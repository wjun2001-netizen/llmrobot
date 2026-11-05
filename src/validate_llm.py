"""Validate LLM transition data without treating unsuccessful tasks as corrupt data."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

FIELDS = ("episode_id", "step", "goal", "observation", "plan", "action", "admissible_actions",
          "previous_actions", "reward", "done", "success", "agent_type", "model_name")


def validate(path, summaries, expected_episodes):
    groups, errors, nulls = {}, [], Counter()
    rows = 0
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except ValueError:
                errors.append(f"line {number}: invalid JSON")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {number}: expected object")
                continue
            rows += 1
            missing = [field for field in FIELDS if field not in row]
            if missing:
                errors.append(f"line {number}: missing fields {missing}")
            nulls.update(field for field in FIELDS if field in row and row[field] is None)
            episode_id = row.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                errors.append(f"line {number}: invalid episode ID")
                continue
            group = groups.setdefault(episode_id, [])
            if row.get("agent_type") != "llm":
                errors.append(f"line {number}: not LLM data")
            if type(row.get("step")) is not int or row["step"] != len(group) + 1:
                errors.append(f"line {number}: nonconsecutive step")
            if row.get("previous_actions") != [previous.get("action") for previous in group]:
                errors.append(f"line {number}: previous_actions mismatch")
            for field in ("observation", "action"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    errors.append(f"line {number}: empty {field}")
            admissible = row.get("admissible_actions")
            if not isinstance(admissible, list) or row.get("action") not in admissible:
                errors.append(f"line {number}: action not admissible")
            if row.get("plan") is not None and not isinstance(row["plan"], str):
                errors.append(f"line {number}: invalid plan type")
            if row.get("model_name") is not None and not isinstance(row["model_name"], str):
                errors.append(f"line {number}: invalid model_name")
            reward = row.get("reward")
            if reward is not None and (type(reward) not in (int, float) or not math.isfinite(reward)):
                errors.append(f"line {number}: invalid reward")
            if type(row.get("done")) is not bool:
                errors.append(f"line {number}: invalid done")
            if row.get("success") is not None and type(row["success"]) is not bool:
                errors.append(f"line {number}: invalid success")
            if group:
                if group[-1].get("next_observation") != row.get("observation"):
                    errors.append(f"line {number}: discontinuous observation")
                if group[-1].get("done") or group[-1].get("truncated"):
                    errors.append(f"line {number}: action after episode end")
            group.append(row)
    ids = [summary["episode_id"] for summary in summaries]
    if len(ids) != len(set(ids)) or len(ids) != expected_episodes:
        errors.append("Summary episode count/uniqueness mismatch")
    for summary in summaries:
        group = groups.get(summary["episode_id"], [])
        if not group and summary["outcome"] in ("success", "task_failure", "step_limit"):
            errors.append(f"{summary['episode_id']}: completed outcome without executed transitions")
        if len(group) != summary["steps"]:
            errors.append(f"{summary['episode_id']}: summary step count mismatch")
        if group:
            last = group[-1]
            if any(summary.get(field) != last.get(field) for field in ("done", "success", "truncated")):
                errors.append(f"{summary['episode_id']}: summary terminal status mismatch")
        if summary["outcome"] == "success" and not (summary.get("done") is True and summary.get("success") is True):
            errors.append(f"{summary['episode_id']}: unsupported success claim")
    if set(groups) - set(ids):
        errors.append("Raw episode absent from summary")
    counts = Counter(summary["outcome"] for summary in summaries)
    complete = len(summaries) == expected_episodes and all(
        summary["outcome"] in ("success", "task_failure", "step_limit") for summary in summaries)
    return {"data_valid": not errors, "experiment_complete": complete,
            "episode_attempt_count": len(summaries), "episodes_with_transitions": len(groups),
            "transition_count": rows, "success_count": counts["success"],
            "task_failure_count": counts["task_failure"] + counts["step_limit"],
            "outcome_counts": dict(counts),
            "mean_executed_steps_per_attempt": sum(s["steps"] for s in summaries) / len(summaries) if summaries else None,
            "episode_steps": {s["episode_id"]: s["steps"] for s in summaries},
            "null_counts": dict(nulls), "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=5)
    args = parser.parse_args()
    result = validate(args.run_dir / "trajectories.jsonl",
                      json.loads((args.run_dir / "episodes.json").read_text()), args.expected_episodes)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["data_valid"] and result["experiment_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
