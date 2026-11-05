"""Read raw JSONL from disk; verify schema, transitions and terminal outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path

REQUIRED = (
    "episode_id", "step", "goal", "observation", "plan", "action",
    "admissible_actions", "reward", "done", "success",
)


def validate(path, expected_episodes=None):
    episodes = {}
    errors, missing = [], []
    nulls = Counter()
    empty_observation = empty_action = rows = 0
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {line_number}: not an object")
                continue
            rows += 1
            for field in REQUIRED:
                if field not in row:
                    missing.append({"line": line_number, "field": field})
                elif row[field] is None:
                    nulls[field] += 1
            episode_id = row.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                errors.append(f"line {line_number}: invalid episode_id")
                continue
            group = episodes.setdefault(episode_id, [])
            if type(row.get("step")) is not int or row["step"] != len(group) + 1:
                errors.append(f"line {line_number}: nonconsecutive step")
            if group and (group[-1].get("done") or group[-1].get("truncated")):
                errors.append(f"line {line_number}: transition after episode ended")
            if not isinstance(row.get("observation"), str) or not row["observation"].strip():
                empty_observation += 1
            if not isinstance(row.get("action"), str) or not row["action"].strip():
                empty_action += 1
            if group and group[-1].get("next_observation") != row.get("observation"):
                errors.append(f"line {line_number}: observation discontinuity")
            for field in ("done", "success"):
                if row.get(field) is not None and type(row[field]) is not bool:
                    errors.append(f"line {line_number}: invalid {field}")
            reward = row.get("reward")
            if reward is not None and (type(reward) not in (int, float)):
                errors.append(f"line {line_number}: invalid reward")
            admissible, plan = row.get("admissible_actions"), row.get("plan")
            if admissible is not None:
                if not isinstance(admissible, list) or row.get("action") not in admissible:
                    errors.append(f"line {line_number}: action not admissible")
            if plan is not None:
                if not isinstance(plan, list) or not plan or plan[0] != row.get("action"):
                    errors.append(f"line {line_number}: action differs from expert plan")
            group.append(row)
    per_episode = []
    for episode_id, group in episodes.items():
        last = group[-1]
        outcome = "success" if last.get("done") is True and last.get("success") is True else "failure"
        per_episode.append({"episode_id": episode_id, "steps": len(group), "outcome": outcome,
                            "terminal_done": last.get("done"), "terminal_success": last.get("success"),
                            "success_values_recorded": all(type(r.get("success")) is bool for r in group)})
        if outcome != "success":
            errors.append(f"{episode_id}: no successful terminal transition")
    if expected_episodes is not None and len(episodes) != expected_episodes:
        errors.append(f"Expected {expected_episodes} episodes, found {len(episodes)}")
    if not rows:
        errors.append("No transitions")
    if missing:
        errors.append("Missing required fields")
    if empty_observation or empty_action:
        errors.append("Empty observation/action")
    if nulls.get("success", 0) or nulls.get("done", 0):
        errors.append("Unobservable terminal status")
    return {"valid": not errors, "episode_count": len(episodes), "transition_count": rows,
            "success_count": sum(e["outcome"] == "success" for e in per_episode),
            "failure_count": sum(e["outcome"] == "failure" for e in per_episode),
            "episodes": per_episode, "missing_fields": missing, "null_counts": dict(nulls),
            "empty_observation_count": empty_observation, "empty_action_count": empty_action,
            "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--report", type=Path, help="New output path; existing reports are never overwritten")
    args = parser.parse_args()
    report = validate(args.path, args.expected_episodes)
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
