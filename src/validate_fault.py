"""Validate fault provenance as well as unchanged local entropy/action invariants."""
import argparse
import json
from pathlib import Path

from src.validate_local import validate as validate_local

FIELDS = ("fault_type", "fault_severity", "fault_active", "injection_step", "original_goal", "agent_goal",
          "original_observation", "agent_observation", "fault_payload")


def validate(run_dir, expected_episodes=1):
    run_dir = Path(run_dir)
    summaries = json.loads((run_dir / "episodes.json").read_text())
    report = validate_local(run_dir / "trajectories.jsonl", summaries, expected_episodes)
    errors = report["errors"]
    active_count = 0
    for filename in ("trajectories.jsonl", "decisions.jsonl"):
        with (run_dir / filename).open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                label = f"{filename} line {number}"
                try:
                    row = json.loads(line)
                except ValueError:
                    errors.append(f"{label}: invalid JSON")
                    continue
                if not isinstance(row, dict):
                    errors.append(f"{label}: expected JSON object")
                    continue
                if any(field not in row for field in (*FIELDS, "goal", "observation", "step")):
                    errors.append(f"{label}: missing fault metadata")
                    continue
                if row["goal"] != row["original_goal"] or row["observation"] != row["original_observation"]:
                    errors.append(f"{label}: original input overwritten")
                if row["fault_type"] not in ("no_fault", "goal_drift", "observation_loss", "context_overload"):
                    errors.append(f"{label}: unknown fault type")
                    continue
                if row["fault_type"] != "no_fault" and (type(row["injection_step"]) is not int or row["injection_step"] < 2):
                    errors.append(f"{label}: invalid injection step")
                    continue
                if type(row["step"]) is not int:
                    errors.append(f"{label}: invalid step")
                    continue
                expected_active = row["fault_type"] != "no_fault" and row["step"] >= row["injection_step"]
                if type(row["fault_active"]) is not bool or row["fault_active"] != expected_active:
                    errors.append(f"{label}: incorrect fault timing")
                if not row["fault_active"] and (row["agent_goal"] != row["original_goal"] or row["agent_observation"] != row["original_observation"]):
                    errors.append(f"{label}: inputs changed outside fault window")
                if row["fault_type"] == "goal_drift" and row["agent_observation"] != row["original_observation"]:
                    errors.append(f"{label}: goal_drift changed observation")
                if row["fault_type"] in ("observation_loss", "context_overload") and row["agent_goal"] != row["original_goal"]:
                    errors.append(f"{label}: observation fault changed true goal")
                if filename == "trajectories.jsonl" and row["fault_active"]:
                    active_count += 1
                payload = row["fault_payload"]
                if not isinstance(payload, dict) or type(payload.get("applied")) is not bool:
                    errors.append(f"{label}: invalid fault payload")
                    continue
                changed = row["agent_goal"] != row["original_goal"] or row["agent_observation"] != row["original_observation"]
                if payload["applied"] != changed or (payload["applied"] and not row["fault_active"]):
                    errors.append(f"{label}: unsupported perturbation claim")
                if "added_context" in payload:
                    if not isinstance(payload["added_context"], str) or payload.get("added_context_char_count") != len(payload["added_context"]):
                        errors.append(f"{label}: context character count mismatch")
    report.update(data_valid=not errors, fault_active_transition_count=active_count)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=1)
    args = parser.parse_args()
    report = validate(args.run_dir, args.expected_episodes)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["data_valid"] and report["experiment_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
