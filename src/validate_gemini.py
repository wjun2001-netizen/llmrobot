"""Validate Gemini-specific probabilities plus the existing provider-neutral transition schema."""
import argparse
import json
from pathlib import Path

from src.gemini_agent import probability_data
from src.validate_llm import validate as validate_transitions


def validate(path, summaries, expected_episodes):
    report = validate_transitions(path, summaries, expected_episodes)
    probability_steps = probability_tokens = top_k_aligned_steps = 0
    errors = report["errors"]
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except ValueError:
                continue  # Base validator already records malformed JSON.
            if not isinstance(row, dict):
                continue
            for field in ("token_logprobs", "logprobs_available"):
                if field not in row:
                    errors.append(f"line {line_number}: missing {field}")
            if row.get("provider") != "google_gemini_developer_api":
                errors.append(f"line {line_number}: not Gemini Developer API data")
            result = row.get("logprobs_result")
            expected = probability_data({"candidates": [{"logprobsResult": result}]})
            if type(row.get("logprobs_available")) is not bool or row["logprobs_available"] != expected["logprobs_available"]:
                errors.append(f"line {line_number}: unsupported logprobs availability claim")
            if row.get("token_logprobs") != expected["token_logprobs"]:
                errors.append(f"line {line_number}: token logprobs differ from preserved result")
            if expected["logprobs_available"]:
                probability_steps += 1
                probability_tokens += len(expected["token_logprobs"])
                tops = result.get("topCandidates")
                if isinstance(tops, list) and len(tops) == len(expected["token_logprobs"]):
                    top_k_aligned_steps += 1
    report.update(data_valid=not errors, logprobs_available_steps=probability_steps,
                  logprobs_unavailable_steps=report["transition_count"] - probability_steps,
                  tokens_with_logprobs=probability_tokens, aligned_top_k_steps=top_k_aligned_steps,
                  full_shannon_entropy_data_sufficient=False,
                  entropy_note="Chosen-token logprobs/top-k alternatives are not a full vocabulary distribution; full Shannon entropy is not established.")
    return report


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
