"""Validate local trajectory entropy and the shared LLM transition schema."""
import argparse
import json
import math
from pathlib import Path

from src.validate_llm import validate as validate_transitions


def validate(path, summaries, expected_episodes):
    report = validate_transitions(path, summaries, expected_episodes)
    errors = report["errors"]
    token_count = 0
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            entropies, mean = row.get("token_entropies"), row.get("mean_entropy")
            ids, vocab = row.get("generated_token_ids"), row.get("score_vocab_size")
            if row.get("backend") != "local" or row.get("entropy_unit") != "nats":
                errors.append(f"line {number}: incorrect backend or entropy unit")
            if not isinstance(entropies, list) or not entropies or not isinstance(ids, list) or len(ids) != len(entropies):
                errors.append(f"line {number}: missing/misaligned generated tokens and entropies")
                continue
            if type(vocab) is not int or vocab < 1:
                errors.append(f"line {number}: invalid score vocabulary size")
                continue
            if not all(type(token) is int and 0 <= token < vocab for token in ids):
                errors.append(f"line {number}: generated token ID outside score vocabulary")
            valid_values = all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= math.log(vocab) + 1e-4 for value in entropies)
            if not valid_values:
                errors.append(f"line {number}: invalid full-vocabulary entropy")
                continue
            if type(mean) not in (int, float) or not math.isfinite(mean) or not math.isclose(mean, math.fsum(entropies) / len(entropies), abs_tol=1e-6):
                errors.append(f"line {number}: mean_entropy mismatch")
            token_count += len(entropies)
    report.update(data_valid=not errors, entropy_token_count=token_count,
                  entropy_unit="nats", entropy_scope="Full vocabulary of unfiltered greedy generate.scores; generated tokens including EOS, not prompt tokens")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=5)
    args = parser.parse_args()
    result = validate(args.run_dir / "trajectories.jsonl", json.loads((args.run_dir / "episodes.json").read_text()), args.expected_episodes)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["data_valid"] and result["experiment_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
