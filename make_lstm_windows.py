"""Create sliding E/C/S training samples only; no LSTM fitting or train/test split."""
import argparse
import json
from pathlib import Path

from src.lstm_windows import WindowError, make_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features/timeseries.csv"))
    parser.add_argument("--output", type=Path, default=Path("features/lstm/windows.csv"))
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--exclude-baseline-already-failed", action="store_true")
    parser.add_argument("--exclude-comparison", action="append", default=[], help="Additional pair comparison labels to exclude")
    parser.add_argument("--missing-comparison-policy", choices=["error", "exclude"], default="error")
    parser.add_argument("--no-fault-failure-policy", choices=["exclude", "positive"], default="exclude")
    parser.add_argument("--post-injection-policy", choices=["all", "end"], default="all",
                        help="all: entire window post-injection (default); end: final step post-injection")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = make_windows(args.input, args.output, window_size=args.window_size,
                              exclude_baseline_already_failed=args.exclude_baseline_already_failed,
                              excluded_comparisons=args.exclude_comparison, no_fault_failure_policy=args.no_fault_failure_policy,
                              post_injection_policy=args.post_injection_policy, missing_comparison_policy=args.missing_comparison_policy,
                              overwrite=args.overwrite)
        print(json.dumps({"output": str(args.output), "samples": report["samples"], "class_counts": report["class_counts"],
                          "positive_samples": report["positive_samples"], "negative_samples": report["negative_samples"],
                          "skipped": report["skipped"], "train_test_split": False}, indent=2))
        return 0
    except (WindowError, OSError) as exc:
        parser.exit(2, f"Window generation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
