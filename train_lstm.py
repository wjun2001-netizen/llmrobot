"""Train the fixed five-step [E,C,S] failure-risk classifier."""
import argparse
import json
import sys


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="features/lstm/windows.csv")
    parser.add_argument("--output-dir", default="features/lstm/results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        from src.lstm_training import TrainingConfig, train
        config = TrainingConfig(args.epochs, args.batch_size, args.learning_rate, args.seed,
                                args.validation_fraction, args.device, args.num_threads)
        run, metrics = train(args.input, args.output_dir, config)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    print(f"RESULT_DIR={run}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
