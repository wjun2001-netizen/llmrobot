"""Extract read-only E/C/S feature CSV from multiple local/fault trajectory runs."""
import argparse
import csv
import json
import sys
from pathlib import Path

from src.timeseries_features import FeatureError, PromptCounter, SentenceSimilarity, DEFAULT_SIMILARITY_MODEL, extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs_fault"))
    parser.add_argument("--output", type=Path, default=Path("features/timeseries.csv"))
    parser.add_argument("--max-context-tokens", type=int, default=4096, help="Fallback only when the generating run has no recorded context budget")
    parser.add_argument("--similarity-model", default=DEFAULT_SIMILARITY_MODEL)
    parser.add_argument("--similarity-revision")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--tokenizer-revision", help="Original revision for old logs missing resolved tokenizer/model revision")
    parser.add_argument("--allow-unverified-reconstruction", action="store_true", help="Explicit override for unknown generating source; marked as unverified in CSV")
    parser.add_argument("--skip-invalid", action="store_true", help="Explicitly skip invalid rows/files and list them in sidecar metadata; never renumber steps")
    parser.add_argument("--overwrite", action="store_true", help="Replace only derived feature output; never trajectory inputs")
    parser.add_argument("--preview-rows", type=int, default=5)
    args = parser.parse_args()
    try:
        if args.batch_size <= 0 or args.preview_rows < 0:
            raise FeatureError("batch-size must be positive and preview-rows nonnegative")
        counter = PromptCounter(default_max_context=args.max_context_tokens, local_files_only=args.local_files_only,
                                tokenizer_revision=args.tokenizer_revision, allow_unverified_reconstruction=args.allow_unverified_reconstruction)
        similarity = SentenceSimilarity(args.similarity_model, revision=args.similarity_revision,
                                        batch_size=args.batch_size, local_files_only=args.local_files_only)
        report = extract(args.runs_dir, args.output, counter=counter, similarity=similarity,
                         skip_invalid=args.skip_invalid, overwrite=args.overwrite)
        print(json.dumps({"output": str(args.output), "rows": report["rows"], "validation_errors": len(report["validation_errors"])}, indent=2))
        with args.output.open(encoding="utf-8") as stream:
            reader = csv.reader(stream)
            writer = csv.writer(sys.stdout, lineterminator="\n")
            for index, row in enumerate(reader):
                if index > args.preview_rows:
                    break
                writer.writerow(row)
        return 0
    except (FeatureError, OSError, ImportError) as exc:
        parser.exit(2, f"Feature extraction failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
