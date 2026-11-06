# Standardized proposed diagnosis verification

This is a development audit, not an actual Qwen intervention experiment.

- Full suite: **217 passed**, zero failures/errors/skips (199 existing tests plus
  18 new calibration/diagnosis tests). The old priority-rule assertion was updated
  to the requested largest-anomaly behavior; all other existing tests remain covered.
- No GPU/Qwen, real MiniLM model download or live API execution. New tests use
  synthetic CSVs/checkpoints and fake agent/environment/embedding backends.
  Historical regression tests also include CPU TextWorld with mock API clients.
- `preservation_check.json` covers 204 pre-existing non-cache files (dataset
  excluded). Only the five intended source/doc/test files changed; original
  trajectory logs, feature data and training outputs were not changed/deleted.
- AST comparisons verify unchanged LSTMRiskDetector, decision_features,
  training_entropy_threshold, other-mode decision branches and risk-gate logic.
  The LSTM classifier, local agent, collector and fault injector are unchanged.
- Calibration uses only saved TRAIN indices; deduplication keys are
  run_dir/episode_id/condition/step. Only label-0-only covered observations are
  admitted; mixed-label overlap is excluded, without validation/label-1 fallback.
- Population mean/std (ddof=0), a fixed 1e-6 std denominator floor, and fixed
  default min_anomaly_z=1.0 are independently tested. Exact score ties alone use
  E, then C, then S; otherwise the largest score wins regardless of feature order.
- Tests verify signed S direction, negative clamping, candidate-versus-final
  feature provenance, max-one-intervention behavior and calibration JSON metadata.
- No empirical thresholds or statistics were fitted to the user's step-13/29
  outcomes. Real calibration is performed read-only before local execution using
  the CSV and split matching the real checkpoint; no synthetic defaults are used.

Test details: `all_tests.txt`, `test_summary.json`. Source integrity details:
`preservation_before.json`, `preservation_check.json`.
