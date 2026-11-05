# Sliding-window data preparation

Added make_lstm_windows.py and src/lstm_windows.py. Pure standard library,
CSV output, default W=5/stride=1. No model fitting, learned classifier, feature
normalization, or train/test split.

Groups are run_dir/episode_id/condition. Original within-group order is retained;
duplicate/reversed steps are errors. Gaps and missing/nonfinite E/C/S invalidate
containing windows and are never bridged/padded/interpolated. Short episodes yield
zero windows rather than mixing episode boundaries.

Success trajectories label 0. Fault failure trajectories label 1 only when all
window steps are at/after injection by default; pre/mixed windows label 0.
No_fault failures are excluded by the user's explicit choice. Other runtime/agent
outcomes are excluded instead of inventing a binary task label.
Optional baseline_already_failed and other comparison exclusions use CSV metadata,
pair.json, or same-pair no_fault outcomes where sufficient. Unknown comparison
data requires an explicit exclusion policy or produces an error.

Outputs include sample metadata, E_1..E_W/C_1..C_W/S_1..S_W, label, class counts,
positive/negative counts, policy/input hashes and exclusion counts. Source CSV and
pair metadata are protected against direct, symlink and hardlink overwrite.

Tests: new window tests 26, existing features 18, local 29 and fault 22 = 95 passing.
Existing features/trajectories/logs (111 files) checked unchanged. Source agents and
collectors were not modified. No real training windows were fabricated because
features/timeseries.csv exists on the user's PC, not this Cloud workspace.
CLI class counts are verified with temporary unit fixtures and are not actual
research class distribution measurements.

Commands and label details: docs/LSTM_WINDOWS.md.
Evidence: tests_final_verified.log, feature_18_tests.log, local_29_tests.log,
fault_22_tests.log and verification.json.
