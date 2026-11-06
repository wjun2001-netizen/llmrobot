# Development verification only

This directory contains implementation audit records, not intervention research
results. No real Qwen/GPU model was loaded or run, no MiniLM weights were
downloaded, and no live OpenAI/Gemini calls were made.

- Final full suite: **199 passed**, zero errors/failures/skips.
- Existing tests: 167, including all fault/local/feature/window/LSTM suites and
  historical baseline/API regression tests. Historical API integration tests use
  fake API clients with CPU TextWorld; new intervention tests use fake environments.
- New intervention tests: 32, using mock agents/embeddings/environments and tiny
  synthetic CPU checkpoint inference. Synthetic results live in temporary
  `runs_intervention/unit_test_fixture_*` directories removed by test cleanup.
- Initial full-suite run also passed (195 tests); four additional safety tests
  were then added and the complete suite rerun. See `all_tests_final.txt` and
  `test_summary_final.json` for the final result.
- The isolated test environment used torch 2.6.0+cpu, Transformers 4.57.6 and
  sentence-transformers 5.1.2. For the full suite, the baseline venv's packages
  were appended to sys.path to reuse TextWorld/ALFWorld dependencies without
  installing or changing the existing environment.
- `--check-imports` passed and reported `model_loaded=false` and
  `gpu_execution_attempted=false`; bitsandbytes GPU execution was not tested.
- SHA256 verification: all **189** pre-existing non-cache source/log/feature files
  covered by the manifest remained unchanged. The dataset directory was excluded
  from the manifest. No existing source files were edited.
- Scan of new untracked files found no contents matching configured API/token
  secret environment variables (secret values are never printed).

The requested real checkpoint
`features/lstm/results/20261106T003226.747846Z_03a0d113/model.pt` is absent in Cloud.
The loader's fixed architecture, sigmoid inference, missing-checkpoint rejection,
and read-only preservation are covered with synthetic CPU checkpoints. Real
weights and task recovery results remain to be verified on the local NVIDIA PC.

The default similarity/context thresholds (0.3/0.8) are documented a priori
choices, not fitted Box→Sofa results. Default entropy calibration is train-only
95th percentile from the matching saved split/CSV; alternatively supply an
explicit predeclared CLI threshold. No test results were used for threshold tuning.
