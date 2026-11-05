# E/C/S feature extraction implementation

Added extract_timeseries.py, src/timeseries_features.py, requirements-features.txt,
configs/prompt_formats.json, feature tests, documentation and an explicitly marked
five-row unit fixture CSV example.

E copies mean_entropy in nats. C uses observed generation-time prompt token count,
or reconstructs the verified historical LocalLLMAgent prompt with the original
tokenizer/revision, actual agent_goal/agent_observation, history, numbered actions
and chat template. Context budget is taken from the run/row (fallback 4096).
S is CPU sentence-transformers embedding cosine(original episode initial goal,
step plan), default all-MiniLM-L6-v2. Negative cosine values are preserved.

LocalLLMAgent adds only prompt_token_count/max_context_tokens/effective_context_limit
metadata after real input tokenization. Entropy function and model.generate call
are unchanged. No source trajectory is edited. Existing logs/trajectories were
verified byte-identical. Output guards prevent overwriting source inputs, default
validation rejects missing/non-finite features and reversed steps, and explicit
skip-invalid mode reports skipped rows without renumbering steps.

Tests: 18 feature + 29 existing local + 22 existing fault = 69 passing. Actual
sentence-transformers package import verified on CPU. Unit tests use a synthetic
byte tokenizer and explicit vector embeddings; no pretrained embedding weights,
Qwen artifacts, actual model inference, GPU execution or ALFWorld episodes were run.

There are no actual runs_fault trajectories in this Cloud workspace, so the default
research output features/timeseries.csv was NOT fabricated. The real CLI reports
the missing input before loading models. features/examples/timeseries.fixture.csv
contains 5 clearly labeled unit_test_fixture rows with fixture tokenizer/model
names; these are examples, NOT research E/C/S measurements. Temporary fixture input
paths in its metadata are not actual experiment run directories.

Real usage on the local PC:

```bash
python -m pip install -r requirements-features.txt
python extract_timeseries.py --runs-dir runs_fault --output features/timeseries.csv
```

Evidence: verification.json, tests_final_verified.log, local_29_tests.log,
fault_22_tests.log, import_check.json, no_real_runs_check.log, raw_preservation_before.json.

No LSTM, learned feature classifier, or intervention was implemented.
