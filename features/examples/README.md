# Test fixture example only

timeseries.fixture.csv contains five synthetic unit-test rows, not real Qwen,
ALFWorld, or MiniLM measurements. Its tokenizer counts come from an artificial
byte tokenizer and its cosine values come from explicit test vectors.
data_kind=unit_test_fixture and fixture model names identify every row.

Do not use this example as research training/evaluation data. Generate real
features/timeseries.csv from your local runs_fault with extract_timeseries.py.
