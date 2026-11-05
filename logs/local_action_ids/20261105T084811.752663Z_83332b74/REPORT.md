# Local ACTION_ID selection change

The local agent now receives zero-based IDs for the current, original
admissible_actions list and emits PLAN / ACTION_ID instead of an action string.
The executable action is taken only from admissible_actions[action_id]. Negative,
out-of-range, missing, ambiguous and non-integer IDs are invalid outputs; no fuzzy
matching, plan-to-command substitution, or PDDL fallback is performed.

New JSONL fields include raw_model_output, action_id and action_selection_format.
Invalid attempts retain raw output and token entropy in decisions.jsonl, and do
not produce fabricated env.step transitions. The validator checks exact original
list mapping and raw output parsing while retaining support for prior log schemas.

29 local unit tests passed with CPU tensors and mocked model/environment.
No actual model, CUDA inference, or ALFWorld episode was run in Cloud.
The entropy function is byte-identical and the generate call AST is unchanged.
92 protected source/log files are unchanged. Previous local sources, tests and
documentation are copied byte-for-byte into archive/local_action_string_v1/.

Evidence: tests.log, preservation_before.json and verification.json.

On the local PC, pull feat/local-llm-agent, run the unit test suite, then:

```bash
python collect_normal.py --agent local --model Qwen/Qwen3-4B --episodes 1
python -m src.validate_local <printed-run-directory> --expected-episodes 1
```

After confirming the single episode, collect five episodes with --episodes 5.
All invocations create separate run directories; previous trajectories are retained.
