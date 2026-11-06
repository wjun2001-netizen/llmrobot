# Context refresh and proposed cooldown verification

Development verification only: no real GPU/Qwen/MiniLM loading or live API calls.

- Full suite: **228 passed**, zero failures/errors/skips; 217 existing tests and
  11 new context-recovery/cooldown tests. See `all_tests.txt`, `test_summary.json`.
- New tests use synthetic feature/calibration fixtures and fake agents/environments.
  Historical API regression integration tests use fake clients with CPU TextWorld.
- Context cleaning retains recent history and refreshes observation from the same
  current true feedback before one recovery generation. Tests use changing feedback
  to prove it is current feedback, not an initial/stale/next-state observation.
- One environment transition per step, zero resets, original true goal and action
  list/mapping preserved. The unchanged fault is reapplied next step, including
  during cooldown. Candidate/final context counts and features remain separately
  logged, together with cleaning operation flags and candidate anomaly evidence.
- Default cooldown is five proposed-only transitions. After intervention at step 5,
  steps 6..10 are suppressed and step 11 is eligible. Tests cover continuous risk,
  feature and diagnosis logging during suppression, zero/custom cooldown, incomplete
  recovery not arming cooldown, and non-proposed modes ignoring the setting.
- Older operator-focused test fixtures explicitly set cooldown=0 to retain focused
  operator assertions; dedicated new tests exercise the production default=5.
- `preservation_check.json`: 211 pre-existing non-cache files checked (dataset
  excluded). Only the five intended source/doc/test files changed; old trajectories,
  features and LSTM results were not modified or deleted.
- AST checks confirm choose_intervention (including risk/anomaly logic), detector,
  feature and entropy calibration helpers, action-validation/generation forwarding,
  goal reinjection and re-observation/replanning operator bodies are unchanged.
- Fault injector and standardized anomaly calibration source files are byte-for-byte
  unchanged. Local agent generation/entropy and environment runner files are unchanged.
- Modified/new-file scan found no contents matching configured API/token secrets;
  secret values were not printed.

No recovery success result is claimed. Actual Box→Sofa/context_overload-high results
must be collected locally using the existing checkpoint and model settings.
