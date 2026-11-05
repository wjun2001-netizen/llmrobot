# Input-only runtime fault implementation

Added goal_drift, observation_loss and context_overload with low/high severity.
Faults transform only goal/observation immediately before the unchanged agent's
decide call. Environment/state, original goal, action order/ACTION_ID mapping,
generation options, entropy implementation and existing normal collector remain
unchanged. No LSTM, feature classifier, intervention, memory_noise or HHI.

Metadata is persisted in both transitions and failed decision attempts. Context
payload text, character count and actual tokenizer count (when available) are
retained. Unknown token counts remain null. Injection starts at step >= 2.
Scheduled, actually active and effectively perturbed input steps are distinguished.

Paired runs reset the same game/seed for no_fault and fault. Timing uses actual
no_fault length; fixed episode ID/game path is supported. Single fault runs can
use a prior baseline's length, a configured reference, or a labeled PDDL length
reference, without sending hidden walkthrough actions to the model.
Already-failed baselines, pre-injection divergence, unexposed episodes, task
failures and model/runtime errors are recorded separately. Pair labels describe
observed outcomes and are not proof of causality or a learned classifier.

Validation: new fault unit tests 22/22, unchanged normal local tests 29/29, total
51 passing. All environment/model calls are mocked. Cloud loaded zero pretrained
models, executed zero actual ALFWorld episodes and attempted no GPU inference.
Existing sources and raw logs were checksum-checked (verification.json).

New workflow and local commands: docs/FAULT_COLLECTION.md.
Actual local fault datasets are not claimed; unit fixtures are temporary and are
never counted as research episodes. Existing runs were not deleted or overwritten.
