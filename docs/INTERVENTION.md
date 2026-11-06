# Preliminary rule-based intervention experiment

This is a small comparison around the existing local Qwen agent and trained
LSTM. There is no reinforcement learning, diagnostic neural network, additional
detector training, tuning loop or hyperparameter search. The original collectors,
fault injector, ACTION_ID mapping and full-vocabulary entropy calculation are
unchanged. All experiment outputs are newly created under `runs_intervention/`.

## Local setup and execution

Use the working NVIDIA/Qwen environment. Install a suitable CUDA PyTorch wheel
first, then dependencies if needed:

```bash
python -m pip install -r requirements-intervention.txt
python run_intervention.py --check-imports
```

The default checkpoint is the user's existing local training result:
`features/lstm/results/20261106T003226.747846Z_03a0d113/model.pt`.
The real checkpoint and research windows are not present in Cloud, and are never
replaced with synthetic weights/data. Missing inputs stop before loading Qwen.
Checkpoint inference uses the unchanged fixed LSTM on CPU, not a new model.

One proposed-mode episode:

```bash
python run_intervention.py \
  --agent local --model Qwen/Qwen3-4B \
  --game data/alfworld/json_2.1.1/train/pick_and_place_simple-Box-None-Sofa-205/trial_T20190907_214830_497445/game.tw-pddl \
  --fault context_overload --severity high --mode proposed \
  --lstm-checkpoint features/lstm/results/20261106T003226.747846Z_03a0d113/model.pt \
  --injection-step 9 --seed 42 --min-anomaly-z 1.0
```

All four conditions, with one independent environment initialization per condition
and identical game, fault, severity, injection step, seed and generation options:

```bash
python run_intervention.py \
  --agent local --model Qwen/Qwen3-4B \
  --fault context_overload --severity high \
  --lstm-checkpoint features/lstm/results/20261106T003226.747846Z_03a0d113/model.pt \
  --injection-step 9 --seed 42 --min-anomaly-z 1.0 --compare-all
```

The game defaults to the same Box→Sofa path shown above. `--compare-all` runs the
four modes in the fixed order below and takes precedence over `--mode`.
`--fault goal_drift|observation_loss|context_overload` and `--severity low|high`
remain available. Existing local model switches are supported, including
`--config`, `--model-revision`, `--load-in-4bit`, `--compute-dtype`, `--device`,
`--max-new-tokens`, `--max-context-tokens`, `--save-logits` and their boolean negations.

Unless timing is explicitly supplied, this exact Box→Sofa game uses the
**user-reported** 16-step baseline reference, hence injection step 9. That is a
timing reference, not a newly measured result. Other games use the existing
collector timing logic: `--baseline-steps`/`--baseline-run`, then
`--reference-steps`, game walkthrough length, or 20. `--injection-step` overrides
the reference. Actual timing and its source are saved before the first condition.

## Fixed modes and rules

| Mode | Trigger | One intervention |
|---|---|---|
| `no_intervention` | Never | Same agent inputs/actions as the fault collector; additional read-only feature logging |
| `periodic_goal_reinjection` | Every 5 fault-active steps, configurable with `--goal-period` | Restore original agent-visible goal for this decision |
| `entropy_only` | Candidate E strictly greater than fixed entropy threshold | Re-use current original observation and generate a fresh plan/action once |
| `proposed` | Full five-step window, sigmoid(logit) ≥ 0.5, maximum standardized anomaly ≥ `--min-anomaly-z 1.0` | Largest one-sided train-standardized anomaly: S → goal reinjection, C → context cleaning, E → re-observation/replanning |

No rule runs before fault activation. There is no padding of LSTM warm-up windows;
proposed first has enough evidence at step 5. A high risk below the minimum
standardized anomaly gate produces no intervention. Diagnosis never consults the injected fault label.
Only proposed loads/calls the LSTM; other modes keep `lstm_risk = null`.

Context cleaning retains the last `--history-keep 3` actions and removes older
actions from future agent-visible history; the full executed action list remains
in the trajectory. History can grow again until another cleaning event. The
currently injected observation noise **remains**. Goal reinjection and
re-observation apply to this decision only: the original fault is independently
applied again at every subsequent step. `fault_payload`, original inputs and
before/after inputs are all retained. Re-observation uses feedback already
available from the unchanged current state; it adds no `look` action, extra
transition, reset, rollback or oracle action.

The agent has no previous-plan input/cache to clear. Replanning simply calls its
unchanged `decide()` again with revised inputs. If those inputs are identical,
greedy generation may produce the same result. Cleaning history alone may also
be insufficient to recover from long observation noise. These are experimental
outcomes, not reasons to change thresholds, remove the fault or substitute actions.

## Causal timing and feature provenance

E and S require a generated plan. At each environment step:

1. Apply the unchanged fault to agent-visible goal/observation.
2. Generate one candidate using the existing local agent and unchanged settings.
3. Compute candidate features using `PromptCounter` and `SentenceSimilarity` from
   `src/timeseries_features.py`. E is its mean entropy in nats; C uses its actual
   recorded generation prompt count divided by max_context_tokens; S is the same
   MiniLM cosine(original initial goal, candidate plan).
4. For proposed, combine the preceding four **final executed decision** features
   with the current candidate features in chronological `[E,C,S]` order. The
   checkpoint receives `[1,5,3]`, eval/inference mode; risk is sigmoid(logit).
5. If one rule triggers, revise only agent-visible context and generate once
   more while the environment has not moved. Never re-evaluate for another trigger
   in the same step. An invalid revised output ends the episode as invalid output;
   there is no fallback to the candidate or an expert answer.
6. Execute exactly one final, originally admissible action. Preserve its full
   normal trajectory information. Store its final E/C/S for subsequent windows.

Top-level `E/C/S` match the **final** logged `mean_entropy`, real prompt and plan.
`lstm_risk` is explicitly the **pre-intervention candidate** risk.
`detection_features` and `lstm_input_window` contain its exact evidence.
`pre_intervention_decision` retains the candidate plan, action and generation
details whenever an intervention occurs. Thus a successful repair may have lower
final E/C/S even though its trigger came from the higher candidate risk. These
are not silently mixed. Logits, if enabled, have separate candidate/final files.
`context_length_before/after` are actual prompt **token** counts; action counts
are separately named `history_count_before/after`. `agent_previous_actions` is
the history really forwarded to the model.

A hard generation context-budget error may occur before E or S exists. Preserve
that generation error; do not guess entropy/risk, silently truncate inputs, or
pretend that a recovery occurred.

## Proposed calibration and standardized cause diagnosis

The checkpoint, LSTM input window, raw E/C/S definitions, sigmoid and risk threshold
0.5 are unchanged. **Only cause diagnosis after the risk gate is standardized**;
LSTM inputs are never normalized by these calibration statistics.

Proposed always reads the `split.json` beside the selected checkpoint and the
matching original `features/lstm/windows.csv`. `--calibration-windows PATH` can
specify that CSV explicitly; otherwise it uses `--entropy-training-windows`.
CSV SHA256 must match both the saved split and checkpoint's training-input hash.
Only TRAIN indices are used; validation never contributes to fitting statistics.

Deduplicate by `(run_dir, episode_id, condition, step)` across TRAIN windows first.
Use observations covered exclusively by label-0 windows for the reference
population; exclude observations covered by both label-0 and label-1 windows.
Window labels are weak outcome-based labels, so this conservative overlap policy
avoids treating the same observation as both normal and risky. Fewer than two
eligible observations stops calibration; no validation/label-1/default-statistics
fallback is performed. Conflicting values for the same observation also stop it.

Calculate population means and standard deviations (`ddof=0`). Preserve the six
raw statistics and use `max(std, 1e-6)` only as the division denominator. The fixed
floor, effective standard deviations, population policy/counts, source paths,
input/split SHA256 and zero validation usage are saved in config and metrics.

For a risk-qualified **candidate**, compute:

```text
z_E = (E - E_mean) / max(E_std, 1e-6)
z_C = (C - C_mean) / max(C_std, 1e-6)
z_S = (S_mean - S) / max(S_std, 1e-6)
a_feature = max(0, z_feature)
```

Choose the largest `a_feature`, and intervene only when its value is at least
`--min-anomaly-z` (default **1.0**, fixed before outcomes). Exact ties alone use
**E, then C, then S**. This is a tie-break, not a cause priority: a larger C score
always wins even if S also has an anomaly. No absolute E/C/S cutoff or injected
fault label is used by proposed. The older similarity/context CLI flags remain
accepted for command compatibility and have no effect on proposed diagnosis.

Each step records signed `diagnostic_E_z/C_z/S_z`, clamped `anomaly_scores`,
`selected_anomaly`, `selected_anomaly_score` and `calibration_E_mean/std`,
`calibration_C_mean/std`, `calibration_S_mean/std`. During warm-up, before fault
activation or below the risk gate, diagnosis fields are null. Below the minimum
anomaly gate, the strongest positive candidate is still logged for audit, but no
intervention occurs. If all clamped scores are zero, selected_anomaly is null and
its score is zero. Candidate evidence and final executed E/C/S remain separate.

## Other modes and thresholds stay unchanged

- Risk: default 0.5 (`--lstm-risk-threshold`).
- Entropy-only: supply a predeclared `--entropy-threshold VALUE` in nats, with
  `--entropy-threshold-basis 'reason'`; otherwise derive one fixed 95th percentile
  from `--entropy-training-windows features/lstm/windows.csv` using the saved
  `split.json` beside the checkpoint (`--entropy-split` can point elsewhere).
  Only saved **training** indices are used, validation excluded. Overlapping
  window steps are deduplicated by run/episode/condition/step, and the quantile
  uses linear interpolation. Source SHA256 must match split metadata. Calibration stops on mismatch.

No-intervention, periodic goal reinjection and entropy-only retain their existing
decision behavior. Any explicit threshold changes must be decided before collecting outcomes. Exact values,
basis and input hashes are frozen in `config.json` and carried into `metrics.json`.
Entropy-only can use an explicit threshold without loading/accessing any LSTM
checkpoint. No-intervention/periodic need no checkpoint or entropy calibration.

## Outputs and interpretation

`runs_intervention/<timestamp_UUID>/` contains `config.json`, combined
`trajectories.jsonl`, `episodes.json`, `comparison.json`, `metrics.json`, `run.log`,
and `intervention_comparison.csv`. Each mode also has its own config, trajectory,
decisions and episode summary. The combined stream is mode-major then step-major;
identify rows by **mode + episode_id + step**. For the existing feature extractor,
point to an individual mode trajectory, not the aggregate root (it has repeated
episode/step identifiers across conditions).

Summary fields include success, executed steps, outcome, attempted/completed
intervention counts, first intervention step, mean candidate LSTM risk, effective
fault exposure, and pre-injection prefix match against no-intervention when present.
`false_intervention_count = null` with an explanation: a per-step causal oracle is
absent, so final task success is not evidence that a trigger was false.
`pre_injection_intervention_count` is separately calculable (expected zero).
`recovery_vs_no_intervention` is available only when the matched no-intervention
condition had a task failure, the prefix matched and fault exposure occurred;
it reports observed success with intervention, not proof of causality. Operational
errors stay distinct from task failures. All success/failure logs are preserved.

The checkpoint is inference-only and protected inputs are re-hashed after the run.
`detector_training_episode_overlap` reports whether the selected episode appears
in adjacent split.json training groups, or null when unknown. A positive overlap
means this is an in-sample preliminary experiment, not held-out detector validation.
GPU nondeterminism and added re-generation cost also limit comparisons; task
steps count actual environment transitions, not model calls.

## Cloud verification

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest discover -s tests -p test_intervention.py -v
```

New tests inject fake environments, agents and embeddings and use tiny synthetic
CPU checkpoints. They do not run Qwen, download model weights or generate research
results. Run the real four-condition comparison on the local NVIDIA PC.
