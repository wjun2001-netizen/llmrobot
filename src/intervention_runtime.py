"""Agent-input-only interventions around the unchanged local ALFWorld runner."""
from dataclasses import asdict, replace
import csv
import json
import logging
from pathlib import Path
import random

from collect_local import run_episode
from local_llm_agent import LocalGenerationError
from src.fault_injection import FaultInjector
from src.fault_runtime import FaultLogStream
from src.intervention import choose_intervention, decision_features
from src.llm_io import append_jsonl, new_run, save_json
from src.timeseries_features import PromptCounter, digest
from src.anomaly_calibration import diagnostic_log_fields

ROOT = Path(__file__).resolve().parents[1]


class InterventionAgent:
    """One candidate and at most one revised decision per environment transition.

    The wrapper has no environment handle. Only final decisions enter the original
    runner. Detector windows use the last four executed features plus this candidate.
    """
    def __init__(self, agent, injector, intervention, similarity, detector=None):
        self.agent, self.config, self.injector = agent, agent.config, injector
        self.intervention, self.similarity, self.detector = intervention, similarity, detector
        if intervention.mode == "proposed" and detector is None:
            raise ValueError("proposed mode requires the trained LSTM detector")
        self.counter = PromptCounter()
        self.metadata, self.history, self.step, self.history_start = {}, [], 0, 0
        self.generation_call_count = 0
        self.cooldown_until_step = 0

    def _decide(self, goal, observation, previous, actions, logits_path):
        original = tuple(actions)
        forwarded = list(original)
        self.generation_call_count += 1
        try:
            decision = self.agent.decide(goal, observation, list(previous), forwarded, logits_path=logits_path)
        finally:
            if tuple(actions) != original or tuple(forwarded) != original:
                raise LocalGenerationError("Admissible action order mutated; refusing execution")
        if type(decision.action_id) is not int or not 0 <= decision.action_id < len(original) or decision.action != original[decision.action_id]:
            raise LocalGenerationError("Action/ID differs from original admissible mapping; refusing execution")
        return decision

    def decide(self, goal, observation, previous_actions, admissible_actions, *, logits_path=None):
        self.step += 1
        metadata = self.injector.apply(goal, observation, self.step)
        metadata.update(mode=self.intervention.mode, condition="fault", E=None, C=None, S=None, lstm_risk=None,
                        intervention_triggered=False, intervention_type=None, intervention_reason="candidate_not_available",
                        intervention_completed=False, intervention_count_this_step=0,
                        agent_goal_before=metadata["agent_goal"], agent_goal_after=metadata["agent_goal"],
                        agent_observation_before=metadata["agent_observation"], agent_observation_after=metadata["agent_observation"],
                        context_length_before=None, context_length_after=None, context_length_unit="actual_prompt_tokens",
                        history_count_before=None, history_count_after=None, agent_previous_actions=None,
                        detection_features=None, lstm_input_window=None, pre_intervention_decision=None,
                        lstm_risk_source="candidate_before_intervention", feature_source="final_selected_generation",
                        fault_injection_preserved=True,
                        context_cleaning_history_trimmed=False, context_cleaning_observation_refreshed=False,
                        candidate_C=None, final_C=None, final_features=None, candidate_selected_anomaly_score=None,
                        cooldown_active=self.intervention.mode == "proposed" and self.step <= self.cooldown_until_step,
                        cooldown_remaining=max(0, self.cooldown_until_step - self.step + 1) if self.intervention.mode == "proposed" else 0,
                        cooldown_until_step=self.cooldown_until_step, intervention_suppressed=False,
                        intervention_suppression_reason=None)
        self.metadata[self.step] = metadata  # Also retains attempted/failed decision provenance.
        metadata.update(diagnostic_log_fields(self.intervention.calibration if self.intervention.mode == "proposed" else None))
        previous = list(previous_actions[self.history_start:])
        metadata.update(history_count_before=len(previous), history_count_after=len(previous), agent_previous_actions=list(previous))
        candidate_path = Path(logits_path).with_name(Path(logits_path).stem + ".candidate.pt") if logits_path else None
        candidate = self._decide(metadata["agent_goal"], metadata["agent_observation"], previous, admissible_actions, candidate_path)
        features = decision_features(candidate, goal, self.config, self.similarity, self.counter)
        metadata.update(features)
        metadata.update(context_length_before=features["prompt_token_count"], context_length_after=features["prompt_token_count"],
                        detection_features=dict(features), candidate_C=features["C"], final_C=features["C"], final_features=dict(features))
        window = [*self.history[-4:], [features[key] for key in ("E", "C", "S")]]
        risk = self.detector.predict(window) if self.intervention.mode == "proposed" and len(window) == 5 else None
        metadata.update(lstm_risk=risk, lstm_input_window=window if self.intervention.mode == "proposed" and len(window) == 5 else None)
        kind, reason = choose_intervention(self.intervention, features, risk, step=self.step,
                                           injection_step=metadata["injection_step"], fault_active=metadata["fault_active"], diagnostics=metadata)
        metadata.update(candidate_selected_anomaly_score=metadata.get("selected_anomaly_score"),
                        diagnosed_intervention_type=kind, diagnosed_intervention_reason=reason)
        if metadata["cooldown_active"]:
            metadata.update(intervention_suppressed=kind is not None, intervention_suppression_reason="intervention_cooldown")
            kind, reason = None, "intervention_cooldown"
        metadata["intervention_reason"] = reason
        final = candidate
        if kind is not None:
            metadata.update(intervention_triggered=True, intervention_type=kind, intervention_count_this_step=1,
                            pre_intervention_decision={"plan": candidate.plan, "action_id": candidate.action_id,
                                                       "action": candidate.action, "generation": dict(candidate.generation)})
            if kind == "goal_reinjection":
                metadata["agent_goal"] = goal
            elif kind == "context_cleaning":
                # Trim history and refresh from feedback already available at this transition.
                self.history_start = max(self.history_start, len(previous_actions) - self.intervention.history_keep)
                previous = list(previous_actions[self.history_start:])
                metadata.update(agent_observation=observation, context_cleaning_history_trimmed=True,
                                context_cleaning_observation_refreshed=True)
            else:
                # Re-use the current true feedback already received; no env.look/reset/rollback.
                metadata["agent_observation"] = observation
            metadata.update(agent_goal_after=metadata["agent_goal"], agent_observation_after=metadata["agent_observation"],
                            history_count_after=len(previous), agent_previous_actions=list(previous),
                            context_length_after=None, E=None, C=None, S=None, prompt_token_count=None,
                            feature_source="final_generation_unavailable", final_C=None, final_features=None)
            final = self._decide(metadata["agent_goal"], metadata["agent_observation"], previous, admissible_actions, logits_path)
            final_features = decision_features(final, goal, self.config, self.similarity, self.counter)
            metadata.update(final_features)
            metadata.update(context_length_after=final_features["prompt_token_count"], intervention_completed=True,
                            feature_source="final_selected_generation", final_C=final_features["C"], final_features=dict(final_features))
            if self.intervention.mode == "proposed":
                # The current env.step still executes once. Suppress the NEXT N transitions.
                self.cooldown_until_step = self.step + self.intervention.intervention_cooldown
                metadata["cooldown_until_step"] = self.cooldown_until_step
        self.history.append([metadata[key] for key in ("E", "C", "S")])
        # There is no second trigger evaluation/retry loop in this step.
        return replace(final, generation={**final.generation, **metadata})


def run_mode(game, episode_id, seed, max_steps, agent, fault, intervention, similarity, output_dir, *, detector=None, env_factory=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = getattr(agent, "tokenizer", None)
    encode = getattr(tokenizer, "encode", None)
    counter = (lambda text: len(encode(text, add_special_tokens=False))) if callable(encode) else None
    adapter = InterventionAgent(agent, FaultInjector(fault, seed, episode_id, counter), intervention, similarity, detector)
    save_json(output_dir / "config.json", {"mode": intervention.mode, "condition": "fault", "seed": seed,
                                           "fault": asdict(fault), "intervention": intervention.public_config(),
                                           "local_llm": agent.config.public_config()})
    with (output_dir / "trajectories.jsonl").open("x", encoding="utf-8") as raw, (output_dir / "decisions.jsonl").open("x", encoding="utf-8") as decisions:
        summary = run_episode(game, episode_id, seed, max_steps, adapter, FaultLogStream(raw, adapter),
                              FaultLogStream(decisions, adapter), logits_dir=output_dir / "logits", env_factory=env_factory)
    events = list(adapter.metadata.values())
    interventions = [step for step, event in adapter.metadata.items() if event["intervention_triggered"]]
    risks = [event["lstm_risk"] for event in events if event["lstm_risk"] is not None]
    summary.update(mode=intervention.mode, fault_type=fault.fault_type, fault_severity=fault.severity, seed=seed,
                   injection_step=fault.injection_step, intervention_count=len(interventions),
                   completed_intervention_count=sum(event["intervention_completed"] for event in events),
                   generation_call_count=adapter.generation_call_count,
                   cooldown_suppressed_count=sum(event["intervention_suppressed"] for event in events),
                   first_intervention_step=min(interventions) if interventions else None,
                   mean_lstm_risk=sum(risks) / len(risks) if risks else None,
                   false_intervention_count=None, false_intervention_reason="No per-step causal oracle labels; cannot infer false positives from final success",
                   pre_injection_intervention_count=sum(event["intervention_triggered"] and not event["fault_active"] for event in events),
                   fault_effectively_reached=any(event["fault_payload"]["applied"] for event in events),
                   recovery_vs_no_intervention=None)
    save_json(output_dir / "episodes.json", [summary])
    return summary


def safe_output_root(path):
    path = Path(path).resolve()
    allowed = (ROOT / "runs_intervention").resolve()
    if not path.is_relative_to(allowed):
        raise ValueError("All new experiment outputs must stay under runs_intervention/")
    return path


def compare_summaries(summaries, rows):
    baseline = next((item for item in summaries if item["mode"] == "no_intervention"), None)
    prefix_keys = ("step", "goal", "observation", "action_id", "action", "admissible_actions", "reward", "done", "success")
    for item in summaries:
        item["pre_injection_prefix_match"] = None
        if baseline is None:
            continue
        before = item["injection_step"] - 1
        def prefix(mode):
            return [{key: row.get(key) for key in prefix_keys} for row in rows if row["mode"] == mode and row["step"] <= before]
        first, other = prefix("no_intervention"), prefix(item["mode"])
        match = len(first) == before and first == other
        item["pre_injection_prefix_match"] = match
        if item["mode"] != "no_intervention" and baseline["outcome"] in ("step_limit", "task_failure") and item["outcome"] in ("success", "task_failure", "step_limit") and match and item["fault_effectively_reached"]:
            item["recovery_vs_no_intervention"] = item["success"] is True and item["intervention_count"] > 0
    return summaries


def execute_experiment(*, game, episode_id, seed, max_steps, agent, fault, interventions, similarity,
                       output_root, metadata, detector=None, env_factory=None, seed_runtime=None):
    """Shared four-mode orchestration. Tests inject fake agent/env/detector/embeddings."""
    if not interventions or len({config.mode for config in interventions}) != len(interventions):
        raise ValueError("Require distinct selected intervention modes")
    run = new_run(safe_output_root(output_root))
    logger = logging.getLogger(str(run))
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(run / "run.log", mode="x", encoding="utf-8")
    logger.addHandler(handler)
    save_json(run / "config.json", {**metadata, "game": str(game), "episode_id": episode_id,
                                    "seed": seed, "max_steps": max_steps, "fault": asdict(fault),
                                    "modes": [config.public_config() for config in interventions],
                                    "local_llm": agent.config.public_config()})
    summaries, rows = [], []
    try:
        logger.info("Run %s; modes=%s", run, [config.mode for config in interventions])
        def check_inputs():
            for path, expected in metadata.get("protected_input_sha256", {}).items():
                if digest(path) != expected:
                    raise ValueError("Protected game/checkpoint/calibration input changed: " + path)
        check_inputs()
        with (run / "trajectories.jsonl").open("x", encoding="utf-8") as combined:
            for config in interventions:
                random.seed(seed)
                if seed_runtime is not None:
                    seed_runtime(seed)
                folder = run / config.mode
                summary = run_mode(game, episode_id, seed, max_steps, agent, fault, config, similarity,
                                   folder, detector=detector, env_factory=env_factory)
                summaries.append(summary)
                with (folder / "trajectories.jsonl").open(encoding="utf-8") as stream:
                    for line in stream:
                        row = json.loads(line)
                        rows.append(row)
                        append_jsonl(combined, row)
                logger.info("Mode outcome: %s", json.dumps(summary, ensure_ascii=False))
        compare_summaries(summaries, rows)
        check_inputs()
        save_json(run / "episodes.json", summaries)
        save_json(run / "comparison.json", {"conditions": summaries, "diagnosis": "largest_one_sided_train_standardized_anomaly", "exact_tie_break": ["E", "C", "S"],
                                            "false_intervention_count_available": False,
                                            "recovery_definition": "Observed success after intervention versus same-seed no_intervention task failure, matched pre-injection prefix, effective fault exposure; not causal proof"})
        complete = all(item["outcome"] in ("success", "task_failure", "step_limit") for item in summaries)
        save_json(run / "metrics.json", {"experiment_complete": complete, "conditions_executed": len(summaries),
                                         "task_success_count": sum(item["success"] is True for item in summaries),
                                         "conditions": summaries, "threshold_basis": metadata.get("threshold_basis"),
                                         "feature_semantics": metadata.get("feature_semantics"),
                                         "diagnostic_calibration": metadata.get("diagnostic_calibration"),
                                         "intervention_cooldown": metadata.get("intervention_cooldown"),
                                         "protected_inputs_unchanged": True,
                                         "detector_training_episode_overlap": metadata.get("detector_training_episode_overlap")})
        columns = ["mode", "fault_type", "fault_severity", "success", "steps", "outcome", "intervention_count",
                   "first_intervention_step", "mean_lstm_risk", "false_intervention_count", "recovery_vs_no_intervention"]
        with (run / "intervention_comparison.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        return run, summaries, complete
    except Exception as exc:
        save_json(run / "fatal_error.json", {"error_type": type(exc).__name__, "partial_results_preserved": True})
        logger.error("Experiment interrupted: %s", type(exc).__name__)
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
