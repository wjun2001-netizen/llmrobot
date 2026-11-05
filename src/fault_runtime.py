"""Wrap the unchanged normal runner at the agent-input and JSONL-output boundaries."""
from dataclasses import asdict, replace
import json
from pathlib import Path

from collect_local import run_episode
from local_llm_agent import LocalGenerationError, LocalInvalidOutput
from src.fault_injection import FaultConfig, FaultInjector
from src.llm_io import save_json


class FaultAgent:
    def __init__(self, agent, injector):
        self.agent, self.injector, self.config = agent, injector, agent.config
        self.metadata = {}
        self.step = 0

    def decide(self, goal, observation, previous_actions, admissible_actions, *, logits_path=None):
        self.step += 1
        metadata = self.injector.apply(goal, observation, self.step)
        self.metadata[self.step] = metadata
        original_actions = tuple(admissible_actions)
        forwarded_actions = list(original_actions)
        try:
            decision = self.agent.decide(metadata["agent_goal"], metadata["agent_observation"],
                                         list(previous_actions), forwarded_actions, logits_path=logits_path)
        except LocalInvalidOutput as exc:
            raise LocalInvalidOutput(str(exc), {**exc.generation, **metadata}, exc.parsed) from None
        if tuple(admissible_actions) != original_actions or tuple(forwarded_actions) != original_actions:
            raise LocalGenerationError("Admissible actions mutated at the agent boundary; refusing execution")
        if type(decision.action_id) is not int or not 0 <= decision.action_id < len(original_actions) or decision.action != original_actions[decision.action_id]:
            raise LocalGenerationError("Agent action/ID does not match original admissible action mapping")
        return replace(decision, generation={**decision.generation, **metadata})


class FaultLogStream:
    """The existing append_jsonl writes one complete JSON line; underlying files retain ownership."""
    def __init__(self, stream, adapter):
        self.stream, self.adapter = stream, adapter

    def write(self, text):
        record = json.loads(text)
        metadata = self.adapter.metadata.get(record.get("step"))
        if metadata is None:
            raise RuntimeError("No input provenance for this decision step")
        if record.get("goal") != metadata["original_goal"] or record.get("observation") != metadata["original_observation"]:
            raise RuntimeError("Normal collector lost original environment inputs")
        record.update(metadata)
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return len(text)

    def flush(self):
        return self.stream.flush()

    def fileno(self):
        return self.stream.fileno()


def run_condition(game, episode_id, seed, max_steps, agent, config, output_dir, *, condition="fault", timing=None, env_factory=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = getattr(agent, "tokenizer", None)
    counter = (lambda text: len(tokenizer.encode(text, add_special_tokens=False))) if callable(getattr(tokenizer, "encode", None)) else None
    actual_config = FaultConfig("no_fault", config.severity, None) if condition == "no_fault" else config
    injector = FaultInjector(actual_config, seed, episode_id, counter)
    adapter = FaultAgent(agent, injector)
    save_json(output_dir / "config.json", {"condition": condition, "episode_id": episode_id, "seed": seed,
                                           "fault": asdict(actual_config), "timing": timing,
                                           "local_llm": agent.config.public_config(),
                                           "input_boundary": "Only agent goal/observation changed; env state, original logs, actions/history order untouched"})
    with (output_dir / "trajectories.jsonl").open("x", encoding="utf-8") as raw, (output_dir / "decisions.jsonl").open("x", encoding="utf-8") as decisions:
        summary = run_episode(game, episode_id, seed, max_steps, adapter, FaultLogStream(raw, adapter), FaultLogStream(decisions, adapter),
                              logits_dir=output_dir / "logits", env_factory=env_factory)
    active_steps = [step for step, value in adapter.metadata.items() if value["fault_active"]]
    effective_steps = [step for step, value in adapter.metadata.items() if value["fault_payload"]["applied"]]
    summary.update(condition=condition, seed=seed, fault_type=actual_config.fault_type,
                   injection_step=actual_config.injection_step, first_active_step=min(active_steps) if active_steps else None,
                   first_effective_input_step=min(effective_steps) if effective_steps else None,
                   injection_reached=bool(active_steps), effective_perturbation_reached=bool(effective_steps))
    save_json(output_dir / "episodes.json", [summary])
    from src.validate_fault import validate
    report = validate(output_dir, expected_episodes=1)
    save_json(output_dir / "validation.json", report)
    return summary, report


def compare_pair(no_fault, fault, no_fault_dir, fault_dir):
    def rows(path):
        with (Path(path) / "trajectories.jsonl").open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream]
    baseline_rows, fault_rows = rows(no_fault_dir), rows(fault_dir)
    before = (fault["injection_step"] or 2) - 1
    keys = ("goal", "observation", "previous_actions", "admissible_actions", "action_id", "action", "reward", "done", "success")
    baseline_prefix = [{key: row.get(key) for key in keys} for row in baseline_rows[:before]]
    fault_prefix = [{key: row.get(key) for key in keys} for row in fault_rows[:before]]
    prefix_match = len(baseline_prefix) == before and baseline_prefix == fault_prefix
    if no_fault["outcome"] in ("task_failure", "step_limit"):
        comparison = "baseline_already_failed"
    elif no_fault["outcome"] != "success":
        comparison = "baseline_agent_or_runtime_error"
    elif not fault["effective_perturbation_reached"]:
        comparison = "fault_not_effectively_reached"
    elif not prefix_match:
        comparison = "pre_injection_divergence"
    elif fault["outcome"] in ("task_failure", "step_limit"):
        comparison = "baseline_success_fault_task_failure"
    elif fault["outcome"] == "success":
        comparison = "both_success"
    else:
        comparison = "fault_condition_agent_or_runtime_error"
    return {"episode_id": no_fault["episode_id"], "seed": no_fault["seed"], "comparison": comparison,
            "pre_injection_prefix_match": prefix_match, "baseline_outcome": no_fault["outcome"],
            "fault_outcome": fault["outcome"], "baseline_success": no_fault["success"], "fault_success": fault["success"],
            "baseline_steps": no_fault["steps"], "fault_steps": fault["steps"],
            "injection_step": fault["injection_step"], "first_active_step": fault["first_active_step"],
            "first_effective_input_step": fault["first_effective_input_step"],
            "interpretation": "Paired observed outcomes, not proof of causality; GPU nondeterminism and pre-injection divergence must be considered"}
