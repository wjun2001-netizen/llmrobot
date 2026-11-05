"""Deterministic input-only faults. This module never receives an environment or action list."""
from dataclasses import dataclass
import hashlib
import json
import random
import re
from typing import Callable

FAULTS = ("goal_drift", "observation_loss", "context_overload")
OBJECTS = ("book", "mug", "watch", "apple", "alarmclock", "keychain", "remotecontrol")
RECEPTACLES = ("desk", "shelf", "cabinet", "coffeetable", "countertop", "sidetable", "drawer")


@dataclass(frozen=True)
class FaultConfig:
    fault_type: str
    severity: str = "low"
    injection_step: int | None = 2

    def __post_init__(self):
        if self.fault_type not in (*FAULTS, "no_fault"):
            raise ValueError("Unsupported fault type")
        if self.severity not in ("low", "high"):
            raise ValueError("Severity must be low or high")
        if self.fault_type != "no_fault" and (type(self.injection_step) is not int or self.injection_step < 2):
            raise ValueError("Injection must start at step 2 or later")


def resolve_timing(requested=None, baseline_steps=None, reference_steps=20, max_steps=50):
    if max_steps < 2 or reference_steps < 1:
        raise ValueError("Need max_steps >= 2 and reference_steps >= 1")
    if requested is not None:
        if type(requested) is not int or not 2 <= requested <= max_steps:
            raise ValueError("injection_step must be between 2 and max_steps")
        return {"injection_step": requested, "source": "explicit", "reference_length": None}
    if baseline_steps is not None and (type(baseline_steps) is not int or baseline_steps < 1):
        raise ValueError("Baseline length must be a positive integer")
    length = baseline_steps if baseline_steps is not None else reference_steps
    return {"injection_step": min(max_steps, max(2, length // 2 + 1)),
            "source": "baseline_actual_steps" if baseline_steps is not None else "configured_reference_steps",
            "reference_length": length}


class FaultInjector:
    def __init__(self, config, seed, episode_id, token_counter: Callable[[str], int] | None = None):
        self.config, self.seed, self.episode_id, self.token_counter = config, seed, episode_id, token_counter

    def _rng(self, tag):
        # Independent RNG: never advance model/environment/global random state.
        identity = json.dumps([self.seed, self.episode_id, self.config.fault_type, self.config.severity, tag], ensure_ascii=False)
        return random.Random(int.from_bytes(hashlib.sha256(identity.encode()).digest(), "big"))

    def apply(self, goal, observation, step):
        active = self.config.fault_type != "no_fault" and step >= self.config.injection_step
        result = {"fault_type": self.config.fault_type, "fault_severity": self.config.severity if self.config.fault_type != "no_fault" else None,
                  "fault_active": active, "injection_step": self.config.injection_step if self.config.fault_type != "no_fault" else None,
                  "original_goal": goal, "agent_goal": goal, "original_observation": observation,
                  "agent_observation": observation, "fault_payload": {"applied": False}}
        if not active:
            return result
        if self.config.fault_type == "goal_drift":
            result["agent_goal"], result["fault_payload"] = self._goal_drift(goal)
        elif self.config.fault_type == "observation_loss":
            result["agent_observation"], result["fault_payload"] = self._observation_loss(observation, step)
        else:
            result["agent_observation"], result["fault_payload"] = self._context_overload(observation)
        return result

    def _goal_drift(self, goal):
        if not isinstance(goal, str) or not goal.strip():
            return goal, {"applied": False, "reason": "goal_unavailable"}
        rng = self._rng("stable_goal")
        changes, result = [], goal
        target = re.search(r"\b(?:in|on|into|onto)\s+(?:(?:a|an|the)\s+)?(?P<target>[a-z][a-z0-9_-]*)[.!]?\s*$", result, re.I)
        if target:
            old = target.group("target")
            new = rng.choice([name for name in RECEPTACLES if name.lower() != old.lower()])
            result = result[:target.start("target")] + new + result[target.end("target"):]
            changes.append({"role": "receptacle", "original": old, "replacement": new})
        obj = re.search(r"\b(?:a|an|the|two)\s+(?:(?:clean|hot|cold|cool|sliced)\s+)*(?P<object>[a-z][a-z0-9_-]*)\b", result, re.I)
        if obj and (self.config.severity == "high" or not changes):
            old = obj.group("object")
            new = rng.choice([name for name in OBJECTS if name.lower() != old.lower()])
            result = result[:obj.start("object")] + new + result[obj.end("object"):]
            changes.append({"role": "object", "original": old, "replacement": new})
        if not changes:
            result = f"Put a {rng.choice(OBJECTS)} in {rng.choice(RECEPTACLES)}."
        return result, {"applied": result != goal, "strategy": "task_entity_replacement" if changes else "alternate_goal_fallback",
                        "changes": changes, "alternate_entities_present_in_scene_verified": False}

    def _observation_loss(self, observation, step):
        if not isinstance(observation, str) or not observation.strip():
            return observation, {"applied": False, "reason": "observation_unavailable"}
        rng = self._rng(["observation", step, observation])
        matches = list(re.finditer(r"\byou see\b(?P<details>[^.!?\n]+)", observation, re.I))
        spans = []
        if matches:
            selected = matches if self.config.severity == "high" else [rng.choice(matches)]
            for match in selected:
                start, end = match.span("details")
                if self.config.severity == "low":
                    tokens = list(re.finditer(r"\S+", observation[start:end]))
                    if tokens:
                        offset = rng.randrange(max(1, len(tokens) // 2))
                        start += tokens[offset].start()
                spans.append((start, end))
            strategy = "visibility_detail_removal"
        else:
            tokens = list(re.finditer(r"\S+", observation))
            if len(tokens) < 2:
                return observation, {"applied": False, "reason": "insufficient_observation_text"}
            count = max(1, len(tokens) * (2 if self.config.severity == "high" else 1) // 3)
            start_index = rng.randrange(len(tokens) - count + 1)
            spans = [(tokens[start_index].start(), tokens[start_index + count - 1].end())]
            strategy = "contiguous_token_span_removal"
        removed = [{"start": start, "end": end, "text": observation[start:end]} for start, end in sorted(spans)]
        result = observation
        for start, end in sorted(spans, reverse=True):
            result = result[:start] + " [OBSERVATION_GAP]" + result[end:]
        return result, {"applied": result != observation, "strategy": strategy, "removed_spans": removed}

    def _context_overload(self, observation):
        if not isinstance(observation, str):
            return observation, {"applied": False, "reason": "observation_unavailable"}
        rng = self._rng("stable_unrelated_context")
        count = 8 if self.config.severity == "low" else 64
        lines = [f"Archive note {i:03d}: station {rng.randrange(100, 999)} recorded a {rng.choice(('blue', 'green', 'silver', 'amber'))} banner during an unrelated weather exhibition."
                 for i in range(count)]
        added = "[UNRELATED_CONTEXT]\n" + "\n".join(lines) + "\n[/UNRELATED_CONTEXT]"
        tokens, token_count_error = None, None
        if self.token_counter is not None:
            try:
                tokens = self.token_counter(added)
            except Exception as exc:
                token_count_error = type(exc).__name__
        return observation + "\n\n" + added, {"applied": True, "added_context": added, "added_context_char_count": len(added),
                                               "added_context_token_count": tokens, "line_count": count,
                                               "token_count_error": token_count_error,
                                               "token_count_scope": "Added block encoded alone with add_special_tokens=False; not full prompt delta"}
