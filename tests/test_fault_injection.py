"""Input-fault unit tests use fake agents/environments only; no model or ALFWorld runtime."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import collect_fault
from local_llm_agent import LocalDecision, LocalGenerationError, LocalInvalidOutput, LocalLLMConfig
from src.fault_injection import FAULTS, FaultConfig, FaultInjector, resolve_timing
from src.fault_runtime import FaultAgent, run_condition
from src.validate_fault import validate

GOAL = "put a watch in shelf."
OBS = "You arrive at shelf 1. On the shelf 1, you see a watch 1, a book 2, and a mug 1."


def generation(action_id):
    text = f"PLAN: test-only plan\nACTION_ID: {action_id}"
    return {"raw_text": text, "raw_model_output": text, "action_selection_format": "action_id_v1",
            "generated_token_ids": [0], "token_entropies": [math.log(2)], "mean_entropy": math.log(2),
            "score_vocab_size": 2, "entropy_unit": "nats", "logits_file": None}


class FakeAgent:
    def __init__(self, *, always_wrong=False, diverge=False, invalid_after=None):
        self.config = LocalLLMConfig()
        self.tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: text.split())
        self.calls, self.always_wrong, self.diverge, self.invalid_after = [], always_wrong, diverge, invalid_after

    def decide(self, goal, observation, previous_actions, admissible_actions, **kwargs):
        self.calls.append(deepcopy((goal, observation, previous_actions, admissible_actions)))
        if self.invalid_after is not None and len(previous_actions) >= self.invalid_after:
            raise LocalInvalidOutput("test-only invalid ID", generation(-1), {"plan": "test-only plan", "action_id": -1, "action": None})
        action_id = 1 if self.always_wrong or goal != GOAL or (self.diverge and len(self.calls) == 4) else 0
        return LocalDecision("test-only plan", admissible_actions[action_id], generation(action_id), action_id)


class FakeEnv:
    def __init__(self, terminal_steps=3):
        self.true_goal, self.actions, self.executed = GOAL, ["look", "inventory"], []
        self.terminal_steps, self.closed = terminal_steps, False

    def initial(self):
        return {"feedback": OBS + "\nYour task is to: " + self.true_goal, "won": False, "admissible_commands": self.actions}

    def step(self, action):
        assert action in self.actions
        self.executed.append(action)
        done = len(self.executed) == self.terminal_steps
        won = done and all(value == "look" for value in self.executed)
        return {"feedback": OBS, "won": won, "admissible_commands": self.actions}, int(won), done

    def close(self):
        self.closed = True


def args(**overrides):
    values = dict(fault="goal_drift", severity="low", injection_step=None, reference_steps=20,
                  max_steps=5, paired=True, condition="fault", baseline_run=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class InjectorTests(unittest.TestCase):
    def test_before_injection_inputs_unchanged(self):
        for name in FAULTS:
            injector = FaultInjector(FaultConfig(name, "low", 3), 42, "fixture")
            for step in (1, 2):
                result = injector.apply(GOAL, OBS, step)
                self.assertEqual(result["agent_goal"], GOAL)
                self.assertEqual(result["agent_observation"], OBS)
                self.assertFalse(result["fault_active"])
                self.assertFalse(result["fault_payload"]["applied"])

    def test_goal_drift_applies_and_original_stable(self):
        injector = FaultInjector(FaultConfig("goal_drift", "low", 2), 42, "fixture")
        first, later = injector.apply(GOAL, OBS, 2), injector.apply(GOAL, "later observation", 7)
        self.assertNotEqual(first["agent_goal"], GOAL)
        self.assertEqual(first["original_goal"], GOAL)
        self.assertEqual(first["agent_observation"], OBS)
        self.assertEqual(first["agent_goal"], later["agent_goal"])
        self.assertIn("watch", first["agent_goal"])
        self.assertEqual(first["fault_payload"]["changes"][0]["role"], "receptacle")

    def test_high_goal_drift_can_change_object_and_receptacle(self):
        result = FaultInjector(FaultConfig("goal_drift", "high", 2), 7, "fixture").apply(GOAL, OBS, 2)
        self.assertEqual({row["role"] for row in result["fault_payload"]["changes"]}, {"object", "receptacle"})
        self.assertFalse(result["fault_payload"]["alternate_entities_present_in_scene_verified"])

    def test_observation_loss_preserves_original(self):
        result = FaultInjector(FaultConfig("observation_loss", "low", 2), 42, "fixture").apply(GOAL, OBS, 2)
        self.assertEqual(result["original_observation"], OBS)
        self.assertEqual(result["agent_goal"], GOAL)
        self.assertNotEqual(result["agent_observation"], OBS)
        self.assertIn("[OBSERVATION_GAP]", result["agent_observation"])
        self.assertTrue(result["fault_payload"]["removed_spans"])

    def test_observation_fallback_removes_span_deterministically(self):
        text = "You pick up the watch from the shelf."
        injector = FaultInjector(FaultConfig("observation_loss", "high", 2), 11, "fixture")
        result = injector.apply(GOAL, text, 2)
        self.assertNotEqual(result["agent_observation"], text)
        self.assertEqual(result, injector.apply(GOAL, text, 2))

    def test_context_overload_severity_and_actual_counts(self):
        results = [FaultInjector(FaultConfig("context_overload", severity, 2), 42, "fixture", lambda text: len(text.split())).apply(GOAL, OBS, 2)
                   for severity in ("low", "high")]
        low, high = [result["fault_payload"] for result in results]
        self.assertGreater(high["added_context_char_count"], low["added_context_char_count"])
        self.assertGreater(high["added_context_token_count"], low["added_context_token_count"])
        for result in results:
            payload = result["fault_payload"]
            self.assertEqual(payload["added_context_char_count"], len(payload["added_context"]))
            self.assertEqual(result["agent_goal"], GOAL)
            self.assertEqual(result["agent_observation"], OBS + "\n\n" + payload["added_context"])

    def test_no_tokenizer_never_invents_token_count(self):
        result = FaultInjector(FaultConfig("context_overload", "low", 2), 42, "fixture").apply(GOAL, OBS, 2)
        self.assertIsNone(result["fault_payload"]["added_context_token_count"])

    def test_deterministic_and_does_not_advance_global_rng(self):
        state = random.getstate()
        for name in FAULTS:
            config = FaultConfig(name, "high", 2)
            first = FaultInjector(config, 123, "fixture").apply(GOAL, OBS, 3)
            second = FaultInjector(config, 123, "fixture").apply(GOAL, OBS, 3)
            self.assertEqual(first, second)
        self.assertEqual(random.getstate(), state)

    def test_unavailable_original_inputs_stay_unavailable(self):
        for name in FAULTS:
            result = FaultInjector(FaultConfig(name, "low", 2), 1, "fixture").apply(None, None, 2)
            self.assertIsNone(result["agent_goal"])
            self.assertIsNone(result["agent_observation"])
            self.assertFalse(result["fault_payload"]["applied"])

    def test_timing_sources_and_early_episode_case(self):
        self.assertEqual(resolve_timing(baseline_steps=3)["injection_step"], 2)
        self.assertEqual(resolve_timing(baseline_steps=4)["injection_step"], 3)
        self.assertEqual(resolve_timing(reference_steps=20)["injection_step"], 11)
        self.assertEqual(resolve_timing(requested=4)["source"], "explicit")
        self.assertEqual(resolve_timing(baseline_steps=1)["injection_step"], 2)
        for step in (0, 1, 51):
            with self.assertRaises(ValueError):
                resolve_timing(requested=step)


class BoundaryTests(unittest.TestCase):
    def test_admissible_actions_history_and_model_config_unchanged(self):
        actions, history = ["look", "inventory"], ["look"]
        original_actions, original_history = list(actions), list(history)
        agent = FakeAgent()
        original_config = asdict(agent.config)
        adapter = FaultAgent(agent, FaultInjector(FaultConfig("goal_drift", "low", 2), 42, "fixture"))
        first = adapter.decide(GOAL, OBS, [], actions)
        second = adapter.decide(GOAL, OBS, history, actions)
        self.assertEqual(first.action, actions[first.action_id])
        self.assertEqual(second.action, actions[second.action_id])
        self.assertEqual(actions, original_actions)
        self.assertEqual(history, original_history)
        self.assertEqual(agent.calls[-1][3], original_actions)
        self.assertEqual(asdict(agent.config), original_config)

    def test_reject_mutating_agent_without_mutating_original_actions(self):
        agent = FakeAgent()
        original = agent.decide
        def bad(goal, observation, previous, actions, **kwargs):
            actions.reverse()
            return original(goal, observation, previous, actions, **kwargs)
        agent.decide = bad
        actions = ["look", "inventory"]
        adapter = FaultAgent(agent, FaultInjector(FaultConfig("no_fault", "low", None), 1, "fixture"))
        with self.assertRaises(LocalGenerationError):
            adapter.decide(GOAL, OBS, [], actions)
        self.assertEqual(actions, ["look", "inventory"])

    def test_metadata_logged_and_true_environment_goal_not_changed(self):
        env = FakeEnv()
        agent = FakeAgent()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fault"
            summary, report = run_condition(None, "fixture", 42, 5, agent, FaultConfig("goal_drift", "low", 2), path,
                                             env_factory=lambda game, seed: (env, env.initial()))
            rows = [json.loads(line) for line in (path / "trajectories.jsonl").read_text().splitlines()]
            decisions = [json.loads(line) for line in (path / "decisions.jsonl").read_text().splitlines()]
            self.assertTrue(report["data_valid"])
            self.assertEqual(summary["outcome"], "task_failure")
            self.assertFalse(summary["success"])
            self.assertEqual(summary["first_active_step"], 2)
            self.assertEqual(env.true_goal, GOAL)
            self.assertTrue(env.closed)
            for records in (rows, decisions):
                self.assertFalse(records[0]["fault_active"])
                self.assertTrue(records[1]["fault_active"])
                self.assertEqual(records[1]["goal"], GOAL)
                self.assertEqual(records[1]["observation"], records[1]["original_observation"])
                self.assertNotEqual(records[1]["agent_goal"], GOAL)
                self.assertEqual(records[1]["admissible_actions"], ["look", "inventory"])
                self.assertEqual(records[1]["token_entropies"], [math.log(2)])
            rows[1]["original_goal"] = "test-only corruption of provenance"
            (path / "trajectories.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertFalse(validate(path)["data_valid"])

    def test_invalid_faulted_attempt_preserves_metadata_without_fake_transition(self):
        env = FakeEnv()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fault"
            summary, report = run_condition(None, "fixture", 42, 5, FakeAgent(invalid_after=1), FaultConfig("goal_drift", "low", 2), path,
                                             env_factory=lambda game, seed: (env, env.initial()))
            self.assertEqual(summary["outcome"], "invalid_output")
            self.assertEqual(summary["steps"], 1)
            self.assertTrue(report["data_valid"])
            self.assertFalse(report["experiment_complete"])
            records = [json.loads(line) for line in (path / "decisions.jsonl").read_text().splitlines()]
            self.assertTrue(records[-1]["fault_active"])
            self.assertEqual(records[-1]["original_goal"], GOAL)
            self.assertIsNone(records[-1]["action"])
            self.assertEqual(len(env.executed), 1)


class PairedTests(unittest.TestCase):
    def run_pair(self, agent, *, terminal_steps=3, **overrides):
        self.envs, self.seeds = [], []
        def factory(game, seed):
            env = FakeEnv(terminal_steps)
            self.envs.append((game, seed, env))
            return env, env.initial()
        with tempfile.TemporaryDirectory() as temp:
            result = collect_fault.run_game("same-game", "same-episode", 17, agent, args(**overrides), Path(temp) / "pair",
                                            seed_runtime=self.seeds.append, env_factory=factory)
            self.assertTrue(all(report["data_valid"] for report in result["reports"]))
            return result

    def test_paired_same_game_seed_and_midpoint_from_real_baseline(self):
        result = self.run_pair(FakeAgent())
        self.assertEqual(self.seeds, [17, 17])
        self.assertEqual([(game, seed) for game, seed, env in self.envs], [("same-game", 17), ("same-game", 17)])
        self.assertEqual(result["comparison"]["comparison"], "baseline_success_fault_task_failure")
        self.assertTrue(result["comparison"]["pre_injection_prefix_match"])
        self.assertEqual(result["comparison"]["injection_step"], 2)

    def test_baseline_failures_not_claimed_as_fault_caused(self):
        result = self.run_pair(FakeAgent(always_wrong=True))
        self.assertEqual(result["comparison"]["comparison"], "baseline_already_failed")

    def test_pre_injection_divergence_not_claimed_as_fault_effect(self):
        result = self.run_pair(FakeAgent(diverge=True))
        self.assertEqual(result["comparison"]["comparison"], "pre_injection_divergence")

    def test_early_completion_records_fault_not_reached(self):
        result = self.run_pair(FakeAgent(), terminal_steps=1)
        fault = result["summaries"][1]
        self.assertEqual(fault["injection_step"], 2)
        self.assertIsNone(fault["first_active_step"])
        self.assertFalse(fault["effective_perturbation_reached"])
        self.assertEqual(result["comparison"]["comparison"], "fault_not_effectively_reached")

    def test_no_fault_only_does_not_transform_inputs(self):
        agent = FakeAgent()
        self.run_pair(agent, paired=False, condition="no_fault")
        self.assertTrue(all(goal == GOAL for goal, obs, previous, actions in agent.calls))
        self.assertEqual(self.seeds, [17])


class CliTests(unittest.TestCase):
    def test_single_condition_auto_reference_is_labeled_and_not_sent_as_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            game = Path(temp) / "game.tw-pddl"
            game.write_text(json.dumps({"walkthrough": ["test-only-hidden-reference"] * 6}))
            timing = collect_fault.resolve_game_timing(args(reference_steps=None), game)
            self.assertEqual(timing["injection_step"], 4)
            self.assertEqual(timing["source"], "pddl_walkthrough_reference_steps")
            self.assertNotIn("test-only-hidden-reference", json.dumps(timing))
            explicit = collect_fault.resolve_game_timing(args(reference_steps=10), game)
            self.assertEqual(explicit["injection_step"], 5)  # Clamped to max_steps=5.
            actual = collect_fault.resolve_game_timing(args(reference_steps=None), game, baseline_steps=3)
            self.assertEqual(actual["source"], "baseline_actual_steps")
            self.assertEqual(actual["injection_step"], 2)

    def test_cli_dispatch_mock_only(self):
        with patch.object(collect_fault, "collect", return_value=0) as collect, patch("sys.argv", ["collect_fault.py", "--agent", "local", "--model", "Qwen/Qwen3-4B", "--fault", "goal_drift", "--severity", "low", "--episodes", "1", "--paired"]):
            self.assertEqual(collect_fault.main(), 0)
        selected, config = collect.call_args.args
        self.assertTrue(selected.paired)
        self.assertEqual(config.model_name, "Qwen/Qwen3-4B")

    def test_fixed_game_and_episode_id_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            game = root / "json_2.1.1/train/task/trial/game.tw-pddl"
            game.parent.mkdir(parents=True)
            game.write_text('{"test_only":true}')
            selected = SimpleNamespace(data_dir=root, game=None, episode_id="train/task/trial", episodes=1)
            self.assertEqual(collect_fault.select_experiment_games(selected), [game.resolve()])
            selected.episode_id = "../../escape"
            with self.assertRaises(ValueError):
                collect_fault.select_experiment_games(selected)
            selected.game, selected.episode_id = game, None
            self.assertEqual(collect_fault.game_identity(game, root), "train/task/trial")
            self.assertEqual(collect_fault.select_experiment_games(selected), [game.resolve()])


if __name__ == "__main__":
    unittest.main()
