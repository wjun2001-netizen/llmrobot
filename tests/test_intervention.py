"""Synthetic CPU/mocked intervention tests; never load Qwen, MiniLM or ALFWorld."""
from copy import deepcopy
from dataclasses import replace
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_intervention
from local_llm_agent import LocalDecision, LocalGenerationError, LocalInvalidOutput, LocalLLMConfig
from src.fault_injection import FaultConfig, FaultInjector
from src.intervention import (InterventionConfig, LSTMRiskDetector, MODES, choose_intervention,
                              decision_features, training_entropy_threshold)
from src.intervention_runtime import (InterventionAgent, ROOT, compare_summaries, execute_experiment,
                                      run_mode, safe_output_root)
from src.lstm_classifier import FailureRiskLSTM
from src.lstm_training_data import group_split, load_windows
from src.timeseries_features import FeatureError, PromptCounter, SentenceSimilarity, digest

GOAL = "put a box on sofa."
OBS = "You see a box 1 on shelf 1."
ACTIONS = ["look", "inventory"]


class FakeSimilarity:
    model_name = "test-only-semantic-backend"
    def __init__(self, score=.9):
        self.score, self.inputs = score, []

    def score_pairs(self, pairs):
        self.inputs.extend(pairs)
        return [self.score for _ in pairs]


class FakeDetector:
    def __init__(self, risk=.8):
        self.risk, self.windows = risk, []

    def predict(self, window):
        self.windows.append(deepcopy(window))
        return self.risk


class FakeAgent:
    def __init__(self, entropy=2., wrong=False, invalid_call=None, mutation=False):
        self.config = LocalLLMConfig()
        self.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
        self.calls, self.entropy, self.wrong = [], entropy, wrong
        self.invalid_call, self.mutation = invalid_call, mutation

    def decide(self, goal, observation, previous_actions, admissible_actions, *, logits_path=None):
        self.calls.append(deepcopy((goal, observation, previous_actions, admissible_actions)))
        if self.mutation:
            admissible_actions.reverse()
        index = 1 if self.wrong else 0
        text = f"PLAN: carry the box to the sofa\nACTION_ID: {index}"
        count = 50 + len(observation.split()) + sum(len(value.split()) for value in previous_actions)
        record = {"raw_text": text, "raw_model_output": text, "token_entropies": [self.entropy],
                  "mean_entropy": self.entropy, "prompt_token_count": count, "max_context_tokens": 4096,
                  "entropy_unit": "nats", "generated_token_ids": [1], "logits_file": None}
        if self.invalid_call == len(self.calls):
            raise LocalInvalidOutput("fixture invalid ACTION_ID", record)
        return LocalDecision("carry the box to the sofa", admissible_actions[index], record, index)


class FakeEnv:
    def __init__(self, steps=7):
        self.goal, self.actions = GOAL, list(ACTIONS)
        self.executed, self.terminal_steps, self.closed, self.reset_count = [], steps, False, 0

    def initial(self):
        return {"feedback": OBS + "\nYour task is to: " + self.goal, "won": False, "admissible_commands": self.actions}

    def reset(self):
        self.reset_count += 1
        raise AssertionError("No reset during intervention")

    def step(self, action):
        assert action in self.actions
        self.executed.append(action)
        done = len(self.executed) >= self.terminal_steps
        success = done and all(value == "look" for value in self.executed)
        return {"feedback": OBS, "won": success, "admissible_commands": self.actions}, int(success), done

    def close(self):
        self.closed = True


def config(mode="proposed", **kwargs):
    if mode == "proposed":
        kwargs.setdefault("calibration", {"statistics": {"E_mean": 1., "E_std": 1., "C_mean": .1, "C_std": .1,
                                                          "S_mean": .9, "S_std": .1}, "std_floor": 1e-6})
    return InterventionConfig(mode=mode, entropy_threshold=1., **kwargs)


def checkpoint(path, bias=0.):
    model = FailureRiskLSTM()
    with torch.no_grad():
        for value in model.parameters():
            value.zero_()
        model.head.bias.fill_(bias)
    architecture = {"input_size": 3, "hidden_size": 16, "num_layers": 1, "batch_first": True,
                    "window_size": 5, "feature_order": ["E", "C", "S"], "output_size": 1}
    torch.save({"state_dict": model.state_dict(), "architecture": architecture, "input_sha256": "fixture"}, path)


class RuleTests(unittest.TestCase):
    def choose(self, cfg, features=None, risk=.8, step=6, active=True):
        return choose_intervention(cfg, features or {"E": 2., "C": .9, "S": .1}, risk,
                                   step=step, injection_step=2, fault_active=active)

    def test_four_modes_and_cli_choices(self):
        for mode in MODES:
            self.assertEqual(run_intervention.build_parser().parse_args(["--mode", mode]).mode, mode)
        self.assertEqual(len(MODES), 4)
        self.assertTrue(run_intervention.build_parser().parse_args(["--compare-all"]).compare_all)

    def test_no_intervention_and_before_injection(self):
        self.assertIsNone(self.choose(config("no_intervention"))[0])
        for mode in MODES:
            self.assertIsNone(self.choose(config(mode), active=False)[0])

    def test_periodic_does_not_require_risk(self):
        cfg = config("periodic_goal_reinjection")
        self.assertEqual(self.choose(cfg, risk=None, step=6)[0], "goal_reinjection")
        self.assertIsNone(self.choose(cfg, risk=1., step=5)[0])
        self.assertEqual(self.choose(replace(cfg, period=2), risk=None, step=3)[0], "goal_reinjection")

    def test_entropy_only_uses_only_E(self):
        cfg = config("entropy_only")
        self.assertEqual(self.choose(cfg, {"E": 1.01, "C": 0., "S": 1.}, risk=None)[0], "reobservation_replanning")
        self.assertIsNone(self.choose(cfg, {"E": 1., "C": 10., "S": -1.}, risk=1.)[0])

    def test_lstm_threshold_and_warmup(self):
        self.assertIsNone(self.choose(config(), risk=.4999)[0])
        self.assertIsNone(self.choose(config(), risk=None)[0])
        self.assertEqual(self.choose(config(), risk=.5)[0], "context_cleaning")

    def test_largest_standardized_anomaly_replaces_old_priority(self):
        self.assertEqual(self.choose(config())[0], "context_cleaning")
        self.assertEqual(self.choose(config(), {"E": 2., "C": .8, "S": .9})[0], "context_cleaning")
        self.assertEqual(self.choose(config(), {"E": 2., "C": .1, "S": .9})[0], "reobservation_replanning")
        self.assertIsNone(self.choose(config(), {"E": .1, "C": .1, "S": .9})[0])

    def test_invalid_thresholds(self):
        for kwargs in ({"lstm_risk_threshold": 2.}, {"similarity_threshold": float("nan")},
                       {"context_threshold": 0}, {"entropy_threshold": -1}, {"period": 0}, {"history_keep": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                InterventionConfig(**kwargs)


class AdapterTests(unittest.TestCase):
    def adapter(self, cfg=None, score=.9, risk=.8, fault="goal_drift", agent=None):
        return InterventionAgent(agent or FakeAgent(), FaultInjector(FaultConfig(fault, "high", 2), 42, "fixture"),
                                 cfg or config(), FakeSimilarity(score), FakeDetector(risk))

    def advance(self, adapter, count=5):
        previous = []
        result = None
        for _ in range(count):
            result = adapter.decide(GOAL, OBS, previous, list(ACTIONS))
            previous.append(result.action)
        return result

    def test_goal_reinjection_and_original_goal(self):
        adapter = self.adapter(score=.1)
        self.advance(adapter)
        event = adapter.metadata[5]
        self.assertNotEqual(event["agent_goal_before"], GOAL)
        self.assertEqual(event["agent_goal_after"], GOAL)
        self.assertEqual(event["original_goal"], GOAL)
        self.assertEqual(event["intervention_type"], "goal_reinjection")
        self.assertEqual(adapter.similarity.inputs[-1][0], GOAL)

    def test_context_cleaning_persists_but_injected_noise_stays(self):
        adapter = self.adapter(config(context_threshold=.001), fault="context_overload")
        self.advance(adapter, 6)
        event = adapter.metadata[5]
        self.assertEqual(event["history_count_before"], 4)
        self.assertEqual(event["history_count_after"], 3)
        self.assertLess(event["context_length_after"], event["context_length_before"])
        self.assertIn("[UNRELATED_CONTEXT]", adapter.agent.calls[-1][1])
        self.assertTrue(adapter.metadata[6]["fault_payload"]["applied"])
        self.assertEqual(adapter.metadata[6]["history_count_before"], 4)
        self.assertEqual(adapter.agent.calls[-1][2], ["look"] * 3)

    def test_reobservation_uses_current_observation(self):
        adapter = self.adapter(config("entropy_only"), fault="observation_loss")
        self.advance(adapter, 2)
        event = adapter.metadata[2]
        self.assertNotEqual(event["agent_observation_before"], OBS)
        self.assertEqual(event["agent_observation_after"], OBS)
        self.assertEqual(event["original_observation"], OBS)
        self.assertEqual(event["intervention_type"], "reobservation_replanning")
        self.assertEqual(adapter.agent.calls[-1][1], OBS)

    def test_no_intervention_inputs_equal_unchanged_injector(self):
        adapter = self.adapter(config("no_intervention"))
        self.advance(adapter, 3)
        self.assertEqual(len(adapter.agent.calls), 3)
        for step, call in enumerate(adapter.agent.calls, 1):
            expected = adapter.injector.apply(GOAL, OBS, step)
            self.assertEqual(call[:2], (expected["agent_goal"], expected["agent_observation"]))
            self.assertEqual(call[2], ["look"] * (step - 1))
            self.assertEqual(call[3], ACTIONS)

    def test_at_most_one_intervention_per_step(self):
        adapter = self.adapter(score=.1, fault="context_overload")
        self.advance(adapter, 7)
        self.assertEqual(len(adapter.agent.calls), 10)  # 7 candidates + 3 one-shot replans.
        self.assertTrue(all(event["intervention_count_this_step"] <= 1 for event in adapter.metadata.values()))

    def test_other_modes_do_not_call_LSTM(self):
        class ForbiddenDetector:
            def predict(self, window):
                raise AssertionError("Baseline must never use LSTM inference")
        for mode in MODES[:-1]:
            with self.subTest(mode=mode):
                adapter = self.adapter(config(mode))
                adapter.detector = ForbiddenDetector()
                self.advance(adapter, 6)
                self.assertTrue(all(event["lstm_risk"] is None for event in adapter.metadata.values()))

    def test_window_is_time_major_and_warmup(self):
        adapter = self.adapter(risk=.1)
        self.advance(adapter, 6)
        self.assertEqual(len(adapter.detector.windows), 2)
        self.assertEqual(adapter.detector.windows[0], [[adapter.metadata[step][key] for key in ("E", "C", "S")] for step in range(1, 6)])
        self.assertTrue(all(adapter.metadata[step]["lstm_risk"] is None for step in range(1, 5)))

    def test_mapping_is_unchanged_and_mutation_rejected(self):
        actions = list(ACTIONS)
        adapter = self.adapter(score=.1)
        decision = adapter.decide(GOAL, OBS, [], actions)
        self.assertEqual(actions, ACTIONS)
        self.assertEqual(decision.action, actions[decision.action_id])
        with self.assertRaises(LocalGenerationError):
            self.adapter(agent=FakeAgent(mutation=True)).decide(GOAL, OBS, [], actions)
        self.assertEqual(actions, ACTIONS)

    def test_wrong_action_id_not_repaired(self):
        agent = FakeAgent()
        original = agent.decide
        def wrong(*args, **kwargs):
            return replace(original(*args, **kwargs), action_id=100)
        agent.decide = wrong
        with self.assertRaises(LocalGenerationError):
            self.adapter(agent=agent).decide(GOAL, OBS, [], list(ACTIONS))

    def test_final_features_and_detector_features_both_preserved(self):
        adapter = self.adapter(config(context_threshold=.001), fault="context_overload")
        decision = self.advance(adapter)
        metadata = adapter.metadata[5]
        self.assertEqual(metadata["E"], decision.generation["mean_entropy"])
        self.assertEqual(metadata["C"], decision.generation["prompt_token_count"] / 4096)
        self.assertNotEqual(metadata["C"], metadata["detection_features"]["C"])
        self.assertEqual(metadata["lstm_input_window"][-1][1], metadata["detection_features"]["C"])
        self.assertEqual(adapter.history[-1][1], metadata["C"])
        self.assertAlmostEqual(metadata["diagnostic_C_z"], (metadata["detection_features"]["C"] - .1) / .1)
        self.assertEqual(metadata["selected_anomaly"], "C")

    def test_missing_live_prompt_count_or_NaN_rejected(self):
        agent = FakeAgent()
        decision = agent.decide(GOAL, OBS, [], ACTIONS)
        for change in ({"mean_entropy": float("nan")}, {"prompt_token_count": None}):
            bad = replace(decision, generation={**decision.generation, **change})
            with self.assertRaises(FeatureError):
                decision_features(bad, GOAL, agent.config, FakeSimilarity(), PromptCounter())
        del decision.generation["prompt_token_count"]
        with self.assertRaises(FeatureError):
            decision_features(decision, GOAL, agent.config, FakeSimilarity(), PromptCounter())


class DiskTests(unittest.TestCase):
    def setUp(self):
        output = ROOT / "runs_intervention"
        output.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="unit_test_fixture_", dir=output)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_real_CPU_checkpoint_sigmoid_and_preservation(self):
        path = self.root / "model.pt"
        checkpoint(path, bias=math.log(3.))
        before = path.read_bytes()
        detector = LSTMRiskDetector(path)
        self.assertAlmostEqual(detector.predict([[1., .5, .2]] * 5), .75, places=6)
        self.assertFalse(detector.model.training)
        self.assertEqual(before, path.read_bytes())
        with self.assertRaises(ValueError):
            detector.predict([[1., 2., 3.]] * 4)

    def test_bad_checkpoint_architecture_rejected(self):
        path = self.root / "bad.pt"
        torch.save({"architecture": {"feature_order": ["E", "S", "C"]}}, path)
        with self.assertRaisesRegex(ValueError, "architecture"):
            LSTMRiskDetector(path)

    def test_episode_no_reset_and_required_fields(self):
        env, agent = FakeEnv(), FakeAgent()
        opens = []
        def factory(game, seed):
            opens.append((game, seed))
            return env, env.initial()
        summary = run_mode("fixture", "ep", 42, 7, agent, FaultConfig("context_overload", "high", 2),
                           config(context_threshold=.001), FakeSimilarity(), self.root / "one", detector=FakeDetector(), env_factory=factory)
        self.assertEqual(len(opens), 1)
        self.assertEqual(env.reset_count, 0)
        self.assertEqual(env.goal, GOAL)
        self.assertEqual(env.actions, ACTIONS)
        self.assertEqual(len(env.executed), 7)
        self.assertEqual(summary["steps"], 7)
        self.assertTrue(summary["success"])
        self.assertTrue(env.closed)
        rows = [json.loads(line) for line in (self.root / "one/trajectories.jsonl").read_text().splitlines()]
        required = ("E", "C", "S", "lstm_risk", "intervention_triggered", "intervention_type", "intervention_reason",
                    "original_goal", "agent_goal_before", "agent_goal_after", "context_length_before", "context_length_after",
                    "fault_type", "fault_severity", "fault_active", "success", "done", "token_entropies", "raw_model_output")
        required += ("diagnostic_E_z", "diagnostic_C_z", "diagnostic_S_z", "selected_anomaly", "selected_anomaly_score",
                     "calibration_E_mean", "calibration_E_std", "calibration_C_mean", "calibration_C_std",
                     "calibration_S_mean", "calibration_S_std")
        for row in rows:
            self.assertTrue(all(key in row for key in required))
            self.assertEqual(row["action"], row["admissible_actions"][row["action_id"]])
            self.assertEqual(row["E"], row["mean_entropy"])
            self.assertEqual(row["C"], row["prompt_token_count"] / row["max_context_tokens"])

    def test_invalid_replan_is_preserved_without_expert_fallback(self):
        env, agent = FakeEnv(), FakeAgent(invalid_call=3)
        summary = run_mode("fixture", "ep", 42, 7, agent, FaultConfig("goal_drift", "high", 2),
                           config("periodic_goal_reinjection", period=1), FakeSimilarity(), self.root / "invalid",
                           env_factory=lambda game, seed: (env, env.initial()))
        self.assertEqual(summary["outcome"], "invalid_output")
        self.assertEqual(env.executed, ["look"])
        records = [json.loads(line) for line in (self.root / "invalid/decisions.jsonl").read_text().splitlines()]
        self.assertTrue(records[-1]["intervention_triggered"])
        self.assertFalse(records[-1]["intervention_completed"])
        self.assertIsNotNone(records[-1]["pre_intervention_decision"])
        self.assertIsNone(records[-1]["prompt_token_count"])
        self.assertIsNone(records[-1]["E"])

    def test_four_mode_experiment_all_files_and_same_seed(self):
        envs, seeds = [], []
        def factory(game, seed):
            env = FakeEnv()
            envs.append(env)
            return env, env.initial()
        kwargs = dict(game="fixture", episode_id="ep", seed=42, max_steps=7, agent=FakeAgent(),
                      fault=FaultConfig("context_overload", "high", 2),
                      interventions=[config(mode) for mode in MODES], similarity=FakeSimilarity(),
                      output_root=self.root, metadata={"data_kind": "unit_test_fixture", "diagnostic_calibration": config().calibration}, detector=FakeDetector(),
                      env_factory=factory, seed_runtime=seeds.append)
        run, summaries, complete = execute_experiment(**kwargs)
        self.assertTrue(complete)
        self.assertEqual(seeds, [42] * 4)
        self.assertEqual(len(envs), 4)
        self.assertTrue(all(env.reset_count == 0 for env in envs))
        self.assertEqual([item["mode"] for item in summaries], list(MODES))
        for name in ("config.json", "trajectories.jsonl", "episodes.json", "comparison.json", "metrics.json", "run.log", "intervention_comparison.csv"):
            self.assertTrue((run / name).is_file(), name)
        with (run / "intervention_comparison.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)
        rows = [json.loads(line) for line in (run / "trajectories.jsonl").read_text().splitlines()]
        payloads = [row["fault_payload"] for row in rows if row["step"] == 2]
        self.assertEqual(len(payloads), 4)
        self.assertTrue(all(payload == payloads[0] for payload in payloads))
        self.assertTrue(all(item["pre_injection_prefix_match"] for item in summaries))
        self.assertTrue(all(item["false_intervention_count"] is None for item in summaries))
        self.assertEqual(json.loads((run / "metrics.json").read_text())["diagnostic_calibration"], config().calibration)
        saved = (run / "metrics.json").read_bytes()
        second, _, _ = execute_experiment(**kwargs)
        self.assertNotEqual(run, second)
        self.assertEqual((run / "metrics.json").read_bytes(), saved)

    def test_failed_episode_preserved(self):
        env = FakeEnv(steps=99)
        summary = run_mode("fixture", "ep", 42, 3, FakeAgent(wrong=True), FaultConfig("goal_drift", "low", 2),
                           config("no_intervention"), FakeSimilarity(), self.root / "failed",
                           env_factory=lambda game, seed: (env, env.initial()))
        self.assertEqual(summary["outcome"], "step_limit")
        self.assertFalse(summary["success"])
        self.assertEqual(len((self.root / "failed/trajectories.jsonl").read_text().splitlines()), 3)

    def test_output_escape_rejected(self):
        with self.assertRaises(ValueError):
            safe_output_root(ROOT / "features")
        with self.assertRaises(ValueError):
            safe_output_root(ROOT / "runs_intervention/../runs_fault")

    def test_entropy_calibration_train_only_deduplicated_and_hashed(self):
        rows = []
        fields = ["run_dir", "episode_id", "condition", "start_step", "end_step", "label"]
        fields += [f"{f}_{s}" for f in ("E", "C", "S") for s in range(1, 6)]
        for group in range(5):
            for start in (1, 2):
                row = dict(run_dir=f"fixture/{group}", episode_id=str(group), condition="fault", start_step=start, end_step=start+4, label=start-1)
                for step in range(1, 6):
                    row.update({f"E_{step}": float(start + step - 1), f"C_{step}": .2, f"S_{step}": .8})
                rows.append(row)
        windows, split_path = self.root / "windows.csv", self.root / "split.json"
        def write():
            with windows.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        write()
        split = group_split(load_windows(windows))
        for index in split["validation_indices"]:
            for step in range(1, 6):
                rows[index][f"E_{step}"] = 999.
        write()
        split["input_sha256"] = digest(windows)
        split_path.write_text(json.dumps(split))
        before = windows.read_bytes()
        threshold, basis = training_entropy_threshold(windows, split_path)
        self.assertEqual(threshold, 6.)
        self.assertEqual(basis["unique_training_step_count"], 24)
        self.assertEqual(basis["validation_samples_used"], 0)
        self.assertEqual(windows.read_bytes(), before)
        windows.write_text(windows.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            training_entropy_threshold(windows, split_path)

    def test_prepare_explicit_threshold_never_loads_LSTM_in_baselines(self):
        game = self.root / "game.tw-pddl"
        game.write_text('{"walkthrough":["look","look","look","look"]}')
        parser = run_intervention.build_parser()
        args = parser.parse_args(["--game", str(game), "--mode", "entropy_only", "--entropy-threshold", "1.5"])
        with patch.object(run_intervention, "LSTMRiskDetector", side_effect=AssertionError("Must not load LSTM")):
            prepared = run_intervention.prepare(args)
        self.assertEqual(prepared[3][0].entropy_threshold, 1.5)
        self.assertIsNone(prepared[4])
        self.assertEqual(prepared[-1]["threshold_basis"]["entropy"]["source"], "explicit_CLI")

    def test_Box_timing_reference_recorded(self):
        game = self.root / "game.tw-pddl"
        game.write_text('{}')
        args = run_intervention.build_parser().parse_args(["--game", str(game), "--mode", "no_intervention"])
        with patch.object(run_intervention, "DEFAULT_GAME", game):
            prepared = run_intervention.prepare(args)
        self.assertEqual(prepared[2].injection_step, 9)
        self.assertEqual(prepared[-1]["timing"]["source"], "user_reported_Box_Sofa_16_step_baseline")

    def test_recovery_requires_matched_prefix_and_task_outcomes(self):
        def summary(mode, outcome, success, interventions):
            return {"mode": mode, "injection_step": 2, "outcome": outcome, "success": success,
                    "intervention_count": interventions, "fault_effectively_reached": True,
                    "recovery_vs_no_intervention": None}
        base = summary("no_intervention", "step_limit", False, 0)
        proposed = summary("proposed", "success", True, 1)
        rows = [{"mode": mode, "step": 1, "action": "look"} for mode in ("no_intervention", "proposed")]
        compared = compare_summaries([deepcopy(base), deepcopy(proposed)], rows)
        self.assertTrue(compared[1]["recovery_vs_no_intervention"])
        rows[1]["action"] = "inventory"
        self.assertIsNone(compare_summaries([deepcopy(base), deepcopy(proposed)], rows)[1]["recovery_vs_no_intervention"])
        rows[1]["action"] = "look"
        proposed["outcome"] = "generation_error"
        self.assertIsNone(compare_summaries([base, proposed], rows)[1]["recovery_vs_no_intervention"])

    def test_protected_source_mutation_is_detected(self):
        source = self.root / "protected.txt"
        source.write_text("source")
        metadata = {"data_kind": "unit_test_fixture", "protected_input_sha256": {str(source): digest(source)}}
        def factory(game, seed):
            source.write_text("changed")
            env = FakeEnv(steps=2)
            return env, env.initial()
        with self.assertRaisesRegex(ValueError, "Protected"):
            execute_experiment(game="fixture", episode_id="ep", seed=42, max_steps=2, agent=FakeAgent(),
                fault=FaultConfig("goal_drift", "low", 2), interventions=[config("no_intervention")],
                similarity=FakeSimilarity(), output_root=self.root, metadata=metadata, env_factory=factory)

    def test_proposed_missing_checkpoint_stops_before_Qwen_load(self):
        game = self.root / "game.tw-pddl"
        game.write_text('{}')
        with patch.object(run_intervention.LocalLLMAgent, "load") as load:
            status = run_intervention.main(["--game", str(game), "--entropy-threshold", "1.5",
                                            "--lstm-checkpoint", str(self.root / "missing.pt")])
        self.assertEqual(status, 1)
        load.assert_not_called()

    def test_semantic_feature_reuses_existing_cosine_backend(self):
        class Embeddings:
            def encode(self, texts, **kwargs):
                return [[1., 0.] if text == GOAL else [.6, .8] for text in texts]
        similarity = SentenceSimilarity(model=Embeddings())
        agent = FakeAgent()
        decision = agent.decide(GOAL, OBS, [], ACTIONS)
        features = decision_features(decision, GOAL, agent.config, similarity, PromptCounter())
        self.assertAlmostEqual(features["S"], .6)


if __name__ == "__main__":
    unittest.main()
