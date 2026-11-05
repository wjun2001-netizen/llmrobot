"""Feature unit tests: synthetic sources, a byte-tokenizer fixture and explicit embedding vectors."""
import csv
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from local_llm_agent import LocalLLMConfig, SYSTEM_PROMPT
from src.timeseries_features import (FeatureError, PromptCounter, SentenceSimilarity, cosine_similarity,
                                     discover_trajectories, extract)

ROOT = Path(__file__).resolve().parents[1]
GOAL = "put a watch in shelf."
PLANS = ["approach the shelf", "take the watch", "put the watch on the shelf", "put the watch on a desk"]


class ByteTokenizerFixture:
    """One token per UTF-8 byte, only for unit tests; NOT the Qwen tokenizer."""
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), kwargs))
        return "<system>" + messages[0]["content"] + "</system><user>" + messages[1]["content"] + "</user><assistant>"

    def __call__(self, prompt, **kwargs):
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
        return {"input_ids": [[0] * len(prompt.encode("utf-8"))]}


class EmbeddingFixture:
    """Artificial vectors test cosine math and plumbing, not semantic model quality."""
    max_seq_length = 256
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        values = {GOAL: [1., 0.], PLANS[0]: [1., 0.], PLANS[1]: [.6, .8], PLANS[2]: [.8, .6], PLANS[3]: [-.8, .6],
                  "put a watch in desk.": [-1., 0.]}
        return np.asarray([values[text] for text in texts])


def write_fixture(root, *, observed=False):
    root = Path(root)
    experiment = root / "fixture_run"
    experiment.mkdir(parents=True)
    config = LocalLLMConfig(model_name="fixture/tokenizer").public_config()
    source_hash = hashlib.sha256((ROOT / "local_llm_agent.py").read_bytes()).hexdigest()
    (experiment / "config.json").write_text(json.dumps({"condition": "paired", "local_llm": config,
                                                       "system_prompt": SYSTEM_PROMPT, "source_sha256": {"local_llm_agent.py": source_hash}}))
    (experiment / "model_info.json").write_text(json.dumps({"resolved_revision": "fixture-commit"}))
    paths = {}
    for condition, count in (("no_fault", 3), ("fault", 2)):
        folder = experiment / "episodes/0000" / condition
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"condition": condition, "local_llm": config}))
        summaries = [{"episode_id": "fixture-episode", "steps": count, "outcome": "success" if condition == "no_fault" else "task_failure"}]
        (folder / "episodes.json").write_text(json.dumps(summaries))
        rows = []
        for index in range(count):
            active = condition == "fault" and index >= 1
            plan = PLANS[3] if active else PLANS[index]
            row = {"episode_id": "fixture-episode", "step": index + 1, "goal": GOAL, "original_goal": GOAL,
                   "observation": "You see a watch on shelf 2.", "original_observation": "You see a watch on shelf 2.",
                   "agent_goal": "put a watch in desk." if active else GOAL, "agent_observation": "You see a watch on shelf 2.",
                   "previous_actions": ["look"] * index, "admissible_actions": ["look", "go to shelf 2"],
                   "plan": plan, "action_id": 0, "action": "look", "action_selection_format": "action_id_v1",
                   "model_name": "fixture/tokenizer", "resolved_revision": "fixture-commit", "mean_entropy": 2.1 if active else 1.2 + .1 * index,
                   "entropy_unit": "nats", "fault_type": "goal_drift" if condition == "fault" else "no_fault",
                   "fault_severity": "low" if condition == "fault" else None, "fault_active": active,
                   "injection_step": 2 if condition == "fault" else None, "success": condition == "no_fault" and index == count - 1}
            if observed:
                row.update(prompt_token_count=101 + index, max_context_tokens=4096)
            rows.append(row)
        path = folder / "trajectories.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths[condition] = path
    return paths


def fixture_dependencies():
    tokenizer = ByteTokenizerFixture()
    counter = PromptCounter(tokenizer_loader=lambda *args, **kwargs: tokenizer)
    similarity = SentenceSimilarity("fixture/embedding-vectors", model=EmbeddingFixture())
    return tokenizer, counter, similarity


def read_csv(path):
    with Path(path).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def generate_example(output):
    with tempfile.TemporaryDirectory(prefix="timeseries-unit-fixture-") as temp:
        write_fixture(temp)
        _, counter, similarity = fixture_dependencies()
        return extract(temp, output, counter=counter, similarity=similarity, data_kind="unit_test_fixture")


class MathTests(unittest.TestCase):
    def test_cosine_is_not_exact_match_or_hamming_and_retains_negative_values(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [.6, .8]), .6)
        self.assertEqual(cosine_similarity([1, 0], [-1, 0]), -1)
        self.assertEqual(cosine_similarity([1, 0], [0, 1]), 0)

    def test_missing_nonfinite_or_zero_embeddings_fail(self):
        for left, right in (([0, 0], [1, 0]), ([float("nan"), 1], [1, 0]), ([1], [1, 2])):
            with self.assertRaises(FeatureError):
                cosine_similarity(left, right)

    def test_sentence_encoder_is_called_with_real_text_and_batched(self):
        model = EmbeddingFixture()
        sim = SentenceSimilarity("fixture-vectors", model=model)
        values = sim.score_pairs([(GOAL, PLANS[0]), (GOAL, PLANS[1])])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][0], [GOAL, PLANS[0], PLANS[1]])
        self.assertEqual(values, [1., .6])
        self.assertFalse(model.calls[0][1]["normalize_embeddings"])


class PromptTests(unittest.TestCase):
    def test_runtime_record_count_matches_actual_encoded_input_length(self):
        from test_local_llm import fake_agent
        agent = fake_agent()
        result = agent.decide(GOAL, "fixture observation", [], ["look"])
        actual = agent.model.generate.call_args.kwargs["input_ids"].shape[-1]
        self.assertEqual(result.generation["prompt_token_count"], actual)
        self.assertEqual(result.generation["max_context_tokens"], agent.config.max_context_tokens)

    def test_fault_prompt_uses_agent_inputs_original_action_order_and_template(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = write_fixture(temp)
            tokenizer, counter, similarity = fixture_dependencies()
            output = Path(temp) / "out.csv"
            report = extract(temp, output, counter=counter, similarity=similarity)
            self.assertEqual(report["rows"], 5)
            inputs = [json.loads(messages[1]["content"]) for messages, kwargs in tokenizer.calls]
            self.assertEqual(inputs[-1]["initial_goal"], "put a watch in desk.")
            self.assertEqual(inputs[-1]["previous_actions"], ["look"])
            self.assertEqual(inputs[-1]["admissible_actions"], [{"action_id": 0, "action": "look"}, {"action_id": 1, "action": "go to shelf 2"}])
            self.assertFalse(tokenizer.calls[-1][1]["enable_thinking"])
            rows = read_csv(output)
            self.assertAlmostEqual(float(rows[-1]["S"]), -.8)  # Original goal, NOT the perturbed goal.
            self.assertEqual(float(rows[-1]["E"]), 2.1)
            self.assertEqual(float(rows[-1]["C"]), int(rows[-1]["prompt_token_count"]) / 4096)

    def test_context_load_changes_with_observation_without_changing_history_count(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = write_fixture(temp)
            tokenizer, counter, _ = fixture_dependencies()
            from src.timeseries_features import load_metadata
            row = json.loads(paths["fault"].read_text().splitlines()[1])
            meta = load_metadata(paths["fault"].parent)
            first = counter.count(row, meta)["prompt_token_count"]
            longer = deepcopy(row)
            longer["agent_observation"] += " [UNRELATED_CONTEXT] " + "noise " * 40
            self.assertGreater(counter.count(longer, meta)["prompt_token_count"], first)
            self.assertEqual(len(longer["previous_actions"]), len(row["previous_actions"]))

    def test_observed_counts_need_no_tokenizer_loading(self):
        counter = PromptCounter(tokenizer_loader=lambda *a, **k: self.fail("must not load tokenizer"))
        result = counter.count({"prompt_token_count": 100, "max_context_tokens": 2048}, {"local_llm": {"max_context_tokens": 2048}})
        self.assertEqual(result["prompt_token_count"], 100)
        self.assertEqual(result["max_context_tokens"], 2048)
        self.assertEqual(result["prompt_count_source"], "recorded_at_generation")

    def test_unknown_history_and_revision_are_not_guessed(self):
        tokenizer, counter, _ = fixture_dependencies()
        row = {"action_selection_format": "action_id_v1", "model_name": "fixture/tokenizer", "goal": GOAL, "observation": "obs",
               "previous_actions": [], "admissible_actions": ["look"]}
        with self.assertRaisesRegex(FeatureError, "source hash"):
            counter.count(row, {"local_llm": {}})
        known = next(iter(counter.formats))
        meta = {"local_llm": {}, "source_sha256": {"local_llm_agent.py": known}}
        with self.assertRaisesRegex(FeatureError, "revision"):
            counter.count(row, meta)

    def test_count_nan_and_context_budget_conflict_fail(self):
        counter = PromptCounter()
        for row in ({"prompt_token_count": float("nan")}, {"prompt_token_count": None}, {"prompt_token_count": True},
                    {"prompt_token_count": 100, "max_context_tokens": 1024}):
            with self.assertRaises(FeatureError):
                counter.count(row, {"local_llm": {"max_context_tokens": 4096}})


class ExtractionTests(unittest.TestCase):
    def extract_fixture(self, root, output, **kwargs):
        _, counter, similarity = fixture_dependencies()
        return extract(root, output, counter=counter, similarity=similarity, **kwargs)

    def test_both_conditions_step_order_and_original_files_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = write_fixture(temp)
            before = {name: path.read_bytes() for name, path in paths.items()}
            output = Path(temp) / "features/timeseries.csv"
            report = self.extract_fixture(temp, output)
            rows = read_csv(output)
            self.assertEqual([(row["condition"], int(row["step"])) for row in rows], [("no_fault", 1), ("no_fault", 2), ("no_fault", 3), ("fault", 1), ("fault", 2)])
            self.assertEqual(report["rows"], 5)
            self.assertTrue(report["source_trajectories_unchanged"])
            self.assertNotIn(b"\r", output.read_bytes())
            self.assertTrue(all(paths[name].read_bytes() == value for name, value in before.items()))
            self.assertEqual(rows[0]["fault_severity"], "")
            self.assertEqual(rows[0]["injection_step"], "")
            self.assertEqual(rows[-1]["outcome"], "task_failure")

    def test_multiple_run_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            write_fixture(Path(temp) / "batchA")
            write_fixture(Path(temp) / "batchB")
            self.assertEqual(self.extract_fixture(temp, Path(temp) / "out.csv")["rows"], 10)

    def test_missing_plan_and_nan_entropy_fail_without_csv(self):
        for change in ({"plan": None}, {"mean_entropy": float("nan")}, {"mean_entropy": float("inf")}, {"mean_entropy": None}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                paths = write_fixture(temp)
                rows = [json.loads(line) for line in paths["fault"].read_text().splitlines()]
                rows[-1].update(change)
                paths["fault"].write_text("".join(json.dumps(row) + "\n" for row in rows))
                output = Path(temp) / "out.csv"
                with self.assertRaises(FeatureError):
                    self.extract_fixture(temp, output)
                self.assertFalse(output.exists())

    def test_skip_invalid_is_explicit_reported_and_not_renumbered(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = write_fixture(temp)
            rows = [json.loads(line) for line in paths["no_fault"].read_text().splitlines()]
            rows[1]["mean_entropy"] = None
            paths["no_fault"].write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = Path(temp) / "out.csv"
            report = self.extract_fixture(temp, output, skip_invalid=True)
            self.assertEqual(report["rows"], 4)
            self.assertTrue(report["validation_errors"])
            self.assertEqual([int(row["step"]) for row in read_csv(output) if row["condition"] == "no_fault"], [1, 3])

    def test_reversed_steps_rejected_not_silently_sorted(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = write_fixture(temp)
            lines = paths["fault"].read_text().splitlines()
            paths["fault"].write_text("\n".join(reversed(lines)) + "\n")
            with self.assertRaisesRegex(FeatureError, "order"):
                self.extract_fixture(temp, Path(temp) / "out.csv")

    def test_missing_fault_input_provenance_not_replaced_by_true_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = write_fixture(temp)
            rows = [json.loads(line) for line in paths["fault"].read_text().splitlines()]
            del rows[-1]["agent_observation"]
            paths["fault"].write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(FeatureError, "provenance"):
                self.extract_fixture(temp, Path(temp) / "out.csv")

    def test_source_can_never_be_overwrite_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = write_fixture(temp)
            before = paths["fault"].read_bytes()
            with self.assertRaises(FeatureError):
                self.extract_fixture(temp, paths["fault"], overwrite=True)
            self.assertEqual(paths["fault"].read_bytes(), before)

    def test_existing_derived_output_not_replaced_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            write_fixture(temp)
            output = Path(temp) / "out.csv"
            self.extract_fixture(temp, output)
            before = output.read_bytes()
            with self.assertRaises(FeatureError):
                self.extract_fixture(temp, output)
            self.assertEqual(output.read_bytes(), before)

    def test_no_real_runs_does_not_load_any_embedding_or_tokenizer(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(SentenceSimilarity, "score_pairs") as similarity, self.assertRaises(FeatureError):
                extract(temp, Path(temp) / "out.csv")
            similarity.assert_not_called()


if __name__ == "__main__":
    unittest.main()
