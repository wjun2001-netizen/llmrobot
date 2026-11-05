"""CPU tensor tests and mocked loaders/generation only. No pretrained model or ALFWorld runs."""
from copy import deepcopy
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

import collect_local
import collect_normal
from local_llm_agent import (LocalConfigurationError, LocalGenerationError, LocalInvalidOutput,
                             LocalLLMAgent, LocalLLMConfig, entropy_from_scores)
from src.validate_local import validate


class FakeTokenizer:
    eos_token_id, pad_token_id, bos_token_id = 2, None, None

    def __init__(self, text=None):
        self.text = text or json.dumps({"plan": "test-only plan", "action": "look"})
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = deepcopy(messages)
        self.template_options = kwargs
        return "test-only prompt"

    def __call__(self, prompt, **kwargs):
        return {"input_ids": torch.tensor([[9, 8]]), "attention_mask": torch.tensor([[1, 1]])}

    def decode(self, ids, **kwargs):
        return self.text


def fake_agent(config=None, text=None):
    model = MagicMock()
    model.config = SimpleNamespace(max_position_embeddings=4096, _commit_hash="test-only-commit")
    # Deliberately unsuitable checkpoint defaults must NOT be inherited.
    model.generation_config = SimpleNamespace(eos_token_id=2, top_k=1, top_p=0.1, do_sample=True,
                                              repetition_penalty=1.5, suppress_tokens=[2])
    model.get_input_embeddings.return_value = SimpleNamespace(weight=torch.zeros(1))
    scores = (torch.zeros(1, 3), torch.tensor([[0., -float("inf"), -float("inf")]]))
    model.generate.return_value = SimpleNamespace(sequences=torch.tensor([[9, 8, 1, 2]]), scores=scores, logits=scores)
    return LocalLLMAgent(config or LocalLLMConfig(), model=model, tokenizer=FakeTokenizer(text))


class EntropyTests(unittest.TestCase):
    def test_uniform_full_vocabulary(self):
        values, mean, vocab = entropy_from_scores((torch.zeros(1, 7),))
        self.assertAlmostEqual(values[0], math.log(7), places=6)
        self.assertEqual(mean, values[0])
        self.assertEqual(vocab, 7)

    def test_skewed_matches_independent_probability_formula(self):
        values, _, _ = entropy_from_scores((torch.tensor([[math.log(.25), math.log(.75)]]),))
        self.assertAlmostEqual(values[0], -.25 * math.log(.25) - .75 * math.log(.75), places=6)

    def test_float16_scores_promoted_and_large_logits_stable(self):
        values, _, _ = entropy_from_scores((torch.tensor([[1000., -1000.]], dtype=torch.float16),))
        self.assertEqual(values[0], 0)

    def test_zero_probability_tokens_have_zero_contribution(self):
        values, _, _ = entropy_from_scores((torch.tensor([[0., -float("inf")]]),))
        self.assertEqual(values[0], 0)

    def test_all_generated_tokens_contribute_to_mean(self):
        values, mean, _ = entropy_from_scores((torch.zeros(1, 3), torch.tensor([[0., -float("inf"), -float("inf")]])))
        self.assertAlmostEqual(mean, math.log(3) / 2, places=6)
        self.assertEqual(len(values), 2)

    def test_invalid_scores_not_replaced_with_fake_entropy(self):
        for score in (torch.tensor([[float("nan"), 0.]]), torch.tensor([[float("inf"), 0.]]),
                      torch.tensor([[-float("inf"), -float("inf")]]), torch.zeros(2, 3)):
            with self.subTest(score=score), self.assertRaises(LocalGenerationError):
                entropy_from_scores((score,))
        with self.assertRaises(LocalGenerationError):
            entropy_from_scores(())


class LocalAgentTests(unittest.TestCase):
    def test_config_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"model_name": "test-only-model", "load_in_4bit": False}))
            config = LocalLLMConfig.from_file(path, model_name="override-model")
            self.assertEqual(config.model_name, "override-model")
            self.assertFalse(config.load_in_4bit)
        self.assertEqual(LocalLLMConfig().model_name, "Qwen/Qwen3-4B")
        self.assertTrue(LocalLLMConfig().load_in_4bit)
        self.assertFalse(LocalLLMConfig().save_logits)

    def test_import_or_construction_never_downloads(self):
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as model, patch("transformers.AutoTokenizer.from_pretrained") as tokenizer:
            importlib.reload(importlib.import_module("collect_local"))
            LocalLLMAgent()
            model.assert_not_called()
            tokenizer.assert_not_called()

    def test_loader_stops_before_download_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False), patch("transformers.AutoModelForCausalLM.from_pretrained") as model, patch("transformers.AutoTokenizer.from_pretrained") as tokenizer:
            with self.assertRaises(LocalConfigurationError):
                LocalLLMAgent().load()
            model.assert_not_called()
            tokenizer.assert_not_called()

    def test_4bit_loader_configuration_mock_only(self):
        with patch("torch.cuda.is_available", return_value=True), patch("transformers.BitsAndBytesConfig") as quant, patch("transformers.AutoModelForCausalLM.from_pretrained") as model, patch("transformers.AutoTokenizer.from_pretrained"):
            LocalLLMAgent().load()
        self.assertTrue(quant.call_args.kwargs["load_in_4bit"])
        self.assertEqual(quant.call_args.kwargs["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(quant.call_args.kwargs["bnb_4bit_use_double_quant"])
        self.assertFalse(model.call_args.kwargs["trust_remote_code"])
        self.assertTrue(model.call_args.kwargs["use_safetensors"])
        self.assertEqual(model.call_args.kwargs["device_map"], {"": "cuda:0"})

    def test_nonquantized_loader_mock_only(self):
        with patch("torch.cuda.is_available", return_value=True), patch("transformers.BitsAndBytesConfig") as quant, patch("transformers.AutoModelForCausalLM.from_pretrained") as model, patch("transformers.AutoTokenizer.from_pretrained"):
            LocalLLMAgent(LocalLLMConfig(load_in_4bit=False)).load()
        quant.assert_not_called()
        self.assertIsNone(model.call_args.kwargs["quantization_config"])

    def test_generation_scores_and_unfiltered_greedy_config(self):
        agent = fake_agent()
        result = agent.decide("goal", "observation", ["inventory"], ["look"])
        kwargs = agent.model.generate.call_args.kwargs
        self.assertTrue(kwargs["return_dict_in_generate"])
        self.assertTrue(kwargs["output_scores"])
        self.assertFalse(kwargs["output_logits"])
        self.assertFalse(kwargs["use_model_defaults"])
        cfg = kwargs["generation_config"]
        self.assertFalse(cfg.do_sample)
        self.assertEqual(cfg.num_beams, 1)
        self.assertEqual(cfg.repetition_penalty, 1)
        self.assertIsNone(cfg.suppress_tokens)
        self.assertIsNone(cfg.forced_eos_token_id)
        self.assertNotIn("prefix_allowed_tokens_fn", kwargs)
        inputs = json.loads(agent.tokenizer.messages[1]["content"])
        self.assertEqual(set(inputs), {"initial_goal", "current_observation", "previous_actions", "admissible_actions"})
        self.assertFalse(agent.tokenizer.template_options["enable_thinking"])
        self.assertEqual(result.action, "look")
        self.assertEqual(result.generation["generated_token_ids"], [1, 2])
        self.assertTrue(result.generation["eos_included"])
        self.assertAlmostEqual(result.generation["mean_entropy"], math.log(3) / 2, places=6)

    def test_transformers_does_not_merge_checkpoint_filters(self):
        from transformers import GenerationConfig
        from transformers.generation.utils import GenerationMixin

        defaults = GenerationConfig(do_sample=True, repetition_penalty=1.5, suppress_tokens=[2], top_k=1)
        host = SimpleNamespace(generation_config=defaults)
        fresh = GenerationConfig(do_sample=False, num_beams=1)
        merged, _ = GenerationMixin._prepare_generation_config(host, fresh, use_model_defaults=False)
        self.assertFalse(merged.do_sample)
        self.assertEqual(merged.repetition_penalty, 1.0)
        self.assertIsNone(merged.suppress_tokens)

    def test_context_limit_fails_without_truncation_or_generate(self):
        agent = fake_agent(LocalLLMConfig(max_new_tokens=3, max_context_tokens=4))
        with self.assertRaises(LocalGenerationError):
            agent.decide("goal", "observation", [], ["look"])
        agent.model.generate.assert_not_called()

    def test_logits_default_no_file_and_opt_in_exclusive_dump(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "logits.pt"
            result = fake_agent().decide("goal", "observation", [], ["look"], logits_path=path)
            self.assertFalse(path.exists())
            self.assertIsNone(result.generation["logits_file"])
            agent = fake_agent(LocalLLMConfig(save_logits=True))
            agent.decide("goal", "observation", [], ["look"], logits_path=path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["logits_kind"], "unprocessed generate.logits")
            self.assertEqual(len(saved["token_logits"]), 2)
            with self.assertRaises(FileExistsError):
                agent.decide("goal", "observation", [], ["look"], logits_path=path)

    def test_invalid_action_or_json_preserves_output_entropy(self):
        for text in ('{"plan":"test","action":"LOOK"}', '```json\n{}\n```', '{"action":"look","action":"inventory"}'):
            with self.subTest(text=text), self.assertRaises(LocalInvalidOutput) as error:
                fake_agent(text=text).decide("goal", "observation", [], ["look"])
            self.assertEqual(error.exception.generation["raw_text"], text)
            self.assertEqual(len(error.exception.generation["token_entropies"]), 2)

    def test_token_score_alignment(self):
        agent = fake_agent()
        agent.model.generate.return_value.sequences = torch.tensor([[9, 8, 1]])
        with self.assertRaises(LocalGenerationError):
            agent.decide("goal", "observation", [], ["look"])


class RouterTests(unittest.TestCase):
    def test_all_backends_dispatch_without_running_them(self):
        original = sys.argv
        for backend, module_name in collect_normal.BACKENDS.items():
            module = SimpleNamespace(main=MagicMock(return_value=0))
            with self.subTest(backend=backend), patch.object(collect_normal.importlib, "import_module", return_value=module) as load:
                self.assertEqual(collect_normal.main(["--agent", backend, "--episodes", "5"]), 0)
                load.assert_called_once_with(module_name)
                module.main.assert_called_once()
            self.assertIs(sys.argv, original)

    def test_compatible_default_and_original_helpers(self):
        import collect_pddl
        self.assertIs(collect_normal.select_games, collect_pddl.select_games)
        with patch.object(collect_normal.importlib, "import_module", return_value=SimpleNamespace(main=lambda: 0)) as load:
            collect_normal.main(["--episodes", "1"])
            load.assert_called_once_with("collect_pddl")

    def test_local_example_and_config_arguments(self):
        with patch.object(collect_local, "collect", return_value=0) as collect, patch.object(sys, "argv", ["collect_local.py", "--model", "Qwen/Qwen3-4B", "--episodes", "5", "--save-logits"]):
            self.assertEqual(collect_local.main(), 0)
        args, config = collect.call_args.args
        self.assertEqual(args.episodes, 5)
        self.assertEqual(config.model_name, "Qwen/Qwen3-4B")
        self.assertTrue(config.save_logits)

    def test_import_check_does_not_load_cuda_or_model(self):
        with patch("torch.cuda.is_available") as cuda, patch("transformers.AutoModelForCausalLM.from_pretrained") as model, patch("transformers.AutoTokenizer.from_pretrained") as tokenizer:
            report = collect_local.check_imports()
        self.assertFalse(report["model_loaded"])
        cuda.assert_not_called()
        model.assert_not_called()
        tokenizer.assert_not_called()


class CollectionUnitTests(unittest.TestCase):
    """Mock environment only: no ALFWorld episode, external API or pretrained inference."""
    def test_local_rows_history_entropy_and_success_roundtrip(self):
        env = MagicMock()
        reset = {"feedback": "Your task is to: fixture goal.", "won": False, "admissible_commands": ["look"]}
        env.step.side_effect = [({"feedback": "test observation", "won": False, "admissible_commands": ["look"]}, 0, False),
                                ({"feedback": "test terminal", "won": True}, 1, True)]
        with tempfile.TemporaryDirectory() as folder:
            raw_path, decisions_path = Path(folder) / "raw.jsonl", Path(folder) / "decisions.jsonl"
            with raw_path.open("x") as raw, decisions_path.open("x") as decisions:
                summary = collect_local.run_episode(None, "test-only-episode", 42, 3, fake_agent(), raw, decisions,
                                                    env_factory=lambda game, seed: (env, reset))
            report = validate(raw_path, [summary], 1)
            self.assertTrue(report["data_valid"])
            self.assertTrue(report["experiment_complete"])
            self.assertEqual(report["success_count"], 1)
            rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
            self.assertEqual(rows[1]["previous_actions"], ["look"])
            self.assertTrue(all(row["mean_entropy"] is not None for row in rows))
            rows[0]["mean_entropy"] = 99
            raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertFalse(validate(raw_path, [summary], 1)["data_valid"])
        env.close.assert_called_once()

    def test_invalid_action_does_not_step_mock_environment(self):
        env = MagicMock()
        reset = {"feedback": "Your task is to: fixture.", "won": False, "admissible_commands": ["look"]}
        with tempfile.TemporaryDirectory() as folder:
            raw_path, decisions_path = Path(folder) / "raw.jsonl", Path(folder) / "decisions.jsonl"
            with raw_path.open("x") as raw, decisions_path.open("x") as decisions:
                summary = collect_local.run_episode(None, "test-only", 42, 3,
                    fake_agent(text='{"plan":"test-only","action":"invented"}'), raw, decisions,
                    env_factory=lambda game, seed: (env, reset))
            self.assertEqual(summary["outcome"], "invalid_output")
            self.assertEqual(raw_path.read_text(), "")
            self.assertEqual(json.loads(decisions_path.read_text())["action"], "invented")
            self.assertFalse(validate(raw_path, [summary], 1)["experiment_complete"])
        env.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
