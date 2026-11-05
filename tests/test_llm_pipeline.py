"""Offline fixtures are ONLY tests, never evidence of real LLM episode execution."""
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import collect_llm
from src.llm_agent import (APIError, ConfigurationError, InvalidOutput, LLMAgent, NoRedirects,
                           OpenAIResponsesClient, OpenAISettings, available_model_ids,
                           has_logprobs, output_blocks, probe_model)
from src.llm_io import new_run, save_json
from src.validate_llm import validate

ROOT = Path(__file__).resolve().parents[1]


def fixture_response(action="look", plan="test-only plan", logprobs=True):
    """Artificial protocol fixture, not a model output or research trajectory."""
    return {"id": "test-only-response", "model": "test-fixture-not-a-real-model", "status": "completed",
            "usage": {"output_tokens": 1}, "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": json.dumps({"plan": plan, "action": action}),
                 "logprobs": [{"token": "test-only", "logprob": -0.2, "bytes": [120],
                               "top_logprobs": [{"token": "test-only", "logprob": -0.2, "bytes": [120]}]}]
                    if logprobs else None}]}]}


class FixtureClient:
    def __init__(self, response=None):
        self.settings = SimpleNamespace(model="test-fixture-not-a-real-model", top_logprobs=5)
        self.response = response if response is not None else fixture_response()
        self.inputs = []

    def respond(self, inputs):
        self.inputs.append(deepcopy(inputs))
        return deepcopy(self.response)


class AgentTests(unittest.TestCase):
    def test_missing_key_blocks_before_network(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(OpenAIResponsesClient, "_request") as network:
            with self.assertRaisesRegex(ConfigurationError, "OPENAI_API_KEY"):
                OpenAISettings.from_env()
            network.assert_not_called()

    def test_no_default_model(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-key"}, clear=True):
            settings = OpenAISettings.from_env()
        self.assertIsNone(settings.model)
        self.assertNotIn("test-only-key", repr(settings))
        self.assertNotIn("test-only-key", json.dumps(settings.public_config()))
        client = OpenAIResponsesClient(settings)
        with self.assertRaises(ConfigurationError):
            client.respond({"admissible_actions": ["look"]})

    def test_responses_payload_and_generated_enum(self):
        client = OpenAIResponsesClient(OpenAISettings(api_key="test-only-key", model="test-fixture"))
        inputs = {"initial_goal": "fixture", "current_observation": "fixture", "previous_actions": [], "admissible_actions": ["look", "inventory"]}
        with patch.object(client, "_request", return_value=fixture_response()) as request:
            client.respond(inputs)
        path, payload = request.call_args.args
        self.assertEqual(path, "/responses")
        self.assertEqual(payload["include"], ["message.output_text.logprobs"])
        self.assertEqual(payload["top_logprobs"], 5)
        self.assertFalse(payload["store"])
        self.assertEqual(payload["text"]["format"]["schema"]["properties"]["action"]["enum"], inputs["admissible_actions"])
        self.assertNotIn("test-only-key", json.dumps(payload))
        self.assertNotIn("seed", payload)

    def test_plan_and_action_untouched_inputs_limited(self):
        plan = "  Test-only plan: preserve\nall spacing.  "
        client = FixtureClient(fixture_response(plan=plan))
        previous = ["inventory"]
        decision = LLMAgent(client).decide("goal", "observation", previous, ["look"])
        self.assertEqual(decision.plan, plan)
        self.assertEqual(decision.action, "look")
        self.assertEqual(set(client.inputs[0]), {"initial_goal", "current_observation", "previous_actions", "admissible_actions"})
        self.assertEqual(previous, ["inventory"])

    def test_no_matching_or_expert_substitution(self):
        for action in ["look ", "LOOK", "invented command", None]:
            response = fixture_response(action=action)
            with self.subTest(action=action), self.assertRaises(InvalidOutput) as error:
                LLMAgent(FixtureClient(response)).decide("goal", "observation", [], ["look"])
            self.assertEqual(error.exception.response, response)

    def test_malformed_and_duplicate_json_never_repaired(self):
        for text in ['```json\n{"action":"look"}\n```', '{"action":"inventory","action":"look"}', 'not JSON']:
            response = fixture_response()
            response["output"][0]["content"][0]["text"] = text
            with self.assertRaises(InvalidOutput):
                LLMAgent(FixtureClient(response)).decide("goal", "obs", [], ["look"])

    def test_incomplete_and_refusal_never_execute(self):
        response = fixture_response()
        response["status"] = "incomplete"
        with self.assertRaises(InvalidOutput):
            LLMAgent.validate_response(response, ["look"])
        response["status"] = "completed"
        response["output"][0]["content"] = [{"type": "refusal", "refusal": "test-only refusal"}]
        with self.assertRaises(InvalidOutput):
            LLMAgent.validate_response(response, ["look"])

    def test_unavailable_plan_stays_null(self):
        result = LLMAgent.validate_response(fixture_response(plan=None), ["look"])
        self.assertIsNone(result.plan)

    def test_actual_logprobs_preserved_not_inferred(self):
        response = fixture_response()
        blocks = output_blocks(response)
        self.assertTrue(has_logprobs(blocks))
        self.assertEqual(blocks[0]["logprobs"], response["output"][0]["content"][0]["logprobs"])
        self.assertFalse(has_logprobs(output_blocks(fixture_response(logprobs=False))))

    def test_model_listing_does_not_imply_logprobs_support(self):
        catalog = {"data": [{"id": "test-fixture-not-a-real-model"}]}
        self.assertEqual(available_model_ids(catalog), ["test-fixture-not-a-real-model"])
        self.assertFalse(probe_model(FixtureClient(fixture_response(logprobs=False)), catalog)["output_logprobs_verified"])
        with self.assertRaises(ConfigurationError):
            probe_model(FixtureClient(), {"data": [{"id": "different-fixture"}]})

    def test_credentials_not_forwarded_by_redirect(self):
        self.assertIsNone(NoRedirects().redirect_request(None, None, 302, "", {}, "https://example.com"))

    def test_exclusive_file_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = new_run(directory), new_run(directory)
            self.assertNotEqual(first, second)
            path = first / "test.json"
            save_json(path, {"test": 1})
            with self.assertRaises(FileExistsError):
                save_json(path, {"test": 2})
            self.assertEqual(json.loads(path.read_text()), {"test": 1})

    def test_zero_step_success_is_not_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty_test_only.jsonl"
            path.write_text("")
            summary = {"episode_id": "test-only", "steps": 0, "outcome": "success", "done": True, "success": True, "truncated": False}
            self.assertFalse(validate(path, [summary], 1)["data_valid"])

    def test_malformed_output_container_is_invalid(self):
        for output in (None, 5, {"not": "a list"}):
            with self.subTest(output=output), self.assertRaises(InvalidOutput):
                LLMAgent.validate_response({"status": "completed", "output": output}, ["look"])


class RealEnvironmentFixtureTests(unittest.TestCase):
    """Real ALFWorld + artificial protocol fixture in TEMP files, NOT a real LLM run."""
    @classmethod
    def setUpClass(cls):
        os.environ["ALFWORLD_DATA"] = str(ROOT / "data/alfworld")
        games = collect_llm.select_games(ROOT / "data/alfworld", "train", "pick_and_place_simple", 1, 42)
        cls.game = games[0]
        cls.episode_id = str(cls.game.parent.relative_to(ROOT / "data/alfworld/json_2.1.1"))

    def test_environment_does_not_expose_expert_plan(self):
        env, state = collect_llm.open_environment(self.game, 42)
        try:
            self.assertIsNone(state.get("extra.expert_plan"))
            self.assertIsNone(state.get("policy_commands"))
            self.assertIsNone(state.get("facts"))
            self.assertIn("look", state["admissible_commands"])
        finally:
            env.close()

    def test_failed_episode_preserved_and_history_correct(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path, decision_path = Path(directory) / "test_only_raw.jsonl", Path(directory) / "test_only_decisions.jsonl"
            client = FixtureClient()
            with raw_path.open("x") as raw, decision_path.open("x") as decisions:
                summary = collect_llm.run_episode(self.game, self.episode_id, 42, 2, LLMAgent(client), raw, decisions)
            rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
            self.assertEqual(summary["outcome"], "step_limit")
            self.assertEqual(summary["steps"], 2)
            self.assertFalse(summary["success"])
            self.assertEqual(rows[0]["previous_actions"], [])
            self.assertEqual(rows[1]["previous_actions"], ["look"])
            self.assertEqual(client.inputs[1]["previous_actions"], ["look"])
            self.assertEqual(len(decision_path.read_text().splitlines()), 2)
            report = validate(raw_path, [summary], 1)
            self.assertTrue(report["data_valid"])
            self.assertTrue(report["experiment_complete"])
            self.assertEqual(report["task_failure_count"], 1)

    def test_invalid_output_saved_without_environment_step(self):
        env, state = collect_llm.open_environment(self.game, 42)
        with tempfile.TemporaryDirectory() as directory:
            raw_path, decision_path = Path(directory) / "test_only_raw.jsonl", Path(directory) / "test_only_decisions.jsonl"
            with patch.object(collect_llm, "open_environment", return_value=(env, state)), patch.object(env, "step", wraps=env.step) as step:
                with raw_path.open("x") as raw, decision_path.open("x") as decisions:
                    summary = collect_llm.run_episode(self.game, self.episode_id, 42, 2,
                        LLMAgent(FixtureClient(fixture_response(action="invented command"))), raw, decisions)
                step.assert_not_called()
            self.assertEqual(summary["outcome"], "invalid_output")
            self.assertEqual(raw_path.read_text(), "")
            decision = json.loads(decision_path.read_text())
            self.assertEqual(decision["action"], "invented command")
            self.assertIsNotNone(decision["raw_response"])
            self.assertFalse(validate(raw_path, [summary], 1)["experiment_complete"])

    def test_api_error_does_not_fabricate_step(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path, decision_path = Path(directory) / "test_only_raw.jsonl", Path(directory) / "test_only_decisions.jsonl"
            client = FixtureClient()
            with patch.object(client, "respond", side_effect=APIError("test-only simulated API failure")):
                with raw_path.open("x") as raw, decision_path.open("x") as decisions:
                    summary = collect_llm.run_episode(self.game, self.episode_id, 42, 2, LLMAgent(client), raw, decisions)
            self.assertEqual(summary["outcome"], "api_error")
            self.assertEqual(summary["steps"], 0)
            self.assertEqual(raw_path.read_text(), "")
            self.assertIsNone(json.loads(decision_path.read_text())["raw_response"])


if __name__ == "__main__":
    unittest.main()
