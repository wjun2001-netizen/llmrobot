"""Artificial Gemini fixtures are test-only; no fixture is a real LLM trajectory."""
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import collect_gemini
from src.gemini_agent import (GeminiAgent, GeminiAPIError, GeminiClient, GeminiConfigurationError,
                              GeminiInvalidOutput, GeminiSettings, NoRedirects, flash_models,
                              probability_data, probe_model)
from src.validate_gemini import validate

ROOT = Path(__file__).resolve().parents[1]


def fixture_response(action="look", plan="Test-only plan", probabilities=True):
    token = {"token": "test-only-token", "tokenId": 1, "logProbability": -0.2}
    return {"modelVersion": "test-only-model-not-real", "responseId": "test-only-response",
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps({"plan": plan, "action": action})}]},
                            "logprobsResult": {"chosenCandidates": [deepcopy(token)],
                                               "topCandidates": [{"candidates": [deepcopy(token)]}]} if probabilities else None}]}


class FixtureClient:
    def __init__(self, response=None):
        self.settings = SimpleNamespace(model="models/test-only-flash", response_logprobs=True)
        self.response = response if response is not None else fixture_response()
        self.inputs = []

    def generate(self, inputs):
        self.inputs.append(deepcopy(inputs))
        return deepcopy(self.response)


class GeminiTests(unittest.TestCase):
    def test_missing_key_blocks_no_network(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(GeminiClient, "_request") as request:
            with self.assertRaises(GeminiConfigurationError):
                GeminiSettings.from_env()
            request.assert_not_called()

    def test_no_default_model_and_secret_not_serialized(self):
        settings = GeminiSettings(api_key="test-only-key", request_interval=0)
        self.assertIsNone(settings.model)
        self.assertNotIn("test-only-key", repr(settings))
        self.assertNotIn("test-only-key", json.dumps(settings.public_config()))
        with self.assertRaises(GeminiConfigurationError):
            GeminiClient(settings).generate({"admissible_actions": ["look"]})

    def test_actual_request_fields_and_enum(self):
        client = GeminiClient(GeminiSettings(api_key="test-only-key", model="test-only-flash", request_interval=0))
        inputs = {"initial_goal": "test", "current_observation": "test", "previous_actions": [], "admissible_actions": ["look", "inventory"]}
        with patch.object(client, "_request", return_value=fixture_response()) as request:
            client.generate(inputs)
        path, payload = request.call_args.args
        self.assertEqual(path, "/models/test-only-flash:generateContent")
        self.assertTrue(payload["generationConfig"]["responseLogprobs"])
        self.assertEqual(payload["generationConfig"]["logprobs"], 5)
        self.assertEqual(payload["generationConfig"]["responseSchema"]["properties"]["action"]["enum"], inputs["admissible_actions"])
        self.assertNotIn("test-only-key", json.dumps(payload))

    def test_explicit_without_logprobs_not_misreported(self):
        client = GeminiClient(GeminiSettings(api_key="test", model="test-only-flash", response_logprobs=False, request_interval=0))
        with patch.object(client, "_request", return_value=fixture_response(probabilities=False)) as request:
            client.generate({"admissible_actions": ["look"]})
        self.assertNotIn("responseLogprobs", request.call_args.args[1]["generationConfig"])
        self.assertFalse(probability_data(fixture_response(probabilities=False))["logprobs_available"])

    def test_inputs_only_four_fields_and_plan_preserved(self):
        plan = "  Test plan\nkeep whitespace  "
        client = FixtureClient(fixture_response(plan=plan))
        result = GeminiAgent(client).decide("goal", "observation", ["inventory"], ["look"])
        self.assertEqual(result.plan, plan)
        self.assertEqual(set(client.inputs[0]), {"initial_goal", "current_observation", "previous_actions", "admissible_actions"})

    def test_no_fuzzy_matching_or_fallback(self):
        for action in ("look ", "LOOK", "invented", None):
            response = fixture_response(action=action)
            with self.subTest(action=action), self.assertRaises(GeminiInvalidOutput) as error:
                GeminiAgent.validate_response(response, ["look"])
            self.assertEqual(error.exception.response, response)

    def test_malformed_json_never_repaired(self):
        for value in ('```json\n{}\n```', '{"action":"inventory","action":"look"}', 'not JSON'):
            response = fixture_response()
            response["candidates"][0]["content"]["parts"][0]["text"] = value
            with self.assertRaises(GeminiInvalidOutput):
                GeminiAgent.validate_response(response, ["look"])

    def test_null_plan_preserved(self):
        self.assertIsNone(GeminiAgent.validate_response(fixture_response(plan=None), ["look"]).plan)

    def test_refusal_incomplete_and_thought_only_not_actions(self):
        response = fixture_response()
        response["candidates"][0]["finishReason"] = "MAX_TOKENS"
        with self.assertRaises(GeminiInvalidOutput):
            GeminiAgent.validate_response(response, ["look"])
        response["candidates"][0]["finishReason"] = "STOP"
        response["candidates"][0]["content"]["parts"][0]["thought"] = True
        with self.assertRaises(GeminiInvalidOutput):
            GeminiAgent.validate_response(response, ["look"])
        with self.assertRaises(GeminiInvalidOutput):
            GeminiAgent.validate_response({"promptFeedback": {"blockReason": "SAFETY"}}, ["look"])

    def test_logprob_tokens_and_alternatives_untouched(self):
        response = fixture_response()
        result = probability_data(response)
        self.assertEqual(result["token_logprobs"], response["candidates"][0]["logprobsResult"]["chosenCandidates"])
        self.assertTrue(result["logprobs_available"])
        self.assertEqual(result["logprobs_result"], response["candidates"][0]["logprobsResult"])
        response["candidates"][0]["logprobsResult"] = {"logProbabilitySum": -1.0}
        self.assertIsNone(probability_data(response)["token_logprobs"])
        self.assertFalse(probability_data(response)["logprobs_available"])

    def test_model_listing_does_not_verify_probabilities_or_free_tier(self):
        catalog = {"models": [{"name": "models/test-only-flash", "supportedGenerationMethods": ["generateContent"]},
                               {"name": "models/not-flash-test", "supportedGenerationMethods": ["embedContent"]}]}
        self.assertEqual(flash_models(catalog), ["models/test-only-flash"])
        result = probe_model(FixtureClient(fixture_response(probabilities=False)), catalog)
        self.assertFalse(result["logprobs_verified"])
        self.assertIsNone(result["free_tier_verified"])

    def test_error_body_key_redacted(self):
        client = GeminiClient(GeminiSettings(api_key="test-only-secret", request_interval=0))
        error = HTTPError("https://example.invalid", 400, "invalid", {}, BytesIO(json.dumps({
            "error": {"status": "INVALID_ARGUMENT", "message": "test-only-secret responseLogprobs unsupported"}}).encode()))
        with patch.object(client.opener, "open", side_effect=error), self.assertRaises(GeminiAPIError) as caught:
            client.list_models()
        self.assertNotIn("test-only-secret", str(caught.exception))
        self.assertIn("responseLogprobs unsupported", str(caught.exception))

    def test_redirect_does_not_forward_key(self):
        self.assertIsNone(NoRedirects().redirect_request(None, None, 302, "", {}, "https://example.com"))


class GeminiRealEnvironmentTests(unittest.TestCase):
    """Actual ALFWorld + artificial API fixture; TEMP output, never a real Gemini run."""
    @classmethod
    def setUpClass(cls):
        os.environ["ALFWORLD_DATA"] = str(ROOT / "data/alfworld")
        cls.game = collect_gemini.select_games(ROOT / "data/alfworld", "train", "pick_and_place_simple", 1, 42)[0]
        cls.episode_id = str(cls.game.parent.relative_to(ROOT / "data/alfworld/json_2.1.1"))

    def test_real_environment_expert_not_requested(self):
        env, state = collect_gemini.open_environment(self.game, 42)
        try:
            self.assertIsNone(state.get("extra.expert_plan"))
            self.assertIsNone(state.get("policy_commands"))
            self.assertIsNone(state.get("facts"))
        finally:
            env.close()

    def run_in_temp(self, client, check):
        with tempfile.TemporaryDirectory() as directory:
            raw_path, decisions_path = Path(directory) / "test-only-raw.jsonl", Path(directory) / "test-only-decisions.jsonl"
            with raw_path.open("x") as raw, decisions_path.open("x") as decisions:
                summary = collect_gemini.run_episode(self.game, self.episode_id, 42, 2, GeminiAgent(client), raw, decisions)
            check(summary, raw_path, decisions_path)

    def test_failure_history_logprobs_and_validator(self):
        def check(summary, raw_path, decisions_path):
            rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
            self.assertEqual(summary["outcome"], "step_limit")
            self.assertEqual(rows[1]["previous_actions"], ["look"])
            self.assertTrue(all(row["logprobs_available"] for row in rows))
            self.assertEqual(len(decisions_path.read_text().splitlines()), 2)
            report = validate(raw_path, [summary], 1)
            self.assertTrue(report["data_valid"])
            self.assertTrue(report["experiment_complete"])
            self.assertEqual(report["task_failure_count"], 1)
            self.assertEqual(report["tokens_with_logprobs"], 2)
            self.assertFalse(report["full_shannon_entropy_data_sufficient"])
            rows[0]["logprobs_available"] = False
            # Only mutate a TEMP fixture copy to test detection, never a real run.
            raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertFalse(validate(raw_path, [summary], 1)["data_valid"])
        self.run_in_temp(FixtureClient(), check)

    def test_missing_probabilities_stay_null(self):
        def check(summary, raw_path, decisions_path):
            rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
            self.assertTrue(all(row["token_logprobs"] is None and row["logprobs_available"] is False for row in rows))
            self.assertEqual(validate(raw_path, [summary], 1)["logprobs_unavailable_steps"], 2)
        self.run_in_temp(FixtureClient(fixture_response(probabilities=False)), check)

    def test_invalid_action_not_executed_and_preserved(self):
        def check(summary, raw_path, decisions_path):
            self.assertEqual(summary["outcome"], "invalid_output")
            self.assertEqual(summary["steps"], 0)
            self.assertEqual(raw_path.read_text(), "")
            self.assertEqual(json.loads(decisions_path.read_text())["action"], "invented")
        self.run_in_temp(FixtureClient(fixture_response(action="invented")), check)


if __name__ == "__main__":
    unittest.main()
