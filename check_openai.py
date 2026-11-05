"""List actual accessible models, or probe one explicitly selected model's Responses/logprobs support."""
import argparse
import json
import os
from pathlib import Path

from src.llm_agent import (APIError, ConfigurationError, InvalidOutput, OpenAIResponsesClient,
                           OpenAISettings, available_model_ids, probe_model)
from src.llm_io import new_run, save_json

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["models", "probe"])
    parser.add_argument("--model", help="Explicit ID from the actual models response; no default")
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs/openai_checks")
    args = parser.parse_args()
    run = new_run(args.output_dir)
    print("CHECK_DIR=" + str(run))
    try:
        settings = OpenAISettings.from_env(model=args.model, top_logprobs=args.top_logprobs,
                                          max_output_tokens=args.max_output_tokens)
        client = OpenAIResponsesClient(settings)
        catalog = client.list_models()
        save_json(run / "models.json", catalog)
        ids = available_model_ids(catalog)
        if args.operation == "models":
            status = {"status": "models_listed", "accessible_model_ids": ids,
                      "logprobs_support": "Not inferable from GET /models; probe an explicit model next"}
        else:
            save_json(run / "config.json", settings.public_config())
            result = probe_model(client, catalog)
            save_json(run / "probe.json", result)
            verified = result["output_logprobs_verified"]
            status = {"status": "verified" if verified else "logprobs_not_verified",
                      "requested_model": settings.model, "response_model": result["response_model"],
                      "responses_verified": True, "output_logprobs_verified": verified}
        save_json(run / "status.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0 if status["status"] in ("models_listed", "verified") else 1
    except (ConfigurationError, APIError, InvalidOutput) as exc:
        status = {"status": "blocked", "error_type": type(exc).__name__, "reason": str(exc),
                  "required_environment_variables": ["OPENAI_API_KEY"] if not os.environ.get("OPENAI_API_KEY") else [], "selected_model": args.model,
                  "llm_episodes_executed": 0}
        if isinstance(exc, InvalidOutput):
            save_json(run / "invalid_probe_response.json", exc.response)
        save_json(run / "status.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
