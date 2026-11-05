"""Check actual Gemini Developer API models and generation/logprobs capabilities."""
import argparse
import json
import os
from pathlib import Path

from src.gemini_agent import (GeminiAPIError, GeminiClient, GeminiConfigurationError,
                              GeminiInvalidOutput, GeminiSettings, flash_models, probe_model)
from src.llm_io import new_run, save_json

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["models", "probe"])
    parser.add_argument("--model", help="Actual listed Flash model ID; no default")
    parser.add_argument("--without-logprobs", action="store_true", help="Explicit text-only capability diagnosis, never silently inferred support")
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--request-interval", type=float, default=13)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs/gemini_checks")
    args = parser.parse_args()
    run = new_run(args.output_dir)
    print("CHECK_DIR=" + str(run), flush=True)
    try:
        settings = GeminiSettings.from_env(model=args.model, response_logprobs=not args.without_logprobs,
                                           top_logprobs=args.top_logprobs, max_output_tokens=args.max_output_tokens,
                                           request_interval=args.request_interval)
        save_json(run / "config.json", settings.public_config())
        client = GeminiClient(settings)
        catalog = client.list_models()
        save_json(run / "models.json", catalog)
        if args.operation == "models":
            status = {"status": "models_listed", "flash_generation_models": flash_models(catalog),
                      "note": "Listing does not verify logprobs or free-tier billing; actual probe is required"}
        else:
            result = probe_model(client, catalog)
            save_json(run / "probe.json", result)
            status = {key: value for key, value in result.items() if key != "response"}
            status["status"] = "verified" if result["logprobs_verified"] else "text_verified_logprobs_unavailable"
        save_json(run / "status.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0 if status["status"] in ("models_listed", "verified") else 1
    except (GeminiConfigurationError, GeminiAPIError, GeminiInvalidOutput) as exc:
        if isinstance(exc, GeminiInvalidOutput):
            save_json(run / "invalid_probe_response.json", exc.response)
        status = {"status": "blocked", "error_type": type(exc).__name__, "reason": str(exc),
                  "http_status": getattr(exc, "http_status", None), "google_status": getattr(exc, "api_status", None),
                  "required_environment_variables": ["GEMINI_API_KEY"] if not os.environ.get("GEMINI_API_KEY") else [],
                  "requested_model": args.model, "episodes_executed": 0}
        save_json(run / "status.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
