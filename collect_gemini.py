"""Separate Gemini Developer API collection; no PDDL/OpenAI implementation is modified."""
import argparse
import hashlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import traceback

from collect_normal import select_games
from src.gemini_agent import (GeminiAgent, GeminiAPIError, GeminiClient, GeminiConfigurationError,
                              GeminiInvalidOutput, GeminiSettings, SYSTEM_PROMPT, probability_data, probe_model)
from src.llm_io import append_jsonl, new_run, save_json
from src.validate_gemini import validate

ROOT = Path(__file__).resolve().parent


def open_environment(game, seed):
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

    env = textworld.start(str(game), textworld.EnvInfos(won=True, admissible_commands=True),
                          wrappers=[AlfredDemangler(shuffle=False)])
    try:
        env.seed(seed)
        return env, env.reset()
    except Exception:
        env.close()
        raise


def run_episode(game, episode_id, seed, max_steps, agent, raw_stream, decision_stream):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    previous, env = [], None
    summary = {"episode_id": episode_id, "steps": 0, "done": None, "success": None,
               "truncated": False, "outcome": "environment_error", "error": None}
    try:
        env, state = open_environment(game, seed)
        summary.update(done=state.get("done"), success=state.get("won"))
        match = re.search(r"Your task is to:\s*([^\n]+)", state.get("feedback") or "")
        goal = match.group(1) if match else None
        for step in range(1, max_steps + 1):
            admissible = state.get("admissible_commands")
            if not isinstance(admissible, list) or not admissible:
                raise RuntimeError("No environment admissible actions")
            event = {"episode_id": episode_id, "step": step, "goal": goal, "observation": state.get("feedback"),
                     "plan": None, "action": None, "admissible_actions": list(admissible), "previous_actions": list(previous),
                     "reward": None, "done": None, "success": None, "agent_type": "llm",
                     "provider": "google_gemini_developer_api", "model_name": None,
                     "requested_model": agent.client.settings.model, "token_logprobs": None,
                     "logprobs_available": False, "logprobs_result": None, "raw_response": None, "validation_error": None}
            try:
                decision = agent.decide(goal, event["observation"], previous, admissible)
            except GeminiInvalidOutput as exc:
                event.update(raw_response=exc.response, model_name=exc.response.get("modelVersion"),
                             validation_error=str(exc), **probability_data(exc.response))
                if exc.parsed is not None:
                    event.update(plan=exc.parsed.get("plan"), action=exc.parsed.get("action"))
                append_jsonl(decision_stream, event)
                summary.update(outcome="invalid_output", error=str(exc))
                break
            except GeminiAPIError as exc:
                event["validation_error"] = str(exc)
                append_jsonl(decision_stream, event)
                summary.update(outcome="api_error", error=str(exc))
                break
            event.update(plan=decision.plan, action=decision.action, raw_response=decision.response,
                         model_name=decision.response.get("modelVersion"), **probability_data(decision.response))
            append_jsonl(decision_stream, event)
            next_state, reward, done = env.step(decision.action)
            success = next_state.get("won")
            truncated = step == max_steps and not done
            row = {key: value for key, value in event.items() if key not in ("raw_response", "validation_error")}
            row.update(reward=reward, done=done, success=success, next_observation=next_state.get("feedback"),
                       truncated=truncated, response_id=decision.response.get("responseId"), usage=decision.response.get("usageMetadata"))
            append_jsonl(raw_stream, row)
            summary.update(steps=step, done=done, success=success, truncated=truncated)
            previous.append(decision.action)
            state = next_state
            if done:
                summary["outcome"] = "success" if success is True else "task_failure" if success is False else "unknown_terminal"
                break
            if truncated:
                summary["outcome"] = "step_limit"
    except Exception as exc:
        summary.update(outcome="environment_error", error=f"Environment/collection exception: {type(exc).__name__}")
        summary["error_frames"] = [{"file": f.filename, "line": f.lineno, "function": f.name} for f in traceback.extract_tb(exc.__traceback__)]
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                summary.update(outcome="environment_error", error=f"Environment close exception: {type(exc).__name__}")
    return summary


def collect(args):
    run = new_run(args.output_dir)
    logger = logging.getLogger(str(run))
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(run / "run.log", mode="x", encoding="utf-8"))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("RUN_DIR=%s", run)
    try:
        settings = GeminiSettings.from_env(model=args.model, response_logprobs=not args.without_logprobs,
                                           top_logprobs=args.top_logprobs, max_output_tokens=args.max_output_tokens,
                                           timeout=args.timeout, request_interval=args.request_interval)
        if not settings.model:
            raise GeminiConfigurationError("List actual models with check_gemini.py models; explicitly select and probe a Flash model first")
        save_json(run / "api_config.json", settings.public_config())
        client = GeminiClient(settings)
        catalog = client.list_models()
        save_json(run / "models.json", catalog)
        capabilities = probe_model(client, catalog)
        save_json(run / "capabilities.json", capabilities)
        if settings.response_logprobs and not capabilities["logprobs_verified"]:
            raise GeminiConfigurationError("Generation succeeded but actual token logprobs were not returned; do not claim support")
        data_root = args.data_dir.resolve()
        os.environ["ALFWORLD_DATA"] = str(data_root)
        games = select_games(data_root, args.split, args.task_type, args.episodes, args.seed)
        sources = [Path(__file__), ROOT / "src/gemini_agent.py", ROOT / "src/validate_gemini.py"]
        config = {"provider": "google_gemini_developer_api", "agent_type": "llm", "api_config": settings.public_config(),
                  "system_prompt": SYSTEM_PROMPT, "seed": args.seed, "episodes_requested": args.episodes,
                  "max_steps": args.max_steps, "split": args.split, "task_type": args.task_type, "python": sys.version,
                  "versions": {name: version(name) for name in ("alfworld", "textworld", "numpy")},
                  "selection": "Same games/seeds as preserved baseline; expert plan/walkthrough/facts never sent to agent",
                  "selected_games": [{"gamefile": str(p.relative_to(data_root)), "episode_seed": args.seed + index,
                                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for index, p in enumerate(games)],
                  "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  "field_timing": "Goal/observation/history/plan/action/admissible actions before action; reward/done/success after action",
                  "model_name_source": "Actual response.modelVersion; null if omitted; requested_model stored separately",
                  "probabilities_source": "Unmodified candidates[0].logprobsResult.chosenCandidates plus full logprobsResult",
                  "entropy_note": "Chosen-token and top-k logprobs are not a full vocabulary probability distribution; full Shannon entropy is not established"}
        save_json(run / "config.json", config)
        agent, summaries = GeminiAgent(client), []
        with (run / "trajectories.jsonl").open("x", encoding="utf-8") as raw, (run / "decisions.jsonl").open("x", encoding="utf-8") as decisions, (run / "episode_outcomes.jsonl").open("x", encoding="utf-8") as outcomes:
            for index, game in enumerate(games):
                episode_id = str(game.parent.relative_to(data_root / "json_2.1.1"))
                summary = run_episode(game, episode_id, args.seed + index, args.max_steps, agent, raw, decisions)
                summaries.append(summary)
                append_jsonl(outcomes, summary)
                logger.info("EPISODE %s", json.dumps(summary))
                if summary["outcome"] == "api_error":
                    break
        save_json(run / "episodes.json", summaries)
        report = validate(run / "trajectories.jsonl", summaries, args.episodes)
        save_json(run / "validation.json", report)
        save_json(run / "status.json", {"status": "complete" if report["data_valid"] and report["experiment_complete"] else "incomplete", "report": report})
        logger.info("RESULT %s", json.dumps(report))
        return 0 if report["data_valid"] and report["experiment_complete"] else 1
    except (GeminiAPIError, GeminiConfigurationError, GeminiInvalidOutput) as exc:
        if isinstance(exc, GeminiInvalidOutput):
            save_json(run / "invalid_probe_response.json", exc.response)
        status = {"status": "blocked_before_episodes", "reason": str(exc), "error_type": type(exc).__name__,
                  "http_status": getattr(exc, "http_status", None), "google_status": getattr(exc, "api_status", None),
                  "episodes_executed": 0, "success_count": 0, "failure_count": 0, "mean_steps": None,
                  "required_environment_variables": ["GEMINI_API_KEY"] if not os.environ.get("GEMINI_API_KEY") else []}
        save_json(run / "status.json", status)
        logger.error("BLOCKED %s", json.dumps(status))
        return 2
    except Exception as exc:
        save_json(run / "fatal_error.json", {"error_type": type(exc).__name__, "note": "Raw files retained; inspect durable episode outcomes for counts"})
        return 1
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Actual probed Flash ID, no default")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--request-interval", type=float, default=13)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--without-logprobs", action="store_true", help="Explicit generation-only mode; never a silent fallback")
    parser.add_argument("--split", choices=["train", "valid_seen", "valid_unseen", "valid_train"], default="train")
    parser.add_argument("--task-type", choices=["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"], default="pick_and_place_simple")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/alfworld")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs_gemini")
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or not 0 <= args.seed < 2**32 - args.episodes:
        parser.error("episodes/max_steps must be positive; seed must fit uint32")
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
