"""Separate OpenAI Responses trajectory collector; preserves unsuccessful runs."""
import argparse
import hashlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import platform
import random
import re
import sys
import traceback

from collect_normal import select_games
from src.llm_agent import (APIError, ConfigurationError, InvalidOutput, LLMAgent,
                           OpenAIResponsesClient, OpenAISettings, SYSTEM_PROMPT,
                           output_blocks, probe_model)
from src.llm_io import append_jsonl, new_run, save_json
from src.validate_llm import validate

ROOT = Path(__file__).resolve().parent


def open_environment(game, seed):
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

    # No AlfredExpert wrapper and no policy_commands/facts/walkthrough infos requested.
    infos = textworld.EnvInfos(won=True, admissible_commands=True)
    env = textworld.start(str(game), infos, wrappers=[AlfredDemangler(shuffle=False)])
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
    summary = {"episode_id": episode_id, "steps": 0, "done": None, "success": None,
               "truncated": False, "outcome": "environment_error", "error": None}
    env = None
    previous = []
    try:
        env, state = open_environment(game, seed)
        summary.update(success=state.get("won"), done=state.get("done"))
        match = re.search(r"Your task is to:\s*([^\n]+)", state.get("feedback") or "")
        goal = match.group(1) if match else None
        for step in range(1, max_steps + 1):
            actions = state.get("admissible_commands")
            if not isinstance(actions, list) or not actions:
                raise RuntimeError("Environment provided no admissible actions")
            event = {"episode_id": episode_id, "step": step, "goal": goal,
                     "observation": state.get("feedback"), "plan": None, "action": None,
                     "admissible_actions": list(actions), "previous_actions": list(previous),
                     "reward": None, "done": None, "success": None, "agent_type": "llm",
                     "model_name": None, "requested_model": agent.client.settings.model,
                     "raw_response": None, "validation_error": None}
            try:
                decision = agent.decide(goal, event["observation"], previous, actions)
            except InvalidOutput as exc:
                event.update(raw_response=exc.response, validation_error=str(exc),
                             model_name=exc.response.get("model"))
                if exc.parsed is not None:
                    event.update(plan=exc.parsed.get("plan"), action=exc.parsed.get("action"))
                append_jsonl(decision_stream, event)
                summary.update(outcome="invalid_output", error=str(exc))
                break
            except APIError as exc:
                event["validation_error"] = str(exc)
                append_jsonl(decision_stream, event)
                summary.update(outcome="api_error", error=str(exc))
                break
            event.update(plan=decision.plan, action=decision.action, raw_response=decision.response,
                         model_name=decision.response.get("model"))
            # Persist raw output before env.step, including outputs preceding an environment error.
            append_jsonl(decision_stream, event)
            next_state, reward, done = env.step(decision.action)
            success = next_state.get("won")
            truncated = step == max_steps and not done
            row = {field: event[field] for field in (
                "episode_id", "step", "goal", "observation", "plan", "action",
                "admissible_actions", "previous_actions", "agent_type", "model_name", "requested_model")}
            blocks = output_blocks(decision.response)
            row.update(reward=reward, done=done, success=success,
                       next_observation=next_state.get("feedback"), truncated=truncated,
                       response_id=decision.response.get("id"), usage=decision.response.get("usage"),
                       output_logprobs=blocks if blocks else None)
            append_jsonl(raw_stream, row)
            previous.append(decision.action)
            state = next_state
            summary.update(steps=step, done=done, success=success, truncated=truncated)
            if done:
                summary["outcome"] = "success" if success is True else "task_failure" if success is False else "unknown_terminal"
                break
            if truncated:
                summary["outcome"] = "step_limit"
    except Exception as exc:
        # Environment exceptions are typed, without logging arbitrary data from underlying code.
        summary.update(outcome="environment_error", error=f"Environment/collection exception: {type(exc).__name__}")
        summary["error_frames"] = [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                                   for frame in traceback.extract_tb(exc.__traceback__)]
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
    handler = logging.FileHandler(run / "run.log", mode="x", encoding="utf-8")
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("RUN_DIR=%s", run)
    try:
        # Missing credentials stop before model lookup, API calls, environment resets, or trajectory creation.
        settings = OpenAISettings.from_env(model=args.model, timeout=args.timeout,
                                          top_logprobs=args.top_logprobs,
                                          max_output_tokens=args.max_output_tokens)
        if not settings.model:
            raise ConfigurationError("List accessible models with check_openai.py models, then explicitly set --model or ALFWORLD_LLM_MODEL")
        client = OpenAIResponsesClient(settings)
        save_json(run / "api_config.json", settings.public_config())
        catalog = client.list_models()
        save_json(run / "models.json", catalog)
        probe = probe_model(client, catalog)
        save_json(run / "capabilities.json", probe)
        if not probe["output_logprobs_verified"]:
            raise ConfigurationError("Selected model did not return output logprobs in the actual probe; model selection remains unverified")
        data_root = args.data_dir.resolve()
        os.environ["ALFWORLD_DATA"] = str(data_root)
        games = select_games(data_root, args.split, args.task_type, args.episodes, args.seed)
        source_files = [Path(__file__), ROOT / "src/llm_agent.py", ROOT / "src/llm_io.py", ROOT / "src/validate_llm.py"]
        config = {"agent_type": "llm", "provider": "OpenAI", "api": "Responses", "seed": args.seed,
                  "episodes_requested": args.episodes, "max_steps": args.max_steps,
                  "split": args.split, "task_type": args.task_type, "python": sys.version,
                  "platform": platform.platform(), "versions": {name: version(name) for name in ("alfworld", "textworld", "numpy")},
                  "system_prompt": SYSTEM_PROMPT, "api_config": settings.public_config(),
                  "selection": "Same deterministic game selection as preserved PDDL baseline; no expert output is sent to LLM",
                  "selected_games": [{"gamefile": str(p.relative_to(data_root)),
                                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                      "episode_seed": args.seed + index} for index, p in enumerate(games)],
                  "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                  "field_timing": "Inputs/plan/action before action; reward/done/success after action",
                  "plan_source": "Model output plan, unmodified; no expert fallback or output repair",
                  "success_source": "Post-action environment won, null if unavailable",
                  "model_name_source": "Actual response.model; null if omitted; requested_model stored separately",
                  "entropy_note": "Raw returned token logprobs/top alternatives only. Top-k is not the full vocabulary distribution; constrained JSON tokens are not an unconstrained action distribution."}
        save_json(run / "config.json", config)
        agent = LLMAgent(client)
        summaries = []
        with (run / "trajectories.jsonl").open("x", encoding="utf-8") as raw, (run / "decisions.jsonl").open("x", encoding="utf-8") as decisions:
            for index, game in enumerate(games):
                episode_id = str(game.parent.relative_to(data_root / "json_2.1.1"))
                summary = run_episode(game, episode_id, args.seed + index, args.max_steps, agent, raw, decisions)
                summaries.append(summary)
                # Durable checkpoint: failed/partial episodes survive later interruption.
                with (run / "episode_outcomes.jsonl").open("a", encoding="utf-8") as outcomes:
                    append_jsonl(outcomes, summary)
                logger.info("EPISODE %s", json.dumps(summary))
                if summary["outcome"] == "api_error":
                    logger.info("Stopping further episodes after API failure; selected but unrun episodes remain in config")
                    break
        save_json(run / "episodes.json", summaries)
        report = validate(run / "trajectories.jsonl", summaries, args.episodes)
        save_json(run / "validation.json", report)
        logger.info("RESULT %s", json.dumps(report))
        status = {"status": "complete" if report["data_valid"] and report["experiment_complete"] else "incomplete",
                  "llm_episodes_executed": len(summaries), "success_count": report["success_count"],
                  "task_failure_count": report["task_failure_count"],
                  "mean_steps": report["mean_executed_steps_per_attempt"]}
        save_json(run / "status.json", status)
        return 0 if status["status"] == "complete" else 1
    except (ConfigurationError, APIError, InvalidOutput) as exc:
        if isinstance(exc, InvalidOutput):
            save_json(run / "invalid_probe_response.json", exc.response)
        status = {"status": "blocked_before_episodes", "reason": str(exc), "error_type": type(exc).__name__,
                  "llm_episodes_executed": 0, "success_count": 0, "failure_count": 0,
                  "mean_steps": None, "model_name": None,
                  "required_environment_variables": ["OPENAI_API_KEY"] if not os.environ.get("OPENAI_API_KEY") else []}
        save_json(run / "status.json", status)
        logger.error("BLOCKED %s", json.dumps(status))
        return 2
    except Exception as exc:
        # Do not overwrite outputs or mislabel a post-start exception as zero executed episodes.
        save_json(run / "fatal_error.json", {"error_type": type(exc).__name__,
                                            "note": "Inspect durable episode_outcomes.jsonl, decisions.jsonl and raw data; execution counts are not inferred"})
        logger.error("Run interrupted: %s; raw outputs retained", type(exc).__name__)
        return 1
    finally:
        for current in list(logger.handlers):
            current.close()
            logger.removeHandler(current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Explicit accessible model ID; no default")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--split", choices=["train", "valid_seen", "valid_unseen", "valid_train"], default="train")
    parser.add_argument("--task-type", choices=["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
                        "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"], default="pick_and_place_simple")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/alfworld")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs_llm")
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or not 0 <= args.seed < 2**32 - args.episodes:
        parser.error("episodes/max_steps must be positive; seeds must fit NumPy uint32")
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
