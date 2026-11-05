"""Local NVIDIA trajectory workflow. No model/environment is started by import or --check-imports."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import traceback

from collect_pddl import select_games
from local_llm_agent import LocalConfigurationError, LocalGenerationError, LocalInvalidOutput, LocalLLMAgent, LocalLLMConfig, SYSTEM_PROMPT
from src.llm_io import append_jsonl, new_run, save_json
from src.validate_local import validate

ROOT = Path(__file__).resolve().parent


def check_imports():
    import torch
    import transformers
    import accelerate
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

    try:
        bits_version = version("bitsandbytes")
    except PackageNotFoundError:
        bits_version = None
    return {"torch": torch.__version__, "transformers": transformers.__version__, "accelerate": accelerate.__version__,
            "bitsandbytes_version": bits_version, "model_loaded": False, "gpu_execution_attempted": False,
            "note": "Import-only check. bitsandbytes runtime, CUDA, model loading and ALFWorld episodes are not verified."}


def open_environment(game, seed):
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

    env = textworld.start(str(game), textworld.EnvInfos(won=True, admissible_commands=True), wrappers=[AlfredDemangler(shuffle=False)])
    try:
        env.seed(seed)
        return env, env.reset()
    except Exception:
        env.close()
        raise


def run_episode(game, episode_id, seed, max_steps, agent, raw_stream, decision_stream, *, logits_dir=None, env_factory=None):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    env, previous = None, []
    summary = {"episode_id": episode_id, "steps": 0, "done": None, "success": None,
               "truncated": False, "outcome": "environment_error", "error": None}
    try:
        env, state = (env_factory or open_environment)(game, seed)
        summary.update(done=state.get("done"), success=state.get("won"))
        match = re.search(r"Your task is to:\s*([^\n]+)", state.get("feedback") or "")
        goal = match.group(1) if match else None
        for step in range(1, max_steps + 1):
            admissible = state.get("admissible_commands")
            if not isinstance(admissible, list) or not admissible:
                raise LocalGenerationError("No admissible actions supplied by environment")
            event = {"episode_id": episode_id, "step": step, "goal": goal, "observation": state.get("feedback"),
                     "plan": None, "action_id": None, "action": None, "raw_model_output": None,
                     "action_selection_format": "action_id_v1", "admissible_actions": list(admissible), "previous_actions": list(previous),
                     "reward": None, "done": None, "success": None, "agent_type": "llm", "backend": "local",
                     "model_name": agent.config.model_name, "token_entropies": None, "mean_entropy": None,
                     "generation": None, "validation_error": None}
            try:
                logits_path = Path(logits_dir) / f"step_{step:05d}.pt" if logits_dir is not None else None
                decision = agent.decide(goal, event["observation"], previous, admissible, logits_path=logits_path)
            except LocalInvalidOutput as exc:
                event.update(generation=exc.generation, token_entropies=exc.generation["token_entropies"],
                             mean_entropy=exc.generation["mean_entropy"], raw_model_output=exc.generation["raw_text"], validation_error=str(exc))
                if exc.parsed is not None:
                    event.update(plan=exc.parsed.get("plan"), action_id=exc.parsed.get("action_id"), action=exc.parsed.get("action"))
                append_jsonl(decision_stream, event)
                summary.update(outcome="invalid_output", error=str(exc))
                break
            except (LocalGenerationError, LocalConfigurationError) as exc:
                event["validation_error"] = str(exc)
                append_jsonl(decision_stream, event)
                summary.update(outcome="generation_error", error=str(exc))
                break
            except Exception as exc:
                event["validation_error"] = f"Local generation exception: {type(exc).__name__}"
                append_jsonl(decision_stream, event)
                summary.update(outcome="generation_error", error=event["validation_error"])
                break
            event.update(plan=decision.plan, action_id=decision.action_id, action=decision.action, raw_model_output=decision.generation["raw_text"], generation=decision.generation,
                         token_entropies=decision.generation["token_entropies"], mean_entropy=decision.generation["mean_entropy"])
            append_jsonl(decision_stream, event)
            next_state, reward, done = env.step(decision.action)
            success, truncated = next_state.get("won"), step == max_steps and not done
            row = {key: value for key, value in event.items() if key not in ("generation", "validation_error")}
            row.update({key: value for key, value in decision.generation.items() if key != "raw_text"})
            row.update(reward=reward, done=done, success=success, truncated=truncated, next_observation=next_state.get("feedback"))
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
        summary.update(outcome="environment_error", error=f"Environment/collection exception: {type(exc).__name__}")
        summary["error_frames"] = [{"file": f.filename, "line": f.lineno, "function": f.name} for f in traceback.extract_tb(exc.__traceback__)]
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                summary.update(outcome="environment_error", error=f"Environment close exception: {type(exc).__name__}")
    return summary


def collect(args, config):
    run = new_run(args.output_dir)
    logger = logging.getLogger(str(run))
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(run / "run.log", mode="x", encoding="utf-8"))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("RUN_DIR=%s", run)
    try:
        data_root = args.data_dir.resolve()
        os.environ["ALFWORLD_DATA"] = str(data_root)
        games = select_games(data_root, args.split, args.task_type, args.episodes, args.seed)
        packages = {}
        for name in ("torch", "transformers", "accelerate", "bitsandbytes", "alfworld", "textworld"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        files = [Path(__file__), ROOT / "local_llm_agent.py", ROOT / "src/validate_local.py"]
        save_json(run / "config.json", {"backend": "local", "local_llm": config.public_config(), "packages": packages,
                                       "python": sys.version, "seed": args.seed, "episodes_requested": args.episodes,
                                       "max_steps": args.max_steps, "system_prompt": SYSTEM_PROMPT,
                                       "action_selection_format": "action_id_v1",
                                       "action_mapping": "Zero-based IDs in current original admissible_actions order; action = admissible_actions[action_id]",
                                       "generation_options": {"do_sample": False, "num_beams": 1, "use_model_defaults": False,
                                                              "return_dict_in_generate": True, "output_scores": True,
                                                              "output_logits": config.save_logits, "vocabulary_mask": None},
                                       "entropy_unit": "nats", "entropy_scope": "Full vocabulary generate.scores; fresh greedy config, no warpers/action-token masks. All generated tokens including EOS, excluding prompt.",
                                       "mean_entropy_scope": "Arithmetic mean of generated PLAN text, ACTION_ID labels/digits/separators and EOS in this one ALFWorld step; not whole-episode average or action-only entropy",
                                       "selected_games": [{"gamefile": str(p.relative_to(data_root)), "episode_seed": args.seed + i,
                                                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for i, p in enumerate(games)],
                                       "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
        import torch
        torch.manual_seed(args.seed)
        agent = LocalLLMAgent(config).load()
        save_json(run / "model_info.json", {"model_name": config.model_name, "requested_revision": config.revision,
                                            "resolved_revision": getattr(agent.model.config, "_commit_hash", None)})
        summaries = []
        with (run / "trajectories.jsonl").open("x", encoding="utf-8") as raw, (run / "decisions.jsonl").open("x", encoding="utf-8") as decisions, (run / "episode_outcomes.jsonl").open("x", encoding="utf-8") as outcomes:
            for index, game in enumerate(games):
                torch.manual_seed(args.seed + index)
                episode_id = str(game.parent.relative_to(data_root / "json_2.1.1"))
                summary = run_episode(game, episode_id, args.seed + index, args.max_steps, agent, raw, decisions,
                                      logits_dir=run / "logits" / f"episode_{index:04d}")
                summaries.append(summary)
                append_jsonl(outcomes, summary)
                logger.info("EPISODE %s", json.dumps(summary))
                if summary["outcome"] == "generation_error":
                    break  # Do not repeat unchanged OOM/model failures across remaining episodes.
        save_json(run / "episodes.json", summaries)
        report = validate(run / "trajectories.jsonl", summaries, args.episodes)
        save_json(run / "validation.json", report)
        save_json(run / "status.json", {"status": "complete" if report["data_valid"] and report["experiment_complete"] else "incomplete", "report": report})
        logger.info("RESULT %s", json.dumps(report))
        return 0 if report["data_valid"] and report["experiment_complete"] else 1
    except Exception as exc:
        save_json(run / "fatal_error.json", {"error_type": type(exc).__name__,
                                            "note": "Inspect config and durable outputs; verify local CUDA, bitsandbytes, disk, model access and context budget. No outputs overwritten."})
        logger.error("Local setup/run interrupted: %s", type(exc).__name__)
        return 1
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-logits", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16"])
    parser.add_argument("--device")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--check-imports", action="store_true", help="Imports only; no model download, CUDA query/execution, or episode")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--split", choices=["train", "valid_seen", "valid_unseen", "valid_train"], default="train")
    parser.add_argument("--task-type", choices=["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"], default="pick_and_place_simple")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/alfworld")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs_local")
    args = parser.parse_args()
    if args.check_imports:
        print(json.dumps(check_imports(), indent=2))
        return 0
    if args.episodes <= 0 or args.max_steps <= 0 or not 0 <= args.seed < 2**32 - args.episodes:
        parser.error("episodes/max_steps must be positive; seed must fit uint32")
    try:
        config = LocalLLMConfig.from_file(args.config, model_name=args.model, revision=args.model_revision,
                                         load_in_4bit=args.load_in_4bit, save_logits=args.save_logits,
                                         compute_dtype=args.compute_dtype, device=args.device,
                                         max_new_tokens=args.max_new_tokens, max_context_tokens=args.max_context_tokens)
    except (LocalConfigurationError, ValueError, OSError) as exc:
        parser.error(f"Invalid local config: {type(exc).__name__}; check configuration fields and file")
    return collect(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
