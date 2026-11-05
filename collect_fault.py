"""Local-only runtime input fault collection. Normal collector/model implementation is untouched."""
import argparse
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

from collect_local import check_imports
from collect_pddl import select_games
from local_llm_agent import LocalLLMAgent, LocalLLMConfig, SYSTEM_PROMPT
from src.fault_injection import FAULTS, FaultConfig, resolve_timing
from src.fault_runtime import compare_pair, run_condition
from src.llm_io import append_jsonl, new_run, save_json

ROOT = Path(__file__).resolve().parent


def select_experiment_games(args):
    data_root = args.data_dir.resolve()
    if args.game is not None or args.episode_id is not None:
        if args.episodes != 1:
            raise ValueError("A fixed game/episode_id requires --episodes 1; paired mode still executes both conditions")
        dataset = (data_root / "json_2.1.1").resolve()
        if args.episode_id is not None:
            game = (dataset / args.episode_id / "game.tw-pddl").resolve()
            if not game.is_relative_to(dataset):
                raise ValueError("episode_id must stay under the dataset root")
        else:
            game = args.game.resolve()
            if game.is_dir():
                game = game / "game.tw-pddl"
        if not game.is_file() or game.name != "game.tw-pddl":
            raise FileNotFoundError("Specify an existing game.tw-pddl or its containing directory")
        return [game]
    return select_games(data_root, args.split, args.task_type, args.episodes, args.seed)


def game_identity(game, data_root):
    dataset = (data_root / "json_2.1.1").resolve()
    game = Path(game).resolve()
    if game.is_relative_to(dataset):
        return str(game.parent.relative_to(dataset))
    return "external/" + hashlib.sha256(game.read_bytes()).hexdigest() + "/" + game.parent.name


def baseline_steps_from_run(run_dir, episode_id):
    if run_dir is None:
        return None
    summaries = json.loads((Path(run_dir) / "episodes.json").read_text())
    matches = [row for row in summaries if row.get("episode_id") == episode_id]
    if len(matches) != 1 or type(matches[0].get("steps")) is not int or matches[0]["steps"] < 1:
        raise ValueError("baseline-run must contain exactly one matching episode with a positive actual step count")
    return matches[0]["steps"]


def resolve_game_timing(args, game, baseline_steps=None):
    reference = args.reference_steps
    source = None
    if args.injection_step is None and baseline_steps is None and reference is None:
        data = json.loads(Path(game).read_text())
        walkthrough = data.get("walkthrough")
        if isinstance(walkthrough, list) and walkthrough:
            reference = len(walkthrough)
            source = "pddl_walkthrough_reference_steps"
    timing = resolve_timing(args.injection_step, baseline_steps, reference if reference is not None else 20, args.max_steps)
    if source is not None:
        timing["source"] = source
    return timing


def run_game(game, episode_id, seed, agent, args, output_dir, *, seed_runtime, env_factory=None):
    """Injectable orchestration for unit tests; actual torch seeding is supplied only by local CLI."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries, reports = [], []
    baseline = None
    if args.paired or args.condition == "no_fault":
        seed_runtime(seed)
        baseline_config = FaultConfig("no_fault", args.severity, None)
        baseline, report = run_condition(game, episode_id, seed, args.max_steps, agent, baseline_config,
                                         output_dir / "no_fault", condition="no_fault", env_factory=env_factory)
        summaries.append(baseline)
        reports.append(report)
        if not args.paired:
            return {"summaries": summaries, "reports": reports, "comparison": None}
        if baseline["outcome"] in ("generation_error", "environment_error"):
            comparison = {"episode_id": episode_id, "seed": seed, "comparison": "paired_incomplete_baseline_runtime_error",
                          "fault_condition_executed": False}
            save_json(output_dir / "pair.json", comparison)
            return {"summaries": summaries, "reports": reports, "comparison": comparison}
    length = baseline["steps"] if baseline is not None and baseline["steps"] > 0 else baseline_steps_from_run(args.baseline_run, episode_id)
    timing = resolve_game_timing(args, game, length)
    save_json(output_dir / "timing.json", timing)
    fault_config = FaultConfig(args.fault, args.severity, timing["injection_step"])
    seed_runtime(seed)
    fault, report = run_condition(game, episode_id, seed, args.max_steps, agent, fault_config,
                                  output_dir / "fault", timing=timing, env_factory=env_factory)
    summaries.append(fault)
    reports.append(report)
    comparison = compare_pair(baseline, fault, output_dir / "no_fault", output_dir / "fault") if baseline is not None else None
    if comparison is not None:
        save_json(output_dir / "pair.json", comparison)
    return {"summaries": summaries, "reports": reports, "comparison": comparison}


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
        games = select_experiment_games(args)
        sources = [Path(__file__), ROOT / "src/fault_injection.py", ROOT / "src/fault_runtime.py", ROOT / "src/validate_fault.py",
                   ROOT / "collect_local.py", ROOT / "local_llm_agent.py"]
        packages = {}
        for name in ("torch", "transformers", "accelerate", "bitsandbytes", "alfworld", "textworld"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        save_json(run / "config.json", {"agent": "local", "local_llm": config.public_config(), "packages": packages, "python": sys.version,
                                       "system_prompt": SYSTEM_PROMPT, "fault_type": args.fault, "severity": args.severity, "paired": args.paired,
                                       "condition": "paired" if args.paired else args.condition, "requested_injection_step": args.injection_step,
                                       "reference_steps": args.reference_steps, "baseline_run": str(args.baseline_run) if args.baseline_run else None,
                                       "max_steps": args.max_steps, "seed": args.seed, "episodes_requested": args.episodes,
                                       "selected_games": [{"episode_id": game_identity(game, data_root), "game_path": str(game),
                                                           "sha256": hashlib.sha256(game.read_bytes()).hexdigest(), "seed": args.seed + index}
                                                          for index, game in enumerate(games)],
                                       "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                                       "invariants": "Environment/true goal/admissible action order/ACTION_ID mapping/generation options/entropy implementation unchanged",
                                       "injection_boundary": "Immediately before agent.decide; context noise appended only to agent_observation",
                                       "scope": "No memory_noise, HHI, LSTM, feature classifier, or intervention"})
        import torch
        torch.manual_seed(args.seed)
        agent = LocalLLMAgent(config).load()
        save_json(run / "model_info.json", {"model_name": config.model_name, "requested_revision": config.revision,
                                            "resolved_revision": getattr(agent.model.config, "_commit_hash", None)})
        results = []
        with (run / "outcomes.jsonl").open("x", encoding="utf-8") as outcomes:
            for index, game in enumerate(games):
                episode_id = game_identity(game, data_root)
                result = run_game(game, episode_id, args.seed + index, agent, args, run / "episodes" / f"{index:04d}", seed_runtime=torch.manual_seed)
                results.append(result)
                append_jsonl(outcomes, result)
                logger.info("RESULT %s", json.dumps(result))
                if any(row["outcome"] in ("generation_error", "environment_error") for row in result["summaries"]):
                    break
        complete = len(results) == len(games) and all(report["data_valid"] and report["experiment_complete"] for result in results for report in result["reports"])
        if args.paired:
            complete = complete and all(len(result["summaries"]) == 2 for result in results)
        save_json(run / "results.json", results)
        fault_conditions = [row for result in results for row in result["summaries"] if row["condition"] == "fault"]
        fault_ready = complete and bool(fault_conditions) and all(row["effective_perturbation_reached"] for row in fault_conditions)
        save_json(run / "status.json", {"status": "complete" if complete else "incomplete", "fault_data_ready": fault_ready, "games_attempted": len(results),
                                       "conditions_executed": sum(len(result["summaries"]) for result in results),
                                       "fault_not_reached_count": sum(not row["effective_perturbation_reached"] for result in results for row in result["summaries"] if row["condition"] == "fault"),
                                       "note": "A completed run may contain no effective exposure if the episode ended early; inspect pair results and first_effective_input_step"})
        return 0 if complete else 1
    except Exception as exc:
        save_json(run / "fatal_error.json", {"error_type": type(exc).__name__, "note": "All outputs retained; inspect per-condition outcomes. No execution counts inferred."})
        logger.error("Fault collection interrupted: %s", type(exc).__name__)
        return 1
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["local"], default="local")
    parser.add_argument("--fault", choices=FAULTS, required=True)
    parser.add_argument("--severity", choices=["low", "high"], default="low")
    parser.add_argument("--injection-step", type=int)
    parser.add_argument("--reference-steps", type=int, help="Explicit timing reference; default uses baseline length, then game walkthrough reference, then 20")
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--condition", choices=["no_fault", "fault"], default="fault")
    game_group = parser.add_mutually_exclusive_group()
    game_group.add_argument("--game", type=Path)
    game_group.add_argument("--episode-id")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-logits", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16"])
    parser.add_argument("--device")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument("--episodes", type=int, default=1, help="Number of distinct games/pairs, not total condition executions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--split", choices=["train", "valid_seen", "valid_unseen", "valid_train"], default="train")
    parser.add_argument("--task-type", choices=["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"], default="pick_and_place_simple")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/alfworld")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs_fault")
    args = parser.parse_args()
    if args.check_imports:
        print(json.dumps(check_imports(), indent=2))
        return 0
    try:
        if args.episodes < 1 or not 0 <= args.seed < 2**32 - args.episodes:
            raise ValueError("episodes must be positive and seed fit uint32")
        resolve_timing(args.injection_step, reference_steps=args.reference_steps if args.reference_steps is not None else 20, max_steps=args.max_steps)
        if args.paired and args.baseline_run:
            raise ValueError("Paired mode uses its freshly collected no_fault length, not an external baseline-run")
        if (args.game or args.episode_id) and args.episodes != 1:
            raise ValueError("Fixed game selection requires --episodes 1")
        config = LocalLLMConfig.from_file(args.config, model_name=args.model, revision=args.model_revision,
                                         load_in_4bit=args.load_in_4bit, save_logits=args.save_logits, compute_dtype=args.compute_dtype,
                                         device=args.device, max_new_tokens=args.max_new_tokens, max_context_tokens=args.max_context_tokens)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return collect(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
