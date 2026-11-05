"""Collect genuine ALFWorld text-only transitions using its privileged PDDL expert."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
from uuid import uuid4

from src.validate import validate

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_games(data_root, split, task_type, count, seed):
    candidates = []
    for path in sorted((data_root / "json_2.1.1" / split).glob(f"{task_type}-*/*/game.tw-pddl")):
        if "movable" in str(path) or "Sliced" in str(path):
            continue
        game = json.loads(path.read_text())
        if game.get("solvable") is not True:
            continue
        if not (path.parent / "traj_data.json").is_file():
            raise FileNotFoundError(f"Missing metadata beside {path}; run scripts/download_data.py")
        candidates.append((path, len(game.get("walkthrough") or []) or float("inf")))
    # Prefer the simplest documented games; seeded tie-breaking is reproducible.
    random.Random(seed).shuffle(candidates)
    candidates.sort(key=lambda item: item[1])
    if len(candidates) < count:
        raise ValueError(f"Need {count} distinct games, found {len(candidates)}; run scripts/download_data.py")
    return [path for path, _ in candidates[:count]]


def collect(args):
    data_root = args.data_dir.resolve()
    os.environ["ALFWORLD_DATA"] = str(data_root)
    # Import only after the writable dataset location is established.
    import numpy as np
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid4().hex[:8]
    run_dir = args.output_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(run_dir / "run.log", mode="x", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("RUN_DIR=%s", run_dir)
    summaries = []
    raw_path = run_dir / "trajectories.jsonl"
    try:
        games = select_games(data_root, args.split, args.task_type, args.episodes, args.seed)
        selected = [{"gamefile": str(p.relative_to(data_root)), "sha256": digest(p),
                     "metadata_sha256": digest(p.parent / "traj_data.json"),
                     "episode_seed": args.seed + index} for index, p in enumerate(games)]
        config = {"run_id": run_id, "schema_version": 1, "command": sys.argv,
                  "python": sys.version, "platform": platform.platform(),
                  "versions": {name: version(name) for name in ("alfworld", "textworld", "fast-downward-textworld", "numpy")},
                  "seed": args.seed, "episodes": args.episodes, "max_steps": args.max_steps,
                  "split": args.split, "task_type": args.task_type, "data_dir": str(data_root),
                  "selection": "shortest stored walkthrough; seeded tie-breaking; distinct game files",
                  "agent": "ALFWorld AlfredExpert(planner), privileged symbolic expert; not an LLM",
                  "selected_games": selected,
                  "collector_sha256": digest(Path(__file__)),
                  "field_timing": "observation/plan/action/admissible_actions before action; reward/done/success after action",
                  "success_source": "post-action TextWorld state.won, not inferred from reward",
                  "goal_source": "exact task text extracted from reset feedback; null if unavailable",
                  "reward_source": "unmodified env.step return (PDDL score, binary 0/1 here)",
                  "plan_source": "pre-action extra.expert_plan; not LLM reasoning",
                  "missing_value_policy": "unavailable values are JSON null",
                  "dataset_manifest": json.loads((data_root / "dataset_manifest.json").read_text())
                      if (data_root / "dataset_manifest.json").exists() else None}
        save_json(run_dir / "config.json", config)
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=True)
        with (run_dir / "packages.txt").open("x") as stream:
            stream.write(frozen.stdout)
        with raw_path.open("x", encoding="utf-8", buffering=1) as stream:
            for index, game in enumerate(games):
                episode_id = str(game.parent.relative_to(data_root / "json_2.1.1"))
                episode_seed = args.seed + index
                random.seed(episode_seed)
                np.random.seed(episode_seed)
                env = None
                summary = {"episode_id": episode_id, "steps": 0, "success": None,
                           "done": None, "truncated": False, "error": None}
                try:
                    infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["expert_plan"])
                    env = textworld.start(str(game), infos, wrappers=[
                        AlfredDemangler(shuffle=False), AlfredExpert(expert_type="planner")])
                    env.seed(episode_seed)
                    state = env.reset()
                    match = re.search(r"Your task is to:\s*([^\n]+)", state.get("feedback") or "")
                    goal = match.group(1) if match else None
                    for step in range(1, args.max_steps + 1):
                        plan = state.get("extra.expert_plan")
                        admissible = state.get("admissible_commands")
                        if not plan:
                            raise RuntimeError("Environment supplied no expert action; refusing to invent one")
                        action = plan[0]
                        if admissible is not None and action not in admissible:
                            raise RuntimeError(f"Expert action is not admissible: {action}")
                        observation = state.get("feedback")
                        next_state, reward, done = env.step(action)
                        success = next_state.get("won")
                        truncated = step == args.max_steps and done is not True
                        row = {"episode_id": episode_id, "step": step, "goal": goal,
                               "observation": observation, "plan": plan, "action": action,
                               "admissible_actions": admissible, "reward": reward, "done": done,
                               "success": success, "next_observation": next_state.get("feedback"),
                               "truncated": truncated}
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        summary.update(steps=step, success=success, done=done, truncated=truncated)
                        state = next_state
                        if done:
                            break
                except Exception as exc:
                    summary["error"] = f"{type(exc).__name__}: {exc}"
                    logger.exception("Episode failed: %s", episode_id)
                finally:
                    if env is not None:
                        env.close()
                summaries.append(summary)
                logger.info("EPISODE %s", json.dumps(summary, ensure_ascii=False))
        save_json(run_dir / "episodes.json", summaries)
        # Reopen raw data, rather than validating an in-memory copy.
        report = validate(raw_path, expected_episodes=args.episodes)
        report["episode_errors"] = [s for s in summaries if s["error"]]
        report["valid"] = report["valid"] and not report["episode_errors"]
        save_json(run_dir / "validation.json", report)
        logger.info("VALIDATION valid=%s episodes=%d successes=%d failures=%d steps=%d",
                    report["valid"], report["episode_count"], report["success_count"],
                    report["failure_count"], report["transition_count"])
        return 0 if report["valid"] else 1
    except Exception:
        logger.exception("Run failed; all existing raw output is retained")
        return 1
    finally:
        for current in list(logger.handlers):
            current.close()
            logger.removeHandler(current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--split", choices=["train", "valid_seen", "valid_unseen", "valid_train"], default="train")
    parser.add_argument("--task-type", default="pick_and_place_simple",
                        choices=["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
                                 "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/alfworld")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or not 0 <= args.seed < 2**32 - args.episodes:
        parser.error("episodes/max-steps must be positive; seeds must fit NumPy uint32")
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
