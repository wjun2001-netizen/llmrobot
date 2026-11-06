"""Run four simple input-context intervention comparisons locally (NVIDIA Qwen)."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys

from collect_fault import baseline_steps_from_run, game_identity, resolve_game_timing
from collect_local import check_imports
from local_llm_agent import LocalLLMAgent, LocalLLMConfig, SYSTEM_PROMPT
from src.fault_injection import FAULTS, FaultConfig
from src.intervention import DEFAULT_CHECKPOINT, MODES, InterventionConfig, LSTMRiskDetector, training_entropy_threshold
from src.intervention_runtime import execute_experiment, safe_output_root
from src.timeseries_features import DEFAULT_SIMILARITY_MODEL, SentenceSimilarity, digest, read_json
from src.anomaly_calibration import calibrate_training_observations

ROOT = Path(__file__).resolve().parent
DEFAULT_GAME = ROOT / "data/alfworld/json_2.1.1/train/pick_and_place_simple-Box-None-Sofa-205/trial_T20190907_214830_497445/game.tw-pddl"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["local"], default="local")
    parser.add_argument("--mode", choices=MODES, default="proposed")
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--game", type=Path, default=DEFAULT_GAME)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/alfworld")
    parser.add_argument("--fault", choices=FAULTS, default="context_overload")
    parser.add_argument("--severity", choices=["low", "high"], default="high")
    parser.add_argument("--injection-step", type=int)
    parser.add_argument("--baseline-steps", type=int)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--reference-steps", type=int)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lstm-checkpoint", type=Path, default=ROOT / DEFAULT_CHECKPOINT)
    parser.add_argument("--lstm-risk-threshold", type=float, default=.5)
    parser.add_argument("--min-anomaly-z", type=float, default=1.)
    parser.add_argument("--calibration-windows", type=Path, help="Proposed TRAIN calibration CSV; default --entropy-training-windows. Split is always beside checkpoint")
    parser.add_argument("--similarity-threshold", type=float, default=.3)
    parser.add_argument("--context-threshold", type=float, default=.8)
    parser.add_argument("--entropy-threshold", type=float)
    parser.add_argument("--entropy-threshold-basis", default="CLI value declared before outcomes; no post-hoc tuning")
    parser.add_argument("--entropy-training-windows", type=Path, default=ROOT / "features/lstm/windows.csv")
    parser.add_argument("--entropy-split", type=Path, help="Default: split.json beside LSTM checkpoint; used only for train-only entropy calibration")
    parser.add_argument("--goal-period", type=int, default=5)
    parser.add_argument("--history-keep", type=int, default=3)
    parser.add_argument("--intervention-cooldown", type=int, default=5,
                        help="Proposed only: suppress interventions for N subsequent environment transitions after completed recovery (0 disables)")
    parser.add_argument("--similarity-revision")
    parser.add_argument("--embedding-local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs_intervention")
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
    return parser


def prepare(args):
    """Read-only validation/calibration before any Qwen model download/load."""
    if not 0 <= args.seed < 2**32 or args.max_steps < 2:
        raise ValueError("seed must fit uint32 and max_steps >= 2")
    safe_output_root(args.output_dir)
    game = args.game.resolve()
    if not game.is_file() or game.name != "game.tw-pddl":
        raise ValueError("Specify an existing game.tw-pddl")
    identity = game_identity(game, args.data_dir)
    if args.baseline_steps is not None and args.baseline_run is not None:
        raise ValueError("Choose baseline-steps or baseline-run")
    length = args.baseline_steps if args.baseline_steps is not None else baseline_steps_from_run(args.baseline_run, identity)
    user_reference = False
    if length is None and args.reference_steps is None and game == DEFAULT_GAME.resolve():
        length, user_reference = 16, True
    timing = resolve_game_timing(args, game, length)
    if user_reference and args.injection_step is None:
        timing["source"] = "user_reported_Box_Sofa_16_step_baseline"
    fault = FaultConfig(args.fault, args.severity, timing["injection_step"])
    modes = MODES if args.compare_all else (args.mode,)
    threshold, entropy_basis = args.entropy_threshold, {"source": "unused_in_selected_modes"}
    assets = {str(game): digest(game)}
    if threshold is not None:
        entropy_basis = {"source": "explicit_CLI", "rationale": args.entropy_threshold_basis, "entropy_unit": "nats"}
    elif "entropy_only" in modes:
        split_path = args.entropy_split or args.lstm_checkpoint.parent / "split.json"
        threshold, entropy_basis = training_entropy_threshold(args.entropy_training_windows, split_path)
        assets.update({entropy_basis["windows_path"]: entropy_basis["windows_sha256"], entropy_basis["split_path"]: entropy_basis["split_sha256"]})
    detector = LSTMRiskDetector(args.lstm_checkpoint) if "proposed" in modes else None
    calibration = None
    if detector:
        calibration = calibrate_training_observations(args.calibration_windows or args.entropy_training_windows,
                                                      args.lstm_checkpoint.parent / "split.json")
        if calibration["input_sha256"] != detector.training_input_sha256:
            raise ValueError("Diagnosis calibration must match the checkpoint's training CSV SHA256")
        assets.update({calibration["windows_path"]: calibration["input_sha256"], calibration["split_path"]: calibration["split_sha256"]})
    configs = [InterventionConfig(mode=mode, lstm_risk_threshold=args.lstm_risk_threshold,
                                  similarity_threshold=args.similarity_threshold, context_threshold=args.context_threshold,
                                  entropy_threshold=threshold, period=args.goal_period, history_keep=args.history_keep,
                                  min_anomaly_z=args.min_anomaly_z,
                                  intervention_cooldown=args.intervention_cooldown,
                                  calibration=calibration if mode == "proposed" else None) for mode in modes]
    if detector:
        assets[str(detector.checkpoint)] = detector.sha256
        if entropy_basis["source"] == "recorded_training_partition_only" and detector.training_input_sha256 != entropy_basis["windows_sha256"]:
            raise ValueError("Detector checkpoint and entropy calibration must reference the same training CSV")
    split_path = args.lstm_checkpoint.parent / "split.json"
    overlap = None
    if detector and split_path.is_file():
        split = read_json(split_path)
        assets[str(split_path.resolve())] = digest(split_path)
        if split.get("input_sha256") != detector.training_input_sha256:
            raise ValueError("Detector checkpoint and adjacent split metadata disagree")
        overlap = any(group["episode_id"] == identity for group in split["train_groups"])
    local_config = LocalLLMConfig.from_file(args.config, model_name=args.model, revision=args.model_revision,
                                            load_in_4bit=args.load_in_4bit, save_logits=args.save_logits,
                                            compute_dtype=args.compute_dtype, device=args.device,
                                            max_new_tokens=args.max_new_tokens, max_context_tokens=args.max_context_tokens)
    sources = [Path(__file__), ROOT / "src/intervention.py", ROOT / "src/intervention_runtime.py", ROOT / "src/anomaly_calibration.py",
               ROOT / "local_llm_agent.py", ROOT / "collect_local.py", ROOT / "src/fault_injection.py",
               ROOT / "src/timeseries_features.py", ROOT / "src/lstm_classifier.py"]
    metadata = {"data_kind": "local_research_experiment", "system_prompt": SYSTEM_PROMPT, "timing": timing,
                "threshold_basis": {"entropy": {**entropy_basis, "value": threshold},
                                    "similarity": {"value": args.similarity_threshold, "default": .3,
                                                   "rationale": "A priori fixed cosine cutoff or CLI override, not fitted to Box-Sofa results"},
                                    "context": {"value": args.context_threshold, "default": .8,
                                                "rationale": "A priori fixed prompt budget fraction or CLI override, not fitted to outcomes"},
                                    "risk": {"value": args.lstm_risk_threshold, "default": .5,
                                             "rationale": "Requested sigmoid cutoff or explicit CLI override"},
                                    "min_anomaly_z": {"value": args.min_anomaly_z, "default": 1.,
                                                      "rationale": "Predeclared one-sided standardized anomaly gate; not tuned to experiment outcomes"},
                                    "absolute_thresholds_used_by_proposed": False,
                                    "frozen_before_first_condition": True},
                "feature_semantics": {"E": "mean_entropy in nats, unchanged full-vocabulary generation",
                                      "C": "actual generation prompt_token_count / max_context_tokens",
                                      "S": "original initial goal vs generated plan, SentenceSimilarity cosine",
                                      "final_features": "E/C/S correspond to final selected decision",
                                      "detector_features": "detection_features and lstm_input_window: four previous final decisions + current candidate",
                                      "lstm_risk": "pre-intervention candidate risk, not re-evaluated for a second trigger"},
                "diagnostic_calibration": calibration,
                "diagnosis": "argmax of clamped one-sided standardized candidate anomalies after risk gate",
                "exact_tie_break": ["E", "C", "S"], "context_cleaning": "Trim action history and refresh observation from the same current true feedback; fault reapplies next step",
                "intervention_cooldown": {"transitions": args.intervention_cooldown, "mode": "proposed_only",
                                          "policy": "Suppress next N transitions after completed intervention; still compute candidate features/risk/diagnosis"},
                "intervention_timing": "After candidate generation, before env.step; at most one context intervention and one re-generation",
                "lstm_checkpoint": str(detector.checkpoint) if detector else None,
                "lstm_checkpoint_sha256": detector.sha256 if detector else None,
                "detector_training_episode_overlap": overlap,
                "similarity_model": DEFAULT_SIMILARITY_MODEL, "similarity_revision": args.similarity_revision,
                "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sources},
                "protected_input_sha256": assets,
                "invariants": "No environment rollback/reset within episode; true goal, action mapping, fault injector and generate/entropy code unchanged"}
    return game, identity, fault, configs, detector, local_config, metadata


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check_imports:
        report = check_imports()
        import sentence_transformers
        from src.lstm_classifier import FailureRiskLSTM
        report.update(sentence_transformers=sentence_transformers.__version__, lstm_class=FailureRiskLSTM.__name__)
        print(json.dumps(report, indent=2))
        return 0
    try:
        game, identity, fault, configs, detector, local_config, metadata = prepare(args)
        import torch
        torch.manual_seed(args.seed)
        similarity = SentenceSimilarity(revision=args.similarity_revision, local_files_only=args.embedding_local_files_only)
        # Load the small embedding model before resetting the per-condition model seed.
        similarity.score_pairs([("initialization", "initialization")])
        metadata["similarity_resolved_revision"] = similarity.resolved_revision
        metadata["similarity_max_seq_length"] = getattr(similarity.model, "max_seq_length", None)
        agent = LocalLLMAgent(local_config).load()
        metadata["local_model_resolved_revision"] = getattr(agent.model.config, "_commit_hash", None)
        metadata["packages"] = {}
        for name in ("torch", "transformers", "sentence-transformers", "alfworld", "textworld", "bitsandbytes"):
            try:
                metadata["packages"][name] = version(name)
            except PackageNotFoundError:
                metadata["packages"][name] = None
        os.environ["ALFWORLD_DATA"] = str(args.data_dir.resolve())
        run, summaries, complete = execute_experiment(game=game, episode_id=identity, seed=args.seed,
            max_steps=args.max_steps, agent=agent, fault=fault, interventions=configs, similarity=similarity,
            output_root=args.output_dir, metadata=metadata, detector=detector, seed_runtime=torch.manual_seed)
        print(f"RUN_DIR={run}")
        for item in summaries:
            print(f"{item['mode']}: success={item['success']} steps={item['steps']} outcome={item['outcome']} interventions={item['intervention_count']}")
        return 0 if complete else 1
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
        print(f"Intervention experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
