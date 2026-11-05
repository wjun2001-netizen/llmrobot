"""Read-only E/C/S extraction. No causal language model or ALFWorld environment is loaded."""
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLUMNS = ["run_dir", "episode_id", "condition", "step", "E", "C", "S", "prompt_token_count", "max_context_tokens",
           "fault_type", "fault_severity", "fault_active", "injection_step", "success", "outcome",
           "prompt_count_source", "tokenizer_name", "tokenizer_revision", "similarity_model", "data_kind"]


class FeatureError(ValueError):
    pass


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FeatureError(f"Cannot read JSON: {path} ({type(exc).__name__})") from None


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def discover_trajectories(root):
    root = Path(root)
    if root.is_file() and root.name == "trajectories.jsonl":
        return [root.resolve()]
    if not root.is_dir():
        raise FeatureError(f"Runs directory does not exist: {root}")
    def order(path):
        return (str(path.parent.parent), {"no_fault": 0, "fault": 1}.get(path.parent.name, 2), str(path))
    paths = sorted((p.resolve() for p in root.rglob("trajectories.jsonl")), key=order)
    if not paths:
        raise FeatureError(f"No trajectories.jsonl files under {root}; no models loaded or example research data invented")
    return paths


def load_metadata(directory):
    """Nearest condition settings take precedence; ancestor experiment supplies prompt/revision."""
    metadata = {"local_llm": {}, "metadata_files": []}
    directory = Path(directory).resolve()
    for folder in (directory, *directory.parents):
        path = folder / "config.json"
        if path.is_file():
            config = read_json(path)
            if not isinstance(config, dict):
                raise FeatureError(f"Expected configuration object: {path}")
            metadata["metadata_files"].append(path)
            local_config = config.get("local_llm") or {}
            if not isinstance(local_config, dict):
                raise FeatureError(f"Expected local_llm configuration object: {path}")
            for key, value in local_config.items():
                metadata["local_llm"].setdefault(key, value)
            for key in ("system_prompt", "action_selection_format", "source_sha256", "condition"):
                if key in config:
                    metadata.setdefault(key, config[key])
        info = folder / "model_info.json"
        if info.is_file():
            values = read_json(info)
            metadata["metadata_files"].append(info)
            if isinstance(values, dict):
                metadata.setdefault("resolved_revision", values.get("resolved_revision"))
        # Do not inspect unrelated ancestors after the experiment's recorded source manifest.
        if metadata.get("source_sha256"):
            break
    return metadata


def _positive_int(value, field):
    if type(value) is not int or value <= 0:
        raise FeatureError(f"{field} must be a positive integer")
    return value


class PromptCounter:
    def __init__(self, *, default_max_context=4096, local_files_only=False, tokenizer_revision=None,
                 allow_unverified_reconstruction=False, tokenizer_loader=None):
        self.default_max_context = _positive_int(default_max_context, "default max_context_tokens")
        self.local_files_only = local_files_only
        self.tokenizer_revision = tokenizer_revision
        self.allow_unverified = allow_unverified_reconstruction
        self.loader, self.cache = tokenizer_loader, {}
        formats = read_json(ROOT / "configs/prompt_formats.json")["formats"]
        from local_llm_agent import SYSTEM_PROMPT
        current = {"source_sha256": digest(ROOT / "local_llm_agent.py"), "format": "action_id_v1", "system_prompt": SYSTEM_PROMPT}
        self.formats = {entry["source_sha256"]: entry for entry in [*formats, current]}

    def count(self, row, metadata):
        config = metadata["local_llm"]
        maximum = row.get("max_context_tokens", config.get("max_context_tokens", self.default_max_context))
        maximum = _positive_int(maximum, "max_context_tokens")
        if "max_context_tokens" in row and "max_context_tokens" in config and row["max_context_tokens"] != config["max_context_tokens"]:
            raise FeatureError("Recorded max_context_tokens conflicts with run configuration")
        if "prompt_token_count" in row:
            count = _positive_int(row["prompt_token_count"], "prompt_token_count")
            return {"prompt_token_count": count, "max_context_tokens": maximum, "prompt_count_source": "recorded_at_generation",
                    "tokenizer_name": config.get("model_name", row.get("model_name")),
                    "tokenizer_revision": row.get("resolved_revision") or metadata.get("resolved_revision") or config.get("revision")}
        source_hash = (metadata.get("source_sha256") or {}).get("local_llm_agent.py")
        spec = self.formats.get(source_hash)
        if spec is None and not self.allow_unverified:
            raise FeatureError("Prompt not recorded and generating source hash is unknown; refuse to guess historical prompt format")
        format_name = row.get("action_selection_format", metadata.get("action_selection_format", spec["format"] if spec else None))
        if format_name != "action_id_v1":
            raise FeatureError(f"Unsupported historical prompt format: {format_name}")
        system_prompt = metadata.get("system_prompt", spec["system_prompt"] if spec else None)
        if not isinstance(system_prompt, str) or not system_prompt:
            raise FeatureError("Historical system prompt is unavailable")
        if spec is not None and system_prompt != spec["system_prompt"] and not self.allow_unverified:
            raise FeatureError("Recorded system prompt conflicts with the known generating source")
        name = config.get("model_name") or row.get("model_name")
        if not isinstance(name, str) or not name:
            raise FeatureError("Cannot reconstruct prompt without tokenizer/model name")
        recorded_revision = row.get("resolved_revision") or metadata.get("resolved_revision") or config.get("revision")
        if self.tokenizer_revision and recorded_revision and self.tokenizer_revision != recorded_revision and not self.allow_unverified:
            raise FeatureError("Tokenizer override conflicts with the original recorded revision")
        revision = self.tokenizer_revision or recorded_revision
        if revision is None and not Path(name).is_dir():
            raise FeatureError("No tokenizer revision recorded: supply the original --tokenizer-revision or use observed prompt counts; do not guess a moving HF revision")
        key = (name, revision)
        if key not in self.cache:
            if self.loader is None:
                from transformers import AutoTokenizer
                loader = AutoTokenizer.from_pretrained
            else:
                loader = self.loader
            self.cache[key] = loader(name, revision=revision, trust_remote_code=False, local_files_only=self.local_files_only)
        tokenizer = self.cache[key]
        for field in ("previous_actions", "admissible_actions"):
            if not isinstance(row.get(field), list) or not all(isinstance(value, str) for value in row[field]):
                raise FeatureError(f"Cannot reconstruct prompt: invalid {field}")
        inputs = {"initial_goal": row["agent_goal"] if "agent_goal" in row else row.get("goal"),
                  "current_observation": row["agent_observation"] if "agent_observation" in row else row.get("observation"),
                  "previous_actions": list(row["previous_actions"]),
                  "admissible_actions": [{"action_id": index, "action": action} for index, action in enumerate(row["admissible_actions"])]}
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(inputs, ensure_ascii=False, allow_nan=False)}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=config.get("enable_thinking", False))
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        ids = encoded["input_ids"]
        if hasattr(ids, "shape"):
            if len(ids.shape) != 2 or ids.shape[0] != 1:
                raise FeatureError("Tokenizer returned a non-single-sequence prompt")
            count = int(ids.shape[-1])
        elif isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
            count = len(ids[0])
        else:
            raise FeatureError("Tokenizer must return one encoded prompt sequence")
        return {"prompt_token_count": _positive_int(count, "prompt_token_count"), "max_context_tokens": maximum,
                "prompt_count_source": "reconstructed_verified_source" if spec is not None and not self.allow_unverified else "reconstructed_unverified_override",
                "tokenizer_name": name, "tokenizer_revision": revision}


def cosine_similarity(first, second):
    import numpy as np
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.ndim != 1 or second.shape != first.shape or not first.size or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise FeatureError("Embedding vectors are missing, non-finite, or incompatible")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not denominator or not math.isfinite(denominator):
        raise FeatureError("Cannot compute cosine similarity for a zero/non-finite norm embedding")
    value = float(np.dot(first, second) / denominator)
    if not math.isfinite(value) or not -1.000001 <= value <= 1.000001:
        raise FeatureError("Non-finite or out-of-range cosine similarity")
    return value  # No absolute value, clipping, exact-text matching, or Hamming distance.


class SentenceSimilarity:
    def __init__(self, model_name=DEFAULT_SIMILARITY_MODEL, *, revision=None, local_files_only=False, batch_size=32, model=None):
        self.model_name, self.revision, self.local_files_only = model_name, revision, local_files_only
        self.batch_size, self.model = batch_size, model
        self.resolved_revision = None

    def score_pairs(self, pairs):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, revision=self.revision, device="cpu", local_files_only=self.local_files_only, trust_remote_code=False)
        if callable(getattr(self.model, "_first_module", None)):
            transformer = self.model._first_module()
            self.resolved_revision = getattr(getattr(getattr(transformer, "auto_model", None), "config", None), "_commit_hash", None)
        unique = list(dict.fromkeys(text for pair in pairs for text in pair))
        vectors = self.model.encode(unique, batch_size=self.batch_size, convert_to_numpy=True,
                                    normalize_embeddings=False, show_progress_bar=False)
        if len(vectors) != len(unique):
            raise FeatureError("Embedding backend returned the wrong number of vectors")
        lookup = dict(zip(unique, vectors))
        return [cosine_similarity(lookup[goal], lookup[plan]) for goal, plan in pairs]


@dataclass
class PreparedRow:
    values: dict
    initial_goal: str
    plan: str


def _prepare_file(path, counter, skip_invalid, errors):
    metadata = load_metadata(path.parent)
    summary_path = path.parent / "episodes.json"
    summaries = read_json(summary_path)
    if not isinstance(summaries, list):
        raise FeatureError(f"Expected episode summary array: {summary_path}")
    outcomes = {}
    for summary in summaries:
        if not isinstance(summary, dict) or not isinstance(summary.get("episode_id"), str) or not summary["episode_id"]:
            raise FeatureError(f"Invalid episode summary: {summary_path}")
        key = summary.get("episode_id")
        if key in outcomes:
            raise FeatureError(f"Duplicate episode summary in {summary_path}")
        outcomes[key] = summary.get("outcome")
    initial_goals, last_steps, result = {}, {}, []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise FeatureError("Expected a JSON object")
                episode_id, step = row.get("episode_id"), row.get("step")
                if not isinstance(episode_id, str) or not episode_id:
                    raise FeatureError("Missing episode_id")
                _positive_int(step, "step")
                if step <= last_steps.get(episode_id, 0):
                    raise FeatureError("Duplicate/out-of-order step: source order is never silently sorted")
                last_steps[episode_id] = step
                goal = row["original_goal"] if "original_goal" in row else row.get("goal")
                initial_goals.setdefault(episode_id, goal)
                initial = initial_goals[episode_id]
                if not isinstance(initial, str) or not initial.strip() or initial != goal:
                    raise FeatureError("Missing/inconsistent original episode initial goal")
                plan = row.get("plan")
                if not isinstance(plan, str) or not plan.strip():
                    raise FeatureError("Missing/empty plan; no synthetic similarity value is assigned")
                entropy = row.get("mean_entropy")
                if type(entropy) not in (int, float) or not math.isfinite(entropy) or entropy < 0:
                    raise FeatureError("mean_entropy must be finite, nonnegative nats")
                if row.get("entropy_unit", "nats") != "nats":
                    raise FeatureError("Entropy unit is not nats")
                condition = metadata.get("condition")
                if condition not in ("fault", "no_fault"):
                    condition = path.parent.name if path.parent.name in ("fault", "no_fault") else ("fault" if row.get("fault_type") not in (None, "no_fault") else "no_fault")
                fault_type = row.get("fault_type", "no_fault" if condition == "no_fault" else None)
                severity = row.get("fault_severity")
                active = row.get("fault_active", False if condition == "no_fault" else None)
                injection = row.get("injection_step")
                if type(active) is not bool:
                    raise FeatureError("Missing/invalid fault_active")
                if condition == "no_fault":
                    if fault_type != "no_fault" or active or severity is not None or injection is not None:
                        raise FeatureError("Conflicting no_fault metadata")
                elif fault_type not in ("goal_drift", "observation_loss", "context_overload") or severity not in ("low", "high") or type(injection) is not int or injection < 2:
                    raise FeatureError("Missing/invalid fault metadata")
                if condition == "fault":
                    if any(field not in row for field in ("original_goal", "original_observation", "agent_goal", "agent_observation")):
                        raise FeatureError("Fault input provenance missing; refuse to reconstruct using unperturbed inputs")
                    if active != (step >= injection):
                        raise FeatureError("fault_active conflicts with injection_step")
                if type(row.get("success")) is not bool:
                    raise FeatureError("Missing/invalid per-step success")
                outcome = outcomes.get(episode_id)
                if not isinstance(outcome, str) or not outcome:
                    raise FeatureError("Missing final episode outcome")
                count = counter.count(row, metadata)
                value = {"run_dir": str(path.parent), "episode_id": episode_id, "condition": condition, "step": step,
                         "E": entropy, "C": count["prompt_token_count"] / count["max_context_tokens"], "S": None,
                         **count, "fault_type": fault_type, "fault_severity": severity, "fault_active": active,
                         "injection_step": injection, "success": row["success"], "outcome": outcome}
                result.append(PreparedRow(value, initial, plan))
            except (FeatureError, ValueError, TypeError, KeyError) as exc:
                issue = {"file": str(path), "line": number, "reason": str(exc)}
                errors.append(issue)
                if not skip_invalid:
                    raise FeatureError(f"{path}:{number}: {exc}") from None
    return result, metadata["metadata_files"] + [summary_path]


def extract(runs_dir, output, *, counter=None, similarity=None, skip_invalid=False, overwrite=False, data_kind="trajectory"):
    output = Path(output).resolve()
    report_path = output.with_suffix(output.suffix + ".metadata.json").resolve()
    if not overwrite and (output.exists() or report_path.exists()):
        raise FeatureError(f"Output already exists: {output}; use a new path or explicit --overwrite")
    paths = discover_trajectories(runs_dir)
    # Protect source files even if a caller supplies a misleading output filename.
    if output in paths or report_path in paths or any(output == path.parent / "episodes.json" or output == path.parent / "config.json" for path in paths):
        raise FeatureError("Output must not replace trajectory/metadata inputs")
    counter, similarity = counter or PromptCounter(), similarity or SentenceSimilarity()
    before = {str(path): digest(path) for path in paths}
    prepared, errors, metadata_files = [], [], set()
    for path in paths:
        try:
            values, configs = _prepare_file(path, counter, skip_invalid, errors)
            prepared.extend(values)
            metadata_files.update(configs)
        except FeatureError as exc:
            if not skip_invalid:
                raise
            errors.append({"file": str(path), "line": None, "reason": str(exc)})
    if not prepared:
        raise FeatureError("No valid feature rows; no CSV or fake rows written")
    metadata_before = {str(path.resolve()): digest(path) for path in metadata_files}
    if any(str(candidate) in metadata_before for candidate in (output, report_path)):
        raise FeatureError("Output must not replace source run metadata")
    scores = similarity.score_pairs([(row.initial_goal, row.plan) for row in prepared])
    if len(scores) != len(prepared):
        raise FeatureError("Similarity backend returned incorrect row count")
    rows = []
    for item, score in zip(prepared, scores):
        if type(score) not in (int, float) or not math.isfinite(score) or not -1.000001 <= score <= 1.000001:
            raise FeatureError("Similarity backend returned a missing/non-finite cosine score")
        item.values.update(S=score, similarity_model=similarity.model_name, data_kind=data_kind)
        rows.append(item.values)
    if any(digest(Path(path)) != expected for path, expected in before.items()):
        raise FeatureError("A trajectory changed during extraction; no CSV published")
    if any(digest(Path(path)) != expected for path, expected in metadata_before.items()):
        raise FeatureError("Source metadata changed during extraction; no CSV published")
    report = {"rows": len(rows), "source_files": before, "source_metadata_files": metadata_before,
              "validation_errors": errors, "skipped_invalid": skip_invalid, "source_trajectories_unchanged": True,
              "data_kind": data_kind, "E_unit": "nats", "C_definition": "actual input prompt tokens / configured max_context_tokens",
              "S_definition": "sentence-transformer cosine(original episode initial goal, step plan), unchanged cosine",
              "similarity_model": similarity.model_name, "similarity_revision": similarity.revision,
              "similarity_resolved_revision": getattr(similarity, "resolved_revision", None),
              "extractor_source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (ROOT / "extract_timeseries.py", ROOT / "src/timeseries_features.py", ROOT / "configs/prompt_formats.json")},
              "similarity_max_seq_length": getattr(getattr(similarity, "model", None), "max_seq_length", None),
              "nullable_columns": ["fault_severity", "injection_step", "tokenizer_revision"],
              "prompt_count_sources": {name: sum(row["prompt_count_source"] == name for row in rows) for name in sorted({row["prompt_count_source"] for row in rows})}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        # Exclusive publication by default, replacing derived output only when explicitly requested.
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
            temporary.unlink()
        mode = "w" if overwrite else "x"
        with report_path.open(mode, encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return report
