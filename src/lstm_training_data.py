"""Validated five-step CSV loading and deterministic group-aware splitting."""
import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random


class TrainingDataError(ValueError):
    pass


@dataclass(frozen=True)
class WindowSample:
    x: list
    label: int
    run_dir: str
    episode_id: str
    condition: str
    start_step: int
    end_step: int
    source_line: int
    metadata: dict

    @property
    def group(self):
        return self.run_dir, self.episode_id, self.condition

    @property
    def split_group(self):
        directory = Path(self.run_dir)
        pair = directory.parent if directory.name in ("no_fault", "fault") else directory
        return str(pair), self.episode_id


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_windows(path):
    samples, identities = [], set()
    required = ["run_dir", "episode_id", "condition", "start_step", "end_step", "label"]
    required += [f"{feature}_{step}" for feature in ("E", "C", "S") for step in range(1, 6)]
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise TrainingDataError("Input must contain metadata, label and E_1..E_5/C_1..C_5/S_1..S_5")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise TrainingDataError("Duplicate CSV columns")
        if any(name in reader.fieldnames for name in ("E_6", "C_6", "S_6")):
            raise TrainingDataError("This fixed model expects five-step windows; regenerate with --window-size 5")
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Malformed CSV width")
                if any(not row[name].strip() for name in ("run_dir", "episode_id", "condition")):
                    raise ValueError("Missing group metadata")
                if row["condition"] not in ("no_fault", "fault") or row["label"] not in ("0", "1"):
                    raise ValueError("Invalid condition or non-binary label")
                if not row["start_step"].isdigit() or not row["end_step"].isdigit():
                    raise ValueError("Step endpoints must be integers")
                start, end = int(row["start_step"]), int(row["end_step"])
                if start < 1 or end - start != 4:
                    raise ValueError("Window endpoints must describe five consecutive steps")
                # Explicit name-based time-major order. NEVER reshape feature-major CSV values.
                x = [[float(row[f"{feature}_{step}"]) for feature in ("E", "C", "S")] for step in range(1, 6)]
                if not all(math.isfinite(value) for state in x for value in state):
                    raise ValueError("NaN/inf features are forbidden")
                sample = WindowSample(x, int(row["label"]), row["run_dir"], row["episode_id"], row["condition"], start, end, line, dict(row))
                identity = (*sample.group, start, end)
                if identity in identities:
                    raise ValueError("Duplicate sample identity")
                identities.add(identity)
                samples.append(sample)
            except (ValueError, TypeError, KeyError) as exc:
                raise TrainingDataError(f"{path}:{line}: {exc}") from None
    if not samples:
        raise TrainingDataError("No samples in windows.csv; collect longer episodes and generate nonempty windows first")
    return samples


def check_group_leakage(samples, train_indices, validation_indices):
    train, validation = set(train_indices), set(validation_indices)
    if len(train) != len(train_indices) or len(validation) != len(validation_indices):
        raise TrainingDataError("Duplicate sample indices inside a partition")
    if not train or not validation or train | validation != set(range(len(samples))) or train & validation:
        raise TrainingDataError("Split must assign every sample exactly once to nonempty disjoint partitions")
    train_groups = {samples[index].group for index in train}
    validation_groups = {samples[index].group for index in validation}
    train_pairs = {samples[index].split_group for index in train}
    validation_pairs = {samples[index].split_group for index in validation}
    report = {"sample_overlap_count": len(train & validation), "group_overlap_count": len(train_groups & validation_groups),
              "paired_group_overlap_count": len(train_pairs & validation_pairs), "all_samples_assigned_once": True}
    if report["group_overlap_count"] or report["paired_group_overlap_count"]:
        raise TrainingDataError("Data leakage: windows from the same group/pair appear in both partitions")
    return report


def group_split(samples, validation_fraction=.2, seed=42):
    if not 0 < validation_fraction < 1:
        raise TrainingDataError("validation_fraction must be between 0 and 1")
    groups = sorted({sample.split_group for sample in samples})
    if len(groups) < 2:
        raise TrainingDataError("Need at least two independent run/episode groups; a single no_fault/fault pair cannot be split without leakage")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, math.ceil(len(groups) * validation_fraction)))
    validation_groups = set(groups[:count])
    train = [index for index, sample in enumerate(samples) if sample.split_group not in validation_groups]
    validation = [index for index, sample in enumerate(samples) if sample.split_group in validation_groups]
    check = check_group_leakage(samples, train, validation)
    def descriptions(indices):
        output = []
        for group in sorted({samples[index].group for index in indices}):
            members = [index for index in indices if samples[index].group == group]
            output.append({"run_dir": group[0], "episode_id": group[1], "condition": group[2],
                           "split_group": list(samples[members[0]].split_group), "sample_indices": members,
                           "source_csv_lines": [samples[index].source_line for index in members]})
        return output
    return {"seed": seed, "requested_train_fraction": 1 - validation_fraction, "requested_validation_fraction": validation_fraction,
            "base_group_key": ["run_dir", "episode_id", "condition"],
            "effective_split_key": ["canonical_pair_run_dir", "episode_id"], "paired_conditions_kept_together": True,
            "train_indices": train, "validation_indices": validation, "train_groups": descriptions(train),
            "validation_groups": descriptions(validation), "train_sample_count": len(train), "validation_sample_count": len(validation),
            "effective_group_count": len(groups), "validation_effective_group_count": count, "leakage_check": check,
            "split_method": "single deterministic shuffle of whole groups, no window-random split or split search"}


def training_pos_weight(samples, train_indices):
    positives = sum(samples[index].label for index in train_indices)
    negatives = len(train_indices) - positives
    if not positives or not negatives:
        raise TrainingDataError("Training partition must contain both labels to compute a meaningful pos_weight. Collect more independent labeled groups; no split search/window leakage fallback is performed")
    return negatives / positives, {"0": negatives, "1": positives}
