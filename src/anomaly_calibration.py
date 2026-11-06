"""Train-only non-risk reference distribution; no inference outcomes or validation fit."""
import math
from pathlib import Path
import statistics

from src.lstm_training_data import check_group_leakage, load_windows
from src.timeseries_features import digest, read_json

STD_FLOOR = 1e-6
TIE_BREAK = ("E", "C", "S")  # Used only for exactly equal standardized scores.


def calibrate_training_observations(windows_path, split_path):
    windows_path, split_path = Path(windows_path).resolve(), Path(split_path).resolve()
    windows_hash, split_hash = digest(windows_path), digest(split_path)
    split = read_json(split_path)
    if split.get("input_sha256") != windows_hash:
        raise ValueError("Calibration CSV SHA256 does not match checkpoint split")
    samples = load_windows(windows_path)
    train, validation = split["train_indices"], split["validation_indices"]
    leakage = check_group_leakage(samples, train, validation)
    observations = {}
    for index in train:
        sample = samples[index]
        for offset, state in enumerate(sample.x):
            key = (*sample.group, sample.start_step + offset)
            value = tuple(state)
            if key in observations and observations[key][0] != value:
                raise ValueError("Conflicting E/C/S for the same training observation")
            observations.setdefault(key, (value, set()))[1].add(sample.label)
    # Window labels are weak labels. Reject observations also covered by label-1
    # windows instead of accidentally admitting risky transition overlap.
    population = [value for value, labels in observations.values() if labels == {0}]
    if len(population) < 2:
        raise ValueError("Need at least two unique label-0-only training observations for calibration; no validation/label-1 fallback")
    stats, effective = {}, {}
    for column, feature in enumerate(("E", "C", "S")):
        values = [state[column] for state in population]
        mean, std = statistics.mean(values), statistics.pstdev(values)
        if not math.isfinite(mean) or not math.isfinite(std):
            raise ValueError("Nonfinite calibration statistics")
        stats.update({f"{feature}_mean": mean, f"{feature}_std": std})
        effective[feature] = max(std, STD_FLOOR)
    if digest(windows_path) != windows_hash or digest(split_path) != split_hash:
        raise ValueError("Calibration sources changed during reading")
    return {"source": "LSTM_recorded_TRAIN_partition_label_0_only_unique_observations",
            "population_policy": "Deduplicate all TRAIN observations; keep only steps covered exclusively by label-0 windows; exclude mixed-label overlap; no fallback",
            "statistics": stats, "std_definition": "population standard deviation (ddof=0)",
            "std_floor": STD_FLOOR, "effective_std": effective,
            "sample_count": len(population), "unique_training_step_count": len(observations),
            "mixed_label_steps_excluded": sum(labels == {0, 1} for _, labels in observations.values()),
            "training_window_count": len(train), "validation_samples_used": 0, "leakage_check": leakage,
            "windows_path": str(windows_path), "input_sha256": windows_hash,
            "split_path": str(split_path), "split_sha256": split_hash}


def standardized_diagnosis(features, calibration):
    if calibration is None:
        raise ValueError("Proposed diagnosis requires train-only calibration; no absolute-threshold fallback")
    stats = calibration["statistics"]
    floor = calibration["std_floor"]
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("Calibration std_floor must be finite positive")
    signed, scores = {}, {}
    for feature in ("E", "C", "S"):
        mean, std, value = stats[f"{feature}_mean"], stats[f"{feature}_std"], features[feature]
        if any(not math.isfinite(number) for number in (mean, std, value)) or std < 0:
            raise ValueError("Invalid calibration or candidate E/C/S")
        z = ((mean - value) if feature == "S" else (value - mean)) / max(std, floor)
        if not math.isfinite(z):
            raise ValueError("Nonfinite standardized anomaly")
        signed[f"diagnostic_{feature}_z"] = z
        scores[feature] = max(0., z)
    winner = max(TIE_BREAK, key=scores.__getitem__)
    return {**signed, "anomaly_scores": scores,
            "selected_anomaly": winner if scores[winner] > 0 else None,
            "selected_anomaly_score": scores[winner]}


def diagnostic_log_fields(calibration=None):
    fields = {f"diagnostic_{feature}_z": None for feature in ("E", "C", "S")}
    fields.update(selected_anomaly=None, selected_anomaly_score=None)
    fields.update({f"calibration_{feature}_{stat}": calibration["statistics"][f"{feature}_{stat}"] if calibration else None
                   for feature in ("E", "C", "S") for stat in ("mean", "std")})
    return fields
