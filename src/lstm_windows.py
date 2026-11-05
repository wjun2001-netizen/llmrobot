"""Pure-stdlib sliding E/C/S windows. No model training, splitting, or source rewriting."""
from collections import Counter, OrderedDict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


class WindowError(ValueError):
    pass


REQUIRED = ("run_dir", "episode_id", "condition", "step", "E", "C", "S", "fault_type", "fault_severity", "injection_step", "outcome")
FAILURES = ("step_limit", "task_failure")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def positive_integer(text, name):
    if not isinstance(text, str) or not re.fullmatch(r"[0-9]+", text.strip()) or int(text) < 1:
        raise WindowError(f"{name} must be a positive integer")
    return int(text)


def read_groups(path):
    groups = OrderedDict()
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in REQUIRED):
            raise WindowError(f"Missing required input columns: {REQUIRED}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise WindowError("Duplicate CSV column names")
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise WindowError("Malformed CSV row width")
                for name in ("run_dir", "episode_id", "condition", "fault_type", "outcome"):
                    if not row[name].strip():
                        raise WindowError(f"Missing {name}")
                if row["condition"] not in ("fault", "no_fault"):
                    raise WindowError("condition must be fault or no_fault")
                step = positive_integer(row["step"], "step")
                injection = positive_integer(row["injection_step"], "injection_step") if row["injection_step"].strip() else None
                if row["condition"] == "fault":
                    if injection is None or injection < 2:
                        raise WindowError("Fault condition requires injection_step >= 2")
                    if row["fault_type"] not in ("goal_drift", "observation_loss", "context_overload") or row["fault_severity"] not in ("low", "high"):
                        raise WindowError("Invalid fault metadata")
                elif injection is not None or row["fault_type"] != "no_fault" or row["fault_severity"].strip():
                    raise WindowError("Conflicting no_fault metadata")
                if "fault_active" in row:
                    flag = row["fault_active"].strip().lower()
                    if flag not in ("true", "false", "1", "0"):
                        raise WindowError("Invalid fault_active")
                    expected = row["condition"] == "fault" and step >= injection
                    if (flag in ("true", "1")) != expected:
                        raise WindowError("fault_active conflicts with injection step")
                key = (row["run_dir"], row["episode_id"], row["condition"])
                group = groups.setdefault(key, {"rows": [], "outcome": row["outcome"], "fault_type": row["fault_type"],
                                               "fault_severity": row["fault_severity"], "injection_step": injection,
                                               "data_kind": row.get("data_kind", "unspecified"), "comparison": row.get("pair_comparison") or row.get("comparison", "")})
                for name, value in (("outcome", row["outcome"]), ("fault_type", row["fault_type"]), ("fault_severity", row["fault_severity"]),
                                    ("injection_step", injection), ("data_kind", row.get("data_kind", "unspecified")),
                                    ("comparison", row.get("pair_comparison") or row.get("comparison", ""))):
                    if group[name] != value:
                        raise WindowError(f"Inconsistent trajectory metadata: {name}")
                if group["rows"] and step <= group["rows"][-1]["step"]:
                    raise WindowError("Duplicate/reversed steps within a run/episode/condition; input is not sorted automatically")
                values = []
                for name in ("E", "C", "S"):
                    try:
                        value = float(row[name])
                    except (ValueError, TypeError):
                        value = None
                    values.append(value if value is not None and math.isfinite(value) else None)
                group["rows"].append({"step": step, "x": values, "source_line": line})
            except WindowError as exc:
                raise WindowError(f"{path}:{line}: {exc}") from None
    return groups


def pair_key(key):
    path = Path(key[0])
    directory = path.parent if path.name in ("no_fault", "fault") else path
    return str(directory), key[1]


def resolve_comparison(key, group, *, required=False, inferred=""):
    value = group["comparison"]
    if value:
        return value, None
    pair = Path(key[0]) / ".." / "pair.json"
    pair = pair.resolve()
    if pair.is_file():
        try:
            record = json.loads(pair.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise WindowError(f"Cannot read comparison metadata: {pair}") from None
        if not isinstance(record, dict) or record.get("episode_id") != key[1] or not isinstance(record.get("comparison"), str):
            raise WindowError(f"Comparison metadata does not match episode: {pair}")
        return record["comparison"], pair
    if inferred:
        return inferred, None
    if required:
        raise WindowError(f"Comparison unavailable for {key}; keep pair.json accessible or add pair_comparison to the CSV before using exclusion flags")
    return "", None


def build_windows(groups, window_size=5, *, excluded_comparisons=(), no_fault_failure_policy="exclude", post_injection_policy="all", missing_comparison_policy="error"):
    if type(window_size) is not int or window_size < 1:
        raise WindowError("window_size must be positive")
    if no_fault_failure_policy not in ("exclude", "positive") or post_injection_policy not in ("all", "end"):
        raise WindowError("Unknown label policy")
    if missing_comparison_policy not in ("error", "exclude"):
        raise WindowError("Unknown missing comparison policy")
    windows, skipped, labels = [], Counter(), Counter({0: 0, 1: 0})
    metadata_files = {}
    baseline_outcomes = {}
    for key, group in groups.items():
        if key[2] == "no_fault":
            identity = pair_key(key)
            if identity in baseline_outcomes:
                raise WindowError("Multiple no_fault baselines map to the same run/episode; comparison is ambiguous")
            baseline_outcomes[identity] = group["outcome"]
    for key, group in groups.items():
        baseline = baseline_outcomes.get(pair_key(key))
        inferred = ""
        if baseline in FAILURES:
            inferred = "baseline_already_failed"
        elif baseline == "success" and set(excluded_comparisons).issubset({"baseline_already_failed"}):
            inferred = "baseline_success_from_csv"
        comparison, pair_file = resolve_comparison(key, group, required=bool(excluded_comparisons) and missing_comparison_policy == "error", inferred=inferred)
        if pair_file is not None:
            metadata_files[str(pair_file)] = digest(pair_file)
        if excluded_comparisons and not comparison and missing_comparison_policy == "exclude":
            skipped["missing_comparison_groups"] += 1
            continue
        if comparison in excluded_comparisons:
            skipped["excluded_comparison_groups"] += 1
            continue
        if group["outcome"] not in (*FAILURES, "success"):
            skipped["unlabelable_outcome_groups"] += 1
            continue
        failure = group["outcome"] in FAILURES
        if failure and key[2] == "no_fault" and no_fault_failure_policy == "exclude":
            skipped["no_fault_failure_groups"] += 1
            continue
        source = group["rows"]
        if len(source) < window_size:
            skipped["short_groups"] += 1
        for start in range(len(source) - window_size + 1):
            chunk = source[start:start + window_size]
            if any(right["step"] != left["step"] + 1 for left, right in zip(chunk, chunk[1:])):
                skipped["gap_windows"] += 1
                continue
            if any(value is None for row in chunk for value in row["x"]):
                skipped["nonfinite_or_missing_windows"] += 1
                continue
            label = 0
            if failure:
                if key[2] == "no_fault":
                    label = 1  # Only with explicit positive policy.
                else:
                    boundary = chunk[0]["step"] if post_injection_policy == "all" else chunk[-1]["step"]
                    label = int(boundary >= group["injection_step"])
            sample = {"run_dir": key[0], "episode_id": key[1], "condition": key[2], "fault_type": group["fault_type"],
                      "fault_severity": group["fault_severity"], "start_step": chunk[0]["step"], "end_step": chunk[-1]["step"],
                      "label": label, "outcome": group["outcome"], "injection_step": group["injection_step"],
                      "pair_comparison": comparison, "data_kind": group["data_kind"]}
            for feature, index in (("E", 0), ("C", 1), ("S", 2)):
                for offset, row in enumerate(chunk, 1):
                    sample[f"{feature}_{offset}"] = row["x"][index]
            windows.append(sample)
            labels[label] += 1
    return windows, {"groups": len(groups), "samples": len(windows), "class_counts": {"0": labels[0], "1": labels[1]},
                     "negative_samples": labels[0], "positive_samples": labels[1], "skipped": dict(skipped),
                     "pair_metadata_files": metadata_files}


def make_windows(input_path, output_path, *, window_size=5, exclude_baseline_already_failed=False,
                 excluded_comparisons=(), no_fault_failure_policy="exclude", post_injection_policy="all", missing_comparison_policy="error", overwrite=False):
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    sidecar = output_path.with_suffix(output_path.suffix + ".metadata.json").resolve()
    if input_path in (output_path, sidecar) or any(path.exists() and input_path.exists() and os.path.samefile(path, input_path) for path in (output_path, sidecar)):
        raise WindowError("Output must never overwrite the input timeseries.csv")
    if output_path.suffix.lower() != ".csv":
        raise WindowError("This generator writes CSV; use a .csv output path")
    if not overwrite and (output_path.exists() or sidecar.exists()):
        raise WindowError("Output exists; use a new path or explicit --overwrite for derived windows only")
    before = digest(input_path)
    groups = read_groups(input_path)
    exclude = set(excluded_comparisons)
    if exclude_baseline_already_failed:
        exclude.add("baseline_already_failed")
    rows, report = build_windows(groups, window_size, excluded_comparisons=exclude,
                                  no_fault_failure_policy=no_fault_failure_policy, post_injection_policy=post_injection_policy,
                                  missing_comparison_policy=missing_comparison_policy)
    if digest(input_path) != before:
        raise WindowError("Input changed during window generation; output not published")
    if any(digest(Path(path)) != expected for path, expected in report["pair_metadata_files"].items()):
        raise WindowError("Pair metadata changed during window generation")
    protected = {input_path, *(Path(path) for path in report["pair_metadata_files"])}
    if output_path in protected or sidecar in protected or any(candidate.exists() and os.path.samefile(candidate, source) for candidate in (output_path, sidecar) for source in protected):
        raise WindowError("Output must not overwrite source metadata")
    report.update(input_path=str(input_path), input_sha256=before, source_unchanged=True, window_size=window_size, stride=1,
                  group_key=["run_dir", "episode_id", "condition"], post_injection_policy=post_injection_policy,
                  no_fault_failure_policy=no_fault_failure_policy, excluded_comparisons=sorted(exclude), train_test_split=False,
                  missing_comparison_policy=missing_comparison_policy,
                  label_definition="Trajectory-outcome weak labels, not measured failure onset. Fault failure windows must satisfy injection policy.",
                  feature_tensor_order="For each time index t: [E_t, C_t, S_t]. CSV columns are feature-major E_1..E_W,C_1..C_W,S_1..S_W.")
    columns = ["run_dir", "episode_id", "condition", "fault_type", "fault_severity", "start_step", "end_step"]
    columns += [f"{feature}_{index}" for feature in ("E", "C", "S") for index in range(1, window_size + 1)]
    columns += ["label", "outcome", "injection_step", "pair_comparison", "data_kind"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        if overwrite:
            os.replace(temporary, output_path)
        else:
            os.link(temporary, output_path)
            temporary.unlink()
        with sidecar.open("w" if overwrite else "x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return report
