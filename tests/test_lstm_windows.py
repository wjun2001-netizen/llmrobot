"""CSV/window unit fixtures only. No model loading, training, splitting or ALFWorld."""
import csv
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from src.lstm_windows import WindowError, build_windows, make_windows, read_groups

COLUMNS = ["run_dir", "episode_id", "condition", "step", "E", "C", "S", "fault_type", "fault_severity",
           "fault_active", "injection_step", "outcome", "pair_comparison", "data_kind"]


def fixture_rows(condition="fault", *, count=9, injection=4, outcome="task_failure", run_dir="fixture-run/fault", episode_id="fixture-episode", comparison=""):
    return [{"run_dir": run_dir, "episode_id": episode_id, "condition": condition, "step": step,
             "E": step / 10, "C": step / 100, "S": -step / 20,
             "fault_type": "goal_drift" if condition == "fault" else "no_fault",
             "fault_severity": "low" if condition == "fault" else "",
             "fault_active": condition == "fault" and step >= injection,
             "injection_step": injection if condition == "fault" else "", "outcome": outcome,
             "pair_comparison": comparison, "data_kind": "unit_test_fixture"} for step in range(1, count + 1)]


def write_fixture(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class WindowTests(unittest.TestCase):
    def build(self, rows, size=5, **kwargs):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "timeseries.csv"
            write_fixture(source, rows)
            return build_windows(read_groups(source), size, **kwargs)

    def test_length_five_stride_one_and_feature_order(self):
        windows, report = self.build(fixture_rows())
        self.assertEqual(len(windows), 5)
        first = windows[0]
        self.assertEqual((first["start_step"], first["end_step"]), (1, 5))
        self.assertEqual([first[f"E_{i}"] for i in range(1, 6)], [.1, .2, .3, .4, .5])
        self.assertEqual([first[f"C_{i}"] for i in range(1, 6)], [.01, .02, .03, .04, .05])
        self.assertEqual(first["S_5"], -.25)
        self.assertEqual(report["samples"], report["positive_samples"] + report["negative_samples"])

    def test_fault_failure_labels_only_fully_post_injection(self):
        windows, report = self.build(fixture_rows(injection=4))
        self.assertEqual([row["label"] for row in windows], [0, 0, 0, 1, 1])
        self.assertEqual(report["class_counts"], {"0": 3, "1": 2})

    def test_end_policy_is_explicit_alternative(self):
        windows, _ = self.build(fixture_rows(injection=4), post_injection_policy="end")
        self.assertTrue(all(row["label"] == 1 for row in windows))

    def test_pre_injection_and_unreached_failure_windows_negative(self):
        windows, report = self.build(fixture_rows(injection=20))
        self.assertEqual(report["positive_samples"], 0)
        self.assertTrue(all(row["label"] == 0 for row in windows))

    def test_success_trajectories_always_negative(self):
        rows = fixture_rows(outcome="success") + fixture_rows("no_fault", outcome="success", run_dir="fixture/no_fault")
        windows, report = self.build(rows)
        self.assertEqual(len(windows), 10)
        self.assertEqual(report["class_counts"], {"0": 10, "1": 0})

    def test_no_fault_failures_excluded_by_default_user_choice(self):
        windows, report = self.build(fixture_rows("no_fault", outcome="step_limit"))
        self.assertEqual(windows, [])
        self.assertEqual(report["skipped"]["no_fault_failure_groups"], 1)

    def test_no_fault_failure_positive_requires_explicit_override(self):
        windows, _ = self.build(fixture_rows("no_fault", outcome="step_limit"), no_fault_failure_policy="positive")
        self.assertTrue(all(row["label"] == 1 for row in windows))

    def test_no_cross_episode_condition_or_run_windows(self):
        rows = fixture_rows(count=4, episode_id="episode-A") + fixture_rows(count=4, episode_id="episode-B")
        rows += fixture_rows("no_fault", count=4, outcome="success")
        rows += fixture_rows(count=4, run_dir="another-run/fault")
        windows, report = self.build(rows)
        self.assertEqual(windows, [])
        self.assertEqual(report["groups"], 4)
        self.assertEqual(report["skipped"]["short_groups"], 4)

    def test_repeated_ids_across_runs_are_separate_samples(self):
        rows = fixture_rows(count=5, run_dir="run-A") + fixture_rows(count=5, run_dir="run-B")
        windows, _ = self.build(rows)
        self.assertEqual([row["run_dir"] for row in windows], ["run-A", "run-B"])
        self.assertEqual(len(windows), 2)

    def test_nan_missing_and_inf_block_windows_without_bridging(self):
        for value in ("NaN", "", "inf", None):
            rows = fixture_rows(count=10)
            rows[4]["C"] = value
            windows, report = self.build(rows)
            self.assertEqual([(row["start_step"], row["end_step"]) for row in windows], [(6, 10)])
            self.assertEqual(report["skipped"]["nonfinite_or_missing_windows"], 5)

    def test_gap_never_bridged(self):
        rows = [row for row in fixture_rows(count=10) if row["step"] != 5]
        windows, report = self.build(rows)
        self.assertEqual([(row["start_step"], row["end_step"]) for row in windows], [(6, 10)])
        self.assertGreater(report["skipped"]["gap_windows"], 0)

    def test_reversed_duplicate_and_inconsistent_outcome_rejected(self):
        bad = []
        rows = fixture_rows()
        bad.append(list(reversed(rows)))
        bad.append(rows + [deepcopy(rows[-1])])
        rows = fixture_rows()
        rows[-1]["outcome"] = "success"
        bad.append(rows)
        for rows in bad:
            with self.assertRaises(WindowError):
                self.build(rows)

    def test_unlabelable_runtime_failure_is_excluded_not_negative(self):
        windows, report = self.build(fixture_rows(outcome="generation_error"))
        self.assertFalse(windows)
        self.assertEqual(report["skipped"]["unlabelable_outcome_groups"], 1)

    def test_csv_comparison_can_exclude_both_conditions(self):
        rows = fixture_rows(comparison="baseline_already_failed")
        rows += fixture_rows("no_fault", outcome="success", run_dir="fixture/no_fault", comparison="baseline_already_failed")
        windows, report = self.build(rows, excluded_comparisons={"baseline_already_failed"})
        self.assertFalse(windows)
        self.assertEqual(report["skipped"]["excluded_comparison_groups"], 2)

    def test_missing_comparison_is_error_when_exclusion_requested(self):
        with self.assertRaisesRegex(WindowError, "Comparison unavailable"):
            self.build(fixture_rows(), excluded_comparisons={"baseline_already_failed"})

    def test_missing_comparison_can_be_explicitly_excluded(self):
        rows, report = self.build(fixture_rows(), excluded_comparisons={"baseline_already_failed"}, missing_comparison_policy="exclude")
        self.assertFalse(rows)
        self.assertEqual(report["skipped"]["missing_comparison_groups"], 1)

    def test_baseline_failure_can_be_inferred_from_same_pair_csv_without_json(self):
        rows = fixture_rows(run_dir="paired-run/fault")
        rows += fixture_rows("no_fault", outcome="step_limit", run_dir="paired-run/no_fault")
        windows, report = self.build(rows, excluded_comparisons={"baseline_already_failed"})
        self.assertFalse(windows)
        self.assertEqual(report["skipped"]["excluded_comparison_groups"], 2)

    def test_csv_baseline_success_does_not_require_pair_json_for_baseline_filter(self):
        rows = fixture_rows(run_dir="paired-run/fault")
        rows += fixture_rows("no_fault", outcome="success", run_dir="paired-run/no_fault")
        windows, report = self.build(rows, excluded_comparisons={"baseline_already_failed"})
        self.assertEqual(len(windows), 10)
        self.assertEqual(report["positive_samples"], 2)

    def test_invalid_window_size(self):
        for size in (0, -1, True):
            with self.assertRaises(WindowError):
                self.build(fixture_rows(), size=size)


class FileTests(unittest.TestCase):
    def test_csv_output_counts_metadata_and_read_only_input(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "timeseries.csv", Path(temp) / "windows.csv"
            write_fixture(source, fixture_rows())
            before = source.read_bytes()
            report = make_windows(source, output)
            rows = read_csv(output)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(rows), 5)
            self.assertEqual(report["positive_samples"], 2)
            self.assertEqual(report["negative_samples"], 3)
            self.assertFalse(report["train_test_split"])
            self.assertIn("E_5", rows[0])
            self.assertIn("S_5", rows[0])
            self.assertTrue(output.with_suffix('.csv.metadata.json').is_file())
            self.assertNotIn(b'\r', output.read_bytes())

    def test_pair_json_baseline_exclusion_without_source_csv_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pair = root / "pair"
            (pair / "fault").mkdir(parents=True)
            (pair / "no_fault").mkdir()
            (pair / "pair.json").write_text(json.dumps({"episode_id": "fixture-episode", "comparison": "baseline_already_failed"}))
            rows = fixture_rows(run_dir=str(pair / "fault"))
            rows += fixture_rows("no_fault", outcome="step_limit", run_dir=str(pair / "no_fault"))
            source, output = root / "input.csv", root / "out.csv"
            write_fixture(source, rows)
            before = source.read_bytes()
            report = make_windows(source, output, exclude_baseline_already_failed=True)
            self.assertEqual(report["samples"], 0)
            self.assertEqual(report["skipped"]["excluded_comparison_groups"], 2)
            self.assertEqual(source.read_bytes(), before)

    def test_derived_output_not_overwritten_unless_requested(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input.csv", Path(temp) / "out.csv"
            write_fixture(source, fixture_rows())
            make_windows(source, output)
            before = output.read_bytes()
            with self.assertRaises(WindowError):
                make_windows(source, output)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(make_windows(source, output, overwrite=True)["samples"], 5)

    def test_input_cannot_be_its_own_output_even_with_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.csv"
            write_fixture(source, fixture_rows())
            before = source.read_bytes()
            with self.assertRaises(WindowError):
                make_windows(source, source, overwrite=True)
            self.assertEqual(source.read_bytes(), before)

    def test_short_fixture_yields_zero_counts_and_header_only_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input.csv", Path(temp) / "out.csv"
            write_fixture(source, fixture_rows(count=4))
            report = make_windows(source, output)
            self.assertEqual(report["class_counts"], {"0": 0, "1": 0})
            self.assertEqual(read_csv(output), [])

    def test_cli_prints_class_counts_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input.csv", Path(temp) / "out.csv"
            write_fixture(source, fixture_rows())
            result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "make_lstm_windows.py"),
                                     "--input", str(source), "--output", str(output), "--window-size", "5"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["class_counts"], {"0": 3, "1": 2})
            self.assertEqual(report["positive_samples"], 2)
            self.assertEqual(report["negative_samples"], 3)
            self.assertFalse(report["train_test_split"])

    def test_input_hardlink_cannot_be_truncated_by_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input.csv", Path(temp) / "out.csv"
            write_fixture(source, fixture_rows())
            before = source.read_bytes()
            os.link(source, output.with_suffix('.csv.metadata.json'))
            with self.assertRaises(WindowError):
                make_windows(source, output, overwrite=True)
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
