"""Train-calibration and standardized diagnosis tests; synthetic CPU fixtures only."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

import run_intervention
from src.anomaly_calibration import calibrate_training_observations, standardized_diagnosis
from src.intervention import InterventionConfig, choose_intervention
from src.intervention_runtime import ROOT
from src.lstm_classifier import FailureRiskLSTM
from src.lstm_training_data import group_split, load_windows
from src.timeseries_features import digest


CALIBRATION = {"statistics": {"E_mean": .2, "E_std": .1, "C_mean": .1, "C_std": .05,
                              "S_mean": .8, "S_std": .1}, "std_floor": 1e-6}


class DiagnosisTests(unittest.TestCase):
    def choose(self, features, risk=.5, **overrides):
        diagnostics = {}
        config = InterventionConfig(calibration=CALIBRATION, **overrides)
        result = choose_intervention(config, features, risk, step=9, injection_step=2,
                                      fault_active=True, diagnostics=diagnostics)
        return result, diagnostics

    def test_C_largest_even_when_S_passes_old_absolute_cutoff(self):
        (kind, _), fields = self.choose({"E": .3, "C": .5, "S": .2})
        self.assertEqual(kind, "context_cleaning")
        self.assertEqual(fields["selected_anomaly"], "C")
        self.assertAlmostEqual(fields["diagnostic_C_z"], 8.)
        self.assertGreater(fields["diagnostic_C_z"], fields["diagnostic_S_z"])

    def test_S_largest(self):
        self.assertEqual(self.choose({"E": .3, "C": .15, "S": .2})[0][0], "goal_reinjection")

    def test_E_largest(self):
        self.assertEqual(self.choose({"E": 1., "C": .15, "S": .7})[0][0], "reobservation_replanning")

    def test_below_risk_does_not_even_diagnose(self):
        result, diagnostics = self.choose({"E": 5., "C": .9, "S": .1}, risk=.499)
        self.assertIsNone(result[0])
        self.assertEqual(diagnostics, {})

    def test_min_anomaly_gate(self):
        result, diagnostics = self.choose({"E": .21, "C": .11, "S": .79})
        self.assertIsNone(result[0])
        self.assertLess(diagnostics["selected_anomaly_score"], 1.)

    def test_exact_minimum_and_tie_break(self):
        calibration = {"statistics": {"E_mean": 0., "E_std": 1., "C_mean": 0., "C_std": 1.,
                                      "S_mean": 1., "S_std": 1.}, "std_floor": 1e-6}
        cfg = InterventionConfig(calibration=calibration)
        diagnostics = {}
        result = choose_intervention(cfg, {"E": 1., "C": 1., "S": 0.}, .5, step=5,
                                      injection_step=2, fault_active=True, diagnostics=diagnostics)
        self.assertEqual(result[0], "reobservation_replanning")
        self.assertEqual(diagnostics["selected_anomaly_score"], 1.)

    def test_signed_S_direction_and_negative_clamp(self):
        result, diagnostics = self.choose({"E": .1, "C": .05, "S": .9})
        self.assertIsNone(result[0])
        self.assertLess(diagnostics["diagnostic_S_z"], 0.)
        self.assertEqual(diagnostics["anomaly_scores"], {"E": 0., "C": 0., "S": 0.})
        self.assertIsNone(diagnostics["selected_anomaly"])

    def test_absolute_thresholds_do_not_control_proposed(self):
        features = {"E": .3, "C": .5, "S": .2}
        first = self.choose(features)[0]
        self.assertEqual(first, self.choose(features, similarity_threshold=-1., context_threshold=100., entropy_threshold=99.)[0])

    def test_zero_and_small_std_floored(self):
        calibration = {"statistics": {"E_mean": 0., "E_std": 0., "C_mean": 0., "C_std": 1e-15,
                                      "S_mean": 1., "S_std": 0.}, "std_floor": 1e-6}
        result = standardized_diagnosis({"E": 2e-6, "C": 1e-6, "S": 1.}, calibration)
        self.assertEqual(result["diagnostic_E_z"], 2.)
        self.assertEqual(result["diagnostic_C_z"], 1.)
        self.assertTrue(all(math.isfinite(result[f"diagnostic_{f}_z"]) for f in ("E", "C", "S")))

    def test_no_uncalibrated_fallback(self):
        with self.assertRaisesRegex(ValueError, "calibration"):
            choose_intervention(InterventionConfig(), {"E": 9., "C": 9., "S": .1}, .9,
                                step=5, injection_step=2, fault_active=True)

    def test_invalid_minimum(self):
        for value in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                InterventionConfig(min_anomaly_z=value)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        root = ROOT / "runs_intervention"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="unit_test_fixture_", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.windows, self.split_path = self.root / "windows.csv", self.root / "split.json"
        self.rows = []
        for group in range(5):
            for start in (1, 2):
                row = dict(run_dir=f"fixture/{group}", episode_id=str(group), condition="fault", start_step=start, end_step=start+4, label=0)
                for offset in range(5):
                    step = start + offset
                    row.update({f"E_{offset+1}": float(step), f"C_{offset+1}": step / 100,
                                f"S_{offset+1}": .9 - step / 100})
                self.rows.append(row)
        self.write_rows()
        self.split = group_split(load_windows(self.windows))
        self.save_split()

    def write_rows(self):
        with self.windows.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def save_split(self):
        self.split["input_sha256"] = digest(self.windows)
        self.split_path.write_text(json.dumps(self.split))

    def calibrate(self):
        return calibrate_training_observations(self.windows, self.split_path)

    def test_validation_ignored_and_steps_deduplicated(self):
        original = self.calibrate()
        for index in self.split["validation_indices"]:
            for feature in ("E", "C", "S"):
                for step in range(1, 6):
                    self.rows[index][f"{feature}_{step}"] = 999.
        self.write_rows()
        self.save_split()
        recalibrated = self.calibrate()
        self.assertEqual(original["statistics"], recalibrated["statistics"])
        self.assertEqual(recalibrated["validation_samples_used"], 0)
        self.assertEqual(recalibrated["sample_count"], 24)  # Four TRAIN groups x six unique steps.
        self.assertEqual(recalibrated["training_window_count"], 8)
        self.assertAlmostEqual(recalibrated["statistics"]["E_mean"], 3.5)
        self.assertAlmostEqual(recalibrated["statistics"]["E_std"], math.sqrt(35 / 12))

    def test_label_1_and_mixed_overlap_excluded(self):
        first_train = self.split["train_indices"][0]
        self.rows[first_train+1]["label"] = 1
        self.write_rows()
        self.save_split()
        result = self.calibrate()
        self.assertEqual(result["mixed_label_steps_excluded"], 4)
        self.assertEqual(result["sample_count"], 19)

    def test_no_label_0_population_fails_without_fallback(self):
        for index in self.split["train_indices"]:
            self.rows[index]["label"] = 1
        self.write_rows()
        self.save_split()
        with self.assertRaisesRegex(ValueError, "no validation/label-1 fallback"):
            self.calibrate()

    def test_input_hash_and_preservation(self):
        before = self.windows.read_bytes(), self.split_path.read_bytes()
        result = self.calibrate()
        self.assertEqual(result["input_sha256"], digest(self.windows))
        self.assertEqual(result["split_sha256"], digest(self.split_path))
        self.assertEqual(before, (self.windows.read_bytes(), self.split_path.read_bytes()))
        self.windows.write_text(self.windows.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.calibrate()

    def test_conflicting_overlapping_features_rejected(self):
        self.rows[self.split["train_indices"][1]]["E_1"] = 100.
        self.write_rows()
        self.save_split()
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.calibrate()

    def test_calibrated_zero_std_is_logged_raw_and_effective(self):
        for row in self.rows:
            for step in range(1, 6):
                row[f"C_{step}"] = .2
        self.write_rows()
        self.save_split()
        result = self.calibrate()
        self.assertEqual(result["statistics"]["C_std"], 0.)
        self.assertEqual(result["effective_std"]["C"], 1e-6)

    def test_CLI_proposed_uses_checkpoint_split_without_absolute_entropy(self):
        model = FailureRiskLSTM()
        path = self.root / "model.pt"
        torch.save({"state_dict": model.state_dict(), "architecture": {"input_size": 3, "hidden_size": 16,
                    "num_layers": 1, "batch_first": True, "window_size": 5, "feature_order": ["E", "C", "S"],
                    "output_size": 1}, "input_sha256": digest(self.windows)}, path)
        game = self.root / "game.tw-pddl"
        game.write_text('{}')
        args = run_intervention.build_parser().parse_args(["--game", str(game), "--lstm-checkpoint", str(path),
                    "--calibration-windows", str(self.windows), "--entropy-training-windows", str(self.root / "unused.csv")])
        prepared = run_intervention.prepare(args)
        cfg = prepared[3][0]
        self.assertEqual(cfg.min_anomaly_z, 1.)
        self.assertEqual(cfg.lstm_risk_threshold, .5)
        self.assertIsNone(cfg.entropy_threshold)
        self.assertEqual(cfg.calibration["input_sha256"], digest(self.windows))
        self.assertEqual(prepared[-1]["diagnostic_calibration"], cfg.calibration)
        args.min_anomaly_z = 2.
        self.assertEqual(run_intervention.prepare(args)[3][0].min_anomaly_z, 2.)


if __name__ == "__main__":
    unittest.main()
