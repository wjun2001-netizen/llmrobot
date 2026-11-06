"""CPU-only synthetic fixtures: no pretrained weights, GPU, API or ALFWorld."""
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from src.lstm_classifier import FailureRiskLSTM
from src.lstm_metrics import binary_metrics
from src.lstm_training import TrainingConfig, train
from src.lstm_training_data import (TrainingDataError, check_group_leakage, group_split,
                                    load_windows, training_pos_weight)
from train_lstm import build_parser


def fixture_rows(groups=5):
    rows = []
    for group in range(groups):
        for offset in range(4):
            row = {"run_dir": f"unit_test_fixture/run{group}", "episode_id": f"ep{group}",
                   "condition": "fault", "fault_type": "goal_drift", "fault_severity": "low",
                   "start_step": offset + 1, "end_step": offset + 5, "label": offset % 2,
                   "data_kind": "unit_test_fixture"}
            for feature, base in (("E", 10), ("C", 20), ("S", 30)):
                row.update({f"{feature}_{step}": base + step + offset / 10 for step in range(1, 6)})
            rows.append(row)
    return rows


def write_fixture(path, rows=None, reverse_columns=False):
    rows = rows if rows is not None else fixture_rows()
    fields = list(rows[0])
    if reverse_columns:
        fields.reverse()
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "windows.csv"
        write_fixture(self.path)

    def test_explicit_time_major(self):
        sample = load_windows(self.path)[0]
        self.assertEqual(sample.x, [[11., 21., 31.], [12., 22., 32.], [13., 23., 33.],
                                    [14., 24., 34.], [15., 25., 35.]])

    def test_column_order_does_not_matter(self):
        original = load_windows(self.path)
        write_fixture(self.path, reverse_columns=True)
        self.assertEqual(load_windows(self.path), original)

    def test_invalid_features(self):
        for invalid in ("nan", "inf", "-inf", "", "bad"):
            with self.subTest(invalid=invalid):
                rows = fixture_rows()
                rows[0]["C_3"] = invalid
                write_fixture(self.path, rows)
                with self.assertRaises(TrainingDataError):
                    load_windows(self.path)

    def test_invalid_metadata(self):
        for field, value in (("label", 2), ("label", "nan"), ("run_dir", ""),
                             ("condition", "other"), ("start_step", 0), ("end_step", 6)):
            with self.subTest(field=field, value=value):
                rows = fixture_rows()
                rows[0][field] = value
                write_fixture(self.path, rows)
                with self.assertRaises(TrainingDataError):
                    load_windows(self.path)

    def test_missing_column(self):
        rows = fixture_rows()
        for row in rows:
            del row["S_5"]
        write_fixture(self.path, rows)
        with self.assertRaises(TrainingDataError):
            load_windows(self.path)

    def test_wrong_window_size(self):
        rows = fixture_rows()
        for row in rows:
            row["E_6"] = 1
        write_fixture(self.path, rows)
        with self.assertRaises(TrainingDataError):
            load_windows(self.path)

    def test_duplicate_sample(self):
        rows = fixture_rows()
        write_fixture(self.path, rows + [rows[0]])
        with self.assertRaisesRegex(TrainingDataError, "Duplicate sample"):
            load_windows(self.path)

    def test_empty_input(self):
        self.path.write_text("", encoding="utf-8")
        with self.assertRaises(TrainingDataError):
            load_windows(self.path)

    def test_split_reproducible_and_disjoint(self):
        samples = load_windows(self.path)
        split = group_split(samples)
        self.assertEqual(split, group_split(samples))
        self.assertEqual(split["train_sample_count"], 16)
        self.assertEqual(split["validation_sample_count"], 4)
        self.assertEqual(split["leakage_check"]["group_overlap_count"], 0)
        self.assertFalse({samples[i].group for i in split["train_indices"]} &
                         {samples[i].group for i in split["validation_indices"]})

    def test_paired_conditions_stay_together(self):
        rows = fixture_rows()
        paired = []
        for row in rows:
            for condition in ("no_fault", "fault"):
                paired.append({**row, "run_dir": f"{row['run_dir']}/{condition}", "condition": condition})
        write_fixture(self.path, paired)
        samples = load_windows(self.path)
        split = group_split(samples)
        train_pairs = {samples[i].split_group for i in split["train_indices"]}
        validation_pairs = {samples[i].split_group for i in split["validation_indices"]}
        self.assertFalse(train_pairs & validation_pairs)
        self.assertEqual(split["effective_group_count"], 5)

    def test_cannot_split_one_group(self):
        write_fixture(self.path, fixture_rows(groups=1))
        with self.assertRaisesRegex(TrainingDataError, "at least two"):
            group_split(load_windows(self.path))

    def test_detect_group_leakage(self):
        samples = load_windows(self.path)
        with self.assertRaisesRegex(TrainingDataError, "Data leakage"):
            check_group_leakage(samples, [0], list(range(1, len(samples))))

    def test_detect_pair_leakage(self):
        samples = load_windows(self.path)[:2]
        samples = [replace(samples[0], run_dir="pair/no_fault", condition="no_fault"),
                   replace(samples[1], run_dir="pair/fault", condition="fault")]
        with self.assertRaisesRegex(TrainingDataError, "Data leakage"):
            check_group_leakage(samples, [0], [1])

    def test_detect_sample_overlap_and_duplicates(self):
        samples = load_windows(self.path)
        for train_indices, val_indices in (([0, 0], list(range(1, 20))), ([0, 1], list(range(1, 20)))):
            with self.assertRaises(TrainingDataError):
                check_group_leakage(samples, train_indices, val_indices)

    def test_weight_uses_only_train(self):
        samples = load_windows(self.path)
        weight, counts = training_pos_weight(samples, [0, 1, 2])
        self.assertEqual(weight, 2.)
        self.assertEqual(counts, {"0": 2, "1": 1})
        changed = [replace(s, label=1 - s.label) if i >= 3 else s for i, s in enumerate(samples)]
        self.assertEqual(training_pos_weight(changed, [0, 1, 2]), (weight, counts))

    def test_single_class_train_rejected(self):
        with self.assertRaisesRegex(TrainingDataError, "both labels"):
            training_pos_weight(load_windows(self.path), [0, 2])


class ModelAndMetricTests(unittest.TestCase):
    def test_model_architecture_and_shapes(self):
        model = FailureRiskLSTM()
        self.assertEqual((model.lstm.input_size, model.lstm.hidden_size, model.lstm.num_layers), (3, 16, 1))
        self.assertTrue(model.lstm.batch_first)
        self.assertEqual((model.head.in_features, model.head.out_features), (16, 1))
        for batch in (1, 4):
            self.assertEqual(model(torch.zeros(batch, 5, 3)).shape, (batch,))

    def test_final_hidden_and_no_sigmoid(self):
        model = FailureRiskLSTM()
        x = torch.randn(2, 5, 3)
        _, (hidden, _) = model.lstm(x)
        torch.testing.assert_close(model(x), model.head(hidden[-1]).squeeze(-1))
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(2.)
        torch.testing.assert_close(model(torch.zeros(2, 5, 3)), torch.full((2,), 2.))

    def test_invalid_model_input(self):
        model = FailureRiskLSTM()
        for x in (torch.zeros(2, 3, 5), torch.zeros(5, 3), torch.full((1, 5, 3), float("nan"))):
            with self.assertRaises(ValueError):
                model(x)

    def test_metrics(self):
        result = binary_metrics([0, 0, 1, 1], [.1, .7, .3, .8])
        for key in ("accuracy", "precision", "recall", "f1_score"):
            self.assertEqual(result[key], .5)
        self.assertEqual(result["roc_auc"], .75)

    def test_auc_ties_and_extremes(self):
        self.assertEqual(binary_metrics([0, 1], [.5, .5])["roc_auc"], .5)
        self.assertEqual(binary_metrics([0, 1], [.1, .9])["roc_auc"], 1.)
        self.assertEqual(binary_metrics([0, 1], [.9, .1])["roc_auc"], 0.)

    def test_single_class_validation(self):
        result = binary_metrics([0, 0], [.1, .2])
        self.assertIsNone(result["roc_auc"])
        self.assertEqual(result["roc_auc_unavailable_reason"], "validation_has_single_class")
        self.assertEqual(result["f1_score"], 0.)

    def test_invalid_metrics(self):
        for labels, probabilities in (([], []), ([1], [float("nan")]), ([2], [.5]), ([1], [2.])):
            with self.assertRaises(ValueError):
                binary_metrics(labels, probabilities)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "windows.csv"
        write_fixture(self.path)

    def test_end_to_end_outputs_and_reproducibility(self):
        before = self.path.read_bytes()
        config = TrainingConfig(epochs=2, batch_size=4)
        run, metrics = train(self.path, self.root / "results", config)
        self.assertEqual(self.path.read_bytes(), before)
        for filename in ("metrics.json", "training_history.csv", "predictions.csv", "model.pt", "split.json"):
            self.assertTrue((run / filename).is_file())
        split = json.loads((run / "split.json").read_text())
        self.assertEqual(split["leakage_check"]["group_overlap_count"], 0)
        self.assertEqual(metrics["pos_weight"], 1.)
        self.assertEqual(metrics["train_class_counts"], {"0": 8, "1": 8})
        self.assertEqual(metrics["input_sha256"], hashlib.sha256(before).hexdigest())
        with (run / "training_history.csv").open() as stream:
            history = list(csv.DictReader(stream))
        self.assertEqual(len(history), 2)
        with (run / "predictions.csv").open() as stream:
            predictions = list(csv.DictReader(stream))
        self.assertEqual(len(predictions), 4)
        self.assertEqual([int(row["sample_index"]) for row in predictions], split["validation_indices"])
        self.assertTrue(all(row["subset"] == "validation" for row in predictions))
        checkpoint = torch.load(run / "model.pt", weights_only=True)
        model = FailureRiskLSTM()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        x = torch.tensor([load_windows(self.path)[i].x for i in split["validation_indices"]])
        with torch.inference_mode():
            logits = model(x).tolist()
        for logit, row in zip(logits, predictions):
            self.assertAlmostEqual(logit, float(row["logit"]), places=6)
        again, repeated_metrics = train(self.path, self.root / "results", config)
        self.assertNotEqual(run, again)
        self.assertEqual(metrics, repeated_metrics)
        self.assertEqual((run / "predictions.csv").read_bytes(), (again / "predictions.csv").read_bytes())
        self.assertEqual(self.path.read_bytes(), before)

    def test_single_class_validation_training_completes(self):
        rows = fixture_rows()
        samples = load_windows(self.path)
        for index in group_split(samples)["validation_indices"]:
            rows[index]["label"] = 0
        write_fixture(self.path, rows)
        run, metrics = train(self.path, self.root / "results", TrainingConfig(epochs=1))
        self.assertIsNone(metrics["roc_auc"])
        json.loads((run / "metrics.json").read_text(), parse_constant=lambda value: self.fail(value))

    def test_float32_overflow_rejected(self):
        rows = fixture_rows()
        rows[0]["E_1"] = "1e100"
        write_fixture(self.path, rows)
        with self.assertRaisesRegex(TrainingDataError, "float32"):
            train(self.path, self.root / "results", TrainingConfig(epochs=1))
        self.assertFalse((self.root / "results").exists())

    def test_invalid_config(self):
        for config in (TrainingConfig(epochs=0), TrainingConfig(batch_size=0), TrainingConfig(learning_rate=float("nan")),
                       TrainingConfig(seed=-1), TrainingConfig(validation_fraction=1), TrainingConfig(device="bad")):
            with self.assertRaises(TrainingDataError):
                config.validate()

    def test_cli_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual((args.epochs, args.batch_size, args.learning_rate, args.seed), (30, 32, .001, 42))
        self.assertEqual(args.device, "cpu")

    def test_cli_cpu(self):
        repo = Path(__file__).resolve().parents[1]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        result = subprocess.run([sys.executable, "train_lstm.py", "--input", str(self.path),
                                 "--output-dir", str(self.root / "results"), "--epochs", "1"],
                                cwd=repo, env=env, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RESULT_DIR=", result.stdout)
        self.assertIn('"group_overlap_count": 0', result.stdout)


if __name__ == "__main__":
    unittest.main()
