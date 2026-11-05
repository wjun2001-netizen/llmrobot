"""Negative checks use temporary copies of a genuine, previously collected run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.validate import validate

ROOT = Path(__file__).resolve().parents[1]


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for report in sorted((ROOT / "runs").glob("*/validation.json"), reverse=True):
            if json.loads(report.read_text())["valid"]:
                cls.rows = [json.loads(s) for s in report.with_name("trajectories.jsonl").read_text().splitlines()]
                cls.episode_count = len({r["episode_id"] for r in cls.rows})
                break
        else:
            raise RuntimeError("First collect at least one genuine successful run")

    def check_rows(self, rows, expected=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copy.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            return validate(path, self.episode_count if expected is None else expected)

    def test_genuine_run(self):
        self.assertTrue(self.check_rows(self.rows)["valid"])

    def test_missing_field(self):
        rows = copy.deepcopy(self.rows)
        del rows[0]["goal"]
        self.assertFalse(self.check_rows(rows)["valid"])

    def test_empty_observation_and_action(self):
        rows = copy.deepcopy(self.rows)
        rows[0].update(observation="", action="")
        report = self.check_rows(rows)
        self.assertFalse(report["valid"])
        self.assertEqual(report["empty_observation_count"], 1)
        self.assertEqual(report["empty_action_count"], 1)

    def test_missing_success_not_replaced_with_false(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["success"] = None
        report = self.check_rows(rows)
        self.assertFalse(report["valid"])
        self.assertEqual(report["null_counts"]["success"], 1)

    def test_failed_terminal(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]["success"] = False
        self.assertEqual(self.check_rows(rows)["failure_count"], 1)

    def test_step_gap(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["step"] = 10
        self.assertFalse(self.check_rows(rows)["valid"])

    def test_wrong_episode_count(self):
        self.assertFalse(self.check_rows(self.rows, self.episode_count + 1)["valid"])

    def test_empty_run(self):
        self.assertFalse(self.check_rows([])["valid"])


if __name__ == "__main__":
    unittest.main()
