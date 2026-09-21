from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.evaluate_top1_ltl import digest, read_run, run_experiment
from jepa_lmc.verification.nuxmv import NuSMVRun, find_nusmv_executable


def make_run(directory: Path) -> None:
    directory.mkdir()
    # Both graphs contain an unreachable-from-Top1 danger state. Querying only
    # designated initial states would miss half the requested comparisons.
    record = {
        "seed": 7,
        "case": "safety_mismatch",
        "states": [[4, 5], [2, 1]],
        "initial": 0,
        "labels": [["safe"], ["danger"]],
        "graphs": {
            "real": [[[0, 1]], [[0, 1]]],
            "top1": [[[0, 0]], [[0, 1]]],
        },
    }
    graphs = directory / "relations.jsonl"
    graphs.write_text(json.dumps(record) + "\n", encoding="utf-8")
    (directory / "report.json").write_text(
        json.dumps(
            {
                "source_revision": "fixture",
                "maps": [{"seed": 7, "case": record["case"], "family": "fixture"}],
                "exports": {"relations.jsonl": digest(graphs)},
            }
        ),
        encoding="utf-8",
    )


class SavedTop1LTLTests(unittest.TestCase):
    def assert_results(self, report: dict, output: Path) -> None:
        self.assertEqual(report["external_backend_calls"], 2)
        self.assertEqual(report["external_ltl_verdicts"], 24)
        self.assertEqual(report["initial"]["comparisons"], 6)
        self.assertEqual(report["initial"]["matched"], 2)
        self.assertEqual(report["all_states"]["comparisons"], 12)
        self.assertEqual(report["all_states"]["matched"], 8)
        self.assertEqual(report["initial"]["false_safe_count"], 2)
        self.assertEqual(report["all_states"]["false_safe_count"], 2)
        self.assertEqual(report["seeds"][0]["initial_all_properties_agree_maps"], 0)
        # No positive real examples: an undefined balanced score is not 100%.
        self.assertIsNone(report["all_states"]["primary_balanced_score"])
        with (output / "queries.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        self.assertEqual((rows[0]["row"], rows[0]["column"]), ("4", "5"))
        self.assertEqual((rows[6]["row"], rows[6]["column"]), ("2", "1"))
        self.assertEqual(rows[6]["initial"], "False")
        for name, expected in report["exports"].items():
            self.assertEqual(digest(output / name), expected)

    def test_same_saved_graphs_and_initial_all_state_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, output = Path(temporary) / "input", Path(temporary) / "output"
            make_run(run)
            before = {p.name: digest(p) for p in run.iterdir()}
            real = NuSMVRun(sys.executable, (False,) * 12, "mock real backend")
            top1 = NuSMVRun(
                sys.executable,
                (True, False, True, False, True, True) + (False,) * 6,
                "mock top1 backend",
            )
            with (
                patch(
                    "experiments.evaluate_top1_ltl.run_nusmv_model",
                    side_effect=[real, top1],
                ) as backend,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = run_experiment(run, output, sys.executable)
            self.assert_results(report, output)
            self.assertEqual(before, {p.name: digest(p) for p in run.iterdir()})
            texts = [call.args[0] for call in backend.call_args_list]
            self.assertIn("state = s0 : {s1};", texts[0])
            self.assertIn("state = s0 : {s0};", texts[1])
            self.assertTrue(all(text.count("LTLSPEC ") == 12 for text in texts))
            self.assertTrue(all("init(state) := {s0, s1}" in t for t in texts))

    def test_tampered_or_incomplete_graph_exports_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "input"
            make_run(run)
            graphs = run / "relations.jsonl"
            original = graphs.read_text(encoding="utf-8")
            graphs.write_text(original + original, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                read_run(run)
            report_path = run / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["exports"]["relations.jsonl"] = digest(graphs)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identities"):
                read_run(run)
            record = json.loads(original)
            record["graphs"]["top1"].pop()
            graphs.write_text(json.dumps(record) + "\n", encoding="utf-8")
            report["exports"]["relations.jsonl"] = digest(graphs)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dimensions"):
                read_run(run)

    def test_missing_backend_or_partial_verdicts_cannot_report_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, output = Path(temporary) / "input", Path(temporary) / "output"
            make_run(run)
            with patch(
                "experiments.evaluate_top1_ltl.find_nusmv_executable", return_value=None
            ):
                with self.assertRaisesRegex(FileNotFoundError, "NUXMV_BINARY"):
                    run_experiment(run, output)
            self.assertFalse(output.exists())
            with patch(
                "experiments.evaluate_top1_ltl.run_nusmv_model",
                return_value=NuSMVRun(sys.executable, (True,), "incomplete"),
            ):
                with self.assertRaisesRegex(RuntimeError, "expected 12 LTL verdicts"):
                    run_experiment(run, output, sys.executable)
            self.assertFalse((output / "report.json").exists())

    @unittest.skipUnless(find_nusmv_executable(), "nuXmv/NuSMV is not installed")
    def test_installed_backend_checks_saved_graphs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, output = Path(temporary) / "input", Path(temporary) / "output"
            make_run(run)
            with contextlib.redirect_stdout(io.StringIO()):
                report = run_experiment(run, output)
            self.assert_results(report, output)


if __name__ == "__main__":
    unittest.main()
