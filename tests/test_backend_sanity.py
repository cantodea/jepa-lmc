from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.backend_sanity import (
    audit_encoding,
    build_model,
    parse_verdicts,
    run_audit,
)
from jepa_lmc.verification.nuxmv import find_nusmv_executable
from tests.test_top1_ltl_experiment import make_run


def make_sanity_run(directory: Path) -> None:
    make_run(directory)
    path = directory / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["maps"][0]["initial_ctl_disagreements"] = [
        "EF danger",
        "AG !danger",
        "EG safe",
    ]
    path.write_text(json.dumps(report), encoding="utf-8")


class BackendSanityTests(unittest.TestCase):
    def test_encoding_audit_catches_graph_label_initial_and_fairness_changes(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            make_sanity_run(directory)
            record = json.loads((directory / "relations.jsonl").read_text())
            graph, model, queries, states, labels = build_model(record, "real")
        self.assertEqual(model.count("MODULE main"), 1)
        self.assertEqual(model.count("init(state)"), 1)
        self.assertEqual(len(queries), 16)
        self.assertEqual(sum(q["logic"] == "ctl" for q in queries), 12)
        self.assertEqual(
            {q["formula"] for q in queries if q["logic"] == "ltl"},
            {"G !danger", "F goal"},
        )
        alternatives = (
            model.replace("state = s0 : {s1};", "state = s0 : {s0};"),
            model.replace("ap0 := state = s1;", "ap0 := FALSE;"),
            model.replace("init(state) := {s0, s1};", "init(state) := {s0};"),
            model + "FAIRNESS TRUE\n",
        )
        for alternative in alternatives:
            self.assertNotEqual(model, alternative)
            with self.assertRaises(AssertionError):
                audit_encoding(graph, alternative, states, labels)

    def test_formula_matching_accepts_reordered_results_and_rejects_bad_coverage(self):
        model = """SPEC NAME ctl_0 := ((state = s0) -> (AG (!(ap0))))
LTLSPEC NAME ltl_0 := ((state = s0) -> (G (!(ap0))))
"""
        output = """-- specification (state = s0 -> G !ap0) is false
-- specification (state = s0 -> AG !ap0) is true
"""
        self.assertEqual(parse_verdicts(model, output), {"ctl_0": True, "ltl_0": False})
        for wrong in (
            output.splitlines()[0],
            output + output,
            output.replace("state = s0", "state = s1"),
        ):
            with self.assertRaises(ValueError):
                parse_verdicts(model, wrong)

    @unittest.skipUnless(find_nusmv_executable(), "nuXmv is not installed")
    def test_actual_backend_all_existing_ctl_and_paired_ltl(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, output = Path(temporary) / "run", Path(temporary) / "output"
            make_sanity_run(run)
            with contextlib.redirect_stdout(io.StringIO()):
                report = run_audit(run, output)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["internal_ctl_vs_nuxmv_ctl"]["comparisons"], 24)
            self.assertEqual(report["internal_ctl_vs_nuxmv_ctl"]["agreement"], 1)
            self.assertEqual(report["external_ltl_verdicts"], 8)
            self.assertTrue(
                all(p["agreement"] == 1 for p in report["paired_formulas"].values())
            )
            self.assertEqual(report["initial_all_six"]["saved"], 0)
            self.assertEqual(report["initial_all_six"]["nuxmv"], 0)
            self.assertEqual(report["initial_all_six"]["changed_formula_agreements"], 0)
            self.assertEqual(report["mismatch_count"], 0)

    @unittest.skipUnless(find_nusmv_executable(), "nuXmv is not installed")
    def test_one_corrupted_verdict_is_reported_even_if_all_six_map_count_stays_equal(
        self,
    ):
        original_run = subprocess.run

        def corrupt_one(*args, **kwargs):
            result = original_run(*args, **kwargs)
            if str(args[0][-1]).endswith("000_real.smv"):
                lines = result.stdout.splitlines()
                affected = [
                    i
                    for i, line in enumerate(lines)
                    if "specification (state = s0 -> AG !ap0)" in line
                ]
                self.assertEqual(len(affected), 1)
                index = affected[0]
                self.assertIn("is false", lines[index])
                lines[index] = lines[index].replace("is false", "is true")
                return subprocess.CompletedProcess(
                    result.args, result.returncode, "\n".join(lines), result.stderr
                )
            return result

        with tempfile.TemporaryDirectory() as temporary:
            run, output = Path(temporary) / "run", Path(temporary) / "output"
            make_sanity_run(run)
            with (
                patch(
                    "experiments.backend_sanity.subprocess.run", side_effect=corrupt_one
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = run_audit(run, output)
            self.assertEqual(report["status"], "mismatch")
            self.assertEqual(report["mismatch_count"], 3)
            self.assertEqual(report["initial_all_six"]["changed_maps"], 0)
            self.assertEqual(report["initial_all_six"]["changed_formula_agreements"], 1)
            self.assertEqual(report["internal_ctl_vs_nuxmv_ctl"]["mismatches"], 1)
            self.assertEqual(
                report["paired_formulas"]["AG !danger <=> G !danger"]["mismatches"], 1
            )


if __name__ == "__main__":
    unittest.main()
