from __future__ import annotations

import unittest
from unittest.mock import patch

from jepa_lmc.benchmarks.ltl_suite import LTLProperty, default_ltl_suite
from jepa_lmc.evaluation.metrics import (
    evaluate_ltl_suite_for_state_pairs,
)
from jepa_lmc.verification.ltl import (
    Atom,
    Eventually,
    Globally,
    Next,
    Not,
    Until,
    atoms,
)
from jepa_lmc.verification.nuxmv import NuSMVLTLReport
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem


def make_system(target: str) -> ExplicitTransitionSystem[str, str]:
    return ExplicitTransitionSystem(
        states={"start", "goal", "danger"},
        initial_states={"start"},
        transitions={
            "start": [("move", target)],
            "goal": [("stay", "goal")],
            "danger": [("stay", "danger")],
        },
        labels={
            "start": {"safe"},
            "goal": {"goal"},
            "danger": {"danger"},
        },
    )


class LTLFormulaTests(unittest.TestCase):
    def test_nested_formula_reports_all_atoms(self) -> None:
        formula = Globally(
            Eventually(
                Until(Not(Atom("danger")), Atom("goal")),
            )
        )

        self.assertEqual(atoms(formula), frozenset({"danger", "goal"}))

    def test_empty_atomic_proposition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have a name"):
            Atom("")

    def test_default_suite_contains_short_and_long_horizon_properties(self) -> None:
        suite = default_ltl_suite()

        self.assertEqual(
            len({property_spec.name for property_spec in suite}), len(suite)
        )
        self.assertTrue(any(isinstance(item.formula, Next) for item in suite))
        self.assertTrue(
            any(
                isinstance(item.formula, Globally)
                and isinstance(item.formula.formula, Eventually)
                for item in suite
            )
        )
        diagnostic_names = {item.name for item in suite if not item.primary_score}
        self.assertEqual(diagnostic_names, {"F goal", "G !danger"})


class LTLEvaluationTests(unittest.TestCase):
    @patch("jepa_lmc.evaluation.metrics.evaluate_ltl_queries_with_nusmv")
    def test_external_verdicts_use_shared_property_scorecard(self, mock_run) -> None:
        ground_truth = make_system("danger")
        learned = make_system("goal")
        properties = (
            LTLProperty(
                "X !danger",
                "safety",
                Next(Not(Atom("danger"))),
                safety_claim=True,
            ),
            LTLProperty(
                "F goal",
                "reachability",
                Eventually(Atom("goal")),
            ),
        )

        def fake_run(system, queries, executable=None):
            del executable
            verdicts = (False, False) if system is ground_truth else (True, True)
            return NuSMVLTLReport(
                queries=tuple(queries),
                verdicts=verdicts,
                executable="nuXmv",
                output="",
            )

        mock_run.side_effect = fake_run
        report = evaluate_ltl_suite_for_state_pairs(
            ground_truth,
            learned,
            (("start", "start"),),
            properties,
        )

        self.assertEqual(report.total, 2)
        self.assertEqual(report.agreement, 0.0)
        self.assertEqual(report.false_safe_count, 1)
        self.assertEqual(mock_run.call_count, 2)

    def test_empty_ltl_property_suite_is_rejected(self) -> None:
        system = make_system("goal")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            evaluate_ltl_suite_for_state_pairs(
                system,
                system,
                (("start", "start"),),
                (),
            )


if __name__ == "__main__":
    unittest.main()
