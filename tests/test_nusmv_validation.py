from __future__ import annotations

import unittest

from jepa_lmc.checking.ctl import AG, EF, EU, And, Atom, Not
from jepa_lmc.checking.transition_system import ExplicitTransitionSystem
from jepa_lmc.validation.nusmv import (
    CTLQuery,
    compare_queries_with_nusmv,
    export_nusmv_queries,
    find_nusmv_executable,
    formula_to_nusmv,
    parse_nusmv_verdicts,
)


def make_reference_system() -> ExplicitTransitionSystem[str, str]:
    return ExplicitTransitionSystem(
        states={"start", "goal"},
        initial_states={"start"},
        transitions={
            "start": [("finish", "goal")],
            "goal": [("stay", "goal")],
        },
        labels={
            "start": {"safe"},
            "goal": {"safe", "goal"},
        },
    )


class NuSMVValidationTests(unittest.TestCase):
    def test_supported_formulae_translate_to_nusmv_syntax(self) -> None:
        safe = Atom("safe")
        goal = Atom("goal")

        self.assertEqual(formula_to_nusmv(EF(goal)), "EF (goal)")
        self.assertEqual(
            formula_to_nusmv(AG(Not(Atom("danger")))),
            "AG (!(danger))",
        )
        self.assertEqual(
            formula_to_nusmv(EU(And(safe, Not(goal)), goal)),
            "E[((safe & !(goal))) U (goal)]",
        )

    def test_invalid_unmapped_proposition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid NuSMV identifier"):
            formula_to_nusmv(Atom("not a legal identifier"))

    def test_export_encodes_state_specific_queries_in_one_model(self) -> None:
        system = make_reference_system()
        model = export_nusmv_queries(
            system,
            (
                CTLQuery("start", EF(Atom("goal"))),
                CTLQuery("goal", AG(Atom("goal"))),
            ),
        )

        self.assertIn("state : {s0, s1};", model)
        self.assertIn("init(state) := {s0, s1};", model)
        self.assertIn("state = s1 : {s0};", model)
        self.assertIn("state = s0 : {s0};", model)
        self.assertIn("ap0 := state = s0;", model)
        self.assertIn("SPEC ((state = s1) -> (EF (ap0)))", model)
        self.assertIn("SPEC ((state = s0) -> (AG (ap0)))", model)

    def test_export_rejects_empty_queries(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            export_nusmv_queries(make_reference_system(), ())

    def test_nusmv_verdicts_are_parsed_in_specification_order(self) -> None:
        output = """
-- specification EF ap0 is true
-- specification AG ap0 is false
"""
        self.assertEqual(parse_nusmv_verdicts(output), (True, False))

    def test_missing_verdicts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "did not contain"):
            parse_nusmv_verdicts("NuSMV finished without specifications")

    @unittest.skipUnless(
        find_nusmv_executable(),
        "nuXmv/NuSMV is not installed",
    )
    def test_installed_nusmv_agrees_on_reference_queries(self) -> None:
        queries = (
            CTLQuery("start", EF(Atom("goal"))),
            CTLQuery("start", AG(Atom("safe"))),
            CTLQuery("goal", AG(Atom("goal"))),
        )
        report = compare_queries_with_nusmv(
            make_reference_system(),
            queries,
        )

        self.assertTrue(report.all_match)
        self.assertEqual(report.agreement, 1.0)


if __name__ == "__main__":
    unittest.main()
