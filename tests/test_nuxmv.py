from __future__ import annotations

import unittest

from jepa_lmc.verification.ctl import AG, EF, EU, And, Atom, Not
from jepa_lmc.verification.ltl import (
    Atom as LTLAtom,
)
from jepa_lmc.verification.ltl import (
    Eventually,
    Globally,
    Next,
    Until,
)
from jepa_lmc.verification.ltl import (
    Not as LTLNot,
)
from jepa_lmc.verification.nuxmv import (
    CTLQuery,
    LTLQuery,
    compare_queries_with_nusmv,
    evaluate_ltl_queries_with_nusmv,
    export_nusmv_ltl_queries,
    export_nusmv_queries,
    find_nusmv_executable,
    formula_to_nusmv,
    ltl_formula_to_nusmv,
    parse_nusmv_verdicts,
)
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem


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

    def test_supported_ltl_formulae_translate_to_nusmv_syntax(self) -> None:
        safe = LTLAtom("safe")
        goal = LTLAtom("goal")

        self.assertEqual(
            ltl_formula_to_nusmv(Globally(Eventually(safe))),
            "G (F (safe))",
        )
        self.assertEqual(
            ltl_formula_to_nusmv(Next(LTLNot(goal))),
            "X (!(goal))",
        )
        self.assertEqual(
            ltl_formula_to_nusmv(Until(safe, goal)),
            "((safe) U (goal))",
        )

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

    def test_export_encodes_state_specific_ltl_queries(self) -> None:
        model = export_nusmv_ltl_queries(
            make_reference_system(),
            (
                LTLQuery("start", Eventually(LTLAtom("goal"))),
                LTLQuery("goal", Globally(LTLAtom("safe"))),
            ),
        )

        self.assertIn("LTLSPEC ((state = s1) -> (F (ap0)))", model)
        self.assertIn("LTLSPEC ((state = s0) -> (G (ap1)))", model)
        self.assertNotIn("\nSPEC ", model)

    def test_ltl_export_rejects_empty_queries(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one LTL"):
            export_nusmv_ltl_queries(make_reference_system(), ())

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

    @unittest.skipUnless(
        find_nusmv_executable(),
        "nuXmv/NuSMV is not installed",
    )
    def test_installed_nusmv_evaluates_reference_ltl_queries(self) -> None:
        report = evaluate_ltl_queries_with_nusmv(
            make_reference_system(),
            (
                LTLQuery("start", Eventually(LTLAtom("goal"))),
                LTLQuery("start", Globally(LTLAtom("safe"))),
                LTLQuery("start", Globally(LTLAtom("goal"))),
                LTLQuery("goal", Globally(LTLAtom("goal"))),
            ),
        )

        self.assertEqual(report.verdicts, (True, True, False, True))


if __name__ == "__main__":
    unittest.main()
