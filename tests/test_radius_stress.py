from __future__ import annotations

import unittest

from jepa_lmc.benchmarks.radius_stress import (
    PRIMARY_PROPERTIES,
    exact_transfer_opportunity,
    label_immediate,
    radius_stress_cases,
)
from jepa_lmc.evaluation.latent_radius import ctl_outcomes
from jepa_lmc.verification.ctl import AG, EF, EU, Atom, CTLModelChecker, Not
from jepa_lmc.verification.gridworld import gridworld_to_transition_system
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem


class RadiusStressTopologyTests(unittest.TestCase):
    def test_all_cases_have_prespecified_truth_and_multistep_routes(self) -> None:
        cases = radius_stress_cases()
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({case.name for case in cases}), 24)
        expected = {
            "sealed_region": (True, False, False),
            "danger_gate": (False, True, False),
            "safe_detour": (False, True, True),
        }
        formulas = (
            AG(Not(Atom("danger"))),
            EF(Atom("goal")),
            EU(Not(Atom("danger")), Atom("goal")),
        )
        for case in cases:
            with self.subTest(case=case.name):
                env = case.make_env()
                checker = CTLModelChecker(gridworld_to_transition_system(env))
                actual = tuple(checker.holds(env.start, f) for f in formulas)
                self.assertEqual(actual, expected[case.family])
                # No start/goal shortcut: Manhattan separation is ten steps.
                self.assertEqual(
                    sum(abs(a - b) for a, b in zip(env.start, env.goal)), 10
                )
                self.assertTrue(
                    all(
                        not label_immediate(env, env.start, name)
                        for name in PRIMARY_PROPERTIES
                    )
                )

    def test_positive_and_negative_controls_and_opportunity_denominators(self) -> None:
        opportunities = 0
        for case in radius_stress_cases():
            env = case.make_env()
            exact = gridworld_to_transition_system(env)
            complete = ExplicitTransitionSystem(
                states=exact.states,
                initial_states=exact.initial_states,
                transitions={s: [(0, t) for t in exact.states] for s in exact.states},
                labels={s: exact.propositions(s) for s in exact.states},
            )
            positive = ctl_outcomes(exact, exact)
            negative = ctl_outcomes(exact, complete)
            for perfect, uninformative in zip(positive, negative, strict=True):
                immediate = label_immediate(env, perfect["state"], perfect["property"])
                perfect["label_immediate"] = immediate
                uninformative["label_immediate"] = immediate
                if exact_transfer_opportunity(perfect):
                    opportunities += 1
                    self.assertTrue(perfect["one_sided_claim"])
                    self.assertFalse(uninformative["one_sided_claim"])
                    self.assertIn(case.family, ("sealed_region", "danger_gate"))
        self.assertEqual(opportunities, 480)

    def test_base_case_filter_handles_goal_as_safe(self) -> None:
        env = radius_stress_cases()[0].make_env()
        self.assertTrue(label_immediate(env, env.goal, "EF goal"))
        self.assertTrue(label_immediate(env, env.goal, "E[!danger U goal]"))
        self.assertFalse(label_immediate(env, env.goal, "AG !danger"))
        with self.assertRaises(ValueError):
            label_immediate(env, env.start, "made up")


if __name__ == "__main__":
    unittest.main()
