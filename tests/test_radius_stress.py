from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jepa_lmc.benchmarks.radius_stress import (
    PRIMARY_PROPERTIES,
    exact_transfer_opportunity,
    label_immediate,
    radius_stress_cases,
)
from jepa_lmc.evaluation.latent_radius import ctl_outcomes
from jepa_lmc.evaluation.radius_stress import (
    evaluate_overall_gate,
    evaluate_seed_gate,
    score_nontrivial,
)
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


class RadiusStressScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.positive_rows = []
        cls.negative_rows = []
        for case in radius_stress_cases():
            env = case.make_env()
            exact = gridworld_to_transition_system(env)
            complete = ExplicitTransitionSystem(
                states=exact.states,
                initial_states=exact.initial_states,
                transitions={s: [(0, t) for t in exact.states] for s in exact.states},
                labels={s: exact.propositions(s) for s in exact.states},
            )
            for model, destination in (
                (exact, cls.positive_rows),
                (complete, cls.negative_rows),
            ):
                destination.extend(
                    {
                        **row,
                        "case": case.name,
                        "family": case.family,
                        "relation_inclusion": True,
                        "label_immediate": label_immediate(
                            env, row["state"], row["property"]
                        ),
                    }
                    for row in ctl_outcomes(exact, model)
                )
        cls.protocol = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "configs/latent_radius_stress_protocol.json"
            ).read_text()
        )

    def test_controls_have_nontrivial_denominators_and_separate_scores(self) -> None:
        positive = score_nontrivial(self.positive_rows)
        negative = score_nontrivial(self.negative_rows)
        self.assertEqual(positive["queries"], 2136)
        self.assertEqual(positive["transfer_opportunities"], 480)
        self.assertEqual(positive["primary_balanced_score"], 1)
        self.assertEqual(positive["opportunity_recall"], 1)
        self.assertEqual(negative["queries"], 2136)
        self.assertEqual(negative["transfer_opportunities"], 480)
        self.assertEqual(negative["primary_balanced_score"], 0.5)
        self.assertEqual(negative["opportunity_recall"], 0)
        self.assertEqual(negative["maps_with_recovered_opportunity"], [])
        counts = {
            family: score_nontrivial(
                row for row in self.positive_rows if row["family"] == family
            )["transfer_opportunities"]
            for family in ("sealed_region", "danger_gate", "safe_detour")
        }
        self.assertEqual(
            counts, {"sealed_region": 360, "danger_gate": 120, "safe_detour": 0}
        )

    def test_correct_predictions_without_inclusion_are_not_proofs(self) -> None:
        score = score_nontrivial(
            {**row, "relation_inclusion": False} for row in self.positive_rows
        )
        self.assertEqual(score["primary_balanced_score"], 1)
        self.assertEqual(score["recovered_opportunities"], 0)
        self.assertEqual(score["opportunity_recall"], 0)

    def threshold_summaries(self) -> dict:
        return {
            "epsilon_max": {
                "ctl_evaluated_maps": 24,
                "successor_coverage": 1,
                "relation_inclusion_maps": 24,
                "ctl": {"one_sided_violations": 0},
                "nontrivial": {
                    "primary_balanced_score": 0.6,
                    "opportunity_recall": 0.1,
                },
                "nontrivial_by_family": {
                    family: {"maps_with_recovered_opportunity": list(range(4))}
                    for family in ("sealed_region", "danger_gate")
                },
            },
            "all_states": {"nontrivial": {"primary_balanced_score": 0.5}},
        }

    def test_inclusive_thresholds_and_each_required_gate(self) -> None:
        summaries = self.threshold_summaries()
        self.assertTrue(evaluate_seed_gate(summaries, self.protocol)["passes"])
        mutations = (
            (("ctl_evaluated_maps",), 23, "all_maps_evaluated"),
            (("successor_coverage",), 0.999, "full_action_coverage"),
            (("relation_inclusion_maps",), 23, "full_relation_inclusion"),
            (("ctl", "one_sided_violations"), 1, "zero_one_sided_violations"),
            (("nontrivial", "primary_balanced_score"), 0.599, "balanced_gain"),
            (("nontrivial", "primary_balanced_score"), None, "balanced_gain"),
            (("nontrivial", "opportunity_recall"), 0.099, "opportunity_recall"),
            (
                (
                    "nontrivial_by_family",
                    "danger_gate",
                    "maps_with_recovered_opportunity",
                ),
                list(range(3)),
                "family_consistency",
            ),
        )
        for path, value, failed_check in mutations:
            with self.subTest(check=failed_check, value=value):
                changed = copy.deepcopy(summaries)
                # An excellent diagnostic quantile cannot override the max gate.
                changed["epsilon_95"] = copy.deepcopy(changed["epsilon_max"])
                target = changed["epsilon_max"]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                gate = evaluate_seed_gate(changed, self.protocol)
                self.assertFalse(gate["passes"])
                self.assertIn(failed_check, gate["failed_checks"])

    def test_all_prespecified_seeds_must_pass(self) -> None:
        gates = {seed: {"passes": True} for seed in self.protocol["model_seeds"]}
        self.assertEqual(
            evaluate_overall_gate(gates, self.protocol)["status"], "continue"
        )
        gates[20260805] = {"passes": False}
        decision = evaluate_overall_gate(gates, self.protocol)
        self.assertEqual(decision["status"], "stop")
        self.assertEqual(decision["failed_seeds"], [20260805])
        gates.pop(20260805)
        with self.assertRaises(ValueError):
            evaluate_overall_gate(gates, self.protocol)


if __name__ == "__main__":
    unittest.main()
