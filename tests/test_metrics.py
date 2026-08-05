from __future__ import annotations

import unittest

from jepa_lmc.benchmarks.ctl_suite import default_ctl_suite
from jepa_lmc.benchmarks.perturbations import block_entries_to_proposition
from jepa_lmc.evaluation.metrics import (
    aggregate_reports,
    evaluate_ctl_suite,
    evaluate_ctl_suite_for_state_pairs,
)
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem


def make_safety_system() -> ExplicitTransitionSystem[str, str]:
    return ExplicitTransitionSystem(
        states={"s0", "loop", "goal", "danger"},
        initial_states={"s0"},
        transitions={
            "s0": [
                ("wait", "loop"),
                ("finish", "goal"),
                ("fail", "danger"),
            ],
            "loop": [("wait", "loop")],
            "goal": [("stay", "goal")],
            "danger": [("stay", "danger")],
        },
        labels={
            "s0": {"safe"},
            "loop": {"safe"},
            "goal": {"safe", "goal"},
            "danger": {"danger"},
        },
    )


class VerificationMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ground_truth = make_safety_system()

    def test_identical_model_scores_one_hundred_percent(self) -> None:
        report = evaluate_ctl_suite(
            self.ground_truth,
            self.ground_truth,
            "s0",
            "s0",
        )

        self.assertEqual(report.total, 6)
        self.assertEqual(report.agreement, 1.0)
        self.assertEqual(report.false_safe_count, 0)
        self.assertEqual(report.unsafe_miss_rate, 0.0)
        self.assertTrue(
            all(score == 1.0 for score in report.agreement_by_category.values())
        )

    def test_danger_blind_model_is_penalized_as_false_safe(self) -> None:
        learned = block_entries_to_proposition(self.ground_truth, "danger")
        report = evaluate_ctl_suite(
            self.ground_truth,
            learned,
            "s0",
            "s0",
        )

        outcomes = {outcome.name: outcome for outcome in report.outcomes}
        self.assertFalse(outcomes["EF danger"].matches)
        self.assertTrue(outcomes["AG !danger"].false_safe)
        self.assertEqual(report.agreement, 4 / 6)
        self.assertEqual(report.false_safe_count, 1)
        self.assertEqual(report.unsafe_miss_rate, 1.0)

    def test_reports_can_be_aggregated_across_maps(self) -> None:
        perfect = evaluate_ctl_suite(
            self.ground_truth,
            self.ground_truth,
            "s0",
            "s0",
        )
        danger_blind = evaluate_ctl_suite(
            self.ground_truth,
            block_entries_to_proposition(self.ground_truth, "danger"),
            "s0",
            "s0",
        )

        aggregate = aggregate_reports([perfect, danger_blind])
        self.assertEqual(aggregate.total, 12)
        self.assertEqual(aggregate.agreement, 10 / 12)
        self.assertEqual(aggregate.false_safe_count, 1)
        self.assertEqual(aggregate.unsafe_miss_rate, 0.5)

    def test_many_states_share_one_pair_of_model_checkers(self) -> None:
        report = evaluate_ctl_suite_for_state_pairs(
            self.ground_truth,
            self.ground_truth,
            tuple((state, state) for state in self.ground_truth.states),
        )

        self.assertEqual(report.total, 24)
        self.assertEqual(report.agreement, 1.0)
        self.assertEqual(report.primary_balanced_score, 1.0)
        self.assertEqual(
            set(report.agreement_by_property.values()), {1.0}
        )

    def test_primary_score_excludes_duplicate_and_diagnostic_formulae(self) -> None:
        primary_names = {
            property_spec.name
            for property_spec in default_ctl_suite()
            if property_spec.primary_score
        }
        self.assertEqual(
            primary_names,
            {"EF goal", "E[!danger U goal]", "AG !danger"},
        )

    def test_confusion_metrics_penalize_false_safe_results(self) -> None:
        learned = block_entries_to_proposition(self.ground_truth, "danger")
        report = evaluate_ctl_suite_for_state_pairs(
            self.ground_truth,
            learned,
            tuple((state, state) for state in self.ground_truth.states),
        )
        safety = report.confusion_by_property["AG !danger"]

        self.assertGreater(safety.false_positive, 0)
        self.assertIsNotNone(safety.balanced_accuracy)
        assert safety.balanced_accuracy is not None
        self.assertLess(safety.balanced_accuracy, 1.0)
        self.assertIsNotNone(report.primary_balanced_score)
        assert report.primary_balanced_score is not None
        self.assertLess(report.primary_balanced_score, 1.0)

    def test_empty_state_pairs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "state pair"):
            evaluate_ctl_suite_for_state_pairs(
                self.ground_truth,
                self.ground_truth,
                (),
            )

    def test_empty_property_suite_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            evaluate_ctl_suite(
                self.ground_truth,
                self.ground_truth,
                "s0",
                "s0",
                properties=(),
            )


if __name__ == "__main__":
    unittest.main()
