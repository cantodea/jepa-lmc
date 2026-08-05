from __future__ import annotations

import unittest

import torch

from jepa_lmc.envs.gridworld import GridWorld
from jepa_lmc.evaluation.dynamics import (
    RolloutReport,
    RolloutRetrieval,
    TransitionRetrieval,
    TransitionRetrievalReport,
    evaluate_gridworld_transitions,
    evaluate_open_loop_rollouts,
    transition_system_from_retrievals,
)
from jepa_lmc.learning.model import ActionJEPA


def make_env() -> GridWorld:
    return GridWorld(
        width=4,
        height=4,
        start=(0, 0),
        goal=(3, 3),
        walls={(1, 1)},
        dangers={(1, 2)},
    )


def exact_retrieval_report(env: GridWorld) -> TransitionRetrievalReport:
    outcomes = []
    for transition in env.all_transitions():
        next_state = transition.next_state
        outcomes.append(
            TransitionRetrieval(
                state=transition.state,
                action=transition.action,
                true_next_state=next_state,
                predicted_next_state=next_state,
                true_rank=1,
                predicted_distance=0.0,
                true_distance=0.0,
                kind="self_loop" if transition.state == next_state else "movement",
                true_next_is_danger=env.is_danger(next_state),
                predicted_next_is_danger=env.is_danger(next_state),
            )
        )
    return TransitionRetrievalReport(tuple(outcomes))


class TransitionReportTests(unittest.TestCase):
    def test_retrieval_metrics_include_rank_and_safety(self) -> None:
        report = TransitionRetrievalReport(
            (
                TransitionRetrieval(
                    (0, 0),
                    0,
                    (0, 0),
                    (0, 0),
                    1,
                    0.1,
                    0.1,
                    "self_loop",
                    False,
                    False,
                ),
                TransitionRetrieval(
                    (0, 0),
                    1,
                    (1, 0),
                    (0, 1),
                    2,
                    0.2,
                    0.3,
                    "danger_entry",
                    True,
                    False,
                ),
            )
        )

        self.assertEqual(report.top1_accuracy, 0.5)
        self.assertEqual(report.top_k_accuracy(2), 1.0)
        self.assertEqual(report.mean_reciprocal_rank, 0.75)
        self.assertEqual(report.unsafe_miss_count, 1)
        self.assertEqual(report.danger_destination_recall, 0.0)

    def test_exact_retrievals_reconstruct_the_real_graph(self) -> None:
        env = make_env()
        report = exact_retrieval_report(env)
        learned = transition_system_from_retrievals(env, report)

        for state in env.all_states():
            expected = {env.transition(state, action) for action in env.ACTIONS}
            self.assertEqual(learned.successors(state), expected)

    def test_missing_transition_is_rejected(self) -> None:
        env = make_env()
        report = exact_retrieval_report(env)
        incomplete = TransitionRetrievalReport(report.outcomes[:-1])

        with self.assertRaisesRegex(ValueError, "do not match"):
            transition_system_from_retrievals(env, incomplete)


class JEPADynamicsEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.env = make_env()
        self.model = ActionJEPA(
            height=4,
            width=4,
            latent_dim=8,
            hidden_channels=4,
            action_dim=4,
            predictor_hidden_dim=12,
        )

    def test_untrained_model_still_produces_complete_ranked_graph(self) -> None:
        report = evaluate_gridworld_transitions(self.model, self.env)

        self.assertEqual(
            report.total,
            len(self.env.all_states()) * len(self.env.ACTIONS),
        )
        self.assertTrue(all(outcome.true_rank >= 1 for outcome in report.outcomes))
        learned = transition_system_from_retrievals(self.env, report)
        self.assertEqual(learned.states, frozenset(self.env.all_states()))

    def test_action_override_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "action override"):
            evaluate_gridworld_transitions(
                self.model,
                self.env,
                action_override=99,
            )

    def test_open_loop_rollouts_report_each_requested_horizon(self) -> None:
        report = evaluate_open_loop_rollouts(
            self.model,
            self.env,
            horizons=(1, 3),
            rollouts_per_state=1,
            seed=5,
        )

        self.assertEqual(set(report.accuracy_by_horizon), {1, 3})
        self.assertEqual(
            len(report.outcomes),
            len(self.env.all_states()) * 2,
        )

    def test_rollout_report_aggregates_by_horizon(self) -> None:
        report = RolloutReport(
            (
                RolloutRetrieval((0, 0), (0,), 1, (0, 0), (0, 0)),
                RolloutRetrieval((0, 0), (0, 1), 2, (1, 0), (0, 0)),
            )
        )
        self.assertEqual(report.accuracy_by_horizon, {1: 1.0, 2: 0.0})


if __name__ == "__main__":
    unittest.main()
