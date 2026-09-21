from __future__ import annotations

import unittest

import torch

from jepa_lmc.envs.gridworld import GridWorld
from jepa_lmc.evaluation.latent_radius import (
    LatentDistances,
    aggregate_candidate_results,
    candidate_metrics,
    candidate_transition_system,
    collect_latent_distances,
    evaluate_candidates,
    oracle_radii,
    summarize_ctl,
    truth_monotonicity,
)
from jepa_lmc.learning.model import ActionJEPA
from jepa_lmc.verification.ctl import AG, EF, Atom, Not


def make_env() -> GridWorld:
    return GridWorld(
        width=2,
        height=2,
        start=(0, 0),
        goal=(1, 1),
        walls=set(),
        dangers={(0, 1)},
    )


def ideal_table(env: GridWorld) -> LatentDistances:
    states = tuple(env.all_states())
    pairs = tuple((state, action) for state in states for action in env.ACTIONS)
    indices = tuple(states.index(env.transition(s, a)) for s, a in pairs)
    distances = torch.full((len(pairs), len(states)), 2.0, dtype=torch.float64)
    distances[torch.arange(len(pairs)), list(indices)] = 0.0
    return LatentDistances(states, pairs, indices, distances)


class RadiusGeometryTests(unittest.TestCase):
    def test_perfect_zero_radius_reconstructs_exact_graph(self) -> None:
        env = make_env()
        table = ideal_table(env)
        self.assertEqual(
            oracle_radii([table]),
            {
                "epsilon_max": 0.0,
                "epsilon_95": 0.0,
                "epsilon_99": 0.0,
            },
        )
        result = evaluate_candidates(env, table, table.candidates(0.0))
        self.assertEqual(result["successor_coverage"], 1.0)
        self.assertEqual(result["mean_candidate_size"], 1.0)
        self.assertEqual(result["singleton_fraction"], 1.0)
        self.assertTrue(result["relation_equal"])
        self.assertEqual(result["ctl"]["agreement"], 1.0)
        self.assertEqual(result["ctl"]["one_sided_violations"], 0)

    def test_quantiles_are_pooled_linear_and_maximum_covers_boundary(self) -> None:
        env = make_env()
        table = ideal_table(env)
        errors = torch.arange(len(table.pairs), dtype=torch.float64)
        table.distances[torch.arange(len(table.pairs)), list(table.true_indices)] = (
            errors
        )
        radii = oracle_radii([table])
        self.assertEqual(radii["epsilon_max"], 15.0)
        self.assertAlmostEqual(radii["epsilon_95"], 14.25)
        self.assertAlmostEqual(radii["epsilon_99"], 14.85)
        result = candidate_metrics(env, table, table.candidates(radii["epsilon_max"]))
        self.assertEqual(result["successor_coverage"], 1.0)
        self.assertTrue(result["action_inclusion"])
        # One small map of zeros must not receive the same weight as this map.
        small = GridWorld(1, 1, (0, 0), (0, 0), set(), set())
        self.assertAlmostEqual(
            oracle_radii([table, ideal_table(small)])["epsilon_95"], 14.05
        )

    def test_latent_collisions_keep_every_candidate_and_exact_labels(self) -> None:
        env = make_env()
        table = ideal_table(env)
        table.distances.zero_()
        result = evaluate_candidates(env, table, table.candidates(0.0))
        self.assertEqual(result["mean_candidate_size"], 4.0)
        self.assertEqual(result["singleton_fraction"], 0.0)
        self.assertIsNone(result["singleton_precision"])
        self.assertTrue(result["action_inclusion"])
        self.assertFalse(result["relation_equal"])
        graph = candidate_transition_system(env, table, table.candidates(0.0))
        self.assertIn("danger", graph.propositions((0, 1)))
        self.assertNotIn("danger", graph.propositions((0, 0)))

    def test_action_coverage_and_unlabelled_inclusion_are_distinct(self) -> None:
        env = make_env()
        table = ideal_table(env)
        mask = table.candidates(0.0)
        # At the top-left corner both up and left produce the same self-loop.
        mask[table.pairs.index(((0, 0), 0))] = False
        result = evaluate_candidates(env, table, mask)
        self.assertFalse(result["action_inclusion"])
        self.assertTrue(result["relation_inclusion"])
        self.assertEqual(result["ctl_status"], "evaluated")
        self.assertEqual(result["empty_count"], 1)

    def test_empty_candidates_are_recorded_without_repair(self) -> None:
        env = make_env()
        table = ideal_table(env)
        mask = torch.zeros_like(table.distances, dtype=torch.bool)
        result = evaluate_candidates(env, table, mask)
        self.assertEqual(result["empty_fraction"], 1.0)
        self.assertEqual(result["successor_coverage"], 0.0)
        self.assertEqual(result["ctl_status"], "not_total")
        self.assertEqual(result["ctl"]["queries"], 0)
        self.assertIsNone(result["ctl"]["agreement"])
        with self.assertRaisesRegex(ValueError, "no successor"):
            candidate_transition_system(env, table, mask)

    def test_invalid_numeric_inputs_and_incomplete_tables_fail(self) -> None:
        env = make_env()
        table = ideal_table(env)
        for epsilon in (-1.0, float("nan"), float("inf")):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                table.candidates(epsilon)
        for value in (float("nan"), float("inf"), -1.0):
            bad = table.distances.clone()
            bad[0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                LatentDistances(table.states, table.pairs, table.true_indices, bad)
        incomplete = LatentDistances(
            table.states,
            table.pairs[:-1],
            table.true_indices[:-1],
            table.distances[:-1],
        )
        with self.assertRaisesRegex(ValueError, "entire environment"):
            evaluate_candidates(env, incomplete, incomplete.candidates(0.0))


class RadiusCTLTests(unittest.TestCase):
    def test_overapproximation_allows_existential_false_positives(self) -> None:
        env = GridWorld(
            width=3,
            height=2,
            start=(0, 0),
            goal=(1, 0),
            walls={(0, 1), (1, 1)},
            dangers={(0, 2)},
        )
        table = ideal_table(env)
        mask = table.candidates(0.0)
        mask[table.pairs.index(((0, 0), 0)), table.states.index((0, 2))] = True
        result = evaluate_candidates(env, table, mask)
        self.assertTrue(result["relation_inclusion"])
        self.assertGreater(result["ctl_by_property"]["EF danger"]["false_positive"], 0)
        self.assertGreater(result["ctl_by_property"]["AG !danger"]["false_negative"], 0)
        self.assertEqual(result["ctl"]["one_sided_violations"], 0)
        self.assertEqual(result["ctl"]["false_safe_count"], 0)

    def test_missing_edges_can_make_universal_claims_unsound(self) -> None:
        env = make_env()
        table = ideal_table(env)
        mask = torch.zeros_like(table.distances, dtype=torch.bool)
        for row, (state, _) in enumerate(table.pairs):
            mask[row, table.states.index(state)] = True
        result = evaluate_candidates(env, table, mask)
        self.assertFalse(result["relation_inclusion"])
        self.assertGreater(result["ctl"]["false_safe_count"], 0)
        self.assertGreater(result["ctl"]["one_sided_violations"], 0)

    def test_mixed_quantifiers_abstain_and_negation_reverses_direction(self) -> None:
        proposition = Atom("goal")
        self.assertEqual(truth_monotonicity(AG(proposition)), -1)
        self.assertEqual(truth_monotonicity(EF(proposition)), 1)
        self.assertEqual(truth_monotonicity(Not(EF(proposition))), -1)
        self.assertIsNone(truth_monotonicity(AG(EF(proposition))))
        self.assertIsNone(summarize_ctl([])["empirical_one_sided_soundness"])

    def test_aggregation_weights_pairs_and_exposes_skipped_maps(self) -> None:
        large = make_env()
        large_table = ideal_table(large)
        small = GridWorld(1, 1, (0, 0), (0, 0), set(), set())
        small_table = ideal_table(small)
        results = [
            evaluate_candidates(large, large_table, large_table.candidates(0.0)),
            evaluate_candidates(
                small,
                small_table,
                torch.zeros_like(small_table.distances, dtype=torch.bool),
            ),
        ]
        summary = aggregate_candidate_results(results)
        self.assertEqual(summary["successor_coverage"], 0.8)
        self.assertEqual(summary["mean_candidate_size"], 0.8)
        self.assertEqual(summary["ctl_evaluated_maps"], 1)
        self.assertEqual(summary["ctl_skipped_maps"], 1)
        self.assertEqual(summary["ctl"]["queries"], 24)


class FrozenModelTests(unittest.TestCase):
    def test_evaluation_uses_target_encoder_and_preserves_weights_and_modes(
        self,
    ) -> None:
        torch.manual_seed(3)
        model = ActionJEPA(height=2, width=2, latent_dim=8, hidden_channels=4)
        # Deliberately separate target/context: using the wrong encoder must fail.
        with torch.no_grad():
            model.target_encoder.features[-1].bias.add_(5.0)
            model.predictor.network[-1].weight.zero_()
            model.predictor.network[-1].bias.zero_()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        modes = [module.training for module in model.modules()]
        calls = []
        handle = model.target_encoder.register_forward_hook(
            lambda _module, _inputs, output: calls.append(output.detach().clone())
        )
        try:
            table = collect_latent_distances(model, make_env())
        finally:
            handle.remove()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(table.pairs), 16)
        # Up at (0, 0) is a self-loop; only the target's bias differs, by 5.
        self.assertAlmostEqual(float(table.errors[0]), 5.0, places=6)
        self.assertEqual(table.distances.dtype, torch.float64)
        self.assertFalse(table.distances.requires_grad)
        self.assertEqual(modes, [module.training for module in model.modules()])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        repeated = collect_latent_distances(model, make_env())
        self.assertTrue(torch.equal(table.distances, repeated.distances))


if __name__ == "__main__":
    unittest.main()
