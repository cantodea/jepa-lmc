from __future__ import annotations

import unittest

import torch

from jepa_lmc.evaluation.latent_radius import LatentDistances
from jepa_lmc.evaluation.local_abstraction import (
    candidate_rows,
    identity_simulation,
    oracle_local_mask,
    ranking_rows,
    summarize_candidates,
)
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem


class LocalAbstractionTests(unittest.TestCase):
    def test_local_bound_retains_complete_boundary_ties_and_zero(self):
        table = LatentDistances(
            ((0, 0), (0, 1), (0, 2)),
            (((0, 0), 0), ((0, 1), 0), ((0, 2), 0)),
            (1, 2, 0),
            torch.tensor(
                [[0.0, 0.0, 3.0], [1.0, 2.0, 2.0], [1.0, 2.0, 4.0]], dtype=torch.float64
            ),
        )
        mask = oracle_local_mask(table)
        self.assertEqual(
            mask.tolist(),
            [[True, True, False], [True, True, True], [True, False, False]],
        )
        ranks = ranking_rows(table)
        self.assertEqual([r["true_rank"] for r in ranks], [2, 3, 1])
        self.assertEqual([r["true_distance_ties"] for r in ranks], [2, 2, 1])
        stats = summarize_candidates(candidate_rows(table, mask))
        self.assertEqual(stats["successor_coverage"], 1.0)
        self.assertEqual(stats["median_size"], 2)
        self.assertEqual(stats["max_size"], 3)
        self.assertEqual(stats["spurious_edge_rate"], 3 / 6)
        self.assertFalse(
            bool((mask & ~table.candidates(float(table.errors.max()))).any())
        )

    def test_spurious_denominator_counts_actions_and_missing_successors(self):
        table = LatentDistances(
            ((0, 0), (0, 1), (0, 2)),
            (((0, 0), 0), ((0, 0), 1)),
            (0, 1),
            torch.ones(2, 3),
        )
        mask = torch.tensor([[False, True, False], [True, True, True]])
        summary = summarize_candidates(candidate_rows(table, mask))
        self.assertEqual(summary["covered_pairs"], 1)
        self.assertEqual(summary["spurious_action_edges"], 3)
        self.assertEqual(summary["spurious_edge_rate"], 3 / 4)
        self.assertEqual(summary["mean_size"], 2)

    def test_identity_checks_action_matching_and_labels(self):
        def graph(edges, labels=None):
            return ExplicitTransitionSystem(
                states=(0, 1),
                initial_states=(0,),
                transitions=edges,
                labels=labels or {0: ("safe",), 1: ("safe",)},
            )

        real = graph({0: [(0, 0), (1, 1)], 1: [(0, 1), (1, 1)]})
        swapped = graph({0: [(0, 1), (1, 0)], 1: [(0, 1), (1, 1)]})
        candidate = graph({0: [(0, 0), (0, 1), (1, 0), (1, 1)], 1: [(0, 1), (1, 1)]})
        self.assertTrue(identity_simulation(real, candidate)["holds"])
        self.assertFalse(identity_simulation(real, swapped)["holds"])
        self.assertTrue(
            identity_simulation(real, swapped, action_sensitive=False)["holds"]
        )
        relabelled = graph(
            {0: [(0, 0), (0, 1), (1, 1)], 1: [(0, 1), (1, 1)]},
            {0: ("danger",), 1: ("safe",)},
        )
        self.assertFalse(identity_simulation(real, relabelled)["holds"])

    def test_oracle_local_is_minimal_in_closed_ball_family(self):
        generator = torch.Generator().manual_seed(8)
        distances = torch.rand(12, 3, generator=generator, dtype=torch.float64)
        pairs = tuple(((i // 4, 0), i % 4) for i in range(12))
        table = LatentDistances(
            ((0, 0), (1, 0), (2, 0)), pairs, tuple(i % 3 for i in range(12)), distances
        )
        mask = oracle_local_mask(table)
        for i, true in enumerate(table.true_indices):
            self.assertTrue(bool(mask[i, true]))
            lower_bound = torch.nextafter(
                table.errors[i], torch.tensor(-torch.inf, dtype=torch.float64)
            )
            self.assertFalse(bool(distances[i, true] <= lower_bound))
            self.assertEqual(int(mask[i].sum()), ranking_rows(table)[i]["true_rank"])


if __name__ == "__main__":
    unittest.main()
