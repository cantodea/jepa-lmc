from __future__ import annotations

import random
import unittest

import torch

from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.evaluation.refinement_priority import (
    latent_rank_costs,
    shuffled_rank_costs,
)
from jepa_lmc.learning.model import ActionJEPA
from jepa_lmc.verification.successor_refinement import (
    SuccessorRefinement,
    task_from_labels,
)


class SuccessorRefinementTests(unittest.TestCase):
    def test_no_oracle_calls_or_pruning_from_neural_costs_at_initialization(
        self,
    ) -> None:
        def forbidden(s, a):
            self.fail("No oracle call is allowed before explicit refinement.")

        refiner = SuccessorRefinement(3, 1, forbidden, costs=((1, 100, 3),) * 3)
        self.assertEqual(refiner.known, {})
        for state in range(3):
            self.assertEqual(refiner.candidates(state, 0), frozenset(range(3)))

    def test_local_exact_update_preserves_true_edges_and_other_actions(self) -> None:
        calls = []

        def oracle(state, action):
            calls.append((state, action))
            return state

        refiner = SuccessorRefinement(3, 2, oracle)
        self.assertEqual(refiner.observe(0, 1, expected=2), 0)
        self.assertEqual(refiner.candidates(0, 1), frozenset((0,)))
        self.assertEqual(refiner.candidates(0, 0), frozenset((0, 1, 2)))
        self.assertEqual(refiner.candidates(1, 1), frozenset((0, 1, 2)))
        self.assertEqual(refiner.observe(0, 1), 0)
        self.assertEqual(calls, [(0, 1)])
        snapshot = refiner.known
        snapshot[(0, 1)] = 2
        self.assertEqual(refiner.known[(0, 1)], 0)

    def test_budget_exhaustion_is_unknown_and_last_query_can_complete_proof(
        self,
    ) -> None:
        labels = (frozenset(), frozenset({"goal"}))
        task = task_from_labels("EF goal", 0, labels)
        blocked = SuccessorRefinement(2, 1, lambda s, a: s, budget=0).solve(task)
        self.assertIsNone(blocked.value)
        self.assertEqual(blocked.reason, "budget_exhausted")
        negative = SuccessorRefinement(2, 1, lambda s, a: s, budget=1).solve(task)
        self.assertFalse(negative.value)
        self.assertEqual(negative.closed_region, frozenset((0,)))
        positive = SuccessorRefinement(2, 1, lambda s, a: 1, budget=1).solve(task)
        self.assertTrue(positive.value)
        self.assertEqual(positive.witness, ((0, 0, 1),))

    def test_one_removed_witness_does_not_prove_unreachability(self) -> None:
        task = task_from_labels("EF goal", 0, (frozenset(), frozenset({"goal"})))
        refiner = SuccessorRefinement(2, 2, lambda s, a: a)
        result = refiner.solve(task)
        self.assertTrue(result.value)
        self.assertEqual(result.queries, 2)
        self.assertEqual(result.spurious_paths, 1)
        self.assertEqual(result.witness, ((0, 1, 1),))

    def test_until_disallows_dangerous_prefix_and_zero_length_uses_labels(self) -> None:
        labels = (frozenset(), frozenset({"danger"}), frozenset({"goal"}))

        def oracle(s, a):
            return min(s + 1, 2)

        for name, expected in (
            ("AG !danger", False),
            ("EF goal", True),
            ("E[!danger U goal]", False),
        ):
            result = SuccessorRefinement(3, 1, oracle).solve(
                task_from_labels(name, 0, labels)
            )
            self.assertEqual(result.value, expected)
        immediate = SuccessorRefinement(3, 1, oracle, budget=0).solve(
            task_from_labels("E[!danger U goal]", 2, labels)
        )
        self.assertTrue(immediate.value)
        self.assertEqual(immediate.queries, 0)
        with self.assertRaises(ValueError):
            task_from_labels("AG(EF goal)", 0, labels)

    def test_invalid_inputs_fail_before_mutating_the_relation(self) -> None:
        for costs in (((0, 1),) * 2, ((float("nan"), 1),) * 2, ((1,),)):
            with self.assertRaises(ValueError):
                SuccessorRefinement(2, 1, lambda s, a: 0, costs=costs)
        refiner = SuccessorRefinement(2, 1, lambda s, a: 99)
        with self.assertRaises(ValueError):
            refiner.observe(0, 0)
        self.assertEqual(refiner.known, {})
        with self.assertRaises(ValueError):
            refiner.candidates(99, 0)

    def test_all_topologies_and_adversarial_priorities_keep_sound_verdicts(
        self,
    ) -> None:
        expected = {
            "sealed_region": (True, False, False),
            "danger_gate": (False, True, False),
            "safe_detour": (False, True, True),
        }
        rng = random.Random(31)
        for case in radius_stress_cases():
            env = case.make_env()
            states = tuple(env.all_states())
            indices = {s: i for i, s in enumerate(states)}
            labels = tuple(frozenset((env.label(s),)) for s in states)
            costs = tuple(
                tuple(rng.randint(1, 100) for _ in states)
                for _ in range(len(states) * 4)
            )
            for name, truth in zip(
                ("AG !danger", "EF goal", "E[!danger U goal]"),
                expected[case.family],
                strict=True,
            ):
                for mode in ("uniform", "adversarial", "bfs"):
                    with self.subTest(case=case.name, property=name, mode=mode):

                        def oracle(s, a):
                            return indices[env.transition(states[s], a)]

                        refiner = SuccessorRefinement(
                            len(states),
                            4,
                            oracle,
                            costs=costs if mode == "adversarial" else None,
                        )
                        task = task_from_labels(name, indices[env.start], labels)
                        result = (
                            refiner.solve_direct_bfs(task)
                            if mode == "bfs"
                            else refiner.solve(task)
                        )
                        self.assertEqual(result.value, truth)
                        self.assertLessEqual(result.queries, len(states) * 4)
                        # Every prefix used for a new query is concretely reachable.
                        reachable = {task.start}
                        for observation in result.observations:
                            self.assertIn(observation.state, reachable)
                            reachable.add(observation.successor)
                        for state in range(len(states)):
                            for action in range(4):
                                self.assertIn(
                                    oracle(state, action),
                                    refiner.candidates(state, action),
                                )
                        if result.witness is not None:
                            for source, action, target in result.witness:
                                self.assertEqual(oracle(source, action), target)

    def test_recorded_wall_jump_is_removed_only_after_exact_query(self) -> None:
        env = next(
            c.make_env()
            for c in radius_stress_cases()
            if c.name == "danger_gate_layout0_rot0"
        )
        states = env.all_states()
        indices = {s: i for i, s in enumerate(states)}
        refiner = SuccessorRefinement(
            len(states), 4, lambda s, a: indices[env.transition(states[s], a)]
        )
        source, bogus = indices[(0, 1)], indices[(0, 3)]
        self.assertIn(bogus, refiner.candidates(source, 3))
        refiner.observe(source, 3, expected=bogus)
        self.assertNotIn(bogus, refiner.candidates(source, 3))
        self.assertIn(source, refiner.candidates(source, 3))
        self.assertIn(bogus, refiner.candidates(source, 0))


class RefinementPriorityTests(unittest.TestCase):
    def test_priorities_are_complete_rankings_and_preserve_the_frozen_model(
        self,
    ) -> None:
        torch.manual_seed(42)
        model = ActionJEPA(height=2, width=2, latent_dim=8)
        observations = torch.zeros(4, 4, 2, 2)
        for i in range(4):
            observations[i, 3, i // 2, i % 2] = 1
        before = {k: v.clone() for k, v in model.state_dict().items()}
        modes = [m.training for m in model.modules()]
        costs = latent_rank_costs(model, observations)
        self.assertEqual(len(costs), 16)
        self.assertTrue(all(sorted(row) == [1, 2, 3, 4] for row in costs))
        self.assertEqual(modes, [m.training for m in model.modules()])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))
        shuffled = shuffled_rank_costs(costs, 99)
        self.assertEqual(shuffled, shuffled_rank_costs(costs, 99))
        self.assertNotEqual(costs, shuffled)
        self.assertTrue(all(sorted(row) == [1, 2, 3, 4] for row in shuffled))


if __name__ == "__main__":
    unittest.main()
