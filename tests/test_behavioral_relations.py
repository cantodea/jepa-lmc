from __future__ import annotations

import itertools
import random
import unittest
from dataclasses import replace

from jepa_lmc.verification.behavioral_relations import (
    Elimination,
    audit_greatest_relation,
    bisimulation_by_partition,
    greatest_relation,
)
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem


def system(transitions, labels=None, initials=(0,)):
    return ExplicitTransitionSystem(
        states=transitions,
        initial_states=initials,
        transitions=transitions,
        labels=labels or {},
    )


def brute_force_maximum(left, right, kind, actions):
    candidates = [
        (s, t)
        for s in left.states
        for t in right.states
        if left.propositions(s) == right.propositions(t)
    ]
    maximum = set()
    for mask in range(1 << len(candidates)):
        relation = {p for i, p in enumerate(candidates) if mask & (1 << i)}
        valid = True
        for s, t in relation:
            for e in left.action_successors(s):
                valid &= any(
                    (not actions or e.action == f.action)
                    and (e.target, f.target) in relation
                    for f in right.action_successors(t)
                )
            if kind == "bisimulation":
                for f in right.action_successors(t):
                    valid &= any(
                        (not actions or e.action == f.action)
                        and (e.target, f.target) in relation
                        for e in left.action_successors(s)
                    )
        if valid:
            maximum.update(relation)
    return maximum


class BehavioralRelationTests(unittest.TestCase):
    def test_different_edges_can_have_all_diagonal_pairs_bisimilar(self):
        labels = {1: {"goal"}, 2: {"goal"}}
        left = system({0: [(0, 1)], 1: [(0, 1)], 2: [(0, 2)]}, labels)
        right = system({0: [(0, 2)], 1: [(0, 1)], 2: [(0, 2)]}, labels)
        result = greatest_relation(left, right, kind="bisimulation")
        self.assertNotEqual(left.successors(0), right.successors(0))
        self.assertTrue(all((s, s) in result.pairs for s in left.states))
        self.assertIn((1, 2), result.pairs)
        self.assertTrue(result.relates_initials(left, right))
        audit_greatest_relation(left, right, result)

    def test_direction_and_initial_quantifiers(self):
        labels = {1: {"p"}}
        small = system({0: [(0, 0)], 1: [(0, 1)]}, labels)
        large = system({0: [(0, 0), (0, 1)], 1: [(0, 1)]}, labels)
        forward = greatest_relation(small, large)
        backward = greatest_relation(large, small)
        self.assertTrue(forward.relates_initials(small, large))
        self.assertFalse(backward.relates_initials(large, small))
        both_initials = system({0: [(0, 0)], 1: [(0, 1)]}, labels, initials=(0, 1))
        self.assertFalse(
            greatest_relation(both_initials, small).relates_initials(
                both_initials, small
            )
        )
        self.assertTrue(
            greatest_relation(small, both_initials).relates_initials(
                small, both_initials
            )
        )
        self.assertFalse(
            greatest_relation(
                small, both_initials, kind="bisimulation"
            ).relates_initials(small, both_initials)
        )

    def test_mutual_simulation_is_not_bisimulation_on_branching_graphs(self):
        left = system(
            {0: [(0, 1)], 1: [(0, 2), (0, 3)], 2: [(0, 2)], 3: [(0, 3)]},
            {0: {"root"}, 2: {"b"}, 3: {"c"}},
        )
        right = system(
            {
                0: [(0, 1), (0, 4), (0, 5)],
                1: [(0, 2), (0, 3)],
                2: [(0, 2)],
                3: [(0, 3)],
                4: [(0, 2)],
                5: [(0, 3)],
            },
            {0: {"root"}, 2: {"b"}, 3: {"c"}},
        )
        self.assertIn((0, 0), greatest_relation(left, right).pairs)
        self.assertIn((0, 0), greatest_relation(right, left).pairs)
        bisim = greatest_relation(left, right, kind="bisimulation")
        self.assertNotIn((0, 0), bisim.pairs)
        audit_greatest_relation(left, right, bisim)
        self.assertEqual(bisim.pairs, bisimulation_by_partition(left, right))

    def test_action_labels_are_optional_but_state_labels_are_required(self):
        left, right = system({0: [("a", 0)]}), system({0: [("b", 0)]})
        self.assertIn((0, 0), greatest_relation(left, right, kind="bisimulation").pairs)
        result = greatest_relation(left, right, action_sensitive=True)
        self.assertNotIn((0, 0), result.pairs)
        audit_greatest_relation(left, right, result)
        labelled = system({0: [("a", 0)]}, {0: {"p"}})
        result = greatest_relation(left, labelled)
        self.assertEqual(result.pairs, frozenset())
        self.assertEqual(result.eliminations, (Elimination(0, 0, 0, "label"),))

    def test_initial_equivalence_does_not_require_unreachable_diagonal_pairs(self):
        labels = {1: {"danger"}}
        left = system({0: [(0, 0)], 1: [(0, 1)], 2: [(0, 2)]}, labels)
        right = system({0: [(0, 0)], 1: [(0, 2)], 2: [(0, 2)]}, labels)
        result = greatest_relation(left, right, kind="bisimulation")
        self.assertTrue(result.relates_initials(left, right))
        self.assertNotIn((1, 1), result.pairs)

    def test_rejection_audit_detects_a_closed_but_nonmaximal_forgery(self):
        graph = system({0: [(0, 0)], 1: [(0, 1)]})
        correct = greatest_relation(graph, graph)
        forged = replace(
            correct,
            pairs=correct.pairs - {(0, 0)},
            eliminations=(Elimination(0, 0, 1, "left", None, 0),),
            rounds=1,
        )
        with self.assertRaisesRegex(AssertionError, "earlier"):
            audit_greatest_relation(graph, graph, forged)
        with self.assertRaises(ValueError):
            greatest_relation(graph, graph, kind="mutual")

    def test_exact_maxima_against_enumerating_all_candidate_relations(self):
        rng = random.Random(2718)
        for trial in range(30):
            graphs = []
            for _ in range(2):
                n = rng.randint(2, 3)
                graphs.append(
                    system(
                        {
                            s: [
                                (rng.randrange(2), rng.randrange(n))
                                for _ in range(rng.randint(1, 3))
                            ]
                            for s in range(n)
                        },
                        {s: {"p"} if rng.randrange(2) else set() for s in range(n)},
                    )
                )
            left, right = graphs
            for kind, actions in itertools.product(
                ("simulation", "bisimulation"), (False, True)
            ):
                with self.subTest(trial=trial, kind=kind, actions=actions):
                    result = greatest_relation(
                        left, right, kind=kind, action_sensitive=actions
                    )
                    self.assertEqual(
                        result.pairs, brute_force_maximum(left, right, kind, actions)
                    )
                    audit_greatest_relation(left, right, result)
                    if kind == "bisimulation":
                        self.assertEqual(
                            result.pairs,
                            bisimulation_by_partition(
                                left, right, action_sensitive=actions
                            ),
                        )

    def test_total_action_determinism_makes_simulation_symmetric(self):
        rng = random.Random(314)
        for _ in range(20):
            graphs = [
                system(
                    {s: [(a, rng.randrange(4)) for a in range(2)] for s in range(4)},
                    {0: {"p"}},
                )
                for _ in range(2)
            ]
            left, right = graphs
            forward = greatest_relation(left, right, action_sensitive=True)
            backward = greatest_relation(right, left, action_sensitive=True)
            bisim = greatest_relation(
                left, right, kind="bisimulation", action_sensitive=True
            )
            self.assertEqual(forward.pairs, {(t, s) for s, t in backward.pairs})
            self.assertEqual(forward.pairs, bisim.pairs)


if __name__ == "__main__":
    unittest.main()
