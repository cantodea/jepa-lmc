from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from experiments.good_bad_trace_sanity import (
    BUDGETS,
    SEEDS,
    aggregate,
    benchmark_catalogue,
    evaluate_budget,
    evaluate_total_model,
    run_experiment,
    stream_seed,
)
from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.envs.gridworld import GridWorld
from jepa_lmc.evaluation.structural import SEMANTICS, evaluate_model_pair
from jepa_lmc.evaluation.trace_sanity import (
    KnownProblem,
    Trace,
    accepts_trace,
    is_execution,
    make_hard_negatives,
    maximally_permissive,
    observed_edges,
    observed_self_loop_completion,
    relation_edges,
    sample_consistency,
    sample_good_traces,
    validate_trace_pairs,
)
from jepa_lmc.verification.gridworld import gridworld_to_transition_system
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem


def catalogue(env):
    real = gridworld_to_transition_system(env)
    known = KnownProblem(
        tuple(sorted(real.states)),
        real.initial_states,
        tuple(sorted(env.ACTIONS)),
        {s: real.propositions(s) for s in real.states},
    )
    return real, known


class TraceSanityTests(unittest.TestCase):
    def setUp(self):
        self.case = radius_stress_cases()[8]
        self.env = self.case.make_env()
        self.real, self.known = catalogue(self.env)
        self.goods = sample_good_traces(self.env, seed=20260804)
        self.negatives = make_hard_negatives(
            self.env, self.goods, self.known.labels, seed=20260805
        )
        self.relation = observed_edges(self.goods)

    def test_sampling_and_corruption_are_reproducible_prefixes(self):
        for budget in BUDGETS:
            self.assertEqual(
                sample_good_traces(self.env, seed=20260804, count=budget),
                self.goods[:budget],
            )
            self.assertEqual(
                make_hard_negatives(
                    self.env, self.goods[:budget], self.known.labels, seed=20260805
                ),
                self.negatives[:budget],
            )
        self.assertNotEqual(self.goods, sample_good_traces(self.env, seed=20260806))
        seed = stream_seed(20260804, self.case.name, "good")
        self.assertEqual(seed, stream_seed(20260804, self.case.name, "good"))
        self.assertNotEqual(seed, stream_seed(20260804, self.case.name, "bad"))
        self.assertNotEqual(seed, stream_seed(20260804, "another_map", "good"))

    def test_every_benchmark_pool_is_valid_and_perfectly_separated(self):
        # Exercise all three label families/rotations and all 72 sampling pools.
        for case in radius_stress_cases():
            env = case.make_env()
            real, known = catalogue(env)
            for seed in SEEDS:
                with self.subTest(case=case.name, seed=seed):
                    goods = sample_good_traces(
                        env, seed=stream_seed(seed, case.name, "good")
                    )
                    negatives = make_hard_negatives(
                        env,
                        goods,
                        known.labels,
                        seed=stream_seed(seed, case.name, "bad"),
                    )
                    validate_trace_pairs(env, goods, negatives)
                    relation = observed_edges(goods)
                    self.assertLessEqual(relation, relation_edges(real))
                    for good, negative in zip(goods, negatives, strict=True):
                        self.assertEqual(len(good.actions), 16)
                        self.assertEqual(len(good.states), 17)
                        self.assertEqual(good.states[0], env.start)
                        self.assertTrue(is_execution(env, good))
                        self.assertFalse(is_execution(env, negative.trace))
                        self.assertTrue(accepts_trace(known, relation, good))
                        self.assertFalse(accepts_trace(known, relation, negative.trace))

    def test_corruption_records_exactly_one_false_transition_and_true_suffix(self):
        for good, negative in zip(self.goods, self.negatives, strict=True):
            bad, step = negative.trace, negative.corrupted_step
            self.assertEqual(good.actions, bad.actions)
            self.assertEqual(good.states[: step + 1], bad.states[: step + 1])
            false = [
                i
                for i, (s, a, t) in enumerate(bad.edges)
                if self.env.transition(s, a) != t
            ]
            self.assertEqual(false, [step])
            self.assertEqual(negative.true_successor, good.states[step + 1])
            self.assertEqual(negative.inserted_successor, bad.states[step + 1])
            self.assertNotEqual(negative.true_successor, negative.inserted_successor)
            matching = [
                s
                for s in self.known.states
                if s != negative.true_successor
                and self.known.labels[s] == self.known.labels[negative.true_successor]
            ]
            if matching:
                self.assertIn(negative.inserted_successor, matching)
                self.assertTrue(negative.same_proposition_labels)

    def test_singleton_label_fallback_is_still_a_false_transition(self):
        env = GridWorld(2, 1, (0, 0), (0, 1), set(), set())
        _, known = catalogue(env)
        good = Trace(((0, 0), (0, 1)), (3,))
        (negative,) = make_hard_negatives(env, [good], known.labels, seed=1)
        self.assertFalse(negative.same_proposition_labels)
        self.assertEqual(negative.inserted_successor, (0, 0))
        validate_trace_pairs(env, [good], [negative])

    def test_validator_rejects_invalid_good_and_corrupt_metadata(self):
        with self.assertRaises(ValueError):
            validate_trace_pairs(
                self.env, [self.negatives[0].trace], [self.negatives[0]]
            )
        with self.assertRaises(ValueError):
            validate_trace_pairs(
                self.env,
                [self.goods[0]],
                [replace(self.negatives[0], corrupted_step=-1)],
            )
        with self.assertRaises(ValueError):
            validate_trace_pairs(
                self.env,
                [self.goods[0]],
                [replace(self.negatives[0], trace=self.goods[0])],
            )
        with self.assertRaises(ValueError):
            sample_good_traces(self.env, seed=1, length=0)
        with self.assertRaises(ValueError):
            Trace(((0, 0),), (0,))

    def test_nested_budgets_have_monotone_partial_relations(self):
        previous = frozenset()
        for budget in BUDGETS:
            relation = observed_edges(self.goods[:budget])
            self.assertLessEqual(previous, relation)
            self.assertLessEqual(relation, relation_edges(self.real))
            self.assertEqual(len(relation), len({(s, a) for s, a, _ in relation}))
            previous = relation

    def test_partial_membership_checks_initial_state_actions_and_catalogue(self):
        full = relation_edges(maximally_permissive(self.known))
        other = next(s for s in self.known.states if s not in self.known.initial_states)
        self.assertFalse(accepts_trace(self.known, full, Trace((other,), ())))
        self.assertTrue(
            accepts_trace(self.known, self.relation, Trace((self.env.start,), ()))
        )
        self.assertFalse(
            accepts_trace(self.known, full, Trace((self.env.start, other), (99,)))
        )
        self.assertFalse(
            accepts_trace(self.known, full, Trace((self.env.start, (-1, -1)), (0,)))
        )
        # The target alone is insufficient: membership must retain action labels.
        good = self.goods[0]
        edge = good.edges[0]
        one = frozenset({edge})
        self.assertFalse(
            accepts_trace(
                self.known, one, Trace((edge[0], edge[2]), ((edge[1] + 1) % 4,))
            )
        )

    def test_completion_is_action_total_preserves_observed_edges_and_known_data(self):
        relation = observed_edges(self.goods[:1])
        completed = observed_self_loop_completion(self.known, relation)
        self.assertIsInstance(completed, ExplicitTransitionSystem)
        self.assertEqual(completed.states, self.real.states)
        self.assertEqual(completed.initial_states, self.real.initial_states)
        observed = {(s, a): t for s, a, t in relation}
        for s in self.known.states:
            self.assertEqual(completed.propositions(s), self.real.propositions(s))
            actual = {e.action: e.target for e in completed.action_successors(s)}
            self.assertEqual(set(actual), set(self.known.actions))
            for a in self.known.actions:
                self.assertEqual(actual[a], observed.get((s, a), s))
        self.assertEqual(relation, observed_edges(self.goods[:1]))
        self.assertLessEqual(relation, relation_edges(completed))
        self.assertGreater(len(relation_edges(completed)), len(relation))

    def test_baseline_construction_does_not_query_transition_oracle(self):
        with patch.object(
            GridWorld, "transition", side_effect=AssertionError("oracle leak")
        ):
            partial = observed_edges(self.goods)
            observed_self_loop_completion(self.known, partial)
            maximally_permissive(self.known)
            sample_consistency(
                self.known, partial, self.goods, [n.trace for n in self.negatives]
            )

    def test_complete_relation_is_cartesian_product_and_accepts_all_samples(self):
        complete = maximally_permissive(self.known)
        edges = relation_edges(complete)
        expected = frozenset(
            (s, a, t)
            for s in self.known.states
            for a in self.known.actions
            for t in self.known.states
        )
        self.assertEqual(edges, expected)
        self.assertEqual(
            len(edges), len(self.known.states) ** 2 * len(self.known.actions)
        )
        rates = sample_consistency(
            self.known, edges, self.goods, [n.trace for n in self.negatives]
        )
        self.assertEqual(rates["good_acceptance"], 1)
        self.assertEqual(rates["bad_rejection"], 0)

    def test_completion_can_accept_a_bad_trace_that_partial_relation_rejects(self):
        env = GridWorld(2, 1, (0, 0), (0, 1), set(), set())
        _, known = catalogue(env)
        good = Trace(((0, 0), (0, 1)), (3,))
        bad = Trace(((0, 0), (0, 1), (0, 1)), (3, 2))
        partial = observed_edges([good])
        self.assertFalse(is_execution(env, bad))
        self.assertFalse(accepts_trace(known, partial, bad))
        completed = relation_edges(observed_self_loop_completion(known, partial))
        self.assertTrue(accepts_trace(known, completed, bad))

    def test_partial_relation_is_never_sent_to_formal_evaluator(self):
        with patch(
            "experiments.good_bad_trace_sanity.evaluate_model_pair",
            wraps=evaluate_model_pair,
        ) as shared:
            controls = {
                "maximally_permissive": evaluate_total_model(
                    self.real, maximally_permissive(self.known)
                ),
                "real_reference": evaluate_total_model(self.real, self.real),
            }
            row = evaluate_budget(
                self.known,
                self.real,
                self.goods[:1],
                self.negatives[:1],
                controls=controls,
            )
        self.assertEqual(shared.call_count, 3)
        for call in shared.call_args_list:
            self.assertTrue(
                all(isinstance(g, ExplicitTransitionSystem) for g in call.args)
            )
        self.assertNotIn("behavioral", row["observed_edge"])
        self.assertNotIn("diagnostics", row["observed_edge"])
        self.assertTrue(row["expected_trivial_separator"])
        self.assertEqual(row["observed_steps"], 16)
        self.assertEqual(
            row["transition_coverage"], row["observed_edge"]["transition_recall"]
        )
        for variant in row["formal"].values():
            self.assertEqual(set(variant["behavioral"]), set(SEMANTICS))
            self.assertEqual(len(variant["diagnostics"]["initial_ctl_verdicts"]), 6)
            self.assertIsNone(variant["diagnostics"]["ltl"])
        summary = aggregate([row])
        self.assertEqual(summary["cases"], 1)
        for semantics in SEMANTICS:
            p = summary["formal"]["real_reference"]["primary_metrics"][semantics]
            self.assertTrue(all(v["fraction"] == 1 for v in p.values()))

    def test_exact_october_benchmark_and_no_overwrite(self):
        cases, source = benchmark_catalogue()
        self.assertEqual(cases, radius_stress_cases())
        self.assertEqual(len(cases), 24)
        self.assertIn("structural_evaluation", source["path"])
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "marker.txt"
            marker.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                run_experiment(directory, progress=False)
            self.assertEqual(marker.read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()
