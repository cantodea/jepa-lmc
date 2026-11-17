from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from jepa_lmc.envs.gridworld import GridWorld
from jepa_lmc.evaluation.trace_sanity import KnownProblem, Trace
from jepa_lmc.learning.finite_trace import FiniteTraceData, fit_finite_model, tensor_digest
from jepa_lmc.learning.passive_negative import (
    analyze_version_space,
    encode_trace,
    fit_with_negative_traces,
    trace_negative_loss,
)


class VersionSpaceTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = (0, 0), (0, 1)
        self.problem = KnownProblem((self.a, self.b), frozenset({self.a}), (0,),
                                    {self.a: frozenset(), self.b: frozenset()})

    def test_paired_negative_is_redundant(self):
        good = Trace((self.a, self.a), (0,))
        bad = Trace((self.a, self.b), (0,))
        result = analyze_version_space(self.problem, [good], [bad])
        self.assertEqual(result["status"], "exact")
        self.assertEqual(result["model_count_good"], "2")
        self.assertEqual(result["model_count_good_bad"], "2")
        self.assertTrue(result["negative_diagnostics"][0]["redundant_given_good"])
        self.assertEqual(result["information_bits"], 0)

    def test_general_negative_reduces_space_without_reducing_marginals(self):
        bad = Trace((self.a, self.b, self.a), (0, 0))
        result = analyze_version_space(self.problem, [], [bad])
        self.assertEqual(result["model_count_good"], "4")
        self.assertEqual(result["model_count_good_bad"], "3")
        self.assertEqual([r["candidate_count"] for r in result["candidate_successors"]], [2, 2])
        self.assertEqual(result["unknown_pairs_with_reduced_support"], 0)
        self.assertAlmostEqual(result["information_bits"], math.log2(4 / 3))

    def test_duplicate_negatives_are_sequentially_and_globally_redundant(self):
        bad = Trace((self.a, self.b), (0,))
        result = analyze_version_space(self.problem, [], [bad, bad])
        self.assertEqual(result["model_count_good_bad"], "2")
        diagnostics = result["negative_diagnostics"]
        self.assertFalse(diagnostics[0]["sequential_redundant"])
        self.assertTrue(diagnostics[1]["sequential_redundant"])
        self.assertTrue(all(d["redundant_given_all_other_bad"] for d in diagnostics))

    def test_unsatisfiable_labels_are_explicit(self):
        trace = Trace((self.a, self.a), (0,))
        result = analyze_version_space(self.problem, [trace], [trace])
        self.assertFalse(result["satisfiable"])
        self.assertEqual(result["model_count_good_bad"], "0")
        self.assertEqual(result["completion_edges"], [])

    def test_limit_is_unknown_and_not_approximation(self):
        bad = Trace((self.a, self.b, self.a), (0, 0))
        result = analyze_version_space(self.problem, [], [bad], max_nodes=1)
        self.assertEqual(result["status"], "resource_limit")
        self.assertIsNone(result["model_count_good_bad"])
        self.assertNotIn("completion_edges", result)

    def test_repeated_conflicting_pair_is_class_wide_impossibility(self):
        bad = Trace((self.a, self.a, self.b), (0, 0))
        self.assertIsNone(encode_trace(self.problem, bad))
        result = analyze_version_space(self.problem, [], [bad])
        self.assertEqual(result["model_count_good_bad"], "4")
        self.assertEqual(result["negative_diagnostics"][0]["reason"],
                         "syntax_or_internal_determinism_conflict")


class TraceLossTests(unittest.TestCase):
    def test_loss_matches_whole_trace_rejection_probability(self):
        logits = torch.tensor([[math.log(.75), math.log(.25)],
                               [math.log(.4), math.log(.6)]], requires_grad=True)
        loss = trace_negative_loss(logits, [((0, 1), (1, 0))])
        self.assertAlmostEqual(float(loss.detach()), -math.log(1 - .25 * .4), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_repeated_equal_pairs_are_not_independent_draws(self):
        a, b = (0, 0), (0, 1)
        p = KnownProblem((a, b), frozenset({a}), (0,), {a: frozenset(), b: frozenset()})
        trace = Trace((a, a, a), (0, 0))
        encoded = encode_trace(p, trace)
        self.assertEqual(encoded, ((0, 0),))
        logits = torch.zeros((2, 2))
        self.assertAlmostEqual(float(trace_negative_loss(logits, [encoded])), math.log(2), places=6)

    def test_impossible_samples_contribute_zero_and_keep_denominator(self):
        logits = torch.zeros((2, 2), requires_grad=True)
        loss = trace_negative_loss(logits, [None, ((0, 0),)])
        self.assertAlmostEqual(float(loss.detach()), math.log(2) / 2, places=6)
        loss.backward()
        self.assertEqual(float(logits.grad[1].abs().sum()), 0)

    def test_saturated_acceptance_remains_finite(self):
        logits = torch.tensor([[1000., -1000.]], requires_grad=True)
        loss = trace_negative_loss(logits, [((0, 0),)])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class NegativeTrainingTests(unittest.TestCase):
    def setUp(self):
        states = ((0, 0), (0, 1), (1, 0), (1, 1))
        self.problem = KnownProblem(states, frozenset({states[0]}), (0, 1),
                                    {s: frozenset() for s in states})
        self.goods = [Trace((states[0], states[1], states[1]), (0, 1))]
        self.bad = [Trace((states[0], states[2], states[1]), (0, 1))]
        self.data = FiniteTraceData.from_traces(self.problem, self.goods, height=2, width=2)
        config_path = Path(__file__).resolve().parents[1] / "configs/finite_trace_mlp900_vs_jepa.json"
        self.config = json.loads(config_path.read_text())
        self.config["training"]["epochs"] = 2
        self.config["mlp"]["training"]["epochs"] = 2

    def test_zero_negative_weight_exactly_reproduces_historical_training(self):
        for kind in ("mlp", "jepa"):
            old, _ = fit_finite_model(self.data, kind=kind, seed=11, config=self.config)
            new, _ = fit_with_negative_traces(self.data, self.bad, kind=kind, seed=11,
                                              config=self.config, negative_weight=0)
            self.assertEqual(tensor_digest(*old.state_dict().items()), tensor_digest(*new.state_dict().items()))

    def test_oracle_unavailable_initialization_identical_and_loss_finite(self):
        for kind in ("mlp", "jepa"):
            _, old = fit_finite_model(self.data, kind=kind, seed=11, config=self.config)
            with patch.object(GridWorld, "transition", side_effect=AssertionError("hidden oracle")), \
                 patch.object(GridWorld, "all_transitions", side_effect=AssertionError("hidden R")):
                model, metadata = fit_with_negative_traces(self.data, self.bad, kind=kind,
                                                           seed=11, config=self.config)
            self.assertEqual(old["initial_state_sha256"], metadata["initial_state_sha256"])
            self.assertEqual(metadata["training_sha256"], self.data.training_sha256)
            self.assertTrue(all(math.isfinite(h["loss"]) for h in metadata["history"]))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
