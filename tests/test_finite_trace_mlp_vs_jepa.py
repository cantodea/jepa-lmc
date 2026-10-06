from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.finite_trace_mlp_vs_jepa import (
    CONFIG,
    aggregate,
    known_catalogue,
    paired_comparison,
    read_training_pools,
    run_experiment,
)
from experiments.good_bad_trace_sanity import stream_seed
from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.envs.gridworld import GridWorld
from jepa_lmc.evaluation.finite_trace import (
    LTLBackendCache,
    disjoint_heldout_traces,
    evaluate_predictor,
    pair_partition,
    prediction_system,
    trace_scores,
    transition_scores,
)
from jepa_lmc.evaluation.structural import SEMANTICS, evaluate_model_pair
from jepa_lmc.evaluation.trace_sanity import (
    is_execution,
    observed_edges,
    observed_self_loop_completion,
    relation_edges,
    sample_good_traces,
)
from jepa_lmc.learning.data import GridWorldTransitionDataset, gridworld_observation
from jepa_lmc.learning.finite_trace import (
    DirectSuccessorMLP,
    FiniteTraceData,
    fit_finite_model,
    known_observations,
    predict_successors,
    tensor_digest,
)
from jepa_lmc.learning.ranking import same_map_ranking_loss
from jepa_lmc.verification.gridworld import gridworld_to_transition_system
from jepa_lmc.verification.nuxmv import find_nusmv_executable
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem


def test_config():
    config = json.loads(CONFIG.read_text())
    config["training"]["epochs"] = 2
    return config


class FiniteTraceLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.case = radius_stress_cases()[16]
        self.env = self.case.make_env()
        self.problem = known_catalogue(self.env)
        self.goods = sample_good_traces(self.env, seed=20260804)
        self.data = FiniteTraceData.from_traces(
            self.problem, self.goods[:1], height=6, width=6
        )
        self.config = test_config()

    def test_representation_equals_existing_jepa_input_from_known_metadata(self):
        expected = torch.stack(
            [gridworld_observation(self.env, s) for s in self.problem.states]
        )
        with patch.object(
            GridWorld, "transition", side_effect=AssertionError("oracle")
        ):
            actual = known_observations(self.problem, height=6, width=6)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.float32)

    def test_dataset_labels_are_exact_observed_trace_steps_not_cartesian_domain(self):
        expected = [e for t in self.goods[:1] for e in t.edges]
        decoded = [
            (
                self.problem.states[int(s)],
                self.problem.actions[int(a)],
                self.problem.states[int(t)],
            )
            for s, a, t in zip(
                self.data.sources, self.data.actions, self.data.successors, strict=True
            )
        ]
        self.assertEqual(decoded, expected)
        self.assertEqual(len(decoded), 16)
        self.assertEqual(
            self.data.observed_pairs, frozenset((s, a) for s, a, _ in expected)
        )
        self.assertLess(len(self.data.observed_pairs), len(self.problem.states) * 4)

    def test_original_pools_reused_and_all_nested_budgets_valid(self):
        pools = read_training_pools()
        self.assertEqual(len(pools), 72)
        maps = {c.name: c for c in radius_stress_cases()}
        for (seed, name), traces in pools.items():
            env = maps[name].make_env()
            problem = known_catalogue(env)
            self.assertEqual(
                traces, sample_good_traces(env, seed=stream_seed(seed, name, "good"))
            )
            previous = frozenset()
            for budget in (1, 2, 4, 8, 16):
                data = FiniteTraceData.from_traces(
                    problem, traces[:budget], height=6, width=6
                )
                self.assertEqual(len(data.actions), 16 * budget)
                self.assertLessEqual(previous, data.observed_pairs)
                previous = data.observed_pairs

    def test_mlp_and_jepa_training_inference_cannot_query_or_load_hidden_truth(self):
        with (
            patch.object(
                GridWorld, "transition", side_effect=AssertionError("hidden T")
            ),
            patch.object(
                GridWorld, "all_transitions", side_effect=AssertionError("hidden R")
            ),
            patch.object(
                GridWorldTransitionDataset,
                "__init__",
                side_effect=AssertionError("exhaustive data"),
            ),
            patch("torch.load", side_effect=AssertionError("pretrained checkpoint")),
        ):
            data = FiniteTraceData.from_traces(
                self.problem, self.goods[:1], height=6, width=6
            )
            hashes = []
            for kind in ("mlp", "jepa"):
                model, metadata = fit_finite_model(
                    data, kind=kind, seed=17, config=self.config
                )
                predictions = predict_successors(model, data, kind=kind)
                graph = prediction_system(self.problem, predictions)
                self.assertIsInstance(graph, ExplicitTransitionSystem)
                self.assertEqual(len(predictions), 4 * len(self.problem.states))
                self.assertEqual(metadata["steps_per_epoch"], 16)
                self.assertEqual(metadata["examples_processed"], 32)
                hashes.append(metadata["training_sha256"])
            self.assertEqual(hashes, [data.training_sha256] * 2)

    def test_mlp_supervision_and_forward_queries_are_observed_samples_only(self):
        original = DirectSuccessorMLP.forward
        seen = []

        def checked_forward(model, observation, action):
            self.assertTrue(
                torch.equal(observation, self.data.observations[self.data.sources])
            )
            self.assertTrue(torch.equal(action, self.data.actions))
            seen.append(len(action))
            return original(model, observation, action)

        with patch.object(DirectSuccessorMLP, "forward", checked_forward):
            model, _ = fit_finite_model(
                self.data, kind="mlp", seed=17, config=self.config
            )
        self.assertEqual(seen, [16, 16])
        self.assertFalse(hasattr(model, "target_encoder"))
        self.assertFalse(hasattr(model, "predictor"))
        self.assertEqual(model.network[0].in_features, 152)
        self.assertEqual(model.network[0].out_features, 64)
        self.assertEqual(model.network[-1].out_features, len(self.problem.states))

    def test_jepa_ranking_positives_only_come_from_observed_successors(self):
        with patch(
            "jepa_lmc.learning.finite_trace.same_map_ranking_loss",
            wraps=same_map_ranking_loss,
        ) as ranking:
            fit_finite_model(self.data, kind="jepa", seed=17, config=self.config)
        self.assertEqual(ranking.call_count, 2)
        for call in ranking.call_args_list:
            prediction, bank, successors, query_maps, candidate_maps = call.args
            self.assertEqual(len(prediction), 16)
            self.assertEqual(len(bank), len(self.problem.states))
            self.assertTrue(torch.equal(successors, self.data.successors))
            self.assertTrue(torch.equal(query_maps, torch.zeros(16, dtype=torch.long)))
            self.assertFalse(bank.requires_grad)
            self.assertEqual(len(candidate_maps), len(self.problem.states))

    def test_fixed_seed_training_reproduces_weights_and_predictions(self):
        for kind in ("mlp", "jepa"):
            first, one = fit_finite_model(
                self.data, kind=kind, seed=31, config=self.config
            )
            second, two = fit_finite_model(
                self.data, kind=kind, seed=31, config=self.config
            )
            self.assertEqual(one["initial_state_sha256"], two["initial_state_sha256"])
            self.assertEqual(one["final_state_sha256"], two["final_state_sha256"])
            self.assertEqual(
                predict_successors(first, self.data, kind=kind),
                predict_successors(second, self.data, kind=kind),
            )
            larger = FiniteTraceData.from_traces(
                self.problem, self.goods[:2], height=6, width=6
            )
            _, other = fit_finite_model(larger, kind=kind, seed=31, config=self.config)
            self.assertEqual(one["initial_state_sha256"], other["initial_state_sha256"])
        self.assertEqual(self.config, test_config())

    def test_tensor_hash_changes_when_only_observed_supervision_changes(self):
        different = self.data.successors.clone()
        different[0] = (different[0] + 1) % len(self.problem.states)
        self.assertNotEqual(
            tensor_digest(("target", self.data.successors)),
            tensor_digest(("target", different)),
        )


class FiniteTraceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.env = GridWorld(3, 3, (0, 0), (2, 2), {(1, 1)}, {(0, 2)})
        self.problem = known_catalogue(self.env)
        self.goods = sample_good_traces(self.env, seed=4)
        self.real = gridworld_to_transition_system(self.env)
        self.truth = {(s, a): t for s, a, t in relation_edges(self.real)}
        self.partial = {(s, a): t for s, a, t in observed_edges(self.goods[:1])}
        self.observed = frozenset(self.partial)

    def test_relative_output_is_resolved_before_historical_hash_protection(self):
        def check_destination(*, executable, log_path):
            self.assertTrue(log_path.is_absolute())
            self.assertEqual(log_path, destination / "ltl_backend.jsonl.gz")
            raise RuntimeError("destination checked before full benchmark")

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            destination = Path(directory).resolve() / "run"
            relative = destination.relative_to(Path.cwd())
            with patch(
                "experiments.finite_trace_mlp_vs_jepa.LTLBackendCache",
                side_effect=check_destination,
            ):
                with self.assertRaisesRegex(RuntimeError, "destination checked"):
                    run_experiment(relative)

    def test_partition_and_abstention_denominators_are_exact(self):
        observed, unseen = pair_partition(self.problem, self.observed)
        self.assertFalse(observed & unseen)
        self.assertEqual(observed | unseen, frozenset(self.truth))
        partial = transition_scores(
            self.problem, self.observed, self.partial, self.real
        )
        self.assertEqual(partial["observed"]["accuracy"], 1)
        self.assertEqual(partial["unseen"]["accuracy"], 0)
        self.assertEqual(partial["unseen"]["predicted"], 0)
        self.assertEqual(
            partial["overall"]["accuracy"], len(observed) / len(self.truth)
        )
        perfect = transition_scores(self.problem, self.observed, self.truth, self.real)
        self.assertEqual(perfect["unseen"]["accuracy"], 1)
        self.assertEqual(perfect["overall"]["accuracy"], 1)
        self.assertIsNone(perfect["unreachable_unseen"]["accuracy"])

    def test_denominator_keeps_unreachable_states(self):
        env = radius_stress_cases()[0].make_env()
        problem = known_catalogue(env)
        real = gridworld_to_transition_system(env)
        observed = observed_edges(sample_good_traces(env, seed=4, count=1))
        predictions = {(s, a): t for s, a, t in observed}
        result = transition_scores(problem, predictions.keys(), predictions, real)
        self.assertEqual(result["overall"]["total"], len(problem.states) * 4)
        self.assertGreater(result["unreachable_unseen"]["total"], 0)

    def test_heldout_sampling_is_real_unique_disjoint_and_reproducible(self):
        heldout = disjoint_heldout_traces(self.env, self.goods, seed=55)
        self.assertEqual(len(heldout), 64)
        self.assertEqual(len(set(heldout)), 64)
        self.assertFalse(set(heldout) & set(self.goods))
        self.assertTrue(all(is_execution(self.env, t) for t in heldout))
        self.assertEqual(
            heldout, disjoint_heldout_traces(self.env, self.goods, seed=55)
        )
        self.assertFalse(set(heldout) & set(self.goods[:1]))

    def test_learned_system_is_total_and_preserves_known_metadata(self):
        graph = prediction_system(self.problem, self.truth)
        self.assertEqual(relation_edges(graph), relation_edges(self.real))
        self.assertEqual(graph.states, self.real.states)
        self.assertEqual(graph.initial_states, self.real.initial_states)
        for s in self.problem.states:
            self.assertEqual(graph.propositions(s), self.real.propositions(s))
            self.assertEqual(
                {e.action for e in graph.action_successors(s)},
                set(self.problem.actions),
            )
        with self.assertRaises(ValueError):
            prediction_system(self.problem, self.partial)
        with self.assertRaises(ValueError):
            prediction_system(
                self.problem, self.truth | {next(iter(self.truth)): (-1, -1)}
            )

    def test_trace_failure_uses_action_labels_and_reports_first_real_edge(self):
        trace = self.goods[0]
        perfect = trace_scores(self.problem, [trace], self.truth, self.observed)
        self.assertEqual(perfect["acceptance"]["accuracy"], 1)
        s, a, t = trace.edges[0]
        wrong = next(u for u in self.problem.states if u != t)
        failed = trace_scores(
            self.problem, [trace], self.truth | {(s, a): wrong}, self.observed
        )
        self.assertEqual(failed["acceptance"]["accuracy"], 0)
        self.assertEqual(failed["first_failure_histogram"], {0: 1})
        self.assertEqual(failed["traces"][0]["first_failure_edge"]["real_successor"], t)
        self.assertEqual(
            failed["traces"][0]["first_failure_edge"]["predicted_successor"], wrong
        )

    @unittest.skipUnless(find_nusmv_executable(), "nuXmv not installed")
    def test_structural_and_actual_ltl_backend_reuse_and_serialization(self):
        data = FiniteTraceData.from_traces(
            self.problem, self.goods[:1], height=3, width=3
        )
        heldout = disjoint_heldout_traces(self.env, self.goods, seed=55, count=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backend.jsonl.gz"
            backend = LTLBackendCache(log_path=path)
            with patch(
                "jepa_lmc.evaluation.finite_trace.evaluate_model_pair",
                wraps=evaluate_model_pair,
            ) as shared:
                model, _ = fit_finite_model(
                    data, kind="jepa", seed=31, config=test_config()
                )
                predictions = predict_successors(model, data, kind="jepa")
                result = evaluate_predictor(
                    self.problem,
                    data.observed_pairs,
                    predictions,
                    real=self.real,
                    formal_system=prediction_system(self.problem, predictions),
                    heldout=heldout,
                    backend=backend,
                )
            self.assertEqual(shared.call_count, 1)
            self.assertEqual(set(result["formal"]["behavioral"]), set(SEMANTICS))
            for metrics in result["formal"]["behavioral"].values():
                for key in (
                    "simulation_real_to_learned",
                    "simulation_learned_to_real",
                    "initial_bisimulation",
                    "identity_simulation_real_to_learned",
                    "identity_simulation_learned_to_real",
                    "identity_bisimulation",
                ):
                    self.assertIsInstance(metrics[key], bool)
            self.assertEqual(
                result["formal"]["diagnostics"]["ltl"]["comparisons"],
                len(self.problem.states) * 6,
            )
            before = backend.calls
            backend.comparison(
                self.real, prediction_system(self.problem, predictions), self.problem
            )
            self.assertEqual(backend.calls, before)
            self.assertGreaterEqual(backend.hits, 2)
            with gzip.open(path, "rt") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(len(records), backend.calls)
            self.assertTrue(all("model" in r and "output" in r for r in records))
            json.dumps(result, allow_nan=False)

    def test_no_overwrite_of_result_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "marker.txt"
            path.write_text("preserve")
            with self.assertRaises(FileExistsError):
                run_experiment(directory)
            self.assertEqual(path.read_text(), "preserve")

    @unittest.skipUnless(find_nusmv_executable(), "nuXmv not installed")
    def test_aggregates_and_paired_comparison_use_case_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = LTLBackendCache(log_path=Path(directory) / "backend.gz")
            rows = []
            for name, predictions, graph in (
                ("mlp", self.truth, self.real),
                ("jepa", self.truth, self.real),
                (
                    "observed_edge",
                    self.partial,
                    observed_self_loop_completion(
                        self.problem, observed_edges(self.goods[:1])
                    ),
                ),
            ):
                rows.append(
                    {
                        "seed": 4,
                        "case": "tiny",
                        "family": "test",
                        "model": name,
                        "observed_pairs": len(self.observed),
                        "unseen_pairs": len(self.truth) - len(self.observed),
                        "training_trace_acceptance": trace_scores(
                            self.problem, self.goods[:1], predictions, self.observed
                        )["acceptance"],
                        **evaluate_predictor(
                            self.problem,
                            self.observed,
                            predictions,
                            real=self.real,
                            formal_system=graph,
                            heldout=self.goods[1:3],
                            backend=backend,
                        ),
                    }
                )
            summary = aggregate(rows[:1])
            self.assertEqual(summary["transitions"]["unseen"]["macro"]["mean"], 1)
            self.assertEqual(
                summary["transitions"]["unreachable_unseen"]["macro"]["evaluated"], 0
            )
            paired = paired_comparison(rows)
            self.assertEqual(paired["ties"], 1)
            self.assertEqual(paired["jepa_wins"], 0)
            self.assertEqual(paired["mlp_wins"], 0)


if __name__ == "__main__":
    unittest.main()
