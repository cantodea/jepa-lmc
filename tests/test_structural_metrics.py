from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.evaluate_structural import (
    evaluate_saved_run,
    point_prediction_counts,
    read_linked_ltl,
)
from experiments.evaluate_top1_ltl import digest, read_run
from experiments.oracle_local_abstraction import analyze_graph
from experiments.top1_quality_diagnosis import evaluate_model
from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.benchmarks.random_gridworld import make_pilot_benchmark_splits
from jepa_lmc.evaluation.latent_radius import LatentDistances
from jepa_lmc.evaluation.local_abstraction import oracle_local_mask
from jepa_lmc.evaluation.structural import (
    behavioral_metrics,
    count_score,
    evaluate_model_pair,
    model_evaluation_report,
    save_model_evaluation,
)
from jepa_lmc.learning.model import ActionJEPA
from jepa_lmc.verification.behavioral_relations import greatest_relation
from jepa_lmc.verification.ctl import EX, Atom, CTLModelChecker
from tests.test_behavioral_relations import system
from tests.test_top1_ltl_experiment import make_run


class StructuralMetricTests(unittest.TestCase):
    def test_different_sizes_and_edges_can_be_bisimilar(self):
        left = system({0: [("go", 1)], 1: [("stay", 1)]}, {1: {"goal"}})
        right = system(
            {10: [("go", 11), ("go", 12)], 11: [("stay", 12)], 12: [("stay", 11)]},
            {11: {"goal"}, 12: {"goal"}},
            (10,),
        )
        result = evaluate_model_pair(left, right)
        for metrics in result["behavioral"].values():
            self.assertTrue(metrics["simulation_real_to_learned"])
            self.assertTrue(metrics["simulation_learned_to_real"])
            self.assertTrue(metrics["initial_bisimulation"])
            self.assertIsNone(metrics["identity_bisimulation"])
        summary = model_evaluation_report([result])["summary"]
        self.assertEqual(
            summary["primary_metrics"]["action_insensitive"]["identity_bisimulation"][
                "evaluated"
            ],
            0,
        )
        self.assertIsNone(
            summary["diagnostics"]["initial_all_six_ltl_agreement"]["fraction"]
        )

    def test_six_ctl_verdicts_agree_but_bisimulation_fails(self):
        labels = {0: {"safe"}, 1: {"safe"}, 2: {"goal"}}
        left = system({0: [(0, 2)], 1: [(0, 2)], 2: [(0, 2)]}, labels)
        right = system({0: [(0, 1)], 1: [(0, 2)], 2: [(0, 2)]}, labels)
        case = {
            "seed": 4,
            "case": "delayed_goal",
            **evaluate_model_pair(
                left, right, top1_counts=point_prediction_counts(left, right)
            ),
        }
        self.assertTrue(case["diagnostics"]["initial_all_six_ctl_agreement"])
        self.assertFalse(
            case["behavioral"]["action_insensitive"]["initial_bisimulation"]
        )
        self.assertTrue(CTLModelChecker(left).holds(0, EX(Atom("goal"))))
        self.assertFalse(CTLModelChecker(right).holds(0, EX(Atom("goal"))))
        report = model_evaluation_report([case])
        self.assertEqual(
            report["summary"]["diagnostics"]["ctl_agrees_but_not_bisimilar"], 1
        )
        self.assertEqual(report["summary"]["diagnostics"]["top1"]["accuracy"], 2 / 3)
        self.assertEqual(report["summary"]["mismatch_cases"][0]["case"], "delayed_goal")

    def test_actions_can_change_bisimulation_without_changing_ctl(self):
        labels = {0: {"safe"}, 1: {"goal"}}
        left = system({0: [("a", 0), ("b", 1)], 1: [("a", 1), ("b", 1)]}, labels)
        right = system({0: [("b", 0), ("a", 1)], 1: [("a", 1), ("b", 1)]}, labels)
        metrics = evaluate_model_pair(left, right)
        self.assertTrue(
            metrics["behavioral"]["action_insensitive"]["identity_bisimulation"]
        )
        self.assertFalse(
            metrics["behavioral"]["action_sensitive"]["initial_bisimulation"]
        )
        self.assertFalse(
            metrics["behavioral"]["action_sensitive"]["identity_bisimulation"]
        )
        self.assertTrue(metrics["diagnostics"]["initial_all_six_ctl_agreement"])

    def test_identity_membership_does_not_mean_equal_edges(self):
        labels = {1: {"goal"}, 2: {"goal"}}
        left = system({0: [(0, 1)], 1: [(0, 1)], 2: [(0, 2)]}, labels)
        right = system({0: [(0, 2)], 1: [(0, 1)], 2: [(0, 2)]}, labels)
        for metric in behavioral_metrics(left, right).values():
            self.assertTrue(metric["identity_bisimulation"])
            self.assertTrue(metric["identity_simulation_real_to_learned"])
            self.assertFalse(metric["strict_identity_bisimulation"])
            self.assertFalse(metric["strict_identity_simulation_real_to_learned"])

    def test_initial_and_global_are_different_with_unreachable_states(self):
        labels = {1: {"danger"}}
        left = system({0: [(0, 0)], 1: [(0, 1)], 2: [(0, 2)]}, labels)
        right = system({0: [(0, 0)], 1: [(0, 2)], 2: [(0, 2)]}, labels)
        metric = behavioral_metrics(left, right)["action_insensitive"]
        self.assertTrue(metric["initial_bisimulation"])
        self.assertFalse(metric["identity_bisimulation"])
        self.assertEqual(
            metric["relations"]["bisimulation"]["unrelated_identity_states"], ["1"]
        )

    def test_simulation_direction_and_nondeterministic_candidate(self):
        labels = {1: {"danger"}}
        real = system({0: [(0, 0)], 1: [(0, 1)]}, labels)
        candidate = system({0: [(0, 0), (0, 1)], 1: [(0, 1)]}, labels)
        for metric in behavioral_metrics(real, candidate).values():
            self.assertTrue(metric["simulation_real_to_learned"])
            self.assertTrue(metric["identity_simulation_real_to_learned"])
            self.assertTrue(metric["strict_identity_simulation_real_to_learned"])
            self.assertFalse(metric["simulation_learned_to_real"])
            self.assertFalse(metric["initial_bisimulation"])
        with self.assertRaisesRegex(ValueError, "deterministic"):
            point_prediction_counts(real, candidate)

    def test_multiple_designated_initials_require_both_sides_for_bisimulation(self):
        transitions, labels = {0: [(0, 0)], 1: [(0, 1)]}, {1: {"p"}}
        left = system(transitions, labels)
        right = system(transitions, labels, (0, 1))
        metric = behavioral_metrics(left, right)["action_insensitive"]
        self.assertTrue(metric["simulation_real_to_learned"])
        self.assertFalse(metric["simulation_learned_to_real"])
        self.assertFalse(metric["initial_bisimulation"])
        self.assertTrue(metric["identity_bisimulation"])

    def test_mutual_simulation_is_not_relabelled_as_bisimulation(self):
        labels = {0: {"root"}, 2: {"b"}, 3: {"c"}}
        left = system(
            {0: [(0, 1)], 1: [(0, 2), (0, 3)], 2: [(0, 2)], 3: [(0, 3)]}, labels
        )
        right = system(
            {
                0: [(0, 1), (0, 4)],
                1: [(0, 2), (0, 3)],
                2: [(0, 2)],
                3: [(0, 3)],
                4: [(0, 2)],
            },
            labels,
        )
        metric = behavioral_metrics(left, right)["action_insensitive"]
        self.assertTrue(metric["simulation_real_to_learned"])
        self.assertTrue(metric["simulation_learned_to_real"])
        self.assertFalse(metric["initial_bisimulation"])

    def test_invalid_or_unavailable_metrics_are_not_counted_as_success(self):
        with self.assertRaises(ValueError):
            count_score(["False"])
        self.assertEqual(count_score([True, False, None])["evaluated"], 2)
        graph = system({0: [(0, 0)]})
        with self.assertRaises(ValueError):
            behavioral_metrics(graph, graph, relations={})
        with self.assertRaises(ValueError):
            evaluate_model_pair(graph, graph, top1_counts=(2, 1))
        relation = greatest_relation(graph, graph, kind="bisimulation")
        self.assertTrue(relation.relates_identity(graph, graph))


class StructuralPipelineTests(unittest.TestCase):
    def test_oracle_pipeline_records_both_semantics_and_predictor_accuracy(self):
        case = radius_stress_cases()[0]
        env = case.make_env()
        states = tuple(env.all_states())
        indices = {s: i for i, s in enumerate(states)}
        pairs = tuple((s, a) for s in states for a in env.ACTIONS)
        true = tuple(indices[env.transition(s, a)] for s, a in pairs)
        distances = torch.full((len(pairs), len(states)), 2.0)
        distances[torch.arange(len(pairs)), list(true)] = 1.0
        # One wrong point prediction; the oracle ball still contains truth.
        distances[0, (true[0] + 1) % len(states)] = 0.0
        table = LatentDistances(states, pairs, true, distances)
        acc = {
            "graphs": io.StringIO(),
            "maps": [],
            "ctl": [],
            "relations": [],
            "certificates": 0,
            "partitions": 0,
            "preservation": 0,
        }
        result, _ = analyze_graph(
            case, table, oracle_local_mask(table), 1, "ranking_local", acc
        )
        evaluation = acc["maps"][0]["evaluation"]
        self.assertEqual(result["covered_pairs"], len(pairs))
        self.assertEqual(evaluation["diagnostics"]["top1"]["correct"], len(pairs) - 1)
        for metrics in evaluation["behavioral"].values():
            self.assertTrue(metrics["identity_simulation_real_to_learned"])
        self.assertEqual(acc["certificates"], 6)
        self.assertEqual(acc["partitions"], 2)

    def test_saved_graph_replay_writes_counts_without_mutating_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "input"
            make_run(run)
            before = {p.name: digest(p) for p in run.iterdir()}
            report = evaluate_saved_run(run)
            self.assertEqual(report["summary"]["cases"], 1)
            self.assertEqual(report["summary"]["diagnostics"]["top1"]["accuracy"], 0.5)
            self.assertFalse(
                report["cases"][0]["behavioral"]["action_insensitive"][
                    "initial_bisimulation"
                ]
            )
            destination = root / "result.json"
            save_model_evaluation(destination, report["cases"])
            self.assertEqual(
                json.loads(destination.read_text())["summary"], report["summary"]
            )
            with self.assertRaises(FileExistsError):
                save_model_evaluation(destination, report["cases"])
            self.assertEqual(before, {p.name: digest(p) for p in run.iterdir()})

    def test_saved_ltl_requires_graph_identity_and_complete_query_coverage(self):
        import csv

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, ltl = root / "input", root / "ltl"
            make_run(run)
            source, records = read_run(run)
            ltl.mkdir()
            report_path = ltl / "report.json"
            report_path.write_text(json.dumps({"source_relations_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "different graph"):
                evaluate_saved_run(run, ltl_dir=ltl)
            query = ltl / "queries.csv"
            with query.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "seed",
                        "case",
                        "state_id",
                        "property",
                        "real",
                        "top1",
                        "agreement",
                        "initial",
                        "row",
                        "column",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "seed": 7,
                        "case": "safety_mismatch",
                        "state_id": 0,
                        "property": "G !danger",
                        "real": False,
                        "top1": True,
                        "agreement": False,
                        "initial": True,
                        "row": 4,
                        "column": 5,
                    }
                )
            source_hash = source["exports"]["relations.jsonl"]
            report_path.write_text(
                json.dumps(
                    {
                        "source_relations_sha256": source_hash,
                        "exports": {"queries.csv": digest(query)},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "missing"):
                read_linked_ltl(ltl, source_hash, records)

    def test_ranking_evaluator_scores_every_map_and_keeps_weights_frozen(self):
        spec = make_pilot_benchmark_splits().validation[0]
        model = ActionJEPA(height=spec.height, width=spec.width, latent_dim=8)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        metadata = {
            "split": "validation",
            "map": "fixture",
            "family": "fixture",
            "rotation": 0,
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "experiments.top1_quality_diagnosis.catalogue",
                return_value=[(metadata, spec)],
            ),
        ):
            report = evaluate_model(model, 9, Path(temporary) / "evaluation")
        scorecard = report["model_evaluation"]
        self.assertEqual(scorecard["summary"]["cases"], 1)
        self.assertEqual(scorecard["cases"][0]["case"], "fixture")
        self.assertAlmostEqual(
            scorecard["summary"]["diagnostics"]["top1"]["accuracy"],
            report["by_map"][0]["accuracy"],
        )
        self.assertTrue(
            all(torch.equal(before[k], v) for k, v in model.state_dict().items())
        )


if __name__ == "__main__":
    unittest.main()
