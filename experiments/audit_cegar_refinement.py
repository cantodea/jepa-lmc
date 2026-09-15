"""Replay every reported query budget and audit the saved CEGAR experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from cegar_refinement import (
    ROOT,
    assess,
    audit_result,
    digest,
    load_checkpoint,
    summarize,
    write_json,
)

from jepa_lmc.benchmarks.radius_stress import PRIMARY_PROPERTIES, radius_stress_cases
from jepa_lmc.evaluation.refinement_priority import (
    latent_rank_costs,
    shuffled_rank_costs,
)
from jepa_lmc.learning.data import gridworld_observation
from jepa_lmc.learning.model import ActionJEPA
from jepa_lmc.verification.successor_refinement import (
    SuccessorRefinement,
    task_from_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    report = json.loads((args.run_dir / "report.json").read_text())
    protocol = report["protocol"]
    for path, expected in report["source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise AssertionError(f"Experiment source changed: {path}")
    for path, expected in report["exports"].items():
        if digest(args.run_dir / path) != expected:
            raise AssertionError(f"Export checksum mismatch: {path}")
    with (args.run_dir / "tasks.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("seed", "oracle_queries", "full_table_queries", "spurious_paths"):
            row[key] = int(row[key])
        for key in ("query_fraction", "mean_upper_candidates", "verification_seconds"):
            row[key] = float(row[key])
        for key in ("value", "ground_truth"):
            row[key] = {"True": True, "False": False, "": None}[row[key]]
    traces = {}
    for line in (args.run_dir / "traces.jsonl").read_text().splitlines():
        trace = json.loads(line)
        key = tuple(trace[k] for k in ("seed", "case", "method", "property"))
        if key in traces:
            raise AssertionError("Duplicate task trace.")
        traces[key] = trace
    expected_tasks = (
        len(protocol["model_seeds"])
        * len(radius_stress_cases())
        * len(protocol["methods"])
        * len(PRIMARY_PROPERTIES)
    )
    if not len(rows) == len(traces) == report["task_count"] == expected_tasks:
        raise AssertionError("Task count mismatch.")
    fractions = protocol["budget_fractions_of_full_state_action_table"]
    summaries = {}
    for seed_report in report["seeds"]:
        seed = seed_report["seed"]
        summaries[seed] = {
            method: summarize(
                [r for r in rows if r["seed"] == seed and r["method"] == method],
                fractions,
            )
            for method in protocol["methods"]
        }
        if summaries[seed] != seed_report["summary"]:
            raise AssertionError("Aggregate does not match task CSV.")
    recomputed_assessment = json.loads(json.dumps(assess(summaries, protocol)))
    if recomputed_assessment != report["assessment"]:
        raise AssertionError("Screen does not match recomputed aggregates.")
    payloads = {}
    for path in args.checkpoints:
        payload = load_checkpoint(path, protocol)
        seed = payload["training"]["seed"]
        reference = next(s for s in report["seeds"] if s["seed"] == seed)
        if seed in payloads or digest(path) != reference["checkpoint_sha256"]:
            raise AssertionError("Duplicate or different checkpoint.")
        payloads[seed] = payload
    if set(payloads) != set(protocol["model_seeds"]):
        raise AssertionError("Supply every recorded checkpoint.")
    torch.set_num_threads(4)
    replayed = Counter()
    resolved = Counter()
    for seed, payload in sorted(payloads.items()):
        model = ActionJEPA(**payload["model_kwargs"])
        model.load_state_dict(payload["model_state_dict"])
        for case_index, case in enumerate(radius_stress_cases()):
            env = case.make_env()
            states = tuple(env.all_states())
            indices = {s: i for i, s in enumerate(states)}
            labels = tuple(frozenset((env.label(s),)) for s in states)
            observations = torch.stack([gridworld_observation(env, s) for s in states])
            costs = latent_rank_costs(model, observations)
            shuffled = shuffled_rank_costs(costs, seed * 1000 + case_index)
            exact = {
                (s, a): indices[env.transition(states[s], a)]
                for s in range(len(states))
                for a in range(4)
            }
            for method in protocol["methods"]:
                priorities = (
                    costs
                    if method == "cegar_jepa"
                    else shuffled
                    if method == "cegar_shuffled"
                    else None
                )
                for name in PRIMARY_PROPERTIES:
                    full = traces[(seed, case.name, method, name)]["result"]
                    task = task_from_labels(name, indices[env.start], labels)
                    for fraction in fractions:
                        budget = math.floor(fraction * len(states) * 4)
                        refiner = SuccessorRefinement(
                            len(states),
                            4,
                            lambda s, a: exact[(s, a)],
                            costs=priorities,
                            budget=budget,
                        )
                        result = (
                            refiner.solve_direct_bfs(task)
                            if method == "direct_bfs"
                            else refiner.solve(task)
                        )
                        audit_result(refiner, result, task, exact)
                        expected_value = (
                            full["value"] if full["queries"] <= budget else None
                        )
                        if result.value != expected_value:
                            raise AssertionError(
                                "Budget resolution estimate was wrong."
                            )
                        if result.queries != min(budget, full["queries"]):
                            raise AssertionError("Budget query accounting mismatch.")
                        observed = [asdict(x) for x in result.observations]
                        if observed != full["observations"][: result.queries]:
                            raise AssertionError("Budget changed the query prefix.")
                        key = f"{seed}/{method}/{fraction}"
                        replayed[key] += 1
                        resolved[key] += result.value is not None
        print(f"Audited seed {seed}", flush=True)
    output = {
        "original_report_sha256": digest(args.run_dir / "report.json"),
        "auditor_sha256": digest(Path(__file__)),
        "export_and_source_checksums_match": True,
        "aggregate_and_screen_recomputation_matches": True,
        "task_count": expected_tasks,
        "actual_budget_replays": sum(replayed.values()),
        "all_budget_verdicts_and_query_prefixes_match": True,
        "all_replayed_certificates_and_inclusion_audits_pass": True,
        "replayed_by_seed_method_fraction": dict(replayed),
        "resolved_by_seed_method_fraction": dict(resolved),
    }
    write_json(args.run_dir / "audit.json", output)
    print(f"Passed {output['actual_budget_replays']} bounded replays.")


if __name__ == "__main__":
    main()
