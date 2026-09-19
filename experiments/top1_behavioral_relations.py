"""Exactly compare frozen Top-1 and real graphs by simulation and bisimulation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from cegar_refinement import digest, load_checkpoint, model_digest, write_json

from jepa_lmc.benchmarks.ctl_suite import default_ctl_suite
from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.evaluation.dynamics import (
    evaluate_gridworld_transitions,
    transition_system_from_retrievals,
)
from jepa_lmc.learning.model import ActionJEPA
from jepa_lmc.verification.behavioral_relations import (
    audit_greatest_relation,
    bisimulation_by_partition,
    greatest_relation,
)
from jepa_lmc.verification.ctl import CTLModelChecker
from jepa_lmc.verification.gridworld import gridworld_to_transition_system

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/top1_relations_protocol.json"


def relation_summary(rows):
    return {
        "maps": len(rows),
        "initial_related_maps": sum(r["initial_related"] for r in rows),
        "all_diagonal_related_maps": sum(r["all_diagonal_related"] for r in rows),
        "diagonal_related_states": sum(r["diagonal_related"] for r in rows),
        "states": sum(r["states"] for r in rows),
        "diagonal_related_fraction": sum(r["diagonal_related"] for r in rows)
        / sum(r["states"] for r in rows),
        "identity_relation_holds_maps": sum(r["identity_relation_holds"] for r in rows),
        "initial_related_despite_changed_edges_maps": sum(
            r["initial_related"] and not r["edges_equal"] for r in rows
        ),
        "retained_pairs": sum(r["retained_pairs"] for r in rows),
        "max_elimination_round": max(r["elimination_rounds"] for r in rows),
    }


def export_relation(result, indices):
    removals = []
    for row in result.eliminations:
        entry = asdict(row)
        entry["left"], entry["right"] = indices[row.left], indices[row.right]
        entry["target"] = None if row.target is None else indices[row.target]
        removals.append(entry)
    return {
        "kind": result.kind,
        "action_sensitive": result.action_sensitive,
        "pairs": sorted((indices[s], indices[t]) for s, t in result.pairs),
        "eliminations": removals,
        "rounds": result.rounds,
    }


def audit_ctl_pairs(result, left_truth, right_truth):
    count = 0
    for name in left_truth:
        universal = name in ("AG !danger", "AF goal")
        for s, t in result.pairs:
            left, right = s in left_truth[name], t in right_truth[name]
            violation = (
                left != right
                if result.kind == "bisimulation"
                else (right and not left)
                if universal
                else (left and not right)
            )
            if violation:
                raise AssertionError(f"Relation violates CTL preservation: {name}")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/behavioral_relations/top1")
    )
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    payloads = [load_checkpoint(p, protocol) for p in args.checkpoints]
    seeds = [p["training"]["seed"] for p in payloads]
    if len(seeds) != len(set(seeds)) or set(seeds) != set(protocol["model_seeds"]):
        raise ValueError("Supply exactly the three fixed checkpoints.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Choose a fresh output directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    sources = [
        "configs/top1_relations_protocol.json",
        "experiments/top1_behavioral_relations.py",
        "experiments/cegar_refinement.py",
        "src/jepa_lmc/verification/behavioral_relations.py",
        "src/jepa_lmc/verification/transition_system.py",
        "src/jepa_lmc/verification/gridworld.py",
        "src/jepa_lmc/verification/ctl.py",
        "src/jepa_lmc/evaluation/dynamics.py",
        "src/jepa_lmc/learning/model.py",
        "src/jepa_lmc/learning/data.py",
        "src/jepa_lmc/envs/gridworld.py",
        "src/jepa_lmc/benchmarks/radius_stress.py",
        "src/jepa_lmc/benchmarks/ctl_suite.py",
    ]
    report = {
        "protocol": protocol,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {p: digest(ROOT / p) for p in sources},
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "threads": 4,
        },
        "seeds": [],
    }
    rows, map_scores = [], []
    certificate_checks = partition_checks = ctl_checks = 0
    suite = default_ctl_suite()
    with (args.output_dir / "relations.jsonl").open("w", encoding="utf-8") as raw:
        for path, payload in sorted(
            zip(args.checkpoints, payloads, strict=True),
            key=lambda x: x[1]["training"]["seed"],
        ):
            seed = payload["training"]["seed"]
            model = ActionJEPA(**payload["model_kwargs"])
            model.load_state_dict(payload["model_state_dict"])
            before = model_digest(model)
            started = time.perf_counter()
            for case in radius_stress_cases():
                env = case.make_env()
                states = tuple(env.all_states())
                indices = {s: i for i, s in enumerate(states)}
                retrieval = evaluate_gridworld_transitions(model, env)
                real = gridworld_to_transition_system(env)
                learned = transition_system_from_retrievals(env, retrieval)
                if (
                    real.states != learned.states
                    or real.initial_states != learned.initial_states
                    or any(
                        real.propositions(s) != learned.propositions(s) for s in states
                    )
                ):
                    raise AssertionError("Only transitions may differ.")
                truths = []
                for graph in (real, learned):
                    checker = CTLModelChecker(graph)
                    truths.append(
                        {p.name: checker.satisfying_states(p.formula) for p in suite}
                    )
                map_scores.append(
                    {
                        "seed": seed,
                        "case": case.name,
                        "family": case.family,
                        "pairs": retrieval.total,
                        "correct_top1_pairs": sum(
                            o.correct for o in retrieval.outcomes
                        ),
                        "initial_ctl_disagreements": [
                            {
                                "property": p.name,
                                "real": env.start in truths[0][p.name],
                                "top1": env.start in truths[1][p.name],
                            }
                            for p in suite
                            if (env.start in truths[0][p.name])
                            != (env.start in truths[1][p.name])
                        ],
                    }
                )
                raw_map = {
                    "seed": seed,
                    "case": case.name,
                    "states": states,
                    "initial": indices[env.start],
                    "actions": list(env.ACTIONS),
                    "labels": [sorted(real.propositions(s)) for s in states],
                    "graphs": {},
                    "relations": [],
                }
                for name, graph in (("real", real), ("top1", learned)):
                    raw_map["graphs"][name] = [
                        [
                            (e.action, indices[e.target])
                            for e in graph.action_successors(s)
                        ]
                        for s in states
                    ]
                for actions in (False, True):
                    results = {}
                    for name in protocol["relations"]:
                        left, right = (
                            (learned, real)
                            if name == "top1_to_real"
                            else (real, learned)
                        )
                        kind = (
                            "bisimulation" if name == "bisimulation" else "simulation"
                        )
                        result = greatest_relation(
                            left, right, kind=kind, action_sensitive=actions
                        )
                        audit_greatest_relation(left, right, result)
                        certificate_checks += 1
                        if kind == "bisimulation":
                            if result.pairs != bisimulation_by_partition(
                                left, right, action_sensitive=actions
                            ):
                                raise AssertionError(
                                    "Independent bisimulation algorithms disagree."
                                )
                            partition_checks += 1
                        ltruth, rtruth = (
                            reversed(truths) if name == "top1_to_real" else truths
                        )
                        ctl_checks += audit_ctl_pairs(result, ltruth, rtruth)
                        edge_sets = [
                            {
                                (s, e.action if actions else None, e.target)
                                for s in states
                                for e in g.action_successors(s)
                            }
                            for g in (left, right)
                        ]
                        missing, extra = (
                            edge_sets[0] - edge_sets[1],
                            edge_sets[1] - edge_sets[0],
                        )
                        diagonal = sum((s, s) in result.pairs for s in states)
                        initial = (env.start, env.start) in result.pairs
                        if initial != result.relates_initials(left, right):
                            raise AssertionError(
                                "Single-initial pointed comparison disagrees."
                            )
                        row = {
                            "seed": seed,
                            "case": case.name,
                            "family": case.family,
                            "semantics": "action_labelled" if actions else "ctl",
                            "relation": name,
                            "states": len(states),
                            "initial_related": initial,
                            "diagonal_related": diagonal,
                            "all_diagonal_related": diagonal == len(states),
                            "retained_pairs": len(result.pairs),
                            "elimination_rounds": result.rounds,
                            "identity_relation_holds": not missing
                            and (kind == "simulation" or not extra),
                            "edges_equal": not missing and not extra,
                            "missing_left_edges": len(missing),
                            "extra_right_edges": len(extra),
                        }
                        rows.append(row)
                        exported = export_relation(result, indices)
                        exported["name"] = name
                        exported["start_failure"] = next(
                            (
                                x
                                for x in exported["eliminations"]
                                if x["left"] == x["right"] == indices[env.start]
                            ),
                            None,
                        )
                        raw_map["relations"].append(exported)
                        results[name] = result
                    if actions:
                        # Both models have one outcome for each GridWorld action.
                        if results["real_to_top1"].pairs != results[
                            "bisimulation"
                        ].pairs or results["real_to_top1"].pairs != {
                            (t, s) for s, t in results["top1_to_real"].pairs
                        }:
                            raise AssertionError(
                                "Total action-deterministic consistency failed."
                            )
                raw.write(json.dumps(raw_map, allow_nan=False) + "\n")
            if model_digest(model) != before:
                raise AssertionError("The frozen JEPA weights changed.")
            seed_rows = [r for r in rows if r["seed"] == seed]
            summary = {
                semantics: {
                    name: relation_summary(
                        [
                            r
                            for r in seed_rows
                            if r["semantics"] == semantics and r["relation"] == name
                        ]
                    )
                    for name in protocol["relations"]
                }
                for semantics in ("ctl", "action_labelled")
            }
            report["seeds"].append(
                {
                    "seed": seed,
                    "checkpoint_sha256": digest(path),
                    "model_state_sha256": before,
                    "weights_unchanged": True,
                    "elapsed_seconds": time.perf_counter() - started,
                    "summary": summary,
                    "by_family": {
                        family: {
                            semantics: {
                                name: relation_summary(
                                    [
                                        r
                                        for r in seed_rows
                                        if r["family"] == family
                                        and r["semantics"] == semantics
                                        and r["relation"] == name
                                    ]
                                )
                                for name in protocol["relations"]
                            }
                            for semantics in ("ctl", "action_labelled")
                        }
                        for family in sorted({r["family"] for r in seed_rows})
                    },
                }
            )
            print(
                seed,
                json.dumps(
                    {
                        s: {n: v["initial_related_maps"] for n, v in methods.items()}
                        for s, methods in summary.items()
                    }
                ),
                flush=True,
            )
    with (args.output_dir / "maps.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report["maps"] = map_scores
    report["audit"] = {
        "greatest_relation_certificates_checked": certificate_checks,
        "independent_partition_comparisons": partition_checks,
        "ctl_pair_formula_checks": ctl_checks,
        "all_passed": True,
    }
    report["exports"] = {
        p: digest(args.output_dir / p) for p in ("maps.csv", "relations.jsonl")
    }
    write_json(args.output_dir / "report.json", report)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
