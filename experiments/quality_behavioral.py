"""Apply the existing exact behavioral algorithms to the selected quality model."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

if __package__:
    from .top1_quality_diagnosis import (
        BASELINES,
        ROOT,
        SEEDS,
        digest,
        fresh_directory,
        load_model,
        write_csv,
        write_json,
    )
else:
    from top1_quality_diagnosis import (
        BASELINES,
        ROOT,
        SEEDS,
        digest,
        fresh_directory,
        load_model,
        write_csv,
        write_json,
    )

from jepa_lmc.benchmarks.ctl_suite import default_ctl_suite
from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.evaluation.dynamics import (
    evaluate_gridworld_transitions,
    transition_system_from_retrievals,
)
from jepa_lmc.verification.behavioral_relations import (
    audit_greatest_relation,
    bisimulation_by_partition,
    greatest_relation,
)
from jepa_lmc.verification.ctl import CTLModelChecker
from jepa_lmc.verification.gridworld import gridworld_to_transition_system

NAMES = ("real_to_top1", "top1_to_real", "bisimulation")


def export_relation(result, indices, name, initial):
    eliminations = []
    for removal in result.eliminations:
        row = asdict(removal)
        row["left"], row["right"] = indices[removal.left], indices[removal.right]
        row["target"] = None if removal.target is None else indices[removal.target]
        eliminations.append(row)
    return {
        "name": name,
        "kind": result.kind,
        "action_sensitive": result.action_sensitive,
        "pairs": sorted([indices[s], indices[t]] for s, t in result.pairs),
        "rounds": result.rounds,
        "eliminations": eliminations,
        "start_failure": next(
            (r for r in eliminations if r["left"] == r["right"] == initial), None
        ),
    }


def run(checkpoints, output, baseline=False):
    output = fresh_directory(output)
    torch.set_num_threads(4)
    saved = {}
    if baseline:
        with (
            ROOT / "outputs/behavioral_relations/top1/relations.jsonl"
        ).open() as handle:
            saved = {(r["seed"], r["case"]): r for r in map(json.loads, handle)}
    maps, relations, ctl = [], [], []
    checkpoint_hashes = {str(s): digest(p) for s, p in checkpoints.items()}
    certificates = partitions = preservation = 0
    with (output / "relations.jsonl").open("w") as raw:
        for seed in SEEDS:
            model, payload = load_model(checkpoints[seed])
            if payload["training"]["seed"] != seed:
                raise AssertionError("Checkpoint seed mismatch.")
            for case in radius_stress_cases():
                env = case.make_env()
                states = tuple(env.all_states())
                indices = {s: i for i, s in enumerate(states)}
                retrieval = evaluate_gridworld_transitions(model, env)
                real = gridworld_to_transition_system(env)
                top1 = transition_system_from_retrievals(env, retrieval)
                if (
                    real.states != top1.states
                    or real.initial_states != top1.initial_states
                    or any(real.propositions(s) != top1.propositions(s) for s in states)
                ):
                    raise AssertionError("State/label/initial identity changed.")
                truths = [
                    {
                        p.name: CTLModelChecker(g).satisfying_states(p.formula)
                        for p in default_ctl_suite()
                    }
                    for g in (real, top1)
                ]
                for state in states:
                    for prop in default_ctl_suite():
                        rv, tv = (
                            state in truths[0][prop.name],
                            state in truths[1][prop.name],
                        )
                        ctl.append(
                            {
                                "seed": seed,
                                "case": case.name,
                                "family": case.family,
                                "row": state[0],
                                "column": state[1],
                                "initial": state == env.start,
                                "property": prop.name,
                                "real": rv,
                                "top1": tv,
                                "agreement": rv == tv,
                            }
                        )
                maps.append(
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
                            for p in default_ctl_suite()
                            if (env.start in truths[0][p.name])
                            != (env.start in truths[1][p.name])
                        ],
                    }
                )
                record = {
                    "seed": seed,
                    "case": case.name,
                    "states": [list(s) for s in states],
                    "initial": indices[env.start],
                    "actions": list(env.ACTIONS),
                    "labels": [sorted(real.propositions(s)) for s in states],
                    "graphs": {
                        name: [
                            [
                                [e.action, indices[e.target]]
                                for e in graph.action_successors(s)
                            ]
                            for s in states
                        ]
                        for name, graph in (("real", real), ("top1", top1))
                    },
                    "relations": [],
                }
                for actions in (False, True):
                    results = {}
                    for name in NAMES:
                        left, right = (
                            (top1, real) if name == "top1_to_real" else (real, top1)
                        )
                        lt, rt = reversed(truths) if name == "top1_to_real" else truths
                        kind = (
                            "bisimulation" if name == "bisimulation" else "simulation"
                        )
                        result = greatest_relation(
                            left, right, kind=kind, action_sensitive=actions
                        )
                        audit_greatest_relation(left, right, result)
                        certificates += 1
                        if kind == "bisimulation":
                            if result.pairs != bisimulation_by_partition(
                                left, right, action_sensitive=actions
                            ):
                                raise AssertionError(
                                    "Independent bisimulation algorithms disagree."
                                )
                            partitions += 1
                        for prop in default_ctl_suite():
                            universal = prop.name in ("AG !danger", "AF goal")
                            for s, t in result.pairs:
                                lv, rv = s in lt[prop.name], t in rt[prop.name]
                                violated = (
                                    lv != rv
                                    if kind == "bisimulation"
                                    else rv and not lv
                                    if universal
                                    else lv and not rv
                                )
                                if violated:
                                    raise AssertionError("CTL preservation violation.")
                                preservation += 1
                        results[name] = result
                        relations.append(
                            {
                                "seed": seed,
                                "case": case.name,
                                "family": case.family,
                                "semantics": "action_labelled" if actions else "ctl",
                                "relation": name,
                                "states": len(states),
                                "initial_related": result.relates_initials(left, right),
                                "diagonal_related": sum(
                                    (s, s) in result.pairs for s in states
                                ),
                                "retained_pairs": len(result.pairs),
                                "rounds": result.rounds,
                            }
                        )
                        record["relations"].append(
                            export_relation(result, indices, name, indices[env.start])
                        )
                    if actions and (
                        results["real_to_top1"].pairs != results["bisimulation"].pairs
                        or results["real_to_top1"].pairs
                        != {(t, s) for s, t in results["top1_to_real"].pairs}
                    ):
                        raise AssertionError(
                            "Total action-deterministic consistency failed."
                        )
                if baseline and record != saved[seed, case.name]:
                    raise AssertionError(
                        "Baseline graph or full greatest relation changed."
                    )
                raw.write(json.dumps(record, allow_nan=False) + "\n")
            print("behavioral", seed, "complete", flush=True)
    if checkpoint_hashes != {str(s): digest(p) for s, p in checkpoints.items()}:
        raise AssertionError("A checkpoint changed during inference.")
    write_csv(output / "maps.csv", relations)
    write_csv(output / "ctl_queries.csv", ctl)
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__).resolve(),
                ROOT / "src/jepa_lmc/verification/behavioral_relations.py",
                ROOT / "src/jepa_lmc/verification/ctl.py",
                ROOT / "src/jepa_lmc/evaluation/dynamics.py",
            )
        },
        "checkpoint_sha256": checkpoint_hashes,
        "maps": maps,
        "audit": {
            "certificates": certificates,
            "partition_comparisons": partitions,
            "ctl_preservation_checks": preservation,
            "passed": True,
            "baseline_full_record_parity": baseline,
        },
        "seeds": [
            {
                "seed": seed,
                "initial_all_six_ctl": sum(
                    not m["initial_ctl_disagreements"]
                    for m in maps
                    if m["seed"] == seed
                ),
                "relations": {
                    semantics: {
                        name: sum(
                            r["initial_related"]
                            for r in relations
                            if r["seed"] == seed
                            and r["semantics"] == semantics
                            and r["relation"] == name
                        )
                        for name in NAMES
                    }
                    for semantics in ("ctl", "action_labelled")
                },
            }
            for seed in SEEDS
        ],
        "exports": {p.name: digest(p) for p in output.iterdir()},
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "outputs/top1_quality/round1"
    )
    parser.add_argument("--model", choices=("baseline", "best"), required=True)
    args = parser.parse_args()
    if args.model == "baseline":
        checkpoints = BASELINES
    else:
        selection = json.loads((args.run_dir / "selection.json").read_text())
        checkpoints = {
            int(s): ROOT / p for s, p in selection["selected_checkpoints"].items()
        }
    run(checkpoints, args.run_dir / "behavioral" / args.model, args.model == "baseline")


if __name__ == "__main__":
    main()
