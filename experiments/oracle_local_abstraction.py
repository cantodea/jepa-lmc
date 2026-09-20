"""Frozen ranking JEPA: oracle local balls, matched global controls and exact audits."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from pathlib import Path

import torch

from experiments.quality_behavioral import NAMES, export_relation
from experiments.top1_quality_diagnosis import (
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
from jepa_lmc.evaluation.latent_radius import (
    candidate_transition_system,
    collect_latent_distances,
    evaluate_candidates,
    oracle_radii,
)
from jepa_lmc.evaluation.local_abstraction import (
    candidate_rows,
    identity_simulation,
    oracle_local_mask,
    ranking_rows,
    summarize_candidates,
)
from jepa_lmc.evaluation.radius_stress import (
    annotate_stress_result,
    summarize_stress_maps,
)
from jepa_lmc.learning.data import gridworld_observation
from jepa_lmc.verification.behavioral_relations import (
    audit_greatest_relation,
    bisimulation_by_partition,
    greatest_relation,
)
from jepa_lmc.verification.gridworld import gridworld_to_transition_system

PROTOCOL = ROOT / "configs/local_abstraction_protocol.json"
DEFAULT_RUN = ROOT / "outputs/local_abstraction/oracle_ranking_v1"
HISTORICAL = ROOT / "outputs/latent_radius/stress"
RANKING_RUN = ROOT / "outputs/top1_quality/round2"
BACKEND_VARIANTS = ("ranking_local", "ranking_global", "regression_global")


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


@torch.no_grad()
def distance_record(model, env, table, metadata):
    observations = torch.stack([gridworld_observation(env, s) for s in table.states])
    sources = torch.arange(len(table.states)).repeat_interleave(4)
    actions = torch.tensor([a for _, a in table.pairs])
    model.eval()
    target = model.target_encoder(observations)
    prediction = model.predictor(model.context_encoder(observations)[sources], actions)
    independent = (
        (prediction.double()[:, None] - target.double()[None]).square().sum(-1).sqrt()
    )
    maximum_error = float((independent - table.distances).abs().max())
    if maximum_error > 1e-12:
        raise AssertionError("Independent distance arithmetic disagrees.")
    order = table.distances.argsort(dim=1, stable=True)
    if not torch.equal(order, independent.argsort(dim=1, stable=True)):
        raise AssertionError("Independent complete distance ordering differs.")
    canonical = torch.cdist(prediction, target).argsort(1)
    if not torch.equal(canonical[:, 0], order[:, 0]):
        raise AssertionError("Oracle table changed canonical Top-1 retrieval.")
    indices = {s: i for i, s in enumerate(table.states)}
    return {
        **metadata,
        "states": table.states,
        "pairs": [(indices[s], a) for s, a in table.pairs],
        "true_indices": table.true_indices,
        "distances": table.distances.tolist(),
        "ordered_indices": order.tolist(),
        "predicted_embeddings": prediction.tolist(),
        "target_embeddings": target.tolist(),
        "independent_max_absolute_error": maximum_error,
        "canonical_top1_disagreements": 0,
    }


def analyze_graph(case, table, mask, seed, variant, accumulator):
    env = case.make_env()
    states = table.states
    indices = {s: i for i, s in enumerate(states)}
    real = gridworld_to_transition_system(env)
    candidate = candidate_transition_system(env, table, mask)
    result = annotate_stress_result(case, evaluate_candidates(env, table, mask))
    direct = identity_simulation(real, candidate)
    plain = identity_simulation(real, candidate, action_sensitive=False)
    if (
        direct["holds"] != result["action_inclusion"]
        or plain["holds"] != result["relation_inclusion"]
    ):
        raise AssertionError("Identity matching and edge inclusion disagree.")
    if variant != "ranking_top1" and (
        not direct["holds"] or result["ctl"]["one_sided_violations"]
    ):
        raise AssertionError("An oracle overapproximation is unsound.")
    truths = [{}, {}]
    initial_disagreements = []
    for prop in default_ctl_suite():
        subset = [r for r in result["ctl_outcomes"] if r["property"] == prop.name]
        truths[0][prop.name] = {r["state"] for r in subset if r["ground_truth"]}
        truths[1][prop.name] = {r["state"] for r in subset if r["learned"]}
        for r in subset:
            rv, cv = r["ground_truth"], r["learned"]
            accumulator["ctl"].append(
                {
                    "seed": seed,
                    "case": case.name,
                    "family": case.family,
                    "row": r["state"][0],
                    "column": r["state"][1],
                    "initial": r["state"] == env.start,
                    "property": prop.name,
                    "real": rv,
                    "top1": cv,
                    "agreement": rv == cv,
                }
            )
            if r["state"] == env.start and rv != cv:
                initial_disagreements.append(
                    {"property": prop.name, "real": rv, "top1": cv}
                )
    record = {
        "seed": seed,
        "case": case.name,
        "candidate_variant": variant,
        "legacy_graph_key": (
            "top1 denotes this variant's graph, including nondeterministic candidates"
        ),
        "states": states,
        "initial": indices[env.start],
        "actions": list(env.ACTIONS),
        "labels": [sorted(real.propositions(s)) for s in states],
        "graphs": {
            name: [
                [[e.action, indices[e.target]] for e in graph.action_successors(s)]
                for s in states
            ]
            for name, graph in (("real", real), ("top1", candidate))
        },
        "identity_action_simulation": direct,
        "identity_plain_simulation": plain,
        "relations": [],
    }
    for actions in (False, True):
        for name in NAMES:
            left, right = (
                (candidate, real) if name == "top1_to_real" else (real, candidate)
            )
            lt, rt = reversed(truths) if name == "top1_to_real" else truths
            kind = "bisimulation" if name == "bisimulation" else "simulation"
            relation = greatest_relation(
                left, right, kind=kind, action_sensitive=actions
            )
            audit_greatest_relation(left, right, relation)
            accumulator["certificates"] += 1
            if kind == "bisimulation":
                if relation.pairs != bisimulation_by_partition(
                    left, right, action_sensitive=actions
                ):
                    raise AssertionError(
                        "Independent bisimulation partition disagrees."
                    )
                accumulator["partitions"] += 1
            if (
                name == "real_to_top1"
                and direct["holds"]
                and any((s, s) not in relation.pairs for s in states)
            ):
                raise AssertionError(
                    "Greatest simulation omitted the identity relation."
                )
            for prop in default_ctl_suite():
                universal = prop.name in ("AG !danger", "AF goal")
                for s, t in relation.pairs:
                    lv, rv = s in lt[prop.name], t in rt[prop.name]
                    violated = (
                        lv != rv
                        if kind == "bisimulation"
                        else (rv and not lv if universal else lv and not rv)
                    )
                    if violated:
                        raise AssertionError(
                            "CTL preservation over a retained relation failed."
                        )
                    accumulator["preservation"] += 1
            accumulator["relations"].append(
                {
                    "seed": seed,
                    "case": case.name,
                    "family": case.family,
                    "semantics": "action_labelled" if actions else "ctl",
                    "relation": name,
                    "initial_related": relation.relates_initials(left, right),
                    "states": len(states),
                    "diagonal_related": sum((s, s) in relation.pairs for s in states),
                    "retained_pairs": len(relation.pairs),
                    "rounds": relation.rounds,
                }
            )
            record["relations"].append(
                export_relation(relation, indices, name, indices[env.start])
            )
    accumulator["graphs"].write(json.dumps(record) + "\n")
    accumulator["maps"].append(
        {
            "seed": seed,
            "case": case.name,
            "family": case.family,
            "pairs": len(table.pairs),
            "covered_pairs": result["covered_pairs"],
            "candidate_action_edges": result["candidate_count"],
            "initial_ctl_disagreements": initial_disagreements,
            "identity_action_simulation": direct["holds"],
            "identity_plain_simulation": plain["holds"],
            "unlabelled_exact_edges": result["exact_edges"],
            "unlabelled_candidate_edges": result["candidate_edges"],
            "unlabelled_spurious_edges": result["extra_edges"],
        }
    )
    return result, record


def run(output):
    output = fresh_directory(output)
    protocol = json.loads(PROTOCOL.read_text())
    bank = json.loads((ROOT / protocol["model_bank"]).read_text())
    checkpoints = {
        name: {
            int(s): ROOT / v["path"]
            for s, v in bank["roles"][role]["checkpoints"].items()
        }
        for name, role in (("ranking", "best"), ("regression", "baseline"))
    }
    hashes = {}
    for name, role in (("ranking", "best"), ("regression", "baseline")):
        for seed, path in checkpoints[name].items():
            sha = digest(path)
            if sha != bank["roles"][role]["checkpoints"][str(seed)]["sha256"]:
                raise AssertionError("Frozen model bank checkpoint mismatch.")
            hashes[f"{name}/{seed}"] = sha
    if hashes != protocol["checkpoint_sha256"]:
        raise AssertionError("Checkpoints differ from the pinned c15c457 model bank.")
    preserved = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (ROOT / "outputs").rglob("*")
        if p.is_file() and not p.is_relative_to(output)
    }
    write_json(output / "preservation_before.json", preserved)
    started = {
        "protocol": protocol,
        "protocol_sha256": digest(PROTOCOL),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__),
                ROOT / "src/jepa_lmc/evaluation/local_abstraction.py",
                ROOT / "src/jepa_lmc/evaluation/latent_radius.py",
                ROOT / "src/jepa_lmc/learning/model.py",
            )
        },
        "checkpoint_sha256": hashes,
        "checkpoint_paths": {
            n: {str(s): str(p.relative_to(ROOT)) for s, p in paths.items()}
            for n, paths in checkpoints.items()
        },
        "historical_report_sha256": digest(HISTORICAL / "report.json"),
        "historical_candidates_sha256": digest(HISTORICAL / "candidate_sets.jsonl"),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "threads": 4,
            "device": "cpu",
        },
    }
    write_json(output / "study_started.json", started)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    historical = json.loads((HISTORICAL / "report.json").read_text())
    old_sets = {
        (r["seed"], r["case"], tuple(r["state"]), r["action"]): {
            tuple(s) for s in r["candidates"]
        }
        for r in read_jsonl(HISTORICAL / "candidate_sets.jsonl")
        if r["variant"] == "epsilon_max"
    }
    old_top1 = {
        (r["seed"], r["case"]): r
        for r in read_jsonl(RANKING_RUN / "behavioral/best/relations.jsonl")
    }
    cases = radius_stress_cases()
    accumulators = {}
    for variant in protocol["variants"]:
        directory = fresh_directory(output / "behavioral" / variant)
        accumulators[variant] = {
            "directory": directory,
            "graphs": (directory / "relations.jsonl").open("w"),
            "maps": [],
            "ctl": [],
            "relations": [],
            "certificates": 0,
            "partitions": 0,
            "preservation": 0,
        }
    pairs, ranks, seed_results, radius_replay = [], [], [], []
    with (output / "distance_tables.jsonl").open("w") as raw:
        for seed in SEEDS:
            tables = {}
            radii = {}
            for predictor in ("ranking", "regression"):
                model, payload = load_model(checkpoints[predictor][seed])
                if payload["training"]["seed"] != seed:
                    raise AssertionError("Seed provenance changed.")
                tables[predictor] = [
                    collect_latent_distances(model, c.make_env()) for c in cases
                ]
                radii[predictor] = oracle_radii(tables[predictor])["epsilon_max"]
                for case, table in zip(cases, tables[predictor], strict=True):
                    metadata = {
                        "predictor": predictor,
                        "seed": seed,
                        "case": case.name,
                        "family": case.family,
                    }
                    raw.write(
                        json.dumps(
                            distance_record(model, case.make_env(), table, metadata)
                        )
                        + "\n"
                    )
                    ranks.extend(metadata | row for row in ranking_rows(table))
            old_seed = next(r for r in historical["seeds"] if r["seed"] == seed)
            saved_radius = old_seed["radii"]["epsilon_max"]
            # Floating inference need not be bitwise identical across CPU builds.
            # This scalar tolerance never relaxes the exact set/CTL replay below.
            radius_replay.append(
                {
                    "seed": seed,
                    "historical": saved_radius,
                    "recomputed": radii["regression"],
                    "absolute_difference": abs(radii["regression"] - saved_radius),
                }
            )
            if not math.isclose(radii["regression"], saved_radius, rel_tol=1e-5):
                raise AssertionError("Historical maximum radius materially changed.")
            for variant in protocol["variants"]:
                predictor = (
                    "regression" if variant.startswith("regression_") else "ranking"
                )
                maps = []
                for case, table in zip(cases, tables[predictor], strict=True):
                    local = oracle_local_mask(table)
                    global_mask = table.candidates(radii[predictor])
                    if bool((local & ~global_mask).any()):
                        raise AssertionError(
                            "Local ball is not contained in pooled maximum ball."
                        )
                    if variant.endswith("_local"):
                        mask = local
                    elif variant.endswith("_global"):
                        mask = global_mask
                    elif variant == "all_states":
                        mask = torch.ones_like(table.distances, dtype=torch.bool)
                    else:
                        mask = torch.zeros_like(table.distances, dtype=torch.bool)
                        indices = (
                            table.distances.argmin(1)
                            if variant == "ranking_top1"
                            else torch.tensor(table.true_indices)
                        )
                        mask[torch.arange(len(table.pairs)), indices] = True
                    if variant == "regression_global":
                        for i, (s, a) in enumerate(table.pairs):
                            actual = {
                                table.states[j]
                                for j in mask[i].nonzero().flatten().tolist()
                            }
                            if actual != old_sets[seed, case.name, s, a]:
                                raise AssertionError(
                                    "Historical global candidate membership changed."
                                )
                    metadata = {
                        "seed": seed,
                        "variant": variant,
                        "case": case.name,
                        "family": case.family,
                    }
                    pairs.extend(metadata | r for r in candidate_rows(table, mask))
                    result, exported = analyze_graph(
                        case, table, mask, seed, variant, accumulators[variant]
                    )
                    maps.append(result)
                    if variant == "ranking_top1":
                        previous = old_top1[seed, case.name]
                        if (
                            exported["graphs"] != previous["graphs"]
                            or exported["labels"] != previous["labels"]
                            or exported["initial"] != previous["initial"]
                        ):
                            raise AssertionError("Frozen ranking Top-1 graph changed.")
                summary = summarize_stress_maps(maps)
                if (
                    variant == "regression_global"
                    and summary != old_seed["summary"]["epsilon_max"]
                ):
                    raise AssertionError(
                        "Historical global CTL/candidate aggregates changed."
                    )
                selected = [
                    r for r in pairs if r["seed"] == seed and r["variant"] == variant
                ]
                seed_results.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "global_radius": radii[predictor],
                        "candidates": summarize_candidates(selected),
                        "ctl": summary,
                    }
                )
                print(
                    seed,
                    variant,
                    "size",
                    summarize_candidates(selected)["mean_size"],
                    "proofs",
                    summary["nontrivial"]["recovered_opportunities"],
                    "/",
                    summary["nontrivial"]["transfer_opportunities"],
                    flush=True,
                )
    write_csv(output / "rankings.csv", ranks)
    write_csv(output / "candidate_pairs.csv", pairs)
    for variant, acc in accumulators.items():
        acc["graphs"].close()
        directory = acc["directory"]
        write_csv(directory / "maps.csv", acc["relations"])
        write_csv(directory / "ctl_queries.csv", acc["ctl"])
        report = {
            "source_revision": started["source_revision"],
            "candidate_variant": variant,
            "graph_semantics": (
                "The legacy top1 key holds the explicitly identified candidate graph."
            ),
            "maps": acc["maps"],
            "audit": {
                "passed": True,
                "certificates": acc["certificates"],
                "partition_comparisons": acc["partitions"],
                "ctl_preservation_checks": acc["preservation"],
            },
            "seeds": [
                {
                    "seed": seed,
                    "initial_all_six_ctl": sum(
                        not m["initial_ctl_disagreements"]
                        for m in acc["maps"]
                        if m["seed"] == seed
                    ),
                    "relations": {
                        semantics: {
                            name: sum(
                                r["initial_related"]
                                for r in acc["relations"]
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
            "exports": {p.name: digest(p) for p in directory.iterdir() if p.is_file()},
        }
        write_json(directory / "report.json", report)
    if hashes != {
        f"{n}/{s}": digest(p) for n, ps in checkpoints.items() for s, p in ps.items()
    }:
        raise AssertionError("A frozen checkpoint changed.")
    write_json(
        output / "build_report.json",
        {
            "status": "completed",
            "protocol_sha256": digest(PROTOCOL),
            "seed_results": seed_results,
            "historical_max_candidate_sets_reproduced": len(old_sets),
            "frozen_ranking_top1_graphs_reproduced": len(old_top1),
            "historical_radius_numerical_replay": radius_replay,
            "all_distance_orderings_independently_verified": True,
            "exports": {
                p.name: digest(p)
                for p in (
                    output / "distance_tables.jsonl",
                    output / "rankings.csv",
                    output / "candidate_pairs.csv",
                )
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == "__main__":
    main()
