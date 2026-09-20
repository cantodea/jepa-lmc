"""Replay frozen local-ball records and apply the predeclared feasibility gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.oracle_local_abstraction import (
    BACKEND_VARIANTS,
    DEFAULT_RUN,
    PROTOCOL,
    RANKING_RUN,
)
from experiments.summarize_top1_quality import (
    behavior_summary,
    read_csv,
    temporal_summary,
)
from experiments.top1_quality_diagnosis import ROOT, SEEDS, digest, write_json
from jepa_lmc.benchmarks.radius_stress import label_immediate, radius_stress_cases
from jepa_lmc.evaluation.local_abstraction import summarize_candidates
from jepa_lmc.evaluation.radius_stress import score_nontrivial
from jepa_lmc.verification.gridworld import gridworld_to_transition_system

UNIVERSAL_CTL = {"AG !danger", "AF goal"}
ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}


def jsonl(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def grouped(rows, fields, summarize):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[f] for f in fields)].append(row)
    return [
        dict(zip(fields, key, strict=True)) | summarize(value)
        for key, value in sorted(buckets.items())
    ]


def rank_summary(rows):
    ranks = [int(r["true_rank"]) for r in rows]
    return {
        "pairs": len(rows),
        "true_rank_histogram": dict(sorted(Counter(ranks).items())),
        "top1_coverage": sum(r <= 1 for r in ranks) / len(rows),
        "top2_coverage": sum(r <= 2 for r in ranks) / len(rows),
        "top3_coverage": sum(r <= 3 for r in ranks) / len(rows),
        "boundary_tie_queries": sum(int(r["true_distance_ties"]) > 1 for r in rows),
    }


def pair_key(row):
    return (
        int(row["seed"]),
        row["case"],
        int(row["state_row"]),
        int(row["state_col"]),
        int(row["action"]),
    )


def query_key(row):
    return (
        int(row["seed"]),
        row["case"],
        int(row["row"]),
        int(row["column"]),
        row["property"],
    )


def replay_geometry(run, build, protocol, cases):
    """Check raw embeddings, full order, closed membership and exported graph edges."""
    for name, sha in build["exports"].items():
        if digest(run / name) != sha:
            raise AssertionError(f"Build export hash changed: {name}")
    ranks = read_csv(run / "rankings.csv")
    pairs = read_csv(run / "candidate_pairs.csv")
    for rows in (ranks, pairs):
        for r in rows:
            for k in ("seed", "state_row", "state_col", "action"):
                r[k] = int(r[k])
    for r in pairs:
        for k in ("size", "spurious_action_edges"):
            r[k] = int(r[k])
    rank_index = {(r["predictor"], *pair_key(r)): r for r in ranks}
    pair_index = {(r["variant"], *pair_key(r)): r for r in pairs}
    if len(rank_index) != len(ranks) or len(pair_index) != len(pairs):
        raise AssertionError("Duplicate query records.")
    tables = {}
    expected_queries = {
        (seed, name, *s, a)
        for seed in SEEDS
        for name, case in cases.items()
        for s in case.make_env().all_states()
        for a in range(4)
    }
    if set(rank_index) != {
        (p, *key) for p in ("ranking", "regression") for key in expected_queries
    } or set(pair_index) != {
        (v, *key) for v in protocol["variants"] for key in expected_queries
    }:
        raise AssertionError("Incomplete state-action catalogue.")
    maximum_error = 0.0
    for record in jsonl(run / "distance_tables.jsonl"):
        name, seed, case_name = record["predictor"], record["seed"], record["case"]
        env = cases[case_name].make_env()
        states = [tuple(s) for s in record["states"]]
        if states != list(env.all_states()):
            raise AssertionError("Distance table state ordering changed.")
        if record["pairs"] != [[i, a] for i in range(len(states)) for a in range(4)]:
            raise AssertionError("Distance table query order changed.")
        d = np.asarray(record["distances"], dtype=np.float64)
        pred = np.asarray(record["predicted_embeddings"], dtype=np.float64)
        target = np.asarray(record["target_embeddings"], dtype=np.float64)
        independent = np.sqrt(np.sum((pred[:, None] - target[None]) ** 2, axis=-1))
        error = float(np.max(np.abs(d - independent)))
        maximum_error = max(maximum_error, error)
        order = np.argsort(d, axis=1, kind="stable")
        if error > 1e-12 or not np.array_equal(order, record["ordered_indices"]):
            raise AssertionError("Saved full distances/order fail numerical replay.")
        if not np.array_equal(order, np.argsort(independent, axis=1, kind="stable")):
            raise AssertionError("Independent arithmetic changes complete rankings.")
        indices = {s: i for i, s in enumerate(states)}
        for i, (source, action) in enumerate(record["pairs"]):
            true = indices[env.transition(states[source], action)]
            if true != record["true_indices"][i]:
                raise AssertionError("Saved oracle successor is incorrect.")
            row = rank_index[name, seed, case_name, *states[source], action]
            rank = int(np.where(order[i] == true)[0][0]) + 1
            if rank != int(row["true_rank"]) or true != int(row["true_index"]):
                raise AssertionError("Saved true rank/index is incorrect.")
            expected = {
                "d1": d[i, order[i, 0]],
                "d2": d[i, order[i, 1]],
                "d3": d[i, order[i, 2]],
                "d_true": d[i, true],
                "margin": d[i, order[i, 1]] - d[i, order[i, 0]],
                "true_distance_ties": int(np.sum(d[i] == d[i, true])),
            }
            if any(float(row[k]) != value for k, value in expected.items()):
                raise AssertionError("Saved distance scalar is incorrect.")
            if any(row[f"top{k}_covered"] != (rank <= k) for k in (1, 2, 3)):
                raise AssertionError("Saved Top-k coverage is incorrect.")
        key = name, seed, case_name
        if key in tables:
            raise AssertionError("Duplicate distance table.")
        tables[key] = record
    if len(tables) != 2 * len(SEEDS) * len(cases):
        raise AssertionError("Missing distance tables.")
    radii = {
        (p, s): max(
            r["distances"][i][t]
            for (name, seed, _), r in tables.items()
            if (name, seed) == (p, s)
            for i, t in enumerate(r["true_indices"])
        )
        for p in ("ranking", "regression")
        for s in SEEDS
    }
    replayed = 0
    for variant in protocol["variants"]:
        seen = set()
        predictor = "regression" if variant.startswith("regression_") else "ranking"
        for graph in jsonl(run / "behavioral" / variant / "relations.jsonl"):
            seed, case_name = graph["seed"], graph["case"]
            if (seed, case_name) in seen or graph["candidate_variant"] != variant:
                raise AssertionError("Graph catalogue/variant mismatch.")
            seen.add((seed, case_name))
            record = tables[predictor, seed, case_name]
            env = cases[case_name].make_env()
            exact = gridworld_to_transition_system(env)
            states = [tuple(s) for s in record["states"]]
            if graph["states"] != record["states"] or graph["initial"] != states.index(
                env.start
            ):
                raise AssertionError("Graph state/initial encoding changed.")
            if graph["labels"] != [sorted(exact.propositions(s)) for s in states]:
                raise AssertionError("Graph labels changed.")
            d = np.asarray(record["distances"])
            n = len(states)
            expected_real = [set() for _ in states]
            expected_candidate = [set() for _ in states]
            for i, (source, action) in enumerate(record["pairs"]):
                true = record["true_indices"][i]
                local = set(np.flatnonzero(d[i] <= d[i, true]).tolist())
                global_set = set(
                    np.flatnonzero(d[i] <= radii[predictor, seed]).tolist()
                )
                if not local <= global_set or true not in local:
                    raise AssertionError("Closed local/global nesting failed.")
                if variant.endswith("_local"):
                    candidates = local
                elif variant.endswith("_global"):
                    candidates = global_set
                elif variant == "all_states":
                    candidates = set(range(n))
                elif variant == "exact_graph":
                    candidates = {true}
                else:
                    candidates = {record["ordered_indices"][i][0]}
                row = pair_index[variant, seed, case_name, *states[source], action]
                if (row["size"], row["covered"], row["spurious_action_edges"]) != (
                    len(candidates),
                    true in candidates,
                    len(candidates - {true}),
                ):
                    raise AssertionError(
                        "Candidate metrics differ from full membership."
                    )
                expected_real[source].add((action, true))
                expected_candidate[source].update((action, u) for u in candidates)
                replayed += 1
            for name, expected in (
                ("real", expected_real),
                ("top1", expected_candidate),
            ):
                for edges, wanted in zip(graph["graphs"][name], expected, strict=True):
                    if {tuple(e) for e in edges} != wanted or len(edges) != len(wanted):
                        raise AssertionError(
                            "Exported backend graph lost/added an edge."
                        )
        if seen != {(s, c) for s in SEEDS for c in cases}:
            raise AssertionError("Missing backend graphs.")
    return (
        ranks,
        pairs,
        {
            "distance_tables": len(tables),
            "ranking_queries": len(ranks),
            "candidate_memberships_replayed": replayed,
            "numpy_distance_max_absolute_error": maximum_error,
            "complete_orderings_and_scalar_fields_verified": True,
            "all_exported_action_edges_and_labels_verified": True,
        },
    )


def universal_proofs(rows, logic, cases):
    if logic == "ctl":
        rows = [r for r in rows if r["property"] in UNIVERSAL_CTL]

    def immediate(r):
        env = cases[r["case"]].make_env()
        s = int(r["row"]), int(r["column"])
        if logic == "ctl":
            return label_immediate(env, s, r["property"])
        if r["property"] == "F goal":
            return env.is_goal(s)
        if r["property"] == "G !danger":
            return env.is_danger(s)
        if r["property"] == "safe U goal":
            return env.is_goal(s) or env.is_danger(s)
        return False

    def stats(subset):
        opportunities = sum(r["real"] for r in subset)
        proven = sum(r["real"] and r["top1"] for r in subset)
        return {
            "queries": len(subset),
            "true_opportunities": opportunities,
            "proven": proven,
            "false_claims": sum(not r["real"] and r["top1"] for r in subset),
            "true_proof_recall": proven / opportunities if opportunities else None,
        }

    return {
        scope: {
            "pooled": stats(subset),
            "by_property": {
                prop: stats([r for r in subset if r["property"] == prop])
                for prop in sorted({r["property"] for r in rows})
            },
        }
        for scope, subset in (
            ("all_states", rows),
            ("initial", [r for r in rows if r["initial"]]),
            ("non_immediate", [r for r in rows if not immediate(r)]),
        )
    }


def primary_score(rows, cases, inclusive):
    return score_nontrivial(
        [
            r
            | {
                "ground_truth": r["real"],
                "learned": r["top1"],
                "false_safe": r["property"] == "AG !danger"
                and r["top1"]
                and not r["real"],
                "state": (int(r["row"]), int(r["column"])),
                "label_immediate": label_immediate(
                    cases[r["case"]].make_env(),
                    (int(r["row"]), int(r["column"])),
                    r["property"],
                ),
                "relation_inclusion": inclusive[int(r["seed"]), r["case"]],
                "monotonicity": -1 if r["property"] in UNIVERSAL_CTL else 1,
                "one_sided_claim": r["top1"]
                if r["property"] in UNIVERSAL_CTL
                else not r["top1"],
            }
            for r in rows
        ]
    )


def apply_gate(protocol, variants):
    local, old = variants["ranking_local"], variants["regression_global"]
    limits = protocol["continue_gate"]["per_seed"]
    per_seed = []
    for seed in SEEDS:
        a = next(r for r in local["by_seed"] if r["seed"] == seed)
        b = next(r for r in old["by_seed"] if r["seed"] == seed)
        c, p = a["candidates"], a["primary_ctl"]
        ag = a["universal_ctl"]["non_immediate"]["by_property"]["AG !danger"]
        families = a["required_family_maps_with_proof"]
        checks = {
            "full_coverage": c["successor_coverage"] == limits["successor_coverage"],
            "identity_simulation_all_maps": a["identity_action_simulation_maps"] == 24,
            "zero_one_sided_violations": a["ctl_one_sided_violations"]
            == a["universal_ltl"]["all_states"]["pooled"]["false_claims"]
            == 0,
            "mean_size": c["mean_size"] <= limits["maximum_mean_candidate_size"],
            "maximum_size": c["max_size"] <= limits["maximum_candidate_size"],
            "singleton_fraction": c["singleton_fraction"]
            >= limits["minimum_singleton_fraction"],
            "safety_proof_recall": ag["true_proof_recall"] is not None
            and ag["true_proof_recall"]
            >= limits["minimum_non_immediate_AG_true_proof_recall"],
            "primary_proof_recall": p["opportunity_recall"] is not None
            and p["opportunity_recall"] >= limits["minimum_primary_proof_recall"],
            "required_family_consistency": all(
                n >= limits["minimum_maps_with_proof_in_each_required_family"]
                for n in families.values()
            ),
            "no_seed_regression": p["opportunity_recall"]
            >= b["primary_ctl"]["opportunity_recall"],
        }
        per_seed.append(
            {
                "seed": seed,
                "checks": checks,
                "passes": all(checks.values()),
                "failed_checks": [k for k, v in checks.items() if not v],
            }
        )
    reduction = (
        1
        - local["candidates"]["candidate_action_edges"]
        / old["candidates"]["candidate_action_edges"]
    )
    gain = (
        local["primary_ctl"]["opportunity_recall"]
        - old["primary_ctl"]["opportunity_recall"]
    )
    limits = protocol["continue_gate"]["pooled"]
    checks = {
        "candidate_count_reduction": reduction
        >= limits["minimum_candidate_count_reduction_vs_historical_global"],
        "primary_proof_recall_gain": gain
        >= limits["minimum_primary_proof_recall_gain_vs_historical_global"],
    }
    passed = all(r["passes"] for r in per_seed) and all(checks.values())
    return {
        "status": "continue_to_consider_bound_estimation"
        if passed
        else "stop_local_ball_route",
        "passes": passed,
        "by_seed": per_seed,
        "pooled_checks": checks,
        "candidate_count_reduction_vs_historical_global": reduction,
        "primary_proof_recall_gain_vs_historical_global": gain,
        "scope": protocol["scope"],
    }


def summarize(run):
    if (run / "summary.json").exists():
        raise ValueError("Summary already exists; refusing to overwrite.")
    build = json.loads((run / "build_report.json").read_text())
    started = json.loads((run / "study_started.json").read_text())
    protocol = started["protocol"]
    if digest(PROTOCOL) != started["protocol_sha256"] or build[
        "protocol_sha256"
    ] != digest(PROTOCOL):
        raise AssertionError("Predeclared protocol changed.")
    cases = {c.name: c for c in radius_stress_cases()}
    ranks, pairs, geometry_audit = replay_geometry(run, build, protocol, cases)
    variants, ctl_tables, ltl_tables = {}, {}, {}
    for variant in protocol["variants"]:
        directory = run / "behavioral" / variant
        report = json.loads((directory / "report.json").read_text())
        for name, sha in report["exports"].items():
            if digest(directory / name) != sha:
                raise AssertionError("Behavioral record hash changed.")
        ctl = read_csv(directory / "ctl_queries.csv")
        relations = read_csv(directory / "maps.csv")
        inclusive = {
            (r["seed"], r["case"]): r["identity_plain_simulation"]
            for r in report["maps"]
        }
        ltl = None
        if variant in BACKEND_VARIANTS:
            for suffix in ("ltl", "sanity"):
                backend_dir = run / "temporal" / f"{variant}_{suffix}"
                backend_report = json.loads((backend_dir / "report.json").read_text())
                for name, sha in backend_report["exports"].items():
                    if digest(backend_dir / name) != sha:
                        raise AssertionError("nuXmv input/output record hash changed.")
            formal, _, _, linked_ctl, ltl = behavior_summary(run, variant)
            if linked_ctl != ctl:
                raise AssertionError("Backend linking read a different CTL export.")
        elif variant == "ranking_top1":
            formal, _, _, old_ctl, ltl = behavior_summary(RANKING_RUN, "best")
            if {query_key(r): (r["real"], r["top1"]) for r in old_ctl} != {
                query_key(r): (r["real"], r["top1"]) for r in ctl
            }:
                raise AssertionError("Frozen Top-1 native verdicts changed.")
            formal["backend_reused_from"] = str(RANKING_RUN.relative_to(ROOT))
        else:
            formal = {
                "ctl": temporal_summary(ctl),
                "relations": {
                    semantics: {
                        name: sum(
                            r["initial_related"]
                            for r in relations
                            if r["semantics"] == semantics and r["relation"] == name
                        )
                        for name in ("real_to_top1", "top1_to_real", "bisimulation")
                    }
                    for semantics in ("ctl", "action_labelled")
                },
                "audits": report["audit"],
                "ltl": None,
                "backend_audit": None,
            }
        selected = [r for r in pairs if r["variant"] == variant]
        value = {
            "candidates": summarize_candidates(selected),
            "formal": formal,
            "identity_action_simulation_maps": sum(
                r["identity_action_simulation"] for r in report["maps"]
            ),
            "unlabelled_edges": {
                k: sum(r[k] for r in report["maps"])
                for k in (
                    "unlabelled_exact_edges",
                    "unlabelled_candidate_edges",
                    "unlabelled_spurious_edges",
                )
            },
            "primary_ctl": primary_score(ctl, cases, inclusive),
            "universal_ctl": universal_proofs(ctl, "ctl", cases),
            "universal_ltl": universal_proofs(ltl, "ltl", cases)
            if ltl is not None
            else None,
            "by_seed": [],
        }
        for seed in SEEDS:
            seed_ctl = [r for r in ctl if int(r["seed"]) == seed]
            original = next(
                r
                for r in build["seed_results"]
                if r["seed"] == seed and r["variant"] == variant
            )
            primary = primary_score(seed_ctl, cases, inclusive)
            if primary != original["ctl"]["nontrivial"]:
                raise AssertionError("Primary proof score differs from replayed CTL.")
            if (
                summarize_candidates([r for r in selected if r["seed"] == seed])
                != original["candidates"]
            ):
                raise AssertionError(
                    "Candidate aggregate differs from full membership replay."
                )
            value["by_seed"].append(
                {
                    "seed": seed,
                    "global_radius": original["global_radius"],
                    "candidates": summarize_candidates(
                        [r for r in selected if r["seed"] == seed]
                    ),
                    "primary_ctl": primary,
                    "universal_ctl": universal_proofs(seed_ctl, "ctl", cases),
                    "universal_ltl": universal_proofs(
                        [r for r in ltl if int(r["seed"]) == seed], "ltl", cases
                    )
                    if ltl is not None
                    else None,
                    "ctl_one_sided_violations": original["ctl"]["ctl"][
                        "one_sided_violations"
                    ],
                    "identity_action_simulation_maps": sum(
                        r["identity_action_simulation"]
                        for r in report["maps"]
                        if r["seed"] == seed
                    ),
                    "required_family_maps_with_proof": {
                        f: len(
                            original["ctl"]["nontrivial_by_family"][f][
                                "maps_with_recovered_opportunity"
                            ]
                        )
                        for f in ("sealed_region", "danger_gate")
                    },
                    "nontrivial_by_family": original["ctl"]["nontrivial_by_family"],
                }
            )
        if (
            ltl is not None
            and variant != "ranking_top1"
            and value["universal_ltl"]["all_states"]["pooled"]["false_claims"]
        ):
            raise AssertionError(
                "Oracle overapproximation makes an unsound LTL true claim."
            )
        variants[variant] = value
        ctl_tables[variant] = {query_key(r): r for r in ctl}
        if ltl is not None:
            ltl_tables[variant] = {query_key(r): r for r in ltl}
    monotonic_checks = 0
    backend_hashes = {
        variants[v]["formal"]["backend_audit"]["nuxmv_sha256"] for v in BACKEND_VARIANTS
    }
    if len(backend_hashes) != 1:
        raise AssertionError("The three abstraction controls used different backends.")
    for tables, logic in ((ctl_tables, "ctl"), (ltl_tables, "ltl")):
        for key, local in tables["ranking_local"].items():
            global_row = tables["ranking_global"][key]
            if local["real"] != global_row["real"]:
                raise AssertionError("Exact model temporal verdict changed.")
            universal = logic == "ltl" or local["property"] in UNIVERSAL_CTL
            bad = (
                global_row["top1"] and not local["top1"]
                if universal
                else local["top1"] and not global_row["top1"]
            )
            if bad:
                raise AssertionError("Nested graphs violate temporal monotonicity.")
            monotonic_checks += 1
    before = json.loads((run / "preservation_before.json").read_text())
    changed = [
        name
        for name, sha in before.items()
        if not (ROOT / name).is_file() or digest(ROOT / name) != sha
    ]
    if changed:
        raise AssertionError(f"Historical artifacts changed: {changed}")
    for predictor, paths in started["checkpoint_paths"].items():
        for seed, path in paths.items():
            if (
                digest(ROOT / path)
                != protocol["checkpoint_sha256"][f"{predictor}/{seed}"]
            ):
                raise AssertionError("Frozen checkpoint changed.")
    for name, sha in started["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise AssertionError("Frozen construction source changed during the run.")
    result = {
        "study": protocol["study"],
        "status": "completed",
        "protocol": protocol,
        "source_revision": started["source_revision"],
        "protocol_sha256": started["protocol_sha256"],
        "runtime": started["runtime"],
        "construction_source_sha256": started["source_sha256"],
        "summary_source_sha256": digest(Path(__file__)),
        "checkpoint_sha256": started["checkpoint_sha256"],
        "historical_radius_numerical_replay": build[
            "historical_radius_numerical_replay"
        ],
        "variants": variants,
        "gate": apply_gate(protocol, variants),
        "ranking_groups": {
            "pooled": grouped(ranks, ("predictor",), rank_summary),
            **{
                name: grouped(ranks, ("predictor", *fields), rank_summary)
                for name, fields in (
                    ("by_seed", ("seed",)),
                    ("by_family", ("family",)),
                    ("by_action", ("action",)),
                    ("by_seed_family_action", ("seed", "family", "action")),
                )
            },
        },
        "candidate_groups": {
            name: grouped(pairs, ("variant", *fields), summarize_candidates)
            for name, fields in (
                ("by_seed", ("seed",)),
                ("by_family", ("family",)),
                ("by_action", ("action",)),
                ("by_seed_family_action", ("seed", "family", "action")),
            )
        },
        "action_names": ACTION_NAMES,
        "audit": geometry_audit
        | {
            "historical_files_preserved": len(before),
            "checkpoint_hashes_preserved": 6,
            "historical_max_candidate_sets_reproduced": build[
                "historical_max_candidate_sets_reproduced"
            ],
            "frozen_ranking_top1_graphs_reproduced": build[
                "frozen_ranking_top1_graphs_reproduced"
            ],
            "nested_graph_temporal_checks": monotonic_checks,
            "shared_nuxmv_sha256": next(iter(backend_hashes)),
            "relation_certificates": sum(
                v["formal"]["audits"]["certificates"] for v in variants.values()
            ),
            "independent_partition_checks": sum(
                v["formal"]["audits"]["partition_comparisons"]
                for v in variants.values()
            ),
        },
        "exports": {
            p.relative_to(run).as_posix(): digest(p)
            for p in sorted(run.rglob("*"))
            if p.is_file()
        },
    }
    write_json(run / "summary.json", result)
    print(json.dumps({"gate": result["gate"], "audit": result["audit"]}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    summarize(args.run_dir.resolve())


if __name__ == "__main__":
    main()
