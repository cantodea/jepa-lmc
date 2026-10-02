"""Re-evaluate saved graphs; no inference or training, with optional saved LTL."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

from experiments.evaluate_top1_ltl import digest, load_system, read_run
from jepa_lmc.benchmarks.ctl_suite import default_ctl_suite
from jepa_lmc.benchmarks.ltl_suite import default_ltl_suite
from jepa_lmc.evaluation.structural import (
    RELATIONS,
    SEMANTICS,
    evaluate_model_pair,
    model_evaluation_report,
)
from jepa_lmc.verification.behavioral_relations import greatest_relation
from jepa_lmc.verification.ctl import CTLModelChecker

ROOT = Path(__file__).resolve().parents[1]


def point_prediction_counts(real, learned):
    """Count exact (s,a)->t matches, refusing set-valued or incompatible graphs."""
    if real.states != learned.states:
        raise ValueError("Top-1 counting needs the shared state catalogue.")
    tables = []
    for graph in (real, learned):
        table = {}
        for s in graph.states:
            for edge in graph.action_successors(s):
                table.setdefault((s, edge.action), set()).add(edge.target)
        if any(len(targets) != 1 for targets in table.values()):
            raise ValueError("Top-1 counting requires action-deterministic graphs.")
        tables.append(table)
    if tables[0].keys() != tables[1].keys():
        raise ValueError("Top-1 action domains differ.")
    return sum(t == tables[1][key] for key, t in tables[0].items()), len(tables[0])


def read_linked_ltl(directory, source_hash, records):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    if report["source_relations_sha256"] != source_hash:
        raise ValueError("Saved LTL was evaluated on a different graph export.")
    query_path = directory / "queries.csv"
    if digest(query_path) != report["exports"]["queries.csv"]:
        raise ValueError("Saved LTL query hash mismatch.")
    catalogue = {(r["seed"], r["case"]): r for r in records}
    names = {p.name for p in default_ltl_suite()}
    expected = {
        (seed, case, s, p)
        for (seed, case), r in catalogue.items()
        for s in range(len(r["states"]))
        for p in names
    }
    seen, grouped = set(), {key: [] for key in catalogue}
    with query_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["seed"]), row["case"], int(row["state_id"]), row["property"])
            if key not in expected or key in seen:
                raise ValueError("Saved LTL query coverage is duplicate or unexpected.")
            seen.add(key)
            record = catalogue[key[:2]]
            for field in ("real", "top1", "agreement", "initial"):
                if row[field] not in ("True", "False"):
                    raise ValueError("Saved LTL verdicts must be boolean.")
                row[field] = row[field] == "True"
            if (
                row["agreement"] != (row["real"] == row["top1"])
                or row["initial"] != (key[2] == record["initial"])
                or [int(row["row"]), int(row["column"])] != record["states"][key[2]]
            ):
                raise ValueError("Saved LTL verdict metadata is inconsistent.")
            grouped[key[:2]].append(row)
    if seen != expected:
        raise ValueError("Saved LTL is missing state/property queries.")
    result = {}
    for key, rows in grouped.items():
        initial = [r for r in rows if r["initial"]]
        result[key] = {
            "status": "reused_hash_verified_saved_backend_verdicts",
            "initial_all_six_agreement": all(r["agreement"] for r in initial),
            "initial_mismatches": [
                {"property": r["property"], "real": r["real"], "learned": r["top1"]}
                for r in initial
                if not r["agreement"]
            ],
            "comparisons": len(rows),
            "matched": sum(r["agreement"] for r in rows),
        }
    return result, {
        "report.json": digest(directory / "report.json"),
        "queries.csv": digest(query_path),
    }


def evaluate_saved_run(run_dir, *, ltl_dir=None, predictor_run_dir=None):
    """Return a fresh primary scorecard; validate all saved input identities."""
    run_dir = Path(run_dir)
    source, records = read_run(run_dir)
    inputs = {
        name: digest(run_dir / name) for name in ("report.json", "relations.jsonl")
    }
    metadata = {(r["seed"], r["case"]): r for r in source["maps"]}
    predictor_records, predictor_hashes = None, None
    if predictor_run_dir is not None:
        predictor_run_dir = Path(predictor_run_dir)
        _, predictor = read_run(predictor_run_dir)
        predictor_records = {(r["seed"], r["case"]): r for r in predictor}
        if predictor_records.keys() != metadata.keys():
            raise ValueError("Predictor and candidate cases differ.")
        predictor_hashes = {n: digest(predictor_run_dir / n) for n in inputs}
    ltl, ltl_hashes = ({}, None)
    if ltl_dir is not None:
        ltl, ltl_hashes = read_linked_ltl(ltl_dir, inputs["relations.jsonl"], records)
    cases, parity = [], 0
    for record in records:
        key = record["seed"], record["case"]
        real, learned = (load_system(record, n) for n in ("real", "top1"))
        top1 = None
        if predictor_records is not None:
            prediction = predictor_records[key]
            if (
                any(prediction[k] != record[k] for k in ("states", "labels", "initial"))
                or prediction["graphs"]["real"] != record["graphs"]["real"]
            ):
                raise ValueError("Predictor and candidate concrete systems differ.")
            top1 = point_prediction_counts(real, load_system(prediction, "top1"))
        elif source.get("candidate_variant") in (None, "ranking_top1"):
            top1 = point_prediction_counts(real, learned)
        elif "evaluation" in metadata[key]:
            # New candidate runs also record the underlying predictor's counts.
            # Never interpret nondeterministic successor coverage as Top-1.
            saved_top1 = metadata[key]["evaluation"]["diagnostics"]["top1"]
            if saved_top1 is not None:
                top1 = saved_top1["correct"], saved_top1["total"]
        relations = {}
        for actions in SEMANTICS.values():
            for name in RELATIONS:
                left, right = (
                    (learned, real) if name == "top1_to_real" else (real, learned)
                )
                relations[actions, name] = greatest_relation(
                    left,
                    right,
                    kind="bisimulation" if name == "bisimulation" else "simulation",
                    action_sensitive=actions,
                )
        saved = record.get("relations", [])
        if saved:
            keys = [(r["action_sensitive"], r["name"]) for r in saved]
            if len(keys) != len(set(keys)) or set(keys) != set(relations):
                raise ValueError("Saved relations have incomplete/duplicate semantics.")
            for previous in saved:
                result = relations[previous["action_sensitive"], previous["name"]]
                if (
                    result.pairs != frozenset(tuple(p) for p in previous["pairs"])
                    or result.rounds != previous["rounds"]
                ):
                    raise AssertionError(
                        "Recomputed greatest relation differs from saved pairs."
                    )
                parity += 1
        case = {
            "seed": key[0],
            "case": key[1],
            "family": metadata[key]["family"],
            **evaluate_model_pair(real, learned, top1_counts=top1, relations=relations),
        }
        if "candidate_variant" in source:
            case["variant"] = source["candidate_variant"]
        previous = metadata[key]
        if "initial_ctl_disagreements" in previous:
            old = {
                (r["property"], r["real"], r["top1"])
                for r in previous["initial_ctl_disagreements"]
            }
            new = {
                (r["property"], r["real"], r["learned"])
                for r in case["diagnostics"]["initial_ctl_mismatches"]
            }
            if old != new:
                raise AssertionError(
                    "Recomputed initial CTL differs from saved diagnostics."
                )
        if "correct_top1_pairs" in previous and top1 != (
            previous["correct_top1_pairs"],
            previous["pairs"],
        ):
            raise AssertionError("Saved Top-1 counts differ from the frozen graph.")
        case["diagnostics"]["ltl"] = ltl.get(key)
        cases.append(case)
    # Check every saved native CTL verdict, including unreachable states.
    ctl_checks = 0
    ctl_path = run_dir / "ctl_queries.csv"
    if ctl_path.exists():
        if digest(ctl_path) != source["exports"]["ctl_queries.csv"]:
            raise ValueError("Saved CTL query hash mismatch.")
        expected = {}
        for record in records:
            checkers = [
                CTLModelChecker(load_system(record, n)) for n in ("real", "top1")
            ]
            for s, coordinates in enumerate(record["states"]):
                for p in default_ctl_suite():
                    k = (record["seed"], record["case"], *coordinates, p.name)
                    expected[k] = tuple(c.holds(s, p.formula) for c in checkers)
        seen = set()
        with ctl_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (
                    int(row["seed"]),
                    row["case"],
                    int(row["row"]),
                    int(row["column"]),
                    row["property"],
                )
                verdicts = (row["real"], row["top1"])
                if (
                    key in seen
                    or key not in expected
                    or verdicts != tuple(str(v) for v in expected[key])
                ):
                    raise AssertionError("Saved native CTL coverage/verdict changed.")
                seen.add(key)
        if seen != expected.keys():
            raise AssertionError("Saved native CTL is incomplete.")
        ctl_checks = len(seen) * 2
        inputs["ctl_queries.csv"] = digest(ctl_path)
    if inputs != {name: digest(run_dir / name) for name in inputs}:
        raise AssertionError("Input artifacts changed during evaluation.")
    return model_evaluation_report(
        cases,
        source_run=str(run_dir),
        source_sha256=inputs,
        predictor_source_sha256=predictor_hashes,
        ltl_source_sha256=ltl_hashes,
        source_revision=source["source_revision"],
        python=platform.python_version(),
        evaluator_sha256={
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__).resolve(),
                ROOT / "src/jepa_lmc/evaluation/structural.py",
                ROOT / "src/jepa_lmc/verification/behavioral_relations.py",
            )
        },
        saved_greatest_relations_reproduced=parity,
        saved_native_ctl_verdicts_reproduced=ctl_checks,
        relation_certificates_checked=6 * len(cases),
        independent_partition_checks=2 * len(cases),
        training=False,
        inference=False,
        top1_role="underlying point predictor; not candidate coverage",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ltl-dir", type=Path)
    parser.add_argument("--predictor-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(
            "Choose a fresh output directory; previous reports are preserved."
        )
    report = evaluate_saved_run(
        args.run_dir, ltl_dir=args.ltl_dir, predictor_run_dir=args.predictor_run_dir
    )
    if (
        args.expected_cases is not None
        and report["summary"]["cases"] != args.expected_cases
    ):
        raise ValueError("Case count differs from the requested evaluation domain.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "mismatches.json").write_text(
        json.dumps(report["summary"]["mismatch_cases"], indent=2) + "\n"
    )
    rows = []
    for case in report["cases"]:
        for name, metrics in case["behavioral"].items():
            rows.append(
                {
                    "seed": case["seed"],
                    "case": case["case"],
                    "family": case["family"],
                    "semantics": name,
                    **{k: v for k, v in metrics.items() if k != "relations"},
                    "all_six_ctl": case["diagnostics"]["initial_all_six_ctl_agreement"],
                    "all_six_ltl": case["diagnostics"]["ltl"][
                        "initial_all_six_agreement"
                    ]
                    if case["diagnostics"]["ltl"]
                    else None,
                }
            )
    with (args.output_dir / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {k: v for k, v in report["summary"].items() if k != "mismatch_cases"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
