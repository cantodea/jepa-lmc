"""Run nuXmv LTL checks on the exact graphs from a saved Top-1 relation run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

from jepa_lmc.benchmarks.ltl_suite import default_ltl_suite
from jepa_lmc.evaluation.metrics import PropertyOutcome, VerificationReport
from jepa_lmc.verification.nuxmv import (
    LTLQuery,
    export_nusmv_ltl_queries,
    find_nusmv_executable,
    run_nusmv_model,
)
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_run(run_dir: Path) -> tuple[dict, list[dict]]:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    graph_path = run_dir / "relations.jsonl"
    if digest(graph_path) != report["exports"]["relations.jsonl"]:
        raise ValueError("relations.jsonl does not match the source report hash.")
    with graph_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    expected = {(row["seed"], row["case"]) for row in report["maps"]}
    actual = [(row["seed"], row["case"]) for row in records]
    if (
        not expected
        or len(expected) != len(report["maps"])
        or len(set(actual)) != len(actual)
        or set(actual) != expected
    ):
        raise ValueError("Exported seed/map identities differ from the source report.")
    # Validate every graph before starting an external process or writing outputs.
    for record in records:
        for name in ("real", "top1"):
            load_system(record, name)
    return report, records


def load_system(record: dict, name: str) -> ExplicitTransitionSystem:
    states = record["states"]
    edges = record["graphs"][name]
    if (
        len({tuple(s) for s in states}) != len(states)
        or len(record["labels"]) != len(states)
        or len(edges) != len(states)
    ):
        raise ValueError("State catalogue, labels and graph dimensions disagree.")
    return ExplicitTransitionSystem(
        states=range(len(states)),
        initial_states=(record["initial"],),
        labels=dict(enumerate(record["labels"])),
        transitions=dict(enumerate(edges)),
    )


def summarize(rows: list[dict]) -> dict:
    report = VerificationReport(
        tuple(
            PropertyOutcome(
                name=row["property"],
                category=row["category"],
                ground_truth=row["real"],
                learned=row["top1"],
                safety_claim=row["safety_claim"],
                primary_score=row["primary_score"],
            )
            for row in rows
        )
    )
    return {
        "comparisons": report.total,
        "matched": report.matched,
        "agreement": report.agreement,
        "false_safe_count": report.false_safe_count,
        "unsafe_miss_rate": report.unsafe_miss_rate,
        "primary_balanced_score": report.primary_balanced_score,
        "by_property": {
            name: {
                **asdict(confusion),
                "agreement": report.agreement_by_property[name],
                "balanced_accuracy": confusion.balanced_accuracy,
                "real_positive_rate": (
                    report.ground_truth_positive_rate_by_property[name]
                ),
            }
            for name, confusion in report.confusion_by_property.items()
        },
    }


def run_experiment(
    run_dir: Path, output_dir: Path, executable: str | Path | None = None
) -> dict:
    resolved = find_nusmv_executable(executable)
    if resolved is None:
        raise FileNotFoundError(
            "nuXmv/NuSMV was not found. Set NUXMV_BINARY or pass --executable PATH."
        )
    resolved = str(Path(resolved).resolve())
    source, records = read_run(run_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose a fresh output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    backend_dir = output_dir / "backend"
    backend_dir.mkdir()
    suite = default_ltl_suite()
    properties = {prop.name: prop for prop in suite}
    metadata = {(r["seed"], r["case"]): r for r in source["maps"]}
    rows, maps = [], []
    for index, record in enumerate(records):
        seed, case = record["seed"], record["case"]
        queries = tuple(
            LTLQuery(state, prop.formula, prop.name)
            for state in range(len(record["states"]))
            for prop in suite
        )
        verdicts = {}
        files = {}
        for name in ("real", "top1"):
            text = export_nusmv_ltl_queries(load_system(record, name), queries)
            stem = f"{index:03d}_{name}"
            smv_path = backend_dir / f"{stem}.smv"
            log_path = backend_dir / f"{stem}.txt"
            smv_path.write_text(text, encoding="utf-8")
            result = run_nusmv_model(text, resolved)
            log_path.write_text(result.output, encoding="utf-8")
            if len(result.verdicts) != len(queries):
                raise RuntimeError(
                    f"{seed}/{case}/{name}: expected {len(queries)} LTL verdicts, "
                    f"got {len(result.verdicts)}."
                )
            verdicts[name] = result.verdicts
            files[name] = smv_path.relative_to(output_dir).as_posix()
        map_rows = []
        for q, real, top1 in zip(
            queries, verdicts["real"], verdicts["top1"], strict=True
        ):
            prop = properties[q.name]
            row = {
                "seed": seed,
                "case": case,
                "family": metadata[seed, case]["family"],
                "state_id": q.state,
                "row": record["states"][q.state][0],
                "column": record["states"][q.state][1],
                "initial": q.state == record["initial"],
                "property": q.name,
                "category": prop.category,
                "safety_claim": prop.safety_claim,
                "primary_score": prop.primary_score,
                "real": real,
                "top1": top1,
                "agreement": real == top1,
            }
            map_rows.append(row)
        initial = [r for r in map_rows if r["initial"]]
        maps.append(
            {
                "seed": seed,
                "case": case,
                "family": metadata[seed, case]["family"],
                "backend_models": files,
                "initial": summarize(initial),
                "all_states": summarize(map_rows),
            }
        )
        rows.extend(map_rows)
        print(
            f"[{index + 1}/{len(records)}] {seed} {case}: initial LTL agreement "
            f"{sum(r['agreement'] for r in initial)}/{len(initial)}",
            flush=True,
        )
    queries_path = output_dir / "queries.csv"
    with queries_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "completed",
        "semantics": "Universal-path LTL; actions ignored; no fairness constraints.",
        "source_run_dir": str(run_dir.resolve()),
        "source_report_sha256": digest(run_dir / "report.json"),
        "source_relations_sha256": source["exports"]["relations.jsonl"],
        "source_revision": source["source_revision"],
        "evaluator_sha256": digest(Path(__file__)),
        "python": platform.python_version(),
        "executable": resolved,
        "executable_sha256": digest(Path(resolved)),
        "external_backend_calls": 2 * len(records),
        "external_ltl_verdicts": 2 * len(rows),
        "properties": [
            {
                "name": p.name,
                "category": p.category,
                "safety_claim": p.safety_claim,
                "primary_score": p.primary_score,
            }
            for p in suite
        ],
        "initial": summarize([r for r in rows if r["initial"]]),
        "all_states": summarize(rows),
        "seeds": [
            {
                "seed": seed,
                "initial": summarize(
                    [r for r in rows if r["seed"] == seed and r["initial"]]
                ),
                "all_states": summarize([r for r in rows if r["seed"] == seed]),
                "initial_all_properties_agree_maps": sum(
                    m["initial"]["agreement"] == 1 for m in maps if m["seed"] == seed
                ),
            }
            for seed in sorted({r["seed"] for r in rows})
        ],
        "maps": maps,
        "exports": {
            path.relative_to(output_dir).as_posix(): digest(path)
            for path in (queries_path, *sorted(backend_dir.iterdir()))
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved LTL report: {(output_dir / 'report.json').resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    run_experiment(args.run_dir, args.output_dir, args.executable)


if __name__ == "__main__":
    main()
