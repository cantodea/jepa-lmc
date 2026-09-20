"""Audit round-2 training, held-out retrieval and unchanged formal pipelines."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import mean

from experiments.improve_top1_ranking import DEFAULT_RUN, PROTOCOL
from experiments.summarize_top1_quality import (
    audit_evaluation,
    behavior_summary,
    read_csv,
    write_model_bank,
)
from experiments.top1_quality_diagnosis import (
    BASELINES,
    ROOT,
    SEEDS,
    digest,
    write_csv,
    write_json,
)

ROUND1 = ROOT / "outputs/top1_quality/round1"


def write_public_summary(result, destination):
    """Publish aggregate evidence; retain full per-run details in the private bundle."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Public summary already exists; refusing to overwrite.")
    public = copy.deepcopy(result)
    for curve in public["convergence"]:
        curve.pop("epoch_metrics")
    diagnostics = []
    for record in public["grouped_diagnostics"]:
        if record["candidate"] not in ("baseline", result["selection"]["selected"]):
            continue
        for key in ("by_family", "by_action", "by_motion", "by_kind"):
            record[key] = [r for r in record[key] if r["split"] == "stress"]
        diagnostics.append(record)
    public["grouped_diagnostics"] = diagnostics
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, public)


def audit_history(directory, metadata, protocol):
    rows = [
        json.loads(r) for r in (directory / "history.jsonl").read_text().splitlines()
    ]
    epochs = metadata["stopped_additional_epoch"]
    if [r["additional_epoch"] for r in rows] != list(range(epochs + 1)):
        raise AssertionError("Training history has omitted or repeated epochs.")
    if any(r["examples"] != 4736 or r["optimizer_steps"] != 10 for r in rows[1:]):
        raise AssertionError("Training data exposure differs from protocol.")
    checked = [r for r in rows if "validation" in r]
    if [r["additional_epoch"] for r in checked] != list(
        range(0, epochs + 1, protocol["validation_interval"])
    ):
        raise AssertionError("Validation schedule changed.")
    best = max(checked, key=lambda r: r["validation"]["accuracy"])
    if (
        best["additional_epoch"] != metadata["best_additional_epoch"]
        or best["validation"] != metadata["validation"]
    ):
        raise AssertionError("Best checkpoint was not selected by validation.")
    stale = 0
    highest = checked[0]["validation"]["accuracy"]
    stop_at = protocol["max_additional_epochs"]
    for row in checked[1:]:
        score = row["validation"]["accuracy"]
        if score > highest:
            highest, stale = score, 0
        else:
            stale += 1
        if (
            row["additional_epoch"] >= protocol["minimum_additional_epochs"]
            and stale >= protocol["patience_checks"]
        ):
            stop_at = row["additional_epoch"]
            break
    if epochs != stop_at:
        raise AssertionError("Early stopping does not match the fixed rule.")
    return {
        "candidate": metadata["candidate"],
        "seed": metadata["seed"],
        "selected_additional_epoch": metadata["best_additional_epoch"],
        "stopped_additional_epoch": epochs,
        "stop_reason": metadata["stop_reason"],
        "training_seconds": metadata["training_seconds"],
        "validation_checkpoints": [
            {"epoch": r["additional_epoch"], "accuracy": r["validation"]["accuracy"]}
            for r in checked
        ],
        "epoch_metrics": [
            {
                k: r[k]
                for k in (
                    "additional_epoch",
                    "loss",
                    "prediction",
                    "ranking",
                    "learning_rate",
                )
            }
            for r in rows[1:]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if (run / "summary.json").exists():
        raise ValueError("Summary already exists; choose a fresh output directory.")
    protocol = json.loads(PROTOCOL.read_text())
    started = json.loads((run / "study_started.json").read_text())
    selection = json.loads((run / "selection.json").read_text())
    if started["protocol_sha256"] != digest(PROTOCOL) or selection[
        "protocol_sha256"
    ] != digest(PROTOCOL):
        raise AssertionError("Protocol changed after training/selection.")
    records = []
    curves = []
    audits = 0
    for seed in SEEDS:
        scores = json.loads(
            (ROUND1 / "diagnosis" / f"seed_{seed}" / "report.json").read_text()
        )
        audits += audit_evaluation(ROUND1 / "diagnosis" / f"seed_{seed}", scores, seed)
        records.append(
            {
                "candidate": "baseline",
                "seed": seed,
                "scores": scores,
                "parameters": 90776,
                "best_epoch": 0,
            }
        )
    for config in protocol["candidates"]:
        for seed in SEEDS:
            directory = run / config["name"] / f"seed_{seed}"
            metadata = json.loads((directory / "report.json").read_text())
            if (
                metadata["protocol_sha256"] != digest(PROTOCOL)
                or metadata["selection_sha256"] != digest(run / "selection.json")
                or metadata["checkpoint_sha256"] != digest(directory / "model.pt")
                or metadata["last_checkpoint_sha256"]
                != digest(directory / "last_model.pt")
                or metadata["baseline_sha256"] != digest(BASELINES[seed])
                or metadata["source_sha256"] != started["source_sha256"]
            ):
                raise AssertionError("Model, baseline, source or selection changed.")
            if (
                metadata["training_tensor_sha256"]
                != "a2184db8da410d7b664a0a8d5e13b4fefc167aa088b676d23f23886aa39f0eaf"
            ):
                raise AssertionError("Training tensors differ from original baseline.")
            curves.append(audit_history(directory, metadata, protocol))
            audits += audit_evaluation(
                directory / "evaluation", metadata["scores"], seed
            )
            records.append(
                {
                    "candidate": config["name"],
                    "seed": seed,
                    "scores": metadata["scores"],
                    "parameters": metadata["trainable_parameters"],
                    "best_epoch": metadata["best_additional_epoch"],
                }
            )
    table = []
    grouped = {}
    for record in records:
        row = {
            "candidate": record["candidate"],
            "seed": record["seed"],
            "parameters": record["parameters"],
            "best_additional_epoch": record["best_epoch"],
        }
        for split in record["scores"]["by_split"]:
            row[f"{split['split']}_accuracy"] = split["accuracy"]
            row[f"{split['split']}_errors"] = split["errors"]
        table.append(row)
        grouped.setdefault(record["candidate"], []).append(record)
    means = {
        name: {
            split: mean(
                next(
                    s["accuracy"]
                    for s in r["scores"]["by_split"]
                    if s["split"] == split
                )
                for r in items
            )
            for split in ("train", "validation", "validation_rotated", "stress")
        }
        for name, items in grouped.items()
    }
    for option in selection["options"]:
        if (
            abs(
                means[option["candidate"]]["validation"]
                - option["mean_validation_accuracy"]
            )
            > 1e-12
        ):
            raise AssertionError("Final scores differ from validation selection.")
    baseline_behavior = behavior_summary(ROUND1, "baseline")[0]
    best_behavior = behavior_summary(run, "best")[0]
    # Compare the actual real graph, labels and initial indices, not just map names.
    baseline_graphs = {
        (r["seed"], r["case"]): r
        for r in map(
            json.loads,
            (ROUND1 / "behavioral/baseline/relations.jsonl").read_text().splitlines(),
        )
    }
    graph_pairs = 0
    for r in map(
        json.loads, (run / "behavioral/best/relations.jsonl").read_text().splitlines()
    ):
        previous = baseline_graphs[r["seed"], r["case"]]
        if (
            any(r[k] != previous[k] for k in ("states", "initial", "actions", "labels"))
            or r["graphs"]["real"] != previous["graphs"]["real"]
        ):
            raise AssertionError("Baseline and best used different concrete systems.")
        graph_pairs += 1
    if graph_pairs != 72:
        raise AssertionError("Formal comparison must contain all 72 graph pairs.")
    # Join all per-map retrieval and temporal outcomes for the selected model.
    selected_records = {r["seed"]: r for r in grouped[selection["selected"]]}
    deltas = []
    relations = [
        read_csv(p / "maps.csv")
        for p in (ROUND1 / "behavioral/baseline", run / "behavioral/best")
    ]
    formal = [
        json.loads((p / "report.json").read_text())
        for p in (ROUND1 / "behavioral/baseline", run / "behavioral/best")
    ]
    ltl_tables = [
        read_csv(p / "queries.csv")
        for p in (ROUND1 / "temporal/baseline_ltl", run / "temporal/best_ltl")
    ]
    for old in grouped["baseline"]:
        seed = old["seed"]
        new_maps = {
            r["map"]: r
            for r in selected_records[seed]["scores"]["by_map"]
            if r["split"] == "stress"
        }
        for m in old["scores"]["by_map"]:
            if m["split"] != "stress":
                continue
            name = m["map"]
            new = new_maps[name]
            row = {
                "seed": seed,
                "map": name,
                "family": m["family"],
                "baseline_accuracy": m["accuracy"],
                "best_accuracy": new["accuracy"],
                "baseline_errors": m["errors"],
                "best_errors": new["errors"],
                "accuracy_gain": new["accuracy"] - m["accuracy"],
            }
            for side, rel, rep, ltl in zip(
                ("baseline", "best"), relations, formal, ltl_tables, strict=True
            ):
                row[f"{side}_initial_bisimulation"] = next(
                    r["initial_related"]
                    for r in rel
                    if int(r["seed"]) == seed
                    and r["case"] == name
                    and r["semantics"] == "ctl"
                    and r["relation"] == "bisimulation"
                )
                fm = next(
                    r for r in rep["maps"] if r["seed"] == seed and r["case"] == name
                )
                row[f"{side}_initial_ctl_disagreements"] = len(
                    fm["initial_ctl_disagreements"]
                )
                initial_ltl = [
                    r
                    for r in ltl
                    if int(r["seed"]) == seed and r["case"] == name and r["initial"]
                ]
                if len(initial_ltl) != 6:
                    raise AssertionError("Initial LTL formula coverage changed.")
                row[f"{side}_initial_ltl_disagreements"] = sum(
                    not r["agreement"] for r in initial_ltl
                )
                expected_errors = m["errors"] if side == "baseline" else new["errors"]
                if fm["pairs"] - fm["correct_top1_pairs"] != expected_errors:
                    raise AssertionError(
                        "Formal graph and retrieval evaluation disagree."
                    )
            deltas.append(row)
    preservation = json.loads((run / "preservation_before.json").read_text())
    changed = [
        name
        for name, sha in preservation["files"].items()
        if digest(ROOT / name) != sha
    ]
    if changed:
        raise AssertionError(f"Existing results changed: {changed}")
    bank = write_model_bank(run, selection)
    result = {
        "status": "completed",
        "protocol_sha256": digest(PROTOCOL),
        "protocol_commit": started["source_revision"],
        "selection": selection,
        "split_audit": started["split_audit"],
        "model_scores": table,
        "means": means,
        "convergence": curves,
        "grouped_diagnostics": [
            {
                "candidate": r["candidate"],
                "seed": r["seed"],
                **{
                    k: r["scores"][k]
                    for k in (
                        "by_split",
                        "by_family",
                        "by_action",
                        "by_motion",
                        "by_kind",
                        "self_retrieval",
                    )
                },
            }
            for r in records
        ],
        "baseline_behavior": baseline_behavior,
        "best_behavior": best_behavior,
        "baseline_behavior_reused_from": "outputs/top1_quality/round1",
        "same_concrete_graph_label_initial_pairs": graph_pairs,
        "map_changes": deltas,
        "model_bank": bank,
        "audited_evaluation_transitions": audits,
        "preservation": {
            "files": len(preservation["files"]),
            "changed": changed,
            "passed": True,
        },
        "tuning_stopped": True,
        "new_abstraction_experiments": False,
        "evaluation_implementation_note": (
            "All training used the locked source hashes in study_started.json. "
            "After training, an evaluation-only guard was corrected to compare "
            "exact error/transition counts: correct/total versus 1-errors/total "
            "differed by one floating-point ULP for the control seed 20260805. "
            "Its raw export was hash/coverage audited and validation replayed "
            "before finalizing the report. Weights and frozen selection are unchanged."
        ),
    }
    write_csv(run / "model_scores.csv", table)
    write_csv(run / "map_deltas.csv", deltas)
    write_json(run / "summary.json", result)
    if args.public_output:
        write_public_summary(result, args.public_output)
    print(
        json.dumps(
            {
                "means": means,
                "selected": selection["selected"],
                "preservation": result["preservation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
