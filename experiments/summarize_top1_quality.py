"""Audit and summarize the completed, bounded model-quality experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

if __package__:
    from .top1_quality_diagnosis import (
        BASELINES,
        ROOT,
        SEEDS,
        digest,
        write_csv,
        write_json,
    )
else:
    from top1_quality_diagnosis import (
        BASELINES,
        ROOT,
        SEEDS,
        digest,
        write_csv,
        write_json,
    )


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {k: (v == "True" if v in ("True", "False") else v) for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


def temporal_summary(rows):
    by_property = {}
    for prop in sorted({r["property"] for r in rows}):
        subset = [r for r in rows if r["property"] == prop]
        tp = sum(r["real"] and r["top1"] for r in subset)
        tn = sum(not r["real"] and not r["top1"] for r in subset)
        fp = sum(not r["real"] and r["top1"] for r in subset)
        fn = sum(r["real"] and not r["top1"] for r in subset)
        by_property[prop] = {
            "comparisons": len(subset),
            "matched": tp + tn,
            "agreement": (tp + tn) / len(subset),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    keys = {(r["seed"], r["case"]) for r in rows}
    initial = [r for r in rows if r["initial"]]
    all_six = sum(
        all(r["agreement"] for r in initial if (r["seed"], r["case"]) == key)
        for key in keys
    )
    return {
        "comparisons": len(rows),
        "matched": sum(r["agreement"] for r in rows),
        "agreement": sum(r["agreement"] for r in rows) / len(rows),
        "initial_formula_agreement": sum(r["agreement"] for r in initial)
        / len(initial),
        "initial_all_six_maps": all_six,
        "maps": len(keys),
        "by_property": by_property,
    }


def behavior_summary(run_dir, model):
    directory = run_dir / "behavioral" / model
    report = json.loads((directory / "report.json").read_text())
    relations = read_csv(directory / "maps.csv")
    ctl = read_csv(directory / "ctl_queries.csv")
    ltl = read_csv(run_dir / "temporal" / f"{model}_ltl" / "queries.csv")
    sanity = json.loads(
        (run_dir / "temporal" / f"{model}_sanity" / "report.json").read_text()
    )
    if sanity["status"] != "passed" or sanity["mismatch_count"]:
        raise AssertionError("Backend sanity check failed.")
    graph_hash = report["exports"]["relations.jsonl"]
    if graph_hash != digest(directory / "relations.jsonl"):
        raise AssertionError("Behavioral graph hash mismatch.")
    ltl_report = json.loads(
        (run_dir / "temporal" / f"{model}_ltl" / "report.json").read_text()
    )
    if (
        graph_hash != ltl_report["source_relations_sha256"]
        or graph_hash != sanity["source_input_sha256"]["relations.jsonl"]
    ):
        raise AssertionError("CTL and LTL checked different graph exports.")
    if ltl_report["executable_sha256"] != sanity["backend"]["sha256"]:
        raise AssertionError("CTL and LTL used different binaries.")
    # Link the existing full LTL pipeline to the strict paired-formula audit.
    pair_lookup = {
        (r["seed"], r["case"], r["row"], r["column"], r["property"]): r for r in ltl
    }
    linked = 0
    for row in read_csv(run_dir / "temporal" / f"{model}_sanity" / "checks.csv"):
        if row["comparison"] != "nuxmv_ctl_vs_nuxmv_ltl":
            continue
        other = pair_lookup[
            row["seed"], row["map"], row["row"], row["column"], row["right_formula"]
        ]
        if row["right_verdict"] != other[row["graph"]]:
            raise AssertionError("Full LTL run differs from paired-formula audit.")
        linked += 1
    # Also link every native CTL state/formula verdict to this strict audit.
    ctl_lookup = {
        (r["seed"], r["case"], r["row"], r["column"], r["property"]): r for r in ctl
    }
    native_links = 0
    for row in read_csv(run_dir / "temporal" / f"{model}_sanity" / "checks.csv"):
        if row["comparison"] != "internal_ctl_vs_nuxmv_ctl":
            continue
        other = ctl_lookup[
            row["seed"], row["map"], row["row"], row["column"], row["left_formula"]
        ]
        if row["left_verdict"] != other[row["graph"]]:
            raise AssertionError("Native CTL export differs from strict audit.")
        native_links += 1
    result = {
        "ctl": temporal_summary(ctl),
        "ltl": temporal_summary(ltl),
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
        "by_seed": [],
        "by_family": {},
        "audits": report["audit"],
        "backend_audit": {
            "internal_ctl_vs_nuxmv_ctl": sanity["internal_ctl_vs_nuxmv_ctl"],
            "paired_formulas": sanity["paired_formulas"],
            "mismatches": sanity["mismatch_count"],
            "full_ltl_vs_paired_audit_verdicts": linked,
            "native_ctl_export_vs_audit_verdicts": native_links,
            "nuxmv_sha256": sanity["backend"]["sha256"],
        },
    }
    for seed in SEEDS:
        result["by_seed"].append(
            {
                "seed": seed,
                "ctl": temporal_summary([r for r in ctl if int(r["seed"]) == seed]),
                "ltl": temporal_summary([r for r in ltl if int(r["seed"]) == seed]),
                "relations": next(
                    r["relations"] for r in report["seeds"] if r["seed"] == seed
                ),
            }
        )
    for family in sorted({r["family"] for r in relations}):
        # The LTL raw table has case names; join via the relation metadata.
        cases = {r["case"] for r in relations if r["family"] == family}
        result["by_family"][family] = {
            "ctl": temporal_summary([r for r in ctl if r["case"] in cases]),
            "ltl": temporal_summary([r for r in ltl if r["case"] in cases]),
            "initial_bisimulation": sum(
                r["initial_related"]
                for r in relations
                if r["family"] == family
                and r["semantics"] == "ctl"
                and r["relation"] == "bisimulation"
            ),
        }
    return result, report, relations, ctl, ltl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "outputs/top1_quality/round1"
    )
    args = parser.parse_args()
    run = args.run_dir
    if (run / "summary.json").exists():
        raise ValueError("Summary already exists; refusing to overwrite.")
    diagnosis = json.loads((run / "diagnosis/report.json").read_text())
    selection = json.loads((run / "selection.json").read_text())
    protocol = json.loads((ROOT / "configs/top1_quality_protocol.json").read_text())
    if selection["protocol_sha256"] != digest(
        ROOT / "configs/top1_quality_protocol.json"
    ):
        raise AssertionError("Frozen protocol changed.")
    rows, histories = [], []
    for name in ("baseline", *(c["name"] for c in protocol["candidates"])):
        for seed in SEEDS:
            if name == "baseline":
                payload = next(m for m in diagnosis["models"] if m["seed"] == seed)
                checkpoint = BASELINES[seed]
                history = payload["training"]["history"]
            else:
                directory = run / name / f"seed_{seed}"
                payload = json.loads((directory / "report.json").read_text())
                if payload["protocol_sha256"] != selection["protocol_sha256"]:
                    raise AssertionError("Training used a different protocol.")
                checkpoint = directory / "model.pt"
                history = [
                    json.loads(line)
                    for line in (directory / "history.jsonl").read_text().splitlines()
                ]
                config = next(c for c in protocol["candidates"] if c["name"] == name)
                if [r["epoch"] for r in history] != list(
                    range(config["initial_epoch"] + 1, 301)
                ):
                    raise AssertionError(
                        "Training epochs differ from the fixed budget."
                    )
            if payload["checkpoint_sha256"] != digest(checkpoint):
                raise AssertionError("Checkpoint digest changed.")
            score = {r["split"]: r for r in payload["scores"]["by_split"]}
            rows.append(
                {
                    "candidate": name,
                    "seed": seed,
                    "epochs": 100 if name == "baseline" else 300,
                    "parameters": payload["trainable_parameters"],
                    **{f"{s}_accuracy": score[s]["accuracy"] for s in score},
                    **{f"{s}_errors": score[s]["errors"] for s in score},
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": digest(checkpoint),
                    "training_seconds_new": 0
                    if name == "baseline"
                    else payload["training_seconds"],
                }
            )
            histories.append(
                {
                    "candidate": name,
                    "seed": seed,
                    "final_loss": history[-1]["loss"],
                    "final_prediction_loss": history[-1]["prediction_loss"],
                    "validation_checkpoints": [
                        {"epoch": r["epoch"], **r["validation"]}
                        for r in history
                        if "validation" in r
                    ],
                }
            )
    baseline = behavior_summary(run, "baseline")
    best = behavior_summary(run, "best")
    transitions = {
        model: {(m["seed"], m["case"]): m for m in result[1]["maps"]}
        for model, result in (("baseline", baseline), ("best", best))
    }
    map_deltas, verdict_deltas = [], []
    for key, before in transitions["baseline"].items():
        after = transitions["best"][key]
        row = {
            "seed": key[0],
            "map": key[1],
            "family": before["family"],
            "transitions": before["pairs"],
            "baseline_correct": before["correct_top1_pairs"],
            "best_correct": after["correct_top1_pairs"],
            "correct_gain": after["correct_top1_pairs"] - before["correct_top1_pairs"],
        }
        for model, value in (("baseline", baseline), ("best", best)):
            rel = next(
                r
                for r in value[2]
                if (int(r["seed"]), r["case"]) == key
                and r["semantics"] == "ctl"
                and r["relation"] == "bisimulation"
            )
            row[f"{model}_bisimulation"] = rel["initial_related"]
            for logic, index in (("ctl", 3), ("ltl", 4)):
                subset = [
                    r
                    for r in value[index]
                    if (int(r["seed"]), r["case"]) == key and r["initial"]
                ]
                row[f"{model}_{logic}_all_six"] = all(r["agreement"] for r in subset)
        map_deltas.append(row)
    for logic, index in (("ctl", 3), ("ltl", 4)):
        left = {
            (r["seed"], r["case"], r["row"], r["column"], r["property"]): r
            for r in baseline[index]
        }
        right = {
            (r["seed"], r["case"], r["row"], r["column"], r["property"]): r
            for r in best[index]
        }
        if left.keys() != right.keys():
            raise AssertionError("Temporal query coverage differs between models.")
        for key, before in left.items():
            after = right[key]
            if before["real"] != after["real"]:
                raise AssertionError("Ground truth changed between models.")
            if before["top1"] != after["top1"]:
                verdict_deltas.append(
                    {
                        "logic": logic,
                        "seed": key[0],
                        "map": key[1],
                        "row": key[2],
                        "column": key[3],
                        "formula": key[4],
                        "initial": before["initial"],
                        "real": before["real"],
                        "baseline": before["top1"],
                        "best": after["top1"],
                    }
                )
    preservation = json.loads((run / "diagnosis/preservation_before.json").read_text())
    changed = [p for p, h in preservation.items() if digest(ROOT / p) != h]
    if changed:
        raise AssertionError(f"Preserved input files changed: {changed}")
    report = {
        "status": "completed",
        "protocol_sha256": selection["protocol_sha256"],
        "selection": selection,
        "split_audit": diagnosis["split_audit"],
        "model_scores": rows,
        "convergence": histories,
        "means": {
            name: {
                s: mean(r[f"{s}_accuracy"] for r in rows if r["candidate"] == name)
                for s in ("train", "validation", "validation_rotated", "stress")
            }
            for name in ("baseline", *(c["name"] for c in protocol["candidates"]))
        },
        "baseline_behavior": baseline[0],
        "best_behavior": best[0],
        "map_changes": {
            "top1_improved": sum(r["correct_gain"] > 0 for r in map_deltas),
            "top1_unchanged": sum(r["correct_gain"] == 0 for r in map_deltas),
            "top1_worsened": sum(r["correct_gain"] < 0 for r in map_deltas),
            **{
                f"{metric}_{change}": sum(
                    (not r[f"baseline_{metric}"] and r[f"best_{metric}"])
                    if change == "gained"
                    else (r[f"baseline_{metric}"] and not r[f"best_{metric}"])
                    for r in map_deltas
                )
                for metric in ("bisimulation", "ctl_all_six", "ltl_all_six")
                for change in ("gained", "lost")
            },
            "better_top1_but_ctl_all_six_lost": sum(
                r["correct_gain"] > 0
                and r["baseline_ctl_all_six"]
                and not r["best_ctl_all_six"]
                for r in map_deltas
            ),
            "better_top1_but_ltl_all_six_lost": sum(
                r["correct_gain"] > 0
                and r["baseline_ltl_all_six"]
                and not r["best_ltl_all_six"]
                for r in map_deltas
            ),
        },
        "preservation": {
            "checked_files": len(preservation),
            "changed_files": changed,
            "passed": True,
        },
        "candidate_runs": len(rows) - 3,
        "tuning_stopped": True,
    }
    write_csv(run / "model_scores.csv", rows)
    write_csv(run / "map_deltas.csv", map_deltas)
    if verdict_deltas:
        write_csv(run / "verdict_deltas.csv", verdict_deltas)
    write_json(run / "summary.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("means", "map_changes", "preservation")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
