"""Evaluator-only unseen edge-type diagnostic from frozen saved predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.finite_trace_mlp_vs_jepa import (
    aggregate_rates,
    distribution,
    write_json,
)
from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.evaluation.finite_trace import rate

COMPLETION = "observed_self_loop_completion_evaluation_only"


def diagnose(result_dir):
    """Read complete results, never train/select/modify any predictor or graph."""
    result_dir = Path(result_dir).resolve()
    report = json.loads((result_dir / "report.json").read_text())
    cases_path = result_dir / "cases.jsonl"
    cases_hash = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    if report["status"] != "complete" or cases_hash != report["exports"]["cases.jsonl"]:
        raise ValueError("Need complete, intact saved benchmark results.")
    rows = [json.loads(line) for line in cases_path.read_text().splitlines()]
    if len(rows) != 1080:
        raise ValueError("Need all 1,080 original model-case records.")
    truth = {
        c.name: {
            (t.state, t.action): t.next_state for t in c.make_env().all_transitions()
        }
        for c in radius_stress_cases()
    }
    partials = {
        (r["case"], r["seed"], r["trajectory_budget"]): {
            (tuple(s), a): tuple(t) for s, a, t in r["predictions"]
        }
        for r in rows
        if r["model"] == "observed_edge"
    }
    scores = []
    for row in rows:
        reference = truth[row["case"]]
        partial = partials[row["case"], row["seed"], row["trajectory_budget"]]
        unseen = set(reference) - set(partial)
        loops = {p for p in unseen if reference[p] == p[0]}
        moving = unseen - loops
        predictions = {(tuple(s), a): tuple(t) for s, a, t in row["predictions"]}
        score = {
            **{
                k: row[k]
                for k in ("case", "seed", "family", "trajectory_budget", "model")
            },
            "true_self_loop": rate(
                sum(predictions.get(p) == reference[p] for p in loops), len(loops)
            ),
            "moving": rate(
                sum(predictions.get(p) == reference[p] for p in moving), len(moving)
            ),
            "unseen": row["transitions"]["unseen"],
        }
        assert (
            score["true_self_loop"]["correct"] + score["moving"]["correct"]
            == score["unseen"]["correct"]
        )
        assert len(loops) + len(moving) == score["unseen"]["total"]
        scores.append(score)
        if row["model"] == "observed_edge":
            scores.append(
                {
                    **{
                        k: score[k]
                        for k in ("case", "seed", "family", "trajectory_budget")
                    },
                    "model": COMPLETION,
                    "true_self_loop": rate(len(loops), len(loops)),
                    "moving": rate(0, len(moving)),
                    "unseen": rate(len(loops), len(unseen)),
                }
            )

    def summarize(selected):
        return {
            k: aggregate_rates(r[k] for r in selected)
            for k in ("true_self_loop", "moving", "unseen")
        }

    models = ("observed_edge", "mlp", "jepa", COMPLETION)
    by_budget, paired = {}, {}
    for budget in report["config"]["trajectory_budgets"]:
        selected = [r for r in scores if r["trajectory_budget"] == budget]
        by_budget[str(budget)] = {
            m: summarize([r for r in selected if r["model"] == m]) for m in models
        }
        lookup = {(r["case"], r["seed"], r["model"]): r for r in selected}
        deltas = [
            r["unseen"]["accuracy"]
            - lookup[r["case"], r["seed"], COMPLETION]["unseen"]["accuracy"]
            for r in selected
            if r["model"] == "jepa"
        ]
        paired[str(budget)] = {
            "jepa_minus_completion_unseen": distribution(deltas),
            "jepa_wins": sum(d > 0 for d in deltas),
            "completion_wins": sum(d < 0 for d in deltas),
            "ties": sum(d == 0 for d in deltas),
        }
    diagnostic = {
        "status": "complete",
        "scope": (
            "Post-inference evaluator-only diagnostic; no training, tuning or selection"
        ),
        "completion_policy": (
            "Keep observed successors; use source state on unseen pairs. "
            "This is an evaluation policy, not learned inference."
        ),
        "denominator": (
            "Unseen pairs in full known S x A, including unreachable states; "
            "stratified by reference successor equal/different to source"
        ),
        "cases_sha256": cases_hash,
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validated_original_records": len(rows),
        "by_budget": by_budget,
        "paired_jepa_vs_completion": paired,
    }
    write_json(result_dir / "unseen_edge_diagnostics.json", diagnostic)
    lines = [
        "## Evaluator-only unseen edge-type diagnostic",
        "",
        "Accuracies use all unseen pairs, including unreachable states. Moving edges "
        "have a real successor different from their source. The self-loop completion "
        "is an **evaluation-only context**, not learned inference or the raw partial "
        "observed-edge baseline. Its moving-edge accuracy is 0% by construction.",
        "",
        "| Budget | Model / evaluation policy | Unseen accuracy | "
        "True self-loop unseen accuracy | Moving unseen accuracy |",
        "|---:|---|---:|---:|---:|",
    ]
    for budget, methods in by_budget.items():
        for model, values in methods.items():
            rates = [
                values[k]["macro"]["mean"]
                for k in ("unseen", "true_self_loop", "moving")
            ]
            lines.append(
                f"| {budget} | {model} | "
                + " | ".join("n/a" if v is None else f"{100 * v:.2f}%" for v in rates)
                + " |"
            )
    lines.append("")
    with (result_dir / "unseen_edge_diagnostics.md").open(
        "x", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines))
    return diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.result_dir)


if __name__ == "__main__":
    main()
