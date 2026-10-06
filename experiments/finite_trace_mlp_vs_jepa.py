"""Finite-trace-only, from-scratch MLP vs existing JEPA on fixed stress maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.good_bad_trace_sanity import benchmark_catalogue, stream_seed
from jepa_lmc.evaluation.finite_trace import (
    LTLBackendCache,
    disjoint_heldout_traces,
    evaluate_predictor,
    prediction_system,
    trace_scores,
)
from jepa_lmc.evaluation.structural import DEFINITIONS, SEMANTICS, summarize_model_pairs
from jepa_lmc.evaluation.trace_sanity import (
    KnownProblem,
    Trace,
    is_execution,
    observed_edges,
    observed_self_loop_completion,
    relation_edges,
    sample_good_traces,
)
from jepa_lmc.learning.finite_trace import (
    FiniteTraceData,
    fit_finite_model,
    predict_successors,
)
from jepa_lmc.verification.gridworld import gridworld_to_transition_system

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/finite_trace_mlp_vs_jepa.json"
MODELS = ("observed_edge", "mlp", "jepa")
OLD_RUN = ROOT / "docs/results/good_bad_trace_sanity"
NEW_SOURCES = (
    "experiments/finite_trace_mlp_vs_jepa.py",
    "src/jepa_lmc/learning/finite_trace.py",
    "src/jepa_lmc/evaluation/finite_trace.py",
    "tests/test_finite_trace_mlp_vs_jepa.py",
)
SHARED_SOURCES = (
    "experiments/good_bad_trace_sanity.py",
    "src/jepa_lmc/evaluation/trace_sanity.py",
    "src/jepa_lmc/benchmarks/radius_stress.py",
    "src/jepa_lmc/learning/model.py",
    "src/jepa_lmc/learning/ranking.py",
    "src/jepa_lmc/learning/training.py",
    "src/jepa_lmc/learning/data.py",
    "src/jepa_lmc/evaluation/structural.py",
    "src/jepa_lmc/verification/behavioral_relations.py",
    "src/jepa_lmc/verification/transition_system.py",
    "src/jepa_lmc/verification/gridworld.py",
    "src/jepa_lmc/envs/gridworld.py",
    "src/jepa_lmc/verification/nuxmv.py",
    "src/jepa_lmc/benchmarks/ctl_suite.py",
    "src/jepa_lmc/benchmarks/ltl_suite.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def read_training_pools():
    report = json.loads((OLD_RUN / "report.json").read_text())
    path = OLD_RUN / "trace_pools.jsonl"
    if digest(path) != report["exports"][path.name]:
        raise ValueError("Original trace-pool integrity check failed.")
    pools = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = row["seed"], row["case"]
        if key in pools:
            raise ValueError("Duplicate source trace pool.")
        pools[key] = tuple(
            Trace(tuple(tuple(s) for s in t["states"]), tuple(t["actions"]))
            for t in row["goods"]
        )
    return pools


def known_catalogue(env):
    """Only known S,I,A,L and dimensions; no enumeration of true transitions."""
    states = tuple(env.all_states())
    labels = {}
    for s in states:
        labels[s] = frozenset(
            ({"danger"} if env.is_danger(s) else {"safe"})
            | ({"goal"} if env.is_goal(s) else set())
        )
    return KnownProblem(states, frozenset((env.start,)), tuple(env.ACTIONS), labels)


def distribution(values):
    values = [v for v in values if v is not None]
    return {
        "evaluated": len(values),
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "std": statistics.pstdev(values) if values else None,
    }


def aggregate_rates(values):
    values = list(values)
    correct, total = sum(v["correct"] for v in values), sum(v["total"] for v in values)
    return {
        "macro": distribution(v["accuracy"] for v in values),
        "pooled": {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else None,
        },
    }


def aggregate(rows):
    rows = list(rows)
    formal = summarize_model_pairs(
        {"seed": r["seed"], "case": r["case"], "family": r["family"], **r["formal"]}
        for r in rows
    )
    formal.pop("mismatch_cases")
    return {
        "cases": len(rows),
        "observed_pairs": distribution(r["observed_pairs"] for r in rows),
        "unseen_pairs": distribution(r["unseen_pairs"] for r in rows),
        "transitions": {
            name: aggregate_rates(r["transitions"][name] for r in rows)
            for name in (
                "observed",
                "unseen",
                "overall",
                "reachable_unseen",
                "unreachable_unseen",
            )
        },
        "heldout": {
            name: aggregate_rates(r["heldout"][name] for r in rows)
            for name in (
                "acceptance",
                "with_unseen_pairs",
                "observed_only",
            )
        },
        "training_trace_acceptance": aggregate_rates(
            r["training_trace_acceptance"] for r in rows
        ),
        "observed_minus_unseen_gap": distribution(
            r["transitions"]["observed"]["accuracy"]
            - r["transitions"]["unseen"]["accuracy"]
            for r in rows
            if r["transitions"]["unseen"]["accuracy"] is not None
        ),
        "formal": formal,
        "initial_ctl_formula_agreement": aggregate_rates(
            {
                "correct": sum(
                    v["real"] == v["learned"]
                    for v in r["formal"]["diagnostics"]["initial_ctl_verdicts"]
                ),
                "total": 6,
                "accuracy": sum(
                    v["real"] == v["learned"]
                    for v in r["formal"]["diagnostics"]["initial_ctl_verdicts"]
                )
                / 6,
            }
            for r in rows
        ),
        "initial_ltl_formula_agreement": aggregate_rates(
            r["formal"]["diagnostics"]["ltl"]["initial_formula_agreement"] for r in rows
        ),
    }


def paired_comparison(rows):
    cases = {(r["seed"], r["case"]): {} for r in rows}
    for row in rows:
        cases[row["seed"], row["case"]][row["model"]] = row
    differences = [
        c["jepa"]["transitions"]["unseen"]["accuracy"]
        - c["mlp"]["transitions"]["unseen"]["accuracy"]
        for c in cases.values()
    ]
    return {
        "cases": len(cases),
        "jepa_minus_mlp_unseen_accuracy": distribution(differences),
        "jepa_wins": sum(d > 0 for d in differences),
        "mlp_wins": sum(d < 0 for d in differences),
        "ties": sum(d == 0 for d in differences),
    }


def run_experiment(output_dir, *, config_path=CONFIG, executable=None):
    config = json.loads(Path(config_path).read_text())
    maps, october = benchmark_catalogue()
    if (
        config["seeds"],
        config["trajectory_budgets"],
        config["trajectory_length"],
        config["training_pool_size"],
    ) != (
        [20260804, 20260805, 20260806],
        [1, 2, 4, 8, 16],
        16,
        16,
    ):
        raise ValueError("The first-round benchmark/data protocol must be retained.")
    source_pools = read_training_pools()
    if set(source_pools) != {(s, c.name) for s in config["seeds"] for c in maps}:
        raise ValueError("Source pools differ from the 72 seed-map cases.")
    # Hash comparisons use absolute ROOT paths; resolve the destination before
    # excluding this run from immutable historical-result protection.
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    backend = LTLBackendCache(
        executable=executable, log_path=output_dir / "ltl_backend.jsonl.gz"
    )
    started = time.perf_counter()
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    source_hashes = {
        name: digest(ROOT / name) for name in (*NEW_SOURCES, *SHARED_SOURCES)
    }
    preserved = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted((ROOT / "docs/results").rglob("*"))
        if p.is_file() and not p.is_relative_to(output_dir)
    }
    write_json(output_dir / "experiment_config.json", config)
    write_json(
        output_dir / "run_started.json",
        {
            "base_revision": base,
            "config_sha256": digest(config_path),
            "source_sha256": source_hashes,
            "october_source": october,
            "training_pool_source_sha256": digest(OLD_RUN / "trace_pools.jsonl"),
            "historical_result_sha256": preserved,
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "device": "cpu",
                "threads": config["training"]["threads"],
                "nuXmv_executable": backend.executable,
                "nuXmv_sha256": digest(backend.executable),
            },
        },
    )
    records, pool_count = [], 0
    with (
        (output_dir / "cases.jsonl").open("x", encoding="utf-8") as cases_file,
        (output_dir / "trace_pools.jsonl").open("x", encoding="utf-8") as pools_file,
    ):
        for case in maps:
            env = case.make_env()
            problem = known_catalogue(env)
            for seed in config["seeds"]:
                goods = source_pools[seed, case.name]
                sampled = sample_good_traces(
                    env, seed=stream_seed(seed, case.name, "good")
                )
                if goods != sampled or not all(is_execution(env, t) for t in goods):
                    raise AssertionError(
                        "Training pools must exactly reproduce "
                        "the original real traces."
                    )
                heldout_seed = stream_seed(seed, case.name, "finite_trace_heldout_real")
                heldout = disjoint_heldout_traces(
                    env,
                    goods,
                    seed=heldout_seed,
                    count=config["heldout_trajectories"],
                    length=config["trajectory_length"],
                )
                pool_count += 1
                pools_file.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "case": case.name,
                            "family": case.family,
                            "training_pool_source": (
                                "unchanged good_bad_trace_sanity/trace_pools.jsonl"
                            ),
                            "heldout_rng_seed": heldout_seed,
                            "goods": [asdict(t) for t in goods],
                            "heldout": [asdict(t) for t in heldout],
                            "all_executions_validated": True,
                            "full_pool_disjoint": True,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                pools_file.flush()
                previous_pairs = frozenset()
                initialization = stream_seed(
                    seed, case.name, "finite_trace_initialization"
                ) % (2**63)
                for budget in config["trajectory_budgets"]:
                    traces = goods[:budget]
                    data = FiniteTraceData.from_traces(
                        problem, traces, height=env.height, width=env.width
                    )
                    if not previous_pairs <= data.observed_pairs:
                        raise AssertionError(
                            "Observed nested budgets must be monotone."
                        )
                    previous_pairs = data.observed_pairs
                    partial = observed_edges(traces)
                    raw = {(s, a): t for s, a, t in partial}
                    predictions, training = (
                        {"observed_edge": raw},
                        {"observed_edge": None},
                    )
                    # Both models are frozen before complete reference evaluation.
                    for kind in ("mlp", "jepa"):
                        model, metadata = fit_finite_model(
                            data, kind=kind, seed=initialization, config=config
                        )
                        predictions[kind] = predict_successors(model, data, kind=kind)
                        training[kind] = metadata
                        del model
                    if (
                        training["mlp"]["training_sha256"]
                        != training["jepa"]["training_sha256"]
                    ):
                        raise AssertionError(
                            "Models received different training examples."
                        )
                    real = gridworld_to_transition_system(env)
                    if not partial <= relation_edges(real):
                        raise AssertionError("Observed edges must be real transitions.")
                    for kind in MODELS:
                        graph = (
                            observed_self_loop_completion(problem, partial)
                            if kind == "observed_edge"
                            else prediction_system(problem, predictions[kind])
                        )
                        row = {
                            "seed": seed,
                            "case": case.name,
                            "family": case.family,
                            "trajectory_budget": budget,
                            "model": kind,
                            "observed_steps": len(data.actions),
                            "observed_pairs": len(data.observed_pairs),
                            "unseen_pairs": len(problem.states) * len(problem.actions)
                            - len(data.observed_pairs),
                            "trace_sha256": data.trace_sha256,
                            "training_sha256": data.training_sha256,
                            "training": training[kind],
                            "predictions": [
                                (s, a, t)
                                for (s, a), t in sorted(predictions[kind].items())
                            ],
                            "formal_model": "observed_self_loop_completion"
                            if kind == "observed_edge"
                            else "predicted_total_model",
                            "training_trace_acceptance": trace_scores(
                                problem, traces, predictions[kind], data.observed_pairs
                            )["acceptance"],
                            **evaluate_predictor(
                                problem,
                                data.observed_pairs,
                                predictions[kind],
                                real=real,
                                formal_system=graph,
                                heldout=heldout,
                                backend=backend,
                            ),
                        }
                        records.append(row)
                        cases_file.write(
                            json.dumps(row, separators=(",", ":"), allow_nan=False)
                            + "\n"
                        )
                        cases_file.flush()
                    print(
                        f"[{len(records)}/1080] {case.name} "
                        f"seed={seed} budget={budget}: "
                        + "; ".join(
                            f"{r['model']} "
                            f"observed={r['transitions']['observed']['accuracy']:.3f} "
                            f"unseen={r['transitions']['unseen']['accuracy']:.3f}"
                            for r in records[-3:]
                        ),
                        flush=True,
                    )
    if source_hashes != {n: digest(ROOT / n) for n in source_hashes}:
        raise AssertionError("Source changed during the locked experiment.")
    if preserved != {n: digest(ROOT / n) for n in preserved}:
        raise AssertionError("A historical result was modified.")
    budgets = config["trajectory_budgets"]
    report = {
        "schema_version": 1,
        "experiment": "finite_trace_mlp_vs_jepa",
        "status": "complete",
        "config": config,
        "base_revision": base,
        "definitions": DEFINITIONS,
        "seed_map_cases": pool_count,
        "budget_cases": pool_count * len(budgets),
        "model_case_records": len(records),
        "from_scratch_fits": 2 * pool_count * len(budgets),
        "validation": {
            "historical_results_unchanged": True,
            "original_training_pools_exactly_reused": True,
            "all_real_training_traces_validated": True,
            "all_real_heldout_traces_validated": True,
            "all_heldout_pools_disjoint_from_full_training_pool": True,
            "training_examples_identical_across_models": True,
            "training_trace_count": pool_count * config["training_pool_size"],
            "heldout_trace_count": pool_count * config["heldout_trajectories"],
        },
        "by_budget": {
            str(b): {
                m: aggregate(
                    r
                    for r in records
                    if r["trajectory_budget"] == b and r["model"] == m
                )
                for m in MODELS
            }
            for b in budgets
        },
        "by_seed": {
            str(s): {
                str(b): {
                    m: aggregate(
                        r
                        for r in records
                        if r["seed"] == s
                        and r["trajectory_budget"] == b
                        and r["model"] == m
                    )
                    for m in MODELS
                }
                for b in budgets
            }
            for s in config["seeds"]
        },
        "by_family": {
            f: {
                str(b): {
                    m: aggregate(
                        r
                        for r in records
                        if r["family"] == f
                        and r["trajectory_budget"] == b
                        and r["model"] == m
                    )
                    for m in MODELS
                }
                for b in budgets
            }
            for f in sorted({c.family for c in maps})
        },
        "paired_unseen_comparison": {
            str(b): paired_comparison(
                [r for r in records if r["trajectory_budget"] == b]
            )
            for b in budgets
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "training_seconds": {
                m: sum(
                    r["training"]["training_seconds"]
                    for r in records
                    if r["model"] == m
                )
                for m in ("mlp", "jepa")
            },
            "evaluate_model_pair_calls": len(records),
            "greatest_relation_audits": 6 * len(records),
            "independent_bisimulation_partition_checks": 2 * len(records),
            "nuXmv_calls": backend.calls,
            "nuXmv_cache_hits": backend.hits,
            "nuXmv_ltl_verdicts": backend.verdict_count,
        },
        "exports": {
            name: digest(output_dir / name)
            for name in (
                "experiment_config.json",
                "run_started.json",
                "cases.jsonl",
                "trace_pools.jsonl",
                "ltl_backend.jsonl.gz",
            )
        },
    }
    write_json(output_dir / "report.json", report)
    with (output_dir / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(render_report(report))
    return report


def render_report(report):
    def pct(value):
        return "n/a" if value is None else f"{value * 100:.2f}%"

    def count(value):
        return f"{value['passed']}/{value['evaluated']}"

    lines = [
        "# Finite-trace successor prediction: MLP vs JEPA",
        "",
        "Complete fixed-protocol run: 24 maps x 3 seeds x 5 budgets, 360 cases, "
        "720 from-scratch fits and 1,080 model-case records. All training data are "
        "finite real trace steps. This evaluates unseen transitions within each "
        "map, not transfer to previously unseen maps.",
        "",
        "Both models receive identical four-channel state images, action indices "
        "and observed steps (including duplicates). The MLP is flatten + 8-D "
        "action embedding → Linear(152,64) → GELU → Linear(64,|S|), with direct "
        "successor CE. JEPA reuses ActionJEPA (32-D latent, original CNN and "
        "fixed position subspace, 64-unit predictor), EMA 0.99, existing "
        "prediction/variance/covariance losses plus ranking weight 0.1. All "
        "known states can be candidates; only observed successors label examples.",
        "",
        "Both methods use AdamW, lr 0.0003, weight decay 0.0001, 300 full-batch "
        "epochs and final-epoch selection. No pretrained checkpoints, early "
        "stopping, oracle repair, heldout selection or hyperparameter search.",
        "",
        "Training pools exactly match the original sanity experiment. Each seed/map "
        "has 64 independently sampled held-out real 16-step traces, unique and "
        "disjoint from all 16 training-pool traces. The pool is fixed across budgets.",
        "",
        "## Interpretation of the baselines and denominators",
        "",
        "The observed-edge baseline is partial. It abstains on unseen queries; "
        "0% unseen accuracy means no correct unseen predictions, not a neural "
        "model selecting wrong successors. Its overall score is edge recall. "
        "Its raw held-out acceptance is action-sensitive membership in R_G.",
        "",
        "Only its explicitly named observed_self_loop_completion is sent to "
        "formal evaluation. Structural and CTL/LTL results for this baseline "
        "hold **under this explicit totalization policy**, not for raw R_G. "
        "Completion is evaluation-only, not learned inference.",
        "",
        "Transition accuracy counts all known S x A once, including unreachable "
        "states. Identity metrics also include unreachable states. Reachable and "
        "unreachable unseen subsets are reported separately. Macro values are "
        "unweighted means over cases; pooled counts and min/max/std are in JSON. "
        "Nested budgets, rotations and reused seeds are not independent trials.",
        "",
    ]
    for semantics in SEMANTICS:
        lines.extend(
            [
                f"## Main comparison: {semantics}",
                "",
                "| Budget | Model | Observed acc | Unseen acc | Overall acc | "
                "Held-out trace acceptance | Initial bisim | Identity bisim |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for b, models in report["by_budget"].items():
            for m, row in models.items():
                p = row["formal"]["primary_metrics"][semantics]
                accuracies = [
                    pct(row["transitions"][k]["macro"]["mean"])
                    for k in ("observed", "unseen", "overall")
                ]
                lines.append(
                    f"| {b} | {m} | "
                    + " | ".join(accuracies)
                    + f" | {pct(row['heldout']['acceptance']['macro']['mean'])} | "
                    + count(p["initial_bisimulation"])
                    + " | "
                    + count(p["identity_bisimulation"])
                    + " |"
                )
        lines.extend(
            [
                "",
                f"## All primary structural metrics: {semantics}",
                "",
                "| Budget | Model | Initial sim M → model | Initial sim model → M | "
                "Initial bisim | Identity sim M → model | "
                "Identity sim model → M | Identity bisim |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for b, models in report["by_budget"].items():
            for m, row in models.items():
                p = row["formal"]["primary_metrics"][semantics]
                metrics = (
                    "simulation_real_to_learned",
                    "simulation_learned_to_real",
                    "initial_bisimulation",
                    "identity_simulation_real_to_learned",
                    "identity_simulation_learned_to_real",
                    "identity_bisimulation",
                )
                lines.append(
                    f"| {b} | {m} | " + " | ".join(count(p[k]) for k in metrics) + " |"
                )
        lines.append("")
    lines.extend(
        [
            "## Additional CTL/LTL diagnostics",
            "",
            "The existing six CTL and six universal-path LTL formulas "
            "remain diagnostic. "
            "LTL verdicts are actually executed with nuXmv on saved predicted graphs; "
            "exact exported-model hashes cache identical queries. Every all-state "
            "AF goal/F goal and AG !danger/G !danger identity is audited.",
            "",
            "| Budget | Model | Initial CTL formula agreement | All-six CTL cases | "
            "Initial LTL formula agreement | All-six LTL cases | "
            "All-state CTL | All-state LTL |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for b, models in report["by_budget"].items():
        for m, r in models.items():
            d = r["formal"]["diagnostics"]
            ctl, ltl = d["all_state_ctl"], d["all_state_ltl"]
            lines.append(
                f"| {b} | {m} | "
                f"{pct(r['initial_ctl_formula_agreement']['macro']['mean'])} | "
                + count(d["initial_all_six_ctl_agreement"])
                + f" | {pct(r['initial_ltl_formula_agreement']['macro']['mean'])} | "
                + count(d["initial_all_six_ltl_agreement"])
                + f" | {pct(ctl['matched'] / ctl['comparisons'])} | "
                + f"{pct(ltl['matched'] / ltl['comparisons'])} |"
            )
    lines.extend(
        [
            "",
            "## Unseen accuracy paired comparison",
            "",
            "| Budget | JEPA minus MLP (percentage points) | "
            "JEPA wins | MLP wins | Ties |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for b, r in report["paired_unseen_comparison"].items():
        lines.append(
            f"| {b} | "
            f"{100 * r['jepa_minus_mlp_unseen_accuracy']['mean']:.2f} | "
            f"{r['jepa_wins']} | {r['mlp_wins']} | {r['ties']} |"
        )
    lines.extend(
        [
            "",
            "## Reachability and held-out difficulty",
            "",
            "| Budget | Model | Reachable unseen acc | Unreachable unseen acc | "
            "Held-out containing unseen pairs | Acceptance among those traces | "
            "Observed-unseen gap |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for b, models in report["by_budget"].items():
        for m, r in models.items():
            unseen = r["heldout"]["with_unseen_pairs"]
            lines.append(
                f"| {b} | {m} | "
                f"{pct(r['transitions']['reachable_unseen']['macro']['mean'])} | "
                f"{pct(r['transitions']['unreachable_unseen']['macro']['mean'])} | "
                f"{unseen['pooled']['total']}/4608 | {pct(unseen['macro']['mean'])} | "
                f"{100 * r['observed_minus_unseen_gap']['mean']:.2f} pp |"
            )
    for grouping in ("by_seed", "by_family"):
        lines.extend(
            [
                "",
                f"## {grouping}",
                "",
                "| Group | Budget | Model | Observed acc | Unseen acc | "
                "Overall acc | Held-out acceptance | "
                "Initial bisim (ignore / respect actions) | "
                "Identity bisim (ignore / respect actions) |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for group, budgets in report[grouping].items():
            for b, models in budgets.items():
                for m, r in models.items():
                    p = r["formal"]["primary_metrics"]
                    parts = [
                        pct(r["transitions"][k]["macro"]["mean"])
                        for k in ("observed", "unseen", "overall")
                    ]
                    lines.append(
                        f"| {group} | {b} | {m} | "
                        + " | ".join(parts)
                        + f" | {pct(r['heldout']['acceptance']['macro']['mean'])} | "
                        + " / ".join(
                            count(p[s]["initial_bisimulation"]) for s in SEMANTICS
                        )
                        + " | "
                        + " / ".join(
                            count(p[s]["identity_bisimulation"]) for s in SEMANTICS
                        )
                        + " |"
                    )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=src NUXMV_BINARY=/path/to/nuXmv \\",
            "  python -m experiments.finite_trace_mlp_vs_jepa \\",
            "  --output-dir outputs/finite_trace_mlp_vs_jepa/new_run",
            "```",
            "",
            "The destination must not exist. experiment_config.json and "
            "run_started.json "
            "lock the protocol, sources and prior result hashes. cases.jsonl records "
            "all predictions, training-data/parameter hashes, observed/unseen scores, "
            "held-out failure steps/edges and both structural semantics. "
            "trace_pools.jsonl "
            "saves exact training and held-out pools. ltl_backend.jsonl.gz saves all "
            "unique symbolic exports and actual backend output/verdicts. Predictions "
            "permit formal replay without retraining; model weights can be reproduced "
            "from the recorded config/seeds/source/runtime, with no checkpoint search.",
            "",
            "This is a comparison of these specified architectures and this fixed "
            "optimization budget. It does not establish superiority over every simple "
            "supervised predictor or identify an optimal learning rule.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    report = run_experiment(
        args.output_dir, config_path=args.config, executable=args.executable
    )
    print(json.dumps(report["paired_unseen_comparison"], indent=2))


if __name__ == "__main__":
    main()
