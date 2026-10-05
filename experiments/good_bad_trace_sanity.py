"""Known-state / hidden-transition problem-definition sanity check; no training.

Run with: python -m experiments.good_bad_trace_sanity --output-dir NEW_DIRECTORY
"""

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

from jepa_lmc.benchmarks.ctl_suite import default_ctl_suite
from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.evaluation.structural import (
    DEFINITIONS,
    SEMANTICS,
    evaluate_model_pair,
    summarize_model_pairs,
)
from jepa_lmc.evaluation.trace_sanity import (
    KnownProblem,
    make_hard_negatives,
    maximally_permissive,
    observed_edges,
    observed_self_loop_completion,
    relation_edges,
    sample_consistency,
    sample_good_traces,
    validate_trace_pairs,
)
from jepa_lmc.verification.gridworld import gridworld_to_transition_system

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260804, 20260805, 20260806)
BUDGETS = (1, 2, 4, 8, 16)
LENGTH = 16
POOL_SIZE = 16
VARIANTS = (
    "observed_self_loop_completion",
    "maximally_permissive",
    "real_reference",
)
POLICY = (
    "The observed-edge relation is partial.\n"
    "Self-loop completion is introduced only to obtain a total transition system\n"
    "compatible with the existing formal evaluation pipeline."
)
SOURCE_FILES = (
    "experiments/good_bad_trace_sanity.py",
    "src/jepa_lmc/evaluation/trace_sanity.py",
    "src/jepa_lmc/benchmarks/radius_stress.py",
    "src/jepa_lmc/envs/gridworld.py",
    "src/jepa_lmc/verification/gridworld.py",
    "src/jepa_lmc/verification/transition_system.py",
    "src/jepa_lmc/evaluation/structural.py",
    "src/jepa_lmc/verification/behavioral_relations.py",
    "src/jepa_lmc/benchmarks/ctl_suite.py",
    "src/jepa_lmc/verification/ctl.py",
    "src/jepa_lmc/verification/nuxmv.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stream_seed(seed: int, case: str, stream: str) -> int:
    """Stable independent per-map streams, not Python's randomized hash()."""
    value = f"good_bad_trace_sanity_v1:{seed}:{case}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def benchmark_catalogue():
    """Reuse the October generator and check its full seed-map identity set."""
    cases = radius_stress_cases()
    source = ROOT / "docs/results/structural_evaluation/ranking/report.json"
    october = json.loads(source.read_text(encoding="utf-8"))
    expected = {(s, c.name, c.family) for s in SEEDS for c in cases}
    actual = {(c["seed"], c["case"], c["family"]) for c in october["cases"]}
    if len(cases) != 24 or len(october["cases"]) != 72 or actual != expected:
        raise ValueError("Benchmark differs from the October 24-map, three-seed suite.")
    return cases, {"path": str(source.relative_to(ROOT)), "sha256": digest(source)}


def evaluate_total_model(real, model):
    """Only total models reach the existing, audited structural evaluator."""
    started = time.perf_counter()
    result = evaluate_model_pair(real, model)
    return {**result, "evaluation_seconds": time.perf_counter() - started}


def evaluate_budget(problem, real, goods, negatives, *, controls):
    """Sample metrics on partial R_G; formal metrics only on named total models."""
    relation = observed_edges(goods)
    truth = relation_edges(real)  # evaluator only, after constructing R_G
    common = len(relation & truth)
    bads = tuple(n.trace for n in negatives)
    partial = {
        **sample_consistency(problem, relation, goods, bads),
        "transition_precision": common / len(relation),
        "transition_recall": common / len(truth),
        "true_positive_transitions": common,
        "false_positive_transitions": len(relation - truth),
    }
    completed = observed_self_loop_completion(problem, relation)
    complete_edges = relation_edges(maximally_permissive(problem))
    complete = sample_consistency(problem, complete_edges, goods, bads)
    return {
        "trajectory_budget": len(goods),
        "observed_steps": sum(len(t.actions) for t in goods),
        "unique_observed_state_action_pairs": len({(s, a) for s, a, _ in relation}),
        "unique_observed_transitions": len(relation),
        "real_transition_count": len(truth),
        "transition_coverage": len(relation) / len(truth),
        "observed_edges": sorted(relation),
        "same_label_negatives": sum(n.same_proposition_labels for n in negatives),
        "observed_edge": partial,
        "maximally_permissive_samples": complete,
        "formal": {
            "observed_self_loop_completion": evaluate_total_model(real, completed),
            **controls,
        },
        "expected_trivial_separator": (
            partial["good_acceptance"] == 1
            and partial["bad_rejection"] == 1
            and partial["transition_precision"] == 1
            and partial["transition_recall"] < 1
        ),
    }


def distribution(values):
    values = list(values)
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}


def aggregate(cases):
    cases = list(cases)
    formal = {}
    for variant in VARIANTS:
        summary = summarize_model_pairs(c["formal"][variant] for c in cases)
        # Full mismatch identities remain in cases.jsonl; don't repeat them in
        # every budget/seed/family aggregate.
        summary.pop("mismatch_cases")
        formal[variant] = summary
    return {
        "cases": len(cases),
        **{
            field: distribution(c[field] for c in cases)
            for field in (
                "observed_steps",
                "unique_observed_state_action_pairs",
                "unique_observed_transitions",
                "real_transition_count",
                "transition_coverage",
            )
        },
        "observed_edge": {
            field: distribution(c["observed_edge"][field] for c in cases)
            for field in (
                "good_acceptance",
                "bad_rejection",
                "transition_precision",
                "transition_recall",
            )
        },
        "maximally_permissive_samples": {
            field: distribution(c["maximally_permissive_samples"][field] for c in cases)
            for field in ("good_acceptance", "bad_rejection")
        },
        "formal": formal,
        "expected_trivial_separator_cases": sum(
            c["expected_trivial_separator"] for c in cases
        ),
        "same_label_negatives": sum(c["same_label_negatives"] for c in cases),
        "negative_count": sum(c["trajectory_budget"] for c in cases),
    }


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, records):
    with Path(path).open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
            )


def run_experiment(output_dir, *, progress=True):
    """Fixed protocol; refuse to overwrite any pre-existing results directory."""
    maps, october_source = benchmark_catalogue()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    records, pools, manifest = [], [], []
    calls = 0
    control_seconds = 0.0
    for case in maps:
        env = case.make_env()
        real = gridworld_to_transition_system(env)
        problem = KnownProblem(
            tuple(sorted(real.states)),
            real.initial_states,
            tuple(sorted(env.ACTIONS)),
            {s: real.propositions(s) for s in real.states},
        )
        controls = {
            "maximally_permissive": evaluate_total_model(
                real, maximally_permissive(problem)
            ),
            "real_reference": evaluate_total_model(real, real),
        }
        calls += 2
        control_seconds += sum(c["evaluation_seconds"] for c in controls.values())
        reachable, pending = set(real.initial_states), list(real.initial_states)
        while pending:
            for successor in real.successors(pending.pop()):
                if successor not in reachable:
                    reachable.add(successor)
                    pending.append(successor)
        manifest.append(
            {
                "case": case.name,
                "family": case.family,
                "layout": case.layout,
                "rotation": case.rotation,
                "states": problem.states,
                "initial": env.start,
                "actions": problem.actions,
                "labels": [sorted(problem.labels[s]) for s in problem.states],
                "walls": sorted(env.walls),
                "goal": env.goal,
                "dangers": sorted(env.dangers),
                "real_edges": sorted(relation_edges(real)),
                "reachable_states": sorted(reachable),
                "reachable_transition_count": len(reachable) * len(problem.actions),
            }
        )
        for seed in SEEDS:
            sampling_seed = stream_seed(seed, case.name, "good")
            corruption_seed = stream_seed(seed, case.name, "bad")
            goods = sample_good_traces(
                env, seed=sampling_seed, count=POOL_SIZE, length=LENGTH
            )
            negatives = make_hard_negatives(
                env, goods, problem.labels, seed=corruption_seed
            )
            validate_trace_pairs(env, goods, negatives)
            pools.append(
                {
                    "seed": seed,
                    "case": case.name,
                    "family": case.family,
                    "sampling_rng_seed": sampling_seed,
                    "corruption_rng_seed": corruption_seed,
                    "goods": [asdict(t) for t in goods],
                    "negatives": [asdict(n) for n in negatives],
                    "validated": True,
                }
            )
            previous = frozenset()
            for budget in BUDGETS:
                subset = goods[:budget]
                relation = observed_edges(subset)
                if not previous <= relation:
                    raise AssertionError("Nested observed relations must be monotone.")
                previous = relation
                result = evaluate_budget(
                    problem, real, subset, negatives[:budget], controls=controls
                )
                records.append(
                    {"seed": seed, "case": case.name, "family": case.family, **result}
                )
                calls += 1
        if progress:
            print(f"{case.name}: {len(records)}/360 cases", flush=True)
    write_jsonl(output_dir / "cases.jsonl", records)
    write_jsonl(output_dir / "trace_pools.jsonl", pools)
    write_json(output_dir / "benchmark.json", manifest)
    by_budget = {
        str(b): aggregate(c for c in records if c["trajectory_budget"] == b)
        for b in BUDGETS
    }
    report = {
        "schema_version": 1,
        "experiment": "good_bad_trace_sanity",
        "training": False,
        "protocol": {
            "sampling_seeds": SEEDS,
            "map_count": len(maps),
            "trajectory_length_in_transitions": LENGTH,
            "trace_pool_size_per_seed_map": POOL_SIZE,
            "nested_trajectory_budgets": BUDGETS,
            "cases_per_budget": 72,
            "case_count": len(records),
            "bad_traces_per_good": 1,
            "sampling": (
                "Uniform random actions; each trace starts at the specified initial."
            ),
            "rng": (
                "random.Random; SHA256-derived independent (seed, map name, good/bad) "
                "streams; first 8 bytes, big endian."
            ),
            "labels": (
                "Full existing verification AP sets; goal has both safe and goal."
            ),
            "bad_trace_policy": (
                "One false successor, same AP set when possible, then replay "
                "remaining actions using true T."
            ),
            "coverage_denominator": (
                "All action-labelled triples in R, including unreachable states "
                "and self-loops."
            ),
            "aggregation": (
                "Unweighted macro mean over seed-map cases; seeds/budgets/rotations "
                "are not independent replicates."
            ),
            "totalization_policy": POLICY,
            "formal_semantics": DEFINITIONS,
            "ctl_formulas": [p.name for p in default_ctl_suite()],
            "ltl": "not run",
            "learner_inputs": (
                "S, I, A, L and good traces; no true R, no oracle queries, "
                "no bad-trace fitting."
            ),
        },
        "provenance": {
            "base_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "october_source": october_source,
            "october_seed_map_catalogue_verified": True,
            "source_sha256": {name: digest(ROOT / name) for name in SOURCE_FILES},
        },
        "validation": {
            "good_execution_checks": POOL_SIZE * len(pools),
            "bad_nonexecution_and_exactly_one_false_edge_checks": POOL_SIZE
            * len(pools),
            "same_label_negatives_in_unique_pools": sum(
                n["same_proposition_labels"]
                for pool in pools
                for n in pool["negatives"]
            ),
            "all_pools_validated": all(p["validated"] for p in pools),
            "nested_budgets_verified": True,
            "expected_trivial_separator_cases": sum(
                c["expected_trivial_separator"] for c in records
            ),
            "all_cases_have_expected_trivial_separator": all(
                c["expected_trivial_separator"] for c in records
            ),
            "all_complete_good_accepted_and_bad_accepted": all(
                c["maximally_permissive_samples"]["good_acceptance"] == 1
                and c["maximally_permissive_samples"]["bad_rejection"] == 0
                for c in records
            ),
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "evaluate_model_pair_calls": calls,
            "greatest_relation_audits": calls * 6,
            "independent_bisimulation_partition_checks": calls * 2,
            "control_evaluation_seconds": control_seconds,
            "control_cache": (
                "Complete and real-reference evaluations computed once per map, "
                "reused across seeds and budgets; not repeated measurements."
            ),
        },
        "by_budget": by_budget,
        "by_seed": {
            str(s): {
                str(b): aggregate(
                    c for c in records if c["seed"] == s and c["trajectory_budget"] == b
                )
                for b in BUDGETS
            }
            for s in SEEDS
        },
        "by_family": {
            f: {
                str(b): aggregate(
                    c
                    for c in records
                    if c["family"] == f and c["trajectory_budget"] == b
                )
                for b in BUDGETS
            }
            for f in sorted({c.family for c in maps})
        },
        "exports": {
            name: digest(output_dir / name)
            for name in ("cases.jsonl", "trace_pools.jsonl", "benchmark.json")
        },
    }
    write_json(output_dir / "report.json", report)
    with (output_dir / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    return report


def render_markdown(report):
    def pct(value):
        return f"{100 * value:.2f}%"

    def count(value):
        return f"{value['passed']}/{value['evaluated']}"

    lines = [
        "# Good / bad trace formulation: sanity experiment",
        "",
        "Known S, I, A, L; hidden R; visible state identities. "
        "No JEPA training or inference.",
        "",
        "The fixed October benchmark has 24 maps, seeds 20260804/20260805/20260806, "
        "16-step trajectories and nested budgets 1/2/4/8/16. Each budget has 72 cases.",
        "",
        "## Partial-relation sample results",
        "",
        "Coverage and recall use the entire action-labelled R, including unreachable "
        "states. Means are unweighted over the 72 seed-map cases. Trajectory counts "
        "are observation budgets, not transition coverage. Bisimulation columns below "
        "refer ONLY to the separately completed total model.",
        "",
    ]
    for semantics in SEMANTICS:
        lines.extend(
            [
                f"### {semantics}",
                "",
                "| Trajectory budget | Mean transition coverage | "
                "Observed-edge good acceptance | Observed-edge bad rejection | "
                "Observed-edge transition recall | "
                "Self-loop completion initial bisimulation | "
                "Complete-model bad rejection |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for budget, row in report["by_budget"].items():
            obs = row["observed_edge"]
            bisim = row["formal"][VARIANTS[0]]["primary_metrics"][semantics][
                "initial_bisimulation"
            ]
            lines.append(
                f"| {budget} | {pct(row['transition_coverage']['mean'])} | "
                f"{pct(obs['good_acceptance']['mean'])} | "
                f"{pct(obs['bad_rejection']['mean'])} | "
                f"{pct(obs['transition_recall']['mean'])} | "
                f"{count(bisim)} ({pct(bisim['fraction'])}) | "
                f"{pct(row['maximally_permissive_samples']['bad_rejection']['mean'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "| Budget | Observed-edge precision min/max | "
            "Complete-model good acceptance min/max |",
            "|---:|---:|---:|",
        ]
    )
    for b, r in report["by_budget"].items():
        precision = r["observed_edge"]["transition_precision"]
        acceptance = r["maximally_permissive_samples"]["good_acceptance"]
        lines.append(
            f"| {b} | {pct(precision['min'])} / {pct(precision['max'])} | "
            f"{pct(acceptance['min'])} / {pct(acceptance['max'])} |"
        )
    lines.extend(
        [
            "",
            "| Budget | Observed steps per case | Mean unique (s,a) | "
            "Mean unique triples | Recall min-max |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for b, r in report["by_budget"].items():
        recall = r["observed_edge"]["transition_recall"]
        lines.append(
            f"| {b} | {r['observed_steps']['mean']:.0f} | "
            f"{r['unique_observed_state_action_pairs']['mean']:.2f} | "
            f"{r['unique_observed_transitions']['mean']:.2f} | "
            f"{pct(recall['min'])}-{pct(recall['max'])} |"
        )
    lines.extend(
        [
            "",
            "## Formal evaluation of total models",
            "",
            POLICY,
            "",
            "For observed (s,a), retain its observed successor; for every "
            "unobserved (s,a), insert (s,a,s). All conclusions in this section "
            "hold **under this explicit totalization policy**. No CTL or "
            "bisimulation claim is assigned to partial R_G.",
            "",
            "The unchanged evaluate_model_pair calls the existing behavioral_relations "
            "fixed points, audits the greatest relations and cross-checks bisimulation "
            "with existing partition refinement. CTL uses the unchanged six-formula "
            "native diagnostic. LTL is not run.",
            "",
        ]
    )
    for semantics in SEMANTICS:
        lines.extend(
            [
                f"### {semantics}",
                "",
                "| Model | Budget | Initial sim M → model | "
                "Initial sim model → M | Initial bisimulation | "
                "Identity bisimulation |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for variant in VARIANTS:
            for b, r in report["by_budget"].items():
                if variant != VARIANTS[0] and b != "1":
                    continue
                p = r["formal"][variant]["primary_metrics"][semantics]
                metrics = (
                    "simulation_real_to_learned",
                    "simulation_learned_to_real",
                    "initial_bisimulation",
                    "identity_bisimulation",
                )
                label = b if variant == VARIANTS[0] else "all"
                lines.append(
                    f"| {variant} | {label} | "
                    + " | ".join(count(p[m]) for m in metrics)
                    + " |"
                )
        lines.append("")
    lines.extend(
        [
            "## Six-formula CTL diagnostic",
            "",
            "Fractions below mean all six initial verdicts agree with M. Full "
            "per-formula verdicts, all-state counts and structural mismatch "
            "states are in cases.jsonl.",
            "",
            "| Budget | Self-loop completion | Maximally permissive | Real reference |",
            "|---:|---:|---:|---:|",
        ]
    )
    for b, r in report["by_budget"].items():
        lines.append(
            f"| {b} | "
            + " | ".join(
                count(r["formal"][v]["diagnostics"]["initial_all_six_ctl_agreement"])
                for v in VARIANTS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Seed and family breakdown",
            "",
            "| Seed | Budget | Mean transition recall | "
            "Initial bisimulation (ignore / respect actions) |",
            "|---:|---:|---:|---:|",
        ]
    )
    for seed, rows in report["by_seed"].items():
        for b, r in rows.items():
            p = r["formal"][VARIANTS[0]]["primary_metrics"]
            lines.append(
                f"| {seed} | {b} | "
                f"{pct(r['observed_edge']['transition_recall']['mean'])} | "
                + " / ".join(count(p[s]["initial_bisimulation"]) for s in SEMANTICS)
                + " |"
            )
    lines.extend(
        [
            "",
            "| Family | Budget | Mean transition recall | "
            "Initial bisimulation (ignore / respect actions) |",
            "|---|---:|---:|---:|",
        ]
    )
    for family, rows in report["by_family"].items():
        for b, r in rows.items():
            p = r["formal"][VARIANTS[0]]["primary_metrics"]
            lines.append(
                f"| {family} | {b} | "
                f"{pct(r['observed_edge']['transition_recall']['mean'])} | "
                + " / ".join(count(p[s]["initial_bisimulation"]) for s in SEMANTICS)
                + " |"
            )
    lines.extend(
        [
            "",
            "## What this establishes",
            "",
            "Expected perfect separation with incomplete R recovery: "
            f"{report['validation']['expected_trivial_separator_cases']}/"
            f"{report['protocol']['case_count']} cases.",
            "",
            "This is also a direct mathematical consequence: every good execution uses "
            "only edges in R, hence R_G is a subset of R. Every good trace is accepted "
            "by construction. Every R-invalid negative with the designated initial "
            "contains an edge outside R and therefore outside R_G. No bad traces need "
            "to be consulted to fit this separator. The result is stronger than merely "
            "rejecting the sampled negatives.",
            "",
            "R_G is minimal by edge inclusion for accepting these identified good "
            "traces; it can still recombine observed edges into additional traces. It "
            "does not reconstruct unseen transitions. This uses partial hypotheses. "
            "If totality is required of the learner, R_G is not itself an admissible "
            "total solution, and completion may invalidate sample separation. We make "
            "no such claim about the completed model.",
            "",
            "In sealed_region, the initial reachable component has only safe labels; "
            "both real and completed systems are action-total on a safe component. "
            "They can be initially bisimilar despite extensive missing edges. "
            "Bisimulation is observational equivalence, not equality of R. Identity "
            "bisimulation in the existing evaluator includes unreachable states and "
            "does not mean strict edge equality either.",
            "",
            "The complete model illustrates good-only over-generalization: it accepts "
            "all syntactically valid negatives, including the same-label corruptions.",
            "",
            "For Luca's formulation, finite good/bad separation alone does not define "
            "transition reconstruction in the known-state setting. A reconstruction "
            "objective needs requirements beyond sample consistency, such as held-out "
            "real transition/trace recovery or explicit structural comparison, and a "
            "clearly stated partial/total hypothesis class and inductive assumptions. "
            "This does not rule out useful learning under additional assumptions, and "
            "the fixed rotated maps and nested samples are not independent "
            "statistical trials.",
            "",
            "## Reproduce and inspect",
            "",
            "```bash",
            "PYTHONPATH=src python -m experiments.good_bad_trace_sanity \\",
            "  --output-dir outputs/good_bad_trace_sanity/new_run",
            "```",
            "",
            "The output directory must not exist. Files: report.json (aggregates, "
            "provenance, hashes), cases.jsonl (360 full records), trace_pools.jsonl "
            "(72 paired pools with corrupted-step metadata), benchmark.json (24 exact "
            "reference graphs) and report.md. Controls are evaluated once per map and "
            "reused, with the cache and timing recorded explicitly.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_experiment(args.output_dir)
    print(json.dumps(report["validation"], indent=2))


if __name__ == "__main__":
    main()
