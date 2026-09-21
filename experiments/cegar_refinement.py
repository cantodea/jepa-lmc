"""Compare JEPA-guided successor refinement with uninformed CEGAR and exact BFS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from statistics import mean, median

import torch

from jepa_lmc.benchmarks.ctl_suite import default_ctl_suite
from jepa_lmc.benchmarks.radius_stress import PRIMARY_PROPERTIES, radius_stress_cases
from jepa_lmc.benchmarks.random_gridworld import make_pilot_benchmark_splits
from jepa_lmc.evaluation.refinement_priority import (
    latent_rank_costs,
    shuffled_rank_costs,
)
from jepa_lmc.learning.data import gridworld_observation
from jepa_lmc.learning.model import ActionJEPA
from jepa_lmc.verification.ctl import CTLModelChecker
from jepa_lmc.verification.gridworld import gridworld_to_transition_system
from jepa_lmc.verification.successor_refinement import (
    RefinementResult,
    SuccessorRefinement,
    task_from_labels,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/cegar_protocol.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_digest(model: ActionJEPA) -> str:
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def json_default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_checkpoint(path: Path, protocol: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["model_kwargs"] != {"height": 6, "width": 6, "latent_dim": 32}:
        raise ValueError("Use the unchanged pilot architecture/checkpoints.")
    training = payload["training"]
    expected = {
        "epochs": 100,
        "batch_size": 512,
        "learning_rate": 0.0003,
        "ema_momentum": 0.99,
    }
    if any(training[k] != v for k, v in expected.items()):
        raise ValueError("Checkpoint training recipe differs from the protocol.")
    if training["seed"] not in protocol["model_seeds"]:
        raise ValueError("Unexpected model seed.")
    maps = make_pilot_benchmark_splits().train[:40]
    if training["train_map_seeds"] != [spec.seed for spec in maps]:
        raise ValueError("Checkpoint training map identities differ.")
    return payload


def audit_result(refiner, result: RefinementResult, task, exact: dict) -> None:
    """Independent concrete audit, called only after a verification task returns."""
    observed = {}
    reachable_prefix = {task.start}
    for row in result.observations:
        pair = (row.state, row.action)
        if pair in observed or row.successor != exact[pair]:
            raise AssertionError("Duplicate query or incorrect oracle observation.")
        if row.state not in reachable_prefix:
            raise AssertionError("A query was not reached by a confirmed prefix.")
        observed[pair] = row.successor
        reachable_prefix.add(row.successor)
        # Every unqueried pair still admits all states; each queried pair is exact.
        if any(target != exact[pair] for pair, target in observed.items()):
            raise AssertionError("An intermediate refinement lost inclusion.")
    if refiner.known != observed or result.queries != len(observed):
        raise AssertionError("Oracle accounting or final relation mismatch.")
    if any(target not in refiner.candidates(*pair) for pair, target in exact.items()):
        raise AssertionError("The final upper relation omits a real edge.")
    if result.value is None:
        return
    if result.value == task.path_value:
        if result.witness is None:
            raise AssertionError("A path verdict needs a concrete witness.")
        current = task.start
        for source, action, target in result.witness:
            if source != current or source not in task.allowed:
                raise AssertionError("Invalid witness prefix.")
            if observed.get((source, action)) != target:
                raise AssertionError("Unconfirmed edge in a returned witness.")
            current = target
        if current not in task.targets:
            raise AssertionError("Witness does not reach its target.")
    else:
        region = result.closed_region
        if region is None or region & task.targets:
            raise AssertionError("Invalid no-path certificate.")
        if task.start in task.allowed and task.start not in region:
            raise AssertionError("The certificate omits the start state.")
        permitted = task.allowed | task.targets
        for state in region:
            for action in range(refiner.num_actions):
                if not (refiner.candidates(state, action) & permitted) <= region:
                    raise AssertionError("The upper region is not closed.")


def summarize(rows: list[dict], fractions: list[float]) -> dict:
    if not rows:
        raise ValueError("Cannot score an empty experiment group.")
    return {
        "tasks": len(rows),
        "wrong_conclusive_verdicts": sum(
            r["value"] is not None and r["value"] != r["ground_truth"] for r in rows
        ),
        "unknown": sum(r["value"] is None for r in rows),
        "total_oracle_queries": sum(r["oracle_queries"] for r in rows),
        "mean_oracle_queries": mean(r["oracle_queries"] for r in rows),
        "median_oracle_queries": median(r["oracle_queries"] for r in rows),
        "mean_query_fraction": mean(r["query_fraction"] for r in rows),
        "mean_upper_candidates": mean(r["mean_upper_candidates"] for r in rows),
        "spurious_paths": sum(r["spurious_paths"] for r in rows),
        "verification_seconds": sum(r["verification_seconds"] for r in rows),
        "resolved_at_budget": {
            str(f): sum(
                r["value"] is not None
                and r["oracle_queries"] <= math.floor(f * r["full_table_queries"])
                for r in rows
            )
            / len(rows)
            for f in fractions
        },
    }


def assess(seed_summaries: dict, protocol: dict) -> dict:
    required = protocol["exploratory_support_rule"][
        "every_seed_reduction_vs_uniform_and_direct_bfs"
    ]
    per_seed = {}
    for seed, methods in seed_summaries.items():
        jepa = methods["cegar_jepa"]["total_oracle_queries"]
        reductions = {
            baseline: 1 - jepa / methods[baseline]["total_oracle_queries"]
            for baseline in ("cegar_uniform", "direct_bfs", "cegar_shuffled")
        }
        per_seed[seed] = {
            "relative_query_reductions": reductions,
            "passes_query_screen": all(
                jepa
                <= (1 - Fraction(str(required))) * methods[b]["total_oracle_queries"]
                for b in ("cegar_uniform", "direct_bfs")
            )
            and reductions["cegar_shuffled"] > 0,
        }
    correct = all(
        m["wrong_conclusive_verdicts"] == 0 and m["unknown"] == 0
        for methods in seed_summaries.values()
        for m in methods.values()
    )
    supported = correct and all(s["passes_query_screen"] for s in per_seed.values())
    return {
        "all_conclusive_verdicts_correct_and_full_runs_resolved": correct,
        "all_intermediate_and_final_inclusion_audits_pass": True,
        "per_seed": per_seed,
        "status": "support_query_efficiency_followup"
        if supported
        else "mechanism_valid_jepa_query_benefit_not_established",
        "scope": "Finite exact successor oracle; no unknown-world guarantee",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/refinement/cegar")
    )
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    payloads = [load_checkpoint(path, protocol) for path in args.checkpoints]
    seeds = [p["training"]["seed"] for p in payloads]
    if len(seeds) != len(set(seeds)) or set(seeds) != set(protocol["model_seeds"]):
        raise ValueError("Supply exactly the three fixed checkpoints.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Choose a fresh output directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    sources = [
        "configs/cegar_protocol.json",
        "experiments/cegar_refinement.py",
        "src/jepa_lmc/verification/successor_refinement.py",
        "src/jepa_lmc/evaluation/refinement_priority.py",
        "src/jepa_lmc/benchmarks/radius_stress.py",
        "src/jepa_lmc/learning/model.py",
    ]
    report = {
        "protocol": protocol,
        "protocol_sha256": digest(PROTOCOL),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {p: digest(ROOT / p) for p in sources},
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "threads": 4,
        },
        "seeds": [],
    }
    all_rows, traces = [], []
    seed_summaries = {}
    formulas = {s.name: s.formula for s in default_ctl_suite()}
    fractions = protocol["budget_fractions_of_full_state_action_table"]
    for path, payload in sorted(
        zip(args.checkpoints, payloads, strict=True),
        key=lambda pair: pair[1]["training"]["seed"],
    ):
        seed = payload["training"]["seed"]
        print(f"Frozen JEPA seed {seed}", flush=True)
        model = ActionJEPA(**payload["model_kwargs"])
        model.load_state_dict(payload["model_state_dict"])
        model_before = model_digest(model)
        preparation_seconds = 0.0
        for case_index, case in enumerate(radius_stress_cases()):
            env = case.make_env()
            states = tuple(env.all_states())
            indices = {s: i for i, s in enumerate(states)}
            labels = tuple(frozenset((env.label(s),)) for s in states)
            prepare_start = time.perf_counter()
            observations = torch.stack([gridworld_observation(env, s) for s in states])
            costs = latent_rank_costs(model, observations)
            shuffled = shuffled_rank_costs(costs, seed * 1000 + case_index)
            preparation_seconds += time.perf_counter() - prepare_start
            pending = []
            for method in protocol["methods"]:
                for name in PRIMARY_PROPERTIES:
                    calls = []

                    def oracle(state, action):
                        calls.append((state, action))
                        return indices[env.transition(states[state], action)]

                    priorities = (
                        costs
                        if method == "cegar_jepa"
                        else (shuffled if method == "cegar_shuffled" else None)
                    )
                    refiner = SuccessorRefinement(
                        len(states), 4, oracle, costs=priorities
                    )
                    task = task_from_labels(name, indices[env.start], labels)
                    started = time.perf_counter()
                    result = (
                        refiner.solve_direct_bfs(task)
                        if method == "direct_bfs"
                        else refiner.solve(task)
                    )
                    elapsed = time.perf_counter() - started
                    if len(calls) != result.queries or len(set(calls)) != len(calls):
                        raise AssertionError(
                            "Unaccounted or duplicate concrete queries."
                        )
                    pending.append((method, name, refiner, task, result, elapsed))
            # Exact dynamics and CTL references are built only after the tasks return.
            exact = {
                (s, a): indices[env.transition(states[s], a)]
                for s in range(len(states))
                for a in range(4)
            }
            checker = CTLModelChecker(gridworld_to_transition_system(env))
            for method, name, refiner, task, result, elapsed in pending:
                audit_result(refiner, result, task, exact)
                truth = checker.holds(env.start, formulas[name])
                if result.value is None or result.value != truth:
                    raise AssertionError(
                        "A full-budget run failed to give exact truth."
                    )
                row = {
                    "seed": seed,
                    "case": case.name,
                    "family": case.family,
                    "method": method,
                    "property": name,
                    "ground_truth": truth,
                    "value": result.value,
                    "reason": result.reason,
                    "oracle_queries": result.queries,
                    "full_table_queries": len(states) * 4,
                    "query_fraction": result.queries / (len(states) * 4),
                    "rounds": result.rounds,
                    "spurious_paths": result.spurious_paths,
                    "mean_upper_candidates": result.mean_upper_candidates,
                    "verification_seconds": elapsed,
                }
                all_rows.append(row)
                traces.append(
                    {
                        "seed": seed,
                        "case": case.name,
                        "method": method,
                        "property": name,
                        "states": states,
                        "task": asdict(task),
                        "result": asdict(result),
                    }
                )
        if model_before != model_digest(model):
            raise AssertionError("Refinement evaluation changed the frozen model.")
        seed_rows = [r for r in all_rows if r["seed"] == seed]
        summaries = {
            method: summarize(
                [r for r in seed_rows if r["method"] == method], fractions
            )
            for method in protocol["methods"]
        }
        seed_summaries[seed] = summaries
        report["seeds"].append(
            {
                "seed": seed,
                "checkpoint_sha256": digest(path),
                "model_state_sha256": model_before,
                "neural_preparation_seconds": preparation_seconds,
                "summary": summaries,
                "by_property": {
                    name: {
                        method: summarize(
                            [
                                r
                                for r in seed_rows
                                if r["method"] == method and r["property"] == name
                            ],
                            fractions,
                        )
                        for method in protocol["methods"]
                    }
                    for name in PRIMARY_PROPERTIES
                },
                "by_family": {
                    family: {
                        method: summarize(
                            [
                                r
                                for r in seed_rows
                                if r["method"] == method and r["family"] == family
                            ],
                            fractions,
                        )
                        for method in protocol["methods"]
                    }
                    for family in sorted({r["family"] for r in seed_rows})
                },
            }
        )
        for method, score in summaries.items():
            print(
                f"  {method}: mean queries={score['mean_oracle_queries']:.2f}, "
                f"resolved at 25%={score['resolved_at_budget']['0.25']:.1%}, "
                f"time={score['verification_seconds']:.2f}s",
                flush=True,
            )
    report["assessment"] = assess(seed_summaries, protocol)
    with (args.output_dir / "tasks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output_dir / "traces.jsonl").write_text(
        "".join(
            json.dumps(t, default=json_default, allow_nan=False) + "\n" for t in traces
        ),
        encoding="utf-8",
    )
    report["exports"] = {
        p: digest(args.output_dir / p) for p in ("tasks.csv", "traces.jsonl")
    }
    report["task_count"] = len(all_rows)
    write_json(args.output_dir / "report.json", report)
    print(report["assessment"]["status"], flush=True)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
