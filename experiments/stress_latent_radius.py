"""Evaluate three frozen checkpoints against the prespecified topology screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.benchmarks.random_gridworld import make_pilot_benchmark_splits
from jepa_lmc.evaluation.latent_radius import (
    collect_latent_distances,
    evaluate_candidates,
    oracle_radii,
)
from jepa_lmc.evaluation.radius_stress import (
    annotate_stress_result,
    evaluate_overall_gate,
    evaluate_seed_gate,
    score_nontrivial,
    summarize_stress_maps,
)
from jepa_lmc.learning.model import ActionJEPA

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/latent_radius_stress_protocol.json"


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_digest(model: ActionJEPA) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Expected nonempty diagnostic export.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_checkpoint(path: Path, protocol: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    training = payload["training"]
    recipe = protocol["training"]
    expected_kwargs = {"height": 6, "width": 6, "latent_dim": recipe["latent_dim"]}
    if payload["model_kwargs"] != expected_kwargs:
        raise ValueError("Checkpoint architecture differs from the fixed protocol.")
    for key in ("epochs", "batch_size", "learning_rate", "ema_momentum"):
        if training[key] != recipe[key]:
            raise ValueError(f"Checkpoint does not match fixed training {key}.")
    expected_specs = make_pilot_benchmark_splits().train[: recipe["train_maps"]]
    if training["train_map_seeds"] != [spec.seed for spec in expected_specs]:
        raise ValueError("Checkpoint training maps differ from the fixed protocol.")
    if training["seed"] not in protocol["model_seeds"]:
        raise ValueError("Unexpected model seed.")
    for case in radius_stress_cases():
        signature = asdict(case.spec)
        signature.pop("seed")
        for spec in expected_specs:
            train_signature = asdict(spec)
            train_signature.pop("seed")
            if signature == train_signature:
                raise ValueError("Stress layout overlaps a training map.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/latent_radius/stress")
    )
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    # Check all inputs before inference: no partial set of seeds yields a decision.
    payloads = [validate_checkpoint(path, protocol) for path in args.checkpoints]
    seeds = [payload["training"]["seed"] for payload in payloads]
    if len(seeds) != len(set(seeds)) or set(seeds) != set(protocol["model_seeds"]):
        raise ValueError(
            "Supply exactly one checkpoint for each of the three fixed seeds."
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Choose a fresh output directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cases = radius_stress_cases()
    assert len(cases) == protocol["maps"]["total"]
    source_paths = (
        "configs/latent_radius_stress_protocol.json",
        "experiments/stress_latent_radius.py",
        "src/jepa_lmc/benchmarks/radius_stress.py",
        "src/jepa_lmc/evaluation/radius_stress.py",
        "src/jepa_lmc/evaluation/latent_radius.py",
        "src/jepa_lmc/learning/model.py",
    )
    report = {
        "schema_version": 1,
        "protocol": protocol,
        "protocol_sha256": digest_file(PROTOCOL_PATH),
        "source_sha256": {path: digest_file(ROOT / path) for path in source_paths},
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "threads": 4,
        },
        "cases": [asdict(case) for case in cases],
        "seeds": [],
    }
    gates = {}
    all_errors, all_ctl, all_candidates = [], [], []
    for path, payload in sorted(
        zip(args.checkpoints, payloads, strict=True),
        key=lambda pair: pair[1]["training"]["seed"],
    ):
        seed = payload["training"]["seed"]
        print(f"Evaluating frozen seed {seed}", flush=True)
        model = ActionJEPA(**payload["model_kwargs"])
        model.load_state_dict(payload["model_state_dict"])
        before = model_digest(model)
        tables = [collect_latent_distances(model, case.make_env()) for case in cases]
        if model_digest(model) != before:
            raise AssertionError("Evaluation changed the frozen model.")
        radii = oracle_radii(tables)
        seed_result = {
            "seed": seed,
            "checkpoint_sha256": digest_file(path),
            "model_state_sha256": before,
            "model_kwargs": payload["model_kwargs"],
            "training": {
                k: v for k, v in payload["training"].items() if k != "history"
            },
            "radii": radii,
            "summary": {},
            "per_map": [],
        }
        for case, table in zip(cases, tables, strict=True):
            for row, (state, action) in enumerate(table.pairs):
                target = table.states[table.true_indices[row]]
                all_errors.append(
                    {
                        "seed": seed,
                        "case": case.name,
                        "family": case.family,
                        "state_row": state[0],
                        "state_col": state[1],
                        "action": action,
                        "true_next_row": target[0],
                        "true_next_col": target[1],
                        "error": float(table.errors[row]),
                    }
                )
        for variant in (
            "top1",
            "epsilon_95",
            "epsilon_99",
            "epsilon_max",
            "all_states",
            "exact_graph",
        ):
            map_results = []
            for case, table in zip(cases, tables, strict=True):
                if variant in ("top1", "exact_graph"):
                    mask = torch.zeros_like(table.distances, dtype=torch.bool)
                    indices = (
                        table.distances.argmin(dim=1)
                        if variant == "top1"
                        else torch.tensor(table.true_indices)
                    )
                    mask[torch.arange(len(table.pairs)), indices] = True
                elif variant == "all_states":
                    mask = torch.ones_like(table.distances, dtype=torch.bool)
                else:
                    mask = table.candidates(radii[variant])
                    for row, (state, action) in enumerate(table.pairs):
                        all_candidates.append(
                            {
                                "seed": seed,
                                "variant": variant,
                                "epsilon": radii[variant],
                                "case": case.name,
                                "state": state,
                                "action": action,
                                "candidates": [
                                    table.states[i]
                                    for i in mask[row].nonzero().flatten().tolist()
                                ],
                                "covered": bool(mask[row, table.true_indices[row]]),
                            }
                        )
                result = annotate_stress_result(
                    case, evaluate_candidates(case.make_env(), table, mask)
                )
                map_results.append(result)
                all_ctl.extend(
                    {"seed": seed, "variant": variant, **row}
                    for row in result["ctl_outcomes"]
                )
                if (
                    result["relation_inclusion"]
                    and result["ctl"]["one_sided_violations"]
                ):
                    raise AssertionError(
                        "One-sided CTL violation despite audited inclusion."
                    )
                if variant == "epsilon_max" and not result["action_inclusion"]:
                    raise AssertionError(
                        "Oracle maximum missed a true action successor."
                    )
                seed_result["per_map"].append(
                    {
                        "variant": variant,
                        "case": case.name,
                        "family": case.family,
                        **{
                            key: result[key]
                            for key in (
                                "pairs",
                                "successor_coverage",
                                "mean_candidate_size",
                                "singleton_fraction",
                                "empty_count",
                                "action_inclusion",
                                "relation_inclusion",
                                "missing_edges",
                                "extra_edges",
                                "ctl_status",
                            )
                        },
                        "nontrivial": score_nontrivial(result["ctl_outcomes"]),
                    }
                )
            summary = summarize_stress_maps(map_results)
            seed_result["summary"][variant] = summary
            nontrivial = summary["nontrivial"]
            print(
                f"  {variant}: coverage={summary['successor_coverage']:.2%}; "
                f"mean |C|={summary['mean_candidate_size']:.3f}; "
                f"balanced={nontrivial['primary_balanced_score']}; "
                f"proofs={nontrivial['recovered_opportunities']}/{nontrivial['transfer_opportunities']}",
                flush=True,
            )
        positive = seed_result["summary"]["exact_graph"]["nontrivial"]
        negative = seed_result["summary"]["all_states"]["nontrivial"]
        if (
            positive["primary_balanced_score"] != 1
            or positive["opportunity_recall"] != 1
        ):
            raise AssertionError("Exact-graph positive control failed.")
        if (
            negative["recovered_opportunities"] != 0
            or negative["primary_balanced_score"] != 0.5
        ):
            raise AssertionError("All-states negative control differs from protocol.")
        gates[seed] = evaluate_seed_gate(seed_result["summary"], protocol)
        seed_result["gate"] = gates[seed]
        report["seeds"].append(seed_result)
    report["decision"] = evaluate_overall_gate(gates, protocol)
    write_csv(args.output_dir / "errors.csv", all_errors)
    write_csv(args.output_dir / "ctl_outcomes.csv", all_ctl)
    candidate_text = "".join(
        json.dumps(row, allow_nan=False) + "\n" for row in all_candidates
    )
    (args.output_dir / "candidate_sets.jsonl").write_text(
        candidate_text, encoding="utf-8"
    )
    report["exports"] = {
        name: digest_file(args.output_dir / name)
        for name in ("errors.csv", "ctl_outcomes.csv", "candidate_sets.jsonl")
    }
    write_json(args.output_dir / "report.json", report)
    print(f"Decision: {report['decision']['status'].upper()}", flush=True)
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
