"""Exhaustive, split-aware diagnostics for the bounded Top-1 quality study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.benchmarks.random_gridworld import make_pilot_benchmark_splits
from jepa_lmc.evaluation.dynamics import evaluate_gridworld_transitions
from jepa_lmc.learning.data import gridworld_observation
from jepa_lmc.learning.model import ActionJEPA

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260804, 20260805, 20260806)
BASELINES = {
    20260804: ROOT / "outputs/latent_radius/oracle/model.pt",
    20260805: ROOT / "outputs/latent_radius/stress_training/seed_20260805/model.pt",
    20260806: ROOT / "outputs/latent_radius/stress_training/seed_20260806/model.pt",
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Refusing to overwrite existing results: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def rotate_spec(spec, turns):
    if spec.width != spec.height:
        raise ValueError("Rotation diagnostics require square maps.")

    def rotate(state):
        row, col = state
        for _ in range(turns % 4):
            row, col = col, spec.height - 1 - row
        return row, col

    return replace(
        spec,
        start=rotate(spec.start),
        goal=rotate(spec.goal),
        walls=frozenset(rotate(s) for s in spec.walls),
        dangers=frozenset(rotate(s) for s in spec.dangers),
    )


def map_fingerprint(spec):
    # The observation does not encode start, so exclude it from leakage checks.
    return (
        spec.height,
        spec.width,
        spec.goal,
        tuple(sorted(spec.walls)),
        tuple(sorted(spec.dangers)),
    )


def catalogue():
    splits = make_pilot_benchmark_splits()
    result = []
    for split, specs, rotations in (
        ("train", splits.train[:40], (0,)),
        ("validation", splits.validation, (0,)),
        ("validation_rotated", splits.validation, (1, 2, 3)),
    ):
        for spec in specs:
            for turn in rotations:
                result.append(
                    (
                        {
                            "split": split,
                            "map": f"random_{spec.seed}_rot{turn * 90}",
                            "map_seed": spec.seed,
                            "family": "random",
                            "rotation": turn * 90,
                        },
                        rotate_spec(spec, turn),
                    )
                )
    for case in radius_stress_cases():
        result.append(
            (
                {
                    "split": "stress",
                    "map": case.name,
                    "map_seed": case.spec.seed,
                    "family": case.family,
                    "rotation": case.rotation * 90,
                },
                case.spec,
            )
        )
    return result


def split_audit():
    splits = make_pilot_benchmark_splits()
    training = [rotate_spec(s, r) for s in splits.train[:40] for r in range(4)]
    evaluation = [s for m, s in catalogue() if m["split"] != "train"]
    untouched_test = list(splits.test)
    overlap = set(map(map_fingerprint, training)) & set(
        map(map_fingerprint, evaluation + untouched_test)
    )
    if overlap:
        raise AssertionError("Training orbit overlaps evaluation observations.")
    return {
        "base_train_maps": 40,
        "original_validation_maps": 20,
        "rotated_validation_maps": 60,
        "stress_maps": 24,
        "untouched_pilot_test_maps": len(untouched_test),
        "train_rotation_orbit_maps": len(training),
        "training_orbit_vs_evaluation_map_overlap": len(overlap),
        "train_map_seeds": [s.seed for s in splits.train[:40]],
        "validation_map_seeds": [s.seed for s in splits.validation],
        "stress_used_for_training_or_model_selection": False,
        "note": "Stress is an already-inspected diagnostic suite, not a fresh test.",
    }


def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = ActionJEPA(**payload["model_kwargs"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


@torch.no_grad()
def inspect_map(model, spec, metadata, seed, detailed=True):
    env = spec.make_env()
    states = tuple(env.all_states())
    index = {s: i for i, s in enumerate(states)}
    report = evaluate_gridworld_transitions(model, env)
    if not detailed:
        return report
    observations = torch.stack([gridworld_observation(env, s) for s in states])
    target = model.target_encoder(observations)
    context = model.context_encoder(observations)
    source = torch.arange(len(states)).repeat_interleave(4)
    action = torch.tensor(list(env.ACTIONS) * len(states))
    prediction = model.predictor(context[source], action)
    # Independent arithmetic: float64 direct subtraction, no cdist/MM shortcut.
    delta = prediction.double()[:, None, :] - target.double()[None, :, :]
    squared = delta.square()
    direct = squared.sum(-1)
    learned_dim = target.shape[1] - model.target_encoder.position_dim
    learned = squared[:, :, :learned_dim].sum(-1)
    position = squared[:, :, learned_dim:].sum(-1)
    ordering = direct.argsort(1)
    true = torch.tensor([index[o.true_next_state] for o in report.outcomes])
    wrong = direct.clone()
    wrong[torch.arange(len(true)), true] = float("inf")
    margins = wrong.min(1).values - direct[torch.arange(len(true)), true]
    self_dist = (target.double()[:, None] - target.double()[None, :]).square().sum(-1)
    self_correct = int((self_dist.argmin(1) == torch.arange(len(states))).sum())
    self_dist.fill_diagonal_(float("inf"))
    rows = []
    for i, outcome in enumerate(report.outcomes):
        dr, dc = env.ACTIONS[outcome.action]
        attempted = (outcome.state[0] + dr, outcome.state[1] + dc)
        motion = (
            "boundary_block"
            if not env.in_bounds(attempted)
            else "wall_block"
            if env.is_wall(attempted)
            else "free_move"
        )
        rows.append(
            {
                "seed": seed,
                **metadata,
                "state_row": outcome.state[0],
                "state_col": outcome.state[1],
                "action": outcome.action,
                "action_name": env.ACTION_NAMES[outcome.action],
                "true_row": outcome.true_next_state[0],
                "true_col": outcome.true_next_state[1],
                "pred_row": outcome.predicted_next_state[0],
                "pred_col": outcome.predicted_next_state[1],
                "correct": outcome.correct,
                "kind": outcome.kind,
                "motion": motion,
                "rank": outcome.true_rank,
                "true_distance": outcome.true_distance,
                "pred_distance": outcome.predicted_distance,
                "direct64_correct": int(ordering[i, 0]) == int(true[i]),
                "float32_vs_direct64_disagree": states[int(ordering[i, 0])]
                != outcome.predicted_next_state,
                "position_only_correct": int(position[i].argmin()) == int(true[i]),
                "learned_only_correct": int(learned[i].argmin()) == int(true[i]),
                "squared_true_vs_nearest_wrong_margin": float(margins[i]),
                "true_squared_learned_error": float(learned[i, true[i]]),
                "true_squared_position_error": float(position[i, true[i]]),
            }
        )
    return rows, {
        "seed": seed,
        **metadata,
        "states": len(states),
        "target_self_retrieval_correct": self_correct,
        "minimum_target_separation": float(self_dist.min().sqrt()),
    }


def summarize(rows):
    count = len(rows)
    errors = sum(not r["correct"] for r in rows)
    result = {"transitions": count, "errors": errors, "accuracy": 1 - errors / count}
    for key in ("direct64_correct", "position_only_correct", "learned_only_correct"):
        result[key.replace("_correct", "_accuracy")] = sum(r[key] for r in rows) / count
    result["top2_accuracy"] = sum(r["rank"] <= 2 for r in rows) / count
    result["top3_accuracy"] = sum(r["rank"] <= 3 for r in rows) / count
    result["float32_vs_direct64_disagreements"] = sum(
        r["float32_vs_direct64_disagree"] for r in rows
    )
    return result


def grouped(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in fields)].append(row)
    return [
        {**dict(zip(fields, key, strict=True)), **summarize(value)}
        for key, value in sorted(groups.items())
    ]


def evaluate_model(model, seed, output_dir):
    output_dir = fresh_directory(output_dir)
    rows, maps = [], []
    for metadata, spec in catalogue():
        outcomes, geometry = inspect_map(model, spec, metadata, seed)
        rows.extend(outcomes)
        maps.append(geometry)
    write_csv(output_dir / "transitions.csv", rows)
    write_csv(output_dir / "map_geometry.csv", maps)
    report = {
        "seed": seed,
        "by_split": grouped(rows, ("split",)),
        "by_map": grouped(rows, ("split", "map", "family", "rotation")),
        "by_family": grouped(rows, ("split", "family")),
        "by_rotation": grouped(rows, ("split", "rotation")),
        "by_action": grouped(rows, ("split", "action", "action_name")),
        "by_motion": grouped(rows, ("split", "motion")),
        "by_kind": grouped(rows, ("split", "kind")),
        "by_state_action": grouped(rows, ("split", "state_row", "state_col", "action")),
        "self_retrieval": {
            "correct": sum(m["target_self_retrieval_correct"] for m in maps),
            "states": sum(m["states"] for m in maps),
            "minimum_target_separation": min(
                m["minimum_target_separation"] for m in maps
            ),
        },
        "exports": {p.name: digest(p) for p in output_dir.iterdir()},
    }
    write_json(output_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = fresh_directory(args.output_dir)
    torch.set_num_threads(4)
    preserved = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted((ROOT / "outputs").rglob("*"))
        if p.is_file() and "top1_quality" not in p.parts
    }
    preserved.update(
        {
            str(p.relative_to(ROOT)): digest(p)
            for folder in ("learning", "verification", "evaluation")
            for p in (ROOT / "src/jepa_lmc" / folder).glob("*.py")
        }
    )
    write_json(output / "preservation_before.json", preserved)
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "script_sha256": digest(__file__),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "threads": 4,
        },
        "split_audit": split_audit(),
        "models": [],
    }
    saved = json.loads(
        (ROOT / "outputs/behavioral_relations/top1/report.json").read_text()
    )
    saved_counts = {
        (r["seed"], r["case"]): r["correct_top1_pairs"] for r in saved["maps"]
    }
    for seed, path in BASELINES.items():
        model, payload = load_model(path)
        scores = evaluate_model(model, seed, output / f"seed_{seed}")
        for row in scores["by_map"]:
            if row["split"] == "stress" and (
                row["transitions"] - row["errors"] != saved_counts[seed, row["map"]]
            ):
                raise AssertionError("Baseline stress predictions changed.")
        history = payload["training"]["history"]
        entry = {
            "seed": seed,
            "checkpoint": str(path.relative_to(ROOT)),
            "checkpoint_sha256": digest(path),
            "model_kwargs": payload["model_kwargs"],
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "payload_keys": sorted(payload),
            "training": payload["training"],
            "last_10_epoch_mean_loss": sum(h["loss"] for h in history[-10:]) / 10,
            "epoch_51_60_mean_loss": sum(h["loss"] for h in history[50:60]) / 10,
            "scores": scores,
        }
        report["models"].append(entry)
        print(seed, json.dumps(scores["by_split"]), flush=True)
    if any(digest(ROOT / p) != expected for p, expected in preserved.items()):
        raise AssertionError("A preserved input changed.")
    report["preserved_files"] = len(preserved)
    report["preservation_passed"] = True
    write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
