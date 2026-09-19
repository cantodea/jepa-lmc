"""Run the frozen-model oracle radius experiment, or validation-radius transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jepa_lmc.benchmarks.random_gridworld import make_pilot_benchmark_splits
from jepa_lmc.evaluation.latent_radius import (
    aggregate_candidate_results,
    collect_latent_distances,
    evaluate_candidates,
    oracle_radii,
)
from jepa_lmc.learning.data import GridWorldTransitionDataset
from jepa_lmc.learning.model import ActionJEPA
from jepa_lmc.learning.training import (
    make_action_jepa_optimizer,
    train_action_jepa_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/latent_radius")
    )
    parser.add_argument(
        "--radius-source", choices=("oracle", "validation"), default="oracle"
    )
    parser.add_argument("--train-maps", type=int, default=40)
    parser.add_argument("--test-maps", type=int, default=10)
    parser.add_argument("--validation-maps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.train_maps, args.test_maps, args.validation_maps, args.threads) <= 0:
        parser.error("Map counts and thread count must be positive.")
    if args.batch_size < 2 or args.latent_dim < 2:
        parser.error("Batch size and latent dimension must be at least two.")
    if args.checkpoint is None and args.epochs <= 0:
        parser.error("Training requires positive epochs, or provide --checkpoint.")
    return args


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=jsonable, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_or_train(args, train_specs, test_specs, validation_specs):
    model_kwargs = {"height": 6, "width": 6, "latent_dim": args.latent_dim}
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        # Accept this experiment's bundle or a plain compatible state_dict.
        bundled = "model_state_dict" in payload
        if bundled:
            model_kwargs = payload["model_kwargs"]
        model = ActionJEPA(**model_kwargs)
        model.load_state_dict(payload["model_state_dict"] if bundled else payload)
        training = payload.get("training", {}) if bundled else {}
        train_seeds = training.get("train_map_seeds")
        if train_seeds is not None:
            evaluation_seeds = {spec.seed for spec in test_specs}
            if args.radius_source == "validation":
                evaluation_seeds.update(spec.seed for spec in validation_specs)
            if evaluation_seeds & set(train_seeds):
                raise ValueError("Checkpoint training maps overlap evaluation maps.")
        metadata = {
            "mode": "checkpoint",
            "checkpoint_sha256": hashlib.sha256(
                args.checkpoint.read_bytes()
            ).hexdigest(),
            "model_kwargs": model_kwargs,
            "training": training,
            "training_split_provenance": "recorded"
            if train_seeds is not None
            else "unknown",
        }
        return model, metadata

    dataset = GridWorldTransitionDataset(train_specs)
    if len(dataset) % args.batch_size == 1:
        raise ValueError(
            "Final training batch would have one sample; change --batch-size."
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = ActionJEPA(**model_kwargs)
    optimizer = make_action_jepa_optimizer(model, learning_rate=args.learning_rate)
    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = train_action_jepa_epoch(
            model,
            loader,
            optimizer,
            ema_momentum=args.ema_momentum,
        )
        history.append({"epoch": epoch, **asdict(metrics)})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}: loss={metrics.loss:.6f}", flush=True)
    training = {
        "train_map_seeds": [spec.seed for spec in train_specs],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "ema_momentum": args.ema_momentum,
        "history": history,
    }
    checkpoint_path = args.output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "training": training,
        },
        checkpoint_path,
    )
    return model, {
        "mode": "trained_with_existing_recipe",
        "model_kwargs": model_kwargs,
        "training": training,
        "training_split_provenance": "recorded",
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    splits = make_pilot_benchmark_splits()
    for count, available in (
        (args.train_maps, splits.train),
        (args.test_maps, splits.test),
        (args.validation_maps, splits.validation),
    ):
        if count > len(available):
            raise ValueError("Requested more maps than the fixed pilot split contains.")
    # Fixed offsets: changing map counts never changes the test-map identities.
    train_specs = splits.train[: args.train_maps]
    test_specs = splits.test[: args.test_maps]
    validation_specs = splits.validation[: args.validation_maps]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in ("report.json", "model.pt")):
        raise ValueError(
            "Output directory contains a previous run; choose a new directory."
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    model, model_metadata = load_or_train(
        args, train_specs, test_specs, validation_specs
    )

    test_tables = [
        collect_latent_distances(model, spec.make_env()) for spec in test_specs
    ]
    radius_specs = test_specs if args.radius_source == "oracle" else validation_specs
    radius_tables = (
        test_tables
        if args.radius_source == "oracle"
        else [collect_latent_distances(model, spec.make_env()) for spec in radius_specs]
    )
    radii = oracle_radii(radius_tables)
    print(f"Radii ({args.radius_source}): {radii}", flush=True)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[1],
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    report = {
        "schema_version": 1,
        "protocol": {
            "radius_source": args.radius_source,
            "uses_test_truth_to_choose_radius": args.radius_source == "oracle",
            "evaluation_scope": "all valid state-action pairs on each listed test map",
            "distance": (
                "unnormalized Euclidean; float64 direct distance of frozen outputs"
            ),
            "quantile_interpolation": "linear",
            "ball": "closed (distance <= epsilon)",
            "candidate_universe": "all valid states on the same map",
            "empty_set_policy": (
                "no repair; skip CTL only if a state has no outgoing edge"
            ),
            "ctl_scope": (
                "all states, including unreachable states; six existing properties"
            ),
            "guarantee_scope": (
                "finite enumerated graphs only; no unseen-map error guarantee"
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": "cpu",
            "threads": args.threads,
            "base_revision": revision,
            "command": sys.argv,
            "evaluation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
                + (
                    Path(__file__).resolve().parents[1]
                    / "src/jepa_lmc/evaluation/latent_radius.py"
                ).read_bytes()
            ).hexdigest(),
        },
        "model": model_metadata,
        "radii": radii,
        "radius_source_pairs": sum(len(table.pairs) for table in radius_tables),
        "radius_source_map_seeds": [spec.seed for spec in radius_specs],
        "test_maps": [asdict(spec) for spec in test_specs],
        "summary": {},
        "per_map": [],
    }
    errors = []
    for split, specs, tables in [("test", test_specs, test_tables)] + (
        [("validation", radius_specs, radius_tables)]
        if args.radius_source == "validation"
        else []
    ):
        for spec, table in zip(specs, tables, strict=True):
            for row, ((state, action), index) in enumerate(
                zip(table.pairs, table.true_indices, strict=True)
            ):
                target = table.states[index]
                errors.append(
                    {
                        "split": split,
                        "map_seed": spec.seed,
                        "state_row": state[0],
                        "state_col": state[1],
                        "action": action,
                        "true_next_row": target[0],
                        "true_next_col": target[1],
                        "error": float(table.errors[row]),
                    }
                )
    write_csv(args.output_dir / "errors.csv", errors)

    ctl_rows = []
    candidates_path = args.output_dir / "candidate_sets.jsonl"
    with candidates_path.open("w", encoding="utf-8") as candidate_file:
        for name in ("top1", "epsilon_95", "epsilon_99", "epsilon_max", "all_states"):
            map_results = []
            for spec, table in zip(test_specs, test_tables, strict=True):
                if name == "top1":
                    mask = torch.zeros_like(table.distances, dtype=torch.bool)
                    mask.scatter_(1, table.distances.argmin(dim=1, keepdim=True), True)
                elif name == "all_states":
                    mask = torch.ones_like(table.distances, dtype=torch.bool)
                else:
                    mask = table.candidates(radii[name])
                    for row, (state, action) in enumerate(table.pairs):
                        candidate_file.write(
                            json.dumps(
                                {
                                    "radius": name,
                                    "epsilon": radii[name],
                                    "map_seed": spec.seed,
                                    "state": state,
                                    "action": action,
                                    "candidates": [
                                        table.states[i]
                                        for i in mask[row].nonzero().flatten().tolist()
                                    ],
                                    "covered": bool(mask[row, table.true_indices[row]]),
                                },
                                allow_nan=False,
                            )
                            + "\n"
                        )
                result = evaluate_candidates(spec.make_env(), table, mask)
                map_results.append(result)
                report["per_map"].append(
                    {
                        "variant": name,
                        "map_seed": spec.seed,
                        **{
                            key: value
                            for key, value in result.items()
                            if key != "ctl_outcomes"
                        },
                    }
                )
                ctl_rows.extend(
                    {"variant": name, "map_seed": spec.seed, **row}
                    for row in result["ctl_outcomes"]
                )
                if (
                    result["relation_inclusion"]
                    and result["ctl"]["one_sided_violations"]
                ):
                    raise AssertionError(
                        "CTL transfer violated despite audited relation inclusion."
                    )
                if (
                    args.radius_source == "oracle"
                    and name == "epsilon_max"
                    and not result["action_inclusion"]
                ):
                    raise AssertionError(
                        "Oracle maximum must cover every enumerated successor."
                    )
            summary = aggregate_candidate_results(map_results)
            report["summary"][name] = summary
            agreement = summary["ctl"]["agreement"]
            agreement_text = "N/A" if agreement is None else f"{agreement:.2%}"
            print(
                f"{name}: coverage={summary['successor_coverage']:.2%}, "
                f"mean |C|={summary['mean_candidate_size']:.3f}, "
                f"singletons={summary['singleton_fraction']:.2%}, "
                f"CTL agreement={agreement_text}, "
                f"inclusion={summary['relation_inclusion_maps']}/"
                f"{summary['maps']} maps",
                flush=True,
            )
    write_csv(args.output_dir / "ctl_outcomes.csv", ctl_rows)
    write_json(args.output_dir / "report.json", report)
    print(f"Saved experiment to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
