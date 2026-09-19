"""Run the three predeclared Top-1 candidates and select using validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__:
    from .top1_quality_diagnosis import (
        BASELINES,
        ROOT,
        SEEDS,
        digest,
        evaluate_model,
        fresh_directory,
        load_model,
        split_audit,
        write_json,
    )
else:
    from top1_quality_diagnosis import (
        BASELINES,
        ROOT,
        SEEDS,
        digest,
        evaluate_model,
        fresh_directory,
        load_model,
        split_audit,
        write_json,
    )

from jepa_lmc.benchmarks.random_gridworld import make_pilot_benchmark_splits
from jepa_lmc.evaluation.dynamics import evaluate_gridworld_transitions
from jepa_lmc.learning.data import GridWorldTransitionDataset
from jepa_lmc.learning.model import ActionJEPA
from jepa_lmc.learning.training import (
    make_action_jepa_optimizer,
    train_action_jepa_epoch,
)

PROTOCOL = ROOT / "configs/top1_quality_protocol.json"


def cached_training_data():
    """Cache exactly the original tensors; do not generate extra examples."""
    specs = make_pilot_benchmark_splits().train[:40]
    source = GridWorldTransitionDataset(specs)
    cached = [source[i] for i in range(len(source))]
    hasher = hashlib.sha256()
    for row in cached:
        for key in ("observation", "action", "next_observation"):
            hasher.update(key.encode())
            hasher.update(row[key].contiguous().numpy().tobytes())
    return cached, hasher.hexdigest()


def learning_rate(config, epoch):
    if epoch <= 100:
        return config["learning_rate"]
    progress = (epoch - 101) / (config["final_epoch"] - 101)
    low, high = config["final_learning_rate"], config["learning_rate"]
    return low + (high - low) * (1 + math.cos(math.pi * progress)) / 2


def accuracy(model, specs):
    correct = total = 0
    for spec in specs:
        result = evaluate_gridworld_transitions(model, spec.make_env())
        correct += sum(o.correct for o in result.outcomes)
        total += result.total
    return {
        "transitions": total,
        "errors": total - correct,
        "accuracy": correct / total,
    }


def run_candidate(config, seed, output, protocol):
    output = fresh_directory(output)
    torch.set_num_threads(protocol["threads"])
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    torch.manual_seed(seed)
    audited_splits = split_audit()
    source_hash = digest(BASELINES[seed])
    data, data_hash = cached_training_data()
    if len(data) != 4736:
        raise AssertionError("Original training transition count changed.")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        data,
        batch_size=protocol["batch_size"],
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    if config["initialization"] == "baseline":
        model, payload = load_model(BASELINES[seed])
        if payload["model_kwargs"] != config["model_kwargs"]:
            raise ValueError("Baseline architecture differs from candidate.")
        if payload["training"]["train_map_seeds"] != audited_splits["train_map_seeds"]:
            raise ValueError("Baseline data provenance differs.")
    else:
        model = ActionJEPA(**config["model_kwargs"])
    optimizer = make_action_jepa_optimizer(
        model,
        learning_rate=config["learning_rate"],
        weight_decay=protocol["weight_decay"],
    )
    validation = make_pilot_benchmark_splits().validation
    metadata = {
        "candidate": config["name"],
        "config": config,
        "seed": seed,
        "protocol_sha256": digest(PROTOCOL),
        "split_audit": audited_splits,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__).resolve(),
                ROOT / "experiments/top1_quality_diagnosis.py",
                ROOT / "src/jepa_lmc/learning/model.py",
                ROOT / "src/jepa_lmc/learning/training.py",
                ROOT / "src/jepa_lmc/learning/data.py",
                ROOT / "src/jepa_lmc/evaluation/dynamics.py",
            )
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "threads": protocol["threads"],
            "device": "cpu",
            "deterministic_algorithms": True,
        },
        "baseline_sha256": source_hash,
        "training_tensor_sha256": data_hash,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "optimizer_restart_epoch": 100,
        "exact_resume": False,
    }
    write_json(output / "run_started.json", metadata)
    history = []
    started = time.perf_counter()
    with (output / "history.jsonl").open("w") as log:
        for epoch in range(config["initial_epoch"] + 1, config["final_epoch"] + 1):
            if epoch == 101 and config["initialization"] == "fresh":
                optimizer = make_action_jepa_optimizer(
                    model,
                    learning_rate=config["learning_rate"],
                    weight_decay=protocol["weight_decay"],
                )
                generator.manual_seed(seed)
            lr = learning_rate(config, epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            metrics = train_action_jepa_epoch(
                model, loader, optimizer, ema_momentum=protocol["ema_momentum"]
            )
            row = {
                "epoch": epoch,
                "learning_rate": lr,
                **asdict(metrics),
                "elapsed_seconds": time.perf_counter() - started,
            }
            if epoch in protocol["evaluation_epochs"]:
                row["validation"] = accuracy(model, validation)
            history.append(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if epoch % 25 == 0 or epoch == config["initial_epoch"] + 1:
                print(config["name"], seed, json.dumps(row), flush=True)
    metadata["training_seconds"] = time.perf_counter() - started
    training = {
        "seed": seed,
        "train_map_seeds": audited_splits["train_map_seeds"],
        "epochs": config["final_epoch"],
        "batch_size": protocol["batch_size"],
        "learning_rate": config["learning_rate"],
        "ema_momentum": protocol["ema_momentum"],
        "candidate": config["name"],
        "baseline_sha256": source_hash,
        "history": history,
        "protocol_sha256": digest(PROTOCOL),
        "initial_epoch": config["initial_epoch"],
        "optimizer_restarted_at_epoch": 100,
    }
    torch.save(
        {
            "model_kwargs": config["model_kwargs"],
            "model_state_dict": model.state_dict(),
            "training": training,
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "loader_rng_state": generator.get_state(),
        },
        output / "model.pt",
    )
    # First commit the selection statistic. Stress is only observed afterwards.
    metadata["validation"] = accuracy(model, validation)
    metadata["checkpoint_sha256"] = digest(output / "model.pt")
    write_json(output / "validation.json", metadata)
    scores = evaluate_model(model, seed, output / "evaluation")
    metadata["scores"] = scores
    if source_hash != digest(BASELINES[seed]):
        raise AssertionError("Baseline checkpoint was modified.")
    write_json(output / "report.json", metadata)
    print("COMPLETED", config["name"], seed, json.dumps(scores["by_split"]), flush=True)


def select_model(run_dir, protocol):
    """This function deliberately reads no stress scores or temporal results."""
    destination = run_dir / "selection.json"
    if destination.exists():
        raise ValueError("Selection already frozen; refusing to change it.")
    diagnosis = json.loads((run_dir / "diagnosis/report.json").read_text())
    baseline = {
        "candidate": "baseline",
        "parameters": diagnosis["models"][0]["trainable_parameters"],
        "validation": {
            str(m["seed"]): next(
                r["accuracy"]
                for r in m["scores"]["by_split"]
                if r["split"] == "validation"
            )
            for m in diagnosis["models"]
        },
        "checkpoints": {str(s): str(p.relative_to(ROOT)) for s, p in BASELINES.items()},
    }
    options = [baseline]
    for config in protocol["candidates"]:
        option = {"candidate": config["name"], "validation": {}, "checkpoints": {}}
        for seed in SEEDS:
            directory = run_dir / config["name"] / f"seed_{seed}"
            value = json.loads((directory / "validation.json").read_text())
            if value["protocol_sha256"] != digest(PROTOCOL):
                raise AssertionError("Candidate used a different protocol.")
            if value["checkpoint_sha256"] != digest(directory / "model.pt"):
                raise AssertionError("Candidate checkpoint changed.")
            option["validation"][str(seed)] = value["validation"]["accuracy"]
            option["parameters"] = value["trainable_parameters"]
            option["checkpoints"][str(seed)] = str(
                (directory / "model.pt").relative_to(ROOT)
            )
        options.append(option)
    for option in options:
        option["mean_validation_accuracy"] = sum(option["validation"].values()) / 3
    best = max(options, key=lambda x: (x["mean_validation_accuracy"], -x["parameters"]))
    improvement = (
        best["mean_validation_accuracy"] - baseline["mean_validation_accuracy"]
    )
    result = {
        "protocol_sha256": digest(PROTOCOL),
        "options": options,
        "selected": best["candidate"],
        "selected_checkpoints": best["checkpoints"],
        "mean_validation_gain": improvement,
        "clear_validation_improvement": improvement >= 0.01
        and all(
            best["validation"][str(s)] > baseline["validation"][str(s)] for s in SEEDS
        ),
        "stress_used_for_selection": False,
    }
    write_json(destination, result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "outputs/top1_quality/round1"
    )
    parser.add_argument("--candidate")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    protocol = json.loads(PROTOCOL.read_text())
    if args.select:
        if args.candidate or args.seed:
            parser.error("Selection is separate from training.")
        select_model(args.run_dir, protocol)
        return
    configurations = {c["name"]: c for c in protocol["candidates"]}
    if args.candidate not in configurations or args.seed is None:
        parser.error("Supply one predeclared --candidate and --seed, or --select.")
    run_candidate(
        configurations[args.candidate],
        args.seed,
        args.run_dir / args.candidate / f"seed_{args.seed}",
        protocol,
    )


if __name__ == "__main__":
    main()
