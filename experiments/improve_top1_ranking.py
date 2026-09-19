"""Bounded ranking/depth fine-tuning; freeze validation selection before testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from experiments.improve_top1 import accuracy
from experiments.top1_quality_diagnosis import (
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
from jepa_lmc.learning.data import gridworld_observation
from jepa_lmc.learning.model import ActionJEPA, JEPAOutput, action_jepa_loss
from jepa_lmc.learning.ranking import same_map_ranking_loss
from jepa_lmc.learning.training import make_action_jepa_optimizer

PROTOCOL = ROOT / "configs/top1_ranking_protocol.json"
DEFAULT_RUN = ROOT / "outputs/top1_quality/round2"


@dataclass
class TrainingMap:
    observations: torch.Tensor
    sources: torch.Tensor
    actions: torch.Tensor
    successors: torch.Tensor


def training_maps():
    """Only the original 40 training maps; hash their original transition tensors."""
    result = []
    hasher = hashlib.sha256()
    for spec in make_pilot_benchmark_splits().train[:40]:
        env = spec.make_env()
        states = tuple(env.all_states())
        indices = {s: i for i, s in enumerate(states)}
        observations = torch.stack([gridworld_observation(env, s) for s in states])
        transitions = tuple(env.all_transitions())
        sources = torch.tensor([indices[t.state] for t in transitions])
        actions = torch.tensor([t.action for t in transitions])
        successors = torch.tensor([indices[t.next_state] for t in transitions])
        for s, a, t in zip(sources, actions, successors, strict=True):
            for key, value in (
                ("observation", observations[s]),
                ("action", a),
                ("next_observation", observations[t]),
            ):
                hasher.update(key.encode())
                hasher.update(value.contiguous().numpy().tobytes())
        result.append(TrainingMap(observations, sources, actions, successors))
    if sum(len(m.actions) for m in result) != 4736:
        raise AssertionError("Original training coverage changed.")
    return result, hasher.hexdigest()


def batch_maps(maps):
    observations, sources, actions, successors, memberships = [], [], [], [], []
    offset = 0
    for i, m in enumerate(maps):
        observations.append(m.observations)
        sources.append(m.sources + offset)
        actions.append(m.actions)
        successors.append(m.successors + offset)
        memberships.append(torch.full((len(m.observations),), i, dtype=torch.long))
        offset += len(m.observations)
    return tuple(
        torch.cat(items)
        for items in (observations, sources, actions, successors, memberships)
    )


def warm_start(seed, depth, observations):
    baseline, payload = load_model(BASELINES[seed])
    kwargs = dict(payload["model_kwargs"])
    if depth:
        kwargs["predictor_residual_blocks"] = depth
        model = ActionJEPA(**kwargs)
        expected_new = {
            k for k in model.state_dict() if k.startswith("predictor.residual_blocks.")
        }
        incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
        if (
            set(incompatible.missing_keys) != expected_new
            or incompatible.unexpected_keys
        ):
            raise AssertionError("Unexpected architecture incompatibility.")
    else:
        model = baseline
    model.eval()
    with torch.no_grad():
        source = torch.arange(len(observations)).repeat_interleave(4)
        actions = torch.arange(4).repeat(len(observations))
        old_context = baseline.context_encoder(observations)
        new_context = model.context_encoder(observations)
        if not torch.equal(old_context, new_context) or not torch.equal(
            baseline.predictor(old_context[source], actions),
            model.predictor(new_context[source], actions),
        ):
            raise AssertionError("Warm-start block changed baseline predictions.")
    return model, kwargs, payload


def learning_rate(protocol, epoch):
    progress = (epoch - 1) / (protocol["max_additional_epochs"] - 1)
    high, low = protocol["learning_rate"], protocol["final_learning_rate"]
    return low + (high - low) * (1 + math.cos(math.pi * progress)) / 2


def train_epoch(model, optimizer, maps, generator, config, protocol):
    model.train()
    totals = dict.fromkeys(
        ("loss", "prediction", "variance", "covariance", "ranking"), 0.0
    )
    examples = steps = 0
    order = torch.randperm(len(maps), generator=generator).tolist()
    for start in range(0, len(order), protocol["maps_per_batch"]):
        obs, source, action, successor, membership = batch_maps(
            [maps[i] for i in order[start : start + protocol["maps_per_batch"]]]
        )
        optimizer.zero_grad(set_to_none=True)
        context = model.context_encoder(obs)
        prediction = model.predictor(context[source], action)
        with torch.no_grad():
            targets = model.target_encoder(obs)
        base = action_jepa_loss(
            JEPAOutput(context[source], prediction, targets[successor]),
            variance_weight=protocol["variance_weight"],
            covariance_weight=protocol["covariance_weight"],
        )
        ranking = (
            same_map_ranking_loss(
                prediction,
                targets,
                successor,
                membership[source],
                membership,
                temperature=protocol["ranking_temperature"],
            )
            if config["ranking_weight"]
            else prediction.new_zeros(())
        )
        loss = base.total + config["ranking_weight"] * ranking
        if not torch.isfinite(loss):
            raise AssertionError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        model.update_target_encoder(protocol["ema_momentum"])
        n = len(action)
        for key, value in (
            ("loss", loss),
            ("prediction", base.prediction),
            ("variance", base.variance),
            ("covariance", base.covariance),
            ("ranking", ranking),
        ):
            totals[key] += float(value.detach()) * n
        examples += n
        steps += 1
    if examples != protocol["train_transitions"] or steps != 10:
        raise AssertionError("Training exposure or optimizer budget changed.")
    return {k: v / examples for k, v in totals.items()} | {
        "examples": examples,
        "optimizer_steps": steps,
    }


def initialize(run_dir):
    fresh_directory(run_dir)
    paths = [p for p in (ROOT / "outputs").rglob("*") if p.is_file()]
    preserved = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted(paths)
        if not p.is_relative_to(run_dir)
    }
    write_json(
        run_dir / "preservation_before.json",
        {
            "files": preserved,
            "baseline": {str(s): digest(p) for s, p in BASELINES.items()},
        },
    )
    write_json(
        run_dir / "study_started.json",
        {
            "protocol": json.loads(PROTOCOL.read_text()),
            "protocol_sha256": digest(PROTOCOL),
            "split_audit": split_audit(),
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_sha256": source_hashes(),
        },
    )


def source_hashes():
    return {
        str(p.relative_to(ROOT)): digest(p)
        for p in (
            Path(__file__).resolve(),
            ROOT / "src/jepa_lmc/learning/model.py",
            ROOT / "src/jepa_lmc/learning/ranking.py",
            ROOT / "src/jepa_lmc/evaluation/dynamics.py",
            ROOT / "experiments/top1_quality_diagnosis.py",
        )
    }


def run_candidate(run_dir, config, seed, protocol):
    started = json.loads((run_dir / "study_started.json").read_text())
    if (
        started["protocol_sha256"] != digest(PROTOCOL)
        or started["source_sha256"] != source_hashes()
    ):
        raise AssertionError("Training protocol or source changed after study lock.")
    output = fresh_directory(run_dir / config["name"] / f"seed_{seed}")
    torch.set_num_threads(protocol["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    maps, tensor_hash = training_maps()
    model, kwargs, original = warm_start(
        seed, config["predictor_residual_blocks"], maps[0].observations
    )
    if (
        original["training"]["train_map_seeds"]
        != started["split_audit"]["train_map_seeds"]
    ):
        raise AssertionError("Baseline training provenance mismatch.")
    optimizer = make_action_jepa_optimizer(
        model,
        learning_rate=protocol["learning_rate"],
        weight_decay=protocol["weight_decay"],
    )
    validation = make_pilot_benchmark_splits().validation
    metadata = {
        "candidate": config["name"],
        "config": config,
        "seed": seed,
        "protocol_sha256": digest(PROTOCOL),
        "source_sha256": source_hashes(),
        "baseline_sha256": digest(BASELINES[seed]),
        "training_tensor_sha256": tensor_hash,
        "model_kwargs": kwargs,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "initial_function_equal": True,
        "optimizer_restarted": True,
        "runtime": {
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "device": "cpu",
            "threads": protocol["threads"],
        },
    }
    write_json(output / "run_started.json", metadata)
    history = []

    def save(name, epoch):
        torch.save(
            {
                "model_kwargs": kwargs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "map_order_rng_state": generator.get_state(),
                "training": {
                    "seed": seed,
                    "candidate": config["name"],
                    "epochs": 100 + epoch,
                    "additional_epochs": epoch,
                    "train_map_seeds": started["split_audit"]["train_map_seeds"],
                    "protocol_sha256": digest(PROTOCOL),
                    "baseline_sha256": metadata["baseline_sha256"],
                    "history": history,
                },
            },
            output / name,
        )

    best_validation = accuracy(model, validation)
    metadata["initial_validation"] = best_validation
    history.append({"additional_epoch": 0, "validation": best_validation})
    best_epoch = stale = 0
    save("model.pt", 0)
    wall_start = time.perf_counter()
    with (output / "history.jsonl").open("w") as handle:
        handle.write(json.dumps(history[0]) + "\n")
        for epoch in range(1, protocol["max_additional_epochs"] + 1):
            lr = learning_rate(protocol, epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            metrics = train_epoch(model, optimizer, maps, generator, config, protocol)
            row = {"additional_epoch": epoch, "learning_rate": lr, **metrics}
            if epoch % protocol["validation_interval"] == 0:
                score = accuracy(model, validation)
                row["validation"] = score
                if score["accuracy"] > best_validation["accuracy"]:
                    best_epoch, best_validation, stale = epoch, score, 0
                else:
                    stale += 1
                row["best_additional_epoch"] = best_epoch
                row["stale_validation_checks"] = stale
            row["elapsed_seconds"] = time.perf_counter() - wall_start
            history.append(row)
            if best_epoch == epoch:
                save("model.pt", epoch)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            if epoch % protocol["validation_interval"] == 0:
                print(config["name"], seed, json.dumps(row), flush=True)
            if (
                epoch >= protocol["minimum_additional_epochs"]
                and stale >= protocol["patience_checks"]
            ):
                break
    save("last_model.pt", epoch)
    best_model, saved = load_model(output / "model.pt")
    if (
        accuracy(best_model, validation) != best_validation
        or saved["training"]["additional_epochs"] != best_epoch
    ):
        raise AssertionError(
            "Saved best checkpoint does not reproduce validation selection."
        )
    if metadata["baseline_sha256"] != digest(BASELINES[seed]):
        raise AssertionError("Baseline checkpoint changed.")
    metadata.update(
        validation=best_validation,
        best_additional_epoch=best_epoch,
        stopped_additional_epoch=epoch,
        stop_reason="patience" if stale >= protocol["patience_checks"] else "budget",
        training_seconds=time.perf_counter() - wall_start,
        checkpoint_sha256=digest(output / "model.pt"),
        last_checkpoint_sha256=digest(output / "last_model.pt"),
    )
    write_json(output / "validation.json", metadata)
    print("COMPLETED", config["name"], seed, json.dumps(best_validation), flush=True)


def select_model(run_dir, protocol):
    """Read validation-only records; raw stress predictions do not yet exist."""
    if (run_dir / "selection.json").exists():
        raise ValueError("Selection already frozen.")
    options = []
    baseline = {
        "candidate": "baseline",
        "parameters": 90776,
        "validation": {},
        "checkpoints": {},
    }
    for config in protocol["candidates"]:
        option = {
            "candidate": config["name"],
            "validation": {},
            "checkpoints": {},
            "best_additional_epochs": {},
        }
        for seed in SEEDS:
            directory = run_dir / config["name"] / f"seed_{seed}"
            value = json.loads((directory / "validation.json").read_text())
            if value["protocol_sha256"] != digest(PROTOCOL) or value[
                "checkpoint_sha256"
            ] != digest(directory / "model.pt"):
                raise AssertionError("Candidate protocol or checkpoint changed.")
            original = value["initial_validation"]["accuracy"]
            if baseline["validation"].setdefault(str(seed), original) != original:
                raise AssertionError("Candidates do not reproduce the same baseline.")
            baseline["checkpoints"][str(seed)] = str(BASELINES[seed].relative_to(ROOT))
            option["validation"][str(seed)] = value["validation"]["accuracy"]
            option["parameters"] = value["trainable_parameters"]
            option["checkpoints"][str(seed)] = str(
                (directory / "model.pt").relative_to(ROOT)
            )
            option["best_additional_epochs"][str(seed)] = value["best_additional_epoch"]
        options.append(option)
    options.insert(0, baseline)
    for option in options:
        option["mean_validation_accuracy"] = sum(option["validation"].values()) / len(
            SEEDS
        )
    best = max(
        options,
        key=lambda o: (
            o["mean_validation_accuracy"],
            -o["parameters"],
            o["candidate"] == "baseline",
        ),
    )
    gain = best["mean_validation_accuracy"] - baseline["mean_validation_accuracy"]
    write_json(
        run_dir / "selection.json",
        {
            "protocol_sha256": digest(PROTOCOL),
            "options": options,
            "selected": best["candidate"],
            "selected_checkpoints": best["checkpoints"],
            "mean_validation_gain": gain,
            "clear_validation_improvement": gain >= 0.01
            and all(
                best["validation"][str(s)] > baseline["validation"][str(s)]
                for s in SEEDS
            ),
            "stress_used_for_selection": False,
        },
    )
    print("SELECTED", best["candidate"], "validation gain", gain, flush=True)


def evaluate_candidate(run_dir, config, seed):
    selection_path = run_dir / "selection.json"
    selection = json.loads(selection_path.read_text())
    if selection["protocol_sha256"] != digest(PROTOCOL):
        raise AssertionError("Protocol changed after selection.")
    output = run_dir / config["name"] / f"seed_{seed}"
    metadata = json.loads((output / "validation.json").read_text())
    if digest(output / "model.pt") != metadata["checkpoint_sha256"]:
        raise AssertionError("Selected checkpoint changed.")
    model, _ = load_model(output / "model.pt")
    torch.set_num_threads(4)
    scores = evaluate_model(model, seed, output / "evaluation")
    if (
        next(s for s in scores["by_split"] if s["split"] == "validation")["accuracy"]
        != metadata["validation"]["accuracy"]
    ):
        raise AssertionError(
            "Final validation differs from frozen selection statistic."
        )
    write_json(
        output / "report.json",
        metadata | {"selection_sha256": digest(selection_path), "scores": scores},
    )
    print("EVALUATED", config["name"], seed, json.dumps(scores["by_split"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--candidate")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument(
        "--phase", choices=("initialize", "train", "select", "evaluate"), required=True
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    protocol = json.loads(PROTOCOL.read_text())
    if args.phase in ("initialize", "select"):
        if args.candidate or args.seed:
            parser.error("Initialization and selection do not take a candidate/seed.")
        if args.phase == "initialize":
            initialize(run_dir)
        else:
            select_model(run_dir, protocol)
        return
    configs = {c["name"]: c for c in protocol["candidates"]}
    if args.candidate not in configs or args.seed is None:
        parser.error("Supply a predeclared --candidate and --seed.")
    if args.phase == "train":
        run_candidate(run_dir, configs[args.candidate], args.seed, protocol)
    else:
        evaluate_candidate(run_dir, configs[args.candidate], args.seed)


if __name__ == "__main__":
    main()
