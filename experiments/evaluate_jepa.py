from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jepa_lmc.benchmarks.random_gridworld import make_pilot_benchmark_splits
from jepa_lmc.evaluation.dynamics import (
    RolloutReport,
    TransitionRetrievalReport,
    aggregate_rollout_reports,
    aggregate_transition_reports,
    evaluate_gridworld_transitions,
    evaluate_open_loop_rollouts,
    transition_system_from_retrievals,
)
from jepa_lmc.evaluation.metrics import (
    VerificationReport,
    aggregate_reports,
    evaluate_ctl_suite_for_state_pairs,
)
from jepa_lmc.evaluation.structural import evaluate_model_pair, save_model_evaluation
from jepa_lmc.learning.data import GridWorldTransitionDataset
from jepa_lmc.learning.model import ActionJEPA
from jepa_lmc.learning.training import (
    make_action_jepa_optimizer,
    train_action_jepa_epoch,
)
from jepa_lmc.verification.gridworld import gridworld_to_transition_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Action-JEPA transitions and CTL fidelity."
    )
    parser.add_argument("--train-maps", type=int, default=40)
    parser.add_argument("--test-maps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--rollouts-per-state", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/model_evaluation/jepa")
    )
    return parser.parse_args()


def print_transition_report(title: str, report: TransitionRetrievalReport) -> None:
    print(f"\n=== {title} ===")
    print(f"Transitions: {report.total}")
    print(f"Top-1 next-state accuracy: {report.top1_accuracy:.1%}")
    print(f"Top-3 next-state accuracy: {report.top_k_accuracy(3):.1%}")
    print(f"Mean reciprocal rank: {report.mean_reciprocal_rank:.3f}")
    danger_recall = report.danger_destination_recall
    print(
        "Danger-destination recall: "
        + ("N/A" if danger_recall is None else f"{danger_recall:.1%}")
    )
    print(f"Unsafe transition misses: {report.unsafe_miss_count}")
    false_danger_rate = report.false_danger_rate
    print(
        "False-danger rate: "
        + ("N/A" if false_danger_rate is None else f"{false_danger_rate:.1%}")
    )
    print("Accuracy by transition kind:")
    for kind, accuracy in report.accuracy_by_kind.items():
        print(f"  {kind}: {accuracy:.1%}")


def print_ctl_report(title: str, report: VerificationReport) -> None:
    print(f"\n=== {title} ===")
    print(f"CTL queries: {report.total}")
    print(f"CTL agreement: {report.agreement:.1%}")
    balanced = report.primary_balanced_score
    print(
        "Additional balanced CTL diagnostic: "
        + ("N/A" if balanced is None else f"{balanced:.1%}")
    )
    print(f"False-safe count: {report.false_safe_count}")
    unsafe_miss_rate = report.unsafe_miss_rate
    print(
        "Unsafe-miss rate: "
        + ("N/A" if unsafe_miss_rate is None else f"{unsafe_miss_rate:.1%}")
    )
    print("Per-property CTL agreement:")
    for name, agreement in report.agreement_by_property.items():
        print(f"  {name}: {agreement:.1%}")


def print_rollout_report(report: RolloutReport) -> None:
    print("\n=== Open-loop latent rollout accuracy ===")
    for horizon, accuracy in report.accuracy_by_horizon.items():
        print(f"Horizon {horizon}: {accuracy:.1%}")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Choose a fresh output directory.")
    if (
        min(
            args.train_maps,
            args.test_maps,
            args.epochs,
            args.rollouts_per_state,
            args.log_every,
        )
        <= 0
    ):
        raise ValueError("Map, epoch, and rollout counts must be positive.")
    if args.batch_size < 2:
        raise ValueError("Batch size must be at least two.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = make_pilot_benchmark_splits()
    if args.train_maps > len(splits.train):
        raise ValueError("Requested more training maps than the benchmark contains.")
    if args.test_maps > len(splits.test):
        raise ValueError("Requested more test maps than the benchmark contains.")

    train_specs = splits.train[: args.train_maps]
    test_specs = splits.test[: args.test_maps]
    dataset = GridWorldTransitionDataset(train_specs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = ActionJEPA(
        height=train_specs[0].height,
        width=train_specs[0].width,
        latent_dim=args.latent_dim,
    )
    optimizer = make_action_jepa_optimizer(
        model,
        learning_rate=args.learning_rate,
    )

    print("=== JEPA transition and CTL evaluation ===")
    print(f"Training maps: {len(train_specs)}")
    print(f"Held-out test maps: {len(test_specs)}")
    print(f"Training transitions: {len(dataset)}")
    print(f"Latent dimension: {args.latent_dim}")
    for epoch in range(1, args.epochs + 1):
        metrics = train_action_jepa_epoch(
            model,
            loader,
            optimizer,
            ema_momentum=args.ema_momentum,
        )
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}: total={metrics.loss:.5f}, "
                f"prediction={metrics.prediction_loss:.5f}, "
                f"variance={metrics.variance_loss:.5f}, "
                f"covariance={metrics.covariance_loss:.5f}"
            )

    jepa_transition_reports: list[TransitionRetrievalReport] = []
    ablated_transition_reports: list[TransitionRetrievalReport] = []
    rollout_reports: list[RolloutReport] = []
    jepa_ctl_reports: list[VerificationReport] = []
    ablated_ctl_reports: list[VerificationReport] = []
    comparisons = {"jepa": [], "action_masked": []}

    for index, spec in enumerate(test_specs):
        env = spec.make_env()
        ground_truth = gridworld_to_transition_system(env)
        state_pairs = tuple((state, state) for state in ground_truth.states)

        jepa_transitions = evaluate_gridworld_transitions(model, env)
        ablated_transitions = evaluate_gridworld_transitions(
            model,
            env,
            action_override=0,
        )
        jepa_transition_reports.append(jepa_transitions)
        ablated_transition_reports.append(ablated_transitions)
        rollout_reports.append(
            evaluate_open_loop_rollouts(
                model,
                env,
                rollouts_per_state=args.rollouts_per_state,
                seed=args.seed + index,
            )
        )

        jepa_system = transition_system_from_retrievals(env, jepa_transitions)
        ablated_system = transition_system_from_retrievals(env, ablated_transitions)
        for name, graph, retrieval in (
            ("jepa", jepa_system, jepa_transitions),
            ("action_masked", ablated_system, ablated_transitions),
        ):
            comparisons[name].append(
                {
                    "seed": args.seed,
                    "case": str(index),
                    **evaluate_model_pair(
                        ground_truth,
                        graph,
                        top1_counts=(
                            sum(o.correct for o in retrieval.outcomes),
                            retrieval.total,
                        ),
                    ),
                }
            )
        jepa_ctl_reports.append(
            evaluate_ctl_suite_for_state_pairs(
                ground_truth,
                jepa_system,
                state_pairs,
            )
        )
        ablated_ctl_reports.append(
            evaluate_ctl_suite_for_state_pairs(
                ground_truth,
                ablated_system,
                state_pairs,
            )
        )

    jepa_transitions = aggregate_transition_reports(jepa_transition_reports)
    ablated_transitions = aggregate_transition_reports(ablated_transition_reports)
    rollouts = aggregate_rollout_reports(rollout_reports)
    jepa_ctl = aggregate_reports(jepa_ctl_reports)
    ablated_ctl = aggregate_reports(ablated_ctl_reports)

    for name, cases in comparisons.items():
        report = save_model_evaluation(args.output_dir / f"{name}.json", cases)
        print(
            f"{name} primary structural metrics: {report['summary']['primary_metrics']}"
        )
    print_transition_report("Action-conditioned JEPA", jepa_transitions)
    print_transition_report("Action-masked ablation", ablated_transitions)
    print_rollout_report(rollouts)
    print_ctl_report("JEPA learned-transition CTL", jepa_ctl)
    print_ctl_report("Action-masked learned-transition CTL", ablated_ctl)

    if not 0.0 <= jepa_transitions.top1_accuracy <= 1.0:
        raise AssertionError("Invalid transition accuracy.")
    if set(rollouts.accuracy_by_horizon) != {1, 2, 4, 8}:
        raise AssertionError("Rollout evaluation did not cover every horizon.")
    if not 0.0 <= jepa_ctl.agreement <= 1.0:
        raise AssertionError("Invalid CTL agreement.")
    danger_recall = jepa_transitions.danger_destination_recall
    if jepa_transitions.top1_accuracy < 0.8:
        raise AssertionError("JEPA did not reach the pilot transition threshold.")
    if danger_recall is None or danger_recall < 0.8:
        raise AssertionError("JEPA did not reach the pilot danger-recall threshold.")
    if jepa_ctl.agreement < 0.85 or jepa_ctl.false_safe_count != 0:
        raise AssertionError("JEPA did not reach the pilot CTL safety threshold.")
    if jepa_transitions.top1_accuracy <= ablated_transitions.top1_accuracy + 0.3:
        raise AssertionError("The learned predictor did not depend enough on action.")
    if jepa_ctl.agreement <= ablated_ctl.agreement + 0.3:
        raise AssertionError("Action conditioning did not improve CTL fidelity enough.")
    print("\nJEPA evaluation passed its pilot thresholds.")


if __name__ == "__main__":
    main()
