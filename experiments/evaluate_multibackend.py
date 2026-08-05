from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jepa_lmc.benchmarks.ltl_suite import default_ltl_suite
from jepa_lmc.benchmarks.random_gridworld import make_pilot_benchmark_splits
from jepa_lmc.evaluation.dynamics import (
    TransitionRetrievalReport,
    aggregate_transition_reports,
    evaluate_gridworld_transitions,
    transition_system_from_retrievals,
)
from jepa_lmc.evaluation.metrics import (
    VerificationReport,
    aggregate_reports,
    evaluate_ctl_suite_for_state_pairs,
    evaluate_ltl_suite_for_state_pairs,
)
from jepa_lmc.learning.data import GridWorldTransitionDataset
from jepa_lmc.learning.model import ActionJEPA
from jepa_lmc.learning.training import (
    make_action_jepa_optimizer,
    train_action_jepa_epoch,
)
from jepa_lmc.verification.gridworld import gridworld_to_transition_system
from jepa_lmc.verification.nuxmv import find_nusmv_executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen Action-JEPA with CTL and LTL backends."
    )
    parser.add_argument("--train-maps", type=int, default=40)
    parser.add_argument("--test-maps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--executable", type=Path)
    return parser.parse_args()


def print_transition_report(report: TransitionRetrievalReport) -> None:
    print("\n=== Frozen JEPA transition fidelity ===")
    print(f"Transitions: {report.total}")
    print(f"Top-1 next-state accuracy: {report.top1_accuracy:.1%}")
    print(f"Top-3 next-state accuracy: {report.top_k_accuracy(3):.1%}")
    print(f"Mean reciprocal rank: {report.mean_reciprocal_rank:.3f}")


def print_logic_report(logic: str, report: VerificationReport) -> None:
    print(f"\n=== Frozen JEPA learned-transition {logic} ===")
    print(f"{logic} queries: {report.total}")
    print(f"{logic} agreement: {report.agreement:.1%}")
    balanced = report.primary_balanced_score
    print(
        f"Primary balanced {logic} score: "
        + ("N/A" if balanced is None else f"{balanced:.1%}")
    )
    print(f"False-safe count: {report.false_safe_count}")
    print(f"Per-property {logic} agreement:")
    for name, agreement in report.agreement_by_property.items():
        print(f"  {name}: {agreement:.1%}")


def main() -> None:
    args = parse_args()
    if min(args.train_maps, args.test_maps, args.epochs, args.log_every) <= 0:
        raise ValueError("Map, epoch, and log counts must be positive.")
    if args.batch_size < 2:
        raise ValueError("Batch size must be at least two.")

    executable = find_nusmv_executable(args.executable)
    if executable is None:
        raise FileNotFoundError(
            "Multi-backend evaluation requires nuXmv/NuSMV for LTL. "
            "Set NUXMV_BINARY or pass --executable PATH."
        )

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

    print("=== Train once, verify with CTL and LTL ===")
    print(f"Training maps: {len(train_specs)}")
    print(f"Held-out test maps: {len(test_specs)}")
    print(f"Training transitions: {len(dataset)}")
    print(f"External LTL backend: {executable}")
    print("Temporal-logic labels used during JEPA training: none")
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

    model.requires_grad_(False)
    model.eval()
    print("Model frozen before CTL/LTL property evaluation: yes")

    transition_reports: list[TransitionRetrievalReport] = []
    ctl_reports: list[VerificationReport] = []
    ltl_reports: list[VerificationReport] = []
    ltl_suite = default_ltl_suite()

    for spec in test_specs:
        env = spec.make_env()
        ground_truth = gridworld_to_transition_system(env)
        state_pairs = tuple((state, state) for state in ground_truth.states)
        retrievals = evaluate_gridworld_transitions(model, env)
        learned_system = transition_system_from_retrievals(env, retrievals)

        transition_reports.append(retrievals)
        ctl_reports.append(
            evaluate_ctl_suite_for_state_pairs(
                ground_truth,
                learned_system,
                state_pairs,
            )
        )
        ltl_reports.append(
            evaluate_ltl_suite_for_state_pairs(
                ground_truth,
                learned_system,
                state_pairs,
                ltl_suite,
                executable,
            )
        )

    transitions = aggregate_transition_reports(transition_reports)
    ctl_report = aggregate_reports(ctl_reports)
    ltl_report = aggregate_reports(ltl_reports)

    print_transition_report(transitions)
    print_logic_report("CTL", ctl_report)
    print_logic_report("LTL", ltl_report)

    expected_ltl_queries = sum(
        len(spec.make_env().all_states()) for spec in test_specs
    ) * len(ltl_suite)
    if ltl_report.total != expected_ltl_queries:
        raise AssertionError("LTL evaluation did not cover every state/property pair.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("The JEPA model changed from frozen evaluation mode.")
    if model.training:
        raise AssertionError("The JEPA model left evaluation mode.")
    if not 0.0 <= ctl_report.agreement <= 1.0:
        raise AssertionError("Invalid CTL agreement.")
    if not 0.0 <= ltl_report.agreement <= 1.0:
        raise AssertionError("Invalid LTL agreement.")

    print("\nOne frozen JEPA model completed both verification backends.")


if __name__ == "__main__":
    main()
