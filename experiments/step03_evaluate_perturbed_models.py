from __future__ import annotations

from pathlib import Path

from jepa_lmc.benchmarks.perturbations import (
    block_entries_to_proposition,
)
from jepa_lmc.checking.gridworld_adapter import (
    gridworld_to_transition_system,
)
from jepa_lmc.envs.factory import make_gridworld_from_config
from jepa_lmc.evaluation.verification_metrics import (
    VerificationReport,
    evaluate_ctl_suite,
)
from jepa_lmc.utils.config import load_yaml


def print_report(title: str, report: VerificationReport) -> None:
    print(f"\n=== {title} ===")
    for outcome in report.outcomes:
        status = "MATCH" if outcome.matches else "MISMATCH"
        warning = " [FALSE SAFE]" if outcome.false_safe else ""
        print(
            f"{outcome.name}: ground_truth={outcome.ground_truth}, "
            f"learned={outcome.learned} -> {status}{warning}"
        )

    print(f"CTL agreement: {report.agreement:.1%}")
    print(f"False-safe count: {report.false_safe_count}")
    if report.unsafe_miss_rate is None:
        print("Unsafe-miss rate: N/A (no unsafe ground-truth cases)")
    else:
        print(f"Unsafe-miss rate: {report.unsafe_miss_rate:.1%}")


def main() -> None:
    config = load_yaml(Path("configs/gridworld_6x6.yaml"))
    env = make_gridworld_from_config(config)
    ground_truth = gridworld_to_transition_system(env)

    perfect_report = evaluate_ctl_suite(
        ground_truth,
        ground_truth,
        env.start,
        env.start,
    )

    danger_blind_model = block_entries_to_proposition(
        ground_truth, "danger"
    )
    danger_blind_report = evaluate_ctl_suite(
        ground_truth,
        danger_blind_model,
        env.start,
        env.start,
    )

    print("=== Step 03A: CTL verification scorecard ===")
    print_report("Perfect model", perfect_report)
    print_report("Danger-blind perturbed model", danger_blind_report)

    assert perfect_report.agreement == 1.0
    assert perfect_report.false_safe_count == 0
    assert danger_blind_report.agreement < 1.0
    assert danger_blind_report.false_safe_count == 1
    assert danger_blind_report.unsafe_miss_rate == 1.0

    print("\nCTL benchmark scoring passed.")


if __name__ == "__main__":
    main()
