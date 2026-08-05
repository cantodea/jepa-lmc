from __future__ import annotations

from jepa_lmc.benchmarks.perturbations import (
    block_entries_to_proposition,
)
from jepa_lmc.benchmarks.random_gridworld import (
    make_pilot_benchmark_splits,
)
from jepa_lmc.evaluation.metrics import (
    VerificationReport,
    aggregate_reports,
    evaluate_ctl_suite_for_state_pairs,
)
from jepa_lmc.verification.gridworld import (
    gridworld_to_transition_system,
)


def print_summary(title: str, report: VerificationReport) -> None:
    print(f"\n=== {title} ===")
    print(f"CTL queries: {report.total}")
    print(f"CTL agreement: {report.agreement:.1%}")
    if report.primary_balanced_score is None:
        print("Primary balanced CTL score: N/A")
    else:
        print(
            "Primary balanced CTL score: "
            f"{report.primary_balanced_score:.1%}"
        )
    print(f"False-safe count: {report.false_safe_count}")
    if report.unsafe_miss_rate is None:
        print("Unsafe-miss rate: N/A")
    else:
        print(f"Unsafe-miss rate: {report.unsafe_miss_rate:.1%}")

    print("\nPer-property agreement / ground-truth True rate:")
    property_agreement = report.agreement_by_property
    positive_rates = report.ground_truth_positive_rate_by_property
    confusion = report.confusion_by_property
    for name in property_agreement:
        balanced_accuracy = confusion[name].balanced_accuracy
        balanced_text = (
            "N/A"
            if balanced_accuracy is None
            else f"{balanced_accuracy:.1%}"
        )
        true_rate = positive_rates[name]
        imbalance_warning = (
            " [IMBALANCED]"
            if true_rate < 0.1 or true_rate > 0.9
            else ""
        )
        print(
            f"{name}: agreement={property_agreement[name]:.1%}, "
            f"balanced={balanced_text}, true_rate={true_rate:.1%}"
            f"{imbalance_warning}"
        )


def main() -> None:
    splits = make_pilot_benchmark_splits()
    perfect_reports: list[VerificationReport] = []
    danger_blind_reports: list[VerificationReport] = []
    state_count = 0

    for spec in splits.test:
        env = spec.make_env()
        ground_truth = gridworld_to_transition_system(env)
        danger_blind = block_entries_to_proposition(
            ground_truth, "danger"
        )
        states = tuple(sorted(ground_truth.states))
        state_pairs = tuple((state, state) for state in states)
        state_count += len(states)

        perfect_reports.append(
            evaluate_ctl_suite_for_state_pairs(
                ground_truth,
                ground_truth,
                state_pairs,
            )
        )
        danger_blind_reports.append(
            evaluate_ctl_suite_for_state_pairs(
                ground_truth,
                danger_blind,
                state_pairs,
            )
        )

    perfect = aggregate_reports(perfect_reports)
    danger_blind = aggregate_reports(danger_blind_reports)

    print("=== Random-map CTL benchmark ===")
    print(f"Train/validation/test maps: {len(splits.train)}/"
          f"{len(splits.validation)}/{len(splits.test)}")
    print(f"Test states: {state_count}")
    print_summary("Perfect model", perfect)
    print_summary("Danger-blind perturbed model", danger_blind)

    assert perfect.agreement == 1.0
    assert perfect.primary_balanced_score == 1.0
    assert perfect.false_safe_count == 0
    assert danger_blind.agreement < 1.0
    assert danger_blind.primary_balanced_score is not None
    assert danger_blind.primary_balanced_score < 1.0
    assert danger_blind.false_safe_count > 0

    print("\nRandom-map CTL benchmark passed.")


if __name__ == "__main__":
    main()
