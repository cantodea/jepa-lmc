from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, TypeVar

from jepa_lmc.benchmarks.ctl_suite import CTLProperty, default_ctl_suite
from jepa_lmc.checking.ctl import CTLModelChecker
from jepa_lmc.checking.transition_system import ExplicitTransitionSystem


GroundStateT = TypeVar("GroundStateT", bound=Hashable)
GroundActionT = TypeVar("GroundActionT", bound=Hashable)
LearnedStateT = TypeVar("LearnedStateT", bound=Hashable)
LearnedActionT = TypeVar("LearnedActionT", bound=Hashable)


@dataclass(frozen=True)
class PropertyOutcome:
    """Ground-truth and learned truth values for one CTL property."""

    name: str
    category: str
    ground_truth: bool
    learned: bool
    safety_claim: bool = False

    @property
    def matches(self) -> bool:
        return self.ground_truth == self.learned

    @property
    def false_safe(self) -> bool:
        return self.safety_claim and not self.ground_truth and self.learned


@dataclass(frozen=True)
class VerificationReport:
    """Property-level comparison used as the main JEPA verification scorecard."""

    outcomes: tuple[PropertyOutcome, ...]

    def __post_init__(self) -> None:
        if not self.outcomes:
            raise ValueError("A verification report requires at least one outcome.")

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def matched(self) -> int:
        return sum(outcome.matches for outcome in self.outcomes)

    @property
    def agreement(self) -> float:
        return self.matched / self.total

    @property
    def false_safe_count(self) -> int:
        return sum(outcome.false_safe for outcome in self.outcomes)

    @property
    def unsafe_case_count(self) -> int:
        return sum(
            outcome.safety_claim and not outcome.ground_truth
            for outcome in self.outcomes
        )

    @property
    def unsafe_miss_rate(self) -> float | None:
        if self.unsafe_case_count == 0:
            return None
        return self.false_safe_count / self.unsafe_case_count

    @property
    def agreement_by_category(self) -> dict[str, float]:
        categories = {outcome.category for outcome in self.outcomes}
        return {
            category: sum(
                outcome.matches
                for outcome in self.outcomes
                if outcome.category == category
            )
            / sum(
                outcome.category == category for outcome in self.outcomes
            )
            for category in sorted(categories)
        }

    @property
    def agreement_by_property(self) -> dict[str, float]:
        property_names = {outcome.name for outcome in self.outcomes}
        return {
            name: sum(
                outcome.matches
                for outcome in self.outcomes
                if outcome.name == name
            )
            / sum(outcome.name == name for outcome in self.outcomes)
            for name in sorted(property_names)
        }

    @property
    def ground_truth_positive_rate_by_property(self) -> dict[str, float]:
        """Expose formula imbalance instead of hiding it in overall accuracy."""
        property_names = {outcome.name for outcome in self.outcomes}
        return {
            name: sum(
                outcome.ground_truth
                for outcome in self.outcomes
                if outcome.name == name
            )
            / sum(outcome.name == name for outcome in self.outcomes)
            for name in sorted(property_names)
        }


def evaluate_ctl_suite(
    ground_truth_system: ExplicitTransitionSystem[GroundStateT, GroundActionT],
    learned_system: ExplicitTransitionSystem[LearnedStateT, LearnedActionT],
    ground_truth_initial_state: GroundStateT,
    learned_initial_state: LearnedStateT,
    properties: Iterable[CTLProperty] | None = None,
) -> VerificationReport:
    """Compare CTL results on exact and learned transition systems."""
    return evaluate_ctl_suite_for_state_pairs(
        ground_truth_system,
        learned_system,
        ((ground_truth_initial_state, learned_initial_state),),
        properties,
    )


def evaluate_ctl_suite_for_state_pairs(
    ground_truth_system: ExplicitTransitionSystem[GroundStateT, GroundActionT],
    learned_system: ExplicitTransitionSystem[LearnedStateT, LearnedActionT],
    state_pairs: Iterable[tuple[GroundStateT, LearnedStateT]],
    properties: Iterable[CTLProperty] | None = None,
) -> VerificationReport:
    """Compare a property suite over many corresponding states efficiently."""
    suite = tuple(default_ctl_suite() if properties is None else properties)
    if not suite:
        raise ValueError("The CTL property suite must not be empty.")

    pairs = tuple(state_pairs)
    if not pairs:
        raise ValueError("At least one corresponding state pair is required.")

    names = [property_spec.name for property_spec in suite]
    if len(names) != len(set(names)):
        raise ValueError("CTL property names must be unique.")

    ground_truth_checker = CTLModelChecker(ground_truth_system)
    learned_checker = CTLModelChecker(learned_system)
    outcomes = tuple(
        PropertyOutcome(
            name=property_spec.name,
            category=property_spec.category,
            ground_truth=ground_truth_checker.holds(
                ground_truth_state, property_spec.formula
            ),
            learned=learned_checker.holds(
                learned_state, property_spec.formula
            ),
            safety_claim=property_spec.safety_claim,
        )
        for ground_truth_state, learned_state in pairs
        for property_spec in suite
    )
    return VerificationReport(outcomes)


def aggregate_reports(
    reports: Iterable[VerificationReport],
) -> VerificationReport:
    """Combine per-map reports without averaging maps of different sizes."""
    outcomes = tuple(
        outcome for report in reports for outcome in report.outcomes
    )
    return VerificationReport(outcomes)
