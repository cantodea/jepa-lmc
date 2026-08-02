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
    primary_score: bool = True

    @property
    def matches(self) -> bool:
        return self.ground_truth == self.learned

    @property
    def false_safe(self) -> bool:
        return self.safety_claim and not self.ground_truth and self.learned


@dataclass(frozen=True)
class BinaryConfusion:
    """Confusion counts for one CTL formula across benchmark cases."""

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def positive_recall(self) -> float | None:
        denominator = self.true_positive + self.false_negative
        if denominator == 0:
            return None
        return self.true_positive / denominator

    @property
    def negative_recall(self) -> float | None:
        denominator = self.true_negative + self.false_positive
        if denominator == 0:
            return None
        return self.true_negative / denominator

    @property
    def balanced_accuracy(self) -> float | None:
        positive_recall = self.positive_recall
        negative_recall = self.negative_recall
        if positive_recall is None or negative_recall is None:
            return None
        return (positive_recall + negative_recall) / 2


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
    def confusion_by_property(self) -> dict[str, BinaryConfusion]:
        property_names = {outcome.name for outcome in self.outcomes}
        return {
            name: BinaryConfusion(
                true_positive=sum(
                    outcome.ground_truth and outcome.learned
                    for outcome in self.outcomes
                    if outcome.name == name
                ),
                true_negative=sum(
                    not outcome.ground_truth and not outcome.learned
                    for outcome in self.outcomes
                    if outcome.name == name
                ),
                false_positive=sum(
                    not outcome.ground_truth and outcome.learned
                    for outcome in self.outcomes
                    if outcome.name == name
                ),
                false_negative=sum(
                    outcome.ground_truth and not outcome.learned
                    for outcome in self.outcomes
                    if outcome.name == name
                ),
            )
            for name in sorted(property_names)
        }

    @property
    def primary_balanced_score(self) -> float | None:
        """Macro balanced accuracy over non-duplicate primary properties."""
        primary_names = {
            outcome.name for outcome in self.outcomes if outcome.primary_score
        }
        if not primary_names:
            return None

        confusion = self.confusion_by_property
        scores = [confusion[name].balanced_accuracy for name in primary_names]
        if any(score is None for score in scores):
            return None
        return sum(score for score in scores if score is not None) / len(scores)

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
            primary_score=property_spec.primary_score,
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
