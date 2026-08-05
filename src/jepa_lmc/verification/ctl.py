from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic

from jepa_lmc.verification.transition_system import (
    ActionT,
    ExplicitTransitionSystem,
    StateT,
)


class Formula:
    """Base class for CTL state formulae."""


@dataclass(frozen=True)
class Atom(Formula):
    name: str


@dataclass(frozen=True)
class Not(Formula):
    formula: Formula


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class EX(Formula):
    formula: Formula


@dataclass(frozen=True)
class AX(Formula):
    formula: Formula


@dataclass(frozen=True)
class EF(Formula):
    formula: Formula


@dataclass(frozen=True)
class AF(Formula):
    formula: Formula


@dataclass(frozen=True)
class EG(Formula):
    formula: Formula


@dataclass(frozen=True)
class AG(Formula):
    formula: Formula


@dataclass(frozen=True)
class EU(Formula):
    condition: Formula
    target: Formula


class CTLModelChecker(Generic[StateT, ActionT]):
    """Explicit-state CTL model checker using finite fixed points."""

    def __init__(
        self, transition_system: ExplicitTransitionSystem[StateT, ActionT]
    ) -> None:
        self.transition_system = transition_system
        self._cache: dict[Formula, frozenset[StateT]] = {}

    def holds(self, state: StateT, formula: Formula) -> bool:
        self.transition_system.validate_state(state)
        return state in self.satisfying_states(formula)

    def satisfying_states(self, formula: Formula) -> frozenset[StateT]:
        cached = self._cache.get(formula)
        if cached is not None:
            return cached

        result = self._evaluate(formula)
        self._cache[formula] = result
        return result

    def _evaluate(self, formula: Formula) -> frozenset[StateT]:
        states = self.transition_system.states

        if isinstance(formula, Atom):
            return frozenset(
                state
                for state in states
                if formula.name in self.transition_system.propositions(state)
            )

        if isinstance(formula, Not):
            return states - self.satisfying_states(formula.formula)

        if isinstance(formula, And):
            return self.satisfying_states(
                formula.left
            ) & self.satisfying_states(formula.right)

        if isinstance(formula, Or):
            return self.satisfying_states(
                formula.left
            ) | self.satisfying_states(formula.right)

        if isinstance(formula, EX):
            return self._pre_exists(self.satisfying_states(formula.formula))

        if isinstance(formula, AX):
            return self._pre_all(self.satisfying_states(formula.formula))

        if isinstance(formula, EF):
            target = self.satisfying_states(formula.formula)
            return self._least_fixed_point(
                target,
                lambda current: target | self._pre_exists(current),
            )

        if isinstance(formula, AF):
            target = self.satisfying_states(formula.formula)
            return self._least_fixed_point(
                target,
                lambda current: target | self._pre_all(current),
            )

        if isinstance(formula, EG):
            condition = self.satisfying_states(formula.formula)
            return self._greatest_fixed_point(
                condition,
                lambda current: condition & self._pre_exists(current),
            )

        if isinstance(formula, AG):
            condition = self.satisfying_states(formula.formula)
            return self._greatest_fixed_point(
                condition,
                lambda current: condition & self._pre_all(current),
            )

        if isinstance(formula, EU):
            condition = self.satisfying_states(formula.condition)
            target = self.satisfying_states(formula.target)
            return self._least_fixed_point(
                target,
                lambda current: target
                | (condition & self._pre_exists(current)),
            )

        raise TypeError(f"Unsupported CTL formula: {formula!r}")

    def _pre_exists(self, target_states: frozenset[StateT]) -> frozenset[StateT]:
        return frozenset(
            state
            for state in self.transition_system.states
            if self.transition_system.successors(state) & target_states
        )

    def _pre_all(self, target_states: frozenset[StateT]) -> frozenset[StateT]:
        return frozenset(
            state
            for state in self.transition_system.states
            if self.transition_system.successors(state) <= target_states
        )

    @staticmethod
    def _least_fixed_point(
        initial: frozenset[StateT],
        update: Callable[[frozenset[StateT]], frozenset[StateT]],
    ) -> frozenset[StateT]:
        current = initial
        while True:
            next_states = update(current)
            if next_states == current:
                return current
            current = next_states

    @staticmethod
    def _greatest_fixed_point(
        initial: frozenset[StateT],
        update: Callable[[frozenset[StateT]], frozenset[StateT]],
    ) -> frozenset[StateT]:
        current = initial
        while True:
            next_states = update(current)
            if next_states == current:
                return current
            current = next_states
