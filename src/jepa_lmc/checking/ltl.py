from __future__ import annotations

from dataclasses import dataclass


class Formula:
    """Base class for the supported LTL syntax tree."""


@dataclass(frozen=True)
class Atom(Formula):
    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("An atomic proposition must have a name.")


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
class Next(Formula):
    formula: Formula


@dataclass(frozen=True)
class Eventually(Formula):
    formula: Formula


@dataclass(frozen=True)
class Globally(Formula):
    formula: Formula


@dataclass(frozen=True)
class Until(Formula):
    condition: Formula
    target: Formula


def atoms(formula: Formula) -> frozenset[str]:
    """Return every atomic proposition referenced by an LTL formula."""
    if isinstance(formula, Atom):
        return frozenset((formula.name,))
    if isinstance(formula, (Not, Next, Eventually, Globally)):
        return atoms(formula.formula)
    if isinstance(formula, (And, Or)):
        return atoms(formula.left) | atoms(formula.right)
    if isinstance(formula, Until):
        return atoms(formula.condition) | atoms(formula.target)
    raise TypeError(f"Unsupported LTL formula: {formula!r}")
