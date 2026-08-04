from __future__ import annotations

from dataclasses import dataclass

from jepa_lmc.checking.ltl import (
    Atom,
    Eventually,
    Formula,
    Globally,
    Next,
    Not,
    Until,
)


@dataclass(frozen=True)
class LTLProperty:
    """A named universal-path LTL property used for evaluation."""

    name: str
    category: str
    formula: Formula
    safety_claim: bool = False
    primary_score: bool = True


def default_ltl_suite() -> tuple[LTLProperty, ...]:
    """Return post-training properties that are never used as JEPA labels.

    ``F goal`` and ``G !danger`` deliberately duplicate CTL ``AF goal`` and
    ``AG !danger`` as backend consistency diagnostics, so they are excluded
    from the primary LTL score.
    """
    danger = Atom("danger")
    goal = Atom("goal")
    safe = Atom("safe")
    not_danger = Not(danger)

    return (
        LTLProperty(
            "X !danger",
            "next_step_safety",
            Next(not_danger),
            safety_claim=True,
        ),
        LTLProperty(
            "F goal",
            "universal_reachability",
            Eventually(goal),
            primary_score=False,
        ),
        LTLProperty(
            "G !danger",
            "safety",
            Globally(not_danger),
            safety_claim=True,
            primary_score=False,
        ),
        LTLProperty(
            "safe U goal",
            "universal_until",
            Until(safe, goal),
        ),
        LTLProperty(
            "G F safe",
            "recurrence",
            Globally(Eventually(safe)),
        ),
        LTLProperty(
            "F G !danger",
            "stabilization",
            Eventually(Globally(not_danger)),
        ),
    )
