from __future__ import annotations

from dataclasses import dataclass

from jepa_lmc.verification.ctl import AF, AG, EF, EG, EU, Atom, Formula, Not


@dataclass(frozen=True)
class CTLProperty:
    """A named CTL property and the role it plays in evaluation."""

    name: str
    category: str
    formula: Formula
    safety_claim: bool = False
    primary_score: bool = True


def default_ctl_suite() -> tuple[CTLProperty, ...]:
    """Return the fixed property suite shared by exact and learned models."""
    danger = Atom("danger")
    goal = Atom("goal")
    safe = Atom("safe")
    not_danger = Not(danger)

    return (
        CTLProperty(
            "EF danger",
            "reachability",
            EF(danger),
            primary_score=False,
        ),
        CTLProperty("EF goal", "reachability", EF(goal)),
        CTLProperty(
            "E[!danger U goal]",
            "constrained_reachability",
            EU(not_danger, goal),
        ),
        CTLProperty(
            "AG !danger",
            "safety",
            AG(not_danger),
            safety_claim=True,
        ),
        CTLProperty(
            "AF goal",
            "inevitability",
            AF(goal),
            primary_score=False,
        ),
        CTLProperty(
            "EG safe",
            "persistence",
            EG(safe),
            primary_score=False,
        ),
    )
