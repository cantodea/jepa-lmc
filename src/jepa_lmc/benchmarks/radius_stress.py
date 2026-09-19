"""Fixed topology stress cases, chosen without looking at model predictions."""

from __future__ import annotations

from dataclasses import dataclass

from jepa_lmc.benchmarks.random_gridworld import GridWorldSpec
from jepa_lmc.envs.gridworld import GridWorld, State

PRIMARY_PROPERTIES = ("AG !danger", "EF goal", "E[!danger U goal]")
OPPORTUNITY_FAMILIES = ("sealed_region", "danger_gate")


@dataclass(frozen=True)
class RadiusStressCase:
    name: str
    family: str
    layout: int
    rotation: int
    spec: GridWorldSpec

    def make_env(self) -> GridWorld:
        return self.spec.make_env()


def _rotate(state: State, turns: int) -> State:
    row, col = state
    for _ in range(turns):
        row, col = col, 5 - row
    return row, col


def radius_stress_cases() -> tuple[RadiusStressCase, ...]:
    """Three families x two wall/gate positions x four rotations, always 6x6.

    A full-height wall separates start and goal in sealed_region. danger_gate
    opens exactly one dangerous gate. safe_detour adds a second, safe gate.
    The four rotations are controlled variants, not independent random samples.
    """
    cases = []
    for family in ("sealed_region", "danger_gate", "safe_detour"):
        for layout, (column, gate_row) in enumerate(((2, 1), (3, 4))):
            walls = {(row, column) for row in range(6)}
            if family == "sealed_region":
                dangers = {(2, 5)}
            else:
                walls.remove((gate_row, column))
                dangers = {(gate_row, column)}
                if family == "safe_detour":
                    walls.remove((5 - gate_row, column))
            for rotation in range(4):
                spec = GridWorldSpec(
                    seed=-1000 - len(cases),
                    width=6,
                    height=6,
                    start=_rotate((0, 0), rotation),
                    goal=_rotate((5, 5), rotation),
                    walls=frozenset(_rotate(s, rotation) for s in walls),
                    dangers=frozenset(_rotate(s, rotation) for s in dangers),
                )
                cases.append(
                    RadiusStressCase(
                        name=f"{family}_layout{layout}_rot{rotation * 90}",
                        family=family,
                        layout=layout,
                        rotation=rotation,
                        spec=spec,
                    )
                )
    return tuple(cases)


def label_immediate(env: GridWorld, state: State, property_name: str) -> bool:
    """Exclude verdicts forced by the source label for the existing CTL suite.

    Goal is safe in the verification labels. This marks only direct base cases,
    not every possible logical simplification of a formula.
    """
    if property_name in ("EF goal", "AF goal"):
        return env.is_goal(state)
    if property_name == "E[!danger U goal]":
        return env.is_goal(state) or env.is_danger(state)
    if property_name in ("AG !danger", "EF danger", "EG safe"):
        return env.is_danger(state)
    raise ValueError(f"Unknown stress property: {property_name}")


def exact_transfer_opportunity(outcome: dict) -> bool:
    """Would an exact candidate graph support a non-immediate one-sided claim?"""
    if outcome["label_immediate"] or outcome["property"] not in PRIMARY_PROPERTIES:
        return False
    direction = outcome["monotonicity"]
    truth = outcome["ground_truth"]
    return (direction == -1 and truth) or (direction == 1 and not truth)
