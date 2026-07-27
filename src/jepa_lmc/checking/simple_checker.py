from __future__ import annotations

from typing import Dict, List

from jepa_lmc.checking.ctl import AG, EF, EU, Atom, CTLModelChecker, Not
from jepa_lmc.checking.gridworld_adapter import (
    gridworld_to_transition_system,
)
from jepa_lmc.checking.witness import find_ef_witness, find_eu_witness
from jepa_lmc.envs.gridworld import GridWorld, State


def _validate_initial_state(env: GridWorld, initial_state: State) -> None:
    if not env.is_valid_state(initial_state):
        raise ValueError(f"Invalid initial state: {initial_state}")


def find_path_to_label(
    env: GridWorld,
    initial_state: State,
    target_label: str,
    avoid_label: str | None = None,
) -> List[State] | None:
    """
    Find one shortest path from the initial state to a state with target_label.

    If avoid_label is provided, states with that label are not visited unless the
    initial state already has the target label. The returned path includes both
    the initial state and the target state. None means no such path exists.
    """
    _validate_initial_state(env, initial_state)
    checker = CTLModelChecker(gridworld_to_transition_system(env))
    target = Atom(target_label)

    if avoid_label is None:
        witness = find_ef_witness(checker, initial_state, target)
    else:
        witness = find_eu_witness(
            checker,
            initial_state,
            condition=Not(Atom(avoid_label)),
            target=target,
        )

    return None if witness is None else list(witness.states)


def is_label_reachable(
    env: GridWorld,
    initial_state: State,
    target_label: str,
    avoid_label: str | None = None,
) -> bool:
    """Return whether a target label is reachable from the initial state."""
    return find_path_to_label(env, initial_state, target_label, avoid_label) is not None


def check_basic_properties(env: GridWorld, initial_state: State) -> Dict[str, bool]:
    """
    Check the basic CTL-fragment properties on the real GridWorld graph.

    Supported properties:
    - EF danger: some reachable path reaches danger.
    - EF goal: some reachable path reaches the goal.
    - E[!danger U goal]: some reachable path reaches goal while avoiding danger.
    - AG !danger: all reachable paths avoid danger; equivalently, danger is not
      reachable in this finite transition system.
    """
    _validate_initial_state(env, initial_state)
    checker = CTLModelChecker(gridworld_to_transition_system(env))
    danger = Atom("danger")
    goal = Atom("goal")
    not_danger = Not(danger)

    return {
        "EF danger": checker.holds(initial_state, EF(danger)),
        "EF goal": checker.holds(initial_state, EF(goal)),
        "E[!danger U goal]": checker.holds(
            initial_state, EU(not_danger, goal)
        ),
        "AG !danger": checker.holds(initial_state, AG(not_danger)),
    }
