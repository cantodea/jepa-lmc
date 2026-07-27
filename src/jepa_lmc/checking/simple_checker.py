from __future__ import annotations

from collections import deque
from typing import Dict, List

from jepa_lmc.envs.gridworld import GridWorld, State


def _successors(env: GridWorld, state: State) -> List[State]:
    """Return deterministic successor states for all available actions."""
    return [env.transition(state, action) for action in env.ACTIONS]


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

    if env.label(initial_state) == target_label:
        return [initial_state]

    if avoid_label is not None and env.label(initial_state) == avoid_label:
        return None

    queue: deque[State] = deque([initial_state])
    parents: Dict[State, State | None] = {initial_state: None}

    while queue:
        state = queue.popleft()

        for next_state in _successors(env, state):
            if next_state in parents:
                continue

            next_label = env.label(next_state)
            if avoid_label is not None and next_label == avoid_label:
                continue

            parents[next_state] = state

            if next_label == target_label:
                return _reconstruct_path(parents, next_state)

            queue.append(next_state)

    return None


def _reconstruct_path(parents: Dict[State, State | None], target: State) -> List[State]:
    path: List[State] = []
    current: State | None = target

    while current is not None:
        path.append(current)
        current = parents[current]

    path.reverse()
    return path


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
    ef_danger = is_label_reachable(env, initial_state, "danger")
    ef_goal = is_label_reachable(env, initial_state, "goal")
    safe_until_goal = is_label_reachable(
        env,
        initial_state,
        target_label="goal",
        avoid_label="danger",
    )

    return {
        "EF danger": ef_danger,
        "EF goal": ef_goal,
        "E[!danger U goal]": safe_until_goal,
        "AG !danger": not ef_danger,
    }