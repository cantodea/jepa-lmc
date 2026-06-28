from __future__ import annotations

from typing import Any, Dict, Set

from jepa_lmc.envs.gridworld import GridWorld, State


def _to_state(x: list[int]) -> State:
    if len(x) != 2:
        raise ValueError(f"State must have two integers, got: {x}")

    return (int(x[0]), int(x[1]))


def _to_state_set(xs: list[list[int]]) -> Set[State]:
    return {_to_state(x) for x in xs}


def make_gridworld_from_config(config: Dict[str, Any]) -> GridWorld:
    env_cfg = config["env"]

    width = int(env_cfg["width"])
    height = int(env_cfg["height"])
    start = _to_state(env_cfg["start"])
    goal = _to_state(env_cfg["goal"])
    walls = _to_state_set(env_cfg.get("walls", []))
    dangers = _to_state_set(env_cfg.get("dangers", []))

    return GridWorld(
        width=width,
        height=height,
        start=start,
        goal=goal,
        walls=walls,
        dangers=dangers,
    )