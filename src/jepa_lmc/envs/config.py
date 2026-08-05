from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from jepa_lmc.envs.gridworld import GridWorld, State


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty config file: {path}")

    return data


def _to_state(value: list[int]) -> State:
    if len(value) != 2:
        raise ValueError(f"State must have two integers, got: {value}")
    return (int(value[0]), int(value[1]))


def _to_state_set(values: list[list[int]]) -> set[State]:
    return {_to_state(value) for value in values}


def make_gridworld_from_config(config: Mapping[str, Any]) -> GridWorld:
    """Build a GridWorld from an already loaded configuration mapping."""
    env_config = config["env"]
    return GridWorld(
        width=int(env_config["width"]),
        height=int(env_config["height"]),
        start=_to_state(env_config["start"]),
        goal=_to_state(env_config["goal"]),
        walls=_to_state_set(env_config.get("walls", [])),
        dangers=_to_state_set(env_config.get("dangers", [])),
    )


def load_gridworld(path: str | Path) -> GridWorld:
    """Load a YAML file and construct its GridWorld in one call."""
    return make_gridworld_from_config(load_yaml(path))
