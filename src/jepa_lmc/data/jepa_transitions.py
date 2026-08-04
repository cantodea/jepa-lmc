from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from jepa_lmc.benchmarks.random_gridworld import GridWorldSpec
from jepa_lmc.envs.gridworld import GridWorld, State


WALL_CHANNEL = 0
DANGER_CHANNEL = 1
GOAL_CHANNEL = 2
AGENT_CHANNEL = 3
OBSERVATION_CHANNELS = 4


def gridworld_observation(env: GridWorld, state: State) -> Tensor:
    """Encode one complete GridWorld state as a four-channel float tensor."""
    if not env.is_valid_state(state):
        raise ValueError(f"Invalid observation state: {state}")

    observation = torch.zeros(
        (OBSERVATION_CHANNELS, env.height, env.width),
        dtype=torch.float32,
    )
    for row, col in env.walls:
        observation[WALL_CHANNEL, row, col] = 1.0
    for row, col in env.dangers:
        observation[DANGER_CHANNEL, row, col] = 1.0

    goal_row, goal_col = env.goal
    observation[GOAL_CHANNEL, goal_row, goal_col] = 1.0
    state_row, state_col = state
    observation[AGENT_CHANNEL, state_row, state_col] = 1.0
    return observation


class GridWorldTransitionDataset(Dataset[dict[str, Tensor]]):
    """All action-labelled transitions from a collection of benchmark maps."""

    def __init__(self, specs: Sequence[GridWorldSpec]) -> None:
        if not specs:
            raise ValueError("At least one GridWorld specification is required.")

        self._examples: list[tuple[GridWorld, State, int, State]] = []
        for spec in specs:
            env = spec.make_env()
            self._examples.extend(
                (env, transition.state, transition.action, transition.next_state)
                for transition in env.all_transitions()
            )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        env, state, action, next_state = self._examples[index]
        return {
            "observation": gridworld_observation(env, state),
            "action": torch.tensor(action, dtype=torch.long),
            "next_observation": gridworld_observation(env, next_state),
        }
