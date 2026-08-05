from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

State = tuple[int, int]


@dataclass(frozen=True)
class Transition:
    state: State
    action: int
    next_state: State


class GridWorld:
    ACTIONS: ClassVar[dict[int, tuple[int, int]]] = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    ACTION_NAMES: ClassVar[dict[int, str]] = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
    }

    def __init__(
        self,
        width: int,
        height: int,
        start: State,
        goal: State,
        walls: set[State],
        dangers: set[State],
    ) -> None:
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.walls = set(walls)
        self.dangers = set(dangers)
        self._validate()

    def _validate(self) -> None:
        all_special = [self.start, self.goal] + list(self.walls) + list(self.dangers)

        for s in all_special:
            if not self.in_bounds(s):
                raise ValueError(f"State {s} is outside the grid.")

        if self.start in self.walls:
            raise ValueError("Start cannot be a wall.")

        if self.goal in self.walls:
            raise ValueError("Goal cannot be a wall.")

        if self.start in self.dangers:
            raise ValueError("Start cannot be danger.")

        if self.goal in self.dangers:
            raise ValueError("Goal cannot be danger.")

    def in_bounds(self, state: State) -> bool:
        row, col = state
        return 0 <= row < self.height and 0 <= col < self.width

    def is_wall(self, state: State) -> bool:
        return state in self.walls

    def is_danger(self, state: State) -> bool:
        return state in self.dangers

    def is_goal(self, state: State) -> bool:
        return state == self.goal

    def is_valid_state(self, state: State) -> bool:
        return self.in_bounds(state) and not self.is_wall(state)

    def all_states(self) -> list[State]:
        states: list[State] = []

        for row in range(self.height):
            for col in range(self.width):
                s = (row, col)
                if self.is_valid_state(s):
                    states.append(s)

        return states

    def transition(self, state: State, action: int) -> State:
        if not self.is_valid_state(state):
            raise ValueError(f"Invalid state: {state}")

        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        dr, dc = self.ACTIONS[action]
        row, col = state
        next_state = (row + dr, col + dc)

        if not self.is_valid_state(next_state):
            return state

        return next_state

    def all_transitions(self) -> list[Transition]:
        transitions: list[Transition] = []

        for s in self.all_states():
            for a in self.ACTIONS:
                s_next = self.transition(s, a)
                transitions.append(Transition(s, a, s_next))

        return transitions

    def label(self, state: State) -> str:
        if not self.is_valid_state(state):
            raise ValueError(f"Invalid state: {state}")

        if self.is_goal(state):
            return "goal"

        if self.is_danger(state):
            return "danger"

        return "safe"

    def state_to_id(self, state: State) -> int:
        row, col = state
        return row * self.width + col

    def id_to_state(self, state_id: int) -> State:
        row = state_id // self.width
        col = state_id % self.width
        state = (row, col)

        if not self.in_bounds(state):
            raise ValueError(f"Invalid state id: {state_id}")

        return state

    def print_map(self) -> None:
        for row in range(self.height):
            symbols = []

            for col in range(self.width):
                s = (row, col)

                if s == self.start:
                    symbols.append("S")
                elif s == self.goal:
                    symbols.append("G")
                elif s in self.walls:
                    symbols.append("#")
                elif s in self.dangers:
                    symbols.append("D")
                else:
                    symbols.append(".")

            print(" ".join(symbols))
