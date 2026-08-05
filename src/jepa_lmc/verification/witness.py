from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Generic

from jepa_lmc.verification.ctl import CTLModelChecker, Formula
from jepa_lmc.verification.transition_system import (
    ActionT,
    ExplicitTransitionSystem,
    StateT,
    TransitionEdge,
)


@dataclass(frozen=True)
class PathStep(Generic[StateT, ActionT]):
    state: StateT
    action: ActionT
    next_state: StateT


@dataclass(frozen=True)
class PathWitness(Generic[StateT, ActionT]):
    initial_state: StateT
    steps: tuple[PathStep[StateT, ActionT], ...]

    @property
    def states(self) -> tuple[StateT, ...]:
        return (self.initial_state,) + tuple(
            step.next_state for step in self.steps
        )


def find_ef_witness(
    checker: CTLModelChecker[StateT, ActionT],
    initial_state: StateT,
    target: Formula,
) -> PathWitness[StateT, ActionT] | None:
    target_states = checker.satisfying_states(target)
    return _find_path(
        checker.transition_system,
        initial_state,
        target_states=target_states,
        allowed_states=checker.transition_system.states,
    )


def find_eu_witness(
    checker: CTLModelChecker[StateT, ActionT],
    initial_state: StateT,
    condition: Formula,
    target: Formula,
) -> PathWitness[StateT, ActionT] | None:
    condition_states = checker.satisfying_states(condition)
    target_states = checker.satisfying_states(target)
    return _find_path(
        checker.transition_system,
        initial_state,
        target_states=target_states,
        allowed_states=condition_states | target_states,
    )


def find_ag_counterexample(
    checker: CTLModelChecker[StateT, ActionT],
    initial_state: StateT,
    invariant: Formula,
) -> PathWitness[StateT, ActionT] | None:
    violating_states = (
        checker.transition_system.states
        - checker.satisfying_states(invariant)
    )
    return _find_path(
        checker.transition_system,
        initial_state,
        target_states=violating_states,
        allowed_states=checker.transition_system.states,
    )


def _find_path(
    transition_system: ExplicitTransitionSystem[StateT, ActionT],
    initial_state: StateT,
    *,
    target_states: frozenset[StateT],
    allowed_states: frozenset[StateT],
) -> PathWitness[StateT, ActionT] | None:
    transition_system.validate_state(initial_state)

    if initial_state in target_states:
        return PathWitness(initial_state=initial_state, steps=())

    if initial_state not in allowed_states:
        return None

    queue: deque[StateT] = deque([initial_state])
    parents: dict[
        StateT, tuple[StateT, TransitionEdge[StateT, ActionT]] | None
    ] = {initial_state: None}

    while queue:
        state = queue.popleft()

        for edge in transition_system.action_successors(state):
            next_state = edge.target

            if next_state in parents or next_state not in allowed_states:
                continue

            parents[next_state] = (state, edge)

            if next_state in target_states:
                return _reconstruct_path(
                    initial_state, next_state, parents
                )

            queue.append(next_state)

    return None


def _reconstruct_path(
    initial_state: StateT,
    target_state: StateT,
    parents: dict[
        StateT, tuple[StateT, TransitionEdge[StateT, ActionT]] | None
    ],
) -> PathWitness[StateT, ActionT]:
    reversed_steps: list[PathStep[StateT, ActionT]] = []
    current = target_state

    while current != initial_state:
        parent = parents[current]
        if parent is None:
            raise RuntimeError("Broken witness parent chain.")

        previous_state, edge = parent
        reversed_steps.append(
            PathStep(
                state=previous_state,
                action=edge.action,
                next_state=current,
            )
        )
        current = previous_state

    reversed_steps.reverse()
    return PathWitness(
        initial_state=initial_state,
        steps=tuple(reversed_steps),
    )
