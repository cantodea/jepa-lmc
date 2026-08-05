from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT", bound=Hashable)


@dataclass(frozen=True)
class TransitionEdge(Generic[StateT, ActionT]):
    action: ActionT
    target: StateT


class ExplicitTransitionSystem(Generic[StateT, ActionT]):
    """Finite, total, action-labelled transition system."""

    def __init__(
        self,
        *,
        states: Iterable[StateT],
        initial_states: Iterable[StateT],
        transitions: Mapping[StateT, Iterable[tuple[ActionT, StateT]]],
        labels: Mapping[StateT, Iterable[str]],
    ) -> None:
        self._states = frozenset(states)
        self._initial_states = frozenset(initial_states)

        if not self._states:
            raise ValueError("A transition system must contain at least one state.")

        if not self._initial_states:
            raise ValueError("A transition system must contain an initial state.")

        unknown_initial_states = self._initial_states - self._states
        if unknown_initial_states:
            raise ValueError(
                f"Initial states are not in the state space: {unknown_initial_states}"
            )

        normalized_transitions: dict[
            StateT, tuple[TransitionEdge[StateT, ActionT], ...]
        ] = {}

        for state in self._states:
            edges = tuple(
                TransitionEdge(action=action, target=target)
                for action, target in transitions.get(state, ())
            )

            if not edges:
                raise ValueError(
                    f"State {state!r} has no successor. "
                    "CTL Kripke structures require a total transition relation."
                )

            for edge in edges:
                if edge.target not in self._states:
                    raise ValueError(
                        f"Transition from {state!r} targets unknown state "
                        f"{edge.target!r}."
                    )

            normalized_transitions[state] = edges

        unknown_sources = set(transitions) - self._states
        if unknown_sources:
            raise ValueError(
                f"Transitions contain unknown source states: {unknown_sources}"
            )

        unknown_labelled_states = set(labels) - self._states
        if unknown_labelled_states:
            raise ValueError(
                f"Labels contain unknown states: {unknown_labelled_states}"
            )

        self._transitions = normalized_transitions
        self._labels = {
            state: frozenset(labels.get(state, ())) for state in self._states
        }

    @property
    def states(self) -> frozenset[StateT]:
        return self._states

    @property
    def initial_states(self) -> frozenset[StateT]:
        return self._initial_states

    def action_successors(
        self, state: StateT
    ) -> tuple[TransitionEdge[StateT, ActionT], ...]:
        self.validate_state(state)
        return self._transitions[state]

    def successors(self, state: StateT) -> frozenset[StateT]:
        return frozenset(edge.target for edge in self.action_successors(state))

    def propositions(self, state: StateT) -> frozenset[str]:
        self.validate_state(state)
        return self._labels[state]

    def validate_state(self, state: StateT) -> None:
        if state not in self._states:
            raise ValueError(f"Unknown state: {state!r}")
