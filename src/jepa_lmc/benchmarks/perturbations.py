from __future__ import annotations

from jepa_lmc.verification.transition_system import (
    ActionT,
    ExplicitTransitionSystem,
    StateT,
)


def block_entries_to_proposition(
    transition_system: ExplicitTransitionSystem[StateT, ActionT],
    proposition: str,
) -> ExplicitTransitionSystem[StateT, ActionT]:
    """Return a copy that cannot enter states carrying ``proposition``.

    This deliberately biased model is a deterministic stand-in for a learned
    model that misses a safety-critical transition. Every blocked transition is
    redirected to a self-loop at its source so the CTL relation stays total.
    """
    transitions: dict[StateT, tuple[tuple[ActionT, StateT], ...]] = {}
    labels = {
        state: transition_system.propositions(state)
        for state in transition_system.states
    }

    for state in transition_system.states:
        transitions[state] = tuple(
            (
                edge.action,
                state
                if proposition in transition_system.propositions(edge.target)
                else edge.target,
            )
            for edge in transition_system.action_successors(state)
        )

    return ExplicitTransitionSystem(
        states=transition_system.states,
        initial_states=transition_system.initial_states,
        transitions=transitions,
        labels=labels,
    )
