from __future__ import annotations

from jepa_lmc.envs.gridworld import GridWorld, State
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem


def gridworld_to_transition_system(
    env: GridWorld,
) -> ExplicitTransitionSystem[State, int]:
    """Convert a GridWorld into the Kripke structure checked by CTL."""
    states = env.all_states()
    transitions = {
        state: tuple(
            (action, env.transition(state, action)) for action in env.ACTIONS
        )
        for state in states
    }

    labels: dict[State, set[str]] = {}
    for state in states:
        propositions = {"danger"} if env.is_danger(state) else {"safe"}
        if env.is_goal(state):
            propositions.add("goal")
        labels[state] = propositions

    return ExplicitTransitionSystem(
        states=states,
        initial_states=[env.start],
        transitions=transitions,
        labels=labels,
    )
