from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from jepa_lmc.checking.gridworld_adapter import gridworld_to_transition_system
from jepa_lmc.checking.transition_system import ExplicitTransitionSystem
from jepa_lmc.data.jepa_transitions import gridworld_observation
from jepa_lmc.envs.gridworld import GridWorld, State
from jepa_lmc.models.action_jepa import ActionJEPA


@dataclass(frozen=True)
class TransitionRetrieval:
    """One decoded JEPA transition and the rank of its true destination."""

    state: State
    action: int
    true_next_state: State
    predicted_next_state: State
    true_rank: int
    predicted_distance: float
    true_distance: float
    kind: str
    true_next_is_danger: bool
    predicted_next_is_danger: bool

    @property
    def correct(self) -> bool:
        return self.predicted_next_state == self.true_next_state


@dataclass(frozen=True)
class TransitionRetrievalReport:
    """Closed-set next-state retrieval metrics for one or more maps."""

    outcomes: tuple[TransitionRetrieval, ...]

    def __post_init__(self) -> None:
        if not self.outcomes:
            raise ValueError("A transition retrieval report requires outcomes.")

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def top1_accuracy(self) -> float:
        return sum(outcome.correct for outcome in self.outcomes) / self.total

    def top_k_accuracy(self, k: int) -> float:
        if k <= 0:
            raise ValueError("k must be positive.")
        return sum(outcome.true_rank <= k for outcome in self.outcomes) / self.total

    @property
    def mean_reciprocal_rank(self) -> float:
        return sum(1.0 / outcome.true_rank for outcome in self.outcomes) / self.total

    @property
    def accuracy_by_kind(self) -> dict[str, float]:
        kinds = {outcome.kind for outcome in self.outcomes}
        return {
            kind: sum(
                outcome.correct for outcome in self.outcomes if outcome.kind == kind
            )
            / sum(outcome.kind == kind for outcome in self.outcomes)
            for kind in sorted(kinds)
        }

    @property
    def unsafe_miss_count(self) -> int:
        return sum(
            outcome.true_next_is_danger and not outcome.predicted_next_is_danger
            for outcome in self.outcomes
        )

    @property
    def danger_destination_recall(self) -> float | None:
        danger_count = sum(outcome.true_next_is_danger for outcome in self.outcomes)
        if danger_count == 0:
            return None
        return 1.0 - self.unsafe_miss_count / danger_count

    @property
    def false_danger_rate(self) -> float | None:
        safe_count = sum(not outcome.true_next_is_danger for outcome in self.outcomes)
        if safe_count == 0:
            return None
        false_danger = sum(
            not outcome.true_next_is_danger and outcome.predicted_next_is_danger
            for outcome in self.outcomes
        )
        return false_danger / safe_count


@dataclass(frozen=True)
class RolloutRetrieval:
    """Decoded state at one horizon of an open-loop latent rollout."""

    initial_state: State
    actions: tuple[int, ...]
    horizon: int
    true_state: State
    predicted_state: State

    @property
    def correct(self) -> bool:
        return self.true_state == self.predicted_state


@dataclass(frozen=True)
class RolloutReport:
    outcomes: tuple[RolloutRetrieval, ...]

    def __post_init__(self) -> None:
        if not self.outcomes:
            raise ValueError("A rollout report requires outcomes.")

    @property
    def accuracy_by_horizon(self) -> dict[int, float]:
        horizons = {outcome.horizon for outcome in self.outcomes}
        return {
            horizon: sum(
                outcome.correct
                for outcome in self.outcomes
                if outcome.horizon == horizon
            )
            / sum(outcome.horizon == horizon for outcome in self.outcomes)
            for horizon in sorted(horizons)
        }


def _state_observations(env: GridWorld, states: tuple[State, ...]) -> Tensor:
    return torch.stack(tuple(gridworld_observation(env, state) for state in states))


def _transition_kind(env: GridWorld, state: State, next_state: State) -> str:
    if next_state == state:
        return "self_loop"
    if env.is_danger(next_state) and not env.is_danger(state):
        return "danger_entry"
    if env.is_goal(next_state) and not env.is_goal(state):
        return "goal_entry"
    return "movement"


@torch.no_grad()
def evaluate_gridworld_transitions(
    model: ActionJEPA,
    env: GridWorld,
    *,
    device: torch.device | str = "cpu",
    action_override: int | None = None,
) -> TransitionRetrievalReport:
    """Decode every state-action prediction by target-latent nearest neighbour.

    ``action_override`` masks the real action with one fixed action and provides an
    action-ablation baseline without retraining or changing candidate states.
    """
    if action_override is not None and action_override not in env.ACTIONS:
        raise ValueError("The action override is outside the environment action space.")

    model.to(device)
    model.eval()
    states = tuple(env.all_states())
    state_to_index = {state: index for index, state in enumerate(states)}
    observations = _state_observations(env, states).to(device)
    target_candidates = model.target_encoder(observations)
    context = model.context_encoder(observations)

    source_indices = torch.arange(len(states), device=device).repeat_interleave(
        len(env.ACTIONS)
    )
    actions = torch.tensor(
        tuple(action for _state in states for action in env.ACTIONS),
        dtype=torch.long,
        device=device,
    )
    predictor_actions = (
        actions
        if action_override is None
        else torch.full_like(actions, action_override)
    )
    predictions = model.predictor(context[source_indices], predictor_actions)
    distances = torch.cdist(predictions, target_candidates)
    ordering = distances.argsort(dim=1)

    outcomes: list[TransitionRetrieval] = []
    for row, (state, action) in enumerate(
        (state, action) for state in states for action in env.ACTIONS
    ):
        true_next_state = env.transition(state, action)
        true_index = state_to_index[true_next_state]
        predicted_index = int(ordering[row, 0].item())
        true_rank = (
            int((ordering[row] == true_index).nonzero(as_tuple=False)[0, 0].item()) + 1
        )
        predicted_next_state = states[predicted_index]
        outcomes.append(
            TransitionRetrieval(
                state=state,
                action=action,
                true_next_state=true_next_state,
                predicted_next_state=predicted_next_state,
                true_rank=true_rank,
                predicted_distance=float(distances[row, predicted_index].item()),
                true_distance=float(distances[row, true_index].item()),
                kind=_transition_kind(env, state, true_next_state),
                true_next_is_danger=env.is_danger(true_next_state),
                predicted_next_is_danger=env.is_danger(predicted_next_state),
            )
        )
    return TransitionRetrievalReport(tuple(outcomes))


def aggregate_transition_reports(
    reports: Iterable[TransitionRetrievalReport],
) -> TransitionRetrievalReport:
    outcomes = tuple(outcome for report in reports for outcome in report.outcomes)
    return TransitionRetrievalReport(outcomes)


def aggregate_rollout_reports(reports: Iterable[RolloutReport]) -> RolloutReport:
    outcomes = tuple(outcome for report in reports for outcome in report.outcomes)
    return RolloutReport(outcomes)


def transition_system_from_retrievals(
    env: GridWorld,
    report: TransitionRetrievalReport,
) -> ExplicitTransitionSystem[State, int]:
    """Build a learned Kripke structure while retaining exact state labels."""
    states = tuple(env.all_states())
    expected_keys = {(state, action) for state in states for action in env.ACTIONS}
    predictions: dict[tuple[State, int], State] = {}
    for outcome in report.outcomes:
        key = (outcome.state, outcome.action)
        if key in predictions:
            raise ValueError(f"Duplicate predicted transition: {key!r}")
        predictions[key] = outcome.predicted_next_state
    if set(predictions) != expected_keys:
        missing = expected_keys - set(predictions)
        extra = set(predictions) - expected_keys
        raise ValueError(
            f"Predicted transitions do not match the environment; "
            f"missing={len(missing)}, extra={len(extra)}."
        )

    exact_system = gridworld_to_transition_system(env)
    return ExplicitTransitionSystem(
        states=states,
        initial_states=[env.start],
        transitions={
            state: tuple(
                (action, predictions[(state, action)]) for action in env.ACTIONS
            )
            for state in states
        },
        labels={state: exact_system.propositions(state) for state in states},
    )


@torch.no_grad()
def evaluate_open_loop_rollouts(
    model: ActionJEPA,
    env: GridWorld,
    *,
    horizons: tuple[int, ...] = (1, 2, 4, 8),
    rollouts_per_state: int = 2,
    seed: int = 20260804,
    device: torch.device | str = "cpu",
) -> RolloutReport:
    """Recursively predict latent states without decoding between steps."""
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("Rollout horizons must contain positive integers.")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Rollout horizons must be unique.")
    if rollouts_per_state <= 0:
        raise ValueError("rollouts_per_state must be positive.")

    sorted_horizons = tuple(sorted(horizons))
    maximum_horizon = sorted_horizons[-1]
    model.to(device)
    model.eval()
    states = tuple(env.all_states())
    observations = _state_observations(env, states).to(device)
    target_candidates = model.target_encoder(observations)
    context_candidates = model.context_encoder(observations)
    state_to_index = {state: index for index, state in enumerate(states)}
    rng = random.Random(seed)
    outcomes: list[RolloutRetrieval] = []

    for initial_state in states:
        for _ in range(rollouts_per_state):
            actions = tuple(
                rng.choice(tuple(env.ACTIONS)) for _step in range(maximum_horizon)
            )
            true_state = initial_state
            predicted_latent = context_candidates[
                state_to_index[initial_state]
            ].unsqueeze(0)
            for step, action in enumerate(actions, start=1):
                action_tensor = torch.tensor([action], dtype=torch.long, device=device)
                predicted_latent = model.predictor(predicted_latent, action_tensor)
                true_state = env.transition(true_state, action)
                if step in sorted_horizons:
                    distance = torch.cdist(predicted_latent, target_candidates)
                    predicted_state = states[int(distance.argmin(dim=1).item())]
                    outcomes.append(
                        RolloutRetrieval(
                            initial_state=initial_state,
                            actions=actions[:step],
                            horizon=step,
                            true_state=true_state,
                            predicted_state=predicted_state,
                        )
                    )
    return RolloutReport(tuple(outcomes))
