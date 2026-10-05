"""Finite trace samples and non-neural baselines for the known-state setting.

Only the sampling/corruption/validation functions receive the real environment.
The baseline constructors receive S, I, A, L and observed edges, never hidden R.
Partial relations are sets of triples and are not ExplicitTransitionSystems.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from jepa_lmc.envs.gridworld import GridWorld, State
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem

Edge = tuple[State, int, State]


@dataclass(frozen=True)
class Trace:
    states: tuple[State, ...]
    actions: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.states) != len(self.actions) + 1:
            raise ValueError("A trace needs one more state than actions.")

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(zip(self.states[:-1], self.actions, self.states[1:], strict=True))


@dataclass(frozen=True)
class KnownProblem:
    """The learner's catalogue; deliberately contains no transition oracle."""

    states: tuple[State, ...]
    initial_states: frozenset[State]
    actions: tuple[int, ...]
    labels: Mapping[State, frozenset[str]]

    def accepts_syntax(self, trace: Trace) -> bool:
        return (
            trace.states[0] in self.initial_states
            and all(s in self.states for s in trace.states)
            and all(a in self.actions for a in trace.actions)
        )


@dataclass(frozen=True)
class HardNegative:
    trace: Trace
    corrupted_step: int
    true_successor: State
    inserted_successor: State
    same_proposition_labels: bool


def sample_good_traces(
    env: GridWorld, *, seed: int, count: int = 16, length: int = 16
) -> tuple[Trace, ...]:
    """Oracle-side sampling, uniformly over actions, resetting to I each time.

    Length counts transitions (16 actions and 17 states). Goal/danger do not
    terminate sampling: the existing GridWorld transition semantics are retained.
    A longer call with the same seed has the shorter call as an exact prefix.
    """
    if count < 1 or length < 1:
        raise ValueError("Trace count and length must be positive.")
    rng = random.Random(seed)
    actions = tuple(sorted(env.ACTIONS))
    traces = []
    for _ in range(count):
        states, sampled = [env.start], []
        for _ in range(length):
            action = rng.choice(actions)
            sampled.append(action)
            states.append(env.transition(states[-1], action))
        traces.append(Trace(tuple(states), tuple(sampled)))
    return tuple(traces)


def is_execution(env: GridWorld, trace: Trace) -> bool:
    """Oracle-side execution check, including the designated initial state."""
    return (
        trace.states[0] == env.start
        and all(env.is_valid_state(s) for s in trace.states)
        and all(a in env.ACTIONS for a in trace.actions)
        and all(env.transition(s, a) == t for s, a, t in trace.edges)
    )


def make_hard_negatives(
    env: GridWorld,
    goods: Sequence[Trace],
    labels: Mapping[State, frozenset[str]],
    *,
    seed: int,
) -> tuple[HardNegative, ...]:
    """Insert exactly one false edge, then re-execute the remaining actions.

    Prefer a replacement with the *full* same AP set (goal is also safe).
    Fall back to any other legal state only when that label class is a singleton.
    Neither these oracle calls nor validation results go into baseline fitting.
    """
    rng = random.Random(seed)
    states = tuple(sorted(env.all_states()))
    if len(states) < 2:
        raise ValueError("A hard negative requires at least two legal states.")
    negatives = []
    for good in goods:
        if not good.actions or not is_execution(env, good):
            raise ValueError("Corruption requires a nonempty real execution.")
        step = rng.randrange(len(good.actions))
        true = good.states[step + 1]
        candidates = [s for s in states if s != true]
        same_labels = [s for s in candidates if labels[s] == labels[true]]
        replacement = rng.choice(same_labels or candidates)
        replay = [*good.states[: step + 1], replacement]
        for action in good.actions[step + 1 :]:
            replay.append(env.transition(replay[-1], action))
        negatives.append(
            HardNegative(
                Trace(tuple(replay), good.actions),
                step,
                true,
                replacement,
                labels[replacement] == labels[true],
            )
        )
    return tuple(negatives)


def validate_trace_pairs(
    env: GridWorld, goods: Sequence[Trace], negatives: Sequence[HardNegative]
) -> None:
    """Fail before evaluation if the samples violate the generation contract."""
    if not goods or len(goods) != len(negatives):
        raise ValueError("Need one negative per good trace, with a nonempty pool.")
    for good, negative in zip(goods, negatives, strict=True):
        bad, step = negative.trace, negative.corrupted_step
        if not is_execution(env, good):
            raise ValueError("Good trace is not a real execution.")
        if not 0 <= step < len(good.actions):
            raise ValueError("Corrupted step is outside the trace.")
        if (
            bad.actions != good.actions
            or bad.states[: step + 1] != good.states[: step + 1]
            or not all(env.is_valid_state(s) for s in bad.states)
        ):
            raise ValueError(
                "Bad trace changed the prefix/actions or has invalid states."
            )
        false_steps = [
            i for i, (s, a, t) in enumerate(bad.edges) if env.transition(s, a) != t
        ]
        if (
            false_steps != [step]
            or negative.true_successor != good.states[step + 1]
            or negative.inserted_successor != bad.states[step + 1]
            or negative.inserted_successor == negative.true_successor
            or is_execution(env, bad)
        ):
            raise ValueError(
                "Bad trace must have exactly the recorded false transition."
            )


def observed_edges(goods: Sequence[Trace]) -> frozenset[Edge]:
    """The learner: union the supplied good edges; no access to R or B."""
    return frozenset(edge for trace in goods for edge in trace.edges)


def accepts_trace(
    problem: KnownProblem, relation: frozenset[Edge], trace: Trace
) -> bool:
    """Finite trace membership in an action-labelled, possibly partial relation."""
    return problem.accepts_syntax(trace) and all(e in relation for e in trace.edges)


def sample_consistency(
    problem: KnownProblem,
    relation: frozenset[Edge],
    goods: Sequence[Trace],
    bads: Sequence[Trace],
) -> dict:
    if not goods or not bads:
        raise ValueError("Sample rates need nonempty good and bad samples.")
    accepted = sum(accepts_trace(problem, relation, t) for t in goods)
    rejected = sum(not accepts_trace(problem, relation, t) for t in bads)
    return {
        "good_accepted": accepted,
        "good_count": len(goods),
        "good_acceptance": accepted / len(goods),
        "bad_rejected": rejected,
        "bad_count": len(bads),
        "bad_rejection": rejected / len(bads),
    }


def relation_edges(system: ExplicitTransitionSystem) -> frozenset[Edge]:
    return frozenset(
        (s, edge.action, edge.target)
        for s in system.states
        for edge in system.action_successors(s)
    )


def observed_self_loop_completion(
    problem: KnownProblem, relation: frozenset[Edge]
) -> ExplicitTransitionSystem:
    """Evaluation-only totalization: unseen (s,a) gets (s,a,s).

    This does not modify the partial observed-edge relation and is not a claim
    about what the learner inferred. Require observed action determinism.
    """
    targets = {}
    for s, a, t in sorted(relation):
        if (
            s not in problem.states
            or t not in problem.states
            or a not in problem.actions
        ):
            raise ValueError("Observed edge is outside the known catalogue.")
        if (s, a) in targets and targets[s, a] != t:
            raise ValueError("Observed relation is not action-deterministic.")
        targets[s, a] = t
    return ExplicitTransitionSystem(
        states=problem.states,
        initial_states=problem.initial_states,
        labels=problem.labels,
        transitions={
            s: [(a, targets.get((s, a), s)) for a in problem.actions]
            for s in problem.states
        },
    )


def maximally_permissive(problem: KnownProblem) -> ExplicitTransitionSystem:
    """S x A x S: constructed only from the known catalogue."""
    return ExplicitTransitionSystem(
        states=problem.states,
        initial_states=problem.initial_states,
        labels=problem.labels,
        transitions={
            s: [(a, t) for a in problem.actions for t in problem.states]
            for s in problem.states
        },
    )
