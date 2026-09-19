"""Exact greatest simulation/bisimulation on two finite labelled state graphs.

Left-to-right simulation means that the right system matches every left step.
Actions are ignored by default, matching this project's ordinary CTL semantics.
The two copies of the state space stay distinct even when state IDs coincide.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal

from jepa_lmc.verification.transition_system import ExplicitTransitionSystem

Pair = tuple[Hashable, Hashable]
Kind = Literal["simulation", "bisimulation"]


@dataclass(frozen=True)
class Elimination:
    left: Hashable
    right: Hashable
    round: int
    side: Literal["label", "left", "right"]
    action: Hashable | None = None
    target: Hashable | None = None


@dataclass(frozen=True)
class BehavioralRelation:
    kind: Kind
    action_sensitive: bool
    pairs: frozenset[Pair]
    eliminations: tuple[Elimination, ...]
    rounds: int

    def relates_initials(self, left, right) -> bool:
        """Simulation covers left initials; bisimulation covers both sets."""
        forward = all(
            any((s, t) in self.pairs for t in right.initial_states)
            for s in left.initial_states
        )
        backward = all(
            any((s, t) in self.pairs for s in left.initial_states)
            for t in right.initial_states
        )
        return forward and (self.kind == "simulation" or backward)


def _edges(system, action_sensitive: bool) -> dict:
    return {
        s: tuple(
            sorted(
                {
                    (e.action if action_sensitive else None, e.target)
                    for e in system.action_successors(s)
                },
                key=repr,
            )
        )
        for s in system.states
    }


def greatest_relation(
    left: ExplicitTransitionSystem,
    right: ExplicitTransitionSystem,
    *,
    kind: Kind = "simulation",
    action_sensitive: bool = False,
) -> BehavioralRelation:
    """Synchronous deletion from label-compatible pairs, with rejection evidence.

    Every failed reply to a round-k deletion was rejected at an earlier round.
    Thus the evidence is a finite DAG, not necessarily a single counterexample
    path. Both directions of a bisimulation use the SAME surviving pair set.
    """
    if kind not in ("simulation", "bisimulation"):
        raise ValueError("Expected simulation or bisimulation.")
    pairs = set()
    eliminations = []
    for s in sorted(left.states, key=repr):
        for t in sorted(right.states, key=repr):
            if left.propositions(s) == right.propositions(t):
                pairs.add((s, t))
            else:
                eliminations.append(Elimination(s, t, 0, "label"))
    left_edges = _edges(left, action_sensitive)
    right_edges = _edges(right, action_sensitive)
    rounds = 0
    while True:
        rejected = []
        for s, t in sorted(pairs, key=repr):
            failure = None
            for action, target in left_edges[s]:
                if not any(
                    a == action and (target, reply) in pairs
                    for a, reply in right_edges[t]
                ):
                    failure = Elimination(s, t, rounds + 1, "left", action, target)
                    break
            if failure is None and kind == "bisimulation":
                for action, target in right_edges[t]:
                    if not any(
                        a == action and (reply, target) in pairs
                        for a, reply in left_edges[s]
                    ):
                        failure = Elimination(s, t, rounds + 1, "right", action, target)
                        break
            if failure is not None:
                rejected.append(failure)
        if not rejected:
            break
        rounds += 1
        for row in rejected:
            pairs.remove((row.left, row.right))
        eliminations.extend(rejected)
    return BehavioralRelation(
        kind, action_sensitive, frozenset(pairs), tuple(eliminations), rounds
    )


def audit_greatest_relation(left, right, result: BehavioralRelation) -> None:
    """Check closure AND maximality from the returned rejection certificate.

    This does not rerun deletion. Surviving pairs must be a relation of the
    claimed kind; every excluded pair needs a strict, earlier-round justification.
    A forged subset cannot pass merely because its surviving pairs are closed.
    """
    if result.kind not in ("simulation", "bisimulation"):
        raise AssertionError("Invalid relation kind.")
    universe = {(s, t) for s in left.states for t in right.states}
    rejected = {(r.left, r.right): r for r in result.eliminations}
    if len(rejected) != len(result.eliminations):
        raise AssertionError("Duplicate elimination.")
    if result.pairs & rejected.keys() or result.pairs | rejected.keys() != universe:
        raise AssertionError("Surviving and rejected pairs must partition the product.")

    def outgoing(system, s):
        return {
            (e.action if result.action_sensitive else None, e.target)
            for e in system.action_successors(s)
        }

    def follows(s, t, reverse=False):
        source, other = (right, left) if reverse else (left, right)
        p, q = (t, s) if reverse else (s, t)
        for action, target in outgoing(source, p):
            if not any(
                a == action
                and ((reply, target) if reverse else (target, reply)) in result.pairs
                for a, reply in outgoing(other, q)
            ):
                return False
        return True

    for s, t in result.pairs:
        if left.propositions(s) != right.propositions(t) or not follows(s, t):
            raise AssertionError("A surviving pair violates simulation.")
        if result.kind == "bisimulation" and not follows(s, t, reverse=True):
            raise AssertionError("A surviving pair violates reverse matching.")
    for (s, t), row in rejected.items():
        if row.side == "label":
            if row.round != 0 or left.propositions(s) == right.propositions(t):
                raise AssertionError("Invalid label-mismatch evidence.")
            continue
        if left.propositions(s) != right.propositions(t) or row.round < 1:
            raise AssertionError("Invalid transition-elimination round.")
        if row.side not in ("left", "right") or (
            row.side == "right" and result.kind != "bisimulation"
        ):
            raise AssertionError("Invalid challenging side.")
        reverse = row.side == "right"
        source, other = (right, left) if reverse else (left, right)
        p, q = (t, s) if reverse else (s, t)
        if (row.action, row.target) not in outgoing(source, p):
            raise AssertionError("The challenging edge does not exist.")
        for action, reply in outgoing(other, q):
            if action != row.action:
                continue
            pair = (reply, row.target) if reverse else (row.target, reply)
            if pair not in rejected or rejected[pair].round >= row.round:
                raise AssertionError(
                    "A reply lacks strictly earlier rejection evidence."
                )
    if result.rounds != max((r.round for r in result.eliminations), default=0):
        raise AssertionError("Incorrect elimination depth.")


def bisimulation_by_partition(
    left, right, *, action_sensitive=False
) -> frozenset[Pair]:
    """Independent bisimulation cross-check by refining the disjoint union.

    Blocks split by current block and successor-block signature. Action labels
    enter the signature only when requested. There is no pair-deletion logic.
    """
    systems = (left, right)
    nodes = [
        (i, s)
        for i, system in enumerate(systems)
        for s in sorted(system.states, key=repr)
    ]
    groups = {}
    blocks = {}
    for node in nodes:
        i, s = node
        label = systems[i].propositions(s)
        blocks[node] = groups.setdefault(label, len(groups))
    while True:
        groups, refined = {}, {}
        for node in nodes:
            i, s = node
            signature = (
                blocks[node],
                frozenset(
                    (e.action if action_sensitive else None, blocks[(i, e.target)])
                    for e in systems[i].action_successors(s)
                ),
            )
            refined[node] = groups.setdefault(signature, len(groups))
        if refined == blocks:
            return frozenset(
                (s, t)
                for s in left.states
                for t in right.states
                if blocks[(0, s)] == blocks[(1, t)]
            )
        blocks = refined
