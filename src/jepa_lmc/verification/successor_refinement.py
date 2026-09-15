"""Property-directed successor refinement with an exact deterministic oracle.

State IDs enumerate a complete finite catalogue. Neural scores may order work;
only a trusted concrete query can remove an upper edge. No neural dependency.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Edge = tuple[int, int, int]


@dataclass(frozen=True)
class ReachabilityTask:
    name: str
    start: int
    targets: frozenset[int]
    allowed: frozenset[int]
    path_value: bool


def task_from_labels(
    name: str, start: int, labels: Sequence[frozenset[str]]
) -> ReachabilityTask:
    states = frozenset(range(len(labels)))
    danger = frozenset(s for s in states if "danger" in labels[s])
    goal = frozenset(s for s in states if "goal" in labels[s])
    if name == "AG !danger":
        return ReachabilityTask(name, start, danger, states, False)
    if name == "EF goal":
        return ReachabilityTask(name, start, goal, states, True)
    if name == "E[!danger U goal]":
        return ReachabilityTask(name, start, goal, states - danger, True)
    raise ValueError("Only AG !danger, EF goal and E[!danger U goal] are supported.")


@dataclass(frozen=True)
class OracleObservation:
    state: int
    action: int
    successor: int
    expected: int | None
    round: int


@dataclass(frozen=True)
class RefinementResult:
    value: bool | None
    reason: str
    queries: int
    rounds: int
    spurious_paths: int
    witness: tuple[Edge, ...] | None
    closed_region: frozenset[int] | None
    observations: tuple[OracleObservation, ...]
    mean_upper_candidates: float


class SuccessorRefinement:
    """Unknown pairs have all successors; queried pairs have one exact successor.

    The caller guarantees deterministic, stationary, total dynamics and an exact
    successor oracle. A sample from nondeterministic dynamics is insufficient.
    Oracle calls occur only in observe(); caches are owned by each instance.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        oracle: Callable[[int, int], int],
        *,
        costs: Sequence[Sequence[float]] | None = None,
        budget: int | None = None,
    ) -> None:
        if num_states < 1 or num_actions < 1:
            raise ValueError("The finite state and action sets must be nonempty.")
        if budget is not None and (not isinstance(budget, int) or budget < 0):
            raise ValueError("Query budget must be a nonnegative integer.")
        self.num_states = num_states
        self.num_actions = num_actions
        self.budget = num_states * num_actions if budget is None else budget
        self._oracle = oracle
        self._known: dict[tuple[int, int], int] = {}
        self._observations: list[OracleObservation] = []
        self.costs = None if costs is None else tuple(tuple(row) for row in costs)
        if self.costs is not None and (
            len(self.costs) != num_states * num_actions
            or any(len(row) != num_states for row in self.costs)
            or any(
                not math.isfinite(value) or value <= 0
                for row in self.costs
                for value in row
            )
        ):
            raise ValueError("Costs must cover every candidate and be finite/positive.")

    @property
    def known(self) -> dict[tuple[int, int], int]:
        return dict(self._known)

    @property
    def query_count(self) -> int:
        return len(self._observations)

    def candidates(self, state: int, action: int) -> frozenset[int]:
        self._validate_pair(state, action)
        target = self._known.get((state, action))
        return (
            frozenset(range(self.num_states))
            if target is None
            else frozenset((target,))
        )

    def _validate_pair(self, state: int, action: int) -> None:
        if not 0 <= state < self.num_states or not 0 <= action < self.num_actions:
            raise ValueError("Unknown state or action ID.")

    def observe(
        self, state: int, action: int, *, expected: int | None = None, round: int = 0
    ) -> int:
        self._validate_pair(state, action)
        if (state, action) in self._known:
            return self._known[(state, action)]
        if self.query_count >= self.budget:
            raise RuntimeError("Exact successor query budget exhausted.")
        target = self._oracle(state, action)
        if not isinstance(target, int) or not 0 <= target < self.num_states:
            raise ValueError("Oracle returned a successor outside the known catalogue.")
        self._known[(state, action)] = target
        self._observations.append(
            OracleObservation(state, action, target, expected, round)
        )
        return target

    def _validate_task(self, task: ReachabilityTask) -> None:
        states = frozenset(range(self.num_states))
        if task.start not in states or not (task.targets | task.allowed) <= states:
            raise ValueError("Task refers to an unknown state.")

    def _find_path(
        self, task: ReachabilityTask, *, upper: bool
    ) -> tuple[tuple[Edge, ...] | None, frozenset[int]]:
        if task.start in task.targets:
            return (), frozenset((task.start,))
        if task.start not in task.allowed:
            return None, frozenset()
        permitted = task.allowed | task.targets
        distances = {task.start: 0.0}
        parents: dict[int, Edge] = {}
        queue = [(0.0, task.start)]
        while queue:
            cost, state = heapq.heappop(queue)
            if cost != distances[state]:
                continue
            if state in task.targets:
                path = []
                while state != task.start:
                    edge = parents[state]
                    path.append(edge)
                    state = edge[0]
                return tuple(reversed(path)), frozenset(distances)
            # Prefer already confirmed edges on equal cost, avoiding wasted queries.
            actions = sorted(
                range(self.num_actions), key=lambda a: (state, a) not in self._known
            )
            for action in actions:
                target = self._known.get((state, action))
                if target is None and not upper:
                    continue
                targets = range(self.num_states) if target is None else (target,)
                for next_state in targets:
                    if next_state not in permitted:
                        continue
                    weight = (
                        self.costs[state * self.num_actions + action][next_state]
                        if target is None and self.costs is not None
                        else 1.0
                    )
                    distance = cost + weight
                    if distance < distances.get(next_state, math.inf):
                        distances[next_state] = distance
                        parents[next_state] = (state, action, next_state)
                        heapq.heappush(queue, (distance, next_state))
        return None, frozenset(distances)

    def _result(
        self,
        value: bool | None,
        reason: str,
        rounds: int,
        spurious: int,
        witness: tuple[Edge, ...] | None = None,
        region: frozenset[int] | None = None,
    ) -> RefinementResult:
        total_pairs = self.num_states * self.num_actions
        candidates = (
            total_pairs - self.query_count
        ) * self.num_states + self.query_count
        return RefinementResult(
            value,
            reason,
            self.query_count,
            rounds,
            spurious,
            witness,
            region,
            tuple(self._observations),
            candidates / total_pairs,
        )

    def solve(self, task: ReachabilityTask) -> RefinementResult:
        """CEGAR-style loop, limited to finite reachability and its negation."""
        self._validate_task(task)
        rounds = spurious = 0
        while True:
            confirmed, _ = self._find_path(task, upper=False)
            if confirmed is not None:
                return self._result(
                    task.path_value,
                    "confirmed_path",
                    rounds,
                    spurious,
                    witness=confirmed,
                )
            candidate, region = self._find_path(task, upper=True)
            if candidate is None:
                return self._result(
                    not task.path_value,
                    "upper_excludes_path",
                    rounds,
                    spurious,
                    region=region,
                )
            if self.query_count >= self.budget:
                return self._result(None, "budget_exhausted", rounds, spurious)
            rounds += 1
            for state, action, expected in candidate:
                if (
                    state,
                    action,
                ) not in self._known and self.query_count >= self.budget:
                    break
                actual = self.observe(state, action, expected=expected, round=rounds)
                if actual != expected:
                    spurious += 1
                    break

    def solve_direct_bfs(self, task: ReachabilityTask) -> RefinementResult:
        """Strong control: query exact successors only during reachable BFS."""
        self._validate_task(task)
        if task.start in task.targets:
            return self._result(task.path_value, "confirmed_path", 0, 0, witness=())
        if task.start not in task.allowed:
            return self._result(
                not task.path_value, "upper_excludes_path", 0, 0, region=frozenset()
            )
        permitted = task.allowed | task.targets
        queue = deque([task.start])
        seen = {task.start}
        parents: dict[int, Edge] = {}
        rounds = 0
        while queue:
            state = queue.popleft()
            rounds += 1
            for action in range(self.num_actions):
                if (
                    state,
                    action,
                ) not in self._known and self.query_count >= self.budget:
                    return self._result(None, "budget_exhausted", rounds, 0)
                target = self.observe(state, action, round=rounds)
                if target not in permitted or target in seen:
                    continue
                seen.add(target)
                parents[target] = (state, action, target)
                if target in task.targets:
                    path = []
                    while target != task.start:
                        edge = parents[target]
                        path.append(edge)
                        target = edge[0]
                    return self._result(
                        task.path_value,
                        "confirmed_path",
                        rounds,
                        0,
                        witness=tuple(reversed(path)),
                    )
                queue.append(target)
        return self._result(
            not task.path_value,
            "upper_excludes_path",
            rounds,
            0,
            region=frozenset(seen),
        )
