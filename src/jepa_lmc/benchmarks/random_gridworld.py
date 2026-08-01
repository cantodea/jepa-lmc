from __future__ import annotations

import random
from dataclasses import dataclass

from jepa_lmc.envs.gridworld import GridWorld, State


@dataclass(frozen=True)
class GridWorldSpec:
    """Serializable ingredients for one reproducible benchmark map."""

    seed: int
    width: int
    height: int
    start: State
    goal: State
    walls: frozenset[State]
    dangers: frozenset[State]

    def make_env(self) -> GridWorld:
        return GridWorld(
            width=self.width,
            height=self.height,
            start=self.start,
            goal=self.goal,
            walls=set(self.walls),
            dangers=set(self.dangers),
        )


@dataclass(frozen=True)
class GridWorldBenchmarkSplits:
    train: tuple[GridWorldSpec, ...]
    validation: tuple[GridWorldSpec, ...]
    test: tuple[GridWorldSpec, ...]


def generate_random_gridworld_spec(
    seed: int,
    *,
    width: int = 6,
    height: int = 6,
    min_walls: int = 2,
    max_walls: int = 10,
    min_dangers: int = 1,
    max_dangers: int = 4,
) -> GridWorldSpec:
    """Generate a deterministic map from a seed without overlapping cells."""
    if width < 2 or height < 2:
        raise ValueError("Grid dimensions must both be at least two.")
    if not 0 <= min_walls <= max_walls:
        raise ValueError("Invalid wall-count range.")
    if not 0 <= min_dangers <= max_dangers:
        raise ValueError("Invalid danger-count range.")

    start = (0, 0)
    goal = (height - 1, width - 1)
    candidates = [
        (row, col)
        for row in range(height)
        for col in range(width)
        if (row, col) not in {start, goal}
    ]
    if max_walls + max_dangers > len(candidates):
        raise ValueError("Requested walls and dangers exceed available cells.")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    wall_count = rng.randint(min_walls, max_walls)
    danger_count = rng.randint(min_dangers, max_dangers)
    walls = frozenset(candidates[:wall_count])
    dangers = frozenset(
        candidates[wall_count : wall_count + danger_count]
    )

    return GridWorldSpec(
        seed=seed,
        width=width,
        height=height,
        start=start,
        goal=goal,
        walls=walls,
        dangers=dangers,
    )


def make_pilot_benchmark_splits(
    *,
    base_seed: int = 20260802,
    train_size: int = 100,
    validation_size: int = 20,
    test_size: int = 50,
) -> GridWorldBenchmarkSplits:
    """Create disjoint map-level splits for the pilot JEPA experiment."""
    sizes = (train_size, validation_size, test_size)
    if any(size <= 0 for size in sizes):
        raise ValueError("Every benchmark split must contain at least one map.")

    specs = tuple(
        generate_random_gridworld_spec(base_seed + offset)
        for offset in range(sum(sizes))
    )
    train_end = train_size
    validation_end = train_end + validation_size
    return GridWorldBenchmarkSplits(
        train=specs[:train_end],
        validation=specs[train_end:validation_end],
        test=specs[validation_end:],
    )
