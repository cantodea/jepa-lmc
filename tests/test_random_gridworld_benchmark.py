from __future__ import annotations

import unittest

from jepa_lmc.benchmarks.random_gridworld import (
    generate_random_gridworld_spec,
    make_pilot_benchmark_splits,
)


class RandomGridWorldBenchmarkTests(unittest.TestCase):
    def test_generation_is_reproducible(self) -> None:
        first = generate_random_gridworld_spec(1234)
        second = generate_random_gridworld_spec(1234)
        different = generate_random_gridworld_spec(1235)

        self.assertEqual(first, second)
        self.assertNotEqual(
            (first.walls, first.dangers),
            (different.walls, different.dangers),
        )

    def test_special_cells_never_overlap(self) -> None:
        for seed in range(100):
            spec = generate_random_gridworld_spec(seed)
            self.assertNotIn(spec.start, spec.walls)
            self.assertNotIn(spec.start, spec.dangers)
            self.assertNotIn(spec.goal, spec.walls)
            self.assertNotIn(spec.goal, spec.dangers)
            self.assertTrue(spec.walls.isdisjoint(spec.dangers))

    def test_pilot_splits_are_map_level_and_disjoint(self) -> None:
        splits = make_pilot_benchmark_splits(
            base_seed=500,
            train_size=5,
            validation_size=3,
            test_size=4,
        )
        train_seeds = {spec.seed for spec in splits.train}
        validation_seeds = {spec.seed for spec in splits.validation}
        test_seeds = {spec.seed for spec in splits.test}

        self.assertEqual(len(splits.train), 5)
        self.assertEqual(len(splits.validation), 3)
        self.assertEqual(len(splits.test), 4)
        self.assertTrue(train_seeds.isdisjoint(validation_seeds))
        self.assertTrue(train_seeds.isdisjoint(test_seeds))
        self.assertTrue(validation_seeds.isdisjoint(test_seeds))

    def test_invalid_capacity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceed available"):
            generate_random_gridworld_spec(
                0,
                width=2,
                height=2,
                min_walls=2,
                max_walls=2,
                min_dangers=1,
                max_dangers=1,
            )


if __name__ == "__main__":
    unittest.main()
