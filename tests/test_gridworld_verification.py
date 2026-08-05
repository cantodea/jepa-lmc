from __future__ import annotations

import unittest

from jepa_lmc.envs.config import load_yaml, make_gridworld_from_config
from jepa_lmc.verification.ctl import (
    AF,
    AG,
    EF,
    EG,
    EU,
    Atom,
    CTLModelChecker,
    Not,
)
from jepa_lmc.verification.gridworld import (
    gridworld_to_transition_system,
)
from jepa_lmc.verification.witness import (
    find_ag_counterexample,
    find_eu_witness,
)


class GridWorldCheckingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_yaml("configs/gridworld_6x6.yaml")
        cls.env = make_gridworld_from_config(config)
        cls.system = gridworld_to_transition_system(cls.env)
        cls.checker = CTLModelChecker(cls.system)

    def test_expected_ctl_properties(self) -> None:
        danger = Atom("danger")
        goal = Atom("goal")
        safe = Atom("safe")
        not_danger = Not(danger)

        expected = {
            EF(danger): True,
            EF(goal): True,
            EU(not_danger, goal): True,
            AG(not_danger): False,
            AF(goal): False,
            EG(safe): True,
        }

        for formula, holds in expected.items():
            with self.subTest(formula=formula):
                self.assertEqual(
                    self.checker.holds(self.env.start, formula),
                    holds,
                )

    def test_safe_witness_replays_in_real_environment(self) -> None:
        witness = find_eu_witness(
            self.checker,
            self.env.start,
            condition=Not(Atom("danger")),
            target=Atom("goal"),
        )
        self.assertIsNotNone(witness)
        assert witness is not None

        for step in witness.steps:
            self.assertEqual(
                self.env.transition(step.state, step.action),
                step.next_state,
            )
            self.assertFalse(self.env.is_danger(step.state))

        self.assertEqual(witness.states[-1], self.env.goal)

    def test_safety_counterexample_replays_in_real_environment(self) -> None:
        counterexample = find_ag_counterexample(
            self.checker,
            self.env.start,
            invariant=Not(Atom("danger")),
        )
        self.assertIsNotNone(counterexample)
        assert counterexample is not None

        for step in counterexample.steps:
            self.assertEqual(
                self.env.transition(step.state, step.action),
                step.next_state,
            )

        self.assertTrue(
            self.env.is_danger(counterexample.states[-1])
        )


if __name__ == "__main__":
    unittest.main()
