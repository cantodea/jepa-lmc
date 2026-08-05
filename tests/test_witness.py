from __future__ import annotations

import unittest

from jepa_lmc.verification.ctl import Atom, CTLModelChecker
from jepa_lmc.verification.witness import (
    find_ag_counterexample,
    find_ef_witness,
    find_eu_witness,
)
from tests.test_ctl import make_branching_system


class WitnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = make_branching_system()
        self.checker = CTLModelChecker(self.system)

    def test_ef_witness_contains_actions(self) -> None:
        witness = find_ef_witness(
            self.checker, "s0", Atom("goal")
        )
        self.assertIsNotNone(witness)
        assert witness is not None
        self.assertEqual(witness.states, ("s0", "goal"))
        self.assertEqual(witness.steps[0].action, "finish")

    def test_eu_witness_respects_condition(self) -> None:
        witness = find_eu_witness(
            self.checker,
            "s0",
            condition=Atom("safe"),
            target=Atom("goal"),
        )
        self.assertIsNotNone(witness)
        assert witness is not None
        self.assertNotIn("danger", witness.states)

    def test_ag_counterexample_reaches_violation(self) -> None:
        counterexample = find_ag_counterexample(
            self.checker,
            "s0",
            invariant=Atom("safe"),
        )
        self.assertIsNotNone(counterexample)
        assert counterexample is not None
        self.assertEqual(counterexample.states[-1], "danger")
        self.assertEqual(counterexample.steps[-1].action, "fail")

    def test_missing_witness_returns_none(self) -> None:
        witness = find_ef_witness(
            self.checker, "loop", Atom("goal")
        )
        self.assertIsNone(witness)


if __name__ == "__main__":
    unittest.main()
