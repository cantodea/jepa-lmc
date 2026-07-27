from __future__ import annotations

import unittest

from jepa_lmc.checking.ctl import (
    AF,
    AG,
    AX,
    EF,
    EG,
    EU,
    EX,
    And,
    Atom,
    CTLModelChecker,
    Not,
    Or,
)
from jepa_lmc.checking.transition_system import ExplicitTransitionSystem


def make_branching_system() -> ExplicitTransitionSystem[str, str]:
    states = {"s0", "loop", "goal", "danger"}
    return ExplicitTransitionSystem(
        states=states,
        initial_states={"s0"},
        transitions={
            "s0": [
                ("wait", "loop"),
                ("finish", "goal"),
                ("fail", "danger"),
            ],
            "loop": [("wait", "loop")],
            "goal": [("stay", "goal")],
            "danger": [("stay", "danger")],
        },
        labels={
            "s0": {"safe"},
            "loop": {"safe"},
            "goal": {"safe", "goal"},
            "danger": {"danger"},
        },
    )


def make_progress_system() -> ExplicitTransitionSystem[str, str]:
    return ExplicitTransitionSystem(
        states={"s0", "s1", "goal"},
        initial_states={"s0"},
        transitions={
            "s0": [("next", "s1")],
            "s1": [("finish", "goal")],
            "goal": [("stay", "goal")],
        },
        labels={
            "s0": {"safe"},
            "s1": {"safe"},
            "goal": {"safe", "goal"},
        },
    )


class CTLModelCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = make_branching_system()
        self.checker = CTLModelChecker(self.system)
        self.safe = Atom("safe")
        self.goal = Atom("goal")
        self.danger = Atom("danger")

    def test_boolean_formulae(self) -> None:
        self.assertTrue(self.checker.holds("s0", self.safe))
        self.assertTrue(
            self.checker.holds(
                "s0", And(self.safe, Not(self.danger))
            )
        )
        self.assertTrue(
            self.checker.holds("danger", Or(self.safe, self.danger))
        )

    def test_next_operators_distinguish_existential_and_universal(self) -> None:
        self.assertTrue(self.checker.holds("s0", EX(self.goal)))
        self.assertFalse(self.checker.holds("s0", AX(self.safe)))

    def test_eventually_and_globally_operators(self) -> None:
        self.assertTrue(self.checker.holds("s0", EF(self.goal)))
        self.assertFalse(self.checker.holds("s0", AF(self.goal)))
        self.assertTrue(self.checker.holds("s0", EG(self.safe)))
        self.assertFalse(self.checker.holds("s0", AG(self.safe)))

    def test_existential_until(self) -> None:
        self.assertTrue(
            self.checker.holds("s0", EU(self.safe, self.goal))
        )
        self.assertFalse(
            self.checker.holds("loop", EU(self.safe, self.goal))
        )

    def test_af_on_deterministic_progress_system(self) -> None:
        checker = CTLModelChecker(make_progress_system())
        self.assertTrue(checker.holds("s0", AF(Atom("goal"))))
        self.assertTrue(checker.holds("s0", AG(Atom("safe"))))

    def test_unknown_state_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.checker.holds("missing", self.safe)

    def test_deadlock_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no successor"):
            ExplicitTransitionSystem(
                states={"deadlock"},
                initial_states={"deadlock"},
                transitions={},
                labels={"deadlock": set()},
            )


if __name__ == "__main__":
    unittest.main()
