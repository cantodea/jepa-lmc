from __future__ import annotations

import random
import unittest

from jepa_lmc.verification.ctl import (
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
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem


def make_random_system(seed: int) -> ExplicitTransitionSystem[int, str]:
    rng = random.Random(seed)
    state_count = rng.randint(2, 8)
    states = tuple(range(state_count))
    transitions: dict[int, list[tuple[str, int]]] = {}
    labels: dict[int, set[str]] = {}

    for state in states:
        successor_count = rng.randint(1, min(3, state_count))
        successors = rng.sample(states, successor_count)
        transitions[state] = [
            (f"edge_{index}", successor)
            for index, successor in enumerate(successors)
        ]
        labels[state] = {
            proposition
            for proposition in ("p", "q")
            if rng.random() < 0.5
        }

    return ExplicitTransitionSystem(
        states=states,
        initial_states={0},
        transitions=transitions,
        labels=labels,
    )


class CTLIdentityTests(unittest.TestCase):
    def test_dualities_hold_on_random_total_kripke_structures(self) -> None:
        for seed in range(50):
            with self.subTest(seed=seed):
                system = make_random_system(seed)
                checker = CTLModelChecker(system)
                p = Atom("p")
                q = Atom("q")
                formulae = (
                    p,
                    Not(p),
                    And(p, q),
                    Or(EX(p), AX(q)),
                    EU(p, q),
                )

                for formula in formulae:
                    self.assertEqual(
                        checker.satisfying_states(AG(formula)),
                        checker.satisfying_states(Not(EF(Not(formula)))),
                    )
                    self.assertEqual(
                        checker.satisfying_states(AF(formula)),
                        checker.satisfying_states(Not(EG(Not(formula)))),
                    )
                    self.assertEqual(
                        checker.satisfying_states(AX(formula)),
                        checker.satisfying_states(Not(EX(Not(formula)))),
                    )

    def test_fixed_point_equations_hold_on_random_systems(self) -> None:
        for seed in range(50, 100):
            with self.subTest(seed=seed):
                system = make_random_system(seed)
                checker = CTLModelChecker(system)
                p = Atom("p")
                q = Atom("q")
                true_formula = Or(p, Not(p))

                ef_p = EF(p)
                af_p = AF(p)
                eg_p = EG(p)
                ag_p = AG(p)
                eu_p_q = EU(p, q)

                self.assertEqual(
                    checker.satisfying_states(ef_p),
                    checker.satisfying_states(Or(p, EX(ef_p))),
                )
                self.assertEqual(
                    checker.satisfying_states(af_p),
                    checker.satisfying_states(Or(p, AX(af_p))),
                )
                self.assertEqual(
                    checker.satisfying_states(eg_p),
                    checker.satisfying_states(And(p, EX(eg_p))),
                )
                self.assertEqual(
                    checker.satisfying_states(ag_p),
                    checker.satisfying_states(And(p, AX(ag_p))),
                )
                self.assertEqual(
                    checker.satisfying_states(eu_p_q),
                    checker.satisfying_states(
                        Or(q, And(p, EX(eu_p_q)))
                    ),
                )
                self.assertEqual(
                    checker.satisfying_states(EF(q)),
                    checker.satisfying_states(EU(true_formula, q)),
                )


if __name__ == "__main__":
    unittest.main()
