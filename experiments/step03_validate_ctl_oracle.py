from __future__ import annotations

import argparse
import random
from pathlib import Path

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
    Formula,
    Not,
    Or,
)
from jepa_lmc.checking.transition_system import ExplicitTransitionSystem
from jepa_lmc.validation.nusmv import (
    CTLQuery,
    compare_queries_with_nusmv,
    find_nusmv_executable,
)


PROPOSITIONS = ("safe", "goal", "danger")


def make_random_system(
    rng: random.Random,
    state_count: int,
) -> ExplicitTransitionSystem[int, str]:
    states = tuple(range(state_count))
    transitions: dict[int, list[tuple[str, int]]] = {}
    labels: dict[int, set[str]] = {}

    for state in states:
        successor_count = rng.randint(1, min(4, state_count))
        transitions[state] = [
            (f"a{index}", successor)
            for index, successor in enumerate(
                rng.sample(states, successor_count)
            )
        ]
        labels[state] = {
            proposition
            for proposition in PROPOSITIONS
            if rng.random() < 0.4
        }

    return ExplicitTransitionSystem(
        states=states,
        initial_states={0},
        transitions=transitions,
        labels=labels,
    )


def make_random_formula(rng: random.Random, depth: int) -> Formula:
    if depth <= 0 or rng.random() < 0.25:
        return Atom(rng.choice(PROPOSITIONS))

    operator = rng.choice(
        (Not, And, Or, EX, AX, EF, AF, EG, AG, EU)
    )
    if operator in (And, Or):
        return operator(
            make_random_formula(rng, depth - 1),
            make_random_formula(rng, depth - 1),
        )
    if operator is EU:
        return EU(
            make_random_formula(rng, depth - 1),
            make_random_formula(rng, depth - 1),
        )
    return operator(make_random_formula(rng, depth - 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Differentially validate JEPA-LMC's CTL checker against "
            "an installed nuXmv/NuSMV executable."
        )
    )
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--systems", type=int, default=100)
    parser.add_argument("--formulas-per-system", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.systems <= 0 or args.formulas_per_system <= 0:
        raise ValueError("System and formula counts must be positive.")
    if args.max_depth < 0:
        raise ValueError("Formula depth must be non-negative.")

    executable = find_nusmv_executable(args.executable)
    if executable is None:
        print("nuXmv/NuSMV executable not found.")
        print("Set NUXMV_BINARY or NUSMV_BINARY, or pass --executable PATH.")
        print("External CTL validation was NOT run.")
        return 2

    rng = random.Random(args.seed)
    checked_queries = 0
    print("=== Step 03C: independent CTL oracle validation ===")
    print(f"External checker: {executable}")
    print(f"Seed: {args.seed}")

    for system_index in range(args.systems):
        state_count = rng.randint(2, 8)
        system = make_random_system(rng, state_count)
        formulae = tuple(
            make_random_formula(rng, args.max_depth)
            for _ in range(args.formulas_per_system)
        )
        queries = tuple(
            CTLQuery(
                state=state,
                formula=formula,
                name=f"system_{system_index}/state_{state}/formula_{index}",
            )
            for state in sorted(system.states)
            for index, formula in enumerate(formulae)
        )
        report = compare_queries_with_nusmv(
            system,
            queries,
            executable,
        )
        checked_queries += len(queries)

        if not report.all_match:
            print("\nMISMATCH detected; the CTL checker is not validated.")
            for query, internal, external, matches in zip(
                report.queries,
                report.internal_verdicts,
                report.external_verdicts,
                report.matches,
            ):
                if not matches:
                    print(
                        f"{query.name}: internal={internal}, "
                        f"NuSMV={external}, formula={query.formula!r}"
                    )
            return 1

    print(f"Random transition systems: {args.systems}")
    print(f"State/formula CTL queries: {checked_queries}")
    print("Agreement with independent checker: 100.0%")
    print("Independent CTL oracle validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
