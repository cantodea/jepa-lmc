from __future__ import annotations

from pathlib import Path

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


def main() -> None:
    config_path = Path("configs/gridworld_6x6.yaml")
    config = load_yaml(config_path)
    env = make_gridworld_from_config(config)

    print("=== GridWorld loaded ===")
    env.print_map()

    transition_system = gridworld_to_transition_system(env)
    checker = CTLModelChecker(transition_system)
    danger = Atom("danger")
    goal = Atom("goal")
    safe = Atom("safe")
    not_danger = Not(danger)

    print("\n=== Ground-truth model checking ===")
    formulas = {
        "EF danger": EF(danger),
        "EF goal": EF(goal),
        "E[!danger U goal]": EU(not_danger, goal),
        "AG !danger": AG(not_danger),
        "AF goal": AF(goal),
        "EG safe": EG(safe),
    }
    results = {
        name: checker.holds(env.start, formula)
        for name, formula in formulas.items()
    }
    for property_name, holds in results.items():
        print(f"{property_name}: {holds}")

    print("\n=== Safe action witness to goal ===")
    safe_witness = find_eu_witness(
        checker,
        env.start,
        condition=not_danger,
        target=goal,
    )

    if safe_witness is None:
        print("No safe path to goal found.")
    else:
        for step in safe_witness.steps:
            action_name = env.ACTION_NAMES[step.action]
            print(
                f"{step.state} --{action_name}--> {step.next_state}"
            )

    print("\n=== AG !danger counterexample ===")
    counterexample = find_ag_counterexample(
        checker,
        env.start,
        invariant=not_danger,
    )
    if counterexample is None:
        print("No counterexample found.")
    else:
        for step in counterexample.steps:
            action_name = env.ACTION_NAMES[step.action]
            print(
                f"{step.state} --{action_name}--> {step.next_state}"
            )

    expected = {
        "EF danger": True,
        "EF goal": True,
        "E[!danger U goal]": True,
        "AG !danger": False,
        "AF goal": False,
        "EG safe": True,
    }
    assert results == expected
    assert safe_witness is not None
    assert counterexample is not None

    print("\nReal-model checking passed.")


if __name__ == "__main__":
    main()
