from __future__ import annotations

from pathlib import Path

from jepa_lmc.checking.simple_checker import (
    check_basic_properties,
    find_path_to_label,
)
from jepa_lmc.envs.factory import make_gridworld_from_config
from jepa_lmc.utils.config import load_yaml


def main() -> None:
    config_path = Path("configs/gridworld_6x6.yaml")
    config = load_yaml(config_path)
    env = make_gridworld_from_config(config)

    print("=== GridWorld loaded ===")
    env.print_map()

    print("\n=== Ground-truth model checking ===")
    results = check_basic_properties(env, env.start)
    for property_name, holds in results.items():
        print(f"{property_name}: {holds}")

    print("\n=== Safe path to goal ===")
    safe_path = find_path_to_label(
        env,
        initial_state=env.start,
        target_label="goal",
        avoid_label="danger",
    )

    if safe_path is None:
        print("No safe path to goal found.")
    else:
        print(" -> ".join(str(state) for state in safe_path))

    expected = {
        "EF danger": True,
        "EF goal": True,
        "E[!danger U goal]": True,
        "AG !danger": False,
    }
    assert results == expected

    print("\nReal-model checking passed.")


if __name__ == "__main__":
    main()