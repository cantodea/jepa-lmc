from __future__ import annotations

from pathlib import Path

from jepa_lmc.data.collect_gridworld import (
    collect_full_transition_table,
    save_transition_dataset,
)
from jepa_lmc.envs.factory import make_gridworld_from_config
from jepa_lmc.utils.config import load_yaml


def main() -> None:
    config_path = Path("configs/gridworld_6x6.yaml")
    config = load_yaml(config_path)

    env = make_gridworld_from_config(config)

    print("=== GridWorld loaded ===")
    env.print_map()

    states = env.all_states()
    transitions = env.all_transitions()

    print("\n=== Basic statistics ===")
    print(f"Width: {env.width}")
    print(f"Height: {env.height}")
    print(f"Start: {env.start}")
    print(f"Goal: {env.goal}")
    print(f"Number of valid states: {len(states)}")
    print(f"Number of transitions: {len(transitions)}")

    print("\n=== First 10 transitions ===")
    for t in transitions[:10]:
        print(
            f"{t.state} --{env.ACTION_NAMES[t.action]}--> "
            f"{t.next_state} | "
            f"{env.label(t.state)} -> {env.label(t.next_state)}"
        )

    df = collect_full_transition_table(env)

    output_path = config["data"]["output_path"]
    save_transition_dataset(df, output_path)

    print("\n=== Dataset saved ===")
    print(output_path)
    print(df.head())

    assert len(df) == len(states) * len(env.ACTIONS)
    assert Path(output_path).exists()

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()