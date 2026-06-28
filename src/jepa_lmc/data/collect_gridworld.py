from __future__ import annotations

from pathlib import Path

import pandas as pd

from jepa_lmc.envs.gridworld import GridWorld


def collect_full_transition_table(env: GridWorld) -> pd.DataFrame:
    """
    Generate all transitions:
        state × action -> next_state
    """
    rows = []

    for t in env.all_transitions():
        row, col = t.state
        next_row, next_col = t.next_state

        rows.append(
            {
                "row": row,
                "col": col,
                "state_id": env.state_to_id(t.state),
                "action": t.action,
                "action_name": env.ACTION_NAMES[t.action],
                "next_row": next_row,
                "next_col": next_col,
                "next_state_id": env.state_to_id(t.next_state),
                "label": env.label(t.state),
                "next_label": env.label(t.next_state),
            }
        )

    return pd.DataFrame(rows)


def save_transition_dataset(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)