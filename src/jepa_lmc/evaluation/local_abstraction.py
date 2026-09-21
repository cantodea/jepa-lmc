"""Oracle local closed balls on a frozen finite-state JEPA distance table."""

from __future__ import annotations

from statistics import median

import torch
from torch import Tensor

from jepa_lmc.evaluation.latent_radius import LatentDistances


def oracle_local_mask(table: LatentDistances) -> Tensor:
    """Smallest closed distance ball per row that contains its true successor.

    The bound and comparison use exactly the same saved distances. Equal-distance
    candidates are all retained; there is no rank truncation or successor repair.
    This consumes oracle truth and is not an unknown-dynamics bound estimator.
    """
    return table.distances <= table.errors[:, None]


def ranking_rows(table: LatentDistances) -> list[dict]:
    if len(table.states) < 3:
        raise ValueError("The d1/d2/d3 diagnostic requires at least three states.")
    order = table.distances.argsort(dim=1, stable=True)
    result = []
    for i, (state, action) in enumerate(table.pairs):
        ranked = order[i].tolist()
        values = table.distances[i, order[i]].tolist()
        target = table.true_indices[i]
        true_distance = float(table.distances[i, target])
        rank = ranked.index(target) + 1
        result.append(
            {
                "state_row": state[0],
                "state_col": state[1],
                "action": action,
                "true_index": target,
                "true_rank": rank,
                "d1": values[0],
                "d2": values[1],
                "d3": values[2],
                "d_true": true_distance,
                "margin": values[1] - values[0],
                "true_distance_ties": values.count(true_distance),
                "top1_covered": rank <= 1,
                "top2_covered": rank <= 2,
                "top3_covered": rank <= 3,
            }
        )
    return result


def candidate_rows(table: LatentDistances, mask: Tensor) -> list[dict]:
    if mask.shape != table.distances.shape or mask.dtype != torch.bool:
        raise ValueError("Mask must be boolean and match the complete table.")
    rows = []
    for i, (state, action) in enumerate(table.pairs):
        size = int(mask[i].sum())
        covered = bool(mask[i, table.true_indices[i]])
        rows.append(
            {
                "state_row": state[0],
                "state_col": state[1],
                "action": action,
                "size": size,
                "covered": covered,
                "spurious_action_edges": size - int(covered),
            }
        )
    return rows


def summarize_candidates(rows) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot summarize an empty query group.")
    sizes = [r["size"] for r in rows]
    count = sum(sizes)
    spurious = sum(r["spurious_action_edges"] for r in rows)
    return {
        "pairs": len(rows),
        "covered_pairs": sum(r["covered"] for r in rows),
        "successor_coverage": sum(r["covered"] for r in rows) / len(rows),
        "candidate_action_edges": count,
        "spurious_action_edges": spurious,
        "spurious_edge_rate": spurious / count if count else None,
        "extra_action_edges_per_true_edge": spurious / len(rows),
        "mean_size": count / len(rows),
        "median_size": median(sizes),
        "max_size": max(sizes),
        "empty_fraction": sizes.count(0) / len(rows),
        "singleton_fraction": sizes.count(1) / len(rows),
        "doubleton_fraction": sizes.count(2) / len(rows),
        "size_histogram": {str(n): sizes.count(n) for n in sorted(set(sizes))},
    }


def identity_simulation(left, right, *, action_sensitive=True) -> dict:
    """Check the identity relation directly, rather than merely initial similarity."""
    if left.states != right.states or left.initial_states != right.initial_states:
        raise ValueError("Identity comparison requires identical state/initial sets.")
    labels_equal = all(
        left.propositions(s) == right.propositions(s) for s in left.states
    )
    missing = []
    steps = 0
    for state in sorted(left.states):
        replies = {
            (e.action if action_sensitive else None, e.target)
            for e in right.action_successors(state)
        }
        for edge in left.action_successors(state):
            steps += 1
            key = (edge.action if action_sensitive else None, edge.target)
            if key not in replies:
                missing.append((state, edge.action, edge.target))
    return {
        "holds": labels_equal and not missing,
        "labels_equal": labels_equal,
        "states": len(left.states),
        "matched_edge_obligations": steps - len(missing),
        "edge_obligations": steps,
        "missing_edges": missing,
    }
