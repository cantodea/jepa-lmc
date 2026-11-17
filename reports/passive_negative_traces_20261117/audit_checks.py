"""Independent verification of the historical paired-negative protocol.

Run from the repository with PYTHONPATH=src. No neural dependency is used.
This regenerates fixed historical pools; it does not retrain historical models.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from unittest.mock import patch

from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.envs.gridworld import GridWorld
from jepa_lmc.evaluation.trace_sanity import (
    KnownProblem, accepts_trace, make_hard_negatives, observed_edges,
    observed_self_loop_completion, relation_edges, sample_good_traces,
    validate_trace_pairs,
)


def stream_seed(seed, case, stream):
    return int.from_bytes(hashlib.sha256(
        f"good_bad_trace_sanity_v1:{seed}:{case}:{stream}".encode()
    ).digest()[:8], "big")


def run():
    rows = []
    same_label = 0
    for case in radius_stress_cases():
        env = case.make_env()
        states = tuple(env.all_states())
        labels = {s: frozenset(
            ({"danger"} if env.is_danger(s) else {"safe"})
            | ({"goal"} if env.is_goal(s) else set())
        ) for s in states}
        known = KnownProblem(states, frozenset([env.start]), tuple(env.ACTIONS), labels)
        for seed in (20260804, 20260805, 20260806):
            goods = sample_good_traces(env, seed=stream_seed(seed, case.name, "good"))
            negatives = make_hard_negatives(
                env, goods, labels, seed=stream_seed(seed, case.name, "bad")
            )
            validate_trace_pairs(env, goods, negatives)
            same_label += sum(b.same_proposition_labels for b in negatives)
            for budget in (1, 2, 4, 8, 16):
                # Expose only positive trace observations and known catalogue.
                with patch.object(GridWorld, "transition", side_effect=AssertionError("R leak")):
                    observed = observed_edges(goods[:budget])
                    fixed = {(s, a): t for s, a, t in observed}
                    complete = observed_self_loop_completion(known, observed)
                    complete_edges = relation_edges(complete)
                    conflicts = [sum(
                        (s, a) in fixed and fixed[s, a] != t
                        for s, a, t in b.trace.edges
                    ) for b in negatives[:budget]]
                    assert all(c >= 1 for c in conflicts)
                    assert all(accepts_trace(known, observed, g) for g in goods[:budget])
                    assert all(not accepts_trace(known, observed, b.trace)
                               for b in negatives[:budget])
                    assert all(not accepts_trace(known, complete_edges, b.trace)
                               for b in negatives[:budget])
                    rows.append({
                        "case": case.name, "family": case.family, "seed": seed,
                        "budget": budget, "observed_pairs": len(fixed),
                        "state_action_pairs": len(states) * len(known.actions),
                        "paired_negatives": budget,
                        "all_redundant_under_determinism": True,
                        "positive_conflict_counts": conflicts,
                        "self_loop_completion_rejects_paired_negatives": True,
                    })
    assert len(rows) == 360
    assert same_label == 1133
    result = {
        "base_commit": "539410dec1d5c4e4433bd9a57b10a309cd9a850d",
        "python": platform.python_version(),
        "scope": "Regenerate historical sampler, inspect exact conflicting positive edge; no neural retraining",
        "unique_pools": 72, "unique_goods": 1152, "unique_negatives": 1152,
        "same_label_negatives": same_label,
        "cases": 360, "all_checks_passed": True, "records": rows,
    }
    out = Path(__file__).with_name("audit_checks.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    run()
