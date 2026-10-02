"""Common learned-model scorecard: behavioral relations first, logic diagnostics.

All fixed-point algorithms remain in verification.behavioral_relations. No
model inference, training, graph repair, or symbolic-backend semantics live here.
"""

from __future__ import annotations

import json
from pathlib import Path

from jepa_lmc.benchmarks.ctl_suite import default_ctl_suite
from jepa_lmc.verification.behavioral_relations import (
    audit_greatest_relation,
    bisimulation_by_partition,
    greatest_relation,
)
from jepa_lmc.verification.ctl import CTLModelChecker

SEMANTICS = {"action_insensitive": False, "action_sensitive": True}
RELATIONS = ("real_to_top1", "top1_to_real", "bisimulation")
METRICS = (
    "simulation_real_to_learned",
    "simulation_learned_to_real",
    "initial_bisimulation",
    "identity_simulation_real_to_learned",
    "identity_simulation_learned_to_real",
    "identity_bisimulation",
    "strict_identity_simulation_real_to_learned",
    "strict_identity_simulation_learned_to_real",
    "strict_identity_bisimulation",
)
DEFINITIONS = {
    "simulation_real_to_learned": (
        "Every real initial has a learned initial in the greatest forward simulation."
    ),
    "simulation_learned_to_real": (
        "Every learned initial has a real initial in the greatest reverse simulation."
    ),
    "initial_bisimulation": (
        "The greatest bisimulation covers both designated initial sets; for "
        "singletons, (i_real, i_learned) belongs to it. This is not mutual "
        "simulation."
    ),
    "identity_bisimulation": (
        "Shared state space only: forall s in S, (s,s) belongs to the greatest "
        "bisimulation, including unreachable states. Otherwise null (not "
        "applicable)."
    ),
    "identity_simulation": (
        "The same diagonal-membership test on the indicated greatest simulation."
    ),
    "strict_identity": (
        "The diagonal alone is a relation: labels match and edges are included "
        "(simulation) or equal (bisimulation), with actions retained only when "
        "requested."
    ),
    "action_insensitive": (
        "Ignore action labels and duplicate targets, matching existing "
        "CTL/Kripke semantics."
    ),
    "action_sensitive": (
        "Match every challenged transition with an edge carrying the same action."
    ),
    "global": (
        "Global means the identity metric over all shared states, not just initials."
    ),
    "diagnostics": (
        "Top-1 and the fixed six CTL/LTL formulas are additional diagnostics. "
        "Finite-suite agreement is not a bisimulation certificate. Missing LTL "
        "is null."
    ),
}


def behavioral_metrics(real, learned, *, relations=None) -> dict:
    """Compute both semantics, or summarize already computed exact relations.

    ``relations`` maps (action_sensitive, legacy_relation_name) to a result.
    Reused results are still audited for closure/maximality and cross-checked
    against partition refinement; this never trusts saved boolean verdicts.
    """
    expected = {(actions, name) for actions in SEMANTICS.values() for name in RELATIONS}
    if relations is not None and set(relations) != expected:
        raise ValueError("Supply all six behavioral relations, with both semantics.")
    shared = real.states == learned.states
    output = {}
    for semantics, actions in SEMANTICS.items():
        details = {}
        strict = {}
        for name in RELATIONS:
            left, right = (learned, real) if name == "top1_to_real" else (real, learned)
            kind = "bisimulation" if name == "bisimulation" else "simulation"
            result = (
                greatest_relation(left, right, kind=kind, action_sensitive=actions)
                if relations is None
                else relations[actions, name]
            )
            if result.kind != kind or result.action_sensitive != actions:
                raise ValueError("Relation kind or action semantics do not match.")
            audit_greatest_relation(left, right, result)
            if kind == "bisimulation" and result.pairs != bisimulation_by_partition(
                left, right, action_sensitive=actions
            ):
                raise AssertionError("Independent bisimulation algorithms disagree.")
            details[name] = {
                "initial_related": result.relates_initials(left, right),
                "identity_related": result.relates_identity(left, right),
                "retained_pairs": len(result.pairs),
                "elimination_rounds": result.rounds,
                "unrelated_identity_states": [
                    repr(s)
                    for s in sorted(left.states, key=repr)
                    if (s, s) not in result.pairs
                ]
                if shared
                else None,
            }
            if shared:
                labels_match = all(
                    left.propositions(s) == right.propositions(s) for s in left.states
                )
                edges = [
                    {
                        (s, e.action if actions else None, e.target)
                        for s in g.states
                        for e in g.action_successors(s)
                    }
                    for g in (left, right)
                ]
                strict[name] = labels_match and (
                    edges[0] == edges[1]
                    if kind == "bisimulation"
                    else edges[0] <= edges[1]
                )
            else:
                strict[name] = None
        forward, reverse, bisim = (details[n] for n in RELATIONS)
        output[semantics] = {
            "action_sensitive": actions,
            "simulation_real_to_learned": forward["initial_related"],
            "simulation_learned_to_real": reverse["initial_related"],
            "initial_bisimulation": bisim["initial_related"],
            "identity_simulation_real_to_learned": forward["identity_related"],
            "identity_simulation_learned_to_real": reverse["identity_related"],
            "identity_bisimulation": bisim["identity_related"],
            "strict_identity_simulation_real_to_learned": strict["real_to_top1"],
            "strict_identity_simulation_learned_to_real": strict["top1_to_real"],
            "strict_identity_bisimulation": strict["bisimulation"],
            "relations": details,
        }
    return output


def count_score(values) -> dict:
    values = [v for v in values if v is not None]
    if any(type(v) is not bool for v in values):
        raise ValueError("Metric values must be bool or None, never truthy strings.")
    return {
        "passed": sum(values),
        "evaluated": len(values),
        "fraction": sum(values) / len(values) if values else None,
    }


def evaluate_model_pair(real, learned, *, top1_counts=None, relations=None) -> dict:
    """Score one graph pair without changing graphs, labels, initials or formulas.

    Top-1 counts describe the point predictor, never candidate-set coverage.
    Callers may attach hash-linked LTL results under diagnostics.ltl later.
    """
    structural = behavioral_metrics(real, learned, relations=relations)
    checkers = [CTLModelChecker(g) for g in (real, learned)]
    initial_rows, all_state_rows = [], []
    for prop in default_ctl_suite():
        truths = [c.satisfying_states(prop.formula) for c in checkers]
        rv, lv = (
            g.initial_states <= t for g, t in zip((real, learned), truths, strict=True)
        )
        initial_rows.append({"property": prop.name, "real": rv, "learned": lv})
        if real.states == learned.states:
            all_state_rows.extend(
                (s in truths[0]) == (s in truths[1]) for s in real.states
            )
    initial_agreement = all(r["real"] == r["learned"] for r in initial_rows)
    if (
        structural["action_insensitive"]["initial_bisimulation"]
        and not initial_agreement
    ):
        raise AssertionError("Bisimulation must preserve the existing CTL formulas.")
    if structural["action_insensitive"]["identity_bisimulation"] and not all(
        all_state_rows
    ):
        raise AssertionError("Identity bisimulation must preserve all-state CTL.")
    top1 = None
    if top1_counts is not None:
        correct, total = top1_counts
        if not (
            type(correct) is int
            and type(total) is int
            and 0 <= correct <= total
            and total > 0
        ):
            raise ValueError(
                "Top-1 requires integer correct/total counts with total > 0."
            )
        top1 = {"correct": correct, "total": total, "accuracy": correct / total}
    return {
        "behavioral": structural,
        "diagnostics": {
            "top1": top1,
            "initial_all_six_ctl_agreement": initial_agreement,
            "initial_ctl_verdicts": initial_rows,
            "initial_ctl_mismatches": [
                r for r in initial_rows if r["real"] != r["learned"]
            ],
            "all_state_ctl": count_score(all_state_rows),
            "ltl": None,
        },
    }


def summarize_model_pairs(cases) -> dict:
    """Uniform counts with explicit denominators and case-level mismatch lists."""
    cases = list(cases)
    primary = {
        semantics: {
            metric: count_score(c["behavioral"][semantics][metric] for c in cases)
            for metric in METRICS
        }
        for semantics in SEMANTICS
    }
    top1 = [
        c["diagnostics"]["top1"] for c in cases if c["diagnostics"]["top1"] is not None
    ]
    total = sum(t["total"] for t in top1)
    correct = sum(t["correct"] for t in top1)
    ctl = [c["diagnostics"]["all_state_ctl"] for c in cases]
    ltl = [
        c["diagnostics"]["ltl"] for c in cases if c["diagnostics"]["ltl"] is not None
    ]
    mismatches = []
    for c in cases:
        failures = [
            f"{s}.{m}"
            for s in SEMANTICS
            for m in METRICS
            if c["behavioral"][s][m] is False
        ]
        diagnostic = c["diagnostics"]
        ctl_gap = (
            diagnostic["initial_all_six_ctl_agreement"]
            and not c["behavioral"]["action_insensitive"]["initial_bisimulation"]
        )
        ltl_failures = (
            diagnostic["ltl"]["initial_mismatches"] if diagnostic["ltl"] else []
        )
        if failures or diagnostic["initial_ctl_mismatches"] or ltl_failures:
            mismatches.append(
                {
                    **{
                        k: c[k]
                        for k in ("seed", "case", "family", "split", "variant")
                        if k in c
                    },
                    "failed_metrics": failures,
                    "ctl_agrees_but_not_bisimilar": ctl_gap,
                    "initial_ctl_mismatches": diagnostic["initial_ctl_mismatches"],
                    "initial_ltl_mismatches": ltl_failures,
                }
            )
    return {
        "cases": len(cases),
        "primary_metrics": primary,
        "diagnostics": {
            "top1": {
                "correct": correct,
                "total": total,
                "evaluated_cases": len(top1),
                "accuracy": correct / total if total else None,
            },
            "initial_all_six_ctl_agreement": count_score(
                c["diagnostics"]["initial_all_six_ctl_agreement"] for c in cases
            ),
            "all_state_ctl": {
                "matched": sum(c["passed"] for c in ctl),
                "comparisons": sum(c["evaluated"] for c in ctl),
            },
            "initial_all_six_ltl_agreement": count_score(
                t["initial_all_six_agreement"] for t in ltl
            ),
            "all_state_ltl": {
                "matched": sum(t["matched"] for t in ltl),
                "comparisons": sum(t["comparisons"] for t in ltl),
            },
            "ctl_agrees_but_not_bisimilar": sum(
                m["ctl_agrees_but_not_bisimilar"] for m in mismatches
            ),
        },
        "mismatch_cases": mismatches,
    }


def model_evaluation_report(cases, **provenance) -> dict:
    cases = list(cases)
    return {
        "schema_version": 1,
        "metric_priority": {
            "primary": "simulation/bisimulation",
            "additional": "Top-1 and CTL/LTL diagnostics",
        },
        "definitions": DEFINITIONS,
        "provenance": provenance,
        "summary": summarize_model_pairs(cases),
        "by_seed": {
            str(s): summarize_model_pairs(c for c in cases if c.get("seed") == s)
            for s in sorted({c["seed"] for c in cases if "seed" in c})
        },
        "by_family": {
            s: summarize_model_pairs(c for c in cases if c.get("family") == s)
            for s in sorted({c["family"] for c in cases if "family" in c})
        },
        "by_split": {
            s: summarize_model_pairs(c for c in cases if c.get("split") == s)
            for s in sorted({c["split"] for c in cases if "split" in c})
        },
        "cases": cases,
    }


def save_model_evaluation(path, cases, **provenance) -> dict:
    """Write a separate report; refuse to overwrite previous experimental evidence."""
    path = Path(path)
    report = model_evaluation_report(cases, **provenance)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
