"""Non-immediate CTL scoring and the prespecified research stopping rule."""

from __future__ import annotations

from collections.abc import Iterable
from math import isclose

from jepa_lmc.benchmarks.radius_stress import (
    PRIMARY_PROPERTIES,
    RadiusStressCase,
    exact_transfer_opportunity,
    label_immediate,
)
from jepa_lmc.evaluation.latent_radius import (
    aggregate_candidate_results,
    summarize_ctl,
)


def annotate_stress_result(case: RadiusStressCase, result: dict) -> dict:
    env = case.make_env()
    rows = [
        {
            **row,
            "case": case.name,
            "family": case.family,
            "label_immediate": label_immediate(env, row["state"], row["property"]),
            "relation_inclusion": result["relation_inclusion"],
        }
        for row in result["ctl_outcomes"]
    ]
    return {**result, "case": case.name, "family": case.family, "ctl_outcomes": rows}


def score_nontrivial(rows: Iterable[dict]) -> dict:
    primary = [
        row
        for row in rows
        if row["property"] in PRIMARY_PROPERTIES and not row["label_immediate"]
    ]
    by_property = {
        name: summarize_ctl(row for row in primary if row["property"] == name)
        for name in PRIMARY_PROPERTIES
    }
    balanced = [value["balanced_accuracy"] for value in by_property.values()]
    opportunities = [row for row in primary if exact_transfer_opportunity(row)]
    recovered = [
        row
        for row in opportunities
        if row["relation_inclusion"]
        and row["one_sided_claim"]
        and row["ground_truth"] == row["learned"]
    ]
    return {
        "queries": len(primary),
        "ctl": summarize_ctl(primary),
        "by_property": by_property,
        "primary_balanced_score": (
            sum(balanced) / len(balanced)
            if all(score is not None for score in balanced)
            else None
        ),
        "transfer_opportunities": len(opportunities),
        "recovered_opportunities": len(recovered),
        "opportunity_recall": (
            len(recovered) / len(opportunities) if opportunities else None
        ),
        "recovered_fraction_of_queries": len(recovered) / len(primary)
        if primary
        else None,
        "maps_with_recovered_opportunity": sorted({row["case"] for row in recovered}),
    }


def summarize_stress_maps(results: Iterable[dict]) -> dict:
    maps = tuple(results)
    if not maps:
        raise ValueError("At least one stress map is required.")
    rows = [row for result in maps for row in result["ctl_outcomes"]]
    return {
        **aggregate_candidate_results(maps),
        "nontrivial": score_nontrivial(rows),
        "nontrivial_by_family": {
            family: score_nontrivial(row for row in rows if row["family"] == family)
            for family in sorted({result["family"] for result in maps})
        },
    }


def evaluate_seed_gate(summaries: dict, protocol: dict) -> dict:
    """A diagnostic quantile or a strong pooled seed cannot rescue a failure."""
    criteria = protocol["continue_only_if_every_seed_passes"]
    primary = summaries[protocol["evaluation"]["primary_variant"]]
    baseline = summaries["all_states"]
    nontrivial = primary["nontrivial"]
    score = nontrivial["primary_balanced_score"]
    baseline_score = baseline["nontrivial"]["primary_balanced_score"]
    gain = (
        score - baseline_score
        if score is not None and baseline_score is not None
        else None
    )
    recall = nontrivial["opportunity_recall"]
    expected_maps = protocol["maps"]["total"]
    family_counts = {
        family: len(
            primary["nontrivial_by_family"][family]["maps_with_recovered_opportunity"]
        )
        for family in criteria["required_opportunity_families"]
    }

    def reaches(value: float | None, threshold: float) -> bool:
        # Inclusive decimal thresholds must survive binary subtraction, e.g. .6-.5.
        return value is not None and (
            value >= threshold or isclose(value, threshold, rel_tol=0, abs_tol=1e-12)
        )

    checks = {
        "all_maps_evaluated": primary["ctl_evaluated_maps"] == expected_maps,
        "full_action_coverage": primary["successor_coverage"]
        == criteria["action_successor_coverage"],
        "full_relation_inclusion": primary["relation_inclusion_maps"]
        == criteria["maps_with_relation_inclusion"],
        "zero_one_sided_violations": primary["ctl"]["one_sided_violations"]
        == criteria["one_sided_violations"],
        "balanced_gain": reaches(
            gain, criteria["minimum_primary_balanced_gain_over_all_states"]
        ),
        "opportunity_recall": reaches(
            recall, criteria["minimum_nontrivial_opportunity_recall"]
        ),
        "family_consistency": all(
            count
            >= criteria["minimum_maps_with_recovered_opportunity_per_required_family"]
            for count in family_counts.values()
        ),
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "balanced_gain_over_all_states": gain,
        "nontrivial_opportunity_recall": recall,
        "recovered_maps_by_required_family": family_counts,
    }


def evaluate_overall_gate(seed_gates: dict[int, dict], protocol: dict) -> dict:
    expected = set(protocol["model_seeds"])
    if set(seed_gates) != expected:
        raise ValueError("The decision requires exactly the prespecified model seeds.")
    passed = all(gate["passes"] for gate in seed_gates.values())
    return {
        "status": "continue" if passed else "stop",
        "all_seeds_pass": passed,
        "failed_seeds": sorted(
            seed for seed, gate in seed_gates.items() if not gate["passes"]
        ),
        "scope": protocol["decision_scope"],
        "failure_action": None if passed else protocol["failure_action"],
    }
