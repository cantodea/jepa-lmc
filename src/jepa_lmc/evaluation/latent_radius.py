"""Oracle radius diagnostics on a fixed JEPA, finite state set and exact labels.

No model fitting, nearest-neighbour fallback or edge repair happens here.
Radii fitted to test errors are oracle diagnostics, not generalization bounds.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from jepa_lmc.benchmarks.ctl_suite import default_ctl_suite
from jepa_lmc.envs.gridworld import GridWorld, State
from jepa_lmc.learning.data import gridworld_observation
from jepa_lmc.learning.model import ActionJEPA
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
    Formula,
    Not,
    Or,
)
from jepa_lmc.verification.gridworld import gridworld_to_transition_system
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem


@dataclass(frozen=True)
class LatentDistances:
    """Rows are all (state, action) pairs; columns are same-map target states."""

    states: tuple[State, ...]
    pairs: tuple[tuple[State, int], ...]
    true_indices: tuple[int, ...]
    distances: Tensor

    def __post_init__(self) -> None:
        if not self.states or not self.pairs:
            raise ValueError("Distance table must not be empty.")
        if len(set(self.states)) != len(self.states):
            raise ValueError("Duplicate candidate states.")
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("Duplicate state-action pairs.")
        if len(self.true_indices) != len(self.pairs) or any(
            not 0 <= index < len(self.states) for index in self.true_indices
        ):
            raise ValueError("Invalid true successor indices.")
        if self.distances.shape != (len(self.pairs), len(self.states)):
            raise ValueError("Distance table shape does not match its indices.")
        if not bool(torch.isfinite(self.distances).all()) or bool(
            (self.distances < 0).any()
        ):
            raise ValueError("Distances must be finite and non-negative.")

    @property
    def errors(self) -> Tensor:
        rows = torch.arange(len(self.pairs), device=self.distances.device)
        return self.distances[rows, list(self.true_indices)]

    def candidates(self, epsilon: float) -> Tensor:
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("Epsilon must be finite and non-negative.")
        # A closed ball, including ties. Errors and membership share one table.
        return self.distances <= epsilon


@torch.no_grad()
def collect_latent_distances(model: ActionJEPA, env: GridWorld) -> LatentDistances:
    """Evaluate frozen encoders/predictor without changing parameters or modes.

    d is unnormalised Euclidean distance, as in the existing retrieval evaluator.
    Compute distances directly in float64 to avoid float32 cdist cancellation
    near zero. This is numerical evaluation, not certified real arithmetic.
    """
    states = tuple(env.all_states())
    pairs = tuple((state, action) for state in states for action in env.ACTIONS)
    indices = {state: index for index, state in enumerate(states)}
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        observations = torch.stack(
            [gridworld_observation(env, state) for state in states]
        ).to(device)
        targets = model.target_encoder(observations)
        context = model.context_encoder(observations)
        sources = torch.tensor([indices[state] for state, _ in pairs], device=device)
        actions = torch.tensor([action for _, action in pairs], device=device)
        predictions = model.predictor(context[sources], actions)
        distances = torch.cdist(
            predictions.double(),
            targets.double(),
            compute_mode="donot_use_mm_for_euclid_dist",
        ).cpu()
    finally:
        for module, training in modes:
            module.training = training
    return LatentDistances(
        states,
        pairs,
        tuple(indices[env.transition(state, action)] for state, action in pairs),
        distances,
    )


def oracle_radii(tables: Iterable[LatentDistances]) -> dict[str, float]:
    """Pool pairs, not map averages; quantiles use linear interpolation."""
    errors = [table.errors.double().cpu() for table in tables]
    if not errors:
        raise ValueError("At least one error table is required.")
    pooled = torch.cat(errors)
    return {
        "epsilon_max": float(pooled.max()),
        "epsilon_95": float(torch.quantile(pooled, 0.95)),
        "epsilon_99": float(torch.quantile(pooled, 0.99)),
    }


def _validate_mask(env: GridWorld, table: LatentDistances, mask: Tensor) -> None:
    expected = {(state, action) for state in env.all_states() for action in env.ACTIONS}
    if set(table.states) != set(env.all_states()) or set(table.pairs) != expected:
        raise ValueError("Distance table must enumerate this entire environment.")
    if any(
        table.states[index] != env.transition(state, action)
        for (state, action), index in zip(table.pairs, table.true_indices, strict=True)
    ):
        raise ValueError("True successors do not match this environment.")
    if mask.shape != table.distances.shape or mask.dtype != torch.bool:
        raise ValueError("Candidate mask must be boolean and match the distance table.")


def candidate_metrics(
    env: GridWorld,
    table: LatentDistances,
    mask: Tensor,
) -> dict:
    """Audit action-labelled inclusion separately from CTL's unlabelled R."""
    _validate_mask(env, table, mask)
    sizes = mask.sum(dim=1).tolist()
    learned_action_edges = {
        (state, action, table.states[index])
        for row, (state, action) in enumerate(table.pairs)
        for index in mask[row].nonzero().flatten().tolist()
    }
    exact_action_edges = {
        (state, action, table.states[index])
        for (state, action), index in zip(table.pairs, table.true_indices, strict=True)
    }
    learned_edges = {(state, target) for state, _, target in learned_action_edges}
    exact_edges = {(state, target) for state, _, target in exact_action_edges}
    missing_action = exact_action_edges - learned_action_edges
    missing = exact_edges - learned_edges
    deadlocks = set(table.states) - {state for state, _ in learned_edges}
    singleton_correct = sum(
        sizes[row] == 1 and bool(mask[row, index])
        for row, index in enumerate(table.true_indices)
    )
    count = len(table.pairs)
    return {
        "pairs": count,
        "covered_pairs": count - len(missing_action),
        "successor_coverage": 1 - len(missing_action) / count,
        "candidate_count": sum(sizes),
        "mean_candidate_size": sum(sizes) / count,
        "mean_candidate_fraction": sum(sizes) / (count * len(table.states)),
        "singleton_count": sizes.count(1),
        "singleton_fraction": sizes.count(1) / count,
        "singleton_correct_count": singleton_correct,
        "singleton_precision": _ratio(singleton_correct, sizes.count(1)),
        "empty_count": sizes.count(0),
        "empty_fraction": sizes.count(0) / count,
        "action_inclusion": not missing_action,
        "relation_inclusion": not missing,
        "relation_equal": learned_edges == exact_edges,
        "exact_edges": len(exact_edges),
        "candidate_edges": len(learned_edges),
        "missing_edges": len(missing),
        "extra_edges": len(learned_edges - exact_edges),
        "deadlock_states": sorted(deadlocks),
        "missing_action_edges": sorted(missing_action),
    }


def candidate_transition_system(
    env: GridWorld,
    table: LatentDistances,
    mask: Tensor,
) -> ExplicitTransitionSystem[State, int]:
    """Build the literal candidate graph; reject non-total graphs, never fill gaps."""
    _validate_mask(env, table, mask)
    exact = gridworld_to_transition_system(env)
    transitions: dict[State, list[tuple[int, State]]] = {
        state: [] for state in table.states
    }
    for row, (state, action) in enumerate(table.pairs):
        transitions[state].extend(
            (action, table.states[index])
            for index in mask[row].nonzero().flatten().tolist()
        )
    return ExplicitTransitionSystem(
        states=table.states,
        initial_states=exact.initial_states,
        transitions=transitions,
        labels={state: exact.propositions(state) for state in table.states},
    )


def truth_monotonicity(formula: Formula) -> int | None:
    """Under edge addition: +1 truth grows, -1 shrinks, 0 unchanged, None unknown.

    This conservative syntactic classifier intentionally abstains for mixed CTL
    such as AG(EF p). It assumes the same states/labels and total relations.
    """
    if isinstance(formula, Atom):
        return 0
    if isinstance(formula, Not):
        child = truth_monotonicity(formula.formula)
        return None if child is None else -child
    if isinstance(formula, (And, Or, EU)):
        children = (
            (formula.condition, formula.target)
            if isinstance(formula, EU)
            else (formula.left, formula.right)
        )
        directions = {truth_monotonicity(child) for child in children} - {0}
        if isinstance(formula, EU):
            directions.add(1)
        if None in directions or len(directions) > 1:
            return None
        return next(iter(directions), 0)
    if isinstance(formula, (EX, EF, EG, AX, AF, AG)):
        direction = 1 if isinstance(formula, (EX, EF, EG)) else -1
        child = truth_monotonicity(formula.formula)
        return direction if child in (0, direction) else None
    return None


def ctl_outcomes(
    exact: ExplicitTransitionSystem,
    learned: ExplicitTransitionSystem,
) -> list[dict]:
    """Empirical query outcomes and one-sided claims conditional on inclusion."""
    if exact.states != learned.states or exact.initial_states != learned.initial_states:
        raise ValueError("CTL transfer requires identical states and initial states.")
    if any(exact.propositions(s) != learned.propositions(s) for s in exact.states):
        raise ValueError("CTL transfer requires identical labels.")
    exact_checker = CTLModelChecker(exact)
    learned_checker = CTLModelChecker(learned)
    rows = []
    for spec in default_ctl_suite():
        direction = truth_monotonicity(spec.formula)
        for state in sorted(exact.states):
            truth = exact_checker.holds(state, spec.formula)
            prediction = learned_checker.holds(state, spec.formula)
            # Universal true and existential false transfer from an overapprox.
            claim = (
                direction == 0
                or (direction == -1 and prediction)
                or (direction == 1 and not prediction)
            )
            rows.append(
                {
                    "state": state,
                    "property": spec.name,
                    "ground_truth": truth,
                    "learned": prediction,
                    "monotonicity": direction,
                    "one_sided_claim": claim,
                    "false_safe": spec.safety_claim and prediction and not truth,
                }
            )
    return rows


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_ctl(rows: Iterable[dict]) -> dict:
    outcomes = tuple(rows)
    tp = sum(row["ground_truth"] and row["learned"] for row in outcomes)
    tn = sum(not row["ground_truth"] and not row["learned"] for row in outcomes)
    fp = sum(not row["ground_truth"] and row["learned"] for row in outcomes)
    fn = sum(row["ground_truth"] and not row["learned"] for row in outcomes)
    claims = [row for row in outcomes if row["one_sided_claim"]]
    violations = sum(row["ground_truth"] != row["learned"] for row in claims)
    positive_recall = _ratio(tp, tp + fn)
    negative_recall = _ratio(tn, tn + fp)
    return {
        "queries": len(outcomes),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "agreement": _ratio(tp + tn, len(outcomes)),
        "positive_precision": _ratio(tp, tp + fp),
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "balanced_accuracy": (
            (positive_recall + negative_recall) / 2
            if positive_recall is not None and negative_recall is not None
            else None
        ),
        "ground_truth_positive_rate": _ratio(tp + fn, len(outcomes)),
        "false_safe_count": sum(row["false_safe"] for row in outcomes),
        "one_sided_claims": len(claims),
        "one_sided_violations": violations,
        "empirical_one_sided_soundness": _ratio(len(claims) - violations, len(claims)),
        "one_sided_decisive_fraction": _ratio(len(claims), len(outcomes)),
    }


def evaluate_candidates(
    env: GridWorld,
    table: LatentDistances,
    mask: Tensor,
) -> dict:
    metrics = candidate_metrics(env, table, mask)
    rows = []
    if not metrics["deadlock_states"]:
        rows = ctl_outcomes(
            gridworld_to_transition_system(env),
            candidate_transition_system(env, table, mask),
        )
    metrics.update(
        {
            "ctl_status": "not_total" if metrics["deadlock_states"] else "evaluated",
            "ctl": summarize_ctl(rows),
            "ctl_by_property": {
                spec.name: summarize_ctl(
                    row for row in rows if row["property"] == spec.name
                )
                for spec in default_ctl_suite()
            },
            "ctl_outcomes": rows,
        }
    )
    return metrics


def aggregate_candidate_results(results: Iterable[Mapping]) -> dict:
    """Micro-average pairs/queries and retain skipped CTL-map denominators."""
    maps = tuple(results)
    if not maps:
        raise ValueError("At least one map result is required.")
    pairs = sum(result["pairs"] for result in maps)
    count = sum(result["candidate_count"] for result in maps)
    singletons = sum(result["singleton_count"] for result in maps)
    rows = [row for result in maps for row in result["ctl_outcomes"]]
    by_property = {
        spec.name: summarize_ctl(row for row in rows if row["property"] == spec.name)
        for spec in default_ctl_suite()
    }
    primary_scores = [
        by_property[spec.name]["balanced_accuracy"]
        for spec in default_ctl_suite()
        if spec.primary_score
    ]
    transfer_claims = sum(
        result["ctl"]["one_sided_claims"]
        for result in maps
        if result["relation_inclusion"]
    )
    return {
        "maps": len(maps),
        "pairs": pairs,
        "successor_coverage": sum(r["covered_pairs"] for r in maps) / pairs,
        "mean_candidate_size": count / pairs,
        "mean_candidate_fraction": sum(
            r["mean_candidate_fraction"] * r["pairs"] for r in maps
        )
        / pairs,
        "singleton_fraction": singletons / pairs,
        "singleton_precision": _ratio(
            sum(r["singleton_correct_count"] for r in maps),
            singletons,
        ),
        "empty_fraction": sum(r["empty_count"] for r in maps) / pairs,
        "action_inclusion_maps": sum(r["action_inclusion"] for r in maps),
        "relation_inclusion_maps": sum(r["relation_inclusion"] for r in maps),
        "relation_equal_maps": sum(r["relation_equal"] for r in maps),
        "missing_edges": sum(r["missing_edges"] for r in maps),
        "extra_edges": sum(r["extra_edges"] for r in maps),
        "ctl_evaluated_maps": sum(r["ctl_status"] == "evaluated" for r in maps),
        "ctl_skipped_maps": sum(r["ctl_status"] != "evaluated" for r in maps),
        "ctl": summarize_ctl(rows),
        "ctl_by_property": by_property,
        "primary_balanced_score": (
            sum(primary_scores) / len(primary_scores)
            if all(score is not None for score in primary_scores)
            else None
        ),
        "audited_transfer_claims": transfer_claims,
        "audited_transfer_fraction": _ratio(transfer_claims, len(rows)),
    }
