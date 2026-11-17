"""Offline learning from whole-trace validity labels, without a transition oracle.

Hypotheses are all total action-deterministic functions on the known catalogue.
An invalid trace is one forbidden conjunction, NOT a collection of bad edges.
Counting uses exact finite-domain decision recursion with unconstrained factors;
the explicit resource limit returns unknown, never an approximate exact count.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache

import torch
from torch import Tensor
from torch.nn import functional as F

from jepa_lmc.evaluation.trace_sanity import KnownProblem, Trace
from jepa_lmc.learning.finite_trace import (
    DirectSuccessorMLP,
    FiniteTraceData,
    fit_finite_model,
    tensor_digest,
    training_settings,
)
from jepa_lmc.learning.model import ActionJEPA, JEPAOutput, action_jepa_loss
from jepa_lmc.learning.ranking import same_map_ranking_loss
from jepa_lmc.learning.training import make_action_jepa_optimizer

Clause = tuple[tuple[int, int], ...]


def encode_trace(problem: KnownProblem, trace: Trace) -> Clause | None:
    """One equality per distinct pair; None means impossible in this class.

Revisiting (s,a) reuses the SAME sampled deterministic successor. Conflicting
requirements therefore have zero acceptance probability under every hypothesis.
"""
    if not problem.accepts_syntax(trace):
        return None
    states = {s: i for i, s in enumerate(problem.states)}
    actions = {a: i for i, a in enumerate(problem.actions)}
    requirements = {}
    for s, a, t in trace.edges:
        pair = states[s] * len(actions) + actions[a]
        target = states[t]
        if pair in requirements and requirements[pair] != target:
            return None
        requirements[pair] = target
    return tuple(sorted(requirements.items()))


def _canonical(clauses):
    return tuple(sorted(set(clauses)))


def _condition(clauses, variable, value):
    result = []
    for clause in clauses:
        requirements = dict(clause)
        if variable in requirements:
            if requirements[variable] != value:
                continue  # This forbidden conjunction cannot now hold.
            del requirements[variable]
        result.append(tuple(sorted(requirements.items())))
    return _canonical(result)


class CountingLimit(RuntimeError):
    pass


class _FiniteCounter:
    def __init__(self, n: int, max_nodes: int):
        self.n, self.max_nodes, self.nodes = n, max_nodes, 0
        self.count_active = lru_cache(maxsize=None)(self._count_active)

    def _count_active(self, clauses):
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise CountingLimit("Exact decision-node budget exhausted")
        if () in clauses:
            return 0
        if not clauses:
            return 1
        frequency = Counter(v for clause in clauses for v, _ in clause)
        variable = min(frequency, key=lambda v: (-frequency[v], v))
        active = set(frequency)
        total = 0
        # All unmentioned values induce the same residual clauses.
        mentioned = {t for clause in clauses for v, t in clause if v == variable}
        values = [(v, 1) for v in sorted(mentioned)]
        if len(mentioned) < self.n:
            other = next(v for v in range(self.n) if v not in mentioned)
            values.append((other, self.n - len(mentioned)))
        for value, multiplicity in values:
            residual = _condition(clauses, variable, value)
            remaining = {v for clause in residual for v, _ in clause}
            free = len(active) - 1 - len(remaining)
            total += multiplicity * self.n ** free * self.count_active(residual)
        return total

    def count(self, clauses, variables):
        active = {v for clause in clauses for v, _ in clause}
        return self.n ** (len(variables) - len(active)) * self.count_active(clauses)


def analyze_version_space(
    problem: KnownProblem,
    good: Sequence[Trace],
    bad: Sequence[Trace],
    *,
    max_nodes: int = 250000,
) -> dict:
    """Exact counts, marginal supports and deterministic symbolic completion.

All returned counts are decimal strings for portable JSON (they can be huge).
Redundancy relative to G, sequential redundancy and leave-one-out redundancy
are different quantities; all are recorded in input order. A count reduction
may occur while every individual marginal support remains unchanged.
"""
    if not problem.states or not problem.actions or max_nodes < 1:
        raise ValueError("Need nonempty finite catalogues and a positive node budget")
    if len(set(problem.states)) != len(problem.states) or len(set(problem.actions)) != len(problem.actions):
        raise ValueError("Catalogue identities must be unique")
    n, na = len(problem.states), len(problem.actions)
    fixed = {}
    for trace in good:
        encoded = encode_trace(problem, trace)
        if encoded is None:
            raise ValueError("Positive traces contradict syntax or determinism")
        for variable, target in encoded:
            if variable in fixed and fixed[variable] != target:
                raise ValueError("Positive traces impose contradictory successors")
            fixed[variable] = target
    variables = tuple(v for v in range(n * na) if v not in fixed)
    constraints, reasons = [], []
    for trace in bad:
        encoded = encode_trace(problem, trace)
        if encoded is None:
            constraints.append(None)
            reasons.append("syntax_or_internal_determinism_conflict")
        elif any(v in fixed and fixed[v] != t for v, t in encoded):
            constraints.append(None)
            reasons.append("contradicts_positive_edge")
        else:
            clause = tuple((v, t) for v, t in encoded if v not in fixed)
            constraints.append(clause)
            reasons.append("forbids_unknown_conjunction" if clause else "contradicts_positive_dataset")
    base = n ** len(variables)
    counter = _FiniteCounter(n, max_nodes)
    started = time.perf_counter()
    result = {
        "hypothesis_class": "all total deterministic functions S x A -> S; fixed state identities; no topology prior",
        "count_method": "exact finite-domain decision recursion with memoization and free-variable factors",
        "max_nodes": max_nodes,
        "model_count_good": str(base),
        "known_pairs": len(fixed),
        "unknown_pairs": len(variables),
        "negative_count": len(bad),
        "negative_diagnostics": [
            {"index": i, "reason": reasons[i], "redundant_given_good": clause is None,
             "unknown_pairs_in_constraint": 0 if clause is None else len(clause),
             "standalone_excluded_fraction": 0.0 if clause is None else n ** (-len(clause))}
            for i, clause in enumerate(constraints)
        ],
    }
    all_clauses = _canonical(c for c in constraints if c is not None)
    try:
        total = counter.count(all_clauses, variables)
        result.update(status="exact", satisfiable=total > 0,
                      model_count_good_bad=str(total),
                      information_bits=None if total == 0 else math.log2(base) - math.log2(total))
        previous, prefix = base, []
        for i, clause in enumerate(constraints):
            if clause is not None:
                prefix.append(clause)
            current = counter.count(_canonical(prefix), variables)
            without = counter.count(_canonical(c for j, c in enumerate(constraints) if j != i and c is not None), variables)
            result["negative_diagnostics"][i].update(
                sequential_redundant=current == previous,
                redundant_given_all_other_bad=without == total,
                sequential_model_count=str(current),
            )
            previous = current
        candidates, completion = [], dict(fixed)
        for variable in range(n * na):
            s, a = problem.states[variable // na], problem.actions[variable % na]
            if total == 0:
                allowed = []
            elif variable in fixed:
                allowed = [fixed[variable]]
            else:
                allowed = [t for t in range(n) if counter.count(_condition(all_clauses, variable, t),
                           tuple(v for v in variables if v != variable)) > 0]
            candidates.append({"source": s, "action": a, "observed": variable in fixed,
                               "candidate_count": len(allowed),
                               "successors": [problem.states[t] for t in allowed]})
        if total:
            residual, pending = all_clauses, list(variables)
            for variable in variables:
                pending.remove(variable)
                for target in range(n):
                    conditioned = _condition(residual, variable, target)
                    if counter.count(conditioned, pending):
                        completion[variable] = target
                        residual = conditioned
                        break
        result.update(
            candidate_successors=candidates,
            unknown_pairs_with_reduced_support=sum(
                not row["observed"] and row["candidate_count"] < n for row in candidates
            ),
            completion_edges=[] if not total else [
                (problem.states[v // na], problem.actions[v % na], problem.states[completion[v]])
                for v in range(n * na)
            ],
            completion_policy="lexicographically smallest feasible target-index vector in catalogue order",
        )
    except CountingLimit as exc:
        result.update(status="resource_limit", error=str(exc), satisfiable=None,
                      model_count_good_bad=None, information_bits=None)
        result.pop("candidate_successors", None)
        result.pop("completion_edges", None)
    result.update(decision_nodes=counter.nodes, analysis_seconds=time.perf_counter() - started)
    return result


def trace_negative_loss(logits: Tensor, encoded_traces: Sequence[Clause | None]) -> Tensor:
    """Mean -log P(trace rejected) under independent categorical function values.

One function value is drawn for each DISTINCT (s,a), not each occurrence.
Impossible traces contribute exactly zero, and stay in the sample denominator.
Log acceptance is capped below zero by 1e-12 for finite machine loss; this
numerical cap is fixed before experimentation, not tuned on evaluation data.
"""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("Need pair-by-successor logits and at least two states")
    zero = logits.sum() * 0.0
    if not encoded_traces:
        return zero
    log_probs = F.log_softmax(logits.double(), dim=1)
    losses = []
    for clause in encoded_traces:
        if clause is None:
            losses.append(zero.double())
        elif not clause:
            raise ValueError("An empty executable trace cannot receive a negative label")
        else:
            rows = torch.tensor([v for v, _ in clause], device=logits.device)
            targets = torch.tensor([t for _, t in clause], device=logits.device)
            log_accept = log_probs[rows, targets].sum().clamp(max=-1e-12)
            losses.append(-torch.log(-torch.expm1(log_accept)))
    return torch.stack(losses).mean().to(logits.dtype)


def fit_with_negative_traces(
    data: FiniteTraceData,
    bad: Sequence[Trace],
    *,
    kind: str,
    seed: int,
    config: dict,
    negative_weight: float = 1.0,
):
    """Same initialization/base loss/schedule as fit_finite_model plus trace NLL.

Inputs contain S,L, state images, positive edges and candidate trace labels only.
No environment, oracle correction, error position, test metric or true R enters.
"""
    if not math.isfinite(negative_weight) or negative_weight < 0:
        raise ValueError("negative_weight must be finite and nonnegative")
    if not bad or negative_weight == 0:
        model, metadata = fit_finite_model(data, kind=kind, seed=seed, config=config)
        metadata.update(negative_weight=negative_weight, negative_trace_count=len(bad))
        return model, metadata
    train = training_settings(config, kind)
    if train["epochs"] < 1 or train["device"] != "cpu":
        raise ValueError("This reproducibility protocol uses positive epochs on CPU")
    encoded = tuple(encode_trace(data.problem, t) for t in bad)
    if any(c == () for c in encoded):
        raise ValueError("Empty executions cannot be negative examples")
    torch.set_num_threads(train["threads"])
    torch.use_deterministic_algorithms(train["deterministic_algorithms"])
    torch.manual_seed(seed)
    height, width = data.observations.shape[-2:]
    kwargs = {"height": height, "width": width, "num_actions": len(data.problem.actions)}
    if kind == "mlp":
        kwargs.update(num_states=len(data.problem.states), hidden_dim=config["mlp"]["hidden_dim"],
                      action_dim=config["mlp"]["action_dim"])
        model = DirectSuccessorMLP(**kwargs)
    else:
        kwargs.update({k: config["jepa"][k] for k in (
            "latent_dim", "hidden_channels", "position_dim", "position_scale", "action_dim",
            "predictor_hidden_dim", "predictor_residual_blocks")})
        model = ActionJEPA(**kwargs)
    initial_sha = tensor_digest(*model.state_dict().items())
    optimizer = make_action_jepa_optimizer(model, learning_rate=train["learning_rate"],
                                          weight_decay=train["weight_decay"])
    n, na = len(data.problem.states), len(data.problem.actions)
    all_sources = torch.arange(n).repeat_interleave(na)
    all_actions = torch.arange(na).repeat(n)
    query_maps = torch.zeros(len(data.actions), dtype=torch.long)
    candidate_maps = torch.zeros(n, dtype=torch.long)
    history, started = [], time.perf_counter()
    for epoch in range(1, train["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if kind == "mlp":
            observed_logits = model(data.observations[data.sources], data.actions)
            base_loss = F.cross_entropy(observed_logits, data.successors)
            all_logits = model(data.observations[all_sources], all_actions)
            losses = {"cross_entropy": base_loss}
        else:
            context = model.context_encoder(data.observations)
            prediction = model.predictor(context[data.sources], data.actions)
            with torch.no_grad():
                bank = model.target_encoder(data.observations)
            jepa = config["jepa"]
            base = action_jepa_loss(JEPAOutput(context[data.sources], prediction, bank[data.successors]),
                                    variance_weight=jepa["variance_weight"], covariance_weight=jepa["covariance_weight"])
            ranking = same_map_ranking_loss(prediction, bank, data.successors, query_maps, candidate_maps,
                                           temperature=jepa["ranking_temperature"])
            base_loss = base.total + jepa["ranking_weight"] * ranking
            predictions = model.predictor(context[all_sources], all_actions)
            all_logits = -(predictions[:, None, :] - bank[None, :, :]).square().sum(-1) / jepa["ranking_temperature"]
            losses = {"prediction": base.prediction, "variance": base.variance,
                      "covariance": base.covariance, "ranking": ranking}
        negative = trace_negative_loss(all_logits, encoded)
        loss = base_loss + negative_weight * negative
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite whole-trace training loss")
        loss.backward()
        optimizer.step()
        if kind == "jepa":
            model.update_target_encoder(config["jepa"]["ema_momentum"])
        if epoch == 1 or epoch % 50 == 0 or epoch == train["epochs"]:
            history.append({"epoch": epoch, "loss": float(loss.detach()),
                            "negative_trace_nll": float(negative.detach()),
                            **{k: float(v.detach()) for k, v in losses.items()}})
    actual = tensor_digest(("known_observations", data.observations), ("observed_sources", data.sources),
                           ("observed_actions", data.actions), ("observed_successors", data.successors))
    if actual != data.training_sha256:
        raise AssertionError("Training data were mutated")
    model.eval()
    negative_json = json.dumps([{"states": t.states, "actions": t.actions} for t in bad], separators=(",", ":"))
    return model, {
        "kind": kind, "initialization_seed": seed, "from_scratch": True, "model_kwargs": kwargs,
        "epochs": train["epochs"], "learning_rate": train["learning_rate"], "weight_decay": train["weight_decay"],
        "optimizer_steps": train["epochs"], "positive_examples_processed": train["epochs"] * len(data.actions),
        "negative_trace_count": len(bad), "negative_weight": negative_weight,
        "negative_traces_processed": train["epochs"] * len(bad),
        "negative_pair_logits_per_epoch": n * na,
        "negative_loss_definition": "mean -log(1-product over distinct pair equalities); conflicting repeats have zero loss",
        "negative_log_acceptance_cap": -1e-12,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "initial_state_sha256": initial_sha, "final_state_sha256": tensor_digest(*model.state_dict().items()),
        "training_sha256": data.training_sha256, "trace_sha256": data.trace_sha256,
        "negative_sha256": hashlib.sha256(negative_json.encode()).hexdigest(),
        "training_seconds": time.perf_counter() - started, "history": history,
        "checkpoint_selection": "fixed final epoch; no evaluator input",
    }
