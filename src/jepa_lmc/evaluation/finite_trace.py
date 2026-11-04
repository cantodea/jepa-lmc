"""Evaluator-only truth comparisons and held-out trace diagnostics.

No training/selection lives here. Raw observed-edge scores remain separate
from the self-loop completion used by the unchanged structural evaluator.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from jepa_lmc.benchmarks.ltl_suite import default_ltl_suite
from jepa_lmc.evaluation.structural import evaluate_model_pair
from jepa_lmc.evaluation.trace_sanity import (
    KnownProblem,
    is_execution,
    relation_edges,
    sample_good_traces,
)
from jepa_lmc.verification.ctl import AF, AG, Atom, CTLModelChecker, Not
from jepa_lmc.verification.nuxmv import (
    LTLQuery,
    export_nusmv_ltl_queries,
    find_nusmv_executable,
    run_nusmv_model,
)
from jepa_lmc.verification.transition_system import ExplicitTransitionSystem


def disjoint_heldout_traces(env, training_pool, *, seed, count=64, length=16):
    """Oracle-side independent stream; exclude full training pool and duplicates."""
    forbidden = set(training_pool)
    size = count + len(training_pool)
    while size <= 100_000:
        result, seen = [], set(forbidden)
        for trace in sample_good_traces(env, seed=seed, count=size, length=length):
            if trace not in seen:
                result.append(trace)
                seen.add(trace)
            if len(result) == count:
                if not all(is_execution(env, t) for t in result):
                    raise AssertionError("Held-out traces must be real executions.")
                return tuple(result)
        size *= 2
    raise ValueError("Cannot generate enough distinct held-out executions.")


def pair_partition(
    problem: KnownProblem, observed_pairs
) -> tuple[frozenset, frozenset]:
    domain = frozenset((s, a) for s in problem.states for a in problem.actions)
    observed = frozenset(observed_pairs)
    if not observed <= domain:
        raise ValueError("Observed pairs are outside known S x A.")
    return observed, domain - observed


def prediction_system(problem: KnownProblem, predictions) -> ExplicitTransitionSystem:
    """A total learned system built only from predictions and known S,I,A,L."""
    domain = {(s, a) for s in problem.states for a in problem.actions}
    if set(predictions) != domain:
        raise ValueError("Predictions must cover exactly known S x A.")
    if any(t not in problem.states for t in predictions.values()):
        raise ValueError("Predictions must select a known legal state.")
    return ExplicitTransitionSystem(
        states=problem.states,
        initial_states=problem.initial_states,
        labels=problem.labels,
        transitions={
            s: [(a, predictions[s, a]) for a in problem.actions] for s in problem.states
        },
    )


def rate(correct, total):
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def transition_scores(problem, observed_pairs, predictions, real) -> dict:
    """Each real (s,a) once. Missing predictions count as abstention/errors."""
    observed, unseen = pair_partition(problem, observed_pairs)
    truth = {(s, a): t for s, a, t in relation_edges(real)}
    if set(truth) != observed | unseen or len(truth) != len(relation_edges(real)):
        raise ValueError("Reference must be total and action deterministic on S x A.")
    if not set(predictions) <= set(truth):
        raise ValueError("Predicted queries are outside the evaluation domain.")
    reachable, todo = set(real.initial_states), list(real.initial_states)
    while todo:
        for t in real.successors(todo.pop()):
            if t not in reachable:
                reachable.add(t)
                todo.append(t)
    subsets = {
        "observed": observed,
        "unseen": unseen,
        "overall": observed | unseen,
        "reachable_unseen": {p for p in unseen if p[0] in reachable},
        "unreachable_unseen": {p for p in unseen if p[0] not in reachable},
    }
    return {
        name: {
            **rate(
                sum(p in predictions and predictions[p] == truth[p] for p in pairs),
                len(pairs),
            ),
            "predicted": sum(p in predictions for p in pairs),
        }
        for name, pairs in subsets.items()
    }


def trace_scores(problem, traces, predictions, observed_pairs) -> dict:
    """Action-sensitive complete-trace membership, plus first failed real edge."""
    records, failure_steps, failing_edges = [], Counter(), Counter()
    for index, trace in enumerate(traces):
        if not problem.accepts_syntax(trace):
            raise ValueError("Evaluation trace has invalid syntax.")
        unseen_count = sum((s, a) not in observed_pairs for s, a, _ in trace.edges)
        failures = [
            i for i, (s, a, t) in enumerate(trace.edges) if predictions.get((s, a)) != t
        ]
        first = failures[0] if failures else None
        record = {
            "trace_index": index,
            "accepted": not failures,
            "unseen_steps": unseen_count,
            "first_failure_step": first,
        }
        if first is not None:
            s, a, t = trace.edges[first]
            record["first_failure_edge"] = {
                "source": s,
                "action": a,
                "real_successor": t,
                "predicted_successor": predictions.get((s, a)),
                "observed_query": (s, a) in observed_pairs,
            }
            failure_steps[first] += 1
            failing_edges[s, a, t, predictions.get((s, a))] += 1
        records.append(record)
    with_unseen = [r for r in records if r["unseen_steps"]]
    only_observed = [r for r in records if not r["unseen_steps"]]
    return {
        "acceptance": rate(sum(r["accepted"] for r in records), len(records)),
        "with_unseen_pairs": rate(
            sum(r["accepted"] for r in with_unseen), len(with_unseen)
        ),
        "observed_only": rate(
            sum(r["accepted"] for r in only_observed), len(only_observed)
        ),
        "first_failure_step_is_zero_based": True,
        "first_failure_histogram": dict(sorted(failure_steps.items())),
        "first_failures_on_observed_pairs": sum(
            r.get("first_failure_edge", {}).get("observed_query", False)
            for r in records
        ),
        "failing_edges": [
            {
                "source": s,
                "action": a,
                "real_successor": t,
                "predicted_successor": p,
                "count": count,
            }
            for (s, a, t, p), count in sorted(
                failing_edges.items(), key=lambda x: repr(x[0])
            )
        ],
        "traces": records,
    }


class LTLBackendCache:
    """Exact cached nuXmv verdicts; compressed models/logs are saved for audit."""

    def __init__(self, *, executable=None, log_path: Path):
        self.executable = find_nusmv_executable(executable)
        if self.executable is None:
            raise FileNotFoundError(
                "LTL requires nuXmv; set NUXMV_BINARY or --executable."
            )
        self.log_path = Path(log_path)
        with self.log_path.open("xb"):
            pass
        self.cache = {}
        self.calls = self.hits = self.verdict_count = 0

    def verdicts(self, system, states):
        suite = default_ltl_suite()
        queries = tuple(LTLQuery(s, p.formula, p.name) for s in states for p in suite)
        model_text = export_nusmv_ltl_queries(system, queries)
        key = hashlib.sha256(model_text.encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            return key, self.cache[key]
        run = run_nusmv_model(model_text, self.executable)
        if len(run.verdicts) != len(queries):
            raise RuntimeError("Incomplete LTL verdict coverage.")
        values = tuple(run.verdicts)
        # Audit the two existing CTL/LTL identities at every state.
        checker = CTLModelChecker(system)
        for i, s in enumerate(states):
            for prop, formula in (
                ("F goal", AF(Atom("goal"))),
                ("G !danger", AG(Not(Atom("danger")))),
            ):
                j = next(j for j, p in enumerate(suite) if p.name == prop)
                if values[i * len(suite) + j] != checker.holds(s, formula):
                    raise AssertionError("nuXmv LTL/CTL identity mismatch.")
        with gzip.open(self.log_path, "at", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "model_sha256": key,
                        "model": model_text,
                        "output": run.output,
                        "verdicts": values,
                        "output_sha256": hashlib.sha256(
                            run.output.encode()
                        ).hexdigest(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
        self.cache[key] = values
        self.calls += 1
        self.verdict_count += len(values)
        return key, values

    def comparison(self, real, learned, problem):
        suite, states = default_ltl_suite(), problem.states
        real_key, real_values = self.verdicts(real, states)
        learned_key, learned_values = self.verdicts(learned, states)
        rows = [
            {"state": s, "property": p.name, "real": rv, "learned": lv}
            for (s, p), rv, lv in zip(
                ((s, p) for s in states for p in suite),
                real_values,
                learned_values,
                strict=True,
            )
        ]
        initial = [r for r in rows if r["state"] in problem.initial_states]
        return {
            "status": "executed_nuxmv_with_model_hash_cache",
            "real_model_sha256": real_key,
            "learned_model_sha256": learned_key,
            "initial_all_six_agreement": all(
                r["real"] == r["learned"] for r in initial
            ),
            "initial_mismatches": [r for r in initial if r["real"] != r["learned"]],
            "initial_verdicts": initial,
            "initial_formula_agreement": rate(
                sum(r["real"] == r["learned"] for r in initial), len(initial)
            ),
            "matched": sum(r["real"] == r["learned"] for r in rows),
            "comparisons": len(rows),
            "by_property": {
                p.name: rate(
                    sum(
                        r["real"] == r["learned"]
                        for r in rows
                        if r["property"] == p.name
                    ),
                    len(states),
                )
                for p in suite
            },
            "state_order": states,
            "property_order": [p.name for p in suite],
            "all_state_truths": {"real": real_values, "learned": learned_values},
        }


def evaluate_predictor(
    problem, observed_pairs, predictions, *, real, formal_system, heldout, backend
):
    """All hidden truth access is confined to this post-inference evaluation."""
    started = time.perf_counter()
    transitions = transition_scores(problem, observed_pairs, predictions, real)
    formal = evaluate_model_pair(real, formal_system)
    formal["diagnostics"]["ltl"] = backend.comparison(real, formal_system, problem)
    heldout_scores = trace_scores(problem, heldout, predictions, observed_pairs)
    # The saved pools allow exact replay; keep per-case failure histograms and
    # failing-edge counts without repeating 64 trajectory records in every row.
    heldout_scores.pop("traces")
    return {
        "transitions": transitions,
        "heldout": heldout_scores,
        "formal": formal,
        "evaluation_seconds": time.perf_counter() - started,
    }
