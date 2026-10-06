"""Finish a complete frozen run: replay records, reproduce weights and save docs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from experiments.finite_trace_mlp_vs_jepa import (
    CONFIG, MODELS, ROOT, aggregate, known_catalogue, paired_comparison,
    read_training_pools, write_json,
)
from jepa_lmc.benchmarks.ltl_suite import default_ltl_suite
from jepa_lmc.benchmarks.radius_stress import radius_stress_cases
from jepa_lmc.evaluation.finite_trace import (
    disjoint_heldout_traces, prediction_system, trace_scores, transition_scores,
)
from jepa_lmc.evaluation.structural import evaluate_model_pair
from jepa_lmc.evaluation.trace_sanity import (
    Trace, is_execution, observed_edges, observed_self_loop_completion,
)
from jepa_lmc.learning.finite_trace import (
    FiniteTraceData, fit_finite_model, predict_successors,
)
from jepa_lmc.verification.gridworld import gridworld_to_transition_system
from jepa_lmc.verification.nuxmv import (
    LTLQuery, export_nusmv_ltl_queries, parse_nusmv_verdicts,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(value):
    return json.loads(json.dumps(value))


def finish(output, test_log, benchmark_log):
    output = Path(output).resolve()
    report = json.loads((output / "report.json").read_text())
    started = json.loads((output / "run_started.json").read_text())
    config = json.loads(CONFIG.read_text())
    assert report["status"] == "complete" and report["config"] == config
    assert report["model_case_records"] == 1080
    for group in ("source_sha256", "historical_result_sha256"):
        for name, expected in started[group].items():
            assert sha(ROOT / name) == expected, name
    assert not any(
        p.startswith("docs/results/finite_trace_mlp_vs_jepa/")
        for p in started["historical_result_sha256"]
    )
    for name, expected in report["exports"].items():
        assert sha(output / name) == expected, name
    rows = [json.loads(line) for line in (output / "cases.jsonl").read_text().splitlines()]
    indexed = {
        (r["case"], r["seed"], r["trajectory_budget"], r["model"]): r for r in rows
    }
    assert len(rows) == len(indexed) == 1080
    pool_rows = [
        json.loads(line) for line in (output / "trace_pools.jsonl").read_text().splitlines()
    ]
    pools = {(r["case"], r["seed"]): r for r in pool_rows}
    assert len(pools) == len(pool_rows) == 72
    original = read_training_pools()
    backend = {}
    with gzip.open(output / "ltl_backend.jsonl.gz", "rt") as handle:
        for line in handle:
            r = json.loads(line)
            assert hashlib.sha256(r["model"].encode()).hexdigest() == r["model_sha256"]
            assert hashlib.sha256(r["output"].encode()).hexdigest() == r["output_sha256"]
            assert list(parse_nusmv_verdicts(r["output"])) == r["verdicts"]
            assert r["model_sha256"] not in backend
            backend[r["model_sha256"]] = r
    assert len(backend) == report["runtime"]["nuXmv_calls"]
    assert sum(len(r["verdicts"]) for r in backend.values()) == report["runtime"]["nuXmv_ltl_verdicts"]
    checked = 0
    data_cache = {}
    for case in radius_stress_cases():
        env = case.make_env()
        problem, real = known_catalogue(env), gridworld_to_transition_system(env)
        queries = tuple(
            LTLQuery(s, p.formula, p.name)
            for s in problem.states for p in default_ltl_suite()
        )
        real_key = hashlib.sha256(
            export_nusmv_ltl_queries(real, queries).encode()
        ).hexdigest()
        for seed in config["seeds"]:
            pool = pools[case.name, seed]
            goods = tuple(
                Trace(tuple(map(tuple, t["states"])), tuple(t["actions"]))
                for t in pool["goods"]
            )
            heldout = tuple(
                Trace(tuple(map(tuple, t["states"])), tuple(t["actions"]))
                for t in pool["heldout"]
            )
            assert goods == original[seed, case.name]
            assert len(heldout) == len(set(heldout)) == 64
            assert not set(goods) & set(heldout)
            assert all(is_execution(env, t) for t in (*goods, *heldout))
            assert heldout == disjoint_heldout_traces(
                env, goods, seed=pool["heldout_rng_seed"]
            )
            previous = frozenset()
            for budget in config["trajectory_budgets"]:
                traces = goods[:budget]
                data = FiniteTraceData.from_traces(
                    problem, traces, height=6, width=6
                )
                data_cache[case.name, seed, budget] = data
                partial = observed_edges(traces)
                assert previous <= data.observed_pairs
                previous = data.observed_pairs
                for kind in MODELS:
                    r = indexed[case.name, seed, budget, kind]
                    assert r["trace_sha256"] == data.trace_sha256
                    assert r["training_sha256"] == data.training_sha256
                    assert r["observed_steps"] == 16 * budget
                    assert r["observed_pairs"] == len(data.observed_pairs)
                    assert r["observed_pairs"] + r["unseen_pairs"] == len(problem.states) * 4
                    predictions = {
                        (tuple(s), a): tuple(t) for s, a, t in r["predictions"]
                    }
                    assert len(predictions) == len(r["predictions"])
                    if kind == "observed_edge":
                        assert {(s, a, t) for (s, a), t in predictions.items()} == partial
                        graph = observed_self_loop_completion(problem, partial)
                        assert r["training"] is None
                    else:
                        graph = prediction_system(problem, predictions)
                        meta = r["training"]
                        assert meta["from_scratch"]
                        assert meta["epochs"] == meta["optimizer_steps"] == 300
                        assert meta["examples_processed"] == 300 * 16 * budget
                        assert meta["training_sha256"] == data.training_sha256
                        assert meta["trace_sha256"] == data.trace_sha256
                        assert meta["checkpoint_selection"] == "fixed final epoch; no evaluator input"
                    assert normalize(transition_scores(
                        problem, data.observed_pairs, predictions, real
                    )) == r["transitions"]
                    scores = trace_scores(
                        problem, heldout, predictions, data.observed_pairs
                    )
                    scores.pop("traces")
                    assert normalize(scores) == r["heldout"]
                    assert normalize(trace_scores(
                        problem, traces, predictions, data.observed_pairs
                    )["acceptance"]) == r["training_trace_acceptance"]
                    replay = evaluate_model_pair(real, graph)
                    assert normalize(replay["behavioral"]) == r["formal"]["behavioral"]
                    for name, value in replay["diagnostics"].items():
                        if name != "ltl":
                            assert normalize(value) == r["formal"]["diagnostics"][name]
                    key = hashlib.sha256(
                        export_nusmv_ltl_queries(graph, queries).encode()
                    ).hexdigest()
                    ltl = r["formal"]["diagnostics"]["ltl"]
                    assert ltl["real_model_sha256"] == real_key
                    assert ltl["learned_model_sha256"] == key
                    assert ltl["all_state_truths"]["real"] == backend[real_key]["verdicts"]
                    assert ltl["all_state_truths"]["learned"] == backend[key]["verdicts"]
                    if r["formal"]["behavioral"]["action_insensitive"]["initial_bisimulation"]:
                        assert ltl["initial_all_six_agreement"]
                    checked += 1
        print(f"Replayed {case.name}: {checked}/1080", flush=True)
    for budget in config["trajectory_budgets"]:
        selected = [r for r in rows if r["trajectory_budget"] == budget]
        for kind in MODELS:
            assert normalize(aggregate(
                r for r in selected if r["model"] == kind
            )) == report["by_budget"][str(budget)][kind]
        assert normalize(paired_comparison(selected)) == report["paired_unseen_comparison"][str(budget)]
    extra = json.loads((output / "unseen_edge_diagnostics.json").read_text())
    assert extra["cases_sha256"] == report["exports"]["cases.jsonl"]
    assert extra["source_script_sha256"] == sha(ROOT / "experiments/diagnose_finite_trace_predictions.py")
    for budget in config["trajectory_budgets"]:
        for kind in MODELS:
            assert extra["by_budget"][str(budget)][kind]["unseen"] == report["by_budget"][str(budget)][kind]["transitions"]["unseen"]
    checks = []
    first = radius_stress_cases()[0].name
    for budget in (1, 16):
        data = data_cache[first, 20260804, budget]
        for kind in ("mlp", "jepa"):
            r = indexed[first, 20260804, budget, kind]
            model, meta = fit_finite_model(
                data, kind=kind, seed=r["training"]["initialization_seed"],
                config=config,
            )
            predicted = predict_successors(model, data, kind=kind)
            saved = {(tuple(s), a): tuple(t) for s, a, t in r["predictions"]}
            assert predicted == saved
            assert meta["final_state_sha256"] == r["training"]["final_state_sha256"]
            checks.append({
                "case": first, "seed": 20260804, "trajectory_budget": budget,
                "model": kind, "final_state_sha256": meta["final_state_sha256"],
                "predictions_identical": True,
            })
    write_json(output / "reproducibility_check.json", {
        "scope": "Independent full-config reruns; no selection or tuning",
        "verified": checks,
    })
    log = Path(test_log).read_text()
    assert "Ran 162 tests" in log and log.rstrip().endswith("OK")
    assert "skipped=" not in log
    shutil.copyfile(test_log, output / "test_log.txt")
    shutil.copyfile(benchmark_log, output / "benchmark_log.txt")
    diagnostic_md = (output / "unseen_edge_diagnostics.md").read_text()
    with (output / "report.md").open("a") as handle:
        handle.write("\n" + diagnostic_md)
    write_json(output / "verification.json", {
        "status": "passed", "model_case_records_replayed": checked,
        "seed_map_pools_checked": 72,
        "structural_replays_using_existing_evaluate_model_pair": checked,
        "unique_nuxmv_outputs_and_verdicts_verified": len(backend),
        "historical_result_files_unchanged": len(started["historical_result_sha256"]),
        "full_tests": {"passed": 162, "failed": 0, "skipped": 0, "new_tests": 17},
        "full_config_training_reruns_with_identical_parameters_and_predictions": 4,
        "shared_observed_examples_only": True,
        "all_heldout_unique_disjoint_real_and_reproducible": True,
        "nested_budgets_verified": True,
        "saved_transition_trace_structural_ctl_and_ltl_results_verified": True,
        "unseen_edge_diagnostic_consistent_with_main_scores": True,
        "postprocessing_source_sha256": {
            "complete_finite_trace_run.py": sha(Path(__file__)),
            "diagnose_finite_trace_predictions.py": extra["source_script_sha256"],
        },
        "artifact_sha256": {
            p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()
        },
    })
    lines = [
        "# Finite-trace successor prediction: MLP vs JEPA", "",
        "Complete fixed-protocol benchmark: 24 existing October stress maps,",
        "three seeds (20260804/20260805/20260806), five nested budgets,",
        "16 transitions per trajectory, 64 disjoint held-out traces per seed/map.",
        "There are 720 from-scratch fits and 1,080 model-case records.", "",
        "The branch is based on main at 0e46680a352efc5ce5aea2e48fdb74dfa2089de4.",
        "It remains independent and is not merged. Previous results are unchanged.", "",
        "## Training and data boundary", "",
        "Both models see identical four-channel wall/danger/goal/agent images,",
        "action indices and observed trace steps including duplicates.",
        "The MLP flattens the 144 values, concatenates an 8-D action embedding,",
        "then uses Linear(152,64), GELU, Linear(64,|S|) and successor cross-entropy.",
        "It has 11,774-11,904 trainable parameters.",
        "JEPA reuses ActionJEPA, the original CNN and fixed 8-D position subspace,",
        "32-D latent and 64-unit action-conditioned residual predictor;",
        "EMA 0.99, prediction/variance/covariance losses and ranking weight 0.1.",
        "It has 90,776 trainable parameters. Both use AdamW, lr 0.0003,",
        "weight decay 0.0001, 300 full-batch epochs and fixed final-epoch selection.",
        "There are no pretrained weights, held-out selection or hyperparameter searches.", "",
        "The learner interface accepts known S,I,A,L, trace tensors, config and seed.",
        "It has no environment, reference R or held-out argument. Ranking positives",
        "are only observed successor indices. Known-state candidate images are permitted;",
        "they do not supply unseen transition labels. Tests prohibit real T/R calls",
        "and exhaustive dataset construction during both training and inference.", "",
        "All predictions are frozen before reference evaluation. Accuracy counts",
        "all known S x A, including unreachable states; separate reachable/unreachable",
        "unseen scores, pooled counts, ranges/std and per-seed/per-family results",
        "are in the detailed report. All held-out traces are unique, real, independent",
        "and disjoint from the entire training pool, across every budget.", "",
        "Raw observed-edge R_G abstains on unseen pairs. Its unseen score is 0%.",
        "The observed-edge relation is partial.",
        "Self-loop completion is introduced only to obtain a total transition system",
        "compatible with the existing formal evaluation pipeline.",
        "Its formal results hold under this explicit totalization policy,",
        "not for raw R_G. Completion is evaluation-only, not learned inference.", "",
        "Both action semantics reuse evaluate_model_pair and behavioral_relations.py.",
        "All six initial/identity simulation/bisimulation metrics are preserved.",
        "The six CTL and six actual nuXmv LTL formulas remain additional diagnostics.",
        "nuxmv.py and existing formal semantics are unchanged.", "",
        "## Main comparison (macro mean over 72 cases)", "",
        "| Budget | Model | Observed acc | Unseen acc | Overall acc | Held-out acceptance | Initial bisim (ignore / respect actions) | Identity bisim (ignore / respect actions) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for budget, methods in report["by_budget"].items():
        for kind, r in methods.items():
            values = [r["transitions"][k]["macro"]["mean"] for k in ("observed", "unseen", "overall")]
            values.append(r["heldout"]["acceptance"]["macro"]["mean"])
            p = r["formal"]["primary_metrics"]
            def count(semantics, name):
                v = p[semantics][name]
                return str(v["passed"]) + "/" + str(v["evaluated"])
            lines.append(
                "| " + budget + " | " + kind + " | "
                + " | ".join(f"{100*v:.2f}%" for v in values) + " | "
                + " / ".join(count(s, "initial_bisimulation") for s in ("action_insensitive", "action_sensitive"))
                + " | " + " / ".join(count(s, "identity_bisimulation") for s in ("action_insensitive", "action_sensitive"))
                + " |"
            )
    lines += [
        "", "## Interpretation", "",
        "JEPA and MLP are compared only under these architectures and this fixed",
        "optimization budget. An MLP with poor observed accuracy is underfitting;",
        "that does not establish that every supervised predictor is inadequate.",
        "High observed accuracy with low unseen accuracy indicates limited recovery",
        "beyond observed examples. Initial bisimulation can hold on a closed safe",
        "region despite incomplete state-identity transition recovery.", "",
        diagnostic_md, "",
        "The moving-edge breakdown is important: correct unseen self-loops alone",
        "do not establish recovery of unseen movement dynamics. The completion",
        "policy is an interpretive evaluator-only context, never learned inference.", "",
        "## Tests and records", "",
        "All 162 tests pass, with zero failures and skips, including real nuXmv.",
        "All 1,080 saved records were replayed against unchanged formal evaluators.",
        "Four independent complete 300-epoch retrains reproduce parameter hashes",
        "and all predictions exactly. Historical result hashes remain unchanged.", "",
        "[Detailed report](results/finite_trace_mlp_vs_jepa/report.md) contains",
        "separate structural tables for both action semantics, all simulation",
        "directions, CTL/LTL, per-seed/per-family tables and failure diagnostics.",
        "[Machine-readable report](results/finite_trace_mlp_vs_jepa/report.json),",
        "cases.jsonl, trace_pools.jsonl, experiment_config.json, run_started.json,",
        "test_log.txt, benchmark_log.txt, verification.json and reproducibility_check.json",
        "are saved in the independent result directory. ltl_backend.jsonl.gz contains",
        "all distinct symbolic models, actual backend output and verdicts.", "",
        "## Reproduce", "",
        "Run python -m experiments.finite_trace_mlp_vs_jepa --output-dir NEW_DIR",
        "from the repository root with PYTHONPATH=src and NUXMV_BINARY set.",
        "Then run python -m experiments.diagnose_finite_trace_predictions",
        "--result-dir NEW_DIR. Use a fresh directory. Runtime versions, config,",
        "sources and historical result hashes are recorded for reproduction.",
        "The branch-specific GitHub workflow runs the same fixed configuration",
        "and commits only complete, verified results to the experiment branch.", "",
    ]
    (ROOT / "docs/finite_trace_mlp_vs_jepa.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--benchmark-log", type=Path, required=True)
    args = parser.parse_args()
    finish(args.result_dir, args.test_log, args.benchmark_log)


if __name__ == "__main__":
    main()
