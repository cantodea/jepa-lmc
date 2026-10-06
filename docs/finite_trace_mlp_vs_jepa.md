# Finite-trace successor prediction: MLP vs JEPA

Complete fixed-protocol benchmark: 24 existing October stress maps,
three seeds (20260804/20260805/20260806), five nested budgets,
16 transitions per trajectory, 64 disjoint held-out traces per seed/map.
There are 720 from-scratch fits and 1,080 model-case records.

The branch is based on main at 0e46680a352efc5ce5aea2e48fdb74dfa2089de4.
It remains independent and is not merged. Previous results are unchanged.

## Training and data boundary

Both models see identical four-channel wall/danger/goal/agent images,
action indices and observed trace steps including duplicates.
The MLP flattens the 144 values, concatenates an 8-D action embedding,
then uses Linear(152,64), GELU, Linear(64,|S|) and successor cross-entropy.
It has 11,774-11,904 trainable parameters.
JEPA reuses ActionJEPA, the original CNN and fixed 8-D position subspace,
32-D latent and 64-unit action-conditioned residual predictor;
EMA 0.99, prediction/variance/covariance losses and ranking weight 0.1.
It has 90,776 trainable parameters. Both use AdamW, lr 0.0003,
weight decay 0.0001, 300 full-batch epochs and fixed final-epoch selection.
There are no pretrained weights, held-out selection or hyperparameter searches.

The learner interface accepts known S,I,A,L, trace tensors, config and seed.
It has no environment, reference R or held-out argument. Ranking positives
are only observed successor indices. Known-state candidate images are permitted;
they do not supply unseen transition labels. Tests prohibit real T/R calls
and exhaustive dataset construction during both training and inference.

All predictions are frozen before reference evaluation. Accuracy counts
all known S x A, including unreachable states; separate reachable/unreachable
unseen scores, pooled counts, ranges/std and per-seed/per-family results
are in the detailed report. All held-out traces are unique, real, independent
and disjoint from the entire training pool, across every budget.

Raw observed-edge R_G abstains on unseen pairs. Its unseen score is 0%.
The observed-edge relation is partial.
Self-loop completion is introduced only to obtain a total transition system
compatible with the existing formal evaluation pipeline.
Its formal results hold under this explicit totalization policy,
not for raw R_G. Completion is evaluation-only, not learned inference.

Both action semantics reuse evaluate_model_pair and behavioral_relations.py.
All six initial/identity simulation/bisimulation metrics are preserved.
The six CTL and six actual nuXmv LTL formulas remain additional diagnostics.
nuxmv.py and existing formal semantics are unchanged.

## Main comparison (macro mean over 72 cases)

| Budget | Model | Observed acc | Unseen acc | Overall acc | Held-out acceptance | Initial bisim (ignore / respect actions) | Identity bisim (ignore / respect actions) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | observed_edge | 100.00% | 0.00% | 9.37% | 0.26% | 24/72 / 24/72 | 0/72 / 0/72 |
| 1 | mlp | 62.43% | 0.88% | 6.60% | 0.17% | 24/72 / 24/72 | 0/72 / 0/72 |
| 1 | jepa | 100.00% | 26.79% | 33.64% | 0.76% | 24/72 / 24/72 | 0/72 / 0/72 |
| 2 | observed_edge | 100.00% | 0.00% | 15.35% | 2.65% | 24/72 / 24/72 | 0/72 / 0/72 |
| 2 | mlp | 42.98% | 0.45% | 6.85% | 0.26% | 24/72 / 24/72 | 0/72 / 0/72 |
| 2 | jepa | 100.00% | 27.83% | 38.87% | 8.01% | 24/72 / 24/72 | 0/72 / 0/72 |
| 4 | observed_edge | 100.00% | 0.00% | 23.34% | 11.74% | 24/72 / 24/72 | 0/72 / 0/72 |
| 4 | mlp | 30.96% | 0.15% | 7.16% | 0.26% | 24/72 / 24/72 | 0/72 / 0/72 |
| 4 | jepa | 100.00% | 28.56% | 45.19% | 21.66% | 24/72 / 24/72 | 0/72 / 0/72 |
| 8 | observed_edge | 100.00% | 0.00% | 32.04% | 36.74% | 24/72 / 24/72 | 0/72 / 0/72 |
| 8 | mlp | 23.97% | 0.05% | 7.47% | 0.67% | 24/72 / 24/72 | 0/72 / 0/72 |
| 8 | jepa | 99.62% | 28.88% | 51.50% | 43.62% | 24/72 / 24/72 | 0/72 / 0/72 |
| 16 | observed_edge | 100.00% | 0.00% | 41.06% | 61.18% | 24/72 / 24/72 | 0/72 / 0/72 |
| 16 | mlp | 18.54% | 0.00% | 7.35% | 0.91% | 24/72 / 24/72 | 0/72 / 0/72 |
| 16 | jepa | 98.14% | 30.07% | 57.92% | 64.67% | 24/72 / 24/72 | 0/72 / 0/72 |

## Interpretation

JEPA and MLP are compared only under these architectures and this fixed
optimization budget. An MLP with poor observed accuracy is underfitting;
that does not establish that every supervised predictor is inadequate.
High observed accuracy with low unseen accuracy indicates limited recovery
beyond observed examples. Initial bisimulation can hold on a closed safe
region despite incomplete state-identity transition recovery.

## Evaluator-only unseen edge-type diagnostic

Accuracies use all unseen pairs, including unreachable states. Moving edges have a real successor different from their source. The self-loop completion is an **evaluation-only context**, not learned inference or the raw partial observed-edge baseline. Its moving-edge accuracy is 0% by construction.

| Budget | Model / evaluation policy | Unseen accuracy | True self-loop unseen accuracy | Moving unseen accuracy |
|---:|---|---:|---:|---:|
| 1 | observed_edge | 0.00% | 0.00% | 0.00% |
| 1 | mlp | 0.88% | 1.22% | 0.75% |
| 1 | jepa | 26.79% | 94.64% | 1.14% |
| 1 | observed_self_loop_completion_evaluation_only | 27.47% | 100.00% | 0.00% |
| 2 | observed_edge | 0.00% | 0.00% | 0.00% |
| 2 | mlp | 0.45% | 0.60% | 0.38% |
| 2 | jepa | 27.83% | 96.16% | 1.40% |
| 2 | observed_self_loop_completion_evaluation_only | 27.91% | 100.00% | 0.00% |
| 4 | observed_edge | 0.00% | 0.00% | 0.00% |
| 4 | mlp | 0.15% | 0.22% | 0.12% |
| 4 | jepa | 28.56% | 96.71% | 2.03% |
| 4 | observed_self_loop_completion_evaluation_only | 28.05% | 100.00% | 0.00% |
| 8 | observed_edge | 0.00% | 0.00% | 0.00% |
| 8 | mlp | 0.05% | 0.06% | 0.05% |
| 8 | jepa | 28.88% | 96.86% | 1.92% |
| 8 | observed_self_loop_completion_evaluation_only | 28.42% | 100.00% | 0.00% |
| 16 | observed_edge | 0.00% | 0.00% | 0.00% |
| 16 | mlp | 0.00% | 0.00% | 0.00% |
| 16 | jepa | 30.07% | 97.26% | 2.05% |
| 16 | observed_self_loop_completion_evaluation_only | 29.43% | 100.00% | 0.00% |


The moving-edge breakdown is important: correct unseen self-loops alone
do not establish recovery of unseen movement dynamics. The completion
policy is an interpretive evaluator-only context, never learned inference.

## Tests and records

All 162 tests pass, with zero failures and skips, including real nuXmv.
All 1,080 saved records were replayed against unchanged formal evaluators.
Four independent complete 300-epoch retrains reproduce parameter hashes
and all predictions exactly. Historical result hashes remain unchanged.

[Detailed report](results/finite_trace_mlp_vs_jepa/report.md) contains
separate structural tables for both action semantics, all simulation
directions, CTL/LTL, per-seed/per-family tables and failure diagnostics.
[Machine-readable report](results/finite_trace_mlp_vs_jepa/report.json),
cases.jsonl, trace_pools.jsonl, experiment_config.json, run_started.json,
test_log.txt, benchmark_log.txt, verification.json and reproducibility_check.json
are saved in the independent result directory. ltl_backend.jsonl.gz contains
all distinct symbolic models, actual backend output and verdicts.

## Reproduce

Run python -m experiments.finite_trace_mlp_vs_jepa --output-dir NEW_DIR
from the repository root with PYTHONPATH=src and NUXMV_BINARY set.
Then run python -m experiments.diagnose_finite_trace_predictions
--result-dir NEW_DIR. Use a fresh directory. Runtime versions, config,
sources and historical result hashes are recorded for reproduction.
The branch-specific GitHub workflow runs the same fixed configuration
and commits only complete, verified results to the experiment branch.
