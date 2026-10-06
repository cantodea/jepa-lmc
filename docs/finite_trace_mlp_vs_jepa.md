# Finite-trace successor prediction: MLP vs JEPA

This independent branch starts from main at 0e46680a352efc5ce5aea2e48fdb74dfa2089de4.
It does not alter the good/bad sanity experiment or any historical result.

## Status

The finite-trace pipeline is implemented and its complete local test suite passed:
162 tests, zero failures and zero skips, including actual nuXmv.
The runtime became unavailable after the last confirmed benchmark progress of
1,011/1,080 model-case records. Its final local files cannot currently be read.
No incomplete aggregate or small-sample result is published as a final result.

A branch-specific GitHub workflow runs the same fixed configuration from scratch,
replays saved predictions and formal verdicts, verifies four complete training
reruns, and commits results only after the full benchmark and checks pass.
This page is replaced by the verified complete report when that job succeeds.
main is not updated or merged.

## Fixed protocol

- Exact 24 October stress maps and original sanity good trace pools.
- Seeds 20260804, 20260805, 20260806; nested budgets 1, 2, 4, 8, 16.
- 16 transitions per training trajectory; 64 independent held-out real traces
  per seed/map, unique and disjoint from the entire 16-trace training pool.
- Both models receive identical known-state images and observed trace steps.
- Training from scratch; 300 full-batch epochs; AdamW lr 0.0003 and decay 0.0001.
- Fixed final epoch, no oracle labels, pretrained weights or outcome selection.

The simple MLP is flatten(4x6x6), learned action embedding(8), Linear(152,64),
GELU, Linear(64,|S|), direct successor cross-entropy. JEPA reuses ActionJEPA,
its CNN and fixed position subspace, EMA 0.99, existing regularization/ranking
and unchanged Top-1 decoding over all known-state target embeddings.
Both fit functions receive only known metadata and observed samples.

Evaluation separates observed/unseen pairs over all S x A, including unreachable
states, and evaluates full held-out trace acceptance. Both action semantics
reuse evaluate_model_pair and behavioral_relations.py; CTL/LTL remain diagnostic.
The raw observed-edge relation is partial. Its separately named self-loop
completion is used only for formal evaluation, not learned inference.
An evaluator-only edge-type diagnostic separates unseen self-loops from movement.

## Entry points

- experiments/finite_trace_mlp_vs_jepa.py
- experiments/diagnose_finite_trace_predictions.py
- experiments/complete_finite_trace_run.py (saved-record replay and report)
- configs/finite_trace_mlp_vs_jepa.json
- .github/workflows/finite-trace-benchmark.yml (this branch only)

Run python -m experiments.finite_trace_mlp_vs_jepa --output-dir NEW_DIR
from the repository root, with PYTHONPATH=src and NUXMV_BINARY set.
The destination must be new. The full result directory is
docs/results/finite_trace_mlp_vs_jepa once a complete verified run is saved.

See [pipeline validation](finite_trace_mlp_vs_jepa_pipeline_validation.md)
for the local test evidence and infrastructure interruption.
