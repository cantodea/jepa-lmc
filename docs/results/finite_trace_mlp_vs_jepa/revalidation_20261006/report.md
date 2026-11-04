# Independent revalidation, 2026-10-06

The requested branch already contained a complete benchmark at
`52bddf724faa84a1c1f2ae657e8fd049e2782369`. This revalidation preserves that
implementation, frozen training protocol and all archived results. It does not
claim to have rerun all 720 training fits in this second environment.

- Full test suite rerun: **162 passed, zero failures or skips**, including actual
  nuXmv integration. Core finite-trace Ruff checks passed.
- All **1,080** saved model records replayed: observed/unseen transition scores,
  held-out acceptance/failures, native CTL and both structural semantics match.
- All 72 training/held-out pools checked for exact source identity, nested
  budgets, real execution, uniqueness, disjointness and sampling reproducibility.
- All **998** archived nuXmv models/outputs verified by hashes and parsed verdicts,
  and linked back to saved predictions. These are output replays, not 998 new
  backend executions. Fresh backend execution is covered by the test suite.
- All archived source hashes, 25 historical result hashes and 12 archived result
  artifact hashes match. Main and historical results remain unchanged.
- Four complete **300-epoch** retrains (first map/seed, budgets 1 and 16, both
  models) reproduce all **480** decoded successor predictions. Training tensor
  hashes and initial parameter hashes also match.

## Bitwise parameter reproducibility caveat

The four final parameter digests differ from the original run. The maximum
absolute difference among logged losses is **2.384185791015625e-7**. There are
zero decoded prediction differences in these four checks. Python, PyTorch and
nuXmv versions match the archived versions, but this does not imply identical
hardware or floating-point kernels. The tiny differences are consistent with
floating-point execution differences; the exact cause was not isolated.

The original strict verifier stopped at its first final-weight hash assertion,
after all 1,080 metric replays and aggregate comparisons passed. The unedited
`replay_log.txt` preserves that assertion failure. `retrain_checks.json` then
records all four reruns, including both expected/actual hashes, loss differences
and empty prediction-mismatch lists. No archived result or original
reproducibility claim was silently rewritten.

The supported new conclusion is **prediction reproducibility for these four
fixed checks**, not universal or cross-environment bitwise weight equality.

## Interpretation retained

JEPA beats this fixed MLP on unseen accuracy, but its correct unseen predictions
are dominated by real self-loops; unseen movement accuracy is only 1.14%-2.05%.
The MLP's low observed accuracy shows underfitting under the frozen configuration.
The comparison therefore does not establish that an adequately fitted ordinary
supervised predictor would be inferior, nor that JEPA has recovered unseen
movement dynamics. No hyperparameters or sample budgets were changed after
seeing these results.
