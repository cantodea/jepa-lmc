# Behavioral fidelity audit — reproducibility and data dictionary

Start with [the Chinese report](research_report_zh.md), [the English thesis text](thesis_findings_en.md), and [the information-flow audit](information_flow_audit.md). The original five branch heads are recorded in the audit. The branch started at `7fdabb95501f9cc78cad119d7dea1f5517c1bc8d`. The primary protocol was committed at `a1f0ec93f3a76013c6dd69b659ac00d2e09b89bc`; the candidate-only witness supplement at `a60997addf691d8d30275b29d8c82fa958e7134a`.

## Delivered tables

|Path relative to this directory|Grain and interpretation|
|---|---|
|`protocol.md`, `witness_protocol.md`|Pre-outcome primary protocol and explicitly later exploratory protocol|
|`transition_analysis.csv`|One raw model, map, seed, budget, source and action per row. E2 shares each E1 row. Includes true, raw, E2 targets, movement/correctness, score and rank when defined|
|`state_catalogue.csv`|Map-local integer state IDs in sorted catalogue order, coordinates, initial and AP labels. Actions 0 up, 1 down, 2 left, 3 right are dataset display names|
|`behavioral_analysis.csv`|One model/variant/setting. Both simulations, initial/identity/strict identity bisimulation, CTL/LTL agreement, safety and directional disagreement. R certificate paths are relative to the separate archive directory below|
|`baseline_comparison.csv`|Case counts and conditional denominators, action shift, self-loops, E2 changes, E3 margin diagnostics. E3 annotates E1 and has identical hard outputs|
|`initial_verdicts.csv`|5,184 initial formula verdict pairs, one row per logic/formula/model/variant. Two of the LTL formulas duplicate CTL semantics|
|`action_results.csv`, `grouped_results.csv`, `quartile_summary.csv`, `uncertainty_quartiles.csv`|Action/seen strata, totals by protocol/arm/variant, budget or map, pooled within-case quartile counts and per-case cut points|
|`information.csv`|36 exact counts, log-ratios and singleton-support counts|
|`witness_counts.csv`, `witnesses.csv`|Feasible candidate search counts, one header-only witness table because no opposite-verdict candidate pair was found|
|`offline_ambiguities.csv`|Additional post-hoc feasible-candidate versus feasible real-system witnesses. Never a deployment input|
|`diagnostics.json`, `execution.json`, `replay_checks.csv`|Derived report quantities; input/checkpoint/backend hashes, runtime; separate per-setting score parity|
|`nuxmv_records.jsonl.gz`|153 distinct exact SMV inputs, their SHA-256 and raw checker output, cached across identical initial queries|
|`logs/`, `pinned_sources.json`, `SHA256SUMS.json`|Tests, completed runs, failed first supplement attempt, frozen Git blob IDs and byte-level release manifest|
|`audit_core.py`, `run_audit.py`, `witnesses.py`, `summarize.py`, `diagnostics.py`, `verify_package.py`, `test_audit.py`|Reconstruction/score and evaluator code, summaries, reproducibility checks and tests|

**Keys and score conventions.** F arm A uses common positives; B adds legacy negatives; C adds natural negatives. R uses the selected ranking weights. The F baselines appear once per setting, with carrier columns `kind=symbolic,arm=A`; R uses `kind=jepa,arm=ranking`. Filter on `variant` before aggregating. E2 on an empty support map is the identity. `changed_from_E1` compares to the carrier's raw table; for a loop variant it is not a training-error count. `seen` means this map/query occurs in training positives. F supports are exact marginal supports under the row's arm. False-safe needs the learned verdict to be true and the real verdict false on a formula marked safety by the original suite. `unsafe_opportunities` is the number of real-false safety verdicts. Every per-case denominator here includes three safety queries and twelve queries of all types. No abstention in E1/E2.

`raw_score1` is the maximum negative ordinary Euclidean distance for JEPA, or the maximum logit for MLP. `gap` subtracts runner-up score from winner score. `true_rank` uses the offline true target; archived R signed squared margin also uses truth and has different units. A blank denotes unavailable/not applicable, including symbolic scores or AUROC with only one outcome class. Quartiles are rank bins on unseen rows per case, ordered by gap then source/action. Neither the score scale nor the quartile cut points are shared across models. Stabilized AP labels and total infinite-path semantics are as in the parent repository, without fairness restrictions.

## Reproduction

A public clone contains all new tables, scripts and checks. Python 3.11+ suffices for the standard-library package check. **Full independent score re-inference requires external historical inputs.** The public finite pilot includes `reports/passive_negative_traces_20261117/run/raw_records.zip`; its Git blob is `2f7d5e80216d64642bdce5c0a62e90192942d5ff`. Despite the older README mentioning a checkpoint tarball, the 72 files are absent from that pinned public directory. During this audit the recovered local `checkpoints.tar.xz` (SHA-256 `2b3d8195558ce4bae977e05c20e97b57626203f57f74b2907a04056f0e5888ae`) contained the 72 matching files. For a fresh re-inference obtain this archive from the maintainer and extract it into a separate read-only-input location. The R input is the maintainer's **original** `jepa-lmc-ranking-original-failure-analysis-bundle.zip` containing `outputs/top1_quality/round2/`, including the three validation-best `model.pt` files. A later replication's model files have different hashes.

Obtain the public certificate/provenance directory via a separate checkout of commit `670e0a003ce7bb1d6b7b270bad43b09fd47d4fec` at `docs/results/bisimulation_failure_analysis`. Do not replace the audit branch with that checkout. Historical default MLP-900 results, top1-local, oracle-local and finite-pilot checkpoints cannot be interchanged. All new computation must target a fresh output directory.

From the repository root with project dependencies, pandas, CPU PyTorch and nuXmv installed (POSIX example; adapt environment variable syntax on Windows):

```sh
D=reports/behavioral_fidelity_audit
F=reports/passive_negative_traces_20261117/run
export PYTHONPATH="$PWD/src:$PWD"
python "$D/verify_package.py" --repo .
python -m unittest discover -s "$D" -p test_audit.py -v
python "$D/run_audit.py" --out audit-replay --backend /path/to/nuXmv --finite "$F" --weights /path/to/checkpoints --ranking /path/to/jepa-lmc-ranking-original-failure-analysis-bundle.zip --archive /path/to/docs/results/bisimulation_failure_analysis
python "$D/witnesses.py" "$F" audit-replay
python "$D/summarize.py" audit-replay
python "$D/diagnostics.py" audit-replay
```

For public-input-only F reconstruction and formal replay, supply `--finite "$F"`, `--out`, `--backend` only. This skips checkpoint scores, so score-dependent combined-cohort summaries do not apply. To reproduce report aggregates without weights or a model checker, copy the published tables to a scratch directory and run `summarize.py` and `diagnostics.py` on that copy. The branch keeps original experiment files intact.

**Execution record.** Windows, Python 3.12.14, CPU PyTorch 2.14.1, one thread, deterministic algorithms, nuXmv 2.1.0. Saved finite weights were originally produced under PyTorch 2.12.1+cpu. Cross-version re-inference matched every archived Top-1; bitwise equality of old score tensors is untested. Two separate complete primary runs took 46.59 and 40.20 seconds wall time inside the runner, excluding startup, supplementary counting, package checks and remote publication. The seven primary CSVs and compressed backend records were byte-identical on repeat. No new training or random sampling. Historical seeds and budgets are explicit in every applicable result. The 33 executed unit/archive tests comprised 5 new tests, 8 archive checks, 8 behavioral-relation tests and 12 negative-trace tests.

**Supplementary provenance.** The first witness verification stopped when comparing freshly computed Python tuples with JSON lists. The retry normalized representation to JSON before comparisons; all 36 exact summaries and supports then matched. The first failed log retains the exception with workspace paths abbreviated. The later real-system existence check is explicitly post hoc, separate from the predeclared candidate scan. It adds no training signal. No all-version-space property solver, alternative trained decoder, new domain or deployable abstention threshold was tested.
