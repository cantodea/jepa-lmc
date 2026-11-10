# Verified bisimulation failure mechanism analysis

This directory archives the completed post-hoc analysis of the **Cross-Map Ranking Round 2 JEPA** predictions. It does not introduce a training run, a model change, additional maps, or an online repair method. The analysis uses the frozen source revision `539410dec1d5c4e4433bd9a57b10a309cd9a850d`.

## Main results

| Quantity | Verified result |
|---|---:|
| Seed–map cases / state–action transitions | 72 / 8,928 |
| Incorrect predicted successors | 96 |
| Correct predictions / Top-1 accuracy | 8,832 / 98.9247311827957% |
| Initial / identity bisimulation | 40/72 / 33/72 |
| Errors that can remain under the studied repair operations | 28, in seven initially bisimilar cases |
| Errors whose isolated retention breaks initial bisimulation | 68, in 32 failing cases |
| Failing cases requiring correction of every error | 32/32 |
| Initial six-CTL agreement without initial bisimulation | 28 cases |
| Exhaustively evaluated repair subsets | 794: 222 in failing cases and 572 in passing cases |

Both action semantics give the same initial decisions for every tested subset. Minimum repair sizes 1, 2, 3, 4, and 5 occur in 13, 9, 6, 1, and 3 failing cases, respectively. These are exact minima **only over replacing already incorrect predicted successors with their true successors**. They are not minima over arbitrary graph edits. True transitions are used only for post-hoc verification and counterfactual intervention.

See [report.md](report.md) for the complete English report, evidence, examples, and limitations.

## Files

- `cases_72_detailed.csv`: per-case counts, reachability, error categories, both bisimulation semantics, elimination depth, and property agreement.
- `errors_96_detailed.csv`: source/action/target identities, coordinates, labels, reachability, motion errors, true-graph successor equivalence, and repair outcomes.
- `repairs_32.csv`: exact minima and all minimum repair sets for failing cases.
- `repair_trials_all.csv`: every one of the 794 subsets and both initial-bisimulation verdicts.
- `passing_cases_7.csv`, `passing_repairs_7.csv`, `single_edge_repairs_13.csv`: focused results.
- `certificates/*.json.gz`: 72 losslessly compressed certificates, each containing both complete greatest bisimulation relations and elimination records. The 32 failing cases contain 64 rooted failure DAGs in total.
- `certificate_summary_64.csv`, `local_successor_certificate.json`: root evidence and the additional within-real-system certificate discussed in the report.
- `initial_properties_864.csv`, `ctl_agree_bisim_fail_28.csv`: temporal-property evidence. CTL was recomputed; LTL reuses the original hash-linked backend verdicts.
- `cases_72.json`, `errors_96.json`, base CSVs, `summary.json`, and other supplementary tables: unchanged machine-readable outputs needed by the verification scripts.
- `code/`: original analysis scripts/tests plus archive preparation and integrity tests. No original project code was edited.
- `provenance.json`: source revision, input hashes, scope, validation, and data availability. `original_provenance.json` preserves the completed run's original metadata.
- `source_files.json`: hashes of the original bytes of copied artifacts, including decompressed certificates. `SHA256SUMS.json`: hashes of published payload files.
- `validation/`: original verification records and the current archival integrity check.

CSV files retain their original UTF-8 BOM, line endings, and numeric precision. A local `.gitattributes` prevents checkout newline conversion. Certificates contain 25,652,683 bytes before lossless compression and 627,262 bytes in Git; the largest compressed certificate is 13,749 bytes. No Git LFS or files near GitHub's large-file limit are needed. The archive contains intentionally published finite-state identifiers, labels, and formal evidence; it contains no credentials, personal filesystem paths, checkpoints, raw private bundles, or original `relations.jsonl`.

## Public integrity checks

These checks require only Python 3.12 and the published archive. They verify file hashes, original artifact identity, the case/edge tables, exhaustive subset coverage, exact minima, and internal consistency of the stored DAGs. They do **not** independently check a challenging edge against an unavailable source graph.

From the repository root:

```bash
python -m unittest discover -s docs/results/bisimulation_failure_analysis/code -p test_archive.py -v
```

To run the four analysis unit tests and the eight existing upstream behavioral-relation tests:

```bash
python -m pip install -r docs/results/bisimulation_failure_analysis/requirements.txt
PYTHONPATH=src python -m unittest discover -s docs/results/bisimulation_failure_analysis/code -p test_analysis.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_behavioral_relations.py -v
```

In PowerShell, set `$env:PYTHONPATH='src'` before the latter commands. The unit fixtures are algorithm tests, not additional experimental maps. Do not run these assertion-based verification scripts with `python -O`.

## Original data and complete reproduction

**A public clone alone is insufficient for full graph-level reproduction.** The original `relations.jsonl` is intentionally not included in ordinary Git. Request the matching **Ranking Round 2 original failure-analysis bundle** from repository maintainer **cantodea**, identifying the study and the graph SHA-256 below. No public download URL or guaranteed access is claimed. A maintainer-approved transfer must provide these original, byte-preserved files under a separate directory `PRIVATE_ROOT`:

```text
outputs/top1_quality/round2/behavioral/best/relations.jsonl
outputs/top1_quality/round2/behavioral/best/report.json
outputs/top1_quality/round2/behavioral/best/ctl_queries.csv
outputs/top1_quality/round2/temporal/best_ltl/report.json
outputs/top1_quality/round2/temporal/best_ltl/queries.csv
outputs/top1_quality/round2/ranking/seed_20260804/evaluation/transitions.csv
outputs/top1_quality/round2/ranking/seed_20260805/evaluation/transitions.csv
outputs/top1_quality/round2/ranking/seed_20260806/evaluation/transitions.csv
```

Required graph SHA-256:

```text
8d51e45e76bf1f5a0c70732fdcfa90f72b08780c0558e6d8bea39793b2eb5d98
```

All eight expected hashes are in `provenance.json`. The first five are also pinned by the original structural-evaluation provenance. The three transition CSVs permit an independent row-level consistency check of the graph export. No checkpoint is required. The supplied recovery archive and extracted inputs were verified during the completed analysis; this archival task rechecks the local graph hash and rereads the saved certificates without running new experiments.

Use a separate, clean source worktree at the **original revision** because the preserved analysis script deliberately checks that exact HEAD. `ARCHIVE` denotes an absolute path to this published directory; `PINNED_SOURCE` and output directories must be outside this archive.

```bash
git worktree add --detach PINNED_SOURCE 539410dec1d5c4e4433bd9a57b10a309cd9a850d
python -m pip install -r "$ARCHIVE/requirements.txt"

# Recheck the published certificates against original graphs, without a new search:
python "$ARCHIVE/code/prepare_public.py" --out PUBLIC_COPY
python "$ARCHIVE/code/verify_outputs.py" --repo PINNED_SOURCE --data PRIVATE_ROOT --results PUBLIC_COPY

# Optional full reproduction of the already completed analysis, not run for this PR:
python "$ARCHIVE/code/analyze.py" --repo PINNED_SOURCE --data PRIVATE_ROOT --out REPRODUCED
python "$ARCHIVE/code/supplement.py" --results REPRODUCED --data PRIVATE_ROOT
python "$ARCHIVE/code/explain_examples.py" --repo PINNED_SOURCE --data PRIVATE_ROOT --results REPRODUCED
python "$ARCHIVE/code/verify_outputs.py" --repo PINNED_SOURCE --data PRIVATE_ROOT --results REPRODUCED
```

Use fresh output directories. `prepare_public.py` creates a detached copy and decompresses certificates without altering the archive. The original scripts are preserved byte-for-byte. The full analysis needs Python 3.12 and PyYAML 6.0.3, but no PyTorch or running nuXmv installation: it does not train or infer, and it verifies/reuses saved LTL results. LTL backend re-execution is not claimed.

## Interpretation

Coordinates are zero-based `[row,column]`; actions 0/1/2/3 mean up/down/left/right. Reachability fields concern the error's **source**. Goal states carry `{goal,safe}`, not an exclusive `goal` label. `targets_bisimilar_in_real` compares the two successors within the true graph. `single_repair_restores` means the verdict after correcting only that edge; in an already-passing case it means preservation, not restoration. `only_this_error_retained_passes` corrects all other errors first. An empty distance means unreachable, not zero.

The elimination DAG records every permitted response to its challenge with strictly decreasing removal ranks. It is a finite bisimulation-failure certificate, not an ordinary error path or a shortest LTL counterexample. Six CTL/LTL formulas agreeing does not establish equivalence for all formulas. The observed reachability/behavior-class separation is a finding on these 72 frozen seed–map cases, not a universal theorem; maps share families, layouts, rotations, and training seeds.
