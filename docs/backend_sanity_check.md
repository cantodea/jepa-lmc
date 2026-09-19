# Strict backend sanity check on the saved Top-1 graphs

**All backend comparisons agree, and the initial-state all-six CTL agreement
remains 52/72.** This checks the existing graph suite and formulas only. It does
not train a model or introduce another research experiment.

## Inputs and backend

The audit reuses the saved `report.json` and `relations.jsonl` from the Top-1
relation experiment: three seeds, 24 maps each, and both the real and learned
graph for every case. That is 72 graph pairs, 144 graphs and 4,464 graph/state
instances. The saved states, coordinates, labels and transitions are read
directly; no weights are loaded and no predictions are decoded again.

The actual run used [official nuXmv 2.2.0](https://nuxmv.fbk.eu/download.html)
on Linux, with the downloaded archive checked against its official SHA-256.
The executable identifies itself as compiled on May 29, 2026. Its SHA-256 is
`758c252e60ad92072e64a8790e4dd8b5c6c1460a1cffdf2ecf889b768a125dfe`.
The checker source was frozen at commit `d4093219` before the completed run.
The run took 29.21 seconds, including output generation.

## Native CTL versus nuXmv CTL

Every existing CTL formula was evaluated at every state of both graphs, using
the project's unchanged `CTLModelChecker` and actual nuXmv execution.

| Existing CTL formula | Matching verdicts / comparisons | Agreement |
|---|---:|---:|
| `EF danger` | 4,464 / 4,464 | 100% |
| `EF goal` | 4,464 / 4,464 | 100% |
| `E[!danger U goal]` | 4,464 / 4,464 | 100% |
| `AG !danger` | 4,464 / 4,464 | 100% |
| `AF goal` | 4,464 / 4,464 | 100% |
| `EG safe` | 4,464 / 4,464 | 100% |
| **Total** | **26,784 / 26,784** | **100%** |

The real graphs and Top-1 graphs each contribute 13,392 comparisons, all matching.
Each seed contributes 8,928, also all matching. Both truth values occur for every
formula; the aggregate records the positive counts as well as agreement.

## CTL versus the corresponding LTL formulas in the same backend

| nuXmv CTL / nuXmv LTL pair | Matching verdicts / comparisons | Agreement |
|---|---:|---:|
| `AG !danger` / `G !danger` | 4,464 / 4,464 | 100% |
| `AF goal` / `F goal` | 4,464 / 4,464 | 100% |

There are 777 true verdicts for the safety pair and 144 for the eventual-goal
pair on either side. The latter is universal eventual reachability, corresponding
to `AF goal`. No other LTL formula was added to this full-suite audit.

## Initial-state real versus Top-1 CTL agreement

This score asks whether **all six formulas agree between M and Top-1 at their
corresponding designated initial states**. It is separate from backend agreement.

| Seed | Previously saved | Recomputed native CTL | Recomputed nuXmv CTL |
|---|---:|---:|---:|
| 20260804 | 21/24 | 21/24 | 21/24 |
| 20260805 | 13/24 | 13/24 | 13/24 |
| 20260806 | 18/24 | 18/24 | 18/24 |
| **Total** | **52/72** | **52/72** | **52/72** |

Zero maps change their all-six status, and zero individual initial-state formula
agreements change relative to the saved report. The original 20 cases with some
real/Top-1 disagreement remain; they are not backend mismatches.

## How the encoding is controlled

Each graph is encoded once, in one SMV file containing both the six CTL formulas
and the two paired LTL formulas for every state. One nuXmv process checks that
file. CTL and LTL therefore share the exact same transition table, proposition
definitions, state identifiers and initialization. Action labels are ignored
uniformly, as in the existing ordinary-CTL comparison.

The existing state-specific encoding makes every state initial and guards each
query by its requested state:

```text
SPEC    (state = s -> CTL_formula)
LTLSPEC (state = s -> LTL_formula)
```

Only paths starting at the requested state have a true antecedent. The originally
designated initial index is retained separately for the 52/72 score. An independent
readback of all 144 generated preambles checks the state domain, full initial
domain, every successor set and every label set against the saved graphs.

There are no `FAIRNESS`, `JUSTICE` or `COMPASSION` declarations. The backend runs
in exact finite-state batch mode with `-s` to disable startup files; no BMC bound
is used. Returned verdicts are matched to their actual state/formula expressions,
with whitespace and parentheses normalized for this fixed formula suite. Missing,
duplicate or unexpected expressions stop the audit rather than counting as
agreement. Each graph has a 120-second process timeout by default.

Input-file hashes, checker-source hashes and the backend executable hash are
checked again after the run. All three checkpoint files and all four learning
source files also have identical before/after hashes. Learning source still has
no diff from the main-branch baseline.

## Mismatches and validation

**Backend mismatches: 0. Paired-formula mismatches: 0. Saved-baseline changes: 0.**
The exported `mismatches.csv` contains its header and no data rows. When a mismatch
exists, the checker records its comparison type, seed, map, graph, state index,
coordinates, both formula names, backend names and both verdicts.

All 32 targeted tests passed with the installed nuXmv backend, including the
previously skipped backend integration tests. Tests detect altered graph/label
encodings, changed initialization, added fairness, missing/reordered backend
output, and a deliberately corrupted verdict even when the total all-six map
count stays unchanged. Ruff passes for the new checker and tests.

A separate replay of the saved records verified all 435 export hashes, reconciled
144 backend logs with 35,712 CSV comparisons, recomputed 26,784 native verdicts,
and checked all 432 initial-state formula rows. It again finds 52/72 and zero
mismatches. The first execution attempt stopped on a reader bug: nonempty saved
disagreements are objects rather than strings. That reader and its fixture were
corrected before the completed run; no CTL algorithm or graph change was needed.

## Reproduce and inspect

After updating this branch, use an installed nuXmv binary and your actual saved
Top-1 output directory. These PowerShell commands follow the earlier local setup:

```powershell
$env:NUXMV_BINARY = "D:\tools\nuXmv\bin\nuXmv.exe"
& .\.venv\Scripts\python.exe experiments\backend_sanity.py --run-dir outputs/behavioral_relations/top1_local --output-dir outputs/behavioral_relations/backend_sanity_local
```

Use a fresh output directory. The script also accepts `--executable PATH` and
`--timeout SECONDS`. It exits nonzero on a mismatch or an incomplete run. The
recorded numerical result above is from Linux nuXmv 2.2.0.

The [public aggregate](results/backend_sanity/summary.json) contains counts,
per-property and per-seed summaries, provenance and validation metadata. Complete
records are delivered separately as `jepa-lmc-backend-sanity-records.zip`:

- `run/report.json`, `checks.csv`, `initial_formulas.csv`, `mismatches.csv`;
- 144 SMV inputs, their query/state manifests and actual backend logs;
- the unchanged original source graph export and its report;
- validation logs, record-replay script/results and weight-preservation hashes.

Within these saved graphs and tested formulas, **no difference is attributable
to the choice of CTL backend, and both corresponding CTL/LTL pairs agree exactly**.
This conclusion is limited to the six CTL and two LTL formulas checked here.
