# Finite-trace pipeline validation before runtime disconnection

Local validation on 2026-10-06 used Python 3.12.14, torch 2.14.1+cpu,
and actual nuXmv 2.2.0. The complete suite passed with no skipped tests.

Command: python -m unittest discover -s tests -v

Cached final stdout:

    Ran 162 tests in 5.539s
    OK

The complete raw log was written in the execution workspace but cannot be
retrieved while that environment reports environment_offline. This page is
a validation summary, not a fabricated replacement for the raw log.

The 17 new tests cover known-only representation, exact observed supervision,
all original nested pools, hidden-T/R and exhaustive-loader prohibitions during
training and inference, observed/unseen partition and unreachable denominators,
held-out uniqueness/disjointness/reproducibility, total predicted models, first
failure statistics, real nuXmv and existing formal evaluator integration,
serialization, fixed-seed weights/predictions and relative output paths.

Two models were independently rerun at budgets 1 and 16 with the full 300-epoch
configuration. All four final parameter hashes and prediction sets matched
exactly. Core new-source Ruff checks and formatting checks also passed.

An output path check initially included its own LTL log as historical data.
That bug was fixed by resolving the destination path, a regression test was
added, the complete suite passed again, and the full benchmark restarted.
The restart reproduced the earlier predictions/parameter hashes on overlapping
records. No training data, epochs or model hyperparameters changed.

The last positively observed progress was 1,011/1,080 records. A subsequent
connection failure returned environment_offline. There is no verified final
aggregate from that local run. The GitHub workflow starts a separate complete
run and records its actual runtime versions; results from different runs are
not mixed. Previous maps, traces and historical result files are unchanged.
