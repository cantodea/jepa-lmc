# Good/bad execution traces: reproducible pilot

Base: `cantodea/jepa-lmc` main commit `539410dec1d5c4e4433bd9a57b10a309cd9a850d`.
Research branch: `research/passive-negative-traces-20261117`.
Historical files and thesis LaTeX are unchanged.

Start with `research_report_zh.md`, `thesis_summary_en.md`, `theory.md`,
`audit.md`, and `independent_validation.md`. All numbers in the summaries
come from the committed `run/` records; this is a small fixed pilot, not the
historical 360-case training benchmark.

## Reproduction

Use Python >=3.11, the project dependencies, and nuXmv 2.1.0. The recorded
environment was Windows, Python 3.11.9 and PyTorch 2.12.1+cpu. Exact tensor
hash reproduction is environment-dependent; predictions and formal verdicts
are separately recorded. No pretrained weights are used for fitting.

```powershell
git clone --branch research/passive-negative-traces-20261117 https://github.com/cantodea/jepa-lmc.git
cd jepa-lmc
python -m pip install -e .
$env:PYTHONPATH = "$PWD/src"
$env:NUXMV_BINARY = 'C:/path/to/nuXmv.exe'
python -m unittest tests.test_passive_negative tests.test_passive_negative_independent tests.test_good_bad_trace_sanity tests.test_finite_trace_mlp_vs_jepa tests.test_behavioral_relations tests.test_structural_metrics
python experiments/passive_negative_traces.py --phase prepare --output outputs/passive_negative_reproduction
python experiments/passive_negative_traces.py --phase train --output outputs/passive_negative_reproduction --nuxmv $env:NUXMV_BINARY
python experiments/summarize_passive_negative.py outputs/passive_negative_reproduction
node reports/passive_negative_traces_20261117/theory_exact_check.js
```

Use a new output directory; the preparation, training, and summary commands
refuse to overwrite existing results. On POSIX use `export PYTHONPATH=src`
and the corresponding nuXmv executable path. If the platform has no nuXmv,
install it before running the full experiment; LTL results must not be
replaced by bounded approximations or labelled as executed.

`run/datasets.json` retains all training candidate queries and labels,
natural goods, accepted candidates, rejected candidates, diagnostic legacy
negatives, disjoint held-out goods/bads, and acquisition costs. The learner
receives only `FiniteTraceData` and negative `Trace` objects. No corrupted
edge index or true successor is passed. `run/information.json` contains
exact model counts, per-query successor supports, redundancy diagnostics,
and symbolic completions before neural training. Resource limits explicitly
return `resource_limit`; this run reached exact answers for all cases.

`run/results.jsonl` contains every completed arm, all predicted transitions,
training losses and timing, checkpoint hashes, and complete formal scores.
`run/ltl_backend.jsonl.gz` retains nuXmv input/output and hashes. Checkpoints
are distributed in `run/checkpoints.zip`; extract into `run/checkpoints/`
to repeat checkpoint replay. `run/pretraining_manifest.json` records the
pretraining data/protocol hashes. CSV files provide per-case, per-family,
per-seed, per-budget and paired contrasts. SVG plots are generated directly
from saved results. See the independent verification record for executed
replays, additional tests, and any limitations.

The first `focused_tests.log` records an unsuccessful environment setup
(missing username for PyTorch and missing historical fixtures in the local
source mirror). It is retained. After restoring the original fixtures and
fixing process environment variables, `regression_tests.log` records the
successful checks. This was an infrastructure correction, not experiment
selection or a change to model hyperparameters.

The GitHub branch uses the actual main parent and its complete original
tree. `source_provenance.json` explains the temporary local partial source
mirror necessitated by unavailable direct Git network access, and verifies
100 original code/configuration files against their Git blob IDs. Local
synthetic mirror commits are not used as the published branch ancestry.
