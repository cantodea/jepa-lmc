# JEPA-LMC

JEPA-LMC studies whether an action-conditioned JEPA model can reduce the concrete
checks needed for finite-state formal verification.

The latest [bounded ranking/depth study](docs/top1_ranking_study.md) finds a better
Top-1 model without changing the selected model's architecture. Across three seeds,
ranking loss improves validation from 92.87% to 98.68% and stress from 94.20% to
98.92%. Initial all-six CTL agreement rises from 52/72 to 68/72 and bisimulation
from 23/72 to 40/72. All-six LTL stays at 70/72; some individual temporal verdicts
regress despite higher Top-1 accuracy. Extra predictor depth gives no further gain.
Both original baseline and ranking checkpoints are retained. This tuning round is
complete; new abstraction experiments remain paused. The earlier negative
[epochs/LR/capacity study](docs/top1_quality_study.md) is preserved unchanged.

The branch also contains **property-directed successor refinement** in
deterministic GridWorld. Frozen JEPA predictions rank candidate paths; counted
exact simulator queries refine the candidate transition relation. Correctness
comes from preserving lower/upper relations around the true dynamics. A query
budget can return `unknown`. Those frozen-checkpoint experiments and their
formal results remain unchanged.

See the [CEGAR literature, assumptions and proof](docs/cegar_literature_and_design.md)
and [fixed experiment results](docs/cegar_results.md). On the three-checkpoint
screen, guidance saves 51–52% of successor queries versus direct BFS. The savings
come from finding concrete paths sooner; path-absence proofs show no query gain.

The branch also directly computes the greatest simulation and bisimulation
relations between the existing Top-1 and real graphs. See the
[behavioral-relation results](docs/top1_behavioral_relations.md): initial
bisimulation holds on 8/24, 7/24 and 8/24 maps, all in uniformly safe reachable
regions. The check distinguishes literal edge equality, initial correspondence
and same-coordinate correspondence throughout the full graph.

At the moment, the project supports:

- an explicit-state CTL model checker;
- witness and counterexample generation;
- nuXmv integration for CTL/LTL validation;
- action-conditioned JEPA training;
- reconstruction of learned transition systems;
- comparison between exact and learned verification results.
- oracle-backed local successor refinement for safety and finite reachability;
- uniform, JEPA-ranked, shuffled-ranking and direct BFS query comparisons.
- exact greatest simulation/bisimulation with independently checked certificates.

The earlier learned-transition reconstruction experiments use

$$
\hat{M} = (S, I, \hat{R}, L),
$$

where the state space, initial state and atomic propositions are taken from the environment, while the transition relation $\hat{R}$ is predicted by the learned dynamics.
These experiments, including the stopped uniform-radius route, remain available
as historical baselines. The refinement loop instead starts with all successors
possible and only removes candidates after a trusted concrete query.

## Setup

Python 3.11 or newer is recommended.

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e .
```

To use the nuXmv backend:

```powershell
$env:NUXMV_BINARY = "path\to\nuXmv.exe"
```

Run the tests with:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -v
```

## Experiments

The main experiment scripts are under `experiments/`.

The current experiment reuses the three frozen checkpoints from the
[checkpoint setup commands](docs/latent_radius_stress_results.md#reproduce-and-inspect):

```powershell
& .\.venv\Scripts\python.exe experiments\cegar_refinement.py --checkpoints outputs/latent_radius/oracle/model.pt outputs/latent_radius/stress_training/seed_20260805/model.pt outputs/latent_radius/stress_training/seed_20260806/model.pt --output-dir outputs/refinement/cegar
& .\.venv\Scripts\python.exe experiments\audit_cegar_refinement.py --run-dir outputs/refinement/cegar --checkpoints outputs/latent_radius/oracle/model.pt outputs/latent_radius/stress_training/seed_20260805/model.pt outputs/latent_radius/stress_training/seed_20260806/model.pt
```

The [fixed protocol](configs/cegar_protocol.json) compares query cost for
`AG !danger`, `EF goal`, and `E[!danger U goal]`. The second command replays every
budget and checks the exported results. Use a fresh directory for a new run.

To compare the existing Top-1 graphs directly with the real graphs:

```powershell
& .\.venv\Scripts\python.exe experiments\top1_behavioral_relations.py --checkpoints outputs/latent_radius/oracle/model.pt outputs/latent_radius/stress_training/seed_20260805/model.pt outputs/latent_radius/stress_training/seed_20260806/model.pt --output-dir outputs/behavioral_relations/top1
```

This computes both simulation directions and bisimulation, primarily with
ordinary CTL semantics and separately with action labels preserved. It exports
every surviving pair and a rejection certificate for every excluded pair.
The [protocol](configs/top1_relations_protocol.json) and
[results/API instructions](docs/top1_behavioral_relations.md) describe the scope.

To add nuXmv LTL verdicts for exactly those saved Top-1/real graphs, set
`NUXMV_BINARY` to your executable (or put nuXmv on PATH), then run:

```powershell
& .\.venv\Scripts\python.exe experiments\evaluate_top1_ltl.py --run-dir outputs/behavioral_relations/top1 --output-dir outputs/behavioral_relations/top1_ltl
```

This reuses the graph export without retraining or decoding again. It writes
initial-state and all-state LTL summaries, per-query CSV results, `.smv` inputs
and backend logs. See the [LTL instructions](docs/top1_behavioral_relations.md#optional-nuxmv-ltl-evaluation-of-the-same-graphs).
The previously recorded Top-1 results contain CTL checks only; LTL requires
this additional command and an installed nuXmv/NuSMV backend.

A completed [strict backend sanity check](docs/backend_sanity_check.md) reuses
all 72 saved graph pairs. Native CTL and nuXmv CTL agree on all 26,784 checks;
`AG !danger` / `G !danger` and `AF goal` / `F goal` each agree on all 4,464 checks
in the same nuXmv process and graph encoding. Initial-state all-six real/Top-1
CTL agreement remains 52/72. Run this specific audit with:

```powershell
& .\.venv\Scripts\python.exe experiments\backend_sanity.py --run-dir outputs/behavioral_relations/top1 --output-dir outputs/behavioral_relations/backend_sanity
```

Earlier exact-checking and learned-graph experiments:

```powershell
& .\.venv\Scripts\python.exe experiments\exact_ctl.py
& .\.venv\Scripts\python.exe experiments\evaluate_jepa.py
& .\.venv\Scripts\python.exe experiments\evaluate_multibackend.py
```

`exact_ctl.py` runs verification on the exact GridWorld model.

`evaluate_jepa.py` trains the JEPA model, reconstructs the learned transition relation, and compares CTL results with the exact model.

`evaluate_multibackend.py` evaluates the same learned transition system with both CTL and LTL backends.

`oracle_latent_radius.py` adds a frozen-model diagnostic: measure latent prediction
errors, form successor sets at the maximum/95th/99th-percentile radii, and compare
coverage, set size, singleton rate and CTL precision/soundness. It audits both
action-labelled and ordinary transition inclusion without changing JEPA.

```powershell
& .\.venv\Scripts\python.exe experiments\oracle_latent_radius.py
```

See the [protocol and Windows commands](docs/oracle_latent_radius.md) and
[recorded pilot results](docs/oracle_latent_radius_pilot.md). Test-derived radii
are oracle diagnostics, not guaranteed error bounds for unseen maps.

`stress_latent_radius.py` applies a fixed 24-map topology screen to three frozen
checkpoints. It excludes source-label base cases, compares exact/all-states
controls, and applies a prespecified stopping rule. See the
[protocol](docs/latent_radius_stress_protocol.md),
[conditional inclusion/CTL theorem](docs/latent_radius_conditional_theorem.md),
and [recorded results with reproducible commands](docs/latent_radius_stress_results.md).

## Current results

The latest ranking model is the validation-selected prediction baseline. The
[report and reproduction commands](docs/top1_ranking_study.md) give every seed,
matched training controls, exhaustive held-out scores and the unchanged formal
pipeline comparison. Validation-best checkpoints are selected before stress
evaluation. Training still uses only the original 40 maps; there is no stress/test
transition training. The following results refer to the older frozen checkpoints.

The CEGAR query-efficiency screen returned **support for further oracle-backed
study** on all three seeds. Across 864 full-budget tasks, every verdict matches
exact CTL truth and every inclusion audit passes. JEPA needs 34.06–34.93 queries
per task, versus uniform CEGAR's 71.14 and direct BFS's 71.33. At 25% of the full
transition-table budget, it resolves 55.56% of tasks versus 11.11% for both
controls. Shuffling the learned rankings removes most of the gain.

The gain is limited to finding real witnesses and safety counterexamples.
Unreachability/safety proofs require the same queries as the controls, and direct
BFS is faster on this cheap simulator. Known finite states, exact labels and a
trusted deterministic successor oracle are assumptions, not outputs of JEPA.
The [results report](docs/cegar_results.md) records these limits and next controls.

The original Top-1 behavioral check finds initial bisimulation only in the sealed-region
family; none of the dangerous-gate or safe-detour starts are bisimilar. Three
actual cases have mutual simulation without bisimulation, and 29 cases agree
on all six existing start-state CTL queries while failing bisimulation. All 432
greatest-relation certificates and 144 independent partition comparisons pass.
These are exact results for the constructed finite graphs, with complete real
dynamics available for comparison.

The earlier three-seed topology screen returned **STOP for the current Yang-style uniform
latent-radius route**. All seeds have 100% oracle successor coverage and zero
one-sided CTL violations; their non-immediate primary balanced scores are 77.50%,
65.42% and 100%, against the all-states control's 50%. Seed 20260805 fails the
fixed family-consistency requirement (3/8 sealed-region and 1/8 dangerous-gate
maps recover a proof; at least 4/8 in each was required). This records useful but
insufficiently stable proving power under that screen, and stops further
uniform-radius certification work for this construction.

In the current deterministic GridWorld pilot, the action-conditioned model reached approximately 94% Top-1 next-state accuracy and 95% CTL agreement on held-out maps.

These experiments are still preliminary. The current GridWorld transition dynamics are relatively simple, and the learned model does not provide a formal equivalence or bisimulation guarantee.

The earlier proposal to test query savings with more expensive concrete checks
and stronger search baselines remains deferred while model-quality comparisons
are assessed.
