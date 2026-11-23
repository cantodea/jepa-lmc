# JEPA-LMC

JEPA-LMC is an experimental project for learning finite-state transition dynamics with an action-conditioned JEPA model and using the learned model for formal verification.

The current implementation uses deterministic GridWorld environments. JEPA is trained from state-action-next-state observations and predicts the next-state representation conditioned on an action. The predicted transitions are then reconstructed as an explicit transition system.

At the moment, the project supports:

- an explicit-state CTL model checker;
- witness and counterexample generation;
- nuXmv integration for CTL/LTL validation;
- action-conditioned JEPA training;
- reconstruction of learned transition systems;
- comparison between exact and learned verification results.

The current learned model has the form

$$
\hat{M} = (S, I, \hat{R}, L),
$$

where the state space, initial state and atomic propositions are taken from the environment, while the transition relation $\hat{R}$ is predicted by the learned dynamics.

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

## Finite-trace MLP baseline (updated 2026-11-04)

The default finite-trace comparison now uses the original MLP architecture with
**900 epochs and learning rate 0.001**. JEPA remains at its original 300-epoch
configuration. All 360 MLP checkpoints and their results are available in the
[MLP-900 model card](docs/results/mlp_900_fairness/README.md).
The MLP fits all observed transitions, but unseen moving accuracy remains 0.277%.
The original 300-epoch configuration and historical results are retained.

## Experiments

The main experiment scripts are under `experiments/`.

```powershell
& .\.venv\Scripts\python.exe experiments\exact_ctl.py
& .\.venv\Scripts\python.exe experiments\evaluate_jepa.py
& .\.venv\Scripts\python.exe experiments\evaluate_multibackend.py
```

`exact_ctl.py` runs verification on the exact GridWorld model.

`evaluate_jepa.py` trains the JEPA model, reconstructs the learned transition relation, and compares CTL results with the exact model.

`evaluate_multibackend.py` evaluates the same learned transition system with both CTL and LTL backends.

## Early pilot results (historical)

In an early deterministic GridWorld pilot, the action-conditioned model reached approximately 94% Top-1 next-state accuracy and 95% CTL agreement on held-out maps.

These experiments are still preliminary. The current GridWorld transition dynamics are relatively simple, and the learned model does not provide a formal equivalence or bisimulation guarantee.

Later work investigates behavioral equivalence, learning from finite traces, and what verification results can be justified from limited observations. See the research branches below.

## Updates (September 2026)

Added [simulation/bisimulation checks](docs/top1_behavioral_relations.md) and a [nuXmv CTL/LTL backend audit](docs/backend_sanity_check.md).
[Ranking-loss training](docs/top1_ranking_study.md) improves stress Top-1 accuracy to 98.92% across three seeds.
The [oracle local abstraction study](docs/oracle_local_abstraction.md) achieves 100% successor coverage with mean candidate size 1.0112; these bounds use oracle truth and do not yet guarantee coverage under unknown dynamics.

## Primary model evaluation (October 2026)

Simulation and bisimulation are now the primary formal comparison of real and
learned models; Top-1 accuracy and the fixed CTL/LTL suites remain additional
diagnostics. The [unified structural report](docs/structural_evaluation.md) defines
initial and identity metrics for both Kripke and action-sensitive semantics,
with complete per-case results and mismatch lists.

Re-evaluating the same frozen 72 graph pairs gives ranking Top-1 **40/72 initial
and 33/72 identity bisimulation**, and oracle-local **49/72 and 46/72** under
ordinary Kripke semantics. All-six initial CTL agreement remains **68/72** and
**69/72**, respectively. No models, checkpoints or data were changed or retrained.
Run `python -m experiments.evaluate_structural --help` for saved-graph evaluation.

The independent [good/bad trace sanity experiment](docs/good_bad_trace_sanity.md)
tests the known-state, hidden-transition formulation without JEPA training.
Across 24 stress maps and three seeds, observed-edge memorisation achieves 100%
good acceptance and bad rejection while recovering only 9.37%-41.06% of real
transitions at budgets of 1-16 trajectories. Sample consistency is evaluated on
the partial relation; simulation/bisimulation and CTL are evaluated separately
under explicit self-loop completion.

## Research branches (November 2026)

The `main` branch contains the core implementation and established experiments. The following research and historical branches are kept separate to preserve their original results.

- **[Behavioral fidelity audit](https://github.com/cantodea/jepa-lmc/tree/research/behavioral-fidelity-audit)** — Examines prediction errors, behavioral equivalence, and which CTL/LTL conclusions are justified by finite observations. A property-directed verification procedure is proposed as future work, not yet implemented.
- **[Positive and negative traces](https://github.com/cantodea/jepa-lmc/tree/research/passive-negative-traces-20261117)** — A pilot study of whether negative traces add information about unobserved transitions and improve reconstruction.
- **[Bisimulation failure analysis (archive)](https://github.com/cantodea/jepa-lmc/tree/docs/archive-bisimulation-failure-analysis-20261109)** — Historical analysis of transition errors and counterfactual repairs.
- **[Finite-trace MLP vs JEPA (historical)](https://github.com/cantodea/jepa-lmc/tree/experiment/finite-trace-mlp-vs-jepa)** — Earlier learning comparison; the updated MLP-900 baseline is documented on `main`.
- **[Good/bad trace sanity check (historical)](https://github.com/cantodea/jepa-lmc/tree/experiment/good-bad-trace-sanity-20261005)** — Initial trace-consistency study, later merged into `main`.
