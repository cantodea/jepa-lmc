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

## Current results

In the current deterministic GridWorld pilot, the action-conditioned model reached approximately 94% Top-1 next-state accuracy and 95% CTL agreement on held-out maps.

These experiments are still preliminary. The current GridWorld transition dynamics are relatively simple, and the learned model does not provide a formal equivalence or bisimulation guarantee.

The next stage of the project is to test learned dynamics across more varied system instances and compare JEPA with simpler neural transition models.

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
