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

The three-seed topology screen returned **STOP for the current Yang-style uniform
latent-radius route**. All seeds have 100% oracle successor coverage and zero
one-sided CTL violations; their non-immediate primary balanced scores are 77.50%,
65.42% and 100%, against the all-states control's 50%. Seed 20260805 fails the
fixed family-consistency requirement (3/8 sealed-region and 1/8 dangerous-gate
maps recover a proof; at least 4/8 in each was required). This records useful but
insufficiently stable proving power under that screen, and stops further
uniform-radius certification work for this construction.

In the current deterministic GridWorld pilot, the action-conditioned model reached approximately 94% Top-1 next-state accuracy and 95% CTL agreement on held-out maps.

These experiments are still preliminary. The current GridWorld transition dynamics are relatively simple, and the learned model does not provide a formal equivalence or bisimulation guarantee.

The next stage of the project is to test learned dynamics across more varied system instances and compare JEPA with simpler neural transition models.
