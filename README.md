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
