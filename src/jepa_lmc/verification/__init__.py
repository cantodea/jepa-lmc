"""Logic and explicit-state model-checking utilities for JEPA-LMC."""

from jepa_lmc.verification.ctl import (
    AF,
    AG,
    AX,
    EF,
    EG,
    EU,
    EX,
    And,
    Atom,
    CTLModelChecker,
    Not,
    Or,
)
from jepa_lmc.verification.gridworld import gridworld_to_transition_system
from jepa_lmc.verification.transition_system import (
    ExplicitTransitionSystem,
    TransitionEdge,
)
from jepa_lmc.verification.witness import (
    PathStep,
    PathWitness,
    find_ag_counterexample,
    find_ef_witness,
    find_eu_witness,
)

__all__ = [
    "AF",
    "AG",
    "AX",
    "EF",
    "EG",
    "EU",
    "EX",
    "And",
    "Atom",
    "CTLModelChecker",
    "ExplicitTransitionSystem",
    "Not",
    "Or",
    "PathStep",
    "PathWitness",
    "TransitionEdge",
    "find_ag_counterexample",
    "find_ef_witness",
    "find_eu_witness",
    "gridworld_to_transition_system",
]
