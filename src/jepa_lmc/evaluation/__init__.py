"""Metrics for comparing learned and ground-truth verification models."""

from jepa_lmc.evaluation.jepa_dynamics import (
    RolloutReport,
    RolloutRetrieval,
    TransitionRetrieval,
    TransitionRetrievalReport,
    aggregate_rollout_reports,
    aggregate_transition_reports,
    evaluate_gridworld_transitions,
    evaluate_open_loop_rollouts,
    transition_system_from_retrievals,
)
from jepa_lmc.evaluation.verification_metrics import (
    BinaryConfusion,
    PropertyOutcome,
    VerificationReport,
    aggregate_reports,
    evaluate_ctl_suite,
    evaluate_ctl_suite_for_state_pairs,
)

__all__ = [
    "BinaryConfusion",
    "PropertyOutcome",
    "RolloutReport",
    "RolloutRetrieval",
    "TransitionRetrieval",
    "TransitionRetrievalReport",
    "VerificationReport",
    "aggregate_reports",
    "aggregate_rollout_reports",
    "aggregate_transition_reports",
    "evaluate_ctl_suite",
    "evaluate_ctl_suite_for_state_pairs",
    "evaluate_gridworld_transitions",
    "evaluate_open_loop_rollouts",
    "transition_system_from_retrievals",
]
