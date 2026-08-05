"""Action-conditioned JEPA data, model, and training utilities."""

from jepa_lmc.learning.data import (
    AGENT_CHANNEL,
    DANGER_CHANNEL,
    GOAL_CHANNEL,
    OBSERVATION_CHANNELS,
    WALL_CHANNEL,
    GridWorldSource,
    GridWorldTransitionDataset,
    gridworld_observation,
)
from jepa_lmc.learning.model import (
    ActionConditionedPredictor,
    ActionJEPA,
    GridStateEncoder,
    JEPALoss,
    JEPAOutput,
    action_jepa_loss,
    covariance_loss,
    variance_floor_loss,
)
from jepa_lmc.learning.training import (
    EmbeddingStatistics,
    EpochMetrics,
    embedding_statistics,
    make_action_jepa_optimizer,
    train_action_jepa_epoch,
)

__all__ = [
    "AGENT_CHANNEL",
    "DANGER_CHANNEL",
    "GOAL_CHANNEL",
    "OBSERVATION_CHANNELS",
    "WALL_CHANNEL",
    "ActionConditionedPredictor",
    "ActionJEPA",
    "EmbeddingStatistics",
    "EpochMetrics",
    "GridStateEncoder",
    "GridWorldSource",
    "GridWorldTransitionDataset",
    "JEPALoss",
    "JEPAOutput",
    "action_jepa_loss",
    "covariance_loss",
    "embedding_statistics",
    "gridworld_observation",
    "make_action_jepa_optimizer",
    "train_action_jepa_epoch",
    "variance_floor_loss",
]
