"""Training loops and diagnostics for JEPA-LMC models."""

from jepa_lmc.training.action_jepa import (
    EpochMetrics,
    EmbeddingStatistics,
    embedding_statistics,
    make_action_jepa_optimizer,
    train_action_jepa_epoch,
)

__all__ = [
    "EmbeddingStatistics",
    "EpochMetrics",
    "embedding_statistics",
    "make_action_jepa_optimizer",
    "train_action_jepa_epoch",
]
