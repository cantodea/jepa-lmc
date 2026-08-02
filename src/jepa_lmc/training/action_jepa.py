from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import AdamW, Optimizer

from jepa_lmc.models.action_jepa import ActionJEPA, action_jepa_loss


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    prediction_loss: float
    variance_loss: float
    covariance_loss: float
    examples: int


@dataclass(frozen=True)
class EmbeddingStatistics:
    mean_dimension_standard_deviation: float
    mean_embedding_norm: float
    effective_rank: float


def make_action_jepa_optimizer(
    model: ActionJEPA,
    *,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
) -> Optimizer:
    if learning_rate <= 0.0:
        raise ValueError("Learning rate must be positive.")
    if weight_decay < 0.0:
        raise ValueError("Weight decay must be non-negative.")
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    return AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def train_action_jepa_epoch(
    model: ActionJEPA,
    batches: Iterable[Mapping[str, Tensor]],
    optimizer: Optimizer,
    *,
    device: torch.device | str = "cpu",
    ema_momentum: float = 0.99,
    variance_weight: float = 0.1,
    covariance_weight: float = 0.01,
) -> EpochMetrics:
    model.train()
    model.to(device)
    totals = {
        "loss": 0.0,
        "prediction": 0.0,
        "variance": 0.0,
        "covariance": 0.0,
    }
    example_count = 0

    for batch in batches:
        observation = batch["observation"].to(device)
        action = batch["action"].to(device)
        next_observation = batch["next_observation"].to(device)
        batch_size = observation.shape[0]
        if batch_size < 2:
            raise ValueError("Action-JEPA training batches must contain >= 2 examples.")

        optimizer.zero_grad(set_to_none=True)
        output = model(observation, action, next_observation)
        loss = action_jepa_loss(
            output,
            variance_weight=variance_weight,
            covariance_weight=covariance_weight,
        )
        loss.total.backward()
        optimizer.step()
        model.update_target_encoder(ema_momentum)

        totals["loss"] += loss.total.item() * batch_size
        totals["prediction"] += loss.prediction.item() * batch_size
        totals["variance"] += loss.variance.item() * batch_size
        totals["covariance"] += loss.covariance.item() * batch_size
        example_count += batch_size

    if example_count == 0:
        raise ValueError("Training requires at least one batch.")

    return EpochMetrics(
        loss=totals["loss"] / example_count,
        prediction_loss=totals["prediction"] / example_count,
        variance_loss=totals["variance"] / example_count,
        covariance_loss=totals["covariance"] / example_count,
        examples=example_count,
    )


@torch.no_grad()
def embedding_statistics(embedding: Tensor) -> EmbeddingStatistics:
    if embedding.ndim != 2 or embedding.shape[0] < 2:
        raise ValueError("Embedding statistics require a 2D batch of size >= 2.")

    centered = embedding - embedding.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    effective_rank = entropy.exp()
    return EmbeddingStatistics(
        mean_dimension_standard_deviation=float(
            embedding.std(dim=0, correction=0).mean().item()
        ),
        mean_embedding_norm=float(embedding.norm(dim=1).mean().item()),
        effective_rank=float(effective_rank.item()),
    )
