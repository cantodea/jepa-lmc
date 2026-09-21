"""Optional same-map ranking objective for the bounded Top-1 study."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def same_map_ranking_loss(
    prediction: Tensor,
    target_bank: Tensor,
    true_indices: Tensor,
    query_maps: Tensor,
    candidate_maps: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Cross-entropy over negative squared distances to detached EMA targets.

    Every valid state of the query's *training* map is a candidate, including
    its source state. Other maps are masked out. The caller owns split isolation.
    The target bank receives no gradients, just like the original JEPA target.
    Squaring distances does not change their nearest-neighbour ordering.
    """
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Temperature must be finite and positive.")
    if (
        prediction.ndim != 2
        or target_bank.ndim != 2
        or prediction.shape[1] != target_bank.shape[1]
        or prediction.shape[0] == 0
        or target_bank.shape[0] == 0
    ):
        raise ValueError("Predictions and targets need nonempty matching latent axes.")
    n, k = prediction.shape[0], target_bank.shape[0]
    if (
        true_indices.shape != (n,)
        or query_maps.shape != (n,)
        or candidate_maps.shape != (k,)
        or true_indices.dtype != torch.long
        or torch.any(true_indices < 0)
        or torch.any(true_indices >= k)
    ):
        raise ValueError("Invalid candidate indices or map membership shapes.")
    allowed = query_maps[:, None] == candidate_maps[None, :]
    if not torch.all(allowed[torch.arange(n, device=prediction.device), true_indices]):
        raise ValueError("A true successor must belong to the query's map.")
    distances = (
        (prediction[:, None, :] - target_bank.detach()[None, :, :]).square().sum(-1)
    )
    logits = (-distances / temperature).masked_fill(~allowed, -torch.inf)
    return F.cross_entropy(logits, true_indices)
