"""Frozen neural search priorities, constructed without access to true dynamics."""

from __future__ import annotations

import random

import torch
from torch import Tensor

from jepa_lmc.learning.model import ActionJEPA


@torch.no_grad()
def latent_rank_costs(
    model: ActionJEPA, observations: Tensor, num_actions: int = 4
) -> tuple[tuple[int, ...], ...]:
    """Return positive rank costs for all candidates; never remove a candidate.

    Inputs are only the state catalogue's observations and frozen model. In
    particular there is no environment, transition function or successor label.
    Rows follow state-major/action-minor order; columns retain catalogue order.
    """
    if len(observations) < 1 or num_actions < 1:
        raise ValueError("Nonempty state/action catalogues are required.")
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    try:
        model.eval()
        observations = observations.to(device)
        targets = model.target_encoder(observations)
        context = model.context_encoder(observations)
        sources = torch.arange(len(observations), device=device).repeat_interleave(
            num_actions
        )
        actions = torch.arange(num_actions, device=device).repeat(len(observations))
        predictions = model.predictor(context[sources], actions)
        distances = torch.cdist(
            predictions.double(),
            targets.double(),
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        if not bool(torch.isfinite(distances).all()):
            raise ValueError("Neural distances must be finite.")
        order = torch.argsort(distances, dim=1, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1,
            order,
            torch.arange(1, len(observations) + 1, device=device).expand_as(order),
        )
        return tuple(tuple(row) for row in ranks.cpu().tolist())
    finally:
        for module, training in modes:
            module.training = training


def shuffled_rank_costs(
    costs: tuple[tuple[int, ...], ...], seed: int
) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    rows = []
    for row in costs:
        values = list(row)
        rng.shuffle(values)
        rows.append(tuple(values))
    return tuple(rows)
