from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GridStateEncoder(nn.Module):
    """Encode map content while retaining an explicit agent-position subspace."""

    def __init__(
        self,
        *,
        height: int,
        width: int,
        input_channels: int = 4,
        latent_dim: int = 32,
        hidden_channels: int = 32,
        position_dim: int | None = None,
        position_scale: float = 4.0,
    ) -> None:
        super().__init__()
        if height <= 0 or width <= 0:
            raise ValueError("Grid dimensions must be positive.")
        if input_channels < 4 or latent_dim <= 1 or hidden_channels <= 0:
            raise ValueError("Encoder dimensions must be positive.")
        resolved_position_dim = (
            min(8, latent_dim - 1) if position_dim is None else position_dim
        )
        if not 1 <= resolved_position_dim < latent_dim or resolved_position_dim > 8:
            raise ValueError(
                "Position dimension must be in [1, min(8, latent_dim - 1)]."
            )
        if position_scale <= 0.0:
            raise ValueError("Position scale must be positive.")

        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.position_dim = resolved_position_dim
        self.position_scale = position_scale
        learned_dim = latent_dim - resolved_position_dim
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(hidden_channels * height * width, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Linear(hidden_channels * 2, learned_dim),
        )
        row_coordinates = torch.linspace(0.0, 1.0, steps=height).view(1, height, 1)
        column_coordinates = torch.linspace(0.0, 1.0, steps=width).view(1, 1, width)
        self.row_coordinates: Tensor
        self.column_coordinates: Tensor
        self.register_buffer("row_coordinates", row_coordinates)
        self.register_buffer("column_coordinates", column_coordinates)

    def forward(self, observation: Tensor) -> Tensor:
        expected_shape = (
            self.input_channels,
            self.height,
            self.width,
        )
        if observation.ndim != 4 or tuple(observation.shape[1:]) != expected_shape:
            raise ValueError(
                "Expected observations with shape "
                f"[batch, {expected_shape[0]}, {expected_shape[1]}, "
                f"{expected_shape[2]}], got {tuple(observation.shape)}."
            )
        learned_features = self.features(observation)
        agent = observation[:, 3]
        row = (agent * self.row_coordinates).sum(dim=(1, 2))
        column = (agent * self.column_coordinates).sum(dim=(1, 2))
        position_features = (
            self.position_scale
            * torch.stack(
                (
                    row,
                    column,
                    torch.sin(torch.pi * row),
                    torch.cos(torch.pi * row),
                    torch.sin(torch.pi * column),
                    torch.cos(torch.pi * column),
                    torch.sin(2.0 * torch.pi * row),
                    torch.sin(2.0 * torch.pi * column),
                ),
                dim=1,
            )[:, : self.position_dim]
        )
        return torch.cat((learned_features, position_features), dim=1)


class ActionConditionedPredictor(nn.Module):
    """Predict the next latent state conditioned on a discrete action."""

    def __init__(
        self,
        *,
        latent_dim: int,
        num_actions: int = 4,
        action_dim: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if min(latent_dim, num_actions, action_dim, hidden_dim) <= 0:
            raise ValueError("Predictor dimensions must be positive.")

        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.action_embedding = nn.Embedding(num_actions, action_dim)
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, context: Tensor, action: Tensor) -> Tensor:
        if context.ndim != 2 or context.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected context shape [batch, {self.latent_dim}], "
                f"got {tuple(context.shape)}."
            )
        if action.ndim != 1 or action.shape[0] != context.shape[0]:
            raise ValueError(
                "Actions must have shape [batch] and match the context batch."
            )
        if torch.any(action < 0) or torch.any(action >= self.num_actions):
            raise ValueError("Action index is outside the configured action space.")

        action_features = self.action_embedding(action)
        delta = self.network(torch.cat((context, action_features), dim=-1))
        return context + delta


@dataclass(frozen=True)
class JEPAOutput:
    context_embedding: Tensor
    predicted_target_embedding: Tensor
    target_embedding: Tensor


@dataclass(frozen=True)
class JEPALoss:
    total: Tensor
    prediction: Tensor
    variance: Tensor
    covariance: Tensor


class ActionJEPA(nn.Module):
    """Minimal action-conditioned JEPA with an EMA target encoder."""

    def __init__(
        self,
        *,
        height: int,
        width: int,
        input_channels: int = 4,
        latent_dim: int = 32,
        hidden_channels: int = 32,
        position_dim: int | None = None,
        position_scale: float = 4.0,
        action_dim: int = 8,
        predictor_hidden_dim: int = 64,
        num_actions: int = 4,
    ) -> None:
        super().__init__()
        self.context_encoder = GridStateEncoder(
            height=height,
            width=width,
            input_channels=input_channels,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            position_dim=position_dim,
            position_scale=position_scale,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = ActionConditionedPredictor(
            latent_dim=latent_dim,
            num_actions=num_actions,
            action_dim=action_dim,
            hidden_dim=predictor_hidden_dim,
        )

    def train(self, mode: bool = True) -> ActionJEPA:
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(
        self,
        observation: Tensor,
        action: Tensor,
        next_observation: Tensor,
    ) -> JEPAOutput:
        context_embedding = self.context_encoder(observation)
        predicted_target = self.predictor(context_embedding, action)
        with torch.no_grad():
            target_embedding = self.target_encoder(next_observation)
        return JEPAOutput(
            context_embedding=context_embedding,
            predicted_target_embedding=predicted_target,
            target_embedding=target_embedding,
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.99) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be between zero and one.")

        for target, context in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
            strict=True,
        ):
            target.mul_(momentum).add_(context, alpha=1.0 - momentum)
        for target, context in zip(
            self.target_encoder.buffers(),
            self.context_encoder.buffers(),
            strict=True,
        ):
            target.copy_(context)


def variance_floor_loss(
    embedding: Tensor,
    *,
    target_standard_deviation: float = 1.0,
    epsilon: float = 1e-4,
) -> Tensor:
    """Penalize latent dimensions whose batch variance collapses."""
    if embedding.ndim != 2 or embedding.shape[0] < 2:
        raise ValueError("Variance regularization requires a 2D batch of size >= 2.")
    if target_standard_deviation <= 0.0:
        raise ValueError("Target standard deviation must be positive.")

    standard_deviation = torch.sqrt(embedding.var(dim=0, correction=0) + epsilon)
    return F.relu(target_standard_deviation - standard_deviation).mean()


def covariance_loss(embedding: Tensor) -> Tensor:
    """Discourage redundant latent dimensions without using negative samples."""
    if embedding.ndim != 2 or embedding.shape[0] < 2:
        raise ValueError("Covariance regularization requires a 2D batch of size >= 2.")

    centered = embedding - embedding.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (embedding.shape[0] - 1)
    diagonal = torch.diag(torch.diagonal(covariance))
    off_diagonal = covariance - diagonal
    return off_diagonal.square().sum() / embedding.shape[1]


def action_jepa_loss(
    output: JEPAOutput,
    *,
    variance_weight: float = 0.1,
    covariance_weight: float = 0.01,
    target_standard_deviation: float = 1.0,
) -> JEPALoss:
    """Latent prediction objective with explicit anti-collapse regularization."""
    if variance_weight < 0.0 or covariance_weight < 0.0:
        raise ValueError("Regularization weights must be non-negative.")

    prediction = F.smooth_l1_loss(
        output.predicted_target_embedding,
        output.target_embedding.detach(),
    )
    variance = 0.5 * (
        variance_floor_loss(
            output.context_embedding,
            target_standard_deviation=target_standard_deviation,
        )
        + variance_floor_loss(
            output.predicted_target_embedding,
            target_standard_deviation=target_standard_deviation,
        )
    )
    covariance = 0.5 * (
        covariance_loss(output.context_embedding)
        + covariance_loss(output.predicted_target_embedding)
    )
    total = prediction + variance_weight * variance + covariance_weight * covariance
    return JEPALoss(
        total=total,
        prediction=prediction,
        variance=variance,
        covariance=covariance,
    )
