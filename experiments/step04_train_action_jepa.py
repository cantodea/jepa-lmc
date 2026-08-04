from __future__ import annotations

import argparse
import random

import torch
from torch.utils.data import DataLoader

from jepa_lmc.benchmarks.random_gridworld import make_pilot_benchmark_splits
from jepa_lmc.data.jepa_transitions import GridWorldTransitionDataset
from jepa_lmc.models.action_jepa import ActionJEPA
from jepa_lmc.training.action_jepa import (
    embedding_statistics,
    make_action_jepa_optimizer,
    train_action_jepa_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the first action-conditioned JEPA GridWorld pilot."
    )
    parser.add_argument("--maps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.maps <= 0 or args.epochs <= 0 or args.batch_size < 2:
        raise ValueError("Maps and epochs must be positive; batch size must be >= 2.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = make_pilot_benchmark_splits()
    if args.maps > len(splits.train):
        raise ValueError(
            f"Requested {args.maps} maps, but only {len(splits.train)} are available."
        )

    specs = splits.train[: args.maps]
    dataset = GridWorldTransitionDataset(specs)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    first_spec = specs[0]
    model = ActionJEPA(
        height=first_spec.height,
        width=first_spec.width,
        latent_dim=args.latent_dim,
    )
    optimizer = make_action_jepa_optimizer(
        model,
        learning_rate=args.learning_rate,
    )

    print("=== Step 04A: action-conditioned JEPA pilot ===")
    print(f"Training maps: {len(specs)}")
    print(f"Transitions: {len(dataset)}")
    print(f"Latent dimension: {args.latent_dim}")

    for epoch in range(1, args.epochs + 1):
        metrics = train_action_jepa_epoch(
            model,
            loader,
            optimizer,
            ema_momentum=args.ema_momentum,
        )
        print(
            f"Epoch {epoch:02d}: total={metrics.loss:.5f}, "
            f"prediction={metrics.prediction_loss:.5f}, "
            f"variance={metrics.variance_loss:.5f}, "
            f"covariance={metrics.covariance_loss:.5f}"
        )

    diagnostic_batch = next(iter(loader))
    model.eval()
    with torch.no_grad():
        context = model.context_encoder(diagnostic_batch["observation"])
        statistics = embedding_statistics(context)
        one_context = context[:1].expand(4, -1)
        action_predictions = model.predictor(
            one_context,
            torch.arange(4, dtype=torch.long),
        )
        action_separation = torch.pdist(action_predictions).mean().item()

    print("\n=== Latent health diagnostics ===")
    print(
        "Mean per-dimension standard deviation: "
        f"{statistics.mean_dimension_standard_deviation:.5f}"
    )
    print(f"Effective rank: {statistics.effective_rank:.2f}")
    print(f"Mean embedding norm: {statistics.mean_embedding_norm:.5f}")
    print(f"Action-conditioned prediction separation: {action_separation:.5f}")

    if statistics.mean_dimension_standard_deviation <= 0.0:
        raise AssertionError("Context embeddings collapsed to a constant.")
    if action_separation <= 0.0:
        raise AssertionError("Predictor ignored the action input.")
    print("\nAction-JEPA architecture smoke test passed.")


if __name__ == "__main__":
    main()
