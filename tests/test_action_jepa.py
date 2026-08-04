from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader

from jepa_lmc.benchmarks.random_gridworld import (
    generate_random_gridworld_spec,
)
from jepa_lmc.data.jepa_transitions import (
    AGENT_CHANNEL,
    DANGER_CHANNEL,
    GOAL_CHANNEL,
    OBSERVATION_CHANNELS,
    WALL_CHANNEL,
    GridWorldTransitionDataset,
    gridworld_observation,
)
from jepa_lmc.envs.gridworld import GridWorld
from jepa_lmc.models.action_jepa import (
    ActionJEPA,
    JEPAOutput,
    action_jepa_loss,
    variance_floor_loss,
)
from jepa_lmc.training.action_jepa import (
    embedding_statistics,
    make_action_jepa_optimizer,
    train_action_jepa_epoch,
)


class GridWorldJEPADataTests(unittest.TestCase):
    def test_observation_channels_encode_map_and_agent(self) -> None:
        env = GridWorld(
            width=4,
            height=3,
            start=(0, 0),
            goal=(2, 3),
            walls={(1, 1)},
            dangers={(1, 2)},
        )
        observation = gridworld_observation(env, (0, 2))

        self.assertEqual(tuple(observation.shape), (OBSERVATION_CHANNELS, 3, 4))
        self.assertEqual(observation[WALL_CHANNEL, 1, 1].item(), 1.0)
        self.assertEqual(observation[DANGER_CHANNEL, 1, 2].item(), 1.0)
        self.assertEqual(observation[GOAL_CHANNEL, 2, 3].item(), 1.0)
        self.assertEqual(observation[AGENT_CHANNEL, 0, 2].item(), 1.0)
        self.assertEqual(observation[AGENT_CHANNEL].sum().item(), 1.0)

    def test_transition_dataset_contains_every_state_action_pair(self) -> None:
        spec = generate_random_gridworld_spec(
            12,
            width=4,
            height=4,
            min_walls=1,
            max_walls=1,
            min_dangers=1,
            max_dangers=1,
        )
        env = spec.make_env()
        dataset = GridWorldTransitionDataset((spec,))

        self.assertEqual(len(dataset), len(env.all_states()) * len(env.ACTIONS))
        sample = dataset[0]
        self.assertEqual(sample["action"].dtype, torch.long)
        self.assertEqual(tuple(sample["observation"].shape), (4, 4, 4))
        self.assertEqual(tuple(sample["next_observation"].shape), (4, 4, 4))

    def test_empty_transition_dataset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            GridWorldTransitionDataset(())


class ActionJEPATests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = ActionJEPA(
            height=6,
            width=6,
            latent_dim=12,
            hidden_channels=8,
            action_dim=4,
            predictor_hidden_dim=16,
        )

    def test_target_encoder_is_frozen_and_kept_in_evaluation_mode(self) -> None:
        self.model.train()

        self.assertFalse(self.model.target_encoder.training)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in self.model.target_encoder.parameters()
            )
        )

    def test_predictor_depends_on_action(self) -> None:
        observation = torch.randn(1, 4, 6, 6).expand(4, -1, -1, -1)
        context = self.model.context_encoder(observation)
        predictions = self.model.predictor(context, torch.arange(4))

        self.assertEqual(tuple(predictions.shape), (4, 12))
        self.assertGreater(torch.pdist(predictions).max().item(), 0.0)

    def test_encoder_preserves_agent_position_before_training(self) -> None:
        env = GridWorld(
            width=6,
            height=6,
            start=(0, 0),
            goal=(5, 5),
            walls=set(),
            dangers=set(),
        )
        observations = torch.stack(
            (
                gridworld_observation(env, (0, 0)),
                gridworld_observation(env, (5, 5)),
            )
        )
        embeddings = self.model.context_encoder(observations)

        self.assertFalse(torch.allclose(embeddings[0], embeddings[1]))
        self.assertEqual(
            embeddings[0, -8:].tolist(),
            [0.0, 0.0, 0.0, 4.0, 0.0, 4.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(embeddings[1, -8].item(), 4.0)
        self.assertAlmostEqual(embeddings[1, -7].item(), 4.0)
        self.assertAlmostEqual(embeddings[1, -5].item(), -4.0, places=5)

    def test_target_encoder_updates_only_through_ema(self) -> None:
        context_parameter = next(self.model.context_encoder.parameters())
        target_parameter = next(self.model.target_encoder.parameters())
        with torch.no_grad():
            context_parameter.fill_(2.0)
            target_parameter.zero_()

        self.model.update_target_encoder(momentum=0.25)

        self.assertTrue(
            torch.allclose(
                target_parameter,
                torch.full_like(target_parameter, 1.5),
            )
        )

    def test_loss_trains_context_and_predictor_but_not_target(self) -> None:
        observation = torch.randn(8, 4, 6, 6)
        next_observation = torch.randn(8, 4, 6, 6)
        output = self.model(observation, torch.arange(8) % 4, next_observation)
        loss = action_jepa_loss(output)
        loss.total.backward()

        self.assertTrue(torch.isfinite(loss.total))
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in self.model.context_encoder.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in self.model.predictor.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in self.model.target_encoder.parameters()
            )
        )

    def test_variance_regularizer_penalizes_collapsed_embeddings(self) -> None:
        collapsed = torch.zeros(16, 8)
        varied = torch.randn(16, 8) * 3.0

        self.assertGreater(
            variance_floor_loss(collapsed).item(),
            variance_floor_loss(varied).item(),
        )

    def test_loss_rejects_single_example_batch(self) -> None:
        embedding = torch.zeros(1, 8)
        output = JEPAOutput(embedding, embedding, embedding)
        with self.assertRaisesRegex(ValueError, "batch of size"):
            action_jepa_loss(output)

    def test_one_training_epoch_reports_finite_diagnostics(self) -> None:
        spec = generate_random_gridworld_spec(
            99,
            min_walls=2,
            max_walls=2,
            min_dangers=1,
            max_dangers=1,
        )
        dataset = GridWorldTransitionDataset((spec,))
        loader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=True)
        optimizer = make_action_jepa_optimizer(self.model, learning_rate=1e-3)

        metrics = train_action_jepa_epoch(
            self.model,
            loader,
            optimizer,
            ema_momentum=0.9,
        )

        self.assertGreater(metrics.examples, 0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics.loss)))
        batch = next(iter(loader))
        with torch.no_grad():
            embedding = self.model.context_encoder(batch["observation"])
        statistics = embedding_statistics(embedding)
        self.assertGreater(statistics.effective_rank, 1.0)


if __name__ == "__main__":
    unittest.main()
