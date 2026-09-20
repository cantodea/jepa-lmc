from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.improve_top1 import cached_training_data
from experiments.improve_top1_ranking import (
    PROTOCOL,
    batch_maps,
    check_validation_counts,
    select_model,
    training_maps,
)
from experiments.top1_quality_diagnosis import SEEDS, digest
from jepa_lmc.learning.model import ActionJEPA
from jepa_lmc.learning.ranking import same_map_ranking_loss


class Top1RankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_ranking_corrects_wrong_choice_and_detaches_target(self):
        prediction = torch.tensor([[0.2]], requires_grad=True)
        targets = torch.tensor([[0.0], [2.0]], requires_grad=True)
        labels = torch.tensor([1])
        maps = torch.zeros(2, dtype=torch.long)
        loss = same_map_ranking_loss(prediction, targets, labels, maps[:1], maps)
        loss.backward()
        self.assertIsNone(targets.grad)
        self.assertLess(float(prediction.grad), 0)
        updated = prediction.detach() - 0.5 * prediction.grad
        self.assertLess(
            float(same_map_ranking_loss(updated, targets, labels, maps[:1], maps)),
            float(loss.detach()),
        )
        self.assertEqual(int((updated - targets.detach().T).square().argmin()), 1)

    def test_validation_guard_accepts_roundoff_but_not_verdict_changes(self):
        frozen = {"transitions": 2404, "errors": 152, "accuracy": 2252 / 2404}
        equivalent = frozen | {"accuracy": 1 - 152 / 2404}
        self.assertNotEqual(frozen["accuracy"], equivalent["accuracy"])
        check_validation_counts(equivalent, frozen)
        with self.assertRaisesRegex(AssertionError, "counts"):
            check_validation_counts(equivalent | {"errors": 151}, frozen)

    def test_other_maps_cannot_act_as_negatives(self):
        prediction = torch.tensor([[0.5]], requires_grad=True)
        targets = torch.tensor([[0.0], [2.0]])
        expected = same_map_ranking_loss(
            prediction,
            targets,
            torch.tensor([0]),
            torch.tensor([4]),
            torch.tensor([4, 4]),
        )
        actual = same_map_ranking_loss(
            prediction,
            torch.cat((targets, torch.tensor([[0.5], [-100.0]]))),
            torch.tensor([0]),
            torch.tensor([4]),
            torch.tensor([4, 4, 5, 5]),
        )
        self.assertEqual(float(expected.detach()), float(actual.detach()))
        self.assertTrue(
            torch.equal(
                torch.autograd.grad(expected, prediction)[0],
                torch.autograd.grad(actual, prediction)[0],
            )
        )
        with self.assertRaisesRegex(ValueError, "query's map"):
            same_map_ranking_loss(
                prediction,
                targets,
                torch.tensor([1]),
                torch.tensor([4]),
                torch.tensor([4, 5]),
            )

    def test_depth_warm_start_preserves_function_and_old_keys(self):
        torch.manual_seed(3)
        baseline = ActionJEPA(height=3, width=3).eval()
        deeper = ActionJEPA(height=3, width=3, predictor_residual_blocks=1).eval()
        missing = deeper.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(missing.unexpected_keys)
        self.assertTrue(missing.missing_keys)
        self.assertTrue(
            all(
                k.startswith("predictor.residual_blocks.") for k in missing.missing_keys
            )
        )
        self.assertFalse(any("residual_blocks" in k for k in baseline.state_dict()))
        observations = torch.randn(12, 4, 3, 3)
        actions = torch.arange(4).repeat(3)
        with torch.no_grad():
            old = baseline(observations, actions, observations)
            new = deeper(observations, actions, observations)
        self.assertTrue(
            torch.equal(old.predicted_target_embedding, new.predicted_target_embedding)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save(deeper.state_dict(), path)
            restored = ActionJEPA(height=3, width=3, predictor_residual_blocks=1)
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)

    def test_map_batches_are_exactly_original_training_examples(self):
        maps, map_hash = training_maps()
        original, original_hash = cached_training_data()
        self.assertEqual(map_hash, original_hash)
        self.assertEqual(len(original), 4736)
        obs, source, action, successor, membership = batch_maps(maps[:4])
        self.assertTrue(torch.equal(membership[source], membership[successor]))
        for i in range(len(action)):
            self.assertTrue(torch.equal(obs[source[i]], original[i]["observation"]))
            self.assertTrue(
                torch.equal(obs[successor[i]], original[i]["next_observation"])
            )
            self.assertEqual(int(action[i]), int(original[i]["action"]))

    def test_selection_uses_validation_and_freezes_once(self):
        protocol = json.loads(PROTOCOL.read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baselines = {s: root / f"baseline_{s}.pt" for s in SEEDS}
            for j, config in enumerate(protocol["candidates"]):
                for seed in SEEDS:
                    output = root / config["name"] / f"seed_{seed}"
                    output.mkdir(parents=True)
                    (output / "model.pt").write_bytes(b"test fixture")
                    (output / "validation.json").write_text(
                        json.dumps(
                            {
                                "protocol_sha256": digest(PROTOCOL),
                                "checkpoint_sha256": digest(output / "model.pt"),
                                "initial_validation": {"accuracy": 0.9},
                                "validation": {"accuracy": [0.91, 0.95, 0.92][j]},
                                "trainable_parameters": 90776 + j,
                                "best_additional_epoch": 25,
                            }
                        )
                    )
                    (output / "report.json").write_text("must not read stress results")
            with (
                patch("experiments.improve_top1_ranking.ROOT", root),
                patch("experiments.improve_top1_ranking.BASELINES", baselines),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                select_model(root, protocol)
                result = json.loads((root / "selection.json").read_text())
                self.assertEqual(result["selected"], "ranking")
                self.assertTrue(result["clear_validation_improvement"])
                self.assertFalse(result["stress_used_for_selection"])
                with self.assertRaisesRegex(ValueError, "frozen"):
                    select_model(root, protocol)
                (root / "selection.json").unlink()
                (root / "ranking" / f"seed_{SEEDS[0]}" / "model.pt").write_bytes(
                    b"changed"
                )
                with self.assertRaisesRegex(AssertionError, "checkpoint changed"):
                    select_model(root, protocol)


if __name__ == "__main__":
    unittest.main()
