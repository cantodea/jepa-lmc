from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from experiments.improve_top1 import PROTOCOL, learning_rate, select_model
from experiments.top1_quality_diagnosis import (
    SEEDS,
    digest,
    fresh_directory,
    map_fingerprint,
    rotate_spec,
    split_audit,
)
from jepa_lmc.benchmarks.random_gridworld import make_pilot_benchmark_splits


class Top1QualityTests(unittest.TestCase):
    def test_observation_level_split_audit_and_rotation_roundtrip(self):
        audit = split_audit()
        self.assertEqual(audit["training_orbit_vs_evaluation_map_overlap"], 0)
        spec = make_pilot_benchmark_splits().train[0]
        self.assertEqual(rotate_spec(spec, 4), spec)
        self.assertEqual(rotate_spec(rotate_spec(spec, 1), 3), spec)
        # Different designated starts do not make observations held out.
        moved = replace(spec, start=spec.goal)
        self.assertEqual(map_fingerprint(spec), map_fingerprint(moved))

    def test_fixed_budget_and_schedule_endpoints(self):
        protocol = json.loads(PROTOCOL.read_text())
        self.assertEqual(len(protocol["candidates"]) * len(SEEDS), 9)
        for config in protocol["candidates"]:
            self.assertEqual(config["final_epoch"], 300)
            self.assertEqual(learning_rate(config, 100), 3e-4)
            self.assertEqual(learning_rate(config, 101), 3e-4)
            self.assertAlmostEqual(
                learning_rate(config, 300), config["final_learning_rate"]
            )

    def test_selection_reads_validation_and_requires_all_seeds_and_intact_weights(self):
        protocol = json.loads(PROTOCOL.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "diagnosis").mkdir()
            baseline = [
                {
                    "seed": s,
                    "trainable_parameters": 10,
                    "scores": {
                        "by_split": [
                            {"split": "validation", "accuracy": 0.9},
                            {"split": "stress", "accuracy": 1.0},
                        ]
                    },
                }
                for s in SEEDS
            ]
            (root / "diagnosis/report.json").write_text(
                json.dumps({"models": baseline})
            )
            for i, config in enumerate(protocol["candidates"]):
                for seed in SEEDS:
                    directory = root / config["name"] / f"seed_{seed}"
                    directory.mkdir(parents=True)
                    (directory / "model.pt").write_bytes(b"fixture weights")
                    metadata = {
                        "protocol_sha256": digest(PROTOCOL),
                        "checkpoint_sha256": digest(directory / "model.pt"),
                        "validation": {"accuracy": (0.92, 0.98, 0.94)[i]},
                        "trainable_parameters": 10 + i,
                    }
                    (directory / "validation.json").write_text(json.dumps(metadata))
                    # Deliberately invalid stress report must never be read.
                    (directory / "report.json").write_text("DO NOT SELECT BY STRESS")
            paths = {s: root / f"baseline_{s}.pt" for s in SEEDS}
            with (
                patch("experiments.improve_top1.ROOT", root),
                patch("experiments.improve_top1.BASELINES", paths),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                select_model(root, protocol)
                selected = json.loads((root / "selection.json").read_text())
                self.assertEqual(selected["selected"], "long_cosine")
                self.assertTrue(selected["clear_validation_improvement"])
                self.assertEqual(len(selected["selected_checkpoints"]), 3)
                with self.assertRaisesRegex(ValueError, "already frozen"):
                    select_model(root, protocol)
                (root / "selection.json").unlink()
                damaged = root / "long_constant" / f"seed_{SEEDS[0]}" / "model.pt"
                damaged.write_bytes(b"changed")
                with self.assertRaisesRegex(AssertionError, "checkpoint changed"):
                    select_model(root, protocol)
                damaged.write_bytes(b"fixture weights")
                (damaged.parent / "validation.json").unlink()
                with self.assertRaises(FileNotFoundError):
                    select_model(root, protocol)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                fresh_directory(root)


if __name__ == "__main__":
    unittest.main()
