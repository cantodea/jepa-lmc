"""Load or verify the 360 frozen Experiment 04 MLP-900 checkpoints.

Uses only the saved known-state catalogue and observed traces for inference.
No environment, unseen labels, held-out traces or formal evaluator is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import torch
from torch.nn import functional as F

from jepa_lmc.evaluation.trace_sanity import KnownProblem, Trace
from jepa_lmc.learning.finite_trace import (
    DirectSuccessorMLP,
    FiniteTraceData,
    predict_successors,
    tensor_digest,
)

DEFAULT_ARTIFACTS = (
    Path(__file__).resolve().parents[1] / "docs/results/mlp_900_fairness"
)


def case_key(row):
    return f"{row['seed']}_{row['case']}_b{row['trajectory_budget']}"


def observed_data(row):
    problem = KnownProblem(
        tuple(map(tuple, row["states"])),
        frozenset(map(tuple, row["initials"])),
        tuple(row["actions"]),
        {tuple(s): frozenset(labels) for s, labels in row["labels"]},
    )
    traces = tuple(
        Trace(tuple(map(tuple, t["states"])), tuple(t["actions"]))
        for t in row["goods"]
    )
    data = FiniteTraceData.from_traces(problem, traces, height=6, width=6)
    if (data.training_sha256, data.trace_sha256) != (
        row["training_sha256"], row["trace_sha256"]
    ):
        raise ValueError("Observed training data do not match the frozen inputs.")
    return data


def verify_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")


def load_checkpoint(archive, row, metadata):
    data = observed_data(row)
    raw = archive.read(f"{case_key(row)}_mlp_A_900.pt")
    if hashlib.sha256(raw).hexdigest() != metadata["checkpoint_file_sha256"]:
        raise ValueError("Checkpoint file hash mismatch.")
    model = DirectSuccessorMLP(
        height=6, width=6, num_states=len(data.problem.states),
        num_actions=len(data.problem.actions), hidden_dim=64, action_dim=8,
    )
    model.load_state_dict(
        torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    )
    if tensor_digest(*model.state_dict().items()) != metadata["final_state_sha256"]:
        raise ValueError("Checkpoint parameter hash mismatch.")
    return model.eval(), data


def verify_checkpoints(root, *, key=None):
    root = Path(root)
    verify_manifest(root)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    inputs = json.loads((root / "training_inputs.json").read_text(encoding="utf-8"))
    runs = {
        case_key(row): row for row in (
            json.loads(line)
            for line in (root / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    selected = [r for r in inputs if key is None or case_key(r) == key]
    if not selected:
        raise ValueError(f"Unknown seed-map-budget: {key}")
    if key is None and (len(inputs) != 360 or len(runs) != 360):
        raise ValueError("Expected all 360 cases.")
    results = []
    with zipfile.ZipFile(root / "weights.zip") as archive, torch.no_grad():
        for row in selected:
            saved = runs[case_key(row)]
            model, data = load_checkpoint(archive, row, saved)
            predictions = predict_successors(model, data, kind="mlp")
            expected = {(tuple(s), a): tuple(t) for s, a, t in saved["predictions"]}
            if predictions != expected:
                raise ValueError(f"Prediction mismatch: {case_key(row)}")
            logits = model(data.observations[data.sources], data.actions)
            correct = int((logits.argmax(dim=1) == data.successors).sum())
            results.append({
                "key": case_key(row), "parameters_sha256": saved["final_state_sha256"],
                "predictions_match": True, "observed_repeated_correct": correct,
                "observed_repeated_total": len(data.actions),
                "observed_ce": float(F.cross_entropy(logits, data.successors)),
                **({"predictions": saved["predictions"]} if key is not None else {}),
            })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--key", help="One seed_map_bBUDGET key; omit to verify all 360"
    )
    args = parser.parse_args()
    results = verify_checkpoints(args.artifacts, key=args.key)
    print(json.dumps({"verified": len(results), "cases": results}, indent=2))


if __name__ == "__main__":
    main()
