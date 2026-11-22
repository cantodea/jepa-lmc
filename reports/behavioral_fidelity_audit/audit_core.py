"""Truth-free observation projection and score replay. States are catalogue IDs."""
from __future__ import annotations

import hashlib
import io

import torch

from jepa_lmc.learning.finite_trace import DirectSuccessorMLP, known_observations
from jepa_lmc.learning.model import ActionJEPA


def observations(goods):
    observed = {}
    for tr in goods:
        states = list(map(tuple, tr["states"]))
        for s, a, t in zip(states[:-1], tr["actions"], states[1:], strict=True):
            if observed.setdefault((s, a), t) != t:
                raise ValueError("Contradictory positive observations")
    return observed


def project(pred, observed):
    if not observed.keys() <= pred.keys():
        raise ValueError("Positive observation outside prediction domain")
    return {**pred, **observed}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_weights(data, expected, kind):
    if sha(data) != expected:
        raise ValueError("Checkpoint SHA-256 mismatch")
    cp = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    kwargs = cp["model_kwargs"] if "model_kwargs" in cp else cp["metadata"]["model_kwargs"]
    model = (ActionJEPA if kind == "jepa" else DirectSuccessorMLP)(**kwargs)
    model.load_state_dict(cp.get("state_dict", cp.get("model_state_dict")), strict=True)
    return model.eval(), kwargs


@torch.inference_mode()
def scores(model, kind, problem, height, width):
    obs = known_observations(problem, height=height, width=width)
    n, k = len(problem.states), len(problem.actions)
    si = torch.arange(n).repeat_interleave(k)
    ai = torch.arange(k).repeat(n)
    if kind == "jepa":
        context = model.context_encoder(obs)
        query = model.predictor(context[si], ai)
        values = -torch.cdist(query, model.target_encoder(obs))
    else:
        values = model(obs[si], ai)
    return values


def error_auc(pairs):
    """P(gap_error < gap_correct), half credit for ties."""
    vals = sorted(pairs)
    total_error = sum(e for _, e in vals)
    total_right = len(vals) - total_error
    if not total_error or not total_right:
        return None
    from itertools import groupby
    errors_before = numerator = 0
    for _, group in groupby(vals, key=lambda p: p[0]):
        block = [e for _, e in group]
        errors = sum(block)
        numerator += (len(block) - errors) * (errors_before + errors / 2)
        errors_before += errors
    return numerator / (total_error * total_right)
