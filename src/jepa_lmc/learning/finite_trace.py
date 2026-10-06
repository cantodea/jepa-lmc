"""Train finite-trace predictors without an environment or transition oracle.

The only labelled training examples are supplied traces. Candidate observations
describe known S,L and never associate an unseen (s,a) with its real successor.
Existing JEPA architecture, losses, EMA, ranking and Top-1 decoder are reused.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from jepa_lmc.evaluation.trace_sanity import KnownProblem, Trace, observed_edges
from jepa_lmc.learning.model import ActionJEPA, JEPAOutput, action_jepa_loss
from jepa_lmc.learning.ranking import same_map_ranking_loss
from jepa_lmc.learning.training import make_action_jepa_optimizer


def known_observations(problem: KnownProblem, *, height: int, width: int) -> Tensor:
    """Reproduce gridworld_observation from the known catalogue, without env/T."""
    if height < 2 or width < 2 or not problem.states:
        raise ValueError("Need a nonempty catalogue and grid dimensions >= 2.")
    states = set(problem.states)
    if len(states) != len(problem.states) or any(
        not (0 <= r < height and 0 <= c < width) for r, c in states
    ):
        raise ValueError("Invalid or duplicate catalogue coordinates.")
    static = torch.zeros(4, height, width, dtype=torch.float32)
    for r in range(height):
        for c in range(width):
            s = r, c
            if s not in states:
                static[0, r, c] = 1
            else:
                static[1, r, c] = float("danger" in problem.labels[s])
                static[2, r, c] = float("goal" in problem.labels[s])
    observations = static.repeat(len(problem.states), 1, 1, 1)
    for i, (r, c) in enumerate(problem.states):
        observations[i, 3, r, c] = 1
    return observations


def tensor_digest(*named_tensors) -> str:
    hasher = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        hasher.update(name.encode())
        hasher.update(str(value.dtype).encode())
        hasher.update(json.dumps(list(value.shape)).encode())
        hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


@dataclass(frozen=True)
class FiniteTraceData:
    """Only known state images and observed labelled examples cross this boundary."""

    problem: KnownProblem
    observations: Tensor
    sources: Tensor
    actions: Tensor
    successors: Tensor
    observed_pairs: frozenset[tuple]
    trace_sha256: str
    training_sha256: str

    @classmethod
    def from_traces(
        cls, problem: KnownProblem, traces: Sequence[Trace], *, height: int, width: int
    ) -> FiniteTraceData:
        if not traces or any(not problem.accepts_syntax(t) for t in traces):
            raise ValueError("Training traces must use known states/actions/initials.")
        edges = [edge for trace in traces for edge in trace.edges]
        if len(edges) < 2:
            raise ValueError("JEPA regularization requires at least two trace steps.")
        relation = observed_edges(traces)
        targets = {}
        for s, a, t in relation:
            if (s, a) in targets and targets[s, a] != t:
                raise ValueError("Observed transitions must be action deterministic.")
            targets[s, a] = t
        indices = {s: i for i, s in enumerate(problem.states)}
        actions = {a: i for i, a in enumerate(problem.actions)}
        observations = known_observations(problem, height=height, width=width)
        source = torch.tensor([indices[s] for s, _, _ in edges], dtype=torch.long)
        action = torch.tensor([actions[a] for _, a, _ in edges], dtype=torch.long)
        target = torch.tensor([indices[t] for _, _, t in edges], dtype=torch.long)
        trace_json = json.dumps(
            [{"states": t.states, "actions": t.actions} for t in traces],
            separators=(",", ":"),
        ).encode()
        return cls(
            problem,
            observations,
            source,
            action,
            target,
            frozenset(targets),
            hashlib.sha256(trace_json).hexdigest(),
            tensor_digest(
                ("known_observations", observations),
                ("observed_sources", source),
                ("observed_actions", action),
                ("observed_successors", target),
            ),
        )


class DirectSuccessorMLP(nn.Module):
    """One hidden layer, direct successor classification; no latent objective."""

    def __init__(
        self,
        *,
        height: int,
        width: int,
        num_states: int,
        num_actions: int = 4,
        hidden_dim: int = 64,
        action_dim: int = 8,
    ) -> None:
        super().__init__()
        self.observation_shape = (4, height, width)
        self.action_embedding = nn.Embedding(num_actions, action_dim)
        self.network = nn.Sequential(
            nn.Linear(4 * height * width + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_states),
        )

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        if (
            observation.ndim != 4
            or tuple(observation.shape[1:]) != self.observation_shape
        ):
            raise ValueError("MLP observations have the wrong shape.")
        features = torch.cat(
            (observation.flatten(1), self.action_embedding(action)), dim=1
        )
        return self.network(features)


def fit_finite_model(data: FiniteTraceData, *, kind: str, seed: int, config: dict):
    """Fit from scratch, fixed epochs, without truth/evaluation/heldout arguments."""
    if kind not in ("mlp", "jepa"):
        raise ValueError("Choose mlp or jepa.")
    train = config["training"]
    if train["epochs"] < 1 or train["device"] != "cpu":
        raise ValueError("This reproducibility protocol uses positive epochs on CPU.")
    torch.set_num_threads(train["threads"])
    torch.use_deterministic_algorithms(train["deterministic_algorithms"])
    torch.manual_seed(seed)
    height, width = data.observations.shape[-2:]
    kwargs = {
        "height": height,
        "width": width,
        "num_actions": len(data.problem.actions),
    }
    if kind == "mlp":
        kwargs.update(
            num_states=len(data.problem.states),
            hidden_dim=config["mlp"]["hidden_dim"],
            action_dim=config["mlp"]["action_dim"],
        )
        model = DirectSuccessorMLP(**kwargs)
    else:
        kwargs.update(
            {
                k: config["jepa"][k]
                for k in (
                    "latent_dim",
                    "hidden_channels",
                    "position_dim",
                    "position_scale",
                    "action_dim",
                    "predictor_hidden_dim",
                    "predictor_residual_blocks",
                )
            }
        )
        model = ActionJEPA(**kwargs)
    initial_sha = tensor_digest(*model.state_dict().items())
    optimizer = make_action_jepa_optimizer(
        model, learning_rate=train["learning_rate"], weight_decay=train["weight_decay"]
    )
    history = []
    candidate_maps = torch.zeros(len(data.problem.states), dtype=torch.long)
    query_maps = torch.zeros(len(data.actions), dtype=torch.long)
    started = time.perf_counter()
    for epoch in range(1, train["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if kind == "mlp":
            logits = model(data.observations[data.sources], data.actions)
            loss = F.cross_entropy(logits, data.successors)
            losses = {"cross_entropy": loss}
        else:
            # As in the existing ranking loop: encode each known state once;
            # select only observed sources and positives for labelled training.
            context = model.context_encoder(data.observations)
            prediction = model.predictor(context[data.sources], data.actions)
            with torch.no_grad():
                bank = model.target_encoder(data.observations)
            jepa = config["jepa"]
            base = action_jepa_loss(
                JEPAOutput(context[data.sources], prediction, bank[data.successors]),
                variance_weight=jepa["variance_weight"],
                covariance_weight=jepa["covariance_weight"],
            )
            ranking = same_map_ranking_loss(
                prediction,
                bank,
                data.successors,
                query_maps,
                candidate_maps,
                temperature=jepa["ranking_temperature"],
            )
            loss = base.total + jepa["ranking_weight"] * ranking
            losses = {
                "prediction": base.prediction,
                "variance": base.variance,
                "covariance": base.covariance,
                "ranking": ranking,
            }
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite finite-trace training loss.")
        loss.backward()
        optimizer.step()
        if kind == "jepa":
            model.update_target_encoder(config["jepa"]["ema_momentum"])
        if epoch == 1 or epoch % 50 == 0 or epoch == train["epochs"]:
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    **{k: float(v.detach()) for k, v in losses.items()},
                }
            )
    # Data mutation would violate the shared-training-example contract.
    actual = tensor_digest(
        ("known_observations", data.observations),
        ("observed_sources", data.sources),
        ("observed_actions", data.actions),
        ("observed_successors", data.successors),
    )
    if actual != data.training_sha256:
        raise AssertionError("Training data were mutated.")
    model.eval()
    return model, {
        "kind": kind,
        "initialization_seed": seed,
        "from_scratch": True,
        "model_kwargs": kwargs,
        "epochs": train["epochs"],
        "optimizer_steps": train["epochs"],
        "steps_per_epoch": len(data.actions),
        "examples_processed": train["epochs"] * len(data.actions),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "initial_state_sha256": initial_sha,
        "final_state_sha256": tensor_digest(*model.state_dict().items()),
        "training_sha256": data.training_sha256,
        "trace_sha256": data.trace_sha256,
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "checkpoint_selection": "fixed final epoch; no evaluator input",
    }


@torch.no_grad()
def predict_successors(model, data: FiniteTraceData, *, kind: str) -> dict:
    """Query all known (s,a); no real targets enter inference or candidate choice."""
    model.eval()
    count, actions = len(data.problem.states), len(data.problem.actions)
    sources = torch.arange(count).repeat_interleave(actions)
    action = torch.arange(actions).repeat(count)
    if kind == "mlp":
        predictions = model(data.observations[sources], action).argmax(dim=1)
    elif kind == "jepa":
        context = model.context_encoder(data.observations)
        targets = model.target_encoder(data.observations)
        prediction = model.predictor(context[sources], action)
        predictions = torch.cdist(prediction, targets).argmin(dim=1)
    else:
        raise ValueError("Choose mlp or jepa.")
    return {
        (s, a): data.problem.states[int(t)]
        for (s, a), t in zip(
            ((s, a) for s in data.problem.states for a in data.problem.actions),
            predictions,
            strict=True,
        )
    }
