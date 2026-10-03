"""Shared deterministic, offline fixtures for the dfl_sim test suite.

Nothing here downloads data or reads files from disk. Datasets are generated
in-process from a seeded :class:`torch.Generator`, and the config builder
reuses the production strict parser so tests exercise the real contract.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from dfl_sim import (
    ATTACK_KAPPAS,
    AttackAction,
    EvaluatedRound,
    ExperimentConfig,
    MessageSummary,
    NodeEvaluation,
    PropagationState,
    RoundResult,
    load_topology,
    parse_config_text,
    write_trace,
)

__all__ = [
    "FailingDataset",
    "MNISTShapedDataset",
    "constant_loss_evaluator_factory",
    "make_config",
    "make_viewer_trace",
]

_BASE_CONFIG: dict = {
    "seed": 20240101,
    "rounds": 2,
    "device": "cpu",
    "bitwise": True,
    "model": {"name": "small_cnn"},
    "data": {
        "dataset": "mnist",
        "partition": "iid",
        "samples_per_node": 32,
        "data_root": "data",
        "download": False,
    },
    "training": {
        "optimizer": "sgd",
        "learning_rate": 0.1,
        "batch_size": 16,
        "local_epochs": 3,
        "num_workers": 0,
    },
    # Honest-by-default D016 attack block so tests can override one fixed
    # per-episode attacker through the same strict parser as production.
    "attack": {"compromised_node": None, "intensity": "zero"},
    "topology": {"kind": "chain3"},
    # Explicit D008 baseline selector; overridable like every other section.
    "aggregation": {"kind": "plain_average"},
    # Bounded privileged-evaluation controls; small defaults keep fixture
    # tests fast while still exercising the real evaluator config contract.
    "evaluation": {"max_samples": 32, "batch_size": 16},
    # Trace snapshot cadence; exercised by the trajectory/CLI slices.
    "trace": {"snapshot_cadence": 1},
}


def make_config(**overrides) -> ExperimentConfig:
    """Build a validated config from a base dict with shallow overrides."""

    data = {
        section: (dict(value) if isinstance(value, dict) else value)
        for section, value in _BASE_CONFIG.items()
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return parse_config_text(yaml.safe_dump(data, sort_keys=True))


def constant_loss_evaluator_factory(config, partition, node_id, round_index):
    """Deterministic stub loss evaluator for scalar-state protocol fixtures.

    Matches the :data:`~dfl_sim.simulator.LossEvaluatorFactory` signature and
    returns a callable that ignores both state and partition, so round-engine
    protocol tests never touch a real dataset. The real observation builder is
    still exercised; only the node-local loss value is stubbed.
    """

    def evaluate(state, *, node_id=node_id, round_index=round_index, context=""):
        return 0.0

    return evaluate


class MNISTShapedDataset(Dataset):
    """In-memory deterministic dataset of MNIST-shaped ``(N, C, H, W)`` images.

    With ``separable=True`` each class has a fixed random prototype plus small
    noise, so a model can genuinely reduce loss (used for overfit/learnability
    tests). Otherwise images are uniform noise. ``fill`` forces every image to
    a constant value (e.g. NaN) to exercise non-finite input handling.
    """

    def __init__(
        self,
        num_examples: int,
        *,
        seed: int = 0,
        num_classes: int = 10,
        channels: int = 1,
        height: int = 28,
        width: int = 28,
        separable: bool = False,
        fill: float | None = None,
    ) -> None:
        if num_examples <= 0:
            raise ValueError("num_examples must be positive")
        generator = torch.Generator().manual_seed(int(seed))
        shape = (num_examples, channels, height, width)
        self.labels = torch.arange(num_examples, dtype=torch.long) % num_classes
        if fill is not None:
            self.images = torch.full(shape, float(fill))
        elif separable:
            prototypes = torch.rand(
                num_classes, channels, height, width, generator=generator
            )
            noise = 0.02 * torch.randn(shape, generator=generator)
            self.images = prototypes[self.labels] + noise
        else:
            self.images = torch.rand(shape, generator=generator)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


# --------------------------------------------------------------------------- #
# Deterministic production-format viewer trace factory (S05/T01)
# --------------------------------------------------------------------------- #
# Typed per-node (loss, accuracy) metrics for the three-node chain used by the
# viewer tests. Values are deliberately distinct per node and improve for the
# benign nodes while the compromised node stays worse, so projection ordering
# and per-node chart series are unambiguous.
_VIEWER_METRICS: dict[int, tuple[tuple[float, float], ...]] = {
    0: ((0.90, 0.30), (0.70, 0.50)),
    1: ((0.80, 0.40), (0.60, 0.55)),
    2: ((1.50, 0.20), (1.30, 0.25)),
}


def _viewer_metrics(node_id: int, round_index: int) -> tuple[float, float]:
    table = _VIEWER_METRICS.get(int(node_id))
    if table is not None and 0 <= round_index < len(table):
        return table[round_index]
    loss = 0.5 + 0.1 * node_id + 0.1 * round_index
    accuracy = min(1.0, max(0.0, 0.5 - 0.05 * node_id + 0.05 * round_index))
    return loss, accuracy


def _viewer_state(node_id: int, step: int) -> dict[str, np.ndarray]:
    """Deterministic, distinct committed state for one node/snapshot step."""

    return {
        "weight": np.array([float(node_id), float(step)], dtype=np.float64),
        "bias": np.array(
            [float(node_id) + 0.25 * float(step)], dtype=np.float64
        ),
    }


def _viewer_messages(
    topology: nx.Graph, nodes: tuple[int, ...]
) -> tuple[MessageSummary, ...]:
    """One directed broadcast summary per sender with at least one neighbor."""

    messages: list[MessageSummary] = []
    for sender in nodes:
        recipients = tuple(
            sorted(int(neighbor) for neighbor in topology.neighbors(sender))
        )
        if recipients:
            messages.append(MessageSummary(sender=sender, recipients=recipients))
    return tuple(messages)


def _viewer_node_evaluation(
    node_id: int,
    round_index: int,
    *,
    topology: nx.Graph,
    compromised_node: int | None,
    kappa: float,
    hops: dict[int, int],
) -> NodeEvaluation:
    loss, accuracy = _viewer_metrics(node_id, round_index)
    if compromised_node is not None and node_id == compromised_node and kappa > 0.0:
        attack = AttackAction(kind="delta_reversal", kappa=kappa)
    else:
        attack = AttackAction(kind="none", kappa=0.0)

    participants = (node_id,) + tuple(
        sorted(int(neighbor) for neighbor in topology.neighbors(node_id))
    )
    count = len(participants)
    multipliers = tuple(1.0 for _ in participants)
    weights = tuple(1.0 / count for _ in participants)

    if compromised_node is None or node_id not in hops:
        propagation = PropagationState(
            reached=False, min_hop=None, first_reached_round=None
        )
    else:
        propagation = PropagationState(
            reached=True, min_hop=int(hops[node_id]), first_reached_round=0
        )

    return NodeEvaluation(
        node_id=node_id,
        loss=loss,
        accuracy=accuracy,
        sample_count=4,
        true_type="malicious" if node_id == compromised_node else "benign",
        attack=attack,
        participant_ids=participants,
        multipliers=multipliers,
        weights=weights,
        propagation=propagation,
    )


def make_viewer_trace(
    destination: str | Path,
    *,
    rounds: int = 2,
    graph: nx.Graph | None = None,
    compromised_node: int | None = 2,
    intensity: str = "weak",
) -> Path:
    """Publish a small deterministic schema-v1 trace for viewer tests.

    The default trace is a three-node chain with two rounds, a malicious node,
    directed broadcasts on every edge, and distinct per-node metrics, built
    purely through the production trajectory writer and evaluator record APIs
    (no datasets, no network, no hand-written JSON). ``graph`` lets callers
    substitute a one-node (or otherwise shaped) topology while still going
    through the real writer/reader contract.
    """

    config = make_config(
        rounds=rounds,
        attack={"compromised_node": compromised_node, "intensity": intensity},
        trace={"snapshot_cadence": 1},
        topology={"kind": "chain3"},
    )
    topology = load_topology(config) if graph is None else graph
    nodes = tuple(sorted(int(node) for node in topology.nodes()))
    if compromised_node is not None and compromised_node not in set(nodes):
        raise ValueError(
            f"compromised_node {compromised_node} is not part of the graph"
        )
    kappa = float(ATTACK_KAPPAS[intensity]) if compromised_node is not None else 0.0
    hops = (
        {}
        if compromised_node is None
        else {
            int(node): int(hop)
            for node, hop in nx.single_source_shortest_path_length(
                topology, compromised_node
            ).items()
        }
    )

    initial = {node: _viewer_state(node, 0) for node in nodes}
    rounds_data: list[tuple[EvaluatedRound, RoundResult]] = []
    for index in range(rounds):
        evaluations = tuple(
            _viewer_node_evaluation(
                node,
                index,
                topology=topology,
                compromised_node=compromised_node,
                kappa=kappa,
                hops=hops,
            )
            for node in nodes
        )
        states = {node: _viewer_state(node, index + 1) for node in nodes}
        result = RoundResult(
            round_index=index,
            states=states,
            stats=(),
            messages=_viewer_messages(topology, nodes),
        )
        rounds_data.append((EvaluatedRound(round_index=index, nodes=evaluations), result))

    destination = Path(destination)
    write_trace(
        destination,
        config=config,
        topology=topology,
        initial_states=initial,
        rounds=rounds_data,
    )
    return destination


class FailingDataset(Dataset):
    """Wraps a dataset and raises on the ``fail_after``-th access.

    Enables deterministic mid-training failure injection: a first full pass
    (e.g. pre-loss evaluation of ``N`` items) succeeds, and training then fails
    once the access counter exceeds ``fail_after``.
    """

    def __init__(self, base: Dataset, *, fail_after: int) -> None:
        self._base = base
        self._fail_after = int(fail_after)
        self._accesses = 0

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int):
        self._accesses += 1
        if self._accesses > self._fail_after:
            raise RuntimeError(f"injected dataset failure at access {self._accesses}")
        return self._base[index]
