"""Privileged post-commit evaluator and weight-aware provenance (R008, D012, D018).

This module is the *observability* half of the S04 contract. It runs strictly
**after** a synchronous round has atomically committed and is structurally
outside the observation/policy call graph (D012): it may consume a validated
:class:`~dfl_sim.config.ExperimentConfig`, the frozen topology, a committed
:class:`~dfl_sim.simulator.RoundResult`, and a privileged global reference
dataset, but it never feeds anything back into training, observation, policy,
aggregation, or control.

Two responsibilities live here:

* :class:`PrivilegedEvaluator` scores **every** committed node model against
  the *same* deterministic, seed-selected, bounded reference subset (never a
  node's own policy/training partition) in ``eval`` mode and ``torch.no_grad``,
  with exact sample accounting and finite guards, and returns immutable
  node-id-sorted :class:`EvaluatedRound` records carrying per-node loss and
  accuracy plus the round's true benign/malicious type, fixed attack action,
  selected participant profile, and propagation state.
* :class:`ProvenanceTracker` implements the D018 stateful, weight-aware
  multi-hop poison tracker: an active ``kappa > 0`` compromised sender seeds
  influence at hop 0, local training preserves prior influence, and only
  participants with a strictly positive selected weight propagate influence
  into a committed recipient. It retains the minimum hop count and the first
  round at which each node was reached and fails closed on malformed,
  graph-inconsistent, skipped, or out-of-order rounds.

All inputs are untrusted boundary values (Q3): mismatched node sets, malformed
profiles/weights/labels, non-finite model tensors or metrics, and impossible
provenance transitions raise :class:`EvaluatorError` with round/node context
*before* any partial record is returned. A failed round evaluation never
advances the tracker, so the next round's provenance cannot be built on a
half-applied update.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .aggregation import AggregationError, validate_state_dict
from .config import ExperimentConfig
from .models import StateDictError, build_model, load_state_into_model
from .reproducibility import ResolvedSeeds
from .simulator import NodeWeightProfile, RoundResult
from .topology import load_topology

__all__ = [
    "ATTACK_DELTA_REVERSAL",
    "ATTACK_NONE",
    "BENIGN",
    "MALICIOUS",
    "AttackAction",
    "EvaluatedRound",
    "EvaluatorError",
    "NodeEvaluation",
    "PropagationState",
    "ProvenanceTracker",
    "PrivilegedEvaluator",
    "select_reference_subset",
    "validate_round_structure",
]

BENIGN = "benign"
MALICIOUS = "malicious"

ATTACK_NONE = "none"
ATTACK_DELTA_REVERSAL = "delta_reversal"

#: Absolute tolerance for the participant-aligned unit-sum weight invariant.
_WEIGHT_SUM_TOLERANCE = 1e-9

#: The stable seed domain used to select the shared reference subset.
_REFERENCE_SEED_LABEL = "evaluation_subset"


class EvaluatorError(RuntimeError):
    """Raised when the privileged evaluator or provenance contract is violated.

    Every instance carries round/node context so a failed evaluation is
    attributable before any event is emitted.
    """


def _suffix(context: str) -> str:
    return f" ({context})" if context else ""


def _require_int(value: object, *, label: str, context: str = "") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluatorError(
            f"{label} must be int, got {type(value).__name__}{_suffix(context)}"
        )
    return int(value)


def _require_finite_float(
    value: object, *, label: str, context: str = ""
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise EvaluatorError(
            f"{label} must be a real number, got {type(value).__name__}"
            f"{_suffix(context)}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise EvaluatorError(f"{label} must be finite, got {result!r}{_suffix(context)}")
    return result


# --------------------------------------------------------------------------- #
# Immutable schema records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class AttackAction:
    """The fixed per-episode attack action applied by one sender (D016).

    ``kind`` is ``"none"`` for every honest sender (and for a selected node
    running the ``kappa == 0`` no-op path) or ``"delta_reversal"`` for the
    single active compromised sender, with the numeric ``kappa`` retained.
    """

    kind: str
    kappa: float

    def __post_init__(self) -> None:
        if self.kind not in (ATTACK_NONE, ATTACK_DELTA_REVERSAL):
            raise EvaluatorError(f"unknown attack action kind {self.kind!r}")
        kappa = _require_finite_float(self.kappa, label="attack kappa")
        if kappa < 0.0:
            raise EvaluatorError(f"attack kappa must be non-negative, got {kappa!r}")
        if self.kind == ATTACK_NONE and kappa != 0.0:
            raise EvaluatorError(
                f"attack kind {ATTACK_NONE!r} requires kappa 0, got {kappa!r}"
            )
        if self.kind == ATTACK_DELTA_REVERSAL and kappa <= 0.0:
            raise EvaluatorError(
                f"attack kind {ATTACK_DELTA_REVERSAL!r} requires kappa > 0, "
                f"got {kappa!r}"
            )
        object.__setattr__(self, "kappa", kappa)


@dataclass(frozen=True, slots=True)
class PropagationState:
    """Immutable weight-aware poison provenance for one node (D018).

    ``reached`` is monotone across the episode (local training preserves prior
    influence). When reached, ``min_hop`` is the minimum number of positive
    aggregation edges the influence crossed from the seed (the compromised
    sender is hop 0) and ``first_reached_round`` is the earliest committed
    round at which this node became reached.
    """

    reached: bool
    min_hop: int | None
    first_reached_round: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.reached, bool):
            raise EvaluatorError("propagation reached must be bool")
        if self.reached:
            if self.min_hop is None or self.first_reached_round is None:
                raise EvaluatorError(
                    "reached propagation requires min_hop and first_reached_round"
                )
            hop = _require_int(self.min_hop, label="min_hop")
            first = _require_int(self.first_reached_round, label="first_reached_round")
            if hop < 0:
                raise EvaluatorError(f"min_hop must be non-negative, got {hop}")
            if first < 0:
                raise EvaluatorError(
                    f"first_reached_round must be non-negative, got {first}"
                )
            object.__setattr__(self, "min_hop", hop)
            object.__setattr__(self, "first_reached_round", first)
        else:
            if self.min_hop is not None or self.first_reached_round is not None:
                raise EvaluatorError(
                    "unreached propagation must not carry hop/first round"
                )


def _normalize_id_tuple(value: object, *, label: str, context: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise EvaluatorError(
            f"{label} must be a tuple of node ids{_suffix(context)}, got "
            f"{type(value).__name__}"
        )
    ids: list[int] = []
    seen: set[int] = set()
    for raw in value:
        node = _require_int(raw, label=label, context=context)
        if node in seen:
            raise EvaluatorError(f"duplicate {label} entry {node}{_suffix(context)}")
        seen.add(node)
        ids.append(node)
    return tuple(ids)


@dataclass(frozen=True, slots=True)
class NodeEvaluation:
    """Immutable per-node evaluator record for one committed round (R008).

    Carries only privileged post-commit truth and never crosses back into the
    decentralized call graph (D012).
    """

    node_id: int
    loss: float
    accuracy: float
    sample_count: int
    true_type: str
    attack: AttackAction
    participant_ids: tuple[int, ...]
    multipliers: tuple[float, ...]
    weights: tuple[float, ...]
    propagation: PropagationState

    def __post_init__(self) -> None:
        node = _require_int(self.node_id, label="node_id")
        if node < 0:
            raise EvaluatorError(f"node_id must be non-negative, got {node}")
        loss = _require_finite_float(self.loss, label="loss")
        if loss < 0.0:
            raise EvaluatorError(f"loss must be non-negative, got {loss}")
        accuracy = _require_finite_float(self.accuracy, label="accuracy")
        if not 0.0 <= accuracy <= 1.0:
            raise EvaluatorError(f"accuracy must be within [0, 1], got {accuracy}")
        samples = _require_int(self.sample_count, label="sample_count")
        if samples < 1:
            raise EvaluatorError(f"sample_count must be positive, got {samples}")
        if self.true_type not in (BENIGN, MALICIOUS):
            raise EvaluatorError(f"unknown true_type {self.true_type!r}")
        if not isinstance(self.attack, AttackAction):
            raise EvaluatorError("attack must be an AttackAction")
        if not isinstance(self.propagation, PropagationState):
            raise EvaluatorError("propagation must be a PropagationState")

        participants = _normalize_id_tuple(
            self.participant_ids, label="participant_ids", context=f"node={node}"
        )
        if not participants or participants[0] != node:
            raise EvaluatorError(
                f"participant_ids must begin with the node id ({node})"
            )
        multipliers = _validate_numeric_vector(
            self.multipliers, label="multiplier", context=f"node={node}"
        )
        weights = _validate_numeric_vector(
            self.weights, label="weight", context=f"node={node}"
        )
        if len(multipliers) != len(participants) or len(weights) != len(participants):
            raise EvaluatorError(
                f"profile vector length mismatch for node={node}: "
                f"participants={len(participants)} multipliers={len(multipliers)} "
                f"weights={len(weights)}"
            )
        if abs(math.fsum(weights) - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise EvaluatorError(
                f"weights must sum to 1 for node={node}, got {math.fsum(weights)!r}"
            )
        object.__setattr__(self, "node_id", node)
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "accuracy", accuracy)
        object.__setattr__(self, "sample_count", samples)
        object.__setattr__(self, "participant_ids", participants)
        object.__setattr__(self, "multipliers", multipliers)
        object.__setattr__(self, "weights", weights)


def _validate_numeric_vector(
    value: object, *, label: str, context: str
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise EvaluatorError(
            f"{label}s must be a tuple of reals{_suffix(context)}, got "
            f"{type(value).__name__}"
        )
    out: list[float] = []
    for raw in value:
        number = _require_finite_float(raw, label=label, context=context)
        if number < 0.0:
            raise EvaluatorError(
                f"{label} must be non-negative, got {number!r}{_suffix(context)}"
            )
        out.append(number)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class EvaluatedRound:
    """One canonical node-id-sorted evaluated-round record (R008/D018).

    ``nodes`` is strictly ascending and duplicate-free so the record serializes
    deterministically regardless of the evaluator's internal iteration order.
    """

    round_index: int
    nodes: tuple[NodeEvaluation, ...]

    def __post_init__(self) -> None:
        index = _require_int(self.round_index, label="round_index")
        if index < 0:
            raise EvaluatorError(f"round_index must be non-negative, got {index}")
        if not isinstance(self.nodes, tuple) or not self.nodes:
            raise EvaluatorError("nodes must be a non-empty tuple")
        previous: int | None = None
        for node in self.nodes:
            if not isinstance(node, NodeEvaluation):
                raise EvaluatorError("nodes must contain NodeEvaluation records")
            if previous is not None and node.node_id <= previous:
                raise EvaluatorError(
                    "nodes must be strictly ascending by node_id"
                )
            previous = node.node_id
        object.__setattr__(self, "round_index", index)

    @property
    def node_ids(self) -> tuple[int, ...]:
        """The evaluated node ids in canonical order."""

        return tuple(node.node_id for node in self.nodes)


# --------------------------------------------------------------------------- #
# Reference subset selection
# --------------------------------------------------------------------------- #
def select_reference_subset(
    dataset: Dataset,
    *,
    seed: int,
    max_samples: int,
) -> Subset:
    """Return a deterministic, seed-selected, bounded reference subset.

    The subset is the first ``min(max_samples, len(dataset))`` indices of a
    seeded permutation, re-sorted ascending so batching (and therefore the
    metric accumulation order) is independent of RNG state. An empty dataset or
    non-positive bound fails closed.
    """

    count = len(dataset)
    if count == 0:
        raise EvaluatorError("evaluation reference dataset is empty")
    bound = _require_int(max_samples, label="max_samples")
    if bound < 1:
        raise EvaluatorError(f"max_samples must be positive, got {bound}")
    selected = min(bound, count)
    rng = np.random.default_rng(int(seed) % (2**32))
    indices = np.sort(rng.permutation(count)[:selected])
    return Subset(dataset, [int(index) for index in indices])


# --------------------------------------------------------------------------- #
# Round structural validation (shared by the tracker and evaluator)
# --------------------------------------------------------------------------- #
def _expected_participants(
    topology: nx.Graph, node: int
) -> tuple[int, ...]:
    return (node,) + tuple(sorted(int(n) for n in topology.neighbors(node)))


def validate_round_structure(
    topology: nx.Graph, result: RoundResult
) -> dict[int, NodeWeightProfile]:
    """Validate a committed round against the topology and return its profiles.

    Enforces that committed states, selected profiles, and message summaries
    cover exactly the topology nodes and edges (graph-consistent senders) and
    that every profile carries a finite, non-negative, unit-sum, participant-
    aligned weight vector. Raises :class:`EvaluatorError` on any violation.
    """

    nodes = tuple(sorted(int(n) for n in topology.nodes()))
    node_set = set(nodes)
    state_nodes = {int(n) for n in result.states}
    if state_nodes != node_set:
        raise EvaluatorError(
            "committed node set does not match topology: "
            f"missing={sorted(node_set - state_nodes)} "
            f"extra={sorted(state_nodes - node_set)}"
        )

    profiles: dict[int, NodeWeightProfile] = {}
    for profile in result.profiles:
        if not isinstance(profile, NodeWeightProfile):
            raise EvaluatorError(
                f"profile must be a NodeWeightProfile, got "
                f"{type(profile).__name__}"
            )
        node = int(profile.node_id)
        if node in profiles:
            raise EvaluatorError(f"duplicate profile for node {node}")
        if node not in node_set:
            raise EvaluatorError(f"profile for unknown node {node}")
        expected = _expected_participants(topology, node)
        participants = tuple(int(p) for p in profile.participant_ids)
        if participants != expected:
            raise EvaluatorError(
                f"profile participants {participants} do not match topology "
                f"neighbors {expected} for node {node}"
            )
        multipliers = _validate_numeric_vector(
            profile.multipliers, label="multiplier", context=f"node={node}"
        )
        weights = _validate_numeric_vector(
            profile.weights, label="weight", context=f"node={node}"
        )
        if len(multipliers) != len(expected) or len(weights) != len(expected):
            raise EvaluatorError(
                f"profile vector length mismatch for node {node}: "
                f"expected {len(expected)}, multipliers={len(multipliers)}, "
                f"weights={len(weights)}"
            )
        total = math.fsum(weights)
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise EvaluatorError(
                f"profile weights must sum to 1 for node {node}, got {total!r}"
            )
        profiles[node] = profile

    if set(profiles) != node_set:
        raise EvaluatorError(
            "profiles do not cover the topology node set: "
            f"missing={sorted(node_set - set(profiles))}"
        )

    seen_senders: set[int] = set()
    for message in result.messages:
        sender = int(message.sender)
        if sender not in node_set:
            raise EvaluatorError(f"message from unknown sender {sender}")
        if sender in seen_senders:
            raise EvaluatorError(f"duplicate message summary for sender {sender}")
        seen_senders.add(sender)
        recipients = tuple(sorted(int(r) for r in message.recipients))
        expected_recipients = tuple(
            sorted(int(n) for n in topology.neighbors(sender))
        )
        if recipients != expected_recipients:
            raise EvaluatorError(
                f"message recipients {recipients} for sender {sender} do not "
                f"match topology neighbors {expected_recipients}"
            )

    return profiles


# --------------------------------------------------------------------------- #
# D018 stateful provenance tracker
# --------------------------------------------------------------------------- #
class ProvenanceTracker:
    """D018 stateful, weight-aware multi-hop poison provenance.

    Construct once per episode from the frozen topology and the resolved attack
    identity/intensity, then call :meth:`update` exactly once per committed
    round in strict round order. The tracker is pure with respect to the round
    it consumes: it validates the committed round shape (node sets, profiles,
    graph-consistent sender edges, finite unit-sum weights) and computes the
    new reach state into a fresh mapping, committing it only after the whole
    round validates, so a malformed round leaves prior state untouched.

    Semantics:

    * An active (``kappa > 0``) compromised sender is a seed at hop 0 from the
      first recorded round onward.
    * Local training preserves prior influence: a node that was reached in a
      previous round remains reached (monotone), keeping its minimum hop.
    * Only a participant with a strictly positive selected weight propagates a
      sender's influence into a committed recipient; a zero weight never does.
    * ``min_hop`` is the minimum number of positive-weight edges from the seed;
      ``first_reached_round`` is the earliest round the node became reached.
    """

    def __init__(
        self,
        *,
        topology: nx.Graph,
        compromised_node: int | None,
        kappa: float,
    ) -> None:
        self._nodes = tuple(sorted(int(n) for n in topology.nodes()))
        node_set = set(self._nodes)
        if compromised_node is not None:
            attacker = _require_int(compromised_node, label="compromised_node")
            if attacker not in node_set:
                raise EvaluatorError(
                    f"compromised_node {attacker} is not in topology nodes "
                    f"{list(self._nodes)}"
                )
        else:
            attacker = None
        intensity = _require_finite_float(kappa, label="kappa")
        if intensity < 0.0:
            raise EvaluatorError(f"kappa must be non-negative, got {intensity!r}")

        self._topology = topology
        self._attacker = attacker
        self._kappa = intensity
        self._active = attacker is not None and intensity > 0.0
        self._state: dict[int, PropagationState] = {
            node: PropagationState(reached=False, min_hop=None, first_reached_round=None)
            for node in self._nodes
        }
        self._next_round = 0

    @classmethod
    def from_experiment(
        cls, config: ExperimentConfig, topology: nx.Graph
    ) -> "ProvenanceTracker":
        """Build a tracker from a resolved config and its frozen topology."""

        return cls(
            topology=topology,
            compromised_node=config.attack.compromised_node,
            kappa=config.attack.kappa,
        )

    @property
    def compromised_node(self) -> int | None:
        """The single compromised node id, or ``None`` for an honest episode."""

        return self._attacker

    @property
    def kappa(self) -> float:
        """The resolved D016 intensity."""

        return self._kappa

    @property
    def active(self) -> bool:
        """Whether an active seed exists (compromised node present and kappa > 0)."""

        return self._active

    @property
    def next_round_index(self) -> int:
        """The next round index this tracker expects to consume."""

        return self._next_round

    def state_for(self, node: int) -> PropagationState:
        """Return the current committed propagation state for ``node``."""

        try:
            return self._state[int(node)]
        except KeyError as exc:
            raise EvaluatorError(f"unknown node {node}") from exc

    def propagation(self) -> tuple[PropagationState, ...]:
        """Return the per-node committed states in canonical node-id order."""

        return tuple(self._state[node] for node in self._nodes)

    def update(self, result: RoundResult) -> tuple[PropagationState, ...]:
        """Consume one committed round and return the new per-node states.

        Raises :class:`EvaluatorError` (leaving prior state unchanged) when the
        round is out of order, graph-inconsistent, or malformed.
        """

        index = _require_int(result.round_index, label="round_index")
        if index != self._next_round:
            raise EvaluatorError(
                f"provenance round out of order: expected {self._next_round}, "
                f"got {index}"
            )
        if tuple(sorted(int(n) for n in result.states)) != self._nodes:
            raise EvaluatorError(
                "provenance round node set does not match topology"
            )
        profiles = validate_round_structure(self._topology, result)

        # Outgoing influence and hop are derived from the prior committed state
        # (local training preserves it); the active attacker is always a hop-0
        # seed regardless of prior state.
        outgoing_reach = {node: self._state[node].reached for node in self._nodes}
        outgoing_hop = {node: self._state[node].min_hop for node in self._nodes}
        if self._active and self._attacker is not None:
            outgoing_reach[self._attacker] = True
            outgoing_hop[self._attacker] = 0

        new_state: dict[int, PropagationState] = {}
        for node in self._nodes:
            prior = self._state[node]
            candidates: list[int] = []
            if self._active and node == self._attacker:
                candidates.append(0)
            if prior.reached and prior.min_hop is not None:
                candidates.append(prior.min_hop)
            profile = profiles[node]
            for sender, weight in zip(
                tuple(int(p) for p in profile.participant_ids[1:]),
                profile.weights[1:],
            ):
                if float(weight) <= 0.0:
                    continue
                if outgoing_reach[sender] and outgoing_hop[sender] is not None:
                    candidates.append(outgoing_hop[sender] + 1)
            if candidates:
                first = (
                    prior.first_reached_round
                    if prior.reached and prior.first_reached_round is not None
                    else index
                )
                new_state[node] = PropagationState(
                    reached=True,
                    min_hop=min(candidates),
                    first_reached_round=first,
                )
            else:
                new_state[node] = PropagationState(
                    reached=False, min_hop=None, first_reached_round=None
                )

        self._state = new_state
        self._next_round = index + 1
        return self.propagation()


# --------------------------------------------------------------------------- #
# Privileged evaluator
# --------------------------------------------------------------------------- #
def _default_model_factory(config: ExperimentConfig) -> Callable[[], nn.Module]:
    model_name = config.model.name

    def factory() -> nn.Module:
        return build_model(model_name, seed=0)

    return factory


class PrivilegedEvaluator:
    """Post-commit evaluator over a shared privileged reference subset (R008).

    Construct once per episode from a resolved config, the frozen topology, a
    privileged global reference ``dataset``, and the episode's resolved seeds.
    ``model_factory`` is an optional seam (defaults to the configured model)
    that lets tests evaluate a tiny architecture without weakening the
    production path.

    :meth:`evaluate_round` must be called once per committed
    :class:`~dfl_sim.simulator.RoundResult` in strict order. It validates the
    round structure, scores every committed node model on the *same* reference
    subset, advances the internal D018 provenance tracker, and returns one
    immutable :class:`EvaluatedRound`. Any failure raises
    :class:`EvaluatorError` and produces no partial record; the tracker is only
    advanced after all metrics succeed.
    """

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        topology: nx.Graph,
        reference_dataset: Dataset,
        seeds: ResolvedSeeds,
        model_factory: Callable[[], nn.Module] | None = None,
    ) -> None:
        self._config = config
        self._topology = topology
        self._nodes = tuple(sorted(int(n) for n in topology.nodes()))
        self._batch_size = int(config.evaluation.batch_size)
        self._model_factory = model_factory or _default_model_factory(config)
        reference_seed = seeds.derive(_REFERENCE_SEED_LABEL)
        self._reference = select_reference_subset(
            reference_dataset,
            seed=reference_seed,
            max_samples=config.evaluation.max_samples,
        )
        self._tracker = ProvenanceTracker.from_experiment(config, topology)
        self._history: list[EvaluatedRound] = []

    @classmethod
    def from_experiment(
        cls,
        config: ExperimentConfig,
        topology: nx.Graph,
        reference_dataset: Dataset,
        seeds: ResolvedSeeds,
        *,
        model_factory: Callable[[], nn.Module] | None = None,
    ) -> "PrivilegedEvaluator":
        """Build an evaluator for a resolved episode configuration."""

        return cls(
            config=config,
            topology=topology,
            reference_dataset=reference_dataset,
            seeds=seeds,
            model_factory=model_factory,
        )

    # -- read-only surface -------------------------------------------------- #
    @property
    def node_ids(self) -> tuple[int, ...]:
        """Topology node ids in canonical order."""

        return self._nodes

    @property
    def reference_sample_count(self) -> int:
        """Number of reference examples scored per node per round."""

        return len(self._reference)

    @property
    def tracker(self) -> ProvenanceTracker:
        """The internal D018 provenance tracker (read-only diagnostics)."""

        return self._tracker

    @property
    def history(self) -> tuple[EvaluatedRound, ...]:
        """All evaluated rounds emitted so far, in execution order (diagnostics)."""

        return tuple(self._history)

    @property
    def next_round_index(self) -> int:
        """The next round index this evaluator expects to consume."""

        return self._tracker.next_round_index

    # -- evaluation --------------------------------------------------------- #
    def evaluate_round(self, result: RoundResult) -> EvaluatedRound:
        """Evaluate one committed round and return its canonical record."""

        index = _require_int(result.round_index, label="round_index")
        if index != self.next_round_index:
            raise EvaluatorError(
                f"evaluator round out of order: expected {self.next_round_index}, "
                f"got {index}"
            )
        profiles = validate_round_structure(self._topology, result)

        metrics: list[tuple[int, float, float, int]] = []
        for node in self._nodes:
            if node not in result.states:
                raise EvaluatorError(
                    f"committed round is missing node {node} at round {index}"
                )
            loss, accuracy, samples = self._evaluate_state(
                result.states[node], node=node, round_index=index
            )
            metrics.append((node, loss, accuracy, samples))

        # Only after every metric succeeded do we advance provenance; a failed
        # round therefore leaves the tracker untouched.
        propagation = self._tracker.update(result)
        propagation_by_node = {
            node: propagation[position] for position, node in enumerate(self._nodes)
        }

        records = tuple(
            NodeEvaluation(
                node_id=node,
                loss=loss,
                accuracy=accuracy,
                sample_count=samples,
                true_type=self._true_type(node),
                attack=self._attack_action(node),
                participant_ids=tuple(int(p) for p in profiles[node].participant_ids),
                multipliers=tuple(float(x) for x in profiles[node].multipliers),
                weights=tuple(float(x) for x in profiles[node].weights),
                propagation=propagation_by_node[node],
            )
            for node, loss, accuracy, samples in metrics
        )
        evaluated = EvaluatedRound(round_index=index, nodes=records)
        self._history.append(evaluated)
        return evaluated

    # -- internals ---------------------------------------------------------- #
    def _true_type(self, node: int) -> str:
        if (
            self._tracker.compromised_node is not None
            and node == self._tracker.compromised_node
        ):
            return MALICIOUS
        return BENIGN

    def _attack_action(self, node: int) -> AttackAction:
        if (
            self._tracker.compromised_node is not None
            and node == self._tracker.compromised_node
            and self._tracker.kappa > 0.0
        ):
            return AttackAction(kind=ATTACK_DELTA_REVERSAL, kappa=self._tracker.kappa)
        return AttackAction(kind=ATTACK_NONE, kappa=0.0)

    def _evaluate_state(
        self, state: Mapping[str, torch.Tensor], *, node: int, round_index: int
    ) -> tuple[float, float, int]:
        """Score one committed state on the shared reference subset.

        Returns ``(mean_loss, accuracy, sample_count)`` with exact accounting.
        """

        context = f"evaluation node={node} round={round_index}"
        try:
            validated = validate_state_dict(
                state, context=f"committed state {context}"
            )
        except AggregationError as exc:
            raise EvaluatorError(
                f"invalid committed model state ({context}): {exc}"
            ) from exc

        model = self._model_factory()
        try:
            load_state_into_model(
                model, validated, context=f"committed state {context}"
            )
        except StateDictError as exc:
            raise EvaluatorError(
                f"could not load committed model state ({context}): {exc}"
            ) from exc

        model.eval()
        loader = DataLoader(
            self._reference,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=0,
        )
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        correct = 0
        count = 0
        try:
            with torch.no_grad():
                for inputs, targets in loader:
                    if not isinstance(inputs, torch.Tensor):
                        raise EvaluatorError(
                            f"evaluation input must be a tensor ({context}), got "
                            f"{type(inputs).__name__}"
                        )
                    if not bool(torch.isfinite(inputs).all()):
                        raise EvaluatorError(
                            f"non-finite evaluation input ({context})"
                        )
                    logits = model(inputs)
                    if not isinstance(logits, torch.Tensor) or logits.dim() != 2:
                        raise EvaluatorError(
                            f"model must return 2-D logits ({context})"
                        )
                    batch = int(logits.shape[0])
                    if batch == 0:
                        continue
                    self._validate_targets(
                        targets,
                        batch=batch,
                        num_classes=int(logits.shape[1]),
                        context=context,
                    )
                    loss = criterion(logits, targets)
                    if not bool(torch.isfinite(loss)):
                        raise EvaluatorError(
                            f"non-finite evaluation loss ({context})"
                        )
                    total_loss += float(loss)
                    count += batch
                    correct += int(
                        (logits.argmax(dim=1) == targets).sum().item()
                    )
        except EvaluatorError:
            raise
        except Exception as exc:
            raise EvaluatorError(
                f"evaluation failed ({context}): {exc}"
            ) from exc

        expected = len(self._reference)
        if count != expected:
            raise EvaluatorError(
                f"evaluation sample accounting mismatch ({context}): "
                f"expected {expected}, counted {count}"
            )
        mean_loss = total_loss / count
        accuracy = correct / count
        if not math.isfinite(mean_loss) or not math.isfinite(accuracy):
            raise EvaluatorError(f"non-finite evaluation metric ({context})")
        return mean_loss, accuracy, count

    @staticmethod
    def _validate_targets(
        targets: object,
        *,
        batch: int,
        num_classes: int,
        context: str,
    ) -> None:
        if not isinstance(targets, torch.Tensor):
            raise EvaluatorError(
                f"evaluation target must be a tensor ({context}), got "
                f"{type(targets).__name__}"
            )
        if targets.dim() != 1 or int(targets.shape[0]) != batch:
            raise EvaluatorError(
                f"evaluation target shape {tuple(targets.shape)} does not match "
                f"batch {batch} ({context})"
            )
        if torch.is_floating_point(targets) or torch.is_complex(targets):
            raise EvaluatorError(
                f"evaluation target must be an integer class index ({context}), "
                f"got {targets.dtype}"
            )
        if num_classes < 1:
            raise EvaluatorError(f"model produced no classes ({context})")
        if int(targets.min()) < 0 or int(targets.max()) >= num_classes:
            raise EvaluatorError(
                f"evaluation target out of range [0, {num_classes}) ({context})"
            )


def build_evaluator(
    config: ExperimentConfig,
    reference_dataset: Dataset,
    seeds: ResolvedSeeds,
    *,
    topology: nx.Graph | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
) -> PrivilegedEvaluator:
    """Convenience constructor resolving topology from ``config`` when omitted."""

    resolved_topology = topology if topology is not None else load_topology(config)
    return PrivilegedEvaluator.from_experiment(
        config,
        resolved_topology,
        reference_dataset,
        seeds,
        model_factory=model_factory,
    )
