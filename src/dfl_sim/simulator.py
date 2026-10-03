"""Snapshot-isolated synchronous DFL round engine (R001, R012, D015).

:func:`run_round` implements R001's exact synchronous semantics as distinct,
independently testable phases:

1. clone every round-start model,
2. local-train every node from its own snapshot,
3. compute each sender's canonical outgoing model exactly once,
4. build and deliver cloned messages **only** across configured graph edges,
5. gather and validate every inbox (all receives complete),
6. build one local observation per node from its own round-start/trained
   models and sorted inbox, select a weight profile through the injected
   policy, and weighted-average self plus inbox into a staged map,
7. validate key/shape/dtype/finiteness invariants, and
8. commit the complete next-state map in one operation.

No node observes another node's partially committed state: aggregation reads
only round-start snapshots and delivered messages, and the next states are
published together as a :class:`RoundResult`. Caller snapshots, local-trained
states, and messages are never mutated; every phase output is a detached clone.

The attack step is injected through a narrow callable
(:class:`~dfl_sim.attacker.FixedAttackController`, matching
:class:`~dfl_sim.attacker.OutgoingController`) that sees only one sender's own
round-start and trained state. The aggregation step is split into two narrow
seams: an :class:`~dfl_sim.observation.ObservationBuilder` that receives only
one node's own state and sorted inbox, and an
:class:`~dfl_sim.policy.AggregationPolicy` whose sole argument is the frozen
:class:`~dfl_sim.observation.LocalObservation`. Neither seam observes the
topology, attack config/compromised node, other inboxes, global results, or an
evaluator object. This module writes no files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, TypeAlias

import networkx as nx
import torch
from torch.utils.data import Dataset

from .aggregation import (
    AggregationError,
    ModelState,
    aggregate_weighted_neighborhood,
    validate_state_dict,
)
from .attacker import FixedAttackController, OutgoingController
from .config import ExperimentConfig
from .models import build_model, load_state_into_model
from .node import (
    LocalTrainResult,
    evaluate_cross_entropy,
    local_train_from_snapshot,
)
from .observation import (
    DefaultObservationBuilder,
    LocalObservation,
    LossEvaluator,
    ObservationBuilder,
)
from .policy import (
    AggregationPolicy,
    WeightProfile,
    build_policy,
)
from .reproducibility import ResolvedSeeds

__all__ = [
    "LossEvaluatorFactory",
    "MessageSummary",
    "NodeRoundStat",
    "NodeWeightProfile",
    "RoundError",
    "RoundMessage",
    "RoundResult",
    "Trainer",
    "build_messages",
    "default_loss_evaluator_factory",
    "deliver_to_inboxes",
    "run_round",
    "validate_node_set",
]

#: Local-training seam. Defaults to the production
#: :func:`~dfl_sim.node.local_train_from_snapshot`; tests inject stubs.
Trainer: TypeAlias = Callable[..., LocalTrainResult]

#: Node-local loss-evaluator seam: given the resolved config, the node's own
#: partition, and node/round context, build the callable passed to the
#: observation builder. Tests inject a deterministic stub so scalar protocol
#: fixtures never touch a real dataset.
LossEvaluatorFactory: TypeAlias = Callable[
    [ExperimentConfig, Dataset, int, int], LossEvaluator
]


def default_loss_evaluator_factory(
    config: ExperimentConfig,
    partition: Dataset,
    node_id: int,
    round_index: int,
) -> LossEvaluator:
    """Build the production per-node loss evaluator over ``partition``.

    The returned callable loads a validated clone of the state into a fresh
    deterministic model and evaluates mean cross-entropy on the node's own
    partition. It receives only one node's state, so local loss evidence never
    crosses the information boundary.
    """

    model_name = config.model.name
    batch_size = int(config.training.batch_size)

    def evaluate(
        state: Mapping[str, torch.Tensor],
        *,
        node_id: int = node_id,
        round_index: int = round_index,
        context: str = "",
    ) -> float:
        model = build_model(model_name, seed=0)
        load_state_into_model(model, state, context=context)
        return evaluate_cross_entropy(
            model, partition, batch_size=batch_size, context=context
        )

    return evaluate


class RoundError(AggregationError):
    """Raised when a round cannot proceed (node set/topology/message issues).

    Subclasses :class:`~dfl_sim.aggregation.AggregationError` so a caller that
    guards the model-payload boundary also catches round-level contract
    violations.
    """


@dataclass(frozen=True, slots=True)
class RoundMessage:
    """One full-model message delivered across one graph edge.

    ``state`` holds detached cloned tensors produced at the delivery boundary;
    it shares no storage with the sender's trained model or with any sibling
    message's state.
    """

    round_index: int
    sender: int
    recipient: int
    state: ModelState


@dataclass(frozen=True, slots=True)
class MessageSummary:
    """Summary of one sender's outgoing messages (sorted recipients)."""

    sender: int
    recipients: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NodeRoundStat:
    """Per-node round statistics needed by the canonical round log (T04)."""

    node_id: int
    pre_loss: float
    post_loss: float
    sample_count: int
    neighbors: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NodeWeightProfile:
    """Immutable selected weight profile for one node in one round (S04 seam).

    Mirrors the action half of :class:`~dfl_sim.policy.WeightProfile` without
    importing any policy internals into the round log: ``participant_ids`` is
    self first then delivered senders ascending, with the aligned finite
    multipliers and normalized weights. ``RoundResult`` exposes these sorted by
    node id; the minimal JSONL writer intentionally ignores them.
    """

    node_id: int
    participant_ids: tuple[int, ...]
    multipliers: tuple[float, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RoundResult:
    """Immutable outcome of one committed synchronous round.

    ``states`` maps every node to its committed next-round state (detached
    clones); ``stats`` and ``profiles`` are sorted by node id and ``messages``
    by sender.
    """

    round_index: int
    states: dict[int, ModelState]
    stats: tuple[NodeRoundStat, ...]
    messages: tuple[MessageSummary, ...]
    profiles: tuple[NodeWeightProfile, ...] = ()


def _topology_nodes(topology: nx.Graph) -> tuple[int, ...]:
    return tuple(sorted(int(node) for node in topology.nodes()))


def _require_node_set(
    topology: nx.Graph,
    states: Mapping[int, object],
    *,
    label: str,
) -> tuple[int, ...]:
    topo_nodes = set(_topology_nodes(topology))
    state_nodes = {int(node) for node in states}
    if state_nodes != topo_nodes:
        raise RoundError(
            f"{label} node set does not match topology: "
            f"missing={sorted(topo_nodes - state_nodes)} "
            f"extra={sorted(state_nodes - topo_nodes)}"
        )
    return tuple(sorted(topo_nodes))


def validate_node_set(
    topology: nx.Graph,
    snapshots: Mapping[int, object],
    partitions: Mapping[int, object],
) -> tuple[int, ...]:
    """Validate that snapshots and partitions cover exactly the topology nodes."""

    nodes = _require_node_set(topology, snapshots, label="snapshot")
    _require_node_set(topology, partitions, label="partition")
    return nodes


def _sorted_edges(topology: nx.Graph) -> list[tuple[int, int]]:
    edges = [tuple(sorted((int(u), int(v)))) for u, v in topology.edges()]
    return sorted(edge for edge in edges if edge[0] != edge[1])


def build_messages(
    outgoing_states: Mapping[int, Mapping[str, torch.Tensor]],
    topology: nx.Graph,
    *,
    round_index: int,
) -> tuple[RoundMessage, ...]:
    """Build one message per sender per graph edge (both directions).

    ``outgoing_states`` holds each sender's already-computed canonical model
    (see :class:`~dfl_sim.attacker.OutgoingController`); this function never
    recomputes a transform. Each message carries a validated, detached clone of
    the sender's canonical model, so equal values fan out identically to every
    neighbor while each recipient tensor is independently stored.
    """

    _require_node_set(topology, outgoing_states, label="outgoing state")
    index = int(round_index)
    messages: list[RoundMessage] = []
    for u, v in _sorted_edges(topology):
        for sender, recipient in ((u, v), (v, u)):
            state = validate_state_dict(
                outgoing_states[sender],
                context=f"outgoing message {sender}->{recipient}",
            )
            messages.append(
                RoundMessage(
                    round_index=index,
                    sender=sender,
                    recipient=recipient,
                    state=state,
                )
            )
    messages.sort(key=lambda message: (message.sender, message.recipient))
    return tuple(messages)


def deliver_to_inboxes(
    messages: Sequence[RoundMessage],
    topology: nx.Graph,
    *,
    round_index: int,
) -> dict[int, tuple[RoundMessage, ...]]:
    """Validate and group messages into per-recipient inboxes.

    Enforces the protocol contract before any arithmetic: every message must
    target a known node, originate from a known node, travel along a configured
    edge, carry the expected round index, and be unique per sender. Every
    expected neighbor message must be present -- this is the synchronous
    receive barrier. Delivered message states are re-validated and cloned at
    the boundary.
    """

    nodes = _topology_nodes(topology)
    node_set = set(nodes)
    expected = {
        node: {int(neighbor) for neighbor in topology.neighbors(node)}
        for node in nodes
    }
    index = int(round_index)

    inboxes: dict[int, list[RoundMessage]] = {node: [] for node in nodes}
    seen: dict[int, set[int]] = {node: set() for node in nodes}

    for message in messages:
        sender = int(message.sender)
        recipient = int(message.recipient)
        if int(message.round_index) != index:
            raise RoundError(
                f"round mismatch: message {sender}->{recipient} has round "
                f"{message.round_index}, expected {index}"
            )
        if recipient not in node_set:
            raise RoundError(f"unknown recipient node {recipient}")
        if sender not in node_set:
            raise RoundError(f"unknown sender node {sender}")
        if sender not in expected[recipient]:
            raise RoundError(
                f"non-edge message {sender}->{recipient}: not a configured edge"
            )
        if sender in seen[recipient]:
            raise RoundError(f"duplicate message {sender}->{recipient}")

        state = validate_state_dict(
            message.state,
            context=f"delivered message {sender}->{recipient}",
        )
        seen[recipient].add(sender)
        inboxes[recipient].append(
            RoundMessage(
                round_index=index,
                sender=sender,
                recipient=recipient,
                state=state,
            )
        )

    for node in nodes:
        missing = expected[node] - seen[node]
        if missing:
            raise RoundError(
                f"missing expected message(s) for node {node} from "
                f"{sorted(missing)}"
            )

    return {
        node: tuple(sorted(inboxes[node], key=lambda message: message.sender))
        for node in nodes
    }


def _summarize_messages(
    messages: Sequence[RoundMessage],
) -> tuple[MessageSummary, ...]:
    grouped: dict[int, list[int]] = {}
    for message in messages:
        grouped.setdefault(int(message.sender), []).append(int(message.recipient))
    return tuple(
        MessageSummary(sender=sender, recipients=tuple(sorted(recipients)))
        for sender, recipients in sorted(grouped.items())
    )


def run_round(
    *,
    config: ExperimentConfig,
    topology: nx.Graph,
    snapshots: Mapping[int, Mapping[str, torch.Tensor]],
    partitions: Mapping[int, Dataset],
    round_index: int,
    seeds: ResolvedSeeds,
    trainer: Trainer | None = None,
    policy: AggregationPolicy | None = None,
    observer: ObservationBuilder | None = None,
    loss_evaluator: LossEvaluatorFactory | None = None,
    controller: OutgoingController | None = None,
) -> RoundResult:
    """Execute one snapshot-isolated synchronous round and commit atomically.

    Every node trains locally from its own round-start snapshot; each sender's
    canonical outgoing model is computed exactly once through ``controller``
    (defaulting to :meth:`FixedAttackController.from_experiment` on ``config``);
    messages are delivered only across configured edges; and, after the
    receive barrier, each node builds a local observation from its own
    round-start/trained models and sorted inbox, selects a participant-aligned
    weight profile through ``policy``, and weighted-averages self plus inbox.
    The complete next-state map is returned in one operation.

    ``policy`` defaults to the config-selected baseline and ``observer`` to the
    production :class:`~dfl_sim.observation.DefaultObservationBuilder`. Neither
    seam sees the topology, attack config, other inboxes, global results, or an
    evaluator object. On any failure (raised/malformed controller, observer, or
    policy output, or a weighted-aggregation error) the round aborts before
    commit and no caller tensor is mutated; the returned :class:`RoundResult`
    additionally carries the sorted selected profiles for S04.
    """

    train = trainer or local_train_from_snapshot
    active_policy = policy if policy is not None else build_policy(config.aggregation)
    active_observer = (
        observer if observer is not None else DefaultObservationBuilder()
    )
    make_evaluator = loss_evaluator or default_loss_evaluator_factory
    outgoing_controller = (
        controller
        if controller is not None
        else FixedAttackController.from_experiment(config)
    )
    index = int(round_index)

    nodes = validate_node_set(topology, snapshots, partitions)

    # Phase 1 -- clone every round-start model so no node can observe or mutate
    # another node's snapshot.
    round_start: dict[int, ModelState] = {
        node: validate_state_dict(
            snapshots[node], context=f"round-start node={node} round={index}"
        )
        for node in nodes
    }

    # Phase 2 -- independent local training from each node's own snapshot.
    trained: dict[int, ModelState] = {}
    stats: dict[int, NodeRoundStat] = {}
    for node in nodes:
        result = train(
            round_start[node],
            partitions[node],
            config=config,
            node_id=node,
            round_index=index,
            train_seed=seeds.for_local_train(index, node),
            data_order_seed=seeds.for_data_order(index, node),
        )
        if int(result.node_id) != node or int(result.round_index) != index:
            raise RoundError(
                f"trainer returned mismatched identity for node={node} "
                f"round={index}: node_id={result.node_id} "
                f"round_index={result.round_index}"
            )
        trained[node] = validate_state_dict(
            result.state,
            reference=round_start[node],
            context=f"trained node={node} round={index}",
        )
        stats[node] = NodeRoundStat(
            node_id=node,
            pre_loss=float(result.pre_loss),
            post_loss=float(result.post_loss),
            sample_count=int(result.sample_count),
            neighbors=tuple(
                sorted(int(neighbor) for neighbor in topology.neighbors(node))
            ),
        )

    # Phase 3 -- compute each sender's canonical outgoing model exactly once
    # through the injected controller. The output is validated against the
    # sender's trained signature at the phase boundary, so a raised or
    # malformed controller aborts before any message or aggregation work.
    outgoing: dict[int, ModelState] = {}
    for node in nodes:
        candidate = outgoing_controller.outgoing_state(
            sender=node,
            round_start=round_start[node],
            trained=trained[node],
            round_index=index,
            context=f"node={node} round={index}",
        )
        if not isinstance(candidate, Mapping):
            raise RoundError(
                f"controller returned a non-mapping outgoing state for "
                f"node={node} round={index}: {type(candidate).__name__}"
            )
        outgoing[node] = validate_state_dict(
            candidate,
            reference=trained[node],
            context=f"outgoing node={node} round={index}",
        )

    # Phase 4 -- one RoundMessage per directed edge, each cloned from the
    # sender's already-computed canonical outgoing model. The transform is
    # never recomputed per recipient: equal values fan out to every neighbor
    # while each recipient tensor has independent storage.
    messages = build_messages(outgoing, topology, round_index=index)

    # Phase 5 -- synchronous receive barrier: validate every inbox completely
    # before any aggregation happens.
    inboxes = deliver_to_inboxes(messages, topology, round_index=index)

    # Phase 6 -- synchronous observe -> policy -> weighted aggregate. Every
    # inbox has already passed the receive barrier, so each node now builds
    # exactly one local observation from its own round-start/trained models and
    # sorted delivered senders, selects one participant-aligned weight profile,
    # and stages a weighted next state. The observer and policy each see only
    # one node's local data; the profile is validated to cover exactly self
    # plus the delivered senders in canonical order before any arithmetic.
    staged: dict[int, ModelState] = {}
    profiles: dict[int, NodeWeightProfile] = {}
    for node in nodes:
        inbox = inboxes[node]
        sender_states = tuple(
            (int(message.sender), message.state) for message in inbox
        )
        expected_participants = (node,) + tuple(
            int(message.sender) for message in inbox
        )

        evaluator = make_evaluator(config, partitions[node], node, index)
        try:
            observation = active_observer.build_observation(
                node_id=node,
                round_index=index,
                round_start=round_start[node],
                trained=trained[node],
                sender_states=sender_states,
                evaluator=evaluator,
            )
        except RoundError:
            raise
        except Exception as exc:
            raise RoundError(
                f"observation failed for node={node} round={index}: {exc}"
            ) from exc
        if not isinstance(observation, LocalObservation):
            raise RoundError(
                f"observer returned {type(observation).__name__} instead of a "
                f"LocalObservation for node={node} round={index}"
            )
        if int(observation.node_id) != node or int(observation.round_index) != index:
            raise RoundError(
                f"observer returned mismatched identity for node={node} "
                f"round={index}: node_id={observation.node_id} "
                f"round_index={observation.round_index}"
            )
        if tuple(observation.participant_ids) != expected_participants:
            raise RoundError(
                f"observation participants {tuple(observation.participant_ids)} "
                f"do not match delivered senders {expected_participants} for "
                f"node={node} round={index}"
            )

        try:
            profile = active_policy.select_weights(observation)
        except RoundError:
            raise
        except Exception as exc:
            raise RoundError(
                f"policy failed for node={node} round={index}: {exc}"
            ) from exc
        if not isinstance(profile, WeightProfile):
            raise RoundError(
                f"policy returned {type(profile).__name__} instead of a "
                f"WeightProfile for node={node} round={index}"
            )
        if tuple(profile.participant_ids) != expected_participants:
            raise RoundError(
                f"policy participants {tuple(profile.participant_ids)} do not "
                f"match delivered senders {expected_participants} for "
                f"node={node} round={index}"
            )

        neighbor_states = [message.state for message in inbox]
        staged[node] = aggregate_weighted_neighborhood(
            trained[node],
            neighbor_states,
            profile.weights,
            node_id=node,
            round_index=index,
            context=f"node={node} round={index}",
        )
        profiles[node] = NodeWeightProfile(
            node_id=node,
            participant_ids=tuple(profile.participant_ids),
            multipliers=tuple(profile.multipliers),
            weights=tuple(profile.weights),
        )

    # Phase 7 -- validate staged invariants against each node's trained model.
    committed: dict[int, ModelState] = {}
    for node in nodes:
        committed[node] = validate_state_dict(
            staged[node],
            reference=trained[node],
            context=f"next-state node={node} round={index}",
        )

    # Phase 8 -- single simultaneous commit.
    return RoundResult(
        round_index=index,
        states=committed,
        stats=tuple(stats[node] for node in nodes),
        messages=_summarize_messages(messages),
        profiles=tuple(profiles[node] for node in nodes),
    )
