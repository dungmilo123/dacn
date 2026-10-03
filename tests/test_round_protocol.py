"""Protocol contract tests for T03: the snapshot-isolated synchronous round engine.

All tests are offline and deterministic. They use stub trainers with scalar or
tiny state dicts to isolate protocol semantics from model/data work, plus one
real small-CNN integration check. Covers R001's snapshot/train/exchange/
aggregate/simultaneous-commit kernel, R012's no-aliasing evidence, and the Q7
negative surface (wrong node sets, non-edge/duplicate/missing messages, key/
shape/dtype mismatches, and non-finite payloads).
"""

from __future__ import annotations

import pytest
import torch

import networkx as nx

from dfl_sim import (
    DEFAULT_EPSILON,
    AggregationError,
    AggregationPolicy,
    AttackConfig,
    AttackError,
    DefaultObservationBuilder,
    FixedAttackController,
    LocalObservation,
    LocalTrainResult,
    MessageSummary,
    NeighborObservation,
    NodeWeightProfile,
    ObservationError,
    OutgoingController,
    PlainAveragePolicy,
    PolicyError,
    RoundError,
    RoundMessage,
    RoundResult,
    TrainingError,
    WeightProfile,
    aggregate_closed_neighborhood,
    aggregate_weighted_neighborhood,
    build_messages,
    build_model,
    build_weight_profile,
    clone_state_dict,
    deliver_to_inboxes,
    load_topology,
    partition_dataset,
    resolve_seeds,
    run_from_config,
    run_round,
    uniform_average,
    validate_state_dict,
)
from helpers import (
    MNISTShapedDataset,
    constant_loss_evaluator_factory,
    make_config,
)

NODES3 = (0, 1, 2)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _config(kind: str = "chain3", **overrides):
    return make_config(topology={"kind": kind}, **overrides)


def _zero_snapshots(nodes):
    return {node: {"w": torch.tensor([0.0])} for node in nodes}


def _stub_trainer(value_fn):
    """Stub trainer: trained state = snapshot + value_fn(node_id)."""

    def trainer(
        snapshot,
        dataset,
        *,
        config,
        node_id,
        round_index,
        train_seed,
        data_order_seed,
    ):
        state = {
            key: value.detach().clone() + float(value_fn(node_id))
            for key, value in snapshot.items()
        }
        return LocalTrainResult(
            node_id=node_id,
            round_index=round_index,
            state=state,
            pre_loss=1.0 + float(node_id),
            post_loss=0.5 + float(node_id),
            sample_count=4,
        )

    return trainer


def _run(
    nodes,
    topology,
    trainer,
    *,
    config=None,
    policy=None,
    observer=None,
    loss_evaluator=None,
    controller=None,
    partitions=None,
    snapshots=None,
    round_index=0,
):
    config = config or _config()
    seeds = resolve_seeds(config)
    if snapshots is None:
        snapshots = _zero_snapshots(nodes)
    if partitions is None:
        partitions = {node: None for node in nodes}
    if loss_evaluator is None:
        loss_evaluator = constant_loss_evaluator_factory
    return run_round(
        config=config,
        topology=topology,
        snapshots=snapshots,
        partitions=partitions,
        round_index=round_index,
        seeds=seeds,
        trainer=trainer,
        policy=policy,
        observer=observer,
        loss_evaluator=loss_evaluator,
        controller=controller,
    )


def _valid_messages(topology, round_index=0):
    trained = {
        node: {"w": torch.tensor([float(node)])} for node in topology.nodes()
    }
    return build_messages(trained, topology, round_index=round_index)


# --------------------------------------------------------------------------- #
# validate_state_dict: boundary contract (Q3/Q7)
# --------------------------------------------------------------------------- #
def test_validate_state_dict_returns_detached_clone() -> None:
    source = {"w": torch.tensor([1.0, 2.0])}
    validated = validate_state_dict(source)
    assert torch.equal(validated["w"], source["w"])
    assert validated["w"].data_ptr() != source["w"].data_ptr()
    validated["w"].add_(10.0)
    assert source["w"].tolist() == [1.0, 2.0]


def test_validate_state_dict_rejects_empty() -> None:
    with pytest.raises(AggregationError, match="empty"):
        validate_state_dict({})


def test_validate_state_dict_rejects_non_tensor() -> None:
    with pytest.raises(AggregationError, match="must be a tensor"):
        validate_state_dict({"w": [1.0, 2.0]})


def test_validate_state_dict_rejects_non_floating() -> None:
    with pytest.raises(AggregationError, match="floating point"):
        validate_state_dict({"w": torch.tensor([1, 2])})


def test_validate_state_dict_rejects_non_finite() -> None:
    with pytest.raises(AggregationError, match="non-finite"):
        validate_state_dict({"w": torch.tensor([float("nan")])})


def test_validate_state_dict_reference_key_mismatch() -> None:
    with pytest.raises(AggregationError, match="key mismatch"):
        validate_state_dict({"b": torch.zeros(1)}, reference={"a": torch.zeros(1)})


def test_validate_state_dict_reference_shape_mismatch() -> None:
    with pytest.raises(AggregationError, match="shape mismatch"):
        validate_state_dict({"w": torch.zeros(3)}, reference={"w": torch.zeros(2)})


def test_validate_state_dict_reference_dtype_mismatch() -> None:
    with pytest.raises(AggregationError, match="dtype mismatch"):
        validate_state_dict(
            {"w": torch.zeros(2, dtype=torch.float32)},
            reference={"w": torch.zeros(2, dtype=torch.float64)},
        )


# --------------------------------------------------------------------------- #
# uniform_average: D015 arithmetic + isolation (R001/R012)
# --------------------------------------------------------------------------- #
def test_uniform_average_arithmetic() -> None:
    averaged = uniform_average(
        [
            {"w": torch.tensor([0.0])},
            {"w": torch.tensor([10.0])},
            {"w": torch.tensor([20.0])},
        ]
    )
    assert averaged["w"].tolist() == [10.0]


def test_uniform_average_single_state_is_identity_clone() -> None:
    source = {"w": torch.tensor([3.0])}
    averaged = uniform_average([source])
    assert averaged["w"].tolist() == [3.0]
    assert averaged["w"].data_ptr() != source["w"].data_ptr()


def test_uniform_average_does_not_mutate_or_alias_inputs() -> None:
    a = {"w": torch.tensor([0.0])}
    b = {"w": torch.tensor([2.0])}
    averaged = uniform_average([a, b])
    assert averaged["w"].data_ptr() != a["w"].data_ptr()
    averaged["w"].add_(5.0)
    assert a["w"].tolist() == [0.0]
    assert b["w"].tolist() == [2.0]


def test_uniform_average_rejects_empty() -> None:
    with pytest.raises(AggregationError, match="at least one state"):
        uniform_average([])


def test_uniform_average_rejects_key_mismatch() -> None:
    with pytest.raises(AggregationError, match="key mismatch"):
        uniform_average([{"a": torch.zeros(1)}, {"b": torch.zeros(1)}])


def test_uniform_average_rejects_non_finite_input() -> None:
    with pytest.raises(AggregationError, match="non-finite"):
        uniform_average(
            [{"w": torch.tensor([0.0])}, {"w": torch.tensor([float("inf")])}]
        )


def test_aggregate_closed_neighborhood_narrow_callable() -> None:
    averaged = aggregate_closed_neighborhood(
        {"w": torch.tensor([0.0])},
        [{"w": torch.tensor([10.0])}],
        node_id=0,
        round_index=0,
    )
    assert averaged["w"].tolist() == [5.0]


# --------------------------------------------------------------------------- #
# run_round: closed-neighborhood arithmetic on exact topologies
# --------------------------------------------------------------------------- #
def test_chain3_closed_neighborhood_average() -> None:
    topology = load_topology(_config("chain3"))
    result = _run(NODES3, topology, _stub_trainer(lambda n: n * 10.0))
    assert sorted(result.states) == [0, 1, 2]
    assert result.states[0]["w"].tolist() == [5.0]
    assert result.states[1]["w"].tolist() == [10.0]
    assert result.states[2]["w"].tolist() == [15.0]
    assert result.round_index == 0


def test_line2_closed_neighborhood_average() -> None:
    config = _config("line2")
    topology = load_topology(config)
    result = _run((0, 1), topology, _stub_trainer(lambda n: n * 10.0), config=config)
    assert result.states[0]["w"].tolist() == [5.0]
    assert result.states[1]["w"].tolist() == [5.0]


def test_isolated_node_averages_self_only() -> None:
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(1, 2)  # node 0 is isolated
    topology = nx.freeze(graph)
    result = _run(NODES3, topology, _stub_trainer(lambda n: n * 10.0))
    assert result.states[0]["w"].tolist() == [0.0]
    assert result.states[1]["w"].tolist() == [15.0]
    assert result.states[2]["w"].tolist() == [15.0]


# --------------------------------------------------------------------------- #
# Messages: honest, edge-only, order-independent (R001/R012)
# --------------------------------------------------------------------------- #
def test_chain3_message_summaries_follow_edges() -> None:
    topology = load_topology(_config("chain3"))
    result = _run(NODES3, topology, _stub_trainer(lambda n: n * 10.0))
    assert result.messages == (
        MessageSummary(sender=0, recipients=(1,)),
        MessageSummary(sender=1, recipients=(0, 2)),
        MessageSummary(sender=2, recipients=(1,)),
    )


def test_build_messages_only_across_edges() -> None:
    topology = load_topology(_config("line2"))
    messages = build_messages(
        {0: {"w": torch.tensor([1.0])}, 1: {"w": torch.tensor([2.0])}},
        topology,
        round_index=0,
    )
    pairs = sorted((message.sender, message.recipient) for message in messages)
    assert pairs == [(0, 1), (1, 0)]
    for message in messages:
        assert message.round_index == 0


def test_build_messages_are_clones_of_sender_state() -> None:
    topology = load_topology(_config("chain3"))
    trained = {node: {"w": torch.tensor([float(node)])} for node in NODES3}
    messages = build_messages(trained, topology, round_index=0)
    for message in messages:
        assert message.state["w"].tolist() == [float(message.sender)]
        assert message.state["w"].data_ptr() != trained[message.sender]["w"].data_ptr()


def test_run_round_order_independent() -> None:
    topology = load_topology(_config("chain3"))
    trainer = _stub_trainer(lambda n: n * 10.0)
    forward = _run(NODES3, topology, trainer)
    reversed_snapshots = {
        node: {"w": torch.tensor([0.0])} for node in reversed(NODES3)
    }
    backward = _run(NODES3, topology, trainer, snapshots=reversed_snapshots)
    for node in NODES3:
        assert torch.equal(forward.states[node]["w"], backward.states[node]["w"])
    assert forward.messages == backward.messages
    assert forward.stats == backward.stats


def test_run_round_stats_sorted_and_match_trainer() -> None:
    topology = load_topology(_config("chain3"))
    result = _run(NODES3, topology, _stub_trainer(lambda n: n * 10.0))
    assert [stat.node_id for stat in result.stats] == [0, 1, 2]
    assert result.stats[0].neighbors == (1,)
    assert result.stats[1].neighbors == (0, 2)
    assert result.stats[2].neighbors == (1,)
    assert result.stats[0].pre_loss == 1.0
    assert result.stats[0].post_loss == 0.5
    assert result.stats[0].sample_count == 4


# --------------------------------------------------------------------------- #
# Isolation / no partial commit (R012 + Q7)
# --------------------------------------------------------------------------- #
def test_run_round_does_not_mutate_snapshots_or_alias_outputs() -> None:
    topology = load_topology(_config("chain3"))
    snapshots = _zero_snapshots(NODES3)
    baseline = {
        node: {key: value.clone() for key, value in state.items()}
        for node, state in snapshots.items()
    }
    result = _run(
        NODES3, topology, _stub_trainer(lambda n: n * 10.0), snapshots=snapshots
    )
    for node in NODES3:
        for key in snapshots[node]:
            assert torch.equal(snapshots[node][key], baseline[node][key])
            assert (
                result.states[node][key].data_ptr()
                != snapshots[node][key].data_ptr()
            )
    result.states[0]["w"].add_(100.0)
    assert torch.equal(snapshots[0]["w"], baseline[0]["w"])


def test_run_round_does_not_mutate_trained_states() -> None:
    topology = load_topology(_config("chain3"))
    captured: dict[int, dict[str, torch.Tensor]] = {}

    def trainer(
        snapshot,
        dataset,
        *,
        config,
        node_id,
        round_index,
        train_seed,
        data_order_seed,
    ):
        state = {
            key: value.detach().clone() + float(node_id) * 10.0
            for key, value in snapshot.items()
        }
        captured[node_id] = state
        return LocalTrainResult(
            node_id=node_id,
            round_index=round_index,
            state=state,
            pre_loss=1.0,
            post_loss=0.5,
            sample_count=4,
        )

    _run(NODES3, topology, trainer)
    assert captured[0]["w"].tolist() == [0.0]
    assert captured[1]["w"].tolist() == [10.0]
    assert captured[2]["w"].tolist() == [20.0]


def test_run_round_late_failure_aborts_without_commit() -> None:
    topology = load_topology(_config("chain3"))
    snapshots = _zero_snapshots(NODES3)
    baseline = {
        node: state["w"].clone() for node, state in snapshots.items()
    }

    def trainer(
        snapshot,
        dataset,
        *,
        config,
        node_id,
        round_index,
        train_seed,
        data_order_seed,
    ):
        if node_id == 2:
            raise TrainingError("injected late failure node=2")
        return LocalTrainResult(
            node_id=node_id,
            round_index=round_index,
            state={key: value.detach().clone() for key, value in snapshot.items()},
            pre_loss=1.0,
            post_loss=0.5,
            sample_count=4,
        )

    with pytest.raises(TrainingError, match="injected late failure"):
        _run(NODES3, topology, trainer, snapshots=snapshots)
    for node in NODES3:
        assert torch.equal(snapshots[node]["w"], baseline[node])


def test_round_result_is_frozen() -> None:
    topology = load_topology(_config("chain3"))
    result = _run(NODES3, topology, _stub_trainer(lambda n: n * 10.0))
    assert isinstance(result, RoundResult)
    with pytest.raises(Exception):
        result.round_index = 5  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Aggregation seam: S03 policy + observation builder replace weight selection
# --------------------------------------------------------------------------- #
def test_custom_policy_seam_self_only() -> None:
    topology = load_topology(_config("chain3"))

    class SelfOnlyPolicy:
        def select_weights(self, observation):
            multipliers = (1.0,) + (0.0,) * observation.degree
            return build_weight_profile(observation.participant_ids, multipliers)

    result = _run(
        NODES3,
        topology,
        _stub_trainer(lambda n: n * 10.0),
        policy=SelfOnlyPolicy(),
    )
    assert result.states[0]["w"].tolist() == [0.0]
    assert result.states[1]["w"].tolist() == [10.0]
    assert result.states[2]["w"].tolist() == [20.0]


# --------------------------------------------------------------------------- #
# Negative surface: node sets, trainer identity, payloads (Q7)
# --------------------------------------------------------------------------- #
def test_run_round_missing_snapshot_node_rejected() -> None:
    topology = load_topology(_config("chain3"))
    with pytest.raises(RoundError, match="snapshot node set"):
        _run(
            NODES3,
            topology,
            _stub_trainer(lambda n: 0.0),
            snapshots={0: {"w": torch.tensor([0.0])}, 1: {"w": torch.tensor([0.0])}},
        )


def test_run_round_extra_snapshot_node_rejected() -> None:
    topology = load_topology(_config("chain3"))
    snapshots = _zero_snapshots(NODES3)
    snapshots[9] = {"w": torch.tensor([0.0])}
    with pytest.raises(RoundError, match="snapshot node set"):
        _run(NODES3, topology, _stub_trainer(lambda n: 0.0), snapshots=snapshots)


def test_run_round_partition_node_mismatch_rejected() -> None:
    topology = load_topology(_config("chain3"))
    with pytest.raises(RoundError, match="partition node set"):
        _run(
            NODES3,
            topology,
            _stub_trainer(lambda n: 0.0),
            partitions={0: None, 1: None},
        )


def test_run_round_wrong_result_identity_rejected() -> None:
    topology = load_topology(_config("chain3"))

    def trainer(
        snapshot,
        dataset,
        *,
        config,
        node_id,
        round_index,
        train_seed,
        data_order_seed,
    ):
        return LocalTrainResult(
            node_id=99,
            round_index=round_index,
            state={key: value.detach().clone() for key, value in snapshot.items()},
            pre_loss=1.0,
            post_loss=0.5,
            sample_count=4,
        )

    with pytest.raises(RoundError, match="mismatched identity"):
        _run(NODES3, topology, trainer)


def test_run_round_trained_key_mismatch_rejected() -> None:
    topology = load_topology(_config("chain3"))

    def trainer(
        snapshot,
        dataset,
        *,
        config,
        node_id,
        round_index,
        train_seed,
        data_order_seed,
    ):
        state = {key: value.detach().clone() for key, value in snapshot.items()}
        if node_id == 1:
            state["extra"] = torch.tensor([1.0])
        return LocalTrainResult(
            node_id=node_id,
            round_index=round_index,
            state=state,
            pre_loss=1.0,
            post_loss=0.5,
            sample_count=4,
        )

    with pytest.raises(AggregationError, match="key mismatch"):
        _run(NODES3, topology, trainer)


def test_run_round_trained_shape_mismatch_rejected() -> None:
    topology = load_topology(_config("chain3"))

    def trainer(
        snapshot,
        dataset,
        *,
        config,
        node_id,
        round_index,
        train_seed,
        data_order_seed,
    ):
        state = {key: value.detach().clone() for key, value in snapshot.items()}
        if node_id == 1:
            state["w"] = torch.zeros(2)
        return LocalTrainResult(
            node_id=node_id,
            round_index=round_index,
            state=state,
            pre_loss=1.0,
            post_loss=0.5,
            sample_count=4,
        )

    with pytest.raises(AggregationError, match="shape mismatch"):
        _run(NODES3, topology, trainer)


def test_run_round_trained_non_finite_rejected() -> None:
    topology = load_topology(_config("chain3"))

    def trainer(
        snapshot,
        dataset,
        *,
        config,
        node_id,
        round_index,
        train_seed,
        data_order_seed,
    ):
        state = {key: value.detach().clone() for key, value in snapshot.items()}
        if node_id == 2:
            state["w"] = torch.tensor([float("nan")])
        return LocalTrainResult(
            node_id=node_id,
            round_index=round_index,
            state=state,
            pre_loss=1.0,
            post_loss=0.5,
            sample_count=4,
        )

    with pytest.raises(AggregationError, match="non-finite"):
        _run(NODES3, topology, trainer)


# --------------------------------------------------------------------------- #
# deliver_to_inboxes: protocol boundary negatives (Q3/Q7)
# --------------------------------------------------------------------------- #
def test_deliver_accepts_valid_chain3_messages() -> None:
    topology = load_topology(_config("chain3"))
    inboxes = deliver_to_inboxes(_valid_messages(topology), topology, round_index=0)
    assert [message.sender for message in inboxes[1]] == [0, 2]
    assert [message.sender for message in inboxes[0]] == [1]
    assert inboxes[0][0].state["w"].tolist() == [1.0]


def test_deliver_rejects_non_edge_message() -> None:
    topology = load_topology(_config("chain3"))
    messages = list(_valid_messages(topology))
    messages.append(
        RoundMessage(
            round_index=0, sender=0, recipient=2, state={"w": torch.tensor([1.0])}
        )
    )
    with pytest.raises(RoundError, match="non-edge"):
        deliver_to_inboxes(messages, topology, round_index=0)


def test_deliver_rejects_duplicate_message() -> None:
    topology = load_topology(_config("chain3"))
    messages = list(_valid_messages(topology))
    messages.append(
        RoundMessage(
            round_index=0, sender=0, recipient=1, state={"w": torch.tensor([1.0])}
        )
    )
    with pytest.raises(RoundError, match="duplicate"):
        deliver_to_inboxes(messages, topology, round_index=0)


def test_deliver_rejects_missing_expected_message() -> None:
    topology = load_topology(_config("chain3"))
    messages = [
        message
        for message in _valid_messages(topology)
        if not (message.sender == 0 and message.recipient == 1)
    ]
    with pytest.raises(RoundError, match="missing expected message"):
        deliver_to_inboxes(messages, topology, round_index=0)


def test_deliver_rejects_unknown_recipient() -> None:
    topology = load_topology(_config("chain3"))
    messages = list(_valid_messages(topology))
    messages.append(
        RoundMessage(
            round_index=0, sender=1, recipient=99, state={"w": torch.tensor([1.0])}
        )
    )
    with pytest.raises(RoundError, match="unknown recipient"):
        deliver_to_inboxes(messages, topology, round_index=0)


def test_deliver_rejects_unknown_sender() -> None:
    topology = load_topology(_config("chain3"))
    messages = list(_valid_messages(topology))
    messages.append(
        RoundMessage(
            round_index=0, sender=99, recipient=1, state={"w": torch.tensor([1.0])}
        )
    )
    with pytest.raises(RoundError, match="unknown sender"):
        deliver_to_inboxes(messages, topology, round_index=0)


def test_deliver_rejects_wrong_round() -> None:
    topology = load_topology(_config("chain3"))
    messages = list(_valid_messages(topology))
    original = messages[0]
    messages[0] = RoundMessage(
        round_index=5,
        sender=original.sender,
        recipient=original.recipient,
        state=original.state,
    )
    with pytest.raises(RoundError, match="round mismatch"):
        deliver_to_inboxes(messages, topology, round_index=0)


def test_deliver_rejects_non_finite_message_state() -> None:
    topology = load_topology(_config("chain3"))
    messages = list(_valid_messages(topology))
    original = messages[0]
    messages[0] = RoundMessage(
        round_index=0,
        sender=original.sender,
        recipient=original.recipient,
        state={"w": torch.tensor([float("nan")])},
    )
    with pytest.raises(AggregationError, match="non-finite"):
        deliver_to_inboxes(messages, topology, round_index=0)


# --------------------------------------------------------------------------- #
# Integration: real small CNN through the production trainer (R001)
# --------------------------------------------------------------------------- #
def test_run_round_default_trainer_real_model() -> None:
    config = make_config(
        topology={"kind": "chain3"},
        training={"local_epochs": 10, "learning_rate": 0.1},
        data={"samples_per_node": 16},
    )
    topology = load_topology(config)
    seeds = resolve_seeds(config)
    nodes = tuple(sorted(topology.nodes()))

    dataset = MNISTShapedDataset(len(nodes) * 16, seed=3, separable=True)
    partitions = partition_dataset(dataset, nodes, 16, seed=seeds.partition)
    snapshot = clone_state_dict(build_model("small_cnn", seed=seeds.torch).state_dict())
    snapshots = {node: clone_state_dict(snapshot) for node in nodes}

    result = run_round(
        config=config,
        topology=topology,
        snapshots=snapshots,
        partitions=partitions,
        round_index=0,
        seeds=seeds,
    )

    assert sorted(result.states) == list(nodes)
    stats_by_node = {stat.node_id: stat for stat in result.stats}
    for node in nodes:
        assert result.states[node].keys() == snapshot.keys()
        for value in result.states[node].values():
            assert bool(torch.isfinite(value).all())
        assert stats_by_node[node].post_loss < stats_by_node[node].pre_loss


# --------------------------------------------------------------------------- #
# Fixed-attack transport: compute-once broadcast, barrier, atomicity (R003/R004)
# --------------------------------------------------------------------------- #
def _attack_config(kind: str = "chain3", node=None, intensity: str = "zero"):
    return make_config(
        topology={"kind": kind},
        attack={"compromised_node": node, "intensity": intensity},
    )


def _bump_trainer(delta: float = 1.0):
    """Stub trainer: trained state = snapshot + delta for every node."""

    def trainer(
        snapshot,
        dataset,
        *,
        config,
        node_id,
        round_index,
        train_seed,
        data_order_seed,
    ):
        state = {
            key: value.detach().clone() + delta for key, value in snapshot.items()
        }
        return LocalTrainResult(
            node_id=node_id,
            round_index=round_index,
            state=state,
            pre_loss=1.0,
            post_loss=0.5,
            sample_count=4,
        )

    return trainer


class _RecordingController:
    """Wrap a real fixed controller, recording call identity and order."""

    def __init__(self, inner, order=None):
        self.inner = inner
        self.calls: list[tuple[int, int | None]] = []
        self._order = order

    @property
    def compromised_node(self):
        return self.inner.compromised_node

    @property
    def kappa(self):
        return self.inner.kappa

    def outgoing_state(
        self, *, sender, round_start, trained, round_index=None, context=""
    ):
        self.calls.append((int(sender), round_index))
        if self._order is not None:
            self._order.append(("controller", int(sender)))
        return self.inner.outgoing_state(
            sender=sender,
            round_start=round_start,
            trained=trained,
            round_index=round_index,
            context=context,
        )


class _RecordingObserver:
    """Wrap the real observation builder, recording delivered sender payloads.

    The observation builder is the first seam that exposes each node's
    post-barrier delivered sender states in canonical order, so the recorder
    captures them (as lists) before delegating. Optional in-place mutation of
    the delivered copies proves per-edge storage isolation.
    """

    def __init__(self, order=None, mutate: bool = False):
        self._inner = DefaultObservationBuilder()
        self.values: dict[int, list[list[float]]] = {}
        self.calls: list[int] = []
        self.observations: dict[int, LocalObservation] = {}
        self._order = order
        self._mutate = mutate

    def build_observation(
        self,
        *,
        node_id,
        round_index,
        round_start,
        trained,
        sender_states,
        evaluator,
    ):
        self.calls.append(int(node_id))
        self.values[int(node_id)] = [
            list(state["w"].tolist()) for _, state in sender_states
        ]
        if self._order is not None:
            self._order.append(("observer", int(node_id)))
        if self._mutate:
            # Mutate this node's delivered copies in place; independent storage
            # means no recipient or sender can observe the change.
            for _, state in sender_states:
                state["w"].add_(1000.0)
        observation = self._inner.build_observation(
            node_id=node_id,
            round_index=round_index,
            round_start=round_start,
            trained=trained,
            sender_states=sender_states,
            evaluator=evaluator,
        )
        self.observations[int(node_id)] = observation
        return observation


class _RecordingPolicy:
    """Record selected profiles and call order, then apply plain averaging."""

    def __init__(self, order=None):
        self._inner = PlainAveragePolicy()
        self.calls: list[int] = []
        self.profiles: dict[int, WeightProfile] = {}
        self.observations: dict[int, LocalObservation] = {}
        self._order = order

    def select_weights(self, observation: LocalObservation) -> WeightProfile:
        self.calls.append(int(observation.node_id))
        self.observations[int(observation.node_id)] = observation
        if self._order is not None:
            self._order.append(("policy", int(observation.node_id)))
        profile = self._inner.select_weights(observation)
        self.profiles[int(observation.node_id)] = profile
        return profile


class _SelfOnlyPolicy:
    """Return exact self-only weights for every observation (empty/all-reject)."""

    def select_weights(self, observation: LocalObservation) -> WeightProfile:
        multipliers = (1.0,) + (0.0,) * observation.degree
        return build_weight_profile(observation.participant_ids, multipliers)


def test_fixed_controller_satisfies_outgoing_protocol() -> None:
    controller = FixedAttackController(attack=AttackConfig())
    assert isinstance(controller, OutgoingController)


def test_chain3_strong_attack_math_in_delivered_inboxes() -> None:
    # snapshots 0 -> trained 1 at every node; node 0 strong reversal = -4.
    config = _attack_config(node=0, intensity="strong")
    topology = load_topology(config)
    recorder = _RecordingObserver()
    result = _run(
        NODES3, topology, _bump_trainer(1.0), config=config, observer=recorder
    )

    assert recorder.values[0] == [[1.0]]
    assert recorder.values[1] == [[-4.0], [1.0]]
    assert recorder.values[2] == [[1.0]]
    # Attack does not change the topology of edge-only delivery.
    assert result.messages == (
        MessageSummary(sender=0, recipients=(1,)),
        MessageSummary(sender=1, recipients=(0, 2)),
        MessageSummary(sender=2, recipients=(1,)),
    )


def test_line2_strong_attack_math_in_delivered_inboxes() -> None:
    config = _attack_config(kind="line2", node=0, intensity="strong")
    topology = load_topology(config)
    recorder = _RecordingObserver()
    _run((0, 1), topology, _bump_trainer(1.0), config=config, observer=recorder)

    assert recorder.values[0] == [[1.0]]
    assert recorder.values[1] == [[-4.0]]


def test_controller_called_exactly_once_per_sender_per_round() -> None:
    # chain3 has four directed edges but only three senders: no per-edge recompute.
    config = _attack_config(node=1, intensity="strong")
    topology = load_topology(config)
    controller = _RecordingController(FixedAttackController.from_experiment(config))
    recorder = _RecordingObserver()
    _run(
        NODES3,
        topology,
        _bump_trainer(1.0),
        config=config,
        controller=controller,
        observer=recorder,
    )

    assert sorted(controller.calls) == [(0, 0), (1, 0), (2, 0)]
    assert len(controller.calls) == 3
    assert recorder.values[0] == [[-4.0]]
    assert recorder.values[1] == [[1.0], [1.0]]
    assert recorder.values[2] == [[-4.0]]


def test_build_messages_siblings_equal_but_independently_stored() -> None:
    topology = load_topology(_config("chain3"))
    canonical = {
        0: {"w": torch.tensor([-4.0])},
        1: {"w": torch.tensor([1.0])},
        2: {"w": torch.tensor([1.0])},
    }
    messages = build_messages(canonical, topology, round_index=0)
    siblings = [m for m in messages if m.sender == 1]

    assert len(siblings) == 2
    assert torch.equal(siblings[0].state["w"], siblings[1].state["w"])
    assert siblings[0].state["w"].data_ptr() != siblings[1].state["w"].data_ptr()
    assert all(
        m.state["w"].data_ptr() != canonical[m.sender]["w"].data_ptr()
        for m in messages
    )


def test_recipient_clone_mutation_does_not_leak_between_neighbors() -> None:
    topology = load_topology(_config("chain3"))
    recorder = _RecordingObserver(mutate=True)
    _run(NODES3, topology, _stub_trainer(lambda n: n * 10.0), observer=recorder)

    # Values are captured before mutation; node 2 still sees node 1's honest 10.0
    # even though node 0's observer mutated its own delivered copy in place.
    assert recorder.values[0] == [[10.0]]
    assert recorder.values[1] == [[0.0], [20.0]]
    assert recorder.values[2] == [[10.0]]


def test_attack_does_not_mutate_snapshots_or_trained_states() -> None:
    config = _attack_config(node=0, intensity="strong")
    topology = load_topology(config)
    snapshots = _zero_snapshots(NODES3)
    baseline = {node: state["w"].clone() for node, state in snapshots.items()}
    captured: dict[int, dict[str, torch.Tensor]] = {}

    def trainer(
        snapshot,
        dataset,
        *,
        config,
        node_id,
        round_index,
        train_seed,
        data_order_seed,
    ):
        state = {
            key: value.detach().clone() + 1.0 for key, value in snapshot.items()
        }
        captured[node_id] = state
        return LocalTrainResult(
            node_id=node_id,
            round_index=round_index,
            state=state,
            pre_loss=1.0,
            post_loss=0.5,
            sample_count=4,
        )

    _run(NODES3, topology, trainer, config=config, snapshots=snapshots)

    for node in NODES3:
        assert torch.equal(snapshots[node]["w"], baseline[node])
    # The compromised node's own trained state is never overwritten by poison.
    assert captured[0]["w"].tolist() == [1.0]


def test_selected_zero_kappa_matches_honest_round_exactly() -> None:
    topology = load_topology(_config("chain3"))
    trainer = _bump_trainer(1.0)
    honest = _run(NODES3, topology, trainer)
    noop = _run(
        NODES3, topology, trainer, config=_attack_config(node=0, intensity="zero")
    )

    for node in NODES3:
        assert torch.equal(honest.states[node]["w"], noop.states[node]["w"])
    assert honest.messages == noop.messages
    assert honest.stats == noop.stats


def test_receive_barrier_precedes_observation_and_policy() -> None:
    config = _attack_config(node=1, intensity="weak")
    topology = load_topology(config)
    order: list[tuple[str, int]] = []
    controller = _RecordingController(
        FixedAttackController.from_experiment(config), order=order
    )
    observer = _RecordingObserver(order=order)
    policy = _RecordingPolicy(order=order)
    _run(
        NODES3,
        topology,
        _bump_trainer(1.0),
        config=config,
        controller=controller,
        observer=observer,
        policy=policy,
    )

    kinds = [kind for kind, _ in order]
    # Every controller call (canonical outgoing computation) happens before any
    # observation or policy call: the receive barrier fully precedes phase 6.
    assert kinds[:3] == ["controller", "controller", "controller"]
    assert "controller" not in kinds[3:]
    assert sorted(kinds[3:]) == sorted(
        ["observer", "observer", "observer", "policy", "policy", "policy"]
    )
    # Each node is observed once and selected once, and in the same order.
    assert observer.calls == policy.calls == [0, 1, 2]


def test_raising_controller_aborts_before_observation() -> None:
    topology = load_topology(_config("chain3"))
    snapshots = _zero_snapshots(NODES3)
    baseline = {node: state["w"].clone() for node, state in snapshots.items()}
    observer = _RecordingObserver()
    policy = _RecordingPolicy()

    class RaisingController:
        def outgoing_state(
            self, *, sender, round_start, trained, round_index=None, context=""
        ):
            if int(sender) == 1:
                raise AttackError("injected controller failure")
            return {key: value.detach().clone() for key, value in trained.items()}

    with pytest.raises(AttackError, match="injected controller failure"):
        _run(
            NODES3,
            topology,
            _bump_trainer(1.0),
            snapshots=snapshots,
            controller=RaisingController(),
            observer=observer,
            policy=policy,
        )

    assert observer.calls == []
    assert policy.calls == []
    for node in NODES3:
        assert torch.equal(snapshots[node]["w"], baseline[node])


def test_malformed_controller_output_aborts_before_observation() -> None:
    topology = load_topology(_config("chain3"))
    observer = _RecordingObserver()
    policy = _RecordingPolicy()

    class MalformedController:
        def outgoing_state(
            self, *, sender, round_start, trained, round_index=None, context=""
        ):
            return {"wrong": torch.tensor([1.0])}

    with pytest.raises(AggregationError, match="key mismatch"):
        _run(
            NODES3,
            topology,
            _bump_trainer(1.0),
            controller=MalformedController(),
            observer=observer,
            policy=policy,
        )

    assert observer.calls == []
    assert policy.calls == []


# --------------------------------------------------------------------------- #
# Observe -> select -> weighted aggregate integration (profiles, weights)
# --------------------------------------------------------------------------- #
def test_round_result_records_sorted_selected_profiles() -> None:
    topology = load_topology(_config("chain3"))
    result = _run(NODES3, topology, _stub_trainer(lambda n: n * 10.0))

    assert [profile.node_id for profile in result.profiles] == [0, 1, 2]
    assert all(isinstance(p, NodeWeightProfile) for p in result.profiles)
    # Self first, then delivered senders in ascending order.
    assert result.profiles[0].participant_ids == (0, 1)
    assert result.profiles[1].participant_ids == (1, 0, 2)
    assert result.profiles[2].participant_ids == (2, 1)
    for profile in result.profiles:
        assert len(profile.weights) == len(profile.multipliers)
        assert len(profile.weights) == len(profile.participant_ids)
        assert abs(sum(profile.weights) - 1.0) <= 1e-12


def test_selected_weights_determine_committed_tensors() -> None:
    topology = load_topology(_config("chain3"))

    class SelectivePolicy:
        """Reject sender 0 when present; accept every other sender."""

        def select_weights(self, observation):
            multipliers = [1.0] + [
                0.0 if sender == 0 else 1.0
                for sender in observation.neighbor_ids
            ]
            return build_weight_profile(
                observation.participant_ids, multipliers
            )

    result = _run(
        NODES3,
        topology,
        _stub_trainer(lambda n: n * 10.0),
        policy=SelectivePolicy(),
    )

    # Node 1 trained 10, sender 0 (0.0) rejected, sender 2 (20.0) kept.
    assert result.profiles[1].multipliers == (1.0, 0.0, 1.0)
    assert result.profiles[1].weights == (0.5, 0.0, 0.5)
    # Committed tensor equals the weighted combination represented by the
    # recorded profile: 0.5*10 + 0.0*0 + 0.5*20 = 15.
    assert result.states[1]["w"].tolist() == [15.0]
    # Nodes with no rejected neighbor stay at the uniform closed mean.
    assert result.states[0]["w"].tolist() == [5.0]
    assert result.states[2]["w"].tolist() == [15.0]


def test_observer_and_policy_called_once_per_node_after_delivery() -> None:
    config = _attack_config(node=1, intensity="weak")
    topology = load_topology(config)
    observer = _RecordingObserver()
    policy = _RecordingPolicy()
    _run(
        NODES3,
        topology,
        _bump_trainer(1.0),
        config=config,
        observer=observer,
        policy=policy,
    )

    assert observer.calls == [0, 1, 2]
    assert policy.calls == [0, 1, 2]
    assert set(observer.observations) == {0, 1, 2}
    assert set(policy.observations) == {0, 1, 2}
    # The policy's sole input is the frozen scalar observation, self first.
    for observation in policy.observations.values():
        assert isinstance(observation, LocalObservation)
        assert observation.participant_ids[0] == observation.node_id


def test_empty_inbox_commits_self_only() -> None:
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(1, 2)  # node 0 is isolated
    topology = nx.freeze(graph)
    result = _run(NODES3, topology, _stub_trainer(lambda n: n * 10.0))

    assert result.profiles[0].participant_ids == (0,)
    assert result.profiles[0].weights == (1.0,)
    assert result.states[0]["w"].tolist() == [0.0]


def test_all_reject_policy_commits_exact_self_state() -> None:
    topology = load_topology(_config("chain3"))
    result = _run(
        NODES3,
        topology,
        _stub_trainer(lambda n: n * 10.0),
        policy=_SelfOnlyPolicy(),
    )
    assert result.states[0]["w"].tolist() == [0.0]
    assert result.states[1]["w"].tolist() == [10.0]
    assert result.states[2]["w"].tolist() == [20.0]
    for profile in result.profiles:
        assert profile.weights[0] == 1.0
        assert all(weight == 0.0 for weight in profile.weights[1:])


def test_fixed_similarity_config_is_applied_end_to_end() -> None:
    config = make_config(
        topology={"kind": "chain3"},
        aggregation={
            "kind": "fixed_similarity",
            "cutoff": 0.0,
            "downweight_multiplier": 0.25,
        },
    )
    topology = load_topology(config)
    result = _run(
        NODES3, topology, _stub_trainer(lambda n: n * 10.0), config=config
    )

    # Node 0's only sender is directionally opposed (disagreement 1.0 > 0.0)
    # so it is rejected -> exact self-only fallback.
    assert result.profiles[0].multipliers == (1.0, 0.0)
    assert result.profiles[0].weights == (1.0, 0.0)
    assert result.states[0]["w"].tolist() == [0.0]
    # Node 2's sender matches direction (disagreement 0.0 <= 0.0) -> accepted.
    assert result.profiles[2].multipliers == (1.0, 1.0)
    assert result.states[2]["w"].tolist() == [15.0]


def test_round_result_profiles_are_immutable() -> None:
    topology = load_topology(_config("chain3"))
    result = _run(NODES3, topology, _stub_trainer(lambda n: n * 10.0))
    with pytest.raises(Exception):
        result.profiles[0].weights = (1.0, 0.0)  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Observer/policy untrusted-boundary negatives (Q3/Q7)
# --------------------------------------------------------------------------- #
def test_policy_wrong_type_aborts_before_commit() -> None:
    topology = load_topology(_config("chain3"))
    snapshots = _zero_snapshots(NODES3)
    baseline = {node: state["w"].clone() for node, state in snapshots.items()}

    class WrongTypePolicy:
        def select_weights(self, observation):
            return {"not": "a profile"}

    with pytest.raises(RoundError, match="instead of a"):
        _run(
            NODES3,
            topology,
            _stub_trainer(lambda n: n * 10.0),
            policy=WrongTypePolicy(),
            snapshots=snapshots,
        )
    for node in NODES3:
        assert torch.equal(snapshots[node]["w"], baseline[node])


def test_policy_exception_aborts_with_node_context() -> None:
    topology = load_topology(_config("chain3"))

    class RaisingPolicy:
        def select_weights(self, observation):
            if observation.node_id == 2:
                raise PolicyError("injected policy failure")
            return PlainAveragePolicy().select_weights(observation)

    with pytest.raises(RoundError, match=r"policy failed for node=2 round=0") as info:
        _run(
            NODES3,
            topology,
            _stub_trainer(lambda n: n * 10.0),
            policy=RaisingPolicy(),
        )
    assert isinstance(info.value.__cause__, PolicyError)


def test_policy_foreign_participant_profile_rejected() -> None:
    topology = load_topology(_config("chain3"))

    class ForeignPolicy:
        def select_weights(self, observation):
            return WeightProfile((99,), (1.0,), (1.0,))

    with pytest.raises(RoundError, match="policy participants"):
        _run(
            NODES3,
            topology,
            _stub_trainer(lambda n: n * 10.0),
            policy=ForeignPolicy(),
        )


def test_policy_reordered_participants_rejected() -> None:
    topology = load_topology(_config("chain3"))

    class ReorderedPolicy:
        def select_weights(self, observation):
            ids = tuple(reversed(observation.participant_ids))
            return build_weight_profile(ids, (1.0,) * len(ids))

    with pytest.raises(RoundError, match="policy participants"):
        _run(
            NODES3,
            topology,
            _stub_trainer(lambda n: n * 10.0),
            policy=ReorderedPolicy(),
        )


def test_observer_wrong_type_aborts_before_commit() -> None:
    topology = load_topology(_config("chain3"))
    snapshots = _zero_snapshots(NODES3)
    baseline = {node: state["w"].clone() for node, state in snapshots.items()}

    class WrongTypeObserver:
        def build_observation(self, **kwargs):
            return object()

    with pytest.raises(RoundError, match="instead of a"):
        _run(
            NODES3,
            topology,
            _stub_trainer(lambda n: n * 10.0),
            observer=WrongTypeObserver(),
            snapshots=snapshots,
        )
    for node in NODES3:
        assert torch.equal(snapshots[node]["w"], baseline[node])


def test_observer_exception_aborts_with_node_context() -> None:
    topology = load_topology(_config("chain3"))
    inner = DefaultObservationBuilder()

    class RaisingObserver:
        def build_observation(self, *, node_id, **kwargs):
            if node_id == 2:
                raise ObservationError("injected observer failure")
            return inner.build_observation(node_id=node_id, **kwargs)

    with pytest.raises(
        RoundError, match=r"observation failed for node=2 round=0"
    ) as info:
        _run(
            NODES3,
            topology,
            _stub_trainer(lambda n: n * 10.0),
            observer=RaisingObserver(),
        )
    assert isinstance(info.value.__cause__, ObservationError)


def test_observer_mismatched_participants_rejected() -> None:
    topology = load_topology(_config("chain3"))

    class StaleObserver:
        def build_observation(
            self,
            *,
            node_id,
            round_index,
            round_start,
            trained,
            sender_states,
            evaluator,
        ):
            # Empty neighbor set even though the inbox is non-empty.
            return LocalObservation(
                node_id=node_id,
                round_index=round_index,
                epsilon=DEFAULT_EPSILON,
                neighbors=(),
            )

    with pytest.raises(RoundError, match="observation participants"):
        _run(
            NODES3,
            topology,
            _stub_trainer(lambda n: n * 10.0),
            observer=StaleObserver(),
        )


def test_failing_local_evaluation_aborts_before_commit() -> None:
    topology = load_topology(_config("chain3"))
    snapshots = _zero_snapshots(NODES3)
    baseline = {node: state["w"].clone() for node, state in snapshots.items()}

    def failing_factory(config, partition, node_id, round_index):
        def evaluate(
            state, *, node_id=node_id, round_index=round_index, context=""
        ):
            raise RuntimeError("injected evaluation failure")

        return evaluate

    with pytest.raises(
        RoundError, match=r"observation failed for node=0 round=0"
    ):
        _run(
            NODES3,
            topology,
            _stub_trainer(lambda n: n * 10.0),
            loss_evaluator=failing_factory,
            snapshots=snapshots,
        )
    for node in NODES3:
        assert torch.equal(snapshots[node]["w"], baseline[node])


def test_run_from_config_reuses_injected_controller_for_every_round() -> None:
    config = make_config(
        rounds=2,
        topology={"kind": "line2"},
        attack={"compromised_node": 0, "intensity": "weak"},
    )
    dataset = MNISTShapedDataset(64, seed=5)
    controller = _RecordingController(FixedAttackController.from_experiment(config))

    result = run_from_config(
        config, dataset=dataset, trainer=_bump_trainer(1.0), controller=controller
    )

    assert len(result.rounds) == 2
    assert controller.calls == [(0, 0), (1, 0), (0, 1), (1, 1)]
