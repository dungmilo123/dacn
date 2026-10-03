"""Offline end-to-end acceptance tests for the S02 fixed-poison episode (T03).

These tests drive the **production** :func:`dfl_sim.run.run_from_config` path --
config loading, topology, real local small-CNN training, attack fanout, the
receive barrier, aggregation, commit, and the canonical JSONL log -- against an
injected deterministic MNIST-shaped fixture. Nothing here downloads data or
reads ``.gsd``/planning artifacts.

They prove the S02 demo end to end:

* a checked-in strong fixed-poison episode (``configs/fixed_poison_3node.yaml``)
  is reproducible and measurably worsens the compromised node's outgoing model,
* a selected-node ``zero`` episode is byte-identical to honest execution,
* transport stays topology-faithful (edge-only, sorted summaries), and
* caller-visible states are never mutated or aliased.

Q5 failure coverage: controller/training failures publish no partial log and
preserve a prior complete log (the atomic writer is also exercised in
``test_clean_episode.py``). Q7 negative coverage: zero-vs-honest exactness
catches accidental attack-path drift, and strong-vs-honest divergence catches a
disconnected controller; malformed transport is covered by the T02 protocol
suite in ``test_round_protocol.py``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from dfl_sim import (
    AttackConfig,
    AttackError,
    DefaultObservationBuilder,
    EpisodeError,
    ModelState,
    build_model,
    canonical_record,
    clone_state_dict,
    evaluate_cross_entropy,
    load_config,
    load_state_into_model,
    load_topology,
    local_train_from_snapshot,
    partition_dataset,
    resolve_seeds,
    run_from_config,
)
from helpers import MNISTShapedDataset, make_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_CONFIG = REPO_ROOT / "configs" / "clean_3node.yaml"
POISON_CONFIG = REPO_ROOT / "configs" / "fixed_poison_3node.yaml"

NODES3 = (0, 1, 2)
# chain3 topology: 0 -- 1 -- 2, so each node's sorted neighbor set is fixed.
CHAIN3_RECIPIENTS = {0: [1], 1: [0, 2], 2: [1]}
SAMPLE_COUNT = 64


def _fixture(num_examples: int = 300, *, seed: int = 7) -> MNISTShapedDataset:
    """Deterministic separable MNIST-shaped fixture (no network, no files)."""

    return MNISTShapedDataset(num_examples, seed=seed, separable=True)


def _zero_variant(config):
    """Return an otherwise identical config with a selected zero-intensity node."""

    return config.model_copy(
        update={"attack": AttackConfig(compromised_node=0, intensity="zero")}
    )


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_states_equal(left: Mapping[int, ModelState], right: Mapping[int, ModelState]) -> None:
    assert set(left) == set(right)
    for node in sorted(left):
        assert set(left[node]) == set(right[node])
        for key in sorted(left[node]):
            assert torch.equal(left[node][key], right[node][key]), (
                f"node {node} tensor {key!r} diverged"
            )


def _evaluate(config, state: Mapping[str, torch.Tensor], partition) -> float:
    """Evaluate a state dict on a partition with the production model/loss path."""

    seeds = resolve_seeds(config)
    model = build_model(config.model.name, seed=seeds.torch)
    load_state_into_model(model, state, context="poison-episode evaluation")
    return evaluate_cross_entropy(
        model,
        partition,
        batch_size=config.training.batch_size,
        context="poison-episode evaluation",
    )


class _InboxObserver:
    """Observation-builder seam that records each node's delivered sender states.

    Wraps the production :class:`~dfl_sim.observation.DefaultObservationBuilder`.
    It records clones of the sorted ``(sender_id, state)`` pairs the round
    engine exposes after the receive barrier -- the exact pre-aggregation
    payload -- without mutating the round, then delegates so the episode behaves
    exactly like production.
    """

    def __init__(self) -> None:
        self._inner = DefaultObservationBuilder()
        self.inboxes: dict[int, dict[int, dict[int, ModelState]]] = {}

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
        record = {
            int(sender): clone_state_dict(state)
            for sender, state in sender_states
        }
        self.inboxes.setdefault(int(round_index), {})[int(node_id)] = record
        return self._inner.build_observation(
            node_id=node_id,
            round_index=round_index,
            round_start=round_start,
            trained=trained,
            sender_states=sender_states,
            evaluator=evaluator,
        )


def _capture_trainer(capture, *, fail_round: int | None = None):
    """Production trainer wrapper capturing round-zero snapshots/trained states."""

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
        if fail_round is not None and int(round_index) >= int(fail_round):
            raise AttackError(f"injected failure at round {round_index}")
        result = local_train_from_snapshot(
            snapshot,
            dataset,
            config=config,
            node_id=node_id,
            round_index=round_index,
            train_seed=train_seed,
            data_order_seed=data_order_seed,
        )
        if int(round_index) == 0:
            capture.setdefault("snapshots", {})[int(node_id)] = clone_state_dict(snapshot)
            capture.setdefault("trained", {})[int(node_id)] = clone_state_dict(result.state)
        return result

    return trainer


# --------------------------------------------------------------------------- #
# Checked-in config contract
# --------------------------------------------------------------------------- #
def test_checked_in_poison_config_is_explicit_and_offline() -> None:
    config = load_config(POISON_CONFIG)

    assert config.attack.compromised_node == 0
    assert config.attack.intensity == "strong"
    assert config.attack.kappa == 5.0
    assert config.data.download is False
    assert config.topology.kind == "chain3"
    # The checked-in YAML spells out every field rather than relying on defaults.
    raw = POISON_CONFIG.read_text(encoding="utf-8")
    for field in (
        "seed:",
        "rounds:",
        "device:",
        "bitwise:",
        "model:",
        "data:",
        "training:",
        "topology:",
        "attack:",
    ):
        assert field in raw


# --------------------------------------------------------------------------- #
# Zero attack is the honest path, byte for byte
# --------------------------------------------------------------------------- #
def test_selected_zero_node_matches_honest_episode_exactly(tmp_path: Path) -> None:
    honest = load_config(CLEAN_CONFIG)
    zero = _zero_variant(honest)
    dataset = _fixture()

    honest_log = tmp_path / "honest.jsonl"
    zero_log = tmp_path / "zero.jsonl"
    honest_result = run_from_config(honest, log_path=honest_log, dataset=dataset)
    zero_result = run_from_config(zero, log_path=zero_log, dataset=dataset)

    # Identical canonical JSONL bytes across independent executions.
    assert honest_log.read_bytes() == zero_log.read_bytes()
    # Identical final committed tensors and per-round records.
    _assert_states_equal(honest_result.final_states, zero_result.final_states)
    assert len(honest_result.rounds) == len(zero_result.rounds)
    for honest_round, zero_round in zip(honest_result.rounds, zero_result.rounds):
        assert canonical_record(honest_round) == canonical_record(zero_round)
        assert honest_round.messages == zero_round.messages
        assert honest_round.stats == zero_round.stats


# --------------------------------------------------------------------------- #
# Strong episode is deterministic and diverges from honest
# --------------------------------------------------------------------------- #
def test_strong_episode_is_reproducible_and_diverges_from_honest(tmp_path: Path) -> None:
    config = load_config(POISON_CONFIG)
    honest = load_config(CLEAN_CONFIG)
    dataset = _fixture()

    strong_a = tmp_path / "strong_a.jsonl"
    strong_b = tmp_path / "strong_b.jsonl"
    honest_log = tmp_path / "honest.jsonl"

    first = run_from_config(config, log_path=strong_a, dataset=dataset)
    second = run_from_config(config, log_path=strong_b, dataset=dataset)
    honest_result = run_from_config(honest, log_path=honest_log, dataset=dataset)

    # Deterministic: two independent strong executions are identical.
    assert strong_a.read_bytes() == strong_b.read_bytes()
    _assert_states_equal(first.final_states, second.final_states)
    assert [canonical_record(r) for r in first.rounds] == [
        canonical_record(r) for r in second.rounds
    ]

    # Divergent: a disconnected controller would leave the logs/states equal.
    assert strong_a.read_bytes() != honest_log.read_bytes()
    any_divergent = any(
        not torch.equal(first.final_states[node][key], honest_result.final_states[node][key])
        for node in NODES3
        for key in first.final_states[node]
    )
    assert any_divergent


# --------------------------------------------------------------------------- #
# Strong payload through the real aggregator seam
# --------------------------------------------------------------------------- #
def test_strong_payload_is_exact_formula_and_worsens_outgoing_model() -> None:
    config = load_config(POISON_CONFIG)
    topology = load_topology(config)
    dataset = _fixture()
    capture: dict = {}
    recorder = _InboxObserver()

    run_from_config(
        config,
        dataset=dataset,
        trainer=_capture_trainer(capture),
        observer=recorder,
    )

    # Node 1's only round-zero inbox entries are senders 0 and 2; sender 0's
    # payload is the poison, node 2's is honest (node 0 is the endpoint victim).
    inbox_round0 = recorder.inboxes[0]
    delivered_0_to_1 = inbox_round0[1][0]
    assert set(inbox_round0) == set(NODES3)

    snapshot_0 = capture["snapshots"][0]
    trained_0 = capture["trained"][0]
    kappa = config.attack.kappa

    # Exact D016 formula: u = w_tilde - kappa * (w_tilde - w), once per tensor.
    assert set(delivered_0_to_1) == set(trained_0)
    for key in sorted(trained_0):
        expected = trained_0[key] - kappa * (trained_0[key] - snapshot_0[key])
        assert torch.equal(delivered_0_to_1[key], expected), key
        assert bool(torch.isfinite(delivered_0_to_1[key]).all())
        # Delivered payload is an independent copy, not the trained/poison state.
        assert delivered_0_to_1[key].data_ptr() != trained_0[key].data_ptr()

    # Poison reaches only node 0's graph neighbor (node 1), never nodes 0 or 2.
    assert sorted(recorder.inboxes[0][0]) == [1]  # node 0 receives from node 1
    assert sorted(recorder.inboxes[0][2]) == [1]  # node 2 receives from node 1
    assert inbox_round0[1][0] is not None  # node 1 receives from node 0 (poison)

    # Measurably worse than the honest locally trained outgoing model on the
    # compromised node's own deterministic partition.
    seeds = resolve_seeds(config)
    partitions = partition_dataset(dataset, NODES3, SAMPLE_COUNT, seed=seeds.partition)
    honest_loss = _evaluate(config, trained_0, partitions[0])
    poison_loss = _evaluate(config, delivered_0_to_1, partitions[0])
    assert poison_loss > honest_loss, (
        f"strong reversal did not worsen loss: poison={poison_loss} honest={honest_loss}"
    )


def test_center_node_strong_broadcast_equal_to_both_neighbors() -> None:
    # Compromise the chain3 center (node 1, two neighbors) so one canonical
    # payload must fan out identically to nodes 0 and 2 without shared storage.
    config = make_config(attack={"compromised_node": 1, "intensity": "strong"})
    topology = load_topology(config)
    dataset = _fixture()
    recorder = _InboxObserver()

    run_from_config(config, dataset=dataset, observer=recorder)

    node0_payload = recorder.inboxes[0][0][1]
    node2_payload = recorder.inboxes[0][2][1]
    for key in sorted(node0_payload):
        assert torch.equal(node0_payload[key], node2_payload[key])
        assert node0_payload[key].data_ptr() != node2_payload[key].data_ptr()

    # No cross-talk: node 0 and node 2 each receive exactly the center's poison.
    assert sorted(recorder.inboxes[0][0]) == [1]
    assert sorted(recorder.inboxes[0][2]) == [1]


# --------------------------------------------------------------------------- #
# Canonical log shape, minimality, and isolation
# --------------------------------------------------------------------------- #
def test_fixed_episode_log_shape_minimality_and_isolation(tmp_path: Path) -> None:
    config = load_config(POISON_CONFIG)
    log = tmp_path / "poison.jsonl"
    result = run_from_config(config, log_path=log, dataset=_fixture())

    data = log.read_bytes()
    assert data.endswith(b"\n")
    records = _read_records(log)
    assert len(records) == config.rounds == len(result.rounds)

    for index, record in enumerate(records):
        assert set(record) == {"round", "nodes", "messages"}
        assert record["round"] == index
        assert [entry["node"] for entry in record["nodes"]] == list(NODES3)
        for entry in record["nodes"]:
            assert set(entry) == {"node", "pre_loss", "post_loss", "sample_count"}
            assert entry["sample_count"] == SAMPLE_COUNT
        assert {
            message["sender"]: message["recipients"] for message in record["messages"]
        } == CHAIN3_RECIPIENTS
        # Senders are sorted and each recipient list is ascending.
        senders = [message["sender"] for message in record["messages"]]
        assert senders == sorted(senders)
        for message in record["messages"]:
            assert message["recipients"] == sorted(message["recipients"])

    # The minimal log carries no attack metadata or tensor payloads.
    text = log.read_text(encoding="utf-8")
    for forbidden in ("attack", "kappa", "compromised", "intensity", "tensor"):
        assert forbidden not in text

    # Complete sorted edge summaries in every committed round.
    for round_result in result.rounds:
        assert [message.sender for message in round_result.messages] == [0, 1, 2]
        for message in round_result.messages:
            assert list(message.recipients) == sorted(message.recipients)

    # Committed tensors are finite and independently stored across nodes.
    final = result.final_states
    for node in NODES3:
        for key, value in final[node].items():
            assert bool(torch.isfinite(value).all()), (node, key)
    for left in NODES3:
        for right in NODES3:
            if left >= right:
                continue
            for key in final[left]:
                assert final[left][key].data_ptr() != final[right][key].data_ptr()

    # Round-zero states share no storage with the final states.
    round0 = result.rounds[0].states
    for node in NODES3:
        for key in round0[node]:
            assert round0[node][key].data_ptr() != final[node][key].data_ptr()

    # Mutating a caller-visible final state cannot leak into the round record.
    before = round0[0]["features.0.weight"].clone()
    final[0]["features.0.weight"].add_(1000.0)
    assert torch.equal(round0[0]["features.0.weight"], before)


# --------------------------------------------------------------------------- #
# Q5: failure paths publish no partial log, preserve a prior complete one
# --------------------------------------------------------------------------- #
def test_controller_failure_publishes_no_partial_log(tmp_path: Path) -> None:
    config = load_config(POISON_CONFIG)
    log = tmp_path / "poison.jsonl"

    class RaisingController:
        def outgoing_state(
            self, *, sender, round_start, trained, round_index=None, context=""
        ):
            if int(sender) == 0 and int(round_index or 0) == 0:
                raise AttackError("injected controller failure")
            return {key: value.detach().clone() for key, value in trained.items()}

    with pytest.raises(EpisodeError, match="round 0 failed"):
        run_from_config(
            config,
            log_path=log,
            dataset=_fixture(),
            controller=RaisingController(),
        )
    assert not log.exists()
    assert list(tmp_path.iterdir()) == []


def test_episode_failure_preserves_prior_complete_log(tmp_path: Path) -> None:
    config = load_config(POISON_CONFIG)
    log = tmp_path / "poison.jsonl"
    log.write_bytes(b"PRIOR-COMPLETE\n")
    capture: dict = {}

    with pytest.raises(EpisodeError, match="round 1 failed"):
        run_from_config(
            config,
            log_path=log,
            dataset=_fixture(),
            trainer=_capture_trainer(capture, fail_round=1),
        )
    assert log.read_bytes() == b"PRIOR-COMPLETE\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["poison.jsonl"]
