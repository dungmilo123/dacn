"""Contract tests for the versioned trajectory store and strict reader (R009/D003).

Covers canonical event serialization, deterministic byte-identical NPZ/ZIP
output, fixed-cadence snapshot selection, atomic non-overwriting publication,
manifest integrity, strict S05 read-back, and the Q3/Q5/Q6/Q7 threat, failure,
load, and negative surfaces. Everything is offline and in-process.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from dfl_sim import (
    ARTIFACT_FILENAMES,
    CONFIG_FILENAME,
    DECLARED_FILENAMES,
    EVENTS_FILENAME,
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    TENSORS_FILENAME,
    AttackAction,
    EvaluatedRound,
    LocalTrainResult,
    NodeEvaluation,
    PropagationState,
    TraceReader,
    TraceWriter,
    TrajectoryError,
    build_evaluator,
    canonical_event,
    digest_inventory,
    entry_name,
    load_topology,
    resolve_seeds,
    run_round,
    trace_inventory,
    validate_trace_path,
    write_trace,
)
from helpers import constant_loss_evaluator_factory, make_config

NODES3 = (0, 1, 2)


# --------------------------------------------------------------------------- #
# Tiny deterministic offline fixtures
# --------------------------------------------------------------------------- #
class TinyNet(nn.Module):
    def __init__(self, in_features: int = 2, num_classes: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class TinyDataset(Dataset):
    def __init__(self, num_examples: int = 16, *, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(int(seed))
        self.features = torch.rand(num_examples, 2, generator=generator)
        self.labels = torch.arange(num_examples, dtype=torch.long) % 2

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def _tiny_state(seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = TinyNet()
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _tiny_trainer(
    snapshot, dataset, *, config, node_id, round_index, train_seed, data_order_seed
):
    state = {key: value.detach().clone() for key, value in snapshot.items()}
    state["linear.bias"] = state["linear.bias"] + 0.05 * float(node_id)
    return LocalTrainResult(
        node_id=node_id,
        round_index=round_index,
        state=state,
        pre_loss=1.0,
        post_loss=0.5,
        sample_count=4,
    )


def _config(*, rounds: int = 2, cadence: int = 1, **overrides):
    base = {
        "rounds": rounds,
        "topology": {"kind": "chain3"},
        "aggregation": {"kind": "plain_average"},
        "evaluation": {"max_samples": 8, "batch_size": 4},
        "trace": {"snapshot_cadence": cadence},
    }
    base.update(overrides)
    return make_config(**base)


def _episode(config, *, rounds: int | None = None):
    """Drive the real runner + evaluator and return (topology, initial, rounds)."""

    total = config.rounds if rounds is None else rounds
    topology = load_topology(config)
    nodes = tuple(sorted(int(node) for node in topology.nodes()))
    seeds = resolve_seeds(config)
    initial = {node: _tiny_state() for node in nodes}
    current = {node: {k: v.clone() for k, v in state.items()} for node, state in initial.items()}
    evaluator = build_evaluator(
        config, TinyDataset(16), seeds, topology=topology, model_factory=TinyNet
    )
    rounds: list[tuple[EvaluatedRound, object]] = []
    for index in range(total):
        result = run_round(
            config=config,
            topology=topology,
            snapshots=current,
            partitions={node: None for node in nodes},
            round_index=index,
            seeds=seeds,
            trainer=_tiny_trainer,
            loss_evaluator=constant_loss_evaluator_factory,
        )
        rounds.append((evaluator.evaluate_round(result), result))
        current = result.states
    return topology, initial, rounds


def _write(config, tmp_path: Path, name: str = "trace"):
    topology, initial, rounds = _episode(config)
    destination = tmp_path / name
    write_trace(
        destination,
        config=config,
        topology=topology,
        initial_states=initial,
        rounds=rounds,
    )
    return destination, topology, initial, rounds


def _node_eval(node_id: int) -> NodeEvaluation:
    return NodeEvaluation(
        node_id=node_id,
        loss=0.5,
        accuracy=0.5,
        sample_count=4,
        true_type="benign",
        attack=AttackAction(kind="none", kappa=0.0),
        participant_ids=(node_id,),
        multipliers=(1.0,),
        weights=(1.0,),
        propagation=PropagationState(reached=False, min_hop=None, first_reached_round=None),
    )


def _evaluated(index: int, nodes) -> EvaluatedRound:
    return EvaluatedRound(
        round_index=index,
        nodes=tuple(_node_eval(node) for node in sorted(nodes)),
    )


# --------------------------------------------------------------------------- #
# Tamper helpers
# --------------------------------------------------------------------------- #
def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def _read_manifest(trace: Path) -> dict:
    return json.loads((trace / MANIFEST_FILENAME).read_text())


def _write_manifest(trace: Path, manifest: dict) -> None:
    (trace / MANIFEST_FILENAME).write_text(_canonical_json(manifest))


def _replace_file(trace: Path, filename: str, data: bytes, *, mutate=None) -> None:
    (trace / filename).write_bytes(data)
    manifest = _read_manifest(trace)
    manifest["artifacts"][filename] = {
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    if mutate is not None:
        mutate(manifest)
    _write_manifest(trace, manifest)


def _rebuild_zip(trace: Path, transform) -> None:
    with zipfile.ZipFile(trace / TENSORS_FILENAME) as archive:
        entries = [(info.filename, archive.read(info.filename)) for info in archive.infolist()]
    entries = transform(entries)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as out:
        for name, data in entries:
            info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            out.writestr(info, data)
    _replace_file(
        trace,
        TENSORS_FILENAME,
        buffer.getvalue(),
        mutate=lambda manifest: manifest.__setitem__("tensor_entry_count", len(entries)),
    )


def _object_npy() -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.array([{"value": 1}], dtype=object), allow_pickle=True)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Canonical serialization / helpers
# --------------------------------------------------------------------------- #
def test_entry_name_is_canonical_and_padded() -> None:
    assert entry_name(round_index=None, node_id=3, param="linear.weight") == (
        "initial/node_000003/linear.weight.npy"
    )
    assert entry_name(round_index=7, node_id=0, param="a") == "round_000007/node_000000/a.npy"


@pytest.mark.parametrize("param", ["../evil", "a/b", "a\\b", "a\x00b", ".hidden", "a..b", ""])
def test_entry_name_rejects_unsafe_parameters(param: str) -> None:
    with pytest.raises(TrajectoryError):
        entry_name(round_index=0, node_id=0, param=param)


def test_canonical_event_shape_and_messages_order() -> None:
    reviewed = _evaluated(0, NODES3)

    class Message:
        def __init__(self, sender, recipients):
            self.sender = sender
            self.recipients = recipients

    event = canonical_event(reviewed, [Message(2, [1]), Message(0, [1, 2])])
    assert event["schema_version"] == SCHEMA_VERSION
    assert [item["sender"] for item in event["messages"]] == [0, 2]
    assert event["messages"][0]["recipients"] == [1, 2]
    assert event["nodes"][0]["node"] == 0


def test_canonical_event_rejects_duplicate_message_sender() -> None:
    class Message:
        sender = 1
        recipients = (0,)

    with pytest.raises(TrajectoryError, match="duplicate message"):
        canonical_event(_evaluated(0, NODES3), [Message(), Message()])


# --------------------------------------------------------------------------- #
# Writer/reader round trip
# --------------------------------------------------------------------------- #
def test_round_trip_writer_reader(tmp_path: Path) -> None:
    config = _config(rounds=2, cadence=1)
    trace, topology, initial, rounds = _write(config, tmp_path)

    with TraceReader(trace) as reader:
        assert reader.schema_version == SCHEMA_VERSION
        assert reader.nodes == NODES3
        assert reader.edges == ((0, 1), (1, 2))
        assert reader.node_types == {0: "benign", 1: "benign", 2: "benign"}
        assert reader.snapshot_rounds == (0, 1)
        assert reader.has_initial_snapshot is True
        assert len(reader.events) == 2
        assert reader.events[0]["round"] == 0

        initial_read = reader.read_initial_snapshot()
        assert set(initial_read) == set(NODES3)
        for node in NODES3:
            for param, value in initial[node].items():
                assert np.array_equal(initial_read[node][param], value.numpy())

        round_one = reader.read_round_snapshot(1)
        assert set(round_one) == set(NODES3)
        committed = rounds[1][1].states
        for node in NODES3:
            for param, value in committed[node].items():
                assert np.array_equal(round_one[node][param], value.numpy())

        assert reader.topology().number_of_edges() == 2


def test_events_match_source_evaluated_rounds(tmp_path: Path) -> None:
    config = _config(rounds=2, cadence=1)
    trace, _topology, _initial, rounds = _write(config, tmp_path)
    with TraceReader(trace) as reader:
        for index, (evaluated, result) in enumerate(rounds):
            assert reader.events[index] == canonical_event(evaluated, result.messages)


def test_config_snapshot_round_trips_and_is_canonical(tmp_path: Path) -> None:
    config = _config(rounds=1, cadence=1)
    trace, _topology, _initial, _rounds = _write(config, tmp_path)
    with TraceReader(trace) as reader:
        assert reader.config == config
    text = (trace / CONFIG_FILENAME).read_text(encoding="utf-8")
    assert text.endswith("\n")


def test_trace_is_byte_identical_across_independent_writes(tmp_path: Path) -> None:
    config = _config(rounds=2, cadence=1)
    topology, initial, rounds = _episode(config)
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_trace(first, config=config, topology=topology, initial_states=initial, rounds=rounds)
    write_trace(second, config=config, topology=topology, initial_states=initial, rounds=rounds)
    assert trace_inventory(first) == trace_inventory(second)
    assert digest_inventory(first) == digest_inventory(second)
    for filename in DECLARED_FILENAMES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_manifest_artifacts_match_files_and_exclude_manifest(tmp_path: Path) -> None:
    trace, _topology, _initial, _rounds = _write(_config(rounds=1, cadence=1), tmp_path)
    manifest = _read_manifest(trace)
    assert set(manifest["artifacts"]) == set(ARTIFACT_FILENAMES)
    assert MANIFEST_FILENAME not in manifest["artifacts"]
    for filename in ARTIFACT_FILENAMES:
        raw = (trace / filename).read_bytes()
        entry = manifest["artifacts"][filename]
        assert entry["size_bytes"] == len(raw)
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["completed"] is True
    assert manifest["snapshot_count"] == 1 + len(manifest["snapshot_rounds"])


# --------------------------------------------------------------------------- #
# Cadence and load profile (Q6)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cadence,expected",
    [
        (1, (0, 1, 2, 3, 4)),
        (2, (0, 2, 4)),
        (3, (0, 3, 4)),
        (10, (0, 4)),
    ],
)
def test_cadence_selects_snapshots_and_always_includes_final(
    tmp_path: Path, cadence: int, expected: tuple[int, ...]
) -> None:
    config = _config(rounds=5, cadence=cadence)
    trace, _topology, _initial, _rounds = _write(config, tmp_path)
    with TraceReader(trace) as reader:
        assert reader.snapshot_rounds == expected
        entries = reader.entry_names()
        # two params per node per (initial + selected rounds)
        assert len(entries) == 3 * 2 * (1 + len(expected))


def test_bounded_twenty_node_synthetic_write(tmp_path: Path) -> None:
    config = _config(
        rounds=2,
        cadence=2,
        topology={"kind": "watts_strogatz", "n": 20, "k": 2, "p": 0.0},
        aggregation={"kind": "local_only"},
    )
    topology = load_topology(config)
    nodes = tuple(sorted(int(node) for node in topology.nodes()))
    assert len(nodes) == 20

    states = {
        node: {"w": torch.arange(3, dtype=torch.float32) + float(node)}
        for node in nodes
    }
    evaluated = [_evaluated(index, nodes) for index in range(2)]
    destination = tmp_path / "twenty"
    with TraceWriter(destination, config=config, topology=topology) as writer:
        writer.write_initial(states)
        for index, record in enumerate(evaluated):
            writer.write_round(record, messages=(), states=states)

    with TraceReader(destination) as reader:
        assert reader.snapshot_rounds == (0, 1)
        # initial + round 0 + final round 1, one param per node each
        assert reader.tensor_entry_count == 20 * 3
        assert len(reader.entry_names()) == 20 * 3
        assert np.array_equal(reader.read_tensor(1, 19, "w"), states[19]["w"].numpy())


# --------------------------------------------------------------------------- #
# Atomic publication and failure modes (Q5)
# --------------------------------------------------------------------------- #
def test_publish_is_atomic_and_leaves_no_staging(tmp_path: Path) -> None:
    config = _config(rounds=1, cadence=1)
    trace, _topology, _initial, _rounds = _write(config, tmp_path)
    assert trace.is_dir()
    assert sorted(item.name for item in trace.iterdir()) == sorted(DECLARED_FILENAMES)
    assert [item for item in tmp_path.iterdir() if item.name.startswith(".trace.staging-")] == []


def test_interrupted_context_aborts_and_cleans_staging(tmp_path: Path) -> None:
    config = _config(rounds=2, cadence=1)
    topology, initial, _rounds = _episode(config)
    destination = tmp_path / "trace"
    with pytest.raises(RuntimeError, match="boom"):
        with TraceWriter(destination, config=config, topology=topology) as writer:
            writer.write_initial(initial)
            raise RuntimeError("boom")
    assert not destination.exists()
    assert [item for item in tmp_path.iterdir() if item.name.startswith(".trace.staging-")] == []


def test_commit_requires_all_expected_rounds(tmp_path: Path) -> None:
    config = _config(rounds=2, cadence=1)
    topology, initial, rounds = _episode(config)
    destination = tmp_path / "trace"
    writer = TraceWriter(destination, config=config, topology=topology)
    writer.__enter__()
    writer.write_initial(initial)
    writer.write_round(rounds[0][0], messages=rounds[0][1].messages, states=rounds[0][1].states)
    with pytest.raises(TrajectoryError, match="incomplete"):
        writer.commit()
    writer.abort()
    assert not destination.exists()
    assert [item for item in tmp_path.iterdir() if item.name.startswith(".trace.staging-")] == []


def test_duplicate_and_out_of_order_rounds_are_rejected(tmp_path: Path) -> None:
    config = _config(rounds=2, cadence=1)
    topology, initial, rounds = _episode(config)
    writer = TraceWriter(tmp_path / "trace", config=config, topology=topology)
    writer.__enter__()
    try:
        writer.write_initial(initial)
        with pytest.raises(TrajectoryError, match="out of order"):
            writer.write_round(
                rounds[1][0], messages=rounds[1][1].messages, states=rounds[1][1].states
            )
        writer.write_round(
            rounds[0][0], messages=rounds[0][1].messages, states=rounds[0][1].states
        )
        with pytest.raises(TrajectoryError, match="out of order"):
            writer.write_round(
                rounds[0][0], messages=rounds[0][1].messages, states=rounds[0][1].states
            )
    finally:
        writer.abort()


def test_snapshot_round_requires_committed_states(tmp_path: Path) -> None:
    config = _config(rounds=2, cadence=1)
    topology, initial, rounds = _episode(config)
    with pytest.raises(TrajectoryError, match="requires committed states"):
        with TraceWriter(tmp_path / "trace", config=config, topology=topology) as writer:
            writer.write_initial(initial)
            writer.write_round(rounds[0][0], messages=rounds[0][1].messages, states=None)


@pytest.mark.parametrize(
    "states",
    [
        {0: {"w": torch.full((2,), float("nan"))}},
        {0: {"w": torch.tensor([float("inf")])}},
        {0: {"../evil": torch.zeros(2)}},
        {0: {"w": np.array([{"x": 1}], dtype=object)}},
        {0: {}},
    ],
)
def test_writer_rejects_untrusted_tensor_states(tmp_path: Path, states) -> None:
    config = _config(rounds=1, cadence=1)
    topology = load_topology(config)
    with pytest.raises(TrajectoryError):
        with TraceWriter(tmp_path / "trace", config=config, topology=topology) as writer:
            writer.write_initial(states)


def test_writer_rejects_state_node_mismatch(tmp_path: Path) -> None:
    config = _config(rounds=1, cadence=1)
    topology = load_topology(config)
    states = {0: {"w": torch.zeros(2)}, 1: {"w": torch.zeros(2)}}
    with pytest.raises(TrajectoryError, match="node set does not match topology"):
        with TraceWriter(tmp_path / "trace", config=config, topology=topology) as writer:
            writer.write_initial(states)


def test_evaluated_round_node_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _config(rounds=2, cadence=1)
    topology = load_topology(config)
    states = {node: {"w": torch.zeros(2)} for node in NODES3}
    with pytest.raises(TrajectoryError, match="node set does not match"):
        with TraceWriter(tmp_path / "trace", config=config, topology=topology) as writer:
            writer.write_initial(states)
            writer.write_round(_evaluated(0, (0, 1)), messages=(), states=states)


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    config = _config(rounds=1, cadence=1)
    topology = load_topology(config)
    destination = tmp_path / "trace"
    destination.mkdir()
    (destination / "keep.txt").write_text("prior")
    with pytest.raises(TrajectoryError, match="already exists"):
        TraceWriter(destination, config=config, topology=topology)
    assert (destination / "keep.txt").read_text() == "prior"


def test_symlink_destination_is_rejected(tmp_path: Path) -> None:
    config = _config(rounds=1, cadence=1)
    topology = load_topology(config)
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    os.symlink(target, link)
    with pytest.raises(TrajectoryError, match="symlink"):
        TraceWriter(link, config=config, topology=topology)


def test_missing_parent_is_rejected(tmp_path: Path) -> None:
    config = _config(rounds=1, cadence=1)
    topology = load_topology(config)
    with pytest.raises(TrajectoryError, match="parent does not exist"):
        TraceWriter(tmp_path / "missing" / "trace", config=config, topology=topology)


def test_config_alias_destination_is_rejected(tmp_path: Path) -> None:
    config = _config(rounds=1, cadence=1)
    topology = load_topology(config)
    destination = tmp_path / "trace"
    with pytest.raises(TrajectoryError, match="alias"):
        validate_trace_path(destination, config_path=destination)
    with pytest.raises(TrajectoryError, match="alias"):
        TraceWriter(destination, config=config, topology=topology, config_path=destination)


# --------------------------------------------------------------------------- #
# Strict reader negatives (Q3/Q7)
# --------------------------------------------------------------------------- #
def test_reader_rejects_unknown_schema_version(tmp_path: Path) -> None:
    trace, _topology, _initial, _rounds = _write(_config(rounds=1, cadence=1), tmp_path)
    manifest = _read_manifest(trace)
    manifest["schema_version"] = SCHEMA_VERSION + 1
    _write_manifest(trace, manifest)
    with pytest.raises(TrajectoryError, match="unsupported trajectory schema"):
        TraceReader(trace)


def test_reader_rejects_tampered_artifact(tmp_path: Path) -> None:
    trace, _topology, _initial, _rounds = _write(_config(rounds=1, cadence=1), tmp_path)
    raw = bytearray((trace / EVENTS_FILENAME).read_bytes())
    raw[-2] ^= 0x01
    (trace / EVENTS_FILENAME).write_bytes(bytes(raw))
    with pytest.raises(TrajectoryError, match="digest mismatch"):
        TraceReader(trace)


def test_reader_rejects_extra_files(tmp_path: Path) -> None:
    trace, _topology, _initial, _rounds = _write(_config(rounds=1, cadence=1), tmp_path)
    (trace / "extra.txt").write_text("unexpected")
    with pytest.raises(TrajectoryError, match="layout mismatch"):
        TraceReader(trace)


def test_reader_rejects_noncanonical_event_order(tmp_path: Path) -> None:
    trace, _topology, _initial, _rounds = _write(_config(rounds=2, cadence=1), tmp_path)
    lines = (trace / EVENTS_FILENAME).read_text(encoding="utf-8").rstrip("\n").split("\n")
    reordered = ("\n".join([lines[1], lines[0]]) + "\n").encode("utf-8")
    _replace_file(trace, EVENTS_FILENAME, reordered)
    with pytest.raises(TrajectoryError, match="order"):
        TraceReader(trace)


def test_reader_rejects_noncanonical_config(tmp_path: Path) -> None:
    trace, _topology, _initial, _rounds = _write(_config(rounds=1, cadence=1), tmp_path)
    text = (trace / CONFIG_FILENAME).read_text(encoding="utf-8")
    pretty = json.dumps(json.loads(text), indent=2) + "\n"
    _replace_file(trace, CONFIG_FILENAME, pretty.encode("utf-8"))
    with pytest.raises(TrajectoryError, match="not canonically encoded"):
        TraceReader(trace)


def test_reader_rejects_duplicate_manifest_json_key(tmp_path: Path) -> None:
    trace, _topology, _initial, _rounds = _write(_config(rounds=1, cadence=1), tmp_path)
    text = (trace / MANIFEST_FILENAME).read_text(encoding="utf-8")
    tampered = text.replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1)
    (trace / MANIFEST_FILENAME).write_text(tampered)
    with pytest.raises(TrajectoryError, match="duplicate JSON key"):
        TraceReader(trace)


def test_reader_rejects_manifest_without_final_snapshot(tmp_path: Path) -> None:
    trace, _topology, _initial, _rounds = _write(_config(rounds=2, cadence=1), tmp_path)
    manifest = _read_manifest(trace)
    manifest["snapshot_rounds"] = manifest["snapshot_rounds"][:-1]
    _write_manifest(trace, manifest)
    with pytest.raises(TrajectoryError, match="final round"):
        TraceReader(trace)


def test_reader_rejects_unsafe_zip_entry(tmp_path: Path) -> None:
    trace, _topology, _initial, _rounds = _write(_config(rounds=1, cadence=1), tmp_path)
    _rebuild_zip(trace, lambda entries: entries + [("../evil.npy", b"junk")])
    with pytest.raises(TrajectoryError, match="unsafe|noncanonical"):
        TraceReader(trace)


def test_reader_rejects_duplicate_zip_entry(tmp_path: Path) -> None:
    trace, _topology, _initial, _rounds = _write(_config(rounds=1, cadence=1), tmp_path)
    _rebuild_zip(trace, lambda entries: entries + [entries[0]])
    with pytest.raises(TrajectoryError, match="duplicate entry"):
        TraceReader(trace)


def test_reader_rejects_object_array_requiring_pickle(tmp_path: Path) -> None:
    trace, _topology, _initial, _rounds = _write(_config(rounds=1, cadence=1), tmp_path)
    replaced: dict[str, str] = {}

    def transform(entries):
        output = []
        for position, (name, data) in enumerate(entries):
            if position == 0:
                replaced["name"] = name
                data = _object_npy()
            output.append((name, data))
        return output

    _rebuild_zip(trace, transform)
    match = re.match(r"initial/node_(\d+)/(.+)\.npy$", replaced["name"])
    assert match is not None
    with TraceReader(trace) as reader:
        with pytest.raises(TrajectoryError, match="pickle"):
            reader.read_tensor(None, int(match.group(1)), match.group(2))


def test_reader_rejects_unknown_snapshot_round(tmp_path: Path) -> None:
    trace, _topology, _initial, _rounds = _write(_config(rounds=1, cadence=1), tmp_path)
    with TraceReader(trace) as reader:
        with pytest.raises(TrajectoryError, match="not a recorded snapshot round"):
            reader.read_tensor(5, 0, "linear.weight")
        with pytest.raises(TrajectoryError, match="does not contain"):
            reader.read_tensor(None, 0, "missing.param")


# --------------------------------------------------------------------------- #
# Injected I/O failures leave no partial destination (Q5)
# --------------------------------------------------------------------------- #
def test_initialization_write_failure_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dfl_sim.trajectory as trajectory

    config = _config(rounds=1, cadence=1)
    topology = load_topology(config)

    def fail_write(path, payload):
        raise TrajectoryError("injected config write failure")

    monkeypatch.setattr(trajectory, "_write_bytes_synced", fail_write)
    destination = tmp_path / "trace"
    with pytest.raises(TrajectoryError, match="injected config write failure"):
        TraceWriter(destination, config=config, topology=topology).__enter__()
    assert not destination.exists()
    assert [item for item in tmp_path.iterdir() if item.name.startswith(".trace.staging-")] == []


def test_rename_failure_leaves_destination_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dfl_sim.trajectory as trajectory

    config = _config(rounds=1, cadence=1)
    topology, initial, rounds = _episode(config)
    destination = tmp_path / "trace"

    def fail_replace(source, target):
        raise OSError("injected rename failure")

    monkeypatch.setattr(trajectory.os, "replace", fail_replace)
    with pytest.raises(TrajectoryError, match="could not publish trace"):
        with TraceWriter(destination, config=config, topology=topology) as writer:
            writer.write_initial(initial)
            writer.write_round(
                rounds[0][0], messages=rounds[0][1].messages, states=rounds[0][1].states
            )
    assert not destination.exists()
    assert [item for item in tmp_path.iterdir() if item.name.startswith(".trace.staging-")] == []
