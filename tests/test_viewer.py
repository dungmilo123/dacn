"""Unit tests for the strict trace-to-replay projection (S05/T01, R011, D020).

Everything here runs the **real** production path: fixtures are published
through :func:`dfl_sim.write_trace` (evaluator/trajectory record APIs) and read
back through the strict :class:`~dfl_sim.trajectory.TraceReader` before the
viewer projects them. There is no network, dataset download, or hand-written
JSON manifest; tamper helpers only rewrite bytes that a real reader must catch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import pytest

from dfl_sim import (
    ATTACK_KAPPAS,
    ReplayModel,
    ViewerError,
    ViewerFrame,
    ViewerNode,
    clamp_round,
    cytoscape_message_edges,
    cytoscape_nodes,
    cytoscape_round_elements,
    cytoscape_static_elements,
    cytoscape_topology_edges,
    describe_round,
    describe_trace,
    load_replay,
    select_frame,
    select_frame_node,
    select_node,
    step_round,
)
from dfl_sim.viewer import NodeFrameState

from helpers import make_viewer_trace

# --------------------------------------------------------------------------- #
# Fixtures and tamper helpers
# --------------------------------------------------------------------------- #
BENIGN = "benign"
MALICIOUS = "malicious"


@pytest.fixture
def viewer_trace(tmp_path: Path) -> Path:
    return make_viewer_trace(tmp_path / "viewer_trace")


def _canonical_json(value: object) -> str:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    )


def _read_events(trace: Path) -> list[dict]:
    text = (trace / "events.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _write_events(trace: Path, events: list[dict]) -> None:
    """Rewrite events.jsonl canonically and refresh its manifest digest."""

    payload = "".join(_canonical_json(event) for event in events).encode("utf-8")
    (trace / "events.jsonl").write_bytes(payload)
    manifest = json.loads((trace / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["events.jsonl"] = {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (trace / "manifest.json").write_bytes(
        _canonical_json(manifest).encode("utf-8")
    )


def _write_manifest(trace: Path, manifest: dict) -> None:
    (trace / "manifest.json").write_bytes(
        _canonical_json(manifest).encode("utf-8")
    )


def _node_frame_state(node_id: int = 0) -> NodeFrameState:
    return NodeFrameState(
        node_id=node_id,
        loss=0.5,
        accuracy=0.5,
        sample_count=1,
        true_type=BENIGN,
        attack_kind="none",
        attack_kappa=0.0,
        participant_ids=(node_id,),
        multipliers=(1.0,),
        weights=(1.0,),
        propagation_reached=False,
        propagation_min_hop=None,
        propagation_first_reached_round=None,
    )


def _mini_model(rounds: tuple[int, ...]) -> ReplayModel:
    frames = tuple(
        ViewerFrame(round_index=index, nodes=(_node_frame_state(),), messages=())
        for index in rounds
    )
    breadth = len(rounds)
    return ReplayModel(
        trace_name="mini",
        schema_version=1,
        aggregation="plain_average",
        attack_kind="none",
        attack_kappa=0.0,
        compromised_node=None,
        nodes=(ViewerNode(0, BENIGN, 0.0, 0.0),),
        edges=(),
        frames=frames,
        loss_history=((0.5,) * breadth,),
        accuracy_history=((0.5,) * breadth,),
        snapshot_rounds=tuple(rounds),
    )


# --------------------------------------------------------------------------- #
# Real writer -> reader -> projection
# --------------------------------------------------------------------------- #
def test_projects_complete_schema_v1_trace(viewer_trace: Path) -> None:
    model = load_replay(viewer_trace)

    assert model.trace_name == "viewer_trace"
    assert model.schema_version == 1
    assert model.node_ids == (0, 1, 2)
    assert model.nodes[0].node_type == BENIGN
    assert model.nodes[1].node_type == BENIGN
    assert model.nodes[2].node_type == MALICIOUS
    assert model.edges == ((0, 1), (1, 2))
    assert model.rounds == (0, 1)
    assert model.first_round == 0
    assert model.last_round == 1
    assert model.attack_kind == "delta_reversal"
    assert model.attack_kappa == ATTACK_KAPPAS["weak"]
    assert model.compromised_node == 2
    assert model.aggregation == "plain_average"
    assert len(model.frames) == 2


def test_projection_never_reads_tensor_payloads(
    viewer_trace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dfl_sim.viewer as viewer

    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("viewer projection must not read tensor payloads")

    monkeypatch.setattr(viewer.TraceReader, "read_tensor", _boom)
    model = load_replay(viewer_trace)
    assert model.frames


def test_reader_closed_after_successful_projection(
    viewer_trace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dfl_sim.viewer as viewer
    from dfl_sim.trajectory import TraceReader as RealReader

    closed: list[bool] = []

    class RecordingReader(RealReader):
        def close(self) -> None:
            closed.append(True)
            super().close()

    monkeypatch.setattr(viewer, "TraceReader", RecordingReader)
    model = load_replay(viewer_trace)

    assert model.frames
    assert closed == [True]


def test_reader_closed_after_failed_projection(
    viewer_trace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dfl_sim.viewer as viewer
    from dfl_sim.trajectory import TraceReader as RealReader

    closed: list[bool] = []

    class RecordingReader(RealReader):
        def close(self) -> None:
            closed.append(True)
            super().close()

    monkeypatch.setattr(viewer, "TraceReader", RecordingReader)
    events = _read_events(viewer_trace)
    events[0]["nodes"][-1]["node"] = 99
    events[0]["nodes"][-1]["profile"]["participants"][0] = 99
    _write_events(viewer_trace, events)

    with pytest.raises(ViewerError):
        load_replay(viewer_trace)
    assert closed == [True]


# --------------------------------------------------------------------------- #
# Node types, topology, and message overlays
# --------------------------------------------------------------------------- #
def test_cytoscape_node_classes_and_stable_ids(viewer_trace: Path) -> None:
    model = load_replay(viewer_trace)
    nodes = cytoscape_nodes(model)

    assert [element["data"]["id"] for element in nodes] == [
        "node-0",
        "node-1",
        "node-2",
    ]
    assert [element["classes"] for element in nodes] == [
        BENIGN,
        BENIGN,
        MALICIOUS,
    ]
    assert nodes[2]["data"]["true_type"] == MALICIOUS
    assert all("position" in element for element in nodes)


def test_topology_edges_are_separate_undirected_class(viewer_trace: Path) -> None:
    model = load_replay(viewer_trace)
    edges = cytoscape_topology_edges(model)

    assert [element["data"]["id"] for element in edges] == [
        "topology-0-1",
        "topology-1-2",
    ]
    assert all(element["classes"] == "topology" for element in edges)
    assert edges[0]["data"]["kind"] == "topology"


def test_message_edges_are_directed_and_separately_classed(viewer_trace: Path) -> None:
    model = load_replay(viewer_trace)
    frame = select_frame(model, 0)
    messages = cytoscape_message_edges(frame)

    # chain3 broadcasts: 0->1, 1->0, 1->2, 2->1
    assert [(m["data"]["sender"], m["data"]["recipient"]) for m in messages] == [
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    ]
    assert all(element["classes"] == "message" for element in messages)
    assert all(element["data"]["kind"] == "message" for element in messages)
    assert messages[0]["data"]["source"] == "node-0"
    assert messages[0]["data"]["target"] == "node-1"
    assert frame.message_count == 4


def test_static_and_round_elements_compose(viewer_trace: Path) -> None:
    model = load_replay(viewer_trace)
    static_ids = {
        element["data"]["id"] for element in cytoscape_static_elements(model)
    }
    round_ids = {
        element["data"]["id"] for element in cytoscape_round_elements(model, 1)
    }

    assert static_ids == {"node-0", "node-1", "node-2", "topology-0-1", "topology-1-2"}
    assert static_ids < round_ids
    assert round_ids - static_ids == {
        "message-1-0-1",
        "message-1-1-0",
        "message-1-1-2",
        "message-1-2-1",
    }


# --------------------------------------------------------------------------- #
# Metric histories and per-node frame state
# --------------------------------------------------------------------------- #
def test_metric_history_ordering(viewer_trace: Path) -> None:
    model = load_replay(viewer_trace)

    assert model.loss_series(0) == (0.90, 0.70)
    assert model.loss_series(1) == (0.80, 0.60)
    assert model.loss_series(2) == (1.50, 1.30)
    assert model.accuracy_series(0) == (0.30, 0.50)
    assert model.accuracy_series(1) == (0.40, 0.55)
    assert model.accuracy_series(2) == (0.20, 0.25)
    assert model.node_index(2) == 2


def test_frame_carries_attack_profile_and_propagation(viewer_trace: Path) -> None:
    model = load_replay(viewer_trace)

    hidden = select_frame_node(model, 0, 2)
    assert hidden.true_type == MALICIOUS
    assert hidden.attack_kind == "delta_reversal"
    assert hidden.attack_kappa == ATTACK_KAPPAS["weak"]
    assert hidden.participant_ids == (2, 1)
    assert hidden.propagation_reached is True
    assert hidden.propagation_min_hop == 0
    assert hidden.propagation_first_reached_round == 0

    far = select_frame_node(model, 0, 0)
    assert far.true_type == BENIGN
    assert far.attack_kind == "none"
    assert far.participant_ids == (0, 1)
    assert far.propagation_min_hop == 2


def test_describe_helpers_are_deterministic(viewer_trace: Path) -> None:
    model = load_replay(viewer_trace)

    summary = describe_trace(model)
    assert "viewer_trace" in summary
    assert "3 node(s) [1 malicious]" in summary
    assert "aggregation=plain_average" in summary
    assert "delta_reversal on node 2" in summary

    status = describe_round(model, 0)
    assert "round 0 of 1" in status
    assert "4 directed message(s) from 3 sender(s)" in status


# --------------------------------------------------------------------------- #
# Selection, bounds, and stepping
# --------------------------------------------------------------------------- #
def test_selection_is_bounds_checked(viewer_trace: Path) -> None:
    model = load_replay(viewer_trace)

    assert select_frame(model, 1).round_index == 1
    assert select_node(model, 2).node_id == 2
    assert select_frame_node(model, 1, 1).node_id == 1

    with pytest.raises(ViewerError):
        select_frame(model, 2)
    with pytest.raises(ViewerError):
        select_node(model, 99)
    with pytest.raises(ViewerError):
        select_frame_node(model, 0, 99)
    with pytest.raises(ViewerError):
        select_frame(model, "not-a-round")
    with pytest.raises(ViewerError):
        select_node(model, True)


def test_stepping_clamps_at_both_boundaries(viewer_trace: Path) -> None:
    model = load_replay(viewer_trace)

    assert step_round(model, 0, -1) == 0
    assert step_round(model, 0, 1) == 1
    assert step_round(model, 1, 1) == 1
    assert step_round(model, 1, -1) == 0
    assert clamp_round(model, -10) == 0
    assert clamp_round(model, 10) == 1


def test_stepping_preserves_noncontiguous_actual_rounds() -> None:
    model = _mini_model((5, 9))

    assert step_round(model, 5, 1) == 9
    assert step_round(model, 9, 1) == 9
    assert step_round(model, 9, -1) == 5
    assert step_round(model, 5, -1) == 5
    assert clamp_round(model, 5) == 5
    assert clamp_round(model, 100) == 9


# --------------------------------------------------------------------------- #
# Boundary shapes: single round, single node, research-scale node count
# --------------------------------------------------------------------------- #
def test_single_round_trace_has_no_movement(tmp_path: Path) -> None:
    trace = make_viewer_trace(tmp_path / "one_round", rounds=1)
    model = load_replay(trace)

    assert model.rounds == (0,)
    assert step_round(model, 0, 1) == 0
    assert step_round(model, 0, -1) == 0
    assert model.loss_series(0) == (0.90,)
    assert model.accuracy_series(2) == (0.20,)


def test_single_node_geometry_is_valid(tmp_path: Path) -> None:
    graph = nx.Graph()
    graph.add_node(0)
    trace = make_viewer_trace(
        tmp_path / "single", rounds=1, graph=graph, compromised_node=0
    )
    model = load_replay(trace)

    assert model.node_ids == (0,)
    assert (model.nodes[0].x, model.nodes[0].y) == (0.0, 0.0)
    assert model.edges == ()
    frame = select_frame(model, 0)
    assert frame.messages == ()
    assert frame.message_count == 0
    assert cytoscape_topology_edges(model) == []


def test_projects_research_scale_node_count(tmp_path: Path) -> None:
    graph = nx.path_graph(20)
    trace = make_viewer_trace(
        tmp_path / "scale", rounds=2, graph=graph, compromised_node=2
    )
    model = load_replay(trace)

    assert len(model.nodes) == 20
    assert len(model.edges) == 19
    assert model.rounds == (0, 1)
    assert all(len(series) == 2 for series in model.loss_history)
    assert model.node_types[2] == MALICIOUS


# --------------------------------------------------------------------------- #
# Negative: missing, incomplete, unsupported, tampered, malformed cross-refs
# --------------------------------------------------------------------------- #
def test_missing_trace_fails_closed_and_is_path_safe(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ViewerError) as info:
        load_replay(missing)

    message = str(info.value)
    assert "does_not_exist" in message
    assert str(tmp_path) not in message
    assert info.value.__cause__ is not None


def test_empty_directory_fails_closed(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ViewerError):
        load_replay(empty)


def test_incomplete_trace_fails_closed(viewer_trace: Path) -> None:
    (viewer_trace / "manifest.json").unlink()
    with pytest.raises(ViewerError):
        load_replay(viewer_trace)


def test_unsupported_schema_fails_closed(viewer_trace: Path) -> None:
    manifest = json.loads((viewer_trace / "manifest.json").read_text())
    manifest["schema_version"] = 99
    _write_manifest(viewer_trace, manifest)
    with pytest.raises(ViewerError, match="unsupported trajectory schema"):
        load_replay(viewer_trace)


def test_tampered_artifact_digest_fails_closed(viewer_trace: Path) -> None:
    events_path = viewer_trace / "events.jsonl"
    events_path.write_bytes(events_path.read_bytes() + b" ")
    with pytest.raises(ViewerError):
        load_replay(viewer_trace)


def test_unknown_node_reference_fails_closed(viewer_trace: Path) -> None:
    events = _read_events(viewer_trace)
    events[0]["nodes"][-1]["node"] = 99
    events[0]["nodes"][-1]["profile"]["participants"][0] = 99
    _write_events(viewer_trace, events)

    with pytest.raises(ViewerError) as info:
        load_replay(viewer_trace)
    assert "round 0" in str(info.value)


def test_type_mismatch_fails_closed(viewer_trace: Path) -> None:
    events = _read_events(viewer_trace)
    events[0]["nodes"][0]["true_type"] = MALICIOUS
    _write_events(viewer_trace, events)

    with pytest.raises(ViewerError, match="true_type"):
        load_replay(viewer_trace)


def test_unknown_message_target_fails_closed(viewer_trace: Path) -> None:
    events = _read_events(viewer_trace)
    events[0]["messages"][0]["recipients"].append(99)
    _write_events(viewer_trace, events)

    with pytest.raises(ViewerError) as info:
        load_replay(viewer_trace)
    assert "unknown node 99" in str(info.value)
