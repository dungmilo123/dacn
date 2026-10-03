"""Endpoint and callback tests for the Dash replay app (S05/T02, R011, D020).

These tests build the *real* Dash application from a production-format schema-v1
trace fixture and exercise:
* the Flask test client surfaces the browser actually fetches (``/``,
  ``/_dash-layout``, ``/_dash-dependencies``);
* the pure callback reducers/redraws that back every Dash callback.

No browser binary is required here: this is the ordinary offline suite. The
Playwright smoke that proves real hydration lives in ``test_viewer_browser.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from dfl_sim import (
    ReplayModel,
    ViewerError,
    build_accuracy_figure,
    build_loss_figure,
    coerce_node_id,
    coerce_round,
    create_viewer_app,
    cytoscape_stylesheet,
    load_replay,
    render_view,
    resolve_round_action,
)
from dfl_sim.viewer import (
    LEGEND_ID,
    NETWORK_ID,
    NEXT_BUTTON_ID,
    NODE_SELECTOR_ID,
    PREV_BUTTON_ID,
    ROUND_LABEL_ID,
    ROUND_SLIDER_ID,
    ROUND_STATUS_ID,
    TRACE_SUMMARY_ID,
    VIEWER_COMPONENT_IDS,
)

from helpers import make_viewer_trace

BENIGN = "benign"
MALICIOUS = "malicious"


# --------------------------------------------------------------------------- #
# Fixtures and layout helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def viewer_trace(tmp_path: Path) -> Path:
    return make_viewer_trace(tmp_path / "viewer_trace")


@pytest.fixture
def model(viewer_trace: Path) -> ReplayModel:
    return load_replay(viewer_trace)


@pytest.fixture
def app(viewer_trace: Path):
    return create_viewer_app(viewer_trace)


def _find(node, target_id):
    """Recursively locate a Dash component node by its ``props.id``."""

    if isinstance(node, dict):
        if node.get("props", {}).get("id") == target_id:
            return node
        for value in node.values():
            found = _find(value, target_id)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find(item, target_id)
            if found is not None:
                return found
    return None


def _layout(client) -> dict:
    response = client.get("/_dash-layout")
    assert response.status_code == 200
    return json.loads(response.get_data(as_text=True))


def _dependencies(client) -> list[dict]:
    response = client.get("/_dash-dependencies")
    assert response.status_code == 200
    payload = json.loads(response.get_data(as_text=True))
    assert isinstance(payload, list)
    return payload


# --------------------------------------------------------------------------- #
# Endpoint surfaces
# --------------------------------------------------------------------------- #
def test_index_served(app) -> None:
    client = app.server.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("Content-Type", "")
    body = response.get_data(as_text=True)
    assert "<!DOCTYPE html>" in body
    # The app title is wired through and the Dash renderer shell is served.
    assert "dfl replay: viewer_trace" in body
    assert "DashRenderer" in body


def test_layout_exposes_every_required_component(app) -> None:
    layout = _layout(app.server.test_client())

    for component_id in VIEWER_COMPONENT_IDS:
        assert _find(layout, component_id) is not None, component_id


def test_layout_summary_and_legend_are_readable(app) -> None:
    layout = _layout(app.server.test_client())

    summary = _find(layout, TRACE_SUMMARY_ID)
    assert "viewer_trace" in summary["props"]["children"]
    assert "3 node(s) [1 malicious]" in summary["props"]["children"]

    legend_text = json.dumps(_find(layout, LEGEND_ID))
    assert "benign node" in legend_text
    assert "malicious node" in legend_text


def test_layout_initial_state_targets_first_round(app) -> None:
    layout = _layout(app.server.test_client())

    slider = _find(layout, ROUND_SLIDER_ID)["props"]
    assert slider["min"] == 0
    assert slider["max"] == 1
    assert slider["value"] == 0
    assert slider["marks"] == {"0": "0", "1": "1"}

    status = _find(layout, ROUND_STATUS_ID)["props"]
    assert "round 0 of 1" in status["children"]
    assert "4 directed message(s) from 3 sender(s)" in status["children"]

    label = _find(layout, ROUND_LABEL_ID)["props"]
    assert label["children"] == "round 0 of 1"

    selector = _find(layout, NODE_SELECTOR_ID)["props"]
    assert selector["value"] == 0
    assert selector["options"] == [
        {"label": "node 0", "value": 0},
        {"label": "node 1", "value": 1},
        {"label": "node 2", "value": 2},
    ]


def test_layout_network_has_static_and_round_flow(app) -> None:
    layout = _layout(app.server.test_client())
    network = _find(layout, NETWORK_ID)
    assert network["type"] == "Cytoscape"

    elements = network["props"]["elements"]
    classes = {element["classes"] for element in elements}
    assert classes == {BENIGN, MALICIOUS, "topology", "message"}

    message_ids = {
        element["data"]["id"]
        for element in elements
        if element["classes"] == "message"
    }
    assert message_ids == {
        "message-0-0-1",
        "message-0-1-0",
        "message-0-1-2",
        "message-0-2-1",
    }


def test_layout_stylesheet_separates_types_and_edges(app) -> None:
    layout = _layout(app.server.test_client())
    stylesheet = _find(layout, NETWORK_ID)["props"]["stylesheet"]
    selectors = {rule["selector"] for rule in stylesheet}

    assert {"node.benign", "node.malicious", "edge.topology", "edge.message"} <= selectors
    malicious = next(
        rule for rule in stylesheet if rule["selector"] == "node.malicious"
    )
    benign = next(rule for rule in stylesheet if rule["selector"] == "node.benign")
    assert malicious["style"]["background-color"] != benign["style"]["background-color"]
    assert malicious["style"]["shape"] != benign["style"]["shape"]

    message_edge = next(
        rule for rule in stylesheet if rule["selector"] == "edge.message"
    )
    assert message_edge["style"]["target-arrow-shape"] == "triangle"


def test_dependencies_wire_round_and_redraw_callbacks(app) -> None:
    dependencies = _dependencies(app.server.test_client())
    assert len(dependencies) == 2

    step = next(
        dep for dep in dependencies if dep["output"] == f"{ROUND_SLIDER_ID}.value"
    )
    assert {item["id"] for item in step["inputs"]} == {
        PREV_BUTTON_ID,
        NEXT_BUTTON_ID,
    }
    assert step["state"] == [{"id": ROUND_SLIDER_ID, "property": "value"}]
    assert step["prevent_initial_call"] is True

    redraw = next(dep for dep in dependencies if dep is not step)
    output_ids = [part for part in redraw["output"].split(".") if part]
    for part in (NETWORK_ID, ROUND_STATUS_ID, ROUND_LABEL_ID):
        assert part in redraw["output"]
    assert {item["id"] for item in redraw["inputs"]} == {
        ROUND_SLIDER_ID,
        NODE_SELECTOR_ID,
    }


def test_every_dependency_output_id_is_a_known_component(app) -> None:
    known = set(VIEWER_COMPONENT_IDS)
    for dependency in _dependencies(app.server.test_client()):
        # Dash joins multi-output specs with runs of dots and mixes in the
        # property name; every component-id token is one of ours ("viewer-*").
        referenced = {
            token
            for token in re.split(r"\.+", dependency["output"])
            if token.startswith("viewer-")
        }
        assert referenced, dependency["output"]
        assert referenced <= known, dependency["output"]


# --------------------------------------------------------------------------- #
# Pure reducers (the callback bodies)
# --------------------------------------------------------------------------- #
def test_resolve_round_action_steps_both_directions(model: ReplayModel) -> None:
    assert (
        resolve_round_action(
            model, trigger_id=NEXT_BUTTON_ID, next_clicks=1, slider_value=0
        )
        == 1
    )
    assert (
        resolve_round_action(
            model, trigger_id=PREV_BUTTON_ID, prev_clicks=1, slider_value=1
        )
        == 0
    )


def test_resolve_round_action_clamps_at_terminal_controls(model: ReplayModel) -> None:
    assert (
        resolve_round_action(
            model, trigger_id=PREV_BUTTON_ID, prev_clicks=1, slider_value=0
        )
        == 0
    )
    assert (
        resolve_round_action(
            model, trigger_id=NEXT_BUTTON_ID, next_clicks=2, slider_value=1
        )
        == 1
    )


def test_resolve_round_action_empty_trigger_returns_first_round(
    model: ReplayModel,
) -> None:
    assert resolve_round_action(model, trigger_id=None, slider_value=1) == 0
    assert resolve_round_action(model) == 0


def test_resolve_round_action_ignores_spurious_clicks(model: ReplayModel) -> None:
    # A triggered button callback with a None click counter keeps the round.
    assert (
        resolve_round_action(
            model, trigger_id=NEXT_BUTTON_ID, next_clicks=None, slider_value=1
        )
        == 1
    )


def test_resolve_round_action_survives_malformed_slider(model: ReplayModel) -> None:
    # A malformed slider payload resolves to the deterministic first round, so
    # the following click steps forward from round 0 without ever raising.
    resolved = resolve_round_action(
        model, trigger_id=NEXT_BUTTON_ID, next_clicks=1, slider_value="oops"
    )
    assert resolved == 1
    assert resolved in model.rounds

    # A malformed current round with a Previous click clamps at the first round.
    assert (
        resolve_round_action(
            model,
            trigger_id=PREV_BUTTON_ID,
            prev_clicks=1,
            slider_value="oops",
        )
        == 0
    )


def test_coerce_helpers_bound_untrusted_payloads(model: ReplayModel) -> None:
    assert coerce_round(model, None) == 0
    assert coerce_round(model, "not-a-round") == 0
    assert coerce_round(model, 999) == 1
    assert coerce_round(model, -5) == 0

    assert coerce_node_id(model, None) == 0
    assert coerce_node_id(model, True) == 0
    assert coerce_node_id(model, 99) == 0
    # Numeric strings (a plausible malformed browser payload) fall back safely.
    assert coerce_node_id(model, "2") == 0
    assert coerce_node_id(model, "not-a-node") == 0
    # Valid integer ids pass through unchanged.
    assert coerce_node_id(model, 2) == 2


# --------------------------------------------------------------------------- #
# One-shot redraw
# --------------------------------------------------------------------------- #
def test_render_view_message_overlay_tracks_round(model: ReplayModel) -> None:
    round_zero = render_view(model, round_index=0, node_id=0)
    round_one = render_view(model, round_index=1, node_id=0)

    assert round_zero.round_index == 0
    assert round_one.round_index == 1
    assert "round 0 of 1" in round_zero.status
    assert "round 1 of 1" in round_one.status
    assert round_zero.round_label == "round 0 of 1"

    static_ids = {"node-0", "node-1", "node-2", "topology-0-1", "topology-1-2"}
    zero_ids = {element["data"]["id"] for element in round_zero.elements}
    one_ids = {element["data"]["id"] for element in round_one.elements}
    assert static_ids <= zero_ids and static_ids <= one_ids
    assert {element["data"]["id"] for element in round_zero.elements if element["classes"] == "message"} == {
        "message-0-0-1",
        "message-0-1-0",
        "message-0-1-2",
        "message-0-2-1",
    }
    assert "message-1-0-1" in one_ids
    assert "message-0-0-1" not in one_ids


def test_render_view_coerces_out_of_range_selection(model: ReplayModel) -> None:
    rendered = render_view(model, round_index=999, node_id=999)

    assert rendered.round_index == 1
    assert rendered.node_id == 0
    assert rendered.status.startswith("round 1")


# --------------------------------------------------------------------------- #
# Plotly charts: node labels, emphasis, cursor, and value
# --------------------------------------------------------------------------- #
def test_charts_label_every_node_and_keep_all_series(model: ReplayModel) -> None:
    figure = build_loss_figure(model, 0, 0)

    assert [trace.name for trace in figure.data] == ["node 0", "node 1", "node 2"]
    assert all(trace.hovertemplate.startswith("node ") for trace in figure.data)
    assert "loss" in figure.layout.title.text


def test_node_selection_emphasizes_one_series(model: ReplayModel) -> None:
    figure = build_loss_figure(model, 1, 2)
    by_name = {trace.name: trace for trace in figure.data}

    assert by_name["node 2"].line.width == 3
    assert by_name["node 2"].opacity == 1.0
    assert by_name["node 0"].line.width < 3
    assert by_name["node 0"].opacity < 1.0
    # Selection emphasizes, it does not drop the other nodes.
    assert set(by_name) == {"node 0", "node 1", "node 2"}


def test_charts_track_round_cursor_and_current_value(model: ReplayModel) -> None:
    fig_zero = build_loss_figure(model, 0, 0)
    fig_one = build_loss_figure(model, 1, 0)

    assert fig_zero.layout.shapes[0].x0 == 0
    assert fig_one.layout.shapes[0].x0 == 1
    assert fig_zero.layout.annotations[0].text == "node 0: 0.900"
    assert fig_one.layout.annotations[0].text == "node 0: 0.700"
    assert list(fig_zero.layout.xaxis.tickvals) == [0, 1]


def test_accuracy_chart_uses_accuracy_series_and_range(model: ReplayModel) -> None:
    figure = build_accuracy_figure(model, 0, 2)

    by_name = {trace.name: trace for trace in figure.data}
    assert list(by_name["node 2"].y) == [0.20, 0.25]
    assert figure.layout.annotations[0].text == "node 2: 0.200"
    assert figure.layout.yaxis.range == (0.0, 1.0)


def test_stylesheet_helper_matches_layout(app) -> None:
    layout = _layout(app.server.test_client())
    assert _find(layout, NETWORK_ID)["props"]["stylesheet"] == cytoscape_stylesheet()


# --------------------------------------------------------------------------- #
# Boundary shapes and fail-closed startup
# --------------------------------------------------------------------------- #
def test_single_round_app_has_valid_state(tmp_path: Path) -> None:
    trace = make_viewer_trace(tmp_path / "one_round", rounds=1)
    app = create_viewer_app(trace)
    layout = _layout(app.server.test_client())

    slider = _find(layout, ROUND_SLIDER_ID)["props"]
    assert slider["min"] == slider["max"] == 0
    assert slider["marks"] == {"0": "0"}

    rendered = render_view(load_replay(trace))
    assert rendered.round_index == 0
    assert rendered.status.startswith("round 0 of 0")
    assert list(build_loss_figure(load_replay(trace), 0, 0).layout.xaxis.tickvals) == [0]


def test_single_node_app_builds(tmp_path: Path) -> None:
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(0)
    trace = make_viewer_trace(
        tmp_path / "single", rounds=1, graph=graph, compromised_node=0
    )
    app = create_viewer_app(trace)
    layout = _layout(app.server.test_client())

    network = _find(layout, NETWORK_ID)["props"]
    assert len(network["elements"]) == 1
    assert network["elements"][0]["classes"] == MALICIOUS


def test_app_factory_accepts_a_projected_model(model: ReplayModel) -> None:
    app = create_viewer_app(model)
    layout = _layout(app.server.test_client())
    assert _find(layout, TRACE_SUMMARY_ID) is not None


def test_app_factory_fails_closed_on_missing_trace(tmp_path: Path) -> None:
    with pytest.raises(ViewerError):
        create_viewer_app(tmp_path / "missing")


def test_app_factory_fails_closed_on_unsupported_schema(viewer_trace: Path) -> None:
    manifest_path = viewer_trace / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 99
    manifest_path.write_bytes(
        (
            json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    )

    with pytest.raises(ViewerError):
        create_viewer_app(viewer_trace)
