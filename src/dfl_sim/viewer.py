"""Framework-neutral replay projection for the local Dash viewer (R011, D020).

This module is the strict, side-effect-free seam between a completed
schema-v1 trajectory and any user interface. It:

* opens a trace **only** through :class:`~dfl_sim.trajectory.TraceReader`, so
  layout, digest, canonical-JSON, ZIP-member, and schema checks are reused
  rather than reimplemented;
* copies only scalar manifest/event data into immutable
  :class:`~dataclasses.dataclass` records and then closes the reader;
* never reads ``tensors.npz`` tensor payloads, never parses
  ``config.json``/``events.jsonl``/``manifest.json`` directly, and never
  exposes simulator/evaluator objects;
* precomputes everything the browser needs once, so view callbacks are pure
  in-memory projections.

The projection is deliberately framework-neutral: the Dash/Plotly/Cytoscape
application factory (:func:`create_viewer_app`) lazily imports its UI packages
**inside** the factory body, so importing :mod:`dfl_sim` (and therefore running
the simulator) never requires a web dependency. Only :func:`render_view` and
the figure builders touch Plotly, and only :func:`create_viewer_app` touches
Dash; the pure reducers and element projectors stay dependency-free.

Q3 threat surface: the trace path and every artifact byte are untrusted local
input. The only filesystem access is through ``TraceReader``; there is no
upload, URL fetch, HTML interpolation, pickle load, arbitrary module import, or
callback that rereads a path supplied by a browser. Callbacks receive only
bounded round/node values and operate on the immutable in-process model.

Q6 load profile: projection is linear in scalar events and messages and stores
no model tensors. M001's 10-20 nodes and ordinary round counts are the target;
thousands of rounds or dense graphs are the likely 10x browser-payload
breakpoint because the full Cytoscape/Plotly JSON eventually dominates.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluator import BENIGN, MALICIOUS
from .trajectory import TraceReader, TrajectoryError

__all__ = [
    "ACCURACY_GRAPH_ID",
    "LEGEND_ID",
    "LOSS_GRAPH_ID",
    "NETWORK_ID",
    "NEXT_BUTTON_ID",
    "NODE_SELECTOR_ID",
    "PREV_BUTTON_ID",
    "ROUND_LABEL_ID",
    "ROUND_SLIDER_ID",
    "ROUND_STATUS_ID",
    "TRACE_SUMMARY_ID",
    "VIEWER_COMPONENT_IDS",
    "DirectedMessage",
    "NodeFrameState",
    "ReplayModel",
    "ViewRender",
    "ViewerError",
    "ViewerFrame",
    "ViewerNode",
    "build_accuracy_figure",
    "build_loss_figure",
    "clamp_round",
    "coerce_node_id",
    "coerce_round",
    "create_viewer_app",
    "cytoscape_message_edges",
    "cytoscape_nodes",
    "cytoscape_round_elements",
    "cytoscape_static_elements",
    "cytoscape_stylesheet",
    "cytoscape_topology_edges",
    "describe_round",
    "describe_trace",
    "load_replay",
    "render_view",
    "resolve_round_action",
    "select_frame",
    "select_frame_node",
    "select_node",
    "step_round",
]

#: Preset-layout radius for the deterministic circular node placement.
_LAYOUT_RADIUS = 180.0
#: First node is placed at the top of the circle for a stable, familiar layout.
_LAYOUT_START_ANGLE = -math.pi / 2.0
#: Accepted node-type classes (mirrors the manifest contract).
_NODE_TYPES = (BENIGN, MALICIOUS)


class ViewerError(RuntimeError):
    """A typed, path-safe viewer startup or projection failure.

    Raised for an absent/incomplete/unsupported/tampered trace or for an
    impossible replay frame (unknown node reference, node/type mismatch,
    invalid frame/node selection). The original :class:`TrajectoryError` is
    preserved as ``__cause__``; the message carries only the trace directory
    name so volatile absolute paths are not surfaced to the UI or CLI.
    """


# --------------------------------------------------------------------------- #
# Immutable replay records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class DirectedMessage:
    """One round's directed sender-to-recipient message fanout."""

    sender: int
    recipients: tuple[int, ...]

    def pairs(self) -> tuple[tuple[int, int], ...]:
        """The directed ``(sender, recipient)`` pairs in recipient order."""

        return tuple((self.sender, recipient) for recipient in self.recipients)


@dataclass(frozen=True, slots=True)
class NodeFrameState:
    """Per-node scalar state captured for one round of the replay."""

    node_id: int
    loss: float
    accuracy: float
    sample_count: int
    true_type: str
    attack_kind: str
    attack_kappa: float
    participant_ids: tuple[int, ...]
    multipliers: tuple[float, ...]
    weights: tuple[float, ...]
    propagation_reached: bool
    propagation_min_hop: int | None
    propagation_first_reached_round: int | None


@dataclass(frozen=True, slots=True)
class ViewerFrame:
    """One round of the replay: node states plus directed message fanout.

    ``nodes`` is sorted ascending by node id and ``messages`` ascending by
    sender, matching the canonical event order, so rendering is deterministic.
    """

    round_index: int
    nodes: tuple[NodeFrameState, ...]
    messages: tuple[DirectedMessage, ...]

    @property
    def node_ids(self) -> tuple[int, ...]:
        return tuple(state.node_id for state in self.nodes)

    @property
    def message_count(self) -> int:
        """Total directed deliveries in this round."""

        return sum(len(message.recipients) for message in self.messages)

    def state_for(self, node_id: int) -> NodeFrameState:
        """Return this round's state for ``node_id`` (bounds-checked)."""

        for state in self.nodes:
            if state.node_id == node_id:
                return state
        raise ViewerError(
            f"round {self.round_index} has no node {node_id}"
        )


@dataclass(frozen=True, slots=True)
class ViewerNode:
    """A stable topology node with its deterministic preset position."""

    node_id: int
    node_type: str
    x: float
    y: float

    @property
    def position(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True, slots=True)
class ReplayModel:
    """Immutable, framework-neutral projection of a completed schema-v1 trace.

    ``loss_history`` and ``accuracy_history`` are aligned to :attr:`nodes` (and
    therefore to :attr:`node_ids`); each inner tuple is aligned to
    :attr:`rounds`. This gives deterministic node/round ordering for the
    per-node metric charts without any dict-order dependence.
    """

    trace_name: str
    schema_version: int
    aggregation: str
    attack_kind: str
    attack_kappa: float
    compromised_node: int | None
    nodes: tuple[ViewerNode, ...]
    edges: tuple[tuple[int, int], ...]
    frames: tuple[ViewerFrame, ...]
    loss_history: tuple[tuple[float, ...], ...]
    accuracy_history: tuple[tuple[float, ...], ...]
    snapshot_rounds: tuple[int, ...]

    @property
    def node_ids(self) -> tuple[int, ...]:
        return tuple(node.node_id for node in self.nodes)

    @property
    def node_types(self) -> dict[int, str]:
        return {node.node_id: node.node_type for node in self.nodes}

    @property
    def rounds(self) -> tuple[int, ...]:
        return tuple(frame.round_index for frame in self.frames)

    @property
    def first_round(self) -> int:
        return self.frames[0].round_index

    @property
    def last_round(self) -> int:
        return self.frames[-1].round_index

    def node_index(self, node_id: int) -> int:
        """Return the history index for ``node_id`` (bounds-checked)."""

        for index, node in enumerate(self.nodes):
            if node.node_id == node_id:
                return index
        raise ViewerError(f"trace {self.trace_name!r} has no node {node_id}")

    def loss_series(self, node_id: int) -> tuple[float, ...]:
        return self.loss_history[self.node_index(node_id)]

    def accuracy_series(self, node_id: int) -> tuple[float, ...]:
        return self.accuracy_history[self.node_index(node_id)]


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def _as_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ViewerError(f"{label} must be an integer, got {value!r}")
    return int(value)


def _sanitize(exc: BaseException, path: Path) -> str:
    """Replace any absolute trace path in ``exc`` with the directory name."""

    detail = str(exc)
    try:
        absolute = str(path.resolve())
    except OSError:  # pragma: no cover - resolve rarely fails
        absolute = str(path)
    for candidate in (str(path), absolute):
        if candidate:
            detail = detail.replace(candidate, path.name or candidate)
    return detail


def _preset_positions(node_ids: tuple[int, ...]) -> dict[int, dict[str, float]]:
    """Deterministic circular preset positions; a lone node sits at the origin."""

    count = len(node_ids)
    if count == 1:
        return {node_ids[0]: {"x": 0.0, "y": 0.0}}
    positions: dict[int, dict[str, float]] = {}
    for index, node in enumerate(node_ids):
        angle = _LAYOUT_START_ANGLE + (2.0 * math.pi * index / count)
        positions[node] = {
            "x": round(_LAYOUT_RADIUS * math.cos(angle), 6),
            "y": round(_LAYOUT_RADIUS * math.sin(angle), 6),
        }
    return positions


def _project_node_state(
    round_index: int, node_id: int, payload: dict
) -> NodeFrameState:
    attack = payload["attack"]
    profile = payload["profile"]
    propagation = payload["propagation"]
    min_hop = propagation["min_hop"]
    first_reached = propagation["first_reached_round"]
    return NodeFrameState(
        node_id=node_id,
        loss=float(payload["loss"]),
        accuracy=float(payload["accuracy"]),
        sample_count=int(payload["sample_count"]),
        true_type=str(payload["true_type"]),
        attack_kind=str(attack["kind"]),
        attack_kappa=float(attack["kappa"]),
        participant_ids=tuple(int(item) for item in profile["participants"]),
        multipliers=tuple(float(item) for item in profile["multipliers"]),
        weights=tuple(float(item) for item in profile["weights"]),
        propagation_reached=bool(propagation["reached"]),
        propagation_min_hop=None if min_hop is None else int(min_hop),
        propagation_first_reached_round=(
            None if first_reached is None else int(first_reached)
        ),
    )


def _project_frame(
    event: dict,
    *,
    trace_name: str,
    node_set: set[int],
    node_types: dict[int, str],
) -> ViewerFrame:
    round_index = int(event["round"])
    states: list[NodeFrameState] = []
    seen: set[int] = set()
    for payload in event["nodes"]:
        node_id = int(payload["node"])
        if node_id not in node_set:
            raise ViewerError(
                f"trace {trace_name!r} round {round_index} references unknown "
                f"node {node_id}"
            )
        if node_id in seen:
            raise ViewerError(
                f"trace {trace_name!r} round {round_index} repeats node {node_id}"
            )
        seen.add(node_id)
        true_type = str(payload["true_type"])
        expected = node_types[node_id]
        if true_type != expected:
            raise ViewerError(
                f"trace {trace_name!r} round {round_index} node {node_id} "
                f"declares true_type {true_type!r} but manifest says {expected!r}"
            )
        states.append(_project_node_state(round_index, node_id, payload))

    missing = sorted(node_set - seen)
    if missing:
        raise ViewerError(
            f"trace {trace_name!r} round {round_index} is missing nodes {missing}"
        )

    messages: list[DirectedMessage] = []
    for message in event["messages"]:
        sender = int(message["sender"])
        if sender not in node_set:
            raise ViewerError(
                f"trace {trace_name!r} round {round_index} has unknown message "
                f"sender {sender}"
            )
        recipients = tuple(int(item) for item in message["recipients"])
        for recipient in recipients:
            if recipient not in node_set:
                raise ViewerError(
                    f"trace {trace_name!r} round {round_index} sender {sender} "
                    f"targets unknown node {recipient}"
                )
        messages.append(DirectedMessage(sender=sender, recipients=recipients))

    ordered_states = tuple(sorted(states, key=lambda state: state.node_id))
    return ViewerFrame(
        round_index=round_index,
        nodes=ordered_states,
        messages=tuple(messages),
    )


def _project(reader: TraceReader, *, name: str) -> ReplayModel:
    node_ids = reader.nodes
    if not node_ids:
        raise ViewerError(f"trace {name!r} has no nodes")
    node_types = reader.node_types
    for node_id in node_ids:
        if node_types.get(node_id) not in _NODE_TYPES:
            raise ViewerError(
                f"trace {name!r} node {node_id} has unsupported type "
                f"{node_types.get(node_id)!r}"
            )

    positions = _preset_positions(node_ids)
    viewer_nodes = tuple(
        ViewerNode(
            node_id=node_id,
            node_type=node_types[node_id],
            x=positions[node_id]["x"],
            y=positions[node_id]["y"],
        )
        for node_id in node_ids
    )

    node_set = set(node_ids)
    edges: list[tuple[int, int]] = []
    for left, right in reader.edges:
        if left not in node_set or right not in node_set:
            raise ViewerError(
                f"trace {name!r} edge ({left}, {right}) references an unknown node"
            )
        edges.append((int(left), int(right)))

    frames = tuple(
        _project_frame(
            event, trace_name=name, node_set=node_set, node_types=node_types
        )
        for event in reader.events
    )
    if not frames:
        raise ViewerError(f"trace {name!r} has no round events")

    loss_by_node: dict[int, list[float]] = {node_id: [] for node_id in node_ids}
    accuracy_by_node: dict[int, list[float]] = {node_id: [] for node_id in node_ids}
    for frame in frames:
        for state in frame.nodes:
            loss_by_node[state.node_id].append(state.loss)
            accuracy_by_node[state.node_id].append(state.accuracy)

    manifest = reader.manifest
    attack = manifest["attack"]
    compromised = manifest["compromised_node"]
    return ReplayModel(
        trace_name=name,
        schema_version=reader.schema_version,
        aggregation=str(manifest["aggregation"]),
        attack_kind=str(attack["kind"]),
        attack_kappa=float(attack["kappa"]),
        compromised_node=None if compromised is None else int(compromised),
        nodes=viewer_nodes,
        edges=tuple(edges),
        frames=frames,
        loss_history=tuple(
            tuple(loss_by_node[node_id]) for node_id in node_ids
        ),
        accuracy_history=tuple(
            tuple(accuracy_by_node[node_id]) for node_id in node_ids
        ),
        snapshot_rounds=reader.snapshot_rounds,
    )


def load_replay(trace_path: str | Path) -> ReplayModel:
    """Project a completed schema-v1 trace into an immutable replay model.

    Opens ``trace_path`` exactly once through :class:`TraceReader`, copies only
    the scalar manifest/event data the UI needs, and closes the reader before
    returning. Any validation or projection failure raises :class:`ViewerError`
    with the trace directory name and no partially projected model.
    """

    base = Path(trace_path)
    name = base.name or str(base)
    try:
        reader = TraceReader(base)
    except TrajectoryError as exc:
        raise ViewerError(
            f"cannot open trace {name!r}: {_sanitize(exc, base)}"
        ) from exc
    try:
        return _project(reader, name=name)
    except ViewerError:
        raise
    except TrajectoryError as exc:  # pragma: no cover - projection reads scalars
        raise ViewerError(
            f"cannot project trace {name!r}: {_sanitize(exc, base)}"
        ) from exc
    finally:
        reader.close()


# --------------------------------------------------------------------------- #
# Pure, bounds-checked selection and stepping helpers
# --------------------------------------------------------------------------- #
def select_frame(model: ReplayModel, round_index: object) -> ViewerFrame:
    """Return the frame with the exact actual round index (bounds-checked)."""

    value = _as_int(round_index, label="round index")
    for frame in model.frames:
        if frame.round_index == value:
            return frame
    raise ViewerError(f"trace {model.trace_name!r} has no round {value}")


def select_node(model: ReplayModel, node_id: object) -> ViewerNode:
    """Return the topology node with ``node_id`` (bounds-checked)."""

    value = _as_int(node_id, label="node id")
    for node in model.nodes:
        if node.node_id == value:
            return node
    raise ViewerError(f"trace {model.trace_name!r} has no node {value}")


def select_frame_node(
    model: ReplayModel, round_index: object, node_id: object
) -> NodeFrameState:
    """Return the per-round state of one node (both axes bounds-checked)."""

    frame = select_frame(model, round_index)
    return frame.state_for(_as_int(node_id, label="node id"))


def clamp_round(model: ReplayModel, round_index: object) -> int:
    """Clamp a round request to the nearest actual event round index.

    Never raises for an out-of-range request; preserves true event round
    indices instead of treating slider positions as array indices.
    """

    value = _as_int(round_index, label="round index")
    rounds = model.rounds
    if value <= rounds[0]:
        return rounds[0]
    if value >= rounds[-1]:
        return rounds[-1]
    return min(rounds, key=lambda candidate: (abs(candidate - value), candidate))


def step_round(model: ReplayModel, current_round: object, delta: object) -> int:
    """Move ``delta`` actual rounds from ``current_round``, clamped at bounds.

    Stepping is positional, so non-contiguous event round indices stay exact
    and ``Previous``/``Next`` can never leave the valid range.
    """

    rounds = model.rounds
    step = _as_int(delta, label="step")
    position = rounds.index(clamp_round(model, current_round))
    target = position + step
    if target < 0:
        target = 0
    elif target >= len(rounds):
        target = len(rounds) - 1
    return rounds[target]


# --------------------------------------------------------------------------- #
# Cytoscape element projection (stable IDs, explicit classes, preset layout)
# --------------------------------------------------------------------------- #
def cytoscape_nodes(model: ReplayModel) -> list[dict[str, Any]]:
    """Cytoscape node elements with stable IDs, classes, and preset positions."""

    return [
        {
            "data": {
                "id": f"node-{node.node_id}",
                "label": f"node {node.node_id}",
                "node_id": node.node_id,
                "true_type": node.node_type,
            },
            "position": {"x": node.x, "y": node.y},
            "classes": node.node_type,
        }
        for node in model.nodes
    ]


def cytoscape_topology_edges(model: ReplayModel) -> list[dict[str, Any]]:
    """Round-independent undirected topology edges (class ``topology``)."""

    return [
        {
            "data": {
                "id": f"topology-{left}-{right}",
                "source": f"node-{left}",
                "target": f"node-{right}",
                "kind": "topology",
            },
            "classes": "topology",
        }
        for left, right in model.edges
    ]


def cytoscape_message_edges(frame: ViewerFrame) -> list[dict[str, Any]]:
    """Round-dependent directed message overlays (class ``message``).

    IDs are stable for ``(round, sender, recipient)`` so a callback can replace
    only these elements without disturbing topology nodes/edges.
    """

    elements: list[dict[str, Any]] = []
    for message in frame.messages:
        for recipient in message.recipients:
            elements.append(
                {
                    "data": {
                        "id": (
                            f"message-{frame.round_index}-{message.sender}-{recipient}"
                        ),
                        "source": f"node-{message.sender}",
                        "target": f"node-{recipient}",
                        "kind": "message",
                        "round": frame.round_index,
                        "sender": message.sender,
                        "recipient": recipient,
                    },
                    "classes": "message",
                }
            )
    return elements


def cytoscape_static_elements(model: ReplayModel) -> list[dict[str, Any]]:
    """Nodes plus topology edges; round-independent part of the network."""

    return [*cytoscape_nodes(model), *cytoscape_topology_edges(model)]


def cytoscape_round_elements(model: ReplayModel, round_index: object) -> list[dict[str, Any]]:
    """Full network element list for one round (static plus directed flow)."""

    frame = select_frame(model, round_index)
    return [*cytoscape_static_elements(model), *cytoscape_message_edges(frame)]


# --------------------------------------------------------------------------- #
# Accessible text summaries (used by the UI and browser tests)
# --------------------------------------------------------------------------- #
def describe_trace(model: ReplayModel) -> str:
    """A concise, deterministic, path-safe summary of the loaded trace."""

    malicious = sum(1 for node in model.nodes if node.node_type == MALICIOUS)
    if model.compromised_node is None:
        attack = f"{model.attack_kind} (no compromised node)"
    else:
        attack = (
            f"{model.attack_kind} on node {model.compromised_node} "
            f"(kappa={model.attack_kappa:g})"
        )
    return (
        f"trace {model.trace_name!r}: schema v{model.schema_version}, "
        f"{len(model.nodes)} node(s) [{malicious} malicious], "
        f"{len(model.edges)} edge(s), {len(model.frames)} round(s); "
        f"aggregation={model.aggregation}; attack={attack}"
    )


def describe_round(model: ReplayModel, round_index: object) -> str:
    """A concise status string for one actual round of the replay."""

    frame = select_frame(model, round_index)
    senders = len(frame.messages)
    return (
        f"round {frame.round_index} of {model.last_round}: "
        f"{frame.message_count} directed message(s) from {senders} sender(s)"
    )


# --------------------------------------------------------------------------- #
# Bounded coercion + callback reducers (pure, framework-neutral)
# --------------------------------------------------------------------------- #
def coerce_round(model: ReplayModel, round_index: object) -> int:
    """Clamp an untrusted round request; malformed input falls back to first.

    Callback payloads (slider/dropdown values) are strings, ``None``, or absent
    entirely. This never raises: an out-of-range integer clamps to the nearest
    actual round and a malformed value resolves to the deterministic first
    round, so a callback can never index outside ``model.frames``.
    """

    try:
        return clamp_round(model, round_index)
    except ViewerError:
        return model.first_round


def coerce_node_id(model: ReplayModel, node_id: object) -> int:
    """Return a valid node id; malformed/unknown input falls back to first."""

    try:
        value = _as_int(node_id, label="node id")
    except ViewerError:
        return model.node_ids[0]
    return value if value in model.node_ids else model.node_ids[0]


def _is_click(value: object) -> bool:
    """True when a Dash click counter represents a real positive click."""

    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, bool)
        and int(value) > 0
    )


def resolve_round_action(
    model: ReplayModel,
    *,
    trigger_id: object = None,
    prev_clicks: object = None,
    next_clicks: object = None,
    slider_value: object = None,
) -> int:
    """Resolve the round after a Previous/Next click or an empty trigger.

    ``slider_value`` is the current round (Dash ``State``). Click counts guard
    against spurious button callbacks, and an empty/unknown trigger returns the
    deterministic initial round so a missing trigger never indexes outside the
    model. Movement is clamped by :func:`step_round`, so terminal clicks are
    no-ops rather than out-of-range requests.
    """

    current = coerce_round(model, slider_value)
    if trigger_id == PREV_BUTTON_ID and _is_click(prev_clicks):
        return step_round(model, current, -1)
    if trigger_id == NEXT_BUTTON_ID and _is_click(next_clicks):
        return step_round(model, current, 1)
    if trigger_id in (PREV_BUTTON_ID, NEXT_BUTTON_ID):
        # A guarded button callback with no real click keeps the current round.
        return current
    return model.first_round


# --------------------------------------------------------------------------- #
# Plotly metric figures (lazy Plotly import; deterministic per-node series)
# --------------------------------------------------------------------------- #
_BENIGN_COLOR = "#2f7d32"
_MALICIOUS_COLOR = "#c62828"
_SELECTED_LINE_WIDTH = 3
_MUTED_LINE_WIDTH = 1.25
_MUTED_OPACITY = 0.35
_CURSOR_COLOR = "#455a64"


def _node_series(model: ReplayModel, node_id: int, metric: str) -> tuple[float, ...]:
    if metric == "loss":
        return model.loss_series(node_id)
    return model.accuracy_series(node_id)


def _metric_figure(
    model: ReplayModel, *, metric: str, round_index: object, node_id: object
):
    """Build one per-node metric figure with a current-round cursor.

    Returns a :class:`plotly.graph_objects.Figure`. Plotly is imported lazily
    so importing :mod:`dfl_sim` stays web-dependency free. Every trace is
    labeled ``node N`` and carries a hover template with the true round index;
    the selected node is emphasized while all nodes remain visible.
    """

    import plotly.graph_objects as go

    round_value = coerce_round(model, round_index)
    selected = coerce_node_id(model, node_id)
    position = model.rounds.index(round_value)

    figure = go.Figure()
    for node in model.nodes:
        series = _node_series(model, node.node_id, metric)
        is_selected = node.node_id == selected
        color = (
            _MALICIOUS_COLOR if node.node_type == MALICIOUS else _BENIGN_COLOR
        )
        figure.add_trace(
            go.Scatter(
                x=list(model.rounds),
                y=list(series),
                mode="lines+markers",
                name=f"node {node.node_id}",
                legendgroup=f"node-{node.node_id}",
                line={
                    "width": (
                        _SELECTED_LINE_WIDTH if is_selected else _MUTED_LINE_WIDTH
                    ),
                    "color": color,
                },
                opacity=1.0 if is_selected else _MUTED_OPACITY,
                marker={"size": 10 if is_selected else 7, "color": color},
                hovertemplate=(
                    f"node {node.node_id}<br>round %{{x}}<br>"
                    f"{metric}=%{{y:.3f}}<extra></extra>"
                ),
            )
        )

    current_value = model.loss_series(selected)[position] if metric == "loss" else (
        model.accuracy_series(selected)[position]
    )
    figure.add_vline(x=round_value, line_dash="dot", line_color=_CURSOR_COLOR)
    figure.add_annotation(
        x=round_value,
        y=current_value,
        text=f"node {selected}: {current_value:.3f}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-30,
        bgcolor="rgba(255,255,255,0.85)",
    )
    figure.update_layout(
        title=f"per-node {metric} (node {selected} emphasized)",
        xaxis={
            "title": "round",
            "tickmode": "array",
            "tickvals": list(model.rounds),
        },
        yaxis={"title": metric},
        margin={"l": 56, "r": 24, "t": 56, "b": 40},
        legend={"orientation": "h", "y": -0.28},
    )
    if metric == "accuracy":
        figure.update_yaxes(range=[0.0, 1.0])
    return figure


def build_loss_figure(
    model: ReplayModel, round_index: object = None, node_id: object = None
):
    """Per-node loss history with the current round and selected node marked."""

    return _metric_figure(
        model, metric="loss", round_index=round_index, node_id=node_id
    )


def build_accuracy_figure(
    model: ReplayModel, round_index: object = None, node_id: object = None
):
    """Per-node accuracy history with the current round and selected node."""

    return _metric_figure(
        model, metric="accuracy", round_index=round_index, node_id=node_id
    )


# --------------------------------------------------------------------------- #
# One-shot view render (pure projection used by the Dash callbacks and tests)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ViewRender:
    """Everything a single round/node redraw needs, in one immutable bundle."""

    round_index: int
    node_id: int
    elements: list[dict[str, Any]]
    status: str
    label: str
    round_label: str
    loss_figure: Any
    accuracy_figure: Any


def render_view(
    model: ReplayModel, *, round_index: object = None, node_id: object = None
) -> ViewRender:
    """Render the network, status text, and both charts for one selection.

    Both inputs are coerced to the valid range, so this is safe to call with
    raw Dash callback payloads. Callbacks therefore never index outside the
    projected rounds/nodes, and malformed payloads degrade to the deterministic
    initial round/first node.
    """

    resolved = coerce_round(model, round_index)
    selected = coerce_node_id(model, node_id)
    return ViewRender(
        round_index=resolved,
        node_id=selected,
        elements=cytoscape_round_elements(model, resolved),
        status=describe_round(model, resolved),
        label=f"round {resolved}",
        round_label=f"round {resolved} of {model.last_round}",
        loss_figure=build_loss_figure(model, resolved, selected),
        accuracy_figure=build_accuracy_figure(model, resolved, selected),
    )


# --------------------------------------------------------------------------- #
# Stable component IDs and Cytoscape styling
# --------------------------------------------------------------------------- #
TRACE_SUMMARY_ID = "viewer-trace-summary"
LEGEND_ID = "viewer-legend"
ROUND_STATUS_ID = "viewer-round-status"
ROUND_LABEL_ID = "viewer-round-label"
ROUND_SLIDER_ID = "viewer-round-slider"
PREV_BUTTON_ID = "viewer-prev-button"
NEXT_BUTTON_ID = "viewer-next-button"
NODE_SELECTOR_ID = "viewer-node-selector"
NETWORK_ID = "viewer-network"
LOSS_GRAPH_ID = "viewer-loss-graph"
ACCURACY_GRAPH_ID = "viewer-accuracy-graph"

#: Every stable id the layout must expose (browser/endpoint tests rely on it).
VIEWER_COMPONENT_IDS: tuple[str, ...] = (
    TRACE_SUMMARY_ID,
    LEGEND_ID,
    ROUND_STATUS_ID,
    ROUND_LABEL_ID,
    ROUND_SLIDER_ID,
    PREV_BUTTON_ID,
    NEXT_BUTTON_ID,
    NODE_SELECTOR_ID,
    NETWORK_ID,
    LOSS_GRAPH_ID,
    ACCURACY_GRAPH_ID,
)


def cytoscape_stylesheet() -> list[dict[str, Any]]:
    """Explicit node/edge classes so type and flow styling are deterministic.

    Benign/malicious nodes differ by fill colour, border, and shape; topology
    edges stay visible and undirected-looking; message edges render as directed
    arrow overlays that are clearly distinguishable from topology.
    """

    return [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "font-size": "12px",
                "text-valign": "center",
                "text-halign": "center",
                "color": "#ffffff",
                "width": 34,
                "height": 34,
            },
        },
        {
            "selector": "node.benign",
            "style": {
                "background-color": _BENIGN_COLOR,
                "border-color": "#1b5e20",
                "border-width": 3,
                "shape": "ellipse",
            },
        },
        {
            "selector": "node.malicious",
            "style": {
                "background-color": _MALICIOUS_COLOR,
                "border-color": "#8e0000",
                "border-width": 4,
                "shape": "diamond",
            },
        },
        {
            "selector": "edge.topology",
            "style": {
                "line-color": "#90a4ae",
                "width": 2,
                "curve-style": "bezier",
                "opacity": 0.7,
            },
        },
        {
            "selector": "edge.message",
            "style": {
                "line-color": "#1565c0",
                "width": 2,
                "curve-style": "bezier",
                "target-arrow-color": "#1565c0",
                "target-arrow-shape": "triangle",
                "arrow-scale": 1.2,
                "opacity": 0.95,
            },
        },
    ]


# --------------------------------------------------------------------------- #
# Dash application factory (lazy Dash/Cytoscape import)
# --------------------------------------------------------------------------- #
def create_viewer_app(
    trace_path: str | Path | ReplayModel, *, title: str | None = None
):
    """Build the read-only local Dash replay app for one validated trace.

    ``trace_path`` is projected once by :func:`load_replay` before any layout
    is built, so an invalid/incomplete/tampered trace fails here with a typed
    :class:`ViewerError` and no partial app. All callbacks close over the
    immutable :class:`ReplayModel`; none rereads a path or accepts browser
    input beyond bounded round/node values.

    Dash, Dash Cytoscape, and Plotly are imported inside this function so the
    simulator's import surface is unchanged.
    """

    import dash
    import dash_cytoscape as cyto
    from dash import Input, Output, State, ctx, dcc, html

    model = trace_path if isinstance(trace_path, ReplayModel) else load_replay(
        trace_path
    )
    initial = render_view(
        model, round_index=model.first_round, node_id=model.node_ids[0]
    )

    def legend_swatch(color: str, label: str):
        return html.Span(
            [
                html.Span(
                    style={
                        "display": "inline-block",
                        "width": "0.9rem",
                        "height": "0.9rem",
                        "backgroundColor": color,
                        "border": "1px solid #37474f",
                        "marginRight": "0.35rem",
                        "verticalAlign": "middle",
                    }
                ),
                label,
            ],
            style={"marginRight": "1rem"},
        )

    app = dash.Dash(__name__, title=title or f"dfl replay: {model.trace_name}")
    app.layout = html.Div(
        [
            html.H1(
                f"DFL replay viewer - {model.trace_name}",
                style={"fontSize": "1.35rem"},
            ),
            html.P(describe_trace(model), id=TRACE_SUMMARY_ID),
            html.Div(
                [
                    html.Span("Legend: ", style={"fontWeight": "bold"}),
                    legend_swatch(_BENIGN_COLOR, "benign node"),
                    legend_swatch(_MALICIOUS_COLOR, "malicious node"),
                ],
                id=LEGEND_ID,
                style={"marginBottom": "0.75rem"},
            ),
            html.Div(
                [
                    html.Button(
                        "Previous",
                        id=PREV_BUTTON_ID,
                        n_clicks=0,
                        style={"marginRight": "0.5rem"},
                    ),
                    html.Button("Next", id=NEXT_BUTTON_ID, n_clicks=0),
                    html.Span(
                        initial.round_label,
                        id=ROUND_LABEL_ID,
                        style={"marginLeft": "0.75rem", "fontWeight": "bold"},
                        **{"aria-live": "polite"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
            dcc.Slider(
                id=ROUND_SLIDER_ID,
                min=model.first_round,
                max=model.last_round,
                step=1,
                marks={round_index: str(round_index) for round_index in model.rounds},
                value=model.first_round,
            ),
            html.Div(
                initial.status,
                id=ROUND_STATUS_ID,
                style={"margin": "0.75rem 0"},
                **{"aria-live": "polite"},
            ),
            cyto.Cytoscape(
                id=NETWORK_ID,
                elements=initial.elements,
                layout={"name": "preset", "fit": True, "padding": 30},
                stylesheet=cytoscape_stylesheet(),
                style={
                    "width": "100%",
                    "height": "460px",
                    "border": "1px solid #cfd8dc",
                },
            ),
            html.Label(
                "Node:",
                htmlFor=NODE_SELECTOR_ID,
                style={"fontWeight": "bold", "marginRight": "0.5rem"},
            ),
            dcc.Dropdown(
                id=NODE_SELECTOR_ID,
                options=[
                    {"label": f"node {node_id}", "value": node_id}
                    for node_id in model.node_ids
                ],
                value=model.node_ids[0],
                clearable=False,
                style={"maxWidth": "16rem"},
            ),
            html.Div(
                [
                    dcc.Graph(
                        id=LOSS_GRAPH_ID,
                        figure=initial.loss_figure,
                        config={"displayModeBar": False},
                        style={"flex": "1 1 32rem", "minWidth": "24rem"},
                    ),
                    dcc.Graph(
                        id=ACCURACY_GRAPH_ID,
                        figure=initial.accuracy_figure,
                        config={"displayModeBar": False},
                        style={"flex": "1 1 32rem", "minWidth": "24rem"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "1rem",
                    "flexWrap": "wrap",
                    "marginTop": "1rem",
                },
            ),
        ],
        style={
            "maxWidth": "1100px",
            "margin": "0 auto",
            "padding": "1rem",
            "fontFamily": "system-ui, -apple-system, sans-serif",
        },
    )

    @app.callback(
        Output(ROUND_SLIDER_ID, "value"),
        Input(PREV_BUTTON_ID, "n_clicks"),
        Input(NEXT_BUTTON_ID, "n_clicks"),
        State(ROUND_SLIDER_ID, "value"),
        prevent_initial_call=True,
    )
    def _step_round(prev_clicks, next_clicks, slider_value):
        return resolve_round_action(
            model,
            trigger_id=ctx.triggered_id,
            prev_clicks=prev_clicks,
            next_clicks=next_clicks,
            slider_value=slider_value,
        )

    @app.callback(
        Output(NETWORK_ID, "elements"),
        Output(ROUND_STATUS_ID, "children"),
        Output(ROUND_LABEL_ID, "children"),
        Output(LOSS_GRAPH_ID, "figure"),
        Output(ACCURACY_GRAPH_ID, "figure"),
        Input(ROUND_SLIDER_ID, "value"),
        Input(NODE_SELECTOR_ID, "value"),
    )
    def _redraw(slider_value, node_value):
        rendered = render_view(
            model, round_index=slider_value, node_id=node_value
        )
        return (
            rendered.elements,
            rendered.status,
            rendered.round_label,
            rendered.loss_figure,
            rendered.accuracy_figure,
        )

    return app
