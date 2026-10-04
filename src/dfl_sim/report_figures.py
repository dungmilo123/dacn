"""Deterministic, dependency-free static figures for the canonical report.

The canonical report compiler (:mod:`dfl_sim.report`) publishes a strict
digest-bound ``canonical-report.json``. This module turns the *validated
semantic* report scalars into a small set of self-contained SVGs. It never
plots from raw traces and never re-derives a value: every bar, whisker and
tick is read directly from a report field, so a figure cannot invent an
outcome the audited report does not contain.

Design constraints (Q3/Q5):

* No plotting, export or browser-service dependency: SVG is emitted as text
  with :func:`xml.sax.saxutils.escape`, so every label is escaped and no
  external resource, ``<script>`` or ``href`` is ever emitted.
* Fully deterministic bytes: no timestamps, no timers, no unordered
  iteration. Policy/series/condition ordering is pinned to the report's own
  declared order (scope.policies) or a stable sort.
* Degenerate inputs are handled honestly: a flat series still renders a real
  axis, a null denominator renders an explicit ``n/a`` marker rather than 0,
  and runtime is never drawn on a byte axis.
* Figure digests are part of report identity; the figures are rendered from
  the semantic payload *before* the ``figures`` block is attached, so the
  report digest never depends on itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from xml.sax.saxutils import escape

__all__ = [
    "FIGURE_DIRNAME",
    "FIGURE_SCHEMA",
    "FIGURE_SCHEMA_VERSION",
    "FigureArtifact",
    "FigureError",
    "attach_figure_identity",
    "render_figures",
]

#: Versioned figure-identity schema, embedded in the canonical report.
FIGURE_SCHEMA = "dfl_canonical_figures"
FIGURE_SCHEMA_VERSION = 1
#: Sub-directory (relative to the report directory) that holds the SVGs.
FIGURE_DIRNAME = "figures"

_MAX_TEXT = 240

# A deterministic, colorblind-reasonable categorical palette; colours are
# assigned by the report's own declared policy order (never by a hash).
_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

_W = 1180.0
_PAD = 24.0
_HEADER_H = 86.0
_FOOTER_H = 40.0
_PANEL_GAP = 24.0
_Y_AXIS_W = 74.0
_X_LABEL_H = 52.0
_PANEL_TITLE_H = 30.0
_PLOT_TOP_PAD = 10.0


class FigureError(RuntimeError):
    """Raised when a figure cannot be rendered fail-closed."""


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _bounded(text: object) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) > _MAX_TEXT:
        return collapsed[: _MAX_TEXT - 1] + "\u2026"
    return collapsed


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise FigureError(detail)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text(value: object) -> str:
    return escape(_bounded(value), {'"': "&quot;", "'": "&apos;"})


def _fmt(value: float, digits: int = 3) -> str:
    if value == 0.0:
        value = 0.0
    text = f"{value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return text


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FigureError(f"{label} must be a finite number")
    result = float(value)
    if not (result == result and result not in (float("inf"), float("-inf"))):
        raise FigureError(f"{label} must be finite")
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FigureError(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> list:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FigureError(f"{label} must be a list")
    return list(value)


def _opt_float(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite(value, label=label)


def _tick_values(y_min: float, y_max: float, count: int = 4) -> list[float]:
    if count < 1:
        return [y_min, y_max]
    span = y_max - y_min
    if span <= 0.0:
        return [y_min]
    return [y_min + span * index / count for index in range(count + 1)]


def _line(x1: float, y1: float, x2: float, y2: float, *, stroke: str, width: float = 1.0) -> str:
    return (
        f'<line x1="{_fmt(x1, 2)}" y1="{_fmt(y1, 2)}" '
        f'x2="{_fmt(x2, 2)}" y2="{_fmt(y2, 2)}" '
        f'stroke="{stroke}" stroke-width="{_fmt(width, 2)}"/>'
    )


def _rect(x: float, y: float, w: float, h: float, *, fill: str, stroke: str = "none", width: float = 0.0) -> str:
    width_attr = f' stroke-width="{_fmt(width, 2)}"' if width else ""
    return (
        f'<rect x="{_fmt(x, 2)}" y="{_fmt(y, 2)}" width="{_fmt(max(w, 0.0), 2)}" '
        f'height="{_fmt(max(h, 0.0), 2)}" fill="{fill}" stroke="{stroke}"{width_attr}/>'
    )


def _text_el(
    x: float,
    y: float,
    value: object,
    *,
    size: float = 12.0,
    anchor: str = "start",
    fill: str = "#222222",
    weight: str = "normal",
    rotate: float | None = None,
) -> str:
    transform = (
        f' transform="rotate({_fmt(rotate, 2)} {_fmt(x, 2)} {_fmt(y, 2)})"'
        if rotate is not None
        else ""
    )
    return (
        f'<text x="{_fmt(x, 2)}" y="{_fmt(y, 2)}" font-family="sans-serif" '
        f'font-size="{_fmt(size, 2)}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}"{transform}>{_text(value)}</text>'
    )


# --------------------------------------------------------------------------- #
# Extraction helpers (read-only, never re-derive)
# --------------------------------------------------------------------------- #
def _policies(scope: Mapping[str, object]) -> list[str]:
    raw = _sequence(scope.get("policies"), label="scope.policies")
    result: list[str] = []
    for item in raw:
        text = str(item)
        if not text:
            raise FigureError("scope.policies contains an empty id")
        if text in result:
            raise FigureError(f"duplicate policy id {text!r}")
        result.append(text)
    _require(bool(result), "scope.policies is empty")
    return result


def _conditions(report: Mapping[str, object]) -> list[str]:
    result = [str(_mapping(item, label="condition")["condition_id"]) for item in _sequence(report.get("conditions"), label="conditions")]
    _require(bool(result), "report has no conditions")
    if len(set(result)) != len(result):
        raise FigureError("duplicate condition ids")
    return sorted(result)


def _aggregates(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    metrics = _mapping(report.get("metrics"), label="metrics")
    return [_mapping(item, label="aggregate") for item in _sequence(metrics.get("aggregates"), label="metrics.aggregates")]


def _deltas(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    metrics = _mapping(report.get("metrics"), label="metrics")
    return [_mapping(item, label="delta") for item in _sequence(metrics.get("paired_deltas"), label="metrics.paired_deltas")]


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def _svg_document(width: float, height: float, title: str, desc: str, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_fmt(width, 2)}" '
        f'height="{_fmt(height, 2)}" viewBox="0 0 {_fmt(width, 2)} {_fmt(height, 2)}" '
        f'role="img" aria-label="{_text(title)}">'
        f"<title>{_text(title)}</title><desc>{_text(desc)}</desc>"
        f'<rect x="0" y="0" width="{_fmt(width, 2)}" height="{_fmt(height, 2)}" fill="#ffffff"/>'
        + "".join(body)
        + "</svg>"
    )


def _legend(x: float, y: float, labels: Sequence[str], colors: Sequence[str]) -> str:
    parts: list[str] = []
    cursor = x
    for index, label in enumerate(labels):
        color = colors[index % len(colors)]
        parts.append(_rect(cursor, y - 9.0, 11.0, 11.0, fill=color))
        parts.append(_text_el(cursor + 16.0, y, label, size=11.0))
        cursor += 16.0 + max(len(label), 3) * 6.6 + 16.0
    return "".join(parts)


def _bar_panel(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    groups: Sequence[str],
    series: Sequence[str],
    values: Sequence[Sequence[float | None]],
    y_min: float,
    y_max: float,
    y_label: str,
    colors: Sequence[str],
    errors: Sequence[Sequence[tuple[float, float] | None]] | None = None,
    label_rotate: float | None = None,
    group_axis_label: str | None = None,
) -> str:
    _require(len(values) == len(groups), "panel values/groups length mismatch")
    for column in values:
        _require(len(column) == len(series), "panel values/series length mismatch")
    _require(y_max > y_min, "panel axis has no span")

    plot_x = x + _Y_AXIS_W
    plot_y = y + _PANEL_TITLE_H + _PLOT_TOP_PAD
    plot_w = max(w - _Y_AXIS_W - 12.0, 40.0)
    plot_h = max(h - _PANEL_TITLE_H - _PLOT_TOP_PAD - _X_LABEL_H, 40.0)
    baseline_y = plot_y + plot_h - (0.0 - y_min) / (y_max - y_min) * plot_h

    parts: list[str] = [_text_el(x, y + 16.0, title, size=13.0, weight="bold")]

    ticks = _tick_values(y_min, y_max, 4)
    for tick in ticks:
        ty = plot_y + (y_max - tick) / (y_max - y_min) * plot_h
        parts.append(_line(plot_x, ty, plot_x + plot_w, ty, stroke="#e6e6e6", width=1.0))
        parts.append(_text_el(plot_x - 8.0, ty + 4.0, _fmt(tick, 3), size=10.0, anchor="end", fill="#555555"))
    parts.append(_line(plot_x, plot_y, plot_x, plot_y + plot_h, stroke="#999999", width=1.0))
    parts.append(_line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, stroke="#999999", width=1.0))
    parts.append(_text_el(x + 4.0, plot_y - 2.0, y_label, size=10.0, fill="#555555"))

    group_span = plot_w / max(len(groups), 1)
    bar_span = group_span * 0.82
    bar_w = bar_span / max(len(series), 1)
    for group_index, group in enumerate(groups):
        gx = plot_x + group_index * group_span
        parts.append(_line(gx, plot_y, gx, plot_y + plot_h, stroke="#f2f2f2", width=1.0))
        center = gx + group_span / 2.0
        for series_index in range(len(series)):
            value = values[group_index][series_index]
            bx = center - bar_span / 2.0 + series_index * bar_w
            if value is None:
                parts.append(_rect(bx + bar_w * 0.15, baseline_y - 10.0, bar_w * 0.7, 10.0, fill="#dddddd"))
                parts.append(
                    _text_el(bx + bar_w / 2.0, baseline_y - 14.0, "n/a", size=8.5, anchor="middle", fill="#777777")
                )
                continue
            top = plot_y + (y_max - value) / (y_max - y_min) * plot_h
            top = min(max(top, plot_y), plot_y + plot_h)
            parts.append(_rect(bx, top, bar_w * 0.86, baseline_y - top, fill=colors[series_index % len(colors)]))
            if errors is not None and errors[group_index][series_index] is not None:
                low, high = errors[group_index][series_index]  # type: ignore[misc]
                low = min(max(low, y_min), y_max)
                high = min(max(high, y_min), y_max)
                low_y = plot_y + (y_max - low) / (y_max - y_min) * plot_h
                high_y = plot_y + (y_max - high) / (y_max - y_min) * plot_h
                cx = bx + bar_w * 0.43
                parts.append(_line(cx, high_y, cx, low_y, stroke="#333333", width=1.0))
                parts.append(_line(cx - 3.0, high_y, cx + 3.0, high_y, stroke="#333333", width=1.0))
                parts.append(_line(cx - 3.0, low_y, cx + 3.0, low_y, stroke="#333333", width=1.0))
        label_y = plot_y + plot_h + 14.0
        parts.append(_text_el(center, label_y, group, size=9.5, anchor="end", rotate=label_rotate))
    if group_axis_label:
        parts.append(
            _text_el(plot_x + plot_w / 2.0, plot_y + plot_h + _X_LABEL_H - 4.0, group_axis_label, size=10.0, anchor="middle", fill="#555555")
        )
    return "".join(parts)


def _panel_grid(count: int, columns: int) -> tuple[int, int]:
    columns = max(1, min(columns, count))
    rows = (count + columns - 1) // columns
    return rows, columns


def _panel_width(columns: int) -> float:
    return (_W - 2.0 * _PAD - (columns - 1) * _PANEL_GAP) / columns


# --------------------------------------------------------------------------- #
# Figure 1: accuracy by condition / schedule
# --------------------------------------------------------------------------- #
def _figure_accuracy(report: Mapping[str, object], scope: Mapping[str, object], provenance: str) -> tuple[str, str, str]:
    policies = _policies(scope)
    conditions = _conditions(report)
    aggregates = _aggregates(report)
    schedules = sorted({str(row["schedule_id"]) for row in aggregates})
    _require(bool(schedules), "no final schedules in report")

    colors = [_PALETTE[index % len(_PALETTE)] for index in range(len(policies))]
    rows, columns = _panel_grid(len(conditions), 3)
    panel_w = _panel_width(columns)
    panel_h = 210.0

    body: list[str] = []
    title = "Final accuracy by condition and schedule"
    body.append(_text_el(_PAD, 30.0, title, size=18.0, weight="bold"))
    body.append(
        _text_el(
            _PAD,
            50.0,
            f"provenance: {provenance} | mean of final seeds; whisker = population std over n final seeds; population spread/IQR, not a confidence interval",
            size=11.0,
            fill="#555555",
        )
    )
    body.append(
        _text_el(
            _PAD,
            66.0,
            f"matrix scope: {len(conditions)} condition(s), final-only categories include held_out; bounded IID headline is reported separately",
            size=11.0,
            fill="#555555",
        )
    )
    body.append(_legend(_PAD, 84.0, policies, colors))

    index: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for row in aggregates:
        key = (str(row["condition_id"]), str(row["schedule_id"]), str(row["policy_id"]))
        _require(key not in index, f"duplicate aggregate {key!r}")
        index[key] = row

    for condition_index, condition in enumerate(conditions):
        grid_row, grid_col = divmod(condition_index, columns)
        x = _PAD + grid_col * (panel_w + _PANEL_GAP)
        y = _HEADER_H + grid_row * (panel_h + _PANEL_GAP)
        values: list[list[float | None]] = []
        errors: list[list[tuple[float, float] | None]] = []
        for schedule in schedules:
            value_row: list[float | None] = []
            error_row: list[tuple[float, float] | None] = []
            for policy in policies:
                row = index.get((condition, schedule, policy))
                if row is None:
                    value_row.append(None)
                    error_row.append(None)
                    continue
                mean = _finite(row["mean_accuracy"], label="mean_accuracy")
                spread = abs(_finite(row.get("std_accuracy", 0.0), label="std_accuracy"))
                value_row.append(min(max(mean, 0.0), 1.0))
                error_row.append((max(mean - spread, 0.0), min(mean + spread, 1.0)))
            values.append(value_row)
            errors.append(error_row)
        body.append(
            _bar_panel(
                x,
                y,
                panel_w,
                panel_h,
                title=f"condition {condition}",
                groups=schedules,
                series=policies,
                values=values,
                y_min=0.0,
                y_max=1.0,
                y_label="mean accuracy [0-1]",
                colors=colors,
                errors=errors,
                label_rotate=-28.0,
                group_axis_label="schedule",
            )
        )

    total_rows = rows
    height = _HEADER_H + total_rows * panel_h + (total_rows - 1) * _PANEL_GAP + _FOOTER_H
    body.append(
        _text_el(
            _PAD,
            height - 12.0,
            "Every policy present in the audited panel is drawn; missing cells show n/a rather than zero.",
            size=10.0,
            fill="#666666",
        )
    )
    desc = "Final-accuracy small multiples for each audited condition and schedule."
    return ("accuracy_by_condition", "accuracy-by-condition.svg", title, _svg_document(_W, height, title, desc, body))


# --------------------------------------------------------------------------- #
# Figure 2: benign exclusions versus malicious acceptance
# --------------------------------------------------------------------------- #
def _figure_rejection(report: Mapping[str, object], scope: Mapping[str, object], provenance: str) -> tuple[str, str, str]:
    policies = _policies(scope)
    conditions = _conditions(report)
    rejection = _mapping(report.get("rejection"), label="rejection")
    records = [_mapping(item, label="rejection row") for item in _sequence(rejection.get("by_condition_policy"), label="rejection.by_condition_policy")]
    index: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in records:
        index[(str(row["condition_id"]), str(row["policy_id"]))] = row

    series = ["benign exclusion", "malicious active acceptance"]
    colors = ["#1f77b4", "#d62728"]
    rows, columns = _panel_grid(len(conditions), 3)
    panel_w = _panel_width(columns)
    panel_h = 220.0

    body: list[str] = []
    title = "Non-IID benign exclusions versus malicious acceptance"
    body.append(_text_el(_PAD, 30.0, title, size=18.0, weight="bold"))
    body.append(
        _text_el(
            _PAD,
            50.0,
            f"provenance: {provenance} | rate = excluded/accepted opportunities; a zero denominator is drawn as n/a, never as 0",
            size=11.0,
            fill="#555555",
        )
    )
    body.append(
        _text_el(
            _PAD,
            66.0,
            "excluded = receiver weight exactly zero (not a detector positive); active = structurally malicious sender with realized kappa > 0",
            size=11.0,
            fill="#555555",
        )
    )
    body.append(_legend(_PAD, 84.0, series, colors))

    for condition_index, condition in enumerate(conditions):
        grid_row, grid_col = divmod(condition_index, columns)
        x = _PAD + grid_col * (panel_w + _PANEL_GAP)
        y = _HEADER_H + grid_row * (panel_h + _PANEL_GAP)
        values: list[list[float | None]] = []
        errors: list[list[tuple[float, float] | None]] = []
        for policy in policies:
            row = index.get((condition, policy))
            if row is None:
                values.append([None, None])
                errors.append([None, None])
                continue
            exclusion = _opt_float(row.get("benign_exclusion_rate"), label="benign_exclusion_rate")
            acceptance = _opt_float(row.get("malicious_active_acceptance_rate"), label="malicious_active_acceptance_rate")
            values.append([exclusion, acceptance])
            errors.append([None, None])
        body.append(
            _bar_panel(
                x,
                y,
                panel_w,
                panel_h,
                title=f"condition {condition}",
                groups=policies,
                series=series,
                values=values,
                y_min=0.0,
                y_max=1.0,
                y_label="rate [0-1]",
                colors=colors,
                errors=errors,
                label_rotate=-32.0,
                group_axis_label="policy",
            )
        )

    height = _HEADER_H + rows * panel_h + (rows - 1) * _PANEL_GAP + _FOOTER_H
    body.append(
        _text_el(
            _PAD,
            height - 12.0,
            "Benign exclusion and malicious acceptance are separate axes of usefulness; neither is pooled into a single defense score.",
            size=10.0,
            fill="#666666",
        )
    )
    desc = "Benign exclusion rates versus malicious active acceptance by condition and policy."
    return ("rejection_acceptance", "rejection-acceptance.svg", title, _svg_document(_W, height, title, desc, body))


# --------------------------------------------------------------------------- #
# Figure 3: placement, propagation and worst-node outcomes
# --------------------------------------------------------------------------- #
def _figure_placement(report: Mapping[str, object], provenance: str) -> tuple[str, str, str]:
    conditions = _conditions(report)
    condition_rows = {
        str(_mapping(item, label="condition")["condition_id"]): _mapping(item, label="condition")
        for item in _sequence(report.get("conditions"), label="conditions")
    }
    aggregates = _aggregates(report)
    propagation = _mapping(report.get("propagation"), label="propagation")
    propagation_rows = [_mapping(item, label="propagation row") for item in _sequence(propagation.get("by_condition_policy"), label="propagation.by_condition_policy")]

    degree_means: list[float | None] = []
    degree_mins: list[float | None] = []
    degree_maxs: list[float | None] = []
    for condition in conditions:
        identities = _sequence(condition_rows[condition].get("placement_identities"), label="placement_identities")
        degrees = [
            _finite(_mapping(item, label="placement identity").get("degree"), label="placement degree")
            for item in identities
        ]
        if not degrees:
            degree_means.append(None)
            degree_mins.append(None)
            degree_maxs.append(None)
            continue
        degree_means.append(sum(degrees) / len(degrees))
        degree_mins.append(min(degrees))
        degree_maxs.append(max(degrees))

    reached_fraction: list[float | None] = []
    for condition in conditions:
        rows = [row for row in propagation_rows if str(row["condition_id"]) == condition]
        reached = sum(int(_finite(row.get("reached_node_rounds", 0), label="reached_node_rounds")) for row in rows)
        total = sum(int(_finite(row.get("total_node_rounds", 0), label="total_node_rounds")) for row in rows)
        reached_fraction.append(None if total == 0 else reached / total)

    worst: list[float | None] = []
    for condition in conditions:
        rows = [row for row in aggregates if str(row["condition_id"]) == condition]
        values = [_finite(row["worst_node_accuracy"], label="worst_node_accuracy") for row in rows]
        worst.append(min(values) if values else None)

    body: list[str] = []
    title = "Placement, propagation and worst-node outcomes"
    body.append(_text_el(_PAD, 30.0, title, size=18.0, weight="bold"))
    body.append(
        _text_el(
            _PAD,
            50.0,
            f"provenance: {provenance} | three independent panels with distinct units; placement is a node-count degree, propagation a fraction, worst-node an accuracy",
            size=11.0,
            fill="#555555",
        )
    )
    panel_h = 240.0
    top = _HEADER_H

    max_degree = max((value for value in (degree_maxs + degree_means) if value is not None), default=1.0)
    degree_axis = max(max_degree * 1.15, 1.0)
    body.append(
        _bar_panel(
            _PAD,
            top,
            _W - 2.0 * _PAD,
            panel_h,
            title="mean placement degree by condition (whisker = min..max node degree)",
            groups=conditions,
            series=["mean degree"],
            values=[[value] for value in degree_means],
            y_min=0.0,
            y_max=degree_axis,
            y_label="degree [count]",
            colors=["#2ca02c"],
            errors=[[((degree_mins[i] or 0.0), (degree_maxs[i] or 0.0)) if degree_means[i] is not None else None] for i in range(len(conditions))],
            label_rotate=-30.0,
            group_axis_label="condition",
        )
    )
    top += panel_h + _PANEL_GAP
    body.append(
        _bar_panel(
            _PAD,
            top,
            _W - 2.0 * _PAD,
            panel_h,
            title="poison propagation: reached node-round fraction (summed over policies)",
            groups=conditions,
            series=["reached fraction"],
            values=[[value] for value in reached_fraction],
            y_min=0.0,
            y_max=1.0,
            y_label="fraction [0-1]",
            colors=["#9467bd"],
            errors=None,
            label_rotate=-30.0,
            group_axis_label="condition",
        )
    )
    top += panel_h + _PANEL_GAP
    body.append(
        _bar_panel(
            _PAD,
            top,
            _W - 2.0 * _PAD,
            panel_h,
            title="worst observed node accuracy (minimum over policies and schedules)",
            groups=conditions,
            series=["worst-node accuracy"],
            values=[[value] for value in worst],
            y_min=0.0,
            y_max=1.0,
            y_label="accuracy [0-1]",
            colors=["#d62728"],
            errors=None,
            label_rotate=-30.0,
            group_axis_label="condition",
        )
    )
    height = _HEADER_H + 3.0 * panel_h + 2.0 * _PANEL_GAP + _FOOTER_H
    body.append(
        _text_el(
            _PAD,
            height - 12.0,
            "Placement degrees come from the frozen per-seed placement identities; propagation is weight-aware and monotone.",
            size=10.0,
            fill="#666666",
        )
    )
    desc = "Placement degree, propagation reach and worst-node accuracy as separate panels."
    return ("placement_propagation_worst_node", "placement-propagation-worst-node.svg", title, _svg_document(_W, height, title, desc, body))


# --------------------------------------------------------------------------- #
# Figure 4: paired ablation deltas
# --------------------------------------------------------------------------- #
def _figure_deltas(report: Mapping[str, object], scope: Mapping[str, object], provenance: str) -> tuple[str, str, str]:
    policies = _policies(scope)
    deltas = _deltas(report)
    references = sorted({str(row["reference_policy"]) for row in deltas})
    _require(bool(references), "no paired delta references in report")

    colors = [_PALETTE[index % len(_PALETTE)] for index in range(len(policies))]
    panel_h = 250.0
    rows, columns = _panel_grid(len(references), 2)
    panel_w = _panel_width(columns)

    body: list[str] = []
    title = "Paired ablation deltas versus reference policies"
    body.append(_text_el(_PAD, 30.0, title, size=18.0, weight="bold"))
    body.append(
        _text_el(
            _PAD,
            50.0,
            f"provenance: {provenance} | paired per-seed accuracy difference on identical (condition, schedule, seed); whisker = population std of deltas",
            size=11.0,
            fill="#555555",
        )
    )
    body.append(
        _text_el(
            _PAD,
            66.0,
            "positive means better than the reference; zero line is drawn explicitly and no winner is claimed",
            size=11.0,
            fill="#555555",
        )
    )
    body.append(_legend(_PAD, 84.0, policies, colors))

    index: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in deltas:
        index.setdefault((str(row["reference_policy"]), str(row["policy_id"])), []).append(row)

    for reference_index, reference in enumerate(references):
        grid_row, grid_col = divmod(reference_index, columns)
        x = _PAD + grid_col * (panel_w + _PANEL_GAP)
        y = _HEADER_H + grid_row * (panel_h + _PANEL_GAP)
        means: list[float | None] = []
        errors: list[tuple[float, float] | None] = []
        for policy in policies:
            matching = index.get((reference, policy), [])
            if not matching:
                means.append(None)
                errors.append(None)
                continue
            mean = sum(_finite(row["mean_delta"], label="mean_delta") for row in matching) / len(matching)
            spread = max(abs(_finite(row.get("std_delta", 0.0), label="std_delta")) for row in matching)
            means.append(mean)
            errors.append((mean - spread, mean + spread))
        span = max([abs(value) for value in means if value is not None] + [abs(lo) for lo, _ in [e for e in errors if e]] + [0.0])
        span = max(span * 1.2, 0.05)
        body.append(
            _bar_panel(
                x,
                y,
                panel_w,
                panel_h,
                title=f"reference {reference}",
                groups=policies,
                series=["mean delta"],
                values=[[value] for value in means],
                y_min=-span,
                y_max=span,
                y_label="accuracy delta [0-1]",
                colors=colors,
                errors=[[error] for error in errors],
                label_rotate=-32.0,
                group_axis_label="policy",
            )
        )

    height = _HEADER_H + rows * panel_h + (rows - 1) * _PANEL_GAP + _FOOTER_H
    body.append(
        _text_el(
            _PAD,
            height - 12.0,
            "Each policy is paired against its reference on the same seed set; a missing pair is drawn as n/a.",
            size=10.0,
            fill="#666666",
        )
    )
    desc = "Paired per-seed accuracy deltas against each declared reference policy."
    return ("paired_ablation_deltas", "paired-ablation-deltas.svg", title, _svg_document(_W, height, title, desc, body))


# --------------------------------------------------------------------------- #
# Figure 5: computation versus communication (independent axes/units)
# --------------------------------------------------------------------------- #
def _figure_costs(report: Mapping[str, object], scope: Mapping[str, object], provenance: str) -> tuple[str, str, str]:
    policies = _policies(scope)
    costs = _mapping(report.get("costs"), label="costs")
    computation = _mapping(costs.get("computation"), label="costs.computation")
    communication = _mapping(costs.get("communication"), label="costs.communication")
    comp_rows = [_mapping(item, label="computation row") for item in _sequence(computation.get("by_condition_policy"), label="computation.by_condition_policy")]
    comm_rows = [_mapping(item, label="communication row") for item in _sequence(communication.get("by_condition_policy"), label="communication.by_condition_policy")]

    def total_by_policy(rows: Sequence[Mapping[str, object]], key: str) -> list[float | None]:
        totals: dict[str, float] = {policy: 0.0 for policy in policies}
        seen: set[str] = set()
        for row in rows:
            policy = str(row["policy_id"])
            if policy not in totals:
                continue
            totals[policy] += _finite(row.get(key, 0), label=key)
            seen.add(policy)
        return [totals[policy] if policy in seen else None for policy in policies]

    comp_values = total_by_policy(comp_rows, "whole_cell_ns")
    comm_values = total_by_policy(comm_rows, "logical_payload_bytes")
    comp_max = max((value for value in comp_values if value is not None), default=0.0)
    comm_max = max((value for value in comm_values if value is not None), default=0.0)

    body: list[str] = []
    title = "Computation and communication are separate measured quantities"
    body.append(_text_el(_PAD, 30.0, title, size=18.0, weight="bold"))
    body.append(
        _text_el(
            _PAD,
            50.0,
            f"provenance: {provenance} | runtime is nanoseconds (perf_counter_ns) and is never drawn on a byte axis; communication is logical payload bytes before receiver filtering",
            size=11.0,
            fill="#555555",
        )
    )
    headline = _mapping(scope.get("headline"), label="scope.headline")
    headline_note = (
        "bounded IID headline cost sidecars are unavailable (reported separately)"
        if not bool(headline.get("available"))
        else "bounded IID headline is verified and scoped separately"
    )
    body.append(_text_el(_PAD, 66.0, headline_note, size=11.0, fill="#555555"))

    panel_h = 300.0
    panel_w = (_W - 2.0 * _PAD - _PANEL_GAP) / 2.0
    body.append(
        _bar_panel(
            _PAD,
            _HEADER_H,
            panel_w,
            panel_h,
            title="measured computation",
            groups=policies,
            series=["whole cell (ns)"],
            values=[[value] for value in comp_values],
            y_min=0.0,
            y_max=max(comp_max * 1.15, 1.0),
            y_label="nanoseconds [ns]",
            colors=["#1f77b4"],
            errors=None,
            label_rotate=-32.0,
            group_axis_label="policy",
        )
    )
    body.append(
        _bar_panel(
            _PAD + panel_w + _PANEL_GAP,
            _HEADER_H,
            panel_w,
            panel_h,
            title="logical communication",
            groups=policies,
            series=["payload (bytes)"],
            values=[[value] for value in comm_values],
            y_min=0.0,
            y_max=max(comm_max * 1.15, 1.0),
            y_label="bytes [B]",
            colors=["#ff7f0e"],
            errors=None,
            label_rotate=-32.0,
            group_axis_label="policy",
        )
    )
    height = _HEADER_H + panel_h + _FOOTER_H
    body.append(
        _text_el(
            _PAD,
            height - 12.0,
            "The two panels share no scale: a nanosecond total and a byte total are different measured quantities.",
            size=10.0,
            fill="#666666",
        )
    )
    desc = "Measured runtime in nanoseconds beside logical payload in bytes, on independent axes."
    return ("compute_vs_communication", "compute-vs-communication.svg", title, _svg_document(_W, height, title, desc, body))


# --------------------------------------------------------------------------- #
# Public render / identity
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FigureArtifact:
    figure_id: str
    filename: str
    title: str
    svg: str
    digest: str


def render_figures(report: Mapping[str, object]) -> tuple[FigureArtifact, ...]:
    """Render every canonical figure from a validated semantic report.

    Raises :class:`FigureError` if a required scalar is missing, non-finite or
    structurally malformed. The semantic report is never mutated.
    """

    report = _mapping(report, label="report")
    scope = _mapping(report.get("scope"), label="scope")
    provenance = str(report.get("data_provenance", scope.get("data_provenance", "unknown")))
    _require(provenance in ("real_mnist", "injected"), f"unknown provenance {provenance!r}")

    builders = (
        _figure_accuracy,
        _figure_rejection,
        _figure_placement,
        _figure_deltas,
        _figure_costs,
    )
    artifacts: list[FigureArtifact] = []
    for builder in builders:
        if builder is _figure_placement:
            figure_id, filename, title, svg = builder(report, provenance)  # type: ignore[misc]
        elif builder is _figure_costs:
            figure_id, filename, title, svg = builder(report, scope, provenance)  # type: ignore[misc]
        else:
            figure_id, filename, title, svg = builder(report, scope, provenance)  # type: ignore[misc]
        artifacts.append(
            FigureArtifact(
                figure_id=figure_id,
                filename=filename,
                title=title,
                svg=svg,
                digest=_sha256_text(svg),
            )
        )
    return tuple(sorted(artifacts, key=lambda artifact: artifact.figure_id))


def attach_figure_identity(
    report: Mapping[str, object],
) -> tuple[dict, tuple[FigureArtifact, ...]]:
    """Attach the figure block and re-bind the report digest.

    Figures are rendered from the semantic payload first; only then is the
    ``figures`` block (ids, filenames and SVG digests) inserted and the report
    digest recomputed over the full body. This ordering makes the identity a
    DAG: the report digest covers the figures, and the figures never cover the
    report digest.
    """

    report = _mapping(report, label="report")
    _require(report.get("figures") in (None,), "report already carries a figures block")
    _require("report_digest" in report, "report is missing its semantic digest")
    figures = render_figures(report)
    entries = [
        {
            "id": artifact.figure_id,
            "filename": artifact.filename,
            "title": artifact.title,
            "digest": artifact.digest,
        }
        for artifact in figures
    ]
    payload = {
        key: value
        for key, value in report.items()
        if key not in ("figures", "report_digest")
    }
    payload["figures"] = {
        "schema": FIGURE_SCHEMA,
        "schema_version": FIGURE_SCHEMA_VERSION,
        "entries": entries,
    }
    payload["report_digest"] = _sha256_json(payload)
    return payload, figures
