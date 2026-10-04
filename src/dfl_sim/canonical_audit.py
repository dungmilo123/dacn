"""Production acceptance audit for the canonical robustness study (M003/S05/T07).

``canonical_audit`` is a small read-only production CLI layered over the two
independent verifiers already shipped in this slice:

* :func:`dfl_sim.report.verify_canonical_report` re-audits the raw matrix,
  the separate bounded IID headline, every published figure and the full
  canonical report body; and
* :func:`dfl_sim.mnist_provision.verify_cache` deep-recomputes the downloaded
  MNIST content identity.

On top of those it adds the exact *acceptance* assertions the slice demo
promises: declared nine-condition coverage, the eight-policy panel, the three
Dirichlet alphas and three degree-placement ranks, actual graph/node/degree
placement identities, a disjoint selection/final/calibration seed split, a
final-only ``held_out`` schedule, real downloaded provenance, and resolvable,
root-confined replay links. It never mutates the study root (or the report);
with ``--output`` it writes exactly one digest-bound ``runtime-audit.json``
atomically.

It is deliberately *not* a pytest: the production path is real-MNIST only and
never reads repository ``runs/``/``data/`` in tests. The offline verifier
suites from T04-T06 already prove the negative behavior; this command is the
real-data acceptance gate and the isolation seams below exist only so a
reduced injected fixture can exercise the same assertion logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .report import (
    CanonicalReportDeps,
    CanonicalReportError,
    verify_canonical_report,
)
from .robustness_config import (
    CANONICAL_ALPHAS,
    CANONICAL_CALIBRATION_SEED,
    CANONICAL_FINAL_SEEDS,
    CANONICAL_PLACEMENTS,
    CANONICAL_SELECTION_SEEDS,
    CANONICAL_TOPOLOGY,
)
from .robustness_study import ROBUSTNESS_PROTOCOL_FILENAME
from .robustness_verify import RobustnessVerifyError

__all__ = [
    "AUDIT_FILENAME",
    "AUDIT_SCHEMA",
    "AUDIT_SCHEMA_VERSION",
    "AUDIT_SUCCESS_MARKER",
    "CanonicalAuditDeps",
    "CanonicalAuditError",
    "CanonicalAuditResult",
    "audit_canonical_study",
    "build_parser",
    "main",
]

AUDIT_SCHEMA = "dfl_canonical_audit"
AUDIT_SCHEMA_VERSION = 1
AUDIT_FILENAME = "runtime-audit.json"
AUDIT_SUCCESS_MARKER = "dfl_sim: canonical audit passed "

#: Frozen canonical acceptance numbers (D040); cross-checked against the
#: on-disk protocol for real_MNIST so a weakened source can never pass.
EXPECTED_ALPHAS: tuple[float, ...] = CANONICAL_ALPHAS
EXPECTED_PLACEMENTS: tuple[str, ...] = CANONICAL_PLACEMENTS
EXPECTED_SELECTION_SEEDS: tuple[int, ...] = CANONICAL_SELECTION_SEEDS
EXPECTED_FINAL_SEEDS: tuple[int, ...] = CANONICAL_FINAL_SEEDS
EXPECTED_CALIBRATION_SEED = CANONICAL_CALIBRATION_SEED
EXPECTED_CONDITIONS = 9
EXPECTED_POLICIES = 8
EXPECTED_SELECTION_CELLS = 576
EXPECTED_FINAL_CELLS = 1080
EXPECTED_AGGREGATES = 360
EXPECTED_ROUNDS = 6
EXPECTED_SAMPLES_PER_NODE = 64
EXPECTED_EVAL_SAMPLES = 128
EXPECTED_TOPOLOGY_N = CANONICAL_TOPOLOGY["n"]
EXPECTED_TOPOLOGY_K = CANONICAL_TOPOLOGY["k"]
EXPECTED_TOPOLOGY_P = CANONICAL_TOPOLOGY["p"]
KNOWN_SCHEDULE_IDS = frozenset({"duty_cycle", "periodic", "delayed_onset", "random"})
HELD_OUT_SCHEDULE_ID = "held_out"

_MAX_ERROR_TEXT = 400


class CanonicalAuditError(RuntimeError):
    """Raised when the canonical study fails independent acceptance."""


# --------------------------------------------------------------------------- #
# Bounded helpers
# --------------------------------------------------------------------------- #
def _bounded(text: object, *, limit: int = _MAX_ERROR_TEXT) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) > limit:
        return collapsed[: limit - 1] + "\u2026"
    return collapsed


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise CanonicalAuditError(detail)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise CanonicalAuditError(f"missing audit source: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanonicalAuditError(f"could not read {path}: {_bounded(exc)}") from exc
    if not isinstance(raw, Mapping):
        raise CanonicalAuditError(f"{path} is not a JSON object")
    return raw


def _confined_realpath(root: Path, relative: object, *, label: str) -> Path:
    """Resolve ``relative`` under ``root`` and reject traversal/symlink escape."""

    if not isinstance(relative, str) or not relative:
        raise CanonicalAuditError(f"{label} path must be a non-empty string")
    if os.path.isabs(relative):
        raise CanonicalAuditError(f"{label} path must be relative, got {relative!r}")
    try:
        real_root = os.path.realpath(root)
        real = os.path.realpath(os.path.join(real_root, relative))
    except OSError as exc:  # pragma: no cover - defensive
        raise CanonicalAuditError(f"could not resolve {label} path: {_bounded(exc)}") from exc
    if real != real_root and not real.startswith(real_root + os.sep):
        raise CanonicalAuditError(f"{label} path escapes the study root")
    return Path(real)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(text: str, base: Path) -> Path:
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate


# --------------------------------------------------------------------------- #
# Injection seams
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CanonicalAuditDeps:
    """Optional injection seams; production uses the real public surfaces."""

    report_deps: CanonicalReportDeps | None = None
    repo_root: str | os.PathLike[str] | None = None
    cache_verify: Callable[..., object] | None = None
    allow_injected: bool = False


@dataclass(frozen=True)
class CanonicalAuditResult:
    root: Path
    status: str
    audit: Mapping[str, object]
    audit_digest: str
    output_path: Path | None
    conditions: int
    policies: int


# --------------------------------------------------------------------------- #
# Deep cache
# --------------------------------------------------------------------------- #
def _deep_cache_digest(data_root: Path, cache_verify: Callable[..., object] | None) -> str:
    verify = cache_verify
    if verify is None:
        from . import mnist_provision

        verify = mnist_provision.verify_cache
    try:
        manifest = verify(str(data_root), deep=True)
    except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
        raise CanonicalAuditError(
            f"deep MNIST cache verification failed closed: {_bounded(exc)}"
        ) from exc
    digest = getattr(manifest, "manifest_digest", None)
    _require(
        isinstance(digest, str) and len(digest) == 64,
        "deep MNIST cache verification returned no content digest",
    )
    return digest


# --------------------------------------------------------------------------- #
# Coverage / identity assertions
# --------------------------------------------------------------------------- #
def _expected_rank(placement: str, node_count: int) -> int:
    if placement == "low":
        return 0
    if placement == "high":
        return node_count - 1
    return (node_count - 1) // 2


def _assert_scope(
    report: Mapping[str, object], protocol: Mapping[str, object], *, real: bool
) -> dict:
    scope = report.get("scope")
    _require(isinstance(scope, Mapping), "canonical report scope is missing")
    assert isinstance(scope, Mapping)
    conditions = report.get("conditions")
    _require(isinstance(conditions, list), "canonical report conditions are missing")
    assert isinstance(conditions, list)

    alphas = tuple(scope.get("alphas", ()))
    placements = tuple(scope.get("placements", ()))
    policies = tuple(scope.get("policies", ()))
    selection_seeds = tuple(scope.get("selection_seeds", ()))
    final_seeds = tuple(scope.get("final_seeds", ()))
    calibration_seed = scope.get("calibration_seed")

    known_schedules = protocol.get("known_schedules")
    _require(isinstance(known_schedules, list), "protocol known_schedules are missing")
    assert isinstance(known_schedules, list)
    held_out = protocol.get("held_out_schedule")
    _require(isinstance(held_out, Mapping), "protocol held_out_schedule is missing")
    assert isinstance(held_out, Mapping)
    held_out_id = str(held_out.get("id"))

    # -- declared dimensions (both real and injected) ---------------------- #
    _require(
        tuple(protocol.get("alphas", ())) == alphas,
        "report alphas disagree with the frozen protocol",
    )
    _require(
        tuple(protocol.get("placements", ())) == placements,
        "report placements disagree with the frozen protocol",
    )
    _require(
        tuple(protocol.get("selection_seeds", ())) == selection_seeds,
        "report selection seeds disagree with the frozen protocol",
    )
    _require(
        tuple(protocol.get("final_seeds", ())) == final_seeds,
        "report final seeds disagree with the frozen protocol",
    )
    _require(
        int(protocol.get("calibration_seed", -1)) == int(calibration_seed),
        "report calibration seed disagrees with the frozen protocol",
    )
    _require(held_out_id == HELD_OUT_SCHEDULE_ID, "held-out schedule id is not 'held_out'")
    _require(
        not any(
            str(entry.get("schedule", {}).get("kind")) == "held_out"
            for entry in known_schedules
            if isinstance(entry.get("schedule"), Mapping)
        ),
        "known schedules include a held_out kind",
    )
    _require(
        len({str(entry.get("id")) for entry in known_schedules}) == len(known_schedules),
        "known schedule ids are not unique",
    )

    # -- seed disjointness ------------------------------------------------- #
    select_set = set(int(seed) for seed in selection_seeds)
    final_set = set(int(seed) for seed in final_seeds)
    _require(select_set, "selection seed set is empty")
    _require(final_set, "final seed set is empty")
    _require(not (select_set & final_set), "selection and final seeds overlap")
    _require(
        int(calibration_seed) not in select_set | final_set,
        "calibration seed collides with a selection/final seed",
    )

    # -- grid coverage ----------------------------------------------------- #
    expected_conditions = len(alphas) * len(placements)
    _require(
        len(conditions) == expected_conditions,
        f"expected {expected_conditions} conditions, found {len(conditions)}",
    )
    _require(len(set(policies)) == len(policies), "policy panel has duplicates")
    _require(len(policies) >= 1, "policy panel is empty")
    policy_count = len(policies)
    seen_pairs = set()
    for condition in conditions:
        pair = (float(condition.get("alpha")), str(condition.get("placement")))
        _require(pair not in seen_pairs, f"duplicate condition {pair!r}")
        seen_pairs.add(pair)
        _require(
            tuple(condition.get("policy_ids", ())) == policies,
            f"condition {condition.get('condition_id')!r} panel disagrees with scope",
        )
    _require(
        seen_pairs == {(float(a), str(p)) for a in alphas for p in placements},
        "condition grid does not cover every alpha x placement cell",
    )

    # -- per-condition cell counts ---------------------------------------- #
    total_selection = 0
    total_final = 0
    total_aggregates = 0
    for condition in conditions:
        label = str(condition.get("condition_id"))
        schedule_ids = tuple(str(s) for s in condition.get("schedule_ids", ()))
        _require(schedule_ids, f"condition {label!r} declares no schedules")
        _require(
            len(set(schedule_ids)) == len(schedule_ids),
            f"condition {label!r} schedule ids are not unique",
        )
        _require(
            held_out_id in schedule_ids,
            f"condition {label!r} is missing the final-only held_out schedule",
        )
        known_count = len(schedule_ids) - 1
        expected_selection = len(selection_seeds) * known_count * policy_count
        expected_final = len(final_seeds) * (known_count + 1) * policy_count
        expected_aggregates = (known_count + 1) * policy_count
        _require(
            int(condition.get("selection_cells", -1)) == expected_selection,
            f"condition {label!r} selection cell count mismatch",
        )
        _require(
            int(condition.get("final_cells", -1)) == expected_final,
            f"condition {label!r} final cell count mismatch",
        )
        _require(
            int(condition.get("aggregates", -1)) == expected_aggregates,
            f"condition {label!r} aggregate count mismatch",
        )
        total_selection += expected_selection
        total_final += expected_final
        total_aggregates += expected_aggregates

    matrix_selection = int(scope.get("matrix_selection_cells", -1))
    matrix_final = int(scope.get("matrix_final_cells", -1))
    matrix_aggregates = int(scope.get("matrix_aggregates", -1))
    _require(matrix_selection == total_selection, "matrix selection cell total mismatch")
    _require(matrix_final == total_final, "matrix final cell total mismatch")
    _require(matrix_aggregates == total_aggregates, "matrix aggregate total mismatch")

    # -- strict canonical acceptance (real-MNIST only) --------------------- #
    if real:
        _require(alphas == EXPECTED_ALPHAS, "real report alphas are not canonical")
        _require(placements == EXPECTED_PLACEMENTS, "real report placements are not canonical")
        _require(
            selection_seeds == EXPECTED_SELECTION_SEEDS,
            "real report selection seeds are not canonical",
        )
        _require(
            final_seeds == EXPECTED_FINAL_SEEDS,
            "real report final seeds are not canonical",
        )
        _require(
            int(calibration_seed) == EXPECTED_CALIBRATION_SEED,
            "real report calibration seed is not canonical",
        )
        _require(len(conditions) == EXPECTED_CONDITIONS, "real report condition count is not canonical")
        _require(policy_count == EXPECTED_POLICIES, "real report policy count is not eight")
        realized_known = {
            str(schedule_id)
            for condition in conditions
            for schedule_id in condition.get("schedule_ids", ())
            if str(schedule_id) != held_out_id
        }
        _require(
            realized_known == set(KNOWN_SCHEDULE_IDS),
            "real report known schedules are not the canonical four",
        )
        _require(
            matrix_selection == EXPECTED_SELECTION_CELLS,
            f"real matrix selection total must be {EXPECTED_SELECTION_CELLS}",
        )
        _require(
            matrix_final == EXPECTED_FINAL_CELLS,
            f"real matrix final total must be {EXPECTED_FINAL_CELLS}",
        )
        _require(
            matrix_aggregates == EXPECTED_AGGREGATES,
            f"real matrix aggregate total must be {EXPECTED_AGGREGATES}",
        )
        _require(int(scope.get("rounds", -1)) == EXPECTED_ROUNDS, "real rounds are not 6")
        _require(
            int(scope.get("samples_per_node", -1)) == EXPECTED_SAMPLES_PER_NODE,
            "real samples_per_node is not 64",
        )
        _require(
            int(scope.get("evaluation_max_samples", -1)) == EXPECTED_EVAL_SAMPLES,
            "real evaluation_max_samples is not 128",
        )

    return {
        "conditions": len(conditions),
        "policies": list(policies),
        "policy_count": policy_count,
        "alphas": list(alphas),
        "placements": list(placements),
        "selection_seeds": list(selection_seeds),
        "final_seeds": list(final_seeds),
        "calibration_seed": int(calibration_seed),
        "matrix_selection_cells": matrix_selection,
        "matrix_final_cells": matrix_final,
        "matrix_aggregates": matrix_aggregates,
    }


def _placement_grid(
    conditions: Sequence[Mapping[str, object]], *, topology_n: int
) -> dict[tuple[float, str], list[Mapping[str, object]]]:
    """Validate every placement identity and return it keyed by (alpha, placement)."""

    grid: dict[tuple[float, str], list[Mapping[str, object]]] = {}
    for condition in conditions:
        kind = str(condition.get("condition_id"))
        placement = str(condition.get("placement"))
        alpha = float(condition.get("alpha"))
        identities = condition.get("placement_identities")
        _require(
            isinstance(identities, list) and identities,
            f"condition {kind!r} has no placement identities",
        )
        assert isinstance(identities, list)
        validated: list[Mapping[str, object]] = []
        for identity in identities:
            _require(isinstance(identity, Mapping), "placement identity is not an object")
            assert isinstance(identity, Mapping)
            _require(
                str(identity.get("placement")) == placement,
                f"placement identity disagrees with condition {kind!r}",
            )
            node = int(identity.get("node", -1))
            node_count = int(identity.get("node_count", -1))
            degree = int(identity.get("degree", -1))
            rank = int(identity.get("rank", -1))
            _require(node_count == topology_n, "placement node_count is not the frozen n")
            _require(0 <= node < node_count, "placement node is out of range")
            _require(degree >= 0, "placement degree is negative")
            _require(rank == _expected_rank(placement, node_count), "placement rank mismatch")
            for digest_key in ("graph_digest", "content_digest"):
                digest = identity.get(digest_key)
                _require(
                    isinstance(digest, str) and len(digest) == 64,
                    f"placement {digest_key} is not a sha256",
                )
            validated.append(dict(identity))
        grid[(alpha, placement)] = validated
    return grid


def _assert_seed_graph_consistency(
    grid: Mapping[tuple[float, str], Sequence[Mapping[str, object]]],
    *,
    placements: Sequence[str],
) -> None:
    """The same final seed must realize the same graph for every alpha.

    Placement is seed-derived on ``(degree, node_id)``, so it depends only on
    the topology seed and never on the Dirichlet alpha. Across alphas the
    per-seed identity must therefore be byte-identical, and degree rank order
    ``low <= mid <= high`` must hold for each seed.
    """

    by_placement: dict[str, list[Sequence[Mapping[str, object]]]] = {}
    for (alpha, placement), identities in grid.items():
        by_placement.setdefault(placement, []).append(identities)
    for placement, identity_lists in by_placement.items():
        _require(identity_lists, f"no placement identities for {placement!r}")
        reference = identity_lists[0]
        for other in identity_lists[1:]:
            _require(
                _canonical_bytes(reference) == _canonical_bytes(other),
                f"seed-derived graph varies across alphas for placement {placement!r}",
            )

    reference_alpha = sorted({alpha for alpha, _ in grid})[0]
    seed_count = len(grid[(reference_alpha, placements[0])])
    for index in range(seed_count):
        degrees = [
            int(grid[(reference_alpha, placement)][index].get("degree", -1))
            for placement in placements
        ]
        _require(
            degrees == sorted(degrees),
            "degree rank ordering low<=mid<=high does not hold for a seed",
        )


def _assert_replay_links(
    conditions: Sequence[Mapping[str, object]],
    *,
    root: Path,
    final_seeds: Sequence[int],
    policies: Sequence[str],
    known_ids: Sequence[str],
    held_out_id: str,
) -> int:
    allowed_schedules = set(known_ids) | {held_out_id}
    final_set = set(int(seed) for seed in final_seeds)
    policy_set = set(policies)
    total = 0
    for condition in conditions:
        links = condition.get("replay_links")
        _require(
            isinstance(links, list) and links,
            f"condition {condition.get('condition_id')!r} has no replay links",
        )
        assert isinstance(links, list)
        _require(
            len(links) == int(condition.get("final_cells", -1)),
            f"condition {condition.get('condition_id')!r} replay link count mismatch",
        )
        for link in links:
            _require(isinstance(link, Mapping), "replay link is not an object")
            assert isinstance(link, Mapping)
            label = f"condition {condition.get('condition_id')!r} replay link"
            _require(str(link.get("policy_id")) in policy_set, f"{label} policy is unknown")
            _require(str(link.get("schedule_id")) in allowed_schedules, f"{label} schedule is unknown")
            seed = int(link.get("seed", -1))
            _require(seed in final_set, f"{label} seed is not a final seed")
            held_out = bool(link.get("held_out", False))
            _require(
                held_out == (str(link.get("schedule_id")) == held_out_id),
                f"{label} held_out flag disagrees with its schedule",
            )
            for key in ("trace", "costs", "decision_audit"):
                relative = link.get(key)
                resolved = _confined_realpath(root, relative, label=f"{label} {key}")
                _require(resolved.exists(), f"{label} {key} does not resolve")
            total += 1
    return total


def _assert_evidence(
    report: Mapping[str, object],
    *,
    real: bool,
    cache_digest: str | None,
) -> dict:
    evidence = report.get("evidence")
    _require(isinstance(evidence, Mapping), "canonical report evidence is missing")
    assert isinstance(evidence, Mapping)
    for block in ("derivation", "solution_quality", "defense_usefulness"):
        _require(isinstance(evidence.get(block), Mapping), f"evidence block {block!r} is missing")
    derivation = evidence["derivation"]
    assert isinstance(derivation, Mapping)
    provenance = str(report.get("data_provenance"))
    _require(str(derivation.get("provenance")) == provenance, "derivation provenance mismatch")
    _require(
        str(evidence["solution_quality"].get("provenance")) == provenance,  # type: ignore[index]
        "solution_quality provenance mismatch",
    )
    _require(
        str(evidence["defense_usefulness"].get("provenance")) == provenance,  # type: ignore[index]
        "defense_usefulness provenance mismatch",
    )

    data_identity = derivation.get("data_identity")
    _require(isinstance(data_identity, Mapping), "derivation data_identity is missing")
    assert isinstance(data_identity, Mapping)
    if real:
        _require(str(data_identity.get("dataset")) == "mnist", "data identity is not MNIST")
        _require(
            str(data_identity.get("partition")) == "dirichlet",
            "data identity partition is not dirichlet",
        )
        topology = data_identity.get("topology")
        _require(isinstance(topology, Mapping), "data identity topology is missing")
        assert isinstance(topology, Mapping)
        _require(str(topology.get("kind")) == "watts_strogatz", "topology is not watts_strogatz")
        _require(int(topology.get("n", -1)) == EXPECTED_TOPOLOGY_N, "topology n mismatch")
        _require(int(topology.get("k", -1)) == EXPECTED_TOPOLOGY_K, "topology k mismatch")
        _require(
            abs(float(topology.get("p", -1.0)) - EXPECTED_TOPOLOGY_P) < 1e-12,
            "topology p mismatch",
        )
        cache = data_identity.get("cache")
        _require(isinstance(cache, Mapping), "data identity cache is missing")
        assert isinstance(cache, Mapping)
        pinned = str(cache.get("manifest_digest"))
        _require(
            isinstance(cache_digest, str) and cache_digest == pinned,
            "deep MNIST cache digest disagrees with the report",
        )

    metrics = report.get("metrics")
    _require(isinstance(metrics, Mapping), "canonical report metrics are missing")
    assert isinstance(metrics, Mapping)
    aggregates = metrics.get("aggregates")
    _require(isinstance(aggregates, list) and aggregates, "report has no final aggregates")
    deltas = metrics.get("paired_deltas")
    _require(isinstance(deltas, list) and deltas, "report has no paired deltas")

    costs = report.get("costs")
    _require(isinstance(costs, Mapping), "canonical report costs are missing")
    assert isinstance(costs, Mapping)
    communication = costs.get("communication")
    _require(isinstance(communication, Mapping), "communication cost block is missing")
    assert isinstance(communication, Mapping)
    totals = communication.get("totals")
    _require(isinstance(totals, Mapping), "communication totals are missing")
    assert isinstance(totals, Mapping)
    _require(
        int(totals.get("directed_messages", -1)) > 0,
        "communication totals report no directed messages",
    )
    _require(
        int(totals.get("logical_payload_bytes", -1)) > 0,
        "communication totals report no logical payload bytes",
    )
    if real:
        headline_costs = costs.get("headline")
        _require(isinstance(headline_costs, Mapping), "headline cost block is missing")
        assert isinstance(headline_costs, Mapping)
        _require(
            headline_costs.get("available") is False,
            "real report must not fabricate unavailable headline costs",
        )

    figures = report.get("figures")
    _require(isinstance(figures, Mapping), "canonical report figures block is missing")
    assert isinstance(figures, Mapping)
    entries = figures.get("entries")
    _require(isinstance(entries, list) and entries, "figure block has no entries")
    assert isinstance(entries, list)
    return {"figures": len(entries), "aggregates": len(aggregates), "paired_deltas": len(deltas)}


# --------------------------------------------------------------------------- #
# Main audit
# --------------------------------------------------------------------------- #
def audit_canonical_study(
    root: str | os.PathLike[str],
    *,
    deps: CanonicalAuditDeps | None = None,
    output: str | os.PathLike[str] | None = None,
    deep_cache: bool = True,
) -> CanonicalAuditResult:
    """Run the read-only production acceptance audit for a canonical study root."""

    deps = deps or CanonicalAuditDeps()
    try:
        root_path = Path(os.path.abspath(os.fspath(root)))
    except TypeError as exc:
        raise CanonicalAuditError(f"invalid study root: {_bounded(exc)}") from exc
    _require(root_path.is_dir(), f"canonical study root not found: {root_path}")
    repo_root = Path(deps.repo_root) if deps.repo_root is not None else _default_repo_root()

    try:
        report_result = verify_canonical_report(root_path, deps=deps.report_deps)
    except (CanonicalReportError, RobustnessVerifyError) as exc:
        raise CanonicalAuditError(
            f"independent report audit failed closed: {_bounded(exc)}"
        ) from exc
    report = report_result.report
    _require(isinstance(report, Mapping), "independent report audit returned no report")
    assert isinstance(report, Mapping)

    frozen = _read_json(root_path / ROBUSTNESS_PROTOCOL_FILENAME)
    _require(
        frozen.get("schema") == "dfl_sim.robustness_protocol/v1",
        "robustness-protocol.json has an unexpected schema",
    )
    protocol = frozen.get("protocol")
    _require(isinstance(protocol, Mapping), "robustness-protocol.json is missing the protocol")
    assert isinstance(protocol, Mapping)

    provenance = str(report.get("data_provenance"))
    _require(
        provenance == "real_mnist" or deps.allow_injected is True,
        "the public canonical audit accepts real_mnist provenance only",
    )
    real = provenance == "real_mnist"

    conditions = report.get("conditions")
    _require(isinstance(conditions, list) and conditions, "canonical report has no conditions")
    assert isinstance(conditions, list)

    coverage = _assert_scope(report, protocol, real=real)
    topology_n = int(protocol.get("topology_n", EXPECTED_TOPOLOGY_N))
    grid = _placement_grid(conditions, topology_n=topology_n)
    _assert_seed_graph_consistency(grid, placements=coverage["placements"])

    known_schedules = protocol.get("known_schedules")
    assert isinstance(known_schedules, list)
    known_ids = tuple(str(entry.get("id")) for entry in known_schedules)
    held_out = protocol.get("held_out_schedule")
    assert isinstance(held_out, Mapping)
    held_out_id = str(held_out.get("id"))
    replayed = _assert_replay_links(
        conditions,
        root=root_path,
        final_seeds=coverage["final_seeds"],
        policies=coverage["policies"],
        known_ids=known_ids,
        held_out_id=held_out_id,
    )

    cache_digest: str | None = None
    deep_verified = False
    if real and deep_cache:
        data_root = _resolve_path(str(protocol.get("data_root", "data")), repo_root)
        cache_digest = _deep_cache_digest(data_root, deps.cache_verify)
        sources = report.get("sources")
        _require(isinstance(sources, Mapping), "canonical report sources are missing")
        assert isinstance(sources, Mapping)
        cache_block = sources.get("cache")
        _require(isinstance(cache_block, Mapping), "report sources cache block is missing")
        assert isinstance(cache_block, Mapping)
        _require(
            cache_digest == str(cache_block.get("manifest_digest")),
            "deep MNIST cache digest disagrees with report sources",
        )
        deep_verified = True

    detail = _assert_evidence(report, real=real, cache_digest=cache_digest)

    audit: dict = {
        "schema": AUDIT_SCHEMA,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed",
        "study_id": str(report.get("study_id")),
        "data_provenance": provenance,
        "report_digest": str(report_result.report_digest),
        "protocol_digest": str(frozen.get("protocol_digest")),
        "coverage": coverage,
        "checks": {
            "conditions": coverage["conditions"],
            "policies": coverage["policy_count"],
            "replay_links": replayed,
            "figures": detail["figures"],
            "aggregates": detail["aggregates"],
            "paired_deltas": detail["paired_deltas"],
            "deep_cache_verified": deep_verified,
        },
        "figures": [
            str(entry.get("filename")) for entry in report["figures"]["entries"]  # type: ignore[index]
        ],
    }
    audit["audit_digest"] = _sha256_json(audit)

    output_path: Path | None = None
    if output is not None:
        output_path = Path(os.path.abspath(os.fspath(output)))
        _publish_audit(output_path, audit)
    return CanonicalAuditResult(
        root=root_path,
        status="passed",
        audit=audit,
        audit_digest=str(audit["audit_digest"]),
        output_path=output_path,
        conditions=coverage["conditions"],
        policies=coverage["policy_count"],
    )


def _publish_audit(destination: Path, audit: Mapping[str, object]) -> None:
    payload = (json.dumps(audit, sort_keys=True, indent=2) + "\n").encode("utf-8")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CanonicalAuditError(
            f"could not create audit output parent: {_bounded(exc)}"
        ) from exc
    try:
        fd, temporary = tempfile.mkstemp(
            dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    except OSError as exc:
        raise CanonicalAuditError(f"could not publish audit result: {_bounded(exc)}") from exc


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _audit_deps() -> CanonicalAuditDeps | None:
    """Production canonical-audit injection seams (none).

    Tests monkeypatch this provider; the public CLI never reaches an injected
    seam.
    """

    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dfl_sim.canonical_audit",
        description=(
            "Independent production acceptance audit of a completed canonical "
            "robustness study: re-audits the raw matrix and headline through "
            "the report verifier, deep-verifies the downloaded MNIST cache, "
            "asserts exact declared coverage and atomically writes an optional "
            "digest-bound runtime audit."
        ),
    )
    parser.add_argument("root", help="A completed canonical robustness study root.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the digest-bound audit-result JSON.",
    )
    parser.add_argument(
        "--shallow-cache",
        action="store_true",
        help="Skip the deep MNIST content-hash re-verification (not canonical).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = audit_canonical_study(
            args.root,
            deps=_audit_deps(),
            output=args.output,
            deep_cache=not args.shallow_cache,
        )
    except CanonicalAuditError as exc:
        print(f"dfl_sim: error: {_bounded(exc)}", file=sys.stderr)
        return 1
    print(
        f"{AUDIT_SUCCESS_MARKER}"
        f"root={result.root} "
        f"conditions={result.conditions} "
        f"policies={result.policies} "
        f"report_digest={result.audit.get('report_digest')} "
        f"audit_digest={result.audit_digest}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
