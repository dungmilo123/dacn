"""Independent audit of a completed raw robustness matrix (M003/S05/T04).

This module reconstructs a published matrix **from disk** and re-derives every
identity it cites. It never trusts a stored provenance string, digest or
success flag: the source protocol, the separate IID headline identity, the
content-digested MNIST cache, each condition's frozen study, and every cell's
trace, cost sidecar and decision audit are re-read and re-hashed. A missing,
mutated, foreign or duplicate source raises :class:`RobustnessVerifyError` and
no evidence marker is written.

The verifier is deliberately independent of the orchestrator: it rebuilds the
expected study for each condition from the frozen protocol plus the *loaded*
calibration artifacts, recomputes the selection/final shared-input digests,
placement identities and per-cell config digests, and compares them to the
published manifests. It relies only on public production surfaces plus a small
set of in-package reconstruction helpers from :mod:`dfl_sim.evaluation`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .costs import (
    CostError,
    file_digest,
    load_cost_payload,
    load_decision_audit_records,
    trace_artifact_digest,
)
from .evaluation import (
    FINAL_MANIFEST_FILENAME,
    SELECTION_MANIFEST_FILENAME,
    _placement_study_payload,
    _redacted_cell_digest,
    _resolve_study_placements,
    build_cell_config,
    final_output_digest,
    read_final_round_metrics,
    selection_output_digest,
    validate_shared_inputs,
)
from .evaluation_config import (
    HeldOutStudyConfig,
    held_out_study_digest,
    parse_held_out_study_text,
    selection_projection,
    selection_spec_digest,
    study_digest,
)
from .robustness_study import (
    CALIBRATION_DIRNAME,
    CONDITIONS_DIRNAME,
    MATRIX_MANIFEST_FILENAME,
    ROBUSTNESS_PROTOCOL_FILENAME,
    STUDY_DIRNAME,
    InProcessStudyRunner,
    RobustnessProtocol,
    RobustnessStudyError,
    StudyRunner,
    alpha_token,
    build_condition_study,
    canonical_protocol,
    protocol_digest,
)
from .study_placement import placement_contract_digest, resolve_placement
from .trajectory import SCHEMA_VERSION, TraceReader, TrajectoryError

__all__ = [
    "MATRIX_EVIDENCE_FILENAME",
    "ROBUSTNESS_EVIDENCE_SCHEMA",
    "RobustnessVerifyDeps",
    "RobustnessVerifyError",
    "RobustnessVerifyResult",
    "build_parser",
    "main",
    "verify_robustness_study",
]

#: Versioned evidence record published next to the completed matrix manifest.
ROBUSTNESS_EVIDENCE_SCHEMA = "dfl_robustness_matrix_evidence/v1"
MATRIX_EVIDENCE_FILENAME = "matrix-evidence.json"

_MAX_ERROR_TEXT = 400
_CALIBRATION_VARIANTS = ("full", "no_belief", "no_strategic")


class RobustnessVerifyError(RuntimeError):
    """Raised when a published robustness matrix fails its independent audit."""


def _bounded(text: object, *, limit: int = _MAX_ERROR_TEXT) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) > limit:
        return collapsed[: limit - 1] + "\u2026"
    return collapsed


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise RobustnessVerifyError(detail)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise RobustnessVerifyError(f"missing matrix artifact: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobustnessVerifyError(
            f"could not read {path}: {_bounded(exc)}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise RobustnessVerifyError(f"{path} is not a JSON object")
    return raw


# --------------------------------------------------------------------------- #
# Injected dependencies
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RobustnessVerifyDeps:
    """Optional injection seams; production uses the real public surfaces."""

    cache_verify: Callable[[str, bool], object] | None = None
    headline_verify: Callable[..., object] | None = None
    load_calibration_config: Callable[..., object] | None = None
    load_artifact: Callable[..., object] | None = None
    trace_reader: Callable[[object], Any] | None = None
    study_builder: (
        Callable[
            [object, RobustnessProtocol, float, str], HeldOutStudyConfig
        ]
        | None
    ) = None
    repo_root: str | os.PathLike[str] | None = None


@dataclass(frozen=True)
class RobustnessVerifyResult:
    output_path: Path
    evidence_path: Path | None
    status: str
    evidence: Mapping[str, object]
    protocol_digest: str
    conditions: int
    selection_cells: int
    final_cells: int
    aggregate_count: int


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(text: str, base: Path) -> Path:
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate


def _confined(root: Path, relative: object, *, label: str) -> Path:
    """Resolve ``relative`` under ``root`` and reject traversal/symlink escape."""

    if not isinstance(relative, str) or not relative:
        raise RobustnessVerifyError(f"{label} path must be a non-empty string")
    if os.path.isabs(relative):
        raise RobustnessVerifyError(f"{label} path must be relative, got {relative!r}")
    try:
        real_root = os.path.realpath(root)
        real = os.path.realpath(os.path.join(real_root, relative))
    except OSError as exc:  # pragma: no cover - defensive
        raise RobustnessVerifyError(f"could not resolve {label} path: {exc}") from exc
    if real != real_root and not real.startswith(real_root + os.sep):
        raise RobustnessVerifyError(f"{label} path escapes the study root")
    return Path(real)


def _shared_input_digest(study: HeldOutStudyConfig, seeds: Sequence[int]) -> str:
    """Recompute the phase shared-input digest exactly as the runner does."""

    placements = _resolve_study_placements(study.study, seeds)
    payload = _placement_study_payload(study.study, placements)
    base = study.study.base
    configs = [
        build_cell_config(base, base.aggregation, int(seed))
        for seed in sorted(int(seed) for seed in seeds)
    ]
    return validate_shared_inputs(configs, placement=payload)


def _placement_lookup(study: HeldOutStudyConfig, seed: int):
    contract = getattr(study.study, "placement", None)
    if contract is None:
        return None
    probe = build_cell_config(study.study.base, study.study.base.aggregation, int(seed))
    return resolve_placement(
        probe,
        contract.placement,
        require_varied=contract.require_varied_degrees,
    )


def _expected_cells(
    study: HeldOutStudyConfig, *, phase: str
) -> set[tuple[int, str, str]]:
    panel = study.study.policy_ids()
    if phase == "selection":
        seeds = tuple(study.study.seeds)
        schedules = tuple(entry.id for entry in study.known_schedules)
    else:
        seeds = tuple(study.final_seeds)
        schedules = tuple(entry.id for entry in study.known_schedules) + (
            study.held_out_schedule.id,
        )
    return {
        (int(seed), str(schedule_id), str(policy_id))
        for seed in seeds
        for schedule_id in schedules
        for policy_id in panel
    }


# --------------------------------------------------------------------------- #
# Manifest entry audit
# --------------------------------------------------------------------------- #
def _verify_entry(
    entry: Mapping[str, object],
    *,
    phase: str,
    phase_root: Path,
    study: HeldOutStudyConfig,
    trace_reader: Callable[[object], Any],
) -> dict:
    policy_id = str(entry.get("policy_id"))
    seed = int(entry.get("seed"))
    schedule_id = str(entry.get("schedule_id"))
    label = f"{phase} policy {policy_id!r} seed {seed} schedule {schedule_id!r}"

    panel = {policy.id: policy for policy in study.study.ordered_panel()}
    _require(policy_id in panel, f"{label} is not in the study panel")
    schedule = _schedule_for(study, schedule_id, phase=phase)
    _require(schedule is not None, f"{label} references an unknown schedule")

    # --- identifier / digest binding -------------------------------------- #
    expected_pairing = f"{policy_id}:seed-{seed}:{schedule_id}"
    _require(entry.get("pairing_key") == expected_pairing, f"{label} pairing key mismatch")
    config = build_cell_config(
        study.study.base,
        panel[policy_id].aggregation,
        seed,
        placement=(
            study.study.placement.placement
            if getattr(study.study, "placement", None) is not None
            else None
        ),
        require_varied_degrees=(
            study.study.placement.require_varied_degrees
            if getattr(study.study, "placement", None) is not None
            else False
        ),
    )
    _require(
        entry.get("config_digest") == _redacted_cell_digest(config),
        f"{label} config digest mismatch",
    )
    _require(
        entry.get("schedule_digest") == _schedule_digest(schedule.schedule),
        f"{label} schedule digest mismatch",
    )
    _require(
        entry.get("schedule") == schedule.schedule.model_dump(mode="json"),
        f"{label} schedule definition mismatch",
    )

    # --- placement identity ------------------------------------------------ #
    resolution = _placement_lookup(study, seed)
    if resolution is not None:
        _require(
            entry.get("placement") == resolution.identity_payload(),
            f"{label} placement identity mismatch",
        )
    else:
        _require("placement" not in entry, f"{label} unexpected placement block")

    # --- artifact identity ------------------------------------------------- #
    artifact_dir = getattr(panel[policy_id].aggregation, "artifact_dir", None)
    if artifact_dir is not None:
        _require(
            isinstance(entry.get("artifact_digest"), str)
            and len(str(entry.get("artifact_digest"))) == 64,
            f"{label} artifact digest missing",
        )
    else:
        _require(entry.get("artifact_digest") is None, f"{label} unexpected artifact digest")

    # --- trace ------------------------------------------------------------- #
    trace_path = _confined(phase_root, entry.get("trace_path"), label=f"{label} trace")
    _require(trace_path.is_dir(), f"{label} trace directory is missing")
    try:
        with trace_reader(trace_path) as reader:
            schema = getattr(reader, "schema_version", SCHEMA_VERSION)
            _require(
                schema == SCHEMA_VERSION,
                f"{label} trace schema {schema!r} is not {SCHEMA_VERSION}",
            )
            manifest = getattr(reader, "manifest", None)
            _require(
                isinstance(manifest, Mapping),
                f"{label} trace reader exposes no manifest",
            )
            trace_digest = trace_artifact_digest(manifest)
    except RobustnessVerifyError:
        raise
    except (TrajectoryError, CostError, OSError) as exc:
        raise RobustnessVerifyError(f"{label} produced an invalid trace: {_bounded(exc)}") from exc

    telemetry = entry.get("telemetry")
    _require(isinstance(telemetry, Mapping), f"{label} is missing a telemetry block")
    _require(
        telemetry.get("trace_digest") == trace_digest,
        f"{label} telemetry trace digest mismatch",
    )
    _require(
        telemetry.get("measurement_clock") == "perf_counter_ns",
        f"{label} telemetry measurement clock mismatch",
    )
    costs = _verify_sidecar(
        phase_root,
        telemetry,
        label=label,
        identity={
            "phase": phase,
            "study_id": study.study.study_id,
            "policy_id": policy_id,
            "seed": seed,
            "schedule_id": schedule_id,
        },
        config_digest=entry.get("config_digest"),
        artifact_digest=entry.get("artifact_digest"),
        trace_digest=trace_digest,
    )

    # --- metrics ----------------------------------------------------------- #
    metrics = read_final_round_metrics(trace_path, trace_reader=trace_reader)
    if phase == "final":
        _require(
            entry.get("metrics") == metrics,
            f"{label} embedded metrics disagree with the validated trace",
        )
    return {
        "policy_id": policy_id,
        "seed": seed,
        "schedule_id": schedule_id,
        "held_out": bool(entry.get("held_out", False)),
        "reused": bool(entry.get("reused", False)),
        "rounds": costs["rounds"],
        "messages": costs["messages"],
        "payload_bytes": costs["payload_bytes"],
    }


def _schedule_for(study: HeldOutStudyConfig, schedule_id: str, *, phase: str):
    for entry in study.known_schedules:
        if entry.id == schedule_id:
            return entry
    if phase == "final" and schedule_id == study.held_out_schedule.id:
        return study.held_out_schedule
    return None


def _schedule_digest(schedule) -> str:
    from .attack_schedule import schedule_digest

    return schedule_digest(schedule)


def _verify_sidecar(
    phase_root: Path,
    telemetry: Mapping[str, object],
    *,
    label: str,
    identity: Mapping[str, object],
    config_digest: object,
    artifact_digest: object,
    trace_digest: str,
) -> dict:
    cost_path = _confined(phase_root, telemetry.get("cost_path"), label=f"{label} cost")
    audit_path = _confined(
        phase_root, telemetry.get("decision_audit_path"), label=f"{label} decision audit"
    )
    _require(cost_path.is_file(), f"{label} cost sidecar is missing")
    _require(audit_path.is_file(), f"{label} decision audit sidecar is missing")
    _require(
        telemetry.get("cost_digest") == file_digest(cost_path),
        f"{label} cost sidecar digest mismatch",
    )
    _require(
        telemetry.get("decision_audit_digest") == file_digest(audit_path),
        f"{label} decision audit digest mismatch",
    )
    try:
        cost = load_cost_payload(cost_path)
        decisions = load_decision_audit_records(audit_path)
    except CostError as exc:
        raise RobustnessVerifyError(f"{label} sidecar failed validation: {_bounded(exc)}") from exc
    canonical_identity = {key: value for key, value in identity.items() if value is not None}
    _require(cost.get("identity") == canonical_identity, f"{label} cost identity mismatch")
    measurement = cost.get("measurement")
    _require(isinstance(measurement, Mapping), f"{label} cost measurement block missing")
    _require(
        measurement.get("config_digest") == config_digest,
        f"{label} cost config digest mismatch",
    )
    _require(
        measurement.get("trace_digest") == trace_digest,
        f"{label} cost trace digest mismatch",
    )
    _require(
        measurement.get("artifact_digests", {}).get("aggregation_artifact")
        == artifact_digest,
        f"{label} cost artifact digest mismatch",
    )
    totals = cost.get("totals")
    _require(isinstance(totals, Mapping), f"{label} cost totals block missing")
    for record in decisions:
        _require(
            record.get("identity") == canonical_identity,
            f"{label} decision audit identity mismatch",
        )
    return {
        "rounds": int(totals.get("rounds", 0)),
        "messages": int(totals.get("directed_messages", 0)),
        "payload_bytes": int(totals.get("logical_payload_bytes", 0)),
    }


# --------------------------------------------------------------------------- #
# Manifest audit
# --------------------------------------------------------------------------- #
def _verify_manifest(
    manifest: Mapping[str, object],
    *,
    phase: str,
    phase_root: Path,
    study: HeldOutStudyConfig,
    expected_shared: str,
    trace_reader: Callable[[object], Any],
) -> tuple[str, dict]:
    if phase == "selection":
        _require(
            manifest.get("schema") == "dfl_held_out_selection",
            "selection manifest schema mismatch",
        )
        _require(manifest.get("status") == "frozen", "selection manifest is not frozen")
        _require(
            manifest.get("shared_input_digest") == expected_shared,
            "selection shared-input digest mismatch",
        )
        _require(
            manifest.get("study_digest") == study_digest(study.study),
            "selection study digest mismatch",
        )
        _require(
            manifest.get("selection_spec_digest")
            == selection_spec_digest(selection_projection(study)),
            "selection spec digest mismatch",
        )
        computed_output_digest = selection_output_digest(manifest)
    else:
        _require(
            manifest.get("schema") == "dfl_held_out_study_manifest",
            "final manifest schema mismatch",
        )
        _require(manifest.get("status") == "complete", "final manifest is not complete")
        _require(
            manifest.get("shared_input_digest") == expected_shared,
            "final shared-input digest mismatch",
        )
        _require(
            manifest.get("study_digest") == study_digest(study.study),
            "final study digest mismatch",
        )
        _require(
            manifest.get("selection_spec_digest")
            == selection_spec_digest(selection_projection(study)),
            "final selection spec digest mismatch",
        )
        _require(
            manifest.get("held_out_study_digest") == held_out_study_digest(study),
            "final held-out study digest mismatch",
        )
        computed_output_digest = final_output_digest(manifest)

    _require(
        manifest.get("panel") is not None
        and sorted(str(x) for x in manifest.get("panel", ()))
        == sorted(study.study.policy_ids()),
        f"{phase} manifest panel mismatch",
    )

    entries = manifest.get("entries")
    _require(isinstance(entries, list), f"{phase} manifest entries are missing")
    expected = _expected_cells(study, phase=phase)
    seen: set[tuple[int, str, str]] = set()
    trace_count = 0
    telemetry_cells = 0
    messages = 0
    payload_bytes = 0
    for entry in entries:
        _require(isinstance(entry, Mapping), f"{phase} manifest entry is not an object")
        key = (
            int(entry.get("seed")),
            str(entry.get("schedule_id")),
            str(entry.get("policy_id")),
        )
        _require(key not in seen, f"{phase} manifest repeats cell {key!r}")
        _require(key in expected, f"{phase} manifest has a foreign cell {key!r}")
        seen.add(key)
        info = _verify_entry(
            entry,
            phase=phase,
            phase_root=phase_root,
            study=study,
            trace_reader=trace_reader,
        )
        trace_count += 1
        telemetry_cells += 1
        messages += info["messages"]
        payload_bytes += info["payload_bytes"]
    missing = expected - seen
    _require(not missing, f"{phase} manifest is missing {len(missing)} cell(s)")
    return computed_output_digest, {
        "cells": trace_count,
        "telemetry_cells": telemetry_cells,
        "messages": messages,
        "payload_bytes": payload_bytes,
    }


def _verify_condition(
    *,
    condition,
    root: Path,
    protocol: RobustnessProtocol,
    canonical,
    deps: RobustnessVerifyDeps,
    include_ablations: bool,
    trace_reader: Callable[[object], Any],
) -> dict:
    condition_id = condition.condition_id
    condition_root = root / CONDITIONS_DIRNAME / condition_id
    _require(condition_root.is_dir(), f"condition {condition_id!r} directory is missing")

    # --- reconstruct the expected study independently --------------------- #
    artifacts = None
    if include_ablations:
        artifacts = _load_condition_artifacts(root, condition.alpha, deps)
    builder = deps.study_builder
    if builder is not None:
        expected = builder(canonical, protocol, condition.alpha, condition.placement)
    else:
        expected = build_condition_study(
            canonical,
            protocol,
            condition.alpha,
            condition.placement,
            artifact_root=root,
            artifacts=artifacts,
        )

    frozen_path = condition_root / "frozen-study.yaml"
    _require(frozen_path.is_file(), f"condition {condition_id!r} frozen study is missing")
    try:
        frozen = parse_held_out_study_text(
            frozen_path.read_text(encoding="utf-8"), source=str(frozen_path)
        )
    except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
        raise RobustnessVerifyError(
            f"condition {condition_id!r} frozen study is invalid: {_bounded(exc)}"
        ) from exc
    _require(
        held_out_study_digest(frozen) == held_out_study_digest(expected),
        f"condition {condition_id!r} frozen study disagrees with the reconstructed study",
    )
    frozen_study_digest = _sha256_bytes(frozen_path.read_bytes())

    study_dir = condition_root / STUDY_DIRNAME
    selection_manifest = _read_json(study_dir / "selection" / SELECTION_MANIFEST_FILENAME)
    final_manifest = _read_json(study_dir / "final" / FINAL_MANIFEST_FILENAME)

    expected_selection_shared = _shared_input_digest(expected, expected.study.seeds)
    expected_final_shared = _shared_input_digest(expected, expected.final_seeds)

    selection_output_digest_value, selection_stats = _verify_manifest(
        selection_manifest,
        phase="selection",
        phase_root=study_dir / "selection",
        study=expected,
        expected_shared=expected_selection_shared,
        trace_reader=trace_reader,
    )
    _require(
        selection_manifest.get("selection_output_digest") == selection_output_digest_value,
        f"condition {condition_id!r} selection output digest mismatch",
    )
    final_output_digest_value, final_stats = _verify_manifest(
        final_manifest,
        phase="final",
        phase_root=study_dir / "final",
        study=expected,
        expected_shared=expected_final_shared,
        trace_reader=trace_reader,
    )
    _require(
        final_manifest.get("final_output_digest") == final_output_digest_value,
        f"condition {condition_id!r} final output digest mismatch",
    )
    _require(
        final_manifest.get("selection_output_digest") == selection_output_digest_value,
        f"condition {condition_id!r} final manifest cites a stale selection digest",
    )

    aggregate_count = len(expected.known_schedules) + 1
    return {
        "condition_id": condition_id,
        "alpha": float(condition.alpha),
        "placement": condition.placement,
        "frozen_study_digest": frozen_study_digest,
        "study_digest": study_digest(expected.study),
        "selection_spec_digest": selection_spec_digest(selection_projection(expected)),
        "held_out_study_digest": held_out_study_digest(expected),
        "selection_shared_input_digest": expected_selection_shared,
        "final_shared_input_digest": expected_final_shared,
        "selection_cells": selection_stats["cells"],
        "final_cells": final_stats["cells"],
        "aggregates": int(aggregate_count * len(expected.study.policy_ids())),
        "selection_output_digest": selection_output_digest_value,
        "final_output_digest": final_output_digest_value,
        "messages": selection_stats["messages"] + final_stats["messages"],
        "payload_bytes": selection_stats["payload_bytes"] + final_stats["payload_bytes"],
    }


def _load_condition_artifacts(
    root: Path, alpha: float, deps: RobustnessVerifyDeps
) -> dict[str, object]:
    if deps.load_artifact is None:
        from .bayesian_artifact import load_bayesian_artifact

        loader = load_bayesian_artifact
    else:
        loader = deps.load_artifact
    token = alpha_token(alpha)
    artifacts: dict[str, object] = {}
    for variant in _CALIBRATION_VARIANTS:
        path = root / CALIBRATION_DIRNAME / token / variant
        _require(path.is_dir(), f"calibration artifact {token}/{variant} is missing")
        try:
            artifacts[variant] = loader(str(path))
        except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
            raise RobustnessVerifyError(
                f"could not load calibration artifact {token}/{variant}: {_bounded(exc)}"
            ) from exc
    return artifacts


# --------------------------------------------------------------------------- #
# Headline / cache
# --------------------------------------------------------------------------- #
def _verify_headline_pin(
    *,
    frozen: Mapping[str, object],
    root: Path,
    protocol: RobustnessProtocol,
    deps: RobustnessVerifyDeps,
    repo_root: Path,
    write: bool,
) -> dict:
    if protocol.data_provenance != "real_mnist":
        return {"data_provenance": "injected", "verified": False}
    pinned = frozen.get("headline")
    _require(isinstance(pinned, Mapping), "frozen protocol is missing the headline pin")
    source_root = _resolve_path(protocol.headline_source_root, repo_root)
    _require(source_root.is_dir(), f"headline source root not found: {source_root!r}")
    verify = deps.headline_verify
    if verify is None:
        from . import headline_verify

        verify = headline_verify.verify_headline
    try:
        result = verify(str(source_root), write=False, deep_cache=False)
    except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
        raise RobustnessVerifyError(
            f"headline re-verification failed closed: {_bounded(exc)}"
        ) from exc
    evidence = getattr(result, "evidence", None)
    evidence_digest = (
        evidence.get("evidence_digest") if isinstance(evidence, Mapping) else None
    )
    _require(
        pinned.get("protocol_digest") == getattr(result, "protocol_digest", None),
        "headline protocol digest no longer matches the pinned identity",
    )
    _require(
        pinned.get("evidence_digest") == evidence_digest,
        "headline evidence digest no longer matches the pinned identity",
    )
    _require(
        int(pinned.get("selection_cells", -1)) == int(getattr(result, "selection_cells", -1)),
        "headline selection cell count mismatch",
    )
    _require(
        int(pinned.get("final_cells", -1)) == int(getattr(result, "final_cells", -1)),
        "headline final cell count mismatch",
    )
    _require(pinned.get("scope") == "iid-chain3", "headline scope is not IID")
    return dict(pinned)


def _verify_cache(
    *,
    frozen: Mapping[str, object],
    protocol: RobustnessProtocol,
    deps: RobustnessVerifyDeps,
    repo_root: Path,
    deep: bool,
) -> dict:
    if protocol.data_provenance != "real_mnist":
        return {"data_provenance": "injected"}
    pinned = frozen.get("cache")
    _require(isinstance(pinned, Mapping), "frozen protocol is missing the cache pin")
    data_root = _resolve_path(protocol.data_root, repo_root)
    verify = deps.cache_verify
    if verify is None:
        from . import mnist_provision

        verify = mnist_provision.verify_cache
    try:
        manifest = verify(str(data_root), deep)
    except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
        raise RobustnessVerifyError(
            f"MNIST cache verification failed closed: {_bounded(exc)}"
        ) from exc
    digest = getattr(manifest, "manifest_digest", None)
    _require(
        digest == pinned.get("manifest_digest"),
        "MNIST cache manifest digest no longer matches the pinned identity",
    )
    return {"data_root": pinned.get("data_root"), "manifest_digest": digest}


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def verify_robustness_study(
    output: str | os.PathLike[str],
    *,
    deps: RobustnessVerifyDeps | None = None,
    write: bool = True,
    deep_cache: bool = False,
) -> RobustnessVerifyResult:
    """Independently verify a completed robustness matrix and publish evidence.

    Every cited source is re-read and re-derived; a missing/mutated/foreign
    source raises :class:`RobustnessVerifyError` and no evidence marker is
    written.
    """

    deps = deps or RobustnessVerifyDeps()
    try:
        root = Path(os.path.abspath(os.fspath(output)))
    except TypeError as exc:
        raise RobustnessVerifyError(f"invalid output path: {_bounded(exc)}") from exc
    _require(root.is_dir(), f"robustness matrix root not found: {root}")

    repo_root = Path(deps.repo_root) if deps.repo_root is not None else _default_repo_root()

    frozen = _read_json(root / ROBUSTNESS_PROTOCOL_FILENAME)
    _require(
        frozen.get("schema") == "dfl_sim.robustness_protocol/v1",
        "robustness-protocol.json has an unexpected schema",
    )
    payload = frozen.get("protocol")
    _require(isinstance(payload, Mapping), "robustness-protocol.json is missing the protocol")
    try:
        protocol = RobustnessProtocol.model_validate(payload)
    except ValidationError as exc:
        raise RobustnessVerifyError(f"frozen protocol is invalid: {_bounded(exc)}") from exc
    try:
        canonical = canonical_protocol(protocol, artifact_root=str(root))
    except RobustnessStudyError as exc:
        raise RobustnessVerifyError(f"frozen protocol is invalid: {_bounded(exc)}") from exc
    _require(
        frozen.get("protocol_digest") == protocol_digest(protocol),
        "frozen protocol digest does not match the resolved protocol",
    )

    include_ablations = protocol.data_provenance == "real_mnist"
    trace_reader = deps.trace_reader or TraceReader

    matrix_manifest = _read_json(root / MATRIX_MANIFEST_FILENAME)
    _require(
        matrix_manifest.get("schema") == "dfl_robustness_matrix",
        "matrix manifest has an unexpected schema",
    )
    _require(matrix_manifest.get("status") == "complete", "matrix manifest is not complete")
    _require(
        matrix_manifest.get("protocol_digest") == frozen.get("protocol_digest"),
        "matrix manifest protocol digest mismatch",
    )
    stored_matrix_digest = matrix_manifest.get("matrix_digest")
    recomputed_matrix = dict(matrix_manifest)
    recomputed_matrix.pop("matrix_digest", None)
    _require(
        stored_matrix_digest == _sha256_json(recomputed_matrix),
        "matrix manifest digest mismatch",
    )

    headline = _verify_headline_pin(
        frozen=frozen,
        root=root,
        protocol=protocol,
        deps=deps,
        repo_root=repo_root,
        write=write,
    )
    cache = _verify_cache(
        frozen=frozen,
        protocol=protocol,
        deps=deps,
        repo_root=repo_root,
        deep=deep_cache,
    )

    conditions = [
        _verify_condition(
            condition=condition,
            root=root,
            protocol=protocol,
            canonical=canonical,
            deps=deps,
            include_ablations=include_ablations,
            trace_reader=trace_reader,
        )
        for condition in canonical.conditions()
    ]

    selection_cells = sum(int(c["selection_cells"]) for c in conditions)
    final_cells = sum(int(c["final_cells"]) for c in conditions)
    aggregate_count = sum(int(c["aggregates"]) for c in conditions)
    totals = matrix_manifest.get("totals")
    _require(isinstance(totals, Mapping), "matrix manifest totals are missing")
    expected_totals = {
        "conditions": len(conditions),
        "selection_cells": selection_cells,
        "final_cells": final_cells,
        "aggregates": aggregate_count,
    }
    for key, value in expected_totals.items():
        _require(
            int(totals.get(key, -1)) == value,
            f"matrix manifest {key} disagrees with the reconstructed matrix",
        )

    published = matrix_manifest.get("conditions")
    _require(isinstance(published, list), "matrix manifest conditions are missing")
    published_by_id = {str(entry.get("condition_id")): entry for entry in published}
    for condition in conditions:
        entry = published_by_id.get(condition["condition_id"])
        _require(entry is not None, f"matrix manifest is missing condition {condition['condition_id']!r}")
        for key in (
            "frozen_study_digest",
            "held_out_study_digest",
            "selection_spec_digest",
            "study_digest",
            "selection_cells",
            "final_cells",
            "aggregates",
            "selection_output_digest",
            "final_output_digest",
        ):
            _require(
                entry.get(key) == condition[key],
                f"matrix manifest condition {condition['condition_id']!r} {key} mismatch",
            )

    evidence: dict = {
        "schema": ROBUSTNESS_EVIDENCE_SCHEMA,
        "status": "verified",
        "study_id": protocol.study_id,
        "data_provenance": protocol.data_provenance,
        "protocol_digest": frozen.get("protocol_digest"),
        "matrix_digest": stored_matrix_digest,
        "headline": dict(headline),
        "cache": dict(cache),
        "conditions": conditions,
        "totals": expected_totals,
    }
    evidence["evidence_digest"] = _sha256_json(evidence)

    evidence_path: Path | None = None
    status = "dry-run"
    if write:
        status, evidence_path = _publish_evidence(root, evidence)
    return RobustnessVerifyResult(
        output_path=root,
        evidence_path=evidence_path,
        status=status,
        evidence=evidence,
        protocol_digest=str(frozen.get("protocol_digest")),
        conditions=len(conditions),
        selection_cells=selection_cells,
        final_cells=final_cells,
        aggregate_count=aggregate_count,
    )


def _publish_evidence(
    root: Path, evidence: Mapping[str, object]
) -> tuple[str, Path | None]:
    destination = root / MATRIX_EVIDENCE_FILENAME
    payload = (json.dumps(evidence, sort_keys=True, indent=2) + "\n").encode("utf-8")
    try:
        fd, temporary = tempfile.mkstemp(
            dir=str(root), prefix=".matrix-evidence.", suffix=".tmp"
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
        raise RobustnessVerifyError(
            f"could not publish matrix evidence: {_bounded(exc)}"
        ) from exc
    return "published", destination


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dfl_sim.robustness_verify",
        description=(
            "Independently verify a completed raw robustness matrix from disk "
            "and publish matrix-evidence.json only when every source matches."
        ),
    )
    parser.add_argument("output", help="A completed robustness matrix root.")
    parser.add_argument(
        "--deep-cache",
        action="store_true",
        help="Recompute the full MNIST cache content identity (slower).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify_robustness_study(
            args.output, write=True, deep_cache=bool(args.deep_cache)
        )
    except (RobustnessVerifyError, RobustnessStudyError) as exc:
        print(f"dfl_sim: error: {_bounded(exc)}", file=sys.stderr)
        return 1
    print(
        "dfl_sim: robustness matrix verified "
        f"root={result.output_path} conditions={result.conditions} "
        f"selection_cells={result.selection_cells} "
        f"final_cells={result.final_cells} "
        f"aggregates={result.aggregate_count}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry
    raise SystemExit(main())
