"""Independent on-disk audit and publication gate for the M003/S04 headline.

``python -m dfl_sim.headline_verify RUNS_DIR`` reconstructs a completed
real-MNIST headline from its published artifacts and re-derives every
identity, schema, count, digest and paired metric *without* importing the
orchestration runner's success state. It is deliberately separate from
:mod:`dfl_sim.headline`:

* the frozen protocol is parsed from ``frozen-protocol.json`` and its digest
  recomputed;
* the verified T01 cache is reloaded and revalidated, and its published
  manifest/preprocessing identity is rebound to the frozen protocol;
* the three calibration artifacts are loaded through the production artifact
  loader and the S03 matched-artifact preflight gate re-runs against a study
  reconstructed from the frozen protocol + the *current* on-disk artifacts;
* the frozen strict study, both phase manifests, every cited schema-v1 trace
  and the paired robustness summary are re-read with the production loaders /
  :class:`~dfl_sim.trajectory.TraceReader`, and all 40 final aggregates are
  recomputed from the traces with the S02 helper;
* only when every check passes is ``headline-evidence.json`` published
  atomically. A mutated source fails closed and never overwrites a stale
  complete record as current.

This module produces S04 *raw headline evidence* (data provenance + matched
calibration identity + scope/limitations), not the S05 canonical report. It
makes no improvement, derivation-validity or strategic-response claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .ablation import isolation_statement
from .ablation_evaluation import (
    AblationEvaluationError,
    AblationMatchSummary,
    ablation_entry_metadata,
    ablation_manifest_metadata,
    preflight_ablation_study,
)
from .attack_schedule import derive_schedule_seed, realized_kappa_sequence, schedule_digest
from .bayesian_artifact import (
    CALIBRATION_VARIANTS,
    MANIFEST_FILENAME as ARTIFACT_MANIFEST_FILENAME,
    BayesianArtifactError,
    load_bayesian_artifact,
)
from .evaluation import (
    FINAL_MANIFEST_FILENAME,
    FINAL_MANIFEST_SCHEMA,
    FINAL_MANIFEST_SCHEMA_VERSION,
    ROBUSTNESS_SUMMARY_FILENAME,
    ROBUSTNESS_SUMMARY_SCHEMA,
    ROBUSTNESS_SUMMARY_SCHEMA_VERSION,
    SELECTION_MANIFEST_FILENAME,
    SELECTION_MANIFEST_SCHEMA,
    SELECTION_MANIFEST_SCHEMA_VERSION,
    EvaluationError,
    TraceReader,
    build_cell_config,
    build_robustness_summary,
    compute_shared_input_digest,
    final_output_digest,
    read_final_round_metrics,
    selection_output_digest,
)
from .evaluation_config import (
    EXTENDED_POLICY_IDS,
    HeldOutStudyConfig,
    HeldOutStudyConfigError,
    held_out_study_digest,
    load_held_out_study_config,
    selection_projection,
    selection_spec_digest,
    study_digest,
)
from .headline import (
    CALIBRATION_DIRNAME,
    FROZEN_PROTOCOL_FILENAME,
    FROZEN_STUDY_FILENAME,
    HEADLINE_PROTOCOL_VERSION,
    MNIST_PROVENANCE_FILENAME,
    STUDY_DIRNAME,
    HeadlineError,
    HeadlineProtocol,
    build_study,
    expected_cell_counts,
    protocol_digest,
)
from .mnist_provision import (
    CANONICAL_COUNTS,
    NUM_CLASSES,
    PREPROCESSING_ID,
    ProvisionManifest,
    ProvisioningError,
    _manifest_digest,
    _preprocessing_digest,
    verify_cache,
)
from .trajectory import MANIFEST_FILENAME as TRACE_MANIFEST_FILENAME, TrajectoryError

__all__ = [
    "HEADLINE_EVIDENCE_FILENAME",
    "HEADLINE_EVIDENCE_SCHEMA",
    "HEADLINE_EVIDENCE_SCHEMA_VERSION",
    "HeadlineVerifyDeps",
    "HeadlineVerifyError",
    "HeadlineVerifyResult",
    "build_parser",
    "main",
    "verify_headline",
]

HEADLINE_EVIDENCE_FILENAME = "headline-evidence.json"
HEADLINE_EVIDENCE_SCHEMA = "dfl_sim.headline_evidence/v1"
HEADLINE_EVIDENCE_SCHEMA_VERSION = 1

MNIST_PROVENANCE_SCHEMA = "dfl_sim.headline_mnist_provenance/v1"

VERIFY_MARKER = "dfl_sim: headline verified "

_MAX_ERROR_TEXT = 400

#: Fields that must be present on every aggregate evidence block.
_AGGREGATE_FIELDS = (
    "policy_id",
    "schedule_id",
    "held_out",
    "seeds",
    "count",
    "node_count",
    "mean_accuracy",
    "std_accuracy",
    "iqr_accuracy",
    "mean_loss",
    "worst_node_accuracy",
    "per_seed",
)

_LIMITATIONS: tuple[str, ...] = (
    "IID chain3 CPU subset headline only; not non-IID, topology, cost or wide "
    "parameter-sweep evidence.",
    "The reported accuracies describe a bounded real-MNIST subset, not full "
    "dataset training, and carry no statistical confidence interval.",
    "Singleton candidate budget: these are matched policy artifacts at a "
    "predeclared budget of one, not a hyperparameter search or winner claim.",
    "no_strategic is a fixed-attacker ablation; the headline does not "
    "establish strategic-response optimization or calibration-derivation "
    "validity.",
    "Equal or poor policy outcomes are honest evidence; no improvement claim "
    "is made.",
)


class HeadlineVerifyError(RuntimeError):
    """Raised when a completed headline cannot be independently verified."""


def _bounded(text: object, *, limit: int = _MAX_ERROR_TEXT) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) > limit:
        return collapsed[: limit - 1] + "\u2026"
    return collapsed


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise HeadlineVerifyError(f"missing headline artifact: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeadlineVerifyError(f"could not read {path}: {_bounded(exc)}") from exc
    if not isinstance(raw, Mapping):
        raise HeadlineVerifyError(f"{path} is not a JSON object")
    return raw


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise HeadlineVerifyError(detail)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HeadlineVerifyError(f"expected an integer, got {value!r}")
    return value


def _as_list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise HeadlineVerifyError(f"{label} must be a sequence")
    return list(value)


# --------------------------------------------------------------------------- #
# Injected dependencies
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeadlineVerifyDeps:
    """Optional injection seams; production uses the real public surfaces.

    The seams exist so offline tests can substitute an isolated provisioner
    validation path and a fixture-labelled artifact preflight. Production never
    passes any of them, so the default path is the unpatched real-data gate.
    """

    cache_verify: Callable[[str, bool], ProvisionManifest] | None = None
    artifact_loader: Callable[[str], object] | None = None
    preflight: Callable[[object], AblationMatchSummary] | None = None
    trace_reader: Callable[[object], Any] | None = None
    study_builder: Callable[[HeadlineProtocol, str, Mapping[str, object]], HeldOutStudyConfig] | None = None


@dataclass(frozen=True)
class HeadlineVerifyResult:
    output_path: Path
    evidence_path: Path | None
    status: str
    evidence: Mapping[str, object]
    protocol_digest: str
    selection_cells: int
    final_cells: int
    aggregate_count: int


def _default_cache_verify(data_root: str, deep: bool) -> ProvisionManifest:
    return verify_cache(data_root, deep=deep)


# --------------------------------------------------------------------------- #
# Protocol / cache resolution
# --------------------------------------------------------------------------- #
def _load_protocol(frozen: Mapping[str, Any]) -> HeadlineProtocol:
    _require(
        frozen.get("schema") == HEADLINE_PROTOCOL_VERSION,
        "frozen-protocol.json has an unexpected schema",
    )
    payload = frozen.get("protocol")
    _require(isinstance(payload, Mapping), "frozen-protocol.json is missing the protocol")
    try:
        protocol = HeadlineProtocol.model_validate(payload)
    except ValidationError as exc:
        raise HeadlineVerifyError(
            f"frozen protocol is invalid: {_bounded(exc)}"
        ) from exc
    _require(
        protocol.data_provenance == "real_mnist",
        "frozen protocol data_provenance is not real_mnist",
    )
    stored_digest = frozen.get("protocol_digest")
    recomputed = protocol_digest(protocol)
    _require(
        stored_digest == recomputed,
        "frozen protocol digest does not match the resolved protocol",
    )
    _require(frozen.get("study_id") == protocol.study_id, "frozen study_id mismatch")
    _require(
        frozen.get("data_provenance") == "real_mnist",
        "frozen data_provenance is not real_mnist",
    )
    _require(
        list(frozen.get("selection_seeds", [])) == list(protocol.selection_seeds),
        "frozen selection_seeds disagree with the protocol",
    )
    _require(
        list(frozen.get("final_seeds", [])) == list(protocol.final_seeds),
        "frozen final_seeds disagree with the protocol",
    )
    _require(
        list(frozen.get("known_schedule_ids", []))
        == [entry.id for entry in protocol.schedules],
        "frozen known_schedule_ids disagree with the protocol",
    )
    _require(
        frozen.get("held_out_schedule_id") == protocol.held_out_schedule.id,
        "frozen held_out_schedule_id disagrees with the protocol",
    )
    _require(
        frozen.get("candidate_budget") == protocol.candidate_budget,
        "frozen candidate_budget disagrees with the protocol",
    )
    cache = frozen.get("cache")
    _require(isinstance(cache, Mapping), "frozen-protocol.json is missing the cache block")
    return protocol


def _load_provenance(
    root: Path, frozen: Mapping[str, Any], protocol: HeadlineProtocol, deps: HeadlineVerifyDeps, deep: bool
) -> tuple[ProvisionManifest, Mapping[str, Any]]:
    provenance = _read_json(root / MNIST_PROVENANCE_FILENAME)
    _require(
        provenance.get("schema") == MNIST_PROVENANCE_SCHEMA,
        "mnist-provenance.json has an unexpected schema",
    )
    _require(provenance.get("real_mnist") is True, "provenance does not declare real MNIST")
    payload = provenance.get("manifest")
    _require(isinstance(payload, Mapping), "provenance is missing the cache manifest")
    try:
        manifest = ProvisionManifest.model_validate_json(
            json.dumps(payload, sort_keys=True)
        )
    except ValidationError as exc:
        raise HeadlineVerifyError(
            f"provenance manifest is invalid: {_bounded(exc)}"
        ) from exc
    _require(
        _manifest_digest(manifest) == manifest.manifest_digest,
        "provenance manifest digest does not match its body",
    )
    _require(
        provenance.get("manifest_digest") == manifest.manifest_digest,
        "provenance manifest_digest field disagrees with the manifest",
    )
    cache = frozen.get("cache")
    assert isinstance(cache, Mapping)
    _require(
        cache.get("manifest_digest") == manifest.manifest_digest,
        "frozen cache manifest_digest disagrees with the provenance",
    )
    _require(
        cache.get("preprocessing_digest") == manifest.preprocessing_digest,
        "frozen cache preprocessing_digest disagrees with the provenance",
    )
    _require(
        cache.get("preprocessing_id") == manifest.preprocessing_id,
        "frozen cache preprocessing_id disagrees with the provenance",
    )

    # Canonical full-dataset identity, independent of the subset protocol.
    _require(manifest.dataset == "mnist", "cache manifest dataset is not mnist")
    _require(
        manifest.train.split == "train" and manifest.test.split == "test",
        "cache manifest split identity is wrong",
    )
    expected_counts = dict(CANONICAL_COUNTS)
    _require(
        manifest.train.count == expected_counts["train"]
        and manifest.test.count == expected_counts["test"],
        "cache manifest does not describe the canonical full MNIST split counts",
    )
    for split in (manifest.train, manifest.test):
        _require(
            tuple(split.image_shape) == (28, 28),
            f"{split.split} image shape is not 28x28",
        )
        _require(
            0 <= split.label_min <= split.label_max < NUM_CLASSES,
            f"{split.split} label bounds are out of range",
        )

    # Preprocessing identity must still match the current code contract.
    _require(
        manifest.preprocessing_id == PREPROCESSING_ID,
        "cache preprocessing_id changed since the headline ran",
    )
    _require(
        manifest.preprocessing_digest == _preprocessing_digest(),
        "cache preprocessing_digest changed since the headline ran",
    )

    data_root = provenance.get("data_root")
    _require(
        isinstance(data_root, str) and data_root,
        "provenance is missing the resolved data root",
    )
    verify = deps.cache_verify or _default_cache_verify
    try:
        actual = verify(data_root, deep)
    except (ProvisioningError, OSError) as exc:
        raise HeadlineVerifyError(
            f"verified cache is unavailable or mutated: {_bounded(exc)}"
        ) from exc
    actual_payload = actual.model_dump(mode="json")
    _require(
        actual_payload.get("manifest_digest") == manifest.manifest_digest,
        "on-disk cache manifest digest no longer matches the published provenance",
    )
    _require(
        _canonical_bytes(actual_payload) == _canonical_bytes(dict(payload)),
        "on-disk cache content identity differs from the published provenance",
    )
    return manifest, actual_payload


# --------------------------------------------------------------------------- #
# Study / matched artifact resolution
# --------------------------------------------------------------------------- #
def _load_artifacts(root: Path, deps: HeadlineVerifyDeps) -> dict[str, object]:
    loader = deps.artifact_loader or load_bayesian_artifact
    artifacts: dict[str, object] = {}
    for variant in CALIBRATION_VARIANTS:
        path = root / CALIBRATION_DIRNAME / variant
        try:
            artifacts[variant] = loader(str(path))
        except (BayesianArtifactError, OSError) as exc:
            raise HeadlineVerifyError(
                f"could not load the {variant!r} calibration artifact: {_bounded(exc)}"
            ) from exc
    return artifacts


def _reconstruct_study(
    protocol: HeadlineProtocol,
    root: Path,
    artifacts: Mapping[str, object],
    deps: HeadlineVerifyDeps,
) -> HeldOutStudyConfig:
    builder = deps.study_builder or build_study
    try:
        return builder(protocol, str(root), artifacts)
    except (HeadlineError, ValidationError) as exc:
        raise HeadlineVerifyError(
            f"could not reconstruct the frozen study from the protocol and artifacts: "
            f"{_bounded(exc)}"
        ) from exc


def _load_frozen_study(root: Path) -> HeldOutStudyConfig:
    try:
        return load_held_out_study_config(root / FROZEN_STUDY_FILENAME)
    except (HeldOutStudyConfigError, OSError) as exc:
        raise HeadlineVerifyError(f"frozen study is invalid: {_bounded(exc)}") from exc


def _bind_study(
    expected: HeldOutStudyConfig, frozen_study: HeldOutStudyConfig
) -> tuple[str, str, str]:
    expected_study_digest = study_digest(expected.study)
    frozen_study_digest = study_digest(frozen_study.study)
    _require(
        expected_study_digest == frozen_study_digest,
        "the frozen study diverges from the protocol plus the current artifacts",
    )
    _require(
        expected.study.base == frozen_study.study.base,
        "the frozen study base config diverges from the frozen protocol",
    )
    expected_selection_digest = selection_spec_digest(selection_projection(expected))
    frozen_selection_digest = selection_spec_digest(selection_projection(frozen_study))
    _require(
        expected_selection_digest == frozen_selection_digest,
        "the frozen selection projection diverges from the protocol",
    )
    expected_held_out_digest = held_out_study_digest(expected)
    frozen_held_out_digest = held_out_study_digest(frozen_study)
    _require(
        expected_held_out_digest == frozen_held_out_digest,
        "the frozen held-out study identity diverges from the protocol",
    )
    return expected_study_digest, expected_selection_digest, expected_held_out_digest


def _matched_artifacts(
    study: HeldOutStudyConfig, deps: HeadlineVerifyDeps
) -> AblationMatchSummary:
    preflight = deps.preflight or preflight_ablation_study
    try:
        match = preflight(study.study)
    except (AblationEvaluationError, OSError) as exc:
        raise HeadlineVerifyError(
            f"matched-artifact preflight failed closed: {_bounded(exc)}"
        ) from exc
    _require(
        match.data_provenance == "production",
        "calibration derivation provenance is injected, not production",
    )
    for policy_id in match.matched_policy_ids:
        identity = match.identity(policy_id)
        _require(
            identity.data_provenance == "production",
            f"calibration derivation for {policy_id!r} is injected, not production",
        )
    return match


def _shared_input_digest(study: HeldOutStudyConfig) -> str:
    return compute_shared_input_digest(study.study.base)


def _cell_config_digest(study: HeldOutStudyConfig, policy_id: str, seed: int) -> str:
    panel = {entry.id: entry for entry in study.study.panel}
    config = build_cell_config(study.study.base, panel[policy_id].aggregation, seed)
    payload = config.model_dump(mode="json")
    aggregation = payload.get("aggregation")
    if isinstance(aggregation, dict) and "artifact_dir" in aggregation:
        aggregation["artifact_dir"] = "<artifact>"
    return _sha256_json(payload)


def _schedule_map(study: HeldOutStudyConfig) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for entry in (*study.known_schedules, study.held_out_schedule):
        mapping[entry.id] = entry
    return mapping


def _expected_cells(
    seeds: Sequence[int], schedule_ids: Sequence[str], study: HeldOutStudyConfig
) -> set[tuple[int, str, str]]:
    panel = tuple(entry.id for entry in study.study.panel)
    return {
        (int(seed), str(schedule_id), str(policy_id))
        for seed in seeds
        for schedule_id in schedule_ids
        for policy_id in panel
    }


# --------------------------------------------------------------------------- #
# Manifest / trace verification
# --------------------------------------------------------------------------- #
def _require_manifest_envelope(
    manifest: Mapping[str, Any],
    *,
    schema: str,
    version: int,
    label: str,
) -> None:
    _require(manifest.get("schema") == schema, f"{label} has an unexpected schema")
    _require(
        manifest.get("schema_version") == version,
        f"{label} has an unexpected schema_version",
    )
    _require(
        manifest.get("data_provenance") == "real_mnist",
        f"{label} does not declare real_mnist provenance",
    )
    panel = manifest.get("panel")
    _require(
        isinstance(panel, Sequence) and not isinstance(panel, (str, bytes)),
        f"{label} panel must be a sequence",
    )
    _require(
        sorted(str(item) for item in panel) == sorted(EXTENDED_POLICY_IDS),
        f"{label} does not declare the eight-policy panel",
    )


def _verify_entry_common(
    entry: Mapping[str, Any],
    *,
    study: HeldOutStudyConfig,
    schedules: Mapping[str, Any],
    match: AblationMatchSummary,
    label: str,
) -> tuple[int, str, str, Any, tuple[float, ...]]:
    try:
        seed = _as_int(entry["seed"])
        policy_id = str(entry["policy_id"])
        schedule_id = str(entry["schedule_id"])
    except (KeyError, HeadlineVerifyError) as exc:
        raise HeadlineVerifyError(f"{label} entry is missing its cell identity") from exc
    cell_label = f"{label} policy {policy_id!r} seed {seed} schedule {schedule_id!r}"

    _require(entry.get("status") == "complete", f"{cell_label} is not complete")
    _require(
        entry.get("pairing_key") == f"{policy_id}:seed-{seed}:{schedule_id}",
        f"{cell_label} pairing_key is wrong",
    )
    schedule = schedules.get(schedule_id)
    _require(schedule is not None, f"{cell_label} cites an unknown schedule")
    _require(
        entry.get("schedule_digest") == schedule_digest(schedule.schedule),
        f"{cell_label} schedule_digest disagrees with the frozen schedule",
    )
    _require(
        entry.get("schedule") == schedule.schedule.model_dump(mode="json"),
        f"{cell_label} schedule payload disagrees with the frozen schedule",
    )
    rounds = int(study.study.base.rounds)
    expected_kappa = realized_kappa_sequence(
        schedule.schedule, rounds, derive_schedule_seed(seed, schedule.schedule)
    )
    realized_raw = _as_list(entry.get("realized_kappa_sequence"), label=f"{cell_label} kappa")
    try:
        realized = tuple(float(value) for value in realized_raw)
    except (TypeError, ValueError) as exc:
        raise HeadlineVerifyError(f"{cell_label} realized kappa is malformed") from exc
    _require(
        realized == tuple(float(value) for value in expected_kappa),
        f"{cell_label} realized_kappa_sequence disagrees with the frozen schedule",
    )
    _require(
        entry.get("config_digest") == _cell_config_digest(study, policy_id, seed),
        f"{cell_label} config_digest disagrees with the reconstructed cell config",
    )
    panel = {item.id: item for item in study.study.panel}
    _require(policy_id in panel, f"{cell_label} cites an unknown policy")
    _require(
        entry.get("policy_kind") == panel[policy_id].aggregation.kind,
        f"{cell_label} policy_kind disagrees with the frozen panel",
    )
    expected_artifact = None
    if policy_id in match.variants:
        expected_artifact = match.identity(policy_id).aggregate_digest
    _require(
        entry.get("artifact_digest") == expected_artifact,
        f"{cell_label} artifact_digest disagrees with the matched artifact",
    )
    expected_ablation = ablation_entry_metadata(match, policy_id)
    if expected_ablation is not None:
        _require(
            entry.get("ablation") == expected_ablation,
            f"{cell_label} ablation identity disagrees with the matched artifact",
        )
    else:
        _require(
            "ablation" not in entry,
            f"{cell_label} carries unexpected ablation metadata",
        )
    return seed, policy_id, schedule_id, schedule, realized


def _resolve_trace_path(
    entry: Mapping[str, Any],
    *,
    phase_dir: Path,
    policy_id: str,
    seed: int,
    schedule_id: str,
    seen: set[Path],
    label: str,
) -> Path:
    expected = f"traces/{schedule_id}/{policy_id}/seed-{seed}"
    recorded = entry.get("trace_path")
    _require(
        recorded == expected,
        f"{label} trace_path {recorded!r} does not match the canonical location {expected!r}",
    )
    candidate = phase_dir / expected
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise HeadlineVerifyError(f"{label} trace path is unresolvable: {_bounded(exc)}") from exc
    try:
        root = phase_dir.resolve()
    except OSError as exc:  # pragma: no cover - defensive
        raise HeadlineVerifyError(f"{label} trace root is unresolvable: {_bounded(exc)}") from exc
    _require(
        resolved.is_relative_to(root),
        f"{label} trace path escapes the phase directory",
    )
    _require(
        resolved not in seen,
        f"{label} aliases a trace already cited by another cell",
    )
    _require(resolved.is_dir(), f"{label} trace directory is missing")
    seen.add(resolved)
    return resolved


def _verify_trace(
    entry: Mapping[str, Any],
    *,
    trace_dir: Path,
    study: HeldOutStudyConfig,
    policy_id: str,
    seed: int,
    schedule: Any,
    realized: tuple[float, ...],
    deps: HeadlineVerifyDeps,
    label: str,
) -> dict[str, Any]:
    reader_factory = deps.trace_reader or TraceReader
    base = study.study.base
    try:
        with reader_factory(trace_dir) as reader:
            schema = int(getattr(reader, "schema_version"))
            _require(schema == 1, f"{label} trace is not schema version 1")
            config = getattr(reader, "config", None)
            events = getattr(reader, "events", None)
            _require(config is not None and events is not None, f"{label} trace exposes no config/events")
            _require(int(config.seed) == seed, f"{label} trace seed disagrees with the manifest")
            _require(
                str(config.aggregation.kind) == str(entry.get("policy_kind")),
                f"{label} trace aggregation disagrees with the manifest policy",
            )
            _require(int(config.rounds) == int(base.rounds), f"{label} trace rounds disagree with the study")
            _require(
                str(config.model.name) == str(base.model.name),
                f"{label} trace model disagrees with the study",
            )
            _require(
                str(config.topology.kind) == str(base.topology.kind),
                f"{label} trace topology disagrees with the study",
            )
            _require(config.device == base.device, f"{label} trace device disagrees with the study")
            _require(bool(config.bitwise) == bool(base.bitwise), f"{label} trace bitwise flag disagrees")
            _require(
                bool(config.data.download) is False,
                f"{label} trace config enables dataset download",
            )
            _require(
                str(config.data.dataset) == str(base.data.dataset),
                f"{label} trace dataset disagrees with the study",
            )
            _require(
                str(config.data.partition) == str(base.data.partition),
                f"{label} trace partition disagrees with the study",
            )
            _require(
                int(config.data.samples_per_node) == int(base.data.samples_per_node),
                f"{label} trace training subset disagrees with the study",
            )
            _require(
                int(config.evaluation.max_samples) == int(base.evaluation.max_samples),
                f"{label} trace evaluation subset disagrees with the study",
            )
            _require(
                config.attack.compromised_node == base.attack.compromised_node,
                f"{label} trace compromised node disagrees with the study",
            )

            events = tuple(events)
            _require(events, f"{label} trace has no evaluated rounds")
            compromised = base.attack.compromised_node
            trace_kappa: list[float] = []
            for event in events:
                nodes = event.get("nodes")
                _require(isinstance(nodes, Sequence) and nodes, f"{label} trace round has no nodes")
                match_node = None
                for node in nodes:
                    if int(node.get("node")) == int(compromised):
                        match_node = node
                        break
                _require(match_node is not None, f"{label} trace is missing the compromised node")
                trace_kappa.append(float(match_node["attack"]["kappa"]))
            _require(
                tuple(trace_kappa) == realized,
                f"{label} trace realized attack disagrees with the manifest",
            )
    except HeadlineVerifyError:
        raise
    except (TrajectoryError, OSError, TypeError, ValueError, KeyError) as exc:
        raise HeadlineVerifyError(f"{label} produced an invalid trace: {_bounded(exc)}") from exc
    return {
        "policy_id": policy_id,
        "seed": seed,
        "schedule_id": schedule.id,
        "trace_manifest_sha256": _file_sha256(trace_dir / TRACE_MANIFEST_FILENAME),
    }


def _verify_cells(
    entries: Sequence[Any],
    *,
    study: HeldOutStudyConfig,
    match: AblationMatchSummary,
    phase_dir: Path,
    expected_cells: set[tuple[int, str, str]],
    label: str,
    deps: HeadlineVerifyDeps,
    seen: set[Path],
    held_out_id: str | None = None,
) -> list[dict[str, Any]]:
    schedules = _schedule_map(study)
    observed: set[tuple[int, str, str]] = set()
    trace_info: list[dict[str, Any]] = []
    attack_by_cell: dict[tuple[int, str], tuple[float, ...]] = {}
    for raw_entry in entries:
        _require(isinstance(raw_entry, Mapping), f"{label} entry must be an object")
        seed, policy_id, schedule_id, schedule, realized = _verify_entry_common(
            raw_entry, study=study, schedules=schedules, match=match, label=label
        )
        cell = (seed, schedule_id, policy_id)
        _require(cell not in observed, f"{label} contains a duplicate cell {cell!r}")
        observed.add(cell)
        if held_out_id is None:
            _require(
                not bool(raw_entry.get("held_out", False)),
                f"{label} selection must not contain a held_out cell",
            )
        else:
            _require(
                bool(raw_entry.get("held_out", False)) == (schedule_id == held_out_id),
                f"{label} cell {cell!r} has an inconsistent held_out flag",
            )
        cell_label = f"{label} policy {policy_id!r} seed {seed} schedule {schedule_id!r}"
        written = attack_by_cell.setdefault((seed, schedule_id), realized)
        _require(
            written == realized,
            f"{label} paired policies disagree on the realized attack for seed/schedule",
        )
        resolved = _resolve_trace_path(
            raw_entry,
            phase_dir=phase_dir,
            policy_id=policy_id,
            seed=seed,
            schedule_id=schedule_id,
            seen=seen,
            label=cell_label,
        )
        trace_info.append(
            _verify_trace(
                raw_entry,
                trace_dir=resolved,
                study=study,
                policy_id=policy_id,
                seed=seed,
                schedule=schedule,
                realized=realized,
                deps=deps,
                label=cell_label,
            )
        )
    _require(observed == expected_cells, f"{label} does not cover the exact expected cell matrix")
    return trace_info


def _verify_selection(
    root: Path,
    *,
    study: HeldOutStudyConfig,
    match: AblationMatchSummary,
    study_digest_value: str,
    selection_spec_value: str,
    deps: HeadlineVerifyDeps,
) -> tuple[Mapping[str, Any], Path, list[dict[str, Any]]]:
    phase_dir = root / STUDY_DIRNAME / "selection"
    manifest = _read_json(phase_dir / SELECTION_MANIFEST_FILENAME)
    _require_manifest_envelope(
        manifest,
        schema=SELECTION_MANIFEST_SCHEMA,
        version=SELECTION_MANIFEST_SCHEMA_VERSION,
        label="selection manifest",
    )
    _require(manifest.get("status") == "frozen", "selection manifest is not frozen")
    _require(manifest.get("study_id") == study.study.study_id, "selection study_id mismatch")
    _require(manifest.get("study_digest") == study_digest_value, "selection study_digest mismatch")
    _require(
        manifest.get("selection_spec_digest") == selection_spec_value,
        "selection selection_spec_digest mismatch",
    )
    _require(
        manifest.get("shared_input_digest") == _shared_input_digest(study),
        "selection shared_input_digest mismatch",
    )
    _require(
        manifest.get("selection_output_digest") == selection_output_digest(manifest),
        "selection manifest self-digest mismatch",
    )
    _require(
        manifest.get("ablation") == ablation_manifest_metadata(match),
        "selection ablation identity mismatch",
    )
    selection_seeds = tuple(sorted(int(seed) for seed in _as_list(manifest.get("selection_seeds"), label="selection_seeds")))
    _require(
        set(selection_seeds) == set(study.study.seeds),
        "selection manifest seed set disagrees with the frozen study",
    )
    schedule_ids = tuple(sorted(str(item) for item in _as_list(manifest.get("schedule_ids"), label="schedule_ids")))
    _require(
        set(schedule_ids) == {entry.id for entry in study.known_schedules},
        "selection schedule coverage disagrees with the frozen study",
    )
    entries = _as_list(manifest.get("entries"), label="selection entries")
    expected = _expected_cells(selection_seeds, schedule_ids, study)
    seen: set[Path] = set()
    traces = _verify_cells(
        entries,
        study=study,
        match=match,
        phase_dir=phase_dir,
        expected_cells=expected,
        label="selection",
        deps=deps,
        seen=seen,
    )
    return manifest, phase_dir, traces


def _verify_final(
    root: Path,
    *,
    study: HeldOutStudyConfig,
    match: AblationMatchSummary,
    selection_manifest: Mapping[str, Any],
    study_digest_value: str,
    selection_spec_value: str,
    held_out_digest_value: str,
    deps: HeadlineVerifyDeps,
) -> tuple[Mapping[str, Any], Path, list[dict[str, Any]]]:
    phase_dir = root / STUDY_DIRNAME / "final"
    manifest = _read_json(phase_dir / FINAL_MANIFEST_FILENAME)
    _require_manifest_envelope(
        manifest,
        schema=FINAL_MANIFEST_SCHEMA,
        version=FINAL_MANIFEST_SCHEMA_VERSION,
        label="final manifest",
    )
    _require(manifest.get("phase") == "final", "final manifest phase is not final")
    _require(manifest.get("status") == "complete", "final manifest is not complete")
    _require(manifest.get("study_id") == study.study.study_id, "final study_id mismatch")
    _require(manifest.get("study_digest") == study_digest_value, "final study_digest mismatch")
    _require(
        manifest.get("selection_spec_digest") == selection_spec_value,
        "final selection_spec_digest mismatch",
    )
    _require(
        manifest.get("held_out_study_digest") == held_out_digest_value,
        "final held_out_study_digest mismatch",
    )
    _require(
        manifest.get("selection_output_digest") == selection_manifest.get("selection_output_digest"),
        "final selection_output_digest disagrees with the frozen selection manifest",
    )
    _require(
        manifest.get("shared_input_digest") == _shared_input_digest(study),
        "final shared_input_digest mismatch",
    )
    _require(
        manifest.get("final_output_digest") == final_output_digest(manifest),
        "final manifest self-digest mismatch",
    )
    _require(
        manifest.get("ablation") == ablation_manifest_metadata(match),
        "final ablation identity mismatch",
    )
    _require(
        manifest.get("held_out_schedule_id") == study.held_out_schedule.id,
        "final held_out_schedule_id disagrees with the frozen study",
    )
    selection_seeds = tuple(sorted(int(seed) for seed in _as_list(manifest.get("selection_seeds"), label="selection_seeds")))
    final_seeds = tuple(sorted(int(seed) for seed in _as_list(manifest.get("final_seeds"), label="final_seeds")))
    _require(
        set(selection_seeds) == set(study.study.seeds),
        "final selection seed set disagrees with the frozen study",
    )
    _require(
        set(final_seeds) == set(study.final_seeds),
        "final seed set disagrees with the frozen study",
    )
    _require(
        not set(selection_seeds) & set(final_seeds),
        "selection and final seeds overlap",
    )
    _require(
        len(final_seeds) == len(set(final_seeds)),
        "final seed set is not unique",
    )
    all_schedule_ids = {entry.id for entry in study.known_schedules} | {study.held_out_schedule.id}
    schedule_ids = tuple(sorted(str(item) for item in _as_list(manifest.get("schedule_ids"), label="schedule_ids")))
    _require(
        set(schedule_ids) == all_schedule_ids,
        "final schedule coverage disagrees with the frozen study",
    )
    entries = _as_list(manifest.get("entries"), label="final entries")
    expected = _expected_cells(final_seeds, schedule_ids, study)
    seen: set[Path] = set()
    traces = _verify_cells(
        entries,
        study=study,
        match=match,
        phase_dir=phase_dir,
        expected_cells=expected,
        label="final",
        deps=deps,
        seen=seen,
        held_out_id=study.held_out_schedule.id,
    )
    held_out_count = sum(1 for trace in traces if trace["schedule_id"] == study.held_out_schedule.id)
    _require(
        held_out_count == len(final_seeds) * len(EXTENDED_POLICY_IDS),
        "final held_out coverage is wrong",
    )
    return manifest, phase_dir, traces


# --------------------------------------------------------------------------- #
# Metric / aggregate recomputation
# --------------------------------------------------------------------------- #
def _compare_exact(actual: Any, expected: Any, *, label: str) -> None:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        _require(set(actual) == set(expected), f"{label} keys disagree")
        for key in expected:
            _compare_exact(actual[key], expected[key], label=f"{label}.{key}")
        return
    if (
        isinstance(actual, Sequence)
        and isinstance(expected, Sequence)
        and not isinstance(actual, (str, bytes))
        and not isinstance(expected, (str, bytes))
    ):
        _require(len(actual) == len(expected), f"{label} length disagrees")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_exact(left, right, label=f"{label}[{index}]")
        return
    if isinstance(actual, float) or isinstance(expected, float):
        _require(
            isinstance(actual, (int, float))
            and isinstance(expected, (int, float))
            and math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12),
            f"{label} value {actual!r} disagrees with {expected!r}",
        )
        return
    _require(actual == expected, f"{label} value {actual!r} disagrees with {expected!r}")


def _verify_embedded_metrics(
    manifest: Mapping[str, Any], phase_dir: Path, deps: HeadlineVerifyDeps
) -> None:
    reader_factory = deps.trace_reader or TraceReader
    for entry in _as_list(manifest.get("entries"), label="final entries"):
        label = (
            f"final policy {entry.get('policy_id')!r} seed {entry.get('seed')} "
            f"schedule {entry.get('schedule_id')!r}"
        )
        recorded = entry.get("metrics")
        _require(isinstance(recorded, Mapping), f"{label} is missing its embedded metrics block")
        trace_path = phase_dir / str(entry["trace_path"])
        try:
            recomputed = read_final_round_metrics(trace_path, trace_reader=reader_factory)
        except (EvaluationError, TrajectoryError, OSError) as exc:
            raise HeadlineVerifyError(f"{label} embedded metrics cannot be recomputed: {_bounded(exc)}") from exc
        _compare_exact(dict(recorded), recomputed, label=f"{label} metrics")


def _verify_summary(
    root: Path,
    *,
    final_manifest: Mapping[str, Any],
    final_dir: Path,
    deps: HeadlineVerifyDeps,
) -> Mapping[str, Any]:
    reader_factory = deps.trace_reader or TraceReader
    try:
        recomputed = build_robustness_summary(
            final_manifest,
            trace_root=final_dir,
            selection_seeds=final_manifest.get("selection_seeds"),
            trace_reader=reader_factory,
        )
    except (EvaluationError, TrajectoryError, OSError) as exc:
        raise HeadlineVerifyError(f"paired metrics could not be recomputed: {_bounded(exc)}") from exc

    on_disk = _read_json(root / STUDY_DIRNAME / ROBUSTNESS_SUMMARY_FILENAME)
    _require(
        on_disk.get("schema") == ROBUSTNESS_SUMMARY_SCHEMA,
        "robustness summary has an unexpected schema",
    )
    _require(
        on_disk.get("schema_version") == ROBUSTNESS_SUMMARY_SCHEMA_VERSION,
        "robustness summary has an unexpected schema_version",
    )
    _require(on_disk.get("status") == "complete", "robustness summary is not complete")
    envelope = (
        "study_id",
        "data_provenance",
        "study_digest",
        "selection_spec_digest",
        "held_out_study_digest",
        "final_output_digest",
        "selection_seeds",
        "final_seeds",
        "panel",
        "schedule_ids",
        "std_method",
        "quantile_method",
        "ablation",
    )
    for key in envelope:
        _require(
            on_disk.get(key) == recomputed.get(key),
            f"robustness summary {key} disagrees with the recomputation",
        )
    on_disk_aggregates = _as_list(on_disk.get("aggregates"), label="summary aggregates")
    expected_aggregates = recomputed["aggregates"]
    _require(
        len(on_disk_aggregates) == len(expected_aggregates),
        "robustness summary aggregate count disagrees with the recomputation",
    )
    by_key = {
        (str(row.get("schedule_id")), str(row.get("policy_id"))): row
        for row in on_disk_aggregates
    }
    _require(len(by_key) == len(on_disk_aggregates), "robustness summary contains duplicate aggregates")
    for expected_row in expected_aggregates:
        key = (str(expected_row["schedule_id"]), str(expected_row["policy_id"]))
        actual_row = by_key.get(key)
        _require(actual_row is not None, f"robustness summary is missing aggregate {key!r}")
        for field_name in _AGGREGATE_FIELDS:
            _require(
                field_name in actual_row,
                f"robustness summary aggregate {key!r} is missing {field_name!r}",
            )
        _compare_exact(dict(actual_row), dict(expected_row), label=f"aggregate {key!r}")
    return on_disk


# --------------------------------------------------------------------------- #
# Evidence assembly
# --------------------------------------------------------------------------- #
def _split_identity(split: object) -> dict[str, Any]:
    dump = split.model_dump(mode="json")  # type: ignore[attr-defined]
    return {
        "count": dump["count"],
        "image_shape": list(dump["image_shape"]),
        "label_min": dump["label_min"],
        "label_max": dump["label_max"],
        "decoded_image_sha256": dump["decoded_image_sha256"],
        "decoded_label_sha256": dump["decoded_label_sha256"],
        "raw_image_sha256": dump["raw_image_sha256"],
        "raw_label_sha256": dump["raw_label_sha256"],
    }


def _aggregate_evidence(aggregates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for row in aggregates:
        block = {
            "policy_id": row["policy_id"],
            "schedule_id": row["schedule_id"],
            "held_out": row["held_out"],
            "seeds": row["seeds"],
            "count": row["count"],
            "mean_accuracy": row["mean_accuracy"],
            "std_accuracy": row["std_accuracy"],
            "iqr_accuracy": row["iqr_accuracy"],
            "mean_loss": row["mean_loss"],
            "worst_node_accuracy": row["worst_node_accuracy"],
        }
        block["digest"] = _sha256_json(block)
        blocks.append(block)
    return blocks


def _build_evidence(
    *,
    root: Path,
    frozen: Mapping[str, Any],
    protocol: HeadlineProtocol,
    manifest: ProvisionManifest,
    frozen_study: HeldOutStudyConfig,
    match: AblationMatchSummary,
    selection_manifest: Mapping[str, Any],
    final_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    selection_traces: Sequence[Mapping[str, Any]],
    final_traces: Sequence[Mapping[str, Any]],
    deep: bool,
) -> dict[str, Any]:
    base = frozen_study.study.base
    threads = int(frozen.get("threads", protocol.threads))
    calibration = protocol.calibration
    evidence_blocks: list[dict[str, Any]] = []
    for kind, path in (
        ("frozen_protocol", root / FROZEN_PROTOCOL_FILENAME),
        ("mnist_provenance", root / MNIST_PROVENANCE_FILENAME),
        ("frozen_study", root / FROZEN_STUDY_FILENAME),
    ):
        evidence_blocks.append(
            {
                "kind": kind,
                "path": str(path.relative_to(root)),
                "sha256": _file_sha256(path),
            }
        )
    for variant in CALIBRATION_VARIANTS:
        artifact_manifest = root / CALIBRATION_DIRNAME / variant / ARTIFACT_MANIFEST_FILENAME
        evidence_blocks.append(
            {
                "kind": "calibration_artifact_manifest",
                "variant": variant,
                "path": str(artifact_manifest.relative_to(root)),
                "aggregate_digest": match.identity(
                    next(
                        policy_id
                        for policy_id in match.matched_policy_ids
                        if match.variant(policy_id) == variant
                    )
                ).aggregate_digest,
                "sha256": _file_sha256(artifact_manifest),
            }
        )
    for kind, path in (
        ("selection_manifest", root / STUDY_DIRNAME / "selection" / SELECTION_MANIFEST_FILENAME),
        ("final_manifest", root / STUDY_DIRNAME / "final" / FINAL_MANIFEST_FILENAME),
        ("robustness_summary", root / STUDY_DIRNAME / ROBUSTNESS_SUMMARY_FILENAME),
    ):
        evidence_blocks.append(
            {
                "kind": kind,
                "path": str(path.relative_to(root)),
                "sha256": _file_sha256(path),
            }
        )

    variants = {
        policy_id: {
            "variant": identity.variant,
            "aggregate_digest": identity.aggregate_digest,
            "calibration_config_digest": identity.calibration_config_digest,
            "dataset_inputs_digest": identity.dataset_inputs_digest,
            "partition_membership_digest": identity.partition_membership_digest,
            "schedule_pairing_digest": identity.schedule_pairing_digest,
            "data_provenance": identity.data_provenance,
            "assumption_version": identity.protocol_version,
            "episodes": identity.episodes,
            "rounds": identity.rounds,
            "design_size": identity.design_size,
            "candidate_protocol": identity.candidate_protocol,
            "candidate_cap": identity.candidate_cap,
            "isolation_statement": isolation_statement(identity.variant),
        }
        for policy_id in match.matched_policy_ids
        for identity in (match.identity(policy_id),)
    }

    selection_seeds = [int(seed) for seed in selection_manifest.get("selection_seeds", [])]
    final_seeds = [int(seed) for seed in final_manifest.get("final_seeds", [])]

    evidence: dict[str, Any] = {
        "schema": HEADLINE_EVIDENCE_SCHEMA,
        "schema_version": HEADLINE_EVIDENCE_SCHEMA_VERSION,
        "status": "complete",
        "study_id": protocol.study_id,
        "data_provenance": "real_mnist",
        "protocol": {
            "protocol_digest": frozen.get("protocol_digest"),
            "source": frozen.get("source"),
            "threads": threads,
            "selection_seeds": list(protocol.selection_seeds),
            "final_seeds": list(protocol.final_seeds),
            "candidate_budget": protocol.candidate_budget,
            "known_schedule_ids": [entry.id for entry in protocol.schedules],
            "held_out_schedule_id": protocol.held_out_schedule.id,
        },
        "data_identity": {
            "manifest_digest": manifest.manifest_digest,
            "preprocessing_id": manifest.preprocessing_id,
            "preprocessing_digest": manifest.preprocessing_digest,
            "preprocessing_unchanged": True,
            "canonical_full_dataset": True,
            "cache_revalidation": "deep" if deep else "shallow",
            "train": _split_identity(manifest.train),
            "test": _split_identity(manifest.test),
        },
        "scope": {
            "training_subset": {
                "dataset": base.data.dataset,
                "partition": base.data.partition,
                "samples_per_node": int(base.data.samples_per_node),
                "node_count": brain_node_count(frozen_study),
                "rounds": int(base.rounds),
                "local_epochs": int(base.training.local_epochs),
                "batch_size": int(base.training.batch_size),
            },
            "evaluation_subset": {
                "max_samples": int(base.evaluation.max_samples),
                "batch_size": int(base.evaluation.batch_size),
            },
            "calibration": {
                "episodes": int(calibration.episodes),
                "rounds": 1,
                "design_split_size": int(calibration.design_split_size),
                "p0": float(calibration.p0),
                "candidate_budget": protocol.candidate_budget,
                "seed": int(calibration.seed),
            },
            "selection": {
                "seeds": selection_seeds,
                "schedule_ids": list(selection_manifest.get("schedule_ids", [])),
                "panel": list(selection_manifest.get("panel", [])),
                "cells": len(selection_traces),
            },
            "final": {
                "seeds": final_seeds,
                "schedule_ids": list(final_manifest.get("schedule_ids", [])),
                "panel": list(final_manifest.get("panel", [])),
                "cells": len(final_traces),
                "held_out_schedule_id": final_manifest.get("held_out_schedule_id"),
                "aggregate_count": len(summary.get("aggregates", [])),
            },
            "subset_note": (
                "Training/evaluation counts describe the bounded real-MNIST subset; "
                "the verified cache covers the full canonical 60000/10000 split."
            ),
        },
        "limitations": list(_LIMITATIONS),
        "calibration_derivation_provenance": {
            "data_provenance": "production",
            "matched_calibration_input_digest": match.matched_calibration_input_digest,
            "assumption_version": match.protocol_version,
            "variants": variants,
            "fixed_behavior_limitation": isolation_statement("no_strategic"),
            "note": (
                "Matched-assumption derivation provenance only; the headline does "
                "not establish derivation validity."
            ),
        },
        "defense_usefulness": {
            "note": (
                "The eight-policy outcomes measure defense usefulness on the frozen "
                "subset; equal or poor outcomes are honest evidence."
            ),
            "no_improvement_claim": True,
            "no_strategic_fixed_kappa": True,
        },
        "selection": {
            "manifest_path": str(
                (Path(STUDY_DIRNAME) / "selection" / SELECTION_MANIFEST_FILENAME)
            ),
            "selection_output_digest": selection_manifest.get("selection_output_digest"),
            "study_digest": selection_manifest.get("study_digest"),
            "selection_spec_digest": selection_manifest.get("selection_spec_digest"),
            "shared_input_digest": selection_manifest.get("shared_input_digest"),
            "seeds": selection_seeds,
            "schedule_ids": list(selection_manifest.get("schedule_ids", [])),
            "panel": list(selection_manifest.get("panel", [])),
            "cells": len(selection_traces),
            "traces": [dict(trace) for trace in selection_traces],
        },
        "final": {
            "manifest_path": str(
                (Path(STUDY_DIRNAME) / "final" / FINAL_MANIFEST_FILENAME)
            ),
            "final_output_digest": final_manifest.get("final_output_digest"),
            "study_digest": final_manifest.get("study_digest"),
            "selection_spec_digest": final_manifest.get("selection_spec_digest"),
            "held_out_study_digest": final_manifest.get("held_out_study_digest"),
            "selection_output_digest": final_manifest.get("selection_output_digest"),
            "shared_input_digest": final_manifest.get("shared_input_digest"),
            "seeds": final_seeds,
            "schedule_ids": list(final_manifest.get("schedule_ids", [])),
            "held_out_schedule_id": final_manifest.get("held_out_schedule_id"),
            "panel": list(final_manifest.get("panel", [])),
            "cells": len(final_traces),
            "aggregates": _aggregate_evidence(summary["aggregates"]),
            "traces": [dict(trace) for trace in final_traces],
        },
        "evidence_blocks": evidence_blocks,
        "runtime": {
            "threads": threads,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": platform.system(),
        },
    }
    identity_payload = {
        key: value for key, value in evidence.items() if key not in ("runtime", "evidence_digest")
    }
    evidence["evidence_digest"] = _sha256_json(identity_payload)
    return evidence


def brain_node_count(study: HeldOutStudyConfig) -> int:
    """Best-effort node count from the frozen topology without importing sim."""

    try:
        from .topology import load_topology

        return len(load_topology(study.study.base).nodes())
    except Exception:  # noqa: BLE001 - evidence is best-effort metadata here
        return 0


# --------------------------------------------------------------------------- #
# Atomic publication
# --------------------------------------------------------------------------- #
def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:  # pragma: no cover - best effort
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - best effort
        pass
    finally:
        os.close(fd)


def _publish_evidence(
    root: Path, evidence: Mapping[str, Any], *, current_evidence_digest: str
) -> tuple[str, Path]:
    destination = root / HEADLINE_EVIDENCE_FILENAME
    data = _canonical_bytes(evidence) + b"\n"
    try:
        fd, staging = tempfile.mkstemp(
            prefix=".headline-evidence.", suffix=".tmp", dir=str(root)
        )
    except OSError as exc:
        raise HeadlineVerifyError(f"could not stage headline evidence: {_bounded(exc)}") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staging, destination)
        except FileExistsError:
            try:
                existing = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise HeadlineVerifyError(
                    f"existing headline evidence is unreadable: {_bounded(exc)}"
                ) from exc
            if not isinstance(existing, Mapping):
                raise HeadlineVerifyError(
                    "existing headline evidence is not a JSON object; refusing to overwrite"
                )
            if existing.get("status") != "complete":
                raise HeadlineVerifyError(
                    "existing headline evidence is incomplete; refusing to overwrite stale evidence"
                )
            if existing.get("evidence_digest") != current_evidence_digest:
                raise HeadlineVerifyError(
                    "existing headline evidence does not match the re-verified sources; "
                    "refusing to overwrite stale evidence"
                )
            return "duplicate", destination
        except OSError as exc:
            raise HeadlineVerifyError(
                f"could not publish headline evidence: {_bounded(exc)}"
            ) from exc
        _fsync_dir(root)
        return "written", destination
    finally:
        try:
            os.unlink(staging)
        except FileNotFoundError:
            pass
        except OSError:  # pragma: no cover - best effort cleanup
            pass


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def verify_headline(
    output: str | os.PathLike[str],
    *,
    deps: HeadlineVerifyDeps | None = None,
    write: bool = True,
    deep_cache: bool = True,
) -> HeadlineVerifyResult:
    """Independently verify a completed headline root and publish evidence.

    ``output`` must be an existing headline directory produced by
    :func:`dfl_sim.headline.run_headline`. Every cited source is re-read and
    re-derived; a missing/mutated source raises :class:`HeadlineVerifyError`
    and no evidence marker is written.
    """

    deps = deps or HeadlineVerifyDeps()
    try:
        root = Path(os.path.abspath(os.fspath(output)))
    except TypeError as exc:
        raise HeadlineVerifyError(f"invalid output path: {_bounded(exc)}") from exc
    _require(root.is_dir(), f"headline output directory not found: {root}")

    frozen = _read_json(root / FROZEN_PROTOCOL_FILENAME)
    protocol = _load_protocol(frozen)
    manifest, _actual = _load_provenance(root, frozen, protocol, deps, deep_cache)

    frozen_study = _load_frozen_study(root)
    artifacts = _load_artifacts(root, deps)
    expected_study = _reconstruct_study(protocol, root, artifacts, deps)
    study_digest_value, selection_spec_value, held_out_digest_value = _bind_study(
        expected_study, frozen_study
    )
    match = _matched_artifacts(expected_study, deps)

    selection_manifest, _, selection_traces = _verify_selection(
        root,
        study=expected_study,
        match=match,
        study_digest_value=study_digest_value,
        selection_spec_value=selection_spec_value,
        deps=deps,
    )
    final_manifest, final_dir, final_traces = _verify_final(
        root,
        study=expected_study,
        match=match,
        selection_manifest=selection_manifest,
        study_digest_value=study_digest_value,
        selection_spec_value=selection_spec_value,
        held_out_digest_value=held_out_digest_value,
        deps=deps,
    )
    _verify_embedded_metrics(final_manifest, final_dir, deps)
    summary = _verify_summary(root, final_manifest=final_manifest, final_dir=final_dir, deps=deps)

    selection_cells, final_cells = expected_cell_counts(protocol)
    _require(
        len(selection_traces) == selection_cells,
        f"selection declared {len(selection_traces)} cells, expected {selection_cells}",
    )
    _require(
        len(final_traces) == final_cells,
        f"final declared {len(final_traces)} cells, expected {final_cells}",
    )

    evidence = _build_evidence(
        root=root,
        frozen=frozen,
        protocol=protocol,
        manifest=manifest,
        frozen_study=frozen_study,
        match=match,
        selection_manifest=selection_manifest,
        final_manifest=final_manifest,
        summary=summary,
        selection_traces=selection_traces,
        final_traces=final_traces,
        deep=deep_cache,
    )
    current_digest = str(evidence["evidence_digest"])

    evidence_path: Path | None = None
    status = "dry-run"
    if write:
        status, evidence_path = _publish_evidence(
            root, evidence, current_evidence_digest=current_digest
        )
    return HeadlineVerifyResult(
        output_path=root,
        evidence_path=evidence_path,
        status=status,
        evidence=evidence,
        protocol_digest=str(frozen.get("protocol_digest")),
        selection_cells=selection_cells,
        final_cells=final_cells,
        aggregate_count=len(summary.get("aggregates", [])),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dfl_sim.headline_verify",
        description=(
            "Independently verify a completed real-MNIST headline from its "
            "on-disk artifacts and publish headline-evidence.json only when "
            "every source still matches."
        ),
    )
    parser.add_argument(
        "output",
        help="Headline output root (e.g. runs/m003-mnist-headline).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Verify only; never publish headline-evidence.json.",
    )
    parser.add_argument(
        "--shallow-cache",
        action="store_true",
        help="Compare raw cache sizes only instead of recomputing content hashes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify_headline(
            args.output,
            write=not args.no_write,
            deep_cache=not args.shallow_cache,
        )
    except HeadlineVerifyError as exc:
        print(f"dfl_sim: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"{VERIFY_MARKER}"
        f"output={result.output_path} "
        f"status={result.status} "
        f"selection_cells={result.selection_cells} "
        f"final_cells={result.final_cells} "
        f"aggregates={result.aggregate_count}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
