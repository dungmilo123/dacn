"""Paired baseline-study runner with atomic manifest publication (M003/S01).

This module turns a validated :class:`~dfl_sim.evaluation_config.BaselineStudyConfig`
into exactly one schema-v1 trace per (seed, policy) cell plus one atomic,
versioned manifest. It is the single fairness-enforcing orchestration layer the
S01 slice introduces; S02 extends it with selection/final splits, resumes, and
adaptive schedules.

Fairness invariants enforced here:

* Every cell config is derived from the study's one frozen base experiment by
  overriding only ``seed`` and ``aggregation``. A shared-input digest over the
  base excluding those two paired dimensions is computed for every cell and all
  cells must agree before any execution starts.
* All cells of one seed therefore share the partition, model initialization,
  and attacker; the aggregation policy is the sole experimental variable.
* Each cell runs through the unchanged production
  :func:`~dfl_sim.run.run_from_config`, so the real policy factory, round
  engine, privileged evaluator, and schema-v1 writer are exercised.

Publication invariants:

* Traces are written into a private same-parent staging directory and validated
  with the existing :class:`~dfl_sim.trajectory.TraceReader`.
* The deterministic manifest is written and fsynced only after every cell
  succeeds, then the whole staging directory is atomically renamed into place.
* Any failure removes the staging directory and leaves the requested output
  absent, so an incomplete panel can never publish a success manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from torch.utils.data import Dataset

from .attacker import AdaptiveAttackController, OutgoingController
from .ablation_evaluation import (
    ABLATION_METADATA_KEY,
    AblationEvaluationError,
    AblationMatchSummary,
    ablation_entry_metadata,
    ablation_manifest_metadata,
    preflight_ablation_study,
)
from .attack_schedule import (
    ScheduleError,
    build_schedule,
    derive_schedule_seed,
    realized_kappa_sequence,
    schedule_digest,
)
from .bayesian_artifact import BayesianArtifactError, load_bayesian_artifact
from .config import AggregationConfig, ExperimentConfig
from .evaluation_config import (
    EXTENDED_POLICY_IDS,
    REQUIRED_POLICY_IDS,
    BaselineStudyConfig,
    HeldOutStudyConfig,
    NamedSchedule,
    PanelPolicy,
    SelectionSpec,
    held_out_study_digest,
    hyperparameter_digest,
    load_study_config,
    policy_identity,
    selected_hyperparameters,
    selection_projection,
    selection_spec_digest,
    study_digest,
)
from .run import DatasetLoader, run_from_config
from .trajectory import SCHEMA_VERSION, TraceReader, TrajectoryError

__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "MANIFEST_SCHEMA_VERSION",
    "FINAL_MANIFEST_FILENAME",
    "FINAL_MANIFEST_SCHEMA",
    "FINAL_MANIFEST_SCHEMA_VERSION",
    "ROBUSTNESS_QUANTILE_METHOD",
    "ROBUSTNESS_STD_METHOD",
    "ROBUSTNESS_SUMMARY_FILENAME",
    "ROBUSTNESS_SUMMARY_SCHEMA",
    "ROBUSTNESS_SUMMARY_SCHEMA_VERSION",
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_SCHEMA_VERSION",
    "SELECTION_MANIFEST_FILENAME",
    "SELECTION_MANIFEST_SCHEMA",
    "SELECTION_MANIFEST_SCHEMA_VERSION",
    "TRACES_DIRNAME",
    "BaselineCell",
    "BaselineStudyResult",
    "EvaluationError",
    "FinalResult",
    "HeldOutStudyResult",
    "SelectionCell",
    "SelectionResult",
    "build_cell_config",
    "build_robustness_summary",
    "compute_shared_input_digest",
    "final_output_digest",
    "load_study_config",
    "read_final_round_metrics",
    "run_baseline_study",
    "run_final_phase",
    "run_held_out_study",
    "run_selection_phase",
    "selection_output_digest",
    "summarize_held_out_study",
    "validate_shared_inputs",
]

#: Manifest file name written at the study output root.
MANIFEST_FILENAME = "baseline-manifest.json"

#: Subdirectory holding one trace per (policy, seed) cell.
TRACES_DIRNAME = "traces"

#: Versioned manifest identity.
MANIFEST_SCHEMA = "dfl_baseline_study_manifest"
MANIFEST_SCHEMA_VERSION = 1

#: Versioned selection-manifest identity (M003/S02).
SELECTION_MANIFEST_FILENAME = "selection-manifest.json"
SELECTION_MANIFEST_SCHEMA = "dfl_held_out_selection"
SELECTION_MANIFEST_SCHEMA_VERSION = 1

#: Versioned final-manifest identity (M003/S02).
FINAL_MANIFEST_FILENAME = "final-manifest.json"
FINAL_MANIFEST_SCHEMA = "dfl_held_out_study_manifest"
FINAL_MANIFEST_SCHEMA_VERSION = 1

#: Versioned paired-metrics summary identity (M003/S02/T06). This is the
#: machine-readable paired-per-cell evidence that the S05 canonical report
#: consumes; it carries no figures and no claim/evidence-class framing.
ROBUSTNESS_SUMMARY_FILENAME = "robustness-summary.json"
ROBUSTNESS_SUMMARY_SCHEMA = "dfl_robustness_summary"
ROBUSTNESS_SUMMARY_SCHEMA_VERSION = 1

#: Pinned spread method: population standard deviation (``ddof == 0``).
ROBUSTNESS_STD_METHOD = "population"

#: Pinned IQR quantile method. ``linear`` is the R-7 / numpy default
#: ``method="linear"`` interpolation, recorded so downstream readers never
#: have to guess how the interquartile range was computed.
ROBUSTNESS_QUANTILE_METHOD = "linear"

#: Private same-parent staging prefix; the final directory is renamed into place.
_STAGING_PREFIX = ".baseline_study.staging-"

#: Private staging prefix for the selection-phase output.
_SELECTION_STAGING_PREFIX = ".selection.staging-"

#: Private staging prefix for the final-phase output.
_FINAL_STAGING_PREFIX = ".final.staging-"

#: Private staging prefix for the whole two-phase held-out study root.
_HELD_OUT_STAGING_PREFIX = ".held_out_study.staging-"

#: Per-cell resume checkpoint file written into each phase staging directory.
#: Canonical JSON Lines: one completed ``(seed, schedule, policy)`` cell per
#: line carrying its relative trace path and the schedule/config/artifact/study
#: digests needed to prove a reused trace belongs to this exact study. Because
#: a successful phase renames staging into place, a published output (or a
#: retained private staging directory) is itself a valid resume checkpoint.
CHECKPOINT_FILENAME = "cells.jsonl"
CHECKPOINT_SCHEMA = "dfl_held_out_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1

#: Canonical panel ordering used for deterministic manifest entry order. The
#: first six indices are the legacy S01 identities, so extending this map to
#: the opt-in eight-policy panel leaves six-policy output ordering unchanged.
_POLICY_ORDER: dict[str, int] = {
    policy_id: index for index, policy_id in enumerate(EXTENDED_POLICY_IDS)
}

#: A validated study, or a path to a study YAML spec.
StudyInput = Union[BaselineStudyConfig, str, "os.PathLike[str]"]

#: Optional per-cell progress hook: ``(seed, policy_id) -> None``.
ProgressHook = Callable[[int, str], None]

#: Trace validation seam (defaults to the production strict reader).
TraceValidator = Callable[[str | os.PathLike], object]

#: Optional per-cell selection/final progress hook:
#: ``(seed, schedule_id, policy_id)``.
SelectionProgressHook = Callable[[int, str, str], None]

#: Optional two-phase progress hook: ``(phase, seed, schedule_id, policy_id)``
#: where ``phase`` is ``"selection"`` or ``"final"``.
HeldOutProgressHook = Callable[[str, int, str, str], None]


class EvaluationError(RuntimeError):
    """Raised when a baseline study cannot be prepared, run, or published."""


@dataclass(frozen=True, slots=True)
class BaselineCell:
    """Outcome of one successfully published (seed, policy) study cell."""

    seed: int
    policy_id: str
    trace_path: Path
    relative_trace_path: str
    manifest_entry: Mapping[str, object]

    @property
    def pairing_key(self) -> str:
        """The stable pairing key for this seed/policy cell."""

        return f"{self.policy_id}:seed-{self.seed}"


@dataclass(frozen=True, slots=True)
class BaselineStudyResult:
    """Immutable outcome of a completed baseline study publication."""

    study: BaselineStudyConfig
    output_path: Path
    manifest_path: Path
    manifest: Mapping[str, object]
    cells: tuple[BaselineCell, ...] = field(default_factory=tuple)
    ablation: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class SelectionCell:
    """Outcome of one successfully published (seed, schedule, policy) cell."""

    seed: int
    policy_id: str
    schedule_id: str
    trace_path: Path
    relative_trace_path: str
    manifest_entry: Mapping[str, object]
    realized_kappa_sequence: tuple[float, ...] = field(default_factory=tuple)

    @property
    def pairing_key(self) -> str:
        """The stable pairing key for this (policy, seed, schedule) cell."""

        return f"{self.policy_id}:seed-{self.seed}:{self.schedule_id}"


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Immutable outcome of a completed frozen selection publication."""

    selection: SelectionSpec
    output_path: Path
    manifest_path: Path
    manifest: Mapping[str, object]
    cells: tuple[SelectionCell, ...] = field(default_factory=tuple)
    ablation: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class FinalResult:
    """Immutable outcome of a completed final-phase publication."""

    study: HeldOutStudyConfig
    selection_output_path: Path
    output_path: Path
    manifest_path: Path
    manifest: Mapping[str, object]
    cells: tuple[SelectionCell, ...] = field(default_factory=tuple)
    ablation: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class HeldOutStudyResult:
    """Immutable outcome of a completed two-phase held-out study publication.

    The study root at :attr:`output_path` holds the frozen selection output
    under ``selection/`` and the final output under ``final/``; the two
    wrapped phase results carry their re-keyed absolute trace paths so callers
    never observe the (removed) private staging location.
    """

    spec: HeldOutStudyConfig
    output_path: Path
    selection_result: SelectionResult
    final_result: FinalResult
    ablation: Mapping[str, object] | None = None

    @property
    def selection_path(self) -> Path:
        """The published ``selection/`` directory."""

        return self.selection_result.output_path

    @property
    def final_path(self) -> Path:
        """The published ``final/`` directory."""

        return self.final_result.output_path

    @property
    def selection_manifest_path(self) -> Path:
        """The frozen selection manifest path."""

        return self.selection_result.manifest_path

    @property
    def final_manifest_path(self) -> Path:
        """The final manifest path."""

        return self.final_result.manifest_path


# --------------------------------------------------------------------------- #
# Derivation, pairing, and identity
# --------------------------------------------------------------------------- #
def build_cell_config(
    base: ExperimentConfig, aggregation: AggregationConfig, seed: int
) -> ExperimentConfig:
    """Derive one cell config from the frozen base.

    Only ``seed`` and ``aggregation`` may differ from the base; both are
    already-validated typed values, so ``model_copy`` is safe and cannot widen
    an invariant the base validator already enforced.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvaluationError(f"seed must be an integer, got {seed!r}")
    return base.model_copy(update={"seed": int(seed), "aggregation": aggregation})


def compute_shared_input_digest(config: ExperimentConfig) -> str:
    """SHA-256 over every config dimension except the paired seed/aggregation.

    The digest captures the shared experiment inputs (data, training, topology,
    model, attack, evaluation, trace) so paired cells can be proven to differ
    only in the two intended dimensions.
    """

    payload = config.model_dump(mode="json")
    payload.pop("seed", None)
    payload.pop("aggregation", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def validate_shared_inputs(
    configs: Sequence[ExperimentConfig],
) -> str:
    """Assert every cell config shares the same non-paired inputs.

    Returns the single shared-input digest. Raises :class:`EvaluationError` when
    any config differs in a dimension other than ``seed``/``aggregation``.
    """

    digests = {compute_shared_input_digest(config) for config in configs}
    if len(digests) != 1:
        raise EvaluationError(
            "paired cell configs differ in a non-paired dimension; expected a "
            "single shared-input digest, got " + ", ".join(sorted(digests))
        )
    return digests.pop()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _redacted_cell_digest(config: ExperimentConfig) -> str:
    """SHA-256 of a cell config with the environment-specific artifact path scrubbed."""

    payload = config.model_dump(mode="json")
    aggregation = payload.get("aggregation")
    if isinstance(aggregation, dict) and "artifact_dir" in aggregation:
        aggregation["artifact_dir"] = "<artifact>"
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _relative_trace_path(policy_id: str, seed: int) -> str:
    return f"{TRACES_DIRNAME}/{policy_id}/seed-{int(seed)}"


def _canonical_panel(study: BaselineStudyConfig) -> tuple[PanelPolicy, ...]:
    return study.ordered_panel()


def _canonical_cells(
    study: BaselineStudyConfig,
) -> tuple[tuple[int, PanelPolicy], ...]:
    """Every (seed, policy) cell in deterministic (seed, canonical panel) order."""

    cells: list[tuple[int, PanelPolicy]] = []
    for seed in sorted(int(seed) for seed in study.seeds):
        for policy in _canonical_panel(study):
            cells.append((seed, policy))
    return tuple(cells)


# --------------------------------------------------------------------------- #
# Output destination safety (Q3: untrusted path)
# --------------------------------------------------------------------------- #
def _validate_output_destination(
    output: str | os.PathLike[str],
    *,
    study_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Return an absolute output path after all pre-publication safety checks."""

    text = os.fspath(output)
    if not text:
        raise EvaluationError("output path is empty")
    absolute = Path(os.path.abspath(text))

    if absolute.name in ("", ".", ".."):
        raise EvaluationError(f"invalid output destination: {text!r}")

    parent = absolute.parent
    try:
        parent_info = os.lstat(parent)
    except FileNotFoundError as exc:
        raise EvaluationError(
            f"output parent does not exist: {parent!r}"
        ) from exc
    except OSError as exc:
        raise EvaluationError(
            f"output parent is unreadable: {parent!r}"
        ) from exc
    if stat.S_ISLNK(parent_info.st_mode):
        raise EvaluationError(f"output parent must not be a symlink: {parent!r}")
    if not stat.S_ISDIR(parent_info.st_mode):
        raise EvaluationError(f"output parent is not a directory: {parent!r}")
    if os.path.lexists(absolute):
        raise EvaluationError(f"output already exists: {absolute!r}")

    if study_path is not None:
        try:
            if os.path.realpath(os.fspath(study_path)) == os.path.realpath(
                absolute
            ):
                raise EvaluationError("output aliases the study config path")
        except OSError:  # pragma: no cover - best-effort alias check
            pass
    return absolute


# --------------------------------------------------------------------------- #
# Data resolution
# --------------------------------------------------------------------------- #
def _resolve_data_sources(
    study: BaselineStudyConfig,
    *,
    dataset: Dataset | None,
    dataset_loader: DatasetLoader | None,
    evaluation_dataset: Dataset | None,
    evaluation_dataset_loader: DatasetLoader | None,
) -> None:
    if dataset is not None and dataset_loader is not None:
        raise EvaluationError("provide either dataset or dataset_loader, not both")
    if evaluation_dataset is not None and evaluation_dataset_loader is not None:
        raise EvaluationError(
            "provide either evaluation_dataset or evaluation_dataset_loader, "
            "not both"
        )
    if study.data_provenance == "injected":
        if dataset is None:
            raise EvaluationError(
                "data_provenance 'injected' requires an explicit in-memory "
                "dataset; no dataset was supplied"
            )
        if evaluation_dataset is None:
            raise EvaluationError(
                "data_provenance 'injected' requires an explicit in-memory "
                "evaluation_dataset; no evaluation dataset was supplied"
            )


# --------------------------------------------------------------------------- #
# Manifest construction
# --------------------------------------------------------------------------- #
def _artifact_digest(
    aggregation: AggregationConfig, cache: dict[str, str]
) -> str | None:
    """The Bayesian artifact aggregate digest, or ``None`` for baseline policies."""

    artifact_dir = getattr(aggregation, "artifact_dir", None)
    if artifact_dir is None:
        return None
    if artifact_dir not in cache:
        try:
            artifact = load_bayesian_artifact(artifact_dir)
        except BayesianArtifactError as exc:
            raise EvaluationError(
                f"could not load bayesian artifact {artifact_dir!r}: {exc}"
            ) from exc
        cache[artifact_dir] = artifact.manifest.aggregate_digest
    return cache[artifact_dir]


def _resolve_ablation_match(
    study: BaselineStudyConfig,
    *,
    provided: AblationMatchSummary | None = None,
) -> AblationMatchSummary | None:
    """Return the validated matched triple for an opt-in study, else ``None``.

    A legacy six-policy study (no ablation contract) never triggers the gate.
    When ``provided`` is supplied the caller already validated this exact study
    within the current invocation, so the immutable summary is reused instead
    of reloading the three artifacts (Q6). Every public entry point revalidates
    the current pinned artifacts when no summary is provided, so a stale cached
    path or a previously frozen selection is never trusted.
    """

    if provided is not None:
        return provided
    if study.ablations is None:
        return None
    return preflight_ablation_study(study)


def _manifest_entry(
    *,
    study: BaselineStudyConfig,
    policy: PanelPolicy,
    seed: int,
    config: ExperimentConfig,
    artifact_digest: str | None,
    match: AblationMatchSummary | None = None,
) -> dict:
    relative = _relative_trace_path(policy.id, seed)
    entry = {
        "pairing_key": f"{policy.id}:seed-{seed}",
        "seed": int(seed),
        "policy_id": policy.id,
        "policy_kind": policy.aggregation.kind,
        "implementation": policy.implementation,
        "citation": policy.citation.model_dump(mode="json"),
        "policy_identity": policy_identity(policy.aggregation),
        "hyperparameters": selected_hyperparameters(policy.aggregation),
        "hyperparameter_digest": hyperparameter_digest(policy.aggregation),
        "tuning": policy.tuning.model_dump(mode="json"),
        "data_provenance": study.data_provenance,
        "config_digest": _redacted_cell_digest(config),
        "artifact_digest": artifact_digest,
        "status": "complete",
        "trace_path": relative,
    }
    ablation = ablation_entry_metadata(match, policy.id)
    if ablation is not None:
        entry[ABLATION_METADATA_KEY] = ablation
    return entry


def _build_manifest(
    *,
    study: BaselineStudyConfig,
    shared_input_digest: str,
    cells: tuple[BaselineCell, ...],
    match: AblationMatchSummary | None = None,
) -> dict:
    entries = sorted(
        (dict(cell.manifest_entry) for cell in cells),
        key=lambda entry: (
            int(entry["seed"]),
            _POLICY_ORDER[str(entry["policy_id"])],
        ),
    )
    manifest: dict = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "study_id": study.study_id,
        "data_provenance": study.data_provenance,
        "study_digest": study_digest(study),
        "shared_input_digest": shared_input_digest,
        "candidate_budget": study.candidate_budget,
        "selection_seed": study.selection_seed,
        "selection_provenance": study.selection_provenance,
        "panel": list(study.policy_ids()),
        "seeds": sorted(int(seed) for seed in study.seeds),
        "entries": entries,
        "status": "complete",
    }
    ablation = ablation_manifest_metadata(match)
    if ablation is not None:
        manifest[ABLATION_METADATA_KEY] = ablation
    return manifest


def manifest_bytes(manifest: Mapping[str, object]) -> bytes:
    """Canonical, deterministic bytes for a study manifest."""

    return (
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _resolve_study(
    study: StudyInput,
) -> tuple[BaselineStudyConfig, Path | None]:
    if isinstance(study, BaselineStudyConfig):
        return study, None
    source_path = Path(study)
    return load_study_config(source_path), source_path


def _write_bytes_synced(path: Path, payload: bytes) -> None:
    try:
        with open(path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise EvaluationError(f"could not write {path}: {exc}") from exc


def _fsync_dir(path: Path) -> None:
    """Best-effort directory fsync; directory durability is platform-dependent."""

    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _prepare_staging(destination: Path, *, prefix: str) -> tuple[Path, Path]:
    """Create a private same-parent staging directory for an atomic publish."""

    try:
        staging = Path(tempfile.mkdtemp(prefix=prefix, dir=str(destination.parent)))
        os.chmod(staging, 0o700)
        staging_root = staging.resolve()
    except OSError as exc:
        raise EvaluationError(
            f"could not create study staging directory under "
            f"{destination.parent}: {exc}"
        ) from exc
    return staging, staging_root


def _execute_cell_run(
    config: ExperimentConfig,
    trace_dir: Path,
    *,
    staging_root: Path,
    label: str,
    dataset: Dataset | None,
    dataset_loader: DatasetLoader | None,
    evaluation_dataset: Dataset | None,
    evaluation_dataset_loader: DatasetLoader | None,
    controller: OutgoingController | None,
) -> None:
    """Run one cell through the production composition and commit its trace.

    ``label`` is the already-formatted bounded cell identity (for example
    ``"policy 'balance' seed 11"``) used verbatim in failure messages, so the
    shared helper reproduces each caller's exact diagnostic wording.
    """

    trace_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved_trace = trace_dir.resolve()
    except OSError as exc:  # pragma: no cover - defensive
        raise EvaluationError(
            f"could not resolve trace destination for {label}: {exc}"
        ) from exc
    if not resolved_trace.is_relative_to(staging_root):
        raise EvaluationError(
            f"trace destination for {label} escapes the staging root"
        )

    try:
        run_from_config(
            config,
            trace_path=trace_dir,
            dataset=dataset,
            dataset_loader=dataset_loader,
            evaluation_dataset=evaluation_dataset,
            evaluation_dataset_loader=evaluation_dataset_loader,
            controller=controller,
        )
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"{label} failed: {exc}") from exc


def _validate_cell_trace(
    validator: TraceValidator,
    trace_dir: Path,
    *,
    label: str,
    measure: bool = False,
) -> dict | None:
    """Validate one committed schema-v1 trace; ``label`` supplies exact wording.

    With ``measure=True`` the validated reader's events are also reduced to the
    final evaluated round's per-node metrics (used by the final phase to embed
    a compact ``metrics`` block per manifest entry). The reduction runs on the
    reader that already validated the trace, so no trace is opened twice.
    """

    try:
        with validator(trace_dir) as reader:
            schema = getattr(reader, "schema_version", SCHEMA_VERSION)
            if schema != SCHEMA_VERSION:
                raise EvaluationError(
                    f"{label} produced schema version {schema!r}, expected "
                    f"{SCHEMA_VERSION}"
                )
            if not measure:
                return None
            events = getattr(reader, "events", None)
            if events is None:
                raise EvaluationError(
                    f"{label} trace reader exposes no events for metrics"
                )
            return _final_round_metrics_from_events(tuple(events), label=label)
    except EvaluationError:
        raise
    except (TrajectoryError, OSError) as exc:
        raise EvaluationError(
            f"{label} produced an invalid trace: {exc}"
        ) from exc


def _publish_staging(
    staging: Path,
    destination: Path,
    manifest: Mapping[str, object],
    manifest_filename: str,
) -> None:
    """Write the manifest into staging, fsync, and atomically rename into place."""

    _write_bytes_synced(staging / manifest_filename, manifest_bytes(manifest))
    _fsync_dir(staging)
    try:
        os.rename(staging, destination)
    except OSError as exc:
        raise EvaluationError(
            f"could not publish study output {destination}: {exc}"
        ) from exc
    _fsync_dir(destination.parent)


# --------------------------------------------------------------------------- #
# Bounded manifest-matched resume (M003/S02/T07)
# --------------------------------------------------------------------------- #
def _checkpoint_pairing_key(policy_id: str, seed: int, schedule_id: str) -> str:
    return f"{policy_id}:seed-{int(seed)}:{schedule_id}"


def _locate_checkpoint(
    resume_from: str | os.PathLike[str], *, phase: str
) -> tuple[Path, Path]:
    """Locate the checkpoint file for ``phase`` under ``resume_from``.

    Accepts either a study root (``<root>/selection`` or ``<root>/final``) or a
    flat phase output directory holding ``cells.jsonl`` directly.
    """

    base = Path(os.path.abspath(os.fspath(resume_from)))
    subdir = "selection" if phase == "selection" else "final"
    for candidate in (
        base / subdir / CHECKPOINT_FILENAME,
        base / CHECKPOINT_FILENAME,
    ):
        if candidate.is_file():
            return candidate.parent, candidate
    raise EvaluationError(
        f"resume checkpoint not found under {base}: expected "
        f"{subdir}/{CHECKPOINT_FILENAME} or {CHECKPOINT_FILENAME}"
    )


def _load_resume_checkpoint(
    resume_from: str | os.PathLike[str],
    *,
    phase: str,
    expected: Mapping[str, object],
) -> tuple[Path, dict[str, dict]]:
    """Read and validate a prior per-cell checkpoint; never trust it blindly.

    Every line must declare the checkpoint schema/version and the requested
    phase, and every study-level digest in ``expected`` must match exactly. A
    mismatch abandons reuse and fails closed naming only the bounded digests,
    so a checkpoint from another study/split/spec can never be mixed in.
    """

    directory, path = _locate_checkpoint(resume_from, phase=phase)
    records: dict[str, dict] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(
                        f"resume checkpoint {path} line {lineno} is not "
                        "valid JSON"
                    ) from exc
                if not isinstance(record, dict):
                    raise EvaluationError(
                        f"resume checkpoint {path} line {lineno} is not an "
                        "object"
                    )
                if (
                    record.get("schema") != CHECKPOINT_SCHEMA
                    or record.get("schema_version")
                    != CHECKPOINT_SCHEMA_VERSION
                ):
                    raise EvaluationError(
                        f"resume checkpoint {path} line {lineno} has an "
                        "unrecognized schema"
                    )
                if record.get("phase") != phase:
                    raise EvaluationError(
                        f"resume checkpoint {path} line {lineno} is a "
                        f"{record.get('phase')!r} checkpoint, expected "
                        f"{phase!r}"
                    )
                for key, value in expected.items():
                    observed = record.get(key)
                    if observed != value:
                        raise EvaluationError(
                            f"resume checkpoint {key} mismatch: checkpoint "
                            f"{observed} does not match current {value}"
                        )
                pairing_key = str(record.get("pairing_key"))
                if pairing_key in records:
                    raise EvaluationError(
                        f"resume checkpoint {path} repeats cell "
                        f"{pairing_key!r}"
                    )
                records[pairing_key] = record
    except OSError as exc:
        raise EvaluationError(
            f"could not read resume checkpoint {path}: {exc}"
        ) from exc
    return directory, records


def _open_checkpoint(path: Path):
    try:
        return open(path, "wb")
    except OSError as exc:
        raise EvaluationError(
            f"could not create resume checkpoint {path}: {exc}"
        ) from exc


def _append_checkpoint(handle, record: Mapping[str, object]) -> None:
    try:
        handle.write(_canonical_bytes(record) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    except OSError as exc:
        raise EvaluationError(
            f"could not write resume checkpoint: {exc}"
        ) from exc


def _checkpoint_record(
    *,
    phase: str,
    study: BaselineStudyConfig,
    selection_digest: str,
    shared_input_digest: str,
    policy: PanelPolicy,
    seed: int,
    schedule: NamedSchedule,
    config: ExperimentConfig,
    artifact_digest: str | None,
    relative: str,
    reused: bool,
    match: AblationMatchSummary | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict:
    """One canonical per-cell checkpoint line.

    For an opt-in ablation study the checkpoint additionally binds the matched
    calibration-input digest and assumption version, so a resume can prove the
    reused trace's artifact set is the one revalidated in this invocation.
    """

    record: dict = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "phase": phase,
        "pairing_key": _checkpoint_pairing_key(policy.id, seed, schedule.id),
        "seed": int(seed),
        "policy_id": policy.id,
        "schedule_id": schedule.id,
        "trace_path": relative,
        "status": "complete",
        "schedule_digest": schedule_digest(schedule.schedule),
        "config_digest": _redacted_cell_digest(config),
        "artifact_digest": artifact_digest,
        "study_digest": study_digest(study),
        "selection_spec_digest": selection_digest,
        "shared_input_digest": shared_input_digest,
        "reused": bool(reused),
    }
    if match is not None:
        record["ablation_match_digest"] = match.matched_calibration_input_digest
        record["ablation_protocol_version"] = match.protocol_version
    if extra is not None:
        record.update(extra)
    return record


def _checkpoint_record_reusable(
    *,
    record: Mapping[str, object] | None,
    expected_schedule_digest: str,
    expected_config_digest: str,
    expected_artifact_digest: str | None,
    label: str,
) -> bool:
    """Decide whether a checkpoint cell may be reused, or fail closed.

    A schedule or artifact digest mismatch means the checkpoint's evidence is
    not from this exact study, so it fails closed rather than mixing evidence.
    A missing record or a redacted-config mismatch simply means the cell is
    recomputed from scratch.
    """

    if record is None:
        return False
    observed_schedule = record.get("schedule_digest")
    if observed_schedule != expected_schedule_digest:
        raise EvaluationError(
            f"{label} resume schedule_digest mismatch: checkpoint "
            f"{observed_schedule} does not match current "
            f"{expected_schedule_digest}"
        )
    observed_artifact = record.get("artifact_digest")
    if observed_artifact != expected_artifact_digest:
        raise EvaluationError(
            f"{label} resume artifact_digest mismatch: checkpoint "
            f"{observed_artifact} does not match current "
            f"{expected_artifact_digest}"
        )
    if record.get("config_digest") != expected_config_digest:
        return False
    if record.get("status") != "complete":
        return False
    return True


def _checkpoint_trace_source(
    directory: Path, record: Mapping[str, object], *, label: str
) -> Path | None:
    """Resolve a recorded relative trace path under the checkpoint root."""

    relative = record.get("trace_path")
    if not isinstance(relative, str) or not relative:
        return None
    if os.path.isabs(relative):
        raise EvaluationError(f"{label} resume trace_path must be relative")
    source = directory / relative
    try:
        resolved = source.resolve()
        root = directory.resolve()
    except OSError as exc:
        raise EvaluationError(
            f"{label} resume trace path is unresolvable: {exc}"
        ) from exc
    if not resolved.is_relative_to(root):
        raise EvaluationError(
            f"{label} resume trace_path escapes the checkpoint directory"
        )
    if not resolved.is_dir():
        return None
    return resolved


def _copy_checkpoint_trace(
    source: Path, trace_dir: Path, *, staging_root: Path, label: str
) -> None:
    """Copy one reused trace into staging at its canonical relative path."""

    trace_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved = trace_dir.resolve()
    except OSError as exc:  # pragma: no cover - defensive
        raise EvaluationError(
            f"could not resolve trace destination for {label}: {exc}"
        ) from exc
    if not resolved.is_relative_to(staging_root):
        raise EvaluationError(
            f"trace destination for {label} escapes the staging root"
        )
    try:
        shutil.copytree(source, trace_dir)
    except OSError as exc:
        raise EvaluationError(
            f"could not reuse checkpoint trace for {label}: {exc}"
        ) from exc


def _resume_cell(
    *,
    checkpoint_directory: Path | None,
    checkpoint_records: Mapping[str, Mapping[str, object]],
    policy: PanelPolicy,
    seed: int,
    schedule: NamedSchedule,
    config: ExperimentConfig,
    artifact_digest: str | None,
    relative: str,
    trace_dir: Path,
    staging_root: Path,
    label: str,
    validator: TraceValidator,
    measure: bool,
) -> tuple[bool, dict | None]:
    """Try to reuse a manifest-matched checkpoint cell.

    Returns ``(reused, metrics)``. A digest-mismatched or unreadable reuse
    candidate is either rejected up front (schedule/artifact mismatch fails
    closed) or silently recomputed (missing trace, unreadable trace, or a
    redacted-config mismatch), so stale evidence is never mixed into a
    published study.
    """

    if checkpoint_directory is None:
        return False, None
    record = checkpoint_records.get(
        _checkpoint_pairing_key(policy.id, seed, schedule.id)
    )
    if not _checkpoint_record_reusable(
        record=record,
        expected_schedule_digest=schedule_digest(schedule.schedule),
        expected_config_digest=_redacted_cell_digest(config),
        expected_artifact_digest=artifact_digest,
        label=label,
    ):
        return False, None
    assert record is not None  # narrowed by _checkpoint_record_reusable
    source = _checkpoint_trace_source(
        checkpoint_directory, record, label=label
    )
    if source is None:
        return False, None
    try:
        _copy_checkpoint_trace(
            source, trace_dir, staging_root=staging_root, label=label
        )
        metrics = _validate_cell_trace(
            validator, trace_dir, label=label, measure=measure
        )
    except EvaluationError:
        shutil.rmtree(trace_dir, ignore_errors=True)
        return False, None
    return True, metrics


def run_baseline_study(
    study: StudyInput,
    output: str | os.PathLike[str],
    *,
    dataset: Dataset | None = None,
    dataset_loader: DatasetLoader | None = None,
    evaluation_dataset: Dataset | None = None,
    evaluation_dataset_loader: DatasetLoader | None = None,
    progress: ProgressHook | None = None,
    trace_validator: TraceValidator | None = None,
) -> BaselineStudyResult:
    """Run every (seed, policy) cell and atomically publish a baseline manifest.

    ``study`` is a validated :class:`BaselineStudyConfig` or a path to a strict
    YAML/JSON study spec. ``output`` is the study destination directory, which
    must not already exist and must not alias the study config; it is created by
    an atomic rename of a private same-parent staging directory only after every
    cell has run and every trace has been validated.

    For ``data_provenance == "injected"`` the caller must supply an in-memory
    ``dataset`` and ``evaluation_dataset`` so no network access occurs; for
    ``real_mnist`` the production loaders are used unless overridden. All cells
    within a seed share the same dataset object, so they share partitions,
    initialization, and attacker and differ only in ``aggregation``.
    """

    resolved, source_path = _resolve_study(study)
    destination = _validate_output_destination(output, study_path=source_path)
    _resolve_data_sources(
        resolved,
        dataset=dataset,
        dataset_loader=dataset_loader,
        evaluation_dataset=evaluation_dataset,
        evaluation_dataset_loader=evaluation_dataset_loader,
    )
    validator = TraceReader if trace_validator is None else trace_validator
    ablation_match = _resolve_ablation_match(resolved)

    # Derive every cell config up front and prove the paired-input invariant
    # before any expensive training or evaluation work begins.
    cells = _canonical_cells(resolved)
    cell_configs = {
        (seed, policy.id): build_cell_config(
            resolved.base, policy.aggregation, seed
        )
        for seed, policy in cells
    }
    shared_input_digest = validate_shared_inputs(list(cell_configs.values()))

    try:
        staging, staging_root = _prepare_staging(
            destination, prefix=_STAGING_PREFIX
        )
    except EvaluationError:
        raise
    published = False
    try:
        artifact_cache: dict[str, str] = {}
        published_cells: list[BaselineCell] = []
        for seed, policy in cells:
            if progress is not None:
                progress(int(seed), policy.id)
            config = cell_configs[(seed, policy.id)]
            relative = _relative_trace_path(policy.id, seed)
            trace_dir = staging / relative
            label = f"policy {policy.id!r} seed {seed}"
            _execute_cell_run(
                config,
                trace_dir,
                staging_root=staging_root,
                label=label,
                dataset=dataset,
                dataset_loader=dataset_loader,
                evaluation_dataset=evaluation_dataset,
                evaluation_dataset_loader=evaluation_dataset_loader,
                controller=None,
            )
            _validate_cell_trace(validator, trace_dir, label=label)

            artifact_digest = _artifact_digest(policy.aggregation, artifact_cache)
            entry = _manifest_entry(
                study=resolved,
                policy=policy,
                seed=seed,
                config=config,
                artifact_digest=artifact_digest,
                match=ablation_match,
            )
            published_cells.append(
                BaselineCell(
                    seed=int(seed),
                    policy_id=policy.id,
                    trace_path=trace_dir,
                    relative_trace_path=relative,
                    manifest_entry=entry,
                )
            )

        manifest = _build_manifest(
            study=resolved,
            shared_input_digest=shared_input_digest,
            cells=tuple(published_cells),
            match=ablation_match,
        )
        _publish_staging(staging, destination, manifest, MANIFEST_FILENAME)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)

    # Re-key the published cells to their final absolute trace paths.
    final_cells = tuple(
        BaselineCell(
            seed=cell.seed,
            policy_id=cell.policy_id,
            trace_path=destination / cell.relative_trace_path,
            relative_trace_path=cell.relative_trace_path,
            manifest_entry=cell.manifest_entry,
        )
        for cell in published_cells
    )
    return BaselineStudyResult(
        study=resolved,
        output_path=destination,
        manifest_path=destination / MANIFEST_FILENAME,
        manifest=manifest,
        cells=final_cells,
        ablation=ablation_manifest_metadata(ablation_match),
    )


# --------------------------------------------------------------------------- #
# Selection-phase runner (M003/S02)
# --------------------------------------------------------------------------- #
def _selection_relative_trace_path(
    policy_id: str, seed: int, schedule_id: str
) -> str:
    """Deterministic trace path for one (schedule, policy, seed) selection cell."""

    return f"{TRACES_DIRNAME}/{schedule_id}/{policy_id}/seed-{int(seed)}"


def _canonical_selection_cells(
    selection: SelectionSpec,
) -> tuple[tuple[int, NamedSchedule, PanelPolicy], ...]:
    """Every (seed, schedule, policy) cell in deterministic order.

    Seeds are ascending, schedules are sorted by id, and policies follow the
    canonical panel order, so the traversal and the resulting manifest are
    independent of the caller's schedule declaration order.
    """

    schedules = tuple(
        sorted(selection.known_schedules, key=lambda entry: entry.id)
    )
    cells: list[tuple[int, NamedSchedule, PanelPolicy]] = []
    for seed in sorted(int(seed) for seed in selection.study.seeds):
        for schedule in schedules:
            for policy in _canonical_panel(selection.study):
                cells.append((seed, schedule, policy))
    return tuple(cells)


def _selection_manifest_entry(
    *,
    study: BaselineStudyConfig,
    policy: PanelPolicy,
    seed: int,
    schedule: NamedSchedule,
    config: ExperimentConfig,
    controller: AdaptiveAttackController,
    artifact_digest: str | None,
    relative: str,
    reused: bool = False,
    match: AblationMatchSummary | None = None,
) -> dict:
    """One per-cell selection manifest entry.

    ``realized_kappa_sequence`` is recomputed from the controller's own schedule
    seed, so the recorded per-round attack is exactly what every policy in the
    cell realized through the :class:`OutgoingController` seam. ``reused`` is
    ``True`` only when the cell's trace came from a manifest-matched resume
    checkpoint instead of being recomputed in this run.
    """

    entry = {
        "pairing_key": f"{policy.id}:seed-{seed}:{schedule.id}",
        "seed": int(seed),
        "policy_id": policy.id,
        "policy_kind": policy.aggregation.kind,
        "schedule_id": schedule.id,
        "schedule": schedule.schedule.model_dump(mode="json"),
        "schedule_digest": schedule_digest(schedule.schedule),
        "realized_kappa_sequence": list(
            realized_kappa_sequence(
                schedule.schedule, study.base.rounds, controller.schedule_seed
            )
        ),
        "config_digest": _redacted_cell_digest(config),
        "artifact_digest": artifact_digest,
        "status": "complete",
        "reused": bool(reused),
        "trace_path": relative,
    }
    ablation = ablation_entry_metadata(match, policy.id)
    if ablation is not None:
        entry[ABLATION_METADATA_KEY] = ablation
    return entry


def selection_output_digest(manifest: Mapping[str, object]) -> str:
    """Canonical SHA-256 over a selection manifest excluding its own digest."""

    payload = {
        key: value
        for key, value in manifest.items()
        if key != "selection_output_digest"
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _build_selection_manifest(
    *,
    selection: SelectionSpec,
    shared_input_digest: str,
    cells: tuple[SelectionCell, ...],
    match: AblationMatchSummary | None = None,
) -> dict:
    entries = sorted(
        (dict(cell.manifest_entry) for cell in cells),
        key=lambda entry: (
            int(entry["seed"]),
            _POLICY_ORDER[str(entry["policy_id"])],
            str(entry["schedule_id"]),
        ),
    )
    study = selection.study
    manifest: dict = {
        "schema": SELECTION_MANIFEST_SCHEMA,
        "schema_version": SELECTION_MANIFEST_SCHEMA_VERSION,
        "status": "frozen",
        "study_id": study.study_id,
        "data_provenance": study.data_provenance,
        "study_digest": study_digest(study),
        "selection_spec_digest": selection_spec_digest(selection),
        "shared_input_digest": shared_input_digest,
        "selection_seeds": sorted(int(seed) for seed in study.seeds),
        "candidate_budget": study.candidate_budget,
        "panel": list(study.policy_ids()),
        "schedule_ids": sorted({str(entry["schedule_id"]) for entry in entries}),
        "entries": entries,
    }
    ablation = ablation_manifest_metadata(match)
    if ablation is not None:
        manifest[ABLATION_METADATA_KEY] = ablation
    manifest["selection_output_digest"] = selection_output_digest(manifest)
    return manifest


def run_selection_phase(
    selection: SelectionSpec,
    output: str | os.PathLike[str],
    *,
    dataset: Dataset | None = None,
    dataset_loader: DatasetLoader | None = None,
    evaluation_dataset: Dataset | None = None,
    evaluation_dataset_loader: DatasetLoader | None = None,
    progress: SelectionProgressHook | None = None,
    trace_validator: TraceValidator | None = None,
    resume_from: str | os.PathLike[str] | None = None,
    ablation_match: AblationMatchSummary | None = None,
) -> SelectionResult:
    """Run the selection panel over known schedules and freeze the output.

    The only accepted input type is a :class:`SelectionSpec`, which structurally
    carries no ``final_seeds`` or ``held_out_schedule``. Final seeds and the
    reserved held-out schedule are therefore unreachable from this runner, which
    enforces the milestone no-leak rule in the type system rather than by
    convention.

    Every (seed, schedule, policy) cell runs through the unchanged production
    :func:`~dfl_sim.run.run_from_config` composition with one immutable
    :class:`~dfl_sim.attacker.AdaptiveAttackController` per (seed, schedule), so
    all policies in a cell share the identical realized per-round attack. A
    private same-parent staging directory is validated with
    :class:`~dfl_sim.trajectory.TraceReader` and atomically renamed into place
    only after every cell succeeds; any failure publishes nothing.

    When ``resume_from`` names a prior selection checkpoint directory (a
    published selection output or a retained private staging directory), every
    recorded cell whose study/schedule/config/artifact digests match is reused
    from that checkpoint after its trace is revalidated with
    :class:`~dfl_sim.trajectory.TraceReader`; only missing or mismatched cells
    are recomputed. A study-level digest mismatch abandons reuse and fails
    closed, so stale or foreign evidence is never mixed into this run.

    ``progress`` receives ``(seed, schedule_id, policy_id)`` before each cell
    runs so callers can emit bounded per-cell progress markers.
    """

    if not isinstance(selection, SelectionSpec):
        raise EvaluationError(
            "run_selection_phase requires a SelectionSpec, got "
            f"{type(selection).__name__}"
        )
    study = selection.study
    destination = _validate_output_destination(output)
    _resolve_data_sources(
        study,
        dataset=dataset,
        dataset_loader=dataset_loader,
        evaluation_dataset=evaluation_dataset,
        evaluation_dataset_loader=evaluation_dataset_loader,
    )
    validator = TraceReader if trace_validator is None else trace_validator
    match = _resolve_ablation_match(study, provided=ablation_match)

    cells = _canonical_selection_cells(selection)
    if not cells:
        raise EvaluationError(
            "selection has no (seed, schedule, policy) cells to run"
        )

    # Derive every cell config and one controller/kappa sequence per
    # (seed, schedule) up front, then prove the paired-input invariant before
    # any expensive training or evaluation work begins.
    cell_configs: dict[tuple[int, str, str], ExperimentConfig] = {}
    controllers: dict[tuple[int, str], AdaptiveAttackController] = {}
    kappa_sequences: dict[tuple[int, str], tuple[float, ...]] = {}
    for seed, schedule, policy in cells:
        cell_configs[(seed, schedule.id, policy.id)] = build_cell_config(
            study.base, policy.aggregation, seed
        )
        key = (seed, schedule.id)
        if key not in controllers:
            controller = AdaptiveAttackController.from_episode(
                compromised_node=study.base.attack.compromised_node,
                schedule=schedule.schedule,
                total_rounds=study.base.rounds,
                master_seed=int(seed),
            )
            controllers[key] = controller
            kappa_sequences[key] = realized_kappa_sequence(
                schedule.schedule, study.base.rounds, controller.schedule_seed
            )
    shared_input_digest = validate_shared_inputs(list(cell_configs.values()))

    checkpoint_directory: Path | None = None
    checkpoint_records: dict[str, dict] = {}
    if resume_from is not None:
        expected: dict[str, object] = {
            "study_digest": study_digest(study),
            "selection_spec_digest": selection_spec_digest(selection),
            "shared_input_digest": shared_input_digest,
        }
        if match is not None:
            expected["ablation_match_digest"] = (
                match.matched_calibration_input_digest
            )
            expected["ablation_protocol_version"] = match.protocol_version
        checkpoint_directory, checkpoint_records = _load_resume_checkpoint(
            resume_from,
            phase="selection",
            expected=expected,
        )

    staging: Path | None = None
    published = False
    checkpoint_handle = None
    try:
        staging, staging_root = _prepare_staging(
            destination, prefix=_SELECTION_STAGING_PREFIX
        )
        checkpoint_handle = _open_checkpoint(staging / CHECKPOINT_FILENAME)
        artifact_cache: dict[str, str] = {}
        published_cells: list[SelectionCell] = []
        for seed, schedule, policy in cells:
            if progress is not None:
                progress(int(seed), schedule.id, policy.id)
            config = cell_configs[(seed, schedule.id, policy.id)]
            controller = controllers[(seed, schedule.id)]
            relative = _selection_relative_trace_path(
                policy.id, seed, schedule.id
            )
            trace_dir = staging / relative
            label = (
                f"policy {policy.id!r} seed {seed} schedule {schedule.id!r}"
            )
            artifact_digest = _artifact_digest(
                policy.aggregation, artifact_cache
            )
            reused, _metrics = _resume_cell(
                checkpoint_directory=checkpoint_directory,
                checkpoint_records=checkpoint_records,
                policy=policy,
                seed=seed,
                schedule=schedule,
                config=config,
                artifact_digest=artifact_digest,
                relative=relative,
                trace_dir=trace_dir,
                staging_root=staging_root,
                label=label,
                validator=validator,
                measure=False,
            )
            if not reused:
                _execute_cell_run(
                    config,
                    trace_dir,
                    staging_root=staging_root,
                    label=label,
                    dataset=dataset,
                    dataset_loader=dataset_loader,
                    evaluation_dataset=evaluation_dataset,
                    evaluation_dataset_loader=evaluation_dataset_loader,
                    controller=controller,
                )
                _validate_cell_trace(validator, trace_dir, label=label)

            _append_checkpoint(
                checkpoint_handle,
                _checkpoint_record(
                    phase="selection",
                    study=study,
                    selection_digest=selection_spec_digest(selection),
                    shared_input_digest=shared_input_digest,
                    policy=policy,
                    seed=seed,
                    schedule=schedule,
                    config=config,
                    artifact_digest=artifact_digest,
                    relative=relative,
                    reused=reused,
                    match=match,
                ),
            )
            entry = _selection_manifest_entry(
                study=study,
                policy=policy,
                seed=seed,
                schedule=schedule,
                config=config,
                controller=controller,
                artifact_digest=artifact_digest,
                relative=relative,
                reused=reused,
                match=match,
            )
            published_cells.append(
                SelectionCell(
                    seed=int(seed),
                    policy_id=policy.id,
                    schedule_id=schedule.id,
                    trace_path=trace_dir,
                    relative_trace_path=relative,
                    manifest_entry=entry,
                    realized_kappa_sequence=kappa_sequences[
                        (seed, schedule.id)
                    ],
                )
            )

        checkpoint_handle.close()
        checkpoint_handle = None
        manifest = _build_selection_manifest(
            selection=selection,
            shared_input_digest=shared_input_digest,
            cells=tuple(published_cells),
            match=match,
        )
        _publish_staging(
            staging, destination, manifest, SELECTION_MANIFEST_FILENAME
        )
        published = True
    finally:
        if checkpoint_handle is not None:
            try:
                checkpoint_handle.close()
            except OSError:  # pragma: no cover - best-effort cleanup
                pass
        if not published and staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    # Re-key the published cells to their final absolute trace paths.
    final_cells = tuple(
        SelectionCell(
            seed=cell.seed,
            policy_id=cell.policy_id,
            schedule_id=cell.schedule_id,
            trace_path=destination / cell.relative_trace_path,
            relative_trace_path=cell.relative_trace_path,
            manifest_entry=cell.manifest_entry,
            realized_kappa_sequence=cell.realized_kappa_sequence,
        )
        for cell in published_cells
    )
    return SelectionResult(
        selection=selection,
        output_path=destination,
        manifest_path=destination / SELECTION_MANIFEST_FILENAME,
        manifest=manifest,
        cells=final_cells,
        ablation=ablation_manifest_metadata(match),
    )


# --------------------------------------------------------------------------- #
# Final-phase runner (M003/S02)
# --------------------------------------------------------------------------- #
def final_output_digest(manifest: Mapping[str, object]) -> str:
    """Canonical SHA-256 over a final manifest excluding its own digest."""

    payload = {
        key: value
        for key, value in manifest.items()
        if key != "final_output_digest"
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _load_frozen_selection_manifest(
    selection_output: str | os.PathLike[str],
) -> tuple[Path, dict]:
    """Load and validate a frozen selection manifest, or fail closed.

    Requires an existing directory holding exactly the schema-v1 selection
    manifest with ``status == "frozen"``. Any missing, unreadable, malformed,
    or non-frozen output raises :class:`EvaluationError` before any final cell
    work begins, so a partial study cannot proceed from stale evidence.
    """

    directory = Path(os.path.abspath(os.fspath(selection_output)))
    if not directory.is_dir():
        raise EvaluationError(
            f"selection output directory not found: {directory}"
        )
    manifest_path = directory / SELECTION_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise EvaluationError(
            "selection output is missing "
            f"{SELECTION_MANIFEST_FILENAME}: {directory}"
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(
            f"could not read frozen selection manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise EvaluationError(
            f"selection manifest {manifest_path} is not a JSON object"
        )
    manifest = dict(raw)
    schema = manifest.get("schema")
    if schema != SELECTION_MANIFEST_SCHEMA:
        raise EvaluationError(
            f"selection manifest schema {schema!r} is not "
            f"{SELECTION_MANIFEST_SCHEMA!r}"
        )
    version = manifest.get("schema_version")
    if version != SELECTION_MANIFEST_SCHEMA_VERSION:
        raise EvaluationError(
            f"selection manifest schema_version {version!r} is not "
            f"{SELECTION_MANIFEST_SCHEMA_VERSION!r}"
        )
    status = manifest.get("status")
    if status != "frozen":
        raise EvaluationError(
            f"selection manifest status {status!r} is not 'frozen'"
        )
    _declared_manifest_panel(manifest, label="selection manifest")
    return directory, manifest


def _canonical_final_cells(
    study: HeldOutStudyConfig,
) -> tuple[tuple[int, NamedSchedule, PanelPolicy], ...]:
    """Every (seed, schedule, policy) final cell in deterministic order.

    Seeds are ascending, the known schedules plus the reserved held-out
    schedule are sorted by id, and policies follow the canonical panel order,
    so the traversal and the resulting manifest are independent of the
    caller's schedule declaration order.
    """

    schedules = tuple(
        sorted(
            (*study.known_schedules, study.held_out_schedule),
            key=lambda entry: entry.id,
        )
    )
    cells: list[tuple[int, NamedSchedule, PanelPolicy]] = []
    for seed in sorted(int(seed) for seed in study.final_seeds):
        for schedule in schedules:
            for policy in _canonical_panel(study.study):
                cells.append((seed, schedule, policy))
    return tuple(cells)


def _final_manifest_entry(
    *,
    study: BaselineStudyConfig,
    policy: PanelPolicy,
    seed: int,
    schedule: NamedSchedule,
    config: ExperimentConfig,
    controller: AdaptiveAttackController,
    artifact_digest: str | None,
    relative: str,
    held_out: bool,
    metrics: Mapping[str, object] | None = None,
    reused: bool = False,
    match: AblationMatchSummary | None = None,
) -> dict:
    """One per-cell final manifest entry.

    Reuses the selection entry shape (including the schedule id/spec/digest and
    the controller-derived ``realized_kappa_sequence``) and adds a boolean
    ``held_out`` flag marking the reserved unseen schedule. When the cell trace
    has been measured, a compact ``metrics`` block (final mean accuracy,
    worst-node accuracy, mean loss, node count) is embedded so downstream
    report compilation can read paired per-cell metrics without reopening every
    trace. ``reused`` records that the trace came from a manifest-matched
    resume checkpoint rather than this run.
    """

    entry = _selection_manifest_entry(
        study=study,
        policy=policy,
        seed=seed,
        schedule=schedule,
        config=config,
        controller=controller,
        artifact_digest=artifact_digest,
        relative=relative,
        reused=reused,
        match=match,
    )
    entry["held_out"] = bool(held_out)
    if metrics is not None:
        entry["metrics"] = dict(metrics)
    return entry


def _build_final_manifest(
    *,
    spec: HeldOutStudyConfig,
    selection_manifest: Mapping[str, object],
    shared_input_digest: str,
    cells: tuple[SelectionCell, ...],
    match: AblationMatchSummary | None = None,
) -> dict:
    entries = sorted(
        (dict(cell.manifest_entry) for cell in cells),
        key=lambda entry: (
            int(entry["seed"]),
            _POLICY_ORDER[str(entry["policy_id"])],
            str(entry["schedule_id"]),
        ),
    )
    study = spec.study
    held_out = spec.held_out_schedule
    manifest: dict = {
        "schema": FINAL_MANIFEST_SCHEMA,
        "schema_version": FINAL_MANIFEST_SCHEMA_VERSION,
        "phase": "final",
        "status": "complete",
        "study_id": study.study_id,
        "data_provenance": study.data_provenance,
        "study_digest": study_digest(study),
        "selection_spec_digest": selection_spec_digest(
            selection_projection(spec)
        ),
        "held_out_study_digest": held_out_study_digest(spec),
        "selection_output_digest": selection_manifest.get(
            "selection_output_digest"
        ),
        "shared_input_digest": shared_input_digest,
        "selection_seeds": sorted(int(seed) for seed in study.seeds),
        "final_seeds": sorted(int(seed) for seed in spec.final_seeds),
        "candidate_budget": study.candidate_budget,
        "panel": list(study.policy_ids()),
        "schedule_ids": sorted(
            {str(entry["schedule_id"]) for entry in entries}
        ),
        "held_out_schedule_id": held_out.id,
        "held_out_schedule": {
            "id": held_out.id,
            "schedule": held_out.schedule.model_dump(mode="json"),
            "schedule_digest": schedule_digest(held_out.schedule),
        },
        "entries": entries,
    }
    ablation = ablation_manifest_metadata(match)
    if ablation is not None:
        manifest[ABLATION_METADATA_KEY] = ablation
    manifest["final_output_digest"] = final_output_digest(manifest)
    return manifest


def run_final_phase(
    spec: HeldOutStudyConfig,
    selection_output: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    dataset: Dataset | None = None,
    dataset_loader: DatasetLoader | None = None,
    evaluation_dataset: Dataset | None = None,
    evaluation_dataset_loader: DatasetLoader | None = None,
    progress: SelectionProgressHook | None = None,
    trace_validator: TraceValidator | None = None,
    resume_from: str | os.PathLike[str] | None = None,
    ablation_match: AblationMatchSummary | None = None,
) -> FinalResult:
    """Run held-out final seeds over known plus reserved schedules.

    Accepts only a :class:`~dfl_sim.evaluation_config.HeldOutStudyConfig` and a
    previously frozen selection output. Before any final cell runs it (1)
    validates the frozen selection manifest schema and ``status == "frozen"``,
    (2) recomputes the selection-side digests and rejects a mutated
    selection-side spec, naming only the bounded digests, and (3) re-asserts
    that the final seeds are disjoint from the selection seeds.

    Every (seed, schedule, policy) cell, including the reserved unseen
    ``held_out_schedule``, runs through the unchanged production
    :func:`~dfl_sim.run.run_from_config` composition with one immutable
    :class:`~dfl_sim.attacker.AdaptiveAttackController` per (seed, schedule), so
    all policies in a cell share the identical realized attack. Each trace is
    validated with :class:`~dfl_sim.trajectory.TraceReader` and published
    atomically into ``output/final-manifest.json`` only after every cell
    succeeds; any failure publishes nothing.

    When ``resume_from`` names a prior final checkpoint directory (a published
    final output or a retained private staging directory), a recorded cell is
    reused only when the frozen selection-side digest, the current
    study/full digest, and the per-cell schedule/config/artifact digests all
    match; its trace is revalidated with
    :class:`~dfl_sim.trajectory.TraceReader` before being copied into staging.
    A global or schedule/artifact digest mismatch abandons reuse and fails
    closed, so stale or foreign evidence is never mixed into a published
    study.

    ``progress`` receives ``(seed, schedule_id, policy_id)`` before each cell
    runs so callers can emit bounded per-cell progress markers.
    """

    if not isinstance(spec, HeldOutStudyConfig):
        raise EvaluationError(
            "run_final_phase requires a HeldOutStudyConfig, got "
            f"{type(spec).__name__}"
        )
    study = spec.study
    destination = _validate_output_destination(output)
    _resolve_data_sources(
        study,
        dataset=dataset,
        dataset_loader=dataset_loader,
        evaluation_dataset=evaluation_dataset,
        evaluation_dataset_loader=evaluation_dataset_loader,
    )
    validator = TraceReader if trace_validator is None else trace_validator
    match = _resolve_ablation_match(study, provided=ablation_match)

    selection_dir, selection_manifest = _load_frozen_selection_manifest(
        selection_output
    )

    # Mutated selection-side spec guard: recompute the frozen digests and
    # reject any mismatch naming only the bounded digests.
    expected_selection_digest = selection_spec_digest(
        selection_projection(spec)
    )
    expected_study_digest = study_digest(study)
    observed_selection_digest = selection_manifest.get("selection_spec_digest")
    observed_study_digest = selection_manifest.get("study_digest")
    if observed_selection_digest != expected_selection_digest:
        raise EvaluationError(
            "selection-side spec mutated: frozen selection_spec_digest "
            f"{observed_selection_digest} does not match current "
            f"{expected_selection_digest}"
        )
    if observed_study_digest != expected_study_digest:
        raise EvaluationError(
            "selection-side spec mutated: frozen study_digest "
            f"{observed_study_digest} does not match current "
            f"{expected_study_digest}"
        )
    frozen_study_id = selection_manifest.get("study_id")
    if frozen_study_id != study.study_id:
        raise EvaluationError(
            "selection-side spec mutated: frozen study_id "
            f"{frozen_study_id!r} does not match current {study.study_id!r}"
        )

    # Re-assert the split. HeldOutStudyConfig already enforces this, but the
    # final runner re-checks so a mutated/injected spec can never run.
    overlap = sorted(
        set(int(seed) for seed in study.seeds)
        & set(int(seed) for seed in spec.final_seeds)
    )
    if overlap:
        raise EvaluationError(
            f"final seeds overlap the selection seeds: {overlap!r}"
        )

    cells = _canonical_final_cells(spec)
    if not cells:
        raise EvaluationError(
            "final phase has no (seed, schedule, policy) cells to run"
        )

    # Derive every cell config and one controller/kappa sequence per
    # (seed, schedule) up front, then prove the paired-input invariant before
    # any expensive training or evaluation work begins.
    held_out_id = spec.held_out_schedule.id
    cell_configs: dict[tuple[int, str, str], ExperimentConfig] = {}
    controllers: dict[tuple[int, str], AdaptiveAttackController] = {}
    kappa_sequences: dict[tuple[int, str], tuple[float, ...]] = {}
    for seed, schedule, policy in cells:
        cell_configs[(seed, schedule.id, policy.id)] = build_cell_config(
            study.base, policy.aggregation, seed
        )
        key = (seed, schedule.id)
        if key not in controllers:
            controller = AdaptiveAttackController.from_episode(
                compromised_node=study.base.attack.compromised_node,
                schedule=schedule.schedule,
                total_rounds=study.base.rounds,
                master_seed=int(seed),
            )
            controllers[key] = controller
            kappa_sequences[key] = realized_kappa_sequence(
                schedule.schedule, study.base.rounds, controller.schedule_seed
            )
    shared_input_digest = validate_shared_inputs(list(cell_configs.values()))

    expected_selection_digest = selection_spec_digest(selection_projection(spec))
    checkpoint_directory: Path | None = None
    checkpoint_records: dict[str, dict] = {}
    if resume_from is not None:
        expected: dict[str, object] = {
            "study_digest": expected_study_digest,
            "selection_spec_digest": expected_selection_digest,
            "held_out_study_digest": held_out_study_digest(spec),
            "shared_input_digest": shared_input_digest,
        }
        if match is not None:
            expected["ablation_match_digest"] = (
                match.matched_calibration_input_digest
            )
            expected["ablation_protocol_version"] = match.protocol_version
        checkpoint_directory, checkpoint_records = _load_resume_checkpoint(
            resume_from,
            phase="final",
            expected=expected,
        )

    staging: Path | None = None
    published = False
    published_cells: list[SelectionCell] = []
    checkpoint_handle = None
    try:
        staging, staging_root = _prepare_staging(
            destination, prefix=_FINAL_STAGING_PREFIX
        )
        checkpoint_handle = _open_checkpoint(staging / CHECKPOINT_FILENAME)
        artifact_cache: dict[str, str] = {}
        for seed, schedule, policy in cells:
            if progress is not None:
                progress(int(seed), schedule.id, policy.id)
            config = cell_configs[(seed, schedule.id, policy.id)]
            controller = controllers[(seed, schedule.id)]
            relative = _selection_relative_trace_path(
                policy.id, seed, schedule.id
            )
            trace_dir = staging / relative
            label = (
                f"policy {policy.id!r} seed {seed} schedule {schedule.id!r}"
            )
            artifact_digest = _artifact_digest(
                policy.aggregation, artifact_cache
            )
            reused, metrics = _resume_cell(
                checkpoint_directory=checkpoint_directory,
                checkpoint_records=checkpoint_records,
                policy=policy,
                seed=seed,
                schedule=schedule,
                config=config,
                artifact_digest=artifact_digest,
                relative=relative,
                trace_dir=trace_dir,
                staging_root=staging_root,
                label=label,
                validator=validator,
                measure=True,
            )
            if not reused:
                _execute_cell_run(
                    config,
                    trace_dir,
                    staging_root=staging_root,
                    label=label,
                    dataset=dataset,
                    dataset_loader=dataset_loader,
                    evaluation_dataset=evaluation_dataset,
                    evaluation_dataset_loader=evaluation_dataset_loader,
                    controller=controller,
                )
                metrics = _validate_cell_trace(
                    validator, trace_dir, label=label, measure=True
                )

            _append_checkpoint(
                checkpoint_handle,
                _checkpoint_record(
                    phase="final",
                    study=study,
                    selection_digest=expected_selection_digest,
                    shared_input_digest=shared_input_digest,
                    policy=policy,
                    seed=seed,
                    schedule=schedule,
                    config=config,
                    artifact_digest=artifact_digest,
                    relative=relative,
                    reused=reused,
                    match=match,
                    extra={
                        "held_out_study_digest": held_out_study_digest(spec),
                        "selection_output_digest": selection_manifest.get(
                            "selection_output_digest"
                        ),
                    },
                ),
            )
            entry = _final_manifest_entry(
                study=study,
                policy=policy,
                seed=seed,
                schedule=schedule,
                config=config,
                controller=controller,
                artifact_digest=artifact_digest,
                relative=relative,
                held_out=schedule.id == held_out_id,
                metrics=metrics,
                reused=reused,
                match=match,
            )
            published_cells.append(
                SelectionCell(
                    seed=int(seed),
                    policy_id=policy.id,
                    schedule_id=schedule.id,
                    trace_path=trace_dir,
                    relative_trace_path=relative,
                    manifest_entry=entry,
                    realized_kappa_sequence=kappa_sequences[
                        (seed, schedule.id)
                    ],
                )
            )

        checkpoint_handle.close()
        checkpoint_handle = None
        manifest = _build_final_manifest(
            spec=spec,
            selection_manifest=selection_manifest,
            shared_input_digest=shared_input_digest,
            cells=tuple(published_cells),
            match=match,
        )
        _publish_staging(
            staging, destination, manifest, FINAL_MANIFEST_FILENAME
        )
        published = True
    finally:
        if checkpoint_handle is not None:
            try:
                checkpoint_handle.close()
            except OSError:  # pragma: no cover - best-effort cleanup
                pass
        if not published and staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    # Re-key the published cells to their final absolute trace paths.
    final_cells = tuple(
        SelectionCell(
            seed=cell.seed,
            policy_id=cell.policy_id,
            schedule_id=cell.schedule_id,
            trace_path=destination / cell.relative_trace_path,
            relative_trace_path=cell.relative_trace_path,
            manifest_entry=cell.manifest_entry,
            realized_kappa_sequence=cell.realized_kappa_sequence,
        )
        for cell in published_cells
    )
    return FinalResult(
        study=spec,
        selection_output_path=selection_dir,
        output_path=destination,
        manifest_path=destination / FINAL_MANIFEST_FILENAME,
        manifest=manifest,
        cells=final_cells,
        ablation=ablation_manifest_metadata(match),
    )


# --------------------------------------------------------------------------- #
# Two-phase held-out study orchestrator (M003/S02/T05)
# --------------------------------------------------------------------------- #
def _rekey_selection_result(
    result: SelectionResult, output_path: Path
) -> SelectionResult:
    """Re-point a published selection result at a new (renamed) output root."""

    cells = tuple(
        SelectionCell(
            seed=cell.seed,
            policy_id=cell.policy_id,
            schedule_id=cell.schedule_id,
            trace_path=output_path / cell.relative_trace_path,
            relative_trace_path=cell.relative_trace_path,
            manifest_entry=cell.manifest_entry,
            realized_kappa_sequence=cell.realized_kappa_sequence,
        )
        for cell in result.cells
    )
    return SelectionResult(
        selection=result.selection,
        output_path=output_path,
        manifest_path=output_path / SELECTION_MANIFEST_FILENAME,
        manifest=result.manifest,
        cells=cells,
        ablation=result.ablation,
    )


def _rekey_final_result(
    result: FinalResult, *, selection_output_path: Path, output_path: Path
) -> FinalResult:
    """Re-point a published final result at its renamed roots."""

    cells = tuple(
        SelectionCell(
            seed=cell.seed,
            policy_id=cell.policy_id,
            schedule_id=cell.schedule_id,
            trace_path=output_path / cell.relative_trace_path,
            relative_trace_path=cell.relative_trace_path,
            manifest_entry=cell.manifest_entry,
            realized_kappa_sequence=cell.realized_kappa_sequence,
        )
        for cell in result.cells
    )
    return FinalResult(
        study=result.study,
        selection_output_path=selection_output_path,
        output_path=output_path,
        manifest_path=output_path / FINAL_MANIFEST_FILENAME,
        manifest=result.manifest,
        cells=cells,
        ablation=result.ablation,
    )


def _phase_progress(
    progress: HeldOutProgressHook | None, phase: str
) -> SelectionProgressHook | None:
    """Adapt the two-phase hook to one phase's ``(seed, schedule, policy)``."""

    if progress is None:
        return None

    def _hook(seed: int, schedule_id: str, policy_id: str) -> None:
        progress(phase, int(seed), schedule_id, policy_id)

    return _hook


def run_held_out_study(
    spec: HeldOutStudyConfig,
    output: str | os.PathLike[str],
    *,
    dataset: Dataset | None = None,
    dataset_loader: DatasetLoader | None = None,
    evaluation_dataset: Dataset | None = None,
    evaluation_dataset_loader: DatasetLoader | None = None,
    progress: HeldOutProgressHook | None = None,
    trace_validator: TraceValidator | None = None,
    resume_from: str | os.PathLike[str] | None = None,
) -> HeldOutStudyResult:
    """Run the full selection -> freeze -> final held-out study in one call.

    A private same-parent staging directory is created, the selection phase is
    run into ``staging/selection`` and its frozen manifest fsynced, the final
    phase then reads that on-disk frozen output and publishes into
    ``staging/final``, and finally the whole staging directory is atomically
    renamed into ``output``. The published study root therefore holds
    ``selection/selection-manifest.json`` and ``final/final-manifest.json``
    plus their traces and a per-cell ``cells.jsonl`` checkpoint per phase.

    This is the only entry point that runs both phases; the phase runners
    themselves never see the final seeds during selection (the selection runner
    accepts only a :class:`SelectionSpec`), and the final runner re-asserts the
    split and the frozen selection digest before running.

    When ``resume_from`` names a prior study root or retained staging root, a
    matching ``selection/cells.jsonl`` resumes the selection phase and a
    matching ``final/cells.jsonl`` (or a flat ``cells.jsonl``) resumes the final
    phase; manifest-matched completed cells are reused and only missing or
    mismatched cells are recomputed. A study-level digest mismatch fails
    closed, so an interrupted study is never continued from stale evidence.

    ``progress`` receives ``(phase, seed, schedule_id, policy_id)`` before each
    cell runs, where ``phase`` is ``"selection"`` or ``"final"``. Any failure
    removes the staging directory, publishes nothing, and re-raises the
    already-wrapped :class:`EvaluationError`.
    """

    if not isinstance(spec, HeldOutStudyConfig):
        raise EvaluationError(
            "run_held_out_study requires a HeldOutStudyConfig, got "
            f"{type(spec).__name__}"
        )
    study = spec.study
    destination = _validate_output_destination(output)
    _resolve_data_sources(
        study,
        dataset=dataset,
        dataset_loader=dataset_loader,
        evaluation_dataset=evaluation_dataset,
        evaluation_dataset_loader=evaluation_dataset_loader,
    )
    validator = TraceReader if trace_validator is None else trace_validator
    # Validate the matched triple once for the whole two-phase invocation and
    # reuse the immutable summary in both phases (Q6). A mismatch fails closed
    # here, before any cell runs or any checkpoint is reused.
    match = _resolve_ablation_match(study)

    selection_resume: Path | None = None
    final_resume: Path | None = None
    if resume_from is not None:
        base = Path(os.path.abspath(os.fspath(resume_from)))
        if (base / "selection" / CHECKPOINT_FILENAME).is_file():
            selection_resume = base / "selection"
        if (base / "final" / CHECKPOINT_FILENAME).is_file():
            final_resume = base / "final"
        elif (base / CHECKPOINT_FILENAME).is_file():
            final_resume = base
        if selection_resume is None and final_resume is None:
            raise EvaluationError(
                f"resume checkpoint not found under {base}: expected "
                f"selection/{CHECKPOINT_FILENAME} or final/"
                f"{CHECKPOINT_FILENAME}"
            )

    staging: Path | None = None
    published = False
    try:
        staging, _staging_root = _prepare_staging(
            destination, prefix=_HELD_OUT_STAGING_PREFIX
        )
        selection_output = staging / "selection"
        selection_result = run_selection_phase(
            selection_projection(spec),
            selection_output,
            dataset=dataset,
            dataset_loader=dataset_loader,
            evaluation_dataset=evaluation_dataset,
            evaluation_dataset_loader=evaluation_dataset_loader,
            progress=_phase_progress(progress, "selection"),
            trace_validator=validator,
            resume_from=selection_resume,
            ablation_match=match,
        )
        # Durably commit the frozen selection output before the final phase
        # reads it back from disk.
        _fsync_dir(selection_output)

        final_output = staging / "final"
        final_result = run_final_phase(
            spec,
            selection_output,
            final_output,
            dataset=dataset,
            dataset_loader=dataset_loader,
            evaluation_dataset=evaluation_dataset,
            evaluation_dataset_loader=evaluation_dataset_loader,
            progress=_phase_progress(progress, "final"),
            trace_validator=validator,
            resume_from=final_resume,
            ablation_match=match,
        )
        _fsync_dir(final_output)
        # Build the paired per-cell metrics summary from the just-written final
        # traces while still inside staging, so the published study root is a
        # single atomic unit containing selection/, final/, and the summary.
        summary = build_robustness_summary(
            final_result.manifest, trace_root=final_output
        )
        _write_bytes_synced(
            staging / ROBUSTNESS_SUMMARY_FILENAME, manifest_bytes(summary)
        )
        _fsync_dir(staging)
        try:
            os.rename(staging, destination)
        except OSError as exc:
            raise EvaluationError(
                f"could not publish study output {destination}: {exc}"
            ) from exc
        _fsync_dir(destination.parent)
        published = True
    finally:
        if not published and staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    selection_path = destination / "selection"
    final_path = destination / "final"
    return HeldOutStudyResult(
        spec=spec,
        output_path=destination,
        selection_result=_rekey_selection_result(
            selection_result, selection_path
        ),
        final_result=_rekey_final_result(
            final_result,
            selection_output_path=selection_path,
            output_path=final_path,
        ),
        ablation=ablation_manifest_metadata(match),
    )


# --------------------------------------------------------------------------- #
# Paired per-cell metrics summary (M003/S02/T06)
# --------------------------------------------------------------------------- #
def _final_round_metrics_from_events(
    events: Sequence[Mapping[str, object]], *, label: str
) -> dict:
    """Reduce validated events to the final evaluated round's per-node metrics.

    Returns the mean and worst-node accuracy, the mean loss, and the node count
    for the last (highest-index) evaluated round. Raises a bounded
    :class:`EvaluationError` when the trace has no rounds or a malformed node
    record instead of silently producing partial evidence.
    """

    if not events:
        raise EvaluationError(f"{label} trace has no evaluated rounds")
    final = events[-1]
    nodes = final.get("nodes") if isinstance(final, Mapping) else None
    if (
        not isinstance(nodes, Sequence)
        or isinstance(nodes, (str, bytes))
        or not nodes
    ):
        raise EvaluationError(f"{label} final round has no per-node metrics")
    accuracies: list[float] = []
    losses: list[float] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            raise EvaluationError(f"{label} final round has a malformed node")
        try:
            accuracy = float(node["accuracy"])
            loss = float(node["loss"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError(
                f"{label} final round has malformed node metrics"
            ) from exc
        if not math.isfinite(accuracy) or not math.isfinite(loss):
            raise EvaluationError(
                f"{label} final round has non-finite node metrics"
            )
        accuracies.append(accuracy)
        losses.append(loss)
    return {
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "worst_node_accuracy": min(accuracies),
        "mean_loss": sum(losses) / len(losses),
        "node_count": len(accuracies),
    }


def read_final_round_metrics(
    trace_path: str | os.PathLike[str],
    *,
    trace_reader: TraceValidator = TraceReader,
) -> dict:
    """Read one schema-v1 trace and return its final-round per-node metrics."""

    display = os.fspath(trace_path)
    try:
        with trace_reader(trace_path) as reader:
            schema = getattr(reader, "schema_version", SCHEMA_VERSION)
            if schema != SCHEMA_VERSION:
                raise EvaluationError(
                    f"trace {display!r} is schema version {schema!r}, "
                    f"expected {SCHEMA_VERSION}"
                )
            events = getattr(reader, "events", None)
            if events is None:
                raise EvaluationError(
                    f"trace {display!r} reader exposes no events for metrics"
                )
            return _final_round_metrics_from_events(
                tuple(events), label=f"trace {display!r}"
            )
    except EvaluationError:
        raise
    except (TrajectoryError, OSError) as exc:
        raise EvaluationError(
            f"could not read trace {display!r}: {exc}"
        ) from exc


def _population_std(values: Sequence[float]) -> float:
    """Population standard deviation (``ddof == 0``) of the given values."""

    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return math.sqrt(variance)


def _linear_quantile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (R-7 / numpy ``method="linear"``)."""

    count = len(ordered)
    if count == 1:
        return float(ordered[0])
    position = q * (count - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    fraction = position - lower
    return float(ordered[lower]) + fraction * (
        float(ordered[upper]) - float(ordered[lower])
    )


def _declared_manifest_panel(
    manifest: Mapping[str, object], *, label: str
) -> tuple[str, ...]:
    """Return the manifest's declared panel after bounded validation.

    The runner always emits the canonical six- or eight-policy panel. A declared
    panel must be a non-empty sequence of known policy ids; if it is exactly the
    eight-policy canonical panel it must carry the ablation identity block, and
    the block may not appear on any other panel. This closes the optional
    metadata coupling without forcing synthetic partial-panel unit fixtures to
    grow to six entries.
    """

    raw_panel = manifest.get("panel")
    if raw_panel is None:
        panel: tuple[str, ...] = REQUIRED_POLICY_IDS
    elif (
        isinstance(raw_panel, Sequence)
        and not isinstance(raw_panel, (str, bytes))
        and raw_panel
    ):
        panel = tuple(str(policy_id) for policy_id in raw_panel)
    else:
        raise EvaluationError(
            f"{label} panel must be a non-empty sequence"
        )
    unknown = sorted(
        policy_id for policy_id in panel if policy_id not in _POLICY_ORDER
    )
    if unknown:
        raise EvaluationError(
            f"{label} panel declares unknown policy ids {unknown!r}"
        )
    ablation = manifest.get(ABLATION_METADATA_KEY)
    if panel == EXTENDED_POLICY_IDS and not isinstance(ablation, Mapping):
        raise EvaluationError(
            f"{label} declares the eight-policy panel without an ablation "
            "identity block"
        )
    if ablation is not None and panel != EXTENDED_POLICY_IDS:
        raise EvaluationError(
            f"{label} declares ablation metadata without the eight-policy "
            "panel"
        )
    return panel


def _validate_final_manifest_for_summary(manifest: Mapping[str, object]) -> None:
    """Validate the final-manifest envelope needed to build paired metrics."""

    if manifest.get("schema") != FINAL_MANIFEST_SCHEMA:
        raise EvaluationError(
            f"final manifest schema {manifest.get('schema')!r} is not "
            f"{FINAL_MANIFEST_SCHEMA!r}"
        )
    if manifest.get("schema_version") != FINAL_MANIFEST_SCHEMA_VERSION:
        raise EvaluationError(
            f"final manifest schema_version {manifest.get('schema_version')!r} "
            f"is not {FINAL_MANIFEST_SCHEMA_VERSION!r}"
        )
    if manifest.get("phase") != "final":
        raise EvaluationError(
            f"final manifest phase {manifest.get('phase')!r} is not 'final'"
        )
    if manifest.get("status") != "complete":
        raise EvaluationError(
            f"final manifest status {manifest.get('status')!r} is not 'complete'"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise EvaluationError("final manifest entries must be a sequence")
    if not entries:
        raise EvaluationError("final manifest has no entries to summarize")
    final_seeds = manifest.get("final_seeds")
    if (
        not isinstance(final_seeds, Sequence)
        or isinstance(final_seeds, (str, bytes))
        or not final_seeds
    ):
        raise EvaluationError("final manifest final_seeds must be a non-empty sequence")
    _declared_manifest_panel(manifest, label="final manifest")


def _load_final_manifest_for_summary(
    study_output: str | os.PathLike[str],
) -> tuple[Path, Path, dict]:
    """Locate and validate the final manifest inside a published study root.

    Accepts either the two-phase study root (``<root>/final/final-manifest.json``
    with traces under ``<root>/final``) or a flat final-only output
    (``<root>/final-manifest.json`` with traces under ``<root>``).
    """

    root = Path(os.path.abspath(os.fspath(study_output)))
    if not root.is_dir():
        raise EvaluationError(f"study output directory not found: {root}")
    nested = root / "final" / FINAL_MANIFEST_FILENAME
    flat = root / FINAL_MANIFEST_FILENAME
    if nested.is_file():
        final_dir, manifest_path = root / "final", nested
    elif flat.is_file():
        final_dir, manifest_path = root, flat
    else:
        raise EvaluationError(
            f"study output is missing {FINAL_MANIFEST_FILENAME}: {root}"
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(
            f"could not read final manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise EvaluationError(
            f"final manifest {manifest_path} is not a JSON object"
        )
    manifest = dict(raw)
    _validate_final_manifest_for_summary(manifest)
    return root, final_dir, manifest


def _coerce_schedule_spec(payload: object, *, label: str):
    """Reconstruct a validated schedule from its dumped manifest payload."""

    if not isinstance(payload, Mapping):
        raise EvaluationError(f"{label} schedule payload is not an object")
    fields = dict(payload)
    kind = fields.pop("kind", None)
    try:
        return build_schedule(str(kind), **fields)
    except (ScheduleError, TypeError, ValueError) as exc:
        raise EvaluationError(
            f"{label} schedule spec is malformed: {exc}"
        ) from exc


def build_robustness_summary(
    final_manifest: Mapping[str, object],
    *,
    trace_root: str | os.PathLike[str],
    selection_seeds: Sequence[int] | None = None,
    trace_reader: TraceValidator = TraceReader,
) -> dict:
    """Aggregate paired per-cell metrics over a validated final manifest.

    Every cited trace is reopened through ``trace_reader`` and reduced to its
    final evaluated round's per-node ``loss``/``accuracy``. Each
    ``(policy_id, schedule_id)`` pair across the paired final seeds yields the
    mean accuracy, population standard deviation, linearly interpolated IQR,
    mean loss, worst-node accuracy (minimum over nodes and seeds), node count,
    and the seed list. Fails closed with a bounded error when a cited trace is
    missing/unreadable, its seed/policy provenance or canonical trace path
    disagrees with the manifest, the recorded schedule is internally
    inconsistent, or the per-policy seed sets differ from the manifest split.

    This is deliberately *not* the S05 canonical report: it carries no figures
    and no claim/evidence-class framing, only deterministic paired metrics.
    """

    manifest = dict(final_manifest)
    _validate_final_manifest_for_summary(manifest)

    entries = tuple(manifest["entries"])
    final_seeds = tuple(sorted(int(seed) for seed in manifest["final_seeds"]))
    if selection_seeds is None:
        raw_selection = manifest.get("selection_seeds")
        if (
            not isinstance(raw_selection, Sequence)
            or isinstance(raw_selection, (str, bytes))
        ):
            raise EvaluationError(
                "final manifest selection_seeds must be a sequence"
            )
        selection_seeds = tuple(int(seed) for seed in raw_selection)
    else:
        selection_seeds = tuple(int(seed) for seed in selection_seeds)
    selection_seeds = tuple(sorted(selection_seeds))
    if set(selection_seeds) & set(final_seeds):
        raise EvaluationError(
            "selection and final seeds overlap in the paired study split"
        )

    root = Path(os.path.abspath(os.fspath(trace_root)))
    if not root.is_dir():
        raise EvaluationError(f"trace root directory not found: {root}")
    resolved_root = root.resolve()

    cells: dict[tuple[str, str], dict[int, dict]] = {}
    held_out_by_schedule: dict[str, bool] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise EvaluationError("final manifest entry must be an object")
        try:
            seed = int(entry["seed"])
            policy_id = str(entry["policy_id"])
            policy_kind = str(entry.get("policy_kind", policy_id))
            schedule_id = str(entry["schedule_id"])
            relative = str(entry["trace_path"])
            held_out = bool(entry["held_out"])
            schedule_payload = entry["schedule"]
            schedule_digest_value = str(entry["schedule_digest"])
            realized = tuple(float(x) for x in entry["realized_kappa_sequence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError("final manifest entry is malformed") from exc
        entry_ablation = entry.get(ABLATION_METADATA_KEY)
        if entry_ablation is not None and not isinstance(entry_ablation, Mapping):
            raise EvaluationError(
                f"final manifest entry {policy_id!r} ablation metadata must be an "
                "object"
            )
        label = f"policy {policy_id!r} seed {seed} schedule {schedule_id!r}"

        expected = _selection_relative_trace_path(policy_id, seed, schedule_id)
        if relative != expected:
            raise EvaluationError(
                f"{label} trace_path {relative!r} does not match the canonical "
                f"schedule/policy/seed location {expected!r}"
            )
        trace_path = root / relative
        try:
            resolved_trace = trace_path.resolve()
        except OSError as exc:
            raise EvaluationError(
                f"{label} trace path is unresolvable: {exc}"
            ) from exc
        if not resolved_trace.is_relative_to(resolved_root):
            raise EvaluationError(f"{label} trace path escapes the trace root")

        schedule_spec = _coerce_schedule_spec(schedule_payload, label=label)
        if schedule_digest(schedule_spec) != schedule_digest_value:
            raise EvaluationError(
                f"{label} schedule_digest disagrees with the recorded schedule"
            )

        try:
            with trace_reader(resolved_trace) as reader:
                schema = getattr(reader, "schema_version", SCHEMA_VERSION)
                if schema != SCHEMA_VERSION:
                    raise EvaluationError(
                        f"{label} trace is schema version {schema!r}, "
                        f"expected {SCHEMA_VERSION}"
                    )
                config = getattr(reader, "config", None)
                events = getattr(reader, "events", None)
                if config is None or events is None:
                    raise EvaluationError(
                        f"{label} trace reader exposes no config/events"
                    )
                trace_seed = int(config.seed)
                trace_kind = str(config.aggregation.kind)
                rounds = int(config.rounds)
                metrics = _final_round_metrics_from_events(
                    tuple(events), label=label
                )
        except EvaluationError:
            raise
        except (TrajectoryError, OSError) as exc:
            raise EvaluationError(
                f"could not read trace for {label}: {exc}"
            ) from exc

        if trace_seed != seed:
            raise EvaluationError(
                f"{label} trace seed {trace_seed} disagrees with the manifest"
            )
        if trace_kind != policy_kind:
            raise EvaluationError(
                f"{label} trace aggregation {trace_kind!r} disagrees with the "
                "manifest policy provenance"
            )
        expected_kappa = realized_kappa_sequence(
            schedule_spec, rounds, derive_schedule_seed(seed, schedule_spec)
        )
        if tuple(expected_kappa) != realized:
            raise EvaluationError(
                f"{label} realized_kappa_sequence disagrees with the recorded "
                "schedule"
            )

        previous_flag = held_out_by_schedule.setdefault(schedule_id, held_out)
        if previous_flag != held_out:
            raise EvaluationError(
                f"schedule {schedule_id!r} has inconsistent held_out flags"
            )
        bucket = cells.setdefault((schedule_id, policy_id), {})
        if seed in bucket:
            raise EvaluationError(f"duplicate manifest cell {label}")
        bucket[seed] = {
            "metrics": metrics,
            "held_out": held_out,
            "ablation": None if entry_ablation is None else dict(entry_ablation),
        }

    panel = _declared_manifest_panel(manifest, label="final manifest")
    schedule_ids = sorted({schedule_id for schedule_id, _ in cells})
    expected_seeds = set(final_seeds)

    aggregates: list[dict] = []
    for schedule_id in schedule_ids:
        for policy_id in panel:
            bucket = cells.get((schedule_id, policy_id))
            if not bucket:
                raise EvaluationError(
                    f"schedule {schedule_id!r} is missing policy {policy_id!r}"
                )
            if set(bucket) != expected_seeds:
                raise EvaluationError(
                    f"schedule {schedule_id!r} policy {policy_id!r} paired seed "
                    f"set {sorted(bucket)!r} disagrees with the final seeds "
                    f"{list(final_seeds)!r}"
                )
            seeds = sorted(bucket)
            per_seed_accuracy = [
                float(bucket[seed]["metrics"]["mean_accuracy"]) for seed in seeds
            ]
            per_seed_loss = [
                float(bucket[seed]["metrics"]["mean_loss"]) for seed in seeds
            ]
            node_counts = {
                int(bucket[seed]["metrics"]["node_count"]) for seed in seeds
            }
            if len(node_counts) != 1:
                raise EvaluationError(
                    f"schedule {schedule_id!r} policy {policy_id!r} has "
                    f"inconsistent node counts {sorted(node_counts)!r}"
                )
            ordered_accuracy = sorted(per_seed_accuracy)
            ablations = [
                bucket[seed]["ablation"] for seed in seeds
            ]
            if any(value is not None for value in ablations):
                non_null = [value for value in ablations if value is not None]
                if len(non_null) != len(seeds) or any(
                    value != non_null[0] for value in non_null[1:]
                ):
                    raise EvaluationError(
                        f"schedule {schedule_id!r} policy {policy_id!r} has "
                        "inconsistent ablation identity across paired seeds"
                    )
            aggregate = {
                "policy_id": policy_id,
                "schedule_id": schedule_id,
                "held_out": bool(bucket[seeds[0]]["held_out"]),
                "seeds": seeds,
                "count": len(seeds),
                "node_count": node_counts.pop(),
                "mean_accuracy": sum(per_seed_accuracy) / len(seeds),
                "std_accuracy": _population_std(per_seed_accuracy),
                "iqr_accuracy": _linear_quantile(ordered_accuracy, 0.75)
                - _linear_quantile(ordered_accuracy, 0.25),
                "mean_loss": sum(per_seed_loss) / len(seeds),
                "worst_node_accuracy": min(
                    float(bucket[seed]["metrics"]["worst_node_accuracy"])
                    for seed in seeds
                ),
                "per_seed": [
                    {
                        "seed": seed,
                        "mean_accuracy": per_seed_accuracy[index],
                        "mean_loss": per_seed_loss[index],
                        "worst_node_accuracy": float(
                            bucket[seed]["metrics"]["worst_node_accuracy"]
                        ),
                    }
                    for index, seed in enumerate(seeds)
                ],
            }
            if ablations[0] is not None:
                aggregate[ABLATION_METADATA_KEY] = ablations[0]
            aggregates.append(aggregate)

    summary: dict = {
        "schema": ROBUSTNESS_SUMMARY_SCHEMA,
        "schema_version": ROBUSTNESS_SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "study_id": manifest.get("study_id"),
        "data_provenance": manifest.get("data_provenance"),
        "study_digest": manifest.get("study_digest"),
        "selection_spec_digest": manifest.get("selection_spec_digest"),
        "held_out_study_digest": manifest.get("held_out_study_digest"),
        "final_output_digest": manifest.get("final_output_digest"),
        "selection_seeds": list(selection_seeds),
        "final_seeds": list(final_seeds),
        "panel": list(panel),
        "schedule_ids": list(schedule_ids),
        "std_method": ROBUSTNESS_STD_METHOD,
        "quantile_method": ROBUSTNESS_QUANTILE_METHOD,
        "aggregates": aggregates,
    }
    # Only the opt-in eight-policy summary gains the ablation block; a legacy
    # six-policy summary keeps its exact prior shape.
    ablation = manifest.get(ABLATION_METADATA_KEY)
    if ablation is not None:
        summary[ABLATION_METADATA_KEY] = ablation
    return summary


def summarize_held_out_study(
    study_output: str | os.PathLike[str],
    *,
    write: bool = True,
    trace_reader: TraceValidator = TraceReader,
) -> dict:
    """Emit the paired per-cell robustness summary for a published study root.

    Loads and validates the final manifest, cross-checks the frozen selection
    manifest when it is present, then builds the paired metrics summary and, by
    default, writes it deterministically to
    ``<study_output>/robustness-summary.json``. Returns the summary mapping so
    callers (e.g. the S05 report compiler) can consume it without re-reading
    the file.
    """

    root, final_dir, manifest = _load_final_manifest_for_summary(study_output)

    selection_manifest_path = (
        root / "selection" / SELECTION_MANIFEST_FILENAME
    )
    if selection_manifest_path.is_file():
        try:
            selection_raw = json.loads(
                selection_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationError(
                f"could not read selection manifest "
                f"{selection_manifest_path}: {exc}"
            ) from exc
        if not isinstance(selection_raw, dict):
            raise EvaluationError(
                f"selection manifest {selection_manifest_path} is not a JSON "
                "object"
            )
        if selection_raw.get("schema") != SELECTION_MANIFEST_SCHEMA:
            raise EvaluationError(
                f"selection manifest schema {selection_raw.get('schema')!r} "
                f"is not {SELECTION_MANIFEST_SCHEMA!r}"
            )
        if selection_raw.get("status") != "frozen":
            raise EvaluationError(
                f"selection manifest status {selection_raw.get('status')!r} "
                "is not 'frozen'"
            )
        frozen_seeds = selection_raw.get("selection_seeds")
        manifest_seeds = manifest.get("selection_seeds")
        if (
            isinstance(frozen_seeds, Sequence)
            and not isinstance(frozen_seeds, (str, bytes))
            and isinstance(manifest_seeds, Sequence)
            and not isinstance(manifest_seeds, (str, bytes))
            and sorted(int(seed) for seed in frozen_seeds)
            != sorted(int(seed) for seed in manifest_seeds)
        ):
            raise EvaluationError(
                "frozen selection seed list disagrees with the final manifest"
            )

    summary = build_robustness_summary(
        manifest,
        trace_root=final_dir,
        selection_seeds=manifest.get("selection_seeds"),
        trace_reader=trace_reader,
    )
    if write:
        _write_bytes_synced(
            root / ROBUSTNESS_SUMMARY_FILENAME, manifest_bytes(summary)
        )
    return summary
