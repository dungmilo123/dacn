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
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from torch.utils.data import Dataset

from .bayesian_artifact import BayesianArtifactError, load_bayesian_artifact
from .config import AggregationConfig, ExperimentConfig
from .evaluation_config import (
    REQUIRED_POLICY_IDS,
    BaselineStudyConfig,
    PanelPolicy,
    hyperparameter_digest,
    load_study_config,
    policy_identity,
    selected_hyperparameters,
    study_digest,
)
from .run import DatasetLoader, run_from_config
from .trajectory import SCHEMA_VERSION, TraceReader, TrajectoryError

__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "MANIFEST_SCHEMA_VERSION",
    "TRACES_DIRNAME",
    "BaselineCell",
    "BaselineStudyResult",
    "EvaluationError",
    "build_cell_config",
    "compute_shared_input_digest",
    "load_study_config",
    "run_baseline_study",
    "validate_shared_inputs",
]

#: Manifest file name written at the study output root.
MANIFEST_FILENAME = "baseline-manifest.json"

#: Subdirectory holding one trace per (policy, seed) cell.
TRACES_DIRNAME = "traces"

#: Versioned manifest identity.
MANIFEST_SCHEMA = "dfl_baseline_study_manifest"
MANIFEST_SCHEMA_VERSION = 1

#: Private same-parent staging prefix; the final directory is renamed into place.
_STAGING_PREFIX = ".baseline_study.staging-"

#: Canonical panel ordering used for deterministic manifest entry order.
_POLICY_ORDER: dict[str, int] = {
    policy_id: index for index, policy_id in enumerate(REQUIRED_POLICY_IDS)
}

#: A validated study, or a path to a study YAML spec.
StudyInput = Union[BaselineStudyConfig, str, "os.PathLike[str]"]

#: Optional per-cell progress hook: ``(seed, policy_id) -> None``.
ProgressHook = Callable[[int, str], None]

#: Trace validation seam (defaults to the production strict reader).
TraceValidator = Callable[[str | os.PathLike], object]


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


def _manifest_entry(
    *,
    study: BaselineStudyConfig,
    policy: PanelPolicy,
    seed: int,
    config: ExperimentConfig,
    artifact_digest: str | None,
) -> dict:
    relative = _relative_trace_path(policy.id, seed)
    return {
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


def _build_manifest(
    *,
    study: BaselineStudyConfig,
    shared_input_digest: str,
    cells: tuple[BaselineCell, ...],
) -> dict:
    entries = sorted(
        (dict(cell.manifest_entry) for cell in cells),
        key=lambda entry: (
            int(entry["seed"]),
            _POLICY_ORDER[str(entry["policy_id"])],
        ),
    )
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "study_id": study.study_id,
        "data_provenance": study.data_provenance,
        "study_digest": study_digest(study),
        "shared_input_digest": shared_input_digest,
        "candidate_budget": study.candidate_budget,
        "selection_seed": study.selection_seed,
        "selection_provenance": study.selection_provenance,
        "panel": list(REQUIRED_POLICY_IDS),
        "seeds": sorted(int(seed) for seed in study.seeds),
        "entries": entries,
        "status": "complete",
    }


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
        staging = Path(
            tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=str(destination.parent))
        )
        os.chmod(staging, 0o700)
        staging_root = staging.resolve()
    except OSError as exc:
        raise EvaluationError(
            f"could not create study staging directory under "
            f"{destination.parent}: {exc}"
        ) from exc
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
            trace_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                resolved_trace = trace_dir.resolve()
            except OSError as exc:  # pragma: no cover - defensive
                raise EvaluationError(
                    f"could not resolve trace destination for policy "
                    f"{policy.id!r} seed {seed}: {exc}"
                ) from exc
            if not resolved_trace.is_relative_to(staging_root):
                raise EvaluationError(
                    f"trace destination for policy {policy.id!r} seed {seed} "
                    "escapes the staging root"
                )

            try:
                run_from_config(
                    config,
                    trace_path=trace_dir,
                    dataset=dataset,
                    dataset_loader=dataset_loader,
                    evaluation_dataset=evaluation_dataset,
                    evaluation_dataset_loader=evaluation_dataset_loader,
                )
            except EvaluationError:
                raise
            except Exception as exc:
                raise EvaluationError(
                    f"policy {policy.id!r} seed {seed} failed: {exc}"
                ) from exc

            try:
                with validator(trace_dir) as reader:
                    schema = getattr(reader, "schema_version", SCHEMA_VERSION)
                    if schema != SCHEMA_VERSION:
                        raise EvaluationError(
                            f"policy {policy.id!r} seed {seed} produced schema "
                            f"version {schema!r}, expected {SCHEMA_VERSION}"
                        )
            except EvaluationError:
                raise
            except (TrajectoryError, OSError) as exc:
                raise EvaluationError(
                    f"policy {policy.id!r} seed {seed} produced an invalid "
                    f"trace: {exc}"
                ) from exc

            artifact_digest = _artifact_digest(policy.aggregation, artifact_cache)
            entry = _manifest_entry(
                study=resolved,
                policy=policy,
                seed=seed,
                config=config,
                artifact_digest=artifact_digest,
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
        )
        _write_bytes_synced(staging / MANIFEST_FILENAME, manifest_bytes(manifest))
        _fsync_dir(staging)

        try:
            os.rename(staging, destination)
        except OSError as exc:
            raise EvaluationError(
                f"could not publish study output {destination}: {exc}"
            ) from exc
        published = True
        _fsync_dir(destination.parent)
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
    )
