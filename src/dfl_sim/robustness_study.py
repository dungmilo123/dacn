"""Canonical real-MNIST robustness matrix orchestration (M003/S05/T04, D040).

This module owns the **bounded raw matrix** the S05 canonical report consumes:
it builds on the frozen T02 protocol (:mod:`dfl_sim.robustness_config`) and the
existing production surfaces (``mnist_provision``, ``headline_verify``,
``dfl_sim calibrate``, and :func:`~dfl_sim.evaluation.run_held_out_study`) to

1. claim a fresh collision-safe output root and lease/deep-verify the
   content-digested MNIST cache;
2. independently reverify the separate IID headline root and pin its evidence
   and source identities;
3. derive a matched full/no-belief/no-strategic calibration triple **per alpha**
   (the normalized input triple is matched within an alpha, never across
   different alphas);
4. freeze and run nine held-out conditions (three Dirichlet alphas crossed with
   low/mid/high degree placement) with opt-in cost/decision telemetry; and
5. publish a strict, digest-covered completed raw matrix manifest only after
   every leaf source passes an independent audit.

The public real-MNIST entry point never accepts injected provenance or a
public canonical fallback: :func:`run_robustness_study` refuses an injected
protocol unless an explicit injected ``StudyRunner`` seam is supplied, and the
module CLI rejects injected protocols outright. This module deliberately does
**not** publish the canonical report or figures; T05/T06 own those surfaces.

Raw success marker (stdout, exactly once): :data:`ROBUSTNESS_RAW_MARKER`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from . import mnist_provision
from .ablation import matched_calibration_input_digest
from .ablation_evaluation import (
    AblationEvaluationError,
    AblationMatchSummary,
    preflight_ablation_study,
)
from .bayesian_artifact import (
    CALIBRATION_VARIANTS,
    FULL_VARIANT,
    BayesianArtifactError,
    load_bayesian_artifact,
)
from .calibration_config import (
    CalibrationConfig,
    CalibrationConfigError,
    load_calibration_config,
)
from .config import (
    MAX_ROUNDS,
    MAX_SAMPLES_PER_NODE,
    MAX_SEED,
    ConfigError,
    ExperimentConfig,
)
from .evaluation import (
    FINAL_MANIFEST_FILENAME,
    SELECTION_MANIFEST_FILENAME,
    EvaluationError,
    final_output_digest,
    run_held_out_study,
    selection_output_digest,
)
from .evaluation_config import (
    MAX_CANDIDATE_BUDGET,
    AblationArtifactPin,
    AblationStudyContract,
    BaselineStudyConfig,
    BaselineStudyConfigError,
    HeldOutStudyConfig,
    HeldOutStudyConfigError,
    NamedSchedule,
    held_out_study_digest,
    parse_held_out_study_text,
    selection_projection,
    selection_spec_digest,
    study_digest,
)
from .headline import StageOutcome, SubprocessStageRunner, build_panel
from .robustness_config import (
    CANONICAL_ALPHAS,
    CANONICAL_CALIBRATION_SEED,
    CANONICAL_FINAL_SEEDS,
    CANONICAL_PLACEMENTS,
    CANONICAL_PROTOCOL_ID,
    CANONICAL_SELECTION_SEEDS,
    CANONICAL_TOPOLOGY,
    CanonicalRobustnessProtocol,
    RobustnessConfigError,
)
from .topology import TopologyError

__all__ = [
    "CONDITIONS_DIRNAME",
    "GENERATED_CONFIGS_DIRNAME",
    "MATRIX_MANIFEST_FILENAME",
    "ROBUSTNESS_PROTOCOL_FILENAME",
    "ROBUSTNESS_PROTOCOL_VERSION",
    "ROBUSTNESS_RAW_MARKER",
    "InProcessStudyRunner",
    "MatrixRunResult",
    "RobustnessCalibrationSpec",
    "RobustnessDeps",
    "RobustnessProtocol",
    "RobustnessStudyError",
    "StudyRunner",
    "StudyStageOutcome",
    "alpha_token",
    "build_condition_study",
    "build_matrix_manifest",
    "build_parser",
    "canonical_protocol",
    "derive_calibration_config",
    "expected_condition_counts",
    "load_robustness_protocol",
    "main",
    "parse_robustness_text",
    "run_robustness_study",
    "validate_protocol",
]

#: Versioned source-protocol schema for the robustness matrix.
ROBUSTNESS_PROTOCOL_VERSION = "dfl_sim.robustness_protocol/v1"
#: Versioned completed raw matrix manifest schema.
MATRIX_SCHEMA = "dfl_robustness_matrix"
MATRIX_SCHEMA_VERSION = 1

ROBUSTNESS_PROTOCOL_FILENAME = "robustness-protocol.json"
MATRIX_MANIFEST_FILENAME = "matrix-manifest.json"
GENERATED_CONFIGS_DIRNAME = "generated-configs"
CONDITIONS_DIRNAME = "conditions"
CALIBRATION_DIRNAME = "calibration"
DIAGNOSTICS_DIRNAME = "diagnostics"
STUDY_DIRNAME = "study"

#: Stable stdout marker for a *raw* (unreported) completed matrix. It is
#: deliberately distinct from any canonical report success marker so a raw
#: matrix can never masquerade as a compiled report.
ROBUSTNESS_RAW_MARKER = "dfl_sim: robustness raw matrix "

DEFAULT_STAGE_TIMEOUT = 86_400.0
_MAX_ERROR_TEXT = 400


class RobustnessStudyError(RuntimeError):
    """Raised when the robustness matrix protocol or a stage fails closed."""


# --------------------------------------------------------------------------- #
# Strict source protocol
# --------------------------------------------------------------------------- #
def _as_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


SeedTuple = Annotated[tuple[int, ...], BeforeValidator(_as_tuple)]
FloatTuple = Annotated[tuple[float, ...], BeforeValidator(_as_tuple)]
ScheduleTuple = Annotated[tuple[NamedSchedule, ...], BeforeValidator(_as_tuple)]
StringTuple = Annotated[tuple[str, ...], BeforeValidator(_as_tuple)]


def _bounded(text: object, *, limit: int = _MAX_ERROR_TEXT) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) > limit:
        return collapsed[: limit - 1] + "\u2026"
    return collapsed


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class RobustnessCalibrationSpec(_StrictModel):
    """Source calibration contract shared by every alpha's triple derivation."""

    config: str = Field(min_length=1, max_length=1024)
    seed: int = Field(ge=0, le=MAX_SEED)
    episodes: int = Field(ge=2, le=512)
    design_split_size: int = Field(ge=1, le=8192)
    p0: float = Field(ge=0.0, le=1.0)
    mu_a: FloatTuple

    @model_validator(mode="after")
    def _pinned_mixture(self) -> "RobustnessCalibrationSpec":
        if tuple(float(value) for value in self.mu_a) != (0.0, 2.0, 5.0):
            raise ValueError("calibration.mu_a must be the pinned (0, 2, 5) mixture")
        return self


class RobustnessProtocol(_StrictModel):
    """Strict, frozen source envelope for the canonical robustness matrix.

    The scientific dimensions (alphas, placements, watts-strogatz family,
    resource envelope, seed split, schedules) are declared explicitly so the
    matrix is self-describing and an injected test twin can pin a reduced
    bounded matrix. For ``data_provenance == "real_mnist"`` every dimension is
    cross-checked against the frozen T02 canonical constants by
    :func:`validate_protocol`, so a hand-edited source file can never silently
    weaken the canonical matrix.
    """

    protocol_version: Literal["dfl_sim.robustness_protocol/v1"]
    study_id: str = Field(min_length=1, max_length=64)
    data_provenance: Literal["real_mnist", "injected"] = "real_mnist"
    headline_source_root: str = Field(min_length=1, max_length=512)
    data_root: str = Field(min_length=1, max_length=256)
    threads: int = Field(default=1, ge=1, le=1)
    alphas: FloatTuple
    placements: StringTuple
    topology_kind: Literal["watts_strogatz"]
    topology_n: int = Field(ge=10, le=20)
    topology_k: int = Field(ge=2)
    topology_p: float = Field(ge=0.0, le=1.0)
    rounds: int = Field(ge=1, le=MAX_ROUNDS)
    samples_per_node: int = Field(ge=1, le=MAX_SAMPLES_PER_NODE)
    evaluation_max_samples: int = Field(ge=1, le=10_000)
    evaluation_batch_size: int = Field(ge=1, le=1024)
    candidate_budget: int = Field(gt=0, le=MAX_CANDIDATE_BUDGET)
    selection_seed: int = Field(ge=0, le=MAX_SEED)
    selection_provenance: str = Field(min_length=1, max_length=256)
    selection_seeds: SeedTuple
    final_seeds: SeedTuple
    calibration_seed: int = Field(ge=0, le=MAX_SEED)
    known_schedules: ScheduleTuple
    held_out_schedule: NamedSchedule
    calibration: RobustnessCalibrationSpec

    @model_validator(mode="after")
    def _bounded_identity(self) -> "RobustnessProtocol":
        safe = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
        if not all(character in safe for character in self.study_id):
            raise ValueError(
                "study_id must contain only lowercase letters, digits, "
                "hyphens, and underscores"
            )
        return self


# --------------------------------------------------------------------------- #
# Parsing / validation
# --------------------------------------------------------------------------- #
def parse_robustness_text(
    text: str, *, source: str = "<string>"
) -> RobustnessProtocol:
    """Parse and strictly validate a robustness protocol from JSON or YAML."""

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RobustnessStudyError(
            f"invalid robustness config in {source}: {_bounded(exc)}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise RobustnessStudyError(
            f"robustness config in {source} must be a mapping, got "
            f"{type(raw).__name__}"
        )
    try:
        return RobustnessProtocol.model_validate(raw)
    except ValidationError as exc:
        raise RobustnessStudyError(
            f"invalid robustness protocol in {source}: {_bounded(exc)}"
        ) from exc


def load_robustness_protocol(path: str | os.PathLike[str]) -> RobustnessProtocol:
    """Read ``path`` and validate it as a :class:`RobustnessProtocol`."""

    source = os.fspath(path)
    try:
        text = Path(source).read_text(encoding="utf-8")
    except OSError as exc:
        raise RobustnessStudyError(
            f"could not read robustness config {source}: {_bounded(exc)}"
        ) from exc
    return parse_robustness_text(text, source=source)


def canonical_protocol(
    protocol: RobustnessProtocol, *, artifact_root: str
) -> CanonicalRobustnessProtocol:
    """Build the frozen T02 canonical envelope from the source protocol.

    Raises :class:`RobustnessStudyError` when the source envelope is
    structurally invalid (the canonical model enforces topology/seed/schedule
    bounds).
    """

    try:
        return CanonicalRobustnessProtocol(
            protocol_id=protocol.study_id,
            data_provenance=protocol.data_provenance,
            headline_source_root=protocol.headline_source_root,
            artifact_root=str(artifact_root),
            data_root=protocol.data_root,
            download=False,
            alphas=tuple(protocol.alphas),
            placements=tuple(protocol.placements),
            topology_kind=protocol.topology_kind,
            topology_n=int(protocol.topology_n),
            topology_k=int(protocol.topology_k),
            topology_p=float(protocol.topology_p),
            rounds=int(protocol.rounds),
            samples_per_node=int(protocol.samples_per_node),
            evaluation_max_samples=int(protocol.evaluation_max_samples),
            evaluation_batch_size=int(protocol.evaluation_batch_size),
            candidate_budget=int(protocol.candidate_budget),
            selection_seed=int(protocol.selection_seed),
            selection_provenance=protocol.selection_provenance,
            selection_seeds=tuple(protocol.selection_seeds),
            final_seeds=tuple(protocol.final_seeds),
            calibration_seed=int(protocol.calibration_seed),
            threads=int(protocol.threads),
            known_schedules=tuple(protocol.known_schedules),
            held_out_schedule=protocol.held_out_schedule,
        )
    except (ValidationError, RobustnessConfigError) as exc:
        raise RobustnessStudyError(
            f"invalid robustness protocol: {_bounded(exc)}"
        ) from exc


def validate_protocol(
    protocol: RobustnessProtocol, *, artifact_root: str = "<validate>"
) -> CanonicalRobustnessProtocol:
    """Fail closed unless every pinned canonical dimension holds.

    Structurally invalid envelopes are rejected by the canonical model. For a
    real-MNIST protocol the declared scientific dimensions are additionally
    cross-checked against the frozen T02 constants, so the source file cannot
    silently alter the canonical matrix.
    """

    canonical = canonical_protocol(protocol, artifact_root=artifact_root)

    if protocol.data_provenance != "real_mnist":
        return canonical

    failures: list[str] = []
    checks = (
        (protocol.study_id == CANONICAL_PROTOCOL_ID, "study_id"),
        (tuple(protocol.alphas) == CANONICAL_ALPHAS, "alphas"),
        (tuple(protocol.placements) == CANONICAL_PLACEMENTS, "placements"),
        (protocol.topology_n == CANONICAL_TOPOLOGY["n"], "topology_n"),
        (protocol.topology_k == CANONICAL_TOPOLOGY["k"], "topology_k"),
        (protocol.topology_p == CANONICAL_TOPOLOGY["p"], "topology_p"),
        (protocol.rounds == 6, "rounds"),
        (protocol.samples_per_node == 64, "samples_per_node"),
        (protocol.evaluation_max_samples == 128, "evaluation_max_samples"),
        (protocol.evaluation_batch_size == 32, "evaluation_batch_size"),
        (protocol.candidate_budget == 1, "candidate_budget"),
        (protocol.selection_seed == 1101, "selection_seed"),
        (
            tuple(protocol.selection_seeds) == tuple(CANONICAL_SELECTION_SEEDS),
            "selection_seeds",
        ),
        (tuple(protocol.final_seeds) == tuple(CANONICAL_FINAL_SEEDS), "final_seeds"),
        (protocol.calibration_seed == CANONICAL_CALIBRATION_SEED, "calibration_seed"),
        (protocol.threads == 1, "threads"),
    )
    for ok, name in checks:
        if not ok:
            failures.append(name)
    if failures:
        raise RobustnessStudyError(
            "invalid canonical robustness protocol: " + ", ".join(failures)
        )
    return canonical


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def alpha_token(alpha: float) -> str:
    """A study-id-safe token for one alpha (``0.1`` -> ``a0p1``)."""

    text = f"{float(alpha):g}"
    return "a" + text.replace("-", "m").replace(".", "p")


def derive_calibration_config(
    template: CalibrationConfig, alpha: float
) -> CalibrationConfig:
    """Derive one alpha's calibration config from the shared source template.

    Only the embedded experiment's data partition/alpha change; the data mode,
    quota, model, features, action space, bins, budget and calibration envelope
    are byte-for-byte the template's, so the normalized input triple is matched
    within an alpha. The result is re-validated through the strict calibration
    contract, so a bad alpha fails closed before any episode.
    """

    if not isinstance(template, CalibrationConfig):
        raise RobustnessStudyError(
            "derive_calibration_config requires a validated CalibrationConfig, "
            f"got {type(template).__name__}"
        )
    payload = template.model_dump(mode="json")
    data = payload["experiment"]["data"]
    data["partition"] = "dirichlet"
    data["dirichlet_alpha"] = float(alpha)
    try:
        return CalibrationConfig.model_validate(payload)
    except ValidationError as exc:
        raise RobustnessStudyError(
            f"derived calibration config for alpha {alpha!r} is invalid: "
            f"{_bounded(exc)}"
        ) from exc


def expected_condition_counts(study: HeldOutStudyConfig) -> dict[str, int]:
    """Exact ``(selection_cells, final_cells, aggregates)`` for one condition."""

    panel = len(study.study.policy_ids())
    selection = len(study.study.seeds) * len(study.known_schedules) * panel
    schedules = len(study.known_schedules) + 1
    final = len(study.final_seeds) * schedules * panel
    return {
        "selection_cells": int(selection),
        "final_cells": int(final),
        "aggregates": int(schedules * panel),
    }


def _artifact_derivation(artifact: object, variant: str):
    manifest = getattr(artifact, "manifest", None)
    derivation = getattr(manifest, "derivation", None)
    if derivation is None:
        raise RobustnessStudyError(
            f"{variant} calibration artifact has no calibration derivation; "
            "refusing to pin an unproven artifact"
        )
    return derivation


def _artifact_aggregate(artifact: object, variant: str) -> str:
    digest = getattr(getattr(artifact, "manifest", None), "aggregate_digest", None)
    if not isinstance(digest, str) or not digest:
        raise RobustnessStudyError(
            f"{variant} calibration artifact lacks an aggregate digest"
        )
    return digest


def build_condition_study(
    canonical: CanonicalRobustnessProtocol,
    protocol: RobustnessProtocol,
    alpha: float,
    placement: str,
    *,
    artifact_root: str | os.PathLike[str],
    artifacts: Mapping[str, object] | None = None,
) -> HeldOutStudyConfig:
    """Build one frozen condition study.

    Without ``artifacts`` the T02 six-policy placed condition is returned
    unchanged (used only by the injected fixture path, which must not require a
    real artifact triple). With a matched ``artifacts`` mapping the baseline is
    rebuilt with the eight-policy panel pointing at the per-alpha calibration
    directories and a digest-pinned :class:`AblationStudyContract`.
    """

    base_condition = canonical.condition_spec(alpha, placement)
    if not artifacts:
        return base_condition

    artifact_root_path = Path(artifact_root)
    artifact_dirs = {
        variant: artifact_root_path / CALIBRATION_DIRNAME / alpha_token(alpha) / variant
        for variant in CALIBRATION_VARIANTS
    }
    for variant in CALIBRATION_VARIANTS:
        if variant not in artifacts:
            raise RobustnessStudyError(f"missing {variant!r} calibration artifact")

    panel = build_panel(
        artifact_dirs,
        candidate_budget=protocol.candidate_budget,
        selection_seed=protocol.selection_seed,
        selection_provenance=protocol.selection_provenance,
    )
    pins = tuple(
        AblationArtifactPin(
            variant=variant,
            expected_aggregate_digest=_artifact_aggregate(artifacts[variant], variant),
        )
        for variant in CALIBRATION_VARIANTS
    )
    contract = AblationStudyContract(
        matched_calibration_input_digest=matched_calibration_input_digest(
            _artifact_derivation(artifacts[FULL_VARIANT], FULL_VARIANT)
        ),
        expected_aggregate_digests=pins,
    )
    payload = base_condition.study.model_dump(mode="json")
    payload["panel"] = [entry.model_dump(mode="json") for entry in panel]
    payload["ablations"] = contract.model_dump(mode="json")
    try:
        study = BaselineStudyConfig.model_validate(payload)
    except ValidationError as exc:
        raise RobustnessStudyError(
            f"rebuilt eight-policy condition study is invalid: {_bounded(exc)}"
        ) from exc
    return HeldOutStudyConfig(
        study=study,
        final_seeds=base_condition.final_seeds,
        known_schedules=base_condition.known_schedules,
        held_out_schedule=base_condition.held_out_schedule,
    )


# --------------------------------------------------------------------------- #
# Stage runners
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StudyStageOutcome:
    """Bounded diagnostic outcome of one condition's held-out study."""

    label: str
    selection_cells: int
    final_cells: int
    reused_final_cells: int
    duration_ms: int


class StudyRunner(Protocol):
    """Injection seam for one condition's production held-out study run."""

    def run(
        self,
        study: HeldOutStudyConfig,
        destination: str | os.PathLike[str],
        diagnostics_dir: Path,
        *,
        resume_from: str | os.PathLike[str] | None = None,
        label: str = "study",
    ) -> StudyStageOutcome: ...


class InProcessStudyRunner:
    """Run the unpatched production ``selection -> freeze -> final`` in-process.

    The production public surfaces are called directly (no dataset/loader/policy
    injection when constructed with no fixture arguments) and opt-in telemetry
    is enabled so the canonical report can separate measured computation and
    logical communication. CPU/BLAS thread pools are capped at ``threads``.
    """

    def __init__(
        self,
        *,
        threads: int = 1,
        dataset: object | None = None,
        dataset_loader: object | None = None,
        evaluation_dataset: object | None = None,
        evaluation_dataset_loader: object | None = None,
    ) -> None:
        self._threads = int(threads)
        self._dataset = dataset
        self._dataset_loader = dataset_loader
        self._evaluation_dataset = evaluation_dataset
        self._evaluation_dataset_loader = evaluation_dataset_loader

    def run(
        self,
        study: HeldOutStudyConfig,
        destination: str | os.PathLike[str],
        diagnostics_dir: Path,
        *,
        resume_from: str | os.PathLike[str] | None = None,
        label: str = "study",
    ) -> StudyStageOutcome:
        self._cap_threads()
        start = time.monotonic()
        try:
            result = run_held_out_study(
                study,
                destination,
                dataset=self._dataset,
                dataset_loader=self._dataset_loader,
                evaluation_dataset=self._evaluation_dataset,
                evaluation_dataset_loader=self._evaluation_dataset_loader,
                resume_from=resume_from,
                collect_telemetry=True,
            )
        except EvaluationError as exc:
            _write_diagnostics(
                diagnostics_dir, f"study-{label}", [], detail=_bounded(exc)
            )
            raise RobustnessStudyError(
                f"condition {label!r} study failed closed: {_bounded(exc)}"
            ) from exc
        duration_ms = int((time.monotonic() - start) * 1000)
        final_entries = tuple(result.final_result.manifest.get("entries", ()))
        reused = sum(1 for entry in final_entries if entry.get("reused"))
        _write_diagnostics(
            diagnostics_dir,
            f"study-{label}",
            [],
            returncode=0,
            duration_ms=duration_ms,
        )
        return StudyStageOutcome(
            label=label,
            selection_cells=len(result.selection_result.cells),
            final_cells=len(result.final_result.cells),
            reused_final_cells=int(reused),
            duration_ms=duration_ms,
        )

    def _cap_threads(self) -> None:
        try:  # pragma: no cover - torch is always present in production
            import torch

            torch.set_num_threads(self._threads)
        except Exception:  # noqa: BLE001 - best effort thread cap
            pass


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _write_diagnostics(
    diagnostics_dir: Path,
    label: str,
    command: Sequence[str],
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int | None = None,
    duration_ms: int | None = None,
    detail: str | None = None,
) -> None:
    try:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": label,
            "argv": list(command),
            "returncode": returncode,
            "duration_ms": duration_ms,
            "detail": detail,
            "stdout_tail": stdout,
            "stderr_tail": stderr,
        }
        path = Path(diagnostics_dir) / f"{label}.json"
        path.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Injected dependencies
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RobustnessDeps:
    """Optional injection seams; production uses the real public surfaces."""

    provision: Callable[..., object] | None = None
    cache_lease: Callable[..., Any] | None = None
    load_calibration_config: Callable[..., object] | None = None
    calibrate: Callable[..., StageOutcome] | None = None
    load_artifact: Callable[..., object] | None = None
    preflight: Callable[..., AblationMatchSummary] | None = None
    verify_headline: Callable[..., object] | None = None
    verify_matrix: Callable[..., object] | None = None
    study_runner: StudyRunner | None = None
    study_builder: (
        Callable[
            [CanonicalRobustnessProtocol, RobustnessProtocol, float, str],
            HeldOutStudyConfig,
        ]
        | None
    ) = None
    repo_root: str | os.PathLike[str] | None = None


@dataclass(frozen=True)
class MatrixRunResult:
    """Bounded result of one raw robustness matrix publication."""

    output_path: Path
    protocol_digest: str
    conditions: tuple[str, ...]
    selection_cells: int
    final_cells: int
    stages: tuple[StudyStageOutcome, ...] = field(default_factory=tuple)
    evidence_path: Path | None = None


# --------------------------------------------------------------------------- #
# Output / path safety
# --------------------------------------------------------------------------- #
def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(text: str, base: Path) -> Path:
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate


def _paths_alias(first: Path, second: Path) -> bool:
    try:
        return os.path.realpath(first) == os.path.realpath(second)
    except OSError:  # pragma: no cover - best effort
        return False


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child_real = os.path.realpath(child)
        parent_real = os.path.realpath(parent)
    except OSError:  # pragma: no cover - best effort
        return False
    return child_real == parent_real or child_real.startswith(parent_real + os.sep)


def _check_output_destination(destination: Path, config_path: Path) -> None:
    if destination.name in ("", ".", ".."):
        raise RobustnessStudyError(f"invalid output destination: {destination}")
    parent = destination.parent
    if not parent.is_dir():
        raise RobustnessStudyError(f"output parent does not exist: {parent!r}")
    if parent.is_symlink():
        raise RobustnessStudyError(f"output parent must not be a symlink: {parent!r}")
    if os.path.lexists(destination):
        raise RobustnessStudyError(f"output already exists: {destination!r}")
    if _paths_alias(destination, config_path):
        raise RobustnessStudyError("output aliases the robustness config path")


def _check_source_roots(
    destination: Path,
    *,
    data_root: Path,
    headline_root: Path,
) -> None:
    """Paths outside the matrix output root that must not alias/traverse it."""

    for label, source in (("data", data_root), ("headline", headline_root)):
        if (
            _paths_alias(destination, source)
            or _is_within(destination, source)
            or _is_within(source, destination)
        ):
            raise RobustnessStudyError(
                f"output root must not alias, contain, or be contained by the "
                f"{label} root"
            )


# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #
def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def protocol_digest(protocol: RobustnessProtocol) -> str:
    """Environment-independent digest of the resolved source protocol."""

    return _sha256_json(protocol.model_dump(mode="json"))


def _write_bytes(path: Path, payload: bytes) -> None:
    try:
        with open(path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise RobustnessStudyError(
            f"could not write {path}: {_bounded(exc)}"
        ) from exc


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _write_bytes(
        path,
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


def _write_text(path: Path, text: str) -> None:
    _write_bytes(path, text.encode("utf-8"))


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise RobustnessStudyError(f"missing robustness artifact: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobustnessStudyError(
            f"could not read {path}: {_bounded(exc)}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise RobustnessStudyError(f"{path} is not a JSON object")
    return raw


# --------------------------------------------------------------------------- #
# Matrix manifest
# --------------------------------------------------------------------------- #
def build_matrix_manifest(
    *,
    protocol: RobustnessProtocol,
    protocol_digest_value: str,
    headline: Mapping[str, object],
    cache: Mapping[str, object] | None,
    calibration: Mapping[str, object],
    conditions: Sequence[Mapping[str, object]],
    totals: Mapping[str, int],
) -> dict:
    """The strict digest-covered completed raw matrix manifest payload."""

    manifest: dict = {
        "schema": MATRIX_SCHEMA,
        "schema_version": MATRIX_SCHEMA_VERSION,
        "status": "complete",
        "study_id": protocol.study_id,
        "data_provenance": protocol.data_provenance,
        "protocol_digest": protocol_digest_value,
        "protocol": protocol.model_dump(mode="json"),
        "headline": dict(headline),
        "calibration": {key: dict(value) for key, value in calibration.items()},
        "conditions": [dict(condition) for condition in conditions],
        "totals": {key: int(value) for key, value in totals.items()},
    }
    if cache is not None:
        manifest["cache"] = dict(cache)
    manifest["matrix_digest"] = _sha256_json(manifest)
    return manifest


# --------------------------------------------------------------------------- #
# Calibration contract cross-check
# --------------------------------------------------------------------------- #
def _topology_signature(topology: object) -> tuple:
    kind = getattr(topology, "kind")
    if kind == "chain3":
        return ("chain3",)
    return ("watts_strogatz", int(topology.n), int(topology.k), float(topology.p))


def _check_calibration_contract(
    protocol: RobustnessProtocol,
    canonical: CanonicalRobustnessProtocol,
    calibration: CalibrationConfig,
    calibration_path: Path,
) -> None:
    """Cross-check the source calibration template against the first alpha."""

    alpha = float(protocol.alphas[0])
    base = canonical.condition_spec(alpha, str(protocol.placements[0])).study.base
    experiment = calibration.experiment
    data = experiment.data
    training = experiment.training
    evaluation = experiment.evaluation

    failures: list[str] = []
    checks = (
        (experiment.device == "cpu", "device"),
        (experiment.bitwise is True, "bitwise"),
        (experiment.model.name == base.model.name, "model"),
        (data.dataset == base.data.dataset, "dataset"),
        (data.partition == "dirichlet", "partition"),
        (float(data.dirichlet_alpha or 0.0) == alpha, "dirichlet_alpha"),
        (data.samples_per_node == base.data.samples_per_node, "samples_per_node"),
        (data.data_root == base.data.data_root, "data_root"),
        (data.download is False, "download"),
        (training.batch_size == base.training.batch_size, "batch_size"),
        (training.local_epochs == base.training.local_epochs, "local_epochs"),
        (training.num_workers == base.training.num_workers, "num_workers"),
        (
            training.learning_rate == base.training.learning_rate,
            "learning_rate",
        ),
        (evaluation.max_samples == base.evaluation.max_samples, "max_samples"),
        (evaluation.batch_size == base.evaluation.batch_size, "evaluation_batch_size"),
        (_topology_signature(experiment.topology) == _topology_signature(base.topology), "topology"),
        (experiment.rounds == 1, "rounds"),
        (experiment.seed == protocol.calibration.seed, "seed"),
        (
            int(getattr(calibration, "episodes")) == protocol.calibration.episodes,
            "episodes",
        ),
        (
            int(getattr(calibration, "design_split_size"))
            == protocol.calibration.design_split_size,
            "design_split_size",
        ),
        (float(getattr(calibration, "p0")) == protocol.calibration.p0, "p0"),
        (
            tuple(float(x) for x in getattr(calibration, "mu_a"))
            == tuple(protocol.calibration.mu_a),
            "mu_a",
        ),
    )
    for ok, name in checks:
        if not ok:
            failures.append(name)
    if failures:
        raise RobustnessStudyError(
            f"calibration contract mismatch for {calibration_path}: "
            + ", ".join(failures)
        )


# --------------------------------------------------------------------------- #
# Leaf publication validation
# --------------------------------------------------------------------------- #
def _validate_condition_counts(
    condition_dir: Path, study: HeldOutStudyConfig, *, label: str
) -> dict[str, int]:
    expected = expected_condition_counts(study)
    selection = _read_json(condition_dir / STUDY_DIRNAME / "selection" / SELECTION_MANIFEST_FILENAME)
    final = _read_json(condition_dir / STUDY_DIRNAME / "final" / FINAL_MANIFEST_FILENAME)
    if selection.get("schema") != "dfl_held_out_selection":
        raise RobustnessStudyError(f"{label} selection manifest has an unexpected schema")
    if final.get("schema") != "dfl_held_out_study_manifest":
        raise RobustnessStudyError(f"{label} final manifest has an unexpected schema")
    if selection.get("status") != "frozen":
        raise RobustnessStudyError(f"{label} selection manifest is not frozen")
    if final.get("status") != "complete":
        raise RobustnessStudyError(f"{label} final manifest is not complete")
    if len(selection.get("entries", [])) != expected["selection_cells"]:
        raise RobustnessStudyError(f"{label} selection cell count is wrong")
    if len(final.get("entries", [])) != expected["final_cells"]:
        raise RobustnessStudyError(f"{label} final cell count is wrong")
    if selection.get("study_digest") != study_digest(study.study):
        raise RobustnessStudyError(f"{label} selection study digest mismatch")
    if final.get("held_out_study_digest") != held_out_study_digest(study):
        raise RobustnessStudyError(f"{label} held-out study digest mismatch")
    if final.get("selection_spec_digest") != selection_spec_digest(
        selection_projection(study)
    ):
        raise RobustnessStudyError(f"{label} selection spec digest mismatch")
    return expected


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #
def run_robustness_study(
    config_path: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    data_root: str | os.PathLike[str] | None = None,
    download: bool = False,
    timeout: float = DEFAULT_STAGE_TIMEOUT,
    deps: RobustnessDeps | None = None,
    resume_from: str | os.PathLike[str] | None = None,
) -> MatrixRunResult:
    """Run the bounded real-MNIST robustness matrix end to end.

    The calibrated per-alpha triples and nine held-out conditions are written
    under a fresh output root and atomically validated on disk; the independent
    matrix audit runs last and, on failure, removes the completed matrix
    manifest so a partial or mismatched study can never publish a raw success.
    """

    deps = deps or RobustnessDeps()
    protocol = load_robustness_protocol(config_path)
    config_abs = Path(os.path.abspath(os.fspath(config_path)))
    destination = Path(os.path.abspath(os.fspath(output)))
    canonical = validate_protocol(protocol, artifact_root=str(destination))

    if protocol.data_provenance == "injected" and deps.study_runner is None:
        raise RobustnessStudyError(
            "injected provenance requires an explicit injected StudyRunner "
            "seam; the public canonical path is real-MNIST only"
        )

    repo_root = Path(deps.repo_root) if deps.repo_root is not None else _default_repo_root()
    resolved_data_root = Path(
        os.path.abspath(os.fspath(data_root or protocol.data_root))
    )
    headline_root = _resolve_path(protocol.headline_source_root, repo_root)

    _check_output_destination(destination, config_abs)
    _check_source_roots(
        destination, data_root=resolved_data_root, headline_root=headline_root
    )

    resume_root = (
        Path(os.path.abspath(os.fspath(resume_from))) if resume_from is not None else None
    )
    _check_resume_protocol(resume_root, protocol)

    if protocol.data_provenance == "real_mnist":
        provision = deps.provision or mnist_provision.provision
        try:
            cache_manifest = provision(str(resolved_data_root), download=download)
        except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
            raise RobustnessStudyError(
                f"MNIST provisioning failed closed: {_bounded(exc)}"
            ) from exc
        lease_factory = deps.cache_lease or mnist_provision.cache_lease
        lease_context = lease_factory(
            str(resolved_data_root), exclusive=True, deep=True
        )
    else:
        cache_manifest = None
        lease_context = _nullcontext()

    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:  # pragma: no cover - checked above
        raise RobustnessStudyError(f"output already exists: {destination!r}") from exc
    except OSError as exc:
        raise RobustnessStudyError(
            f"could not create output root {destination!r}: {_bounded(exc)}"
        ) from exc
    diagnostics = destination / DIAGNOSTICS_DIRNAME
    diagnostics.mkdir(parents=True, exist_ok=True)
    generated = destination / GENERATED_CONFIGS_DIRNAME
    generated.mkdir(parents=True, exist_ok=True)

    stages: list[StudyStageOutcome] = []
    try:
        with lease_context:
            headline = _verify_headline(
                deps, headline_root, protocol, write=False
            )
            _write_json(
                destination / ROBUSTNESS_PROTOCOL_FILENAME,
                {
                    "schema": ROBUSTNESS_PROTOCOL_VERSION,
                    "source": str(config_path),
                    "protocol_digest": protocol_digest(protocol),
                    "study_id": protocol.study_id,
                    "data_provenance": protocol.data_provenance,
                    "protocol": protocol.model_dump(mode="json"),
                    "headline": dict(headline),
                    "cache": _cache_block(protocol, cache_manifest),
                },
            )

            include_ablations = protocol.data_provenance == "real_mnist"
            calibration_block = _prepare_calibrations(
                deps,
                protocol,
                canonical,
                repo_root,
                destination,
                resume_root,
                diagnostics,
                include_ablations=include_ablations,
            )
            study_builder = deps.study_builder
            load_artifact = deps.load_artifact or load_bayesian_artifact
            conditions_meta: list[dict] = []
            for condition in canonical.conditions():
                condition_id = condition.condition_id
                condition_root = destination / CONDITIONS_DIRNAME / condition_id
                condition_root.mkdir(parents=True, exist_ok=True)
                artifacts = (
                    _alpha_artifacts(destination, condition.alpha, load_artifact)
                    if include_ablations
                    else None
                )
                if study_builder is not None:
                    study = study_builder(
                        canonical, protocol, condition.alpha, condition.placement
                    )
                else:
                    study = build_condition_study(
                        canonical,
                        protocol,
                        condition.alpha,
                        condition.placement,
                        artifact_root=destination,
                        artifacts=artifacts,
                    )
                if include_ablations:
                    preflight = deps.preflight or preflight_ablation_study
                    try:
                        preflight(study.study)
                    except AblationEvaluationError as exc:
                        raise RobustnessStudyError(
                            f"condition {condition_id} matched-artifact preflight "
                            f"failed closed: {_bounded(exc)}"
                        ) from exc

                frozen_study_path = condition_root / "frozen-study.yaml"
                _write_text(
                    frozen_study_path,
                    yaml.safe_dump(study.model_dump(mode="json"), sort_keys=True),
                )

                prior_study = (
                    resume_root
                    / CONDITIONS_DIRNAME
                    / condition_id
                    / STUDY_DIRNAME
                    if resume_root is not None
                    else None
                )
                runner = deps.study_runner or InProcessStudyRunner(
                    threads=protocol.threads
                )
                outcome = runner.run(
                    study,
                    condition_root / STUDY_DIRNAME,
                    diagnostics,
                    resume_from=prior_study if _is_dir(prior_study) else None,
                    label=condition_id,
                )
                stages.append(outcome)
                counts = _validate_condition_counts(
                    condition_root, study, label=condition_id
                )
                conditions_meta.append(
                    _condition_meta(
                        condition_id=condition_id,
                        alpha=condition.alpha,
                        placement=condition.placement,
                        study=study,
                        counts=counts,
                        condition_root=condition_root,
                        outcome=outcome,
                    )
                )

            totals = {
                "conditions": len(conditions_meta),
                "selection_cells": sum(int(c["selection_cells"]) for c in conditions_meta),
                "final_cells": sum(int(c["final_cells"]) for c in conditions_meta),
                "aggregates": sum(int(c["aggregates"]) for c in conditions_meta),
            }
            manifest = build_matrix_manifest(
                protocol=protocol,
                protocol_digest_value=protocol_digest(protocol),
                headline=headline,
                cache=_cache_block(protocol, cache_manifest),
                calibration=calibration_block,
                conditions=conditions_meta,
                totals=totals,
            )
            _write_json(destination / MATRIX_MANIFEST_FILENAME, manifest)

            try:
                if deps.verify_matrix is not None:
                    verify_result = deps.verify_matrix(
                        str(destination), write=True, deep_cache=False
                    )
                else:
                    from . import robustness_verify

                    verify_deps = robustness_verify.RobustnessVerifyDeps(
                        headline_verify=deps.verify_headline,
                        load_artifact=deps.load_artifact,
                        study_builder=deps.study_builder,
                        repo_root=deps.repo_root,
                    )
                    verify_result = robustness_verify.verify_robustness_study(
                        str(destination), deps=verify_deps, write=True, deep_cache=False
                    )
            except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
                _remove_if_exists(destination / MATRIX_MANIFEST_FILENAME)
                raise RobustnessStudyError(
                    f"independent matrix audit failed closed: {_bounded(exc)}"
                ) from exc
    except RobustnessStudyError:
        raise
    except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
        raise RobustnessStudyError(
            f"robustness matrix stage failed closed: {_bounded(exc)}"
        ) from exc

    evidence_path = Path(
        getattr(verify_result, "evidence_path", None)
        or (destination / "matrix-evidence.json")
    )
    if not evidence_path.is_file():
        evidence_path = None

    return MatrixRunResult(
        output_path=destination,
        protocol_digest=protocol_digest(protocol),
        conditions=tuple(meta["condition_id"] for meta in conditions_meta),
        selection_cells=totals["selection_cells"],
        final_cells=totals["final_cells"],
        stages=tuple(stages),
        evidence_path=evidence_path,
    )


# --------------------------------------------------------------------------- #
# Orchestration helpers
# --------------------------------------------------------------------------- #
class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc_info) -> bool:
        return False


def _is_dir(path: Path | None) -> bool:
    return path is not None and path.is_dir()


def _remove_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:  # pragma: no cover - best effort cleanup
        pass


def _check_resume_protocol(
    resume_root: Path | None, protocol: RobustnessProtocol
) -> None:
    if resume_root is None:
        return
    if not resume_root.is_dir():
        raise RobustnessStudyError(f"resume root not found: {resume_root!r}")
    frozen = _read_json(resume_root / ROBUSTNESS_PROTOCOL_FILENAME)
    stored = frozen.get("protocol_digest")
    current = protocol_digest(protocol)
    if stored != current:
        raise RobustnessStudyError(
            "resume protocol digest mismatch: prior matrix "
            f"{_bounded(stored)} does not match current {_bounded(current)}"
        )


def _cache_block(
    protocol: RobustnessProtocol, cache_manifest: object | None
) -> dict:
    if cache_manifest is None:
        return {"data_root": protocol.data_root, "data_provenance": protocol.data_provenance}
    return {
        "data_root": protocol.data_root,
        "manifest_digest": getattr(cache_manifest, "manifest_digest", None),
        "preprocessing_digest": getattr(cache_manifest, "preprocessing_digest", None),
        "preprocessing_id": getattr(cache_manifest, "preprocessing_id", None),
    }


def _verify_headline(
    deps: RobustnessDeps,
    headline_root: Path,
    protocol: RobustnessProtocol,
    *,
    write: bool,
) -> dict:
    if protocol.data_provenance != "real_mnist":
        return {"data_provenance": "injected", "verified": False}
    if not headline_root.is_dir():
        raise RobustnessStudyError(f"headline root not found: {headline_root!r}")
    verify = deps.verify_headline
    if verify is None:
        from . import headline_verify

        verify = headline_verify.verify_headline
    try:
        result = verify(str(headline_root), write=write, deep_cache=False)
    except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
        raise RobustnessStudyError(
            f"headline verification failed closed: {_bounded(exc)}"
        ) from exc
    evidence = getattr(result, "evidence", None)
    return {
        "data_provenance": "real_mnist",
        "source_root": str(headline_root),
        "protocol_digest": getattr(result, "protocol_digest", None),
        "evidence_digest": (
            evidence.get("evidence_digest") if isinstance(evidence, Mapping) else None
        ),
        "scope": "iid-chain3",
        "selection_cells": int(getattr(result, "selection_cells", 0)),
        "final_cells": int(getattr(result, "final_cells", 0)),
        "aggregate_count": int(getattr(result, "aggregate_count", 0)),
    }


def _prepare_calibrations(
    deps: RobustnessDeps,
    protocol: RobustnessProtocol,
    canonical: CanonicalRobustnessProtocol,
    repo_root: Path,
    destination: Path,
    resume_root: Path | None,
    diagnostics: Path,
    *,
    include_ablations: bool,
) -> dict:
    if not include_ablations:
        return {}
    load_calibration = deps.load_calibration_config or load_calibration_config
    calibration_path = _resolve_path(protocol.calibration.config, repo_root)
    try:
        template = load_calibration(str(calibration_path))
    except (CalibrationConfigError, ConfigError) as exc:
        raise RobustnessStudyError(
            f"could not load calibration config {calibration_path}: {_bounded(exc)}"
        ) from exc
    _check_calibration_contract(protocol, canonical, template, calibration_path)

    calibrate = deps.calibrate
    if calibrate is None:
        stage_runner = SubprocessStageRunner(
            repo_root=repo_root, threads=protocol.threads
        )
        calibrate = stage_runner.calibrate
    load_artifact = deps.load_artifact or load_bayesian_artifact

    block: dict[str, object] = {}
    for alpha in protocol.alphas:
        token = alpha_token(alpha)
        derived = derive_calibration_config(template, alpha)
        config_path = destination / GENERATED_CONFIGS_DIRNAME / f"calibrate-{token}.yaml"
        _write_text(config_path, yaml.safe_dump(derived.model_dump(mode="json"), sort_keys=True))
        variants: dict[str, object] = {}
        for variant in CALIBRATION_VARIANTS:
            variant_dest = destination / CALIBRATION_DIRNAME / token / variant
            prior = (
                resume_root / CALIBRATION_DIRNAME / token / variant
                if resume_root is not None
                else None
            )
            reused = _try_reuse_calibration(prior, variant_dest, load_artifact)
            if not reused:
                outcome = calibrate(
                    str(config_path), str(variant_dest), variant, diagnostics
                )
                if getattr(outcome, "returncode", 0) != 0:
                    raise RobustnessStudyError(
                        f"calibration {token}/{variant} failed closed"
                    )
            artifact = load_artifact(str(variant_dest))
            variants[variant] = {
                "path": f"{CALIBRATION_DIRNAME}/{token}/{variant}",
                "aggregate_digest": _artifact_aggregate(artifact, variant),
                "matched_calibration_input_digest": matched_calibration_input_digest(
                    _artifact_derivation(artifact, variant)
                ),
            }
        block[token] = {
            "alpha": float(alpha),
            "config_path": f"{GENERATED_CONFIGS_DIRNAME}/calibrate-{token}.yaml",
            "config_digest": _sha256_json(derived.model_dump(mode="json")),
            "variants": variants,
        }
    return block


def _try_reuse_calibration(
    prior: Path | None,
    destination: Path,
    load_artifact: Callable[..., object],
) -> bool:
    """Copy a prior manifest-matched calibration artifact into the new root."""

    if prior is None or not prior.is_dir():
        return False
    try:
        load_artifact(str(prior))
    except (BayesianArtifactError, OSError):
        return False
    try:
        import shutil

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(prior, destination)
    except OSError:
        return False
    try:
        load_artifact(str(destination))
    except (BayesianArtifactError, OSError):
        return False
    return True


def _alpha_artifacts(
    destination: Path,
    alpha: float,
    loader: Callable[..., object],
) -> dict[str, object]:
    token = alpha_token(alpha)
    artifacts: dict[str, object] = {}
    for variant in CALIBRATION_VARIANTS:
        path = destination / CALIBRATION_DIRNAME / token / variant
        artifacts[variant] = loader(str(path))
    return artifacts


def _condition_meta(
    *,
    condition_id: str,
    alpha: float,
    placement: str,
    study: HeldOutStudyConfig,
    counts: Mapping[str, int],
    condition_root: Path,
    outcome: StudyStageOutcome,
) -> dict:
    study_dir = condition_root / STUDY_DIRNAME
    selection_manifest = _read_json(
        study_dir / "selection" / SELECTION_MANIFEST_FILENAME
    )
    final_manifest = _read_json(study_dir / "final" / FINAL_MANIFEST_FILENAME)
    return {
        "condition_id": condition_id,
        "alpha": float(alpha),
        "placement": placement,
        "frozen_study_digest": _sha256_bytes(
            (condition_root / "frozen-study.yaml").read_bytes()
        ),
        "study_path": f"{CONDITIONS_DIRNAME}/{condition_id}/{STUDY_DIRNAME}",
        "study_digest": study_digest(study.study),
        "selection_spec_digest": selection_spec_digest(selection_projection(study)),
        "held_out_study_digest": held_out_study_digest(study),
        "selection_cells": counts["selection_cells"],
        "final_cells": counts["final_cells"],
        "aggregates": counts["aggregates"],
        "selection_output_digest": selection_output_digest(selection_manifest),
        "final_output_digest": final_output_digest(final_manifest),
        "selection_manifest_digest": _sha256_json(selection_manifest),
        "final_manifest_digest": _sha256_json(final_manifest),
        "reused_final_cells": outcome.reused_final_cells,
        "duration_ms": outcome.duration_ms,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dfl_sim.robustness_study",
        description=(
            "Run the bounded real-MNIST robustness matrix and publish a raw, "
            "independently audited matrix manifest. This emits a raw success "
            "marker only; it never compiles the canonical report (T05/T06)."
        ),
    )
    parser.add_argument("config", help="Robustness source-protocol YAML/JSON path.")
    parser.add_argument(
        "--output", required=True, help="Fresh output root for the raw matrix."
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Verified MNIST cache root (defaults to the protocol data_root).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Permit provisioning to download missing MNIST (default: verify only).",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="A prior matrix root whose manifest-matched cells may be reused.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_STAGE_TIMEOUT,
        help=f"Per-calibration wall-clock bound in seconds (default {DEFAULT_STAGE_TIMEOUT:g}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_robustness_protocol(args.config)
        if protocol.data_provenance != "real_mnist":
            raise RobustnessStudyError(
                "the public robustness command accepts real_mnist provenance only"
            )
        result = run_robustness_study(
            args.config,
            args.output,
            data_root=args.data_root,
            download=bool(args.download),
            timeout=float(args.timeout),
            resume_from=args.resume_from,
        )
    except (RobustnessStudyError, ConfigError, TopologyError) as exc:
        print(f"dfl_sim: error: {_bounded(exc)}", file=sys.stderr)
        return 1
    print(
        f"{ROBUSTNESS_RAW_MARKER}"
        f"root={result.output_path} "
        f"conditions={len(result.conditions)} "
        f"selection_cells={result.selection_cells} "
        f"final_cells={result.final_cells} "
        f"raw=true"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry
    raise SystemExit(main())
