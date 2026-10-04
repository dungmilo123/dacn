"""Frozen eight-policy real-MNIST headline orchestration (M003/S04/T02).

``python -m dfl_sim.headline CONFIG --output DIR`` composes the *public*
production surfaces that already exist (T01 provisioning, S02 ``dfl calibrate``,
S03 matched-artifact preflight, S01/S02 ``dfl evaluate`` held-out study) into a
single reproducible real-data headline. It deliberately adds no simulator,
defense, optimizer or dataset code:

* The protocol is parsed from a strict, frozen :class:`HeadlineProtocol` and
  cross-checked against the existing schedule/config constraints (CPU bitwise
  SmallCNN, IID chain3, six rounds, 64 samples/node, 128 evaluation samples,
  batch 32, one local epoch, zero/weak/strong schedules, held_out reserved for
  the final phase, two selection seeds and three disjoint final seeds).
* The verified T01 MNIST cache is provisioned once and leased; every production
  calibration/evaluation subprocess runs with ``download: false`` against that
  verified root and never receives an injected dataset, trainer, loader or
  policy.
* The three calibration variants run as real ``dfl_sim calibrate`` subprocesses;
  their freshly published artifacts are loaded through the production artifact
  loader and pinned (matched-input digest plus one aggregate digest per variant)
  into a generated ``frozen-study.yaml`` :class:`HeldOutStudyConfig`.
* The S03 preflight validates the matched triple before the real
  ``dfl_sim evaluate`` held-out study runs.

The runner fails closed on an unsafe/aliased output path, an existing output,
a missing or mutated cache, a non-zero stage, a mutated protocol or a missing
split. It retains bounded stage diagnostics and partial raw outputs for
diagnosis but never writes the independent ``headline-evidence.json`` (that is
T03's verification gate) or a headline *acceptance* marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
    NO_BELIEF_VARIANT,
    NO_STRATEGIC_VARIANT,
    load_bayesian_artifact,
)
from .calibration_config import load_calibration_config
from .config import (
    MAX_SEED,
    BalanceAggregation,
    BayesianAggregation,
    ConfigError,
    ExperimentConfig,
    FixedSimilarityAggregation,
    LocalOnlyAggregation,
    PlainAverageAggregation,
    TrustWeightedAggregation,
)
from .evaluation import (
    FINAL_MANIFEST_FILENAME,
    SELECTION_MANIFEST_FILENAME,
    summarize_held_out_study,
)
from .evaluation_config import (
    BALANCE_CITATION,
    EXTENDED_POLICY_IDS,
    MAX_CANDIDATE_BUDGET,
    AblationArtifactPin,
    AblationStudyContract,
    BaselineStudyConfig,
    CitationMetadata,
    HeldOutStudyConfig,
    HeldOutStudyConfigError,
    NamedSchedule,
    PanelPolicy,
    TuningProvenance,
    parse_held_out_study_text,
)

__all__ = [
    "FROZEN_PROTOCOL_FILENAME",
    "FROZEN_STUDY_FILENAME",
    "HEADLINE_PROTOCOL_VERSION",
    "HEADLINE_RAW_MARKER",
    "MNIST_PROVENANCE_FILENAME",
    "HeadlineError",
    "HeadlineProtocol",
    "HeadlineRunResult",
    "HeadlineStageError",
    "SubprocessStageRunner",
    "build_calibrate_command",
    "build_evaluate_command",
    "build_parser",
    "build_panel",
    "build_study",
    "expected_cell_counts",
    "load_headline_config",
    "main",
    "parse_headline_text",
    "run_headline",
    "validate_protocol",
]

#: Versioned headline protocol tag. Bumping it invalidates any prior headline.
HEADLINE_PROTOCOL_VERSION = "dfl_sim.headline_protocol/v1"

FROZEN_PROTOCOL_FILENAME = "frozen-protocol.json"
MNIST_PROVENANCE_FILENAME = "mnist-provenance.json"
FROZEN_STUDY_FILENAME = "frozen-study.yaml"
DIAGNOSTICS_DIRNAME = "diagnostics"
CALIBRATION_DIRNAME = "calibration"
STUDY_DIRNAME = "study"

#: Bounded stdout marker for the raw (not yet independently verified) study.
#: T03 replaces this with the headline acceptance gate; T02 never claims more
#: than a raw completed study.
HEADLINE_RAW_MARKER = "dfl_sim: headline raw study "

#: Default per-stage wall-clock bound (seconds); override with ``--timeout``.
DEFAULT_STAGE_TIMEOUT = 86_400.0

_MAX_ERROR_TEXT = 400
_MAX_DIAG_BYTES = 16_384

_FULL = FULL_VARIANT
_NO_BELIEF = NO_BELIEF_VARIANT
_NO_STRATEGIC = NO_STRATEGIC_VARIANT

_VARIANT_POLICY_ID: dict[str, str] = {
    _FULL: "bayesian",
    _NO_BELIEF: "bayesian_no_belief",
    _NO_STRATEGIC: "bayesian_no_strategic",
}
_POLICY_IMPLEMENTATION: dict[str, str] = {
    "bayesian": "Bayesian",
    "bayesian_no_belief": "uniform-posterior ablation",
    "bayesian_no_strategic": "fixed-attacker ablation",
}
_POLICY_CITATION: dict[str, object] = {
    "kind": "project",
    "title": "project comparator",
}

#: The four D033 known schedule kinds. Each named schedule's id must equal its
#: kind so the panel and manifest are self-describing.
_KNOWN_SCHEDULE_KINDS: tuple[str, ...] = (
    "duty_cycle",
    "periodic",
    "delayed_onset",
    "random",
)
_HELD_OUT_SCHEDULE_ID = "held_out"


class HeadlineError(RuntimeError):
    """Raised when the headline protocol or a stage fails closed."""


class HeadlineStageError(HeadlineError):
    """A production subprocess stage returned non-zero or could not launch."""

    def __init__(self, label: str, detail: str) -> None:
        self.label = label
        super().__init__(f"{label}: {detail}")


# --------------------------------------------------------------------------- #
# Strict protocol schema
# --------------------------------------------------------------------------- #
def _as_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


SeedTuple = Annotated[tuple[int, ...], BeforeValidator(_as_tuple)]
FloatTuple = Annotated[tuple[float, ...], BeforeValidator(_as_tuple)]
ScheduleTuple = Annotated[tuple[NamedSchedule, ...], BeforeValidator(_as_tuple)]


class _StrictHeadlineModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class HeadlineCalibrationSpec(_StrictHeadlineModel):
    """The pinned calibration source contract."""

    config: str = Field(min_length=1, max_length=1024)
    seed: int = Field(ge=0, le=MAX_SEED)
    episodes: int = Field(ge=2, le=512)
    design_split_size: int = Field(ge=1, le=8192)
    p0: float = Field(ge=0.0, le=1.0)
    mu_a: FloatTuple


class HeadlineProtocol(_StrictHeadlineModel):
    """Strict, frozen headline protocol before the calibration digests exist."""

    protocol_version: Literal["dfl_sim.headline_protocol/v1"]
    study_id: str = Field(min_length=1, max_length=64)
    data_provenance: Literal["real_mnist"] = "real_mnist"
    threads: int = Field(default=1, ge=1, le=64)
    base: ExperimentConfig
    selection_seeds: SeedTuple
    final_seeds: SeedTuple
    candidate_budget: int = Field(ge=1, le=MAX_CANDIDATE_BUDGET)
    selection_seed: int = Field(ge=0, le=MAX_SEED)
    selection_provenance: str = Field(min_length=1, max_length=256)
    schedules: ScheduleTuple
    held_out_schedule: NamedSchedule
    calibration: HeadlineCalibrationSpec

    @model_validator(mode="after")
    def _bounded_ids(self) -> "HeadlineProtocol":
        safe = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
        if not all(character in safe for character in self.study_id):
            raise ValueError(
                "study_id must contain only lowercase letters, digits, "
                "hyphens, and underscores"
            )
        if not (
            self.held_out_schedule.id
            and all(
                character in safe or character == "_"
                for character in self.held_out_schedule.id
            )
        ):
            raise ValueError("held_out_schedule.id is not a safe schedule id")
        return self


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _bounded(text: object, *, limit: int = _MAX_ERROR_TEXT) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) > limit:
        return collapsed[: limit - 1] + "\u2026"
    return collapsed


def _parse_mapping(text: str, *, source: str) -> Mapping[str, Any]:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise HeadlineError(f"invalid headline config in {source}: {_bounded(exc)}") from exc
    if not isinstance(raw, Mapping):
        raise HeadlineError(
            f"headline config in {source} must be a mapping, got "
            f"{type(raw).__name__}"
        )
    return raw


def parse_headline_text(text: str, *, source: str = "<string>") -> HeadlineProtocol:
    """Parse and strictly validate a headline protocol from JSON or safe YAML."""

    raw = _parse_mapping(text, source=source)
    try:
        return HeadlineProtocol.model_validate(raw)
    except ValidationError as exc:
        raise HeadlineError(
            f"invalid headline protocol in {source}: {_bounded(exc)}"
        ) from exc


def load_headline_config(path: str | os.PathLike[str]) -> HeadlineProtocol:
    """Read ``path`` and validate it as a :class:`HeadlineProtocol`."""

    source = os.fspath(path)
    try:
        text = Path(source).read_text(encoding="utf-8")
    except OSError as exc:
        raise HeadlineError(f"could not read headline config {source}: {_bounded(exc)}") from exc
    return parse_headline_text(text, source=source)


# --------------------------------------------------------------------------- #
# Protocol validation
# --------------------------------------------------------------------------- #
def build_panel(
    artifact_dirs: Mapping[str, object],
    *,
    candidate_budget: int,
    selection_seed: int,
    selection_provenance: str,
) -> tuple[PanelPolicy, ...]:
    """Build the canonical eight-policy panel with pinned tuning provenance.

    The six baseline identities carry predeclared hyperparameters and the two
    parameter-free policies are exempt; every tunable policy declares exactly
    one candidate, so the panel asserts no search or winner claim. The three
    Bayesian entries point at the full / no-belief / no-strategic artifact
    directories derived from the headline root.
    """

    project = CitationMetadata(**_POLICY_CITATION)
    tuned = TuningProvenance(
        mode="tuned",
        candidate_budget=candidate_budget,
        selection_seed=selection_seed,
        selection_provenance=selection_provenance,
    )
    exempt = TuningProvenance(mode="exempt", candidate_budget=0)

    def _bayesian(variant: str) -> PanelPolicy:
        policy_id = _VARIANT_POLICY_ID[variant]
        return PanelPolicy(
            id=policy_id,
            implementation=_POLICY_IMPLEMENTATION[policy_id],
            aggregation=BayesianAggregation(
                kind="bayesian",
                artifact_dir=str(artifact_dirs[variant]),
                omega=0.5,
                degree_cap=4,
                tie_tolerance=1.0e-9,
            ),
            citation=project,
            tuning=tuned,
        )

    return (
        PanelPolicy(
            id="local_only",
            implementation="local only",
            aggregation=LocalOnlyAggregation(),
            citation=project,
            tuning=exempt,
        ),
        PanelPolicy(
            id="plain_average",
            implementation="plain average",
            aggregation=PlainAverageAggregation(),
            citation=project,
            tuning=exempt,
        ),
        PanelPolicy(
            id="fixed_similarity",
            implementation="fixed similarity",
            aggregation=FixedSimilarityAggregation(
                kind="fixed_similarity", cutoff=0.5, downweight_multiplier=0.5
            ),
            citation=project,
            tuning=tuned,
        ),
        PanelPolicy(
            id="trust_weighted",
            implementation="trust weighted",
            aggregation=TrustWeightedAggregation(beta=1.0),
            citation=project,
            tuning=tuned,
        ),
        PanelPolicy(
            id="balance",
            implementation="faithful BALANCE",
            aggregation=BalanceAggregation(),
            citation=CitationMetadata(**BALANCE_CITATION),
            tuning=tuned,
        ),
        _bayesian(_FULL),
        _bayesian(_NO_BELIEF),
        _bayesian(_NO_STRATEGIC),
    )


def _provisional_study(protocol: HeadlineProtocol) -> HeldOutStudyConfig:
    """Validate seed split, schedules and panel by building a throwaway study."""

    placeholder_dirs = {variant: f"<placeholder-{variant}>" for variant in CALIBRATION_VARIANTS}
    panel = build_panel(
        placeholder_dirs,
        candidate_budget=protocol.candidate_budget,
        selection_seed=protocol.selection_seed,
        selection_provenance=protocol.selection_provenance,
    )
    contract = AblationStudyContract(
        matched_calibration_input_digest="0" * 64,
        expected_aggregate_digests=tuple(
            AblationArtifactPin(
                variant=variant, expected_aggregate_digest="0" * 64
            )
            for variant in CALIBRATION_VARIANTS
        ),
    )
    study = BaselineStudyConfig(
        study_id=protocol.study_id,
        data_provenance=protocol.data_provenance,
        base=protocol.base,
        seeds=protocol.selection_seeds,
        candidate_budget=protocol.candidate_budget,
        selection_seed=protocol.selection_seed,
        selection_provenance=protocol.selection_provenance,
        panel=panel,
        ablations=contract,
    )
    return HeldOutStudyConfig(
        study=study,
        final_seeds=protocol.final_seeds,
        known_schedules=tuple(protocol.schedules),
        held_out_schedule=protocol.held_out_schedule,
    )


def validate_protocol(protocol: HeadlineProtocol) -> None:
    """Fail closed unless every pinned headline dimension holds.

    Most structural constraints (seed disjointness, schedule intensity
    coverage, peak intensity match, total-seed floor, panel identity/budget)
    are enforced by building a provisional :class:`HeldOutStudyConfig`. The
    remaining checks pin the deliberately small real-subset protocol itself.
    """

    base = protocol.base
    failures: list[str] = []
    if base.device != "cpu":
        failures.append("device must be cpu")
    if not base.bitwise:
        failures.append("bitwise must be true")
    if base.model.name != "small_cnn":
        failures.append("model must be small_cnn")
    if base.rounds != 6:
        failures.append("rounds must be 6")
    if base.data.dataset != "mnist":
        failures.append("dataset must be mnist")
    if base.data.partition != "iid":
        failures.append("partition must be iid")
    if base.data.samples_per_node != 64:
        failures.append("samples_per_node must be 64")
    if base.data.download:
        failures.append("base.data.download must be false (headline leases a verified cache)")
    if base.training.batch_size != 32:
        failures.append("training.batch_size must be 32")
    if base.training.local_epochs != 1:
        failures.append("training.local_epochs must be 1")
    if base.training.num_workers != 0:
        failures.append("training.num_workers must be 0")
    if base.evaluation.max_samples != 128:
        failures.append("evaluation.max_samples must be 128")
    if base.topology.kind != "chain3":
        failures.append("topology must be chain3")
    if base.attack.compromised_node is None:
        failures.append("base.attack.compromised_node must be set")
    if base.attack.intensity != "strong":
        failures.append("base.attack.intensity must be the strong peak")
    if not isinstance(base.aggregation, PlainAverageAggregation):
        failures.append("base.aggregation must be plain_average")
    if len(protocol.selection_seeds) != 2:
        failures.append("selection_seeds must declare exactly two seeds")
    if len(protocol.final_seeds) < 3:
        failures.append("final_seeds must declare at least three seeds")
    if protocol.candidate_budget != 1:
        failures.append("candidate_budget must be exactly one")
    if protocol.threads < 1:
        failures.append("threads must be >= 1")
    reserved_seeds = set(protocol.selection_seeds) | set(protocol.final_seeds)
    if protocol.calibration.seed in reserved_seeds:
        failures.append(
            "calibration seed must be separate from the selection and final seeds"
        )

    schedule_kinds = tuple((entry.id, entry.schedule.kind) for entry in protocol.schedules)
    if len(protocol.schedules) != len(_KNOWN_SCHEDULE_KINDS):
        failures.append("schedules must declare exactly the four known schedules")
    elif set(entry.id for entry in protocol.schedules) != set(_KNOWN_SCHEDULE_KINDS):
        failures.append(f"schedules must be {list(_KNOWN_SCHEDULE_KINDS)!r}")
    for schedule_id, kind in schedule_kinds:
        if schedule_id != kind:
            failures.append(f"schedule {schedule_id!r} must use kind {schedule_id!r}")
    if protocol.held_out_schedule.schedule.kind != "held_out":
        failures.append("held_out_schedule must use the reserved held_out kind")
    if protocol.held_out_schedule.id in {entry.id for entry in protocol.schedules}:
        failures.append("held_out_schedule.id must not collide with a known schedule")

    if failures:
        raise HeadlineError("invalid headline protocol: " + "; ".join(failures))

    try:
        _provisional_study(protocol)
    except (ValidationError, ConfigError) as exc:
        raise HeadlineError(
            f"invalid headline protocol: {_bounded(exc)}"
        ) from exc


# --------------------------------------------------------------------------- #
# Pure helpers: identity, provenance, study generation, counts
# --------------------------------------------------------------------------- #
def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def protocol_digest(protocol: HeadlineProtocol) -> str:
    """Environment-independent digest of the resolved source protocol."""

    return _sha256_json(protocol.model_dump(mode="json"))


def expected_cell_counts(protocol: HeadlineProtocol) -> tuple[int, int]:
    """Return ``(selection_cells, final_cells)`` for the frozen matrix."""

    panel_size = len(EXTENDED_POLICY_IDS)
    selection = len(protocol.selection_seeds) * len(protocol.schedules) * panel_size
    final = (
        len(protocol.final_seeds)
        * (len(protocol.schedules) + 1)
        * panel_size
    )
    return selection, final


def _manifest_dump(manifest: object) -> Mapping[str, Any]:
    dump = getattr(manifest, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(manifest, Mapping):
        return dict(manifest)
    raise HeadlineError("cache manifest is not serialisable")


def _manifest_digest(manifest: object) -> str | None:
    digest = getattr(manifest, "manifest_digest", None)
    return digest if isinstance(digest, str) else None


def build_frozen_protocol(
    protocol: HeadlineProtocol,
    *,
    source: str,
    cache_manifest: object,
    threads: int,
) -> dict:
    """The frozen-protocol.json payload binding spec, cache identity and split."""

    cache = _manifest_dump(cache_manifest)
    return {
        "schema": HEADLINE_PROTOCOL_VERSION,
        "source": source,
        "protocol_digest": protocol_digest(protocol),
        "study_id": protocol.study_id,
        "data_provenance": protocol.data_provenance,
        "threads": int(threads),
        "selection_seeds": list(protocol.selection_seeds),
        "final_seeds": list(protocol.final_seeds),
        "candidate_budget": protocol.candidate_budget,
        "selection_seed": protocol.selection_seed,
        "selection_provenance": protocol.selection_provenance,
        "known_schedule_ids": [entry.id for entry in protocol.schedules],
        "held_out_schedule_id": protocol.held_out_schedule.id,
        "calibration": protocol.calibration.model_dump(mode="json"),
        "cache": {
            "data_root": protocol.base.data.data_root,
            "manifest_digest": cache.get("manifest_digest"),
            "preprocessing_digest": cache.get("preprocessing_digest"),
            "preprocessing_id": cache.get("preprocessing_id"),
        },
        "protocol": protocol.model_dump(mode="json"),
    }


def build_mnist_provenance(
    protocol: HeadlineProtocol, *, data_root: str, cache_manifest: object
) -> dict:
    """The mnist-provenance.json payload recording the verified T01 snapshot."""

    return {
        "schema": "dfl_sim.headline_mnist_provenance/v1",
        "data_root": data_root,
        "manifest_digest": _manifest_digest(cache_manifest),
        "real_mnist": True,
        "manifest": _manifest_dump(cache_manifest),
    }


def _artifact_derivation(artifact: object, variant: str):
    manifest = getattr(artifact, "manifest", None)
    derivation = getattr(manifest, "derivation", None)
    if derivation is None:
        raise HeadlineError(
            f"{variant} calibration artifact has no calibration derivation; "
            "refusing to pin an unproven artifact"
        )
    return derivation


def _artifact_aggregate(artifact: object, variant: str) -> str:
    digest = getattr(getattr(artifact, "manifest", None), "aggregate_digest", None)
    if not isinstance(digest, str) or not digest:
        raise HeadlineError(f"{variant} calibration artifact lacks an aggregate digest")
    return digest


def build_study(
    protocol: HeadlineProtocol,
    root: str | os.PathLike[str],
    artifacts: Mapping[str, object],
) -> HeldOutStudyConfig:
    """Generate the digest-pinned executable held-out study."""

    root_path = Path(root)
    artifact_dirs = {
        variant: root_path / CALIBRATION_DIRNAME / variant
        for variant in CALIBRATION_VARIANTS
    }
    for variant in CALIBRATION_VARIANTS:
        if variant not in artifacts:
            raise HeadlineError(f"missing {variant!r} calibration artifact")
        _artifact_derivation(artifacts[variant], variant)

    matched = matched_calibration_input_digest(
        _artifact_derivation(artifacts[_FULL], _FULL)
    )
    pins = tuple(
        AblationArtifactPin(
            variant=variant,
            expected_aggregate_digest=_artifact_aggregate(artifacts[variant], variant),
        )
        for variant in CALIBRATION_VARIANTS
    )
    contract = AblationStudyContract(
        matched_calibration_input_digest=matched,
        expected_aggregate_digests=pins,
    )
    panel = build_panel(
        artifact_dirs,
        candidate_budget=protocol.candidate_budget,
        selection_seed=protocol.selection_seed,
        selection_provenance=protocol.selection_provenance,
    )
    study = BaselineStudyConfig(
        study_id=protocol.study_id,
        data_provenance=protocol.data_provenance,
        base=protocol.base,
        seeds=protocol.selection_seeds,
        candidate_budget=protocol.candidate_budget,
        selection_seed=protocol.selection_seed,
        selection_provenance=protocol.selection_provenance,
        panel=panel,
        ablations=contract,
    )
    return HeldOutStudyConfig(
        study=study,
        final_seeds=protocol.final_seeds,
        known_schedules=tuple(protocol.schedules),
        held_out_schedule=protocol.held_out_schedule,
    )


def serialize_study(study: HeldOutStudyConfig) -> str:
    """Serialise a study to YAML and re-validate the exact text round-trip."""

    text = yaml.safe_dump(study.model_dump(mode="json"), sort_keys=True)
    try:
        parse_held_out_study_text(text, source="<frozen-study>")
    except HeldOutStudyConfigError as exc:
        raise HeadlineError(f"generated study is not valid: {_bounded(exc)}") from exc
    return text


# --------------------------------------------------------------------------- #
# Subprocess stage runner
# --------------------------------------------------------------------------- #
def build_calibrate_command(
    python: str, config_path: str, destination: str, variant: str
) -> list[str]:
    """Production ``dfl calibrate`` argv: no dataset/trainer/policy injection."""

    return [
        str(python),
        "-m",
        "dfl_sim",
        "calibrate",
        str(config_path),
        "--output",
        str(destination),
        "--variant",
        str(variant),
    ]


def build_evaluate_command(
    python: str, study_path: str, destination: str
) -> list[str]:
    """Production ``dfl evaluate`` argv: no dataset/trainer/policy injection."""

    return [
        str(python),
        "-m",
        "dfl_sim",
        "evaluate",
        str(study_path),
        "--output",
        str(destination),
    ]


def _stage_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    bound = str(int(threads))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = bound
    return env


def _tail(text: str | None, *, limit: int = _MAX_DIAG_BYTES) -> str:
    if not text:
        return ""
    if len(text) > limit:
        return "\u2026" + text[-limit:]
    return text


@dataclass(frozen=True)
class StageOutcome:
    label: str
    returncode: int
    duration_ms: int
    stdout_tail: str
    stderr_tail: str


class StageRunner(Protocol):
    """Injection seam for the two production subprocess stages."""

    def calibrate(
        self,
        config_path: str,
        destination: str,
        variant: str,
        diagnostics_dir: Path,
    ) -> StageOutcome: ...

    def evaluate(
        self, study_path: str, destination: str, diagnostics_dir: Path
    ) -> StageOutcome: ...


class SubprocessStageRunner:
    """Run the public ``dfl_sim calibrate`` / ``dfl_sim evaluate`` entry points.

    The repository root is always the child working directory so package
    imports and repository-relative paths resolve; the child environment caps
    BLAS thread pools at ``threads`` so the run is reproducible and feasible on
    CPU. Bounded stdout/stderr tails are persisted under ``diagnostics/`` and a
    non-zero exit raises :class:`HeadlineStageError`.
    """

    def __init__(
        self,
        *,
        repo_root: str | os.PathLike[str],
        threads: int = 1,
        timeout: float = DEFAULT_STAGE_TIMEOUT,
        python: str | None = None,
    ) -> None:
        self._repo_root = Path(repo_root)
        self._threads = int(threads)
        self._timeout = float(timeout)
        self._python = python or sys.executable

    def calibrate(
        self,
        config_path: str,
        destination: str,
        variant: str,
        diagnostics_dir: Path,
    ) -> StageOutcome:
        command = build_calibrate_command(
            self._python, config_path, destination, variant
        )
        return self._run(
            command, label=f"calibrate-{variant}", diagnostics_dir=diagnostics_dir
        )

    def evaluate(
        self, study_path: str, destination: str, diagnostics_dir: Path
    ) -> StageOutcome:
        command = build_evaluate_command(self._python, study_path, destination)
        return self._run(command, label="evaluate", diagnostics_dir=diagnostics_dir)

    def _run(
        self, command: Sequence[str], *, label: str, diagnostics_dir: Path
    ) -> StageOutcome:
        env = _stage_env(self._threads)
        start = time.monotonic()
        try:
            process = subprocess.run(
                list(command),
                cwd=str(self._repo_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            detail = f"timed out after {self._timeout:g}s"
            _write_diagnostics(diagnostics_dir, label, command, detail=detail)
            raise HeadlineStageError(label, detail) from exc
        except OSError as exc:
            detail = f"could not launch: {_bounded(exc)}"
            _write_diagnostics(diagnostics_dir, label, command, detail=detail)
            raise HeadlineStageError(label, detail) from exc
        duration_ms = int((time.monotonic() - start) * 1000)
        stdout_tail = _tail(process.stdout)
        stderr_tail = _tail(process.stderr)
        _write_diagnostics(
            diagnostics_dir,
            label,
            command,
            stdout=stdout_tail,
            stderr=stderr_tail,
            returncode=process.returncode,
            duration_ms=duration_ms,
        )
        if process.returncode != 0:
            detail = stderr_tail.strip() or stdout_tail.strip() or "no output"
            raise HeadlineStageError(
                label, f"exit {process.returncode}: {_bounded(detail)}"
            )
        return StageOutcome(
            label=label,
            returncode=process.returncode,
            duration_ms=duration_ms,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )


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
        # Diagnostics are best-effort; never mask the original stage failure.
        pass


# --------------------------------------------------------------------------- #
# Injected dependencies
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeadlineDeps:
    """Optional injection seams; production uses the real public surfaces."""

    provision: Callable[..., object] | None = None
    cache_lease: Callable[..., Any] | None = None
    load_calibration_config: Callable[..., object] | None = None
    load_artifact: Callable[..., object] | None = None
    preflight: Callable[..., AblationMatchSummary] | None = None
    summarize: Callable[..., object] | None = None
    verify: Callable[..., object] | None = None
    stage_runner: StageRunner | None = None
    repo_root: str | os.PathLike[str] | None = None


@dataclass(frozen=True)
class HeadlineRunResult:
    output_path: Path
    frozen_protocol_path: Path
    provenance_path: Path
    frozen_study_path: Path
    study_path: Path
    protocol_digest: str
    selection_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    selection_cells: int
    final_cells: int
    stages: tuple[StageOutcome, ...] = field(default_factory=tuple)
    evidence_path: Path | None = None


# --------------------------------------------------------------------------- #
# Output / path safety
# --------------------------------------------------------------------------- #
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
        raise HeadlineError(f"invalid output destination: {destination}")
    parent = destination.parent
    if not parent.is_dir():
        raise HeadlineError(f"output parent does not exist: {parent!r}")
    if parent.is_symlink():
        raise HeadlineError(f"output parent must not be a symlink: {parent!r}")
    if os.path.lexists(destination):
        raise HeadlineError(f"output already exists: {destination!r}")
    if _paths_alias(destination, config_path):
        raise HeadlineError("output aliases the headline config path")


def _check_calibration_contract(
    protocol: HeadlineProtocol, calibration: object, calibration_path: Path
) -> None:
    """Cross-check the embedded calibration experiment against the headline."""

    base = protocol.base
    try:
        experiment = calibration.experiment  # type: ignore[attr-defined]
        data = experiment.data
        training = experiment.training
        evaluation = experiment.evaluation
    except AttributeError as exc:  # pragma: no cover - defensive
        raise HeadlineError(
            f"calibration config {calibration_path} is malformed: {_bounded(exc)}"
        ) from exc

    checks = (
        (experiment.device == "cpu", "device"),
        (experiment.bitwise is True, "bitwise"),
        (experiment.model.name == base.model.name, "model"),
        (experiment.data.dataset == base.data.dataset, "dataset"),
        (experiment.data.partition == base.data.partition, "partition"),
        (data.samples_per_node == base.data.samples_per_node, "samples_per_node"),
        (data.data_root == base.data.data_root, "data_root"),
        (data.download is False, "download"),
        (training.batch_size == base.training.batch_size, "batch_size"),
        (training.local_epochs == 1, "local_epochs"),
        (training.num_workers == 0, "num_workers"),
        (evaluation.max_samples == base.evaluation.max_samples, "max_samples"),
        (experiment.topology.kind == base.topology.kind, "topology"),
        (experiment.rounds == 1, "rounds"),
        (experiment.seed == protocol.calibration.seed, "seed"),
        (int(getattr(calibration, "episodes")) == protocol.calibration.episodes, "episodes"),
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
    failures = [name for ok, name in checks if not ok]
    if failures:
        raise HeadlineError(
            "calibration contract mismatch for "
            f"{calibration_path}: {', '.join(failures)}"
        )


# --------------------------------------------------------------------------- #
# Published study validation
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise HeadlineError(f"missing headline artifact: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeadlineError(f"could not read {path}: {_bounded(exc)}") from exc
    if not isinstance(raw, Mapping):
        raise HeadlineError(f"{path} is not a JSON object")
    return raw


def _verify_manifest_counts(
    manifest: Mapping[str, object],
    *,
    expected_schema: str,
    expected_cells: int,
    expected_seeds: Sequence[int],
    expected_schedules: Sequence[str],
    label: str,
) -> None:
    if manifest.get("schema") != expected_schema:
        raise HeadlineError(f"{label} has an unexpected schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != expected_cells:
        raise HeadlineError(
            f"{label} declares "
            f"{len(entries) if isinstance(entries, list) else 'no'} cells, "
            f"expected {expected_cells}"
        )
    panel = manifest.get("panel")
    if not isinstance(panel, list) or sorted(map(str, panel)) != sorted(EXTENDED_POLICY_IDS):
        raise HeadlineError(f"{label} does not declare the eight-policy panel")
    schedule_ids = manifest.get("schedule_ids")
    if not isinstance(schedule_ids, list) or sorted(map(str, schedule_ids)) != sorted(expected_schedules):
        raise HeadlineError(f"{label} schedule coverage is wrong")
    seed_field = "selection_seeds" if label == "selection" else "final_seeds"
    seeds = manifest.get(seed_field)
    if not isinstance(seeds, list) or sorted(int(s) for s in seeds) != sorted(int(s) for s in expected_seeds):
        raise HeadlineError(f"{label} seed set is wrong")


def validate_published_study(
    study_root: Path, protocol: HeadlineProtocol
) -> tuple[int, int]:
    """Fail closed unless the published study matches the frozen matrix."""

    selection = _read_json(study_root / "selection" / SELECTION_MANIFEST_FILENAME)
    final = _read_json(study_root / "final" / FINAL_MANIFEST_FILENAME)

    selection_cells, final_cells = expected_cell_counts(protocol)
    known_ids = [entry.id for entry in protocol.schedules]
    _verify_manifest_counts(
        selection,
        expected_schema="dfl_held_out_selection",
        expected_cells=selection_cells,
        expected_seeds=protocol.selection_seeds,
        expected_schedules=known_ids,
        label="selection",
    )
    _verify_manifest_counts(
        final,
        expected_schema="dfl_held_out_study_manifest",
        expected_cells=final_cells,
        expected_seeds=protocol.final_seeds,
        expected_schedules=[*known_ids, protocol.held_out_schedule.id],
        label="final",
    )

    held_out = [entry for entry in final.get("entries", []) if entry.get("held_out")]
    if len(held_out) != len(protocol.final_seeds) * len(EXTENDED_POLICY_IDS):
        raise HeadlineError("final manifest held_out coverage is wrong")
    if any(entry.get("held_out") for entry in selection.get("entries", [])):
        raise HeadlineError("selection manifest must not contain a held_out cell")
    return selection_cells, final_cells


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #
def _default_repo_root() -> Path:
    # src/dfl_sim/headline.py -> repo root
    return Path(__file__).resolve().parents[2]


def _write_bytes(path: Path, payload: bytes) -> None:
    try:
        with open(path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise HeadlineError(f"could not write {path}: {_bounded(exc)}") from exc


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _write_bytes(
        path,
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


def _write_text(path: Path, text: str) -> None:
    _write_bytes(path, text.encode("utf-8"))


def run_headline(
    config_path: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    data_root: str | os.PathLike[str] | None = None,
    download: bool = True,
    timeout: float = DEFAULT_STAGE_TIMEOUT,
    deps: HeadlineDeps | None = None,
) -> HeadlineRunResult:
    """Run the frozen headline protocol end to end.

    Provisioning is idempotent and verified; the cache is leased across the
    three calibrations and the held-out study so a mutation fails closed. The
    generated study is re-validated through the real S03 preflight before the
    production ``dfl evaluate`` stage runs. No dataset, trainer, loader or
    policy is ever injected into the production command path.
    """

    protocol = load_headline_config(config_path)
    validate_protocol(protocol)
    config_abs = Path(os.path.abspath(os.fspath(config_path)))
    destination = Path(os.path.abspath(os.fspath(output)))
    _check_output_destination(destination, config_abs)

    deps = deps or HeadlineDeps()
    repo_root = Path(deps.repo_root) if deps.repo_root is not None else _default_repo_root()
    provision = deps.provision or mnist_provision.provision
    cache_lease = deps.cache_lease or mnist_provision.cache_lease
    load_calibration = deps.load_calibration_config or load_calibration_config
    load_artifact = deps.load_artifact or load_bayesian_artifact
    preflight = deps.preflight or preflight_ablation_study
    summarize = deps.summarize or summarize_held_out_study
    stage_runner = deps.stage_runner or SubprocessStageRunner(
        repo_root=repo_root, threads=protocol.threads, timeout=timeout
    )

    calibration_path = _resolve_path(protocol.calibration.config, repo_root)
    try:
        calibration = load_calibration(str(calibration_path))
    except (HeadlineError, ConfigError) as exc:
        raise HeadlineError(
            f"could not load calibration config {calibration_path}: {_bounded(exc)}"
        ) from exc
    _check_calibration_contract(protocol, calibration, calibration_path)

    resolved_root = Path(
        os.path.abspath(os.fspath(data_root or protocol.base.data.data_root))
    )
    if _paths_alias(destination, resolved_root) or _is_within(destination, resolved_root) or _is_within(resolved_root, destination):
        raise HeadlineError(
            "output root must not alias or contain the verified data root"
        )

    # Provision once, before creating the output tree, and fail closed.
    try:
        cache_manifest = provision(str(resolved_root), download=download)
    except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
        raise HeadlineError(
            f"MNIST provisioning failed closed: {_bounded(exc)}"
        ) from exc

    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:  # pragma: no cover - checked above
        raise HeadlineError(f"output already exists: {destination!r}") from exc
    except OSError as exc:
        raise HeadlineError(
            f"could not create output root {destination!r}: {_bounded(exc)}"
        ) from exc
    diagnostics = destination / DIAGNOSTICS_DIRNAME
    diagnostics.mkdir(parents=True, exist_ok=True)

    frozen = build_frozen_protocol(
        protocol,
        source=str(config_path),
        cache_manifest=cache_manifest,
        threads=protocol.threads,
    )
    _write_json(destination / FROZEN_PROTOCOL_FILENAME, frozen)
    _write_json(
        destination / MNIST_PROVENANCE_FILENAME,
        build_mnist_provenance(
            protocol, data_root=str(resolved_root), cache_manifest=cache_manifest
        ),
    )

    stages: list[StageOutcome] = []
    try:
        with cache_lease(str(resolved_root), exclusive=True, deep=True):
            artifacts: dict[str, object] = {}
            calibration_root = destination / CALIBRATION_DIRNAME
            calibration_root.mkdir(parents=True, exist_ok=True)
            for variant in CALIBRATION_VARIANTS:
                variant_destination = calibration_root / variant
                stages.append(
                    stage_runner.calibrate(
                        str(calibration_path),
                        str(variant_destination),
                        variant,
                        diagnostics,
                    )
                )
                artifacts[variant] = load_artifact(str(variant_destination))

            study = build_study(protocol, destination, artifacts)
            try:
                preflight(study)
            except AblationEvaluationError as exc:
                raise HeadlineError(
                    f"matched-artifact preflight failed closed: {_bounded(exc)}"
                ) from exc

            frozen_study_path = destination / FROZEN_STUDY_FILENAME
            _write_text(frozen_study_path, serialize_study(study))

            stages.append(
                stage_runner.evaluate(
                    str(frozen_study_path),
                    str(destination / STUDY_DIRNAME),
                    diagnostics,
                )
            )
            summarize(str(destination / STUDY_DIRNAME))
            selection_cells, final_cells = validate_published_study(
                destination / STUDY_DIRNAME, protocol
            )
    except HeadlineError:
        raise
    except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
        raise HeadlineError(f"headline stage failed closed: {_bounded(exc)}") from exc

    # Independently audit the published study from disk and publish the
    # headline evidence before the caller can emit the success marker. A
    # missing/mutated source fails closed here and never overwrites a stale
    # complete record.
    verify = deps.verify
    if verify is None:
        from . import headline_verify

        verify = headline_verify.verify_headline
    try:
        verify(str(destination))
    except HeadlineError:
        raise
    except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
        raise HeadlineError(
            f"independent headline audit failed closed: {_bounded(exc)}"
        ) from exc
    evidence_path = destination / "headline-evidence.json"
    if not evidence_path.is_file():
        evidence_path = None

    return HeadlineRunResult(
        output_path=destination,
        frozen_protocol_path=destination / FROZEN_PROTOCOL_FILENAME,
        provenance_path=destination / MNIST_PROVENANCE_FILENAME,
        frozen_study_path=destination / FROZEN_STUDY_FILENAME,
        study_path=destination / STUDY_DIRNAME,
        protocol_digest=frozen["protocol_digest"],
        selection_seeds=tuple(protocol.selection_seeds),
        final_seeds=tuple(protocol.final_seeds),
        selection_cells=selection_cells,
        final_cells=final_cells,
        stages=tuple(stages),
        evidence_path=evidence_path,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dfl_sim.headline",
        description=(
            "Run the frozen eight-policy real-MNIST headline: verify/lease the "
            "T01 MNIST cache, run three production calibrations, pin and "
            "preflight a generated held-out study, then run the production "
            "held-out evaluation."
        ),
    )
    parser.add_argument(
        "config",
        help="Path to a strict headline protocol YAML (e.g. configs/headline_mnist.yaml).",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Destination headline root. Must not exist; created fail-closed.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        metavar="DIR",
        help="Verified T01 MNIST cache root (defaults to base.data.data_root).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Reuse an already verified cache and never download.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_STAGE_TIMEOUT,
        help=f"Per-stage wall-clock bound in seconds (default {DEFAULT_STAGE_TIMEOUT:g}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_headline(
            args.config,
            args.output,
            data_root=args.data_root,
            download=not args.no_download,
            timeout=args.timeout,
        )
    except (HeadlineError, ConfigError, OSError) as exc:
        print(f"dfl_sim: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"{HEADLINE_RAW_MARKER}"
        f"output={result.output_path} "
        f"frozen_study={result.frozen_study_path} "
        f"study={result.study_path} "
        f"selection_cells={result.selection_cells} "
        f"final_cells={result.final_cells}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
