"""Strict, frozen offline calibration configuration (R013/R014/D021, Q3).

This module owns the *offline* calibration config contract. It is deliberately
separate from the online :class:`~dfl_sim.config.ExperimentConfig`: the
calibration envelope embeds a complete, already-validated experiment and adds
the sampler, smoothing, design-split, and table-bin controls that only the
offline calibrator may hold. None of these symbols are importable from the
online policy / artifact-loader boundary.

Configuration is treated as untrusted structured input (Q3 threat surface):

* Canonical JSON is tried first and ``yaml.safe_load`` is the fallback, so a
  canonical snapshot round-trips exactly while ordinary YAML keeps working.
* Every model forbids unknown keys (``extra="forbid"``), is frozen, is strict,
  and rejects non-finite floats (``allow_inf_nan=False``).
* Episode count, design-split size, smoothing alpha, bin cardinality, and the
  required training-example budget are all bounded.
* No config value ever selects a Python class or executes a tag.
* Error text is bounded so a hostile or accidental invalid config cannot flood
  logs with an unbounded validation dump.

The embedded experiment must use plain averaging, must not preselect a
compromised node, and (with the design split) must fit inside the fixed MNIST
training split. ``mu_a`` and the action-space provenance are pinned to the
M002 first-pass values so the generated artifact is accepted unchanged by the
S01 loader / policy.
"""

from __future__ import annotations

import json
import math
import os
from typing import Annotated

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from .ablation import (
    FULL_VARIANT,
    SOURCE_MU_A,
    Variant,
    effective_mixture,
)
from .bayesian_artifact import (
    ActionSpaceProvenance,
    BinSet,
    CategoricalBins,
    IntervalBins,
)
from .config import (
    BAYESIAN_DEGREE_CAP,
    BAYESIAN_OMEGA,
    BAYESIAN_TIE_TOLERANCE,
    ConfigError,
    ExperimentConfig,
    PlainAverageAggregation,
    topology_node_count,
)

__all__ = [
    "DEGREE_LABELS",
    "MAX_CALIBRATION_EPISODES",
    "MAX_DESIGN_SPLIT_SIZE",
    "MAX_SMOOTHING_ALPHA",
    "MNIST_TRAIN_EXAMPLES",
    "PROGRESS_LABELS",
    "CalibrationConfig",
    "CalibrationConfigError",
    "load_calibration_config",
    "parse_calibration_text",
]

# --------------------------------------------------------------------------- #
# Bounds and pinned literals (Q3/Q6: out-of-range values are rejected).
# --------------------------------------------------------------------------- #
#: A calibration needs at least one clean and one attacked stratum, so a
#: bounded episode count can never be smaller than two. The upper bound keeps
#: one offline calibration run finite and auditable.
MAX_CALIBRATION_EPISODES = 512

#: Upper bound on the clean global design split (calibrator-only population).
MAX_DESIGN_SPLIT_SIZE = 8192

#: Additive (Laplace/Dirichlet) smoothing bound. Alpha is strictly positive so
#: no cell can ever be assigned zero probability by the estimator.
MIN_SMOOTHING_ALPHA = 1e-6
MAX_SMOOTHING_ALPHA = 100.0

#: Maximum number of interval edges per evidence feature (edges + 1 labels).
MAX_EVIDENCE_EDGES = 31

#: The MNIST training split length. The node-local partition plus the design
#: split must fit inside it; the final-test split is never used.
MNIST_TRAIN_EXAMPLES = 60_000

#: Pinned M002 context labels (CONTEXT "First-pass numeric defaults" / S01
#: fixture). The artifact contract and the online policy resolve exactly these.
DEGREE_LABELS: tuple[str, ...] = ("1", "2", "3-4", "5+")
PROGRESS_LABELS: tuple[str, ...] = ("early", "mid", "late")

#: Pinned finite attack-mixture component values, in canonical order. These map
#: one-to-one onto the D016 intensities zero/weak/strong (kappa 0/2/5). The
#: canonical source protocol always carries this mixture; the no-strategic
#: variant derives its effective degenerate mixture from the variant instead of
#: accepting an arbitrary attacker setting.
PINNED_MU_A: tuple[float, ...] = SOURCE_MU_A

#: Maximum rendered error-message length (Q3 bounded diagnostics).
MAX_ERROR_TEXT = 400


class CalibrationConfigError(ConfigError):
    """Raised when calibration YAML/JSON is missing, malformed, or invalid."""


def _as_tuple(value: object) -> object:
    """Convert JSON/YAML lists to tuples before strict tuple validation."""

    if isinstance(value, list):
        return tuple(value)
    return value


#: Pinned mixture as a strict tuple that accepts JSON/YAML lists.
FloatTuple = Annotated[tuple[float, ...], BeforeValidator(_as_tuple)]


class _StrictCalibrationModel(BaseModel):
    """Base for calibration leaves: unknown keys forbidden, frozen, strict."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


def _validate_interval_bins(bins: IntervalBins, name: str) -> None:
    """Reject non-finite, duplicate, or unsorted evidence-bin edges."""

    edges = bins.edges
    labels = bins.labels
    if len(edges) > MAX_EVIDENCE_EDGES:
        raise ValueError(
            f"{name} bins declare {len(edges)} edges, limit is {MAX_EVIDENCE_EDGES}"
        )
    if len(labels) != len(edges) + 1:
        raise ValueError(
            f"{name} bins must declare exactly len(edges)+1 labels, "
            f"got {len(labels)} labels for {len(edges)} edges"
        )
    previous: float | None = None
    for edge in edges:
        if not math.isfinite(edge):
            raise ValueError(f"{name} bin edges must be finite")
        if previous is not None and edge <= previous:
            raise ValueError(f"{name} bin edges must be strictly increasing")
        previous = edge
    if len(set(labels)) != len(labels) or not all(labels):
        raise ValueError(f"{name} bin labels must be unique and non-empty")


def _validate_categorical_bins(
    bins: CategoricalBins, name: str, *, expected: tuple[str, ...] | None = None
) -> None:
    labels = bins.labels
    if not labels:
        raise ValueError(f"{name} bins must declare at least one label")
    if len(set(labels)) != len(labels) or not all(labels):
        raise ValueError(f"{name} bin labels must be unique and non-empty")
    if expected is not None and tuple(labels) != expected:
        raise ValueError(
            f"{name} bins must be pinned to {expected!r}, got {tuple(labels)!r}"
        )


def _validate_bins(bins: BinSet) -> None:
    _validate_interval_bins(bins.disagreement, "disagreement")
    _validate_interval_bins(bins.relative_norm, "relative_norm")
    _validate_interval_bins(bins.loss_delta, "loss_delta")
    _validate_categorical_bins(bins.degree, "degree", expected=DEGREE_LABELS)
    _validate_categorical_bins(bins.progress, "progress", expected=PROGRESS_LABELS)


def _validate_action_space(action_space: ActionSpaceProvenance) -> None:
    """Pin the calibration action space to the S01/D021 first-pass values."""

    if action_space.omega != BAYESIAN_OMEGA:
        raise ValueError(
            "action_space.omega must equal the pinned first-pass value "
            f"{BAYESIAN_OMEGA!r}, got {action_space.omega!r}"
        )
    if action_space.degree_cap != BAYESIAN_DEGREE_CAP:
        raise ValueError(
            "action_space.degree_cap must equal the pinned first-pass value "
            f"{BAYESIAN_DEGREE_CAP!r}, got {action_space.degree_cap!r}"
        )
    if action_space.tie_tolerance != BAYESIAN_TIE_TOLERANCE:
        raise ValueError(
            "action_space.tie_tolerance must equal the pinned first-pass value "
            f"{BAYESIAN_TIE_TOLERANCE!r}, got {action_space.tie_tolerance!r}"
        )
    if action_space.self_multiplier != 1.0:
        raise ValueError("action_space.self_multiplier must be exactly 1.0")
    expected = sorted((0.0, BAYESIAN_OMEGA, 1.0))
    if list(action_space.multiplier_set) != expected:
        raise ValueError(
            "action_space.multiplier_set must be exactly {0, omega, 1}, got "
            f"{list(action_space.multiplier_set)!r}"
        )


class CalibrationConfig(_StrictCalibrationModel):
    """Top-level strict offline calibration envelope (R013/R014).

    ``experiment`` is a complete validated :class:`ExperimentConfig`. The
    sampler-derived fields are:

    ``episodes``
        Number of reference episodes (>= 2 so both strata exist).
    ``p0``
        Target *clean* episode fraction in ``[0, 1]``. The schedule rounds it
        and clamps so a valid small run still contains at least one clean and
        one attacked episode.
    ``variant``
        The calibration ablation variant (default ``full``). ``full`` and
        ``no_belief`` use the pinned source mixture ``(0, 2, 5)``; the
        ``no_strategic`` variant derives the degenerate strong mixture ``(5,)``
        at schedule-build time while retaining clean/attacked counts, seeds,
        placements, rounds, bins, and action space.
    ``mu_a``
        Pinned finite attack-mixture component values ``(0, 2, 5)`` mapped
        exactly onto the D016 zero/weak/strong intensities. Conditional on an
        attacked episode, the sampler cycles the *effective* mixture as a
        deterministic balanced mixture.
    ``max_compromised``
        Structurally pinned to ``1`` (at most one compromised node per episode).
    ``smoothing_alpha``
        Strictly positive additive smoothing applied by the estimator.
    ``design_split_size``
        Bounded clean global design-split size drawn from unused MNIST-training
        indices after every node-local partition is allocated.
    ``bins`` / ``action_space``
        Explicit S01-compatible feature/context bins and pinned action-space
        provenance so the generated artifact loads through the S01 loader.
    """

    experiment: ExperimentConfig
    variant: Variant = FULL_VARIANT
    episodes: int = Field(ge=2, le=MAX_CALIBRATION_EPISODES)
    p0: float = Field(ge=0.0, le=1.0)
    mu_a: FloatTuple
    max_compromised: int = Field(default=1, ge=1, le=1)
    smoothing_alpha: float = Field(
        ge=MIN_SMOOTHING_ALPHA, le=MAX_SMOOTHING_ALPHA
    )
    design_split_size: int = Field(ge=1, le=MAX_DESIGN_SPLIT_SIZE)
    bins: BinSet
    action_space: ActionSpaceProvenance

    @model_validator(mode="after")
    def _validate_calibration(self) -> "CalibrationConfig":
        experiment = self.experiment
        if not isinstance(experiment.aggregation, PlainAverageAggregation):
            raise ValueError(
                "calibration experiment.aggregation must be plain_average; "
                f"got {type(experiment.aggregation).__name__}"
            )
        if experiment.attack.compromised_node is not None:
            raise ValueError(
                "calibration experiment.attack must not preselect a "
                "compromised node"
            )
        if experiment.attack.intensity != "zero":
            raise ValueError(
                "calibration experiment.attack.intensity must be 'zero'"
            )
        if tuple(self.mu_a) != PINNED_MU_A:
            raise ValueError(
                f"mu_a must be the pinned mixture {PINNED_MU_A!r}, got "
                f"{tuple(self.mu_a)!r}"
            )
        _validate_bins(self.bins)
        _validate_action_space(self.action_space)

        # Validate the declared variant and derive its effective mixture up
        # front so an unknown/wrongly-typed variant fails closed at parse time.
        try:
            effective_mixture(self.variant, tuple(self.mu_a))
        except Exception as exc:  # AblationError, bounded and re-raised
            raise ValueError(str(exc)) from None

        node_count = topology_node_count(experiment.topology)
        required = node_count * experiment.data.samples_per_node + self.design_split_size
        if required > MNIST_TRAIN_EXAMPLES:
            raise ValueError(
                "calibration needs "
                f"{required} MNIST training examples "
                f"({node_count} nodes x {experiment.data.samples_per_node} "
                f"+ design {self.design_split_size}), but the training split "
                f"only has {MNIST_TRAIN_EXAMPLES}"
            )
        return self

    @property
    def effective_mu_a(self) -> tuple[float, ...]:
        """Return the variant-derived effective attacker mixture (D036).

        Full and no-belief variants return the pinned source mixture; the
        no-strategic variant returns the degenerate strong mixture ``(5,)``.
        """

        return effective_mixture(self.variant, tuple(self.mu_a))


class _JsonConstant(ValueError):
    """Internal marker: JSON contained NaN/Infinity."""


def _reject_json_constant(value: str) -> float:
    raise _JsonConstant(f"non-finite JSON constant {value!r}")


def _bounded(detail: str) -> str:
    """Collapse whitespace and truncate a diagnostic to a fixed bound."""

    collapsed = " ".join(str(detail).split())
    if len(collapsed) > MAX_ERROR_TEXT:
        collapsed = collapsed[:MAX_ERROR_TEXT] + "..."
    return collapsed


def _parse_structured(text: str, *, source: str) -> object:
    try:
        raw = json.loads(text, parse_constant=_reject_json_constant)
    except _JsonConstant as exc:
        raise CalibrationConfigError(
            f"invalid JSON in {source}: {_bounded(str(exc))}"
        ) from None
    except json.JSONDecodeError:
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise CalibrationConfigError(
                f"invalid YAML in {source}: {_bounded(str(exc))}"
            ) from exc
    return raw


def parse_calibration_text(
    text: str, *, source: str = "<string>"
) -> CalibrationConfig:
    """Parse and validate canonical JSON (preferred) or safe YAML.

    Raises :class:`CalibrationConfigError` with the source path and a bounded
    offending-field summary on any malformed input, non-mapping root,
    non-finite value, unknown key, or invalid field.
    """

    raw = _parse_structured(text, source=source)
    if raw is None:
        raise CalibrationConfigError(
            f"empty calibration config in {source}: expected a mapping"
        )
    if not isinstance(raw, dict):
        raise CalibrationConfigError(
            f"calibration config root in {source} must be a mapping, got "
            f"{type(raw).__name__}"
        )
    try:
        return CalibrationConfig.model_validate(raw)
    except ValidationError as exc:
        raise CalibrationConfigError(
            f"invalid calibration config in {source}: {_bounded(str(exc))}"
        ) from exc


def load_calibration_config(
    path: str | os.PathLike[str],
) -> CalibrationConfig:
    """Read ``path`` from disk and validate it as a :class:`CalibrationConfig`."""

    from pathlib import Path

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CalibrationConfigError(
            f"calibration config file not found: {config_path}"
        ) from exc
    except OSError as exc:
        raise CalibrationConfigError(
            f"could not read calibration config file {config_path}: "
            f"{_bounded(str(exc))}"
        ) from exc
    return parse_calibration_text(text, source=str(config_path))
