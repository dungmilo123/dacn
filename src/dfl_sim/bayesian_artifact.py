"""Strict frozen Bayesian calibration artifact contract (R013/R015/R016/D021).

This module owns the *offline/online boundary* for M002's approximate myopic
Bayesian best-response defense. The artifact is a small, human-auditable,
versioned JSON directory fixed at exactly three files:

``manifest.json``
    Schema identity, sampler/calibration provenance, feature and context bins,
    the explicit feasible joint-type prior, action-space provenance,
    support/coverage metadata, declared payload digests, and an aggregate
    artifact digest.
``likelihood.json``
    List of per-neighbor likelihood rows ``P(x_j | theta_j, c)`` keyed by
    ``(degree_bin, progress, local_type, evidence_bin)``.
``payoff.json``
    List of per-neighbor conditional-payoff rows ``q(x_j, c, m_j, theta_j)``
    keyed by ``(degree_bin, progress, local_type, evidence_bin, multiplier)``.

The tables use **list/tuple row records** (never encoded composite dict keys),
so degree/state/evidence/action cardinalities can be validated explicitly and
the artifact stays diffable.

Model factorization
-------------------
The hidden joint type is ``theta in {0, 1}^d`` (0 benign, 1 malicious) with the
at-most-one-compromised-node support (at most a single 1). The artifact stores
an explicit prior over feasible joint states. Because the evidence and payoff
only depend on a neighbor through that neighbor's own type and multiplier, the
stored tables are per-neighbor and the policy composes them:

* ``P(x | theta, c) = product_j L(x_j | theta_j, c)``
* ``Q(x, c, d, theta) = sum_j q(x_j, c, d_j, theta_j)``

so ``sum_theta posterior(theta) * Q(x, c, d, theta)`` remains exactly
computable while every stored row is degree-independent and hand-checkable.

Untrusted input
---------------
Artifact paths and JSON are untrusted local input (Q3 threat surface). Every
file is read **once** as bounded bytes and those same bytes are used both for
the SHA-256 digest checks and for parsing (no TOCTOU re-read). Symlinked
directories/files, non-regular files, oversized files, duplicate JSON keys,
non-finite constants, unknown fields, unsupported schema/version,
non-probabilities, duplicate rows, infeasible states with more than one
malicious neighbor, invalid multiplier vectors, inconsistent bins/cardinalities,
and coverage outside ``[0, 1]`` all fail closed with a dedicated
:class:`BayesianArtifactError` that names the fixed file and validation class
without echoing table contents.

Nothing here imports the evaluator, simulator, topology, datasets, or model
tensors; the contract is directly reusable by the S02 offline estimator.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

__all__ = [
    "ABLATION_PROTOCOL_VERSION",
    "ARTIFACT_FILENAMES",
    "ARTIFACT_SCHEMA",
    "ARTIFACT_SCHEMA_VERSION",
    "CALIBRATION_VARIANTS",
    "CANDIDATE_PROTOCOL",
    "CANDIDATE_PROTOCOLS",
    "FULL_VARIANT",
    "ISOLATION_STATEMENTS",
    "LIKELIHOOD_FILENAME",
    "MANIFEST_FILENAME",
    "MAX_ARTIFACT_FILE_BYTES",
    "NO_BELIEF_VARIANT",
    "NO_STRATEGIC_EFFECTIVE_MU_A",
    "NO_STRATEGIC_KAPPA",
    "NO_STRATEGIC_VARIANT",
    "PAYOFF_FILENAME",
    "SOURCE_MU_A",
    "ActionSpaceProvenance",
    "ArtifactManifest",
    "BayesianArtifact",
    "BayesianArtifactError",
    "BinSet",
    "CalibrationDerivation",
    "CategoricalBins",
    "CoverageMetadata",
    "IntervalBins",
    "LikelihoodCell",
    "PayloadDigest",
    "PayoffCell",
    "PriorCell",
    "SamplerProvenance",
    "SupportMetadata",
    "canonical_json_bytes",
    "compute_aggregate_digest",
    "interval_bin_index",
    "load_bayesian_artifact",
]

ARTIFACT_SCHEMA = "dfl_bayesian_artifact"
ARTIFACT_SCHEMA_VERSION = 1

MANIFEST_FILENAME = "manifest.json"
LIKELIHOOD_FILENAME = "likelihood.json"
PAYOFF_FILENAME = "payoff.json"
#: Fixed artifact members, in canonical order (manifest first).
ARTIFACT_FILENAMES: tuple[str, ...] = (
    MANIFEST_FILENAME,
    LIKELIHOOD_FILENAME,
    PAYOFF_FILENAME,
)

#: Per-file byte cap (Q6 resource-exhaustion guard). The JSON tables are tiny
#: at M002 scale; anything larger is rejected before parsing.
MAX_ARTIFACT_FILE_BYTES = 8 * 1024 * 1024

#: Hex SHA-256 pattern used for payload and aggregate digests.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Absolute tolerance for prior normalization (unit mass per degree).
_PRIOR_SUM_TOLERANCE = 1e-9

# --------------------------------------------------------------------------- #
# Calibration ablation assumptions and derivation metadata (S03/T01, D036/D037)
# --------------------------------------------------------------------------- #
#: Version of the calibration-derivation assumption protocol. Bumping it
#: invalidates previously pinned match identities.
ABLATION_PROTOCOL_VERSION = "calibration-derivation-v1"

#: The canonical full/source attack mixture (D016 zero/weak/strong kappa 0/2/5).
SOURCE_MU_A: tuple[float, ...] = (0.0, 2.0, 5.0)

#: The pinned strong attacker used by the no-strategic ablation (D036).
NO_STRATEGIC_KAPPA: float = 5.0
NO_STRATEGIC_EFFECTIVE_MU_A: tuple[float, ...] = (NO_STRATEGIC_KAPPA,)

FULL_VARIANT: str = "full"
NO_BELIEF_VARIANT: str = "no_belief"
NO_STRATEGIC_VARIANT: str = "no_strategic"

Variant = Literal["full", "no_belief", "no_strategic"]
CALIBRATION_VARIANTS: tuple[str, ...] = (
    FULL_VARIANT,
    NO_BELIEF_VARIANT,
    NO_STRATEGIC_VARIANT,
)

#: The single feasible-candidate enumeration protocol the action space pins.
CANDIDATE_PROTOCOL = "enumerate_candidate_actions"
CANDIDATE_PROTOCOLS: tuple[str, ...] = (CANDIDATE_PROTOCOL,)

#: Canonical, human-auditable isolation statement per variant.
ISOLATION_STATEMENTS: Mapping[str, str] = MappingProxyType(
    {
        FULL_VARIANT: (
            "Full belief and fixed source attacker mixture (kappa 0/2/5); no "
            "component removed."
        ),
        NO_BELIEF_VARIANT: (
            "Belief component removed: exact uniform posterior over feasible "
            "joint states; payoff tables, action space and fixed source "
            "attacker mixture retained."
        ),
        NO_STRATEGIC_VARIANT: (
            "Fixed-behavior sensitivity: every attacked episode uses the "
            "pinned strong attacker (kappa=5); the belief component is "
            "retained. Calibration attack composition and trajectories differ "
            "from the full source protocol, so the strategic-response "
            "contribution is not fully isolated."
        ),
    }
)


class BayesianArtifactError(RuntimeError):
    """Raised when a frozen artifact is missing, malformed, or tampered.

    Every instance carries the fixed ``filename`` (or ``"artifact"`` for
    directory-level failures) and a machine-checkable ``validation_class`` so a
    caller can localize the failure. Messages name the file and validation
    class only; likelihood/payoff table contents are never echoed.
    """

    def __init__(
        self,
        filename: str,
        validation_class: str,
        detail: str = "",
    ) -> None:
        message = f"{filename}: {validation_class}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.filename = filename
        self.validation_class = validation_class


# --------------------------------------------------------------------------- #
# Strict frozen models
# --------------------------------------------------------------------------- #
def _as_tuple(value: object) -> object:
    """Convert JSON lists to tuples before strict tuple validation."""

    if isinstance(value, list):
        return tuple(value)
    return value


_TupleBefore = BeforeValidator(_as_tuple)
IntTuple = Annotated[tuple[int, ...], _TupleBefore]
FloatTuple = Annotated[tuple[float, ...], _TupleBefore]
StrTuple = Annotated[tuple[str, ...], _TupleBefore]


class _ArtifactModel(BaseModel):
    """Base for every artifact model: unknown keys forbidden, frozen, strict."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CalibrationDerivation(_ArtifactModel):
    """Optional schema-v1 manifest metadata binding a calibrated artifact.

    Every digest is a lowercase SHA-256 hex string. ``source_mu_a`` is pinned
    to the canonical source mixture; ``effective_mu_a`` is derived from
    ``variant``. When present, the artifact loader additionally enforces that
    the declared effective mixture matches the sampler provenance. Legacy M002
    manifests omit this block entirely and load unchanged.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )

    protocol_version: str
    variant: Variant
    isolation_statement: str
    source_mu_a: FloatTuple
    effective_mu_a: FloatTuple
    seed: int = Field(ge=0)
    calibration_config_digest: str
    dataset_inputs_digest: str
    partition_membership_digest: str
    schedule_pairing_digest: str
    episodes: int = Field(ge=0)
    rounds: int = Field(ge=0)
    design_size: int = Field(ge=1)
    candidate_protocol: str
    candidate_cap: int = Field(ge=1)
    data_provenance: Literal["injected", "production"]

    @field_validator(
        "calibration_config_digest",
        "dataset_inputs_digest",
        "partition_membership_digest",
        "schedule_pairing_digest",
    )
    @classmethod
    def _valid_digest(cls, value: str) -> str:
        if _SHA256_RE.match(value) is None:
            raise ValueError(
                "derivation digest must be a lowercase SHA-256 hex string"
            )
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "CalibrationDerivation":
        if self.protocol_version != ABLATION_PROTOCOL_VERSION:
            raise ValueError(
                "unsupported derivation protocol version "
                f"{self.protocol_version!r}"
            )
        if tuple(self.source_mu_a) != SOURCE_MU_A:
            raise ValueError("source_mu_a must be the canonical source mixture")
        expected = (
            NO_STRATEGIC_EFFECTIVE_MU_A
            if self.variant == NO_STRATEGIC_VARIANT
            else SOURCE_MU_A
        )
        if tuple(self.effective_mu_a) != expected:
            raise ValueError(
                "effective_mu_a does not match the declared variant assumption"
            )
        if self.isolation_statement != ISOLATION_STATEMENTS[self.variant]:
            raise ValueError(
                "isolation_statement does not match the declared variant"
            )
        if self.candidate_protocol not in CANDIDATE_PROTOCOLS:
            raise ValueError(
                f"unknown candidate protocol {self.candidate_protocol!r}"
            )
        return self


class SamplerProvenance(_ArtifactModel):
    """Declared sampler/calibration provenance (never trusted, only validated)."""

    reference_policy: Literal["plain_average", "local_only"]
    p0: float
    mu_a: FloatTuple
    placement: Literal["uniform_non_isolated"]
    max_compromised: int
    smoothing_alpha: float
    design_split: Literal["clean_global_heldout"]


class ActionSpaceProvenance(_ArtifactModel):
    """Finite action-space provenance shared with the online policy (D017/D022)."""

    multiplier_set: FloatTuple
    self_multiplier: float
    omega: float
    degree_cap: int
    tie_tolerance: float
    ordering: Literal["fewer_rejected_then_lexicographic"]
    pruning: Literal["directional_disagreement_desc_sender_id_asc"]


class IntervalBins(_ArtifactModel):
    """Feature bins as strictly increasing finite ``edges`` plus ``len+1`` labels.

    Bin ``i`` covers values in ``[edges[i - 1], edges[i])`` (with implicit
    ``-inf`` / ``+inf`` outer half-open ends). The infinity endpoints are never
    serialized, so the artifact stays finite. ``interval_bin_index`` applies the
    convention ``bisect_right(edges, value)``.
    """

    edges: FloatTuple
    labels: StrTuple


class CategoricalBins(_ArtifactModel):
    """Context bins with explicit, unique string labels (no numeric edges)."""

    labels: StrTuple


class BinSet(_ArtifactModel):
    """The feature and context bin declarations referenced by every table row."""

    disagreement: IntervalBins
    relative_norm: IntervalBins
    loss_delta: IntervalBins
    degree: CategoricalBins
    progress: CategoricalBins


class PriorCell(_ArtifactModel):
    """One feasible joint-type prior entry ``P(theta = state)``."""

    degree: int
    state: IntTuple
    probability: float


class LikelihoodCell(_ArtifactModel):
    """One per-neighbor likelihood row ``P(x_j | theta_j, c)``."""

    degree_bin: str
    progress: str
    local_type: int
    evidence_bin: IntTuple
    probability: float


class PayoffCell(_ArtifactModel):
    """One per-neighbor conditional-payoff row ``q(x_j, c, m_j, theta_j)``."""

    degree_bin: str
    progress: str
    local_type: int
    evidence_bin: IntTuple
    multiplier: float
    value: float


class SupportMetadata(_ArtifactModel):
    """Cell/collection counts used to cross-check declared table cardinalities."""

    calibration_episodes: int
    calibration_rounds: int
    prior_cells: int
    likelihood_cells: int
    payoff_cells: int
    min_cell_count: int


class CoverageMetadata(_ArtifactModel):
    """Coverage fractions in ``[0, 1]`` for the frozen tables.

    ``posterior_calibration`` is the bounded ``1 - mean Brier`` belief score.
    It is only meaningful when the prior/likelihood are actually estimated from
    observations; uniform-posterior variants set
    ``posterior_calibration_applicable=False`` so a full-policy Brier score is
    never presented as meaningful no-belief evidence (D036).
    """

    likelihood: float
    payoff: float
    posterior_calibration: float
    posterior_calibration_applicable: bool = True


class PayloadDigest(_ArtifactModel):
    """Declared size and SHA-256 of one non-manifest payload file."""

    size_bytes: int
    sha256: str


class ArtifactManifest(_ArtifactModel):
    """Strict schema-v1 manifest envelope.

    ``derivation`` is an optional, additive calibration-ablation metadata block
    (D037). Legacy M002 manifests that omit it still load and verify exactly as
    before; newly calibrated artifacts (T02) populate it. When present the
    loader enforces its internal variant/assumption consistency and binds its
    effective attacker mixture to the sampler provenance.
    """

    schema_name: str = Field(alias="schema")
    schema_version: int
    provenance: SamplerProvenance
    action_space: ActionSpaceProvenance
    bins: BinSet
    prior: Annotated[tuple[PriorCell, ...], _TupleBefore]
    support: SupportMetadata
    coverage: CoverageMetadata
    payloads: dict[str, PayloadDigest]
    derivation: CalibrationDerivation | None = None
    aggregate_digest: str

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, populate_by_name=True
    )

    @model_validator(mode="after")
    def _payload_set(self) -> "ArtifactManifest":
        if set(self.payloads) != {LIKELIHOOD_FILENAME, PAYOFF_FILENAME}:
            raise ValueError("manifest payload set must be exactly the two tables")
        return self


# --------------------------------------------------------------------------- #
# Canonical digest helpers (shared with the S02 producer)
# --------------------------------------------------------------------------- #
def canonical_json_bytes(obj: Any) -> bytes:
    """Return the canonical, finite JSON encoding used by digest computation."""

    try:
        text = json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise BayesianArtifactError(
            MANIFEST_FILENAME, "schema", "artifact is not canonically encodable"
        ) from exc
    return text.encode("utf-8")


def compute_aggregate_digest(manifest_body: Mapping[str, Any]) -> str:
    """Return the aggregate digest over ``manifest_body`` minus its own digest.

    The digest covers every manifest field (schema/version, provenance, bins,
    prior, support, coverage, action-space, and the declared payload digests)
    except ``aggregate_digest`` itself, so a tampered manifest body that keeps
    structural validity is still detected.
    """

    body = {key: value for key, value in manifest_body.items() if key != "aggregate_digest"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# JSON reading (same bytes for digest and parse)
# --------------------------------------------------------------------------- #
class _JsonRejection(Exception):
    """Internal marker raised by the strict JSON hooks."""

    def __init__(self, validation_class: str) -> None:
        super().__init__(validation_class)
        self.validation_class = validation_class


def _reject_json_constant(value: str) -> Any:
    raise _JsonRejection("non_finite")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _JsonRejection("duplicate_key")
        result[key] = value
    return result


def _parse_json(data: bytes, *, filename: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BayesianArtifactError(
            filename, "malformed_json", "payload is not valid UTF-8"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise BayesianArtifactError(filename, "unreadable", "could not read payload") from exc

    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except _JsonRejection as exc:
        raise BayesianArtifactError(filename, exc.validation_class) from None
    except json.JSONDecodeError as exc:
        raise BayesianArtifactError(
            filename, "malformed_json", f"line {exc.lineno} column {exc.colno}"
        ) from None


def _first_error(exc: ValidationError) -> tuple[str, str]:
    """Return a sanitized ``(location, error type)`` for the first failure."""

    error = exc.errors()[0]
    location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
    return location, str(error.get("type", "invalid"))


def _validate_model(
    model: type[BaseModel],
    data: Any,
    *,
    filename: str,
    validation_class: str,
) -> BaseModel:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        location, error_type = _first_error(exc)
        raise BayesianArtifactError(
            filename, validation_class, f"invalid field {location!r} ({error_type})"
        ) from None


# --------------------------------------------------------------------------- #
# Bounded, symlink-safe file reads
# --------------------------------------------------------------------------- #
def _read_fixed_file(base: str, filename: str) -> bytes:
    """Read one fixed artifact file once as bounded bytes.

    Rejects symlinks, non-regular files, and oversized inputs before reading,
    then re-checks the opened descriptor so a race cannot substitute a
    different inode between ``lstat`` and ``read``.
    """

    path = os.path.join(base, filename)
    try:
        before = os.lstat(path)
    except FileNotFoundError as exc:
        raise BayesianArtifactError(filename, "missing") from exc
    except OSError as exc:
        raise BayesianArtifactError(filename, "unreadable") from exc

    if stat.S_ISLNK(before.st_mode):
        raise BayesianArtifactError(filename, "symlink")
    if not stat.S_ISREG(before.st_mode):
        raise BayesianArtifactError(filename, "non_regular")
    if before.st_size > MAX_ARTIFACT_FILE_BYTES:
        raise BayesianArtifactError(
            filename,
            "oversized",
            f"limit {MAX_ARTIFACT_FILE_BYTES} bytes",
        )

    try:
        with open(path, "rb") as handle:
            after = os.fstat(handle.fileno())
            if stat.S_ISLNK(after.st_mode) or not stat.S_ISREG(after.st_mode):
                raise BayesianArtifactError(filename, "non_regular")
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise BayesianArtifactError(filename, "changed")
            data = handle.read(MAX_ARTIFACT_FILE_BYTES + 1)
    except BayesianArtifactError:
        raise
    except OSError as exc:
        raise BayesianArtifactError(filename, "unreadable") from exc

    if len(data) > MAX_ARTIFACT_FILE_BYTES:
        raise BayesianArtifactError(
            filename,
            "oversized",
            f"limit {MAX_ARTIFACT_FILE_BYTES} bytes",
        )
    return data


# --------------------------------------------------------------------------- #
# Semantic validation
# --------------------------------------------------------------------------- #
def _check(condition: bool, filename: str, validation_class: str, detail: str = "") -> None:
    if not condition:
        raise BayesianArtifactError(filename, validation_class, detail)


def _is_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_interval_bins(bins: IntervalBins, *, name: str) -> None:
    _check(
        len(bins.labels) == len(bins.edges) + 1,
        MANIFEST_FILENAME,
        "bins",
        f"{name} label/edge cardinality mismatch",
    )
    previous: float | None = None
    for edge in bins.edges:
        _check(_is_finite(edge), MANIFEST_FILENAME, "bins", f"{name} non-finite edge")
        if previous is not None:
            _check(edge > previous, MANIFEST_FILENAME, "bins", f"{name} edges not increasing")
        previous = edge
    _check_labels_unique(bins.labels, name)


def _check_labels_unique(labels: tuple[str, ...], name: str) -> None:
    _check(
        len(labels) == len(set(labels)),
        MANIFEST_FILENAME,
        "bins",
        f"{name} labels not unique",
    )
    _check(
        all(isinstance(label, str) and label for label in labels),
        MANIFEST_FILENAME,
        "bins",
        f"{name} has an empty label",
    )


def _validate_action_space(action_space: ActionSpaceProvenance) -> None:
    cls = "action_space"
    omega = action_space.omega
    _check(_is_finite(omega) and 0.0 < omega < 1.0, MANIFEST_FILENAME, cls, "omega must be in (0, 1)")
    _check(
        action_space.self_multiplier == 1.0,
        MANIFEST_FILENAME,
        cls,
        "self_multiplier must be exactly 1.0",
    )
    _check(
        action_space.degree_cap >= 1,
        MANIFEST_FILENAME,
        cls,
        "degree_cap must be >= 1",
    )
    _check(
        _is_finite(action_space.tie_tolerance) and action_space.tie_tolerance >= 0.0,
        MANIFEST_FILENAME,
        cls,
        "tie_tolerance must be finite and >= 0",
    )
    multipliers = action_space.multiplier_set
    _check(
        len(multipliers) == 3 and all(_is_finite(m) for m in multipliers),
        MANIFEST_FILENAME,
        cls,
        "multiplier_set must hold three finite multipliers",
    )
    _check(
        sorted(multipliers) == [0.0, omega, 1.0],
        MANIFEST_FILENAME,
        cls,
        "multiplier_set must be exactly {0, omega, 1}",
    )


def _validate_provenance(provenance: SamplerProvenance) -> None:
    cls = "provenance"
    _check(
        _is_finite(provenance.p0) and 0.0 <= provenance.p0 <= 1.0,
        MANIFEST_FILENAME,
        cls,
        "p0 must be in [0, 1]",
    )
    _check(
        len(provenance.mu_a) > 0 and all(_is_finite(v) and v >= 0.0 for v in provenance.mu_a),
        MANIFEST_FILENAME,
        cls,
        "mu_a must be a non-empty list of non-negative finite values",
    )
    _check(
        provenance.max_compromised == 1,
        MANIFEST_FILENAME,
        cls,
        "max_compromised must be exactly 1",
    )
    _check(
        _is_finite(provenance.smoothing_alpha) and provenance.smoothing_alpha > 0.0,
        MANIFEST_FILENAME,
        cls,
        "smoothing_alpha must be finite and > 0",
    )


def _validate_support(support: SupportMetadata) -> None:
    cls = "support"
    for name in (
        "calibration_episodes",
        "calibration_rounds",
        "prior_cells",
        "likelihood_cells",
        "payoff_cells",
        "min_cell_count",
    ):
        value = getattr(support, name)
        _check(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            MANIFEST_FILENAME,
            cls,
            f"{name} must be a non-negative integer",
        )


def _validate_coverage(coverage: CoverageMetadata) -> None:
    cls = "coverage"
    for name in ("likelihood", "payoff", "posterior_calibration"):
        value = getattr(coverage, name)
        _check(
            _is_finite(value) and 0.0 <= value <= 1.0,
            MANIFEST_FILENAME,
            cls,
            f"{name} must be in [0, 1]",
        )
    _check(
        isinstance(coverage.posterior_calibration_applicable, bool),
        MANIFEST_FILENAME,
        cls,
        "posterior_calibration_applicable must be a boolean",
    )


def _validate_prior(prior: tuple[PriorCell, ...]) -> None:
    seen: set[tuple[int, tuple[int, ...]]] = set()
    totals: dict[int, float] = {}
    for index, cell in enumerate(prior):
        _check(cell.degree >= 0, MANIFEST_FILENAME, "state", f"prior row {index} negative degree")
        _check(
            len(cell.state) == cell.degree,
            MANIFEST_FILENAME,
            "cardinality",
            f"prior row {index} state length does not match degree",
        )
        _check(
            all(bit in (0, 1) for bit in cell.state),
            MANIFEST_FILENAME,
            "state",
            f"prior row {index} state must be binary",
        )
        _check(
            sum(cell.state) <= 1,
            MANIFEST_FILENAME,
            "state",
            f"prior row {index} has more than one malicious neighbor",
        )
        _check(
            _is_finite(cell.probability) and 0.0 <= cell.probability <= 1.0,
            MANIFEST_FILENAME,
            "probability",
            f"prior row {index} probability out of range",
        )
        key = (cell.degree, tuple(cell.state))
        _check(key not in seen, MANIFEST_FILENAME, "duplicate_row", f"prior row {index} duplicated")
        seen.add(key)
        totals[cell.degree] = totals.get(cell.degree, 0.0) + cell.probability

    for degree, total in totals.items():
        _check(
            abs(total - 1.0) <= _PRIOR_SUM_TOLERANCE,
            MANIFEST_FILENAME,
            "prior",
            f"degree {degree} prior mass does not sum to 1",
        )


def _validate_derivation(manifest: ArtifactManifest) -> None:
    """Bind optional calibration-derivation metadata to the sampler provenance.

    Internal (variant/source/effective/isolation/budget) consistency is already
    enforced by :class:`CalibrationDerivation`. Here we check the cross-field
    binding that only the whole manifest can verify: the declared effective
    attacker mixture must equal the sampler provenance's recorded mixture, and
    the candidate cap must equal the action-space degree cap.
    """

    derivation = manifest.derivation
    if derivation is None:
        return
    _check(
        tuple(manifest.provenance.mu_a) == tuple(derivation.effective_mu_a),
        MANIFEST_FILENAME,
        "derivation",
        "declared effective mixture does not match sampler provenance",
    )
    _check(
        derivation.candidate_cap == manifest.action_space.degree_cap,
        MANIFEST_FILENAME,
        "derivation",
        "declared candidate cap does not match action-space degree cap",
    )


def _validate_manifest(manifest: ArtifactManifest) -> None:
    _validate_provenance(manifest.provenance)
    _validate_action_space(manifest.action_space)
    _validate_derivation(manifest)
    _validate_interval_bins(manifest.bins.disagreement, name="disagreement")
    _validate_interval_bins(manifest.bins.relative_norm, name="relative_norm")
    _validate_interval_bins(manifest.bins.loss_delta, name="loss_delta")
    _check_labels_unique(manifest.bins.degree.labels, "degree")
    _check_labels_unique(manifest.bins.progress.labels, "progress")
    _validate_prior(manifest.prior)
    _validate_support(manifest.support)
    _validate_coverage(manifest.coverage)
    _check(
        _SHA256_RE.match(manifest.aggregate_digest) is not None,
        MANIFEST_FILENAME,
        "aggregate_digest",
        "aggregate digest is not a SHA-256 hex string",
    )
    for filename, digest in manifest.payloads.items():
        _check(
            digest.size_bytes >= 0,
            MANIFEST_FILENAME,
            "support",
            f"declared size for {filename} is negative",
        )
        _check(
            _SHA256_RE.match(digest.sha256) is not None,
            MANIFEST_FILENAME,
            "digest",
            f"declared digest for {filename} is not a SHA-256 hex string",
        )


def _evidence_cardinality(manifest: ArtifactManifest) -> tuple[int, int, int]:
    bins = manifest.bins
    return (
        len(bins.disagreement.edges) + 1,
        len(bins.relative_norm.edges) + 1,
        len(bins.loss_delta.edges) + 1,
    )


def _validate_evidence(
    evidence: tuple[int, ...],
    *,
    manifest: ArtifactManifest,
    filename: str,
    row_index: int,
) -> tuple[int, int, int]:
    _check(
        len(evidence) == 3,
        filename,
        "cardinality",
        f"row {row_index} evidence_bin must have three indices",
    )
    cardinality = _evidence_cardinality(manifest)
    for value, limit in zip(evidence, cardinality):
        _check(
            0 <= value < limit,
            filename,
            "bins",
            f"row {row_index} evidence index out of range",
        )
    return (evidence[0], evidence[1], evidence[2])


def _validate_context(
    *,
    degree_bin: str,
    progress: str,
    local_type: int,
    manifest: ArtifactManifest,
    filename: str,
    row_index: int,
) -> None:
    _check(
        degree_bin in manifest.bins.degree.labels,
        filename,
        "bins",
        f"row {row_index} unknown degree_bin",
    )
    _check(
        progress in manifest.bins.progress.labels,
        filename,
        "bins",
        f"row {row_index} unknown progress bin",
    )
    _check(local_type in (0, 1), filename, "state", f"row {row_index} local_type must be 0 or 1")


def _validate_likelihood_rows(
    rows: object,
    *,
    manifest: ArtifactManifest,
    filename: str,
) -> tuple[tuple[tuple[str, str, int, tuple[int, int, int]], float], ...]:
    _check(isinstance(rows, list), filename, "schema", "payload must be a JSON array")
    entries: list[tuple[tuple[str, str, int, tuple[int, int, int]], float]] = []
    seen: set[tuple[str, str, int, tuple[int, int, int]]] = set()
    for index, raw in enumerate(rows):
        row = _validate_model(LikelihoodCell, raw, filename=filename, validation_class="schema")
        assert isinstance(row, LikelihoodCell)
        _validate_context(
            degree_bin=row.degree_bin,
            progress=row.progress,
            local_type=row.local_type,
            manifest=manifest,
            filename=filename,
            row_index=index,
        )
        evidence = _validate_evidence(
            row.evidence_bin, manifest=manifest, filename=filename, row_index=index
        )
        _check(
            _is_finite(row.probability) and 0.0 <= row.probability <= 1.0,
            filename,
            "probability",
            f"row {index} probability out of range",
        )
        key = (row.degree_bin, row.progress, row.local_type, evidence)
        _check(key not in seen, filename, "duplicate_row", f"row {index} duplicated")
        seen.add(key)
        entries.append((key, row.probability))
    return tuple(entries)


def _validate_payoff_rows(
    rows: object,
    *,
    manifest: ArtifactManifest,
    filename: str,
) -> tuple[tuple[tuple[str, str, int, tuple[int, int, int], float], float], ...]:
    _check(isinstance(rows, list), filename, "schema", "payload must be a JSON array")
    allowed = set(manifest.action_space.multiplier_set)
    entries: list[tuple[tuple[str, str, int, tuple[int, int, int], float], float]] = []
    seen: set[tuple[str, str, int, tuple[int, int, int], float]] = set()
    for index, raw in enumerate(rows):
        row = _validate_model(PayoffCell, raw, filename=filename, validation_class="schema")
        assert isinstance(row, PayoffCell)
        _validate_context(
            degree_bin=row.degree_bin,
            progress=row.progress,
            local_type=row.local_type,
            manifest=manifest,
            filename=filename,
            row_index=index,
        )
        evidence = _validate_evidence(
            row.evidence_bin, manifest=manifest, filename=filename, row_index=index
        )
        _check(
            _is_finite(row.multiplier) and row.multiplier in allowed,
            filename,
            "action_space",
            f"row {index} multiplier is not in the declared action set",
        )
        _check(_is_finite(row.value), filename, "non_finite", f"row {index} value is not finite")
        key = (row.degree_bin, row.progress, row.local_type, evidence, row.multiplier)
        _check(key not in seen, filename, "duplicate_row", f"row {index} duplicated")
        seen.add(key)
        entries.append((key, row.value))
    return tuple(entries)


#: Absolute tolerance for the uniform-posterior ablation consistency checks.
_UNIFORM_TOLERANCE = 1e-9


def _feasible_prior_states(degree: int) -> tuple[tuple[int, ...], ...]:
    """Return the at-most-one-malicious joint states for ``degree``."""

    states: list[tuple[int, ...]] = [(0,) * degree]
    for position in range(degree):
        state = [0] * degree
        state[position] = 1
        states.append(tuple(state))
    return tuple(states)


def _validate_no_belief_tables(
    manifest: ArtifactManifest,
    likelihood_entries: tuple[
        tuple[tuple[str, str, int, tuple[int, int, int]], float], ...
    ],
) -> None:
    """Enforce the exact uniform-posterior contract on a no-belief artifact.

    A ``no_belief`` artifact must carry a uniform prior over the *complete*
    feasible joint-state support per degree and a uniform likelihood over the
    *complete* evidence support for both local types in every context. A
    nonuniform prior or an evidence-dependent likelihood contradicts the
    declared variant and is rejected rather than silently accepted as
    no-belief evidence (D036).
    """

    prior_by_degree: dict[int, list[float]] = {}
    for cell in manifest.prior:
        prior_by_degree.setdefault(cell.degree, []).append(cell.probability)
    for degree, probabilities in prior_by_degree.items():
        _check(
            len(probabilities) == len(_feasible_prior_states(degree)),
            MANIFEST_FILENAME,
            "derivation",
            f"no-belief prior degree {degree} does not cover the feasible support",
        )
        _check(
            max(probabilities) - min(probabilities) <= _UNIFORM_TOLERANCE,
            MANIFEST_FILENAME,
            "derivation",
            f"no-belief prior degree {degree} is not uniform",
        )

    cardinality = _evidence_cardinality(manifest)
    support_size = cardinality[0] * cardinality[1] * cardinality[2]
    by_context: dict[tuple[str, str, int], list[float]] = {}
    for (degree_bin, progress, local_type, _evidence), probability in likelihood_entries:
        by_context.setdefault((degree_bin, progress, local_type), []).append(probability)
    for context, probabilities in by_context.items():
        _check(
            len(probabilities) == support_size,
            LIKELIHOOD_FILENAME,
            "derivation",
            "no-belief likelihood does not cover the full evidence support",
        )
        _check(
            max(probabilities) - min(probabilities) <= _UNIFORM_TOLERANCE,
            LIKELIHOOD_FILENAME,
            "derivation",
            "no-belief likelihood is evidence-dependent",
        )


def _validate_belief_applicability(
    manifest: ArtifactManifest,
    likelihood_entries: tuple[
        tuple[tuple[str, str, int, tuple[int, int, int]], float], ...
    ],
) -> None:
    """Bind the declared variant to the honest belief-metric applicability.

    ``no_belief`` artifacts must mark the Brier-style posterior diagnostic
    non-applicable and prove the exact uniform-posterior table contract; every
    other variant (and a legacy artifact with no derivation) must leave it
    applicable.
    """

    derivation = manifest.derivation
    if derivation is not None and derivation.variant == NO_BELIEF_VARIANT:
        _check(
            manifest.coverage.posterior_calibration_applicable is False,
            MANIFEST_FILENAME,
            "derivation",
            "no-belief coverage must mark belief calibration non-applicable",
        )
        _validate_no_belief_tables(manifest, likelihood_entries)
    else:
        _check(
            manifest.coverage.posterior_calibration_applicable is True,
            MANIFEST_FILENAME,
            "derivation",
            "belief variants must mark posterior calibration applicable",
        )


# --------------------------------------------------------------------------- #
# Immutable indexed artifact
# --------------------------------------------------------------------------- #
LikelihoodKey = tuple[str, str, int, tuple[int, int, int]]
PayoffKey = tuple[str, str, int, tuple[int, int, int], float]


@dataclass(frozen=True, slots=True)
class BayesianArtifact:
    """Immutable, indexed view of a validated schema-v1 artifact.

    Construction is private to :func:`load_bayesian_artifact`; the exposed
    tables are read-only mappings and every row model is frozen.
    """

    manifest: ArtifactManifest
    prior_index: Mapping[int, tuple[PriorCell, ...]]
    likelihood_index: Mapping[LikelihoodKey, float]
    payoff_index: Mapping[PayoffKey, float]

    # -- action-space / numerics accessors ---------------------------------- #
    @property
    def omega(self) -> float:
        return self.manifest.action_space.omega

    @property
    def multiplier_set(self) -> tuple[float, ...]:
        return self.manifest.action_space.multiplier_set

    @property
    def self_multiplier(self) -> float:
        return self.manifest.action_space.self_multiplier

    @property
    def degree_cap(self) -> int:
        return self.manifest.action_space.degree_cap

    @property
    def tie_tolerance(self) -> float:
        return self.manifest.action_space.tie_tolerance

    # -- table lookups ------------------------------------------------------ #
    def prior_for_degree(self, degree: int) -> tuple[PriorCell, ...]:
        """Return the feasible prior cells for ``degree`` (empty when absent)."""

        return self.prior_index.get(degree, ())

    def prior_probability(self, degree: int, state: tuple[int, ...]) -> float | None:
        """Return ``P(theta = state)`` for a degree, or ``None`` when absent."""

        for cell in self.prior_for_degree(degree):
            if cell.state == state:
                return cell.probability
        return None

    def likelihood_for(
        self,
        degree_bin: str,
        progress: str,
        local_type: int,
        evidence_bin: tuple[int, int, int],
    ) -> float | None:
        """Return ``P(x | local_type, c)`` or ``None`` for a missing cell."""

        return self.likelihood_index.get((degree_bin, progress, local_type, evidence_bin))

    def payoff_for(
        self,
        degree_bin: str,
        progress: str,
        local_type: int,
        evidence_bin: tuple[int, int, int],
        multiplier: float,
    ) -> float | None:
        """Return ``q(x, c, multiplier, local_type)`` or ``None`` when absent."""

        return self.payoff_index.get(
            (degree_bin, progress, local_type, evidence_bin, multiplier)
        )


def interval_bin_index(bins: IntervalBins, value: float) -> int:
    """Return the bin index for ``value`` under the artifact interval convention.

    Bins are half-open ``[edges[i - 1], edges[i])`` with implicit infinite outer
    ends; an exact edge value falls into the upper bin (``bisect_right``).
    """

    _check(_is_finite(value), MANIFEST_FILENAME, "non_finite", "cannot bin a non-finite value")
    return bisect.bisect_right(bins.edges, value)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def _read_directory(base: str) -> None:
    try:
        info = os.lstat(base)
    except FileNotFoundError as exc:
        raise BayesianArtifactError("artifact", "missing", "directory does not exist") from exc
    except OSError as exc:
        raise BayesianArtifactError("artifact", "unreadable") from exc

    if stat.S_ISLNK(info.st_mode):
        raise BayesianArtifactError("artifact", "symlink", "directory must not be a symlink")
    if not stat.S_ISDIR(info.st_mode):
        raise BayesianArtifactError("artifact", "non_regular", "path is not a directory")

    try:
        names = os.listdir(base)
    except OSError as exc:
        raise BayesianArtifactError("artifact", "unreadable") from exc

    missing = sorted(set(ARTIFACT_FILENAMES) - set(names))
    if missing:
        raise BayesianArtifactError(missing[0], "missing")
    extra = sorted(set(names) - set(ARTIFACT_FILENAMES))
    if extra:
        raise BayesianArtifactError(extra[0], "unexpected_file")


def load_bayesian_artifact(path: str | os.PathLike[str]) -> BayesianArtifact:
    """Load and strictly validate a schema-v1 Bayesian artifact directory.

    Reads ``manifest.json`` first, validates schema/provenance/bins/prior/
    support/coverage/action-space, then reads and digest-checks the two payload
    files once, parses and validates their rows, verifies the aggregate digest
    over the whole manifest body, and only then builds the immutable indexes.

    Any failure raises :class:`BayesianArtifactError` naming the fixed file and
    a validation class, before any policy is constructed.
    """

    base = os.fspath(path)
    _read_directory(base)

    manifest_bytes = _read_fixed_file(base, MANIFEST_FILENAME)
    raw_manifest = _parse_json(manifest_bytes, filename=MANIFEST_FILENAME)
    _check(
        isinstance(raw_manifest, dict),
        MANIFEST_FILENAME,
        "schema",
        "manifest root must be a JSON object",
    )
    manifest = _validate_model(
        ArtifactManifest, raw_manifest, filename=MANIFEST_FILENAME, validation_class="schema"
    )
    assert isinstance(manifest, ArtifactManifest)

    if manifest.schema_name != ARTIFACT_SCHEMA:
        raise BayesianArtifactError(MANIFEST_FILENAME, "schema", "unsupported artifact schema")
    if manifest.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise BayesianArtifactError(
            MANIFEST_FILENAME, "version", "unsupported artifact schema version"
        )
    _validate_manifest(manifest)

    payload_bytes: dict[str, bytes] = {}
    for filename in (LIKELIHOOD_FILENAME, PAYOFF_FILENAME):
        data = _read_fixed_file(base, filename)
        payload_bytes[filename] = data
        declared = manifest.payloads[filename]
        _check(
            len(data) == declared.size_bytes,
            filename,
            "digest",
            "declared size does not match payload",
        )
        _check(
            _sha256(data) == declared.sha256,
            filename,
            "digest",
            "declared digest does not match payload",
        )

    raw_likelihood = _parse_json(payload_bytes[LIKELIHOOD_FILENAME], filename=LIKELIHOOD_FILENAME)
    raw_payoff = _parse_json(payload_bytes[PAYOFF_FILENAME], filename=PAYOFF_FILENAME)
    likelihood_entries = _validate_likelihood_rows(
        raw_likelihood, manifest=manifest, filename=LIKELIHOOD_FILENAME
    )
    payoff_entries = _validate_payoff_rows(
        raw_payoff, manifest=manifest, filename=PAYOFF_FILENAME
    )

    _validate_belief_applicability(manifest, likelihood_entries)

    _check(
        manifest.support.prior_cells == len(manifest.prior),
        MANIFEST_FILENAME,
        "support",
        "declared prior_cells does not match manifest prior",
    )
    _check(
        manifest.support.likelihood_cells == len(likelihood_entries),
        MANIFEST_FILENAME,
        "support",
        "declared likelihood_cells does not match payload",
    )
    _check(
        manifest.support.payoff_cells == len(payoff_entries),
        MANIFEST_FILENAME,
        "support",
        "declared payoff_cells does not match payload",
    )

    _check(
        compute_aggregate_digest(raw_manifest) == manifest.aggregate_digest,
        MANIFEST_FILENAME,
        "aggregate_digest",
        "aggregate digest does not match manifest body",
    )

    return _build_artifact(
        manifest=manifest,
        likelihood=dict(likelihood_entries),
        payoff=dict(payoff_entries),
    )


def _build_artifact(
    *,
    manifest: ArtifactManifest,
    likelihood: Mapping[LikelihoodKey, float],
    payoff: Mapping[PayoffKey, float],
) -> BayesianArtifact:
    prior: dict[int, list[PriorCell]] = {}
    for cell in manifest.prior:
        prior.setdefault(cell.degree, []).append(cell)
    prior_index = {degree: tuple(cells) for degree, cells in prior.items()}
    return BayesianArtifact(
        manifest=manifest,
        prior_index=MappingProxyType(prior_index),
        likelihood_index=MappingProxyType(dict(likelihood)),
        payoff_index=MappingProxyType(dict(payoff)),
    )
