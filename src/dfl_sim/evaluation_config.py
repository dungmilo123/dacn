"""Strict, frozen baseline-study configuration for the M003 evaluation harness.

This module owns the *study spec* contract that S01 introduces and S02 extends.
A study is the fairness-enforcing envelope around the unchanged
:class:`~dfl_sim.config.ExperimentConfig`: it declares one frozen base
experiment, a bounded set of seeds, and a complete six-policy baseline panel.
Every cell a runner derives from the study differs from the base only in
``seed`` and ``aggregation``, so paired seeds share partitions,
initialization, and attacker while the aggregation policy is the sole
experimental variable.

Configuration is treated as untrusted structured input (Q3 threat surface):

* Canonical JSON is tried first and :func:`yaml.safe_load` is the fallback, so
  a canonical snapshot round-trips exactly while ordinary YAML keeps working.
  ``yaml.safe_load`` never constructs Python objects from ``!!python/...`` tags.
* Every model forbids unknown keys (``extra="forbid"``), is frozen, is strict,
  and rejects non-finite floats (``allow_inf_nan=False``).
* Seed count, panel cardinality, identifier length, labels, and the matched
  candidate budget are all bounded.
* Error text is bounded so a hostile or accidental invalid spec cannot flood
  logs with an unbounded validation dump.

The panel must contain exactly the six M003 baseline identities
(:data:`REQUIRED_POLICY_IDS`). Tunable policies must declare the study's single
positive candidate budget and selection-seed provenance; parameter-free
policies must explicitly declare zero/exempt rather than pretending to tune.
The BALANCE entry's citation is pinned to the D028 external reference.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .config import (
    MAX_SEED,
    AggregationConfig,
    ConfigError,
    ExperimentConfig,
    PlainAverageAggregation,
)

__all__ = [
    "BALANCE_CITATION",
    "MAX_CANDIDATE_BUDGET",
    "MAX_STUDY_ID_LEN",
    "MAX_STUDY_SEEDS",
    "REQUIRED_POLICY_IDS",
    "BaselineStudyConfig",
    "BaselineStudyConfigError",
    "CitationMetadata",
    "PanelPolicy",
    "TuningProvenance",
    "hyperparameter_digest",
    "load_study_config",
    "parse_study_text",
    "policy_identity",
    "selected_hyperparameters",
    "study_digest",
    "study_identity_payload",
]

# --------------------------------------------------------------------------- #
# Bounds and pinned literals (Q3/Q6: out-of-range values are rejected)
# --------------------------------------------------------------------------- #
#: A study needs at least one seed and is capped so a single run stays bounded.
MIN_STUDY_SEEDS = 1
MAX_STUDY_SEEDS = 64

#: Maximum length of the study identifier.
MAX_STUDY_ID_LEN = 64

#: Maximum length of any human-readable label/implementation string.
MAX_LABEL_LEN = 256

#: Maximum number of tuning candidates a single study may declare.
MAX_CANDIDATE_BUDGET = 4096

#: Maximum rendered error-message length (Q3 bounded diagnostics).
MAX_ERROR_TEXT = 400

#: Canonical policy order used for deterministic manifest entry ordering.
REQUIRED_POLICY_IDS: tuple[str, ...] = (
    "local_only",
    "plain_average",
    "fixed_similarity",
    "trust_weighted",
    "balance",
    "bayesian",
)

#: Panel identities whose hyperparameters are selected under the matched
#: candidate budget, versus the structurally parameter-free baselines.
_TUNABLE_POLICY_IDS: frozenset[str] = frozenset(
    {"fixed_similarity", "trust_weighted", "balance", "bayesian"}
)
_PARAMETER_FREE_POLICY_IDS: frozenset[str] = frozenset(
    {"local_only", "plain_average"}
)

#: Expected aggregation config ``kind`` for each stable policy id. Identity and
#: aggregation kind are cross-checked so a panel cannot mislabel a policy.
_POLICY_KINDS: dict[str, str] = {
    "local_only": "local_only",
    "plain_average": "plain_average",
    "fixed_similarity": "fixed_similarity",
    "trust_weighted": "trust_weighted",
    "balance": "balance",
    "bayesian": "bayesian",
}

#: D028 pinned BALANCE citation (Fang et al., ACM CCS 2024, arXiv:2406.10416v4,
#: Algorithm 3). A BALANCE panel entry must reproduce these fields exactly.
BALANCE_CITATION: dict[str, object] = {
    "kind": "external",
    "title": "Byzantine-Robust Decentralized Federated Learning",
    "venue": "ACM CCS 2024",
    "year": 2024,
    "identifier": "arXiv:2406.10416v4",
    "algorithm": "Algorithm 3",
    "url": "https://arxiv.org/abs/2406.10416",
}


class BaselineStudyConfigError(ConfigError):
    """Raised when a baseline-study YAML/JSON spec is missing or invalid."""


def _as_tuple(value: object) -> object:
    """Convert JSON/YAML lists to tuples before strict tuple validation."""

    if isinstance(value, list):
        return tuple(value)
    return value


SeedTuple = Annotated[tuple[int, ...], BeforeValidator(_as_tuple)]


class _StrictStudyModel(BaseModel):
    """Base for study leaves: unknown keys forbidden, frozen, strict, finite."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class CitationMetadata(_StrictStudyModel):
    """Provenance/version metadata for one panel policy.

    ``kind="external"`` identifies an external published method reproduction and
    therefore requires a non-empty ``identifier`` (for example an arXiv id);
    ``kind="project"`` identifies a project-local comparator (D029) and requires
    only a title. Fields are optional so a project method need not fabricate a
    venue.
    """

    kind: Literal["project", "external"]
    title: str = Field(min_length=1, max_length=MAX_LABEL_LEN)
    venue: str | None = Field(default=None, max_length=MAX_LABEL_LEN)
    year: int | None = Field(default=None, ge=1900, le=2100)
    identifier: str | None = Field(default=None, max_length=MAX_LABEL_LEN)
    algorithm: str | None = Field(default=None, max_length=MAX_LABEL_LEN)
    url: str | None = Field(default=None, max_length=MAX_LABEL_LEN)

    @model_validator(mode="after")
    def _external_requires_identifier(self) -> "CitationMetadata":
        if self.kind == "external" and not (self.identifier or "").strip():
            raise ValueError("external citations require a non-empty identifier")
        return self


class TuningProvenance(_StrictStudyModel):
    """Matched-budget provenance for one panel policy.

    A tuned policy must declare a strictly positive candidate budget and the
    selection seed/provenance it was chosen under. A parameter-free policy must
    declare ``mode="exempt"`` with a zero budget and no selection seed, so the
    study never pretends a fixed baseline was tuned.
    """

    mode: Literal["tuned", "exempt"]
    candidate_budget: int = Field(ge=0, le=MAX_CANDIDATE_BUDGET)
    selection_seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    selection_provenance: str | None = Field(
        default=None, max_length=MAX_LABEL_LEN
    )

    @model_validator(mode="after")
    def _consistent_mode(self) -> "TuningProvenance":
        if self.mode == "exempt":
            if self.candidate_budget != 0:
                raise ValueError(
                    "exempt tuning provenance must declare candidate_budget 0"
                )
            if self.selection_seed is not None:
                raise ValueError(
                    "exempt tuning provenance must not declare a selection_seed"
                )
        else:
            if self.candidate_budget <= 0:
                raise ValueError(
                    "tuned tuning provenance requires a positive candidate_budget"
                )
            if self.selection_seed is None:
                raise ValueError(
                    "tuned tuning provenance requires a selection_seed"
                )
            if not (self.selection_provenance or "").strip():
                raise ValueError(
                    "tuned tuning provenance requires a selection_provenance"
                )
        return self


class PanelPolicy(_StrictStudyModel):
    """One baseline identity in the study panel.

    ``id`` is a stable identifier from :data:`REQUIRED_POLICY_IDS` and must
    agree with the ``aggregation`` selector's ``kind``. ``implementation`` is a
    bounded human-readable label, ``citation`` carries the reproduction/version
    metadata, and ``tuning`` carries the matched-budget provenance.
    """

    id: str = Field(min_length=1, max_length=MAX_STUDY_ID_LEN)
    implementation: str = Field(min_length=1, max_length=MAX_LABEL_LEN)
    aggregation: AggregationConfig
    citation: CitationMetadata
    tuning: TuningProvenance

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        if not all(
            character.islower() or character.isdigit() or character == "_"
            for character in value
        ):
            raise ValueError(
                "policy id must contain only lowercase letters, digits, "
                "and underscores"
            )
        if value not in _POLICY_KINDS:
            raise ValueError(f"unknown policy id: {value!r}")
        return value

    @model_validator(mode="after")
    def _identity_matches_aggregation(self) -> "PanelPolicy":
        expected = _POLICY_KINDS[self.id]
        if self.aggregation.kind != expected:
            raise ValueError(
                f"policy {self.id!r} aggregation.kind must be {expected!r}, "
                f"got {self.aggregation.kind!r}"
            )
        if self.id == "balance":
            actual = self.citation.model_dump(mode="json")
            if actual != BALANCE_CITATION:
                raise ValueError(
                    "balance citation must be the pinned D028 reference "
                    f"{BALANCE_CITATION!r}, got {actual!r}"
                )
        return self


class BaselineStudyConfig(_StrictStudyModel):
    """Top-level strict baseline-study envelope (R017/S01).

    ``base`` is a complete validated :class:`ExperimentConfig` whose
    ``aggregation`` must be plain averaging; it is the frozen source every cell
    is derived from by overriding only ``seed`` and ``aggregation``.

    ``data_provenance`` is explicit: ``injected`` requires ``base.data.download``
    to be false (an in-memory MNIST-shaped dataset is supplied directly), while
    ``real_mnist`` documents that the production dataset loader provides the
    data. ``seeds`` is a non-empty bounded unique tuple.

    ``candidate_budget`` / ``selection_seed`` / ``selection_provenance`` are the
    single matched tuning budget every tunable policy must declare; each
    parameter-free policy must instead declare ``exempt`` with a zero budget.
    """

    study_id: str = Field(min_length=1, max_length=MAX_STUDY_ID_LEN)
    data_provenance: Literal["injected", "real_mnist"]
    base: ExperimentConfig
    seeds: SeedTuple
    candidate_budget: int = Field(gt=0, le=MAX_CANDIDATE_BUDGET)
    selection_seed: int = Field(ge=0, le=MAX_SEED)
    selection_provenance: str = Field(min_length=1, max_length=MAX_LABEL_LEN)
    panel: Annotated[tuple[PanelPolicy, ...], BeforeValidator(_as_tuple)]

    @field_validator("study_id")
    @classmethod
    def _safe_study_id(cls, value: str) -> str:
        if not all(
            character.islower() or character.isdigit() or character in "_-"
            for character in value
        ):
            raise ValueError(
                "study_id must contain only lowercase letters, digits, "
                "hyphens, and underscores"
            )
        return value

    @model_validator(mode="after")
    def _validate_study(self) -> "BaselineStudyConfig":
        if not isinstance(self.base.aggregation, PlainAverageAggregation):
            raise ValueError(
                "base.aggregation must be plain_average; every panel entry "
                "overrides it per cell"
            )
        if self.data_provenance == "injected" and self.base.data.download:
            raise ValueError(
                "data_provenance 'injected' requires base.data.download false"
            )

        seeds = tuple(self.seeds)
        if not seeds:
            raise ValueError("seeds must declare at least one seed")
        if len(seeds) > MAX_STUDY_SEEDS:
            raise ValueError(
                f"seeds declare {len(seeds)} entries, limit is {MAX_STUDY_SEEDS}"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f"seeds must be integers, got {seed!r}")
            if not 0 <= seed <= MAX_SEED:
                raise ValueError(
                    f"seed {seed!r} is outside [0, {MAX_SEED}]"
                )

        ids = tuple(entry.id for entry in self.panel)
        if len(ids) != len(REQUIRED_POLICY_IDS):
            raise ValueError(
                "panel must declare exactly the "
                f"{len(REQUIRED_POLICY_IDS)} required policies "
                f"{REQUIRED_POLICY_IDS!r}, got {ids!r}"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(f"panel policy ids must be unique, got {ids!r}")
        if set(ids) != set(REQUIRED_POLICY_IDS):
            missing = sorted(set(REQUIRED_POLICY_IDS) - set(ids))
            extra = sorted(set(ids) - set(REQUIRED_POLICY_IDS))
            raise ValueError(
                f"panel policy set mismatch (missing={missing}, unexpected={extra})"
            )

        for entry in self.panel:
            tuning = entry.tuning
            if entry.id in _TUNABLE_POLICY_IDS:
                if tuning.mode != "tuned":
                    raise ValueError(
                        f"tunable policy {entry.id!r} must declare tuned "
                        "tuning provenance"
                    )
                if tuning.candidate_budget != self.candidate_budget:
                    raise ValueError(
                        f"policy {entry.id!r} candidate_budget "
                        f"{tuning.candidate_budget!r} does not match the study "
                        f"matched budget {self.candidate_budget!r}"
                    )
                if tuning.selection_seed != self.selection_seed:
                    raise ValueError(
                        f"policy {entry.id!r} selection_seed "
                        f"{tuning.selection_seed!r} does not match the study "
                        f"selection seed {self.selection_seed!r}"
                    )
                if tuning.selection_provenance != self.selection_provenance:
                    raise ValueError(
                        f"policy {entry.id!r} selection_provenance does not "
                        "match the study selection provenance"
                    )
            elif entry.id in _PARAMETER_FREE_POLICY_IDS:
                if tuning.mode != "exempt":
                    raise ValueError(
                        f"parameter-free policy {entry.id!r} must declare "
                        "exempt tuning provenance"
                    )
        return self

    # -- deterministic ordering helper ------------------------------------- #
    def ordered_panel(self) -> tuple[PanelPolicy, ...]:
        """The panel in the canonical :data:`REQUIRED_POLICY_IDS` order."""

        by_id = {entry.id: entry for entry in self.panel}
        return tuple(by_id[policy_id] for policy_id in REQUIRED_POLICY_IDS)


# --------------------------------------------------------------------------- #
# Deterministic hyperparameter identity (environment-independent)
# --------------------------------------------------------------------------- #
def selected_hyperparameters(aggregation: AggregationConfig) -> dict:
    """The numeric hyperparameters of ``aggregation`` (no kind/path fields)."""

    payload = aggregation.model_dump(mode="json")
    payload.pop("kind", None)
    payload.pop("artifact_dir", None)
    return payload


def policy_identity(aggregation: AggregationConfig) -> dict:
    """A deterministic identity payload for ``aggregation``.

    The environment-specific ``artifact_dir`` path is replaced by a fixed
    placeholder; the artifact's own digest is recorded separately, so identity
    and manifest bytes stay independent of the checkout location.
    """

    payload = aggregation.model_dump(mode="json")
    if "artifact_dir" in payload:
        payload["artifact_dir"] = "<artifact>"
    return payload


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def hyperparameter_digest(aggregation: AggregationConfig) -> str:
    """Stable SHA-256 over the selected (environment-independent) hyperparameters."""

    return hashlib.sha256(
        _canonical_bytes(selected_hyperparameters(aggregation))
    ).hexdigest()


def study_identity_payload(study: BaselineStudyConfig) -> dict:
    """A deterministic, environment-independent payload for ``study``.

    Every ``artifact_dir`` (base and panel) is replaced by the fixed
    placeholder so two studies with the same logical spec on different
    machines hash identically. The artifact's own aggregate digest is recorded
    per manifest entry instead.
    """

    payload = study.model_dump(mode="json")
    return _scrub_artifact_paths(payload)


def _scrub_artifact_paths(payload: object) -> object:
    if isinstance(payload, dict):
        return {
            key: ("<artifact>" if key == "artifact_dir" else _scrub_artifact_paths(value))
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_scrub_artifact_paths(item) for item in payload]
    return payload


def study_digest(study: BaselineStudyConfig) -> str:
    """SHA-256 of the environment-independent canonical study payload."""

    return hashlib.sha256(
        _canonical_bytes(study_identity_payload(study))
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Strict loader (JSON first, safe YAML fallback)
# --------------------------------------------------------------------------- #
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
        raise BaselineStudyConfigError(
            f"invalid JSON in {source}: {_bounded(str(exc))}"
        ) from None
    except json.JSONDecodeError:
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise BaselineStudyConfigError(
                f"invalid YAML in {source}: {_bounded(str(exc))}"
            ) from exc
    return raw


def parse_study_text(
    text: str, *, source: str = "<string>"
) -> BaselineStudyConfig:
    """Parse and validate canonical JSON (preferred) or safe YAML.

    Raises :class:`BaselineStudyConfigError` with the source path and a bounded
    offending-field summary on any malformed input, non-mapping root,
    non-finite value, unknown key, unsafe tag, or invalid field.
    """

    raw = _parse_structured(text, source=source)
    if raw is None:
        raise BaselineStudyConfigError(
            f"empty study config in {source}: expected a mapping"
        )
    if not isinstance(raw, dict):
        raise BaselineStudyConfigError(
            f"study config root in {source} must be a mapping, got "
            f"{type(raw).__name__}"
        )
    try:
        return BaselineStudyConfig.model_validate(raw)
    except ValidationError as exc:
        raise BaselineStudyConfigError(
            f"invalid study config in {source}: {_bounded(str(exc))}"
        ) from exc


def load_study_config(path: str | os.PathLike[str]) -> BaselineStudyConfig:
    """Read ``path`` from disk and validate it as a :class:`BaselineStudyConfig`."""

    from pathlib import Path

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BaselineStudyConfigError(
            f"study config file not found: {config_path}"
        ) from exc
    except OSError as exc:
        raise BaselineStudyConfigError(
            f"could not read study config file {config_path}: "
            f"{_bounded(str(exc))}"
        ) from exc
    return parse_study_text(text, source=str(config_path))
