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
(:data:`REQUIRED_POLICY_IDS`); when the optional :class:`AblationStudyContract`
is present it must instead contain exactly the eight :data:`EXTENDED_POLICY_IDS`
(the same six plus ``bayesian_no_belief`` and ``bayesian_no_strategic``).
Tunable policies must declare the study's single positive candidate budget and
selection-seed provenance; parameter-free policies must explicitly declare
zero/exempt rather than pretending to tune. The BALANCE entry's citation is
pinned to the D028 external reference.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from .ablation import (
    CALIBRATION_VARIANTS,
    FULL_VARIANT,
    NO_BELIEF_VARIANT,
    NO_STRATEGIC_VARIANT,
)
from .attack_schedule import (
    ScheduleConfig,
    intensity_name,
    realized_kappa_sequence,
    schedule_digest,
)
from .config import (
    MAX_SEED,
    AggregationConfig,
    ConfigError,
    ExperimentConfig,
    PlainAverageAggregation,
)

__all__ = [
    "ABLATION_POLICY_IDS",
    "ABLATION_VARIANT_BY_POLICY_ID",
    "ARTIFACT_POLICY_IDS",
    "BALANCE_CITATION",
    "EXTENDED_POLICY_IDS",
    "MAX_CANDIDATE_BUDGET",
    "MAX_KNOWN_SCHEDULES",
    "MAX_STUDY_ID_LEN",
    "MAX_STUDY_SEEDS",
    "MIN_TOTAL_SEEDS_INJECTED",
    "MIN_TOTAL_SEEDS_REAL_MNIST",
    "REQUIRED_POLICY_IDS",
    "AblationArtifactPin",
    "AblationStudyContract",
    "BaselineStudyConfig",
    "BaselineStudyConfigError",
    "CitationMetadata",
    "HeldOutStudyConfig",
    "HeldOutStudyConfigError",
    "NamedSchedule",
    "PanelPolicy",
    "PlacementContract",
    "SelectionSpec",
    "TuningProvenance",
    "held_out_study_digest",
    "held_out_study_identity_payload",
    "hyperparameter_digest",
    "load_any_study_config",
    "load_held_out_study_config",
    "load_study_config",
    "parse_held_out_study_text",
    "parse_study_text",
    "policy_identity",
    "selected_hyperparameters",
    "selection_identity_payload",
    "selection_projection",
    "selection_spec_digest",
    "study_digest",
    "study_envelope_kind",
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

#: Maximum number of named adaptive schedules a held-out study may carry.
MAX_KNOWN_SCHEDULES = 16

#: R022 seed floors: paired selection + final seeds must total at least these
#: counts (five for injected provenance, three for real MNIST data).
MIN_TOTAL_SEEDS_INJECTED = 5
MIN_TOTAL_SEEDS_REAL_MNIST = 3

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
    {
        "fixed_similarity",
        "trust_weighted",
        "balance",
        "bayesian",
        "bayesian_no_belief",
        "bayesian_no_strategic",
    }
)
_PARAMETER_FREE_POLICY_IDS: frozenset[str] = frozenset(
    {"local_only", "plain_average"}
)

#: Opt-in ablation panel identities (R021/D036/D037). They may only appear when
#: a study declares the :class:`AblationStudyContract`; both map onto the
#: existing ``bayesian`` aggregation kind (the artifact is the experimental
#: variable, not a new runtime policy class).
ABLATION_POLICY_IDS: tuple[str, ...] = (
    "bayesian_no_belief",
    "bayesian_no_strategic",
)

#: The canonical eight-policy panel an opt-in ablation study must declare.
EXTENDED_POLICY_IDS: tuple[str, ...] = REQUIRED_POLICY_IDS + ABLATION_POLICY_IDS

#: The three artifact-backed panel identities, in canonical order. ``bayesian``
#: is the full artifact; the other two are the ablations.
ARTIFACT_POLICY_IDS: tuple[str, ...] = (
    "bayesian",
    "bayesian_no_belief",
    "bayesian_no_strategic",
)

#: Canonical variant for each panel identity that consumes an artifact.
ABLATION_VARIANT_BY_POLICY_ID: dict[str, str] = {
    "bayesian": FULL_VARIANT,
    "bayesian_no_belief": NO_BELIEF_VARIANT,
    "bayesian_no_strategic": NO_STRATEGIC_VARIANT,
}

#: Expected aggregation config ``kind`` for each stable policy id. Identity and
#: aggregation kind are cross-checked so a panel cannot mislabel a policy.
_POLICY_KINDS: dict[str, str] = {
    "local_only": "local_only",
    "plain_average": "plain_average",
    "fixed_similarity": "fixed_similarity",
    "trust_weighted": "trust_weighted",
    "balance": "balance",
    "bayesian": "bayesian",
    "bayesian_no_belief": "bayesian",
    "bayesian_no_strategic": "bayesian",
}

#: Lowercase SHA-256 hex, the only accepted digest format for pinned ablated
#: artifact and matched calibration-input identities.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

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


class HeldOutStudyConfigError(BaselineStudyConfigError):
    """Raised when a held-out study YAML/JSON spec is missing or invalid."""


def _as_tuple(value: object) -> object:
    """Convert JSON/YAML lists to tuples before strict tuple validation."""

    if isinstance(value, list):
        return tuple(value)
    return value


SeedTuple = Annotated[tuple[int, ...], BeforeValidator(_as_tuple)]


def _validate_seed_tuple(seeds, *, label: str = "seeds") -> tuple[int, ...]:
    """Validate a bounded, unique, in-range seed tuple.

    Shared by the S01 baseline study (``seeds``) and the S02 held-out study
    (``final_seeds``) so both top-level validators enforce the identical seed
    contract without duplicating the bounds.
    """

    values = tuple(seeds)
    if not values:
        raise ValueError(f"{label} must declare at least one seed")
    if len(values) > MAX_STUDY_SEEDS:
        raise ValueError(
            f"{label} declare {len(values)} entries, limit is {MAX_STUDY_SEEDS}"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
    for seed in values:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{label} must be integers, got {seed!r}")
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed {seed!r} is outside [0, {MAX_SEED}]")
    return values


def _validate_panel_budget(study) -> None:
    """Validate the six/eight-policy panel and its matched-budget provenance.

    Without an :class:`AblationStudyContract` exactly the six canonical
    :data:`REQUIRED_POLICY_IDS` are required and the S01 behavior is
    byte-for-byte unchanged. With the opt-in contract the panel must declare
    exactly the eight :data:`EXTENDED_POLICY_IDS`; the three artifact-backed
    Bayesian entries must point at distinct artifact directories, so a reused
    or swapped artifact path cannot masquerade as a matched triple.
    """

    ablations = getattr(study, "ablations", None)
    expected_ids = (
        EXTENDED_POLICY_IDS if ablations is not None else REQUIRED_POLICY_IDS
    )
    ids = tuple(entry.id for entry in study.panel)
    if len(ids) != len(expected_ids):
        raise ValueError(
            "panel must declare exactly the "
            f"{len(expected_ids)} required policies "
            f"{expected_ids!r}, got {ids!r}"
        )
    if len(set(ids)) != len(ids):
        raise ValueError(f"panel policy ids must be unique, got {ids!r}")
    if set(ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(ids))
        extra = sorted(set(ids) - set(expected_ids))
        raise ValueError(
            f"panel policy set mismatch (missing={missing}, unexpected={extra})"
        )

    for entry in study.panel:
        tuning = entry.tuning
        if entry.id in _TUNABLE_POLICY_IDS:
            if tuning.mode != "tuned":
                raise ValueError(
                    f"tunable policy {entry.id!r} must declare tuned "
                    "tuning provenance"
                )
            if tuning.candidate_budget != study.candidate_budget:
                raise ValueError(
                    f"policy {entry.id!r} candidate_budget "
                    f"{tuning.candidate_budget!r} does not match the study "
                    f"matched budget {study.candidate_budget!r}"
                )
            if tuning.selection_seed != study.selection_seed:
                raise ValueError(
                    f"policy {entry.id!r} selection_seed "
                    f"{tuning.selection_seed!r} does not match the study "
                    f"selection seed {study.selection_seed!r}"
                )
            if tuning.selection_provenance != study.selection_provenance:
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

    if ablations is not None:
        artifact_dirs = tuple(
            entry.aggregation.artifact_dir
            for entry in study.panel
            if entry.id in ABLATION_VARIANT_BY_POLICY_ID
        )
        if len(set(artifact_dirs)) != len(artifact_dirs):
            raise ValueError(
                "ablation panel must reference distinct artifact directories "
                "for the full/no_belief/no_strategic artifacts"
            )


def _validate_study_content(study) -> None:
    """Validate base provenance, seeds, panel, and budget for a study.

    Shared by :class:`BaselineStudyConfig` and :class:`HeldOutStudyConfig` so
    both top-level validators enforce the same study contract.
    """

    if not isinstance(study.base.aggregation, PlainAverageAggregation):
        raise ValueError(
            "base.aggregation must be plain_average; every panel entry "
            "overrides it per cell"
        )
    if study.data_provenance == "injected" and study.base.data.download:
        raise ValueError(
            "data_provenance 'injected' requires base.data.download false"
        )
    _validate_seed_tuple(study.seeds, label="seeds")
    _validate_panel_budget(study)


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

    ``id`` is a stable identifier from :data:`REQUIRED_POLICY_IDS` (or, when
    an :class:`AblationStudyContract` is present, from :data:`EXTENDED_POLICY_IDS`)
    and must agree with the ``aggregation`` selector's ``kind``. ``implementation``
    is a bounded human-readable label, ``citation`` carries the reproduction/version
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


class AblationArtifactPin(_StrictStudyModel):
    """One pinned expected aggregate digest for a calibration variant (R021).

    The digest is the artifact's own schema-v1 ``aggregate_digest``; it binds
    the exact published tables and assumptions the study was frozen against.
    The artifact path is deliberately *not* duplicated here: the matching panel
    entry already carries ``artifact_dir``.
    """

    variant: Literal["full", "no_belief", "no_strategic"]
    expected_aggregate_digest: str

    @field_validator("expected_aggregate_digest")
    @classmethod
    def _valid_aggregate_digest(cls, value: str) -> str:
        if _SHA256_RE.match(value) is None:
            raise ValueError(
                "expected_aggregate_digest must be a lowercase SHA-256 hex "
                "string"
            )
        return value


class AblationStudyContract(_StrictStudyModel):
    """Opt-in matched-ablation contract for the eight-policy panel (R021).

    Requires exactly one pinned aggregate digest for each of the ``full``,
    ``no_belief`` and ``no_strategic`` artifacts plus one pinned
    ``matched_calibration_input_digest`` (the shared, variant-normalized
    calibration input identity). It carries no artifact paths, so a study
    cannot declare a second, conflicting pointer to the same artifact.
    """

    matched_calibration_input_digest: str
    expected_aggregate_digests: Annotated[
        tuple[AblationArtifactPin, ...], BeforeValidator(_as_tuple)
    ]

    @field_validator("matched_calibration_input_digest")
    @classmethod
    def _valid_matched_digest(cls, value: str) -> str:
        if _SHA256_RE.match(value) is None:
            raise ValueError(
                "matched_calibration_input_digest must be a lowercase "
                "SHA-256 hex string"
            )
        return value

    @model_validator(mode="after")
    def _complete_distinct_variants(self) -> "AblationStudyContract":
        variants = tuple(pin.variant for pin in self.expected_aggregate_digests)
        if len(variants) != len(set(variants)):
            raise ValueError(
                "expected_aggregate_digests must pin each variant at most once"
            )
        if set(variants) != set(CALIBRATION_VARIANTS):
            raise ValueError(
                "expected_aggregate_digests must pin exactly one digest for "
                "each of full, no_belief and no_strategic"
            )
        return self

    def expected_digest(self, variant: str) -> str:
        """Return the pinned aggregate digest for ``variant``."""

        for pin in self.expected_aggregate_digests:
            if pin.variant == variant:
                return pin.expected_aggregate_digest
        raise KeyError(variant)


class PlacementContract(_StrictStudyModel):
    """Optional seed-derived degree-placement contract for a study (R020/D040).

    When present, the compromised node is **not** a prechosen scalar: the
    per-cell node is resolved by the named degree rank (``low``/``mid``/
    ``high``) on the realized, seed-derived topology. ``graph_kind`` pins the
    topology family the contract is valid for; the study base must declare that
    same kind. ``require_varied_degrees`` (default true) makes a regular
    realized graph fail closed, because its three canonical ranks could never
    represent three distinct degree groups.
    """

    placement: Literal["low", "mid", "high"]
    graph_kind: Literal["watts_strogatz"] = "watts_strogatz"
    require_varied_degrees: bool = True


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

    ``ablations`` is the optional opt-in matched-ablation contract (R021). When
    absent the panel must be exactly the six S01 identities and the canonical
    payload is byte-for-byte what it was before this field existed (the
    serializer omits an absent contract rather than emitting a null key). When
    present the panel must be exactly the eight identities and the three
    artifact-backed entries must reference distinct, digest-pinned artifacts.

    ``placement`` is the optional seed-derived degree-placement contract (R020).
    When present the base topology must match its ``graph_kind`` and the base
    attack's scalar ``compromised_node`` is a schema placeholder that the
    contract deterministically overrides per cell; the serializer omits an
    absent contract so legacy canonical bytes are stable.
    """

    study_id: str = Field(min_length=1, max_length=MAX_STUDY_ID_LEN)
    data_provenance: Literal["injected", "real_mnist"]
    base: ExperimentConfig
    seeds: SeedTuple
    candidate_budget: int = Field(gt=0, le=MAX_CANDIDATE_BUDGET)
    selection_seed: int = Field(ge=0, le=MAX_SEED)
    selection_provenance: str = Field(min_length=1, max_length=MAX_LABEL_LEN)
    panel: Annotated[tuple[PanelPolicy, ...], BeforeValidator(_as_tuple)]
    ablations: AblationStudyContract | None = None
    placement: PlacementContract | None = None

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

    @model_serializer(mode="wrap")
    def _omit_absent_opt_ins(self, handler, info):
        """Serialize without absent opt-in keys (legacy canonical form)."""

        data = handler(self)
        if self.ablations is None:
            data.pop("ablations", None)
        if self.placement is None:
            data.pop("placement", None)
        return data

    @model_validator(mode="after")
    def _validate_study(self) -> "BaselineStudyConfig":
        _validate_study_content(self)
        if self.placement is not None:
            if self.base.topology.kind != self.placement.graph_kind:
                raise ValueError(
                    "placement contract graph_kind "
                    f"{self.placement.graph_kind!r} does not match the base "
                    f"topology kind {self.base.topology.kind!r}"
                )
        return self

    # -- deterministic ordering helpers ------------------------------------ #
    def policy_ids(self) -> tuple[str, ...]:
        """The ordered panel identity set for this study (six or eight ids)."""

        return (
            EXTENDED_POLICY_IDS
            if self.ablations is not None
            else REQUIRED_POLICY_IDS
        )

    def ordered_panel(self) -> tuple[PanelPolicy, ...]:
        """The panel in canonical (six- or eight-identity) order."""

        by_id = {entry.id: entry for entry in self.panel}
        return tuple(by_id[policy_id] for policy_id in self.policy_ids())


class NamedSchedule(_StrictStudyModel):
    """A named adaptive attack schedule (R018/D033).

    ``id`` is a bounded study-local label and ``schedule`` is a validated
    :data:`~dfl_sim.attack_schedule.ScheduleConfig`. The name is what the
    selection/final manifests record as ``schedule_id``.
    """

    id: str = Field(min_length=1, max_length=MAX_STUDY_ID_LEN)
    schedule: ScheduleConfig

    @field_validator("id")
    @classmethod
    def _safe_schedule_id(cls, value: str) -> str:
        if not all(
            character.islower() or character.isdigit() or character in "_-"
            for character in value
        ):
            raise ValueError(
                "schedule id must contain only lowercase letters, digits, "
                "hyphens, and underscores"
            )
        return value


class SelectionSpec(_StrictStudyModel):
    """The selection-side projection of a held-out study (R018/R022).

    Carries only the selection-phase content: the S01 baseline study (whose
    ``seeds`` are the selection seeds) and the known adaptive schedules. It
    deliberately has **no** ``final_seeds`` or ``held_out_schedule`` attribute,
    so a selection runner typed to accept only a :class:`SelectionSpec` cannot
    structurally read the final seeds or the reserved held-out schedule. That
    is the milestone no-leak rule enforced at the type boundary, not by
    convention.
    """

    study: BaselineStudyConfig
    known_schedules: Annotated[
        tuple[NamedSchedule, ...], BeforeValidator(_as_tuple)
    ]

    @model_validator(mode="after")
    def _validate_selection(self) -> "SelectionSpec":
        _validate_known_schedules(self.known_schedules)
        return self


class HeldOutStudyConfig(_StrictStudyModel):
    """Top-level strict held-out study envelope (R018/R022, S02/T02).

    Composes the S01 :class:`BaselineStudyConfig` as the field ``study``; its
    ``seeds`` are the *selection-phase* seeds. Because
    :func:`study_digest` operates on that nested study alone, the selection-side
    digest covers only selection content and excludes ``final_seeds`` by
    construction.

    ``final_seeds`` are disjoint from ``study.seeds``. ``known_schedules`` are
    the adaptive schedules the selection phase may build; ``held_out_schedule``
    is the reserved unseen schedule that only the final phase may build. The
    attack provenance must declare the compromised node and the peak realized
    intensity, and the paired seed totals must satisfy the R022 floors.
    """

    study: BaselineStudyConfig
    final_seeds: SeedTuple
    known_schedules: Annotated[
        tuple[NamedSchedule, ...], BeforeValidator(_as_tuple)
    ]
    held_out_schedule: NamedSchedule

    @model_validator(mode="after")
    def _validate_held_out(self) -> "HeldOutStudyConfig":
        # Reuse the S01 study contract helper so both top-level validators
        # enforce identical base/seed/panel/budget rules.
        _validate_study_content(self.study)
        _validate_known_schedules(self.known_schedules)

        final_seeds = _validate_seed_tuple(self.final_seeds, label="final_seeds")
        selection_seeds = tuple(self.study.seeds)

        overlap = sorted(set(final_seeds) & set(selection_seeds))
        if overlap:
            raise ValueError(
                "final_seeds must be disjoint from the selection seeds, "
                f"overlap={overlap!r}"
            )

        held_out_id = self.held_out_schedule.id
        known_ids = tuple(entry.id for entry in self.known_schedules)
        if held_out_id in known_ids:
            raise ValueError(
                f"held_out schedule id {held_out_id!r} collides with a known "
                f"schedule id {known_ids!r}"
            )
        known_digests = {
            schedule_digest(entry.schedule) for entry in self.known_schedules
        }
        if schedule_digest(self.held_out_schedule.schedule) in known_digests:
            raise ValueError(
                "held_out schedule must be structurally distinct from every "
                "known schedule"
            )

        attack = self.study.base.attack
        if attack.compromised_node is None and self.study.placement is None:
            raise ValueError(
                "study.base.attack.compromised_node must be set for an "
                "adaptive held-out study (or a placement contract must "
                "derive it)"
            )

        rounds = self.study.base.rounds
        # Coverage is a plan-level property, so it is checked under the
        # canonical seed 0: periodic/duty/delayed schedules are seed-invariant,
        # and a random schedule must still cover all three under this pinned
        # draw. This keeps the coverage verdict deterministic and independent
        # of any particular selection seed.
        realized: set[str] = set()
        peak_kappa = 0.0
        for entry in self.known_schedules:
            for kappa in realized_kappa_sequence(entry.schedule, rounds, 0):
                realized.add(intensity_name(kappa))
                peak_kappa = max(peak_kappa, kappa)
        missing = sorted({"zero", "weak", "strong"} - realized)
        if missing:
            raise ValueError(
                "known schedules must collectively realize zero, weak, and "
                f"strong over {rounds} rounds; missing={missing!r}"
            )
        peak_name = intensity_name(peak_kappa)
        if attack.intensity != peak_name:
            raise ValueError(
                f"study.base.attack.intensity {attack.intensity!r} does not "
                f"match the peak realized intensity {peak_name!r}"
            )

        total_seeds = len(selection_seeds) + len(final_seeds)
        floor = (
            MIN_TOTAL_SEEDS_INJECTED
            if self.study.data_provenance == "injected"
            else MIN_TOTAL_SEEDS_REAL_MNIST
        )
        if total_seeds < floor:
            raise ValueError(
                f"{self.study.data_provenance} study declares {total_seeds} "
                f"total selection+final seeds, need at least {floor}"
            )
        return self

    def ordered_known_schedules(self) -> tuple[NamedSchedule, ...]:
        """The known schedules sorted by id (deterministic iteration order)."""

        return tuple(sorted(self.known_schedules, key=lambda entry: entry.id))


def _validate_known_schedules(schedules) -> None:
    """Validate a bounded, id-unique set of known adaptive schedules."""

    values = tuple(schedules)
    if not values:
        raise ValueError("known_schedules must declare at least one schedule")
    if len(values) > MAX_KNOWN_SCHEDULES:
        raise ValueError(
            f"known_schedules declares {len(values)} entries, limit is "
            f"{MAX_KNOWN_SCHEDULES}"
        )
    ids = tuple(entry.id for entry in values)
    if len(set(ids)) != len(ids):
        raise ValueError(f"known schedule ids must be unique, got {ids!r}")
    for entry in values:
        if entry.schedule.kind == "held_out":
            raise ValueError(
                f"known schedule {entry.id!r} must not use the reserved "
                "held_out kind"
            )


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


def selection_projection(study: HeldOutStudyConfig) -> SelectionSpec:
    """Project a held-out study onto its selection-side content.

    The result carries the nested selection study and the known schedules and
    nothing else, so final seeds and the held-out schedule are structurally
    unavailable during selection. This is the type-level enforcement of the
    milestone no-leak rule (R018/R022).
    """

    if not isinstance(study, HeldOutStudyConfig):
        raise HeldOutStudyConfigError(
            "selection_projection requires a HeldOutStudyConfig, got "
            f"{type(study).__name__}"
        )
    return SelectionSpec(
        study=study.study, known_schedules=study.known_schedules
    )


def selection_identity_payload(selection: SelectionSpec) -> dict:
    """Environment-independent canonical payload for a selection projection."""

    if not isinstance(selection, SelectionSpec):
        raise HeldOutStudyConfigError(
            "selection_identity_payload requires a SelectionSpec, got "
            f"{type(selection).__name__}"
        )
    return _scrub_artifact_paths(selection.model_dump(mode="json"))


def selection_spec_digest(selection: SelectionSpec) -> str:
    """SHA-256 over the selection-side projection (excludes final seeds)."""

    return hashlib.sha256(
        _canonical_bytes(selection_identity_payload(selection))
    ).hexdigest()


def held_out_study_identity_payload(study: HeldOutStudyConfig) -> dict:
    """Environment-independent canonical payload for a held-out study."""

    if not isinstance(study, HeldOutStudyConfig):
        raise HeldOutStudyConfigError(
            "held_out_study_identity_payload requires a HeldOutStudyConfig, "
            f"got {type(study).__name__}"
        )
    return _scrub_artifact_paths(study.model_dump(mode="json"))


def held_out_study_digest(study: HeldOutStudyConfig) -> str:
    """SHA-256 over the full held-out study payload (selection + final)."""

    return hashlib.sha256(
        _canonical_bytes(held_out_study_identity_payload(study))
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


def _parse_mapping(text: str, *, source: str) -> dict:
    """Parse structured text once and require a mapping root."""

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
    return raw


def parse_study_text(
    text: str, *, source: str = "<string>"
) -> BaselineStudyConfig:
    """Parse and validate canonical JSON (preferred) or safe YAML.

    Raises :class:`BaselineStudyConfigError` with the source path and a bounded
    offending-field summary on any malformed input, non-mapping root,
    non-finite value, unknown key, unsafe tag, or invalid field.
    """

    raw = _parse_mapping(text, source=source)
    try:
        return BaselineStudyConfig.model_validate(raw)
    except ValidationError as exc:
        raise BaselineStudyConfigError(
            f"invalid study config in {source}: {_bounded(str(exc))}"
        ) from exc


def parse_held_out_study_text(
    text: str, *, source: str = "<string>"
) -> HeldOutStudyConfig:
    """Parse and validate a held-out study as canonical JSON or safe YAML.

    Raises :class:`HeldOutStudyConfigError` (a bounded subclass of
    :class:`BaselineStudyConfigError`) on malformed/unsafe input or any invalid
    field.
    """

    try:
        raw = _parse_mapping(text, source=source)
    except BaselineStudyConfigError as exc:
        raise HeldOutStudyConfigError(str(exc)) from None
    try:
        return HeldOutStudyConfig.model_validate(raw)
    except ValidationError as exc:
        raise HeldOutStudyConfigError(
            f"invalid held-out study config in {source}: {_bounded(str(exc))}"
        ) from exc


def _read_config_text(path: str | os.PathLike[str]) -> str:
    """Read a config file, raising a bounded baseline error on any failure."""

    from pathlib import Path

    config_path = Path(path)
    try:
        return config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BaselineStudyConfigError(
            f"study config file not found: {config_path}"
        ) from exc
    except OSError as exc:
        raise BaselineStudyConfigError(
            f"could not read study config file {config_path}: "
            f"{_bounded(str(exc))}"
        ) from exc


def load_study_config(path: str | os.PathLike[str]) -> BaselineStudyConfig:
    """Read ``path`` from disk and validate it as a :class:`BaselineStudyConfig`."""

    return parse_study_text(_read_config_text(path), source=str(path))


def load_held_out_study_config(
    path: str | os.PathLike[str],
) -> HeldOutStudyConfig:
    """Read ``path`` and validate it as a :class:`HeldOutStudyConfig`."""

    try:
        text = _read_config_text(path)
    except BaselineStudyConfigError as exc:
        raise HeldOutStudyConfigError(str(exc)) from None
    return parse_held_out_study_text(text, source=str(path))


#: The strict canonical robustness source-envelope version literal. The
#: robustness matrix protocol declares this at its root, so the public CLI can
#: route it to the canonical matrix command without validating it first.
ROBUSTNESS_PROTOCOL_SOURCE_VERSION = "dfl_sim.robustness_protocol/v1"


def study_envelope_kind(path: str | os.PathLike[str]) -> str:
    """Classify a study-source file as ``"robustness"`` or ``"study"``.

    This is a cheap, non-validating structural sniff: the strict canonical
    robustness envelope declares the versioned ``protocol_version`` key, while
    a baseline/held-out study spec does not. It reads and parses the file once
    and raises :class:`BaselineStudyConfigError` on an unreadable or malformed
    source, so the public CLI can report one bounded error before any work.
    """

    text = _read_config_text(path)
    raw = _parse_mapping(text, source=str(path))
    if raw.get("protocol_version") == ROBUSTNESS_PROTOCOL_SOURCE_VERSION:
        return "robustness"
    return "study"


def load_any_study_config(
    path: str | os.PathLike[str],
) -> BaselineStudyConfig | HeldOutStudyConfig:
    """Read ``path`` once and dispatch to the baseline or held-out contract.

    Structured text is parsed exactly once; the mapping is validated as a
    :class:`BaselineStudyConfig` first, and only if that fails as a
    :class:`HeldOutStudyConfig`. This keeps the S01 `dfl evaluate` baseline
    path byte-identical while letting the CLI accept both spec kinds from one
    loader.
    """

    text = _read_config_text(path)
    try:
        raw = _parse_mapping(text, source=str(path))
    except BaselineStudyConfigError as exc:
        raise BaselineStudyConfigError(str(exc)) from None
    try:
        return BaselineStudyConfig.model_validate(raw)
    except ValidationError as baseline_exc:
        try:
            return HeldOutStudyConfig.model_validate(raw)
        except ValidationError as held_exc:
            raise BaselineStudyConfigError(
                f"invalid study config in {path}: neither baseline nor "
                f"held-out (baseline: {_bounded(str(baseline_exc))}; "
                f"held-out: {_bounded(str(held_exc))})"
            ) from None
