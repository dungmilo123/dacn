"""Pure matched-artifact preflight gate for the opt-in ablation study (S03/T03).

The opt-in eight-policy study (R021/D036/D037) is only a *matched* study when
its three artifact-backed panel entries -- ``bayesian`` (full),
``bayesian_no_belief`` and ``bayesian_no_strategic`` -- are genuinely derived
from one calibration run and carry no legacy, relabelled, or mismatched
artifact. This module loads each artifact exactly once through the unchanged
production :func:`~dfl_sim.bayesian_artifact.load_bayesian_artifact` loader and
returns an immutable identity summary the runners can reuse for the whole
invocation (Q6).

What is checked:

* every artifact's aggregate digest equals the study's pinned expected digest;
* the three artifacts have distinct aggregate digests and distinct variants;
* the declared source/effective mixtures match the variant assumptions;
* every artifact carries fresh :class:`~dfl_sim.bayesian_artifact.CalibrationDerivation`
  metadata (an unproven legacy full artifact is rejected for ablation
  comparison, while ordinary legacy studies are never routed here);
* the normalized calibration-config digest, actual dataset-inputs,
  partition/design membership and schedule-pairing fingerprints are identical
  across the three variants and equal the pinned
  ``matched_calibration_input_digest``;
* feature bins and the finite action space are identical across the three
  artifacts and agree with the pinned panel selectors;
* the sampler settings are identical except for the declared effective attacker
  mixture, which is bound to each artifact's derivation (so the fixed-strong
  no-strategic mixture is allowed while any other sampler drift is rejected);
* the declared episode/round/design-size/candidate budgets are identical;
* the artifacts' declared data provenance matches the study's declared
  provenance (``injected`` -> ``injected``, ``real_mnist`` -> ``production``).

Only the declared uniform-belief difference and the fixed-attack intensity /
trajectory difference are whitelisted: the likelihood and payoff *table bytes*
are never compared across variants. Evaluation seeds and runtime attacker
schedules are intentionally not required to equal the calibration seed or the
calibration mixture; that separation is what keeps a legitimate held-out
schedule from being mistaken for a broken match.

Failures raise :class:`AblationEvaluationError` with a bounded
``validation_class`` and a policy-id label; loader diagnostics are reduced to
the fixed filename and validation class so no raw path, YAML/JSON, or tensor
content can leak into a study error.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .ablation import (
    NO_STRATEGIC_EFFECTIVE_MU_A,
    NO_STRATEGIC_VARIANT,
    SOURCE_MU_A,
    matched_calibration_input_digest,
)
from .bayesian_artifact import (
    BayesianArtifact,
    BayesianArtifactError,
    CalibrationDerivation,
    load_bayesian_artifact,
)
from .config import BayesianAggregation
from .evaluation_config import (
    ABLATION_VARIANT_BY_POLICY_ID,
    ARTIFACT_POLICY_IDS,
    AblationStudyContract,
    BaselineStudyConfig,
    PanelPolicy,
)

__all__ = [
    "AblationEvaluationError",
    "AblationMatchSummary",
    "AblationVariantIdentity",
    "preflight_ablation_study",
]

#: Maximum rendered diagnostic detail length (bounded study errors, Q5).
_MAX_DETAIL = 400

#: The sampler-provenance fields that must be identical across the three
#: matched artifacts. ``mu_a`` is deliberately excluded: it is the declared
#: effective attacker mixture, bound separately to the derivation.
_SAMPLER_FIELDS = (
    "reference_policy",
    "p0",
    "placement",
    "max_compromised",
    "smoothing_alpha",
    "design_split",
)

#: The declared derivation budget fields that must be identical.
_BUDGET_FIELDS = (
    "episodes",
    "rounds",
    "design_size",
    "candidate_protocol",
    "candidate_cap",
)

#: The shared calibration-input fingerprints that must be identical.
_INPUT_FINGERPRINT_FIELDS = (
    "calibration_config_digest",
    "dataset_inputs_digest",
    "partition_membership_digest",
    "schedule_pairing_digest",
)


def _bounded(detail: object) -> str:
    collapsed = " ".join(str(detail).split())
    if len(collapsed) > _MAX_DETAIL:
        collapsed = collapsed[: _MAX_DETAIL - 3] + "..."
    return collapsed


class AblationEvaluationError(RuntimeError):
    """Raised when an opt-in ablation study fails matched-artifact preflight.

    Carries a machine-checkable ``validation_class``; the message is bounded and
    names only the offending policy id, never a filesystem path or payload.
    """

    def __init__(self, validation_class: str, detail: str = "") -> None:
        message = f"ablation_preflight: {validation_class}"
        if detail:
            message = f"{message}: {_bounded(detail)}"
        super().__init__(message)
        self.validation_class = validation_class


@dataclass(frozen=True)
class AblationVariantIdentity:
    """Immutable matched identity of one artifact-backed policy variant."""

    policy_id: str
    variant: str
    aggregate_digest: str
    matched_calibration_input_digest: str
    calibration_config_digest: str
    dataset_inputs_digest: str
    partition_membership_digest: str
    schedule_pairing_digest: str
    source_mu_a: tuple[float, ...]
    effective_mu_a: tuple[float, ...]
    seed: int
    episodes: int
    rounds: int
    design_size: int
    candidate_protocol: str
    candidate_cap: int
    data_provenance: str
    protocol_version: str


@dataclass(frozen=True)
class AblationMatchSummary:
    """Immutable, reusable result of one matched-artifact preflight.

    ``artifacts`` holds the three already-loaded production artifacts so a
    runner can reuse the validated triple within one invocation instead of
    reloading per cell (Q6). ``variants`` maps each canonical policy id to its
    :class:`AblationVariantIdentity`.
    """

    matched_calibration_input_digest: str
    protocol_version: str
    data_provenance: str
    matched_policy_ids: tuple[str, ...]
    variants: Mapping[str, AblationVariantIdentity]
    artifacts: Mapping[str, BayesianArtifact]

    def policy_ids(self) -> tuple[str, ...]:
        """The three canonical artifact-backed policy ids, in order."""

        return self.matched_policy_ids

    def identity(self, policy_id: str) -> AblationVariantIdentity:
        """Return the matched identity for ``policy_id``."""

        return self.variants[policy_id]

    def artifact(self, policy_id: str) -> BayesianArtifact:
        """Return the already-loaded artifact for ``policy_id``."""

        return self.artifacts[policy_id]

    def variant(self, policy_id: str) -> str:
        """Return the calibration variant for ``policy_id``."""

        return self.variants[policy_id].variant


# --------------------------------------------------------------------------- #
# Per-artifact checks
# --------------------------------------------------------------------------- #
def _require_contract(study: BaselineStudyConfig) -> AblationStudyContract:
    if not isinstance(study, BaselineStudyConfig):
        raise AblationEvaluationError(
            "study_type", "requires a BaselineStudyConfig study"
        )
    if study.ablations is None:
        raise AblationEvaluationError(
            "not_ablation_study", "study does not declare an ablation contract"
        )
    return study.ablations


def _load_artifact(
    policy_id: str, entry: PanelPolicy
) -> BayesianArtifact:
    aggregation = entry.aggregation
    if not isinstance(aggregation, BayesianAggregation):
        raise AblationEvaluationError(
            "kind_mismatch",
            f"policy {policy_id!r} is not backed by a bayesian artifact",
        )
    try:
        return load_bayesian_artifact(aggregation.artifact_dir)
    except BayesianArtifactError as exc:
        # Reduce the loader diagnostic to its fixed filename + validation class:
        # never surface the configured path or payload content.
        raise AblationEvaluationError(
            "artifact_load",
            f"policy {policy_id!r} failed to load "
            f"({exc.filename}: {exc.validation_class})",
        ) from None


def _require_fresh_derivation(
    policy_id: str, artifact: BayesianArtifact
) -> CalibrationDerivation:
    derivation = artifact.manifest.derivation
    if derivation is None:
        raise AblationEvaluationError(
            "unproven_artifact",
            f"policy {policy_id!r} is a legacy artifact without "
            "calibration-derivation metadata",
        )
    return derivation


def _require_variant(
    policy_id: str, derivation: CalibrationDerivation, expected: str
) -> None:
    if derivation.variant != expected:
        raise AblationEvaluationError(
            "variant_mismatch",
            f"policy {policy_id!r} expected a {expected!r} artifact but the "
            f"loaded derivation declares {derivation.variant!r}",
        )


def _require_assumptions(
    policy_id: str, variant: str, derivation: CalibrationDerivation
) -> None:
    if tuple(derivation.source_mu_a) != SOURCE_MU_A:
        raise AblationEvaluationError(
            "assumption_mismatch",
            f"policy {policy_id!r} source mixture is not the canonical source",
        )
    expected = (
        NO_STRATEGIC_EFFECTIVE_MU_A
        if variant == NO_STRATEGIC_VARIANT
        else SOURCE_MU_A
    )
    if tuple(derivation.effective_mu_a) != expected:
        raise AblationEvaluationError(
            "assumption_mismatch",
            f"policy {policy_id!r} effective mixture does not match the "
            f"{variant!r} variant assumption",
        )


def _require_panel_action_space(
    policy_id: str, artifact: BayesianArtifact, entry: PanelPolicy
) -> None:
    aggregation = entry.aggregation
    assert isinstance(aggregation, BayesianAggregation)
    space = artifact.manifest.action_space
    for name, artifact_value, panel_value in (
        ("omega", space.omega, aggregation.omega),
        ("degree_cap", space.degree_cap, aggregation.degree_cap),
        ("tie_tolerance", space.tie_tolerance, aggregation.tie_tolerance),
    ):
        if artifact_value != panel_value:
            raise AblationEvaluationError(
                "action_space_mismatch",
                f"policy {policy_id!r} artifact {name} does not match the "
                "pinned panel selector",
            )


def _build_identity(
    policy_id: str,
    variant: str,
    artifact: BayesianArtifact,
    derivation: CalibrationDerivation,
) -> AblationVariantIdentity:
    return AblationVariantIdentity(
        policy_id=policy_id,
        variant=variant,
        aggregate_digest=artifact.manifest.aggregate_digest,
        matched_calibration_input_digest=matched_calibration_input_digest(
            derivation
        ),
        calibration_config_digest=derivation.calibration_config_digest,
        dataset_inputs_digest=derivation.dataset_inputs_digest,
        partition_membership_digest=derivation.partition_membership_digest,
        schedule_pairing_digest=derivation.schedule_pairing_digest,
        source_mu_a=tuple(derivation.source_mu_a),
        effective_mu_a=tuple(derivation.effective_mu_a),
        seed=derivation.seed,
        episodes=derivation.episodes,
        rounds=derivation.rounds,
        design_size=derivation.design_size,
        candidate_protocol=derivation.candidate_protocol,
        candidate_cap=derivation.candidate_cap,
        data_provenance=derivation.data_provenance,
        protocol_version=derivation.protocol_version,
    )


# --------------------------------------------------------------------------- #
# Cross-artifact checks
# --------------------------------------------------------------------------- #
def _verify_pinned_digests(
    contract: AblationStudyContract,
    loaded: Mapping[str, BayesianArtifact],
) -> None:
    for policy_id, artifact in loaded.items():
        variant = ABLATION_VARIANT_BY_POLICY_ID[policy_id]
        expected = contract.expected_digest(variant)
        actual = artifact.manifest.aggregate_digest
        if actual != expected:
            raise AblationEvaluationError(
                "aggregate_digest_mismatch",
                f"policy {policy_id!r} aggregate digest does not match the "
                "pinned expected digest",
            )


def _verify_distinct_artifacts(
    loaded: Mapping[str, BayesianArtifact],
) -> None:
    digests = [artifact.manifest.aggregate_digest for artifact in loaded.values()]
    if len(set(digests)) != len(digests):
        raise AblationEvaluationError(
            "variant_identity_collision",
            "the three ablation artifacts must have distinct aggregate digests",
        )


def _verify_shared_inputs(
    contract: AblationStudyContract,
    identities: Mapping[str, AblationVariantIdentity],
) -> None:
    for policy_id, identity in identities.items():
        if (
            identity.matched_calibration_input_digest
            != contract.matched_calibration_input_digest
        ):
            raise AblationEvaluationError(
                "matched_input_mismatch",
                f"policy {policy_id!r} calibration input identity does not "
                "match the pinned matched_calibration_input_digest",
            )
    for field in _INPUT_FINGERPRINT_FIELDS:
        values = {getattr(identity, field) for identity in identities.values()}
        if len(values) != 1:
            raise AblationEvaluationError(
                "input_identity_divergence",
                f"ablation artifacts disagree on {field}",
            )


def _verify_shared_structure(
    loaded: Mapping[str, BayesianArtifact],
) -> None:
    reference = loaded[ARTIFACT_POLICY_IDS[0]].manifest
    for policy_id in ARTIFACT_POLICY_IDS[1:]:
        manifest = loaded[policy_id].manifest
        if manifest.bins != reference.bins:
            raise AblationEvaluationError(
                "bins_mismatch",
                f"policy {policy_id!r} feature bins do not match the matched "
                "artifact set",
            )
        if manifest.action_space != reference.action_space:
            raise AblationEvaluationError(
                "action_space_mismatch",
                f"policy {policy_id!r} action space does not match the matched "
                "artifact set",
            )


def _verify_sampler(
    loaded: Mapping[str, BayesianArtifact],
) -> None:
    reference = loaded[ARTIFACT_POLICY_IDS[0]].manifest.provenance
    for policy_id in ARTIFACT_POLICY_IDS:
        manifest = loaded[policy_id].manifest
        provenance = manifest.provenance
        derivation = manifest.derivation
        assert derivation is not None  # guaranteed by _require_fresh_derivation
        if tuple(provenance.mu_a) != tuple(derivation.effective_mu_a):
            raise AblationEvaluationError(
                "sampler_mismatch",
                f"policy {policy_id!r} sampler mixture does not match its "
                "declared effective mixture",
            )
        if policy_id == ARTIFACT_POLICY_IDS[0]:
            continue
        for field in _SAMPLER_FIELDS:
            if getattr(provenance, field) != getattr(reference, field):
                raise AblationEvaluationError(
                    "sampler_mismatch",
                    f"policy {policy_id!r} sampler setting {field!r} differs "
                    "from the matched artifact set",
                )


def _verify_budgets(
    identities: Mapping[str, AblationVariantIdentity],
) -> None:
    reference = identities[ARTIFACT_POLICY_IDS[0]]
    for policy_id in ARTIFACT_POLICY_IDS[1:]:
        identity = identities[policy_id]
        for field in _BUDGET_FIELDS:
            if getattr(identity, field) != getattr(reference, field):
                raise AblationEvaluationError(
                    "budget_mismatch",
                    f"policy {policy_id!r} derivation budget {field!r} differs "
                    "from the matched artifact set",
                )


def _verify_study_provenance(
    study: BaselineStudyConfig,
    identities: Mapping[str, AblationVariantIdentity],
) -> None:
    expected = (
        "injected" if study.data_provenance == "injected" else "production"
    )
    for policy_id, identity in identities.items():
        if identity.data_provenance != expected:
            raise AblationEvaluationError(
                "data_provenance_mismatch",
                f"policy {policy_id!r} artifact provenance "
                f"{identity.data_provenance!r} does not match study "
                f"data_provenance {study.data_provenance!r}",
            )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def preflight_ablation_study(
    study: BaselineStudyConfig,
) -> AblationMatchSummary:
    """Load and validate the study's matched full/no-belief/no-strategic triple.

    Returns an immutable :class:`AblationMatchSummary` carrying the validated
    identities and the three loaded artifacts. Raises
    :class:`AblationEvaluationError` (bounded, path-free) on any mismatch, so a
    runner can fail closed before executing a single cell.
    """

    contract = _require_contract(study)
    panel = {entry.id: entry for entry in study.panel}
    for policy_id in ARTIFACT_POLICY_IDS:
        if policy_id not in panel:
            raise AblationEvaluationError(
                "missing_policy",
                f"panel is missing artifact-backed policy {policy_id!r}",
            )

    loaded: dict[str, BayesianArtifact] = {}
    identities: dict[str, AblationVariantIdentity] = {}
    for policy_id in ARTIFACT_POLICY_IDS:
        variant = ABLATION_VARIANT_BY_POLICY_ID[policy_id]
        entry = panel[policy_id]
        artifact = _load_artifact(policy_id, entry)
        derivation = _require_fresh_derivation(policy_id, artifact)
        _require_variant(policy_id, derivation, variant)
        _require_assumptions(policy_id, variant, derivation)
        _require_panel_action_space(policy_id, artifact, entry)
        loaded[policy_id] = artifact
        identities[policy_id] = _build_identity(
            policy_id, variant, artifact, derivation
        )

    _verify_pinned_digests(contract, loaded)
    _verify_distinct_artifacts(loaded)
    _verify_shared_inputs(contract, identities)
    _verify_shared_structure(loaded)
    _verify_sampler(loaded)
    _verify_budgets(identities)
    _verify_study_provenance(study, identities)

    protocol_version = identities[ARTIFACT_POLICY_IDS[0]].protocol_version
    data_provenance = identities[ARTIFACT_POLICY_IDS[0]].data_provenance
    return AblationMatchSummary(
        matched_calibration_input_digest=(
            contract.matched_calibration_input_digest
        ),
        protocol_version=protocol_version,
        data_provenance=data_provenance,
        matched_policy_ids=ARTIFACT_POLICY_IDS,
        variants=MappingProxyType(dict(identities)),
        artifacts=MappingProxyType(dict(loaded)),
    )
