"""Deterministic smoothed calibration-table estimator (T03, R013/R014/R016).

This module turns the offline records collected by
:mod:`dfl_sim.calibration_scoring` into the exact schema-v1 row models owned by
:mod:`dfl_sim.bayesian_artifact`:

* an explicit feasible at-most-one-compromised **prior** over joint neighbor
  types, derived from the sampler's per-degree state counts with symmetric
  additive smoothing and normalized per degree;
* normalized **current-evidence likelihood** rows ``P(x_j | theta_j, c)`` with
  the full configured evidence-bin Cartesian support per reached context; and
* finite, zero-shrunk **conditional-payoff** rows ``q(x_j, c, m_j, theta_j)``
  estimated from the collected self-only-anchored neighbor marginals.

Estimation is deterministic: rows are sorted by explicit tuple keys, floats are
finite, per-context probabilities are re-normalized and re-checked, and the
payload bytes are the canonical JSON encoding the S01 loader re-validates. Any
duplicate key, unsupported context, cardinality drift, non-finite arithmetic,
or normalization error raises :class:`CalibrationEstimationError` *before* any
byte is serialized, so a malformed table can never reach publication.

Support and coverage are computed from **empirical (pre-pseudocount)** counts.
Likelihood coverage counts observed ``(degree_bin, progress, local_type,
evidence_bin)`` cells against the eligible cells of every reached
``(degree_bin, progress)`` context (both local types), and payoff coverage
counts observed ``(..., multiplier)`` cells against the same eligible context
space times the configured multipliers. ``posterior_calibration`` is the
bounded ``1 - mean Brier`` score of the estimated prior/likelihood over the
collected typed neighbor observations.

Nothing here is importable from the online policy / artifact-loader boundary;
the estimator is offline-only and consumes no dataset, tensor, or privileged
label.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .ablation import calibration_config_digest
from .bayesian_artifact import (
    ABLATION_PROTOCOL_VERSION,
    ARTIFACT_SCHEMA,
    ARTIFACT_SCHEMA_VERSION,
    CANDIDATE_PROTOCOL,
    ISOLATION_STATEMENTS,
    LIKELIHOOD_FILENAME,
    NO_BELIEF_VARIANT,
    PAYOFF_FILENAME,
    ArtifactManifest,
    CalibrationDerivation,
    CoverageMetadata,
    LikelihoodCell,
    PayoffCell,
    PriorCell,
    SamplerProvenance,
    SupportMetadata,
    canonical_json_bytes,
    compute_aggregate_digest,
)
from .calibration import (
    PriorStateCounts,
    count_prior_states,
    feasible_prior_states,
)
from .calibration_config import CalibrationConfig
from .calibration_scoring import (
    CalibrationCollection,
    CollectionProvenance,
    NodeRoundRecord,
)

__all__ = [
    "CalibrationEstimationError",
    "EstimatedArtifact",
    "estimate_artifact",
    "estimate_calibration",
]

#: Tolerance for per-context/per-degree normalization checks.
_NORMALIZATION_TOLERANCE = 1e-9

EvidenceKey = tuple[int, int, int]
ContextKey = tuple[str, str, int]
LikelihoodKey = tuple[str, str, int, EvidenceKey]
PayoffKey = tuple[str, str, int, EvidenceKey, float]


class CalibrationEstimationError(RuntimeError):
    """Raised when collected records cannot form a valid schema-v1 artifact.

    Raised for an unsupported or duplicate context, evidence/degree/progress
    cardinality drift, non-finite arithmetic, a normalization failure, or empty
    empirical support. Every failure aborts the calibration before any
    serialization.
    """


@dataclass(frozen=True, slots=True)
class EstimatedArtifact:
    """A fully assembled, canonical, loader-valid artifact (bytes included).

    ``manifest`` is the S01 schema-v1 envelope with its declared payload
    digests and aggregate digest; ``likelihood_bytes`` / ``payoff_bytes`` are
    the exact canonical payload bytes the manifest declares. Publication writes
    these same bytes and re-validates the staged directory through the S01
    loader.
    """

    manifest: ArtifactManifest
    likelihood_bytes: bytes
    payoff_bytes: bytes

    @property
    def aggregate_digest(self) -> str:
        return self.manifest.aggregate_digest

    @property
    def support(self) -> SupportMetadata:
        return self.manifest.support

    @property
    def coverage(self) -> CoverageMetadata:
        return self.manifest.coverage


# --------------------------------------------------------------------------- #
# Small numeric / validation helpers
# --------------------------------------------------------------------------- #
def _finite(value: object, *, label: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationEstimationError(
            f"{label} is not numeric ({context}): {value!r}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationEstimationError(
            f"{label} is not finite ({context}): {result!r}"
        )
    return result


def _evidence_cardinality(config: CalibrationConfig) -> tuple[int, int, int]:
    bins = config.bins
    return (
        len(bins.disagreement.labels),
        len(bins.relative_norm.labels),
        len(bins.loss_delta.labels),
    )


def _evidence_support(cardinality: tuple[int, int, int]) -> tuple[EvidenceKey, ...]:
    return tuple(
        itertools.product(
            range(cardinality[0]),
            range(cardinality[1]),
            range(cardinality[2]),
        )
    )


def _validate_evidence(
    evidence: object, *, cardinality: tuple[int, int, int], where: str
) -> EvidenceKey:
    if not isinstance(evidence, tuple) or len(evidence) != 3:
        raise CalibrationEstimationError(
            f"evidence_bin must be a 3-tuple ({where}): {evidence!r}"
        )
    out: list[int] = []
    for value, limit in zip(evidence, cardinality):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CalibrationEstimationError(
                f"evidence index is not an int ({where}): {value!r}"
            )
        if not 0 <= value < limit:
            raise CalibrationEstimationError(
                f"evidence index {value} out of range 0..{limit - 1} ({where})"
            )
        out.append(value)
    return (out[0], out[1], out[2])


def _validate_context(
    *,
    degree_bin: object,
    progress: object,
    local_type: object,
    config: CalibrationConfig,
    where: str,
) -> ContextKey:
    if degree_bin not in config.bins.degree.labels:
        raise CalibrationEstimationError(
            f"unsupported degree_bin ({where}): {degree_bin!r}"
        )
    if progress not in config.bins.progress.labels:
        raise CalibrationEstimationError(
            f"unsupported progress context ({where}): {progress!r}"
        )
    if local_type not in (0, 1):
        raise CalibrationEstimationError(
            f"local_type must be 0 or 1 ({where}): {local_type!r}"
        )
    return (str(degree_bin), str(progress), int(local_type))


def _require_config(config: object) -> CalibrationConfig:
    if not isinstance(config, CalibrationConfig):
        raise CalibrationEstimationError(
            f"estimate_artifact requires a CalibrationConfig, got "
            f"{type(config).__name__}"
        )
    return config


# --------------------------------------------------------------------------- #
# Prior
# --------------------------------------------------------------------------- #
def _estimate_prior(
    config: CalibrationConfig,
    prior_counts: PriorStateCounts,
    *,
    uniform: bool = False,
) -> tuple[tuple[PriorCell, ...], list[int]]:
    if not isinstance(prior_counts, PriorStateCounts):
        raise CalibrationEstimationError(
            f"prior_counts must be PriorStateCounts, got "
            f"{type(prior_counts).__name__}"
        )
    alpha = float(config.smoothing_alpha)
    cells: list[PriorCell] = []
    observed: list[int] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()

    for degree in sorted(prior_counts.counts):
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
            raise CalibrationEstimationError(
                f"prior count degree must be a positive int, got {degree!r}"
            )
        feasible = feasible_prior_states(degree)
        feasible_set = set(feasible)
        raw = prior_counts.counts[degree]
        if not isinstance(raw, Mapping):
            raise CalibrationEstimationError(
                f"prior counts for degree {degree} must be a mapping"
            )
        for state in raw:
            if not isinstance(state, tuple) or state not in feasible_set:
                raise CalibrationEstimationError(
                    f"unsupported prior state for degree {degree}: {state!r}"
                )
            count = raw[state]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise CalibrationEstimationError(
                    f"prior count for degree {degree} must be non-negative, "
                    f"got {count!r}"
                )
        total = 0
        for state in feasible:
            total += int(raw.get(state, 0))
        if uniform:
            # Exact uniform mass over the complete feasible joint-state support
            # per degree (D036). The empirical counts above are still measured
            # and reported through the support diagnostics, never synthesized.
            uniform_probability = 1.0 / len(feasible)
            denominator = None
        else:
            uniform_probability = 0.0
            denominator = total + alpha * len(feasible)
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise CalibrationEstimationError(
                    "prior normalization denominator for degree "
                    f"{degree} is invalid"
                )
        running = 0.0
        for state in feasible:
            count = int(raw.get(state, 0))
            probability = (
                uniform_probability
                if uniform
                else (count + alpha) / float(denominator)  # type: ignore[arg-type]
            )
            _finite(probability, label="prior probability", context=f"degree {degree}")
            if probability < 0.0 or probability > 1.0:
                raise CalibrationEstimationError(
                    f"prior probability out of range for degree {degree}"
                )
            key = (degree, state)
            if key in seen:  # pragma: no cover - feasibility is unique
                raise CalibrationEstimationError(
                    f"duplicate prior record for degree {degree}"
                )
            seen.add(key)
            cells.append(
                PriorCell(degree=degree, state=state, probability=probability)
            )
            running += probability
            if count > 0:
                observed.append(count)
        if abs(running - 1.0) > _NORMALIZATION_TOLERANCE:
            raise CalibrationEstimationError(
                f"prior mass for degree {degree} does not sum to 1 "
                f"(got {running!r})"
            )

    if not cells:
        raise CalibrationEstimationError("empty prior support: no degree counts")
    cells.sort(key=lambda cell: (cell.degree, cell.state))
    return tuple(cells), observed


# --------------------------------------------------------------------------- #
# Likelihood
# --------------------------------------------------------------------------- #
def _estimate_likelihood(
    config: CalibrationConfig,
    records: Sequence[NodeRoundRecord],
    *,
    uniform: bool = False,
) -> tuple[tuple[LikelihoodCell, ...], int, int, list[int]]:
    alpha = float(config.smoothing_alpha)
    cardinality = _evidence_cardinality(config)
    support = _evidence_support(cardinality)
    counts: dict[LikelihoodKey, int] = {}
    pairs: set[tuple[str, str]] = set()

    for record in records:
        if len(record.evidence_bins) != record.degree or len(
            record.hidden_types
        ) != record.degree:
            raise CalibrationEstimationError(
                f"record cardinality drift at node {record.node_id} "
                f"round {record.round_index}"
            )
        for position in range(record.degree):
            context = _validate_context(
                degree_bin=record.degree_bin,
                progress=record.progress,
                local_type=record.hidden_types[position],
                config=config,
                where=f"node {record.node_id} round {record.round_index}",
            )
            evidence = _validate_evidence(
                record.evidence_bins[position],
                cardinality=cardinality,
                where=f"node {record.node_id} round {record.round_index}",
            )
            key = (context[0], context[1], context[2], evidence)
            counts[key] = counts.get(key, 0) + 1
            pairs.add((context[0], context[1]))

    if not counts:
        raise CalibrationEstimationError("empty likelihood support: no observations")

    contexts = sorted((pair[0], pair[1], local_type) for pair in pairs for local_type in (0, 1))
    rows: list[LikelihoodCell] = []
    seen: set[LikelihoodKey] = set()
    for degree_bin, progress, local_type in contexts:
        total = sum(
            count
            for (db, pg, lt, _evidence), count in counts.items()
            if db == degree_bin and pg == progress and lt == local_type
        )
        if uniform:
            # Exact uniform mass over the complete evidence support for both
            # local types in every reached context (D036). Observed counts are
            # still measured below and reported through support diagnostics.
            uniform_probability = 1.0 / len(support)
            denominator = None
        else:
            uniform_probability = 0.0
            denominator = total + alpha * len(support)
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise CalibrationEstimationError(
                    f"likelihood normalization denominator invalid for "
                    f"({degree_bin!r}, {progress!r}, {local_type})"
                )
        running = 0.0
        for evidence in support:
            key = (degree_bin, progress, local_type, evidence)
            count = counts.get(key, 0)
            probability = (
                uniform_probability
                if uniform
                else (count + alpha) / float(denominator)  # type: ignore[arg-type]
            )
            _finite(
                probability,
                label="likelihood probability",
                context=f"{key!r}",
            )
            if key in seen:  # pragma: no cover - sorted unique contexts
                raise CalibrationEstimationError(
                    f"duplicate likelihood record: {key!r}"
                )
            seen.add(key)
            rows.append(
                LikelihoodCell(
                    degree_bin=degree_bin,
                    progress=progress,
                    local_type=local_type,
                    evidence_bin=evidence,
                    probability=probability,
                )
            )
            running += probability
        if abs(running - 1.0) > _NORMALIZATION_TOLERANCE:
            raise CalibrationEstimationError(
                f"likelihood mass for {key!r} context does not sum to 1 "
                f"(got {running!r})"
            )

    rows.sort(
        key=lambda row: (
            row.degree_bin,
            row.progress,
            row.local_type,
            row.evidence_bin,
        )
    )
    observed = [count for count in counts.values() if count > 0]
    return tuple(rows), len(pairs), len(support), observed


# --------------------------------------------------------------------------- #
# Payoff
# --------------------------------------------------------------------------- #
def _estimate_payoff(
    config: CalibrationConfig, records: Sequence[NodeRoundRecord]
) -> tuple[tuple[PayoffCell, ...], int, int, list[int]]:
    alpha = float(config.smoothing_alpha)
    multipliers = tuple(
        sorted(float(value) for value in config.action_space.multiplier_set)
    )
    sums: dict[PayoffKey, float] = {}
    counts: dict[PayoffKey, int] = {}
    cells: set[LikelihoodKey] = set()

    for record in records:
        for marginal in record.marginals:
            context = _validate_context(
                degree_bin=marginal.degree_bin,
                progress=marginal.progress,
                local_type=marginal.hidden_type,
                config=config,
                where=f"node {record.node_id} round {record.round_index}",
            )
            evidence = _validate_evidence(
                marginal.evidence_bin,
                cardinality=_evidence_cardinality(config),
                where=f"node {record.node_id} round {record.round_index}",
            )
            multiplier = _finite(
                marginal.multiplier,
                label="payoff multiplier",
                context=f"node {record.node_id}",
            )
            if multiplier not in multipliers:
                raise CalibrationEstimationError(
                    f"payoff multiplier {multiplier!r} is not in the declared "
                    f"action set"
                )
            utility = _finite(
                marginal.utility,
                label="payoff utility",
                context=f"node {record.node_id}",
            )
            cell = (context[0], context[1], context[2], evidence)
            key = (cell[0], cell[1], cell[2], cell[3], multiplier)
            sums[key] = sums.get(key, 0.0) + utility
            counts[key] = counts.get(key, 0) + 1
            cells.add(cell)

    if not cells:
        raise CalibrationEstimationError("empty payoff support: no marginals")

    rows: list[PayoffCell] = []
    seen: set[PayoffKey] = set()
    for cell in sorted(cells):
        for multiplier in multipliers:
            key = (cell[0], cell[1], cell[2], cell[3], multiplier)
            count = counts.get(key, 0)
            total = sums.get(key, 0.0)
            # Zero-shrunk conditional mean: sum / (count + alpha); an
            # unobserved multiplier collapses to exactly zero.
            value = total / (count + alpha)
            _finite(value, label="payoff value", context=f"{key!r}")
            if key in seen:  # pragma: no cover - sorted unique cells
                raise CalibrationEstimationError(f"duplicate payoff record: {key!r}")
            seen.add(key)
            rows.append(
                PayoffCell(
                    degree_bin=cell[0],
                    progress=cell[1],
                    local_type=cell[2],
                    evidence_bin=cell[3],
                    multiplier=multiplier,
                    value=value,
                )
            )

    rows.sort(
        key=lambda row: (
            row.degree_bin,
            row.progress,
            row.local_type,
            row.evidence_bin,
            row.multiplier,
        )
    )
    observed = [count for count in counts.values() if count > 0]
    return tuple(rows), len(cells), len(multipliers), observed


# --------------------------------------------------------------------------- #
# Posterior calibration (Brier score)
# --------------------------------------------------------------------------- #
def _posterior_calibration(
    *,
    records: Sequence[NodeRoundRecord],
    prior: Sequence[PriorCell],
    likelihood: Sequence[LikelihoodCell],
) -> float:
    prior_by_degree: dict[int, dict[tuple[int, ...], float]] = {}
    for cell in prior:
        prior_by_degree.setdefault(cell.degree, {})[cell.state] = cell.probability
    likelihood_map: dict[LikelihoodKey, float] = {
        (row.degree_bin, row.progress, row.local_type, row.evidence_bin): row.probability
        for row in likelihood
    }

    brier_total = 0.0
    pairs = 0
    for record in records:
        cells = prior_by_degree.get(record.degree)
        if not cells:
            raise CalibrationEstimationError(
                f"empty-support posterior: no prior for degree {record.degree} "
                f"(node {record.node_id})"
            )
        weights: list[tuple[tuple[int, ...], float]] = []
        for state, probability in cells.items():
            factors: list[float] = []
            for position, (local_type, evidence) in enumerate(
                zip(state, record.evidence_bins)
            ):
                likelihood_value = likelihood_map.get(
                    (record.degree_bin, record.progress, local_type, evidence)
                )
                if likelihood_value is None:
                    raise CalibrationEstimationError(
                        "empty-support posterior: missing likelihood for "
                        f"degree {record.degree} local_type {local_type} "
                        f"(node {record.node_id}, position {position})"
                    )
                factors.append(likelihood_value)
            product = math.prod(factors)
            weight = probability * product
            _finite(
                weight, label="posterior weight", context=f"node {record.node_id}"
            )
            weights.append((state, weight))
        mass = math.fsum(weight for _state, weight in weights)
        if not math.isfinite(mass) or mass <= 0.0:
            raise CalibrationEstimationError(
                f"empty-support posterior: zero mass (node {record.node_id})"
            )
        for position in range(record.degree):
            positive = math.fsum(
                weight
                for state, weight in weights
                if state[position] == 1
            )
            posterior_positive = positive / mass
            _finite(
                posterior_positive,
                label="posterior probability",
                context=f"node {record.node_id}",
            )
            observed = int(record.hidden_types[position])
            brier_total += (posterior_positive - observed) ** 2
            pairs += 1

    if pairs == 0:
        raise CalibrationEstimationError(
            "empty-support posterior: no typed neighbor observations"
        )
    brier = brier_total / pairs
    score = 1.0 - brier
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


# --------------------------------------------------------------------------- #
# Derivation metadata assembly
# --------------------------------------------------------------------------- #
def _bounded(detail: object, *, limit: int = 400) -> str:
    """Collapse whitespace and truncate a derivation diagnostic."""

    collapsed = " ".join(str(detail).split())
    if len(collapsed) > limit:
        collapsed = collapsed[:limit] + "..."
    return collapsed


def build_derivation(
    config: CalibrationConfig,
    provenance: CollectionProvenance,
    *,
    calibration_episodes: int,
    calibration_rounds: int,
) -> CalibrationDerivation:
    """Build the digest-covered derivation metadata for one calibrated artifact.

    The declared calibration-config digest is path-independent and
    variant-normalized (full/no_belief/no_strategic share one source identity);
    the dataset/membership/pairing digests are the immutable content
    fingerprints captured at collection. Every field is validated by the strict
    :class:`CalibrationDerivation` contract, so a malformed fingerprint or an
    inconsistent variant/effective mixture fails closed here, before any byte
    is serialized.
    """

    if not isinstance(provenance, CollectionProvenance):
        raise CalibrationEstimationError(
            "derivation requires a CollectionProvenance, got "
            f"{type(provenance).__name__}"
        )
    try:
        config_digest = calibration_config_digest(config.model_dump(mode="json"))
    except Exception as exc:  # AblationError, bounded and re-raised
        raise CalibrationEstimationError(
            f"calibration config identity failed: {_bounded(exc)}"
        ) from exc
    try:
        return CalibrationDerivation(
            protocol_version=ABLATION_PROTOCOL_VERSION,
            variant=config.variant,
            isolation_statement=ISOLATION_STATEMENTS[config.variant],
            source_mu_a=tuple(float(value) for value in config.mu_a),
            effective_mu_a=tuple(float(value) for value in config.effective_mu_a),
            seed=int(config.experiment.seed),
            calibration_config_digest=config_digest,
            dataset_inputs_digest=provenance.dataset_inputs_digest,
            partition_membership_digest=provenance.partition_membership_digest,
            schedule_pairing_digest=provenance.schedule_pairing_digest,
            episodes=int(calibration_episodes),
            rounds=int(calibration_rounds),
            design_size=int(config.design_split_size),
            candidate_protocol=CANDIDATE_PROTOCOL,
            candidate_cap=int(config.action_space.degree_cap),
            data_provenance=provenance.data_provenance,
        )
    except ValidationError as exc:
        raise CalibrationEstimationError(
            f"derivation metadata invalid: {_bounded(exc)}"
        ) from exc


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def estimate_artifact(
    config: CalibrationConfig,
    records: Sequence[NodeRoundRecord],
    prior_counts: PriorStateCounts,
    *,
    calibration_episodes: int,
    calibration_rounds: int,
    provenance: CollectionProvenance | None = None,
) -> EstimatedArtifact:
    """Estimate every schema-v1 table from collected records and freeze bytes.

    Builds the feasible prior, the smoothed full-support likelihood rows, the
    zero-shrunk payoff rows, and the support/coverage/calibration diagnostics,
    then returns the canonical payload bytes plus the exact S01 manifest with
    declared payload and aggregate digests. Raises
    :class:`CalibrationEstimationError` on any invalid input before
    serialization.

    The ``variant`` on ``config`` selects the belief-ablation transform:

    * ``full`` and ``no_strategic`` keep the smoothed empirical prior and
      likelihood; ``no_strategic`` only changes the effective attacker mixture
      used at schedule build time, and its fresh records re-estimate all three
      tables.
    * ``no_belief`` replaces the prior with exact uniform mass over the
      feasible joint states per degree and the likelihood with exact uniform
      mass over the complete evidence support for both local types in every
      reached context. The payoff table is still estimated from the unchanged
      records, so full and no-belief share byte-identical payoff tables. Its
      Brier-style posterior diagnostic is explicitly marked non-applicable
      rather than presented as meaningful belief calibration.

    When ``provenance`` is supplied (the production calibration path always
    supplies it) the manifest also carries digest-covered
    :class:`CalibrationDerivation` metadata. Callers that omit it get a
    loader-valid legacy-style artifact without derivation metadata, which the
    ablation match gate rejects for comparison.
    """

    config = _require_config(config)
    if isinstance(calibration_episodes, bool) or not isinstance(
        calibration_episodes, int
    ):
        raise CalibrationEstimationError("calibration_episodes must be an int")
    if isinstance(calibration_rounds, bool) or not isinstance(
        calibration_rounds, int
    ):
        raise CalibrationEstimationError("calibration_rounds must be an int")
    if calibration_episodes < 0 or calibration_rounds < 0:
        raise CalibrationEstimationError(
            "calibration episode/round counts must be non-negative"
        )
    if provenance is not None and not isinstance(provenance, CollectionProvenance):
        raise CalibrationEstimationError(
            "provenance must be a CollectionProvenance when supplied, got "
            f"{type(provenance).__name__}"
        )

    uniform_belief = config.variant == NO_BELIEF_VARIANT
    if uniform_belief and provenance is None:
        # A no-belief artifact marks its belief metric non-applicable, which the
        # loader only accepts alongside digest-covered derivation metadata that
        # explains the variant. Without provenance the artifact could not be
        # loaded, so fail closed instead of emitting an unloadable table set.
        raise CalibrationEstimationError(
            "no_belief estimation requires collection provenance so the "
            "non-applicable belief metric is self-describing"
        )

    prior, prior_counts_observed = _estimate_prior(
        config, prior_counts, uniform=uniform_belief
    )
    likelihood, pair_count, evidence_count, likelihood_counts = (
        _estimate_likelihood(config, records, uniform=uniform_belief)
    )
    payoff, payoff_cell_count, multiplier_count, payoff_counts = _estimate_payoff(
        config, records
    )

    if uniform_belief:
        # A full-policy Brier score is not meaningful no-belief evidence, so it
        # is explicitly marked non-applicable instead of being presented as a
        # calibrated belief score.
        posterior_calibration = 0.0
        posterior_applicable = False
    else:
        posterior_calibration = _posterior_calibration(
            records=records, prior=prior, likelihood=likelihood
        )
        posterior_applicable = True

    empirical_counts = prior_counts_observed + likelihood_counts + payoff_counts
    if not empirical_counts:
        raise CalibrationEstimationError("empty support: no empirical observations")
    min_cell_count = min(empirical_counts)

    contexts = pair_count * 2
    eligible_likelihood = contexts * evidence_count
    eligible_payoff = eligible_likelihood * multiplier_count
    observed_likelihood = len(likelihood_counts)
    observed_payoff = len(payoff_counts)
    if observed_likelihood > eligible_likelihood or observed_payoff > eligible_payoff:
        raise CalibrationEstimationError("coverage numerator exceeds eligible cells")
    likelihood_coverage = (
        observed_likelihood / eligible_likelihood if eligible_likelihood else 0.0
    )
    payoff_coverage = (
        observed_payoff / eligible_payoff if eligible_payoff else 0.0
    )

    support = SupportMetadata(
        calibration_episodes=int(calibration_episodes),
        calibration_rounds=int(calibration_rounds),
        prior_cells=len(prior),
        likelihood_cells=len(likelihood),
        payoff_cells=len(payoff),
        min_cell_count=int(min_cell_count),
    )
    coverage = CoverageMetadata(
        likelihood=float(likelihood_coverage),
        payoff=float(payoff_coverage),
        posterior_calibration=float(posterior_calibration),
        posterior_calibration_applicable=bool(posterior_applicable),
    )

    likelihood_bytes = canonical_json_bytes(
        [row.model_dump(mode="json") for row in likelihood]
    )
    payoff_bytes = canonical_json_bytes(
        [row.model_dump(mode="json") for row in payoff]
    )

    sampler_provenance = SamplerProvenance(
        reference_policy="plain_average",
        p0=float(config.p0),
        # The effective mixture (not the pinned source mixture) is what the
        # sampler actually scheduled; recording it keeps no-strategic
        # self-describing and lets the loader bind provenance to the declared
        # derivation.
        mu_a=tuple(float(value) for value in config.effective_mu_a),
        placement="uniform_non_isolated",
        max_compromised=int(config.max_compromised),
        smoothing_alpha=float(config.smoothing_alpha),
        design_split="clean_global_heldout",
    )

    derivation = (
        None
        if provenance is None
        else build_derivation(
            config,
            provenance,
            calibration_episodes=int(calibration_episodes),
            calibration_rounds=int(calibration_rounds),
        )
    )

    body: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "provenance": sampler_provenance.model_dump(mode="json"),
        "action_space": config.action_space.model_dump(mode="json"),
        "bins": config.bins.model_dump(mode="json"),
        "prior": [cell.model_dump(mode="json") for cell in prior],
        "support": support.model_dump(mode="json"),
        "coverage": coverage.model_dump(mode="json"),
        # Optional schema-v1 ablation provenance (T01/T02). Always present in
        # the canonical body so the aggregate digest matches a default
        # ``model_dump`` of the parsed manifest, which emits the defaulted
        # field as JSON null when absent.
        "derivation": None
        if derivation is None
        else derivation.model_dump(mode="json"),
        "payloads": {
            LIKELIHOOD_FILENAME: {
                "size_bytes": len(likelihood_bytes),
                "sha256": _sha256(likelihood_bytes),
            },
            PAYOFF_FILENAME: {
                "size_bytes": len(payoff_bytes),
                "sha256": _sha256(payoff_bytes),
            },
        },
    }
    aggregate_digest = compute_aggregate_digest(body)
    full = {**body, "aggregate_digest": aggregate_digest}
    manifest = ArtifactManifest.model_validate(full)

    return EstimatedArtifact(
        manifest=manifest,
        likelihood_bytes=likelihood_bytes,
        payoff_bytes=payoff_bytes,
    )


def estimate_calibration(
    config: CalibrationConfig,
    collection: CalibrationCollection,
    *,
    prior_counts: PriorStateCounts | None = None,
) -> EstimatedArtifact:
    """Estimate the artifact for a collected calibration (convenience wrapper)."""

    config = _require_config(config)
    if not isinstance(collection, CalibrationCollection):
        raise CalibrationEstimationError(
            f"estimate_calibration requires a CalibrationCollection, got "
            f"{type(collection).__name__}"
        )
    if prior_counts is None:
        prior_counts = count_prior_states(config, collection.schedule.specs)
    return estimate_artifact(
        config,
        collection.records,
        prior_counts,
        calibration_episodes=collection.episode_count,
        calibration_rounds=collection.round_count,
        provenance=collection.provenance,
    )


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
