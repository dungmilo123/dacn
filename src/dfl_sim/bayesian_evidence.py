"""Offline on-policy evidence aggregation for the Bayesian policy (R015/R016/T02).

This module lives strictly **offline**. It consumes the structured
:class:`~dfl_sim.policy.BayesianDiagnostic` records emitted by the online
seam, the resolved experiment config (which owns the fixed per-episode attack
identity/intensity), and an already-loaded frozen
:class:`~dfl_sim.bayesian_artifact.BayesianArtifact`, and returns one immutable
:class:`BayesianEvidenceReport`.

Nothing here can reach the online decision path: the collector never appears
in ``policy.py``/``bayesian_artifact.py``, it never mutates the artifact or the
config, and the report carries only bounded aggregate scalars -- no tensors,
examples, labels, per-example data, or filesystem paths. The small set of
fields that do cross the boundary are the artifact aggregate digest, the run
seed/provenance, and exact count/fraction aggregates.

Two held-out on-policy quantities are defined over **benign receivers only**
(a receiver equal to the configured compromised node under an active attack is
excluded, because its own selection is not an honest defense decision):

* **coverage** -- the fraction of benign-receiver neighbor observations whose
  declared ``(degree_bin, progress, hidden neighbor type, evidence cell)``
  likelihood row *and* every required payoff cell (one per declared multiplier)
  exist. A context that never resolved (``degree_bin``/``progress`` absent) or
  a missing row counts as uncovered.
* **distribution mismatch** -- the support-count-weighted mean total-variation
  distance between each non-empty held-out empirical evidence distribution and
  the artifact likelihood for the same
  ``(degree_bin, progress, hidden neighbor type)`` group, where the artifact
  likelihood is normalized over its declared evidence cells and undeclared
  cells take zero mass. Groups for which the artifact declares no likelihood
  mass are omitted (the ratio is undefined), and only groups with a positive
  observation count are included.

Operational aggregates (fallback reasons/rate, pruning, candidate counts, and
elapsed selection time) cover every recorded selection. Fallback counts are
reported for all documented reasons, including explicit zeros, so an absent
condition is distinguishable from an unreported one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .bayesian_artifact import BayesianArtifact
from .config import ExperimentConfig, topology_node_count
from .policy import (
    FALLBACK_CONTEXT_UNAVAILABLE,
    FALLBACK_MISSING_LIKELIHOOD,
    FALLBACK_MISSING_PAYOFF,
    FALLBACK_MISSING_PRIOR,
    FALLBACK_NON_FINITE,
    FALLBACK_ZERO_MASS,
    BayesianDiagnostic,
)

__all__ = [
    "BayesianEvidenceReport",
    "EvidenceError",
    "FALLBACK_REASONS",
    "REPORT_SCHEMA",
    "collect_bayesian_evidence",
    "render_bayesian_report",
]

#: One canonical schema name for the rendered stdout record.
REPORT_SCHEMA = "dfl_bayesian_evidence_report"
REPORT_SCHEMA_VERSION = 1

#: Every documented Bayesian fallback reason, in a stable rendered order.
FALLBACK_REASONS: tuple[str, ...] = (
    FALLBACK_CONTEXT_UNAVAILABLE,
    FALLBACK_MISSING_PRIOR,
    FALLBACK_MISSING_LIKELIHOOD,
    FALLBACK_MISSING_PAYOFF,
    FALLBACK_ZERO_MASS,
    FALLBACK_NON_FINITE,
)

_REASON_SET = frozenset(FALLBACK_REASONS)


class EvidenceError(RuntimeError):
    """Raised when on-policy evidence cannot be aggregated consistently.

    Covers empty/duplicated/out-of-order/non-Bayesian diagnostics, a
    non-Bayesian config, a malformed sender/evidence alignment, unknown context
    or evidence cells, and invalid durations.
    """


def _require_int(value: object, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{label} must be int, got {type(value).__name__}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise EvidenceError(f"{label} must be >= {minimum}, got {result}")
    return result


def _evidence_cardinality(artifact: BayesianArtifact) -> tuple[int, int, int]:
    """Return the inclusive evidence-bin limits implied by the artifact bins."""

    bins = artifact.manifest.bins
    return (
        len(bins.disagreement.edges) + 1,
        len(bins.relative_norm.edges) + 1,
        len(bins.loss_delta.edges) + 1,
    )


def _validate_diagnostic(
    diagnostic: object,
    *,
    artifact: BayesianArtifact,
    cardinality: tuple[int, int, int],
) -> BayesianDiagnostic:
    if type(diagnostic) is not BayesianDiagnostic:
        raise EvidenceError(
            "diagnostics must be exact BayesianDiagnostic records, got "
            f"{type(diagnostic).__name__}"
        )
    record = diagnostic
    _require_int(record.node_id, label="diagnostic node_id", minimum=0)
    _require_int(record.round_index, label="diagnostic round_index", minimum=0)
    degree = _require_int(record.degree, label="diagnostic degree", minimum=0)

    senders = record.sender_ids
    if not isinstance(senders, tuple) or len(senders) != degree:
        raise EvidenceError(
            "diagnostic sender/evidence misalignment: sender_ids must have "
            f"degree={degree} entries, got {len(senders) if isinstance(senders, tuple) else senders!r}"
        )
    previous: int | None = None
    for sender in senders:
        value = _require_int(sender, label="sender_id", minimum=0)
        if previous is not None and value <= previous:
            raise EvidenceError(
                "diagnostic sender_ids must be strictly ascending"
            )
        previous = value

    evidence = record.evidence_bins
    if not isinstance(evidence, tuple) or len(evidence) != degree:
        raise EvidenceError(
            "diagnostic sender/evidence misalignment: evidence_bins must have "
            f"degree={degree} entries"
        )
    for row in evidence:
        if (
            not isinstance(row, tuple)
            or len(row) != 3
            or any(isinstance(v, bool) or not isinstance(v, int) for v in row)
        ):
            raise EvidenceError(
                f"diagnostic evidence_bins row must be a 3-tuple of ints, got {row!r}"
            )
        for value, limit in zip(row, cardinality):
            if not 0 <= value < limit:
                raise EvidenceError(
                    "diagnostic evidence cell is outside the artifact bins: "
                    f"{row!r} (limit {cardinality!r})"
                )

    labels = artifact.manifest.bins
    if record.degree_bin is not None and record.degree_bin not in labels.degree.labels:
        raise EvidenceError(
            f"diagnostic degree_bin {record.degree_bin!r} is not declared by the artifact"
        )
    if record.progress is not None and record.progress not in labels.progress.labels:
        raise EvidenceError(
            f"diagnostic progress {record.progress!r} is not declared by the artifact"
        )

    if record.fallback_reason is not None and record.fallback_reason not in _REASON_SET:
        raise EvidenceError(
            f"diagnostic fallback_reason {record.fallback_reason!r} is unknown"
        )
    _require_int(record.candidate_count, label="diagnostic candidate_count", minimum=0)
    if not isinstance(record.pruned, bool):
        raise EvidenceError("diagnostic pruned must be bool")
    _require_int(record.elapsed_ns, label="diagnostic elapsed_ns", minimum=0)
    return record


def _cell_exists(
    artifact: BayesianArtifact,
    *,
    degree_bin: str,
    progress: str,
    local_type: int,
    evidence: tuple[int, int, int],
) -> bool:
    """Return whether the likelihood and every required payoff cell exist."""

    if (
        artifact.likelihood_for(degree_bin, progress, local_type, evidence)
        is None
    ):
        return False
    multipliers = artifact.multiplier_set
    return all(
        artifact.payoff_for(
            degree_bin, progress, local_type, evidence, float(multiplier)
        )
        is not None
        for multiplier in multipliers
    )


def _artifact_group_distribution(
    artifact: BayesianArtifact,
    *,
    degree_bin: str,
    progress: str,
    local_type: int,
) -> dict[tuple[int, int, int], float]:
    """Return the artifact's *normalized* evidence distribution for one group.

    The result maps each declared evidence cell to its normalized likelihood
    mass; undeclared cells are implicitly zero. An empty mapping means the
    artifact declares no likelihood mass for the group, so the group is
    omitted from the mismatch aggregate.
    """

    raw: dict[tuple[int, int, int], float] = {}
    for key, probability in artifact.likelihood_index.items():
        k_degree, k_progress, k_type, k_evidence = key
        if (
            k_degree == degree_bin
            and k_progress == progress
            and k_type == local_type
        ):
            raw[k_evidence] = raw.get(k_evidence, 0.0) + float(probability)
    total = math.fsum(raw.values())
    if total <= 0.0 or not math.isfinite(total):
        return {}
    return {evidence: mass / total for evidence, mass in raw.items()}


def _total_variation(
    empirical: Mapping[tuple[int, int, int], float],
    artifact: Mapping[tuple[int, int, int], float],
) -> float:
    """Return the total-variation distance between two evidence distributions."""

    support = set(empirical) | set(artifact)
    return 0.5 * math.fsum(
        abs(empirical.get(cell, 0.0) - artifact.get(cell, 0.0))
        for cell in support
    )


@dataclass(frozen=True, slots=True)
class BayesianEvidenceReport:
    """Immutable, bounded aggregate of one completed Bayesian run (R016).

    Every field is a scalar or an exact count; the record carries no tensors,
    examples, labels, per-example data, or unbounded paths. ``fallback_counts``
    always includes all documented reasons (with explicit zeros).
    """

    artifact_digest: str
    seed: int
    rounds: int
    node_count: int
    compromised_node: int | None
    attack_kappa: float
    selection_count: int
    benign_receiver_count: int
    neighbor_observation_count: int
    covered_observation_count: int
    coverage: float
    mismatch: float
    mismatch_group_count: int
    fallback_counts: tuple[tuple[str, int], ...]
    fallback_total: int
    fallback_rate: float
    pruned_count: int
    candidate_count_total: int
    candidate_count_max: int
    elapsed_ns_total: int
    elapsed_ns_min: int
    elapsed_ns_max: int

    def canonical_json(self) -> str:
        """Return the one canonical bounded JSON rendering of this report."""

        return render_bayesian_report(self)


def render_bayesian_report(report: BayesianEvidenceReport) -> str:
    """Render a report as canonical, finite, bounded, sorted-key JSON."""

    if not isinstance(report, BayesianEvidenceReport):
        raise EvidenceError(
            f"expected a BayesianEvidenceReport, got {type(report).__name__}"
        )
    payload = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_digest": report.artifact_digest,
        "seed": report.seed,
        "rounds": report.rounds,
        "node_count": report.node_count,
        "compromised_node": report.compromised_node,
        "attack_kappa": report.attack_kappa,
        "selection_count": report.selection_count,
        "benign_receiver_count": report.benign_receiver_count,
        "neighbor_observation_count": report.neighbor_observation_count,
        "covered_observation_count": report.covered_observation_count,
        "coverage": report.coverage,
        "mismatch": report.mismatch,
        "mismatch_group_count": report.mismatch_group_count,
        "fallback": {reason: count for reason, count in report.fallback_counts},
        "fallback_total": report.fallback_total,
        "fallback_rate": report.fallback_rate,
        "pruned_count": report.pruned_count,
        "candidate_count_total": report.candidate_count_total,
        "candidate_count_max": report.candidate_count_max,
        "elapsed_ns_total": report.elapsed_ns_total,
        "elapsed_ns_min": report.elapsed_ns_min,
        "elapsed_ns_max": report.elapsed_ns_max,
    }
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise EvidenceError(f"could not render evidence report: {exc}") from exc


def collect_bayesian_evidence(
    diagnostics: Iterable[BayesianDiagnostic],
    *,
    config: ExperimentConfig,
    artifact: BayesianArtifact,
) -> BayesianEvidenceReport:
    """Aggregate a completed Bayesian run's diagnostics into one bounded report.

    ``config`` must be a validated :class:`~dfl_sim.config.ExperimentConfig`
    whose aggregation is ``bayesian`` (its fixed attack identity/intensity
    supplies the hidden sender types); ``artifact`` must be the already-loaded
    frozen artifact the policy consumed. ``diagnostics`` must be the exact
    per-selection records in strictly ascending ``(round_index, node_id)``
    order with no duplicates; any inconsistency fails closed with
    :class:`EvidenceError` rather than yielding a misleading report.
    """

    if not isinstance(config, ExperimentConfig):
        raise EvidenceError(
            f"config must be an ExperimentConfig, got {type(config).__name__}"
        )
    aggregation = config.aggregation
    if getattr(aggregation, "kind", None) != "bayesian":
        raise EvidenceError(
            "bayesian evidence requires a bayesian aggregation config, got "
            f"{type(aggregation).__name__}"
        )
    if not isinstance(artifact, BayesianArtifact):
        raise EvidenceError(
            f"artifact must be a loaded BayesianArtifact, got {type(artifact).__name__}"
        )

    records = tuple(diagnostics)
    if not records:
        raise EvidenceError("no Bayesian diagnostics were recorded")

    cardinality = _evidence_cardinality(artifact)
    validated: list[BayesianDiagnostic] = []
    previous_key: tuple[int, int] | None = None
    seen_keys: set[tuple[int, int]] = set()
    for raw in records:
        record = _validate_diagnostic(
            raw, artifact=artifact, cardinality=cardinality
        )
        key = (int(record.round_index), int(record.node_id))
        if key in seen_keys:
            raise EvidenceError(
                f"duplicate Bayesian selection key round={key[0]} node={key[1]}"
            )
        if previous_key is not None and key < previous_key:
            raise EvidenceError(
                "Bayesian diagnostics must be in ascending (round, node) order"
            )
        seen_keys.add(key)
        previous_key = key
        validated.append(record)

    compromised = config.attack.compromised_node
    kappa = float(config.attack.kappa)

    def is_malicious(node: object) -> bool:
        return (
            compromised is not None
            and kappa > 0.0
            and int(node) == int(compromised)
        )

    coverage_covered = 0
    coverage_total = 0
    group_counts: dict[tuple[str, str, int], dict[tuple[int, int, int], int]] = {}

    for record in validated:
        if is_malicious(record.node_id):
            continue
        for index, sender in enumerate(record.sender_ids):
            coverage_total += 1
            evidence = record.evidence_bins[index]
            if record.degree_bin is None or record.progress is None:
                continue
            local_type = 1 if is_malicious(sender) else 0
            if _cell_exists(
                artifact,
                degree_bin=record.degree_bin,
                progress=record.progress,
                local_type=local_type,
                evidence=evidence,
            ):
                coverage_covered += 1
            bucket = group_counts.setdefault(
                (record.degree_bin, record.progress, local_type), {}
            )
            bucket[evidence] = bucket.get(evidence, 0) + 1

    coverage = 1.0 if coverage_total == 0 else coverage_covered / coverage_total

    mismatch_weighted = 0.0
    mismatch_support = 0
    mismatch_groups = 0
    for (degree_bin, progress, local_type), counts in group_counts.items():
        support = sum(counts.values())
        if support <= 0:
            continue
        artifact_distribution = _artifact_group_distribution(
            artifact,
            degree_bin=degree_bin,
            progress=progress,
            local_type=local_type,
        )
        if not artifact_distribution:
            continue
        empirical = {
            evidence: count / support for evidence, count in counts.items()
        }
        mismatch_weighted += support * _total_variation(
            empirical, artifact_distribution
        )
        mismatch_support += support
        mismatch_groups += 1
    mismatch = (
        0.0 if mismatch_support == 0 else mismatch_weighted / mismatch_support
    )

    fallback_lookup = {reason: 0 for reason in FALLBACK_REASONS}
    for record in validated:
        reason = record.fallback_reason
        if reason is not None:
            fallback_lookup[reason] += 1
    fallback_total = sum(fallback_lookup.values())

    elapsed_values = [int(record.elapsed_ns) for record in validated]
    candidate_counts = [int(record.candidate_count) for record in validated]

    return BayesianEvidenceReport(
        artifact_digest=str(artifact.manifest.aggregate_digest),
        seed=int(config.seed),
        rounds=int(config.rounds),
        node_count=int(topology_node_count(config.topology)),
        compromised_node=None if compromised is None else int(compromised),
        attack_kappa=kappa,
        selection_count=len(validated),
        benign_receiver_count=sum(
            1 for record in validated if not is_malicious(record.node_id)
        ),
        neighbor_observation_count=coverage_total,
        covered_observation_count=coverage_covered,
        coverage=_bounded_unit(coverage),
        mismatch=_bounded_unit(mismatch),
        mismatch_group_count=mismatch_groups,
        fallback_counts=tuple(
            (reason, fallback_lookup[reason]) for reason in FALLBACK_REASONS
        ),
        fallback_total=fallback_total,
        fallback_rate=fallback_total / len(validated),
        pruned_count=sum(1 for record in validated if record.pruned),
        candidate_count_total=sum(candidate_counts),
        candidate_count_max=max(candidate_counts),
        elapsed_ns_total=sum(elapsed_values),
        elapsed_ns_min=min(elapsed_values),
        elapsed_ns_max=max(elapsed_values),
    )


def _bounded_unit(value: float) -> float:
    """Return ``value`` clamped to the closed unit interval (finite by construction)."""

    if not math.isfinite(value):
        raise EvidenceError(f"aggregate {value!r} is not finite")
    return min(1.0, max(0.0, float(value)))
