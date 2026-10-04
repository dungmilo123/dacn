"""Finite weight profiles and baseline aggregation policies (R005/R006/R007/D008/D017).

This module closes the second half of the S03 contract:

* :class:`WeightProfile` is the immutable, participant-aligned action record
  returned by every policy. It carries the finite multiplier action
  (``multipliers``) and the normalized aggregation weights (``weights``) in the
  exact canonical participant order supplied by a
  :class:`~dfl_sim.observation.LocalObservation`.
* :func:`build_weight_profile` is the policy-factory-independent finite action
  helper: it validates participant ids and multipliers and folds a multiplier
  vector into a normalized profile *without* enumerating the ``3**degree``
  joint action space (D017).
* :class:`AggregationPolicy` is a runtime-checkable protocol exposing only
  ``select_weights(observation)``. Its sole argument is a frozen
  :class:`~dfl_sim.observation.LocalObservation`, so a policy structurally
  cannot receive model tensors, datasets, topology, attack identity, peer
  observations, evaluator state, or global metrics (R005/D012).
* :class:`LocalOnlyPolicy`, :class:`PlainAveragePolicy`, and
  :class:`FixedSimilarityFilterPolicy` are the three M001 baselines (D008) over
  that single seam.
* :class:`BayesianPolicy` (R015/R016/D022/D023) is M002's exact
  approximate-myopic Bayesian best response. It consumes only a loaded frozen
  artifact, the immutable total-round context fixed at construction, and each
  scalar :class:`~dfl_sim.observation.LocalObservation`; it derives the
  posterior over artifact-declared feasible joint types, computes the
  posterior-weighted expected payoff of every bounded D022 candidate action,
  and selects the first maximum under the artifact tie tolerance. Every
  selection emits one structured :class:`BayesianDiagnostic`; unsupported
  cells, absent context, or zero mass fall back to exact self-only weights with
  an explicit reason.

Every built-in policy output is treated as untrusted boundary data even though
it is produced in-process: :func:`build_weight_profile` validates the finite
self multiplier (exactly ``1``), finite non-negative multipliers, optional
D017 membership in ``{0, omega, 1}`` for neighbors, and normalization before a
profile exists. Empty and all-rejected neighborhoods are *not* errors and
resolve to exactly self-only weights ``(1.0, 0.0, ...)``/``(1.0,)``.
"""

from __future__ import annotations

import itertools
import logging
import math
import numbers
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .bayesian_artifact import (
    BayesianArtifact,
    interval_bin_index,
    load_bayesian_artifact,
)
from .config import (
    AggregationConfig,
    BalanceAggregation,
    BayesianAggregation,
    FixedSimilarityAggregation,
    LocalOnlyAggregation,
    PlainAverageAggregation,
    TrustWeightedAggregation,
    ValidationReferenceAggregation,
)
from .observation import LocalObservation

__all__ = [
    "AggregationPolicy",
    "BalancePolicy",
    "BayesianDiagnostic",
    "BayesianPolicy",
    "FALLBACK_CONTEXT_UNAVAILABLE",
    "FALLBACK_MISSING_LIKELIHOOD",
    "FALLBACK_MISSING_PAYOFF",
    "FALLBACK_MISSING_PRIOR",
    "FALLBACK_NON_FINITE",
    "FALLBACK_ZERO_MASS",
    "FixedSimilarityFilterPolicy",
    "LocalOnlyPolicy",
    "PlainAveragePolicy",
    "PolicyError",
    "PosteriorEntry",
    "TrustWeightedPolicy",
    "ValidationReferencePolicy",
    "WeightProfile",
    "build_bayesian_policy",
    "build_policy",
    "build_weight_profile",
    "enumerate_candidate_actions",
]

#: Module logger for the structured Bayesian selection diagnostics. One record
#: is emitted per selection; the payload is attached under the
#: ``bayesian_diagnostic`` extra so a host can capture it without parsing the
#: human-readable message. The diagnostics carry only policy-visible fields
#: (node/round, chosen multipliers, candidate count, pruning flag, a bounded
#: posterior summary, and an explicit fallback reason).
_LOGGER = logging.getLogger("dfl_sim.policy")

#: Fallback reason codes: the only ways a Bayesian selection may fall back to
#: exact self-only weights. Each maps to a documented, non-privileged cause.
FALLBACK_CONTEXT_UNAVAILABLE = "context_unavailable"
FALLBACK_MISSING_PRIOR = "missing_prior"
FALLBACK_MISSING_LIKELIHOOD = "missing_likelihood"
FALLBACK_MISSING_PAYOFF = "missing_payoff"
FALLBACK_ZERO_MASS = "zero_mass"
FALLBACK_NON_FINITE = "non_finite"

#: Degree label grammar: an exact integer, an inclusive ``a-b`` range, or an
#: open-ended ``N+`` range. The labels are ordered and the first match wins.
_DEGREE_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_DEGREE_PLUS_RE = re.compile(r"^(\d+)\s*\+$")

#: Absolute tolerance for the unit-sum and multiplier-normalization invariants.
_WEIGHT_SUM_TOLERANCE = 1e-9


class PolicyError(RuntimeError):
    """Raised when a policy action or returned weight profile is malformed."""


def _require_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyError(f"{label} must be int, got {type(value).__name__}")
    return int(value)


def _require_finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise PolicyError(
            f"{label} must be a real number, got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise PolicyError(f"{label} must be finite, got {result!r}")
    return result


def _suffix(context: str) -> str:
    return f" ({context})" if context else ""


def _normalize_participant_ids(
    participant_ids: object, *, context: str = ""
) -> tuple[int, ...]:
    """Validate participant ids into a non-empty, duplicate-free int tuple."""

    suffix = _suffix(context)
    if isinstance(participant_ids, (str, bytes)) or not isinstance(
        participant_ids, Sequence
    ):
        raise PolicyError(
            "participant_ids must be a sequence of ints"
            f"{suffix}, got {type(participant_ids).__name__}"
        )
    if not participant_ids:
        raise PolicyError(f"participant_ids must be non-empty{suffix}")

    seen: set[int] = set()
    ids: list[int] = []
    for raw in participant_ids:
        value = _require_int(raw, label="participant_id")
        if value in seen:
            raise PolicyError(f"duplicate participant id {value}{suffix}")
        seen.add(value)
        ids.append(value)
    return tuple(ids)


def _normalize_multipliers(
    multipliers: object, *, count: int, context: str = ""
) -> tuple[float, ...]:
    """Validate a finite, non-negative multiplier vector of the expected length."""

    suffix = _suffix(context)
    if isinstance(multipliers, (str, bytes)) or not isinstance(
        multipliers, Sequence
    ):
        raise PolicyError(
            "multipliers must be a sequence of reals"
            f"{suffix}, got {type(multipliers).__name__}"
        )
    if len(multipliers) != count:
        raise PolicyError(
            f"multiplier count {len(multipliers)} does not match participant "
            f"count {count}{suffix}"
        )
    values: list[float] = []
    for raw in multipliers:
        value = _require_finite_float(raw, label="multiplier")
        if value < 0.0:
            raise PolicyError(
                f"multiplier must be non-negative, got {value!r}{suffix}"
            )
        values.append(value)
    return tuple(values)


def _normalize_weights(
    weights: object, *, count: int, context: str = ""
) -> tuple[float, ...]:
    """Validate a finite, non-negative, unit-sum weight vector."""

    suffix = _suffix(context)
    if isinstance(weights, (str, bytes)) or not isinstance(weights, Sequence):
        raise PolicyError(
            "weights must be a sequence of reals"
            f"{suffix}, got {type(weights).__name__}"
        )
    if len(weights) != count:
        raise PolicyError(
            f"weight count {len(weights)} does not match participant "
            f"count {count}{suffix}"
        )
    values: list[float] = []
    for raw in weights:
        value = _require_finite_float(raw, label="weight")
        if value < 0.0:
            raise PolicyError(
                f"weight must be non-negative, got {value!r}{suffix}"
            )
        values.append(value)
    total = math.fsum(values)
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise PolicyError(f"weights must sum to 1, got {total!r}{suffix}")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class WeightProfile:
    """Immutable participant-aligned finite action and weight profile (D017).

    ``participant_ids`` is the canonical order (node first, then received
    senders ascending) exactly as exposed by
    :attr:`~dfl_sim.observation.LocalObservation.participant_ids`.
    ``multipliers`` is the finite action (self fixed at ``1``; neighbors in
    ``{0, omega, 1}`` when validated through :func:`build_weight_profile` with
    an ``omega``). ``weights`` is the normalized non-negative unit-sum vector
    actually consumed by weighted aggregation, and must equal the normalization
    of ``multipliers`` so the two representations can never disagree.
    """

    participant_ids: tuple[int, ...]
    multipliers: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        ids = _normalize_participant_ids(self.participant_ids)
        count = len(ids)
        multipliers = _normalize_multipliers(self.multipliers, count=count)
        weights = _normalize_weights(self.weights, count=count)

        if multipliers[0] != 1.0:
            raise PolicyError(
                "self multiplier must be exactly 1, got "
                f"{multipliers[0]!r} (node={ids[0]})"
            )
        total = math.fsum(multipliers)
        if total <= 0.0:
            raise PolicyError(
                f"multipliers must sum to a positive value (node={ids[0]})"
            )
        expected = tuple(value / total for value in multipliers)
        for participant, got, want in zip(ids, weights, expected):
            if abs(got - want) > _WEIGHT_SUM_TOLERANCE:
                raise PolicyError(
                    "weights must equal the normalization of multipliers: "
                    f"participant {participant} has weight {got!r}, expected "
                    f"{want!r}"
                )

        object.__setattr__(self, "participant_ids", ids)
        object.__setattr__(self, "multipliers", multipliers)
        object.__setattr__(self, "weights", weights)

    @property
    def node_id(self) -> int:
        """The observing node: the first canonical participant."""

        return self.participant_ids[0]

    @property
    def degree(self) -> int:
        """Number of neighbor participants (excluding self)."""

        return len(self.participant_ids) - 1

    def weight_for(self, participant_id: int) -> float:
        """Return the aligned weight for ``participant_id`` (raises if absent)."""

        try:
            index = self.participant_ids.index(participant_id)
        except ValueError as exc:
            raise PolicyError(
                f"participant {participant_id} is not in this profile "
                f"(participants={self.participant_ids})"
            ) from exc
        return self.weights[index]


def build_weight_profile(
    participant_ids: Sequence[int],
    multipliers: Sequence[float],
    *,
    omega: float | None = None,
    context: str = "",
) -> WeightProfile:
    """Policy-factory-independent finite action helper (D017).

    Validates participant ids and the finite multiplier action, then normalizes
    once in canonical participant order. The self multiplier must be exactly
    ``1``. When ``omega`` is supplied it must satisfy ``0 < omega < 1`` and
    every neighbor multiplier must be exactly ``0``, ``omega``, or ``1``.

    This is a closed-form fold over the action vector: it never enumerates the
    ``3**degree`` joint action space (Q6 load profile).
    """

    suffix = _suffix(context)
    ids = _normalize_participant_ids(participant_ids, context=context)
    mults = _normalize_multipliers(
        multipliers, count=len(ids), context=context
    )

    omega_value: float | None = None
    if omega is not None:
        omega_value = _require_finite_float(omega, label="omega")
        if not 0.0 < omega_value < 1.0:
            raise PolicyError(
                f"omega must be in (0, 1), got {omega_value!r}{suffix}"
            )

    if mults[0] != 1.0:
        raise PolicyError(
            f"self multiplier must be exactly 1, got {mults[0]!r}{suffix}"
        )
    if omega_value is not None:
        allowed = (0.0, omega_value, 1.0)
        for sender, value in zip(ids[1:], mults[1:]):
            if not any(value == candidate for candidate in allowed):
                raise PolicyError(
                    f"neighbor multiplier {value!r} for sender {sender} is not "
                    f"one of {{0, {omega_value!r}, 1}}{suffix}"
                )

    total = math.fsum(mults)
    if total <= 0.0:
        raise PolicyError(
            f"multipliers must sum to a positive value{suffix}"
        )
    weights = tuple(value / total for value in mults)
    return WeightProfile(participant_ids=ids, multipliers=mults, weights=weights)


def _require_observation(
    observation: object, *, policy_name: str
) -> LocalObservation:
    """Reject anything that is not the frozen local observation record (R005).

    The check is *exact*. A duck-typed lookalike and an enriched subclass (which
    reintroduces a mutable ``__dict__`` and could carry unused privileged
    sentinels) are both refused, so the online policy seam cannot be widened by
    an observation-shaped object from outside ``dfl_sim.observation``.
    """

    if type(observation) is not LocalObservation:
        raise PolicyError(
            f"{policy_name}.select_weights requires a LocalObservation, got "
            f"{type(observation).__name__}"
        )
    return observation


def _observation_context(observation: LocalObservation) -> str:
    return f"node={observation.node_id} round={observation.round_index}"


@runtime_checkable
class AggregationPolicy(Protocol):
    """Narrow aggregation seam: ``select_weights(observation) -> profile`` (R007/D008).

    The protocol exposes only this method and accepts only a
    :class:`~dfl_sim.observation.LocalObservation`, so any policy selected by
    the round engine structurally cannot read privileged evaluator or
    network-wide state (R005/D012).
    """

    def select_weights(self, observation: LocalObservation) -> WeightProfile:
        """Select a participant-aligned weight profile for one node/round."""

        ...


@dataclass(frozen=True, slots=True)
class LocalOnlyPolicy:
    """Baseline: ignore all neighbors, aggregate the locally trained model (D008)."""

    def select_weights(self, observation: LocalObservation) -> WeightProfile:
        obs = _require_observation(observation, policy_name="LocalOnlyPolicy")
        multipliers = (1.0,) + (0.0,) * obs.degree
        return build_weight_profile(
            obs.participant_ids,
            multipliers,
            context=f"LocalOnlyPolicy {_observation_context(obs)}",
        )


@dataclass(frozen=True, slots=True)
class PlainAveragePolicy:
    """Baseline: uniform mean of self plus every received neighbor (D008/D015)."""

    def select_weights(self, observation: LocalObservation) -> WeightProfile:
        obs = _require_observation(observation, policy_name="PlainAveragePolicy")
        # D017 uniform reference profile: every participant multiplier is 1.
        multipliers = (1.0,) * len(obs.participant_ids)
        return build_weight_profile(
            obs.participant_ids,
            multipliers,
            context=f"PlainAveragePolicy {_observation_context(obs)}",
        )


@dataclass(frozen=True, slots=True)
class FixedSimilarityFilterPolicy:
    """Baseline: keep only neighbors whose directional disagreement is <= cutoff (D008).

    A neighbor is accepted exactly when
    ``neighbor.directional_disagreement <= cutoff`` (inclusive, deterministic
    equality) and rejected otherwise. Rejection is a documented non-error
    fallback: an all-rejected or empty neighborhood resolves to exactly
    self-only weights ``(1.0, 0.0, ...)``.

    ``downweight_multiplier`` is the D017 reference ``omega`` in ``(0, 1)``.
    The M001 fixed filter emits binary multipliers (``1`` accept / ``0``
    reject); when ``omega`` is configured it is passed through to
    :func:`build_weight_profile` so every emitted neighbor multiplier is
    checked against the finite action set ``{0, omega, 1}``.
    """

    cutoff: float = 1.0
    downweight_multiplier: float | None = None

    def __post_init__(self) -> None:
        cutoff = _require_finite_float(self.cutoff, label="cutoff")
        if not 0.0 <= cutoff <= 2.0:
            raise PolicyError(
                f"cutoff must be within [0, 2], got {cutoff!r}"
            )
        downweight = self.downweight_multiplier
        if downweight is not None:
            downweight = _require_finite_float(
                downweight, label="downweight_multiplier"
            )
            if not 0.0 < downweight < 1.0:
                raise PolicyError(
                    "downweight_multiplier must be within (0, 1), got "
                    f"{downweight!r}"
                )
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "downweight_multiplier", downweight)

    def select_weights(self, observation: LocalObservation) -> WeightProfile:
        obs = _require_observation(
            observation, policy_name="FixedSimilarityFilterPolicy"
        )
        multipliers = (1.0,) + tuple(
            1.0 if neighbor.directional_disagreement <= self.cutoff else 0.0
            for neighbor in obs.neighbors
        )
        return build_weight_profile(
            obs.participant_ids,
            multipliers,
            omega=self.downweight_multiplier,
            context=f"FixedSimilarityFilterPolicy {_observation_context(obs)}",
        )


@dataclass(frozen=True, slots=True)
class ValidationReferencePolicy:
    """Baseline: keep neighbors whose delivered loss delta is <= tolerance (D026).

    A neighbor is accepted exactly when ``neighbor.loss_delta <= tolerance``
    (inclusive, deterministic equality) and rejected otherwise. ``tolerance``
    must be finite and non-negative; it is validated once at construction so a
    malformed value fails before any round. This is the simple documented
    validation-loss reference comparator for the fixed-poison acceptance: it
    shares the exact online authority of the other built-ins -- only the
    frozen scalar :class:`~dfl_sim.observation.LocalObservation` seam -- and
    never inspects evaluator labels, empirical counters, or timing.

    Rejection is a documented non-error fallback: an all-rejected or empty
    neighborhood resolves to exactly self-only weights ``(1.0, 0.0, ...)`` /
    ``(1.0,)`` via the shared :func:`build_weight_profile` normalization, so
    the reference can never silently substitute plain averaging.
    """

    tolerance: float = 0.0

    def __post_init__(self) -> None:
        tolerance = _require_finite_float(self.tolerance, label="tolerance")
        if tolerance < 0.0:
            raise PolicyError(
                f"tolerance must be non-negative, got {tolerance!r}"
            )
        object.__setattr__(self, "tolerance", tolerance)

    def select_weights(self, observation: LocalObservation) -> WeightProfile:
        obs = _require_observation(
            observation, policy_name="ValidationReferencePolicy"
        )
        multipliers = (1.0,) + tuple(
            1.0 if neighbor.loss_delta <= self.tolerance else 0.0
            for neighbor in obs.neighbors
        )
        return build_weight_profile(
            obs.participant_ids,
            multipliers,
            context=f"ValidationReferencePolicy {_observation_context(obs)}",
        )


@dataclass(frozen=True, slots=True)
class TrustWeightedPolicy:
    """D029 local validation-loss trust comparator over the shared seam.

    The local model keeps multiplier ``1`` and each delivered neighbor ``j`` is
    scaled by ``exp(-beta * max(loss_delta_j, 0))``. An improving or equal
    neighbor (``loss_delta <= 0``) therefore stays at multiplier ``1`` (equal to
    plain averaging), while a strictly worse neighbor is continuously
    downweighted toward, but never reaching, ``0``. ``beta`` is a strict
    positive finite rate validated once at construction; an empty neighborhood
    resolves to exact self-only weights. Every emitted multiplier is folded
    through :func:`build_weight_profile`, so the participant alignment and
    unit-sum invariants are centralized and the local-information boundary is
    unchanged (only the frozen scalar observation is read).
    """

    beta: float = 1.0

    def __post_init__(self) -> None:
        beta = _require_finite_float(self.beta, label="beta")
        if beta <= 0.0:
            raise PolicyError(f"beta must be positive, got {beta!r}")
        object.__setattr__(self, "beta", beta)

    def select_weights(self, observation: LocalObservation) -> WeightProfile:
        obs = _require_observation(
            observation, policy_name="TrustWeightedPolicy"
        )
        multipliers = (1.0,) + tuple(
            math.exp(-self.beta * max(neighbor.loss_delta, 0.0))
            for neighbor in obs.neighbors
        )
        return build_weight_profile(
            obs.participant_ids,
            multipliers,
            context=f"TrustWeightedPolicy {_observation_context(obs)}",
        )


@dataclass(frozen=True, slots=True)
class BalancePolicy:
    """D028 faithful BALANCE aggregation (Fang et al., ACM CCS 2024 Algorithm 3).

    A delivered neighbor ``j`` is accepted exactly when its scalar
    ``relative_norm_j <= gamma * exp(-kappa * round_index / total_rounds)``;
    the threshold is inclusive and decays geometrically over the fixed episode
    length. The accepted set is then mixed with exact self weight ``alpha`` and
    uniform ``(1 - alpha) / accepted_count`` neighbor weights, encoded as
    multipliers through :func:`build_weight_profile` so the returned weights are
    normalized and participant-aligned. An empty or all-rejected neighborhood
    resolves to exact self-only weights ``(1.0, 0.0, ...)`` -- the BALANCE
    fallback -- regardless of ``alpha``.

    ``total_rounds`` is the validated episode round count fixed at construction
    (the same D023 pattern as :class:`BayesianPolicy`), so ``LocalObservation``
    stays unchanged. ``alpha`` in ``(0, 1)``, ``gamma > 0``, and ``kappa >= 0``
    are strict finite values validated once; a malformed or impossible input
    fails closed before any round.
    """

    total_rounds: int
    alpha: float = 0.5
    gamma: float = 0.3
    kappa: float = 1.0

    def __post_init__(self) -> None:
        rounds = _require_int(self.total_rounds, label="total_rounds")
        if rounds < 1:
            raise PolicyError(
                f"total_rounds must be >= 1, got {rounds!r}"
            )
        alpha = _require_finite_float(self.alpha, label="alpha")
        if not 0.0 < alpha < 1.0:
            raise PolicyError(f"alpha must be in (0, 1), got {alpha!r}")
        gamma = _require_finite_float(self.gamma, label="gamma")
        if gamma <= 0.0:
            raise PolicyError(f"gamma must be positive, got {gamma!r}")
        decay = _require_finite_float(self.kappa, label="kappa")
        if decay < 0.0:
            raise PolicyError(f"kappa must be non-negative, got {decay!r}")
        object.__setattr__(self, "total_rounds", rounds)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "kappa", decay)

    def threshold(self, round_index: int) -> float:
        """Return the BALANCE acceptance threshold at ``round_index`` (D028)."""

        return self.gamma * math.exp(
            -self.kappa * round_index / self.total_rounds
        )

    def select_weights(self, observation: LocalObservation) -> WeightProfile:
        obs = _require_observation(observation, policy_name="BalancePolicy")
        threshold = self.threshold(obs.round_index)
        accepted = [
            neighbor.relative_norm <= threshold
            for neighbor in obs.neighbors
        ]
        count = sum(1 for keep in accepted if keep)
        if count == 0:
            multipliers = (1.0,) + (0.0,) * obs.degree
        else:
            neighbor_multiplier = (1.0 - self.alpha) / (self.alpha * count)
            multipliers = (1.0,) + tuple(
                neighbor_multiplier if keep else 0.0 for keep in accepted
            )
        return build_weight_profile(
            obs.participant_ids,
            multipliers,
            context=f"BalancePolicy {_observation_context(obs)}",
        )


def _resolve_degree_bin(
    labels: Sequence[str], degree: int
) -> str | None:
    """Resolve ``degree`` to one declared categorical label or ``None``.

    Labels are ordered and the first match wins. Supported grammars: an exact
    integer (``"2"``), an inclusive range (``"3-4"``), or an open-ended range
    (``"5+"``). Labels that match none of these -- or a degree no label covers
    -- resolve to ``None`` so the caller falls back honestly rather than
    guessing a bin (R016).
    """

    for label in labels:
        candidate = label.strip()
        if candidate.isdigit():
            if int(candidate) == degree:
                return label
            continue
        match = _DEGREE_RANGE_RE.match(candidate)
        if match is not None:
            if int(match.group(1)) <= degree <= int(match.group(2)):
                return label
            continue
        match = _DEGREE_PLUS_RE.match(candidate)
        if match is not None and degree >= int(match.group(1)):
            return label
    return None


def _resolve_progress_bin(
    labels: Sequence[str], round_index: int, total_rounds: int
) -> str | None:
    """Resolve round progress to one declared progress label or ``None``.

    The ordered categorical labels are spread evenly over the fixed run length:
    ``index = (round_index * len(labels)) // total_rounds``. A round index
    outside ``[0, total_rounds)`` resolves to ``None`` (absent context).
    """

    if total_rounds < 1 or not labels:
        return None
    if round_index < 0 or round_index >= total_rounds:
        return None
    count = len(labels)
    index = (round_index * count) // total_rounds
    if index >= count:
        index = count - 1
    return labels[index]


def _neighbor_evidence(
    artifact: BayesianArtifact, neighbor: object
) -> tuple[int, int, int]:
    """Bin one neighbor's scalar feature triple into the artifact's bins."""

    manifest = artifact.manifest
    disagreement = float(getattr(neighbor, "directional_disagreement"))
    relative_norm = float(getattr(neighbor, "relative_norm"))
    loss_delta = float(getattr(neighbor, "loss_delta"))
    return (
        interval_bin_index(manifest.bins.disagreement, disagreement),
        interval_bin_index(manifest.bins.relative_norm, relative_norm),
        interval_bin_index(manifest.bins.loss_delta, loss_delta),
    )


def _finite_product(factors: Sequence[float]) -> float | None:
    """Return the product of finite non-negative ``factors`` or ``None``."""

    product = 1.0
    for factor in factors:
        if not math.isfinite(factor) or factor < 0.0:
            return None
        product *= factor
        if not math.isfinite(product):
            return None
    return product


def _action_sort_key(action: tuple[float, ...]) -> tuple[int, tuple[float, ...]]:
    """Canonical D022 action order: fewer zero multipliers, then ascending."""

    rejected = 0
    for multiplier in action:
        if multiplier == 0.0:
            rejected += 1
    return (rejected, action)


def _select_variable_positions(
    disagreements: Sequence[float],
    sender_ids: Sequence[int],
    degree_cap: int,
) -> tuple[int, ...]:
    """Rank neighbor positions by D022 then keep the first ``degree_cap``.

    Neighbors are ranked by descending directional disagreement with sender id
    ascending as the deterministic tie-break; only the kept positions are
    varied, and the returned positions are sorted ascending so the variable
    set (not the assignment order) is canonical.
    """

    order = sorted(
        range(len(disagreements)),
        key=lambda index: (-disagreements[index], sender_ids[index]),
    )
    return tuple(sorted(order[:degree_cap]))


def enumerate_candidate_actions(
    degree: int,
    multiplier_set: Sequence[float],
    *,
    degree_cap: int | None = None,
    disagreements: Sequence[float] | None = None,
    sender_ids: Sequence[int] | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Enumerate the ordered, deterministic finite action candidates (D022/Q6).

    For ``degree <= degree_cap`` every neighbor multiplier vector in
    ``multiplier_set ** degree`` is enumerated. Above the cap only the first
    ``degree_cap`` neighbors -- ranked by descending directional disagreement
    then ascending sender id -- are varied; every other neighbor is forced to
    ``1``. The returning order is always canonical (fewer zero multipliers
    first, then lexicographic ascending), so candidate count is exactly
    ``len(multiplier_set) ** degree`` inside the cap and at most
    ``len(multiplier_set) ** degree_cap`` above it.
    """

    degree = _require_int(degree, label="degree")
    if degree < 0:
        raise PolicyError(f"degree must be non-negative, got {degree!r}")

    if isinstance(multiplier_set, (str, bytes)) or not isinstance(
        multiplier_set, Sequence
    ):
        raise PolicyError("multiplier_set must be a sequence of reals")
    multipliers = tuple(
        sorted(
            _require_finite_float(value, label="multiplier")
            for value in multiplier_set
        )
    )
    if not multipliers:
        raise PolicyError("multiplier_set must be non-empty")
    if len(set(multipliers)) != len(multipliers):
        raise PolicyError("multiplier_set must not contain duplicates")

    if degree == 0:
        return ((),)

    if degree_cap is not None:
        cap = _require_int(degree_cap, label="degree_cap")
        if cap < 1:
            raise PolicyError(f"degree_cap must be >= 1, got {cap!r}")
    else:
        cap = degree

    if degree > cap:
        if disagreements is None or sender_ids is None:
            raise PolicyError(
                "pruned enumeration requires disagreements and sender_ids"
            )
        if len(disagreements) != degree or len(sender_ids) != degree:
            raise PolicyError(
                "disagreements/sender_ids length must match degree"
            )
        positions = _select_variable_positions(
            tuple(float(value) for value in disagreements),
            tuple(_require_int(value, label="sender_id") for value in sender_ids),
            cap,
        )
        actions: list[tuple[float, ...]] = []
        for combination in itertools.product(multipliers, repeat=len(positions)):
            vector = [1.0] * degree
            for position, multiplier in zip(positions, combination):
                vector[position] = multiplier
            actions.append(tuple(vector))
        return tuple(sorted(actions, key=_action_sort_key))

    enumerated = itertools.product(multipliers, repeat=degree)
    return tuple(sorted((tuple(action) for action in enumerated), key=_action_sort_key))


def _compute_posterior(
    artifact: BayesianArtifact,
    *,
    degree_bin: str,
    progress: str,
    evidence_bins: tuple[tuple[int, int, int], ...],
) -> tuple[tuple["PosteriorEntry", ...] | None, float | None, str | None]:
    """Compute ``posterior(theta) proportional to prior * product_j L``.

    Returns ``(posteriors, mass, reason)``: on success ``reason`` is ``None``;
    on any missing likelihood, non-finite factor, or zero normalization mass the
    posteriors are ``None`` and ``reason`` names the fallback cause.
    """

    cells = artifact.prior_for_degree(len(evidence_bins))
    if not cells:
        return None, None, FALLBACK_MISSING_PRIOR

    weights: list[float] = []
    for cell in cells:
        factors: list[float] = []
        for theta_j, evidence in zip(cell.state, evidence_bins):
            likelihood = artifact.likelihood_for(
                degree_bin, progress, theta_j, evidence
            )
            if likelihood is None:
                return None, None, FALLBACK_MISSING_LIKELIHOOD
            factors.append(likelihood)
        product = _finite_product(factors)
        if product is None:
            return None, None, FALLBACK_NON_FINITE
        weight = cell.probability * product
        if not math.isfinite(weight):
            return None, None, FALLBACK_NON_FINITE
        weights.append(weight)

    try:
        mass = math.fsum(weights)
    except OverflowError:
        return None, None, FALLBACK_NON_FINITE
    if not math.isfinite(mass):
        return None, None, FALLBACK_NON_FINITE
    if mass <= 0.0:
        return None, None, FALLBACK_ZERO_MASS

    posteriors = tuple(
        PosteriorEntry(state=cell.state, probability=weight / mass)
        for cell, weight in zip(cells, weights)
    )
    return posteriors, mass, None


def _expected_payoffs(
    artifact: BayesianArtifact,
    *,
    degree_bin: str,
    progress: str,
    evidence_bins: tuple[tuple[int, int, int], ...],
    posteriors: tuple["PosteriorEntry", ...],
    actions: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...] | None, str | None]:
    """Compute ``E[Q] = sum_theta posterior(theta) * sum_j q(x_j, c, d_j, theta_j)``.

    Returns ``(values, reason)`` where ``reason`` is ``None`` on success or a
    fallback cause for a missing payoff cell or a non-finite sum.
    """

    values: list[float] = []
    for action in actions:
        expected = 0.0
        for entry in posteriors:
            terms: list[float] = []
            for theta_j, evidence, multiplier in zip(
                entry.state, evidence_bins, action
            ):
                value = artifact.payoff_for(
                    degree_bin, progress, theta_j, evidence, multiplier
                )
                if value is None:
                    return None, FALLBACK_MISSING_PAYOFF
                terms.append(value)
            try:
                total = math.fsum(terms)
            except OverflowError:
                return None, FALLBACK_NON_FINITE
            if not math.isfinite(total):
                return None, FALLBACK_NON_FINITE
            expected += entry.probability * total
            if not math.isfinite(expected):
                return None, FALLBACK_NON_FINITE
        values.append(expected)
    return tuple(values), None


def _first_argmax(values: Sequence[float], tolerance: float) -> int:
    """Return the first index whose value is within ``tolerance`` of the max."""

    best_index = 0
    best_value = values[0]
    for index in range(1, len(values)):
        value = values[index]
        if value > best_value + tolerance:
            best_value = value
            best_index = index
    return best_index


@dataclass(frozen=True, slots=True)
class PosteriorEntry:
    """One bounded posterior-summary row: a feasible joint state and mass."""

    state: tuple[int, ...]
    probability: float


@dataclass(frozen=True, slots=True)
class BayesianDiagnostic:
    """Structured, policy-visible record of one Bayesian selection (R015/R016).

    Carries only node/round context, the chosen finite multipliers, the
    candidate count and whether D022 pruning was applied, a bounded posterior
    summary over feasible joint states, the normalization mass, the canonical
    per-neighbor sender/evidence alignment, and an integer elapsed selection
    time. It never contains true attacker types, evaluator state, topology,
    tensors, or global metrics.

    ``sender_ids`` mirrors :attr:`~dfl_sim.observation.LocalObservation.neighbor_ids`
    (strictly ascending, exactly ``degree`` entries) and ``evidence_bins`` is
    the aligned per-neighbor artifact evidence cell. ``elapsed_ns`` is measured
    around the selection but never enters any arithmetic or tie-breaking, so an
    episode's chosen profiles are timing-independent. All three are bounded by
    the observation's degree; the offline evidence collector consumes them
    without widening the online seam.
    """

    node_id: int
    round_index: int
    degree: int
    degree_bin: str | None
    progress: str | None
    chosen_multipliers: tuple[float, ...]
    candidate_count: int
    pruned: bool
    posterior: tuple[PosteriorEntry, ...]
    fallback_reason: str | None
    normalization_mass: float | None
    sender_ids: tuple[int, ...] = ()
    evidence_bins: tuple[tuple[int, int, int], ...] = ()
    elapsed_ns: int = 0

    def __post_init__(self) -> None:
        degree = _require_int(self.degree, label="degree")
        if degree < 0:
            raise PolicyError(f"degree must be non-negative, got {degree!r}")

        senders = self.sender_ids
        if not isinstance(senders, tuple) or len(senders) != degree:
            raise PolicyError(
                "sender_ids must be a tuple aligned with degree "
                f"(degree={degree}, senders={senders!r})"
            )
        previous: int | None = None
        for sender in senders:
            value = _require_int(sender, label="sender_id")
            if value < 0:
                raise PolicyError(f"sender_id must be non-negative, got {value!r}")
            if previous is not None and value <= previous:
                raise PolicyError("sender_ids must be strictly ascending")
            previous = value

        evidence = self.evidence_bins
        if not isinstance(evidence, tuple) or len(evidence) != degree:
            raise PolicyError(
                "evidence_bins must be a tuple aligned with degree "
                f"(degree={degree}, bins={evidence!r})"
            )
        for row in evidence:
            if (
                not isinstance(row, tuple)
                or len(row) != 3
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in row
                )
            ):
                raise PolicyError(
                    f"evidence_bins rows must be 3-tuples of ints, got {row!r}"
                )
            if any(value < 0 for value in row):
                raise PolicyError(
                    f"evidence_bins rows must be non-negative, got {row!r}"
                )

        elapsed = _require_int(self.elapsed_ns, label="elapsed_ns")
        if elapsed < 0:
            raise PolicyError(f"elapsed_ns must be non-negative, got {elapsed!r}")

        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "elapsed_ns", elapsed)


@dataclass(frozen=True, slots=True)
class BayesianPolicy:
    """Exact approximate-myopic Bayesian best response over the local seam (R015).

    Consumes only the frozen artifact, the immutable total-round context fixed
    at construction, and each frozen scalar
    :class:`~dfl_sim.observation.LocalObservation`. It returns an ordinary
    :class:`WeightProfile` built through :func:`build_weight_profile`, so self
    stays multiplier ``1`` and participant alignment/unit-sum invariants remain
    centralized. Empty neighborhoods resolve normally to self-only; missing
    cells, absent context, or zero posterior mass fall back to exact self-only
    weights with one structured diagnostic.
    """

    artifact: BayesianArtifact
    total_rounds: int
    diagnostic_sink: Callable[[BayesianDiagnostic], None] | None = None
    clock: Callable[[], int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, BayesianArtifact):
            raise PolicyError(
                "BayesianPolicy requires a loaded BayesianArtifact, got "
                f"{type(self.artifact).__name__}"
            )
        rounds = _require_int(self.total_rounds, label="total_rounds")
        if rounds < 1:
            raise PolicyError(
                f"total_rounds must be >= 1, got {rounds!r}"
            )
        sink = self.diagnostic_sink
        if sink is not None and not callable(sink):
            raise PolicyError("diagnostic_sink must be callable")
        clock = self.clock
        if clock is not None and not callable(clock):
            raise PolicyError("clock must be callable")
        if clock is None:
            clock = time.perf_counter_ns
        object.__setattr__(self, "total_rounds", rounds)
        object.__setattr__(self, "clock", clock)

    def _elapsed_ns(self, start: int) -> int:
        """Return the integer nanoseconds elapsed since ``start`` (timing only)."""

        clock = self.clock
        assert clock is not None  # set in __post_init__
        return int(clock()) - int(start)

    def _emit(self, diagnostic: BayesianDiagnostic) -> None:
        sink = self.diagnostic_sink
        if sink is not None:
            try:
                sink(diagnostic)
            except Exception as exc:
                # A reporter failure must never yield a silently incomplete
                # diagnostics stream: fail closed before the caller can treat
                # the already-selected profile as fully accounted for.
                raise PolicyError(
                    f"diagnostic sink failed for node={diagnostic.node_id} "
                    f"round={diagnostic.round_index}: {exc}"
                ) from exc
        _LOGGER.debug(
            "bayesian selection node=%s round=%s fallback=%s",
            diagnostic.node_id,
            diagnostic.round_index,
            diagnostic.fallback_reason,
            extra={"bayesian_diagnostic": diagnostic},
        )

    def _fallback(
        self,
        observation: LocalObservation,
        *,
        context: str,
        reason: str,
        degree_bin: str | None,
        progress: str | None,
        candidate_count: int,
        pruned: bool,
        posterior: tuple[PosteriorEntry, ...],
        mass: float | None,
        evidence_bins: tuple[tuple[int, int, int], ...],
        elapsed_ns: int,
    ) -> WeightProfile:
        multipliers = (1.0,) + (0.0,) * observation.degree
        profile = build_weight_profile(
            observation.participant_ids,
            multipliers,
            omega=self.artifact.omega,
            context=f"{context} fallback={reason}",
        )
        self._emit(
            BayesianDiagnostic(
                node_id=observation.node_id,
                round_index=observation.round_index,
                degree=observation.degree,
                degree_bin=degree_bin,
                progress=progress,
                chosen_multipliers=profile.multipliers,
                candidate_count=candidate_count,
                pruned=pruned,
                posterior=posterior,
                fallback_reason=reason,
                normalization_mass=mass,
                sender_ids=observation.neighbor_ids,
                evidence_bins=evidence_bins,
                elapsed_ns=elapsed_ns,
            )
        )
        return profile

    def select_weights(self, observation: LocalObservation) -> WeightProfile:
        """Select the posterior-weighted joint-action argmax for one node/round."""

        start = self.clock()
        obs = _require_observation(observation, policy_name="BayesianPolicy")
        context = f"BayesianPolicy {_observation_context(obs)}"
        degree = obs.degree

        # An empty neighborhood is the normal self-only case, not a fallback.
        if degree == 0:
            profile = build_weight_profile(
                obs.participant_ids,
                (1.0,),
                omega=self.artifact.omega,
                context=context,
            )
            self._emit(
                BayesianDiagnostic(
                    node_id=obs.node_id,
                    round_index=obs.round_index,
                    degree=0,
                    degree_bin=None,
                    progress=None,
                    chosen_multipliers=profile.multipliers,
                    candidate_count=0,
                    pruned=False,
                    posterior=(),
                    fallback_reason=None,
                    normalization_mass=None,
                    sender_ids=(),
                    evidence_bins=(),
                    elapsed_ns=self._elapsed_ns(start),
                )
            )
            return profile

        # Evidence cells are computable from the artifact bins alone, so they
        # are captured on every non-empty diagnostic (including fallbacks) and
        # let the offline collector measure on-policy table support.
        evidence_bins = tuple(
            _neighbor_evidence(self.artifact, neighbor)
            for neighbor in obs.neighbors
        )
        degree_bin = _resolve_degree_bin(
            self.artifact.manifest.bins.degree.labels, degree
        )
        progress = _resolve_progress_bin(
            self.artifact.manifest.bins.progress.labels,
            obs.round_index,
            self.total_rounds,
        )
        if degree_bin is None or progress is None:
            return self._fallback(
                obs,
                context=context,
                reason=FALLBACK_CONTEXT_UNAVAILABLE,
                degree_bin=degree_bin,
                progress=progress,
                candidate_count=0,
                pruned=degree > self.artifact.degree_cap,
                posterior=(),
                mass=None,
                evidence_bins=evidence_bins,
                elapsed_ns=self._elapsed_ns(start),
            )

        posteriors, mass, reason = _compute_posterior(
            self.artifact,
            degree_bin=degree_bin,
            progress=progress,
            evidence_bins=evidence_bins,
        )
        if reason is not None:
            return self._fallback(
                obs,
                context=context,
                reason=reason,
                degree_bin=degree_bin,
                progress=progress,
                candidate_count=0,
                pruned=degree > self.artifact.degree_cap,
                posterior=(),
                mass=mass,
                evidence_bins=evidence_bins,
                elapsed_ns=self._elapsed_ns(start),
            )
        assert posteriors is not None

        disagreements = tuple(
            neighbor.directional_disagreement for neighbor in obs.neighbors
        )
        sender_ids = tuple(neighbor.sender_id for neighbor in obs.neighbors)
        pruned = degree > self.artifact.degree_cap
        actions = enumerate_candidate_actions(
            degree,
            self.artifact.multiplier_set,
            degree_cap=self.artifact.degree_cap,
            disagreements=disagreements,
            sender_ids=sender_ids,
        )
        candidate_count = len(actions)

        values, reason = _expected_payoffs(
            self.artifact,
            degree_bin=degree_bin,
            progress=progress,
            evidence_bins=evidence_bins,
            posteriors=posteriors,
            actions=actions,
        )
        if reason is not None:
            return self._fallback(
                obs,
                context=context,
                reason=reason,
                degree_bin=degree_bin,
                progress=progress,
                candidate_count=candidate_count,
                pruned=pruned,
                posterior=posteriors,
                mass=mass,
                evidence_bins=evidence_bins,
                elapsed_ns=self._elapsed_ns(start),
            )
        assert values is not None

        best_index = _first_argmax(values, self.artifact.tie_tolerance)
        chosen = (1.0,) + tuple(actions[best_index])
        profile = build_weight_profile(
            obs.participant_ids,
            chosen,
            omega=self.artifact.omega,
            context=context,
        )
        self._emit(
            BayesianDiagnostic(
                node_id=obs.node_id,
                round_index=obs.round_index,
                degree=degree,
                degree_bin=degree_bin,
                progress=progress,
                chosen_multipliers=profile.multipliers,
                candidate_count=candidate_count,
                pruned=pruned,
                posterior=posteriors,
                fallback_reason=None,
                normalization_mass=mass,
                sender_ids=obs.neighbor_ids,
                evidence_bins=evidence_bins,
                elapsed_ns=self._elapsed_ns(start),
            )
        )
        return profile


def build_bayesian_policy(
    artifact: BayesianArtifact,
    total_rounds: int,
    *,
    diagnostic_sink: Callable[[BayesianDiagnostic], None] | None = None,
    clock: Callable[[], int] | None = None,
) -> BayesianPolicy:
    """Construct one immutable Bayesian policy from a loaded artifact (D023).

    ``total_rounds`` is the validated experiment round count fixed at episode
    construction; it is combined with each observation's ``round_index`` to
    select the early/mid/late progress bin without widening ``LocalObservation``.
    ``diagnostic_sink`` is an optional construction-time observability hook and
    ``clock`` an optional monotonic nanosecond source (defaults to
    :func:`time.perf_counter_ns`) used only to timestamp selections.
    """

    return BayesianPolicy(
        artifact=artifact,
        total_rounds=total_rounds,
        diagnostic_sink=diagnostic_sink,
        clock=clock,
    )


def _require_bayesian_provenance_match(
    config: BayesianAggregation, artifact: BayesianArtifact
) -> None:
    """Reject a config whose pinned action space disagrees with the artifact.

    The config pins omega/degree_cap/tie_tolerance to the D021 first-pass
    values; the loaded artifact declares its own. A mismatch means the config
    and artifact were produced by incompatible revisions, so the episode fails
    closed before any round rather than silently using one side's values.
    """

    mismatches: list[str] = []
    if config.omega != artifact.omega:
        mismatches.append(f"omega config={config.omega!r} artifact={artifact.omega!r}")
    if config.degree_cap != artifact.degree_cap:
        mismatches.append(
            f"degree_cap config={config.degree_cap!r} artifact={artifact.degree_cap!r}"
        )
    if config.tie_tolerance != artifact.tie_tolerance:
        mismatches.append(
            "tie_tolerance config="
            f"{config.tie_tolerance!r} artifact={artifact.tie_tolerance!r}"
        )
    if mismatches:
        raise PolicyError(
            "bayesian config/artifact provenance mismatch: " + "; ".join(mismatches)
        )


def _build_bayesian_from_config(
    config: BayesianAggregation,
    total_rounds: int | None,
    *,
    diagnostic_sink: Callable[[BayesianDiagnostic], None] | None = None,
) -> BayesianPolicy:
    """Load the frozen artifact once and build the Bayesian policy (D021/D023).

    ``total_rounds`` is required: the policy combines it with each observation's
    ``round_index`` to resolve the early/mid/late progress bin, so it must be
    validated before any selection. The artifact is read and validated exactly
    once here, before the policy exists and before any round starts; a malformed
    or missing artifact raises the loader's localized
    :class:`~dfl_sim.bayesian_artifact.BayesianArtifactError`.
    """

    if total_rounds is None:
        raise PolicyError(
            "bayesian aggregation requires the validated total_rounds at "
            "construction (D023)"
        )
    rounds = _require_int(total_rounds, label="total_rounds")
    if rounds < 1:
        raise PolicyError(f"total_rounds must be >= 1, got {rounds!r}")

    artifact = load_bayesian_artifact(config.artifact_dir)
    _require_bayesian_provenance_match(config, artifact)
    return build_bayesian_policy(
        artifact, rounds, diagnostic_sink=diagnostic_sink
    )


def build_policy(
    config: AggregationConfig,
    *,
    total_rounds: int | None = None,
    diagnostic_sink: Callable[[BayesianDiagnostic], None] | None = None,
) -> AggregationPolicy:
    """Pure config-to-policy factory (D008/R007/D023).

    Maps one validated aggregation selector to a frozen built-in policy. The
    result is stateless and immutable, so an episode constructs it exactly once
    and reuses it for every node and round, mirroring the fixed attacker
    controller. Unknown or malformed selectors fail closed before any round.

    The three M001 baselines take no additional context. The D026 validation
    reference, D029 trust comparator, and D028 BALANCE selector likewise select
    from the frozen scalar observation; BALANCE additionally requires the
    validated ``total_rounds`` fixed at episode construction (the same D023
    pattern as the Bayesian selector) so its round-progress decay is
    well-defined. The Bayesian selector also requires the validated
    ``total_rounds`` (D023): it loads and validates the configured artifact
    exactly once, cross-checks the config's pinned action-space provenance
    against the artifact, and returns the immutable :class:`BayesianPolicy`.
    ``diagnostic_sink`` is an optional construction-time hook forwarded only to
    the Bayesian policy; baseline policies ignore it and ordinary callers
    remain unchanged.
    """

    if isinstance(config, LocalOnlyAggregation):
        return LocalOnlyPolicy()
    if isinstance(config, PlainAverageAggregation):
        return PlainAveragePolicy()
    if isinstance(config, FixedSimilarityAggregation):
        return FixedSimilarityFilterPolicy(
            cutoff=float(config.cutoff),
            downweight_multiplier=float(config.downweight_multiplier),
        )
    if isinstance(config, ValidationReferenceAggregation):
        return ValidationReferencePolicy(tolerance=float(config.tolerance))
    if isinstance(config, TrustWeightedAggregation):
        return TrustWeightedPolicy(beta=float(config.beta))
    if isinstance(config, BalanceAggregation):
        if total_rounds is None:
            raise PolicyError(
                "balance aggregation requires the validated total_rounds at "
                "construction (D028)"
            )
        return BalancePolicy(
            total_rounds=total_rounds,
            alpha=float(config.alpha),
            gamma=float(config.gamma),
            kappa=float(config.kappa),
        )
    if isinstance(config, BayesianAggregation):
        return _build_bayesian_from_config(
            config, total_rounds, diagnostic_sink=diagnostic_sink
        )
    raise PolicyError(
        f"unsupported aggregation config: {type(config).__name__}"
    )
