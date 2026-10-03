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

Every built-in policy output is treated as untrusted boundary data even though
it is produced in-process: :func:`build_weight_profile` validates the finite
self multiplier (exactly ``1``), finite non-negative multipliers, optional
D017 membership in ``{0, omega, 1}`` for neighbors, and normalization before a
profile exists. Empty and all-rejected neighborhoods are *not* errors and
resolve to exactly self-only weights ``(1.0, 0.0, ...)``/``(1.0,)``.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .config import (
    AggregationConfig,
    FixedSimilarityAggregation,
    LocalOnlyAggregation,
    PlainAverageAggregation,
)
from .observation import LocalObservation

__all__ = [
    "AggregationPolicy",
    "FixedSimilarityFilterPolicy",
    "LocalOnlyPolicy",
    "PlainAveragePolicy",
    "PolicyError",
    "WeightProfile",
    "build_policy",
    "build_weight_profile",
]

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
    """Reject anything that is not the frozen local observation record (R005)."""

    if not isinstance(observation, LocalObservation):
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


def build_policy(config: AggregationConfig) -> AggregationPolicy:
    """Pure config-to-policy factory (D008/R007).

    Maps one validated aggregation selector to a frozen built-in policy. The
    result is stateless and immutable, so an episode constructs it exactly once
    and reuses it for every node and round, mirroring the fixed attacker
    controller. Unknown or malformed selectors fail closed before any round.
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
    raise PolicyError(
        f"unsupported aggregation config: {type(config).__name__}"
    )
