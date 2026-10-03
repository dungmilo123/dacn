"""Model aggregation contracts for the synchronous DFL round (R001, R012, D015).

This module owns the arithmetic boundary of the round. Model state dicts are
treated as untrusted boundary payloads (Q3): before any arithmetic we verify
that they are non-empty mappings of string keys to **floating, finite** tensors
with an exact key/shape/dtype signature. Averaging (D015 uniform closed-
neighborhood mean) accumulates in a stable, explicit order and always returns
detached clones, so no caller-owned tensor is mutated or aliased.

The aggregation callable is deliberately narrow --
:func:`aggregate_closed_neighborhood` -- so S03 can replace weight selection
without rewriting the round loop.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Callable, TypeAlias

import torch

__all__ = [
    "AggregationError",
    "Aggregator",
    "ModelState",
    "aggregate_closed_neighborhood",
    "aggregate_weighted_neighborhood",
    "uniform_average",
    "validate_state_dict",
    "weighted_average",
]

#: Absolute tolerance for the participant-aligned unit-sum weight invariant.
_WEIGHT_SUM_TOLERANCE = 1e-9

#: A full-model payload: tensor state-dict keyed by parameter/buffer name.
ModelState: TypeAlias = dict[str, torch.Tensor]

#: Aggregation seam. ``run_round`` calls this with the node's own trained model
#: plus its neighbor models in sorted sender order.
Aggregator: TypeAlias = Callable[..., ModelState]


class AggregationError(RuntimeError):
    """Raised when a model payload violates the aggregation contract."""


def validate_state_dict(
    state: Mapping[str, torch.Tensor],
    *,
    reference: Mapping[str, torch.Tensor] | None = None,
    context: str = "",
) -> ModelState:
    """Validate a model state payload and return a detached clone.

    Checks that ``state`` is a non-empty mapping of string keys to floating,
    finite tensors. When ``reference`` is supplied, the key set, per-key shape,
    and dtype must match exactly. Raises :class:`AggregationError` on any
    violation so malformed payloads fail at the boundary, before arithmetic.
    """

    suffix = f" ({context})" if context else ""
    if not state:
        raise AggregationError(f"state dict is empty{suffix}")

    validated: ModelState = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise AggregationError(
                f"state key must be str, got {type(key).__name__}{suffix}"
            )
        if not isinstance(value, torch.Tensor):
            raise AggregationError(
                f"state entry {key!r} must be a tensor, got "
                f"{type(value).__name__}{suffix}"
            )
        if not torch.is_floating_point(value):
            raise AggregationError(
                f"state entry {key!r} must be floating point, got "
                f"{value.dtype}{suffix}"
            )
        if not bool(torch.isfinite(value).all()):
            raise AggregationError(
                f"non-finite values in state entry {key!r}{suffix}"
            )
        validated[key] = value.detach().clone()

    if reference is not None:
        missing = sorted(set(reference) - set(validated))
        extra = sorted(set(validated) - set(reference))
        if missing or extra:
            raise AggregationError(
                f"state key mismatch{suffix}: missing={missing} extra={extra}"
            )
        for key in sorted(validated):
            expected = reference[key]
            got = validated[key]
            if tuple(got.shape) != tuple(expected.shape):
                raise AggregationError(
                    f"shape mismatch for {key!r}{suffix}: got "
                    f"{tuple(got.shape)}, expected {tuple(expected.shape)}"
                )
            if got.dtype != expected.dtype:
                raise AggregationError(
                    f"dtype mismatch for {key!r}{suffix}: got {got.dtype}, "
                    f"expected {expected.dtype}"
                )

    return validated


def uniform_average(
    states: Sequence[Mapping[str, torch.Tensor]],
    *,
    context: str = "",
) -> ModelState:
    """D015 uniform mean of full models, in stable input order.

    Every state is validated against the first state's signature, then summed
    with an explicit ordered accumulation (never ``stack``/``mean``, whose
    reduction order is not part of this contract). The returned dict holds
    detached clones and shares no storage with any input.
    """

    suffix = f" ({context})" if context else ""
    if not states:
        raise AggregationError(
            f"uniform average requires at least one state{suffix}"
        )

    first = validate_state_dict(states[0], context=f"state[0]{suffix}")
    count = len(states)
    accum: ModelState = {key: first[key] for key in sorted(first)}

    for index in range(1, count):
        validated = validate_state_dict(
            states[index],
            reference=first,
            context=f"state[{index}]{suffix}",
        )
        for key in accum:
            accum[key].add_(validated[key])

    averaged: ModelState = {}
    for key in accum:
        accum[key].div_(count)
        if not bool(torch.isfinite(accum[key]).all()):
            raise AggregationError(
                f"non-finite average for {key!r}{suffix}"
            )
        averaged[key] = accum[key].detach().clone()
    return averaged


def _validate_weights(
    weights: object, *, context: str = ""
) -> tuple[float, ...]:
    """Validate a finite, non-negative, unit-sum weight sequence (Q3 boundary).

    Weight profiles are untrusted even when produced in-process by a built-in
    policy, so shape/type/finiteness/non-negativity/sum are all checked before
    any tensor arithmetic.
    """

    suffix = f" ({context})" if context else ""
    if isinstance(weights, (str, bytes)) or not isinstance(weights, Sequence):
        raise AggregationError(
            f"weights must be a sequence of reals{suffix}, got "
            f"{type(weights).__name__}"
        )
    values: list[float] = []
    total = 0.0
    for raw in weights:
        if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
            raise AggregationError(
                f"weight must be a real number{suffix}, got "
                f"{type(raw).__name__}"
            )
        value = float(raw)
        if not math.isfinite(value):
            raise AggregationError(f"weight must be finite{suffix}, got {value!r}")
        if value < 0.0:
            raise AggregationError(
                f"weight must be non-negative{suffix}, got {value!r}"
            )
        values.append(value)
        total += value
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise AggregationError(f"weights must sum to 1{suffix}, got {total!r}")
    return tuple(values)


def weighted_average(
    states: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
    *,
    context: str = "",
) -> ModelState:
    """Participant-aligned weighted mean of full models, in stable order (R006/R012).

    ``states`` and ``weights`` are positionally aligned: ``states[k]`` is the
    model of ``weights[k]`` in the caller's canonical participant order (self
    first, then received senders ascending). Weights are validated for
    finiteness, non-negativity, and unit sum *before* any arithmetic. Every
    state is validated against the first state's exact key/shape/dtype
    signature (and finiteness), accumulated in explicit input order, and the
    result is returned as detached clones that alias no input.
    """

    suffix = f" ({context})" if context else ""
    if not states:
        raise AggregationError(
            f"weighted average requires at least one state{suffix}"
        )
    validated_weights = _validate_weights(weights, context=context)
    if len(states) != len(validated_weights):
        raise AggregationError(
            f"weight count {len(validated_weights)} does not match state count "
            f"{len(states)}{suffix}"
        )

    first = validate_state_dict(states[0], context=f"state[0]{suffix}")
    accum: ModelState = {
        key: first[key] * validated_weights[0] for key in sorted(first)
    }

    for index in range(1, len(states)):
        validated = validate_state_dict(
            states[index],
            reference=first,
            context=f"state[{index}]{suffix}",
        )
        weight = validated_weights[index]
        for key in accum:
            accum[key] = accum[key] + validated[key] * weight

    averaged: ModelState = {}
    for key in accum:
        if not bool(torch.isfinite(accum[key]).all()):
            raise AggregationError(
                f"non-finite weighted average for {key!r}{suffix}"
            )
        averaged[key] = accum[key].detach().clone()
    return averaged


def aggregate_weighted_neighborhood(
    self_state: Mapping[str, torch.Tensor],
    neighbor_states: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
    *,
    node_id: int | None = None,
    round_index: int | None = None,
    context: str = "",
) -> ModelState:
    """Participant-aligned weighted aggregation of a node's closed neighborhood.

    ``weights`` is the full aligned profile (self first, then neighbors in
    canonical sender order) as carried by
    :attr:`~dfl_sim.policy.WeightProfile.weights`. This is the weighted
    replacement for :func:`aggregate_closed_neighborhood` on the same round
    seam: the plain uniform profile reproduces the D015 closed-neighborhood
    mean.
    """

    if not context:
        context = f"node={node_id} round={round_index}"
    states = [self_state, *neighbor_states]
    validated = _validate_weights(weights, context=context)
    if len(validated) != len(states):
        raise AggregationError(
            f"weight count {len(validated)} does not match participant count "
            f"{len(states)} ({context})"
        )
    return weighted_average(states, validated, context=context)


def aggregate_closed_neighborhood(
    self_state: Mapping[str, torch.Tensor],
    neighbor_states: Sequence[Mapping[str, torch.Tensor]],
    *,
    node_id: int | None = None,
    round_index: int | None = None,
    context: str = "",
) -> ModelState:
    """D015 uniform mean of a node's own model plus its neighbor models.

    ``neighbor_states`` must already be in the caller's stable order (the round
    engine supplies one message per graph neighbor sorted by sender). This is
    the narrow replacement seam for S03 weight selection.
    """

    if not context:
        context = f"node={node_id} round={round_index}"
    return uniform_average([self_state, *neighbor_states], context=context)
