"""Local observation feature contract for the synchronous DFL round (R005/D012/D017).

This module owns the *executable* information boundary of the simulator. A
:class:`LocalObservation` is an immutable, slotted record carrying only scalar
features for one node in one round:

* ``node_id`` / ``round_index`` context,
* the participant order (self first, then received sender ids ascending), and
* one :class:`NeighborObservation` per received sender with three scalar
  features derived from the node's own round-start ``w_i`` and locally trained
  ``w_tilde_i`` models and the sender's delivered ``u_j_to_i`` model:

    delta_ii      = w_tilde_i - w_i
    delta_ij      = u_j_to_i - w_i
    disagreement  = 1 - clamp(cos(delta_ii, delta_ij), -1, 1)
    relative_norm = ||delta_ij|| / max(||delta_ii||, epsilon)
    loss_delta    = L_i(u_j_to_i) - L_i(w_tilde_i)

Model payloads are untrusted boundary input (Q3): every state is validated for
an exact key/shape/dtype signature and finiteness *before* arithmetic, flattened
over sorted keys, and never aliased or mutated. The node-local loss evaluation
is injected through a narrow seam (:data:`LossEvaluator`) and receives only a
detached clone, so no caller-owned tensor becomes policy-visible mutable state.
All exposed features are guaranteed finite; malformed payloads, non-finite
deltas/losses, invalid epsilon, or a raised/malformed evaluator produce a typed
:class:`ObservationError` with node/sender/round context and no partial result.

Deterministic zero handling (D017): when both ``delta_ii`` and ``delta_ij`` are
zero, disagreement is ``0`` and relative norm is ``0``; when exactly one is
zero, disagreement is ``1``. This module deliberately imports neither
``simulator`` (avoiding a circular import) nor any evaluator/trajectory code.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Protocol, TypeAlias, runtime_checkable

import torch

from .aggregation import AggregationError, validate_state_dict

__all__ = [
    "DEFAULT_EPSILON",
    "DefaultObservationBuilder",
    "LocalObservation",
    "LossEvaluator",
    "NeighborObservation",
    "ObservationBuilder",
    "ObservationError",
    "build_local_observation",
]

#: Default additive floor for the relative-norm denominator (D017).
DEFAULT_EPSILON = 1e-12


class ObservationError(RuntimeError):
    """Raised when the local observation contract is violated.

    Every instance carries node/round context and, where applicable, the
    offending sender id, so a round-level failure is attributable before any
    state is committed.
    """


#: Node-local loss seam: ``L_i(state)`` over the node's own partition. It
#: receives a validated detached clone plus node/round context and must return
#: a finite real number.
LossEvaluator: TypeAlias = Callable[..., float]


def _require_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ObservationError(
            f"{label} must be int, got {type(value).__name__}"
        )
    return int(value)


def _require_finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ObservationError(
            f"{label} must be a real number, got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ObservationError(f"{label} must be finite, got {result!r}")
    return result


def _context(node_id: int, round_index: int, sender_id: int | None = None) -> str:
    parts = [f"node={node_id}", f"round={round_index}"]
    if sender_id is not None:
        parts.append(f"sender={sender_id}")
    return " ".join(parts)


@dataclass(frozen=True, slots=True)
class NeighborObservation:
    """Immutable scalar feature triple for one received sender (D017).

    Holds no tensors and shares no storage with any model: the policy boundary
    can consume it without gaining access to privileged simulator/evaluator
    state.
    """

    sender_id: int
    directional_disagreement: float
    relative_norm: float
    loss_delta: float

    def __post_init__(self) -> None:
        sender = _require_int(self.sender_id, label="sender_id")
        disagreement = _require_finite_float(
            self.directional_disagreement, label="directional_disagreement"
        )
        relative = _require_finite_float(self.relative_norm, label="relative_norm")
        delta = _require_finite_float(self.loss_delta, label="loss_delta")
        if not 0.0 <= disagreement <= 2.0:
            raise ObservationError(
                "directional_disagreement must be within [0, 2], got "
                f"{disagreement!r} (sender={sender})"
            )
        if relative < 0.0:
            raise ObservationError(
                "relative_norm must be non-negative, got "
                f"{relative!r} (sender={sender})"
            )
        object.__setattr__(self, "sender_id", sender)
        object.__setattr__(self, "directional_disagreement", disagreement)
        object.__setattr__(self, "relative_norm", relative)
        object.__setattr__(self, "loss_delta", delta)


@dataclass(frozen=True, slots=True)
class LocalObservation:
    """Immutable local evidence for one node/round (R005/D012/D017).

    The participant order is the node's own id first, then every received
    sender id in strictly ascending order. Only scalar context and features are
    exposed -- never a model tensor, dataset, topology, attack identity, or
    evaluator object.
    """

    node_id: int
    round_index: int
    epsilon: float
    neighbors: tuple[NeighborObservation, ...]

    def __post_init__(self) -> None:
        node = _require_int(self.node_id, label="node_id")
        index = _require_int(self.round_index, label="round_index")
        epsilon = _require_finite_float(self.epsilon, label="epsilon")
        if epsilon <= 0.0:
            raise ObservationError(f"epsilon must be positive, got {epsilon!r}")
        if not isinstance(self.neighbors, tuple) or any(
            not isinstance(neighbor, NeighborObservation)
            for neighbor in self.neighbors
        ):
            raise ObservationError(
                "neighbors must be a tuple of NeighborObservation records"
            )
        previous: int | None = None
        for neighbor in self.neighbors:
            if neighbor.sender_id == node:
                raise ObservationError(
                    f"neighbor sender {neighbor.sender_id} must not equal node "
                    f"id ({node})"
                )
            if previous is not None and neighbor.sender_id <= previous:
                raise ObservationError(
                    "neighbors must be strictly ascending by sender_id"
                )
            previous = neighbor.sender_id
        object.__setattr__(self, "node_id", node)
        object.__setattr__(self, "round_index", index)
        object.__setattr__(self, "epsilon", epsilon)

    @property
    def neighbor_ids(self) -> tuple[int, ...]:
        """Received sender ids in canonical ascending order."""

        return tuple(neighbor.sender_id for neighbor in self.neighbors)

    @property
    def participant_ids(self) -> tuple[int, ...]:
        """Self-first participant order: ``(node_id, *neighbor_ids)``."""

        return (self.node_id, *self.neighbor_ids)

    @property
    def degree(self) -> int:
        """Number of received senders in this observation."""

        return len(self.neighbors)


@runtime_checkable
class ObservationBuilder(Protocol):
    """Narrow seam building a node's :class:`LocalObservation` for one round.

    The round engine invokes :meth:`build_observation` exactly once per node
    after the receive barrier and passes only that node's own round-start and
    trained models, its sorted received ``(sender_id, state)`` pairs, and its
    own loss evaluator. A builder instance therefore never observes the
    topology, attack config, other inboxes, global results, or any evaluator
    object owned by another node.
    """

    def build_observation(
        self,
        *,
        node_id: int,
        round_index: int,
        round_start: Mapping[str, torch.Tensor],
        trained: Mapping[str, torch.Tensor],
        sender_states: Sequence[tuple[int, Mapping[str, torch.Tensor]]],
        evaluator: LossEvaluator,
    ) -> LocalObservation:
        """Build ``node_id``'s immutable local observation for this round."""

        ...


def _require_mapping(value: object, *, label: str, ctx: str) -> None:
    if not isinstance(value, Mapping):
        raise ObservationError(
            f"{label} must be a mapping ({ctx}), got {type(value).__name__}"
        )


def _validated_state(
    raw: object,
    *,
    reference: Mapping[str, torch.Tensor] | None,
    label: str,
    ctx: str,
) -> dict[str, torch.Tensor]:
    try:
        return validate_state_dict(
            raw, reference=reference, context=f"{label} {ctx}"
        )
    except AggregationError as exc:
        raise ObservationError(f"invalid {label} ({ctx}): {exc}") from exc


def _flatten(state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Canonical sorted-key flattening to a float64 vector.

    Keys are visited in sorted order and each tensor is reshaped row-major; the
    float64 promotion makes mixed-precision payloads comparable and keeps the
    norm/dot arithmetic deterministic.
    """

    parts = [
        state[key].reshape(-1).to(torch.float64) for key in sorted(state)
    ]
    return torch.cat(parts)


def _finite_vector(
    vector: torch.Tensor, *, label: str, ctx: str
) -> torch.Tensor:
    if not bool(torch.isfinite(vector).all()):
        raise ObservationError(f"non-finite {label} vector ({ctx})")
    return vector


def _finite_norm(vector: torch.Tensor, *, label: str, ctx: str) -> float:
    norm = float(torch.linalg.vector_norm(vector))
    if not math.isfinite(norm):
        raise ObservationError(f"non-finite {label} norm ({ctx})")
    return norm


def _normalize_sender_states(
    sender_states: object,
    *,
    start: Mapping[str, torch.Tensor],
    node: int,
    index: int,
) -> list[tuple[int, dict[str, torch.Tensor]]]:
    """Validate received ``(sender_id, state)`` pairs into sorted clones.

    Rejects non-sequence input, malformed pairs, non-integer/self/duplicate
    sender ids, non-mapping states, and any state that does not exactly match
    the node's round-start signature.
    """

    ctx = _context(node, index)
    if isinstance(sender_states, (str, bytes)) or not isinstance(
        sender_states, Sequence
    ):
        raise ObservationError(
            f"sender_states must be a sequence of (sender_id, state) pairs "
            f"({ctx}), got {type(sender_states).__name__}"
        )

    raw: dict[int, object] = {}
    for entry in sender_states:
        if (
            isinstance(entry, (str, bytes))
            or not isinstance(entry, Sequence)
            or len(entry) != 2
        ):
            raise ObservationError(
                f"each sender entry must be a (sender_id, state) pair ({ctx})"
            )
        sender = _require_int(entry[0], label="sender_id")
        if sender == node:
            raise ObservationError(
                f"received sender {sender} must not equal node id ({ctx})"
            )
        if sender in raw:
            raise ObservationError(
                f"duplicate received sender {sender} ({ctx})"
            )
        raw[sender] = entry[1]

    validated: list[tuple[int, dict[str, torch.Tensor]]] = []
    for sender in sorted(raw):
        sender_ctx = _context(node, index, sender)
        state = raw[sender]
        _require_mapping(state, label="sender state", ctx=sender_ctx)
        validated.append(
            (
                sender,
                _validated_state(
                    state,
                    reference=start,
                    label="sender state",
                    ctx=sender_ctx,
                ),
            )
        )
    return validated


def _directional_disagreement(
    delta_ii: torch.Tensor,
    delta_ij: torch.Tensor,
    *,
    ii_norm: float,
    ij_norm: float,
    ctx: str,
) -> float:
    """D017 directional disagreement with deterministic zero handling."""

    if ii_norm == 0.0 and ij_norm == 0.0:
        return 0.0
    if ii_norm == 0.0 or ij_norm == 0.0:
        return 1.0
    # Normalize before the dot product so extreme finite vectors cannot
    # overflow into a non-finite cosine.
    cosine = float(torch.dot(delta_ii / ii_norm, delta_ij / ij_norm))
    if not math.isfinite(cosine):
        raise ObservationError(f"non-finite cosine ({ctx})")
    cosine = max(-1.0, min(1.0, cosine))
    return 1.0 - cosine


def _evaluate_loss(
    evaluator: LossEvaluator,
    state: Mapping[str, torch.Tensor],
    *,
    node: int,
    index: int,
    sender: int | None,
    ctx: str,
) -> float:
    """Call the node-local loss seam and validate its scalar output."""

    try:
        value = evaluator(
            state,
            node_id=node,
            round_index=index,
            context=f"{ctx} local-loss",
        )
    except ObservationError:
        raise
    except Exception as exc:
        raise ObservationError(
            f"local loss evaluation failed ({ctx}): {exc}"
        ) from exc

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ObservationError(
            f"evaluator returned non-numeric loss {value!r} ({ctx})"
        )
    loss = float(value)
    if not math.isfinite(loss):
        raise ObservationError(f"non-finite local loss {loss!r} ({ctx})")
    return loss


def build_local_observation(
    *,
    node_id: int,
    round_index: int,
    round_start: Mapping[str, torch.Tensor],
    trained: Mapping[str, torch.Tensor],
    sender_states: Sequence[tuple[int, Mapping[str, torch.Tensor]]],
    evaluator: LossEvaluator,
    epsilon: float = DEFAULT_EPSILON,
) -> LocalObservation:
    """Build one node's immutable, finite, scalar-only local observation.

    ``sender_states`` is the node's received ``(sender_id, state)`` set; it is
    validated, cloned, and canonicalized to ascending sender order during the
    build, so the caller's delivery order does not leak into policy input.
    Neither ``round_start``, ``trained``, nor any sender state is mutated or
    aliased; the returned observation holds no tensor storage at all.
    """

    node = _require_int(node_id, label="node_id")
    index = _require_int(round_index, label="round_index")
    eps = _require_finite_float(epsilon, label="epsilon")
    if eps <= 0.0:
        raise ObservationError(f"epsilon must be positive, got {eps!r}")

    ctx = _context(node, index)
    _require_mapping(round_start, label="round_start", ctx=ctx)
    _require_mapping(trained, label="trained", ctx=ctx)

    start = _validated_state(
        round_start, reference=None, label="round-start state", ctx=ctx
    )
    trained_state = _validated_state(
        trained, reference=start, label="trained state", ctx=ctx
    )

    start_flat = _flatten(start)
    delta_ii = _finite_vector(
        _flatten(trained_state) - start_flat, label="delta_ii", ctx=ctx
    )
    ii_norm = _finite_norm(delta_ii, label="delta_ii", ctx=ctx)

    pairs = _normalize_sender_states(
        sender_states, start=start, node=node, index=index
    )

    # One local loss evaluation of the node's own trained model, reused for
    # every sender's loss delta (Q6 load profile).
    trained_loss = _evaluate_loss(
        evaluator, trained_state, node=node, index=index, sender=None, ctx=ctx
    )

    neighbors: list[NeighborObservation] = []
    for sender, state in pairs:
        sender_ctx = _context(node, index, sender)
        delta_ij = _finite_vector(
            _flatten(state) - start_flat, label="delta_ij", ctx=sender_ctx
        )
        ij_norm = _finite_norm(delta_ij, label="delta_ij", ctx=sender_ctx)

        disagreement = _directional_disagreement(
            delta_ii, delta_ij, ii_norm=ii_norm, ij_norm=ij_norm, ctx=sender_ctx
        )
        relative_norm = ij_norm / max(ii_norm, eps)
        if not math.isfinite(relative_norm):
            raise ObservationError(f"non-finite relative norm ({sender_ctx})")

        sender_loss = _evaluate_loss(
            evaluator, state, node=node, index=index, sender=sender, ctx=sender_ctx
        )
        loss_delta = sender_loss - trained_loss
        if not math.isfinite(loss_delta):
            raise ObservationError(f"non-finite loss delta ({sender_ctx})")

        neighbors.append(
            NeighborObservation(
                sender_id=sender,
                directional_disagreement=disagreement,
                relative_norm=relative_norm,
                loss_delta=loss_delta,
            )
        )

    return LocalObservation(
        node_id=node,
        round_index=index,
        epsilon=eps,
        neighbors=tuple(neighbors),
    )


@dataclass(frozen=True, slots=True)
class DefaultObservationBuilder:
    """Production :class:`ObservationBuilder` with a fixed epsilon.

    Immutable and constructed once per episode, mirroring the fixed attacker
    controller. Its only configuration is the relative-norm epsilon, which is
    validated at construction so a malformed value fails before any round.
    """

    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        eps = _require_finite_float(self.epsilon, label="epsilon")
        if eps <= 0.0:
            raise ObservationError(f"epsilon must be positive, got {eps!r}")
        object.__setattr__(self, "epsilon", eps)

    def build_observation(
        self,
        *,
        node_id: int,
        round_index: int,
        round_start: Mapping[str, torch.Tensor],
        trained: Mapping[str, torch.Tensor],
        sender_states: Sequence[tuple[int, Mapping[str, torch.Tensor]]],
        evaluator: LossEvaluator,
    ) -> LocalObservation:
        """Build one observation using this builder's configured epsilon."""

        return build_local_observation(
            node_id=node_id,
            round_index=round_index,
            round_start=round_start,
            trained=trained,
            sender_states=sender_states,
            evaluator=evaluator,
            epsilon=self.epsilon,
        )
