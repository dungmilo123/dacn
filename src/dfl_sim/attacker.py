"""Fixed per-episode delta cancellation/reversal attacker (R003, D007, D016).

The attack family is a one-shot transform of a sender's canonical outgoing
model computed from its round-start snapshot ``w`` and its locally trained
model ``w_tilde``:

    u = w_tilde - kappa * (w_tilde - w)

``kappa`` is fixed per episode from the D016 table (:data:`ATTACK_KAPPAS`): no
round observes a different intensity, and only one scalar ``compromised_node``
can be poisoned. The controller is *pure*: it validates the two input state
maps, computes the transformed tensor exactly once per floating parameter, and
returns detached clones. It never mutates or aliases either input, so the round
engine can compute the canonical outgoing value once and fan it out to every
neighbor without weakening snapshot isolation. It satisfies the narrow
:class:`OutgoingController` protocol consumed by the round engine.

Non-floating tensors (e.g. integer bookkeeping buffers) have no meaningful
linear interpolation, so they are copied through unchanged for every sender.
Malformed payloads -- empty maps, key/shape/dtype mismatches, non-tensor
entries, or non-finite floating values -- raise :class:`AttackError` before any
output is produced, so a controller failure returns no partial state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from .aggregation import ModelState
from .config import ExperimentConfig, AttackConfig

__all__ = ["AttackError", "FixedAttackController", "OutgoingController"]


class AttackError(RuntimeError):
    """Raised when an outgoing-state attack contract is violated."""


@runtime_checkable
class OutgoingController(Protocol):
    """Narrow seam for one round's canonical sender-to-neighbors transform.

    The round engine invokes :meth:`outgoing_state` exactly once per sender per
    round and passes only that sender's own round-start and locally trained
    state maps. An injected controller therefore never observes the topology,
    recipient identity, inboxes, evaluator metrics, or any other node's state.
    """

    def outgoing_state(
        self,
        *,
        sender: int,
        round_start: Mapping[str, torch.Tensor],
        trained: Mapping[str, torch.Tensor],
        round_index: int | None = None,
        context: str = "",
    ) -> ModelState:
        """Return ``sender``'s canonical outgoing model as detached clones."""

        ...


def _validate_state_pair(
    round_start: Mapping[str, torch.Tensor],
    trained: Mapping[str, torch.Tensor],
    *,
    suffix: str,
) -> None:
    """Validate two state maps share a non-empty, matching, finite signature.

    Raises :class:`AttackError` with sender/tensor context on the first
    violation. Key types are checked before any sorting so mixed key types fail
    loudly instead of raising an incidental ``TypeError``.
    """

    if not round_start:
        raise AttackError(f"round-start state is empty{suffix}")
    if not trained:
        raise AttackError(f"trained state is empty{suffix}")

    for label, state in (("round-start", round_start), ("trained", trained)):
        for key in state:
            if not isinstance(key, str):
                raise AttackError(
                    f"{label} state key must be str, got "
                    f"{type(key).__name__}{suffix}"
                )

    missing = sorted(set(round_start) - set(trained))
    extra = sorted(set(trained) - set(round_start))
    if missing or extra:
        raise AttackError(
            f"state key mismatch{suffix}: missing={missing} extra={extra}"
        )

    for key in sorted(trained):
        for label, value in (
            ("round-start", round_start[key]),
            ("trained", trained[key]),
        ):
            if not isinstance(value, torch.Tensor):
                raise AttackError(
                    f"{label} state entry {key!r} must be a tensor, got "
                    f"{type(value).__name__}{suffix}"
                )

        reference = round_start[key]
        current = trained[key]
        if tuple(current.shape) != tuple(reference.shape):
            raise AttackError(
                f"shape mismatch for {key!r}{suffix}: round-start="
                f"{tuple(reference.shape)} trained={tuple(current.shape)}"
            )
        if current.dtype != reference.dtype:
            raise AttackError(
                f"dtype mismatch for {key!r}{suffix}: round-start="
                f"{reference.dtype} trained={current.dtype}"
            )
        for label, value in (
            ("round-start", reference),
            ("trained", current),
        ):
            if torch.is_floating_point(value) and not bool(
                torch.isfinite(value).all()
            ):
                raise AttackError(
                    f"non-finite values in {label} state entry {key!r}{suffix}"
                )


@dataclass(frozen=True, slots=True)
class FixedAttackController:
    """Immutable D016 controller producing one canonical outgoing state.

    Construct from a resolved episode config via :meth:`from_experiment` or
    directly from a validated :class:`~dfl_sim.config.AttackConfig`. The
    controller holds no mutable state and retains no round history, so calling
    :meth:`outgoing_state` for every sender in one round is O(P) for the
    configured sender and clone-only for everyone else.
    """

    attack: AttackConfig

    @classmethod
    def from_experiment(cls, config: ExperimentConfig) -> "FixedAttackController":
        """Build the controller for a resolved episode configuration."""

        return cls(attack=config.attack)

    @property
    def compromised_node(self) -> int | None:
        """The single poisoned node id, or ``None`` for an honest episode."""

        return self.attack.compromised_node

    @property
    def kappa(self) -> float:
        """The D016 numeric reversal intensity for this episode."""

        return self.attack.kappa

    def outgoing_state(
        self,
        *,
        sender: int,
        round_start: Mapping[str, torch.Tensor],
        trained: Mapping[str, torch.Tensor],
        round_index: int | None = None,
        context: str = "",
    ) -> ModelState:
        """Return ``sender``'s canonical outgoing model as detached clones.

        Honest senders receive a clone of ``trained``. The configured sender
        receives ``trained - kappa * (trained - round_start)`` computed exactly
        once per floating tensor; non-floating tensors are copied unchanged.
        Neither input is mutated or aliased and no partial state is returned on
        failure.
        """

        sender_id = int(sender)
        ctx = f"sender={sender_id}"
        if round_index is not None:
            ctx += f" round={int(round_index)}"
        if context:
            ctx += f" {context}"
        suffix = f" ({ctx})"

        # Validate the full pair before computing anything: a malformed payload
        # fails here with no output produced.
        _validate_state_pair(round_start, trained, suffix=suffix)

        kappa = self.kappa
        # kappa == 0.0 is an exact no-op: short-circuit so a selected node with
        # zero intensity is tensor-identical to the honest outgoing model.
        poisoned = self.compromised_node == sender_id and kappa != 0.0

        outgoing: ModelState = {}
        for key in sorted(trained):
            value = trained[key]
            if not poisoned or not torch.is_floating_point(value):
                outgoing[key] = value.detach().clone()
                continue
            reference = round_start[key]
            outgoing[key] = (value - kappa * (value - reference)).detach().clone()
        return outgoing
