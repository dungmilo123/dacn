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
from .attack_schedule import (
    ScheduleConfig,
    ScheduleModel,
    derive_schedule_seed,
    kappa_for_round,
    realized_kappa_sequence,
)
from .config import ExperimentConfig, AttackConfig

__all__ = [
    "AdaptiveAttackController",
    "AttackError",
    "FixedAttackController",
    "OutgoingController",
]


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


@dataclass(frozen=True, slots=True)
class AdaptiveAttackController:
    """Immutable schedule-driven D016 controller (R018, D031).

    Unlike :class:`FixedAttackController`, which holds one static per-episode
    ``kappa``, this controller selects the D016 intensity *per round* from a
    validated :class:`~dfl_sim.attack_schedule.ScheduleConfig`. It holds the
    single ``compromised_node``, the schedule, the episode ``total_rounds``,
    and the schedule seed derived from the episode master seed, and it applies
    the exact same transform ``w_tilde - kappa * (w_tilde - w)`` with a
    tensor-identical no-op at ``kappa == 0.0``.

    The controller is pure and immutable: :meth:`outgoing_state` reads only the
    sender id, the two provided state maps, and the round context. It never
    reads the topology, evaluator, inboxes, or any defender feedback, so it
    satisfies the unchanged :class:`OutgoingController` seam.
    """

    compromised_node: int | None
    schedule: ScheduleConfig
    total_rounds: int
    schedule_seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, ScheduleModel):
            raise AttackError(
                "schedule must be a validated ScheduleConfig, got "
                f"{type(self.schedule).__name__}"
            )
        node = self.compromised_node
        if node is not None and (
            isinstance(node, bool) or not isinstance(node, int) or node < 0
        ):
            raise AttackError(
                "compromised_node must be a non-negative int or None, got "
                f"{node!r}"
            )
        if (
            isinstance(self.total_rounds, bool)
            or not isinstance(self.total_rounds, int)
            or self.total_rounds < 1
        ):
            raise AttackError(
                "total_rounds must be a positive int, got "
                f"{self.total_rounds!r}"
            )
        if (
            isinstance(self.schedule_seed, bool)
            or not isinstance(self.schedule_seed, int)
            or self.schedule_seed < 0
        ):
            raise AttackError(
                "schedule_seed must be a non-negative int, got "
                f"{self.schedule_seed!r}"
            )

    @classmethod
    def from_episode(
        cls,
        *,
        compromised_node: int | None,
        schedule: ScheduleConfig,
        total_rounds: int,
        master_seed: int,
    ) -> "AdaptiveAttackController":
        """Build a controller whose schedule seed derives from the episode seed."""

        return cls(
            compromised_node=compromised_node,
            schedule=schedule,
            total_rounds=total_rounds,
            schedule_seed=derive_schedule_seed(master_seed, schedule),
        )

    @property
    def peak_kappa(self) -> float:
        """The highest kappa the bound schedule can realize over the episode."""

        return max(
            realized_kappa_sequence(
                self.schedule, self.total_rounds, self.schedule_seed
            )
        )

    def kappa_for(self, round_index: int) -> float:
        """Return the selected D016 kappa for ``round_index`` (0 for honest)."""

        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index < 0
        ):
            raise AttackError(
                f"round_index must be a non-negative int, got {round_index!r}"
            )
        if self.compromised_node is None:
            return 0.0
        return kappa_for_round(
            self.schedule, round_index, self.total_rounds, self.schedule_seed
        )

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

        The configured sender receives the schedule-selected per-round
        transform; the exact ``kappa == 0.0`` case is a tensor-identical clone
        of ``trained``. A schedule without a round index cannot be resolved, so
        ``round_index is None`` fails closed. The full state pair is validated
        before any output is produced.
        """

        sender_id = int(sender)
        if round_index is None:
            raise AttackError(
                f"AdaptiveAttackController requires a round_index "
                f"(sender={sender_id})"
            )
        index = int(round_index)
        ctx = f"sender={sender_id} round={index}"
        if context:
            ctx += f" {context}"
        suffix = f" ({ctx})"

        _validate_state_pair(round_start, trained, suffix=suffix)

        kappa = self.kappa_for(index)
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
