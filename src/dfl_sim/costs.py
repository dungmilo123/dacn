"""Scalar-only computation and logical-communication cost sidecars (R023/D041).

This module owns the *measurement* channel of the simulator. It is deliberately
separate from the deterministic trace schema: timing values are nondeterministic
and must never enter schema-v1 trace bytes, policy arithmetic, tie-breaking, the
frozen selection/final identities, or artifact calibration-input identity
(D041). Every record here is a plain, strict, scalar-only mapping -- no tensors
are ever retained, and candidate/posterior rows are bounded by the existing
observation degree (and, for diagnostics, by ``nodes * rounds``).

Two opt-in channels are published per completed cell:

* a **cost sidecar** (``costs.json``) carrying per-round logical message counts
  and uncompressed tensor payload bytes plus per-boundary nanosecond timings;
* a **decision audit** (``decision-audit.jsonl``) carrying, for every
  node/round, the policy-visible scalar observation and (when the policy emits
  one) the existing :class:`~dfl_sim.policy.BayesianDiagnostic`, so candidate
  enumeration, degree-cap pruning, fallback and expected-utility selection can
  be reconstructed offline without widening the online policy seam.

The default :class:`NullCostCollector`/``telemetry=None`` path performs no clock
call at all, so enabling or disabling telemetry cannot change training,
selection, or trace bytes. A :class:`CellTelemetry` instance is the single
opt-in object threaded through :func:`~dfl_sim.run.run_from_config` and
:func:`~dfl_sim.simulator.run_round`; the policy's existing
``diagnostic_sink`` is reused by making :class:`CellTelemetry` callable, so no
policy input is widened and no evaluator/design/topology type reaches it.

Logical communication is the *actual* directed message count and the sum of
``numel * element_size`` over every message's state tensors at construction
time -- before any receiver-side policy filtering. A zero-weight or
``local_only`` neighbor still incurs exactly the bytes the simulator actually
sends; bytes are never inferred from post-filter weights. Headers, retransmits
and physical-network overhead are out of scope by construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import stat
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from .observation import LocalObservation
from .policy import BayesianDiagnostic, WeightProfile

__all__ = [
    "COST_SCHEMA",
    "COST_SCHEMA_VERSION",
    "COST_SIDECAR_FILENAME",
    "DECISION_AUDIT_SCHEMA",
    "DECISION_AUDIT_SCHEMA_VERSION",
    "DECISION_AUDIT_SIDECAR_FILENAME",
    "MEASUREMENT_CLOCK",
    "WALL_PHASES",
    "CellTelemetry",
    "CostError",
    "DecisionAuditCollector",
    "DecisionAuditRecord",
    "NodeRoundCost",
    "NullCostCollector",
    "RecordingCostCollector",
    "RoundTelemetry",
    "SidecarPaths",
    "canonical_json_bytes",
    "default_clock",
    "load_cost_payload",
    "load_decision_audit_records",
    "state_dict_payload_bytes",
    "trace_artifact_digest",
    "validate_artifact_digests",
    "validate_cost_payload",
    "validate_decision_audit_record",
    "validate_measurement_identity",
]

#: Versioned cost sidecar identity.
COST_SCHEMA = "dfl_cell_cost"
COST_SCHEMA_VERSION = 1

#: Versioned decision-audit sidecar identity.
DECISION_AUDIT_SCHEMA = "dfl_cell_decision_audit"
DECISION_AUDIT_SCHEMA_VERSION = 1

#: Sidecar file names written next to (never inside) a schema-v1 trace.
COST_SIDECAR_FILENAME = "costs.json"
DECISION_AUDIT_SIDECAR_FILENAME = "decision-audit.jsonl"

#: The only monotonic clock used in production; recorded verbatim in sidecars.
MEASUREMENT_CLOCK = "perf_counter_ns"

#: Wall-time phases a runner may measure. All values are integer nanoseconds.
#: ``train`` is deliberately *not* a wall phase here: local-training time is
#: measured per round inside the round engine and summed into ``train_ns`` so a
#: single wall accumulator cannot double-count or disagree with the rounds.
WALL_PHASES: tuple[str, ...] = ("setup", "evaluation", "whole_cell")

#: Identity keys every sidecar must carry (values are bounded scalars).
_REQUIRED_IDENTITY_KEYS = ("phase", "policy_id", "seed")
_ALLOWED_IDENTITY_KEYS = (
    "phase",
    "study_id",
    "condition_id",
    "policy_id",
    "schedule_id",
    "seed",
)

_COST_KEYS = {
    "schema",
    "schema_version",
    "identity",
    "measurement",
    "totals",
    "rounds",
}
_MEASUREMENT_KEYS = {
    "clock",
    "config_digest",
    "artifact_digests",
    "trace_digest",
}
_TOTALS_KEYS = {
    "rounds",
    "undirected_links",
    "directed_messages",
    "logical_payload_bytes",
    "model_payload_bytes",
    "observation_ns",
    "policy_ns",
    "aggregation_ns",
    "selection_ns",
    "setup_ns",
    "train_ns",
    "evaluation_ns",
    "whole_cell_ns",
}
_ROUND_KEYS = {
    "round_index",
    "undirected_links",
    "directed_messages",
    "logical_payload_bytes",
    "model_payload_bytes",
    "train_ns",
    "nodes",
}
_NODE_COST_KEYS = {
    "node_id",
    "round_index",
    "degree",
    "delivered_messages",
    "delivered_payload_bytes",
    "observation_ns",
    "policy_ns",
    "aggregation_ns",
}
_DECISION_KEYS = {
    "schema",
    "schema_version",
    "identity",
    "node_id",
    "round_index",
    "degree",
    "sender_ids",
    "disagreement",
    "relative_norm",
    "loss_delta",
    "chosen_multipliers",
    "weights",
    "candidate_count",
    "pruned",
    "posterior",
    "fallback_reason",
    "normalization_mass",
    "evidence_bins",
    "selection_ns",
}

_SHA256_RE = None  # lazily compiled to avoid a re import at module scope


class CostError(RuntimeError):
    """Raised when a cost/decision sidecar contract is violated."""


Clock = Callable[[], int]


def default_clock() -> int:
    """Production monotonic clock: integer nanoseconds from ``perf_counter_ns``."""

    return int(time.perf_counter_ns())


# --------------------------------------------------------------------------- #
# Scalar validation helpers
# --------------------------------------------------------------------------- #
def _require_int(value: object, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CostError(f"{label} must be int, got {type(value).__name__}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise CostError(f"{label} must be >= {minimum}, got {result!r}")
    return result


def _require_finite_float(
    value: object, *, label: str, minimum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise CostError(
            f"{label} must be a real number, got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise CostError(f"{label} must be finite, got {result!r}")
    if minimum is not None and result < minimum:
        raise CostError(f"{label} must be >= {minimum}, got {result!r}")
    return result


def _require_str(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CostError(f"{label} must be a non-empty string")
    return value


def _require_bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise CostError(f"{label} must be bool, got {type(value).__name__}")
    return bool(value)


def _require_exact_keys(
    payload: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise CostError(
            f"{label} key mismatch (missing={missing}, unexpected={extra})"
        )


def _require_digest(value: object, *, label: str) -> str:
    text = _require_str(value, label=label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise CostError(f"{label} must be a lowercase 64-character sha256 hex")
    return text


def canonical_json_bytes(payload: object) -> bytes:
    """Canonical deterministic JSON bytes for sidecar payloads."""

    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )


# --------------------------------------------------------------------------- #
# Logical payload accounting
# --------------------------------------------------------------------------- #
def state_dict_payload_bytes(state: Mapping[str, object]) -> int:
    """Sum ``numel * element_size`` over every tensor in one validated state.

    This is the *logical uncompressed tensor payload* of one model message. It
    counts exactly the bytes the configured model sends and is computed before
    any receiver-side filtering, so a neighbor that ends up with weight ``0``
    still counts. Mixed-dtype states are summed per tensor with their own
    ``element_size``.
    """

    if not isinstance(state, Mapping):
        raise CostError(
            f"state must be a mapping, got {type(state).__name__}"
        )
    total = 0
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise CostError(
                f"state entry {key!r} must be a torch.Tensor, got "
                f"{type(tensor).__name__}"
            )
        total += int(tensor.numel()) * int(tensor.element_size())
    return total


# --------------------------------------------------------------------------- #
# Trace identity
# --------------------------------------------------------------------------- #
def trace_artifact_digest(manifest: Mapping[str, object]) -> str:
    """Canonical digest of a TraceReader-validated trace manifest.

    The manifest embeds the SHA-256 digest and size of every trace artifact and
    is only accepted here after :class:`~dfl_sim.trajectory.TraceReader` has
    verified it. Hashing the canonical manifest therefore binds the complete
    validated trace content without re-reading multi-gigabyte tensor archives.
    """

    if not isinstance(manifest, Mapping):
        raise CostError("trace manifest must be a mapping")
    return hashlib.sha256(canonical_json_bytes(dict(manifest))).hexdigest()


# --------------------------------------------------------------------------- #
# Measurement identity
# --------------------------------------------------------------------------- #
def validate_measurement_identity(identity: Mapping[str, object]) -> dict:
    """Validate and canonicalize a sidecar identity mapping.

    Required keys: ``phase``, ``policy_id``, ``seed``. Optional keys are
    ``study_id``, ``condition_id`` and ``schedule_id``. All values must be
    bounded scalars (non-empty strings or non-negative integers).
    """

    if not isinstance(identity, Mapping):
        raise CostError("identity must be a mapping")
    unknown = sorted(set(identity) - set(_ALLOWED_IDENTITY_KEYS))
    if unknown:
        raise CostError(f"identity has unexpected keys: {unknown}")
    missing = [key for key in _REQUIRED_IDENTITY_KEYS if key not in identity]
    if missing:
        raise CostError(f"identity is missing required keys: {missing}")

    canonical: dict[str, object] = {}
    for key in _ALLOWED_IDENTITY_KEYS:
        if key not in identity:
            continue
        value = identity[key]
        if key == "seed":
            canonical[key] = _require_int(value, label="identity.seed", minimum=0)
        else:
            canonical[key] = _require_str(value, label=f"identity.{key}")
    return canonical


def validate_artifact_digests(artifact_digests: Mapping[str, str | None]) -> dict:
    """Validate a mapping of artifact identity name -> digest (or ``None``)."""

    if not isinstance(artifact_digests, Mapping):
        raise CostError("artifact_digests must be a mapping")
    canonical: dict[str, str | None] = {}
    for key in sorted(artifact_digests):
        name = _require_str(key, label="artifact_digests key")
        value = artifact_digests[key]
        if value is None:
            canonical[name] = None
        else:
            canonical[name] = _require_digest(
                value, label=f"artifact_digests[{name}]"
            )
    return canonical


# --------------------------------------------------------------------------- #
# Cost records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class NodeRoundCost:
    """Per-node scalar cost row for one round (bounded by ``nodes * rounds``)."""

    node_id: int
    round_index: int
    degree: int
    delivered_messages: int
    delivered_payload_bytes: int
    observation_ns: int
    policy_ns: int
    aggregation_ns: int

    def __post_init__(self) -> None:
        node = _require_int(self.node_id, label="node_id", minimum=0)
        index = _require_int(self.round_index, label="round_index", minimum=0)
        degree = _require_int(self.degree, label="degree", minimum=0)
        delivered = _require_int(
            self.delivered_messages, label="delivered_messages", minimum=0
        )
        if delivered != degree:
            raise CostError(
                f"delivered_messages ({delivered}) must equal degree ({degree})"
            )
        payload = _require_int(
            self.delivered_payload_bytes,
            label="delivered_payload_bytes",
            minimum=0,
        )
        observation = _require_int(
            self.observation_ns, label="observation_ns", minimum=0
        )
        policy = _require_int(self.policy_ns, label="policy_ns", minimum=0)
        aggregation = _require_int(
            self.aggregation_ns, label="aggregation_ns", minimum=0
        )
        object.__setattr__(self, "node_id", node)
        object.__setattr__(self, "round_index", index)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "delivered_messages", delivered)
        object.__setattr__(self, "delivered_payload_bytes", payload)
        object.__setattr__(self, "observation_ns", observation)
        object.__setattr__(self, "policy_ns", policy)
        object.__setattr__(self, "aggregation_ns", aggregation)

    def to_payload(self) -> dict:
        return {
            "node_id": self.node_id,
            "round_index": self.round_index,
            "degree": self.degree,
            "delivered_messages": self.delivered_messages,
            "delivered_payload_bytes": self.delivered_payload_bytes,
            "observation_ns": self.observation_ns,
            "policy_ns": self.policy_ns,
            "aggregation_ns": self.aggregation_ns,
        }


@dataclass(frozen=True, slots=True)
class RoundTelemetry:
    """Scalar communication + timing accounting for exactly one committed round.

    ``messages`` is the number of directed messages the simulator actually
    built (one per directed edge) and ``payload_bytes`` is the sum of every
    message's ``numel * element_size`` at construction, before receiver
    filtering. ``model_payload_bytes`` is the payload of one message (all
    messages carry the same-architecture model).
    """

    round_index: int
    undirected_links: int
    messages: int
    payload_bytes: int
    model_payload_bytes: int
    train_ns: int
    nodes: tuple[NodeRoundCost, ...]

    def __post_init__(self) -> None:
        index = _require_int(self.round_index, label="round_index", minimum=0)
        links = _require_int(self.undirected_links, label="undirected_links", minimum=0)
        messages = _require_int(self.messages, label="messages", minimum=0)
        payload = _require_int(self.payload_bytes, label="payload_bytes", minimum=0)
        model_payload = _require_int(
            self.model_payload_bytes, label="model_payload_bytes", minimum=0
        )
        train = _require_int(self.train_ns, label="train_ns", minimum=0)
        if not isinstance(self.nodes, tuple) or any(
            not isinstance(node, NodeRoundCost) for node in self.nodes
        ):
            raise CostError("nodes must be a tuple of NodeRoundCost records")
        delivered = sum(node.delivered_messages for node in self.nodes)
        if delivered != messages:
            raise CostError(
                f"delivered messages ({delivered}) must equal messages ({messages})"
            )
        delivered_bytes = sum(node.delivered_payload_bytes for node in self.nodes)
        if delivered_bytes != payload:
            raise CostError(
                f"delivered payload bytes ({delivered_bytes}) must equal "
                f"payload_bytes ({payload})"
            )
        seen: set[int] = set()
        for node in self.nodes:
            if node.round_index != index:
                raise CostError(
                    "every node row must carry the round's round_index"
                )
            if node.node_id in seen:
                raise CostError(f"duplicate node row {node.node_id}")
            seen.add(node.node_id)
        object.__setattr__(self, "round_index", index)
        object.__setattr__(self, "undirected_links", links)
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "payload_bytes", payload)
        object.__setattr__(self, "model_payload_bytes", model_payload)
        object.__setattr__(self, "train_ns", train)
        object.__setattr__(self, "nodes", tuple(sorted(self.nodes, key=lambda n: n.node_id)))

    def to_payload(self) -> dict:
        return {
            "round_index": self.round_index,
            "undirected_links": self.undirected_links,
            "directed_messages": self.messages,
            "logical_payload_bytes": self.payload_bytes,
            "model_payload_bytes": self.model_payload_bytes,
            "train_ns": self.train_ns,
            "nodes": [node.to_payload() for node in self.nodes],
        }


# --------------------------------------------------------------------------- #
# Collectors
# --------------------------------------------------------------------------- #
class NullCostCollector:
    """No-op default collector: performs no clock call and stores nothing.

    ``enabled`` is ``False`` so the round engine can skip every timing and
    payload branch entirely; this is what guarantees telemetry-off behavior is
    byte-for-byte identical to the pre-telemetry engine.
    """

    enabled = False

    def record_round(self, telemetry: RoundTelemetry) -> None:  # pragma: no cover
        pass

    def add_wall(self, phase: str, elapsed_ns: int) -> None:  # pragma: no cover
        pass


#: Shared no-op instance (stateless, safe to reuse).
NULL_COST_COLLECTOR = NullCostCollector()


class RecordingCostCollector:
    """Accumulates scalar cost telemetry for one completed cell.

    The clock is injectable so tests can assert exact nanosecond values with a
    fake clock; production uses :func:`default_clock`. Only integer
    nanoseconds and non-negative counts are stored.
    """

    enabled = True

    def __init__(self, *, clock: Clock | None = None) -> None:
        resolved = default_clock if clock is None else clock
        if not callable(resolved):
            raise CostError("clock must be callable")
        self._clock = resolved
        self._rounds: list[RoundTelemetry] = []
        self._wall: dict[str, int] = {}

    @property
    def clock(self) -> Clock:
        return self._clock

    def now(self) -> int:
        value = _require_int(self._clock(), label="clock value", minimum=0)
        return value

    def record_round(self, telemetry: RoundTelemetry) -> None:
        if not isinstance(telemetry, RoundTelemetry):
            raise CostError(
                f"record_round requires a RoundTelemetry, got "
                f"{type(telemetry).__name__}"
            )
        if any(
            existing.round_index == telemetry.round_index
            for existing in self._rounds
        ):
            raise CostError(
                f"duplicate round telemetry {telemetry.round_index}"
            )
        self._rounds.append(telemetry)

    def add_wall(self, phase: str, elapsed_ns: int) -> None:
        if phase not in WALL_PHASES:
            raise CostError(f"unknown wall phase {phase!r}")
        value = _require_int(elapsed_ns, label=f"wall[{phase}]", minimum=0)
        self._wall[phase] = self._wall.get(phase, 0) + value

    @property
    def rounds(self) -> tuple[RoundTelemetry, ...]:
        return tuple(sorted(self._rounds, key=lambda row: row.round_index))

    @property
    def wall_ns(self) -> dict[str, int]:
        return dict(self._wall)

    def totals(self) -> dict:
        rounds = self.rounds
        return {
            "rounds": len(rounds),
            "undirected_links": sum(row.undirected_links for row in rounds),
            "directed_messages": sum(row.messages for row in rounds),
            "logical_payload_bytes": sum(row.payload_bytes for row in rounds),
            "model_payload_bytes": (
                rounds[0].model_payload_bytes if rounds else 0
            ),
            "observation_ns": sum(
                node.observation_ns for row in rounds for node in row.nodes
            ),
            "policy_ns": sum(
                node.policy_ns for row in rounds for node in row.nodes
            ),
            "aggregation_ns": sum(
                node.aggregation_ns for row in rounds for node in row.nodes
            ),
            "train_ns": sum(row.train_ns for row in rounds),
        }


# --------------------------------------------------------------------------- #
# Decision audit
# --------------------------------------------------------------------------- #
def _feature_tuple(values: tuple[float, ...], *, label: str) -> tuple[float, ...]:
    return tuple(
        _require_finite_float(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    )


@dataclass(frozen=True, slots=True)
class DecisionAuditRecord:
    """One node/round selection audit row (scalar-only, bounded by degree).

    Carries the exact policy-visible scalar observation (disagreement,
    relative norm, loss delta per sender) plus the selected action and, for a
    Bayesian policy, the merged :class:`~dfl_sim.policy.BayesianDiagnostic`
    fields. It holds no tensors and no privileged evaluator/topology state, so
    it can be persisted and replayed freely.
    """

    node_id: int
    round_index: int
    degree: int
    sender_ids: tuple[int, ...]
    disagreement: tuple[float, ...]
    relative_norm: tuple[float, ...]
    loss_delta: tuple[float, ...]
    chosen_multipliers: tuple[float, ...]
    weights: tuple[float, ...]
    candidate_count: int | None
    pruned: bool | None
    posterior: tuple[tuple[tuple[int, ...], float], ...]
    fallback_reason: str | None
    normalization_mass: float | None
    evidence_bins: tuple[tuple[int, int, int], ...]
    selection_ns: int

    def __post_init__(self) -> None:
        node = _require_int(self.node_id, label="node_id", minimum=0)
        index = _require_int(self.round_index, label="round_index", minimum=0)
        degree = _require_int(self.degree, label="degree", minimum=0)

        if not isinstance(self.sender_ids, tuple) or len(self.sender_ids) != degree:
            raise CostError("sender_ids must be a tuple aligned with degree")
        senders: list[int] = []
        previous: int | None = None
        for sender in self.sender_ids:
            value = _require_int(sender, label="sender_id", minimum=0)
            if previous is not None and value <= previous:
                raise CostError("sender_ids must be strictly ascending")
            previous = value
            senders.append(value)

        disagreement = _feature_tuple(self.disagreement, label="disagreement")
        relative = _feature_tuple(self.relative_norm, label="relative_norm")
        loss_delta = _feature_tuple(self.loss_delta, label="loss_delta")
        for label, values in (
            ("disagreement", disagreement),
            ("relative_norm", relative),
            ("loss_delta", loss_delta),
        ):
            if len(values) != degree:
                raise CostError(f"{label} must be aligned with degree")

        multipliers = _feature_tuple(
            self.chosen_multipliers, label="chosen_multipliers"
        )
        weights = _feature_tuple(self.weights, label="weights")
        if len(multipliers) != degree + 1 or len(weights) != degree + 1:
            raise CostError(
                "chosen_multipliers and weights must have degree+1 entries "
                "(self plus senders)"
            )
        if any(value < 0.0 for value in multipliers):
            raise CostError("chosen_multipliers must be non-negative")
        if any(value < 0.0 for value in weights):
            raise CostError("weights must be non-negative")

        candidate = self.candidate_count
        if candidate is None:
            candidate_count = None
        else:
            candidate_count = _require_int(
                candidate, label="candidate_count", minimum=0
            )
        pruned = None if self.pruned is None else _require_bool(
            self.pruned, label="pruned"
        )

        if not isinstance(self.posterior, tuple):
            raise CostError("posterior must be a tuple of (state, probability)")
        posterior: list[tuple[tuple[int, ...], float]] = []
        for entry in self.posterior:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise CostError("posterior entries must be (state, probability)")
            state, probability = entry
            if not isinstance(state, tuple):
                raise CostError("posterior state must be a tuple of ints")
            for value in state:
                _require_int(value, label="posterior state entry", minimum=0)
            posterior.append(
                (
                    tuple(int(value) for value in state),
                    _require_finite_float(
                        probability,
                        label="posterior probability",
                        minimum=0.0,
                    ),
                )
            )

        fallback = self.fallback_reason
        if fallback is not None:
            fallback = _require_str(fallback, label="fallback_reason")
        mass = self.normalization_mass
        if mass is not None:
            mass = _require_finite_float(
                mass, label="normalization_mass", minimum=0.0
            )

        if not isinstance(self.evidence_bins, tuple):
            raise CostError("evidence_bins must be a tuple")
        bins: list[tuple[int, int, int]] = []
        for row in self.evidence_bins:
            if not isinstance(row, tuple) or len(row) != 3:
                raise CostError("evidence_bins rows must be 3-tuples of ints")
            bins.append(
                tuple(
                    _require_int(value, label="evidence_bins entry", minimum=0)
                    for value in row
                )
            )
        expected_bins = 0 if candidate_count is None else degree
        if len(bins) != expected_bins:
            raise CostError(
                "evidence_bins must be aligned with degree for a Bayesian "
                "selection and empty for a non-Bayesian selection"
            )

        selection_ns = _require_int(
            self.selection_ns, label="selection_ns", minimum=0
        )

        object.__setattr__(self, "node_id", node)
        object.__setattr__(self, "round_index", index)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "sender_ids", tuple(senders))
        object.__setattr__(self, "disagreement", disagreement)
        object.__setattr__(self, "relative_norm", relative)
        object.__setattr__(self, "loss_delta", loss_delta)
        object.__setattr__(self, "chosen_multipliers", multipliers)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "candidate_count", candidate_count)
        object.__setattr__(self, "pruned", pruned)
        object.__setattr__(self, "posterior", tuple(posterior))
        object.__setattr__(self, "fallback_reason", fallback)
        object.__setattr__(self, "normalization_mass", mass)
        object.__setattr__(self, "evidence_bins", tuple(bins))
        object.__setattr__(self, "selection_ns", selection_ns)

    def to_payload(self, *, identity: Mapping[str, object]) -> dict:
        return {
            "schema": DECISION_AUDIT_SCHEMA,
            "schema_version": DECISION_AUDIT_SCHEMA_VERSION,
            "identity": dict(identity),
            "node_id": self.node_id,
            "round_index": self.round_index,
            "degree": self.degree,
            "sender_ids": list(self.sender_ids),
            "disagreement": list(self.disagreement),
            "relative_norm": list(self.relative_norm),
            "loss_delta": list(self.loss_delta),
            "chosen_multipliers": list(self.chosen_multipliers),
            "weights": list(self.weights),
            "candidate_count": self.candidate_count,
            "pruned": self.pruned,
            "posterior": [
                {"state": list(state), "probability": probability}
                for state, probability in self.posterior
            ],
            "fallback_reason": self.fallback_reason,
            "normalization_mass": self.normalization_mass,
            "evidence_bins": [list(row) for row in self.evidence_bins],
            "selection_ns": self.selection_ns,
        }


class DecisionAuditCollector:
    """Aligns policy-visible observations with optional Bayesian diagnostics.

    Observations are recorded by the round engine (:meth:`record_observation`)
    and diagnostics by the policy's existing ``diagnostic_sink``
    (:meth:`record_diagnostic`); the two are keyed by ``(node_id, round_index)``
    so the sidecar can reconstruct exactly which scalar features produced which
    selection. No tensors or privileged state are retained.
    """

    def __init__(self, *, expected_rows: int | None = None) -> None:
        self._expected_rows = (
            None
            if expected_rows is None
            else _require_int(expected_rows, label="expected_rows", minimum=0)
        )
        self._observations: dict[tuple[int, int], tuple[LocalObservation, WeightProfile]] = {}
        self._diagnostics: dict[tuple[int, int], BayesianDiagnostic] = {}

    def set_expected_rows(self, expected_rows: int) -> None:
        if self._expected_rows is not None:
            raise CostError("expected_rows is already set")
        self._expected_rows = _require_int(
            expected_rows, label="expected_rows", minimum=0
        )

    def record_observation(
        self, observation: LocalObservation, profile: WeightProfile
    ) -> None:
        if not isinstance(observation, LocalObservation):
            raise CostError("record_observation requires a LocalObservation")
        if not isinstance(profile, WeightProfile):
            raise CostError("record_observation requires a WeightProfile")
        key = (int(observation.node_id), int(observation.round_index))
        if key in self._observations:
            raise CostError(
                f"duplicate observation for node={key[0]} round={key[1]}"
            )
        self._observations[key] = (observation, profile)

    def record_diagnostic(self, diagnostic: BayesianDiagnostic) -> None:
        if not isinstance(diagnostic, BayesianDiagnostic):
            raise CostError("record_diagnostic requires a BayesianDiagnostic")
        key = (int(diagnostic.node_id), int(diagnostic.round_index))
        if key in self._diagnostics:
            raise CostError(
                f"duplicate diagnostic for node={key[0]} round={key[1]}"
            )
        self._diagnostics[key] = diagnostic

    def records(self) -> tuple[DecisionAuditRecord, ...]:
        orphan = sorted(set(self._diagnostics) - set(self._observations))
        if orphan:
            raise CostError(
                f"diagnostics without a matching observation: {orphan}"
            )
        if (
            self._expected_rows is not None
            and len(self._observations) != self._expected_rows
        ):
            raise CostError(
                f"expected {self._expected_rows} decision rows, collected "
                f"{len(self._observations)}"
            )
        rows: list[DecisionAuditRecord] = []
        for key in sorted(self._observations):
            observation, profile = self._observations[key]
            diagnostic = self._diagnostics.get(key)
            rows.append(_merge_decision(key, observation, profile, diagnostic))
        return tuple(rows)


def _merge_decision(
    key: tuple[int, int],
    observation: LocalObservation,
    profile: WeightProfile,
    diagnostic: BayesianDiagnostic | None,
) -> DecisionAuditRecord:
    node, index = key
    neighbors = observation.neighbors
    sender_ids = tuple(neighbor.sender_id for neighbor in neighbors)
    if tuple(profile.participant_ids) != (node, *sender_ids):
        raise CostError(
            f"profile participants do not match observation for node={node} "
            f"round={index}"
        )
    disagreement = tuple(neighbor.directional_disagreement for neighbor in neighbors)
    relative = tuple(neighbor.relative_norm for neighbor in neighbors)
    loss_delta = tuple(neighbor.loss_delta for neighbor in neighbors)

    if diagnostic is not None:
        candidate_count: int | None = diagnostic.candidate_count
        pruned: bool | None = diagnostic.pruned
        posterior = tuple(
            (tuple(int(value) for value in entry.state), float(entry.probability))
            for entry in diagnostic.posterior
        )
        fallback_reason = diagnostic.fallback_reason
        normalization_mass = diagnostic.normalization_mass
        evidence_bins = tuple(tuple(int(v) for v in row) for row in diagnostic.evidence_bins)
        selection_ns = int(diagnostic.elapsed_ns)
        if tuple(diagnostic.sender_ids) != sender_ids:
            raise CostError(
                f"diagnostic senders do not match observation for node={node} "
                f"round={index}"
            )
    else:
        candidate_count = None
        pruned = None
        posterior = ()
        fallback_reason = None
        normalization_mass = None
        evidence_bins = ()
        selection_ns = 0

    return DecisionAuditRecord(
        node_id=node,
        round_index=index,
        degree=len(neighbors),
        sender_ids=sender_ids,
        disagreement=disagreement,
        relative_norm=relative,
        loss_delta=loss_delta,
        chosen_multipliers=tuple(profile.multipliers),
        weights=tuple(profile.weights),
        candidate_count=candidate_count,
        pruned=pruned,
        posterior=posterior,
        fallback_reason=fallback_reason,
        normalization_mass=normalization_mass,
        evidence_bins=evidence_bins,
        selection_ns=selection_ns,
    )


# --------------------------------------------------------------------------- #
# Cell telemetry facade
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SidecarPaths:
    """Published sidecar locations and content digests for one cell."""

    cost_path: Path
    cost_digest: str
    decision_audit_path: Path
    decision_audit_digest: str


class CellTelemetry:
    """Single opt-in object threading measurement through one cell.

    It is callable so it can be passed as the existing
    ``diagnostic_sink`` of :func:`~dfl_sim.policy.build_policy` without adding a
    policy input. The round engine records observations and timings on it, and
    :func:`~dfl_sim.run.run_from_config` measures the setup/evaluation/whole-cell
    boundaries. Timing values never feed back into any decision.
    """

    enabled = True

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._costs = RecordingCostCollector(clock=clock)
        self._decisions = DecisionAuditCollector()

    # -- clock -------------------------------------------------------------- #
    def now(self) -> int:
        return self._costs.now()

    # -- cost recording ----------------------------------------------------- #
    def add_wall(self, phase: str, elapsed_ns: int) -> None:
        self._costs.add_wall(phase, elapsed_ns)

    def record_round(self, telemetry: RoundTelemetry) -> None:
        self._costs.record_round(telemetry)

    # -- decision recording ------------------------------------------------- #
    def record_observation(
        self, observation: LocalObservation, profile: WeightProfile
    ) -> None:
        self._decisions.record_observation(observation, profile)

    def record_diagnostic(self, diagnostic: BayesianDiagnostic) -> None:
        self._decisions.record_diagnostic(diagnostic)

    def __call__(self, diagnostic: BayesianDiagnostic) -> None:
        """Accept the policy's ``BayesianDiagnostic`` sink callback."""

        self.record_diagnostic(diagnostic)

    # -- lifecycle ---------------------------------------------------------- #
    def begin_cell(self, *, rounds: int, nodes: int) -> None:
        """Declare the exact expected decision-row bound (``rounds * nodes``)."""

        count_rounds = _require_int(rounds, label="rounds", minimum=1)
        count_nodes = _require_int(nodes, label="nodes", minimum=1)
        self._decisions.set_expected_rows(count_rounds * count_nodes)

    @property
    def rounds(self) -> tuple[RoundTelemetry, ...]:
        return self._costs.rounds

    def decision_records(self) -> tuple[DecisionAuditRecord, ...]:
        return self._decisions.records()

    def cost_payload(
        self,
        *,
        identity: Mapping[str, object],
        config_digest: str,
        artifact_digests: Mapping[str, str | None],
        trace_digest: str,
    ) -> dict:
        canonical_identity = validate_measurement_identity(identity)
        canonical_artifacts = validate_artifact_digests(artifact_digests)
        totals = self._costs.totals()
        decisions = self._decisions.records()
        totals["selection_ns"] = sum(row.selection_ns for row in decisions)
        wall = self._costs.wall_ns
        for phase in WALL_PHASES:
            totals[f"{phase}_ns"] = int(wall.get(phase, 0))
        payload = {
            "schema": COST_SCHEMA,
            "schema_version": COST_SCHEMA_VERSION,
            "identity": canonical_identity,
            "measurement": {
                "clock": MEASUREMENT_CLOCK,
                "config_digest": _require_digest(
                    config_digest, label="config_digest"
                ),
                "artifact_digests": canonical_artifacts,
                "trace_digest": _require_digest(
                    trace_digest, label="trace_digest"
                ),
            },
            "totals": totals,
            "rounds": [row.to_payload() for row in self._costs.rounds],
        }
        validate_cost_payload(payload)
        return payload

    def decision_payloads(
        self, *, identity: Mapping[str, object]
    ) -> tuple[dict, ...]:
        canonical_identity = validate_measurement_identity(identity)
        return tuple(
            row.to_payload(identity=canonical_identity)
            for row in self._decisions.records()
        )

    def write_sidecars(
        self,
        directory: str | os.PathLike[str],
        *,
        identity: Mapping[str, object],
        config_digest: str,
        artifact_digests: Mapping[str, str | None],
        trace_digest: str,
    ) -> SidecarPaths:
        """Atomically publish both sidecars for this cell, or raise.

        Nothing is published unless both payloads validate and both files can
        be written and fsynced; a failure aborts the cell (the caller removes
        staging), so a partial cell never yields a "free" baseline or a silently
        zeroed measurement.
        """

        base = Path(os.path.abspath(os.fspath(directory)))
        cost_payload = self.cost_payload(
            identity=identity,
            config_digest=config_digest,
            artifact_digests=artifact_digests,
            trace_digest=trace_digest,
        )
        decision_payloads = self.decision_payloads(identity=identity)
        cost_path, cost_digest = _write_atomic(
            base / COST_SIDECAR_FILENAME, canonical_json_bytes(cost_payload) + b"\n"
        )
        lines = b"".join(
            canonical_json_bytes(payload) + b"\n" for payload in decision_payloads
        )
        audit_path, audit_digest = _write_atomic(
            base / DECISION_AUDIT_SIDECAR_FILENAME, lines
        )
        return SidecarPaths(
            cost_path=cost_path,
            cost_digest=cost_digest,
            decision_audit_path=audit_path,
            decision_audit_digest=audit_digest,
        )


# --------------------------------------------------------------------------- #
# Atomic sidecar publication
# --------------------------------------------------------------------------- #
def _write_atomic(path: Path, payload: bytes) -> tuple[Path, str]:
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CostError(
            f"could not create sidecar directory {parent}: {exc}"
        ) from exc
    try:
        parent_info = os.lstat(parent)
    except OSError as exc:
        raise CostError(f"sidecar parent is unreadable: {parent}") from exc
    if stat.S_ISLNK(parent_info.st_mode):
        raise CostError(f"sidecar parent must not be a symlink: {parent}")
    if not stat.S_ISDIR(parent_info.st_mode):
        raise CostError(f"sidecar parent is not a directory: {parent}")
    if os.path.lexists(path):
        raise CostError(f"sidecar already exists: {path}")

    digest = hashlib.sha256(payload).hexdigest()
    handle = None
    temp_name: str | None = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix=".costs-", dir=str(parent))
        handle = os.fdopen(fd, "wb")
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.rename(temp_name, path)
        temp_name = None
    except OSError as exc:
        raise CostError(f"could not publish sidecar {path}: {exc}") from exc
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:  # pragma: no cover - best-effort cleanup
                pass
        if temp_name is not None and os.path.lexists(temp_name):
            try:
                os.unlink(temp_name)
            except OSError:  # pragma: no cover - best-effort cleanup
                pass
    _fsync_dir(parent)
    return path, digest


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- #
# Strict readers/validators (resume + downstream verification)
# --------------------------------------------------------------------------- #
def validate_cost_payload(payload: object) -> dict:
    """Strictly validate a cost sidecar mapping, rejecting unknown keys/values."""

    if not isinstance(payload, Mapping):
        raise CostError("cost payload must be a JSON object")
    _require_exact_keys(payload, _COST_KEYS, label="cost payload")
    if payload.get("schema") != COST_SCHEMA:
        raise CostError(f"cost schema {payload.get('schema')!r} is not {COST_SCHEMA!r}")
    version = payload.get("schema_version")
    if isinstance(version, bool) or version != COST_SCHEMA_VERSION:
        raise CostError(
            f"cost schema_version {version!r} is not {COST_SCHEMA_VERSION}"
        )
    identity = validate_measurement_identity(payload.get("identity"))  # type: ignore[arg-type]

    measurement = payload.get("measurement")
    if not isinstance(measurement, Mapping):
        raise CostError("cost measurement must be an object")
    _require_exact_keys(measurement, _MEASUREMENT_KEYS, label="cost measurement")
    if measurement.get("clock") != MEASUREMENT_CLOCK:
        raise CostError(
            f"cost clock {measurement.get('clock')!r} is not {MEASUREMENT_CLOCK!r}"
        )
    _require_digest(measurement.get("config_digest"), label="measurement.config_digest")
    validate_artifact_digests(measurement.get("artifact_digests"))  # type: ignore[arg-type]
    _require_digest(measurement.get("trace_digest"), label="measurement.trace_digest")

    totals = payload.get("totals")
    if not isinstance(totals, Mapping):
        raise CostError("cost totals must be an object")
    _require_exact_keys(totals, _TOTALS_KEYS, label="cost totals")
    for key, value in totals.items():
        _require_int(value, label=f"totals.{key}", minimum=0)

    rounds = payload.get("rounds")
    if not isinstance(rounds, list):
        raise CostError("cost rounds must be a list")
    seen_rounds: set[int] = set()
    aggregated_messages = 0
    aggregated_bytes = 0
    aggregated_observation = 0
    aggregated_policy = 0
    aggregated_aggregation = 0
    aggregated_train = 0
    in_degrees: dict[tuple[int, int], int] = {}
    for row in rounds:
        _validate_round_row(row)
        index = int(row["round_index"])
        if index in seen_rounds:
            raise CostError(f"duplicate cost round {index}")
        seen_rounds.add(index)
        aggregated_messages += int(row["directed_messages"])
        aggregated_bytes += int(row["logical_payload_bytes"])
        aggregated_train += int(row["train_ns"])
        for node in row["nodes"]:
            aggregated_observation += int(node["observation_ns"])
            aggregated_policy += int(node["policy_ns"])
            aggregated_aggregation += int(node["aggregation_ns"])
            in_degrees[(int(node["node_id"]), index)] = int(node["degree"])
    if aggregated_messages != totals["directed_messages"]:
        raise CostError("cost totals.directed_messages disagrees with rounds")
    if aggregated_bytes != totals["logical_payload_bytes"]:
        raise CostError("cost totals.logical_payload_bytes disagrees with rounds")
    if aggregated_observation != totals["observation_ns"]:
        raise CostError("cost totals.observation_ns disagrees with rounds")
    if aggregated_policy != totals["policy_ns"]:
        raise CostError("cost totals.policy_ns disagrees with rounds")
    if aggregated_aggregation != totals["aggregation_ns"]:
        raise CostError("cost totals.aggregation_ns disagrees with rounds")
    if aggregated_train != totals["train_ns"]:
        raise CostError("cost totals.train_ns disagrees with rounds")
    if len(rounds) != totals["rounds"]:
        raise CostError("cost totals.rounds disagrees with rounds")
    return dict(payload)


def _validate_round_row(row: object) -> None:
    if not isinstance(row, Mapping):
        raise CostError("cost round row must be an object")
    _require_exact_keys(row, _ROUND_KEYS, label="cost round row")
    _require_int(row["round_index"], label="round_index", minimum=0)
    _require_int(row["undirected_links"], label="undirected_links", minimum=0)
    _require_int(row["directed_messages"], label="directed_messages", minimum=0)
    _require_int(row["logical_payload_bytes"], label="logical_payload_bytes", minimum=0)
    _require_int(row["model_payload_bytes"], label="model_payload_bytes", minimum=0)
    _require_int(row["train_ns"], label="train_ns", minimum=0)
    nodes = row["nodes"]
    if not isinstance(nodes, list):
        raise CostError("cost round nodes must be a list")
    delivered = 0
    delivered_bytes = 0
    seen: set[int] = set()
    for node in nodes:
        if not isinstance(node, Mapping):
            raise CostError("cost node row must be an object")
        _require_exact_keys(node, _NODE_COST_KEYS, label="cost node row")
        node_id = _require_int(node["node_id"], label="node_id", minimum=0)
        if node_id in seen:
            raise CostError(f"duplicate cost node row {node_id}")
        seen.add(node_id)
        if node["round_index"] != row["round_index"]:
            raise CostError("cost node round mismatch")
        degree = _require_int(node["degree"], label="degree", minimum=0)
        messages = _require_int(
            node["delivered_messages"], label="delivered_messages", minimum=0
        )
        if messages != degree:
            raise CostError("delivered_messages must equal degree")
        delivered += messages
        delivered_bytes += _require_int(
            node["delivered_payload_bytes"], label="delivered_payload_bytes", minimum=0
        )
        _require_int(node["observation_ns"], label="observation_ns", minimum=0)
        _require_int(node["policy_ns"], label="policy_ns", minimum=0)
        _require_int(node["aggregation_ns"], label="aggregation_ns", minimum=0)
    if delivered != row["directed_messages"]:
        raise CostError("round directed_messages disagrees with node rows")
    if delivered_bytes != row["logical_payload_bytes"]:
        raise CostError("round logical_payload_bytes disagrees with node rows")


def validate_decision_audit_record(record: object) -> dict:
    """Strictly validate one decision-audit JSONL record."""

    if not isinstance(record, Mapping):
        raise CostError("decision record must be a JSON object")
    _require_exact_keys(record, _DECISION_KEYS, label="decision record")
    if record.get("schema") != DECISION_AUDIT_SCHEMA:
        raise CostError("decision record schema mismatch")
    version = record.get("schema_version")
    if isinstance(version, bool) or version != DECISION_AUDIT_SCHEMA_VERSION:
        raise CostError("decision record schema_version mismatch")
    identity = validate_measurement_identity(record.get("identity"))  # type: ignore[arg-type]
    node = _require_int(record["node_id"], label="node_id", minimum=0)
    _require_int(record["round_index"], label="round_index", minimum=0)
    degree = _require_int(record["degree"], label="degree", minimum=0)
    for key in ("sender_ids", "disagreement", "relative_norm", "loss_delta"):
        value = record[key]
        if not isinstance(value, list) or len(value) != degree:
            raise CostError(f"decision {key} must be a list aligned with degree")
    for key in ("chosen_multipliers", "weights"):
        value = record[key]
        if not isinstance(value, list) or len(value) != degree + 1:
            raise CostError(f"decision {key} must have degree+1 entries")
    for value in record["disagreement"]:
        _require_finite_float(value, label="disagreement")
    for value in record["relative_norm"]:
        _require_finite_float(value, label="relative_norm")
    for value in record["loss_delta"]:
        _require_finite_float(value, label="loss_delta")
    for value in record["chosen_multipliers"]:
        _require_finite_float(value, label="chosen_multipliers", minimum=0.0)
    for value in record["weights"]:
        _require_finite_float(value, label="weights", minimum=0.0)
    if record["candidate_count"] is not None:
        _require_int(record["candidate_count"], label="candidate_count", minimum=0)
    if record["pruned"] is not None:
        _require_bool(record["pruned"], label="pruned")
    posterior = record["posterior"]
    if not isinstance(posterior, list):
        raise CostError("decision posterior must be a list")
    for entry in posterior:
        if not isinstance(entry, Mapping) or set(entry) != {"state", "probability"}:
            raise CostError("decision posterior entry must carry state+probability")
        if not isinstance(entry["state"], list):
            raise CostError("decision posterior state must be a list")
        for value in entry["state"]:
            _require_int(value, label="posterior state entry", minimum=0)
        _require_finite_float(entry["probability"], label="posterior probability", minimum=0.0)
    if record["fallback_reason"] is not None:
        _require_str(record["fallback_reason"], label="fallback_reason")
    if record["normalization_mass"] is not None:
        _require_finite_float(record["normalization_mass"], label="normalization_mass", minimum=0.0)
    evidence = record["evidence_bins"]
    candidate = record["candidate_count"]
    expected_bins = 0 if candidate is None else degree
    if not isinstance(evidence, list) or len(evidence) != expected_bins:
        raise CostError(
            "decision evidence_bins must be aligned with degree for a "
            "Bayesian selection and empty otherwise"
        )
    for row in evidence:
        if not isinstance(row, list) or len(row) != 3:
            raise CostError("decision evidence row must have 3 entries")
        for value in row:
            _require_int(value, label="evidence_bins entry", minimum=0)
    _require_int(record["selection_ns"], label="selection_ns", minimum=0)
    del identity  # validated for shape only
    del node
    return dict(record)


# --------------------------------------------------------------------------- #
# Reading published sidecars
# --------------------------------------------------------------------------- #
def _read_file(path: Path, *, label: str) -> bytes:
    if not isinstance(path, Path):
        path = Path(path)
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise CostError(f"{label} does not exist: {path}") from exc
    except OSError as exc:
        raise CostError(f"{label} is unreadable: {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise CostError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise CostError(f"{label} is not a regular file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CostError(f"{label} could not be read: {path}: {exc}") from exc


def file_digest(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a regular sidecar file (rejects symlinks/non-files)."""

    data = _read_file(Path(path), label="sidecar")
    return hashlib.sha256(data).hexdigest()


def load_cost_payload(path: str | os.PathLike[str]) -> dict:
    """Load and strictly validate one published cost sidecar."""

    target = Path(path)
    data = _read_file(target, label="cost sidecar")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CostError(f"cost sidecar {target} is not valid JSON: {exc}") from exc
    return validate_cost_payload(payload)


def load_decision_audit_records(
    path: str | os.PathLike[str],
) -> tuple[dict, ...]:
    """Load and strictly validate a decision-audit JSONL sidecar."""

    target = Path(path)
    data = _read_file(target, label="decision audit sidecar")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CostError(f"decision audit {target} is not valid UTF-8: {exc}") from exc
    records: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CostError(
                f"decision audit {target} line {lineno} is not valid JSON: {exc}"
            ) from exc
        records.append(validate_decision_audit_record(record))
    return tuple(records)
