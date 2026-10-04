"""Deterministic adaptive attack schedules (R018, D031-D033).

This module owns the *study-level* adaptive attack contract that S02 injects
through the unchanged :class:`~dfl_sim.attacker.OutgoingController` seam. It is
deliberately separate from :class:`~dfl_sim.config.AttackConfig`: an episode
still resolves to a single ``ExperimentConfig`` with one static D016 intensity,
while an adaptive study wraps a schedule around that seam and selects the
per-round intensity without touching the frozen schema-v1 config snapshot or
the trace byte contract.

Design constraints:

* **Strict and frozen.** Every schedule is a pydantic v2 model with
  ``extra="forbid"``, ``strict=True``, ``allow_inf_nan=False``, bounded fields,
  and a discriminated ``kind`` tag. Malformed, unknown, out-of-range, or
  non-finite specs fail closed.
* **Pure.** Each schedule is a deterministic map
  ``(schedule, round_index, total_rounds, seed) -> kappa`` drawn only from the
  D016 :data:`~dfl_sim.config.ATTACK_KAPPAS` values ``{0.0, 2.0, 5.0}``. The
  per-round draw is derived from a domain-separated stable seed, never from a
  mutable running RNG, so repeated calls and process order cannot change the
  realized sequence.
* **No defender feedback.** A schedule observes only the round index and its
  own spec; it never reads the topology, inboxes, evaluator, or any defender
  signal. The held-out schedule is a reserved deterministic non-stationary
  pattern keyed by the domain-separated label ``held_out_schedule`` that the
  selection phase never uses.

Naming note: :func:`build_schedule` mirrors the calibration sampler's helper of
the same name, but the package-level export is aliased to
``build_attack_schedule`` so it never shadows the S01/R013 calibration
``build_schedule`` (which rebuilds the offline reference-episode schedule).
Import the module path directly (``dfl_sim.attack_schedule.build_schedule``)
when the attack-specific factory is intended.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from .config import ATTACK_KAPPAS
from .reproducibility import stable_seed

__all__ = [
    "ATTACK_INTENSITIES",
    "HELD_OUT_SCHEDULE_LABEL",
    "INTENSITY_BY_KAPPA",
    "KAPPA_VALUES",
    "MAX_SCHEDULE_ROUNDS",
    "SCHEDULE_SEED_DOMAIN",
    "DelayedOnsetSchedule",
    "DutyCycleSchedule",
    "HeldOutSchedule",
    "Intensity",
    "PeriodicSchedule",
    "RandomSchedule",
    "ScheduleConfig",
    "ScheduleError",
    "ScheduleKind",
    "ScheduleModel",
    "build_schedule",
    "derive_schedule_seed",
    "intensity_name",
    "kappa_for_round",
    "peak_intensity",
    "realized_intensity_sequence",
    "realized_kappa_sequence",
    "schedule_digest",
]

#: Upper bound on any schedule dimension; keeps one schedule finite/auditable.
MAX_SCHEDULE_ROUNDS = 100_000

#: Domain-separated label that keys the reserved held-out pattern. The selection
#: phase must never build this schedule.
HELD_OUT_SCHEDULE_LABEL = "held_out_schedule"

#: Domain label under which a study derives a per-schedule seed from an episode
#: master seed (``master_seed -> attack_schedule:<kind>``).
SCHEDULE_SEED_DOMAIN = "attack_schedule"

#: Ordered named intensities, matching the D016 table.
ATTACK_INTENSITIES: tuple[str, ...] = ("zero", "weak", "strong")

#: Inverse of the D016 intensity table: kappa value -> named intensity.
INTENSITY_BY_KAPPA: dict[float, str] = {
    float(value): name for name, value in ATTACK_KAPPAS.items()
}

#: The only numeric intensities any schedule may emit (R018 / D016).
KAPPA_VALUES = frozenset(INTENSITY_BY_KAPPA)

Intensity = Literal["zero", "weak", "strong"]
ScheduleKind = Literal[
    "duty_cycle", "periodic", "delayed_onset", "random", "held_out"
]


class ScheduleError(RuntimeError):
    """Raised when an adaptive schedule spec or round query is invalid."""


class ScheduleModel(BaseModel):
    """Base for schedule leaves: unknown keys forbidden, frozen, strict."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class DutyCycleSchedule(ScheduleModel):
    """Repeating on/off duty cycle (default 3 rounds strong, 2 rounds zero)."""

    kind: Literal["duty_cycle"] = "duty_cycle"
    on_rounds: int = Field(default=3, ge=1, le=MAX_SCHEDULE_ROUNDS)
    off_rounds: int = Field(default=2, ge=1, le=MAX_SCHEDULE_ROUNDS)
    intensity: Intensity = "strong"


class PeriodicSchedule(ScheduleModel):
    """Period with leading strong, then weak, then zero rounds.

    Defaults are ``period=6``, ``strong_rounds=2``, ``weak_rounds=2`` so one
    period realizes all three D016 intensities (strong, weak, zero).
    """

    kind: Literal["periodic"] = "periodic"
    period: int = Field(default=6, ge=1, le=MAX_SCHEDULE_ROUNDS)
    strong_rounds: int = Field(default=2, ge=0, le=MAX_SCHEDULE_ROUNDS)
    weak_rounds: int = Field(default=2, ge=0, le=MAX_SCHEDULE_ROUNDS)

    @model_validator(mode="after")
    def _validate_period(self) -> "PeriodicSchedule":
        if self.strong_rounds + self.weak_rounds > self.period:
            raise ValueError(
                "periodic strong_rounds + weak_rounds must be <= period, got "
                f"{self.strong_rounds} + {self.weak_rounds} > {self.period}"
            )
        return self


class DelayedOnsetSchedule(ScheduleModel):
    """Zero intensity until ``onset_round``, then a constant intensity."""

    kind: Literal["delayed_onset"] = "delayed_onset"
    onset_round: int = Field(default=2, ge=0, le=MAX_SCHEDULE_ROUNDS)
    intensity: Intensity = "strong"


class RandomSchedule(ScheduleModel):
    """Per-round Bernoulli on/off draw keyed by a domain-separated seed."""

    kind: Literal["random"] = "random"
    p_on: float = Field(default=0.5, ge=0.0, le=1.0)
    intensity: Intensity = "weak"


class HeldOutSchedule(ScheduleModel):
    """Reserved deterministic non-stationary pattern for the final phase only.

    The pattern cycles through zero/weak/strong with an offset derived from the
    domain-separated ``held_out_schedule`` label, so it is deterministic yet
    never seen during selection.
    """

    kind: Literal["held_out"] = "held_out"


#: Discriminated union on the ``kind`` tag; unknown kinds fail fast.
ScheduleConfig = Annotated[
    Union[
        DutyCycleSchedule,
        PeriodicSchedule,
        DelayedOnsetSchedule,
        RandomSchedule,
        HeldOutSchedule,
    ],
    Field(discriminator="kind"),
]

_SCHEDULE_ADAPTER: TypeAdapter[ScheduleConfig] = TypeAdapter(ScheduleConfig)

#: Pinned D033 defaults applied by :func:`build_schedule`.
_DEFAULT_PARAMS: dict[str, dict[str, object]] = {
    "duty_cycle": {"on_rounds": 3, "off_rounds": 2, "intensity": "strong"},
    "periodic": {"period": 6, "strong_rounds": 2, "weak_rounds": 2},
    "delayed_onset": {"onset_round": 2, "intensity": "strong"},
    "random": {"p_on": 0.5, "intensity": "weak"},
    "held_out": {},
}


def _require_plain_int(value: object, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ScheduleError(
            f"{label} must be an int >= {minimum}, got {value!r}"
        )
    return value


def _as_schedule(schedule: object) -> ScheduleModel:
    if not isinstance(schedule, ScheduleModel):
        raise ScheduleError(
            "schedule must be a validated ScheduleConfig, got "
            f"{type(schedule).__name__}"
        )
    return schedule


def build_schedule(kind: str, **overrides: object) -> ScheduleConfig:
    """Build a schedule for ``kind`` with the pinned D033 defaults.

    ``overrides`` replace individual default fields; unknown keys and
    out-of-range/non-finite values are rejected by the strict contract. Use the
    module path ``dfl_sim.attack_schedule.build_schedule`` when the attack
    factory is intended (the package export is ``build_attack_schedule``).
    """

    if not isinstance(kind, str) or kind not in _DEFAULT_PARAMS:
        raise ScheduleError(
            f"unknown schedule kind {kind!r}; expected one of "
            f"{sorted(_DEFAULT_PARAMS)}"
        )
    payload: dict[str, object] = {"kind": kind, **_DEFAULT_PARAMS[kind]}
    payload.update(overrides)
    try:
        return _SCHEDULE_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise ScheduleError(
            f"invalid {kind!r} schedule: {_bounded(str(exc))}"
        ) from exc


def derive_schedule_seed(master: int, schedule: ScheduleConfig | str) -> int:
    """Derive a stable per-schedule seed from an episode master seed.

    The kind is embedded in the domain label so two schedules in one episode
    draw independent (but reproducible) patterns.
    """

    if isinstance(schedule, ScheduleModel):
        kind = schedule.kind
    elif isinstance(schedule, str):
        kind = schedule
    else:
        raise ScheduleError(
            "schedule must be a ScheduleConfig or a kind string, got "
            f"{type(schedule).__name__}"
        )
    if kind not in _DEFAULT_PARAMS:
        raise ScheduleError(f"unknown schedule kind {kind!r}")
    return stable_seed(int(master), f"{SCHEDULE_SEED_DOMAIN}:{kind}")


def kappa_for_round(
    schedule: ScheduleConfig,
    round_index: int,
    total_rounds: int,
    seed: int = 0,
) -> float:
    """Return the D016 kappa for one round of ``schedule`` (pure map).

    The result is one of :data:`KAPPA_VALUES`; ``total_rounds`` bounds the
    schedule horizon and the derived ``seed`` keys any per-round draw. The same
    arguments always yield the same kappa regardless of call order.
    """

    spec = _as_schedule(schedule)
    index = _require_plain_int(round_index, label="round_index", minimum=0)
    _require_plain_int(total_rounds, label="total_rounds", minimum=1)
    draw_seed = _require_plain_int(seed, label="seed", minimum=0)

    if spec.kind == "duty_cycle":
        cycle = spec.on_rounds + spec.off_rounds
        if (index % cycle) < spec.on_rounds:
            return float(ATTACK_KAPPAS[spec.intensity])
        return 0.0

    if spec.kind == "periodic":
        phase = index % spec.period
        if phase < spec.strong_rounds:
            return float(ATTACK_KAPPAS["strong"])
        if phase < spec.strong_rounds + spec.weak_rounds:
            return float(ATTACK_KAPPAS["weak"])
        return 0.0

    if spec.kind == "delayed_onset":
        if index >= spec.onset_round:
            return float(ATTACK_KAPPAS[spec.intensity])
        return 0.0

    if spec.kind == "random":
        draw = stable_seed(draw_seed, f"{SCHEDULE_SEED_DOMAIN}:random:{index}")
        threshold = int(spec.p_on * (2**31))
        if draw < threshold:
            return float(ATTACK_KAPPAS[spec.intensity])
        return 0.0

    if spec.kind == "held_out":
        offset = stable_seed(draw_seed, HELD_OUT_SCHEDULE_LABEL) % 3
        ordered = (
            0.0,
            float(ATTACK_KAPPAS["weak"]),
            float(ATTACK_KAPPAS["strong"]),
        )
        return ordered[(offset + index) % 3]

    raise ScheduleError(f"unsupported schedule kind {spec.kind!r}")  # pragma: no cover


def realized_kappa_sequence(
    schedule: ScheduleConfig, total_rounds: int, seed: int = 0
) -> tuple[float, ...]:
    """Return the full per-round kappa sequence for ``schedule``."""

    rounds = _require_plain_int(total_rounds, label="total_rounds", minimum=1)
    draw_seed = _require_plain_int(seed, label="seed", minimum=0)
    return tuple(
        kappa_for_round(schedule, index, rounds, draw_seed)
        for index in range(rounds)
    )


def intensity_name(kappa: float) -> str:
    """Return the named D016 intensity for one of :data:`KAPPA_VALUES`."""

    try:
        return INTENSITY_BY_KAPPA[float(kappa)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ScheduleError(
            f"kappa must be one of {sorted(KAPPA_VALUES)}, got {kappa!r}"
        ) from exc


def realized_intensity_sequence(
    schedule: ScheduleConfig, total_rounds: int, seed: int = 0
) -> tuple[str, ...]:
    """Return the named per-round intensity sequence for ``schedule``."""

    return tuple(
        intensity_name(value)
        for value in realized_kappa_sequence(schedule, total_rounds, seed)
    )


def peak_intensity(
    schedule: ScheduleConfig, total_rounds: int, seed: int = 0
) -> str:
    """Return the highest realized intensity name (the D031 attack provenance)."""

    sequence = realized_kappa_sequence(schedule, total_rounds, seed)
    return intensity_name(max(sequence))


def schedule_digest(schedule: ScheduleConfig) -> str:
    """Canonical SHA-256 over the environment-independent schedule payload."""

    spec = _as_schedule(schedule)
    payload = spec.model_dump(mode="json")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bounded(detail: str, *, limit: int = 300) -> str:
    collapsed = " ".join(str(detail).split())
    if len(collapsed) > limit:
        collapsed = collapsed[:limit] + "..."
    return collapsed
