"""Contract tests for deterministic adaptive attack schedules (R018, D031-D033).

These tests exercise the schedule contract and the schedule-driven
:class:`~dfl_sim.attacker.AdaptiveAttackController` in isolation from the round
engine: pinned defaults, pure per-round kappa maps, collective zero/weak/strong
coverage, real per-round variation, exact no-op semantics, malformed-state and
malformed-spec rejection, immutability, and the absence of any privileged call
surface. Nothing here reads files or the network; every payload is generated
in-process.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

from dfl_sim import (
    ATTACK_KAPPAS,
    AdaptiveAttackController,
    AttackError,
    DelayedOnsetSchedule,
    DutyCycleSchedule,
    HeldOutSchedule,
    OutgoingController,
    PeriodicSchedule,
    RandomSchedule,
    ScheduleConfig,
    ScheduleError,
    ScheduleModel,
    build_attack_schedule,
    derive_schedule_seed,
    intensity_name,
    kappa_for_round,
    peak_intensity,
    realized_intensity_sequence,
    realized_kappa_sequence,
    schedule_digest,
)
from dfl_sim.attack_schedule import HELD_OUT_SCHEDULE_LABEL
from dfl_sim.attack_schedule import build_schedule as module_build_schedule

KAPPAS = frozenset(float(value) for value in ATTACK_KAPPAS.values())
ZERO = float(ATTACK_KAPPAS["zero"])
WEAK = float(ATTACK_KAPPAS["weak"])
STRONG = float(ATTACK_KAPPAS["strong"])

#: Fixture horizon long enough to realize every preset's full period.
FIXTURE_ROUNDS = 12


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _kinds() -> dict[str, ScheduleConfig]:
    return {
        "duty_cycle": build_attack_schedule("duty_cycle"),
        "periodic": build_attack_schedule("periodic"),
        "delayed_onset": build_attack_schedule("delayed_onset"),
        "random": build_attack_schedule("random"),
    }


def _pair(seed: int = 0) -> tuple[dict, dict]:
    generator = torch.Generator().manual_seed(seed)
    round_start = {
        "weight": torch.randn(4, 3, generator=generator, dtype=torch.float64),
        "bias": torch.randn(4, generator=generator, dtype=torch.float64),
    }
    trained = {
        "weight": torch.randn(4, 3, generator=generator, dtype=torch.float64),
        "bias": torch.randn(4, generator=generator, dtype=torch.float64),
    }
    return round_start, trained


def _controller(
    *,
    kind: str = "duty_cycle",
    node: int | None = 0,
    total_rounds: int = FIXTURE_ROUNDS,
    master_seed: int = 1234,
    **overrides: object,
) -> AdaptiveAttackController:
    schedule = build_attack_schedule(kind, **overrides)
    return AdaptiveAttackController.from_episode(
        compromised_node=node,
        schedule=schedule,
        total_rounds=total_rounds,
        master_seed=master_seed,
    )


def _snapshot(state: dict) -> dict:
    return {key: value.clone() for key, value in state.items()}


# --------------------------------------------------------------------------- #
# Exports and namespace integrity
# --------------------------------------------------------------------------- #
def test_attack_factory_is_aliased_and_calibration_schedule_is_preserved() -> None:
    # The attack factory must not shadow the S01/R013 calibration builder.
    import dfl_sim
    from dfl_sim.calibration import build_schedule as calibration_build_schedule

    assert dfl_sim.build_attack_schedule is module_build_schedule
    assert dfl_sim.build_schedule is calibration_build_schedule
    assert dfl_sim.build_schedule is not module_build_schedule


def test_schedule_summary_and_calibration_schedule_names_still_exported() -> None:
    import dfl_sim

    assert "ScheduleSummary" in dfl_sim.__all__
    assert "CalibrationSchedule" in dfl_sim.__all__
    assert dfl_sim.ScheduleSummary is not None
    assert dfl_sim.CalibrationSchedule is not None


# --------------------------------------------------------------------------- #
# Pinned defaults (D033)
# --------------------------------------------------------------------------- #
def test_pinned_defaults() -> None:
    duty = build_attack_schedule("duty_cycle")
    assert isinstance(duty, DutyCycleSchedule)
    assert (duty.on_rounds, duty.off_rounds, duty.intensity) == (3, 2, "strong")

    periodic = build_attack_schedule("periodic")
    assert isinstance(periodic, PeriodicSchedule)
    assert (periodic.period, periodic.strong_rounds, periodic.weak_rounds) == (
        6,
        2,
        2,
    )

    delayed = build_attack_schedule("delayed_onset")
    assert isinstance(delayed, DelayedOnsetSchedule)
    assert (delayed.onset_round, delayed.intensity) == (2, "strong")

    random = build_attack_schedule("random")
    assert isinstance(random, RandomSchedule)
    assert (random.p_on, random.intensity) == (0.5, "weak")

    held = build_attack_schedule("held_out")
    assert isinstance(held, HeldOutSchedule)
    assert held.kind == "held_out"
    assert HELD_OUT_SCHEDULE_LABEL == "held_out_schedule"


def test_default_schema_round_trips_through_model_validate() -> None:
    duty = build_attack_schedule("duty_cycle")
    rebuilt = TypeAdapter(ScheduleConfig).validate_python(
        duty.model_dump(mode="json")
    )
    assert rebuilt == duty
    assert schedule_digest(rebuilt) == schedule_digest(duty)


# --------------------------------------------------------------------------- #
# Pure deterministic kappa maps (R018)
# --------------------------------------------------------------------------- #
def test_presets_collectively_realize_zero_weak_and_strong() -> None:
    realized: set[float] = set()
    for schedule in _kinds().values():
        realized.update(realized_kappa_sequence(schedule, FIXTURE_ROUNDS, seed=7))
    assert realized == {ZERO, WEAK, STRONG}
    assert realized <= KAPPAS


def test_held_out_schedule_realizes_all_three_and_is_nonstationary() -> None:
    sequence = realized_kappa_sequence(
        build_attack_schedule("held_out"), FIXTURE_ROUNDS, seed=99
    )
    assert set(sequence) == {ZERO, WEAK, STRONG}
    assert len(set(sequence)) == 3
    assert len(set(sequence)) > 1


def test_duty_cycle_expected_pattern() -> None:
    sequence = realized_kappa_sequence(
        build_attack_schedule("duty_cycle"), 10, seed=0
    )
    assert sequence == (
        STRONG,
        STRONG,
        STRONG,
        ZERO,
        ZERO,
        STRONG,
        STRONG,
        STRONG,
        ZERO,
        ZERO,
    )


def test_periodic_expected_pattern() -> None:
    sequence = realized_kappa_sequence(
        build_attack_schedule("periodic"), 6, seed=0
    )
    assert sequence == (STRONG, STRONG, WEAK, WEAK, ZERO, ZERO)


def test_delayed_onset_expected_pattern() -> None:
    sequence = realized_kappa_sequence(
        build_attack_schedule("delayed_onset"), 5, seed=0
    )
    assert sequence == (ZERO, ZERO, STRONG, STRONG, STRONG)


def test_every_preset_varies_per_round_not_a_constant_relabel() -> None:
    for name, schedule in _kinds().items():
        sequence = realized_kappa_sequence(schedule, FIXTURE_ROUNDS, seed=3)
        assert len(set(sequence)) > 1, f"{name} is constant: {sequence}"
    held = realized_kappa_sequence(
        build_attack_schedule("held_out"), FIXTURE_ROUNDS, seed=3
    )
    assert len(set(held)) > 1


def test_sequence_is_order_independent() -> None:
    schedule = build_attack_schedule("random", p_on=0.5)
    expected = realized_kappa_sequence(schedule, FIXTURE_ROUNDS, seed=42)
    shuffled = list(reversed(range(FIXTURE_ROUNDS)))
    recomputed = {
        index: kappa_for_round(schedule, index, FIXTURE_ROUNDS, 42)
        for index in shuffled
    }
    assert tuple(recomputed[index] for index in range(FIXTURE_ROUNDS)) == expected


def test_kappa_for_round_repeated_calls_are_equal() -> None:
    schedule = build_attack_schedule("held_out")
    first = kappa_for_round(schedule, 5, FIXTURE_ROUNDS, 11)
    second = kappa_for_round(schedule, 5, FIXTURE_ROUNDS, 11)
    assert first == second
    assert first in KAPPAS


def test_random_boundaries_always_on_and_always_off() -> None:
    always = realized_kappa_sequence(
        build_attack_schedule("random", p_on=1.0), 8, seed=1
    )
    never = realized_kappa_sequence(
        build_attack_schedule("random", p_on=0.0), 8, seed=1
    )
    assert always == (WEAK,) * 8
    assert never == (ZERO,) * 8


def test_random_seed_changes_the_draw() -> None:
    schedule = build_attack_schedule("random")
    first = realized_kappa_sequence(schedule, 24, seed=1)
    second = realized_kappa_sequence(schedule, 24, seed=2)
    assert first != second


def test_sequence_length_matches_total_rounds() -> None:
    schedule = build_attack_schedule("periodic")
    assert len(realized_kappa_sequence(schedule, 1, seed=0)) == 1
    assert len(realized_kappa_sequence(schedule, 17, seed=0)) == 17


def test_derive_schedule_seed_is_deterministic_and_domain_separated() -> None:
    duty = build_attack_schedule("duty_cycle")
    periodic = build_attack_schedule("periodic")
    assert derive_schedule_seed(5, duty) == derive_schedule_seed(5, "duty_cycle")
    assert derive_schedule_seed(5, duty) != derive_schedule_seed(5, periodic)
    assert derive_schedule_seed(5, duty) != derive_schedule_seed(6, duty)


def test_intensity_and_peak_helpers() -> None:
    assert intensity_name(ZERO) == "zero"
    assert intensity_name(WEAK) == "weak"
    assert intensity_name(STRONG) == "strong"
    assert peak_intensity(build_attack_schedule("duty_cycle"), FIXTURE_ROUNDS) == (
        "strong"
    )
    assert realized_intensity_sequence(
        build_attack_schedule("delayed_onset"), 4, seed=0
    ) == ("zero", "zero", "strong", "strong")
    with pytest.raises(ScheduleError):
        intensity_name(1.0)


# --------------------------------------------------------------------------- #
# Canonical digest
# --------------------------------------------------------------------------- #
def test_schedule_digest_is_stable_for_equal_specs() -> None:
    first = build_attack_schedule("periodic")
    second = build_attack_schedule("periodic")
    assert schedule_digest(first) == schedule_digest(second)
    assert len(schedule_digest(first)) == 64


def test_schedule_digest_changes_with_any_field() -> None:
    base = build_attack_schedule("duty_cycle")
    changed = build_attack_schedule("duty_cycle", on_rounds=4)
    assert schedule_digest(base) != schedule_digest(changed)
    assert schedule_digest(
        build_attack_schedule("duty_cycle", intensity="weak")
    ) != schedule_digest(base)


def test_schedule_digest_rejects_unvalidated_input() -> None:
    with pytest.raises(ScheduleError):
        schedule_digest({"kind": "duty_cycle"})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Controller: protocol, per-round selection, exact math
# --------------------------------------------------------------------------- #
def test_controller_implements_outgoing_protocol() -> None:
    controller = _controller()
    assert isinstance(controller, OutgoingController)


def test_controller_selects_kappa_per_round() -> None:
    controller = _controller(kind="duty_cycle")
    assert controller.kappa_for(0) == STRONG
    assert controller.kappa_for(2) == STRONG
    assert controller.kappa_for(3) == ZERO
    assert controller.kappa_for(5) == STRONG


def test_controller_peak_kappa_matches_schedule() -> None:
    controller = _controller(kind="periodic")
    assert controller.peak_kappa == STRONG


def test_controller_exact_no_op_at_zero_kappa() -> None:
    round_start, trained = _pair(seed=2)
    controller = _controller(kind="duty_cycle")  # round 3 is off
    assert controller.kappa_for(3) == ZERO

    outgoing = controller.outgoing_state(
        sender=0, round_start=round_start, trained=trained, round_index=3
    )
    for key in trained:
        assert torch.equal(outgoing[key], trained[key])
        assert outgoing[key].data_ptr() != trained[key].data_ptr()


def test_controller_applies_exact_formula_at_peak_round() -> None:
    round_start, trained = _pair(seed=7)
    controller = _controller(kind="duty_cycle")  # round 0 is strong
    outgoing = controller.outgoing_state(
        sender=0, round_start=round_start, trained=trained, round_index=0
    )
    for key in trained:
        expected = trained[key] - STRONG * (trained[key] - round_start[key])
        assert torch.equal(outgoing[key], expected)
        assert not torch.equal(outgoing[key], trained[key])


def test_controller_honest_node_never_poisoned_on_attack_round() -> None:
    round_start, trained = _pair(seed=4)
    controller = _controller(kind="duty_cycle", node=0)
    outgoing = controller.outgoing_state(
        sender=1, round_start=round_start, trained=trained, round_index=0
    )
    for key in trained:
        assert torch.equal(outgoing[key], trained[key])


def test_controller_without_compromised_node_never_poisons() -> None:
    round_start, trained = _pair(seed=4)
    controller = _controller(kind="duty_cycle", node=None)
    assert controller.kappa_for(0) == ZERO
    outgoing = controller.outgoing_state(
        sender=0, round_start=round_start, trained=trained, round_index=0
    )
    for key in trained:
        assert torch.equal(outgoing[key], trained[key])


def test_controller_integer_tensors_are_copied_unchanged() -> None:
    round_start, trained = _pair(seed=1)
    round_start["counts"] = torch.tensor([1, 2, 3], dtype=torch.int64)
    trained["counts"] = torch.tensor([4, 5, 6], dtype=torch.int64)
    controller = _controller(kind="duty_cycle")
    outgoing = controller.outgoing_state(
        sender=0, round_start=round_start, trained=trained, round_index=0
    )
    assert outgoing["counts"].dtype == torch.int64
    assert torch.equal(outgoing["counts"], trained["counts"])


def test_controller_inputs_are_never_mutated() -> None:
    round_start, trained = _pair(seed=11)
    round_before = _snapshot(round_start)
    trained_before = _snapshot(trained)
    _controller().outgoing_state(
        sender=0, round_start=round_start, trained=trained, round_index=0
    )
    for key in round_before:
        assert torch.equal(round_start[key], round_before[key])
        assert torch.equal(trained[key], trained_before[key])


def test_controller_repeated_calls_are_bitwise_deterministic() -> None:
    round_start, trained = _pair(seed=3)
    controller = _controller(kind="periodic")
    first = controller.outgoing_state(
        sender=0, round_start=round_start, trained=trained, round_index=0
    )
    second = controller.outgoing_state(
        sender=0, round_start=round_start, trained=trained, round_index=0
    )
    for key in trained:
        assert torch.equal(first[key], second[key])
        assert first[key].data_ptr() != second[key].data_ptr()


# --------------------------------------------------------------------------- #
# Controller call surface: no privileged signal
# --------------------------------------------------------------------------- #
def test_controller_call_surface_is_narrow() -> None:
    params = list(inspect.signature(
        AdaptiveAttackController.outgoing_state
    ).parameters)
    assert params == [
        "self",
        "sender",
        "round_start",
        "trained",
        "round_index",
        "context",
    ]
    for keyword in inspect.signature(
        AdaptiveAttackController.outgoing_state
    ).parameters.values():
        assert keyword.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )


def test_controller_exposes_no_privileged_channel_attributes() -> None:
    controller = _controller()
    for forbidden in (
        "topology",
        "evaluator",
        "inbox",
        "inboxes",
        "defender_feedback",
        "other_inboxes",
        "global_metrics",
    ):
        assert not hasattr(controller, forbidden)


def test_controller_rejects_unknown_call_arguments() -> None:
    round_start, trained = _pair()
    controller = _controller()
    with pytest.raises(TypeError):
        controller.outgoing_state(  # type: ignore[call-arg]
            sender=0,
            round_start=round_start,
            trained=trained,
            round_index=0,
            topology={"x": 1},  # type: ignore[call-arg]
        )


def test_controller_requires_a_round_index() -> None:
    round_start, trained = _pair()
    with pytest.raises(AttackError, match="requires a round_index"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained=trained
        )


# --------------------------------------------------------------------------- #
# Controller validation (Q7)
# --------------------------------------------------------------------------- #
def test_controller_rejects_negative_round_index() -> None:
    round_start, trained = _pair()
    with pytest.raises(AttackError, match="round_index"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained=trained, round_index=-1
        )


def test_controller_rejects_invalid_construction() -> None:
    schedule = build_attack_schedule("duty_cycle")
    with pytest.raises(AttackError, match="total_rounds"):
        AdaptiveAttackController(
            compromised_node=0, schedule=schedule, total_rounds=0
        )
    with pytest.raises(AttackError, match="schedule_seed"):
        AdaptiveAttackController(
            compromised_node=0,
            schedule=schedule,
            total_rounds=4,
            schedule_seed=-1,
        )
    with pytest.raises(AttackError, match="compromised_node"):
        AdaptiveAttackController(
            compromised_node=-2, schedule=schedule, total_rounds=4
        )
    with pytest.raises(AttackError, match="schedule must be"):
        AdaptiveAttackController(
            compromised_node=0,
            schedule={"kind": "duty_cycle"},  # type: ignore[arg-type]
            total_rounds=4,
        )


def test_controller_empty_round_start_rejected() -> None:
    _, trained = _pair()
    with pytest.raises(AttackError, match="round-start state is empty"):
        _controller().outgoing_state(
            sender=0, round_start={}, trained=trained, round_index=0
        )


def test_controller_empty_trained_rejected() -> None:
    round_start, _ = _pair()
    with pytest.raises(AttackError, match="trained state is empty"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained={}, round_index=0
        )


def test_controller_key_mismatch_rejected() -> None:
    round_start, trained = _pair()
    del trained["bias"]
    trained["extra"] = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(AttackError, match="key mismatch"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained=trained, round_index=0
        )


def test_controller_shape_mismatch_rejected() -> None:
    round_start, trained = _pair()
    trained["weight"] = trained["weight"][:2]
    with pytest.raises(AttackError, match="shape mismatch"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained=trained, round_index=0
        )


def test_controller_dtype_mismatch_rejected() -> None:
    round_start, trained = _pair()
    trained["weight"] = trained["weight"].to(torch.float32)
    with pytest.raises(AttackError, match="dtype mismatch"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained=trained, round_index=0
        )


def test_controller_non_tensor_entry_rejected() -> None:
    round_start, trained = _pair()
    trained["weight"] = [1.0, 2.0]
    with pytest.raises(AttackError, match="must be a tensor"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained=trained, round_index=0
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_controller_non_finite_state_rejected(bad: float) -> None:
    round_start, trained = _pair()
    round_start["weight"] = round_start["weight"].clone()
    round_start["weight"][0, 0] = bad
    with pytest.raises(AttackError, match="non-finite"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained=trained, round_index=0
        )


def test_controller_error_context_includes_sender_and_round() -> None:
    round_start, trained = _pair()
    trained["weight"] = trained["weight"][:2]
    with pytest.raises(AttackError) as excinfo:
        _controller().outgoing_state(
            sender=3,
            round_start=round_start,
            trained=trained,
            round_index=2,
            context="cell-x",
        )
    message = str(excinfo.value)
    assert "sender=3" in message
    assert "round=2" in message
    assert "cell-x" in message


# --------------------------------------------------------------------------- #
# Immutability
# --------------------------------------------------------------------------- #
def test_schedule_models_are_frozen() -> None:
    schedule = build_attack_schedule("duty_cycle")
    with pytest.raises(Exception):
        schedule.on_rounds = 9  # type: ignore[misc]


def test_controller_is_frozen() -> None:
    controller = _controller()
    with pytest.raises(Exception):
        controller.schedule_seed = 5  # type: ignore[misc]


def test_schedule_classes_share_a_strict_base() -> None:
    for model in (
        DutyCycleSchedule,
        PeriodicSchedule,
        DelayedOnsetSchedule,
        RandomSchedule,
        HeldOutSchedule,
    ):
        assert issubclass(model, ScheduleModel)


# --------------------------------------------------------------------------- #
# Malformed schedule specs (Q7)
# --------------------------------------------------------------------------- #
def test_build_schedule_rejects_unknown_kind() -> None:
    with pytest.raises(ScheduleError, match="unknown schedule kind"):
        build_attack_schedule("not_a_schedule")


def test_build_schedule_rejects_out_of_range_fields() -> None:
    with pytest.raises(ScheduleError):
        build_attack_schedule("duty_cycle", on_rounds=0)
    with pytest.raises(ScheduleError):
        build_attack_schedule("delayed_onset", onset_round=-1)
    with pytest.raises(ScheduleError):
        build_attack_schedule("random", p_on=1.5)
    with pytest.raises(ScheduleError):
        build_attack_schedule("random", p_on=-0.1)
    with pytest.raises(ScheduleError):
        build_attack_schedule("periodic", period=3, strong_rounds=2, weak_rounds=2)


def test_build_schedule_rejects_unknown_keys_and_wrong_types() -> None:
    with pytest.raises(ScheduleError):
        build_attack_schedule("duty_cycle", surprise=1)
    with pytest.raises(ScheduleError):
        build_attack_schedule("duty_cycle", on_rounds="3")
    with pytest.raises(ScheduleError):
        build_attack_schedule("duty_cycle", intensity="peak")


def test_direct_schedule_validation_rejects_malformed_configs() -> None:
    adapter = TypeAdapter(ScheduleConfig)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "unknown_kind"})
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "duty_cycle", "on_rounds": 0})
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "duty_cycle", "extra_field": 1})
    with pytest.raises(ValidationError):
        DutyCycleSchedule(on_rounds=0)
    with pytest.raises(ValidationError):
        PeriodicSchedule(period=2, strong_rounds=2, weak_rounds=2)


def test_kappa_for_round_rejects_bad_scalars() -> None:
    schedule = build_attack_schedule("duty_cycle")
    with pytest.raises(ScheduleError):
        kappa_for_round(schedule, -1, FIXTURE_ROUNDS, 0)
    with pytest.raises(ScheduleError):
        kappa_for_round(schedule, 0, 0, 0)
    with pytest.raises(ScheduleError):
        kappa_for_round(schedule, 0, FIXTURE_ROUNDS, -1)
    with pytest.raises(ScheduleError):
        kappa_for_round({"kind": "duty_cycle"}, 0, 1, 0)  # type: ignore[arg-type]
