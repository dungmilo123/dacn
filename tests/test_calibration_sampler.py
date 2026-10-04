"""Deterministic calibration-sampler and prior-count tests (T01, R013/R014).

Offline and deterministic: the schedule and the topology are pure functions of
the validated config, so no dataset, tensor, or network is involved. The
builder is reused from the sibling config-contract test module.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from dfl_sim import (
    CalibrationError,
    CalibrationSchedule,
    EpisodeSpec,
    build_schedule,
    build_schedule_and_counts,
    count_prior_states,
    feasible_prior_states,
    summarize_schedule,
)
from test_calibration_config import base_dict, parse_dict

PINNED_VALUES = {0.0, 2.0, 5.0}


def make_schedule(**overrides) -> CalibrationSchedule:
    return build_schedule(parse_dict(base_dict(**overrides)))


# --------------------------------------------------------------------------- #
# Stratified counts and clamping
# --------------------------------------------------------------------------- #
def test_even_split_builds_both_strata() -> None:
    schedule = make_schedule(episodes=8, p0=0.5)
    summary = schedule.summary
    assert summary.episodes == 8
    assert summary.clean == 4
    assert summary.attacked == 4
    assert len(schedule.specs) == 8
    assert {spec.stratum for spec in schedule.specs} == {"clean", "attacked"}


@pytest.mark.parametrize(
    "episodes,p0,clean,attacked",
    [
        (3, 0.0, 1, 2),
        (3, 1.0, 2, 1),
        (2, 0.9, 1, 1),
        (2, 0.1, 1, 1),
        (5, 0.5, 2, 3),
    ],
)
def test_rounding_is_clamped_to_both_strata(
    episodes: int, p0: float, clean: int, attacked: int
) -> None:
    summary = make_schedule(episodes=episodes, p0=p0).summary
    assert summary.clean == clean
    assert summary.attacked == attacked


def test_indices_are_contiguous_and_ordered() -> None:
    specs = make_schedule(episodes=6, p0=0.5).specs
    assert [spec.index for spec in specs] == list(range(6))


# --------------------------------------------------------------------------- #
# Balanced mixture and uniform placement
# --------------------------------------------------------------------------- #
def test_attacked_mixture_is_balanced() -> None:
    schedule = make_schedule(episodes=11, p0=0.5)
    counts = dict(schedule.summary.intensity_counts)
    assert schedule.summary.clean == 6
    assert schedule.summary.attacked == 5
    assert set(counts) <= {"zero", "weak", "strong"}
    assert sum(counts.values()) == 5
    assert max(counts.values()) - min(counts.values()) <= 1


def test_attack_values_are_pinned() -> None:
    schedule = make_schedule(episodes=9, p0=0.5)
    for spec in schedule.specs:
        assert spec.value in PINNED_VALUES
        if spec.stratum == "clean":
            assert spec.node is None
            assert spec.intensity == "zero"
            assert spec.value == 0.0
        else:
            assert spec.node is not None
            assert spec.intensity in {"zero", "weak", "strong"}


def test_placement_covers_non_isolated_nodes_uniformly() -> None:
    # chain3 has three non-isolated nodes; with six attacked episodes the
    # cycling placement covers every node exactly twice.
    schedule = make_schedule(episodes=12, p0=0.5)
    placements = dict(schedule.summary.placement_counts)
    assert schedule.summary.attacked == 6
    assert set(placements) == {0, 1, 2}
    assert sum(placements.values()) == 6
    assert max(placements.values()) - min(placements.values()) <= 1


def test_compromised_node_within_topology() -> None:
    for spec in make_schedule(episodes=8, p0=0.5).specs:
        if spec.node is not None:
            assert 0 <= spec.node < 3


# --------------------------------------------------------------------------- #
# Determinism and stable episode seeds
# --------------------------------------------------------------------------- #
def test_schedule_is_deterministic() -> None:
    config = parse_dict(base_dict(episodes=8, p0=0.5))
    first = build_schedule(config)
    second = build_schedule(config)
    assert first.specs == second.specs
    assert first.summary == second.summary


def test_experiment_seed_changes_schedule() -> None:
    from helpers import make_config

    config_a = parse_dict(
        base_dict(experiment=make_config(seed=1).model_dump(mode="json"))
    )
    config_b = parse_dict(
        base_dict(experiment=make_config(seed=2).model_dump(mode="json"))
    )
    assert build_schedule(config_a).specs != build_schedule(config_b).specs


def test_episode_seeds_are_stable_and_distinct() -> None:
    specs = make_schedule(episodes=8, p0=0.5).specs
    seeds = [spec.seed for spec in specs]
    assert len(seeds) == len(set(seeds))
    assert all(0 <= seed < 2**31 for seed in seeds)
    assert seeds == [spec.seed for spec in make_schedule(episodes=8, p0=0.5).specs]


def test_summarize_schedule_accepts_plain_sequence() -> None:
    specs = make_schedule(episodes=4, p0=0.5).specs
    assert summarize_schedule(list(specs)) == summarize_schedule(specs)


# --------------------------------------------------------------------------- #
# Immutability / no tensors
# --------------------------------------------------------------------------- #
def test_specs_are_frozen_and_tensor_free() -> None:
    specs = make_schedule(episodes=4, p0=0.5).specs
    for spec in specs:
        assert isinstance(spec, EpisodeSpec)
        with pytest.raises(Exception):
            spec.index = 999  # type: ignore[misc]
        for field in dataclasses.fields(spec):
            assert not isinstance(getattr(spec, field.name), torch.Tensor)


def test_build_schedule_rejects_unvalidated_type() -> None:
    with pytest.raises(CalibrationError, match="CalibrationConfig"):
        build_schedule({"episodes": 4})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Prior-count inputs
# --------------------------------------------------------------------------- #
def test_feasible_prior_states_cover_at_most_one_malicious() -> None:
    assert feasible_prior_states(0) == ((),)
    assert feasible_prior_states(1) == ((0,), (1,))
    assert feasible_prior_states(2) == ((0, 0), (1, 0), (0, 1))
    assert feasible_prior_states(3) == (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    )


def test_feasible_prior_states_rejects_negative() -> None:
    with pytest.raises(CalibrationError):
        feasible_prior_states(-1)


def test_prior_counts_preserve_sender_order_and_support() -> None:
    config = parse_dict(base_dict(episodes=8, p0=0.5))
    schedule, counts = build_schedule_and_counts(config)
    assert counts.episodes == 8

    # chain3: 4 clean eps x 3 benign receivers + 4 attacked x 2 benign = 20.
    assert counts.benign_observations == 20
    assert set(counts.counts) == {1, 2}

    total = 0
    for degree, states in counts.counts.items():
        feasible = set(feasible_prior_states(degree))
        for state, count in states.items():
            assert len(state) == degree
            assert state in feasible
            assert sum(state) <= 1
            assert count >= 1
            total += count
    assert total == counts.benign_observations


def test_prior_counts_condition_on_benign_receiver() -> None:
    # With a single attacked episode, the compromised node itself is never
    # counted as a receiver; only its two benign peers contribute.
    config = parse_dict(base_dict(episodes=2, p0=0.0))  # clean clamped to 1
    schedule = build_schedule(config)
    counts = count_prior_states(config, schedule.specs)
    assert schedule.summary.clean == 1
    assert schedule.summary.attacked == 1
    # chain3 has no isolated nodes: 1 clean * 3 benign receivers plus the single
    # attacked episode's 2 benign peers = 5 observations.
    assert counts.benign_observations == 5


def test_count_prior_states_rejects_unvalidated_type() -> None:
    with pytest.raises(CalibrationError, match="CalibrationConfig"):
        count_prior_states({"episodes": 2}, ())  # type: ignore[arg-type]
