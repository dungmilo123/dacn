"""Offline reference-episode collection tests (T02, R013/R014).

Deterministic in-process MNIST-shaped dataset and the real
:func:`~dfl_sim.run.run_from_config` round engine with the plain-average
reference policy. No files, network, or final-test loader. These tests prove
clean and attacked episodes traverse the real engine, only plain averaging
advances the trajectory, the design split is provably disjoint from every
node-local partition, records are tensor-free/deterministic, and failures
abort before any record is fabricated.
"""

from __future__ import annotations

import torch

import pytest

from dfl_sim import (
    NO_BELIEF_VARIANT,
    NO_STRATEGIC_VARIANT,
    CalibrationScoringError,
    CalibrationError,
    CalibrationObservationBuilder,
    CalibrationCollection,
    DataError,
    collect_calibration,
    collect_episode,
    derive_experiment_config,
    partition_indices,
    resolve_seeds,
)
from dfl_sim import calibration_scoring as cs
from helpers import MNISTShapedDataset, make_config
from test_calibration_config import base_dict, parse_dict


def _make_calibration_config(**overrides):
    experiment = make_config(
        rounds=1,
        training={"local_epochs": 1, "batch_size": 8},
        data={"samples_per_node": 8},
        evaluation={"max_samples": 8, "batch_size": 8},
    )
    kwargs = {
        "experiment": experiment.model_dump(mode="json"),
        "episodes": 2,
        "p0": 0.5,
        "design_split_size": 8,
    }
    kwargs.update(overrides)
    return parse_dict(base_dict(**kwargs))


def _dataset(size: int = 200):
    return MNISTShapedDataset(size, separable=True, seed=11)


# --------------------------------------------------------------------------- #
# Clean + attacked collection through the real engine
# --------------------------------------------------------------------------- #
def test_collects_both_strata_through_the_real_engine() -> None:
    config = _make_calibration_config()
    collection = collect_calibration(config, dataset=_dataset())
    assert isinstance(collection, CalibrationCollection)
    assert collection.episode_count == 2
    assert {episode.spec.stratum for episode in collection.episodes} == {
        "clean",
        "attacked",
    }
    # chain3 has three connected nodes and one round -> three records each.
    for episode in collection.episodes:
        assert len(episode.records) == 3
        assert {record.node_id for record in episode.records} == {0, 1, 2}
    assert len(collection.records) == 6
    assert collection.round_count == 6


def test_candidate_counts_match_degree_and_action_space() -> None:
    config = _make_calibration_config()
    collection = collect_calibration(config, dataset=_dataset())
    multiplier_count = len(config.action_space.multiplier_set)
    for record in collection.records:
        assert len(record.candidates) == multiplier_count ** record.degree
        assert len(record.marginals) == record.degree * multiplier_count
        assert record.plain_full_multipliers == (1.0,) * (record.degree + 1)


def test_hidden_types_follow_sampled_compromised_node() -> None:
    config = _make_calibration_config()
    collection = collect_calibration(config, dataset=_dataset())
    for episode in collection.episodes:
        compromised = episode.spec.node
        for record in episode.records:
            assert sum(record.hidden_types) <= 1
            expected = tuple(
                1 if compromised is not None and sender == compromised else 0
                for sender in record.sender_ids
            )
            assert record.hidden_types == expected
            if compromised is None:
                assert sum(record.hidden_types) == 0


def test_only_plain_average_advances_the_trajectory() -> None:
    config = _make_calibration_config()
    collection = collect_calibration(config, dataset=_dataset())
    for episode in collection.episodes:
        assert episode.advanced_profiles
        for profile in episode.advanced_profiles:
            assert profile.multipliers == (1.0,) * len(profile.multipliers)


def test_zero_utility_anchors_hold_on_real_aggregation() -> None:
    config = _make_calibration_config()
    collection = collect_calibration(config, dataset=_dataset())
    for record in collection.records:
        zero_candidate = next(
            score
            for score in record.candidates
            if all(value == 0.0 for value in score.multipliers)
        )
        assert zero_candidate.utility == pytest.approx(0.0, abs=1e-12)
        assert zero_candidate.design_loss == pytest.approx(
            record.self_design_loss, abs=1e-12
        )
        for marginal in record.marginals:
            if marginal.multiplier == 0.0:
                assert marginal.utility == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Design-split disjointness
# --------------------------------------------------------------------------- #
def test_design_split_is_disjoint_from_every_node_partition() -> None:
    config = _make_calibration_config()
    dataset = _dataset()
    collection = collect_calibration(config, dataset=dataset)
    assert len(collection.episodes) == 2
    for episode in collection.episodes:
        resolved = derive_experiment_config(config, episode.spec)
        seeds = resolve_seeds(resolved)
        node_indices = partition_indices(
            len(dataset),
            (0, 1, 2),
            resolved.data.samples_per_node,
            seed=seeds.partition,
        )
        design = set(episode.design_indices)
        assert len(design) == config.design_split_size
        for block in node_indices.values():
            assert design.isdisjoint(set(block))
        # The design split is clean MNIST-training data, never the final-test
        # split: every design index is a valid training index.
        assert all(0 <= index < len(dataset) for index in episode.design_indices)


# --------------------------------------------------------------------------- #
# Determinism / records
# --------------------------------------------------------------------------- #
def test_collection_is_deterministic() -> None:
    config = _make_calibration_config()
    first = collect_calibration(config, dataset=_dataset())
    second = collect_calibration(config, dataset=_dataset())
    assert first.records == second.records
    assert [e.design_indices for e in first.episodes] == [
        e.design_indices for e in second.episodes
    ]


def test_records_are_tensor_free() -> None:
    config = _make_calibration_config()
    collection = collect_calibration(config, dataset=_dataset())
    for record in collection.records:
        for field_name in record.__slots__:
            value = getattr(record, field_name)
            assert not isinstance(value, torch.Tensor)


# --------------------------------------------------------------------------- #
# Per-episode derivation
# --------------------------------------------------------------------------- #
def test_derive_experiment_config_maps_seed_and_attack() -> None:
    config = _make_calibration_config()
    collection = collect_calibration(config, dataset=_dataset())
    for episode in collection.episodes:
        resolved = derive_experiment_config(config, episode.spec)
        assert resolved.seed == episode.spec.seed
        assert resolved.attack.compromised_node == episode.spec.node
        assert resolved.attack.intensity == episode.spec.intensity
        # Non-varying sections are preserved.
        assert resolved.topology == config.experiment.topology
        assert resolved.rounds == config.experiment.rounds
        assert resolved.aggregation == config.experiment.aggregation


def test_collect_episode_returns_a_calibration_observation_builder_record() -> None:
    config = _make_calibration_config()
    from dfl_sim import build_schedule

    spec = build_schedule(config).specs[0]
    episode = collect_episode(config, spec, dataset=_dataset())
    assert episode.spec == spec
    assert len(episode.records) == 3
    assert all(
        record.candidates and record.marginals for record in episode.records
    )


# --------------------------------------------------------------------------- #
# Negative surface (Q5/Q7)
# --------------------------------------------------------------------------- #
def test_too_small_dataset_aborts_before_records() -> None:
    config = _make_calibration_config()
    # 3 nodes * 8 + 8 design = 32 required; 16 cannot fit.
    with pytest.raises(DataError):
        collect_calibration(config, dataset=_dataset(16))


def test_collect_rejects_non_calibration_config() -> None:
    with pytest.raises(CalibrationScoringError, match="CalibrationConfig"):
        collect_calibration(make_config(), dataset=_dataset())


def test_derive_rejects_non_calibration_config() -> None:
    with pytest.raises(CalibrationError, match="CalibrationConfig"):
        derive_experiment_config(make_config(), None)  # type: ignore[arg-type]


def test_non_finite_design_loss_aborts_the_episode() -> None:
    config = _make_calibration_config()
    from dfl_sim import build_schedule

    spec = build_schedule(config).specs[0]
    dataset = _dataset()

    # Confirm a poisoned design loss is a hard offline error and leaves no
    # fabricated record behind (the real injection point is exercised above).
    resolved = derive_experiment_config(config, spec)
    sink = []
    builder = CalibrationObservationBuilder(
        config=config,
        compromised_node=spec.node,
        total_rounds=resolved.rounds,
        design_loss=lambda *a, **k: float("inf"),
        sink=sink,
    )
    round_start = {"w": torch.zeros(2, dtype=torch.float64)}
    trained = {"w": torch.ones(2, dtype=torch.float64)}
    senders = ((1, {"w": torch.tensor([1.0, 0.0], dtype=torch.float64)}),)
    with pytest.raises(CalibrationScoringError, match="design loss"):
        builder.build_observation(
            node_id=0,
            round_index=0,
            round_start=round_start,
            trained=trained,
            sender_states=senders,
            evaluator=lambda *a, **k: 0.0,
        )
    assert sink == []


# --------------------------------------------------------------------------- #
# Collection provenance fingerprints (S03/T01, R021/D037)
# --------------------------------------------------------------------------- #
def test_collection_captures_injected_content_and_membership_provenance() -> None:
    config = _make_calibration_config()
    collection = collect_calibration(config, dataset=_dataset())
    provenance = collection.provenance

    assert provenance.data_provenance == "injected"
    assert provenance.consumed_examples > 0
    # consumed examples are the union of node partitions + design splits.
    expected = {
        index
        for episode in collection.episodes
        for _node, indices in episode.node_indices
        for index in indices
    } | {
        index for episode in collection.episodes for index in episode.design_indices
    }
    assert provenance.consumed_examples == len(expected)
    for digest in (
        provenance.dataset_inputs_digest,
        provenance.partition_membership_digest,
        provenance.schedule_pairing_digest,
    ):
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_three_variant_provenance_matches_on_source_inputs() -> None:
    dataset = _dataset()
    full = collect_calibration(_make_calibration_config(), dataset=dataset)
    no_belief = collect_calibration(
        _make_calibration_config(variant=NO_BELIEF_VARIANT), dataset=dataset
    )
    no_strategic = collect_calibration(
        _make_calibration_config(variant=NO_STRATEGIC_VARIANT), dataset=dataset
    )

    for other in (no_belief, no_strategic):
        assert (
            other.provenance.dataset_inputs_digest
            == full.provenance.dataset_inputs_digest
        )
        assert (
            other.provenance.partition_membership_digest
            == full.provenance.partition_membership_digest
        )
        assert (
            other.provenance.schedule_pairing_digest
            == full.provenance.schedule_pairing_digest
        )
        assert other.provenance.consumed_examples == full.provenance.consumed_examples
        assert other.provenance.data_provenance == "injected"

    # The no-strategic variant really does change the sampled attacker mix.
    assert full.schedule.summary.intensity_counts != (
        no_strategic.schedule.summary.intensity_counts
    )


def test_changed_dataset_content_changes_content_digest_only() -> None:
    config = _make_calibration_config()
    first = collect_calibration(
        config, dataset=MNISTShapedDataset(200, separable=True, seed=11)
    )
    second = collect_calibration(
        config, dataset=MNISTShapedDataset(200, separable=True, seed=12)
    )
    # Same partition scheme / seeds -> same membership and pairing identity.
    assert (
        first.provenance.partition_membership_digest
        == second.provenance.partition_membership_digest
    )
    assert (
        first.provenance.schedule_pairing_digest
        == second.provenance.schedule_pairing_digest
    )
    # Different labels/tensors at the same indices -> different content digest.
    assert (
        first.provenance.dataset_inputs_digest
        != second.provenance.dataset_inputs_digest
    )


# --------------------------------------------------------------------------- #
# Provenance failure surface (Q5)
# --------------------------------------------------------------------------- #
def test_unsupported_tensor_content_is_a_bounded_error() -> None:
    with pytest.raises(CalibrationScoringError, match="unsupported dataset tensor"):
        cs._canonical_example_bytes("not-a-tensor", where="index 0")


def test_non_finite_tensor_content_is_a_bounded_error() -> None:
    tensor = torch.full((1, 2, 2), float("nan"))
    with pytest.raises(CalibrationScoringError, match="non-finite"):
        cs._canonical_example_bytes(tensor, where="index 0")


def test_non_integral_label_is_a_bounded_error() -> None:
    with pytest.raises(CalibrationScoringError, match="must be an integer"):
        cs._canonical_label("seven", where="index 0")


def test_unavailable_dataset_example_is_a_bounded_error() -> None:
    class Broken:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index):
            raise RuntimeError("boom")

    with pytest.raises(CalibrationScoringError, match="unavailable"):
        cs._consumed_dataset_examples(Broken(), [0])
