"""Contract tests for S03 finite weight profiles and baseline policies (R005/R006/R007/D008/D017).

Deterministic, in-process tensor/scalar fixtures only: no files, network, or
MNIST. Coverage mirrors the T02 contract:

* the three baselines (local-only, plain average, fixed similarity filter) over
  the shared :class:`~dfl_sim.policy.AggregationPolicy` seam produce distinct,
  participant-aligned profiles;
* the policy-factory-independent finite action helper validates and normalizes
  D017 multipliers without enumerating the joint action space;
* participant-aligned weighted aggregation validates weights and state
  signatures before arithmetic, is deterministic, non-mutating and
  non-aliasing, and reproduces the D015 uniform mean on the plain profile;
* the Q7 negative surface: invalid omega/cutoff, malformed observation/profile
  objects, duplicate/foreign/reordered participants, forbidden multiplier
  values, bad sums, negative/NaN weights, empty/all-reject fallbacks, and
  tensor key/shape/dtype/finiteness mismatches.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from dfl_sim.aggregation import (
    AggregationError,
    aggregate_weighted_neighborhood,
    uniform_average,
    weighted_average,
)
from dfl_sim.observation import (
    DEFAULT_EPSILON,
    LocalObservation,
    NeighborObservation,
)
from dfl_sim.policy import (
    AggregationPolicy,
    FixedSimilarityFilterPolicy,
    LocalOnlyPolicy,
    PlainAveragePolicy,
    PolicyError,
    WeightProfile,
    build_weight_profile,
)

_TOL = 1e-12


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _neighbor(
    sender_id: int,
    *,
    disagreement: float = 0.0,
    relative_norm: float = 0.0,
    loss_delta: float = 0.0,
) -> NeighborObservation:
    return NeighborObservation(
        sender_id=sender_id,
        directional_disagreement=disagreement,
        relative_norm=relative_norm,
        loss_delta=loss_delta,
    )


def _obs(
    node_id: int = 3,
    *,
    round_index: int = 1,
    neighbors: tuple[NeighborObservation, ...] = (),
) -> LocalObservation:
    return LocalObservation(
        node_id=node_id,
        round_index=round_index,
        epsilon=DEFAULT_EPSILON,
        neighbors=neighbors,
    )


def _state(**parts: list[float]) -> dict[str, torch.Tensor]:
    return {
        key: torch.tensor(value, dtype=torch.float32)
        for key, value in parts.items()
    }


def _assert_valid_profile(
    profile: WeightProfile, participant_ids: tuple[int, ...]
) -> None:
    assert profile.participant_ids == participant_ids
    assert len(profile.weights) == len(participant_ids)
    assert len(profile.multipliers) == len(participant_ids)
    assert profile.multipliers[0] == 1.0
    for weight in profile.weights:
        assert weight >= 0.0
        assert weight < float("inf")
    assert abs(sum(profile.weights) - 1.0) <= _TOL


# --------------------------------------------------------------------------- #
# Baseline profiles: local-only, plain average, fixed similarity filter
# --------------------------------------------------------------------------- #
def test_local_only_empty_neighborhood_is_exactly_self_only() -> None:
    profile = LocalOnlyPolicy().select_weights(_obs(node_id=7))
    _assert_valid_profile(profile, (7,))
    assert profile.multipliers == (1.0,)
    assert profile.weights == (1.0,)
    assert profile.degree == 0
    assert profile.node_id == 7


def test_local_only_zeroes_every_neighbor() -> None:
    obs = _obs(
        node_id=7,
        neighbors=(
            _neighbor(2, disagreement=0.1),
            _neighbor(5, disagreement=0.9),
        ),
    )
    profile = LocalOnlyPolicy().select_weights(obs)
    _assert_valid_profile(profile, (7, 2, 5))
    assert profile.multipliers == (1.0, 0.0, 0.0)
    assert profile.weights == (1.0, 0.0, 0.0)
    assert profile.degree == 2


def test_plain_average_is_uniform_closed_neighborhood() -> None:
    obs = _obs(
        node_id=7,
        neighbors=(_neighbor(2, disagreement=0.3), _neighbor(5, disagreement=1.4)),
    )
    profile = PlainAveragePolicy().select_weights(obs)
    _assert_valid_profile(profile, (7, 2, 5))
    assert profile.multipliers == (1.0, 1.0, 1.0)
    assert profile.weights == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=_TOL)


def test_fixed_filter_partially_rejects_neighbors() -> None:
    obs = _obs(
        node_id=7,
        neighbors=(
            _neighbor(2, disagreement=0.25),  # <= 0.5 -> accept
            _neighbor(5, disagreement=0.75),  # > 0.5  -> reject
        ),
    )
    profile = FixedSimilarityFilterPolicy(cutoff=0.5).select_weights(obs)
    _assert_valid_profile(profile, (7, 2, 5))
    assert profile.multipliers == (1.0, 1.0, 0.0)
    assert profile.weights == pytest.approx((0.5, 0.5, 0.0), abs=_TOL)


def test_fixed_filter_includes_neighbor_exactly_at_cutoff() -> None:
    obs = _obs(node_id=7, neighbors=(_neighbor(2, disagreement=0.5),))
    profile = FixedSimilarityFilterPolicy(cutoff=0.5).select_weights(obs)
    assert profile.multipliers == (1.0, 1.0)
    assert profile.weights == pytest.approx((0.5, 0.5), abs=_TOL)


def test_fixed_filter_all_rejected_is_exactly_self_only() -> None:
    obs = _obs(
        node_id=7,
        neighbors=(
            _neighbor(2, disagreement=0.9),
            _neighbor(5, disagreement=1.5),
        ),
    )
    profile = FixedSimilarityFilterPolicy(cutoff=0.4).select_weights(obs)
    _assert_valid_profile(profile, (7, 2, 5))
    assert profile.weights == (1.0, 0.0, 0.0)


def test_fixed_filter_empty_neighborhood_is_exactly_self_only() -> None:
    profile = FixedSimilarityFilterPolicy(cutoff=0.0).select_weights(_obs(node_id=9))
    assert profile.participant_ids == (9,)
    assert profile.weights == (1.0,)


def test_three_baselines_produce_distinct_profiles() -> None:
    obs = _obs(
        node_id=7,
        neighbors=(
            _neighbor(2, disagreement=0.25),
            _neighbor(5, disagreement=0.75),
        ),
    )
    local = LocalOnlyPolicy().select_weights(obs)
    plain = PlainAveragePolicy().select_weights(obs)
    filtered = FixedSimilarityFilterPolicy(cutoff=0.5).select_weights(obs)
    assert local.weights != plain.weights
    assert filtered.weights != local.weights
    assert filtered.weights != plain.weights
    # Participant alignment is identical across all three baselines.
    assert local.participant_ids == plain.participant_ids == filtered.participant_ids


def test_policies_satisfy_runtime_checkable_protocol() -> None:
    policies = (
        LocalOnlyPolicy(),
        PlainAveragePolicy(),
        FixedSimilarityFilterPolicy(cutoff=0.5),
    )
    for policy in policies:
        assert isinstance(policy, AggregationPolicy)


def test_select_weights_rejects_non_observation_objects() -> None:
    for policy in (LocalOnlyPolicy(), PlainAveragePolicy()):
        with pytest.raises(PolicyError):
            policy.select_weights(object())
        with pytest.raises(PolicyError):
            policy.select_weights(None)
    with pytest.raises(PolicyError):
        FixedSimilarityFilterPolicy(cutoff=0.5).select_weights("not-an-observation")


def test_profiles_and_policies_are_immutable() -> None:
    profile = PlainAveragePolicy().select_weights(_obs(node_id=1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.weights = (0.0, 1.0)  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.multipliers = (1.0,)  # type: ignore[misc]
    # Frozen+slots dataclasses only guard declared fields; use a declared field
    # to prove the policy itself is immutable (unknown attrs raise TypeError).
    policy = FixedSimilarityFilterPolicy(cutoff=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.cutoff = 0.9  # type: ignore[misc]


def test_weight_for_returns_aligned_weight_and_rejects_foreign_id() -> None:
    profile = FixedSimilarityFilterPolicy(cutoff=0.5).select_weights(
        _obs(node_id=7, neighbors=(_neighbor(2), _neighbor(5, disagreement=1.9)))
    )
    assert profile.weight_for(7) == pytest.approx(0.5, abs=_TOL)
    assert profile.weight_for(2) == pytest.approx(0.5, abs=_TOL)
    assert profile.weight_for(5) == 0.0
    with pytest.raises(PolicyError):
        profile.weight_for(99)


# --------------------------------------------------------------------------- #
# Finite action helper (D017): omega downweight + membership validation
# --------------------------------------------------------------------------- #
def test_finite_action_helper_proves_omega_downweight() -> None:
    profile = build_weight_profile(
        (0, 1, 2), (1.0, 0.25, 1.0), omega=0.25
    )
    _assert_valid_profile(profile, (0, 1, 2))
    assert profile.multipliers == (1.0, 0.25, 1.0)
    total = 2.25
    assert profile.weights == pytest.approx(
        (1.0 / total, 0.25 / total, 1.0 / total), abs=_TOL
    )
    # The downweighted neighbor receives strictly less than an accepted peer.
    assert profile.weights[1] < profile.weights[2]


def test_finite_action_helper_accepts_zero_and_one_neighbors() -> None:
    profile = build_weight_profile((0, 1, 2), (1.0, 0.0, 1.0), omega=0.5)
    assert profile.multipliers == (1.0, 0.0, 1.0)
    assert profile.weights == pytest.approx((0.5, 0.0, 0.5), abs=_TOL)


@pytest.mark.parametrize("omega", [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf"), "0.5", True])
def test_finite_action_helper_rejects_invalid_omega(omega) -> None:
    with pytest.raises(PolicyError):
        build_weight_profile((0, 1), (1.0, 1.0), omega=omega)


@pytest.mark.parametrize("bad", [0.3, 0.75, -1.0, 2.0, float("nan"), float("inf")])
def test_finite_action_helper_rejects_off_space_neighbor_multiplier(bad) -> None:
    with pytest.raises(PolicyError):
        build_weight_profile((0, 1), (1.0, bad), omega=0.5)


@pytest.mark.parametrize("self_multiplier", [0.0, 0.5, 2.0, -1.0])
def test_finite_action_helper_requires_self_multiplier_one(self_multiplier) -> None:
    with pytest.raises(PolicyError):
        build_weight_profile((0, 1), (self_multiplier, 1.0))


def test_finite_action_helper_validates_participants_and_lengths() -> None:
    with pytest.raises(PolicyError):
        build_weight_profile((), ())
    with pytest.raises(PolicyError):
        build_weight_profile((0, 0), (1.0, 1.0))  # duplicate id
    with pytest.raises(PolicyError):
        build_weight_profile((0, 1), (1.0,))  # length mismatch
    with pytest.raises(PolicyError):
        build_weight_profile((0, True), (1.0, 1.0))  # bool id
    with pytest.raises(PolicyError):
        build_weight_profile("01", (1.0, 1.0))  # str is not a participant seq


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, "1.0", True])
def test_finite_action_helper_rejects_bad_multiplier_scalars(bad) -> None:
    with pytest.raises(PolicyError):
        build_weight_profile((0, 1), (1.0, bad))


# --------------------------------------------------------------------------- #
# WeightProfile invariants (malformed profile objects)
# --------------------------------------------------------------------------- #
def test_weight_profile_rejects_bad_sum() -> None:
    with pytest.raises(PolicyError):
        WeightProfile((0, 1), (1.0, 1.0), (0.4, 0.4))


def test_weight_profile_rejects_negative_or_nonfinite_weights() -> None:
    with pytest.raises(PolicyError):
        WeightProfile((0, 1), (1.0, 1.0), (-0.5, 1.5))
    with pytest.raises(PolicyError):
        WeightProfile((0, 1), (1.0, 1.0), (float("nan"), 1.0))
    with pytest.raises(PolicyError):
        WeightProfile((0, 1), (1.0, 1.0), (float("inf"), 0.0))


def test_weight_profile_rejects_multiplier_weight_disagreement() -> None:
    with pytest.raises(PolicyError):
        WeightProfile((0, 1), (1.0, 1.0), (0.9, 0.1))


def test_weight_profile_rejects_duplicate_and_empty_participants() -> None:
    with pytest.raises(PolicyError):
        WeightProfile((0, 0), (1.0, 1.0), (0.5, 0.5))
    with pytest.raises(PolicyError):
        WeightProfile((), (), ())


def test_weight_profile_requires_self_multiplier_one() -> None:
    with pytest.raises(PolicyError):
        WeightProfile((0, 1), (0.5, 0.5), (0.5, 0.5))


# --------------------------------------------------------------------------- #
# Weighted aggregation arithmetic
# --------------------------------------------------------------------------- #
def test_weighted_average_hand_computed() -> None:
    self_state = _state(w=[0.0, 2.0])
    peer = _state(w=[2.0, 0.0])
    out = weighted_average([self_state, peer], [0.25, 0.75])
    assert torch.allclose(out["w"], torch.tensor([1.5, 0.5]))


def test_weighted_average_is_deterministic_and_non_aliasing() -> None:
    self_state = _state(w=[1.0, 2.0], b=[3.0])
    peer = _state(w=[4.0, 5.0], b=[6.0])
    before_self = {k: v.clone() for k, v in self_state.items()}
    before_peer = {k: v.clone() for k, v in peer.items()}

    first = weighted_average([self_state, peer], [0.4, 0.6])
    second = weighted_average([self_state, peer], [0.4, 0.6])

    for key in first:
        assert torch.equal(first[key], second[key])
        # Output shares no storage with any input.
        assert first[key].data_ptr() != self_state[key].data_ptr()
        assert first[key].data_ptr() != peer[key].data_ptr()
    for key in self_state:
        assert torch.equal(self_state[key], before_self[key])
        assert torch.equal(peer[key], before_peer[key])


def test_weighted_average_matches_uniform_average_on_plain_profile() -> None:
    self_state = _state(w=[1.0, 2.0, 3.0], b=[0.5])
    neighbors = [
        _state(w=[4.0, 5.0, 6.0], b=[0.7]),
        _state(w=[7.0, 8.0, 9.0], b=[0.9]),
    ]
    obs = _obs(node_id=1, neighbors=(_neighbor(0), _neighbor(2)))
    profile = PlainAveragePolicy().select_weights(obs)

    weighted = weighted_average([self_state, *neighbors], list(profile.weights))
    uniform = uniform_average([self_state, *neighbors])
    for key in uniform:
        assert torch.allclose(weighted[key], uniform[key], rtol=1e-5, atol=1e-6)


def test_weighted_average_with_single_participant_returns_clone() -> None:
    self_state = _state(w=[1.0, -2.0])
    out = weighted_average([self_state], [1.0])
    assert torch.equal(out["w"], self_state["w"])
    assert out["w"].data_ptr() != self_state["w"].data_ptr()


def test_aggregate_weighted_neighborhood_aligns_and_matches_core() -> None:
    self_state = _state(w=[0.0])
    neighbors = [_state(w=[4.0]), _state(w=[8.0])]
    weights = [0.5, 0.25, 0.25]
    out = aggregate_weighted_neighborhood(
        self_state, neighbors, weights, node_id=1, round_index=2
    )
    expected = weighted_average([self_state, *neighbors], weights)
    assert torch.allclose(out["w"], expected["w"])
    assert torch.allclose(out["w"], torch.tensor([3.0]))


# --------------------------------------------------------------------------- #
# Negative: weight validation before arithmetic
# --------------------------------------------------------------------------- #
def test_weighted_average_rejects_wrong_weight_count() -> None:
    with pytest.raises(AggregationError):
        weighted_average([_state(w=[1.0])], [0.5, 0.5])
    with pytest.raises(AggregationError):
        aggregate_weighted_neighborhood(
            _state(w=[1.0]), [_state(w=[1.0])], [1.0, 0.0, 0.0]
        )


@pytest.mark.parametrize(
    "weights",
    [
        [0.4, 0.4],  # bad sum
        [0.5, 0.6],  # bad sum
        [-0.5, 1.5],  # negative
        [float("nan"), 1.0],
        [float("inf"), 0.0],
        "not-a-sequence",
        [True, 0.0],  # bool
        ["0.5", 0.5],  # str entry
    ],
)
def test_weighted_average_rejects_malformed_weights(weights) -> None:
    with pytest.raises(AggregationError):
        weighted_average([_state(w=[1.0]), _state(w=[2.0])], weights)


def test_weighted_average_rejects_empty_states() -> None:
    with pytest.raises(AggregationError):
        weighted_average([], [])


# --------------------------------------------------------------------------- #
# Negative: tensor signature validation before arithmetic
# --------------------------------------------------------------------------- #
def test_weighted_average_rejects_key_signature_mismatch() -> None:
    with pytest.raises(AggregationError):
        weighted_average([_state(w=[1.0]), _state(other=[2.0])], [0.5, 0.5])


def test_weighted_average_rejects_shape_mismatch() -> None:
    with pytest.raises(AggregationError):
        weighted_average([_state(w=[1.0, 2.0]), _state(w=[1.0])], [0.5, 0.5])


def test_weighted_average_rejects_dtype_mismatch() -> None:
    first = {"w": torch.tensor([1.0], dtype=torch.float32)}
    second = {"w": torch.tensor([2.0], dtype=torch.float64)}
    with pytest.raises(AggregationError):
        weighted_average([first, second], [0.5, 0.5])


def test_weighted_average_rejects_non_finite_state() -> None:
    with pytest.raises(AggregationError):
        weighted_average(
            [{"w": torch.tensor([1.0])}, {"w": torch.tensor([float("nan")])}],
            [0.5, 0.5],
        )


def test_weighted_average_rejects_non_tensor_state() -> None:
    with pytest.raises(AggregationError):
        weighted_average([{"w": 1.0}], [1.0])


# --------------------------------------------------------------------------- #
# Negative: cutoff bounds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cutoff", [-0.1, 2.1, float("nan"), float("inf"), float("-inf"), "0.5", True]
)
def test_fixed_filter_rejects_invalid_cutoff(cutoff) -> None:
    with pytest.raises(PolicyError):
        FixedSimilarityFilterPolicy(cutoff=cutoff)
