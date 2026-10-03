"""Hand-computable contract tests for the S03 local observation module (R005/D012/D017).

Every case is an in-process, deterministic tensor fixture -- no files, network,
or MNIST. Coverage mirrors the T01 contract: exact formula math, canonical
sender ordering, clamping/epsilon/zero-norm semantics, finite scalar output,
immutability/repeatability/input non-mutation, the narrow builder protocol, and
the Q7 malformed-input surface (duplicate/self senders, key/shape/dtype/finite
mismatches, invalid epsilon, raised/malformed evaluator output).
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from dfl_sim.observation import (
    DEFAULT_EPSILON,
    DefaultObservationBuilder,
    LocalObservation,
    NeighborObservation,
    ObservationBuilder,
    ObservationError,
    build_local_observation,
)

_TOL = 1e-12


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _state(**parts: list[float]) -> dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64) for key, value in parts.items()}


def _sum_evaluator(state, *, node_id, round_index, context=""):
    """Deterministic hand-computable node-local loss: sum of all parameters."""

    return float(sum(float(value.sum()) for value in state.values()))


class _RecordingEvaluator:
    """Records each call's context and the data pointers of the received tensors."""

    def __init__(self, fn=None) -> None:
        self._fn = fn or _sum_evaluator
        self.calls: list[dict] = []

    def __call__(self, state, *, node_id, round_index, context=""):
        self.calls.append(
            {
                "context": context,
                "ptrs": {key: value.data_ptr() for key, value in state.items()},
                "state": state,
            }
        )
        return self._fn(
            state, node_id=node_id, round_index=round_index, context=context
        )


def _build(
    *,
    node_id=3,
    round_index=1,
    round_start=None,
    trained=None,
    senders=(),
    evaluator=None,
    epsilon=DEFAULT_EPSILON,
):
    return build_local_observation(
        node_id=node_id,
        round_index=round_index,
        round_start=round_start if round_start is not None else _state(w=[0.0, 0.0]),
        trained=trained if trained is not None else _state(w=[1.0, 0.0]),
        sender_states=senders,
        evaluator=evaluator or _sum_evaluator,
        epsilon=epsilon,
    )


# --------------------------------------------------------------------------- #
# Participant ordering + self-only fallback
# --------------------------------------------------------------------------- #
def test_participant_order_is_self_first_then_senders_ascending() -> None:
    obs = _build(
        node_id=7,
        senders=(
            (9, _state(w=[2.0, 0.0])),
            (2, _state(w=[0.0, 2.0])),
            (5, _state(w=[3.0, 0.0])),
        ),
    )
    assert obs.node_id == 7
    assert obs.round_index == 1
    assert obs.neighbor_ids == (2, 5, 9)
    assert obs.participant_ids == (7, 2, 5, 9)
    assert obs.degree == 3


def test_empty_neighborhood_is_self_only_and_never_evaluates_neighbors() -> None:
    recorder = _RecordingEvaluator()
    obs = _build(node_id=0, senders=(), evaluator=recorder)
    assert obs.neighbor_ids == ()
    assert obs.participant_ids == (0,)
    assert obs.degree == 0
    # Only the node's own trained model is evaluated once.
    assert len(recorder.calls) == 1
    assert "sender=" not in recorder.calls[0]["context"]


# --------------------------------------------------------------------------- #
# Exact formula math (D017)
# --------------------------------------------------------------------------- #
def test_hand_computed_features() -> None:
    obs = _build(
        node_id=3,
        round_start=_state(w=[0.0, 0.0]),
        trained=_state(w=[1.0, 0.0]),
        senders=(
            (2, _state(w=[0.0, 2.0])),  # delta_ij = [0, 2]: orthogonal
            (5, _state(w=[2.0, 0.0])),  # delta_ij = [2, 0]: parallel
        ),
    )
    first, second = obs.neighbors
    assert first.sender_id == 2
    assert first.directional_disagreement == pytest.approx(1.0, abs=_TOL)
    assert first.relative_norm == pytest.approx(2.0, abs=_TOL)
    assert first.loss_delta == pytest.approx(1.0, abs=_TOL)  # sums 2 - 1

    assert second.sender_id == 5
    assert second.directional_disagreement == pytest.approx(0.0, abs=_TOL)
    assert second.relative_norm == pytest.approx(2.0, abs=_TOL)
    assert second.loss_delta == pytest.approx(1.0, abs=_TOL)


def test_two_key_sorted_flattening_matches_single_vector_math() -> None:
    obs = _build(
        node_id=0,
        round_start=_state(a=[0.0, 0.0], b=[0.0, 0.0]),
        trained=_state(a=[1.0, 0.0], b=[0.0, 0.0]),
        senders=((1, _state(a=[0.0, 0.0], b=[0.0, 2.0])),),
    )
    neighbor = obs.neighbors[0]
    assert neighbor.directional_disagreement == pytest.approx(1.0, abs=_TOL)
    assert neighbor.relative_norm == pytest.approx(2.0, abs=_TOL)
    assert neighbor.loss_delta == pytest.approx(1.0, abs=_TOL)


def test_both_deltas_zero_gives_zero_disagreement_and_norm() -> None:
    obs = _build(
        node_id=0,
        round_start=_state(w=[0.0, 0.0]),
        trained=_state(w=[0.0, 0.0]),
        senders=((1, _state(w=[0.0, 0.0])),),
    )
    neighbor = obs.neighbors[0]
    assert neighbor.directional_disagreement == 0.0
    assert neighbor.relative_norm == 0.0


def test_self_delta_zero_neighbor_nonzero_gives_disagreement_one() -> None:
    obs = _build(
        node_id=0,
        round_start=_state(w=[0.0, 0.0]),
        trained=_state(w=[0.0, 0.0]),
        senders=((1, _state(w=[1.0, 1.0])),),
        epsilon=1.0,
    )
    neighbor = obs.neighbors[0]
    assert neighbor.directional_disagreement == 1.0
    assert neighbor.relative_norm == pytest.approx(math.sqrt(2.0), abs=_TOL)


def test_self_delta_nonzero_neighbor_zero_gives_one_and_zero_norm() -> None:
    obs = _build(
        node_id=0,
        round_start=_state(w=[0.0, 0.0]),
        trained=_state(w=[1.0, 0.0]),
        senders=((1, _state(w=[0.0, 0.0])),),
    )
    neighbor = obs.neighbors[0]
    assert neighbor.directional_disagreement == 1.0
    assert neighbor.relative_norm == 0.0


@pytest.mark.parametrize(
    "sender_vector, expected_disagreement",
    [
        ([2.0, 0.0], 0.0),  # parallel, cos = 1 -> clamp low
        ([-1.0, 0.0], 2.0),  # antiparallel, cos = -1 -> clamp high
        ([10.0, 0.0], 0.0),  # parallel, |cos| would exceed 1 without clamp
    ],
)
def test_cosine_clipping_bounds(sender_vector, expected_disagreement) -> None:
    obs = _build(
        node_id=0,
        round_start=_state(w=[0.0, 0.0]),
        trained=_state(w=[1.0, 0.0]),
        senders=((1, _state(w=sender_vector)),),
    )
    assert obs.neighbors[0].directional_disagreement == pytest.approx(
        expected_disagreement, abs=_TOL
    )


def test_epsilon_floor_governs_relative_norm_when_self_norm_is_zero() -> None:
    obs = _build(
        node_id=0,
        round_start=_state(w=[0.0, 0.0]),
        trained=_state(w=[0.0, 0.0]),
        senders=((1, _state(w=[3.0, 4.0])),),
        epsilon=0.5,
    )
    # ||delta_ij|| = 5, max(0, 0.5) = 0.5 -> 10.0
    assert obs.neighbors[0].relative_norm == pytest.approx(10.0, abs=_TOL)


def test_all_exposed_values_are_finite() -> None:
    obs = _build(
        senders=(
            (1, _state(w=[1e150, -1e150])),
            (2, _state(w=[0.0, 0.0])),
        ),
    )
    assert math.isfinite(obs.epsilon)
    for neighbor in obs.neighbors:
        assert math.isfinite(neighbor.directional_disagreement)
        assert math.isfinite(neighbor.relative_norm)
        assert math.isfinite(neighbor.loss_delta)


# --------------------------------------------------------------------------- #
# Immutability, determinism, non-aliasing
# --------------------------------------------------------------------------- #
def test_observations_are_frozen() -> None:
    obs = _build(senders=((1, _state(w=[0.0, 2.0])),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.node_id = 99  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.neighbors[0].loss_delta = 0.0  # type: ignore[misc]


def test_repeatable_across_calls() -> None:
    senders = ((1, _state(w=[0.0, 2.0])), (4, _state(w=[2.0, 0.0])))
    first = _build(senders=senders)
    second = _build(senders=senders)
    assert first == second


def test_caller_states_are_not_mutated_and_evaluator_receives_clones() -> None:
    round_start = _state(w=[0.0, 0.0])
    trained = _state(w=[1.0, 0.0])
    sender = _state(w=[0.0, 2.0])
    origin_ptrs = {
        "round_start": round_start["w"].data_ptr(),
        "trained": trained["w"].data_ptr(),
        "sender": sender["w"].data_ptr(),
    }
    before = {
        "round_start": round_start["w"].clone(),
        "trained": trained["w"].clone(),
        "sender": sender["w"].clone(),
    }

    probe = _RecordingEvaluator(fn=None)

    def mutating(state, *, node_id, round_index, context=""):
        for value in state.values():
            value.add_(999.0)
        return 0.0

    probe._fn = mutating
    obs = build_local_observation(
        node_id=3,
        round_index=0,
        round_start=round_start,
        trained=trained,
        sender_states=[(1, sender)],
        evaluator=probe,
    )

    assert obs.participant_ids == (3, 1)
    # Inputs untouched despite the evaluator mutating everything it saw.
    assert torch.equal(round_start["w"], before["round_start"])
    assert torch.equal(trained["w"], before["trained"])
    assert torch.equal(sender["w"], before["sender"])
    # Every evaluator call received an independent clone, never the caller tensor.
    assert len(probe.calls) == 2
    for call in probe.calls:
        assert call["ptrs"]["w"] != origin_ptrs["round_start"]
        assert call["ptrs"]["w"] != origin_ptrs["trained"]
        assert call["ptrs"]["w"] != origin_ptrs["sender"]


def test_observation_exposes_no_tensor_storage() -> None:
    obs = _build(senders=((1, _state(w=[0.0, 2.0])),))

    def assert_scalar(value) -> None:
        assert not isinstance(value, torch.Tensor)
        assert not isinstance(value, dict)
        assert not isinstance(value, list)

    for field in dataclasses.fields(obs):
        value = getattr(obs, field.name)
        assert_scalar(value)
        if field.name == "neighbors":
            for neighbor in value:
                for inner in dataclasses.fields(neighbor):
                    assert_scalar(getattr(neighbor, inner.name))


# --------------------------------------------------------------------------- #
# Evaluator seam behavior (Q5/Q6)
# --------------------------------------------------------------------------- #
def test_evaluator_called_once_for_self_then_once_per_sender_in_order() -> None:
    recorder = _RecordingEvaluator()
    obs = _build(
        node_id=3,
        round_index=4,
        senders=(
            (9, _state(w=[1.0, 0.0])),
            (2, _state(w=[0.0, 1.0])),
            (5, _state(w=[2.0, 0.0])),
        ),
        evaluator=recorder,
    )
    assert obs.neighbor_ids == (2, 5, 9)
    assert len(recorder.calls) == 1 + 3
    # Baseline first, without a sender tag.
    assert recorder.calls[0]["context"] == "node=3 round=4 local-loss"
    sender_tags = [call["context"] for call in recorder.calls[1:]]
    assert "sender=2" in sender_tags[0]
    assert "sender=5" in sender_tags[1]
    assert "sender=9" in sender_tags[2]


def test_evaluator_failure_is_wrapped_with_node_sender_round_context() -> None:
    calls = {"count": 0}

    def boom_on_sender(state, *, node_id, round_index, context=""):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("evaluator exploded")
        return 0.0

    with pytest.raises(ObservationError, match="local loss evaluation failed") as excinfo:
        _build(
            node_id=4,
            round_index=3,
            senders=((8, _state(w=[0.0, 2.0])),),
            evaluator=boom_on_sender,
        )
    message = str(excinfo.value)
    assert "node=4" in message
    assert "round=3" in message
    assert "sender=8" in message


def test_baseline_evaluator_failure_includes_node_round() -> None:
    def boom(state, *, node_id, round_index, context=""):
        raise ValueError("nope")

    with pytest.raises(ObservationError) as excinfo:
        _build(node_id=6, round_index=2, evaluator=boom)
    message = str(excinfo.value)
    assert "node=6" in message
    assert "round=2" in message


@pytest.mark.parametrize("bad", [None, "x", True, object(), float("nan"), float("inf")])
def test_malformed_evaluator_output_rejected(bad) -> None:
    def bad_evaluator(state, *, node_id, round_index, context=""):
        return bad

    with pytest.raises(ObservationError):
        _build(evaluator=bad_evaluator)


def test_non_finite_loss_delta_from_overflow_rejected() -> None:
    losses = iter([1e308, -1e308])

    def overflowing(state, *, node_id, round_index, context=""):
        return next(losses)

    with pytest.raises(ObservationError, match="non-finite loss delta"):
        _build(
            node_id=0,
            round_index=0,
            senders=((1, _state(w=[0.0, 2.0])),),
            evaluator=overflowing,
        )


def test_non_callable_evaluator_rejected() -> None:
    with pytest.raises(ObservationError, match="local loss evaluation failed"):
        _build(evaluator=5)


# --------------------------------------------------------------------------- #
# Builder protocol
# --------------------------------------------------------------------------- #
def test_default_builder_satisfies_protocol_and_uses_configured_epsilon() -> None:
    builder = DefaultObservationBuilder(epsilon=0.5)
    assert isinstance(builder, ObservationBuilder)
    obs = builder.build_observation(
        node_id=0,
        round_index=0,
        round_start=_state(w=[0.0, 0.0]),
        trained=_state(w=[0.0, 0.0]),
        sender_states=[(1, _state(w=[1.0, 0.0]))],
        evaluator=_sum_evaluator,
    )
    assert obs.epsilon == 0.5
    assert obs.neighbors[0].relative_norm == pytest.approx(2.0, abs=_TOL)


def test_structural_builder_satisfies_protocol() -> None:
    class _StructuralBuilder:
        def build_observation(self, **kwargs):
            return build_local_observation(**kwargs)

    assert isinstance(_StructuralBuilder(), ObservationBuilder)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_default_builder_rejects_invalid_epsilon(bad) -> None:
    with pytest.raises(ObservationError):
        DefaultObservationBuilder(epsilon=bad)


# --------------------------------------------------------------------------- #
# Invalid sender sets (Q7)
# --------------------------------------------------------------------------- #
def test_duplicate_sender_rejected() -> None:
    with pytest.raises(ObservationError, match="duplicate received sender"):
        _build(senders=((2, _state(w=[0.0, 2.0])), (2, _state(w=[1.0, 0.0]))))


def test_self_sender_rejected() -> None:
    with pytest.raises(ObservationError, match="must not equal node id"):
        _build(node_id=2, senders=((2, _state(w=[0.0, 2.0])),))


@pytest.mark.parametrize("bad_sender", ["2", 2.0, None, True])
def test_non_integer_sender_rejected(bad_sender) -> None:
    with pytest.raises(ObservationError, match="sender_id must be int"):
        _build(senders=((bad_sender, _state(w=[0.0, 2.0])),))


@pytest.mark.parametrize("bad_entry", [(2,), (2, {}, {}), 2, "2"])
def test_malformed_sender_entry_rejected(bad_entry) -> None:
    with pytest.raises(ObservationError, match="pair"):
        _build(senders=(bad_entry,))


@pytest.mark.parametrize("bad_container", [None, 5, {"sender": 2}, "abc"])
def test_sender_states_must_be_a_sequence(bad_container) -> None:
    with pytest.raises(ObservationError, match="must be a sequence"):
        _build(senders=bad_container)


# --------------------------------------------------------------------------- #
# Invalid state payloads (Q7)
# --------------------------------------------------------------------------- #
def test_sender_key_mismatch_rejected() -> None:
    with pytest.raises(ObservationError, match="key mismatch"):
        _build(senders=((2, {"other": torch.zeros(2, dtype=torch.float64)}),))


def test_sender_shape_mismatch_rejected() -> None:
    with pytest.raises(ObservationError, match="shape mismatch"):
        _build(senders=((2, _state(w=[0.0, 2.0, 3.0])),))


def test_sender_dtype_mismatch_rejected() -> None:
    with pytest.raises(ObservationError, match="dtype mismatch"):
        _build(
            senders=((2, {"w": torch.zeros(2, dtype=torch.float32)}),)
        )


def test_sender_non_tensor_entry_rejected() -> None:
    with pytest.raises(ObservationError, match="must be a tensor"):
        _build(senders=((2, {"w": [0.0, 2.0]}),))


def test_sender_non_string_key_rejected() -> None:
    with pytest.raises(ObservationError, match="key must be str"):
        _build(senders=((2, {0: torch.zeros(2, dtype=torch.float64)}),))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_sender_non_finite_rejected(bad) -> None:
    state = _state(w=[0.0, 2.0])
    state["w"] = state["w"].clone()
    state["w"][0] = bad
    with pytest.raises(ObservationError, match="non-finite"):
        _build(senders=((2, state),))


def test_empty_round_start_rejected() -> None:
    with pytest.raises(ObservationError, match="invalid round-start state"):
        _build(round_start={})


def test_empty_trained_rejected() -> None:
    with pytest.raises(ObservationError, match="invalid trained state"):
        _build(trained={})


@pytest.mark.parametrize("label", ["round_start", "trained"])
def test_non_mapping_states_rejected(label: str) -> None:
    kwargs = {label: [("w", torch.zeros(2, dtype=torch.float64))]}
    with pytest.raises(ObservationError, match=f"{label} must be a mapping"):
        _build(**kwargs)


def test_state_error_context_includes_node_and_round() -> None:
    with pytest.raises(ObservationError) as excinfo:
        _build(
            node_id=5,
            round_index=2,
            round_start=_state(w=[0.0, 0.0]),
            trained=_state(w=[1.0, 0.0]),
            senders=((2, _state(w=[0.0, 2.0, 3.0])),),
        )
    message = str(excinfo.value)
    assert "node=5" in message
    assert "round=2" in message
    assert "sender=2" in message


def test_non_finite_delta_ii_from_extreme_finite_inputs_rejected() -> None:
    with pytest.raises(ObservationError, match="non-finite delta_ii"):
        _build(
            node_id=0,
            round_index=0,
            round_start={"w": torch.tensor([1e308], dtype=torch.float64)},
            trained={"w": torch.tensor([-1e308], dtype=torch.float64)},
            senders=(),
        )


def test_non_finite_delta_ij_from_extreme_finite_inputs_rejected() -> None:
    with pytest.raises(ObservationError, match="non-finite delta_ij"):
        _build(
            node_id=0,
            round_index=0,
            round_start={"w": torch.tensor([1e308], dtype=torch.float64)},
            trained={"w": torch.tensor([1e308], dtype=torch.float64)},
            senders=(
                (1, {"w": torch.tensor([-1e308], dtype=torch.float64)}),
            ),
        )


# --------------------------------------------------------------------------- #
# Invalid epsilon (Q7)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "x", True, None])
def test_invalid_epsilon_rejected(bad) -> None:
    with pytest.raises(ObservationError):
        _build(epsilon=bad)


# --------------------------------------------------------------------------- #
# Record-level validation
# --------------------------------------------------------------------------- #
def test_neighbor_observation_validates_scalar_ranges() -> None:
    with pytest.raises(ObservationError, match="within \\[0, 2\\]"):
        NeighborObservation(1, 2.5, 0.0, 0.0)
    with pytest.raises(ObservationError, match="within \\[0, 2\\]"):
        NeighborObservation(1, -0.5, 0.0, 0.0)
    with pytest.raises(ObservationError, match="non-negative"):
        NeighborObservation(1, 0.0, -1.0, 0.0)
    with pytest.raises(ObservationError):
        NeighborObservation(1, float("nan"), 0.0, 0.0)
    with pytest.raises(ObservationError, match="sender_id must be int"):
        NeighborObservation(True, 0.0, 0.0, 0.0)


def test_local_observation_validates_sender_ordering() -> None:
    descending = (
        NeighborObservation(2, 0.0, 0.0, 0.0),
        NeighborObservation(1, 0.0, 0.0, 0.0),
    )
    with pytest.raises(ObservationError, match="strictly ascending"):
        LocalObservation(node_id=0, round_index=0, epsilon=1e-12, neighbors=descending)

    duplicated = (
        NeighborObservation(1, 0.0, 0.0, 0.0),
        NeighborObservation(1, 0.0, 0.0, 0.0),
    )
    with pytest.raises(ObservationError, match="strictly ascending"):
        LocalObservation(node_id=0, round_index=0, epsilon=1e-12, neighbors=duplicated)


def test_local_observation_rejects_self_sender() -> None:
    with pytest.raises(ObservationError, match="must not equal node id"):
        LocalObservation(
            node_id=2,
            round_index=0,
            epsilon=1e-12,
            neighbors=(NeighborObservation(2, 0.0, 0.0, 0.0),),
        )


def test_local_observation_rejects_non_tuple_neighbors() -> None:
    with pytest.raises(ObservationError, match="tuple of NeighborObservation"):
        LocalObservation(
            node_id=0,
            round_index=0,
            epsilon=1e-12,
            neighbors=[NeighborObservation(1, 0.0, 0.0, 0.0)],  # type: ignore[arg-type]
        )


def test_local_observation_rejects_non_positive_epsilon() -> None:
    with pytest.raises(ObservationError, match="epsilon must be positive"):
        LocalObservation(node_id=0, round_index=0, epsilon=0.0, neighbors=())
