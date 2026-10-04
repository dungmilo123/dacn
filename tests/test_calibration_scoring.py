"""Hand-computable offline counterfactual scoring tests (T02, R013/R014).

Deterministic in-process tensor fixtures only -- no files, network, or MNIST.
The :class:`~dfl_sim.calibration_scoring.CalibrationObservationBuilder` is
driven directly with a hand-computable design-loss stub so candidate ordering,
loss deltas, evidence/context binning, hidden types, and the self-only-anchored
marginal projection can be checked exactly. Mutation sentinels prove the
counterfactual branches neither alias nor alter any input snapshot.
"""

from __future__ import annotations

import pytest
import torch

from dfl_sim import (
    CalibrationScoringError,
    CandidateProfileScore,
    NeighborPayoffMarginal,
    NodeRoundRecord,
    bin_evidence,
)
from dfl_sim.calibration_scoring import CalibrationObservationBuilder
from dfl_sim.observation import (
    DefaultObservationBuilder,
    LocalObservation,
    NeighborObservation,
)
from dfl_sim.policy import enumerate_candidate_actions
from test_calibration_config import base_dict, parse_dict

_TOL = 1e-12


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _state(*values: float) -> dict[str, torch.Tensor]:
    return {"w": torch.tensor(list(values), dtype=torch.float64)}


def _sum_loss(state, *, node_id=None, round_index=None, context="", **kwargs) -> float:
    """Hand-computable design loss: sum of every parameter value."""

    return float(sum(float(value.sum()) for value in state.values()))


def _zero_evaluator(state, *, node_id=None, round_index=None, context=""):
    return 0.0


def _make_builder(*, compromised_node=5, total_rounds=2, design_loss=_sum_loss):
    config = parse_dict(base_dict())
    sink: list[NodeRoundRecord] = []
    builder = CalibrationObservationBuilder(
        config=config,
        compromised_node=compromised_node,
        total_rounds=total_rounds,
        design_loss=design_loss,
        sink=sink,
    )
    return config, builder, sink


# round_start w=0, trained w=[2,0]; sender 5 w=[0,4]; sender 7 w=[0,0].
_ROUND_START = _state(0.0, 0.0)
_TRAINED = _state(2.0, 0.0)
_SENDER_5 = _state(0.0, 4.0)
_SENDER_7 = _state(0.0, 0.0)


def _build(builder):
    return builder.build_observation(
        node_id=3,
        round_index=1,
        round_start=_ROUND_START,
        trained=_TRAINED,
        sender_states=((5, _SENDER_5), (7, _SENDER_7)),
        evaluator=_zero_evaluator,
    )


# --------------------------------------------------------------------------- #
# Evidence binning
# --------------------------------------------------------------------------- #
def test_bin_evidence_uses_interval_right_convention() -> None:
    config = parse_dict(base_dict())
    neighbor = NeighborObservation(
        sender_id=5,
        directional_disagreement=0.6,
        relative_norm=1.0,
        loss_delta=0.0,
    )
    # disagreement edges [0.25, 0.5, 0.75, 1.0] -> 0.6 in [0.5, 0.75) -> 2
    # relative_norm edges [0.5, 1.0, 2.0] -> exact edge falls upper -> 2
    # loss_delta edges [-0.05, 0.0, 0.05] -> exact edge falls upper -> 2
    assert bin_evidence(config.bins, neighbor) == (2, 2, 2)


# --------------------------------------------------------------------------- #
# Delegation / boundary
# --------------------------------------------------------------------------- #
def test_returned_observation_is_the_default_one_unchanged() -> None:
    _, builder, sink = _make_builder()
    observation = _build(builder)
    expected = DefaultObservationBuilder().build_observation(
        node_id=3,
        round_index=1,
        round_start=_ROUND_START,
        trained=_TRAINED,
        sender_states=((5, _SENDER_5), (7, _SENDER_7)),
        evaluator=_zero_evaluator,
    )
    assert type(observation) is LocalObservation
    assert observation == expected
    assert observation.participant_ids == (3, 5, 7)
    assert len(sink) == 1


def test_degree_zero_returns_observation_without_record() -> None:
    _, builder, sink = _make_builder()
    observation = builder.build_observation(
        node_id=3,
        round_index=1,
        round_start=_ROUND_START,
        trained=_TRAINED,
        sender_states=(),
        evaluator=_zero_evaluator,
    )
    assert observation.degree == 0
    assert sink == []


# --------------------------------------------------------------------------- #
# Candidate ordering / loss deltas
# --------------------------------------------------------------------------- #
def test_candidate_ordering_and_utilities_are_exact() -> None:
    _, builder, sink = _make_builder()
    _build(builder)
    record = sink[0]
    assert record.self_design_loss == pytest.approx(2.0, abs=_TOL)

    expected_order = (
        (0.5, 0.5),
        (0.5, 1.0),
        (1.0, 0.5),
        (1.0, 1.0),
        (0.0, 0.5),
        (0.0, 1.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.0, 0.0),
    )
    assert tuple(score.multipliers for score in record.candidates) == expected_order

    utilities = {
        score.multipliers: score.utility for score in record.candidates
    }
    expected_utilities = {
        (0.5, 0.5): 0.0,
        (0.5, 1.0): 0.4,
        (1.0, 0.5): -0.4,
        (1.0, 1.0): 0.0,
        (0.0, 0.5): 2.0 / 3.0,
        (0.0, 1.0): 1.0,
        (0.5, 0.0): -2.0 / 3.0,
        (1.0, 0.0): -1.0,
        (0.0, 0.0): 0.0,
    }
    for multipliers, expected in expected_utilities.items():
        assert utilities[multipliers] == pytest.approx(expected, abs=_TOL)


def test_candidate_enumeration_reuses_shared_helper() -> None:
    _, builder, sink = _make_builder()
    observation = _build(builder)
    config = builder.config
    actions = enumerate_candidate_actions(
        observation.degree,
        tuple(config.action_space.multiplier_set),
        degree_cap=config.action_space.degree_cap,
        disagreements=tuple(
            neighbor.directional_disagreement for neighbor in observation.neighbors
        ),
        sender_ids=observation.neighbor_ids,
    )
    assert tuple(score.multipliers for score in sink[0].candidates) == actions


def test_all_zero_candidate_reproduces_self_only_anchor() -> None:
    _, builder, sink = _make_builder()
    _build(builder)
    record = sink[0]
    zero_candidate = next(
        score for score in record.candidates if score.multipliers == (0.0, 0.0)
    )
    assert zero_candidate.design_loss == pytest.approx(
        record.self_design_loss, abs=_TOL
    )
    assert zero_candidate.utility == pytest.approx(0.0, abs=_TOL)


# --------------------------------------------------------------------------- #
# Anchored single-neighbor marginals
# --------------------------------------------------------------------------- #
def test_anchored_marginals_are_exact() -> None:
    _, builder, sink = _make_builder(compromised_node=5)
    _build(builder)
    record = sink[0]
    assert len(record.marginals) == 2 * 3
    assert all(isinstance(m, NeighborPayoffMarginal) for m in record.marginals)

    by_key = {
        (m.hidden_type, m.multiplier): m.utility for m in record.marginals
    }
    # position 0 is sender 5 (malicious); position 1 is sender 7 (benign).
    expected = {
        (1, 0.0): 0.0,
        (1, 0.5): -2.0 / 3.0,
        (1, 1.0): -1.0,
        (0, 0.0): 0.0,
        (0, 0.5): 2.0 / 3.0,
        (0, 1.0): 1.0,
    }
    for key, value in expected.items():
        assert by_key[key] == pytest.approx(value, abs=_TOL)


def test_zero_multiplier_marginals_are_exactly_anchored() -> None:
    _, builder, sink = _make_builder(compromised_node=None)
    _build(builder)
    record = sink[0]
    for marginal in record.marginals:
        if marginal.multiplier == 0.0:
            assert marginal.utility == pytest.approx(0.0, abs=_TOL)
            assert marginal.design_loss == pytest.approx(
                record.self_design_loss, abs=_TOL
            )


def test_marginal_keys_match_anchored_profile_losses() -> None:
    _, builder, sink = _make_builder(compromised_node=7)
    _build(builder)
    record = sink[0]
    # Sender 5 is benign, sender 7 is malicious: hidden types flip position 1.
    assert record.hidden_types == (0, 1)
    assert [m.hidden_type for m in record.marginals] == [0, 0, 0, 1, 1, 1]
    assert [m.multiplier for m in record.marginals] == [0.0, 0.5, 1.0] * 2


# --------------------------------------------------------------------------- #
# Provenance / context
# --------------------------------------------------------------------------- #
def test_evidence_context_and_hidden_types_recorded() -> None:
    config = parse_dict(base_dict())
    _, builder, sink = _make_builder(compromised_node=5)
    observation = _build(builder)
    record = sink[0]
    assert record.node_id == 3
    assert record.round_index == 1
    assert record.degree == 2
    assert record.degree_bin == "2"
    assert record.progress == "mid"
    assert record.participant_ids == (3, 5, 7)
    assert record.sender_ids == (5, 7)
    assert record.evidence_bins == tuple(
        bin_evidence(config.bins, neighbor) for neighbor in observation.neighbors
    )
    assert record.hidden_types == (1, 0)
    assert record.plain_full_multipliers == (1.0, 1.0, 1.0)


def test_clean_episode_has_no_hidden_malicious_type() -> None:
    _, builder, sink = _make_builder(compromised_node=None)
    _build(builder)
    assert sink[0].hidden_types == (0, 0)
    assert all(m.hidden_type == 0 for m in sink[0].marginals)


def test_records_are_tensor_free() -> None:
    _, builder, sink = _make_builder()
    _build(builder)
    record = sink[0]

    def _assert_no_tensor(value) -> None:
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, (tuple, list)):
            for item in value:
                _assert_no_tensor(item)

    for field_name in record.__slots__:
        _assert_no_tensor(getattr(record, field_name))
    assert all(isinstance(c, CandidateProfileScore) for c in record.candidates)


# --------------------------------------------------------------------------- #
# Immutability / mutation sentinels
# --------------------------------------------------------------------------- #
def test_counterfactual_branches_do_not_mutate_inputs() -> None:
    _, builder, sink = _make_builder(compromised_node=5)
    round_start = _state(0.0, 0.0)
    trained = _state(2.0, 0.0)
    sender_5 = _state(0.0, 4.0)
    sender_7 = _state(0.0, 0.0)
    before = {
        "start": round_start["w"].clone(),
        "trained": trained["w"].clone(),
        "s5": sender_5["w"].clone(),
        "s7": sender_7["w"].clone(),
    }
    before_ptrs = {
        "trained": trained["w"].data_ptr(),
        "s5": sender_5["w"].data_ptr(),
        "s7": sender_7["w"].data_ptr(),
    }

    builder.build_observation(
        node_id=3,
        round_index=1,
        round_start=round_start,
        trained=trained,
        sender_states=((5, sender_5), (7, sender_7)),
        evaluator=_zero_evaluator,
    )

    assert torch.equal(round_start["w"], before["start"])
    assert torch.equal(trained["w"], before["trained"])
    assert torch.equal(sender_5["w"], before["s5"])
    assert torch.equal(sender_7["w"], before["s7"])
    assert trained["w"].data_ptr() == before_ptrs["trained"]
    assert sender_5["w"].data_ptr() == before_ptrs["s5"]
    assert sender_7["w"].data_ptr() == before_ptrs["s7"]
    assert len(sink) == 1


# --------------------------------------------------------------------------- #
# Negative surface (Q7)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_design_loss_is_a_hard_error(bad) -> None:
    _, builder, _ = _make_builder(design_loss=lambda *a, **k: bad)
    with pytest.raises(CalibrationScoringError, match="design loss"):
        _build(builder)


def test_non_numeric_design_loss_is_a_hard_error() -> None:
    _, builder, _ = _make_builder(design_loss=lambda *a, **k: "not-a-number")
    with pytest.raises(CalibrationScoringError, match="design loss"):
        _build(builder)


def test_builder_rejects_invalid_total_rounds() -> None:
    config = parse_dict(base_dict())
    with pytest.raises(CalibrationScoringError, match="total_rounds"):
        CalibrationObservationBuilder(
            config=config,
            compromised_node=None,
            total_rounds=0,
            design_loss=_sum_loss,
            sink=[],
        )


def test_builder_rejects_non_mapping_design_loss_sink() -> None:
    config = parse_dict(base_dict())
    with pytest.raises(CalibrationScoringError, match="sink"):
        CalibrationObservationBuilder(
            config=config,
            compromised_node=None,
            total_rounds=1,
            design_loss=_sum_loss,
            sink=(),  # type: ignore[arg-type]
        )
