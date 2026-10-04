"""Hand-computable tests for the exact Bayesian best-response policy (T02/R015/R016).

The checked-in schema-v1 fixture is deliberately small enough to recompute the
posterior and the posterior-weighted expected payoff by hand. These tests
independently derive those rational values, assert the exact selected
:class:`~dfl_sim.policy.WeightProfile` and structured diagnostic, and prove the
D022 candidate enumeration is exact inside the degree cap and bounded and
deterministic above it. The Q7 negative surface covers missing cells, absent
context, zero mass, non-finite arithmetic, malformed observations, and exact
self-only fallback. Tests read only checked-in fixtures plus self-consistent
temporary artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

from dfl_sim.bayesian_artifact import (
    LIKELIHOOD_FILENAME,
    MANIFEST_FILENAME,
    PAYOFF_FILENAME,
    compute_aggregate_digest,
    load_bayesian_artifact,
)
from dfl_sim.observation import LocalObservation, NeighborObservation
from dfl_sim.policy import (
    FALLBACK_CONTEXT_UNAVAILABLE,
    FALLBACK_MISSING_LIKELIHOOD,
    FALLBACK_MISSING_PAYOFF,
    FALLBACK_MISSING_PRIOR,
    FALLBACK_NON_FINITE,
    FALLBACK_ZERO_MASS,
    BayesianDiagnostic,
    BayesianPolicy,
    PolicyError,
    build_bayesian_policy,
    enumerate_candidate_actions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"

MULTIPLIERS = (0.0, 0.5, 1.0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_json(name: str) -> object:
    return json.loads((FIXTURE / name).read_text(encoding="utf-8"))


def _obs(
    node_id: int,
    round_index: int,
    neighbors: tuple[tuple[int, float, float, float], ...],
) -> LocalObservation:
    return LocalObservation(
        node_id=node_id,
        round_index=round_index,
        epsilon=1e-12,
        neighbors=tuple(NeighborObservation(*entry) for entry in neighbors),
    )


def _collector() -> tuple[list[BayesianDiagnostic], object]:
    records: list[BayesianDiagnostic] = []
    return records, records.append


def _write_artifact(
    tmp_path: Path,
    *,
    prior: list | None = None,
    likelihood: list | None = None,
    payoff: list | None = None,
    degree_cap: int | None = None,
    name: str = "artifact",
):
    """Write a fully self-consistent artifact and load it through the strict path."""

    body = _load_json(MANIFEST_FILENAME)
    assert isinstance(body, dict)
    if degree_cap is not None:
        body["action_space"]["degree_cap"] = degree_cap
    if prior is not None:
        body["prior"] = prior
    lik = (
        likelihood
        if likelihood is not None
        else _load_json(LIKELIHOOD_FILENAME)
    )
    pay = payoff if payoff is not None else _load_json(PAYOFF_FILENAME)
    body["support"] = {
        **body["support"],
        "prior_cells": len(body["prior"]),
        "likelihood_cells": len(lik),
        "payoff_cells": len(pay),
    }
    lik_bytes = (json.dumps(lik, indent=2) + "\n").encode()
    pay_bytes = (json.dumps(pay, indent=2) + "\n").encode()
    body["payloads"] = {
        LIKELIHOOD_FILENAME: {
            "size_bytes": len(lik_bytes),
            "sha256": hashlib.sha256(lik_bytes).hexdigest(),
        },
        PAYOFF_FILENAME: {
            "size_bytes": len(pay_bytes),
            "sha256": hashlib.sha256(pay_bytes).hexdigest(),
        },
    }
    body["aggregate_digest"] = compute_aggregate_digest(body)

    dest = tmp_path / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / LIKELIHOOD_FILENAME).write_bytes(lik_bytes)
    (dest / PAYOFF_FILENAME).write_bytes(pay_bytes)
    (dest / MANIFEST_FILENAME).write_text(
        json.dumps(body, indent=2) + "\n", encoding="utf-8"
    )
    return load_bayesian_artifact(dest)


@pytest.fixture(scope="module")
def fixture():
    return load_bayesian_artifact(FIXTURE)


def _one_neighbor_evidence0() -> tuple[tuple[int, float, float, float], ...]:
    # disagreement 0.2 -> "lo" (index 0); relative_norm 0.5 -> index 0;
    # loss_delta -0.3 -> index 0, i.e. evidence_bin (0, 0, 0).
    return ((7, 0.2, 0.5, -0.3),)


# --------------------------------------------------------------------------- #
# Positive: posterior normalization and exact payoff argmax
# --------------------------------------------------------------------------- #
def test_degree1_posterior_and_full_acceptance(fixture) -> None:
    records, sink = _collector()
    policy = build_bayesian_policy(fixture, 3, diagnostic_sink=sink)
    profile = policy.select_weights(
        _obs(0, 1, _one_neighbor_evidence0())
    )

    # prior(0)=0.8, prior(1)=0.2; L(0)=0.6, L(1)=0.1
    # weights: 0.48, 0.02 -> mass 0.5 -> posterior 0.96, 0.04.
    # E[m=0]=0; E[m=0.5]=0.96*0.5+0.04*(-0.5)=0.46; E[m=1]=0.96-0.04=0.92.
    assert profile.participant_ids == (0, 7)
    assert profile.multipliers == (1.0, 1.0)
    assert profile.weights == pytest.approx((0.5, 0.5))

    assert len(records) == 1
    diagnostic = records[0]
    assert diagnostic.fallback_reason is None
    assert diagnostic.degree_bin == "1"
    assert diagnostic.progress == "mid"
    assert diagnostic.candidate_count == 3
    assert diagnostic.pruned is False
    assert diagnostic.normalization_mass == pytest.approx(0.5)
    assert [(entry.state, entry.probability) for entry in diagnostic.posterior] == [
        ((0,), pytest.approx(0.96)),
        ((1,), pytest.approx(0.04)),
    ]
    assert diagnostic.chosen_multipliers == (1.0, 1.0)


def test_degree1_different_evidence_selects_rejection(fixture) -> None:
    records, sink = _collector()
    policy = build_bayesian_policy(fixture, 3, diagnostic_sink=sink)
    # disagreement 0.75 -> "mid" evidence index 1 -> evidence (1, 0, 0).
    profile = policy.select_weights(_obs(0, 1, ((7, 0.75, 0.5, -0.3),)))

    # posterior 0.64/0.36; E[m=0.5]=0.16-0.36=-0.20, E[m=1]=0.32-0.72=-0.40,
    # so the exact maximum is m=0 -> self-only.
    assert profile.multipliers == (1.0, 0.0)
    assert profile.weights == pytest.approx((1.0, 0.0))
    assert records[0].fallback_reason is None
    assert [(e.state, e.probability) for e in records[0].posterior] == [
        ((0,), pytest.approx(0.64)),
        ((1,), pytest.approx(0.36)),
    ]


def test_degree2_exact_expected_payoff_selects_mixed_action(fixture) -> None:
    # A: evidence (0,0,0); B: evidence (1,0,0).
    # prior (0,0)=0.7, (1,0)=0.18, (0,1)=0.12
    # joint: 0.24, 0.04, 0.54 -> scaled 0.168, 0.0072, 0.0648 -> mass 0.24
    # posterior: 0.7, 0.03, 0.27
    # E(dA,dB) = A0(dA)*0.97 + A1(dA)*0.03 + B0(dB)*0.73 + B1(dB)*0.27
    #   dA: 0 -> 0; 0.5 -> 0.47; 1 -> 0.94 (max)
    #   dB: 0 -> 0; 0.5 -> -0.0875; 1 -> -0.175 (max)
    # -> chosen (1, 0), multipliers (1, 1, 0).
    records, sink = _collector()
    policy = build_bayesian_policy(fixture, 3, diagnostic_sink=sink)
    profile = policy.select_weights(
        _obs(0, 1, ((7, 0.2, 0.5, -0.3), (9, 0.75, 0.5, -0.3)))
    )

    assert profile.participant_ids == (0, 7, 9)
    assert profile.multipliers == (1.0, 1.0, 0.0)
    assert profile.weights == pytest.approx((0.5, 0.5, 0.0))

    diagnostic = records[0]
    assert diagnostic.fallback_reason is None
    assert diagnostic.degree_bin == "2"
    assert diagnostic.candidate_count == 9
    assert diagnostic.pruned is False
    assert diagnostic.normalization_mass == pytest.approx(0.24)
    assert [(e.state, e.probability) for e in diagnostic.posterior] == [
        ((0, 0), pytest.approx(0.7)),
        ((1, 0), pytest.approx(0.03)),
        ((0, 1), pytest.approx(0.27)),
    ]
    # at-most-one-malicious support: (1, 1) is never feasible.
    assert all(sum(entry.state) <= 1 for entry in diagnostic.posterior)
    assert (1, 1) not in {entry.state for entry in diagnostic.posterior}


def test_degree2_all_benign_evidence_accepts_everything(fixture) -> None:
    records, sink = _collector()
    policy = build_bayesian_policy(fixture, 3, diagnostic_sink=sink)
    profile = policy.select_weights(
        _obs(0, 1, ((7, 0.2, 0.5, -0.3), (9, 0.2, 0.5, -0.3)))
    )
    assert profile.multipliers == (1.0, 1.0, 1.0)
    assert records[0].candidate_count == 9
    assert records[0].fallback_reason is None


# --------------------------------------------------------------------------- #
# Tie tolerance and canonical action order
# --------------------------------------------------------------------------- #
def _tie_artifact(tmp_path: Path, higher: float):
    prior = [
        {"degree": 1, "state": [0], "probability": 1.0},
        {"degree": 1, "state": [1], "probability": 0.0},
    ]
    likelihood = [
        {
            "degree_bin": "1",
            "progress": "mid",
            "local_type": 0,
            "evidence_bin": [0, 0, 0],
            "probability": 1.0,
        },
        {
            "degree_bin": "1",
            "progress": "mid",
            "local_type": 1,
            "evidence_bin": [0, 0, 0],
            "probability": 1.0,
        },
    ]
    payoff = [
        {
            "degree_bin": "1",
            "progress": "mid",
            "local_type": 0,
            "evidence_bin": [0, 0, 0],
            "multiplier": 0.0,
            "value": 0.0,
        },
        {
            "degree_bin": "1",
            "progress": "mid",
            "local_type": 0,
            "evidence_bin": [0, 0, 0],
            "multiplier": 0.5,
            "value": 1.0,
        },
        {
            "degree_bin": "1",
            "progress": "mid",
            "local_type": 0,
            "evidence_bin": [0, 0, 0],
            "multiplier": 1.0,
            "value": higher,
        },
    ] + [
        {
            "degree_bin": "1",
            "progress": "mid",
            "local_type": 1,
            "evidence_bin": [0, 0, 0],
            "multiplier": multiplier,
            "value": 0.0,
        }
        for multiplier in (0.0, 0.5, 1.0)
    ]
    return _write_artifact(
        tmp_path, prior=prior, likelihood=likelihood, payoff=payoff, name="tie"
    )


def test_within_tolerance_tie_prefers_canonical_first(tmp_path: Path) -> None:
    # E(0.5)=1.0 and E(1.0)=1.0+5e-10 differ below the 1e-9 tolerance, so the
    # canonical order (fewer rejects, then ascending) picks 0.5 first.
    artifact = _tie_artifact(tmp_path, 1.0 + 5e-10)
    records, sink = _collector()
    policy = build_bayesian_policy(artifact, 3, diagnostic_sink=sink)
    profile = policy.select_weights(_obs(0, 1, _one_neighbor_evidence0()))
    assert profile.multipliers == (1.0, 0.5)
    assert records[0].fallback_reason is None


def test_outside_tolerance_tie_selects_higher(tmp_path: Path) -> None:
    artifact = _tie_artifact(tmp_path, 1.0 + 1e-6)
    records, sink = _collector()
    policy = build_bayesian_policy(artifact, 3, diagnostic_sink=sink)
    profile = policy.select_weights(_obs(0, 1, _one_neighbor_evidence0()))
    assert profile.multipliers == (1.0, 1.0)
    assert records[0].fallback_reason is None


def test_canonical_action_order_is_fewer_rejects_then_lexicographic() -> None:
    assert enumerate_candidate_actions(0, MULTIPLIERS) == ((),)
    assert enumerate_candidate_actions(1, MULTIPLIERS) == (
        (0.5,),
        (1.0,),
        (0.0,),
    )
    assert enumerate_candidate_actions(2, MULTIPLIERS) == (
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


def test_zero_mass_falls_back_to_self_only(tmp_path: Path) -> None:
    likelihood = _load_json(LIKELIHOOD_FILENAME)
    assert isinstance(likelihood, list)
    for row in likelihood:
        row["probability"] = 0.0
    artifact = _write_artifact(tmp_path, likelihood=likelihood, name="zero")
    records, sink = _collector()
    policy = build_bayesian_policy(artifact, 3, diagnostic_sink=sink)
    profile = policy.select_weights(_obs(0, 1, _one_neighbor_evidence0()))

    assert profile.multipliers == (1.0, 0.0)
    assert profile.weights == (1.0, 0.0)
    diagnostic = records[0]
    assert diagnostic.fallback_reason == FALLBACK_ZERO_MASS
    assert diagnostic.candidate_count == 0
    assert diagnostic.posterior == ()
    assert diagnostic.chosen_multipliers == (1.0, 0.0)
    assert diagnostic.degree_bin == "1"
    assert diagnostic.progress == "mid"


# --------------------------------------------------------------------------- #
# Q7 negative surface: missing context / cells, non-finite, malformed input
# --------------------------------------------------------------------------- #
def test_missing_likelihood_cell_falls_back(fixture) -> None:
    # total_rounds=1 puts round 0 in the "early" progress bin, for which the
    # fixture has no likelihood rows -> documented local-only fallback.
    records, sink = _collector()
    policy = build_bayesian_policy(fixture, 1, diagnostic_sink=sink)
    profile = policy.select_weights(_obs(0, 0, _one_neighbor_evidence0()))
    assert profile.multipliers == (1.0, 0.0)
    diagnostic = records[0]
    assert diagnostic.fallback_reason == FALLBACK_MISSING_LIKELIHOOD
    assert diagnostic.progress == "early"
    assert diagnostic.candidate_count == 0
    assert diagnostic.posterior == ()


def test_missing_prior_for_degree_falls_back(fixture) -> None:
    # The fixture declares degree bins up to "5+" but priors only for degrees
    # 1 and 2, so a 3-neighbor observation has no feasible support.
    records, sink = _collector()
    policy = build_bayesian_policy(fixture, 3, diagnostic_sink=sink)
    neighborhood = (
        (7, 0.2, 0.5, -0.3),
        (9, 0.2, 0.5, -0.3),
        (11, 0.2, 0.5, -0.3),
    )
    profile = policy.select_weights(_obs(0, 1, neighborhood))
    assert profile.multipliers == (1.0, 0.0, 0.0, 0.0)
    diagnostic = records[0]
    assert diagnostic.fallback_reason == FALLBACK_MISSING_PRIOR
    assert diagnostic.degree_bin == "3-4"
    assert diagnostic.candidate_count == 0


def test_missing_payoff_cell_falls_back(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path, payoff=[], name="no_payoff")
    records, sink = _collector()
    policy = build_bayesian_policy(artifact, 3, diagnostic_sink=sink)
    profile = policy.select_weights(_obs(0, 1, _one_neighbor_evidence0()))
    assert profile.multipliers == (1.0, 0.0)
    diagnostic = records[0]
    assert diagnostic.fallback_reason == FALLBACK_MISSING_PAYOFF
    # Posterior was computable, so it is retained on the fallback diagnostic.
    assert diagnostic.candidate_count == 3
    assert diagnostic.posterior != ()
    assert diagnostic.chosen_multipliers == (1.0, 0.0)


def test_non_finite_expected_payoff_falls_back(tmp_path: Path) -> None:
    prior = [{"degree": 2, "state": [0, 0], "probability": 1.0}]
    likelihood = [
        {
            "degree_bin": "2",
            "progress": "mid",
            "local_type": 0,
            "evidence_bin": [0, 0, 0],
            "probability": 1.0,
        }
    ]
    huge = 1e308
    payoff = [
        {
            "degree_bin": "2",
            "progress": "mid",
            "local_type": 0,
            "evidence_bin": [0, 0, 0],
            "multiplier": 0.0,
            "value": 0.0,
        },
        {
            "degree_bin": "2",
            "progress": "mid",
            "local_type": 0,
            "evidence_bin": [0, 0, 0],
            "multiplier": 0.5,
            "value": huge,
        },
        {
            "degree_bin": "2",
            "progress": "mid",
            "local_type": 0,
            "evidence_bin": [0, 0, 0],
            "multiplier": 1.0,
            "value": huge,
        },
    ]
    artifact = _write_artifact(
        tmp_path, prior=prior, likelihood=likelihood, payoff=payoff, name="overflow"
    )
    records, sink = _collector()
    policy = build_bayesian_policy(artifact, 3, diagnostic_sink=sink)
    profile = policy.select_weights(
        _obs(0, 1, ((7, 0.2, 0.5, -0.3), (9, 0.2, 0.5, -0.3)))
    )
    # sum of two 1e308 payoffs overflows the guarded fsum -> documented fallback
    # rather than an exception or a non-finite profile.
    assert profile.multipliers == (1.0, 0.0, 0.0)
    diagnostic = records[0]
    assert diagnostic.fallback_reason == FALLBACK_NON_FINITE
    assert diagnostic.candidate_count == 9
    assert all(
        entry.probability == entry.probability  # finite posterior preserved
        for entry in diagnostic.posterior
    )


def test_non_observation_input_is_a_boundary_error(fixture) -> None:
    policy = build_bayesian_policy(fixture, 3)
    with pytest.raises(PolicyError):
        policy.select_weights(object())  # type: ignore[arg-type]
    with pytest.raises(PolicyError):
        policy.select_weights(float("nan"))  # type: ignore[arg-type]


def test_round_out_of_range_is_absent_context(fixture) -> None:
    records, sink = _collector()
    policy = build_bayesian_policy(fixture, 3, diagnostic_sink=sink)
    profile = policy.select_weights(_obs(0, 3, _one_neighbor_evidence0()))
    assert profile.multipliers == (1.0, 0.0)
    assert records[0].fallback_reason == FALLBACK_CONTEXT_UNAVAILABLE


# --------------------------------------------------------------------------- #
# Empty neighborhood and construction guards
# --------------------------------------------------------------------------- #
def test_empty_neighborhood_resolves_self_only_without_fallback(fixture) -> None:
    records, sink = _collector()
    policy = build_bayesian_policy(fixture, 3, diagnostic_sink=sink)
    profile = policy.select_weights(_obs(5, 0, ()))
    assert profile.participant_ids == (5,)
    assert profile.multipliers == (1.0,)
    assert profile.weights == (1.0,)
    diagnostic = records[0]
    assert diagnostic.fallback_reason is None
    assert diagnostic.degree == 0
    assert diagnostic.candidate_count == 0
    assert diagnostic.pruned is False
    assert diagnostic.posterior == ()


def test_policy_construction_guards(fixture) -> None:
    with pytest.raises(PolicyError):
        build_bayesian_policy(fixture, 0)
    with pytest.raises(PolicyError):
        BayesianPolicy(artifact=object(), total_rounds=3)  # type: ignore[arg-type]
    with pytest.raises(PolicyError):
        build_bayesian_policy(fixture, 3, diagnostic_sink=object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Q6 load profile: exact inside the cap, bounded and deterministic above it
# --------------------------------------------------------------------------- #
def test_candidate_count_exact_inside_and_bounded_above_cap() -> None:
    for degree in range(0, 5):
        actions = enumerate_candidate_actions(degree, MULTIPLIERS)
        assert len(actions) == 3**degree
        assert len(set(actions)) == len(actions)

    for degree in (5, 6, 10):
        disagreements = [0.1] * degree
        sender_ids = list(range(1, degree + 1))
        actions = enumerate_candidate_actions(
            degree,
            MULTIPLIERS,
            degree_cap=4,
            disagreements=disagreements,
            sender_ids=sender_ids,
        )
        assert len(actions) == 3**4 == 81
        assert actions == enumerate_candidate_actions(
            degree,
            MULTIPLIERS,
            degree_cap=4,
            disagreements=disagreements,
            sender_ids=sender_ids,
        )


def test_pruning_requires_pruning_arguments() -> None:
    with pytest.raises(PolicyError):
        enumerate_candidate_actions(5, MULTIPLIERS, degree_cap=4)
    with pytest.raises(PolicyError):
        enumerate_candidate_actions(
            5, MULTIPLIERS, degree_cap=4, disagreements=[0.1] * 5, sender_ids=[1]
        )
    with pytest.raises(PolicyError):
        enumerate_candidate_actions(2, MULTIPLIERS, degree_cap=0)


def test_sender_id_ascending_breaks_pruning_ties() -> None:
    # Equal disagreement everywhere: the D022 tie-break keeps the lowest
    # sender ids, so the highest-id neighbor is the one forced to 1.
    degree = 5
    low_first = enumerate_candidate_actions(
        degree,
        MULTIPLIERS,
        degree_cap=4,
        disagreements=[0.5] * degree,
        sender_ids=[10, 20, 30, 40, 50],
    )
    assert all(action[4] == 1.0 for action in low_first)
    assert any(action[4] != 1.0 for action in low_first) is False
    assert (0.0, 0.0, 0.0, 0.0, 1.0) in low_first
    assert (1.0, 1.0, 1.0, 1.0, 1.0) in low_first

    high_first = enumerate_candidate_actions(
        degree,
        MULTIPLIERS,
        degree_cap=4,
        disagreements=[0.5] * degree,
        sender_ids=[50, 40, 30, 20, 10],
    )
    assert all(action[0] == 1.0 for action in high_first)
    assert (1.0, 0.0, 0.0, 0.0, 0.0) in high_first


def test_above_cap_end_to_end_pruned_selection(tmp_path: Path) -> None:
    prior = [
        {"degree": 5, "state": [0, 0, 0, 0, 0], "probability": 0.5},
        {"degree": 5, "state": [1, 0, 0, 0, 0], "probability": 0.1},
        {"degree": 5, "state": [0, 1, 0, 0, 0], "probability": 0.1},
        {"degree": 5, "state": [0, 0, 1, 0, 0], "probability": 0.1},
        {"degree": 5, "state": [0, 0, 0, 1, 0], "probability": 0.1},
        {"degree": 5, "state": [0, 0, 0, 0, 1], "probability": 0.1},
    ]
    likelihood = [
        {
            "degree_bin": "5+",
            "progress": "mid",
            "local_type": 0,
            "evidence_bin": [0, 0, 0],
            "probability": 0.6,
        },
        {
            "degree_bin": "5+",
            "progress": "mid",
            "local_type": 1,
            "evidence_bin": [0, 0, 0],
            "probability": 0.1,
        },
    ]
    payoff = [
        {
            "degree_bin": "5+",
            "progress": "mid",
            "local_type": local_type,
            "evidence_bin": [0, 0, 0],
            "multiplier": multiplier,
            "value": value,
        }
        for local_type, values in ((0, (0.0, 0.5, 1.0)), (1, (0.0, -0.5, -1.0)))
        for multiplier, value in zip((0.0, 0.5, 1.0), values)
    ]
    artifact = _write_artifact(
        tmp_path, prior=prior, likelihood=likelihood, payoff=payoff, name="pruned"
    )
    records, sink = _collector()
    policy = build_bayesian_policy(artifact, 3, diagnostic_sink=sink)
    neighborhood = tuple(
        (sender, 0.2, 0.5, -0.3) for sender in (1, 2, 3, 4, 5)
    )
    profile = policy.select_weights(_obs(0, 1, neighborhood))

    assert profile.participant_ids == (0, 1, 2, 3, 4, 5)
    assert profile.multipliers[0] == 1.0
    # All benign evidence favors full acceptance; pruning still bounds search.
    assert profile.multipliers == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert profile.weights == pytest.approx((1 / 6,) * 6)
    diagnostic = records[0]
    assert diagnostic.pruned is True
    assert diagnostic.candidate_count == 81
    assert diagnostic.degree_bin == "5+"
    assert diagnostic.fallback_reason is None


# --------------------------------------------------------------------------- #
# Observability: structured diagnostics reach the logging surface
# --------------------------------------------------------------------------- #
def test_diagnostic_is_emitted_to_logger(fixture, caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="dfl_sim.policy")
    policy = build_bayesian_policy(fixture, 3)
    policy.select_weights(_obs(0, 1, _one_neighbor_evidence0()))
    payloads = [
        record.bayesian_diagnostic
        for record in caplog.records
        if hasattr(record, "bayesian_diagnostic")
    ]
    assert len(payloads) == 1
    assert isinstance(payloads[0], BayesianDiagnostic)
    assert payloads[0].fallback_reason is None
    assert payloads[0].node_id == 0
