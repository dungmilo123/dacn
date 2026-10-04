"""Hand-computable smoothed calibration-estimator tests (T03, R013/R014/R016).

Deterministic in-process fixtures only. The estimator is driven with explicit
:class:`~dfl_sim.calibration_scoring.NodeRoundRecord` /
:class:`~dfl_sim.calibration.PriorStateCounts` inputs so every smoothing,
normalization, shrinkage, ordering, coverage, and Brier value is checkable by
hand. A final integration test proves the estimator accepts a real collected
clean+attacked calibration and produces a directory the S01 loader accepts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dfl_sim import (
    CalibrationEstimationError,
    NeighborPayoffMarginal,
    NodeRoundRecord,
    PriorStateCounts,
    collect_calibration,
    estimate_artifact,
    estimate_calibration,
)
from dfl_sim.bayesian_artifact import (
    MANIFEST_FILENAME,
    canonical_json_bytes,
    load_bayesian_artifact,
)
from dfl_sim.calibration_config import CalibrationConfig
from helpers import MNISTShapedDataset, make_config
from test_calibration_config import base_dict, parse_dict


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _small_bins() -> dict:
    return {
        "disagreement": {"edges": [0.5], "labels": ["lo", "hi"]},
        "relative_norm": {"edges": [1.0], "labels": ["small", "large"]},
        "loss_delta": {"edges": [0.0], "labels": ["neg", "pos"]},
        "degree": {"labels": ["1", "2", "3-4", "5+"]},
        "progress": {"labels": ["early", "mid", "late"]},
    }


def _config() -> CalibrationConfig:
    return parse_dict(base_dict(bins=_small_bins()))


def _marginal(degree_bin, progress, evidence, hidden, multiplier, utility):
    return NeighborPayoffMarginal(
        evidence_bin=evidence,
        degree_bin=degree_bin,
        progress=progress,
        hidden_type=hidden,
        multiplier=multiplier,
        design_loss=0.0,
        utility=utility,
    )


def _record(
    *,
    degree,
    degree_bin,
    progress,
    evidence_bins,
    hidden_types,
    marginals,
    node_id=0,
    round_index=0,
) -> NodeRoundRecord:
    return NodeRoundRecord(
        node_id=node_id,
        round_index=round_index,
        degree=degree,
        degree_bin=degree_bin,
        progress=progress,
        participant_ids=tuple(range(degree + 1)),
        sender_ids=tuple(range(1, degree + 1)),
        evidence_bins=tuple(evidence_bins),
        hidden_types=tuple(hidden_types),
        self_design_loss=0.0,
        plain_full_multipliers=(1.0,) * (degree + 1),
        candidates=(),
        marginals=tuple(marginals),
    )


def _prior_counts() -> PriorStateCounts:
    return PriorStateCounts(
        counts={
            1: {(0,): 3, (1,): 1},
            2: {(0, 0): 2, (1, 0): 1},
        },
        episodes=2,
        benign_observations=6,
    )


def _records():
    """Two degree-1 observations: a benign and a malicious one, same context."""

    record_a = _record(
        degree=1,
        degree_bin="1",
        progress="mid",
        evidence_bins=[(0, 0, 0)],
        hidden_types=[0],
        marginals=[
            _marginal("1", "mid", (0, 0, 0), 0, 0.0, 0.0),
            _marginal("1", "mid", (0, 0, 0), 0, 0.5, 2.0),
            _marginal("1", "mid", (0, 0, 0), 0, 1.0, -4.0),
        ],
        node_id=0,
    )
    record_b = _record(
        degree=1,
        degree_bin="1",
        progress="mid",
        evidence_bins=[(1, 1, 1)],
        hidden_types=[1],
        marginals=[
            _marginal("1", "mid", (1, 1, 1), 1, 0.0, 0.0),
            _marginal("1", "mid", (1, 1, 1), 1, 0.5, -1.0),
            _marginal("1", "mid", (1, 1, 1), 1, 1.0, 3.0),
        ],
        node_id=1,
    )
    return (record_a, record_b)


def _estimate():
    return estimate_artifact(
        _config(),
        _records(),
        _prior_counts(),
        calibration_episodes=2,
        calibration_rounds=2,
    )


# --------------------------------------------------------------------------- #
# Prior smoothing
# --------------------------------------------------------------------------- #
def test_prior_smoothing_is_exact() -> None:
    artifact = _estimate()
    by_degree = {}
    for cell in artifact.manifest.prior:
        by_degree.setdefault(cell.degree, {})[cell.state] = cell.probability
    # degree 1: (0,): 3, (1,): 1, alpha 1, |feasible| 2 -> /6
    assert by_degree[1][(0,)] == pytest.approx(4.0 / 6.0, abs=1e-15)
    assert by_degree[1][(1,)] == pytest.approx(2.0 / 6.0, abs=1e-15)
    # degree 2: (0,0): 2, (1,0): 1, (0,1): 0, alpha 1, |feasible| 3 -> /6
    assert by_degree[2][(0, 0)] == pytest.approx(3.0 / 6.0, abs=1e-15)
    assert by_degree[2][(1, 0)] == pytest.approx(2.0 / 6.0, abs=1e-15)
    assert by_degree[2][(0, 1)] == pytest.approx(1.0 / 6.0, abs=1e-15)
    # Every feasible degree-specific row exists and sums to one.
    assert len(artifact.manifest.prior) == 5
    for degree, probabilities in by_degree.items():
        assert len(probabilities) == degree + 1
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-15)


def test_prior_rows_are_sorted_and_unique() -> None:
    artifact = _estimate()
    keys = [(cell.degree, cell.state) for cell in artifact.manifest.prior]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


# --------------------------------------------------------------------------- #
# Likelihood normalization
# --------------------------------------------------------------------------- #
def test_likelihood_full_support_and_normalization_are_exact() -> None:
    artifact = _estimate()
    rows = json.loads(artifact.likelihood_bytes)
    # one reached context pair ("1", "mid") x two local types x 8 evidence cells
    assert len(rows) == 16
    index = {
        (r["degree_bin"], r["progress"], r["local_type"], tuple(r["evidence_bin"])): r[
            "probability"
        ]
        for r in rows
    }
    # Type 0: evidence (0,0,0) counted once, total 1, alpha 1, support 8 -> /9.
    assert index[("1", "mid", 0, (0, 0, 0))] == pytest.approx(2.0 / 9.0, abs=1e-15)
    assert index[("1", "mid", 0, (1, 1, 1))] == pytest.approx(1.0 / 9.0, abs=1e-15)
    # Type 1: evidence (1,1,1) counted once.
    assert index[("1", "mid", 1, (1, 1, 1))] == pytest.approx(2.0 / 9.0, abs=1e-15)
    assert index[("1", "mid", 1, (0, 0, 0))] == pytest.approx(1.0 / 9.0, abs=1e-15)
    for local_type in (0, 1):
        total = sum(
            probability
            for (db, pg, lt, _ev), probability in index.items()
            if (db, pg, lt) == ("1", "mid", local_type)
        )
        assert total == pytest.approx(1.0, abs=1e-15)
    # Evidence indices are canonical ascending tuples within each local type.
    for local_type in (0, 1):
        evidence = [
            tuple(r["evidence_bin"]) for r in rows if r["local_type"] == local_type
        ]
        assert evidence == sorted(evidence)


# --------------------------------------------------------------------------- #
# Payoff shrinkage
# --------------------------------------------------------------------------- #
def test_payoff_shrinkage_is_exact_and_ordered() -> None:
    artifact = _estimate()
    rows = json.loads(artifact.payoff_bytes)
    # two reached cells x three configured multipliers
    assert len(rows) == 6
    index = {
        (
            r["degree_bin"],
            r["progress"],
            r["local_type"],
            tuple(r["evidence_bin"]),
            r["multiplier"],
        ): r["value"]
        for r in rows
    }
    zero_cell = ("1", "mid", 0, (0, 0, 0))
    assert index[(*zero_cell, 0.0)] == pytest.approx(0.0, abs=1e-15)
    assert index[(*zero_cell, 0.5)] == pytest.approx(2.0 / 2.0, abs=1e-15)
    assert index[(*zero_cell, 1.0)] == pytest.approx(-4.0 / 2.0, abs=1e-15)
    one_cell = ("1", "mid", 1, (1, 1, 1))
    assert index[(*one_cell, 0.0)] == pytest.approx(0.0, abs=1e-15)
    assert index[(*one_cell, 0.5)] == pytest.approx(-1.0 / 2.0, abs=1e-15)
    assert index[(*one_cell, 1.0)] == pytest.approx(3.0 / 2.0, abs=1e-15)
    keys = [
        (
            r["degree_bin"],
            r["progress"],
            r["local_type"],
            tuple(r["evidence_bin"]),
            r["multiplier"],
        )
        for r in rows
    ]
    assert keys == sorted(keys)


# --------------------------------------------------------------------------- #
# Support / coverage / Brier
# --------------------------------------------------------------------------- #
def test_support_coverage_and_brier_are_exact() -> None:
    artifact = _estimate()
    support = artifact.support
    assert support.calibration_episodes == 2
    assert support.calibration_rounds == 2
    assert support.prior_cells == 5
    assert support.likelihood_cells == 16
    assert support.payoff_cells == 6
    assert support.min_cell_count == 1
    coverage = artifact.coverage
    # Likelihood: 2 observed cells / (1 pair x 2 types x 8 evidence) = 2/16.
    assert coverage.likelihood == pytest.approx(2.0 / 16.0, abs=1e-15)
    # Payoff: 6 observed cells / (16 x 3 multipliers) = 6/48.
    assert coverage.payoff == pytest.approx(6.0 / 48.0, abs=1e-15)
    # Brier terms: record A p1=1/5 vs 0 -> 1/25; record B p1=1/2 vs 1 -> 1/4.
    expected_brier = (0.04 + 0.25) / 2.0
    assert coverage.posterior_calibration == pytest.approx(
        1.0 - expected_brier, abs=1e-15
    )


def test_payload_digests_and_aggregate_digest_match_bytes() -> None:
    artifact = _estimate()
    manifest = artifact.manifest
    assert (
        manifest.payloads["likelihood.json"].sha256
        == hashlib.sha256(artifact.likelihood_bytes).hexdigest()
    )
    assert (
        manifest.payloads["payoff.json"].sha256
        == hashlib.sha256(artifact.payoff_bytes).hexdigest()
    )
    assert manifest.payloads["likelihood.json"].size_bytes == len(
        artifact.likelihood_bytes
    )
    assert manifest.payloads["payoff.json"].size_bytes == len(artifact.payoff_bytes)
    assert len(manifest.aggregate_digest) == 64


def test_estimation_is_deterministic() -> None:
    first = _estimate()
    second = _estimate()
    assert first.likelihood_bytes == second.likelihood_bytes
    assert first.payoff_bytes == second.payoff_bytes
    assert first.manifest == second.manifest
    assert first.aggregate_digest == second.aggregate_digest


# --------------------------------------------------------------------------- #
# Generated directory loads through the untouched S01 loader
# --------------------------------------------------------------------------- #
def _write_artifact(directory: Path, artifact) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "likelihood.json").write_bytes(artifact.likelihood_bytes)
    (directory / "payoff.json").write_bytes(artifact.payoff_bytes)
    (directory / MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(artifact.manifest.model_dump(by_alias=True, mode="json"))
    )
    return directory


def test_estimated_artifact_loads_through_s01(tmp_path: Path) -> None:
    artifact = _estimate()
    directory = _write_artifact(tmp_path / "artifact", artifact)
    loaded = load_bayesian_artifact(directory)
    assert loaded.manifest.support.prior_cells == 5
    assert loaded.likelihood_for("1", "mid", 0, (0, 0, 0)) == pytest.approx(
        2.0 / 9.0, abs=1e-15
    )
    assert loaded.payoff_for("1", "mid", 1, (1, 1, 1), 1.0) == pytest.approx(
        1.5, abs=1e-15
    )
    assert loaded.prior_probability(1, (1,)) == pytest.approx(1.0 / 3.0, abs=1e-15)


def test_real_collection_estimates_and_loads(tmp_path: Path) -> None:
    experiment = make_config(
        rounds=1,
        training={"local_epochs": 1, "batch_size": 8},
        data={"samples_per_node": 8},
        evaluation={"max_samples": 8, "batch_size": 8},
    )
    config = parse_dict(
        base_dict(
            experiment=experiment.model_dump(mode="json"),
            episodes=2,
            p0=0.5,
            design_split_size=8,
        )
    )
    collection = collect_calibration(
        config, dataset=MNISTShapedDataset(200, separable=True, seed=11)
    )
    artifact = estimate_calibration(config, collection)
    assert artifact.support.calibration_episodes == 2
    assert artifact.support.calibration_rounds == 6
    directory = _write_artifact(tmp_path / "real", artifact)
    loaded = load_bayesian_artifact(directory)
    assert loaded.manifest.schema_version == 1
    assert 0.0 <= loaded.manifest.coverage.posterior_calibration <= 1.0


# --------------------------------------------------------------------------- #
# Negative surface (Q7)
# --------------------------------------------------------------------------- #
def test_unsupported_context_is_rejected() -> None:
    records = list(_records())
    records[0] = _record(
        degree=1,
        degree_bin="99",
        progress="mid",
        evidence_bins=[(0, 0, 0)],
        hidden_types=[0],
        marginals=[_marginal("99", "mid", (0, 0, 0), 0, 1.0, 1.0)],
    )
    with pytest.raises(CalibrationEstimationError, match="degree_bin"):
        estimate_artifact(
            _config(),
            records,
            _prior_counts(),
            calibration_episodes=2,
            calibration_rounds=2,
        )


def test_evidence_cardinality_drift_is_rejected() -> None:
    records = list(_records())
    records[0] = _record(
        degree=1,
        degree_bin="1",
        progress="mid",
        evidence_bins=[(0, 0, 9)],
        hidden_types=[0],
        marginals=[_marginal("1", "mid", (0, 0, 0), 0, 1.0, 1.0)],
    )
    with pytest.raises(CalibrationEstimationError, match="out of range"):
        estimate_artifact(
            _config(),
            records,
            _prior_counts(),
            calibration_episodes=2,
            calibration_rounds=2,
        )


def test_record_length_mismatch_is_rejected() -> None:
    bad = _record(
        degree=1,
        degree_bin="1",
        progress="mid",
        evidence_bins=[(0, 0, 0), (1, 1, 1)],
        hidden_types=[0, 1],
        marginals=[],
    )
    with pytest.raises(CalibrationEstimationError, match="cardinality drift"):
        estimate_artifact(
            _config(),
            [bad],
            _prior_counts(),
            calibration_episodes=1,
            calibration_rounds=1,
        )


def test_non_finite_utility_is_rejected() -> None:
    record = _record(
        degree=1,
        degree_bin="1",
        progress="mid",
        evidence_bins=[(0, 0, 0)],
        hidden_types=[0],
        marginals=[_marginal("1", "mid", (0, 0, 0), 0, 0.5, float("inf"))],
    )
    with pytest.raises(CalibrationEstimationError, match="not finite"):
        estimate_artifact(
            _config(),
            [record],
            _prior_counts(),
            calibration_episodes=1,
            calibration_rounds=1,
        )


def test_undeclared_multiplier_is_rejected() -> None:
    record = _record(
        degree=1,
        degree_bin="1",
        progress="mid",
        evidence_bins=[(0, 0, 0)],
        hidden_types=[0],
        marginals=[_marginal("1", "mid", (0, 0, 0), 0, 0.25, 1.0)],
    )
    with pytest.raises(CalibrationEstimationError, match="action set"):
        estimate_artifact(
            _config(),
            [record],
            _prior_counts(),
            calibration_episodes=1,
            calibration_rounds=1,
        )


def test_empty_records_are_rejected() -> None:
    with pytest.raises(CalibrationEstimationError, match="likelihood support"):
        estimate_artifact(
            _config(),
            [],
            _prior_counts(),
            calibration_episodes=0,
            calibration_rounds=0,
        )


def test_unsupported_prior_state_is_rejected() -> None:
    bad_prior = PriorStateCounts(
        counts={1: {(1, 1): 1}}, episodes=1, benign_observations=1
    )
    with pytest.raises(CalibrationEstimationError, match="unsupported prior state"):
        estimate_artifact(
            _config(),
            _records(),
            bad_prior,
            calibration_episodes=2,
            calibration_rounds=2,
        )


def test_non_config_is_rejected() -> None:
    with pytest.raises(CalibrationEstimationError, match="CalibrationConfig"):
        estimate_artifact(  # type: ignore[arg-type]
            make_config(),
            _records(),
            _prior_counts(),
            calibration_episodes=2,
            calibration_rounds=2,
        )
