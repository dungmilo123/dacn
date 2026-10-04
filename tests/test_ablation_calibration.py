"""Offline integration tests for the three calibration artifact variants (S03/T02).

These tests drive the **real production calibration path**
(``calibrate_from_config`` -> collect -> estimate -> publish -> loader) over an
explicitly injected deterministic in-process dataset for the ``full``,
``no_belief`` and ``no_strategic`` variants. They prove:

* each newly calibrated artifact is loader-valid and carries self-describing,
  digest-covered derivation metadata;
* the three variants share one path-independent calibration-config identity and
  identical dataset/partition/design/schedule-pairing fingerprints, while their
  aggregate digests are distinct;
* ``full`` and ``no_belief`` share byte-identical payoff tables, differing only
  in the exact uniform prior/likelihood and the honestly non-applicable belief
  metric;
* the direct Bayesian posterior over distinct evidence bins is exactly uniform
  for ``no_belief``;
* the ``no_strategic`` fixed strong-attacker behaviour is observed in the
  *collected* schedule, not merely a metadata tag;
* the loader and estimator fail closed on relabelled variants, nonuniform
  no-belief tables, wrong effective mixtures, and malformed provenance.

No network, no MNIST download, no GSD/ignored paths: the dataset is generated
in-process and every artifact is written under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import pytest

from dfl_sim import (
    FULL_VARIANT,
    ISOLATION_STATEMENTS,
    NO_BELIEF_VARIANT,
    NO_STRATEGIC_EFFECTIVE_MU_A,
    NO_STRATEGIC_VARIANT,
    SOURCE_MU_A,
    CalibrationEstimationError,
    CollectionProvenance,
    build_schedule,
    calibrate_from_config,
    collect_calibration,
    count_prior_states,
    estimate_artifact,
    publish_calibration_artifact,
)
from dfl_sim.bayesian_artifact import (
    LIKELIHOOD_FILENAME,
    MANIFEST_FILENAME,
    PAYOFF_FILENAME,
    BayesianArtifactError,
    canonical_json_bytes,
    compute_aggregate_digest,
    load_bayesian_artifact,
)
from dfl_sim.calibration_estimator import build_derivation
from dfl_sim.observation import LocalObservation, NeighborObservation
from dfl_sim.policy import build_bayesian_policy
from helpers import MNISTShapedDataset, make_config
from test_calibration_config import base_dict, parse_dict

#: Fixed seed chosen (and asserted below) so a benign degree-2 receiver in the
#: chain3 topology observes a malicious neighbor during attacked episodes.
SEED = 20240101
EPISODES = 8
DESIGN_SPLIT = 8

#: Representative degree for each degree-bin label (chain3 only reaches 1 and 2).
_DEGREE_BY_BIN = {"1": 1, "2": 2, "3-4": 3, "5+": 5}


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
def _experiment(seed: int = SEED):
    return make_config(
        seed=seed,
        rounds=1,
        training={"local_epochs": 1, "batch_size": 8},
        data={"samples_per_node": 8},
        evaluation={"max_samples": 8, "batch_size": 8},
    )


def _config(variant: str, *, seed: int = SEED):
    return parse_dict(
        base_dict(
            experiment=_experiment(seed).model_dump(mode="json"),
            episodes=EPISODES,
            p0=0.5,
            design_split_size=DESIGN_SPLIT,
            variant=variant,
        )
    )


def _dataset() -> MNISTShapedDataset:
    return MNISTShapedDataset(400, separable=True, seed=11)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Calibrate all three variants once through the production path."""

    dataset = _dataset()
    artifacts: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for variant in (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT):
        dest = tmp_path_factory.mktemp("calibration") / variant
        calibrate_from_config(_config(variant), str(dest), dataset=dataset)
        paths[variant] = dest
        artifacts[variant] = load_bayesian_artifact(dest)
    return {"artifacts": artifacts, "paths": paths}


def _rewrite_artifact(
    source: Path,
    dest: Path,
    mutate: Callable[[dict, list, list], None],
) -> Path:
    """Rewrite an artifact with recomputed, self-consistent digests.

    This represents a hostile artifact whose JSON is structurally valid and
    whose declared digests match its bytes, so only the loader's *semantic*
    variant/table consistency checks can reject it.
    """

    dest.mkdir(parents=True)
    manifest = json.loads((source / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    likelihood = json.loads((source / LIKELIHOOD_FILENAME).read_text(encoding="utf-8"))
    payoff = json.loads((source / PAYOFF_FILENAME).read_text(encoding="utf-8"))
    mutate(manifest, likelihood, payoff)

    likelihood_bytes = canonical_json_bytes(likelihood)
    payoff_bytes = canonical_json_bytes(payoff)
    manifest["payloads"] = {
        LIKELIHOOD_FILENAME: {
            "size_bytes": len(likelihood_bytes),
            "sha256": hashlib.sha256(likelihood_bytes).hexdigest(),
        },
        PAYOFF_FILENAME: {
            "size_bytes": len(payoff_bytes),
            "sha256": hashlib.sha256(payoff_bytes).hexdigest(),
        },
    }
    manifest["aggregate_digest"] = compute_aggregate_digest(manifest)
    (dest / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
    (dest / LIKELIHOOD_FILENAME).write_bytes(likelihood_bytes)
    (dest / PAYOFF_FILENAME).write_bytes(payoff_bytes)
    return dest


def _feature_value(
    bins,
    index: int,
    *,
    zero_floor: bool = False,
    upper: float | None = None,
) -> float:
    """Return a scalar falling in interval bin ``index`` (matches bisect_right).

    The representative respects the observation's field constraints: a
    ``zero_floor`` field never goes negative and an ``upper`` bound (2.0 for
    directional disagreement) falls back to the exact top edge when the natural
    representative would exceed it.
    """

    edges = list(bins.edges)
    if index == 0:
        value = edges[0] - 1.0 if edges else -1.0
        if zero_floor:
            value = max(value, 0.0)
        if edges and value >= edges[0]:
            value = edges[0] / 2.0
        return value
    if index == len(edges):
        value = edges[-1] + 1.0 if edges else 1.0
        if upper is not None and value > upper:
            # The exact top edge still bins to the top index (bisect_right).
            value = edges[-1]
        return value
    return (edges[index - 1] + edges[index]) / 2.0


def _observation(artifact, degree: int, evidence: tuple[int, ...]) -> LocalObservation:
    bins = artifact.manifest.bins
    disagreement = _feature_value(
        bins.disagreement, evidence[0], zero_floor=True, upper=2.0
    )
    relative_norm = _feature_value(bins.relative_norm, evidence[1], zero_floor=True)
    loss_delta = _feature_value(bins.loss_delta, evidence[2])
    neighbors = tuple(
        NeighborObservation(7 + position, disagreement, relative_norm, loss_delta)
        for position in range(degree)
    )
    return LocalObservation(
        node_id=0, round_index=0, epsilon=1e-12, neighbors=neighbors
    )


def _supported_evidence(artifact):
    """Find a (degree, evidence) cell with payoff rows for both local types."""

    by_cell: dict[tuple[str, str, tuple[int, ...]], set] = {}
    for degree_bin, progress, local_type, evidence, _multiplier in artifact.payoff_index:
        by_cell.setdefault((degree_bin, progress, evidence), set()).add(local_type)
    for (degree_bin, progress, evidence), types in sorted(by_cell.items()):
        if types != {0, 1} or progress != "early":
            continue
        degree = _DEGREE_BY_BIN[degree_bin]
        if degree in artifact.prior_index:
            return degree, evidence
    return None


# --------------------------------------------------------------------------- #
# Positive: self-describing, matched, distinct artifacts
# --------------------------------------------------------------------------- #
def test_three_variants_are_loader_valid_and_self_describing(pipeline) -> None:
    artifacts = pipeline["artifacts"]
    assert set(artifacts) == {FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT}

    digests = {}
    for variant, artifact in artifacts.items():
        derivation = artifact.manifest.derivation
        assert derivation is not None
        assert derivation.variant == variant
        assert tuple(derivation.source_mu_a) == SOURCE_MU_A
        assert derivation.isolation_statement == ISOLATION_STATEMENTS[variant]
        assert derivation.seed == SEED
        assert derivation.data_provenance == "injected"
        assert derivation.design_size == DESIGN_SPLIT
        assert derivation.episodes == EPISODES
        assert derivation.rounds == artifact.manifest.support.calibration_rounds
        assert derivation.candidate_cap == artifact.manifest.action_space.degree_cap
        digests[variant] = artifact.manifest.aggregate_digest

    expected_effective = {
        FULL_VARIANT: SOURCE_MU_A,
        NO_BELIEF_VARIANT: SOURCE_MU_A,
        NO_STRATEGIC_VARIANT: NO_STRATEGIC_EFFECTIVE_MU_A,
    }
    for variant, expected in expected_effective.items():
        artifact = artifacts[variant]
        assert tuple(artifact.manifest.provenance.mu_a) == expected
        assert tuple(artifact.manifest.derivation.effective_mu_a) == expected

    # Distinct self-describing variants never share an aggregate digest.
    assert len(set(digests.values())) == len(digests)


def test_matched_input_identity_across_variants(pipeline) -> None:
    artifacts = pipeline["artifacts"]
    config_digests = {
        variant: artifact.manifest.derivation.calibration_config_digest
        for variant, artifact in artifacts.items()
    }
    dataset_digests = {
        variant: artifact.manifest.derivation.dataset_inputs_digest
        for variant, artifact in artifacts.items()
    }
    membership_digests = {
        variant: artifact.manifest.derivation.partition_membership_digest
        for variant, artifact in artifacts.items()
    }
    pairing_digests = {
        variant: artifact.manifest.derivation.schedule_pairing_digest
        for variant, artifact in artifacts.items()
    }
    assert len(set(config_digests.values())) == 1
    assert len(set(dataset_digests.values())) == 1
    assert len(set(membership_digests.values())) == 1
    assert len(set(pairing_digests.values())) == 1

    # Declared budgets are equal across all three (Q6: no smaller subset).
    for name in ("episodes", "rounds", "design_size", "candidate_cap"):
        values = {
            getattr(artifact.manifest.derivation, name)
            for artifact in artifacts.values()
        }
        assert len(values) == 1, name
    # Actual support is measured separately and also equal for full/no_belief.
    assert (
        artifacts[FULL_VARIANT].manifest.support.likelihood_cells
        == artifacts[NO_BELIEF_VARIANT].manifest.support.likelihood_cells
    )
    assert (
        artifacts[FULL_VARIANT].manifest.support.payoff_cells
        == artifacts[NO_BELIEF_VARIANT].manifest.support.payoff_cells
    )


def test_full_and_no_belief_share_byte_identical_payoff_tables(pipeline) -> None:
    full = pipeline["artifacts"][FULL_VARIANT]
    no_belief = pipeline["artifacts"][NO_BELIEF_VARIANT]
    assert full.payoff_index == no_belief.payoff_index
    assert (
        full.manifest.payloads[PAYOFF_FILENAME].sha256
        == no_belief.manifest.payloads[PAYOFF_FILENAME].sha256
    )
    assert full.manifest.bins == no_belief.manifest.bins
    assert full.manifest.action_space == no_belief.manifest.action_space
    # The belief tables and diagnostics are what differ.
    assert full.manifest.aggregate_digest != no_belief.manifest.aggregate_digest
    assert full.manifest.payloads[LIKELIHOOD_FILENAME].sha256 != (
        no_belief.manifest.payloads[LIKELIHOOD_FILENAME].sha256
    )


# --------------------------------------------------------------------------- #
# no_belief: exact uniform prior/likelihood and uniform posterior
# --------------------------------------------------------------------------- #
def test_no_belief_prior_and_likelihood_are_exactly_uniform(pipeline) -> None:
    artifact = pipeline["artifacts"][NO_BELIEF_VARIANT]
    assert artifact.manifest.coverage.posterior_calibration_applicable is False

    by_degree: dict[int, list] = defaultdict(list)
    for cell in artifact.manifest.prior:
        by_degree[cell.degree].append(cell)
    for degree, cells in by_degree.items():
        assert len(cells) == degree + 1
        assert {cell.probability for cell in cells} == {1.0 / (degree + 1)}

    bins = artifact.manifest.bins
    support_size = (
        (len(bins.disagreement.edges) + 1)
        * (len(bins.relative_norm.edges) + 1)
        * (len(bins.loss_delta.edges) + 1)
    )
    by_context: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for (degree_bin, progress, local_type, _evidence), probability in (
        artifact.likelihood_index.items()
    ):
        by_context[(degree_bin, progress, local_type)].append(probability)
    assert by_context
    for probabilities in by_context.values():
        assert len(probabilities) == support_size
        assert {round(value, 15) for value in probabilities} == {
            round(1.0 / support_size, 15)
        }


def test_no_belief_posterior_is_uniform_for_distinct_evidence(pipeline) -> None:
    artifact = pipeline["artifacts"][NO_BELIEF_VARIANT]

    def posterior_for(degree: int, evidence: tuple[int, ...]):
        sink: list = []
        policy = build_bayesian_policy(artifact, 1, diagnostic_sink=sink.append)
        policy.select_weights(_observation(artifact, degree, evidence))
        return {entry.state: entry.probability for entry in sink[0].posterior}

    # Degree 1: two distinct evidence cells must both yield {(): 0.5, (1,): 0.5}
    # independent of the evidence, because the likelihood is uniform and the
    # prior is uniform over the two feasible states.
    first = posterior_for(1, (0, 0, 0))
    second = posterior_for(1, (1, 1, 1))
    assert first == second == {(0,): 0.5, (1,): 0.5}

    # Degree 2: three feasible states each receive exactly one third.
    degree2 = posterior_for(2, (1, 1, 1))
    assert degree2 == {
        (0, 0): pytest.approx(1.0 / 3.0),
        (1, 0): pytest.approx(1.0 / 3.0),
        (0, 1): pytest.approx(1.0 / 3.0),
    }


# --------------------------------------------------------------------------- #
# Informative fixture + non-fallback full action
# --------------------------------------------------------------------------- #
def test_fixture_has_benign_degree2_receiver_of_malicious_neighbor() -> None:
    collection = collect_calibration(_config(FULL_VARIANT), dataset=_dataset())
    # chain3: node 1 has degree 2; any attacked episode whose compromised node
    # is not 1 leaves node 1 benign with a directly adjacent malicious sender.
    assert any(
        spec.is_attacked and spec.node != 1 for spec in collection.schedule.specs
    )
    # The collected records include a degree-2 benign receiver.
    assert any(record.degree == 2 for record in collection.records)


def test_full_artifact_produces_a_non_fallback_action(pipeline) -> None:
    artifact = pipeline["artifacts"][FULL_VARIANT]
    # Type-1 payoff support exists.
    assert any(key[2] == 1 for key in artifact.payoff_index)

    supported = _supported_evidence(artifact)
    assert supported is not None, "fixture has no jointly supported evidence cell"
    degree, evidence = supported

    sink: list = []
    policy = build_bayesian_policy(artifact, 1, diagnostic_sink=sink.append)
    profile = policy.select_weights(_observation(artifact, degree, evidence))

    diagnostic = sink[0]
    assert diagnostic.fallback_reason is None
    assert diagnostic.posterior
    # WeightProfile invariants hold without widening the online seam.
    assert profile.multipliers[0] == 1.0
    assert sum(profile.weights) == pytest.approx(1.0)
    assert profile.participant_ids == (0,) + tuple(
        neighbor.sender_id for neighbor in _observation(artifact, degree, evidence).neighbors
    )


# --------------------------------------------------------------------------- #
# no_strategic: fixed strong attacker observed in the collected schedule
# --------------------------------------------------------------------------- #
def test_no_strategic_fixed_schedule_and_fresh_tables() -> None:
    dataset = _dataset()
    full = collect_calibration(_config(FULL_VARIANT), dataset=dataset)
    no_strategic = collect_calibration(_config(NO_STRATEGIC_VARIANT), dataset=dataset)

    # Pairing, clean/attacked counts and placements are unchanged.
    full_pairing = [
        (spec.index, spec.seed, spec.stratum, spec.node)
        for spec in full.schedule.specs
    ]
    no_strategic_pairing = [
        (spec.index, spec.seed, spec.stratum, spec.node)
        for spec in no_strategic.schedule.specs
    ]
    assert full_pairing == no_strategic_pairing
    assert full.schedule.summary.clean == no_strategic.schedule.summary.clean
    assert full.schedule.summary.attacked == no_strategic.schedule.summary.attacked

    attacked = [spec for spec in no_strategic.schedule.specs if spec.is_attacked]
    assert attacked
    assert all(spec.intensity == "strong" and spec.value == 5.0 for spec in attacked)

    # Dataset/membership/pairing fingerprints are preserved; the effective
    # mixture and the actual trajectories are what change.
    assert full.provenance.dataset_inputs_digest == no_strategic.provenance.dataset_inputs_digest
    assert full.provenance.partition_membership_digest == no_strategic.provenance.partition_membership_digest
    assert full.provenance.schedule_pairing_digest == no_strategic.provenance.schedule_pairing_digest
    assert [spec.value for spec in full.schedule.specs if spec.is_attacked] != [
        spec.value for spec in no_strategic.schedule.specs if spec.is_attacked
    ]

    # Re-estimation from the fresh records produces a distinct artifact.
    full_artifact = estimate_artifact(
        _config(FULL_VARIANT),
        full.records,
        count_prior_states(_config(FULL_VARIANT), full.schedule.specs),
        calibration_episodes=full.episode_count,
        calibration_rounds=full.round_count,
        provenance=full.provenance,
    )
    no_strategic_artifact = estimate_artifact(
        _config(NO_STRATEGIC_VARIANT),
        no_strategic.records,
        count_prior_states(_config(NO_STRATEGIC_VARIANT), no_strategic.schedule.specs),
        calibration_episodes=no_strategic.episode_count,
        calibration_rounds=no_strategic.round_count,
        provenance=no_strategic.provenance,
    )
    assert tuple(no_strategic_artifact.manifest.provenance.mu_a) == (
        NO_STRATEGIC_EFFECTIVE_MU_A
    )
    assert (
        full_artifact.manifest.aggregate_digest
        != no_strategic_artifact.manifest.aggregate_digest
    )


# --------------------------------------------------------------------------- #
# Legacy compatibility + estimator provenance validation
# --------------------------------------------------------------------------- #
def test_legacy_artifact_without_derivation_still_loads(tmp_path: Path) -> None:
    dataset = _dataset()
    config = _config(FULL_VARIANT)
    collection = collect_calibration(config, dataset=dataset)
    artifact = estimate_artifact(
        config,
        collection.records,
        count_prior_states(config, collection.schedule.specs),
        calibration_episodes=collection.episode_count,
        calibration_rounds=collection.round_count,
    )
    assert artifact.manifest.derivation is None
    destination = tmp_path / "legacy"
    publish_calibration_artifact(destination, artifact)
    loaded = load_bayesian_artifact(destination)
    assert loaded.manifest.derivation is None
    assert loaded.manifest.coverage.posterior_calibration_applicable is True


def _synthetic_prior_counts():
    from dfl_sim.calibration import PriorStateCounts

    return PriorStateCounts(
        counts={1: {(0,): 1, (1,): 1}},
        episodes=1,
        benign_observations=2,
    )


def test_build_derivation_rejects_bad_provenance() -> None:
    bad = CollectionProvenance(
        data_provenance="synthetic",
        dataset_inputs_digest="a" * 64,
        partition_membership_digest="b" * 64,
        schedule_pairing_digest="c" * 64,
        consumed_examples=1,
    )
    with pytest.raises(CalibrationEstimationError, match="derivation metadata invalid"):
        build_derivation(_config(FULL_VARIANT), bad, calibration_episodes=8, calibration_rounds=24)


def test_build_derivation_rejects_non_provenance() -> None:
    with pytest.raises(CalibrationEstimationError, match="CollectionProvenance"):
        build_derivation(  # type: ignore[arg-type]
            _config(FULL_VARIANT), object(), calibration_episodes=8, calibration_rounds=24
        )


def test_estimate_artifact_rejects_bad_provenance_type() -> None:
    with pytest.raises(CalibrationEstimationError, match="CollectionProvenance"):
        estimate_artifact(  # type: ignore[arg-type]
            _config(FULL_VARIANT),
            [],
            _synthetic_prior_counts(),
            calibration_episodes=1,
            calibration_rounds=1,
            provenance=object(),
        )


def test_no_belief_estimation_requires_provenance() -> None:
    with pytest.raises(
        CalibrationEstimationError, match="no_belief estimation requires"
    ):
        estimate_artifact(
            _config(NO_BELIEF_VARIANT),
            [],
            _synthetic_prior_counts(),
            calibration_episodes=1,
            calibration_rounds=1,
        )


# --------------------------------------------------------------------------- #
# Negative: loader rejects relabelled / inconsistent variant tables (Q5/Q7)
# --------------------------------------------------------------------------- #
def _expect_derivation_rejection(path: Path) -> None:
    with pytest.raises(BayesianArtifactError) as excinfo:
        load_bayesian_artifact(path)
    assert excinfo.value.validation_class == "derivation"
    assert excinfo.value.filename in (MANIFEST_FILENAME, LIKELIHOOD_FILENAME)


def test_loader_rejects_full_relabelled_as_no_belief(pipeline, tmp_path: Path) -> None:
    source = pipeline["paths"][FULL_VARIANT]

    def mutate(manifest: dict, _lik: list, _pay: list) -> None:
        derivation = manifest["derivation"]
        derivation["variant"] = NO_BELIEF_VARIANT
        derivation["isolation_statement"] = ISOLATION_STATEMENTS[NO_BELIEF_VARIANT]
        derivation["effective_mu_a"] = list(SOURCE_MU_A)
        manifest["coverage"]["posterior_calibration_applicable"] = False

    dest = _rewrite_artifact(source, tmp_path / "relabelled", mutate)
    _expect_derivation_rejection(dest)


def test_loader_rejects_nonuniform_no_belief_prior(pipeline, tmp_path: Path) -> None:
    source = pipeline["paths"][NO_BELIEF_VARIANT]

    def mutate(manifest: dict, _lik: list, _pay: list) -> None:
        degree_one = [row for row in manifest["prior"] if row["degree"] == 1]
        degree_one[0]["probability"] = 0.6
        degree_one[1]["probability"] = 0.4

    dest = _rewrite_artifact(source, tmp_path / "nonuniform", mutate)
    _expect_derivation_rejection(dest)


def test_loader_rejects_evidence_dependent_no_belief_likelihood(
    pipeline, tmp_path: Path
) -> None:
    source = pipeline["paths"][NO_BELIEF_VARIANT]

    def mutate(_manifest: dict, likelihood: list, _pay: list) -> None:
        likelihood[0]["probability"] = 0.5

    dest = _rewrite_artifact(source, tmp_path / "dependent", mutate)
    _expect_derivation_rejection(dest)


def test_loader_rejects_wrong_no_strategic_effective_mixture(
    pipeline, tmp_path: Path
) -> None:
    source = pipeline["paths"][NO_STRATEGIC_VARIANT]

    def mutate(manifest: dict, _lik: list, _pay: list) -> None:
        manifest["provenance"]["mu_a"] = list(SOURCE_MU_A)

    dest = _rewrite_artifact(source, tmp_path / "wrong_mixture", mutate)
    _expect_derivation_rejection(dest)


def test_loader_rejects_misreported_belief_applicability(
    pipeline, tmp_path: Path
) -> None:
    source = pipeline["paths"][NO_BELIEF_VARIANT]

    def mutate(manifest: dict, _lik: list, _pay: list) -> None:
        manifest["coverage"]["posterior_calibration_applicable"] = True

    dest = _rewrite_artifact(source, tmp_path / "misreported", mutate)
    _expect_derivation_rejection(dest)


# --------------------------------------------------------------------------- #
# Negative: corrupted records / missing payoff support fail closed (Q5)
# --------------------------------------------------------------------------- #
def test_estimation_rejects_empty_payoff_support(pipeline) -> None:
    config = _config(FULL_VARIANT)
    collection = collect_calibration(config, dataset=_dataset())
    # Strip every marginal so the payoff table has no support.
    stripped = []
    for record in collection.records:
        stripped.append(
            type(record)(
                node_id=record.node_id,
                round_index=record.round_index,
                degree=record.degree,
                degree_bin=record.degree_bin,
                progress=record.progress,
                participant_ids=record.participant_ids,
                sender_ids=record.sender_ids,
                evidence_bins=record.evidence_bins,
                hidden_types=record.hidden_types,
                self_design_loss=record.self_design_loss,
                plain_full_multipliers=record.plain_full_multipliers,
                candidates=record.candidates,
                marginals=(),
            )
        )
    with pytest.raises(CalibrationEstimationError, match="payoff support"):
        estimate_artifact(
            config,
            stripped,
            count_prior_states(config, collection.schedule.specs),
            calibration_episodes=collection.episode_count,
            calibration_rounds=collection.round_count,
            provenance=collection.provenance,
        )


def test_no_strategic_uses_same_full_schedule_shape() -> None:
    full = build_schedule(_config(FULL_VARIANT))
    no_strategic = build_schedule(_config(NO_STRATEGIC_VARIANT))
    assert [
        (spec.index, spec.seed, spec.stratum, spec.node) for spec in full.specs
    ] == [
        (spec.index, spec.seed, spec.stratum, spec.node) for spec in no_strategic.specs
    ]
