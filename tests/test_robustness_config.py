"""Canonical robustness protocol envelope tests (S05/T02, R019/R020/D040).

These tests freeze the source protocol *before* any artifact digest exists:
nine real-MNIST conditions (three Dirichlet alphas x three seed-derived degree
placements), a disjoint selection/final seed split, a separate calibration seed,
four known schedules plus a final-only held-out schedule, singleton tuning
budget, and a pinned CPU SmallCNN envelope. All checks are pure and offline.
"""

from __future__ import annotations

import inspect
import re

import pytest
from pydantic import ValidationError

from dfl_sim import (
    CANONICAL_ALPHAS,
    CANONICAL_CALIBRATION_SEED,
    CANONICAL_FINAL_SEEDS,
    CANONICAL_PLACEMENTS,
    CANONICAL_SELECTION_SEEDS,
    CANONICAL_TOPOLOGY,
    MAX_CANDIDATE_BUDGET,
    MAX_ROUNDS,
    CanonicalRobustnessProtocol,
    HeldOutStudyConfig,
    PlacementContract,
    RobustnessConfigError,
    build_canonical_protocol,
    build_injected_protocol,
    intensity_name,
    realized_kappa_sequence,
)

HEADLINE_ROOT = "runs/m003-mnist-headline"
ARTIFACT_ROOT = "runs/m003-canonical-robustness/calibration"


def _canonical() -> CanonicalRobustnessProtocol:
    return build_canonical_protocol(
        headline_source_root=HEADLINE_ROOT, artifact_root=ARTIFACT_ROOT
    )


def _kwargs(**overrides):
    protocol = _canonical()
    values = {
        name: getattr(protocol, name)
        for name in type(protocol).model_fields
    }
    values.update(overrides)
    return values


# --------------------------------------------------------------------------- #
# Frozen envelope
# --------------------------------------------------------------------------- #
def test_canonical_protocol_builds_nine_conditions() -> None:
    conditions = _canonical().conditions()
    assert len(conditions) == 9
    assert {(c.alpha, c.placement) for c in conditions} == {
        (alpha, placement)
        for alpha in CANONICAL_ALPHAS
        for placement in CANONICAL_PLACEMENTS
    }
    assert len({c.condition_id for c in conditions}) == 9


def test_condition_specs_are_validated_held_out_studies() -> None:
    for condition in _canonical().conditions():
        spec = condition.spec
        assert isinstance(spec, HeldOutStudyConfig)
        study = spec.study
        assert isinstance(study.placement, PlacementContract)
        assert study.placement.placement == condition.placement
        assert study.placement.graph_kind == "watts_strogatz"
        # The canonical matrix deliberately opts into honest tie disclosure.
        assert study.placement.require_varied_degrees is False
        base = study.base
        assert base.data.partition == "dirichlet"
        assert base.data.dirichlet_alpha == condition.alpha
        assert base.topology.kind == "watts_strogatz"
        assert base.topology.n == 10
        assert base.topology.k == 2
        assert base.topology.p == 0.3
        assert base.device == "cpu"
        assert base.model.name == "small_cnn"
        assert base.rounds == 6
        assert base.data.samples_per_node == 64
        assert base.evaluation.max_samples == 128
        assert base.attack.intensity == "strong"


def test_seed_split_is_disjoint_and_calibration_seed_separate() -> None:
    for condition in _canonical().conditions():
        spec = condition.spec
        selection = set(spec.study.seeds)
        final = set(spec.final_seeds)
        assert selection == set(CANONICAL_SELECTION_SEEDS)
        assert final == set(CANONICAL_FINAL_SEEDS)
        assert selection.isdisjoint(final)
        # Injected floor is 5; real-MNIST floor is 3. The canonical split
        # satisfies the stricter injected floor.
        assert len(selection) + len(final) >= 5
        assert CANONICAL_CALIBRATION_SEED not in selection | final


def test_known_schedules_cover_intensities_and_held_out_is_final_only() -> None:
    for condition in _canonical().conditions():
        spec = condition.spec
        ids = [entry.id for entry in spec.known_schedules]
        assert len(ids) == 4
        assert len(set(ids)) == 4
        assert all(entry.schedule.kind != "held_out" for entry in spec.known_schedules)
        assert spec.held_out_schedule.schedule.kind == "held_out"
        assert spec.held_out_schedule.id not in set(ids)
        # The four known D033 schedules collectively realize zero/weak/strong.
        realized = set()
        for entry in spec.known_schedules:
            for kappa in realized_kappa_sequence(
                entry.schedule, spec.study.base.rounds, 0
            ):
                realized.add(intensity_name(kappa))
        assert realized == {"zero", "weak", "strong"}


def test_resource_and_identity_pins() -> None:
    protocol = _canonical()
    assert protocol.protocol_id == "m003-canonical-robustness"
    assert protocol.data_provenance == "real_mnist"
    assert protocol.headline_source_root == HEADLINE_ROOT
    assert protocol.artifact_root == ARTIFACT_ROOT
    assert protocol.threads == 1
    assert protocol.candidate_budget == 1 <= MAX_CANDIDATE_BUDGET
    assert protocol.rounds == 6 <= MAX_ROUNDS
    assert protocol.samples_per_node == 64
    assert protocol.evaluation_max_samples == 128
    assert protocol.calibration_seed == CANONICAL_CALIBRATION_SEED
    assert protocol.selection_seeds == CANONICAL_SELECTION_SEEDS
    assert protocol.final_seeds == CANONICAL_FINAL_SEEDS
    assert protocol.topology_n == CANONICAL_TOPOLOGY["n"] == 10
    assert protocol.topology_k == CANONICAL_TOPOLOGY["k"] == 2
    assert protocol.topology_p == CANONICAL_TOPOLOGY["p"] == 0.3
    assert protocol.alphas == CANONICAL_ALPHAS
    assert protocol.placements == CANONICAL_PLACEMENTS


def test_condition_study_ids_are_safe_and_unique() -> None:
    safe = re.compile(r"^[a-z0-9_-]+$")
    ids = []
    for condition in _canonical().conditions():
        study_id = condition.spec.study.study_id
        assert safe.match(study_id), study_id
        assert condition.placement in study_id
        ids.append(study_id)
    assert len(set(ids)) == 9


def test_public_factory_has_no_injected_escape_hatch() -> None:
    signature = inspect.signature(build_canonical_protocol)
    assert "data_provenance" not in signature.parameters
    assert "alphas" not in signature.parameters
    assert "placements" not in signature.parameters
    assert _canonical().data_provenance == "real_mnist"


# --------------------------------------------------------------------------- #
# Injected internal API
# --------------------------------------------------------------------------- #
def test_injected_factory_is_a_distinct_internal_api() -> None:
    injected = build_injected_protocol(artifact_root=ARTIFACT_ROOT)
    assert injected.data_provenance == "injected"
    assert injected.download is False
    conditions = injected.conditions()
    assert len(conditions) == 9
    assert all(
        condition.spec.study.data_provenance == "injected"
        for condition in conditions
    )


# --------------------------------------------------------------------------- #
# Placement resolution determinism and tie disclosure
# --------------------------------------------------------------------------- #
def test_placement_resolves_per_seed_and_discloses_ties() -> None:
    from dfl_sim.evaluation import _resolve_study_placements

    condition = _canonical().conditions()[0]
    study = condition.spec.study
    selection = _resolve_study_placements(study, study.seeds)
    final = _resolve_study_placements(study, condition.spec.final_seeds)
    assert set(selection) == set(CANONICAL_SELECTION_SEEDS)
    assert set(final) == set(CANONICAL_FINAL_SEEDS)
    digests = {r.graph_digest for r in selection.values()}
    digests |= {r.graph_digest for r in final.values()}
    # Seed-derived graphs vary; the resolver is deterministic but not constant.
    assert len(digests) > 1
    for resolution in (*selection.values(), *final.values()):
        assert resolution.placement == condition.placement
        # Ties are disclosed, never falsely claimed as distinct groups.
        if resolution.regular:
            assert resolution.distinct_degree_groups is False
        for key in ("node", "degree", "rank", "graph_digest"):
            assert key in resolution.identity_payload()


# --------------------------------------------------------------------------- #
# Invalid envelopes fail closed
# --------------------------------------------------------------------------- #
def test_envelope_rejects_non_singleton_threads() -> None:
    with pytest.raises(ValidationError):
        CanonicalRobustnessProtocol(**_kwargs(threads=2))


def test_envelope_rejects_download() -> None:
    with pytest.raises(ValidationError):
        CanonicalRobustnessProtocol(**_kwargs(download=True))


def test_envelope_rejects_overlapping_seeds() -> None:
    with pytest.raises(ValidationError):
        CanonicalRobustnessProtocol(**_kwargs(final_seeds=(1101,)))


def test_envelope_rejects_calibration_seed_collision() -> None:
    with pytest.raises(ValidationError):
        CanonicalRobustnessProtocol(**_kwargs(calibration_seed=1101))


def test_envelope_rejects_invalid_placements() -> None:
    with pytest.raises(ValidationError):
        CanonicalRobustnessProtocol(**_kwargs(placements=("low", "extreme")))
    with pytest.raises(ValidationError):
        CanonicalRobustnessProtocol(**_kwargs(placements=("low", "low")))


def test_envelope_rejects_invalid_alpha_and_topology() -> None:
    with pytest.raises(ValidationError):
        CanonicalRobustnessProtocol(**_kwargs(alphas=(0.0,)))
    with pytest.raises(ValidationError):
        CanonicalRobustnessProtocol(**_kwargs(alphas=(-0.1,)))
    with pytest.raises(ValidationError):
        CanonicalRobustnessProtocol(**_kwargs(topology_k=3))
    with pytest.raises(ValidationError):
        CanonicalRobustnessProtocol(**_kwargs(topology_k=10))


def test_envelope_rejects_out_of_range_resources() -> None:
    with pytest.raises(ValidationError):
        CanonicalRobustnessProtocol(**_kwargs(rounds=MAX_ROUNDS + 1))
    with pytest.raises(ValidationError):
        CanonicalRobustnessProtocol(**_kwargs(candidate_budget=0))


def test_real_mnist_matrix_requires_nine_conditions() -> None:
    # A real-MNIST envelope with a non-3x3 matrix is rejected at condition build.
    truncated = _canonical().model_copy(update={"alphas": (0.1, 0.5)})
    with pytest.raises(RobustnessConfigError):
        truncated.conditions()


def test_condition_spec_rejects_unknown_dimensions() -> None:
    protocol = _canonical()
    with pytest.raises(RobustnessConfigError):
        protocol.condition_spec(0.1, "extreme")
    with pytest.raises(RobustnessConfigError):
        protocol.condition_spec(0.25, "low")
