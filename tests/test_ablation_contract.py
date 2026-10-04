"""Calibration-ablation identity and derivation-metadata contract tests (S03/T01).

Offline and deterministic. These tests cover the pure identity helpers in
:mod:`dfl_sim.ablation`, the variant/effective-mixture assumptions (D036), the
schedule intensity normalization, and the strict optional
:class:`~dfl_sim.ablation.CalibrationDerivation` model. No files, network,
tensors, or GSD/ignored paths are read; every fixture is built in-process or
from the tracked calibration-config builder.
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from dfl_sim import (
    ABLATION_PROTOCOL_VERSION,
    CALIBRATION_VARIANTS,
    CANDIDATE_PROTOCOL,
    FULL_VARIANT,
    ISOLATION_STATEMENTS,
    NO_BELIEF_VARIANT,
    NO_STRATEGIC_EFFECTIVE_MU_A,
    NO_STRATEGIC_KAPPA,
    NO_STRATEGIC_VARIANT,
    SOURCE_MU_A,
    AblationError,
    CalibrationDerivation,
    build_schedule,
    calibration_config_digest,
    dataset_content_digest,
    effective_mixture,
    isolation_statement,
    membership_digest,
    normalize_calibration_config,
    schedule_pairing_digest,
    sha256_hex,
)
from helpers import make_config
from test_calibration_config import base_dict, parse_dict


# --------------------------------------------------------------------------- #
# Variant / mixture assumptions
# --------------------------------------------------------------------------- #
def test_effective_mixture_pins_source_and_derives_no_strategic() -> None:
    assert effective_mixture(FULL_VARIANT) == SOURCE_MU_A
    assert effective_mixture(NO_BELIEF_VARIANT) == SOURCE_MU_A
    assert effective_mixture(NO_STRATEGIC_VARIANT) == (5.0,)
    assert NO_STRATEGIC_EFFECTIVE_MU_A == (NO_STRATEGIC_KAPPA,)


@pytest.mark.parametrize("bad", [True, False, 1, 1.0, None, ["full"], "bogus"])
def test_effective_mixture_rejects_non_variants(bad: object) -> None:
    with pytest.raises(AblationError):
        effective_mixture(bad)  # type: ignore[arg-type]


def test_isolation_statement_is_canonical_per_variant() -> None:
    for variant in CALIBRATION_VARIANTS:
        assert isolation_statement(variant) == ISOLATION_STATEMENTS[variant]
    assert "not fully isolated" in ISOLATION_STATEMENTS[NO_STRATEGIC_VARIANT]
    with pytest.raises(AblationError):
        isolation_statement("nope")


# --------------------------------------------------------------------------- #
# Config-level variant plumbing
# --------------------------------------------------------------------------- #
def test_config_defaults_to_full_and_derives_effective_mixture() -> None:
    config = parse_dict(base_dict())
    assert config.variant == FULL_VARIANT
    assert tuple(config.mu_a) == SOURCE_MU_A
    assert config.effective_mu_a == SOURCE_MU_A

    no_belief = parse_dict(base_dict(variant=NO_BELIEF_VARIANT))
    assert no_belief.effective_mu_a == SOURCE_MU_A

    no_strategic = parse_dict(base_dict(variant=NO_STRATEGIC_VARIANT))
    assert tuple(no_strategic.mu_a) == SOURCE_MU_A
    assert no_strategic.effective_mu_a == NO_STRATEGIC_EFFECTIVE_MU_A


# --------------------------------------------------------------------------- #
# Schedule intensity-only normalization
# --------------------------------------------------------------------------- #
def _pairing(specs) -> list[tuple]:
    return [
        (spec.index, spec.seed, spec.stratum, spec.node) for spec in specs
    ]


def test_full_and_no_belief_schedules_are_identical() -> None:
    full = build_schedule(parse_dict(base_dict(episodes=9, p0=0.5)))
    no_belief = build_schedule(
        parse_dict(base_dict(episodes=9, p0=0.5, variant=NO_BELIEF_VARIANT))
    )
    assert full.specs == no_belief.specs
    assert full.summary == no_belief.summary


def test_no_strategic_keeps_pairing_and_pins_strong_attacker() -> None:
    full = build_schedule(parse_dict(base_dict(episodes=9, p0=0.5)))
    no_strategic = build_schedule(
        parse_dict(base_dict(episodes=9, p0=0.5, variant=NO_STRATEGIC_VARIANT))
    )

    assert _pairing(full.specs) == _pairing(no_strategic.specs)
    assert full.summary.clean == no_strategic.summary.clean
    assert full.summary.attacked == no_strategic.summary.attacked
    assert full.summary.placement_counts == no_strategic.summary.placement_counts

    attacked = [spec for spec in no_strategic.specs if spec.is_attacked]
    assert attacked
    assert all(spec.intensity == "strong" for spec in attacked)
    assert all(spec.value == NO_STRATEGIC_KAPPA for spec in attacked)
    assert dict(no_strategic.summary.intensity_counts) == {
        "strong": len(attacked)
    }

    # Pairing digest excludes intensity, so it must match across all variants.
    assert schedule_pairing_digest(full.specs) == schedule_pairing_digest(
        no_strategic.specs
    )


def test_schedule_pairing_digest_changes_with_seed() -> None:
    first = build_schedule(parse_dict(base_dict(episodes=6)))
    second = build_schedule(
        parse_dict(base_dict(episodes=6, experiment=make_config(seed=77).model_dump(mode="json")))
    )
    assert schedule_pairing_digest(first.specs) != schedule_pairing_digest(
        second.specs
    )


# --------------------------------------------------------------------------- #
# Path-independent, input-sensitive config identity
# --------------------------------------------------------------------------- #
def _config_body(**overrides) -> dict:
    return parse_dict(base_dict(**overrides)).model_dump(mode="json")


def test_config_digest_is_path_independent() -> None:
    body = _config_body()
    relocated = copy.deepcopy(body)
    relocated["experiment"]["data"]["data_root"] = "/some/other/environment"
    assert calibration_config_digest(body) == calibration_config_digest(relocated)


def test_config_digest_normalizes_variant_and_effective_intensity() -> None:
    full = calibration_config_digest(_config_body(variant=FULL_VARIANT))
    no_belief = calibration_config_digest(_config_body(variant=NO_BELIEF_VARIANT))
    no_strategic = calibration_config_digest(
        _config_body(variant=NO_STRATEGIC_VARIANT)
    )
    assert full == no_belief == no_strategic
    # The variant key itself is scrubbed from the normalized identity.
    assert "variant" not in normalize_calibration_config(_config_body())


def test_config_digest_is_input_sensitive() -> None:
    base = _config_body()

    seed_changed = _config_body(
        experiment=make_config(seed=424242).model_dump(mode="json")
    )
    smoothing_changed = copy.deepcopy(base)
    smoothing_changed["smoothing_alpha"] = 2.5
    bins_changed = copy.deepcopy(base)
    bins_changed["bins"]["degree"]["labels"] = ["1", "2", "3", "4"]
    action_changed = copy.deepcopy(base)
    action_changed["action_space"]["tie_tolerance"] = 1e-6

    baseline = calibration_config_digest(base)
    for mutated in (seed_changed, smoothing_changed, bins_changed, action_changed):
        assert calibration_config_digest(mutated) != baseline


def test_normalize_rejects_non_mapping() -> None:
    with pytest.raises(AblationError):
        normalize_calibration_config(["not", "a", "mapping"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Content fingerprints
# --------------------------------------------------------------------------- #
def test_dataset_content_digest_is_order_independent_and_content_sensitive() -> None:
    entries = [(0, 1, b"first"), (2, 7, b"second")]
    reordered = [(2, 7, b"second"), (0, 1, b"first")]
    assert dataset_content_digest(entries) == dataset_content_digest(reordered)

    assert dataset_content_digest(entries) != dataset_content_digest(
        [(0, 1, b"first"), (2, 7, b"changed")]
    )
    assert dataset_content_digest(entries) != dataset_content_digest(
        [(0, 2, b"first"), (2, 7, b"second")]
    )


def test_dataset_content_digest_rejects_duplicate_index() -> None:
    with pytest.raises(AblationError):
        dataset_content_digest([(0, 1, b"a"), (0, 1, b"b")])


def test_dataset_content_digest_rejects_bad_shapes() -> None:
    with pytest.raises(AblationError):
        dataset_content_digest([(0, 1)])  # type: ignore[list-item]
    with pytest.raises(AblationError):
        dataset_content_digest([(True, 1, b"a")])
    with pytest.raises(AblationError):
        dataset_content_digest([(0, 1, "not-bytes")])  # type: ignore[list-item]


def test_membership_digest_tracks_partitions_and_design() -> None:
    episode = [
        {"episode": 0, "nodes": {0: [3, 1], 1: [4]}, "design": [9, 8]},
        {"episode": 1, "nodes": {0: [5], 1: [6]}, "design": [7]},
    ]
    same = [
        {"episode": 0, "nodes": {1: [4], 0: [1, 3]}, "design": [8, 9]},
        {"episode": 1, "nodes": {0: [5], 1: [6]}, "design": [7]},
    ]
    assert membership_digest(episode) == membership_digest(same)

    changed_partition = copy.deepcopy(episode)
    changed_partition[0]["nodes"][0] = [3, 2]
    changed_design = copy.deepcopy(episode)
    changed_design[1]["design"] = [7, 10]
    baseline = membership_digest(episode)
    assert membership_digest(changed_partition) != baseline
    assert membership_digest(changed_design) != baseline


def test_sha256_hex_is_stable_and_rejects_non_bytes() -> None:
    assert sha256_hex(b"abc") == sha256_hex(b"abc")
    assert sha256_hex(b"abc") != sha256_hex(b"abd")
    with pytest.raises(AblationError):
        sha256_hex("abc")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Strict CalibrationDerivation model
# --------------------------------------------------------------------------- #
def _valid_derivation(**overrides: object) -> dict:
    base: dict = {
        "protocol_version": ABLATION_PROTOCOL_VERSION,
        "variant": FULL_VARIANT,
        "isolation_statement": ISOLATION_STATEMENTS[FULL_VARIANT],
        "source_mu_a": list(SOURCE_MU_A),
        "effective_mu_a": list(SOURCE_MU_A),
        "seed": 20240101,
        "calibration_config_digest": "a" * 64,
        "dataset_inputs_digest": "b" * 64,
        "partition_membership_digest": "c" * 64,
        "schedule_pairing_digest": "d" * 64,
        "episodes": 8,
        "rounds": 3,
        "design_size": 32,
        "candidate_protocol": CANDIDATE_PROTOCOL,
        "candidate_cap": 4,
        "data_provenance": "injected",
    }
    base.update(overrides)
    return base


def test_valid_derivation_model_round_trips_and_is_frozen() -> None:
    derivation = CalibrationDerivation.model_validate(_valid_derivation())
    assert derivation.variant == FULL_VARIANT
    assert tuple(derivation.effective_mu_a) == SOURCE_MU_A
    assert derivation.data_provenance == "injected"
    with pytest.raises(ValidationError):
        derivation.seed = 1  # type: ignore[misc]


def test_no_strategic_derivation_accepts_pinned_strong_mixture() -> None:
    derivation = CalibrationDerivation.model_validate(
        _valid_derivation(
            variant=NO_STRATEGIC_VARIANT,
            effective_mu_a=list(NO_STRATEGIC_EFFECTIVE_MU_A),
            isolation_statement=ISOLATION_STATEMENTS[NO_STRATEGIC_VARIANT],
        )
    )
    assert tuple(derivation.effective_mu_a) == (5.0,)


@pytest.mark.parametrize("bad", [True, False, 1, None, "bogus", ["full"]])
def test_derivation_rejects_non_variant(bad: object) -> None:
    with pytest.raises(ValidationError):
        CalibrationDerivation.model_validate(_valid_derivation(variant=bad))


def test_derivation_rejects_inconsistent_effective_mixture() -> None:
    with pytest.raises(ValidationError):
        CalibrationDerivation.model_validate(
            _valid_derivation(variant=FULL_VARIANT, effective_mu_a=[5.0])
        )
    with pytest.raises(ValidationError):
        CalibrationDerivation.model_validate(
            _valid_derivation(
                variant=NO_STRATEGIC_VARIANT,
                effective_mu_a=list(SOURCE_MU_A),
            )
        )


def test_derivation_rejects_wrong_isolation_statement() -> None:
    with pytest.raises(ValidationError):
        CalibrationDerivation.model_validate(
            _valid_derivation(isolation_statement="totally isolated")
        )


def test_derivation_rejects_wrong_protocol_version() -> None:
    with pytest.raises(ValidationError):
        CalibrationDerivation.model_validate(
            _valid_derivation(protocol_version="v0")
        )


def test_derivation_rejects_malformed_digest() -> None:
    with pytest.raises(ValidationError):
        CalibrationDerivation.model_validate(
            _valid_derivation(calibration_config_digest="not-a-digest")
        )
    with pytest.raises(ValidationError):
        CalibrationDerivation.model_validate(
            _valid_derivation(dataset_inputs_digest="A" * 64)
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_derivation_rejects_non_finite_numerics(bad: float) -> None:
    with pytest.raises(ValidationError):
        CalibrationDerivation.model_validate(
            _valid_derivation(source_mu_a=[bad, 2.0, 5.0])
        )


def test_derivation_rejects_non_finite_seed_budget() -> None:
    with pytest.raises(ValidationError):
        CalibrationDerivation.model_validate(_valid_derivation(episodes=float("nan")))
    with pytest.raises(ValidationError):
        CalibrationDerivation.model_validate(_valid_derivation(rounds=-1))
    with pytest.raises(ValidationError):
        CalibrationDerivation.model_validate(_valid_derivation(design_size=0))


def test_derivation_rejects_unknown_candidate_protocol() -> None:
    with pytest.raises(ValidationError):
        CalibrationDerivation.model_validate(
            _valid_derivation(candidate_protocol="arbitrary")
        )


def test_derivation_rejects_unknown_data_provenance() -> None:
    with pytest.raises(ValidationError):
        CalibrationDerivation.model_validate(
            _valid_derivation(data_provenance="synthetic")
        )


def test_derivation_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        CalibrationDerivation.model_validate(_valid_derivation(surprise=True))
