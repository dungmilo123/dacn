"""Strict offline calibration-config contract tests (T01, R013/R014, Q3/Q7).

All tests are offline and deterministic. The valid builder wraps a real
validated :class:`~dfl_sim.config.ExperimentConfig` produced by the existing
strict parser, so the calibration envelope is exercised through the same
contract as production.
"""

from __future__ import annotations

import json

import pytest
import yaml

from dfl_sim import (
    FULL_VARIANT,
    NO_BELIEF_VARIANT,
    NO_STRATEGIC_VARIANT,
    SOURCE_MU_A,
    CalibrationConfig,
    CalibrationConfigError,
    load_calibration_config,
    parse_calibration_text,
)
from helpers import make_config


def _bins() -> dict:
    return {
        "disagreement": {
            "edges": [0.25, 0.5, 0.75, 1.0],
            "labels": ["vlo", "lo", "mid", "hi", "vhi"],
        },
        "relative_norm": {
            "edges": [0.5, 1.0, 2.0],
            "labels": ["tiny", "small", "large", "huge"],
        },
        "loss_delta": {
            "edges": [-0.05, 0.0, 0.05],
            "labels": ["drop", "flatneg", "flatpos", "rise"],
        },
        "degree": {"labels": ["1", "2", "3-4", "5+"]},
        "progress": {"labels": ["early", "mid", "late"]},
    }


def _action_space() -> dict:
    return {
        "multiplier_set": [0.0, 0.5, 1.0],
        "self_multiplier": 1.0,
        "omega": 0.5,
        "degree_cap": 4,
        "tie_tolerance": 1e-9,
        "ordering": "fewer_rejected_then_lexicographic",
        "pruning": "directional_disagreement_desc_sender_id_asc",
    }


def base_dict(**overrides) -> dict:
    """Build a valid calibration config mapping with shallow overrides."""

    base: dict = {
        "experiment": make_config().model_dump(mode="json"),
        "episodes": 8,
        "p0": 0.5,
        "mu_a": [0.0, 2.0, 5.0],
        "max_compromised": 1,
        "smoothing_alpha": 1.0,
        "design_split_size": 32,
        "bins": _bins(),
        "action_space": _action_space(),
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def parse_dict(mapping: dict) -> CalibrationConfig:
    return parse_calibration_text(json.dumps(mapping, sort_keys=True))


# --------------------------------------------------------------------------- #
# Valid parsing paths
# --------------------------------------------------------------------------- #
def test_valid_canonical_json_parses() -> None:
    config = parse_dict(base_dict())
    assert isinstance(config, CalibrationConfig)
    assert config.episodes == 8
    assert config.p0 == 0.5
    assert tuple(config.mu_a) == (0.0, 2.0, 5.0)
    assert config.max_compromised == 1
    assert config.smoothing_alpha == 1.0
    assert config.design_split_size == 32
    assert config.experiment.rounds >= 1
    assert config.bins.progress.labels == ("early", "mid", "late")


def test_yaml_fallback_parses() -> None:
    text = yaml.safe_dump(base_dict(), sort_keys=True)
    config = parse_calibration_text(text)
    assert config.episodes == 8
    assert tuple(config.bins.disagreement.labels) == (
        "vlo",
        "lo",
        "mid",
        "hi",
        "vhi",
    )


def test_canonical_json_wins_over_yaml_exponent() -> None:
    # YAML 1.1 resolves a bare ``1e-09`` as a *string*; canonical JSON resolves
    # it as a float. The parser must try JSON first so the pinned tolerance
    # round-trips instead of failing strict float validation.
    mapping = base_dict()
    mapping["action_space"]["tie_tolerance"] = 1e-9
    text = json.dumps(mapping, sort_keys=True)
    assert "1e-09" in text
    config = parse_calibration_text(text)
    assert config.action_space.tie_tolerance == 1e-9


def test_load_from_file(tmp_path) -> None:
    path = tmp_path / "calibrate.yaml"
    path.write_text(yaml.safe_dump(base_dict(), sort_keys=True), encoding="utf-8")
    config = load_calibration_config(path)
    assert config.episodes == 8


def test_missing_file_raises(tmp_path) -> None:
    with pytest.raises(CalibrationConfigError, match="not found"):
        load_calibration_config(tmp_path / "absent.yaml")


# --------------------------------------------------------------------------- #
# Malformed structured input (Q7)
# --------------------------------------------------------------------------- #
def test_unknown_top_level_key_rejected() -> None:
    with pytest.raises(CalibrationConfigError, match="invalid calibration config"):
        parse_dict(base_dict(unexpected=1))


def test_unknown_experiment_key_rejected() -> None:
    experiment = make_config().model_dump(mode="json")
    experiment["surprise"] = True
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(experiment=experiment))


def test_empty_config_rejected() -> None:
    with pytest.raises(CalibrationConfigError, match="empty"):
        parse_calibration_text("")


def test_non_mapping_root_rejected() -> None:
    with pytest.raises(CalibrationConfigError, match="must be a mapping"):
        parse_calibration_text("[1, 2, 3]")


def test_malformed_yaml_rejected() -> None:
    # Not valid JSON, and an unclosed flow sequence is not valid YAML either.
    with pytest.raises(CalibrationConfigError, match="invalid YAML"):
        parse_calibration_text("a: [1, 2")


def test_non_finite_yaml_value_rejected() -> None:
    text = yaml.safe_dump(base_dict(), sort_keys=True).replace("0.5", ".nan", 1)
    with pytest.raises(CalibrationConfigError):
        parse_calibration_text(text)


def test_non_finite_json_constant_rejected() -> None:
    # json.dumps emits a bare NaN which strict JSON must refuse rather than
    # silently accept.
    raw = base_dict()
    raw["p0"] = float("nan")
    text = json.dumps(raw, sort_keys=True)
    with pytest.raises(CalibrationConfigError, match="non-finite"):
        parse_calibration_text(text)


def test_unsorted_bin_edges_rejected() -> None:
    bins = _bins()
    bins["disagreement"]["edges"] = [0.75, 0.25, 0.5, 1.0]
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(bins=bins))


def test_duplicate_bin_edges_rejected() -> None:
    bins = _bins()
    bins["relative_norm"]["edges"] = [0.5, 0.5, 2.0]
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(bins=bins))


def test_bin_label_cardinality_rejected() -> None:
    bins = _bins()
    bins["loss_delta"]["labels"] = ["only", "two"]
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(bins=bins))


def test_duplicate_bin_labels_rejected() -> None:
    bins = _bins()
    bins["disagreement"]["labels"] = ["lo", "lo", "mid", "hi", "vhi"]
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(bins=bins))


def test_degree_labels_pinned() -> None:
    bins = _bins()
    bins["degree"]["labels"] = ["1", "2", "3+"]
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(bins=bins))


def test_progress_labels_pinned() -> None:
    bins = _bins()
    bins["progress"]["labels"] = ["start", "end"]
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(bins=bins))


# --------------------------------------------------------------------------- #
# Envelope semantics (R014/Q4)
# --------------------------------------------------------------------------- #
def test_non_plain_average_experiment_rejected() -> None:
    experiment = make_config(aggregation={"kind": "local_only"})
    with pytest.raises(CalibrationConfigError, match="plain_average"):
        parse_dict(base_dict(experiment=experiment.model_dump(mode="json")))


def test_preselected_attacker_rejected() -> None:
    experiment = make_config(
        attack={"compromised_node": 2, "intensity": "weak"}
    )
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(experiment=experiment.model_dump(mode="json")))


def test_wrong_mixture_rejected() -> None:
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(mu_a=[0.0, 2.0]))


def test_single_episode_rejected() -> None:
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(episodes=1))


def test_non_positive_smoothing_rejected() -> None:
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(smoothing_alpha=0.0))


def test_design_split_must_be_positive() -> None:
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(design_split_size=0))


def test_insufficient_training_data_rejected() -> None:
    experiment = make_config(
        topology={"kind": "watts_strogatz", "n": 20, "k": 4, "p": 0.2},
        data={"samples_per_node": 4000},
    )
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(experiment=experiment.model_dump(mode="json")))


def test_pinned_omega_rejected() -> None:
    action_space = _action_space()
    action_space["omega"] = 0.4
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(action_space=action_space))


def test_pinned_degree_cap_rejected() -> None:
    action_space = _action_space()
    action_space["degree_cap"] = 3
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(action_space=action_space))


def test_multiplier_set_must_match_omega() -> None:
    action_space = _action_space()
    action_space["multiplier_set"] = [0.0, 0.25, 1.0]
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(action_space=action_space))


def test_error_text_is_bounded() -> None:
    # A giant unknown key produces a large pydantic dump; the raised message
    # must stay within a small fixed bound.
    with pytest.raises(CalibrationConfigError) as excinfo:
        parse_dict(base_dict(**{"x" * 5000: "y"}))
    assert len(str(excinfo.value)) <= 500


# --------------------------------------------------------------------------- #
# Ablation variant plumbing (S03/T01)
# --------------------------------------------------------------------------- #
def test_variant_defaults_to_full_and_derives_source_mixture() -> None:
    config = parse_dict(base_dict())
    assert config.variant == FULL_VARIANT
    assert tuple(config.mu_a) == SOURCE_MU_A
    assert config.effective_mu_a == SOURCE_MU_A


def test_no_belief_variant_keeps_source_mixture() -> None:
    config = parse_dict(base_dict(variant=NO_BELIEF_VARIANT))
    assert config.variant == NO_BELIEF_VARIANT
    assert config.effective_mu_a == SOURCE_MU_A


def test_no_strategic_variant_derives_pinned_strong_mixture() -> None:
    config = parse_dict(base_dict(variant=NO_STRATEGIC_VARIANT))
    assert config.variant == NO_STRATEGIC_VARIANT
    # The input mixture stays the pinned source protocol; only the derived
    # effective mixture changes.
    assert tuple(config.mu_a) == SOURCE_MU_A
    assert config.effective_mu_a == (5.0,)


@pytest.mark.parametrize("bad", [True, False, 1, 0, None, ["full"], "bogus"])
def test_invalid_variant_rejected(bad: object) -> None:
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(variant=bad))


def test_arbitrary_attacker_mixture_still_rejected_for_no_strategic() -> None:
    # no_strategic derives (5,); the canonical source mixture is still required
    # as input, so an arbitrary attacker setting cannot be smuggled in.
    with pytest.raises(CalibrationConfigError):
        parse_dict(base_dict(variant=NO_STRATEGIC_VARIANT, mu_a=[5.0]))
