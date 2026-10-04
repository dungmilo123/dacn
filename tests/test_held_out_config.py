"""Held-out study contract: disjoint split, projection, and digest tests (S02/T02).

The held-out study is the fairness envelope for the S02 runner: it composes the
S01 :class:`BaselineStudyConfig` (whose ``seeds`` are the selection seeds) with a
disjoint ``final_seeds`` tuple, the known adaptive schedules the selection phase
may build, and the reserved unseen ``held_out_schedule`` that only the final
phase may build. :class:`SelectionSpec` projects only the selection-side content,
so final seeds and the held-out schedule are structurally unreachable during
selection (milestone no-leak rule, R018/R022).

Everything here is offline and in-process: specs are strict JSON/YAML mappings,
the base reuses the production parser, and no runner or dataset is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dfl_sim import (
    BALANCE_CITATION,
    MAX_KNOWN_SCHEDULES,
    MAX_STUDY_SEEDS,
    MIN_TOTAL_SEEDS_INJECTED,
    MIN_TOTAL_SEEDS_REAL_MNIST,
    BaselineStudyConfig,
    BaselineStudyConfigError,
    HeldOutStudyConfig,
    HeldOutStudyConfigError,
    SelectionSpec,
    held_out_study_digest,
    load_any_study_config,
    load_held_out_study_config,
    load_study_config,
    parse_held_out_study_text,
    parse_study_text,
    selection_projection,
    selection_spec_digest,
)
from helpers import make_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BAYESIAN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"

_TUNED_PROVENANCE = "selection-seed-budget"


# --------------------------------------------------------------------------- #
# Fixture builders (mirror the S01 study shape, plus adaptive schedules)
# --------------------------------------------------------------------------- #
def _base(**overrides):
    defaults = {
        "rounds": 6,
        "data": {"samples_per_node": 8},
        "training": {"local_epochs": 1, "batch_size": 4},
        "evaluation": {"max_samples": 8, "batch_size": 4},
        "attack": {"compromised_node": 0, "intensity": "strong"},
    }
    defaults.update(overrides)
    return make_config(**defaults)


def _panel(*, artifact_dir=None, tuned_budget=3, tuned_seed=7):
    artifact = str(BAYESIAN_FIXTURE if artifact_dir is None else artifact_dir)
    project = {"kind": "project", "title": "project comparator"}
    tuned = {
        "mode": "tuned",
        "candidate_budget": tuned_budget,
        "selection_seed": tuned_seed,
        "selection_provenance": _TUNED_PROVENANCE,
    }
    exempt = {
        "mode": "exempt",
        "candidate_budget": 0,
        "selection_seed": None,
        "selection_provenance": None,
    }
    return [
        {
            "id": "local_only",
            "implementation": "local only",
            "aggregation": {"kind": "local_only"},
            "citation": project,
            "tuning": exempt,
        },
        {
            "id": "plain_average",
            "implementation": "plain average",
            "aggregation": {"kind": "plain_average"},
            "citation": project,
            "tuning": exempt,
        },
        {
            "id": "fixed_similarity",
            "implementation": "fixed similarity",
            "aggregation": {
                "kind": "fixed_similarity",
                "cutoff": 0.5,
                "downweight_multiplier": 0.5,
            },
            "citation": project,
            "tuning": tuned,
        },
        {
            "id": "trust_weighted",
            "implementation": "trust weighted",
            "aggregation": {"kind": "trust_weighted", "beta": 1.0},
            "citation": project,
            "tuning": tuned,
        },
        {
            "id": "balance",
            "implementation": "faithful BALANCE",
            "aggregation": {"kind": "balance"},
            "citation": dict(BALANCE_CITATION),
            "tuning": tuned,
        },
        {
            "id": "bayesian",
            "implementation": "M002 Bayesian",
            "aggregation": {
                "kind": "bayesian",
                "artifact_dir": artifact,
                "omega": 0.5,
                "degree_cap": 4,
                "tie_tolerance": 1e-9,
            },
            "citation": project,
            "tuning": tuned,
        },
    ]


def _study_mapping(
    *,
    seeds=(11, 12, 13),
    artifact_dir=None,
    data_provenance="injected",
    base=None,
    **overrides,
):
    mapping = {
        "study_id": "held_out_fixture",
        "data_provenance": data_provenance,
        "base": (base or _base()).model_dump(mode="json"),
        "seeds": list(seeds),
        "candidate_budget": 3,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
        "panel": _panel(artifact_dir=artifact_dir),
    }
    mapping.update(overrides)
    return mapping


def _known_schedules():
    # periodic default (period=6) realizes strong/weak/zero over 6 rounds.
    return [
        {"id": "periodic_known", "schedule": {"kind": "periodic"}},
        {"id": "duty_known", "schedule": {"kind": "duty_cycle"}},
    ]


def _held_out_schedule():
    return {"id": "held_out_schedule", "schedule": {"kind": "held_out"}}


def _held_out_mapping(
    *,
    selection_seeds=(11, 12, 13),
    final_seeds=(21, 22),
    known_schedules=None,
    held_out_schedule=None,
    study_mapping=None,
    **overrides,
):
    study = study_mapping or _study_mapping(seeds=selection_seeds)
    mapping = {
        "study": study,
        "final_seeds": list(final_seeds),
        "known_schedules": (
            _known_schedules() if known_schedules is None else known_schedules
        ),
        "held_out_schedule": (
            _held_out_schedule()
            if held_out_schedule is None
            else held_out_schedule
        ),
    }
    mapping.update(overrides)
    return mapping


def _parse(mapping) -> HeldOutStudyConfig:
    return parse_held_out_study_text(yaml.safe_dump(mapping, sort_keys=True))


def _invalid(mapping) -> HeldOutStudyConfigError:
    with pytest.raises(HeldOutStudyConfigError) as info:
        _parse(mapping)
    return info.value


# --------------------------------------------------------------------------- #
# Round trip and structure
# --------------------------------------------------------------------------- #
def test_held_out_spec_parses_and_round_trips() -> None:
    study = _parse(_held_out_mapping())
    assert study.study.study_id == "held_out_fixture"
    assert study.study.seeds == (11, 12, 13)
    assert study.final_seeds == (21, 22)
    assert [entry.id for entry in study.known_schedules] == [
        "periodic_known",
        "duty_known",
    ]
    assert study.held_out_schedule.id == "held_out_schedule"
    assert study.held_out_schedule.schedule.kind == "held_out"

    snapshot = json.dumps(study.model_dump(mode="json"))
    assert held_out_study_digest(parse_held_out_study_text(snapshot)) == (
        held_out_study_digest(study)
    )


def test_ordered_known_schedules_is_stable() -> None:
    study = _parse(_held_out_mapping())
    assert [entry.id for entry in study.ordered_known_schedules()] == [
        "duty_known",
        "periodic_known",
    ]


# --------------------------------------------------------------------------- #
# Split guards: overlap / short / duplicate / out-of-range final seeds
# --------------------------------------------------------------------------- #
def test_final_seeds_overlapping_selection_rejected() -> None:
    message = str(
        _invalid(_held_out_mapping(selection_seeds=(11, 12, 13), final_seeds=(13, 21)))
    )
    assert "disjoint" in message or "overlap" in message


def test_short_final_seeds_rejected() -> None:
    assert "at least one seed" in str(_invalid(_held_out_mapping(final_seeds=())))


def test_duplicate_final_seeds_rejected() -> None:
    assert "unique" in str(_invalid(_held_out_mapping(final_seeds=(21, 21))))


def test_out_of_range_final_seed_rejected() -> None:
    assert "outside" in str(_invalid(_held_out_mapping(final_seeds=(2**31,))))


def test_too_many_final_seeds_rejected() -> None:
    seeds = tuple(range(100, 100 + MAX_STUDY_SEEDS + 1))
    assert "limit" in str(_invalid(_held_out_mapping(final_seeds=seeds)))


# --------------------------------------------------------------------------- #
# Schedule plan guards
# --------------------------------------------------------------------------- #
def test_held_out_id_colliding_with_known_rejected() -> None:
    known = _known_schedules()
    mapping = _held_out_mapping(
        known_schedules=known,
        held_out_schedule={"id": "periodic_known", "schedule": {"kind": "held_out"}},
    )
    assert "collides" in str(_invalid(mapping))


def test_duplicate_known_schedule_ids_rejected() -> None:
    known = [
        {"id": "same", "schedule": {"kind": "periodic"}},
        {"id": "same", "schedule": {"kind": "duty_cycle"}},
    ]
    assert "unique" in str(_invalid(_held_out_mapping(known_schedules=known)))


def test_reserved_held_out_kind_in_known_schedules_rejected() -> None:
    known = [
        {"id": "periodic_known", "schedule": {"kind": "periodic"}},
        {"id": "sneaky", "schedule": {"kind": "held_out"}},
    ]
    assert "reserved" in str(_invalid(_held_out_mapping(known_schedules=known)))


def test_empty_known_schedules_rejected() -> None:
    assert "at least one schedule" in str(
        _invalid(_held_out_mapping(known_schedules=[]))
    )


def test_too_many_known_schedules_rejected() -> None:
    known = [
        {"id": f"s{index}", "schedule": {"kind": "duty_cycle"}}
        for index in range(MAX_KNOWN_SCHEDULES + 1)
    ]
    assert "limit" in str(_invalid(_held_out_mapping(known_schedules=known)))


def test_held_out_schedule_duplicate_of_known_rejected() -> None:
    mapping = _held_out_mapping(
        held_out_schedule={"id": "held_out_schedule", "schedule": {"kind": "periodic"}}
    )
    # A periodic held-out schedule duplicates the known periodic schedule.
    assert "distinct" in str(_invalid(mapping))


# --------------------------------------------------------------------------- #
# Intensity coverage and attack provenance
# --------------------------------------------------------------------------- #
def test_zero_weak_strong_coverage_failure_rejected() -> None:
    # duty_cycle alone realizes only strong and zero (no weak).
    known = [{"id": "duty_only", "schedule": {"kind": "duty_cycle"}}]
    message = str(_invalid(_held_out_mapping(known_schedules=known)))
    assert "collectively realize" in message
    assert "weak" in message


def test_attack_intensity_must_match_peak_realized() -> None:
    mapping = _held_out_mapping()
    mapping["study"]["base"]["attack"]["intensity"] = "weak"
    assert "peak realized intensity" in str(_invalid(mapping))


def test_attack_provenance_requires_compromised_node() -> None:
    mapping = _held_out_mapping()
    mapping["study"]["base"]["attack"] = {
        "compromised_node": None,
        "intensity": "strong",
    }
    assert "compromised_node" in str(_invalid(mapping))


# --------------------------------------------------------------------------- #
# R022 seed floors
# --------------------------------------------------------------------------- #
def test_injected_provenance_seed_floor_rejected() -> None:
    mapping = _held_out_mapping(selection_seeds=(11,), final_seeds=(21,))
    assert str(MIN_TOTAL_SEEDS_INJECTED) in str(_invalid(mapping))


def test_injected_provenance_seed_floor_accepted_at_boundary() -> None:
    study = _parse(
        _held_out_mapping(selection_seeds=(11, 12, 13), final_seeds=(21, 22))
    )
    assert len(study.study.seeds) + len(study.final_seeds) == MIN_TOTAL_SEEDS_INJECTED


def test_real_mnist_provenance_seed_floor_rejected() -> None:
    mapping = _held_out_mapping(
        study_mapping=_study_mapping(
            seeds=(11,),
            data_provenance="real_mnist",
            base=_base(data={"download": True}),
        ),
        final_seeds=(21,),
    )
    assert str(MIN_TOTAL_SEEDS_REAL_MNIST) in str(_invalid(mapping))


def test_real_mnist_provenance_seed_floor_accepted() -> None:
    study = _parse(
        _held_out_mapping(
            study_mapping=_study_mapping(
                seeds=(11,),
                data_provenance="real_mnist",
                base=_base(data={"download": True}),
            ),
            final_seeds=(21, 22),
        )
    )
    assert study.study.data_provenance == "real_mnist"
    assert len(study.final_seeds) == 2


# --------------------------------------------------------------------------- #
# Selection projection and digests
# --------------------------------------------------------------------------- #
def test_selection_spec_exposes_no_final_or_held_out_attribute() -> None:
    selection = selection_projection(_parse(_held_out_mapping()))
    assert isinstance(selection, SelectionSpec)
    assert not hasattr(selection, "final_seeds")
    assert not hasattr(selection, "held_out_schedule")
    assert "final_seeds" not in SelectionSpec.model_fields
    assert "held_out_schedule" not in SelectionSpec.model_fields
    assert set(SelectionSpec.model_fields) == {"study", "known_schedules"}


def test_selection_projection_preserves_selection_content() -> None:
    study = _parse(_held_out_mapping())
    selection = selection_projection(study)
    assert selection.study == study.study
    assert selection.known_schedules == study.known_schedules


def test_selection_spec_digest_invariant_to_final_seeds() -> None:
    first = selection_projection(
        _parse(_held_out_mapping(final_seeds=(21, 22)))
    )
    second = selection_projection(
        _parse(_held_out_mapping(final_seeds=(31, 32)))
    )
    assert first == second
    assert selection_spec_digest(first) == selection_spec_digest(second)
    assert held_out_study_digest(
        _parse(_held_out_mapping(final_seeds=(21, 22)))
    ) != held_out_study_digest(_parse(_held_out_mapping(final_seeds=(31, 32))))


def test_selection_spec_digest_sensitive_to_selection_content() -> None:
    first = selection_projection(_parse(_held_out_mapping()))
    second = selection_projection(
        _parse(_held_out_mapping(selection_seeds=(41, 42, 43)))
    )
    assert selection_spec_digest(first) != selection_spec_digest(second)


def test_selection_projection_rejects_wrong_type() -> None:
    with pytest.raises(HeldOutStudyConfigError):
        selection_projection(object())  # type: ignore[arg-type]


def test_selection_spec_round_trips_through_json() -> None:
    selection = selection_projection(_parse(_held_out_mapping()))
    snapshot = json.dumps(selection.model_dump(mode="json"))
    parsed = SelectionSpec.model_validate(json.loads(snapshot))
    assert selection_spec_digest(parsed) == selection_spec_digest(selection)


# --------------------------------------------------------------------------- #
# Strict parsing: unknown keys, non-finite values, malformed input
# --------------------------------------------------------------------------- #
def test_unknown_top_level_key_rejected() -> None:
    mapping = _held_out_mapping()
    mapping["unexpected"] = 1
    assert "unexpected" in str(_invalid(mapping))


def test_unknown_study_key_rejected() -> None:
    mapping = _held_out_mapping()
    mapping["study"]["unexpected"] = 1
    assert "unexpected" in str(_invalid(mapping))


def test_unknown_schedule_key_rejected() -> None:
    known = [{"id": "periodic_known", "schedule": {"kind": "periodic", "junk": 1}}]
    assert "junk" in str(_invalid(_held_out_mapping(known_schedules=known)))


def test_non_finite_schedule_field_rejected() -> None:
    known = [
        {"id": "periodic_known", "schedule": {"kind": "periodic"}},
        {"id": "random_known", "schedule": {"kind": "random", "p_on": float("nan")}},
    ]
    assert "finite" in str(_invalid(_held_out_mapping(known_schedules=known))).lower()


def test_malformed_json_and_yaml_rejected() -> None:
    with pytest.raises(HeldOutStudyConfigError):
        parse_held_out_study_text("{")
    with pytest.raises(HeldOutStudyConfigError):
        parse_held_out_study_text(
            "study: !!python/object/apply:os.system ['echo hacked']"
        )
    with pytest.raises(HeldOutStudyConfigError):
        parse_held_out_study_text("not a mapping")


def test_error_text_is_bounded() -> None:
    mapping = _held_out_mapping()
    mapping["study"]["study_id"] = "A" * 5000
    message = str(_invalid(mapping))
    assert len(message) < 800


# --------------------------------------------------------------------------- #
# Loaders and dispatch
# --------------------------------------------------------------------------- #
def test_load_held_out_study_config_round_trip(tmp_path: Path) -> None:
    spec = tmp_path / "held_out.yaml"
    spec.write_text(
        yaml.safe_dump(_held_out_mapping(), sort_keys=True), encoding="utf-8"
    )
    loaded = load_held_out_study_config(spec)
    assert isinstance(loaded, HeldOutStudyConfig)
    assert held_out_study_digest(loaded) == held_out_study_digest(
        _parse(_held_out_mapping())
    )


def test_load_any_study_config_dispatches_both_kinds(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.yaml"
    baseline_path.write_text(
        yaml.safe_dump(_study_mapping(), sort_keys=True), encoding="utf-8"
    )
    held_out_path = tmp_path / "held_out.yaml"
    held_out_path.write_text(
        yaml.safe_dump(_held_out_mapping(), sort_keys=True), encoding="utf-8"
    )

    baseline = load_any_study_config(baseline_path)
    held_out = load_any_study_config(held_out_path)
    assert isinstance(baseline, BaselineStudyConfig)
    assert isinstance(held_out, HeldOutStudyConfig)

    # The baseline remains loadable through its own strict loader unchanged.
    assert isinstance(load_study_config(baseline_path), BaselineStudyConfig)
    assert isinstance(parse_study_text(baseline_path.read_text()), BaselineStudyConfig)


def test_load_any_study_config_reports_combined_failure(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("mystery: 1", encoding="utf-8")
    with pytest.raises(BaselineStudyConfigError) as info:
        load_any_study_config(bad)
    message = str(info.value)
    assert "neither baseline nor held-out" in message


def test_load_held_out_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(HeldOutStudyConfigError):
        load_held_out_study_config(tmp_path / "does-not-exist.yaml")
