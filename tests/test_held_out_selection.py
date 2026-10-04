"""Selection-phase runner and frozen selection output tests (M003/S02/T03).

These tests drive the **production** selection composition closure offline:

    held-out spec -> SelectionSpec projection -> per-(seed, schedule) adaptive
    controller -> run_from_config -> schema-v1 TraceReader validation ->
    atomic frozen selection manifest

The input type is :class:`~dfl_sim.evaluation_config.SelectionSpec`, which
structurally carries no ``final_seeds`` or ``held_out_schedule``; the tests
assert that no-leak property directly and prove an overlapping split is
rejected before any episode runs. Dataset and evaluator reference are tiny
in-memory MNIST-shaped tensors, and the Bayesian cell loads the git-tracked
M002 fixture, so no network access occurs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

import dfl_sim.evaluation as evaluation_module
from dfl_sim import (
    BALANCE_CITATION,
    REQUIRED_POLICY_IDS,
    SELECTION_MANIFEST_FILENAME,
    SELECTION_MANIFEST_SCHEMA,
    SELECTION_MANIFEST_SCHEMA_VERSION,
    EvaluationError,
    HeldOutStudyConfigError,
    SelectionSpec,
    TraceReader,
    TrajectoryError,
    compute_shared_input_digest,
    derive_schedule_seed,
    intensity_name,
    parse_held_out_study_text,
    parse_study_text,
    realized_kappa_sequence,
    run_selection_phase,
    schedule_digest,
    selection_output_digest,
    selection_projection,
    selection_spec_digest,
    study_digest,
)
from helpers import MNISTShapedDataset, make_config
from dfl_sim.ablation_evaluation import ABLATION_METADATA_KEY

REPO_ROOT = Path(__file__).resolve().parents[1]
BAYESIAN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"

_TUNED_PROVENANCE = "selection-seed-budget"
_SELECTION_SEEDS = (11,)
_FINAL_SEEDS = (21, 22, 23, 24)


# --------------------------------------------------------------------------- #
# Fixture builders (mirror the S02/T02 held-out shape)
# --------------------------------------------------------------------------- #
def _base(**overrides):
    defaults = {
        "rounds": 6,
        "data": {"samples_per_node": 4},
        "training": {"local_epochs": 1, "batch_size": 4},
        "evaluation": {"max_samples": 4, "batch_size": 4},
        "attack": {"compromised_node": 0, "intensity": "strong"},
    }
    defaults.update(overrides)
    return make_config(**defaults)


def _panel(*, artifact_dir=None):
    artifact = str(BAYESIAN_FIXTURE if artifact_dir is None else artifact_dir)
    project = {"kind": "project", "title": "project comparator"}
    tuned = {
        "mode": "tuned",
        "candidate_budget": 3,
        "selection_seed": 7,
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


def _study_mapping(*, seeds=_SELECTION_SEEDS):
    return {
        "study_id": "held_out_fixture",
        "data_provenance": "injected",
        "base": _base().model_dump(mode="json"),
        "seeds": list(seeds),
        "candidate_budget": 3,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
        "panel": _panel(),
    }


def _known_schedules():
    return [
        {"id": "periodic_known", "schedule": {"kind": "periodic"}},
        {"id": "duty_known", "schedule": {"kind": "duty_cycle"}},
    ]


def _held_out_mapping(*, selection_seeds=_SELECTION_SEEDS, final_seeds=_FINAL_SEEDS):
    return {
        "study": _study_mapping(seeds=selection_seeds),
        "final_seeds": list(final_seeds),
        "known_schedules": _known_schedules(),
        "held_out_schedule": {
            "id": "held_out_schedule",
            "schedule": {"kind": "held_out"},
        },
    }


def _held_out(*, selection_seeds=_SELECTION_SEEDS, final_seeds=_FINAL_SEEDS):
    return parse_held_out_study_text(
        yaml.safe_dump(
            _held_out_mapping(
                selection_seeds=selection_seeds, final_seeds=final_seeds
            ),
            sort_keys=True,
        )
    )


def _injected_selection(**kwargs):
    selection = selection_projection(_held_out(**kwargs))
    return selection


def _run(selection, output, **overrides):
    return run_selection_phase(
        selection,
        output,
        dataset=MNISTShapedDataset(64, seed=1),
        evaluation_dataset=MNISTShapedDataset(32, seed=2),
        **overrides,
    )


@pytest.fixture(scope="module")
def selection_run(tmp_path_factory):
    output = tmp_path_factory.mktemp("held-out-selection") / "selection"
    return _run(_injected_selection(), output)


# --------------------------------------------------------------------------- #
# Structural no-leak property
# --------------------------------------------------------------------------- #
def test_selection_spec_structurally_hides_final_seeds() -> None:
    selection = _injected_selection()
    assert isinstance(selection, SelectionSpec)
    assert not hasattr(selection, "final_seeds")
    assert not hasattr(selection, "held_out_schedule")
    assert "final_seeds" not in SelectionSpec.model_fields
    assert "held_out_schedule" not in SelectionSpec.model_fields
    assert set(SelectionSpec.model_fields) == {"study", "known_schedules"}


def test_selection_rejects_non_selection_spec(tmp_path: Path) -> None:
    study = parse_study_text(
        yaml.safe_dump(_study_mapping(), sort_keys=True)
    )
    with pytest.raises(EvaluationError):
        run_selection_phase(
            study,  # type: ignore[arg-type]
            tmp_path / "selection",
            dataset=MNISTShapedDataset(64, seed=1),
            evaluation_dataset=MNISTShapedDataset(32, seed=2),
        )


def test_overlapping_split_rejected_before_any_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple] = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("run_from_config must never be reached")

    monkeypatch.setattr(evaluation_module, "run_from_config", forbidden)
    mapping = _held_out_mapping(
        selection_seeds=(11,), final_seeds=(11, 21, 22, 23)
    )
    with pytest.raises(HeldOutStudyConfigError) as info:
        parse_held_out_study_text(yaml.safe_dump(mapping, sort_keys=True))
    assert "overlap" in str(info.value) or "disjoint" in str(info.value)
    assert calls == []
    assert not (tmp_path / "selection").exists()


# --------------------------------------------------------------------------- #
# Frozen selection publication
# --------------------------------------------------------------------------- #
def test_selection_manifest_is_frozen_and_complete(selection_run) -> None:
    manifest = selection_run.manifest
    assert manifest["schema"] == SELECTION_MANIFEST_SCHEMA
    assert manifest["schema_version"] == SELECTION_MANIFEST_SCHEMA_VERSION
    assert manifest["status"] == "frozen"
    assert manifest["study_id"] == "held_out_fixture"
    assert manifest["data_provenance"] == "injected"
    assert manifest["panel"] == list(REQUIRED_POLICY_IDS)
    assert manifest["selection_seeds"] == list(_SELECTION_SEEDS)
    assert manifest["candidate_budget"] == 3
    assert manifest["schedule_ids"] == ["duty_known", "periodic_known"]

    selection = selection_run.selection
    assert manifest["study_digest"] == study_digest(selection.study)
    assert manifest["selection_spec_digest"] == selection_spec_digest(selection)
    assert manifest["shared_input_digest"] == compute_shared_input_digest(
        selection.study.base
    )

    expected_cells = (
        len(_SELECTION_SEEDS)
        * len(selection.known_schedules)
        * len(REQUIRED_POLICY_IDS)
    )
    assert len(manifest["entries"]) == expected_cells
    assert len(selection_run.cells) == expected_cells

    # The manifest is written at the selection output root.
    assert selection_run.manifest_path == (
        selection_run.output_path / SELECTION_MANIFEST_FILENAME
    )
    assert selection_run.manifest_path.is_file()
    assert selection_run.manifest_path.parent == selection_run.output_path


def test_selection_manifest_has_no_ablation_metadata(selection_run) -> None:
    manifest = selection_run.manifest
    assert manifest["panel"] == list(REQUIRED_POLICY_IDS)
    assert ABLATION_METADATA_KEY not in manifest
    assert selection_run.ablation is None
    for entry in manifest["entries"]:
        assert ABLATION_METADATA_KEY not in entry


def test_selection_entries_are_complete_and_scheduled(selection_run) -> None:
    selection = selection_run.selection
    schedules = {entry.id: entry for entry in selection.known_schedules}
    for entry in selection_run.manifest["entries"]:
        assert entry["status"] == "complete"
        assert entry["seed"] == 11
        assert entry["policy_kind"] == entry["policy_id"]
        assert entry["config_digest"]
        schedule = schedules[entry["schedule_id"]]
        assert entry["schedule"] == schedule.schedule.model_dump(mode="json")
        assert entry["schedule_digest"] == schedule_digest(schedule.schedule)
        assert entry["pairing_key"] == (
            f"{entry['policy_id']}:seed-{entry['seed']}:{entry['schedule_id']}"
        )
        assert entry["trace_path"].startswith("traces/")


def test_selection_traces_are_readable_schema_v1(selection_run) -> None:
    assert all(cell.trace_path.is_dir() for cell in selection_run.cells)
    relative = [cell.relative_trace_path for cell in selection_run.cells]
    assert len(relative) == len(set(relative))
    assert all(".." not in part for part in relative)
    for cell in selection_run.cells:
        with TraceReader(cell.trace_path) as reader:
            assert reader.schema_version == 1
            assert reader.config.seed == cell.seed
            assert reader.config.aggregation.kind == cell.policy_id


def test_selection_records_realized_kappa_sequences(selection_run) -> None:
    selection = selection_run.selection
    rounds = selection.study.base.rounds
    schedules = {entry.id: entry for entry in selection.known_schedules}
    realized_intensities: set[str] = set()
    per_schedule_sequences: dict[str, set[tuple[float, ...]]] = {}
    for cell in selection_run.cells:
        schedule = schedules[cell.schedule_id]
        expected = realized_kappa_sequence(
            schedule.schedule,
            rounds,
            derive_schedule_seed(cell.seed, schedule.schedule),
        )
        assert cell.realized_kappa_sequence == expected
        assert tuple(cell.manifest_entry["realized_kappa_sequence"]) == expected
        per_schedule_sequences.setdefault(cell.schedule_id, set()).add(expected)
        for kappa in expected:
            realized_intensities.add(intensity_name(kappa))

    # The known schedules collectively realize the full D016 intensity table.
    assert {"zero", "weak", "strong"} <= realized_intensities
    # The recorded sequences are genuinely non-stationary, not a constant label,
    # and there is exactly one realized sequence per (seed, schedule).
    for sequences in per_schedule_sequences.values():
        assert len(sequences) == 1
        assert len(set(next(iter(sequences)))) > 1


def test_selection_pairs_identical_attack_across_policies(selection_run) -> None:
    groups: dict[tuple[int, str], set[tuple[float, ...]]] = {}
    counts: dict[tuple[int, str], int] = {}
    for cell in selection_run.cells:
        key = (cell.seed, cell.schedule_id)
        groups.setdefault(key, set()).add(cell.realized_kappa_sequence)
        counts[key] = counts.get(key, 0) + 1
    # Every policy in a (seed, schedule) cell shares the identical realized
    # attack, and every cell carries the full six-policy panel.
    assert all(len(sequences) == 1 for sequences in groups.values())
    assert set(counts.values()) == {len(REQUIRED_POLICY_IDS)}


def test_selection_manifest_digest_recomputes(selection_run) -> None:
    manifest = selection_run.manifest
    assert manifest["selection_output_digest"] == selection_output_digest(
        manifest
    )
    on_disk = json.loads(selection_run.manifest_path.read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert on_disk["selection_output_digest"] == selection_output_digest(on_disk)


def test_selection_output_is_byte_stable(tmp_path: Path) -> None:
    selection = _injected_selection()
    first = _run(selection, tmp_path / "first")
    second = _run(selection, tmp_path / "second")
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.manifest == second.manifest


def test_selection_progress_hook_reports_every_cell(tmp_path: Path) -> None:
    # A single-schedule projection is enough to prove per-cell traversal.
    base_selection = _injected_selection()
    selection = SelectionSpec(
        study=base_selection.study,
        known_schedules=(base_selection.known_schedules[0],),
    )
    seen: list[tuple[int, str, str]] = []
    _run(
        selection,
        tmp_path / "progress",
        progress=lambda seed, schedule_id, policy_id: seen.append(
            (seed, schedule_id, policy_id)
        ),
    )
    assert len(seen) == len(REQUIRED_POLICY_IDS)
    assert {policy_id for _, _, policy_id in seen} == set(REQUIRED_POLICY_IDS)
    assert {schedule_id for _, schedule_id, _ in seen} == {
        selection.known_schedules[0].id
    }


# --------------------------------------------------------------------------- #
# Failure modes: never publish a partial selection
# --------------------------------------------------------------------------- #
def test_selection_requires_injected_dataset(tmp_path: Path) -> None:
    selection = _injected_selection()
    with pytest.raises(EvaluationError):
        run_selection_phase(selection, tmp_path / "a")
    with pytest.raises(EvaluationError):
        run_selection_phase(
            selection,
            tmp_path / "b",
            dataset=MNISTShapedDataset(64, seed=1),
        )


def test_selection_existing_output_rejected(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(EvaluationError):
        _run(_injected_selection(), output)


def test_selection_middle_cell_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "selection"
    real = evaluation_module.run_from_config

    def flaky(config, **kwargs):
        if config.aggregation.kind == "balance":
            raise RuntimeError("injected mid-selection failure")
        return real(config, **kwargs)

    monkeypatch.setattr(evaluation_module, "run_from_config", flaky)
    with pytest.raises(EvaluationError) as info:
        _run(_injected_selection(), output)
    message = str(info.value)
    assert "balance" in message
    assert "11" in message
    assert not output.exists()
    assert not (output / SELECTION_MANIFEST_FILENAME).exists()
    assert [n for n in os.listdir(tmp_path) if "staging" in n] == []


def test_selection_invalid_trace_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "selection"

    def bad_validator(_path):
        raise TrajectoryError("injected invalid trace")

    with pytest.raises(EvaluationError):
        _run(_injected_selection(), output, trace_validator=bad_validator)
    assert not output.exists()
    assert not (output / SELECTION_MANIFEST_FILENAME).exists()
    assert [n for n in os.listdir(tmp_path) if "staging" in n] == []
