"""Final-phase runner, immutable freeze guards, and publication (M003/S02/T04).

These tests drive the **production** final composition offline:

    frozen selection output -> HeldOutStudyConfig -> mutated-spec / split
    guards -> per-(seed, schedule) adaptive controller over known + reserved
    held-out schedules -> run_from_config -> schema-v1 TraceReader validation
    -> atomic final manifest

The final runner accepts only a :class:`~dfl_sim.evaluation_config.HeldOutStudyConfig`
and a previously frozen selection output. It refuses a missing/incomplete
selection output, a mutated selection-side spec, and an overlapping split, and
it is the only phase that builds the reserved unseen ``held_out_schedule``.
Dataset and evaluator reference are tiny in-memory MNIST-shaped tensors, and the
Bayesian cell loads the git-tracked M002 fixture, so no network access occurs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import dfl_sim.evaluation as evaluation_module
from dfl_sim import (
    BALANCE_CITATION,
    FINAL_MANIFEST_FILENAME,
    FINAL_MANIFEST_SCHEMA,
    FINAL_MANIFEST_SCHEMA_VERSION,
    REQUIRED_POLICY_IDS,
    SELECTION_MANIFEST_FILENAME,
    SELECTION_MANIFEST_SCHEMA,
    EvaluationError,
    HeldOutStudyConfigError,
    TraceReader,
    TrajectoryError,
    compute_shared_input_digest,
    derive_schedule_seed,
    final_output_digest,
    held_out_study_digest,
    parse_held_out_study_text,
    realized_kappa_sequence,
    run_final_phase,
    run_selection_phase,
    schedule_digest,
    selection_projection,
    selection_spec_digest,
    study_digest,
)
from helpers import MNISTShapedDataset, make_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BAYESIAN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"

_TUNED_PROVENANCE = "selection-seed-budget"
# Four selection seeds plus one final seed satisfies the injected floor of five
# total paired seeds while keeping the offline fixture run bounded.
_SELECTION_SEEDS = (11, 12, 13, 14)
_FINAL_SEEDS = (21,)
_HELD_OUT_ID = "held_out_schedule"


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
    # periodic alone realizes zero, weak, and strong over six rounds, so a single
    # known schedule is sufficient coverage while keeping the fixture bounded.
    return [{"id": "periodic_known", "schedule": {"kind": "periodic"}}]


def _held_out_mapping(*, selection_seeds=_SELECTION_SEEDS, final_seeds=_FINAL_SEEDS):
    return {
        "study": _study_mapping(seeds=selection_seeds),
        "final_seeds": list(final_seeds),
        "known_schedules": _known_schedules(),
        "held_out_schedule": {
            "id": _HELD_OUT_ID,
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


def _mutated_panel_held_out():
    """A valid held-out spec whose panel tuning was changed after freezing.

    Both the study budget and every tuned panel entry move together so the
    mutation is a legal spec; it still changes ``study_digest`` relative to the
    frozen selection output.
    """

    mapping = _held_out_mapping()
    mapping["study"]["candidate_budget"] = 4
    for entry in mapping["study"]["panel"]:
        if entry["tuning"]["mode"] == "tuned":
            entry["tuning"]["candidate_budget"] = 4
    return parse_held_out_study_text(yaml.safe_dump(mapping, sort_keys=True))


def _mutated_schedule_held_out():
    """A valid held-out spec whose known schedule set changed after freezing."""

    mapping = _held_out_mapping()
    mapping["known_schedules"] = [
        {"id": "periodic_known", "schedule": {"kind": "periodic"}},
        {"id": "duty_known", "schedule": {"kind": "delayed_onset"}},
    ]
    return parse_held_out_study_text(yaml.safe_dump(mapping, sort_keys=True))


def _dataset():
    return MNISTShapedDataset(64, seed=1)


def _evaluation_dataset():
    return MNISTShapedDataset(32, seed=2)


def _run_final(spec, selection_output, output, **overrides):
    return run_final_phase(
        spec,
        selection_output,
        output,
        dataset=_dataset(),
        evaluation_dataset=_evaluation_dataset(),
        **overrides,
    )


# --------------------------------------------------------------------------- #
# Module fixtures: one frozen selection and one published final phase
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def held_out_spec():
    return _held_out()


@pytest.fixture(scope="module")
def frozen_selection(tmp_path_factory, held_out_spec):
    output = tmp_path_factory.mktemp("final-selection-root") / "selection"
    run_selection_phase(
        selection_projection(held_out_spec),
        output,
        dataset=_dataset(),
        evaluation_dataset=_evaluation_dataset(),
    )
    return output


@pytest.fixture(scope="module")
def final_run(tmp_path_factory, held_out_spec, frozen_selection):
    output = tmp_path_factory.mktemp("final-output-root") / "final"
    result = _run_final(held_out_spec, frozen_selection, output)
    return SimpleNamespace(
        spec=held_out_spec,
        selection_output=frozen_selection,
        output=output,
        result=result,
    )


# --------------------------------------------------------------------------- #
# Publication identity and digests
# --------------------------------------------------------------------------- #
def test_final_manifest_identity(final_run) -> None:
    manifest = final_run.result.manifest
    assert manifest["schema"] == FINAL_MANIFEST_SCHEMA
    assert manifest["schema_version"] == FINAL_MANIFEST_SCHEMA_VERSION
    assert manifest["phase"] == "final"
    assert manifest["status"] == "complete"
    assert manifest["study_id"] == "held_out_fixture"
    assert manifest["data_provenance"] == "injected"
    assert manifest["panel"] == list(REQUIRED_POLICY_IDS)
    assert manifest["selection_seeds"] == sorted(_SELECTION_SEEDS)
    assert manifest["final_seeds"] == sorted(_FINAL_SEEDS)
    assert manifest["held_out_schedule_id"] == _HELD_OUT_ID

    assert final_run.result.manifest_path == (
        final_run.output / FINAL_MANIFEST_FILENAME
    )
    assert final_run.result.manifest_path.is_file()
    assert final_run.result.manifest_path.parent == final_run.output
    assert final_run.result.selection_output_path.resolve() == (
        final_run.selection_output.resolve()
    )


def test_final_digests_bind_spec_and_selection(final_run) -> None:
    manifest = final_run.result.manifest
    spec = final_run.spec
    selection_manifest = json.loads(
        (final_run.selection_output / SELECTION_MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert manifest["study_digest"] == study_digest(spec.study)
    assert manifest["selection_spec_digest"] == selection_spec_digest(
        selection_projection(spec)
    )
    assert manifest["held_out_study_digest"] == held_out_study_digest(spec)
    # The full digest covers final seeds and the held-out schedule, so it must
    # differ from the selection-side study digest.
    assert manifest["held_out_study_digest"] != manifest["study_digest"]
    assert manifest["selection_output_digest"] == (
        selection_manifest["selection_output_digest"]
    )
    assert manifest["shared_input_digest"] == compute_shared_input_digest(
        spec.study.base
    )


def test_held_out_schedule_appears_only_in_final(final_run) -> None:
    spec = final_run.spec
    held = spec.held_out_schedule
    selection_manifest = json.loads(
        (final_run.selection_output / SELECTION_MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    # The reserved unseen schedule never appears in the frozen selection output.
    assert held.id not in selection_manifest["schedule_ids"]
    assert all(
        entry["schedule_id"] != held.id
        for entry in selection_manifest["entries"]
    )

    manifest = final_run.result.manifest
    assert held.id in manifest["schedule_ids"]
    assert manifest["held_out_schedule"] == {
        "id": held.id,
        "schedule": held.schedule.model_dump(mode="json"),
        "schedule_digest": schedule_digest(held.schedule),
    }

    held_cells = [
        cell for cell in final_run.result.cells if cell.schedule_id == held.id
    ]
    assert held_cells
    for cell in held_cells:
        entry = cell.manifest_entry
        assert entry["held_out"] is True
        assert entry["schedule"] == held.schedule.model_dump(mode="json")
        assert entry["schedule_digest"] == schedule_digest(held.schedule)
        with TraceReader(cell.trace_path) as reader:
            assert reader.schema_version == 1
            assert reader.config.seed == cell.seed
            assert reader.config.aggregation.kind == cell.policy_id

    assert all(
        cell.manifest_entry["held_out"] is False
        for cell in final_run.result.cells
        if cell.schedule_id != held.id
    )


def test_final_entries_complete_and_scheduled(final_run) -> None:
    spec = final_run.spec
    schedules = {
        entry.id: entry
        for entry in (*spec.known_schedules, spec.held_out_schedule)
    }
    expected_cells = (
        len(spec.final_seeds) * len(schedules) * len(REQUIRED_POLICY_IDS)
    )
    assert len(final_run.result.manifest["entries"]) == expected_cells
    assert len(final_run.result.cells) == expected_cells

    for entry in final_run.result.manifest["entries"]:
        assert entry["status"] == "complete"
        assert entry["seed"] in spec.final_seeds
        assert entry["policy_kind"] == entry["policy_id"]
        assert entry["trace_path"].startswith("traces/")
        schedule = schedules[entry["schedule_id"]]
        assert entry["schedule"] == schedule.schedule.model_dump(mode="json")
        assert entry["schedule_digest"] == schedule_digest(schedule.schedule)
        assert entry["pairing_key"] == (
            f"{entry['policy_id']}:seed-{entry['seed']}:{entry['schedule_id']}"
        )


def test_final_traces_are_readable_schema_v1(final_run) -> None:
    cells = final_run.result.cells
    assert all(cell.trace_path.is_dir() for cell in cells)
    relative = [cell.relative_trace_path for cell in cells]
    assert len(relative) == len(set(relative))
    assert all(".." not in part for part in relative)
    for cell in cells:
        with TraceReader(cell.trace_path) as reader:
            assert reader.schema_version == 1
            assert reader.config.seed == cell.seed
            assert reader.config.aggregation.kind == cell.policy_id


def test_final_records_realized_kappa_sequences(final_run) -> None:
    spec = final_run.spec
    rounds = spec.study.base.rounds
    schedules = {
        entry.id: entry
        for entry in (*spec.known_schedules, spec.held_out_schedule)
    }
    for cell in final_run.result.cells:
        schedule = schedules[cell.schedule_id]
        expected = realized_kappa_sequence(
            schedule.schedule,
            rounds,
            derive_schedule_seed(cell.seed, schedule.schedule),
        )
        assert cell.realized_kappa_sequence == expected
        assert tuple(cell.manifest_entry["realized_kappa_sequence"]) == expected


def test_final_pairs_identical_attack_across_policies(final_run) -> None:
    groups: dict[tuple[int, str], set[tuple[float, ...]]] = {}
    counts: dict[tuple[int, str], int] = {}
    for cell in final_run.result.cells:
        key = (cell.seed, cell.schedule_id)
        groups.setdefault(key, set()).add(cell.realized_kappa_sequence)
        counts[key] = counts.get(key, 0) + 1
    assert all(len(sequences) == 1 for sequences in groups.values())
    assert set(counts.values()) == {len(REQUIRED_POLICY_IDS)}


def test_final_manifest_digest_recomputes(final_run) -> None:
    manifest = final_run.result.manifest
    assert manifest["final_output_digest"] == final_output_digest(manifest)
    on_disk = json.loads(
        final_run.result.manifest_path.read_text(encoding="utf-8")
    )
    assert on_disk == manifest
    assert on_disk["final_output_digest"] == final_output_digest(on_disk)


def test_final_output_is_byte_stable(
    tmp_path_factory, held_out_spec, frozen_selection
) -> None:
    first = tmp_path_factory.mktemp("final-byte-first") / "final"
    second = tmp_path_factory.mktemp("final-byte-second") / "final"
    first_result = _run_final(held_out_spec, frozen_selection, first)
    second_result = _run_final(held_out_spec, frozen_selection, second)
    assert first_result.manifest_path.read_bytes() == (
        second_result.manifest_path.read_bytes()
    )
    assert first_result.manifest == second_result.manifest


def test_final_progress_hook_reports_every_cell(
    tmp_path, frozen_selection, held_out_spec
) -> None:
    output = tmp_path / "progress"
    seen: list[tuple[int, str, str]] = []
    _run_final(
        held_out_spec,
        frozen_selection,
        output,
        progress=lambda seed, schedule_id, policy_id: seen.append(
            (seed, schedule_id, policy_id)
        ),
    )
    schedule_ids = {
        entry.id
        for entry in (*held_out_spec.known_schedules, held_out_spec.held_out_schedule)
    }
    expected = (
        len(held_out_spec.final_seeds) * len(schedule_ids) * len(REQUIRED_POLICY_IDS)
    )
    assert len(seen) == expected
    assert {policy_id for _, _, policy_id in seen} == set(REQUIRED_POLICY_IDS)
    assert {schedule_id for _, schedule_id, _ in seen} == schedule_ids


# --------------------------------------------------------------------------- #
# Fail-closed guards: never publish a partial or unauthorised final phase
# --------------------------------------------------------------------------- #
def test_final_rejects_non_held_out_spec(
    tmp_path, frozen_selection, held_out_spec
) -> None:
    selection = selection_projection(held_out_spec)
    with pytest.raises(EvaluationError):
        run_final_phase(
            selection,  # type: ignore[arg-type]
            frozen_selection,
            tmp_path / "final",
            dataset=_dataset(),
            evaluation_dataset=_evaluation_dataset(),
        )
    assert not (tmp_path / "final").exists()


def test_final_missing_selection_output_fails_closed(
    tmp_path, held_out_spec
) -> None:
    output = tmp_path / "final"
    with pytest.raises(EvaluationError) as info:
        _run_final(held_out_spec, tmp_path / "absent", output)
    assert "not found" in str(info.value)
    assert not output.exists()
    assert [n for n in os.listdir(tmp_path) if "staging" in n] == []


def test_final_incomplete_selection_output_fails_closed(
    tmp_path, held_out_spec
) -> None:
    output = tmp_path / "final"

    # Missing manifest file entirely.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(EvaluationError):
        _run_final(held_out_spec, empty, output)
    assert not output.exists()

    # Manifest present but not frozen.
    unfrozen = tmp_path / "unfrozen"
    unfrozen.mkdir()
    (unfrozen / SELECTION_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema": SELECTION_MANIFEST_SCHEMA,
                "schema_version": 1,
                "status": "draft",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError) as info:
        _run_final(held_out_spec, unfrozen, output)
    assert "frozen" in str(info.value)
    assert not output.exists()
    assert not (output / FINAL_MANIFEST_FILENAME).exists()


def test_final_rejects_mutated_panel_spec(tmp_path, frozen_selection) -> None:
    mutated = _mutated_panel_held_out()
    output = tmp_path / "final"
    with pytest.raises(EvaluationError) as info:
        _run_final(mutated, frozen_selection, output)
    message = str(info.value)
    assert "mutated" in message
    assert selection_spec_digest(selection_projection(mutated)) in message
    assert selection_spec_digest(selection_projection(_held_out())) in message
    # The diagnostic names only bounded digests, never the full spec.
    assert len(message) < 400
    assert not output.exists()
    assert [n for n in os.listdir(tmp_path) if "staging" in n] == []


def test_final_rejects_mutated_known_schedule(tmp_path, frozen_selection) -> None:
    mutated = _mutated_schedule_held_out()
    output = tmp_path / "final"
    with pytest.raises(EvaluationError) as info:
        _run_final(mutated, frozen_selection, output)
    message = str(info.value)
    assert "mutated" in message
    assert selection_spec_digest(selection_projection(mutated)) in message
    assert not output.exists()


def test_final_overlapping_split_rejected_at_parse() -> None:
    mapping = _held_out_mapping(
        selection_seeds=(11,), final_seeds=(11, 21, 22, 23)
    )
    with pytest.raises(HeldOutStudyConfigError) as info:
        parse_held_out_study_text(yaml.safe_dump(mapping, sort_keys=True))
    assert "overlap" in str(info.value) or "disjoint" in str(info.value)


def test_final_reasserts_disjoint_split(
    tmp_path, frozen_selection, held_out_spec
) -> None:
    # model_copy skips validation, standing in for a mutated/injected spec that
    # slips past the type boundary; the runner must re-assert the split.
    overlapping = held_out_spec.model_copy(update={"final_seeds": (11,)})
    output = tmp_path / "final"
    with pytest.raises(EvaluationError) as info:
        _run_final(overlapping, frozen_selection, output)
    assert "overlap" in str(info.value)
    assert not output.exists()
    assert [n for n in os.listdir(tmp_path) if "staging" in n] == []


def test_final_mid_cell_failure_publishes_nothing(
    tmp_path, monkeypatch, frozen_selection, held_out_spec
) -> None:
    output = tmp_path / "final"
    real = evaluation_module.run_from_config

    def flaky(config, **kwargs):
        if config.aggregation.kind == "balance":
            raise RuntimeError("injected mid-final failure")
        return real(config, **kwargs)

    monkeypatch.setattr(evaluation_module, "run_from_config", flaky)
    with pytest.raises(EvaluationError) as info:
        _run_final(held_out_spec, frozen_selection, output)
    message = str(info.value)
    assert "balance" in message
    assert str(_FINAL_SEEDS[0]) in message
    assert not output.exists()
    assert not (output / FINAL_MANIFEST_FILENAME).exists()
    assert [n for n in os.listdir(tmp_path) if "staging" in n] == []


def test_final_invalid_trace_fails_closed(
    tmp_path, frozen_selection, held_out_spec
) -> None:
    output = tmp_path / "final"

    def bad_validator(_path):
        raise TrajectoryError("injected invalid trace")

    with pytest.raises(EvaluationError):
        _run_final(
            held_out_spec,
            frozen_selection,
            output,
            trace_validator=bad_validator,
        )
    assert not output.exists()
    assert not (output / FINAL_MANIFEST_FILENAME).exists()
    assert [n for n in os.listdir(tmp_path) if "staging" in n] == []


def test_final_requires_injected_dataset(tmp_path, frozen_selection, held_out_spec) -> None:
    with pytest.raises(EvaluationError):
        run_final_phase(held_out_spec, frozen_selection, tmp_path / "a")
    with pytest.raises(EvaluationError):
        run_final_phase(
            held_out_spec,
            frozen_selection,
            tmp_path / "b",
            dataset=_dataset(),
        )
    assert not (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()


def test_final_existing_output_rejected(
    tmp_path, frozen_selection, held_out_spec
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(EvaluationError):
        _run_final(held_out_spec, frozen_selection, output)
    assert [n for n in os.listdir(tmp_path) if "staging" in n] == []
