"""Bounded manifest-matched resume for the held-out study (M003/S02/T07).

These tests drive the **production** resume composition offline:

    a prior per-cell ``cells.jsonl`` checkpoint -> manifest-matched reuse of
    completed (seed, schedule, policy) traces -> revalidation with TraceReader
    -> recompute only missing/mismatched cells -> atomic publication

The held-out runner writes a canonical JSON Lines checkpoint into each phase
staging directory as cells complete; because a successful phase renames staging
into place, a published output (or a retained private staging directory) is
itself a valid checkpoint. Resume never trusts a checkpoint blindly: the frozen
selection-side digest, the current study/full digest, and every per-cell
schedule/config/artifact digest must match, the split must stay disjoint, and
every reused trace is revalidated before it is copied into the new staging.

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
import dfl_sim.run as run_module
from dfl_sim import (
    BALANCE_CITATION,
    CHECKPOINT_FILENAME,
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_VERSION,
    FINAL_MANIFEST_FILENAME,
    REQUIRED_POLICY_IDS,
    EvaluationError,
    TraceReader,
    compute_shared_input_digest,
    final_output_digest,
    held_out_study_digest,
    parse_held_out_study_text,
    run_final_phase,
    run_held_out_study,
    selection_projection,
    selection_spec_digest,
    study_digest,
)
from dfl_sim.__main__ import HELD_OUT_SUCCESS_MARKER, main
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
# Fixture builders (mirror the S02/T04 held-out shape)
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
        "study_id": "held_out_resume_fixture",
        "data_provenance": "injected",
        "base": _base().model_dump(mode="json"),
        "seeds": list(seeds),
        "candidate_budget": 3,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
        "panel": _panel(),
    }


def _known_schedules():
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


def _dataset():
    return MNISTShapedDataset(64, seed=1)


def _evaluation_dataset():
    return MNISTShapedDataset(32, seed=2)


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def _read_checkpoint_lines(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_checkpoint_lines(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _clone_checkpoint(source: Path, destination: Path) -> Path:
    """Copy a phase output (checkpoint + traces) into a private clone."""

    import shutil

    shutil.copytree(source, destination)
    return destination


def _trace_bytes(root: Path) -> dict[str, bytes]:
    """Map every file under ``root/traces`` to its relative path and bytes."""

    traces = root / "traces"
    return {
        str(path.relative_to(traces)): path.read_bytes()
        for path in sorted(traces.rglob("*"))
        if path.is_file()
    }


def _strip_reuse(manifest: dict) -> dict:
    """Return a manifest copy with the resume provenance scrubbed."""

    payload = json.loads(json.dumps(manifest))
    payload.pop("final_output_digest", None)
    payload.pop("selection_output_digest", None)
    for entry in payload.get("entries", []):
        entry.pop("reused", None)
    return payload


def _reused_flags(manifest: dict) -> dict[str, bool]:
    return {
        str(entry["pairing_key"]): bool(entry.get("reused"))
        for entry in manifest["entries"]
    }


# --------------------------------------------------------------------------- #
# Module fixtures: one clean held-out study reused by every resume test
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def held_out_spec():
    return _held_out()


@pytest.fixture(scope="module")
def clean_study(tmp_path_factory, held_out_spec):
    output = tmp_path_factory.mktemp("resume-clean-root") / "study"
    result = run_held_out_study(
        held_out_spec,
        output,
        dataset=_dataset(),
        evaluation_dataset=_evaluation_dataset(),
    )
    return SimpleNamespace(spec=held_out_spec, output=output, result=result)


@pytest.fixture(scope="module")
def frozen_selection(clean_study):
    return clean_study.output / "selection"


@pytest.fixture(scope="module")
def clean_final(clean_study):
    return clean_study.output / "final"


# --------------------------------------------------------------------------- #
# Checkpoint shape
# --------------------------------------------------------------------------- #
def test_checkpoint_is_canonical_jsonl_per_cell(clean_study) -> None:
    selection_checkpoint = (
        clean_study.output / "selection" / CHECKPOINT_FILENAME
    )
    final_checkpoint = clean_study.output / "final" / CHECKPOINT_FILENAME
    assert selection_checkpoint.is_file()
    assert final_checkpoint.is_file()

    selection_records = _read_checkpoint_lines(selection_checkpoint)
    assert len(selection_records) == (
        len(_SELECTION_SEEDS) * len(_known_schedules()) * len(REQUIRED_POLICY_IDS)
    )
    final_records = _read_checkpoint_lines(final_checkpoint)
    assert len(final_records) == (
        len(_FINAL_SEEDS)
        * (len(_known_schedules()) + 1)
        * len(REQUIRED_POLICY_IDS)
    )

    for record in selection_records:
        assert record["schema"] == CHECKPOINT_SCHEMA
        assert record["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert record["phase"] == "selection"
        assert record["status"] == "complete"
        assert record["trace_path"].startswith("traces/")
        assert record["schedule_digest"]
        assert record["config_digest"]
        assert record["reused"] is False
        assert "held_out_study_digest" not in record

    for record in final_records:
        assert record["phase"] == "final"
        assert record["held_out_study_digest"]
        assert record["selection_output_digest"]
        assert record["reused"] is False


def test_checkpoint_globals_bind_the_study(clean_study) -> None:
    spec = clean_study.spec
    selection_records = _read_checkpoint_lines(
        clean_study.output / "selection" / CHECKPOINT_FILENAME
    )
    final_records = _read_checkpoint_lines(
        clean_study.output / "final" / CHECKPOINT_FILENAME
    )
    assert {r["study_digest"] for r in selection_records} == {
        study_digest(spec.study)
    }
    assert {r["selection_spec_digest"] for r in selection_records} == {
        selection_spec_digest(selection_projection(spec))
    }
    assert {r["shared_input_digest"] for r in selection_records} == {
        compute_shared_input_digest(spec.study.base)
    }
    assert {r["held_out_study_digest"] for r in final_records} == {
        held_out_study_digest(spec)
    }


def test_checkpoint_has_no_ablation_identity_for_six_policy(
    clean_study,
) -> None:
    for phase in ("selection", "final"):
        records = _read_checkpoint_lines(
            clean_study.output / phase / CHECKPOINT_FILENAME
        )
        assert records
        for record in records:
            assert "ablation_match_digest" not in record
            assert "ablation_protocol_version" not in record


# --------------------------------------------------------------------------- #
# Partial resume: reuse matched cells, recompute only the missing ones
# --------------------------------------------------------------------------- #
def test_partial_final_resume_recomputes_only_missing(
    tmp_path, monkeypatch, held_out_spec, frozen_selection, clean_final
) -> None:
    # Emulate an interruption after the first six final cells by retaining a
    # checkpoint whose cells.jsonl names only those completed cells.
    checkpoint = _clone_checkpoint(clean_final, tmp_path / "retained-final")
    checkpoint_file = checkpoint / CHECKPOINT_FILENAME
    records = _read_checkpoint_lines(checkpoint_file)
    assert len(records) == 12
    retained = records[:6]
    _write_checkpoint_lines(checkpoint_file, retained)
    retained_keys = {record["pairing_key"] for record in retained}

    calls: list[str] = []
    real = evaluation_module.run_from_config

    def counting(config, **kwargs):
        calls.append(config.aggregation.kind)
        return real(config, **kwargs)

    monkeypatch.setattr(evaluation_module, "run_from_config", counting)

    output = tmp_path / "resumed-final"
    result = run_final_phase(
        held_out_spec,
        frozen_selection,
        output,
        dataset=_dataset(),
        evaluation_dataset=_evaluation_dataset(),
        resume_from=checkpoint,
    )

    # Only the six missing cells were recomputed; the retained six were reused.
    assert len(calls) == 6
    flags = _reused_flags(result.manifest)
    assert sum(1 for value in flags.values() if value) == 6
    assert sum(1 for value in flags.values() if not value) == 6
    for entry in result.manifest["entries"]:
        assert flags[entry["pairing_key"]] is (
            entry["pairing_key"] in retained_keys
        )

    # Every reused trace revalidates under the production strict reader.
    reused_cells = [
        cell for cell in result.cells if cell.manifest_entry.get("reused")
    ]
    assert len(reused_cells) == 6
    for cell in reused_cells:
        with TraceReader(cell.trace_path) as reader:
            assert reader.schema_version == 1
            assert reader.config.seed == cell.seed
            assert reader.config.aggregation.kind == cell.policy_id


def test_full_final_resume_runs_no_cells_and_matches_clean(
    tmp_path, monkeypatch, held_out_spec, frozen_selection, clean_final
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("run_from_config must not be reached on full reuse")

    monkeypatch.setattr(evaluation_module, "run_from_config", forbidden)

    output = tmp_path / "full-resumed-final"
    result = run_final_phase(
        held_out_spec,
        frozen_selection,
        output,
        dataset=_dataset(),
        evaluation_dataset=_evaluation_dataset(),
        resume_from=clean_final,
    )

    assert all(
        entry.get("reused") is True for entry in result.manifest["entries"]
    )
    # Traces are byte-for-byte identical to the clean run.
    assert _trace_bytes(output) == _trace_bytes(clean_final)
    # The manifest differs only by the reused provenance flag and its digest.
    clean_manifest = json.loads(
        (clean_final / FINAL_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert _strip_reuse(result.manifest) == _strip_reuse(clean_manifest)
    assert result.manifest["final_output_digest"] == final_output_digest(
        result.manifest
    )


# --------------------------------------------------------------------------- #
# Mismatch handling: absent/corrupt reuse candidates recompute, digests fail closed
# --------------------------------------------------------------------------- #
def test_missing_reused_trace_is_recomputed(
    tmp_path, monkeypatch, held_out_spec, frozen_selection, clean_final
) -> None:
    checkpoint = _clone_checkpoint(clean_final, tmp_path / "missing-trace")
    records = _read_checkpoint_lines(checkpoint / CHECKPOINT_FILENAME)
    victim = records[0]
    import shutil

    shutil.rmtree(checkpoint / victim["trace_path"])

    calls: list[str] = []
    real = evaluation_module.run_from_config

    def counting(config, **kwargs):
        calls.append(config.aggregation.kind)
        return real(config, **kwargs)

    monkeypatch.setattr(evaluation_module, "run_from_config", counting)

    output = tmp_path / "missing-trace-output"
    result = run_final_phase(
        held_out_spec,
        frozen_selection,
        output,
        dataset=_dataset(),
        evaluation_dataset=_evaluation_dataset(),
        resume_from=checkpoint,
    )
    assert len(calls) == 1
    flags = _reused_flags(result.manifest)
    assert flags[victim["pairing_key"]] is False
    assert sum(1 for value in flags.values() if value) == 11


def test_corrupt_reused_trace_is_recomputed(
    tmp_path, monkeypatch, held_out_spec, frozen_selection, clean_final
) -> None:
    checkpoint = _clone_checkpoint(clean_final, tmp_path / "corrupt-trace")
    records = _read_checkpoint_lines(checkpoint / CHECKPOINT_FILENAME)
    victim = records[0]
    # Corrupt the trace's events file so TraceReader refuses it; the runner
    # must not mix the unreadable evidence in.
    (checkpoint / victim["trace_path"] / "events.jsonl").write_text(
        "not valid json\n", encoding="utf-8"
    )

    calls: list[str] = []
    real = evaluation_module.run_from_config

    def counting(config, **kwargs):
        calls.append(config.aggregation.kind)
        return real(config, **kwargs)

    monkeypatch.setattr(evaluation_module, "run_from_config", counting)

    output = tmp_path / "corrupt-trace-output"
    result = run_final_phase(
        held_out_spec,
        frozen_selection,
        output,
        dataset=_dataset(),
        evaluation_dataset=_evaluation_dataset(),
        resume_from=checkpoint,
    )
    assert len(calls) == 1
    assert _reused_flags(result.manifest)[victim["pairing_key"]] is False
    # The recomputed cell still publishes a valid trace.
    for cell in result.cells:
        with TraceReader(cell.trace_path) as reader:
            assert reader.schema_version == 1


def test_tampered_selection_side_digest_fails_closed(
    tmp_path, held_out_spec, frozen_selection, clean_final
) -> None:
    checkpoint = _clone_checkpoint(clean_final, tmp_path / "tampered-selection")
    checkpoint_file = checkpoint / CHECKPOINT_FILENAME
    records = _read_checkpoint_lines(checkpoint_file)
    records[0]["selection_spec_digest"] = "0" * 64
    _write_checkpoint_lines(checkpoint_file, records)

    output = tmp_path / "tampered-selection-output"
    with pytest.raises(EvaluationError) as info:
        run_final_phase(
            held_out_spec,
            frozen_selection,
            output,
            dataset=_dataset(),
            evaluation_dataset=_evaluation_dataset(),
            resume_from=checkpoint,
        )
    assert "selection_spec_digest" in str(info.value)
    assert not output.exists()
    assert not (output / FINAL_MANIFEST_FILENAME).exists()
    assert [n for n in os.listdir(tmp_path) if "staging" in n] == []


def test_tampered_schedule_digest_fails_closed(
    tmp_path, held_out_spec, frozen_selection, clean_final
) -> None:
    checkpoint = _clone_checkpoint(clean_final, tmp_path / "tampered-schedule")
    checkpoint_file = checkpoint / CHECKPOINT_FILENAME
    records = _read_checkpoint_lines(checkpoint_file)
    records[0]["schedule_digest"] = "0" * 64
    _write_checkpoint_lines(checkpoint_file, records)

    output = tmp_path / "tampered-schedule-output"
    with pytest.raises(EvaluationError) as info:
        run_final_phase(
            held_out_spec,
            frozen_selection,
            output,
            dataset=_dataset(),
            evaluation_dataset=_evaluation_dataset(),
            resume_from=checkpoint,
        )
    assert "schedule_digest" in str(info.value)
    assert not output.exists()
    assert [n for n in os.listdir(tmp_path) if "staging" in n] == []


def test_tampered_artifact_digest_fails_closed(
    tmp_path, held_out_spec, frozen_selection, clean_final
) -> None:
    checkpoint = _clone_checkpoint(clean_final, tmp_path / "tampered-artifact")
    checkpoint_file = checkpoint / CHECKPOINT_FILENAME
    records = _read_checkpoint_lines(checkpoint_file)
    # The bayesian cell carries a real artifact digest; tamper that one.
    victim = next(r for r in records if r["artifact_digest"] is not None)
    victim["artifact_digest"] = "0" * 64
    _write_checkpoint_lines(checkpoint_file, records)

    output = tmp_path / "tampered-artifact-output"
    with pytest.raises(EvaluationError) as info:
        run_final_phase(
            held_out_spec,
            frozen_selection,
            output,
            dataset=_dataset(),
            evaluation_dataset=_evaluation_dataset(),
            resume_from=checkpoint,
        )
    assert "artifact_digest" in str(info.value)
    assert not output.exists()
    assert [n for n in os.listdir(tmp_path) if "staging" in n] == []


def test_missing_checkpoint_fails_closed(tmp_path, held_out_spec) -> None:
    empty = tmp_path / "empty-checkpoint"
    empty.mkdir()
    output = tmp_path / "no-checkpoint-output"
    with pytest.raises(EvaluationError) as info:
        run_held_out_study(
            held_out_spec,
            output,
            dataset=_dataset(),
            evaluation_dataset=_evaluation_dataset(),
            resume_from=empty,
        )
    assert "resume checkpoint not found" in str(info.value)
    assert not output.exists()


# --------------------------------------------------------------------------- #
# Whole-study resume through the two-phase orchestrator
# --------------------------------------------------------------------------- #
def test_study_resume_reuses_selection_and_final(
    tmp_path, monkeypatch, held_out_spec, clean_study
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("run_from_config must not be reached on full reuse")

    monkeypatch.setattr(evaluation_module, "run_from_config", forbidden)

    output = tmp_path / "resumed-study"
    result = run_held_out_study(
        held_out_spec,
        output,
        dataset=_dataset(),
        evaluation_dataset=_evaluation_dataset(),
        resume_from=clean_study.output,
    )

    assert result.final_result.manifest["status"] == "complete"
    assert all(
        entry.get("reused") is True
        for entry in result.final_result.manifest["entries"]
    )
    assert all(
        entry.get("reused") is True
        for entry in result.selection_result.manifest["entries"]
    )
    # Both phases reproduce byte-identical traces and summary evidence.
    assert _trace_bytes(output / "final") == _trace_bytes(
        clean_study.output / "final"
    )
    assert _trace_bytes(output / "selection") == _trace_bytes(
        clean_study.output / "selection"
    )
    resumed_final = json.loads(
        result.final_manifest_path.read_text(encoding="utf-8")
    )
    clean_final_manifest = json.loads(
        clean_study.result.final_manifest_path.read_text(encoding="utf-8")
    )
    assert _strip_reuse(resumed_final) == _strip_reuse(clean_final_manifest)

    # The published resumed study is itself a valid checkpoint for the future.
    assert (output / "selection" / CHECKPOINT_FILENAME).is_file()
    assert (output / "final" / CHECKPOINT_FILENAME).is_file()


def test_partial_study_resume_only_recomputes_missing_final_cells(
    tmp_path, monkeypatch, held_out_spec, clean_study
) -> None:
    # Retain the complete selection checkpoint but only half the final cells.
    checkpoint = tmp_path / "partial-study-checkpoint"
    checkpoint.mkdir()
    import shutil

    shutil.copytree(
        clean_study.output / "selection", checkpoint / "selection"
    )
    shutil.copytree(clean_study.output / "final", checkpoint / "final")
    final_file = checkpoint / "final" / CHECKPOINT_FILENAME
    records = _read_checkpoint_lines(final_file)
    retained = records[:6]
    _write_checkpoint_lines(final_file, retained)

    calls: list[str] = []
    real = evaluation_module.run_from_config

    def counting(config, **kwargs):
        calls.append(config.aggregation.kind)
        return real(config, **kwargs)

    monkeypatch.setattr(evaluation_module, "run_from_config", counting)

    output = tmp_path / "partial-study-resumed"
    result = run_held_out_study(
        held_out_spec,
        output,
        dataset=_dataset(),
        evaluation_dataset=_evaluation_dataset(),
        resume_from=checkpoint,
    )
    # Selection was fully reused (no runs); only the six missing final cells ran.
    assert len(calls) == 6
    assert all(
        entry.get("reused") is True
        for entry in result.selection_result.manifest["entries"]
    )
    final_flags = _reused_flags(result.final_result.manifest)
    assert sum(1 for value in final_flags.values() if value) == 6


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #
def _real_mnist_mapping(*, selection_seeds=(11, 12), final_seeds=(21,)):
    mapping = _held_out_mapping(
        selection_seeds=selection_seeds, final_seeds=final_seeds
    )
    mapping["study"]["data_provenance"] = "real_mnist"
    return mapping


def _patch_loaders(monkeypatch) -> None:
    monkeypatch.setattr(
        run_module,
        "load_mnist_dataset",
        lambda config, **kwargs: MNISTShapedDataset(64, seed=1),
    )
    monkeypatch.setattr(
        run_module,
        "load_mnist_evaluation_dataset",
        lambda config, **kwargs: MNISTShapedDataset(32, seed=2),
    )


def test_cli_resume_from_publishes_and_reports_success(
    tmp_path, monkeypatch, capsys
) -> None:
    _patch_loaders(monkeypatch)
    spec_path = tmp_path / "study.yaml"
    spec_path.write_text(
        yaml.safe_dump(_real_mnist_mapping(), sort_keys=True),
        encoding="utf-8",
    )
    first = tmp_path / "first-study"
    status = main(["evaluate", str(spec_path), "--output", str(first)])
    assert status == 0
    capsys.readouterr()

    resumed = tmp_path / "resumed-study"
    status = main(
        [
            "evaluate",
            str(spec_path),
            "--output",
            str(resumed),
            "--resume-from",
            str(first),
        ]
    )
    out = capsys.readouterr().out
    assert status == 0
    assert HELD_OUT_SUCCESS_MARKER in out
    final_manifest = json.loads(
        (resumed / "final" / FINAL_MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert final_manifest["status"] == "complete"
    assert all(entry.get("reused") is True for entry in final_manifest["entries"])
