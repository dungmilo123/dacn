"""``dfl evaluate`` held-out end-to-end CLI tests (M003/S02/T05, R018/R022).

These tests drive the *real* ``dfl evaluate`` command over a strict held-out
spec: the two-phase orchestrator (selection -> freeze -> final), the unchanged
:func:`~dfl_sim.run.run_from_config` composition, the TraceReader validation,
and the atomic publication of ``selection/selection-manifest.json`` plus
``final/final-manifest.json``.

The production MNIST loaders are replaced with in-process MNIST-shaped fixtures
(the same seam the S01 evaluate CLI tests use) and the Bayesian cell loads the
git-tracked M002 fixture, so nothing touches the network. The negative cases
(malformed/overlap/mutated/missing-selection/baseline-plus-flag) prove the
command fails closed with a single concise stderr line and never publishes a
partial study.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import dfl_sim.evaluation as evaluation_module
import dfl_sim.run as run_module
from dfl_sim import (
    BALANCE_CITATION,
    FINAL_MANIFEST_FILENAME,
    FINAL_MANIFEST_SCHEMA,
    REQUIRED_POLICY_IDS,
    SELECTION_MANIFEST_FILENAME,
    SELECTION_MANIFEST_SCHEMA,
    TraceReader,
    final_output_digest,
    load_any_study_config,
    run_held_out_study,
    selection_output_digest,
)
from dfl_sim.evaluation import MANIFEST_FILENAME as BASELINE_MANIFEST_FILENAME
from dfl_sim.__main__ import (
    EVALUATE_SUCCESS_MARKER,
    HELD_OUT_FINAL_PROGRESS_MARKER,
    HELD_OUT_SELECTION_PROGRESS_MARKER,
    HELD_OUT_SUCCESS_MARKER,
    build_evaluate_parser,
    main,
)
from helpers import MNISTShapedDataset, make_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BAYESIAN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"

_TUNED_PROVENANCE = "selection-seed-budget"
_SELECTION_SEEDS = (11, 12)
_FINAL_SEEDS = (21,)
_HELD_OUT_ID = "held_out_schedule"
_KNOWN_SCHEDULE_ID = "periodic_known"
_STAGING_PREFIX = ".held_out_study.staging-"


# --------------------------------------------------------------------------- #
# Fixture builders (real_mnist provenance; loaders patched in-process)
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
        "study_id": "held_out_cli_fixture",
        "data_provenance": "real_mnist",
        "base": _base().model_dump(mode="json"),
        "seeds": list(seeds),
        "candidate_budget": 3,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
        "panel": _panel(),
    }


def _known_schedules():
    # periodic alone realizes zero, weak, and strong over six rounds, so one
    # known schedule is enough coverage while keeping the fixture bounded.
    return [{"id": _KNOWN_SCHEDULE_ID, "schedule": {"kind": "periodic"}}]


def _held_out_mapping(
    *, selection_seeds=_SELECTION_SEEDS, final_seeds=_FINAL_SEEDS
):
    return {
        "study": _study_mapping(seeds=selection_seeds),
        "final_seeds": list(final_seeds),
        "known_schedules": _known_schedules(),
        "held_out_schedule": {
            "id": _HELD_OUT_ID,
            "schedule": {"kind": "held_out"},
        },
    }


def _write_spec(tmp_path: Path, mapping, *, name: str = "study.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(mapping, sort_keys=True), encoding="utf-8")
    return path


def _mutated_held_out_mapping():
    """A valid held-out spec whose selection-side budget changed after freezing.

    Both the nested study budget and every tuned panel entry move together so
    the mutation is a legal spec; it still changes ``study_digest`` relative to
    the frozen selection output.
    """

    mapping = _held_out_mapping()
    mapping["study"]["candidate_budget"] = 4
    for entry in mapping["study"]["panel"]:
        if entry["tuning"]["mode"] == "tuned":
            entry["tuning"]["candidate_budget"] = 4
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


def _staging_leftovers(parent: Path) -> list[Path]:
    return sorted(parent.glob(f"{_STAGING_PREFIX}*"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_dirs(root: Path) -> list[Path]:
    traces = root / "traces"
    if not traces.is_dir():
        return []
    return sorted(
        path for path in traces.rglob("seed-*") if path.is_dir()
    )


# --------------------------------------------------------------------------- #
# Happy path: run the real two-phase command once and reuse the published study
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def published(tmp_path_factory):
    spec_dir = tmp_path_factory.mktemp("held-out-cli-spec")
    spec = _write_spec(spec_dir, _held_out_mapping())
    output = tmp_path_factory.mktemp("held-out-cli-out") / "study"

    original_train = run_module.load_mnist_dataset
    original_eval = run_module.load_mnist_evaluation_dataset
    run_module.load_mnist_dataset = (
        lambda config, **kw: MNISTShapedDataset(64, seed=1)
    )
    run_module.load_mnist_evaluation_dataset = (
        lambda config, **kw: MNISTShapedDataset(32, seed=2)
    )
    out_buf, err_buf = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            status = main(["evaluate", str(spec), "--output", str(output)])
    finally:
        run_module.load_mnist_dataset = original_train
        run_module.load_mnist_evaluation_dataset = original_eval

    assert status == 0, (
        f"dfl evaluate held-out failed: status={status}\n"
        f"stdout:\n{out_buf.getvalue()}\nstderr:\n{err_buf.getvalue()}"
    )
    selection_output = output / "selection"
    final_output = output / "final"
    selection_manifest = _read_json(
        selection_output / SELECTION_MANIFEST_FILENAME
    )
    final_manifest = _read_json(final_output / FINAL_MANIFEST_FILENAME)
    return SimpleNamespace(
        status=status,
        spec=spec,
        output=output,
        selection_output=selection_output,
        final_output=final_output,
        stdout=out_buf.getvalue(),
        stderr=err_buf.getvalue(),
        selection_manifest=selection_manifest,
        final_manifest=final_manifest,
    )


def test_cli_held_out_publishes_two_phase_study(published) -> None:
    assert published.status == 0
    selection_manifest = published.selection_manifest
    final_manifest = published.final_manifest

    assert selection_manifest["schema"] == SELECTION_MANIFEST_SCHEMA
    assert selection_manifest["status"] == "frozen"
    assert final_manifest["schema"] == FINAL_MANIFEST_SCHEMA
    assert final_manifest["phase"] == "final"
    assert final_manifest["status"] == "complete"

    assert selection_manifest["panel"] == list(REQUIRED_POLICY_IDS)
    assert final_manifest["panel"] == list(REQUIRED_POLICY_IDS)
    assert selection_manifest["selection_seeds"] == list(_SELECTION_SEEDS)
    assert final_manifest["selection_seeds"] == list(_SELECTION_SEEDS)
    assert final_manifest["final_seeds"] == list(_FINAL_SEEDS)
    assert final_manifest["held_out_schedule_id"] == _HELD_OUT_ID

    # Output layout is a study root with both phase directories.
    assert (published.output / "selection").is_dir()
    assert (published.output / "final").is_dir()
    assert not _staging_leftovers(published.output.parent)


def test_cli_held_out_success_marker_is_single_and_names_both(
    published,
) -> None:
    assert published.stdout.count(HELD_OUT_SUCCESS_MARKER) == 1
    assert EVALUATE_SUCCESS_MARKER not in published.stdout
    assert SELECTION_MANIFEST_FILENAME in published.stdout
    assert FINAL_MANIFEST_FILENAME in published.stdout
    assert str(published.output) in published.stdout
    assert "policies=6" in published.stdout
    assert "selection_seeds=2" in published.stdout
    assert "final_seeds=1" in published.stdout
    assert "selection_cells=12" in published.stdout
    assert "final_cells=12" in published.stdout


def test_cli_held_out_progress_markers_are_bounded_per_cell(
    published,
) -> None:
    selection_lines = [
        line
        for line in published.stderr.splitlines()
        if line.startswith(HELD_OUT_SELECTION_PROGRESS_MARKER)
    ]
    final_lines = [
        line
        for line in published.stderr.splitlines()
        if line.startswith(HELD_OUT_FINAL_PROGRESS_MARKER)
    ]
    assert len(selection_lines) == 12
    assert len(final_lines) == 12
    for line in selection_lines:
        assert f"schedule={_KNOWN_SCHEDULE_ID}" in line
    assert {
        line.split("schedule=", 1)[1] for line in final_lines
    } == {_KNOWN_SCHEDULE_ID, _HELD_OUT_ID}
    for policy_id in REQUIRED_POLICY_IDS:
        assert f"policy={policy_id}" in published.stderr
    # Bounded: no artifact path or YAML contents leak into stderr.
    assert "artifact_dir" not in published.stderr
    assert str(BAYESIAN_FIXTURE) not in published.stderr


def test_cli_held_out_all_traces_are_readable_schema_v1(published) -> None:
    selection_traces = _trace_dirs(published.selection_output)
    final_traces = _trace_dirs(published.final_output)
    assert len(selection_traces) == 12
    assert len(final_traces) == 12
    # The reserved held-out schedule is never built during selection.
    assert all(
        _HELD_OUT_ID not in trace.parts for trace in selection_traces
    )
    assert any(_HELD_OUT_ID in trace.parts for trace in final_traces)

    for trace in (*selection_traces, *final_traces):
        with TraceReader(trace) as reader:
            assert reader.schema_version == 1
            assert reader.config.aggregation.kind in REQUIRED_POLICY_IDS


def test_cli_held_out_manifests_bind_the_freeze(published) -> None:
    selection_manifest = published.selection_manifest
    final_manifest = published.final_manifest
    # The frozen selection digest recomputes and is carried into the final
    # manifest, so the final phase is bound to the exact frozen output.
    assert selection_manifest["selection_output_digest"] == (
        selection_output_digest(selection_manifest)
    )
    assert final_manifest["selection_output_digest"] == (
        selection_manifest["selection_output_digest"]
    )
    assert final_manifest["final_output_digest"] == final_output_digest(
        final_manifest
    )
    # The reserved unseen schedule appears only in the final manifest.
    assert _HELD_OUT_ID not in selection_manifest["schedule_ids"]
    assert _HELD_OUT_ID in final_manifest["schedule_ids"]
    assert all(
        entry["schedule_id"] != _HELD_OUT_ID
        for entry in selection_manifest["entries"]
    )
    assert all(
        entry["held_out"] == (entry["schedule_id"] == _HELD_OUT_ID)
        for entry in final_manifest["entries"]
    )


def test_cli_held_out_manifest_bytes_match_disk(published) -> None:
    selection_on_disk = _read_json(
        published.selection_output / SELECTION_MANIFEST_FILENAME
    )
    final_on_disk = _read_json(
        published.final_output / FINAL_MANIFEST_FILENAME
    )
    assert selection_on_disk == published.selection_manifest
    assert final_on_disk == published.final_manifest


def test_orchestrator_is_byte_stable_and_rekeyed(tmp_path_factory) -> None:
    # Injected datasets let the orchestrator run offline without CLI patching.
    spec = load_any_study_config(_write_spec(
        tmp_path_factory.mktemp("held-out-stable-spec"), _held_out_mapping()
    ))
    assert not hasattr(spec, "study_id")  # held-out envelope, not baseline

    first = run_held_out_study(
        spec,
        tmp_path_factory.mktemp("held-out-stable-a") / "study",
        dataset=MNISTShapedDataset(64, seed=1),
        evaluation_dataset=MNISTShapedDataset(32, seed=2),
    )
    second = run_held_out_study(
        spec,
        tmp_path_factory.mktemp("held-out-stable-b") / "study",
        dataset=MNISTShapedDataset(64, seed=1),
        evaluation_dataset=MNISTShapedDataset(32, seed=2),
    )
    assert first.selection_manifest_path.read_bytes() == (
        second.selection_manifest_path.read_bytes()
    )
    assert first.final_manifest_path.read_bytes() == (
        second.final_manifest_path.read_bytes()
    )
    # Published cell paths are re-keyed out of the removed staging directory.
    for cell in first.selection_result.cells:
        assert cell.trace_path.is_dir()
        assert cell.trace_path.is_relative_to(first.output_path)
    for cell in first.final_result.cells:
        assert cell.trace_path.is_dir()
        assert cell.trace_path.is_relative_to(first.final_path)
    assert first.selection_path.parent == first.output_path
    assert first.final_path.parent == first.output_path
    assert not _staging_leftovers(first.output_path.parent)


# --------------------------------------------------------------------------- #
# Dispatch: baseline spec keeps the S01 surface; --selection-output is held-out only
# --------------------------------------------------------------------------- #
def test_baseline_spec_still_uses_baseline_surface(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_loaders(monkeypatch)
    spec = _write_spec(tmp_path, _study_mapping())
    output = tmp_path / "baseline"

    status = main(["evaluate", str(spec), "--output", str(output)])
    captured = capsys.readouterr()
    assert status == 0, captured.err
    assert EVALUATE_SUCCESS_MARKER in captured.out
    assert HELD_OUT_SUCCESS_MARKER not in captured.out
    assert (output / BASELINE_MANIFEST_FILENAME).is_file()
    assert not (output / "selection").exists()


def test_selection_output_flag_rejected_for_baseline_spec(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_loaders(monkeypatch)
    spec = _write_spec(tmp_path, _study_mapping())
    output = tmp_path / "baseline"
    frozen = tmp_path / "frozen"
    frozen.mkdir()

    status = main(
        [
            "evaluate",
            str(spec),
            "--output",
            str(output),
            "--selection-output",
            str(frozen),
        ]
    )
    captured = capsys.readouterr()
    assert status != 0
    assert "--selection-output" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


def test_cli_held_out_final_only_against_frozen_selection(
    published, tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_loaders(monkeypatch)
    spec = _write_spec(tmp_path, _held_out_mapping())
    output = tmp_path / "final-only"

    status = main(
        [
            "evaluate",
            str(spec),
            "--output",
            str(output),
            "--selection-output",
            str(published.selection_output),
        ]
    )
    captured = capsys.readouterr()
    assert status == 0, captured.err
    assert captured.out.count(HELD_OUT_SUCCESS_MARKER) == 1
    # Final-only publishes final-manifest.json directly into --output.
    assert (output / FINAL_MANIFEST_FILENAME).is_file()
    assert not (output / SELECTION_MANIFEST_FILENAME).exists()
    assert str(published.selection_output) in captured.out
    assert FINAL_MANIFEST_FILENAME in captured.out
    assert "final_cells=12" in captured.out

    final_manifest = _read_json(output / FINAL_MANIFEST_FILENAME)
    assert final_manifest["schema"] == FINAL_MANIFEST_SCHEMA
    assert final_manifest["final_output_digest"] == final_output_digest(
        final_manifest
    )


# --------------------------------------------------------------------------- #
# Parser surface
# --------------------------------------------------------------------------- #
def test_build_evaluate_parser_accepts_selection_output() -> None:
    parser = build_evaluate_parser()
    args = parser.parse_args(
        ["study.yaml", "--output", "out", "--selection-output", "frozen"]
    )
    assert args.study == "study.yaml"
    assert args.output == "out"
    assert args.selection_output == "frozen"

    default = parser.parse_args(["study.yaml", "--output", "out"])
    assert default.selection_output is None


def test_cli_held_out_help_lists_selection_output(capsys) -> None:
    with pytest.raises(SystemExit) as info:
        main(["evaluate", "--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "--selection-output" in out


# --------------------------------------------------------------------------- #
# Negative paths (Q5/Q7): one concise stderr line, nonzero, no publication
# --------------------------------------------------------------------------- #
def _assert_single_line_failure(status: int, err: str, output: Path) -> None:
    assert status != 0
    assert "error" in err
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
    assert not output.exists()
    assert not (output / FINAL_MANIFEST_FILENAME).exists()
    assert not (output / SELECTION_MANIFEST_FILENAME).exists()


def test_cli_held_out_malformed_spec_is_concise(
    tmp_path: Path, capsys
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("study: [unclosed\n", encoding="utf-8")
    output = tmp_path / "study"

    status = main(["evaluate", str(bad), "--output", str(output)])
    err = capsys.readouterr().err
    _assert_single_line_failure(status, err, output)
    assert "invalid YAML" in err
    assert _staging_leftovers(tmp_path) == []


def test_cli_held_out_overlapping_split_is_rejected(
    tmp_path: Path, capsys
) -> None:
    mapping = _held_out_mapping(
        selection_seeds=(11,), final_seeds=(11, 21)
    )
    spec = _write_spec(tmp_path, mapping)
    output = tmp_path / "study"

    status = main(["evaluate", str(spec), "--output", str(output)])
    err = capsys.readouterr().err
    _assert_single_line_failure(status, err, output)
    assert "overlap" in err or "disjoint" in err
    assert _staging_leftovers(tmp_path) == []


def test_cli_held_out_missing_selection_output_fails_closed(
    tmp_path: Path, capsys
) -> None:
    spec = _write_spec(tmp_path, _held_out_mapping())
    output = tmp_path / "final"

    status = main(
        [
            "evaluate",
            str(spec),
            "--output",
            str(output),
            "--selection-output",
            str(tmp_path / "absent"),
        ]
    )
    err = capsys.readouterr().err
    _assert_single_line_failure(status, err, output)
    assert "not found" in err
    assert _staging_leftovers(tmp_path) == []


def test_cli_held_out_mutated_spec_fails_closed(
    published, tmp_path: Path, capsys
) -> None:
    spec = _write_spec(tmp_path, _mutated_held_out_mapping())
    output = tmp_path / "final"

    status = main(
        [
            "evaluate",
            str(spec),
            "--output",
            str(output),
            "--selection-output",
            str(published.selection_output),
        ]
    )
    err = capsys.readouterr().err
    _assert_single_line_failure(status, err, output)
    assert "mutated" in err
    assert _staging_leftovers(tmp_path) == []


def test_cli_held_out_mid_run_failure_publishes_nothing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_loaders(monkeypatch)
    spec = _write_spec(tmp_path, _held_out_mapping())
    output = tmp_path / "study"
    real = evaluation_module.run_from_config

    def flaky(config, **kwargs):
        if config.aggregation.kind == "balance":
            raise RuntimeError("injected mid-study failure")
        return real(config, **kwargs)

    monkeypatch.setattr(evaluation_module, "run_from_config", flaky)

    status = main(["evaluate", str(spec), "--output", str(output)])
    captured = capsys.readouterr()
    assert status != 0
    assert "balance" in captured.err
    assert "Traceback" not in captured.err
    assert HELD_OUT_SUCCESS_MARKER not in captured.out
    assert not output.exists()
    assert not (output / "selection").exists()
    assert not (output / "final").exists()
    assert _staging_leftovers(tmp_path) == []
