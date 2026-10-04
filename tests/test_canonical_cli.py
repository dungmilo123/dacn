"""Canonical robustness CLI integration tests (M003/S05/T06).

These tests drive the *real* unified command end to end: the argparse family,
the strict envelope dispatch, the real T04 matrix orchestrator, the independent
matrix audit, the T05 canonical report compiler and the static figure
publisher. The heavy positive path builds the reduced **injected** matrix once
through ``dfl evaluate`` with explicitly injected fixture seams; every negative
case either fails before any work or copies the published tree and mutates
exactly one source. No test reads downloaded MNIST, repository ``runs/``,
``data/`` or ``.gsd``.
"""

from __future__ import annotations

import io
import json
import shutil
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import dfl_sim.__main__ as cli
from dfl_sim.__main__ import (
    CANONICAL_EVALUATE_SUCCESS_MARKER,
    CANONICAL_PROGRESS_MARKER,
    REPORT_REGENERATE_MARKER,
    REPORT_VERIFY_MARKER,
    main,
)
from dfl_sim.report import (
    CANONICAL_REPORT_FILENAME,
    CanonicalReportDeps,
    _check_report_destination,
    validate_canonical_report,
)
from dfl_sim.report_figures import FIGURE_DIRNAME
from dfl_sim.robustness_study import (
    ROBUSTNESS_RAW_MARKER,
    InProcessStudyRunner,
    RobustnessDeps,
)
from dfl_sim.robustness_verify import RobustnessVerifyDeps
from test_robustness_study import (
    BAYESIAN_FIXTURE,
    REPO_ROOT,
    _dataset,
    _evaluation_dataset,
    fixture_study,
    injected_protocol,
)

_REPORT_DEPS = CanonicalReportDeps(
    verify_deps=RobustnessVerifyDeps(study_builder=fixture_study, repo_root=REPO_ROOT),
    allow_injected=True,
    verified_roots=(str(BAYESIAN_FIXTURE.parent),),
    repo_root=REPO_ROOT,
)


def _matrix_deps() -> RobustnessDeps:
    return RobustnessDeps(
        study_runner=InProcessStudyRunner(
            dataset=_dataset(), evaluation_dataset=_evaluation_dataset()
        ),
        study_builder=fixture_study,
        repo_root=REPO_ROOT,
    )


def _write_protocol(tmp_path: Path) -> Path:
    path = tmp_path / "robustness-injected.yaml"
    path.write_text(
        yaml.safe_dump(injected_protocol().model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    return path


def _run_capture(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        status = main(argv)
    return status, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# One real CLI publication, reused by the report-audit cases
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def canonical_cli(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("canonical-cli")
    config = _write_protocol(tmp)
    output = tmp / "matrix"

    original_matrix, original_report = (
        cli._canonical_matrix_deps,
        cli._canonical_report_deps,
    )
    cli._canonical_matrix_deps = _matrix_deps
    cli._canonical_report_deps = lambda: _REPORT_DEPS
    try:
        status, stdout, stderr = _run_capture(
            ["evaluate", str(config), "--output", str(output)]
        )
    finally:
        cli._canonical_matrix_deps = original_matrix
        cli._canonical_report_deps = original_report

    assert status == 0, f"canonical evaluate failed: {stdout}\n{stderr}"
    report = json.loads((output / "report" / CANONICAL_REPORT_FILENAME).read_text())
    return SimpleNamespace(
        tmp=tmp,
        config=config,
        output=output,
        status=status,
        stdout=stdout,
        stderr=stderr,
        report=report,
    )


@pytest.fixture()
def report_deps(monkeypatch):
    monkeypatch.setattr(cli, "_canonical_report_deps", lambda: _REPORT_DEPS)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_cli_publishes_report_and_all_figures(canonical_cli) -> None:
    assert CANONICAL_EVALUATE_SUCCESS_MARKER in canonical_cli.stdout
    report_path = canonical_cli.output / "report" / CANONICAL_REPORT_FILENAME
    assert str(report_path) in canonical_cli.stdout
    assert "figures=5" in canonical_cli.stdout
    assert "conditions=1" in canonical_cli.stdout
    assert "final_cells=18" in canonical_cli.stdout
    assert "provenance=injected" in canonical_cli.stdout

    validate_canonical_report(canonical_cli.report)
    report = canonical_cli.report
    assert report["schema_version"] == 1
    assert len(report["figures"]["entries"]) == 5
    for entry in report["figures"]["entries"]:
        figure = canonical_cli.output / "report" / FIGURE_DIRNAME / entry["filename"]
        assert figure.is_file()
        assert "<svg" in figure.read_text()


def test_cli_never_leaks_the_raw_raw_matrix_marker(canonical_cli) -> None:
    assert ROBUSTNESS_RAW_MARKER not in canonical_cli.stdout
    assert ROBUSTNESS_RAW_MARKER not in canonical_cli.stderr


def test_cli_emits_bounded_per_condition_progress(canonical_cli) -> None:
    lines = [
        line
        for line in canonical_cli.stderr.splitlines()
        if line.startswith(CANONICAL_PROGRESS_MARKER)
    ]
    assert len(lines) == 1
    assert "condition=a0p5__mid" in lines[0]
    assert "selection_cells=48" in lines[0]
    assert "final_cells=18" in lines[0]
    assert "artifact_dir" not in canonical_cli.stderr


# --------------------------------------------------------------------------- #
# dfl report --verify-only / --output
# --------------------------------------------------------------------------- #
def test_report_verify_only_succeeds(canonical_cli, report_deps) -> None:
    status, stdout, stderr = _run_capture(
        ["report", str(canonical_cli.output), "--verify-only"]
    )
    assert status == 0, stderr
    assert REPORT_VERIFY_MARKER in stdout
    assert "figures=5" in stdout
    assert str(canonical_cli.output / "report" / CANONICAL_REPORT_FILENAME) in stdout


def test_report_verify_only_detects_tampered_report(
    canonical_cli, report_deps, tmp_path
) -> None:
    root = _copy_matrix(canonical_cli.output, tmp_path)
    path = root / "report" / CANONICAL_REPORT_FILENAME
    payload = json.loads(path.read_text())
    payload["scope"]["matrix_final_cells"] += 1
    path.write_text(json.dumps(payload))
    status, stdout, stderr = _run_capture(["report", str(root), "--verify-only"])
    assert status == 1
    assert REPORT_VERIFY_MARKER not in stdout
    assert "dfl_sim: error:" in stderr


def test_report_verify_only_detects_tampered_figure(
    canonical_cli, report_deps, tmp_path
) -> None:
    root = _copy_matrix(canonical_cli.output, tmp_path)
    entry = json.loads(
        (root / "report" / CANONICAL_REPORT_FILENAME).read_text()
    )["figures"]["entries"][0]
    figure = root / "report" / FIGURE_DIRNAME / entry["filename"]
    figure.write_bytes(figure.read_bytes() + b"<!-- tampered -->")
    status, _stdout, stderr = _run_capture(["report", str(root), "--verify-only"])
    assert status == 1
    assert "dfl_sim: error:" in stderr


def test_report_regenerates_into_new_directory(canonical_cli, report_deps, tmp_path) -> None:
    destination = tmp_path / "regenerated"
    status, stdout, stderr = _run_capture(
        ["report", str(canonical_cli.output), "--output", str(destination)]
    )
    assert status == 0, stderr
    assert REPORT_REGENERATE_MARKER in stdout
    regenerated = json.loads(
        (destination / CANONICAL_REPORT_FILENAME).read_text()
    )
    assert regenerated["report_digest"] == canonical_cli.report["report_digest"]
    for entry in regenerated["figures"]["entries"]:
        assert (destination / FIGURE_DIRNAME / entry["filename"]).is_file()


def test_report_regeneration_is_byte_deterministic(canonical_cli, report_deps, tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert _run_capture(["report", str(canonical_cli.output), "--output", str(first)])[0] == 0
    assert _run_capture(["report", str(canonical_cli.output), "--output", str(second)])[0] == 0
    assert (first / CANONICAL_REPORT_FILENAME).read_bytes() == (
        second / CANONICAL_REPORT_FILENAME
    ).read_bytes()
    for entry in canonical_cli.report["figures"]["entries"]:
        a = (first / FIGURE_DIRNAME / entry["filename"]).read_bytes()
        b = (second / FIGURE_DIRNAME / entry["filename"]).read_bytes()
        assert a == b


# --------------------------------------------------------------------------- #
# Invalid flags / unsafe destinations
# --------------------------------------------------------------------------- #
def test_report_rejects_verify_and_output_together(canonical_cli) -> None:
    status, stdout, stderr = _run_capture(
        ["report", str(canonical_cli.output), "--verify-only", "--output", "x"]
    )
    assert status == 2
    assert REPORT_VERIFY_MARKER not in stdout
    assert "mutually exclusive" in stderr


def test_report_requires_a_mode(canonical_cli) -> None:
    status, stdout, stderr = _run_capture(["report", str(canonical_cli.output)])
    assert status == 2
    assert REPORT_VERIFY_MARKER not in stdout
    assert "--verify-only" in stderr


def test_report_output_collision_with_matrix_root_is_rejected(
    canonical_cli, report_deps
) -> None:
    status, stdout, stderr = _run_capture(
        ["report", str(canonical_cli.output), "--output", str(canonical_cli.output)]
    )
    assert status == 1
    assert REPORT_REGENERATE_MARKER not in stdout
    assert "matrix root" in stderr


def test_check_report_destination_rejects_ancestor_and_root(tmp_path) -> None:
    root = tmp_path / "matrix"
    root.mkdir()
    from dfl_sim.report import CanonicalReportError

    with pytest.raises(CanonicalReportError):
        _check_report_destination(root, root)
    with pytest.raises(CanonicalReportError):
        _check_report_destination(tmp_path, root)
    link = tmp_path / "link"
    try:
        link.symlink_to(root)
    except OSError:  # pragma: no cover - platform without symlinks
        pytest.skip("symlinks unavailable")
    with pytest.raises(CanonicalReportError):
        _check_report_destination(link, root)


def test_canonical_evaluate_rejects_selection_output(canonical_cli, tmp_path) -> None:
    status, stdout, stderr = _run_capture(
        [
            "evaluate",
            str(canonical_cli.config),
            "--output",
            str(tmp_path / "never"),
            "--selection-output",
            str(tmp_path / "sel"),
        ]
    )
    assert status == 1
    assert CANONICAL_EVALUATE_SUCCESS_MARKER not in stdout
    assert "--selection-output" in stderr
    assert not (tmp_path / "never").exists()


# --------------------------------------------------------------------------- #
# Public provenance gates cannot be bypassed from the CLI
# --------------------------------------------------------------------------- #
def test_cli_evaluate_refuses_injected_without_seam(tmp_path) -> None:
    config = _write_protocol(tmp_path)
    output = tmp_path / "matrix"
    status, stdout, stderr = _run_capture(
        ["evaluate", str(config), "--output", str(output)]
    )
    assert status == 1
    assert CANONICAL_EVALUATE_SUCCESS_MARKER not in stdout
    assert "injected" in stderr
    assert not output.exists()


def test_cli_report_refuses_injected_without_seam(canonical_cli) -> None:
    status, stdout, stderr = _run_capture(
        ["report", str(canonical_cli.output), "--verify-only"]
    )
    assert status == 1
    assert REPORT_VERIFY_MARKER not in stdout
    assert "real_mnist" in stderr


# --------------------------------------------------------------------------- #
# Envelope dispatch stays backwards compatible
# --------------------------------------------------------------------------- #
def test_baseline_spec_still_classifies_as_study(tmp_path) -> None:
    from dfl_sim.evaluation_config import study_envelope_kind

    baseline = {
        "study_id": "cli_fixture_study",
        "data_provenance": "injected",
        "base": {},
        "seeds": [1],
        "candidate_budget": 1,
        "selection_seed": 1,
        "selection_provenance": "fixture",
        "panel": [],
    }
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(baseline), encoding="utf-8")
    assert study_envelope_kind(path) == "study"
    assert study_envelope_kind(_write_protocol(tmp_path)) == "robustness"


def test_report_help_lists_modes(capsys) -> None:
    import pytest as _pytest

    from dfl_sim.__main__ import build_report_parser

    with _pytest.raises(SystemExit):
        build_report_parser().parse_args(["--help"])
    captured = capsys.readouterr()
    assert "--verify-only" in captured.out
    assert "--output" in captured.out


def _copy_matrix(source: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "matrix"
    shutil.copytree(source, destination)
    return destination
