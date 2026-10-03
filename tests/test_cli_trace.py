"""Offline integration tests for the S04 CLI and trace publication wiring (T03).

These tests drive the *real* ``dfl run`` command and the real
:func:`dfl_sim.run.run_from_config` orchestrator with the production data
loaders patched to in-process MNIST-shaped fixtures, so nothing touches the
network. They prove that YAML -> runner -> privileged evaluation -> atomic
schema-v1 trajectory publication executes end to end, that the backward
compatible legacy log form still works, and that every requested failure path
returns nonzero without publishing a partial trace or replacing a prior log.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import dfl_sim.run as run_module
from dfl_sim import (
    DataError,
    EpisodeError,
    TraceReader,
    TrajectoryError,
    load_config,
    run_from_config,
)
from dfl_sim.__main__ import main
from helpers import MNISTShapedDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_CONFIG = REPO_ROOT / "configs" / "clean_3node.yaml"


def _train() -> MNISTShapedDataset:
    # clean_3node.yaml needs 3 nodes x 64 samples; separable so loss is finite.
    return MNISTShapedDataset(300, seed=7, separable=True)


def _evaluation(num_examples: int = 64) -> MNISTShapedDataset:
    return MNISTShapedDataset(num_examples, seed=11)


def _patch_loaders(monkeypatch, *, evaluation: MNISTShapedDataset | None = None) -> None:
    """Patch production dataset loaders with in-process offline fixtures."""

    monkeypatch.setattr(
        run_module, "load_mnist_dataset", lambda config, **kwargs: _train()
    )
    monkeypatch.setattr(
        run_module,
        "load_mnist_evaluation_dataset",
        lambda config, **kwargs: evaluation if evaluation is not None else _evaluation(),
    )


# --------------------------------------------------------------------------- #
# Happy path: CLI -> real runner -> complete trace
# --------------------------------------------------------------------------- #
def test_cli_run_publishes_complete_trace(tmp_path: Path, monkeypatch, capsys) -> None:
    _patch_loaders(monkeypatch, evaluation=_evaluation())
    output = tmp_path / "trace"

    status = main(["run", str(CLEAN_CONFIG), "--output", str(output)])

    assert status == 0
    stdout = capsys.readouterr().out
    assert "wrote trace" in stdout
    assert str(output) in stdout

    with TraceReader(output) as reader:
        assert reader.schema_version == 1
        assert reader.nodes == (0, 1, 2)
        assert reader.edges == ((0, 1), (1, 2))
        assert reader.has_initial_snapshot is True
        assert len(reader.events) == 2
        assert reader.snapshot_rounds == (0, 1)
        assert reader.manifest["completed"] is True
        assert reader.manifest["recorded_rounds"] == 2
        assert reader.manifest["event_count"] == 2
        assert reader.config.rounds == 2
        # Exactly one initial + two committed snapshots for every node.
        assert len(reader.read_initial_snapshot()) == 3
        assert len(reader.read_round_snapshot(1)) == 3
        assert reader.tensor_entry_count > 0


def test_cli_run_trace_reports_round_and_node_counts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_loaders(monkeypatch, evaluation=_evaluation())
    output = tmp_path / "trace"

    assert main(["run", str(CLEAN_CONFIG), "--output", str(output)]) == 0
    stdout = capsys.readouterr().out
    assert "2 round(s)" in stdout
    assert "3 node(s)" in stdout


# --------------------------------------------------------------------------- #
# Direct runner seams for trace + evaluation injection
# --------------------------------------------------------------------------- #
def test_run_from_config_returns_trace_path_and_uses_loader_seam(
    tmp_path: Path,
) -> None:
    seen: dict[str, bool] = {}

    def loader(config):
        seen["called"] = True
        return _evaluation()

    result = run_from_config(
        load_config(CLEAN_CONFIG),
        dataset=_train(),
        evaluation_dataset_loader=loader,
        trace_path=tmp_path / "trace",
    )

    assert seen.get("called") is True
    assert result.trace_path == tmp_path / "trace"
    assert result.log_path is None
    assert (tmp_path / "trace" / "manifest.json").is_file()


def test_run_from_config_publishes_log_and_trace_together(tmp_path: Path) -> None:
    log = tmp_path / "episode.jsonl"
    result = run_from_config(
        load_config(CLEAN_CONFIG),
        dataset=_train(),
        evaluation_dataset=_evaluation(),
        log_path=log,
        trace_path=tmp_path / "trace",
    )

    assert result.log_path == log
    assert result.trace_path == tmp_path / "trace"
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2
    assert (tmp_path / "trace" / "manifest.json").is_file()


def test_evaluation_dataset_and_loader_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(EpisodeError, match="not both"):
        run_from_config(
            load_config(CLEAN_CONFIG),
            dataset=_train(),
            evaluation_dataset=_evaluation(),
            evaluation_dataset_loader=lambda config: _evaluation(),
            trace_path=tmp_path / "trace",
        )


# --------------------------------------------------------------------------- #
# Destination safety (Q3/Q5/Q7)
# --------------------------------------------------------------------------- #
def test_cli_run_refuses_existing_output(tmp_path: Path, monkeypatch, capsys) -> None:
    _patch_loaders(monkeypatch)
    output = tmp_path / "trace"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    status = main(["run", str(CLEAN_CONFIG), "--output", str(output)])

    assert status == 1
    assert "already exists" in capsys.readouterr().err
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_cli_run_rejects_symlink_output(tmp_path: Path, monkeypatch, capsys) -> None:
    _patch_loaders(monkeypatch)
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)

    status = main(["run", str(CLEAN_CONFIG), "--output", str(link)])

    assert status == 1
    assert "symlink" in capsys.readouterr().err


def test_trace_and_log_may_not_alias(tmp_path: Path) -> None:
    shared = tmp_path / "shared.out"
    with pytest.raises(TrajectoryError, match="alias"):
        run_from_config(
            load_config(CLEAN_CONFIG),
            dataset=_train(),
            evaluation_dataset=_evaluation(),
            log_path=shared,
            trace_path=shared,
        )
    assert not shared.exists()


def test_trace_missing_parent_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TrajectoryError, match="does not exist"):
        run_from_config(
            load_config(CLEAN_CONFIG),
            dataset=_train(),
            evaluation_dataset=_evaluation(),
            trace_path=tmp_path / "missing" / "trace",
        )


def test_cli_run_requires_output_argument() -> None:
    with pytest.raises(SystemExit) as info:
        main(["run", str(CLEAN_CONFIG)])
    assert info.value.code == 2


# --------------------------------------------------------------------------- #
# Failure paths: nonzero and no published target (Q5/Q7)
# --------------------------------------------------------------------------- #
def test_cli_run_missing_config_returns_nonzero(tmp_path: Path, monkeypatch, capsys) -> None:
    _patch_loaders(monkeypatch)
    output = tmp_path / "trace"

    status = main(["run", str(tmp_path / "nope.yaml"), "--output", str(output)])

    assert status == 1
    assert "not found" in capsys.readouterr().err
    assert not output.exists()


def test_cli_run_malformed_yaml_returns_nonzero(tmp_path: Path, capsys) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("rounds: [unclosed\n", encoding="utf-8")
    output = tmp_path / "trace"

    status = main(["run", str(config), "--output", str(output)])

    assert status == 1
    assert "invalid YAML" in capsys.readouterr().err
    assert not output.exists()


def test_cli_run_unavailable_evaluation_split_returns_nonzero(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        run_module, "load_mnist_dataset", lambda config, **kwargs: _train()
    )

    def unavailable(config, **kwargs):
        raise DataError("could not load MNIST (evaluation split)")

    monkeypatch.setattr(run_module, "load_mnist_evaluation_dataset", unavailable)
    output = tmp_path / "trace"

    status = main(["run", str(CLEAN_CONFIG), "--output", str(output)])

    assert status == 1
    assert "evaluation split" in capsys.readouterr().err
    assert not output.exists()


def test_cli_run_mid_episode_failure_leaves_no_trace(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_loaders(monkeypatch)
    real_run_round = run_module.run_round
    calls = {"count": 0}

    def flaky(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] > 1:
            raise RuntimeError("injected round failure")
        return real_run_round(*args, **kwargs)

    monkeypatch.setattr(run_module, "run_round", flaky)
    output = tmp_path / "trace"

    status = main(["run", str(CLEAN_CONFIG), "--output", str(output)])

    assert status == 1
    assert not output.exists()
    # Writer-owned staging is cleaned; nothing else lingers.
    assert list(tmp_path.iterdir()) == []


def test_cli_run_trace_write_failure_leaves_no_trace(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_loaders(monkeypatch)
    original = run_module.TraceWriter.write_round
    calls = {"count": 0}

    def failing(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] > 1:
            raise TrajectoryError("injected write failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(run_module.TraceWriter, "write_round", failing)
    output = tmp_path / "trace"

    status = main(["run", str(CLEAN_CONFIG), "--output", str(output)])

    assert status == 1
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_cli_run_evaluator_failure_leaves_no_trace(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_loaders(
        monkeypatch,
        evaluation=MNISTShapedDataset(64, seed=2, fill=float("nan")),
    )
    output = tmp_path / "trace"

    status = main(["run", str(CLEAN_CONFIG), "--output", str(output)])

    assert status == 1
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_trace_failure_preserves_prior_complete_log(tmp_path: Path, monkeypatch) -> None:
    def failing(self, *args, **kwargs):
        raise TrajectoryError("injected write failure")

    monkeypatch.setattr(run_module.TraceWriter, "write_round", failing)
    log = tmp_path / "episode.jsonl"
    log.write_bytes(b"PRIOR-COMPLETE\n")

    with pytest.raises(EpisodeError, match="trace emission failed"):
        run_from_config(
            load_config(CLEAN_CONFIG),
            dataset=_train(),
            evaluation_dataset=_evaluation(),
            log_path=log,
            trace_path=tmp_path / "trace",
        )

    assert log.read_bytes() == b"PRIOR-COMPLETE\n"
    assert not (tmp_path / "trace").exists()


# --------------------------------------------------------------------------- #
# Backward compatibility and help surfaces
# --------------------------------------------------------------------------- #
def test_legacy_invocation_still_writes_minimal_log(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        run_module, "load_mnist_dataset", lambda config, **kwargs: _train()
    )
    log = tmp_path / "episode.jsonl"

    status = main([str(CLEAN_CONFIG), "--log", str(log)])

    assert status == 0
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2
    assert "wrote 2 round(s)" in capsys.readouterr().out


def test_module_help_exposes_both_forms() -> None:
    legacy = subprocess.run(
        [sys.executable, "-m", "dfl_sim", "--help"],
        capture_output=True,
        text=True,
    )
    assert legacy.returncode == 0, legacy.stderr
    assert "--log" in legacy.stdout
    assert "run CONFIG --output" in legacy.stdout

    run_help = subprocess.run(
        [sys.executable, "-m", "dfl_sim", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert run_help.returncode == 0, run_help.stderr
    assert "--output" in run_help.stdout
    assert "TRACE_DIR" in run_help.stdout


def test_console_script_help_when_installed() -> None:
    executable = shutil.which("dfl")
    if executable is None:  # pragma: no cover - depends on the local venv
        pytest.skip("console script 'dfl' is not installed on PATH")
    proc = subprocess.run([executable, "--help"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "run CONFIG --output" in proc.stdout
