"""Atomic no-overwrite artifact publication tests (T03, R013/R014, Q3/Q7).

Deterministic in-process fixtures. Each test publishes a hand-built
:class:`~dfl_sim.calibration_estimator.EstimatedArtifact` and asserts the
atomicity, no-overwrite, digest-integrity, and failure-cleanup guarantees of
the publisher. Failure injection uses the module's documented seams so a write,
validation, or rename failure can be forced and inspected.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

import dfl_sim.calibration_artifact_writer as writer_module
from dfl_sim import (
    CalibrationPublishError,
    EstimatedArtifact,
    publish_calibration_artifact,
)
from dfl_sim.bayesian_artifact import (
    ARTIFACT_FILENAMES,
    MANIFEST_FILENAME,
    load_bayesian_artifact,
)
from test_calibration_estimator import _estimate

_STAGING_PREFIX = ".dfl-calibration-stage-"


def _artifact() -> EstimatedArtifact:
    return _estimate()


def _staging_leftovers(parent: Path) -> list[Path]:
    return sorted(parent.glob(f"{_STAGING_PREFIX}*"))


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_publish_writes_exactly_three_files_and_loads(tmp_path: Path) -> None:
    artifact = _artifact()
    published = publish_calibration_artifact(tmp_path / "artifact", artifact)
    directory = tmp_path / "artifact"
    assert published.path == directory
    assert sorted(p.name for p in directory.iterdir()) == sorted(ARTIFACT_FILENAMES)
    loaded = load_bayesian_artifact(directory)
    assert loaded.manifest.aggregate_digest == published.aggregate_digest
    assert loaded.manifest.support.prior_cells == 5
    assert _staging_leftovers(tmp_path) == []


def test_two_publications_are_byte_identical(tmp_path: Path) -> None:
    artifact = _artifact()
    first = publish_calibration_artifact(tmp_path / "out1", artifact)
    second = publish_calibration_artifact(tmp_path / "out2", artifact)
    for name in ARTIFACT_FILENAMES:
        assert (first.path / name).read_bytes() == (second.path / name).read_bytes()
    assert first.aggregate_digest == second.aggregate_digest


# --------------------------------------------------------------------------- #
# No-overwrite / path safety (Q3)
# --------------------------------------------------------------------------- #
def test_existing_output_is_never_replaced(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    destination.mkdir()
    marker = destination / "user-data.txt"
    marker.write_text("keep me", encoding="utf-8")
    with pytest.raises(CalibrationPublishError, match="already exists"):
        publish_calibration_artifact(destination, _artifact())
    assert marker.read_text(encoding="utf-8") == "keep me"
    assert sorted(p.name for p in destination.iterdir()) == ["user-data.txt"]
    assert _staging_leftovers(tmp_path) == []


def test_symlink_parent_is_rejected(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(CalibrationPublishError, match="symlink"):
        publish_calibration_artifact(link / "artifact", _artifact())
    assert not (real_parent / "artifact").exists()


def test_config_alias_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "config.json"
    with pytest.raises(CalibrationPublishError, match="aliases"):
        publish_calibration_artifact(
            destination, _artifact(), config_path=destination
        )
    assert not destination.exists()


def test_data_root_containment_is_rejected(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(CalibrationPublishError, match="inside the data root"):
        publish_calibration_artifact(
            data_root / "artifact", _artifact(), data_root=data_root
        )
    assert not (data_root / "artifact").exists()


def test_missing_parent_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CalibrationPublishError, match="does not exist"):
        publish_calibration_artifact(tmp_path / "absent" / "artifact", _artifact())


def test_non_artifact_argument_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CalibrationPublishError, match="EstimatedArtifact"):
        publish_calibration_artifact(
            tmp_path / "artifact", object()  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# Digest drift (Q3)
# --------------------------------------------------------------------------- #
def test_payload_digest_drift_is_rejected(tmp_path: Path) -> None:
    artifact = _artifact()
    tampered = dataclasses.replace(
        artifact, likelihood_bytes=artifact.likelihood_bytes + b" "
    )
    with pytest.raises(CalibrationPublishError, match="does not match payload"):
        publish_calibration_artifact(tmp_path / "artifact", tampered)
    assert not (tmp_path / "artifact").exists()
    assert _staging_leftovers(tmp_path) == []


def test_tampered_published_bytes_fail_the_s01_loader(tmp_path: Path) -> None:
    publish_calibration_artifact(tmp_path / "artifact", _artifact())
    likelihood = tmp_path / "artifact" / "likelihood.json"
    payload = bytearray(likelihood.read_bytes())
    payload[-1] = payload[-1] ^ 0x01
    likelihood.write_bytes(bytes(payload))
    with pytest.raises(Exception):
        load_bayesian_artifact(tmp_path / "artifact")


# --------------------------------------------------------------------------- #
# Failure injection: no final or complete-looking partial directory
# --------------------------------------------------------------------------- #
def _assert_clean(tmp_path: Path) -> None:
    assert not (tmp_path / "artifact").exists()
    assert _staging_leftovers(tmp_path) == []


def test_injected_write_failure_is_atomic(tmp_path: Path, monkeypatch) -> None:
    def boom(path: str, data: bytes) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(writer_module, "_write_staged_file", boom)
    with pytest.raises(OSError, match="injected write failure"):
        publish_calibration_artifact(tmp_path / "artifact", _artifact())
    _assert_clean(tmp_path)


def test_injected_validation_failure_is_atomic(tmp_path: Path, monkeypatch) -> None:
    def boom(path: str) -> None:
        raise RuntimeError("injected validation failure")

    monkeypatch.setattr(writer_module, "_validate_staged", boom)
    with pytest.raises(CalibrationPublishError, match="S01 validation"):
        publish_calibration_artifact(tmp_path / "artifact", _artifact())
    _assert_clean(tmp_path)


def test_injected_rename_failure_is_atomic(tmp_path: Path, monkeypatch) -> None:
    def boom(source: str, destination: str) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(writer_module, "_rename", boom)
    with pytest.raises(OSError, match="injected rename failure"):
        publish_calibration_artifact(tmp_path / "artifact", _artifact())
    _assert_clean(tmp_path)


def test_injected_directory_sync_failure_leaves_no_partial(
    tmp_path: Path, monkeypatch
) -> None:
    def boom(path: str) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(writer_module, "_fsync_dir", boom)
    with pytest.raises(OSError, match="injected fsync failure"):
        publish_calibration_artifact(tmp_path / "artifact", _artifact())
    _assert_clean(tmp_path)


def test_staged_files_use_bounded_permissions(tmp_path: Path) -> None:
    publish_calibration_artifact(tmp_path / "artifact", _artifact())
    for name in ARTIFACT_FILENAMES:
        mode = os.stat(tmp_path / "artifact" / name).st_mode & 0o777
        assert mode == 0o600


# --------------------------------------------------------------------------- #
# Full orchestrator entry point (shared with the T04 CLI)
# --------------------------------------------------------------------------- #
def test_calibrate_from_config_is_byte_identical_end_to_end(tmp_path: Path) -> None:
    from dfl_sim import calibrate_from_config
    from helpers import MNISTShapedDataset, make_config
    from test_calibration_config import base_dict, parse_dict

    experiment = make_config(
        rounds=1,
        training={"local_epochs": 1, "batch_size": 8},
        data={"samples_per_node": 8},
        evaluation={"max_samples": 8, "batch_size": 8},
    )
    config = parse_dict(
        base_dict(
            experiment=experiment.model_dump(mode="json"),
            episodes=2,
            p0=0.5,
            design_split_size=8,
        )
    )
    dataset = MNISTShapedDataset(200, separable=True, seed=11)
    first = calibrate_from_config(config, str(tmp_path / "a"), dataset=dataset)
    second = calibrate_from_config(config, str(tmp_path / "b"), dataset=dataset)
    for name in ARTIFACT_FILENAMES:
        assert (first.path / name).read_bytes() == (second.path / name).read_bytes()
    assert first.aggregate_digest == second.aggregate_digest
    loaded = load_bayesian_artifact(first.path)
    assert loaded.manifest.support.calibration_episodes == 2
    assert 0.0 <= loaded.manifest.coverage.posterior_calibration <= 1.0
