"""Contract tests for T01: integrity-checked MNIST provisioning (S04/R017).

All fixtures are synthetic, in-process IDX payloads written under ``tmp_path``;
nothing here downloads data, reads the real ``data/`` cache, ``.gsd`` or any
other ignored project path, and no fixture is ever labelled as actual
real-MNIST execution. The real download is performed operationally in T04.

Covered contract: canonical manifest hashing/reuse, both-split validation,
disabled-download failure, corrupt resource/content, bad labels/shapes/counts,
interrupted provisioning, foreign/tampered manifest, concurrent lease
collision, lock cleanup and mutation detection across a lease.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

import dfl_sim.mnist_provision as mnist_provision
from dfl_sim.mnist_provision import (
    CANONICAL_COUNTS,
    LOCK_FILENAME,
    MANIFEST_FILENAME,
    PREPROCESSING_ID,
    PROVISION_SCHEMA_VERSION,
    CacheLockError,
    ManifestError,
    ProvisioningError,
    ResourceSpec,
    build_manifest,
    cache_lease,
    load_manifest,
    main,
    manifest_path,
    provision,
    raw_dir,
    verify_cache,
    _Decoder,
    _group_specs,
)

IMG_TRAIN = "train-images-idx3-ubyte.gz"
LBL_TRAIN = "train-labels-idx1-ubyte.gz"
IMG_TEST = "t10k-images-idx3-ubyte.gz"
LBL_TEST = "t10k-labels-idx1-ubyte.gz"
DUMMY_MD5 = "0" * 32
TRAIN_IMAGES = "train-images-idx3-ubyte"
TRAIN_LABELS = "train-labels-idx1-ubyte"
TEST_IMAGES = "t10k-images-idx3-ubyte"
TEST_LABELS = "t10k-labels-idx1-ubyte"


# --------------------------------------------------------------------------- #
# Synthetic IDX fixtures (never real MNIST)
# --------------------------------------------------------------------------- #
def _images(count: int, seed: int, *, height: int = 28, width: int = 28) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(count, height, width), dtype=np.uint8)


def _labels(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 10, size=count, dtype=np.uint8)


def _idx_images(images: np.ndarray) -> bytes:
    count, height, width = images.shape
    header = struct.pack(">IIII", 2051, count, height, width)
    return header + np.ascontiguousarray(images, dtype=np.uint8).tobytes()


def _idx_labels(labels: np.ndarray) -> bytes:
    labels = np.ascontiguousarray(labels, dtype=np.uint8)
    header = struct.pack(">II", 2049, labels.shape[0])
    return header + labels.tobytes()


def _specs() -> tuple[ResourceSpec, ...]:
    return (
        ResourceSpec(IMG_TRAIN, DUMMY_MD5, "train", "image"),
        ResourceSpec(LBL_TRAIN, DUMMY_MD5, "train", "label"),
        ResourceSpec(IMG_TEST, DUMMY_MD5, "test", "image"),
        ResourceSpec(LBL_TEST, DUMMY_MD5, "test", "label"),
    )


def _write_raw(root: Path, split: str, images: np.ndarray, labels: np.ndarray) -> None:
    directory = Path(raw_dir(root))
    directory.mkdir(parents=True, exist_ok=True)
    if split == "train":
        image_name, label_name = TRAIN_IMAGES, TRAIN_LABELS
    else:
        image_name, label_name = TEST_IMAGES, TEST_LABELS
    (directory / image_name).write_bytes(_idx_images(images))
    (directory / label_name).write_bytes(_idx_labels(labels))


def _make_cache(
    root: Path,
    *,
    train: int = 6,
    test: int = 4,
    train_images: np.ndarray | None = None,
    train_labels: np.ndarray | None = None,
    test_images: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
) -> tuple[ResourceSpec, ...]:
    _write_raw(
        root,
        "train",
        _images(train, 1) if train_images is None else train_images,
        _labels(train, 1) if train_labels is None else train_labels,
    )
    _write_raw(
        root,
        "test",
        _images(test, 2) if test_images is None else test_images,
        _labels(test, 2) if test_labels is None else test_labels,
    )
    return _specs()


def _provision(root: Path, *, train: int = 6, test: int = 4, specs=None, **kwargs):
    return provision(
        root,
        resources=specs if specs is not None else _specs(),
        expected_counts={"train": train, "test": test},
        require_archive=False,
        **kwargs,
    )


def _build(root: Path, *, train: int = 6, test: int = 4, specs=None, **kwargs):
    return build_manifest(
        root,
        resources=specs if specs is not None else _specs(),
        expected_counts={"train": train, "test": test},
        require_archive=False,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Happy path: publish, reuse, verify
# --------------------------------------------------------------------------- #
def test_provision_publishes_verified_manifest(tmp_path: Path) -> None:
    specs = _make_cache(tmp_path)
    manifest = _provision(tmp_path, specs=specs)

    assert manifest.schema_version == PROVISION_SCHEMA_VERSION
    assert manifest.dataset == "mnist"
    assert manifest.preprocessing_id == PREPROCESSING_ID
    assert len(manifest.preprocessing_digest) == 64
    assert manifest.train.split == "train" and manifest.test.split == "test"
    assert manifest.train.count == 6 and manifest.test.count == 4
    assert manifest.train.image_shape == (28, 28)
    assert manifest.train.image_dtype == "uint8"
    assert manifest.train.label_dtype == "int64"
    assert 0 <= manifest.train.label_min <= manifest.train.label_max <= 9
    assert Path(manifest_path(tmp_path)).is_file()
    assert load_manifest(tmp_path) == manifest

    # Reuse revalidates and must not rewrite the published artifact.
    published = Path(manifest_path(tmp_path)).read_bytes()
    reused = _provision(tmp_path, specs=specs)
    assert reused == manifest
    assert Path(manifest_path(tmp_path)).read_bytes() == published


def test_verify_cache_deep_and_fast(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    manifest = _provision(tmp_path)
    assert verify_cache(tmp_path, deep=True) == manifest
    assert verify_cache(tmp_path, deep=False) == manifest
    with cache_lease(tmp_path, exclusive=True) as leased:
        assert leased == manifest


def test_manifest_is_path_independent(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _make_cache(left)
    _make_cache(right)
    first = _provision(left)
    second = _provision(right)
    assert first.manifest_digest == second.manifest_digest
    assert first.train.decoded_image_sha256 == second.train.decoded_image_sha256


def test_both_splits_have_distinct_content_identities(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    manifest = _provision(tmp_path)
    assert manifest.train.decoded_image_sha256 != manifest.test.decoded_image_sha256
    assert manifest.train.raw_image_sha256 != manifest.test.raw_image_sha256
    assert manifest.train.raw_image_bytes > 0 and manifest.test.raw_image_bytes > 0


# --------------------------------------------------------------------------- #
# Disabled download and missing data fail closed
# --------------------------------------------------------------------------- #
def test_disabled_download_missing_data_fails_closed(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ProvisioningError, match="raw MNIST directory not found"):
        provision(
            empty,
            download=False,
            resources=_specs(),
            expected_counts={"train": 6, "test": 4},
            require_archive=False,
        )
    assert not (empty / MANIFEST_FILENAME).exists()


def test_missing_raw_file_fails_before_publish(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    (Path(raw_dir(tmp_path)) / TRAIN_LABELS).unlink()
    with pytest.raises(ProvisioningError, match="missing train label file"):
        _provision(tmp_path)
    assert not (tmp_path / MANIFEST_FILENAME).exists()


# --------------------------------------------------------------------------- #
# Corrupt resources and content
# --------------------------------------------------------------------------- #
def test_archive_checksum_verified_when_present(tmp_path: Path) -> None:
    specs = _make_cache(tmp_path)
    directory = Path(raw_dir(tmp_path))
    verified: list[ResourceSpec] = []
    for spec in specs:
        raw_file = directory / os.path.splitext(spec.filename)[0]
        archive = directory / spec.filename
        archive.write_bytes(gzip.compress(raw_file.read_bytes()))
        verified.append(
            ResourceSpec(
                spec.filename,
                hashlib.md5(archive.read_bytes()).hexdigest(),
                spec.split,
                spec.kind,
            )
        )
    manifest = provision(
        tmp_path,
        resources=tuple(verified),
        expected_counts={"train": 6, "test": 4},
        require_archive=True,
    )
    assert manifest.train.count == 6


def test_archive_checksum_mismatch_fails(tmp_path: Path) -> None:
    specs = _make_cache(tmp_path)
    directory = Path(raw_dir(tmp_path))
    (directory / IMG_TRAIN).write_bytes(gzip.compress(b"not the real resource"))
    with pytest.raises(ProvisioningError, match="checksum mismatch"):
        build_manifest(
            tmp_path,
            resources=specs,
            expected_counts={"train": 6, "test": 4},
            require_archive=True,
        )


def test_required_archive_missing_fails(tmp_path: Path) -> None:
    specs = _make_cache(tmp_path)
    with pytest.raises(ProvisioningError, match="missing upstream archive"):
        build_manifest(
            tmp_path,
            resources=specs,
            expected_counts={"train": 6, "test": 4},
            require_archive=True,
        )


def test_corrupt_compressed_content_fails(tmp_path: Path) -> None:
    specs = _make_cache(tmp_path)
    directory = Path(raw_dir(tmp_path))
    # Authentic archives for every resource ...
    verified: list[ResourceSpec] = []
    for spec in specs:
        raw_file = directory / os.path.splitext(spec.filename)[0]
        archive = directory / spec.filename
        archive.write_bytes(gzip.compress(raw_file.read_bytes()))
        verified.append(
            ResourceSpec(
                spec.filename,
                hashlib.md5(archive.read_bytes()).hexdigest(),
                spec.split,
                spec.kind,
            )
        )
    # ... but the decompressed image bytes are not a valid IDX payload.
    (directory / TRAIN_IMAGES).write_bytes(b"not an idx file")
    with pytest.raises(ProvisioningError, match="could not decode train"):
        build_manifest(
            tmp_path,
            resources=tuple(verified),
            expected_counts={"train": 6, "test": 4},
            require_archive=True,
        )


def test_truncated_image_file_fails(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    image_file = Path(raw_dir(tmp_path)) / TRAIN_IMAGES
    image_file.write_bytes(image_file.read_bytes()[:-64])
    with pytest.raises(ProvisioningError):
        _build(tmp_path)


def test_bad_label_value_fails(tmp_path: Path) -> None:
    labels = np.full(6, 5, dtype=np.uint8)
    labels[0] = 10
    _write_raw(tmp_path, "train", _images(6, 3), labels)
    _write_raw(tmp_path, "test", _images(4, 4), _labels(4, 4))
    with pytest.raises(ProvisioningError, match="train labels out of range"):
        _build(tmp_path)


def test_bad_test_split_fails(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    bad = np.full(4, 10, dtype=np.uint8)
    _write_raw(tmp_path, "test", _images(4, 2), bad)
    with pytest.raises(ProvisioningError, match="test labels out of range"):
        _build(tmp_path)


def test_wrong_image_shape_fails(tmp_path: Path) -> None:
    _write_raw(tmp_path, "train", _images(6, 5, height=14, width=14), _labels(6, 5))
    _write_raw(tmp_path, "test", _images(4, 6), _labels(4, 6))
    with pytest.raises(ProvisioningError, match="is not"):
        _build(tmp_path)


def test_wrong_count_fails(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    with pytest.raises(ProvisioningError, match="count"):
        _build(tmp_path, train=5)


def test_non_positive_expected_count_fails(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    with pytest.raises(ProvisioningError, match="positive integer"):
        _build(tmp_path, train=0)


def test_decoder_seam_rejects_wrong_dtype(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    bad = _Decoder(
        read_images=lambda path: torch.zeros(6, 28, 28, dtype=torch.float32),
        read_labels=lambda path: torch.zeros(6, dtype=torch.int64),
    )
    with pytest.raises(ProvisioningError, match="image dtype"):
        _build(tmp_path, decoder=bad)


def test_decoder_seam_rejects_label_dtype(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    bad = _Decoder(
        read_images=lambda path: torch.zeros(6, 28, 28, dtype=torch.uint8),
        read_labels=lambda path: torch.zeros(6, dtype=torch.float32),
    )
    with pytest.raises(ProvisioningError, match="label dtype"):
        _build(tmp_path, decoder=bad)


def test_decoder_seam_failure_is_bounded(tmp_path: Path) -> None:
    _make_cache(tmp_path)

    def _boom(path: str):
        raise RuntimeError("no decoder available " + "x" * 1000)

    bad = _Decoder(read_images=_boom, read_labels=lambda path: torch.zeros(6, dtype=torch.int64))
    with pytest.raises(ProvisioningError, match="could not decode train"):
        _build(tmp_path, decoder=bad)


# --------------------------------------------------------------------------- #
# Interrupted provisioning and foreign manifests
# --------------------------------------------------------------------------- #
def test_interrupted_publication_leaves_no_manifest(tmp_path: Path, monkeypatch) -> None:
    specs = _make_cache(tmp_path)

    def _interrupt(destination: str, data: bytes) -> None:
        raise ProvisioningError("interrupted publish")

    monkeypatch.setattr(mnist_provision, "_publish_manifest_bytes", _interrupt)
    with pytest.raises(ProvisioningError, match="interrupted"):
        _provision(tmp_path, specs=specs)
    assert not (tmp_path / MANIFEST_FILENAME).exists()
    leftovers = [name for name in os.listdir(tmp_path) if name.startswith(".mnist-provision.")]
    assert leftovers == []


def test_publish_never_overwrites_existing_manifest(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    foreign = '{"schema_version":"not-ours"}'
    (tmp_path / MANIFEST_FILENAME).write_text(foreign)
    with pytest.raises(ManifestError):
        _provision(tmp_path)
    assert (tmp_path / MANIFEST_FILENAME).read_text() == foreign


def test_malformed_manifest_rejected(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    (tmp_path / MANIFEST_FILENAME).write_text("{ this is not json")
    with pytest.raises(ManifestError):
        load_manifest(tmp_path)


def test_unknown_schema_version_rejected(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    manifest = _provision(tmp_path)
    payload = json.loads(mnist_provision._manifest_bytes(manifest).decode("utf-8"))
    payload["schema_version"] = "dfl_sim.mnist_provision/v2"
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(payload))
    with pytest.raises(ManifestError, match="schema"):
        load_manifest(tmp_path)


def test_tampered_manifest_body_rejected(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    manifest = _provision(tmp_path)
    payload = json.loads(mnist_provision._manifest_bytes(manifest).decode("utf-8"))
    payload["train"]["count"] = 60000
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(payload))
    with pytest.raises(ManifestError, match="digest"):
        load_manifest(tmp_path)


def test_manifest_symlink_rejected(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    target = tmp_path / "elsewhere.json"
    target.write_text("{}")
    os.symlink(target, tmp_path / MANIFEST_FILENAME)
    with pytest.raises(ManifestError, match="symbolic link"):
        load_manifest(tmp_path)


def test_raw_dir_symlink_escape_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    (root / "MNIST").mkdir(parents=True)
    os.symlink(outside, root / "MNIST" / "raw")
    with pytest.raises(ProvisioningError, match="escapes the data root"):
        _build(root)


def test_unsafe_resource_filename_rejected() -> None:
    specs = list(_specs())
    specs[0] = ResourceSpec("../evil.gz", DUMMY_MD5, "train", "image")
    with pytest.raises(ProvisioningError, match="unsafe"):
        _group_specs(specs)


# --------------------------------------------------------------------------- #
# Mutation detection and lease semantics
# --------------------------------------------------------------------------- #
def test_noncooperating_writer_causes_identity_mismatch(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    manifest = _provision(tmp_path)
    raw = Path(raw_dir(tmp_path))
    (raw / TEST_IMAGES).write_bytes(_idx_images(np.zeros((4, 28, 28), dtype=np.uint8)))
    with pytest.raises(ProvisioningError, match="no longer matches"):
        verify_cache(tmp_path, deep=True)
    # The published manifest itself remains untouched.
    assert load_manifest(tmp_path) == manifest


def test_fast_verify_detects_size_change(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    _provision(tmp_path)
    raw = Path(raw_dir(tmp_path))
    label_file = raw / TRAIN_LABELS
    label_file.write_bytes(label_file.read_bytes() + b"\x00")
    with pytest.raises(ProvisioningError, match="size changed"):
        verify_cache(tmp_path, deep=False)


def test_cache_lease_detects_mutation_during_lease(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    _provision(tmp_path)
    raw = Path(raw_dir(tmp_path))
    mutated = _labels(6, 99)
    with pytest.raises(ProvisioningError, match="no longer matches"):
        with cache_lease(tmp_path, exclusive=True):
            (raw / TRAIN_LABELS).write_bytes(_idx_labels(mutated))


def test_cache_lease_exclusive_collision(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    _provision(tmp_path)
    with cache_lease(tmp_path, exclusive=True):
        with pytest.raises(CacheLockError):
            with cache_lease(tmp_path, exclusive=True):
                pass


def test_cache_lease_shared_then_exclusive_collision(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    _provision(tmp_path)
    with cache_lease(tmp_path, exclusive=False):
        with pytest.raises(CacheLockError):
            with cache_lease(tmp_path, exclusive=True):
                pass
        # Shared leases do not exclude each other.
        with cache_lease(tmp_path, exclusive=False):
            pass


def test_lock_released_after_body_exception(tmp_path: Path) -> None:
    _make_cache(tmp_path)
    _provision(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with cache_lease(tmp_path, exclusive=True):
            raise RuntimeError("boom")
    with cache_lease(tmp_path, exclusive=True) as manifest:
        assert manifest.schema_version == PROVISION_SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_help_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--data-root" in out
    assert "--verify-only" in out
    assert "--no-download" in out


def test_cli_missing_data_fails_nonzero(tmp_path: Path, capsys) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--data-root", str(empty), "--no-download"]) == 2
    assert "error" in capsys.readouterr().err
    assert not (empty / MANIFEST_FILENAME).exists()


def test_cli_verify_only_missing_manifest_fails(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--data-root", str(empty), "--verify-only"]) == 2


def test_cli_rejects_conflicting_download_modes(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--data-root", str(tmp_path), "--download", "--no-download"])
    assert excinfo.value.code == 2


def test_cli_verify_only_json_on_synthetic_cache(tmp_path: Path, capsys) -> None:
    _make_cache(tmp_path)
    _provision(tmp_path)
    assert main(["--data-root", str(tmp_path), "--verify-only", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["schema_version"] == PROVISION_SCHEMA_VERSION
    assert payload["train"]["count"] == 6
    assert payload["test"]["count"] == 4


# --------------------------------------------------------------------------- #
# Contract constants
# --------------------------------------------------------------------------- #
def test_canonical_counts_are_pinned() -> None:
    assert CANONICAL_COUNTS == {"train": 60000, "test": 10000}
    assert LOCK_FILENAME == "mnist-provision.lock"
