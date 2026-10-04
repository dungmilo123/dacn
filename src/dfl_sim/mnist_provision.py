"""Integrity-checked MNIST provisioning and cache identity (M003/S04, R017).

This module provisions the *actual* torchvision MNIST dataset (downloaded
through torchvision's pinned resources and mirrors) and publishes a strict,
versioned manifest that binds the raw archives and decoded tensors to
path-independent SHA-256 content identities. It never substitutes injected or
synthetic tensors: an absent, corrupt, truncated or unavailable cache fails
closed with a bounded error before any calibration runs.

The manifest is immutable once published. Reuse revalidates the on-disk content
against the recorded identities, so a non-cooperating writer produces a
detected identity mismatch instead of silently re-labelling evidence.
:func:`cache_lease` holds a file lock so cooperating writers sharing one data
root cannot interleave, and re-verifies the identity when the lease exits so a
change during a headline run is detected.

Production never trusts a ``real_mnist`` string or an ``isinstance`` check: the
published identities are computed from the downloaded bytes and the decoded
``uint8`` tensors, and both splits must validate before the manifest is
atomically linked into place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:  # pragma: no cover - exercised implicitly on POSIX
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "CANONICAL_COUNTS",
    "IMAGE_SHAPE",
    "LOCK_FILENAME",
    "MANIFEST_FILENAME",
    "MNIST_SUBDIR",
    "PREPROCESSING_ID",
    "PROVISION_SCHEMA_VERSION",
    "RAW_SUBDIR",
    "CacheLockError",
    "ManifestError",
    "ProvisionManifest",
    "ProvisioningError",
    "ResourceSpec",
    "SplitIdentity",
    "build_manifest",
    "cache_lease",
    "load_manifest",
    "main",
    "manifest_path",
    "provision",
    "raw_dir",
    "verify_cache",
]

# --------------------------------------------------------------------------- #
# Contract constants
# --------------------------------------------------------------------------- #
PROVISION_SCHEMA_VERSION = "dfl_sim.mnist_provision/v1"
PREPROCESSING_ID = "torchvision.transforms.ToTensor:v1"
MNIST_SUBDIR = "MNIST"
RAW_SUBDIR = "raw"
MANIFEST_FILENAME = "mnist-provision.json"
LOCK_FILENAME = "mnist-provision.lock"

CANONICAL_COUNTS: Mapping[str, int] = {"train": 60000, "test": 10000}
IMAGE_SHAPE = (28, 28)
NUM_CLASSES = 10

#: Bounded read ceiling for one raw idx file (train images are ~47 MiB).
MAX_RAW_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
_MAX_TEXT = 400

#: Canonical, path-independent description of the production preprocessing.
_PREPROCESSING_SPEC = b"ToTensor|uint8:HxW->float32:1xHxW|scale=1/255|layout=CHW"

_SPLITS = ("train", "test")
_KINDS = ("image", "label")


class ProvisioningError(RuntimeError):
    """Raised when MNIST cannot be provisioned or verified; bounded and safe."""


class ManifestError(ProvisioningError):
    """Raised when a published manifest is missing, foreign or tampered with."""


class CacheLockError(ProvisioningError):
    """Raised when a data-root cache lease cannot be acquired."""


# --------------------------------------------------------------------------- #
# Manifest schema (strict, frozen, versioned)
# --------------------------------------------------------------------------- #
class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SplitIdentity(_StrictModel):
    """One split's authenticated resources plus content identities."""

    split: Literal["train", "test"]
    image_resource: str = Field(min_length=1, max_length=128)
    label_resource: str = Field(min_length=1, max_length=128)
    image_archive_md5: str = Field(min_length=32, max_length=32)
    label_archive_md5: str = Field(min_length=32, max_length=32)
    raw_image_sha256: str = Field(min_length=64, max_length=64)
    raw_label_sha256: str = Field(min_length=64, max_length=64)
    decoded_image_sha256: str = Field(min_length=64, max_length=64)
    decoded_label_sha256: str = Field(min_length=64, max_length=64)
    raw_image_bytes: int = Field(ge=1, le=MAX_RAW_BYTES)
    raw_label_bytes: int = Field(ge=1, le=MAX_RAW_BYTES)
    count: int = Field(ge=1)
    image_shape: tuple[int, int]
    label_min: int = Field(ge=0, le=NUM_CLASSES - 1)
    label_max: int = Field(ge=0, le=NUM_CLASSES - 1)
    image_dtype: Literal["uint8"] = "uint8"
    label_dtype: Literal["int64"] = "int64"


class ProvisionManifest(_StrictModel):
    """Versioned, digest-sealed provisioning manifest for one data root."""

    schema_version: Literal["dfl_sim.mnist_provision/v1"] = PROVISION_SCHEMA_VERSION
    dataset: Literal["mnist"] = "mnist"
    preprocessing_id: str = Field(min_length=1, max_length=128)
    preprocessing_digest: str = Field(min_length=64, max_length=64)
    train: SplitIdentity
    test: SplitIdentity
    manifest_digest: str = Field(min_length=64, max_length=64)


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """A pinned upstream MNIST resource (archive filename + MD5 checksum)."""

    filename: str
    md5: str
    split: str
    kind: str


@dataclass(frozen=True, slots=True)
class _Decoder:
    read_images: Callable[[str], Any]
    read_labels: Callable[[str], Any]


# --------------------------------------------------------------------------- #
# Bounded helpers
# --------------------------------------------------------------------------- #
def _bounded(detail: object, *, limit: int = _MAX_TEXT) -> str:
    """Collapse whitespace and truncate untrusted error text."""

    collapsed = " ".join(str(detail).split())
    if len(collapsed) > limit:
        collapsed = collapsed[:limit] + "..."
    return collapsed


def _torch() -> Any:
    import torch

    return torch


def _resolve_root(data_root: Any) -> str:
    """Normalize a data root, rejecting empty/NUL/oversized inputs."""

    if data_root is None:
        raise ProvisioningError("data root is required")
    try:
        text = os.fspath(data_root)
    except TypeError as exc:
        raise ProvisioningError(
            f"data root must be a path, got {type(data_root).__name__}"
        ) from exc
    if not isinstance(text, str) or not text:
        raise ProvisioningError("data root must be a non-empty path")
    if "\x00" in text:
        raise ProvisioningError("data root must not contain NUL bytes")
    if len(text) > 1024:
        raise ProvisioningError("data root path is too long")
    return os.path.realpath(text)


def manifest_path(data_root: Any) -> str:
    """Return the resolved manifest path for ``data_root``."""

    return os.path.join(_resolve_root(data_root), MANIFEST_FILENAME)


def raw_dir(data_root: Any) -> str:
    """Return the torchvision raw directory for ``data_root``."""

    return os.path.join(_resolve_root(data_root), MNIST_SUBDIR, RAW_SUBDIR)


def _lock_path(root: str) -> str:
    return os.path.join(root, LOCK_FILENAME)


def _canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _md5_file(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bounded(path: str) -> bytes:
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise ProvisioningError(
            _bounded(f"could not stat raw file {os.path.basename(path)!r}: {exc}")
        ) from exc
    if size > MAX_RAW_BYTES:
        raise ProvisioningError(
            _bounded(f"raw file {os.path.basename(path)!r} exceeds the size bound")
        )
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise ProvisioningError(
            _bounded(f"could not read raw file {os.path.basename(path)!r}: {exc}")
        ) from exc


def _confined(root: str, name: str) -> str:
    """Resolve a bare resource filename inside ``root`` with alias rejection."""

    if not name or name != os.path.basename(name) or name in (".", ".."):
        raise ProvisioningError(_bounded(f"unsafe resource name: {name!r}"))
    root_real = os.path.realpath(root)
    candidate = os.path.join(root_real, name)
    resolved = os.path.realpath(candidate)
    try:
        common = os.path.commonpath([root_real, resolved])
    except ValueError as exc:  # pragma: no cover - different drives (Windows)
        raise ProvisioningError(
            _bounded(f"resource path cannot be confined: {name!r}")
        ) from exc
    if common != root_real:
        raise ProvisioningError(
            _bounded(f"resource path escapes the data root: {name!r}")
        )
    return candidate


def _confined_raw_dir(root: str) -> str:
    """Return the raw directory after rejecting symlink/path escapes."""

    root_real = os.path.realpath(root)
    candidate = os.path.join(root_real, MNIST_SUBDIR, RAW_SUBDIR)
    resolved = os.path.realpath(candidate)
    try:
        common = os.path.commonpath([root_real, resolved])
    except ValueError as exc:  # pragma: no cover - different drives (Windows)
        raise ProvisioningError(
            "raw MNIST directory cannot be confined to the data root"
        ) from exc
    if common != root_real:
        raise ProvisioningError("raw MNIST directory escapes the data root")
    return candidate


def _raw_basename(filename: str) -> str:
    """Drop the ``.gz`` suffix to reach torchvision's decompressed filename."""

    base = os.path.splitext(filename)[0]
    if not base:
        raise ProvisioningError(_bounded(f"malformed resource filename: {filename!r}"))
    return base


def _preprocessing_digest() -> str:
    return hashlib.sha256(_PREPROCESSING_SPEC).hexdigest()


# --------------------------------------------------------------------------- #
# Resource specs / decoder seams
# --------------------------------------------------------------------------- #
def default_resources() -> tuple[ResourceSpec, ...]:
    """Return the pinned MNIST resources declared by torchvision."""

    from torchvision.datasets import MNIST

    specs: list[ResourceSpec] = []
    for filename, md5 in MNIST.resources:
        name = str(filename)
        if name.startswith("train-"):
            split = "train"
        elif name.startswith("t10k-"):
            split = "test"
        else:
            raise ProvisioningError(_bounded(f"unexpected MNIST resource: {name!r}"))
        kind = "image" if "images" in name else "label"
        specs.append(ResourceSpec(name, str(md5), split, kind))
    grouped = _group_specs(specs)
    if set(grouped) != set(_SPLITS):  # pragma: no cover - defensive
        raise ProvisioningError("torchvision MNIST resources are incomplete")
    return tuple(specs)


def _default_decoder() -> _Decoder:
    from torchvision.datasets.mnist import read_image_file, read_label_file

    return _Decoder(read_images=read_image_file, read_labels=read_label_file)


def _group_specs(
    resources: Sequence[ResourceSpec],
) -> dict[str, dict[str, ResourceSpec]]:
    grouped: dict[str, dict[str, ResourceSpec]] = {split: {} for split in _SPLITS}
    for spec in resources:
        if not isinstance(spec, ResourceSpec):
            raise ProvisioningError(
                f"resource specs must be ResourceSpec, got {type(spec).__name__}"
            )
        if spec.split not in grouped:
            raise ProvisioningError(_bounded(f"unknown resource split: {spec.split!r}"))
        if spec.kind not in _KINDS:
            raise ProvisioningError(_bounded(f"unknown resource kind: {spec.kind!r}"))
        if not spec.filename or spec.filename != os.path.basename(spec.filename):
            raise ProvisioningError(_bounded(f"unsafe resource filename: {spec.filename!r}"))
        if len(spec.md5) != 32 or any(c not in "0123456789abcdef" for c in spec.md5.lower()):
            raise ProvisioningError(_bounded(f"invalid MD5 for {spec.filename!r}"))
        if spec.kind in grouped[spec.split]:
            raise ProvisioningError(
                _bounded(f"duplicate {spec.kind} resource for {spec.split} split")
            )
        grouped[spec.split][spec.kind] = spec
    for split in _SPLITS:
        if set(grouped[split]) != set(_KINDS):
            raise ProvisioningError(
                _bounded(f"missing image/label resources for {split} split")
            )
    return grouped


def _resources_from_manifest(manifest: ProvisionManifest) -> tuple[ResourceSpec, ...]:
    specs: list[ResourceSpec] = []
    for split, recorded in (("train", manifest.train), ("test", manifest.test)):
        specs.append(
            ResourceSpec(recorded.image_resource, recorded.image_archive_md5, split, "image")
        )
        specs.append(
            ResourceSpec(recorded.label_resource, recorded.label_archive_md5, split, "label")
        )
    return tuple(specs)


# --------------------------------------------------------------------------- #
# Split validation
# --------------------------------------------------------------------------- #
def _verify_archive(raw_root: str, spec: ResourceSpec, *, require_archive: bool) -> None:
    archive = _confined(raw_root, spec.filename)
    if os.path.isfile(archive):
        actual = _md5_file(archive)
        if actual != spec.md5:
            raise ProvisioningError(
                _bounded(
                    f"{spec.filename} checksum mismatch: expected {spec.md5}, got {actual}"
                )
            )
        return
    if os.path.exists(archive):
        raise ProvisioningError(
            _bounded(f"{spec.filename} is not a regular file")
        )
    if require_archive:
        raise ProvisioningError(
            _bounded(f"missing upstream archive {spec.filename!r}")
        )


def _check_preprocessing(sample: Any, *, split: str) -> None:
    try:
        import numpy as np
        from PIL import Image
        from torchvision import transforms
    except Exception as exc:
        raise ProvisioningError(
            _bounded(f"preprocessing dependency unavailable: {exc}")
        ) from exc

    if not isinstance(sample, np.ndarray):
        raise ProvisioningError(f"{split} preprocessing sample is not a numpy array")
    if tuple(sample.shape) != IMAGE_SHAPE:
        raise ProvisioningError(
            f"{split} preprocessing sample shape {tuple(sample.shape)} is not {IMAGE_SHAPE}"
        )
    expected = transforms.ToTensor()(Image.fromarray(sample.astype(np.uint8, copy=False), mode="L"))
    if tuple(expected.shape) != (1, *IMAGE_SHAPE) or str(expected.dtype) != "torch.float32":
        raise ProvisioningError(f"{split} ToTensor preprocessing shape/dtype mismatch")
    low = float(expected.min())
    high = float(expected.max())
    if low < 0.0 or high > 1.0:
        raise ProvisioningError(f"{split} ToTensor preprocessing range is not [0, 1]")


def _validate_split(
    raw_root: str,
    split: str,
    specs: Mapping[str, ResourceSpec],
    *,
    expected_count: int,
    decoder: _Decoder,
    require_archive: bool,
) -> SplitIdentity:
    if set(specs) != set(_KINDS):
        raise ProvisioningError(_bounded(f"missing resources for {split} split"))
    image_spec = specs["image"]
    label_spec = specs["label"]

    _verify_archive(raw_root, image_spec, require_archive=require_archive)
    _verify_archive(raw_root, label_spec, require_archive=require_archive)

    raw_image_path = _confined(raw_root, _raw_basename(image_spec.filename))
    raw_label_path = _confined(raw_root, _raw_basename(label_spec.filename))
    if not os.path.isfile(raw_image_path):
        raise ProvisioningError(_bounded(f"missing {split} image file"))
    if not os.path.isfile(raw_label_path):
        raise ProvisioningError(_bounded(f"missing {split} label file"))

    raw_image_bytes = _read_bounded(raw_image_path)
    raw_label_bytes = _read_bounded(raw_label_path)

    try:
        images = decoder.read_images(raw_image_path)
        labels = decoder.read_labels(raw_label_path)
    except Exception as exc:
        raise ProvisioningError(
            _bounded(f"could not decode {split} split: {exc}")
        ) from exc

    torch = _torch()
    try:
        image_shape = tuple(int(v) for v in images.shape)
        label_count = int(labels.shape[0])
    except Exception as exc:  # pragma: no cover - defensive
        raise ProvisioningError(
            _bounded(f"decoded {split} split has no readable shape: {exc}")
        ) from exc

    if images.dtype != torch.uint8:
        raise ProvisioningError(
            _bounded(f"{split} image dtype is {images.dtype}, expected uint8")
        )
    if labels.dtype != torch.int64:
        raise ProvisioningError(
            _bounded(f"{split} label dtype is {labels.dtype}, expected int64")
        )
    if len(image_shape) != 3 or image_shape[1:] != IMAGE_SHAPE:
        raise ProvisioningError(
            _bounded(f"{split} image shape {image_shape} is not (N, 28, 28)")
        )
    if image_shape[0] != expected_count:
        raise ProvisioningError(
            _bounded(f"{split} image count {image_shape[0]} != expected {expected_count}")
        )
    if label_count != expected_count:
        raise ProvisioningError(
            _bounded(f"{split} label count {label_count} != expected {expected_count}")
        )
    label_min = int(labels.min())
    label_max = int(labels.max())
    if label_min < 0 or label_max >= NUM_CLASSES:
        raise ProvisioningError(
            _bounded(f"{split} labels out of range [{label_min}, {label_max}]")
        )

    _check_preprocessing(images[0].numpy(), split=split)

    return SplitIdentity(
        split=split,
        image_resource=image_spec.filename,
        label_resource=label_spec.filename,
        image_archive_md5=image_spec.md5,
        label_archive_md5=label_spec.md5,
        raw_image_sha256=hashlib.sha256(raw_image_bytes).hexdigest(),
        raw_label_sha256=hashlib.sha256(raw_label_bytes).hexdigest(),
        decoded_image_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),
        decoded_label_sha256=hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
        raw_image_bytes=len(raw_image_bytes),
        raw_label_bytes=len(raw_label_bytes),
        count=expected_count,
        image_shape=IMAGE_SHAPE,
        label_min=label_min,
        label_max=label_max,
    )


# --------------------------------------------------------------------------- #
# Manifest seal / load / publish
# --------------------------------------------------------------------------- #
def _manifest_body_bytes(manifest: ProvisionManifest) -> bytes:
    body = manifest.model_dump(mode="json", exclude={"manifest_digest"})
    return _canonical_bytes(body)


def _manifest_bytes(manifest: ProvisionManifest) -> bytes:
    return _canonical_bytes(manifest.model_dump(mode="json"))


def _manifest_digest(manifest: ProvisionManifest) -> str:
    return hashlib.sha256(_manifest_body_bytes(manifest)).hexdigest()


def load_manifest(data_root: Any) -> ProvisionManifest:
    """Load and validate a published manifest without touching raw content."""

    root = _resolve_root(data_root)
    path = os.path.join(root, MANIFEST_FILENAME)
    if os.path.islink(path):
        raise ManifestError("manifest must not be a symbolic link")
    if not os.path.exists(path):
        raise ManifestError(_bounded("mnist provisioning manifest is not published"))
    if not os.path.isfile(path):
        raise ManifestError("mnist provisioning manifest is not a regular file")
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise ManifestError(_bounded(f"could not read manifest: {exc}")) from exc
    if len(data) > MAX_MANIFEST_BYTES:
        raise ManifestError("manifest exceeds the size bound")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError("manifest is not valid UTF-8") from exc
    try:
        manifest = ProvisionManifest.model_validate_json(text)
    except ValidationError as exc:
        raise ManifestError(_bounded(f"manifest schema invalid: {exc}")) from exc
    if _manifest_digest(manifest) != manifest.manifest_digest:
        raise ManifestError("manifest digest does not match its body")
    return manifest


def _publish_manifest_bytes(destination: str, data: bytes) -> None:
    """Atomically link a manifest into place without overwriting (seam)."""

    parent = os.path.dirname(destination)
    try:
        fd, staging = tempfile.mkstemp(
            prefix=".mnist-provision.", suffix=".tmp", dir=parent
        )
    except OSError as exc:
        raise ProvisioningError(_bounded(f"could not stage manifest: {exc}")) from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staging, destination)
        except FileExistsError as exc:
            raise ManifestError("manifest already exists; refusing to overwrite") from exc
        except OSError as exc:
            raise ProvisioningError(_bounded(f"could not publish manifest: {exc}")) from exc
        _fsync_dir(parent)
    finally:
        try:
            os.unlink(staging)
        except FileNotFoundError:  # pragma: no cover - already linked away
            pass
        except OSError:  # pragma: no cover - best-effort cleanup
            pass


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - best effort
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- #
# Provisioning
# --------------------------------------------------------------------------- #
def build_manifest(
    data_root: Any,
    *,
    resources: Sequence[ResourceSpec] | None = None,
    decoder: _Decoder | None = None,
    expected_counts: Mapping[str, int] | None = None,
    require_archive: bool = False,
) -> ProvisionManifest:
    """Validate both splits on disk and return a sealed manifest (no write)."""

    root = _resolve_root(data_root)
    specs = tuple(resources) if resources is not None else default_resources()
    grouped = _group_specs(specs)
    dec = decoder or _default_decoder()
    counts = dict(CANONICAL_COUNTS if expected_counts is None else expected_counts)
    if set(counts) != set(_SPLITS):
        raise ProvisioningError("expected_counts must cover both train and test")
    for split in _SPLITS:
        try:
            if int(counts[split]) < 1:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ProvisioningError(
                f"expected count for {split} must be a positive integer"
            ) from exc

    root_raw = _confined_raw_dir(root)
    if not os.path.isdir(root_raw):
        raise ProvisioningError(_bounded(f"raw MNIST directory not found: {root_raw!r}"))

    identities: dict[str, SplitIdentity] = {}
    for split in _SPLITS:
        identities[split] = _validate_split(
            root_raw,
            split,
            grouped[split],
            expected_count=int(counts[split]),
            decoder=dec,
            require_archive=require_archive,
        )

    partial = ProvisionManifest(
        preprocessing_id=PREPROCESSING_ID,
        preprocessing_digest=_preprocessing_digest(),
        train=identities["train"],
        test=identities["test"],
        manifest_digest="0" * 64,
    )
    return partial.model_copy(update={"manifest_digest": _manifest_digest(partial)})


def _download_mnist(root: str) -> None:
    """Download missing resources through torchvision's pinned mirrors."""

    try:
        os.makedirs(raw_dir(root), exist_ok=True)
    except OSError as exc:
        raise ProvisioningError(_bounded(f"could not create data root: {exc}")) from exc
    try:
        from torchvision.datasets import MNIST

        MNIST(root=root, train=True, download=True)
    except Exception as exc:
        raise ProvisioningError(_bounded(f"MNIST download failed: {exc}")) from exc


def _revalidate_against_manifest(
    root: str, manifest: ProvisionManifest, *, decoder: _Decoder | None
) -> None:
    dec = decoder or _default_decoder()
    grouped = _group_specs(_resources_from_manifest(manifest))
    root_raw = _confined_raw_dir(root)
    if not os.path.isdir(root_raw):
        raise ProvisioningError(_bounded(f"raw MNIST directory not found: {root_raw!r}"))
    for split, recorded in (("train", manifest.train), ("test", manifest.test)):
        recomputed = _validate_split(
            root_raw,
            split,
            grouped[split],
            expected_count=recorded.count,
            decoder=dec,
            require_archive=False,
        )
        if recomputed != recorded:
            raise ProvisioningError(
                _bounded(f"{split} cache content no longer matches the published manifest")
            )


def _fast_revalidate(root: str, manifest: ProvisionManifest) -> None:
    root_raw = _confined_raw_dir(root)
    for split, recorded in (("train", manifest.train), ("test", manifest.test)):
        for resource, expected_bytes in (
            (recorded.image_resource, recorded.raw_image_bytes),
            (recorded.label_resource, recorded.raw_label_bytes),
        ):
            raw = _confined(root_raw, _raw_basename(resource))
            if not os.path.isfile(raw):
                raise ProvisioningError(_bounded(f"missing {split} raw file for {resource!r}"))
            if os.path.getsize(raw) != expected_bytes:
                raise ProvisioningError(
                    _bounded(f"{split} raw file {resource!r} size changed since publication")
                )


def verify_cache(
    data_root: Any,
    *,
    deep: bool = True,
    decoder: _Decoder | None = None,
) -> ProvisionManifest:
    """Load the manifest and revalidate the on-disk content.

    ``deep=True`` recomputes the content identities (bounded O(70000) once);
    ``deep=False`` compares raw file sizes only, for cheap between-cell checks.
    Either way a mutation raises :class:`ProvisioningError`.
    """

    root = _resolve_root(data_root)
    manifest = load_manifest(root)
    if deep:
        _revalidate_against_manifest(root, manifest, decoder=decoder)
    else:
        _fast_revalidate(root, manifest)
    return manifest


def provision(
    data_root: Any,
    *,
    download: bool = False,
    resources: Sequence[ResourceSpec] | None = None,
    decoder: _Decoder | None = None,
    expected_counts: Mapping[str, int] | None = None,
    require_archive: bool = True,
    publish: bool = True,
) -> ProvisionManifest:
    """Provision or reuse a verified cache and expose its identity.

    A published manifest is never overwritten: reuse revalidates content
    against it and fails closed on any mismatch. When no manifest exists, the
    current raw content is validated and atomically published.
    """

    root = _resolve_root(data_root)
    if download:
        try:
            os.makedirs(raw_dir(root), exist_ok=True)
        except OSError as exc:
            raise ProvisioningError(
                _bounded(f"could not create data root: {exc}")
            ) from exc
    with _file_lock(root, exclusive=True):
        published = os.path.join(root, MANIFEST_FILENAME)
        if os.path.exists(published):
            manifest = load_manifest(root)
            _revalidate_against_manifest(root, manifest, decoder=decoder)
            return manifest
        if download:
            _download_mnist(root)
        manifest = build_manifest(
            root,
            resources=resources,
            decoder=decoder,
            expected_counts=expected_counts,
            require_archive=require_archive,
        )
        if publish:
            _publish_manifest_bytes(published, _manifest_bytes(manifest))
        return manifest


# --------------------------------------------------------------------------- #
# Cache lease
# --------------------------------------------------------------------------- #
@contextmanager
def _file_lock(root: str, *, exclusive: bool) -> Iterator[None]:
    if fcntl is None:  # pragma: no cover - non-POSIX fallback
        yield
        return
    path = _lock_path(root)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise ProvisioningError(_bounded(f"could not open cache lock: {exc}")) from exc
    try:
        flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(fd, flags | fcntl.LOCK_NB)
        except OSError as exc:
            raise CacheLockError(
                _bounded("cache is locked by another process")
            ) from exc
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:  # pragma: no cover - best effort
                pass
    finally:
        os.close(fd)


@contextmanager
def cache_lease(
    data_root: Any,
    *,
    exclusive: bool = True,
    deep: bool = True,
    decoder: _Decoder | None = None,
) -> Iterator[ProvisionManifest]:
    """Hold a lock over a verified cache and detect mutation across the lease.

    The lease verifies before entering and re-verifies on exit. A change during
    the leased work (or a non-cooperating writer) raises
    :class:`ProvisioningError` unless the body itself already raised, in which
    case the body error is preserved.
    """

    root = _resolve_root(data_root)
    with _file_lock(root, exclusive=exclusive):
        manifest = verify_cache(root, deep=deep, decoder=decoder)
        try:
            yield manifest
        finally:
            active = sys.exc_info()[0] is not None
            try:
                verify_cache(root, deep=deep, decoder=decoder)
            except ProvisioningError:
                if not active:
                    raise


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _summary(manifest: ProvisionManifest, root: str) -> str:
    return (
        "mnist_provision: verified "
        f"data_root={root} "
        f"train={manifest.train.count} test={manifest.test.count} "
        f"train_images_sha256={manifest.train.decoded_image_sha256} "
        f"test_images_sha256={manifest.test.decoded_image_sha256} "
        f"manifest_digest={manifest.manifest_digest}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dfl_sim.mnist_provision",
        description=(
            "Provision and verify integrity-checked MNIST without injected fallback."
        ),
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="data root that holds MNIST/raw and the provisioning manifest",
    )
    download_group = parser.add_mutually_exclusive_group()
    download_group.add_argument(
        "--download",
        dest="download",
        action="store_true",
        help="allow torchvision to download missing pinned resources (default)",
    )
    download_group.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="never touch the network; reuse or verify the existing cache only",
    )
    parser.set_defaults(download=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the published manifest and cache without writing anything",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the verified manifest JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the human-readable summary",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.verify_only:
            manifest = verify_cache(args.data_root, deep=True)
        else:
            manifest = provision(args.data_root, download=args.download)
    except ProvisioningError as exc:
        print(f"mnist_provision: error: {_bounded(exc)}", file=sys.stderr)
        return 2
    if args.json:
        print(_manifest_bytes(manifest).decode("utf-8"))
    if not args.quiet and not args.json:
        print(_bounded(_summary(manifest, _resolve_root(args.data_root)), limit=2000))
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI entry point
    raise SystemExit(main())
