"""Atomic, no-overwrite publisher for the frozen calibration artifact (T03).

The publisher writes exactly the three fixed schema-v1 files
(``manifest.json``, ``likelihood.json``, ``payoff.json``) as canonical JSON into
a **private same-parent staging directory**, fsyncs every file and the staging
directory, re-validates the staged directory through the S01
:func:`~dfl_sim.bayesian_artifact.load_bayesian_artifact` loader, and only then
atomically renames the staging directory into place. The requested output must
not already exist, so a prior complete artifact is never replaced.

Untrusted filesystem input (Q3): the destination path and any pre-existing
object at it are untrusted. The publisher rejects a missing/non-directory
parent, a symlinked parent or destination, an existing destination, a
destination that aliases the calibration config or data root, and extra files
or declared-digest drift inside the staging directory. Every failure removes
only the staging directory; user-visible state is left unchanged.

Test-only injection seams are module-level ``_write_staged_file``,
``_fsync_dir``, ``_validate_staged``, and ``_rename`` so a caller can force a
write, validation, or rename failure and assert the atomic cleanup guarantee.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .bayesian_artifact import (
    ARTIFACT_FILENAMES,
    MAX_ARTIFACT_FILE_BYTES,
    MANIFEST_FILENAME,
    LIKELIHOOD_FILENAME,
    PAYOFF_FILENAME,
    ArtifactManifest,
    canonical_json_bytes,
    compute_aggregate_digest,
    load_bayesian_artifact,
)
from .calibration_estimator import EstimatedArtifact

__all__ = [
    "CalibrationPublishError",
    "PublishedArtifact",
    "publish_calibration_artifact",
]

#: Staging-directory prefix (hidden, same parent as the destination).
_STAGING_PREFIX = ".dfl-calibration-stage-"


class CalibrationPublishError(RuntimeError):
    """Raised when the artifact directory cannot be published atomically.

    Raised for an unsafe or aliased destination, an existing output, a declared
    digest that does not match the payload bytes, an extra staged file, or a
    staged loader-validation failure. The staging directory is removed on every
    failure and the destination is never created or replaced.
    """


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    """The result of one successful publication.

    ``manifest`` is the re-parsed S01 manifest of the published directory;
    ``likelihood_bytes`` / ``payoff_bytes`` are the exact bytes written. The
    bounded scalar diagnostics (episodes, rounds, cell counts, coverage, and
    aggregate digest) are all reachable through ``manifest``.
    """

    path: Path
    manifest: ArtifactManifest
    likelihood_bytes: bytes
    payoff_bytes: bytes

    @property
    def aggregate_digest(self) -> str:
        return self.manifest.aggregate_digest


# --------------------------------------------------------------------------- #
# Injection seams (monkeypatched by failure-injection tests)
# --------------------------------------------------------------------------- #
def _write_staged_file(path: str, data: bytes) -> None:
    """Write one bounded regular file with private mode and fsync it."""

    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:  # pragma: no cover - already closed
            pass
        raise


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_staged(path: str) -> None:
    load_bayesian_artifact(path)


def _rename(source: str, destination: str) -> None:
    os.rename(source, destination)


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_aliases(
    destination_real: str,
    *,
    config_path: str | os.PathLike[str] | None,
    data_root: str | os.PathLike[str] | None,
) -> None:
    if config_path is not None:
        config_real = os.path.realpath(os.fspath(config_path))
        if config_real == destination_real:
            raise CalibrationPublishError(
                "output aliases the calibration config path"
            )
    if data_root is not None:
        data_real = os.path.realpath(os.fspath(data_root))
        if data_real == destination_real:
            raise CalibrationPublishError("output aliases the data root")
        try:
            common = os.path.commonpath([destination_real, data_real])
        except ValueError:  # pragma: no cover - different drives (Windows)
            common = ""
        if common and common == data_real:
            raise CalibrationPublishError("output is inside the data root")


def _validate_destination(destination: str | os.PathLike[str]) -> str:
    """Return the absolute destination after all pre-publication safety checks."""

    destination_text = os.fspath(destination)
    if not destination_text:
        raise CalibrationPublishError("output path is empty")
    absolute = os.path.abspath(destination_text)

    parent = os.path.dirname(absolute)
    if not parent or parent == absolute:
        raise CalibrationPublishError(f"output has no usable parent: {absolute!r}")
    try:
        parent_info = os.lstat(parent)
    except FileNotFoundError as exc:
        raise CalibrationPublishError(
            f"output parent does not exist: {parent!r}"
        ) from exc
    except OSError as exc:
        raise CalibrationPublishError(
            f"output parent is unreadable: {parent!r}"
        ) from exc
    if stat.S_ISLNK(parent_info.st_mode):
        raise CalibrationPublishError(
            f"output parent must not be a symlink: {parent!r}"
        )
    if not stat.S_ISDIR(parent_info.st_mode):
        raise CalibrationPublishError(
            f"output parent is not a directory: {parent!r}"
        )

    if os.path.lexists(absolute):
        raise CalibrationPublishError(f"output already exists: {absolute!r}")
    return absolute


# --------------------------------------------------------------------------- #
# Manifest / payload validation
# --------------------------------------------------------------------------- #
def _verify_payload(
    declared: object, data: bytes, *, filename: str
) -> None:
    size = getattr(declared, "size_bytes", None)
    digest = getattr(declared, "sha256", None)
    if size is None or digest is None:
        raise CalibrationPublishError(
            f"manifest is missing the declared digest for {filename}"
        )
    if len(data) > MAX_ARTIFACT_FILE_BYTES:
        raise CalibrationPublishError(
            f"{filename} exceeds the artifact file size bound"
        )
    if int(size) != len(data):
        raise CalibrationPublishError(
            f"declared size for {filename} does not match payload bytes"
        )
    if str(digest) != _sha256(data):
        raise CalibrationPublishError(
            f"declared digest for {filename} does not match payload bytes"
        )


def _manifest_bytes(manifest: ArtifactManifest) -> bytes:
    """Re-canonicalize the manifest and verify its declared aggregate digest."""

    body = manifest.model_dump(by_alias=True, mode="json")
    recomputed = compute_aggregate_digest(body)
    if recomputed != manifest.aggregate_digest:
        raise CalibrationPublishError(
            "manifest aggregate digest does not match manifest body"
        )
    return canonical_json_bytes(body)


# --------------------------------------------------------------------------- #
# Publisher
# --------------------------------------------------------------------------- #
def publish_calibration_artifact(
    destination: str | os.PathLike[str],
    artifact: EstimatedArtifact,
    *,
    config_path: str | os.PathLike[str] | None = None,
    data_root: str | os.PathLike[str] | None = None,
) -> PublishedArtifact:
    """Publish ``artifact`` to ``destination`` atomically and never overwrite.

    Writes the three fixed files into a private same-parent staging directory,
    fsyncs them and the directory, validates the staged directory through the
    S01 loader, and atomically renames it into place. Any failure removes the
    staging directory and leaves the requested output untouched.
    """

    if not isinstance(artifact, EstimatedArtifact):
        raise CalibrationPublishError(
            f"publish requires an EstimatedArtifact, got {type(artifact).__name__}"
        )

    absolute = _validate_destination(destination)
    destination_real = os.path.realpath(absolute)
    _reject_aliases(
        destination_real, config_path=config_path, data_root=data_root
    )

    likelihood_bytes = artifact.likelihood_bytes
    payoff_bytes = artifact.payoff_bytes
    declared = artifact.manifest.payloads
    _verify_payload(
        declared.get(LIKELIHOOD_FILENAME), likelihood_bytes, filename=LIKELIHOOD_FILENAME
    )
    _verify_payload(
        declared.get(PAYOFF_FILENAME), payoff_bytes, filename=PAYOFF_FILENAME
    )
    manifest_bytes = _manifest_bytes(artifact.manifest)

    parent = os.path.dirname(absolute)
    staging: str | None = None
    try:
        staging = tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=parent)
        os.chmod(staging, 0o700)
        _write_staged_file(
            os.path.join(staging, MANIFEST_FILENAME), manifest_bytes
        )
        _write_staged_file(
            os.path.join(staging, LIKELIHOOD_FILENAME), likelihood_bytes
        )
        _write_staged_file(
            os.path.join(staging, PAYOFF_FILENAME), payoff_bytes
        )
        _fsync_dir(staging)

        try:
            names = set(os.listdir(staging))
        except OSError as exc:  # pragma: no cover - staging is private
            raise CalibrationPublishError(
                f"could not inspect staging directory: {exc}"
            ) from exc
        if names != set(ARTIFACT_FILENAMES):
            raise CalibrationPublishError(
                "staging directory does not hold exactly the three fixed files"
            )

        try:
            _validate_staged(staging)
        except Exception as exc:
            if isinstance(exc, CalibrationPublishError):
                raise
            raise CalibrationPublishError(
                f"staged artifact failed S01 validation: {exc}"
            ) from exc

        if os.path.lexists(absolute):
            raise CalibrationPublishError(f"output already exists: {absolute!r}")
        _rename(staging, absolute)
        staging = None
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        _fsync_dir(parent)
    except OSError:  # pragma: no cover - durability best effort
        pass

    loaded = load_bayesian_artifact(absolute)
    return PublishedArtifact(
        path=Path(absolute),
        manifest=loaded.manifest,
        likelihood_bytes=likelihood_bytes,
        payoff_bytes=payoff_bytes,
    )
