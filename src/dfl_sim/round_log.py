"""Canonical deterministic JSONL round log (R001, R012, T04).

The round log is the observable artifact boundary of an episode. It is
deliberately minimal and unversioned: one JSON object per round with a
zero-based round index, per-node pre/post local losses and sample counts, and
one sender-to-sorted-recipient summary per sender. It contains **no**
timestamps, durations, absolute paths, random UUIDs, or host metadata, so the
same config and dataset always produce byte-identical output. Attack metadata
and tensor payloads are intentionally excluded from this minimal log.

Writes go through a same-directory temporary file and are published with
:func:`os.replace` only after every round has succeeded (atomic, no partial
result is ever visible). Any failure removes the temporary file and leaves a
prior complete log untouched. Paths are treated as untrusted input (Q3): a log
target that aliases the input config, is a symlink, is a non-regular file, or
lives under a missing parent is rejected before any training starts.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from .simulator import RoundResult

__all__ = [
    "LogError",
    "RoundLogWriter",
    "canonical_line",
    "canonical_record",
    "validate_log_path",
    "write_round_log",
]


class LogError(RuntimeError):
    """Raised when a round log cannot be validated, written, or published."""


def canonical_record(result: RoundResult) -> dict:
    """Render one round as a canonical, deterministically-ordered mapping.

    Nodes are sorted by node id and messages by sender; each sender's recipient
    list is sorted ascending. Floats are finite by construction (the round
    engine rejects non-finite losses), so :func:`json.dumps` with
    ``allow_nan=False`` never has to reject a legitimate record.
    """

    nodes = sorted(result.stats, key=lambda stat: int(stat.node_id))
    messages = sorted(result.messages, key=lambda message: int(message.sender))
    return {
        "round": int(result.round_index),
        "nodes": [
            {
                "node": int(stat.node_id),
                "pre_loss": float(stat.pre_loss),
                "post_loss": float(stat.post_loss),
                "sample_count": int(stat.sample_count),
            }
            for stat in nodes
        ],
        "messages": [
            {
                "sender": int(message.sender),
                "recipients": sorted(int(recipient) for recipient in message.recipients),
            }
            for message in messages
        ],
    }


def canonical_line(record: Mapping) -> str:
    """Serialize one record: sorted keys, compact separators, ASCII, no NaN."""

    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def validate_log_path(
    log_path: str | os.PathLike,
    *,
    config_path: str | os.PathLike | None = None,
) -> Path:
    """Validate a log destination and return it as a :class:`Path`.

    Rejects a missing/non-directory parent, a symlink target, an existing
    non-regular file, and a path that resolves to the same file as
    ``config_path``.
    """

    path = Path(log_path)
    parent = path.parent if str(path.parent) else Path(".")
    if not parent.exists():
        raise LogError(f"log directory does not exist: {parent}")
    if not parent.is_dir():
        raise LogError(f"log parent is not a directory: {parent}")
    if path.is_symlink():
        raise LogError(f"log path must not be a symlink: {path}")
    if path.exists() and not path.is_file():
        raise LogError(f"log path exists and is not a regular file: {path}")

    if config_path is not None:
        config = Path(config_path)
        try:
            if path.resolve() == config.resolve():
                raise LogError("log path must not alias the config path")
        except OSError as exc:  # pragma: no cover - resolve rarely fails
            raise LogError(
                f"could not resolve log/config paths for alias check: {exc}"
            ) from exc
    return path


class RoundLogWriter:
    """Stream canonical round records to a temporary file, publish atomically.

    Usage::

        with RoundLogWriter(path, config_path=config_path) as writer:
            for result in results:
                writer.write_round(result)
            writer.commit()

    Entering the context creates a hidden temporary file beside the final path.
    :meth:`commit` flushes/fsyncs it and atomically replaces the final path.
    On exception (or :meth:`abort`) the temporary file is removed and any prior
    complete log is left untouched.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        config_path: str | os.PathLike | None = None,
    ) -> None:
        self._path = validate_log_path(path, config_path=config_path)
        self._tmp_path: Path | None = None
        self._handle = None
        self._closed = False

    @property
    def path(self) -> Path:
        """The final publish path."""

        return self._path

    def __enter__(self) -> "RoundLogWriter":
        directory = self._path.parent if str(self._path.parent) else Path(".")
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                dir=str(directory),
            )
        except OSError as exc:
            raise LogError(
                f"could not create temporary log in {directory}: {exc}"
            ) from exc
        self._tmp_path = Path(tmp_name)
        try:
            self._handle = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
        except OSError as exc:  # pragma: no cover - fdopen almost never fails
            os.close(fd)
            self._discard()
            raise LogError(f"could not open temporary log: {exc}") from exc
        return self

    def write_round(self, result: RoundResult) -> None:
        """Append one canonical round record (with a trailing newline)."""

        if self._closed or self._handle is None:
            raise LogError("round log writer is not open")
        try:
            line = canonical_line(canonical_record(result))
        except (TypeError, ValueError) as exc:
            raise LogError(
                f"could not serialize round {int(result.round_index)}: {exc}"
            ) from exc
        self._handle.write(line)
        self._handle.write("\n")

    def commit(self) -> None:
        """Flush the temporary file and atomically publish it."""

        if self._closed:
            raise LogError("round log writer is already closed")
        handle = self._handle
        if handle is None:  # pragma: no cover - guarded by write_round/enter
            raise LogError("round log writer was not opened")
        try:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        except OSError as exc:
            self._discard()
            self._closed = True
            raise LogError(f"could not flush round log: {exc}") from exc

        tmp_path = self._tmp_path
        try:
            os.replace(tmp_path, self._path)
        except OSError as exc:
            self._discard()
            self._closed = True
            raise LogError(
                f"could not publish round log to {self._path}: {exc}"
            ) from exc
        self._tmp_path = None
        self._closed = True

    def abort(self) -> None:
        """Remove the temporary file; never touches the final path."""

        self._discard()
        self._closed = True

    def _discard(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None and not handle.closed:
            try:
                handle.close()
            except OSError:  # pragma: no cover - best effort cleanup
                pass
        tmp_path = self._tmp_path
        self._tmp_path = None
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:  # pragma: no cover - best effort cleanup
                pass

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.abort()
            return False
        if not self._closed:
            self.commit()
        return False


def write_round_log(
    results: Sequence[RoundResult],
    path: str | os.PathLike,
    *,
    config_path: str | os.PathLike | None = None,
) -> Path:
    """Write ``results`` to ``path`` atomically; return the published path."""

    with RoundLogWriter(path, config_path=config_path) as writer:
        for result in results:
            writer.write_round(result)
        writer.commit()
    return writer.path
