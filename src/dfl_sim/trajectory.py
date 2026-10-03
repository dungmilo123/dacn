"""Versioned, deterministic trajectory store and strict reader (R009, D003, D018).

The trajectory store is the single self-describing, corruption-detectable
artifact contract produced by an episode and consumed by S05. It is a
*directory* with a fixed layout and a terminal completion manifest::

    <trace>/config.json     canonical ExperimentConfig snapshot
    <trace>/events.jsonl    one canonical evaluated-round event per line
    <trace>/tensors.npz     deterministic, non-pickle NPZ ZIP of committed states
    <trace>/manifest.json   terminal schema-versioned index with digests

Two properties are load-bearing:

* **Byte reproducibility.** Every JSON payload is canonically encoded
  (sorted keys, compact separators, ASCII, trailing newline, ``allow_nan=False``)
  and every ZIP entry uses fixed metadata (stored, fixed timestamp, fixed
  create-system/mode, little-endian contiguous CPU tensors). The trace carries
  no timestamps, UUIDs, hostnames, absolute/staging paths, source/output paths,
  or any other volatile field, so two clean runs of the same config produce
  byte-identical files.
* **Atomic, non-overwriting publication.** The writer validates the
  destination up front (must not exist, must not be a symlink, must not alias
  the config or any declared source path), stages every artifact in a freshly
  created sibling directory, fsyncs each one, and only then atomically renames
  the complete directory into place. A failure at any point leaves the
  destination absent and removes only the writer-owned staging directory, so an
  incomplete run can never be mistaken for a completed trace.

Both the writer and the reader treat every boundary value as untrusted (Q3):
output paths, JSON/manifest content, and NPZ/ZIP member names. The writer
rejects unsafe targets, object/non-finite tensors, and duplicate entries; the
reader rejects unknown schema versions, malformed manifests, digest/size/count
mismatches, noncanonical event order or encoding, unexpected files, unsafe ZIP
member names, and any tensor payload that would require pickle to load.

The legacy minimal round log in :mod:`dfl_sim.round_log` is intentionally left
unchanged; this module is the richer, versioned successor.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch

from .config import (
    ConfigError,
    ExperimentConfig,
    canonical_dump,
    parse_config_text,
)
from .evaluator import (
    ATTACK_DELTA_REVERSAL,
    ATTACK_NONE,
    BENIGN,
    MALICIOUS,
    EvaluatedRound,
    NodeEvaluation,
)

__all__ = [
    "ARTIFACT_FILENAMES",
    "CONFIG_FILENAME",
    "DECLARED_FILENAMES",
    "EVENTS_FILENAME",
    "MANIFEST_FILENAME",
    "SCHEMA_VERSION",
    "TENSORS_FILENAME",
    "TraceReader",
    "TraceWriter",
    "TrajectoryError",
    "canonical_event",
    "digest_inventory",
    "entry_name",
    "trace_inventory",
    "validate_trace_path",
    "write_trace",
]

#: The one explicit, supported trajectory schema version. Any other value is
#: rejected by both writer and reader; a bump is a deliberate migration.
SCHEMA_VERSION = 1

CONFIG_FILENAME = "config.json"
EVENTS_FILENAME = "events.jsonl"
TENSORS_FILENAME = "tensors.npz"
MANIFEST_FILENAME = "manifest.json"

#: Non-manifest artifacts covered by manifest digests, in canonical order.
ARTIFACT_FILENAMES: tuple[str, ...] = (
    CONFIG_FILENAME,
    EVENTS_FILENAME,
    TENSORS_FILENAME,
)
#: Every file a completed trace directory may contain (manifest included).
DECLARED_FILENAMES: tuple[str, ...] = ARTIFACT_FILENAMES + (MANIFEST_FILENAME,)

_INITIAL_GROUP = "initial"

#: State-dict parameter names that are safe as a single archive path segment.
#: Must start alphanumeric/underscore and may then contain word chars and dots.
_PARAM_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.]*$")
#: Canonical archive member name: ``initial/node_000000/p.npy`` or
#: ``round_000000/node_000000/p.npy``.
_ENTRY_RE = re.compile(
    r"^(?P<group>initial|round_(?P<round>\d{6}))"
    r"/node_(?P<node>\d{6})"
    r"/(?P<param>[A-Za-z0-9_][A-Za-z0-9_.]*)\.npy$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
#: Fixed DOS timestamp for every ZIP entry: the ZIP epoch. Volatile real
#: timestamps would break byte-for-byte reproducibility.
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

_EVENT_KEYS = {"schema_version", "round", "nodes", "messages"}
_NODE_EVENT_KEYS = {
    "node",
    "loss",
    "accuracy",
    "sample_count",
    "true_type",
    "attack",
    "profile",
    "propagation",
}
_MANIFEST_KEYS = {
    "schema_version",
    "nodes",
    "edges",
    "node_types",
    "compromised_node",
    "attack",
    "aggregation",
    "snapshot_cadence",
    "recorded_rounds",
    "event_count",
    "snapshot_rounds",
    "snapshot_count",
    "has_initial_snapshot",
    "tensor_entry_count",
    "completed",
    "artifacts",
}


class TrajectoryError(RuntimeError):
    """Raised when a trajectory cannot be validated, written, read, or published."""


# --------------------------------------------------------------------------- #
# Canonical JSON helpers
# --------------------------------------------------------------------------- #
def _canonical_json(value: Any) -> str:
    """Encode ``value`` as canonical, finite, ASCII, sorted-key compact JSON."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TrajectoryError(f"could not canonically encode JSON: {exc}") from exc


def _reject_json_constant(value: str) -> Any:
    raise TrajectoryError(f"non-finite JSON constant {value!r} is not allowed")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrajectoryError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json(text: str, *, label: str) -> Any:
    """Parse strict JSON, rejecting non-finite constants and duplicate keys."""

    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except TrajectoryError:
        raise
    except json.JSONDecodeError as exc:
        raise TrajectoryError(f"malformed JSON in {label}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Typed value validation
# --------------------------------------------------------------------------- #
def _require_int(value: object, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrajectoryError(f"{label} must be int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise TrajectoryError(f"{label} must be >= {minimum}, got {value}")
    return int(value)


def _require_finite_float(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrajectoryError(f"{label} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not np.isfinite(result):
        raise TrajectoryError(f"{label} must be finite, got {result!r}")
    if minimum is not None and result < minimum:
        raise TrajectoryError(f"{label} must be >= {minimum}, got {result!r}")
    return result


def _validate_param_name(name: object, *, context: str) -> str:
    if not isinstance(name, str) or not name:
        raise TrajectoryError(f"tensor parameter name must be a non-empty string ({context})")
    if ".." in name or "/" in name or "\\" in name or "\x00" in name:
        raise TrajectoryError(f"unsafe tensor parameter name {name!r} ({context})")
    if not _PARAM_RE.match(name):
        raise TrajectoryError(f"unsafe tensor parameter name {name!r} ({context})")
    return name


def entry_name(*, round_index: int | None, node_id: int, param: str) -> str:
    """Return the canonical NPZ member name for one committed tensor.

    ``round_index`` is ``None`` for the mandatory initial snapshot, otherwise a
    non-negative round index. Round and node ids are zero-padded to six digits
    so lexical order matches numeric order.
    """

    node = _require_int(node_id, label="node_id", minimum=0)
    safe_param = _validate_param_name(param, context=f"node={node}")
    if round_index is None:
        prefix = _INITIAL_GROUP
    else:
        index = _require_int(round_index, label="round_index", minimum=0)
        prefix = f"round_{index:06d}"
    return f"{prefix}/node_{node:06d}/{safe_param}.npy"


# --------------------------------------------------------------------------- #
# Tensor normalization and serialization
# --------------------------------------------------------------------------- #
def _tensor_to_array(tensor: object, *, label: str) -> np.ndarray:
    """Normalize a committed tensor to a deterministic, picklable-free array.

    Rejects object/complex/unsupported dtypes and non-finite floats, forces a
    little-endian dtype for multi-byte values (cross-machine byte stability),
    and makes the array C-contiguous so :func:`numpy.lib.format.write_array`
    always emits the same header and payload.
    """

    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        array = tensor
    else:
        raise TrajectoryError(
            f"{label} must be a torch.Tensor or numpy.ndarray, got "
            f"{type(tensor).__name__}"
        )
    if array.dtype.hasobject:
        raise TrajectoryError(f"{label} must not be an object array")
    if np.iscomplexobj(array):
        raise TrajectoryError(f"{label} must be real-valued, got {array.dtype}")
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TrajectoryError(f"{label} has unsupported dtype {array.dtype}")
    if np.issubdtype(array.dtype, np.floating) and not bool(np.isfinite(array).all()):
        raise TrajectoryError(f"{label} contains non-finite values")
    if array.dtype.itemsize > 1:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(array)


def _encode_npy(array: np.ndarray, *, label: str) -> bytes:
    buffer = io.BytesIO()
    try:
        np.lib.format.write_array(buffer, array, allow_pickle=False)
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
        raise TrajectoryError(f"could not serialize {label}: {exc}") from exc
    return buffer.getvalue()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 0
    info.external_attr = 0o600 << 16
    return info


# --------------------------------------------------------------------------- #
# Canonical per-round event serialization
# --------------------------------------------------------------------------- #
def _canonical_messages(messages: object) -> list[dict]:
    if messages is None:
        return []
    if isinstance(messages, (str, bytes)) or not isinstance(messages, (tuple, list)):
        raise TrajectoryError(
            "messages must be a sequence of MessageSummary, got "
            f"{type(messages).__name__}"
        )
    payloads: list[dict] = []
    seen: set[int] = set()
    for message in messages:
        sender = getattr(message, "sender", None)
        recipients = getattr(message, "recipients", None)
        if sender is None or recipients is None:
            raise TrajectoryError(f"message must expose sender/recipients, got {message!r}")
        sender_id = _require_int(sender, label="message sender", minimum=0)
        if sender_id in seen:
            raise TrajectoryError(f"duplicate message summary for sender {sender_id}")
        seen.add(sender_id)
        ordered = sorted(
            _require_int(item, label="message recipient", minimum=0)
            for item in recipients
        )
        if len(set(ordered)) != len(ordered):
            raise TrajectoryError(f"duplicate recipient for sender {sender_id}")
        payloads.append({"sender": sender_id, "recipients": ordered})
    payloads.sort(key=lambda item: item["sender"])
    return payloads


def _node_event(node: NodeEvaluation) -> dict:
    if not isinstance(node, NodeEvaluation):
        raise TrajectoryError("evaluated round nodes must be NodeEvaluation records")
    return {
        "node": int(node.node_id),
        "loss": float(node.loss),
        "accuracy": float(node.accuracy),
        "sample_count": int(node.sample_count),
        "true_type": node.true_type,
        "attack": {"kind": node.attack.kind, "kappa": float(node.attack.kappa)},
        "profile": {
            "participants": [int(item) for item in node.participant_ids],
            "multipliers": [float(item) for item in node.multipliers],
            "weights": [float(item) for item in node.weights],
        },
        "propagation": {
            "reached": bool(node.propagation.reached),
            "min_hop": (
                None if node.propagation.min_hop is None else int(node.propagation.min_hop)
            ),
            "first_reached_round": (
                None
                if node.propagation.first_reached_round is None
                else int(node.propagation.first_reached_round)
            ),
        },
    }


def canonical_event(evaluated: EvaluatedRound, messages: object = ()) -> dict:
    """Render one evaluated round as its canonical event mapping (schema v1)."""

    if not isinstance(evaluated, EvaluatedRound):
        raise TrajectoryError(
            f"evaluated round must be an EvaluatedRound, got {type(evaluated).__name__}"
        )
    nodes = [_node_event(node) for node in evaluated.nodes]
    previous: int | None = None
    for payload in nodes:
        node_id = payload["node"]
        if previous is not None and node_id <= previous:
            raise TrajectoryError("event nodes must be strictly ascending by node id")
        previous = node_id
    return {
        "schema_version": SCHEMA_VERSION,
        "round": int(evaluated.round_index),
        "nodes": nodes,
        "messages": _canonical_messages(messages),
    }


def canonical_event_line(evaluated: EvaluatedRound, messages: object = ()) -> str:
    """Return the canonical JSON line (without newline) for one round."""

    return _canonical_json(canonical_event(evaluated, messages))


# --------------------------------------------------------------------------- #
# Destination validation
# --------------------------------------------------------------------------- #
def validate_trace_path(
    path: str | os.PathLike,
    *,
    config_path: str | os.PathLike | None = None,
    alias_paths: Sequence[str | os.PathLike] = (),
) -> Path:
    """Validate a trace destination that must be created, never overwritten.

    Rejects an empty/relative-dot target, a symlink, an existing path of any
    kind, a missing/non-directory parent, and any target that resolves to the
    same file as ``config_path`` or one of ``alias_paths`` (untrusted Q3 input).
    """

    destination = Path(path)
    if not str(destination) or destination.name in ("", ".", ".."):
        raise TrajectoryError(f"invalid trace destination: {destination!r}")
    if destination.is_symlink():
        raise TrajectoryError(f"trace destination must not be a symlink: {destination}")
    parent = destination.parent if str(destination.parent) else Path(".")
    if not parent.exists():
        raise TrajectoryError(f"trace destination parent does not exist: {parent}")
    if not parent.is_dir():
        raise TrajectoryError(f"trace destination parent is not a directory: {parent}")
    if destination.exists():
        raise TrajectoryError(f"trace destination already exists: {destination}")

    try:
        resolved = destination.resolve()
    except OSError as exc:  # pragma: no cover - resolve rarely fails
        raise TrajectoryError(
            f"could not resolve trace destination {destination}: {exc}"
        ) from exc

    aliases: list[Path] = []
    if config_path is not None:
        aliases.append(Path(config_path))
    aliases.extend(Path(item) for item in alias_paths)
    for alias in aliases:
        try:
            if alias.resolve() == resolved:
                raise TrajectoryError(
                    f"trace destination must not alias {alias}"
                )
        except OSError:  # pragma: no cover - best-effort alias check
            continue
    return destination


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TrajectoryError(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _write_bytes_synced(path: Path, payload: bytes) -> None:
    try:
        with open(path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise TrajectoryError(f"could not write {path}: {exc}") from exc


def _fsync_path(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError as exc:
        raise TrajectoryError(f"could not open {path} for fsync: {exc}") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise TrajectoryError(f"could not fsync {path}: {exc}") from exc
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    """Best-effort directory fsync; directory durability is platform-dependent."""

    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- #
# Trace writer
# --------------------------------------------------------------------------- #
class TraceWriter:
    """Stage and atomically publish a complete schema-v1 trajectory.

    Usage::

        with TraceWriter(path, config=config, topology=topology) as writer:
            writer.write_initial(initial_states)
            for evaluated, result in rounds:
                writer.write_round(
                    evaluated, messages=result.messages, states=result.states
                )

    The initial snapshot is mandatory; exactly ``expected_rounds`` (default
    ``config.rounds``) rounds must be written; snapshots are taken for round 0
    and every ``trace.snapshot_cadence`` rounds plus the final round. The
    context manager commits on clean exit and aborts on exception, leaving the
    destination absent and removing only the sibling staging directory.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        config: ExperimentConfig,
        topology: nx.Graph,
        config_path: str | os.PathLike | None = None,
        alias_paths: Sequence[str | os.PathLike] = (),
        expected_rounds: int | None = None,
    ) -> None:
        if not isinstance(config, ExperimentConfig):
            raise TrajectoryError("config must be an ExperimentConfig")
        if not isinstance(topology, nx.Graph):
            raise TrajectoryError("topology must be a networkx.Graph")
        self._config = config
        self._topology = topology
        self._nodes = tuple(sorted(int(node) for node in topology.nodes()))
        if not self._nodes:
            raise TrajectoryError("topology must have at least one node")
        self._node_set = set(self._nodes)
        if expected_rounds is None:
            expected = int(config.rounds)
        else:
            expected = _require_int(expected_rounds, label="expected_rounds", minimum=1)
        self._expected_rounds = expected
        self._cadence = int(config.trace.snapshot_cadence)
        if self._cadence < 1:
            raise TrajectoryError(f"snapshot_cadence must be positive, got {self._cadence}")

        self._path = validate_trace_path(
            path, config_path=config_path, alias_paths=alias_paths
        )
        self._staging: Path | None = None
        self._events_handle = None
        self._zip: zipfile.ZipFile | None = None
        self._entry_names: set[str] = set()
        self._has_initial = False
        self._next_round = 0
        self._written_rounds = 0
        self._snapshot_rounds: set[int] = set()
        self._closed = False
        self._committed = False

    # -- introspection ------------------------------------------------------ #
    @property
    def path(self) -> Path:
        """The final publish destination (never created until :meth:`commit`)."""

        return self._path

    @property
    def staged_path(self) -> Path | None:
        """The sibling staging directory while open, else ``None``."""

        return self._staging

    @property
    def expected_rounds(self) -> int:
        return self._expected_rounds

    @property
    def next_round_index(self) -> int:
        return self._next_round

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def snapshot_rounds(self) -> tuple[int, ...]:
        return tuple(sorted(self._snapshot_rounds))

    def selected_rounds(self) -> tuple[int, ...]:
        """The round indices that will carry a committed tensor snapshot."""

        return tuple(
            index
            for index in range(self._expected_rounds)
            if self._selects_snapshot(index)
        )

    # -- context management ------------------------------------------------- #
    def __enter__(self) -> "TraceWriter":
        if self._closed or self._staging is not None:
            raise TrajectoryError("trace writer is already open")
        parent = self._path.parent if str(self._path.parent) else Path(".")
        try:
            staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{self._path.name}.staging-", dir=str(parent)
                )
            )
        except OSError as exc:
            raise TrajectoryError(
                f"could not create staging directory for {self._path}: {exc}"
            ) from exc
        self._staging = staging
        try:
            _write_bytes_synced(
                staging / CONFIG_FILENAME,
                (canonical_dump(self._config) + "\n").encode("utf-8"),
            )
            self._events_handle = open(
                staging / EVENTS_FILENAME, "w", encoding="utf-8", newline="\n"
            )
            self._zip = zipfile.ZipFile(
                staging / TENSORS_FILENAME,
                "w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            )
        except TrajectoryError:
            self.abort()
            raise
        except OSError as exc:
            self.abort()
            raise TrajectoryError(
                f"could not initialize trace staging at {staging}: {exc}"
            ) from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.abort()
            return False
        if not self._closed:
            self.commit()
        return False

    # -- streaming API ------------------------------------------------------ #
    def write_initial(self, states: Mapping[int, Mapping[str, object]]) -> None:
        """Stream the mandatory initial committed-state snapshot."""

        self._ensure_open()
        if self._has_initial:
            raise TrajectoryError("initial snapshot has already been written")
        normalized = self._normalize_states(states, context="initial snapshot")
        for node in self._nodes:
            self._write_node_tensors(round_index=None, node_id=node, state=normalized[node])
        self._has_initial = True

    def write_round(
        self,
        evaluated: EvaluatedRound,
        *,
        messages: object = (),
        states: Mapping[int, Mapping[str, object]] | None = None,
    ) -> None:
        """Write one canonical evaluated-round event and any selected snapshot."""

        self._ensure_open()
        if not isinstance(evaluated, EvaluatedRound):
            raise TrajectoryError(
                f"evaluated round must be an EvaluatedRound, got "
                f"{type(evaluated).__name__}"
            )
        index = int(evaluated.round_index)
        if index != self._next_round:
            raise TrajectoryError(
                f"trajectory round out of order: expected {self._next_round}, got {index}"
            )
        if evaluated.node_ids != self._nodes:
            raise TrajectoryError(
                "evaluated round node set does not match the resolved topology: "
                f"expected {list(self._nodes)}, got {list(evaluated.node_ids)}"
            )

        line = canonical_event_line(evaluated, messages)
        try:
            self._events_handle.write(line)
            self._events_handle.write("\n")
        except OSError as exc:
            raise TrajectoryError(f"could not write event for round {index}: {exc}") from exc
        self._written_rounds += 1

        if self._selects_snapshot(index):
            if states is None:
                raise TrajectoryError(
                    f"round {index} is snapshot-selected and requires committed states"
                )
            normalized = self._normalize_states(states, context=f"round {index} snapshot")
            for node in self._nodes:
                self._write_node_tensors(
                    round_index=index, node_id=node, state=normalized[node]
                )
            self._snapshot_rounds.add(index)

        self._next_round = index + 1

    # -- publication -------------------------------------------------------- #
    def commit(self) -> Path:
        """Finalize, fsync, compute digests, and atomically publish the trace."""

        self._ensure_open()
        if not self._has_initial:
            raise TrajectoryError("initial snapshot is required before commit")
        if self._written_rounds != self._expected_rounds:
            raise TrajectoryError(
                f"trajectory is incomplete: expected {self._expected_rounds} rounds, "
                f"got {self._written_rounds}"
            )
        expected_snapshots = set(self.selected_rounds())
        if self._snapshot_rounds != expected_snapshots:
            raise TrajectoryError(
                "snapshot selection mismatch: expected "
                f"{sorted(expected_snapshots)}, got {sorted(self._snapshot_rounds)}"
            )

        staging = self._staging
        assert staging is not None  # guarded by _ensure_open
        self._close_events()
        try:
            self._zip.close()
        except OSError as exc:
            self.abort()
            raise TrajectoryError(f"could not finalize {TENSORS_FILENAME}: {exc}") from exc
        self._zip = None
        try:
            _fsync_path(staging / TENSORS_FILENAME)
        except TrajectoryError:
            self.abort()
            raise

        artifacts: dict[str, dict] = {}
        for filename in ARTIFACT_FILENAMES:
            artifact_path = staging / filename
            if not artifact_path.is_file():
                self.abort()
                raise TrajectoryError(f"missing staged artifact {filename}")
            try:
                size = artifact_path.stat().st_size
            except OSError as exc:
                self.abort()
                raise TrajectoryError(f"could not stat {filename}: {exc}") from exc
            artifacts[filename] = {
                "size_bytes": int(size),
                "sha256": _file_sha256(artifact_path),
            }

        manifest = self._build_manifest(artifacts)
        try:
            _write_bytes_synced(
                staging / MANIFEST_FILENAME,
                (_canonical_json(manifest) + "\n").encode("utf-8"),
            )
        except TrajectoryError:
            self.abort()
            raise

        _fsync_dir(staging)
        try:
            os.replace(staging, self._path)
        except OSError as exc:
            self.abort()
            raise TrajectoryError(
                f"could not publish trace to {self._path}: {exc}"
            ) from exc
        self._staging = None
        self._committed = True
        self._closed = True
        _fsync_dir(self._path.parent if str(self._path.parent) else Path("."))
        return self._path

    def abort(self) -> None:
        """Close handles and remove only the writer-owned staging directory."""

        self._closed = True
        handle = self._events_handle
        self._events_handle = None
        if handle is not None and not handle.closed:
            try:
                handle.close()
            except OSError:  # pragma: no cover - best-effort cleanup
                pass
        zf = self._zip
        self._zip = None
        if zf is not None:
            try:
                zf.close()
            except OSError:  # pragma: no cover - best-effort cleanup
                pass
        staging = self._staging
        self._staging = None
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    # -- internals ---------------------------------------------------------- #
    def _ensure_open(self) -> None:
        if self._closed or self._staging is None:
            raise TrajectoryError("trace writer is not open")

    def _close_events(self) -> None:
        handle = self._events_handle
        if handle is None:
            return
        self._events_handle = None
        try:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        except OSError as exc:
            self.abort()
            raise TrajectoryError(f"could not finalize {EVENTS_FILENAME}: {exc}") from exc

    def _selects_snapshot(self, index: int) -> bool:
        return index % self._cadence == 0 or index == self._expected_rounds - 1

    def _normalize_states(
        self, states: Mapping[int, Mapping[str, object]], *, context: str
    ) -> dict[int, dict[str, np.ndarray]]:
        if not isinstance(states, Mapping):
            raise TrajectoryError(
                f"{context} must be a node-id mapping, got {type(states).__name__}"
            )
        normalized_keys: dict[int, Mapping[str, object]] = {}
        for raw_node, state in states.items():
            node = _require_int(raw_node, label="state node id", minimum=0)
            if node in normalized_keys:
                raise TrajectoryError(f"duplicate state for node {node} ({context})")
            normalized_keys[node] = state
        if set(normalized_keys) != self._node_set:
            missing = sorted(self._node_set - set(normalized_keys))
            extra = sorted(set(normalized_keys) - self._node_set)
            raise TrajectoryError(
                f"{context} node set does not match topology "
                f"(missing={missing}, extra={extra})"
            )

        result: dict[int, dict[str, np.ndarray]] = {}
        for node in self._nodes:
            state = normalized_keys[node]
            if not isinstance(state, Mapping):
                raise TrajectoryError(
                    f"state for node {node} must be a parameter mapping ({context})"
                )
            params: dict[str, np.ndarray] = {}
            for raw_name, tensor in state.items():
                name = _validate_param_name(raw_name, context=f"{context} node={node}")
                if name in params:
                    raise TrajectoryError(
                        f"duplicate tensor {name!r} for node {node} ({context})"
                    )
                params[name] = _tensor_to_array(
                    tensor, label=f"tensor {node}/{name} ({context})"
                )
            if not params:
                raise TrajectoryError(f"state for node {node} has no tensors ({context})")
            result[node] = params
        return result

    def _write_node_tensors(
        self, *, round_index: int | None, node_id: int, state: Mapping[str, np.ndarray]
    ) -> None:
        assert self._zip is not None  # guarded by open
        for param in sorted(state):
            name = entry_name(round_index=round_index, node_id=node_id, param=param)
            if name in self._entry_names:
                raise TrajectoryError(f"duplicate trajectory tensor entry {name}")
            payload = _encode_npy(state[param], label=name)
            try:
                self._zip.writestr(_zip_info(name), payload)
            except OSError as exc:
                raise TrajectoryError(f"could not write tensor entry {name}: {exc}") from exc
            self._entry_names.add(name)

    def _resolve_node_types(self) -> dict[str, str]:
        attacker = self._config.attack.compromised_node
        types: dict[str, str] = {}
        for node in self._nodes:
            if attacker is not None and node == attacker:
                types[str(node)] = MALICIOUS
            else:
                types[str(node)] = BENIGN
        return types

    def _build_manifest(self, artifacts: Mapping[str, Mapping[str, Any]]) -> dict:
        attacker = self._config.attack.compromised_node
        kappa = float(self._config.attack.kappa)
        if attacker is not None and kappa > 0.0:
            attack_kind = ATTACK_DELTA_REVERSAL
        else:
            attack_kind = ATTACK_NONE
            kappa = 0.0
        edges = sorted(
            [min(int(a), int(b)), max(int(a), int(b))]
            for a, b in self._topology.edges()
        )
        node_types = self._resolve_node_types()
        return {
            "schema_version": SCHEMA_VERSION,
            "nodes": list(self._nodes),
            "edges": edges,
            "node_types": {str(node): node_types[str(node)] for node in self._nodes},
            "compromised_node": None if attacker is None else int(attacker),
            "attack": {"kind": attack_kind, "kappa": kappa},
            "aggregation": str(self._config.aggregation.kind),
            "snapshot_cadence": self._cadence,
            "recorded_rounds": self._expected_rounds,
            "event_count": self._written_rounds,
            "snapshot_rounds": sorted(self._snapshot_rounds),
            "snapshot_count": (1 if self._has_initial else 0)
            + len(self._snapshot_rounds),
            "has_initial_snapshot": self._has_initial,
            "tensor_entry_count": len(self._entry_names),
            "completed": True,
            "artifacts": {
                filename: {
                    "size_bytes": int(artifacts[filename]["size_bytes"]),
                    "sha256": str(artifacts[filename]["sha256"]),
                }
                for filename in ARTIFACT_FILENAMES
            },
        }


def write_trace(
    path: str | os.PathLike,
    *,
    config: ExperimentConfig,
    topology: nx.Graph,
    initial_states: Mapping[int, Mapping[str, object]],
    rounds: Iterable[tuple[EvaluatedRound, object]],
    config_path: str | os.PathLike | None = None,
    alias_paths: Sequence[str | os.PathLike] = (),
) -> Path:
    """Write a complete trajectory from evaluated round/committed-result pairs.

    ``rounds`` yields ``(evaluated_round, result)`` pairs where ``result``
    exposes ``messages`` and ``states`` (a
    :class:`~dfl_sim.simulator.RoundResult`). Returns the published path.
    """

    with TraceWriter(
        path,
        config=config,
        topology=topology,
        config_path=config_path,
        alias_paths=alias_paths,
    ) as writer:
        writer.write_initial(initial_states)
        for evaluated, result in rounds:
            writer.write_round(
                evaluated,
                messages=getattr(result, "messages", ()),
                states=getattr(result, "states", None),
            )
    return writer.path


# --------------------------------------------------------------------------- #
# Trace inventory helpers (downstream golden/reproducibility proofs)
# --------------------------------------------------------------------------- #
def trace_inventory(path: str | os.PathLike) -> tuple[str, ...]:
    """Return the sorted relative file inventory of a trace directory."""

    base = Path(path)
    if not base.is_dir():
        raise TrajectoryError(f"trace path is not a directory: {base}")
    return tuple(
        sorted(
            str(item.relative_to(base))
            for item in base.rglob("*")
            if item.is_file()
        )
    )


def digest_inventory(path: str | os.PathLike) -> dict[str, str]:
    """Return a relative-path -> SHA-256 map of every file under ``path``."""

    base = Path(path)
    return {relative: _file_sha256(base / relative) for relative in trace_inventory(path)}


# --------------------------------------------------------------------------- #
# Strict reader
# --------------------------------------------------------------------------- #
def _validate_read_path(path: str | os.PathLike) -> Path:
    base = Path(path)
    if base.is_symlink():
        raise TrajectoryError(f"trace path must not be a symlink: {base}")
    if not base.exists():
        raise TrajectoryError(f"trace path does not exist: {base}")
    if not base.is_dir():
        raise TrajectoryError(f"trace path is not a directory: {base}")
    return base


def _check_layout(base: Path) -> None:
    names = sorted(item.name for item in base.iterdir())
    expected = sorted(DECLARED_FILENAMES)
    if names != expected:
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise TrajectoryError(
            f"trace directory layout mismatch (missing={missing}, unexpected={extra})"
        )
    for filename in DECLARED_FILENAMES:
        if not (base / filename).is_file():
            raise TrajectoryError(f"trace entry is not a regular file: {filename}")


def _validate_artifacts(manifest: Mapping[str, Any], base: Path) -> None:
    artifacts = manifest["artifacts"]
    for filename in ARTIFACT_FILENAMES:
        entry = artifacts[filename]
        path = base / filename
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            raise TrajectoryError(f"could not stat artifact {filename}: {exc}") from exc
        if actual_size != entry["size_bytes"]:
            raise TrajectoryError(
                f"artifact {filename} size mismatch: manifest "
                f"{entry['size_bytes']} != actual {actual_size}"
            )
        actual_digest = _file_sha256(path)
        if actual_digest != entry["sha256"]:
            raise TrajectoryError(
                f"artifact {filename} digest mismatch: manifest "
                f"{entry['sha256']} != actual {actual_digest}"
            )


def _validate_manifest(manifest: Any) -> dict:
    if not isinstance(manifest, dict):
        raise TrajectoryError("manifest must be a JSON object")
    if set(manifest) != _MANIFEST_KEYS:
        missing = sorted(_MANIFEST_KEYS - set(manifest))
        extra = sorted(set(manifest) - _MANIFEST_KEYS)
        raise TrajectoryError(
            f"manifest key mismatch (missing={missing}, unexpected={extra})"
        )
    schema = manifest["schema_version"]
    if isinstance(schema, bool) or schema != SCHEMA_VERSION:
        raise TrajectoryError(
            f"unsupported trajectory schema version {schema!r} "
            f"(supported: {SCHEMA_VERSION})"
        )

    nodes = manifest["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise TrajectoryError("manifest nodes must be a non-empty list")
    for node in nodes:
        _require_int(node, label="manifest node", minimum=0)
    if nodes != sorted(set(nodes)):
        raise TrajectoryError("manifest nodes must be unique and strictly ascending")
    node_set = set(nodes)

    edges = manifest["edges"]
    if not isinstance(edges, list):
        raise TrajectoryError("manifest edges must be a list")
    seen_edges: list[list[int]] = []
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise TrajectoryError(f"manifest edge must be a 2-list, got {edge!r}")
        left = _require_int(edge[0], label="edge endpoint", minimum=0)
        right = _require_int(edge[1], label="edge endpoint", minimum=0)
        if left >= right:
            raise TrajectoryError(f"manifest edge must be ascending, got {edge!r}")
        if left not in node_set or right not in node_set:
            raise TrajectoryError(f"manifest edge references unknown node: {edge!r}")
        seen_edges.append([left, right])
    if seen_edges != sorted(seen_edges):
        raise TrajectoryError("manifest edges must be sorted")
    if len({tuple(edge) for edge in seen_edges}) != len(seen_edges):
        raise TrajectoryError("manifest edges must be unique")

    node_types = manifest["node_types"]
    if not isinstance(node_types, dict):
        raise TrajectoryError("manifest node_types must be an object")
    parsed_types: dict[int, str] = {}
    for key, value in node_types.items():
        if not isinstance(key, str) or not key.isdigit():
            raise TrajectoryError(f"manifest node_types key {key!r} is not a node id")
        if value not in (BENIGN, MALICIOUS):
            raise TrajectoryError(f"manifest node_types[{key!r}] must be benign/malicious")
        parsed_types[int(key)] = value
    if set(parsed_types) != node_set:
        raise TrajectoryError("manifest node_types keys must match nodes")

    compromised = manifest["compromised_node"]
    if compromised is not None:
        compromised = _require_int(compromised, label="manifest compromised_node", minimum=0)
        if compromised not in node_set:
            raise TrajectoryError("manifest compromised_node is not in nodes")
        if parsed_types[compromised] != MALICIOUS:
            raise TrajectoryError("manifest compromised_node must be marked malicious")
    elif any(value == MALICIOUS for value in parsed_types.values()):
        raise TrajectoryError("manifest marks a malicious node without a compromised_node")

    attack = manifest["attack"]
    if not isinstance(attack, dict) or set(attack) != {"kind", "kappa"}:
        raise TrajectoryError("manifest attack must carry exactly kind and kappa")
    if attack["kind"] not in (ATTACK_NONE, ATTACK_DELTA_REVERSAL):
        raise TrajectoryError(f"unknown manifest attack kind {attack['kind']!r}")
    kappa = _require_finite_float(attack["kappa"], label="manifest attack kappa", minimum=0.0)
    if attack["kind"] == ATTACK_NONE and kappa != 0.0:
        raise TrajectoryError("manifest attack kind 'none' requires kappa 0")
    if attack["kind"] == ATTACK_DELTA_REVERSAL and kappa <= 0.0:
        raise TrajectoryError("manifest attack kind 'delta_reversal' requires kappa > 0")

    if not isinstance(manifest["aggregation"], str) or not manifest["aggregation"]:
        raise TrajectoryError("manifest aggregation must be a non-empty string")
    cadence = _require_int(manifest["snapshot_cadence"], label="snapshot_cadence", minimum=1)
    recorded = _require_int(manifest["recorded_rounds"], label="recorded_rounds", minimum=1)
    event_count = _require_int(manifest["event_count"], label="event_count", minimum=0)
    if event_count != recorded:
        raise TrajectoryError("manifest event_count must equal recorded_rounds")

    snapshot_rounds = manifest["snapshot_rounds"]
    if not isinstance(snapshot_rounds, list) or not snapshot_rounds:
        raise TrajectoryError("manifest snapshot_rounds must be a non-empty list")
    for index in snapshot_rounds:
        _require_int(index, label="snapshot_round", minimum=0)
        if index >= recorded:
            raise TrajectoryError("manifest snapshot_round is outside the recorded rounds")
    if snapshot_rounds != sorted(set(snapshot_rounds)):
        raise TrajectoryError("manifest snapshot_rounds must be unique and ascending")
    if snapshot_rounds[-1] != recorded - 1:
        raise TrajectoryError("manifest snapshot_rounds must include the final round")
    expected_selected = [
        index for index in range(recorded) if index % cadence == 0 or index == recorded - 1
    ]
    if snapshot_rounds != expected_selected:
        raise TrajectoryError(
            "manifest snapshot_rounds do not match the configured cadence"
        )

    if manifest["has_initial_snapshot"] is not True:
        raise TrajectoryError("manifest must declare the initial snapshot")
    snapshot_count = _require_int(manifest["snapshot_count"], label="snapshot_count", minimum=1)
    if snapshot_count != 1 + len(snapshot_rounds):
        raise TrajectoryError("manifest snapshot_count is inconsistent")
    _require_int(manifest["tensor_entry_count"], label="tensor_entry_count", minimum=1)
    if manifest["completed"] is not True:
        raise TrajectoryError("manifest is not completed")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_FILENAMES):
        raise TrajectoryError("manifest artifacts must cover exactly the declared files")
    for filename, entry in artifacts.items():
        if not isinstance(entry, dict) or set(entry) != {"size_bytes", "sha256"}:
            raise TrajectoryError(f"manifest artifact {filename!r} is malformed")
        _require_int(entry["size_bytes"], label=f"artifact {filename} size", minimum=0)
        digest = entry["sha256"]
        if not isinstance(digest, str) or not _SHA256_RE.match(digest):
            raise TrajectoryError(f"manifest artifact {filename!r} has an invalid digest")
    return manifest


def _validate_event_node(node: Any, *, position: int, previous: int | None) -> int:
    if not isinstance(node, dict) or set(node) != _NODE_EVENT_KEYS:
        raise TrajectoryError(f"event {position} node payload has unexpected keys")
    node_id = _require_int(node["node"], label="event node id", minimum=0)
    if previous is not None and node_id <= previous:
        raise TrajectoryError(
            f"event {position} nodes must be strictly ascending by node id"
        )
    _require_finite_float(node["loss"], label="event node loss", minimum=0.0)
    accuracy = _require_finite_float(node["accuracy"], label="event node accuracy")
    if not 0.0 <= accuracy <= 1.0:
        raise TrajectoryError("event node accuracy must be within [0, 1]")
    _require_int(node["sample_count"], label="event sample_count", minimum=1)
    if node["true_type"] not in (BENIGN, MALICIOUS):
        raise TrajectoryError(f"event node true_type {node['true_type']!r} is invalid")

    attack = node["attack"]
    if not isinstance(attack, dict) or set(attack) != {"kind", "kappa"}:
        raise TrajectoryError("event node attack must carry exactly kind and kappa")
    if attack["kind"] not in (ATTACK_NONE, ATTACK_DELTA_REVERSAL):
        raise TrajectoryError(f"event node attack kind {attack['kind']!r} is invalid")
    _require_finite_float(attack["kappa"], label="event node attack kappa", minimum=0.0)

    profile = node["profile"]
    if not isinstance(profile, dict) or set(profile) != {
        "participants",
        "multipliers",
        "weights",
    }:
        raise TrajectoryError("event node profile is malformed")
    participants = profile["participants"]
    multipliers = profile["multipliers"]
    weights = profile["weights"]
    for vector, label in (
        (participants, "participants"),
        (multipliers, "multipliers"),
        (weights, "weights"),
    ):
        if not isinstance(vector, list) or not vector:
            raise TrajectoryError(f"event node profile {label} must be a non-empty list")
    if len(participants) != len(multipliers) or len(participants) != len(weights):
        raise TrajectoryError("event node profile vector lengths must match")
    for value in participants:
        _require_int(value, label="event profile participant", minimum=0)
    if participants[0] != node_id:
        raise TrajectoryError("event profile participants must begin with the node id")
    for value in multipliers:
        _require_finite_float(value, label="event profile multiplier", minimum=0.0)
    for value in weights:
        _require_finite_float(value, label="event profile weight", minimum=0.0)

    propagation = node["propagation"]
    if not isinstance(propagation, dict) or set(propagation) != {
        "reached",
        "min_hop",
        "first_reached_round",
    }:
        raise TrajectoryError("event node propagation is malformed")
    reached = propagation["reached"]
    if not isinstance(reached, bool):
        raise TrajectoryError("event propagation reached must be bool")
    if reached:
        _require_int(propagation["min_hop"], label="event min_hop", minimum=0)
        _require_int(
            propagation["first_reached_round"],
            label="event first_reached_round",
            minimum=0,
        )
    elif propagation["min_hop"] is not None or propagation["first_reached_round"] is not None:
        raise TrajectoryError("unreached event propagation must not carry hop fields")
    return node_id


def _validate_event(event: Any, *, position: int) -> None:
    if not isinstance(event, dict) or set(event) != _EVENT_KEYS:
        raise TrajectoryError(f"event {position} has unexpected keys")
    if event["schema_version"] != SCHEMA_VERSION:
        raise TrajectoryError(
            f"event {position} has unsupported schema version "
            f"{event['schema_version']!r}"
        )
    if event["round"] != position:
        raise TrajectoryError(
            f"event {position} declares round {event['round']!r} (noncanonical order)"
        )
    nodes = event["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise TrajectoryError(f"event {position} nodes must be a non-empty list")
    previous: int | None = None
    for node in nodes:
        previous = _validate_event_node(node, position=position, previous=previous)

    messages = event["messages"]
    if not isinstance(messages, list):
        raise TrajectoryError(f"event {position} messages must be a list")
    previous_sender: int | None = None
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"sender", "recipients"}:
            raise TrajectoryError(f"event {position} message is malformed")
        sender = _require_int(message["sender"], label="event message sender", minimum=0)
        if previous_sender is not None and sender <= previous_sender:
            raise TrajectoryError(
                f"event {position} messages must be strictly ascending by sender"
            )
        previous_sender = sender
        recipients = message["recipients"]
        if not isinstance(recipients, list):
            raise TrajectoryError("event message recipients must be a list")
        for recipient in recipients:
            _require_int(recipient, label="event recipient", minimum=0)
        if recipients != sorted(set(recipients)):
            raise TrajectoryError("event message recipients must be sorted and unique")


def _validate_entry_name(name: Any) -> tuple[str, int | None, int, str]:
    if not isinstance(name, str) or not name:
        raise TrajectoryError(f"archive entry name must be a non-empty string, got {name!r}")
    if name.startswith("/") or "\\" in name or "\x00" in name:
        raise TrajectoryError(f"unsafe archive entry name {name!r}")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise TrajectoryError(f"unsafe archive entry path {name!r}")
    match = _ENTRY_RE.match(name)
    if match is None:
        raise TrajectoryError(f"noncanonical archive entry name {name!r}")
    param = match.group("param")
    if ".." in param:
        raise TrajectoryError(f"unsafe archive entry name {name!r}")
    group = match.group("group")
    round_index = int(match.group("round")) if match.group("round") is not None else None
    return group, round_index, int(match.group("node")), param


class TraceReader:
    """Strict, validating reader for a completed schema-v1 trajectory (S05).

    Construction verifies the directory layout, manifest structure, artifact
    sizes and SHA-256 digests, canonical config/event encoding and round order,
    and the exact NPZ/ZIP member set. Subsequent snapshot reads are lazy: a
    single named tensor is loaded and validated on demand, never the whole
    archive at once.
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = _validate_read_path(path)
        _check_layout(self._path)
        manifest_bytes = (self._path / MANIFEST_FILENAME).read_bytes()
        try:
            manifest_text = manifest_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TrajectoryError(f"manifest is not valid UTF-8: {exc}") from exc
        self._manifest = _validate_manifest(
            _parse_json(manifest_text, label=MANIFEST_FILENAME)
        )
        _validate_artifacts(self._manifest, self._path)

        try:
            self._config_text = (self._path / CONFIG_FILENAME).read_text(encoding="utf-8")
            self._events_text = (self._path / EVENTS_FILENAME).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise TrajectoryError(f"could not read trace artifacts: {exc}") from exc

        self._config = self._parse_config_snapshot()
        self._events = self._parse_events()

        try:
            self._zip = zipfile.ZipFile(self._path / TENSORS_FILENAME, "r")
        except (OSError, zipfile.BadZipFile) as exc:
            raise TrajectoryError(f"tensors.npz is not a readable ZIP: {exc}") from exc
        self._closed = False
        self._entry_names: set[str] = set()
        try:
            self._archive = self._validate_archive()
        except TrajectoryError:
            self._zip.close()
            raise

    # -- context management ------------------------------------------------- #
    def __enter__(self) -> "TraceReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._zip.close()

    # -- validated surface -------------------------------------------------- #
    @property
    def path(self) -> Path:
        return self._path

    @property
    def manifest(self) -> dict:
        """The validated manifest mapping."""

        return dict(self._manifest)

    @property
    def config(self) -> ExperimentConfig:
        """The validated ``ExperimentConfig`` snapshot from ``config.json``."""

        return self._config

    @property
    def events(self) -> tuple[dict, ...]:
        """All validated events in canonical round order."""

        return tuple(dict(event) for event in self._events)

    @property
    def nodes(self) -> tuple[int, ...]:
        return tuple(int(node) for node in self._manifest["nodes"])

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(left), int(right)) for left, right in self._manifest["edges"]
        )

    @property
    def node_types(self) -> dict[int, str]:
        return {
            int(key): str(value) for key, value in self._manifest["node_types"].items()
        }

    @property
    def snapshot_rounds(self) -> tuple[int, ...]:
        return tuple(int(index) for index in self._manifest["snapshot_rounds"])

    @property
    def has_initial_snapshot(self) -> bool:
        return bool(self._manifest["has_initial_snapshot"])

    @property
    def tensor_entry_count(self) -> int:
        return int(self._manifest["tensor_entry_count"])

    @property
    def schema_version(self) -> int:
        return int(self._manifest["schema_version"])

    def topology(self) -> nx.Graph:
        """Resolve the manifest into a node/edge graph annotated with types."""

        graph = nx.Graph()
        for node in self.nodes:
            graph.add_node(node, true_type=self.node_types[node])
        graph.add_edges_from(self.edges)
        return graph

    # -- lazy snapshot reads ------------------------------------------------ #
    def entry_names(self) -> tuple[str, ...]:
        """The sorted canonical archive member names."""

        return tuple(sorted(self._entry_names))

    def snapshot_entry_names(self, round_index: int | None) -> tuple[str, ...]:
        """The canonical member names for one snapshot group."""

        group = self._group_for(round_index)
        return tuple(sorted(name for name in self._entry_names if name.startswith(f"{group}/")))

    def read_tensor(
        self, round_index: int | None, node_id: int, param: str
    ) -> np.ndarray:
        """Lazily load and validate one named snapshot tensor."""

        self._group_for(round_index)
        name = entry_name(round_index=round_index, node_id=node_id, param=param)
        if name not in self._entry_names:
            raise TrajectoryError(f"trajectory does not contain tensor {name}")
        try:
            raw = self._zip.read(name)
        except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
            raise TrajectoryError(f"could not read tensor {name}: {exc}") from exc
        try:
            array = np.load(io.BytesIO(raw), allow_pickle=False)
        except ValueError as exc:
            raise TrajectoryError(
                f"tensor {name} cannot be loaded without pickle: {exc}"
            ) from exc
        if not isinstance(array, np.ndarray):
            raise TrajectoryError(f"tensor {name} did not decode to an array")
        if array.dtype.hasobject:
            raise TrajectoryError(f"tensor {name} must not be an object array")
        if np.issubdtype(array.dtype, np.floating) and not bool(np.isfinite(array).all()):
            raise TrajectoryError(f"tensor {name} contains non-finite values")
        return array

    def read_node_snapshot(
        self, round_index: int | None, node_id: int
    ) -> dict[str, np.ndarray]:
        """Lazily load every tensor for one node in one snapshot group."""

        group = self._group_for(round_index)
        if node_id not in self._node_params(group):
            raise TrajectoryError(
                f"trajectory snapshot {group} has no node {node_id}"
            )
        return {
            param: self.read_tensor(round_index, node_id, param)
            for param in self._node_params(group)[node_id]
        }

    def read_round_snapshot(
        self, round_index: int
    ) -> dict[int, dict[str, np.ndarray]]:
        """Lazily load a full committed snapshot keyed by node id."""

        return {
            node: self.read_node_snapshot(round_index, node) for node in self.nodes
        }

    def read_initial_snapshot(self) -> dict[int, dict[str, np.ndarray]]:
        """Lazily load the mandatory initial snapshot keyed by node id."""

        return {
            node: self.read_node_snapshot(None, node) for node in self.nodes
        }

    # -- internals ---------------------------------------------------------- #
    def _parse_config_snapshot(self) -> ExperimentConfig:
        try:
            config = parse_config_text(
                self._config_text, source=f"{self._path / CONFIG_FILENAME}"
            )
        except ConfigError as exc:
            raise TrajectoryError(f"invalid config snapshot: {exc}") from exc
        if canonical_dump(config) + "\n" != self._config_text:
            raise TrajectoryError("config.json is not canonically encoded")
        return config

    def _parse_events(self) -> list[dict]:
        text = self._events_text
        if not text.endswith("\n"):
            raise TrajectoryError("events.jsonl must end with a trailing newline")
        lines = text[:-1].split("\n")
        expected = int(self._manifest["recorded_rounds"])
        if len(lines) != expected:
            raise TrajectoryError(
                f"events.jsonl line count mismatch: manifest {expected}, actual {len(lines)}"
            )
        events: list[dict] = []
        for position, raw in enumerate(lines):
            if not raw:
                raise TrajectoryError(f"events.jsonl line {position + 1} is blank")
            event = _parse_json(raw, label=f"{EVENTS_FILENAME}:{position + 1}")
            _validate_event(event, position=position)
            if _canonical_json(event) != raw:
                raise TrajectoryError(
                    f"event {position} is not canonically encoded"
                )
            events.append(event)
        return events

    def _validate_archive(self) -> dict[str, dict[int, tuple[str, ...]]]:
        infos = self._zip.infolist()
        names = [info.filename for info in infos]
        if len(set(names)) != len(names):
            raise TrajectoryError("tensors.npz contains duplicate entry names")
        self._entry_names = set(names)
        node_set = set(self.nodes)
        has_initial = self.has_initial_snapshot
        snapshot_rounds = set(self.snapshot_rounds)

        groups: dict[str, dict[int, set[str]]] = {}
        for name in names:
            group, round_index, node, param = _validate_entry_name(name)
            if node not in node_set:
                raise TrajectoryError(f"archive entry {name!r} references unknown node")
            if group == _INITIAL_GROUP:
                if not has_initial:
                    raise TrajectoryError(
                        f"archive entry {name!r} present without an initial snapshot"
                    )
            else:
                if round_index not in snapshot_rounds:
                    raise TrajectoryError(
                        f"archive entry {name!r} references unrecorded snapshot round"
                    )
            groups.setdefault(group, {}).setdefault(node, set()).add(param)

        expected_groups = {_INITIAL_GROUP} | {
            f"round_{index:06d}" for index in snapshot_rounds
        }
        if set(groups) != expected_groups:
            missing = sorted(expected_groups - set(groups))
            extra = sorted(set(groups) - expected_groups)
            raise TrajectoryError(
                f"archive snapshot group mismatch (missing={missing}, unexpected={extra})"
            )
        for group, by_node in groups.items():
            if set(by_node) != node_set:
                missing = sorted(node_set - set(by_node))
                raise TrajectoryError(
                    f"archive group {group} is missing nodes {missing}"
                )
            param_sets = {frozenset(params) for params in by_node.values()}
            if len(param_sets) != 1:
                raise TrajectoryError(
                    f"archive group {group} has inconsistent per-node parameter sets"
                )
        if len(names) != int(self._manifest["tensor_entry_count"]):
            raise TrajectoryError(
                "archive entry count does not match the manifest tensor_entry_count"
            )
        return {
            group: {
                node: tuple(sorted(params)) for node, params in by_node.items()
            }
            for group, by_node in groups.items()
        }

    def _group_for(self, round_index: int | None) -> str:
        if round_index is None:
            if not self.has_initial_snapshot:
                raise TrajectoryError("trace has no initial snapshot")
            return _INITIAL_GROUP
        index = _require_int(round_index, label="round_index", minimum=0)
        if index not in set(self.snapshot_rounds):
            raise TrajectoryError(f"round {index} is not a recorded snapshot round")
        return f"round_{index:06d}"

    def _node_params(self, group: str) -> dict[int, tuple[str, ...]]:
        if group not in self._archive:
            raise TrajectoryError(f"trajectory has no snapshot group {group!r}")
        return self._archive[group]
