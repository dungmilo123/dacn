"""Golden whole-trace reproducibility and atomic-publication tests (T04).

These are the strongest integrated S04 acceptance tests. Two clean traces are
produced in **independent subprocesses** (each a fresh interpreter driving
:func:`dfl_sim.run.run_from_config` through ``tests/trace_fixture_runner.py``),
then compared byte-for-byte, reopened through the strict :class:`TraceReader`,
and checked against a checked-in SHA-256 golden digest JSON. The golden holds
only hashes and structural expectations, never a bulky tensor archive.

Atomic negative end-to-end tests then inject a late evaluator/writer failure and
a pre-existing destination and assert that no newly published trace or
completion manifest appears at the requested target.

Q3 threat surface: golden tests inspect only declared trace files through the
strict reader and hash raw bytes; subprocess arguments are list-form local
paths; fixtures never read ``.gsd``/``.gitignore``/network/credentials/external
state.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dfl_sim import (
    ARTIFACT_FILENAMES,
    DECLARED_FILENAMES,
    EpisodeError,
    EvaluatorError,
    TrajectoryError,
    TraceReader,
    TraceWriter,
    digest_inventory,
    load_config,
    run_from_config,
    trace_inventory,
)
from dfl_sim.evaluator import PrivilegedEvaluator
from helpers import MNISTShapedDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_CONFIG = REPO_ROOT / "configs" / "clean_3node.yaml"
RUNNER = REPO_ROOT / "tests" / "trace_fixture_runner.py"
GOLDEN = REPO_ROOT / "tests" / "golden" / "clean_3node_trace.sha256.json"

#: Files that must not appear in a canonical trace directory. Any text artifact
#: containing one of these tokens is non-reproducible.
_VOLATILE_TOKENS = ("timestamp", "created_at", "uuid", "hostname", "datetime", "mtime")
_TEXT_ARTIFACTS = ("config.json", "events.jsonl", "manifest.json")


def _load_golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _run_fixture(destination: Path) -> Path:
    """Publish one clean trace through a fresh interpreter."""

    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--config",
            str(CLEAN_CONFIG),
            "--output",
            str(destination),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"fixture runner failed ({proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return destination


def _fixture_datasets():
    return (
        MNISTShapedDataset(192, seed=5, separable=True),
        MNISTShapedDataset(96, seed=9, separable=True),
    )


def _assert_no_target(root: Path, target: Path) -> None:
    assert not target.exists()
    assert not (target / "manifest.json").exists()
    # Writer-owned staging is cleaned; nothing else lingers next to the target.
    assert list(root.iterdir()) == [], f"lingering staging entries: {list(root.iterdir())}"


# --------------------------------------------------------------------------- #
# Bitwise whole-trace reproducibility + golden digest
# --------------------------------------------------------------------------- #
def test_golden_trace_is_bitwise_reproducible_and_matches_golden(tmp_path: Path) -> None:
    golden = _load_golden()
    first = _run_fixture(tmp_path / "first")
    second = _run_fixture(tmp_path / "second")

    # Sorted relative inventory: declared layout only, no extra/volatile files.
    inventory = trace_inventory(first)
    assert inventory == tuple(golden["inventory"])
    assert inventory == tuple(sorted(DECLARED_FILENAMES))
    assert trace_inventory(second) == inventory
    assert sorted(path.name for path in first.iterdir()) == sorted(DECLARED_FILENAMES)

    # Every declared file is byte-for-byte identical across the two runs.
    for relative in inventory:
        assert (first / relative).read_bytes() == (second / relative).read_bytes(), (
            f"non-reproducible trace file: {relative}"
        )

    # The complete SHA-256 map is stable and matches the checked-in golden.
    digests = digest_inventory(first)
    assert digests == digest_inventory(second)
    if digests != golden["digests"]:
        mismatches = {
            name: (golden["digests"].get(name), digests.get(name))
            for name in sorted(set(golden["digests"]) | set(digests))
            if golden["digests"].get(name) != digests.get(name)
        }
        pytest.fail(f"golden digest mismatch: {mismatches}")

    # Both traces reopen through the strict reader with identical content.
    with TraceReader(first) as r1, TraceReader(second) as r2:
        assert r1.schema_version == r2.schema_version == golden["expected"]["schema_version"]
        assert r1.manifest == r2.manifest
        assert r1.events == r2.events
        assert r1.nodes == r2.nodes
        assert r1.edges == r2.edges
        assert r1.node_types == r2.node_types
        assert r1.snapshot_rounds == r2.snapshot_rounds
        assert r1.tensor_entry_count == r2.tensor_entry_count


# --------------------------------------------------------------------------- #
# Structure, cadence, digests, and absence of volatile fields
# --------------------------------------------------------------------------- #
def test_golden_trace_structure_cadence_and_no_volatile_fields(tmp_path: Path) -> None:
    golden = _load_golden()
    expected = golden["expected"]
    trace = _run_fixture(tmp_path / "trace")

    with TraceReader(trace) as reader:
        assert reader.schema_version == expected["schema_version"]
        assert reader.nodes == tuple(expected["nodes"])
        assert reader.edges == tuple(tuple(edge) for edge in expected["edges"])
        assert reader.has_initial_snapshot is expected["has_initial_snapshot"]
        assert reader.snapshot_rounds == tuple(expected["snapshot_rounds"])
        assert reader.tensor_entry_count == expected["tensor_entry_count"]
        assert {
            node: kind for node, kind in reader.node_types.items()
        } == {int(node): kind for node, kind in expected["node_types"].items()}
        assert len(reader.events) == expected["event_count"]

        manifest = reader.manifest
        assert manifest["schema_version"] == expected["schema_version"]
        assert manifest["nodes"] == expected["nodes"]
        assert manifest["edges"] == expected["edges"]
        assert manifest["compromised_node"] == expected["compromised_node"]
        assert manifest["attack"]["kind"] == expected["attack_kind"]
        assert manifest["aggregation"] == expected["aggregation"]
        assert manifest["snapshot_cadence"] == expected["snapshot_cadence"]
        assert manifest["recorded_rounds"] == expected["recorded_rounds"]
        assert manifest["event_count"] == expected["event_count"]
        assert manifest["snapshot_rounds"] == expected["snapshot_rounds"]
        assert manifest["snapshot_count"] == expected["snapshot_count"]
        assert manifest["has_initial_snapshot"] is expected["has_initial_snapshot"]
        assert manifest["tensor_entry_count"] == expected["tensor_entry_count"]
        assert manifest["completed"] is expected["completed"]
        assert set(manifest["artifacts"]) == set(ARTIFACT_FILENAMES)
        assert all(
            entry["size_bytes"] > 0 and len(entry["sha256"]) == 64
            for entry in manifest["artifacts"].values()
        )

    # Exactly one trailing JSONL newline and one canonical event per round.
    events_bytes = (trace / "events.jsonl").read_bytes()
    assert events_bytes.endswith(b"\n")
    assert not events_bytes.endswith(b"\n\n")
    lines = events_bytes.decode("utf-8").splitlines()
    assert len(lines) == expected["event_count"]
    for line in lines:
        payload = json.loads(line)
        assert payload["schema_version"] == expected["schema_version"]

    # No volatile tokens in any text artifact (path/timestamp/uuid/host leaks).
    for name in _TEXT_ARTIFACTS:
        lowered = (trace / name).read_text(encoding="utf-8").lower()
        for token in _VOLATILE_TOKENS:
            assert token not in lowered, f"{name} contains volatile token {token!r}"


# --------------------------------------------------------------------------- #
# Atomic negative end-to-end tests (Q5/Q7)
# --------------------------------------------------------------------------- #
def test_late_evaluator_failure_leaves_no_trace_or_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    train, reference = _fixture_datasets()
    original = PrivilegedEvaluator.evaluate_round
    calls = {"count": 0}

    def flaky(self, result):
        calls["count"] += 1
        if calls["count"] > 1:
            raise EvaluatorError("injected late evaluator failure")
        return original(self, result)

    monkeypatch.setattr(PrivilegedEvaluator, "evaluate_round", flaky)
    target = tmp_path / "trace"

    with pytest.raises(EpisodeError, match="evaluation failed"):
        run_from_config(
            load_config(CLEAN_CONFIG),
            dataset=train,
            evaluation_dataset=reference,
            trace_path=target,
        )

    _assert_no_target(tmp_path, target)


def test_late_writer_failure_leaves_no_trace_or_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    train, reference = _fixture_datasets()
    original = TraceWriter.write_round
    calls = {"count": 0}

    def flaky(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] > 1:
            raise TrajectoryError("injected late writer failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TraceWriter, "write_round", flaky)
    target = tmp_path / "trace"

    with pytest.raises(EpisodeError, match="trace emission failed"):
        run_from_config(
            load_config(CLEAN_CONFIG),
            dataset=train,
            evaluation_dataset=reference,
            trace_path=target,
        )

    _assert_no_target(tmp_path, target)


def test_pre_existing_destination_is_left_untouched(tmp_path: Path) -> None:
    train, reference = _fixture_datasets()
    target = tmp_path / "trace"
    target.mkdir()
    (target / "sentinel.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(TrajectoryError, match="already exists"):
        run_from_config(
            load_config(CLEAN_CONFIG),
            dataset=train,
            evaluation_dataset=reference,
            trace_path=target,
        )

    assert (target / "sentinel.txt").read_text(encoding="utf-8") == "keep"
    assert not (target / "manifest.json").exists()
    assert sorted(path.name for path in target.iterdir()) == ["sentinel.txt"]


def test_repeated_cli_publish_is_refused_and_preserves_first_trace(
    tmp_path: Path,
) -> None:
    target = tmp_path / "trace"
    _run_fixture(target)
    before = digest_inventory(target)

    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--config",
            str(CLEAN_CONFIG),
            "--output",
            str(target),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert proc.returncode != 0
    assert "already exists" in proc.stderr
    assert digest_inventory(target) == before
