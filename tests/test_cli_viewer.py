"""CLI tests for ``dfl view`` (S05/T02, R011, D020).

Only the final server run boundary (``dash.Dash.run``) is monkeypatched: the
parser, port validation, trace projection, and error mapping all run for real.
This proves the command validates the trace and port *before* touching the
network, fixes loopback/debug/reloader settings, and leaves ``dfl run`` and the
legacy ``python -m dfl_sim`` invocation untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import dash
import pytest

from dfl_sim.__main__ import VIEWER_DEFAULT_PORT, VIEWER_HOST, main

from helpers import make_viewer_trace


@pytest.fixture
def viewer_trace(tmp_path: Path) -> Path:
    return make_viewer_trace(tmp_path / "viewer_trace")


@pytest.fixture
def recorded_runs(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture the arguments passed to the final ``app.run`` boundary."""

    calls: list[dict] = []

    def fake_run(self, **kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(dash.Dash, "run", fake_run)
    return calls


# --------------------------------------------------------------------------- #
# Happy path and fixed settings
# --------------------------------------------------------------------------- #
def test_view_launches_loopback_server(
    viewer_trace: Path, recorded_runs: list[dict], capsys
) -> None:
    status = main(["view", str(viewer_trace), "--port", "8123"])

    assert status == 0
    assert len(recorded_runs) == 1
    kwargs = recorded_runs[0]
    assert kwargs == {
        "host": VIEWER_HOST,
        "port": 8123,
        "debug": False,
        "use_reloader": False,
    }
    assert VIEWER_HOST == "127.0.0.1"

    out = capsys.readouterr().out
    assert "http://127.0.0.1:8123/" in out
    assert "read-only" in out


def test_view_defaults_to_documented_port(
    viewer_trace: Path, recorded_runs: list[dict]
) -> None:
    assert VIEWER_DEFAULT_PORT == 8050
    assert main(["view", str(viewer_trace)]) == 0
    assert recorded_runs[0]["port"] == 8050


def test_view_help_lists_trace_and_port(capsys) -> None:
    with pytest.raises(SystemExit) as info:
        main(["view", "--help"])

    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "TRACE_DIR" in out
    assert "--port" in out
    assert "read-only" in out


# --------------------------------------------------------------------------- #
# Port validation stops before startup
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("port", [0, -1, 65536, 99999])
def test_view_rejects_invalid_ports(
    port: int, viewer_trace: Path, recorded_runs: list[dict], capsys
) -> None:
    status = main(["view", str(viewer_trace), "--port", str(port)])

    assert status == 2
    assert recorded_runs == []
    err = capsys.readouterr().err
    assert "port must be between 1 and 65535" in err
    assert "Traceback" not in err


def test_view_non_numeric_port_is_usage_error(viewer_trace: Path, recorded_runs: list[dict]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["view", str(viewer_trace), "--port", "eight"])

    assert info.value.code == 2
    assert recorded_runs == []


# --------------------------------------------------------------------------- #
# Trace validation stops before startup
# --------------------------------------------------------------------------- #
def test_view_missing_trace_fails_before_server(
    tmp_path: Path, recorded_runs: list[dict], capsys
) -> None:
    status = main(["view", str(tmp_path / "missing")])

    assert status == 1
    assert recorded_runs == []
    err = capsys.readouterr().err
    assert "missing" in err
    assert "Traceback" not in err
    assert str(tmp_path) not in err  # path-safe: no absolute parent leak


def test_view_unsupported_schema_fails_before_server(
    viewer_trace: Path, recorded_runs: list[dict], capsys
) -> None:
    manifest_path = viewer_trace / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 99
    manifest_path.write_bytes(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )

    status = main(["view", str(viewer_trace)])

    assert status == 1
    assert recorded_runs == []
    err = capsys.readouterr().err
    assert "unsupported trajectory schema" in err
    assert "Traceback" not in err


# --------------------------------------------------------------------------- #
# Server startup failure is reported cleanly
# --------------------------------------------------------------------------- #
def test_view_bind_failure_is_reported_cleanly(
    monkeypatch: pytest.MonkeyPatch, viewer_trace: Path, capsys
) -> None:
    def boom(self, **kwargs):
        raise OSError("address already in use")

    monkeypatch.setattr(dash.Dash, "run", boom)

    status = main(["view", str(viewer_trace), "--port", "8123"])

    assert status == 1
    err = capsys.readouterr().err
    assert "address already in use" in err
    assert "Traceback" not in err


# --------------------------------------------------------------------------- #
# Preserved run/legacy dispatch
# --------------------------------------------------------------------------- #
def test_run_and_legacy_dispatch_are_unchanged(capsys) -> None:
    with pytest.raises(SystemExit) as run_info:
        main(["run", "--help"])
    assert run_info.value.code == 0
    assert "--output" in capsys.readouterr().out

    with pytest.raises(SystemExit) as legacy_info:
        main(["--help"])
    assert legacy_info.value.code == 0
    out = capsys.readouterr().out
    assert "--log" in out
    assert "run CONFIG --output" in out
