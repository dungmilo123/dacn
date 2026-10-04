"""Canonical acceptance-audit tests (M003/S05/T07).

These drive the production acceptance CLI over the *reduced injected* matrix
already published by the canonical CLI fixture in ``test_canonical_cli``. The
audit logic is exercised through the same seams the public command uses; no
test reads downloaded MNIST, repository ``runs/``, ``data/`` or ``.gsd``.
"""

from __future__ import annotations

import io
import json
import shutil
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

import dfl_sim.canonical_audit as audit_mod
from dfl_sim.canonical_audit import (
    AUDIT_SUCCESS_MARKER,
    CanonicalAuditDeps,
    CanonicalAuditError,
    _assert_replay_links,
    _assert_scope,
    audit_canonical_study,
)
from dfl_sim.robustness_study import ROBUSTNESS_PROTOCOL_FILENAME
from dfl_sim.robustness_verify import _sha256_json
from test_canonical_cli import _REPORT_DEPS, canonical_cli  # noqa: F401 - fixture

REPO_ROOT = Path(__file__).resolve().parents[1]

_AUDIT_DEPS = CanonicalAuditDeps(
    report_deps=_REPORT_DEPS,
    repo_root=REPO_ROOT,
    allow_injected=True,
)


def _run_capture(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        status = audit_mod.main(argv)
    return status, out.getvalue(), err.getvalue()


def _protocol(canonical_cli) -> dict:
    payload = json.loads(
        (canonical_cli.output / ROBUSTNESS_PROTOCOL_FILENAME).read_text()
    )
    return payload["protocol"]


def _copy(source: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "matrix"
    shutil.copytree(source, destination)
    return destination


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_audit_injected_matrix_passes(canonical_cli) -> None:
    result = audit_canonical_study(str(canonical_cli.output), deps=_AUDIT_DEPS)
    assert result.status == "passed"
    assert result.conditions == 1
    assert result.policies == 6
    assert len(result.audit_digest) == 64
    body = dict(result.audit)
    stored = body.pop("audit_digest")
    assert stored == _sha256_json(body)
    assert result.audit["checks"]["deep_cache_verified"] is False


def test_audit_cli_prints_marker(canonical_cli, monkeypatch) -> None:
    monkeypatch.setattr(audit_mod, "_audit_deps", lambda: _AUDIT_DEPS)
    status, stdout, stderr = _run_capture([str(canonical_cli.output)])
    assert status == 0, stderr
    assert AUDIT_SUCCESS_MARKER in stdout
    assert "conditions=1" in stdout
    assert "policies=6" in stdout


def test_audit_publishes_digest_bound_output(canonical_cli, tmp_path) -> None:
    output = tmp_path / "runtime-audit.json"
    result = audit_canonical_study(
        str(canonical_cli.output), deps=_AUDIT_DEPS, output=str(output)
    )
    assert result.output_path == output
    payload = json.loads(output.read_text())
    assert payload["status"] == "passed"
    body = dict(payload)
    stored = body.pop("audit_digest")
    assert stored == _sha256_json(body)
    assert payload["coverage"]["policy_count"] == 6


# --------------------------------------------------------------------------- #
# Bypassed public provenance gate
# --------------------------------------------------------------------------- #
def test_audit_cli_refuses_injected_without_seam(canonical_cli) -> None:
    status, stdout, stderr = _run_capture([str(canonical_cli.output)])
    assert status == 1
    assert AUDIT_SUCCESS_MARKER not in stdout
    assert "real_mnist" in stderr


def test_audit_missing_root_fails(tmp_path) -> None:
    status, stdout, stderr = _run_capture([str(tmp_path / "absent")])
    assert status == 1
    assert AUDIT_SUCCESS_MARKER not in stdout
    assert "dfl_sim: error:" in stderr


# --------------------------------------------------------------------------- #
# Coverage / identity assertions
# --------------------------------------------------------------------------- #
def test_assert_scope_rejects_duplicate_condition(canonical_cli) -> None:
    report = json.loads(json.dumps(canonical_cli.report))
    protocol = _protocol(canonical_cli)
    report["conditions"].append(dict(report["conditions"][0]))
    with pytest.raises(CanonicalAuditError):
        _assert_scope(report, protocol, real=False)


def test_assert_scope_rejects_seed_overlap(canonical_cli) -> None:
    report = json.loads(json.dumps(canonical_cli.report))
    protocol = dict(_protocol(canonical_cli))
    protocol["selection_seeds"] = list(protocol["final_seeds"])
    with pytest.raises(CanonicalAuditError):
        _assert_scope(report, protocol, real=False)


def test_assert_replay_links_rejects_escape(canonical_cli) -> None:
    report = json.loads(json.dumps(canonical_cli.report))
    condition = report["conditions"][0]
    condition["replay_links"][0]["trace"] = "../../escape"
    with pytest.raises(CanonicalAuditError):
        _assert_replay_links(
            [condition],
            root=canonical_cli.output,
            final_seeds=report["scope"]["final_seeds"],
            policies=report["scope"]["policies"],
            known_ids=("duty_cycle", "periodic"),
            held_out_id="held_out",
        )


def test_audit_detects_tampered_report_copy(canonical_cli, tmp_path) -> None:
    root = _copy(canonical_cli.output, tmp_path)
    path = root / "report" / "canonical-report.json"
    payload = json.loads(path.read_text())
    payload["scope"]["matrix_final_cells"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(CanonicalAuditError):
        audit_canonical_study(str(root), deps=_AUDIT_DEPS)


def test_deep_cache_digest_passes_deep_as_keyword(tmp_path: Path) -> None:
    """Regression (M003/S05/T07): production ``verify_cache`` takes ``deep``
    keyword-only, so the audit must not pass it positionally. The injected
    fixture skips this branch, so assert the convention against a keyword-only
    fake."""

    class _Manifest:
        manifest_digest = "b" * 64

    def fake_verify(data_root, *, deep=True):  # keyword-only, like production
        assert deep is True
        return _Manifest()

    assert audit_mod._deep_cache_digest(tmp_path, fake_verify) == "b" * 64
