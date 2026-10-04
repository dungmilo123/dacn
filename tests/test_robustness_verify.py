"""Independent robustness matrix verifier tests (M003/S05/T04).

The positive path builds the reduced injected matrix once with the real
orchestrator and then re-audits it with the real verifier. Every negative case
copies that published tree into ``tmp_path`` and mutates exactly one source,
proving the verifier fails closed on tampered provenance, missing cells,
tampered sidecars/traces and unsafe paths. No test reads downloaded MNIST,
repository ``runs/``, ``data/`` or ``.gsd``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dfl_sim.robustness_study import (
    MATRIX_MANIFEST_FILENAME,
    ROBUSTNESS_PROTOCOL_FILENAME,
)
from dfl_sim.robustness_verify import (
    MATRIX_EVIDENCE_FILENAME,
    RobustnessVerifyDeps,
    RobustnessVerifyError,
    _confined,
    _verify_cache,
    verify_robustness_study,
)
from test_robustness_study import REPO_ROOT, fixture_study, run_injected_matrix

#: The injected fixture matrix is only reconstructible with the same fixture
#: study seam, so every verification call must forward it (production forwards
#: the real ``build_condition_study`` instead).
_DEPS = RobustnessVerifyDeps(study_builder=fixture_study, repo_root=REPO_ROOT)


def _verify(root: Path) -> object:
    return verify_robustness_study(str(root), deps=_DEPS, write=False)


@pytest.fixture(scope="module")
def published_matrix(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("published-matrix")
    _result, _config_path, output = run_injected_matrix(tmp)
    return output


def _copy_matrix(source: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "matrix"
    shutil.copytree(source, destination)
    return destination


def _condition_study(root: Path) -> Path:
    return root / "conditions" / "a0p5__mid" / "study"


def _condition_root(root: Path) -> Path:
    return root / "conditions" / "a0p5__mid"


# --------------------------------------------------------------------------- #
# Confinement
# --------------------------------------------------------------------------- #
def test_confined_rejects_absolute_and_traversal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(RobustnessVerifyError):
        _confined(root, "/etc/passwd", label="trace")
    with pytest.raises(RobustnessVerifyError):
        _confined(root, "../escape", label="trace")
    with pytest.raises(RobustnessVerifyError):
        _confined(root, "", label="trace")


def test_confined_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    try:
        os.symlink(outside, root / "link")
    except OSError:  # pragma: no cover - platform without symlinks
        pytest.skip("symlinks unavailable")
    with pytest.raises(RobustnessVerifyError):
        _confined(root, "link/secret.txt", label="trace")


# --------------------------------------------------------------------------- #
# Positive path
# --------------------------------------------------------------------------- #
def test_verify_publishes_evidence(published_matrix: Path) -> None:
    result = verify_robustness_study(str(published_matrix), deps=_DEPS, write=True)
    assert result.status == "published"
    assert result.conditions == 1
    assert result.selection_cells > 0 and result.final_cells > 0
    evidence = json.loads((published_matrix / MATRIX_EVIDENCE_FILENAME).read_text())
    assert evidence["status"] == "verified"
    assert evidence["conditions"][0]["condition_id"] == "a0p5__mid"
    assert evidence["cache"] == {"data_provenance": "injected"}


# --------------------------------------------------------------------------- #
# Negative paths
# --------------------------------------------------------------------------- #
def test_verify_missing_protocol_fails(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(RobustnessVerifyError):
        verify_robustness_study(str(root), deps=_DEPS, write=False)


def test_verify_rejects_tampered_protocol_digest(
    published_matrix: Path, tmp_path: Path
) -> None:
    root = _copy_matrix(published_matrix, tmp_path)
    path = root / ROBUSTNESS_PROTOCOL_FILENAME
    payload = json.loads(path.read_text())
    payload["protocol_digest"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(RobustnessVerifyError):
        verify_robustness_study(str(root), deps=_DEPS, write=False)


def test_verify_rejects_tampered_matrix_manifest(
    published_matrix: Path, tmp_path: Path
) -> None:
    root = _copy_matrix(published_matrix, tmp_path)
    path = root / MATRIX_MANIFEST_FILENAME
    payload = json.loads(path.read_text())
    payload["totals"]["final_cells"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(RobustnessVerifyError):
        verify_robustness_study(str(root), deps=_DEPS, write=False)


def test_verify_rejects_tampered_frozen_study(
    published_matrix: Path, tmp_path: Path
) -> None:
    root = _copy_matrix(published_matrix, tmp_path)
    frozen = _condition_root(root) / "frozen-study.yaml"
    frozen.write_text(frozen.read_text() + "\n# tampered\n")
    with pytest.raises(RobustnessVerifyError):
        verify_robustness_study(str(root), deps=_DEPS, write=False)


def test_verify_rejects_missing_final_cell(
    published_matrix: Path, tmp_path: Path
) -> None:
    root = _copy_matrix(published_matrix, tmp_path)
    path = _condition_study(root) / "final" / "final-manifest.json"
    payload = json.loads(path.read_text())
    payload["entries"] = payload["entries"][:-1]
    path.write_text(json.dumps(payload))
    with pytest.raises(RobustnessVerifyError):
        verify_robustness_study(str(root), deps=_DEPS, write=False)


def test_verify_rejects_duplicate_final_cell(
    published_matrix: Path, tmp_path: Path
) -> None:
    root = _copy_matrix(published_matrix, tmp_path)
    path = _condition_study(root) / "final" / "final-manifest.json"
    payload = json.loads(path.read_text())
    payload["entries"] = payload["entries"] + [dict(payload["entries"][0])]
    path.write_text(json.dumps(payload))
    with pytest.raises(RobustnessVerifyError):
        verify_robustness_study(str(root), deps=_DEPS, write=False)


def test_verify_rejects_tampered_sidecar(
    published_matrix: Path, tmp_path: Path
) -> None:
    root = _copy_matrix(published_matrix, tmp_path)
    sidecars = sorted(_condition_study(root).rglob("costs.json"))
    assert sidecars, "published matrix must contain cost sidecars"
    with open(sidecars[0], "ab") as handle:
        handle.write(b" ")
    with pytest.raises(RobustnessVerifyError):
        verify_robustness_study(str(root), deps=_DEPS, write=False)


def test_verify_rejects_tampered_trace(
    published_matrix: Path, tmp_path: Path
) -> None:
    root = _copy_matrix(published_matrix, tmp_path)
    manifests = sorted(_condition_study(root).rglob("manifest.json"))
    assert manifests, "published matrix must contain trace manifests"
    target = manifests[0]
    payload = json.loads(target.read_text())
    payload["tampered"] = True
    target.write_text(json.dumps(payload))
    with pytest.raises(RobustnessVerifyError):
        verify_robustness_study(str(root), deps=_DEPS, write=False)


def test_verify_rejects_foreign_trace_path(
    published_matrix: Path, tmp_path: Path
) -> None:
    root = _copy_matrix(published_matrix, tmp_path)
    path = _condition_study(root) / "final" / "final-manifest.json"
    payload = json.loads(path.read_text())
    payload["entries"][0]["trace_path"] = "../../../../etc/passwd"
    path.write_text(json.dumps(payload))
    with pytest.raises(RobustnessVerifyError):
        verify_robustness_study(str(root), deps=_DEPS, write=False)


# --------------------------------------------------------------------------- #
# Real-MNIST cache call convention
# --------------------------------------------------------------------------- #
def test_verify_cache_passes_deep_as_keyword(tmp_path: Path) -> None:
    """Regression (M003/S05/T07): production ``verify_cache`` is
    ``verify_cache(data_root, *, deep=...)``. Passing ``deep`` positionally
    raised ``TypeError`` and failed the real matrix audit closed only after all
    nine conditions had completed. The injected fixture never reaches this
    branch, so assert the call convention directly against a keyword-only
    fake."""

    class _Manifest:
        manifest_digest = "a" * 64

    seen: dict = {}

    def fake_verify(data_root, *, deep=True):  # keyword-only, like production
        seen["data_root"] = data_root
        seen["deep"] = deep
        return _Manifest()

    protocol = SimpleNamespace(data_provenance="real_mnist", data_root=str(tmp_path))
    frozen = {"cache": {"data_root": str(tmp_path), "manifest_digest": "a" * 64}}
    deps = RobustnessVerifyDeps(cache_verify=fake_verify)

    result = _verify_cache(
        frozen=frozen,
        protocol=protocol,
        deps=deps,
        repo_root=tmp_path,
        deep=True,
    )

    assert seen == {"data_root": str(tmp_path), "deep": True}
    assert result["manifest_digest"] == "a" * 64
