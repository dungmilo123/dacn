"""Static canonical figure tests (M003/S05/T06).

Everything here is offline and deterministic. The heavy positive path builds
the reduced **injected** robustness matrix once with the real T04 orchestrator
and compiles it with the real T05 compiler; the published report is then used
both to re-derive figure identity and as a base for in-memory mutations that
exercise escaping, flat series, null denominators and unit separation. No test
reads downloaded MNIST, repository ``runs/``, ``data/`` or ``.gsd``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from dfl_sim.report import (
    CANONICAL_REPORT_FILENAME,
    CanonicalReportDeps,
    compile_canonical_report,
    validate_canonical_report,
)
from dfl_sim.report_figures import (
    FIGURE_SCHEMA,
    FigureError,
    attach_figure_identity,
    render_figures,
)
from dfl_sim.robustness_verify import RobustnessVerifyDeps
from test_robustness_study import (
    BAYESIAN_FIXTURE,
    REPO_ROOT,
    fixture_study,
    run_injected_matrix,
)

EXPECTED_FIGURES = {
    "accuracy_by_condition": "accuracy-by-condition.svg",
    "rejection_acceptance": "rejection-acceptance.svg",
    "placement_propagation_worst_node": "placement-propagation-worst-node.svg",
    "paired_ablation_deltas": "paired-ablation-deltas.svg",
    "compute_vs_communication": "compute-vs-communication.svg",
}

_DEPS = CanonicalReportDeps(
    verify_deps=RobustnessVerifyDeps(study_builder=fixture_study, repo_root=REPO_ROOT),
    allow_injected=True,
    verified_roots=(str(BAYESIAN_FIXTURE.parent),),
    repo_root=REPO_ROOT,
)


@pytest.fixture(scope="module")
def published_report(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("figures-report")
    _result, _config_path, matrix = run_injected_matrix(tmp)
    result = compile_canonical_report(str(matrix), str(matrix / "report"), deps=_DEPS)
    assert result.status == "published"
    report = json.loads((matrix / "report" / CANONICAL_REPORT_FILENAME).read_text())
    return report


def _artifact(artifacts, figure_id: str):
    for item in artifacts:
        if item.figure_id == figure_id:
            return item
    raise AssertionError(f"missing figure {figure_id}")


# --------------------------------------------------------------------------- #
# Identity and digest binding
# --------------------------------------------------------------------------- #
def test_attach_figure_identity_reproduces_published_identity(published_report) -> None:
    semantic = {key: value for key, value in published_report.items() if key != "figures"}
    payload, figures = attach_figure_identity(semantic)
    assert payload["figures"]["schema"] == FIGURE_SCHEMA
    assert payload["report_digest"] == published_report["report_digest"]
    assert {item.figure_id: item.filename for item in figures} == EXPECTED_FIGURES
    for entry in payload["figures"]["entries"]:
        artifact = _artifact(figures, entry["id"])
        assert entry["filename"] == artifact.filename
        assert entry["digest"] == hashlib.sha256(artifact.svg.encode("utf-8")).hexdigest()


def test_attached_payload_revalidates(published_report) -> None:
    semantic = {key: value for key, value in published_report.items() if key != "figures"}
    payload, _figures = attach_figure_identity(semantic)
    assert validate_canonical_report(payload)["report_digest"] == payload["report_digest"]


# --------------------------------------------------------------------------- #
# Determinism, parseability, no external resources
# --------------------------------------------------------------------------- #
def test_figures_are_deterministic_and_parseable(published_report) -> None:
    first = render_figures(published_report)
    second = render_figures(published_report)
    assert [item.digest for item in first] == [item.digest for item in second]
    assert [item.svg for item in first] == [item.svg for item in second]
    for artifact in first:
        ET.fromstring(artifact.svg)  # raises on malformed XML


def test_figures_emit_no_external_resource_or_script(published_report) -> None:
    for artifact in render_figures(published_report):
        svg = artifact.svg
        assert "<script" not in svg
        assert "href" not in svg
        assert "<image" not in svg
        assert "xlink:" not in svg
        assert "url(" not in svg


def test_figures_label_every_declared_policy(published_report) -> None:
    artifacts = render_figures(published_report)
    accuracy = _artifact(artifacts, "accuracy_by_condition")
    for policy in published_report["scope"]["policies"]:
        assert policy in accuracy.svg
    for condition in published_report["scope"]["placements"]:
        assert condition in accuracy.svg


def test_figures_label_eight_policy_panel_without_data(published_report) -> None:
    mutated = copy.deepcopy(published_report)
    mutated["scope"]["policies"] = [
        "local_only",
        "plain_average",
        "fixed_similarity",
        "trust_weighted",
        "balance",
        "bayesian",
        "bayesian_no_belief",
        "bayesian_no_strategic",
    ]
    accuracy = _artifact(render_figures(mutated), "accuracy_by_condition")
    for policy in mutated["scope"]["policies"]:
        assert policy in accuracy.svg
    assert "n/a" in accuracy.svg  # the two extra policies have no audited cell


# --------------------------------------------------------------------------- #
# Escaping, flat series, null denominators, units
# --------------------------------------------------------------------------- #
def test_figures_escape_malicious_labels(published_report) -> None:
    mutated = copy.deepcopy(published_report)
    malicious = "<script>alert('x')</script>"
    mutated["scope"]["policies"] = [malicious, "plain_average"]
    artifacts = render_figures(mutated)
    for artifact in artifacts:
        assert "<script" not in artifact.svg
    accuracy = _artifact(artifacts, "accuracy_by_condition")
    assert "&lt;script&gt;" in accuracy.svg


def test_figures_handle_flat_identical_values(published_report) -> None:
    mutated = copy.deepcopy(published_report)
    for aggregate in mutated["metrics"]["aggregates"]:
        aggregate["mean_accuracy"] = 0.5
        aggregate["std_accuracy"] = 0.0
        aggregate["worst_node_accuracy"] = 0.5
    for delta in mutated["metrics"]["paired_deltas"]:
        delta["mean_delta"] = 0.0
        delta["std_delta"] = 0.0
    for row in mutated["propagation"]["by_condition_policy"]:
        row["reached_node_rounds"] = 0
        row["total_node_rounds"] = 0
    artifacts = render_figures(mutated)
    for artifact in artifacts:
        assert ">nan<" not in artifact.svg.lower()
        assert ">inf<" not in artifact.svg.lower()
        assert ">-inf<" not in artifact.svg.lower()
        ET.fromstring(artifact.svg)


def test_figures_mark_null_denominators_as_na(published_report) -> None:
    mutated = copy.deepcopy(published_report)
    for row in mutated["rejection"]["by_condition_policy"]:
        row["benign_exclusion_rate"] = None
        row["malicious_active_acceptance_rate"] = None
    rejection = _artifact(render_figures(mutated), "rejection_acceptance")
    assert "n/a" in rejection.svg


def test_cost_figures_keep_runtime_and_bytes_on_separate_axes(published_report) -> None:
    costs = _artifact(render_figures(published_report), "compute_vs_communication")
    assert "nanoseconds [ns]" in costs.svg
    assert "bytes [B]" in costs.svg
    assert "measured computation" in costs.svg
    assert "logical communication" in costs.svg
    # Runtime and bytes get distinct series colours/axes.
    assert "#1f77b4" in costs.svg and "#ff7f0e" in costs.svg


def test_figures_disclose_headline_scope_and_held_out(published_report) -> None:
    artifacts = render_figures(published_report)
    accuracy = _artifact(artifacts, "accuracy_by_condition")
    assert "held_out" in accuracy.svg
    assert "bounded IID headline" in accuracy.svg
    assert published_report["data_provenance"] in accuracy.svg
    costs = _artifact(artifacts, "compute_vs_communication")
    assert "unavailable" in costs.svg


# --------------------------------------------------------------------------- #
# Fail-closed malformed input
# --------------------------------------------------------------------------- #
def test_render_rejects_empty_and_malformed_report() -> None:
    with pytest.raises(FigureError):
        render_figures({})
    with pytest.raises(FigureError):
        render_figures({"scope": {}})


def test_render_rejects_nonfinite_scalar(published_report) -> None:
    mutated = copy.deepcopy(published_report)
    mutated["metrics"]["aggregates"][0]["mean_accuracy"] = float("nan")
    with pytest.raises(FigureError):
        render_figures(mutated)
