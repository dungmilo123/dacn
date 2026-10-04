"""Canonical report compiler tests (M003/S05/T05).

Everything here is offline and deterministic. The heavy positive path builds
the reduced **injected** robustness matrix once with the real T04 orchestrator
and compiles it with the real report compiler; every negative case copies that
published tree into ``tmp_path`` and mutates exactly one source. Pure reducers
are exercised with hand-computed unequal seed/node values and synthetic event
records. No test reads downloaded MNIST, repository ``runs/``, ``data/`` or
``.gsd``.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from dfl_sim.bayesian_artifact import load_bayesian_artifact
from dfl_sim.policy import enumerate_candidate_actions
from dfl_sim.report import (
    CANONICAL_REPORT_FILENAME,
    CanonicalReportDeps,
    CanonicalReportError,
    _local_compute_posterior,
    _local_enumerate_candidates,
    _local_expected_payoffs,
    _local_argmax,
    audit_bayesian_decisions,
    build_report_payload,
    compile_canonical_report,
    paired_policy_deltas,
    reduce_final_aggregates,
    reduce_propagation,
    reduce_rejection,
    validate_canonical_report,
    verify_canonical_report,
)
from dfl_sim.robustness_verify import MATRIX_EVIDENCE_FILENAME, RobustnessVerifyDeps
from test_robustness_study import (
    BAYESIAN_FIXTURE,
    REPO_ROOT,
    fixture_study,
    run_injected_matrix,
)

_DEPS = CanonicalReportDeps(
    verify_deps=RobustnessVerifyDeps(study_builder=fixture_study, repo_root=REPO_ROOT),
    allow_injected=True,
    verified_roots=(str(BAYESIAN_FIXTURE.parent),),
    repo_root=REPO_ROOT,
)


# --------------------------------------------------------------------------- #
# Positive path fixture
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def published_report(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("canonical-report")
    _result, _config_path, matrix = run_injected_matrix(tmp)
    result = compile_canonical_report(str(matrix), str(matrix / "report"), deps=_DEPS)
    report = json.loads((matrix / "report" / CANONICAL_REPORT_FILENAME).read_text())
    return matrix, result, report


def _copy_matrix(source: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "matrix"
    shutil.copytree(source, destination)
    return destination


def _final_manifest(root: Path) -> Path:
    return root / "conditions" / "a0p5__mid" / "study" / "final" / "final-manifest.json"


# --------------------------------------------------------------------------- #
# Pure reducers: final aggregates
# --------------------------------------------------------------------------- #
def _cell(policy: str, seed: int, accuracy: float, loss: float, worst: float) -> dict:
    return {
        "condition_id": "c",
        "policy_id": policy,
        "schedule_id": "s",
        "seed": seed,
        "held_out": False,
        "provenance": "injected",
        "metrics": {
            "mean_accuracy": accuracy,
            "mean_loss": loss,
            "worst_node_accuracy": worst,
            "node_count": 3,
        },
    }


def test_reduce_final_aggregates_hand_computed() -> None:
    cells = [
        _cell("a", 1, 0.2, 1.0, 0.1),
        _cell("a", 2, 0.5, 0.8, 0.4),
        _cell("a", 3, 0.8, 0.6, 0.7),
    ]
    aggregates = reduce_final_aggregates(cells)
    assert len(aggregates) == 1
    aggregate = aggregates[0]
    assert aggregate["seeds"] == [1, 2, 3]
    assert aggregate["count"] == 3
    assert aggregate["mean_accuracy"] == pytest.approx(0.5)
    assert aggregate["std_accuracy"] == pytest.approx(math.sqrt(0.06))
    assert aggregate["iqr_accuracy"] == pytest.approx(0.3)
    assert aggregate["mean_loss"] == pytest.approx(0.8)
    assert aggregate["worst_node_accuracy"] == pytest.approx(0.1)
    assert [row["seed"] for row in aggregate["per_seed"]] == [1, 2, 3]


def test_reduce_final_aggregates_rejects_duplicate_seed() -> None:
    with pytest.raises(CanonicalReportError):
        reduce_final_aggregates([_cell("a", 1, 0.2, 1.0, 0.1), _cell("a", 1, 0.5, 1.0, 0.1)])


def test_reduce_final_aggregates_rejects_inconsistent_node_counts() -> None:
    first = _cell("a", 1, 0.2, 1.0, 0.1)
    second = _cell("a", 2, 0.5, 1.0, 0.1)
    second["metrics"]["node_count"] = 4
    with pytest.raises(CanonicalReportError):
        reduce_final_aggregates([first, second])


# --------------------------------------------------------------------------- #
# Pure reducers: paired deltas
# --------------------------------------------------------------------------- #
def test_paired_policy_deltas_use_identical_seed_keys() -> None:
    cells: list[dict] = []
    for seed, (accuracy, reference) in enumerate(((0.2, 0.5), (0.5, 0.5), (0.8, 0.5)), start=1):
        cells.append(_cell("policy", seed, accuracy, 1.0, accuracy))
        cells.append(_cell("plain_average", seed, reference, 1.0, reference))
        cells.append(_cell("bayesian", seed, reference, 1.0, reference))
    aggregates = reduce_final_aggregates(cells)
    deltas = paired_policy_deltas(aggregates)
    plain = [d for d in deltas if d["reference_policy"] == "plain_average" and d["policy_id"] == "policy"]
    assert len(plain) == 1
    per_seed = {row["seed"]: row["delta"] for row in plain[0]["per_seed"]}
    assert per_seed[1] == pytest.approx(-0.3)
    assert per_seed[2] == pytest.approx(0.0)
    assert per_seed[3] == pytest.approx(0.30000000000000004)
    assert plain[0]["mean_delta"] == pytest.approx(0.0)


def test_paired_policy_deltas_reject_unpaired_seed_sets() -> None:
    cells = [
        _cell("policy", 1, 0.2, 1.0, 0.1),
        _cell("policy", 2, 0.5, 1.0, 0.1),
        _cell("plain_average", 1, 0.5, 1.0, 0.1),
    ]
    aggregates = reduce_final_aggregates(cells)
    with pytest.raises(CanonicalReportError):
        paired_policy_deltas(aggregates)


# --------------------------------------------------------------------------- #
# Pure reducers: rejection / acceptance
# --------------------------------------------------------------------------- #
def _node(node_id, true_type, kappa, participants, weights) -> dict:
    return {
        "node": node_id,
        "loss": 0.0,
        "accuracy": 0.5,
        "sample_count": 1,
        "true_type": true_type,
        "attack": {"kind": "delta_reversal" if kappa > 0 else "none", "kappa": kappa},
        "profile": {"participants": participants, "multipliers": [1.0] * len(participants), "weights": weights},
        "propagation": {"reached": False, "min_hop": None, "first_reached_round": None},
    }


def test_reduce_rejection_active_vs_inactive_and_benign() -> None:
    events = [
        {
            "round": 0,
            "nodes": [
                _node(0, "malicious", 1.0, [0], [1.0]),
                _node(1, "benign", 0.0, [1, 0], [0.5, 0.5]),
                _node(2, "benign", 0.0, [2, 0], [1.0, 0.0]),
                _node(3, "benign", 0.0, [3, 4], [0.0, 1.0]),
                _node(4, "malicious", 0.0, [4], [1.0]),
                _node(5, "benign", 0.0, [5, 1], [1.0, 0.0]),
            ]
        }
    ]
    counters = reduce_rejection(events, label="t")
    assert counters["delivered_opportunities"] == 4
    assert counters["benign_sender_opportunities"] == 1
    assert counters["benign_excluded"] == 1
    assert counters["malicious_structural_opportunities"] == 3
    assert counters["malicious_active_opportunities"] == 2
    assert counters["malicious_active_accepted"] == 1
    assert counters["malicious_active_rejected"] == 1
    assert counters["malicious_inactive_opportunities"] == 1
    assert counters["malicious_inactive_accepted"] == 1


def test_reduce_rejection_zero_opportunity_is_null() -> None:
    events = [
        {
            "round": 0,
            "nodes": [
                _node(0, "benign", 0.0, [0], [1.0]),
            ],
        }
    ]
    counters = reduce_rejection(events, label="t")
    assert counters["benign_sender_opportunities"] == 0
    assert counters["malicious_active_opportunities"] == 0
    from dfl_sim.report import _rejection_rates

    rates = _rejection_rates(counters)
    assert rates["benign_exclusion_rate"] is None
    assert rates["malicious_active_rejection_rate"] is None


def test_reduce_rejection_rejects_unknown_sender() -> None:
    events = [
        {
            "round": 0,
            "nodes": [
                _node(1, "benign", 0.0, [1, 9], [1.0, 0.0]),
            ],
        }
    ]
    with pytest.raises(CanonicalReportError):
        reduce_rejection(events, label="t")


# --------------------------------------------------------------------------- #
# Pure reducers: propagation
# --------------------------------------------------------------------------- #
def test_reduce_propagation_semantics() -> None:
    events = [
        {
            "round": 0,
            "nodes": [
                {
                    "node": 0,
                    "propagation": {"reached": True, "min_hop": 0, "first_reached_round": 0},
                },
                {
                    "node": 1,
                    "propagation": {"reached": True, "min_hop": 1, "first_reached_round": 0},
                },
                {
                    "node": 2,
                    "propagation": {"reached": False, "min_hop": None, "first_reached_round": None},
                },
            ],
        },
        {
            "round": 1,
            "nodes": [
                {
                    "node": 0,
                    "propagation": {"reached": True, "min_hop": 0, "first_reached_round": 0},
                },
                {
                    "node": 2,
                    "propagation": {"reached": True, "min_hop": 2, "first_reached_round": 1},
                },
            ],
        },
    ]
    summary = reduce_propagation(events, label="t")
    assert summary["reached_node_rounds"] == 4
    assert summary["total_node_rounds"] == 5
    assert summary["reached_nodes_ever"] == 3
    assert summary["min_hop"] == 0
    assert summary["max_hop"] == 2
    assert summary["first_reached_rounds"] == [0, 1]
    assert {"round", "node", "min_hop", "first_reached_round"} <= set(summary["sources"][0])


# --------------------------------------------------------------------------- #
# Independent Bayesian solution-quality audit
# --------------------------------------------------------------------------- #
def _synthetic_bayesian_record() -> tuple[object, dict]:
    artifact = load_bayesian_artifact(str(BAYESIAN_FIXTURE))
    degree = 2
    evidence = [(0, 0, 0), (0, 0, 0)]
    posterior, reason = _local_compute_posterior(
        artifact, degree=degree, degree_bin="2", progress="mid", evidence_bins=evidence
    )
    assert reason is None
    assert posterior is not None
    actions = _local_enumerate_candidates(
        degree,
        artifact.multiplier_set,
        degree_cap=artifact.degree_cap,
        disagreements=[0.5, 0.5],
        sender_ids=[1, 2],
    )
    values, reason = _local_expected_payoffs(
        artifact,
        degree_bin="2",
        progress="mid",
        evidence_bins=evidence,
        posterior=posterior,
        actions=actions,
    )
    assert reason is None
    assert values is not None
    best = _local_argmax(values, artifact.tie_tolerance)
    record = {
        "schema": "dfl_cell_decision_audit",
        "schema_version": 1,
        "identity": {"phase": "final", "policy_id": "bayesian", "seed": 21},
        "node_id": 3,
        "round_index": 1,
        "degree": degree,
        "sender_ids": [1, 2],
        "disagreement": [0.5, 0.5],
        "relative_norm": [0.1, 0.2],
        "loss_delta": [0.0, 0.0],
        "chosen_multipliers": [1.0, *actions[best]],
        "weights": [1.0, 0.0, 0.0],
        "candidate_count": len(actions),
        "pruned": False,
        "posterior": [
            {"state": list(state), "probability": probability}
            for state, probability in posterior
        ],
        "fallback_reason": None,
        "normalization_mass": None,
        "evidence_bins": [list(row) for row in evidence],
        "selection_ns": 1,
    }
    return artifact, record


def test_bayesian_audit_recomputes_and_agrees() -> None:
    artifact, record = _synthetic_bayesian_record()
    audit = audit_bayesian_decisions(
        artifact, [record], total_rounds=3, policy_id="bayesian", provenance="injected"
    )
    assert audit["applicable"] is True
    assert audit["recomputed_rows"] == 1
    assert audit["posterior_agreement_rows"] == 1
    assert audit["action_agreement_rows"] == 1
    assert audit["action_mismatch_rows"] == 0
    assert audit["fallback_rows"] == 0
    assert audit["min_candidates"] == 9 and audit["max_candidates"] == 9


def test_bayesian_audit_detects_action_mismatch() -> None:
    artifact, record = _synthetic_bayesian_record()
    record["chosen_multipliers"] = [1.0, 1.0, 0.0]
    with pytest.raises(CanonicalReportError):
        audit_bayesian_decisions(
            artifact, [record], total_rounds=3, policy_id="bayesian", provenance="injected"
        )


def test_bayesian_audit_detects_posterior_tamper() -> None:
    artifact, record = _synthetic_bayesian_record()
    record["posterior"][0]["probability"] += 0.1
    with pytest.raises(CanonicalReportError):
        audit_bayesian_decisions(
            artifact, [record], total_rounds=3, policy_id="bayesian", provenance="injected"
        )


def test_bayesian_audit_counts_fallback_and_enforces_self_only() -> None:
    artifact, record = _synthetic_bayesian_record()
    record["candidate_count"] = 0
    record["fallback_reason"] = "missing_prior"
    record["chosen_multipliers"] = [1.0, 0.0, 0.0]
    record["posterior"] = []
    audit = audit_bayesian_decisions(
        artifact, [record], total_rounds=3, policy_id="bayesian", provenance="injected"
    )
    assert audit["fallback_rows"] == 1
    assert audit["recomputed_rows"] == 0

    bad = dict(record)
    bad["chosen_multipliers"] = [1.0, 0.5, 0.0]
    with pytest.raises(CanonicalReportError):
        audit_bayesian_decisions(
            artifact, [bad], total_rounds=3, policy_id="bayesian", provenance="injected"
        )


def test_local_candidate_enumeration_matches_production_and_prunes() -> None:
    disagreements = [0.1, 0.9, 0.5, 0.7, 0.3]
    sender_ids = [1, 2, 3, 4, 5]
    for degree, cap in ((2, 4), (5, 2)):
        local = _local_enumerate_candidates(
            degree,
            (0.0, 0.5, 1.0),
            degree_cap=cap,
            disagreements=disagreements[:degree],
            sender_ids=sender_ids[:degree],
        )
        production = enumerate_candidate_actions(
            degree,
            (0.0, 0.5, 1.0),
            degree_cap=cap,
            disagreements=disagreements[:degree],
            sender_ids=sender_ids[:degree],
        )
        assert tuple(local) == tuple(production)
    assert len(local) == 9  # 3 ** degree_cap, not 3 ** degree


# --------------------------------------------------------------------------- #
# End-to-end positive path
# --------------------------------------------------------------------------- #
def test_published_report_is_strict_and_self_consistent(published_report) -> None:
    _matrix, result, report = published_report
    assert result.status == "published"
    validate_canonical_report(report)
    assert report["schema"] == "dfl_canonical_report"
    assert report["schema_version"] == 1
    assert report["status"] == "complete"
    assert report["data_provenance"] == "injected"
    assert set(report["evidence"]) == {
        "derivation",
        "solution_quality",
        "defense_usefulness",
    }
    assert report["conditions"][0]["condition_id"] == "a0p5__mid"
    assert report["conditions"][0]["final_cells"] == 18
    assert report["conditions"][0]["selection_cells"] == 48
    assert len(report["metrics"]["aggregates"]) == 18
    assert len(report["metrics"]["paired_deltas"]) == 30
    assert report["metrics"]["std_method"] == "population"
    assert report["costs"]["computation"]["clock"] == "perf_counter_ns"
    assert report["costs"]["communication"]["totals"]["directed_messages"] > 0
    assert report["costs"]["calibration_stage_wall"]["available"] is False
    assert report["costs"]["process_rss"]["available"] is False
    assert report["costs"]["headline"]["available"] is False
    assert report["evidence"]["derivation"]["provenance"] == "injected"
    assert report["limitations"]


def test_replay_links_are_relative_and_confined(published_report) -> None:
    matrix, _result, report = published_report
    root = matrix.resolve()
    for condition in report["conditions"]:
        for link in condition["replay_links"]:
            for key in ("trace", "costs", "decision_audit"):
                value = link[key]
                assert not value.startswith("/")
                resolved = (matrix / value).resolve()
                assert resolved == root or str(resolved).startswith(str(root) + "/")
                assert resolved.exists()


def test_compiled_report_verifies(published_report) -> None:
    matrix, _result, report = published_report
    verified = verify_canonical_report(str(matrix), deps=_DEPS)
    assert verified.status == "verified"
    assert verified.report_digest == report["report_digest"]


def test_compile_is_idempotent_when_matched(published_report) -> None:
    matrix, _result, report = published_report
    second = compile_canonical_report(str(matrix), str(matrix / "report"), deps=_DEPS)
    assert second.status in ("unchanged", "published")
    assert second.report_digest == report["report_digest"]


def test_solution_quality_documents_fallback_rows(published_report) -> None:
    _matrix, _result, report = published_report
    policies = {row["policy_id"]: row for row in report["evidence"]["solution_quality"]["policies"]}
    assert policies["bayesian"]["applicable"] is True
    assert policies["bayesian"]["fallback_rows"] > 0
    assert policies["local_only"]["applicable"] is False


def test_rejection_and_propagation_are_separate(published_report) -> None:
    _matrix, _result, report = published_report
    rejection = report["rejection"]["totals"]
    assert rejection["benign_sender_opportunities"] > 0
    assert rejection["malicious_active_opportunities"] > 0
    assert rejection["benign_exclusion_rate"] is not None
    propagation = report["propagation"]["totals"]
    assert propagation["reached_node_rounds"] > 0
    assert propagation["total_node_rounds"] >= propagation["reached_node_rounds"]


# --------------------------------------------------------------------------- #
# Fail-closed negative paths
# --------------------------------------------------------------------------- #
def test_public_path_refuses_injected(published_report, tmp_path: Path) -> None:
    matrix, _result, _report = published_report
    with pytest.raises(CanonicalReportError):
        compile_canonical_report(
            str(matrix),
            str(tmp_path / "report"),
            deps=CanonicalReportDeps(
                verify_deps=_DEPS.verify_deps,
                verified_roots=_DEPS.verified_roots,
                repo_root=REPO_ROOT,
            ),
        )


def test_missing_matrix_evidence_fails(published_report, tmp_path: Path) -> None:
    matrix, _result, _report = published_report
    root = _copy_matrix(matrix, tmp_path)
    (root / MATRIX_EVIDENCE_FILENAME).unlink()
    with pytest.raises(CanonicalReportError):
        build_report_payload(str(root), deps=_DEPS)


def test_tampered_matrix_evidence_fails(published_report, tmp_path: Path) -> None:
    matrix, _result, _report = published_report
    root = _copy_matrix(matrix, tmp_path)
    path = root / MATRIX_EVIDENCE_FILENAME
    payload = json.loads(path.read_text())
    payload["status"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(CanonicalReportError):
        build_report_payload(str(root), deps=_DEPS)


def test_tampered_trace_fails(published_report, tmp_path: Path) -> None:
    matrix, _result, _report = published_report
    root = _copy_matrix(matrix, tmp_path)
    trace_manifests = sorted((root / "conditions").rglob("manifest.json"))
    assert trace_manifests
    target = trace_manifests[0]
    payload = json.loads(target.read_text())
    payload["tampered"] = True
    target.write_text(json.dumps(payload))
    with pytest.raises(CanonicalReportError):
        build_report_payload(str(root), deps=_DEPS)


def test_tampered_embedded_metrics_fails(published_report, tmp_path: Path) -> None:
    matrix, _result, _report = published_report
    root = _copy_matrix(matrix, tmp_path)
    path = _final_manifest(root)
    payload = json.loads(path.read_text())
    payload["entries"][0]["metrics"]["mean_accuracy"] += 0.25
    path.write_text(json.dumps(payload))
    with pytest.raises(CanonicalReportError):
        build_report_payload(str(root), deps=_DEPS)


def test_path_escape_fails(published_report, tmp_path: Path) -> None:
    matrix, _result, _report = published_report
    root = _copy_matrix(matrix, tmp_path)
    path = _final_manifest(root)
    payload = json.loads(path.read_text())
    payload["entries"][0]["trace_path"] = "../../escape"
    path.write_text(json.dumps(payload))
    with pytest.raises(CanonicalReportError):
        build_report_payload(str(root), deps=_DEPS)


def test_artifact_outside_verified_roots_fails(published_report, tmp_path: Path) -> None:
    matrix, _result, _report = published_report
    deps = CanonicalReportDeps(
        verify_deps=_DEPS.verify_deps,
        allow_injected=True,
        verified_roots=(),
        repo_root=REPO_ROOT,
    )
    with pytest.raises(CanonicalReportError):
        build_report_payload(str(matrix), deps=deps)


def test_tampered_report_fails_verify(published_report, tmp_path: Path) -> None:
    matrix, _result, _report = published_report
    root = _copy_matrix(matrix, tmp_path)
    path = root / "report" / CANONICAL_REPORT_FILENAME
    payload = json.loads(path.read_text())
    payload["scope"]["matrix_final_cells"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(CanonicalReportError):
        verify_canonical_report(str(root), deps=_DEPS)


def test_missing_report_fails_verify(published_report, tmp_path: Path) -> None:
    matrix, _result, _report = published_report
    root = _copy_matrix(matrix, tmp_path)
    (root / "report" / CANONICAL_REPORT_FILENAME).unlink()
    with pytest.raises(CanonicalReportError):
        verify_canonical_report(str(root), deps=_DEPS)


def test_validate_rejects_nonfinite(published_report) -> None:
    _matrix, _result, report = published_report
    mutated = json.loads(json.dumps(report))
    mutated["metrics"]["aggregates"][0]["mean_accuracy"] = float("nan")
    with pytest.raises(CanonicalReportError):
        validate_canonical_report(mutated)
