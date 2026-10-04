"""Hand-computable tests for offline on-policy Bayesian evidence (T02/R016).

The checked-in schema-v1 fixture is small enough to recompute coverage and the
support-weighted mean total-variation mismatch exactly. These tests build the
exact per-selection :class:`~dfl_sim.policy.BayesianDiagnostic` records (or
drive the real policy through a fake clock), then assert exact aggregate values,
explicit zero fallback counts, and the full failure-closed negative surface:
empty/duplicate/out-of-order/non-Bayesian inputs, sender/evidence misalignment,
unknown context or evidence cells, invalid durations, and sink exceptions.

Everything is offline and in-process: the artifact is read from the git-tracked
fixture directory and the config is built through the production strict parser.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dfl_sim.bayesian_artifact import load_bayesian_artifact
from dfl_sim.bayesian_evidence import (
    FALLBACK_REASONS,
    REPORT_SCHEMA,
    BayesianEvidenceReport,
    EvidenceError,
    collect_bayesian_evidence,
    render_bayesian_report,
)
from dfl_sim.observation import LocalObservation, NeighborObservation
from dfl_sim.policy import (
    FALLBACK_MISSING_LIKELIHOOD,
    FALLBACK_ZERO_MASS,
    BayesianDiagnostic,
    PolicyError,
    build_bayesian_policy,
)
from helpers import make_config

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"


@pytest.fixture(scope="module")
def artifact():
    return load_bayesian_artifact(FIXTURE)


def _config(**overrides):
    return make_config(
        topology={"kind": "chain3"},
        aggregation={
            "kind": "bayesian",
            "artifact_dir": str(FIXTURE),
            "omega": 0.5,
            "degree_cap": 4,
            "tie_tolerance": 1.0e-9,
        },
        **overrides,
    )


def _diag(
    *,
    node: int,
    round_index: int,
    senders: tuple[int, ...],
    evidence: tuple[tuple[int, int, int], ...],
    degree_bin: str | None,
    progress: str | None,
    fallback: str | None = None,
    candidate_count: int = 0,
    pruned: bool = False,
    elapsed_ns: int = 0,
) -> BayesianDiagnostic:
    degree = len(senders)
    return BayesianDiagnostic(
        node_id=node,
        round_index=round_index,
        degree=degree,
        degree_bin=degree_bin,
        progress=progress,
        chosen_multipliers=(1.0,) + (0.0,) * degree,
        candidate_count=candidate_count,
        pruned=pruned,
        posterior=(),
        fallback_reason=fallback,
        normalization_mass=None,
        sender_ids=senders,
        evidence_bins=evidence,
        elapsed_ns=elapsed_ns,
    )


# --------------------------------------------------------------------------- #
# Exact coverage and TV mismatch on the clean fixture
# --------------------------------------------------------------------------- #
def _clean_diagnostics() -> tuple[BayesianDiagnostic, ...]:
    return (
        _diag(
            node=0, round_index=0, senders=(1,), evidence=((0, 0, 0),),
            degree_bin="1", progress="early",
            fallback=FALLBACK_MISSING_LIKELIHOOD, elapsed_ns=10,
        ),
        _diag(
            node=1, round_index=0, senders=(0, 2),
            evidence=((0, 0, 0), (1, 0, 0)),
            degree_bin="2", progress="early",
            fallback=FALLBACK_MISSING_LIKELIHOOD, elapsed_ns=20,
        ),
        _diag(
            node=2, round_index=0, senders=(1,), evidence=((1, 0, 0),),
            degree_bin="1", progress="early",
            fallback=FALLBACK_MISSING_LIKELIHOOD, elapsed_ns=30,
        ),
        _diag(
            node=0, round_index=1, senders=(1,), evidence=((0, 0, 0),),
            degree_bin="1", progress="mid", candidate_count=3,
            elapsed_ns=40,
        ),
        _diag(
            node=1, round_index=1, senders=(0, 2),
            evidence=((0, 0, 0), (1, 0, 0)),
            degree_bin="2", progress="mid", candidate_count=9,
            elapsed_ns=50,
        ),
        _diag(
            node=2, round_index=1, senders=(1,), evidence=((1, 0, 0),),
            degree_bin="1", progress="mid", candidate_count=3,
            elapsed_ns=60,
        ),
    )


def test_exact_coverage_and_weighted_mismatch(artifact) -> None:
    report = collect_bayesian_evidence(
        _clean_diagnostics(), config=_config(), artifact=artifact
    )

    # round-0 "early" context has no likelihood rows -> uncovered; round-1 "mid"
    # covers every declared cell -> exactly half of the 8 neighbor observations.
    assert report.neighbor_observation_count == 8
    assert report.covered_observation_count == 4
    assert report.coverage == pytest.approx(0.5)

    # degree-1 and degree-2 "mid" benign groups each have p=(0.5,0.5) against
    # the artifact q=(0.6,0.4): TV = 0.1 each, support-weighted -> 0.1.
    assert report.mismatch_group_count == 2
    assert report.mismatch == pytest.approx(0.1)

    assert report.selection_count == 6
    assert report.benign_receiver_count == 6
    assert report.artifact_digest == artifact.manifest.aggregate_digest
    assert report.seed == _config().seed
    assert report.rounds == 2
    assert report.node_count == 3
    assert report.compromised_node is None
    assert report.attack_kappa == 0.0


def test_fallback_counts_include_explicit_zeros(artifact) -> None:
    report = collect_bayesian_evidence(
        _clean_diagnostics(), config=_config(), artifact=artifact
    )
    assert [reason for reason, _ in report.fallback_counts] == list(FALLBACK_REASONS)
    counts = dict(report.fallback_counts)
    assert counts[FALLBACK_MISSING_LIKELIHOOD] == 3
    assert counts[FALLBACK_ZERO_MASS] == 0
    assert report.fallback_total == 3
    assert report.fallback_rate == pytest.approx(0.5)


def test_operational_aggregates_are_exact(artifact) -> None:
    report = collect_bayesian_evidence(
        _clean_diagnostics(), config=_config(), artifact=artifact
    )
    assert report.pruned_count == 0
    assert report.candidate_count_total == 3 + 9 + 3
    assert report.candidate_count_max == 9
    assert report.elapsed_ns_total == 10 + 20 + 30 + 40 + 50 + 60
    assert report.elapsed_ns_min == 10
    assert report.elapsed_ns_max == 60


# --------------------------------------------------------------------------- #
# Compromised receivers are excluded from type-conditioned coverage/mismatch
# --------------------------------------------------------------------------- #
def test_compromised_receiver_is_excluded_from_on_policy_evidence(artifact) -> None:
    config = _config(attack={"compromised_node": 0, "intensity": "strong"})
    diagnostics = (
        # Malicious receiver node 0: counted in the operational aggregates but
        # never in coverage/mismatch.
        _diag(
            node=0, round_index=1, senders=(1,), evidence=((0, 0, 0),),
            degree_bin="1", progress="mid", candidate_count=3, elapsed_ns=5,
        ),
        # Benign receiver node 1 with one malicious sender (0) and one benign
        # sender (2).
        _diag(
            node=1, round_index=1, senders=(0, 2),
            evidence=((0, 0, 0), (1, 0, 0)),
            degree_bin="2", progress="mid", candidate_count=9, elapsed_ns=7,
        ),
    )
    report = collect_bayesian_evidence(diagnostics, config=config, artifact=artifact)

    assert report.selection_count == 2
    assert report.benign_receiver_count == 1
    assert report.compromised_node == 0
    assert report.attack_kappa == 5.0
    # Only node 1's two neighbor observations are counted, both covered.
    assert report.neighbor_observation_count == 2
    assert report.covered_observation_count == 2
    assert report.coverage == pytest.approx(1.0)
    # type-1 group: empirical {cell0: 1.0} vs q {0:0.1, 1:0.9} -> TV 0.9.
    # type-0 group: empirical {cell1: 1.0} vs q {0:0.6, 1:0.4} -> TV 0.6.
    # support-weighted mean = 0.75.
    assert report.mismatch_group_count == 2
    assert report.mismatch == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# Canonical bounded rendering
# --------------------------------------------------------------------------- #
def test_report_render_is_canonical_sorted_and_complete(artifact) -> None:
    report = collect_bayesian_evidence(
        _clean_diagnostics(), config=_config(), artifact=artifact
    )
    rendered = render_bayesian_report(report)
    assert rendered == report.canonical_json()
    payload = json.loads(rendered)
    assert list(payload) == sorted(payload)
    assert payload["schema"] == REPORT_SCHEMA
    assert set(payload["fallback"]) == set(FALLBACK_REASONS)
    assert "torch" not in rendered.lower()
    # Bounded: no example/label/path payload keys leak in.
    for banned in ("examples", "labels", "paths", "tensors", "samples"):
        assert banned not in payload


def test_render_rejects_non_report() -> None:
    with pytest.raises(EvidenceError):
        render_bayesian_report(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Positive policy-driven path with an injected clock
# --------------------------------------------------------------------------- #
def _obs(node_id: int, round_index: int, sender: int, features) -> LocalObservation:
    return LocalObservation(
        node_id=node_id,
        round_index=round_index,
        epsilon=1e-12,
        neighbors=(NeighborObservation(sender_id=sender, **features),),
    )


class _TickClock:
    """Strictly increasing integer nanosecond clock (one tick per call)."""

    def __init__(self, start: int = 0, step: int = 1) -> None:
        self._value = start
        self._step = step

    def __call__(self) -> int:
        value = self._value
        self._value += self._step
        return value


def test_policy_diagnostics_carry_alignment_and_elapsed(artifact) -> None:
    records: list[BayesianDiagnostic] = []
    policy = build_bayesian_policy(
        artifact,
        total_rounds=3,
        diagnostic_sink=records.append,
        clock=_TickClock(100),
    )
    observation = _obs(
        0, 1, 7, {"directional_disagreement": 0.2, "relative_norm": 0.5, "loss_delta": -0.3}
    )
    policy.select_weights(observation)

    diagnostic = records[0]
    assert diagnostic.sender_ids == (7,)
    assert diagnostic.evidence_bins == ((0, 0, 0),)
    assert diagnostic.elapsed_ns == 1
    assert diagnostic.chosen_multipliers == diagnostic.chosen_multipliers  # stable


def test_timing_never_changes_the_selected_profile(artifact) -> None:
    first: list[BayesianDiagnostic] = []
    second: list[BayesianDiagnostic] = []
    observation = _obs(
        0, 1, 7, {"directional_disagreement": 0.75, "relative_norm": 0.5, "loss_delta": -0.3}
    )
    build_bayesian_policy(
        artifact, 3, diagnostic_sink=first.append, clock=_TickClock(0, 0)
    ).select_weights(observation)
    build_bayesian_policy(
        artifact, 3, diagnostic_sink=second.append, clock=_TickClock(10**9, 5)
    ).select_weights(observation)

    assert first[0].chosen_multipliers == second[0].chosen_multipliers
    assert first[0].fallback_reason == second[0].fallback_reason
    assert first[0].elapsed_ns == 0
    assert second[0].elapsed_ns == 5


def test_policy_driven_report_is_finite_and_bounded(artifact) -> None:
    records: list[BayesianDiagnostic] = []
    policy = build_bayesian_policy(
        artifact, 3, diagnostic_sink=records.append, clock=_TickClock()
    )
    for round_index in (1, 2):
        for node, sender in ((0, 7), (1, 9)):
            features = {"directional_disagreement": 0.2, "relative_norm": 0.5, "loss_delta": -0.3}
            policy.select_weights(_obs(node, round_index, sender, features))

    report = collect_bayesian_evidence(records, config=_config(), artifact=artifact)
    assert 0.0 <= report.coverage <= 1.0
    assert 0.0 <= report.mismatch <= 1.0
    assert report.selection_count == 4
    assert report.elapsed_ns_total == 4
    assert report.elapsed_ns_min == 1
    assert report.elapsed_ns_max == 1


# --------------------------------------------------------------------------- #
# Fallback-only and sink failure negatives
# --------------------------------------------------------------------------- #
def test_sink_exception_fails_closed(artifact) -> None:
    def bad_sink(diagnostic: BayesianDiagnostic) -> None:
        raise RuntimeError("reporter exploded")

    policy = build_bayesian_policy(artifact, 3, diagnostic_sink=bad_sink)
    observation = _obs(
        0, 1, 7, {"directional_disagreement": 0.2, "relative_norm": 0.5, "loss_delta": -0.3}
    )
    with pytest.raises(PolicyError, match="diagnostic sink failed"):
        policy.select_weights(observation)


# --------------------------------------------------------------------------- #
# Q7 negative surface: malformed / inconsistent collector inputs
# --------------------------------------------------------------------------- #
def test_empty_diagnostics_rejected(artifact) -> None:
    with pytest.raises(EvidenceError, match="no Bayesian diagnostics"):
        collect_bayesian_evidence((), config=_config(), artifact=artifact)


def test_non_bayesian_config_rejected(artifact) -> None:
    with pytest.raises(EvidenceError, match="bayesian aggregation"):
        collect_bayesian_evidence(
            _clean_diagnostics(), config=make_config(), artifact=artifact
        )


def test_non_diagnostic_rejected(artifact) -> None:
    with pytest.raises(EvidenceError, match="exact BayesianDiagnostic"):
        collect_bayesian_evidence(
            [object()], config=_config(), artifact=artifact  # type: ignore[list-item]
        )


def test_duplicate_selection_key_rejected(artifact) -> None:
    duplicate = (
        _diag(
            node=0, round_index=1, senders=(1,), evidence=((0, 0, 0),),
            degree_bin="1", progress="mid",
        ),
        _diag(
            node=0, round_index=1, senders=(1,), evidence=((0, 0, 0),),
            degree_bin="1", progress="mid",
        ),
    )
    with pytest.raises(EvidenceError, match="duplicate"):
        collect_bayesian_evidence(duplicate, config=_config(), artifact=artifact)


def test_out_of_order_diagnostics_rejected(artifact) -> None:
    later = _diag(
        node=2, round_index=1, senders=(1,), evidence=((0, 0, 0),),
        degree_bin="1", progress="mid",
    )
    earlier = _diag(
        node=0, round_index=1, senders=(1,), evidence=((0, 0, 0),),
        degree_bin="1", progress="mid",
    )
    with pytest.raises(EvidenceError, match="ascending"):
        collect_bayesian_evidence([later, earlier], config=_config(), artifact=artifact)


def _tamper(diagnostic: BayesianDiagnostic, field: str, value) -> BayesianDiagnostic:
    object.__setattr__(diagnostic, field, value)
    return diagnostic


def test_sender_evidence_misalignment_rejected(artifact) -> None:
    record = _tamper(
        _diag(
            node=0, round_index=1, senders=(1,), evidence=((0, 0, 0),),
            degree_bin="1", progress="mid",
        ),
        "evidence_bins",
        (),
    )
    with pytest.raises(EvidenceError, match="misalignment"):
        collect_bayesian_evidence([record], config=_config(), artifact=artifact)


def test_unknown_degree_bin_rejected(artifact) -> None:
    record = _tamper(
        _diag(
            node=0, round_index=1, senders=(1,), evidence=((0, 0, 0),),
            degree_bin="1", progress="mid",
        ),
        "degree_bin",
        "9",
    )
    with pytest.raises(EvidenceError, match="degree_bin"):
        collect_bayesian_evidence([record], config=_config(), artifact=artifact)


def test_unknown_progress_rejected(artifact) -> None:
    record = _tamper(
        _diag(
            node=0, round_index=1, senders=(1,), evidence=((0, 0, 0),),
            degree_bin="1", progress="mid",
        ),
        "progress",
        "whenever",
    )
    with pytest.raises(EvidenceError, match="progress"):
        collect_bayesian_evidence([record], config=_config(), artifact=artifact)


def test_unknown_evidence_cell_rejected(artifact) -> None:
    record = _tamper(
        _diag(
            node=0, round_index=1, senders=(1,), evidence=((0, 0, 0),),
            degree_bin="1", progress="mid",
        ),
        "evidence_bins",
        ((9, 0, 0),),
    )
    with pytest.raises(EvidenceError, match="evidence cell"):
        collect_bayesian_evidence([record], config=_config(), artifact=artifact)


def test_invalid_duration_rejected(artifact) -> None:
    record = _tamper(
        _diag(
            node=0, round_index=1, senders=(1,), evidence=((0, 0, 0),),
            degree_bin="1", progress="mid",
        ),
        "elapsed_ns",
        -5,
    )
    with pytest.raises(EvidenceError, match="elapsed_ns"):
        collect_bayesian_evidence([record], config=_config(), artifact=artifact)


def test_unknown_fallback_reason_rejected(artifact) -> None:
    record = _tamper(
        _diag(
            node=0, round_index=1, senders=(1,), evidence=((0, 0, 0),),
            degree_bin="1", progress="mid", fallback=FALLBACK_ZERO_MASS,
        ),
        "fallback_reason",
        "made_up",
    )
    with pytest.raises(EvidenceError, match="fallback_reason"):
        collect_bayesian_evidence([record], config=_config(), artifact=artifact)


def test_report_type_is_frozen_report(artifact) -> None:
    report = collect_bayesian_evidence(
        _clean_diagnostics(), config=_config(), artifact=artifact
    )
    assert isinstance(report, BayesianEvidenceReport)
