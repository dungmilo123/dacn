"""Scalar cost / decision-audit telemetry tests (R023/D041, S05/T03).

These tests exercise the opt-in measurement channel end to end without any
network or repository-data dependency:

* strict scalar record validation (timings, counts, identities, digests),
* exact fake-clock accounting through the real round engine,
* logical payload bytes at message construction (mixed dtypes; no-op attack;
  zeroed-neighbor policies still transmit),
* per-round observation/Bayesian-diagnostic alignment for every policy,
* the byte-for-byte telemetry-off/on equivalence of traces and policy choices,
* atomic sidecar publication and strict reader negatives,
* evaluation-runner wiring plus manifest-matched resume sidecar reuse,
  tampering and missing-sidecar fail-closed behavior.

Every fixture is in-memory or git-tracked; nothing reads ``runs/``, ``data/`` or
``.gsd``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from helpers import MNISTShapedDataset, make_config

from dfl_sim import (
    BALANCE_CITATION,
    CellTelemetry,
    CostError,
    DecisionAuditRecord,
    EvaluationError,
    NamedSchedule,
    NodeRoundCost,
    NullCostCollector,
    RecordingCostCollector,
    RoundTelemetry,
    SelectionSpec,
    build_attack_schedule,
    canonical_json_bytes,
    load_cost_payload,
    load_decision_audit_records,
    parse_study_text,
    run_baseline_study,
    run_from_config,
    run_selection_phase,
    state_dict_payload_bytes,
    validate_cost_payload,
    validate_decision_audit_record,
    validate_measurement_identity,
    WeightProfile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BAYESIAN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"
_TUNED_PROVENANCE = "selection-seed-budget"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _base(**overrides):
    defaults = {
        "rounds": 3,
        "data": {"samples_per_node": 8},
        "training": {"local_epochs": 1, "batch_size": 4},
        "evaluation": {"max_samples": 8, "batch_size": 4},
        "attack": {"compromised_node": 0, "intensity": "strong"},
    }
    defaults.update(overrides)
    return make_config(**defaults)


def _panel(*, artifact_dir=None):
    artifact = str(BAYESIAN_FIXTURE if artifact_dir is None else artifact_dir)
    project = {"kind": "project", "title": "project comparator"}
    tuned = {
        "mode": "tuned",
        "candidate_budget": 3,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
    }
    exempt = {
        "mode": "exempt",
        "candidate_budget": 0,
        "selection_seed": None,
        "selection_provenance": None,
    }
    return [
        {
            "id": "local_only",
            "implementation": "local only",
            "aggregation": {"kind": "local_only"},
            "citation": project,
            "tuning": exempt,
        },
        {
            "id": "plain_average",
            "implementation": "plain average",
            "aggregation": {"kind": "plain_average"},
            "citation": project,
            "tuning": exempt,
        },
        {
            "id": "fixed_similarity",
            "implementation": "fixed similarity",
            "aggregation": {
                "kind": "fixed_similarity",
                "cutoff": 0.5,
                "downweight_multiplier": 0.5,
            },
            "citation": project,
            "tuning": tuned,
        },
        {
            "id": "trust_weighted",
            "implementation": "trust weighted",
            "aggregation": {"kind": "trust_weighted", "beta": 1.0},
            "citation": project,
            "tuning": tuned,
        },
        {
            "id": "balance",
            "implementation": "faithful BALANCE",
            "aggregation": {"kind": "balance"},
            "citation": dict(BALANCE_CITATION),
            "tuning": tuned,
        },
        {
            "id": "bayesian",
            "implementation": "M002 Bayesian",
            "aggregation": {
                "kind": "bayesian",
                "artifact_dir": artifact,
                "omega": 0.5,
                "degree_cap": 4,
                "tie_tolerance": 1e-9,
            },
            "citation": project,
            "tuning": tuned,
        },
    ]


def _study_mapping(*, seeds=(11,)):
    return {
        "study_id": "costs_fixture",
        "data_provenance": "injected",
        "base": _base().model_dump(mode="json"),
        "seeds": list(seeds),
        "candidate_budget": 3,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
        "panel": _panel(),
    }


def _study(*, seeds=(11,)):
    return parse_study_text(yaml.safe_dump(_study_mapping(seeds=seeds), sort_keys=True))


def _selection(*, seeds=(11,)):
    study = _study(seeds=seeds)
    schedule = NamedSchedule(
        id="periodic_known", schedule=build_attack_schedule("periodic")
    )
    return SelectionSpec(study=study, known_schedules=(schedule,))


def _policy_config(kind: str):
    aggregation = {"kind": kind}
    if kind == "fixed_similarity":
        aggregation = {
            "kind": "fixed_similarity",
            "cutoff": 0.5,
            "downweight_multiplier": 0.5,
        }
    elif kind == "trust_weighted":
        aggregation = {"kind": "trust_weighted", "beta": 1.0}
    elif kind == "bayesian":
        aggregation = {
            "kind": "bayesian",
            "artifact_dir": str(BAYESIAN_FIXTURE),
            "omega": 0.5,
            "degree_cap": 4,
            "tie_tolerance": 1e-9,
        }
    return make_config(
        rounds=3,
        topology={"kind": "chain3"},
        aggregation=aggregation,
        attack={"compromised_node": 0, "intensity": "strong"},
    )


def _trace_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


class _CounterClock:
    """Deterministic monotonic fake clock: ``value`` increments per call."""

    def __init__(self, step: int = 1000) -> None:
        self._step = int(step)
        self._value = 0
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        self._value += self._step
        return self._value


class _SelfOnlyPolicy:
    """A policy that zeroes every neighbor but still returns a valid profile."""

    def select_weights(self, observation):
        degree = observation.degree
        multipliers = (1.0,) + (0.0,) * degree
        weights = (1.0,) + (0.0,) * degree
        return WeightProfile(
            participant_ids=observation.participant_ids,
            multipliers=multipliers,
            weights=weights,
        )


# --------------------------------------------------------------------------- #
# Payload accounting
# --------------------------------------------------------------------------- #
def test_state_dict_payload_bytes_mixed_dtype() -> None:
    state = {
        "a": torch.zeros(2, 3, dtype=torch.float32),
        "b": torch.zeros(4, dtype=torch.float64),
        "c": torch.zeros(5, dtype=torch.int8),
    }
    assert state_dict_payload_bytes(state) == 24 + 32 + 5


def test_state_dict_payload_bytes_zero_and_empty() -> None:
    assert state_dict_payload_bytes({}) == 0
    assert state_dict_payload_bytes({"z": torch.zeros(0, dtype=torch.float32)}) == 0


def test_state_dict_payload_bytes_rejects_non_tensor_and_non_mapping() -> None:
    with pytest.raises(CostError):
        state_dict_payload_bytes({"a": [1, 2, 3]})
    with pytest.raises(CostError):
        state_dict_payload_bytes([1, 2, 3])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Strict record validation
# --------------------------------------------------------------------------- #
def test_node_round_cost_rejects_negative_timing() -> None:
    with pytest.raises(CostError):
        NodeRoundCost(
            node_id=0,
            round_index=0,
            degree=1,
            delivered_messages=1,
            delivered_payload_bytes=10,
            observation_ns=-1,
            policy_ns=0,
            aggregation_ns=0,
        )


def test_node_round_cost_rejects_message_degree_mismatch() -> None:
    with pytest.raises(CostError):
        NodeRoundCost(
            node_id=0,
            round_index=0,
            degree=2,
            delivered_messages=1,
            delivered_payload_bytes=10,
            observation_ns=0,
            policy_ns=0,
            aggregation_ns=0,
        )


def test_round_telemetry_rejects_negative_counter() -> None:
    with pytest.raises(CostError):
        RoundTelemetry(
            round_index=0,
            undirected_links=1,
            messages=2,
            payload_bytes=10,
            model_payload_bytes=5,
            train_ns=-5,
            nodes=(),
        )


def test_round_telemetry_rejects_mismatched_totals() -> None:
    node = NodeRoundCost(
        node_id=0,
        round_index=0,
        degree=1,
        delivered_messages=1,
        delivered_payload_bytes=5,
        observation_ns=1,
        policy_ns=1,
        aggregation_ns=1,
    )
    with pytest.raises(CostError):
        RoundTelemetry(
            round_index=0,
            undirected_links=1,
            messages=2,
            payload_bytes=5,
            model_payload_bytes=5,
            train_ns=1,
            nodes=(node,),
        )


def test_recording_collector_rejects_duplicate_round_and_bad_phase() -> None:
    collector = RecordingCostCollector(clock=_CounterClock())
    row = RoundTelemetry(
        round_index=0,
        undirected_links=0,
        messages=0,
        payload_bytes=0,
        model_payload_bytes=0,
        train_ns=0,
        nodes=(),
    )
    collector.record_round(row)
    with pytest.raises(CostError):
        collector.record_round(row)
    with pytest.raises(CostError):
        collector.add_wall("nonsense", 1)


def test_decision_record_rejects_nan_and_length_mismatch() -> None:
    with pytest.raises(CostError):
        DecisionAuditRecord(
            node_id=0,
            round_index=0,
            degree=1,
            sender_ids=(1,),
            disagreement=(float("nan"),),
            relative_norm=(0.0,),
            loss_delta=(0.0,),
            chosen_multipliers=(1.0, 0.0),
            weights=(1.0, 0.0),
            candidate_count=None,
            pruned=None,
            posterior=(),
            fallback_reason=None,
            normalization_mass=None,
            evidence_bins=(),
            selection_ns=0,
        )
    with pytest.raises(CostError):
        DecisionAuditRecord(
            node_id=0,
            round_index=0,
            degree=2,
            sender_ids=(1,),
            disagreement=(0.0,),
            relative_norm=(0.0,),
            loss_delta=(0.0,),
            chosen_multipliers=(1.0, 0.0, 0.0),
            weights=(1.0, 0.0, 0.0),
            candidate_count=None,
            pruned=None,
            posterior=(),
            fallback_reason=None,
            normalization_mass=None,
            evidence_bins=(),
            selection_ns=0,
        )


def test_measurement_identity_validation() -> None:
    with pytest.raises(CostError):
        validate_measurement_identity({"phase": "baseline", "seed": 1})
    with pytest.raises(CostError):
        validate_measurement_identity(
            {"phase": "baseline", "policy_id": "p", "seed": 1, "extra": "x"}
        )
    with pytest.raises(CostError):
        validate_measurement_identity(
            {"phase": "baseline", "policy_id": "p", "seed": -1}
        )
    with pytest.raises(CostError):
        validate_measurement_identity(
            {"phase": "baseline", "policy_id": "", "seed": 1}
        )
    canonical = validate_measurement_identity(
        {"phase": "selection", "policy_id": "p", "seed": 3, "schedule_id": "s"}
    )
    assert canonical == {
        "phase": "selection",
        "policy_id": "p",
        "seed": 3,
        "schedule_id": "s",
    }


def test_null_collector_is_noop() -> None:
    collector = NullCostCollector()
    assert collector.enabled is False
    collector.record_round(None)  # type: ignore[arg-type]
    collector.add_wall("setup", 123)


# --------------------------------------------------------------------------- #
# Exact fake-clock accounting through the real engine
# --------------------------------------------------------------------------- #
def test_fake_clock_exact_nanosecond_boundaries() -> None:
    clock = _CounterClock(step=1000)
    telemetry = CellTelemetry(clock=clock)
    result = run_from_config(
        _policy_config("plain_average"),
        dataset=MNISTShapedDataset(96, seed=1),
        telemetry=telemetry,
    )
    assert len(result.rounds) == 3
    assert len(telemetry.rounds) == 3
    for round_row in telemetry.rounds:
        # Training is measured as one start/end pair per round.
        assert round_row.train_ns == 1000
        for node in round_row.nodes:
            assert node.observation_ns == 1000
            assert node.policy_ns == 1000
            assert node.aggregation_ns == 1000
    payload = telemetry.cost_payload(
        identity={"phase": "baseline", "policy_id": "plain_average", "seed": 1},
        config_digest="a" * 64,
        artifact_digests={"aggregation_artifact": None},
        trace_digest="b" * 64,
    )
    assert payload["totals"]["train_ns"] == 3000
    assert payload["totals"]["observation_ns"] == 9000
    assert payload["totals"]["setup_ns"] == 1000


def test_begin_cell_bounds_decision_rows() -> None:
    telemetry = CellTelemetry(clock=_CounterClock())
    telemetry.begin_cell(rounds=2, nodes=3)
    # No observations recorded -> finalization must fail closed, not emit zeros.
    with pytest.raises(CostError):
        telemetry.decision_records()


# --------------------------------------------------------------------------- #
# Communication counting and all-policy measurement
# --------------------------------------------------------------------------- #
def test_local_only_still_transmits_and_counts_bytes() -> None:
    telemetry = CellTelemetry(clock=_CounterClock())
    run_from_config(
        _policy_config("local_only"),
        dataset=MNISTShapedDataset(96, seed=1),
        telemetry=telemetry,
    )
    totals = telemetry.cost_payload(
        identity={"phase": "baseline", "policy_id": "local_only", "seed": 1},
        config_digest="a" * 64,
        artifact_digests={},
        trace_digest="b" * 64,
    )["totals"]
    # chain3 has two undirected edges -> four directed messages per round.
    assert totals["directed_messages"] == 12
    assert totals["undirected_links"] == 6
    assert totals["logical_payload_bytes"] > 0
    # Every node's selected neighbor weight is zero, yet bytes are still counted.
    for record in telemetry.decision_records():
        assert all(weight == 0.0 for weight in record.weights[1:])


def test_self_only_policy_counts_messages_before_filtering() -> None:
    telemetry = CellTelemetry(clock=_CounterClock())
    run_from_config(
        _policy_config("plain_average"),
        dataset=MNISTShapedDataset(96, seed=1),
        policy=_SelfOnlyPolicy(),
        telemetry=telemetry,
    )
    totals = telemetry.cost_payload(
        identity={"phase": "baseline", "policy_id": "self_only", "seed": 1},
        config_digest="a" * 64,
        artifact_digests={},
        trace_digest="b" * 64,
    )["totals"]
    assert totals["directed_messages"] == 12
    assert totals["logical_payload_bytes"] > 0


@pytest.mark.parametrize(
    "kind",
    ["local_only", "plain_average", "fixed_similarity", "trust_weighted", "balance", "bayesian"],
)
def test_every_policy_produces_aligned_measurements(kind: str) -> None:
    telemetry = CellTelemetry(clock=_CounterClock())
    result = run_from_config(
        _policy_config(kind),
        dataset=MNISTShapedDataset(96, seed=1),
        telemetry=telemetry,
    )
    nodes = len(result.final_states)
    assert len(telemetry.rounds) == 3
    records = telemetry.decision_records()
    assert len(records) == nodes * 3
    for record in records:
        assert record.degree == len(record.sender_ids)
        assert list(record.sender_ids) == sorted(record.sender_ids)
        assert len(record.weights) == record.degree + 1
        assert abs(sum(record.weights) - 1.0) < 1e-9
    if kind == "bayesian":
        for record in records:
            assert record.candidate_count is not None
            assert len(record.evidence_bins) == record.degree
    else:
        for record in records:
            assert record.candidate_count is None
            assert record.evidence_bins == ()


def test_per_round_audit_alignment() -> None:
    telemetry = CellTelemetry(clock=_CounterClock())
    result = run_from_config(
        _policy_config("plain_average"),
        dataset=MNISTShapedDataset(96, seed=1),
        telemetry=telemetry,
    )
    nodes = tuple(sorted(result.final_states))
    seen = {(record.node_id, record.round_index) for record in telemetry.decision_records()}
    assert seen == {(node, index) for index in range(3) for node in nodes}


# --------------------------------------------------------------------------- #
# Telemetry off/on equivalence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["plain_average", "bayesian"])
def test_telemetry_off_on_trace_bytes_and_profiles_identical(
    kind: str, tmp_path: Path
) -> None:
    config = _policy_config(kind)
    dataset = MNISTShapedDataset(96, seed=1)
    off_dir = tmp_path / f"{kind}-off"
    on_dir = tmp_path / f"{kind}-on"
    off = run_from_config(config, dataset=dataset, trace_path=off_dir)
    on = run_from_config(
        config,
        dataset=dataset,
        trace_path=on_dir,
        telemetry=CellTelemetry(clock=_CounterClock()),
    )
    assert _trace_digest(off_dir) == _trace_digest(on_dir)
    for off_round, on_round in zip(off.rounds, on.rounds):
        assert off_round.profiles == on_round.profiles
        assert off_round.messages == on_round.messages


# --------------------------------------------------------------------------- #
# Sidecar publication and strict readers
# --------------------------------------------------------------------------- #
def _telemetry_payload(telemetry: CellTelemetry) -> tuple[dict, tuple[dict, ...]]:
    identity = {"phase": "baseline", "policy_id": "plain_average", "seed": 1}
    payload = telemetry.cost_payload(
        identity=identity,
        config_digest="a" * 64,
        artifact_digests={"aggregation_artifact": None},
        trace_digest="b" * 64,
    )
    return payload, telemetry.decision_payloads(identity=identity)


def test_sidecar_roundtrip_digests_and_validation(tmp_path: Path) -> None:
    telemetry = CellTelemetry(clock=_CounterClock())
    run_from_config(
        _policy_config("plain_average"),
        dataset=MNISTShapedDataset(96, seed=1),
        telemetry=telemetry,
    )
    sidecars = telemetry.write_sidecars(
        tmp_path / "cell",
        identity={"phase": "baseline", "policy_id": "plain_average", "seed": 1},
        config_digest="a" * 64,
        artifact_digests={"aggregation_artifact": None},
        trace_digest="b" * 64,
    )
    assert sidecars.cost_path.is_file()
    assert sidecars.decision_audit_path.is_file()
    assert hashlib.sha256(sidecars.cost_path.read_bytes()).hexdigest() == sidecars.cost_digest
    payload = load_cost_payload(sidecars.cost_path)
    assert payload["schema"] == "dfl_cell_cost"
    assert payload["measurement"]["trace_digest"] == "b" * 64
    records = load_decision_audit_records(sidecars.decision_audit_path)
    assert len(records) == 9
    # Writing again must not silently overwrite.
    with pytest.raises(CostError):
        telemetry.write_sidecars(
            tmp_path / "cell",
            identity={"phase": "baseline", "policy_id": "plain_average", "seed": 1},
            config_digest="a" * 64,
            artifact_digests={"aggregation_artifact": None},
            trace_digest="b" * 64,
        )


def test_write_sidecars_rejects_symlinked_parent(tmp_path: Path) -> None:
    telemetry = CellTelemetry(clock=_CounterClock())
    run_from_config(
        _policy_config("plain_average"),
        dataset=MNISTShapedDataset(96, seed=1),
        telemetry=telemetry,
    )
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(CostError):
        telemetry.write_sidecars(
            link,
            identity={"phase": "baseline", "policy_id": "plain_average", "seed": 1},
            config_digest="a" * 64,
            artifact_digests={},
            trace_digest="b" * 64,
        )


def test_validate_cost_payload_rejects_mutations(tmp_path: Path) -> None:
    telemetry = CellTelemetry(clock=_CounterClock())
    run_from_config(
        _policy_config("plain_average"),
        dataset=MNISTShapedDataset(96, seed=1),
        telemetry=telemetry,
    )
    payload, _ = _telemetry_payload(telemetry)
    validate_cost_payload(payload)

    unknown = json.loads(json.dumps(payload))
    unknown["surprise"] = 1
    with pytest.raises(CostError):
        validate_cost_payload(unknown)

    missing = json.loads(json.dumps(payload))
    del missing["totals"]
    with pytest.raises(CostError):
        validate_cost_payload(missing)

    bad_clock = json.loads(json.dumps(payload))
    bad_clock["measurement"]["clock"] = "wall"
    with pytest.raises(CostError):
        validate_cost_payload(bad_clock)

    bad_digest = json.loads(json.dumps(payload))
    bad_digest["measurement"]["trace_digest"] = "not-a-digest"
    with pytest.raises(CostError):
        validate_cost_payload(bad_digest)

    negative = json.loads(json.dumps(payload))
    negative["totals"]["observation_ns"] = -1
    with pytest.raises(CostError):
        validate_cost_payload(negative)

    inconsistent = json.loads(json.dumps(payload))
    inconsistent["totals"]["directed_messages"] += 1
    with pytest.raises(CostError):
        validate_cost_payload(inconsistent)


def test_validate_decision_record_rejects_mutations(tmp_path: Path) -> None:
    telemetry = CellTelemetry(clock=_CounterClock())
    run_from_config(
        _policy_config("bayesian"),
        dataset=MNISTShapedDataset(96, seed=1),
        telemetry=telemetry,
    )
    _, records = _telemetry_payload(telemetry)
    validate_decision_audit_record(records[0])

    unknown = json.loads(json.dumps(records[0]))
    unknown["surprise"] = True
    with pytest.raises(CostError):
        validate_decision_audit_record(unknown)

    bad_degree = json.loads(json.dumps(records[0]))
    bad_degree["degree"] = bad_degree["degree"] + 1
    with pytest.raises(CostError):
        validate_decision_audit_record(bad_degree)

    bad_value = json.loads(json.dumps(records[0]))
    bad_value["weights"][0] = -0.5
    with pytest.raises(CostError):
        validate_decision_audit_record(bad_value)


def test_validate_decision_record_rejects_nonfinite_json() -> None:
    # NaN cannot round-trip through canonical JSON; the strict reader rejects a
    # hand-built non-finite feature.
    payload = {
        "schema": "dfl_cell_decision_audit",
        "schema_version": 1,
        "identity": {"phase": "baseline", "policy_id": "p", "seed": 1},
        "node_id": 0,
        "round_index": 0,
        "degree": 1,
        "sender_ids": [1],
        "disagreement": [float("nan")],
        "relative_norm": [0.0],
        "loss_delta": [0.0],
        "chosen_multipliers": [1.0, 0.0],
        "weights": [1.0, 0.0],
        "candidate_count": None,
        "pruned": None,
        "posterior": [],
        "fallback_reason": None,
        "normalization_mass": None,
        "evidence_bins": [],
        "selection_ns": 0,
    }
    with pytest.raises(CostError):
        validate_decision_audit_record(payload)


def test_canonical_json_bytes_is_deterministic() -> None:
    first = canonical_json_bytes({"b": 1, "a": [2, 3]})
    second = canonical_json_bytes({"a": [2, 3], "b": 1})
    assert first == second


# --------------------------------------------------------------------------- #
# Evaluation-runner wiring
# --------------------------------------------------------------------------- #
def test_baseline_study_telemetry_publishes_sidecars(tmp_path: Path) -> None:
    output = tmp_path / "study"
    result = run_baseline_study(
        _study(),
        output,
        dataset=MNISTShapedDataset(64, seed=1),
        evaluation_dataset=MNISTShapedDataset(32, seed=2),
        collect_telemetry=True,
    )
    assert len(result.cells) == 6
    for cell in result.cells:
        block = cell.manifest_entry.get("telemetry")
        assert isinstance(block, dict)
        payload = load_cost_payload(output / block["cost_path"])
        records = load_decision_audit_records(output / block["decision_audit_path"])
        assert payload["identity"]["phase"] == "baseline"
        assert payload["identity"]["policy_id"] == cell.policy_id
        assert payload["measurement"]["trace_digest"] == block["trace_digest"]
        assert len(records) == 9


def test_baseline_study_without_telemetry_has_no_blocks(tmp_path: Path) -> None:
    output = tmp_path / "study"
    result = run_baseline_study(
        _study(),
        output,
        dataset=MNISTShapedDataset(64, seed=1),
        evaluation_dataset=MNISTShapedDataset(32, seed=2),
    )
    for cell in result.cells:
        assert "telemetry" not in cell.manifest_entry
    assert not (output / "telemetry").exists()


# --------------------------------------------------------------------------- #
# Manifest-matched resume sidecar reuse / tampering / missing
# --------------------------------------------------------------------------- #
def _run_selection(destination: Path, *, telemetry: bool, resume_from=None):
    return run_selection_phase(
        _selection(),
        destination,
        dataset=MNISTShapedDataset(64, seed=1),
        evaluation_dataset=MNISTShapedDataset(32, seed=2),
        collect_telemetry=telemetry,
        resume_from=resume_from,
    )


def test_selection_resume_reuses_matched_sidecars(tmp_path: Path) -> None:
    first = tmp_path / "first"
    _run_selection(first, telemetry=True)
    second = tmp_path / "second"
    result = _run_selection(second, telemetry=True, resume_from=first)
    assert all(cell.manifest_entry["reused"] for cell in result.cells)
    for cell in result.cells:
        block = cell.manifest_entry["telemetry"]
        assert (second / block["cost_path"]).is_file()
        assert (second / block["decision_audit_path"]).is_file()
        load_cost_payload(second / block["cost_path"])
        load_decision_audit_records(second / block["decision_audit_path"])


def test_selection_resume_sidecar_tamper_fails_closed(tmp_path: Path) -> None:
    first = tmp_path / "first"
    _run_selection(first, telemetry=True)
    target = next((first / "telemetry").rglob("costs.json"))
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(EvaluationError):
        _run_selection(tmp_path / "second", telemetry=True, resume_from=first)
    assert not (tmp_path / "second").exists()


def test_selection_resume_missing_sidecar_fails_closed(tmp_path: Path) -> None:
    first = tmp_path / "first"
    _run_selection(first, telemetry=True)
    target = next((first / "telemetry").rglob("decision-audit.jsonl"))
    target.unlink()
    with pytest.raises(EvaluationError):
        _run_selection(tmp_path / "second", telemetry=True, resume_from=first)
    assert not (tmp_path / "second").exists()


def test_selection_resume_without_telemetry_ignores_sidecars(tmp_path: Path) -> None:
    first = tmp_path / "first"
    _run_selection(first, telemetry=True)
    second = tmp_path / "second"
    result = _run_selection(second, telemetry=False, resume_from=first)
    assert all(cell.manifest_entry["reused"] for cell in result.cells)
    for cell in result.cells:
        assert "telemetry" not in cell.manifest_entry


def test_selection_resume_legacy_checkpoint_recomputes_when_telemetry_required(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    _run_selection(first, telemetry=False)
    second = tmp_path / "second"
    result = _run_selection(second, telemetry=True, resume_from=first)
    # Legacy checkpoint cells carry no sidecars, so they are recomputed rather
    # than reused; the fresh output still carries matched sidecars.
    for cell in result.cells:
        assert cell.manifest_entry["reused"] is False
        assert "telemetry" in cell.manifest_entry
