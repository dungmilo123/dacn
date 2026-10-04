"""Paired per-cell metrics and mean/spread/worst-node summary (M003/S02/T06).

These tests exercise the deterministic paired-metrics layer that S05 consumes:

* the pure :func:`~dfl_sim.evaluation.build_robustness_summary` over a
  validated final manifest and its cited schema-v1 traces, and
* the :func:`~dfl_sim.evaluation.summarize_held_out_study` wrapper that emits
  ``robustness-summary.json`` for a published study root.

The synthetic study root is built entirely from in-process, deterministic
schema-v1 traces (no datasets, network, or subprocesses) so the expected mean,
population standard deviation, linearly interpolated IQR, and worst-node
accuracy can be hand-computed exactly. A second, real end-to-end held-out study
(five total paired seeds, one known plus one reserved schedule) proves the
orchestrator embeds a compact ``metrics`` block per final entry and writes a
summary that matches a fresh recomputation.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dfl_sim import (
    AttackAction,
    EvaluatedRound,
    EvaluationError,
    FINAL_MANIFEST_FILENAME,
    FINAL_MANIFEST_SCHEMA,
    FINAL_MANIFEST_SCHEMA_VERSION,
    MessageSummary,
    NodeEvaluation,
    PropagationState,
    ROBUSTNESS_QUANTILE_METHOD,
    ROBUSTNESS_STD_METHOD,
    ROBUSTNESS_SUMMARY_FILENAME,
    ROBUSTNESS_SUMMARY_SCHEMA,
    ROBUSTNESS_SUMMARY_SCHEMA_VERSION,
    RoundResult,
    SELECTION_MANIFEST_FILENAME,
    SELECTION_MANIFEST_SCHEMA,
    SELECTION_MANIFEST_SCHEMA_VERSION,
    TraceReader,
    build_attack_schedule,
    build_robustness_summary,
    derive_schedule_seed,
    load_topology,
    realized_kappa_sequence,
    run_held_out_study,
    schedule_digest,
    summarize_held_out_study,
    write_trace,
)
from helpers import make_config
from test_held_out_final import _dataset, _evaluation_dataset, _held_out

# --------------------------------------------------------------------------- #
# Synthetic deterministic study root (exact hand-computable metrics)
# --------------------------------------------------------------------------- #
SELECTION_SEEDS = (11, 12, 13)
FINAL_SEEDS = (21, 22, 23)
ROUNDS = 6
POLICIES = ("local_only", "plain_average")

# (schedule_id -> (schedule, held_out))
_SCHEDULES = {
    "known_periodic": (build_attack_schedule("periodic"), False),
    "held_out_schedule": (build_attack_schedule("held_out"), True),
}

# Per (schedule_id, policy_id) per-seed node-mean accuracy. Every node in a cell
# shares the value, so the final-round mean accuracy equals the value and the
# worst-node accuracy equals the same value.
_ACCURACY = {
    ("known_periodic", "local_only"): {21: 0.2, 22: 0.5, 23: 0.8},
    ("known_periodic", "plain_average"): {21: 0.4, 22: 0.4, 23: 0.4},
    ("held_out_schedule", "local_only"): {21: 0.1, 22: 0.5, 23: 0.9},
    ("held_out_schedule", "plain_average"): {21: 0.3, 22: 0.5, 23: 0.7},
}

# Per (schedule_id, policy_id) node-mean loss (seed-invariant).
_LOSS = {
    ("known_periodic", "local_only"): 2.0,
    ("known_periodic", "plain_average"): 1.0,
    ("held_out_schedule", "local_only"): 3.0,
    ("held_out_schedule", "plain_average"): 4.0,
}

NODE_COUNT = 3


def _write_metric_trace(
    destination: Path,
    *,
    seed: int,
    policy_id: str,
    final_accuracy: float,
    final_loss: float,
    rounds: int = ROUNDS,
) -> Path:
    """Write a deterministic schema-v1 trace whose final round has known metrics.

    Every node reports ``(final_loss, final_accuracy)`` in the last evaluated
    round, so the trace's final-round aggregate is exact and hand-computable.
    Earlier rounds are fixed neutral values and never enter the summary.
    """

    config = make_config(
        seed=seed,
        rounds=rounds,
        aggregation={"kind": policy_id},
        trace={"snapshot_cadence": 1},
    )
    topology = load_topology(config)
    nodes = tuple(sorted(int(node) for node in topology.nodes()))
    assert len(nodes) == NODE_COUNT

    initial = {node: {"weight": np.array([float(node), 0.0])} for node in nodes}
    rounds_data: list[tuple[EvaluatedRound, RoundResult]] = []
    for index in range(rounds):
        is_final = index == rounds - 1
        evaluations = []
        for node in nodes:
            loss = final_loss if is_final else 1.0
            accuracy = final_accuracy if is_final else 0.0
            participants = (node,) + tuple(
                sorted(int(neighbor) for neighbor in topology.neighbors(node))
            )
            count = len(participants)
            evaluations.append(
                NodeEvaluation(
                    node_id=node,
                    loss=float(loss),
                    accuracy=float(accuracy),
                    sample_count=4,
                    true_type="benign",
                    attack=AttackAction(kind="none", kappa=0.0),
                    participant_ids=participants,
                    multipliers=tuple(1.0 for _ in participants),
                    weights=tuple(1.0 / count for _ in participants),
                    propagation=PropagationState(
                        reached=False, min_hop=None, first_reached_round=None
                    ),
                )
            )
        states = {
            node: {"weight": np.array([float(node), float(index + 1)])}
            for node in nodes
        }
        messages = tuple(
            MessageSummary(
                sender=node,
                recipients=tuple(
                    sorted(int(neighbor) for neighbor in topology.neighbors(node))
                ),
            )
            for node in nodes
            if tuple(topology.neighbors(node))
        )
        result = RoundResult(
            round_index=index, states=states, stats=(), messages=messages
        )
        rounds_data.append(
            (
                EvaluatedRound(round_index=index, nodes=tuple(evaluations)),
                result,
            )
        )

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_trace(
        destination,
        config=config,
        topology=topology,
        initial_states=initial,
        rounds=rounds_data,
    )
    return destination


def _entry(
    *,
    seed: int,
    policy_id: str,
    schedule_id: str,
    trace_root: Path,
) -> dict:
    schedule, held_out = _SCHEDULES[schedule_id]
    accuracy = _ACCURACY[(schedule_id, policy_id)][seed]
    loss = _LOSS[(schedule_id, policy_id)]
    relative = f"traces/{schedule_id}/{policy_id}/seed-{seed}"
    _write_metric_trace(
        trace_root / relative,
        seed=seed,
        policy_id=policy_id,
        final_accuracy=accuracy,
        final_loss=loss,
    )
    return {
        "pairing_key": f"{policy_id}:seed-{seed}:{schedule_id}",
        "seed": seed,
        "policy_id": policy_id,
        "schedule_id": schedule_id,
        "schedule": schedule.model_dump(mode="json"),
        "schedule_digest": schedule_digest(schedule),
        "realized_kappa_sequence": list(
            realized_kappa_sequence(
                schedule, ROUNDS, derive_schedule_seed(seed, schedule)
            )
        ),
        "held_out": held_out,
        "trace_path": relative,
        "status": "complete",
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _build_synthetic_root(root: Path) -> Path:
    final_dir = root / "final"
    trace_root = final_dir
    real_final_dir = final_dir
    entries = [
        _entry(
            seed=seed,
            policy_id=policy_id,
            schedule_id=schedule_id,
            trace_root=trace_root,
        )
        for schedule_id in sorted(_SCHEDULES)
        for policy_id in POLICIES
        for seed in FINAL_SEEDS
    ]
    _write_json(
        root / "selection" / SELECTION_MANIFEST_FILENAME,
        {
            "schema": SELECTION_MANIFEST_SCHEMA,
            "schema_version": SELECTION_MANIFEST_SCHEMA_VERSION,
            "status": "frozen",
            "selection_seeds": list(SELECTION_SEEDS),
        },
    )
    _write_json(
        real_final_dir / FINAL_MANIFEST_FILENAME,
        {
            "schema": FINAL_MANIFEST_SCHEMA,
            "schema_version": FINAL_MANIFEST_SCHEMA_VERSION,
            "phase": "final",
            "status": "complete",
            "study_id": "synthetic_metrics",
            "data_provenance": "injected",
            "study_digest": "0" * 64,
            "selection_spec_digest": "1" * 64,
            "held_out_study_digest": "2" * 64,
            "final_output_digest": "3" * 64,
            "panel": list(POLICIES),
            "selection_seeds": list(SELECTION_SEEDS),
            "final_seeds": list(FINAL_SEEDS),
            "schedule_ids": sorted(_SCHEDULES),
            "entries": entries,
        },
    )
    return root


def _synthetic_summary(tmp_path: Path) -> tuple[Path, dict]:
    root = _build_synthetic_root(tmp_path / "study")
    summary = summarize_held_out_study(root)
    return root, summary


def _aggregate(summary: dict, schedule_id: str, policy_id: str) -> dict:
    matches = [
        row
        for row in summary["aggregates"]
        if row["schedule_id"] == schedule_id and row["policy_id"] == policy_id
    ]
    assert len(matches) == 1
    return matches[0]


# --------------------------------------------------------------------------- #
# Hand-computed aggregate values
# --------------------------------------------------------------------------- #
def test_summary_identity_and_seed_split(tmp_path: Path) -> None:
    root, summary = _synthetic_summary(tmp_path)
    assert summary["schema"] == ROBUSTNESS_SUMMARY_SCHEMA
    assert summary["schema_version"] == ROBUSTNESS_SUMMARY_SCHEMA_VERSION
    assert summary["status"] == "complete"
    assert summary["std_method"] == ROBUSTNESS_STD_METHOD == "population"
    assert summary["quantile_method"] == ROBUSTNESS_QUANTILE_METHOD == "linear"
    assert summary["selection_seeds"] == list(SELECTION_SEEDS)
    assert summary["final_seeds"] == list(FINAL_SEEDS)
    assert summary["schedule_ids"] == sorted(_SCHEDULES)
    assert summary["panel"] == list(POLICIES)
    assert root / ROBUSTNESS_SUMMARY_FILENAME in {p for p in root.iterdir()}


def test_aggregates_match_hand_computed_values(tmp_path: Path) -> None:
    _, summary = _synthetic_summary(tmp_path)

    known_local = _aggregate(summary, "known_periodic", "local_only")
    assert known_local["held_out"] is False
    assert known_local["seeds"] == list(FINAL_SEEDS)
    assert known_local["count"] == 3
    assert known_local["node_count"] == NODE_COUNT
    assert known_local["mean_accuracy"] == pytest.approx(0.5)
    assert known_local["std_accuracy"] == pytest.approx(0.2449489742783178)
    assert known_local["iqr_accuracy"] == pytest.approx(0.3)
    assert known_local["mean_loss"] == pytest.approx(2.0)
    assert known_local["worst_node_accuracy"] == pytest.approx(0.2)
    assert [row["mean_accuracy"] for row in known_local["per_seed"]] == pytest.approx(
        [0.2, 0.5, 0.8]
    )

    known_plain = _aggregate(summary, "known_periodic", "plain_average")
    assert known_plain["std_accuracy"] == pytest.approx(0.0)
    assert known_plain["iqr_accuracy"] == pytest.approx(0.0)
    assert known_plain["worst_node_accuracy"] == pytest.approx(0.4)
    assert known_plain["mean_loss"] == pytest.approx(1.0)

    held_local = _aggregate(summary, "held_out_schedule", "local_only")
    assert held_local["held_out"] is True
    assert held_local["mean_accuracy"] == pytest.approx(0.5)
    assert held_local["std_accuracy"] == pytest.approx(0.32659863237109044)
    assert held_local["iqr_accuracy"] == pytest.approx(0.4)
    assert held_local["mean_loss"] == pytest.approx(3.0)
    assert held_local["worst_node_accuracy"] == pytest.approx(0.1)

    held_plain = _aggregate(summary, "held_out_schedule", "plain_average")
    assert held_plain["mean_accuracy"] == pytest.approx(0.5)
    assert held_plain["std_accuracy"] == pytest.approx(0.1632993161855452)
    assert held_plain["iqr_accuracy"] == pytest.approx(0.2)
    assert held_plain["worst_node_accuracy"] == pytest.approx(0.3)


def test_summary_is_deterministic(tmp_path: Path) -> None:
    root = _build_synthetic_root(tmp_path / "study")
    first = summarize_held_out_study(root)
    first_bytes = (root / ROBUSTNESS_SUMMARY_FILENAME).read_bytes()
    second = summarize_held_out_study(root)
    second_bytes = (root / ROBUSTNESS_SUMMARY_FILENAME).read_bytes()
    assert first == second
    assert first_bytes == second_bytes
    assert build_robustness_summary(
        json.loads((root / "final" / FINAL_MANIFEST_FILENAME).read_text()),
        trace_root=root / "final",
    ) == first


def test_summary_is_distinct_from_canonical_report(tmp_path: Path) -> None:
    _, summary = _synthetic_summary(tmp_path)

    forbidden = {
        "figures",
        "figure",
        "figure_paths",
        "evidence_class",
        "evidence_classes",
        "claim_class",
        "claim_classes",
    }

    def collect_keys(payload: object) -> set[str]:
        if isinstance(payload, dict):
            keys = set(payload)
            for value in payload.values():
                keys |= collect_keys(value)
            return keys
        if isinstance(payload, list):
            keys: set[str] = set()
            for value in payload:
                keys |= collect_keys(value)
            return keys
        return set()

    keys = collect_keys(summary)
    assert not (keys & forbidden)
    # It is a metrics artifact, not the report compiler's output schema.
    assert summary["schema"] == ROBUSTNESS_SUMMARY_SCHEMA


# --------------------------------------------------------------------------- #
# Fail-closed negative coverage
# --------------------------------------------------------------------------- #
def _replace_trace(
    destination: Path, *, seed: int, policy_id: str
) -> None:
    """Remove an existing trace directory, then rewrite it with new provenance."""

    if destination.exists():
        for item in sorted(destination.rglob("*"), reverse=True):
            if item.is_file():
                item.unlink()
            else:
                item.rmdir()
        destination.rmdir()
    _write_metric_trace(
        destination,
        seed=seed,
        policy_id=policy_id,
        final_accuracy=0.2,
        final_loss=2.0,
    )


def test_missing_trace_fails_closed(tmp_path: Path) -> None:
    root = _build_synthetic_root(tmp_path / "study")
    victim = root / "final" / "traces" / "known_periodic" / "local_only" / "seed-21"
    for item in sorted(victim.rglob("*"), reverse=True):
        if item.is_file():
            item.unlink()
        else:
            item.rmdir()
    with pytest.raises(EvaluationError):
        summarize_held_out_study(root, write=False)


def test_seed_provenance_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _build_synthetic_root(tmp_path / "study")
    victim = root / "final" / "traces" / "known_periodic" / "local_only" / "seed-21"
    _replace_trace(victim, seed=999, policy_id="local_only")
    with pytest.raises(EvaluationError, match="seed"):
        summarize_held_out_study(root, write=False)


def test_policy_provenance_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _build_synthetic_root(tmp_path / "study")
    victim = root / "final" / "traces" / "known_periodic" / "local_only" / "seed-21"
    _replace_trace(victim, seed=21, policy_id="plain_average")
    with pytest.raises(EvaluationError, match="aggregation"):
        summarize_held_out_study(root, write=False)


def test_inconsistent_paired_seed_sets_fail_closed(tmp_path: Path) -> None:
    root = _build_synthetic_root(tmp_path / "study")
    manifest_path = root / "final" / FINAL_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"] = [
        entry
        for entry in manifest["entries"]
        if not (
            entry["schedule_id"] == "known_periodic"
            and entry["policy_id"] == "local_only"
            and entry["seed"] == 23
        )
    ]
    _write_json(manifest_path, manifest)
    with pytest.raises(EvaluationError, match="seed"):
        summarize_held_out_study(root, write=False)


def test_schedule_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _build_synthetic_root(tmp_path / "study")
    manifest_path = root / "final" / FINAL_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["schedule_digest"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(EvaluationError, match="schedule_digest"):
        summarize_held_out_study(root, write=False)


def test_non_frozen_selection_manifest_fails_closed(tmp_path: Path) -> None:
    root = _build_synthetic_root(tmp_path / "study")
    selection_path = root / "selection" / SELECTION_MANIFEST_FILENAME
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["status"] = "draft"
    _write_json(selection_path, selection)
    with pytest.raises(EvaluationError, match="frozen"):
        summarize_held_out_study(root, write=False)


def test_missing_final_manifest_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(EvaluationError, match="final-manifest.json"):
        summarize_held_out_study(root, write=False)


# --------------------------------------------------------------------------- #
# Real end-to-end integration: metrics block + written summary
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_study(tmp_path_factory):
    spec = _held_out(selection_seeds=(11, 12, 13), final_seeds=(21, 22))
    output = tmp_path_factory.mktemp("metrics-real-root") / "study"
    result = run_held_out_study(
        spec,
        output,
        dataset=_dataset(),
        evaluation_dataset=_evaluation_dataset(),
    )
    return SimpleNamespace(spec=spec, output=output, result=result)


def test_real_entries_carry_compact_metrics(real_study) -> None:
    manifest = json.loads(
        real_study.result.final_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["entries"]
    for entry in manifest["entries"]:
        metrics = entry.get("metrics")
        assert isinstance(metrics, dict)
        assert set(metrics) == {
            "mean_accuracy",
            "worst_node_accuracy",
            "mean_loss",
            "node_count",
        }
        trace_dir = real_study.result.final_path / entry["trace_path"]
        with TraceReader(trace_dir) as reader:
            nodes = reader.events[-1]["nodes"]
        assert metrics["mean_accuracy"] == pytest.approx(
            sum(node["accuracy"] for node in nodes) / len(nodes)
        )
        assert metrics["worst_node_accuracy"] == pytest.approx(
            min(node["accuracy"] for node in nodes)
        )
        assert metrics["mean_loss"] == pytest.approx(
            sum(node["loss"] for node in nodes) / len(nodes)
        )
        assert metrics["node_count"] == len(nodes)


def test_real_summary_matches_fresh_recomputation(real_study) -> None:
    summary_path = real_study.output / ROBUSTNESS_SUMMARY_FILENAME
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema"] == ROBUSTNESS_SUMMARY_SCHEMA
    assert summary["selection_seeds"] == [11, 12, 13]
    assert summary["final_seeds"] == [21, 22]

    manifest = json.loads(
        real_study.result.final_manifest_path.read_text(encoding="utf-8")
    )
    expected_pairs = {
        (entry["policy_id"], entry["schedule_id"]) for entry in manifest["entries"]
    }
    got_pairs = {
        (row["policy_id"], row["schedule_id"]) for row in summary["aggregates"]
    }
    assert got_pairs == expected_pairs
    assert all(row["count"] == 2 for row in summary["aggregates"])

    again = summarize_held_out_study(real_study.output, write=False)
    assert again == summary
    assert build_robustness_summary(
        manifest, trace_root=real_study.result.final_path
    ) == summary


def test_real_summary_has_no_canonical_report_framing(real_study) -> None:
    summary = json.loads(
        (real_study.output / ROBUSTNESS_SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    text = json.dumps(summary, sort_keys=True)
    assert "figure" not in text
    assert "evidence_class" not in text
    assert "claim_class" not in text
