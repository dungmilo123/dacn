"""Independent headline-evidence audit and publication-gate tests (S04/T03).

Everything in this module is an **offline fixture**: the calibration inputs are
the in-process ``MNISTShapedDataset`` and the verified-cache validator is
replaced with an isolated manifest-only seam. Passing checks here are labelled
fixture-only and are *not* real-MNIST evidence; the genuine downloaded-data
acceptance path is exercised by ``python -m dfl_sim.headline_verify`` on the
T04 output root.

The module builds one complete, self-consistent headline tree (frozen protocol,
provenance, three matched calibration artifacts, frozen strict study, the full
64 selection + 120 final cell matrix, schema-v1 traces and the paired summary)
and then exercises acceptance, idempotent re-verification, atomic collision and
a broad negative matrix that must each fail closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import yaml

from dfl_sim import (
    AttackAction,
    EvaluatedRound,
    MessageSummary,
    NodeEvaluation,
    PropagationState,
    RoundResult,
    load_topology,
    write_trace,
)
from dfl_sim.ablation_evaluation import (
    AblationMatchSummary,
    ablation_entry_metadata,
    ablation_manifest_metadata,
    preflight_ablation_study,
)
from dfl_sim.attack_schedule import (
    derive_schedule_seed,
    realized_kappa_sequence,
    schedule_digest,
)
from dfl_sim.bayesian_artifact import CALIBRATION_VARIANTS
from dfl_sim.calibration import calibrate_from_config
from dfl_sim.evaluation import (
    FINAL_MANIFEST_FILENAME,
    ROBUSTNESS_SUMMARY_FILENAME,
    SELECTION_MANIFEST_FILENAME,
    TraceReader,
    build_cell_config,
    build_robustness_summary,
    compute_shared_input_digest,
    final_output_digest,
    selection_output_digest,
)
from dfl_sim.evaluation_config import (
    EXTENDED_POLICY_IDS,
    held_out_study_digest,
    selection_projection,
    selection_spec_digest,
    study_digest,
)
from dfl_sim.headline import (
    CALIBRATION_DIRNAME,
    FROZEN_PROTOCOL_FILENAME,
    FROZEN_STUDY_FILENAME,
    MNIST_PROVENANCE_FILENAME,
    STUDY_DIRNAME,
    build_frozen_protocol,
    build_study,
    expected_cell_counts,
    load_headline_config,
    serialize_study,
)
from dfl_sim.headline_verify import (
    HEADLINE_EVIDENCE_FILENAME,
    VERIFY_MARKER,
    HeadlineVerifyDeps,
    HeadlineVerifyError,
    verify_headline,
)
from dfl_sim.mnist_provision import (
    PREPROCESSING_ID,
    ProvisionManifest,
    SplitIdentity,
    _manifest_bytes,
    _manifest_digest,
    _preprocessing_digest,
    _publish_manifest_bytes,
    load_manifest,
)
from helpers import MNISTShapedDataset, make_config
from test_calibration_config import base_dict, parse_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
HEADLINE_CONFIG = REPO_ROOT / "configs" / "headline_mnist.yaml"

VARIANTS = CALIBRATION_VARIANTS
CAL_SEED = 20240101
CAL_EPISODES = 8
CAL_DESIGN = 8

# The known schedule id whose final plain_average aggregate is hand-computed.
_HAND_SCHEDULE = "duty_cycle"
_HAND_POLICY = "plain_average"

# Fixture-only note stamped into assert messages so a green suite is never
# mistaken for real-MNIST acceptance.
FIXTURE_ONLY = "fixture-only: not real-MNIST evidence"


# --------------------------------------------------------------------------- #
# Cache manifest (genuine schema, fixture-only content identities)
# --------------------------------------------------------------------------- #
def _split_identity(split: str, count: int) -> SplitIdentity:
    return SplitIdentity(
        split=split,
        image_resource=f"{split}-images-idx3-ubyte.gz",
        label_resource=f"{split}-labels-idx1-ubyte.gz",
        image_archive_md5=("a" if split == "train" else "b") * 32,
        label_archive_md5=("c" if split == "train" else "d") * 32,
        raw_image_sha256=("1" if split == "train" else "2") * 64,
        raw_label_sha256=("3" if split == "train" else "4") * 64,
        decoded_image_sha256=("5" if split == "train" else "6") * 64,
        decoded_label_sha256=("7" if split == "train" else "8") * 64,
        raw_image_bytes=1000 if split == "train" else 500,
        raw_label_bytes=100 if split == "train" else 50,
        count=count,
        image_shape=(28, 28),
        label_min=0,
        label_max=9,
    )


def _cache_manifest() -> ProvisionManifest:
    partial = ProvisionManifest(
        preprocessing_id=PREPROCESSING_ID,
        preprocessing_digest=_preprocessing_digest(),
        train=_split_identity("train", 60000),
        test=_split_identity("test", 10000),
        manifest_digest="0" * 64,
    )
    return partial.model_copy(update={"manifest_digest": _manifest_digest(partial)})


def _fixture_cache_verify(data_root: str, _deep: bool) -> ProvisionManifest:
    """Isolated manifest-only cache seam: fixture-only, no dataset hashing."""

    return load_manifest(data_root)


# --------------------------------------------------------------------------- #
# Calibration triple (genuine production pipeline, injected offline dataset)
# --------------------------------------------------------------------------- #
def _calibration_config(variant: str):
    experiment = make_config(
        seed=CAL_SEED,
        rounds=1,
        training={"local_epochs": 1, "batch_size": 8},
        data={"samples_per_node": 8},
        evaluation={"max_samples": 8, "batch_size": 8},
    )
    return parse_dict(
        base_dict(
            experiment=experiment.model_dump(mode="json"),
            episodes=CAL_EPISODES,
            p0=0.5,
            design_split_size=CAL_DESIGN,
            variant=variant,
        )
    )


def _build_artifacts(destination: Path) -> dict[str, object]:
    dataset = MNISTShapedDataset(400, separable=True, seed=11)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, object] = {}
    for variant in VARIANTS:
        dest = destination / variant
        calibrate_from_config(_calibration_config(variant), str(dest), dataset=dataset)
        from dfl_sim.bayesian_artifact import load_bayesian_artifact

        artifacts[variant] = load_bayesian_artifact(dest)
    return artifacts


def _fixture_preflight(study) -> AblationMatchSummary:
    """Real preflight, re-labelled production for the offline fixture only.

    The offline calibration inputs are injected, so the derivation provenance
    reads ``injected``; the verifier's no-injection gate is exercised against
    the *unpatched* preflight in a dedicated negative test. Here we run the
    genuine matched-artifact gate against an injected-provenance view of the
    study and then relabel the result so the acceptance path can be exercised
    offline.
    """

    injected_view = study.model_copy(update={"data_provenance": "injected"})
    match = preflight_ablation_study(injected_view)
    variants = MappingProxyType(
        {
            policy_id: replace(identity, data_provenance="production")
            for policy_id, identity in match.variants.items()
        }
    )
    return replace(match, data_provenance="production", variants=variants)


# --------------------------------------------------------------------------- #
# Schema-v1 trace synthesis
# --------------------------------------------------------------------------- #
def _write_cell_trace(
    destination: Path,
    *,
    config,
    final_accuracy: float,
    final_loss: float,
) -> None:
    topology = load_topology(config)
    nodes = tuple(sorted(int(node) for node in topology.nodes()))
    compromised = config.attack.compromised_node
    rounds = int(config.rounds)
    # Schema-v1 traces snapshot the *static* study attack: the runtime
    # controller applies the per-round schedule without mutating the frozen
    # config, so the traced node attack is constant across rounds (matching
    # production and the S02 metric-trace fixture).
    base_kappa = float(config.attack.kappa) if compromised is not None else 0.0
    if base_kappa <= 0.0:
        base_kappa = 0.0
    base_kind = "delta_reversal" if base_kappa > 0.0 else "none"
    initial = {node: {"weight": np.array([float(node), 0.0])} for node in nodes}
    rounds_data = []
    for index in range(rounds):
        is_final = index == rounds - 1
        evaluations = []
        for node in nodes:
            attacker = compromised is not None and node == int(compromised)
            node_kappa = base_kappa if attacker else 0.0
            node_kind = base_kind if attacker else "none"
            participants = (node,) + tuple(
                sorted(int(neighbour) for neighbour in topology.neighbors(node))
            )
            evaluations.append(
                NodeEvaluation(
                    node_id=node,
                    loss=float(final_loss if is_final else 1.0),
                    accuracy=float(final_accuracy if is_final else 0.0),
                    sample_count=4,
                    true_type="malicious" if attacker else "benign",
                    attack=AttackAction(kind=node_kind, kappa=node_kappa),
                    participant_ids=participants,
                    multipliers=tuple(1.0 for _ in participants),
                    weights=tuple(1.0 / len(participants) for _ in participants),
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
                    sorted(int(neighbour) for neighbour in topology.neighbors(node))
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_trace(
        destination,
        config=config,
        topology=topology,
        initial_states=initial,
        rounds=rounds_data,
    )


# --------------------------------------------------------------------------- #
# Manifest / tree construction
# --------------------------------------------------------------------------- #
def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _cell_config_digest(study, policy_id: str, seed: int) -> str:
    panel = {entry.id: entry for entry in study.study.panel}
    config = build_cell_config(study.study.base, panel[policy_id].aggregation, seed)
    payload = config.model_dump(mode="json")
    aggregation = payload.get("aggregation")
    if isinstance(aggregation, dict) and "artifact_dir" in aggregation:
        aggregation["artifact_dir"] = "<artifact>"
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _accuracy(schedule_id: str, policy_id: str, seed: int, rank: int) -> float:
    if schedule_id == _HAND_SCHEDULE and policy_id == _HAND_POLICY:
        return (0.2, 0.5, 0.8)[rank]
    return 0.5


def _base_entry(
    *, study, match, seed: int, schedule, policy, rounds: int, config_digest: str
) -> dict:
    entry = {
        "pairing_key": f"{policy.id}:seed-{seed}:{schedule.id}",
        "seed": seed,
        "policy_id": policy.id,
        "policy_kind": policy.aggregation.kind,
        "schedule_id": schedule.id,
        "schedule": schedule.schedule.model_dump(mode="json"),
        "schedule_digest": schedule_digest(schedule.schedule),
        "realized_kappa_sequence": list(
            realized_kappa_sequence(
                schedule.schedule, rounds, derive_schedule_seed(seed, schedule.schedule)
            )
        ),
        "config_digest": config_digest,
        "artifact_digest": (
            match.identity(policy.id).aggregate_digest
            if policy.id in match.variants
            else None
        ),
        "status": "complete",
        "reused": False,
        "trace_path": f"traces/{schedule.id}/{policy.id}/seed-{seed}",
    }
    ablation = ablation_entry_metadata(match, policy.id)
    if ablation is not None:
        entry["ablation"] = ablation
    return entry


def _build_selection(section: SimpleNamespace) -> dict:
    study = section.study
    match = section.match
    rounds = int(study.study.base.rounds)
    schedules = {entry.id: entry for entry in study.known_schedules}
    panel = {entry.id: entry for entry in study.study.panel}
    entries = []
    for seed in sorted(int(seed) for seed in study.study.seeds):
        for schedule_id in sorted(schedules):
            for policy_id in EXTENDED_POLICY_IDS:
                policy = panel[policy_id]
                schedule = schedules[schedule_id]
                _write_cell_trace(
                    section.selection_dir
                    / "traces"
                    / schedule_id
                    / policy_id
                    / f"seed-{seed}",
                    config=build_cell_config(study.study.base, policy.aggregation, seed),
                    final_accuracy=0.5,
                    final_loss=1.0,
                )
                entries.append(
                    _base_entry(
                        study=study,
                        match=match,
                        seed=seed,
                        schedule=schedule,
                        policy=policy,
                        rounds=rounds,
                        config_digest=_cell_config_digest(study, policy_id, seed),
                    )
                )
    entries.sort(key=lambda row: (row["seed"], row["schedule_id"], row["policy_id"]))
    manifest = {
        "schema": "dfl_held_out_selection",
        "schema_version": 1,
        "status": "frozen",
        "study_id": study.study.study_id,
        "data_provenance": "real_mnist",
        "study_digest": study_digest(study.study),
        "selection_spec_digest": selection_spec_digest(selection_projection(study)),
        "shared_input_digest": compute_shared_input_digest(study.study.base),
        "selection_seeds": sorted(int(seed) for seed in study.study.seeds),
        "candidate_budget": study.study.candidate_budget,
        "panel": list(EXTENDED_POLICY_IDS),
        "schedule_ids": sorted(schedules),
        "entries": entries,
    }
    ablation = ablation_manifest_metadata(match)
    if ablation is not None:
        manifest["ablation"] = ablation
    manifest["selection_output_digest"] = selection_output_digest(manifest)
    return manifest


def _build_final(section: SimpleNamespace, selection_manifest: dict) -> dict:
    study = section.study
    match = section.match
    rounds = int(study.study.base.rounds)
    schedules = {
        entry.id: entry for entry in (*study.known_schedules, study.held_out_schedule)
    }
    panel = {entry.id: entry for entry in study.study.panel}
    final_seeds = sorted(int(seed) for seed in study.final_seeds)
    entries = []
    for rank, seed in enumerate(final_seeds):
        for schedule_id in sorted(schedules):
            for policy_id in EXTENDED_POLICY_IDS:
                policy = panel[policy_id]
                schedule = schedules[schedule_id]
                accuracy = _accuracy(schedule_id, policy_id, seed, rank)
                node_count = len(load_topology(study.study.base).nodes())
                _write_cell_trace(
                    section.final_dir
                    / "traces"
                    / schedule_id
                    / policy_id
                    / f"seed-{seed}",
                    config=build_cell_config(study.study.base, policy.aggregation, seed),
                    final_accuracy=accuracy,
                    final_loss=1.0 if schedule_id == _HAND_SCHEDULE else 2.0,
                )
                entry = _base_entry(
                    study=study,
                    match=match,
                    seed=seed,
                    schedule=schedule,
                    policy=policy,
                    rounds=rounds,
                    config_digest=_cell_config_digest(study, policy_id, seed),
                )
                entry["held_out"] = schedule_id == study.held_out_schedule.id
                entry["metrics"] = {
                    "mean_accuracy": accuracy,
                    "worst_node_accuracy": accuracy,
                    "mean_loss": 1.0 if schedule_id == _HAND_SCHEDULE else 2.0,
                    "node_count": node_count,
                }
                entries.append(entry)
    entries.sort(key=lambda row: (row["seed"], row["schedule_id"], row["policy_id"]))
    manifest = {
        "schema": "dfl_held_out_study_manifest",
        "schema_version": 1,
        "phase": "final",
        "status": "complete",
        "study_id": study.study.study_id,
        "data_provenance": "real_mnist",
        "study_digest": study_digest(study.study),
        "selection_spec_digest": selection_spec_digest(selection_projection(study)),
        "held_out_study_digest": held_out_study_digest(study),
        "selection_output_digest": selection_manifest["selection_output_digest"],
        "shared_input_digest": compute_shared_input_digest(study.study.base),
        "selection_seeds": sorted(int(seed) for seed in study.study.seeds),
        "final_seeds": final_seeds,
        "candidate_budget": study.study.candidate_budget,
        "panel": list(EXTENDED_POLICY_IDS),
        "schedule_ids": sorted(schedules),
        "held_out_schedule_id": study.held_out_schedule.id,
        "held_out_schedule": {
            "id": study.held_out_schedule.id,
            "schedule": study.held_out_schedule.schedule.model_dump(mode="json"),
            "schedule_digest": schedule_digest(study.held_out_schedule.schedule),
        },
        "entries": entries,
    }
    ablation = ablation_manifest_metadata(match)
    if ablation is not None:
        manifest["ablation"] = ablation
    manifest["final_output_digest"] = final_output_digest(manifest)
    return manifest


def _build_tree(root: Path, protocol) -> SimpleNamespace:
    root.mkdir(parents=True)
    data_root = root.parent / "cache"
    data_root.mkdir(parents=True, exist_ok=True)
    manifest = _cache_manifest()
    _publish_manifest_bytes(
        str(data_root / "mnist-provision.json"), _manifest_bytes(manifest)
    )

    _write_json(
        root / FROZEN_PROTOCOL_FILENAME,
        build_frozen_protocol(
            protocol,
            source="configs/headline_mnist.yaml",
            cache_manifest=manifest,
            threads=protocol.threads,
        ),
    )
    _write_json(
        root / MNIST_PROVENANCE_FILENAME,
        {
            "schema": "dfl_sim.headline_mnist_provenance/v1",
            "data_root": str(data_root),
            "manifest_digest": manifest.manifest_digest,
            "real_mnist": True,
            "manifest": manifest.model_dump(mode="json"),
        },
    )

    artifacts = _build_artifacts(root / CALIBRATION_DIRNAME)
    study = build_study(protocol, root, artifacts)
    (root / FROZEN_STUDY_FILENAME).write_text(
        serialize_study(study), encoding="utf-8"
    )
    match = _fixture_preflight(study.study)

    section = SimpleNamespace(
        root=root,
        study=study,
        match=match,
        protocol=protocol,
        manifest=manifest,
        data_root=data_root,
        selection_dir=root / STUDY_DIRNAME / "selection",
        final_dir=root / STUDY_DIRNAME / "final",
    )
    selection_manifest = _build_selection(section)
    final_manifest = _build_final(section, selection_manifest)
    _write_json(
        section.selection_dir / SELECTION_MANIFEST_FILENAME, selection_manifest
    )
    _write_json(section.final_dir / FINAL_MANIFEST_FILENAME, final_manifest)

    summary = build_robustness_summary(
        final_manifest,
        trace_root=section.final_dir,
        selection_seeds=final_manifest["selection_seeds"],
        trace_reader=TraceReader,
    )
    _write_json(root / STUDY_DIRNAME / ROBUSTNESS_SUMMARY_FILENAME, summary)
    section.selection_manifest = selection_manifest
    section.final_manifest = final_manifest
    section.summary = summary
    return section


@pytest.fixture(scope="module")
def baseline_tree(tmp_path_factory):
    protocol = load_headline_config(HEADLINE_CONFIG)
    root = tmp_path_factory.mktemp("headline-verify") / "headline"
    return _build_tree(root, protocol)


@pytest.fixture
def tree(tmp_path, baseline_tree):
    """A pristine per-test copy of the fixture headline tree."""

    dest = tmp_path / "headline"
    shutil.copytree(baseline_tree.root, dest)
    section = SimpleNamespace(**vars(baseline_tree))
    section.root = dest
    section.selection_dir = dest / STUDY_DIRNAME / "selection"
    section.final_dir = dest / STUDY_DIRNAME / "final"
    return section


def _deps(*, preflight=_fixture_preflight, cache_verify=_fixture_cache_verify):
    return HeadlineVerifyDeps(cache_verify=cache_verify, preflight=preflight)


# --------------------------------------------------------------------------- #
# Mutation helpers
# --------------------------------------------------------------------------- #
@contextmanager
def _mutated_bytes(path: Path, data: bytes):
    original = path.read_bytes()
    path.write_bytes(data)
    try:
        yield
    finally:
        path.write_bytes(original)


@contextmanager
def _mutated_json(path: Path, mutate):
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with _mutated_bytes(path, data):
        yield


@contextmanager
def _missing_dir(path: Path):
    hidden = path.with_name(path.name + ".hidden")
    path.rename(hidden)
    try:
        yield
    finally:
        hidden.rename(path)


# --------------------------------------------------------------------------- #
# Positive acceptance
# --------------------------------------------------------------------------- #
def test_verify_accepts_complete_fixture_and_writes_evidence(tree):
    result = verify_headline(tree.root, deps=_deps(), write=True)

    assert result.status == "written"
    assert result.selection_cells == 64
    assert result.final_cells == 120
    assert result.aggregate_count == 40
    evidence_path = tree.root / HEADLINE_EVIDENCE_FILENAME
    assert evidence_path.is_file()
    assert result.evidence_path == evidence_path

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "complete"
    assert evidence["data_provenance"] == "real_mnist"
    assert evidence["study_id"] == "m003-mnist-headline"
    assert evidence["data_identity"]["canonical_full_dataset"] is True
    assert evidence["data_identity"]["train"]["count"] == 60000
    assert evidence["data_identity"]["test"]["count"] == 10000
    assert evidence["calibration_derivation_provenance"]["data_provenance"] == "production"
    assert evidence["defense_usefulness"]["no_strategic_fixed_kappa"] is True
    assert any("no_strategic" in item for item in evidence["limitations"])
    assert evidence["scope"]["selection"]["cells"] == 64
    assert evidence["scope"]["final"]["cells"] == 120
    assert len(evidence["final"]["aggregates"]) == 40
    assert len(evidence["selection"]["traces"]) == 64
    assert len(evidence["final"]["traces"]) == 120
    # Every cited manifest/trace and aggregate block carries a digest.
    assert all("sha256" in block for block in evidence["evidence_blocks"])
    assert all("digest" in block for block in evidence["final"]["aggregates"])
    assert all("trace_manifest_sha256" in trace for trace in evidence["selection"]["traces"])


def test_verify_is_idempotent_when_every_source_still_matches(tree):
    first = verify_headline(tree.root, deps=_deps(), write=True)
    assert first.status == "written"
    evidence_path = tree.root / HEADLINE_EVIDENCE_FILENAME
    original = evidence_path.read_bytes()

    second = verify_headline(tree.root, deps=_deps(), write=True)
    assert second.status == "duplicate"
    assert evidence_path.read_bytes() == original


def test_verify_no_write_leaves_no_marker(tree):
    result = verify_headline(tree.root, deps=_deps(), write=False)
    assert result.status == "dry-run"
    assert not (tree.root / HEADLINE_EVIDENCE_FILENAME).exists()


def test_hand_computed_final_aggregate_matches_evidence(tree):
    result = verify_headline(tree.root, deps=_deps(), write=False)
    evidence = result.evidence
    block = next(
        row
        for row in evidence["final"]["aggregates"]
        if row["schedule_id"] == _HAND_SCHEDULE and row["policy_id"] == _HAND_POLICY
    )
    assert block["count"] == 3
    assert block["mean_accuracy"] == pytest.approx(0.5, abs=1e-12)
    assert block["std_accuracy"] == pytest.approx(0.2449489742783178, abs=1e-12)
    assert block["iqr_accuracy"] == pytest.approx(0.3, abs=1e-12)
    assert block["mean_loss"] == pytest.approx(1.0, abs=1e-12)
    assert block["worst_node_accuracy"] == pytest.approx(0.2, abs=1e-12)


# --------------------------------------------------------------------------- #
# Negative matrix
# --------------------------------------------------------------------------- #
def test_missing_selection_trace_fails_closed(tree):
    trace_dir = tree.selection_dir / "traces" / "duty_cycle" / "plain_average" / "seed-1101"
    with _missing_dir(trace_dir):
        with pytest.raises(HeadlineVerifyError):
            verify_headline(tree.root, deps=_deps(), write=False)


def test_trace_attack_mismatch_fails_closed(tree):
    """A trace asserting a non-base static attack is rejected.

    schema-v1 traces snapshot the study's *static* base attack (the runtime
    schedule is applied without mutating the frozen config), so the verifier
    must reject a trace whose declared attack disagrees with that base.
    """

    policy_id = "plain_average"
    seed = 1101
    schedule_id = "duty_cycle"
    trace_dir = (
        tree.selection_dir / "traces" / schedule_id / policy_id / f"seed-{seed}"
    )
    panel = {entry.id: entry for entry in tree.study.study.panel}
    config = build_cell_config(
        tree.study.study.base, panel[policy_id].aggregation, seed
    )
    # Same compromised node, but a non-base intensity: only the traced attack
    # declaration differs, so the base-attack check is what must fire.
    weakened = config.model_copy(
        update={
            "attack": config.attack.model_copy(update={"intensity": "weak"})
        }
    )
    shutil.rmtree(trace_dir)
    _write_cell_trace(
        trace_dir, config=weakened, final_accuracy=0.5, final_loss=1.0
    )
    with pytest.raises(HeadlineVerifyError) as info:
        verify_headline(tree.root, deps=_deps(), write=False)
    assert "trace attack disagrees with the study base attack" in str(info.value)


def test_foreign_trace_path_fails_closed(tree):
    path = tree.final_dir / FINAL_MANIFEST_FILENAME

    def mutate(payload):
        payload["entries"][0]["trace_path"] = "traces/duty_cycle/plain_average/seed-2102"
        payload.pop("final_output_digest", None)
        payload["final_output_digest"] = final_output_digest(payload)

    with _mutated_json(path, mutate):
        with pytest.raises(HeadlineVerifyError) as info:
            verify_headline(tree.root, deps=_deps(), write=False)
    assert "canonical" in str(info.value) or "alias" in str(info.value)


def test_path_escape_trace_fails_closed(tree):
    path = tree.final_dir / FINAL_MANIFEST_FILENAME

    def mutate(payload):
        payload["entries"][0]["trace_path"] = "../../../etc/passwd"
        payload.pop("final_output_digest", None)
        payload["final_output_digest"] = final_output_digest(payload)

    with _mutated_json(path, mutate):
        with pytest.raises(HeadlineVerifyError):
            verify_headline(tree.root, deps=_deps(), write=False)


def test_duplicate_trace_reference_fails_closed(tree):
    path = tree.final_dir / FINAL_MANIFEST_FILENAME

    def mutate(payload):
        first = payload["entries"][0]
        payload["entries"][1]["trace_path"] = first["trace_path"]
        payload.pop("final_output_digest", None)
        payload["final_output_digest"] = final_output_digest(payload)

    with _mutated_json(path, mutate):
        with pytest.raises(HeadlineVerifyError) as info:
            verify_headline(tree.root, deps=_deps(), write=False)
    assert "canonical" in str(info.value) or "alias" in str(info.value)


def test_wrong_summary_aggregate_fails_closed(tree):
    path = tree.root / STUDY_DIRNAME / ROBUSTNESS_SUMMARY_FILENAME

    def mutate(payload):
        payload["aggregates"][0]["mean_accuracy"] = 0.123456

    with _mutated_json(path, mutate):
        with pytest.raises(HeadlineVerifyError):
            verify_headline(tree.root, deps=_deps(), write=False)


def test_wrong_embedded_metrics_fails_closed(tree):
    path = tree.final_dir / FINAL_MANIFEST_FILENAME

    def mutate(payload):
        payload["entries"][0]["metrics"]["mean_accuracy"] = 0.999
        # Reseal the manifest so its own self-digest cannot be the excuse.
        payload.pop("final_output_digest", None)
        payload["final_output_digest"] = final_output_digest(payload)

    with _mutated_json(path, mutate):
        with pytest.raises(HeadlineVerifyError) as info:
            verify_headline(tree.root, deps=_deps(), write=False)
    assert "metrics" in str(info.value)


def test_altered_final_self_digest_fails_closed(tree):
    path = tree.final_dir / FINAL_MANIFEST_FILENAME

    def mutate(payload):
        payload["final_output_digest"] = "f" * 64

    with _mutated_json(path, mutate):
        with pytest.raises(HeadlineVerifyError):
            verify_headline(tree.root, deps=_deps(), write=False)


def test_split_overlap_fails_closed(tree):
    path = tree.final_dir / FINAL_MANIFEST_FILENAME

    def mutate(payload):
        payload["final_seeds"] = [1101, 2102, 2103]
        payload.pop("final_output_digest", None)
        payload["final_output_digest"] = final_output_digest(payload)

    with _mutated_json(path, mutate):
        with pytest.raises(HeadlineVerifyError):
            verify_headline(tree.root, deps=_deps(), write=False)


def test_held_out_leakage_in_selection_fails_closed(tree):
    path = tree.selection_dir / SELECTION_MANIFEST_FILENAME

    def mutate(payload):
        payload["entries"][0]["held_out"] = True
        payload.pop("selection_output_digest", None)
        payload["selection_output_digest"] = selection_output_digest(payload)

    with _mutated_json(path, mutate):
        with pytest.raises(HeadlineVerifyError) as info:
            verify_headline(tree.root, deps=_deps(), write=False)
    assert "held_out" in str(info.value)


def test_spoofed_real_label_with_injected_derivation_is_rejected(tree):
    # Use the unpatched production preflight; the fixture artifacts are injected.
    with pytest.raises(HeadlineVerifyError) as info:
        verify_headline(tree.root, deps=_deps(preflight=None), write=False)
    assert "injected" in str(info.value)


def test_incomplete_selection_status_fails_closed(tree):
    path = tree.selection_dir / SELECTION_MANIFEST_FILENAME

    def mutate(payload):
        payload["status"] = "partial"

    with _mutated_json(path, mutate):
        with pytest.raises(HeadlineVerifyError) as info:
            verify_headline(tree.root, deps=_deps(), write=False)
    assert "frozen" in str(info.value)


def test_altered_cache_identity_fails_closed(tree):
    path = tree.data_root / "mnist-provision.json"
    with _mutated_bytes(path, b'{"schema_version":"dfl_sim.mnist_provision/v1"}\n'):
        with pytest.raises(HeadlineVerifyError):
            verify_headline(tree.root, deps=_deps(), write=False)


def test_altered_protocol_digest_fails_closed(tree):
    path = tree.root / FROZEN_PROTOCOL_FILENAME

    def mutate(payload):
        payload["protocol_digest"] = "f" * 64

    with _mutated_json(path, mutate):
        with pytest.raises(HeadlineVerifyError) as info:
            verify_headline(tree.root, deps=_deps(), write=False)
    assert "digest" in str(info.value)


def test_altered_frozen_study_fails_closed(tree):
    path = tree.root / FROZEN_STUDY_FILENAME
    original = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(original)
    payload["held_out_schedule"]["id"] = "held_out_tampered"
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    try:
        with pytest.raises(HeadlineVerifyError):
            verify_headline(tree.root, deps=_deps(), write=False)
    finally:
        path.write_text(original, encoding="utf-8")


def test_missing_selection_cell_fails_closed(tree):
    path = tree.selection_dir / SELECTION_MANIFEST_FILENAME

    def mutate(payload):
        payload["entries"].pop()

    with _mutated_json(path, mutate):
        with pytest.raises(HeadlineVerifyError):
            verify_headline(tree.root, deps=_deps(), write=False)


def test_tampered_final_manifest_self_digest_cannot_bypass_binding(tree):
    path = tree.final_dir / FINAL_MANIFEST_FILENAME

    def mutate(payload):
        payload["held_out_study_digest"] = "0" * 64
        payload.pop("final_output_digest", None)
        payload["final_output_digest"] = final_output_digest(payload)

    with _mutated_json(path, mutate):
        with pytest.raises(HeadlineVerifyError) as info:
            verify_headline(tree.root, deps=_deps(), write=False)
    assert "held_out_study_digest" in str(info.value)


def test_atomic_collision_refuses_to_overwrite_stale_record(tmp_path, baseline_tree):
    root = tmp_path / "headline"
    shutil.copytree(baseline_tree.root, root)
    assert verify_headline(root, deps=_deps(), write=True).status == "written"
    evidence_path = root / HEADLINE_EVIDENCE_FILENAME
    original = evidence_path.read_bytes()

    # Mutate a source so the recomputed evidence necessarily differs.
    summary_path = root / STUDY_DIRNAME / ROBUSTNESS_SUMMARY_FILENAME
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["aggregates"][0]["mean_accuracy"] = 0.111111
    summary_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(HeadlineVerifyError):
        verify_headline(root, deps=_deps(), write=True)
    # The stale complete record is preserved, never overwritten as current.
    assert evidence_path.read_bytes() == original


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_help_exits_zero():
    from dfl_sim.headline_verify import main

    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0


def test_cli_module_help_via_subprocess():
    process = subprocess.run(
        [sys.executable, "-m", "dfl_sim.headline_verify", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0
    assert "output" in process.stdout
