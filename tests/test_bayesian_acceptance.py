"""End-to-end M002 acceptance: generated artifact, poison divergence, clean gate.

This suite drives the **real production CLI** (``dfl calibrate`` and ``dfl run``)
end to end over deterministic, in-process MNIST-shaped data. Only the dataset
loaders are replaced through their existing injection seams, because the
repository intentionally has no local MNIST cache and the acceptance must stay
fully offline. Everything else is production code: strict config parsing,
topology, local training, fixed attack delivery, artifact publication and
loading, policy construction, privileged evaluation, schema-v1 ``TraceWriter``,
``load_replay``, and the viewer render helpers.

The flow:

1. ``dfl calibrate`` publishes a fresh frozen artifact into the documented
   generated-artifact location. The final-test loader is provably never reached
   during calibration, and the published artifact is loader-valid and distinct
   from the S01 hand fixture.
2. The held-out fixed-poison pair (Bayesian vs. the D026 local validation-loss
   reference) differs only in aggregation. At least one benign ``(round, node)``
   selects a different participant-aligned multiplier vector.
3. The opt-in ``--bayesian-report`` record is finite and bounded, reports exact
   fallback reason counts (including explicit zeros), pruning/candidate metrics,
   and positive, well-ordered timing aggregates.
4. The held-out clean pair (Bayesian vs. M001 plain average) differs only in
   aggregation and satisfies the documented non-degradation gate:
   final-round mean post-commit evaluation loss ``bayesian <= plain + 1e-9``.
   Final mean accuracy is recorded but is not a second gate.
5. Both Bayesian traces replay with the live selected weights and rejections
   visible through ``load_replay`` and the viewer helpers, and carry no extra
   diagnostics file.

The deterministic fixture data is MNIST-shaped ``(N, 1, 28, 28)`` tensors. The
calibration split is large enough for the checked-in calibration schedule
(3 nodes x 16 samples + a 16-example design split). Acceptance seeds are the
checked-in ``20240101``, which is distinct from every calibration episode seed
(those are derived under a separate domain label).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import dfl_sim.calibration_scoring as scoring_module
import dfl_sim.run as run_module
from dfl_sim import (
    FALLBACK_REASONS,
    BayesianArtifactError,
    TraceReader,
    digest_inventory,
    load_bayesian_artifact,
    load_calibration_config,
    load_config,
    load_replay,
    render_view,
    run_from_config,
    select_frame,
    select_frame_node,
)
from dfl_sim.__main__ import BAYESIAN_REPORT_MARKER, main
from dfl_sim.bayesian_artifact import ARTIFACT_FILENAMES
from dfl_sim.calibration import build_schedule
from helpers import MNISTShapedDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATE_CONFIG = REPO_ROOT / "configs" / "calibrate_chain3.yaml"
POISON_BAYES = REPO_ROOT / "configs" / "bayesian_fixed_poison_3node.yaml"
POISON_REFERENCE = (
    REPO_ROOT / "configs" / "validation_reference_fixed_poison_3node.yaml"
)
CLEAN_BAYES = REPO_ROOT / "configs" / "bayesian_clean_3node.yaml"
CLEAN_PLAIN = REPO_ROOT / "configs" / "clean_3node.yaml"
HAND_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"

#: The documented generated-artifact location, relative to the run cwd.
GENERATED_ARTIFACT_REL = Path("runs") / "bayesian_chain3_artifact"

#: Deterministic, offline MNIST-shaped fixture definitions. The values are
#: pinned so the acceptance is bitwise reproducible; the clean split is chosen
#: so the honest local-only Bayesian fallback meets the non-degradation rule.
CALIBRATION_SEED = 11
RUN_SEED = 381610
EVALUATION_SEED = 100648


def _calibration_dataset() -> MNISTShapedDataset:
    # 200 >= 3 * 16 + 16 (node blocks plus the disjoint design split).
    return MNISTShapedDataset(200, seed=CALIBRATION_SEED, separable=True)


def _run_dataset() -> MNISTShapedDataset:
    # 192 == 3 * 64 exactly covers the checked-in run configs' partitions.
    return MNISTShapedDataset(192, seed=RUN_SEED, separable=True)


def _evaluation_dataset() -> MNISTShapedDataset:
    return MNISTShapedDataset(16, seed=EVALUATION_SEED, separable=True)


def _install_loaders(monkeypatch, eval_calls: list[bool]) -> None:
    """Inject offline datasets and make final-test calibration access observable."""

    monkeypatch.setattr(
        scoring_module, "load_mnist_dataset", lambda config, **kw: _calibration_dataset()
    )
    monkeypatch.setattr(
        run_module, "load_mnist_dataset", lambda config, **kw: _run_dataset()
    )

    def _evaluation_loader(config, **kw):
        eval_calls.append(True)
        return _evaluation_dataset()

    monkeypatch.setattr(
        run_module, "load_mnist_evaluation_dataset", _evaluation_loader
    )


def _report_records(stdout: str) -> list[dict]:
    lines = [
        line
        for line in stdout.splitlines()
        if line.startswith(BAYESIAN_REPORT_MARKER)
    ]
    return [json.loads(line[len(BAYESIAN_REPORT_MARKER):]) for line in lines]


def _final_metrics(trace_path: Path) -> tuple[float, float]:
    """Return (mean post-commit loss, mean accuracy) over the final round."""

    model = load_replay(trace_path)
    nodes = model.node_ids
    loss = sum(model.loss_series(node)[-1] for node in nodes) / len(nodes)
    accuracy = sum(model.accuracy_series(node)[-1] for node in nodes) / len(nodes)
    return loss, accuracy


def _aggregation_free_dump(config_path: Path) -> dict:
    """Return the resolved config with only the aggregation block removed."""

    payload = load_config(config_path).model_dump(mode="json")
    payload.pop("aggregation")
    return payload


def _diverging_profiles(
    bayesian_path: Path, reference_path: Path
) -> list[tuple[int, int, tuple[float, ...], tuple[float, ...]]]:
    bayesian = load_replay(bayesian_path)
    reference = load_replay(reference_path)
    diverging: list[tuple[int, int, tuple[float, ...], tuple[float, ...]]] = []
    for round_index in bayesian.rounds:
        frame = select_frame(bayesian, round_index)
        other = select_frame(reference, round_index)
        for state in frame.nodes:
            replayed = other.state_for(state.node_id)
            if state.multipliers != replayed.multipliers:
                diverging.append(
                    (
                        round_index,
                        state.node_id,
                        state.multipliers,
                        replayed.multipliers,
                    )
                )
    return diverging


def test_calibrate_to_run_acceptance(tmp_path: Path, monkeypatch, capsys) -> None:
    """The full calibrate -> poison pair -> clean pair acceptance flow."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    eval_calls: list[bool] = []
    _install_loaders(monkeypatch, eval_calls)

    artifact_dir = tmp_path / GENERATED_ARTIFACT_REL

    # --- 1. Calibrate into a fresh expected directory --------------------- #
    status = main(
        ["calibrate", str(CALIBRATE_CONFIG), "--output", str(artifact_dir)]
    )
    out = capsys.readouterr()
    assert status == 0, out.err
    assert artifact_dir.is_dir()
    assert eval_calls == []  # final-test loader is never reached in calibration
    assert "wrote calibration artifact" in out.out

    # The published artifact is exactly the three schema files, loader-valid,
    # and generated (a different aggregate digest than the S01 hand fixture).
    assert sorted(p.name for p in artifact_dir.iterdir()) == sorted(
        ARTIFACT_FILENAMES
    )
    artifact = load_bayesian_artifact(artifact_dir)
    fixture = load_bayesian_artifact(HAND_FIXTURE)
    assert artifact.manifest.aggregate_digest != fixture.manifest.aggregate_digest
    assert len(artifact.manifest.aggregate_digest) == 64

    # --- 2. Held-out fixed-poison pair ------------------------------------ #
    poison_bayesian = tmp_path / "runs" / "poison_bayesian"
    poison_reference = tmp_path / "runs" / "poison_reference"
    assert (
        main(
            [
                "run",
                str(POISON_BAYES),
                "--output",
                str(poison_bayesian),
                "--bayesian-report",
            ]
        )
        == 0
    )
    report_stdout = capsys.readouterr().out
    assert main(["run", str(POISON_REFERENCE), "--output", str(poison_reference)]) == 0

    # The two configs differ only in aggregation, so divergence is attributable
    # to the selector. At least one *benign* node diverges (node 0 is poisoned).
    assert _aggregation_free_dump(POISON_BAYES) == _aggregation_free_dump(
        POISON_REFERENCE
    )
    diverging = _diverging_profiles(poison_bayesian, poison_reference)
    benign_divergence = [entry for entry in diverging if entry[1] != 0]
    assert benign_divergence, diverging

    # --- 3. Bounded on-policy evidence report ----------------------------- #
    records = _report_records(report_stdout)
    assert len(records) == 1
    report = records[0]
    assert report["schema"] == "dfl_bayesian_evidence_report"
    assert list(report) == sorted(report)  # canonical sorted-key rendering
    assert report["artifact_digest"] == artifact.manifest.aggregate_digest
    assert report["seed"] == load_config(POISON_BAYES).seed
    assert report["rounds"] == 2
    assert report["node_count"] == 3
    assert report["compromised_node"] == 0
    assert report["attack_kappa"] == 5.0

    assert report["selection_count"] == 6  # nodes x rounds
    assert 0.0 < report["coverage"] <= 1.0
    assert math.isfinite(report["coverage"])
    assert 0.0 <= report["mismatch"] <= 1.0
    assert math.isfinite(report["mismatch"])
    assert report["mismatch_group_count"] >= 1

    fallback = report["fallback"]
    assert set(fallback) == set(FALLBACK_REASONS)  # explicit zeros are present
    assert sum(fallback.values()) == report["fallback_total"]
    assert 0.0 <= report["fallback_rate"] <= 1.0
    assert report["fallback_rate"] == report["fallback_total"] / report["selection_count"]

    assert 0 <= report["pruned_count"] <= report["selection_count"]
    assert report["candidate_count_max"] >= 1
    assert report["candidate_count_total"] >= report["candidate_count_max"]
    # Every non-empty selection enumerates within the D022 bound (3 ** degree).
    assert report["candidate_count_max"] <= 3 ** 2

    assert report["elapsed_ns_min"] >= 0
    assert report["elapsed_ns_max"] >= report["elapsed_ns_min"]
    assert report["elapsed_ns_total"] >= report["elapsed_ns_max"]
    assert report["elapsed_ns_total"] > 0

    # --- 4. Held-out clean pair and the non-degradation gate -------------- #
    clean_bayesian = tmp_path / "runs" / "clean_bayesian"
    clean_plain = tmp_path / "runs" / "clean_plain"
    assert main(["run", str(CLEAN_BAYES), "--output", str(clean_bayesian)]) == 0
    assert main(["run", str(CLEAN_PLAIN), "--output", str(clean_plain)]) == 0

    assert _aggregation_free_dump(CLEAN_BAYES) == _aggregation_free_dump(CLEAN_PLAIN)
    bayesian_loss, bayesian_accuracy = _final_metrics(clean_bayesian)
    plain_loss, plain_accuracy = _final_metrics(clean_plain)
    assert bayesian_loss <= plain_loss + 1e-9, (bayesian_loss, plain_loss)
    # Accuracy is recorded but deliberately not a second gate.
    for accuracy in (bayesian_accuracy, plain_accuracy):
        assert 0.0 <= accuracy <= 1.0

    # Acceptance seeds are distinct from every calibration episode seed.
    episode_seeds = {
        spec.seed
        for spec in build_schedule(load_calibration_config(CALIBRATE_CONFIG)).specs
    }
    assert load_config(POISON_BAYES).seed not in episode_seeds
    assert load_config(CLEAN_BAYES).seed not in episode_seeds

    # --- 5. Replay continuity, trace integrity, determinism --------------- #
    for path, kind in (
        (poison_bayesian, "bayesian"),
        (poison_reference, "validation_reference"),
    ):
        with TraceReader(path) as reader:
            assert reader.manifest["aggregation"] == kind
        # Exactly the four schema-v1 files; no diagnostics sidecar appears.
        assert set(digest_inventory(path)) == {
            "config.json",
            "events.jsonl",
            "tensors.npz",
            "manifest.json",
        }
    with TraceReader(poison_bayesian) as reader:
        assert reader.config.aggregation.kind == "bayesian"
        assert reader.config.aggregation.artifact_dir == str(GENERATED_ARTIFACT_REL)

    # Live selected profiles equal the replayed viewer frame values.
    live = run_from_config(
        load_config(POISON_BAYES),
        dataset=_run_dataset(),
        evaluation_dataset=_evaluation_dataset(),
    )
    replay_model = load_replay(poison_bayesian)
    for result in live.rounds:
        frame = select_frame(replay_model, result.round_index)
        for profile in result.profiles:
            state = frame.state_for(profile.node_id)
            assert state.participant_ids == tuple(profile.participant_ids)
            assert state.multipliers == tuple(profile.multipliers)
            assert state.weights == pytest.approx(tuple(profile.weights), abs=1e-12)
            # Rejections are visible as zero multipliers.
            assert sum(1 for value in state.multipliers if value == 0.0) >= 1

    # The viewer render helper exposes the same frame for a real selection.
    round_index, node_id = benign_divergence[0][0], benign_divergence[0][1]
    rendered = render_view(
        replay_model, round_index=round_index, node_id=node_id
    )
    assert rendered.round_index == round_index
    assert rendered.node_id == node_id
    rendered_state = select_frame_node(replay_model, round_index, node_id)
    assert rendered_state.multipliers == benign_divergence[0][2]
    assert any(
        element["data"].get("id", "").startswith("message-")
        for element in rendered.elements
    )

    # Reporting is observability only: a plain re-run is byte-identical.
    poison_repeat = tmp_path / "runs" / "poison_repeat"
    assert main(["run", str(POISON_BAYES), "--output", str(poison_repeat)]) == 0
    assert digest_inventory(poison_bayesian) == digest_inventory(poison_repeat)


# --------------------------------------------------------------------------- #
# Negative / boundary paths (Q5/Q7)
# --------------------------------------------------------------------------- #
def test_bayesian_report_rejected_for_reference_config(
    tmp_path: Path, capsys
) -> None:
    """A report request on a non-Bayesian config fails before publishing."""

    output = tmp_path / "trace"
    status = main(
        [
            "run",
            str(POISON_REFERENCE),
            "--output",
            str(output),
            "--bayesian-report",
        ]
    )
    err = capsys.readouterr().err
    assert status == 1
    assert "bayesian" in err
    assert not output.exists()


def test_missing_generated_artifact_fails_before_trace(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A Bayesian run with an absent generated artifact fails loudly, no trace."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    _install_loaders(monkeypatch, [])
    config = load_config(POISON_BAYES).model_copy(
        update={
            "aggregation": load_config(POISON_BAYES).aggregation.model_copy(
                update={"artifact_dir": "runs/does-not-exist"}
            )
        }
    )
    output = tmp_path / "runs" / "trace"
    with pytest.raises(BayesianArtifactError):
        run_from_config(
            config,
            dataset=_run_dataset(),
            evaluation_dataset=_evaluation_dataset(),
            trace_path=output,
        )
    assert not output.exists()


def test_changed_non_aggregation_field_invalidates_pair() -> None:
    """A mutation outside the aggregation block is detected, not silently ignored."""

    payload = load_config(POISON_BAYES).model_dump(mode="json")
    payload.pop("aggregation")
    reference = _aggregation_free_dump(POISON_REFERENCE)
    assert payload == reference

    mutated = dict(payload)
    mutated["seed"] = int(payload["seed"]) + 1
    assert mutated != reference
    # The changed seed is not the checked-in acceptance seed either.
    assert mutated["seed"] != load_config(POISON_BAYES).seed
