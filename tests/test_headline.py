"""Offline contract/CLI tests for the real-MNIST headline (S04/T02).

Every test here is fully offline and uses temporary paths or the checked-in
``configs/`` files only. The expensive production stages (three calibrations
plus a 184-cell held-out study) are stubbed behind the explicit
:class:`~dfl_sim.headline.HeadlineDeps` injection seams and the fixtures are
labelled *injected*; nothing in this module is evidence of a real-MNIST run.
The actual downloaded-data execution and independent verification live in
T03/T04.
"""

from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from dfl_sim.ablation import effective_mixture, isolation_statement
from dfl_sim.bayesian_artifact import (
    ABLATION_PROTOCOL_VERSION,
    CALIBRATION_VARIANTS,
    CANDIDATE_PROTOCOL,
    CalibrationDerivation,
)
from dfl_sim.evaluation_config import (
    EXTENDED_POLICY_IDS,
    load_held_out_study_config,
)
from dfl_sim.headline import (
    FROZEN_PROTOCOL_FILENAME,
    FROZEN_STUDY_FILENAME,
    HEADLINE_PROTOCOL_VERSION,
    MNIST_PROVENANCE_FILENAME,
    HeadlineDeps,
    HeadlineError,
    HeadlineStageError,
    StageOutcome,
    SubprocessStageRunner,
    build_calibrate_command,
    build_evaluate_command,
    build_frozen_protocol,
    build_panel,
    build_study,
    expected_cell_counts,
    load_headline_config,
    main,
    parse_headline_text,
    protocol_digest,
    run_headline,
    serialize_study,
    validate_protocol,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HEADLINE_CONFIG = REPO_ROOT / "configs" / "headline_mnist.yaml"

SELECTION_SEEDS = (1101, 1102)
FINAL_SEEDS = (2101, 2102, 2103)
KNOWN_SCHEDULE_IDS = ("duty_cycle", "periodic", "delayed_onset", "random")
HELD_OUT_ID = "held_out"
_VARIANT_TAG = {"full": "a", "no_belief": "b", "no_strategic": "c"}


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
def _protocol():
    return load_headline_config(HEADLINE_CONFIG)


def _mapping() -> dict:
    return yaml.safe_load(HEADLINE_CONFIG.read_text(encoding="utf-8"))


def _parse(mapping: dict):
    return parse_headline_text(yaml.safe_dump(mapping, sort_keys=True))


class _FakeManifest:
    """Injected stand-in for the verified T01 ProvisionManifest."""

    manifest_digest = "f" * 64
    preprocessing_id = "mnist-totensor-v1"
    preprocessing_digest = "0" * 64

    def model_dump(self, mode: str = "json") -> dict:
        return {
            "schema_version": "dfl_sim.mnist_provision/v1",
            "dataset": "mnist",
            "preprocessing_id": self.preprocessing_id,
            "preprocessing_digest": self.preprocessing_digest,
            "manifest_digest": self.manifest_digest,
            "train": {"count": 60000},
            "test": {"count": 10000},
        }


class _ProvisionRecorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def provision(self, data_root: str, *, download: bool = False):
        self.calls.append({"data_root": data_root, "download": download})
        return _FakeManifest()


@contextmanager
def _fake_lease(*args: object, **kwargs: object):
    yield _FakeManifest()


@contextmanager
def _mutating_lease(*args: object, **kwargs: object):
    yield _FakeManifest()
    raise RuntimeError("cache mutated during the lease")


def _derivation(variant: str) -> CalibrationDerivation:
    return CalibrationDerivation(
        protocol_version=ABLATION_PROTOCOL_VERSION,
        variant=variant,
        isolation_statement=isolation_statement(variant),
        source_mu_a=(0.0, 2.0, 5.0),
        effective_mu_a=tuple(effective_mixture(variant)),
        seed=20240101,
        # Shared across the triple so the matched-input identity agrees.
        calibration_config_digest="c" * 64,
        dataset_inputs_digest="d" * 64,
        partition_membership_digest="b" * 64,
        schedule_pairing_digest="e" * 64,
        episodes=32,
        rounds=1,
        design_size=16,
        candidate_protocol=CANDIDATE_PROTOCOL,
        candidate_cap=1,
        data_provenance="production",
    )


def _artifact(variant: str):
    tag = _VARIANT_TAG[variant]
    return types.SimpleNamespace(
        manifest=types.SimpleNamespace(
            derivation=_derivation(variant), aggregate_digest=tag * 64
        )
    )


def _fake_artifact(path: str):
    variant = Path(path).name
    return _artifact(variant)


def _fake_summarize(study_root: str) -> dict:
    directory = Path(study_root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "robustness-summary.json").write_text(
        json.dumps({"schema": "dfl_robustness_summary", "status": "complete"}) + "\n",
        encoding="utf-8",
    )
    return {"status": "complete"}


def _selection_manifest(spec, *, drop_last: bool = False) -> dict:
    known = sorted(entry.id for entry in spec.known_schedules)
    entries = [
        {
            "policy_id": policy_id,
            "seed": seed,
            "schedule_id": schedule_id,
            "held_out": False,
        }
        for seed in sorted(spec.study.seeds)
        for schedule_id in known
        for policy_id in EXTENDED_POLICY_IDS
    ]
    if drop_last:
        entries = entries[:-1]
    return {
        "schema": "dfl_held_out_selection",
        "schema_version": 1,
        "status": "frozen",
        "panel": list(EXTENDED_POLICY_IDS),
        "selection_seeds": sorted(spec.study.seeds),
        "schedule_ids": known,
        "entries": entries,
    }


def _final_manifest(spec, *, drop_last: bool = False) -> dict:
    known = sorted(entry.id for entry in spec.known_schedules)
    schedules = [*known, spec.held_out_schedule.id]
    entries = [
        {
            "policy_id": policy_id,
            "seed": seed,
            "schedule_id": schedule_id,
            "held_out": schedule_id == spec.held_out_schedule.id,
        }
        for seed in sorted(spec.final_seeds)
        for schedule_id in schedules
        for policy_id in EXTENDED_POLICY_IDS
    ]
    if drop_last:
        entries = entries[:-1]
    return {
        "schema": "dfl_held_out_study_manifest",
        "schema_version": 1,
        "phase": "final",
        "status": "complete",
        "panel": list(EXTENDED_POLICY_IDS),
        "final_seeds": sorted(spec.final_seeds),
        "schedule_ids": schedules,
        "entries": entries,
    }


class _FakeStageRunner:
    """Injected stand-in for the two production subprocess stages."""
    def __init__(self, *, fail_on: str | None = None, drop_last_cell: bool = False):
        self.fail_on = fail_on
        self.drop_last_cell = drop_last_cell
        self.calibrate_calls: list[tuple[str, str, str]] = []
        self.evaluate_calls: list[tuple[str, str]] = []

    def calibrate(
        self, config_path: str, destination: str, variant: str, diagnostics_dir: Path
    ) -> StageOutcome:
        self.calibrate_calls.append((config_path, destination, variant))
        label = f"calibrate-{variant}"
        if self.fail_on == label:
            raise HeadlineStageError(label, "injected failure")
        Path(destination).mkdir(parents=True, exist_ok=True)
        return StageOutcome(label, 0, 0, "", "")

    def evaluate(
        self, study_path: str, destination: str, diagnostics_dir: Path
    ) -> StageOutcome:
        self.evaluate_calls.append((study_path, destination))
        if self.fail_on == "evaluate":
            raise HeadlineStageError("evaluate", "injected failure")
        spec = load_held_out_study_config(study_path)
        root = Path(destination)
        (root / "selection").mkdir(parents=True, exist_ok=True)
        (root / "final").mkdir(parents=True, exist_ok=True)
        (root / "selection" / "selection-manifest.json").write_text(
            json.dumps(_selection_manifest(spec, drop_last=self.drop_last_cell)),
            encoding="utf-8",
        )
        (root / "final" / "final-manifest.json").write_text(
            json.dumps(_final_manifest(spec, drop_last=self.drop_last_cell)),
            encoding="utf-8",
        )
        return StageOutcome("evaluate", 0, 0, "", "")


def _noop_verify(_root: str) -> None:
    """Injected stand-in for the independent audit; records nothing."""

    return None


class _RecordingVerify:
    """Injected audit seam that records the audited root and any failure."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self._fail = fail

    def __call__(self, root: str) -> None:
        self.calls.append(root)
        if self._fail:
            raise RuntimeError("injected audit failure")


def _deps(
    stage_runner: _FakeStageRunner,
    *,
    recorder: _ProvisionRecorder | None = None,
    lease=_fake_lease,
    load_artifact=_fake_artifact,
    preflight=lambda study: None,
    summarize=_fake_summarize,
    verify=_noop_verify,
) -> HeadlineDeps:
    recorder = recorder or _ProvisionRecorder()
    return HeadlineDeps(
        provision=recorder.provision,
        cache_lease=lease,
        load_artifact=load_artifact,
        preflight=preflight,
        summarize=summarize,
        verify=verify,
        stage_runner=stage_runner,
        repo_root=REPO_ROOT,
    )


# --------------------------------------------------------------------------- #
# Strict protocol
# --------------------------------------------------------------------------- #
def test_checked_in_protocol_validates():
    protocol = _protocol()
    validate_protocol(protocol)
    assert protocol.study_id == "m003-mnist-headline"
    assert protocol.data_provenance == "real_mnist"
    assert protocol.threads == 1
    assert tuple(protocol.selection_seeds) == SELECTION_SEEDS
    assert tuple(protocol.final_seeds) == FINAL_SEEDS
    assert protocol.candidate_budget == 1


def test_unknown_field_rejected():
    mapping = _mapping()
    mapping["unexpected"] = 1
    with pytest.raises(HeadlineError):
        _parse(mapping)


@pytest.mark.parametrize(
    "mutate,label",
    [
        (lambda m: m["base"].__setitem__("rounds", 5), "rounds"),
        (lambda m: m["base"]["data"].__setitem__("download", True), "download"),
        (lambda m: m["base"]["data"].__setitem__("samples_per_node", 8), "samples"),
        (lambda m: m["base"]["training"].__setitem__("batch_size", 8), "batch"),
        (lambda m: m["base"]["evaluation"].__setitem__("max_samples", 64), "eval"),
        (lambda m: m["base"]["topology"].__setitem__("kind", "line2"), "topology"),
        (lambda m: m["base"].__setitem__("bitwise", False), "bitwise"),
        (lambda m: m.__setitem__("candidate_budget", 2), "budget"),
    ],
)
def test_pinned_dimension_mutations_rejected(mutate, label):
    mapping = _mapping()
    mutate(mapping)
    protocol = _parse(mapping)
    with pytest.raises(HeadlineError):
        validate_protocol(protocol)


def test_seed_split_is_immutable():
    mapping = _mapping()
    mapping["final_seeds"] = [1101, 2101, 2102]  # overlaps selection seed 1101
    protocol = _parse(mapping)
    with pytest.raises(HeadlineError):
        validate_protocol(protocol)


def test_calibration_seed_must_be_separate():
    mapping = _mapping()
    mapping["calibration"]["seed"] = 1101  # collides with a selection seed
    protocol = _parse(mapping)
    with pytest.raises(HeadlineError):
        validate_protocol(protocol)


def test_known_schedules_must_cover_all_intensities():
    mapping = _mapping()
    # Replace the four known schedules with a single strong duty cycle.
    mapping["schedules"] = [mapping["schedules"][0]]
    protocol = _parse(mapping)
    with pytest.raises(HeadlineError):
        validate_protocol(protocol)


def test_held_out_kind_is_reserved():
    mapping = _mapping()
    mapping["held_out_schedule"]["schedule"] = {
        "kind": "duty_cycle",
        "on_rounds": 3,
        "off_rounds": 2,
        "intensity": "strong",
    }
    protocol = _parse(mapping)
    with pytest.raises(HeadlineError):
        validate_protocol(protocol)


# --------------------------------------------------------------------------- #
# Panel / generated study / matrix
# --------------------------------------------------------------------------- #
def test_expected_matrix_is_64_plus_120():
    selection, final = expected_cell_counts(_protocol())
    assert selection == 64
    assert final == 120
    assert selection + final == 184


def test_panel_declares_eight_policies_with_singleton_budget():
    panel = build_panel(
        {variant: f"<{variant}>" for variant in CALIBRATION_VARIANTS},
        candidate_budget=1,
        selection_seed=1101,
        selection_provenance="singleton",
    )
    assert tuple(entry.id for entry in panel) == EXTENDED_POLICY_IDS
    for entry in panel:
        if entry.id in ("local_only", "plain_average"):
            assert entry.tuning.mode == "exempt"
            assert entry.tuning.candidate_budget == 0
        else:
            assert entry.tuning.mode == "tuned"
            assert entry.tuning.candidate_budget == 1


def test_generated_study_pins_matrix_and_artifact_dirs(tmp_path):
    study = build_study(_protocol(), tmp_path / "root", {
        variant: _artifact(variant) for variant in CALIBRATION_VARIANTS
    })
    assert study.study.policy_ids() == EXTENDED_POLICY_IDS
    assert tuple(study.study.seeds) == SELECTION_SEEDS
    assert tuple(study.final_seeds) == FINAL_SEEDS
    assert study.study.candidate_budget == 1
    assert study.study.ablations is not None
    assert len(study.study.ablations.expected_aggregate_digests) == 3
    assert study.study.data_provenance == "real_mnist"
    assert study.study.base.data.download is False

    by_id = {entry.id: entry for entry in study.study.panel}
    for variant in CALIBRATION_VARIANTS:
        policy_id = {
            "full": "bayesian",
            "no_belief": "bayesian_no_belief",
            "no_strategic": "bayesian_no_strategic",
        }[variant]
        expected = str(tmp_path / "root" / "calibration" / variant)
        assert by_id[policy_id].aggregation.artifact_dir == expected


def test_generated_study_round_trips_and_reserves_held_out(tmp_path):
    study = build_study(_protocol(), tmp_path / "root", {
        variant: _artifact(variant) for variant in CALIBRATION_VARIANTS
    })
    text = serialize_study(study)
    reparsed = load_held_out_study_config(
        _write_tmp(tmp_path, text)
    )
    assert tuple(entry.id for entry in reparsed.ordered_known_schedules()) == tuple(
        sorted(KNOWN_SCHEDULE_IDS)
    )
    assert reparsed.held_out_schedule.id == HELD_OUT_ID
    assert reparsed.held_out_schedule.schedule.kind == "held_out"
    assert HELD_OUT_ID not in {entry.id for entry in reparsed.known_schedules}


def _write_tmp(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "frozen-study.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_protocol_digest_is_stable_and_environment_independent():
    assert protocol_digest(_protocol()) == protocol_digest(_protocol())
    assert len(protocol_digest(_protocol())) == 64


def test_frozen_protocol_payload_binds_cache_identity():
    payload = build_frozen_protocol(
        _protocol(),
        source=str(HEADLINE_CONFIG),
        cache_manifest=_FakeManifest(),
        threads=1,
    )
    assert payload["schema"] == HEADLINE_PROTOCOL_VERSION
    assert payload["cache"]["manifest_digest"] == _FakeManifest.manifest_digest
    assert payload["selection_seeds"] == list(SELECTION_SEEDS)
    assert payload["final_seeds"] == list(FINAL_SEEDS)
    assert payload["threads"] == 1


# --------------------------------------------------------------------------- #
# Subprocess wiring / no injection
# --------------------------------------------------------------------------- #
def test_calibrate_command_wiring_has_no_injection():
    command = build_calibrate_command("/py", "cfg.yaml", "out", "full")
    assert command == [
        "/py", "-m", "dfl_sim", "calibrate", "cfg.yaml",
        "--output", "out", "--variant", "full",
    ]
    joined = " ".join(command)
    for forbidden in ("dataset", "trainer", "loader", "policy", "download"):
        assert forbidden not in joined


def test_evaluate_command_wiring_has_no_injection():
    command = build_evaluate_command("/py", "study.yaml", "out")
    assert command == [
        "/py", "-m", "dfl_sim", "evaluate", "study.yaml", "--output", "out",
    ]
    joined = " ".join(command)
    for forbidden in ("dataset", "trainer", "loader", "policy", "download"):
        assert forbidden not in joined


def test_subprocess_stage_bounds_threads_and_succeeds(tmp_path):
    shim = tmp_path / "py.sh"
    shim.write_text('#!/bin/sh\necho "threads=$OMP_NUM_THREADS"\n', encoding="utf-8")
    shim.chmod(0o755)
    runner = SubprocessStageRunner(repo_root=REPO_ROOT, python=str(shim), threads=1)
    outcome = runner.calibrate("cfg", "out", "full", tmp_path / "diag")
    assert outcome.returncode == 0
    assert "threads=1" in outcome.stdout_tail


def test_subprocess_stage_nonzero_writes_bounded_diagnostics(tmp_path):
    shim = tmp_path / "py.sh"
    shim.write_text("#!/bin/sh\necho boom >&2\nexit 3\n", encoding="utf-8")
    shim.chmod(0o755)
    runner = SubprocessStageRunner(repo_root=REPO_ROOT, python=str(shim), threads=1)
    diagnostics = tmp_path / "diag"
    with pytest.raises(HeadlineStageError) as info:
        runner.calibrate("cfg", "out", "full", diagnostics)
    assert "exit 3" in str(info.value)
    payload = json.loads((diagnostics / "calibrate-full.json").read_text(encoding="utf-8"))
    assert payload["returncode"] == 3
    assert payload["stderr_tail"].strip() == "boom"


# --------------------------------------------------------------------------- #
# End-to-end orchestration with injected stages
# --------------------------------------------------------------------------- #
def test_run_headline_happy_path(tmp_path):
    recorder = _ProvisionRecorder()
    runner = _FakeStageRunner()
    deps = _deps(runner, recorder=recorder)
    output = tmp_path / "headline"

    result = run_headline(
        HEADLINE_CONFIG,
        output,
        data_root=tmp_path / "cache",
        download=True,
        deps=deps,
    )

    assert result.selection_cells == 64
    assert result.final_cells == 120
    assert (output / FROZEN_PROTOCOL_FILENAME).is_file()
    assert (output / MNIST_PROVENANCE_FILENAME).is_file()
    assert (output / FROZEN_STUDY_FILENAME).is_file()
    assert (output / "study" / "robustness-summary.json").is_file()
    # headline-evidence.json is T03's independent verification gate.
    assert not (output / "headline-evidence.json").exists()

    # Provisioned exactly once with the requested download policy.
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["download"] is True
    assert recorder.calls[0]["data_root"] == str(tmp_path / "cache")

    # Three calibrations plus one held-out evaluation, no in-process injection.
    assert [call[2] for call in runner.calibrate_calls] == [
        "full", "no_belief", "no_strategic",
    ]
    assert len(runner.evaluate_calls) == 1

    spec = load_held_out_study_config(output / FROZEN_STUDY_FILENAME)
    assert spec.study.policy_ids() == EXTENDED_POLICY_IDS
    assert tuple(spec.study.seeds) == SELECTION_SEEDS
    assert tuple(spec.final_seeds) == FINAL_SEEDS


def test_run_headline_invokes_audit_after_evaluate(tmp_path):
    recorder = _RecordingVerify()
    runner = _FakeStageRunner()
    output = tmp_path / "headline"
    result = run_headline(
        HEADLINE_CONFIG,
        output,
        data_root=tmp_path / "cache",
        deps=_deps(runner, verify=recorder),
    )
    # The audit runs exactly once, after the evaluate stage published the study.
    assert recorder.calls == [str(output)]
    assert (output / "study" / "final" / "final-manifest.json").is_file()
    # The stub does not write evidence, so the result records none.
    assert result.evidence_path is None


def test_run_headline_audit_failure_blocks_success(tmp_path):
    output = tmp_path / "headline"
    recorder = _RecordingVerify(fail=True)
    with pytest.raises(HeadlineError) as info:
        run_headline(
            HEADLINE_CONFIG,
            output,
            data_root=tmp_path / "cache",
            deps=_deps(_FakeStageRunner(), verify=recorder),
        )
    assert "independent headline audit failed closed" in str(info.value)
    assert recorder.calls == [str(output)]
    # The raw study survives for diagnosis but no acceptance marker is written.
    assert (output / "study" / "robustness-summary.json").is_file()
    assert not (output / "headline-evidence.json").exists()


def test_run_headline_no_download_after_provisioning(tmp_path):
    recorder = _ProvisionRecorder()
    deps = _deps(_FakeStageRunner(), recorder=recorder)
    run_headline(
        HEADLINE_CONFIG,
        tmp_path / "headline",
        data_root=tmp_path / "cache",
        download=False,
        deps=deps,
    )
    assert recorder.calls == [{"data_root": str(tmp_path / "cache"), "download": False}]


def test_run_headline_output_collision_fails_closed(tmp_path):
    output = tmp_path / "headline"
    output.mkdir()
    with pytest.raises(HeadlineError):
        run_headline(HEADLINE_CONFIG, output, deps=_deps(_FakeStageRunner()))


def test_run_headline_missing_parent_fails_closed(tmp_path):
    with pytest.raises(HeadlineError):
        run_headline(
            HEADLINE_CONFIG,
            tmp_path / "missing" / "headline",
            deps=_deps(_FakeStageRunner()),
        )


def test_run_headline_provisioning_failure_fails_closed(tmp_path):
    def bad_provision(data_root: str, *, download: bool = False):
        raise RuntimeError("no verified cache and download disabled")

    deps = HeadlineDeps(
        provision=bad_provision,
        cache_lease=_fake_lease,
        load_artifact=_fake_artifact,
        preflight=lambda study: None,
        summarize=_fake_summarize,
        stage_runner=_FakeStageRunner(),
        repo_root=REPO_ROOT,
    )
    output = tmp_path / "headline"
    with pytest.raises(HeadlineError) as info:
        run_headline(HEADLINE_CONFIG, output, data_root=tmp_path / "cache", deps=deps)
    assert "provisioning failed closed" in str(info.value)
    assert not output.exists()


def test_run_headline_stage_failure_keeps_partial_diagnostics(tmp_path):
    runner = _FakeStageRunner(fail_on="calibrate-no_belief")
    output = tmp_path / "headline"
    with pytest.raises(HeadlineError):
        run_headline(
            HEADLINE_CONFIG,
            output,
            data_root=tmp_path / "cache",
            deps=_deps(runner),
        )
    # Frozen protocol/provenance survive for diagnosis; no study is published.
    assert (output / FROZEN_PROTOCOL_FILENAME).is_file()
    assert (output / MNIST_PROVENANCE_FILENAME).is_file()
    assert not (output / FROZEN_STUDY_FILENAME).exists()
    assert not (output / "study").exists()
    assert not (output / "headline-evidence.json").exists()


def test_run_headline_unproven_artifact_fails_closed(tmp_path):
    def unproven(path: str):
        return types.SimpleNamespace(
            manifest=types.SimpleNamespace(derivation=None, aggregate_digest="a" * 64)
        )

    deps = _deps(_FakeStageRunner(), load_artifact=unproven)
    with pytest.raises(HeadlineError) as info:
        run_headline(
            HEADLINE_CONFIG,
            tmp_path / "headline",
            data_root=tmp_path / "cache",
            deps=deps,
        )
    assert "no calibration derivation" in str(info.value)


def test_run_headline_wrong_cell_count_fails_closed(tmp_path):
    runner = _FakeStageRunner(drop_last_cell=True)
    with pytest.raises(HeadlineError) as info:
        run_headline(
            HEADLINE_CONFIG,
            tmp_path / "headline",
            data_root=tmp_path / "cache",
            deps=_deps(runner),
        )
    assert "cells" in str(info.value)


def test_run_headline_cache_mutation_fails_closed(tmp_path):
    deps = _deps(_FakeStageRunner(), lease=_mutating_lease)
    with pytest.raises(HeadlineError) as info:
        run_headline(
            HEADLINE_CONFIG,
            tmp_path / "headline",
            data_root=tmp_path / "cache",
            deps=deps,
        )
    assert "failed closed" in str(info.value)


def test_run_headline_calibration_contract_mismatch_fails_closed(tmp_path):
    mapping = _mapping()
    mapping["calibration"]["seed"] = 999
    protocol_path = tmp_path / "headline.yaml"
    protocol_path.write_text(yaml.safe_dump(mapping, sort_keys=True), encoding="utf-8")
    with pytest.raises(HeadlineError) as info:
        run_headline(
            protocol_path,
            tmp_path / "headline",
            data_root=tmp_path / "cache",
            deps=_deps(_FakeStageRunner()),
        )
    assert "calibration contract mismatch" in str(info.value)
    assert "seed" in str(info.value)


def test_run_headline_rejects_data_root_overlap(tmp_path):
    output = tmp_path / "headline"
    data_root = output / "cache"  # output contains the data root
    with pytest.raises(HeadlineError):
        run_headline(
            HEADLINE_CONFIG,
            output,
            data_root=data_root,
            deps=_deps(_FakeStageRunner()),
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_help_exits_zero():
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0


def test_cli_reports_fail_closed_output_collision(tmp_path, capsys):
    output = tmp_path / "headline"
    output.mkdir()
    code = main([str(HEADLINE_CONFIG), "--output", str(output)])
    assert code == 1
    captured = capsys.readouterr()
    assert "dfl_sim: error:" in captured.err


def test_cli_module_help_via_subprocess():
    import subprocess

    process = subprocess.run(
        [sys.executable, "-m", "dfl_sim.headline", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0
    assert "--output" in process.stdout
