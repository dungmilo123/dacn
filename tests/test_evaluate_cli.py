"""``dfl evaluate`` end-to-end CLI integration tests (M003/S01/T03, R017).

These tests drive the *real* ``dfl evaluate`` command end to end: the argparse
family, the strict baseline-study loader, the fairness-enforcing paired runner,
the unchanged :func:`~dfl_sim.run.run_from_config` composition (real policy
factory, round engine, privileged evaluator, schema-v1 writer), the
TraceReader validation, and the atomic versioned manifest publication.

The production MNIST loaders are replaced with in-process MNIST-shaped
fixtures (the same seam the existing run/calibrate CLI tests use) and the
Bayesian cell loads the git-tracked minimal M002 fixture, so nothing touches
the network. The negative cases (malformed study YAML, existing output,
matched-budget mismatch, missing artifact, injected mid-panel run failure)
prove the command fails closed with a concise contextual stderr line and never
publishes a success manifest.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import dfl_sim.evaluation as evaluation_module
import dfl_sim.run as run_module
from dfl_sim import (
    BALANCE_CITATION,
    REQUIRED_POLICY_IDS,
    TraceReader,
    compute_shared_input_digest,
    load_bayesian_artifact,
    load_study_config,
)
from dfl_sim.__main__ import (
    EVALUATE_PROGRESS_MARKER,
    EVALUATE_SUCCESS_MARKER,
    build_evaluate_parser,
    main,
)
from dfl_sim.evaluation import MANIFEST_FILENAME as BASELINE_MANIFEST_FILENAME
from helpers import MNISTShapedDataset, make_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BAYESIAN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"

_TUNED_PROVENANCE = "selection-seed-budget"
_STAGING_PREFIX = ".baseline_study.staging-"
_DEFAULT_SEED = 11


# --------------------------------------------------------------------------- #
# Fixture builders (real_mnist provenance; loaders are patched in-process)
# --------------------------------------------------------------------------- #
def _base(**overrides):
    defaults = {
        "rounds": 2,
        "data": {"samples_per_node": 8},
        "training": {"local_epochs": 1, "batch_size": 4},
        "evaluation": {"max_samples": 8, "batch_size": 4},
    }
    defaults.update(overrides)
    return make_config(**defaults)


def _panel(
    *,
    artifact_dir=None,
    tuned_budget=3,
    tuned_seed=7,
    provenance=_TUNED_PROVENANCE,
):
    artifact = str(BAYESIAN_FIXTURE if artifact_dir is None else artifact_dir)
    project = {"kind": "project", "title": "project comparator"}
    tuned = {
        "mode": "tuned",
        "candidate_budget": tuned_budget,
        "selection_seed": tuned_seed,
        "selection_provenance": provenance,
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


def _study_mapping(*, seeds=(_DEFAULT_SEED,), artifact_dir=None):
    return {
        "study_id": "cli_fixture_study",
        "data_provenance": "real_mnist",
        "base": _base().model_dump(mode="json"),
        "seeds": list(seeds),
        "candidate_budget": 3,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
        "panel": _panel(artifact_dir=artifact_dir),
    }


def _write_study(tmp_path: Path, mapping, *, name: str = "study.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(mapping, sort_keys=True), encoding="utf-8")
    return path


def _patch_loaders(monkeypatch) -> None:
    monkeypatch.setattr(
        run_module,
        "load_mnist_dataset",
        lambda config, **kwargs: MNISTShapedDataset(64, seed=1),
    )
    monkeypatch.setattr(
        run_module,
        "load_mnist_evaluation_dataset",
        lambda config, **kwargs: MNISTShapedDataset(32, seed=2),
    )


def _staging_leftovers(parent: Path) -> list[Path]:
    return sorted(parent.glob(f"{_STAGING_PREFIX}*"))


# --------------------------------------------------------------------------- #
# Happy path: run the real command once and reuse the published study
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def published(tmp_path_factory):
    spec_dir = tmp_path_factory.mktemp("cli-study-spec")
    spec = _write_study(spec_dir, _study_mapping())
    output = tmp_path_factory.mktemp("cli-study-out") / "study"

    original_train = run_module.load_mnist_dataset
    original_eval = run_module.load_mnist_evaluation_dataset
    run_module.load_mnist_dataset = lambda config, **kw: MNISTShapedDataset(64, seed=1)
    run_module.load_mnist_evaluation_dataset = (
        lambda config, **kw: MNISTShapedDataset(32, seed=2)
    )
    out_buf, err_buf = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            status = main(["evaluate", str(spec), "--output", str(output)])
    finally:
        run_module.load_mnist_dataset = original_train
        run_module.load_mnist_evaluation_dataset = original_eval

    assert status == 0, (
        f"dfl evaluate failed: status={status}\n"
        f"stdout:\n{out_buf.getvalue()}\nstderr:\n{err_buf.getvalue()}"
    )
    manifest = json.loads(
        (output / BASELINE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    return SimpleNamespace(
        status=status,
        spec=spec,
        output=output,
        stdout=out_buf.getvalue(),
        stderr=err_buf.getvalue(),
        manifest=manifest,
        manifest_path=output / BASELINE_MANIFEST_FILENAME,
        entries={entry["policy_id"]: entry for entry in manifest["entries"]},
    )


def test_cli_evaluate_publishes_complete_six_policy_manifest(published) -> None:
    assert published.status == 0
    # One stable stdout marker naming the manifest and output directory.
    assert EVALUATE_SUCCESS_MARKER in published.stdout
    assert str(published.output) in published.stdout
    assert published.manifest_path.name in published.stdout
    assert "policies=6" in published.stdout
    assert "seeds=1" in published.stdout
    assert "cells=6" in published.stdout

    manifest = published.manifest
    assert manifest["schema"] == "dfl_baseline_study_manifest"
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "complete"
    assert manifest["panel"] == list(REQUIRED_POLICY_IDS)
    assert manifest["seeds"] == [_DEFAULT_SEED]
    assert len(manifest["entries"]) == 6
    assert set(published.entries) == set(REQUIRED_POLICY_IDS)


def test_cli_evaluate_progress_is_bounded_per_cell(published) -> None:
    # Progress goes to stderr, one bounded line per (policy, seed) cell, only
    # policy id and seed (no YAML/artifact contents).
    lines = [
        line for line in published.stderr.splitlines()
        if line.startswith(EVALUATE_PROGRESS_MARKER)
    ]
    assert len(lines) == 6
    for policy_id in REQUIRED_POLICY_IDS:
        assert (
            f"{EVALUATE_PROGRESS_MARKER}policy={policy_id} seed={_DEFAULT_SEED}"
            in published.stderr
        )
    assert "artifact_dir" not in published.stderr
    assert str(BAYESIAN_FIXTURE) not in published.stderr


def test_cli_evaluate_emits_one_readable_schema_v1_trace_per_policy(
    published,
) -> None:
    for policy_id in REQUIRED_POLICY_IDS:
        trace = published.output / "traces" / policy_id / f"seed-{_DEFAULT_SEED}"
        assert trace.is_dir()
        with TraceReader(trace) as reader:
            assert reader.schema_version == 1
            assert reader.config.seed == _DEFAULT_SEED
            assert reader.config.aggregation.kind == policy_id
        assert published.entries[policy_id]["trace_path"] == (
            f"traces/{policy_id}/seed-{_DEFAULT_SEED}"
        )


def test_cli_evaluate_manifest_proves_paired_common_inputs(published) -> None:
    study = load_study_config(published.spec)
    manifest = published.manifest
    # One shared-input digest for the whole panel, equal to the base digest.
    assert manifest["shared_input_digest"] == compute_shared_input_digest(study.base)
    # Six aggregation-only variants => six distinct cell config digests.
    digests = {entry["config_digest"] for entry in manifest["entries"]}
    assert len(digests) == len(REQUIRED_POLICY_IDS)
    for entry in manifest["entries"]:
        assert entry["hyperparameter_digest"]
        assert entry["data_provenance"] == "real_mnist"


def test_cli_evaluate_pins_balance_citation_and_artifact(published) -> None:
    balance = published.entries["balance"]
    assert balance["citation"]["identifier"] == "arXiv:2406.10416v4"
    assert balance["citation"]["algorithm"] == "Algorithm 3"
    assert balance["citation"]["venue"] == "ACM CCS 2024"
    assert balance["citation"]["kind"] == "external"

    artifact_digest = load_bayesian_artifact(
        BAYESIAN_FIXTURE
    ).manifest.aggregate_digest
    assert published.entries["bayesian"]["artifact_digest"] == artifact_digest
    assert published.entries["local_only"]["artifact_digest"] is None


def test_cli_evaluate_matches_tunable_budgets_and_exempt_baselines(
    published,
) -> None:
    study = load_study_config(published.spec)
    for policy_id in REQUIRED_POLICY_IDS:
        tuning = published.entries[policy_id]["tuning"]
        if policy_id in {"local_only", "plain_average"}:
            assert tuning == {
                "mode": "exempt",
                "candidate_budget": 0,
                "selection_seed": None,
                "selection_provenance": None,
            }
        else:
            assert tuning["mode"] == "tuned"
            assert tuning["candidate_budget"] == study.candidate_budget
            assert tuning["selection_seed"] == study.selection_seed
            assert tuning["selection_provenance"] == study.selection_provenance


# --------------------------------------------------------------------------- #
# Parser surface
# --------------------------------------------------------------------------- #
def test_build_evaluate_parser_requires_output() -> None:
    parser = build_evaluate_parser()
    with pytest.raises(SystemExit) as info:
        parser.parse_args(["study.yaml"])
    assert info.value.code == 2

    args = parser.parse_args(["study.yaml", "--output", "out"])
    assert args.study == "study.yaml"
    assert args.output == "out"


def test_cli_evaluate_help_lists_study_and_output(capsys) -> None:
    with pytest.raises(SystemExit) as info:
        main(["evaluate", "--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "--output" in out
    assert "DIR" in out
    assert "study" in out.lower()


# --------------------------------------------------------------------------- #
# Negative paths (Q5/Q7): concise stderr, nonzero, no success manifest
# --------------------------------------------------------------------------- #
def _assert_failure(status: int, err: str, output: Path) -> None:
    assert status != 0
    assert "error" in err
    assert "Traceback" not in err
    assert not output.exists()
    assert not (output / BASELINE_MANIFEST_FILENAME).exists()


def test_cli_evaluate_malformed_study_is_concise(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("study_id: [unclosed\n", encoding="utf-8")
    output = tmp_path / "study"

    status = main(["evaluate", str(bad), "--output", str(output)])
    err = capsys.readouterr().err
    _assert_failure(status, err, output)
    assert "invalid YAML" in err


def test_cli_evaluate_missing_study_is_concise(tmp_path: Path, capsys) -> None:
    output = tmp_path / "study"
    status = main(
        ["evaluate", str(tmp_path / "absent.yaml"), "--output", str(output)]
    )
    err = capsys.readouterr().err
    _assert_failure(status, err, output)
    assert "not found" in err


def test_cli_evaluate_existing_output_is_never_replaced(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_loaders(monkeypatch)
    spec = _write_study(tmp_path, _study_mapping())
    output = tmp_path / "study"
    output.mkdir()
    marker = output / "user-data.txt"
    marker.write_text("keep me", encoding="utf-8")

    status = main(["evaluate", str(spec), "--output", str(output)])
    err = capsys.readouterr().err
    assert status != 0
    assert "already exists" in err
    assert "Traceback" not in err
    assert marker.read_text(encoding="utf-8") == "keep me"
    assert sorted(p.name for p in output.iterdir()) == ["user-data.txt"]
    assert _staging_leftovers(tmp_path) == []


def test_cli_evaluate_budget_mismatch_is_rejected(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_loaders(monkeypatch)
    mapping = _study_mapping()
    mapping["panel"][2]["tuning"]["candidate_budget"] = 4
    spec = _write_study(tmp_path, mapping)
    output = tmp_path / "study"

    status = main(["evaluate", str(spec), "--output", str(output)])
    err = capsys.readouterr().err
    _assert_failure(status, err, output)
    assert "candidate_budget" in err


def test_cli_evaluate_missing_bayesian_artifact_fails_closed(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_loaders(monkeypatch)
    spec = _write_study(
        tmp_path, _study_mapping(artifact_dir=tmp_path / "does-not-exist")
    )
    output = tmp_path / "study"

    status = main(["evaluate", str(spec), "--output", str(output)])
    err = capsys.readouterr().err
    _assert_failure(status, err, output)
    assert "bayesian" in err
    assert str(_DEFAULT_SEED) in err
    assert _staging_leftovers(tmp_path) == []


def test_cli_evaluate_injected_run_failure_is_visible_and_non_publishing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_loaders(monkeypatch)
    spec = _write_study(tmp_path, _study_mapping())
    output = tmp_path / "study"
    real = evaluation_module.run_from_config

    def flaky(config, **kwargs):
        if config.aggregation.kind == "balance":
            raise RuntimeError("injected mid-panel failure")
        return real(config, **kwargs)

    monkeypatch.setattr(evaluation_module, "run_from_config", flaky)

    status = main(["evaluate", str(spec), "--output", str(output)])
    err = capsys.readouterr().err
    assert status != 0
    assert "balance" in err
    assert str(_DEFAULT_SEED) in err
    assert "Traceback" not in err
    assert not output.exists()
    assert not (output / BASELINE_MANIFEST_FILENAME).exists()
    assert _staging_leftovers(tmp_path) == []


def test_cli_evaluate_no_success_marker_on_failure(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_loaders(monkeypatch)
    spec = _write_study(tmp_path, _study_mapping())
    output = tmp_path / "study"
    output.mkdir()

    status = main(["evaluate", str(spec), "--output", str(output)])
    captured = capsys.readouterr()
    assert status != 0
    assert EVALUATE_SUCCESS_MARKER not in captured.out


# --------------------------------------------------------------------------- #
# Preserved command dispatch
# --------------------------------------------------------------------------- #
def test_existing_commands_still_dispatch(capsys) -> None:
    with pytest.raises(SystemExit) as run_info:
        main(["run", "--help"])
    assert run_info.value.code == 0
    assert "--output" in capsys.readouterr().out

    with pytest.raises(SystemExit) as calibrate_info:
        main(["calibrate", "--help"])
    assert calibrate_info.value.code == 0
    assert "ARTIFACT_DIR" in capsys.readouterr().out

    with pytest.raises(SystemExit) as legacy_info:
        main(["--help"])
    assert legacy_info.value.code == 0
    out = capsys.readouterr().out
    assert "--log" in out
    assert "run CONFIG --output" in out
