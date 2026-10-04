"""``dfl calibrate`` end-to-end CLI integration tests (T04, R013/R014/R005/R015).

These tests drive the *real* command and the real
:func:`~dfl_sim.calibration.calibrate_from_config` orchestrator over the real
synchronous round engine. The production MNIST loader is replaced with an
in-process MNIST-shaped fixture, so nothing touches the network and the final
test loader is provably never consulted.

Coverage:

* the checked-in ``configs/calibrate_chain3.yaml`` drives a complete calibration
  that publishes exactly the three fixed files, loads unchanged through the S01
  loader, and is byte-identical across two independent runs;
* both clean and attacked strata are actually collected;
* the success summary exposes bounded sampler/support/coverage/calibration
  scalars and the aggregate digest;
* every negative path (malformed config, impossible split, non-finite scorer,
  existing/symlink/aliased output, injected publication failure) returns a
  nonzero status with a concise stderr line, no traceback, and no partial or
  replacement output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

import dfl_sim.calibration_artifact_writer as writer_module
import dfl_sim.calibration_scoring as scoring_module
import dfl_sim.run as run_module
from dfl_sim import load_bayesian_artifact
from dfl_sim.__main__ import build_calibrate_parser, main
from dfl_sim.bayesian_artifact import (
    ARTIFACT_FILENAMES,
    LIKELIHOOD_FILENAME,
    PAYOFF_FILENAME,
)
from dfl_sim.calibration_config import load_calibration_config
from helpers import MNISTShapedDataset, make_config
from test_calibration_config import base_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAIN3_CONFIG = REPO_ROOT / "configs" / "calibrate_chain3.yaml"
_STAGING_PREFIX = ".dfl-calibration-stage-"


def _train(size: int = 64) -> MNISTShapedDataset:
    return MNISTShapedDataset(size, separable=True, seed=11)


def _patch_training_dataset(monkeypatch, dataset: MNISTShapedDataset) -> None:
    """Inject an in-process training split and forbid the final-test loader."""

    monkeypatch.setattr(
        scoring_module, "load_mnist_dataset", lambda config, **kwargs: dataset
    )
    monkeypatch.setattr(
        run_module,
        "load_mnist_evaluation_dataset",
        lambda config, **kwargs: (_ for _ in ()).throw(
            AssertionError("final-test loader must never run during calibration")
        ),
    )


def _spy_strata(monkeypatch) -> list[str]:
    """Wrap the real collection so the CLI's episode strata are observable."""

    original = scoring_module.collect_calibration
    seen: list[str] = []

    def spy(config, **kwargs):
        collection = original(config, **kwargs)
        seen.extend(episode.spec.stratum for episode in collection.episodes)
        return collection

    monkeypatch.setattr(scoring_module, "collect_calibration", spy)
    return seen


def _staging_leftovers(parent: Path) -> list[Path]:
    return sorted(parent.glob(f"{_STAGING_PREFIX}*"))


def _load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Happy path: checked-in config -> real command -> frozen artifact
# --------------------------------------------------------------------------- #
def test_cli_calibrate_publishes_byte_identical_s01_valid_artifacts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_training_dataset(monkeypatch, _train())
    strata = _spy_strata(monkeypatch)

    out1 = tmp_path / "artifact-one"
    out2 = tmp_path / "artifact-two"

    assert main(["calibrate", str(CHAIN3_CONFIG), "--output", str(out1)]) == 0
    first_stdout = capsys.readouterr().out
    assert main(["calibrate", str(CHAIN3_CONFIG), "--output", str(out2)]) == 0
    second_stdout = capsys.readouterr().out

    # Both clean and attacked episodes were collected through the real CLI path
    # (two runs, each 2 clean + 2 attacked).
    assert set(strata) == {"clean", "attacked"}
    assert strata.count("clean") == 4
    assert strata.count("attacked") == 4

    # Exactly the three fixed files, byte-identical across independent runs.
    assert sorted(p.name for p in out1.iterdir()) == sorted(ARTIFACT_FILENAMES)
    for name in ARTIFACT_FILENAMES:
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()

    loaded1 = load_bayesian_artifact(out1)
    loaded2 = load_bayesian_artifact(out2)
    assert loaded1.manifest.aggregate_digest == loaded2.manifest.aggregate_digest
    # The two summaries differ only in their destination path; the digest-bearing
    # diagnostics are identical.
    assert (
        f"aggregate_digest={loaded1.manifest.aggregate_digest}" in first_stdout
    )
    assert (
        f"aggregate_digest={loaded2.manifest.aggregate_digest}" in second_stdout
    )

    manifest = loaded1.manifest
    # Exact provenance/action-space contract from the checked-in config.
    provenance = manifest.provenance
    assert provenance.reference_policy == "plain_average"
    assert provenance.p0 == 0.5
    assert tuple(provenance.mu_a) == (0.0, 2.0, 5.0)
    assert provenance.placement == "uniform_non_isolated"
    assert provenance.max_compromised == 1
    assert provenance.design_split == "clean_global_heldout"
    assert manifest.action_space.omega == 0.5
    assert manifest.action_space.degree_cap == 4
    assert manifest.action_space.tie_tolerance == 1.0e-9
    assert tuple(manifest.action_space.multiplier_set) == (0.0, 0.5, 1.0)

    # Sampler/support counts: 4 episodes x 3 nodes x 2 rounds = 24 records.
    assert manifest.support.calibration_episodes == 4
    assert manifest.support.calibration_rounds == 24
    assert manifest.support.prior_cells == len(manifest.prior)
    assert manifest.support.min_cell_count >= 1

    # Feasible at-most-one-malicious prior, normalized per degree.
    prior_by_degree: dict[int, list] = {}
    for cell in manifest.prior:
        assert len(cell.state) == cell.degree
        assert sum(cell.state) <= 1
        prior_by_degree.setdefault(cell.degree, []).append(cell)
    assert set(prior_by_degree) == {1, 2}
    for cells in prior_by_degree.values():
        assert math.isclose(
            sum(cell.probability for cell in cells), 1.0, abs_tol=1e-9
        )

    # Likelihood rows are nonempty and normalized over the full evidence support.
    likelihood_rows = _load_rows(out1 / LIKELIHOOD_FILENAME)
    assert likelihood_rows
    grouped: dict[tuple, float] = {}
    for row in likelihood_rows:
        key = (row["degree_bin"], row["progress"], row["local_type"])
        grouped[key] = grouped.get(key, 0.0) + row["probability"]
    for total in grouped.values():
        assert math.isclose(total, 1.0, abs_tol=1e-9)

    # Payoff rows are finite and only use the declared multipliers.
    payoff_rows = _load_rows(out1 / PAYOFF_FILENAME)
    assert payoff_rows
    allowed = set(manifest.action_space.multiplier_set)
    for row in payoff_rows:
        assert math.isfinite(row["value"])
        assert row["multiplier"] in allowed
    assert manifest.support.likelihood_cells == len(likelihood_rows)
    assert manifest.support.payoff_cells == len(payoff_rows)

    # Bounded coverage/calibration diagnostics.
    assert 0.0 < manifest.coverage.likelihood <= 1.0
    assert 0.0 < manifest.coverage.payoff <= 1.0
    assert 0.0 <= manifest.coverage.posterior_calibration <= 1.0

    # One stable success summary with the required bounded scalars.
    assert "wrote calibration artifact" in first_stdout
    assert "episodes=4" in first_stdout
    assert "rounds=24" in first_stdout
    assert "prior_cells=5" in first_stdout
    assert "likelihood_coverage=" in first_stdout
    assert "payoff_coverage=" in first_stdout
    assert "posterior_calibration=" in first_stdout
    assert f"aggregate_digest={manifest.aggregate_digest}" in first_stdout
    assert _staging_leftovers(tmp_path) == []


def test_cli_calibrate_does_not_call_the_final_test_loader(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    # The sentinel installed by ``_patch_training_dataset`` raises if the
    # evaluation/final-test loader is ever reached; a clean run proves it is not.
    _patch_training_dataset(monkeypatch, _train())
    output = tmp_path / "artifact"
    assert main(["calibrate", str(CHAIN3_CONFIG), "--output", str(output)]) == 0
    capsys.readouterr()
    assert output.is_dir()


# --------------------------------------------------------------------------- #
# Parser surface
# --------------------------------------------------------------------------- #
def test_build_calibrate_parser_requires_output() -> None:
    parser = build_calibrate_parser()
    with pytest.raises(SystemExit) as info:
        parser.parse_args(["config.yaml"])
    assert info.value.code == 2

    args = parser.parse_args(["config.yaml", "--output", "out"])
    assert args.config == "config.yaml"
    assert args.output == "out"
    # ``--variant`` is optional and defaults to the config's own variant.
    assert args.variant is None


def test_build_calibrate_parser_accepts_variant_choices() -> None:
    parser = build_calibrate_parser()
    for variant in ("full", "no_belief", "no_strategic"):
        args = parser.parse_args(
            ["config.yaml", "--output", "out", "--variant", variant]
        )
        assert args.variant == variant
    with pytest.raises(SystemExit) as info:
        parser.parse_args(
            ["config.yaml", "--output", "out", "--variant", "bogus"]
        )
    assert info.value.code == 2


def test_cli_calibrate_variant_override_revalidates_through_model(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    # The same checked-in full config revalidated as ``no_belief`` publishes a
    # loader-valid uniform-posterior artifact and keeps one success marker.
    _patch_training_dataset(monkeypatch, _train())
    output = tmp_path / "artifact"
    status = main(
        [
            "calibrate",
            str(CHAIN3_CONFIG),
            "--output",
            str(output),
            "--variant",
            "no_belief",
        ]
    )
    out = capsys.readouterr().out
    assert status == 0
    loaded = load_bayesian_artifact(output)
    assert loaded.manifest.derivation.variant == "no_belief"
    assert loaded.manifest.coverage.posterior_calibration_applicable is False
    assert "wrote calibration artifact" in out
    assert f"aggregate_digest={loaded.manifest.aggregate_digest}" in out


# --------------------------------------------------------------------------- #
# Negative paths (Q5/Q7): concise stderr, no partial publication
# --------------------------------------------------------------------------- #
def _assert_failure(status: int, err: str, output: Path) -> None:
    assert status != 0
    assert "error" in err
    assert "Traceback" not in err
    assert not output.exists()


def test_cli_calibrate_missing_config_is_concise(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "artifact"
    status = main(
        ["calibrate", str(tmp_path / "absent.yaml"), "--output", str(output)]
    )
    err = capsys.readouterr().err
    _assert_failure(status, err, output)
    assert "not found" in err


def test_cli_calibrate_malformed_config_is_concise(
    tmp_path: Path, capsys
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_real_calibration_key: 1\n", encoding="utf-8")
    output = tmp_path / "artifact"
    status = main(["calibrate", str(bad), "--output", str(output)])
    err = capsys.readouterr().err
    _assert_failure(status, err, output)
    assert "invalid calibration config" in err


def test_cli_calibrate_impossible_split_is_concise(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    # 32 examples cannot cover 3 nodes x 16 samples plus a 16-example design.
    _patch_training_dataset(monkeypatch, _train(size=32))
    output = tmp_path / "artifact"
    status = main(["calibrate", str(CHAIN3_CONFIG), "--output", str(output)])
    err = capsys.readouterr().err
    _assert_failure(status, err, output)
    assert "too few examples" in err
    assert _staging_leftovers(tmp_path) == []


def test_cli_calibrate_non_finite_scorer_is_concise(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_training_dataset(monkeypatch, _train())
    monkeypatch.setattr(
        scoring_module,
        "make_design_loss_evaluator",
        lambda *args, **kwargs: (lambda *a, **k: float("inf")),
    )
    output = tmp_path / "artifact"
    status = main(["calibrate", str(CHAIN3_CONFIG), "--output", str(output)])
    err = capsys.readouterr().err
    _assert_failure(status, err, output)
    assert "not finite" in err
    assert _staging_leftovers(tmp_path) == []


def test_cli_calibrate_existing_output_is_never_replaced(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_training_dataset(monkeypatch, _train())
    output = tmp_path / "artifact"
    output.mkdir()
    marker = output / "user-data.txt"
    marker.write_text("keep me", encoding="utf-8")

    status = main(["calibrate", str(CHAIN3_CONFIG), "--output", str(output)])
    err = capsys.readouterr().err
    assert status != 0
    assert "already exists" in err
    assert "Traceback" not in err
    assert marker.read_text(encoding="utf-8") == "keep me"
    assert sorted(p.name for p in output.iterdir()) == ["user-data.txt"]
    assert _staging_leftovers(tmp_path) == []


def test_cli_calibrate_symlink_output_parent_is_rejected(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_training_dataset(monkeypatch, _train())
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_parent, target_is_directory=True)

    status = main(
        ["calibrate", str(CHAIN3_CONFIG), "--output", str(link / "artifact")]
    )
    err = capsys.readouterr().err
    assert status != 0
    assert "symlink" in err
    assert "Traceback" not in err
    assert not (real_parent / "artifact").exists()
    assert _staging_leftovers(tmp_path) == []


def test_cli_calibrate_aliased_output_is_rejected(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    # A calibration config whose declared data root is a temp directory: an
    # output *inside* that root is an alias hazard the publisher must refuse.
    data_root = tmp_path / "data"
    data_root.mkdir()
    experiment = make_config(data={"data_root": str(data_root)})
    mapping = base_dict(
        experiment=experiment.model_dump(mode="json"),
        episodes=2,
        p0=0.5,
        design_split_size=16,
    )
    config_path = tmp_path / "alias.yaml"
    config_path.write_text(yaml.safe_dump(mapping, sort_keys=True), encoding="utf-8")

    _patch_training_dataset(monkeypatch, _train(size=128))
    output = data_root / "artifact"
    status = main(["calibrate", str(config_path), "--output", str(output)])
    err = capsys.readouterr().err
    assert status != 0
    assert "inside the data root" in err
    assert "Traceback" not in err
    assert not output.exists()
    assert _staging_leftovers(data_root) == []


def test_cli_calibrate_injected_publication_failure_leaves_no_partial(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _patch_training_dataset(monkeypatch, _train())

    def boom(source, destination):
        raise OSError("injected rename failure")

    monkeypatch.setattr(writer_module, "_rename", boom)
    output = tmp_path / "artifact"

    status = main(["calibrate", str(CHAIN3_CONFIG), "--output", str(output)])
    err = capsys.readouterr().err
    _assert_failure(status, err, output)
    assert "injected rename failure" in err
    assert _staging_leftovers(tmp_path) == []


# --------------------------------------------------------------------------- #
# Config round-trip guard
# --------------------------------------------------------------------------- #
def test_checked_in_config_is_a_valid_calibration_config() -> None:
    config = load_calibration_config(CHAIN3_CONFIG)
    assert config.episodes == 4
    assert config.p0 == 0.5
    assert tuple(config.mu_a) == (0.0, 2.0, 5.0)
    assert config.design_split_size == 16
    assert config.experiment.aggregation.kind == "plain_average"
    assert config.experiment.attack.compromised_node is None
