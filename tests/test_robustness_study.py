"""Robustness matrix orchestration tests (M003/S05/T04).

Everything here is offline and deterministic. The canonical real-MNIST protocol
is exercised only for its strict *source* structure (parsing, pinning, path
safety); the executable matrix is driven through explicitly injected seams:

* a reduced, injected :class:`RobustnessProtocol` (one alpha x one placement);
* an in-memory ``MNISTShapedDataset`` and the git-tracked Bayesian fixture; and
* the real production ``run_held_out_study`` runner plus the real independent
  verifier.

No test reads downloaded MNIST, repository ``runs/``, ``data/`` or ``.gsd``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dfl_sim import BALANCE_CITATION
from dfl_sim.attack_schedule import build_schedule
from dfl_sim.evaluation_config import (
    BaselineStudyConfig,
    CitationMetadata,
    HeldOutStudyConfig,
    NamedSchedule,
    PanelPolicy,
    PlacementContract,
    TuningProvenance,
)
from dfl_sim.robustness_study import (
    MATRIX_MANIFEST_FILENAME,
    MATRIX_SCHEMA,
    ROBUSTNESS_PROTOCOL_FILENAME,
    InProcessStudyRunner,
    RobustnessDeps,
    RobustnessProtocol,
    RobustnessStudyError,
    _check_calibration_contract,
    _check_output_destination,
    _check_resume_protocol,
    _check_source_roots,
    alpha_token,
    build_matrix_manifest,
    derive_calibration_config,
    load_robustness_protocol,
    main as robustness_main,
    protocol_digest,
    run_robustness_study,
    validate_protocol,
)
from helpers import MNISTShapedDataset, make_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BAYESIAN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"
CANONICAL_CONFIG = REPO_ROOT / "configs" / "robustness_mnist.yaml"
CALIBRATION_TEMPLATE = REPO_ROOT / "configs" / "calibrate_robustness_mnist.yaml"
SELECTION_SEEDS = (11, 12, 13, 14)
FINAL_SEEDS = (21,)


# --------------------------------------------------------------------------- #
# Injected fixture builders (shared with test_robustness_verify)
# --------------------------------------------------------------------------- #
def _dataset() -> MNISTShapedDataset:
    return MNISTShapedDataset(256, seed=1)


def _evaluation_dataset() -> MNISTShapedDataset:
    return MNISTShapedDataset(64, seed=2)


def injected_protocol(**overrides) -> RobustnessProtocol:
    payload = yaml.safe_load(CANONICAL_CONFIG.read_text(encoding="utf-8"))
    payload.update(
        {
            "study_id": "m003-robustness-injected",
            "data_provenance": "injected",
            "headline_source_root": "<injected>",
            "data_root": "data",
            "alphas": [0.5],
            "placements": ["mid"],
            "rounds": 3,
            "samples_per_node": 2,
            "evaluation_max_samples": 2,
            "evaluation_batch_size": 2,
            "selection_seed": 11,
            "selection_seeds": list(SELECTION_SEEDS),
            "final_seeds": list(FINAL_SEEDS),
        }
    )
    payload.update(overrides)
    return RobustnessProtocol.model_validate(payload)


def _panel() -> tuple[PanelPolicy, ...]:
    project = CitationMetadata(kind="project", title="project comparator")
    tuned = TuningProvenance(
        mode="tuned",
        candidate_budget=1,
        selection_seed=11,
        selection_provenance="fixture",
    )
    exempt = TuningProvenance(mode="exempt", candidate_budget=0)
    return (
        PanelPolicy(
            id="local_only",
            implementation="local only",
            aggregation={"kind": "local_only"},
            citation=project,
            tuning=exempt,
        ),
        PanelPolicy(
            id="plain_average",
            implementation="plain average",
            aggregation={"kind": "plain_average"},
            citation=project,
            tuning=exempt,
        ),
        PanelPolicy(
            id="fixed_similarity",
            implementation="fixed similarity",
            aggregation={
                "kind": "fixed_similarity",
                "cutoff": 0.5,
                "downweight_multiplier": 0.5,
            },
            citation=project,
            tuning=tuned,
        ),
        PanelPolicy(
            id="trust_weighted",
            implementation="trust weighted",
            aggregation={"kind": "trust_weighted", "beta": 1.0},
            citation=project,
            tuning=tuned,
        ),
        PanelPolicy(
            id="balance",
            implementation="faithful BALANCE",
            aggregation={"kind": "balance"},
            citation=CitationMetadata(**BALANCE_CITATION),
            tuning=tuned,
        ),
        PanelPolicy(
            id="bayesian",
            implementation="M002 Bayesian",
            aggregation={
                "kind": "bayesian",
                "artifact_dir": str(BAYESIAN_FIXTURE),
                "omega": 0.5,
                "degree_cap": 4,
                "tie_tolerance": 1.0e-9,
            },
            citation=project,
            tuning=tuned,
        ),
    )


def _known_schedules() -> tuple[NamedSchedule, ...]:
    # duty_cycle + periodic collectively realize zero/weak/strong over 3 rounds.
    return (
        NamedSchedule(
            id="duty_cycle",
            schedule=build_schedule(
                "duty_cycle", on_rounds=1, off_rounds=1, intensity="strong"
            ),
        ),
        NamedSchedule(
            id="periodic",
            schedule=build_schedule(
                "periodic", period=2, strong_rounds=1, weak_rounds=1
            ),
        ),
    )


def _held_out_schedule() -> NamedSchedule:
    return NamedSchedule(id="held_out", schedule=build_schedule("held_out"))


def fixture_study(canonical, protocol, alpha: float, placement: str) -> HeldOutStudyConfig:
    contract = PlacementContract(
        placement=placement,
        graph_kind="watts_strogatz",
        require_varied_degrees=False,
    )
    base = make_config(
        rounds=3,
        data={
            "partition": "dirichlet",
            "dirichlet_alpha": float(alpha),
            "samples_per_node": 2,
            "download": False,
            "data_root": "data",
        },
        topology={"kind": "watts_strogatz", "n": 10, "k": 2, "p": 0.3},
        attack={"compromised_node": 0, "intensity": "strong"},
        training={"batch_size": 2, "local_epochs": 1},
        evaluation={"max_samples": 2, "batch_size": 2},
    )
    study = BaselineStudyConfig(
        study_id=f"m003-robustness-injected-{alpha_token(alpha)}-{placement}",
        data_provenance="injected",
        base=base,
        seeds=SELECTION_SEEDS,
        candidate_budget=1,
        selection_seed=11,
        selection_provenance="fixture",
        panel=_panel(),
        placement=contract,
    )
    return HeldOutStudyConfig(
        study=study,
        final_seeds=FINAL_SEEDS,
        known_schedules=_known_schedules(),
        held_out_schedule=_held_out_schedule(),
    )


def run_injected_matrix(tmp_path: Path):
    """Run the real orchestrator over the reduced injected fixture matrix."""

    protocol = injected_protocol()
    config_path = tmp_path / "robustness-injected.yaml"
    config_path.write_text(
        yaml.safe_dump(protocol.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    output = tmp_path / "matrix"
    deps = RobustnessDeps(
        study_runner=InProcessStudyRunner(
            dataset=_dataset(), evaluation_dataset=_evaluation_dataset()
        ),
        study_builder=fixture_study,
        repo_root=REPO_ROOT,
    )
    result = run_robustness_study(str(config_path), str(output), deps=deps)
    return result, config_path, output


# --------------------------------------------------------------------------- #
# Source protocol
# --------------------------------------------------------------------------- #
def test_canonical_config_parses_to_nine_conditions() -> None:
    protocol = load_robustness_protocol(CANONICAL_CONFIG)
    canonical = validate_protocol(protocol, artifact_root="<validate>")
    conditions = canonical.conditions()
    assert len(conditions) == 9
    assert [c.condition_id for c in conditions] == [
        f"{alpha_token(alpha)}__{placement}"
        for alpha in (0.1, 0.5, 1.0)
        for placement in ("low", "mid", "high")
    ]
    assert protocol.threads == 1
    assert tuple(protocol.selection_seeds) == (1101, 1102)
    assert tuple(protocol.final_seeds) == (2101, 2102, 2103)


def test_weakened_canonical_dimension_is_rejected() -> None:
    protocol = injected_protocol(data_provenance="real_mnist", study_id="m003-canonical-robustness")
    # Injected protocol declares one alpha/one placement; validate must reject it
    # against the frozen canonical constants.
    with pytest.raises(RobustnessStudyError):
        validate_protocol(protocol, artifact_root="<validate>")


def test_alpha_token_matches_canonical_condition_ids() -> None:
    assert alpha_token(0.1) == "a0p1"
    assert alpha_token(0.5) == "a0p5"
    assert alpha_token(1.0) == "a1"


def test_derive_calibration_config_changes_only_alpha() -> None:
    from dfl_sim.calibration_config import load_calibration_config

    template = load_calibration_config(CALIBRATION_TEMPLATE)
    derived = derive_calibration_config(template, 0.5)
    assert derived.experiment.data.dirichlet_alpha == 0.5
    assert derived.experiment.data.partition == "dirichlet"
    before = template.model_dump(mode="json")
    after = derived.model_dump(mode="json")
    before["experiment"]["data"].pop("dirichlet_alpha")
    after["experiment"]["data"].pop("dirichlet_alpha")
    assert before == after


def test_calibration_contract_mismatch_is_rejected() -> None:
    from dfl_sim.calibration_config import CalibrationConfig, load_calibration_config

    protocol = load_robustness_protocol(CANONICAL_CONFIG)
    canonical = validate_protocol(protocol, artifact_root="<validate>")
    template = load_calibration_config(CALIBRATION_TEMPLATE)
    mutated = CalibrationConfig.model_validate(
        {
            **template.model_dump(mode="json"),
            "experiment": {
                **template.experiment.model_dump(mode="json"),
                "data": {
                    **template.experiment.data.model_dump(mode="json"),
                    "samples_per_node": 8,
                },
            },
        }
    )
    with pytest.raises(RobustnessStudyError):
        _check_calibration_contract(protocol, canonical, mutated, CALIBRATION_TEMPLATE)


def test_prepare_calibrations_creates_alpha_parent(
    tmp_path: Path, monkeypatch
) -> None:
    # Regression (M003/S05/T07): the production calibration publisher refuses
    # an absent output parent, so the matrix runner must materialize
    # ``calibration/<alpha>/`` before dispatching each variant. The injected
    # end-to-end matrix skips ablation calibration, so this ordering was only
    # exercised by the real canonical execution.
    from types import SimpleNamespace

    import dfl_sim.robustness_study as rs

    monkeypatch.setattr(rs, "_artifact_derivation", lambda artifact, variant: object())
    monkeypatch.setattr(rs, "_artifact_aggregate", lambda artifact, variant: "a" * 64)
    monkeypatch.setattr(
        rs, "matched_calibration_input_digest", lambda derivation: "b" * 64
    )

    protocol = load_robustness_protocol(CANONICAL_CONFIG)
    destination = tmp_path / "matrix"
    (destination / "generated-configs").mkdir(parents=True)
    (destination / "diagnostics").mkdir(parents=True)
    canonical = validate_protocol(protocol, artifact_root=str(destination))
    seen: list[bool] = []

    def fake_calibrate(config_path, out, variant, diagnostics_dir):
        seen.append(Path(out).parent.is_dir())
        return SimpleNamespace(returncode=0)

    block = rs._prepare_calibrations(
        RobustnessDeps(
            calibrate=fake_calibrate,
            load_artifact=lambda path: SimpleNamespace(),
            repo_root=REPO_ROOT,
        ),
        protocol,
        canonical,
        REPO_ROOT,
        destination,
        None,
        destination / "diagnostics",
        include_ablations=True,
    )
    assert seen and all(seen)
    assert set(block) == {"a0p1", "a0p5", "a1"}
    for token in ("a0p1", "a0p5", "a1"):
        assert (destination / "calibration" / token).is_dir()


def test_injected_protocol_requires_explicit_runner(tmp_path: Path) -> None:
    protocol = injected_protocol()
    config_path = tmp_path / "p.yaml"
    config_path.write_text(yaml.safe_dump(protocol.model_dump(mode="json"), sort_keys=True))
    with pytest.raises(RobustnessStudyError):
        run_robustness_study(
            str(config_path), str(tmp_path / "out"), deps=RobustnessDeps()
        )


def test_public_cli_rejects_injected_protocol(tmp_path: Path, capsys) -> None:
    protocol = injected_protocol()
    config_path = tmp_path / "p.yaml"
    config_path.write_text(yaml.safe_dump(protocol.model_dump(mode="json"), sort_keys=True))
    assert robustness_main([str(config_path), "--output", str(tmp_path / "out")]) == 1
    assert "real_mnist" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Path safety / resume
# --------------------------------------------------------------------------- #
def test_output_collision_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "p.yaml"
    config.write_text("x")
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(RobustnessStudyError):
        _check_output_destination(destination, config)


def test_source_root_confinement_is_rejected(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    destination = data_root / "out"
    with pytest.raises(RobustnessStudyError):
        _check_source_roots(
            destination, data_root=data_root, headline_root=tmp_path / "headline"
        )


def test_resume_protocol_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    resume = tmp_path / "prior"
    resume.mkdir()
    (resume / ROBUSTNESS_PROTOCOL_FILENAME).write_text(
        json.dumps({"protocol_digest": "0" * 64}), encoding="utf-8"
    )
    with pytest.raises(RobustnessStudyError):
        _check_resume_protocol(resume, injected_protocol())


def test_matrix_manifest_digest_excludes_itself() -> None:
    protocol = injected_protocol()
    manifest = build_matrix_manifest(
        protocol=protocol,
        protocol_digest_value=protocol_digest(protocol),
        headline={"data_provenance": "injected"},
        cache=None,
        calibration={},
        conditions=[],
        totals={"conditions": 0, "selection_cells": 0, "final_cells": 0, "aggregates": 0},
    )
    assert manifest["schema"] == MATRIX_SCHEMA
    stored = manifest["matrix_digest"]
    recomputed = dict(manifest)
    recomputed.pop("matrix_digest")
    from dfl_sim.robustness_verify import _sha256_json

    assert stored == _sha256_json(recomputed)


# --------------------------------------------------------------------------- #
# End-to-end injected orchestration
# --------------------------------------------------------------------------- #
def test_injected_matrix_end_to_end(tmp_path: Path) -> None:
    result, _config_path, output = run_injected_matrix(tmp_path)

    assert result.conditions == ("a0p5__mid",)
    assert result.selection_cells == len(SELECTION_SEEDS) * 2 * 6
    assert result.final_cells == len(FINAL_SEEDS) * 3 * 6

    manifest = json.loads((output / MATRIX_MANIFEST_FILENAME).read_text())
    assert manifest["status"] == "complete"
    assert manifest["data_provenance"] == "injected"
    assert manifest["totals"] == {
        "conditions": 1,
        "selection_cells": result.selection_cells,
        "final_cells": result.final_cells,
        "aggregates": 3 * 6,
    }
    condition = manifest["conditions"][0]
    assert condition["condition_id"] == "a0p5__mid"
    assert condition["alpha"] == 0.5
    assert condition["placement"] == "mid"

    evidence = json.loads((output / "matrix-evidence.json").read_text())
    assert evidence["status"] == "verified"
    assert evidence["conditions"][0]["condition_id"] == "a0p5__mid"
    assert evidence["conditions"][0]["messages"] > 0

    frozen = json.loads((output / ROBUSTNESS_PROTOCOL_FILENAME).read_text())
    assert frozen["data_provenance"] == "injected"
    assert frozen["protocol_digest"] == protocol_digest(injected_protocol())
