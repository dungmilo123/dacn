"""Strict baseline-study contract and atomic fixture-runner tests (M003/S01/T02).

These tests drive the **production** composition closure for the S01 harness:

    strict study spec -> paired cell configs -> run_from_config -> schema-v1
    TraceWriter -> TraceReader validation -> atomic baseline manifest

Everything is offline and in-process: the study base uses the production strict
parser, the dataset and evaluator reference are tiny in-memory MNIST-shaped
tensors, and the artifact-backed Bayesian cell loads the git-tracked minimal
M002 fixture. No policy is injected, so the real policy factory, round engine,
privileged evaluator, and trajectory writer are all exercised.

The negative cases (duplicate/missing policies, unknown keys, unsafe YAML tags,
non-finite values, budget/selection mismatch, shared-input mismatch, existing
output, missing artifact, middle-cell failure, invalid trace) prove the runner
fails closed and never publishes a success manifest for an incomplete panel.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

import dfl_sim.evaluation as evaluation_module
from dfl_sim import (
    BALANCE_CITATION,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    MANIFEST_SCHEMA_VERSION,
    MAX_STUDY_SEEDS,
    REQUIRED_POLICY_IDS,
    BaselineStudyConfigError,
    EvaluationError,
    TraceReader,
    TrajectoryError,
    build_cell_config,
    compute_shared_input_digest,
    hyperparameter_digest,
    load_bayesian_artifact,
    load_study_config,
    parse_study_text,
    policy_identity,
    run_baseline_study,
    selected_hyperparameters,
    study_digest,
    validate_shared_inputs,
)
from dfl_sim.evaluation import manifest_bytes
from helpers import MNISTShapedDataset, make_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BAYESIAN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"

_TUNED_PROVENANCE = "selection-seed-budget"


# --------------------------------------------------------------------------- #
# Fixture builders
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


def _panel(*, artifact_dir=None, tuned_budget=3, tuned_seed=7, provenance=_TUNED_PROVENANCE):
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


def _mapping(*, seeds=(11,), artifact_dir=None, **overrides):
    mapping = {
        "study_id": "fixture_study",
        "data_provenance": "injected",
        "base": _base().model_dump(mode="json"),
        "seeds": list(seeds),
        "candidate_budget": 3,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
        "panel": _panel(artifact_dir=artifact_dir),
    }
    mapping.update(overrides)
    return mapping


def _study(*, seeds=(11,), artifact_dir=None, **overrides):
    return parse_study_text(
        yaml.safe_dump(_mapping(seeds=seeds, artifact_dir=artifact_dir, **overrides), sort_keys=True)
    )


def _invalid(mapping) -> BaselineStudyConfigError:
    with pytest.raises(BaselineStudyConfigError) as info:
        parse_study_text(yaml.safe_dump(mapping, sort_keys=True))
    return info.value


def _unpaired_dump(config) -> dict:
    payload = config.model_dump(mode="json")
    payload.pop("seed", None)
    payload.pop("aggregation", None)
    return payload


@pytest.fixture(scope="module")
def study_run(tmp_path_factory):
    output = tmp_path_factory.mktemp("baseline-study") / "study"
    result = run_baseline_study(
        _study(),
        output,
        dataset=MNISTShapedDataset(64, seed=1),
        evaluation_dataset=MNISTShapedDataset(32, seed=2),
    )
    return result


# --------------------------------------------------------------------------- #
# Study config: strict parsing and panel contract
# --------------------------------------------------------------------------- #
def test_study_validates_and_canonical_panel_order() -> None:
    study = _study()
    assert study.study_id == "fixture_study"
    assert study.data_provenance == "injected"
    assert tuple(entry.id for entry in study.ordered_panel()) == REQUIRED_POLICY_IDS
    assert study.seeds == (11,)


def test_duplicate_policy_rejected() -> None:
    mapping = _mapping()
    mapping["panel"][1] = dict(mapping["panel"][0])
    message = str(_invalid(mapping))
    assert "panel" in message


def test_missing_policy_rejected() -> None:
    mapping = _mapping()
    mapping["panel"] = mapping["panel"][:5]
    assert "panel" in str(_invalid(mapping))


def test_unknown_policy_id_rejected() -> None:
    mapping = _mapping()
    mapping["panel"][0]["id"] = "mystery"
    assert "unknown policy id" in str(_invalid(mapping))


def test_wrong_aggregation_kind_rejected() -> None:
    mapping = _mapping()
    mapping["panel"][0]["aggregation"] = {"kind": "plain_average"}
    assert "aggregation.kind" in str(_invalid(mapping))


def test_unknown_top_level_key_rejected() -> None:
    mapping = _mapping()
    mapping["unexpected"] = 1
    assert "unexpected" in str(_invalid(mapping))


def test_unknown_panel_key_rejected() -> None:
    mapping = _mapping()
    mapping["panel"][0]["unexpected"] = 1
    assert "unexpected" in str(_invalid(mapping))


def test_unknown_aggregation_kind_rejected() -> None:
    mapping = _mapping()
    mapping["panel"][0]["aggregation"] = {"kind": "mystery"}
    assert "mystery" in str(_invalid(mapping))


def test_unsafe_yaml_tag_rejected() -> None:
    with pytest.raises(BaselineStudyConfigError):
        parse_study_text("study_id: !!python/object/apply:os.system ['echo hacked']")


def test_non_finite_hyperparameter_rejected() -> None:
    mapping = _mapping()
    mapping["panel"][3]["aggregation"]["beta"] = float("nan")
    assert "finite" in str(_invalid(mapping)).lower()


def test_infinite_cutoff_rejected() -> None:
    mapping = _mapping()
    mapping["panel"][2]["aggregation"]["cutoff"] = float("inf")
    assert "less than or equal" in str(_invalid(mapping)).lower()


def test_candidate_budget_mismatch_rejected() -> None:
    mapping = _mapping()
    mapping["panel"][3]["tuning"]["candidate_budget"] = 4
    assert "candidate_budget" in str(_invalid(mapping))


def test_selection_seed_mismatch_rejected() -> None:
    mapping = _mapping()
    mapping["panel"][2]["tuning"]["selection_seed"] = 99
    assert "selection_seed" in str(_invalid(mapping))


def test_parameter_free_policy_cannot_claim_tuning() -> None:
    mapping = _mapping()
    mapping["panel"][0]["tuning"] = {
        "mode": "tuned",
        "candidate_budget": 3,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
    }
    assert "parameter-free" in str(_invalid(mapping))


def test_tunable_policy_cannot_claim_exempt() -> None:
    mapping = _mapping()
    mapping["panel"][3]["tuning"] = {
        "mode": "exempt",
        "candidate_budget": 0,
        "selection_seed": None,
        "selection_provenance": None,
    }
    assert "tuned" in str(_invalid(mapping))


def test_balance_citation_pinned() -> None:
    mapping = _mapping()
    mapping["panel"][4]["citation"]["title"] = "some other paper"
    assert "balance citation" in str(_invalid(mapping)).lower()


def test_external_citation_requires_identifier() -> None:
    mapping = _mapping()
    mapping["panel"][2]["citation"] = {"kind": "external", "title": "no id"}
    assert "identifier" in str(_invalid(mapping))


def test_empty_seeds_rejected() -> None:
    assert "at least one seed" in str(_invalid(_mapping(seeds=())))


def test_duplicate_seeds_rejected() -> None:
    assert "unique" in str(_invalid(_mapping(seeds=(1, 1))))


def test_out_of_range_seed_rejected() -> None:
    assert "outside" in str(_invalid(_mapping(seeds=(2**31,))))


def test_too_many_seeds_rejected() -> None:
    seeds = tuple(range(MAX_STUDY_SEEDS + 1))
    assert "limit" in str(_invalid(_mapping(seeds=seeds)))


def test_injected_provenance_rejects_download() -> None:
    mapping = _mapping()
    mapping["base"]["data"]["download"] = True
    assert "download" in str(_invalid(mapping))


def test_invalid_data_provenance_rejected() -> None:
    mapping = _mapping()
    mapping["data_provenance"] = "synthetic"
    assert "data_provenance" in str(_invalid(mapping))


def test_base_aggregation_must_be_plain_average() -> None:
    mapping = _mapping()
    mapping["base"]["aggregation"] = {"kind": "trust_weighted", "beta": 1.0}
    assert "plain_average" in str(_invalid(mapping))


def test_error_text_is_bounded() -> None:
    mapping = _mapping()
    mapping["study_id"] = "A" * 5000
    message = str(_invalid(mapping))
    assert len(message) < 600


def test_canonical_json_round_trip() -> None:
    study = _study()
    snapshot = json.dumps(study.model_dump(mode="json"))
    assert study_digest(parse_study_text(snapshot)) == study_digest(study)


def test_load_study_config_round_trip(tmp_path: Path) -> None:
    study = _study()
    spec = tmp_path / "study.yaml"
    spec.write_text(yaml.safe_dump(_mapping(), sort_keys=True), encoding="utf-8")
    loaded = load_study_config(spec)
    assert study_digest(loaded) == study_digest(study)


def test_hyperparameter_digest_tracks_values() -> None:
    study = _study()
    trust = next(e for e in study.panel if e.id == "trust_weighted")
    assert hyperparameter_digest(trust.aggregation) == hyperparameter_digest(
        trust.aggregation
    )
    assert selected_hyperparameters(trust.aggregation) == {"beta": 1.0}
    local = next(e for e in study.panel if e.id == "local_only")
    assert selected_hyperparameters(local.aggregation) == {}


def test_policy_identity_scrubs_artifact_dir() -> None:
    study = _study()
    bayesian = next(e for e in study.panel if e.id == "bayesian")
    assert policy_identity(bayesian.aggregation)["artifact_dir"] == "<artifact>"
    assert "artifact_dir" not in selected_hyperparameters(bayesian.aggregation)


def test_balance_default_hyperparameters() -> None:
    study = _study()
    balance = next(e for e in study.panel if e.id == "balance")
    assert selected_hyperparameters(balance.aggregation) == {
        "alpha": 0.5,
        "gamma": 0.3,
        "kappa": 1.0,
    }


# --------------------------------------------------------------------------- #
# Pairing helpers
# --------------------------------------------------------------------------- #
def test_cell_configs_differ_only_in_seed_and_aggregation() -> None:
    study = _study()
    base_dump = _unpaired_dump(study.base)
    for policy in study.ordered_panel():
        config = build_cell_config(study.base, policy.aggregation, 11)
        assert config.seed == 11
        assert config.aggregation.kind == policy.id
        assert _unpaired_dump(config) == base_dump


def test_validate_shared_inputs_detects_mismatch() -> None:
    first = build_cell_config(_base(), _study().panel[0].aggregation, 11)
    second = build_cell_config(_base(rounds=3), _study().panel[1].aggregation, 12)
    with pytest.raises(EvaluationError):
        validate_shared_inputs([first, second])
    assert compute_shared_input_digest(first) != compute_shared_input_digest(second)


def test_validate_shared_inputs_accepts_paired_cells() -> None:
    study = _study()
    configs = [
        build_cell_config(study.base, policy.aggregation, 11)
        for policy in study.ordered_panel()
    ]
    digest = validate_shared_inputs(configs)
    assert digest == compute_shared_input_digest(study.base)


# --------------------------------------------------------------------------- #
# End-to-end fixture study
# --------------------------------------------------------------------------- #
def test_fixture_study_publishes_six_paired_traces(study_run) -> None:
    manifest = study_run.manifest
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["status"] == "complete"
    assert manifest["panel"] == list(REQUIRED_POLICY_IDS)
    assert manifest["seeds"] == [11]
    assert len(study_run.cells) == 6
    ids = [cell.policy_id for cell in study_run.cells]
    assert ids == list(REQUIRED_POLICY_IDS)
    for cell in study_run.cells:
        assert cell.trace_path.is_dir()
        assert cell.relative_trace_path == f"traces/{cell.policy_id}/seed-11"
        entry = cell.manifest_entry
        assert entry["status"] == "complete"
        assert entry["seed"] == 11
        assert entry["data_provenance"] == "injected"
        assert entry["policy_kind"] == cell.policy_id


def test_fixture_traces_are_readable_schema_v1(study_run) -> None:
    for cell in study_run.cells:
        with TraceReader(cell.trace_path) as reader:
            assert reader.schema_version == 1
            assert reader.config.aggregation.kind == cell.policy_id
            assert reader.config.seed == 11


def test_trace_paths_are_unique_and_collision_safe(study_run) -> None:
    relative = [cell.relative_trace_path for cell in study_run.cells]
    assert len(relative) == len(set(relative))
    assert all(".." not in part for part in relative)


def test_manifest_records_citation_budget_and_artifact(study_run) -> None:
    entries = {entry["policy_id"]: entry for entry in study_run.manifest["entries"]}
    balance = entries["balance"]
    assert balance["citation"]["identifier"] == "arXiv:2406.10416v4"
    assert balance["citation"]["algorithm"] == "Algorithm 3"
    assert balance["citation"]["venue"] == "ACM CCS 2024"

    artifact_digest = load_bayesian_artifact(BAYESIAN_FIXTURE).manifest.aggregate_digest
    assert entries["bayesian"]["artifact_digest"] == artifact_digest
    assert entries["local_only"]["artifact_digest"] is None

    for policy_id in REQUIRED_POLICY_IDS:
        tuning = entries[policy_id]["tuning"]
        if policy_id in {"local_only", "plain_average"}:
            assert tuning == {
                "mode": "exempt",
                "candidate_budget": 0,
                "selection_seed": None,
                "selection_provenance": None,
            }
        else:
            assert tuning["mode"] == "tuned"
            assert tuning["candidate_budget"] == 3
            assert tuning["selection_seed"] == 7
            assert tuning["selection_provenance"] == _TUNED_PROVENANCE


def test_manifest_shared_input_and_config_digests(study_run) -> None:
    shared = study_run.manifest["shared_input_digest"]
    assert shared == compute_shared_input_digest(study_run.study.base)
    for entry in study_run.manifest["entries"]:
        assert entry["config_digest"]
        assert entry["hyperparameter_digest"]


def test_manifest_is_deterministic_across_runs(tmp_path: Path) -> None:
    study = _study()
    first = run_baseline_study(
        study,
        tmp_path / "first",
        dataset=MNISTShapedDataset(64, seed=1),
        evaluation_dataset=MNISTShapedDataset(32, seed=2),
    )
    second = run_baseline_study(
        study,
        tmp_path / "second",
        dataset=MNISTShapedDataset(64, seed=1),
        evaluation_dataset=MNISTShapedDataset(32, seed=2),
    )
    first_bytes = (first.manifest_path).read_bytes()
    second_bytes = (second.manifest_path).read_bytes()
    assert first_bytes == second_bytes
    assert first_bytes == manifest_bytes(first.manifest)


def test_two_seeds_produce_twelve_paired_cells(tmp_path: Path) -> None:
    mapping = _mapping(seeds=(11, 12))
    mapping["base"]["rounds"] = 1
    study = parse_study_text(yaml.safe_dump(mapping, sort_keys=True))
    result = run_baseline_study(
        study,
        tmp_path / "two-seed",
        dataset=MNISTShapedDataset(64, seed=1),
        evaluation_dataset=MNISTShapedDataset(32, seed=2),
    )
    assert len(result.cells) == 12
    per_seed = {11: set(), 12: set()}
    for cell in result.cells:
        per_seed[cell.seed].add(cell.policy_id)
    assert per_seed[11] == set(REQUIRED_POLICY_IDS)
    assert per_seed[12] == set(REQUIRED_POLICY_IDS)
    # Same seed -> same shared inputs; only aggregation differs per cell.
    for seed in (11, 12):
        configs = [
            build_cell_config(study.base, policy.aggregation, seed)
            for policy in study.ordered_panel()
        ]
        assert len({compute_shared_input_digest(c) for c in configs}) == 1


def test_progress_hook_reports_every_cell(tmp_path: Path) -> None:
    seen: list[tuple[int, str]] = []
    run_baseline_study(
        _study(seeds=(11, 12)),
        tmp_path / "progress",
        dataset=MNISTShapedDataset(64, seed=1),
        evaluation_dataset=MNISTShapedDataset(32, seed=2),
        progress=lambda seed, policy_id: seen.append((seed, policy_id)),
    )
    assert len(seen) == 12
    assert set(seen) == {
        (seed, policy_id) for seed in (11, 12) for policy_id in REQUIRED_POLICY_IDS
    }


def test_staging_directory_is_cleaned(tmp_path: Path) -> None:
    run_baseline_study(
        _study(),
        tmp_path / "clean",
        dataset=MNISTShapedDataset(64, seed=1),
        evaluation_dataset=MNISTShapedDataset(32, seed=2),
    )
    leftovers = [name for name in os.listdir(tmp_path) if "staging" in name]
    assert leftovers == []


# --------------------------------------------------------------------------- #
# Failure modes: the panel must never publish when it is incomplete
# --------------------------------------------------------------------------- #
def test_existing_output_rejected(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(EvaluationError):
        run_baseline_study(
            _study(),
            output,
            dataset=MNISTShapedDataset(64, seed=1),
            evaluation_dataset=MNISTShapedDataset(32, seed=2),
        )


def test_missing_output_parent_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError):
        run_baseline_study(
            _study(),
            tmp_path / "missing" / "study",
            dataset=MNISTShapedDataset(64, seed=1),
            evaluation_dataset=MNISTShapedDataset(32, seed=2),
        )


def test_injected_provenance_requires_dataset(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError):
        run_baseline_study(_study(), tmp_path / "study")
    with pytest.raises(EvaluationError):
        run_baseline_study(
            _study(),
            tmp_path / "study",
            dataset=MNISTShapedDataset(64, seed=1),
        )


def test_missing_bayesian_artifact_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "study"
    with pytest.raises(EvaluationError) as info:
        run_baseline_study(
            _study(artifact_dir=tmp_path / "does-not-exist"),
            output,
            dataset=MNISTShapedDataset(64, seed=1),
            evaluation_dataset=MNISTShapedDataset(32, seed=2),
        )
    assert "bayesian" in str(info.value)
    assert "11" in str(info.value)
    assert not output.exists()
    assert [n for n in os.listdir(tmp_path) if "staging" in n] == []


def test_middle_cell_failure_is_visible_and_non_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "study"
    real = evaluation_module.run_from_config

    def flaky(config, **kwargs):
        if config.aggregation.kind == "balance":
            raise RuntimeError("injected mid-panel failure")
        return real(config, **kwargs)

    monkeypatch.setattr(evaluation_module, "run_from_config", flaky)
    with pytest.raises(EvaluationError) as info:
        run_baseline_study(
            _study(),
            output,
            dataset=MNISTShapedDataset(64, seed=1),
            evaluation_dataset=MNISTShapedDataset(32, seed=2),
        )
    message = str(info.value)
    assert "balance" in message
    assert "11" in message
    assert not output.exists()
    assert [n for n in os.listdir(tmp_path) if "staging" in n] == []


def test_invalid_trace_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "study"

    def bad_validator(_path):
        raise TrajectoryError("injected invalid trace")

    with pytest.raises(EvaluationError):
        run_baseline_study(
            _study(),
            output,
            dataset=MNISTShapedDataset(64, seed=1),
            evaluation_dataset=MNISTShapedDataset(32, seed=2),
            trace_validator=bad_validator,
        )
    assert not output.exists()
    assert [n for n in os.listdir(tmp_path) if "staging" in n] == []


def test_no_study_manifest_without_complete_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "study"

    def always_fail(config, **kwargs):
        raise RuntimeError("failure")

    monkeypatch.setattr(evaluation_module, "run_from_config", always_fail)
    with pytest.raises(EvaluationError):
        run_baseline_study(
            _study(),
            output,
            dataset=MNISTShapedDataset(64, seed=1),
            evaluation_dataset=MNISTShapedDataset(32, seed=2),
        )
    assert not (output / MANIFEST_FILENAME).exists()
