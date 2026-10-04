"""Runner wiring for the opt-in matched ablation study (S03/T04).

This module drives the **public** study entry points end to end:

* :func:`run_baseline_study` on an eight-policy study,
* :func:`run_selection_phase` / :func:`run_final_phase` and the two-phase
  :func:`run_held_out_study`, plus their resume paths,
* :func:`build_robustness_summary` over the published evidence.

The genuine ``full`` / ``no_belief`` / ``no_strategic`` triple is produced by
the real ``calibrate_from_config`` path over an injected deterministic dataset
and loaded through the production artifact loader. Every negative case proves
that a valid-but-mismatched or stale-across-boundary artifact set is rejected
*before* a single ``run_from_config`` cell executes, that no partial manifest
or success marker escapes, and that the ordinary six-policy outputs are never
reformatted by the optional ablation metadata.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

import dfl_sim.evaluation as evaluation_module
from dfl_sim import (
    BALANCE_CITATION,
    FULL_VARIANT,
    NO_BELIEF_VARIANT,
    NO_STRATEGIC_VARIANT,
    EvaluationError,
    TraceReader,
    calibrate_from_config,
    compute_shared_input_digest,
    load_bayesian_artifact,
    parse_held_out_study_text,
    parse_study_text,
    run_baseline_study,
    run_final_phase,
    run_held_out_study,
    run_selection_phase,
    selection_projection,
    summarize_held_out_study,
)
from dfl_sim.ablation import matched_calibration_input_digest
from dfl_sim.ablation_evaluation import (
    ABLATION_METADATA_KEY,
    AblationEvaluationError,
    preflight_ablation_study,
)
from dfl_sim.bayesian_artifact import (
    LIKELIHOOD_FILENAME,
    MANIFEST_FILENAME,
    PAYOFF_FILENAME,
    canonical_json_bytes,
    compute_aggregate_digest,
    load_bayesian_artifact as load_artifact_strict,
)
from dfl_sim.evaluation import (
    CHECKPOINT_FILENAME,
    FINAL_MANIFEST_FILENAME,
    ROBUSTNESS_SUMMARY_FILENAME,
    SELECTION_MANIFEST_FILENAME,
    build_robustness_summary,
    manifest_bytes,
)
from dfl_sim.evaluation_config import (
    ABLATION_VARIANT_BY_POLICY_ID,
    ARTIFACT_POLICY_IDS,
    EXTENDED_POLICY_IDS,
    REQUIRED_POLICY_IDS,
)
from helpers import MNISTShapedDataset, make_config
from test_calibration_config import base_dict, parse_dict

VARIANTS = (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT)
SEED = 20240101
EPISODES = 8
DESIGN_SPLIT = 8
_TUNED_PROVENANCE = "selection-seed-budget"


# --------------------------------------------------------------------------- #
# Genuine production calibration triple
# --------------------------------------------------------------------------- #
def _calibration_experiment(seed: int = SEED):
    return make_config(
        seed=seed,
        rounds=1,
        training={"local_epochs": 1, "batch_size": 8},
        data={"samples_per_node": 8},
        evaluation={"max_samples": 8, "batch_size": 8},
    )


def _calibration_config(variant: str, *, seed: int = SEED):
    return parse_dict(
        base_dict(
            experiment=_calibration_experiment(seed).model_dump(mode="json"),
            episodes=EPISODES,
            p0=0.5,
            design_split_size=DESIGN_SPLIT,
            variant=variant,
        )
    )


@pytest.fixture(scope="module")
def triple(tmp_path_factory):
    """Calibrate and load the genuine full/no-belief/no-strategic triple once."""

    dataset = MNISTShapedDataset(400, separable=True, seed=11)
    root = tmp_path_factory.mktemp("t04-ablation-triple")
    paths: dict[str, str] = {}
    artifacts: dict[str, object] = {}
    for variant in VARIANTS:
        dest = root / variant
        calibrate_from_config(_calibration_config(variant), str(dest), dataset=dataset)
        paths[variant] = str(dest)
        artifacts[variant] = load_bayesian_artifact(dest)
    return {"paths": paths, "artifacts": artifacts}


@pytest.fixture(scope="module")
def dataset() -> MNISTShapedDataset:
    return MNISTShapedDataset(32, separable=True, seed=3)


# --------------------------------------------------------------------------- #
# Study mapping builders
# --------------------------------------------------------------------------- #
def _base(**overrides):
    defaults = {
        "rounds": 1,
        "data": {"samples_per_node": 4},
        "training": {"local_epochs": 1, "batch_size": 4},
        "evaluation": {"max_samples": 4, "batch_size": 4},
        "attack": {"compromised_node": 0, "intensity": "strong"},
    }
    defaults.update(overrides)
    return make_config(**defaults)


def _tuned() -> dict:
    return {
        "mode": "tuned",
        "candidate_budget": 3,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
    }


def _exempt() -> dict:
    return {
        "mode": "exempt",
        "candidate_budget": 0,
        "selection_seed": None,
        "selection_provenance": None,
    }


def _project() -> dict:
    return {"kind": "project", "title": "project comparator"}


def _bayesian_entry(artifact: str, *, id_: str, implementation: str) -> dict:
    return {
        "id": id_,
        "implementation": implementation,
        "aggregation": {
            "kind": "bayesian",
            "artifact_dir": artifact,
            "omega": 0.5,
            "degree_cap": 4,
            "tie_tolerance": 1e-9,
        },
        "citation": _project(),
        "tuning": _tuned(),
    }


def _eight_panel(full: str, no_belief: str, no_strategic: str) -> list[dict]:
    project = _project()
    tuned = _tuned()
    return [
        {
            "id": "local_only",
            "implementation": "local only",
            "aggregation": {"kind": "local_only"},
            "citation": project,
            "tuning": _exempt(),
        },
        {
            "id": "plain_average",
            "implementation": "plain average",
            "aggregation": {"kind": "plain_average"},
            "citation": project,
            "tuning": _exempt(),
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
        _bayesian_entry(full, id_="bayesian", implementation="M002 Bayesian"),
        _bayesian_entry(
            no_belief,
            id_="bayesian_no_belief",
            implementation="uniform-posterior ablation",
        ),
        _bayesian_entry(
            no_strategic,
            id_="bayesian_no_strategic",
            implementation="fixed-attacker ablation",
        ),
    ]


def _pinned(artifacts) -> dict[str, str]:
    return {
        variant: artifacts[variant].manifest.aggregate_digest for variant in VARIANTS
    }


def _matched(artifacts) -> str:
    return matched_calibration_input_digest(
        artifacts[FULL_VARIANT].manifest.derivation
    )


def _study_mapping(
    triple,
    *,
    base=None,
    seeds=(11,),
    full: str | None = None,
    no_belief: str | None = None,
    no_strategic: str | None = None,
    pinned: dict[str, str] | None = None,
    matched: str | None = None,
    panel: list[dict] | None = None,
    **overrides,
) -> dict:
    full = full if full is not None else triple["paths"][FULL_VARIANT]
    no_belief = (
        no_belief if no_belief is not None else triple["paths"][NO_BELIEF_VARIANT]
    )
    no_strategic = (
        no_strategic
        if no_strategic is not None
        else triple["paths"][NO_STRATEGIC_VARIANT]
    )
    if pinned is None:
        pinned = _pinned(triple["artifacts"])
    if matched is None:
        matched = _matched(triple["artifacts"])
    mapping = {
        "study_id": "t04_ablation_study",
        "data_provenance": "injected",
        "base": (base if base is not None else _base()).model_dump(mode="json"),
        "seeds": list(seeds),
        "candidate_budget": 3,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
        "panel": (
            panel
            if panel is not None
            else _eight_panel(full, no_belief, no_strategic)
        ),
        "ablations": {
            "matched_calibration_input_digest": matched,
            "expected_aggregate_digests": [
                {"variant": variant, "expected_aggregate_digest": pinned[variant]}
                for variant in VARIANTS
            ],
        },
    }
    mapping.update(overrides)
    return mapping


def _study(triple, **kwargs):
    return parse_study_text(yaml.safe_dump(_study_mapping(triple, **kwargs)))


def _held_out_mapping(
    triple,
    *,
    selection_seeds=(11, 12, 13),
    final_seeds=(21, 22),
    base=None,
    **kwargs,
) -> dict:
    return {
        "study": _study_mapping(
            triple,
            base=(base if base is not None else _base(rounds=6)),
            seeds=selection_seeds,
            **kwargs,
        ),
        "final_seeds": list(final_seeds),
        "known_schedules": [
            {"id": "periodic_known", "schedule": {"kind": "periodic"}}
        ],
        "held_out_schedule": {
            "id": "held_out_schedule",
            "schedule": {"kind": "held_out"},
        },
    }


def _held_out(triple, **kwargs):
    return parse_held_out_study_text(
        yaml.safe_dump(_held_out_mapping(triple, **kwargs))
    )


def _swapped(triple, **kwargs):
    """A valid-to-parse study whose ablation paths are swapped (variant mismatch)."""

    return _study(
        triple,
        no_belief=triple["paths"][NO_STRATEGIC_VARIANT],
        no_strategic=triple["paths"][NO_BELIEF_VARIANT],
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Loader-valid artifact mutation helper
# --------------------------------------------------------------------------- #
def _mutate_artifact(source: Path, dest: Path, mutate) -> str:
    """Copy ``source`` to ``dest``, apply ``mutate``, and reseal all digests."""

    dest.mkdir(parents=True)
    manifest = json.loads((source / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    likelihood = json.loads((source / LIKELIHOOD_FILENAME).read_text(encoding="utf-8"))
    payoff = json.loads((source / PAYOFF_FILENAME).read_text(encoding="utf-8"))
    mutate(manifest, likelihood, payoff)

    likelihood_bytes = canonical_json_bytes(likelihood)
    payoff_bytes = canonical_json_bytes(payoff)
    manifest["payloads"] = {
        LIKELIHOOD_FILENAME: {
            "size_bytes": len(likelihood_bytes),
            "sha256": hashlib.sha256(likelihood_bytes).hexdigest(),
        },
        PAYOFF_FILENAME: {
            "size_bytes": len(payoff_bytes),
            "sha256": hashlib.sha256(payoff_bytes).hexdigest(),
        },
    }
    manifest["aggregate_digest"] = compute_aggregate_digest(manifest)
    (dest / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
    (dest / LIKELIHOOD_FILENAME).write_bytes(likelihood_bytes)
    (dest / PAYOFF_FILENAME).write_bytes(payoff_bytes)
    return str(dest)


# --------------------------------------------------------------------------- #
# Shared published fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def baseline_run(triple, dataset, tmp_path_factory):
    study = _study(triple)
    output = tmp_path_factory.mktemp("t04-baseline") / "baseline"
    result = run_baseline_study(
        study, output, dataset=dataset, evaluation_dataset=dataset
    )
    return {"study": study, "result": result, "output": output}


@pytest.fixture(scope="module")
def held_out_run(triple, dataset, tmp_path_factory):
    spec = _held_out(triple)
    output = tmp_path_factory.mktemp("t04-heldout") / "study"
    result = run_held_out_study(
        spec, output, dataset=dataset, evaluation_dataset=dataset
    )
    return {"spec": spec, "result": result, "output": output}


def _entry(manifest, policy_id: str, seed: int | None = None, schedule_id=None):
    matches = [
        entry
        for entry in manifest["entries"]
        if entry["policy_id"] == policy_id
        and (seed is None or entry["seed"] == seed)
        and (schedule_id is None or entry["schedule_id"] == schedule_id)
    ]
    assert len(matches) == 1, (policy_id, seed, schedule_id, len(matches))
    return matches[0]


def _spy(monkeypatch):
    calls: list[str] = []
    real = evaluation_module.run_from_config

    def counting(config, **kwargs):
        calls.append(config.aggregation.kind)
        return real(config, **kwargs)

    monkeypatch.setattr(evaluation_module, "run_from_config", counting)
    return calls


# --------------------------------------------------------------------------- #
# Preflight sanity
# --------------------------------------------------------------------------- #
def test_preflight_accepts_genuine_triple(triple) -> None:
    summary = preflight_ablation_study(_study(triple))
    assert summary.matched_policy_ids == ARTIFACT_POLICY_IDS
    assert summary.matched_calibration_input_digest == _matched(triple["artifacts"])
    for policy_id in ARTIFACT_POLICY_IDS:
        assert summary.artifact(policy_id).manifest.aggregate_digest == (
            triple["artifacts"][
                ABLATION_VARIANT_BY_POLICY_ID[policy_id]
            ].manifest.aggregate_digest
        )


# --------------------------------------------------------------------------- #
# Eight-policy baseline publication
# --------------------------------------------------------------------------- #
def test_baseline_eight_policy_manifest_propagates_identity(triple, baseline_run) -> None:
    manifest = baseline_run["result"].manifest
    assert manifest["panel"] == list(EXTENDED_POLICY_IDS)
    assert len(manifest["entries"]) == len(EXTENDED_POLICY_IDS)
    assert manifest["status"] == "complete"

    ablation = manifest[ABLATION_METADATA_KEY]
    assert ablation["matched_calibration_input_digest"] == _matched(
        triple["artifacts"]
    )
    assert ablation["assumption_version"] == "calibration-derivation-v1"
    assert "not fully isolated" in ablation["fixed_behavior_limitation"]
    assert set(ablation["ablated_aggregate_digests"]) == {
        "bayesian_no_belief",
        "bayesian_no_strategic",
    }
    assert set(ablation["variants"]) == set(ARTIFACT_POLICY_IDS)

    for policy_id in ARTIFACT_POLICY_IDS:
        entry = _entry(manifest, policy_id, seed=11)
        assert entry[ABLATION_METADATA_KEY]["variant"] == (
            ABLATION_VARIANT_BY_POLICY_ID[policy_id]
        )
        assert entry[ABLATION_METADATA_KEY]["assumption_version"] == (
            ablation["assumption_version"]
        )
        assert entry[ABLATION_METADATA_KEY]["matched_calibration_input_digest"] == (
            ablation["matched_calibration_input_digest"]
        )
    for policy_id in ("local_only", "plain_average", "balance"):
        assert ABLATION_METADATA_KEY not in _entry(manifest, policy_id, seed=11)

    # The result-level projection matches the published manifest block.
    assert baseline_run["result"].ablation == ablation

    # Every artifact-backed cell carries the pinned aggregate digest in its
    # manifest entry, and each trace revalidates under the strict reader.
    for policy_id in ARTIFACT_POLICY_IDS:
        entry = _entry(manifest, policy_id, seed=11)
        variant = ABLATION_VARIANT_BY_POLICY_ID[policy_id]
        assert entry["artifact_digest"] == (
            triple["artifacts"][variant].manifest.aggregate_digest
        )
    for cell in baseline_run["result"].cells:
        with TraceReader(cell.trace_path) as reader:
            assert reader.schema_version == 1
            assert reader.config.seed == cell.seed


def test_baseline_eight_policy_is_byte_deterministic(triple, dataset, tmp_path) -> None:
    study = _study(triple)
    first = run_baseline_study(
        study, tmp_path / "a", dataset=dataset, evaluation_dataset=dataset
    )
    second = run_baseline_study(
        study, tmp_path / "b", dataset=dataset, evaluation_dataset=dataset
    )
    assert manifest_bytes(first.manifest) == manifest_bytes(second.manifest)
    assert first.ablation == second.ablation


# --------------------------------------------------------------------------- #
# Fail-closed preflight on every public entry point
# --------------------------------------------------------------------------- #
def test_baseline_rejects_mismatched_triple_before_any_cell(
    triple, dataset, tmp_path, monkeypatch
) -> None:
    study = _swapped(triple)
    calls = _spy(monkeypatch)
    output = tmp_path / "baseline"
    with pytest.raises(AblationEvaluationError) as info:
        run_baseline_study(
            study, output, dataset=dataset, evaluation_dataset=dataset
        )
    assert info.value.validation_class == "variant_mismatch"
    assert calls == []
    assert not output.exists()
    assert [n for n in tmp_path.iterdir() if "staging" in n.name] == []


def test_selection_rejects_mismatched_triple_before_any_cell(
    triple, dataset, tmp_path, monkeypatch
) -> None:
    spec = _held_out(triple, no_belief=triple["paths"][NO_STRATEGIC_VARIANT],
                     no_strategic=triple["paths"][NO_BELIEF_VARIANT])
    calls = _spy(monkeypatch)
    output = tmp_path / "selection"
    with pytest.raises(AblationEvaluationError) as info:
        run_selection_phase(
            selection_projection(spec),
            output,
            dataset=dataset,
            evaluation_dataset=dataset,
        )
    assert info.value.validation_class == "variant_mismatch"
    assert calls == []
    assert not output.exists()


def test_final_rejects_mismatched_triple_before_any_cell(
    triple, held_out_run, dataset, tmp_path, monkeypatch
) -> None:
    mismatched = _held_out(
        triple,
        no_belief=triple["paths"][NO_STRATEGIC_VARIANT],
        no_strategic=triple["paths"][NO_BELIEF_VARIANT],
    )
    calls = _spy(monkeypatch)
    output = tmp_path / "final"
    with pytest.raises(AblationEvaluationError) as info:
        run_final_phase(
            mismatched,
            held_out_run["result"].selection_path,
            output,
            dataset=dataset,
            evaluation_dataset=dataset,
        )
    assert info.value.validation_class == "variant_mismatch"
    assert calls == []
    assert not output.exists()


def test_held_out_rejects_mismatched_triple_before_any_cell(
    triple, dataset, tmp_path, monkeypatch
) -> None:
    spec = _held_out(
        triple,
        no_belief=triple["paths"][NO_STRATEGIC_VARIANT],
        no_strategic=triple["paths"][NO_BELIEF_VARIANT],
    )
    calls = _spy(monkeypatch)
    output = tmp_path / "study"
    with pytest.raises(AblationEvaluationError) as info:
        run_held_out_study(
            spec, output, dataset=dataset, evaluation_dataset=dataset
        )
    assert info.value.validation_class == "variant_mismatch"
    assert calls == []
    assert not output.exists()
    assert [n for n in tmp_path.iterdir() if "staging" in n.name] == []


# --------------------------------------------------------------------------- #
# Two-phase eight-policy evidence
# --------------------------------------------------------------------------- #
def test_held_out_eight_policy_selection_and_final_identity(held_out_run) -> None:
    selection = held_out_run["result"].selection_result
    final = held_out_run["result"].final_result

    assert selection.manifest["panel"] == list(EXTENDED_POLICY_IDS)
    assert final.manifest["panel"] == list(EXTENDED_POLICY_IDS)
    assert selection.manifest[ABLATION_METADATA_KEY] == final.manifest[
        ABLATION_METADATA_KEY
    ]
    assert held_out_run["result"].ablation == final.manifest[ABLATION_METADATA_KEY]

    for manifest in (selection.manifest, final.manifest):
        for entry in manifest["entries"]:
            if entry["policy_id"] in ARTIFACT_POLICY_IDS:
                assert entry[ABLATION_METADATA_KEY]["variant"] == (
                    ABLATION_VARIANT_BY_POLICY_ID[entry["policy_id"]]
                )
            else:
                assert ABLATION_METADATA_KEY not in entry

    # Selection and final splits are disjoint and the reserved schedule is
    # only present in the final phase.
    assert set(selection.manifest["selection_seeds"]).isdisjoint(
        set(final.manifest["final_seeds"])
    )
    selection_schedules = set(selection.manifest["schedule_ids"])
    assert selection.manifest["panel"] == list(EXTENDED_POLICY_IDS)
    assert any(entry["held_out"] for entry in final.manifest["entries"])
    assert final.manifest["held_out_schedule_id"] not in selection_schedules

    # The published robustness summary carries the ablation identity through
    # both the envelope and the artifact-backed aggregates.
    summary = json.loads(
        (held_out_run["output"] / ROBUSTNESS_SUMMARY_FILENAME).read_text()
    )
    assert summary["panel"] == list(EXTENDED_POLICY_IDS)
    assert summary[ABLATION_METADATA_KEY] == final.manifest[ABLATION_METADATA_KEY]
    for aggregate in summary["aggregates"]:
        if aggregate["policy_id"] in ARTIFACT_POLICY_IDS:
            assert aggregate[ABLATION_METADATA_KEY]["variant"] == (
                ABLATION_VARIANT_BY_POLICY_ID[aggregate["policy_id"]]
            )
        else:
            assert ABLATION_METADATA_KEY not in aggregate
    # Re-summarizing from disk reproduces the same deterministic evidence.
    assert summarize_held_out_study(held_out_run["output"]) == summary


def test_full_held_out_resume_reuses_every_cell(
    held_out_run, dataset, tmp_path, monkeypatch
) -> None:
    calls = _spy(monkeypatch)
    output = tmp_path / "resumed"
    result = run_held_out_study(
        held_out_run["spec"],
        output,
        dataset=dataset,
        evaluation_dataset=dataset,
        resume_from=held_out_run["output"],
    )
    assert calls == []
    assert all(
        entry.get("reused") is True
        for entry in (
            list(result.selection_result.manifest["entries"])
            + list(result.final_result.manifest["entries"])
        )
    )
    # Reused traces are byte-for-byte identical to the clean run.
    for clean, resumed in zip(
        held_out_run["result"].final_result.cells, result.final_result.cells
    ):
        assert _trace_bytes(clean.trace_path) == _trace_bytes(resumed.trace_path)


def _trace_bytes(trace_dir: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(Path(trace_dir).rglob("*.json*"))
        if path.is_file()
    }


# --------------------------------------------------------------------------- #
# Stale checkpoint / partial evidence fail-closed
# --------------------------------------------------------------------------- #
def test_selection_resume_rejects_stale_ablation_checkpoint(
    triple, held_out_run, dataset, tmp_path, monkeypatch
) -> None:
    tampered = tmp_path / "tampered-selection"
    shutil.copytree(held_out_run["result"].selection_path, tampered)
    checkpoint = tampered / CHECKPOINT_FILENAME
    lines = [
        json.loads(line)
        for line in checkpoint.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in lines:
        record["ablation_match_digest"] = "0" * 64
    checkpoint.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in lines),
        encoding="utf-8",
    )

    calls = _spy(monkeypatch)
    with pytest.raises(EvaluationError, match="ablation_match_digest"):
        run_selection_phase(
            selection_projection(held_out_run["spec"]),
            tmp_path / "output",
            dataset=dataset,
            evaluation_dataset=dataset,
            resume_from=tampered,
        )
    assert calls == []


def test_selection_resume_rejects_stale_artifact_checkpoint(
    triple, held_out_run, dataset, tmp_path, monkeypatch
) -> None:
    tampered = tmp_path / "tampered-selection"
    shutil.copytree(held_out_run["result"].selection_path, tampered)
    checkpoint = tampered / CHECKPOINT_FILENAME
    lines = [
        json.loads(line)
        for line in checkpoint.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lines[0]["artifact_digest"] = "f" * 64
    lines[0]["reused"] = False
    checkpoint.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in lines),
        encoding="utf-8",
    )
    calls = _spy(monkeypatch)
    with pytest.raises(EvaluationError, match="artifact_digest"):
        run_selection_phase(
            selection_projection(held_out_run["spec"]),
            tmp_path / "output",
            dataset=dataset,
            evaluation_dataset=dataset,
            resume_from=tampered,
        )
    assert calls == []


def test_summary_requires_every_ablation_variant(held_out_run) -> None:
    final = held_out_run["result"].final_result
    manifest = json.loads(json.dumps(final.manifest))
    manifest["entries"] = [
        entry
        for entry in manifest["entries"]
        if entry["policy_id"] != "bayesian_no_belief"
    ]
    with pytest.raises(EvaluationError, match="missing policy"):
        build_robustness_summary(
            manifest, trace_root=held_out_run["output"] / "final"
        )


def test_summary_rejects_ablation_block_without_eight_panel(held_out_run) -> None:
    final = held_out_run["result"].final_result
    manifest = json.loads(json.dumps(final.manifest))
    manifest["panel"] = list(REQUIRED_POLICY_IDS)
    with pytest.raises(EvaluationError, match="ablation"):
        build_robustness_summary(
            manifest, trace_root=held_out_run["output"] / "final"
        )


# --------------------------------------------------------------------------- #
# Loader-valid but mismatched artifact is rejected by the runner gate
# --------------------------------------------------------------------------- #
def test_baseline_rejects_loader_valid_bins_mismatch(
    triple, dataset, tmp_path, monkeypatch
) -> None:
    def mutate(manifest, likelihood, payoff):
        manifest["bins"]["disagreement"]["edges"][-1] = 1.25

    mutated = _mutate_artifact(
        Path(triple["paths"][NO_BELIEF_VARIANT]),
        tmp_path / "bins_mutated",
        mutate,
    )
    pinned = _pinned(triple["artifacts"])
    pinned[NO_BELIEF_VARIANT] = load_artifact_strict(mutated).manifest.aggregate_digest
    study = _study(triple, no_belief=mutated, pinned=pinned)
    calls = _spy(monkeypatch)
    output = tmp_path / "baseline"
    with pytest.raises(AblationEvaluationError) as info:
        run_baseline_study(
            study, output, dataset=dataset, evaluation_dataset=dataset
        )
    assert info.value.validation_class == "bins_mismatch"
    assert calls == []
    assert not output.exists()


# --------------------------------------------------------------------------- #
# Legacy six-policy outputs are never reformatted
# --------------------------------------------------------------------------- #
def test_six_policy_output_has_no_ablation_metadata(
    triple, dataset, tmp_path
) -> None:
    mapping = _study_mapping(triple)
    mapping = json.loads(json.dumps(mapping))
    mapping["panel"] = mapping["panel"][:6]
    mapping.pop("ablations")
    study = parse_study_text(yaml.safe_dump(mapping))
    result = run_baseline_study(
        study, tmp_path / "legacy", dataset=dataset, evaluation_dataset=dataset
    )
    assert result.manifest["panel"] == list(REQUIRED_POLICY_IDS)
    assert ABLATION_METADATA_KEY not in result.manifest
    assert result.ablation is None
    assert all(
        ABLATION_METADATA_KEY not in entry for entry in result.manifest["entries"]
    )
