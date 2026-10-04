"""Opt-in eight-policy ablation study contract and matched-artifact gate (S03/T03).

Two independent surfaces are exercised here:

1. **Study schema.** A legacy six-policy :class:`BaselineStudyConfig` must still
   parse, serialise, and digest byte-for-byte as before (the optional
   ``ablations`` contract is omitted rather than serialised as ``null``). An
   opt-in study must declare exactly the eight canonical ids, one pinned
   aggregate digest per full/no-belief/no-strategic artifact, and one pinned
   matched calibration-input digest; malformed contracts, wrong kinds, duplicate
   variant pins, reused artifact paths, and mis-budgeted ablation entries are all
   rejected at parse time.

2. **Matched-artifact preflight.** The pure gate in
   :mod:`dfl_sim.ablation_evaluation` loads the three *real* artifacts produced
   by the production calibration path exactly once, accepts a genuine matched
   triple, and rejects an unproven legacy artifact, a wrong pinned digest, a
   relabelled/reused variant, and loader-valid but mismatched bins, action-space
   settings, sampler settings, and derivation budgets.

Everything is offline and in-process: the injected dataset is generated from a
seed, artifacts are written under ``tmp_path``/``tmp_path_factory``, and no GSD
or ignored path is read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from dfl_sim import (
    BALANCE_CITATION,
    NO_BELIEF_VARIANT,
    NO_STRATEGIC_VARIANT,
    FULL_VARIANT,
    BaselineStudyConfigError,
    HeldOutStudyConfigError,
    calibrate_from_config,
    held_out_study_digest,
    load_bayesian_artifact,
    parse_held_out_study_text,
    parse_study_text,
    selection_projection,
    selection_spec_digest,
    study_digest,
    study_identity_payload,
)
from dfl_sim.ablation import matched_calibration_input_digest
from dfl_sim.ablation_evaluation import (
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
from dfl_sim.evaluation_config import (
    ABLATION_POLICY_IDS,
    ARTIFACT_POLICY_IDS,
    EXTENDED_POLICY_IDS,
    REQUIRED_POLICY_IDS,
)
from helpers import MNISTShapedDataset, make_config
from test_calibration_config import base_dict, parse_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"

SEED = 20240101
EPISODES = 8
DESIGN_SPLIT = 8
_TUNED_PROVENANCE = "selection-seed-budget"


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _base(**overrides):
    defaults = {
        "rounds": 6,
        "data": {"samples_per_node": 8},
        "training": {"local_epochs": 1, "batch_size": 4},
        "evaluation": {"max_samples": 8, "batch_size": 4},
    }
    defaults.update(overrides)
    return make_config(**defaults)


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


def _project_citation() -> dict:
    return {"kind": "project", "title": "project comparator"}


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


def _bayesian_entry(*, artifact_dir: str, tuning: dict | None = None, id_: str = "bayesian", implementation: str = "Bayesian") -> dict:
    return {
        "id": id_,
        "implementation": implementation,
        "aggregation": {
            "kind": "bayesian",
            "artifact_dir": artifact_dir,
            "omega": 0.5,
            "degree_cap": 4,
            "tie_tolerance": 1e-9,
        },
        "citation": _project_citation(),
        "tuning": tuning if tuning is not None else _tuned(),
    }


def _six_panel(*, artifact_dir: str, tuned: dict | None = None) -> list[dict]:
    tuned = tuned if tuned is not None else _tuned()
    project = _project_citation()
    exempt = _exempt()
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
        _bayesian_entry(artifact_dir=artifact_dir, tuning=tuned),
    ]


def _eight_panel(*, full: str, no_belief: str, no_strategic: str) -> list[dict]:
    panel = _six_panel(artifact_dir=full)
    panel.append(
        _bayesian_entry(
            artifact_dir=no_belief,
            id_="bayesian_no_belief",
            implementation="uniform-posterior ablation",
        )
    )
    panel.append(
        _bayesian_entry(
            artifact_dir=no_strategic,
            id_="bayesian_no_strategic",
            implementation="fixed-attacker ablation",
        )
    )
    return panel


def _contract(
    pinned: dict[str, str],
    *,
    matched: str,
    variants: tuple[str, ...] = (
        FULL_VARIANT,
        NO_BELIEF_VARIANT,
        NO_STRATEGIC_VARIANT,
    ),
) -> dict:
    return {
        "matched_calibration_input_digest": matched,
        "expected_aggregate_digests": [
            {"variant": variant, "expected_aggregate_digest": pinned[variant]}
            for variant in variants
        ],
    }


def _legacy_mapping(*, artifact_dir, tuned=None, **overrides) -> dict:
    mapping = {
        "study_id": "legacy_fixture_study",
        "data_provenance": "injected",
        "base": _base().model_dump(mode="json"),
        "seeds": [11, 12, 13],
        "candidate_budget": 3,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
        "panel": _six_panel(artifact_dir=artifact_dir, tuned=tuned),
    }
    mapping.update(overrides)
    return mapping


def _ablation_mapping(
    *,
    full: str,
    no_belief: str,
    no_strategic: str,
    pinned: dict[str, str],
    matched: str,
    contract: dict | None = None,
    panel: list[dict] | None = None,
    data_provenance: str = "injected",
    **overrides,
) -> dict:
    mapping = {
        "study_id": "ablation_fixture_study",
        "data_provenance": data_provenance,
        "base": _base().model_dump(mode="json"),
        "seeds": [11, 12, 13],
        "candidate_budget": 3,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
        "panel": (
            panel
            if panel is not None
            else _eight_panel(full=full, no_belief=no_belief, no_strategic=no_strategic)
        ),
        "ablations": (
            contract
            if contract is not None
            else _contract(pinned, matched=matched)
        ),
    }
    mapping.update(overrides)
    return mapping


def _study(mapping) -> object:
    return parse_study_text(yaml.safe_dump(mapping, sort_keys=True))


def _invalid(mapping) -> BaselineStudyConfigError:
    with pytest.raises(BaselineStudyConfigError) as info:
        _study(mapping)
    return info.value


# --------------------------------------------------------------------------- #
# Real calibration triple fixture
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def triple(tmp_path_factory):
    """Calibrate and load the genuine full/no-belief/no-strategic triple once."""

    dataset = MNISTShapedDataset(400, separable=True, seed=11)
    root = tmp_path_factory.mktemp("ablation-triple")
    paths: dict[str, str] = {}
    artifacts: dict[str, object] = {}
    for variant in (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT):
        dest = root / variant
        calibrate_from_config(_calibration_config(variant), str(dest), dataset=dataset)
        paths[variant] = str(dest)
        artifacts[variant] = load_bayesian_artifact(dest)
    return {"paths": paths, "artifacts": artifacts}


def _pinned(artifacts) -> dict[str, str]:
    return {
        variant: artifacts[variant].manifest.aggregate_digest
        for variant in (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT)
    }


def _matched(artifacts) -> str:
    return matched_calibration_input_digest(
        artifacts[FULL_VARIANT].manifest.derivation
    )


def _triple_study(triple, **overrides):
    kwargs = {
        "full": triple["paths"][FULL_VARIANT],
        "no_belief": triple["paths"][NO_BELIEF_VARIANT],
        "no_strategic": triple["paths"][NO_STRATEGIC_VARIANT],
        "pinned": _pinned(triple["artifacts"]),
        "matched": _matched(triple["artifacts"]),
    }
    kwargs.update(overrides)
    return _study(_ablation_mapping(**kwargs))


# --------------------------------------------------------------------------- #
# Loader-valid artifact mutation helper
# --------------------------------------------------------------------------- #
def _mutate_artifact(source: Path, dest: Path, mutate) -> str:
    """Copy ``source`` to ``dest``, apply ``mutate``, and reseal all digests.

    The result is structurally valid and loader-valid: only the gate's semantic
    matching checks can reject it.
    """

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
# Study schema: legacy behavior preserved
# --------------------------------------------------------------------------- #
def test_legacy_six_policy_config_still_parses_and_omits_ablations(tmp_path) -> None:
    study = _study(
        _legacy_mapping(artifact_dir=str(tmp_path / "bayesian"))
    )
    assert tuple(entry.id for entry in study.ordered_panel()) == REQUIRED_POLICY_IDS
    assert study.policy_ids() == REQUIRED_POLICY_IDS
    assert study.ablations is None
    payload = study.model_dump(mode="json")
    assert "ablations" not in payload
    assert "ablations" not in study_identity_payload(study)
    snapshot = json.dumps(payload)
    assert study_digest(parse_study_text(snapshot)) == study_digest(study)


def test_ablation_config_requires_the_eight_policy_panel(tmp_path) -> None:
    study = _study(
        _ablation_mapping(
            full=str(tmp_path / "full"),
            no_belief=str(tmp_path / "no_belief"),
            no_strategic=str(tmp_path / "no_strategic"),
            pinned={
                FULL_VARIANT: "a" * 64,
                NO_BELIEF_VARIANT: "b" * 64,
                NO_STRATEGIC_VARIANT: "c" * 64,
            },
            matched="d" * 64,
        )
    )
    assert tuple(entry.id for entry in study.ordered_panel()) == EXTENDED_POLICY_IDS
    assert study.policy_ids() == EXTENDED_POLICY_IDS
    assert study.ablations is not None
    assert "ablations" in study.model_dump(mode="json")


def test_ablation_contract_pinned_by_exact_absence_of_six_panel(tmp_path) -> None:
    # Eight-policy panel but no contract must be rejected.
    mapping = _ablation_mapping(
        full=str(tmp_path / "full"),
        no_belief=str(tmp_path / "no_belief"),
        no_strategic=str(tmp_path / "no_strategic"),
        pinned={v: "a" * 64 for v in (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT)},
        matched="d" * 64,
    )
    mapping.pop("ablations")
    assert "panel" in str(_invalid(mapping))


def test_ablation_contract_rejects_six_panel(tmp_path) -> None:
    # Contract present but only the six-policy panel must be rejected.
    mapping = _legacy_mapping(artifact_dir=str(tmp_path / "full"))
    mapping["ablations"] = _contract(
        {v: "a" * 64 for v in (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT)},
        matched="d" * 64,
    )
    assert "panel" in str(_invalid(mapping))


def test_ablation_entries_must_be_tuned_not_exempt(tmp_path) -> None:
    panel = _eight_panel(
        full=str(tmp_path / "full"),
        no_belief=str(tmp_path / "no_belief"),
        no_strategic=str(tmp_path / "no_strategic"),
    )
    for entry in panel:
        if entry["id"] == "bayesian_no_belief":
            entry["tuning"] = _exempt()
    mapping = _ablation_mapping(
        full=str(tmp_path / "full"),
        no_belief=str(tmp_path / "no_belief"),
        no_strategic=str(tmp_path / "no_strategic"),
        pinned={v: "a" * 64 for v in (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT)},
        matched="d" * 64,
        panel=panel,
    )
    message = str(_invalid(mapping))
    assert "bayesian_no_belief" in message


def test_ablation_entries_must_match_matched_budget(tmp_path) -> None:
    panel = _eight_panel(
        full=str(tmp_path / "full"),
        no_belief=str(tmp_path / "no_belief"),
        no_strategic=str(tmp_path / "no_strategic"),
    )
    for entry in panel:
        if entry["id"] == "bayesian_no_strategic":
            entry["tuning"] = {
                "mode": "tuned",
                "candidate_budget": 9,
                "selection_seed": 7,
                "selection_provenance": _TUNED_PROVENANCE,
            }
    mapping = _ablation_mapping(
        full=str(tmp_path / "full"),
        no_belief=str(tmp_path / "no_belief"),
        no_strategic=str(tmp_path / "no_strategic"),
        pinned={v: "a" * 64 for v in (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT)},
        matched="d" * 64,
        panel=panel,
    )
    message = str(_invalid(mapping))
    assert "candidate_budget" in message or "bayesian_no_strategic" in message


def test_ablation_entries_must_use_distinct_artifact_paths(tmp_path) -> None:
    shared = str(tmp_path / "shared")
    mapping = _ablation_mapping(
        full=shared,
        no_belief=shared,
        no_strategic=str(tmp_path / "no_strategic"),
        pinned={v: "a" * 64 for v in (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT)},
        matched="d" * 64,
    )
    assert "distinct" in str(_invalid(mapping))


def test_ablation_entry_wrong_kind_rejected(tmp_path) -> None:
    panel = _eight_panel(
        full=str(tmp_path / "full"),
        no_belief=str(tmp_path / "no_belief"),
        no_strategic=str(tmp_path / "no_strategic"),
    )
    for entry in panel:
        if entry["id"] == "bayesian_no_belief":
            entry["aggregation"] = {"kind": "balance"}
    mapping = _ablation_mapping(
        full=str(tmp_path / "full"),
        no_belief=str(tmp_path / "no_belief"),
        no_strategic=str(tmp_path / "no_strategic"),
        pinned={v: "a" * 64 for v in (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT)},
        matched="d" * 64,
        panel=panel,
    )
    assert "bayesian" in str(_invalid(mapping))


def test_contract_rejects_duplicate_variant_pin(tmp_path) -> None:
    mapping = _ablation_mapping(
        full=str(tmp_path / "full"),
        no_belief=str(tmp_path / "no_belief"),
        no_strategic=str(tmp_path / "no_strategic"),
        pinned={v: "a" * 64 for v in (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT)},
        matched="d" * 64,
        contract={
            "matched_calibration_input_digest": "d" * 64,
            "expected_aggregate_digests": [
                {"variant": FULL_VARIANT, "expected_aggregate_digest": "a" * 64},
                {"variant": FULL_VARIANT, "expected_aggregate_digest": "b" * 64},
                {"variant": NO_STRATEGIC_VARIANT, "expected_aggregate_digest": "c" * 64},
            ],
        },
    )
    assert "at most once" in str(_invalid(mapping)) or "exactly one" in str(_invalid(mapping))


def test_contract_rejects_incomplete_variant_pin(tmp_path) -> None:
    mapping = _ablation_mapping(
        full=str(tmp_path / "full"),
        no_belief=str(tmp_path / "no_belief"),
        no_strategic=str(tmp_path / "no_strategic"),
        pinned={v: "a" * 64 for v in (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT)},
        matched="d" * 64,
        contract={
            "matched_calibration_input_digest": "d" * 64,
            "expected_aggregate_digests": [
                {"variant": FULL_VARIANT, "expected_aggregate_digest": "a" * 64},
                {"variant": NO_BELIEF_VARIANT, "expected_aggregate_digest": "b" * 64},
            ],
        },
    )
    assert "exactly one" in str(_invalid(mapping))


@pytest.mark.parametrize("bad", ["not-a-digest", "A" * 64, "a" * 63])
def test_contract_rejects_bad_digest_format(tmp_path, bad) -> None:
    mapping = _ablation_mapping(
        full=str(tmp_path / "full"),
        no_belief=str(tmp_path / "no_belief"),
        no_strategic=str(tmp_path / "no_strategic"),
        pinned={v: "a" * 64 for v in (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT)},
        matched=bad,
    )
    assert "SHA-256" in str(_invalid(mapping))


def test_ablation_digest_sensitive_to_contract(tmp_path) -> None:
    def build(matched: str):
        return _study(
            _ablation_mapping(
                full=str(tmp_path / "full"),
                no_belief=str(tmp_path / "no_belief"),
                no_strategic=str(tmp_path / "no_strategic"),
                pinned={v: "a" * 64 for v in (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT)},
                matched=matched,
            )
        )

    first = build("d" * 64)
    second = build("e" * 64)
    assert study_digest(first) != study_digest(second)


def test_selection_digest_ignores_final_seeds_but_tracks_contract(tmp_path) -> None:
    study = _ablation_mapping(
        full=str(tmp_path / "full"),
        no_belief=str(tmp_path / "no_belief"),
        no_strategic=str(tmp_path / "no_strategic"),
        pinned={v: "a" * 64 for v in (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT)},
        matched="d" * 64,
        base=_base(
            attack={"compromised_node": 0, "intensity": "strong"}
        ).model_dump(mode="json"),
    )

    def held_out(study_mapping, final_seeds):
        mapping = {
            "study": study_mapping,
            "final_seeds": list(final_seeds),
            "known_schedules": [
                {"id": "periodic_known", "schedule": {"kind": "periodic"}},
                {"id": "duty_known", "schedule": {"kind": "duty_cycle"}},
            ],
            "held_out_schedule": {
                "id": "held_out_schedule",
                "schedule": {"kind": "held_out"},
            },
        }
        return parse_held_out_study_text(
            yaml.safe_dump(mapping, sort_keys=True)
        )

    first = held_out(study, (21, 22))
    second = held_out(study, (31, 32))
    assert first.study.ablations is not None
    assert selection_projection(first).study.ablations is not None
    assert selection_spec_digest(selection_projection(first)) == selection_spec_digest(
        selection_projection(second)
    )
    assert held_out_study_digest(first) != held_out_study_digest(second)


# --------------------------------------------------------------------------- #
# Matched-artifact preflight: accept
# --------------------------------------------------------------------------- #
def test_preflight_accepts_genuine_matched_triple(triple) -> None:
    study = _triple_study(triple)
    summary = preflight_ablation_study(study)
    assert summary.policy_ids() == ARTIFACT_POLICY_IDS
    assert tuple(summary.variants) == ARTIFACT_POLICY_IDS
    assert summary.artifacts["bayesian"] is not None
    assert isinstance(summary.variants, type(summary.variants))
    assert {v.variant for v in summary.variants.values()} == {
        FULL_VARIANT,
        NO_BELIEF_VARIANT,
        NO_STRATEGIC_VARIANT,
    }
    assert summary.matched_calibration_input_digest == _matched(triple["artifacts"])
    assert summary.data_provenance == "injected"
    assert summary.protocol_version.startswith("calibration-derivation-")
    # one immutable identity per artifact, with the shared fingerprints recorded
    assert len({i.matched_calibration_input_digest for i in summary.variants.values()}) == 1
    assert len({i.dataset_inputs_digest for i in summary.variants.values()}) == 1
    assert len({i.aggregate_digest for i in summary.variants.values()}) == 3
    # only the declared belief/fixed-attack differences remain
    assert summary.identity("bayesian").effective_mu_a == (0.0, 2.0, 5.0)
    assert summary.identity("bayesian_no_belief").effective_mu_a == (0.0, 2.0, 5.0)
    assert summary.identity("bayesian_no_strategic").effective_mu_a == (5.0,)
    # the returned mappings are read-only
    with pytest.raises(TypeError):
        summary.artifacts["bayesian"] = None  # type: ignore[index]


# --------------------------------------------------------------------------- #
# Matched-artifact preflight: reject
# --------------------------------------------------------------------------- #
def test_preflight_rejects_unproven_legacy_full(triple, tmp_path) -> None:
    study = _triple_study(
        triple,
        full=str(LEGACY_FIXTURE),
    )
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study(study)
    assert info.value.validation_class == "unproven_artifact"


def test_preflight_maps_missing_artifact_to_bounded_error(triple, tmp_path) -> None:
    missing = str(tmp_path / "does-not-exist")
    study = _triple_study(triple, no_belief=missing)
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study(study)
    assert info.value.validation_class == "artifact_load"
    assert str(tmp_path) not in str(info.value)


def test_preflight_rejects_wrong_pinned_digest(triple) -> None:
    pinned = _pinned(triple["artifacts"])
    pinned[NO_BELIEF_VARIANT] = "0" * 64
    study = _triple_study(triple, pinned=pinned)
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study(study)
    assert info.value.validation_class == "aggregate_digest_mismatch"


def test_preflight_rejects_wrong_matched_digest(triple) -> None:
    study = _triple_study(triple, matched="0" * 64)
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study(study)
    assert info.value.validation_class == "matched_input_mismatch"


def test_preflight_rejects_relabelled_reused_variant(triple, tmp_path) -> None:
    # A distinct directory holding a byte-identical *full* artifact offered as
    # the no-belief entry is rejected on its declared variant.
    reused = _mutate_artifact(
        Path(triple["paths"][FULL_VARIANT]),
        tmp_path / "reused_full",
        lambda manifest, likelihood, payoff: None,
    )
    study = _triple_study(triple, no_belief=reused)
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study(study)
    assert info.value.validation_class == "variant_mismatch"


def test_preflight_rejects_mismatched_bins(triple, tmp_path) -> None:
    def mutate(manifest, likelihood, payoff):
        manifest["bins"]["disagreement"]["edges"][-1] = 1.25

    mutated = _mutate_artifact(
        Path(triple["paths"][NO_BELIEF_VARIANT]),
        tmp_path / "bins_mutated",
        mutate,
    )
    pinned = _pinned(triple["artifacts"])
    pinned[NO_BELIEF_VARIANT] = load_artifact_strict(mutated).manifest.aggregate_digest
    study = _triple_study(triple, no_belief=mutated, pinned=pinned)
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study(study)
    assert info.value.validation_class == "bins_mismatch"


def test_preflight_rejects_mismatched_action_space(triple, tmp_path) -> None:
    def mutate(manifest, likelihood, payoff):
        manifest["action_space"]["tie_tolerance"] = 1e-6

    mutated = _mutate_artifact(
        Path(triple["paths"][NO_STRATEGIC_VARIANT]),
        tmp_path / "tie_mutated",
        mutate,
    )
    pinned = _pinned(triple["artifacts"])
    pinned[NO_STRATEGIC_VARIANT] = load_artifact_strict(mutated).manifest.aggregate_digest
    study = _triple_study(triple, no_strategic=mutated, pinned=pinned)
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study(study)
    assert info.value.validation_class == "action_space_mismatch"


def test_preflight_rejects_mismatched_sampler(triple, tmp_path) -> None:
    def mutate(manifest, likelihood, payoff):
        manifest["provenance"]["smoothing_alpha"] = 2.0

    mutated = _mutate_artifact(
        Path(triple["paths"][NO_BELIEF_VARIANT]),
        tmp_path / "smoothing_mutated",
        mutate,
    )
    pinned = _pinned(triple["artifacts"])
    pinned[NO_BELIEF_VARIANT] = load_artifact_strict(mutated).manifest.aggregate_digest
    study = _triple_study(triple, no_belief=mutated, pinned=pinned)
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study(study)
    assert info.value.validation_class == "sampler_mismatch"


def test_preflight_rejects_mismatched_budget(triple, tmp_path) -> None:
    def mutate(manifest, likelihood, payoff):
        manifest["derivation"]["rounds"] = manifest["derivation"]["rounds"] + 1

    mutated = _mutate_artifact(
        Path(triple["paths"][NO_STRATEGIC_VARIANT]),
        tmp_path / "rounds_mutated",
        mutate,
    )
    pinned = _pinned(triple["artifacts"])
    pinned[NO_STRATEGIC_VARIANT] = load_artifact_strict(mutated).manifest.aggregate_digest
    study = _triple_study(triple, no_strategic=mutated, pinned=pinned)
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study(study)
    assert info.value.validation_class == "budget_mismatch"


def test_preflight_rejects_data_provenance_mismatch(triple) -> None:
    study = _triple_study(triple, data_provenance="real_mnist")
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study(study)
    assert info.value.validation_class == "data_provenance_mismatch"


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("dataset_inputs_digest", "1" * 64),
        ("partition_membership_digest", "2" * 64),
        ("schedule_pairing_digest", "3" * 64),
        ("calibration_config_digest", "4" * 64),
        ("seed", 424242),
    ],
)
def test_preflight_rejects_mismatched_calibration_input(
    triple, tmp_path, field, replacement
) -> None:
    def mutate(manifest, likelihood, payoff):
        manifest["derivation"][field] = replacement

    mutated = _mutate_artifact(
        Path(triple["paths"][NO_BELIEF_VARIANT]),
        tmp_path / f"input_{field}",
        mutate,
    )
    pinned = _pinned(triple["artifacts"])
    pinned[NO_BELIEF_VARIANT] = load_artifact_strict(mutated).manifest.aggregate_digest
    study = _triple_study(triple, no_belief=mutated, pinned=pinned)
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study(study)
    assert info.value.validation_class == "matched_input_mismatch"


def test_preflight_rejects_swapped_ablation_variants(triple) -> None:
    study = _triple_study(
        triple,
        no_belief=triple["paths"][NO_STRATEGIC_VARIANT],
        no_strategic=triple["paths"][NO_BELIEF_VARIANT],
    )
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study(study)
    assert info.value.validation_class == "variant_mismatch"


def test_preflight_rejects_legacy_study_without_contract(triple) -> None:
    study = _study(
        _legacy_mapping(artifact_dir=triple["paths"][FULL_VARIANT])
    )
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study(study)
    assert info.value.validation_class == "not_ablation_study"


def test_preflight_rejects_wrong_study_type(triple) -> None:
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study({"not": "a study"})  # type: ignore[arg-type]
    assert info.value.validation_class == "study_type"


def test_preflight_errors_are_bounded_and_path_free(triple, tmp_path) -> None:
    def mutate(manifest, likelihood, payoff):
        manifest["bins"]["disagreement"]["edges"][-1] = 1.25

    mutated = _mutate_artifact(
        Path(triple["paths"][NO_BELIEF_VARIANT]),
        tmp_path / "bins_mutated_bounded",
        mutate,
    )
    pinned = _pinned(triple["artifacts"])
    pinned[NO_BELIEF_VARIANT] = load_artifact_strict(mutated).manifest.aggregate_digest
    study = _triple_study(triple, no_belief=mutated, pinned=pinned)
    with pytest.raises(AblationEvaluationError) as info:
        preflight_ablation_study(study)
    message = str(info.value)
    assert str(tmp_path) not in message
    assert len(message) < 500


# --------------------------------------------------------------------------- #
# Adjacent held-out contract wiring
# --------------------------------------------------------------------------- #
def test_unknown_policy_id_rejected(tmp_path) -> None:
    mapping = _legacy_mapping(artifact_dir=str(tmp_path / "bayesian"))
    mapping["panel"][0]["id"] = "mystery_policy"
    assert "unknown policy id" in str(_invalid(mapping))


def test_ablation_policy_ids_are_exported_consistently() -> None:
    assert ABLATION_POLICY_IDS == ("bayesian_no_belief", "bayesian_no_strategic")
    assert EXTENDED_POLICY_IDS == REQUIRED_POLICY_IDS + ABLATION_POLICY_IDS
    assert ARTIFACT_POLICY_IDS == (
        "bayesian",
        "bayesian_no_belief",
        "bayesian_no_strategic",
    )
