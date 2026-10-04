"""``dfl calibrate --variant`` + opt-in ablation ``dfl evaluate`` CLI tests (S03/T05).

These tests close the **public command workflow** the slice demo promises:

1. three real ``dfl calibrate`` dispatches (``--variant full``/``no_belief``/
   ``no_strategic``) over an explicitly injected in-process dataset produce the
   matched artifact triple;
2. a temporary eight-policy held-out study spec built from those artifacts'
   *actual* aggregate and matched-input digests runs the real ``dfl evaluate``
   path (selection -> freeze -> final) and publishes TraceReader-valid traces,
   complete matched identity metadata, and the existing single success marker;
3. deliberate mismatch is rejected before a single cell runs.

Everything is offline and deterministic: the production loaders are replaced
through in-process test seams that inject ``MNISTShapedDataset`` fixtures, so
nothing touches the network. The study declares ``data_provenance: injected``
honestly -- these are not downloaded-MNIST results (that is S04 evidence). All
generated artifacts/specs/evidence live under ``tmp_path``; no GSD or ignored
path is ever read.
"""

from __future__ import annotations

import hashlib
import io
import json
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import dfl_sim.__main__ as cli_module
import dfl_sim.evaluation as evaluation_module
from dfl_sim import (
    ABLATION_METADATA_KEY,
    ABLATION_VARIANT_BY_POLICY_ID,
    ARTIFACT_POLICY_IDS,
    BALANCE_CITATION,
    EXTENDED_POLICY_IDS,
    FULL_VARIANT,
    NO_BELIEF_VARIANT,
    NO_STRATEGIC_VARIANT,
    SOURCE_MU_A,
    TraceReader,
    build_bayesian_policy,
    collect_calibration,
    load_bayesian_artifact,
    load_calibration_config,
    matched_calibration_input_digest,
)
from dfl_sim.__main__ import (
    HELD_OUT_FINAL_PROGRESS_MARKER,
    HELD_OUT_SELECTION_PROGRESS_MARKER,
    HELD_OUT_SUCCESS_MARKER,
    build_calibrate_parser,
    main,
)
from dfl_sim.bayesian_artifact import (
    ARTIFACT_FILENAMES,
    LIKELIHOOD_FILENAME,
    MANIFEST_FILENAME,
    PAYOFF_FILENAME,
    canonical_json_bytes,
    compute_aggregate_digest,
)
from dfl_sim.evaluation import (
    FINAL_MANIFEST_FILENAME,
    SELECTION_MANIFEST_FILENAME,
)
from helpers import MNISTShapedDataset, make_config
from test_ablation_calibration import _observation, _supported_evidence

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION_CONFIG = REPO_ROOT / "configs" / "calibrate_ablation_chain3.yaml"
CHAIN3_CONFIG = REPO_ROOT / "configs" / "calibrate_chain3.yaml"

VARIANTS = (FULL_VARIANT, NO_BELIEF_VARIANT, NO_STRATEGIC_VARIANT)
SELECTION_SEEDS = (11, 12, 13)
FINAL_SEEDS = (21, 22)
KNOWN_SCHEDULE_ID = "periodic_known"
HELD_OUT_ID = "held_out_schedule"
_TUNED_PROVENANCE = "selection-seed-budget"
_STAGING_PREFIX = ".held_out_study.staging-"


# --------------------------------------------------------------------------- #
# In-process test seams (inject deterministic datasets, never the network)
# --------------------------------------------------------------------------- #
def _calibration_dataset() -> MNISTShapedDataset:
    # Same informative fixture the T02 calibration tests use: a benign degree-2
    # chain3 receiver sees a malicious neighbor and type-1 payoff support exists.
    return MNISTShapedDataset(400, separable=True, seed=11)


def _evaluation_dataset() -> MNISTShapedDataset:
    return MNISTShapedDataset(64, seed=1)


def _evaluation_test_dataset() -> MNISTShapedDataset:
    return MNISTShapedDataset(32, seed=2)


@contextmanager
def _calibration_seam(dataset: MNISTShapedDataset):
    """Inject ``dataset`` through the real calibration orchestrator seam."""

    real = cli_module.calibrate_from_config

    def injected(config, output, **kwargs):
        return real(config, output, dataset=dataset, **kwargs)

    cli_module.calibrate_from_config = injected
    try:
        yield
    finally:
        cli_module.calibrate_from_config = real


@contextmanager
def _evaluate_seam(dataset: MNISTShapedDataset, test_dataset: MNISTShapedDataset):
    """Inject datasets through the real held-out/final runner seams."""

    real_held = cli_module.run_held_out_study
    real_final = cli_module.run_final_phase

    def held(*args, **kwargs):
        kwargs.setdefault("dataset", dataset)
        kwargs.setdefault("evaluation_dataset", test_dataset)
        return real_held(*args, **kwargs)

    def final(*args, **kwargs):
        kwargs.setdefault("dataset", dataset)
        kwargs.setdefault("evaluation_dataset", test_dataset)
        return real_final(*args, **kwargs)

    cli_module.run_held_out_study = held
    cli_module.run_final_phase = final
    try:
        yield
    finally:
        cli_module.run_held_out_study = real_held
        cli_module.run_final_phase = real_final


def _run(argv, *, stdout: io.StringIO | None = None, stderr: io.StringIO | None = None) -> int:
    out_buf = stdout if stdout is not None else io.StringIO()
    err_buf = stderr if stderr is not None else io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        return main(argv)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_dirs(root: Path) -> list[Path]:
    traces = root / "traces"
    if not traces.is_dir():
        return []
    return sorted(path for path in traces.rglob("seed-*") if path.is_dir())


def _staging_leftovers(parent: Path) -> list[Path]:
    return sorted(parent.glob(f"{_STAGING_PREFIX}*"))


# --------------------------------------------------------------------------- #
# Shared calibration triple: three real CLI dispatches, once
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def triple(tmp_path_factory):
    dataset = _calibration_dataset()
    root = tmp_path_factory.mktemp("ablation-cli-calibration")
    paths: dict[str, str] = {}
    artifacts: dict[str, object] = {}
    stdout: dict[str, str] = {}
    with _calibration_seam(dataset):
        for variant in VARIANTS:
            out = root / variant
            buf, err = io.StringIO(), io.StringIO()
            status = _run(
                [
                    "calibrate",
                    str(ABLATION_CONFIG),
                    "--output",
                    str(out),
                    "--variant",
                    variant,
                ],
                stdout=buf,
                stderr=err,
            )
            assert status == 0, (variant, status, err.getvalue())
            paths[variant] = str(out)
            artifacts[variant] = load_bayesian_artifact(out)
            stdout[variant] = buf.getvalue()
    return {"paths": paths, "artifacts": artifacts, "stdout": stdout}


# --------------------------------------------------------------------------- #
# Study mapping builders (eight-policy, injected provenance)
# --------------------------------------------------------------------------- #
def _base(**overrides):
    defaults = {
        "rounds": 6,
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


def _pinned(triple) -> dict[str, str]:
    return {
        variant: triple["artifacts"][variant].manifest.aggregate_digest
        for variant in VARIANTS
    }


def _matched(triple) -> str:
    return matched_calibration_input_digest(
        triple["artifacts"][FULL_VARIANT].manifest.derivation
    )


def _study_mapping(triple, *, panel=None, pinned=None, matched=None, **overrides) -> dict:
    if pinned is None:
        pinned = _pinned(triple)
    if matched is None:
        matched = _matched(triple)
    mapping = {
        "study_id": "t05_ablation_cli",
        "data_provenance": "injected",
        "base": _base().model_dump(mode="json"),
        "seeds": list(SELECTION_SEEDS),
        "candidate_budget": 3,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
        "panel": (
            panel
            if panel is not None
            else _eight_panel(
                triple["paths"][FULL_VARIANT],
                triple["paths"][NO_BELIEF_VARIANT],
                triple["paths"][NO_STRATEGIC_VARIANT],
            )
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


def _held_out_mapping(triple, **kwargs) -> dict:
    return {
        "study": _study_mapping(triple, **kwargs),
        "final_seeds": list(FINAL_SEEDS),
        "known_schedules": [
            {"id": KNOWN_SCHEDULE_ID, "schedule": {"kind": "periodic"}}
        ],
        "held_out_schedule": {
            "id": HELD_OUT_ID,
            "schedule": {"kind": "held_out"},
        },
    }


def _write_spec(tmp_path: Path, mapping, *, name: str = "study.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(mapping, sort_keys=True), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Artifact mutation helpers (loader-valid, digest-resealed)
# --------------------------------------------------------------------------- #
def _reseed(manifest: dict, likelihood: list, payoff: list) -> None:
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


def _mutate_artifact(source: Path, dest: Path, mutate) -> str:
    """Copy ``source`` to ``dest``, apply ``mutate``, and reseal all digests."""

    dest.mkdir(parents=True)
    manifest = json.loads((source / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    likelihood = json.loads((source / LIKELIHOOD_FILENAME).read_text(encoding="utf-8"))
    payoff = json.loads((source / PAYOFF_FILENAME).read_text(encoding="utf-8"))
    mutate(manifest, likelihood, payoff)
    _reseed(manifest, likelihood, payoff)
    (dest / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
    (dest / LIKELIHOOD_FILENAME).write_bytes(canonical_json_bytes(likelihood))
    (dest / PAYOFF_FILENAME).write_bytes(canonical_json_bytes(payoff))
    return str(dest)


def _mutate_in_place(source: Path, mutate) -> dict[str, bytes]:
    """Mutate ``source`` in place; return the original bytes for restoration."""

    original = {name: (source / name).read_bytes() for name in ARTIFACT_FILENAMES}
    manifest = json.loads((source / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    likelihood = json.loads((source / LIKELIHOOD_FILENAME).read_text(encoding="utf-8"))
    payoff = json.loads((source / PAYOFF_FILENAME).read_text(encoding="utf-8"))
    mutate(manifest, likelihood, payoff)
    _reseed(manifest, likelihood, payoff)
    (source / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
    (source / LIKELIHOOD_FILENAME).write_bytes(canonical_json_bytes(likelihood))
    (source / PAYOFF_FILENAME).write_bytes(canonical_json_bytes(payoff))
    return original


def _restore_in_place(source: Path, original: dict[str, bytes]) -> None:
    for name, data in original.items():
        (source / name).write_bytes(data)


def _counting_spy(monkeypatch) -> list[str]:
    calls: list[str] = []
    real = evaluation_module.run_from_config

    def counting(config, **kwargs):
        calls.append(config.aggregation.kind)
        return real(config, **kwargs)

    monkeypatch.setattr(evaluation_module, "run_from_config", counting)
    return calls


def _assert_cli_failure(status: int, err: str, output: Path) -> None:
    assert status != 0
    assert "error" in err
    assert "Traceback" not in err
    assert not output.exists()


# --------------------------------------------------------------------------- #
# Calibration: three variants, matched identities, honest metadata
# --------------------------------------------------------------------------- #
def test_cli_calibrate_three_variants_are_matched_and_self_describing(triple) -> None:
    artifacts = triple["artifacts"]
    assert set(artifacts) == set(VARIANTS)

    digests = set()
    for variant, artifact in artifacts.items():
        derivation = artifact.manifest.derivation
        assert derivation is not None
        assert derivation.variant == variant
        assert tuple(derivation.source_mu_a) == SOURCE_MU_A
        assert derivation.data_provenance == "injected"
        digests.add(artifact.manifest.aggregate_digest)
        # One bounded success marker per dispatch, naming the aggregate digest.
        assert "wrote calibration artifact" in triple["stdout"][variant]
        assert f"aggregate_digest={artifact.manifest.aggregate_digest}" in (
            triple["stdout"][variant]
        )
    # Distinct variant identities, one shared matched calibration-input digest.
    assert len(digests) == 3
    assert len({_matched(triple) for _ in VARIANTS}) == 1


def test_cli_calibrate_variant_override_controls_effective_mixture(triple) -> None:
    expected = {
        FULL_VARIANT: SOURCE_MU_A,
        NO_BELIEF_VARIANT: SOURCE_MU_A,
        NO_STRATEGIC_VARIANT: (5.0,),
    }
    for variant, mixture in expected.items():
        derivation = triple["artifacts"][variant].manifest.derivation
        assert tuple(derivation.effective_mu_a) == mixture
        assert tuple(
            triple["artifacts"][variant].manifest.provenance.mu_a
        ) == mixture
    assert (
        triple["artifacts"][NO_BELIEF_VARIANT]
        .manifest.coverage.posterior_calibration_applicable
        is False
    )


def test_cli_calibrate_without_variant_uses_config_default(tmp_path, monkeypatch) -> None:
    # A separate one-off calibration proves the flag is optional and that the
    # config's own variant (default full) is used when it is absent.
    dataset = _calibration_dataset()
    out = tmp_path / "default"
    buf, err = io.StringIO(), io.StringIO()
    with _calibration_seam(dataset):
        status = _run(
            ["calibrate", str(ABLATION_CONFIG), "--output", str(out)],
            stdout=buf,
            stderr=err,
        )
    assert status == 0, err.getvalue()
    artifact = load_bayesian_artifact(out)
    assert artifact.manifest.derivation.variant == FULL_VARIANT
    assert "wrote calibration artifact" in buf.getvalue()


def test_ablation_config_is_valid_and_informative(triple) -> None:
    config = load_calibration_config(ABLATION_CONFIG)
    assert config.variant == FULL_VARIANT
    assert config.episodes == 8
    assert config.p0 == 0.5
    assert config.design_split_size == 8

    dataset = _calibration_dataset()
    collection = collect_calibration(config, dataset=dataset)
    # A benign degree-2 chain3 receiver observes a directly adjacent malicious
    # neighbor, and a degree-2 benign record is actually collected.
    assert any(
        spec.is_attacked and spec.node != 1 for spec in collection.schedule.specs
    )
    assert any(record.degree == 2 for record in collection.records)

    # The CLI-calibrated full artifact (same config/dataset) carries type-1
    # payoff support and yields a non-fallback action; the defense/ablations are
    # not required to improve accuracy.
    artifact = triple["artifacts"][FULL_VARIANT]
    assert any(key[2] == 1 for key in artifact.payoff_index)
    supported = _supported_evidence(artifact)
    assert supported is not None
    degree, evidence = supported
    sink: list = []
    policy = build_bayesian_policy(artifact, 1, diagnostic_sink=sink.append)
    policy.select_weights(_observation(artifact, degree, evidence))
    assert sink[0].fallback_reason is None
    assert sink[0].posterior


# --------------------------------------------------------------------------- #
# Happy path: the documented workflow publishes a complete matched study
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def published(triple, tmp_path_factory):
    spec_dir = tmp_path_factory.mktemp("ablation-cli-spec")
    spec = _write_spec(spec_dir, _held_out_mapping(triple))
    output = tmp_path_factory.mktemp("ablation-cli-out") / "study"

    out_buf, err_buf = io.StringIO(), io.StringIO()
    with _evaluate_seam(_evaluation_dataset(), _evaluation_test_dataset()):
        status = _run(
            ["evaluate", str(spec), "--output", str(output)],
            stdout=out_buf,
            stderr=err_buf,
        )
    assert status == 0, (
        f"dfl evaluate ablation study failed: status={status}\n"
        f"stdout:\n{out_buf.getvalue()}\nstderr:\n{err_buf.getvalue()}"
    )
    selection_output = output / "selection"
    final_output = output / "final"
    return SimpleNamespace(
        status=status,
        spec=spec,
        output=output,
        selection_output=selection_output,
        final_output=final_output,
        stdout=out_buf.getvalue(),
        stderr=err_buf.getvalue(),
        selection_manifest=_read_json(
            selection_output / SELECTION_MANIFEST_FILENAME
        ),
        final_manifest=_read_json(final_output / FINAL_MANIFEST_FILENAME),
    )


def test_cli_evaluate_emits_one_bounded_success_marker(published) -> None:
    assert published.status == 0
    assert published.stdout.count(HELD_OUT_SUCCESS_MARKER) == 1
    marker = published.stdout.strip()
    assert "policies=8" in marker
    assert f"selection_seeds={len(SELECTION_SEEDS)}" in marker
    assert f"final_seeds={len(FINAL_SEEDS)}" in marker
    assert "selection_cells=24" in marker
    assert "final_cells=32" in marker
    assert not _staging_leftovers(published.output.parent)


def test_cli_evaluate_progress_markers_are_bounded_per_cell(published) -> None:
    assert published.stderr.count(HELD_OUT_SELECTION_PROGRESS_MARKER) == 24
    assert published.stderr.count(HELD_OUT_FINAL_PROGRESS_MARKER) == 32
    # Progress lines never echo YAML/artifact content.
    for line in published.stderr.splitlines():
        assert "artifact_dir" not in line
        assert "study_id" not in line


def test_cli_evaluate_publishes_every_policy_and_matched_identity(published) -> None:
    selection = published.selection_manifest
    final = published.final_manifest
    assert selection["panel"] == list(EXTENDED_POLICY_IDS)
    assert final["panel"] == list(EXTENDED_POLICY_IDS)
    assert len(selection["entries"]) == 24
    assert len(final["entries"]) == 32

    for manifest in (selection, final):
        ablation = manifest[ABLATION_METADATA_KEY]
        assert ablation["assumption_version"] == "calibration-derivation-v1"
        assert set(ablation["variants"]) == set(ARTIFACT_POLICY_IDS)
        assert "not fully isolated" in ablation["fixed_behavior_limitation"]
        assert set(ablation["ablated_aggregate_digests"]) == {
            "bayesian_no_belief",
            "bayesian_no_strategic",
        }
        for entry in manifest["entries"]:
            if entry["policy_id"] in ARTIFACT_POLICY_IDS:
                meta = entry[ABLATION_METADATA_KEY]
                assert meta["variant"] == ABLATION_VARIANT_BY_POLICY_ID[
                    entry["policy_id"]
                ]
                assert meta["matched_calibration_input_digest"] == (
                    ablation["matched_calibration_input_digest"]
                )
                assert entry["artifact_digest"] in (
                    set(ablation["ablated_aggregate_digests"].values())
                    | {ablation["full_aggregate_digest"]}
                )
            else:
                assert ABLATION_METADATA_KEY not in entry

    # Frozen split: disjoint selection/final seeds and a reserved schedule that
    # only the final phase builds.
    assert set(selection["selection_seeds"]).isdisjoint(final["final_seeds"])
    assert HELD_OUT_ID not in selection["schedule_ids"]
    assert HELD_OUT_ID in final["schedule_ids"]
    assert all(entry["schedule_id"] != HELD_OUT_ID for entry in selection["entries"])
    assert any(entry["held_out"] for entry in final["entries"])


def test_cli_evaluate_paired_cells_share_inputs_and_attack(published) -> None:
    for manifest in (published.selection_manifest, published.final_manifest):
        # The phase-level shared-input digest proves every cell ran from the
        # same realized input (partitions/initialization) for the whole phase.
        assert manifest["shared_input_digest"]
        cells: dict[tuple, list[dict]] = {}
        for entry in manifest["entries"]:
            cells.setdefault((entry["seed"], entry["schedule_id"]), []).append(entry)
        assert cells
        for key, entries in cells.items():
            assert len(entries) == 8, key
            kappas = {tuple(entry["realized_kappa_sequence"]) for entry in entries}
            digests = {entry["schedule_digest"] for entry in entries}
            assert len(kappas) == 1, key
            assert len(digests) == 1, key


def test_cli_evaluate_all_traces_are_readable_schema_v1(published) -> None:
    selection_traces = _trace_dirs(published.selection_output)
    final_traces = _trace_dirs(published.final_output)
    assert len(selection_traces) == 24
    assert len(final_traces) == 32
    assert all(HELD_OUT_ID not in trace.parts for trace in selection_traces)
    assert any(HELD_OUT_ID in trace.parts for trace in final_traces)
    for trace in (*selection_traces, *final_traces):
        with TraceReader(trace) as reader:
            assert reader.schema_version == 1


# --------------------------------------------------------------------------- #
# Negatives: fail closed before any cell, with bounded diagnostics
# --------------------------------------------------------------------------- #
def test_cli_evaluate_swapped_variant_artifact_is_rejected_before_any_cell(
    triple, tmp_path, monkeypatch
) -> None:
    mapping = _held_out_mapping(
        triple,
        panel=_eight_panel(
            triple["paths"][FULL_VARIANT],
            triple["paths"][NO_STRATEGIC_VARIANT],  # swapped
            triple["paths"][NO_BELIEF_VARIANT],  # swapped
        ),
    )
    spec = _write_spec(tmp_path, mapping)
    calls = _counting_spy(monkeypatch)
    output = tmp_path / "study"
    buf, err = io.StringIO(), io.StringIO()
    with _evaluate_seam(_evaluation_dataset(), _evaluation_test_dataset()):
        status = _run(
            ["evaluate", str(spec), "--output", str(output)], stdout=buf, stderr=err
        )
    _assert_cli_failure(status, err.getvalue(), output)
    assert "variant_mismatch" in err.getvalue()
    assert buf.getvalue() == ""
    assert calls == []
    assert not (output / SELECTION_MANIFEST_FILENAME).exists()


def test_cli_evaluate_loader_valid_bins_mismatch_is_rejected(
    triple, tmp_path, monkeypatch
) -> None:
    def mutate(manifest, likelihood, payoff):
        manifest["bins"]["disagreement"]["edges"][-1] = 1.25

    mutated = _mutate_artifact(
        Path(triple["paths"][NO_BELIEF_VARIANT]),
        tmp_path / "bins_mutated",
        mutate,
    )
    pinned = _pinned(triple)
    pinned[NO_BELIEF_VARIANT] = load_bayesian_artifact(mutated).manifest.aggregate_digest
    mapping = _held_out_mapping(
        triple,
        panel=_eight_panel(
            triple["paths"][FULL_VARIANT],
            mutated,
            triple["paths"][NO_STRATEGIC_VARIANT],
        ),
        pinned=pinned,
    )
    spec = _write_spec(tmp_path, mapping)
    calls = _counting_spy(monkeypatch)
    output = tmp_path / "study"
    buf, err = io.StringIO(), io.StringIO()
    with _evaluate_seam(_evaluation_dataset(), _evaluation_test_dataset()):
        status = _run(
            ["evaluate", str(spec), "--output", str(output)], stdout=buf, stderr=err
        )
    _assert_cli_failure(status, err.getvalue(), output)
    assert "bins_mismatch" in err.getvalue()
    assert buf.getvalue() == ""
    assert calls == []


def test_cli_evaluate_missing_provenance_legacy_full_is_rejected(
    triple, tmp_path, monkeypatch
) -> None:
    def strip(manifest, likelihood, payoff):
        manifest.pop("derivation", None)

    legacy = _mutate_artifact(
        Path(triple["paths"][FULL_VARIANT]), tmp_path / "legacy_full", strip
    )
    pinned = _pinned(triple)
    pinned[FULL_VARIANT] = load_bayesian_artifact(legacy).manifest.aggregate_digest
    mapping = _held_out_mapping(
        triple,
        panel=_eight_panel(
            legacy,
            triple["paths"][NO_BELIEF_VARIANT],
            triple["paths"][NO_STRATEGIC_VARIANT],
        ),
        pinned=pinned,
    )
    spec = _write_spec(tmp_path, mapping)
    calls = _counting_spy(monkeypatch)
    output = tmp_path / "study"
    buf, err = io.StringIO(), io.StringIO()
    with _evaluate_seam(_evaluation_dataset(), _evaluation_test_dataset()):
        status = _run(
            ["evaluate", str(spec), "--output", str(output)], stdout=buf, stderr=err
        )
    _assert_cli_failure(status, err.getvalue(), output)
    assert "unproven_artifact" in err.getvalue()
    assert buf.getvalue() == ""
    assert calls == []


def test_cli_evaluate_replaced_artifact_at_final_and_resume_is_rejected(
    triple, published, tmp_path, monkeypatch
) -> None:
    no_belief = Path(triple["paths"][NO_BELIEF_VARIANT])

    def mutate(manifest, likelihood, payoff):
        manifest["bins"]["disagreement"]["edges"][-1] = 1.25

    original = _mutate_in_place(no_belief, mutate)
    try:
        # (a) final-only against the frozen selection output.
        calls = _counting_spy(monkeypatch)
        final_out = tmp_path / "final"
        buf, err = io.StringIO(), io.StringIO()
        with _evaluate_seam(_evaluation_dataset(), _evaluation_test_dataset()):
            status = _run(
                [
                    "evaluate",
                    str(published.spec),
                    "--output",
                    str(final_out),
                    "--selection-output",
                    str(published.selection_output),
                ],
                stdout=buf,
                stderr=err,
            )
        _assert_cli_failure(status, err.getvalue(), final_out)
        assert buf.getvalue() == ""
        assert calls == []

        # (b) resume of the full two-phase study.
        monkeypatch.undo()
        calls = _counting_spy(monkeypatch)
        resume_out = tmp_path / "resume"
        buf, err = io.StringIO(), io.StringIO()
        with _evaluate_seam(_evaluation_dataset(), _evaluation_test_dataset()):
            status = _run(
                [
                    "evaluate",
                    str(published.spec),
                    "--output",
                    str(resume_out),
                    "--resume-from",
                    str(published.output),
                ],
                stdout=buf,
                stderr=err,
            )
        _assert_cli_failure(status, err.getvalue(), resume_out)
        assert buf.getvalue() == ""
        assert calls == []
    finally:
        _restore_in_place(no_belief, original)


def test_cli_evaluate_interrupted_run_publishes_nothing(
    triple, tmp_path, monkeypatch
) -> None:
    spec = _write_spec(tmp_path, _held_out_mapping(triple))
    real = evaluation_module.run_from_config
    state = {"count": 0}

    def flaky(config, **kwargs):
        state["count"] += 1
        if state["count"] >= 3:
            raise RuntimeError("injected mid-panel failure")
        return real(config, **kwargs)

    monkeypatch.setattr(evaluation_module, "run_from_config", flaky)
    output = tmp_path / "study"
    buf, err = io.StringIO(), io.StringIO()
    with _evaluate_seam(_evaluation_dataset(), _evaluation_test_dataset()):
        status = _run(
            ["evaluate", str(spec), "--output", str(output)], stdout=buf, stderr=err
        )
    _assert_cli_failure(status, err.getvalue(), output)
    assert HELD_OUT_SUCCESS_MARKER not in buf.getvalue()
    assert _staging_leftovers(tmp_path) == []


# --------------------------------------------------------------------------- #
# Calibration CLI negatives (Q5/Q7)
# --------------------------------------------------------------------------- #
def test_cli_calibrate_unknown_variant_flag_is_rejected(tmp_path, capsys) -> None:
    output = tmp_path / "artifact"
    with pytest.raises(SystemExit) as info:
        main(
            [
                "calibrate",
                str(ABLATION_CONFIG),
                "--output",
                str(output),
                "--variant",
                "bogus",
            ]
        )
    assert info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "bogus" in captured.err
    assert not output.exists()


def test_cli_calibrate_config_declared_unknown_variant_is_bounded(
    tmp_path, capsys
) -> None:
    mapping = yaml.safe_load(ABLATION_CONFIG.read_text(encoding="utf-8"))
    mapping["variant"] = "bogus"
    bad = tmp_path / "bad_variant.yaml"
    bad.write_text(yaml.safe_dump(mapping, sort_keys=True), encoding="utf-8")
    output = tmp_path / "artifact"
    status = main(["calibrate", str(bad), "--output", str(output)])
    err = capsys.readouterr().err
    _assert_cli_failure(status, err, output)
    assert "invalid calibration config" in err
    assert "bogus" in err


def test_build_calibrate_parser_accepts_variant_choices() -> None:
    parser = build_calibrate_parser()
    for variant in VARIANTS:
        args = parser.parse_args(
            ["config.yaml", "--output", "out", "--variant", variant]
        )
        assert args.variant == variant
    args = parser.parse_args(["config.yaml", "--output", "out"])
    assert args.variant is None


# --------------------------------------------------------------------------- #
# Legacy behavior is untouched
# --------------------------------------------------------------------------- #
def test_legacy_calibrate_without_variant_still_works(tmp_path) -> None:
    # The original chain3 config (no --variant) still calibrates and publishes.
    dataset = MNISTShapedDataset(64, separable=True, seed=11)
    out = tmp_path / "legacy"
    buf, err = io.StringIO(), io.StringIO()
    with _calibration_seam(dataset):
        status = _run(
            ["calibrate", str(CHAIN3_CONFIG), "--output", str(out)],
            stdout=buf,
            stderr=err,
        )
    assert status == 0, err.getvalue()
    assert sorted(p.name for p in out.iterdir()) == sorted(ARTIFACT_FILENAMES)
    assert load_bayesian_artifact(out).manifest.derivation.variant == FULL_VARIANT
