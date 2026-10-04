"""Seed-derived degree placement resolution and runner integration (S05/T02, R020/D040).

Everything here is offline and deterministic: the topology is the production
seeded Watts-Strogatz family, the datasets are in-memory MNIST-shaped tensors,
and the Bayesian cell (when present) loads the git-tracked M002 fixture. The
tests cover the pure resolver (ranks, middle convention, regular graphs, degree
ties, digest stability), the common per-cell config integration, the
shared-input/placement binding, and the selection-phase manifest/checkpoint
placement identity including resume tamper rejection.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dfl_sim import (
    BALANCE_CITATION,
    CHECKPOINT_FILENAME,
    MIDDLE_RANK_CONVENTION,
    PLACEMENT_RANKS,
    EvaluationError,
    HeldOutStudyConfig,
    NamedSchedule,
    PlacementError,
    SelectionSpec,
    TraceReader,
    build_attack_schedule,
    build_cell_config,
    compute_shared_input_digest,
    graph_content_digest,
    graph_structure_digest,
    load_topology,
    parse_study_text,
    placement_contract_digest,
    placement_contract_payload,
    placement_freeze_payload,
    resolve_placement,
    run_baseline_study,
    run_final_phase,
    run_selection_phase,
    validate_shared_inputs,
)
from helpers import MNISTShapedDataset, make_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BAYESIAN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"

_TUNED_PROVENANCE = "singleton-candidate-frozen-before-final"
_WS = {"kind": "watts_strogatz", "n": 10, "k": 2, "p": 0.3}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _base(**overrides):
    defaults = {
        "rounds": 6,
        "data": {"samples_per_node": 4},
        "training": {"local_epochs": 1, "batch_size": 4},
        "evaluation": {"max_samples": 4, "batch_size": 4},
        "topology": dict(_WS),
        "attack": {"compromised_node": 0, "intensity": "strong"},
    }
    defaults.update(overrides)
    return make_config(**defaults)


def _panel(*, budget=1, selection_seed=7, artifact_dir=None):
    artifact = str(BAYESIAN_FIXTURE if artifact_dir is None else artifact_dir)
    project = {"kind": "project", "title": "project comparator"}
    tuned = {
        "mode": "tuned",
        "candidate_budget": budget,
        "selection_seed": selection_seed,
        "selection_provenance": _TUNED_PROVENANCE,
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


def _placement_study(
    *,
    placement="low",
    seeds=(11, 12),
    rounds=6,
    require_varied=False,
    topology=None,
):
    mapping = {
        "study_id": "placement_fixture",
        "data_provenance": "injected",
        "base": _base(rounds=rounds, topology=topology or dict(_WS)).model_dump(
            mode="json"
        ),
        "seeds": list(seeds),
        "candidate_budget": 1,
        "selection_seed": 7,
        "selection_provenance": _TUNED_PROVENANCE,
        "panel": _panel(budget=1),
        "placement": {
            "placement": placement,
            "graph_kind": "watts_strogatz",
            "require_varied_degrees": require_varied,
        },
    }
    return parse_study_text(yaml.safe_dump(mapping, sort_keys=True))


def _ws_config(seed=2101, *, n=10, k=2, p=0.3):
    return make_config(
        seed=seed,
        topology={"kind": "watts_strogatz", "n": n, "k": k, "p": p},
        attack={"compromised_node": 0, "intensity": "strong"},
    )


def _dataset():
    return MNISTShapedDataset(64, seed=1)


def _evaluation_dataset():
    return MNISTShapedDataset(32, seed=2)


# --------------------------------------------------------------------------- #
# Pure resolver
# --------------------------------------------------------------------------- #
def test_middle_rank_convention_is_lower_middle() -> None:
    from dfl_sim.study_placement import middle_rank_index

    assert MIDDLE_RANK_CONVENTION == "lower-middle"
    assert middle_rank_index(10) == 4
    assert middle_rank_index(3) == 1
    assert middle_rank_index(2) == 0
    assert middle_rank_index(1) == 0
    with pytest.raises(PlacementError):
        middle_rank_index(0)
    with pytest.raises(PlacementError):
        middle_rank_index(True)


def test_canonical_edges_are_min_max_sorted_unique() -> None:
    from dfl_sim.study_placement import canonical_edges

    graph = load_topology(_ws_config())
    edges = canonical_edges(graph)
    assert edges == tuple(sorted(set(edges)))
    assert all(left < right for left, right in edges)
    # Undirected: the raw graph may expose either orientation, but the
    # canonical form is stable.
    assert set(edges) == {
        (min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges()
    }


def test_resolve_placement_ranks_on_realized_graph() -> None:
    graph = load_topology(_ws_config(2101))
    ordered = tuple(
        sorted(
            (int(node) for node in graph.nodes()),
            key=lambda node: (int(graph.degree(node)), node),
        )
    )
    expected_rank = {
        "low": 0,
        "mid": (len(ordered) - 1) // 2,
        "high": len(ordered) - 1,
    }
    for placement in PLACEMENT_RANKS:
        resolution = resolve_placement(_ws_config(2101), placement)
        node = ordered[expected_rank[placement]]
        assert resolution.node == node
        assert resolution.rank == expected_rank[placement]
        assert resolution.degree == int(graph.degree(node))
        assert resolution.node_count == len(ordered)
        assert resolution.ordered_edges == tuple(
            sorted(
                (min(int(u), int(v)), max(int(u), int(v)))
                for u, v in graph.edges()
            )
        )
    # The three realized ranks are distinct degree groups for this seed.
    assert resolve_placement(_ws_config(2101), "low").distinct_degree_groups


def test_resolve_placement_is_deterministic_and_pure() -> None:
    config = _ws_config(1101)
    first = resolve_placement(config, "high")
    second = resolve_placement(config, "high")
    assert first == second
    assert first.graph_digest == second.graph_digest
    assert first.content_digest == second.content_digest
    # Different seeds realize different graphs.
    other = resolve_placement(_ws_config(2101), "high")
    assert other.graph_digest != first.graph_digest


def test_regular_graph_reports_ties_and_fails_closed_when_required() -> None:
    # p=0.0 is a pure ring: every node has degree 2.
    ring = _ws_config(7, p=0.0)
    honest = resolve_placement(ring, "mid")
    assert honest.regular is True
    assert honest.degree_tie is True
    assert honest.distinct_degree_groups is False
    # The three rank labels are still distinct nodes, but not distinct degrees.
    low = resolve_placement(ring, "low")
    mid = resolve_placement(ring, "mid")
    high = resolve_placement(ring, "high")
    assert len({low.node, mid.node, high.node}) == 3
    assert {low.degree, mid.degree, high.degree} == {2}
    with pytest.raises(PlacementError):
        resolve_placement(ring, "low", require_varied=True)


def test_invalid_placement_name_rejected() -> None:
    with pytest.raises(PlacementError):
        resolve_placement(_ws_config(), "highest")
    with pytest.raises(PlacementError):
        resolve_placement(_ws_config(), None)


def test_graph_digests_bind_content() -> None:
    graph = load_topology(_ws_config(2101))
    assert graph_structure_digest(graph) == graph_structure_digest(graph)
    assert graph_content_digest(graph) == graph_content_digest(graph)
    # Structure excludes degrees but content includes them.
    resolution = resolve_placement(_ws_config(2101), "low")
    assert resolution.graph_digest != resolution.content_digest


def test_contract_and_freeze_payload_deterministic() -> None:
    payload = placement_contract_payload("mid")
    assert payload == {"placement": "mid", "graph_kind": "watts_strogatz"}
    assert placement_contract_digest("mid") == placement_contract_digest("mid")
    assert placement_contract_digest("mid") != placement_contract_digest("low")
    resolutions = {
        11: resolve_placement(_ws_config(11), "low"),
        12: resolve_placement(_ws_config(12), "low"),
    }
    frozen = placement_freeze_payload(resolutions)
    assert set(frozen["seeds"]) == {"11", "12"}
    assert frozen == placement_freeze_payload(dict(reversed(list(resolutions.items()))))


# --------------------------------------------------------------------------- #
# Common per-cell config integration
# --------------------------------------------------------------------------- #
def test_build_cell_config_applies_resolved_node() -> None:
    study = _placement_study(placement="high")
    resolution = resolve_placement(_ws_config(11), "high")
    config = build_cell_config(
        study.base, study.base.aggregation, 11, placement="high"
    )
    assert config.attack.compromised_node == resolution.node
    assert config.seed == 11
    # Legacy call (no placement) keeps the base scalar untouched.
    legacy = build_cell_config(study.base, study.base.aggregation, 11)
    assert legacy.attack.compromised_node == study.base.attack.compromised_node


def test_shared_input_digest_binds_placement_but_legacy_unchanged() -> None:
    config = _ws_config(11)
    legacy = compute_shared_input_digest(config)
    assert legacy == compute_shared_input_digest(config, placement=None)
    payload = {
        "contract": placement_contract_payload("low"),
        "resolutions": placement_freeze_payload(
            {11: resolve_placement(config, "low")}
        ),
    }
    bound = compute_shared_input_digest(config, placement=payload)
    assert bound != legacy
    # Stable under a caller-order-independent rebuild.
    assert bound == compute_shared_input_digest(
        config,
        placement={
            "contract": placement_contract_payload("low"),
            "resolutions": placement_freeze_payload(
                {11: resolve_placement(config, "low")}
            ),
        },
    )


def test_legacy_study_bytes_omit_placement_field() -> None:
    mapping = yaml.safe_load(
        yaml.safe_dump(
            {
                "study_id": "legacy_fixture",
                "data_provenance": "injected",
                "base": _base().model_dump(mode="json"),
                "seeds": [11, 12],
                "candidate_budget": 1,
                "selection_seed": 7,
                "selection_provenance": _TUNED_PROVENANCE,
                "panel": _panel(),
            },
            sort_keys=True,
        )
    )
    study = parse_study_text(yaml.safe_dump(mapping, sort_keys=True))
    assert study.placement is None
    assert "placement" not in study.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Baseline runner integration (manifest + trace identity)
# --------------------------------------------------------------------------- #
def test_baseline_study_manifest_records_realized_placement(tmp_path) -> None:
    study = _placement_study(placement="high", rounds=2)
    output = tmp_path / "placement-baseline"
    result = run_baseline_study(
        study,
        output,
        dataset=_dataset(),
        evaluation_dataset=_evaluation_dataset(),
    )
    manifest = result.manifest
    assert manifest["placement"]["contract"]["placement"] == "high"
    assert set(manifest["placement"]["resolutions"]["seeds"]) == {"11", "12"}
    assert manifest["shared_input_digest"] == compute_shared_input_digest(
        study.base, placement=manifest["placement"]
    )
    # Every cell in a seed shares the one realized node, and the trace snapshot
    # carries exactly that scalar attacker.
    by_seed: dict[int, set[int]] = {}
    for entry in manifest["entries"]:
        seed = int(entry["seed"])
        placement = entry["placement"]
        assert placement["graph_digest"]
        assert placement["node"] == manifest["placement"]["resolutions"][
            "seeds"
        ][str(seed)]["node"]
        by_seed.setdefault(seed, set()).add(int(placement["node"]))
    for seed, nodes in by_seed.items():
        assert len(nodes) == 1
    for cell in result.cells:
        with TraceReader(cell.trace_path) as reader:
            assert reader.config.attack.compromised_node == int(
                cell.manifest_entry["placement"]["node"]
            )


def test_inconsistent_placement_binding_fails_shared_inputs() -> None:
    # Two seed configs with different realized nodes cannot share the *base*
    # digest when a caller wrongly inserts the resolved node into the digest.
    low = build_cell_config(
        _placement_study(placement="low", seeds=(11, 12)).base,
        _placement_study().base.aggregation,
        11,
        placement="low",
    )
    high = build_cell_config(
        _placement_study(placement="high", seeds=(11, 12)).base,
        _placement_study().base.aggregation,
        11,
        placement="high",
    )
    with pytest.raises(EvaluationError):
        validate_shared_inputs([low, high])


# --------------------------------------------------------------------------- #
# Selection runner + checkpoint placement identity
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def selection_fixture(tmp_path_factory):
    study = _placement_study(placement="low", seeds=(11, 12), rounds=6)
    spec = SelectionSpec(
        study=study,
        known_schedules=(
            NamedSchedule(
                id="periodic_known", schedule=build_attack_schedule("periodic")
            ),
        ),
    )
    output = tmp_path_factory.mktemp("placement-selection") / "selection"
    result = run_selection_phase(
        spec,
        output,
        dataset=_dataset(),
        evaluation_dataset=_evaluation_dataset(),
    )
    return SimpleNamespace(spec=spec, output=output, result=result)


def _read_checkpoint(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_selection_manifest_and_checkpoint_bind_placement(
    selection_fixture,
) -> None:
    manifest = selection_fixture.result.manifest
    assert manifest["placement"]["contract"]["placement"] == "low"
    assert manifest["placement"]["contract_digest"] == placement_contract_digest(
        "low"
    )
    expected_contract = placement_contract_digest("low")
    records = _read_checkpoint(
        selection_fixture.output / CHECKPOINT_FILENAME
    )
    assert records
    for record in records:
        assert record["placement_contract_digest"] == expected_contract
        assert record["placement"]["placement"] == "low"
    for entry in manifest["entries"]:
        assert entry["placement"]["placement"] == "low"
        assert entry["placement"]["rank"] == 0
    # All policies/schedules within one seed use the same realized node.
    nodes_by_seed: dict[int, set[int]] = {}
    for entry in manifest["entries"]:
        nodes_by_seed.setdefault(int(entry["seed"]), set()).add(
            int(entry["placement"]["node"])
        )
    for seed, nodes in nodes_by_seed.items():
        assert len(nodes) == 1


def test_selection_traces_carry_resolved_attacker(selection_fixture) -> None:
    for cell in selection_fixture.result.cells:
        with TraceReader(cell.trace_path) as reader:
            assert reader.config.attack.compromised_node == int(
                cell.manifest_entry["placement"]["node"]
            )


def test_full_selection_resume_reuses_all_cells(
    tmp_path, monkeypatch, selection_fixture
) -> None:
    import dfl_sim.evaluation as evaluation_module

    def forbidden(*args, **kwargs):
        raise AssertionError("run_from_config must not run on full reuse")

    monkeypatch.setattr(evaluation_module, "run_from_config", forbidden)
    output = tmp_path / "resumed-selection"
    result = run_selection_phase(
        selection_fixture.spec,
        output,
        dataset=_dataset(),
        evaluation_dataset=_evaluation_dataset(),
        resume_from=selection_fixture.output,
    )
    assert all(
        entry.get("reused") is True for entry in result.manifest["entries"]
    )


def test_tampered_placement_identity_fails_closed(
    tmp_path, selection_fixture
) -> None:
    import shutil

    clone = tmp_path / "tampered-placement"
    shutil.copytree(selection_fixture.output, clone)
    checkpoint = clone / CHECKPOINT_FILENAME
    records = _read_checkpoint(checkpoint)
    records[0]["placement"]["node"] = int(records[0]["placement"]["node"]) + 1
    checkpoint.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    output = tmp_path / "tampered-placement-output"
    with pytest.raises(EvaluationError) as info:
        run_selection_phase(
            selection_fixture.spec,
            output,
            dataset=_dataset(),
            evaluation_dataset=_evaluation_dataset(),
            resume_from=clone,
        )
    assert "placement" in str(info.value)
    assert not output.exists()


def test_tampered_placement_contract_digest_fails_closed(
    tmp_path, selection_fixture
) -> None:
    import shutil

    clone = tmp_path / "tampered-contract"
    shutil.copytree(selection_fixture.output, clone)
    checkpoint = clone / CHECKPOINT_FILENAME
    records = _read_checkpoint(checkpoint)
    records[0]["placement_contract_digest"] = "0" * 64
    checkpoint.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    output = tmp_path / "tampered-contract-output"
    with pytest.raises(EvaluationError) as info:
        run_selection_phase(
            selection_fixture.spec,
            output,
            dataset=_dataset(),
            evaluation_dataset=_evaluation_dataset(),
            resume_from=clone,
        )
    assert "placement_contract_digest" in str(info.value)
    assert not output.exists()


# --------------------------------------------------------------------------- #
# Final-phase placement
# --------------------------------------------------------------------------- #
def test_final_phase_uses_resolved_placement(selection_fixture, tmp_path) -> None:
    spec = HeldOutStudyConfig(
        study=selection_fixture.spec.study,
        final_seeds=(21, 22, 23),
        known_schedules=selection_fixture.spec.known_schedules,
        held_out_schedule=NamedSchedule(
            id="held_out", schedule=build_attack_schedule("held_out")
        ),
    )
    final_output = tmp_path / "final-placement"
    result = run_final_phase(
        spec,
        selection_fixture.output,
        final_output,
        dataset=_dataset(),
        evaluation_dataset=_evaluation_dataset(),
    )
    manifest = result.manifest
    assert manifest["placement"]["contract"]["placement"] == "low"
    assert set(manifest["placement"]["resolutions"]["seeds"]) == {
        "21",
        "22",
        "23",
    }
    for entry in manifest["entries"]:
        assert entry["placement"]["placement"] == "low"
        assert entry["placement"]["graph_digest"]
    for cell in result.cells:
        with TraceReader(cell.trace_path) as reader:
            assert reader.config.attack.compromised_node == int(
                cell.manifest_entry["placement"]["node"]
            )
