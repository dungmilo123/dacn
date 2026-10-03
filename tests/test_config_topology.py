"""Contract tests for T01 configuration, seed resolution, and topology.

Covers R002 topology scope, R012 determinism/no-aliasing evidence at the
config layer, and the Q7 negative surface (unsafe YAML, unknown fields,
invalid discriminators, out-of-bounds numbers, inconsistent graph params).

These tests are offline and never download data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
import yaml

from dfl_sim import (
    ATTACK_KAPPAS,
    MAX_EVAL_BATCH_SIZE,
    MAX_EVAL_SAMPLES,
    MAX_ROUNDS,
    MAX_SAMPLES_PER_NODE,
    MAX_SNAPSHOT_CADENCE,
    AttackConfig,
    Chain3Topology,
    ConfigError,
    DeterminismError,
    EvaluationConfig,
    ExperimentConfig,
    FixedSimilarityAggregation,
    FixedSimilarityFilterPolicy,
    Line2Topology,
    LocalOnlyAggregation,
    LocalOnlyPolicy,
    PlainAverageAggregation,
    PlainAveragePolicy,
    TraceConfig,
    WattsStrogatzTopology,
    build_policy,
    canonical_dump,
    configure_determinism,
    load_config,
    load_topology,
    parse_config_text,
    resolve_seeds,
    stable_seed,
    topology_node_count,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_3NODE = REPO_ROOT / "configs" / "clean_3node.yaml"
POISON_3NODE = REPO_ROOT / "configs" / "fixed_poison_3node.yaml"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _base_dict() -> dict:
    return {
        "seed": 20240101,
        "rounds": 2,
        "device": "cpu",
        "bitwise": True,
        "model": {"name": "small_cnn"},
        "data": {
            "dataset": "mnist",
            "partition": "iid",
            "samples_per_node": 64,
            "data_root": "data",
            "download": False,
        },
        "training": {
            "optimizer": "sgd",
            "learning_rate": 0.05,
            "batch_size": 16,
            "local_epochs": 1,
            "num_workers": 0,
        },
        "topology": {"kind": "chain3"},
    }


def make(**overrides) -> ExperimentConfig:
    """Build a validated config from the base dict with shallow overrides."""

    data = _base_dict()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return parse_config_text(yaml.safe_dump(data, sort_keys=True))


def raw(text: str) -> ExperimentConfig:
    return parse_config_text(text)


# --------------------------------------------------------------------------- #
# Happy path + canonical dump
# --------------------------------------------------------------------------- #
def test_clean_config_loads() -> None:
    config = load_config(CLEAN_3NODE)
    assert isinstance(config, ExperimentConfig)
    assert config.topology.kind == "chain3"
    assert config.rounds >= 1
    assert config.bitwise is True
    assert config.data.partition == "iid"


def test_canonical_dump_is_stable_and_json() -> None:
    first = canonical_dump(load_config(CLEAN_3NODE))
    second = canonical_dump(load_config(CLEAN_3NODE))
    assert first == second
    parsed = json.loads(first)
    assert parsed["topology"]["kind"] == "chain3"
    # Canonical form is compact and sorted.
    assert "\n" not in first
    assert ": " not in first
    assert first.index('"bitwise"') < first.index('"data"')


def test_canonical_dump_equal_across_independent_yaml_loads() -> None:
    text = yaml.safe_dump(_base_dict(), sort_keys=True)
    a = canonical_dump(parse_config_text(text))
    b = canonical_dump(parse_config_text(text))
    assert a == b


def test_defaults_fill_optional_fields() -> None:
    config = make()
    assert config.device == "cpu"
    assert config.model.name == "small_cnn"
    assert config.data.data_root == "data"
    assert config.data.download is False
    assert config.training.num_workers == 0


# --------------------------------------------------------------------------- #
# Malformed / unsafe YAML (Q3 + Q7)
# --------------------------------------------------------------------------- #
def test_missing_file_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(REPO_ROOT / "configs" / "does-not-exist.yaml")


def test_malformed_yaml_rejected() -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        raw("seed: 1\nrounds: [unclosed\n")


def test_empty_yaml_rejected() -> None:
    with pytest.raises(ConfigError, match="empty config"):
        raw("")


def test_non_mapping_root_rejected() -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        raw("- just\n- a\n- list\n")


def test_unsafe_yaml_tag_rejected() -> None:
    # safe_load must refuse arbitrary Python object construction.
    payload = "!!python/object/apply:os.system ['echo pwned']\n"
    with pytest.raises(ConfigError, match="invalid YAML"):
        raw(payload)


def test_custom_tag_rejected() -> None:
    payload = "!!python/name:os.system ''\n"
    with pytest.raises(ConfigError, match="invalid YAML"):
        raw(payload)


# --------------------------------------------------------------------------- #
# Unknown fields / invalid discriminators (Q3 + Q7)
# --------------------------------------------------------------------------- #
def test_unknown_top_level_field_rejected() -> None:
    with pytest.raises(ConfigError):
        make(bogus=1)


def test_unknown_nested_field_rejected() -> None:
    with pytest.raises(ConfigError):
        make(training={"unknown_knob": 3})


def test_invalid_dataset_rejected() -> None:
    with pytest.raises(ConfigError):
        make(data={"dataset": "cifar10"})


def test_non_iid_partition_rejected() -> None:
    with pytest.raises(ConfigError):
        make(data={"partition": "dirichlet"})


def test_invalid_model_rejected() -> None:
    with pytest.raises(ConfigError):
        make(model={"name": "resnet"})


def test_unsupported_device_rejected() -> None:
    with pytest.raises(ConfigError):
        make(device="cuda")


def test_invalid_optimizer_rejected() -> None:
    with pytest.raises(ConfigError):
        make(training={"optimizer": "rmsprop"})


# --------------------------------------------------------------------------- #
# Numeric bounds (Q6 + Q7)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rounds", [0, -1, MAX_ROUNDS + 1])
def test_invalid_rounds_rejected(rounds: int) -> None:
    with pytest.raises(ConfigError):
        make(rounds=rounds)


@pytest.mark.parametrize("samples", [0, -5, MAX_SAMPLES_PER_NODE + 1])
def test_invalid_samples_per_node_rejected(samples: int) -> None:
    with pytest.raises(ConfigError):
        make(data={"samples_per_node": samples})


@pytest.mark.parametrize("batch", [0, -1, 1025])
def test_invalid_batch_size_rejected(batch: int) -> None:
    with pytest.raises(ConfigError):
        make(training={"batch_size": batch})


@pytest.mark.parametrize("epochs", [0, -1, 51])
def test_invalid_local_epochs_rejected(epochs: int) -> None:
    with pytest.raises(ConfigError):
        make(training={"local_epochs": epochs})


@pytest.mark.parametrize("lr", [0.0, -0.1, 1.5])
def test_invalid_learning_rate_rejected(lr: float) -> None:
    with pytest.raises(ConfigError):
        make(training={"learning_rate": lr})


@pytest.mark.parametrize("seed", [-1, 2**31])
def test_invalid_seed_rejected(seed: int) -> None:
    with pytest.raises(ConfigError):
        make(seed=seed)


def test_bitwise_requires_zero_workers() -> None:
    with pytest.raises(ConfigError, match="num_workers"):
        make(training={"num_workers": 2})


def test_non_bitwise_allows_workers() -> None:
    config = make(bitwise=False, training={"num_workers": 2})
    assert config.bitwise is False
    assert config.training.num_workers == 2


# --------------------------------------------------------------------------- #
# Topology discriminators + param validation (Q7)
# --------------------------------------------------------------------------- #
def test_line2_rejects_node_override() -> None:
    with pytest.raises(ConfigError):
        make(topology={"kind": "line2", "n": 2})


def test_chain3_rejects_node_override() -> None:
    with pytest.raises(ConfigError):
        make(topology={"kind": "chain3", "nodes": 3})


def test_unknown_topology_kind_rejected() -> None:
    with pytest.raises(ConfigError):
        make(topology={"kind": "complete"})


@pytest.mark.parametrize("n", [9, 21, 0, -3])
def test_ws_node_count_out_of_range_rejected(n: int) -> None:
    with pytest.raises(ConfigError):
        make(topology={"kind": "watts_strogatz", "n": n, "k": 4, "p": 0.2})


@pytest.mark.parametrize("k", [3, 5, 11])
def test_ws_odd_degree_rejected(k: int) -> None:
    with pytest.raises(ConfigError, match="even"):
        make(topology={"kind": "watts_strogatz", "n": 12, "k": k, "p": 0.2})


def test_ws_degree_must_be_less_than_n() -> None:
    with pytest.raises(ConfigError, match="k < n"):
        make(topology={"kind": "watts_strogatz", "n": 10, "k": 10, "p": 0.2})


@pytest.mark.parametrize("k", [0, 1, -2])
def test_ws_degree_too_small_rejected(k: int) -> None:
    with pytest.raises(ConfigError):
        make(topology={"kind": "watts_strogatz", "n": 12, "k": k, "p": 0.2})


@pytest.mark.parametrize("p", [-0.1, 1.1, 2.0])
def test_ws_probability_out_of_range_rejected(p: float) -> None:
    with pytest.raises(ConfigError):
        make(topology={"kind": "watts_strogatz", "n": 12, "k": 4, "p": p})


def test_ws_boundary_probabilities_accepted() -> None:
    assert make(topology={"kind": "watts_strogatz", "n": 10, "k": 4, "p": 0.0}).topology.p == 0.0
    assert make(topology={"kind": "watts_strogatz", "n": 20, "k": 4, "p": 1.0}).topology.p == 1.0


# --------------------------------------------------------------------------- #
# Exact topology edges (R002)
# --------------------------------------------------------------------------- #
def test_line2_exact_edges() -> None:
    config = make(topology={"kind": "line2"})
    assert isinstance(config.topology, Line2Topology)
    graph = load_topology(config)
    assert sorted(graph.nodes()) == [0, 1]
    assert sorted(map(sorted, graph.edges())) == [[0, 1]]


def test_chain3_exact_edges() -> None:
    config = make(topology={"kind": "chain3"})
    assert isinstance(config.topology, Chain3Topology)
    graph = load_topology(config)
    assert sorted(graph.nodes()) == [0, 1, 2]
    assert sorted(map(sorted, graph.edges())) == [[0, 1], [1, 2]]


@pytest.mark.parametrize("n", [10, 14, 20])
def test_ws_connected_contiguous_and_even_degree(n: int) -> None:
    config = make(topology={"kind": "watts_strogatz", "n": n, "k": 4, "p": 0.2})
    assert isinstance(config.topology, WattsStrogatzTopology)
    graph = load_topology(config)
    assert graph.number_of_nodes() == n
    assert sorted(graph.nodes()) == list(range(n))
    assert nx.is_connected(graph)
    # The configured ring degree is even; connectivity is the invariant that
    # matters across rewiring (rewiring can change individual degrees).
    assert config.topology.k == 4
    assert config.topology.k % 2 == 0


def test_ws_repeatable_across_independent_loads() -> None:
    text = yaml.safe_dump(
        _base_dict() | {"topology": {"kind": "watts_strogatz", "n": 16, "k": 4, "p": 0.3}},
        sort_keys=True,
    )
    first = load_topology(parse_config_text(text))
    second = load_topology(parse_config_text(text))
    assert sorted(first.edges()) == sorted(second.edges())
    assert first.graph["generator_seed"] == second.graph["generator_seed"]


def test_ws_seed_change_changes_generator_seed() -> None:
    a = make(seed=1, topology={"kind": "watts_strogatz", "n": 14, "k": 4, "p": 0.2})
    b = make(seed=2, topology={"kind": "watts_strogatz", "n": 14, "k": 4, "p": 0.2})
    assert resolve_seeds(a).topology != resolve_seeds(b).topology


def test_topology_is_frozen_and_detached() -> None:
    config = make(topology={"kind": "chain3"})
    graph = load_topology(config)
    assert getattr(graph, "frozen", False) is True
    with pytest.raises(nx.NetworkXError):
        graph.add_node(99)
    # A second call returns an equal but separate snapshot.
    other = load_topology(config)
    assert graph is not other
    assert sorted(graph.edges()) == sorted(other.edges())
    assert graph.graph["kind"] == "chain3"
    assert graph.graph["node_count"] == 3


# --------------------------------------------------------------------------- #
# Seed resolution (D009)
# --------------------------------------------------------------------------- #
def test_stable_seed_is_deterministic_and_label_sensitive() -> None:
    assert stable_seed(7, "alpha") == stable_seed(7, "alpha")
    assert stable_seed(7, "alpha") != stable_seed(7, "beta")
    assert stable_seed(7, "alpha") != stable_seed(8, "alpha")
    assert 0 <= stable_seed(2**31 - 1, "x") < 2**31


def test_resolved_seeds_stable_across_independent_loads() -> None:
    text = yaml.safe_dump(_base_dict(), sort_keys=True)
    assert resolve_seeds(parse_config_text(text)) == resolve_seeds(parse_config_text(text))


def test_resolved_seed_domains_are_distinct() -> None:
    seeds = resolve_seeds(make())
    values = [seeds.python, seeds.numpy, seeds.torch, seeds.topology, seeds.partition]
    assert len(set(values)) == len(values)
    assert all(value != seeds.master for value in values)


def test_resolved_seed_changes_with_master_seed() -> None:
    assert resolve_seeds(make(seed=10)).torch != resolve_seeds(make(seed=11)).torch


def test_local_seeds_are_stable_and_per_node_round() -> None:
    seeds = resolve_seeds(make())
    assert seeds.for_local_train(0, 0) == seeds.for_local_train(0, 0)
    assert seeds.for_local_train(0, 0) != seeds.for_local_train(0, 1)
    assert seeds.for_local_train(0, 0) != seeds.for_local_train(1, 0)
    assert seeds.for_data_order(0, 0) != seeds.for_local_train(0, 0)


# --------------------------------------------------------------------------- #
# Deterministic setup (D009)
# --------------------------------------------------------------------------- #
def test_configure_determinism_is_repeatable() -> None:
    seeds = resolve_seeds(make())
    configure_determinism(seeds)
    py_a = [random.random() for _ in range(4)]
    np_a = np.random.rand(4).tolist()
    torch_a = torch.rand(4)
    configure_determinism(seeds)
    py_b = [random.random() for _ in range(4)]
    np_b = np.random.rand(4).tolist()
    torch_b = torch.rand(4)
    assert py_a == py_b
    assert np_a == np_b
    assert torch.equal(torch_a, torch_b)


def test_configure_determinism_report() -> None:
    seeds = resolve_seeds(make())
    report = configure_determinism(seeds, bitwise=True, num_workers=0)
    assert report.bitwise is True
    assert report.master_seed == seeds.master
    assert report.num_threads == 1
    assert report.num_workers == 0
    assert report.deterministic_algorithms is True


def test_configure_determinism_rejects_workers_in_bitwise() -> None:
    seeds = resolve_seeds(make())
    with pytest.raises(DeterminismError, match="num_workers"):
        configure_determinism(seeds, bitwise=True, num_workers=2)


def test_configure_determinism_lenient_mode() -> None:
    report = configure_determinism(resolve_seeds(make()), bitwise=False, num_workers=2)
    assert report.bitwise is False
    assert report.deterministic_algorithms is False


# --------------------------------------------------------------------------- #
# Fixed attack config (R003, D007, D016) + Q7 negative surface
# --------------------------------------------------------------------------- #
def test_default_attack_is_zero_and_unselected() -> None:
    config = make()
    assert isinstance(config.attack, AttackConfig)
    assert config.attack.compromised_node is None
    assert config.attack.intensity == "zero"
    assert config.attack.kappa == 0.0
    assert config.attack.kappa in ATTACK_KAPPAS.values()


def test_attack_kappa_table_is_exact() -> None:
    assert ATTACK_KAPPAS == {"zero": 0.0, "weak": 2.0, "strong": 5.0}


def test_old_config_without_attack_key_still_loads() -> None:
    # ``_base_dict`` predates the attack block; the strict default must keep it
    # loadable byte-for-byte in behaviour.
    config = parse_config_text(yaml.safe_dump(_base_dict(), sort_keys=True))
    assert config.attack.compromised_node is None
    assert config.attack.intensity == "zero"


def test_selected_node_with_zero_intensity_allowed() -> None:
    config = make(attack={"compromised_node": 0, "intensity": "zero"})
    assert config.attack.compromised_node == 0
    assert config.attack.kappa == 0.0


@pytest.mark.parametrize(
    "intensity, kappa", [("weak", 2.0), ("strong", 5.0)]
)
def test_selected_node_with_active_intensity(intensity: str, kappa: float) -> None:
    config = make(attack={"compromised_node": 2, "intensity": intensity})
    assert config.attack.compromised_node == 2
    assert config.attack.kappa == kappa


def test_active_intensity_without_selected_node_rejected() -> None:
    with pytest.raises(ConfigError, match="compromised_node"):
        make(attack={"intensity": "weak"})
    with pytest.raises(ConfigError, match="compromised_node"):
        make(attack={"intensity": "strong"})


def test_unknown_attack_intensity_rejected() -> None:
    with pytest.raises(ConfigError):
        make(attack={"compromised_node": 0, "intensity": "max"})


def test_unknown_attack_field_rejected() -> None:
    with pytest.raises(ConfigError):
        make(attack={"compromised_node": 0, "kappa": 3.0})


def test_attack_node_wrong_type_rejected() -> None:
    with pytest.raises(ConfigError):
        make(attack={"compromised_node": "0"})
    with pytest.raises(ConfigError):
        make(attack={"compromised_node": [0]})
    with pytest.raises(ConfigError):
        make(attack={"compromised_node": 1.5})


@pytest.mark.parametrize("node", [-1, 3, 99])
def test_chain3_attack_node_out_of_range_rejected(node: int) -> None:
    # Negative ids fail the scalar attacker's own guard; ids above the
    # resolved topology node count fail the experiment-level range check.
    with pytest.raises(ConfigError, match="node range|non-negative"):
        make(
            topology={"kind": "chain3"},
            attack={"compromised_node": node, "intensity": "weak"},
        )


def test_line2_attack_node_out_of_range_rejected() -> None:
    # Node 1 is valid on line2; node 2 is not.
    assert (
        make(
            topology={"kind": "line2"},
            attack={"compromised_node": 1, "intensity": "strong"},
        ).attack.compromised_node
        == 1
    )
    with pytest.raises(ConfigError, match="node range"):
        make(
            topology={"kind": "line2"},
            attack={"compromised_node": 2, "intensity": "strong"},
        )


def test_ws_attack_node_validated_against_configured_n() -> None:
    ws = {"kind": "watts_strogatz", "n": 12, "k": 4, "p": 0.2}
    config = make(
        topology=ws,
        attack={"compromised_node": 11, "intensity": "weak"},
    )
    assert config.attack.compromised_node == 11
    with pytest.raises(ConfigError, match="node range"):
        make(
            topology=ws,
            attack={"compromised_node": 12, "intensity": "weak"},
        )


def test_topology_node_count_matches_families() -> None:
    assert topology_node_count(make(topology={"kind": "line2"}).topology) == 2
    assert topology_node_count(make(topology={"kind": "chain3"}).topology) == 3
    assert (
        topology_node_count(
            make(
                topology={"kind": "watts_strogatz", "n": 14, "k": 4, "p": 0.2}
            ).topology
        )
        == 14
    )


def test_attack_survives_canonical_dump() -> None:
    config = make(attack={"compromised_node": 1, "intensity": "strong"})
    parsed = json.loads(canonical_dump(config))
    assert parsed["attack"] == {"compromised_node": 1, "intensity": "strong"}
    assert canonical_dump(config) == canonical_dump(
        make(attack={"compromised_node": 1, "intensity": "strong"})
    )


def test_attack_config_is_frozen() -> None:
    config = make(attack={"compromised_node": 0, "intensity": "weak"})
    with pytest.raises(Exception):
        config.attack.intensity = "strong"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Aggregation selector (R005/R006/R007, D008/D017) + Q7 negative surface
# --------------------------------------------------------------------------- #
def test_default_aggregation_is_plain_average() -> None:
    # ``_base_dict`` predates the aggregation union; the strict default must
    # keep existing programmatic configs on the D015 baseline.
    config = make()
    assert isinstance(config.aggregation, PlainAverageAggregation)
    assert config.aggregation.kind == "plain_average"


@pytest.mark.parametrize("kind", ["local_only", "plain_average"])
def test_simple_aggregation_kinds_load(kind: str) -> None:
    config = make(aggregation={"kind": kind})
    assert config.aggregation.kind == kind


def test_local_only_aggregation_type() -> None:
    config = make(aggregation={"kind": "local_only"})
    assert isinstance(config.aggregation, LocalOnlyAggregation)


def test_fixed_similarity_aggregation_loads() -> None:
    config = make(
        aggregation={
            "kind": "fixed_similarity",
            "cutoff": 0.5,
            "downweight_multiplier": 0.25,
        }
    )
    assert isinstance(config.aggregation, FixedSimilarityAggregation)
    assert config.aggregation.cutoff == 0.5
    assert config.aggregation.downweight_multiplier == 0.25


def test_unknown_aggregation_kind_rejected() -> None:
    with pytest.raises(ConfigError):
        make(aggregation={"kind": "complete"})


def test_unknown_aggregation_field_rejected() -> None:
    with pytest.raises(ConfigError):
        make(aggregation={"kind": "plain_average", "cutoff": 0.5})


def test_fixed_similarity_requires_cutoff_and_downweight() -> None:
    with pytest.raises(ConfigError):
        make(
            aggregation={
                "kind": "fixed_similarity",
                "downweight_multiplier": 0.5,
            }
        )
    with pytest.raises(ConfigError):
        make(aggregation={"kind": "fixed_similarity", "cutoff": 0.5})


@pytest.mark.parametrize("cutoff", [-0.1, 2.1, float("nan"), float("inf")])
def test_fixed_similarity_cutoff_out_of_range_rejected(cutoff: float) -> None:
    with pytest.raises(ConfigError):
        make(
            aggregation={
                "kind": "fixed_similarity",
                "cutoff": cutoff,
                "downweight_multiplier": 0.5,
            }
        )


def test_fixed_similarity_cutoff_boundaries_accepted() -> None:
    low = make(
        aggregation={
            "kind": "fixed_similarity",
            "cutoff": 0.0,
            "downweight_multiplier": 0.5,
        }
    )
    high = make(
        aggregation={
            "kind": "fixed_similarity",
            "cutoff": 2.0,
            "downweight_multiplier": 0.5,
        }
    )
    assert low.aggregation.cutoff == 0.0
    assert high.aggregation.cutoff == 2.0


@pytest.mark.parametrize(
    "downweight", [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")]
)
def test_fixed_similarity_downweight_out_of_range_rejected(
    downweight: float,
) -> None:
    with pytest.raises(ConfigError):
        make(
            aggregation={
                "kind": "fixed_similarity",
                "cutoff": 0.5,
                "downweight_multiplier": downweight,
            }
        )


def test_aggregation_survives_canonical_dump() -> None:
    config = make(
        aggregation={
            "kind": "fixed_similarity",
            "cutoff": 0.5,
            "downweight_multiplier": 0.25,
        }
    )
    parsed = json.loads(canonical_dump(config))
    assert parsed["aggregation"] == {
        "kind": "fixed_similarity",
        "cutoff": 0.5,
        "downweight_multiplier": 0.25,
    }
    assert canonical_dump(config) == canonical_dump(
        make(
            aggregation={
                "kind": "fixed_similarity",
                "cutoff": 0.5,
                "downweight_multiplier": 0.25,
            }
        )
    )


def test_aggregation_config_is_frozen() -> None:
    config = make(aggregation={"kind": "plain_average"})
    with pytest.raises(Exception):
        config.aggregation.kind = "local_only"  # type: ignore[misc]


def test_build_policy_selects_each_baseline() -> None:
    assert isinstance(
        build_policy(make(aggregation={"kind": "local_only"}).aggregation),
        LocalOnlyPolicy,
    )
    assert isinstance(
        build_policy(make(aggregation={"kind": "plain_average"}).aggregation),
        PlainAveragePolicy,
    )
    policy = build_policy(
        make(
            aggregation={
                "kind": "fixed_similarity",
                "cutoff": 0.5,
                "downweight_multiplier": 0.25,
            }
        ).aggregation
    )
    assert isinstance(policy, FixedSimilarityFilterPolicy)
    assert policy.cutoff == 0.5
    assert policy.downweight_multiplier == 0.25


def test_checked_in_configs_declare_plain_average() -> None:
    for path in (CLEAN_3NODE, POISON_3NODE):
        config = load_config(path)
        assert config.aggregation.kind == "plain_average"
        # The checked-in YAML spells the selector out rather than relying on
        # the programmatic default.
        assert "aggregation:" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Evaluation + trace config (R008/R009, D018) + Q7 negative surface
# --------------------------------------------------------------------------- #
def test_evaluation_and_trace_defaults_apply_and_dump() -> None:
    config = make()
    assert isinstance(config.evaluation, EvaluationConfig)
    assert config.evaluation.max_samples == 512
    assert config.evaluation.batch_size == 64
    assert isinstance(config.trace, TraceConfig)
    assert config.trace.snapshot_cadence == 1
    parsed = json.loads(canonical_dump(config))
    assert parsed["evaluation"] == {"max_samples": 512, "batch_size": 64}
    assert parsed["trace"] == {"snapshot_cadence": 1}


def test_old_config_without_evaluation_or_trace_still_loads() -> None:
    # ``_base_dict`` predates both blocks; the strict defaults must keep it
    # loadable without behaviour change.
    config = parse_config_text(yaml.safe_dump(_base_dict(), sort_keys=True))
    assert config.evaluation.max_samples == 512
    assert config.evaluation.batch_size == 64
    assert config.trace.snapshot_cadence == 1


def test_explicit_evaluation_and_trace_load() -> None:
    config = make(
        evaluation={"max_samples": 10, "batch_size": 2},
        trace={"snapshot_cadence": 3},
    )
    assert config.evaluation.max_samples == 10
    assert config.evaluation.batch_size == 2
    assert config.trace.snapshot_cadence == 3
    parsed = json.loads(canonical_dump(config))
    assert parsed["evaluation"] == {"max_samples": 10, "batch_size": 2}
    assert parsed["trace"] == {"snapshot_cadence": 3}


def test_unknown_evaluation_field_rejected() -> None:
    with pytest.raises(ConfigError):
        make(evaluation={"bogus": 1})


def test_unknown_trace_field_rejected() -> None:
    with pytest.raises(ConfigError):
        make(trace={"bogus": 1})


@pytest.mark.parametrize("samples", [0, -1, MAX_EVAL_SAMPLES + 1])
def test_invalid_eval_max_samples_rejected(samples: int) -> None:
    with pytest.raises(ConfigError):
        make(evaluation={"max_samples": samples})


@pytest.mark.parametrize("batch", [0, -1, MAX_EVAL_BATCH_SIZE + 1])
def test_invalid_eval_batch_size_rejected(batch: int) -> None:
    with pytest.raises(ConfigError):
        make(evaluation={"batch_size": batch})


@pytest.mark.parametrize(
    "evaluation",
    [
        {"max_samples": "many", "batch_size": 1},
        {"max_samples": 1, "batch_size": 1.5},
        {"max_samples": float("nan"), "batch_size": 1},
        {"max_samples": True, "batch_size": 1},
    ],
)
def test_eval_wrong_types_rejected(evaluation: dict) -> None:
    with pytest.raises(ConfigError):
        make(evaluation=evaluation)


@pytest.mark.parametrize("cadence", [0, -1, MAX_SNAPSHOT_CADENCE + 1])
def test_invalid_snapshot_cadence_rejected(cadence: int) -> None:
    with pytest.raises(ConfigError):
        make(trace={"snapshot_cadence": cadence})


@pytest.mark.parametrize("cadence", [1.5, "1", True])
def test_snapshot_cadence_wrong_type_rejected(cadence: object) -> None:
    with pytest.raises(ConfigError):
        make(trace={"snapshot_cadence": cadence})


def test_evaluation_and_trace_boundaries_accepted() -> None:
    low = make(evaluation={"max_samples": 1, "batch_size": 1})
    high = make(
        evaluation={"max_samples": MAX_EVAL_SAMPLES, "batch_size": MAX_EVAL_BATCH_SIZE},
        trace={"snapshot_cadence": MAX_SNAPSHOT_CADENCE},
    )
    assert low.evaluation.max_samples == 1
    assert low.evaluation.batch_size == 1
    assert high.evaluation.max_samples == MAX_EVAL_SAMPLES
    assert high.evaluation.batch_size == MAX_EVAL_BATCH_SIZE
    assert high.trace.snapshot_cadence == MAX_SNAPSHOT_CADENCE


def test_evaluation_and_trace_are_frozen() -> None:
    config = make()
    with pytest.raises(Exception):
        config.evaluation.max_samples = 1  # type: ignore[misc]
    with pytest.raises(Exception):
        config.trace.snapshot_cadence = 2  # type: ignore[misc]
