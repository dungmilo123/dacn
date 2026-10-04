"""Deterministic bounded Dirichlet partition tests (T01, R019/D039).

All fixtures are generated in-process from a seeded RNG; nothing downloads data
or reads ``.gsd``/gitignored runs. The suite covers the opt-in config contract,
the capacity-constrained class-wise allocator (exact quotas, determinism,
order-independence, alpha sensitivity, disjoint design complement), the
library-level dispatch used by both production simulation and calibration, the
reconstructible realized-label/digest metadata, and the fail-closed negative
surface.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Subset

from dfl_sim import (
    ConfigError,
    DataError,
    MNIST_NUM_CLASSES,
    build_schedule,
    canonical_dump,
    collect_calibration,
    collect_episode,
    derive_experiment_config,
    extract_dataset_labels,
    partition_dataset,
    partition_dataset_with_design,
    partition_indices,
    partition_indices_dirichlet,
    partition_indices_dirichlet_with_design,
    partition_membership_digest,
    realized_label_counts,
    resolve_seeds,
)
from helpers import MNISTShapedDataset, make_config
from test_calibration_config import base_dict, parse_dict

NODES = [0, 1, 2]


def _labels(num_examples: int, *, num_classes: int = MNIST_NUM_CLASSES) -> np.ndarray:
    return (np.arange(num_examples) % num_classes).astype(np.int64)


def _assert_exact_partition(
    partition: dict[int, tuple[int, ...]],
    *,
    num_examples: int,
    per_node: int,
) -> None:
    assert set(partition) == set(NODES)
    union: list[int] = []
    for indices in partition.values():
        assert len(indices) == per_node
        assert len(set(indices)) == len(indices)
        assert all(0 <= index < num_examples for index in indices)
        union.extend(indices)
    assert len(union) == len(set(union)) == len(NODES) * per_node


def _mean_max_fraction(counts: dict[int, tuple[int, ...]], per_node: int) -> float:
    return float(np.mean([max(block) / per_node for block in counts.values()]))


# --------------------------------------------------------------------------- #
# Config contract (opt-in, alpha required/forbidden, legacy bytes)
# --------------------------------------------------------------------------- #
def test_dirichlet_config_parses_with_alpha() -> None:
    config = make_config(
        data={"partition": "dirichlet", "dirichlet_alpha": 0.5}
    )
    assert config.data.partition == "dirichlet"
    assert config.data.dirichlet_alpha == 0.5
    payload = canonical_dump(config)
    assert '"dirichlet_alpha":0.5' in payload
    assert '"partition":"dirichlet"' in payload


def test_dirichlet_config_requires_alpha() -> None:
    with pytest.raises(ConfigError, match="dirichlet_alpha is required"):
        make_config(data={"partition": "dirichlet"})


def test_iid_config_forbids_alpha() -> None:
    with pytest.raises(ConfigError, match="only valid when"):
        make_config(data={"partition": "iid", "dirichlet_alpha": 0.5})


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan"), float("inf")])
def test_dirichlet_config_rejects_nonpositive_or_nonfinite_alpha(alpha: float) -> None:
    with pytest.raises(ConfigError):
        make_config(data={"partition": "dirichlet", "dirichlet_alpha": alpha})


def test_iid_config_bytes_omit_absent_alpha() -> None:
    config = make_config()
    payload = canonical_dump(config)
    assert "dirichlet_alpha" not in payload
    assert '"partition":"iid"' in payload
    # The legacy IID canonical bytes are exactly the pre-extension form.
    assert config.data.dirichlet_alpha is None


def test_dirichlet_config_round_trips_canonical_dump() -> None:
    from dfl_sim import parse_config_text

    original = make_config(
        data={"partition": "dirichlet", "dirichlet_alpha": 1.0}
    )
    reparsed = parse_config_text(canonical_dump(original))
    assert reparsed.data.partition == "dirichlet"
    assert reparsed.data.dirichlet_alpha == 1.0
    assert canonical_dump(reparsed) == canonical_dump(original)


# --------------------------------------------------------------------------- #
# Allocator: exact quotas, disjointness, determinism, ordering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
def test_dirichlet_exact_quotas_and_disjoint(alpha: float) -> None:
    labels = _labels(200)
    partition = partition_indices_dirichlet(
        labels, NODES, 20, seed=7, alpha=alpha
    )
    _assert_exact_partition(partition, num_examples=200, per_node=20)


def test_dirichlet_repeatable_across_calls() -> None:
    labels = _labels(200)
    first = partition_indices_dirichlet(labels, NODES, 20, seed=7, alpha=0.5)
    second = partition_indices_dirichlet(labels, NODES, 20, seed=7, alpha=0.5)
    assert first == second


def test_dirichlet_order_independent() -> None:
    labels = _labels(200)
    forward = partition_indices_dirichlet(labels, NODES, 20, seed=7, alpha=0.5)
    reversed_nodes = partition_indices_dirichlet(
        labels, list(reversed(NODES)), 20, seed=7, alpha=0.5
    )
    assert forward == reversed_nodes


def test_dirichlet_seed_changes_assignment() -> None:
    labels = _labels(200)
    a = partition_indices_dirichlet(labels, NODES, 20, seed=1, alpha=0.5)
    b = partition_indices_dirichlet(labels, NODES, 20, seed=2, alpha=0.5)
    assert a != b


def test_dirichlet_accepts_torch_tensor_labels() -> None:
    labels = torch.tensor(_labels(200), dtype=torch.long)
    partition = partition_indices_dirichlet(labels, NODES, 20, seed=7, alpha=0.5)
    numpy_partition = partition_indices_dirichlet(
        _labels(200), NODES, 20, seed=7, alpha=0.5
    )
    assert partition == numpy_partition


def test_dirichlet_alpha_changes_realized_distribution() -> None:
    labels = _labels(600)
    low = partition_indices_dirichlet(labels, NODES, 50, seed=11, alpha=0.1)
    high = partition_indices_dirichlet(labels, NODES, 50, seed=11, alpha=1.0)
    low_counts = realized_label_counts(labels, low)
    high_counts = realized_label_counts(labels, high)
    assert low_counts != high_counts
    # Stronger heterogeneity concentrates each node on fewer classes.
    assert _mean_max_fraction(low_counts, 50) > _mean_max_fraction(high_counts, 50)


def test_dirichlet_uses_all_ten_classes_when_present() -> None:
    labels = _labels(300)
    partition = partition_indices_dirichlet(labels, NODES, 30, seed=3, alpha=0.5)
    per_node = realized_label_counts(labels, partition)
    for counts in per_node.values():
        assert len(counts) == MNIST_NUM_CLASSES
        assert sum(counts) == 30


# --------------------------------------------------------------------------- #
# Design complement
# --------------------------------------------------------------------------- #
def test_dirichlet_design_disjoint_and_local_stable() -> None:
    labels = _labels(200)
    split = partition_indices_dirichlet_with_design(
        labels, NODES, 20, seed=7, alpha=0.5, design_size=15
    )
    base = partition_indices_dirichlet(labels, NODES, 20, seed=7, alpha=0.5)
    # Node blocks do not depend on the design request, so run and calibration
    # always agree on the training membership.
    assert split.node_indices == base
    assert len(split.design_indices) == 15
    allocated = split.all_indices
    assert len(allocated) == len(set(allocated)) == 75
    design = set(split.design_indices)
    for block in split.node_indices.values():
        assert design.isdisjoint(set(block))


def test_dirichlet_design_repeatable_and_order_independent() -> None:
    labels = _labels(200)
    first = partition_indices_dirichlet_with_design(
        labels, NODES, 20, seed=7, alpha=0.5, design_size=15
    )
    second = partition_indices_dirichlet_with_design(
        labels, list(reversed(NODES)), 20, seed=7, alpha=0.5, design_size=15
    )
    assert first == second


def test_dirichlet_design_seed_changes_selection() -> None:
    labels = _labels(300)
    a = partition_indices_dirichlet_with_design(
        labels, NODES, 20, seed=1, alpha=0.5, design_size=15
    )
    b = partition_indices_dirichlet_with_design(
        labels, NODES, 20, seed=2, alpha=0.5, design_size=15
    )
    assert a.design_indices != b.design_indices


# --------------------------------------------------------------------------- #
# Library dispatch (same mechanism for simulation and calibration)
# --------------------------------------------------------------------------- #
def test_partition_dataset_dirichlet_subsets_match_indices() -> None:
    dataset = MNISTShapedDataset(200, seed=5)
    subsets = partition_dataset(
        dataset, NODES, 20, seed=7, partition="dirichlet", alpha=0.5
    )
    labels = extract_dataset_labels(dataset)
    expected = partition_indices_dirichlet(
        labels, NODES, 20, seed=7, alpha=0.5
    )
    actual = {
        node: tuple(int(index) for index in subset.indices)
        for node, subset in subsets.items()
    }
    assert actual == expected


def test_partition_dataset_with_design_dirichlet() -> None:
    dataset = MNISTShapedDataset(200, seed=5)
    split = partition_dataset_with_design(
        dataset,
        NODES,
        20,
        seed=7,
        design_size=10,
        partition="dirichlet",
        alpha=0.5,
    )
    expected = partition_indices_dirichlet_with_design(
        extract_dataset_labels(dataset),
        NODES,
        20,
        seed=7,
        alpha=0.5,
        design_size=10,
    )
    assert split.node_indices == expected.node_indices
    assert split.design_indices == expected.design_indices
    assert len(split.design_subset) == 10


def test_iid_dispatch_unchanged_by_extension() -> None:
    dataset = MNISTShapedDataset(200, seed=5)
    subsets = partition_dataset(dataset, NODES, 20, seed=7)
    expected = partition_indices(200, NODES, 20, seed=7)
    actual = {
        node: tuple(int(index) for index in subset.indices)
        for node, subset in subsets.items()
    }
    assert actual == expected


def test_dispatch_rejects_alpha_for_iid() -> None:
    dataset = MNISTShapedDataset(200, seed=5)
    with pytest.raises(DataError, match="only valid"):
        partition_dataset(dataset, NODES, 20, seed=7, alpha=0.5)
    with pytest.raises(DataError, match="only valid"):
        partition_dataset_with_design(
            dataset, NODES, 20, seed=7, design_size=10, alpha=0.5
        )


def test_dispatch_rejects_unknown_partition() -> None:
    dataset = MNISTShapedDataset(200, seed=5)
    with pytest.raises(DataError, match="unsupported partition"):
        partition_dataset(dataset, NODES, 20, seed=7, partition="shards")


# --------------------------------------------------------------------------- #
# Reconstructible metadata
# --------------------------------------------------------------------------- #
def test_realized_label_counts_are_reconstructible() -> None:
    labels = _labels(200)
    partition = partition_indices_dirichlet(labels, NODES, 20, seed=7, alpha=0.5)
    counts = realized_label_counts(labels, partition)
    for node, indices in partition.items():
        expected = tuple(
            sum(1 for index in indices if int(labels[index]) == class_index)
            for class_index in range(MNIST_NUM_CLASSES)
        )
        assert counts[node] == expected


def test_partition_membership_digest_is_order_independent() -> None:
    labels = _labels(200)
    split = partition_indices_dirichlet_with_design(
        labels, NODES, 20, seed=7, alpha=0.5, design_size=10
    )
    first = partition_membership_digest(split.node_indices, split.design_indices)
    second = partition_membership_digest(
        dict(reversed(list(split.node_indices.items()))), split.design_indices
    )
    assert first == second
    assert len(first) == 64
    assert all(character in "0123456789abcdef" for character in first)
    # A changed alpha changes the realized membership digest.
    other = partition_indices_dirichlet(labels, NODES, 20, seed=7, alpha=1.0)
    assert partition_membership_digest(other) != partition_membership_digest(
        split.node_indices
    )


# --------------------------------------------------------------------------- #
# Label extraction
# --------------------------------------------------------------------------- #
def test_extract_labels_from_attributes_and_subset() -> None:
    dataset = MNISTShapedDataset(100, seed=1)
    direct = extract_dataset_labels(dataset)
    assert direct.shape == (100,)
    subset = Subset(dataset, [3, 7, 11, 40])
    sub_labels = extract_dataset_labels(subset)
    assert sub_labels.tolist() == [int(direct[i]) for i in (3, 7, 11, 40)]


def test_extract_labels_falls_back_to_iteration() -> None:
    class _NoLabelAttribute(Dataset):
        def __init__(self, base: Dataset) -> None:
            self._base = base

        def __len__(self) -> int:
            return len(self._base)

        def __getitem__(self, index):
            return self._base[index]

    base = MNISTShapedDataset(50, seed=2)
    wrapped = _NoLabelAttribute(base)
    assert extract_dataset_labels(wrapped).tolist() == extract_dataset_labels(
        base
    ).tolist()


def test_extract_labels_rejects_out_of_range() -> None:
    dataset = MNISTShapedDataset(20, seed=1)
    dataset.labels = torch.full((20,), 10, dtype=torch.long)
    with pytest.raises(DataError, match="MNIST labels"):
        extract_dataset_labels(dataset)


# --------------------------------------------------------------------------- #
# Integration: production simulation and calibration share the mechanism
# --------------------------------------------------------------------------- #
def test_run_from_config_accepts_dirichlet_partition() -> None:
    from dfl_sim import run_from_config

    config = make_config(
        rounds=1,
        data={"partition": "dirichlet", "dirichlet_alpha": 0.5, "samples_per_node": 8},
        training={"local_epochs": 1, "batch_size": 8},
        evaluation={"max_samples": 8, "batch_size": 8},
    )
    dataset = MNISTShapedDataset(120, separable=True, seed=3)
    result = run_from_config(config, dataset=dataset)
    assert result.node_ids == (0, 1, 2)
    assert len(result.rounds) == 1


def _dirichlet_calibration_config(**overrides):
    experiment = make_config(
        rounds=1,
        training={"local_epochs": 1, "batch_size": 8},
        data={
            "samples_per_node": 8,
            "partition": "dirichlet",
            "dirichlet_alpha": 0.5,
        },
        evaluation={"max_samples": 8, "batch_size": 8},
    )
    kwargs = {
        "experiment": experiment.model_dump(mode="json"),
        "episodes": 2,
        "p0": 0.5,
        "design_split_size": 8,
    }
    kwargs.update(overrides)
    return parse_dict(base_dict(**kwargs))


def test_collect_episode_uses_dirichlet_partition() -> None:
    config = _dirichlet_calibration_config()
    dataset = MNISTShapedDataset(200, separable=True, seed=11)
    spec = build_schedule(config).specs[0]
    episode = collect_episode(config, spec, dataset=dataset)

    resolved = derive_experiment_config(config, spec)
    seeds = resolve_seeds(resolved)
    expected = partition_indices_dirichlet(
        extract_dataset_labels(dataset),
        (0, 1, 2),
        resolved.data.samples_per_node,
        seed=seeds.partition,
        alpha=0.5,
    )
    actual = dict(episode.node_indices)
    assert actual == expected
    design = set(episode.design_indices)
    assert len(design) == config.design_split_size
    for block in expected.values():
        assert design.isdisjoint(set(block))


def test_collect_calibration_dirichlet_is_deterministic_and_disjoint() -> None:
    config = _dirichlet_calibration_config()
    dataset = MNISTShapedDataset(200, separable=True, seed=11)
    first = collect_calibration(config, dataset=dataset)
    second = collect_calibration(config, dataset=dataset)
    assert first.records == second.records
    assert [e.design_indices for e in first.episodes] == [
        e.design_indices for e in second.episodes
    ]
    for episode in first.episodes:
        design = set(episode.design_indices)
        for _node, indices in episode.node_indices:
            assert design.isdisjoint(set(indices))


# --------------------------------------------------------------------------- #
# Negative surface (Q3/Q7)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan"), float("inf"), None, "x"])
def test_allocator_rejects_invalid_alpha(alpha) -> None:
    labels = _labels(100)
    with pytest.raises(DataError, match="alpha"):
        partition_indices_dirichlet(labels, NODES, 10, seed=1, alpha=alpha)


def test_allocator_rejects_non_integral_labels() -> None:
    labels = np.array([0.0, 1.0, 0.5, 2.0] * 25)
    with pytest.raises(DataError, match="integral"):
        partition_indices_dirichlet(labels, NODES, 10, seed=1, alpha=0.5)


def test_allocator_rejects_out_of_range_labels() -> None:
    labels = _labels(100)
    labels[0] = 10
    with pytest.raises(DataError, match="MNIST labels"):
        partition_indices_dirichlet(labels, NODES, 10, seed=1, alpha=0.5)
    labels[0] = -1
    with pytest.raises(DataError, match="MNIST labels"):
        partition_indices_dirichlet(labels, NODES, 10, seed=1, alpha=0.5)


def test_allocator_rejects_boolean_labels() -> None:
    labels = np.zeros(100, dtype=bool)
    with pytest.raises(DataError, match="boolean"):
        partition_indices_dirichlet(labels, NODES, 10, seed=1, alpha=0.5)


def test_extract_labels_rejects_length_mismatch() -> None:
    class _WrongLabels(Dataset):
        def __init__(self) -> None:
            self.labels = torch.zeros(5, dtype=torch.long)

        def __len__(self) -> int:
            return 10

        def __getitem__(self, index):
            return torch.zeros(1, 1, 1, 1), 0

    with pytest.raises(DataError, match="does not match|length"):
        extract_dataset_labels(_WrongLabels())


def test_allocator_rejects_impossible_quota() -> None:
    with pytest.raises(DataError, match="too few examples"):
        partition_indices_dirichlet(
            _labels(20), NODES, 10, seed=1, alpha=0.5
        )


def test_allocator_rejects_duplicate_and_empty_nodes() -> None:
    with pytest.raises(DataError, match="duplicate node id"):
        partition_indices_dirichlet(
            _labels(100), [0, 1, 1], 10, seed=1, alpha=0.5
        )
    with pytest.raises(DataError, match="at least one node"):
        partition_indices_dirichlet(_labels(100), [], 10, seed=1, alpha=0.5)


def test_design_split_rejects_oversized_and_nonpositive() -> None:
    labels = _labels(100)
    with pytest.raises(DataError, match="too few examples"):
        partition_indices_dirichlet_with_design(
            labels, NODES, 30, seed=1, alpha=0.5, design_size=20
        )
    with pytest.raises(DataError, match="design_size"):
        partition_indices_dirichlet_with_design(
            labels, NODES, 10, seed=1, alpha=0.5, design_size=0
        )


def test_dataset_dispatch_rejects_invalid_alpha_via_dirichlet() -> None:
    dataset = MNISTShapedDataset(100, seed=1)
    with pytest.raises(DataError, match="alpha"):
        partition_dataset(
            dataset, NODES, 10, seed=1, partition="dirichlet", alpha=None
        )
