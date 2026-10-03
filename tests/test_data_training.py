"""Contract tests for T02: seeded partitioning and local training primitives.

All tests are offline and deterministic: datasets are generated in-process and
never downloaded, so the suite runs without network access or a local MNIST
copy. Covers R001's local-training phase, R012's determinism/no-aliasing
evidence, and the Q7 negative surface (impossible partitions, malformed input
shapes, mismatched state dicts, non-finite values, injected failures).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import Subset

from dfl_sim import (
    DataError,
    NonFiniteError,
    StateDictError,
    TrainingError,
    build_model,
    clone_state_dict,
    ensure_finite_state,
    load_mnist_dataset,
    load_state_into_model,
    local_train_from_snapshot,
    partition_dataset,
    partition_indices,
    subset_for_node,
)
from helpers import FailingDataset, MNISTShapedDataset, make_config

NODES = [0, 1, 2]


def snapshot(seed: int = 1234) -> dict[str, torch.Tensor]:
    """Fresh deterministic round-start state built from a seeded model."""

    return clone_state_dict(build_model("small_cnn", seed=seed).state_dict())


def _state_equal(a, b) -> bool:
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[key], b[key]) for key in a)


def _same_values(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Exact tensor equality that treats matching NaNs as equal."""

    return a.shape == b.shape and torch.allclose(
        a.to(torch.float64),
        b.to(torch.float64),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


# --------------------------------------------------------------------------- #
# Partitioning: exact bounds, disjointness, stable ordering
# --------------------------------------------------------------------------- #
def test_partition_exact_counts_and_disjoint() -> None:
    partition = partition_indices(100, NODES, 20, seed=7)
    assert set(partition) == set(NODES)
    assert all(len(indices) == 20 for indices in partition.values())
    union = [index for indices in partition.values() for index in indices]
    assert len(union) == len(set(union)) == 60
    assert all(0 <= index < 100 for index in union)


def test_partition_repeatable_across_calls() -> None:
    first = partition_indices(100, NODES, 20, seed=7)
    second = partition_indices(100, NODES, 20, seed=7)
    assert first == second


def test_partition_is_order_independent() -> None:
    forward = partition_indices(100, NODES, 20, seed=7)
    reversed_nodes = partition_indices(100, list(reversed(NODES)), 20, seed=7)
    assert forward == reversed_nodes


def test_partition_seed_changes_assignment() -> None:
    a = partition_indices(100, NODES, 20, seed=1)
    b = partition_indices(100, NODES, 20, seed=2)
    assert a != b


def test_partition_allows_exact_fit() -> None:
    partition = partition_indices(60, NODES, 20, seed=3)
    assert all(len(indices) == 20 for indices in partition.values())


def test_partition_dataset_returns_subsets() -> None:
    dataset = MNISTShapedDataset(100, seed=5)
    subsets = partition_dataset(dataset, NODES, 20, seed=7)
    assert set(subsets) == set(NODES)
    for subset in subsets.values():
        assert isinstance(subset, Subset)
        assert len(subset) == 20


# --------------------------------------------------------------------------- #
# Partitioning negative surface (Q7)
# --------------------------------------------------------------------------- #
def test_partition_insufficient_examples_raises() -> None:
    with pytest.raises(DataError, match="too few examples"):
        partition_indices(59, NODES, 20, seed=1)


@pytest.mark.parametrize("per_node", [0, -1, -20])
def test_partition_non_positive_samples_rejected(per_node: int) -> None:
    with pytest.raises(DataError, match="samples_per_node"):
        partition_indices(100, NODES, per_node, seed=1)


def test_partition_negative_examples_rejected() -> None:
    with pytest.raises(DataError, match="num_examples"):
        partition_indices(-1, NODES, 1, seed=1)


def test_partition_duplicate_nodes_rejected() -> None:
    with pytest.raises(DataError, match="duplicate node id"):
        partition_indices(100, [0, 1, 1], 10, seed=1)


def test_partition_empty_nodes_rejected() -> None:
    with pytest.raises(DataError, match="at least one node"):
        partition_indices(100, [], 10, seed=1)


def test_subset_for_node_rejects_empty() -> None:
    dataset = MNISTShapedDataset(10, seed=1)
    with pytest.raises(DataError, match="empty index list"):
        subset_for_node(dataset, [])


def test_subset_for_node_rejects_out_of_range() -> None:
    dataset = MNISTShapedDataset(10, seed=1)
    with pytest.raises(DataError, match="out of range"):
        subset_for_node(dataset, [0, 10])


def test_load_mnist_missing_without_download_raises(tmp_path) -> None:
    config = make_config(data={"data_root": str(tmp_path / "absent-mnist")})
    with pytest.raises(DataError, match="could not load MNIST"):
        load_mnist_dataset(config)


# --------------------------------------------------------------------------- #
# Model construction + state dict helpers (R012)
# --------------------------------------------------------------------------- #
def test_build_model_identical_seeded_state() -> None:
    a = build_model("small_cnn", seed=42)
    b = build_model("small_cnn", seed=42)
    assert _state_equal(a.state_dict(), b.state_dict())


def test_build_model_seed_changes_state() -> None:
    a = build_model("small_cnn", seed=42)
    b = build_model("small_cnn", seed=43)
    assert not _state_equal(a.state_dict(), b.state_dict())


def test_build_model_does_not_disturb_global_rng() -> None:
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    build_model("small_cnn", seed=999)
    assert torch.equal(torch.rand(3), expected)


def test_build_model_unknown_name_rejected() -> None:
    with pytest.raises(Exception, match="unsupported model"):
        build_model("resnet")


def test_clone_state_dict_deep_copies_without_aliasing() -> None:
    original = build_model("small_cnn", seed=1).state_dict()
    cloned = clone_state_dict(original)
    for key in original:
        assert cloned[key].data_ptr() != original[key].data_ptr()
        assert torch.equal(cloned[key], original[key])
    # Mutating the clone must not touch the source.
    first = next(iter(cloned))
    cloned[first].add_(1.0)
    assert not torch.equal(cloned[first], original[first])


def test_clone_state_dict_rejects_empty_and_non_tensor() -> None:
    with pytest.raises(StateDictError, match="empty"):
        clone_state_dict({})
    with pytest.raises(StateDictError, match="must be a tensor"):
        clone_state_dict({"w": [1, 2, 3]})


def test_ensure_finite_state_detects_nan() -> None:
    state = snapshot()
    key = next(iter(state))
    state[key][0, 0] = float("nan")
    with pytest.raises(StateDictError, match="non-finite"):
        ensure_finite_state(state)


def test_load_state_into_model_rejects_missing_key() -> None:
    model = build_model("small_cnn", seed=1)
    state = snapshot()
    state.pop(next(iter(state)))
    with pytest.raises(StateDictError):
        load_state_into_model(model, state)


def test_load_state_into_model_rejects_shape_mismatch() -> None:
    model = build_model("small_cnn", seed=1)
    state = snapshot()
    key = next(iter(state))
    state[key] = torch.zeros((1, 1, 1, 1))
    with pytest.raises(StateDictError):
        load_state_into_model(model, state)


def test_load_state_into_model_rejects_non_finite() -> None:
    model = build_model("small_cnn", seed=1)
    state = snapshot()
    state[next(iter(state))].fill_(float("inf"))
    with pytest.raises(StateDictError, match="non-finite"):
        load_state_into_model(model, state)


# --------------------------------------------------------------------------- #
# Local training: repeatability + loss reduction
# --------------------------------------------------------------------------- #
def test_local_train_reduces_loss_on_separable_fixture() -> None:
    config = make_config(training={"local_epochs": 20, "learning_rate": 0.1})
    dataset = MNISTShapedDataset(40, seed=11, separable=True)
    result = local_train_from_snapshot(
        snapshot(),
        dataset,
        config=config,
        node_id=0,
        round_index=0,
        train_seed=101,
        data_order_seed=202,
    )
    assert result.pre_loss > 0.0
    assert result.post_loss < result.pre_loss
    assert math.isfinite(result.pre_loss) and math.isfinite(result.post_loss)
    assert result.sample_count == len(dataset)


def test_local_train_repeatable_with_same_seeds() -> None:
    config = make_config(training={"local_epochs": 4})
    dataset = MNISTShapedDataset(32, seed=9, separable=True)
    kwargs = dict(
        config=config, node_id=1, round_index=0, train_seed=5, data_order_seed=6
    )
    first = local_train_from_snapshot(snapshot(), dataset, **kwargs)
    second = local_train_from_snapshot(snapshot(), dataset, **kwargs)
    assert first.pre_loss == second.pre_loss
    assert first.post_loss == second.post_loss
    assert _state_equal(first.state, second.state)


def test_local_train_different_data_order_diverges() -> None:
    config = make_config(training={"local_epochs": 4})
    dataset = MNISTShapedDataset(32, seed=9, separable=True)
    base = dict(config=config, node_id=0, round_index=0, train_seed=5)
    a = local_train_from_snapshot(snapshot(), dataset, data_order_seed=6, **base)
    b = local_train_from_snapshot(snapshot(), dataset, data_order_seed=77, **base)
    assert not _state_equal(a.state, b.state)


def test_local_train_result_is_frozen() -> None:
    config = make_config(training={"local_epochs": 1})
    dataset = MNISTShapedDataset(16, seed=1, separable=True)
    result = local_train_from_snapshot(
        snapshot(),
        dataset,
        config=config,
        node_id=0,
        round_index=0,
        train_seed=1,
        data_order_seed=2,
    )
    with pytest.raises(Exception):
        result.pre_loss = 0.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Snapshot purity / no aliasing (R012 + Q7)
# --------------------------------------------------------------------------- #
def test_local_train_preserves_snapshot_and_does_not_alias() -> None:
    config = make_config(training={"local_epochs": 3})
    dataset = MNISTShapedDataset(32, seed=4, separable=True)
    source = snapshot()
    baseline = {key: value.clone() for key, value in source.items()}

    result = local_train_from_snapshot(
        source,
        dataset,
        config=config,
        node_id=2,
        round_index=1,
        train_seed=8,
        data_order_seed=13,
    )

    for key in source:
        assert torch.equal(source[key], baseline[key])
        assert result.state[key].data_ptr() != source[key].data_ptr()

    # Mutating the returned state must not affect the caller's snapshot.
    first = next(iter(result.state))
    result.state[first].add_(10.0)
    assert torch.equal(source[first], baseline[first])


def test_local_train_rejects_non_finite_snapshot_and_preserves_it() -> None:
    config = make_config(training={"local_epochs": 1})
    dataset = MNISTShapedDataset(16, seed=2, separable=True)
    source = snapshot()
    source[next(iter(source))].fill_(float("nan"))
    baseline = {key: value.clone() for key, value in source.items()}

    with pytest.raises(TrainingError):
        local_train_from_snapshot(
            source,
            dataset,
            config=config,
            node_id=0,
            round_index=0,
            train_seed=1,
            data_order_seed=2,
        )
    for key in source:
        assert _same_values(source[key], baseline[key])


def test_local_train_rejects_nan_input_and_preserves_snapshot() -> None:
    config = make_config(training={"local_epochs": 2})
    dataset = MNISTShapedDataset(16, seed=2, fill=float("nan"))
    source = snapshot()
    baseline = {key: value.clone() for key, value in source.items()}

    with pytest.raises(NonFiniteError, match="non-finite input"):
        local_train_from_snapshot(
            source,
            dataset,
            config=config,
            node_id=0,
            round_index=0,
            train_seed=1,
            data_order_seed=2,
        )
    for key in source:
        assert torch.equal(source[key], baseline[key])


def test_local_train_rejects_malformed_image_shape() -> None:
    config = make_config(training={"local_epochs": 1})
    dataset = MNISTShapedDataset(16, seed=2, channels=3)
    with pytest.raises(TrainingError, match="forward failed"):
        local_train_from_snapshot(
            snapshot(),
            dataset,
            config=config,
            node_id=0,
            round_index=0,
            train_seed=1,
            data_order_seed=2,
        )


def test_local_train_rejects_mismatched_state() -> None:
    config = make_config(training={"local_epochs": 1})
    dataset = MNISTShapedDataset(16, seed=2, separable=True)
    bad = snapshot()
    bad.pop(next(iter(bad)))
    with pytest.raises(TrainingError):
        local_train_from_snapshot(
            bad,
            dataset,
            config=config,
            node_id=0,
            round_index=0,
            train_seed=1,
            data_order_seed=2,
        )


def test_local_train_rejects_empty_partition() -> None:
    config = make_config(training={"local_epochs": 1})
    base = MNISTShapedDataset(16, seed=2)
    empty = Subset(base, [])
    with pytest.raises(TrainingError, match="empty local partition"):
        local_train_from_snapshot(
            snapshot(),
            empty,
            config=config,
            node_id=0,
            round_index=0,
            train_seed=1,
            data_order_seed=2,
        )


def test_local_train_injected_failure_preserves_snapshot() -> None:
    config = make_config(training={"local_epochs": 3, "batch_size": 8})
    base = MNISTShapedDataset(16, seed=2, separable=True)
    # The pre-loss pass consumes all 16 items; training then fails on access 17.
    failing = FailingDataset(base, fail_after=len(base))
    source = snapshot()
    baseline = {key: value.clone() for key, value in source.items()}

    with pytest.raises(TrainingError, match="local training failed"):
        local_train_from_snapshot(
            source,
            failing,
            config=config,
            node_id=1,
            round_index=2,
            train_seed=3,
            data_order_seed=4,
        )
    for key in source:
        assert torch.equal(source[key], baseline[key])
