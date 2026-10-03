"""Seeded IID MNIST partitioning and dataset loading (D006, R001).

The partition is a pure function of ``(num_examples, node_ids,
samples_per_node, seed)``: it permutes the dataset once, takes exactly
``len(node_ids) * samples_per_node`` indices, and hands each node a contiguous
block in **sorted node order**. Reversing the caller's node order therefore
yields the identical mapping, and every node receives exactly its configured
sample bound with no overlap.

Production data loading uses :class:`torchvision.datasets.MNIST` with explicit
``root``/``download`` settings; torchvision verifies the archive integrity
(checksum) on download. Tests inject an in-process :class:`~torch.utils.data.Dataset`
through :func:`partition_dataset` / :func:`subset_for_node`, so the suite never
touches the network. No arbitrary model or data files are deserialized here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from torch.utils.data import Dataset, Subset

from .config import ExperimentConfig

__all__ = [
    "DataError",
    "load_mnist_dataset",
    "load_mnist_evaluation_dataset",
    "partition_dataset",
    "partition_indices",
    "subset_for_node",
]


class DataError(RuntimeError):
    """Raised when data bounds are impossible or the dataset cannot be loaded."""


def partition_indices(
    num_examples: int,
    node_ids: Sequence[int],
    samples_per_node: int,
    *,
    seed: int,
) -> dict[int, tuple[int, ...]]:
    """Return a seeded, disjoint IID index partition keyed by node id.

    Each node receives exactly ``samples_per_node`` distinct indices. Node IDs
    are normalized to sorted order before assignment, so the mapping is
    independent of the caller's ordering. Raises :class:`DataError` when the
    bounds are impossible (empty node set, non-positive samples, duplicates, or
    fewer examples than ``len(node_ids) * samples_per_node``).
    """

    try:
        n_examples = int(num_examples)
        per_node = int(samples_per_node)
    except (TypeError, ValueError) as exc:
        raise DataError(f"partition bounds must be integers: {exc}") from exc

    if n_examples < 0:
        raise DataError(f"num_examples must be non-negative, got {n_examples}")
    if per_node < 1:
        raise DataError(f"samples_per_node must be >= 1, got {per_node}")

    ordered: list[int] = []
    seen: set[int] = set()
    for raw in node_ids:
        try:
            node = int(raw)
        except (TypeError, ValueError) as exc:
            raise DataError(f"node id must be an integer, got {raw!r}") from exc
        if node in seen:
            raise DataError(f"duplicate node id in partition request: {node}")
        seen.add(node)
        ordered.append(node)
    if not ordered:
        raise DataError("partition requires at least one node id")
    ordered.sort()

    total = len(ordered) * per_node
    if total > n_examples:
        raise DataError(
            "too few examples for the requested partition: "
            f"need {total} ({len(ordered)} nodes x {per_node}), have {n_examples}"
        )

    # Permute once with a stable-seed RNG; slice contiguous blocks in sorted
    # node order so the assignment is reproducible and ordering-independent.
    rng = np.random.default_rng(int(seed))
    selected = rng.permutation(n_examples)[:total]

    partition: dict[int, tuple[int, ...]] = {}
    for position, node in enumerate(ordered):
        start = position * per_node
        block = selected[start : start + per_node]
        partition[node] = tuple(int(index) for index in block)
    return partition


def subset_for_node(dataset: Dataset, indices: Sequence[int]) -> Subset:
    """Wrap ``dataset`` with a node's index list, rejecting empty partitions."""

    index_list = [int(index) for index in indices]
    if not index_list:
        raise DataError("cannot build a node subset from an empty index list")
    length = len(dataset)
    for index in index_list:
        if index < 0 or index >= length:
            raise DataError(
                f"partition index {index} out of range for dataset of size {length}"
            )
    return Subset(dataset, index_list)


def partition_dataset(
    dataset: Dataset,
    node_ids: Sequence[int],
    samples_per_node: int,
    *,
    seed: int,
) -> dict[int, Subset]:
    """Partition ``dataset`` into one :class:`Subset` per node (stable order)."""

    indices = partition_indices(
        len(dataset), node_ids, samples_per_node, seed=seed
    )
    return {node: subset_for_node(dataset, idx) for node, idx in indices.items()}


def load_mnist_dataset(
    config: ExperimentConfig, *, train: bool = True, transform=None
) -> Dataset:
    """Load the production MNIST split described by ``config``.

    ``train=True`` (the default) loads the training split; ``train=False``
    loads the held-out evaluation split used by the privileged post-commit
    evaluator. Imports ``torchvision`` lazily so partition-only tests avoid the
    cost. The ``download`` flag comes from validated config; when downloads are
    disabled and files are missing, the underlying :class:`RuntimeError` is
    wrapped in :class:`DataError` naming the root so the failure is actionable.
    """

    if config.data.dataset != "mnist":  # pragma: no cover - schema enforced
        raise DataError(f"unsupported dataset: {config.data.dataset!r}")

    from torchvision import datasets, transforms

    if transform is None:
        transform = transforms.ToTensor()

    try:
        return datasets.MNIST(
            root=config.data.data_root,
            train=bool(train),
            download=config.data.download,
            transform=transform,
        )
    except (RuntimeError, OSError) as exc:
        split = "train" if train else "evaluation"
        raise DataError(
            f"could not load MNIST ({split} split) from "
            f"{config.data.data_root!r} (download={config.data.download}): {exc}"
        ) from exc


def load_mnist_evaluation_dataset(config: ExperimentConfig, *, transform=None) -> Dataset:
    """Load the MNIST ``train=False`` evaluation split with the training split's
    typed :class:`DataError` failure behavior.
    """

    return load_mnist_dataset(config, train=False, transform=transform)
