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
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset, Subset

from .config import ExperimentConfig

__all__ = [
    "DataError",
    "DatasetSplit",
    "TrainingSplit",
    "load_mnist_dataset",
    "load_mnist_evaluation_dataset",
    "partition_dataset",
    "partition_dataset_with_design",
    "partition_indices",
    "partition_indices_with_design",
    "subset_for_node",
]


class DataError(RuntimeError):
    """Raised when data bounds are impossible or the dataset cannot be loaded."""


@dataclass(frozen=True, slots=True)
class TrainingSplit:
    """Disjoint node-local index blocks plus a clean global design split.

    ``node_indices`` maps each sorted node id to exactly ``samples_per_node``
    distinct MNIST-training indices. ``design_indices`` is the bounded global
    clean design split drawn from the indices left over after every node-local
    block was allocated. All blocks are pairwise disjoint by construction and
    by the defensive disjointness check in
    :func:`partition_indices_with_design`.
    """

    node_indices: dict[int, tuple[int, ...]]
    design_indices: tuple[int, ...]

    @property
    def all_indices(self) -> tuple[int, ...]:
        """Every allocated index in deterministic node-then-design order."""

        ordered: list[int] = []
        for node in sorted(self.node_indices):
            ordered.extend(self.node_indices[node])
        ordered.extend(self.design_indices)
        return tuple(ordered)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Node-local :class:`Subset` blocks plus the clean design :class:`Subset`."""

    node_subsets: dict[int, Subset]
    design_subset: Subset
    design_indices: tuple[int, ...]


def _partition_bounds(
    num_examples: int,
    node_ids: Sequence[int],
    samples_per_node: int,
) -> tuple[int, list[int], int]:
    """Validate and normalize partition arguments (shared by both variants)."""

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
    return n_examples, ordered, per_node


def _seeded_blocks(
    n_examples: int,
    ordered: list[int],
    per_node: int,
    *,
    seed: int,
    design_size: int,
) -> tuple[np.ndarray, int]:
    """Draw one seeded permutation and split it into local then design blocks."""

    local_total = len(ordered) * per_node
    total = local_total + design_size
    if total > n_examples:
        raise DataError(
            "too few examples for the requested partition"
            + (" plus design split" if design_size else "")
            + f": need {total} ({len(ordered)} nodes x {per_node}"
            + (f" + design {design_size}" if design_size else "")
            + f"), have {n_examples}"
        )
    rng = np.random.default_rng(int(seed))
    selected = rng.permutation(n_examples)[:total]
    return selected, local_total


def _assert_disjoint(
    blocks: Sequence[Sequence[int]], *, context: str
) -> None:
    """Defensive check: no index may appear in two allocation blocks."""

    seen: set[int] = set()
    for block in blocks:
        for index in block:
            value = int(index)
            if value in seen:
                raise DataError(
                    f"allocation overlap at index {value} while building {context}"
                )
            seen.add(value)


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

    n_examples, ordered, per_node = _partition_bounds(
        num_examples, node_ids, samples_per_node
    )
    # Permute once with a stable-seed RNG; slice contiguous blocks in sorted
    # node order so the assignment is reproducible and ordering-independent.
    selected, _ = _seeded_blocks(
        n_examples, ordered, per_node, seed=seed, design_size=0
    )

    partition: dict[int, tuple[int, ...]] = {}
    for position, node in enumerate(ordered):
        start = position * per_node
        block = selected[start : start + per_node]
        partition[node] = tuple(int(index) for index in block)
    return partition


def partition_indices_with_design(
    num_examples: int,
    node_ids: Sequence[int],
    samples_per_node: int,
    *,
    seed: int,
    design_size: int,
) -> TrainingSplit:
    """Allocate node-local blocks then a disjoint clean global design split.

    A **single** seeded permutation of the MNIST-training indices is consumed
    front-to-back: the first ``len(node_ids) * samples_per_node`` indices become
    the node-local blocks (in sorted node order, exactly as
    :func:`partition_indices`) and the next ``design_size`` unused indices become
    the clean global design split. All three blocks are pairwise disjoint and
    deterministic; the returned ``design_indices`` preserve the permutation
    order so repeat calls are byte-stable. The final-test loader is never
    consulted.
    """

    n_examples, ordered, per_node = _partition_bounds(
        num_examples, node_ids, samples_per_node
    )
    try:
        requested_design = int(design_size)
    except (TypeError, ValueError) as exc:
        raise DataError(f"design_size must be an integer: {exc}") from exc
    if requested_design < 1:
        raise DataError(
            f"design_size must be >= 1, got {requested_design}"
        )

    selected, local_total = _seeded_blocks(
        n_examples,
        ordered,
        per_node,
        seed=seed,
        design_size=requested_design,
    )

    node_indices: dict[int, tuple[int, ...]] = {}
    for position, node in enumerate(ordered):
        start = position * per_node
        node_indices[node] = tuple(
            int(index) for index in selected[start : start + per_node]
        )
    design_indices = tuple(
        int(index) for index in selected[local_total : local_total + requested_design]
    )

    # Defensive: a single permutation cannot overlap, but assert it so a future
    # refactor cannot silently introduce aliasing across the local/design split.
    _assert_disjoint(
        [*node_indices.values(), design_indices], context="training split"
    )
    return TrainingSplit(
        node_indices=node_indices, design_indices=design_indices
    )


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


def partition_dataset_with_design(
    dataset: Dataset,
    node_ids: Sequence[int],
    samples_per_node: int,
    *,
    seed: int,
    design_size: int,
) -> DatasetSplit:
    """Dataset-level variant of :func:`partition_indices_with_design`.

    Returns the node-local :class:`Subset` blocks and the clean global design
    :class:`Subset` carved from the same single seeded permutation, so the
    local partitions and the design split are guaranteed disjoint and never
    touch the final-test loader.
    """

    split = partition_indices_with_design(
        len(dataset),
        node_ids,
        samples_per_node,
        seed=seed,
        design_size=design_size,
    )
    node_subsets = {
        node: subset_for_node(dataset, indices)
        for node, indices in split.node_indices.items()
    }
    design_subset = subset_for_node(dataset, split.design_indices)
    return DatasetSplit(
        node_subsets=node_subsets,
        design_subset=design_subset,
        design_indices=split.design_indices,
    )


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
