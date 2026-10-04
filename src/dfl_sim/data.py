"""Seeded MNIST partitioning (IID and opt-in Dirichlet) and dataset loading.

The IID partition (D006, R001) is a pure function of ``(num_examples, node_ids,
samples_per_node, seed)``: it permutes the dataset once, takes exactly
``len(node_ids) * samples_per_node`` indices, and hands each node a contiguous
block in **sorted node order**. Reversing the caller's node order therefore
yields the identical mapping, and every node receives exactly its configured
sample bound with no overlap.

The opt-in Dirichlet partition (D039) adds a pure, bounded, capacity-constrained
class-wise allocation: each class's node affinity is a deterministic
``Dirichlet(alpha)`` draw and a fixed-point repair forces every node to exactly
``samples_per_node`` unique examples, with the clean design split carved from the
unallocated complement so it is always disjoint from the node blocks. Both
modes are wired through :func:`partition_dataset` /
:func:`partition_dataset_with_design` so simulation and calibration share one
mechanism.

Production data loading uses :class:`torchvision.datasets.MNIST` with explicit
``root``/``download`` settings; torchvision verifies the archive integrity
(checksum) on download. Tests inject an in-process :class:`~torch.utils.data.Dataset`
through :func:`partition_dataset` / :func:`subset_for_node`, so the suite never
touches the network. No arbitrary model or data files are deserialized here.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from .config import ExperimentConfig
from .reproducibility import stable_seed

__all__ = [
    "DataError",
    "DatasetSplit",
    "MNIST_NUM_CLASSES",
    "TrainingSplit",
    "extract_dataset_labels",
    "load_mnist_dataset",
    "load_mnist_evaluation_dataset",
    "partition_dataset",
    "partition_dataset_with_design",
    "partition_indices",
    "partition_indices_dirichlet",
    "partition_indices_dirichlet_with_design",
    "partition_indices_with_design",
    "partition_membership_digest",
    "realized_label_counts",
    "subset_for_node",
]

#: MNIST has ten digit classes; labels must be integral in ``[0, 10)``.
MNIST_NUM_CLASSES = 10

#: Domain-separated labels for the deterministic Dirichlet draws. The class
#: affinity and the concrete index slotting use separate domains so the two
#: draws can never coincide, and the clean design split uses a third.
_DIRICHLET_CLASS_DOMAIN = "dirichlet-class"
_DIRICHLET_SLOT_DOMAIN = "dirichlet-slot"
_DIRICHLET_DESIGN_DOMAIN = "dirichlet-design"


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

    @property
    def node_indices(self) -> dict[int, tuple[int, ...]]:
        """The concrete index membership behind each node :class:`Subset`.

        Derived (not stored) so a dataset split and an index split can never
        disagree; ``Subset.indices`` is the single source of truth.
        """

        return {
            node: tuple(int(index) for index in subset.indices)
            for node, subset in self.node_subsets.items()
        }


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


def _coerce_label_array(labels) -> np.ndarray:
    """Return a 1-D int64 label view, rejecting non-integral/boolean input."""

    if isinstance(labels, torch.Tensor):
        array = labels.detach().to("cpu").contiguous().numpy()
    else:
        array = np.asarray(labels)
    if array.ndim != 1:
        raise DataError("labels must be a one-dimensional sequence")
    if array.dtype.kind == "b":
        raise DataError("labels must be integers, got booleans")
    if array.dtype.kind == "f":
        if not bool(np.isfinite(array).all()):
            raise DataError("labels must be finite")
        if not bool(np.equal(array, np.floor(array)).all()):
            raise DataError("labels must be integral")
        array = array.astype(np.int64)
    elif array.dtype.kind in "iu":
        array = array.astype(np.int64)
    else:
        raise DataError(f"labels must be integers, got dtype {array.dtype}")
    return array


def _validate_mnist_labels(array: np.ndarray, num_examples: int) -> np.ndarray:
    """Validate length and MNIST class range, returning the same array."""

    if array.shape[0] != num_examples:
        raise DataError(
            f"labels length {array.shape[0]} does not match dataset length "
            f"{num_examples}"
        )
    if array.size and (
        int(array.min()) < 0 or int(array.max()) >= MNIST_NUM_CLASSES
    ):
        raise DataError(
            f"MNIST labels must be integers in [0, {MNIST_NUM_CLASSES}), got "
            f"range [{int(array.min())}, {int(array.max())}]"
        )
    return array


def _validated_alpha(alpha) -> float:
    """Return a finite, strictly positive Dirichlet concentration or raise."""

    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
        raise DataError(
            f"dirichlet alpha must be a finite positive number, got {alpha!r}"
        )
    value = float(alpha)
    if not math.isfinite(value) or value <= 0.0:
        raise DataError(
            f"dirichlet alpha must be finite and positive, got {value!r}"
        )
    return value


def extract_dataset_labels(dataset: Dataset) -> np.ndarray:
    """Return a validated 1-D int64 MNIST label array for ``dataset``.

    Recognizes torchvision's ``.targets`` and in-memory ``.labels`` attributes
    (including through :class:`Subset` wrappers) without reading image tensors;
    otherwise falls back to reading each ``(image, label)`` pair once. The
    result is validated as integral MNIST labels in ``[0, 10)`` before use, so
    a malformed label source fails closed instead of silently partitioning.
    """

    n_examples = len(dataset)
    if isinstance(dataset, Subset):
        base = extract_dataset_labels(dataset.dataset)
        indices = np.asarray([int(i) for i in dataset.indices], dtype=np.int64)
        if indices.size and (
            int(indices.min()) < 0 or int(indices.max()) >= base.shape[0]
        ):
            raise DataError("Subset indices out of range for label extraction")
        return _validate_mnist_labels(base[indices], indices.shape[0])

    for attr in ("targets", "labels"):
        value = getattr(dataset, attr, None)
        if value is None:
            continue
        array = _coerce_label_array(value)
        if array.shape[0] != n_examples:
            raise DataError(
                f"dataset label attribute {attr!r} has length "
                f"{array.shape[0]}, expected {n_examples}"
            )
        return _validate_mnist_labels(array, n_examples)

    collected = np.empty(n_examples, dtype=np.int64)
    for index in range(n_examples):
        try:
            sample = dataset[index]
        except Exception as exc:  # pragma: no cover - defensive path
            raise DataError(
                f"could not read dataset label at index {index}: {exc}"
            ) from exc
        label = sample[-1] if isinstance(sample, tuple) else sample
        if isinstance(label, torch.Tensor):
            if label.numel() != 1:
                raise DataError(
                    f"dataset label at index {index} must be scalar"
                )
            label = label.item()
        if isinstance(label, bool):
            raise DataError(f"dataset label at index {index} must be an integer")
        if isinstance(label, numbers.Integral):
            collected[index] = int(label)
        elif (
            isinstance(label, numbers.Real)
            and math.isfinite(float(label))
            and float(label).is_integer()
        ):
            collected[index] = int(label)
        else:
            raise DataError(
                f"dataset label at index {index} must be an integer, got "
                f"{label!r}"
            )
    return _validate_mnist_labels(collected, n_examples)


def _class_usage(supply: np.ndarray, total_quota: int) -> np.ndarray:
    """Choose how many examples of each class the local quota consumes.

    The total quota is split across classes proportionally to their supply via
    the largest-remainder method, so the realized class mix tracks the pool
    exactly and every class usage stays within its supply. Deterministic ties
    break toward the smaller class index.
    """

    pool_size = int(supply.sum())
    if total_quota > pool_size:
        raise DataError(
            "too few examples for the requested Dirichlet partition: need "
            f"{total_quota}, have {pool_size}"
        )
    usage = np.zeros_like(supply)
    if total_quota == 0:
        return usage
    exact = supply.astype(np.float64) * (float(total_quota) / float(pool_size))
    usage = np.floor(exact).astype(np.int64)
    remainder = int(total_quota - int(usage.sum()))
    if remainder > 0:
        candidates = np.where(supply > 0)[0]
        fractional = exact - usage
        order = candidates[np.lexsort((candidates, -fractional[candidates]))]
        for index in order[:remainder]:
            usage[index] += 1
    if int(usage.sum()) != total_quota:
        raise DataError("internal Dirichlet class usage did not reach the quota")
    if bool((usage > supply).any()):
        raise DataError("internal Dirichlet class usage exceeded supply")
    return usage


def _node_class_fractions(
    num_nodes: int, alpha: float, seed: int
) -> np.ndarray:
    """Draw one ``Dirichlet(alpha)`` node-affinity vector per class.

    Each class draws independently from a domain-separated stable seed, so the
    matrix is deterministic and reconstructible from ``(seed, alpha)`` alone;
    smaller alpha concentrates a class on fewer nodes, larger alpha approaches
    a uniform split.
    """

    fractions = np.empty((MNIST_NUM_CLASSES, num_nodes), dtype=np.float64)
    for class_index in range(MNIST_NUM_CLASSES):
        rng = np.random.default_rng(
            stable_seed(int(seed), f"{_DIRICHLET_CLASS_DOMAIN}:{class_index}")
        )
        fractions[class_index] = rng.dirichlet(
            np.full(num_nodes, alpha, dtype=np.float64)
        )
    return fractions


def _initial_counts(
    fractions: np.ndarray, usage: np.ndarray, num_nodes: int
) -> np.ndarray:
    """Round each class's node fractions to integer counts summing to usage."""

    counts = np.zeros((MNIST_NUM_CLASSES, num_nodes), dtype=np.int64)
    for class_index in range(MNIST_NUM_CLASSES):
        class_usage = int(usage[class_index])
        if class_usage == 0:
            continue
        exact = fractions[class_index] * float(class_usage)
        base = np.floor(exact).astype(np.int64)
        remainder = class_usage - int(base.sum())
        if remainder > 0:
            fractional = exact - base
            order = np.lexsort((np.arange(num_nodes), -fractional))
            base[order[:remainder]] += 1
        if int(base.sum()) != class_usage:
            raise DataError("internal Dirichlet class slotting failed")
        counts[class_index] = base
    return counts


def _balance_counts(counts: np.ndarray, quota: int) -> np.ndarray:
    """Move same-class units between nodes until every node has exactly quota.

    The per-class rounding above yields exact class usage but only approximate
    per-node totals. Moving a unit of a class from an over-full to an
    under-full node preserves that class's usage and the global total while
    driving every node column to the exact quota; no retry or IID fallback is
    involved and the tie rules are deterministic.
    """

    counts = counts.copy()
    num_nodes = counts.shape[1]
    column = counts.sum(axis=0)
    over = [k for k in range(num_nodes) if int(column[k]) > quota]
    under = [k for k in range(num_nodes) if int(column[k]) < quota]
    i = 0
    j = 0
    while i < len(over) and j < len(under):
        source = over[i]
        target = under[j]
        class_index = int(np.argmax(counts[:, source]))
        if int(counts[class_index, source]) <= 0:  # pragma: no cover - invariant
            raise DataError("internal Dirichlet repair found no movable class")
        counts[class_index, source] -= 1
        counts[class_index, target] += 1
        column[source] -= 1
        column[target] += 1
        if int(column[source]) == quota:
            i += 1
        if int(column[target]) == quota:
            j += 1
    if i != len(over) or j != len(under):
        raise DataError("internal Dirichlet repair failed to reach node quotas")
    if bool((counts.sum(axis=0) != quota).any()):  # pragma: no cover - invariant
        raise DataError("internal Dirichlet repair left a node off-quota")
    return counts


def _dirichlet_index_partition(
    labels: np.ndarray,
    ordered: list[int],
    per_node: int,
    seed: int,
    alpha: float,
) -> dict[int, tuple[int, ...]]:
    """Pure class-wise Dirichlet allocation over the full example pool.

    The concrete index membership is independent of any design split (design
    is carved from the leftover complement afterwards), so the runner and the
    calibration design split always agree on the node blocks. Within each
    class, a domain-separated seeded permutation slots the class's indices into
    contiguous per-node blocks, so every index appears exactly once.
    """

    num_nodes = len(ordered)
    quota = per_node
    pool = np.arange(labels.shape[0], dtype=np.int64)
    pool_labels = labels[pool]
    supply = np.asarray(
        [int(np.count_nonzero(pool_labels == c)) for c in range(MNIST_NUM_CLASSES)],
        dtype=np.int64,
    )
    usage = _class_usage(supply, num_nodes * quota)
    fractions = _node_class_fractions(num_nodes, alpha, seed)
    counts = _balance_counts(_initial_counts(fractions, usage, num_nodes), quota)

    node_pools: dict[int, list[int]] = {node: [] for node in ordered}
    for class_index in range(MNIST_NUM_CLASSES):
        class_indices = pool[pool_labels == class_index]
        rng = np.random.default_rng(
            stable_seed(int(seed), f"{_DIRICHLET_SLOT_DOMAIN}:{class_index}")
        )
        permuted = class_indices[rng.permutation(class_indices.shape[0])]
        offset = 0
        for position, node in enumerate(ordered):
            take = int(counts[class_index, position])
            if take:
                node_pools[node].extend(int(v) for v in permuted[offset : offset + take])
            offset += take

    return {node: tuple(sorted(node_pools[node])) for node in ordered}


def _design_complement(
    node_indices: dict[int, tuple[int, ...]],
    num_examples: int,
    seed: int,
    design_size: int,
) -> tuple[int, ...]:
    """Reserve a clean design split from the unallocated complement.

    Only indices not already in a node block are eligible, so the design split
    is disjoint from every node-local block by construction. A domain-separated
    seeded permutation selects exactly ``design_size`` of them; the result is
    sorted so repeat calls are byte-stable.
    """

    allocated: set[int] = set()
    for indices in node_indices.values():
        allocated.update(int(index) for index in indices)
    leftover = np.asarray(
        sorted(set(range(num_examples)) - allocated), dtype=np.int64
    )
    if design_size > leftover.shape[0]:
        total_local = sum(len(indices) for indices in node_indices.values())
        raise DataError(
            "too few examples for the requested partition plus design split: "
            f"need {total_local} local + {design_size} design, have "
            f"{num_examples}"
        )
    rng = np.random.default_rng(stable_seed(int(seed), _DIRICHLET_DESIGN_DOMAIN))
    chosen = leftover[rng.permutation(leftover.shape[0])[:design_size]]
    return tuple(sorted(int(index) for index in chosen))


def partition_indices_dirichlet(
    labels,
    node_ids: Sequence[int],
    samples_per_node: int,
    *,
    seed: int,
    alpha: float,
) -> dict[int, tuple[int, ...]]:
    """Return a capacity-constrained class-wise Dirichlet index partition.

    ``labels`` is a 1-D integral MNIST label source (array-like, NumPy, or
    torch tensor). The rules are:

    * Node IDs are sorted, so the mapping is independent of the caller order.
    * The total ``len(node_ids) * samples_per_node`` quota is split across the
      ten classes proportionally to their supply (largest remainder, ties to
      the smaller class index).
    * For each class, a ``Dirichlet(alpha, ..., alpha)`` vector over the sorted
      nodes gives the node affinity; a largest-remainder rounding turns it into
      integer counts and a deterministic repair moves same-class units between
      nodes until **every** node has exactly ``samples_per_node`` unique
      examples. There is no retry loop and no IID fallback: impossible bounds
      raise :class:`DataError`.
    * Within a class, a domain-separated seeded permutation slots the chosen
      indices into contiguous per-node blocks, so realized membership is
      reconstructible from ``(labels, node_ids, samples_per_node, seed, alpha)``.

    Smaller alpha concentrates each class on fewer nodes; larger alpha spreads
    it. This alpha is unrelated to the offline calibration smoothing alpha.
    """

    array = _coerce_label_array(labels)
    n_examples, ordered, per_node = _partition_bounds(
        array.shape[0], node_ids, samples_per_node
    )
    alpha_value = _validated_alpha(alpha)
    _validate_mnist_labels(array, n_examples)
    if n_examples < len(ordered) * per_node:
        raise DataError(
            "too few examples for the requested Dirichlet partition: need "
            f"{len(ordered) * per_node}, have {n_examples}"
        )
    return _dirichlet_index_partition(
        array, ordered, per_node, int(seed), alpha_value
    )


def partition_indices_dirichlet_with_design(
    labels,
    node_ids: Sequence[int],
    samples_per_node: int,
    *,
    seed: int,
    alpha: float,
    design_size: int,
) -> TrainingSplit:
    """Dirichlet node blocks plus a disjoint clean global design split.

    Node blocks are identical to :func:`partition_indices_dirichlet` (design is
    carved from the complement, never from the local allocation), so the design
    split is guaranteed disjoint from every local block and never touches the
    final-test loader. ``design_size`` must be a positive integer that fits in
    the unallocated complement.
    """

    array = _coerce_label_array(labels)
    n_examples, ordered, per_node = _partition_bounds(
        array.shape[0], node_ids, samples_per_node
    )
    alpha_value = _validated_alpha(alpha)
    _validate_mnist_labels(array, n_examples)
    try:
        requested_design = int(design_size)
    except (TypeError, ValueError) as exc:
        raise DataError(f"design_size must be an integer: {exc}") from exc
    if requested_design < 1:
        raise DataError(f"design_size must be >= 1, got {requested_design}")
    if n_examples < len(ordered) * per_node:
        raise DataError(
            "too few examples for the requested Dirichlet partition: need "
            f"{len(ordered) * per_node}, have {n_examples}"
        )
    node_indices = _dirichlet_index_partition(
        array, ordered, per_node, int(seed), alpha_value
    )
    design_indices = _design_complement(
        node_indices, n_examples, int(seed), requested_design
    )
    _assert_disjoint(
        [*node_indices.values(), design_indices],
        context="dirichlet training split",
    )
    return TrainingSplit(
        node_indices=node_indices, design_indices=design_indices
    )


def realized_label_counts(
    labels, node_indices: dict[int, Sequence[int]]
) -> dict[int, tuple[int, ...]]:
    """Return each node's realized per-class counts from a membership map.

    Reconstructible report metadata: the counts follow purely from the label
    source and the assigned indices, so an auditor can recompute them without
    trusting a stored copy.
    """

    array = _coerce_label_array(labels)
    counts: dict[int, tuple[int, ...]] = {}
    for node in sorted(node_indices):
        block = np.asarray(
            [int(index) for index in node_indices[node]], dtype=np.int64
        )
        counts[node] = tuple(
            int(np.count_nonzero(array[block] == c))
            for c in range(MNIST_NUM_CLASSES)
        )
    return counts


def partition_membership_digest(
    node_indices: dict[int, Sequence[int]],
    design_indices: Sequence[int] = (),
) -> str:
    """Return a canonical SHA-256 digest of a partition/design membership.

    Node keys are sorted and the payload is canonical compact JSON, so the
    digest is independent of dict ordering and reconstructible by any auditor
    from the published membership alone.
    """

    payload = {
        "nodes": {
            str(node): [int(index) for index in node_indices[node]]
            for node in sorted(node_indices)
        },
        "design": [int(index) for index in design_indices],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    partition: str = "iid",
    alpha: float | None = None,
) -> dict[int, Subset]:
    """Partition ``dataset`` into one :class:`Subset` per node (stable order).

    ``partition="iid"`` (the default) is the legacy single-permutation
    allocation; ``partition="dirichlet"`` extracts the dataset labels and uses
    the capacity-constrained class-wise allocator with the required
    ``alpha``. ``alpha`` is rejected for IID so the modes can never be mixed.
    """

    if partition == "dirichlet":
        indices = partition_indices_dirichlet(
            extract_dataset_labels(dataset),
            node_ids,
            samples_per_node,
            seed=seed,
            alpha=alpha,
        )
    elif partition == "iid":
        if alpha is not None:
            raise DataError("alpha is only valid for the dirichlet partition")
        indices = partition_indices(
            len(dataset), node_ids, samples_per_node, seed=seed
        )
    else:
        raise DataError(f"unsupported partition mode: {partition!r}")
    return {node: subset_for_node(dataset, idx) for node, idx in indices.items()}


def partition_dataset_with_design(
    dataset: Dataset,
    node_ids: Sequence[int],
    samples_per_node: int,
    *,
    seed: int,
    design_size: int,
    partition: str = "iid",
    alpha: float | None = None,
) -> DatasetSplit:
    """Dataset-level variant of the index split with a disjoint design split.

    Returns the node-local :class:`Subset` blocks and the clean global design
    :class:`Subset` carved from the unallocated complement, so the local
    partitions and the design split are guaranteed disjoint and never touch the
    final-test loader. Honours the same ``partition``/``alpha`` dispatch as
    :func:`partition_dataset`, so calibration and simulation agree on the node
    blocks for both IID and Dirichlet.
    """

    if partition == "dirichlet":
        split = partition_indices_dirichlet_with_design(
            extract_dataset_labels(dataset),
            node_ids,
            samples_per_node,
            seed=seed,
            alpha=alpha,
            design_size=design_size,
        )
    elif partition == "iid":
        if alpha is not None:
            raise DataError("alpha is only valid for the dirichlet partition")
        split = partition_indices_with_design(
            len(dataset),
            node_ids,
            samples_per_node,
            seed=seed,
            design_size=design_size,
        )
    else:
        raise DataError(f"unsupported partition mode: {partition!r}")

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
