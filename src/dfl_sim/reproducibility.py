"""Deterministic setup and stable seed resolution (D009).

Seeds are derived with a stable BLAKE2b mix over ``<master>:<domain-label>``
rather than Python's per-process randomized :func:`hash`, so the same config
yields identical seeds across processes, machines, and interpreter runs.

:func:`configure_determinism` centralizes Python/NumPy/Torch seeding, one CPU
thread, and (in bitwise mode) Torch deterministic algorithms. It returns a
:class:`DeterminismReport` describing what was applied.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from .config import ExperimentConfig

__all__ = [
    "DeterminismError",
    "DeterminismReport",
    "ResolvedSeeds",
    "SEED_DOMAINS",
    "configure_determinism",
    "resolve_seeds",
    "stable_seed",
]

# Stable domain labels. Adding a label must never renumber existing ones.
SEED_DOMAINS: tuple[str, ...] = (
    "python",
    "numpy",
    "torch",
    "topology",
    "partition",
)

_SEED_MODULUS = 2**31


class DeterminismError(RuntimeError):
    """Raised when the requested deterministic mode cannot be established."""


def stable_seed(master: int, label: str) -> int:
    """Derive a stable seed in ``[0, 2**31)`` from ``master`` and ``label``.

    Uses BLAKE2b over a UTF-8 ``"<master>:<label>"`` domain string. The result
    is intentionally independent of Python's randomized ``hash``.
    """

    digest = hashlib.blake2b(
        f"{int(master)}:{label}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS


@dataclass(frozen=True, slots=True)
class ResolvedSeeds:
    """Frozen set of domain-separated seeds derived from one experiment seed."""

    master: int
    python: int
    numpy: int
    torch: int
    topology: int
    partition: int

    def derive(self, label: str) -> int:
        """Derive a new stable seed for an arbitrary domain ``label``."""

        return stable_seed(self.master, label)

    def for_local_train(self, round_index: int, node_id: int) -> int:
        """Per-node, per-round local-training seed (stable across processes)."""

        return self.derive(f"local_train:{int(round_index)}:{int(node_id)}")

    def for_data_order(self, round_index: int, node_id: int) -> int:
        """Per-node, per-round data-ordering seed used by local loaders."""

        return self.derive(f"data_order:{int(round_index)}:{int(node_id)}")


def resolve_seeds(config: ExperimentConfig) -> ResolvedSeeds:
    """Resolve the frozen seed set for ``config`` using stable arithmetic."""

    master = int(config.seed)
    return ResolvedSeeds(
        master=master,
        python=stable_seed(master, "python"),
        numpy=stable_seed(master, "numpy"),
        torch=stable_seed(master, "torch"),
        topology=stable_seed(master, "topology"),
        partition=stable_seed(master, "partition"),
    )


@dataclass(frozen=True, slots=True)
class DeterminismReport:
    """Outcome of :func:`configure_determinism`, for diagnostics/tests."""

    bitwise: bool
    master_seed: int
    num_threads: int
    num_workers: int
    deterministic_algorithms: bool


def configure_determinism(
    seeds: ResolvedSeeds,
    *,
    bitwise: bool = True,
    num_workers: int = 0,
) -> DeterminismReport:
    """Apply process-wide deterministic settings and return a report.

    Sets ``PYTHONHASHSEED`` (for any child process), Python/NumPy/Torch seeds,
    one CPU thread, and -- in bitwise mode -- Torch deterministic algorithms
    plus deterministic cuDNN. Raises :class:`DeterminismError` if bitwise
    reproducibility is requested but cannot be established.
    """

    if bitwise and num_workers != 0:
        raise DeterminismError(
            "bitwise mode requires num_workers == 0, "
            f"got {num_workers}"
        )

    os.environ["PYTHONHASHSEED"] = str(seeds.python)
    random.seed(seeds.python)
    np.random.seed(seeds.numpy % (2**32))
    torch.manual_seed(seeds.torch)
    torch.set_num_threads(1)

    if torch.cuda.is_available():  # pragma: no cover - CPU-first environment
        torch.cuda.manual_seed_all(seeds.torch)

    deterministic_algorithms = False
    if bitwise:
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except Exception as exc:  # pragma: no cover - depends on torch build
            raise DeterminismError(
                f"failed to enable deterministic torch algorithms: {exc}"
            ) from exc
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        deterministic_algorithms = True

    return DeterminismReport(
        bitwise=bitwise,
        master_seed=seeds.master,
        num_threads=torch.get_num_threads(),
        num_workers=num_workers,
        deterministic_algorithms=deterministic_algorithms,
    )
