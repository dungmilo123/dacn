"""Per-node local training from an immutable round-start snapshot (R001).

:func:`local_train_from_snapshot` is the pure local phase of the synchronous
DFL loop. It:

1. builds a fresh model and loads a **clone** of the caller's round-start
   state (the input snapshot is never mutated or aliased),
2. resets the configured optimizer,
3. trains for the configured bounded epochs with a per-node/per-round
   data-order seed, and
4. evaluates cross-entropy on the identical local partition *before* and
   *after* training, rejecting non-finite losses/parameters.

It returns a detached cloned trained state plus stable scalar statistics and
does not write files or communicate with other nodes -- the round engine (T03)
owns message exchange and simultaneous commit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .models import (
    StateDictError,
    build_model,
    clone_state_dict,
    ensure_finite_state,
    load_state_into_model,
)

__all__ = [
    "LocalTrainResult",
    "NonFiniteError",
    "TrainingError",
    "evaluate_cross_entropy",
    "local_train_from_snapshot",
]

# Torch generator seeds are signed 64-bit; stable seeds are < 2**31.
_GENERATOR_MODULUS = 2**63


class TrainingError(RuntimeError):
    """Raised when local training cannot proceed (data/shape/state problems)."""


class NonFiniteError(TrainingError):
    """Raised when inputs, losses, or trained parameters become non-finite."""


@dataclass(frozen=True, slots=True)
class LocalTrainResult:
    """Immutable outcome of one node's local training for one round.

    ``state`` holds detached cloned tensors; it shares no storage with the
    input snapshot or the training model.
    """

    node_id: int
    round_index: int
    state: dict[str, torch.Tensor]
    pre_loss: float
    post_loss: float
    sample_count: int


def _build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    optimizer_name = config.training.optimizer
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=config.training.learning_rate
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=config.training.learning_rate
        )
    # Schema discriminates the optimizer, so this is defensive only.
    raise TrainingError(f"unsupported optimizer: {optimizer_name!r}")


def _forward(model: nn.Module, inputs: torch.Tensor, *, context: str) -> torch.Tensor:
    try:
        return model(inputs)
    except RuntimeError as exc:
        raise TrainingError(
            f"model forward failed ({context}): expected {model.__class__.__name__} "
            f"input of shape (N, 1, 28, 28), got {tuple(inputs.shape)}: {exc}"
        ) from exc


def _ensure_finite_inputs(inputs: torch.Tensor, *, context: str) -> None:
    if not bool(torch.isfinite(inputs).all()):
        raise NonFiniteError(f"non-finite input values ({context})")


def evaluate_cross_entropy(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    context: str,
) -> float:
    """Mean cross-entropy over ``dataset`` in eval mode (no grad, no shuffle)."""

    if len(dataset) == 0:
        raise TrainingError(f"empty local partition cannot be evaluated ({context})")

    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    model.eval()
    total = 0.0
    count = 0
    try:
        with torch.no_grad():
            for inputs, targets in loader:
                _ensure_finite_inputs(inputs, context=context)
                logits = _forward(model, inputs, context=context)
                loss = criterion(logits, targets)
                if not bool(torch.isfinite(loss)):
                    raise NonFiniteError(f"non-finite evaluation loss ({context})")
                batch = int(inputs.shape[0])
                total += float(loss) * batch
                count += batch
    finally:
        model.train()

    if count == 0:  # pragma: no cover - empty partition already rejected
        raise TrainingError(f"no examples evaluated ({context})")
    return total / count


def local_train_from_snapshot(
    snapshot: Mapping[str, torch.Tensor],
    dataset: Dataset,
    *,
    config: ExperimentConfig,
    node_id: int,
    round_index: int,
    train_seed: int,
    data_order_seed: int,
) -> LocalTrainResult:
    """Train one node locally from ``snapshot`` on its ``dataset`` partition.

    ``snapshot`` is treated as immutable input: it is validated, deep-cloned,
    and loaded into a freshly built model. On every success and error path the
    caller's snapshot tensors remain untouched and un-aliased.
    """

    context = f"node={int(node_id)} round={int(round_index)}"

    if not snapshot:
        raise TrainingError(f"empty round-start snapshot ({context})")
    if len(dataset) == 0:
        raise TrainingError(f"empty local partition ({context})")

    # Build + load only a cloned snapshot. build_model uses a private generator
    # seeded from train_seed, so initialization is deterministic and does not
    # disturb global RNG state.
    try:
        model = build_model(config.model.name, seed=int(train_seed))
        load_state_into_model(model, snapshot, context=context)
    except StateDictError as exc:
        raise TrainingError(f"invalid round-start snapshot ({context}): {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive model construction
        raise TrainingError(f"could not build model ({context}): {exc}") from exc

    pre_loss = evaluate_cross_entropy(
        model, dataset, batch_size=config.training.batch_size, context=context
    )

    optimizer = _build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(int(data_order_seed) % _GENERATOR_MODULUS)
    torch.manual_seed(int(train_seed) % _GENERATOR_MODULUS)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    try:
        model.train()
        for _epoch in range(config.training.local_epochs):
            for inputs, targets in loader:
                _ensure_finite_inputs(inputs, context=context)
                optimizer.zero_grad(set_to_none=True)
                logits = _forward(model, inputs, context=context)
                loss = criterion(logits, targets)
                if not bool(torch.isfinite(loss)):
                    raise NonFiniteError(f"non-finite training loss ({context})")
                loss.backward()
                optimizer.step()
    except (TrainingError, NonFiniteError, StateDictError):
        raise
    except Exception as exc:
        raise TrainingError(f"local training failed ({context}): {exc}") from exc

    post_loss = evaluate_cross_entropy(
        model, dataset, batch_size=config.training.batch_size, context=context
    )

    if not math.isfinite(pre_loss) or not math.isfinite(post_loss):
        raise NonFiniteError(
            f"non-finite local loss ({context}): pre={pre_loss} post={post_loss}"
        )

    trained_state = clone_state_dict(model.state_dict())
    try:
        ensure_finite_state(trained_state, context=f"trained state {context}")
    except StateDictError as exc:
        raise NonFiniteError(str(exc)) from exc

    return LocalTrainResult(
        node_id=int(node_id),
        round_index=int(round_index),
        state=trained_state,
        pre_loss=float(pre_loss),
        post_loss=float(post_loss),
        sample_count=int(len(dataset)),
    )
