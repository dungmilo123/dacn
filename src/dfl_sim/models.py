"""Compact deterministic CNN and tensor state-dict helpers (D006, R001/R012).

M001 ships exactly one model, :class:`SmallCNN`, a small fully-convolutional
network for ``1x28x28`` grayscale MNIST inputs. Construction can be seeded with
a local :class:`torch.Generator` so two builds with the same seed produce
bitwise-identical parameters without touching global RNG state.

State dicts are treated as boundary payloads (Q3): helpers clone them deeply
(cross-node/round isolation) and verify keys, shapes, dtypes, and finiteness
before any arithmetic. Nothing here reads a model file from disk -- weights
are always produced in-process.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

__all__ = [
    "MNIST_CHANNELS",
    "MNIST_SIZE",
    "NUM_CLASSES",
    "ModelError",
    "SmallCNN",
    "StateDictError",
    "build_model",
    "clone_state_dict",
    "ensure_finite_state",
    "load_state_into_model",
    "state_dict_signature",
]

MNIST_CHANNELS = 1
MNIST_SIZE = 28
NUM_CLASSES = 10

# Torch generators accept a signed 64-bit seed; stable seeds are < 2**31.
_GENERATOR_MODULUS = 2**63


class ModelError(RuntimeError):
    """Raised for unsupported model selectors or construction failures."""


class StateDictError(ModelError):
    """Raised when a state dict is malformed, mismatched, or non-finite."""


class SmallCNN(nn.Module):
    """Compact deterministic CNN for ``1x28x28`` MNIST input.

    Two max-pooled convolution blocks followed by a two-layer classifier. The
    architecture is fixed so state dicts from the same config are always
    load-compatible; small enough to fit many cloned instances in CPU memory.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(MNIST_CHANNELS, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.classifier(self.features(inputs))


_MODEL_REGISTRY: dict[str, type[nn.Module]] = {"small_cnn": SmallCNN}


def _initialize_parameters(model: nn.Module, generator: torch.Generator) -> None:
    """Reset Conv/Linear parameters from ``generator`` (no global RNG use)."""

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            fan_in = int(module.weight[0].numel())
            bound = 1.0 / (fan_in**0.5)
            with torch.no_grad():
                module.weight.uniform_(-bound, bound, generator=generator)
                if module.bias is not None:
                    module.bias.uniform_(-bound, bound, generator=generator)


def build_model(
    model_name: str = "small_cnn",
    *,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> nn.Module:
    """Construct a model, optionally with a seeded local parameter init.

    When ``seed`` is provided (and no generator), a private generator is used
    so initialization is repeatable and does not perturb global RNG state.
    """

    model_cls = _MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ModelError(
            f"unsupported model {model_name!r}; expected one of "
            f"{sorted(_MODEL_REGISTRY)}"
        )

    if seed is not None and generator is None:
        generator = torch.Generator().manual_seed(int(seed) % _GENERATOR_MODULUS)

    if generator is None:
        return model_cls()

    # Construct and re-initialize under a saved global CPU RNG state: model
    # construction consumes global RNG for default init before we overwrite it
    # from the private generator, so restore the state to leave no side effect.
    saved_state = torch.get_rng_state()
    try:
        model = model_cls()
        _initialize_parameters(model, generator)
    finally:
        torch.set_rng_state(saved_state)
    return model


def clone_state_dict(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Deep-clone a tensor state dict (detached, contiguous-free copies).

    Raises :class:`StateDictError` on non-tensor values or an empty mapping so
    a malformed payload fails at the boundary instead of during arithmetic.
    """

    if not state:
        raise StateDictError("state dict is empty")
    cloned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise StateDictError(f"state dict key must be str, got {type(key).__name__}")
        if not isinstance(value, torch.Tensor):
            raise StateDictError(
                f"state dict entry {key!r} must be a tensor, got {type(value).__name__}"
            )
        cloned[key] = value.detach().clone()
    return cloned


def ensure_finite_state(
    state: Mapping[str, torch.Tensor], *, context: str = ""
) -> None:
    """Raise :class:`StateDictError` if any tensor contains NaN/Inf."""

    suffix = f" ({context})" if context else ""
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise StateDictError(f"state entry {key!r} is not a tensor{suffix}")
        if not bool(torch.isfinite(value).all()):
            raise StateDictError(f"non-finite values in state entry {key!r}{suffix}")


def state_dict_signature(
    state: Mapping[str, torch.Tensor],
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """Return a canonical ``(key, shape, dtype)`` signature for comparison."""

    return tuple(
        (key, tuple(state[key].shape), str(state[key].dtype)) for key in sorted(state)
    )


def load_state_into_model(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    context: str = "",
) -> None:
    """Load ``state`` into ``model`` strictly, with boundary validation.

    The state is fully validated (keys/shapes/finiteness via ``strict=True``
    load plus a finiteness scan) *before* the model is mutated, so a bad
    payload cannot leave a partially-loaded model behind.
    """

    suffix = f" ({context})" if context else ""
    ensure_finite_state(state, context=f"state load{suffix}")
    try:
        model.load_state_dict(dict(state), strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise StateDictError(f"incompatible state dict{suffix}: {exc}") from exc
