"""Strict, frozen experiment configuration contracts for the DFL simulator.

Configuration is treated as untrusted structured input (Q3 threat surface):

* YAML is parsed with :func:`yaml.safe_load` only -- never ``load``/``full_load``.
* Every model forbids unknown keys (``extra="forbid"``) and is frozen.
* All numeric and structural values are bounded.
* No config value ever selects a Python class or executes a tag.

Public entry points: :func:`load_config`, :func:`parse_config_text`,
:func:`canonical_dump`, and the :class:`ExperimentConfig` model tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

__all__ = [
    "ATTACK_KAPPAS",
    "MAX_BATCH_SIZE",
    "MAX_EVAL_BATCH_SIZE",
    "MAX_EVAL_SAMPLES",
    "MAX_LOCAL_EPOCHS",
    "MAX_NUM_WORKERS",
    "MAX_ROUNDS",
    "MAX_SAMPLES_PER_NODE",
    "MAX_SEED",
    "MAX_SNAPSHOT_CADENCE",
    "MAX_WS_NODES",
    "MIN_WS_NODES",
    "AggregationConfig",
    "AttackConfig",
    "Chain3Topology",
    "ConfigError",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FixedSimilarityAggregation",
    "Line2Topology",
    "LocalOnlyAggregation",
    "ModelConfig",
    "PlainAverageAggregation",
    "TopologyConfig",
    "TraceConfig",
    "TrainingConfig",
    "WattsStrogatzTopology",
    "canonical_dump",
    "load_config",
    "parse_config_text",
    "topology_node_count",
]

# --------------------------------------------------------------------------- #
# Bounds (Q6 load profile: values outside these are rejected, not degraded).
# --------------------------------------------------------------------------- #
MAX_SEED = 2**31 - 1
MAX_ROUNDS = 1000
MAX_SAMPLES_PER_NODE = 10_000
MAX_BATCH_SIZE = 1024
MAX_LOCAL_EPOCHS = 50
MAX_NUM_WORKERS = 8
MIN_WS_NODES = 10
MAX_WS_NODES = 20

#: Bounds for the privileged post-commit evaluator (R008/D018, Q6).
#: ``max_samples`` caps the shared reference subset evaluated per node per
#: recorded round, and ``batch_size`` caps the evaluation batch. Both are
#: deliberately separate from the training bounds so evaluation cost can be
#: bounded independently of local training.
MAX_EVAL_SAMPLES = 10_000
MAX_EVAL_BATCH_SIZE = 1024

#: Bound for the trace snapshot cadence: the number of rounds between
#: cadence-selected tensor snapshots. Always bounded by the round count.
MAX_SNAPSHOT_CADENCE = MAX_ROUNDS

#: D016 fixed intensity table: the only legal kappa values, keyed by name.
ATTACK_KAPPAS: dict[str, float] = {"zero": 0.0, "weak": 2.0, "strong": 5.0}


class ConfigError(ValueError):
    """Raised when configuration YAML is missing, malformed, or invalid."""


class _StrictModel(BaseModel):
    """Base for every config leaf: unknown keys forbidden, frozen, strict types."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelConfig(_StrictModel):
    """Model selector. M001 ships exactly one deterministic small CNN."""

    name: Literal["small_cnn"] = "small_cnn"


class DataConfig(_StrictModel):
    """Dataset and partition bounds. M001 is MNIST + IID only (D006)."""

    dataset: Literal["mnist"] = "mnist"
    partition: Literal["iid"] = "iid"
    samples_per_node: int = Field(ge=1, le=MAX_SAMPLES_PER_NODE)
    data_root: str = Field(default="data", min_length=1, max_length=256)
    download: bool = False

    @field_validator("data_root")
    @classmethod
    def _reject_control_chars(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("data_root must not contain NUL bytes")
        return value


class TrainingConfig(_StrictModel):
    """Local training hyper-parameters; all values are bounded (Q6)."""

    optimizer: Literal["sgd", "adam"] = "sgd"
    learning_rate: float = Field(gt=0.0, le=1.0)
    batch_size: int = Field(ge=1, le=MAX_BATCH_SIZE)
    local_epochs: int = Field(ge=1, le=MAX_LOCAL_EPOCHS)
    num_workers: int = Field(default=0, ge=0, le=MAX_NUM_WORKERS)


class EvaluationConfig(_StrictModel):
    """Bounded privileged-evaluation controls (R008/Q6).

    ``max_samples`` caps the deterministic, seed-selected reference subset
    that the post-commit evaluator scores every committed node model against,
    and ``batch_size`` caps the evaluation batch. Both are strict-typed,
    bounded integers with backward-compatible defaults, so pre-S04 configs
    keep loading unchanged and the canonical dump always carries resolved
    values.
    """

    max_samples: int = Field(default=512, ge=1, le=MAX_EVAL_SAMPLES)
    batch_size: int = Field(default=64, ge=1, le=MAX_EVAL_BATCH_SIZE)


class TraceConfig(_StrictModel):
    """Trace publication controls (D018/R009).

    ``snapshot_cadence`` is the number of rounds between cadence-selected
    tensor snapshots in the versioned trajectory store; initial and final
    committed states are always included regardless of cadence. It is bounded
    by :data:`MAX_ROUNDS` and defaulted for backward compatibility.
    """

    snapshot_cadence: int = Field(default=1, ge=1, le=MAX_SNAPSHOT_CADENCE)


class Line2Topology(_StrictModel):
    """Exact 2-node line: 0 -- 1. Node count is fixed and cannot be overridden."""

    kind: Literal["line2"] = "line2"


class Chain3Topology(_StrictModel):
    """Exact 3-node chain: 0 -- 1 -- 2. Node count is fixed, no overrides."""

    kind: Literal["chain3"] = "chain3"


class WattsStrogatzTopology(_StrictModel):
    """D014 seeded connected Watts-Strogatz family constrained to 10-20 nodes.

    ``k`` is the even ring neighborhood degree and ``2 <= k < n``; ``p`` is the
    rewiring probability in ``[0, 1]``.
    """

    kind: Literal["watts_strogatz"]
    n: int = Field(ge=MIN_WS_NODES, le=MAX_WS_NODES)
    k: int = Field(ge=2)
    p: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_degree(self) -> "WattsStrogatzTopology":
        if self.k % 2 != 0:
            raise ValueError(f"watts_strogatz requires an even neighborhood degree k, got {self.k}")
        if self.k >= self.n:
            raise ValueError(
                f"watts_strogatz requires k < n, got k={self.k} for n={self.n}"
            )
        return self


# Discriminated union on the ``kind`` tag; unknown kinds fail fast.
TopologyConfig = Annotated[
    Union[Line2Topology, Chain3Topology, WattsStrogatzTopology],
    Field(discriminator="kind"),
]


class AttackConfig(_StrictModel):
    """D016 strict fixed per-episode delta-cancellation/reversal attacker.

    A single optional **scalar** compromised node (never a list) plus one named
    finite intensity from :data:`ATTACK_KAPPAS`. Omitting the node requires
    ``intensity == "zero"``; a selected node may still carry ``"zero"`` so an
    episode can prove the no-op path through the real attack seam.
    """

    compromised_node: int | None = None
    intensity: Literal["zero", "weak", "strong"] = "zero"

    @model_validator(mode="after")
    def _zero_required_without_selected_node(self) -> "AttackConfig":
        if self.compromised_node is None and self.intensity != "zero":
            raise ValueError(
                "attack.intensity must be 'zero' when attack.compromised_node "
                "is not selected"
            )
        if self.compromised_node is not None and self.compromised_node < 0:
            raise ValueError(
                "attack.compromised_node must be a non-negative node id, got "
                f"{self.compromised_node}"
            )
        return self

    @property
    def kappa(self) -> float:
        """The D016 numeric reversal intensity for this episode's attack."""

        return ATTACK_KAPPAS[self.intensity]


class LocalOnlyAggregation(_StrictModel):
    """D008 baseline selector: aggregate only the node's own trained model."""

    kind: Literal["local_only"] = "local_only"


class PlainAverageAggregation(_StrictModel):
    """D008/D015 baseline selector: uniform mean of self plus every sender.

    This is the programmatic default so existing configs that predate the
    aggregation union keep byte-for-byte D015 behavior.
    """

    kind: Literal["plain_average"] = "plain_average"


class FixedSimilarityAggregation(_StrictModel):
    """D008/D017 fixed directional-similarity filter selector.

    ``cutoff`` is the maximum directional disagreement (in the feature's
    ``[0, 2]`` range) at which a delivered sender is accepted. The configured
    ``downweight_multiplier`` is the D017 reference ``omega`` in ``(0, 1)`` for
    the finite neighbor multiplier set ``{0, omega, 1}``; the M001 fixed filter
    is binary (reject to ``0``) and validates its emitted multipliers against
    that set so the finite-action contract cannot silently drift.
    """

    kind: Literal["fixed_similarity"]
    cutoff: float = Field(ge=0.0, le=2.0)
    downweight_multiplier: float = Field(gt=0.0, lt=1.0)


# Discriminated union on the ``kind`` tag; unknown aggregation kinds fail fast.
AggregationConfig = Annotated[
    Union[
        LocalOnlyAggregation,
        PlainAverageAggregation,
        FixedSimilarityAggregation,
    ],
    Field(discriminator="kind"),
]


_TOPOLOGY_NODE_COUNTS = {"line2": 2, "chain3": 3}


def topology_node_count(topology: TopologyConfig) -> int:
    """Return the fixed node count implied by a resolved topology config."""

    if isinstance(topology, WattsStrogatzTopology):
        return int(topology.n)
    return _TOPOLOGY_NODE_COUNTS[topology.kind]


class ExperimentConfig(_StrictModel):
    """Top-level validated experiment envelope."""

    seed: int = Field(ge=0, le=MAX_SEED)
    rounds: int = Field(ge=1, le=MAX_ROUNDS)
    device: Literal["cpu"] = "cpu"
    bitwise: bool = True
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig
    training: TrainingConfig
    topology: TopologyConfig
    attack: AttackConfig = Field(default_factory=AttackConfig)
    aggregation: AggregationConfig = Field(
        default_factory=PlainAverageAggregation
    )
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)

    @model_validator(mode="after")
    def _attack_node_within_topology(self) -> "ExperimentConfig":
        node = self.attack.compromised_node
        if node is None:
            return self
        count = topology_node_count(self.topology)
        if not 0 <= node < count:
            raise ValueError(
                f"attack.compromised_node {node} is outside the "
                f"{self.topology.kind} node range [0, {count})"
            )
        return self

    @model_validator(mode="after")
    def _bitwise_requires_serial_loading(self) -> "ExperimentConfig":
        # Bitwise reproducibility requires a single-process, single-thread,
        # zero-worker data path (D009).
        if self.bitwise and self.training.num_workers != 0:
            raise ValueError(
                "bitwise mode requires training.num_workers == 0, "
                f"got {self.training.num_workers}"
            )
        return self


def parse_config_text(text: str, *, source: str = "<string>") -> ExperimentConfig:
    """Parse and validate YAML ``text`` into an :class:`ExperimentConfig`.

    Raises :class:`ConfigError` with the source path and offending fields on any
    malformed YAML, non-mapping root, or validation failure.
    """

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"empty config in {source}: expected a mapping")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config root in {source} must be a mapping, got {type(raw).__name__}"
        )

    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {source}: {exc}") from exc


def load_config(path: str | Path) -> ExperimentConfig:
    """Read ``path`` from disk and validate it as an :class:`ExperimentConfig`."""

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {config_path}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read config file {config_path}: {exc}") from exc
    return parse_config_text(text, source=str(config_path))


def canonical_dump(config: ExperimentConfig) -> str:
    """Return a canonical, deterministic JSON rendering of ``config``.

    Keys are sorted, separators are compact, and the encoding is ASCII so two
    independent loads of the same config produce byte-identical strings.
    """

    return json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
