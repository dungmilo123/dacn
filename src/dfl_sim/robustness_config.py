"""Pre-calibration canonical robustness protocol envelope (S05/T02, D040).

This module freezes the bounded **source protocol** for the S05 canonical
robustness matrix *before* any artifact digest exists:

* three Dirichlet class-heterogeneity alphas (0.1, 0.5, 1.0) crossed with three
  seed-derived Watts-Strogatz degree placements (low/mid/high), exactly nine
  conditions;
* real-MNIST identity with an explicit verified headline source root and a
  separate calibration seed;
* a pinned CPU SmallCNN base (six rounds, 64 samples per node, 128 evaluation
  samples, singleton candidate budget, one thread) and the four known D033
  schedules plus the reserved held-out final-only schedule.

Every condition is a fully validated :class:`HeldOutStudyConfig` whose base is
a ``partition="dirichlet"`` experiment on an ``n=10, k=2, p=0.3`` Watts-Strogatz
topology and whose :class:`~dfl_sim.evaluation_config.PlacementContract`
resolves the compromised node by realized degree rank per seed.

Injected (non-real-MNIST) protocols are available only through the explicitly
named :func:`build_injected_protocol` internal test API; the public
:func:`build_canonical_protocol` always builds the real-MNIST protocol and
rejects injected provenance. The envelope carries no artifact digests yet: T04
generates and pins the matched full/no_belief/no_strategic calibration triple
and rebuilds the eight-policy conditions on top of this source contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from .attack_schedule import build_schedule
from .config import (
    MAX_ROUNDS,
    MAX_SAMPLES_PER_NODE,
    MAX_SEED,
    AttackConfig,
    ConfigError,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PlainAverageAggregation,
    TraceConfig,
    TrainingConfig,
    WattsStrogatzTopology,
)
from .evaluation_config import (
    BALANCE_CITATION,
    MAX_CANDIDATE_BUDGET,
    MAX_KNOWN_SCHEDULES,
    MAX_STUDY_SEEDS,
    BaselineStudyConfig,
    CitationMetadata,
    HeldOutStudyConfig,
    NamedSchedule,
    PanelPolicy,
    PlacementContract,
    TuningProvenance,
)

__all__ = [
    "CANONICAL_ALPHAS",
    "CANONICAL_CALIBRATION_SEED",
    "CANONICAL_FINAL_SEEDS",
    "CANONICAL_PLACEMENTS",
    "CANONICAL_PROTOCOL_ID",
    "CANONICAL_SELECTION_SEEDS",
    "CANONICAL_TOPOLOGY",
    "RobustnessCondition",
    "RobustnessConfigError",
    "CanonicalRobustnessProtocol",
    "build_canonical_protocol",
    "build_injected_protocol",
]

#: Canonical Dirichlet class-heterogeneity alphas (D039/D040).
CANONICAL_ALPHAS: tuple[float, ...] = (0.1, 0.5, 1.0)

#: Canonical realized-graph degree placement ranks.
CANONICAL_PLACEMENTS: tuple[str, ...] = ("low", "mid", "high")

#: Pinned real-MNIST seed split (disjoint from the calibration seed).
CANONICAL_SELECTION_SEEDS: tuple[int, ...] = (1101, 1102)
CANONICAL_FINAL_SEEDS: tuple[int, ...] = (2101, 2102, 2103)
CANONICAL_CALIBRATION_SEED = 20240101

#: Pinned Watts-Strogatz family (n, k, p).
CANONICAL_TOPOLOGY = {"n": 10, "k": 2, "p": 0.3}

CANONICAL_PROTOCOL_ID = "m003-canonical-robustness"

#: Pinned base experiment resources.
_CANONICAL_ROUNDS = 6
_CANONICAL_SAMPLES_PER_NODE = 64
_CANONICAL_EVAL_SAMPLES = 128
_CANONICAL_EVAL_BATCH = 32
_CANONICAL_TRAIN_BATCH = 32
_CANONICAL_LEARNING_RATE = 0.05
_CANONICAL_CANDIDATE_BUDGET = 1
_CANONICAL_SELECTION_SEED = 1101
_CANONICAL_SELECTION_PROVENANCE = "singleton-candidate-frozen-before-final"
_CANONICAL_BASE_SEED = 0

#: The only supported placement graph kind for the canonical protocol.
_GRAPH_KIND = "watts_strogatz"


class RobustnessConfigError(ConfigError):
    """Raised when a canonical robustness protocol envelope is invalid."""


class _StrictEnvelope(BaseModel):
    """Base envelope: unknown keys forbidden, frozen, strict, finite."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class CanonicalRobustnessProtocol(_StrictEnvelope):
    """Strict, frozen source protocol for the canonical robustness matrix."""

    protocol_id: str = Field(min_length=1, max_length=64)
    data_provenance: Literal["real_mnist", "injected"]
    headline_source_root: str = Field(min_length=1, max_length=512)
    artifact_root: str = Field(min_length=1, max_length=1024)
    data_root: str = Field(min_length=1, max_length=256)
    download: bool
    alphas: tuple[float, ...]
    placements: tuple[str, ...]
    topology_kind: Literal["watts_strogatz"]
    topology_n: int = Field(ge=10, le=20)
    topology_k: int = Field(ge=2)
    topology_p: float = Field(ge=0.0, le=1.0)
    rounds: int = Field(ge=1, le=MAX_ROUNDS)
    samples_per_node: int = Field(ge=1, le=MAX_SAMPLES_PER_NODE)
    evaluation_max_samples: int = Field(ge=1, le=10_000)
    evaluation_batch_size: int = Field(ge=1, le=1024)
    candidate_budget: int = Field(gt=0, le=MAX_CANDIDATE_BUDGET)
    selection_seed: int = Field(ge=0, le=MAX_SEED)
    selection_provenance: str = Field(min_length=1, max_length=256)
    selection_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    calibration_seed: int = Field(ge=0, le=MAX_SEED)
    threads: int = Field(ge=1, le=1)
    known_schedules: tuple[NamedSchedule, ...]
    held_out_schedule: NamedSchedule

    @model_validator(mode="after")
    def _validate_envelope(self) -> "CanonicalRobustnessProtocol":
        if self.topology_k >= self.topology_n:
            raise ValueError("topology_k must be strictly less than topology_n")
        if self.topology_k % 2 != 0:
            raise ValueError("topology_k must be even")
        if not self.alphas:
            raise ValueError("alphas must be non-empty")
        for alpha in self.alphas:
            if not (0.0 < float(alpha) < float("inf")):
                raise ValueError(f"alpha must be finite and positive, got {alpha!r}")
        if not self.placements:
            raise ValueError("placements must be non-empty")
        if len(set(self.placements)) != len(self.placements):
            raise ValueError("placements must be unique")
        for placement in self.placements:
            if placement not in CANONICAL_PLACEMENTS:
                raise ValueError(
                    f"placement must be one of {CANONICAL_PLACEMENTS}, got "
                    f"{placement!r}"
                )
        _validate_seed_tuple(self.selection_seeds, label="selection_seeds")
        _validate_seed_tuple(self.final_seeds, label="final_seeds")
        overlap = sorted(set(self.selection_seeds) & set(self.final_seeds))
        if overlap:
            raise ValueError(
                f"final_seeds must be disjoint from selection_seeds, overlap={overlap!r}"
            )
        if self.calibration_seed in set(self.selection_seeds) | set(
            self.final_seeds
        ):
            raise ValueError(
                "calibration_seed must be distinct from every selection/final seed"
            )
        if self.download:
            raise ValueError("canonical protocol must not enable dataset download")
        schedules = tuple(self.known_schedules)
        if not schedules:
            raise ValueError("known_schedules must be non-empty")
        if len(schedules) > MAX_KNOWN_SCHEDULES:
            raise ValueError(
                f"known_schedules exceeds the {MAX_KNOWN_SCHEDULES}-entry limit"
            )
        ids = tuple(entry.id for entry in schedules)
        if len(set(ids)) != len(ids):
            raise ValueError(f"known schedule ids must be unique, got {ids!r}")
        if any(entry.schedule.kind == "held_out" for entry in schedules):
            raise ValueError("known schedules must not use the held_out kind")
        if self.held_out_schedule.schedule.kind != "held_out":
            raise ValueError("held_out_schedule must use the held_out kind")
        if self.held_out_schedule.id in set(ids):
            raise ValueError("held_out schedule id collides with a known schedule")
        if self.threads != 1:
            raise ValueError("canonical protocol pins threads=1")
        return self

    # -- builders ---------------------------------------------------------- #
    def _base_experiment(self, alpha: float) -> ExperimentConfig:
        return ExperimentConfig(
            seed=_CANONICAL_BASE_SEED,
            rounds=self.rounds,
            device="cpu",
            bitwise=True,
            model=ModelConfig(),
            data=DataConfig(
                dataset="mnist",
                partition="dirichlet",
                dirichlet_alpha=float(alpha),
                samples_per_node=self.samples_per_node,
                data_root=self.data_root,
                download=self.download,
            ),
            training=TrainingConfig(
                optimizer="sgd",
                learning_rate=_CANONICAL_LEARNING_RATE,
                batch_size=_CANONICAL_TRAIN_BATCH,
                local_epochs=1,
                num_workers=0,
            ),
            topology=WattsStrogatzTopology(
                kind="watts_strogatz",
                n=self.topology_n,
                k=self.topology_k,
                p=self.topology_p,
            ),
            # Schema placeholder: ``AttackConfig`` requires a selected node for
            # a non-zero intensity. The placement contract deterministically
            # overrides this per cell, so the placeholder is never executed.
            attack=AttackConfig(compromised_node=0, intensity="strong"),
            aggregation=PlainAverageAggregation(),
            evaluation=EvaluationConfig(
                max_samples=self.evaluation_max_samples,
                batch_size=self.evaluation_batch_size,
            ),
            trace=TraceConfig(snapshot_cadence=1),
        )

    def _default_panel(self) -> tuple[PanelPolicy, ...]:
        project = CitationMetadata(kind="project", title="project comparator")
        tuned = TuningProvenance(
            mode="tuned",
            candidate_budget=self.candidate_budget,
            selection_seed=self.selection_seed,
            selection_provenance=self.selection_provenance,
        )
        exempt = TuningProvenance(
            mode="exempt",
            candidate_budget=0,
            selection_seed=None,
            selection_provenance=None,
        )
        balance_citation = CitationMetadata.model_validate(BALANCE_CITATION)
        return (
            PanelPolicy(
                id="local_only",
                implementation="local only",
                aggregation={"kind": "local_only"},
                citation=project,
                tuning=exempt,
            ),
            PanelPolicy(
                id="plain_average",
                implementation="plain average",
                aggregation={"kind": "plain_average"},
                citation=project,
                tuning=exempt,
            ),
            PanelPolicy(
                id="fixed_similarity",
                implementation="fixed similarity",
                aggregation={
                    "kind": "fixed_similarity",
                    "cutoff": 0.5,
                    "downweight_multiplier": 0.5,
                },
                citation=project,
                tuning=tuned,
            ),
            PanelPolicy(
                id="trust_weighted",
                implementation="trust weighted",
                aggregation={"kind": "trust_weighted", "beta": 1.0},
                citation=project,
                tuning=tuned,
            ),
            PanelPolicy(
                id="balance",
                implementation="faithful BALANCE",
                aggregation={"kind": "balance"},
                citation=balance_citation,
                tuning=tuned,
            ),
            PanelPolicy(
                id="bayesian",
                implementation="M002 Bayesian",
                aggregation={
                    "kind": "bayesian",
                    "artifact_dir": f"{self.artifact_root.rstrip('/')}/full",
                    "omega": 0.5,
                    "degree_cap": 4,
                    "tie_tolerance": 1e-9,
                },
                citation=project,
                tuning=tuned,
            ),
        )

    def condition_spec(
        self, alpha: float, placement: str
    ) -> HeldOutStudyConfig:
        """Build and validate one (alpha, placement) held-out condition."""

        if placement not in self.placements:
            raise RobustnessConfigError(
                f"placement {placement!r} is not part of this protocol "
                f"{self.placements!r}"
            )
        if alpha not in self.alphas:
            raise RobustnessConfigError(
                f"alpha {alpha!r} is not part of this protocol {self.alphas!r}"
            )
        study = BaselineStudyConfig(
            study_id=f"{self.protocol_id}-{_alpha_token(alpha)}-{placement}",
            data_provenance=self.data_provenance,
            base=self._base_experiment(alpha),
            seeds=self.selection_seeds,
            candidate_budget=self.candidate_budget,
            selection_seed=self.selection_seed,
            selection_provenance=self.selection_provenance,
            panel=self._default_panel(),
            placement=PlacementContract(
                placement=placement,
                graph_kind=_GRAPH_KIND,
                # D040 pins the seed set and requires degree-rank ties to be
                # disclosed rather than silently rejected: one declared
                # topology seed realizes a regular graph, so the contract must
                # not fail closed on it. The resolver still records
                # regular=True and distinct_degree_groups=False honestly.
                require_varied_degrees=False,
            ),
        )
        return HeldOutStudyConfig(
            study=study,
            final_seeds=self.final_seeds,
            known_schedules=self.known_schedules,
            held_out_schedule=self.held_out_schedule,
        )

    def conditions(self) -> tuple["RobustnessCondition", ...]:
        """Every condition in the frozen alpha x placement matrix."""

        built: list[RobustnessCondition] = []
        for alpha in self.alphas:
            for placement in self.placements:
                built.append(
                    RobustnessCondition(
                        condition_id=f"{_alpha_token(alpha)}__{placement}",
                        alpha=float(alpha),
                        placement=placement,
                        spec=self.condition_spec(alpha, placement),
                    )
                )
        if self.data_provenance == "real_mnist" and len(built) != 9:
            raise RobustnessConfigError(
                "the canonical real-MNIST matrix must declare exactly nine "
                f"conditions (3 alphas x 3 placements), got {len(built)}"
            )
        return tuple(built)


@dataclass(frozen=True, slots=True)
class RobustnessCondition:
    """One frozen (alpha, placement) condition and its validated study spec."""

    condition_id: str
    alpha: float
    placement: str
    spec: HeldOutStudyConfig


def _validate_seed_tuple(seeds: tuple[int, ...], *, label: str) -> None:
    if not seeds:
        raise ValueError(f"{label} must declare at least one seed")
    if len(seeds) > MAX_STUDY_SEEDS:
        raise ValueError(
            f"{label} declares {len(seeds)} entries, limit is {MAX_STUDY_SEEDS}"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{label} must be unique")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{label} must be integers, got {seed!r}")
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed {seed!r} is outside [0, {MAX_SEED}]")


def _alpha_token(alpha: float) -> str:
    """A study-id-safe token for one alpha (``0.1`` -> ``a0p1``)."""

    text = f"{float(alpha):g}"
    return "a" + text.replace("-", "m").replace(".", "p")


def _canonical_known_schedules() -> tuple[NamedSchedule, ...]:
    return (
        NamedSchedule(
            id="duty_cycle",
            schedule=build_schedule(
                "duty_cycle", on_rounds=3, off_rounds=2, intensity="strong"
            ),
        ),
        NamedSchedule(
            id="periodic",
            schedule=build_schedule(
                "periodic", period=6, strong_rounds=2, weak_rounds=2
            ),
        ),
        NamedSchedule(
            id="delayed_onset",
            schedule=build_schedule(
                "delayed_onset", onset_round=2, intensity="strong"
            ),
        ),
        NamedSchedule(
            id="random",
            schedule=build_schedule("random", p_on=0.5, intensity="weak"),
        ),
    )


def _canonical_held_out_schedule() -> NamedSchedule:
    return NamedSchedule(id="held_out", schedule=build_schedule("held_out"))


def build_canonical_protocol(
    *,
    headline_source_root: str,
    artifact_root: str,
    data_root: str = "data",
) -> CanonicalRobustnessProtocol:
    """Build the public real-MNIST canonical robustness protocol.

    Injected provenance and dataset download can never be selected here; the
    internal :func:`build_injected_protocol` exists only for isolated tests.
    """

    if not isinstance(headline_source_root, str) or not headline_source_root:
        raise RobustnessConfigError("headline_source_root must be a non-empty string")
    if not isinstance(artifact_root, str) or not artifact_root:
        raise RobustnessConfigError("artifact_root must be a non-empty string")
    return CanonicalRobustnessProtocol(
        protocol_id=CANONICAL_PROTOCOL_ID,
        data_provenance="real_mnist",
        headline_source_root=headline_source_root,
        artifact_root=artifact_root,
        data_root=data_root,
        download=False,
        alphas=CANONICAL_ALPHAS,
        placements=CANONICAL_PLACEMENTS,
        topology_kind=_GRAPH_KIND,
        topology_n=CANONICAL_TOPOLOGY["n"],
        topology_k=CANONICAL_TOPOLOGY["k"],
        topology_p=CANONICAL_TOPOLOGY["p"],
        rounds=_CANONICAL_ROUNDS,
        samples_per_node=_CANONICAL_SAMPLES_PER_NODE,
        evaluation_max_samples=_CANONICAL_EVAL_SAMPLES,
        evaluation_batch_size=_CANONICAL_EVAL_BATCH,
        candidate_budget=_CANONICAL_CANDIDATE_BUDGET,
        selection_seed=_CANONICAL_SELECTION_SEED,
        selection_provenance=_CANONICAL_SELECTION_PROVENANCE,
        selection_seeds=CANONICAL_SELECTION_SEEDS,
        final_seeds=CANONICAL_FINAL_SEEDS,
        calibration_seed=CANONICAL_CALIBRATION_SEED,
        threads=1,
        known_schedules=_canonical_known_schedules(),
        held_out_schedule=_canonical_held_out_schedule(),
    )


def build_injected_protocol(
    *,
    artifact_root: str,
    headline_source_root: str = "<injected>",
    data_root: str = "data",
) -> CanonicalRobustnessProtocol:
    """Internal test API: an injected-provenance twin of the canonical protocol.

    This exists so isolated tests can exercise the full nine-condition matrix
    without downloaded MNIST. It is deliberately a separate, explicitly named
    constructor and never reachable from the public real-MNIST entry point.
    """

    return CanonicalRobustnessProtocol(
        protocol_id="m003-canonical-robustness-injected",
        data_provenance="injected",
        headline_source_root=headline_source_root,
        artifact_root=artifact_root,
        data_root=data_root,
        download=False,
        alphas=CANONICAL_ALPHAS,
        placements=CANONICAL_PLACEMENTS,
        topology_kind=_GRAPH_KIND,
        topology_n=CANONICAL_TOPOLOGY["n"],
        topology_k=CANONICAL_TOPOLOGY["k"],
        topology_p=CANONICAL_TOPOLOGY["p"],
        rounds=_CANONICAL_ROUNDS,
        samples_per_node=_CANONICAL_SAMPLES_PER_NODE,
        evaluation_max_samples=_CANONICAL_EVAL_SAMPLES,
        evaluation_batch_size=_CANONICAL_EVAL_BATCH,
        candidate_budget=_CANONICAL_CANDIDATE_BUDGET,
        selection_seed=_CANONICAL_SELECTION_SEED,
        selection_provenance=_CANONICAL_SELECTION_PROVENANCE,
        selection_seeds=CANONICAL_SELECTION_SEEDS,
        final_seeds=CANONICAL_FINAL_SEEDS,
        calibration_seed=CANONICAL_CALIBRATION_SEED,
        threads=1,
        known_schedules=_canonical_known_schedules(),
        held_out_schedule=_canonical_held_out_schedule(),
    )
