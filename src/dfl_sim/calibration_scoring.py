"""Offline calibration observation collection and counterfactual scoring (T02).

This module is the *offline-only* bridge between the existing synchronous round
engine and the S01 artifact estimator. It adds no new protocol: for each
sampled :class:`~dfl_sim.calibration.EpisodeSpec` it derives the per-episode
:class:`~dfl_sim.config.ExperimentConfig`, runs the real
:func:`~dfl_sim.run.run_from_config` path with the plain-average reference
policy, and injects a
:class:`CalibrationObservationBuilder` through the existing narrow observer
seam. The builder first delegates to
:class:`~dfl_sim.observation.DefaultObservationBuilder`, so the online policy
still receives exactly the frozen scalar
:class:`~dfl_sim.observation.LocalObservation`; types, design data, tensors,
and counterfactual scores never cross that boundary.

At each node/round the builder then performs **offline** counterfactual
scoring against the node's own immutable snapshot:

* it records the ordered hidden sender types (from the sampled single
  compromised node) and each sender's binned evidence triple;
* it enumerates every D022-eligible joint neighbor-multiplier action with the
  exact shared :func:`~dfl_sim.policy.enumerate_candidate_actions` helper and
  scores each by aggregating detached clones through
  :func:`~dfl_sim.aggregation.aggregate_weighted_neighborhood` and evaluating
  the resulting model on the one clean calibrator design split;
* it records the **self-only-anchored single-neighbor marginals** for every
  ``(evidence, context, hidden type, multiplier)`` so the estimator can compose
  the per-neighbor ``q`` rows under S01's explicit additive ``Q = sum_j q_j``
  approximation;
* utility is ``self_only_design_loss - candidate_design_loss`` and any
  non-finite loss or utility is a hard offline error, never a fabricated row.

Only the independently selected plain-average profile advances the real
trajectory: the counterfactual branches read cloned state and are never
committed. Records are immutable, tensor-free value objects bounded by the
configured episode/round/degree/candidate limits.

Nothing in this module is importable from the online policy or artifact-loader
boundary; the design split and hidden types stay entirely offline.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from torch.utils.data import Dataset

from .aggregation import aggregate_weighted_neighborhood
from .bayesian_artifact import BinSet, interval_bin_index
from .calibration import CalibrationSchedule, EpisodeSpec, derive_experiment_config
from .calibration_config import CalibrationConfig
from .config import ExperimentConfig
from .data import (
    load_mnist_dataset,
    partition_indices,
    partition_indices_with_design,
    subset_for_node,
)
from .models import ModelError, StateDictError, build_model, load_state_into_model
from .node import evaluate_cross_entropy
from .observation import (
    DefaultObservationBuilder,
    LocalObservation,
    LossEvaluator,
    NeighborObservation,
    ObservationBuilder,
)
from .policy import (
    PlainAveragePolicy,
    _resolve_degree_bin,
    _resolve_progress_bin,
    build_weight_profile,
    enumerate_candidate_actions,
)
from .reproducibility import resolve_seeds
from .run import run_from_config
from .topology import load_topology

__all__ = [
    "AdvancedProfile",
    "CalibrationCollection",
    "CalibrationObservationBuilder",
    "CalibrationScoringError",
    "CandidateProfileScore",
    "DesignLossEvaluator",
    "EpisodeCollection",
    "NeighborPayoffMarginal",
    "NodeRoundRecord",
    "bin_evidence",
    "collect_calibration",
    "collect_episode",
    "make_design_loss_evaluator",
]

#: Offline design-loss seam: ``L_design(state) -> float`` over the clean
#: calibrator design split. It receives a detached aggregate clone plus
#: ``node_id``/``round_index``/``context`` and must return a finite real number.
DesignLossEvaluator: TypeAlias = Callable[..., float]


class CalibrationScoringError(RuntimeError):
    """Raised when offline collection or counterfactual scoring fails.

    Raised for a malformed observation context, a non-finite design loss or
    utility, or a broken snapshot invariant. Every failure aborts the
    calibration before publication rather than emitting a fabricated row.
    """


# --------------------------------------------------------------------------- #
# Immutable, tensor-free records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CandidateProfileScore:
    """One scored joint neighbor-multiplier action (self fixed at 1).

    ``multipliers`` holds only the neighbor multipliers in canonical ascending
    sender order; the self multiplier is always exactly ``1``. ``design_loss``
    is the aggregate model's mean cross-entropy on the clean design split and
    ``utility`` is the self-only-anchored delta ``self_loss - design_loss``.
    """

    multipliers: tuple[float, ...]
    design_loss: float
    utility: float


@dataclass(frozen=True, slots=True)
class NeighborPayoffMarginal:
    """One self-only-anchored single-neighbor payoff marginal (T02/T03).

    Keyed by the neighbor's binned evidence triple, the receiver context
    ``(degree_bin, progress)``, the offline hidden neighbor type, and the
    applied neighbor multiplier. The value is the exact anchored utility
    ``self_only_design_loss - single_neighbor_design_loss``. A multiplier of
    ``0`` reproduces the self-only aggregate, so its utility is exactly zero.
    """

    evidence_bin: tuple[int, int, int]
    degree_bin: str
    progress: str
    hidden_type: int
    multiplier: float
    design_loss: float
    utility: float


@dataclass(frozen=True, slots=True)
class NodeRoundRecord:
    """Immutable offline record for one scored node/round (no tensors).

    Carries the frozen observation context, the ordered sender evidence and
    hidden types, the self-only anchor loss, the plain-average profile the real
    trajectory advanced, every scored joint candidate (for coverage/audit), and
    the anchored single-neighbor marginals the estimator consumes.
    """

    node_id: int
    round_index: int
    degree: int
    degree_bin: str
    progress: str
    participant_ids: tuple[int, ...]
    sender_ids: tuple[int, ...]
    evidence_bins: tuple[tuple[int, int, int], ...]
    hidden_types: tuple[int, ...]
    self_design_loss: float
    plain_full_multipliers: tuple[float, ...]
    candidates: tuple[CandidateProfileScore, ...]
    marginals: tuple[NeighborPayoffMarginal, ...]


@dataclass(frozen=True, slots=True)
class AdvancedProfile:
    """The participant-aligned multipliers the real round actually committed."""

    node_id: int
    round_index: int
    multipliers: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EpisodeCollection:
    """Collected offline records for one reference episode."""

    spec: EpisodeSpec
    design_indices: tuple[int, ...]
    records: tuple[NodeRoundRecord, ...]
    advanced_profiles: tuple[AdvancedProfile, ...]


@dataclass(frozen=True, slots=True)
class CalibrationCollection:
    """Collected offline records for every schedule episode (R013/R014)."""

    config: CalibrationConfig
    schedule: CalibrationSchedule
    episodes: tuple[EpisodeCollection, ...]

    @property
    def records(self) -> tuple[NodeRoundRecord, ...]:
        """Every node/round record across all episodes, in collection order."""

        ordered: list[NodeRoundRecord] = []
        for episode in self.episodes:
            ordered.extend(episode.records)
        return tuple(ordered)

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def round_count(self) -> int:
        """Total node/round records collected (support ``calibration_rounds``)."""

        return sum(len(episode.records) for episode in self.episodes)


# --------------------------------------------------------------------------- #
# Evidence binning (shared with the online policy convention)
# --------------------------------------------------------------------------- #
def bin_evidence(
    bins: BinSet, neighbor: NeighborObservation
) -> tuple[int, int, int]:
    """Bin one neighbor's scalar feature triple into the configured bins.

    Uses the exact :func:`~dfl_sim.bayesian_artifact.interval_bin_index`
    convention the online policy applies, so calibration evidence rows and
    online lookups can never disagree about a cell boundary.
    """

    return (
        interval_bin_index(bins.disagreement, neighbor.directional_disagreement),
        interval_bin_index(bins.relative_norm, neighbor.relative_norm),
        interval_bin_index(bins.loss_delta, neighbor.loss_delta),
    )


# --------------------------------------------------------------------------- #
# Design-loss evaluator
# --------------------------------------------------------------------------- #
def make_design_loss_evaluator(
    config: ExperimentConfig,
    design_dataset: Dataset,
    *,
    label: str = "calibration-design",
) -> DesignLossEvaluator:
    """Build a deterministic design-split loss evaluator (T02).

    The returned callable loads a validated state clone into a fresh
    deterministic model and evaluates mean cross-entropy on the clean design
    subset. It is stateless across calls (a fresh model per call), so a
    counterfactual evaluation can never contaminate the next one.
    """

    if not isinstance(config, ExperimentConfig):  # pragma: no cover - defensive
        raise CalibrationScoringError(
            "make_design_loss_evaluator requires an ExperimentConfig, got "
            f"{type(config).__name__}"
        )
    if len(design_dataset) == 0:
        raise CalibrationScoringError("design split is empty")
    model_name = config.model.name
    batch_size = int(config.training.batch_size)

    def evaluate(
        state: Mapping,
        *,
        node_id: int | None = None,
        round_index: int | None = None,
        context: str = "",
    ) -> float:
        suffix = f" {context}" if context else ""
        ctx = f"{label} node={node_id} round={round_index}{suffix}"
        try:
            model = build_model(model_name, seed=0)
            load_state_into_model(model, state, context=ctx)
        except (ModelError, StateDictError) as exc:
            raise CalibrationScoringError(
                f"design model load failed ({ctx}): {exc}"
            ) from exc
        return float(
            evaluate_cross_entropy(
                model, design_dataset, batch_size=batch_size, context=ctx
            )
        )

    return evaluate


# --------------------------------------------------------------------------- #
# Observation builder (narrow offline seam)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CalibrationObservationBuilder:
    """Offline observer that delegates, then scores counterfactuals (T02).

    Constructed once per episode and injected through the existing
    ``observer`` seam of :func:`~dfl_sim.run.run_from_config`. Its
    :meth:`build_observation` returns the *exact* frozen
    :class:`~dfl_sim.observation.LocalObservation` produced by
    :class:`~dfl_sim.observation.DefaultObservationBuilder`; the online policy
    therefore observes no design data, hidden type, tensor, or score. The
    counterfactual scoring happens purely locally and appends one immutable
    :class:`NodeRoundRecord` per non-isolated node/round to ``sink``.
    """

    config: CalibrationConfig
    compromised_node: int | None
    total_rounds: int
    design_loss: DesignLossEvaluator
    sink: list[NodeRoundRecord]
    delegate: ObservationBuilder = field(default_factory=DefaultObservationBuilder)

    def __post_init__(self) -> None:
        if not isinstance(self.config, CalibrationConfig):
            raise CalibrationScoringError(
                "CalibrationObservationBuilder requires a CalibrationConfig, "
                f"got {type(self.config).__name__}"
            )
        node = self.compromised_node
        if node is not None and (isinstance(node, bool) or not isinstance(node, int)):
            raise CalibrationScoringError(
                f"compromised_node must be int or None, got {node!r}"
            )
        rounds = self.total_rounds
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
            raise CalibrationScoringError(
                f"total_rounds must be a positive int, got {rounds!r}"
            )
        if not callable(self.design_loss):
            raise CalibrationScoringError("design_loss must be callable")
        if not isinstance(self.sink, list):
            raise CalibrationScoringError("sink must be a list")

    # -- observation seam --------------------------------------------------- #
    def build_observation(
        self,
        *,
        node_id: int,
        round_index: int,
        round_start: Mapping,
        trained: Mapping,
        sender_states: Sequence[tuple[int, Mapping]],
        evaluator: LossEvaluator,
    ) -> LocalObservation:
        """Delegate the online observation, then record offline scores."""

        observation = self.delegate.build_observation(
            node_id=node_id,
            round_index=round_index,
            round_start=round_start,
            trained=trained,
            sender_states=sender_states,
            evaluator=evaluator,
        )
        if observation.degree > 0:
            record = self._score_observation(
                observation,
                trained=trained,
                sender_states=sender_states,
            )
            self.sink.append(record)
        return observation

    # -- offline scoring ---------------------------------------------------- #
    def _score_observation(
        self,
        observation: LocalObservation,
        *,
        trained: Mapping,
        sender_states: Sequence[tuple[int, Mapping]],
    ) -> NodeRoundRecord:
        action_space = self.config.action_space
        multiplier_set = tuple(float(value) for value in action_space.multiplier_set)
        omega = float(action_space.omega)
        degree = observation.degree

        degree_bin = _resolve_degree_bin(
            tuple(self.config.bins.degree.labels), degree
        )
        progress = _resolve_progress_bin(
            tuple(self.config.bins.progress.labels),
            observation.round_index,
            self.total_rounds,
        )
        if degree_bin is None:
            raise CalibrationScoringError(
                f"degree {degree} has no configured degree bin "
                f"(node={observation.node_id} round={observation.round_index})"
            )
        if progress is None:
            raise CalibrationScoringError(
                f"round {observation.round_index} has no configured progress "
                f"context (total_rounds={self.total_rounds})"
            )

        state_by_sender = {int(sender): state for sender, state in sender_states}
        sender_ids = observation.neighbor_ids
        try:
            neighbor_states = [state_by_sender[sender] for sender in sender_ids]
        except KeyError as exc:  # pragma: no cover - engine guarantees coverage
            raise CalibrationScoringError(
                f"observation references missing sender {exc.args[0]!r} "
                f"(node={observation.node_id} round={observation.round_index})"
            ) from exc

        evidence_bins = tuple(
            bin_evidence(self.config.bins, neighbor)
            for neighbor in observation.neighbors
        )
        hidden_types = tuple(
            1 if self.compromised_node is not None and sender == self.compromised_node else 0
            for sender in sender_ids
        )
        if sum(hidden_types) > 1:  # pragma: no cover - at-most-one by construction
            raise CalibrationScoringError(
                "more than one compromised sender observed "
                f"(node={observation.node_id} round={observation.round_index})"
            )

        participants = observation.participant_ids
        ctx = (
            f"node={observation.node_id} round={observation.round_index} "
            f"degree={degree}"
        )

        self_loss = self._anchored_loss(
            participants,
            trained,
            neighbor_states,
            (1.0,) + (0.0,) * degree,
            omega=omega,
            context=f"self-only {ctx}",
        )

        disagreements = tuple(
            neighbor.directional_disagreement for neighbor in observation.neighbors
        )
        actions = enumerate_candidate_actions(
            degree,
            multiplier_set,
            degree_cap=action_space.degree_cap,
            disagreements=disagreements,
            sender_ids=sender_ids,
        )
        candidates: list[CandidateProfileScore] = []
        for action in actions:
            candidate_loss = self._anchored_loss(
                participants,
                trained,
                neighbor_states,
                (1.0,) + tuple(action),
                omega=omega,
                context=f"candidate {tuple(action)} {ctx}",
            )
            utility = self_loss - candidate_loss
            _require_finite(utility, label="candidate utility", context=ctx)
            candidates.append(
                CandidateProfileScore(
                    multipliers=tuple(float(value) for value in action),
                    design_loss=candidate_loss,
                    utility=utility,
                )
            )

        marginals: list[NeighborPayoffMarginal] = []
        for position, sender in enumerate(sender_ids):
            for multiplier in multiplier_set:
                vector = [1.0] + [0.0] * degree
                vector[1 + position] = multiplier
                marginal_loss = self._anchored_loss(
                    participants,
                    trained,
                    neighbor_states,
                    tuple(vector),
                    omega=omega,
                    context=(
                        f"marginal sender={sender} m={multiplier} {ctx}"
                    ),
                )
                utility = self_loss - marginal_loss
                _require_finite(utility, label="marginal utility", context=ctx)
                marginals.append(
                    NeighborPayoffMarginal(
                        evidence_bin=evidence_bins[position],
                        degree_bin=degree_bin,
                        progress=progress,
                        hidden_type=hidden_types[position],
                        multiplier=float(multiplier),
                        design_loss=marginal_loss,
                        utility=utility,
                    )
                )

        return NodeRoundRecord(
            node_id=observation.node_id,
            round_index=observation.round_index,
            degree=degree,
            degree_bin=degree_bin,
            progress=progress,
            participant_ids=participants,
            sender_ids=sender_ids,
            evidence_bins=evidence_bins,
            hidden_types=hidden_types,
            self_design_loss=self_loss,
            plain_full_multipliers=(1.0,) * len(participants),
            candidates=tuple(candidates),
            marginals=tuple(marginals),
        )

    def _anchored_loss(
        self,
        participants: tuple[int, ...],
        trained: Mapping,
        neighbor_states: Sequence[Mapping],
        full_multipliers: Sequence[float],
        *,
        omega: float,
        context: str,
    ) -> float:
        """Aggregate detached clones under one action and evaluate the design loss."""

        profile = build_weight_profile(
            participants,
            tuple(full_multipliers),
            omega=omega,
            context=context,
        )
        aggregate = aggregate_weighted_neighborhood(
            trained,
            neighbor_states,
            profile.weights,
            context=context,
        )
        try:
            value = self.design_loss(aggregate, context=context)
        except CalibrationScoringError:
            raise
        except Exception as exc:
            raise CalibrationScoringError(
                f"design loss evaluation failed ({context}): {exc}"
            ) from exc
        return _require_finite(value, label="design loss", context=context)


def _require_finite(value: object, *, label: str, context: str) -> float:
    """Return ``value`` as a finite float or raise an offline error."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise CalibrationScoringError(
            f"{label} is not a real number ({context}): {value!r}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationScoringError(
            f"{label} is not finite ({context}): {result!r}"
        )
    return result


# --------------------------------------------------------------------------- #
# Collection orchestration
# --------------------------------------------------------------------------- #
def collect_episode(
    config: CalibrationConfig,
    spec: EpisodeSpec,
    *,
    dataset: Dataset,
    policy: object | None = None,
    trainer: object | None = None,
) -> EpisodeCollection:
    """Run one reference episode and collect offline counterfactual records.

    The episode uses the real :func:`~dfl_sim.run.run_from_config` path with the
    plain-average reference policy. The clean design split is carved with
    :func:`~dfl_sim.data.partition_indices_with_design` from the **same** seeded
    permutation the runner uses for the node partitions (the episode seed's
    partition domain), so it is provably disjoint from every node-local block
    while never touching the final-test loader.
    """

    if not isinstance(config, CalibrationConfig):
        raise CalibrationScoringError(
            f"collect_episode requires a CalibrationConfig, got "
            f"{type(config).__name__}"
        )
    if not isinstance(spec, EpisodeSpec):
        raise CalibrationScoringError(
            f"collect_episode requires an EpisodeSpec, got {type(spec).__name__}"
        )

    resolved = derive_experiment_config(config, spec)
    seeds = resolve_seeds(resolved)
    topology = load_topology(resolved)
    node_ids = tuple(sorted(int(node) for node in topology.nodes()))

    split = partition_indices_with_design(
        len(dataset),
        node_ids,
        resolved.data.samples_per_node,
        seed=seeds.partition,
        design_size=config.design_split_size,
    )
    # Defensive: the runner's own partitioning must reproduce the node blocks
    # we subtracted from the single permutation, or the design split would not
    # actually be disjoint.
    expected_nodes = partition_indices(
        len(dataset),
        node_ids,
        resolved.data.samples_per_node,
        seed=seeds.partition,
    )
    if expected_nodes != split.node_indices:  # pragma: no cover - invariant
        raise CalibrationScoringError(
            "design split node blocks disagree with the runner partition"
        )

    design_subset = subset_for_node(dataset, split.design_indices)
    design_loss = make_design_loss_evaluator(resolved, design_subset)
    sink: list[NodeRoundRecord] = []
    builder = CalibrationObservationBuilder(
        config=config,
        compromised_node=spec.node,
        total_rounds=resolved.rounds,
        design_loss=design_loss,
        sink=sink,
    )

    active_policy = policy if policy is not None else PlainAveragePolicy()
    result = run_from_config(
        resolved,
        dataset=dataset,
        observer=builder,
        policy=active_policy,  # type: ignore[arg-type]
        trainer=trainer,  # type: ignore[arg-type]
    )

    advanced = tuple(
        AdvancedProfile(
            node_id=int(profile.node_id),
            round_index=index,
            multipliers=tuple(float(value) for value in profile.multipliers),
        )
        for index, round_result in enumerate(result.rounds)
        for profile in round_result.profiles
    )
    return EpisodeCollection(
        spec=spec,
        design_indices=tuple(split.design_indices),
        records=tuple(sink),
        advanced_profiles=advanced,
    )


def collect_calibration(
    config: CalibrationConfig,
    *,
    dataset: Dataset | None = None,
    policy: object | None = None,
    trainer: object | None = None,
) -> CalibrationCollection:
    """Collect offline records for every episode in the sampled schedule.

    ``dataset`` defaults to the production MNIST training split described by
    the embedded experiment; tests inject a deterministic in-process dataset.
    The schedule itself is built with :func:`~dfl_sim.calibration.build_schedule`
    so the same config, seed, and dataset always produce the same records.
    """

    if not isinstance(config, CalibrationConfig):
        raise CalibrationScoringError(
            f"collect_calibration requires a CalibrationConfig, got "
            f"{type(config).__name__}"
        )
    from .calibration import build_schedule

    schedule = build_schedule(config)
    if dataset is None:
        dataset = load_mnist_dataset(config.experiment)
    episodes = tuple(
        collect_episode(
            config, spec, dataset=dataset, policy=policy, trainer=trainer
        )
        for spec in schedule.specs
    )
    return CalibrationCollection(
        config=config, schedule=schedule, episodes=episodes
    )
