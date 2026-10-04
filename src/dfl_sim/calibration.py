"""Domain-seeded deterministic calibration schedule and prior inputs (R013/R014).

This module owns the *offline population model*: it turns a validated
:class:`~dfl_sim.calibration_config.CalibrationConfig` into an immutable,
reproducible list of reference-episode specifications, and derives the
joint-type prior-count inputs the estimator consumes.

Design constraints:

* **Deterministic and stratified.** The clean/attacked allocation is a rounded
  ``p0`` split clamped so a valid small run always contains at least one clean
  and one attacked episode. Conditional on an attacked episode the *effective*
  attack mixture (``config.effective_mu_a``) is cycled as a balanced mixture,
  so the pinned intensities are used as evenly as the episode count allows.
  Full and no-belief variants share the pinned source mixture; the
  no-strategic variant derives the degenerate strong mixture ``(5,)`` without
  changing counts, seeds, strata, or placements.
* **Uniform non-isolated placement.** Eligible (degree >= 1) nodes are shuffled
  once with a sampler-only, domain-labelled seed and then cycled, so placement
  coverage is uniform and independent of experiment training order.
* **Stable episode seeds.** Each episode seed is derived with
  :func:`~dfl_sim.reproducibility.stable_seed` under a calibration domain label,
  so the same config yields the same seeds across processes and machines.
* **No tensors or datasets.** Episode specs and prior counts contain only
  scalars, strings, and small tuples; no model state, dataset, or topology
  object is retained.
* **At most one compromised node.** Every attacked spec carries exactly one
  scalar node id; clean specs carry ``None``. The prior count therefore only
  ever sees the at-most-one-malicious support.

Nothing here is importable from the online policy / artifact-loader boundary.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np

from pydantic import ValidationError

from .calibration_config import CalibrationConfig
from .config import ATTACK_KAPPAS, ExperimentConfig
from .reproducibility import stable_seed
from .topology import TopologyError, load_topology

__all__ = [
    "EPISODE_SEED_DOMAIN",
    "SAMPLER_PLACEMENT_DOMAIN",
    "CalibrationError",
    "CalibrationSchedule",
    "EpisodeSpec",
    "PriorStateCounts",
    "ScheduleSummary",
    "build_schedule",
    "build_schedule_and_counts",
    "calibrate_from_config",
    "count_prior_states",
    "derive_experiment_config",
    "feasible_prior_states",
    "summarize_schedule",
]

#: Domain labels for stable seed derivation. Adding a label must never renumber
#: existing labels, so the schedules stay bitwise reproducible.
SAMPLER_PLACEMENT_DOMAIN = "calibration:sampler:placement"
EPISODE_SEED_DOMAIN = "calibration:episode"

#: Inverse of the D016 intensity table: kappa value -> named intensity.
INTENSITY_BY_KAPPA: dict[float, str] = {
    float(value): name for name, value in ATTACK_KAPPAS.items()
}


class CalibrationError(RuntimeError):
    """Raised when a calibration schedule or prior count cannot be built."""


Stratum = Literal["clean", "attacked"]
Intensity = Literal["zero", "weak", "strong"]


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """One immutable reference-episode specification.

    Contains only scalars and strings: no tensors, datasets, topology objects,
    or mutable containers. ``stratum`` is ``"clean"`` for a no-attack episode
    (``node`` is ``None``) or ``"attacked"`` for an episode with exactly one
    compromised ``node`` and one pinned ``intensity`` / numeric ``value``.
    """

    index: int
    seed: int
    stratum: Stratum
    node: int | None
    intensity: Intensity
    value: float

    @property
    def is_attacked(self) -> bool:
        return self.stratum == "attacked"


@dataclass(frozen=True, slots=True)
class ScheduleSummary:
    """Auditable counts for one built schedule (records finite-sample rounding)."""

    episodes: int
    clean: int
    attacked: int
    intensity_counts: tuple[tuple[str, int], ...]
    placement_counts: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class CalibrationSchedule:
    """A built schedule plus its summary (deterministic and immutable)."""

    specs: tuple[EpisodeSpec, ...]
    summary: ScheduleSummary


@dataclass(frozen=True, slots=True)
class PriorStateCounts:
    """Joint-type prior-count inputs derived from the sampled schedule.

    ``counts`` maps local degree to an immutable mapping from the receiver's
    ordered neighbor-type state (one bit per neighbor, in ascending sender
    order) to the number of sampled episodes in which a *benign* receiver of
    that degree observed it. Each state has at most one set bit (at-most-one
    compromised node per episode).
    """

    counts: Mapping[int, Mapping[tuple[int, ...], int]]
    episodes: int
    benign_observations: int


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _clean_count(config: CalibrationConfig) -> int:
    """Rounded, clamped clean allocation (always ``1 <= clean < episodes``)."""

    target = int(round(config.p0 * config.episodes))
    return _clamp(target, 1, config.episodes - 1)


def _eligible_nodes(topology) -> tuple[int, ...]:
    return tuple(
        sorted(
            int(node)
            for node, degree in topology.degree()
            if int(degree) >= 1
        )
    )


def build_schedule(config: CalibrationConfig) -> CalibrationSchedule:
    """Build the deterministic stratified calibration schedule.

    The returned :class:`CalibrationSchedule` carries the immutable episode
    specs plus an auditable summary of the actual counts. Raises
    :class:`CalibrationError` when the config's topology cannot be built or no
    non-isolated node exists for placement.
    """

    if not isinstance(config, CalibrationConfig):
        raise CalibrationError(
            "build_schedule requires a validated CalibrationConfig, got "
            f"{type(config).__name__}"
        )

    episodes = config.episodes
    clean_count = _clean_count(config)
    attacked_count = episodes - clean_count

    try:
        topology = load_topology(config.experiment)
    except TopologyError as exc:
        raise CalibrationError(f"calibration topology failed: {exc}") from exc

    eligible = _eligible_nodes(topology)
    if attacked_count > 0 and not eligible:
        raise CalibrationError(
            "calibration has attacked episodes but the topology has no "
            "non-isolated node for uniform placement"
        )

    master = int(config.experiment.seed)
    placement_seed = stable_seed(master, SAMPLER_PLACEMENT_DOMAIN)
    rng = np.random.default_rng(placement_seed)
    if eligible:
        order = rng.permutation(len(eligible))
        placement = tuple(eligible[int(position)] for position in order)
    else:  # pragma: no cover - only a fully isolated topology reaches here
        placement = ()

    mixture = tuple(float(value) for value in config.effective_mu_a)

    specs: list[EpisodeSpec] = []
    for index in range(clean_count):
        specs.append(
            EpisodeSpec(
                index=index,
                seed=stable_seed(master, f"{EPISODE_SEED_DOMAIN}:{index}"),
                stratum="clean",
                node=None,
                intensity="zero",
                value=0.0,
            )
        )
    for ordinal in range(attacked_count):
        index = clean_count + ordinal
        value = mixture[ordinal % len(mixture)]
        intensity = INTENSITY_BY_KAPPA[value]
        node = placement[ordinal % len(placement)]
        specs.append(
            EpisodeSpec(
                index=index,
                seed=stable_seed(master, f"{EPISODE_SEED_DOMAIN}:{index}"),
                stratum="attacked",
                node=node,
                intensity=intensity,  # type: ignore[arg-type]
                value=value,
            )
        )

    ordered = tuple(specs)
    return CalibrationSchedule(
        specs=ordered, summary=summarize_schedule(ordered)
    )


def summarize_schedule(specs: Sequence[EpisodeSpec]) -> ScheduleSummary:
    """Return the deterministic count summary for ``specs``."""

    clean = 0
    intensities: Counter[str] = Counter()
    placements: Counter[int] = Counter()
    for spec in specs:
        if spec.stratum == "clean":
            clean += 1
            continue
        intensities[spec.intensity] += 1
        if spec.node is None:  # pragma: no cover - guarded by construction
            raise CalibrationError("attacked episode is missing a compromised node")
        placements[int(spec.node)] += 1
    attacked = len(specs) - clean
    return ScheduleSummary(
        episodes=len(specs),
        clean=clean,
        attacked=attacked,
        intensity_counts=tuple(sorted(intensities.items())),
        placement_counts=tuple(sorted(placements.items())),
    )


def feasible_prior_states(degree: int) -> tuple[tuple[int, ...], ...]:
    """Return the at-most-one-malicious joint states for ``degree``.

    Ordered as the all-benign state first, then one malicious position at a
    time in ascending position order, matching the artifact's feasible prior
    support and the sender order preserved by :func:`count_prior_states`.
    """

    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise CalibrationError(f"degree must be a non-negative int, got {degree!r}")
    states: list[tuple[int, ...]] = [(0,) * degree]
    for position in range(degree):
        state = [0] * degree
        state[position] = 1
        states.append(tuple(state))
    return tuple(states)


def count_prior_states(
    config: CalibrationConfig, specs: Sequence[EpisodeSpec]
) -> PriorStateCounts:
    """Count each benign receiver's ordered neighbor-type state once per episode.

    The count is per sampled episode, never per round. Neighbor order is the
    ascending sender order the round engine uses, so ``state[j]`` is the type
    of the ``j``-th neighbor. Only benign receivers contribute (the
    receiver-is-benign conditioning), and each state contains at most one
    malicious bit because exactly one node is compromised per attacked episode.
    """

    if not isinstance(config, CalibrationConfig):
        raise CalibrationError(
            "count_prior_states requires a validated CalibrationConfig, got "
            f"{type(config).__name__}"
        )
    try:
        topology = load_topology(config.experiment)
    except TopologyError as exc:
        raise CalibrationError(f"calibration topology failed: {exc}") from exc

    neighbors = {
        int(node): tuple(sorted(int(nb) for nb in topology.neighbors(node)))
        for node in topology.nodes()
    }

    raw: dict[int, Counter[tuple[int, ...]]] = {}
    benign_observations = 0
    for spec in specs:
        compromised = spec.node
        for receiver, receiver_neighbors in neighbors.items():
            if not receiver_neighbors:
                continue
            if compromised is not None and receiver == compromised:
                continue
            state = tuple(
                1 if neighbor == compromised else 0
                for neighbor in receiver_neighbors
            )
            raw.setdefault(len(receiver_neighbors), Counter())[state] += 1
            benign_observations += 1

    counts = {
        degree: MappingProxyType(dict(bucket))
        for degree, bucket in sorted(raw.items())
    }
    return PriorStateCounts(
        counts=MappingProxyType(counts),
        episodes=len(specs),
        benign_observations=benign_observations,
    )


def build_schedule_and_counts(
    config: CalibrationConfig,
) -> tuple[CalibrationSchedule, PriorStateCounts]:
    """Convenience: build the schedule and its prior-count inputs together."""

    schedule = build_schedule(config)
    return schedule, count_prior_states(config, schedule.specs)


def derive_experiment_config(
    config: CalibrationConfig, spec: EpisodeSpec
) -> ExperimentConfig:
    """Derive the per-episode :class:`ExperimentConfig` for one spec (T02).

    The embedded calibration experiment is copied and its ``seed`` and
    ``attack`` are replaced with the spec's stable episode seed and sampled
    fixed attack (``compromised_node=None`` and intensity ``"zero"`` for a
    clean episode). The result is re-validated through the same strict
    contract as production, so an out-of-range seed or an attack node outside
    the topology fails closed before any round. Only the two episode-varying
    fields change; topology, data, training, aggregation, evaluation, and trace
    settings are preserved byte-for-byte.
    """

    if not isinstance(config, CalibrationConfig):
        raise CalibrationError(
            "derive_experiment_config requires a validated CalibrationConfig, "
            f"got {type(config).__name__}"
        )
    if not isinstance(spec, EpisodeSpec):
        raise CalibrationError(
            "derive_experiment_config requires an EpisodeSpec, got "
            f"{type(spec).__name__}"
        )

    payload = config.experiment.model_dump(mode="json")
    payload["seed"] = int(spec.seed)
    payload["attack"] = {
        "compromised_node": spec.node,
        "intensity": spec.intensity,
    }
    try:
        return ExperimentConfig.model_validate(payload)
    except ValidationError as exc:
        raise CalibrationError(
            f"episode {spec.index} experiment derivation failed: "
            f"{_bounded_detail(exc)}"
        ) from exc


def calibrate_from_config(
    config: CalibrationConfig,
    output: str,
    *,
    dataset: object | None = None,
    config_path: str | None = None,
    data_root: str | None = None,
    policy: object | None = None,
    trainer: object | None = None,
) -> object:
    """Run the full offline calibration pipeline and publish the artifact.

    The orchestrator is the single entry point behind ``dfl calibrate``: it
    builds the deterministic schedule, collects real clean/attacked reference
    episodes through the existing round engine, estimates the smoothed
    schema-v1 tables, and atomically publishes the frozen artifact directory.
    The returned value is the publisher's :class:`PublishedArtifact`; it is
    annotated as ``object`` here only to keep this module free of a top-level
    import cycle with the offline scoring/estimation modules.
    """

    from .calibration_artifact_writer import publish_calibration_artifact
    from .calibration_estimator import estimate_artifact
    from .calibration_scoring import collect_calibration

    if not isinstance(config, CalibrationConfig):
        raise CalibrationError(
            "calibrate_from_config requires a validated CalibrationConfig, got "
            f"{type(config).__name__}"
        )
    if data_root is None:
        data_root = config.experiment.data.data_root

    collection = collect_calibration(
        config, dataset=dataset, policy=policy, trainer=trainer
    )
    schedule = collection.schedule
    prior_counts = count_prior_states(config, schedule.specs)
    artifact = estimate_artifact(
        config,
        collection.records,
        prior_counts,
        calibration_episodes=collection.episode_count,
        calibration_rounds=collection.round_count,
        provenance=collection.provenance,
    )
    return publish_calibration_artifact(
        output,
        artifact,
        config_path=config_path,
        data_root=data_root,
    )


def _bounded_detail(detail: object, *, limit: int = 400) -> str:
    """Collapse whitespace and truncate a derivation diagnostic."""

    collapsed = " ".join(str(detail).split())
    if len(collapsed) > limit:
        collapsed = collapsed[:limit] + "..."
    return collapsed
