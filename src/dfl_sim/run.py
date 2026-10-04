"""End-to-end episode wiring: YAML config -> synchronous DFL run -> artifacts.

:func:`run_from_config` is the single orchestrator that closes the S01 loop and
the S04 production path. It resets deterministic state, validates the config and
every requested output destination, builds/freezes the topology, loads and
seeds the configured IID or Dirichlet partition of the dataset, initializes one
deterministic small-CNN state cloned to every node, executes the configured
rounds through
:func:`~dfl_sim.simulator.run_round`, and (optionally) publishes either the
canonical minimal round log, a complete versioned schema-v1 trajectory, or both.

When a trace destination is requested the privileged post-commit evaluator and
the :class:`~dfl_sim.trajectory.TraceWriter` are constructed *outside* the round
engine: the evaluator sees only committed :class:`~dfl_sim.simulator.RoundResult`
objects and the global evaluation dataset, the writer sees only evaluated
records plus committed states, and neither is ever passed through the
observation/policy/controller seams. After each successful atomic round commit
the evaluator scores the round and exactly one trajectory event is written.

The dataset is injectable through ``dataset``/``dataset_loader`` and the
evaluation (privileged reference) dataset through
``evaluation_dataset``/``evaluation_dataset_loader`` so offline integration tests
never touch the network; the user-facing CLI only ever supplies configured
MNIST. The runner holds no hidden global state: the only process-wide settings
applied are the explicit :func:`~dfl_sim.reproducibility.configure_determinism`
values.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import networkx as nx
from torch.utils.data import Dataset

from .aggregation import ModelState
from .attacker import FixedAttackController, OutgoingController
from .config import ExperimentConfig, load_config
from .data import (
    load_mnist_dataset,
    load_mnist_evaluation_dataset,
    partition_dataset,
)
from .evaluator import EvaluatorError, PrivilegedEvaluator
from .models import build_model, clone_state_dict
from .observation import DefaultObservationBuilder, ObservationBuilder
from .policy import AggregationPolicy, BayesianDiagnostic, build_policy
from .reproducibility import configure_determinism, resolve_seeds
from .round_log import RoundLogWriter, validate_log_path
from .simulator import LossEvaluatorFactory, RoundResult, Trainer, run_round
from .topology import load_topology
from .trajectory import TraceWriter, TrajectoryError, validate_trace_path

__all__ = [
    "ConfigInput",
    "DatasetLoader",
    "EpisodeError",
    "EpisodeResult",
    "run_from_config",
]

#: A validated config, or a path to a YAML config to load.
ConfigInput = Union[ExperimentConfig, str, "os.PathLike[str]"]

#: Dataset injection seam: a callable that builds the dataset for a config.
DatasetLoader = Callable[[ExperimentConfig], Dataset]


class EpisodeError(RuntimeError):
    """Raised when an episode cannot be prepared or a round fails mid-run."""


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Immutable outcome of a completed episode.

    ``final_states`` maps every node to its committed state after the last
    round (detached clones). ``rounds`` holds the per-round summaries in
    execution order, ``log_path`` is the published minimal log path (or
    ``None`` when no log was requested), and ``trace_path`` is the published
    schema-v1 trajectory directory (or ``None`` when no trace was requested).
    """

    config: ExperimentConfig
    final_states: dict[int, ModelState]
    rounds: tuple[RoundResult, ...]
    log_path: Path | None
    trace_path: Path | None = None

    @property
    def node_ids(self) -> tuple[int, ...]:
        """Sorted node ids present in the episode."""

        return tuple(sorted(self.final_states))

    @property
    def final_round(self) -> RoundResult:
        """The last committed round summary."""

        if not self.rounds:  # pragma: no cover - schema guarantees >= 1 round
            raise EpisodeError("episode has no rounds")
        return self.rounds[-1]


def _resolve_config(config: ConfigInput) -> tuple[ExperimentConfig, Path | None]:
    if isinstance(config, ExperimentConfig):
        return config, None
    source_path = Path(config)
    return load_config(source_path), source_path


def run_from_config(
    config: ConfigInput,
    *,
    log_path: str | os.PathLike | None = None,
    trace_path: str | os.PathLike | None = None,
    dataset: Dataset | None = None,
    dataset_loader: DatasetLoader | None = None,
    evaluation_dataset: Dataset | None = None,
    evaluation_dataset_loader: DatasetLoader | None = None,
    trainer: Trainer | None = None,
    policy: AggregationPolicy | None = None,
    observer: ObservationBuilder | None = None,
    loss_evaluator: LossEvaluatorFactory | None = None,
    controller: OutgoingController | None = None,
    diagnostic_sink: Callable[[BayesianDiagnostic], None] | None = None,
) -> EpisodeResult:
    """Run one deterministic synchronous DFL episode and return its result.

    ``config`` is a validated :class:`~dfl_sim.config.ExperimentConfig` or a
    path to a strict YAML config. At most one of ``dataset`` (an in-memory
    :class:`~torch.utils.data.Dataset`) and ``dataset_loader`` (a callable
    ``config -> Dataset``) may be supplied; when both are omitted the
    production :func:`~dfl_sim.data.load_mnist_dataset` is used. The
    ``trainer``/``observer``/``loss_evaluator`` seams are forwarded to the
    round engine.

    One immutable outgoing controller is constructed from the resolved episode
    config (or supplied via ``controller``) and one immutable aggregation
    policy from the config's aggregation selector (or supplied via ``policy``);
    both are reused for every round, so neither a fixed attack nor a baseline
    can change mid-episode. Passing a custom policy or observer never exposes
    the topology, recipient identity, inboxes, evaluator metrics, trajectory
    records, or other nodes' states to the policy: the policy's sole input is
    the frozen per-node :class:`~dfl_sim.observation.LocalObservation`.

    When ``log_path`` is given, the canonical JSONL log is published atomically
    only after every round succeeds; a failure removes the temporary file and
    leaves any prior log untouched. When ``trace_path`` is given, a complete
    schema-v1 trajectory is published atomically after every round has been
    evaluated and written; a failure removes only the writer-owned staging
    directory and leaves the requested destination absent. Both destinations
    are validated before any training or evaluation work begins, and the trace
    is committed before the log so a trace failure never replaces a prior log.
    """

    if dataset is not None and dataset_loader is not None:
        raise EpisodeError("provide either dataset or dataset_loader, not both")
    if evaluation_dataset is not None and evaluation_dataset_loader is not None:
        raise EpisodeError(
            "provide either evaluation_dataset or evaluation_dataset_loader, not both"
        )

    resolved, source_path = _resolve_config(config)
    source_label = str(source_path) if source_path is not None else "<config>"

    # Fail fast on unsafe output targets (alias/non-regular/missing parent)
    # before any deterministic setup, data loading, or training work happens.
    validated_log: Path | None = None
    if log_path is not None:
        validated_log = validate_log_path(log_path, config_path=source_path)

    validated_trace: Path | None = None
    if trace_path is not None:
        aliases: tuple[str | os.PathLike, ...] = (
            (validated_log,) if validated_log is not None else ()
        )
        validated_trace = validate_trace_path(
            trace_path, config_path=source_path, alias_paths=aliases
        )

    seeds = resolve_seeds(resolved)
    configure_determinism(
        seeds,
        bitwise=resolved.bitwise,
        num_workers=resolved.training.num_workers,
    )

    topology: nx.Graph = load_topology(resolved)
    node_ids = tuple(sorted(int(node) for node in topology.nodes()))

    if dataset is not None:
        data = dataset
    elif dataset_loader is not None:
        data = dataset_loader(resolved)
    else:
        data = load_mnist_dataset(resolved)

    partitions = partition_dataset(
        data,
        node_ids,
        resolved.data.samples_per_node,
        seed=seeds.partition,
        partition=resolved.data.partition,
        alpha=resolved.data.dirichlet_alpha,
    )

    # One immutable controller for the whole episode: the D016 intensity is
    # fixed per episode and every round reuses the same object.
    outgoing_controller = (
        controller
        if controller is not None
        else FixedAttackController.from_experiment(resolved)
    )

    # One immutable policy and observer for the whole episode, built from the
    # validated aggregation selector, mirroring the fixed attacker controller.
    # D023: the validated round count is supplied at construction so the
    # Bayesian policy can resolve early/mid/late progress from the immutable
    # public run context while `select_weights` still receives only a
    # LocalObservation. Baselines ignore the extra argument.
    episode_policy = (
        policy
        if policy is not None
        else build_policy(
            resolved.aggregation,
            total_rounds=resolved.rounds,
            diagnostic_sink=diagnostic_sink,
        )
    )
    episode_observer = (
        observer if observer is not None else DefaultObservationBuilder()
    )

    # The privileged evaluator is built before training and lives outside the
    # round engine: it consumes only committed RoundResults and the global
    # evaluation dataset, so nothing it holds can reach the policy seam.
    evaluator: PrivilegedEvaluator | None = None
    if validated_trace is not None:
        if evaluation_dataset is not None:
            reference = evaluation_dataset
        elif evaluation_dataset_loader is not None:
            reference = evaluation_dataset_loader(resolved)
        else:
            reference = load_mnist_evaluation_dataset(resolved)
        evaluator = PrivilegedEvaluator.from_experiment(
            resolved, topology, reference, seeds
        )

    # One deterministic initial state, deep-cloned per node so the nodes share
    # no storage and cannot observe each other's mutation.
    base_model = build_model(resolved.model.name, seed=seeds.torch)
    initial_state = clone_state_dict(base_model.state_dict())

    def _execute(
        log_writer: RoundLogWriter | None,
        trace_writer: TraceWriter | None,
    ) -> tuple[dict[int, ModelState], list[RoundResult]]:
        current: dict[int, ModelState] = {
            node: clone_state_dict(initial_state) for node in node_ids
        }
        if trace_writer is not None:
            trace_writer.write_initial(current)
        results: list[RoundResult] = []
        for index in range(resolved.rounds):
            try:
                result = run_round(
                    config=resolved,
                    topology=topology,
                    snapshots=current,
                    partitions=partitions,
                    round_index=index,
                    seeds=seeds,
                    trainer=trainer,
                    policy=episode_policy,
                    observer=episode_observer,
                    loss_evaluator=loss_evaluator,
                    controller=outgoing_controller,
                )
            except EpisodeError:
                raise
            except Exception as exc:
                raise EpisodeError(
                    f"round {index} failed for {source_label}: {exc}"
                ) from exc

            if trace_writer is not None:
                assert evaluator is not None  # built whenever a trace is set
                try:
                    evaluated = evaluator.evaluate_round(result)
                except EvaluatorError as exc:
                    raise EpisodeError(
                        f"evaluation failed for round {index} "
                        f"in {source_label}: {exc}"
                    ) from exc
                try:
                    trace_writer.write_round(
                        evaluated,
                        messages=result.messages,
                        states=result.states,
                    )
                except TrajectoryError as exc:
                    raise EpisodeError(
                        f"trace emission failed for round {index} "
                        f"in {source_label}: {exc}"
                    ) from exc

            current = result.states
            results.append(result)
            if log_writer is not None:
                log_writer.write_round(result)
        return current, results

    # Enter the log writer first and the trace writer second so the ExitStack
    # unwinds the trace first: the trajectory is committed before the log, and
    # a trace failure aborts the log instead of replacing a prior log.
    with ExitStack() as stack:
        log_writer: RoundLogWriter | None = None
        trace_writer: TraceWriter | None = None
        if validated_log is not None:
            log_writer = stack.enter_context(RoundLogWriter(validated_log))
        if validated_trace is not None:
            trace_writer = stack.enter_context(
                TraceWriter(
                    validated_trace,
                    config=resolved,
                    topology=topology,
                    config_path=source_path,
                    alias_paths=(
                        (validated_log,) if validated_log is not None else ()
                    ),
                )
            )
        final_states, round_results = _execute(log_writer, trace_writer)

    return EpisodeResult(
        config=resolved,
        final_states=final_states,
        rounds=tuple(round_results),
        log_path=validated_log,
        trace_path=validated_trace,
    )
