"""End-to-end integration tests for the artifact-backed Bayesian policy (T03).

These tests drive the **production** composition closure:

    strict config -> one-time artifact load -> build_policy -> LocalObservation
    -> WeightProfile -> round aggregation -> schema-v1 TraceWriter -> ReplayModel

Everything is offline and in-process: the checked-in hand-computable artifact is
loaded from the git-tracked fixture directory, the dataset and evaluator
reference are tiny in-memory tensors, and a controlled scalar
:class:`~dfl_sim.observation.ObservationBuilder` deterministically lands every
neighbor on fixture evidence cells. No policy is injected, so this proves the
ordinary config path selects, builds, and reuses the Bayesian policy exactly as
a baseline would.

The checked-in fixture declares only the ``mid`` progress bin, so a
``rounds=2`` episode is deliberate: round 0 honestly falls back to exact
self-only weights and round 1 performs the posterior-weighted argmax. Both are
visible in the unchanged schema-v1 trace and replay through the existing viewer
helpers without a schema fork.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dfl_sim import (
    DEFAULT_EPSILON,
    BayesianArtifactError,
    LocalObservation,
    LocalTrainResult,
    NeighborObservation,
    load_replay,
    run_from_config,
    select_frame,
    select_frame_node,
    step_round,
)
from helpers import MNISTShapedDataset, constant_loss_evaluator_factory, make_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BAYESIAN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"

#: Per-sender fixture features chosen to land on declared evidence bins.
#: Disagreement edges [0.5, 1.0] -> 0.1 is "lo" (index 0) and 0.7 is "mid"
#: (index 1); relative_norm edge [1.0] -> 0.5 is "small" (index 0); loss_delta
#: edge [0.0] -> -0.5 is "neg" (index 0). Sender 0/2 land on (0, 0, 0) and
#: sender 1 lands on (1, 0, 0), giving one accept and one reject path.
_FEATURES: dict[int, tuple[float, float, float]] = {
    0: (0.1, 0.5, -0.5),  # evidence bin (0, 0, 0)
    1: (0.7, 0.5, -0.5),  # evidence bin (1, 0, 0)
    2: (0.1, 0.5, -0.5),  # evidence bin (0, 0, 0)
}


class FixtureObservationBuilder:
    """Controlled scalar observer that lands every neighbor on fixture cells.

    It ignores the round-start/trained models, sender values, and evaluator and
    returns only the frozen scalar :class:`LocalObservation`, so the selected
    profiles are hand-computable from the artifact tables alone.
    """

    def build_observation(
        self,
        *,
        node_id: int,
        round_index: int,
        round_start,
        trained,
        sender_states,
        evaluator,
    ) -> LocalObservation:
        neighbors: list[NeighborObservation] = []
        for sender, _state in sorted(sender_states, key=lambda item: int(item[0])):
            disagreement, relative_norm, loss_delta = _FEATURES[int(sender)]
            neighbors.append(
                NeighborObservation(
                    sender_id=int(sender),
                    directional_disagreement=disagreement,
                    relative_norm=relative_norm,
                    loss_delta=loss_delta,
                )
            )
        return LocalObservation(
            node_id=int(node_id),
            round_index=int(round_index),
            epsilon=DEFAULT_EPSILON,
            neighbors=tuple(neighbors),
        )


def _offset_trainer(scale: float = 10.0):
    """Tiny deterministic trainer so committed states are hand-computable."""

    def trainer(
        snapshot,
        dataset,
        *,
        config,
        node_id,
        round_index,
        train_seed,
        data_order_seed,
    ):
        state = {
            key: value.detach().clone() + scale * float(node_id)
            for key, value in snapshot.items()
        }
        return LocalTrainResult(
            node_id=node_id,
            round_index=round_index,
            state=state,
            pre_loss=1.0,
            post_loss=0.5,
            sample_count=4,
        )

    return trainer


def _bayesian_config(*, artifact_dir: Path = BAYESIAN_FIXTURE, rounds: int = 2):
    return make_config(
        rounds=rounds,
        data={"samples_per_node": 8},
        aggregation={
            "kind": "bayesian",
            "artifact_dir": str(artifact_dir),
            "omega": 0.5,
            "degree_cap": 4,
            "tie_tolerance": 1.0e-9,
        },
    )


def _run_episode(tmp_path: Path, *, artifact_dir: Path = BAYESIAN_FIXTURE):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = _bayesian_config(artifact_dir=artifact_dir)
    trace_dir = tmp_path / "trace"
    result = run_from_config(
        config,
        dataset=MNISTShapedDataset(48, seed=11, separable=True),
        evaluation_dataset=MNISTShapedDataset(32, seed=12, separable=True),
        trainer=_offset_trainer(),
        observer=FixtureObservationBuilder(),
        loss_evaluator=constant_loss_evaluator_factory,
        trace_path=trace_dir,
    )
    return result, trace_dir


# --------------------------------------------------------------------------- #
# Ordinary config -> policy -> trace -> replay, no injected policy
# --------------------------------------------------------------------------- #
def test_bayesian_episode_publishes_replayable_schema_v1_trace(tmp_path: Path) -> None:
    result, trace_dir = _run_episode(tmp_path)

    # The strict config selected the Bayesian variant; the trace was published.
    assert result.config.aggregation.kind == "bayesian"
    assert result.trace_path == trace_dir
    assert trace_dir.is_dir()

    model = load_replay(trace_dir)
    assert model.aggregation == "bayesian"
    assert model.rounds == (0, 1)
    assert model.node_ids == (0, 1, 2)
    assert model.compromised_node is None

    # Round 1 resolves the fixture's only ("mid") progress bin and selects the
    # posterior-weighted argmax: sender 1 lands on the rejection evidence cell.
    node0 = select_frame_node(model, 1, 0)
    assert node0.participant_ids == (0, 1)
    assert node0.multipliers == (1.0, 0.0)
    assert node0.weights == (1.0, 0.0)
    assert node0.multipliers[0] == 1.0  # self stays multiplier 1

    node2 = select_frame_node(model, 1, 2)
    assert node2.participant_ids == (2, 1)
    assert node2.multipliers == (1.0, 0.0)
    assert node2.weights == (1.0, 0.0)

    # Node 1 accepts both fixture-cell (0,0,0) senders: uniform three-way mean.
    node1 = select_frame_node(model, 1, 1)
    assert node1.participant_ids == (1, 0, 2)
    assert node1.multipliers == (1.0, 1.0, 1.0)
    assert node1.weights == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=1e-12)

    # Round 0 resolves to "early", which the fixture does not cover, so the
    # policy honestly falls back to exact self-only weights -- visible in the
    # unchanged schema-v1 trace.
    node1_round0 = select_frame_node(model, 0, 1)
    assert node1_round0.multipliers == (1.0, 0.0, 0.0)
    assert node1_round0.weights == (1.0, 0.0, 0.0)

    # Round stepping uses the real event round indices.
    assert step_round(model, 0, 1) == 1
    assert step_round(model, 1, 1) == 1
    assert step_round(model, 0, -1) == 0
    assert select_frame(model, 1).round_index == 1


def test_live_profiles_match_replayed_profiles(tmp_path: Path) -> None:
    result, trace_dir = _run_episode(tmp_path)
    model = load_replay(trace_dir)

    for record in result.rounds:
        frame = select_frame(model, record.round_index)
        for profile in record.profiles:
            assert profile.participant_ids[0] == profile.node_id
            assert abs(sum(profile.weights) - 1.0) <= 1e-9
            state = frame.state_for(profile.node_id)
            assert state.participant_ids == tuple(profile.participant_ids)
            assert state.multipliers == tuple(profile.multipliers)
            assert state.weights == pytest.approx(tuple(profile.weights), abs=1e-12)


def test_bayesian_episode_is_bitwise_deterministic(tmp_path: Path) -> None:
    first, _ = _run_episode(tmp_path / "a")
    second, _ = _run_episode(tmp_path / "b")
    assert first.final_round.profiles == second.final_round.profiles
    for node in first.node_ids:
        left = first.final_states[node]
        right = second.final_states[node]
        assert left.keys() == right.keys()
        for key in left:
            assert (left[key] == right[key]).all().item()


# --------------------------------------------------------------------------- #
# Construction failures leave no apparently complete trace (Q5)
# --------------------------------------------------------------------------- #
def test_missing_artifact_fails_before_trace_publication(tmp_path: Path) -> None:
    missing = tmp_path / "not-an-artifact"
    trace_dir = tmp_path / "trace"
    config = _bayesian_config(artifact_dir=missing)

    with pytest.raises(BayesianArtifactError) as excinfo:
        run_from_config(
            config,
            dataset=MNISTShapedDataset(32, seed=5),
            evaluation_dataset=MNISTShapedDataset(32, seed=6),
            trainer=_offset_trainer(),
            observer=FixtureObservationBuilder(),
            loss_evaluator=constant_loss_evaluator_factory,
            trace_path=trace_dir,
        )
    # Localized diagnostic names the artifact member/class, not table contents.
    assert excinfo.value.filename == "artifact"
    assert not trace_dir.exists()
