"""Contract tests for the privileged post-commit evaluator and D018 provenance.

Covers R008 evaluator-only truth and global metrics, D012 structural boundary
preservation, D018 weight-aware multi-hop propagation, and the Q5/Q6/Q7
failure, load, and negative surfaces.

Everything is offline and in-process: a tiny injected model/dataset drives the
real :func:`~dfl_sim.simulator.run_round` commit path, and a few tests build
:class:`~dfl_sim.simulator.RoundResult` variants via ``dataclasses.replace`` to
prove the evaluator fails closed on malformed rounds.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from dfl_sim import (
    ATTACK_DELTA_REVERSAL,
    ATTACK_NONE,
    BENIGN,
    MALICIOUS,
    AttackAction,
    EvaluatedRound,
    EvaluatorError,
    LocalTrainResult,
    NodeEvaluation,
    PlainAveragePolicy,
    PropagationState,
    ProvenanceTracker,
    PrivilegedEvaluator,
    RoundResult,
    build_evaluator,
    load_topology,
    resolve_seeds,
    run_round,
    select_reference_subset,
    validate_round_structure,
)
from helpers import constant_loss_evaluator_factory, make_config

NODES3 = (0, 1, 2)


# --------------------------------------------------------------------------- #
# Tiny offline fixtures
# --------------------------------------------------------------------------- #
class TinyNet(nn.Module):
    """Two-feature/two-class linear model with a small state dict."""

    def __init__(self, in_features: int = 2, num_classes: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class TinyDataset(Dataset):
    """Deterministic two-feature dataset with controllable labels/finiteness."""

    def __init__(
        self,
        num_examples: int = 16,
        *,
        seed: int = 0,
        num_classes: int = 2,
        fill: float | None = None,
        label_offset: int = 0,
    ) -> None:
        generator = torch.Generator().manual_seed(int(seed))
        if fill is None:
            self.features = torch.rand(num_examples, 2, generator=generator)
        else:
            self.features = torch.full((num_examples, 2), float(fill))
        self.labels = (
            torch.arange(num_examples, dtype=torch.long) % num_classes
        ) + int(label_offset)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def _tiny_state(seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = TinyNet()
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _tiny_trainer(
    snapshot,
    dataset,
    *,
    config,
    node_id,
    round_index,
    train_seed,
    data_order_seed,
):
    """Deterministic trainer producing a valid, node-distinct tiny state."""

    state = {key: value.detach().clone() for key, value in snapshot.items()}
    state["linear.bias"] = state["linear.bias"] + 0.05 * float(node_id)
    return LocalTrainResult(
        node_id=node_id,
        round_index=round_index,
        state=state,
        pre_loss=1.0,
        post_loss=0.5,
        sample_count=4,
    )


def _plain(**overrides):
    base = {
        "topology": {"kind": "chain3"},
        "aggregation": {"kind": "plain_average"},
        "evaluation": {"max_samples": 8, "batch_size": 4},
    }
    base.update(overrides)
    return make_config(**base)


def _drive_rounds(config, rounds: int = 2):
    topology = load_topology(config)
    nodes = tuple(sorted(int(node) for node in topology.nodes()))
    seeds = resolve_seeds(config)
    current = {node: _tiny_state() for node in nodes}
    results: list[RoundResult] = []
    for index in range(rounds):
        result = run_round(
            config=config,
            topology=topology,
            snapshots=current,
            partitions={node: None for node in nodes},
            round_index=index,
            seeds=seeds,
            trainer=_tiny_trainer,
            loss_evaluator=constant_loss_evaluator_factory,
        )
        results.append(result)
        current = result.states
    return topology, results


def _evaluator(config, topology, *, reference=None, model_factory=None):
    return PrivilegedEvaluator.from_experiment(
        config,
        topology,
        reference if reference is not None else TinyDataset(16),
        resolve_seeds(config),
        model_factory=model_factory or TinyNet,
    )


def _find(record: EvaluatedRound, node: int) -> NodeEvaluation:
    matches = [item for item in record.nodes if item.node_id == node]
    assert len(matches) == 1
    return matches[0]


# --------------------------------------------------------------------------- #
# Reference subset selection (Q6)
# --------------------------------------------------------------------------- #
def test_reference_subset_is_bounded_seeded_and_ascending() -> None:
    dataset = TinyDataset(20)
    first = select_reference_subset(dataset, seed=123, max_samples=8)
    second = select_reference_subset(dataset, seed=123, max_samples=8)
    assert len(first) == 8
    assert list(first.indices) == list(second.indices)
    assert list(first.indices) == sorted(first.indices)
    assert len(set(first.indices)) == 8


def test_reference_subset_clamps_to_dataset_size() -> None:
    dataset = TinyDataset(5)
    subset = select_reference_subset(dataset, seed=1, max_samples=100)
    assert len(subset) == 5
    assert sorted(subset.indices) == [0, 1, 2, 3, 4]


def test_reference_seed_changes_selection() -> None:
    dataset = TinyDataset(64)
    a = select_reference_subset(dataset, seed=1, max_samples=16)
    b = select_reference_subset(dataset, seed=2, max_samples=16)
    assert list(a.indices) != list(b.indices)


def test_reference_subset_rejects_empty_dataset() -> None:
    with pytest.raises(EvaluatorError, match="empty"):
        select_reference_subset(TinyDataset(0), seed=1, max_samples=8)


@pytest.mark.parametrize("bound", [0, -1])
def test_reference_subset_rejects_non_positive_bound(bound: int) -> None:
    with pytest.raises(EvaluatorError, match="positive"):
        select_reference_subset(TinyDataset(4), seed=1, max_samples=bound)


# --------------------------------------------------------------------------- #
# Deterministic finite metrics + sample accounting
# --------------------------------------------------------------------------- #
def test_evaluator_metrics_are_deterministic_and_finite() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    first = _evaluator(config, topology).evaluate_round(results[0])
    second = _evaluator(config, topology).evaluate_round(results[0])
    assert isinstance(first, EvaluatedRound)
    assert first.node_ids == NODES3
    for a, b in zip(first.nodes, second.nodes):
        assert a.loss == b.loss
        assert a.accuracy == b.accuracy
        assert math.isfinite(a.loss)
        assert 0.0 <= a.accuracy <= 1.0


def test_evaluator_scores_all_nodes_on_the_same_reference_subset() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    evaluator = _evaluator(config, topology, reference=TinyDataset(20))
    record = evaluator.evaluate_round(results[0])
    assert evaluator.reference_sample_count == 8
    assert {node.sample_count for node in record.nodes} == {8}
    assert evaluator.history == (record,)


def test_evaluator_does_not_mutate_committed_states() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    before = {
        node: {key: value.clone() for key, value in state.items()}
        for node, state in results[0].states.items()
    }
    _evaluator(config, topology).evaluate_round(results[0])
    for node, state in results[0].states.items():
        for key, value in state.items():
            assert torch.equal(value, before[node][key])


# --------------------------------------------------------------------------- #
# True types, fixed attack actions, selected profiles
# --------------------------------------------------------------------------- #
def test_honest_episode_reports_benign_none() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    record = _evaluator(config, topology).evaluate_round(results[0])
    for node in record.nodes:
        assert node.true_type == BENIGN
        assert node.attack == AttackAction(kind=ATTACK_NONE, kappa=0.0)


def test_strong_attacker_reports_malicious_and_reversal() -> None:
    config = _plain(
        rounds=1, attack={"compromised_node": 0, "intensity": "strong"}
    )
    topology, results = _drive_rounds(config, 1)
    record = _evaluator(config, topology).evaluate_round(results[0])
    attacker = _find(record, 0)
    assert attacker.true_type == MALICIOUS
    assert attacker.attack.kind == ATTACK_DELTA_REVERSAL
    assert attacker.attack.kappa == 5.0
    for node in (1, 2):
        assert _find(record, node).true_type == BENIGN
        assert _find(record, node).attack.kind == ATTACK_NONE


def test_selected_node_with_zero_intensity_is_malicious_but_no_op() -> None:
    config = _plain(
        rounds=1, attack={"compromised_node": 1, "intensity": "zero"}
    )
    topology, results = _drive_rounds(config, 1)
    record = _evaluator(config, topology).evaluate_round(results[0])
    selected = _find(record, 1)
    assert selected.true_type == MALICIOUS
    assert selected.attack.kind == ATTACK_NONE
    assert selected.attack.kappa == 0.0


def test_selected_profiles_are_carried_for_every_node() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    record = _evaluator(config, topology).evaluate_round(results[0])
    node1 = _find(record, 1)
    assert node1.participant_ids == (1, 0, 2)
    assert node1.weights == pytest.approx((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    assert node1.multipliers == (1.0, 1.0, 1.0)
    assert _find(record, 0).participant_ids == (0, 1)


# --------------------------------------------------------------------------- #
# D018 weight-aware multi-hop propagation
# --------------------------------------------------------------------------- #
def test_plain_average_poison_reaches_hop_two_on_chain3() -> None:
    config = _plain(
        rounds=2, attack={"compromised_node": 0, "intensity": "strong"}
    )
    topology, results = _drive_rounds(config, 2)
    evaluator = _evaluator(config, topology)
    first = evaluator.evaluate_round(results[0])
    second = evaluator.evaluate_round(results[1])

    # Round 0: seed at hop 0, adjacent benign node reached at hop 1, the
    # non-adjacent node is not reached yet.
    assert _find(first, 0).propagation == PropagationState(True, 0, 0)
    assert _find(first, 1).propagation == PropagationState(True, 1, 0)
    assert _find(first, 2).propagation == PropagationState(False, None, None)

    # Round 1: influence crosses the second positive aggregation edge.
    assert _find(second, 2).propagation == PropagationState(True, 2, 1)
    # Retained minimum hop for the already-reached hop-1 node.
    assert _find(second, 1).propagation == PropagationState(True, 1, 0)


def test_local_only_prevents_poison_propagation() -> None:
    config = make_config(
        topology={"kind": "chain3"},
        aggregation={"kind": "local_only"},
        evaluation={"max_samples": 8, "batch_size": 4},
        rounds=2,
        attack={"compromised_node": 0, "intensity": "strong"},
    )
    topology, results = _drive_rounds(config, 2)
    evaluator = _evaluator(config, topology)
    first = evaluator.evaluate_round(results[0])
    second = evaluator.evaluate_round(results[1])

    # The active attacker seeds hop 0 regardless of aggregation, but zero
    # neighbor weights never propagate to the benign nodes.
    for record in (first, second):
        assert _find(record, 0).propagation == PropagationState(True, 0, 0)
        for node in (1, 2):
            assert _find(record, node).propagation == PropagationState(
                False, None, None
            )


def test_zero_kappa_attacker_never_seeds() -> None:
    config = _plain(
        rounds=1, attack={"compromised_node": 0, "intensity": "zero"}
    )
    topology, results = _drive_rounds(config, 1)
    record = _evaluator(config, topology).evaluate_round(results[0])
    for node in record.nodes:
        assert node.propagation == PropagationState(False, None, None)


def test_no_attacker_means_no_propagation() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    tracker = ProvenanceTracker.from_experiment(config, topology)
    assert tracker.active is False
    states = tracker.update(results[0])
    assert all(state.reached is False for state in states)


def test_tracker_does_not_advance_on_out_of_order_round() -> None:
    config = _plain(
        rounds=2, attack={"compromised_node": 0, "intensity": "strong"}
    )
    topology, results = _drive_rounds(config, 2)
    tracker = ProvenanceTracker.from_experiment(config, topology)
    tracker.update(results[0])
    before = tracker.propagation()
    with pytest.raises(EvaluatorError, match="out of order"):
        tracker.update(results[0])
    assert tracker.propagation() == before
    assert tracker.next_round_index == 1


def test_evaluator_rejects_skipped_first_round() -> None:
    config = _plain(
        rounds=2, attack={"compromised_node": 0, "intensity": "strong"}
    )
    topology, results = _drive_rounds(config, 2)
    evaluator = _evaluator(config, topology)
    with pytest.raises(EvaluatorError, match="out of order"):
        evaluator.evaluate_round(results[1])


def test_evaluator_does_not_advance_tracker_when_metrics_fail() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    evaluator = _evaluator(
        config, topology, reference=TinyDataset(8, fill=float("nan"))
    )
    with pytest.raises(EvaluatorError):
        evaluator.evaluate_round(results[0])
    assert evaluator.next_round_index == 0
    assert evaluator.history == ()


# --------------------------------------------------------------------------- #
# Q7 negative surface: malformed rounds and data
# --------------------------------------------------------------------------- #
def test_evaluator_rejects_non_finite_input() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    evaluator = _evaluator(
        config, topology, reference=TinyDataset(8, fill=float("nan"))
    )
    with pytest.raises(EvaluatorError, match="non-finite evaluation input"):
        evaluator.evaluate_round(results[0])


def test_evaluator_rejects_out_of_range_labels() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    evaluator = _evaluator(
        config,
        topology,
        reference=TinyDataset(8, num_classes=2, label_offset=2),
    )
    with pytest.raises(EvaluatorError, match="out of range"):
        evaluator.evaluate_round(results[0])


def test_evaluator_rejects_empty_reference_dataset() -> None:
    config = _plain(rounds=1)
    topology, _ = _drive_rounds(config, 1)
    with pytest.raises(EvaluatorError, match="empty"):
        _evaluator(config, topology, reference=TinyDataset(0))


def test_evaluator_rejects_non_finite_committed_state() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    states = dict(results[0].states)
    node_state = dict(states[0])
    key = sorted(node_state)[0]
    node_state[key] = torch.full_like(node_state[key], float("nan"))
    states[0] = node_state
    corrupted = dataclasses.replace(results[0], states=states)
    with pytest.raises(EvaluatorError, match="invalid committed model state"):
        _evaluator(config, topology).evaluate_round(corrupted)


def test_validate_round_structure_rejects_missing_node() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    states = {node: state for node, state in results[0].states.items() if node != 2}
    corrupted = dataclasses.replace(results[0], states=states)
    with pytest.raises(EvaluatorError, match="node set"):
        validate_round_structure(topology, corrupted)


def test_validate_round_structure_rejects_foreign_profile() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    profiles = list(results[0].profiles)
    target = next(index for index, p in enumerate(profiles) if p.node_id == 0)
    bad = dataclasses.replace(
        profiles[target],
        participant_ids=(0, 99),
        multipliers=(1.0, 1.0),
        weights=(0.5, 0.5),
    )
    profiles[target] = bad
    corrupted = dataclasses.replace(results[0], profiles=tuple(profiles))
    with pytest.raises(EvaluatorError, match="participants"):
        validate_round_structure(topology, corrupted)


def test_validate_round_structure_rejects_nan_weight() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    profiles = list(results[0].profiles)
    target = next(index for index, p in enumerate(profiles) if p.node_id == 0)
    original = profiles[target]
    bad = dataclasses.replace(
        original, weights=(float("nan"),) + tuple(original.weights[1:])
    )
    profiles[target] = bad
    corrupted = dataclasses.replace(results[0], profiles=tuple(profiles))
    with pytest.raises(EvaluatorError, match="finite"):
        validate_round_structure(topology, corrupted)


def test_validate_round_structure_rejects_wrong_recipients() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    messages = list(results[0].messages)
    target = next(index for index, m in enumerate(messages) if m.sender == 0)
    bad = dataclasses.replace(messages[target], recipients=(0,))
    messages[target] = bad
    corrupted = dataclasses.replace(results[0], messages=tuple(messages))
    with pytest.raises(EvaluatorError, match="recipients"):
        validate_round_structure(topology, corrupted)


def test_validate_round_structure_rejects_duplicate_profile() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    profiles = list(results[0].profiles)
    profiles.append(profiles[0])
    corrupted = dataclasses.replace(results[0], profiles=tuple(profiles))
    with pytest.raises(EvaluatorError, match="duplicate profile"):
        validate_round_structure(topology, corrupted)


# --------------------------------------------------------------------------- #
# Immutable schema records and input validation
# --------------------------------------------------------------------------- #
def test_records_are_immutable_and_canonically_ordered() -> None:
    config = _plain(rounds=1)
    topology, results = _drive_rounds(config, 1)
    record = _evaluator(config, topology).evaluate_round(results[0])
    assert list(record.node_ids) == sorted(record.node_ids)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.nodes[0].loss = 9.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.round_index = 5  # type: ignore[misc]
    with pytest.raises(EvaluatorError):
        EvaluatedRound(round_index=0, nodes=(record.nodes[1], record.nodes[0]))
    with pytest.raises(EvaluatorError):
        EvaluatedRound(round_index=0, nodes=())


@pytest.mark.parametrize(
    "kind, kappa",
    [
        ("bogus", 1.0),
        (ATTACK_NONE, 1.0),
        (ATTACK_DELTA_REVERSAL, 0.0),
        (ATTACK_NONE, float("nan")),
        (ATTACK_NONE, -1.0),
        (ATTACK_NONE, float("inf")),
    ],
)
def test_attack_action_rejects_invalid_combinations(kind: str, kappa: float) -> None:
    # Construction itself validates; every malformed combination must fail
    # closed rather than producing a silently-malformed action record.
    with pytest.raises(EvaluatorError):
        AttackAction(kind=kind, kappa=kappa)


def test_valid_attack_actions() -> None:
    assert AttackAction(kind=ATTACK_NONE, kappa=0.0).kappa == 0.0
    assert AttackAction(kind=ATTACK_DELTA_REVERSAL, kappa=5.0).kappa == 5.0


@pytest.mark.parametrize(
    "reached, min_hop, first_reached_round",
    [
        (True, None, None),
        (True, 1, None),
        (True, None, 0),
        (True, -1, 0),
        (False, 0, 0),
        (False, None, -1),
    ],
)
def test_propagation_state_rejects_incoherent_values(
    reached: bool, min_hop: int | None, first_reached_round: int | None
) -> None:
    with pytest.raises(EvaluatorError):
        PropagationState(
            reached=reached,
            min_hop=min_hop,
            first_reached_round=first_reached_round,
        )


def test_valid_propagation_states() -> None:
    assert PropagationState(True, 2, 1).min_hop == 2
    assert PropagationState(False, None, None).reached is False


def test_tracker_rejects_compromised_node_outside_topology() -> None:
    config = _plain(rounds=1)
    topology = load_topology(config)
    with pytest.raises(EvaluatorError, match="not in topology"):
        ProvenanceTracker(topology=topology, compromised_node=9, kappa=1.0)


def test_evaluator_from_convenience_builder_loads_topology() -> None:
    config = _plain(rounds=1)
    topology = load_topology(config)
    _, results = _drive_rounds(config, 1)
    evaluator = build_evaluator(
        config, TinyDataset(16), resolve_seeds(config), model_factory=TinyNet
    )
    assert evaluator.node_ids == NODES3
    record = evaluator.evaluate_round(results[0])
    assert record.node_ids == NODES3
    assert topology.number_of_nodes() == 3


# --------------------------------------------------------------------------- #
# Runtime boundary: privileged records never reach a policy observation
# --------------------------------------------------------------------------- #
class _ObsRecorder:
    """Record policy observations and delegate to plain averaging."""

    def __init__(self) -> None:
        self._inner = PlainAveragePolicy()
        self.seen: list = []

    def select_weights(self, observation):
        self.seen.append(observation)
        return self._inner.select_weights(observation)


def test_active_evaluator_records_truth_without_reaching_policy_observations() -> None:
    config = _plain(
        rounds=1, attack={"compromised_node": 0, "intensity": "strong"}
    )
    topology = load_topology(config)
    nodes = tuple(sorted(int(node) for node in topology.nodes()))
    recorder = _ObsRecorder()
    result = run_round(
        config=config,
        topology=topology,
        snapshots={node: _tiny_state() for node in nodes},
        partitions={node: None for node in nodes},
        round_index=0,
        seeds=resolve_seeds(config),
        trainer=_tiny_trainer,
        loss_evaluator=constant_loss_evaluator_factory,
        policy=recorder,
    )
    record = _evaluator(config, topology).evaluate_round(result)
    # The privileged evaluator exposes the true malicious identity...
    assert _find(record, 0).true_type == MALICIOUS
    # ...while every policy observation stays scalar-only and privilege-free.
    assert len(recorder.seen) == 3
    for observation in recorder.seen:
        assert type(observation).__module__ == "dfl_sim.observation"
        assert observation.participant_ids[0] == observation.node_id
        for forbidden in (
            "true_type",
            "attack",
            "propagation",
            "loss",
            "accuracy",
            "evaluator",
        ):
            assert not hasattr(observation, forbidden)
