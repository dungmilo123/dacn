"""End-to-end baseline acceptance tests for S03 (R005/R006/R007, D008/D017).

These tests drive the **production** config -> policy -> observe -> weighted-
aggregate seam for all three M001 baselines (local-only, plain-average, fixed
directional-similarity) over one *controlled* chain episode, then through the
full :func:`~dfl_sim.run.run_from_config` episode path for deterministic
repeatability.

Everything is offline and in-process: no files (other than the git-tracked
production sources under ``src/``), no network, and no local MNIST download.
State dicts are tiny scalar ``{"w": tensor([x])}`` payloads and the loss seam is
stubbed, so every observation feature, selected profile, and committed tensor is
hand-computable.

Coverage:

* the three baselines produce three distinct, participant-aligned profiles over
  the real config/round seam, and each committed state equals the weighted
  combination represented by its recorded profile;
* empty/isolated inboxes and a fixed filter that rejects every neighbor both
  commit the locally trained self state exactly;
* each baseline is bitwise-deterministic across two full episodes;
* a selected-node ``zero`` intensity is identical to honest execution for every
  baseline, and a strong attack preserves edge-only transport while propagating
  through the real round (the S02 kappa-zero/strong-attack invariants).
"""

from __future__ import annotations

import networkx as nx
import torch

from dfl_sim import (
    LocalTrainResult,
    MessageSummary,
    NodeWeightProfile,
    RoundResult,
    aggregate_weighted_neighborhood,
    load_topology,
    resolve_seeds,
    run_from_config,
    run_round,
)
from helpers import (
    MNISTShapedDataset,
    constant_loss_evaluator_factory,
    make_config,
)

NODES3 = (0, 1, 2)
BASELINE_KINDS = ("local_only", "plain_average", "fixed_similarity")


# --------------------------------------------------------------------------- #
# Controlled episode helpers
# --------------------------------------------------------------------------- #
def _aggregation(kind: str) -> dict:
    if kind == "fixed_similarity":
        return {
            "kind": "fixed_similarity",
            "cutoff": 0.0,
            "downweight_multiplier": 0.25,
        }
    return {"kind": kind}


def _baseline_config(kind: str, *, attack: dict | None = None):
    overrides: dict = {
        "topology": {"kind": "chain3"},
        "aggregation": _aggregation(kind),
    }
    if attack is not None:
        overrides["attack"] = attack
    return make_config(**overrides)


def _offset_trainer(scale: float = 10.0):
    """Stub trainer: trained state = snapshot + scale * node_id.

    With zero snapshots this yields trained values ``0, 10, 20`` and therefore
    deterministic, hand-computable directional disagreements on chain3.
    """

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


def _signed_trainer(scale: float = 10.0):
    """Stub trainer: alternating sign per node so every edge is opposed.

    Every neighbor is directionally opposite its recipient, so a fixed filter
    at ``cutoff=0.0`` rejects all of them and each node commits its own state.
    """

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
        sign = 1.0 if node_id % 2 == 0 else -1.0
        state = {
            key: value.detach().clone() + sign * scale
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


def _controlled_round(
    kind: str,
    *,
    trainer=None,
    topology: nx.Graph | None = None,
    attack: dict | None = None,
) -> RoundResult:
    """Run one controlled chain round with the config-selected policy."""

    config = _baseline_config(kind, attack=attack)
    active_topology = topology if topology is not None else load_topology(config)
    snapshots = {node: {"w": torch.tensor([0.0])} for node in NODES3}
    return run_round(
        config=config,
        topology=active_topology,
        snapshots=snapshots,
        partitions={node: None for node in NODES3},
        round_index=0,
        seeds=resolve_seeds(config),
        trainer=trainer if trainer is not None else _offset_trainer(),
        loss_evaluator=constant_loss_evaluator_factory,
    )


def _profile_map(result: RoundResult) -> dict[int, NodeWeightProfile]:
    return {profile.node_id: profile for profile in result.profiles}


def _expected_from_profile(profile: NodeWeightProfile):
    """Rebuild the committed state implied by a recorded profile.

    Under the honest offset trainer a sender ``s`` delivers ``10 * s``; the
    observing node's own trained state is ``10 * node``. Both are recoverable
    from ``participant_ids`` (self first, then senders ascending).
    """

    states = [
        {"w": torch.tensor([10.0 * participant])}
        for participant in profile.participant_ids
    ]
    return aggregate_weighted_neighborhood(
        states[0], states[1:], profile.weights
    )


def _production_episode(kind: str):
    """Full production episode with real local training on a tiny fixture."""

    config = make_config(
        rounds=1,
        data={"samples_per_node": 16},
        training={"local_epochs": 1, "batch_size": 8},
        aggregation=_aggregation(kind),
    )
    dataset = MNISTShapedDataset(48, seed=11, separable=True)
    return run_from_config(config, dataset=dataset)


# --------------------------------------------------------------------------- #
# Three baselines over the real config/round seam
# --------------------------------------------------------------------------- #
def test_three_baselines_produce_distinct_profiles_end_to_end() -> None:
    local = _profile_map(_controlled_round("local_only"))
    plain = _profile_map(_controlled_round("plain_average"))
    fixed = _profile_map(_controlled_round("fixed_similarity"))

    # Node 1 has two delivered senders, so its profile is the discriminating one:
    # local-only rejects both, plain-average keeps both, and the fixed filter
    # keeps exactly one (a controlled accept/reject mixture).
    assert local[1].participant_ids == (1, 0, 2)
    assert local[1].multipliers == (1.0, 0.0, 0.0)
    assert plain[1].multipliers == (1.0, 1.0, 1.0)
    assert fixed[1].multipliers == (1.0, 0.0, 1.0)

    assert local[1] != plain[1]
    assert plain[1] != fixed[1]
    assert local[1] != fixed[1]

    # All three profiles stay participant-aligned and unit-sum.
    for profile in (*local.values(), *plain.values(), *fixed.values()):
        assert profile.participant_ids[0] == profile.node_id
        assert abs(sum(profile.weights) - 1.0) <= 1e-9


def test_committed_states_match_selected_profiles() -> None:
    for kind in BASELINE_KINDS:
        result = _controlled_round(kind)
        assert [p.node_id for p in result.profiles] == list(NODES3)
        for profile in result.profiles:
            expected = _expected_from_profile(profile)
            assert torch.equal(result.states[profile.node_id]["w"], expected["w"]), (
                f"{kind} node {profile.node_id}: committed state does not match "
                f"the weighted combination of the recorded profile"
            )


def test_baseline_committed_values_are_exact() -> None:
    local = _controlled_round("local_only")
    assert local.states[0]["w"].tolist() == [0.0]
    assert local.states[1]["w"].tolist() == [10.0]
    assert local.states[2]["w"].tolist() == [20.0]

    plain = _controlled_round("plain_average")
    assert plain.states[0]["w"].tolist() == [5.0]
    assert plain.states[1]["w"].tolist() == [10.0]
    assert plain.states[2]["w"].tolist() == [15.0]

    # fixed: node 0's own delta_ii is zero (trained == round-start), so its
    # only sender is rejected as directionally undefined -> exact self [0.0].
    # Node 1 rejects the opposed sender 0 and keeps sender 2 -> 15.
    fixed = _controlled_round("fixed_similarity")
    assert fixed.states[0]["w"].tolist() == [0.0]
    assert fixed.states[1]["w"].tolist() == [15.0]
    assert fixed.states[2]["w"].tolist() == [15.0]


# --------------------------------------------------------------------------- #
# Self-only fallback: empty inbox and all-rejected neighbors
# --------------------------------------------------------------------------- #
def test_empty_inbox_commits_exact_self_state_for_every_baseline() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(NODES3)
    graph.add_edge(1, 2)  # node 0 is isolated -> empty inbox
    topology = nx.freeze(graph)

    for kind in BASELINE_KINDS:
        result = _controlled_round(kind, topology=topology)
        profile = _profile_map(result)[0]
        assert profile.participant_ids == (0,)
        assert profile.weights == (1.0,)
        assert result.states[0]["w"].tolist() == [0.0]


def test_fixed_filter_rejecting_every_neighbor_commits_exact_self_state() -> None:
    result = _controlled_round("fixed_similarity", trainer=_signed_trainer())

    for profile in result.profiles:
        assert profile.weights[0] == 1.0
        assert all(weight == 0.0 for weight in profile.weights[1:])

    assert result.states[0]["w"].tolist() == [10.0]
    assert result.states[1]["w"].tolist() == [-10.0]
    assert result.states[2]["w"].tolist() == [10.0]


# --------------------------------------------------------------------------- #
# Deterministic repeatability under bitwise mode
# --------------------------------------------------------------------------- #
def _state_dicts_equal(left, right) -> bool:
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def test_each_baseline_is_deterministic_under_bitwise_mode() -> None:
    for kind in BASELINE_KINDS:
        first = _production_episode(kind)
        second = _production_episode(kind)

        assert first.config.bitwise is True
        assert len(first.rounds) == 1

        assert first.final_round.profiles == second.final_round.profiles
        for node in first.node_ids:
            assert _state_dicts_equal(
                first.final_states[node], second.final_states[node]
            ), f"{kind}: node {node} final tensor diverged across bitwise runs"


# --------------------------------------------------------------------------- #
# S02 attack invariants retained at the baseline seam
# --------------------------------------------------------------------------- #
def test_selected_node_zero_intensity_matches_honest_for_every_baseline() -> None:
    for kind in BASELINE_KINDS:
        honest = _controlled_round(kind)
        zero = _controlled_round(
            kind, attack={"compromised_node": 0, "intensity": "zero"}
        )
        assert zero.profiles == honest.profiles
        assert zero.messages == honest.messages
        for node in NODES3:
            assert torch.equal(zero.states[node]["w"], honest.states[node]["w"])


def test_strong_attack_transport_is_edge_only_and_propagates() -> None:
    expected_messages = (
        MessageSummary(sender=0, recipients=(1,)),
        MessageSummary(sender=1, recipients=(0, 2)),
        MessageSummary(sender=2, recipients=(1,)),
    )
    for kind in BASELINE_KINDS:
        honest = _controlled_round(kind)
        attacked = _controlled_round(
            kind, attack={"compromised_node": 1, "intensity": "strong"}
        )

        # Attack never adds or removes an edge; delivery stays topology-faithful.
        assert attacked.messages == expected_messages
        for profile in attacked.profiles:
            assert profile.participant_ids[0] == profile.node_id

        # Plain averaging always keeps every delivered sender, so the poisoned
        # sender's transported value necessarily changes a committed state.
        if kind == "plain_average":
            assert any(
                not torch.equal(attacked.states[node]["w"], honest.states[node]["w"])
                for node in NODES3
            )
