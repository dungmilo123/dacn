"""End-to-end chain propagation acceptance over the production runner (T04).

Offline, deterministic integration tests that drive :func:`run_from_config`
with injected in-memory fixtures and read the published schema-v1 trajectory.
They prove the slice demo and validate R008/R009/R010/R012:

* a strong (``kappa == 5.0``) delta-reversal poison seeded at compromised node
  0 traverses two positive-weight aggregation edges on ``chain3`` and reaches
  the non-adjacent benign node 2 only at the second synchronous round with
  minimum hop 2 under ``plain_average``;
* that reach genuinely degrades node 2's shared-reference loss versus the
  honest/plain control while remaining bitwise-different at the state level;
* ``local_only`` blocks propagation entirely: node 1 and node 2 remain exactly
  (bitwise) equal to the honest/local-only control, so no poison leaks past the
  compromised sender;
* the ``kappa == 0`` path is equivalent to the honest control;
* every message stays on a configured edge and every provenance transition is
  explained by the round's selected participant weights.
"""

from __future__ import annotations

import pytest
import torch

from dfl_sim import TraceReader, run_from_config
from helpers import MNISTShapedDataset, make_config

CHAIN_NODES = (0, 1, 2)
CHAIN_EDGES = frozenset({(0, 1), (1, 2)})
COMPROMISED = 0

HONEST = {"compromised_node": None, "intensity": "zero"}
STRONG = {"compromised_node": COMPROMISED, "intensity": "strong"}
ZERO = {"compromised_node": COMPROMISED, "intensity": "zero"}

# name -> (attack block, aggregation kind)
_VARIANTS = {
    "honest_plain": (HONEST, "plain_average"),
    "strong_plain": (STRONG, "plain_average"),
    "zero_plain": (ZERO, "plain_average"),
    "honest_local": (HONEST, "local_only"),
    "strong_local": (STRONG, "local_only"),
}


def _config(attack: dict, aggregation: str):
    return make_config(
        rounds=2,
        topology={"kind": "chain3"},
        attack=attack,
        aggregation={"kind": aggregation},
        data={"samples_per_node": 32},
        evaluation={"max_samples": 32, "batch_size": 16},
        trace={"snapshot_cadence": 1},
    )


@pytest.fixture(scope="module")
def episodes(tmp_path_factory):
    """Run every variant once through the production runner and read its trace."""

    root = tmp_path_factory.mktemp("propagation")
    train = MNISTShapedDataset(160, seed=5, separable=True)
    reference = MNISTShapedDataset(96, seed=9, separable=True)

    out: dict[str, dict] = {}
    for name, (attack, aggregation) in _VARIANTS.items():
        destination = root / name
        result = run_from_config(
            _config(attack, aggregation),
            dataset=train,
            evaluation_dataset=reference,
            trace_path=destination,
        )
        with TraceReader(destination) as reader:
            events = tuple(reader.events)
            manifest = dict(reader.manifest)
        out[name] = {
            "result": result,
            "events": events,
            "manifest": manifest,
        }
    return out


def _node(event: dict, node_id: int) -> dict:
    return next(item for item in event["nodes"] if item["node"] == node_id)


def _loss(episode: dict, node_id: int) -> float:
    return float(_node(episode["events"][-1], node_id)["loss"])


def _final_state(episode: dict, node_id: int):
    return episode["result"].final_states[node_id]


def _state_equal(left, right) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[key], right[key]) for key in left
    )


def _all_unreached(episode: dict) -> bool:
    return all(
        not item["propagation"]["reached"]
        for event in episode["events"]
        for item in event["nodes"]
    )


def _nodes_unreached(episode: dict, node_ids) -> bool:
    return all(
        not _node(event, node_id)["propagation"]["reached"]
        for event in episode["events"]
        for node_id in node_ids
    )


# --------------------------------------------------------------------------- #
# Two-hop propagation under plain averaging
# --------------------------------------------------------------------------- #
def test_strong_plain_reaches_non_adjacent_node_at_hop_two(episodes) -> None:
    episode = episodes["strong_plain"]
    events = episode["events"]
    assert [event["round"] for event in events] == [0, 1]

    attacker = _node(events[0], COMPROMISED)
    assert attacker["true_type"] == "malicious"
    assert attacker["attack"]["kind"] == "delta_reversal"
    assert attacker["attack"]["kappa"] == pytest.approx(5.0)

    # Node 0 seeds influence at round 0, hop 0.
    for event in events:
        propagation = _node(event, COMPROMISED)["propagation"]
        assert propagation == {
            "reached": True,
            "min_hop": 0,
            "first_reached_round": 0,
        }

    # Node 1 is adjacent: reached at hop 1 in the first round.
    first_hop = _node(events[0], 1)["propagation"]
    assert first_hop == {
        "reached": True,
        "min_hop": 1,
        "first_reached_round": 0,
    }

    # Node 2 is *not* adjacent: it must stay unreached after round 0 and can
    # only be reached through node 1 in the second synchronous round (Q7).
    assert _node(events[0], 2)["propagation"]["reached"] is False
    second_hop = _node(events[1], 2)["propagation"]
    assert second_hop == {
        "reached": True,
        "min_hop": 2,
        "first_reached_round": 1,
    }
    assert _node(events[1], 2)["true_type"] == "benign"


def test_strong_plain_degrades_final_loss_versus_honest(episodes) -> None:
    strong = episodes["strong_plain"]
    honest = episodes["honest_plain"]

    assert _all_unreached(honest)
    assert _loss(strong, 2) > _loss(honest, 2)
    # The poisoned state is genuinely different from the honest control.
    assert not _state_equal(_final_state(strong, 2), _final_state(honest, 2))


# --------------------------------------------------------------------------- #
# Local-only isolation
# --------------------------------------------------------------------------- #
def test_local_only_blocks_propagation_and_matches_honest(episodes) -> None:
    strong = episodes["strong_local"]
    honest = episodes["honest_local"]

    # The active attacker is always a hop-0 seed, but no benign node may be
    # reached under local-only aggregation.
    for event in strong["events"]:
        assert _node(event, COMPROMISED)["propagation"] == {
            "reached": True,
            "min_hop": 0,
            "first_reached_round": 0,
        }
    assert _nodes_unreached(strong, (1, 2))
    assert _all_unreached(honest)

    for node_id in CHAIN_NODES:
        strong_node = _final_state(strong, node_id)
        honest_node = _final_state(honest, node_id)
        assert _state_equal(strong_node, honest_node), (
            f"node {node_id} state diverged under local_only"
        )

    # Metrics are exactly equal for every node; benign propagation is identical
    # (unreached everywhere), proving the poison never leaked past node 0.
    for strong_event, honest_event in zip(strong["events"], honest["events"]):
        assert strong_event["round"] == honest_event["round"]
        for node_id in CHAIN_NODES:
            strong_node = _node(strong_event, node_id)
            honest_node = _node(honest_event, node_id)
            assert strong_node["loss"] == honest_node["loss"]
            assert strong_node["accuracy"] == honest_node["accuracy"]
            assert strong_node["sample_count"] == honest_node["sample_count"]
        for node_id in (1, 2):
            assert (
                _node(strong_event, node_id)["propagation"]
                == _node(honest_event, node_id)["propagation"]
            )


# --------------------------------------------------------------------------- #
# kappa-zero equivalence
# --------------------------------------------------------------------------- #
def test_zero_intensity_is_equivalent_to_honest(episodes) -> None:
    zero = episodes["zero_plain"]
    honest = episodes["honest_plain"]

    assert _all_unreached(zero)
    for node_id in CHAIN_NODES:
        assert _state_equal(_final_state(zero, node_id), _final_state(honest, node_id))
        assert _loss(zero, node_id) == _loss(honest, node_id)


# --------------------------------------------------------------------------- #
# Edge-only messages and profile-explained transitions
# --------------------------------------------------------------------------- #
def _assert_messages_are_edge_only(events) -> None:
    for event in events:
        for message in event["messages"]:
            sender = message["sender"]
            recipients = message["recipients"]
            assert recipients, f"round {event['round']} sender {sender} has no recipients"
            assert len(set(recipients)) == len(recipients)
            for recipient in recipients:
                assert recipient != sender
                assert (min(sender, recipient), max(sender, recipient)) in CHAIN_EDGES


def _assert_profiles_explain_transitions(events, attacker: int) -> None:
    """Every reached node's min-hop must be justified by a positive-weight edge.

    The active attacker is always an outgoing hop-0 seed (even in round 0,
    before any committed state records it), so it can explain hop 1 exactly.
    """

    prior = {node: {"reached": False, "min_hop": None} for node in CHAIN_NODES}
    for event in events:
        current: dict[int, dict] = {}
        for item in event["nodes"]:
            node_id = item["node"]
            propagation = item["propagation"]
            if not propagation["reached"]:
                current[node_id] = {"reached": False, "min_hop": None}
                continue

            hop = propagation["min_hop"]
            profile = item["profile"]
            explained = False

            # The compromised sender seeds influence at hop 0 every round.
            if node_id == attacker and hop == 0:
                explained = True
            # Monotone local training preserves a node's prior minimal hop.
            if prior[node_id]["reached"] and prior[node_id]["min_hop"] == hop:
                explained = True

            for participant, weight in zip(
                profile["participants"], profile["weights"]
            ):
                if participant == node_id or weight <= 0.0:
                    continue
                if participant == attacker:
                    if hop == 1:  # seeded hop 0 -> +1 edge
                        explained = True
                elif (
                    prior[participant]["reached"]
                    and prior[participant]["min_hop"] == hop - 1
                ):
                    explained = True

            assert explained, (
                f"round {event['round']} node {node_id} reached hop {hop} "
                f"but no positive-weight predecessor explains it "
                f"(prior={prior}, profile={profile})"
            )
            current[node_id] = {"reached": True, "min_hop": hop}
        prior = current


def test_messages_stay_on_edges_and_profiles_explain_transitions(episodes) -> None:
    for name in ("honest_plain", "strong_plain", "honest_local", "strong_local"):
        _assert_messages_are_edge_only(episodes[name]["events"])
    _assert_profiles_explain_transitions(episodes["strong_plain"]["events"], COMPROMISED)
    _assert_profiles_explain_transitions(episodes["honest_plain"]["events"], COMPROMISED)
