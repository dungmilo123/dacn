"""Episode-level coverage closing the M001 round-1 validation gaps.

The round-0/round-1 milestone validation returned NEEDS-ATTENTION because
success criterion C1 (a clean episode runs deterministically on the 2-node
line and a 10-20 node IID graph and local training reduces loss) had evidence
only for the 3-node chain, and the Integration planned check had no test that
consumed an *episode-emitted* schema-v1 trace with the viewer.

These tests run the production :func:`~dfl_sim.run.run_from_config` orchestrator
with injected in-process fixtures (no network), assert two independent reruns
are byte-identical and that first-round local loss decreases at every node, and
drive the real ``dfl run`` -> ``dfl view`` CLI chain so the viewer's strict
``TraceReader`` projection consumes the trace the episode actually emitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import dash
import networkx as nx
import pytest

import dfl_sim.run as run_module
from dfl_sim import load_topology, run_from_config
from dfl_sim.__main__ import main
from dfl_sim.viewer import load_replay
from helpers import MNISTShapedDataset, make_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_CONFIG = REPO_ROOT / "configs" / "clean_3node.yaml"
POISON_CONFIG = REPO_ROOT / "configs" / "fixed_poison_3node.yaml"


def _train(num_examples: int = 400, *, seed: int = 7) -> MNISTShapedDataset:
    """Separable MNIST-shaped fixture so local training can genuinely learn."""

    return MNISTShapedDataset(num_examples, seed=seed, separable=True)


def _evaluation(num_examples: int = 64, *, seed: int = 11) -> MNISTShapedDataset:
    return MNISTShapedDataset(num_examples, seed=seed)


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_first_round_loss_decreases(result) -> None:
    first = result.rounds[0]
    assert first.stats, "first round produced no node stats"
    for stat in first.stats:
        assert stat.post_loss < stat.pre_loss, (
            f"node {stat.node_id} did not reduce loss: "
            f"pre={stat.pre_loss} post={stat.post_loss}"
        )


# --------------------------------------------------------------------------- #
# C1: deterministic clean episodes with loss reduction on line2 and WS 10-20
# --------------------------------------------------------------------------- #
def test_line2_clean_episode_is_deterministic_and_reduces_loss(tmp_path: Path) -> None:
    config = make_config(topology={"kind": "line2"}, rounds=2)
    dataset = _train()
    log_a = tmp_path / "line2_a.jsonl"
    log_b = tmp_path / "line2_b.jsonl"

    result_a = run_from_config(config, log_path=log_a, dataset=dataset)
    result_b = run_from_config(config, log_path=log_b, dataset=dataset)

    # 2-node line, fixed per episode.
    assert result_a.node_ids == (0, 1)
    assert tuple(sorted(load_topology(config).edges())) == ((0, 1),)
    # Bitwise-identical rerun of the canonical artifact.
    assert log_a.read_bytes() == log_b.read_bytes()
    assert len(_read_records(log_a)) == 2
    # Local training reduces loss at every node in the first round.
    _assert_first_round_loss_decreases(result_a)


@pytest.mark.parametrize("n", [10, 20])
def test_watts_strogatz_clean_episode_is_deterministic_and_reduces_loss(
    tmp_path: Path, n: int
) -> None:
    config = make_config(
        topology={"kind": "watts_strogatz", "n": n, "k": 4, "p": 0.2},
        rounds=2,
    )
    topology = load_topology(config)
    assert topology.number_of_nodes() == n
    assert nx.is_connected(topology)

    dataset = _train(num_examples=n * 64)
    log_a = tmp_path / f"ws{n}_a.jsonl"
    log_b = tmp_path / f"ws{n}_b.jsonl"

    result_a = run_from_config(config, log_path=log_a, dataset=dataset)
    result_b = run_from_config(config, log_path=log_b, dataset=dataset)

    assert result_a.node_ids == tuple(range(n))
    assert log_a.read_bytes() == log_b.read_bytes()
    assert len(_read_records(log_a)) == 2
    _assert_first_round_loss_decreases(result_a)


# --------------------------------------------------------------------------- #
# Integration: an episode-emitted trace is consumed through the dfl view CLI
# --------------------------------------------------------------------------- #
def test_run_cli_emits_trace_consumed_by_view_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patch the production loaders with offline fixtures so the real `dfl run`
    # command publishes a complete trace without touching the network.
    monkeypatch.setattr(
        run_module, "load_mnist_dataset", lambda config, **kwargs: _train()
    )
    monkeypatch.setattr(
        run_module,
        "load_mnist_evaluation_dataset",
        lambda config, **kwargs: _evaluation(),
    )

    output = tmp_path / "episode_trace"
    # fixed_poison_3node => node 0 is malicious, so the viewer must classify it.
    assert main(["run", str(POISON_CONFIG), "--output", str(output)]) == 0
    assert (output / "manifest.json").is_file()

    # The viewer projection must read the episode-emitted trace through its
    # strict TraceReader (no hand-built write_trace fixture).
    model = load_replay(output)
    assert model.node_ids == (0, 1, 2)
    assert model.rounds == (0, 1)
    assert model.node_types == {0: "malicious", 1: "benign", 2: "benign"}
    assert model.compromised_node == 0
    assert model.edges == ((0, 1), (1, 2))
    for frame in model.frames:
        assert frame.message_count > 0
        assert frame.node_ids == (0, 1, 2)
        assert all(state.loss > 0.0 for state in frame.nodes)
        assert all(0.0 <= state.accuracy <= 1.0 for state in frame.nodes)

    # The real `dfl view` command must serve (loopback only, debug off) the
    # episode-emitted trace; only the final server run boundary is stubbed.
    recorded: list[dict] = []
    monkeypatch.setattr(dash.Dash, "run", lambda self, **kwargs: recorded.append(kwargs))
    assert main(["view", str(output), "--port", "8123"]) == 0
    assert recorded == [
        {"host": "127.0.0.1", "port": 8123, "debug": False, "use_reloader": False}
    ]
