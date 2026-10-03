"""Offline end-to-end integration tests for T04.

These tests wire ``configs/clean_3node.yaml`` through the full episode runner
with an injected deterministic MNIST-shaped fixture, so nothing touches the
network. They prove the artifact boundary: two independent runs of the same
config and dataset produce **byte-identical** canonical JSONL, every configured
round/node/message is represented, first-round local loss decreases at every
node, and no log is ever published after a failure (while a prior complete log
survives untouched). They also cover the Q7 negative surface: config/log path
aliasing, invalid targets, mid-episode failures, NaN rejection, order
independence, and seed divergence.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dfl_sim import (
    EpisodeError,
    LogError,
    MessageSummary,
    NodeRoundStat,
    RoundResult,
    canonical_line,
    canonical_record,
    load_config,
    load_topology,
    local_train_from_snapshot,
    run_from_config,
    validate_log_path,
    write_round_log,
)
from helpers import MNISTShapedDataset, make_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "clean_3node.yaml"

# chain3 -> 0-1-2, so each node's sorted neighbor set is fixed.
CHAIN3_RECIPIENTS = {0: [1], 1: [0, 2], 2: [1]}


def _fixture(num_examples: int = 300, *, seed: int = 7) -> MNISTShapedDataset:
    return MNISTShapedDataset(num_examples, seed=seed, separable=True)


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #
def test_module_help_succeeds() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "dfl_sim", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--log" in proc.stdout
    assert "PATH" in proc.stdout


def test_cli_missing_config_returns_nonzero() -> None:
    from dfl_sim.__main__ import main

    status = main(["does-not-exist.yaml", "--log", "unused.jsonl"])
    assert status == 1


# --------------------------------------------------------------------------- #
# Deterministic artifact boundary
# --------------------------------------------------------------------------- #
def test_two_runs_are_byte_identical_and_canonical(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    dataset = _fixture()
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"

    result_a = run_from_config(CONFIG_PATH, log_path=log_a, dataset=dataset)
    result_b = run_from_config(CONFIG_PATH, log_path=log_b, dataset=dataset)

    bytes_a = log_a.read_bytes()
    assert bytes_a == log_b.read_bytes()
    assert bytes_a.endswith(b"\n")

    lines = bytes_a.decode("utf-8").splitlines()
    assert len(lines) == config.rounds == len(result_a.rounds) == len(result_b.rounds)
    # Canonical: sorted keys, compact separators, and stable re-encoding.
    for line in lines:
        assert line == canonical_line(json.loads(line))


def test_round_log_represents_every_round_node_and_message(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    log = tmp_path / "episode.jsonl"
    result = run_from_config(CONFIG_PATH, log_path=log, dataset=_fixture())

    topology = load_topology(config)
    node_ids = sorted(int(node) for node in topology.nodes())
    records = _read_records(log)

    assert len(records) == config.rounds == len(result.rounds)
    for index, record in enumerate(records):
        assert record["round"] == index
        assert [entry["node"] for entry in record["nodes"]] == node_ids
        for entry in record["nodes"]:
            assert entry["sample_count"] == config.data.samples_per_node
            assert entry["pre_loss"] > 0
            assert entry["post_loss"] > 0
        assert {
            entry["sender"]: entry["recipients"] for entry in record["messages"]
        } == CHAIN3_RECIPIENTS


def test_first_round_local_loss_decreases_for_every_node(tmp_path: Path) -> None:
    log = tmp_path / "episode.jsonl"
    result = run_from_config(CONFIG_PATH, log_path=log, dataset=_fixture())

    first = result.rounds[0]
    assert len(first.stats) == 3
    for stat in first.stats:
        assert stat.post_loss < stat.pre_loss, (
            f"node {stat.node_id} did not reduce loss: "
            f"pre={stat.pre_loss} post={stat.post_loss}"
        )

    # The published artifact must agree with the in-memory summary.
    record = _read_records(log)[0]
    for entry in record["nodes"]:
        assert entry["post_loss"] < entry["pre_loss"]


def test_final_states_cover_topology_without_aliasing(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    result = run_from_config(CONFIG_PATH, log_path=tmp_path / "e.jsonl", dataset=_fixture())

    topology = load_topology(config)
    assert result.node_ids == tuple(sorted(int(node) for node in topology.nodes()))
    states = list(result.final_states.values())
    assert len(states) == 3
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            key = "features.0.weight"
            assert states[left][key].data_ptr() != states[right][key].data_ptr()


def test_run_without_log_writes_nothing(tmp_path: Path) -> None:
    result = run_from_config(load_config(CONFIG_PATH), dataset=_fixture())
    assert result.log_path is None
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# Failure paths (Q5/Q7): no partial publication
# --------------------------------------------------------------------------- #
def _fail_on_round(target_round: int):
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
        if int(round_index) >= target_round:
            raise RuntimeError(f"injected failure at round {round_index}")
        return local_train_from_snapshot(
            snapshot,
            dataset,
            config=config,
            node_id=node_id,
            round_index=round_index,
            train_seed=train_seed,
            data_order_seed=data_order_seed,
        )

    return trainer


def test_mid_episode_failure_publishes_no_log(tmp_path: Path) -> None:
    log = tmp_path / "episode.jsonl"
    with pytest.raises(EpisodeError, match="round 1 failed"):
        run_from_config(
            CONFIG_PATH,
            log_path=log,
            dataset=_fixture(),
            trainer=_fail_on_round(1),
        )
    assert not log.exists()
    # Temporary output is cleaned; nothing else remains.
    assert list(tmp_path.iterdir()) == []


def test_failure_preserves_prior_complete_log(tmp_path: Path) -> None:
    log = tmp_path / "episode.jsonl"
    log.write_bytes(b"PRIOR-COMPLETE\n")
    with pytest.raises(EpisodeError, match="round 0 failed"):
        run_from_config(
            CONFIG_PATH,
            log_path=log,
            dataset=_fixture(),
            trainer=_fail_on_round(0),
        )
    assert log.read_bytes() == b"PRIOR-COMPLETE\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["episode.jsonl"]


def test_successful_run_replaces_preexisting_log(tmp_path: Path) -> None:
    log = tmp_path / "episode.jsonl"
    log.write_bytes(b"STALE\n")
    result = run_from_config(CONFIG_PATH, log_path=log, dataset=_fixture())
    data = log.read_bytes()
    assert data != b"STALE\n"
    assert data.endswith(b"\n")
    assert len(data.splitlines()) == result.config.rounds


def test_non_finite_dataset_aborts_without_log(tmp_path: Path) -> None:
    log = tmp_path / "episode.jsonl"
    nan_dataset = MNISTShapedDataset(300, seed=1, fill=float("nan"))
    with pytest.raises(EpisodeError):
        run_from_config(CONFIG_PATH, log_path=log, dataset=nan_dataset)
    assert not log.exists()
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# Path safety (Q3/Q7)
# --------------------------------------------------------------------------- #
def test_log_path_aliasing_config_is_rejected(tmp_path: Path) -> None:
    config_copy = tmp_path / "cfg.yaml"
    config_copy.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(LogError, match="alias"):
        run_from_config(config_copy, log_path=config_copy, dataset=_fixture())
    # The config file is untouched and no log was written.
    assert config_copy.read_text(encoding="utf-8") == CONFIG_PATH.read_text(
        encoding="utf-8"
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["cfg.yaml"]


def test_log_path_directory_is_rejected(tmp_path: Path) -> None:
    target_dir = tmp_path / "outdir"
    target_dir.mkdir()
    with pytest.raises(LogError, match="not a regular file"):
        run_from_config(CONFIG_PATH, log_path=target_dir, dataset=_fixture())


def test_log_path_missing_parent_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(LogError, match="does not exist"):
        run_from_config(
            CONFIG_PATH,
            log_path=tmp_path / "missing" / "episode.jsonl",
            dataset=_fixture(),
        )


def test_log_path_symlink_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real.jsonl"
    real.write_bytes(b"x\n")
    link = tmp_path / "link.jsonl"
    link.symlink_to(real)
    with pytest.raises(LogError, match="symlink"):
        validate_log_path(link)


# --------------------------------------------------------------------------- #
# Order independence and divergence
# --------------------------------------------------------------------------- #
def test_canonical_record_is_order_independent() -> None:
    stats = (
        NodeRoundStat(node_id=2, pre_loss=1.0, post_loss=0.5, sample_count=3, neighbors=(1,)),
        NodeRoundStat(node_id=0, pre_loss=2.0, post_loss=1.5, sample_count=3, neighbors=(1,)),
        NodeRoundStat(node_id=1, pre_loss=3.0, post_loss=2.5, sample_count=3, neighbors=(0, 2)),
    )
    messages = (
        MessageSummary(sender=2, recipients=(1,)),
        MessageSummary(sender=0, recipients=(1,)),
        MessageSummary(sender=1, recipients=(2, 0)),
    )
    shuffled = RoundResult(round_index=0, states={}, stats=stats, messages=messages)
    ordered = RoundResult(
        round_index=0,
        states={},
        stats=tuple(sorted(stats, key=lambda stat: stat.node_id)),
        messages=tuple(sorted(messages, key=lambda message: message.sender)),
    )
    assert canonical_record(shuffled) == canonical_record(ordered)
    assert canonical_line(canonical_record(shuffled)) == canonical_line(
        canonical_record(ordered)
    )
    # Senders are sorted and recipient lists normalized to ascending order.
    record = canonical_record(shuffled)
    assert [message["sender"] for message in record["messages"]] == [0, 1, 2]
    assert record["messages"][1]["recipients"] == [0, 2]


def test_write_round_log_helper_roundtrips(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    result = run_from_config(config, dataset=_fixture())
    log = tmp_path / "helper.jsonl"
    published = write_round_log(result.rounds, log)
    assert published == log
    assert len(_read_records(log)) == config.rounds


def test_different_seed_diverges(tmp_path: Path) -> None:
    dataset = _fixture()
    log_a = tmp_path / "seed_a.jsonl"
    log_b = tmp_path / "seed_b.jsonl"
    run_from_config(make_config(seed=111), log_path=log_a, dataset=dataset)
    run_from_config(make_config(seed=222), log_path=log_b, dataset=dataset)
    assert log_a.read_bytes() != log_b.read_bytes()


# --------------------------------------------------------------------------- #
# Dataset injection seam
# --------------------------------------------------------------------------- #
def test_dataset_loader_seam_is_used(tmp_path: Path) -> None:
    seen: dict[str, bool] = {}

    def loader(config):
        seen["called"] = True
        return _fixture()

    result = run_from_config(
        load_config(CONFIG_PATH),
        log_path=tmp_path / "loader.jsonl",
        dataset_loader=loader,
    )
    assert seen.get("called") is True
    assert len(result.rounds) == result.config.rounds


def test_dataset_and_loader_are_mutually_exclusive() -> None:
    with pytest.raises(EpisodeError, match="not both"):
        run_from_config(
            load_config(CONFIG_PATH),
            dataset=_fixture(),
            dataset_loader=lambda config: _fixture(),
        )
