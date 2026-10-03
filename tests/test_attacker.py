"""Pure contract tests for the D016 fixed delta/reversal attacker (R003).

These tests exercise :class:`~dfl_sim.attacker.FixedAttackController` in
isolation from the round engine: exact zero/weak/strong math, integer
preservation, deterministic repeatability, strict input/output/sibling
no-aliasing, and the Q7 malformed-state surface. Nothing here reads files or
the network; every payload is generated in-process.

Layer ownership: config-level negatives (unknown fields, bad intensity,
out-of-topology node ids) live in ``test_config_topology.py``.
"""

from __future__ import annotations

import pytest
import torch

from dfl_sim import (
    ATTACK_KAPPAS,
    AttackConfig,
    AttackError,
    FixedAttackController,
)

from helpers import make_config


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _pair(seed: int = 0) -> tuple[dict, dict]:
    """Two distinct float64 parameter maps with matched keys/shapes/dtypes."""

    generator = torch.Generator().manual_seed(seed)
    round_start = {
        "weight": torch.randn(4, 3, generator=generator, dtype=torch.float64),
        "bias": torch.randn(4, generator=generator, dtype=torch.float64),
    }
    trained = {
        "weight": torch.randn(4, 3, generator=generator, dtype=torch.float64),
        "bias": torch.randn(4, generator=generator, dtype=torch.float64),
    }
    return round_start, trained


def _controller(*, intensity: str = "strong", node: int | None = 0):
    return FixedAttackController(
        attack=AttackConfig(compromised_node=node, intensity=intensity)
    )


def _snapshot(state: dict) -> dict:
    return {key: value.clone() for key, value in state.items()}


# --------------------------------------------------------------------------- #
# Public properties
# --------------------------------------------------------------------------- #
def test_kappa_and_compromised_node_properties() -> None:
    honest = _controller(intensity="zero", node=None)
    assert honest.compromised_node is None
    assert honest.kappa == 0.0

    weak = _controller(intensity="weak", node=1)
    assert weak.compromised_node == 1
    assert weak.kappa == ATTACK_KAPPAS["weak"] == 2.0

    strong = _controller(intensity="strong", node=2)
    assert strong.compromised_node == 2
    assert strong.kappa == ATTACK_KAPPAS["strong"] == 5.0


def test_controller_is_frozen() -> None:
    controller = _controller()
    with pytest.raises(Exception):
        controller.attack = AttackConfig()  # type: ignore[misc]


def test_from_experiment_builds_matching_controller() -> None:
    config = make_config(
        attack={"compromised_node": 2, "intensity": "strong"},
    )
    controller = FixedAttackController.from_experiment(config)
    assert controller.compromised_node == 2
    assert controller.kappa == 5.0
    assert controller.attack is config.attack


# --------------------------------------------------------------------------- #
# Honest path + exact zero no-op (R003)
# --------------------------------------------------------------------------- #
def test_honest_sender_clones_trained_exactly() -> None:
    round_start, trained = _pair()
    controller = _controller(intensity="strong", node=0)

    outgoing = controller.outgoing_state(
        sender=1, round_start=round_start, trained=trained
    )

    assert set(outgoing) == set(trained)
    for key in trained:
        assert torch.equal(outgoing[key], trained[key])
        assert outgoing[key].dtype == trained[key].dtype


def test_unselected_controller_never_poisons() -> None:
    round_start, trained = _pair()
    controller = _controller(intensity="zero", node=None)
    outgoing = controller.outgoing_state(
        sender=0, round_start=round_start, trained=trained
    )
    for key in trained:
        assert torch.equal(outgoing[key], trained[key])


def test_selected_node_zero_intensity_is_tensor_identical_to_honest() -> None:
    round_start, trained = _pair()
    honest = _controller(intensity="zero", node=None)
    noop = _controller(intensity="zero", node=0)

    honest_out = honest.outgoing_state(
        sender=0, round_start=round_start, trained=trained
    )
    noop_out = noop.outgoing_state(
        sender=0, round_start=round_start, trained=trained
    )

    for key in trained:
        assert torch.equal(noop_out[key], honest_out[key])
        assert torch.equal(noop_out[key], trained[key])


# --------------------------------------------------------------------------- #
# Exact attack math (D007/D016)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "intensity, kappa", [("weak", 2.0), ("strong", 5.0)]
)
def test_formula_is_exact_for_configured_sender(
    intensity: str, kappa: float
) -> None:
    round_start, trained = _pair(seed=7)
    controller = _controller(intensity=intensity, node=0)

    outgoing = controller.outgoing_state(
        sender=0, round_start=round_start, trained=trained
    )

    for key in trained:
        expected = trained[key] - kappa * (trained[key] - round_start[key])
        assert torch.equal(outgoing[key], expected)
        # The reversal is a real change from the honest trailing model.
        assert not torch.equal(outgoing[key], trained[key])


def test_non_compromised_nodes_are_untouched_in_strong_episode() -> None:
    round_start, trained = _pair()
    controller = _controller(intensity="strong", node=0)
    for sender in (1, 2):
        outgoing = controller.outgoing_state(
            sender=sender, round_start=round_start, trained=trained
        )
        for key in trained:
            assert torch.equal(outgoing[key], trained[key])


# --------------------------------------------------------------------------- #
# Non-floating tensors (Q7)
# --------------------------------------------------------------------------- #
def test_integer_tensors_are_copied_unchanged() -> None:
    round_start, trained = _pair()
    round_start["counts"] = torch.tensor([1, 2, 3], dtype=torch.int64)
    trained["counts"] = torch.tensor([4, 5, 6], dtype=torch.int64)

    controller = _controller(intensity="strong", node=0)
    outgoing = controller.outgoing_state(
        sender=0, round_start=round_start, trained=trained
    )

    assert outgoing["counts"].dtype == torch.int64
    assert torch.equal(outgoing["counts"], trained["counts"])
    # Floating parameters are still poisoned in the same call.
    expected = trained["weight"] - 5.0 * (trained["weight"] - round_start["weight"])
    assert torch.equal(outgoing["weight"], expected)


# --------------------------------------------------------------------------- #
# Determinism + no-aliasing (R012)
# --------------------------------------------------------------------------- #
def test_repeated_calls_are_bitwise_deterministic() -> None:
    round_start, trained = _pair(seed=3)
    controller = _controller(intensity="weak", node=0)

    first = controller.outgoing_state(
        sender=0, round_start=round_start, trained=trained
    )
    second = controller.outgoing_state(
        sender=0, round_start=round_start, trained=trained
    )

    for key in trained:
        assert torch.equal(first[key], second[key])
        assert first[key].data_ptr() != second[key].data_ptr()


def test_inputs_are_never_mutated() -> None:
    round_start, trained = _pair(seed=11)
    round_before = _snapshot(round_start)
    trained_before = _snapshot(trained)

    controller = _controller(intensity="strong", node=0)
    controller.outgoing_state(sender=0, round_start=round_start, trained=trained)

    for key in round_before:
        assert torch.equal(round_start[key], round_before[key])
        assert torch.equal(trained[key], trained_before[key])


def test_output_does_not_alias_inputs() -> None:
    round_start, trained = _pair(seed=5)
    controller = _controller(intensity="strong", node=0)

    outgoing = controller.outgoing_state(
        sender=0, round_start=round_start, trained=trained
    )

    for key in trained:
        assert outgoing[key].data_ptr() != trained[key].data_ptr()
        assert outgoing[key].data_ptr() != round_start[key].data_ptr()

    # Mutating the output must not leak back into either input.
    outgoing["weight"].add_(100.0)
    assert not torch.equal(outgoing["weight"], trained["weight"])
    assert torch.equal(trained["weight"], _pair(seed=5)[1]["weight"])


def test_sibling_outputs_share_no_storage() -> None:
    round_start, trained = _pair(seed=9)
    controller = _controller(intensity="weak", node=0)

    first = controller.outgoing_state(
        sender=0, round_start=round_start, trained=trained
    )
    second = controller.outgoing_state(
        sender=0, round_start=round_start, trained=trained
    )

    for key in trained:
        assert first[key].data_ptr() != second[key].data_ptr()
        assert torch.equal(first[key], second[key])

    first["bias"].add_(42.0)
    assert not torch.equal(first["bias"], second["bias"])


# --------------------------------------------------------------------------- #
# Invalid payloads (Q7)
# --------------------------------------------------------------------------- #
def test_empty_round_start_rejected() -> None:
    _, trained = _pair()
    with pytest.raises(AttackError, match="round-start state is empty"):
        _controller().outgoing_state(sender=0, round_start={}, trained=trained)


def test_empty_trained_rejected() -> None:
    round_start, _ = _pair()
    with pytest.raises(AttackError, match="trained state is empty"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained={}
        )


def test_key_mismatch_rejected() -> None:
    round_start, trained = _pair()
    del trained["bias"]
    trained["extra"] = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(AttackError, match="key mismatch"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained=trained
        )


def test_shape_mismatch_rejected() -> None:
    round_start, trained = _pair()
    trained["weight"] = trained["weight"][:2]
    with pytest.raises(AttackError, match="shape mismatch"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained=trained
        )


def test_dtype_mismatch_rejected() -> None:
    round_start, trained = _pair()
    trained["weight"] = trained["weight"].to(torch.float32)
    with pytest.raises(AttackError, match="dtype mismatch"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained=trained
        )


def test_non_tensor_entry_rejected() -> None:
    round_start, trained = _pair()
    trained["weight"] = [1.0, 2.0]
    with pytest.raises(AttackError, match="must be a tensor"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained=trained
        )


def test_non_string_key_rejected() -> None:
    round_start, trained = _pair()
    round_start[0] = round_start.pop("weight")
    trained[0] = trained.pop("weight")
    with pytest.raises(AttackError, match="key must be str"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained=trained
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_round_start_rejected(bad: float) -> None:
    round_start, trained = _pair()
    round_start["weight"] = round_start["weight"].clone()
    round_start["weight"][0, 0] = bad
    with pytest.raises(AttackError, match="non-finite"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained=trained
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_trained_rejected(bad: float) -> None:
    round_start, trained = _pair()
    trained["bias"] = trained["bias"].clone()
    trained["bias"][0] = bad
    with pytest.raises(AttackError, match="non-finite"):
        _controller().outgoing_state(
            sender=0, round_start=round_start, trained=trained
        )


def test_error_context_includes_sender_round_and_tensor() -> None:
    round_start, trained = _pair()
    trained["weight"] = trained["weight"][:2]
    with pytest.raises(AttackError) as excinfo:
        _controller().outgoing_state(
            sender=2,
            round_start=round_start,
            trained=trained,
            round_index=7,
        )
    message = str(excinfo.value)
    assert "sender=2" in message
    assert "round=7" in message
    assert "'weight'" in message


def test_caller_context_is_appended_to_errors() -> None:
    round_start, _ = _pair()
    with pytest.raises(AttackError, match="attempt-3"):
        _controller().outgoing_state(
            sender=0,
            round_start=round_start,
            trained={},
            context="attempt-3",
        )
