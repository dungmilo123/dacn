"""Runtime and static information-boundary acceptance tests (R005, D012, T04).

The S03 contract claims the policy seam is an *executable* boundary, not a
naming convention. These tests prove that claim two independent ways:

1. **Runtime object capability.** A recording policy is invoked by the real
   round engine and asserts that its sole argument is the frozen, scalar-only
   :class:`~dfl_sim.observation.LocalObservation`: no model tensor, dataset,
   topology, attack config/identity, peer observation, global metric, evaluator
   label/state, or contamination flag is reachable from it -- including when the
   episode is run under a configured compromised node.
2. **Static module dependency.** A narrow AST contract over the production
   ``observation.py``/``policy.py`` modules forbids imports from
   evaluator/trajectory/simulator/attack/topology modules and any read of
   banned privileged identifiers, while still allowing explicitly local node
   evaluation helpers and the legitimate ``aggregation``/``config``/``observation``
   dependencies.

Negative coverage proves fail-closed behavior with no partial commit: stale and
foreign participant profiles, post-construction NaN weight injection, a policy
that raises on the final node, and frozen-record mutation are all rejected before
any caller tensor is written. A fixture with representative forbidden imports and
attribute names proves the boundary checker itself detects leaks.

Everything is offline and in-process: no ``.gsd``/planning files, no ignored
artifacts, no network, and no local MNIST download.
"""

from __future__ import annotations

import ast
import dataclasses
import gc
import inspect
import textwrap
import types as pytypes
from collections.abc import Mapping
from pathlib import Path

import networkx as nx
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dfl_sim import (
    AggregationError,
    AttackConfig,
    ExperimentConfig,
    FixedSimilarityFilterPolicy,
    LocalObservation,
    LocalOnlyPolicy,
    LocalTrainResult,
    NeighborObservation,
    PlainAveragePolicy,
    PolicyError,
    RoundError,
    WeightProfile,
    build_weight_profile,
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

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_MODULES = (
    REPO_ROOT / "src" / "dfl_sim" / "observation.py",
    REPO_ROOT / "src" / "dfl_sim" / "policy.py",
)

#: The privileged evaluator may consume only these dfl_sim modules (plus
#: stdlib/third-party); it must never reach back into observation/policy in a
#: way that would let its records become policy input.
EVALUATOR_MODULE = REPO_ROOT / "src" / "dfl_sim" / "evaluator.py"
EVALUATOR_ALLOWED_DFL_IMPORTS = {
    "dfl_sim.aggregation",
    "dfl_sim.config",
    "dfl_sim.models",
    "dfl_sim.reproducibility",
    "dfl_sim.simulator",
    "dfl_sim.topology",
}

# Static dependency boundary: observation/policy may not reach into evaluator,
# trajectory, simulator/round, attack, topology, data, or logging modules.
FORBIDDEN_MODULE_PREFIXES = (
    "dfl_sim.simulator",
    "dfl_sim.run",
    "dfl_sim.attacker",
    "dfl_sim.trajectory",
    "dfl_sim.evaluator",
    "dfl_sim.round_log",
    "dfl_sim.topology",
    "dfl_sim.data",
    "dfl_sim.reproducibility",
    "torch.utils.data",
    "torchvision",
)

# The only dfl_sim sibling modules the boundary modules may depend on.
ALLOWED_DFL_IMPORTS = {
    "dfl_sim.aggregation",
    "dfl_sim.config",
    "dfl_sim.observation",
}

# Privileged identifiers a boundary module or observation must never expose.
BANNED_NAMES = frozenset(
    {
        "compromised_node",
        "kappa",
        "ground_truth",
        "contamination",
        "contaminated",
        "global_metrics",
        "peer_observations",
        "other_inboxes",
        "network_state",
        "all_snapshots",
        "attack_config",
        "evaluator_state",
        "evaluator_labels",
        "trajectory",
    }
)

NODES3 = (0, 1, 2)


# --------------------------------------------------------------------------- #
# Static AST/import contract
# --------------------------------------------------------------------------- #
def _full_module(node: ast.ImportFrom) -> str:
    if node.level:
        return f"dfl_sim.{node.module}" if node.module else "dfl_sim"
    return node.module or ""


def _is_forbidden_module(module: str) -> bool:
    return any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in FORBIDDEN_MODULE_PREFIXES
    )


def _dfl_imports(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = _full_module(node)
            if module == "dfl_sim" or module.startswith("dfl_sim."):
                found.add(module)
    return found


def _boundary_violations(source: str) -> list[str]:
    """Return boundary violations in ``source`` (empty when clean)."""

    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_module(alias.name):
                    violations.append(f"forbidden import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = _full_module(node)
            if _is_forbidden_module(module):
                violations.append(f"forbidden import {module}")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                violations.append(f"banned name {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_NAMES:
                violations.append(f"banned attribute {node.attr}")
    return violations


def test_production_modules_respect_static_dependency_boundary() -> None:
    for path in PRODUCTION_MODULES:
        source = path.read_text(encoding="utf-8")
        assert _boundary_violations(source) == [], f"{path.name} leaks privileges"

        tree = ast.parse(source)
        assert _dfl_imports(tree) <= ALLOWED_DFL_IMPORTS, (
            f"{path.name} imports outside the allowed boundary: "
            f"{_dfl_imports(tree) - ALLOWED_DFL_IMPORTS}"
        )


def test_observation_and_policy_never_import_the_evaluator() -> None:
    for path in PRODUCTION_MODULES:
        imports = _dfl_imports(ast.parse(path.read_text(encoding="utf-8")))
        assert "dfl_sim.evaluator" not in imports, f"{path.name} imports evaluator"


def test_evaluator_imports_only_allowed_privileged_sources() -> None:
    tree = ast.parse(EVALUATOR_MODULE.read_text(encoding="utf-8"))
    imports = _dfl_imports(tree)
    assert imports <= EVALUATOR_ALLOWED_DFL_IMPORTS, (
        f"evaluator.py imports outside its privileged allowlist: "
        f"{imports - EVALUATOR_ALLOWED_DFL_IMPORTS}"
    )
    # The evaluator consumes the committed round + topology, not the policy
    # or observation modules it must stay above.
    assert "dfl_sim.observation" not in imports
    assert "dfl_sim.policy" not in imports


def test_boundary_checker_detects_representative_leaks() -> None:
    source = textwrap.dedent(
        """
        import torch.utils.data
        import torchvision
        from dfl_sim.simulator import RoundResult
        from .attacker import FixedAttackController
        from .evaluator import score
        from .trajectory import Trace

        def leak(config, observation):
            node = config.compromised_node
            weight = config.kappa
            g = config.topology
            contamination = observation.contamination
            return node, weight, g, contamination
        """
    )
    violations = _boundary_violations(source)

    assert any("torch.utils.data" in v for v in violations)
    assert any("torchvision" in v for v in violations)
    assert any("dfl_sim.simulator" in v for v in violations)
    assert any("dfl_sim.attacker" in v for v in violations)
    assert any("dfl_sim.evaluator" in v for v in violations)
    assert any("dfl_sim.trajectory" in v for v in violations)
    # Attribute reads of privileged names are caught too.
    assert any("compromised_node" in v for v in violations)
    assert any("kappa" in v for v in violations)
    assert any("contamination" in v for v in violations)


def test_boundary_checker_accepts_local_node_evaluation_helpers() -> None:
    # A local loss helper named ``evaluate`` and a local ``node_id`` argument are
    # explicitly permitted: only privileged *state* names are banned.
    source = textwrap.dedent(
        """
        from .observation import LocalObservation

        def evaluate(state, *, node_id, round_index, context=""):
            return float(state["w"].sum())

        def local_only(observation: LocalObservation):
            return observation.node_id, observation.degree
        """
    )
    assert _boundary_violations(source) == []


# --------------------------------------------------------------------------- #
# Runtime reachability helper
# --------------------------------------------------------------------------- #
def _forbidden_reachable(root: object) -> list[str]:
    """Collect privileged objects reachable from ``root`` (boundary walk).

    Traversal stops at primitive scalars, strings, and type/module objects so
    the walk stays bounded by the observation's flat scalar schema.
    """

    found: set[str] = set()
    seen: set[int] = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))

        if isinstance(obj, torch.Tensor):
            found.add(f"torch.Tensor:{type(obj).__name__}")
        if isinstance(obj, Mapping) and not isinstance(obj, (str, bytes)):
            found.add(f"Mapping:{type(obj).__name__}")
        if isinstance(obj, Dataset):
            found.add(f"Dataset:{type(obj).__name__}")
        if isinstance(obj, nx.Graph):
            found.add(f"Graph:{type(obj).__name__}")
        if isinstance(obj, (ExperimentConfig, AttackConfig)):
            found.add(type(obj).__name__)
        if isinstance(obj, nn.Module):
            found.add(f"Module:{type(obj).__name__}")

        if isinstance(
            obj,
            (type, pytypes.ModuleType, str, bytes, int, float, bool, type(None)),
        ):
            continue
        for referent in gc.get_referents(obj):
            stack.append(referent)
    return sorted(found)


def _schema_violations(observation: object) -> list[str]:
    """Return reasons ``observation`` violates the scalar-only boundary."""

    violations: list[str] = []
    if not isinstance(observation, LocalObservation):
        return [f"not a LocalObservation: {type(observation).__name__}"]
    if type(observation).__module__ != "dfl_sim.observation":
        violations.append(f"unexpected observation module {type(observation).__module__}")

    if not isinstance(observation.node_id, int):
        violations.append("node_id is not int")
    if not isinstance(observation.round_index, int):
        violations.append("round_index is not int")
    if not isinstance(observation.epsilon, float):
        violations.append("epsilon is not float")
    if not isinstance(observation.neighbors, tuple):
        violations.append("neighbors is not a tuple")

    for neighbor in observation.neighbors:
        if not isinstance(neighbor, NeighborObservation):
            violations.append(f"neighbor is not NeighborObservation: {type(neighbor).__name__}")
            continue
        for name in ("sender_id",):
            if not isinstance(getattr(neighbor, name), int):
                violations.append(f"neighbor.{name} is not int")
        for name in ("directional_disagreement", "relative_norm", "loss_delta"):
            if not isinstance(getattr(neighbor, name), float):
                violations.append(f"neighbor.{name} is not float")

    if hasattr(observation, "__dict__"):
        violations.append("observation exposes a mutable __dict__")
    for name in BANNED_NAMES:
        if hasattr(observation, name):
            violations.append(f"privileged attribute {name}")

    for leaked in _forbidden_reachable(observation):
        violations.append(f"reachable privileged object {leaked}")
    return violations


def _schema_signature(observation: LocalObservation) -> tuple:
    return (
        f"{type(observation).__module__}.{type(observation).__name__}",
        tuple(observation.__slots__),
        tuple(
            (type(neighbor).__name__, tuple(neighbor.__slots__))
            for neighbor in observation.neighbors
        ),
    )


# --------------------------------------------------------------------------- #
# Round-driving helpers
# --------------------------------------------------------------------------- #
def _config(**overrides):
    return make_config(topology={"kind": "chain3"}, **overrides)


def _offset_trainer(scale: float = 10.0):
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


def _zero_snapshots() -> dict[int, dict[str, torch.Tensor]]:
    return {node: {"w": torch.tensor([0.0])} for node in NODES3}


def _run(*, config=None, policy=None, snapshots=None):
    config = config or _config()
    return run_round(
        config=config,
        topology=load_topology(config),
        snapshots=snapshots if snapshots is not None else _zero_snapshots(),
        partitions={node: None for node in NODES3},
        round_index=0,
        seeds=resolve_seeds(config),
        trainer=_offset_trainer(),
        loss_evaluator=constant_loss_evaluator_factory,
        policy=policy,
    )


class _BoundaryRecorder:
    """Record every policy observation and delegate to plain averaging."""

    def __init__(self) -> None:
        self._inner = PlainAveragePolicy()
        self.calls: list[int] = []
        self.seen: list[LocalObservation] = []

    def select_weights(self, observation: LocalObservation) -> WeightProfile:
        self.calls.append(int(observation.node_id))
        self.seen.append(observation)
        return self._inner.select_weights(observation)


# --------------------------------------------------------------------------- #
# Runtime object capability boundary
# --------------------------------------------------------------------------- #
def test_policy_receives_only_scalar_local_observation() -> None:
    recorder = _BoundaryRecorder()
    _run(policy=recorder)

    assert recorder.calls == [0, 1, 2]
    assert len(recorder.seen) == 3
    for observation in recorder.seen:
        assert _schema_violations(observation) == []
        assert _forbidden_reachable(observation) == []
        # A tensor/dataset/topology is not merely ``None`` -- it is absent.
        for name in BANNED_NAMES:
            assert not hasattr(observation, name)


def test_policy_input_schema_is_identical_under_compromised_node() -> None:
    honest = _BoundaryRecorder()
    _run(policy=honest)

    attacked_config = _config(
        attack={"compromised_node": 1, "intensity": "strong"}
    )
    attacked = _BoundaryRecorder()
    _run(config=attacked_config, policy=attacked)

    assert attacked_config.attack.compromised_node == 1
    for node in NODES3:
        honest_observation = honest.seen[node]
        attacked_observation = attacked.seen[node]
        assert _schema_signature(attacked_observation) == _schema_signature(
            honest_observation
        )
        assert _schema_violations(attacked_observation) == []
        assert not hasattr(attacked_observation, "compromised_node")

    # The compromised sender's different outgoing state still reaches the honest
    # neighbour's *scalar* observation (relative delta norm changes), proving the
    # attacked episode exercised the real attack seam.
    assert (
        honest.seen[0].neighbors[0].relative_norm
        != attacked.seen[0].neighbors[0].relative_norm
    )


def test_builtin_policies_reject_non_observation_arguments() -> None:
    policies = (
        LocalOnlyPolicy(),
        PlainAveragePolicy(),
        FixedSimilarityFilterPolicy(cutoff=1.0, downweight_multiplier=0.5),
    )
    for policy in policies:
        signature = inspect.signature(type(policy).select_weights)
        assert list(signature.parameters) == ["self", "observation"]

        with pytest.raises(PolicyError):
            policy.select_weights(torch.zeros(2))  # type: ignore[arg-type]
        with pytest.raises(PolicyError):
            policy.select_weights({"w": torch.zeros(2)})  # type: ignore[arg-type]


def test_observations_and_profiles_are_immutable() -> None:
    recorder = _BoundaryRecorder()
    _run(policy=recorder)
    observation = recorder.seen[1]

    with pytest.raises(dataclasses.FrozenInstanceError):
        observation.node_id = 99  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        observation.neighbors[0].loss_delta = 9.0  # type: ignore[misc]

    profile = PlainAveragePolicy().select_weights(observation)
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.weights = (1.0,)  # type: ignore[misc]


def test_isolated_empty_inbox_is_still_a_valid_scalar_observation() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(NODES3)
    graph.add_edge(1, 2)  # node 0 has an empty inbox
    topology = nx.freeze(graph)

    config = _config()
    recorder = _BoundaryRecorder()
    result = run_round(
        config=config,
        topology=topology,
        snapshots=_zero_snapshots(),
        partitions={node: None for node in NODES3},
        round_index=0,
        seeds=resolve_seeds(config),
        trainer=_offset_trainer(),
        loss_evaluator=constant_loss_evaluator_factory,
        policy=recorder,
    )

    isolated = recorder.seen[0]
    assert isolated.degree == 0
    assert isolated.neighbors == ()
    assert _schema_violations(isolated) == []
    assert result.states[0]["w"].tolist() == [0.0]


# --------------------------------------------------------------------------- #
# Fail-closed negatives (no partial commit)
# --------------------------------------------------------------------------- #
def test_foreign_participant_profile_aborts_before_commit() -> None:
    snapshots = _zero_snapshots()
    baseline = {node: snapshots[node]["w"].clone() for node in NODES3}

    class ForeignPolicy:
        def select_weights(self, observation):
            return WeightProfile(
                (observation.node_id + 100,), (1.0,), (1.0,)
            )

    with pytest.raises(RoundError, match="policy participants"):
        _run(policy=ForeignPolicy(), snapshots=snapshots)
    for node in NODES3:
        assert torch.equal(snapshots[node]["w"], baseline[node])


def test_stale_participant_profile_aborts_before_commit() -> None:
    snapshots = _zero_snapshots()
    baseline = {node: snapshots[node]["w"].clone() for node in NODES3}

    class StalePolicy:
        def select_weights(self, observation):
            # Claim self-only even though senders were delivered.
            return build_weight_profile(
                observation.participant_ids[:1], (1.0,)
            )

    with pytest.raises(RoundError, match="policy participants"):
        _run(policy=StalePolicy(), snapshots=snapshots)
    for node in NODES3:
        assert torch.equal(snapshots[node]["w"], baseline[node])


def test_nan_weight_injection_aborts_before_commit() -> None:
    snapshots = _zero_snapshots()
    baseline = {node: snapshots[node]["w"].clone() for node in NODES3}

    class NaNWieghtPolicy:
        def select_weights(self, observation):
            profile = PlainAveragePolicy().select_weights(observation)
            # Bypass frozen/validated construction to inject a non-finite weight.
            object.__setattr__(
                profile,
                "weights",
                (float("nan"),) + tuple(profile.weights[1:]),
            )
            return profile

    with pytest.raises(AggregationError, match="finite"):
        _run(policy=NaNWieghtPolicy(), snapshots=snapshots)
    for node in NODES3:
        assert torch.equal(snapshots[node]["w"], baseline[node])


def test_policy_raising_on_final_node_aborts_without_partial_commit() -> None:
    snapshots = _zero_snapshots()
    baseline = {node: snapshots[node]["w"].clone() for node in NODES3}

    class RaisingPolicy:
        def select_weights(self, observation):
            if observation.node_id == 2:
                raise PolicyError("injected boundary failure")
            return PlainAveragePolicy().select_weights(observation)

    with pytest.raises(
        RoundError, match=r"policy failed for node=2 round=0"
    ) as info:
        _run(policy=RaisingPolicy(), snapshots=snapshots)
    assert isinstance(info.value.__cause__, PolicyError)
    for node in NODES3:
        assert torch.equal(snapshots[node]["w"], baseline[node])


# --------------------------------------------------------------------------- #
# Load profile: policy calls scale as nodes x rounds, not network-wide state
# --------------------------------------------------------------------------- #
def test_policy_calls_scale_as_nodes_times_rounds() -> None:
    config = _config(rounds=2, data={"samples_per_node": 16})
    recorder = _BoundaryRecorder()
    dataset = MNISTShapedDataset(48, seed=3)

    run_from_config(
        config,
        dataset=dataset,
        trainer=_offset_trainer(),
        policy=recorder,
    )

    # chain3: 3 nodes x 2 rounds policy calls; 4 directed messages per round.
    assert len(recorder.calls) == 3 * 2
    assert sum(len(observation.neighbors) for observation in recorder.seen) == 4 * 2
    assert all(_schema_violations(observation) == [] for observation in recorder.seen)


def test_policy_boundary_holds_when_trace_emission_is_active(tmp_path: Path) -> None:
    """The privileged evaluator/writer must never leak into the policy seam.

    With schema-v1 trace emission active the runner holds an evaluator, a
    TraceWriter, the global evaluation dataset, true node types, and prior
    evaluated rounds. None of those may become reachable from the policy's
    observation, so the same scalar-only capability check must still pass.
    """

    config = _config(rounds=2, data={"samples_per_node": 16})
    recorder = _BoundaryRecorder()
    dataset = MNISTShapedDataset(48, seed=3)
    evaluation = MNISTShapedDataset(32, seed=4)
    trace_dir = tmp_path / "trace"

    result = run_from_config(
        config,
        dataset=dataset,
        evaluation_dataset=evaluation,
        trace_path=trace_dir,
        trainer=_offset_trainer(),
        policy=recorder,
    )

    # The real trace was published and every policy call still saw only a
    # frozen, scalar-only LocalObservation.
    assert result.trace_path == trace_dir
    assert (trace_dir / "manifest.json").is_file()
    assert len(recorder.calls) == 3 * 2
    assert all(_schema_violations(observation) == [] for observation in recorder.seen)
