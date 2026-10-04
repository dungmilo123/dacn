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
from collections.abc import Mapping, Sequence
from pathlib import Path

import networkx as nx
import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError
from torch.utils.data import Dataset

from dfl_sim import (
    DEFAULT_EPSILON,
    AggregationError,
    AttackConfig,
    BalanceAggregation,
    BayesianAggregation,
    BayesianArtifact,
    BayesianDiagnostic,
    BayesianPolicy,
    ExperimentConfig,
    FixedSimilarityFilterPolicy,
    LocalObservation,
    LocalOnlyPolicy,
    LocalTrainResult,
    NeighborObservation,
    PlainAveragePolicy,
    PolicyError,
    RoundError,
    TrustWeightedAggregation,
    ValidationReferencePolicy,
    WeightProfile,
    build_bayesian_policy,
    build_policy,
    build_weight_profile,
    collect_calibration,
    estimate_calibration,
    load_bayesian_artifact,
    load_calibration_config,
    load_topology,
    publish_calibration_artifact,
    resolve_seeds,
    run_from_config,
    run_round,
)
from dfl_sim.policy import (
    FALLBACK_CONTEXT_UNAVAILABLE,
    FALLBACK_MISSING_LIKELIHOOD,
    FALLBACK_MISSING_PRIOR,
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
# ``bayesian_artifact`` is a leaf contract module (stdlib/pydantic only) that
# the Bayesian policy consumes offline; it exposes no privileged runtime state.
ALLOWED_DFL_IMPORTS = {
    "dfl_sim.aggregation",
    "dfl_sim.bayesian_artifact",
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

#: BALANCE Algorithm 3's exponential decay coefficient is named ``kappa`` in the
#: cited paper (D028). It is a construction-time local hyperparameter that grants
#: no access to the D016 attack intensity: ``attack_config``, ``intensity``, and
#: ``compromised_node`` remain banned, so a real attack-state read is still
#: caught. The production policy-module check therefore allows the bare
#: ``kappa`` identifier while the generic checker stays strict.
POLICY_LOCAL_HYPERPARAMETER_NAMES = frozenset({"kappa"})


# --------------------------------------------------------------------------- #
# Static AST/import contract
# --------------------------------------------------------------------------- #
def _full_module(node: ast.ImportFrom) -> str:
    if node.level:
        return f"dfl_sim.{node.module}" if node.module else "dfl_sim"
    return node.module or ""


def _is_forbidden_module(
    module: str, prefixes: Sequence[str] = FORBIDDEN_MODULE_PREFIXES
) -> bool:
    return any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in prefixes
    )


def _dfl_imports(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = _full_module(node)
            if module == "dfl_sim" or module.startswith("dfl_sim."):
                found.add(module)
    return found


def _boundary_violations(
    source: str,
    *,
    forbidden: Sequence[str] = FORBIDDEN_MODULE_PREFIXES,
    banned: frozenset[str] = BANNED_NAMES,
    forbidden_string_keys: frozenset[str] = frozenset(),
    allowed: frozenset[str] = frozenset(),
) -> list[str]:
    """Return boundary violations in ``source`` (empty when clean).

    ``allowed`` names are exempted from the ``banned``/``forbidden_string_keys``
    checks for callers that deliberately permit a documented local
    hyperparameter (for example BALANCE's ``kappa``).
    """

    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_module(alias.name, forbidden):
                    violations.append(f"forbidden import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = _full_module(node)
            if _is_forbidden_module(module, forbidden):
                violations.append(f"forbidden import {module}")
        elif isinstance(node, ast.Name):
            if node.id in banned and node.id not in allowed:
                violations.append(f"banned name {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr in banned and node.attr not in allowed:
                violations.append(f"banned attribute {node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in forbidden_string_keys and node.value not in allowed:
                violations.append(f"forbidden string key {node.value!r}")
    return violations


def test_production_modules_respect_static_dependency_boundary() -> None:
    for path in PRODUCTION_MODULES:
        source = path.read_text(encoding="utf-8")
        allowed = (
            POLICY_LOCAL_HYPERPARAMETER_NAMES
            if path.name == "policy.py"
            else frozenset()
        )
        assert _boundary_violations(source, allowed=allowed) == [], (
            f"{path.name} leaks privileges: "
            f"{_boundary_violations(source, allowed=allowed)}"
        )

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
        ValidationReferencePolicy(tolerance=0.0),
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


# --------------------------------------------------------------------------- #
# Bayesian artifact/policy boundary (R013/R015/R016, T04)
# --------------------------------------------------------------------------- #
POLICY_MODULE = REPO_ROOT / "src" / "dfl_sim" / "policy.py"
BAYESIAN_ARTIFACT_MODULE = REPO_ROOT / "src" / "dfl_sim" / "bayesian_artifact.py"
BAYESIAN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"

# The artifact contract and the Bayesian policy are online modules: they may
# never import the evaluator, attacker, topology, datasets, model definitions,
# or tensors. ``observation.py`` legitimately imports torch for the production
# observation *builder*, so torch is forbidden only for the Bayesian modules.
BAYESIAN_FORBIDDEN_MODULE_PREFIXES = (
    "dfl_sim.simulator",
    "dfl_sim.run",
    "dfl_sim.attacker",
    "dfl_sim.trajectory",
    "dfl_sim.evaluator",
    "dfl_sim.round_log",
    "dfl_sim.topology",
    "dfl_sim.data",
    "dfl_sim.models",
    "dfl_sim.reproducibility",
    # The offline calibration family owns the reference population model, the
    # design split, the hidden sender types, and the counterfactual scores; the
    # online policy/loader may never import or reach any of it (R005).
    "dfl_sim.calibration",
    "dfl_sim.calibration_config",
    "dfl_sim.calibration_scoring",
    "dfl_sim.calibration_estimator",
    "dfl_sim.calibration_artifact_writer",
    "torch",
    "torchvision",
)

# Additive privileged names for the Bayesian online path: true attacker types,
# global/network metrics, raw state dictionaries, model/message tensors, and
# calibration design *data*. ``design_split`` is deliberately absent: the
# artifact records the immutable provenance literal ``"clean_global_heldout"``
# as metadata, which names no runtime data.
BAYESIAN_BANNED_NAMES = BANNED_NAMES | frozenset(
    {
        "true_type",
        "true_types",
        "attacker_type",
        "attacker_types",
        "true_attacker_type",
        "true_attacker_types",
        "true_label",
        "true_labels",
        "evaluator_label",
        "evaluator_labels",
        "global_metric",
        "network_metrics",
        "raw_state",
        "raw_states",
        "state_dict",
        "message_tensor",
        "message_tensors",
        "model_tensor",
        "model_tensors",
        "calibration_data",
        "calibration_split",
        "design_data",
        "design_partition",
        "design_partitions",
    }
)

# Privileged *string keys* the online path must never use to index a raw dict.
BAYESIAN_FORBIDDEN_STRING_KEYS = frozenset(
    {
        "true_type",
        "true_types",
        "attacker_type",
        "attacker_types",
        "ground_truth",
        "compromised_node",
        "kappa",
        "global_metrics",
        "network_state",
        "state_dict",
        "message",
        "tensor",
        "label",
        "labels",
        "design_split",
    }
)

# Bayesian online entry points and the exact, bounded set of in-module helpers
# they are allowed to execute. Reachability is computed from call and
# function-reference edges; the artifact leaf is reached only through the three
# approved imports below.
BAYESIAN_POLICY_ROOTS = frozenset(
    {
        "BayesianPolicy.select_weights",
        "build_bayesian_policy",
        "_build_bayesian_from_config",
        "build_policy",
        "_require_bayesian_provenance_match",
    }
)

BAYESIAN_ALLOWED_ONLINE_HELPERS = frozenset(
    {
        "BayesianPolicy._emit",
        "BayesianPolicy._elapsed_ns",
        "BayesianPolicy._fallback",
        "BayesianPolicy.select_weights",
        "_action_sort_key",
        "_build_bayesian_from_config",
        "_compute_posterior",
        "_expected_payoffs",
        "_finite_product",
        "_first_argmax",
        "_neighbor_evidence",
        "_normalize_multipliers",
        "_normalize_participant_ids",
        "_observation_context",
        "_require_bayesian_provenance_match",
        "_require_finite_float",
        "_require_int",
        "_require_observation",
        "_resolve_degree_bin",
        "_resolve_progress_bin",
        "_select_variable_positions",
        "_suffix",
        "build_bayesian_policy",
        "build_policy",
        "build_weight_profile",
        "enumerate_candidate_actions",
    }
)

# The key math/selection helpers that must remain on the online path, so the
# reachability test cannot silently degrade to a smaller set.
BAYESIAN_REQUIRED_ONLINE_HELPERS = frozenset(
    {
        "BayesianPolicy._fallback",
        "build_weight_profile",
        "enumerate_candidate_actions",
        "_compute_posterior",
        "_expected_payoffs",
        "_select_variable_positions",
    }
)

# The D026 validation-loss reference is a fourth built-in baseline over the same
# narrow runtime seam. Its whole implementation may reach only the frozen local
# observation guard, the scalar-observation context suffix, and the shared
# finite weight-profile helpers -- never evaluator labels, attack config, timing,
# or any calibration/report state.
VALIDATION_REFERENCE_ROOTS = frozenset(
    {
        "ValidationReferencePolicy.select_weights",
        "ValidationReferencePolicy.__post_init__",
    }
)

VALIDATION_REFERENCE_ALLOWED_HELPERS = frozenset(
    {
        "ValidationReferencePolicy.__post_init__",
        "ValidationReferencePolicy.select_weights",
        "_normalize_multipliers",
        "_normalize_participant_ids",
        "_observation_context",
        "_require_finite_float",
        "_require_int",
        "_require_observation",
        "_suffix",
        "build_weight_profile",
    }
)

VALIDATION_REFERENCE_REQUIRED_HELPERS = frozenset(
    {
        "ValidationReferencePolicy.select_weights",
        "ValidationReferencePolicy.__post_init__",
        "build_weight_profile",
        "_require_observation",
    }
)

# The D029 trust comparator and the D028 BALANCE reproduction are two more
# built-in policies over the same narrow runtime seam. Each may reach only the
# frozen local observation guard, scalar-observation helpers, and the shared
# finite weight-profile fold -- never evaluator labels, attack config, topology,
# data, or any calibration/report state.
TRUST_WEIGHTED_ROOTS = frozenset(
    {
        "TrustWeightedPolicy.select_weights",
        "TrustWeightedPolicy.__post_init__",
    }
)

TRUST_WEIGHTED_ALLOWED_HELPERS = frozenset(
    {
        "TrustWeightedPolicy.__post_init__",
        "TrustWeightedPolicy.select_weights",
        "_normalize_multipliers",
        "_normalize_participant_ids",
        "_observation_context",
        "_require_finite_float",
        "_require_int",
        "_require_observation",
        "_suffix",
        "build_weight_profile",
    }
)

TRUST_WEIGHTED_REQUIRED_HELPERS = frozenset(
    {
        "TrustWeightedPolicy.select_weights",
        "TrustWeightedPolicy.__post_init__",
        "build_weight_profile",
        "_require_observation",
    }
)

BALANCE_ROOTS = frozenset(
    {
        "BalancePolicy.select_weights",
        "BalancePolicy.__post_init__",
        "BalancePolicy.threshold",
    }
)

BALANCE_ALLOWED_HELPERS = frozenset(
    {
        "BalancePolicy.__post_init__",
        "BalancePolicy.select_weights",
        "BalancePolicy.threshold",
        "_normalize_multipliers",
        "_normalize_participant_ids",
        "_observation_context",
        "_require_finite_float",
        "_require_int",
        "_require_observation",
        "_suffix",
        "build_weight_profile",
    }
)

BALANCE_REQUIRED_HELPERS = frozenset(
    {
        "BalancePolicy.select_weights",
        "BalancePolicy.__post_init__",
        "BalancePolicy.threshold",
        "build_weight_profile",
        "_require_observation",
    }
)

BAYESIAN_ALLOWED_ARTIFACT_IMPORTS = frozenset(
    {"BayesianArtifact", "interval_bin_index", "load_bayesian_artifact"}
)

BAYESIAN_DIAGNOSTIC_FIELDS = (
    "node_id",
    "round_index",
    "degree",
    "degree_bin",
    "progress",
    "chosen_multipliers",
    "candidate_count",
    "pruned",
    "posterior",
    "fallback_reason",
    "normalization_mass",
    "sender_ids",
    "evidence_bins",
    "elapsed_ns",
)

ALLOWED_FALLBACK_REASONS = frozenset(
    {
        FALLBACK_CONTEXT_UNAVAILABLE,
        FALLBACK_MISSING_LIKELIHOOD,
        FALLBACK_MISSING_PRIOR,
    }
)

# Offline-only calibration module prefixes. The online policy/loader must not
# import them; kept separate so the static test can assert the exact set.
CALIBRATION_MODULE_PREFIXES = (
    "dfl_sim.calibration",
    "dfl_sim.calibration_config",
    "dfl_sim.calibration_scoring",
    "dfl_sim.calibration_estimator",
    "dfl_sim.calibration_artifact_writer",
)

# Offline-only calibration/collection/design symbols that must never appear by
# name on the online path. These are *not* artifact-contract field names (the
# manifest legitimately records aggregate support/coverage scalars), so a
# positive hit means a real privilege leak.
CALIBRATION_OFFLINE_SYMBOLS = frozenset(
    {
        "collect_calibration",
        "collect_episode",
        "estimate_artifact",
        "estimate_calibration",
        "publish_calibration_artifact",
        "build_schedule",
        "build_schedule_and_counts",
        "count_prior_states",
        "derive_experiment_config",
        "feasible_prior_states",
        "design_indices",
        "design_dataset",
        "design_loss",
        "design_split_size",
        "hidden_types",
        "marginals",
        "episode_spec",
        "candidate_scores",
    }
)

# The checked-in offline calibration config drives the generated-artifact
# runtime walk below, so the online surface is exercised against a real
# published artifact rather than only the hand-written fixture.
CALIBRATION_CONFIG_PATH = REPO_ROOT / "configs" / "calibrate_chain3.yaml"


def _module_definitions(
    tree: ast.Module,
) -> tuple[dict[str, ast.AST], dict[str, ast.AST]]:
    """Return (top-level functions, methods) keyed by name / ``Class.method``."""

    top: dict[str, ast.AST] = {}
    methods: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[f"{node.name}.{child.name}"] = child
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top[node.name] = node
    return top, methods


def _callee_names(
    node: ast.AST,
    top: Mapping[str, ast.AST],
    methods: Mapping[str, ast.AST],
) -> set[str]:
    """Resolve the in-module call/reference targets of one function body."""

    by_simple: dict[str, list[str]] = {}
    for qualified in methods:
        by_simple.setdefault(qualified.rsplit(".", 1)[-1], []).append(qualified)

    called: set[str] = set()
    referenced: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
        elif isinstance(child, ast.Name):
            referenced.add(child.id)

    targets: set[str] = set()
    for name in called:
        if name in top or name in methods:
            targets.add(name)
        targets.update(by_simple.get(name, ()))
    # Only top-level functions resolve from bare name references, so a local
    # variable named ``degree`` cannot pull in a ``WeightProfile.degree``
    # property and inflate the reachable set.
    for name in referenced:
        if name in top:
            targets.add(name)
    return targets


def _reachable_definitions(
    source: str, roots: Sequence[str]
) -> tuple[frozenset[str], dict[str, ast.AST]]:
    """Return the in-module definitions reachable from ``roots``."""

    tree = ast.parse(source)
    top, methods = _module_definitions(tree)
    definitions = {**top, **methods}
    missing = [root for root in roots if root not in definitions]
    assert not missing, f"call-graph roots not found: {missing}"

    seen: set[str] = set()
    stack = list(roots)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for target in _callee_names(definitions[current], top, methods):
            if target not in seen:
                stack.append(target)
    return frozenset(seen), definitions


def test_bayesian_artifact_module_is_a_privilege_free_leaf() -> None:
    source = BAYESIAN_ARTIFACT_MODULE.read_text(encoding="utf-8")
    assert (
        _boundary_violations(
            source,
            forbidden=BAYESIAN_FORBIDDEN_MODULE_PREFIXES,
            banned=BAYESIAN_BANNED_NAMES,
            forbidden_string_keys=BAYESIAN_FORBIDDEN_STRING_KEYS,
        )
        == []
    )
    # The artifact contract is a leaf: it imports no dfl_sim module at all.
    assert _dfl_imports(ast.parse(source)) == set()


def test_bayesian_policy_online_path_is_bounded_and_privilege_free() -> None:
    source = POLICY_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert _dfl_imports(tree) <= ALLOWED_DFL_IMPORTS
    assert (
        _boundary_violations(
            source,
            forbidden=BAYESIAN_FORBIDDEN_MODULE_PREFIXES,
            banned=BAYESIAN_BANNED_NAMES,
            forbidden_string_keys=BAYESIAN_FORBIDDEN_STRING_KEYS,
            allowed=POLICY_LOCAL_HYPERPARAMETER_NAMES,
        )
        == []
    )

    reachable, _ = _reachable_definitions(source, sorted(BAYESIAN_POLICY_ROOTS))
    assert reachable <= BAYESIAN_ALLOWED_ONLINE_HELPERS, (
        "Bayesian online path reached un-approved helpers: "
        f"{sorted(reachable - BAYESIAN_ALLOWED_ONLINE_HELPERS)}"
    )
    assert BAYESIAN_REQUIRED_ONLINE_HELPERS <= reachable, (
        "Bayesian online path lost required helpers: "
        f"{sorted(BAYESIAN_REQUIRED_ONLINE_HELPERS - reachable)}"
    )

    # The only artifact leaf surface the policy may import is the approved trio.
    imported: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and _full_module(node) == "dfl_sim.bayesian_artifact"
        ):
            imported.update(alias.name for alias in node.names)
    assert imported == BAYESIAN_ALLOWED_ARTIFACT_IMPORTS


def test_validation_reference_policy_reaches_only_local_seam_helpers() -> None:
    source = POLICY_MODULE.read_text(encoding="utf-8")
    reachable, _ = _reachable_definitions(
        source, sorted(VALIDATION_REFERENCE_ROOTS)
    )
    assert reachable <= VALIDATION_REFERENCE_ALLOWED_HELPERS, (
        "validation reference reached un-approved helpers: "
        f"{sorted(reachable - VALIDATION_REFERENCE_ALLOWED_HELPERS)}"
    )
    assert VALIDATION_REFERENCE_REQUIRED_HELPERS <= reachable, (
        "validation reference lost required helpers: "
        f"{sorted(VALIDATION_REFERENCE_REQUIRED_HELPERS - reachable)}"
    )


def test_validation_reference_boundary_checker_detects_planted_leak() -> None:
    # A validation-reference-shaped implementation that reaches privileged
    # evaluator/attack state must be rejected by the same static checker.
    source = textwrap.dedent(
        """
        from dfl_sim.evaluator import GlobalMetrics

        class ValidationReferencePolicy:
            def select_weights(self, observation):
                labels = observation.evaluator_labels
                kappa = observation.attack_config.kappa
                return labels, kappa
        """
    )
    violations = _boundary_violations(
        source,
        forbidden=BAYESIAN_FORBIDDEN_MODULE_PREFIXES,
        banned=BAYESIAN_BANNED_NAMES,
        forbidden_string_keys=BAYESIAN_FORBIDDEN_STRING_KEYS,
    )
    assert any("dfl_sim.evaluator" in v for v in violations)
    assert any("evaluator_labels" in v for v in violations)
    assert any("attack_config" in v for v in violations)
    assert any("kappa" in v for v in violations)


def test_trust_weighted_policy_reaches_only_local_seam_helpers() -> None:
    source = POLICY_MODULE.read_text(encoding="utf-8")
    reachable, _ = _reachable_definitions(source, sorted(TRUST_WEIGHTED_ROOTS))
    assert reachable <= TRUST_WEIGHTED_ALLOWED_HELPERS, (
        "trust weighted reached un-approved helpers: "
        f"{sorted(reachable - TRUST_WEIGHTED_ALLOWED_HELPERS)}"
    )
    assert TRUST_WEIGHTED_REQUIRED_HELPERS <= reachable, (
        "trust weighted lost required helpers: "
        f"{sorted(TRUST_WEIGHTED_REQUIRED_HELPERS - reachable)}"
    )


def test_balance_policy_reaches_only_local_seam_helpers() -> None:
    source = POLICY_MODULE.read_text(encoding="utf-8")
    reachable, _ = _reachable_definitions(source, sorted(BALANCE_ROOTS))
    assert reachable <= BALANCE_ALLOWED_HELPERS, (
        "balance reached un-approved helpers: "
        f"{sorted(reachable - BALANCE_ALLOWED_HELPERS)}"
    )
    assert BALANCE_REQUIRED_HELPERS <= reachable, (
        "balance lost required helpers: "
        f"{sorted(BALANCE_REQUIRED_HELPERS - reachable)}"
    )


def test_policy_kappa_exemption_does_not_hide_attack_state_reads() -> None:
    # The documented ``kappa`` exemption covers only the local BALANCE decay
    # hyperparameter; a real attack-intensity read still fails closed because
    # ``attack_config`` remains banned.
    source = textwrap.dedent(
        """
        class BalancePolicy:
            def select_weights(self, observation):
                kappa = observation.attack_config.kappa
                return kappa
        """
    )
    violations = _boundary_violations(
        source,
        banned=BAYESIAN_BANNED_NAMES,
        forbidden_string_keys=BAYESIAN_FORBIDDEN_STRING_KEYS,
        allowed=POLICY_LOCAL_HYPERPARAMETER_NAMES,
    )
    assert any("attack_config" in v for v in violations)


def test_trust_and_balance_policies_accept_only_scalar_local_observation() -> None:
    policies = (
        build_policy(TrustWeightedAggregation(beta=1.0)),
        build_policy(BalanceAggregation(), total_rounds=2),
    )
    observation = _fixture_observation(neighbors=(_fixture_neighbor(1),))
    assert _schema_violations(observation) == []
    assert _forbidden_reachable(observation) == []

    for policy in policies:
        signature = inspect.signature(type(policy).select_weights)
        assert list(signature.parameters) == ["self", "observation"]
        profile = policy.select_weights(observation)
        assert isinstance(profile, WeightProfile)
        assert _forbidden_reachable(profile) == []
        for bad in (torch.zeros(2), {"w": torch.zeros(2)}, None, "obs"):
            with pytest.raises(PolicyError):
                policy.select_weights(bad)  # type: ignore[arg-type]


def test_online_modules_cannot_import_or_reach_calibration_symbols() -> None:
    """policy.py and bayesian_artifact.py stay outside the offline calibrator."""

    for path in (POLICY_MODULE, BAYESIAN_ARTIFACT_MODULE):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = _dfl_imports(tree)
        for prefix in CALIBRATION_MODULE_PREFIXES:
            assert prefix not in imports, f"{path.name} imports {prefix}"

        found = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in CALIBRATION_OFFLINE_SYMBOLS
        } | {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr in CALIBRATION_OFFLINE_SYMBOLS
        }
        assert found == set(), (
            f"{path.name} names offline calibration symbols: {sorted(found)}"
        )


def test_calibration_boundary_checker_detects_representative_leaks() -> None:
    source = textwrap.dedent(
        """
        from dfl_sim.calibration_scoring import collect_calibration
        from dfl_sim.calibration_estimator import estimate_artifact
        from dfl_sim.calibration_config import load_calibration_config

        def leak(config, dataset):
            collection = collect_calibration(config, dataset=dataset)
            design_indices = collection.design_indices
            return collection, design_indices
        """
    )
    violations = _boundary_violations(
        source,
        forbidden=CALIBRATION_MODULE_PREFIXES,
        banned=CALIBRATION_OFFLINE_SYMBOLS,
    )
    assert any("dfl_sim.calibration_scoring" in v for v in violations)
    assert any("dfl_sim.calibration_estimator" in v for v in violations)
    assert any("dfl_sim.calibration_config" in v for v in violations)
    assert any("collect_calibration" in v for v in violations)
    assert any("design_indices" in v for v in violations)


def test_bayesian_boundary_checker_detects_representative_leaks() -> None:
    source = textwrap.dedent(
        """
        import torch
        from dfl_sim.evaluator import GlobalMetrics
        from dfl_sim.data import load_design_split
        from dfl_sim.models import Model
        from dfl_sim.topology import load_topology

        def leak(state_dict, artifact, observation):
            true_types = state_dict["true_types"]
            metrics = artifact["global_metrics"]
            tensor = state_dict["tensor"]
            labels = artifact["labels"]
            return true_types, metrics, tensor, labels
        """
    )
    violations = _boundary_violations(
        source,
        forbidden=BAYESIAN_FORBIDDEN_MODULE_PREFIXES,
        banned=BAYESIAN_BANNED_NAMES,
        forbidden_string_keys=BAYESIAN_FORBIDDEN_STRING_KEYS,
    )
    assert any("import torch" in v for v in violations)
    assert any("dfl_sim.evaluator" in v for v in violations)
    assert any("dfl_sim.data" in v for v in violations)
    assert any("dfl_sim.models" in v for v in violations)
    assert any("dfl_sim.topology" in v for v in violations)
    assert any("state_dict" in v for v in violations)
    assert any("true_types" in v for v in violations)
    assert any("global_metrics" in v for v in violations)
    # Privileged raw-dict string keys are caught even when spelled as literals.
    assert any("forbidden string key" in v for v in violations)


# --------------------------------------------------------------------------- #
# Fixture-built Bayesian policy runtime capability walk
# --------------------------------------------------------------------------- #
def _fixture_bayesian_config() -> BayesianAggregation:
    return BayesianAggregation(
        artifact_dir=str(BAYESIAN_FIXTURE),
        omega=0.5,
        degree_cap=4,
        tie_tolerance=1.0e-9,
    )


def _fixture_neighbor(
    sender: int = 1,
    *,
    disagreement: float = 0.7,
    relative_norm: float = 0.5,
    loss_delta: float = -0.5,
) -> NeighborObservation:
    return NeighborObservation(
        sender_id=sender,
        directional_disagreement=disagreement,
        relative_norm=relative_norm,
        loss_delta=loss_delta,
    )


def _fixture_observation(
    *,
    node_id: int = 0,
    round_index: int = 1,
    neighbors: Sequence[NeighborObservation] = (),
) -> LocalObservation:
    return LocalObservation(
        node_id=node_id,
        round_index=round_index,
        epsilon=DEFAULT_EPSILON,
        neighbors=tuple(neighbors),
    )


class _BayesianRecorder:
    """Record the real policy's observations while delegating every call."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.seen: list[LocalObservation] = []

    def select_weights(self, observation: LocalObservation) -> WeightProfile:
        self.seen.append(observation)
        return self._inner.select_weights(observation)


class _PrivilegedSentinel:
    """Unused privileged state, used to prove the seam rejects enrichment."""

    def __init__(self) -> None:
        self.ground_truth = "benign"
        self.compromised_node = 1
        self.global_metrics = {"accuracy": 1.0}
        self.model = nn.Linear(1, 1)
        self.dataset = MNISTShapedDataset(8, seed=7)
        self.topology = nx.Graph()
        self.tensor = torch.zeros(2)


def test_fixture_built_bayesian_policy_accepts_only_scalar_local_observation() -> None:
    policy = build_policy(_fixture_bayesian_config(), total_rounds=2)
    assert isinstance(policy, BayesianPolicy)
    # Construction-time state is exactly the frozen artifact, the pinned run
    # length, and an optional observability sink -- never a parameter of
    # ``select_weights``.
    assert tuple(f.name for f in dataclasses.fields(BayesianPolicy)) == (
        "artifact",
        "total_rounds",
        "diagnostic_sink",
        "clock",
    )
    signature = inspect.signature(type(policy).select_weights)
    assert list(signature.parameters) == ["self", "observation"]

    observation = _fixture_observation(neighbors=(_fixture_neighbor(1),))
    assert _schema_violations(observation) == []
    assert _forbidden_reachable(observation) == []

    profile = policy.select_weights(observation)
    assert isinstance(profile, WeightProfile)
    assert profile.participant_ids == (0, 1)
    assert profile.multipliers[0] == 1.0
    assert abs(sum(profile.weights) - 1.0) <= 1e-9
    assert _forbidden_reachable(profile) == []

    # Non-LocalObservation inputs fail closed, including duck-typed and
    # privileged-enriched lookalikes and any model tensor / raw state dict.
    sentinel = _PrivilegedSentinel()
    assert _forbidden_reachable(sentinel)  # the sentinel really carries privilege
    duck = pytypes.SimpleNamespace(
        node_id=0,
        round_index=1,
        epsilon=DEFAULT_EPSILON,
        neighbors=(_fixture_neighbor(1),),
        ground_truth="benign",
        evaluator=sentinel,
    )
    for bad in (duck, sentinel, torch.zeros(2), {"w": torch.zeros(2)}, None, "obs"):
        with pytest.raises(PolicyError):
            policy.select_weights(bad)  # type: ignore[arg-type]


def test_enriched_local_observation_subclass_is_rejected() -> None:
    policy = build_policy(_fixture_bayesian_config(), total_rounds=2)
    sentinel = _PrivilegedSentinel()

    class EnrichedObservation(LocalObservation):
        pass

    enriched = EnrichedObservation(
        node_id=0,
        round_index=1,
        epsilon=DEFAULT_EPSILON,
        neighbors=(_fixture_neighbor(1),),
    )
    object.__setattr__(enriched, "ground_truth", "benign")
    object.__setattr__(enriched, "evaluator", sentinel)

    # The subclass would otherwise smuggle an unused privileged graph past the
    # seam; the exact-type guard refuses it.
    assert _forbidden_reachable(enriched)
    with pytest.raises(PolicyError):
        policy.select_weights(enriched)


def test_bayesian_selection_diagnostics_expose_only_bounded_fields() -> None:
    records: list[BayesianDiagnostic] = []
    artifact = load_bayesian_artifact(BAYESIAN_FIXTURE)
    assert isinstance(artifact, BayesianArtifact)
    policy = build_bayesian_policy(
        artifact, total_rounds=2, diagnostic_sink=records.append
    )

    # Round 1 resolves the fixture's only covered progress bin -> success.
    success_profile = policy.select_weights(
        _fixture_observation(round_index=1, neighbors=(_fixture_neighbor(1),))
    )
    success = records[-1]
    assert success.fallback_reason is None
    assert success.chosen_multipliers == success_profile.multipliers
    assert success.candidate_count == 3  # |{0, omega, 1}| ** degree 1
    assert success.pruned is False
    assert success.degree_bin == "1"
    assert success.progress == "mid"

    # A missing likelihood cell (uncovered evidence bin) falls back explicitly.
    policy.select_weights(
        _fixture_observation(
            round_index=1,
            neighbors=(
                _fixture_neighbor(1, relative_norm=2.0, loss_delta=0.5),
            ),
        )
    )
    assert records[-1].fallback_reason == FALLBACK_MISSING_LIKELIHOOD

    # An uncovered degree (3) has no prior cells -> explicit missing prior.
    policy.select_weights(
        _fixture_observation(
            round_index=1,
            neighbors=(
                _fixture_neighbor(1),
                _fixture_neighbor(2),
                _fixture_neighbor(3),
            ),
        )
    )
    assert records[-1].fallback_reason == FALLBACK_MISSING_PRIOR

    # A round index outside the run length has no progress context.
    policy.select_weights(
        _fixture_observation(round_index=5, neighbors=(_fixture_neighbor(1),))
    )
    assert records[-1].fallback_reason == FALLBACK_CONTEXT_UNAVAILABLE

    for record in records:
        assert (
            tuple(f.name for f in dataclasses.fields(record))
            == BAYESIAN_DIAGNOSTIC_FIELDS
        )
        assert record.fallback_reason is None or (
            record.fallback_reason in ALLOWED_FALLBACK_REASONS
        )
        for entry in record.posterior:
            assert tuple(f.name for f in dataclasses.fields(entry)) == (
                "state",
                "probability",
            )
            assert isinstance(entry.state, tuple)
            assert all(isinstance(value, int) for value in entry.state)
            assert isinstance(entry.probability, float)
        assert _forbidden_reachable(record) == []
        for name in BAYESIAN_BANNED_NAMES:
            assert not hasattr(record, name)


def test_bayesian_policy_receives_clean_observations_through_real_round_engine() -> None:
    records: list[BayesianDiagnostic] = []
    artifact = load_bayesian_artifact(BAYESIAN_FIXTURE)
    inner = build_bayesian_policy(
        artifact, total_rounds=2, diagnostic_sink=records.append
    )
    recorder = _BayesianRecorder(inner)

    _run(policy=recorder)

    assert len(recorder.seen) == 3
    for observation in recorder.seen:
        assert _schema_violations(observation) == []
        assert _forbidden_reachable(observation) == []

    assert len(records) == 3
    for record in records:
        assert (
            tuple(f.name for f in dataclasses.fields(record))
            == BAYESIAN_DIAGNOSTIC_FIELDS
        )
        assert _forbidden_reachable(record) == []


def test_generated_calibration_artifact_drives_only_scalar_loader_policy(
    tmp_path: Path,
) -> None:
    """A real published artifact feeds only the existing scalar loader/policy.

    Builds the artifact through the real offline pipeline from the checked-in
    config, loads it back through the S01 loader, and drives the Bayesian policy
    through the real round engine. The recorder proves the policy still receives
    exactly the frozen scalar ``LocalObservation`` and that no calibration
    design data, hidden type, or tensor is reachable from the observation,
    profile, or diagnostics.
    """

    config = load_calibration_config(CALIBRATION_CONFIG_PATH)
    dataset = MNISTShapedDataset(64, separable=True, seed=11)
    collection = collect_calibration(config, dataset=dataset)
    estimated = estimate_calibration(config, collection)
    published = publish_calibration_artifact(tmp_path / "artifact", estimated)

    loaded = load_bayesian_artifact(published.path)
    assert isinstance(loaded, BayesianArtifact)
    assert loaded.manifest.aggregate_digest == published.aggregate_digest
    # The frozen artifact reaches no privileged runtime object: no dataset,
    # experiment/attack config, graph, module, or tensor.
    leaked = _forbidden_reachable(loaded)
    assert not any(
        kind.startswith(
            (
                "torch.Tensor",
                "Dataset:",
                "ExperimentConfig",
                "AttackConfig",
                "Module:",
                "Graph:",
            )
        )
        for kind in leaked
    ), leaked

    records: list[BayesianDiagnostic] = []
    inner = build_bayesian_policy(
        loaded,
        total_rounds=config.experiment.rounds,
        diagnostic_sink=records.append,
    )
    recorder = _BayesianRecorder(inner)

    run_from_config(
        config.experiment,
        dataset=dataset,
        trainer=_offset_trainer(),
        policy=recorder,
    )

    assert len(recorder.seen) == 3 * config.experiment.rounds
    for observation in recorder.seen:
        assert _schema_violations(observation) == []
        assert _forbidden_reachable(observation) == []
    for record in records:
        assert (
            tuple(f.name for f in dataclasses.fields(record))
            == BAYESIAN_DIAGNOSTIC_FIELDS
        )
        assert _forbidden_reachable(record) == []


def test_bayesian_config_artifact_and_observation_reject_mutation() -> None:
    observation = _fixture_observation(neighbors=(_fixture_neighbor(1),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        observation.neighbors[0].loss_delta = 9.0  # type: ignore[misc]

    config = _fixture_bayesian_config()
    with pytest.raises(ValidationError):
        config.omega = 0.25  # type: ignore[misc]

    artifact = load_bayesian_artifact(BAYESIAN_FIXTURE)
    with pytest.raises(ValidationError):
        artifact.manifest.action_space.omega = 0.25  # type: ignore[misc]
    with pytest.raises(TypeError):
        artifact.prior_index[1] = ()  # type: ignore[index]
