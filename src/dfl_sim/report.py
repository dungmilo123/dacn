"""Strict, independently reconstructed canonical evidence and metrics (M003/S05/T05).

This module owns the **canonical report** the S05 demo promises. It consumes a
completed, independently audited raw robustness matrix (T04), freshly re-audits
it, re-audits the separate bounded IID headline, and re-reads every raw trace,
cost sidecar and decision-audit sidecar to compile one strict, versioned JSON
document. It never trusts a runner success flag or a stored summary: every
number in the report is recomputed from the validated sources, every block
states its ``real_mnist`` / ``injected`` provenance, and every cited run is
linked by a relative path confined to a declared verified source root.

The report separates three evidence classes that must not be conflated:

* **derivation** -- dataset/partition identity, calibration provenance and the
  artifact assumptions a posterior answer depends on;
* **solution_quality** -- the *finite-action* audit: candidate enumeration,
  degree-cap pruning, fallback and the posterior-weighted expected-payoff
  argmax recomputed independently from the pinned artifact tables;
* **defense_usefulness** -- the *observed final outcomes* plus separate benign
  exclusion and malicious rejection/acceptance accounting and propagation.

Computation and logical communication are reported in separate blocks with
explicit units and semantics. Timings are never used to tune selection, and a
logical-payload byte count is never presented as a network-traffic claim.

The public real-MNIST path refuses injected inputs (:func:`compile_canonical_report`
without an explicit internal fixture seam). Read-only verification
(:func:`verify_canonical_report`) recompiles the report and compares digests
without mutating any source; no report is published if any source is missing,
mutated, foreign, or unsafe.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .bayesian_artifact import (
    BayesianArtifact,
    BayesianArtifactError,
    interval_bin_index,
    load_bayesian_artifact,
)
from .costs import (
    CostError,
    load_cost_payload,
    load_decision_audit_records,
)
from .evaluation_config import parse_held_out_study_text
from .robustness_config import CanonicalRobustnessProtocol
from .robustness_study import (
    CONDITIONS_DIRNAME,
    MATRIX_MANIFEST_FILENAME,
    ROBUSTNESS_PROTOCOL_FILENAME,
    RobustnessProtocol,
    canonical_protocol,
)
from .robustness_verify import (
    MATRIX_EVIDENCE_FILENAME,
    RobustnessVerifyDeps,
    RobustnessVerifyError,
    verify_robustness_study,
)
from .trajectory import SCHEMA_VERSION, TraceReader, TrajectoryError

__all__ = [
    "CANONICAL_REPORT_FILENAME",
    "CANONICAL_REPORT_SCHEMA",
    "CANONICAL_REPORT_SCHEMA_VERSION",
    "CanonicalReportDeps",
    "CanonicalReportError",
    "CanonicalReportResult",
    "audit_bayesian_decisions",
    "build_report_payload",
    "compile_canonical_report",
    "paired_policy_deltas",
    "reduce_final_aggregates",
    "reduce_propagation",
    "reduce_rejection",
    "validate_canonical_report",
    "verify_canonical_report",
]

#: Versioned canonical report identity.
CANONICAL_REPORT_SCHEMA = "dfl_canonical_report"
CANONICAL_REPORT_SCHEMA_VERSION = 1
CANONICAL_REPORT_FILENAME = "canonical-report.json"

#: Evidence provenance labels used by every block.
PROVENANCE = ("real_mnist", "injected")

#: Predeclared paired references (full Bayesian + plain average).
PAIR_REFERENCES: tuple[str, ...] = ("plain_average", "bayesian")

#: The bounded matrix scope statement (nine WS / non-IID conditions).
_MATRIX_SCOPE_KIND = "watts_strogatz_non_iid_alpha_x_placement"
_HEADLINE_SCOPE_KIND = "iid_chain3_bounded"

_MAX_ERROR_TEXT = 400
_TIE_TOLERANCE = 1.0e-9

_LIMITATIONS: tuple[str, ...] = (
    "Finite-action solution quality only: no equilibrium, regret bound or "
    "attacker-response solver is claimed.",
    "The three final seeds give a population spread and IQR, not a confidence "
    "interval or a winner claim.",
    "Degree-cap pruning and missing/unsupported artifact support are disclosed "
    "per decision row and are not fabricated into full-action optimality.",
    "The no_strategic calibration variant fixes kappa and is an assumption, "
    "not a measured strategic response.",
    "Communication is the actual logical uncompressed tensor payload and "
    "directed message count before receiver filtering; it is not a physical "
    "network-traffic measurement.",
    "Injected fixture provenance proves mechanism only and is never real-MNIST "
    "evidence.",
    "The bounded IID headline is scoped separately and is never pooled with "
    "the WS/non-IID matrix conditions.",
)

_DEGREE_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_DEGREE_PLUS_RE = re.compile(r"^(\d+)\s*\+$")


class CanonicalReportError(RuntimeError):
    """Raised when the canonical report cannot be compiled fail-closed."""


# --------------------------------------------------------------------------- #
# Errors / primitives
# --------------------------------------------------------------------------- #
def _bounded(text: object, *, limit: int = _MAX_ERROR_TEXT) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) > limit:
        return collapsed[: limit - 1] + "\u2026"
    return collapsed


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise CanonicalReportError(detail)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(payload: object) -> str:
    return _sha256_bytes(_canonical_bytes(payload))


def _reject_constant(value: str) -> Any:
    raise CanonicalReportError(f"non-finite JSON constant {value!r} is not allowed")


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise CanonicalReportError(f"missing report source: {path}")
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except CanonicalReportError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise CanonicalReportError(f"could not read {path}: {_bounded(exc)}") from exc
    if not isinstance(raw, Mapping):
        raise CanonicalReportError(f"{path} is not a JSON object")
    return raw


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _confined(root: Path, relative: object, *, label: str) -> Path:
    """Resolve ``relative`` under ``root`` and reject traversal/symlink escape."""

    if not isinstance(relative, str) or not relative:
        raise CanonicalReportError(f"{label} path must be a non-empty string")
    if os.path.isabs(relative):
        raise CanonicalReportError(f"{label} path must be relative, got {relative!r}")
    try:
        real_root = os.path.realpath(root)
        real = os.path.realpath(os.path.join(real_root, relative))
    except OSError as exc:  # pragma: no cover - defensive
        raise CanonicalReportError(f"could not resolve {label} path: {exc}") from exc
    if real != real_root and not real.startswith(real_root + os.sep):
        raise CanonicalReportError(f"{label} path escapes the declared root")
    return Path(real)


def _relative_within(root: Path, path: Path, *, label: str) -> str:
    real_root = os.path.realpath(root)
    real = os.path.realpath(path)
    if real != real_root and not real.startswith(real_root + os.sep):
        raise CanonicalReportError(f"{label} path escapes the declared root")
    return os.path.relpath(real, real_root).replace(os.sep, "/")


# --------------------------------------------------------------------------- #
# Pure statistics reducers
# --------------------------------------------------------------------------- #
def _population_std(values: Sequence[float]) -> float:
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return math.sqrt(variance)


def _linear_quantile(ordered: Sequence[float], q: float) -> float:
    count = len(ordered)
    if count == 1:
        return float(ordered[0])
    position = q * (count - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    fraction = position - lower
    return float(ordered[lower]) + fraction * (
        float(ordered[upper]) - float(ordered[lower])
    )


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanonicalReportError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise CanonicalReportError(f"{label} must be finite")
    return result


def _nonneg_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CanonicalReportError(f"{label} must be an int")
    if value < 0:
        raise CanonicalReportError(f"{label} must be non-negative")
    return int(value)


def reduce_final_aggregates(cells: Sequence[Mapping[str, object]]) -> list[dict]:
    """Reduce final-only per-cell metrics to per-(condition,policy,schedule).

    Every seed value is retained; the mean, population standard deviation,
    linearly interpolated IQR, mean loss and global minimum node accuracy are
    recomputed from the per-seed values. The seed set is required to be
    identical for every cell of one group, so no unpaired or partial cell can
    inflate a mean.
    """

    buckets: dict[tuple[str, str, str], dict[int, dict]] = {}
    flags: dict[tuple[str, str, str], bool] = {}
    for cell in cells:
        condition_id = str(cell["condition_id"])
        policy_id = str(cell["policy_id"])
        schedule_id = str(cell["schedule_id"])
        key = (condition_id, policy_id, schedule_id)
        seed = _nonneg_int(cell["seed"], label="seed")
        metrics = cell["metrics"]
        if not isinstance(metrics, Mapping):
            raise CanonicalReportError("cell metrics must be a mapping")
        accuracy = _finite_float(metrics["mean_accuracy"], label="mean_accuracy")
        loss = _finite_float(metrics["mean_loss"], label="mean_loss")
        worst = _finite_float(
            metrics["worst_node_accuracy"], label="worst_node_accuracy"
        )
        node_count = _nonneg_int(metrics["node_count"], label="node_count")
        if node_count < 1:
            raise CanonicalReportError("node_count must be positive")
        bucket = buckets.setdefault(key, {})
        if seed in bucket:
            raise CanonicalReportError(
                f"duplicate cell {(condition_id, policy_id, schedule_id, seed)!r}"
            )
        bucket[seed] = {
            "mean_accuracy": accuracy,
            "mean_loss": loss,
            "worst_node_accuracy": worst,
            "node_count": node_count,
        }
        held_out = bool(cell.get("held_out", False))
        previous = flags.setdefault(key, held_out)
        if previous != held_out:
            raise CanonicalReportError(
                f"inconsistent held_out flag for {(condition_id, policy_id, schedule_id)!r}"
            )

    aggregates: list[dict] = []
    for key in sorted(buckets):
        condition_id, policy_id, schedule_id = key
        bucket = buckets[key]
        seeds = sorted(bucket)
        accuracies = [bucket[seed]["mean_accuracy"] for seed in seeds]
        losses = [bucket[seed]["mean_loss"] for seed in seeds]
        node_counts = {bucket[seed]["node_count"] for seed in seeds}
        if len(node_counts) != 1:
            raise CanonicalReportError(
                f"inconsistent node counts for {key!r}: {sorted(node_counts)!r}"
            )
        ordered = sorted(accuracies)
        aggregates.append(
            {
                "condition_id": condition_id,
                "policy_id": policy_id,
                "schedule_id": schedule_id,
                "held_out": flags[key],
                "provenance": str(cell_provenance(cells, condition_id, policy_id)),
                "seeds": seeds,
                "count": len(seeds),
                "node_count": node_counts.pop(),
                "mean_accuracy": sum(accuracies) / len(seeds),
                "std_accuracy": _population_std(accuracies),
                "iqr_accuracy": _linear_quantile(ordered, 0.75)
                - _linear_quantile(ordered, 0.25),
                "mean_loss": sum(losses) / len(seeds),
                "worst_node_accuracy": min(
                    bucket[seed]["worst_node_accuracy"] for seed in seeds
                ),
                "per_seed": [
                    {
                        "seed": seed,
                        "mean_accuracy": bucket[seed]["mean_accuracy"],
                        "mean_loss": bucket[seed]["mean_loss"],
                        "worst_node_accuracy": bucket[seed]["worst_node_accuracy"],
                    }
                    for seed in seeds
                ],
            }
        )
    return aggregates


def cell_provenance(
    cells: Sequence[Mapping[str, object]], condition_id: str, policy_id: str
) -> str:
    for cell in cells:
        if (
            str(cell["condition_id"]) == condition_id
            and str(cell["policy_id"]) == policy_id
        ):
            return str(cell.get("provenance", "injected"))
    return "injected"


def paired_policy_deltas(
    aggregates: Sequence[Mapping[str, object]],
    *,
    references: Sequence[str] = PAIR_REFERENCES,
) -> list[dict]:
    """Paired per-seed accuracy deltas on identical (condition, schedule, seed).

    For each declared reference policy, every other policy's per-seed accuracy
    is differenced against the reference on the *same* (condition, schedule,
    seed) key, so no unpaired seed can enter a delta. The mean and population
    spread of the seed deltas are reported without any confidence or winner
    claim.
    """

    lookup: dict[tuple[str, str, str], dict[int, float]] = {}
    for aggregate in aggregates:
        key = (
            str(aggregate["condition_id"]),
            str(aggregate["policy_id"]),
            str(aggregate["schedule_id"]),
        )
        lookup[key] = {
            int(entry["seed"]): _finite_float(
                entry["mean_accuracy"], label="paired accuracy"
            )
            for entry in aggregate["per_seed"]
        }

    keys: dict[tuple[str, str], list[str]] = {}
    for condition_id, policy_id, schedule_id in lookup:
        keys.setdefault((condition_id, schedule_id), []).append(policy_id)
    for bucket in keys.values():
        bucket.sort()

    deltas: list[dict] = []
    for (condition_id, schedule_id), policies in sorted(keys.items()):
        for reference in references:
            reference_key = (condition_id, reference, schedule_id)
            if reference_key not in lookup:
                continue
            reference_seeds = lookup[reference_key]
            for policy_id in policies:
                if policy_id == reference:
                    continue
                policy_seeds = lookup[(condition_id, policy_id, schedule_id)]
                if set(policy_seeds) != set(reference_seeds):
                    raise CanonicalReportError(
                        f"paired seed mismatch for {condition_id!r} "
                        f"{policy_id!r} vs {reference!r} schedule {schedule_id!r}"
                    )
                seeds = sorted(policy_seeds)
                per_seed = [
                    {
                        "seed": seed,
                        "delta": policy_seeds[seed] - reference_seeds[seed],
                    }
                    for seed in seeds
                ]
                values = [entry["delta"] for entry in per_seed]
                ordered = sorted(values)
                deltas.append(
                    {
                        "condition_id": condition_id,
                        "schedule_id": schedule_id,
                        "reference_policy": reference,
                        "policy_id": policy_id,
                        "seeds": seeds,
                        "count": len(seeds),
                        "mean_delta": sum(values) / len(values),
                        "std_delta": _population_std(values),
                        "iqr_delta": _linear_quantile(ordered, 0.75)
                        - _linear_quantile(ordered, 0.25),
                        "per_seed": per_seed,
                    }
                )
    return deltas


# --------------------------------------------------------------------------- #
# Rejection / acceptance reducer
# --------------------------------------------------------------------------- #
_REJECTION_COUNTERS = (
    "delivered_opportunities",
    "benign_sender_opportunities",
    "benign_excluded",
    "malicious_structural_opportunities",
    "malicious_active_opportunities",
    "malicious_active_rejected",
    "malicious_active_accepted",
    "malicious_inactive_opportunities",
    "malicious_inactive_rejected",
    "malicious_inactive_accepted",
)


def _empty_rejection() -> dict:
    return {name: 0 for name in _REJECTION_COUNTERS}


def reduce_rejection(events: Sequence[Mapping[str, object]], *, label: str) -> dict:
    """Count benign exclusion and malicious rejection/acceptance opportunities.

    A *delivered message opportunity* is one directed edge into a benign
    receiver, excluding self: the receiver selected a weight for each sender in
    its profile. A zero (or absent) weight is an exclusion/rejection; a positive
    weight is contribution/acceptance. ``local_only`` and continuous weighting
    therefore show up honestly as zero contribution rather than being relabelled
    detector positives. A malicious sender is *active* exactly when its realized
    ``kappa > 0`` in that round; a structurally malicious sender with kappa 0 is
    counted separately as inactive.
    """

    counters = _empty_rejection()
    for event in events:
        nodes = event.get("nodes")
        if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
            raise CanonicalReportError(f"{label} round has no node records")
        by_id = {int(node["node"]): node for node in nodes}
        for receiver in nodes:
            if str(receiver.get("true_type")) != "benign":
                continue
            profile = receiver.get("profile")
            if not isinstance(profile, Mapping):
                raise CanonicalReportError(f"{label} receiver has no profile")
            participants = profile.get("participants")
            weights = profile.get("weights")
            if (
                not isinstance(participants, Sequence)
                or not isinstance(weights, Sequence)
                or len(participants) != len(weights)
            ):
                raise CanonicalReportError(f"{label} profile alignment is invalid")
            for index in range(1, len(participants)):
                sender = int(participants[index])
                weight = _finite_float(weights[index], label="sender weight")
                if weight < 0.0:
                    raise CanonicalReportError(f"{label} negative sender weight")
                sender_node = by_id.get(sender)
                if sender_node is None:
                    raise CanonicalReportError(
                        f"{label} receiver cites absent sender {sender}"
                    )
                counters["delivered_opportunities"] += 1
                sender_type = str(sender_node.get("true_type"))
                attack = sender_node.get("attack")
                kappa = _finite_float(
                    (attack or {}).get("kappa", 0.0), label="sender kappa"
                )
                positive = weight > 0.0
                if sender_type == "benign":
                    counters["benign_sender_opportunities"] += 1
                    if not positive:
                        counters["benign_excluded"] += 1
                elif sender_type == "malicious":
                    counters["malicious_structural_opportunities"] += 1
                    if kappa > 0.0:
                        counters["malicious_active_opportunities"] += 1
                        if positive:
                            counters["malicious_active_accepted"] += 1
                        else:
                            counters["malicious_active_rejected"] += 1
                    else:
                        counters["malicious_inactive_opportunities"] += 1
                        if positive:
                            counters["malicious_inactive_accepted"] += 1
                        else:
                            counters["malicious_inactive_rejected"] += 1
                else:
                    raise CanonicalReportError(
                        f"{label} sender {sender} has unknown type {sender_type!r}"
                    )
    return counters


def _merge_rejection(target: dict, source: Mapping[str, object]) -> None:
    for name in _REJECTION_COUNTERS:
        target[name] += int(source[name])


def _rejection_rates(counters: Mapping[str, object]) -> dict:
    benign = int(counters["benign_sender_opportunities"])
    active = int(counters["malicious_active_opportunities"])
    structural = int(counters["malicious_structural_opportunities"])
    return {
        "benign_exclusion_rate": (
            None if benign == 0 else int(counters["benign_excluded"]) / benign
        ),
        "malicious_active_rejection_rate": (
            None
            if active == 0
            else int(counters["malicious_active_rejected"]) / active
        ),
        "malicious_active_acceptance_rate": (
            None
            if active == 0
            else int(counters["malicious_active_accepted"]) / active
        ),
        "malicious_structural_rejection_rate": (
            None
            if structural == 0
            else (
                int(counters["malicious_inactive_rejected"])
                + int(counters["malicious_active_rejected"])
            )
            / structural
        ),
    }


def reduce_propagation(events: Sequence[Mapping[str, object]], *, label: str) -> dict:
    """Summarize weight-aware poison propagation straight from trace fields.

    ``reached`` is monotone; ``min_hop`` counts positive aggregation edges from
    the active seed (hop 0) and ``first_reached_round`` is the earliest committed
    round at which the node became reached. A bounded list of
    ``(round, node, min_hop)`` source links is retained for audit.
    """

    reached_node_rounds = 0
    total_node_rounds = 0
    reached_nodes: set[int] = set()
    hops: list[int] = []
    first_rounds: list[int] = []
    sources: list[dict] = []
    for event in events:
        nodes = event.get("nodes")
        if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
            raise CanonicalReportError(f"{label} round has no node records")
        round_index = _nonneg_int(event.get("round"), label="round")
        for node in nodes:
            total_node_rounds += 1
            propagation = node.get("propagation")
            if not isinstance(propagation, Mapping):
                raise CanonicalReportError(f"{label} node has no propagation state")
            if not bool(propagation.get("reached")):
                continue
            node_id = int(node["node"])
            reached_node_rounds += 1
            reached_nodes.add(node_id)
            hop = propagation.get("min_hop")
            first = propagation.get("first_reached_round")
            if hop is None or first is None:
                raise CanonicalReportError(f"{label} reached node lacks hop/first")
            hops.append(int(hop))
            first_rounds.append(int(first))
            if len(sources) < 32:
                sources.append(
                    {
                        "round": round_index,
                        "node": node_id,
                        "min_hop": int(hop),
                        "first_reached_round": int(first),
                    }
                )
    return {
        "reached_node_rounds": reached_node_rounds,
        "total_node_rounds": total_node_rounds,
        "reached_nodes_ever": len(reached_nodes),
        "min_hop": min(hops) if hops else None,
        "max_hop": max(hops) if hops else None,
        "first_reached_rounds": sorted(set(first_rounds)),
        "sources": sources,
    }


# --------------------------------------------------------------------------- #
# Independent Bayesian solution-quality audit
# --------------------------------------------------------------------------- #
def _local_resolve_degree_bin(labels: Sequence[str], degree: int) -> str | None:
    for label in labels:
        candidate = label.strip()
        if candidate.isdigit():
            if int(candidate) == degree:
                return label
            continue
        match = _DEGREE_RANGE_RE.match(candidate)
        if match is not None and int(match.group(1)) <= degree <= int(match.group(2)):
            return label
        match = _DEGREE_PLUS_RE.match(candidate)
        if match is not None and degree >= int(match.group(1)):
            return label
    return None


def _local_resolve_progress_bin(
    labels: Sequence[str], round_index: int, total_rounds: int
) -> str | None:
    if total_rounds < 1 or not labels:
        return None
    if round_index < 0 or round_index >= total_rounds:
        return None
    index = (round_index * len(labels)) // total_rounds
    if index >= len(labels):
        index = len(labels) - 1
    return labels[index]


def _local_action_sort_key(action: tuple[float, ...]) -> tuple:
    rejected = sum(1 for multiplier in action if multiplier == 0.0)
    return (rejected, action)


def _local_enumerate_candidates(
    degree: int,
    multiplier_set: Sequence[float],
    *,
    degree_cap: int,
    disagreements: Sequence[float],
    sender_ids: Sequence[int],
) -> tuple[tuple[float, ...], ...]:
    """Independent D022 candidate enumeration (pruned above the degree cap)."""

    import itertools

    multipliers = tuple(sorted(float(value) for value in multiplier_set))
    if degree == 0:
        return ((),)
    if degree <= degree_cap:
        return tuple(
            sorted(
                (tuple(action) for action in itertools.product(multipliers, repeat=degree)),
                key=_local_action_sort_key,
            )
        )
    order = sorted(
        range(degree),
        key=lambda index: (-float(disagreements[index]), int(sender_ids[index])),
    )
    positions = tuple(sorted(order[:degree_cap]))
    actions: list[tuple[float, ...]] = []
    for combination in itertools.product(multipliers, repeat=len(positions)):
        vector = [1.0] * degree
        for position, multiplier in zip(positions, combination):
            vector[position] = multiplier
        actions.append(tuple(vector))
    return tuple(sorted(actions, key=_local_action_sort_key))


def _local_compute_posterior(
    artifact: BayesianArtifact,
    *,
    degree: int,
    degree_bin: str,
    progress: str,
    evidence_bins: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[tuple[int, ...], float], ...] | None, str | None]:
    cells = artifact.prior_for_degree(degree)
    if not cells:
        return None, "missing_prior"
    weights: list[float] = []
    for cell in cells:
        product = 1.0
        for theta, evidence in zip(cell.state, evidence_bins):
            likelihood = artifact.likelihood_for(
                degree_bin, progress, int(theta), tuple(int(v) for v in evidence)
            )
            if likelihood is None:
                return None, "missing_likelihood"
            product *= float(likelihood)
            if not math.isfinite(product):
                return None, "non_finite"
        weight = float(cell.probability) * product
        if not math.isfinite(weight):
            return None, "non_finite"
        weights.append(weight)
    mass = math.fsum(weights)
    if not math.isfinite(mass) or mass <= 0.0:
        return None, "zero_mass"
    return (
        tuple(
            (tuple(int(v) for v in cell.state), weight / mass)
            for cell, weight in zip(cells, weights)
        ),
        None,
    )


def _local_expected_payoffs(
    artifact: BayesianArtifact,
    *,
    degree_bin: str,
    progress: str,
    evidence_bins: Sequence[Sequence[int]],
    posterior: Sequence[tuple[tuple[int, ...], float]],
    actions: Sequence[tuple[float, ...]],
) -> tuple[tuple[float, ...] | None, str | None]:
    values: list[float] = []
    for action in actions:
        expected = 0.0
        for state, probability in posterior:
            total = 0.0
            for theta, evidence, multiplier in zip(state, evidence_bins, action):
                value = artifact.payoff_for(
                    degree_bin,
                    progress,
                    int(theta),
                    tuple(int(v) for v in evidence),
                    float(multiplier),
                )
                if value is None:
                    return None, "missing_payoff"
                total += float(value)
            if not math.isfinite(total):
                return None, "non_finite"
            expected += float(probability) * total
            if not math.isfinite(expected):
                return None, "non_finite"
        values.append(expected)
    return tuple(values), None


def _local_argmax(values: Sequence[float], tolerance: float) -> int:
    best_index = 0
    best_value = values[0]
    for index in range(1, len(values)):
        if values[index] > best_value + tolerance:
            best_value = values[index]
            best_index = index
    return best_index


def audit_bayesian_decisions(
    artifact: BayesianArtifact,
    records: Sequence[Mapping[str, object]],
    *,
    total_rounds: int,
    policy_id: str,
    provenance: str,
) -> dict:
    """Independently reconstruct and check every Bayesian selection row.

    For each decision-audit row the posterior is recomputed from the recorded
    evidence cells and the pinned artifact prior/likelihood tables, the finite
    candidate set is re-enumerated under the artifact degree cap, and every
    candidate's posterior-weighted expected payoff is recomputed from the
    pinned payoff table. The recorded posterior and chosen action must agree
    within the artifact tie tolerance or the audit fails closed.
    """

    from .policy import enumerate_candidate_actions as _production_enumerate

    applicable = 0
    fallback_rows = 0
    unsupported_rows = 0
    recomputed_rows = 0
    posterior_agreement = 0
    action_agreement = 0
    action_mismatch = 0
    candidate_count_mismatch = 0
    pruned_mismatch = 0
    candidate_counts: list[int] = []
    pruned_rows = 0
    min_candidates: int | None = None
    max_candidates: int | None = None

    for record in records:
        degree = _nonneg_int(record.get("degree"), label="decision degree")
        candidate_count = record.get("candidate_count")
        if candidate_count is None:
            continue
        applicable += 1
        candidate_count = _nonneg_int(candidate_count, label="candidate_count")
        round_index = _nonneg_int(record.get("round_index"), label="round_index")
        sender_ids = [int(v) for v in record["sender_ids"]]
        disagreements = [float(v) for v in record["disagreement"]]
        evidence_bins = [tuple(int(v) for v in row) for row in record["evidence_bins"]]
        chosen = tuple(float(v) for v in record["chosen_multipliers"])
        recorded_posterior = tuple(
            (tuple(int(v) for v in entry["state"]), float(entry["probability"]))
            for entry in record["posterior"]
        )
        fallback_reason = record.get("fallback_reason")
        degree_bin = _local_resolve_degree_bin(
            artifact.manifest.bins.degree.labels, degree
        )
        progress = _local_resolve_progress_bin(
            artifact.manifest.bins.progress.labels, round_index, total_rounds
        )
        if degree == 0:
            continue
        if degree_bin is None or progress is None:
            unsupported_rows += 1
            continue
        if fallback_reason is not None:
            fallback_rows += 1
            _require(
                chosen == (1.0,) + (0.0,) * degree,
                f"policy {policy_id!r} fallback did not select self-only",
            )
            continue

        posterior, reason = _local_compute_posterior(
            artifact,
            degree=degree,
            degree_bin=degree_bin,
            progress=progress,
            evidence_bins=evidence_bins,
        )
        if reason is not None:
            unsupported_rows += 1
            continue
        assert posterior is not None
        if len(posterior) != len(recorded_posterior) or any(
            state != rec_state or abs(probability - rec_probability) > 1.0e-9
            for (state, probability), (rec_state, rec_probability) in zip(
                posterior, recorded_posterior
            )
        ):
            raise CanonicalReportError(
                f"policy {policy_id!r} row node={record.get('node_id')} "
                "round=" + str(round_index) + " posterior disagrees with the artifact"
            )
        posterior_agreement += 1

        actions = _local_enumerate_candidates(
            degree,
            artifact.multiplier_set,
            degree_cap=artifact.degree_cap,
            disagreements=disagreements,
            sender_ids=sender_ids,
        )
        production_actions = _production_enumerate(
            degree,
            artifact.multiplier_set,
            degree_cap=artifact.degree_cap,
            disagreements=disagreements,
            sender_ids=sender_ids,
        )
        if tuple(actions) != tuple(production_actions):
            raise CanonicalReportError(
                f"policy {policy_id!r} independent candidate enumeration "
                "disagrees with the production enumeration"
            )
        if len(actions) != candidate_count:
            candidate_count_mismatch += 1
        expected_pruned = degree > artifact.degree_cap
        if bool(record.get("pruned")) != expected_pruned:
            pruned_mismatch += 1
        if expected_pruned:
            pruned_rows += 1

        values, reason = _local_expected_payoffs(
            artifact,
            degree_bin=degree_bin,
            progress=progress,
            evidence_bins=evidence_bins,
            posterior=posterior,
            actions=actions,
        )
        if reason is not None:
            unsupported_rows += 1
            continue
        assert values is not None
        recomputed_rows += 1
        candidate_counts.append(len(actions))
        chosen_neighbors = chosen[1:]
        chosen_index = next(
            (index for index, action in enumerate(actions) if action == chosen_neighbors),
            None,
        )
        best = _local_argmax(values, artifact.tie_tolerance)
        if chosen_index is not None and (
            chosen_index == best
            or abs(values[chosen_index] - values[best]) <= artifact.tie_tolerance
        ):
            action_agreement += 1
        else:
            action_mismatch += 1

    if candidate_counts:
        min_candidates = min(candidate_counts)
        max_candidates = max(candidate_counts)
    if action_mismatch or candidate_count_mismatch or pruned_mismatch:
        raise CanonicalReportError(
            f"policy {policy_id!r} solution-quality audit failed: "
            f"action_mismatch={action_mismatch} "
            f"candidate_count_mismatch={candidate_count_mismatch} "
            f"pruned_mismatch={pruned_mismatch}"
        )
    return {
        "policy_id": policy_id,
        "provenance": provenance,
        "applicable": True,
        "decision_rows": len(records),
        "finite_action_rows": applicable,
        "recomputed_rows": recomputed_rows,
        "posterior_agreement_rows": posterior_agreement,
        "action_agreement_rows": action_agreement,
        "action_mismatch_rows": action_mismatch,
        "fallback_rows": fallback_rows,
        "context_unsupported_rows": unsupported_rows,
        "pruned_rows": pruned_rows,
        "candidate_count_mismatch_rows": candidate_count_mismatch,
        "min_candidates": min_candidates,
        "max_candidates": max_candidates,
        "tie_tolerance": float(artifact.tie_tolerance),
        "degree_cap": int(artifact.degree_cap),
        "artifact_aggregate_digest": str(artifact.manifest.aggregate_digest),
        "artifact_variant": (
            None
            if artifact.manifest.derivation is None
            else str(artifact.manifest.derivation.variant)
        ),
    }


# --------------------------------------------------------------------------- #
# Strict versioned report models
# --------------------------------------------------------------------------- #
class _ReportModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        populate_by_name=True,
    )


class ReportHeadlineScope(_ReportModel):
    kind: str
    data_provenance: str
    verified: bool
    available: bool
    selection_cells: int | None
    final_cells: int | None
    aggregate_count: int | None
    protocol_digest: str | None
    evidence_digest: str | None


class ReportScope(_ReportModel):
    study_id: str
    data_provenance: Literal["real_mnist", "injected"]
    alphas: tuple[float, ...]
    placements: tuple[str, ...]
    policies: tuple[str, ...]
    selection_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    calibration_seed: int
    rounds: int
    samples_per_node: int
    evaluation_max_samples: int
    matrix_selection_cells: int
    matrix_final_cells: int
    matrix_aggregates: int
    headline: ReportHeadlineScope


class ReportSources(_ReportModel):
    protocol_digest: str
    matrix_digest: str
    matrix_manifest_digest: str
    matrix_evidence_digest: str
    headline: dict[str, Any]
    cache: dict[str, Any]
    verified_roots: dict[str, str]


class ReplayLink(_ReportModel):
    policy_id: str
    schedule_id: str
    seed: int
    held_out: bool
    trace: str
    costs: str
    decision_audit: str
    reused: bool


class ConditionRecord(_ReportModel):
    condition_id: str
    alpha: float
    placement: str
    provenance: Literal["real_mnist", "injected"]
    study_digest: str
    held_out_study_digest: str
    frozen_study_digest: str
    selection_output_digest: str
    final_output_digest: str
    policy_ids: tuple[str, ...]
    schedule_ids: tuple[str, ...]
    selection_cells: int
    final_cells: int
    aggregates: int
    messages: int
    payload_bytes: int
    placement_identities: tuple[dict[str, Any], ...]
    replay_links: tuple[ReplayLink, ...]


class ReportEvidence(_ReportModel):
    derivation: dict[str, Any]
    solution_quality: dict[str, Any]
    defense_usefulness: dict[str, Any]


class ReportMetrics(_ReportModel):
    std_method: str
    quantile_method: str
    disclaimer: str
    aggregates: tuple[dict[str, Any], ...]
    paired_deltas: tuple[dict[str, Any], ...]


class ReportCosts(_ReportModel):
    computation: dict[str, Any]
    communication: dict[str, Any]
    calibration_stage_wall: dict[str, Any]
    process_rss: dict[str, Any]
    headline: dict[str, Any]
    resumed_cells: int
    new_cells: int


class ReportRejection(_ReportModel):
    semantics: dict[str, Any]
    by_condition_policy: tuple[dict[str, Any], ...]
    totals: dict[str, Any]


class ReportPropagation(_ReportModel):
    semantics: dict[str, Any]
    by_condition_policy: tuple[dict[str, Any], ...]
    totals: dict[str, Any]


class CanonicalReport(_ReportModel):
    schema_name: Literal["dfl_canonical_report"] = Field(alias="schema")
    schema_version: Literal[1]
    status: Literal["complete"]
    study_id: str
    data_provenance: Literal["real_mnist", "injected"]
    scope: ReportScope
    sources: ReportSources
    conditions: tuple[ConditionRecord, ...]
    evidence: ReportEvidence
    metrics: ReportMetrics
    costs: ReportCosts
    rejection: ReportRejection
    propagation: ReportPropagation
    limitations: tuple[str, ...]
    report_digest: str


def _assert_finite_json(value: object, *, path: str = "$") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalReportError(f"{path} is not finite")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalReportError(f"{path} has a non-string key")
            _assert_finite_json(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_json(item, path=f"{path}[{index}]")
    elif isinstance(value, (str, int, type(None), bool)):
        return
    else:
        raise CanonicalReportError(
            f"{path} has an unsupported {type(value).__name__} value"
        )


def validate_canonical_report(payload: object) -> dict:
    """Strictly validate a canonical report payload, returning its canonical body."""

    _assert_finite_json(payload)
    try:
        report = CanonicalReport.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
        raise CanonicalReportError(f"invalid canonical report: {_bounded(exc)}") from exc
    body = report.model_dump(mode="json", by_alias=True)
    stored = body.pop("report_digest", None)
    _require(_is_sha256(stored), "canonical report is missing its digest")
    _require(
        stored == _sha256_json(body),
        "canonical report digest does not bind its body",
    )
    return report.model_dump(mode="json", by_alias=True)


# --------------------------------------------------------------------------- #
# Injection seams
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CanonicalReportDeps:
    """Optional injection seams; production uses the real public surfaces."""

    verify_matrix: Callable[..., object] | None = None
    headline_verify: Callable[..., object] | None = None
    load_artifact: Callable[[str], object] | None = None
    trace_reader: Callable[[object], Any] | None = None
    verify_deps: RobustnessVerifyDeps | None = None
    repo_root: str | os.PathLike[str] | None = None
    allow_injected: bool = False
    verified_roots: tuple[str, ...] = ()


@dataclass(frozen=True)
class CanonicalReportResult:
    output_path: Path
    report_path: Path | None
    status: str
    report: Mapping[str, object]
    report_digest: str
    conditions: int
    aggregates: int
    paired_deltas: int


# --------------------------------------------------------------------------- #
# Headline / source audit
# --------------------------------------------------------------------------- #
def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(text: str, base: Path) -> Path:
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate


def _audit_headline(
    *,
    protocol: RobustnessProtocol,
    frozen: Mapping[str, object],
    deps: CanonicalReportDeps,
    repo_root: Path,
    verified_roots: dict[str, str],
) -> dict:
    pinned = frozen.get("headline")
    if protocol.data_provenance != "real_mnist":
        return {
            "kind": _HEADLINE_SCOPE_KIND,
            "data_provenance": "injected",
            "verified": False,
            "available": False,
            "selection_cells": None,
            "final_cells": None,
            "aggregate_count": None,
            "protocol_digest": None,
            "evidence_digest": None,
        }
    _require(isinstance(pinned, Mapping), "frozen protocol is missing the headline pin")
    source_root = _resolve_path(protocol.headline_source_root, repo_root)
    _require(source_root.is_dir(), f"headline source root not found: {source_root!r}")
    verify = deps.headline_verify
    if verify is None:
        from . import headline_verify

        verify = headline_verify.verify_headline
    try:
        result = verify(str(source_root), write=False, deep_cache=False)
    except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
        raise CanonicalReportError(
            f"fresh headline audit failed closed: {_bounded(exc)}"
        ) from exc
    evidence = getattr(result, "evidence", None)
    evidence_digest = (
        evidence.get("evidence_digest") if isinstance(evidence, Mapping) else None
    )
    _require(
        pinned.get("protocol_digest") == getattr(result, "protocol_digest", None),
        "headline protocol digest no longer matches the matrix pin",
    )
    _require(
        pinned.get("evidence_digest") == evidence_digest,
        "headline evidence digest no longer matches the matrix pin",
    )
    _require(
        int(pinned.get("selection_cells", -1))
        == int(getattr(result, "selection_cells", -1)),
        "headline selection cell count mismatch",
    )
    _require(
        int(pinned.get("final_cells", -1)) == int(getattr(result, "final_cells", -1)),
        "headline final cell count mismatch",
    )
    _require(pinned.get("scope") == "iid-chain3", "headline scope is not IID")
    verified_roots["headline"] = str(source_root)
    return {
        "kind": _HEADLINE_SCOPE_KIND,
        "data_provenance": "real_mnist",
        "verified": True,
        "available": True,
        "selection_cells": int(getattr(result, "selection_cells", 0)),
        "final_cells": int(getattr(result, "final_cells", 0)),
        "aggregate_count": int(getattr(result, "aggregate_count", 0)),
        "protocol_digest": getattr(result, "protocol_digest", None),
        "evidence_digest": evidence_digest,
    }


# --------------------------------------------------------------------------- #
# Condition loading
# --------------------------------------------------------------------------- #
@dataclass
class _ConditionEvidence:
    record: dict
    cells: list[dict]
    costs: list[dict]
    rejection: dict
    propagation: dict


def _load_panel_artifacts(
    study, deps: CanonicalReportDeps, root: Path
) -> dict[str, object]:
    """Load each Bayesian policy's artifact, confined to a verified root."""

    loader = deps.load_artifact
    if loader is None:
        loader = load_bayesian_artifact
    allowed_roots = [root]
    for extra in deps.verified_roots:
        allowed_roots.append(Path(os.path.abspath(extra)))
    artifacts: dict[str, object] = {}
    for policy in study.study.ordered_panel():
        aggregation = policy.aggregation
        if getattr(aggregation, "kind", None) != "bayesian":
            continue
        artifact_dir = str(getattr(aggregation, "artifact_dir", ""))
        resolved = Path(os.path.abspath(artifact_dir))
        confined = any(
            os.path.realpath(resolved) == os.path.realpath(candidate)
            or os.path.realpath(resolved).startswith(
                os.path.realpath(candidate) + os.sep
            )
            for candidate in allowed_roots
        )
        if not confined:
            raise CanonicalReportError(
                f"policy {policy.id!r} artifact directory is outside the "
                "declared verified roots"
            )
        try:
            artifacts[policy.id] = loader(str(resolved))
        except (BayesianArtifactError, OSError) as exc:
            raise CanonicalReportError(
                f"could not load policy {policy.id!r} artifact: {_bounded(exc)}"
            ) from exc
    return artifacts


def _reduce_cell(
    *,
    entry: Mapping[str, object],
    phase_root: Path,
    condition_id: str,
    provenance: str,
    trace_reader: Callable[[object], Any],
) -> dict:
    policy_id = str(entry["policy_id"])
    schedule_id = str(entry["schedule_id"])
    seed = _nonneg_int(entry["seed"], label="seed")
    held_out = bool(entry.get("held_out", False))
    label = f"{condition_id} policy {policy_id!r} seed {seed} schedule {schedule_id!r}"

    trace_path = _confined(phase_root, entry.get("trace_path"), label=f"{label} trace")
    _require(trace_path.is_dir(), f"{label} trace directory is missing")
    try:
        with trace_reader(str(trace_path)) as reader:
            schema = getattr(reader, "schema_version", SCHEMA_VERSION)
            _require(
                schema == SCHEMA_VERSION,
                f"{label} trace schema {schema!r} is not {SCHEMA_VERSION}",
            )
            config = getattr(reader, "config", None)
            events = getattr(reader, "events", None)
            _require(
                config is not None and events is not None,
                f"{label} trace exposes no config/events",
            )
            if int(getattr(config, "seed", -1)) != seed:
                raise CanonicalReportError(f"{label} trace seed disagrees with manifest")
            metrics = _final_metrics(tuple(events), label=label)
            rejection = reduce_rejection(tuple(events), label=label)
            propagation = reduce_propagation(tuple(events), label=label)
    except CanonicalReportError:
        raise
    except (TrajectoryError, OSError) as exc:
        raise CanonicalReportError(f"{label} produced an invalid trace: {_bounded(exc)}") from exc

    embedded = entry.get("metrics")
    _require(
        isinstance(embedded, Mapping) and dict(embedded) == metrics,
        f"{label} embedded metrics disagree with the validated trace",
    )

    telemetry = entry.get("telemetry")
    _require(isinstance(telemetry, Mapping), f"{label} is missing a telemetry block")
    cost_path = _confined(phase_root, telemetry.get("cost_path"), label=f"{label} costs")
    audit_path = _confined(
        phase_root, telemetry.get("decision_audit_path"), label=f"{label} audit"
    )
    try:
        cost = load_cost_payload(cost_path)
        decisions = load_decision_audit_records(audit_path)
    except CostError as exc:
        raise CanonicalReportError(f"{label} sidecar failed validation: {_bounded(exc)}") from exc
    identity = cost.get("identity")
    _require(
        isinstance(identity, Mapping)
        and identity.get("policy_id") == policy_id
        and int(identity.get("seed", -1)) == seed,
        f"{label} cost identity disagrees with the manifest",
    )
    _require(
        identity.get("schedule_id") == schedule_id,
        f"{label} cost schedule identity disagrees with the manifest",
    )
    return {
        "condition_id": condition_id,
        "policy_id": policy_id,
        "schedule_id": schedule_id,
        "seed": seed,
        "held_out": held_out,
        "provenance": provenance,
        "reused": bool(entry.get("reused", False)),
        "metrics": metrics,
        "rejection": rejection,
        "propagation": propagation,
        "cost": cost,
        "decisions": decisions,
        "trace_path": str(trace_path),
        "cost_path": str(cost_path),
        "audit_path": str(audit_path),
    }


def _final_metrics(
    events: Sequence[Mapping[str, object]], *, label: str
) -> dict:
    if not events:
        raise CanonicalReportError(f"{label} trace has no evaluated rounds")
    final = events[-1]
    nodes = final.get("nodes") if isinstance(final, Mapping) else None
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)) or not nodes:
        raise CanonicalReportError(f"{label} final round has no per-node metrics")
    accuracies: list[float] = []
    losses: list[float] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            raise CanonicalReportError(f"{label} final round has a malformed node")
        accuracies.append(_finite_float(node.get("accuracy"), label="accuracy"))
        losses.append(_finite_float(node.get("loss"), label="loss"))
    return {
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "worst_node_accuracy": min(accuracies),
        "mean_loss": sum(losses) / len(losses),
        "node_count": len(accuracies),
    }


def _load_condition(
    *,
    condition,
    root: Path,
    provenance: str,
    deps: CanonicalReportDeps,
    trace_reader: Callable[[object], Any],
    expected_rounds: int,
) -> _ConditionEvidence:
    condition_id = condition.condition_id
    condition_root = root / CONDITIONS_DIRNAME / condition_id
    _require(condition_root.is_dir(), f"condition {condition_id!r} is missing")
    frozen_path = condition_root / "frozen-study.yaml"
    _require(frozen_path.is_file(), f"condition {condition_id!r} frozen study is missing")
    try:
        study = parse_held_out_study_text(
            frozen_path.read_text(encoding="utf-8"), source=str(frozen_path)
        )
    except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
        raise CanonicalReportError(
            f"condition {condition_id!r} frozen study is invalid: {_bounded(exc)}"
        ) from exc
    study_dir = condition_root / "study"
    final_manifest = _read_json(study_dir / "final" / "final-manifest.json")
    _require(
        final_manifest.get("schema") == "dfl_held_out_study_manifest",
        f"condition {condition_id!r} final manifest has an unexpected schema",
    )
    _require(
        final_manifest.get("status") == "complete",
        f"condition {condition_id!r} final manifest is not complete",
    )
    entries = final_manifest.get("entries")
    _require(isinstance(entries, list) and entries, f"condition {condition_id!r} has no final cells")
    panel = tuple(str(policy_id) for policy_id in final_manifest.get("panel", ()))
    _require(bool(panel), f"condition {condition_id!r} declares no policy panel")

    artifacts = _load_panel_artifacts(study, deps, root)
    artifacts_by_policy = dict(artifacts)

    phase_root = study_dir / "final"
    cells: list[dict] = []
    replay_links: list[ReplayLink] = []
    placement_by_seed: dict[int, dict] = {}
    for entry in entries:
        _require(isinstance(entry, Mapping), f"condition {condition_id!r} entry is not an object")
        cell = _reduce_cell(
            entry=entry,
            phase_root=phase_root,
            condition_id=condition_id,
            provenance=provenance,
            trace_reader=trace_reader,
        )
        cells.append(cell)
        seed = int(entry["seed"])
        placement = entry.get("placement")
        if isinstance(placement, Mapping):
            placement_by_seed[seed] = dict(placement)
        replay_links.append(
            {
                "policy_id": str(entry["policy_id"]),
                "schedule_id": str(entry["schedule_id"]),
                "seed": seed,
                "held_out": bool(entry.get("held_out", False)),
                "trace": _relative_within(root, Path(cell["trace_path"]), label="trace"),
                "costs": _relative_within(root, Path(cell["cost_path"]), label="costs"),
                "decision_audit": _relative_within(
                    root, Path(cell["audit_path"]), label="decision audit"
                ),
                "reused": bool(entry.get("reused", False)),
            }
        )

    # -- solution quality: one audit per Bayesian policy -------------------- #
    quality_records: list[dict] = []
    for policy_id in panel:
        policy_records = [
            record
            for cell in cells
            if cell["policy_id"] == policy_id
            for record in cell["decisions"]
        ]
        artifact = artifacts_by_policy.get(policy_id)
        if artifact is None:
            quality_records.append(
                {
                    "policy_id": policy_id,
                    "provenance": provenance,
                    "applicable": False,
                    "reason": "policy has no artifact-backed finite-action audit",
                }
            )
            continue
        audit = audit_bayesian_decisions(
            artifact,
            policy_records,
            total_rounds=expected_rounds,
            policy_id=policy_id,
            provenance=provenance,
        )
        quality_records.append(audit)

    # -- costs / rejection / propagation aggregates ------------------------ #
    cost_rows: list[dict] = []
    rejection = _empty_rejection()
    propagation_totals = {
        "reached_node_rounds": 0,
        "total_node_rounds": 0,
        "reached_nodes_ever": 0,
    }
    for cell in cells:
        totals = cell["cost"]["totals"]
        identity = cell["cost"]["identity"]
        cost_rows.append(
            {
                "condition_id": condition_id,
                "policy_id": cell["policy_id"],
                "schedule_id": cell["schedule_id"],
                "seed": cell["seed"],
                "reused": cell["reused"],
                "observation_ns": int(totals["observation_ns"]),
                "policy_ns": int(totals["policy_ns"]),
                "aggregation_ns": int(totals["aggregation_ns"]),
                "selection_ns": int(totals["selection_ns"]),
                "setup_ns": int(totals["setup_ns"]),
                "train_ns": int(totals["train_ns"]),
                "evaluation_ns": int(totals["evaluation_ns"]),
                "whole_cell_ns": int(totals["whole_cell_ns"]),
                "directed_messages": int(totals["directed_messages"]),
                "logical_payload_bytes": int(totals["logical_payload_bytes"]),
                "model_payload_bytes": int(totals["model_payload_bytes"]),
                "clock": str(cell["cost"]["measurement"]["clock"]),
                "study_id": identity.get("study_id"),
            }
        )
        _merge_rejection(rejection, cell["rejection"])
        propagation_totals["reached_node_rounds"] += int(
            cell["propagation"]["reached_node_rounds"]
        )
        propagation_totals["total_node_rounds"] += int(
            cell["propagation"]["total_node_rounds"]
        )
        propagation_totals["reached_nodes_ever"] = max(
            propagation_totals["reached_nodes_ever"],
            int(cell["propagation"]["reached_nodes_ever"]),
        )

    selection_manifest_path = study_dir / "selection" / "selection-manifest.json"
    selection_cells = 0
    if selection_manifest_path.is_file():
        selection_manifest = _read_json(selection_manifest_path)
        selection_cells = len(selection_manifest.get("entries", ()))
    record = {
        "condition_id": condition_id,
        "alpha": float(condition.alpha),
        "placement": condition.placement,
        "provenance": provenance,
        "study_digest": str(final_manifest.get("study_digest")),
        "held_out_study_digest": str(final_manifest.get("held_out_study_digest")),
        "frozen_study_digest": _sha256_bytes(frozen_path.read_bytes()),
        "selection_output_digest": str(final_manifest.get("selection_output_digest")),
        "final_output_digest": str(final_manifest.get("final_output_digest")),
        "policy_ids": panel,
        "schedule_ids": tuple(
            sorted({str(entry["schedule_id"]) for entry in entries})
        ),
        "selection_cells": selection_cells,
        "final_cells": len(entries),
        "aggregates": len({str(e["schedule_id"]) for e in entries}) * len(panel),
        "messages": sum(row["directed_messages"] for row in cost_rows),
        "payload_bytes": sum(row["logical_payload_bytes"] for row in cost_rows),
        "placement_identities": tuple(
            placement_by_seed[seed] for seed in sorted(placement_by_seed)
        ),
        "replay_links": tuple(replay_links),
        "quality_records": quality_records,
        "cost_rows": cost_rows,
        "rejection": rejection,
        "propagation": propagation_totals,
    }
    return _ConditionEvidence(
        record=record,
        cells=cells,
        costs=cost_rows,
        rejection=rejection,
        propagation=propagation_totals,
    )


# --------------------------------------------------------------------------- #
# Block builders
# --------------------------------------------------------------------------- #
_COST_TOTAL_KEYS = (
    "observation_ns",
    "policy_ns",
    "aggregation_ns",
    "selection_ns",
    "setup_ns",
    "train_ns",
    "evaluation_ns",
    "whole_cell_ns",
    "directed_messages",
    "logical_payload_bytes",
    "model_payload_bytes",
)


def _aggregate_costs(condition_records: Sequence[Mapping[str, object]]) -> dict:
    by_key: dict[tuple[str, str], dict] = {}
    totals = {key: 0 for key in _COST_TOTAL_KEYS}
    resumed = 0
    new = 0
    for record in condition_records:
        for row in record["cost_rows"]:
            key = (str(row["condition_id"]), str(row["policy_id"]))
            bucket = by_key.setdefault(
                key,
                {
                    "condition_id": key[0],
                    "policy_id": key[1],
                    "cells": 0,
                    "reused_cells": 0,
                    "new_cells": 0,
                    **{name: 0 for name in _COST_TOTAL_KEYS},
                },
            )
            bucket["cells"] += 1
            if row["reused"]:
                bucket["reused_cells"] += 1
                resumed += 1
            else:
                bucket["new_cells"] += 1
                new += 1
            for name in _COST_TOTAL_KEYS:
                bucket[name] += int(row[name])
                totals[name] += int(row[name])
    computation = {
        "units": "nanoseconds",
        "clock": "perf_counter_ns",
        "environment": {
            "device": "cpu",
            "threads": 1,
            "note": "measured at the shared round/evaluation seam; not tuning inputs",
        },
        "phases": [
            "observation_ns",
            "policy_ns",
            "aggregation_ns",
            "selection_ns",
            "setup_ns",
            "train_ns",
            "evaluation_ns",
            "whole_cell_ns",
        ],
        "totals": {name: totals[name] for name in _COST_TOTAL_KEYS if name.endswith("_ns")},
        "by_condition_policy": [
            by_key[key] for key in sorted(by_key)
        ],
    }
    communication = {
        "semantics": (
            "actual directed messages and the sum of numel*element_size over "
            "every message's state tensors at construction, before receiver "
            "filtering; headers/retransmits/network are excluded"
        ),
        "totals": {
            "directed_messages": totals["directed_messages"],
            "logical_payload_bytes": totals["logical_payload_bytes"],
            "model_payload_bytes": totals["model_payload_bytes"],
        },
        "by_condition_policy": [
            {
                "condition_id": key[0],
                "policy_id": key[1],
                "directed_messages": by_key[key]["directed_messages"],
                "logical_payload_bytes": by_key[key]["logical_payload_bytes"],
                "model_payload_bytes": by_key[key]["model_payload_bytes"],
            }
            for key in sorted(by_key)
        ],
    }
    return {
        "computation": computation,
        "communication": communication,
        "resumed_cells": resumed,
        "new_cells": new,
    }


def _aggregate_rejection(condition_records: Sequence[Mapping[str, object]]) -> dict:
    by_key: dict[tuple[str, str], dict] = {}
    totals = _empty_rejection()
    for record in condition_records:
        condition_id = str(record["condition_id"])
        policies = {str(policy_id) for policy_id in record["policy_ids"]}
        # Split each condition's totals back per policy using the raw cells is
        # not available here; the caller passes per-condition records whose
        # rejection is the condition total over all policies. To keep per-policy
        # accounting exact the loader also stores per-policy counters.
        per_policy = record.get("rejection_by_policy", {})
        for policy_id in policies:
            counters = per_policy.get(policy_id, _empty_rejection())
            key = (condition_id, policy_id)
            bucket = by_key.setdefault(
                key,
                {
                    "condition_id": condition_id,
                    "policy_id": policy_id,
                    "provenance": record["provenance"],
                    **{name: 0 for name in _REJECTION_COUNTERS},
                },
            )
            _merge_rejection(bucket, counters)
            _merge_rejection(totals, counters)
    rows: list[dict] = []
    for key in sorted(by_key):
        bucket = by_key[key]
        rows.append({**bucket, **_rejection_rates(bucket)})
    return {"by_condition_policy": rows, "totals": {**totals, **_rejection_rates(totals)}}


def _aggregate_propagation(condition_records: Sequence[Mapping[str, object]]) -> dict:
    by_key: dict[tuple[str, str], dict] = {}
    totals = {
        "reached_node_rounds": 0,
        "total_node_rounds": 0,
        "reached_nodes_ever": 0,
        "min_hop": None,
        "max_hop": None,
        "first_reached_rounds": set(),
        "sources": [],
    }
    for record in condition_records:
        per_policy = record.get("propagation_by_policy", {})
        for policy_id, summary in per_policy.items():
            key = (str(record["condition_id"]), str(policy_id))
            bucket = by_key.setdefault(
                key,
                {
                    "condition_id": key[0],
                    "policy_id": key[1],
                    "provenance": record["provenance"],
                    "reached_node_rounds": 0,
                    "total_node_rounds": 0,
                    "reached_nodes_ever": 0,
                    "min_hop": None,
                    "max_hop": None,
                    "first_reached_rounds": set(),
                    "sources": [],
                },
            )
            bucket["reached_node_rounds"] += int(summary["reached_node_rounds"])
            bucket["total_node_rounds"] += int(summary["total_node_rounds"])
            bucket["reached_nodes_ever"] = max(
                bucket["reached_nodes_ever"], int(summary["reached_nodes_ever"])
            )
            hops = [
                value
                for value in (summary.get("min_hop"), summary.get("max_hop"))
                if value is not None
            ]
            if hops:
                bucket["min_hop"] = (
                    min(hops)
                    if bucket["min_hop"] is None
                    else min(bucket["min_hop"], min(hops))
                )
                bucket["max_hop"] = (
                    max(hops)
                    if bucket["max_hop"] is None
                    else max(bucket["max_hop"], max(hops))
                )
            bucket["first_reached_rounds"].update(summary.get("first_reached_rounds", ()))
            bucket["sources"].extend(summary.get("sources", ())[:8])
    rows: list[dict] = []
    for key in sorted(by_key):
        bucket = by_key[key]
        totals["reached_node_rounds"] += bucket["reached_node_rounds"]
        totals["total_node_rounds"] += bucket["total_node_rounds"]
        totals["reached_nodes_ever"] = max(
            totals["reached_nodes_ever"], bucket["reached_nodes_ever"]
        )
        for value in (bucket["min_hop"], bucket["max_hop"]):
            if value is None:
                continue
            totals["min_hop"] = (
                value if totals["min_hop"] is None else min(totals["min_hop"], value)
            )
            totals["max_hop"] = (
                value if totals["max_hop"] is None else max(totals["max_hop"], value)
            )
        totals["first_reached_rounds"].update(bucket["first_reached_rounds"])
        totals["sources"].extend(bucket["sources"][:4])
        rows.append(
            {
                **bucket,
                "first_reached_rounds": sorted(bucket["first_reached_rounds"]),
            }
        )
    totals["first_reached_rounds"] = sorted(totals["first_reached_rounds"])
    totals["sources"] = totals["sources"][:16]
    return {"by_condition_policy": rows, "totals": totals}


# --------------------------------------------------------------------------- #
# Public compile / verify
# --------------------------------------------------------------------------- #
def _build_blocks(
    *,
    root: Path,
    protocol: RobustnessProtocol,
    canonical: CanonicalRobustnessProtocol,
    evidence: Mapping[str, object],
    headline: Mapping[str, object],
    verified_roots: Mapping[str, str],
    deps: CanonicalReportDeps,
    trace_reader: Callable[[object], Any],
) -> dict:
    conditions_payload = [
        _load_condition(
            condition=condition,
            root=root,
            provenance=protocol.data_provenance,
            deps=deps,
            trace_reader=trace_reader,
            expected_rounds=int(protocol.rounds),
        )
        for condition in canonical.conditions()
    ]

    all_cells = [cell for evidence_item in conditions_payload for cell in evidence_item.cells]
    aggregates = reduce_final_aggregates(all_cells)
    deltas = paired_policy_deltas(aggregates)

    internal_records: list[dict] = []
    condition_records: list[dict] = []
    quality: list[dict] = []
    for item in conditions_payload:
        record = dict(item.record)
        record["rejection_by_policy"] = _rejection_by_policy(item.cells)
        record["propagation_by_policy"] = _propagation_by_policy(item.cells)
        internal_records.append(record)
        quality.extend(record["quality_records"])
        published = {
            key: value
            for key, value in record.items()
            if key
            in {
                "condition_id",
                "alpha",
                "placement",
                "provenance",
                "study_digest",
                "held_out_study_digest",
                "frozen_study_digest",
                "selection_output_digest",
                "final_output_digest",
                "policy_ids",
                "schedule_ids",
                "selection_cells",
                "final_cells",
                "aggregates",
                "messages",
                "payload_bytes",
                "placement_identities",
                "replay_links",
            }
        }
        condition_records.append(published)

    cost_block = _aggregate_costs(internal_records)
    rejection_block = _aggregate_rejection(internal_records)
    propagation_block = _aggregate_propagation(internal_records)

    derivation = {
        "provenance": protocol.data_provenance,
        "data_identity": {
            "dataset": "mnist",
            "partition": "dirichlet",
            "alphas": list(protocol.alphas),
            "topology": {
                "kind": protocol.topology_kind,
                "n": int(protocol.topology_n),
                "k": int(protocol.topology_k),
                "p": float(protocol.topology_p),
            },
            "cache": dict(evidence.get("cache", {})),
            "note": (
                "injected fixture provenance"
                if protocol.data_provenance == "injected"
                else "content-digested downloaded MNIST cache"
            ),
        },
        "calibration": [
            {
                "condition_id": record["condition_id"],
                "alpha": record["alpha"],
                "note": "per-alpha matched full/no_belief/no_strategic triple",
            }
            for record in condition_records
        ],
        "artifact_assumptions": [
            "posterior is proportional to artifact prior times per-neighbor likelihood over the artifact's feasible joint types",
            "expected payoff uses the pinned artifact payoff table and the frozen finite action space",
            "the no_strategic variant fixes the attacker mixture and is an assumption, not measured",
        ],
        "headline": dict(headline),
        "unavailable": [
            "historical S04 headline decision-audit and cost channels are not present in the raw evidence",
        ],
    }

    solution_quality = {
        "provenance": protocol.data_provenance,
        "policies": quality,
        "limitations": [
            "finite-action solution quality only; no equilibrium/regret/strategic-solver claim",
            "degree-cap pruning reduces the variable neighbor set above the artifact cap",
            "fallback rows select exact self-only weights and are counted, not hidden",
        ],
    }
    defense = {
        "provenance": protocol.data_provenance,
        "summary": {
            "matrix_final_aggregates": len(aggregates),
            "paired_delta_records": len(deltas),
            "conditions": len(condition_records),
            "note": "observed final outcomes only; selection is never mixed in",
        },
    }

    scope = {
        "study_id": protocol.study_id,
        "data_provenance": protocol.data_provenance,
        "alphas": list(protocol.alphas),
        "placements": list(protocol.placements),
        "policies": list(condition_records[0]["policy_ids"]) if condition_records else [],
        "selection_seeds": list(protocol.selection_seeds),
        "final_seeds": list(protocol.final_seeds),
        "calibration_seed": int(protocol.calibration_seed),
        "rounds": int(protocol.rounds),
        "samples_per_node": int(protocol.samples_per_node),
        "evaluation_max_samples": int(protocol.evaluation_max_samples),
        "matrix_selection_cells": sum(r["selection_cells"] for r in condition_records),
        "matrix_final_cells": sum(r["final_cells"] for r in condition_records),
        "matrix_aggregates": len(aggregates),
        "headline": dict(headline),
    }

    sources = {
        "protocol_digest": str(evidence.get("protocol_digest")),
        "matrix_digest": str(evidence.get("matrix_digest")),
        "matrix_manifest_digest": str(
            evidence.get("matrix_manifest_digest")
            or _sha256_json(_read_json(root / MATRIX_MANIFEST_FILENAME))
        ),
        "matrix_evidence_digest": str(evidence.get("evidence_digest")),
        "headline": dict(evidence.get("headline", {})),
        "cache": dict(evidence.get("cache", {})),
        "verified_roots": verified_roots,
    }

    metrics = {
        "std_method": "population",
        "quantile_method": "linear",
        "disclaimer": (
            "population spread and IQR over the declared final seeds; no "
            "confidence interval or winner claim"
        ),
        "aggregates": aggregates,
        "paired_deltas": deltas,
    }

    report = {
        "schema": CANONICAL_REPORT_SCHEMA,
        "schema_version": CANONICAL_REPORT_SCHEMA_VERSION,
        "status": "complete",
        "study_id": protocol.study_id,
        "data_provenance": protocol.data_provenance,
        "scope": scope,
        "sources": sources,
        "conditions": condition_records,
        "evidence": {
            "derivation": derivation,
            "solution_quality": solution_quality,
            "defense_usefulness": defense,
        },
        "metrics": metrics,
        "costs": {
            "computation": cost_block["computation"],
            "communication": cost_block["communication"],
            "calibration_stage_wall": {
                "available": False,
                "reason": "per-alpha calibration stage wall time is not recorded in the raw matrix manifest",
            },
            "process_rss": {
                "available": False,
                "reason": "process RSS is not recorded in the raw matrix evidence",
            },
            "headline": {
                "available": False,
                "reason": "the bounded IID headline carries no cost/decision sidecars",
            },
            "resumed_cells": cost_block["resumed_cells"],
            "new_cells": cost_block["new_cells"],
        },
        "rejection": {
            "semantics": {
                "opportunity": "one directed edge into a benign receiver, excluding self",
                "excluded": "receiver weight for that sender is exactly zero",
                "accepted": "receiver weight for that sender is positive",
                "active": "structurally malicious sender whose realized kappa > 0 in that round",
                "inactive": "structurally malicious sender with kappa == 0 in that round",
                "note": "local_only and continuous weighting show as zero contribution, not detector positives",
            },
            **rejection_block,
        },
        "propagation": {
            "semantics": {
                "reached": "monotone weight-aware poison provenance",
                "min_hop": "positive aggregation edges from the active seed (hop 0)",
                "first_reached_round": "earliest committed round the node became reached",
            },
            **propagation_block,
        },
        "limitations": list(_LIMITATIONS),
    }
    body = dict(report)
    report["report_digest"] = _sha256_json(body)
    return validate_canonical_report(report)


def _rejection_by_policy(cells: Sequence[Mapping[str, object]]) -> dict:
    by_policy: dict[str, dict] = {}
    for cell in cells:
        policy_id = str(cell["policy_id"])
        bucket = by_policy.setdefault(policy_id, _empty_rejection())
        _merge_rejection(bucket, cell["rejection"])
    return by_policy


def _propagation_by_policy(cells: Sequence[Mapping[str, object]]) -> dict:
    by_policy: dict[str, dict] = {}
    for cell in cells:
        policy_id = str(cell["policy_id"])
        bucket = by_policy.setdefault(
            policy_id,
            {
                "reached_node_rounds": 0,
                "total_node_rounds": 0,
                "reached_nodes_ever": 0,
                "min_hop": None,
                "max_hop": None,
                "first_reached_rounds": set(),
                "sources": [],
            },
        )
        summary = cell["propagation"]
        bucket["reached_node_rounds"] += int(summary["reached_node_rounds"])
        bucket["total_node_rounds"] += int(summary["total_node_rounds"])
        bucket["reached_nodes_ever"] = max(
            bucket["reached_nodes_ever"], int(summary["reached_nodes_ever"])
        )
        for value in (summary.get("min_hop"), summary.get("max_hop")):
            if value is None:
                continue
            bucket["min_hop"] = (
                value if bucket["min_hop"] is None else min(bucket["min_hop"], value)
            )
            bucket["max_hop"] = (
                value if bucket["max_hop"] is None else max(bucket["max_hop"], value)
            )
        bucket["first_reached_rounds"].update(summary.get("first_reached_rounds", ()))
        bucket["sources"].extend(summary.get("sources", ())[:4])
    return {
        policy_id: {
            "reached_node_rounds": bucket["reached_node_rounds"],
            "total_node_rounds": bucket["total_node_rounds"],
            "reached_nodes_ever": bucket["reached_nodes_ever"],
            "min_hop": bucket["min_hop"],
            "max_hop": bucket["max_hop"],
            "first_reached_rounds": sorted(bucket["first_reached_rounds"]),
            "sources": bucket["sources"][:16],
        }
        for policy_id, bucket in by_policy.items()
    }


def build_report_payload(
    root: str | os.PathLike[str],
    *,
    deps: CanonicalReportDeps | None = None,
) -> dict:
    """Compile the canonical report payload in memory (no publication)."""

    deps = deps or CanonicalReportDeps()
    root_path = Path(os.path.abspath(os.fspath(root)))
    _require(root_path.is_dir(), f"robustness matrix root not found: {root_path}")

    frozen = _read_json(root_path / ROBUSTNESS_PROTOCOL_FILENAME)
    _require(
        frozen.get("schema") == "dfl_sim.robustness_protocol/v1",
        "robustness-protocol.json has an unexpected schema",
    )
    payload = frozen.get("protocol")
    _require(isinstance(payload, Mapping), "frozen protocol payload is missing")
    try:
        protocol = RobustnessProtocol.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - bounded fail-closed boundary
        raise CanonicalReportError(f"frozen protocol is invalid: {_bounded(exc)}") from exc
    _require(
        protocol.data_provenance == "real_mnist" or deps.allow_injected is True,
        "the public canonical report accepts real_mnist provenance only",
    )
    canonical = canonical_protocol(protocol, artifact_root=str(root_path))

    verify = deps.verify_matrix
    if verify is None:
        verify = verify_robustness_study
    verify_deps = deps.verify_deps or RobustnessVerifyDeps(repo_root=deps.repo_root)
    try:
        result = verify(str(root_path), deps=verify_deps, write=False, deep_cache=False)
    except RobustnessVerifyError as exc:
        raise CanonicalReportError(
            f"raw matrix audit failed closed: {_bounded(exc)}"
        ) from exc
    evidence = getattr(result, "evidence", None)
    _require(isinstance(evidence, Mapping), "matrix audit produced no evidence")
    on_disk = _read_json(root_path / MATRIX_EVIDENCE_FILENAME)
    stored_digest = on_disk.get("evidence_digest")
    on_disk_body = {key: value for key, value in on_disk.items() if key != "evidence_digest"}
    _require(
        stored_digest == _sha256_json(on_disk_body),
        "the published matrix evidence digest does not bind its body",
    )
    _require(
        stored_digest == evidence.get("evidence_digest"),
        "the published matrix evidence no longer matches a fresh audit",
    )

    verified_roots: dict[str, str] = {"matrix": str(root_path)}
    headline = _audit_headline(
        protocol=protocol,
        frozen=frozen,
        deps=deps,
        repo_root=(
            Path(deps.repo_root) if deps.repo_root is not None else _default_repo_root()
        ),
        verified_roots=verified_roots,
    )

    trace_reader = deps.trace_reader or TraceReader
    return _build_blocks(
        root=root_path,
        protocol=protocol,
        canonical=canonical,
        evidence=evidence,
        headline=headline,
        verified_roots=verified_roots,
        deps=deps,
        trace_reader=trace_reader,
    )


def _publish_report(output_dir: Path, payload: Mapping[str, object]) -> tuple[str, Path]:
    if os.path.lexists(output_dir) and os.path.islink(output_dir):
        raise CanonicalReportError(f"report destination must not be a symlink: {output_dir}")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CanonicalReportError(
            f"could not create report directory {output_dir}: {_bounded(exc)}"
        ) from exc
    destination = output_dir / CANONICAL_REPORT_FILENAME
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if destination.is_file() and destination.read_bytes() == raw:
        return "unchanged", destination
    try:
        fd, temporary = tempfile.mkstemp(
            dir=str(output_dir), prefix=".canonical-report.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    except OSError as exc:
        raise CanonicalReportError(
            f"could not publish canonical report: {_bounded(exc)}"
        ) from exc
    return "published", destination


def compile_canonical_report(
    root: str | os.PathLike[str],
    output: str | os.PathLike[str] | None = None,
    *,
    deps: CanonicalReportDeps | None = None,
) -> CanonicalReportResult:
    """Compile and atomically publish ``canonical-report.json``.

    The public path refuses injected inputs; an isolated fixture seam must pass
    ``allow_injected=True`` explicitly. Nothing is published unless the raw
    matrix audit, the fresh headline audit (real path) and every reducer pass.
    """

    deps = deps or CanonicalReportDeps()
    root_path = Path(os.path.abspath(os.fspath(root)))
    report = build_report_payload(root_path, deps=deps)
    destination_dir = (
        Path(os.path.abspath(os.fspath(output)))
        if output is not None
        else root_path / "report"
    )
    status, report_path = _publish_report(destination_dir, report)
    return CanonicalReportResult(
        output_path=destination_dir,
        report_path=report_path,
        status=status,
        report=report,
        report_digest=str(report["report_digest"]),
        conditions=len(report["conditions"]),
        aggregates=len(report["metrics"]["aggregates"]),
        paired_deltas=len(report["metrics"]["paired_deltas"]),
    )


def verify_canonical_report(
    root: str | os.PathLike[str],
    *,
    deps: CanonicalReportDeps | None = None,
) -> CanonicalReportResult:
    """Read-only revalidation of a published canonical report.

    Recompiles the report from the audited sources and compares the full
    canonical payload and digest against the on-disk document without mutating
    any source or the report.
    """

    deps = deps or CanonicalReportDeps()
    root_path = Path(os.path.abspath(os.fspath(root)))
    report_dir = root_path / "report"
    on_disk_path = report_dir / CANONICAL_REPORT_FILENAME
    _require(on_disk_path.is_file(), f"canonical report not found: {on_disk_path}")
    on_disk = validate_canonical_report(_read_json(on_disk_path))
    fresh = build_report_payload(root_path, deps=deps)
    _require(
        on_disk.get("report_digest") == fresh.get("report_digest"),
        "published canonical report digest disagrees with a fresh compilation",
    )
    _require(
        _canonical_bytes(on_disk) == _canonical_bytes(fresh),
        "published canonical report body disagrees with a fresh compilation",
    )
    return CanonicalReportResult(
        output_path=report_dir,
        report_path=on_disk_path,
        status="verified",
        report=fresh,
        report_digest=str(fresh["report_digest"]),
        conditions=len(fresh["conditions"]),
        aggregates=len(fresh["metrics"]["aggregates"]),
        paired_deltas=len(fresh["metrics"]["paired_deltas"]),
    )
