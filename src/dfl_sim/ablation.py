"""Canonical calibration-ablation identity helpers (S03/T01).

This offline-only module owns the pure, deterministic identity helpers shared by
the calibration config, the offline collection layer, and the artifact manifest.
The ablation *assumption contract* itself (the variant enum, pinned mixtures,
isolation statements, and the :class:`CalibrationDerivation` metadata model)
lives in :mod:`dfl_sim.bayesian_artifact`, which is the privilege-free leaf the
artifact loader already depends on. This module re-exports that contract and
adds:

* :class:`AblationError` and bounded variant validation;
* path-independent, variant-normalized calibration-config identity;
* content fingerprints of the consumed dataset inputs and partition membership;
* an intensity-excluding schedule pairing digest.

Honesty boundaries (R021/D036/D037):

* A *calibration-config identity* (:func:`calibration_config_digest`) is a
  declared path/config identity, never a content fingerprint. Actual data
  content is fingerprinted separately by :func:`dataset_content_digest`.
* The no-strategic variant pins the degenerate strong attacker ``kappa=5``. It
  changes calibration attack composition/trajectories, so an equal measured
  outcome is reported as *no measured contribution on this fixture*, never as
  proof that a strategic-response solver adds nothing.

Every helper is deterministic and pure; digests are SHA-256 over canonical JSON
(or raw example bytes for dataset content). Nothing here touches the runtime
policy, evaluator, simulator, or datasets.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .bayesian_artifact import (
    ABLATION_PROTOCOL_VERSION,
    CALIBRATION_VARIANTS,
    CANDIDATE_PROTOCOL,
    CANDIDATE_PROTOCOLS,
    FULL_VARIANT,
    ISOLATION_STATEMENTS,
    NO_BELIEF_VARIANT,
    NO_STRATEGIC_EFFECTIVE_MU_A,
    NO_STRATEGIC_KAPPA,
    NO_STRATEGIC_VARIANT,
    SOURCE_MU_A,
    CalibrationDerivation,
    Variant,
)

__all__ = [
    "ABLATION_PROTOCOL_VERSION",
    "CALIBRATION_VARIANTS",
    "CANDIDATE_PROTOCOLS",
    "CANDIDATE_PROTOCOL",
    "FULL_VARIANT",
    "NO_BELIEF_VARIANT",
    "NO_STRATEGIC_EFFECTIVE_MU_A",
    "NO_STRATEGIC_KAPPA",
    "NO_STRATEGIC_VARIANT",
    "SOURCE_MU_A",
    "ISOLATION_STATEMENTS",
    "AblationError",
    "CalibrationDerivation",
    "Variant",
    "calibration_config_digest",
    "dataset_content_digest",
    "effective_mixture",
    "isolation_statement",
    "matched_calibration_input_digest",
    "membership_digest",
    "normalize_calibration_config",
    "require_variant",
    "schedule_pairing_digest",
    "sha256_hex",
]

#: Keys whose values are environmental locations and are scrubbed to a fixed
#: placeholder when building a path-independent config identity.
_ENVIRONMENTAL_KEYS = frozenset(
    {
        "data_root",
        "config_path",
        "config_dir",
        "output",
        "output_dir",
        "output_path",
        "artifact_dir",
        "artifact_path",
    }
)

#: Keys that encode the declared variant/effective intensity difference and are
#: normalized away so the three variants share one source calibration identity.
_NORMALIZED_KEYS = frozenset({"variant", "ablation_variant", "effective_mu_a"})


class AblationError(RuntimeError):
    """Raised when an ablation identity/variant operation cannot be satisfied.

    Raised for an unknown or wrongly typed variant, a malformed derivation
    payload, or a non-canonical identity input. Failures are bounded and never
    fabricate provenance.
    """


def _bounded(detail: object, *, limit: int = 400) -> str:
    collapsed = " ".join(str(detail).split())
    if len(collapsed) > limit:
        collapsed = collapsed[:limit] + "..."
    return collapsed


# --------------------------------------------------------------------------- #
# Variant / mixture assumptions (re-exported contract with bounded errors)
# --------------------------------------------------------------------------- #
def require_variant(value: object) -> str:
    """Return ``value`` as a known variant or raise :class:`AblationError`."""

    if isinstance(value, bool) or not isinstance(value, str):
        raise AblationError(
            f"calibration variant must be a string, got {type(value).__name__}"
        )
    if value not in CALIBRATION_VARIANTS:
        raise AblationError(
            f"unknown calibration variant {value!r}; expected one of "
            f"{CALIBRATION_VARIANTS!r}"
        )
    return value


def effective_mixture(
    variant: str, source_mu_a: Sequence[float] = SOURCE_MU_A
) -> tuple[float, ...]:
    """Return the effective attacker mixture for ``variant`` (D036).

    The canonical source protocol always carries the pinned ``(0, 2, 5)``
    mixture. Only the no-strategic variant derives the degenerate strong mixture
    ``(5,)``; full and no-belief stay byte-for-byte identical to the source.
    """

    variant = require_variant(variant)
    source = tuple(float(value) for value in source_mu_a)
    if variant == NO_STRATEGIC_VARIANT:
        return NO_STRATEGIC_EFFECTIVE_MU_A
    return source


def isolation_statement(variant: str) -> str:
    """Return the canonical isolation statement for ``variant``."""

    variant = require_variant(variant)
    return ISOLATION_STATEMENTS[variant]


# --------------------------------------------------------------------------- #
# Canonical hashing helpers
# --------------------------------------------------------------------------- #
def sha256_hex(data: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``data``."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise AblationError("sha256_hex requires bytes-like input")
    return hashlib.sha256(bytes(data)).hexdigest()


def canonical_json_text(obj: Any) -> str:
    """Return the canonical finite JSON text used by identity digests."""

    try:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AblationError(
            f"identity payload is not canonically encodable: {_bounded(exc)}"
        ) from exc


def _sha256_json(obj: Any) -> str:
    return sha256_hex(canonical_json_text(obj).encode("utf-8"))


# --------------------------------------------------------------------------- #
# Path-independent calibration-config identity
# --------------------------------------------------------------------------- #
def _scrub(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in _ENVIRONMENTAL_KEYS:
                out[str(key)] = "<environment>"
            elif key in _NORMALIZED_KEYS:
                continue
            else:
                out[str(key)] = _scrub(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    return value


def normalize_calibration_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a path-independent, variant-normalized copy of a config body.

    Only declared environmental locations (data root, config/output paths) and
    the declared variant/effective-intensity difference are normalized. Every
    learning/data/topology/bin/action/smoothing/seed field is retained, so the
    returned mapping is a *config identity*, not a data-content fingerprint.
    """

    if not isinstance(payload, Mapping):
        raise AblationError(
            f"calibration config payload must be a mapping, got "
            f"{type(payload).__name__}"
        )
    scrubbed = _scrub(payload)
    # The effective mixture is derived from the variant. Force the canonical
    # source mixture so full/no_belief/no_strategic share one source identity.
    if "mu_a" in scrubbed:
        scrubbed["mu_a"] = [float(value) for value in SOURCE_MU_A]
    return scrubbed


def calibration_config_digest(payload: Mapping[str, Any]) -> str:
    """Return the path-independent identity of a calibration config body."""

    return _sha256_json(normalize_calibration_config(payload))


def matched_calibration_input_digest(
    derivation: CalibrationDerivation,
) -> str:
    """Return the shared calibration-input identity of a derivation.

    This combines the *declared* normalized calibration-config identity, the
    calibration seed, the actual dataset and partition/design membership
    fingerprints, and the intensity-excluding schedule-pairing fingerprint.
    Variant / effective-mixture fields, every table byte, and the declared
    derivation budgets are excluded (the budgets are separately cross-checked by
    the preflight and are already bound by the aggregate digest and, for
    episodes/design size, by the config digest). ``full``, ``no_belief`` and
    ``no_strategic`` artifacts derived from one calibration run therefore share
    exactly one matched identity while genuinely different data, partitions,
    seeds, or configs change it.

    The digest is a matching identity, not a content fingerprint: the actual
    dataset content is fingerprinted separately by
    :func:`dataset_content_digest` and carried inside this value.
    """

    if not isinstance(derivation, CalibrationDerivation):
        raise AblationError(
            "matched_calibration_input_digest requires a CalibrationDerivation"
        )
    return _sha256_json(
        {
            "protocol_version": derivation.protocol_version,
            "seed": derivation.seed,
            "calibration_config_digest": derivation.calibration_config_digest,
            "dataset_inputs_digest": derivation.dataset_inputs_digest,
            "partition_membership_digest": derivation.partition_membership_digest,
            "schedule_pairing_digest": derivation.schedule_pairing_digest,
        }
    )


# --------------------------------------------------------------------------- #
# Content fingerprints captured at collection
# --------------------------------------------------------------------------- #
def dataset_content_digest(
    examples: Iterable[tuple[int, int, bytes]],
) -> str:
    """Fingerprint canonical ``(index, label, tensor-bytes)`` dataset inputs.

    Examples are hashed in ascending index order exactly once; duplicate
    indices are rejected. The digest is content-sensitive (labels/tensor bytes)
    and independent of any filesystem path or environment root.
    """

    hasher = hashlib.sha256()
    entries: list[tuple[int, int, bytes]] = []
    for entry in examples:
        try:
            index, label, data = entry
        except (TypeError, ValueError) as exc:
            raise AblationError(
                "dataset content entries must be (index, label, bytes) triples"
            ) from exc
        if isinstance(index, bool) or not isinstance(index, int):
            raise AblationError(f"dataset index must be an int, got {index!r}")
        if isinstance(label, bool) or not isinstance(label, int):
            raise AblationError(f"dataset label must be an int, got {label!r}")
        if isinstance(data, (bytearray, memoryview)):
            data = bytes(data)
        if not isinstance(data, bytes):
            raise AblationError(f"dataset content for index {index} must be bytes")
        entries.append((index, label, data))

    entries.sort(key=lambda item: item[0])
    seen: set[int] = set()
    for index, label, data in entries:
        if index in seen:
            raise AblationError(f"duplicate consumed dataset index {index}")
        seen.add(index)
        hasher.update(b"example")
        hasher.update(index.to_bytes(8, "big"))
        hasher.update(label.to_bytes(8, "big", signed=True))
        hasher.update(len(data).to_bytes(8, "big"))
        hasher.update(data)
    return hasher.hexdigest()


def membership_digest(episodes: Iterable[Mapping[str, Any]]) -> str:
    """Fingerprint the per-episode node-partition and design membership.

    ``episodes`` is an ordered iterable of mappings with ``episode``, ``nodes``
    (node id -> index sequence) and ``design`` (index sequence). No dataset
    content is read; only which indices were allocated to which consumer.
    Within each consumer the indices are treated as a membership *set*, so
    ordering inside a node block or the design block never changes the digest;
    episode order and the partition boundaries do.
    """

    hasher = hashlib.sha256()
    for position, entry in enumerate(episodes):
        if not isinstance(entry, Mapping):
            raise AblationError(
                f"membership episode {position} must be a mapping, got "
                f"{type(entry).__name__}"
            )
        try:
            episode_index = int(entry["episode"])
            nodes = entry["nodes"]
            design = entry["design"]
        except (KeyError, TypeError, ValueError) as exc:
            raise AblationError(
                f"membership episode {position} is missing required fields"
            ) from exc
        if not isinstance(nodes, Mapping):
            raise AblationError(
                f"membership episode {position} nodes must be a mapping"
            )
        node_payload = {
            str(int(node)): sorted(int(index) for index in indices)
            for node, indices in sorted(nodes.items(), key=lambda kv: int(kv[0]))
        }
        payload = {
            "episode": episode_index,
            "nodes": node_payload,
            "design": sorted(int(index) for index in design),
        }
        hasher.update(b"episode")
        hasher.update(canonical_json_text(payload).encode("utf-8"))
    return hasher.hexdigest()


def _spec_field(spec: object, name: str) -> Any:
    if isinstance(spec, Mapping):
        try:
            return spec[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AblationError(f"schedule spec is missing {name!r}") from exc
    try:
        return getattr(spec, name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AblationError(f"schedule spec is missing {name!r}") from exc


def schedule_pairing_digest(specs: Iterable[object]) -> str:
    """Fingerprint a schedule's index/seed/stratum/placement pairing.

    The sampled attack intensity and value are deliberately excluded, so a
    variant that only changes effective intensity selection (no-strategic)
    yields the same pairing digest while genuinely different seeds, strata, or
    placements change it.
    """

    records: list[dict[str, Any]] = []
    for spec in specs:
        node = _spec_field(spec, "node")
        records.append(
            {
                "index": int(_spec_field(spec, "index")),
                "seed": int(_spec_field(spec, "seed")),
                "stratum": str(_spec_field(spec, "stratum")),
                "node": None if node is None else int(node),
            }
        )
    return _sha256_json(records)
