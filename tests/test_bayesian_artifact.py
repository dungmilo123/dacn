"""Contract tests for the strict frozen Bayesian artifact (T01/R013/R016).

Covers the positive hand-computable fixture load and indexing, digest and
immutability guarantees, and the Q7 negative surface: missing/extra files,
symlinked file/directory, non-regular and oversized inputs, malformed JSON,
duplicate keys, non-finite constants, unknown fields, unsupported schema or
version, byte/stale/aggregate tampering, infeasible hidden states, bad
probabilities/support/coverage/bins/cardinalities, invalid multipliers, and
duplicate rows. These tests are offline and read only checked-in fixtures.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from dfl_sim import bayesian_artifact as bayesian_artifact_module
from dfl_sim.bayesian_artifact import (
    ARTIFACT_SCHEMA,
    ARTIFACT_SCHEMA_VERSION,
    LIKELIHOOD_FILENAME,
    MANIFEST_FILENAME,
    MAX_ARTIFACT_FILE_BYTES,
    PAYOFF_FILENAME,
    BayesianArtifactError,
    canonical_json_bytes,
    compute_aggregate_digest,
    interval_bin_index,
    load_bayesian_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bayesian_v1"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_json(name: str) -> object:
    return json.loads((FIXTURE / name).read_text(encoding="utf-8"))


def _finalize(dest: Path, body: dict, lik_bytes: bytes, pay_bytes: bytes) -> Path:
    """Write a fully self-consistent artifact directory from raw payload bytes."""

    body = copy.deepcopy(body)
    body["payloads"] = {
        LIKELIHOOD_FILENAME: {
            "size_bytes": len(lik_bytes),
            "sha256": hashlib.sha256(lik_bytes).hexdigest(),
        },
        PAYOFF_FILENAME: {
            "size_bytes": len(pay_bytes),
            "sha256": hashlib.sha256(pay_bytes).hexdigest(),
        },
    }
    body["aggregate_digest"] = compute_aggregate_digest(body)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / LIKELIHOOD_FILENAME).write_bytes(lik_bytes)
    (dest / PAYOFF_FILENAME).write_bytes(pay_bytes)
    (dest / MANIFEST_FILENAME).write_text(
        json.dumps(body, indent=2) + "\n", encoding="utf-8"
    )
    return dest


def _write_artifact(
    dest: Path,
    *,
    manifest: dict | None = None,
    likelihood: list | None = None,
    payoff: list | None = None,
    support: dict | None = None,
) -> Path:
    """Write a self-consistent artifact, defaulting to the checked-in fixture."""

    body = copy.deepcopy(manifest if manifest is not None else _load_json(MANIFEST_FILENAME))
    lik = likelihood if likelihood is not None else _load_json(LIKELIHOOD_FILENAME)
    pay = payoff if payoff is not None else _load_json(PAYOFF_FILENAME)
    if support is not None:
        body["support"] = support
    else:
        existing = body.get("support", {})
        body["support"] = {
            **existing,
            "prior_cells": len(body.get("prior", [])),
            "likelihood_cells": len(lik),
            "payoff_cells": len(pay),
        }
    lik_bytes = (json.dumps(lik, indent=2) + "\n").encode()
    pay_bytes = (json.dumps(pay, indent=2) + "\n").encode()
    return _finalize(dest, body, lik_bytes, pay_bytes)


def _copy_fixture(dest: Path) -> Path:
    shutil.copytree(FIXTURE, dest)
    return dest


def _error_for(path: Path) -> BayesianArtifactError:
    with pytest.raises(BayesianArtifactError) as info:
        load_bayesian_artifact(path)
    return info.value


def _rewrite_manifest_text(path: Path, mutate) -> None:
    manifest_path = path / MANIFEST_FILENAME
    text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(mutate(text), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Positive: load, index, reproducibility, immutability
# --------------------------------------------------------------------------- #
def test_fixture_loads_and_indexes_hand_computable_cells() -> None:
    artifact = load_bayesian_artifact(FIXTURE)

    assert artifact.manifest.schema_name == ARTIFACT_SCHEMA
    assert artifact.manifest.schema_version == ARTIFACT_SCHEMA_VERSION
    assert artifact.omega == 0.5
    assert artifact.degree_cap == 4
    assert artifact.tie_tolerance == 1e-9
    assert artifact.multiplier_set == (0.0, 0.5, 1.0)
    assert artifact.self_multiplier == 1.0

    assert [(cell.state, cell.probability) for cell in artifact.prior_for_degree(1)] == [
        ((0,), 0.8),
        ((1,), 0.2),
    ]
    assert [(cell.state, cell.probability) for cell in artifact.prior_for_degree(2)] == [
        ((0, 0), 0.7),
        ((1, 0), 0.18),
        ((0, 1), 0.12),
    ]
    assert artifact.prior_for_degree(0) == ()
    assert artifact.prior_probability(1, (1,)) == 0.2
    assert artifact.prior_probability(1, (0, 1)) is None

    assert artifact.likelihood_for("1", "mid", 0, (0, 0, 0)) == 0.6
    assert artifact.likelihood_for("1", "mid", 1, (0, 0, 0)) == 0.1
    assert artifact.likelihood_for("2", "mid", 0, (1, 0, 0)) == 0.4
    # Missing cells resolve to None so the policy can fall back honestly (R016).
    assert artifact.likelihood_for("1", "early", 0, (0, 0, 0)) is None
    assert artifact.likelihood_for("1", "mid", 0, (2, 0, 0)) is None

    assert artifact.payoff_for("1", "mid", 0, (0, 0, 0), 1.0) == 1.0
    assert artifact.payoff_for("1", "mid", 1, (1, 0, 0), 1.0) == -2.0
    assert artifact.payoff_for("1", "mid", 1, (0, 0, 0), 0.5) == -0.5
    assert artifact.payoff_for("1", "mid", 0, (0, 0, 0), 0.25) is None


def test_fixture_load_is_reproducible_and_immutable() -> None:
    first = load_bayesian_artifact(FIXTURE)
    second = load_bayesian_artifact(FIXTURE)

    assert first == second
    assert first.manifest == second.manifest
    assert first.likelihood_index == second.likelihood_index
    assert first.payoff_index == second.payoff_index

    with pytest.raises(TypeError):
        first.likelihood_index[("1", "mid", 0, (0, 0, 0))] = 0.0  # type: ignore[index]
    with pytest.raises(ValidationError):
        first.manifest.provenance.p0 = 0.0  # type: ignore[misc]


def test_declared_support_counts_match_tables() -> None:
    artifact = load_bayesian_artifact(FIXTURE)

    assert artifact.manifest.support.prior_cells == len(artifact.manifest.prior)
    assert artifact.manifest.support.likelihood_cells == len(artifact.likelihood_index)
    assert artifact.manifest.support.payoff_cells == len(artifact.payoff_index)
    assert 0.0 <= artifact.manifest.coverage.likelihood <= 1.0
    assert 0.0 <= artifact.manifest.coverage.payoff <= 1.0


def test_aggregate_digest_ignores_only_its_own_field() -> None:
    body = _load_json(MANIFEST_FILENAME)
    assert isinstance(body, dict)
    expected = compute_aggregate_digest(body)
    assert expected == body["aggregate_digest"]
    tampered = {**body, "aggregate_digest": "0" * 64}
    assert compute_aggregate_digest(tampered) == expected
    assert compute_aggregate_digest({**body, "provenance": {}}) != expected


def test_interval_bin_index_convention_and_guard() -> None:
    artifact = load_bayesian_artifact(FIXTURE)
    bins = artifact.manifest.bins.disagreement
    assert [interval_bin_index(bins, v) for v in (0.0, 0.49, 0.5, 0.99, 1.0, 3.0)] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    with pytest.raises(BayesianArtifactError) as info:
        interval_bin_index(bins, float("nan"))
    assert info.value.validation_class == "non_finite"


def test_canonical_json_bytes_rejects_non_finite() -> None:
    with pytest.raises(BayesianArtifactError) as info:
        canonical_json_bytes({"x": float("inf")})
    assert info.value.validation_class == "schema"


# --------------------------------------------------------------------------- #
# Negative: directory and file-level failures
# --------------------------------------------------------------------------- #
def test_missing_directory_fails_closed(tmp_path: Path) -> None:
    error = _error_for(tmp_path / "nope")
    assert error.validation_class == "missing"


def test_missing_fixed_file_names_the_file(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path / "artifact")
    os.remove(path / PAYOFF_FILENAME)
    error = _error_for(path)
    assert error.validation_class == "missing"
    assert error.filename == PAYOFF_FILENAME


def test_extra_file_is_rejected(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path / "artifact")
    (path / "sneaky.json").write_text("{}", encoding="utf-8")
    error = _error_for(path)
    assert error.validation_class == "unexpected_file"


def test_symlinked_directory_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "link"
    os.symlink(FIXTURE, link)
    error = _error_for(link)
    assert error.validation_class == "symlink"


def test_symlinked_file_is_rejected(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path / "artifact")
    os.remove(path / LIKELIHOOD_FILENAME)
    os.symlink(FIXTURE / LIKELIHOOD_FILENAME, path / LIKELIHOOD_FILENAME)
    error = _error_for(path)
    assert error.validation_class == "symlink"
    assert error.filename == LIKELIHOOD_FILENAME


def test_non_regular_file_is_rejected(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path / "artifact")
    os.remove(path / LIKELIHOOD_FILENAME)
    os.mkfifo(path / LIKELIHOOD_FILENAME)
    error = _error_for(path)
    assert error.validation_class == "non_regular"


def test_oversized_payload_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _copy_fixture(tmp_path / "artifact")
    (path / LIKELIHOOD_FILENAME).write_bytes(b"x" * 5000)
    monkeypatch.setattr(bayesian_artifact_module, "MAX_ARTIFACT_FILE_BYTES", 4096)
    assert MAX_ARTIFACT_FILE_BYTES > 4096
    error = _error_for(path)
    assert error.validation_class == "oversized"
    assert error.filename == LIKELIHOOD_FILENAME


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path / "artifact")
    _rewrite_manifest_text(path, lambda _text: "{ not json")
    error = _error_for(path)
    assert error.validation_class == "malformed_json"


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path / "artifact")
    _rewrite_manifest_text(
        path,
        lambda text: text.replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 2,'),
    )
    error = _error_for(path)
    assert error.validation_class == "duplicate_key"


def test_non_finite_json_constant_is_rejected(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path / "artifact")
    _rewrite_manifest_text(path, lambda text: text.replace('"p0": 0.5', '"p0": NaN'))
    error = _error_for(path)
    assert error.validation_class == "non_finite"


# --------------------------------------------------------------------------- #
# Negative: manifest schema / provenance
# --------------------------------------------------------------------------- #
def test_unknown_manifest_field_is_rejected(tmp_path: Path) -> None:
    manifest = _load_json(MANIFEST_FILENAME)
    assert isinstance(manifest, dict)
    manifest["surprise"] = 1
    path = _write_artifact(tmp_path / "artifact", manifest=manifest)
    error = _error_for(path)
    assert error.validation_class == "schema"


def test_unsupported_schema_name_is_rejected(tmp_path: Path) -> None:
    manifest = _load_json(MANIFEST_FILENAME)
    assert isinstance(manifest, dict)
    manifest["schema"] = "some_other_artifact"
    path = _write_artifact(tmp_path / "artifact", manifest=manifest)
    error = _error_for(path)
    assert error.validation_class == "schema"


def test_unsupported_schema_version_is_rejected(tmp_path: Path) -> None:
    manifest = _load_json(MANIFEST_FILENAME)
    assert isinstance(manifest, dict)
    manifest["schema_version"] = 2
    path = _write_artifact(tmp_path / "artifact", manifest=manifest)
    error = _error_for(path)
    assert error.validation_class == "version"


def test_invalid_action_space_multiplier_set_is_rejected(tmp_path: Path) -> None:
    manifest = _load_json(MANIFEST_FILENAME)
    assert isinstance(manifest, dict)
    manifest["action_space"]["multiplier_set"] = [0.0, 0.5, 0.9]
    path = _write_artifact(tmp_path / "artifact", manifest=manifest)
    error = _error_for(path)
    assert error.validation_class == "action_space"


def test_coverage_outside_unit_interval_is_rejected(tmp_path: Path) -> None:
    manifest = _load_json(MANIFEST_FILENAME)
    assert isinstance(manifest, dict)
    manifest["coverage"]["likelihood"] = 1.5
    path = _write_artifact(tmp_path / "artifact", manifest=manifest)
    error = _error_for(path)
    assert error.validation_class == "coverage"


def test_support_cardinality_mismatch_is_rejected(tmp_path: Path) -> None:
    path = _write_artifact(
        tmp_path / "artifact",
        support={
            "calibration_episodes": 4,
            "calibration_rounds": 12,
            "prior_cells": 99,
            "likelihood_cells": 8,
            "payoff_cells": 24,
            "min_cell_count": 1,
        },
    )
    error = _error_for(path)
    assert error.validation_class == "support"


# --------------------------------------------------------------------------- #
# Negative: prior semantics
# --------------------------------------------------------------------------- #
def test_infeasible_prior_state_is_rejected(tmp_path: Path) -> None:
    manifest = _load_json(MANIFEST_FILENAME)
    assert isinstance(manifest, dict)
    manifest["prior"][3]["state"] = [1, 1]
    path = _write_artifact(tmp_path / "artifact", manifest=manifest)
    error = _error_for(path)
    assert error.validation_class == "state"


def test_prior_state_cardinality_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest = _load_json(MANIFEST_FILENAME)
    assert isinstance(manifest, dict)
    manifest["prior"][3]["state"] = [1, 0, 0]
    path = _write_artifact(tmp_path / "artifact", manifest=manifest)
    error = _error_for(path)
    assert error.validation_class == "cardinality"


def test_prior_probability_out_of_range_is_rejected(tmp_path: Path) -> None:
    manifest = _load_json(MANIFEST_FILENAME)
    assert isinstance(manifest, dict)
    manifest["prior"][1]["probability"] = 1.5
    path = _write_artifact(tmp_path / "artifact", manifest=manifest)
    error = _error_for(path)
    assert error.validation_class == "probability"


def test_prior_mass_not_normalized_is_rejected(tmp_path: Path) -> None:
    manifest = _load_json(MANIFEST_FILENAME)
    assert isinstance(manifest, dict)
    manifest["prior"][1]["probability"] = 0.1
    path = _write_artifact(tmp_path / "artifact", manifest=manifest)
    error = _error_for(path)
    assert error.validation_class == "prior"


# --------------------------------------------------------------------------- #
# Negative: payload tampering and row semantics
# --------------------------------------------------------------------------- #
def test_payload_byte_tamper_is_rejected(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path / "artifact")
    payload_path = path / LIKELIHOOD_FILENAME
    data = bytearray(payload_path.read_bytes())
    data[-2] = data[-2] ^ 0xFF
    payload_path.write_bytes(bytes(data))
    error = _error_for(path)
    assert error.validation_class == "digest"
    assert error.filename == LIKELIHOOD_FILENAME


def test_stale_declared_digest_is_rejected(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path / "artifact")
    manifest = json.loads((path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["payloads"][LIKELIHOOD_FILENAME]["sha256"] = "0" * 64
    (path / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    error = _error_for(path)
    assert error.validation_class == "digest"


def test_aggregate_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path / "artifact")
    manifest = json.loads((path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["provenance"]["p0"] = 0.6
    (path / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    error = _error_for(path)
    assert error.validation_class == "aggregate_digest"


def test_duplicate_likelihood_row_is_rejected(tmp_path: Path) -> None:
    likelihood = _load_json(LIKELIHOOD_FILENAME)
    assert isinstance(likelihood, list)
    likelihood.append(copy.deepcopy(likelihood[0]))
    path = _write_artifact(tmp_path / "artifact", likelihood=likelihood)
    error = _error_for(path)
    assert error.validation_class == "duplicate_row"


def test_duplicate_payoff_row_is_rejected(tmp_path: Path) -> None:
    payoff = _load_json(PAYOFF_FILENAME)
    assert isinstance(payoff, list)
    payoff.append(copy.deepcopy(payoff[0]))
    path = _write_artifact(tmp_path / "artifact", payoff=payoff)
    error = _error_for(path)
    assert error.validation_class == "duplicate_row"


def test_evidence_bin_out_of_range_is_rejected(tmp_path: Path) -> None:
    likelihood = _load_json(LIKELIHOOD_FILENAME)
    assert isinstance(likelihood, list)
    likelihood[0]["evidence_bin"] = [9, 0, 0]
    path = _write_artifact(tmp_path / "artifact", likelihood=likelihood)
    error = _error_for(path)
    assert error.validation_class == "bins"


def test_unknown_context_label_is_rejected(tmp_path: Path) -> None:
    likelihood = _load_json(LIKELIHOOD_FILENAME)
    assert isinstance(likelihood, list)
    likelihood[0]["degree_bin"] = "99"
    path = _write_artifact(tmp_path / "artifact", likelihood=likelihood)
    error = _error_for(path)
    assert error.validation_class == "bins"


def test_invalid_payoff_multiplier_is_rejected(tmp_path: Path) -> None:
    payoff = _load_json(PAYOFF_FILENAME)
    assert isinstance(payoff, list)
    payoff[0]["multiplier"] = 0.7
    path = _write_artifact(tmp_path / "artifact", payoff=payoff)
    error = _error_for(path)
    assert error.validation_class == "action_space"


def test_bad_payload_probability_does_not_echo_table_contents(tmp_path: Path) -> None:
    likelihood = _load_json(LIKELIHOOD_FILENAME)
    assert isinstance(likelihood, list)
    likelihood[0]["probability"] = 1.5
    path = _write_artifact(tmp_path / "artifact", likelihood=likelihood)
    error = _error_for(path)
    assert error.validation_class == "probability"
    message = str(error)
    assert "1.5" not in message
    assert "0.6" not in message
    assert "degree_bin" not in message


def test_duplicate_json_key_in_payload_is_rejected(tmp_path: Path) -> None:
    body = copy.deepcopy(_load_json(MANIFEST_FILENAME))
    assert isinstance(body, dict)
    pay_bytes = (json.dumps(_load_json(PAYOFF_FILENAME), indent=2) + "\n").encode()
    lik_bytes = (
        b'[{"degree_bin":"1","progress":"mid","local_type":0,'
        b'"evidence_bin":[0,0,0],"probability":0.6,"probability":0.6}]'
    )
    path = _finalize(tmp_path / "artifact", body, lik_bytes, pay_bytes)
    error = _error_for(path)
    assert error.validation_class == "duplicate_key"
    assert error.filename == LIKELIHOOD_FILENAME
