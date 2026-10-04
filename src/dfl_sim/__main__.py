"""Command-line entry points for the DFL simulator.

The following invocation forms are supported:

* ``dfl run CONFIG --output TRACE_DIR`` (console script ``dfl``) is the S04
  production command: it runs the episode and atomically publishes a complete
  versioned schema-v1 trajectory directory.
* ``dfl calibrate CONFIG --output ARTIFACT_DIR`` (console script ``dfl``) is
  the S02 offline-calibration command: it runs the strict calibration config
  through the real reference simulator and atomically publishes a frozen
  schema-v1 Bayesian artifact directory.
* ``dfl evaluate STUDY.yaml --output DIR`` (console script ``dfl``) is the S01
  baseline-study command: it runs the strict six-policy paired panel through
  the real policy factory/simulator and atomically publishes one schema-v1
  trace per (policy, seed) cell plus a versioned baseline manifest.
* ``dfl view TRACE_DIR [--port PORT]`` (console script ``dfl``) is the S05
  read-only local replay server: it validates and fully projects the trace
  once, then serves the Dash viewer on ``127.0.0.1`` only.
* ``python -m dfl_sim CONFIG --log PATH`` is the backward-compatible minimal-log
  form retained from S01.

All forms accept only local paths, reuse the strict config parser, and return
a nonzero status on any failure. Failures are reported on stderr with the
underlying context and never expose a stack trace or a staging path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .ablation import CALIBRATION_VARIANTS
from .ablation_evaluation import AblationEvaluationError
from .aggregation import AggregationError
from .bayesian_artifact import BayesianArtifactError, load_bayesian_artifact
from .bayesian_evidence import EvidenceError, collect_bayesian_evidence
from .calibration import CalibrationError, calibrate_from_config
from .calibration_artifact_writer import CalibrationPublishError
from .calibration_config import (
    CalibrationConfigError,
    load_calibration_config,
    parse_calibration_text,
)
from .calibration_estimator import CalibrationEstimationError
from .calibration_scoring import CalibrationScoringError
from .config import BayesianAggregation, ConfigError, load_config
from .data import DataError
from .evaluation import (
    SELECTION_MANIFEST_FILENAME,
    EvaluationError,
    run_baseline_study,
    run_final_phase,
    run_held_out_study,
)
from .evaluation_config import (
    BaselineStudyConfig,
    BaselineStudyConfigError,
    HeldOutStudyConfig,
    HeldOutStudyConfigError,
    load_any_study_config,
)
from .evaluator import EvaluatorError
from .models import ModelError
from .node import TrainingError
from .policy import BayesianDiagnostic
from .reproducibility import DeterminismError
from .round_log import LogError
from .run import EpisodeError, run_from_config
from .topology import TopologyError
from .trajectory import TrajectoryError
from .viewer import ViewerError, create_viewer_app

__all__ = [
    "build_calibrate_parser",
    "build_evaluate_parser",
    "build_parser",
    "build_run_parser",
    "build_view_parser",
    "main",
]

#: The viewer is loopback-only; it never binds a routable interface.
VIEWER_HOST = "127.0.0.1"
#: Default local port for ``dfl view``.
VIEWER_DEFAULT_PORT = 8050

#: Stable stdout marker prefixing the one canonical Bayesian evidence record.
BAYESIAN_REPORT_MARKER = "dfl_sim: bayesian_report "

#: Per-cell progress marker emitted to stderr as each (policy, seed) cell runs.
#: It carries only the bounded policy id and seed, never YAML/artifact content,
#: and is written to stderr so stdout stays a single machine-readable success
#: marker.
EVALUATE_PROGRESS_MARKER = "dfl_sim: evaluate cell "

#: Stable stdout marker for a published baseline study; it names the manifest
#: and output directory so the command's one success line is unambiguous.
EVALUATE_SUCCESS_MARKER = "dfl_sim: baseline study "

#: Per-cell selection-phase progress marker emitted to stderr (M003/S02).
#: It carries only the bounded policy id, seed, and schedule id.
HELD_OUT_SELECTION_PROGRESS_MARKER = "dfl_sim: selection cell "

#: Per-cell final-phase progress marker emitted to stderr (M003/S02).
HELD_OUT_FINAL_PROGRESS_MARKER = "dfl_sim: final cell "

#: Stable stdout marker for a published held-out study; it names both manifest
#: basenames, the output root, panel size, seed counts, and cell counts.
HELD_OUT_SUCCESS_MARKER = "dfl_sim: held out study "

# Domain failures that should be reported as a clean CLI error (nonzero exit)
# rather than an uncaught traceback.
_CLI_ERRORS = (
    ConfigError,
    DataError,
    DeterminismError,
    LogError,
    TopologyError,
    EpisodeError,
    EvaluatorError,
    TrajectoryError,
    AggregationError,
    ModelError,
    TrainingError,
    OSError,
)

# The bounded offline-calibration failure family: a malformed calibration
# config, a data/simulation failure while collecting reference episodes, a
# scoring or estimation error, a publication/atomicity failure, or an
# underlying filesystem error. Every one is reported as a concise stderr line
# with a nonzero exit and never a stack trace or partial publication.
_CALIBRATION_ERRORS = _CLI_ERRORS + (
    CalibrationConfigError,
    CalibrationError,
    CalibrationScoringError,
    CalibrationEstimationError,
    CalibrationPublishError,
)

# The bounded baseline-study failure family: a malformed strict study spec, a
# non-publishing safety rejection, or any per-cell simulation/artifact/trace
# failure the runner already wrapped with its policy id and seed. Each is
# reported as a concise stderr line with a nonzero exit and never a stack trace
# or a partial/success manifest. The held-out spec errors are subclasses of
# ConfigError but are listed explicitly so the bounded family mirrors the
# held-out contract.
_EVALUATE_ERRORS = _CLI_ERRORS + (
    EvaluationError,
    BaselineStudyConfigError,
    HeldOutStudyConfigError,
    # The opt-in matched-ablation preflight raises a bounded RuntimeError whose
    # validation class names the mismatch. It must be reported as a clean CLI
    # error (nonzero exit, no traceback, no partial publication) rather than
    # escaping as an uncaught exception.
    AblationEvaluationError,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the backward-compatible ``python -m dfl_sim`` parser."""

    parser = argparse.ArgumentParser(
        prog="python -m dfl_sim",
        description=(
            "Run a deterministic synchronous DFL episode from a strict YAML "
            "config and write a canonical JSONL round log."
        ),
        epilog=(
            "For the S04 production command, use "
            "'dfl run CONFIG --output TRACE_DIR' to publish a versioned "
            "schema-v1 trajectory directory."
        ),
    )
    parser.add_argument(
        "config",
        help="Path to a strict YAML experiment config (e.g. configs/clean_3node.yaml).",
    )
    parser.add_argument(
        "--log",
        metavar="PATH",
        default=None,
        help="Write the canonical JSONL round log to PATH (atomic replace).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"dfl-sim {__version__}",
    )
    return parser


def build_run_parser() -> argparse.ArgumentParser:
    """Build the ``dfl run`` parser that publishes a trace directory."""

    parser = argparse.ArgumentParser(
        prog="dfl run",
        description=(
            "Run a deterministic synchronous DFL episode from a strict YAML "
            "config and publish a complete schema-v1 trajectory directory."
        ),
    )
    parser.add_argument(
        "config",
        help="Path to a strict YAML experiment config (e.g. configs/clean_3node.yaml).",
    )
    parser.add_argument(
        "--output",
        metavar="TRACE_DIR",
        required=True,
        help=(
            "Destination directory for the versioned trace. Must not exist "
            "and must not alias the config; published atomically."
        ),
    )
    parser.add_argument(
        "--bayesian-report",
        action="store_true",
        help=(
            "After publishing the trace, print one canonical bounded on-policy "
            "evidence record (coverage, mismatch, fallback, pruning, candidate, "
            "and timing aggregates). Requires a bayesian aggregation config; "
            "diagnostics are never written into the trace directory."
        ),
    )
    return parser


def build_calibrate_parser() -> argparse.ArgumentParser:
    """Build the ``dfl calibrate`` parser that publishes a frozen artifact."""

    parser = argparse.ArgumentParser(
        prog="dfl calibrate",
        description=(
            "Run a strict offline calibration from a dedicated calibration "
            "config and atomically publish a frozen schema-v1 Bayesian "
            "artifact directory accepted by the online loader."
        ),
    )
    parser.add_argument(
        "config",
        help=(
            "Path to a strict calibration YAML/JSON config "
            "(e.g. configs/calibrate_chain3.yaml)."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="ARTIFACT_DIR",
        required=True,
        help=(
            "Destination directory for the frozen artifact. Must not exist "
            "and must not alias the config or data root; published "
            "atomically after staged loader validation."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=CALIBRATION_VARIANTS,
        default=None,
        help=(
            "Override the loaded calibration config's ablation variant "
            "(full, no_belief, or no_strategic). Without this flag the "
            "config's own variant is used (default full). The override is "
            "revalidated through the strict calibration model and is the "
            "sole requested variant, not a second competing sampler setting."
        ),
    )
    return parser


def build_evaluate_parser() -> argparse.ArgumentParser:
    """Build the ``dfl evaluate`` parser that publishes a baseline study."""

    parser = argparse.ArgumentParser(
        prog="dfl evaluate",
        description=(
            "Run a strict six-policy paired baseline study through the real "
            "policy factory and simulator, and atomically publish one "
            "schema-v1 trace per (policy, seed) cell plus a versioned "
            "baseline manifest."
        ),
    )
    parser.add_argument(
        "study",
        help=(
            "Path to a strict baseline-study YAML/JSON spec declaring the "
            "frozen base experiment, bounded seeds, and the six-policy panel."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        required=True,
        help=(
            "Destination directory for the study. Must not exist and must "
            "not alias the study spec; published atomically after every cell "
            "and every trace validates."
        ),
    )
    parser.add_argument(
        "--selection-output",
        metavar="DIR",
        default=None,
        help=(
            "Run only the final phase against a previously frozen selection "
            "output directory (held-out study specs only). --output then "
            "receives final-manifest.json directly instead of a two-phase "
            "study root."
        ),
    )
    parser.add_argument(
        "--resume-from",
        metavar="DIR",
        default=None,
        help=(
            "Resume a held-out study from a prior study root or retained "
            "staging directory. Manifest-matched completed cells are reused "
            "after their traces are revalidated; only missing or mismatched "
            "cells are recomputed. A digest mismatch fails closed."
        ),
    )
    return parser


def build_view_parser() -> argparse.ArgumentParser:
    """Build the ``dfl view`` parser for the read-only local replay server."""

    parser = argparse.ArgumentParser(
        prog="dfl view",
        description=(
            "Serve a read-only local replay of a completed schema-v1 trace. "
            "The trace is validated and projected once at startup, and the "
            "server binds to 127.0.0.1 only; nothing is uploaded or rewritten."
        ),
    )
    parser.add_argument(
        "trace_dir",
        metavar="TRACE_DIR",
        help="Path to a completed schema-v1 trajectory directory.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=VIEWER_DEFAULT_PORT,
        help=(
            "Local TCP port to serve on, 1-65535 "
            f"(default {VIEWER_DEFAULT_PORT})."
        ),
    )
    return parser


def _run_legacy(config_path: str, log_path: str | None) -> int:
    try:
        result = run_from_config(config_path, log_path=log_path)
    except _CLI_ERRORS as exc:
        print(f"dfl_sim: error: {exc}", file=sys.stderr)
        return 1

    node_count = len(result.final_states)
    round_count = len(result.rounds)
    if result.log_path is not None:
        print(
            f"dfl_sim: wrote {round_count} round(s) for {node_count} node(s) "
            f"to {result.log_path}"
        )
    else:
        print(f"dfl_sim: completed {round_count} round(s) for {node_count} node(s)")
    return 0


def _run_trace(
    config_path: str, output: str, bayesian_report: bool = False
) -> int:
    diagnostics: list[BayesianDiagnostic] = []

    # Reject a report request for a non-Bayesian config *before* running, so a
    # trace is never published for a command that cannot honor its flag.
    if bayesian_report:
        try:
            config = load_config(config_path)
        except _CLI_ERRORS as exc:
            print(f"dfl_sim: error: {exc}", file=sys.stderr)
            return 1
        if not isinstance(config.aggregation, BayesianAggregation):
            print(
                "dfl_sim: error: --bayesian-report requires a bayesian "
                f"aggregation config, got {type(config.aggregation).__name__}",
                file=sys.stderr,
            )
            return 1

    try:
        result = run_from_config(
            config_path,
            trace_path=output,
            diagnostic_sink=diagnostics.append if bayesian_report else None,
        )
    except _CLI_ERRORS as exc:
        print(f"dfl_sim: error: {exc}", file=sys.stderr)
        return 1

    node_count = len(result.final_states)
    round_count = len(result.rounds)
    print(
        f"dfl_sim: wrote trace with {round_count} round(s) for "
        f"{node_count} node(s) to {result.trace_path}"
    )

    # The report is emitted only after the trace has been atomically published.
    if bayesian_report:
        aggregation = result.config.aggregation
        assert isinstance(aggregation, BayesianAggregation)
        try:
            artifact = load_bayesian_artifact(aggregation.artifact_dir)
            report = collect_bayesian_evidence(
                diagnostics, config=result.config, artifact=artifact
            )
        except (EvidenceError, BayesianArtifactError) as exc:
            print(
                f"dfl_sim: error: bayesian report failed: {exc}",
                file=sys.stderr,
            )
            return 1
        print(f"{BAYESIAN_REPORT_MARKER}{report.canonical_json()}")
    return 0


def _run_calibrate(
    config_path: str, output: str, variant: str | None = None
) -> int:
    """Run the offline calibration pipeline and publish the frozen artifact.

    ``variant`` is the optional ``--variant`` override. When supplied it
    replaces the loaded config's declared variant and the whole config is
    revalidated through the strict calibration model, so an override can never
    smuggle in an invalid variant/mixture combination. The variant itself is
    reported through the published artifact metadata, never through a raw
    config dump on stdout.
    """

    try:
        config = load_calibration_config(config_path)
        if variant is not None:
            payload = config.model_dump(mode="json")
            payload["variant"] = variant
            config = parse_calibration_text(json.dumps(payload, sort_keys=True))
        result = calibrate_from_config(
            config,
            output,
            config_path=config_path,
            data_root=config.experiment.data.data_root,
        )
    except _CALIBRATION_ERRORS as exc:
        print(f"dfl_sim: error: {exc}", file=sys.stderr)
        return 1

    manifest = result.manifest
    support = manifest.support
    coverage = manifest.coverage
    print(
        f"dfl_sim: wrote calibration artifact to {result.path} "
        f"(episodes={support.calibration_episodes} "
        f"rounds={support.calibration_rounds} "
        f"prior_cells={support.prior_cells} "
        f"likelihood_cells={support.likelihood_cells} "
        f"payoff_cells={support.payoff_cells} "
        f"min_cell_count={support.min_cell_count} "
        f"likelihood_coverage={coverage.likelihood:.6f} "
        f"payoff_coverage={coverage.payoff:.6f} "
        f"posterior_calibration={coverage.posterior_calibration:.6f} "
        f"aggregate_digest={manifest.aggregate_digest})"
    )
    return 0


def _run_baseline_evaluate(spec: BaselineStudyConfig, output: str) -> int:
    """Run the S01 paired baseline study (unchanged success surface)."""

    def _progress(seed: int, policy_id: str) -> None:
        # Progress is bounded (policy id + seed only) and goes to stderr so
        # stdout stays a single stable success marker. It never echoes the
        # untrusted study YAML or artifact contents.
        print(
            f"{EVALUATE_PROGRESS_MARKER}policy={policy_id} seed={int(seed)}",
            file=sys.stderr,
        )

    try:
        result = run_baseline_study(spec, output, progress=_progress)
    except _EVALUATE_ERRORS as exc:
        print(f"dfl_sim: error: {exc}", file=sys.stderr)
        return 1

    manifest = result.manifest
    print(
        f"{EVALUATE_SUCCESS_MARKER}manifest={result.manifest_path} "
        f"output={result.output_path} "
        f"policies={len(manifest['panel'])} "
        f"seeds={len(manifest['seeds'])} "
        f"cells={len(result.cells)}"
    )
    return 0


def _held_out_progress(
    phase: str, seed: int, schedule_id: str, policy_id: str
) -> None:
    """Emit one bounded per-cell held-out progress line to stderr."""

    marker = (
        HELD_OUT_SELECTION_PROGRESS_MARKER
        if phase == "selection"
        else HELD_OUT_FINAL_PROGRESS_MARKER
    )
    print(
        f"{marker}policy={policy_id} seed={int(seed)} "
        f"schedule={schedule_id}",
        file=sys.stderr,
    )


def _held_out_success_marker(
    spec: HeldOutStudyConfig,
    *,
    selection_manifest: Path,
    final_manifest: Path,
    output: Path,
    selection_cells: int,
    final_cells: int,
) -> str:
    """Build the one bounded held-out success line."""

    return (
        f"{HELD_OUT_SUCCESS_MARKER}"
        f"selection_manifest={selection_manifest} "
        f"final_manifest={final_manifest} "
        f"output={output} "
        f"policies={len(spec.study.policy_ids())} "
        f"selection_seeds={len(spec.study.seeds)} "
        f"final_seeds={len(spec.final_seeds)} "
        f"selection_cells={selection_cells} "
        f"final_cells={final_cells}"
    )


def _run_held_out_evaluate(
    spec: HeldOutStudyConfig,
    output: str,
    selection_output: str | None,
    resume_from: str | None = None,
) -> int:
    """Run the held-out study (full two-phase, or final-only with a freeze)."""

    if selection_output is not None:
        # Final-only: read an existing frozen selection output and publish
        # final-manifest.json directly into ``output``.
        try:
            result = run_final_phase(
                spec,
                selection_output,
                output,
                progress=lambda seed, schedule_id, policy_id: _held_out_progress(
                    "final", seed, schedule_id, policy_id
                ),
                resume_from=resume_from,
            )
        except _EVALUATE_ERRORS as exc:
            print(f"dfl_sim: error: {exc}", file=sys.stderr)
            return 1
        selection_cells = (
            len(spec.study.seeds)
            * len(spec.known_schedules)
            * len(spec.study.policy_ids())
        )
        marker = _held_out_success_marker(
            spec,
            selection_manifest=(
                Path(selection_output) / SELECTION_MANIFEST_FILENAME
            ),
            final_manifest=result.manifest_path,
            output=result.output_path,
            selection_cells=selection_cells,
            final_cells=len(result.cells),
        )
        print(marker)
        return 0

    try:
        study_result = run_held_out_study(
            spec, output, progress=_held_out_progress, resume_from=resume_from
        )
    except _EVALUATE_ERRORS as exc:
        print(f"dfl_sim: error: {exc}", file=sys.stderr)
        return 1

    marker = _held_out_success_marker(
        spec,
        selection_manifest=study_result.selection_manifest_path,
        final_manifest=study_result.final_manifest_path,
        output=study_result.output_path,
        selection_cells=len(study_result.selection_result.cells),
        final_cells=len(study_result.final_result.cells),
    )
    print(marker)
    return 0


def _run_evaluate(
    study_path: str,
    output: str,
    selection_output: str | None = None,
    resume_from: str | None = None,
) -> int:
    """Dispatch ``dfl evaluate`` to the baseline or held-out study contract."""

    try:
        spec = load_any_study_config(study_path)
    except _EVALUATE_ERRORS as exc:
        print(f"dfl_sim: error: {exc}", file=sys.stderr)
        return 1

    if isinstance(spec, HeldOutStudyConfig):
        return _run_held_out_evaluate(
            spec, output, selection_output, resume_from
        )

    if selection_output is not None:
        print(
            "dfl_sim: error: --selection-output is only valid for a "
            "held-out study spec; a baseline study runs selection and final "
            "in one command",
            file=sys.stderr,
        )
        return 1

    if resume_from is not None:
        print(
            "dfl_sim: error: --resume-from is only valid for a held-out "
            "study spec",
            file=sys.stderr,
        )
        return 1

    return _run_baseline_evaluate(spec, output)


def _run_view(trace_dir: str, port: int) -> int:
    """Validate a trace, then serve the read-only replay on loopback."""

    if not 1 <= port <= 65535:
        print(
            f"dfl_sim: error: port must be between 1 and 65535, got {port}",
            file=sys.stderr,
        )
        return 2

    # Project the trace fully before touching the network; an invalid trace
    # stops here with a concise message and never starts a server.
    try:
        app = create_viewer_app(trace_dir)
    except (ViewerError, TrajectoryError) as exc:
        print(f"dfl_sim: error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        # A missing Dash/Plotly install is a view-command startup failure, not
        # a simulator import failure; report it without a stack trace.
        print(
            f"dfl_sim: error: viewer UI dependencies unavailable: {exc}",
            file=sys.stderr,
        )
        return 1

    url = f"http://{VIEWER_HOST}:{port}/"
    print(f"dfl_sim: serving read-only replay at {url}")
    try:
        # Loopback only, no debug console, and no reloader (single process).
        app.run(host=VIEWER_HOST, port=port, debug=False, use_reloader=False)
    except OSError as exc:
        print(f"dfl_sim: error: cannot start viewer on {url}: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; return the process exit status."""

    args_list = list(sys.argv[1:] if argv is None else argv)

    if args_list and args_list[0] == "run":
        run_parser = build_run_parser()
        run_args = run_parser.parse_args(args_list[1:])
        return _run_trace(
            run_args.config, run_args.output, run_args.bayesian_report
        )

    if args_list and args_list[0] == "calibrate":
        calibrate_parser = build_calibrate_parser()
        calibrate_args = calibrate_parser.parse_args(args_list[1:])
        return _run_calibrate(
            calibrate_args.config, calibrate_args.output, calibrate_args.variant
        )

    if args_list and args_list[0] == "evaluate":
        evaluate_parser = build_evaluate_parser()
        evaluate_args = evaluate_parser.parse_args(args_list[1:])
        return _run_evaluate(
            evaluate_args.study,
            evaluate_args.output,
            evaluate_args.selection_output,
            evaluate_args.resume_from,
        )

    if args_list and args_list[0] == "view":
        view_parser = build_view_parser()
        view_args = view_parser.parse_args(args_list[1:])
        return _run_view(view_args.trace_dir, view_args.port)

    parser = build_parser()
    args = parser.parse_args(args_list)
    return _run_legacy(args.config, args.log)


if __name__ == "__main__":
    raise SystemExit(main())
