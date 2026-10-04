"""Command-line entry points for the DFL simulator.

Three invocation forms are supported:

* ``dfl run CONFIG --output TRACE_DIR`` (console script ``dfl``) is the S04
  production command: it runs the episode and atomically publishes a complete
  versioned schema-v1 trajectory directory.
* ``dfl calibrate CONFIG --output ARTIFACT_DIR`` (console script ``dfl``) is
  the S02 offline-calibration command: it runs the strict calibration config
  through the real reference simulator and atomically publishes a frozen
  schema-v1 Bayesian artifact directory.
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
import sys
from collections.abc import Sequence

from . import __version__
from .aggregation import AggregationError
from .bayesian_artifact import BayesianArtifactError, load_bayesian_artifact
from .bayesian_evidence import EvidenceError, collect_bayesian_evidence
from .calibration import CalibrationError, calibrate_from_config
from .calibration_artifact_writer import CalibrationPublishError
from .calibration_config import CalibrationConfigError, load_calibration_config
from .calibration_estimator import CalibrationEstimationError
from .calibration_scoring import CalibrationScoringError
from .config import BayesianAggregation, ConfigError, load_config
from .data import DataError
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


def _run_calibrate(config_path: str, output: str) -> int:
    """Run the offline calibration pipeline and publish the frozen artifact."""

    try:
        config = load_calibration_config(config_path)
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
        return _run_calibrate(calibrate_args.config, calibrate_args.output)

    if args_list and args_list[0] == "view":
        view_parser = build_view_parser()
        view_args = view_parser.parse_args(args_list[1:])
        return _run_view(view_args.trace_dir, view_args.port)

    parser = build_parser()
    args = parser.parse_args(args_list)
    return _run_legacy(args.config, args.log)


if __name__ == "__main__":
    raise SystemExit(main())
