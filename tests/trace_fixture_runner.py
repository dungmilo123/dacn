"""Standalone subprocess fixture runner for golden-trace acceptance tests (T04).

This module is *not* a test module: it is a tiny, dependency-light executable
that a test spawns in a **fresh interpreter** so a complete schema-v1 trace is
produced through the real production orchestrator
(:func:`dfl_sim.run.run_from_config`) rather than by serializing hand-built
records. It deliberately:

* accepts only list-form local paths (``--config``/``--output``);
* builds deterministic, in-memory MNIST-shaped fixtures (no network, no
  ``.gsd``/``.gitignore``/credential/external-state reads);
* never patches the runner: the only seams used are the documented
  ``dataset``/``evaluation_dataset`` injection arguments;
* prints a fixed, path-free status line so stdout cannot leak volatile data.

Running it as ``python tests/trace_fixture_runner.py`` places this file's
directory on ``sys.path``, so ``helpers`` (the shared deterministic fixture
module) imports without any environment manipulation.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from dfl_sim import load_config, run_from_config
from helpers import MNISTShapedDataset

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the fixture-runner argument parser."""

    parser = argparse.ArgumentParser(
        prog="trace_fixture_runner",
        description=(
            "Run one deterministic clean episode through run_from_config and "
            "publish a schema-v1 trace directory (offline fixtures only)."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a strict YAML experiment config (e.g. configs/clean_3node.yaml).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination trace directory (must not exist; published atomically).",
    )
    parser.add_argument(
        "--train-examples",
        type=int,
        default=192,
        help="Deterministic separable training examples (>= nodes x samples_per_node).",
    )
    parser.add_argument("--train-seed", type=int, default=5)
    parser.add_argument(
        "--eval-examples",
        type=int,
        default=96,
        help="Deterministic separable reference examples for privileged evaluation.",
    )
    parser.add_argument("--eval-seed", type=int, default=9)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one clean episode; return the process exit status."""

    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    train = MNISTShapedDataset(
        args.train_examples, seed=args.train_seed, separable=True
    )
    reference = MNISTShapedDataset(
        args.eval_examples, seed=args.eval_seed, separable=True
    )
    run_from_config(
        config,
        dataset=train,
        evaluation_dataset=reference,
        trace_path=args.output,
    )
    # Fixed, path-free status so stdout cannot leak a volatile temp path.
    print("trace_fixture_runner: published")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
