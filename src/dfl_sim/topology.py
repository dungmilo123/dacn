"""Topology construction (R002, D013/D014).

``load_topology(config)`` returns a *frozen*, node-normalized, connected
:class:`networkx.Graph`:

* ``line2`` -- exact edge ``0 -- 1``.
* ``chain3`` -- exact edges ``0 -- 1 -- 2``.
* ``watts_strogatz`` -- a D014 seeded connected Watts-Strogatz graph with
  10-20 nodes, an even ring degree, and a bounded number of generation
  attempts.

Node IDs are normalized to contiguous integers and the returned graph is
frozen with :func:`networkx.freeze`, so downstream phases cannot mutate the
episode's communication topology.
"""

from __future__ import annotations

import networkx as nx

from .config import (
    Chain3Topology,
    ExperimentConfig,
    Line2Topology,
    WattsStrogatzTopology,
)
from .reproducibility import resolve_seeds

__all__ = ["MAX_WS_TRIES", "TopologyError", "load_topology"]

# Finite generation attempts: an impossible (n, k, p) fails loudly instead of
# looping unbounded (Q6 load profile).
MAX_WS_TRIES = 100

_EXPECTED_NODE_COUNTS = {"line2": 2, "chain3": 3}


class TopologyError(RuntimeError):
    """Raised when a configured topology cannot be constructed."""


def _generate_connected_ws(topology: WattsStrogatzTopology, seed: int) -> nx.Graph:
    try:
        return nx.connected_watts_strogatz_graph(
            topology.n,
            topology.k,
            topology.p,
            tries=MAX_WS_TRIES,
            seed=int(seed),
        )
    except nx.NetworkXError as exc:
        raise TopologyError(
            "could not generate a connected Watts-Strogatz graph for "
            f"n={topology.n}, k={topology.k}, p={topology.p} within "
            f"{MAX_WS_TRIES} attempts"
        ) from exc


def _validate(graph: nx.Graph, kind: str) -> None:
    if graph.number_of_nodes() == 0:
        raise TopologyError(f"{kind} topology produced no nodes")
    expected = _EXPECTED_NODE_COUNTS.get(kind)
    if expected is not None and graph.number_of_nodes() != expected:
        raise TopologyError(
            f"{kind} topology must have exactly {expected} nodes, "
            f"got {graph.number_of_nodes()}"
        )
    if not nx.is_connected(graph):
        raise TopologyError(f"{kind} topology must be connected")


def load_topology(config: ExperimentConfig) -> nx.Graph:
    """Build and freeze the topology described by ``config``."""

    topology = config.topology
    generator_seed: int | None = None

    if isinstance(topology, Line2Topology):
        graph = nx.Graph()
        graph.add_edges_from([(0, 1)])
    elif isinstance(topology, Chain3Topology):
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2)])
    elif isinstance(topology, WattsStrogatzTopology):
        generator_seed = resolve_seeds(config).topology
        graph = _generate_connected_ws(topology, generator_seed)
    else:  # pragma: no cover - discriminated union prevents this
        raise TopologyError(f"unsupported topology: {topology!r}")

    # Normalize to contiguous integer node IDs (stable, sorted ordering).
    normalized = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    _validate(normalized, topology.kind)

    normalized.graph["kind"] = topology.kind
    normalized.graph["node_count"] = normalized.number_of_nodes()
    if generator_seed is not None:
        normalized.graph["generator_seed"] = int(generator_seed)

    return nx.freeze(normalized)
