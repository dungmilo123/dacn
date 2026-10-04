"""Deterministic seed-derived degree-placement resolution (R020, D040).

An adaptive study resolves the one compromised node by *degree rank* on the
**realized** seed-derived topology, not by a prechosen scalar node id. This
module owns that pure resolution contract:

* ``low`` / ``mid`` / ``high`` name ranks on the deterministic
  ``(degree, node_id)`` ordering of the frozen :func:`~dfl_sim.topology.load_topology`
  graph. The graph is never resampled, regenerated, or selected by a metric.
* The middle rank for an even node count is pinned to the lower middle
  (:data:`MIDDLE_RANK_CONVENTION`), so the choice is reproducible.
* Rank ties and degree collisions are reported honestly: a graph whose three
  canonical ranks do not carry three distinct degrees never claims three
  distinct degree groups, and a regular graph is flagged ``regular=True``.

The resolver never touches privileged evaluator state or policy observations:
it consumes only the validated :class:`~dfl_sim.config.ExperimentConfig`
topology seed and returns evaluator-only metadata (node, degree, rank, ordered
edges, graph/content digests).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import networkx as nx

from .config import ExperimentConfig
from .topology import load_topology

__all__ = [
    "MIDDLE_RANK_CONVENTION",
    "PLACEMENT_RANKS",
    "PlacementError",
    "PlacementResolution",
    "canonical_edges",
    "graph_content_digest",
    "graph_structure_digest",
    "middle_rank_index",
    "ordered_nodes_by_degree",
    "placement_contract_digest",
    "placement_contract_payload",
    "placement_freeze_payload",
    "resolve_placement",
]


class PlacementError(RuntimeError):
    """Raised when a placement request cannot be resolved on a realized graph."""


#: The only accepted placement rank names, in canonical order.
PLACEMENT_RANKS: tuple[str, ...] = ("low", "mid", "high")

#: Pinned convention for even node counts: the lower of the two middle ranks.
MIDDLE_RANK_CONVENTION = "lower-middle"


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _require_placement(placement: object) -> str:
    if not isinstance(placement, str) or placement not in PLACEMENT_RANKS:
        raise PlacementError(
            f"placement must be one of {PLACEMENT_RANKS}, got {placement!r}"
        )
    return placement


def middle_rank_index(node_count: int) -> int:
    """Return the pinned middle rank for ``node_count`` nodes.

    Uses the lower middle for an even count (``(node_count - 1) // 2``), e.g.
    rank 4 for ten nodes, so the convention is deterministic and documented.
    """

    if isinstance(node_count, bool) or not isinstance(node_count, int):
        raise PlacementError(
            f"node_count must be a positive int, got {node_count!r}"
        )
    if node_count < 1:
        raise PlacementError(
            f"node_count must be a positive int, got {node_count!r}"
        )
    return (node_count - 1) // 2


def canonical_edges(graph: nx.Graph) -> tuple[tuple[int, int], ...]:
    """Return the unique undirected edges as sorted ``(min, max)`` integer pairs."""

    edges: set[tuple[int, int]] = set()
    for left, right in graph.edges():
        a, b = int(left), int(right)
        if a == b:
            continue
        edges.add((a, b) if a < b else (b, a))
    return tuple(sorted(edges))


def ordered_nodes_by_degree(graph: nx.Graph) -> tuple[int, ...]:
    """Return node ids ordered by ``(degree, node_id)`` (deterministic)."""

    nodes = sorted(int(node) for node in graph.nodes())
    return tuple(sorted(nodes, key=lambda node: (int(graph.degree(node)), node)))


def _structure_payload(graph: nx.Graph, kind: str) -> dict:
    return {
        "kind": kind,
        "node_count": int(graph.number_of_nodes()),
        "edges": [list(edge) for edge in canonical_edges(graph)],
    }


def graph_structure_digest(graph: nx.Graph, *, kind: str = "watts_strogatz") -> str:
    """SHA-256 over the graph kind, node count, and canonical edge list."""

    return hashlib.sha256(
        _canonical_bytes(_structure_payload(graph, kind))
    ).hexdigest()


def graph_content_digest(graph: nx.Graph, *, kind: str = "watts_strogatz") -> str:
    """SHA-256 over the graph kind, node count, per-node degrees, and edges."""

    nodes = sorted(int(node) for node in graph.nodes())
    payload = {
        "kind": kind,
        "node_count": len(nodes),
        "degrees": [int(graph.degree(node)) for node in nodes],
        "edges": [list(edge) for edge in canonical_edges(graph)],
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class PlacementResolution:
    """The realized placement of one rank on one seed-derived graph."""

    placement: str
    node: int
    degree: int
    rank: int
    node_count: int
    degrees: tuple[int, ...]
    degree_tie: bool
    regular: bool
    distinct_degree_groups: bool
    ordered_edges: tuple[tuple[int, int], ...]
    graph_digest: str
    content_digest: str

    def identity_payload(self) -> dict:
        """Environment-independent canonical placement identity."""

        return {
            "placement": self.placement,
            "node": int(self.node),
            "degree": int(self.degree),
            "rank": int(self.rank),
            "node_count": int(self.node_count),
            "degree_tie": bool(self.degree_tie),
            "regular": bool(self.regular),
            "distinct_degree_groups": bool(self.distinct_degree_groups),
            "graph_digest": self.graph_digest,
            "content_digest": self.content_digest,
        }

    @property
    def placement_digest(self) -> str:
        """SHA-256 over :meth:`identity_payload`."""

        return hashlib.sha256(
            _canonical_bytes(self.identity_payload())
        ).hexdigest()


def resolve_placement(
    config: ExperimentConfig,
    placement: str,
    *,
    require_varied: bool = False,
    kind: str | None = None,
) -> PlacementResolution:
    """Resolve one degree rank on the realized graph for ``config``.

    ``config`` supplies the frozen topology and the seed that derives the
    Watts-Strogatz graph. The graph is built exactly once with
    :func:`~dfl_sim.topology.load_topology`; it is never resampled or chosen by
    a metric. With ``require_varied=True`` a regular realized graph (every node
    has the same degree) fails closed, because low/mid/high could not then
    represent three distinct degree groups.
    """

    rank_name = _require_placement(placement)
    graph = load_topology(config)
    node_count = int(graph.number_of_nodes())
    if node_count < 1:
        raise PlacementError("realized graph has no nodes")
    graph_kind = kind or str(
        getattr(graph, "graph", {}).get("kind") or config.topology.kind
    )

    nodes = sorted(int(node) for node in graph.nodes())
    degrees = tuple(int(graph.degree(node)) for node in nodes)
    regular = len(set(degrees)) == 1
    if require_varied and regular:
        unique = degrees[0] if degrees else 0
        raise PlacementError(
            "realized graph is regular (all node degrees == "
            f"{unique}); low/mid/high cannot represent three distinct "
            "degree groups"
        )

    ordered = ordered_nodes_by_degree(graph)
    ordered_degrees = tuple(int(graph.degree(node)) for node in ordered)
    middle = middle_rank_index(node_count)
    rank_index = {"low": 0, "mid": middle, "high": node_count - 1}[rank_name]
    node = int(ordered[rank_index])
    degree = int(graph.degree(node))
    degree_tie = Counter(degrees)[degree] > 1
    canonical_group_degrees = {
        ordered_degrees[0],
        ordered_degrees[middle],
        ordered_degrees[node_count - 1],
    }
    distinct_degree_groups = (
        node_count >= 3 and len(canonical_group_degrees) == 3
    )

    return PlacementResolution(
        placement=rank_name,
        node=node,
        degree=degree,
        rank=int(rank_index),
        node_count=node_count,
        degrees=degrees,
        degree_tie=degree_tie,
        regular=regular,
        distinct_degree_groups=distinct_degree_groups,
        ordered_edges=canonical_edges(graph),
        graph_digest=graph_structure_digest(graph, kind=graph_kind),
        content_digest=graph_content_digest(graph, kind=graph_kind),
    )


def placement_contract_payload(
    placement: str, *, graph_kind: str = "watts_strogatz"
) -> dict:
    """Environment-independent payload for one placement contract."""

    _require_placement(placement)
    return {"placement": placement, "graph_kind": graph_kind}


def placement_contract_digest(
    placement: str, *, graph_kind: str = "watts_strogatz"
) -> str:
    """SHA-256 over :func:`placement_contract_payload`."""

    return hashlib.sha256(
        _canonical_bytes(
            placement_contract_payload(placement, graph_kind=graph_kind)
        )
    ).hexdigest()


def placement_freeze_payload(
    resolutions: Mapping[int, PlacementResolution],
) -> dict:
    """Canonical seed -> placement identity mapping bound into freeze digests."""

    seeds = sorted(int(seed) for seed in resolutions)
    return {
        "seeds": {
            str(seed): resolutions[seed].identity_payload() for seed in seeds
        }
    }
