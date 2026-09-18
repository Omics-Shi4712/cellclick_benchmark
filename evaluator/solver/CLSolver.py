"""Utilities for querying the Cell Ontology ``is_a`` hierarchy.

The graph used here is deliberately directed from a child term to its parent
term.  This makes an upward ontology path a normal directed shortest path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import networkx as nx


class CLSolver:
    """Load a pinned Cell Ontology OBO file and query its ``is_a`` DAG.

    Parameters
    ----------
    obo_path:
        Path to a Cell Ontology OBO file.  By default, the repository's pinned
        file, ``evaluator/ref_data/cl.obo``, is used.

    Attributes
    ----------
    cl_graph_raw:
        The graph returned by :func:`obonet.read_obo`.
    cl_graph:
        A :class:`networkx.DiGraph` containing *only* ``is_a`` edges, directed
        ``child -> parent``.
    """

    DEFAULT_OBO_PATH = Path(__file__).resolve().parents[1] / "ref_data" / "cl.obo"

    def __init__(self, obo_path: str | Path | None = None) -> None:
        self.obo_path = Path(obo_path) if obo_path is not None else self.DEFAULT_OBO_PATH
        if not self.obo_path.is_file():
            raise FileNotFoundError(f"Cell Ontology OBO file not found: {self.obo_path}")

        try:
            import obonet
        except ImportError as exc:
            raise ImportError(
                "CLSolver requires 'obonet'. Install it with: pip install obonet networkx"
            ) from exc

        self.cl_graph_raw = obonet.read_obo(str(self.obo_path))
        self.cl_graph = self._build_is_a_graph(self.cl_graph_raw)

    @staticmethod
    def _build_is_a_graph(raw_graph: nx.MultiDiGraph) -> nx.DiGraph:
        """Return the child-to-parent DAG containing only ``is_a`` relations."""
        graph = nx.DiGraph()
        # Preserve terms that have no is_a relation as known ontology nodes.
        graph.add_nodes_from(raw_graph.nodes(data=True))

        for child, parent, key, data in raw_graph.edges(keys=True, data=True):
            relation_values = (
                str(key),
                str(data.get("relation", "")),
                str(data.get("typedef", "")),
            )
            if "is_a" in relation_values:
                graph.add_edge(child, parent)

        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("The Cell Ontology is_a graph must be a DAG.")
        return graph

    def get_ancestors(self, cl_id: str) -> set[str]:
        """Return ``cl_id`` and all of its ancestors.

        Including the input term makes an exact term match a valid common
        ancestor with distance zero.
        """
        if cl_id not in self.cl_graph:
            return set()
        return {cl_id, *nx.descendants(self.cl_graph, cl_id)}

    def get_path_to_ancestor(
        self, child_id: str, ancestor_id: str
    ) -> Optional[list[dict[str, str]]]:
        """Return one shortest child-to-ancestor path, with term labels."""
        try:
            path_ids = nx.shortest_path(self.cl_graph, child_id, ancestor_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        return [
            {
                "cl_id": term_id,
                "label": self.cl_graph_raw.nodes[term_id].get("name", term_id),
            }
            for term_id in path_ids
        ]

    def get_common_parents(
        self, first_id: str, second_id: str
    ) -> Optional[dict[str, Any]]:
        """Return the nearest common ancestor and distances from both terms.

        A term is included in its own ancestor set, so equal IDs return that
        term with both distances equal to zero.  The nearest common ancestor is
        chosen by minimum average upward distance; ties are resolved by total
        distance and then CL identifier for deterministic results.  Distances
        are numbers of ``is_a`` edges.  Unknown or disconnected IDs return
        ``None``.
        """
        if first_id not in self.cl_graph or second_id not in self.cl_graph:
            return None

        first_ancestors = self.get_ancestors(first_id)
        second_ancestors = self.get_ancestors(second_id)
        common_parents = first_ancestors & second_ancestors
        results: list[dict[str, Any]] = []

        for parent_id in common_parents:
            first_distance = nx.shortest_path_length(self.cl_graph, first_id, parent_id)
            second_distance = nx.shortest_path_length(self.cl_graph, second_id, parent_id)
            results.append(
                {
                    "cl_id": parent_id,
                    "label": self.cl_graph_raw.nodes[parent_id].get("name", parent_id),
                    "distance_first": first_distance,
                    "distance_second": second_distance,
                    "average_distance": (first_distance + second_distance) / 2,
                }
            )

        if not results:
            return None
        return min(
            results,
            key=lambda item: (
                item["average_distance"],
                item["distance_first"] + item["distance_second"],
                item["cl_id"],
            ),
        )

    # Common terminology for ontology ancestry.
    get_common_ancestors = get_common_parents
    common_parents = get_common_parents
