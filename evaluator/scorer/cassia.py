"""CASSIA's Cell Ontology based annotation agreement score.

The scorer accepts Cell Ontology identifiers (for example ``CL:0000236``),
not free-text annotation labels.  CASSIA assigns full credit to an identical
CL term and half credit only where the two terms have a direct parent--child
``is_a`` relation in the pinned Cell Ontology.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import json
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from evaluator.solver.CLSolver import CLSolver


Score = Optional[float]
OLS_SEARCH_URL = "https://www.ebi.ac.uk/ols/api/search"


@lru_cache(maxsize=4096)
def get_cell_type_info(
    cell_type_name: str, ontology: str = "CL", *, timeout: float = 10.0
) -> tuple[Optional[str], Optional[str]]:
    """Look up the top EBI OLS hit for a cell-type label.

    The lookup is intentionally separate from ontology scoring: callers can
    persist the returned CL ID and label alongside a frozen prediction before
    using :class:`CassiaScorer`.  ``(None, None)`` means OLS returned no usable
    hit; network and HTTP failures are allowed to raise rather than silently
    becoming a score of zero.
    """
    if not isinstance(cell_type_name, str) or not cell_type_name.strip():
        raise ValueError("cell_type_name must be a non-empty string")
    if not isinstance(ontology, str) or not ontology.strip():
        raise ValueError("ontology must be a non-empty string")

    query = urlencode({"q": cell_type_name, "ontology": ontology, "rows": 1})
    request = Request(
        f"{OLS_SEARCH_URL}?{query}",
        headers={"Accept": "application/json", "User-Agent": "AI-CTA-benchmark"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)

    response_block = payload.get("response", {}) if isinstance(payload, dict) else {}
    documents = response_block.get("docs", []) if isinstance(response_block, dict) else []
    if not documents or not isinstance(documents[0], dict):
        return None, None
    first_doc = documents[0]
    obo_id = first_doc.get("obo_id")
    label = first_doc.get("label")
    return (
        obo_id if isinstance(obo_id, str) and obo_id else None,
        label if isinstance(label, str) and label else None,
    )


class CassiaScorer:
    """Score CL-ID pairs according to CASSIA's published agreement rule.

    ``None`` represents a missing, hence unscorable, annotation.  Empty,
    malformed, and unknown submitted identifiers are scored as ``0.0``:
    none can establish an ontology identity or direct parent--child relation.
    """

    def __init__(self, solver: CLSolver | None = None) -> None:
        if solver is None:
            from evaluator.solver.CLSolver import CLSolver

            solver = CLSolver()
        self.solver = solver

    @staticmethod
    def _cl_id(value: object) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            return ""
        return value.strip()

    def score_one(self, first_id: object, second_id: object) -> Score:
        """Return ``1.0``, ``0.5``, or ``0.0`` for one CL-ID pair."""
        first = self._cl_id(first_id)
        second = self._cl_id(second_id)
        if first is None or second is None:
            return None
        if not first or not second:
            return 0.0

        graph = self.solver.cl_graph
        if first not in graph or second not in graph:
            return 0.0
        if first == second:
            return 1.0
        if graph.has_edge(first, second) or graph.has_edge(second, first):
            return 0.5
        return 0.0

    def score(self, first_ids: Sequence[object], second_ids: Sequence[object]) -> list[Score]:
        """Score aligned CL-ID pairs without applying dataset aggregation."""
        if len(first_ids) != len(second_ids):
            raise ValueError(
                "first_ids and second_ids must have the same length; got "
                f"{len(first_ids)} and {len(second_ids)}."
            )
        return [
            self.score_one(first_id, second_id)
            for first_id, second_id in zip(first_ids, second_ids)
        ]


@lru_cache(maxsize=1)
def default_scorer() -> CassiaScorer:
    """Return the cached scorer backed by the pinned Cell Ontology."""
    return CassiaScorer()


def evaluate_one(first_id: object, second_id: object) -> Score:
    """Score one CASSIA CL-ID pair using the pinned Cell Ontology."""
    return default_scorer().score_one(first_id, second_id)


def evaluate(first_ids: Sequence[object], second_ids: Sequence[object]) -> list[Score]:
    """Score aligned CASSIA CL-ID pairs using the pinned Cell Ontology."""
    return default_scorer().score(first_ids, second_ids)
