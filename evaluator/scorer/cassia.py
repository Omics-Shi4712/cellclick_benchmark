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
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from evaluator.solver.CLSolver import CLSolver


Score = Optional[float]
OLS_SEARCH_URL = "https://www.ebi.ac.uk/ols/api/search"
MAPPING_PATH = Path(__file__).resolve().parents[1] / "ref_data" / "CILO_mapping.json"


def _mapping_key(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _load_mapping() -> dict[str, str | None]:
    if not MAPPING_PATH.is_file():
        return {}
    try:
        payload = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_mapping(mapping: dict[str, str | None]) -> None:
    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MAPPING_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(MAPPING_PATH)


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


def label_to_cl_id(label: object) -> Optional[str]:
    """Resolve a natural-language label through the persistent offline cache."""
    if label is None:
        return None
    if not isinstance(label, str) or not label.strip():
        return None
    value = label.strip()
    if value.startswith("CL:"):
        return value
    key = _mapping_key(value)
    mapping = _load_mapping()
    if key in mapping:
        result = mapping[key]
        return result if isinstance(result, str) and result else None
    cl_id, _ = get_cell_type_info(value)
    mapping[key] = cl_id
    _save_mapping(mapping)
    return cl_id


def _resolve_id(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return ""
    return label_to_cl_id(value)


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
        first = self._cl_id(_resolve_id(first_id))
        second = self._cl_id(_resolve_id(second_id))
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

    def score_by_dataset(
        self, first_ids: Sequence[object], second_ids: Sequence[object], datasets: Sequence[str]
    ) -> dict[str, Optional[float]]:
        if len(first_ids) != len(second_ids) or len(first_ids) != len(datasets):
            raise ValueError("first_ids, second_ids, and datasets must have the same length")
        grouped: dict[str, list[float]] = {}
        for first, second, dataset in zip(first_ids, second_ids, datasets):
            score = self.score_one(first, second)
            if score is None:
                return {dataset: None for dataset in sorted(set(datasets))}
            grouped.setdefault(dataset, []).append(score)
        return {dataset: sum(values) / len(values) for dataset, values in grouped.items()}


@lru_cache(maxsize=1)
def default_scorer() -> CassiaScorer:
    """Return the cached scorer backed by the pinned Cell Ontology."""
    return CassiaScorer()


def evaluate_one(first_id: object, second_id: object) -> Score:
    """Score one pair, resolving natural-language labels to cached CL IDs."""
    return default_scorer().score_one(first_id, second_id)


def evaluate(first_ids: Sequence[object], second_ids: Sequence[object]) -> list[Score]:
    """Score aligned pairs, resolving labels through the persistent cache."""
    return [evaluate_one(a, b) for a, b in zip(first_ids, second_ids)]
