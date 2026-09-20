"""CellTypeAgent caller adapter for live annotation strategies.

The expression-reranking helpers in this module deliberately accept only
already-normalized candidate records.  Parsing paper artifacts and comparing
against their published labels belongs in ``test/``.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


_STRATEGIES = frozenset({"mean", "max", "sum"})


@dataclass(frozen=True)
class ExpressionRerankRecord:
    """One label-free, normalized frozen-candidate record.

    ``candidates`` and ``expression_candidates`` are parallel ordered
    candidate groups. A singleton group represents one ordinary cell type;
    a multi-element group represents an upstream mixture candidate.
    """

    dataset: str
    tissue: str
    markers: tuple[str, ...]
    candidates: tuple[tuple[str, ...], ...]
    expression_candidates: tuple[tuple[str, ...], ...]
    agreement_scores: tuple[float, ...]


@dataclass(frozen=True)
class ExpressionRerankResult:
    """Diagnostic output from one expression-based candidate reranking."""

    original_candidates: tuple[tuple[str, ...], ...]
    expression_ranks: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    base_ranks: tuple[float, ...]
    combined_scores: tuple[float, ...]
    selected_index: int
    prediction: str
    selected_agreement_score: float


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_record(record: ExpressionRerankRecord, top_n: int) -> None:
    _require_text(record.dataset, "dataset")
    _require_text(record.tissue, "tissue")
    if not record.markers or any(not isinstance(gene, str) or not gene.strip() for gene in record.markers):
        raise ValueError("markers must contain non-empty gene symbols")
    if len(record.candidates) != top_n or len(record.expression_candidates) != top_n:
        raise ValueError(f"record must contain exactly {top_n} candidates")
    if len(record.agreement_scores) != top_n:
        raise ValueError(f"record must contain exactly {top_n} agreement scores")
    for groups, field in ((record.candidates, "candidates"), (record.expression_candidates, "expression_candidates")):
        if any(not group or any(not isinstance(label, str) or not label.strip() for label in group) for group in groups):
            raise ValueError(f"{field} contains an empty candidate label")


def _load_upstream_functions(code_dir: Path) -> tuple[Callable[..., Any], Mapping[str, Sequence[str]], Callable[..., Any]]:
    """Load the installed/upstream implementation without modifying it."""
    if not (code_dir / "get_selection.py").is_file():
        raise FileNotFoundError(f"celltypeagent_code_dir lacks get_selection.py: {code_dir}")
    directory = str(code_dir)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    selection = importlib.import_module("get_selection")
    expression = importlib.import_module("get_expression_score")
    return selection.get_expression_rank, selection.expression_file_names, expression.load_expression_data


def _load_expression_data(
    datasets: set[str], expression_dir: Path, file_names: Mapping[str, Sequence[str]], loader: Callable[..., Any],
) -> dict[str, Any]:
    if not expression_dir.is_dir():
        raise FileNotFoundError(f"Expression directory not found: {expression_dir}")
    loaded: dict[str, Any] = {}
    for dataset in sorted(datasets):
        names = file_names.get(dataset)
        if names is None:
            raise ValueError(f"CellTypeAgent has no expression file list for {dataset}")
        missing = [name for name in names if not (expression_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Expression files missing for {dataset}: {', '.join(missing)}")
        frame = loader(names, str(expression_dir))
        frame["Expressed in Cells"] = (frame["Number of Cells Expressing Genes"] / frame["Cell Count"]).fillna(0)
        frame["Expression, Scaled"] = frame["Expression, Scaled"].fillna(0)
        loaded[dataset] = frame
    return loaded


def rerank_with_expression(
    records: Sequence[ExpressionRerankRecord],
    *,
    celltypeagent_code_dir: Path,
    expression_dir: Path,
    top_n: int = 3,
    max_markers: int | None = None,
    tissue_flag: bool = True,
    mixture_strategy: str = "mean",
) -> list[ExpressionRerankResult]:
    """Rerank prepared frozen candidates with CellTypeAgent expression evidence.

    The caller never reads a paper CSV or accepts reference labels. The test
    fixture is responsible for creating ``ExpressionRerankRecord`` objects.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    if max_markers is not None and max_markers < 1:
        raise ValueError("max_markers must be positive when set")
    if mixture_strategy not in _STRATEGIES:
        raise ValueError(f"mixture_strategy must be one of: {', '.join(sorted(_STRATEGIES))}")
    if not records:
        return []
    for record in records:
        _validate_record(record, top_n)

    get_expression_rank, expression_file_names, load_expression_data = _load_upstream_functions(celltypeagent_code_dir)
    expression_data = _load_expression_data(
        {record.dataset for record in records}, expression_dir, expression_file_names, load_expression_data,
    )
    # This is the upstream ``calculate_agreement_scores`` positional prior.
    base_ranks = tuple(float(value * 3 / top_n) for value in range(top_n - 1, -1, -1))
    results: list[ExpressionRerankResult] = []
    for record in records:
        upstream_sample = {
            "dataset": record.dataset,
            "tissue": record.tissue,
            "marker": ",".join(record.markers),
            "cell_type_pred_CLname": [list(group) for group in record.expression_candidates],
        }
        ranks = tuple(
            tuple(int(value) for value in get_expression_rank(
                record.dataset, upstream_sample, expression_data[record.dataset], term, use_tissue,
                "averaged", max_markers, mixture_strategy=mixture_strategy,
            )[0])
            for term, use_tissue in (
                ("Expression, Scaled", tissue_flag),
                ("Expressed in Cells", tissue_flag),
                ("Expression, Scaled", False),
            )
        )
        if any(len(rank) != top_n or any(value < 0 or value >= top_n for value in rank) for rank in ranks):
            raise ValueError("CellTypeAgent returned invalid expression ranks")
        combined = tuple(float(sum(rank[index] for rank in ranks) + base_ranks[index]) for index in range(top_n))
        selected_index = max(range(top_n), key=combined.__getitem__)
        if not isinstance(selected_index, Integral):  # Defensive: preserve an explicit output contract.
            raise ValueError("CellTypeAgent returned an invalid selected candidate index")
        prediction = " + ".join(record.candidates[selected_index]).strip()
        results.append(ExpressionRerankResult(
            original_candidates=record.candidates,
            expression_ranks=ranks,
            base_ranks=base_ranks,
            combined_scores=combined,
            selected_index=int(selected_index),
            prediction=prediction,
            selected_agreement_score=record.agreement_scores[selected_index],
        ))
    return results


def run_task(task: dict[str, Any]) -> None:
    source = task.get("method_config", {}).get("candidate_source")
    if source in {"frozen", "frozen_reproduction"}:
        raise ValueError(
            "candidate_source=frozen_reproduction is a paper-reproduction input; "
            "run it with runner --mode test, not runner --mode run"
        )
    raise NotImplementedError("No live CellTypeAgent candidate caller is implemented yet")
