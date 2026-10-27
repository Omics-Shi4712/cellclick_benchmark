"""CellTypeAgent adapter for the upstream two-stage annotation protocol.

The caller intentionally reproduces the upstream candidate generation and
expression reranking protocol. Its expression reference can overlap the
evaluated data; every live task records that the overlap risk is not excluded.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


STANDARD_FIELDS = (
    "source_row_id", "dataset", "tissue", "prediction", "method", "model", "task_id",
)
QUERY_FIELDS = ("source_row_id", "dataset", "tissue", "marker")
EXPRESSION_FIELDS = (
    "Tissue", "Cell Type", "Cell Count", "Tissue Composition", "Gene Symbol",
    "Expression", "Expression, Scaled", "Number of Cells Expressing Genes",
)
_STRATEGIES = frozenset({"mean", "max", "sum"})


@dataclass(frozen=True)
class ExpressionRerankRecord:
    """One normalized candidate record without evaluation fields."""

    dataset: str
    tissue: str
    markers: tuple[str, ...]
    candidates: tuple[tuple[str, ...], ...]
    expression_candidates: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ExpressionRerankResult:
    """Diagnostic output from one expression-based candidate reranking."""

    original_candidates: tuple[tuple[str, ...], ...]
    expression_ranks: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    base_ranks: tuple[float, ...]
    combined_scores: tuple[float, ...]
    selected_index: int
    prediction: str


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _candidate_groups(value: Any, top_n: int, field: str) -> tuple[tuple[str, ...], ...]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"CellTypeAgent {field} is not a list literal") from error
    if not isinstance(value, list) or len(value) != top_n:
        raise ValueError(f"CellTypeAgent {field} must contain exactly {top_n} candidates")
    groups: list[tuple[str, ...]] = []
    for candidate in value:
        values = candidate if isinstance(candidate, list) else [candidate]
        if not values or any(not isinstance(label, str) or not label.strip() for label in values):
            raise ValueError(f"CellTypeAgent {field} contains an empty candidate label")
        groups.append(tuple(label.strip() for label in values))
    return tuple(groups)


def _validate_record(record: ExpressionRerankRecord, top_n: int) -> None:
    _require_text(record.dataset, "dataset")
    _require_text(record.tissue, "tissue")
    if not record.markers or any(not isinstance(gene, str) or not gene.strip() for gene in record.markers):
        raise ValueError("markers must contain non-empty gene symbols")
    if len(record.candidates) != top_n or len(record.expression_candidates) != top_n:
        raise ValueError(f"record must contain exactly {top_n} candidates")
    for groups, field in ((record.candidates, "candidates"), (record.expression_candidates, "expression_candidates")):
        if any(not group or any(not isinstance(label, str) or not label.strip() for label in group) for group in groups):
            raise ValueError(f"{field} contains an empty candidate label")


def read_query(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or tuple(rows[0]) != QUERY_FIELDS:
        raise ValueError("CellTypeAgent query TSV must contain exactly source_row_id, dataset, tissue, and marker")
    if len({row["source_row_id"] for row in rows}) != len(rows):
        raise ValueError("CellTypeAgent query TSV has duplicate source_row_id values")
    for row in rows:
        for field in QUERY_FIELDS:
            _require_text(row.get(field), f"query {field}")
    return rows


def resolve_config_path(value: str, task: Mapping[str, Any]) -> Path:
    path = Path(_require_text(value, "path"))
    if path.is_absolute():
        return path
    return (Path(_require_text(task.get("config_dir"), "task config_dir")) / path).resolve()


def _load_upstream_functions(code_dir: Path) -> tuple[Callable[..., Any], Mapping[str, Sequence[str]], Callable[..., Any]]:
    if not (code_dir / "get_selection.py").is_file() or not (code_dir / "get_prediction.py").is_file():
        raise FileNotFoundError(f"celltypeagent_code_dir lacks upstream modules: {code_dir}")
    directory = str(code_dir)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    selection = importlib.import_module("get_selection")
    expression = importlib.import_module("get_expression_score")
    return selection.get_expression_rank, selection.expression_file_names, expression.load_expression_data


def _upstream_prediction(code_dir: Path) -> Callable[..., Any]:
    _load_upstream_functions(code_dir)
    return importlib.import_module("get_prediction").get_prediction


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
        loaded[dataset] = loader(names, str(expression_dir))
    return loaded


def _load_task_expression(expression_file: Path, loader: Callable[..., Any]) -> Any:
    if not expression_file.is_file():
        raise FileNotFoundError(f"CellTypeAgent expression_file not found: {expression_file}")
    frame = loader([expression_file.name], str(expression_file.parent))
    missing = set(EXPRESSION_FIELDS) - set(getattr(frame, "columns", ()))
    if missing:
        raise ValueError("CellTypeAgent expression_file lacks required columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("CellTypeAgent expression_file is empty")
    if frame["Tissue"].isna().any() or frame["Cell Type"].isna().any():
        raise ValueError("CellTypeAgent expression_file has empty Tissue or Cell Type values")
    frame["Expressed in Cells"] = (frame["Number of Cells Expressing Genes"] / frame["Cell Count"]).fillna(0)
    frame["Expression, Scaled"] = frame["Expression, Scaled"].fillna(0)
    return frame


def _register_task_tissues(dataset: str, tissues: set[str], expression: Any) -> None:
    selection = importlib.import_module("get_selection")
    available = {str(value).strip() for value in expression["Tissue"] if str(value).strip()}
    missing = sorted(tissue for tissue in tissues if tissue not in available)
    if missing:
        raise ValueError("CellTypeAgent expression_file lacks query tissue values: " + ", ".join(missing))
    selection.data_to_expression_tissue_map[dataset] = {tissue: tissue for tissue in tissues}


def rerank_with_expression(
    records: Sequence[ExpressionRerankRecord], *, celltypeagent_code_dir: Path,
    expression_dir: Path | None = None, expression_file: Path | None = None, top_n: int = 3,
    max_markers: int | None = None, tissue_flag: bool = True, mixture_strategy: str = "mean",
) -> list[ExpressionRerankResult]:
    """Apply upstream expression ranks and its non-oracle rank combiner."""
    if top_n != 3:
        raise ValueError("CellTypeAgent faithful reproduction requires top_n=3")
    if max_markers is not None and max_markers < 1:
        raise ValueError("max_markers must be positive when set")
    if mixture_strategy not in _STRATEGIES:
        raise ValueError(f"mixture_strategy must be one of: {', '.join(sorted(_STRATEGIES))}")
    if bool(expression_dir) == bool(expression_file):
        raise ValueError("provide exactly one of expression_dir or expression_file")
    if not records:
        return []
    for record in records:
        _validate_record(record, top_n)
    datasets = {record.dataset for record in records}
    get_expression_rank, file_names, loader = _load_upstream_functions(celltypeagent_code_dir)
    if expression_file is not None:
        reference = _load_task_expression(expression_file, loader)
        for dataset in datasets:
            _register_task_tissues(dataset, {record.tissue for record in records if record.dataset == dataset}, reference)
        expression_data = {dataset: reference for dataset in datasets}
    else:
        assert expression_dir is not None
        expression_data = _load_expression_data(datasets, expression_dir, file_names, loader)
        for frame in expression_data.values():
            frame["Expressed in Cells"] = (frame["Number of Cells Expressing Genes"] / frame["Cell Count"]).fillna(0)
            frame["Expression, Scaled"] = frame["Expression, Scaled"].fillna(0)
    base_ranks = tuple(float(value * 3 / top_n) for value in range(top_n - 1, -1, -1))
    modes = (
        (("Expression, Scaled", True, "averaged"), ("Expressed in Cells", True, "averaged"), ("Expression, Scaled", False, "averaged"))
        if tissue_flag else
        (("Expression, Scaled", False, "averaged"), ("Expressed in Cells", False, "averaged"), ("Expression, Scaled", False, "tissue_wise"))
    )
    results: list[ExpressionRerankResult] = []
    for record in records:
        sample = {
            "dataset": record.dataset, "tissue": record.tissue, "marker": ",".join(record.markers),
            "cell_type_pred_CLname": [list(group) for group in record.expression_candidates],
        }
        ranks = tuple(
            tuple(int(value) for value in get_expression_rank(
                record.dataset, sample, expression_data[record.dataset], term, use_tissue, aggregation,
                max_markers, mixture_strategy=mixture_strategy,
            )[0])
            for term, use_tissue, aggregation in modes
        )
        if any(len(rank) != top_n or any(value < 0 or value >= top_n for value in rank) for rank in ranks):
            raise ValueError("CellTypeAgent returned invalid expression ranks")
        combined = tuple(float(sum(rank[index] for rank in ranks) + base_ranks[index]) for index in range(top_n))
        selected_index = max(range(top_n), key=combined.__getitem__)
        if not isinstance(selected_index, Integral):
            raise ValueError("CellTypeAgent returned an invalid selected candidate index")
        results.append(ExpressionRerankResult(
            original_candidates=record.candidates, expression_ranks=ranks, base_ranks=base_ranks,
            combined_scores=combined, selected_index=int(selected_index),
            prediction=" + ".join(record.candidates[selected_index]).strip(),
        ))
    return results


def _synthetic_dataset_name(rows: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256("\n".join(row["source_row_id"] for row in rows).encode("utf-8")).hexdigest()[:16]
    return f"benchmark_{digest}"


def generate_candidates(
    rows: Sequence[dict[str, str]], *, code_dir: Path, work_dir: Path, model: str, species: str,
    top_n: int, max_markers: int | None,
) -> list[ExpressionRerankRecord]:
    if top_n != 3:
        raise ValueError("CellTypeAgent faithful reproduction requires top_n=3")
    if len({row["dataset"] for row in rows}) != 1:
        raise ValueError("CellTypeAgent tasks must contain exactly one dataset")
    dataset_name = _synthetic_dataset_name(rows)
    input_dir = work_dir / "upstream_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    with (input_dir / f"{dataset_name}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "tissue", "marker"))
        writer.writeheader()
        writer.writerows({field: row[field] for field in ("dataset", "tissue", "marker")} for row in rows)
    predicted, _ = _upstream_prediction(code_dir)(
        model, dataset_name, number_of_candidates=top_n, data_dir=str(input_dir),
        log_dir=str((work_dir / "upstream_logs").resolve()), max_markers=max_markers, species=species,
    )
    if len(predicted) != len(rows) or list(predicted.index) != list(range(len(rows))):
        raise ValueError("CellTypeAgent prediction rows are not aligned with the submitted query")
    required = {"cell_type_pred", "cell_type_pred_CLname"}
    if not required.issubset(predicted.columns):
        raise ValueError("CellTypeAgent prediction lacks candidate columns")
    return [ExpressionRerankRecord(
        dataset=row["dataset"], tissue=row["tissue"],
        markers=tuple(gene.strip() for gene in row["marker"].split(",") if gene.strip()),
        candidates=_candidate_groups(predicted.iloc[index]["cell_type_pred"], top_n, "cell_type_pred"),
        expression_candidates=_candidate_groups(predicted.iloc[index]["cell_type_pred_CLname"], top_n, "cell_type_pred_CLname"),
    ) for index, row in enumerate(rows)]


def write_standard_predictions(
    output_path: Path, rows: Sequence[Mapping[str, str]], results: Sequence[ExpressionRerankResult], *,
    model: str, task_id: str,
) -> None:
    """Write caller-normalized predictions for live and reproduction callers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STANDARD_FIELDS, delimiter="\t")
        writer.writeheader()
        for row, result in zip(rows, results):
            writer.writerow({"source_row_id": row["source_row_id"], "dataset": row["dataset"], "tissue": row["tissue"],
                             "prediction": result.prediction, "method": "celltypeagent", "model": model,
                             "task_id": task_id})


def _write_outputs(task: Mapping[str, Any], rows: Sequence[Mapping[str, str]], results: Sequence[ExpressionRerankResult], *, model: str) -> None:
    output_path = Path(_require_text(task.get("output_tsv"), "task output_tsv"))
    write_standard_predictions(output_path, rows, results, model=model, task_id=_require_text(task.get("task_id"), "task task_id"))
    work_dir = Path(_require_text(task.get("work_dir"), "task work_dir"))
    diagnostics_dir = work_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    with (diagnostics_dir / "reranking.jsonl").open("w", encoding="utf-8") as handle:
        for row, result in zip(rows, results):
            handle.write(json.dumps({"source_row_id": row["source_row_id"], "candidates": result.original_candidates,
                                     "expression_ranks": result.expression_ranks, "base_ranks": result.base_ranks,
                                     "combined_scores": result.combined_scores, "selected_index": result.selected_index,
                                     "prediction": result.prediction}, ensure_ascii=True) + "\n")
    config = task.get("method_config", {})
    (diagnostics_dir / "protocol.json").write_text(json.dumps({
        "protocol": "celltypeagent-upstream-two-stage", "reference_overlap_risk": "not excluded",
        "expression_file": _require_text(config.get("expression_file"), "expression_file"),
        "celltypeagent_code_dir": _require_text(config.get("celltypeagent_code_dir"), "celltypeagent_code_dir"),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_task(task: dict[str, Any]) -> None:
    config = dict(task.get("method_config", {}))
    forbidden = {"candidate_source", "candidate_file", "agreement_score", "final_score"} & set(config)
    if forbidden:
        raise ValueError("CellTypeAgent live caller rejects reproduction-only settings: " + ", ".join(sorted(forbidden)))
    rows = read_query(Path(_require_text(task.get("query_tsv"), "task query_tsv")))
    code_dir = resolve_config_path(_require_text(config.get("celltypeagent_code_dir"), "celltypeagent_code_dir"), task)
    expression_file = resolve_config_path(_require_text(config.get("expression_file"), "expression_file"), task)
    work_dir = Path(_require_text(task.get("work_dir"), "task work_dir"))
    work_dir.mkdir(parents=True, exist_ok=True)
    model, species = _require_text(config.get("model"), "model"), _require_text(config.get("species"), "species")
    top_n = int(config.get("top_n", 3))
    max_markers = config.get("max_markers")
    max_markers = None if max_markers is None else int(max_markers)
    records = generate_candidates(rows, code_dir=code_dir, work_dir=work_dir, model=model, species=species,
                                  top_n=top_n, max_markers=max_markers)
    results = rerank_with_expression(
        records, celltypeagent_code_dir=code_dir, expression_file=expression_file, top_n=top_n,
        max_markers=max_markers, tissue_flag=bool(config.get("tissue_flag", True)),
        mixture_strategy=str(config.get("mixture_strategy", "mean")),
    )
    _write_outputs(task, rows, results, model=model)
