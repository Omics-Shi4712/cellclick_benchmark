"""Reproduce CellTypeAgent reranking from paper-frozen candidates.

This module is the sole place that understands paper candidate/final-result
CSV columns. The live caller only receives normalized candidate records.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from caller.celltypeagent.celltypeagent import (
    ExpressionRerankRecord,
    _candidate_groups,
    rerank_with_expression,
    write_standard_predictions,
)
from runner.reproduce.common import read_reference_rows, require_label_free_queries, resolve_reference_file, write_comparison_summary


_CANDIDATE_COLUMNS = {"dataset", "tissue", "marker", "cell_type_pred", "cell_type_pred_CLname"}
_FINAL_COLUMNS = {"dataset", "tissue", "marker", "best_candidate_index", "final_annotation", "final_score"}


def _markers(value: str) -> tuple[str, ...]:
    result = tuple(gene.strip() for gene in value.split(",") if gene.strip())
    if not result:
        raise ValueError("CellTypeAgent marker list is empty")
    return result


def _fingerprint(dataset: str, tissue: str, markers: tuple[str, ...]) -> str:
    payload = "\0".join((dataset.strip(), tissue.strip(), *[gene.upper() for gene in markers]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _paper_candidates(path: Path, top_n: int) -> dict[str, ExpressionRerankRecord]:
    rows = read_reference_rows(path)
    missing = _CANDIDATE_COLUMNS - set(rows[0])
    if missing:
        raise ValueError("CellTypeAgent candidate_file lacks columns: " + ", ".join(sorted(missing)))
    result: dict[str, ExpressionRerankRecord] = {}
    for row in rows:
        markers = _markers(row["marker"])
        key = _fingerprint(row["dataset"], row["tissue"], markers)
        if key in result:
            raise ValueError("CellTypeAgent candidate_file has duplicate dataset/tissue/marker rows")
        result[key] = ExpressionRerankRecord(
            dataset=row["dataset"].strip(), tissue=row["tissue"].strip(), markers=markers,
            candidates=_candidate_groups(row["cell_type_pred"], top_n, "cell_type_pred"),
            expression_candidates=_candidate_groups(row["cell_type_pred_CLname"], top_n, "cell_type_pred_CLname"),
        )
    return result


def _resolve(value: Any, config_path: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"CellTypeAgent test requires {name}")
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _write_comparison(path: Path, rows: list[dict[str, str]], results: list[Any], reference_rows: list[dict[str, str]]) -> None:
    by_key = {
        _fingerprint(row["dataset"], row["tissue"], _markers(row["marker"])): row
        for row in reference_rows
    }
    fields = (
        "source_row_id", "dataset", "tissue", "marker", "selected_index", "prediction",
        "published_candidate_index", "published_final_annotation", "published_final_score",
        "candidate_index_match", "prediction_match",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, result in zip(rows, results):
            published = by_key[_fingerprint(row["dataset"], row["tissue"], _markers(row["marker"]))]
            published_index = int(published["best_candidate_index"])
            writer.writerow({
                "source_row_id": row["source_row_id"], "dataset": row["dataset"], "tissue": row["tissue"],
                "marker": row["marker"], "selected_index": result.selected_index, "prediction": result.prediction,
                "published_candidate_index": published_index, "published_final_annotation": published["final_annotation"],
                "published_final_score": published["final_score"],
                "candidate_index_match": result.selected_index == published_index,
                "prediction_match": result.prediction.casefold() == published["final_annotation"].strip().casefold(),
            })


def run(config, adapter, queries, callers, config_path, output_dir, *, resume=False, force=False, max_parallel_tasks=1):
    del config, adapter, resume, force
    require_label_free_queries(queries)
    for name, settings in callers.items():
        if settings.get("implementation", name) != "celltypeagent":
            continue
        if settings.get("candidate_source") not in {"frozen", "frozen_reproduction"}:
            raise ValueError("CellTypeAgent test mode requires candidate_source=frozen_reproduction")
        top_n = int(settings.get("top_n", 3))
        candidates = _paper_candidates(_resolve(settings.get("candidate_file"), config_path, "candidate_file"), top_n)
        reference_file = resolve_reference_file(settings, config_path, "celltypeagent")
        reference_rows = read_reference_rows(reference_file)
        missing = _FINAL_COLUMNS - set(reference_rows[0])
        if missing:
            raise ValueError("CellTypeAgent reference_file lacks columns: " + ", ".join(sorted(missing)))
        reference_keys = {_fingerprint(row["dataset"], row["tissue"], _markers(row["marker"])) for row in reference_rows}
        if len(reference_keys) != len(reference_rows):
            raise ValueError("CellTypeAgent reference_file has duplicate dataset/tissue/marker rows")
        expression_dir = _resolve(settings.get("expression_dir"), config_path, "expression_dir")
        code_dir = _resolve(settings.get("celltypeagent_code_dir"), config_path, "celltypeagent_code_dir")
        all_rows: list[dict[str, str]] = []
        all_results: list[Any] = []
        for query in queries:
            rows = [{"source_row_id": row.source_row_id, "dataset": row.dataset, "tissue": row.tissue, "marker": row.marker} for row in query.rows]
            records = []
            for row in rows:
                key = _fingerprint(row["dataset"], row["tissue"], _markers(row["marker"]))
                if key not in candidates or key not in reference_keys:
                    raise ValueError(f"CellTypeAgent paper rows do not match query {row['source_row_id']}")
                records.append(candidates[key])
            results = rerank_with_expression(
                records, celltypeagent_code_dir=code_dir, expression_dir=expression_dir, top_n=top_n,
                max_markers=settings.get("max_markers"), tissue_flag=bool(settings.get("tissue_flag", True)),
                mixture_strategy=str(settings.get("mixture_strategy", "mean")),
            )
            task_path = output_dir / "predictions" / name / f"{query.task_id.replace('/', '_')}.tsv"
            write_standard_predictions(task_path, rows, results, model=str(settings.get("model", "o1-preview-frozen")), task_id=query.task_id)
            all_rows.extend(rows)
            all_results.extend(results)
        comparison_file = output_dir / "comparisons" / "celltypeagent.csv"
        _write_comparison(comparison_file, all_rows, all_results, reference_rows)
        summary_file = write_comparison_summary(
            output_dir, "celltypeagent", comparison_file, reference_file, rows=len(all_rows),
            comparison="paper frozen candidates reranked before published-result comparison",
            reference_overlap_risk="not excluded",
        )
        return {"method": "celltypeagent", "rows": len(all_rows), "status": "validated",
                "output_file": str(comparison_file), "reference_file": str(reference_file), "summary_file": str(summary_file)}
    raise ValueError("No CellTypeAgent caller selected")
