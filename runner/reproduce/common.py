"""Shared helpers for method-specific reproduction checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def require_label_free_queries(queries: list[Any]) -> None:
    for query in queries:
        for row in query.rows:
            if not row.source_row_id or not row.dataset or not row.tissue or not row.marker:
                raise ValueError(f"Invalid label-free query row in {query.task_id}")


def resolve_reference_file(settings: dict[str, Any], config_path: Path, method: str) -> Path:
    value = settings.get("reference_file")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{method} test mode requires --reference-file")
    path = Path(value)
    path = path if path.is_absolute() else (config_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{method} reference_file not found: {path}")
    return path


def read_reference_rows(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows or not rows[0]:
        raise ValueError(f"Reference file is empty or has no header: {path}")
    return rows


def write_selected_rows(path: Path, queries: list[Any], extra_fields: dict[str, dict[str, str]] | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("source_row_id", "dataset", "tissue", "marker")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        count = 0
        for query in queries:
            for row in query.rows:
                writer.writerow({field: getattr(row, field) for field in fields})
                count += 1
    return count


def write_comparison_summary(
    output_dir: Path, method: str, output_file: Path, reference_file: Path, **comparison: Any
) -> Path:
    summary_file = output_dir / "comparisons" / f"{method}.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "output_file": str(output_file.resolve()),
        "reference_file": str(reference_file.resolve()),
        "summary_file": str(summary_file.resolve()),
        **comparison,
    }
    summary_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_file


def require_exact_source_row_ids(queries: list[Any], reference_rows: list[dict[str, str]], column: str, method: str) -> None:
    if column not in reference_rows[0]:
        raise ValueError(f"{method} reference_file lacks required column: {column}")
    expected = {row.source_row_id for query in queries for row in query.rows}
    actual = {row[column].strip() for row in reference_rows}
    if len(actual) != len(reference_rows):
        raise ValueError(f"{method} reference_file contains duplicate {column} values")
    if actual != expected:
        raise ValueError(f"{method} reference_file {column} values do not exactly match selected source rows")
