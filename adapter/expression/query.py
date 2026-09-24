"""Runner-side view of label-free queries exported from expression data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from adapter.marker.marker import AnnotationQuery, QueryRow


QUERY_FIELDS = ("source_row_id", "dataset", "tissue", "marker")


class ExpressionQueryAdapter:
    """Read an expression export while retaining truth outside caller inputs."""

    def __init__(self, exports: list[dict[str, str]]) -> None:
        self.exports = exports
        self._truth_by_id: dict[str, str] = {}

    def generate_query(self, method_name: str) -> list[AnnotationQuery]:
        queries: list[AnnotationQuery] = []
        for export in self.exports:
            query_path = Path(export["query_tsv"])
            evaluation_path = Path(export["evaluation_json"])
            with query_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            if not rows or tuple(rows[0]) != QUERY_FIELDS:
                raise ValueError(f"Expression query TSV has unexpected columns: {query_path}")
            with evaluation_path.open(encoding="utf-8") as handle:
                evaluation = json.load(handle)
            truth = evaluation.get("ground_truth_by_source_row_id") if isinstance(evaluation, dict) else None
            if not isinstance(truth, dict) or set(truth) != {row["source_row_id"] for row in rows}:
                raise ValueError(f"Expression evaluation mapping does not align with {query_path}")
            query_rows = []
            for row in rows:
                if any(not row[field].strip() for field in QUERY_FIELDS):
                    raise ValueError(f"Expression query TSV has an empty required value: {query_path}")
                genes = tuple(gene for gene in row["marker"].split(",") if gene)
                if not genes:
                    raise ValueError(f"Expression query TSV has no marker genes: {query_path}")
                query_rows.append(QueryRow(row["source_row_id"], row["dataset"], row["tissue"], genes))
            task_id = export["task_id"]
            context = export["tissue_context"]
            queries.append(AnnotationQuery(method_name, task_id, context, tuple(query_rows)))
            self._truth_by_id.update({key: str(value) for key, value in truth.items()})
        if not queries:
            raise ValueError("Expression adapter produced no annotation queries")
        return queries

    def evaluation_rows(self, query: AnnotationQuery) -> list[dict[str, str]]:
        return [
            {"source_row_id": row.source_row_id, "ground_truth": self._truth_by_id[row.source_row_id]}
            for row in query.rows
        ]

    @staticmethod
    def write_query_tsv(query: AnnotationQuery, path: str | Path) -> None:
        from adapter.marker.marker import MarkerTableAdapter

        MarkerTableAdapter.write_query_tsv(query, path)
