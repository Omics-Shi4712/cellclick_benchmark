"""Leakage-safe marker-table adapter for cell-type annotation benchmarks.

The adapter deliberately keeps reference labels out of generated queries.  They
remain available only through ``evaluation_rows`` after predictions are made.
It uses only the Python standard library so it can be shared by every method.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


QUERY_COLUMNS = ("dataset", "tissue", "marker")
EVALUATION_COLUMNS = ("ground_truth", "clid", "synonyms", "broadtype")


@dataclass(frozen=True)
class QueryRow:
    """One cluster presented to an annotation method."""

    source_row_id: str
    dataset: str
    tissue: str
    marker_genes: tuple[str, ...]

    @property
    def marker(self) -> str:
        return ",".join(self.marker_genes)


@dataclass(frozen=True)
class AnnotationQuery:
    """A leakage-free, method-ready annotation task."""

    method_name: str
    task_id: str
    tissue_context: str
    rows: tuple[QueryRow, ...]


class MarkerTableAdapter:
    """Read a configured marker table and generate annotation queries.

    The JSON configuration maps semantic field names to source column names.
    A query contains only identifiers, cohort metadata, and ordered marker
    genes.  Reference labels can be obtained separately with
    :meth:`evaluation_rows` after model output has been frozen.
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        with self.config_path.open(encoding="utf-8") as handle:
            self.config: dict[str, Any] = json.load(handle)

        self.columns: dict[str, str] = self.config["columns"]
        missing = [name for name in QUERY_COLUMNS if not self.columns.get(name)]
        if missing:
            raise ValueError(f"Missing required column mappings: {', '.join(missing)}")

        source = Path(self.config["source"])
        self.source_path = source if source.is_absolute() else self.config_path.parent / source
        self.delimiter = self.config.get("delimiter", "\t")
        self.marker_split = self.config.get("marker_split", r"\s*,\s*")
        self.top_n = self.config.get("top_n", 10)
        self.required_marker_count = self.config.get("required_marker_count")
        self._source_rows: list[dict[str, str]] | None = None

    def _load_source_rows(self) -> list[dict[str, str]]:
        if self._source_rows is not None:
            return self._source_rows
        with self.source_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=self.delimiter)
            source_headers = set(reader.fieldnames or [])
            expected = set(self.columns.values())
            missing = sorted(expected - source_headers)
            if missing:
                raise ValueError(f"Configured columns not in {self.source_path}: {missing}")
            self._source_rows = list(reader)
        return self._source_rows

    def _split_marker(self, value: str, row_id: str) -> tuple[str, ...]:
        genes = tuple(gene.strip() for gene in re.split(self.marker_split, value.strip()) if gene.strip())
        if self.top_n is not None:
            genes = genes[: int(self.top_n)]
        if not genes:
            raise ValueError(f"{row_id} has no marker genes")
        if self.required_marker_count is not None and len(genes) != int(self.required_marker_count):
            raise ValueError(
                f"{row_id} has {len(genes)} marker genes; expected {self.required_marker_count}"
            )
        return genes

    def _task_key(self, dataset: str, tissue: str) -> str:
        aggregate = set(self.config.get("grouping", {}).get("aggregate_all_tissues", []))
        return dataset if dataset in aggregate else f"{dataset}__{tissue}"

    def marker_count_audit(self) -> list[dict[str, int | str]]:
        """Report marker counts before and after ``top_n`` selection.

        A short marker list is valid input and must not be padded.  Callers can
        use this report to decide whether short rows should be excluded by a
        documented benchmark rule.
        """
        audit = []
        for line_number, row in enumerate(self._load_source_rows(), start=2):
            raw_genes = tuple(
                gene.strip()
                for gene in re.split(self.marker_split, row[self.columns["marker"]].strip())
                if gene.strip()
            )
            used_genes = self._split_marker(
                row[self.columns["marker"]], f"row_{line_number - 1:04d}"
            )
            audit.append(
                {
                    "source_row_id": f"row_{line_number - 1:04d}",
                    "raw_marker_count": len(raw_genes),
                    "used_marker_count": len(used_genes),
                }
            )
        return audit

    def generate_query(
        self,
        method_name: str,
        *,
        dataset: str | None = None,
        tissue: str | None = None,
    ) -> list[AnnotationQuery]:
        """Return grouped queries without ground-truth or ontology fields."""
        if not method_name.strip():
            raise ValueError("method_name must not be empty")

        grouped: dict[str, list[QueryRow]] = {}
        for line_number, row in enumerate(self._load_source_rows(), start=2):
            row_dataset = row[self.columns["dataset"]].strip()
            row_tissue = row[self.columns["tissue"]].strip()
            if dataset is not None and row_dataset != dataset:
                continue
            if tissue is not None and row_tissue != tissue:
                continue
            row_id = f"row_{line_number - 1:04d}"
            query_row = QueryRow(
                source_row_id=row_id,
                dataset=row_dataset,
                tissue=row_tissue,
                marker_genes=self._split_marker(row[self.columns["marker"]], row_id),
            )
            grouped.setdefault(self._task_key(row_dataset, row_tissue), []).append(query_row)

        if not grouped:
            raise ValueError("No rows match the requested dataset/tissue selection")
        organism = self.config.get("organism", "")
        aggregate_context = self.config.get("grouping", {}).get("aggregate_tissue_context", "{organism} cells")
        queries = []
        for task_id, rows in grouped.items():
            is_aggregate = task_id in set(self.config.get("grouping", {}).get("aggregate_all_tissues", []))
            template = aggregate_context if is_aggregate else self.config.get(
                "tissue_context", "{organism} {tissue}"
            )
            context = template.format(organism=organism, tissue=rows[0].tissue).strip()
            queries.append(AnnotationQuery(method_name, task_id, context, tuple(rows)))
        return queries

    def evaluation_rows(self, query: AnnotationQuery) -> list[dict[str, str]]:
        """Return truth fields for a completed query, aligned by source_row_id."""
        wanted = {row.source_row_id for row in query.rows}
        records: list[dict[str, str]] = []
        for line_number, row in enumerate(self._load_source_rows(), start=2):
            row_id = f"row_{line_number - 1:04d}"
            if row_id not in wanted:
                continue
            record = {"source_row_id": row_id}
            for semantic_name in EVALUATION_COLUMNS:
                source_name = self.columns.get(semantic_name)
                if source_name:
                    record[semantic_name] = row[source_name]
                else:
                    record[semantic_name] = ""
            records.append(record)
        return records

    def result_fieldnames(self) -> tuple[str, ...]:
        """Return final-result evaluation columns using configured source names.

        Each evaluation field is always represented. A missing JSON mapping uses
        its semantic name and is populated with empty strings when results are
        merged.
        """
        return tuple(self.columns.get(name) or name for name in EVALUATION_COLUMNS)

    def merge_predictions(
        self,
        query: AnnotationQuery,
        prediction_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Join frozen predictions to configured evaluation fields by row ID.

        Prediction files remain label-free while GPTCelltype is running. This
        method performs the only permitted truth join, validates a one-to-one
        match, and writes one complete benchmark result TSV. If ``output_path``
        is omitted, the frozen prediction file is replaced in place.
        """
        prediction_path = Path(prediction_path)
        destination = Path(output_path) if output_path is not None else prediction_path
        with prediction_path.open(encoding="utf-8-sig", newline="") as handle:
            predictions = list(csv.DictReader(handle, delimiter="\t"))
        required_prediction_columns = {"source_row_id", "gpt_annotation_raw", "model", "tissue_context"}
        if not predictions or not required_prediction_columns.issubset(predictions[0]):
            raise ValueError("Prediction TSV lacks required GPTCelltype output columns")

        prediction_by_id = {record["source_row_id"]: record for record in predictions}
        query_ids = [row.source_row_id for row in query.rows]
        if len(prediction_by_id) != len(predictions):
            raise ValueError("Prediction TSV has duplicate source_row_id values")
        if set(prediction_by_id) != set(query_ids):
            raise ValueError("Prediction TSV source_row_id values do not exactly match the query")

        truth_by_id = {record["source_row_id"]: record for record in self.evaluation_rows(query)}
        if set(truth_by_id) != set(query_ids):
            raise ValueError("Configured marker table does not align with the query")

        evaluation_fields = self.result_fieldnames()
        fieldnames = (
            "source_row_id", "dataset", "tissue", "marker", "gpt_annotation_raw",
            "model", "tissue_context", *evaluation_fields,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for query_row in query.rows:
                prediction = prediction_by_id[query_row.source_row_id]
                truth = truth_by_id[query_row.source_row_id]
                record = {
                    "source_row_id": query_row.source_row_id,
                    "dataset": query_row.dataset,
                    "tissue": query_row.tissue,
                    "marker": query_row.marker,
                    "gpt_annotation_raw": prediction["gpt_annotation_raw"],
                    "model": prediction["model"],
                    "tissue_context": prediction["tissue_context"],
                }
                for semantic_name, output_name in zip(EVALUATION_COLUMNS, evaluation_fields):
                    record[output_name] = truth[semantic_name]
                writer.writerow(record)
        return destination

    @staticmethod
    def write_query_tsv(query: AnnotationQuery, path: str | Path) -> None:
        """Write the R/Python exchange contract, containing no reference labels."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("source_row_id", "dataset", "tissue", "marker"),
                delimiter="\t",
            )
            writer.writeheader()
            for row in query.rows:
                writer.writerow(
                    {
                        "source_row_id": row.source_row_id,
                        "dataset": row.dataset,
                        "tissue": row.tissue,
                        "marker": row.marker,
                    }
                )

    @staticmethod
    def batches(query: AnnotationQuery, batch_size: int = 30) -> Iterable[AnnotationQuery]:
        """Split a query explicitly so each GPTCelltype call has bounded size."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(query.rows), batch_size):
            yield AnnotationQuery(
                method_name=query.method_name,
                task_id=f"{query.task_id}__batch_{start // batch_size + 1:03d}",
                tissue_context=query.tissue_context,
                rows=query.rows[start : start + batch_size],
            )
