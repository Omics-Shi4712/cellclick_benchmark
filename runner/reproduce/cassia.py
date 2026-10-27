"""Execute, natively score, and compare a live CASSIA reproduction."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from runner.reproduce.common import read_reference_rows, resolve_reference_file, write_comparison_summary


REQUIRED_TOP_N = 50
REFERENCE_CONTEXT_COLUMNS = ("Dataset", "Tissue")
REFERENCE_EVALUATION_COLUMN = "Evaluation"
REFERENCE_SCORE_COLUMN = "Score"
REFERENCE_DATASET_ALIASES = {"tsv1": "ts"}


def _normalize_context_value(value: str) -> str:
    """Normalize presentation-only variation in historical context names."""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _reference_context(dataset: str, tissue: str) -> tuple[str, str]:
    normalized_dataset = _normalize_context_value(dataset)
    return REFERENCE_DATASET_ALIASES.get(normalized_dataset, normalized_dataset), _normalize_context_value(tissue)


def _historical_reference_summaries(reference_file: Path, queries: list[Any]) -> dict[tuple[str, str], dict[str, object]]:
    """Summarize published CASSIA rows by context without matching marker lists.

    Expression-derived queries use independently recomputed markers and cannot
    be row-matched to the publication's frozen marker table.  The reference is
    therefore a historical context-level comparator only.
    """
    rows = read_reference_rows(reference_file)
    required = {*REFERENCE_CONTEXT_COLUMNS, REFERENCE_EVALUATION_COLUMN, REFERENCE_SCORE_COLUMN}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"cassia reference_file lacks required columns: {', '.join(sorted(missing))}")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not row["Dataset"].strip() or not row["Tissue"].strip():
            raise ValueError("cassia reference_file contains an empty Dataset or Tissue")
        try:
            float(row[REFERENCE_EVALUATION_COLUMN])
            float(row[REFERENCE_SCORE_COLUMN])
        except (TypeError, ValueError) as error:
            raise ValueError("cassia reference_file Evaluation and Score must be numeric") from error
        grouped[_reference_context(row["Dataset"], row["Tissue"])].append(row)

    selected = {_reference_context(row.dataset, row.tissue) for query in queries for row in query.rows}
    result: dict[tuple[str, str], dict[str, object]] = {}
    for context in selected & set(grouped):
        context_rows = grouped[context]
        result[context] = {
            "published_dataset": context_rows[0]["Dataset"].strip(),
            "published_tissue": context_rows[0]["Tissue"].strip(),
            "published_rows": len(context_rows),
            "published_evaluation": sum(float(row[REFERENCE_EVALUATION_COLUMN]) for row in context_rows) / len(context_rows),
            "published_score": sum(float(row[REFERENCE_SCORE_COLUMN]) for row in context_rows) / len(context_rows),
        }
    if not result:
        raise ValueError("cassia reference_file has no shared dataset/tissue contexts after alias normalization")
    return result


def _pipeline_scores(queries: list[Any], caller: str, output_dir: Path) -> dict[str, float]:
    """Read CASSIA's Boost-preceding native pipeline scores after annotation."""
    scores: dict[str, float] = {}
    for query in queries:
        stem = query.task_id.replace("/", "_")
        work_dir = output_dir / "work" / caller / stem
        input_path = work_dir / "cassia_native_scores.csv"
        with input_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_ids = {row.source_row_id for row in query.rows}
        by_id = {row.get("Cluster ID", "").strip(): row for row in rows}
        if set(by_id) != expected_ids or len(by_id) != len(rows):
            raise ValueError(f"CASSIA pipeline score output does not align with {query.task_id}")
        for source_row_id, row in by_id.items():
            try:
                scores[source_row_id] = float(row.get("Score", ""))
            except (TypeError, ValueError) as error:
                raise ValueError(f"CASSIA pipeline score output has no numeric Score for {source_row_id}") from error
    return scores


def _write_evaluations(adapter: Any, queries: list[Any], predictions: dict[str, dict[str, str]], reference_summaries: dict[tuple[str, str], dict[str, object]], scores: dict[str, float], output_dir: Path) -> tuple[Path, Path]:
    """Write live per-row results and shared-context historical summaries."""
    rows: list[dict[str, object]] = []
    by_context: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for query in queries:
        truth = {record["source_row_id"]: record for record in adapter.evaluation_rows(query)}
        expected = {row.source_row_id for row in query.rows}
        if set(truth) != expected:
            raise ValueError(f"Adapter evaluation rows do not exactly match {query.task_id}")
        for query_row in query.rows:
            source_row_id = query_row.source_row_id
            prediction = predictions[source_row_id]["prediction"]
            result = {
                "dataset": query_row.dataset, "tissue": query_row.tissue, "source_row_id": source_row_id,
                "ground_truth": truth[source_row_id]["ground_truth"], "cassia_prediction": prediction,
                "cassia_native_score": scores[source_row_id],
            }
            rows.append(result)
            by_context[_reference_context(query_row.dataset, query_row.tissue)].append(result)

    one_file = output_dir / "evaluation_one.csv"
    fields = ("dataset", "tissue", "source_row_id", "ground_truth", "cassia_prediction", "cassia_native_score")
    with one_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows: list[dict[str, object]] = []
    for context in sorted(set(by_context) & set(reference_summaries)):
        live_rows = by_context[context]
        published = reference_summaries[context]
        aggregate_rows.append({
            "live_rows": len(live_rows), "live_native_score": sum(float(row["cassia_native_score"]) for row in live_rows) / len(live_rows),
            **published, "comparison_status": "historical_context_only",
        })
    aggregate_file = output_dir / "evaluation.csv"
    aggregate_fields = ("published_dataset", "published_tissue", "live_rows", "live_native_score", "published_rows", "published_evaluation", "published_score", "comparison_status")
    with aggregate_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return one_file, aggregate_file


def run(config, adapter, queries, callers, config_path, output_dir, *, resume: bool = False, force: bool = False, max_parallel_tasks: int = 1):
    """Run CASSIA and compare shared contexts with historical publication summaries."""
    from runner.run import (
        collect_validated_predictions, config_digest, execute_tasks, initialize_output,
        write_combined_predictions,
    )

    for name, settings in callers.items():
        if settings.get("implementation", name) != "cassia":
            continue
        adapter_top_n = getattr(adapter, "top_n", None)
        if adapter_top_n is not None and int(adapter_top_n or 0) != REQUIRED_TOP_N:
            raise ValueError(f"CASSIA reproduction requires adapter top_n={REQUIRED_TOP_N}")
        manifest = initialize_output(output_dir, config_digest(config_path), resume=resume, force=force)
        conda_executable = str(config.get("run", {}).get("conda_executable", "conda"))
        failures = execute_tasks(
            adapter, queries, {name: settings}, output_dir, config_path, conda_executable,
            manifest, resume=resume, force=force,
            max_parallel_tasks=max_parallel_tasks,
        )
        if failures:
            raise RuntimeError(f"cassia test failed for {failures} task(s); see {output_dir / 'manifest.json'}")
        prediction_file = write_combined_predictions(queries, name, output_dir)
        predictions = collect_validated_predictions(queries, name, output_dir)
        scores = _pipeline_scores(queries, name, output_dir)
        reference_file = resolve_reference_file(settings, config_path, "cassia")
        reference_summaries = _historical_reference_summaries(reference_file, queries)
        evaluation_one_file, evaluation_file = _write_evaluations(adapter, queries, predictions, reference_summaries, scores, output_dir)
        summary_file = write_comparison_summary(output_dir, "cassia", prediction_file, reference_file, rows=len(predictions), shared_contexts=len(reference_summaries), reference_id_column="Dataset/Tissue (alias-normalized)", comparison="live expression-derived CASSIA run versus published context-level historical summaries; marker lists and predictions are not row-matched")
        return {"method": "cassia", "tasks": len(queries), "rows": len(predictions), "status": "historical_comparison", "prediction_file": str(prediction_file), "evaluation_one_file": str(evaluation_one_file), "evaluation_file": str(evaluation_file), "reference_file": str(reference_file), "summary_file": str(summary_file)}
    raise ValueError("No CASSIA caller selected")
