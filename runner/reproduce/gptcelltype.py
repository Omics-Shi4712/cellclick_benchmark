"""Execute and compare a GPTCelltype benchmark reproduction.

Test mode uses the same adapter, caller, prediction validation, and evaluator
as a normal run. The supplied reference is used only after live predictions
have been frozen.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluator.scorer.celltypegpt import DATASET_NAME_ALIASES, CellTypeGPTScorer
from runner.reproduce.common import read_reference_rows, resolve_reference_file, write_comparison_summary


REFERENCE_MATCH_COLUMNS = ("dataset", "tissue", "marker")
REFERENCE_PREDICTION_COLUMN = "GPT-4 (June 13, 2023) annotation"


def _reference_predictions(
    reference_file: Path, queries: list[Any], marker_top_n: int | None = None,
) -> dict[str, str]:
    """Read the fixed published CellTypeGPT marker-table result format."""
    rows = read_reference_rows(reference_file)
    missing = {REFERENCE_PREDICTION_COLUMN, *REFERENCE_MATCH_COLUMNS} - set(rows[0])
    if missing:
        raise ValueError(f"gptcelltype reference_file lacks required columns: {', '.join(sorted(missing))}")
    def marker_key(value: str) -> str:
        # The adapter serializes the original ordered genes as comma-separated
        # tokens without spaces. Reference tables may retain display spaces.
        genes = [gene.strip() for gene in value.split(",")]
        return ",".join(genes[:marker_top_n] if marker_top_n is not None else genes)

    def reference_key(row: dict[str, str]) -> tuple[str, ...]:
        return tuple(
            marker_key(row[column]) if column == "marker" else row[column].strip()
            for column in REFERENCE_MATCH_COLUMNS
        )

    result_by_key = {reference_key(row): row[REFERENCE_PREDICTION_COLUMN].strip() for row in rows}
    expected_by_key = {
        (row.dataset, row.tissue, row.marker)
        for query in queries for row in query.rows
    }
    if len(result_by_key) != len(rows):
        raise ValueError("gptcelltype reference_file contains duplicate dataset/tissue/marker values")
    if not expected_by_key.issubset(result_by_key):
        raise ValueError("gptcelltype reference_file lacks selected dataset/tissue/marker values")
    result = {
        row.source_row_id: result_by_key[(row.dataset, row.tissue, row.marker)]
        for query in queries for row in query.rows
    }
    if any(not prediction for prediction in result.values()):
        raise ValueError(f"gptcelltype reference_file contains an empty {REFERENCE_PREDICTION_COLUMN}")
    return result


def _write_evaluations(
    adapter: Any,
    queries: list[Any],
    predictions: dict[str, dict[str, str]],
    references: dict[str, str],
    output_dir: Path,
) -> tuple[Path, Path]:
    scorer = CellTypeGPTScorer()
    rows: list[dict[str, object]] = []
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for query in queries:
        truth = {record["source_row_id"]: record for record in adapter.evaluation_rows(query)}
        expected = {row.source_row_id for row in query.rows}
        if set(truth) != expected:
            raise ValueError(f"Adapter evaluation rows do not exactly match {query.task_id}")
        for query_row in query.rows:
            source_row_id = query_row.source_row_id
            prediction = predictions[source_row_id]["prediction"]
            reference_prediction = references[source_row_id]
            ground_truth = truth[source_row_id]["ground_truth"]
            current_score = scorer.score_one(prediction, ground_truth)
            reference_score = scorer.score_one(reference_prediction, ground_truth)
            row = {
                "dataset": query_row.dataset,
                "tissue": query_row.tissue,
                "source_row_id": source_row_id,
                "ground_truth": ground_truth,
                "gptcelltype_prediction": prediction,
                "reference_prediction": reference_prediction,
                "prediction_match": prediction == reference_prediction,
                "gptcelltype_evaluate_one": "" if current_score is None else current_score,
                "reference_evaluate_one": "" if reference_score is None else reference_score,
                "score_difference": "" if current_score is None or reference_score is None else current_score - reference_score,
            }
            rows.append(row)
            by_dataset[query_row.dataset].append(row)

    one_file = output_dir / "evaluation_one.csv"
    fields = (
        "dataset", "tissue", "source_row_id", "ground_truth", "gptcelltype_prediction",
        "reference_prediction", "prediction_match", "gptcelltype_evaluate_one",
        "reference_evaluate_one", "score_difference",
    )
    with one_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows: list[dict[str, object]] = []
    for dataset in sorted(by_dataset):
        dataset_rows = by_dataset[dataset]
        current_scores = scorer.score_by_dataset(
            [str(row["gptcelltype_prediction"]) for row in dataset_rows],
            [str(row["ground_truth"]) for row in dataset_rows],
            [dataset] * len(dataset_rows),
        )
        reference_scores = scorer.score_by_dataset(
            [str(row["reference_prediction"]) for row in dataset_rows],
            [str(row["ground_truth"]) for row in dataset_rows],
            [dataset] * len(dataset_rows),
        )
        score_key = DATASET_NAME_ALIASES.get(dataset, dataset)
        current_value = current_scores[score_key]
        reference_value = reference_scores[score_key]
        aggregate_rows.append({
            "dataset": dataset,
            "gptcelltype_evaluate": "" if current_value is None else current_value,
            "reference_evaluate": "" if reference_value is None else reference_value,
            "score_difference": "" if current_value is None or reference_value is None else current_value - reference_value,
            "gptcelltype_status": "ok" if current_value is not None else "unscorable",
            "reference_status": "ok" if reference_value is not None else "unscorable",
        })
    aggregate_file = output_dir / "evaluation.csv"
    aggregate_fields = (
        "dataset", "gptcelltype_evaluate", "reference_evaluate", "score_difference",
        "gptcelltype_status", "reference_status",
    )
    with aggregate_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return one_file, aggregate_file


def run(config, adapter, queries, callers, config_path, output_dir):
    """Run GPTCelltype, score it, and compare the result to a reference."""
    from runner.run import collect_validated_predictions, execute_tasks, write_combined_predictions, write_json

    for name, settings in callers.items():
        if settings.get("implementation", name) != "gptcelltype":
            continue
        batch_size = int(settings.get("batch_size", 30))
        if not 1 <= batch_size <= 30:
            raise ValueError("GPTCelltype reproduction requires batch_size between 1 and 30")
        reference_file = resolve_reference_file(settings, config_path, "gptcelltype")
        references = _reference_predictions(
            reference_file, queries, getattr(adapter, "top_n", None),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"config_digest": "test", "tasks": {}}
        write_json(output_dir / "manifest.json", manifest)
        conda_executable = str(config.get("run", {}).get("conda_executable", "conda"))
        failures = execute_tasks(adapter, queries, {name: settings}, output_dir, config_path, conda_executable, manifest)
        if failures:
            raise RuntimeError(f"gptcelltype test failed for {failures} task(s); see {output_dir / 'manifest.json'}")
        prediction_file = write_combined_predictions(queries, name, output_dir)
        predictions = collect_validated_predictions(queries, name, output_dir)
        evaluation_one_file, evaluation_file = _write_evaluations(adapter, queries, predictions, references, output_dir)
        summary_file = write_comparison_summary(
            output_dir, "gptcelltype", prediction_file, reference_file,
            rows=len(predictions), reference_rows=len(references),
            reference_id_column="dataset/tissue/marker",
            reference_prediction_column=REFERENCE_PREDICTION_COLUMN,
            comparison="live caller predictions and CellTypeGPT scores",
        )
        return {
            "method": "gptcelltype", "tasks": len(queries), "rows": len(predictions), "status": "validated",
            "prediction_file": str(prediction_file), "evaluation_one_file": str(evaluation_one_file),
            "evaluation_file": str(evaluation_file), "reference_file": str(reference_file),
            "summary_file": str(summary_file),
        }
    raise ValueError("No GPTCelltype caller selected")
