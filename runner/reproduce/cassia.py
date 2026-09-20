"""Execute, natively score, and compare a live CASSIA reproduction."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from runner.reproduce.common import read_reference_rows, resolve_reference_file, write_comparison_summary


REQUIRED_TOP_N = 50
REFERENCE_MATCH_COLUMNS = ("Dataset", "Tissue", "Marker List")
REFERENCE_PREDICTION_COLUMN = "Predicted Main Cell Type"
REFERENCE_EVALUATION_COLUMN = "Evaluation"
REFERENCE_SCORE_COLUMN = "Score"
SCORE_TOLERANCE = 1e-6


def _marker_key(value: str, top_n: int) -> str:
    genes = [gene.strip() for gene in value.split(",") if gene.strip()]
    return ",".join(genes[:top_n])


def _reference_records(reference_file: Path, queries: list[Any], top_n: int) -> dict[str, dict[str, str]]:
    """Read the fixed CASSIA reference format after live predictions freeze."""
    rows = read_reference_rows(reference_file)
    required = {*REFERENCE_MATCH_COLUMNS, REFERENCE_PREDICTION_COLUMN, REFERENCE_EVALUATION_COLUMN, REFERENCE_SCORE_COLUMN}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"cassia reference_file lacks required columns: {', '.join(sorted(missing))}")

    def key(row: dict[str, str]) -> tuple[str, str, str]:
        return row["Dataset"].strip(), row["Tissue"].strip(), _marker_key(row["Marker List"], top_n)

    by_key = {key(row): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("cassia reference_file contains duplicate Dataset/Tissue/Marker List rows")
    result: dict[str, dict[str, str]] = {}
    for query in queries:
        for row in query.rows:
            reference = by_key.get((row.dataset, row.tissue, row.marker))
            if reference is None:
                raise ValueError("cassia reference_file lacks a selected Dataset/Tissue/Marker List row")
            for column in (REFERENCE_PREDICTION_COLUMN, REFERENCE_EVALUATION_COLUMN, REFERENCE_SCORE_COLUMN):
                if not reference[column].strip():
                    raise ValueError(f"cassia reference_file contains an empty {column}")
            try:
                float(reference[REFERENCE_EVALUATION_COLUMN])
                float(reference[REFERENCE_SCORE_COLUMN])
            except ValueError as error:
                raise ValueError("cassia reference_file Evaluation and Score must be numeric") from error
            result[row.source_row_id] = reference
    return result


def _score_tasks(queries: list[Any], caller: str, settings: dict[str, Any], output_dir: Path, conda_executable: str) -> dict[str, float]:
    """Call CASSIA's native score batch for each frozen task summary."""
    from runner.run import invoke_caller, write_json

    scores: dict[str, float] = {}
    for query in queries:
        stem = query.task_id.replace("/", "_")
        work_dir = output_dir / "work" / caller / stem
        input_path = work_dir / "cassia_summary.csv"
        output_path = output_dir / "scores" / caller / f"{stem}.csv"
        task_path = output_dir / "tasks" / caller / f"{stem}.score.json"
        task = {
            "action": "score", "caller": settings["implementation"],
            "score_input_csv": str(input_path.resolve()), "score_output_csv": str(output_path.resolve()),
            "conversations_json_path": str((work_dir / "cassia_conversations.json").resolve()),
            "method_config": {key: value for key, value in settings.items() if key not in {"enabled", "environment", "implementation", "reference_file"}},
        }
        write_json(task_path, task)
        invoke_caller(caller, settings, task_path, output_dir / "logs" / caller / f"{stem}.score.log", conda_executable)
        with output_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_ids = {row.source_row_id for row in query.rows}
        by_id = {row.get("Cluster ID", "").strip(): row for row in rows}
        if set(by_id) != expected_ids or len(by_id) != len(rows):
            raise ValueError(f"CASSIA score output does not align with {query.task_id}")
        for source_row_id, row in by_id.items():
            try:
                scores[source_row_id] = float(row.get("Score", ""))
            except (TypeError, ValueError) as error:
                raise ValueError(f"CASSIA score output has no numeric Score for {source_row_id}") from error
    return scores


def _write_evaluations(adapter: Any, queries: list[Any], predictions: dict[str, dict[str, str]], references: dict[str, dict[str, str]], scores: dict[str, float], output_dir: Path) -> tuple[Path, Path]:
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
            reference = references[source_row_id]
            reference_evaluation = float(reference[REFERENCE_EVALUATION_COLUMN])
            current_score, reference_score = scores[source_row_id], float(reference[REFERENCE_SCORE_COLUMN])
            result = {
                "dataset": query_row.dataset, "tissue": query_row.tissue, "source_row_id": source_row_id,
                "ground_truth": truth[source_row_id]["ground_truth"], "cassia_prediction": prediction,
                "reference_prediction": reference[REFERENCE_PREDICTION_COLUMN].strip(),
                "prediction_match": prediction == reference[REFERENCE_PREDICTION_COLUMN].strip(),
                "reference_evaluation": reference_evaluation,
                "cassia_score": current_score, "reference_score": reference_score,
                "score_difference": current_score - reference_score,
                "score_match": abs(current_score - reference_score) <= SCORE_TOLERANCE,
            }
            rows.append(result)
            by_dataset[query_row.dataset].append(result)

    one_file = output_dir / "evaluation_one.csv"
    fields = ("dataset", "tissue", "source_row_id", "ground_truth", "cassia_prediction", "reference_prediction", "prediction_match", "reference_evaluation", "cassia_score", "reference_score", "score_difference", "score_match")
    with one_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows: list[dict[str, object]] = []
    for dataset in sorted(by_dataset):
        dataset_rows = by_dataset[dataset]
        reference_evaluations = [float(row["reference_evaluation"]) for row in dataset_rows]
        current_scores = [float(row["cassia_score"]) for row in dataset_rows]
        reference_scores = [float(row["reference_score"]) for row in dataset_rows]
        reference_evaluation = sum(reference_evaluations) / len(reference_evaluations)
        current_score, reference_score = sum(current_scores) / len(current_scores), sum(reference_scores) / len(reference_scores)
        aggregate_rows.append({
            "dataset": dataset, "reference_evaluate": reference_evaluation,
            "cassia_score": current_score, "reference_score": reference_score, "score_difference": current_score - reference_score,
            "prediction_match_rate": sum(bool(row["prediction_match"]) for row in dataset_rows) / len(dataset_rows),
            "cassia_status": "ok", "reference_status": "ok",
        })
    aggregate_file = output_dir / "evaluation.csv"
    aggregate_fields = ("dataset", "reference_evaluate", "cassia_score", "reference_score", "score_difference", "prediction_match_rate", "cassia_status", "reference_status")
    with aggregate_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return one_file, aggregate_file


def run(config, adapter, queries, callers, config_path, output_dir):
    """Run CASSIA, validate predictions, score them, then read the reference."""
    from runner.run import collect_validated_predictions, execute_tasks, write_combined_predictions, write_json

    for name, settings in callers.items():
        if settings.get("implementation", name) != "cassia":
            continue
        if int(getattr(adapter, "top_n", 0) or 0) != REQUIRED_TOP_N:
            raise ValueError(f"CASSIA reproduction requires adapter top_n={REQUIRED_TOP_N}")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"config_digest": "test", "tasks": {}}
        write_json(output_dir / "manifest.json", manifest)
        conda_executable = str(config.get("run", {}).get("conda_executable", "conda"))
        failures = execute_tasks(adapter, queries, {name: settings}, output_dir, config_path, conda_executable, manifest)
        if failures:
            raise RuntimeError(f"cassia test failed for {failures} task(s); see {output_dir / 'manifest.json'}")
        prediction_file = write_combined_predictions(queries, name, output_dir)
        predictions = collect_validated_predictions(queries, name, output_dir)
        scores = _score_tasks(queries, name, settings, output_dir, conda_executable)
        reference_file = resolve_reference_file(settings, config_path, "cassia")
        references = _reference_records(reference_file, queries, REQUIRED_TOP_N)
        evaluation_one_file, evaluation_file = _write_evaluations(adapter, queries, predictions, references, scores, output_dir)
        summary_file = write_comparison_summary(output_dir, "cassia", prediction_file, reference_file, rows=len(predictions), reference_rows=len(references), reference_id_column="Dataset/Tissue/Marker List", reference_prediction_column=REFERENCE_PREDICTION_COLUMN, comparison="live CASSIA predictions, native scores, and benchmark evaluations")
        return {"method": "cassia", "tasks": len(queries), "rows": len(predictions), "status": "validated", "prediction_file": str(prediction_file), "evaluation_one_file": str(evaluation_one_file), "evaluation_file": str(evaluation_file), "reference_file": str(reference_file), "summary_file": str(summary_file)}
    raise ValueError("No CASSIA caller selected")
