#!/usr/bin/env python3
"""Run configured cell-type annotation methods in isolated Conda environments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STANDARD_FIELDS = (
    "source_row_id",
    "dataset",
    "tissue",
    "prediction",
    "method",
    "model",
    "task_id",
)
SUPPORTED_METHODS = frozenset({"cassia", "gptcelltype"})


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Runner config must be a YAML mapping")
    return config


def config_digest(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def resolve_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def enabled_methods(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    methods = config.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("config.methods must be a mapping")
    selected: dict[str, dict[str, Any]] = {}
    for name, settings in methods.items():
        if name not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method {name!r}; supported: {', '.join(sorted(SUPPORTED_METHODS))}")
        if not isinstance(settings, dict):
            raise ValueError(f"methods.{name} must be a mapping")
        if settings.get("enabled", True):
            environment = settings.get("environment")
            if not isinstance(environment, str) or not environment.strip():
                raise ValueError(f"methods.{name}.environment must be a non-empty string")
            selected[name] = settings
    if not selected:
        raise ValueError("At least one enabled method is required")
    return selected


def select_queries(adapter: Any, selection: Any) -> list[Any]:
    if selection is None:
        selection = []
    if not isinstance(selection, list):
        raise ValueError("data.selection must be a list")
    if not selection:
        return adapter.generate_query("benchmark")
    queries: list[Any] = []
    seen: set[str] = set()
    for item in selection:
        if not isinstance(item, dict):
            raise ValueError("Each data.selection item must be a mapping")
        dataset, tissue = item.get("dataset"), item.get("tissue")
        if dataset is not None and not isinstance(dataset, str):
            raise ValueError("selection.dataset must be a string")
        if tissue is not None and not isinstance(tissue, str):
            raise ValueError("selection.tissue must be a string")
        for query in adapter.generate_query("benchmark", dataset=dataset, tissue=tissue):
            if query.task_id not in seen:
                queries.append(query)
                seen.add(query.task_id)
    return queries


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_standard_predictions(path: Path, query: Any) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Method did not create prediction TSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or not set(STANDARD_FIELDS).issubset(rows[0]):
        raise ValueError(f"{path} lacks standard prediction fields")
    expected_ids = [row.source_row_id for row in query.rows]
    result = {row["source_row_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path} contains duplicate source_row_id values")
    if set(result) != set(expected_ids):
        raise ValueError(f"{path} source_row_id values do not exactly match its query")
    forbidden = {"ground_truth", "clid", "synonyms", "broadtype", "manual annotation"}
    if forbidden & set(rows[0]):
        raise ValueError(f"{path} contains forbidden evaluation columns")
    return result


def task_payload(query: Any, method: str, settings: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    stem = query.task_id.replace("/", "_")
    return {
        "task_id": query.task_id,
        "query_tsv": str((output_dir / "inputs" / f"{stem}.tsv").resolve()),
        "output_tsv": str((output_dir / "predictions" / method / f"{stem}.tsv").resolve()),
        "work_dir": str((output_dir / "work" / method / stem).resolve()),
        "tissue_context": query.tissue_context,
        "method": method,
        "method_config": {key: value for key, value in settings.items() if key not in {"enabled", "environment"}},
    }


def prepare_run(config: dict[str, Any], config_path: Path, output_dir: Path) -> tuple[Any, list[Any], dict[str, dict[str, Any]]]:
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict) or evaluation.get("type", "celltypegpt") != "celltypegpt":
        raise ValueError("Only evaluation.type=celltypegpt is supported")
    data = config.get("data")
    if not isinstance(data, dict) or not data.get("marker_adapter_config"):
        raise ValueError("data.marker_adapter_config is required")
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from adapter.marker.marker import MarkerTableAdapter

    adapter = MarkerTableAdapter(resolve_path(data["marker_adapter_config"], config_path))
    queries = select_queries(adapter, data.get("selection", []))
    return adapter, queries, enabled_methods(config)


def initialize_output(output_dir: Path, digest: str, resume: bool) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"Output directory already exists: {output_dir}; use --resume to continue")
        if not manifest_path.is_file():
            raise FileExistsError(f"Cannot resume {output_dir}: manifest.json is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_digest") != digest:
            raise ValueError("Cannot resume: config content differs from the existing run")
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"config_digest": digest, "tasks": {}}
    write_json(manifest_path, manifest)
    return manifest


def invoke_method(method: str, settings: dict[str, Any], task_path: Path, log_path: Path, conda_executable: str) -> None:
    command = [
        conda_executable,
        "run",
        "--no-capture-output",
        "-n",
        settings["environment"],
        "python",
        str((Path(__file__).parent / "methods" / f"{method}_runner.py").resolve()),
        "--task",
        str(task_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=REPOSITORY_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError(f"{method} failed with exit code {completed.returncode}; see {log_path}")


def score_tables(adapter: Any, queries: list[Any], methods: dict[str, dict[str, Any]], output_dir: Path) -> None:
    from evaluator.scorer.celltypegpt import CellTypeGPTScorer

    scorer = CellTypeGPTScorer()
    prediction_maps: dict[str, dict[str, dict[str, str]]] = {}
    for method in methods:
        collected: dict[str, dict[str, str]] = {}
        for query in queries:
            path = output_dir / "predictions" / method / f"{query.task_id.replace('/', '_')}.tsv"
            try:
                collected.update(read_standard_predictions(path, query))
            except (FileNotFoundError, ValueError):
                continue
        prediction_maps[method] = collected

    rows: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        truth = {record["source_row_id"]: record for record in adapter.evaluation_rows(query)}
        for query_row in query.rows:
            record: dict[str, Any] = {
                "dataset": query_row.dataset,
                "tissue": query_row.tissue,
                "source_row_id": query_row.source_row_id,
                "ground_truth": truth[query_row.source_row_id]["ground_truth"],
            }
            for method in methods:
                prediction = prediction_maps[method].get(query_row.source_row_id)
                record[f"{method}_prediction"] = prediction["prediction"] if prediction else ""
                score = scorer.score_one(record[f"{method}_prediction"], record["ground_truth"]) if prediction else None
                record[f"{method}_evaluate_one"] = "" if score is None else score
            rows.append(record)
            by_dataset[query_row.dataset].append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    one_fields = ["dataset", "tissue", "source_row_id", "ground_truth"] + [
        field for method in methods for field in (f"{method}_prediction", f"{method}_evaluate_one")
    ]
    with (output_dir / "evaluation_one.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=one_fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows: list[dict[str, Any]] = []
    for dataset in sorted(by_dataset):
        aggregate: dict[str, Any] = {"dataset": dataset}
        dataset_rows = by_dataset[dataset]
        for method in methods:
            predictions = [row[f"{method}_prediction"] for row in dataset_rows]
            if any(not prediction for prediction in predictions):
                aggregate[f"{method}_evaluate"] = ""
                aggregate[f"{method}_status"] = "incomplete"
                continue
            scores = scorer.score_by_dataset(predictions, [row["ground_truth"] for row in dataset_rows], [dataset] * len(dataset_rows))
            value = scores.get(dataset)
            aggregate[f"{method}_evaluate"] = "" if value is None else value
            aggregate[f"{method}_status"] = "ok" if value is not None else "unscorable"
        aggregate_rows.append(aggregate)
    aggregate_fields = ["dataset"] + [field for method in methods for field in (f"{method}_evaluate", f"{method}_status")]
    with (output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)


def run(config_path: Path, resume: bool = False, dry_run: bool = False) -> int:
    config = load_config(config_path)
    output_setting = config.get("run", {}).get("output_dir") if isinstance(config.get("run", {}), dict) else None
    if not output_setting:
        raise ValueError("run.output_dir is required")
    output_dir = resolve_path(output_setting, config_path)
    adapter, queries, methods = prepare_run(config, config_path, output_dir)
    if dry_run:
        print(f"Validated {len(queries)} task groups for methods: {', '.join(methods)}")
        return 0
    manifest = initialize_output(output_dir, config_digest(config_path), resume)
    conda_executable = config.get("run", {}).get("conda_executable", "conda")
    failures = 0
    for query in queries:
        input_path = output_dir / "inputs" / f"{query.task_id.replace('/', '_')}.tsv"
        adapter.write_query_tsv(query, input_path)
        for method, settings in methods.items():
            key = f"{method}:{query.task_id}"
            task = task_payload(query, method, settings, output_dir)
            task_path = output_dir / "tasks" / method / f"{query.task_id.replace('/', '_')}.json"
            write_json(task_path, task)
            output_path = Path(task["output_tsv"])
            previous = manifest["tasks"].get(key, {})
            if resume and previous.get("status") == "success":
                try:
                    read_standard_predictions(output_path, query)
                    continue
                except (FileNotFoundError, ValueError):
                    pass
            try:
                invoke_method(method, settings, task_path, output_dir / "logs" / method / f"{query.task_id}.log", str(conda_executable))
                read_standard_predictions(output_path, query)
                manifest["tasks"][key] = {"status": "success", "output": str(output_path.relative_to(output_dir))}
            except Exception as error:
                failures += 1
                manifest["tasks"][key] = {"status": "failed", "error": str(error)}
            write_json(output_dir / "manifest.json", manifest)
    score_tables(adapter, queries, methods, output_dir)
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        raise SystemExit(run(args.config.resolve(), resume=args.resume, dry_run=args.dry_run))
    except Exception as error:
        print(f"runner error: {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
