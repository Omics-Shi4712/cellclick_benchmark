#!/usr/bin/env python3
"""GPTCelltype environment-side runner with bounded R invocations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any


_LEADING_LIST_MARKER = re.compile(r"^\s*(?:[-*]+|\d+[.)])\s+")
_PARENTHETICAL_QUALIFIER = re.compile(r"\s*\([^)]*\)")


def normalize_prediction(value: object) -> str:
    label = _LEADING_LIST_MARKER.sub("", str(value)).strip()
    return _PARENTHETICAL_QUALIFIER.sub("", label).strip()


def read_query(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"source_row_id", "dataset", "tissue", "marker"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Query TSV lacks required label-free columns")
    if len({row["source_row_id"] for row in rows}) != len(rows):
        raise ValueError("Query TSV has duplicate source_row_id values")
    return rows


def write_query_batch(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_row_id", "dataset", "tissue", "marker"), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run_task(task: dict[str, Any]) -> None:
    data_adapter = task.get("data_adapter", "tableAdapter")
    if data_adapter not in {"tableAdapter", "expressionAdapter"}:
        raise ValueError(f"Unsupported data_adapter {data_adapter!r}")
    # Both adapters deliberately converge to the same label-free TSV contract.
    rows = read_query(Path(task["query_tsv"]))
    config = task.get("method_config", {})
    batch_size = int(config.get("batch_size", 30))
    if not 1 <= batch_size <= 30:
        raise ValueError("gptcelltype batch_size must be between 1 and 30")
    work_dir = Path(task["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    predicted: dict[str, str] = {}
    bridge = Path(__file__).with_name("gptcelltype_runner.R")
    model = str(config.get("model", "gpt-4"))
    for index in range(0, len(rows), batch_size):
        batch = rows[index : index + batch_size]
        query_path = work_dir / f"batch_{index // batch_size + 1:03d}.tsv"
        prediction_path = work_dir / f"batch_{index // batch_size + 1:03d}.predictions.tsv"
        write_query_batch(batch, query_path)
        subprocess.run(["Rscript", str(bridge), str(query_path), str(prediction_path), task["tissue_context"], model], check=True)
        with prediction_path.open(encoding="utf-8-sig", newline="") as handle:
            result = list(csv.DictReader(handle, delimiter="\t"))
        by_id = {row.get("source_row_id", ""): normalize_prediction(row.get("prediction", "")) for row in result}
        if set(by_id) != {row["source_row_id"] for row in batch} or any(not value for value in by_id.values()):
            raise ValueError("GPTCelltype batch does not align one-to-one with the query")
        predicted.update(by_id)
    output_path = Path(task["output_tsv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_row_id", "dataset", "tissue", "prediction", "method", "model", "task_id"), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({"source_row_id": row["source_row_id"], "dataset": row["dataset"], "tissue": row["tissue"], "prediction": predicted[row["source_row_id"]], "method": "gptcelltype", "model": model, "task_id": task["task_id"]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path)
    args = parser.parse_args()
    run_task(json.loads(args.task.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
