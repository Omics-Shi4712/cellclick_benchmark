"""Environment-side bridge for the pinned ScType reference implementation."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


MODEL = "sctype-v1.0"
QUERY_FIELDS = ("source_row_id", "dataset", "tissue", "marker")
OUTPUT_FIELDS = ("source_row_id", "dataset", "tissue", "prediction", "method", "model", "task_id")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_query(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != QUERY_FIELDS:
            raise ValueError("ScType query must contain exactly the label-free query columns")
        rows = list(reader)
    ids = [row["source_row_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)) or any(
        not row[field].strip() for row in rows for field in QUERY_FIELDS
    ):
        raise ValueError("ScType query is empty, malformed, or has duplicate IDs")
    if len({row["dataset"] for row in rows}) != 1 or len({row["tissue"] for row in rows}) != 1:
        raise ValueError("ScType task must contain exactly one dataset and tissue")
    return rows


def run_task(task: dict[str, Any]) -> None:
    if task.get("data_adapter") != "expressionAdapter":
        raise ValueError("ScType requires expressionAdapter")
    rows = _read_query(Path(task["query_tsv"]))
    config = task.get("method_config")
    if not isinstance(config, dict):
        raise ValueError("ScType method_config must be a mapping")
    allowed = {
        "implementation", "expression_file", "database_file", "database_sha256",
        "database_tissue", "cell_block_size",
    }
    if set(config) - allowed:
        raise ValueError(f"Unsupported ScType parameters: {sorted(set(config) - allowed)}")
    if config.get("database_tissue") != "All":
        raise ValueError("ScType database_tissue must be 'All'")
    block_size = config.get("cell_block_size", 2000)
    if type(block_size) is not int or block_size < 1:
        raise ValueError("ScType cell_block_size must be a positive integer")

    def input_file(key: str) -> Path:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"ScType {key} is required")
        path = Path(value)
        if not path.is_absolute():
            path = Path(task["config_dir"]) / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ScType {key} not found: {path}")
        return path

    expression = input_file("expression_file")
    database = input_file("database_file")
    digest = file_sha256(database)
    if config.get("database_sha256") != digest:
        raise ValueError("ScType database content changed after task creation")
    work_dir = Path(task["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = work_dir / "population_predictions.tsv"
    result_path.unlink(missing_ok=True)
    subprocess.run([
        "Rscript", str(Path(__file__).with_name("sctype_runner.R")), str(expression),
        str(database), digest, "All", str(block_size), str(result_path), str(work_dir),
    ], check=True)
    with result_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ("source_row_id", "prediction", "n_cells"):
            raise ValueError("ScType result has unexpected columns")
        results = list(reader)
    predictions = {row["source_row_id"]: row["prediction"] for row in results}
    expected = {row["source_row_id"] for row in rows}
    if len(predictions) != len(results) or set(predictions) != expected or any(
        not value.strip() for value in predictions.values()
    ):
        raise ValueError("ScType results do not align one-to-one with the query")
    protocol_path = work_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update({"database_file": str(database), "database_sha256": digest, "model": MODEL})
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = Path(task["output_tsv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "source_row_id": row["source_row_id"], "dataset": row["dataset"],
                "tissue": row["tissue"], "prediction": predictions[row["source_row_id"]],
                "method": "sctype", "model": MODEL, "task_id": task["task_id"],
            })
