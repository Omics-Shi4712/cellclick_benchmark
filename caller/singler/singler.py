"""Standard-library bridge to SingleR 1.4.1 inside the singler environment."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

MODEL = "SingleR-1.4.1_HPCA-celldex-1.0.0_label.main"
QUERY_FIELDS = ("source_row_id", "dataset", "tissue", "marker")
OUTPUT_FIELDS = ("source_row_id", "dataset", "tissue", "prediction", "method", "model", "task_id")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_task(task: dict[str, Any]) -> None:
    if task.get("data_adapter") != "expressionAdapter":
        raise ValueError("SingleR requires expressionAdapter")
    with Path(task["query_tsv"]).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != QUERY_FIELDS:
            raise ValueError("SingleR query must contain exactly the label-free query columns")
        rows = list(reader)
    ids = [row["source_row_id"] for row in rows]
    if not rows or len(set(ids)) != len(ids) or any(
        not row[field] or not row[field].strip() for row in rows for field in QUERY_FIELDS
    ):
        raise ValueError("SingleR query is empty, malformed, or has duplicate IDs")
    if len({row["tissue"] for row in rows}) != 1 or len({row["dataset"] for row in rows}) != 1:
        raise ValueError("SingleR task must contain exactly one dataset and tissue")
    config = task["method_config"]
    allowed = {"implementation", "expression_file", "hpca_rds", "hpca_sha256", "num_workers", "save_fullpred"}
    if set(config) - allowed:
        raise ValueError(f"Unsupported SingleR parameters: {sorted(set(config) - allowed)}")
    workers = config.get("num_workers", 1)
    save_fullpred = config.get("save_fullpred", True)
    if type(workers) is not int or workers < 1:
        raise ValueError("SingleR num_workers must be a positive integer")
    if type(save_fullpred) is not bool:
        raise ValueError("SingleR save_fullpred must be a boolean")

    def resolve_input(key: str) -> Path:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"SingleR {key} is required")
        path = Path(value)
        if not path.is_absolute():
            path = Path(task["config_dir"]) / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SingleR {key} not found: {path}")
        return path

    expression = resolve_input("expression_file")
    hpca = resolve_input("hpca_rds")
    digest = file_sha256(hpca)
    if config.get("hpca_sha256", digest) != digest:
        raise ValueError("SingleR HPCA content changed after task creation")
    work = Path(task["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    result_path = work / "population_predictions.tsv"
    result_path.unlink(missing_ok=True)
    subprocess.run([
        "Rscript", str(Path(__file__).with_name("singler_runner.R")),
        str(expression), str(hpca), str(result_path), str(work),
        str(workers), str(save_fullpred).lower(), rows[0]["tissue"],
    ], check=True)
    with result_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ("source_row_id", "prediction", "n_cells"):
            raise ValueError("SingleR result has unexpected columns")
        results = list(reader)
    predictions = {row["source_row_id"]: row["prediction"] for row in results}
    if len(predictions) != len(results) or set(predictions) != set(ids) or any(
        not value.strip() for value in predictions.values()
    ):
        raise ValueError("SingleR results do not align one-to-one with query IDs")
    protocol_path = work / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update({"hpca_rds": str(hpca), "hpca_sha256": digest, "model": MODEL})
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
                "method": "singler", "model": MODEL, "task_id": task["task_id"],
            })
