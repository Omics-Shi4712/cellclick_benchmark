"""Benchmark-contract adapter for the CellClick V6 protocol."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "cellclick"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def run_task(task: dict[str, Any]) -> None:
    from cellclick.v6 import contract_fingerprint, protocol_fingerprint, run_task as run_v6
    from cellclick.v6.contract import PROMPT_VERSION
    # Prepare the submitted cluster workspaces before executing the V6 graph.
    work_dir = Path(task["work_dir"])
    with Path(task["query_tsv"]).open(encoding="utf-8-sig", newline="") as f:
        submitted = list(csv.DictReader(f, delimiter="\t"))
    for row in submitted:
        (work_dir / "clusters" / row["source_row_id"]).mkdir(parents=True, exist_ok=True)
    run_v6(task)
    expected = ("source_row_id", "dataset", "tissue", "prediction", "method", "model", "task_id")
    with Path(task["output_tsv"]).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError("CellClick V6 wrote an invalid prediction schema")
        rows = list(reader)
    expected_ids = [row["source_row_id"] for row in submitted]
    actual_ids = [row["source_row_id"] for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("CellClick V6 predictions must cover query clusters exactly once in query order")
    if any(not row["prediction"].strip() or row["task_id"] != str(task.get("task_id", "")) for row in rows):
        raise ValueError("CellClick V6 prediction rows have empty labels or unexpected task_id")
    manifest = {
        "protocol_version": "v6", "prompt_version": PROMPT_VERSION,
        "protocol_fingerprint": protocol_fingerprint(),
        "contract_fingerprint": contract_fingerprint(),
        "source_revision": "local",
    }
    (Path(task["work_dir"]) / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
