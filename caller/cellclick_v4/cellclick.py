"""Adapt the upstream CellClick V4 workflow to the benchmark caller contract."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CELLCLICK_SOURCE = REPOSITORY_ROOT / "src" / "cellclick"
if str(CELLCLICK_SOURCE) not in sys.path:
    sys.path.insert(0, str(CELLCLICK_SOURCE))

QUERY_FIELDS = ("source_row_id", "dataset", "tissue", "marker")
PREDICTION_FIELDS = ("source_row_id", "dataset", "tissue", "prediction", "method", "model", "task_id")


def read_query(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != QUERY_FIELDS:
            raise ValueError("CellClick V4 query TSV must contain exactly source_row_id, dataset, tissue, and marker")
        rows = list(reader)
    if not rows:
        raise ValueError("CellClick V4 query TSV contains no rows")
    if len({row["source_row_id"] for row in rows}) != len(rows):
        raise ValueError("CellClick V4 query TSV has duplicate source_row_id values")
    if any(not row[field].strip() for row in rows for field in QUERY_FIELDS):
        raise ValueError("CellClick V4 query TSV contains an empty required value")
    return rows


def _source_revision() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _metrics_path(output: Path) -> Path:
    for parent in output.parents:
        if parent.name == "predictions":
            return parent.parent / "cellclick_metrics.jsonl"
    return output.parent / "cellclick_metrics.jsonl"


def _append_metrics(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def run_task(task: dict[str, Any]) -> None:
    from cellclick.llm import OpenAIChatClient
    from cellclick.metrics import append_metrics
    from caller.cellclick_v4 import v45

    config = task.get("method_config", {})
    if not isinstance(config, dict):
        raise ValueError("CellClick V4 method_config must be a mapping")
    configured_protocol = config.get("protocol_version")
    if configured_protocol == "v4.5":
        expected_contract = {
            "prompt_version": "v4.5", "annotation_candidates": v45.ANNOTATION_CANDIDATES,
            "max_rounds": v45.MAX_QUERY_ROUNDS_TOTAL, "request_retries": v45.REQUEST_RETRIES,
            "max_parallel_clusters": v45.MAX_PARALLEL_CLUSTERS,
        }
        for field, expected in expected_contract.items():
            if config.get(field, expected) != expected:
                raise ValueError(f"CellClick V4.5 requires {field}={expected}")
        protocol = None
    else:
        from cellclick.registry import get_protocol
        if configured_protocol != "v4.4" or config.get("prompt_version") != "v4.4":
            raise ValueError("CellClick V4 requires protocol_version=v4.4 or v4.5 with matching prompt_version")
        protocol = get_protocol("v4.4")
        for field in ("annotation_candidates", "max_rounds", "request_retries", "max_parallel_clusters"):
            expected = getattr(protocol, field)
            if int(config.get(field, expected)) != expected:
                raise ValueError(f"CellClick V4 requires {field}={expected}")

    species = config.get("species")
    if not isinstance(species, str) or not species.strip():
        raise ValueError("CellClick V4 task lacks metadata-injected species")
    expression_file = config.get("expression_file")
    expression_manifest = config.get("expression_manifest")
    if not isinstance(expression_file, str) or not expression_file.strip():
        raise ValueError("CellClick V4 requires data.expressionAdapter and an expression_file")
    if not isinstance(expression_manifest, str) or not expression_manifest.strip():
        raise ValueError("CellClick V4 requires an expression_manifest")

    rows = read_query(Path(task["query_tsv"]))
    task_id = str(task.get("task_id", "")).strip()
    if not task_id:
        raise ValueError("CellClick V4 task_id must be non-empty")
    model = str(config.get("model", "gpt-4o")).strip()
    if not model:
        raise ValueError("CellClick V4 model must be non-empty")
    temperature = float(config.get("temperature", 0.0))
    if not 0 <= temperature <= 2:
        raise ValueError("CellClick V4 temperature must be between 0 and 2")

    work_dir = Path(task["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    if configured_protocol == "v4.5":
        annotate_cluster = v45.annotate_cluster
        fingerprint = v45.protocol_fingerprint()
        contract_fingerprint = v45.contract_fingerprint()
    else:
        module = __import__("cellclick.v4.graph", fromlist=["annotate_cluster"])
        annotate_cluster = module.annotate_cluster
        fingerprint_module = __import__("cellclick.v4.contract", fromlist=["protocol_fingerprint", "contract_fingerprint"])
        fingerprint = fingerprint_module.protocol_fingerprint()
        contract_fingerprint = fingerprint_module.contract_fingerprint()
    revision = _source_revision()
    (work_dir / "protocol_manifest.json").write_text(
        json.dumps({
            "protocol_version": configured_protocol, "prompt_version": config.get("prompt_version"),
            "protocol_fingerprint": fingerprint, "contract_fingerprint": contract_fingerprint,
            "source_revision": revision,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    predictions: list[dict[str, str]] = []
    metrics: list[dict[str, Any]] = []
    for row in rows:
        started = time.perf_counter()
        try:
            markers = [gene.strip() for gene in row["marker"].split(",") if gene.strip()]
            if not markers:
                raise ValueError(f"CellClick V4 row {row['source_row_id']!r} has no marker genes")
            v45_client_options = (
                {"subtype_client_factory": OpenAIChatClient}
                if configured_protocol == "v4.5" else {}
            )
            result = annotate_cluster(
                cluster_id=row["source_row_id"], dataset=row["dataset"], species=species.strip(),
                tissue=row["tissue"], markers=markers,
                client=OpenAIChatClient(), model=model, temperature=temperature,
                **v45_client_options,
                retries=(protocol.request_retries if protocol is not None else v45.REQUEST_RETRIES), work_dir=work_dir,
                prompt_version=("v4.4" if configured_protocol == "v4.4" else "v4.5"),
                protocol_version=configured_protocol,
                expression_file=expression_file, expression_manifest=expression_manifest,
            )
            record = result.get("final_record") or {}
            prediction = record.get("prediction")
            decision_status = result.get("status")
            if decision_status not in {"validated", "unresolved"} or not isinstance(prediction, str) or not prediction.strip():
                raise RuntimeError(f"CellClick V4 did not produce a recommended prediction for {row['source_row_id']!r}")
            call_counts = record.get("call_counts") if isinstance(record.get("call_counts"), dict) else {}
            metrics.append({
                "source_row_id": row["source_row_id"], "dataset": row["dataset"], "tissue": row["tissue"],
                "task_id": task_id, "status": decision_status, "decision_status": decision_status,
                "rounds_completed": result.get("round_count", 0),
                "boost_cycles_completed": result.get("round_count", 0),
                "annotation_calls": sum(int(call_counts.get(stage, 0)) for stage in ("a1", "a2", "a3")),
                "validation_calls": int(call_counts.get("a4", 0)), "formatting_calls": 0,
                "total_calls": call_counts.get("total", 0), "elapsed_seconds": round(time.perf_counter() - started, 6),
                "model": model, "protocol_version": configured_protocol, "prompt_version": config.get("prompt_version"),
                "protocol_fingerprint": fingerprint, "contract_fingerprint": contract_fingerprint,
                "source_revision": revision,
                "decision_history": record.get("decision_history", []),
            })
            predictions.append({
                "source_row_id": row["source_row_id"], "dataset": row["dataset"], "tissue": row["tissue"],
                "prediction": prediction.strip(), "method": "cellclick", "model": model, "task_id": task_id,
            })
        except Exception as error:
            metrics.append({
                "source_row_id": row["source_row_id"], "dataset": row["dataset"], "tissue": row["tissue"],
                "task_id": task_id, "status": "error", "error_type": type(error).__name__,
                "elapsed_seconds": round(time.perf_counter() - started, 6), "model": model,
                "protocol_version": configured_protocol, "prompt_version": config.get("prompt_version"), "protocol_fingerprint": fingerprint,
                "contract_fingerprint": contract_fingerprint, "source_revision": revision,
            })
            _append_metrics(_metrics_path(Path(task["output_tsv"])), metrics)
            raise

    output = Path(task["output_tsv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(predictions)
    append_metrics(_metrics_path(output), metrics)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    args = parser.parse_args()
    run_task(json.loads(args.task.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
