"""Adapt CellClick V5 to the benchmark prediction contract."""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CELLCLICK_SOURCE = REPOSITORY_ROOT / "src" / "cellclick"
if str(CELLCLICK_SOURCE) not in sys.path:
    sys.path.insert(0, str(CELLCLICK_SOURCE))

QUERY_FIELDS = ("source_row_id", "dataset", "tissue", "marker")
STANDARD_FIELDS = ("source_row_id", "dataset", "tissue", "prediction", "method", "model", "task_id")
CONTRACT = {"protocol_version": "v5", "prompt_version": "v5.10", "annotation_candidates": 5,
            "request_retries": 2, "max_parallel_clusters": 2}


def _read_query(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or tuple(rows[0]) != QUERY_FIELDS:
        raise ValueError("CellClick V5 query TSV must contain exactly source_row_id, dataset, tissue, and marker")
    if len({row["source_row_id"] for row in rows}) != len(rows):
        raise ValueError("CellClick V5 query TSV has duplicate source_row_id values")
    if any(not row[field].strip() for row in rows for field in QUERY_FIELDS):
        raise ValueError("CellClick V5 query TSV contains an empty required value")
    return rows


def _source_revision() -> str:
    result = subprocess.run(["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


_CANDIDATE_HANDLE = re.compile(r"^H\d+$", re.IGNORECASE)


def _prediction(record: dict[str, Any], result: dict[str, Any] | None = None) -> str | None:
    if record.get("verification_status") not in {"solved", "resolved"} or "population_annotations" in record:
        return None
    status = record.get("annotation_status")
    if status not in {"accepted", "major_accepted_subtype_unresolved"}:
        return None
    subtype = record.get("subtype")
    identity = record.get("identity")
    value = subtype if status == "accepted" and isinstance(subtype, str) and subtype.strip() else identity
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if _CANDIDATE_HANDLE.fullmatch(value):
        return None
    return value


def _write_metrics(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _metrics_path(output: Path) -> Path:
    for parent in output.parents:
        if parent.name == "predictions":
            return parent.parent / "cellclick_metrics.jsonl"
    return output.parent / "cellclick_metrics.jsonl"


def _write_prediction_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STANDARD_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_task(task: dict[str, Any]) -> None:
    from cellclick.llm import OpenAIChatClient
    from cellclick.registry import get_protocol
    from .schema_compat import V5SchemaCompatibleClient

    config = task.get("method_config", {})
    if not isinstance(config, dict):
        raise ValueError("CellClick V5 method_config must be a mapping")
    rounds = config.get("verifier_max_rounds", 6)
    if "max_rounds" in config or type(rounds) is not int or not 1 <= rounds <= 6:
        raise ValueError("CellClick V5 requires verifier_max_rounds between 1 and 6; max_rounds was removed")
    for field, expected in CONTRACT.items():
        actual = config.get(field, expected)
        if actual != expected:
            raise ValueError(f"CellClick V5 requires {field}={expected!r}")
    if not isinstance(config.get("cellmarker_reference_path"), str) or not config["cellmarker_reference_path"].strip():
        raise ValueError("CellClick V5 requires cellmarker_reference_path")
    if type(config.get("cellmarker_top_k", 5)) is not int or config.get("cellmarker_top_k", 5) < 1:
        raise ValueError("CellClick V5 cellmarker_top_k must be a positive integer")
    reference_max = config.get("reference_max", 3)
    if type(reference_max) is not int or not 0 <= reference_max <= 5:
        raise ValueError("CellClick V5 reference_max must be an integer between 0 and 5")
    subtype_enabled = config.get("subtype", True)
    if type(subtype_enabled) is not bool:
        raise ValueError("CellClick V5 subtype must be a boolean")
    comparison_mapping = config.get("comparison_reference_by_tissue", {})
    if not isinstance(comparison_mapping, dict):
        raise ValueError("CellClick V5 comparison_reference_by_tissue must be a mapping")
    for tissue, binding in comparison_mapping.items():
        if not isinstance(tissue, str) or not isinstance(binding, dict) or not all(
            isinstance(binding.get(key), str) and binding[key].strip() for key in ("ref_path", "h5ad_path")
        ):
            raise ValueError("CellClick V5 comparison references must contain ref_path and h5ad_path")
    protocol = get_protocol("v5")
    if config.get("prompt_version") != protocol.prompt_version:
        raise ValueError("CellClick V5 prompt_version does not match protocol")
    species = config.get("species")
    if not isinstance(species, str) or not species.strip():
        raise ValueError("CellClick V5 task lacks metadata-injected species")
    model = str(config.get("model", "gpt-4o")).strip()
    if not model:
        raise ValueError("CellClick V5 model must be non-empty")
    temperature = float(config.get("temperature", 0.0))
    if not 0 <= temperature <= 2:
        raise ValueError("CellClick V5 temperature must be between 0 and 2")
    rows = _read_query(Path(task["query_tsv"]))
    task_id = str(task.get("task_id", "")).strip()
    if not task_id:
        raise ValueError("CellClick V5 task_id must be non-empty")
    work_dir = Path(task["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = protocol.protocol_fingerprint()
    contract_fingerprint = protocol.contract_fingerprint()
    (work_dir / "protocol_manifest.json").write_text(json.dumps({
        "protocol_version": "v5", "prompt_version": protocol.prompt_version,
        "protocol_fingerprint": fingerprint, "contract_fingerprint": contract_fingerprint,
        "source_revision": _source_revision(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions: list[dict[str, str]] = []
    failures: list[str] = []
    metrics_path = _metrics_path(Path(task["output_tsv"]))
    checkpoint_path = work_dir / "predictions.partial.tsv"
    _write_prediction_rows(checkpoint_path, predictions)
    from cellclick.v5.runtime import DatasetResources, Runtime, Telemetry
    from cellclick.v5.candidate_registry import MAPPING_CACHE, EMBEDDING_CACHE, _get_index, _load_cl_names
    import httpx
    import shutil
    cache_mb = config.get("expression_cache_max_mb", 512)
    connect_timeout = config.get("request_connect_timeout_seconds", 10)
    read_timeout = config.get("request_read_timeout_seconds", 600)
    total_timeout = config.get("request_total_timeout_seconds", 610)
    for name, value in (("expression_cache_max_mb", cache_mb), ("connect timeout", connect_timeout),
                        ("read timeout", read_timeout), ("total timeout", total_timeout)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf"):
            raise ValueError(f"V5 {name} must be a finite positive number")
    dataset_started = time.perf_counter()
    resources = DatasetResources(config.get("expression_file"), config.get("expression_manifest"), cache_max_mb=cache_mb)
    sink = Telemetry(work_dir / "performance.jsonl")
    sink.write({"event": "dataset_load", "duration_seconds": resources.load_seconds})
    # Initialize shared reference/index state before any worker starts.
    reference_started = time.perf_counter()
    try:
        resources.retriever(config["cellmarker_reference_path"])
    except Exception as error:
        sink.write({"event": "reference_load", "status": "unavailable", "error_type": type(error).__name__})
    for binding in comparison_mapping.values():
        try:
            resources.json(binding["ref_path"])
        except (OSError, ValueError) as error:
            sink.write({"event": "reference_load", "status": "error", "error_type": type(error).__name__,
                        "reason": str(error)})
    mapping_path = work_dir / "label_to_clid.tsv"
    if not mapping_path.exists() and MAPPING_CACHE.is_file():
        shutil.copyfile(MAPPING_CACHE, mapping_path)
    obo_path = REPOSITORY_ROOT / "evaluator" / "ref_data" / "cl.obo"
    _load_cl_names(obo_path)
    try:
        _get_index(EMBEDDING_CACHE, obo_path)
    except ValueError:
        pass  # Existing missing-reference behavior is handled by the graph.
    sink.write({"event": "reference_init", "duration_seconds": time.perf_counter() - reference_started})
    (work_dir / "execution_manifest.json").write_text(json.dumps({
        "model": model, "max_parallel_clusters": 2, "expression_cache_max_mb": cache_mb,
        "request_connect_timeout_seconds": connect_timeout, "request_read_timeout_seconds": read_timeout,
        "request_total_timeout_seconds": total_timeout, "sdk_retries": 0,
        "request_retries": 2, "expression_fingerprint": resources.fingerprint,
        "subtype": subtype_enabled,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def execute(row):
        started = time.perf_counter()
        client = None
        try:
            client = V5SchemaCompatibleClient(OpenAIChatClient(
                timeout=httpx.Timeout(read_timeout, connect=connect_timeout), max_retries=0))
            from cellclick.v5.graph import annotate_cluster
            cluster_dir = work_dir / "clusters" / row["source_row_id"]
            cluster_dir.mkdir(parents=True, exist_ok=True)
            result = annotate_cluster(cluster_id=row["source_row_id"], dataset=row["dataset"], species=species.strip(),
                                      tissue=row["tissue"], markers=row["marker"].split(","), client=client,
                                      model=model, temperature=temperature, retries=CONTRACT["request_retries"],
                                      work_dir=cluster_dir, prompt_version=protocol.prompt_version, protocol_version="v5",
                                      runtime=Runtime(resources, sink, row["source_row_id"], mapping_path, total_timeout,
                                                      connect_timeout, read_timeout),
                                      expression_file=config.get("expression_file"),
                                      expression_manifest=config.get("expression_manifest"),
                                      cellmarker_reference_path=config.get("cellmarker_reference_path"),
                                      cellmarker_top_k=config.get("cellmarker_top_k", 5),
                                      reference_max=reference_max,
                                      subtype_enabled=subtype_enabled,
                                      verifier_max_rounds=rounds,
                                      comparison_reference_binding=(config.get("comparison_reference_by_tissue") or {}).get(row["tissue"]))
        except Exception as error:
            try:
                import asyncio
                from cellclick.llm import close_json_client
                asyncio.run(close_json_client(client))
            except Exception:
                pass
            from cellclick.v5.tools import ReferenceComparisonError
            reason = str(error) if isinstance(error, ReferenceComparisonError) else None
            return row, None, {"error_type": type(error).__name__, "reason": reason}, time.perf_counter() - started
        return row, result, None, time.perf_counter() - started

    completed = {}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cellclick-v5") as executor:
        futures = [executor.submit(execute, row) for row in rows]
        for future in as_completed(futures):
            row, result, error_type, elapsed = future.result()
            completed[row["source_row_id"]] = (result, error_type, elapsed)
            # Only the main thread writes aggregate state; preserve query order.
            partial = []
            for original in rows:
                item = completed.get(original["source_row_id"])
                prediction = _prediction((item[0] or {}).get("final_record") or {}) if item else None
                if prediction:
                    partial.append({**{key: original[key] for key in QUERY_FIELDS[:3]}, "prediction": prediction,
                                    "method": "cellclick", "model": model, "task_id": task_id})
            _write_prediction_rows(checkpoint_path, partial)
            if result is not None:
                (work_dir / f"{row['source_row_id']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            _write_metrics(metrics_path, [{"source_row_id": row["source_row_id"], "task_id": task_id,
                "status": (result or {}).get("status") if result and _prediction(result.get("final_record") or {}) else "error",
                "error_type": error_type.get("error_type") if error_type else None,
                "error_reason": error_type.get("reason") if error_type else None,
                "rounds_completed": (result or {}).get("round_count", 0),
                "rounds": (result or {}).get("rounds", {}),
                **{key: ((result or {}).get("final_record") or {}).get(key)
                   for key in ("annotation_status", "accepted_level", "stop_reason", "evaluation_mode")},
                "model": model, "protocol_version": "v5", "elapsed_seconds": round(elapsed, 6)}])
    import resource
    sink.write({"event": "dataset_complete", "wall_seconds": time.perf_counter() - dataset_started,
                "cluster_seconds_sum": sum(item[2] for item in completed.values()),
                "derived_cache_bytes": resources.cache.size,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    for row in rows:
        result, error_type, elapsed = completed[row["source_row_id"]]
        if error_type:
            failures.append(row["source_row_id"])
            continue
        final = result.get("final_record") or {}
        prediction = _prediction(final, result)
        if prediction is None:
            failures.append(row["source_row_id"])
            continue
        predictions.append({"source_row_id": row["source_row_id"], "dataset": row["dataset"], "tissue": row["tissue"],
                            "prediction": prediction, "method": "cellclick", "model": model, "task_id": task_id})
    if failures:
        raise RuntimeError(f"CellClick V5 execution incomplete for clusters: {', '.join(failures)}")
    output = Path(task["output_tsv"])
    _write_prediction_rows(output, predictions)
