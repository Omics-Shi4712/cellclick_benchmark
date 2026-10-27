"""Adapt CellClick LangGraph annotations to the repository prediction contract."""

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
STANDARD_FIELDS = ("source_row_id", "dataset", "tissue", "prediction", "method", "model", "task_id")


def read_query(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or tuple(rows[0]) != QUERY_FIELDS:
        raise ValueError("CellClick query TSV must contain exactly source_row_id, dataset, tissue, and marker")
    if len({row["source_row_id"] for row in rows}) != len(rows):
        raise ValueError("CellClick query TSV has duplicate source_row_id values")
    if any(not row[field].strip() for row in rows for field in QUERY_FIELDS):
        raise ValueError("CellClick query TSV contains an empty required value")
    return rows


def _source_revision() -> str:
    """Record the checkout revision when available without making it a runtime dependency."""
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _result_dir(output: Path) -> Path:
    """Find the runner output root while retaining a safe fallback for direct caller tests."""
    for parent in output.parents:
        if parent.name == "predictions":
            return parent.parent
    return output.parent


def run_task(task: dict[str, Any]) -> None:
    from caller.registry import caller_spec, normalize_caller_name
    from cellclick.llm import OpenAIChatClient
    from cellclick.metrics import append_metrics
    from cellclick.registry import get_protocol
    # Select the explicitly configured protocol; v1 remains frozen and v2 is independent.

    configured_caller = task.get("caller", "cellclick_v1")
    canonical_caller = normalize_caller_name(configured_caller)
    if canonical_caller not in {"cellclick_v1", "cellclick_v2", "cellclick_v3"}:
        raise ValueError(f"CellClick caller requires a versioned implementation, got {configured_caller!r}")
    caller_metadata = caller_spec(canonical_caller)

    rows = read_query(Path(task["query_tsv"]))
    task_id = str(task.get("task_id", "")).strip()
    if not task_id:
        raise ValueError("CellClick task_id must be a non-empty string")
    attempt = int(task.get("attempt", 1))
    if attempt < 1:
        raise ValueError("CellClick attempt must be a positive integer")
    config = task.get("method_config", {})
    species = config.get("species")
    if not isinstance(species, str) or not species.strip():
        raise ValueError("CellClick task lacks metadata-injected species")
    model = str(config.get("model", "gpt-4o"))
    configured_version = config.get("protocol_version")
    if not isinstance(configured_version, str) or not configured_version.strip():
        raise ValueError("CellClick requires an explicit protocol_version")
    protocol = get_protocol(configured_version)
    protocol_module = __import__(f"cellclick.{configured_version}", fromlist=["annotate_cluster"])
    annotate_cluster = protocol_module.annotate_cluster
    protocol_version = protocol.version
    retries = int(config.get("request_retries", protocol.request_retries))
    if retries != protocol.request_retries:
        raise ValueError(
            f"CellClick {protocol_version} requires request_retries={protocol.request_retries}"
        )
    if (
        int(config.get("annotation_candidates", protocol.annotation_candidates)) != protocol.annotation_candidates
        or int(config.get("max_rounds", protocol.max_rounds)) != protocol.max_rounds
    ):
        raise ValueError(
            f"CellClick {protocol_version} requires annotation_candidates={protocol.annotation_candidates} "
            f"and max_rounds={protocol.max_rounds}"
        )
    if int(config.get("max_parallel_clusters", protocol.max_parallel_clusters)) != protocol.max_parallel_clusters:
        raise ValueError(
            f"CellClick {protocol_version} requires max_parallel_clusters={protocol.max_parallel_clusters}"
        )
    prompt_version = config.get("prompt_version")
    if not isinstance(prompt_version, str) or prompt_version != protocol.prompt_version:
        raise ValueError(
            f"CellClick {protocol_version} requires prompt_version={protocol.prompt_version!r}"
        )
    temperature = float(config.get("temperature", 0.7))
    if not 0 <= temperature <= 2:
        raise ValueError("CellClick temperature must be between 0 and 2")
    work_dir = Path(task["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    rag_config: dict[str, Any] = {}
    if canonical_caller in {"cellclick_v2", "cellclick_v3"}:
        from cellclick.reference import DEFAULT_TOP_K, default_reference_path, reference_sha256

        reference_path = Path(config.get("rag_reference_path", default_reference_path())).resolve()
        rag_top_k = int(config.get("rag_top_k", DEFAULT_TOP_K))
        if rag_top_k < 1:
            raise ValueError("CellClick v2 rag_top_k must be positive")
        rag_config = {"rag_reference_path": reference_path, "rag_top_k": rag_top_k}
        if canonical_caller == "cellclick_v3":
            rag_config["embedding_cache_path"] = work_dir / "embedding_cache.json"
    fingerprint = protocol.protocol_fingerprint()
    contract_fingerprint = protocol.contract_fingerprint()
    revision = _source_revision()
    (work_dir / "protocol_manifest.json").write_text(
        json.dumps({
            "protocol_version": protocol_version,
            "prompt_version": prompt_version,
            "protocol_fingerprint": fingerprint,
            "contract_fingerprint": contract_fingerprint,
            "source_revision": revision,
            "rag": ({"reference_path": str(reference_path), "reference_sha256": reference_sha256(reference_path), "top_k": rag_top_k}
                    if canonical_caller in {"cellclick_v2", "cellclick_v3"} else None),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    predictions: list[dict[str, str]] = []
    metrics: list[dict[str, Any]] = []
    dataset_context = [
        {"cluster_id": item["source_row_id"], "ordered_markers": item["marker"].split(",")}
        for item in rows
    ]
    for row in rows:
        started = time.perf_counter()
        try:
            # Each protocol call owns its event loop via asyncio.run().  AsyncOpenAI
            # transports are bound to that loop, so a client must not survive into
            # the next cluster invocation after the previous loop has closed.
            client = OpenAIChatClient()
            v3_context = {"dataset_context": dataset_context} if canonical_caller == "cellclick_v3" else {}
            result = annotate_cluster(cluster_id=row["source_row_id"], dataset=row["dataset"], species=species.strip(),
                                      tissue=row["tissue"], markers=row["marker"].split(","), client=client, model=model,
                                      temperature=temperature, retries=retries, work_dir=work_dir, prompt_version=prompt_version,
                                      protocol_version=protocol_version, **rag_config, **v3_context)
        except Exception as error:
            metrics.append({"source_row_id": row["source_row_id"], "dataset": row["dataset"], "tissue": row["tissue"],
                            "task_id": task_id, "attempt": attempt, "status": "error", "error_type": type(error).__name__,
                            "elapsed_seconds": round(time.perf_counter() - started, 6),
                            "model": model, "protocol_version": protocol_version, "prompt_version": prompt_version,
                            "protocol_fingerprint": fingerprint, "contract_fingerprint": contract_fingerprint,
                            "source_revision": revision})
            output = Path(task["output_tsv"])
            append_metrics(_result_dir(output) / "cellclick_metrics.jsonl", metrics)
            raise
        final_record = result.get("final_record") or {}
        rounds = int(result.get("round_count", 0))
        call_counts = final_record.get("call_counts") if isinstance(final_record.get("call_counts"), dict) else {}
        annotation_calls = int(call_counts.get("candidates", rounds * protocol.annotation_candidates))
        validation_calls = int(call_counts.get("judgment", rounds))
        total_calls = int(call_counts.get("total", rounds * (protocol.annotation_candidates + 1)))
        metrics.append({"source_row_id": row["source_row_id"], "dataset": row["dataset"], "tissue": row["tissue"],
                        "task_id": task_id, "attempt": attempt, "status": result.get("status"),
                        "rounds_completed": rounds, "annotation_calls": annotation_calls,
                        "validation_calls": validation_calls, "formatting_calls": 0,
                        "total_calls": total_calls,
                        "selected_candidate_id": final_record.get("selected_candidate_id"),
                        "fallback_policy": (final_record.get("fallback") or {}).get("policy"),
                        "elapsed_seconds": round(time.perf_counter() - started, 6), "model": model,
                        "protocol_version": protocol_version, "prompt_version": prompt_version,
                        "protocol_fingerprint": fingerprint, "contract_fingerprint": contract_fingerprint,
                        "source_revision": revision})
        valid_statuses = {"validated", "fallback"}
        if result.get("status") not in valid_statuses or not isinstance(final_record.get("prediction"), str) or not final_record["prediction"].strip():
            output = Path(task["output_tsv"])
            append_metrics(_result_dir(output) / "cellclick_metrics.jsonl", metrics)
            raise RuntimeError(f"CellClick did not validate source_row_id {row['source_row_id']!r}; see {work_dir / 'traces'}")
        predictions.append({"source_row_id": row["source_row_id"], "dataset": row["dataset"], "tissue": row["tissue"],
                            "prediction": final_record["prediction"].strip(),
                            "method": caller_metadata["prediction_method"], "model": model,
                            "task_id": task_id})
    output = Path(task["output_tsv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STANDARD_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(predictions)
    append_metrics(_result_dir(output) / "cellclick_metrics.jsonl", metrics)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    args = parser.parse_args()
    run_task(json.loads(args.task.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
