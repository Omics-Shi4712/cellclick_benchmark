"""Stable semantic fingerprints for resumable benchmark tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def semantic_digest(payload: Any) -> str:
    """Hash JSON-compatible semantics independent of mapping presentation."""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_file_digest(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    return file_digest(resolved) if resolved.is_file() else None


def _path_signature(value: Any) -> dict[str, Any] | str | None:
    if not isinstance(value, str):
        return value
    path = Path(value)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "missing": True}
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "kind": "directory" if path.is_dir() else "file",
    }


def _expression_export_semantics(export: Any) -> dict[str, Any] | None:
    if not isinstance(export, dict):
        return None
    return {
        key: export.get(key)
        for key in ("dataset", "dataname", "groupby", "task_id", "tissue_context", "tissue_column")
        if key in export
    }


def expression_task_fingerprint(task: dict[str, Any], environment: str) -> str:
    expression = dict(task.get("expression", {}))
    marker_statistics_dir = expression.get("marker_statistics_dir")
    # The cache location itself is not semantic, but its filesystem identity
    # is: a changed cache must not silently reuse a previous marker result.
    expression["marker_statistics_dir_signature"] = _path_signature(marker_statistics_dir)
    expression.pop("marker_statistics_dir", None)
    expression.pop("skip_marker_gene", None)
    return semantic_digest({
        "schema": 1,
        "environment": environment,
        "h5ad": _path_signature(task.get("h5ad")),
        "expression": expression,
        "annotation_export": _expression_export_semantics(task.get("annotation_export")),
    })


def caller_task_semantics(
    task: dict[str, Any], *, environment: str | None = None, base_url: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "task_id": task.get("task_id"),
        "tissue_context": task.get("tissue_context"),
        "method": task.get("method"),
        "caller": task.get("caller"),
        "data_adapter": task.get("data_adapter"),
        "method_config": task.get("method_config", {}),
        "environment": environment,
        "base_url": base_url,
    }


def caller_task_fingerprint(
    task: dict[str, Any], *, environment: str, base_url: str | None,
    query_digest: str, expression_export_digest: str | None,
    upstream_fingerprint: str | None = None,
) -> str:
    return semantic_digest({
        **caller_task_semantics(task, environment=environment, base_url=base_url),
        "query_digest": query_digest,
        "expression_export_digest": expression_export_digest,
        "upstream_fingerprint": upstream_fingerprint,
    })


def legacy_caller_task_fingerprint(
    task: dict[str, Any], *, query_digest: str, expression_export_digest: str | None,
) -> str:
    """Fingerprint fields persisted by schema-v1 task JSON files."""
    return semantic_digest({
        **caller_task_semantics(task),
        "query_digest": query_digest,
        "expression_export_digest": expression_export_digest,
    })
