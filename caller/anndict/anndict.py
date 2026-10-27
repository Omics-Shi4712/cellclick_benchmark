"""Run AnnDictionary's marker-only annotation and label simplification flow."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any


QUERY_FIELDS = ("source_row_id", "dataset", "tissue", "marker")
OUTPUT_FIELDS = (
    "source_row_id", "dataset", "tissue", "prediction", "method", "model", "task_id",
)
SIMPLIFICATION_LEVEL = "redundancy-removed"
PROTOCOL_VERSION = "anndict.marker-label-simplification.v1"


def _read_query(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != QUERY_FIELDS:
            raise ValueError("AnnDictionary query must contain exactly the label-free query columns")
        rows = list(reader)
    identifiers = [row.get("source_row_id", "") for row in rows]
    if not rows or len(identifiers) != len(set(identifiers)) or any(
        not row.get(field, "").strip() for row in rows for field in QUERY_FIELDS
    ):
        raise ValueError("AnnDictionary query is empty, malformed, or has duplicate IDs")
    if len({row["dataset"] for row in rows}) != 1 or len({row["tissue"] for row in rows}) != 1:
        raise ValueError("AnnDictionary task must contain exactly one dataset and tissue")
    return rows


def _configure_runtime(work_dir: Path) -> None:
    cache_root = work_dir / "runtime_cache"
    numba_cache = cache_root / "numba"
    matplotlib_cache = cache_root / "matplotlib"
    numba_cache.mkdir(parents=True, exist_ok=True)
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
    # AnnDictionary otherwise writes a default response.txt in the process cwd.
    os.environ["RESPONSE_PATH"] = str(work_dir / "last_upstream_response.txt")


def _configure_anndict(config: dict[str, Any]) -> tuple[Any, str, str, str]:
    provider = config.get("provider", "openai")
    model = config.get("model")
    if provider != "openai":
        raise ValueError("AnnDictionary caller currently supports provider='openai' only")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("AnnDictionary model must be a non-empty string")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required for AnnDictionary provider=openai")

    import anndict as adt

    kwargs: dict[str, str] = {"api_key": api_key}
    base_url = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "").strip()
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    adt.configure_llm_backend(provider="openai", model=model.strip(), **kwargs)
    try:
        version = importlib.metadata.version("anndict")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return adt, model.strip(), version, str(Path(adt.__file__).resolve())


def _write_protocol_manifest(
    work_dir: Path, *, model: str, package_version: str, module_path: str,
) -> None:
    fingerprint_input = {
        "protocol_version": PROTOCOL_VERSION,
        "model": model,
        "provider": "openai",
        "simplification_level": SIMPLIFICATION_LEVEL,
        "annotation_api": "ai_cell_type",
        "package_version": package_version,
        "module_path": module_path,
        "python": platform.python_version(),
    }
    encoded = json.dumps(fingerprint_input, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": "anndict.ai_cell_type.upstream",
        "protocol_fingerprint": digest,
        "contract_fingerprint": hashlib.sha256(b"ai-cta-anndict-standard-prediction-v1").hexdigest(),
        "source_revision": f"anndict-{package_version}@{module_path};python-{platform.python_version()}",
    }
    (work_dir / "protocol_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def run_task(task: dict[str, Any]) -> None:
    if task.get("data_adapter") != "expressionAdapter":
        raise ValueError("AnnDictionary requires expressionAdapter")
    config = task.get("method_config")
    if not isinstance(config, dict):
        raise ValueError("AnnDictionary method_config must be a mapping")
    allowed = {"implementation", "provider", "model"}
    unexpected = set(config).difference(allowed)
    if unexpected:
        raise ValueError(f"Unsupported AnnDictionary parameters: {sorted(unexpected)}")

    rows = _read_query(Path(task["query_tsv"]))
    work_dir = Path(task["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    _configure_runtime(work_dir)
    adt, model, package_version, module_path = _configure_anndict(config)

    raw_predictions: dict[str, str] = {}
    for row in rows:
        genes = [gene.strip() for gene in row["marker"].split(",") if gene.strip()]
        if not genes:
            raise ValueError(f"AnnDictionary query row {row['source_row_id']!r} has no usable marker genes")
        raw = adt.ai_cell_type(genes, tissue=row["tissue"])
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"AnnDictionary returned an empty annotation for {row['source_row_id']!r}")
        raw_predictions[row["source_row_id"]] = raw.strip()

    raw_path = work_dir / "raw_predictions.json"
    raw_path.write_text(json.dumps(raw_predictions, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mapping = adt.map_cell_type_labels_to_simplified_set(
        list(raw_predictions.values()), simplification_level=SIMPLIFICATION_LEVEL,
    )
    if not isinstance(mapping, dict):
        raise ValueError("AnnDictionary label simplification did not return a mapping")
    simplified: dict[str, str] = {}
    for source_row_id, raw in raw_predictions.items():
        label = mapping.get(raw)
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"AnnDictionary simplification lacks a non-empty label for {source_row_id!r}")
        simplified[source_row_id] = label.strip()
    (work_dir / "raw_to_simplified.json").write_text(
        json.dumps({raw: mapping.get(raw) for raw in sorted(set(raw_predictions.values()))}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_protocol_manifest(work_dir, model=model, package_version=package_version, module_path=module_path)

    destination = Path(task["output_tsv"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "source_row_id": row["source_row_id"], "dataset": row["dataset"],
                "tissue": row["tissue"], "prediction": simplified[row["source_row_id"]],
                "method": "anndict", "model": model, "task_id": task["task_id"],
            })
