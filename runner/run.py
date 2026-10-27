#!/usr/bin/env python3
"""Run or reproduce configured cell-type annotation workflows.

``run`` is method-agnostic orchestration: adapter -> caller -> evaluator.
``test`` dispatches to a method-specific reproduction check under
``runner.reproduce``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from caller.registry import CALLER_REGISTRY, caller_spec, normalize_caller_name
from runner.fingerprint import (
    caller_task_fingerprint,
    expression_task_fingerprint,
    file_digest,
    legacy_caller_task_fingerprint,
    optional_file_digest,
    semantic_digest,
)
STANDARD_FIELDS = (
    "source_row_id",
    "dataset",
    "tissue",
    "prediction",
    "method",
    "model",
    "task_id",
)
SUPPORTED_CALLERS = frozenset((*CALLER_REGISTRY, "cellclick"))
SUPPORTED_DATA_ADAPTERS = frozenset({"expressionAdapter", "tableAdapter"})
# Compatibility alias for downstream code that imported the old constant.
SUPPORTED_METHODS = SUPPORTED_CALLERS
_ENVIRONMENT_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CALLER_SECRET_SETTINGS = frozenset({"api_key", "api_key_env", "base_url"})
CELLCLICK_PROTOCOL_CONTRACTS = {
    "v1": {
        "prompt_version": "v1",
        "annotation_candidates": 5,
        "max_rounds": 3,
        "request_retries": 2,
        "max_parallel_clusters": 1,
    },
    "v2": {
        "prompt_version": "v2", "annotation_candidates": 3, "max_rounds": 1,
        "request_retries": 2, "max_parallel_clusters": 1,
    },
    "v3": {
        "prompt_version": "v3", "annotation_candidates": 3, "max_rounds": 1,
        "request_retries": 2, "max_parallel_clusters": 1,
    },
    "v4.4": {
        "prompt_version": "v4.4", "annotation_candidates": 3, "max_rounds": 5,
        "request_retries": 2, "max_parallel_clusters": 1,
    },
    "v4.5": {
        "prompt_version": "v4.5", "annotation_candidates": 5, "max_rounds": 5,
        "request_retries": 2, "max_parallel_clusters": 1,
    },
    "v5": {
        "prompt_version": "v5.10", "annotation_candidates": 5,
        "request_retries": 2, "max_parallel_clusters": 2,
    },
    "v6": {
        "prompt_version": "v6.12", "annotation_candidates": 5,
        "request_retries": 2, "max_parallel_clusters": 1,
    },
    "v7": {
        "prompt_version": "v7.1", "annotation_candidates": 5,
        "request_retries": 2, "max_parallel_clusters": 1,
    },
    "v7.2": {
        "prompt_version": "v7.2", "annotation_candidates": 5,
        "request_retries": 2, "max_parallel_clusters": 1,
    },
    "v7.3": {
        "prompt_version": "v7.3", "annotation_candidates": 5,
        "request_retries": 2, "max_parallel_clusters": 1,
    },
}


def _is_cellclick(settings: dict[str, Any]) -> bool:
    return normalize_caller_name(settings["implementation"]) in {"cellclick_v1", "cellclick_v2", "cellclick_v3", "cellclick_v4", "cellclick_v5", "cellclick_v6", "cellclick_v7", "cellclick_v7_2", "cellclick_v7_3"}
_PROTOCOL_MANIFEST_FIELDS = frozenset({
    "protocol_version",
    "prompt_version",
    "protocol_fingerprint",
    "contract_fingerprint",
    "source_revision",
})
_RAG_MANIFEST_FIELDS = frozenset({"reference_path", "reference_sha256", "top_k"})


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


def _expression_settings(data: dict[str, Any]) -> dict[str, Any] | None:
    """Return enabled expression settings, preserving the optional contract."""
    settings = data.get("expressionAdapter")
    if settings is None:
        return None
    if not isinstance(settings, dict) or not settings.get("enabled", True):
        if isinstance(settings, dict):
            return None
        raise ValueError("data.expressionAdapter must be a mapping")
    environment = settings.get("environment")
    if not isinstance(environment, str) or not environment.strip():
        raise ValueError("data.expressionAdapter.environment must be a non-empty string")
    return settings


def _read_expression_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Expression metadata file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Expression metadata is not valid JSON: {path}") from error
    if not isinstance(metadata, dict) or not isinstance(metadata.get("datasets"), dict):
        raise ValueError("Expression metadata must contain a datasets mapping")
    return metadata


def dataset_species_from_metadata(path: Path) -> tuple[dict[str, str], str | None]:
    """Read public dataset-level species metadata for CASSIA prompts."""
    metadata = _read_expression_metadata(path)
    default_species = metadata.get("default_species")
    if default_species is not None and (not isinstance(default_species, str) or not default_species.strip()):
        raise ValueError("Expression metadata default_species must be a non-empty string")
    species_by_dataset: dict[str, str] = {}
    for dataset_name, dataset in metadata["datasets"].items():
        if not isinstance(dataset, dict):
            raise ValueError(f"Expression metadata dataset {dataset_name!r} must be a mapping")
        species = dataset.get("species", default_species)
        if not isinstance(species, str) or not species.strip():
            raise ValueError(f"Expression metadata dataset {dataset_name!r} lacks a non-empty species")
        species_by_dataset[dataset_name] = species.strip()
    return species_by_dataset, default_species.strip() if isinstance(default_species, str) else None


def configure_metadata_species(
    callers: dict[str, dict[str, Any]], data: dict[str, Any], config_path: Path,
) -> dict[str, dict[str, Any]]:
    """Attach public metadata species to callers that require it, without serializing labels."""
    species_callers = {"cassia", "cellclick_v1", "cellclick_v2", "cellclick_v3", "cellclick_v4", "cellclick_v5", "cellclick_v6", "cellclick_v7", "cellclick_v7_2", "cellclick_v7_3"}
    if not any(normalize_caller_name(settings["implementation"]) in species_callers for settings in callers.values()):
        return callers
    configured_path = data.get("dataset_metadata_file")
    metadata_path = (
        resolve_path(configured_path, config_path)
        if isinstance(configured_path, str) and configured_path.strip()
        else REPOSITORY_ROOT / "data" / "expression_data" / "h5ad_metadata.json"
    )
    species_by_dataset, default_species = dataset_species_from_metadata(metadata_path)
    return {
        name: (
            {**settings, "_runner_species_by_dataset": species_by_dataset, "_runner_default_species": default_species}
            if normalize_caller_name(settings["implementation"]) in species_callers else settings
        )
        for name, settings in callers.items()
    }


def configure_cassia_species(
    callers: dict[str, dict[str, Any]], data: dict[str, Any], config_path: Path,
) -> dict[str, dict[str, Any]]:
    """Backward-compatible alias for metadata species injection."""
    return configure_metadata_species(callers, data, config_path)


def _metadata_expression_tasks(
    settings: dict[str, Any], config_path: Path, *, skip_marker_gene: bool | None,
    output_dir: Path | None = None, implementations: set[str] | None = None,
) -> list[tuple[str, dict[str, Any], str]]:
    metadata_value = settings.get("metadata_file")
    if not isinstance(metadata_value, str) or not metadata_value.strip():
        raise ValueError("data.expressionAdapter.metadata_file must be a non-empty string")
    marker_root = settings.get("marker_statistics_root")
    if not isinstance(marker_root, str) or not marker_root.strip():
        raise ValueError("data.expressionAdapter.marker_statistics_root must be a non-empty string")
    metadata = _read_expression_metadata(resolve_path(metadata_value, config_path))
    selected = settings.get("datasets", sorted(metadata["datasets"]))
    if not isinstance(selected, list) or not selected or any(not isinstance(name, str) or not name.strip() for name in selected):
        raise ValueError("data.expressionAdapter.datasets must be a non-empty list of dataset names")
    if len(set(selected)) != len(selected):
        raise ValueError("data.expressionAdapter.datasets contains duplicates")

    ignored = {"enabled", "environment", "metadata_file", "datasets", "marker_statistics_root"}
    common = {key: value for key, value in settings.items() if key not in ignored}
    root = resolve_path(marker_root, config_path)
    environment = settings["environment"]
    canonical_implementations = {
        normalize_caller_name(implementation) for implementation in (implementations or set())
    }
    uses_sctype = "sctype" in canonical_implementations
    tasks: list[tuple[str, dict[str, Any], str]] = []
    for dataset_name in selected:
        dataset = metadata["datasets"].get(dataset_name)
        if not isinstance(dataset, dict):
            raise ValueError(f"Expression metadata has no dataset {dataset_name!r}")
        h5ad_files = dataset.get("h5ad_files")
        obs_keys = dataset.get("obs_keys")
        raw_count_layer = dataset.get("raw_counts_location")
        species = dataset.get("species", metadata.get("default_species", "human"))
        if not isinstance(species, str) or not species.strip():
            raise ValueError(f"Expression metadata dataset {dataset_name!r} lacks a non-empty species")
        if not isinstance(h5ad_files, dict) or not h5ad_files:
            raise ValueError(f"Expression metadata dataset {dataset_name!r} lacks h5ad_files")
        if not isinstance(obs_keys, dict) or not all(isinstance(obs_keys.get(key), str) and obs_keys[key].strip() for key in ("tissue", "cell_type")):
            raise ValueError(f"Expression metadata dataset {dataset_name!r} lacks obs_keys.tissue or obs_keys.cell_type")
        if uses_sctype:
            # The benchmark's native marker protocol defines populations from
            # the reference cell-type partition.  An optional annotation_group
            # supports a separately supplied clustering partition instead.
            annotation_group = obs_keys.get("annotation_group", obs_keys["cell_type"])
            if not isinstance(annotation_group, str) or not annotation_group.strip():
                raise ValueError(f"ScType annotation_group for {dataset_name!r} must be a non-empty string")
            if species.strip().lower() != "human":
                raise ValueError(f"ScType supports only human expression datasets, got {species!r} for {dataset_name!r}")
            if "singler" in canonical_implementations and annotation_group != obs_keys["cell_type"]:
                raise ValueError(
                    "ScType and SingleR must share data.expressionAdapter.obs_keys.cell_type "
                    "when enabled in the same run"
                )
        if not isinstance(raw_count_layer, str) or not raw_count_layer.strip():
            raise ValueError(f"Expression metadata dataset {dataset_name!r} lacks raw_counts_location")
        for dataname, h5ad_value in h5ad_files.items():
            if not isinstance(dataname, str) or not dataname.strip() or Path(dataname).name != dataname:
                raise ValueError(f"Expression metadata has invalid data name in {dataset_name!r}")
            if not isinstance(h5ad_value, str) or not h5ad_value.strip():
                raise ValueError(f"Expression metadata has invalid h5ad path for {dataname!r}")
            h5ad = Path(h5ad_value)
            if not h5ad.is_file():
                raise FileNotFoundError(f"Expression h5ad file does not exist: {h5ad}")
            expression = {
                **common,
                "dataname": dataname,
                "marker_statistics_dir": str(root / dataset_name / "statistics"),
                # Every caller receives the same metadata-defined analysis
                # unit. ScType's optional annotation_group is respected when
                # explicitly configured, while SingleR follows cell_type.
                "groupby": annotation_group if uses_sctype else obs_keys["cell_type"],
                "raw_count_layer": raw_count_layer,
                "tissue_column": obs_keys["tissue"],
                "cell_type_column": obs_keys["cell_type"],
            }
            if skip_marker_gene is not None:
                expression["skip_marker_gene"] = skip_marker_gene
            task_name = f"{dataset_name}/{dataname}"
            task = {"h5ad": str(h5ad), "expression": expression}
            if output_dir is not None:
                stem = task_name.replace("/", "_")
                task["work_dir"] = str((output_dir / "expression" / "work" / stem).resolve())
                task["annotation_export"] = {
                    "dataset": dataset_name,
                    "dataname": dataname,
                    "groupby": expression["groupby"],
                    "tissue_column": obs_keys["tissue"],
                    "task_id": task_name,
                    "tissue_context": f"{species.strip()} {dataname.replace('_', ' ')}",
                    "query_tsv": str((output_dir / "expression_queries" / f"{stem}.tsv").resolve()),
                    "evaluation_json": str((output_dir / "expression_evaluation" / f"{stem}.json").resolve()),
                }
                task["method_exports"] = {
                    implementation: str((
                        output_dir / f"expression_{implementation}"
                        / f"{stem}{'.h5' if normalize_caller_name(implementation) in {'singler', 'sctype'} else ('.h5ad' if normalize_caller_name(implementation) in {'cellclick_v4', 'cellclick_v5', 'cellclick_v6', 'cellclick_v7', 'cellclick_v7_2', 'cellclick_v7_3'} else '.csv')}"
                    ).resolve())
                    for implementation in sorted(implementations or ())
                }
            tasks.append((task_name, task, environment))
    return tasks


def expression_task_configs(
    data: dict[str, Any], config_path: Path, *, skip_marker_gene: bool | None = None,
    output_dir: Path | None = None, implementations: set[str] | None = None,
) -> list[tuple[str, dict[str, Any], str]] | None:
    """Resolve legacy single-h5ad or metadata-expanded expression marker jobs."""
    settings = _expression_settings(data)
    if settings is None:
        return None
    if "metadata_file" in settings:
        return _metadata_expression_tasks(
            settings, config_path, skip_marker_gene=skip_marker_gene,
            output_dir=output_dir, implementations=implementations,
        )
    required = ("h5ad", "dataname", "marker_statistics_dir", "groupby")
    missing = [name for name in required if not settings.get(name)]
    if missing:
        raise ValueError("data.expressionAdapter lacks required field(s): " + ", ".join(missing))
    h5ad = resolve_path(settings["h5ad"], config_path)
    if not h5ad.is_file():
        raise FileNotFoundError(f"Expression h5ad file does not exist: {h5ad}")
    expression = {key: value for key, value in settings.items() if key not in {"enabled", "environment", "h5ad"}}
    expression["marker_statistics_dir"] = str(resolve_path(expression["marker_statistics_dir"], config_path))
    if skip_marker_gene is not None:
        expression["skip_marker_gene"] = skip_marker_gene
    task = {"h5ad": str(h5ad), "expression": expression}
    if output_dir is not None:
        task["work_dir"] = str((output_dir / "expression" / "work" / "expression").resolve())
    return [("expression", task, settings["environment"])]


def expression_task_config(
    data: dict[str, Any], config_path: Path, *, skip_marker_gene: bool | None = None,
) -> tuple[dict[str, Any], str] | None:
    """Backward-compatible accessor for callers that expect one expression task."""
    tasks = expression_task_configs(data, config_path, skip_marker_gene=skip_marker_gene)
    if tasks is None:
        return None
    if len(tasks) != 1:
        raise ValueError("Expression configuration expands to multiple tasks; use expression_task_configs")
    _, task, environment = tasks[0]
    return task, environment


def invoke_expression_adapter(
    task: dict[str, Any], environment: str, output_dir: Path, conda_executable: str, *, task_name: str = "expression",
) -> str:
    """Run expression preprocessing in its declared Conda environment."""
    stem = task_name.replace("/", "_")
    task_path = output_dir / "expression" / "tasks" / f"{stem}.json"
    log_path = output_dir / "logs" / "expression" / f"{stem}.log"
    write_json(task_path, task)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        conda_executable, "run", "--no-capture-output", "-n", environment,
        "python", "-m", "runner.expression_task", "--task", str(task_path),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=REPOSITORY_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError(f"ExpressionDataAdapter failed with exit code {completed.returncode}; see {log_path}")
    return str(task_path.relative_to(output_dir))


def _expression_task_path(output_dir: Path, task_name: str) -> Path:
    return output_dir / "expression" / "tasks" / f"{task_name.replace('/', '_')}.json"


def _load_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _expression_core_outputs(task: dict[str, Any]) -> dict[str, str]:
    export = task.get("annotation_export")
    if not isinstance(export, dict):
        return {}
    return {
        name: value for name, value in {
            "query_tsv": export.get("query_tsv"),
            "evaluation_json": export.get("evaluation_json"),
        }.items() if isinstance(value, str)
    }


def _expression_exports(task: dict[str, Any]) -> dict[str, str]:
    exports = task.get("method_exports", {})
    return {
        str(name): str(path) for name, path in exports.items()
        if isinstance(name, str) and isinstance(path, str)
    } if isinstance(exports, dict) else {}


def _recorded_artifact_matches(record: Any, path: Path) -> bool:
    if not path.is_file():
        return False
    if isinstance(record, dict) and isinstance(record.get("sha256"), str):
        return file_digest(path) == record["sha256"]
    return True


def execute_expression_tasks(
    tasks: list[tuple[str, dict[str, Any], str]], output_dir: Path, conda_executable: str,
    manifest: dict[str, Any], *, resume: bool = False, force: bool = False,
    max_parallel_tasks: int = 1,
) -> int:
    """Run every configured h5ad marker task and retain per-file manifest status."""
    if type(max_parallel_tasks) is not int or max_parallel_tasks < 1:
        raise ValueError("max_parallel_tasks must be a positive integer")
    section = manifest.setdefault("expression_adapter", {"tasks": {}})
    task_statuses = section.setdefault("tasks", {})
    jobs: list[tuple[str, dict[str, Any], dict[str, Any], str, Path, dict[str, str], dict[str, str], int, bool]] = []
    for task_name, task, environment in tasks:
        previous = task_statuses.get(task_name, {})
        fingerprint = expression_task_fingerprint(task, environment)
        task_path = _expression_task_path(output_dir, task_name)
        core_outputs = _expression_core_outputs(task)
        current_exports = _expression_exports(task)
        core_matches = previous.get("fingerprint") == fingerprint
        if not previous.get("fingerprint") and previous.get("status") == "success":
            stored_task = _load_json_mapping(task_path)
            core_matches = (
                stored_task is not None
                and expression_task_fingerprint(stored_task, environment) == fingerprint
            )
        core_artifacts_valid = all(
            _recorded_artifact_matches(previous.get("core_outputs", {}).get(name), Path(path))
            for name, path in core_outputs.items()
        ) and len(core_outputs) == 2
        missing_exports = {
            implementation: path for implementation, path in current_exports.items()
            if not _recorded_artifact_matches(previous.get("exports", {}).get(implementation), Path(path))
            or (normalize_caller_name(implementation) == "cassia"
                and previous.get("exports", {}).get(implementation, {}).get("marker_scope") != "all_genes_v1")
            or (normalize_caller_name(implementation) == "singler"
                and previous.get("exports", {}).get(implementation, {}).get("schema_version") != 2)
            or (normalize_caller_name(implementation) in {"cellclick_v4", "cellclick_v5", "cellclick_v6", "cellclick_v7", "cellclick_v7_2", "cellclick_v7_3"}
                and previous.get("exports", {}).get(implementation, {}).get("manifest_schema_version")
                != "cellclick.expression.2")
        }
        if resume and not force and previous.get("status") == "success" and core_matches and core_artifacts_valid:
            if not missing_exports:
                previous["active_in_current_config"] = True
                previous["fingerprint"] = fingerprint
                print(f"[REUSE] expression {task_name}: semantic fingerprint matched")
                continue
            print(
                f"[EXPORT] expression {task_name}: generating "
                + ", ".join(sorted(missing_exports))
            )
            invocation_task = {**task, "method_exports": missing_exports}
        else:
            reason = "forced" if force else "new or changed data task"
            print(f"[RUN]   expression {task_name}: {reason}")
            invocation_task = task
        attempt = int(previous.get("attempt", 0)) + 1
        write_json(task_path, task)
        jobs.append((task_name, task, invocation_task, environment, task_path, core_outputs,
                     current_exports, attempt, force))

    def run_job(job: tuple[str, dict[str, Any], dict[str, Any], str, Path, dict[str, str], dict[str, str], int, bool]) -> tuple[str, dict[str, Any]]:
        task_name, desired_task, invocation_task, environment, task_path, core_outputs, current_exports, attempt, forced = job
        try:
            relative_task_path = invoke_expression_adapter(
                invocation_task, environment, output_dir, conda_executable, task_name=task_name,
            )
            # Persist the complete desired task even when the worker only had
            # to materialize newly requested caller exports.
            write_json(task_path, desired_task)
            final_fingerprint = expression_task_fingerprint(desired_task, environment)
            record = {
                "status": "success", "task": relative_task_path,
                "fingerprint": final_fingerprint,
                "core_outputs": {
                    name: {"path": path, "sha256": file_digest(path)}
                    for name, path in core_outputs.items()
                },
                "exports": {
                    implementation: {
                        "path": path, "sha256": file_digest(path),
                        **({"marker_scope": "all_genes_v1"} if normalize_caller_name(implementation) == "cassia" else {}),
                        **({"schema_version": 2} if normalize_caller_name(implementation) == "singler" else {}),
                        **({"manifest_schema_version": "cellclick.expression.2"}
                           if normalize_caller_name(implementation) in {"cellclick_v4", "cellclick_v5", "cellclick_v6", "cellclick_v7", "cellclick_v7_2", "cellclick_v7_3"} else {}),
                    }
                    for implementation, path in current_exports.items()
                },
                "attempt": attempt,
                "forced": forced,
                "active_in_current_config": True,
            }
        except Exception as error:
            record = {
                "status": "failed", "error": str(error),
                "fingerprint": expression_task_fingerprint(desired_task, environment),
                "attempt": attempt,
                "forced": forced,
                "active_in_current_config": True,
            }
        return task_name, record

    failures = 0
    if jobs:
        with ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            futures = [executor.submit(run_job, job) for job in jobs]
            for future in as_completed(futures):
                task_name, record = future.result()
                task_statuses[task_name] = record
                if record["status"] == "failed":
                    failures += 1
                write_json(output_dir / "manifest.json", manifest)
    return failures


def selected_data_adapter(data: Any) -> tuple[str, dict[str, Any]]:
    """Require exactly one registered adapter under ``data``."""
    if not isinstance(data, dict):
        raise ValueError("data must be a mapping")
    unknown = sorted(set(data) - SUPPORTED_DATA_ADAPTERS)
    if unknown:
        raise ValueError(
            "data supports only adapter mappings "
            f"({', '.join(sorted(SUPPORTED_DATA_ADAPTERS))}); unsupported: {', '.join(unknown)}"
        )
    selected = [(name, settings) for name, settings in data.items() if isinstance(settings, dict) and settings.get("enabled", True)]
    if len(selected) != 1:
        raise ValueError("data must contain exactly one enabled adapter mapping")
    name, settings = selected[0]
    if not isinstance(settings, dict):
        raise ValueError(f"data.{name} must be a mapping")
    return name, settings


def enabled_callers(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return enabled callers, accepting legacy ``methods`` YAML temporarily."""
    callers = config.get("callers", config.get("methods"))
    if not isinstance(callers, dict):
        raise ValueError("config.callers must be a mapping")
    selected: dict[str, dict[str, Any]] = {}
    for name, settings in callers.items():
        implementation = settings.get("implementation", name) if isinstance(settings, dict) else name
        if not isinstance(settings, dict):
            raise ValueError(f"callers.{name} must be a mapping")
        try:
            canonical_implementation = normalize_caller_name(implementation)
        except ValueError as error:
            raise ValueError(str(error)) from error
        if settings.get("enabled", True):
            timeout = settings.get("caller_timeout_seconds", 1800)
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ValueError(f"callers.{name}.caller_timeout_seconds must be a positive number")
            if canonical_implementation == "sctype":
                allowed = {"implementation", "enabled", "environment", "caller_timeout_seconds", "database_file", "database_tissue", "cell_block_size"}
                if set(settings) - allowed:
                    raise ValueError(f"Unsupported ScType settings: {sorted(set(settings) - allowed)}")
                if settings.get("environment") != "sctype":
                    raise ValueError("ScType environment must be 'sctype'")
                if not isinstance(settings.get("database_file"), str) or not settings["database_file"].strip():
                    raise ValueError("ScType database_file is required")
                if settings.get("database_tissue") != "All":
                    raise ValueError("ScType database_tissue must be 'All' for the pinned benchmark protocol")
                block_size = settings.get("cell_block_size", 2000)
                if type(block_size) is not int or block_size < 1:
                    raise ValueError("ScType cell_block_size must be a positive integer")
            if canonical_implementation == "singler":
                allowed = {"implementation", "enabled", "environment", "caller_timeout_seconds", "hpca_rds", "num_workers", "save_fullpred"}
                if set(settings) - allowed:
                    raise ValueError(f"Unsupported SingleR settings: {sorted(set(settings) - allowed)}")
                if settings.get("environment") != "singler":
                    raise ValueError("SingleR environment must be 'singler'")
                if not isinstance(settings.get("hpca_rds"), str) or not settings["hpca_rds"].strip():
                    raise ValueError("SingleR hpca_rds is required")
                workers = settings.get("num_workers", 1)
                if type(workers) is not int or workers < 1:
                    raise ValueError("SingleR num_workers must be a positive integer")
                if type(settings.get("save_fullpred", True)) is not bool:
                    raise ValueError("SingleR save_fullpred must be a boolean")
            environment = settings.get("environment")
            if not isinstance(environment, str) or not environment.strip():
                raise ValueError(f"callers.{name}.environment must be a non-empty string")
            if canonical_implementation in {"cellclick_v1", "cellclick_v2", "cellclick_v3", "cellclick_v4", "cellclick_v5", "cellclick_v6", "cellclick_v7", "cellclick_v7_2", "cellclick_v7_3"}:
                expected_environment = caller_spec(canonical_implementation)["environment"]
                if environment != expected_environment:
                    raise ValueError(
                        f"callers.{name}.environment must be {expected_environment!r} for {canonical_implementation}"
                    )
                protocol_version = settings.get("protocol_version")
                if not isinstance(protocol_version, str) or not protocol_version.strip():
                    raise ValueError(f"callers.{name}.protocol_version must be explicitly configured")
                contract = CELLCLICK_PROTOCOL_CONTRACTS.get(protocol_version)
                if contract is None:
                    supported = ", ".join(sorted(CELLCLICK_PROTOCOL_CONTRACTS))
                    raise ValueError(
                        f"Unsupported CellClick protocol_version {protocol_version!r}; supported: {supported}"
                    )
                for field, expected in contract.items():
                    if settings.get(field) != expected:
                        raise ValueError(
                            f"callers.{name}.{field} must be {expected!r} for CellClick {protocol_version}"
                        )
                if canonical_implementation in {"cellclick_v5", "cellclick_v6"}:
                    if "max_rounds" in settings:
                        raise ValueError("CellClick V5 uses verifier_max_rounds, not max_rounds")
                    rounds = settings.get("verifier_max_rounds", 6)
                    if type(rounds) is not int or not 1 <= rounds <= 6:
                        raise ValueError("CellClick V5 verifier_max_rounds must be an integer between 1 and 6")
                    settings = {**settings, "verifier_max_rounds": rounds}
                    subtype = settings.get("subtype", True)
                    if type(subtype) is not bool:
                        raise ValueError("CellClick V5 subtype must be a boolean")
                    settings = {**settings, "subtype": subtype}
                    for field, default in (("expression_cache_max_mb", 512),
                            ("request_connect_timeout_seconds", 10), ("request_read_timeout_seconds", 600),
                            ("request_total_timeout_seconds", 610)):
                        value = settings.get(field, default)
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf"):
                            raise ValueError(f"CellClick V5 {field} must be finite and positive")
                        settings[field] = value
                    reference_max = settings.get("reference_max", 3)
                    if type(reference_max) is not int or not 0 <= reference_max <= 5:
                        raise ValueError("CellClick V5 reference_max must be an integer between 0 and 5")
                    if not isinstance(settings.get("cellmarker_reference_path"), str) or not settings["cellmarker_reference_path"].strip():
                        raise ValueError("CellClick V5 requires cellmarker_reference_path")
                    if canonical_implementation == "cellclick_v6":
                        if subtype and (not isinstance(settings.get("llmc_model"), str) or not settings["llmc_model"].strip()):
                            raise ValueError("CellClick V6 requires llmc_model when subtype is enabled")
                        cosg_top_n = settings.get("cosg_top_n", 50)
                        if type(cosg_top_n) is not int or cosg_top_n < 1:
                            raise ValueError("CellClick V6 cosg_top_n must be a positive integer")
                    top_k = settings.get("cellmarker_top_k", 5)
                    if type(top_k) is not int or top_k < 1:
                        raise ValueError("CellClick V5 cellmarker_top_k must be a positive integer")
                    mapping = settings.get("comparison_reference_by_tissue", {})
                    if not isinstance(mapping, dict):
                        raise ValueError("CellClick V5 comparison_reference_by_tissue must be a mapping")
                    for tissue, binding in mapping.items():
                        if not isinstance(tissue, str) or not tissue.strip() or not isinstance(binding, dict):
                            raise ValueError("Each CellClick V5 comparison reference must contain ref_path and h5ad_path")
                        if not all(isinstance(binding.get(key), str) and binding[key].strip() for key in ("ref_path", "h5ad_path")):
                            raise ValueError("Each CellClick V5 comparison reference must contain ref_path and h5ad_path")
                elif canonical_implementation in {"cellclick_v7", "cellclick_v7_2", "cellclick_v7_3"}:
                    if canonical_implementation == "cellclick_v7_3":
                        mapping = settings.get("comparison_reference_by_tissue", {})
                        if not isinstance(mapping, dict):
                            raise ValueError("CellClick V7.3 comparison_reference_by_tissue must be a mapping")
                        normalized_tissues = [
                            tissue.strip().casefold() for tissue in mapping
                            if isinstance(tissue, str) and tissue.strip()
                        ]
                        if len(normalized_tissues) != len(mapping) or len(normalized_tissues) != len(set(normalized_tissues)):
                            raise ValueError("CellClick V7.3 comparison reference tissue keys must be non-empty and unique ignoring case")
                        for tissue, binding in mapping.items():
                            if not isinstance(tissue, str) or not tissue.strip() or not isinstance(binding, dict):
                                raise ValueError("CellClick V7.3 comparison reference requires a tissue mapping")
                            paths = binding.get("ref_path")
                            if isinstance(paths, str):
                                paths = [paths]
                            if not isinstance(paths, list) or not paths or any(
                                not isinstance(path, str) or not path.strip() for path in paths
                            ) or not isinstance(binding.get("h5ad_path"), str) or not binding["h5ad_path"].strip():
                                raise ValueError("CellClick V7.3 comparison reference requires ref_path and h5ad_path")
                            ids = [Path(path).stem for path in paths]
                            if len(ids) != len(set(ids)):
                                raise ValueError("CellClick V7.3 comparison reference IDs must be unique")
                    for field, minimum in (("session_max_llm_calls", 2), ("finalize_reserved_calls", 1)):
                        value = settings.get(field, 8 if field == "session_max_llm_calls" else 2)
                        if type(value) is not int or value < minimum:
                            raise ValueError(f"CellClick V7 {field} must be an integer >= {minimum}")
                        settings[field] = value
                    if settings["finalize_reserved_calls"] >= settings["session_max_llm_calls"]:
                        raise ValueError("CellClick V7 finalize_reserved_calls must leave a regular call available")
                    candidates = settings.get("annotation_candidates", 5)
                    reference_max = settings.get("reference_max", 3)
                    if type(candidates) is not int or not 1 <= candidates <= 20:
                        raise ValueError("CellClick V7 annotation_candidates must be between 1 and 20")
                    if type(reference_max) is not int or not 0 <= reference_max <= candidates:
                        raise ValueError("CellClick V7 reference_max must be between 0 and annotation_candidates")
                    if reference_max and not isinstance(settings.get("cellmarker_reference_path"), str):
                        raise ValueError("CellClick V7 requires cellmarker_reference_path when reference_max > 0")
                    if type(settings.get("cosg_top_n", 50)) is not int or settings.get("cosg_top_n", 50) < 1:
                        raise ValueError("CellClick V7 cosg_top_n must be positive")
                    if type(settings.get("max_parallel_clusters", 1)) is not int or settings.get("max_parallel_clusters", 1) < 1:
                        raise ValueError("CellClick V7 max_parallel_clusters must be positive")
            if canonical_implementation in {"cellclick_v7_2", "cellclick_v7_3"}:
                policy = (
                    "cellclick-v7.2-label-embedding-v1"
                    if canonical_implementation == "cellclick_v7_2"
                    else "cellclick-v7.3-label-embedding-v1"
                )
                if settings.get("mapping_policy", policy) != policy:
                    raise ValueError(f"CellClick {canonical_implementation} mapping_policy is incompatible")
                settings = {**settings, "mapping_policy": policy}
            selected[name] = {**settings, "implementation": canonical_implementation}
    if not selected:
        raise ValueError("At least one enabled caller is required")
    return selected


enabled_methods = enabled_callers


def select_configured_methods(
    config: dict[str, Any], methods_arg: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Select enabled YAML callers for one invocation without mutating config."""
    configured = enabled_callers(config)
    if methods_arg is None:
        return configured
    if not isinstance(methods_arg, str) or not methods_arg.strip():
        raise ValueError("--methods must be a non-empty comma-separated list")
    requested = [value.strip() for value in methods_arg.split(",")]
    if any(not value for value in requested):
        raise ValueError("--methods contains an empty method name")
    if len(set(requested)) != len(requested):
        raise ValueError("--methods contains duplicate method names")

    selected: dict[str, dict[str, Any]] = {}
    for name in requested:
        if name in configured:
            selected[name] = configured[name]
            continue
        try:
            canonical = normalize_caller_name(name)
        except ValueError as error:
            raise ValueError(f"--methods names an unknown caller {name!r}") from error
        matches = [key for key, settings in configured.items() if settings["implementation"] == canonical]
        if not matches:
            raise ValueError(f"--methods caller {name!r} is not enabled in the YAML")
        if len(matches) > 1:
            raise ValueError(
                f"--methods caller {name!r} is ambiguous; use the YAML caller key: {', '.join(matches)}"
            )
        selected[matches[0]] = configured[matches[0]]
    return selected


def _config_with_methods(
    config: dict[str, Any], methods: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a shallow config view whose enabled callers are exactly selected."""
    result = dict(config)
    caller_key = "callers" if "callers" in config else "methods"
    result[caller_key] = methods
    return result


def _record_selected_methods(manifest: dict[str, Any], methods: dict[str, dict[str, Any]]) -> None:
    manifest["selected_methods"] = list(methods)
    manifest["active_callers"] = list(methods)


def _record_parallelism(manifest: dict[str, Any], max_parallel_tasks: int) -> None:
    manifest.setdefault("execution", {})["max_parallel_tasks"] = max_parallel_tasks


def excluded_source_row_ids(data: dict[str, Any]) -> set[str]:
    excluded = data.get("exclude_source_row_ids", [])
    if not isinstance(excluded, list) or any(not isinstance(row_id, str) or not row_id.strip() for row_id in excluded):
        raise ValueError("data.exclude_source_row_ids must be a list of non-empty strings")
    normalized = {row_id.strip() for row_id in excluded}
    if len(normalized) != len(excluded):
        raise ValueError("data.exclude_source_row_ids contains duplicates")
    return normalized


def exclude_query_rows(queries: list[Any], excluded_ids: set[str]) -> list[Any]:
    if not excluded_ids:
        return queries
    available_ids = {row.source_row_id for query in queries for row in query.rows}
    unknown_ids = sorted(excluded_ids - available_ids)
    if unknown_ids:
        raise ValueError(f"data.exclude_source_row_ids does not match selected rows: {', '.join(unknown_ids)}")
    filtered = []
    for query in queries:
        rows = tuple(row for row in query.rows if row.source_row_id not in excluded_ids)
        if rows:
            filtered.append(replace(query, rows=rows))
    if not filtered:
        raise ValueError("data.exclude_source_row_ids removes every selected row")
    return filtered


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


def max_parallel_tasks_from_config(config: dict[str, Any], override: int | None = None) -> int:
    """Return the bounded runner-level task concurrency."""
    value = override
    if value is None:
        run_settings = config.get("run", {})
        value = run_settings.get("max_parallel_tasks", 1) if isinstance(run_settings, dict) else 1
    if type(value) is not int or value < 1:
        raise ValueError("run.max_parallel_tasks must be a positive integer")
    return value


def read_standard_predictions(path: Path, query: Any) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Method did not create prediction TSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or set(rows[0]) != set(STANDARD_FIELDS):
        raise ValueError(f"{path} lacks standard prediction fields")
    expected_ids = [row.source_row_id for row in query.rows]
    result = {row["source_row_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path} contains duplicate source_row_id values")
    if set(result) != set(expected_ids):
        raise ValueError(f"{path} source_row_id values do not exactly match its query")
    if any(not row["prediction"].strip() for row in rows):
        raise ValueError(f"{path} contains an empty prediction")
    if any(row["task_id"] != query.task_id for row in rows):
        raise ValueError(f"{path} contains an unexpected task_id")
    forbidden = {"ground_truth", "clid", "synonyms", "broadtype", "manual annotation"}
    if forbidden & set(rows[0]):
        raise ValueError(f"{path} contains forbidden evaluation columns")
    return result


def task_payload(
    query: Any, caller: str, settings: dict[str, Any], output_dir: Path, config_path: Path,
    data_adapter: str = "tableAdapter",
) -> dict[str, Any]:
    stem = query.task_id.replace("/", "_")
    method_config = {
        key: value for key, value in settings.items()
        if key not in {"enabled", "environment", "caller_timeout_seconds", *_CALLER_SECRET_SETTINGS}
        and not key.startswith("reference_") and not key.startswith("_runner_")
    }
    implementation = normalize_caller_name(settings["implementation"])
    if implementation in {"cassia", "cellclick_v1", "cellclick_v2", "cellclick_v3", "cellclick_v4", "cellclick_v5", "cellclick_v6", "cellclick_v7", "cellclick_v7_2", "cellclick_v7_3"}:
        datasets = {row.dataset for row in query.rows}
        if len(datasets) != 1:
            raise ValueError(f"{settings['implementation']} task {query.task_id!r} contains multiple datasets")
        dataset = datasets.pop()
        species = settings.get("_runner_species_by_dataset", {}).get(dataset, settings.get("_runner_default_species"))
        if not isinstance(species, str) or not species.strip():
            raise ValueError(f"No species metadata is available for {settings['implementation']} dataset {dataset!r}")
        method_config["species"] = species.strip()
        if implementation == "cellclick_v4" and data_adapter == "expressionAdapter":
            expression_file = (output_dir / "expression_cellclick_v4" / f"{stem}.h5ad").resolve()
            method_config["expression_file"] = str(expression_file)
            method_config["expression_manifest"] = str(expression_file.with_suffix(".manifest.json"))
        if implementation in {"cellclick_v5", "cellclick_v6"} and data_adapter == "expressionAdapter":
            expression_file = (output_dir / f"expression_{implementation}" / f"{stem}.h5ad").resolve()
            method_config["expression_file"] = str(expression_file)
            method_config["expression_manifest"] = str(expression_file.with_suffix(".manifest.json"))
            cellmarker_path = resolve_path(settings["cellmarker_reference_path"], config_path).resolve()
            if not cellmarker_path.is_file():
                raise FileNotFoundError(f"CellClick V5 cellmarker_reference_path not found: {cellmarker_path}")
            method_config["cellmarker_reference_path"] = str(cellmarker_path)
            method_config["cellmarker_reference_sha256"] = file_digest(cellmarker_path)
            comparison_paths: dict[str, dict[str, str]] = {}
            comparison_hashes: dict[str, str] = {}
            manifest = json.loads(Path(method_config["expression_manifest"]).read_text(encoding="utf-8"))
            source_h5ad = Path(str(manifest.get("source_h5ad_path", ""))).resolve()
            for tissue, binding in settings.get("comparison_reference_by_tissue", {}).items():
                ref_path = resolve_path(binding["ref_path"], config_path).resolve()
                h5ad_path = resolve_path(binding["h5ad_path"], config_path).resolve()
                if not ref_path.is_file():
                    raise FileNotFoundError(f"CellClick V5 comparison reference not found for {tissue!r}: {ref_path}")
                if tissue == query.rows[0].tissue and h5ad_path == source_h5ad:
                    comparison_paths[tissue] = {"ref_path": str(ref_path), "h5ad_path": str(h5ad_path)}
                    comparison_hashes[tissue] = file_digest(ref_path)
            if implementation in {"cellclick_v5", "cellclick_v6"}:
                method_config["comparison_reference_by_tissue"] = comparison_paths
                method_config["comparison_reference_sha256_by_tissue"] = comparison_hashes
        elif implementation in {"cellclick_v7", "cellclick_v7_2", "cellclick_v7_3"} and data_adapter == "expressionAdapter":
            expression_file = (output_dir / f"expression_{implementation}" / f"{stem}.h5ad").resolve()
            method_config["expression_file"] = str(expression_file)
            method_config["expression_manifest"] = str(expression_file.with_suffix(".manifest.json"))
            ontology_path = resolve_path(settings.get("ontology_path", "../evaluator/ref_data/cl.obo"), config_path).resolve()
            if not ontology_path.is_file():
                raise FileNotFoundError(f"CellClick V7 ontology_path not found: {ontology_path}")
            method_config["ontology_path"] = str(ontology_path)
            method_config["ontology_sha256"] = file_digest(ontology_path)
            if settings.get("cellmarker_reference_path"):
                cellmarker_path = resolve_path(settings["cellmarker_reference_path"], config_path).resolve()
                if not cellmarker_path.is_file():
                    raise FileNotFoundError(f"CellClick V7 cellmarker_reference_path not found: {cellmarker_path}")
                method_config["cellmarker_reference_path"] = str(cellmarker_path)
                method_config["cellmarker_reference_sha256"] = file_digest(cellmarker_path)
            comparison_paths = {}
            expression_manifest = json.loads(Path(method_config["expression_manifest"]).read_text(encoding="utf-8"))
            source_h5ad = Path(str(expression_manifest.get("source_h5ad_path", ""))).resolve()
            query_tissue = query.rows[0].tissue
            for tissue, binding in settings.get("comparison_reference_by_tissue", {}).items():
                h5ad_path = resolve_path(binding["h5ad_path"], config_path).resolve()
                if tissue.strip().casefold() == query_tissue.strip().casefold():
                    if h5ad_path != source_h5ad:
                        raise ValueError(f"CellClick V7 comparison reference source binding mismatch for {tissue!r}")
                    if implementation == "cellclick_v7_3" and isinstance(binding["ref_path"], list):
                        references = []
                        for path_value in binding["ref_path"]:
                            ref_path = resolve_path(path_value, config_path).resolve()
                            if not ref_path.is_file():
                                raise FileNotFoundError(f"CellClick V7.3 comparison reference not found: {ref_path}")
                            reference = json.loads(ref_path.read_text(encoding="utf-8"))
                            if not isinstance(reference, dict) or not reference or any(
                                not isinstance(label, str) or not label.strip() or not isinstance(genes, list)
                                or any(not isinstance(gene, str) or not gene.strip() for gene in genes)
                                for label, genes in reference.items()
                            ):
                                raise ValueError(f"CellClick V7.3 comparison reference has invalid marker lists: {ref_path}")
                            references.append({"reference_id": ref_path.stem, "ref_path": str(ref_path),
                                               "sha256": file_digest(ref_path)})
                        comparison_paths[query_tissue] = {"references": references, "h5ad_path": str(h5ad_path)}
                    else:
                        ref_path = resolve_path(binding["ref_path"], config_path).resolve()
                        if not ref_path.is_file():
                            raise FileNotFoundError(f"CellClick V7 comparison reference not found: {ref_path}")
                        comparison_paths[query_tissue] = {"ref_path": str(ref_path), "h5ad_path": str(h5ad_path),
                                                    "sha256": file_digest(ref_path)}
            method_config["comparison_reference_by_tissue"] = comparison_paths
        if implementation == "cassia" and data_adapter == "expressionAdapter":
            marker_file = output_dir / "expression_cassia" / f"{stem}.csv"
            method_config["marker_file"] = str(marker_file.resolve())
            method_config["marker_scope"] = "all_genes_v1"
    elif implementation == "celltypeagent" and data_adapter == "expressionAdapter":
        method_config["expression_file"] = str(
            (output_dir / "expression_celltypeagent" / f"{stem}.csv").resolve()
        )
    elif implementation == "singler" and data_adapter == "expressionAdapter":
        hpca = resolve_path(settings["hpca_rds"], config_path).resolve()
        method_config["hpca_rds"] = str(hpca)
        method_config["hpca_sha256"] = file_digest(hpca)
        method_config["expression_file"] = str(
            (output_dir / "expression_singler" / f"{stem}.h5").resolve()
        )
    elif implementation == "sctype" and data_adapter == "expressionAdapter":
        database = resolve_path(settings["database_file"], config_path).resolve()
        if not database.is_file():
            raise FileNotFoundError(f"ScType database_file not found: {database}")
        method_config["database_file"] = str(database)
        method_config["database_sha256"] = file_digest(database)
        method_config["cell_block_size"] = settings.get("cell_block_size", 2000)
        method_config["expression_file"] = str(
            (output_dir / "expression_sctype" / f"{stem}.h5").resolve()
        )
    return {
        "task_id": query.task_id,
        "config_dir": str(config_path.parent.resolve()),
        "query_tsv": str((output_dir / "inputs" / f"{stem}.tsv").resolve()),
        "output_tsv": str((output_dir / "predictions" / caller / f"{stem}.tsv").resolve()),
        "work_dir": str((output_dir / "work" / caller / stem).resolve()),
        "tissue_context": query.tissue_context,
        "method": caller,
        "caller": settings["implementation"],
        "data_adapter": data_adapter,
        "method_config": method_config,
    }


def validate_method_expression_export(query: Any, implementation: str, output_dir: Path) -> Path:
    """Ensure the expression stage produced the caller's declared artifact."""
    canonical_implementation = normalize_caller_name(implementation)
    suffix = ".h5" if canonical_implementation in {"singler", "sctype"} else (".h5ad" if canonical_implementation in {"cellclick_v4", "cellclick_v5", "cellclick_v6", "cellclick_v7", "cellclick_v7_2", "cellclick_v7_3"} else ".csv")
    path = output_dir / f"expression_{canonical_implementation}" / f"{query.task_id.replace('/', '_')}{suffix}"
    if not path.is_file():
        raise FileNotFoundError(f"Expression export is missing for {implementation} task {query.task_id}: {path}")
    if canonical_implementation == "singler":
        _validate_singler_expression_export(path)
        return path
    if canonical_implementation == "sctype":
        _validate_sctype_expression_export(path)
        return path
    if canonical_implementation in {"cellclick_v4", "cellclick_v5", "cellclick_v6", "cellclick_v7", "cellclick_v7_2", "cellclick_v7_3"}:
        _validate_cellclick_v4_expression_export(path)
        return path
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    expected = {
        "anndict": ("source_row_id",),
        "gptcelltype": ("source_row_id",),
        "cellclick_v1": ("source_row_id",),
        "cellclick_v2": ("source_row_id",),
        "cellclick_v3": ("source_row_id",),
        "cassia": ("cluster", "gene", "avg_log2FC", "p_val_adj", "pct.1", "pct.2", "p_val"),
        "celltypeagent": (
            "Tissue", "Cell Type", "Cell Count", "Tissue Composition", "Gene Symbol",
            "Expression", "Expression, Scaled", "Number of Cells Expressing Genes",
        ),
    }.get(canonical_implementation)
    if expected is None:
        raise ValueError(f"No expression export contract is registered for caller {implementation!r}")
    if tuple(header or ()) != expected:
        raise ValueError(f"Expression export has unexpected columns: {path}")
    return path


def _validate_singler_expression_export(path: Path) -> None:
    """Reject missing or non-HDF5 artifacts without adding h5py to the parent runner."""
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89HDF\r\n\x1a\n":
            raise ValueError(f"SingleR expression export is not an HDF5 file: {path}")


def _validate_sctype_expression_export(path: Path) -> None:
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89HDF\r\n\x1a\n":
            raise ValueError(f"ScType expression export is not an HDF5 file: {path}")


def _validate_cellclick_v4_expression_export(path: Path) -> None:
    """Check the V4 binding without importing AnnData in the parent process."""
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89HDF\r\n\x1a\n":
            raise ValueError(f"CellClick V4 expression export is not an HDF5 file: {path}")
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CellClick V4 expression manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"CellClick V4 expression manifest is invalid JSON: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"CellClick V4 expression manifest must be an object: {manifest_path}")
    if manifest.get("schema_version") != "cellclick.expression.2":
        raise ValueError(f"CellClick V4 expression manifest has an unsupported schema: {manifest_path}")
    if manifest.get("h5ad_path") != str(path.resolve()):
        raise ValueError(f"CellClick V4 expression manifest does not bind to its h5ad: {manifest_path}")
    if manifest.get("groupby") != "cellclick_group_id":
        raise ValueError(f"CellClick V4 expression manifest has an unexpected group key: {manifest_path}")
    if manifest.get("statistics_layer") != "normalization+log1p":
        raise ValueError(f"CellClick V4 expression manifest has an unexpected statistics layer: {manifest_path}")


def prepare_table_run(config: dict[str, Any], config_path: Path) -> tuple[Any, list[Any], dict[str, dict[str, Any]]]:
    _evaluation_types(config)
    adapter_name, data = selected_data_adapter(config.get("data"))
    if not isinstance(data.get("marker_adapter_config"), str) or not data["marker_adapter_config"].strip():
        raise ValueError("data.tableAdapter.marker_adapter_config is required")
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from adapter.marker.marker import MarkerTableAdapter

    adapter = MarkerTableAdapter(resolve_path(data["marker_adapter_config"], config_path))
    queries = select_queries(adapter, data.get("selection", []))
    queries = exclude_query_rows(queries, excluded_source_row_ids(data))
    callers = configure_cassia_species(enabled_callers(config), data, config_path)
    validate_caller_adapter_compatibility(adapter_name, callers)
    return adapter, queries, callers


def validate_caller_adapter_compatibility(
    adapter_name: str, callers: dict[str, dict[str, Any]],
) -> None:
    expression_only = {"cellclick_v4", "cellclick_v5", "cellclick_v6", "cellclick_v7", "cellclick_v7_2", "cellclick_v7_3", "sctype"}
    selected = {value.get("implementation") for value in callers.values()}
    if adapter_name == "tableAdapter" and selected & expression_only:
        names = ", ".join(sorted(selected & expression_only))
        raise ValueError(f"{names} requires data.expressionAdapter; tableAdapter has no expression matrix")


def expression_query_adapter(tasks: list[tuple[str, dict[str, Any], str]]) -> Any:
    """Open only caller-safe expression TSVs plus runner-private truth maps."""
    from adapter.expression.query import ExpressionQueryAdapter

    exports = [task["annotation_export"] for _, task, _ in tasks if isinstance(task.get("annotation_export"), dict)]
    if len(exports) != len(tasks):
        raise ValueError("Expression annotation tasks lack annotation exports")
    return ExpressionQueryAdapter(exports)


def initialize_output(output_dir: Path, digest: str, resume: bool, *, force: bool = False) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not (resume or force):
            raise FileExistsError(f"Output directory already exists: {output_dir}; use --resume to continue")
        if not manifest_path.is_file():
            raise FileExistsError(f"Cannot resume {output_dir}: manifest.json is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Cannot resume {output_dir}: manifest.json must contain a mapping")
        history = manifest.setdefault("config_history", [])
        previous_digest = manifest.get("current_config_digest", manifest.get("config_digest"))
        if isinstance(previous_digest, str) and previous_digest and previous_digest not in history:
            history.append(previous_digest)
        if digest not in history:
            history.append(digest)
        manifest["schema_version"] = 2
        manifest["current_config_digest"] = digest
        # Keep the legacy field for readers that have not yet migrated.
        manifest["config_digest"] = digest
        manifest.setdefault("tasks", {})
        write_json(manifest_path, manifest)
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "config_digest": digest,
        "current_config_digest": digest,
        "config_history": [digest],
        "tasks": {},
    }
    write_json(manifest_path, manifest)
    return manifest


def caller_environment(settings: dict[str, Any]) -> dict[str, str]:
    """Build a child-only OpenAI environment from non-secret YAML settings."""
    if "api_key" in settings:
        raise ValueError("API keys must not be stored in YAML; use callers.<name>.api_key_env instead")
    environment = os.environ.copy()
    api_key_env = settings.get("api_key_env")
    if api_key_env is None and normalize_caller_name(str(settings.get("implementation", ""))) == "cassia":
        api_key_env = "OPENAI_API_KEY_LLM"
    if api_key_env is not None:
        if not isinstance(api_key_env, str) or not _ENVIRONMENT_VARIABLE.fullmatch(api_key_env):
            raise ValueError("callers.<name>.api_key_env must be an environment variable name")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Configured API key environment variable {api_key_env!r} is not set")
        environment["OPENAI_API_KEY"] = api_key

    base_url = settings.get("base_url")
    if base_url is not None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("callers.<name>.base_url must be a non-empty URL string")
        environment["OPENAI_BASE_URL"] = base_url.strip().rstrip("/")
        environment.pop("OPENAI_API_BASE", None)
    return environment


def invoke_caller(caller: str, settings: dict[str, Any], task_path: Path, log_path: Path, conda_executable: str) -> None:
    command = [
        conda_executable,
        "run",
        "--no-capture-output",
        "-n",
        settings["environment"],
        "python",
        "-m", "caller.dispatch", "--task",
        str(task_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = settings.get("caller_timeout_seconds", 1800)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=REPOSITORY_ROOT, env=caller_environment(settings),
            stdout=log, stderr=subprocess.STDOUT, text=True,
            start_new_session=(os.name == "posix"),
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            # conda run may spawn a Python child; terminate the whole process
            # group so a timed-out caller cannot survive the runner.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            raise RuntimeError(
                f"{caller} timed out after {timeout:g} seconds; see {log_path}"
            ) from error
    if returncode:
        raise RuntimeError(f"{caller} failed with exit code {returncode}; see {log_path}")


def read_protocol_manifest(work_dir: Path, *, required: bool = False) -> dict[str, Any] | None:
    """Read a caller-written, label-free protocol contract manifest."""
    path = work_dir / "protocol_manifest.json"
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Caller did not create protocol manifest: {path}")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed_fields = _PROTOCOL_MANIFEST_FIELDS | {"rag"}
    if not isinstance(payload, dict) or not set(payload).issubset(allowed_fields) or not _PROTOCOL_MANIFEST_FIELDS.issubset(payload):
        raise ValueError(f"Protocol manifest has unexpected fields: {path}")
    if any(not isinstance(payload[field], str) or not payload[field].strip() for field in _PROTOCOL_MANIFEST_FIELDS):
        raise ValueError(f"Protocol manifest contains an empty field: {path}")
    rag = payload.get("rag")
    if rag is not None:
        if (
            not isinstance(rag, dict)
            or set(rag) != _RAG_MANIFEST_FIELDS
            or not isinstance(rag["reference_path"], str)
            or not rag["reference_path"].strip()
            or not isinstance(rag["reference_sha256"], str)
            or len(rag["reference_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in rag["reference_sha256"].lower())
            or not isinstance(rag["top_k"], int)
            or isinstance(rag["top_k"], bool)
            or rag["top_k"] < 1
        ):
            raise ValueError(f"Protocol manifest contains an invalid rag contract: {path}")
    result = {field: payload[field] for field in sorted(_PROTOCOL_MANIFEST_FIELDS)}
    if rag is not None:
        result["rag"] = {field: rag[field] for field in sorted(_RAG_MANIFEST_FIELDS)}
    return result


invoke_method = invoke_caller


def score_tables(
    adapter: Any, queries: list[Any], methods: dict[str, dict[str, Any]], output_dir: Path,
    scorer: Any | None = None, output_namespace: str | None = None,
) -> None:
    if scorer is None:
        from evaluator.scorer.celltypegpt import CellTypeGPTScorer
        scorer = CellTypeGPTScorer()
    prediction_maps: dict[str, dict[str, dict[str, str]]] = {}
    decision_statuses: dict[str, dict[str, str]] = {}
    for method in methods:
        collected: dict[str, dict[str, str]] = {}
        for query in queries:
            path = output_dir / "predictions" / method / f"{query.task_id.replace('/', '_')}.tsv"
            try:
                collected.update(read_standard_predictions(path, query))
            except (FileNotFoundError, ValueError):
                continue
        prediction_maps[method] = collected
        statuses: dict[str, str] = {}
        metric_paths = [output_dir / f"{method}_metrics.jsonl"]
        if method.startswith("cellclick"):
            metric_paths.append(output_dir / "cellclick_metrics.jsonl")
        for metric_path in metric_paths:
            if not metric_path.is_file():
                continue
            with metric_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        metric = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    source_row_id = metric.get("source_row_id")
                    status = metric.get("decision_status", metric.get("status"))
                    if isinstance(source_row_id, str) and isinstance(status, str):
                        statuses[source_row_id] = status
        decision_statuses[method] = statuses

    unresolved_labels: set[str] = set()
    is_label_mapping_scorer = hasattr(scorer, "unresolved_labels")
    if is_label_mapping_scorer:
        values: list[object] = []
        for method in methods:
            values.extend(record["prediction"] for record in prediction_maps[method].values())
        for query in queries:
            values.extend(record["ground_truth"] for record in adapter.evaluation_rows(query))
        if hasattr(scorer, "ensure_mapping"):
            unresolved_labels.update(scorer.ensure_mapping(values))
        else:
            unresolved_labels.update(scorer.unresolved_labels(values))

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
            row_unresolved: set[str] = set()
            if is_label_mapping_scorer:
                for value in (record["ground_truth"],):
                    if (isinstance(value, str) and value.strip() in unresolved_labels) or (
                        isinstance(value, str) and scorer.resolve_id(value) is None
                    ):
                        row_unresolved.add(value.strip())
            truth_unresolved = bool(row_unresolved)
            for method in methods:
                prediction = prediction_maps[method].get(query_row.source_row_id)
                record[f"{method}_prediction"] = prediction["prediction"] if prediction else ""
                prediction_label = record[f"{method}_prediction"]
                if is_label_mapping_scorer and (
                    prediction_label in unresolved_labels or scorer.resolve_id(prediction_label) is None
                ):
                    row_unresolved.add(prediction_label)
                if not prediction:
                    score = None
                elif is_label_mapping_scorer:
                    score = scorer.score_one(prediction_label, record["ground_truth"])
                    if score is None:
                        score = 0.0
                else:
                    score = scorer.score_one(prediction_label, record["ground_truth"])
                record[f"{method}_evaluate_one"] = "" if score is None else score
            if is_label_mapping_scorer:
                record["unresolved_labels"] = "; ".join(sorted(row_unresolved, key=str.casefold))
            rows.append(record)
            by_dataset[query_row.dataset].append(record)

    result_dir = output_dir if output_namespace is None else output_dir / "evaluation" / output_namespace
    result_dir.mkdir(parents=True, exist_ok=True)
    one_fields = ["dataset", "tissue", "source_row_id", "ground_truth"] + [
        field for method in methods for field in (f"{method}_prediction", f"{method}_evaluate_one")
    ]
    if is_label_mapping_scorer:
        one_fields.append("unresolved_labels")
    with (result_dir / "evaluation_one.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=one_fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows: list[dict[str, Any]] = []
    for dataset in sorted(by_dataset):
        aggregate: dict[str, Any] = {"dataset": dataset}
        dataset_rows = by_dataset[dataset]
        dataset_unresolved_labels = {
            label
            for row in dataset_rows
            for label in str(row.get("unresolved_labels", "")).split("; ")
            if label
        }
        for method in methods:
            predictions = [row[f"{method}_prediction"] for row in dataset_rows]
            if any(not prediction for prediction in predictions):
                aggregate[f"{method}_evaluate"] = ""
                aggregate[f"{method}_status"] = "incomplete"
                continue
            if is_label_mapping_scorer:
                numeric_scores = [float(row[f"{method}_evaluate_one"]) for row in dataset_rows]
                value = sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
            else:
                scores = scorer.score_by_dataset(predictions, [row["ground_truth"] for row in dataset_rows], [dataset] * len(dataset_rows))
                value = scores.get(dataset)
            aggregate[f"{method}_evaluate"] = "" if value is None else value
            unresolved = bool(dataset_unresolved_labels) or any(
                decision_statuses[method].get(row["source_row_id"]) == "unresolved"
                for row in dataset_rows
            )
            if value is None:
                aggregate[f"{method}_status"] = "unscorable"
            else:
                aggregate[f"{method}_status"] = "review" if unresolved else "ok"
        if is_label_mapping_scorer:
            aggregate["unresolved_labels"] = "; ".join(sorted(dataset_unresolved_labels, key=str.casefold))
        aggregate_rows.append(aggregate)
    aggregate_fields = ["dataset"] + [field for method in methods for field in (f"{method}_evaluate", f"{method}_status")]
    if is_label_mapping_scorer:
        aggregate_fields.append("unresolved_labels")
    with (result_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)


def collect_validated_predictions(queries: list[Any], method: str, output_dir: Path) -> dict[str, dict[str, str]]:
    """Read one method's task outputs after enforcing the shared contract."""
    collected: dict[str, dict[str, str]] = {}
    for query in queries:
        path = output_dir / "predictions" / method / f"{query.task_id.replace('/', '_')}.tsv"
        collected.update(read_standard_predictions(path, query))
    return collected


def write_combined_predictions(
    queries: list[Any], method: str, output_dir: Path,
) -> Path:
    """Write one label-free, normalized prediction table for a completed method."""
    predictions = collect_validated_predictions(queries, method, output_dir)
    path = output_dir / "predictions" / f"{method}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STANDARD_FIELDS, delimiter="\t")
        writer.writeheader()
        for query in queries:
            for row in query.rows:
                writer.writerow(predictions[row.source_row_id])
    return path


def execute_tasks(
    adapter: Any,
    queries: list[Any],
    methods: dict[str, dict[str, Any]],
    output_dir: Path,
    config_path: Path,
    conda_executable: str,
    manifest: dict[str, Any],
    *,
    resume: bool = False, force: bool = False, data_adapter: str = "tableAdapter",
    max_parallel_tasks: int = 1,
) -> int:
    """Execute callers and record every task result using the normal run contract."""
    if type(max_parallel_tasks) is not int or max_parallel_tasks < 1:
        raise ValueError("max_parallel_tasks must be a positive integer")
    failures = 0
    jobs: list[dict[str, Any]] = []
    for query in queries:
        input_path = output_dir / "inputs" / f"{query.task_id.replace('/', '_')}.tsv"
        previous_query_digest = optional_file_digest(input_path)
        adapter.write_query_tsv(query, input_path)
        query_digest = file_digest(input_path)
        for method, settings in methods.items():
            expression_export_path = None
            if data_adapter == "expressionAdapter":
                expression_export_path = validate_method_expression_export(
                    query, settings["implementation"], output_dir,
                )
            expression_export_digest = optional_file_digest(expression_export_path)
            upstream_fingerprint = None
            if data_adapter == "expressionAdapter":
                expression_record = manifest.get("expression_adapter", {}).get("tasks", {}).get(query.task_id, {})
                if isinstance(expression_record, dict):
                    upstream_fingerprint = expression_record.get("fingerprint")
            key = f"{method}:{query.task_id}"
            task = task_payload(query, method, settings, output_dir, config_path, data_adapter)
            task_path = output_dir / "tasks" / method / f"{query.task_id.replace('/', '_')}.json"
            output_path = Path(task["output_tsv"])
            previous = manifest["tasks"].get(key, {})
            fingerprint = caller_task_fingerprint(
                task,
                environment=str(settings["environment"]),
                base_url=settings.get("base_url"),
                query_digest=query_digest,
                expression_export_digest=expression_export_digest,
                upstream_fingerprint=upstream_fingerprint,
            )
            fingerprint_matches = previous.get("fingerprint") == fingerprint
            migrated_from_v1 = False
            if (
                resume and not force and previous.get("status") == "success"
                and not previous.get("fingerprint") and previous_query_digest is not None
            ):
                stored_task = _load_json_mapping(task_path)
                if stored_task is not None:
                    old_fingerprint = legacy_caller_task_fingerprint(
                        stored_task,
                        query_digest=previous_query_digest,
                        expression_export_digest=expression_export_digest,
                    )
                    new_fingerprint = legacy_caller_task_fingerprint(
                        task,
                        query_digest=query_digest,
                        expression_export_digest=expression_export_digest,
                    )
                    fingerprint_matches = old_fingerprint == new_fingerprint
                    migrated_from_v1 = fingerprint_matches
            if resume and not force and previous.get("status") == "success":
                try:
                    if not fingerprint_matches:
                        raise ValueError("semantic fingerprint changed")
                    read_standard_predictions(output_path, query)
                    recorded_digest = previous.get("output_sha256")
                    if isinstance(recorded_digest, str) and file_digest(output_path) != recorded_digest:
                        raise ValueError("prediction digest changed")
                    protocol_manifest = read_protocol_manifest(
                        Path(task["work_dir"]),
                        required=bool(caller_spec(settings["implementation"]).get("requires_protocol_manifest")),
                    )
                    previous.update({
                        "fingerprint": fingerprint,
                        "output_sha256": file_digest(output_path),
                        "active_in_current_config": True,
                    })
                    if protocol_manifest is not None:
                        previous["protocol"] = protocol_manifest
                    if migrated_from_v1:
                        previous["migrated_from_schema_v1"] = True
                    write_json(output_dir / "manifest.json", manifest)
                    print(f"[REUSE] caller {key}: semantic fingerprint matched")
                    continue
                except (FileNotFoundError, ValueError):
                    pass
            reason = "forced" if force else "new, changed, or invalid task"
            print(f"[RUN]   caller {key}: {reason}")
            attempt = int(previous.get("attempt", 0)) + 1
            task["attempt"] = attempt
            write_json(task_path, task)
            jobs.append({
                "key": key, "method": method, "settings": settings, "task": task,
                "task_path": task_path, "output_path": output_path, "query": query,
                "fingerprint": fingerprint, "attempt": attempt, "forced": force,
            })

    def run_job(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        key = job["key"]
        method = job["method"]
        settings = job["settings"]
        task = job["task"]
        task_path = job["task_path"]
        output_path = job["output_path"]
        query = job["query"]
        try:
            # A failed replacement attempt must not leave a stale result
            # available to the evaluation stage.
            output_path.unlink(missing_ok=True)
            invoke_caller(
                method, settings, task_path,
                output_dir / "logs" / method / f"{query.task_id}.log", conda_executable,
            )
            read_standard_predictions(output_path, query)
            task_record: dict[str, Any] = {
                "status": "success", "output": str(output_path.relative_to(output_dir)),
                "fingerprint": job["fingerprint"],
                "output_sha256": file_digest(output_path),
                "attempt": job["attempt"],
                "forced": job["forced"],
                "active_in_current_config": True,
            }
            protocol_manifest = read_protocol_manifest(
                Path(task["work_dir"]),
                required=bool(caller_spec(settings["implementation"]).get("requires_protocol_manifest")),
            )
            if protocol_manifest is not None:
                task_record["protocol"] = protocol_manifest
            if settings["implementation"] == "celltypeagent":
                task_record["reference_overlap_risk"] = "not excluded"
        except Exception as error:
            task_record = {
                "status": "failed", "error": str(error),
                "fingerprint": job["fingerprint"],
                "attempt": job["attempt"],
                "forced": job["forced"],
                "active_in_current_config": True,
            }
        return key, task_record

    if jobs:
        with ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            futures = [executor.submit(run_job, job) for job in jobs]
            for future in as_completed(futures):
                key, task_record = future.result()
                manifest["tasks"][key] = task_record
                if task_record["status"] == "failed":
                    failures += 1
                write_json(output_dir / "manifest.json", manifest)
    return failures


def test_output_dir(config: dict[str, Any], config_path: Path) -> Path:
    run_settings = config.get("run", {})
    output_setting = run_settings.get("output_dir") if isinstance(run_settings, dict) else None
    if not output_setting:
        raise ValueError("run.output_dir is required for --mode test")
    output_dir = resolve_path(output_setting, config_path)
    return output_dir.with_name(f"{output_dir.name}_test")


def test_reference_files(values: list[str] | None, callers: dict[str, dict[str, Any]]) -> dict[str, str]:
    if not values:
        raise ValueError("--reference-file is required with --mode test")
    if len(callers) == 1 and len(values) == 1 and "=" not in values[0]:
        return {next(iter(callers)): str(Path(values[0]).expanduser().resolve())}
    references: dict[str, str] = {}
    for value in values:
        caller, separator, path = value.partition("=")
        if not separator or not caller or not path:
            raise ValueError("Use --reference-file CALLER=PATH when testing multiple callers")
        if caller not in callers:
            raise ValueError(f"--reference-file names an unselected caller: {caller}")
        if caller in references:
            raise ValueError(f"--reference-file was supplied more than once for {caller}")
        references[caller] = str(Path(path).expanduser().resolve())
    missing = sorted(set(callers) - set(references))
    if missing:
        raise ValueError(f"--reference-file is missing for callers: {', '.join(missing)}")
    return references


def run_test(
    config_path: Path, reference_files: list[str] | None = None, *, resume: bool = False, force: bool = False,
    max_parallel_tasks: int | None = None,
) -> int:
    """Compare method-specific reproduction inputs with user-supplied results."""
    import importlib

    config = load_config(config_path)
    parallel_tasks = max_parallel_tasks_from_config(config, max_parallel_tasks)
    output_dir = test_output_dir(config, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Reproduction modules decide their execution protocol. GPTCelltype test
    # mode deliberately invokes its configured caller in the normal Conda env.
    # Expression-backed tests must first materialize their label-free marker
    # exports in the declared expression environment.
    data = config.get("data")
    adapter_name, adapter_settings = selected_data_adapter(data)
    callers = configure_cassia_species(enabled_callers(config), adapter_settings, config_path)
    if adapter_name == "tableAdapter":
        adapter, queries, _ = prepare_table_run(config, config_path)
    else:
        expression_tasks = expression_task_configs(
            data, config_path, skip_marker_gene=None, output_dir=output_dir,
            implementations={settings["implementation"] for settings in callers.values()},
        )
        assert expression_tasks is not None
        manifest = initialize_output(output_dir, config_digest(config_path), resume=resume, force=force)
        _record_parallelism(manifest, parallel_tasks)
        conda_executable = str(config.get("run", {}).get("conda_executable", "conda"))
        expression_failures = execute_expression_tasks(
            expression_tasks, output_dir, conda_executable, manifest, resume=resume, force=force,
            max_parallel_tasks=parallel_tasks,
        )
        if expression_failures:
            raise RuntimeError(
                f"{expression_failures} ExpressionDataAdapter task(s) failed; "
                f"see {output_dir / 'manifest.json'}"
            )
        adapter = expression_query_adapter(expression_tasks)
        queries = adapter.generate_query("benchmark")
    references = test_reference_files(reference_files, callers)
    results = []
    # The expression stage has already initialized the shared output manifest.
    # Let the reproduction stage open that manifest even on a fresh run; its
    # caller task records are empty at that point and therefore nothing is
    # reused unless --resume was explicitly requested.
    reproduction_resume = resume or force or adapter_name == "expressionAdapter"
    for caller_name, settings in callers.items():
        implementation = settings["implementation"]
        module = importlib.import_module(f"runner.reproduce.{implementation}")
        results.append(module.run(
            config, adapter, queries, {caller_name: {**settings, "reference_file": references[caller_name]}},
            config_path, output_dir, resume=reproduction_resume, force=force,
            max_parallel_tasks=parallel_tasks,
        ))
    for result in results:
        print(
            f"test validated {result['method']}: "
            f"{result.get('rows', result.get('tasks', 0))} selected rows/tasks; "
            f"summary: {result['summary_file']}"
        )
    return 0


def _run_settings(config: dict[str, Any], config_path: Path) -> tuple[Path, str]:
    setting = config.get("run", {}).get("output_dir") if isinstance(config.get("run", {}), dict) else None
    if not setting:
        raise ValueError("run.output_dir is required")
    return resolve_path(setting, config_path), str(config.get("run", {}).get("conda_executable", "conda"))


def _expression_context(
    config: dict[str, Any], config_path: Path, output_dir: Path, *, skip_marker_gene: bool | None,
    selected_methods: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, list[Any], dict[str, dict[str, Any]], list[tuple[str, dict[str, Any], str]], str]:
    data = config.get("data")
    adapter_name, settings = selected_data_adapter(data)
    methods = configure_cassia_species(
        selected_methods or enabled_callers(config), settings, config_path,
    )
    if adapter_name != "expressionAdapter":
        raise ValueError("This stage requires data.expressionAdapter")
    if any(value["implementation"] == "singler" for value in methods.values()):
        if settings.get("target_sum", 10000) != 10000:
            raise ValueError("SingleR paper protocol requires target_sum=10000")
        for value in methods.values():
            if value["implementation"] == "singler":
                hpca = resolve_path(value["hpca_rds"], config_path)
                if not hpca.is_file():
                    raise FileNotFoundError(f"SingleR HPCA file not found: {hpca}")
    tasks = expression_task_configs(
        data, config_path, skip_marker_gene=skip_marker_gene, output_dir=output_dir,
        implementations={value["implementation"] for value in methods.values()},
    )
    assert tasks is not None
    adapter = expression_query_adapter(tasks)
    return adapter, [], methods, tasks, adapter_name


def run_expression_stage(
    config: dict[str, Any], config_path: Path, output_dir: Path, conda_executable: str, *, resume: bool,
    force: bool, dry_run: bool, skip_marker_gene: bool | None,
    selected_methods: dict[str, dict[str, Any]] | None = None,
    max_parallel_tasks: int = 1,
) -> int:
    _, _, methods, tasks, _ = _expression_context(
        config, config_path, output_dir, skip_marker_gene=False if force else skip_marker_gene,
        selected_methods=selected_methods,
    )
    if dry_run:
        print(f"Validated {len(tasks)} expression task(s) for methods: {', '.join(methods)}")
        return 0
    manifest = initialize_output(output_dir, config_digest(config_path), resume, force=force)
    _record_selected_methods(manifest, methods)
    _record_parallelism(manifest, max_parallel_tasks)
    write_json(output_dir / "manifest.json", manifest)
    return 1 if execute_expression_tasks(
        tasks, output_dir, conda_executable, manifest, resume=resume, force=force,
        max_parallel_tasks=max_parallel_tasks,
    ) else 0


def _call_or_evaluation_context(
    config: dict[str, Any], config_path: Path, output_dir: Path,
    *, selected_methods: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, list[Any], dict[str, dict[str, Any]], str]:
    data = config.get("data")
    adapter_name, settings = selected_data_adapter(data)
    methods = configure_cassia_species(
        selected_methods or enabled_callers(config), settings, config_path,
    )
    if adapter_name == "tableAdapter":
        adapter, queries, _ = prepare_table_run(_config_with_methods(config, methods), config_path)
        return adapter, queries, methods, adapter_name
    adapter, _, _, tasks, _ = _expression_context(
        config, config_path, output_dir, skip_marker_gene=None, selected_methods=methods,
    )
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Expression stage manifest is missing; run --mode expression first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    statuses = manifest.get("expression_adapter", {}).get("tasks", {})
    for task_name, task, _ in tasks:
        if statuses.get(task_name, {}).get("status") != "success":
            raise RuntimeError(f"Expression task is not successful: {task_name}; run --mode expression first")
        export = task.get("annotation_export", {})
        for field in ("query_tsv", "evaluation_json"):
            if not isinstance(export.get(field), str) or not Path(export[field]).is_file():
                raise FileNotFoundError(f"Expression task {task_name} lacks {field}")
    queries = adapter.generate_query("benchmark")
    for query in queries:
        for settings in methods.values():
            validate_method_expression_export(query, settings["implementation"], output_dir)
    return adapter, queries, methods, adapter_name


def _evaluation_types(config: dict[str, Any]) -> list[str]:
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a mapping")
    if "type" in evaluation and "types" in evaluation:
        raise ValueError("evaluation.type and evaluation.types are mutually exclusive")
    values = evaluation.get("types") if "types" in evaluation else [evaluation.get("type", "celltypegpt")]
    if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("evaluation.types must be a non-empty list of strings")
    if len(set(values)) != len(values):
        raise ValueError("evaluation.types contains duplicates")
    unknown = sorted(set(values) - {"celltypegpt", "cassia"})
    if unknown:
        raise ValueError(f"unsupported evaluation type(s): {', '.join(unknown)}")
    return list(values)


def _scorers(config: dict[str, Any]) -> dict[str, Any]:
    scorers: dict[str, Any] = {}
    for evaluation_type in _evaluation_types(config):
        if evaluation_type == "celltypegpt":
            from evaluator.scorer.celltypegpt import CellTypeGPTScorer
            scorers[evaluation_type] = CellTypeGPTScorer()
        elif evaluation_type == "cassia":
            from evaluator.scorer.cassia import CassiaScorer
            settings = config.get("evaluation", {}).get("cassia", {})
            if not isinstance(settings, dict):
                raise ValueError("evaluation.cassia must be a mapping")
            scorers[evaluation_type] = CassiaScorer(
                auto_extend_mapping=bool(settings.get("auto_extend_mapping", True)),
                mapping_path=settings.get("mapping_path") or None,
                ontology_path=settings.get("ontology_path"),
                embedding_model=str(settings.get("embedding_model", "text-embedding-3-small")),
                api_key_env=str(settings.get("api_key_env", "OPENAI_API_KEY_LLM")),
                branch_penalty=settings.get("branch_penalty", 3.0),
            )
    return scorers


def _scorer(config: dict[str, Any]) -> Any:
    """Backward-compatible single scorer accessor."""
    scorers = _scorers(config)
    if len(scorers) != 1:
        raise ValueError("_scorer requires exactly one evaluation type")
    return next(iter(scorers.values()))


def run_call_stage(
    config: dict[str, Any], config_path: Path, output_dir: Path, conda_executable: str, *, resume: bool,
    force: bool, dry_run: bool, selected_methods: dict[str, dict[str, Any]] | None = None,
    max_parallel_tasks: int = 1,
) -> int:
    adapter, queries, methods, adapter_name = _call_or_evaluation_context(
        config, config_path, output_dir, selected_methods=selected_methods,
    )
    if dry_run:
        print(f"Validated {len(queries)} call task group(s) for methods: {', '.join(methods)}")
        return 0
    manifest = initialize_output(output_dir, config_digest(config_path), resume, force=force)
    _record_selected_methods(manifest, methods)
    _record_parallelism(manifest, max_parallel_tasks)
    write_json(output_dir / "manifest.json", manifest)
    failures = execute_tasks(adapter, queries, methods, output_dir, config_path, conda_executable, manifest,
                             resume=resume, force=force, data_adapter=adapter_name,
                             max_parallel_tasks=max_parallel_tasks)
    if not failures:
        for method in methods:
            write_combined_predictions(queries, method, output_dir)
    return 1 if failures else 0


def run_evaluation_stage(
    config: dict[str, Any], config_path: Path, output_dir: Path, *, force: bool, dry_run: bool,
    selected_methods: dict[str, dict[str, Any]] | None = None,
) -> int:
    adapter, queries, methods, _ = _call_or_evaluation_context(
        config, config_path, output_dir, selected_methods=selected_methods,
    )
    if dry_run:
        print(f"Validated {len(queries)} evaluation task group(s) for methods: {', '.join(methods)}")
        return 0
    manifest = initialize_output(output_dir, config_digest(config_path), resume=True, force=force)
    _record_selected_methods(manifest, methods)
    active_task_keys = {f"{method}:{query.task_id}" for method in methods for query in queries}
    for key, record in manifest.get("tasks", {}).items():
        if isinstance(record, dict):
            record["active_in_current_config"] = key in active_task_keys
    prediction_digests: dict[str, dict[str, str | None]] = {}
    for method in methods:
        prediction_digests[method] = {
            query.task_id: optional_file_digest(
                output_dir / "predictions" / method / f"{query.task_id.replace('/', '_')}.tsv"
            )
            for query in queries
        }
    evaluation_types = _evaluation_types(config)
    evaluation_fingerprint = semantic_digest({
        "schema": 3,
        "evaluation_types": evaluation_types,
        "evaluation_settings": config.get("evaluation", {}),
        "queries": [
            {
                "task_id": query.task_id,
                "rows": [
                    [row.source_row_id, row.dataset, row.tissue]
                    for row in query.rows
                ],
            }
            for query in queries
        ],
        "prediction_digests": prediction_digests,
    })
    previous_evaluation = manifest.get("evaluation", {})
    reason = (
        "inputs unchanged; refreshing deterministic outputs"
        if previous_evaluation.get("fingerprint") == evaluation_fingerprint
        else "configuration, callers, or predictions changed"
    )
    print(f"[SCORE] evaluation: {reason}")
    try:
        for method in methods:
            try:
                write_combined_predictions(queries, method, output_dir)
            except (FileNotFoundError, ValueError):
                # score_tables records the affected dataset as incomplete instead
                # of dropping rows or manufacturing a prediction table.
                pass
        evaluator_results: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        for evaluator_name, scorer in _scorers(config).items():
            try:
                score_tables(adapter, queries, methods, output_dir, scorer, output_namespace=evaluator_name)
                result_dir = output_dir / "evaluation" / evaluator_name
                evaluator_results[evaluator_name] = {
                    "status": "success",
                    **({"mapping_extension": getattr(scorer, "mapping_extension", {})} if evaluator_name == "cassia" else {}),
                    "outputs": {
                        "one": {"path": str(result_dir.relative_to(output_dir) / "evaluation_one.csv"), "sha256": file_digest(result_dir / "evaluation_one.csv")},
                        "aggregate": {"path": str(result_dir.relative_to(output_dir) / "evaluation.csv"), "sha256": file_digest(result_dir / "evaluation.csv")},
                    },
                }
            except Exception as error:
                failures.append(evaluator_name)
                evaluator_results[evaluator_name] = {"status": "failed", "error": str(error)}
        manifest["evaluation"] = {
            "status": "failed" if failures else "success",
            "fingerprint": evaluation_fingerprint,
            "types": evaluation_types,
            "active_callers": list(methods),
            "evaluators": evaluator_results,
        }
        if failures:
            write_json(output_dir / "manifest.json", manifest)
            return 1
    except Exception as error:
        manifest["evaluation"] = {
            "status": "failed", "fingerprint": evaluation_fingerprint, "error": str(error),
        }
        write_json(output_dir / "manifest.json", manifest)
        raise
    write_json(output_dir / "manifest.json", manifest)
    return 0


def run(
    config_path: Path, resume: bool = False, dry_run: bool = False, mode: str = "run",
    reference_files: list[str] | None = None, skip_marker_gene: bool | None = None, force: bool = False,
    methods_arg: str | None = None, max_parallel_tasks: int | None = None,
) -> int:
    if mode not in {"expression", "call", "evaluate", "run", "test"}:
        raise ValueError("mode must be 'expression', 'call', 'evaluate', 'run', or 'test'")
    if force and skip_marker_gene:
        raise ValueError("--force and --skip-marker-gene cannot be used together")
    if mode == "test":
        if dry_run:
            raise ValueError("--dry-run is not valid with --mode test")
        if methods_arg is not None:
            raise ValueError("--methods is not supported with --mode test")
        config = load_config(config_path)
        parallel_tasks = max_parallel_tasks_from_config(config, max_parallel_tasks)
        return run_test(config_path, reference_files, resume=resume, force=force, max_parallel_tasks=parallel_tasks)
    config = load_config(config_path)
    parallel_tasks = max_parallel_tasks_from_config(config, max_parallel_tasks)
    selected_methods = select_configured_methods(config, methods_arg)
    output_dir, conda_executable = _run_settings(config, config_path)
    if mode == "expression":
        return run_expression_stage(config, config_path, output_dir, conda_executable, resume=resume,
                                    force=force, dry_run=dry_run, skip_marker_gene=skip_marker_gene,
                                    selected_methods=selected_methods, max_parallel_tasks=parallel_tasks)
    if mode == "call":
        return run_call_stage(config, config_path, output_dir, conda_executable, resume=resume, force=force,
                              dry_run=dry_run, selected_methods=selected_methods,
                              max_parallel_tasks=parallel_tasks)
    if mode == "evaluate":
        return run_evaluation_stage(config, config_path, output_dir, force=force, dry_run=dry_run,
                                    selected_methods=selected_methods)
    adapter_name, _ = selected_data_adapter(config.get("data"))
    if adapter_name == "tableAdapter":
        result = run_call_stage(config, config_path, output_dir, conda_executable, resume=resume, force=force,
                                dry_run=dry_run, selected_methods=selected_methods,
                                max_parallel_tasks=parallel_tasks)
        if result or dry_run:
            return result
        return run_evaluation_stage(config, config_path, output_dir, force=force, dry_run=False,
                                    selected_methods=selected_methods)
    result = run_expression_stage(config, config_path, output_dir, conda_executable, resume=resume,
                                  force=force, dry_run=dry_run, skip_marker_gene=skip_marker_gene,
                                  selected_methods=selected_methods, max_parallel_tasks=parallel_tasks)
    if result or dry_run:
        return result
    result = run_call_stage(config, config_path, output_dir, conda_executable, resume=True, force=force,
                            dry_run=False, selected_methods=selected_methods,
                            max_parallel_tasks=parallel_tasks)
    if result:
        return result
    return run_evaluation_stage(config, config_path, output_dir, force=force, dry_run=False,
                                selected_methods=selected_methods)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=("expression", "call", "evaluate", "run", "test"), default="run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun successful tasks in an existing output directory")
    parser.add_argument(
        "--methods",
        help="comma-separated enabled caller names to run; omitted means all enabled callers",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-parallel-tasks", type=int,
        help="maximum number of independent expression/caller tasks to run concurrently",
    )
    parser.add_argument(
        "--skip-marker-gene", action="store_true", default=None,
        help=(
            "reuse a complete parameter-matched ExpressionDataAdapter marker cache; "
            "compute and write it when either cache file is missing"
        ),
    )
    parser.add_argument(
        "--reference-file", action="append", metavar="[CALLER=]PATH",
        help="Original result file required by --mode test; repeat as CALLER=PATH for multiple callers",
    )
    args = parser.parse_args()
    try:
        raise SystemExit(run(
            args.config.resolve(), resume=args.resume, dry_run=args.dry_run, mode=args.mode,
            reference_files=args.reference_file, skip_marker_gene=args.skip_marker_gene, force=args.force,
            methods_arg=args.methods, max_parallel_tasks=args.max_parallel_tasks,
        ))
    except Exception as error:
        print(f"runner error: {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
