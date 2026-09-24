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
import subprocess
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
STANDARD_FIELDS = (
    "source_row_id",
    "dataset",
    "tissue",
    "prediction",
    "method",
    "model",
    "task_id",
)
SUPPORTED_CALLERS = frozenset({"cassia", "celltypeagent", "gptcelltype"})
SUPPORTED_DATA_ADAPTERS = frozenset({"expressionAdapter", "tableAdapter"})
# Compatibility alias for downstream code that imported the old constant.
SUPPORTED_METHODS = SUPPORTED_CALLERS
_ENVIRONMENT_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CALLER_SECRET_SETTINGS = frozenset({"api_key", "api_key_env", "base_url"})


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


def configure_cassia_species(
    callers: dict[str, dict[str, Any]], data: dict[str, Any], config_path: Path,
) -> dict[str, dict[str, Any]]:
    """Attach dataset species lookup data to CASSIA settings without serializing it."""
    if not any(settings["implementation"] == "cassia" for settings in callers.values()):
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
            if settings["implementation"] == "cassia" else settings
        )
        for name, settings in callers.items()
    }


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
                "groupby": obs_keys["cell_type"],
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
                task["annotation_export"] = {
                    "dataset": dataset_name,
                    "dataname": dataname,
                    "groupby": obs_keys["cell_type"],
                    "tissue_column": obs_keys["tissue"],
                    "task_id": task_name,
                    "tissue_context": f"{species.strip()} {dataname.replace('_', ' ')}",
                    "query_tsv": str((output_dir / "expression_queries" / f"{stem}.tsv").resolve()),
                    "evaluation_json": str((output_dir / "expression_evaluation" / f"{stem}.json").resolve()),
                }
                task["method_exports"] = {
                    implementation: str((output_dir / f"expression_{implementation}" / f"{stem}.csv").resolve())
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
    return [("expression", {"h5ad": str(h5ad), "expression": expression}, settings["environment"])]


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


def execute_expression_tasks(
    tasks: list[tuple[str, dict[str, Any], str]], output_dir: Path, conda_executable: str,
    manifest: dict[str, Any], *, resume: bool = False,
) -> int:
    """Run every configured h5ad marker task and retain per-file manifest status."""
    section = manifest.setdefault("expression_adapter", {"tasks": {}})
    task_statuses = section.setdefault("tasks", {})
    failures = 0
    for task_name, task, environment in tasks:
        previous = task_statuses.get(task_name, {})
        if resume and previous.get("status") == "success":
            # A prior manifest may predate annotation exports (or the files
            # may have been removed).  Reuse only a success whose generated
            # caller input is still present; otherwise regenerate it.
            export = task.get("annotation_export")
            query_tsv = export.get("query_tsv") if isinstance(export, dict) else None
            exports = task.get("method_exports", {})
            exports_exist = isinstance(exports, dict) and all(
                isinstance(path, str) and Path(path).is_file() for path in exports.values()
            )
            evaluation_json = export.get("evaluation_json") if isinstance(export, dict) else None
            if (
                isinstance(query_tsv, str) and Path(query_tsv).is_file()
                and isinstance(evaluation_json, str) and Path(evaluation_json).is_file()
                and exports_exist
            ):
                continue
        try:
            task_path = invoke_expression_adapter(
                task, environment, output_dir, conda_executable, task_name=task_name,
            )
            task_statuses[task_name] = {
                "status": "success", "task": task_path,
                "exports": task.get("method_exports", {}),
            }
        except Exception as error:
            failures += 1
            task_statuses[task_name] = {"status": "failed", "error": str(error)}
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
        if implementation not in SUPPORTED_CALLERS:
            raise ValueError(f"Unsupported caller {implementation!r}; supported: {', '.join(sorted(SUPPORTED_CALLERS))}")
        if not isinstance(settings, dict):
            raise ValueError(f"callers.{name} must be a mapping")
        if settings.get("enabled", True):
            environment = settings.get("environment")
            if not isinstance(environment, str) or not environment.strip():
                raise ValueError(f"callers.{name}.environment must be a non-empty string")
            selected[name] = {**settings, "implementation": implementation}
    if not selected:
        raise ValueError("At least one enabled caller is required")
    return selected


enabled_methods = enabled_callers


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
        if key not in {"enabled", "environment", *_CALLER_SECRET_SETTINGS}
        and not key.startswith("reference_") and not key.startswith("_runner_")
    }
    if settings["implementation"] == "cassia":
        datasets = {row.dataset for row in query.rows}
        if len(datasets) != 1:
            raise ValueError(f"CASSIA task {query.task_id!r} contains multiple datasets")
        dataset = datasets.pop()
        species = settings.get("_runner_species_by_dataset", {}).get(dataset, settings.get("_runner_default_species"))
        if not isinstance(species, str) or not species.strip():
            raise ValueError(f"No species metadata is available for CASSIA dataset {dataset!r}")
        method_config["species"] = species.strip()
        if data_adapter == "expressionAdapter":
            marker_file = output_dir / "expression_cassia" / f"{stem}.csv"
            method_config["marker_file"] = str(marker_file.resolve())
    elif settings["implementation"] == "celltypeagent" and data_adapter == "expressionAdapter":
        method_config["expression_file"] = str(
            (output_dir / "expression_celltypeagent" / f"{stem}.csv").resolve()
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
    path = output_dir / f"expression_{implementation}" / f"{query.task_id.replace('/', '_')}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Expression export is missing for {implementation} task {query.task_id}: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    expected = {
        "gptcelltype": ("source_row_id",),
        "cassia": ("cluster", "gene", "avg_log2FC", "p_val_adj", "pct.1", "pct.2", "p_val"),
        "celltypeagent": ("source_row_id", "gene", "mean_normalized_expression", "expressed_fraction"),
    }.get(implementation)
    if expected is None:
        raise ValueError(f"No expression export contract is registered for caller {implementation!r}")
    if tuple(header or ()) != expected:
        raise ValueError(f"Expression export has unexpected columns: {path}")
    return path


def prepare_table_run(config: dict[str, Any], config_path: Path) -> tuple[Any, list[Any], dict[str, dict[str, Any]]]:
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict) or evaluation.get("type", "celltypegpt") not in {"celltypegpt", "cassia"}:
        raise ValueError("evaluation.type must be 'celltypegpt' or 'cassia'")
    adapter_name, data = selected_data_adapter(config.get("data"))
    if not isinstance(data.get("marker_adapter_config"), str) or not data["marker_adapter_config"].strip():
        raise ValueError("data.tableAdapter.marker_adapter_config is required")
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from adapter.marker.marker import MarkerTableAdapter

    adapter = MarkerTableAdapter(resolve_path(data["marker_adapter_config"], config_path))
    queries = select_queries(adapter, data.get("selection", []))
    queries = exclude_query_rows(queries, excluded_source_row_ids(data))
    callers = configure_cassia_species(enabled_callers(config), data, config_path)
    return adapter, queries, callers


def expression_query_adapter(tasks: list[tuple[str, dict[str, Any], str]]) -> Any:
    """Open only caller-safe expression TSVs plus runner-private truth maps."""
    from adapter.expression.query import ExpressionQueryAdapter

    exports = [task["annotation_export"] for _, task, _ in tasks if isinstance(task.get("annotation_export"), dict)]
    if len(exports) != len(tasks):
        raise ValueError("Expression annotation tasks lack annotation exports")
    return ExpressionQueryAdapter(exports)


def initialize_output(output_dir: Path, digest: str, resume: bool) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"Output directory already exists: {output_dir}; use --resume to continue")
        if not manifest_path.is_file():
            raise FileExistsError(f"Cannot resume {output_dir}: manifest.json is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_digest") != digest:
            raise ValueError("Cannot resume: config content differs from the existing run")
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"config_digest": digest, "tasks": {}}
    write_json(manifest_path, manifest)
    return manifest


def caller_environment(settings: dict[str, Any]) -> dict[str, str]:
    """Build a child-only OpenAI environment from non-secret YAML settings."""
    if "api_key" in settings:
        raise ValueError("API keys must not be stored in YAML; use callers.<name>.api_key_env instead")
    environment = os.environ.copy()
    api_key_env = settings.get("api_key_env")
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
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=REPOSITORY_ROOT, env=caller_environment(settings),
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )
    if completed.returncode:
        raise RuntimeError(f"{caller} failed with exit code {completed.returncode}; see {log_path}")


invoke_method = invoke_caller


def score_tables(adapter: Any, queries: list[Any], methods: dict[str, dict[str, Any]], output_dir: Path, scorer: Any | None = None) -> None:
    if scorer is None:
        from evaluator.scorer.celltypegpt import CellTypeGPTScorer
        scorer = CellTypeGPTScorer()
    prediction_maps: dict[str, dict[str, dict[str, str]]] = {}
    for method in methods:
        collected: dict[str, dict[str, str]] = {}
        for query in queries:
            path = output_dir / "predictions" / method / f"{query.task_id.replace('/', '_')}.tsv"
            try:
                collected.update(read_standard_predictions(path, query))
            except (FileNotFoundError, ValueError):
                continue
        prediction_maps[method] = collected

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
            for method in methods:
                prediction = prediction_maps[method].get(query_row.source_row_id)
                record[f"{method}_prediction"] = prediction["prediction"] if prediction else ""
                score = scorer.score_one(record[f"{method}_prediction"], record["ground_truth"]) if prediction else None
                record[f"{method}_evaluate_one"] = "" if score is None else score
            rows.append(record)
            by_dataset[query_row.dataset].append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    one_fields = ["dataset", "tissue", "source_row_id", "ground_truth"] + [
        field for method in methods for field in (f"{method}_prediction", f"{method}_evaluate_one")
    ]
    with (output_dir / "evaluation_one.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=one_fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows: list[dict[str, Any]] = []
    for dataset in sorted(by_dataset):
        aggregate: dict[str, Any] = {"dataset": dataset}
        dataset_rows = by_dataset[dataset]
        for method in methods:
            predictions = [row[f"{method}_prediction"] for row in dataset_rows]
            if any(not prediction for prediction in predictions):
                aggregate[f"{method}_evaluate"] = ""
                aggregate[f"{method}_status"] = "incomplete"
                continue
            scores = scorer.score_by_dataset(predictions, [row["ground_truth"] for row in dataset_rows], [dataset] * len(dataset_rows))
            value = scores.get(dataset)
            aggregate[f"{method}_evaluate"] = "" if value is None else value
            aggregate[f"{method}_status"] = "ok" if value is not None else "unscorable"
        aggregate_rows.append(aggregate)
    aggregate_fields = ["dataset"] + [field for method in methods for field in (f"{method}_evaluate", f"{method}_status")]
    with (output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
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
    resume: bool = False, data_adapter: str = "tableAdapter",
) -> int:
    """Execute callers and record every task result using the normal run contract."""
    failures = 0
    for query in queries:
        input_path = output_dir / "inputs" / f"{query.task_id.replace('/', '_')}.tsv"
        adapter.write_query_tsv(query, input_path)
        for method, settings in methods.items():
            if data_adapter == "expressionAdapter":
                validate_method_expression_export(query, settings["implementation"], output_dir)
            key = f"{method}:{query.task_id}"
            task = task_payload(query, method, settings, output_dir, config_path, data_adapter)
            task_path = output_dir / "tasks" / method / f"{query.task_id.replace('/', '_')}.json"
            write_json(task_path, task)
            output_path = Path(task["output_tsv"])
            previous = manifest["tasks"].get(key, {})
            if resume and previous.get("status") == "success":
                try:
                    read_standard_predictions(output_path, query)
                    continue
                except (FileNotFoundError, ValueError):
                    pass
            try:
                invoke_caller(method, settings, task_path, output_dir / "logs" / method / f"{query.task_id}.log", conda_executable)
                read_standard_predictions(output_path, query)
                manifest["tasks"][key] = {"status": "success", "output": str(output_path.relative_to(output_dir))}
            except Exception as error:
                failures += 1
                manifest["tasks"][key] = {"status": "failed", "error": str(error)}
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
    config_path: Path, reference_files: list[str] | None = None, *, resume: bool = False,
) -> int:
    """Compare method-specific reproduction inputs with user-supplied results."""
    import importlib

    config = load_config(config_path)
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
        manifest = initialize_output(output_dir, config_digest(config_path), resume=resume)
        conda_executable = str(config.get("run", {}).get("conda_executable", "conda"))
        expression_failures = execute_expression_tasks(
            expression_tasks, output_dir, conda_executable, manifest, resume=resume,
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
    reproduction_resume = resume or adapter_name == "expressionAdapter"
    for caller_name, settings in callers.items():
        implementation = settings["implementation"]
        module = importlib.import_module(f"runner.reproduce.{implementation}")
        results.append(module.run(
            config, adapter, queries, {caller_name: {**settings, "reference_file": references[caller_name]}},
            config_path, output_dir, resume=reproduction_resume,
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
) -> tuple[Any, list[Any], dict[str, dict[str, Any]], list[tuple[str, dict[str, Any], str]], str]:
    data = config.get("data")
    adapter_name, settings = selected_data_adapter(data)
    methods = configure_cassia_species(enabled_callers(config), settings, config_path)
    if adapter_name != "expressionAdapter":
        raise ValueError("This stage requires data.expressionAdapter")
    tasks = expression_task_configs(
        data, config_path, skip_marker_gene=skip_marker_gene, output_dir=output_dir,
        implementations={value["implementation"] for value in methods.values()},
    )
    assert tasks is not None
    adapter = expression_query_adapter(tasks)
    return adapter, [], methods, tasks, adapter_name


def run_expression_stage(
    config: dict[str, Any], config_path: Path, output_dir: Path, conda_executable: str, *, resume: bool,
    dry_run: bool, skip_marker_gene: bool | None,
) -> int:
    _, _, methods, tasks, _ = _expression_context(
        config, config_path, output_dir, skip_marker_gene=skip_marker_gene,
    )
    if dry_run:
        print(f"Validated {len(tasks)} expression task(s) for methods: {', '.join(methods)}")
        return 0
    manifest = initialize_output(output_dir, config_digest(config_path), resume)
    return 1 if execute_expression_tasks(tasks, output_dir, conda_executable, manifest, resume=resume) else 0


def _call_or_evaluation_context(
    config: dict[str, Any], config_path: Path, output_dir: Path,
) -> tuple[Any, list[Any], dict[str, dict[str, Any]], str]:
    data = config.get("data")
    adapter_name, settings = selected_data_adapter(data)
    methods = configure_cassia_species(enabled_callers(config), settings, config_path)
    if adapter_name == "tableAdapter":
        adapter, queries, _ = prepare_table_run(config, config_path)
        return adapter, queries, methods, adapter_name
    adapter, _, _, tasks, _ = _expression_context(config, config_path, output_dir, skip_marker_gene=None)
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


def _scorer(config: dict[str, Any]) -> Any:
    evaluation_type = config.get("evaluation", {}).get("type", "celltypegpt")
    if evaluation_type == "celltypegpt":
        from evaluator.scorer.celltypegpt import CellTypeGPTScorer
        return CellTypeGPTScorer()
    if evaluation_type == "cassia":
        from evaluator.scorer.cassia import CassiaScorer
        return CassiaScorer()
    raise ValueError("evaluation.type must be 'celltypegpt' or 'cassia'")


def run_call_stage(
    config: dict[str, Any], config_path: Path, output_dir: Path, conda_executable: str, *, resume: bool, dry_run: bool,
) -> int:
    adapter, queries, methods, adapter_name = _call_or_evaluation_context(config, config_path, output_dir)
    if dry_run:
        print(f"Validated {len(queries)} call task group(s) for methods: {', '.join(methods)}")
        return 0
    manifest = initialize_output(output_dir, config_digest(config_path), resume)
    failures = execute_tasks(adapter, queries, methods, output_dir, config_path, conda_executable, manifest,
                             resume=resume, data_adapter=adapter_name)
    if not failures:
        for method in methods:
            write_combined_predictions(queries, method, output_dir)
    return 1 if failures else 0


def run_evaluation_stage(config: dict[str, Any], config_path: Path, output_dir: Path, *, dry_run: bool) -> int:
    adapter, queries, methods, _ = _call_or_evaluation_context(config, config_path, output_dir)
    if dry_run:
        print(f"Validated {len(queries)} evaluation task group(s) for methods: {', '.join(methods)}")
        return 0
    for method in methods:
        try:
            write_combined_predictions(queries, method, output_dir)
        except (FileNotFoundError, ValueError):
            # score_tables records the affected dataset as incomplete instead
            # of dropping rows or manufacturing a prediction table.
            pass
    score_tables(adapter, queries, methods, output_dir, _scorer(config))
    return 0


def run(
    config_path: Path, resume: bool = False, dry_run: bool = False, mode: str = "run",
    reference_files: list[str] | None = None, skip_marker_gene: bool | None = None,
) -> int:
    if mode not in {"expression", "call", "evaluate", "run", "test"}:
        raise ValueError("mode must be 'expression', 'call', 'evaluate', 'run', or 'test'")
    if mode == "test":
        if dry_run:
            raise ValueError("--dry-run is not valid with --mode test")
        return run_test(config_path, reference_files, resume=resume)
    config = load_config(config_path)
    output_dir, conda_executable = _run_settings(config, config_path)
    if mode == "expression":
        return run_expression_stage(config, config_path, output_dir, conda_executable, resume=resume,
                                    dry_run=dry_run, skip_marker_gene=skip_marker_gene)
    if mode == "call":
        return run_call_stage(config, config_path, output_dir, conda_executable, resume=resume, dry_run=dry_run)
    if mode == "evaluate":
        return run_evaluation_stage(config, config_path, output_dir, dry_run=dry_run)
    adapter_name, _ = selected_data_adapter(config.get("data"))
    if adapter_name == "tableAdapter":
        result = run_call_stage(config, config_path, output_dir, conda_executable, resume=resume, dry_run=dry_run)
        if result or dry_run:
            return result
        return run_evaluation_stage(config, config_path, output_dir, dry_run=False)
    result = run_expression_stage(config, config_path, output_dir, conda_executable, resume=resume,
                                  dry_run=dry_run, skip_marker_gene=skip_marker_gene)
    if result or dry_run:
        return result
    result = run_call_stage(config, config_path, output_dir, conda_executable, resume=True, dry_run=False)
    if result:
        return result
    return run_evaluation_stage(config, config_path, output_dir, dry_run=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=("expression", "call", "evaluate", "run", "test"), default="run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-marker-gene", action="store_true", default=None,
        help="reuse only a parameter-matched ExpressionDataAdapter marker cache",
    )
    parser.add_argument(
        "--reference-file", action="append", metavar="[CALLER=]PATH",
        help="Original result file required by --mode test; repeat as CALLER=PATH for multiple callers",
    )
    args = parser.parse_args()
    try:
        raise SystemExit(run(
            args.config.resolve(), resume=args.resume, dry_run=args.dry_run, mode=args.mode,
            reference_files=args.reference_file, skip_marker_gene=args.skip_marker_gene,
        ))
    except Exception as error:
        print(f"runner error: {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
