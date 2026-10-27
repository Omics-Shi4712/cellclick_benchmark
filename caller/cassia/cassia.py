#!/usr/bin/env python3
"""CASSIA pipeline adapter for the benchmark caller contract."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


STANDARD_FIELDS = ("source_row_id", "dataset", "tissue", "prediction", "method", "model", "task_id")
_FINAL_TAG = re.compile(r"<FINAL_ANNOTATION>\s*(.*?)\s*</FINAL_ANNOTATION>", re.IGNORECASE | re.DOTALL)
_LABEL_LINE = re.compile(
    r"(?:final\s+(?:cell\s+type|annotation)|cell\s+type)(?:\s*[*_]{1,3})?\s*(?::|\-|\bis\b)\s*([^\n.]+)",
    re.IGNORECASE,
)


def read_query(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"source_row_id", "dataset", "tissue", "marker"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Query TSV lacks required label-free columns")
    if len({row["source_row_id"] for row in rows}) != len(rows):
        raise ValueError("Query TSV has duplicate source_row_id values")
    return rows


def provider_from_environment(config: dict[str, Any]) -> str:
    provider = str(config.get("provider", "openai"))
    base_url = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "").strip()
    if base_url:
        if provider != "openai":
            raise ValueError("OPENAI_BASE_URL requires provider=openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is required with OPENAI_BASE_URL")
        os.environ["CUSTOMIZED_API_KEY"] = api_key
        # CASSIA passes custom providers directly to the OpenAI client.  Keep
        # only the API base path; passing a full endpoint makes the client
        # append /chat/completions twice on some relay versions.
        base_url = re.sub(r"/chat/completions/?$", "", base_url)
        return base_url.rstrip("/")
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required for CASSIA provider=openai")
    return provider


def _pipeline_directory(work_dir: Path) -> Path:
    candidates = [path for path in work_dir.glob("CASSIA_Pipeline_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"CASSIA pipeline output directory not found under {work_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _by_cluster(rows: list[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    result = {row.get("Cluster ID", "").strip(): row for row in rows}
    if not result or "" in result or len(result) != len(rows):
        raise ValueError(f"CASSIA output has missing or duplicate Cluster ID values: {path}")
    return result


def _boost_folder(cluster_id: str) -> str:
    return "".join(char for char in cluster_id if char.isalnum() or char in (" ", "-", "_")).strip()


def _boosted_prediction(raw_conversation: Path) -> str:
    text = raw_conversation.read_text(encoding="utf-8")
    tagged = _FINAL_TAG.findall(text)
    marker = text.upper().rfind("FINAL ANNOTATION COMPLETED")
    if not tagged and marker < 0:
        raise ValueError(f"CASSIA Boost did not produce a final annotation in {raw_conversation}")
    candidate = tagged[-1] if tagged else text[marker:]
    label = _LABEL_LINE.search(candidate)
    if label:
        return label.group(1).strip(" \t:;.-")
    raise ValueError(f"Could not extract a final Boost annotation from {raw_conversation}")


def _native_scores(scored_path: Path, expected_ids: set[str]) -> dict[str, float]:
    by_id = _by_cluster(_read_csv(scored_path), scored_path)
    if set(by_id) != expected_ids:
        raise ValueError("CASSIA scored output does not align one-to-one with the query")
    scores: dict[str, float] = {}
    for cluster_id, row in by_id.items():
        try:
            scores[cluster_id] = float(row["Score"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"CASSIA scored output has no numeric Score for {cluster_id}") from error
    return scores


def _alignment_error(kind: str, observed: set[str], expected: set[str]) -> ValueError:
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    details = []
    if missing:
        details.append(f"missing={missing}")
    if extra:
        details.append(f"extra={extra}")
    suffix = f" ({'; '.join(details)})" if details else ""
    return ValueError(f"CASSIA {kind} does not align one-to-one with the query{suffix}")


def _run_expression_pipeline(pipeline: Any, arguments: dict[str, Any], rows: list[dict[str, str]]) -> None:
    """Submit ordered query markers directly; retain full statistics for Boost.

    Re-filtering the benchmark query as a Seurat DE table can remove entire
    clusters. Use CASSIA's supported preformatted input for initial annotation.
    The full table remains the pipeline input used by Boost.
    """
    engine = importlib.import_module("CASSIA.engine.tools_function")
    original_batch = engine.runCASSIA_batch
    allowed = {
        row["source_row_id"]: {gene.strip() for gene in row["marker"].split(",") if gene.strip()}
        for row in rows
    }

    def initial_batch(*args: Any, **kwargs: Any) -> Any:
        frame = kwargs["marker"]
        clusters = frame["cluster"].astype(str)
        unknown = set(clusters).difference(allowed)
        if unknown:
            raise ValueError(f"CASSIA initial marker table contains unknown clusters: {sorted(unknown)}")
        import pandas as pd
        kwargs["marker"] = pd.DataFrame({
            "cluster": [row["source_row_id"] for row in rows],
            "markers": [row["marker"] for row in rows],
        })
        return original_batch(*args, **kwargs)

    engine.runCASSIA_batch = initial_batch
    try:
        pipeline(**arguments)
    finally:
        engine.runCASSIA_batch = original_batch


def run_task(task: dict[str, Any]) -> None:
    rows = read_query(Path(task["query_tsv"]))
    config = task.get("method_config", {})
    species = config.get("species")
    if not isinstance(species, str) or not species.strip():
        raise ValueError("CASSIA requires species injected from dataset metadata")
    import pandas as pd
    import CASSIA

    work_dir = Path(task["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    marker_file = config.get("marker_file")
    if marker_file:
        marker_frame = pd.read_csv(Path(str(marker_file)))
        required = {"cluster", "gene", "avg_log2FC", "p_val_adj", "pct.1", "pct.2", "p_val"}
        if not required.issubset(marker_frame.columns):
            raise ValueError("CASSIA marker_file lacks the required Seurat marker columns")
        marker_frame = marker_frame.loc[:, ["cluster", "gene", "avg_log2FC", "p_val_adj", "pct.1", "pct.2", "p_val"]]
        if set(marker_frame["cluster"].astype(str)) != {row["source_row_id"] for row in rows}:
            raise ValueError("CASSIA marker_file clusters do not align with the query")
    else:
        input_path = work_dir / "cassia_input.csv"
        with input_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("cluster_id", "markers"))
            writer.writeheader()
            writer.writerows({"cluster_id": row["source_row_id"], "markers": row["marker"]} for row in rows)
        marker_frame = pd.read_csv(input_path)

    provider = provider_from_environment(config)
    model = str(config.get("model", "gpt-4o-2024-08-06"))
    configured_workers = int(config.get("max_workers", 4))
    pipeline_args: dict[str, Any] = {
        "output_file_name": "cassia", "output_dir": str(work_dir), "marker": marker_frame,
        "tissue": rows[0]["tissue"], "species": species.strip(), "overall_provider": provider,
        "annotation_model": model, "annotation_provider": provider,
        "score_model": str(config.get("score_model", model)), "score_provider": str(config.get("score_provider", provider)),
        "score_threshold": float(config.get("score_threshold", 75)), "max_workers": max(1, configured_workers),
        "max_retries": int(config.get("max_retries", 2)), "validator_involvement": str(config.get("validator_involvement", "v1")),
        "auto_convert_ids": bool(config.get("auto_convert_ids", False)),
    }
    reasoning = config.get("reasoning")
    if reasoning is not None:
        reasoning = str(reasoning).strip().lower()
        if reasoning not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("CASSIA reasoning must be one of none, minimal, low, medium, high, xhigh, max")
        pipeline_args["overall_reasoning"] = {"effort": reasoning}
    for name in ("annotationboost_model", "annotationboost_provider", "merge_model", "merge_provider"):
        if config.get(name) is not None:
            pipeline_args[name] = str(config[name])
    if marker_file:
        pipeline_args["n_genes"] = max(
            len([gene for gene in row["marker"].split(",") if gene.strip()]) for row in rows
        )
        _run_expression_pipeline(CASSIA.runCASSIA_pipeline, pipeline_args, rows)
    else:
        CASSIA.runCASSIA_pipeline(**pipeline_args)

    pipeline_dir = _pipeline_directory(work_dir)
    csv_dir = pipeline_dir / "03_csv_files"
    summary_path, scored_path = csv_dir / "cassia_summary.csv", csv_dir / "cassia_scored.csv"
    if not summary_path.is_file() or not scored_path.is_file():
        raise FileNotFoundError(f"CASSIA pipeline CSV output is incomplete under {csv_dir}")
    initial = _by_cluster(_read_csv(summary_path), summary_path)
    expected_ids = {row["source_row_id"] for row in rows}
    if set(initial) != expected_ids:
        raise _alignment_error("summary", set(initial), expected_ids)
    scores, threshold = _native_scores(scored_path, expected_ids), float(config.get("score_threshold", 75))

    lineage: list[dict[str, str]] = []
    final_predictions: dict[str, str] = {}
    for cluster_id in expected_ids:
        initial_prediction = initial[cluster_id].get("Predicted General Cell Type", "").strip()
        if not initial_prediction:
            raise ValueError(f"CASSIA returned an empty general cell type for {cluster_id}")
        boost_applied, boosted_prediction = scores[cluster_id] < threshold, ""
        if boost_applied:
            boost_dir = pipeline_dir / "02_annotation_boost" / _boost_folder(cluster_id)
            raw_files = sorted(boost_dir.glob("*_raw_conversation.txt")) if boost_dir.is_dir() else []
            if len(raw_files) != 1:
                raise ValueError(f"CASSIA Boost output is missing or ambiguous for {cluster_id}")
            boosted_prediction = _boosted_prediction(raw_files[0])
        final_predictions[cluster_id] = boosted_prediction or initial_prediction
        lineage.append({"source_row_id": cluster_id, "initial_prediction": initial_prediction,
                        "initial_cassia_score": str(scores[cluster_id]), "boost_applied": str(boost_applied).lower(),
                        "boosted_prediction": boosted_prediction, "final_prediction": final_predictions[cluster_id]})

    with (work_dir / "cassia_annotation_lineage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(lineage[0]))
        writer.writeheader()
        writer.writerows(lineage)
    shutil.copyfile(scored_path, work_dir / "cassia_native_scores.csv")

    output_path = Path(task["output_tsv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STANDARD_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({"source_row_id": row["source_row_id"], "dataset": row["dataset"], "tissue": row["tissue"],
                             "prediction": final_predictions[row["source_row_id"]], "method": "cassia", "model": model,
                             "task_id": task["task_id"]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path)
    args = parser.parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    if task.get("action", "annotate") != "annotate":
        raise ValueError("CASSIA caller supports only annotation pipeline tasks")
    run_task(task)


if __name__ == "__main__":
    main()
