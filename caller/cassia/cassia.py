#!/usr/bin/env python3
"""CASSIA annotation and score adapters for the benchmark caller contract."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


def read_query(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"source_row_id", "dataset", "tissue", "marker"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Query TSV lacks required label-free columns")
    if len({row["source_row_id"] for row in rows}) != len(rows):
        raise ValueError("Query TSV has duplicate source_row_id values")
    return rows


def species_for_dataset(dataset: str, configured: dict[str, str]) -> str:
    return configured.get(dataset, "human")


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
        return base_url.rstrip("/")
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required for CASSIA provider=openai")
    return provider


def run_task(task: dict[str, Any]) -> None:
    import pandas as pd
    import CASSIA

    rows = read_query(Path(task["query_tsv"]))
    config = task.get("method_config", {})
    work_dir = Path(task["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "cassia_input.csv"
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("cluster_id", "markers"))
        writer.writeheader()
        writer.writerows({"cluster_id": row["source_row_id"], "markers": row["marker"]} for row in rows)

    dataset = rows[0]["dataset"]
    species_mapping = {"MCA": "mouse", "mammal": "mammal", **config.get("dataset_species", {})}
    output_name = work_dir / "cassia"
    provider = provider_from_environment(config)
    CASSIA.runCASSIA_batch(
        marker=pd.read_csv(input_path),
        output_name=str(output_name),
        celltype_column="cluster_id",
        gene_column_name="markers",
        tissue=rows[0]["tissue"],
        species=species_for_dataset(dataset, species_mapping),
        model=str(config.get("model", "gpt-4o-2024-08-06")),
        provider=provider,
        temperature=float(config.get("temperature", 0)),
        validator_involvement=str(config.get("validator_involvement", "v1")),
        max_workers=int(config.get("max_workers", 4)),
        max_retries=int(config.get("max_retries", 2)),
        auto_convert_ids=bool(config.get("auto_convert_ids", False)),
    )
    summary_path = Path(f"{output_name}_summary.csv")
    if not summary_path.is_file():
        raise FileNotFoundError(f"CASSIA summary not found: {summary_path}")
    with summary_path.open(encoding="utf-8-sig", newline="") as handle:
        summary = list(csv.DictReader(handle))
    by_id = {row.get("Cluster ID", ""): row for row in summary}
    expected_ids = {row["source_row_id"] for row in rows}
    if set(by_id) != expected_ids or len(by_id) != len(summary):
        raise ValueError("CASSIA summary does not align one-to-one with the query")
    output_path = Path(task["output_tsv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_row_id", "dataset", "tissue", "prediction", "method", "model", "task_id"), delimiter="\t")
        writer.writeheader()
        for row in rows:
            prediction = by_id[row["source_row_id"]].get("Predicted General Cell Type", "").strip()
            if not prediction:
                raise ValueError(f"CASSIA returned an empty general cell type for {row['source_row_id']}")
            writer.writerow({
                "source_row_id": row["source_row_id"], "dataset": row["dataset"], "tissue": row["tissue"],
                "prediction": prediction, "method": "cassia", "model": config.get("model", "gpt-4o-2024-08-06"), "task_id": task["task_id"],
            })


def run_score_task(task: dict[str, Any]) -> None:
    """Score one frozen CASSIA summary without receiving benchmark truth."""
    import CASSIA

    input_path = Path(task["score_input_csv"])
    output_path = Path(task["score_output_csv"])
    if not input_path.is_file():
        raise FileNotFoundError(f"CASSIA score input not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = task.get("method_config", {})
    CASSIA.runCASSIA_score_batch(
        input_file=str(input_path),
        output_file=str(output_path),
        max_workers=int(config.get("max_workers", 4)),
        model=str(config.get("model", "gpt-4o-2024-08-06")),
        temperature=float(config.get("temperature", 0)),
        provider=provider_from_environment(config),
        max_retries=int(config.get("max_retries", 2)),
        generate_report=False,
        conversations_json_path=task.get("conversations_json_path"),
        reasoning=config.get("reasoning"),
    )
    if not output_path.is_file():
        raise FileNotFoundError(f"CASSIA score output not found: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path)
    args = parser.parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    if task.get("action", "annotate") == "score":
        run_score_task(task)
    else:
        run_task(task)


if __name__ == "__main__":
    main()
