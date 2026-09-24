"""Isolated expression-marker preprocessing worker used by ``runner.run``."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


# ``runner.expression_task`` is executed inside a method-specific Conda
# environment.  In some environments the subprocess working directory is not
# retained on ``sys.path`` even though Python can locate the ``runner`` module
# itself.  Anchor imports to this checkout so the repository adapters remain
# available without installing the benchmark as a package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def run_task(task: dict[str, Any]) -> None:
    from adapter.expression.expression import ExpressionDataAdapter

    expression = task["expression"]
    adapter = ExpressionDataAdapter(
        task["h5ad"],
        raw_count_layer=expression.get("raw_count_layer", "raw_counts"),
        target_sum=expression.get("target_sum", 10_000.0),
        tissue_column=expression.get("tissue_column", "tissue"),
        cell_type_column=expression.get("cell_type_column", "cell_type"),
    )
    marker_query = adapter.generate_query(
        "gptcelltype",
        groupby=expression["groupby"],
        marker_method=expression.get("marker_method", "t-test-seurat3"),
        marker_top_n=expression.get("marker_top_n", 10),
        marker_method_parameters=expression.get("marker_method_parameters", {}),
        dataname=expression.get("dataname"),
        marker_statistics_dir=expression.get("marker_statistics_dir"),
        skip_marker_gene=expression.get("skip_marker_gene", False),
    )
    markers = marker_query.marker_genes
    export = task.get("annotation_export")
    if export is not None:
        if not isinstance(export, dict):
            raise ValueError("annotation_export must be a mapping")
        _write_annotation_export(adapter, markers, export, expression, task.get("method_exports", {}))


def _write_annotation_export(
    adapter: Any, markers: Any, export: dict[str, Any], expression: dict[str, Any], method_exports: Any,
) -> None:
    """Export caller-safe markers and a runner-private truth mapping."""
    required = ("dataset", "dataname", "query_tsv", "evaluation_json", "task_id", "tissue_context")
    missing = [key for key in required if not isinstance(export.get(key), str) or not export[key].strip()]
    if missing:
        raise ValueError("annotation_export lacks required field(s): " + ", ".join(missing))
    tissue_column = export.get("tissue_column", adapter.tissue_column)
    if tissue_column not in adapter.adata.obs:
        raise KeyError(f"adata.obs lacks tissue column {tissue_column!r}")
    tissues = sorted({str(value).strip() for value in adapter.adata.obs[tissue_column] if str(value).strip()})
    if len(tissues) != 1:
        raise ValueError("Each expression h5ad must contain exactly one non-empty tissue for annotation export")
    if not isinstance(method_exports, dict) or any(
        not isinstance(name, str) or not isinstance(path, str) or not path.strip()
        for name, path in method_exports.items()
    ):
        raise ValueError("method_exports must map implementation names to output paths")
    marker_rows = []
    truth: dict[str, str] = {}
    source_by_group: dict[str, str] = {}
    ordered_cell_types = sorted({str(value) for value in adapter.adata.obs[expression_groupby(export)].astype(str)})
    for group, group_rows in markers.groupby("group", sort=True):
        genes = group_rows.sort_values("rank", kind="stable")["gene"].astype(str).tolist()
        if not genes:
            raise ValueError(f"Expression group {group!r} has no marker genes")
        try:
            original_group = ordered_cell_types[int(str(group)) - 1]
        except (ValueError, IndexError) as error:
            raise ValueError(f"Expression marker group {group!r} is not a valid cached group ID") from error
        cell_type_values = adapter.adata.obs.loc[
            adapter.adata.obs[expression_groupby(export)].astype(str) == original_group,
            adapter.cell_type_column,
        ]
        labels = sorted({str(value).strip() for value in cell_type_values if str(value).strip()})
        if len(labels) != 1:
            raise ValueError(f"Expression marker group {group!r} does not map to one cell type")
        identity = "\0".join((export["dataset"], export["dataname"], str(group)))
        source_row_id = "expr_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        marker_rows.append((source_row_id, export["dataset"], tissues[0], ",".join(genes)))
        truth[source_row_id] = labels[0]
        source_by_group[str(group)] = source_row_id
        source_by_group[original_group] = source_row_id
    query_path = Path(export["query_tsv"])
    query_path.parent.mkdir(parents=True, exist_ok=True)
    with query_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("source_row_id", "dataset", "tissue", "marker"))
        writer.writerows(marker_rows)
    evaluation_path = Path(export["evaluation_json"])
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps({"ground_truth_by_source_row_id": truth}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_method_exports(adapter, markers, source_by_group, expression, method_exports)


def _write_method_exports(
    adapter: Any, markers: Any, source_by_group: dict[str, str], expression: dict[str, Any], method_exports: dict[str, str],
) -> None:
    """Write caller-specific, label-free expression artifacts."""
    for implementation, output_value in method_exports.items():
        destination = Path(output_value)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if implementation == "gptcelltype":
            with destination.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(("source_row_id",))
            continue
        if implementation == "cassia":
            _write_cassia_export(adapter, markers, source_by_group, expression, destination)
            continue
        if implementation == "celltypeagent":
            _write_celltypeagent_export(adapter, markers, source_by_group, expression, destination)
            continue
        raise ValueError(f"Expression export is not defined for caller {implementation!r}")


def _write_cassia_export(
    adapter: Any, markers: Any, source_by_group: dict[str, str], expression: dict[str, Any], destination: Path,
) -> None:
    cassia_query = adapter.generate_query(
        "cassia", groupby=str(expression["groupby"]),
        marker_method=str(expression.get("marker_method", "t-test-seurat3")),
        marker_top_n=int(expression.get("marker_top_n", 10)),
        marker_method_parameters=dict(expression.get("marker_method_parameters", {})),
        dataname=expression.get("dataname"), marker_statistics_dir=expression.get("marker_statistics_dir"),
        skip_marker_gene=bool(expression.get("skip_marker_gene", False)),
    )
    cassia_markers = cassia_query.evidence.copy()
    cassia_markers["cluster"] = cassia_markers["cluster"].astype(str).map(source_by_group)
    if cassia_markers["cluster"].isna().any():
        raise ValueError("CASSIA marker groups do not align with annotation source rows")
    cassia_markers.loc[:, ["cluster", "gene", "avg_log2FC", "p_val_adj", "pct.1", "pct.2", "p_val"]].to_csv(
        destination, index=False,
    )


def _write_celltypeagent_export(
    adapter: Any, markers: Any, source_by_group: dict[str, str], expression: dict[str, Any], destination: Path,
) -> None:
    """Export per-submitted-group expression summaries without reference labels."""
    import numpy as np
    import pandas as pd
    from scipy import sparse

    groupby = str(expression["groupby"])
    groups = sorted({str(value) for value in adapter.adata.obs[groupby].astype(str)})
    group_ids = {group: str(index) for index, group in enumerate(groups, start=1)}
    genes = markers.sort_values("rank", kind="stable")["gene"].drop_duplicates().astype(str).tolist()
    positions = adapter._gene_positions(adapter.adata.var_names.astype(str), genes)
    matrix = adapter.adata.layers["normalization+log1p"][:, positions]
    values = adapter.adata.obs[groupby].astype(str).to_numpy()
    records = []
    for group, group_id in group_ids.items():
        subset = matrix[values == group]
        size = int((values == group).sum())
        if sparse.issparse(subset):
            mean = np.asarray(subset.mean(axis=0)).ravel()
            fraction = np.asarray(subset.getnnz(axis=0)).ravel() / size
        else:
            mean = np.asarray(subset).mean(axis=0)
            fraction = np.count_nonzero(subset, axis=0) / size
        for gene, mean_value, fraction_value in zip(genes, mean, fraction):
            records.append({"source_row_id": source_by_group[group_id], "gene": gene,
                            "mean_normalized_expression": float(mean_value), "expressed_fraction": float(fraction_value)})
    pd.DataFrame.from_records(records, columns=("source_row_id", "gene", "mean_normalized_expression", "expressed_fraction")).to_csv(destination, index=False)


def expression_groupby(export: dict[str, Any]) -> str:
    groupby = export.get("groupby")
    if not isinstance(groupby, str) or not groupby.strip():
        raise ValueError("annotation_export.groupby must be a non-empty string")
    return groupby


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    args = parser.parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    if not isinstance(task, dict) or not isinstance(task.get("expression"), dict):
        raise ValueError("expression task must contain an expression mapping")
    run_task(task)


if __name__ == "__main__":
    main()
