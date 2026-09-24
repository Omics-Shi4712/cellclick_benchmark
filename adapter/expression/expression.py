"""Build label-free expression evidence from one preselected AnnData object.

Analysis-unit selection (including tissue splitting) is deliberately the
caller's responsibility.  This adapter filters to the pinned protein-coding
gene list, derives a normalized/log-transformed layer, and ranks markers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


NORMALIZED_COUNTS_LAYER = "normalization"
NORMALIZED_LOG_LAYER = "normalization+log1p"
T_TEST_SEURAT3_LOG_LAYER = "normalization+log2p"
PROTEIN_CODING_GENE_FILE = Path(__file__).with_name("protein_coding_genes_list.csv")
MARKER_COLUMNS = ("groupby", "group", "rank", "gene", "score", "pvals_adj")
CASSIA_MARKER_COLUMNS = ("cluster", "gene", "avg_log2FC", "p_val_adj", "pct.1", "pct.2", "p_val")
RICH_MARKER_COLUMNS = MARKER_COLUMNS + ("pvals", "log2fc", "pct_in", "pct_rest")
MARKER_CACHE_COLUMNS = ("cell_type_id", "cell_type", "marker_genes")
CELLTYPEAGENT_COLUMNS = (
    "Tissue", "Cell Type", "Cell Count", "Tissue Composition", "Gene Symbol",
    "Expression", "Expression, Scaled", "Number of Cells Expressing Genes",
)
DEFAULT_MARKER_WORKERS = 5


def _rank_one_t_test_seurat3_worker(payload: tuple[Any, str, str, int]) -> Any:
    """Rank one group in an isolated process using the fixed t-test protocol."""
    adata, groupby, group, top_n = payload
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy import sparse

    labels = adata.obs[groupby].astype(str)
    target = labels == str(group)
    if not bool(target.any()) or bool(target.all()):
        raise ValueError(f"group {group!r} must have both target and rest cells")
    worker_groupby = "__benchmark_group__"
    adata.obs[worker_groupby] = np.where(target, "target", "rest")
    key_added = "__benchmark_markers__"
    sc.tl.rank_genes_groups(
        adata, groupby=worker_groupby, groups=["target"], reference="rest", method="t-test",
        corr_method="bonferroni", use_raw=False, layer=T_TEST_SEURAT3_LOG_LAYER,
        n_genes=adata.n_vars, key_added=key_added,
    )
    result = sc.get.rank_genes_groups_df(adata, group="target", key=key_added).set_index("names")
    normalized_counts = adata.layers[NORMALIZED_COUNTS_LAYER]
    mean_in, pct_in = ExpressionDataAdapter._mean_and_detected(normalized_counts[target, :], sparse)
    mean_out, pct_out = ExpressionDataAdapter._mean_and_detected(normalized_counts[~target, :], sparse)
    log2fc = np.log2((mean_in + 1.0) / (mean_out + 1.0))
    extra = pd.DataFrame(
        {"log2fc": log2fc, "pct_in": pct_in, "pct_rest": pct_out},
        index=adata.var_names.astype(str),
    )
    result = result.join(extra, how="left").reset_index(names="gene")
    result = result.loc[
        (result[["pct_in", "pct_rest"]].max(axis=1) >= 0.10)
        & (result["log2fc"].abs() >= 0.25)
        & (result["pvals_adj"] < 0.01)
    ].sort_values(["pvals", "log2fc"], ascending=[True, False], kind="mergesort")
    records = []
    for rank, row in enumerate(result.head(top_n).itertuples(index=False), start=1):
        records.append({
            "groupby": groupby, "group": str(group), "rank": rank,
            "gene": str(row.gene), "score": float(row.scores), "pvals_adj": float(row.pvals_adj),
        })
    return records


@dataclass(frozen=True)
class ExpressionQuery:
    """Label-free marker evidence and the representation for one method.

    ``marker_genes`` always retains the ranking output from
    :meth:`ExpressionDataAdapter.generate_markers`.  ``evidence`` is the
    method-specific representation derived from the same selected AnnData.
    """

    marker_genes: Any
    evidence: Any


class ExpressionDataAdapter:
    """Prepare expression evidence from a caller-selected AnnData object.

    ``raw_count_layer`` should normally be configured as ``raw_counts``. Use
    ``"X"`` when the raw counts are stored in :attr:`AnnData.X` rather than
    an AnnData layer.
    Library-size normalization uses all genes in the raw-count matrix before
    protein-coding filtering. ``raw_layers`` is a backward-compatible alias
    retained for existing callers.
    """

    def __init__(
        self,
        data: Any,
        raw_count_layer: str | None = None,
        *,
        raw_layers: str | None = None,
        target_sum: float = 10_000.0,
        tissue_column: str = "tissue",
        cell_type_column: str = "cell_type",
    ) -> None:
        if raw_count_layer is not None and raw_layers is not None and raw_count_layer != raw_layers:
            raise ValueError("raw_count_layer and raw_layers must match when both are supplied")
        resolved_layer = raw_count_layer if raw_count_layer is not None else raw_layers
        if not isinstance(resolved_layer, str) or not resolved_layer.strip():
            raise ValueError("raw_count_layer must name a raw-count layer")
        if target_sum <= 0:
            raise ValueError("target_sum must be positive")

        self.adata = self._read_adata(data)
        self.raw_count_layer = resolved_layer
        self.raw_layers = resolved_layer
        self.target_sum = float(target_sum)
        self.tissue_column = tissue_column
        self.cell_type_column = cell_type_column
        self._genes_before_filter = int(self.adata.shape[1])
        # Preserve the established normalization semantics: total UMI counts
        # are calculated from every raw feature.  Subset before materializing
        # normalized layers, however, so large h5ad files do not hold full
        # 60k-feature normalized copies in memory.
        self._raw_count_totals = self._raw_count_library_totals()
        self._filter_protein_coding_genes()
        self._create_normalized_layers()
        self._generators: dict[str, Callable[..., ExpressionQuery]] = {
            "celltypeagent": self._generate_celltypeagent_query,
            "cassia": self._generate_cassia_query,
            "gptcelltype": self._generate_gptcelltype_query,
        }

    @staticmethod
    def _read_adata(data: Any) -> Any:
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix != ".h5ad":
                raise ValueError(f"Expected an .h5ad file, got {path}")
            try:
                import anndata as ad
            except (ImportError, ValueError) as error:
                raise RuntimeError("Reading h5ad requires working anndata and h5py") from error
            return ad.read_h5ad(path)
        required = ("layers", "shape", "obs", "var_names", "__getitem__")
        if any(not hasattr(data, name) for name in required):
            raise TypeError("data must be AnnData-compatible or an .h5ad path")
        return data

    @staticmethod
    def _protein_coding_genes() -> set[str]:
        """Load ``Gene name`` from the repository-pinned gene list."""
        try:
            with PROTEIN_CODING_GENE_FILE.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "Gene name" not in reader.fieldnames:
                    raise ValueError("protein-coding gene list lacks 'Gene name'")
                genes = {row["Gene name"].strip() for row in reader if row.get("Gene name", "").strip()}
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Protein-coding gene list not found: {PROTEIN_CODING_GENE_FILE}") from error
        if not genes:
            raise ValueError("protein-coding gene list contains no usable gene symbols")
        return genes

    def _filter_protein_coding_genes(self) -> None:
        """Subset every variable-axis matrix while retaining input gene order."""
        keep = self.adata.var_names.astype(str).isin(self._protein_coding_genes())
        if not bool(keep.any()):
            raise ValueError("No adata.var_names match the pinned protein-coding gene list")
        self.adata = self.adata[:, keep].copy()
        self._genes_after_filter = int(self.adata.shape[1])

    def _raw_count_library_totals(self) -> Any:
        """Validate raw counts and return per-cell totals before subsetting.

        The returned totals retain the all-feature library-size denominator
        required by the benchmark protocol while allowing normalized layers to
        be built only for retained protein-coding genes.
        """
        try:
            import numpy as np
            from scipy import sparse
        except (ImportError, ValueError) as error:
            raise RuntimeError("ExpressionDataAdapter requires numpy and scipy") from error
        counts = self._raw_counts()
        if counts.shape != self.adata.shape:
            raise ValueError("raw_count_layer matrix shape must match adata.shape")
        if sparse.issparse(counts):
            if counts.data.size and (not np.isfinite(counts.data).all() or (counts.data < 0).any()):
                raise ValueError("raw count layer must contain finite, non-negative values")
            return np.asarray(counts.sum(axis=1)).ravel()
        dense_counts = np.asarray(counts)
        if not np.isfinite(dense_counts).all() or (dense_counts < 0).any():
            raise ValueError("raw count layer must contain finite, non-negative values")
        return dense_counts.sum(axis=1, keepdims=True)

    def _raw_counts(self) -> Any:
        if self.raw_count_layer == "X":
            return self.adata.X
        if self.raw_count_layer not in self.adata.layers:
            raise KeyError(f"Raw count layer {self.raw_count_layer!r} is not present in adata.layers")
        return self.adata.layers[self.raw_count_layer]

    @property
    def preprocessing_summary(self) -> dict[str, Any]:
        """Local preprocessing facts suitable for reproducibility records."""
        return {
            "raw_count_layer": self.raw_count_layer,
            "protein_coding_genes_before": self._genes_before_filter,
            "protein_coding_genes_after": self._genes_after_filter,
            "target_sum": self.target_sum,
            "normalized_counts_layer": NORMALIZED_COUNTS_LAYER,
            "normalized_log_layer": NORMALIZED_LOG_LAYER,
            "t_test_seurat3_log_layer": T_TEST_SEURAT3_LOG_LAYER,
        }

    def _create_normalized_layers(self) -> None:
        """Normalize all raw-count genes, then create natural- and base-2-log layers.

        The unlogged normalized matrix is retained for Seurat-3-compatible
        detection fractions and log2 fold changes. Zero-total cells remain
        zero and ``adata.X`` is not modified.
        """
        try:
            import numpy as np
            from scipy import sparse
        except (ImportError, ValueError) as error:
            raise RuntimeError("ExpressionDataAdapter requires numpy and scipy") from error
        counts = self._raw_counts()
        if counts.shape != self.adata.shape:
            raise ValueError("raw_count_layer matrix shape must match adata.shape")
        if sparse.issparse(counts):
            normalized = counts.astype(np.float64, copy=True).tocsr()
            if normalized.data.size and (not np.isfinite(normalized.data).all() or (normalized.data < 0).any()):
                raise ValueError("raw count layer must contain finite, non-negative values")
            totals = self._raw_count_totals
            factors = np.divide(self.target_sum, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
            normalized = sparse.diags(factors) @ normalized
            self.adata.layers[NORMALIZED_COUNTS_LAYER] = normalized.copy()
            self.adata.layers[NORMALIZED_LOG_LAYER] = normalized.copy()
            self.adata.layers[NORMALIZED_LOG_LAYER].data = np.log1p(self.adata.layers[NORMALIZED_LOG_LAYER].data)
            normalized.data = np.log2(normalized.data + 1.0)
        else:
            normalized = np.asarray(counts, dtype=np.float64)
            if not np.isfinite(normalized).all() or (normalized < 0).any():
                raise ValueError("raw count layer must contain finite, non-negative values")
            totals = self._raw_count_totals
            normalized = np.divide(normalized * self.target_sum, totals, out=np.zeros_like(normalized), where=totals > 0)
            self.adata.layers[NORMALIZED_COUNTS_LAYER] = normalized.copy()
            self.adata.layers[NORMALIZED_LOG_LAYER] = np.log1p(normalized)
            normalized = np.log2(normalized + 1.0)
        self.adata.layers[T_TEST_SEURAT3_LOG_LAYER] = normalized

    def generate_markers(
        self, *, groupby: str, method: str = "t-test-seurat3", top_n: int = 10,
        key_added: str | None = None, workers: int = 1, include_statistics: bool = False,
        **method_kwargs: Any,
    ) -> Any:
        """Return ordered top markers for an ``obs`` grouping column.

        ``t-test-seurat3`` and ``wilcoxon-seurat`` apply the fixed TS marker
        protocols; ``cosg`` uses the COSG package. No tissue splitting is done
        here.
        """
        if not isinstance(groupby, str) or not groupby.strip():
            raise ValueError("groupby must be a non-empty adata.obs column name")
        if groupby not in self.adata.obs:
            raise KeyError(f"adata.obs lacks requested grouping column {groupby!r}")
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise ValueError("workers must be a positive integer")
        normalized_method = method.strip().lower() if isinstance(method, str) else ""
        if normalized_method not in {"t-test-seurat3", "wilcoxon-seurat", "cosg"}:
            raise ValueError("method must be one of: 't-test-seurat3', 'wilcoxon-seurat', 'cosg'")
        if self.adata.obs[groupby].dropna().astype(str).nunique() < 2:
            raise ValueError("marker ranking requires at least two non-null groups")
        result_key = key_added or f"expression_markers_{normalized_method.replace('-', '_')}"
        if not isinstance(result_key, str) or not result_key.strip():
            raise ValueError("key_added must be a non-empty string when supplied")
        if normalized_method == "t-test-seurat3":
            if workers > 1:
                return self._rank_markers_t_test_seurat3_parallel(
                    groupby=groupby, top_n=top_n, workers=workers,
                )
            return self._rank_markers_t_test_seurat3(
                groupby=groupby, top_n=top_n, key_added=result_key,
                include_statistics=include_statistics, **method_kwargs,
            )
        if normalized_method == "wilcoxon-seurat":
            if workers != 1:
                raise ValueError("workers > 1 is not supported for wilcoxon-seurat")
            return self._rank_markers_wilcoxon_seurat(
                groupby=groupby, top_n=top_n, key_added=result_key,
                include_statistics=include_statistics, **method_kwargs,
            )
        if workers != 1:
            raise ValueError("workers > 1 is currently supported only for t-test-seurat3")
        return self._rank_markers_cosg(groupby=groupby, top_n=top_n, key_added=result_key, **method_kwargs)

    def _rank_markers_t_test_seurat3_parallel(self, *, groupby: str, top_n: int, workers: int) -> Any:
        """Run one-vs-rest marker jobs in processes and merge deterministically."""
        try:
            import pandas as pd
        except (ImportError, ValueError) as error:
            raise RuntimeError("Parallel marker ranking requires pandas") from error
        groups = sorted(self.adata.obs[groupby].dropna().astype(str).unique().tolist())
        if len(groups) < 2:
            raise ValueError("marker ranking requires at least two non-null groups")
        max_workers = min(workers, len(groups), max(1, os.cpu_count() or 1))
        payloads = [(self.adata.copy(), groupby, group, top_n) for group in groups]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            chunks = list(executor.map(_rank_one_t_test_seurat3_worker, payloads))
        records = [record for chunk in chunks for record in chunk]
        return pd.DataFrame.from_records(records, columns=MARKER_COLUMNS)

    def _rank_markers_t_test_seurat3(
        self, *, groupby: str, top_n: int, key_added: str,
        include_statistics: bool = False, **kwargs: Any,
    ) -> Any:
        """Run the fixed TS t-test protocol before selecting the requested top N."""
        if kwargs:
            raise ValueError(
                "t-test-seurat3 does not accept method parameters; its statistical protocol is fixed"
            )
        try:
            import numpy as np
            import pandas as pd
            import scanpy as sc
            from scipy import sparse
        except (ImportError, ValueError) as error:
            raise RuntimeError("t-test-seurat3 marker ranking requires scanpy, numpy, pandas, and scipy") from error
        sc.tl.rank_genes_groups(
            self.adata, groupby=groupby, reference="rest", method="t-test",
            corr_method="bonferroni", use_raw=False, layer=T_TEST_SEURAT3_LOG_LAYER,
            n_genes=self.adata.n_vars, key_added=key_added,
        )
        rankings = self.adata.uns[key_added]
        names = rankings.get("names") if hasattr(rankings, "get") else None
        if names is None or not getattr(names.dtype, "names", None):
            raise ValueError("Marker ranking result lacks structured 'names' fields")

        normalized_counts = self.adata.layers[NORMALIZED_COUNTS_LAYER]
        labels = self.adata.obs[groupby].astype(str).to_numpy()
        records: list[dict[str, Any]] = []
        for group in names.dtype.names:
            inside = labels == str(group)
            outside = ~inside
            mean_in, pct_in = self._mean_and_detected(normalized_counts[inside, :], sparse)
            mean_out, pct_out = self._mean_and_detected(normalized_counts[outside, :], sparse)
            log2fc = np.log2((mean_in + 1.0) / (mean_out + 1.0))
            extra = pd.DataFrame(
                {"log2fc": log2fc, "pct_in": pct_in, "pct_rest": pct_out},
                index=self.adata.var_names.astype(str),
            )
            result = sc.get.rank_genes_groups_df(self.adata, group=group, key=key_added)
            result = result.set_index("names").join(extra, how="left").reset_index(names="gene")
            result = result.loc[
                (result[["pct_in", "pct_rest"]].max(axis=1) >= 0.10)
                & (result["log2fc"].abs() >= 0.25)
                & (result["pvals_adj"] < 0.01)
            ].sort_values(["pvals", "log2fc"], ascending=[True, False], kind="mergesort")
            for rank, row in enumerate(result.head(top_n).itertuples(index=False), start=1):
                record = {
                    "groupby": groupby, "group": str(group), "rank": rank,
                    "gene": str(row.gene), "score": float(row.scores), "pvals_adj": float(row.pvals_adj),
                }
                if include_statistics:
                    record.update({
                        "pvals": float(row.pvals), "log2fc": float(row.log2fc),
                        "pct_in": float(row.pct_in), "pct_rest": float(row.pct_rest),
                    })
                records.append(record)
        columns = RICH_MARKER_COLUMNS if include_statistics else MARKER_COLUMNS
        return pd.DataFrame.from_records(records, columns=columns)

    def _rank_markers_wilcoxon_seurat(
        self, *, groupby: str, top_n: int, key_added: str,
        include_statistics: bool = False, **kwargs: Any,
    ) -> Any:
        """Rank TS markers with Scanpy's Wilcoxon implementation.

        The expression layer follows Seurat ``NormalizeData`` with
        ``LogNormalize`` and scale factor 10,000.  Fold changes are computed
        from the unlogged normalized expression using the Seurat-style
        average-expression definition, while p-values and adjusted p-values
        come from ``scanpy.tl.rank_genes_groups(method='wilcoxon')``.
        """
        if kwargs:
            raise ValueError("wilcoxon-seurat does not accept method parameters; its protocol is fixed")
        try:
            import numpy as np
            import pandas as pd
            import scanpy as sc
            from scipy import sparse
        except (ImportError, ValueError) as error:
            raise RuntimeError("wilcoxon-seurat requires scanpy, numpy, pandas, and scipy") from error
        sc.tl.rank_genes_groups(
            self.adata, groupby=groupby, reference="rest", method="wilcoxon",
            corr_method="bonferroni", use_raw=False, layer=NORMALIZED_LOG_LAYER,
            n_genes=self.adata.n_vars, key_added=key_added,
        )
        rankings = self.adata.uns[key_added]
        names = rankings.get("names") if hasattr(rankings, "get") else None
        if names is None or not getattr(names.dtype, "names", None):
            raise ValueError("Marker ranking result lacks structured 'names' fields")

        normalized_counts = self.adata.layers[NORMALIZED_COUNTS_LAYER]
        labels = self.adata.obs[groupby].astype(str).to_numpy()
        records: list[dict[str, Any]] = []
        for group in names.dtype.names:
            inside = labels == str(group)
            outside = ~inside
            mean_in, pct_in = self._mean_and_detected(normalized_counts[inside, :], sparse)
            mean_out, pct_out = self._mean_and_detected(normalized_counts[outside, :], sparse)
            log2fc = np.log2((mean_in + 1.0) / (mean_out + 1.0))
            extra = pd.DataFrame(
                {"log2fc": log2fc, "pct_in": pct_in, "pct_rest": pct_out},
                index=self.adata.var_names.astype(str),
            )
            result = sc.get.rank_genes_groups_df(self.adata, group=group, key=key_added)
            result = result.set_index("names").join(extra, how="left").reset_index(names="gene")
            result = result.loc[
                (result[["pct_in", "pct_rest"]].max(axis=1) >= 0.10)
                & (result["log2fc"].abs() >= 0.25)
                & (result["pvals_adj"] < 0.01)
            ].sort_values(["pvals", "log2fc"], ascending=[True, False], kind="mergesort")
            for rank, row in enumerate(result.head(top_n).itertuples(index=False), start=1):
                record = {
                    "groupby": groupby, "group": str(group), "rank": rank,
                    "gene": str(row.gene), "score": float(row.scores), "pvals_adj": float(row.pvals_adj),
                }
                if include_statistics:
                    record.update({
                        "pvals": float(row.pvals), "log2fc": float(row.log2fc),
                        "pct_in": float(row.pct_in), "pct_rest": float(row.pct_rest),
                    })
                records.append(record)
        columns = RICH_MARKER_COLUMNS if include_statistics else MARKER_COLUMNS
        return pd.DataFrame.from_records(records, columns=columns)

    @staticmethod
    def _mean_and_detected(matrix: Any, sparse: Any) -> tuple[Any, Any]:
        """Return per-gene mean normalized expression and nonzero fraction."""
        import numpy as np

        if sparse.issparse(matrix):
            return np.asarray(matrix.mean(axis=0)).ravel(), np.asarray(matrix.getnnz(axis=0)).ravel() / matrix.shape[0]
        values = np.asarray(matrix)
        return values.mean(axis=0), np.count_nonzero(values, axis=0) / values.shape[0]

    def _rank_markers_cosg(self, *, groupby: str, top_n: int, key_added: str, **kwargs: Any) -> Any:
        forbidden = {"adata", "groupby", "n_genes_user", "key_added", "use_raw", "layer", "copy"}
        overlap = forbidden.intersection(kwargs)
        if overlap:
            raise ValueError(f"COSG options cannot override protocol fields: {', '.join(sorted(overlap))}")
        try:
            import cosg
        except (ImportError, ValueError) as error:
            raise RuntimeError("COSG marker ranking requires cosg") from error
        cosg.cosg(
            self.adata, groupby=groupby, n_genes_user=top_n, key_added=key_added,
            use_raw=False, layer=NORMALIZED_LOG_LAYER, copy=False, **kwargs,
        )
        return self._marker_table_from_rankings(self.adata.uns[key_added], groupby=groupby, top_n=top_n)

    @staticmethod
    def _marker_table_from_rankings(rankings: Any, *, groupby: str, top_n: int) -> Any:
        """Create a common marker table without reordering implementation output."""
        try:
            import numpy as np
            import pandas as pd
        except (ImportError, ValueError) as error:
            raise RuntimeError("Marker-table conversion requires pandas and numpy") from error
        names = rankings.get("names") if hasattr(rankings, "get") else None
        if names is None or not getattr(names.dtype, "names", None):
            raise ValueError("Marker ranking result lacks structured 'names' fields")
        scores, adjusted = rankings.get("scores"), rankings.get("pvals_adj")
        records: list[dict[str, Any]] = []
        for group in names.dtype.names:
            for position in range(min(top_n, len(names[group]))):
                records.append({
                    "groupby": groupby,
                    "group": str(group),
                    "rank": position + 1,
                    "gene": str(names[group][position]),
                    "score": float(scores[group][position]) if scores is not None else np.nan,
                    "pvals_adj": float(adjusted[group][position]) if adjusted is not None else np.nan,
                })
        return pd.DataFrame.from_records(records, columns=MARKER_COLUMNS)

    def generate_markers_from_config(
        self, config: Mapping[str, Any], *, skip_marker_gene: bool | None = None,
        workers: int = DEFAULT_MARKER_WORKERS,
    ) -> Any:
        """Generate or reuse precomputed marker genes described by ``config``.

        Required configuration fields are ``dataname``,
        ``marker_statistics_dir``, and ``groupby``. Optional fields are
        ``marker_method`` (``t-test-seurat3`` by default), ``marker_top_n`` (10),
        ``marker_method_parameters`` (a JSON-compatible mapping), and
        ``skip_marker_gene``. The cache retains the real ``cell_type`` for
        offline reference, but loads only numeric ``cell_type_id`` values as
        marker groups for later annotation. ``skip_marker_gene`` supplied by
        a CLI takes precedence over the config value.
        """
        if not isinstance(config, Mapping):
            raise TypeError("marker configuration must be a mapping")
        required = ("dataname", "marker_statistics_dir", "groupby")
        missing = [name for name in required if name not in config]
        if missing:
            raise KeyError(f"marker configuration lacks required field(s): {', '.join(missing)}")
        dataname = str(config["dataname"]).strip()
        if not dataname or Path(dataname).name != dataname:
            raise ValueError("dataname must be a non-empty filename component")
        method = str(config.get("marker_method", "t-test-seurat3")).strip().lower()
        top_n = config.get("marker_top_n", 10)
        groupby = config["groupby"]
        method_parameters = config.get("marker_method_parameters", {})
        if not isinstance(method_parameters, Mapping):
            raise TypeError("marker_method_parameters must be a mapping")
        method_parameters = dict(method_parameters)
        try:
            json.dumps(method_parameters, sort_keys=True)
        except TypeError as error:
            raise TypeError("marker_method_parameters must be JSON-serializable") from error
        use_cache = config.get("skip_marker_gene", False) if skip_marker_gene is None else skip_marker_gene
        if not isinstance(use_cache, bool):
            raise TypeError("skip_marker_gene must be boolean")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise ValueError("workers must be a positive integer")
        cache_dir = Path(config["marker_statistics_dir"])
        csv_path, metadata_path = self._marker_cache_paths(cache_dir, dataname, method, top_n)
        output_ids = self._marker_output_ids(groupby)
        parameters = self._marker_cache_parameters(
            dataname=dataname, groupby=groupby, method=method, top_n=top_n, method_parameters=method_parameters,
            output_ids=output_ids,
        )
        if use_cache:
            return self._read_marker_cache(csv_path, metadata_path, parameters)
        marker_workers = workers if method == "t-test-seurat3" else 1
        markers = self.generate_markers(
            groupby=groupby, method=method, top_n=top_n, workers=marker_workers, **method_parameters,
        )
        self._write_marker_cache(
            markers, csv_path, metadata_path, parameters,
            output_ids=output_ids,
        )
        return self._read_marker_cache(csv_path, metadata_path, parameters)

    @staticmethod
    def _marker_cache_paths(cache_dir: Path, dataname: str, method: str, top_n: int) -> tuple[Path, Path]:
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("marker_top_n must be a positive integer")
        if method not in {"t-test-seurat3", "wilcoxon-seurat", "cosg"}:
            raise ValueError("marker_method must be one of: 't-test-seurat3', 'wilcoxon-seurat', 'cosg'")
        stem = f"{dataname}_{method}_{top_n}"
        return cache_dir / f"{stem}.csv", cache_dir / f"{stem}.json"

    def _marker_cache_parameters(
        self, *, dataname: str, groupby: Any, method: str, top_n: int,
        method_parameters: Mapping[str, Any], output_ids: Mapping[str, str],
    ) -> dict[str, Any]:
        var_names = "\n".join(self.adata.var_names.astype(str)).encode("utf-8")
        return {
            "schema_version": 2,
            "dataname": dataname,
            "groupby": groupby,
            "output_ids_sha256": hashlib.sha256(
                json.dumps(dict(sorted(output_ids.items())), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "method": method,
            "top_n": top_n,
            "method_parameters": dict(method_parameters),
            "raw_count_layer": self.raw_count_layer,
            "target_sum": self.target_sum,
            "normalized_counts_layer": NORMALIZED_COUNTS_LAYER,
            "normalized_log_layer": NORMALIZED_LOG_LAYER,
            "t_test_seurat3_log_layer": T_TEST_SEURAT3_LOG_LAYER,
            "t_test_seurat3_protocol": {
                "correction": "bonferroni",
                "min_pct": 0.10,
                "log2fc_threshold": 0.25,
                "return_thresh": 0.01,
                "sort": ["pvals_asc", "log2fc_desc"],
            },
            "wilcoxon_seurat_protocol": {
                "test": "scanpy.tl.rank_genes_groups(method='wilcoxon')",
                "correction": "bonferroni",
                "min_pct": 0.10,
                "log2fc_threshold": 0.25,
                "return_thresh": 0.01,
                "fold_change": "log2((mean(normalized_counts_in)+1)/(mean(normalized_counts_rest)+1))",
                "sort": ["pvals_asc", "log2fc_desc"],
            },
            "n_obs": int(self.adata.shape[0]),
            "n_vars": int(self.adata.shape[1]),
            "var_names_sha256": hashlib.sha256(var_names).hexdigest(),
        }

    def _marker_output_ids(self, groupby: str) -> dict[str, str]:
        """Assign stable numeric IDs to reference-defined DE groups."""
        groups = sorted({str(group) for group in self.adata.obs[groupby].astype(str)})
        return {group: str(index) for index, group in enumerate(groups, start=1)}

    @staticmethod
    def _write_marker_cache(
        markers: Any, csv_path: Path, metadata_path: Path, parameters: Mapping[str, Any], *, output_ids: Mapping[str, str],
    ) -> None:
        try:
            import pandas as pd
        except (ImportError, ValueError) as error:
            raise RuntimeError("Writing marker caches requires pandas") from error
        cache_rows = []
        for group, group_rows in markers.groupby("group", sort=False):
            cache_rows.append({
                "cell_type_id": output_ids[str(group)],
                "cell_type": str(group),
                "marker_genes": json.dumps(group_rows.sort_values("rank", kind="stable")["gene"].tolist()),
            })
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(cache_rows, columns=MARKER_CACHE_COLUMNS).to_csv(csv_path, index=False)
        metadata_path.write_text(
            json.dumps({"parameters": parameters}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _read_marker_cache(csv_path: Path, metadata_path: Path, parameters: Mapping[str, Any]) -> Any:
        try:
            import numpy as np
            import pandas as pd
        except (ImportError, ValueError) as error:
            raise RuntimeError("Reading marker caches requires pandas and numpy") from error
        if not csv_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Marker cache is incomplete: expected {csv_path} and {metadata_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid marker cache metadata JSON: {metadata_path}") from error
        if metadata.get("parameters") != dict(parameters):
            raise ValueError("Marker cache parameters do not match the active configuration")
        cached = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        if tuple(cached.columns) != MARKER_CACHE_COLUMNS:
            raise ValueError(f"Marker cache has unexpected columns: {csv_path}")
        records: list[dict[str, Any]] = []
        for row in cached.itertuples(index=False):
            try:
                genes = json.loads(row.marker_genes)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid marker_genes JSON for {row.cell_type_id!r}") from error
            if not isinstance(genes, list) or not all(isinstance(gene, str) and gene for gene in genes):
                raise ValueError(f"Invalid ordered marker list for {row.cell_type_id!r}")
            for rank, gene in enumerate(genes, start=1):
                records.append({
                    "groupby": parameters["groupby"], "group": row.cell_type_id, "rank": rank,
                    "gene": gene, "score": np.nan, "pvals_adj": np.nan,
                })
        return pd.DataFrame.from_records(records, columns=MARKER_COLUMNS)

    def generate_query(
        self,
        method_name: str,
        *,
        groupby: str,
        marker_method: str = "t-test-seurat3",
        marker_top_n: int = 10,
        marker_key_added: str | None = None,
        marker_method_parameters: Mapping[str, Any] | None = None,
        dataname: str | None = None,
        marker_statistics_dir: str | Path | None = None,
        skip_marker_gene: bool = False,
        workers: int = DEFAULT_MARKER_WORKERS,
        **generator_kwargs: Any,
    ) -> ExpressionQuery:
        """Generate markers, then derive one method's evidence from them.

        ``groupby`` defines the submitted analysis units and must be selected
        by the caller.  It is deliberately required so this adapter never
        infers a reference-label column.  Method-specific options (for
        example ``output_path`` for CellTypeAgent) belong in
        ``generator_kwargs``.
        """
        if not isinstance(method_name, str) or not method_name.strip():
            raise ValueError("method_name must be a non-empty string")
        if marker_method_parameters is None:
            marker_method_parameters = {}
        if not isinstance(marker_method_parameters, Mapping):
            raise TypeError("marker_method_parameters must be a mapping")
        try:
            generator = self._generators[method_name.strip().lower()]
        except KeyError as error:
            supported = ", ".join(sorted(self._generators))
            raise ValueError(f"Unsupported expression method {method_name!r}; supported: {supported}") from error
        method = method_name.strip().lower()
        if marker_statistics_dir is not None or dataname is not None:
            if not isinstance(marker_statistics_dir, (str, Path)) or not str(marker_statistics_dir).strip():
                raise ValueError("marker_statistics_dir is required when using marker cache configuration")
            if not isinstance(dataname, str) or not dataname.strip():
                raise ValueError("dataname is required when using marker cache configuration")
            config = {
                "dataname": dataname, "marker_statistics_dir": str(marker_statistics_dir),
                "groupby": groupby, "marker_method": marker_method,
                "marker_top_n": marker_top_n, "marker_method_parameters": dict(marker_method_parameters),
                "skip_marker_gene": skip_marker_gene,
            }
            if method == "cassia":
                if marker_method.strip().lower() != "t-test-seurat3":
                    raise ValueError("CASSIA expression queries require marker_method='t-test-seurat3'")
                markers = self._generate_cassia_markers_from_config(config, workers=workers)
            else:
                markers = self.generate_markers_from_config(config, workers=workers)
        else:
            markers = self.generate_markers(
                groupby=groupby, method=marker_method, top_n=marker_top_n,
                key_added=marker_key_added,
                workers=(1 if method == "cassia" or marker_method.strip().lower() in {"cosg", "wilcoxon-seurat"} else workers),
                include_statistics=(method == "cassia"), **dict(marker_method_parameters),
            )
        return generator(marker_genes=markers, **generator_kwargs)

    def _generate_cassia_markers_from_config(self, config: Mapping[str, Any], *, workers: int) -> Any:
        """Generate or load the rich DE table required by CASSIA."""
        method = str(config.get("marker_method", "t-test-seurat3")).strip().lower()
        top_n = config.get("marker_top_n", 10)
        dataname = str(config["dataname"])
        cache_dir = Path(config["marker_statistics_dir"])
        csv_path = cache_dir / f"{dataname}_{method}_{top_n}_cassia.csv"
        metadata_path = cache_dir / f"{dataname}_{method}_{top_n}_cassia.json"
        output_ids = self._marker_output_ids(config["groupby"])
        parameters = self._marker_cache_parameters(
            dataname=dataname, groupby=config["groupby"], method=method, top_n=top_n,
            method_parameters=config.get("marker_method_parameters", {}), output_ids=output_ids,
        )
        parameters = {**parameters, "output_format": "cassia-rich", "schema_version": 3}
        if bool(config.get("skip_marker_gene", False)):
            return self._read_cassia_cache(csv_path, metadata_path, parameters)
        markers = self.generate_markers(
            groupby=config["groupby"], method=method, top_n=top_n, workers=1,
            include_statistics=True, **dict(config.get("marker_method_parameters", {})),
        )
        self._write_cassia_cache(markers, csv_path, metadata_path, parameters)
        return self._read_cassia_cache(csv_path, metadata_path, parameters)

    @staticmethod
    def _write_cassia_cache(markers: Any, csv_path: Path, metadata_path: Path, parameters: Mapping[str, Any]) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        markers.to_csv(csv_path, index=False)
        metadata_path.write_text(json.dumps({"parameters": dict(parameters)}, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _read_cassia_cache(csv_path: Path, metadata_path: Path, parameters: Mapping[str, Any]) -> Any:
        try:
            import pandas as pd
        except (ImportError, ValueError) as error:
            raise RuntimeError("Reading CASSIA marker caches requires pandas") from error
        if not csv_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"CASSIA marker cache is incomplete: expected {csv_path} and {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("parameters") != dict(parameters):
            raise ValueError("CASSIA marker cache parameters do not match the active configuration")
        cached = pd.read_csv(csv_path)
        required = set(RICH_MARKER_COLUMNS)
        if not required.issubset(cached.columns):
            raise ValueError(f"CASSIA marker cache lacks required statistics: {csv_path}")
        return cached.loc[:, RICH_MARKER_COLUMNS]

    def _generate_cassia_query(self, *, marker_genes: Any, **kwargs: Any) -> ExpressionQuery:
        if kwargs:
            raise TypeError("cassia does not accept expression export options")
        missing = [column for column in ("group", "gene", "log2fc", "pvals_adj", "pct_in", "pct_rest", "pvals") if column not in marker_genes]
        if missing:
            raise ValueError("CASSIA marker table lacks required statistics: " + ", ".join(missing))
        evidence = marker_genes.loc[:, ["group", "gene", "log2fc", "pvals_adj", "pct_in", "pct_rest", "pvals"]].rename(
            columns={"group": "cluster", "log2fc": "avg_log2FC", "pvals_adj": "p_val_adj", "pct_in": "pct.1", "pct_rest": "pct.2", "pvals": "p_val"}
        )
        return ExpressionQuery(marker_genes=marker_genes, evidence=evidence.loc[:, CASSIA_MARKER_COLUMNS])

    def _generate_gptcelltype_query(self, *, marker_genes: Any, **kwargs: Any) -> ExpressionQuery:
        if kwargs:
            raise TypeError("gptcelltype does not accept expression export options")
        return ExpressionQuery(marker_genes=marker_genes, evidence=marker_genes.loc[:, ["gene"]].copy())

    def _generate_celltypeagent_query(
        self,
        *,
        marker_genes: Any,
        output_path: str | Path | None = None,
        genes: Iterable[str] | None = None,
    ) -> ExpressionQuery:
        """Create CellTypeAgent's CELLxGENE-style group-by-gene evidence table."""
        try:
            import numpy as np
            import pandas as pd
            from scipy import sparse
        except (ImportError, ValueError) as error:
            raise RuntimeError("CellTypeAgent expression export requires pandas, numpy, and scipy") from error
        for column in (self.tissue_column, self.cell_type_column):
            if column not in self.adata.obs:
                raise KeyError(f"adata.obs lacks required CellTypeAgent column {column!r}")
        if self.adata.var_names.has_duplicates:
            raise ValueError("adata.var_names must be unique for CellTypeAgent export")
        var_names = self.adata.var_names.astype(str)
        if genes is None:
            genes = marker_genes.sort_values("rank", kind="stable")["gene"].drop_duplicates().tolist()
        positions = self._gene_positions(var_names, genes)
        matrix = self.adata.layers[NORMALIZED_LOG_LAYER][:, positions]
        selected_genes = var_names[positions]
        tissues = self.adata.obs[self.tissue_column].astype(str).to_numpy()
        cell_types = self.adata.obs[self.cell_type_column].astype(str).to_numpy()
        groups = pd.DataFrame({"Tissue": tissues, "Cell Type": cell_types}).drop_duplicates().itertuples(index=False, name=None)
        records: list[dict[str, Any]] = []
        for tissue, cell_type in groups:
            mask = (tissues == tissue) & (cell_types == cell_type)
            group_matrix, cell_count = matrix[mask], int(mask.sum())
            if sparse.issparse(group_matrix):
                mean, expressing = np.asarray(group_matrix.mean(axis=0)).ravel(), np.asarray(group_matrix.getnnz(axis=0)).ravel()
            else:
                mean, expressing = np.asarray(group_matrix).mean(axis=0), np.count_nonzero(group_matrix, axis=0)
            for gene, value, count in zip(selected_genes, mean, expressing):
                records.append({
                    "Tissue": tissue, "Cell Type": cell_type, "Cell Count": cell_count,
                    "Tissue Composition": f"{int(count) / cell_count:.2%}", "Gene Symbol": gene,
                    "Expression": float(value), "Number of Cells Expressing Genes": int(count),
                })
        result = pd.DataFrame.from_records(records)
        if result.empty:
            result = pd.DataFrame(columns=CELLTYPEAGENT_COLUMNS)
        else:
            maxima = result.groupby("Gene Symbol")["Expression"].transform("max")
            result["Expression, Scaled"] = np.divide(result["Expression"], maxima, out=np.zeros(len(result), dtype=float), where=maxima.to_numpy() > 0)
            result = result.loc[:, CELLTYPEAGENT_COLUMNS]
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(destination, index=False)
        return ExpressionQuery(marker_genes=marker_genes, evidence=result)

    @staticmethod
    def _gene_positions(var_names: Any, genes: Iterable[str] | None) -> list[int]:
        if genes is None:
            return list(range(len(var_names)))
        requested = [str(gene).strip() for gene in genes if str(gene).strip()]
        if not requested:
            raise ValueError("genes must contain at least one non-empty gene symbol")
        positions_by_name = {name: position for position, name in enumerate(var_names)}
        missing = [gene for gene in requested if gene not in positions_by_name]
        if missing:
            raise ValueError(f"Requested genes are absent from filtered adata.var_names: {', '.join(missing)}")
        return [positions_by_name[gene] for gene in requested]
