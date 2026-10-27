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


def _log2fc_class(log2fc: float) -> str:
    if log2fc >= 0.1:
        return "positive"
    if log2fc <= -0.1:
        return "negative"
    return "neutral"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        & (result["log2fc"] >= 0.25)
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
    """Expression evidence and the representation for one method.

    ``marker_genes`` retains the ranking output for marker-based methods and
    is ``None`` for full-expression methods such as SingleR. ``evidence`` is
    the method-specific representation derived from the selected AnnData.
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
        protein_coding_only: bool = True,
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

        self._source_data = data
        self.adata = self._read_adata(data)
        self.raw_count_layer = resolved_layer
        self.raw_layers = resolved_layer
        self.target_sum = float(target_sum)
        if not isinstance(protein_coding_only, bool):
            raise TypeError("protein_coding_only must be boolean")
        self.protein_coding_only = protein_coding_only
        self.tissue_column = tissue_column
        self.cell_type_column = cell_type_column
        self._genes_before_filter = int(self.adata.shape[1])
        # Preserve the established normalization semantics: total UMI counts
        # are calculated from every raw feature.  Subset before materializing
        # normalized layers, however, so large h5ad files do not hold full
        # 60k-feature normalized copies in memory.
        self._raw_count_totals = self._raw_count_library_totals()
        if self.protein_coding_only:
            self._filter_protein_coding_genes()
        else:
            self._genes_after_filter = int(self.adata.shape[1])
        self._create_normalized_layers()
        self._generators: dict[str, Callable[..., ExpressionQuery]] = {
            "anndict": self._generate_anndict_query,
            "celltypeagent": self._generate_celltypeagent_query,
            "cassia": self._generate_cassia_query,
            "cellclick_v4": self._generate_cellclick_v4_query,
            "cellclick_v5": self._generate_cellclick_v4_query,
            "cellclick_v6": self._generate_cellclick_v4_query,
            "cellclick_v7": self._generate_cellclick_v4_query,
            "cellclick_v7_2": self._generate_cellclick_v4_query,
            "cellclick_v7_3": self._generate_cellclick_v4_query,
            "gptcelltype": self._generate_gptcelltype_query,
            "singler": self._generate_singler_query,
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
            "protein_coding_only": self.protein_coding_only,
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
        include_all_genes: bool = False,
        **method_kwargs: Any,
    ) -> Any:
        """Return ordered top markers for an ``obs`` grouping column.

        ``t-test-seurat3`` and the Wilcoxon variants apply fixed TS marker
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
        wilcoxon_methods = {
            "wilcoxon-seurat", "wilcoxon-seurat-v2",
            "wilcoxon-seurat-v2-pos-neg", "wilcoxon-seurat-v2-pos-neutral",
        }
        if include_all_genes and (workers != 1 or normalized_method not in {"t-test-seurat3", "t-test-seurat3_v2", *wilcoxon_methods}):
            raise ValueError("include_all_genes requires serial t-test-seurat3 or a Wilcoxon variant")
        if normalized_method not in {"t-test-seurat3", "t-test-seurat3_v2", *wilcoxon_methods, "cosg"}:
            raise ValueError("unsupported marker method")
        if self.adata.obs[groupby].dropna().astype(str).nunique() < 2:
            raise ValueError("marker ranking requires at least two non-null groups")
        missing = self.adata.obs[groupby].isna().to_numpy()
        if bool(missing.any()):
            self.adata = self.adata[~missing].copy()
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
                include_statistics=include_statistics, include_all_genes=include_all_genes, **method_kwargs,
            )
        if normalized_method == "t-test-seurat3_v2":
            if workers != 1:
                raise ValueError("workers > 1 is not supported for t-test-seurat3_v2")
            return self._rank_markers_t_test_seurat3_v2(
                groupby=groupby, top_n=top_n, key_added=result_key,
                include_statistics=include_statistics, include_all_genes=include_all_genes, **method_kwargs,
            )
        if normalized_method == "wilcoxon-seurat":
            if workers != 1:
                raise ValueError("workers > 1 is not supported for wilcoxon-seurat")
            return self._rank_markers_wilcoxon_seurat(
                groupby=groupby, top_n=top_n, key_added=result_key,
                include_statistics=include_statistics, include_all_genes=include_all_genes, **method_kwargs,
            )
        if normalized_method == "wilcoxon-seurat-v2":
            if workers != 1:
                raise ValueError("workers > 1 is not supported for wilcoxon-seurat-v2")
            return self._rank_markers_wilcoxon_seurat_v2(
                groupby=groupby, top_n=top_n, key_added=result_key,
                include_statistics=include_statistics, include_all_genes=include_all_genes, **method_kwargs,
            )
        if normalized_method in {"wilcoxon-seurat-v2-pos-neg", "wilcoxon-seurat-v2-pos-neutral"}:
            if workers != 1:
                raise ValueError(f"workers > 1 is not supported for {normalized_method}")
            return self._rank_markers_wilcoxon_seurat_v2_balanced(
                groupby=groupby, method=normalized_method, top_n=top_n, key_added=result_key,
                include_statistics=include_statistics, include_all_genes=include_all_genes, **method_kwargs,
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
        include_statistics: bool = False, include_all_genes: bool = False, **kwargs: Any,
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
            if not include_all_genes:
                result = result.loc[
                    (result[["pct_in", "pct_rest"]].max(axis=1) >= 0.10)
                    & (result["log2fc"] >= 0.25)
                    & (result["pvals_adj"] < 0.01)
                ]
            result = result.sort_values(["pvals", "log2fc"], ascending=[True, False], kind="mergesort")
            selected = result if include_all_genes else result.head(top_n)
            for rank, row in enumerate(selected.itertuples(index=False), start=1):
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

    def _rank_markers_t_test_seurat3_v2(
        self, *, groupby: str, top_n: int, key_added: str,
        include_statistics: bool = False, include_all_genes: bool = False, **kwargs: Any,
    ) -> Any:
        """Approximate Seurat 3 FindAllMarkers(test.use='t') with all features.

        Seurat 3 filters rounded detection rates and natural-log fold changes
        before testing, then applies the adjusted-P return threshold.
        Scanpy computes the Welch t statistics; a pinned Seurat 3 run is
        needed to establish exact cross-language numerical agreement.
        """
        if kwargs:
            raise ValueError("t-test-seurat3_v2 does not accept method parameters")
        if self.protein_coding_only:
            raise ValueError("t-test-seurat3_v2 requires protein_coding_only=False")
        try:
            import numpy as np
            import pandas as pd
            import scanpy as sc
            from scipy import sparse
        except (ImportError, ValueError) as error:
            raise RuntimeError("t-test-seurat3_v2 requires scanpy, numpy, pandas, and scipy") from error
        counts = self.adata.layers[NORMALIZED_COUNTS_LAYER]
        labels = self.adata.obs[groupby].astype(str).to_numpy()
        records: list[dict[str, Any]] = []
        for group in sorted(set(labels)):
            inside = labels == group
            outside = ~inside
            mean_in, pct_in = self._mean_and_detected(counts[inside, :], sparse)
            mean_out, pct_out = self._mean_and_detected(counts[outside, :], sparse)
            pct_in, pct_out = np.round(pct_in, 3), np.round(pct_out, 3)
            logfc = np.log((mean_in + 1.0) / (mean_out + 1.0))
            eligible = (np.maximum(pct_in, pct_out) > 0.10) & (logfc > 0.25)
            if not eligible.any():
                continue
            subset = self.adata[:, eligible].copy()
            subset.obs["__benchmark_group__"] = np.where(inside, "target", "rest")
            sc.tl.rank_genes_groups(
                subset, groupby="__benchmark_group__", groups=["target"],
                reference="rest", method="t-test", corr_method="bonferroni",
                use_raw=False, layer=NORMALIZED_LOG_LAYER,
                n_genes=subset.n_vars, key_added=key_added,
            )
            extra = pd.DataFrame(
                {"logfc": logfc[eligible], "pct_in": pct_in[eligible], "pct_rest": pct_out[eligible]},
                index=self.adata.var_names[eligible].astype(str),
            )
            result = sc.get.rank_genes_groups_df(subset, group="target", key=key_added)
            result = result.set_index("names").join(extra, how="left").reset_index(names="gene")
            result = result.loc[(result["logfc"] > 0.0) & (result["pvals_adj"] < 0.01)]
            result = result.sort_values(["pvals", "logfc"], ascending=[True, False], kind="mergesort")
            selected = result if include_all_genes else result.head(top_n)
            for rank, row in enumerate(selected.itertuples(index=False), start=1):
                record = {
                    "groupby": groupby, "group": group, "rank": rank,
                    "gene": str(row.gene), "score": float(row.scores),
                    "pvals_adj": float(row.pvals_adj),
                }
                if include_statistics:
                    record.update({
                        "pvals": float(row.pvals), "log2fc": float(row.logfc / np.log(2.0)),
                        "pct_in": float(row.pct_in), "pct_rest": float(row.pct_rest),
                    })
                records.append(record)
        return pd.DataFrame.from_records(
            records, columns=RICH_MARKER_COLUMNS if include_statistics else MARKER_COLUMNS,
        )

    def _rank_markers_wilcoxon_seurat(
        self, *, groupby: str, top_n: int, key_added: str,
        include_statistics: bool = False, include_all_genes: bool = False, **kwargs: Any,
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
            if not include_all_genes:
                result = result.loc[
                    (result[["pct_in", "pct_rest"]].max(axis=1) >= 0.10)
                    & (result["log2fc"] >= 0.25)
                    & (result["pvals_adj"] < 0.01)
                ]
            result = result.sort_values(["pvals", "log2fc"], ascending=[True, False], kind="mergesort")
            selected = result if include_all_genes else result.head(top_n)
            for rank, row in enumerate(selected.itertuples(index=False), start=1):
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

    def _rank_markers_wilcoxon_seurat_v2(
        self, *, groupby: str, top_n: int, key_added: str,
        include_statistics: bool = False, include_all_genes: bool = False, **kwargs: Any,
    ) -> Any:
        """Run a positive-only, reference-like Seurat Wilcoxon protocol.

        The gene universe remains the adapter's protein-coding subset by
        design. Candidate filtering precedes testing, and only positive
        log2FC genes are returned.
        """
        if kwargs:
            raise ValueError("wilcoxon-seurat-v2 does not accept method parameters")
        try:
            import numpy as np
            import pandas as pd
            import scanpy as sc
            from scipy import sparse
        except (ImportError, ValueError) as error:
            raise RuntimeError("wilcoxon-seurat-v2 requires scanpy, numpy, pandas, and scipy") from error
        counts = self.adata.layers[NORMALIZED_COUNTS_LAYER]
        labels = self.adata.obs[groupby].astype(str).to_numpy()
        records: list[dict[str, Any]] = []
        for group in sorted(set(labels)):
            inside = labels == group
            outside = ~inside
            mean_in, pct_in = self._mean_and_detected(counts[inside, :], sparse)
            mean_out, pct_out = self._mean_and_detected(counts[outside, :], sparse)
            log2fc = np.log2((mean_in + 1.0) / (mean_out + 1.0))
            eligible = (np.maximum(pct_in, pct_out) >= 0.01) & (log2fc >= 0.1)
            if not eligible.any():
                continue
            subset = self.adata[:, eligible].copy()
            subset.obs["__benchmark_group__"] = np.where(inside, "target", "rest")
            sc.tl.rank_genes_groups(
                subset, groupby="__benchmark_group__", groups=["target"],
                reference="rest", method="wilcoxon", corr_method="bonferroni",
                use_raw=False, layer=T_TEST_SEURAT3_LOG_LAYER, n_genes=subset.n_vars,
                key_added=key_added,
            )
            extra = pd.DataFrame(
                {"log2fc": log2fc[eligible], "pct_in": pct_in[eligible], "pct_rest": pct_out[eligible]},
                index=self.adata.var_names[eligible].astype(str),
            )
            result = sc.get.rank_genes_groups_df(subset, group="target", key=key_added)
            result = result.set_index("names").join(extra, how="left").reset_index(names="gene")
            result = result.loc[(result["log2fc"] > 0) & (result["pvals_adj"] < 0.01)]
            result = result.sort_values(["pvals", "log2fc"], ascending=[True, False], kind="mergesort")
            selected = result if include_all_genes else result.head(top_n)
            for rank, row in enumerate(selected.itertuples(index=False), start=1):
                record = {
                    "groupby": groupby, "group": group, "rank": rank,
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

    def _rank_markers_wilcoxon_seurat_v2_balanced(
        self, *, groupby: str, method: str, top_n: int, key_added: str,
        include_statistics: bool = False, include_all_genes: bool = False, **kwargs: Any,
    ) -> Any:
        """Select balanced marker sets from Seurat-style log2FC classes."""
        if kwargs:
            raise ValueError(f"{method} does not accept method parameters")
        try:
            import numpy as np
            import pandas as pd
            import scanpy as sc
            from scipy import sparse
        except (ImportError, ValueError) as error:
            raise RuntimeError(f"{method} requires scanpy, numpy, pandas, and scipy") from error

        counts = self.adata.layers[NORMALIZED_COUNTS_LAYER]
        labels = self.adata.obs[groupby].astype(str).to_numpy()
        other_class = "negative" if method.endswith("pos-neg") else "neutral"
        quota = top_n // 2
        records: list[dict[str, Any]] = []
        for group in sorted(set(labels)):
            inside = labels == group
            outside = ~inside
            mean_in, pct_in = self._mean_and_detected(counts[inside, :], sparse)
            mean_out, pct_out = self._mean_and_detected(counts[outside, :], sparse)
            log2fc = np.log2((mean_in + 1.0) / (mean_out + 1.0))
            eligible = np.maximum(pct_in, pct_out) >= 0.01
            if not eligible.any():
                continue
            subset = self.adata[:, eligible].copy()
            subset.obs["__benchmark_group__"] = np.where(inside, "target", "rest")
            sc.tl.rank_genes_groups(
                subset, groupby="__benchmark_group__", groups=["target"], reference="rest",
                method="wilcoxon", corr_method="bonferroni", use_raw=False,
                layer=T_TEST_SEURAT3_LOG_LAYER, n_genes=subset.n_vars, key_added=key_added,
            )
            extra = pd.DataFrame(
                {"log2fc": log2fc[eligible], "pct_in": pct_in[eligible], "pct_rest": pct_out[eligible]},
                index=self.adata.var_names[eligible].astype(str),
            )
            result = sc.get.rank_genes_groups_df(subset, group="target", key=key_added)
            result = result.set_index("names").join(extra, how="left").reset_index(names="gene")
            result = result.loc[result["pvals_adj"] < 0.01].copy()
            result["gene"] = result["gene"].astype(str)
            result["_class"] = result["log2fc"].map(_log2fc_class)
            result = result.sort_values(
                ["pvals", "log2fc", "gene"], ascending=[True, False, True], kind="mergesort",
            )
            chosen = pd.concat([
                result.loc[result["_class"] == "positive"].head(quota),
                result.loc[result["_class"] == other_class].head(quota),
            ], ignore_index=True)
            for rank, row in enumerate(chosen.itertuples(index=False), start=1):
                record = {
                    "groupby": groupby, "group": group, "rank": rank,
                    "gene": str(row.gene), "score": float(row.scores), "pvals_adj": float(row.pvals_adj),
                }
                if include_statistics:
                    record.update({
                        "pvals": float(row.pvals), "log2fc": float(row.log2fc),
                        "pct_in": float(row.pct_in), "pct_rest": float(row.pct_rest),
                    })
                records.append(record)
        return pd.DataFrame.from_records(
            records, columns=RICH_MARKER_COLUMNS if include_statistics else MARKER_COLUMNS,
        )

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
        a CLI takes precedence over the config value.  Cache reuse is
        best-effort: when either cache file is missing, markers are computed
        and a complete cache is written.  Existing complete caches remain
        subject to strict parameter and content validation.
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
        if use_cache and csv_path.is_file() and metadata_path.is_file():
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
        supported_methods = {
            "t-test-seurat3", "t-test-seurat3_v2", "wilcoxon-seurat", "wilcoxon-seurat-v2",
            "wilcoxon-seurat-v2-pos-neg", "wilcoxon-seurat-v2-pos-neutral", "cosg",
        }
        if method not in supported_methods:
            raise ValueError("unsupported marker_method")
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
            "protein_coding_only": self.protein_coding_only,
            "t_test_seurat3_v2_protocol": {
                "gene_universe": "all_raw_count_features",
                "logfc_base": "e", "min_pct": 0.10,
                "logfc_threshold": 0.25, "return_thresh_pval": 0.01,
                "bonferroni_n": "all_features", "sort": ["pvals_asc", "logfc_desc"],
            },
            "t_test_seurat3_protocol": {
                "correction": "bonferroni",
                "min_pct": 0.10,
                "log2fc_threshold": 0.25,
                "return_thresh": 0.01,
                "only_pos": True,
                "sort": ["pvals_asc", "log2fc_desc"],
            },
            "wilcoxon_seurat_protocol": {
                "test": "scanpy.tl.rank_genes_groups(method='wilcoxon')",
                "correction": "bonferroni",
                "min_pct": 0.10,
                "log2fc_threshold": 0.25,
                "return_thresh": 0.01,
                "only_pos": True,
                "fold_change": "log2((mean(normalized_counts_in)+1)/(mean(normalized_counts_rest)+1))",
                "sort": ["pvals_asc", "log2fc_desc"],
            },
            "wilcoxon_seurat_v2_protocol": {
                "gene_universe": "protein_coding_features",
                "normalization": "log2(1 + target_sum * raw_count / cell_library_size)",
                "test": "scanpy.tl.rank_genes_groups(method='wilcoxon')",
                "only_pos": True,
                "min_pct": 0.01,
                "log2fc_threshold": 0.1,
                "return_thresh": 0.01,
                "candidate_filter_before_test": True,
                "sort": ["pvals_asc", "log2fc_desc"],
            },
            **({"wilcoxon_seurat_v2_balanced_protocol": {
                "gene_universe": "protein_coding_features",
                "normalization": "log2(1 + target_sum * raw_count / cell_library_size)",
                "test": "scanpy.tl.rank_genes_groups(method='wilcoxon')",
                "correction": "bonferroni",
                "candidate_min_pct": 0.01,
                "return_thresh": 0.01,
                "classes": {"positive": "log2fc >= 0.1", "neutral": "-0.1 < log2fc < 0.1", "negative": "log2fc <= -0.1"},
                "sort": ["pvals_asc", "log2fc_desc", "gene_asc"],
                "quota": "floor(top_n/2) per selected class; no cross-class fill",
                "selected_classes": ["positive", "negative"] if method.endswith("pos-neg") else ["positive", "neutral"],
            }} if method in {"wilcoxon-seurat-v2-pos-neg", "wilcoxon-seurat-v2-pos-neutral"} else {}),
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
        """Derive one method's evidence, generating markers when required.

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
        if method == "singler":
            return generator(marker_genes=None, groupby=groupby, **generator_kwargs)
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
                markers = self._generate_cassia_markers_from_config(config, workers=workers)
            else:
                markers = self.generate_markers_from_config(config, workers=workers)
        else:
            markers = self.generate_markers(
                groupby=groupby, method=marker_method, top_n=marker_top_n,
                key_added=marker_key_added,
                workers=(1 if method == "cassia" or marker_method.strip().lower() in {"cosg", "wilcoxon-seurat", "wilcoxon-seurat-v2", "t-test-seurat3_v2"} else workers),
                include_statistics=(method == "cassia"), include_all_genes=(method == "cassia"),
                **dict(marker_method_parameters),
            )
        if method == "cassia":
            return generator(marker_genes=markers, top_n=marker_top_n, **generator_kwargs)
        if method in {"cellclick_v4", "cellclick_v5", "cellclick_v6", "cellclick_v7", "cellclick_v7_2", "cellclick_v7_3"}:
            return generator(marker_genes=markers, groupby=groupby, **generator_kwargs)
        return generator(marker_genes=markers, **generator_kwargs)

    def _generate_singler_query(
        self,
        *,
        marker_genes: Any,
        groupby: str,
        output_path: str | Path,
        source_row_id_by_cell_type: Mapping[str, str],
    ) -> ExpressionQuery:
        """Write full-gene, tissue-level expression using the SingleR protocol."""
        import h5py
        import numpy as np
        import pandas as pd
        from scipy import sparse

        if marker_genes is not None:
            raise ValueError("SingleR expression generation does not accept marker genes")
        if self.target_sum != 10000:
            raise ValueError("SingleR paper protocol requires target_sum=10000")
        # The common adapter filters its working AnnData to protein-coding
        # genes. SingleR instead consumes every source gene, so reopen the
        # original input rather than exporting ``self.adata``.
        source = self._read_adata(self._source_data)
        missing = {self.tissue_column, self.cell_type_column}.difference(source.obs.columns)
        if missing:
            raise KeyError(f"adata.obs lacks required column(s): {sorted(missing)}")
        if groupby != self.cell_type_column:
            raise ValueError("SingleR groupby must match the configured manual cell-type column")
        if not source.obs_names.is_unique:
            raise ValueError("SingleR input contains duplicate cell IDs")
        if not source.var_names.is_unique:
            raise ValueError("SingleR input contains duplicate gene names")

        cell_ids = pd.Index(source.obs_names).astype(str).tolist()
        genes = pd.Index(source.var_names).astype(str).tolist()
        tissues = source.obs[self.tissue_column]
        cell_types = source.obs[self.cell_type_column]
        if tissues.isna().any() or cell_types.isna().any():
            raise ValueError("SingleR tissue and manual cell-type metadata cannot contain missing values")
        tissue_values = tissues.astype(str).str.strip().tolist()
        cell_type_values = cell_types.astype(str).str.strip().tolist()
        if any(not value for value in cell_ids):
            raise ValueError("SingleR cell IDs cannot be empty")
        if any(not value for value in genes):
            raise ValueError("SingleR gene names cannot be empty")
        if any(not value for value in tissue_values):
            raise ValueError("SingleR tissue metadata cannot be empty")
        if any(not value for value in cell_type_values):
            raise ValueError("SingleR manual cell-type metadata cannot be empty")
        if len(set(tissue_values)) != 1:
            raise ValueError("Each SingleR expression export must contain exactly one tissue")
        if set(source_row_id_by_cell_type) != set(cell_type_values):
            raise ValueError("SingleR population mapping does not exactly match manual cell types")
        if len(set(source_row_id_by_cell_type.values())) != len(source_row_id_by_cell_type):
            raise ValueError("SingleR population mapping contains duplicate source_row_id values")
        if any(not isinstance(value, str) or not value.strip() for value in source_row_id_by_cell_type.values()):
            raise ValueError("SingleR population mapping contains an empty source_row_id")

        if self.raw_count_layer == "X":
            counts = source.X
        elif self.raw_count_layer in source.layers:
            counts = source.layers[self.raw_count_layer]
        else:
            raise KeyError(f"Raw count layer {self.raw_count_layer!r} is not present in adata.layers")
        if counts.shape != source.shape:
            raise ValueError("raw-count matrix shape must match adata.shape")
        if sparse.issparse(counts):
            normalized = counts.astype(np.float64, copy=True).tocsr()
            if normalized.data.size and (
                not np.isfinite(normalized.data).all() or (normalized.data < 0).any()
            ):
                raise ValueError("raw counts must contain finite, non-negative values")
            library_sizes = np.asarray(normalized.sum(axis=1)).ravel()
            if np.any(library_sizes <= 0):
                raise ValueError("SingleR input contains cells with zero total expression")
            normalized = sparse.diags(self.target_sum / library_sizes) @ normalized
            normalized.data = np.log2(normalized.data + 1.0)
            expression = normalized.transpose().tocsc()
        else:
            normalized = np.asarray(counts, dtype=np.float64)
            if not np.isfinite(normalized).all() or (normalized < 0).any():
                raise ValueError("raw counts must contain finite, non-negative values")
            library_sizes = normalized.sum(axis=1)
            if np.any(library_sizes <= 0):
                raise ValueError("SingleR input contains cells with zero total expression")
            normalized = normalized * (self.target_sum / library_sizes)[:, None]
            expression = sparse.csc_matrix(np.log2(normalized + 1.0).T)

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        try:
            with h5py.File(temporary, "w") as handle:
                handle.attrs["schema"] = "ai_cta_singler_expression"
                handle.attrs["schema_version"] = 2
                handle.attrs["orientation"] = "genes_by_cells"
                handle.attrs["normalization"] = "log2(1 + target_sum * raw_count / cell_library_size)"
                handle.attrs["target_sum"] = self.target_sum
                handle.attrs["raw_count_layer"] = self.raw_count_layer
                matrix = handle.create_group("expression")
                matrix.attrs["format"] = "csc_matrix"
                matrix.create_dataset("data", data=expression.data, compression="gzip")
                matrix.create_dataset("indices", data=expression.indices.astype(np.int64), compression="gzip")
                matrix.create_dataset("indptr", data=expression.indptr.astype(np.int64), compression="gzip")
                matrix.create_dataset("shape", data=np.asarray(expression.shape, dtype=np.int64))
                handle.create_dataset("genes", data=np.asarray(genes, dtype=object), dtype=string_dtype)
                metadata = handle.create_group("meta")
                metadata.attrs["columns"] = np.asarray(
                    ("cell_id", "tissue", "source_row_id"), dtype=object,
                )
                metadata.create_dataset("cell_id", data=np.asarray(cell_ids, dtype=object), dtype=string_dtype)
                metadata.create_dataset("tissue", data=np.asarray(tissue_values, dtype=object), dtype=string_dtype)
                metadata.create_dataset(
                    "source_row_id",
                    data=np.asarray([source_row_id_by_cell_type[value] for value in cell_type_values], dtype=object),
                    dtype=string_dtype,
                )
                populations = handle.create_group("populations")
                populations.create_dataset(
                    "source_row_id",
                    data=np.asarray(sorted(source_row_id_by_cell_type.values()), dtype=object),
                    dtype=string_dtype,
                )
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return ExpressionQuery(marker_genes=None, evidence=destination)

    def _generate_cassia_markers_from_config(self, config: Mapping[str, Any], *, workers: int) -> Any:
        """Generate or load the rich DE table required by CASSIA."""
        method = str(config.get("marker_method", "t-test-seurat3")).strip().lower()
        if method not in {"t-test-seurat3", "wilcoxon-seurat", "wilcoxon-seurat-v2"}:
            raise ValueError(
                "CASSIA expression queries require a marker method with rich DE statistics: "
                "'t-test-seurat3', 'wilcoxon-seurat', or 'wilcoxon-seurat-v2'"
            )
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
        parameters = {**parameters, "output_format": "cassia-rich", "marker_scope": "all_genes", "schema_version": 4}
        if (
            bool(config.get("skip_marker_gene", False))
            and csv_path.is_file()
            and metadata_path.is_file()
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached_parameters = metadata.get("parameters", {})
            if (
                cached_parameters.get("schema_version") == 4
                and cached_parameters.get("marker_scope") == "all_genes"
            ):
                return self._read_cassia_cache(csv_path, metadata_path, parameters)
        markers = self.generate_markers(
            groupby=config["groupby"], method=method, top_n=top_n, workers=1,
            include_statistics=True, include_all_genes=True, **dict(config.get("marker_method_parameters", {})),
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

    def write_sctype_expression(
        self,
        *,
        groupby: str,
        output_path: str | Path,
        source_row_id_by_group: Mapping[str, str],
    ) -> Path:
        """Export an anonymous, full-gene CP10K/log1p matrix for ScType.

        The export deliberately reopens the source AnnData, because ScType's
        input must retain all genes while the common adapter works on a
        protein-coding subset for marker ranking.
        """
        import h5py
        import numpy as np
        import pandas as pd
        from scipy import sparse

        source = self._read_adata(self._source_data)
        required = {self.tissue_column, groupby}
        missing = required.difference(source.obs.columns)
        if missing:
            raise KeyError(f"adata.obs lacks required column(s): {sorted(missing)}")
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError("ScType input requires unique cell IDs and gene names")
        cell_ids = pd.Index(source.obs_names).astype(str).tolist()
        genes = pd.Index(source.var_names).astype(str).tolist()
        tissues = source.obs[self.tissue_column].astype(str).str.strip().tolist()
        groups = source.obs[groupby].astype(str).str.strip().tolist()
        if any(not value for value in cell_ids + genes + tissues + groups):
            raise ValueError("ScType expression identifiers and metadata cannot be empty")
        if len(set(tissues)) != 1:
            raise ValueError("Each ScType expression export must contain exactly one tissue")
        if set(groups) != set(source_row_id_by_group):
            raise ValueError("ScType population mapping does not exactly match annotation groups")
        if len(set(source_row_id_by_group.values())) != len(source_row_id_by_group) or any(
            not isinstance(value, str) or not value.strip() for value in source_row_id_by_group.values()
        ):
            raise ValueError("ScType population mapping must contain unique non-empty source_row_id values")
        if self.raw_count_layer == "X":
            counts = source.X
        elif self.raw_count_layer in source.layers:
            counts = source.layers[self.raw_count_layer]
        else:
            raise KeyError(f"Raw count layer {self.raw_count_layer!r} is not present in adata.layers")
        if counts.shape != source.shape:
            raise ValueError("raw-count matrix shape must match adata.shape")
        if sparse.issparse(counts):
            normalized = counts.astype(np.float64, copy=True).tocsr()
            if normalized.data.size and (not np.isfinite(normalized.data).all() or (normalized.data < 0).any()):
                raise ValueError("raw counts must contain finite, non-negative values")
            totals = np.asarray(normalized.sum(axis=1)).ravel()
            if np.any(totals <= 0):
                raise ValueError("ScType input contains cells with zero total expression")
            normalized = sparse.diags(self.target_sum / totals) @ normalized
            normalized.data = np.log1p(normalized.data)
            expression = normalized.transpose().tocsc()
        else:
            normalized = np.asarray(counts, dtype=np.float64)
            if not np.isfinite(normalized).all() or (normalized < 0).any():
                raise ValueError("raw counts must contain finite, non-negative values")
            totals = normalized.sum(axis=1)
            if np.any(totals <= 0):
                raise ValueError("ScType input contains cells with zero total expression")
            expression = sparse.csc_matrix(np.log1p(normalized * (self.target_sum / totals)[:, None]).T)

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        try:
            with h5py.File(temporary, "w") as handle:
                handle.attrs["schema"] = "ai_cta_sctype_expression"
                handle.attrs["schema_version"] = 1
                handle.attrs["orientation"] = "genes_by_cells"
                handle.attrs["normalization"] = "log1p(target_sum * raw_count / cell_library_size)"
                handle.attrs["target_sum"] = self.target_sum
                handle.attrs["raw_count_layer"] = self.raw_count_layer
                matrix = handle.create_group("expression")
                matrix.attrs["format"] = "csc_matrix"
                matrix.create_dataset("data", data=expression.data, compression="gzip")
                matrix.create_dataset("indices", data=expression.indices.astype(np.int64), compression="gzip")
                matrix.create_dataset("indptr", data=expression.indptr.astype(np.int64), compression="gzip")
                matrix.create_dataset("shape", data=np.asarray(expression.shape, dtype=np.int64))
                handle.create_dataset("genes", data=np.asarray(genes, dtype=object), dtype=string_dtype)
                metadata = handle.create_group("meta")
                metadata.create_dataset("cell_id", data=np.asarray(cell_ids, dtype=object), dtype=string_dtype)
                metadata.create_dataset("tissue", data=np.asarray(tissues, dtype=object), dtype=string_dtype)
                metadata.create_dataset(
                    "source_row_id",
                    data=np.asarray([source_row_id_by_group[group] for group in groups], dtype=object),
                    dtype=string_dtype,
                )
                populations = handle.create_group("populations")
                populations.create_dataset(
                    "source_row_id",
                    data=np.asarray(sorted(source_row_id_by_group.values()), dtype=object), dtype=string_dtype,
                )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def _generate_cassia_query(self, *, marker_genes: Any, top_n: int = 10, **kwargs: Any) -> ExpressionQuery:
        if kwargs:
            raise TypeError("cassia does not accept expression export options")
        missing = [column for column in ("group", "gene", "log2fc", "pvals_adj", "pct_in", "pct_rest", "pvals") if column not in marker_genes]
        if missing:
            raise ValueError("CASSIA marker table lacks required statistics: " + ", ".join(missing))
        evidence = marker_genes.loc[:, ["group", "gene", "log2fc", "pvals_adj", "pct_in", "pct_rest", "pvals"]].rename(
            columns={"group": "cluster", "log2fc": "avg_log2FC", "pvals_adj": "p_val_adj", "pct_in": "pct.1", "pct_rest": "pct.2", "pvals": "p_val"}
        )
        selected = marker_genes.loc[
            (marker_genes[["pct_in", "pct_rest"]].max(axis=1) >= 0.10)
            & (marker_genes["log2fc"] >= 0.25) & (marker_genes["pvals_adj"] < 0.01)
        ].copy()
        selected = selected.groupby("group", sort=False).head(top_n).copy()
        selected["rank"] = selected.groupby("group", sort=False).cumcount() + 1
        return ExpressionQuery(marker_genes=selected, evidence=evidence.loc[:, CASSIA_MARKER_COLUMNS])

    def _generate_gptcelltype_query(self, *, marker_genes: Any, **kwargs: Any) -> ExpressionQuery:
        if kwargs:
            raise TypeError("gptcelltype does not accept expression export options")
        return ExpressionQuery(marker_genes=marker_genes, evidence=marker_genes.loc[:, ["gene"]].copy())

    def _generate_anndict_query(self, *, marker_genes: Any, **kwargs: Any) -> ExpressionQuery:
        """Expose ordered marker genes through AnnDictionary's explicit profile.

        AnnDictionary consumes the same label-free marker lists as other
        marker-only callers, but retains a dedicated generator so its
        benchmark protocol is explicit rather than coupled to GPTCelltype.
        """
        if kwargs:
            raise TypeError("anndict does not accept expression export options")
        required = {"group", "rank", "gene"}
        missing = required.difference(marker_genes.columns)
        if missing:
            raise ValueError("AnnDictionary marker table lacks required columns: " + ", ".join(sorted(missing)))
        return ExpressionQuery(marker_genes=marker_genes, evidence=marker_genes.loc[:, ["gene"]].copy())

    def _generate_cellclick_v4_query(
        self,
        *,
        marker_genes: Any,
        groupby: str,
        output_path: str | Path,
        source_row_id_by_group: Mapping[str, str],
        manifest_path: str | Path | None = None,
        manifest_schema_version: str = "cellclick.expression.2",
    ) -> ExpressionQuery:
        """Save CellClick V4's label-free queryable AnnData and manifest.

        V4 deliberately stores the selected AnnData instead of materializing
        a full cluster-by-gene statistics table.  Later trusted tool calls can
        calculate statistics only for genes requested during verification.
        ``source_row_id_by_group`` binds the marker-ranking groups to the
        benchmark rows without deriving IDs from annotations.
        """
        required = {"group", "rank", "gene"}
        missing = required.difference(marker_genes.columns)
        if missing:
            raise ValueError("CellClick V4 marker table lacks required columns: " + ", ".join(sorted(missing)))
        if groupby not in self.adata.obs:
            raise KeyError(f"adata.obs lacks requested grouping column {groupby!r}")
        if not isinstance(source_row_id_by_group, Mapping):
            raise TypeError("source_row_id_by_group must be a mapping")
        if not isinstance(manifest_schema_version, str) or not manifest_schema_version.strip():
            raise ValueError("manifest_schema_version must be a non-empty string")

        groups = sorted({str(group) for group in marker_genes["group"]})
        normalized_mapping = {str(group): str(source_row_id).strip() for group, source_row_id in source_row_id_by_group.items()}
        if set(normalized_mapping) != set(groups):
            raise ValueError("CellClick V4 source-row mapping must exactly match marker groups")
        if any(not source_row_id for source_row_id in normalized_mapping.values()):
            raise ValueError("CellClick V4 source-row mapping contains an empty source_row_id")
        if len(set(normalized_mapping.values())) != len(normalized_mapping):
            raise ValueError("CellClick V4 source-row mapping contains duplicate source_row_id values")

        try:
            import anndata as ad
            import pandas as pd
        except (ImportError, ValueError) as error:
            raise RuntimeError("CellClick V4 expression export requires anndata and pandas") from error

        output_ids = self._marker_output_ids(groupby)
        source_group_by_marker_group: dict[str, str] = {}
        for marker_group in groups:
            if marker_group in output_ids:
                source_group_by_marker_group[marker_group] = marker_group
                continue
            source_groups = [source for source, output_id in output_ids.items() if output_id == marker_group]
            if len(source_groups) != 1:
                raise ValueError(f"CellClick V4 marker group {marker_group!r} cannot be bound to {groupby!r}")
            source_group_by_marker_group[marker_group] = source_groups[0]
        if set(source_group_by_marker_group.values()) != set(output_ids):
            raise ValueError("CellClick V4 marker groups do not cover the submitted expression groups")

        anonymous_by_marker_group = {
            marker_group: f"G{index:03d}" for index, marker_group in enumerate(sorted(groups), start=1)
        }
        source_labels = self.adata.obs[groupby].astype(str)
        group_to_anonymous = {
            source_group_by_marker_group[marker_group]: anonymous_by_marker_group[marker_group]
            for marker_group in groups
        }
        anonymous_labels = source_labels.map(group_to_anonymous)
        if anonymous_labels.isna().any():
            raise ValueError("CellClick V4 could not anonymize every expression cell group")
        statistics = self.adata.layers[NORMALIZED_LOG_LAYER]
        cell_ids = [f"cell_{index:09d}" for index in range(self.adata.n_obs)]
        export = ad.AnnData(
            X=statistics.copy(),
            obs=pd.DataFrame({"cellclick_group_id": anonymous_labels.to_numpy()}, index=cell_ids),
            var=pd.DataFrame(index=self.adata.var_names.astype(str).copy()),
        )
        export.layers[NORMALIZED_LOG_LAYER] = statistics.copy()
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        export.write_h5ad(destination)
        resolved_destination = destination.resolve()
        resolved_manifest = (
            Path(manifest_path) if manifest_path is not None
            else destination.with_suffix(".manifest.json")
        )
        resolved_manifest.parent.mkdir(parents=True, exist_ok=True)

        initial_markers: dict[str, list[str]] = {}
        for group, group_rows in marker_genes.groupby("group", sort=True):
            source_row_id = normalized_mapping[str(group)]
            initial_markers[source_row_id] = (
                group_rows.sort_values("rank", kind="stable")["gene"].astype(str).tolist()
            )
        manifest = {
            "schema_version": manifest_schema_version.strip(),
            "h5ad_path": str(resolved_destination),
            "source_h5ad_path": str(Path(self._source_data).resolve()) if isinstance(self._source_data, (str, Path)) else None,
            "source_h5ad_sha256": _sha256_file(self._source_data) if isinstance(self._source_data, (str, Path)) else None,
            "groupby": "cellclick_group_id",
            "statistics_layer": NORMALIZED_LOG_LAYER,
            "source_row_id_by_group": {
                anonymous_by_marker_group[marker_group]: normalized_mapping[marker_group]
                for marker_group in sorted(groups)
            },
            "initial_markers_by_source_row_id": initial_markers,
        }
        resolved_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return ExpressionQuery(marker_genes=marker_genes, evidence=manifest)

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
