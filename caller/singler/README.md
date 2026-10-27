# SingleR

The caller uses the TS algorithm from GPTCelltype_Paper: SingleR 1.4.1,
HPCA `label.main`, one tissue per task, per-cell scores summed by population,
and the first reference label for an argmax tie. It does not use query markers,
majority voting, `clusters=`, or perform normalization again.

`ExpressionDataAdapter.generate_query("singler", ...)` exports all source
genes from an explicit raw-count layer, normalized to 10,000 per cell and
transformed with `log2(x + 1)`. Its version-2 HDF5 holds a genes-by-cells CSC
matrix and `cell_id`, `tissue`, and opaque `source_row_id` metadata. Manual
labels stay in adapter/runner evaluation data; the caller only sees group IDs.

Configuration parameters are `hpca_rds` (required; relative to YAML),
`num_workers` (positive integer, default 1), and `save_fullpred` (boolean,
default true). Algorithm defaults and SingleR version are fixed. The Python
wrapper needs only the standard library; R needs SingleR, SummarizedExperiment,
Matrix, rhdf5, jsonlite, and BiocParallel in Conda environment `singler`.

Run from the repository root:

```bash
python runner/run.py runner/experiment/test/singler/TS_V1.yaml --dry-run
python runner/run.py runner/experiment/test/singler/TS_V1.yaml --mode run
```

This configuration selects the existing 24 TS_V1 tissue files, covering 375
populations, rather than the paper's full 471 evaluation populations. Keep
that partial coverage distinction when comparing results. Directory `test`
does not imply `--mode test` support: a frozen-result reproduction module is
not included.

The caller preserves HPCA labels in the seven-column prediction TSV. Runner
validates IDs, joins truth, and scores through `evaluation.type: celltypegpt`,
with equal weight per population. It uses the pinned scorer lookup rules,
including their handling of unmapped labels; no label aliases are invented.

Task work artifacts are `population_predictions.tsv`, `summed_scores.rds`,
`time.rds`, `sessionInfo.txt`, `protocol.json`, and optional `fullpred.rds`.
The protocol records runtime versions, parallelism, and HPCA SHA-256. Runner
fingerprints the reference digest, so replacing an RDS invalidates predictions
on resume. Version-1 expression exports are regenerated on resume.

Offline checks:

```bash
python -m unittest discover -s test -p 'test_singler_caller.py'
conda run -n singler Rscript test/test_singler_bridge.R
```
