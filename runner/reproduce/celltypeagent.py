"""Validate CellTypeAgent's paper-frozen candidate input before reranking.

The original first-stage LLM output is intentionally frozen.  This check
proves that the adapter-selected marker rows match that artifact one-to-one,
that every top-N candidate group is structurally valid, and that no reference
label is consumed to perform the match.  The upstream reranker remains in
``caller.celltypeagent_frozen`` for an explicit, future model-backed
reproduction execution step.
"""

from __future__ import annotations

from pathlib import Path

from caller import celltypeagent_frozen as frozen
from runner.reproduce.common import (
    read_reference_rows, require_label_free_queries, resolve_reference_file,
    write_comparison_summary, write_selected_rows,
)


def run(config, adapter, queries, callers, config_path, output_dir):
    require_label_free_queries(queries)
    checked = 0
    for name, settings in callers.items():
        if settings.get("implementation", name) != "celltypeagent":
            continue
        source = settings.get("candidate_source")
        if source not in {"frozen", "frozen_reproduction"}:
            raise ValueError("CellTypeAgent test mode requires candidate_source=frozen_reproduction")
        candidate_file = settings.get("candidate_file")
        if not isinstance(candidate_file, str) or not candidate_file:
            raise ValueError("CellTypeAgent frozen reproduction requires candidate_file")
        path = Path(candidate_file)
        if not path.is_absolute():
            path = (config_path.parent / path).resolve()
        provider = frozen.FrozenCandidateProvider(path, int(settings.get("top_n", 3)))
        reference_file = resolve_reference_file(settings, config_path, "celltypeagent")
        reference_rows = read_reference_rows(reference_file)
        required_reference_columns = {"dataset", "tissue", "marker", "final_score"}
        missing_columns = required_reference_columns - set(reference_rows[0])
        if missing_columns:
            raise ValueError(
                "celltypeagent reference_file lacks required columns: "
                f"{', '.join(sorted(missing_columns))}"
            )
        reference_fingerprints = {
            frozen.row_fingerprint(
                row["dataset"], row["tissue"], frozen.canonicalize_markers(row["marker"])
            )
            for row in reference_rows
        }
        if len(reference_fingerprints) != len(reference_rows):
            raise ValueError("celltypeagent reference_file contains duplicate dataset/tissue/marker rows")
        for query in queries:
            rows = [
                {"source_row_id": row.source_row_id, "dataset": row.dataset,
                 "tissue": row.tissue, "marker": row.marker}
                for row in query.rows
            ]
            records = provider.get_candidates(rows)
            if len(records) != len(rows):
                raise ValueError("Frozen CellTypeAgent candidates do not align with adapter queries")
            missing_references = [record.source_row_id for record in records if record.fingerprint not in reference_fingerprints]
            if missing_references:
                raise ValueError(
                    "CellTypeAgent reference_file lacks selected rows: " + ", ".join(missing_references)
                )
            checked += len(records)
        output_file = output_dir / "outputs" / "celltypeagent_selected_rows.tsv"
        write_selected_rows(output_file, queries)
        summary_file = write_comparison_summary(
            output_dir, "celltypeagent", output_file, reference_file, rows=checked,
            reference_rows=len(reference_rows), comparison="dataset/tissue/marker coverage and frozen candidate structure",
        )
        return {"method": "celltypeagent", "rows": checked, "status": "validated",
                "output_file": str(output_file), "reference_file": str(reference_file), "summary_file": str(summary_file)}
    raise ValueError("No CellTypeAgent caller selected")
