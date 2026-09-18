"""CellTypeGPT annotation agreement score.

This reproduces the pairwise scoring rules in
``src/GPTCelltype_Paper/anno/code/gpt4topgenenumber.R`` using the repository
copies of ``compiled.csv`` and ``relation.csv``.  Inputs are annotation labels
from the ``originalname`` column, not CL IDs.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, Optional


DEFAULT_REF_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "ref_data"
DEFAULT_COMPILED_PATH: Final[Path] = DEFAULT_REF_DATA_DIR / "compiled.csv"
DEFAULT_RELATION_PATH: Final[Path] = DEFAULT_REF_DATA_DIR / "relation.csv"
DATASET_NAME_ALIASES: Final[dict[str, str]] = {
    "HCA": "GTEx",
    "tabulasapiens": "TS",
}


@dataclass(frozen=True)
class CompiledTerm:
    """One normalized row from CellTypeGPT's ``compiled.csv`` table."""

    original_name: str
    cl_names: tuple[str, ...]
    cl_ids: tuple[str, ...]
    broad_types: tuple[str, ...]

    @classmethod
    def from_reference_fields(
        cls,
        annotation: object,
        cl_name: Optional[str],
        cl_id: Optional[str],
        broad_type: Optional[str],
    ) -> "CompiledTerm":
        """Create a term from the columns in CellTypeGPT's result table.

        This is useful when a caller already has the ``annotation``, ``CLname``,
        ``CLID``, and ``broadtype`` values supplied by CellTypeGPT, avoiding a
        second lookup through ``compiled.csv``.
        """
        original_name = _normalise_label(annotation)
        if original_name is None:
            raise ValueError("annotation must be a non-empty string")
        return cls(
            original_name=original_name,
            cl_names=_split_r_cell(cl_name),
            cl_ids=_split_r_cell(cl_id),
            broad_types=_split_r_cell(broad_type),
        )


def _normalise_label(value: object) -> Optional[str]:
    """Match the R lookup expression: ``sub('cells', 'cell', tolower(x))``."""
    if value is None:
        return None
    raw_label = str(value)
    # read.csv(..., na.strings = "NA") is the reference script's default.
    if raw_label == "NA":
        return None
    label = raw_label.lower().replace("cells", "cell", 1)
    return label if label else None


def _compiled_lookup_key(value: object) -> Optional[str]:
    """Return an ``originalname`` key exactly as R's ``match`` uses it.

    The R script normalises the queried annotation but does not normalise
    ``cl[, 1]`` (the ``compiled.csv`` ``originalname`` column).  In particular,
    plural tokens beyond the first occurrence must remain unchanged here.
    """
    if value is None:
        return None
    label = str(value)
    if label == "NA":
        return None
    return label.lower() or None


def _split_r_cell(value: Optional[str]) -> tuple[str, ...]:
    """Split a CSV field as used by the reference R code.

    In R, a field whose complete value is ``NA`` is read as a missing value;
    an ``NA`` token embedded in a comma-separated field remains a string.
    """
    if value is None or value == "" or value == "NA":
        return ()
    return tuple(value.split(","))


class CellTypeGPTScorer:
    """Score label pairs using configurable copies of the two reference CSVs.

    Parameters default to the pinned files in ``evaluator/ref_data``.  Passing
    explicit paths makes the scorer suitable for an alternate fixed reference
    table without changing global state.
    """

    def __init__(
        self,
        compiled_path: str | Path = DEFAULT_COMPILED_PATH,
        relation_path: str | Path = DEFAULT_RELATION_PATH,
    ) -> None:
        self.compiled_path = Path(compiled_path)
        self.relation_path = Path(relation_path)
        self.compiled_terms = self._load_compiled(self.compiled_path)
        self.related_broad_type = self._load_relations(self.relation_path)

    @staticmethod
    def _load_compiled(path: Path) -> dict[str, CompiledTerm]:
        if not path.is_file():
            raise FileNotFoundError(f"CellTypeGPT compiled table not found: {path}")

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"originalname", "Clname", "CLID", "type"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path} must contain columns: {sorted(required)}")

            compiled: dict[str, CompiledTerm] = {}
            for row in reader:
                original_name = _compiled_lookup_key(row["originalname"])
                if original_name is None:
                    continue
                # The R implementation uses match(), which resolves a lookup
                # key collision to the first row.
                if original_name in compiled:
                    continue
                compiled[original_name] = CompiledTerm(
                    original_name=original_name,
                    cl_names=_split_r_cell(row["Clname"]),
                    cl_ids=_split_r_cell(row["CLID"]),
                    broad_types=_split_r_cell(row["type"]),
                )
        return compiled

    @staticmethod
    def _load_relations(path: Path) -> dict[str, str]:
        """Load the R script's one-hop, bidirectional relation lookup.

        The reference constructs a named R vector after appending reverse rows.
        A name occurring multiple times resolves to its first value, which is
        preserved with ``setdefault`` below; this is deliberately not a
        transitive ontology traversal.
        """
        if not path.is_file():
            raise FileNotFoundError(f"CellTypeGPT relation table not found: {path}")

        rows: list[tuple[str, str]] = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) != 2:
                    raise ValueError(f"Each row in {path} must have exactly two columns.")
                rows.append((row[0], row[1]))

        relation_lookup: dict[str, str] = {}
        for source, target in [*rows, *((target, source) for source, target in rows)]:
            relation_lookup.setdefault(target, source)
        return relation_lookup

    def _term(self, label: object) -> Optional[CompiledTerm]:
        if isinstance(label, CompiledTerm):
            return label
        normalised = _normalise_label(label)
        return self.compiled_terms.get(normalised) if normalised is not None else None

    def score_one(self, term_a: object, term_b: object) -> Optional[float]:
        """Return the CellTypeGPT pair score: ``0.0``, ``0.5``, or ``1.0``.

        Either input may be an annotation label to look up in ``compiled.csv``
        or a :class:`CompiledTerm` made from pre-mapped reference fields.  A
        ``None`` input returns ``None``.  An empty annotation string scores
        ``0.0`` because the reference TSV/R workflow treats it as a submitted
        but unmapped label in some result columns.
        """
        if term_a is None or term_b is None:
            return None
        if (
            not isinstance(term_a, CompiledTerm)
            and _normalise_label(term_a) is None
        ) or (
            not isinstance(term_b, CompiledTerm)
            and _normalise_label(term_b) is None
        ):
            return None
        entry_a = self._term(term_a)
        entry_b = self._term(term_b)

        broad_a = set(entry_a.broad_types) if entry_a else set()
        broad_b = set(entry_b.broad_types) if entry_b else set()
        expanded_a = broad_a | {
            self.related_broad_type[name]
            for name in broad_a
            if name in self.related_broad_type
        }
        expanded_b = broad_b | {
            self.related_broad_type[name]
            for name in broad_b
            if name in self.related_broad_type
        }
        partial_score = bool(expanded_a & expanded_b)

        cl_names_a = set(entry_a.cl_names) if entry_a else set()
        cl_names_b = set(entry_b.cl_names) if entry_b else set()
        same_cl_names = bool(cl_names_a) and cl_names_a == cl_names_b

        label_a = entry_a.original_name if entry_a else _normalise_label(term_a)
        label_b = entry_b.original_name if entry_b else _normalise_label(term_b)
        same_label = (
            label_a is not None
            and label_b is not None
            and label_a.removesuffix("s") == label_b.removesuffix("s")
        )
        both_malignant = (
            entry_a is not None
            and entry_b is not None
            and entry_a.broad_types == ("malignant cell",)
            and entry_b.broad_types == ("malignant cell",)
        )
        full_score = same_cl_names or same_label or both_malignant

        if full_score:
            return 1.0
        if partial_score:
            return 0.5
        return 0.0

    def score_by_dataset(
        self,
        terms_a: Sequence[object],
        terms_b: Sequence[object],
        datasets: Sequence[str],
    ) -> dict[str, Optional[float]]:
        """Return CellTypeGPT macro-average scores grouped by dataset.

        This reproduces ``analysis/compareperf.R``: each row receives the
        pairwise score and every dataset's score is the arithmetic mean across
        its cell-type populations.  Hence every row has equal weight; cell
        counts are not used.  If any score in a dataset is ``None``, that
        dataset's result is ``None`` because R's ``mean()`` is called without
        ``na.rm = TRUE``.  Output keys follow R's dataset aliases ``HCA`` →
        ``GTEx`` and ``tabulasapiens`` → ``TS``.
        """
        if len(terms_a) != len(terms_b) or len(terms_a) != len(datasets):
            raise ValueError(
                "terms_a, terms_b, and datasets must have the same length; got "
                f"{len(terms_a)}, {len(terms_b)}, and {len(datasets)}."
            )

        grouped_scores: dict[str, list[Optional[float]]] = {}
        for term_a, term_b, dataset in zip(terms_a, terms_b, datasets):
            if not isinstance(dataset, str) or not dataset:
                raise ValueError("Each dataset must be a non-empty string.")
            grouped_scores.setdefault(dataset, []).append(self.score_one(term_a, term_b))

        results: dict[str, Optional[float]] = {}
        for dataset in sorted(grouped_scores):
            scores = grouped_scores[dataset]
            name = DATASET_NAME_ALIASES.get(dataset, dataset)
            if name in results:
                raise ValueError(f"Dataset name collision after aliasing: {name!r}")
            if any(score is None for score in scores):
                results[name] = None
            else:
                results[name] = sum(
                    score for score in scores if score is not None
                ) / len(scores)
        return results


@lru_cache(maxsize=1)
def default_scorer() -> CellTypeGPTScorer:
    """Return the cached scorer backed by the repository's pinned CSV files."""
    return CellTypeGPTScorer()


def evaluate_one(termA: object, termB: object) -> Optional[float]:
    """Score one CellTypeGPT annotation pair using the default reference data."""
    return default_scorer().score_one(termA, termB)


def evaluate(
    listA: Sequence[object],
    listB: Sequence[object],
    datasets: Sequence[str],
) -> dict[str, Optional[float]]:
    """Return CellTypeGPT's dataset-level scores using pinned reference data."""
    return default_scorer().score_by_dataset(listA, listB, datasets)
