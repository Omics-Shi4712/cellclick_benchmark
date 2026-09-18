"""Dispatch annotation evaluation to one of the supported tool strategies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final, Optional

from evaluator.scorer.celltypegpt import evaluate_one as _celltypegpt_evaluate_one
from evaluator.scorer.cassia import evaluate as _cassia_evaluate
from evaluator.scorer.cassia import evaluate_one as _cassia_evaluate_one


Score = Optional[float]
OneEvaluator = Callable[[Any, Any], Score]
BatchEvaluator = Callable[[Sequence[Any], Sequence[Any]], list[Score]]

# This is the sole hard-coded allow-list for the ``evaluate_func`` argument.
AVAILABLE_EVALUATE_FUNCS: Final[frozenset[str]] = frozenset(
    {"celltypegpt", "cassia", "cellclick"}
)


def _celltypegpt_one(term_a: Any, term_b: Any) -> Score:
    """Return the CellTypeGPT score of one annotation-label pair."""
    return _celltypegpt_evaluate_one(term_a, term_b)


def _cassia_one(term_a: Any, term_b: Any) -> Score:
    """Return CASSIA's CL-ID score for one annotation pair."""
    return _cassia_evaluate_one(term_a, term_b)


def _cellclick_one(term_a: Any, term_b: Any) -> Score:
    """Placeholder for the CellClick score of one annotation pair."""
    # PSEUDOCODE: use CLSolver to find the nearest common CL ancestor and
    # transform its ontology distance into a CellClick score.
    return None


def _score_each(
    list_a: Sequence[Any], list_b: Sequence[Any], evaluator: OneEvaluator
) -> list[Score]:
    """Shared placeholder batch behaviour: score aligned annotation pairs."""
    if len(list_a) != len(list_b):
        raise ValueError(
            f"listA and listB must have the same length; got {len(list_a)} and {len(list_b)}."
        )
    return [evaluator(term_a, term_b) for term_a, term_b in zip(list_a, list_b)]


def _celltypegpt_batch(list_a: Sequence[Any], list_b: Sequence[Any]) -> list[Score]:
    """Placeholder for aggregate CellTypeGPT evaluation."""
    # PSEUDOCODE: replace with CellTypeGPT's dataset-level aggregation rule.
    return _score_each(list_a, list_b, _celltypegpt_one)


def _cassia_batch(list_a: Sequence[Any], list_b: Sequence[Any]) -> list[Score]:
    """Return CASSIA's scores for aligned CL-ID pairs."""
    return _cassia_evaluate(list_a, list_b)


def _cellclick_batch(list_a: Sequence[Any], list_b: Sequence[Any]) -> list[Score]:
    """Placeholder for aggregate CellClick evaluation."""
    # PSEUDOCODE: replace with CellClick's dataset-level aggregation rule.
    return _score_each(list_a, list_b, _cellclick_one)


# Keep dispatch hard-coded: ``evaluate_func`` is not a callable supplied by a
# caller, but one of the fixed names above.
_EVALUATION_STRATEGIES: Final[dict[str, tuple[OneEvaluator, BatchEvaluator]]] = {
    "celltypegpt": (_celltypegpt_one, _celltypegpt_batch),
    "cassia": (_cassia_one, _cassia_batch),
    "cellclick": (_cellclick_one, _cellclick_batch),
}


def _get_strategy(evaluate_func: str) -> tuple[OneEvaluator, BatchEvaluator]:
    if not isinstance(evaluate_func, str) or evaluate_func not in AVAILABLE_EVALUATE_FUNCS:
        allowed = ", ".join(sorted(AVAILABLE_EVALUATE_FUNCS))
        raise ValueError(
            f"evaluate_func must be one of: {allowed}; got {evaluate_func!r}."
        )
    return _EVALUATION_STRATEGIES[evaluate_func]


def evaluate_one(termA: Any, termB: Any, evaluate_func: str) -> Score:
    """Return the score for one pair under ``evaluate_func``.

    ``evaluate_func`` must be ``celltypegpt``, ``cassia``, or ``cellclick``.
    CASSIA inputs must be CL IDs.  CellClick remains a placeholder.
    """
    one_evaluator, _ = _get_strategy(evaluate_func)
    return one_evaluator(termA, termB)


def evaluate(
    listA: Sequence[Any], listB: Sequence[Any], evaluate_func: str
) -> list[Score]:
    """Return scores for aligned annotation pairs.

    The selected method controls its pairwise scoring function.  CASSIA
    expects CL IDs; CellClick remains a placeholder.
    """
    _, batch_evaluator = _get_strategy(evaluate_func)
    return batch_evaluator(listA, listB)


# Compatibility with the original misspelled public functions.
evalate_one = evaluate_one
evalate = evaluate
