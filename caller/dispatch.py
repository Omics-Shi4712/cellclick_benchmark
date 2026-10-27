#!/usr/bin/env python3
"""Execute one configured caller inside its declared Conda environment."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from caller.registry import AVAILABLE_CALLERS, caller_spec, normalize_caller_name


AVAILABLE_DATA_ADAPTERS = frozenset({"expressionAdapter", "tableAdapter"})


def run_task(task: dict[str, Any]) -> None:
    implementation = task.get("caller") or task.get("method")
    canonical = normalize_caller_name(implementation)
    data_adapter = task.get("data_adapter")
    if data_adapter not in AVAILABLE_DATA_ADAPTERS:
        raise ValueError(
            f"Unsupported data_adapter {data_adapter!r}; supported: {', '.join(sorted(AVAILABLE_DATA_ADAPTERS))}"
        )
    module = importlib.import_module(caller_spec(canonical)["module"])
    action = task.get("action", "annotate")
    if action == "annotate":
        module.run_task(task)
    elif action == "score":
        scorer = getattr(module, "run_score_task", None)
        if scorer is None:
            raise ValueError(f"Caller {implementation!r} does not support score tasks")
        scorer(task)
    else:
        raise ValueError(f"Unsupported caller action {action!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    args = parser.parse_args()
    run_task(json.loads(args.task.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
