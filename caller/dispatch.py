#!/usr/bin/env python3
"""Execute one configured caller inside its declared Conda environment."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any


AVAILABLE_CALLERS = frozenset({"cassia", "celltypeagent", "gptcelltype"})
AVAILABLE_DATA_ADAPTERS = frozenset({"expressionAdapter", "tableAdapter"})
CALLER_MODULES = {
    "cassia": "caller.cassia.cassia",
    "celltypeagent": "caller.celltypeagent.celltypeagent",
    "gptcelltype": "caller.celltypegpt.gptcelltype",
}


def run_task(task: dict[str, Any]) -> None:
    implementation = task.get("caller") or task.get("method")
    if not isinstance(implementation, str) or implementation not in AVAILABLE_CALLERS:
        raise ValueError(f"Unsupported caller {implementation!r}; supported: {', '.join(sorted(AVAILABLE_CALLERS))}")
    data_adapter = task.get("data_adapter")
    if data_adapter not in AVAILABLE_DATA_ADAPTERS:
        raise ValueError(
            f"Unsupported data_adapter {data_adapter!r}; supported: {', '.join(sorted(AVAILABLE_DATA_ADAPTERS))}"
        )
    module = importlib.import_module(CALLER_MODULES[implementation])
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
