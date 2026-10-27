"""Adapt CellClick V7.3 to the benchmark prediction contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "cellclick"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def run_task(task):
    from cellclick.v7_3 import run_task as run_v7_3
    run_v7_3(task)

    # The upstream V7.3 runner records an internal reference-selection field.
    # Keep the parent runner's protocol manifest limited to its public contract.
    manifest_path = Path(task["work_dir"]) / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("reference_selection_policy", None)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
