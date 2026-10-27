"""Adapt CellClick V7.2 to the benchmark prediction contract."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "cellclick"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def run_task(task):
    from cellclick.v7_2 import run_task as run_v7_2
    run_v7_2(task)
