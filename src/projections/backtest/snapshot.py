"""Snapshot file IO + diff for the walk-forward gate.

Plan 3c Phase 4. Snapshot is a JSON list of
{"position", "year", "metric", "value"} entries, sorted lexicographically
by (metric, position, year) so PR diffs stay clean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

_SCHEMA_COLUMNS: tuple[str, ...] = ("position", "year", "metric", "value")


def write_snapshot(metrics: pd.DataFrame, path: Path) -> None:
    """Serialize a long-form metrics DataFrame to JSON, sorted by
    (metric, position, year)."""
    if set(metrics.columns) != set(_SCHEMA_COLUMNS):
        raise ValueError(
            f"metrics must have columns {_SCHEMA_COLUMNS}, got {tuple(metrics.columns)}"
        )
    sorted_df = metrics.sort_values(["metric", "position", "year"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _idx, row in sorted_df.iterrows():
        rows.append(
            {
                "position": str(row["position"]),
                "year": int(row["year"]),
                "metric": str(row["metric"]),
                "value": float(row["value"]),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def read_snapshot(path: Path) -> pd.DataFrame:
    """Load a snapshot JSON file into a long-form metrics DataFrame."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(raw, columns=list(_SCHEMA_COLUMNS))
