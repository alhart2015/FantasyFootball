"""Unit tests for src/projections/backtest/snapshot.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from projections.backtest.snapshot import read_snapshot, write_snapshot


def test_write_then_read_roundtrips_a_metrics_df(tmp_path: Path) -> None:
    """write_snapshot serializes a long-form metrics DataFrame; read_snapshot
    returns the same columns + values, sorted by (metric, position, year)."""
    df = pd.DataFrame(
        [
            {"position": "WR", "year": 2024, "metric": "composite_rmse", "value": 6.78},
            {"position": "QB", "year": 2021, "metric": "spearman_topN", "value": 0.928},
        ]
    )
    path = tmp_path / "snap.json"
    write_snapshot(df, path)

    out = read_snapshot(path)
    assert set(out.columns) == {"position", "year", "metric", "value"}
    assert len(out) == 2
    # Sorted by (metric, position, year): composite_rmse-WR-2024 then spearman_topN-QB-2021
    assert out["metric"].tolist() == ["composite_rmse", "spearman_topN"]


def test_write_snapshot_emits_human_readable_json(tmp_path: Path) -> None:
    """The on-disk JSON is a list of objects (not pandas-serialized) with
    a 2-space indent so PR diffs stay clean."""
    df = pd.DataFrame([{"position": "WR", "year": 2024, "metric": "composite_rmse", "value": 6.78}])
    path = tmp_path / "snap.json"
    write_snapshot(df, path)

    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert isinstance(parsed, list)
    assert parsed[0] == {
        "position": "WR",
        "year": 2024,
        "metric": "composite_rmse",
        "value": 6.78,
    }
    # Indented for readability.
    assert "\n  " in raw
