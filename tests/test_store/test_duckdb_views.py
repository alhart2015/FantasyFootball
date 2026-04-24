"""DuckDB view layer — query parquet partitions as SQL tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.store import query, write_partition


def _seed(root: Path) -> None:
    df1 = pd.DataFrame({"gsis_id": ["00-0036322"], "season": [2024], "week": [1], "mean": [18.0]})
    df2 = pd.DataFrame({"gsis_id": ["00-0036322"], "season": [2024], "week": [2], "mean": [22.5]})
    write_partition(root, "projections_weekly", df1, season=2024, week=1)
    write_partition(root, "projections_weekly", df2, season=2024, week=2)


def test_query_combines_partitions(tmp_path: Path) -> None:
    _seed(tmp_path)
    out = query(
        tmp_path,
        "SELECT week, mean FROM projections_weekly ORDER BY week",
    )
    assert out["week"].tolist() == [1, 2]
    assert out["mean"].tolist() == [18.0, 22.5]


def test_query_handles_missing_table(tmp_path: Path) -> None:
    import duckdb
    import pytest

    with pytest.raises(duckdb.Error):
        query(tmp_path, "SELECT * FROM no_such_table")
