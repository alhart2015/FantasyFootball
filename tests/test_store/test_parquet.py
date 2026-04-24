"""Parquet store tests — partitioned writes/reads, idempotency."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.store import read_partition, write_partition


def _frame(season: int, week: int, n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": [f"00-00{i:05d}" for i in range(n)],
            "season": [season] * n,
            "week": [week] * n,
            "value": [float(i) for i in range(n)],
        }
    )


def test_write_and_read_single_partition(tmp_path: Path) -> None:
    df = _frame(2024, 3)
    write_partition(tmp_path, "weekly_stats", df, season=2024, week=3)
    out = read_partition(tmp_path, "weekly_stats", season=2024, week=3)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), df.reset_index(drop=True), check_like=True
    )


def test_write_idempotent_overwrites_partition(tmp_path: Path) -> None:
    write_partition(tmp_path, "weekly_stats", _frame(2024, 3, n=5), season=2024, week=3)
    write_partition(tmp_path, "weekly_stats", _frame(2024, 3, n=2), season=2024, week=3)
    out = read_partition(tmp_path, "weekly_stats", season=2024, week=3)
    assert len(out) == 2  # second write replaces first


def test_read_across_weeks(tmp_path: Path) -> None:
    write_partition(tmp_path, "weekly_stats", _frame(2024, 1), season=2024, week=1)
    write_partition(tmp_path, "weekly_stats", _frame(2024, 2), season=2024, week=2)
    write_partition(tmp_path, "weekly_stats", _frame(2024, 3), season=2024, week=3)
    all_2024 = read_partition(tmp_path, "weekly_stats", season=2024)
    assert sorted(all_2024["week"].unique().tolist()) == [1, 2, 3]


def test_season_only_partition(tmp_path: Path) -> None:
    df = pd.DataFrame({"gsis_id": ["00-0036322"], "espn_id": ["4262921"]})
    write_partition(tmp_path, "id_map", df, season=None, week=None)
    out = read_partition(tmp_path, "id_map", season=None, week=None)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), df.reset_index(drop=True), check_like=True
    )
