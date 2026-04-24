"""Weekly-stats ingest tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import refresh_weekly_stats
from projections.schemas import WeeklyStatsSchema
from projections.store import read_partition


def test_refresh_weekly_stats_writes_partitioned_parquet(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: fake_weekly_df,
    )
    written = refresh_weekly_stats(tmp_path, seasons=[2024])
    assert len(written) == 1  # one season => one parquet file (week-level not split here)

    df = read_partition(tmp_path / "raw", "weekly_stats", season=2024)
    WeeklyStatsSchema.validate(df)
    assert set(df["gsis_id"]) == {"00-0036322", "00-0034857"}


def test_refresh_weekly_stats_renames_columns(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: fake_weekly_df,
    )
    refresh_weekly_stats(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "weekly_stats", season=2024)
    assert "gsis_id" in df.columns
    assert "team" in df.columns
    assert "opponent" in df.columns
    assert "player_id" not in df.columns
    assert "recent_team" not in df.columns
    assert "opponent_team" not in df.columns


def test_refresh_weekly_stats_normalizes_team_codes(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliased = fake_weekly_df.copy()
    aliased.loc[0, "recent_team"] = "JAX"   # alias for JAC
    aliased.loc[1, "opponent_team"] = "LA"  # alias for LAR
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: aliased,
    )
    refresh_weekly_stats(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "weekly_stats", season=2024)
    assert "JAX" not in df["team"].tolist()
    assert "JAC" in df["team"].tolist()
    assert "LAR" in df["opponent"].tolist()


def test_refresh_weekly_stats_idempotent(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: fake_weekly_df,
    )
    refresh_weekly_stats(tmp_path, seasons=[2024])
    refresh_weekly_stats(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "weekly_stats", season=2024)
    assert len(df) == 2  # not 4 — second run replaced the partition
