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
    aliased.loc[0, "recent_team"] = "JAX"  # alias for JAC
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


def test_refresh_weekly_stats_filters_unsupported_positions(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fb_row = {
        "player_id": "00-0099999",
        "season": 2024,
        "week": 3,
        "position": "FB",
        "recent_team": "MIN",
        "opponent_team": "HOU",
        "passing_yards": 0.0,
        "passing_tds": 0,
        "interceptions": 0,
        "rushing_yards": 5.0,
        "rushing_tds": 0,
        "receptions": 1,
        "receiving_yards": 8.0,
        "receiving_tds": 0,
        "fumbles_lost": 0,
    }
    with_fb = pd.concat([fake_weekly_df, pd.DataFrame([fb_row])], ignore_index=True)
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: with_fb,
    )
    refresh_weekly_stats(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "weekly_stats", season=2024)
    assert "00-0099999" not in df["gsis_id"].tolist()
    assert len(df) == 2


def test_refresh_weekly_stats_persists_new_columns(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Targets, carries, receiving_air_yards must round-trip through ingest."""
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: fake_weekly_df,
    )
    refresh_weekly_stats(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "weekly_stats", season=2024)
    assert "targets" in df.columns
    assert "carries" in df.columns
    assert "receiving_air_yards" in df.columns
    # WR row: 12 targets, 0 carries, 145 air yards
    wr_row = df[df["gsis_id"] == "00-0036322"].iloc[0]
    assert int(wr_row["targets"]) == 12
    assert int(wr_row["carries"]) == 0
    assert float(wr_row["receiving_air_yards"]) == 145.0
    # QB row: 0 targets, 3 carries
    qb_row = df[df["gsis_id"] == "00-0034857"].iloc[0]
    assert int(qb_row["carries"]) == 3
