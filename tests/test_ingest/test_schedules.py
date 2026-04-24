"""Schedule ingest tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import refresh_schedules
from projections.schemas import SchedulesSchema
from projections.store import read_partition


def test_refresh_schedules_writes_partition(
    tmp_path: Path,
    fake_schedules_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.schedules._fetch_raw_schedules",
        lambda seasons: fake_schedules_df,
    )
    written = refresh_schedules(tmp_path, seasons=[2024])
    assert len(written) == 1

    df = read_partition(tmp_path / "raw", "schedules", season=2024)
    SchedulesSchema.validate(df)
    assert set(df["game_id"]) == {"2024_03_KC_ATL", "2024_03_MIN_HOU"}


def test_refresh_schedules_constructs_kickoff_from_gameday_and_gametime(
    tmp_path: Path,
    fake_schedules_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.schedules._fetch_raw_schedules",
        lambda seasons: fake_schedules_df,
    )
    refresh_schedules(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "schedules", season=2024)
    # Kickoff combined from "2024-09-22" + "20:20" -> 2024-09-22 20:20 UTC
    kc_atl = df[df["game_id"] == "2024_03_KC_ATL"].iloc[0]
    assert pd.Timestamp(kc_atl["kickoff"]) == pd.Timestamp("2024-09-22 20:20:00", tz="UTC")


def test_refresh_schedules_normalizes_team_codes(
    tmp_path: Path,
    fake_schedules_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliased = fake_schedules_df.copy()
    aliased.loc[0, "home_team"] = "JAX"  # alias for JAC
    aliased.loc[1, "away_team"] = "LA"  # alias for LAR
    monkeypatch.setattr(
        "projections.ingest.schedules._fetch_raw_schedules",
        lambda seasons: aliased,
    )
    refresh_schedules(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "schedules", season=2024)
    assert "JAX" not in df["home_team"].tolist()
    assert "JAC" in df["home_team"].tolist()
    assert "LAR" in df["away_team"].tolist()


def test_refresh_schedules_idempotent(
    tmp_path: Path,
    fake_schedules_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.schedules._fetch_raw_schedules",
        lambda seasons: fake_schedules_df,
    )
    refresh_schedules(tmp_path, seasons=[2024])
    refresh_schedules(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "schedules", season=2024)
    assert len(df) == 2  # not 4


def test_refresh_schedules_allows_nullable_lines(
    tmp_path: Path,
    fake_schedules_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future-week games may have NaN spread/total/temp/wind."""
    nan_lines = fake_schedules_df.copy()
    nan_lines.loc[0, "spread_line"] = pd.NA
    nan_lines.loc[0, "total_line"] = pd.NA
    nan_lines.loc[0, "temp"] = pd.NA
    nan_lines.loc[0, "wind"] = pd.NA
    monkeypatch.setattr(
        "projections.ingest.schedules._fetch_raw_schedules",
        lambda seasons: nan_lines,
    )
    refresh_schedules(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "schedules", season=2024)
    SchedulesSchema.validate(df)
    assert pd.isna(df.iloc[0]["spread_line"])
