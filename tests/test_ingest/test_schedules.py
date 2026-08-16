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
    assert set(df["game_id"]) == {
        "2024_03_KC_ATL",
        "2024_03_MIN_HOU",
        "2024_03_PHI_TB",
    }


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
    # `gametime` is ET wall-clock per nfl_data_py. SNF on 2024-09-22 at 8:20pm
    # ET (EDT, UTC-4) is 00:20 UTC the following calendar day.
    kc_atl = df[df["game_id"] == "2024_03_KC_ATL"].iloc[0]
    assert pd.Timestamp(kc_atl["kickoff"]) == pd.Timestamp("2024-09-23 00:20:00", tz="UTC")
    # 1pm ET in September (EDT, UTC-4) is 17:00 UTC same day.
    min_hou = df[df["game_id"] == "2024_03_MIN_HOU"].iloc[0]
    assert pd.Timestamp(min_hou["kickoff"]) == pd.Timestamp("2024-09-22 17:00:00", tz="UTC")


def test_build_kickoff_localizes_et_wall_clock_to_utc() -> None:
    """nfl_data_py publishes gametime as ET wall-clock. _build_kickoff must
    localize to America/New_York then convert to UTC. Catches the regression
    where the old code mis-tagged ET wall-clock as UTC.

    Test scenarios:
    - Sept SNF (EDT, UTC-4): 8:20pm ET = 00:20 UTC next day.
    - Nov MNF (EST, UTC-5): 8:15pm ET = 01:15 UTC next day.
    - Sun 1pm ET in Sept (EDT): 17:00 UTC.
    - Sun 1pm ET in Nov (EST): 18:00 UTC.
    - Missing gameday or gametime -> NaT.
    """
    from projections.ingest.schedules import _build_kickoff

    gameday = pd.Series(
        [
            "2024-09-08",  # Sept SNF (EDT)
            "2024-11-11",  # Nov MNF (EST)
            "2024-09-22",  # Sun 1pm ET in Sept (EDT)
            "2024-11-10",  # Sun 1pm ET in Nov (EST)
            None,  # missing gameday
            "2024-09-08",  # missing gametime
        ]
    )
    gametime = pd.Series(
        [
            "20:20",
            "20:15",
            "13:00",
            "13:00",
            "13:00",
            None,
        ]
    )
    out = _build_kickoff(gameday, gametime)

    assert pd.Timestamp(out.iloc[0]) == pd.Timestamp("2024-09-09 00:20:00", tz="UTC")
    assert pd.Timestamp(out.iloc[1]) == pd.Timestamp("2024-11-12 01:15:00", tz="UTC")
    assert pd.Timestamp(out.iloc[2]) == pd.Timestamp("2024-09-22 17:00:00", tz="UTC")
    assert pd.Timestamp(out.iloc[3]) == pd.Timestamp("2024-11-10 18:00:00", tz="UTC")
    assert pd.isna(out.iloc[4])
    assert pd.isna(out.iloc[5])
    # Output dtype is timezone-aware UTC at us resolution (matches SchedulesSchema).
    assert str(out.dtype) == "datetime64[us, UTC]"


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
    assert len(df) == 3  # not 6


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


def test_refresh_schedules_keeps_scores_and_game_type(
    tmp_path: Path,
    fake_schedules_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scores and game_type survive the round-trip as nullable Int64 / string.

    Unplayed games (PHI@TB in the fixture) carry NA scores rather than NaN
    floats — `result` is deliberately not stored, so downstream code derives the
    margin from these two columns and needs them integral.
    """
    monkeypatch.setattr(
        "projections.ingest.schedules._fetch_raw_schedules",
        lambda seasons: fake_schedules_df,
    )
    refresh_schedules(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "schedules", season=2024)
    SchedulesSchema.validate(df)

    assert df["home_score"].dtype == pd.Int64Dtype()
    assert df["away_score"].dtype == pd.Int64Dtype()
    assert set(df["game_type"]) == {"REG"}

    kc_atl = df[df["game_id"] == "2024_03_KC_ATL"].iloc[0]
    assert kc_atl["home_score"] == 17
    assert kc_atl["away_score"] == 22
    # Margin is derived, never stored.
    assert kc_atl["home_score"] - kc_atl["away_score"] == -5

    unplayed = df[df["game_id"] == "2024_03_PHI_TB"].iloc[0]
    assert pd.isna(unplayed["home_score"])
    assert pd.isna(unplayed["away_score"])


def test_refresh_schedules_moneylines_agree_with_spread_sign(
    tmp_path: Path,
    fake_schedules_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`spread_line > 0` means the home team is favored, so the home moneyline
    must be the shorter (more negative) price. Guards the fixture against
    drifting back to the contradictory values it once held."""
    monkeypatch.setattr(
        "projections.ingest.schedules._fetch_raw_schedules",
        lambda seasons: fake_schedules_df,
    )
    refresh_schedules(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "schedules", season=2024)

    for row in df.itertuples():
        if pd.isna(row.spread_line) or row.spread_line == 0:
            continue
        home_favored = row.spread_line > 0
        assert (row.home_moneyline < row.away_moneyline) == home_favored, (
            f"{row.game_id}: spread_line={row.spread_line} disagrees with "
            f"moneylines {row.home_moneyline}/{row.away_moneyline}"
        )
