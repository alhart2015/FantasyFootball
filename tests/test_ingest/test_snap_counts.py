"""Snap counts ingest tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import build_id_map, refresh_snap_counts
from projections.schemas import SnapCountsSchema
from projections.store import read_partition


def _setup_id_map(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Helper: populate the id_map partition that snap_counts ingest will join on."""
    monkeypatch.setattr(
        "projections.ingest.id_map._fetch_raw_id_map",
        lambda: fake_id_map_df,
    )
    build_id_map(tmp_path)


def test_refresh_snap_counts_writes_partition(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    fake_snap_counts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)
    monkeypatch.setattr(
        "projections.ingest.snap_counts._fetch_raw_snap_counts",
        lambda seasons: fake_snap_counts_df,
    )
    written = refresh_snap_counts(tmp_path, seasons=[2024])
    assert len(written) == 1

    df = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    SnapCountsSchema.validate(df)
    # 3 rows in fixture, all match id_map; all should survive the join.
    assert set(df["gsis_id"]) == {"00-0034857", "00-0036322", "00-0030506"}


def test_refresh_snap_counts_resolves_pfr_to_gsis(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    fake_snap_counts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pfr_player_id column from raw must be resolved to gsis_id via id_map."""
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)
    monkeypatch.setattr(
        "projections.ingest.snap_counts._fetch_raw_snap_counts",
        lambda seasons: fake_snap_counts_df,
    )
    refresh_snap_counts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    assert "gsis_id" in df.columns
    assert "pfr_player_id" not in df.columns
    # Mahomes pfr_id MahoPa00 -> gsis 00-0034857
    mahomes = df[df["gsis_id"] == "00-0034857"].iloc[0]
    assert mahomes["team"] == "KC"


def test_refresh_snap_counts_drops_rows_unmatched_in_id_map(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    fake_snap_counts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bench/practice players (not in id_map) get dropped silently."""
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)

    extra_row = pd.DataFrame(
        [
            {
                "game_id": "2024_03_MIN_HOU",
                "season": 2024,
                "week": 3,
                "player": "Some Bench Guy",
                "position": "WR",
                "team": "MIN",
                "opponent": "HOU",
                "offense_snaps": 5,
                "offense_pct": 0.08,
                "defense_snaps": 0,
                "defense_pct": 0.0,
                "st_snaps": 1,
                "st_pct": 0.03,
                "pfr_player_id": "UnknownXXXX",  # not in fake_id_map
            }
        ]
    )
    with_unknown = pd.concat([fake_snap_counts_df, extra_row], ignore_index=True)

    monkeypatch.setattr(
        "projections.ingest.snap_counts._fetch_raw_snap_counts",
        lambda seasons: with_unknown,
    )
    refresh_snap_counts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    # 3 known players survive; bench guy with no id_map match is dropped.
    assert len(df) == 3


def test_refresh_snap_counts_normalizes_team_codes(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    fake_snap_counts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)
    aliased = fake_snap_counts_df.copy()
    aliased.loc[0, "team"] = "OAK"  # historical alias for LV
    aliased.loc[1, "opponent"] = "WSH"  # historical alias for WAS
    monkeypatch.setattr(
        "projections.ingest.snap_counts._fetch_raw_snap_counts",
        lambda seasons: aliased,
    )
    refresh_snap_counts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    assert "OAK" not in df["team"].tolist()
    assert "LV" in df["team"].tolist()
    assert "WAS" in df["opponent"].tolist()


def test_refresh_snap_counts_filters_unsupported_positions(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    fake_snap_counts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)
    # Add a row with a position that's not in the Position enum (e.g., OL).
    # Use a pfr_id that IS in fake_id_map (BarkSa00) so the row survives the
    # id_map join; the position filter is what should drop it.
    ol_row = pd.DataFrame(
        [
            {
                "game_id": "2024_03_MIN_HOU",
                "season": 2024,
                "week": 3,
                "player": "Some Lineman",
                "position": "OL",
                "team": "MIN",
                "opponent": "HOU",
                "offense_snaps": 70,
                "offense_pct": 1.0,
                "defense_snaps": 0,
                "defense_pct": 0.0,
                "st_snaps": 0,
                "st_pct": 0.0,
                "pfr_player_id": "BarkSa00",
            }
        ]
    )
    with_ol = pd.concat([fake_snap_counts_df, ol_row], ignore_index=True)
    monkeypatch.setattr(
        "projections.ingest.snap_counts._fetch_raw_snap_counts",
        lambda seasons: with_ol,
    )
    refresh_snap_counts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    # OL row was dropped because its position isn't in Position enum.
    assert "OL" not in df["position"].tolist()


def test_refresh_snap_counts_idempotent(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    fake_snap_counts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)
    monkeypatch.setattr(
        "projections.ingest.snap_counts._fetch_raw_snap_counts",
        lambda seasons: fake_snap_counts_df,
    )
    refresh_snap_counts(tmp_path, seasons=[2024])
    refresh_snap_counts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    assert len(df) == 3


def test_refresh_snap_counts_fills_nan_pct_columns(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    fake_snap_counts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nfl_data_py returns NaN for *_pct when team total for that side is 0
    (e.g., offensive player on a defense-only snap-count row). Schema requires
    non-nullable floats, so ingest must fillna(0.0)."""
    _setup_id_map(tmp_path, fake_id_map_df, monkeypatch)
    nan_pcts = fake_snap_counts_df.copy()
    nan_pcts.loc[0, "defense_pct"] = float("nan")
    nan_pcts.loc[0, "st_pct"] = float("nan")
    monkeypatch.setattr(
        "projections.ingest.snap_counts._fetch_raw_snap_counts",
        lambda seasons: nan_pcts,
    )
    refresh_snap_counts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "snap_counts", season=2024)
    SnapCountsSchema.validate(df)
    # NaN pcts filled to 0.0 (and validation passes)
    assert df.iloc[0]["defense_pct"] == 0.0
    assert df.iloc[0]["st_pct"] == 0.0
