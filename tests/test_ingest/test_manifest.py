"""Ingest manifest tests — every refresh records (table, season, fetched_at, rowcount, checksum)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import build_id_map, refresh_weekly_stats
from projections.ingest.manifest import read_manifest


def test_id_map_refresh_records_manifest_entry(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("projections.ingest.id_map._fetch_raw_id_map", lambda: fake_id_map_df)
    build_id_map(tmp_path)
    manifest = read_manifest(tmp_path)
    rows = manifest[manifest["table"] == "id_map"]
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["rowcount"] == 4
    assert pd.isna(row["season"])
    assert isinstance(row["checksum"], str) and len(row["checksum"]) == 64


def test_weekly_refresh_records_one_entry_per_season(
    tmp_path: Path,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly",
        lambda seasons: fake_weekly_df,
    )
    refresh_weekly_stats(tmp_path, seasons=[2023, 2024])
    manifest = read_manifest(tmp_path)
    rows = manifest[manifest["table"] == "weekly_stats"]
    assert sorted(rows["season"].tolist()) == [2023, 2024]
    assert (rows["rowcount"] == 3).all()


def test_re_refresh_replaces_manifest_entry(
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
    manifest = read_manifest(tmp_path)
    rows = manifest[(manifest["table"] == "weekly_stats") & (manifest["season"] == 2024)]
    assert len(rows) == 1  # second run replaced, not appended


def test_manifest_season_dtype_is_nullable_int(
    tmp_path: Path,
    fake_id_map_df: pd.DataFrame,
    fake_weekly_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("projections.ingest.id_map._fetch_raw_id_map", lambda: fake_id_map_df)
    monkeypatch.setattr(
        "projections.ingest.weekly_stats._fetch_raw_weekly", lambda seasons: fake_weekly_df
    )
    build_id_map(tmp_path)  # writes a row with season=NA
    refresh_weekly_stats(tmp_path, seasons=[2024])  # writes a row with season=2024
    manifest = read_manifest(tmp_path)
    assert manifest["season"].dtype == pd.Int64Dtype()
    # Both rows present
    assert len(manifest) == 2
