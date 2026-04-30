"""PBP ingest tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import refresh_pbp
from projections.schemas import PbpSchema
from projections.store import read_partition


def test_refresh_pbp_writes_partitioned_parquet(
    tmp_path: Path,
    fake_pbp_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.pbp._fetch_raw_pbp",
        lambda seasons: fake_pbp_df,
    )
    written = refresh_pbp(tmp_path, seasons=[2024])
    assert len(written) == 1

    df = read_partition(tmp_path / "raw", "pbp", season=2024)
    PbpSchema.validate(df)


def test_refresh_pbp_idempotent(
    tmp_path: Path,
    fake_pbp_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.pbp._fetch_raw_pbp",
        lambda seasons: fake_pbp_df,
    )
    refresh_pbp(tmp_path, seasons=[2024])
    n_first = len(read_partition(tmp_path / "raw", "pbp", season=2024))
    refresh_pbp(tmp_path, seasons=[2024])
    n_second = len(read_partition(tmp_path / "raw", "pbp", season=2024))
    assert n_first == n_second


def test_refresh_pbp_normalizes_team_codes(
    tmp_path: Path,
    fake_pbp_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliased = fake_pbp_df.copy()
    # First non-null posteam row gets a JAX alias.
    first_idx = aliased[aliased["posteam"].notna()].index[0]
    aliased.loc[first_idx, "posteam"] = "JAX"
    aliased.loc[first_idx, "defteam"] = "LA"
    monkeypatch.setattr(
        "projections.ingest.pbp._fetch_raw_pbp",
        lambda seasons: aliased,
    )
    refresh_pbp(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "pbp", season=2024)
    assert "JAX" not in set(df["posteam"].dropna())
    assert "JAC" in set(df["posteam"].dropna())
    assert "LAR" in set(df["defteam"].dropna())


def test_refresh_pbp_curates_columns(
    tmp_path: Path,
    fake_pbp_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict='filter' drops upstream columns we didn't keep."""
    polluted = fake_pbp_df.copy()
    polluted["unwanted_extra_column"] = 0.0
    monkeypatch.setattr(
        "projections.ingest.pbp._fetch_raw_pbp",
        lambda seasons: polluted,
    )
    refresh_pbp(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "pbp", season=2024)
    assert "unwanted_extra_column" not in df.columns


def test_refresh_pbp_preserves_no_play_rows(
    tmp_path: Path,
    fake_pbp_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """no_play rows are preserved at ingest; feature-time filters remove them
    where appropriate. This guards against accidentally narrowing the ingest."""
    monkeypatch.setattr(
        "projections.ingest.pbp._fetch_raw_pbp",
        lambda seasons: fake_pbp_df,
    )
    refresh_pbp(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "pbp", season=2024)
    assert (df["play_type"] == "no_play").any()


def test_refresh_pbp_writes_manifest(
    tmp_path: Path,
    fake_pbp_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.pbp._fetch_raw_pbp",
        lambda seasons: fake_pbp_df,
    )
    refresh_pbp(tmp_path, seasons=[2024])
    manifest = pd.read_parquet(tmp_path / "manifests" / "ingest_manifest.parquet")
    assert ((manifest["table"] == "pbp") & (manifest["season"] == 2024)).any()
