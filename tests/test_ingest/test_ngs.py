"""NGS ingest tests — one parameterized module covering passing/rushing/receiving."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import refresh_ngs
from projections.ingest.ngs import NgsStatType
from projections.schemas import (
    NgsPassingSchema,
    NgsReceivingSchema,
    NgsRushingSchema,
)
from projections.store import read_partition

_SCHEMA_FOR = {
    "passing": NgsPassingSchema,
    "rushing": NgsRushingSchema,
    "receiving": NgsReceivingSchema,
}

_FIXTURE_NAME_FOR = {
    "passing": "fake_ngs_passing_df",
    "rushing": "fake_ngs_rushing_df",
    "receiving": "fake_ngs_receiving_df",
}


@pytest.mark.parametrize("stat_type", ["passing", "rushing", "receiving"])
def test_refresh_ngs_writes_distinct_partition_per_stat_type(
    tmp_path: Path,
    stat_type: NgsStatType,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_df = request.getfixturevalue(_FIXTURE_NAME_FOR[stat_type])
    monkeypatch.setattr(
        "projections.ingest.ngs._fetch_raw_ngs",
        lambda st, seasons: fake_df,
    )
    written = refresh_ngs(tmp_path, stat_type=stat_type, seasons=[2024])
    assert len(written) == 1
    assert f"ngs_{stat_type}" in str(written[0])

    table = f"ngs_{stat_type}"
    df = read_partition(tmp_path / "raw", table, season=2024)
    _SCHEMA_FOR[stat_type].validate(df)


@pytest.mark.parametrize("stat_type", ["passing", "rushing", "receiving"])
def test_refresh_ngs_renames_player_gsis_id_to_gsis_id(
    tmp_path: Path,
    stat_type: NgsStatType,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_df = request.getfixturevalue(_FIXTURE_NAME_FOR[stat_type])
    monkeypatch.setattr(
        "projections.ingest.ngs._fetch_raw_ngs",
        lambda st, seasons: fake_df,
    )
    refresh_ngs(tmp_path, stat_type=stat_type, seasons=[2024])
    df = read_partition(tmp_path / "raw", f"ngs_{stat_type}", season=2024)
    assert "gsis_id" in df.columns
    assert "player_gsis_id" not in df.columns
    assert "team" in df.columns
    assert "team_abbr" not in df.columns
    assert "position" in df.columns
    assert "player_position" not in df.columns


def test_refresh_ngs_rejects_unknown_stat_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stat_type must be one of"):
        refresh_ngs(tmp_path, stat_type="kicking", seasons=[2024])  # type: ignore[arg-type]


@pytest.mark.parametrize("stat_type", ["passing", "rushing", "receiving"])
def test_refresh_ngs_idempotent(
    tmp_path: Path,
    stat_type: NgsStatType,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_df = request.getfixturevalue(_FIXTURE_NAME_FOR[stat_type])
    monkeypatch.setattr(
        "projections.ingest.ngs._fetch_raw_ngs",
        lambda st, seasons: fake_df,
    )
    refresh_ngs(tmp_path, stat_type=stat_type, seasons=[2024])
    refresh_ngs(tmp_path, stat_type=stat_type, seasons=[2024])
    df = read_partition(tmp_path / "raw", f"ngs_{stat_type}", season=2024)
    assert len(df) == 1


def test_refresh_ngs_three_stat_types_produce_independent_partitions(
    tmp_path: Path,
    fake_ngs_passing_df: pd.DataFrame,
    fake_ngs_rushing_df: pd.DataFrame,
    fake_ngs_receiving_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fakes = {
        "passing": fake_ngs_passing_df,
        "rushing": fake_ngs_rushing_df,
        "receiving": fake_ngs_receiving_df,
    }
    monkeypatch.setattr(
        "projections.ingest.ngs._fetch_raw_ngs",
        lambda st, seasons: fakes[st],
    )
    for st in ("passing", "rushing", "receiving"):
        refresh_ngs(tmp_path, stat_type=st, seasons=[2024])

    for table, schema in [
        ("ngs_passing", NgsPassingSchema),
        ("ngs_rushing", NgsRushingSchema),
        ("ngs_receiving", NgsReceivingSchema),
    ]:
        df = read_partition(tmp_path / "raw", table, season=2024)
        assert len(df) == 1
        schema.validate(df)
