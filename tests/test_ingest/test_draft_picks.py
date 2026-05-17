"""Synthetic-fixture tests for the draft_picks ingest module.

No network calls — _fetch_raw_draft_picks is monkey-patched to return a
hand-crafted DataFrame mirroring nfl_data_py.import_draft_picks's output
shape (verified empirically: 36 columns, str gsis_id / pfr_player_id,
float64 age, int32 season/round/pick).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest.draft_picks import _normalize_one_season, refresh_draft_picks
from projections.schemas import DraftPicksSchema
from projections.store import read_partition


def _fake_raw(seasons: list[int]) -> pd.DataFrame:
    rows = []
    for s in seasons:
        rows.append(
            {
                "season": s,
                "round": 1,
                "pick": 10,
                "team": "KC",
                "gsis_id": f"00-003{3000 + s}",
                "pfr_player_id": f"Pl{s}00",
                "pfr_player_name": "Test Player",
                "position": "QB",
                "age": 22.0,
            }
        )
    return pd.DataFrame(rows)


def test_normalize_keeps_canonical_columns() -> None:
    raw = _fake_raw([2022])
    df = _normalize_one_season(raw)
    assert list(df.columns) == [
        "gsis_id",
        "draft_year",
        "draft_round",
        "draft_overall_pick",
        "pfr_id",
        "draft_age",
    ]


def test_normalize_drops_malformed_gsis_id_rows() -> None:
    raw = pd.DataFrame(
        {
            "season": [2022, 2022, 2022],
            "round": [1, 1, 1],
            "pick": [10, 11, 12],
            "team": ["KC", "BUF", "MIA"],
            "gsis_id": ["00-0033000", "malformed", None],
            "pfr_player_id": ["A", "B", "C"],
            "pfr_player_name": ["x", "y", "z"],
            "position": ["QB", "RB", "WR"],
            "age": [22.0, 23.0, 21.0],
        }
    )
    df = _normalize_one_season(raw)
    # Only the well-formed row survives.
    assert len(df) == 1
    assert df.iloc[0]["gsis_id"] == "00-0033000"


def test_normalize_handles_missing_pfr_id() -> None:
    raw = _fake_raw([2022])
    raw["pfr_player_id"] = None
    df = _normalize_one_season(raw)
    assert pd.isna(df.iloc[0]["pfr_id"])


def test_normalize_handles_missing_age() -> None:
    raw = _fake_raw([2022])
    raw["age"] = None
    df = _normalize_one_season(raw)
    assert pd.isna(df.iloc[0]["draft_age"])


def test_normalize_validates_against_schema() -> None:
    raw = _fake_raw([2018, 2019, 2020])
    df = _normalize_one_season(raw)
    DraftPicksSchema.validate(df)


def test_refresh_draft_picks_writes_one_partition_per_season(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_fetch(seasons: list[int]) -> pd.DataFrame:
        return _fake_raw(seasons)

    monkeypatch.setattr("projections.ingest.draft_picks._fetch_raw_draft_picks", fake_fetch)
    written = refresh_draft_picks(tmp_path, seasons=[2018, 2019, 2020])
    assert len(written) == 3
    for s in (2018, 2019, 2020):
        df = read_partition(tmp_path / "raw", "draft_picks", season=s)
        assert len(df) == 1
        DraftPicksSchema.validate(df)


def test_refresh_draft_picks_idempotent_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_fetch(seasons: list[int]) -> pd.DataFrame:
        return _fake_raw(seasons)

    monkeypatch.setattr("projections.ingest.draft_picks._fetch_raw_draft_picks", fake_fetch)
    refresh_draft_picks(tmp_path, seasons=[2018])
    refresh_draft_picks(tmp_path, seasons=[2018])
    df = read_partition(tmp_path / "raw", "draft_picks", season=2018)
    assert len(df) == 1


def test_refresh_draft_picks_empty_seasons_no_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_fetch(seasons: list[int]) -> pd.DataFrame:
        return pd.DataFrame()

    monkeypatch.setattr("projections.ingest.draft_picks._fetch_raw_draft_picks", fake_fetch)
    written = refresh_draft_picks(tmp_path, seasons=[])
    assert written == []


def test_normalize_warns_on_placeholder_gsis_ids(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Mirrors the pre-camp-rookie shape: PFR-style placeholders that nflverse
    # carries for the current draft class until NFL assigns real gsis_ids.
    raw = pd.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "round": [1, 1, 2],
            "pick": [1, 2, 33],
            "team": ["LVR", "NYJ", "CHI"],
            "gsis_id": ["MEN516487", "BailDa02", "00-0033000"],
            "pfr_player_id": ["MendFe00", "BailDa02", "RealOk00"],
            "pfr_player_name": ["Fernando Mendoza", "David Bailey", "Real Veteran"],
            "position": ["QB", "OLB", "QB"],
            "age": [22.0, 23.0, 27.0],
        }
    )
    import logging

    with caplog.at_level(logging.WARNING, logger="projections.ingest.draft_picks"):
        df = _normalize_one_season(raw)
    assert len(df) == 1
    assert df.iloc[0]["gsis_id"] == "00-0033000"
    assert any(
        "filtered 2 row(s) with non-GSIS placeholder ids" in r.message for r in caplog.records
    )
