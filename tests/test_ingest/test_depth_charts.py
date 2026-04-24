"""Depth chart ingest tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.ingest import refresh_depth_charts
from projections.ingest.depth_charts import _parse_depth_rank
from projections.schemas import DepthChartsSchema
from projections.store import read_partition


def test_refresh_depth_charts_writes_partition(
    tmp_path: Path,
    fake_depth_charts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: fake_depth_charts_df,
    )
    written = refresh_depth_charts(tmp_path, seasons=[2024])
    assert len(written) == 1

    df = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    DepthChartsSchema.validate(df)
    assert set(df["gsis_id"]) == {"00-0036322", "00-0034857", "00-0034796"}


def test_refresh_depth_charts_renames_club_code_to_team(
    tmp_path: Path,
    fake_depth_charts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: fake_depth_charts_df,
    )
    refresh_depth_charts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    assert "team" in df.columns
    assert "club_code" not in df.columns
    assert set(df["team"]) == {"MIN", "KC", "PHI"}


def test_refresh_depth_charts_filters_unsupported_positions(
    tmp_path: Path,
    fake_depth_charts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OL, IDP positions are dropped — depth_charts also lists non-fantasy positions."""
    extra = pd.DataFrame(
        [
            {
                "season": 2024,
                "club_code": "MIN",
                "week": 3,
                "depth_team": "LT1",
                "last_name": "Doe",
                "first_name": "John",
                "formation": "Offense",
                "gsis_id": "00-0099998",
                "jersey_number": 71,
                "position": "OL",
                "elias_id": "DOE99998",
                "depth_position": 1,
                "football_name": "John Doe",
            }
        ]
    )
    with_ol = pd.concat([fake_depth_charts_df, extra], ignore_index=True)
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: with_ol,
    )
    refresh_depth_charts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    assert "00-0099998" not in df["gsis_id"].tolist()


def test_refresh_depth_charts_idempotent(
    tmp_path: Path,
    fake_depth_charts_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.depth_charts._fetch_raw_depth_charts",
        lambda seasons: fake_depth_charts_df,
    )
    refresh_depth_charts(tmp_path, seasons=[2024])
    refresh_depth_charts(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "depth_charts", season=2024)
    assert len(df) == 3


# --- _parse_depth_rank unit tests ---


def test_parse_depth_rank_prefers_numeric_depth_position() -> None:
    """If depth_position is a non-null int, that's the rank."""
    rank, warned = _parse_depth_rank(depth_team="LWR", depth_position=2)
    assert rank == 2
    assert warned is False


def test_parse_depth_rank_parses_trailing_digit_from_depth_team() -> None:
    """Falls back to the trailing digit in depth_team when depth_position is null."""
    rank, warned = _parse_depth_rank(depth_team="WR3", depth_position=None)
    assert rank == 3
    assert warned is False


def test_parse_depth_rank_falls_back_to_one_for_unrankable_label() -> None:
    """Unrankable label (no trailing digit, no depth_position) → 1, warned."""
    rank, warned = _parse_depth_rank(depth_team="LWR", depth_position=None)
    assert rank == 1
    assert warned is True


def test_parse_depth_rank_clamps_above_ten() -> None:
    """Parsed rank > 10 (impossible per schema) clamps to 10."""
    rank, warned = _parse_depth_rank(depth_team="WR99", depth_position=None)
    assert rank == 10
    assert warned is True
