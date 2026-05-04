"""Unit tests for src/projections/features/cache.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projections.features.cache import read_features
from projections.schemas import Position
from projections.store import write_partition


def _minimal_wr_features_row(week: int) -> dict[str, object]:
    """Construct one fully-populated WrFeaturesSchema row.
    Mirrors the column set used by `tests/test_features/conftest.py` fixtures.
    """
    return {
        "gsis_id": "00-0036322",
        "season": 2024,
        "week": week,
        "position": Position.WR.value,
        "team": "MIN",
        "depth_rank": 1,
        "targets_per_game_l4": 8.0,
        "targets_per_game_std": 1.0,
        "target_share_l4": 0.30,
        "air_yards_share_l4": 0.35,
        "receptions_per_game_l4": 6.0,
        "receiving_yards_per_game_l4": 80.0,
        "receiving_tds_per_game_l4": 0.5,
        "rushing_attempts_per_game_l4": 0.0,
        "rushing_yards_per_game_l4": 0.0,
        "designed_rusher": False,
        "snap_pct_l4": 0.95,
        "avg_separation_std": 2.5,
        "avg_intended_air_yards_std": 12.0,
        "percent_share_intended_air_yards_std": 0.30,
        "avg_yac_above_expectation_std": 0.3,
        "implied_team_total": 24.5,
        "spread": -3.0,
        "is_home": True,
        "roof_dome": False,
        "opp_allowed_wr_fppg_l4": 35.0,
        "opponent": "GB",
        "age": 25.0,
        "is_rookie": 0.0,
        "volume_trend_l4_minus_prior_l4": 0.0,
        "snap_pct_change_l4_vs_prior_l4": 0.0,
    }


def test_read_features_validates_and_returns_concatenated_weeks(
    tmp_path: Path,
) -> None:
    """read_features returns one DataFrame across all available weeks for a
    (position, season) and validates against WrFeaturesSchema."""
    features_root = tmp_path
    for week in (1, 2, 3):
        df = pd.DataFrame([_minimal_wr_features_row(week)])
        write_partition(features_root, "wr", df, season=2024, week=week)

    out = read_features(Position.WR, 2024, features_root=features_root)
    assert len(out) == 3
    assert sorted(out["week"].tolist()) == [1, 2, 3]
    assert out["gsis_id"].iloc[0] == "00-0036322"


def test_read_features_filters_to_requested_weeks(tmp_path: Path) -> None:
    """The optional `weeks` kwarg returns only those week partitions."""
    features_root = tmp_path
    for week in (1, 2, 3):
        df = pd.DataFrame([_minimal_wr_features_row(week)])
        write_partition(features_root, "wr", df, season=2024, week=week)

    out = read_features(Position.WR, 2024, weeks=[2, 3], features_root=features_root)
    assert sorted(out["week"].tolist()) == [2, 3]


def test_read_features_raises_when_cache_missing(tmp_path: Path) -> None:
    """If the (position, season) directory has no parquet partitions, raise
    FileNotFoundError with a clear message."""
    with pytest.raises(FileNotFoundError, match="No feature cache"):
        read_features(Position.WR, 2024, features_root=tmp_path)
