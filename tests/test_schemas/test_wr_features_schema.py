"""WrFeaturesSchema tests — verify the Vegas team-context columns added in
TODO #33c integration (preseason_implied_team_total, preseason_spread,
season_avg_implied_team_total, season_avg_spread)."""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from projections.schemas import _PYARROW_STR, WrFeaturesSchema


def _wr_features_minimal_row() -> dict[str, object]:
    """A minimal valid WrFeaturesSchema row."""
    return {
        "gsis_id": "00-0034857",
        "season": 2024,
        "week": 5,
        "team": "KC",
        "opponent": "CHI",
        "targets_per_game_l4": 8.0,
        "targets_per_game_std": 7.5,
        "target_share_l4": 0.22,
        "air_yards_share_l4": 0.28,
        "receptions_per_game_l4": 5.0,
        "receiving_yards_per_game_l4": 70.0,
        "receiving_tds_per_game_l4": 0.5,
        "rushing_attempts_per_game_l4": 0.5,
        "rushing_yards_per_game_l4": 3.0,
        "designed_rusher": False,
        "snap_pct_l4": 0.85,
        "depth_rank": 1,
        "avg_separation_std": 2.8,
        "avg_intended_air_yards_std": 11.5,
        "percent_share_intended_air_yards_std": 0.28,
        "avg_yac_above_expectation_std": 0.5,
        "implied_team_total": 29.25,
        "spread": -7.5,
        "is_home": False,
        "roof_dome": False,
        "opp_allowed_wr_fppg_l4": 32.0,
        "age": 27.0,
        "is_rookie": 0.0,
        "volume_trend_l4_minus_prior_l4": 0.5,
        "snap_pct_change_l4_vs_prior_l4": 0.0,
        "wind_speed_mph": 8.0,
        "is_high_wind": 0.0,
        "temperature_f": 55.0,
        "is_grass_surface": 1.0,
    }


def _to_typed_df(row: dict[str, object]) -> pd.DataFrame:
    df = pd.DataFrame([row])
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    df["gsis_id"] = df["gsis_id"].astype(str)
    df["depth_rank"] = df["depth_rank"].astype(pd.Int64Dtype())
    return df


def test_wr_features_schema_accepts_vegas_team_context_cols() -> None:
    row = _wr_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": 28.0,
            "preseason_spread": -6.5,
            "season_avg_implied_team_total": 27.5,
            "season_avg_spread": -5.0,
        }
    )
    out = WrFeaturesSchema.validate(_to_typed_df(row))
    for col in (
        "preseason_implied_team_total",
        "preseason_spread",
        "season_avg_implied_team_total",
        "season_avg_spread",
    ):
        assert col in out.columns


def test_wr_features_schema_accepts_nan_on_vegas_cols() -> None:
    row = _wr_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": float("nan"),
            "preseason_spread": float("nan"),
            "season_avg_implied_team_total": float("nan"),
            "season_avg_spread": float("nan"),
        }
    )
    WrFeaturesSchema.validate(_to_typed_df(row))


def test_wr_features_schema_rejects_negative_preseason_implied_total() -> None:
    row = _wr_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": -1.0,
            "preseason_spread": -6.5,
            "season_avg_implied_team_total": 27.5,
            "season_avg_spread": -5.0,
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        WrFeaturesSchema.validate(_to_typed_df(row))


def test_wr_features_schema_rejects_negative_season_avg_implied_total() -> None:
    row = _wr_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": 28.0,
            "preseason_spread": -6.5,
            "season_avg_implied_team_total": -1.0,
            "season_avg_spread": -5.0,
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        WrFeaturesSchema.validate(_to_typed_df(row))
