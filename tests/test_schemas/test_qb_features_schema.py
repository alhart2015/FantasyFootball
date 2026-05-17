"""QbFeaturesSchema tests — verify the Vegas team-context columns added in
TODO #33c integration (preseason_implied_team_total, preseason_spread,
season_avg_implied_team_total, season_avg_spread)."""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from projections.schemas import _PYARROW_STR, QbFeaturesSchema


def _qb_features_minimal_row() -> dict[str, object]:
    """A minimal valid QbFeaturesSchema row. Mirrors the conftest qb_schedules
    week-5-2024 KC-vs-CHI fixture's downstream shape; values chosen to land
    inside every existing pa.Field bound."""
    return {
        "gsis_id": "00-0034857",
        "season": 2024,
        "week": 5,
        "team": "KC",
        "opponent": "CHI",
        "pass_attempts_per_game_l4": 32.0,
        "passing_yards_per_game_l4": 280.0,
        "passing_tds_per_game_l4": 2.0,
        "interceptions_per_game_l4": 0.5,
        "sacks_per_game_l4": 1.5,
        "passing_yards_per_game_std": 270.0,
        "rushing_attempts_per_game_l4": 5.0,
        "rushing_yards_per_game_l4": 20.0,
        "rushing_qb": True,
        "snap_pct_l4": 0.95,
        "depth_rank": 1,
        "aggressiveness_std": 12.5,
        "completion_percentage_above_expectation_std": 1.2,
        "avg_intended_air_yards_std": 8.5,
        "avg_time_to_throw_std": 2.8,
        "implied_team_total": 29.25,
        "spread": -7.5,
        "is_home": False,
        "roof_dome": False,
        "opp_allowed_qb_fppg_l4": 15.0,
    }


def _to_typed_df(row: dict[str, object]) -> pd.DataFrame:
    """Build a 1-row DataFrame with the dtype conventions the schema expects."""
    df = pd.DataFrame([row])
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    df["gsis_id"] = df["gsis_id"].astype(str)
    df["depth_rank"] = df["depth_rank"].astype(pd.Int64Dtype())
    return df


def test_qb_features_schema_accepts_vegas_team_context_cols() -> None:
    row = _qb_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": 28.0,
            "preseason_spread": -6.5,
            "season_avg_implied_team_total": 27.5,
            "season_avg_spread": -5.0,
        }
    )
    df = _to_typed_df(row)
    out = QbFeaturesSchema.validate(df)
    for col in (
        "preseason_implied_team_total",
        "preseason_spread",
        "season_avg_implied_team_total",
        "season_avg_spread",
    ):
        assert col in out.columns


def test_qb_features_schema_accepts_nan_on_vegas_cols() -> None:
    row = _qb_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": float("nan"),
            "preseason_spread": float("nan"),
            "season_avg_implied_team_total": float("nan"),
            "season_avg_spread": float("nan"),
        }
    )
    df = _to_typed_df(row)
    QbFeaturesSchema.validate(df)


def test_qb_features_schema_rejects_negative_preseason_implied_total() -> None:
    row = _qb_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": -1.0,
            "preseason_spread": -6.5,
            "season_avg_implied_team_total": 27.5,
            "season_avg_spread": -5.0,
        }
    )
    df = _to_typed_df(row)
    with pytest.raises(pandera.errors.SchemaError):
        QbFeaturesSchema.validate(df)


def test_qb_features_schema_rejects_negative_season_avg_implied_total() -> None:
    row = _qb_features_minimal_row()
    row.update(
        {
            "preseason_implied_team_total": 28.0,
            "preseason_spread": -6.5,
            "season_avg_implied_team_total": -1.0,
            "season_avg_spread": -5.0,
        }
    )
    df = _to_typed_df(row)
    with pytest.raises(pandera.errors.SchemaError):
        QbFeaturesSchema.validate(df)
