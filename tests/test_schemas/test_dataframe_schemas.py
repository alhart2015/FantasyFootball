"""Pandera schema tests — verify schemas accept good DataFrames and reject bad ones."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError

from projections.schemas import (
    _PYARROW_STR,
    DepthChartsSchema,
    DraftPicksSchema,
    IdMapSchema,
    NgsPassingSchema,
    NgsReceivingSchema,
    NgsRushingSchema,
    PbpSchema,
    ProjectionWeeklySchema,
    QbFeaturesSchema,
    RbFeaturesSchema,
    SchedulesSchema,
    SnapCountsSchema,
    TeFeaturesSchema,
    WeeklyStatsSchema,
    WrFeaturesSchema,
)


def _good_weekly_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": ["00-0036322"],
            "season": [2024],
            "week": [3],
            "position": ["WR"],
            "team": ["MIN"],
            "opponent": ["HOU"],
            "passing_yards": [0.0],
            "passing_tds": [0],
            "interceptions": [0],
            "attempts": [0],
            "completions": [0],
            "sacks": [0],
            "rushing_yards": [0.0],
            "rushing_tds": [0],
            "carries": [0],
            "receptions": [9],
            "receiving_yards": [110.0],
            "receiving_tds": [1],
            "receiving_air_yards": [145.0],
            "targets": [12],
            "fumbles_lost": [0],
        }
    )


def test_weekly_stats_accepts_good_frame() -> None:
    WeeklyStatsSchema.validate(_good_weekly_stats())


def test_weekly_stats_rejects_bad_position() -> None:
    bad = _good_weekly_stats()
    bad["position"] = "FB"
    with pytest.raises(SchemaError):
        WeeklyStatsSchema.validate(bad)


def test_weekly_stats_rejects_week_out_of_range() -> None:
    bad = _good_weekly_stats()
    bad["week"] = 99
    with pytest.raises(SchemaError):
        WeeklyStatsSchema.validate(bad)


def test_weekly_stats_rejects_bad_gsis_id() -> None:
    bad = _good_weekly_stats()
    bad["gsis_id"] = "not-an-id"
    with pytest.raises(SchemaError):
        WeeklyStatsSchema.validate(bad)


def test_id_map_schema() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": ["00-0036322"],
            "espn_id": ["4262921"],
            "sleeper_id": ["6794"],
            "pfr_id": ["JeffJu00"],
            "full_name": ["Justin Jefferson"],
            "position": ["WR"],
            "team": ["MIN"],
        }
    )
    IdMapSchema.validate(df)


def test_projection_weekly_schema() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": ["00-0036322"],
            "season": [2026],
            "week": [3],
            "position": ["WR"],
            "team": ["MIN"],
            "opponent": ["HOU"],
            "ruleset": ["ESPN_PPR"],
            "family": ["GAMMA"],
            "params": [b"\x00"],
            "mean": [18.4],
            "p10": [6.1],
            "p50": [17.2],
            "p90": [33.5],
            "model_id": ["baseline:abc123:2014-2025"],
            "generated_at": [pd.Timestamp("2026-09-01", tz="UTC")],
        }
    )
    ProjectionWeeklySchema.validate(df)


def test_schedules_schema_accepts_valid_frame() -> None:
    df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [3, 3],
            "game_id": pd.array(["2024_03_KC_ATL", "2024_03_MIN_HOU"], dtype=_PYARROW_STR),
            "home_team": pd.array(["ATL", "HOU"], dtype=_PYARROW_STR),
            "away_team": pd.array(["KC", "MIN"], dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(
                ["2024-09-22T17:00:00Z", "2024-09-22T17:00:00Z"], utc=True
            ).as_unit("us"),
            "spread_line": [3.5, -2.5],
            "total_line": [48.5, 44.0],
            "home_moneyline": pd.array([155, -125], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-180, 105], dtype=pd.Int64Dtype()),
            "surface": pd.array(["fieldturf", "matrixturf"], dtype=_PYARROW_STR),
            "roof": pd.array(["dome", "dome"], dtype=_PYARROW_STR),
            "temp": pd.array([72, 72], dtype=pd.Int64Dtype()),
            "wind": pd.array([0, 0], dtype=pd.Int64Dtype()),
        }
    )
    SchedulesSchema.validate(df)


def test_schedules_schema_rejects_unknown_team_code() -> None:
    df = pd.DataFrame(
        {
            "season": [2024],
            "week": [3],
            "game_id": pd.array(["2024_03_XX_ATL"], dtype=_PYARROW_STR),
            "home_team": pd.array(["ATL"], dtype=_PYARROW_STR),
            "away_team": pd.array(["XX"], dtype=_PYARROW_STR),  # invalid
            "kickoff": pd.to_datetime(["2024-09-22T17:00:00Z"], utc=True).as_unit("us"),
            "spread_line": [3.5],
            "total_line": [48.5],
            "home_moneyline": pd.array([155], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-180], dtype=pd.Int64Dtype()),
            "surface": pd.array(["fieldturf"], dtype=_PYARROW_STR),
            "roof": pd.array(["dome"], dtype=_PYARROW_STR),
            "temp": pd.array([72], dtype=pd.Int64Dtype()),
            "wind": pd.array([0], dtype=pd.Int64Dtype()),
        }
    )
    with pytest.raises(SchemaError):
        SchedulesSchema.validate(df)


def test_schedules_schema_allows_nullable_lines() -> None:
    """Future-week games may have NaN spread/total/kickoff."""
    df = pd.DataFrame(
        {
            "season": [2024],
            "week": [18],
            "game_id": pd.array(["2024_18_TBD_TBD"], dtype=_PYARROW_STR),
            "home_team": pd.array(["KC"], dtype=_PYARROW_STR),
            "away_team": pd.array(["DEN"], dtype=_PYARROW_STR),
            "kickoff": pd.array([pd.NaT], dtype="datetime64[us, UTC]"),
            # spread_line/total_line are Series[float] (float64). Use float("nan")
            # (not pd.NA) so the column stays float64 rather than object — this
            # matches what nfl_data_py's import_schedules actually returns for
            # missing lines.
            "spread_line": [float("nan")],
            "total_line": [float("nan")],
            "home_moneyline": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "surface": pd.array([pd.NA], dtype=_PYARROW_STR),
            "roof": pd.array([pd.NA], dtype=_PYARROW_STR),
            "temp": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "wind": pd.array([pd.NA], dtype=pd.Int64Dtype()),
        }
    )
    SchedulesSchema.validate(df)


def test_snap_counts_schema_accepts_valid_frame() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036322", "00-0034857"], dtype=_PYARROW_STR),
            "season": [2024, 2024],
            "week": [3, 3],
            "team": pd.array(["MIN", "KC"], dtype=_PYARROW_STR),
            "opponent": pd.array(["HOU", "ATL"], dtype=_PYARROW_STR),
            "position": pd.array(["WR", "QB"], dtype=_PYARROW_STR),
            "offense_snaps": [62, 71],
            "offense_pct": [0.95, 1.0],
            "defense_snaps": [0, 0],
            "defense_pct": [0.0, 0.0],
            "st_snaps": [3, 0],
            "st_pct": [0.10, 0.0],
        }
    )
    SnapCountsSchema.validate(df)


def test_snap_counts_schema_rejects_pct_over_one() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036322"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [3],
            "team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "opponent": pd.array(["HOU"], dtype=_PYARROW_STR),
            "position": pd.array(["WR"], dtype=_PYARROW_STR),
            "offense_snaps": [62],
            "offense_pct": [1.5],  # invalid: > 1
            "defense_snaps": [0],
            "defense_pct": [0.0],
            "st_snaps": [3],
            "st_pct": [0.10],
        }
    )
    with pytest.raises(SchemaError):
        SnapCountsSchema.validate(df)


def test_snap_counts_schema_rejects_unsupported_position() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0099999"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [3],
            "team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "opponent": pd.array(["HOU"], dtype=_PYARROW_STR),
            "position": pd.array(["FB"], dtype=_PYARROW_STR),  # not in Position enum
            "offense_snaps": [12],
            "offense_pct": [0.20],
            "defense_snaps": [0],
            "defense_pct": [0.0],
            "st_snaps": [4],
            "st_pct": [0.13],
        }
    )
    with pytest.raises(SchemaError):
        SnapCountsSchema.validate(df)


def test_depth_charts_schema_accepts_valid_frame() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036322", "00-0034857"], dtype=_PYARROW_STR),
            "season": [2024, 2024],
            "week": [3, 3],
            "team": pd.array(["MIN", "KC"], dtype=_PYARROW_STR),
            "position": pd.array(["WR", "QB"], dtype=_PYARROW_STR),
            "depth_team": pd.array(["WR1", "QB1"], dtype=_PYARROW_STR),
            "depth_rank": [1, 1],
        }
    )
    DepthChartsSchema.validate(df)


def test_depth_charts_schema_rejects_rank_out_of_range() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036322"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [3],
            "team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "position": pd.array(["WR"], dtype=_PYARROW_STR),
            "depth_team": pd.array(["WR1"], dtype=_PYARROW_STR),
            "depth_rank": [12],  # > 10, invalid
        }
    )
    with pytest.raises(SchemaError):
        DepthChartsSchema.validate(df)


def test_ngs_passing_schema_accepts_valid_frame() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0034857"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [3],
            "team": pd.array(["KC"], dtype=_PYARROW_STR),
            "position": pd.array(["QB"], dtype=_PYARROW_STR),
            "avg_time_to_throw": [2.71],
            "avg_completed_air_yards": [6.2],
            "avg_intended_air_yards": [8.1],
            "avg_air_yards_differential": [-1.9],
            "aggressiveness": [12.5],
            "max_completed_air_distance": [42.0],
            "avg_air_yards_to_sticks": [-0.4],
            "completion_percentage": [68.5],
            "expected_completion_percentage": [65.2],
            "completion_percentage_above_expectation": [3.3],
            "avg_air_distance": [9.5],
            "max_air_distance": [55.0],
        }
    )
    NgsPassingSchema.validate(df)


def test_ngs_rushing_schema_accepts_valid_frame() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0034796"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [3],
            "team": pd.array(["PHI"], dtype=_PYARROW_STR),
            "position": pd.array(["RB"], dtype=_PYARROW_STR),
            "efficiency": [3.1],
            "percent_attempts_gte_eight_defenders": [22.5],
            "avg_time_to_los": [2.95],
            "rush_attempts": pd.array([18], dtype=pd.Int64Dtype()),
            "rush_yards": pd.array([102], dtype=pd.Int64Dtype()),
            "expected_rush_yards": [85.4],
            "rush_yards_over_expected": [16.6],
            "avg_rush_yards": [5.7],
            "rush_yards_over_expected_per_att": [0.9],
            "rush_pct_over_expected": [12.0],
        }
    )
    NgsRushingSchema.validate(df)


def test_ngs_receiving_schema_accepts_valid_frame() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036322"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [3],
            "team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "position": pd.array(["WR"], dtype=_PYARROW_STR),
            "avg_cushion": [5.4],
            "avg_separation": [3.2],
            "avg_intended_air_yards": [12.1],
            "percent_share_of_intended_air_yards": [29.5],
            "receptions": pd.array([9], dtype=pd.Int64Dtype()),
            "targets": pd.array([12], dtype=pd.Int64Dtype()),
            "catch_percentage": [75.0],
            "yards": pd.array([110], dtype=pd.Int64Dtype()),
            "rec_touchdowns": pd.array([1], dtype=pd.Int64Dtype()),
            "avg_yac": [4.0],
            "avg_expected_yac": [3.5],
            "avg_yac_above_expectation": [0.5],
        }
    )
    NgsReceivingSchema.validate(df)


def test_ngs_receiving_schema_allows_nan_below_threshold() -> None:
    """Players who don't meet NGS qualifying thresholds have NaN columns."""
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0099999"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [3],
            "team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "position": pd.array(["WR"], dtype=_PYARROW_STR),
            "avg_cushion": [float("nan")],
            "avg_separation": [float("nan")],
            "avg_intended_air_yards": [float("nan")],
            "percent_share_of_intended_air_yards": [float("nan")],
            "receptions": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "targets": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "catch_percentage": [float("nan")],
            "yards": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "rec_touchdowns": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "avg_yac": [float("nan")],
            "avg_expected_yac": [float("nan")],
            "avg_yac_above_expectation": [float("nan")],
        }
    )
    NgsReceivingSchema.validate(df)


def test_wr_features_schema_accepts_valid_row() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036322"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "opponent": pd.array(["DET"], dtype=_PYARROW_STR),
            "targets_per_game_l4": [10.5],
            "targets_per_game_std": [9.8],
            "target_share_l4": [0.32],
            "air_yards_share_l4": [0.41],
            "receptions_per_game_l4": [7.25],
            "receiving_yards_per_game_l4": [98.5],
            "receiving_tds_per_game_l4": [0.75],
            "rushing_attempts_per_game_l4": [0.5],
            "rushing_yards_per_game_l4": [3.2],
            "designed_rusher": [False],
            "snap_pct_l4": [0.92],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "avg_separation_std": [3.1],
            "avg_intended_air_yards_std": [11.8],
            "percent_share_intended_air_yards_std": [0.40],
            "avg_yac_above_expectation_std": [0.6],
            "implied_team_total": [24.5],
            "spread": [-3.5],
            "is_home": [True],
            "roof_dome": [False],
            "opp_allowed_wr_fppg_l4": [22.5],
            # Trajectory cols (PR #25 + 2026-05-03 WR integration). All
            # nullable=True; happy-path test covers the populated case.
            "age": [25.0],
            "is_rookie": [0.0],
            "volume_trend_l4_minus_prior_l4": [1.5],
            "snap_pct_change_l4_vs_prior_l4": [0.05],
            # Weather cols (PR #28 + 2026-05-08 RB+WR integration). All
            # nullable=True; happy-path test covers the populated case.
            "wind_speed_mph": [8.0],
            "is_high_wind": [0.0],
            "temperature_f": [65.0],
            "is_grass_surface": [1.0],
        }
    )
    WrFeaturesSchema.validate(df)


def test_wr_features_schema_rejects_target_share_over_one() -> None:
    """target_share is a proportion; > 1 is impossible."""
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036322"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "opponent": pd.array(["DET"], dtype=_PYARROW_STR),
            "targets_per_game_l4": [10.5],
            "targets_per_game_std": [9.8],
            "target_share_l4": [1.5],  # invalid
            "air_yards_share_l4": [0.41],
            "receptions_per_game_l4": [7.25],
            "receiving_yards_per_game_l4": [98.5],
            "receiving_tds_per_game_l4": [0.75],
            "rushing_attempts_per_game_l4": [0.5],
            "rushing_yards_per_game_l4": [3.2],
            "designed_rusher": [False],
            "snap_pct_l4": [0.92],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "avg_separation_std": [3.1],
            "avg_intended_air_yards_std": [11.8],
            "percent_share_intended_air_yards_std": [0.40],
            "avg_yac_above_expectation_std": [0.6],
            "implied_team_total": [24.5],
            "spread": [-3.5],
            "is_home": [True],
            "roof_dome": [False],
            "opp_allowed_wr_fppg_l4": [22.5],
            # Trajectory cols — provided so the failure is on target_share_l4,
            # not on a missing trajectory column.
            "age": [25.0],
            "is_rookie": [0.0],
            "volume_trend_l4_minus_prior_l4": [1.5],
            "snap_pct_change_l4_vs_prior_l4": [0.05],
            # Weather cols — provided so the failure is on target_share_l4,
            # not on a missing weather column.
            "wind_speed_mph": [8.0],
            "is_high_wind": [0.0],
            "temperature_f": [65.0],
            "is_grass_surface": [1.0],
        }
    )
    with pytest.raises(SchemaError):
        WrFeaturesSchema.validate(df)


def test_qb_features_schema_accepts_valid_row() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0034857"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["KC"], dtype=_PYARROW_STR),
            "opponent": pd.array(["DET"], dtype=_PYARROW_STR),
            "pass_attempts_per_game_l4": [38.5],
            "passing_yards_per_game_l4": [285.0],
            "passing_tds_per_game_l4": [2.25],
            "interceptions_per_game_l4": [0.5],
            "sacks_per_game_l4": [2.0],
            "passing_yards_per_game_std": [275.0],
            "rushing_attempts_per_game_l4": [4.5],
            "rushing_yards_per_game_l4": [22.0],
            "rushing_qb": [False],
            "snap_pct_l4": [1.0],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "aggressiveness_std": [12.5],
            "completion_percentage_above_expectation_std": [3.3],
            "avg_intended_air_yards_std": [8.1],
            "avg_time_to_throw_std": [2.71],
            "implied_team_total": [27.5],
            "spread": [-3.5],
            "is_home": [True],
            "roof_dome": [False],
            "opp_allowed_qb_fppg_l4": [18.5],
        }
    )
    QbFeaturesSchema.validate(df)


def test_qb_features_schema_rejects_negative_pass_attempts() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0034857"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["KC"], dtype=_PYARROW_STR),
            "opponent": pd.array(["DET"], dtype=_PYARROW_STR),
            "pass_attempts_per_game_l4": [-1.0],  # invalid
            "passing_yards_per_game_l4": [285.0],
            "passing_tds_per_game_l4": [2.25],
            "interceptions_per_game_l4": [0.5],
            "sacks_per_game_l4": [2.0],
            "passing_yards_per_game_std": [275.0],
            "rushing_attempts_per_game_l4": [4.5],
            "rushing_yards_per_game_l4": [22.0],
            "rushing_qb": [False],
            "snap_pct_l4": [1.0],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "aggressiveness_std": [12.5],
            "completion_percentage_above_expectation_std": [3.3],
            "avg_intended_air_yards_std": [8.1],
            "avg_time_to_throw_std": [2.71],
            "implied_team_total": [27.5],
            "spread": [-3.5],
            "is_home": [True],
            "roof_dome": [False],
            "opp_allowed_qb_fppg_l4": [18.5],
        }
    )
    with pytest.raises(SchemaError):
        QbFeaturesSchema.validate(df)


def test_rb_features_schema_accepts_valid_row() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0034796"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["PHI"], dtype=_PYARROW_STR),
            "opponent": pd.array(["DAL"], dtype=_PYARROW_STR),
            "carries_per_game_l4": [18.5],
            "rushing_yards_per_game_l4": [98.0],
            "rushing_tds_per_game_l4": [0.75],
            "rush_share_l4": [0.72],
            "targets_per_game_l4": [4.5],
            "receptions_per_game_l4": [3.5],
            "receiving_yards_per_game_l4": [28.0],
            "target_share_l4": [0.12],
            "targets_per_game_std": [4.0],
            "snap_pct_l4": [0.85],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "passing_down_back": [True],
            "efficiency_std": [3.1],
            "rush_yards_over_expected_per_att_std": [0.9],
            "percent_attempts_gte_eight_defenders_std": [22.5],
            "implied_team_total": [25.0],
            "spread": [-2.5],
            "is_home": [False],
            "roof_dome": [False],
            "opp_allowed_rb_fppg_l4": [20.5],
            "pace_l4": [65.0],
            "proe_l4": [2.5],
            "team_ayps_l4": [7.8],
            "team_def_epa_resid_l4": [-0.05],
            # Weather cols (PR #28 + 2026-05-08 RB+WR integration). All
            # nullable=True; happy-path test covers the populated case.
            "wind_speed_mph": [8.0],
            "is_high_wind": [0.0],
            "temperature_f": [65.0],
            "is_grass_surface": [1.0],
        }
    )
    RbFeaturesSchema.validate(df)


def test_rb_features_schema_rejects_rush_share_over_one() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0034796"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["PHI"], dtype=_PYARROW_STR),
            "opponent": pd.array(["DAL"], dtype=_PYARROW_STR),
            "carries_per_game_l4": [18.5],
            "rushing_yards_per_game_l4": [98.0],
            "rushing_tds_per_game_l4": [0.75],
            "rush_share_l4": [1.5],  # invalid, must be <= 1
            "targets_per_game_l4": [4.5],
            "receptions_per_game_l4": [3.5],
            "receiving_yards_per_game_l4": [28.0],
            "target_share_l4": [0.12],
            "targets_per_game_std": [4.0],
            "snap_pct_l4": [0.85],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "passing_down_back": [True],
            "efficiency_std": [3.1],
            "rush_yards_over_expected_per_att_std": [0.9],
            "percent_attempts_gte_eight_defenders_std": [22.5],
            "implied_team_total": [25.0],
            "spread": [-2.5],
            "is_home": [False],
            "roof_dome": [False],
            "opp_allowed_rb_fppg_l4": [20.5],
            "pace_l4": [65.0],
            "proe_l4": [2.5],
            "team_ayps_l4": [7.8],
            "team_def_epa_resid_l4": [-0.05],
            # Weather cols — provided so the failure is on rush_share_l4, not
            # on a missing weather column.
            "wind_speed_mph": [8.0],
            "is_high_wind": [0.0],
            "temperature_f": [65.0],
            "is_grass_surface": [1.0],
        }
    )
    with pytest.raises(SchemaError):
        RbFeaturesSchema.validate(df)


def test_te_features_schema_accepts_valid_row() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036440"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["SF"], dtype=_PYARROW_STR),
            "opponent": pd.array(["SEA"], dtype=_PYARROW_STR),
            "targets_per_game_l4": [7.5],
            "targets_per_game_std": [7.0],
            "target_share_l4": [0.18],
            "receptions_per_game_l4": [5.5],
            "receiving_yards_per_game_l4": [62.0],
            "receiving_tds_per_game_l4": [0.5],
            "rushing_attempts_per_game_l4": [0.0],
            "rushing_yards_per_game_l4": [0.0],
            "snap_pct_l4": [0.92],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "avg_separation_std": [2.8],
            "avg_intended_air_yards_std": [9.0],
            "avg_yac_above_expectation_std": [0.4],
            "implied_team_total": [24.0],
            "spread": [-1.5],
            "is_home": [True],
            "roof_dome": [False],
            "opp_allowed_te_fppg_l4": [10.5],
            # Trajectory cols (PR #25 + 2026-05-04 TE integration). All
            # nullable=True; happy-path test covers the populated case.
            "age": [25.0],
            "is_rookie": [0.0],
            "volume_trend_l4_minus_prior_l4": [1.5],
            "snap_pct_change_l4_vs_prior_l4": [0.05],
        }
    )
    TeFeaturesSchema.validate(df)


def test_te_features_schema_rejects_target_share_over_one() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0036440"], dtype=_PYARROW_STR),
            "season": [2024],
            "week": [6],
            "team": pd.array(["SF"], dtype=_PYARROW_STR),
            "opponent": pd.array(["SEA"], dtype=_PYARROW_STR),
            "targets_per_game_l4": [7.5],
            "targets_per_game_std": [7.0],
            "target_share_l4": [1.2],  # invalid
            "receptions_per_game_l4": [5.5],
            "receiving_yards_per_game_l4": [62.0],
            "receiving_tds_per_game_l4": [0.5],
            "rushing_attempts_per_game_l4": [0.0],
            "rushing_yards_per_game_l4": [0.0],
            "snap_pct_l4": [0.92],
            "depth_rank": pd.array([1], dtype=pd.Int64Dtype()),
            "avg_separation_std": [2.8],
            "avg_intended_air_yards_std": [9.0],
            "avg_yac_above_expectation_std": [0.4],
            "implied_team_total": [24.0],
            "spread": [-1.5],
            "is_home": [True],
            "roof_dome": [False],
            "opp_allowed_te_fppg_l4": [10.5],
            # Trajectory cols — provided so the failure is on target_share_l4,
            # not on a missing trajectory column.
            "age": [25.0],
            "is_rookie": [0.0],
            "volume_trend_l4_minus_prior_l4": [1.5],
            "snap_pct_change_l4_vs_prior_l4": [0.05],
        }
    )
    with pytest.raises(SchemaError):
        TeFeaturesSchema.validate(df)


def test_pbp_schema_validates_minimal_row() -> None:
    """A row with required fields and nullable extras passes."""
    df = pd.DataFrame(
        {
            "play_id": [1],
            "game_id": ["2024_03_KC_ATL"],
            "season": [2024],
            "week": [3],
            "posteam": pd.array(["KC"], dtype=pd.StringDtype("pyarrow")),
            "defteam": pd.array(["ATL"], dtype=pd.StringDtype("pyarrow")),
            "play_type": pd.array(["pass"], dtype=pd.StringDtype("pyarrow")),
            "qb_dropback": [1.0],
            "qb_scramble": [0.0],
            "sack": [0.0],
            "rush_attempt": [0.0],
            "pass_attempt": [1.0],
            "epa": [0.42],
            "wpa": [0.05],
            "success": [1.0],
            "air_yards": [12.0],
            "yards_after_catch": [3.0],
            "complete_pass": [1.0],
            "xpass": [0.65],
            "pass_oe": [0.10],
            "down": [1.0],
            "ydstogo": [10],
            "yardline_100": [75.0],
            "half_seconds_remaining": [1200.0],
            "passer_player_id": pd.array(["00-0034857"], dtype=pd.StringDtype("pyarrow")),
            "rusher_player_id": pd.array([None], dtype=pd.StringDtype("pyarrow")),
            "receiver_player_id": pd.array(["00-0036322"], dtype=pd.StringDtype("pyarrow")),
        }
    )
    PbpSchema.validate(df)


def test_pbp_schema_rejects_invalid_team_code() -> None:
    """defteam must be in canonical Team enum (alias 'JAX' should fail)."""
    df = pd.DataFrame(
        {
            "play_id": [1],
            "game_id": ["x"],
            "season": [2024],
            "week": [3],
            "posteam": pd.array(["KC"], dtype=pd.StringDtype("pyarrow")),
            "defteam": pd.array(["JAX"], dtype=pd.StringDtype("pyarrow")),
            "play_type": pd.array(["pass"], dtype=pd.StringDtype("pyarrow")),
            "qb_dropback": [1.0],
            "qb_scramble": [0.0],
            "sack": [0.0],
            "rush_attempt": [0.0],
            "pass_attempt": [1.0],
            "epa": [0.0],
            "wpa": [0.0],
            "success": [0.0],
            "air_yards": [0.0],
            "yards_after_catch": [0.0],
            "complete_pass": [0.0],
            "xpass": [0.5],
            "pass_oe": [0.0],
            "down": [1.0],
            "ydstogo": [10],
            "yardline_100": [75.0],
            "half_seconds_remaining": [1200.0],
            "passer_player_id": pd.array([None], dtype=pd.StringDtype("pyarrow")),
            "rusher_player_id": pd.array([None], dtype=pd.StringDtype("pyarrow")),
            "receiver_player_id": pd.array([None], dtype=pd.StringDtype("pyarrow")),
        }
    )
    with pytest.raises(SchemaError):
        PbpSchema.validate(df)


def test_draft_picks_schema_accepts_valid_row() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0033873"], dtype=_PYARROW_STR),
            "draft_year": pd.array([2017], dtype=pd.Int64Dtype()),
            "draft_round": pd.array([1], dtype=pd.Int64Dtype()),
            "draft_overall_pick": pd.array([10], dtype=pd.Int64Dtype()),
            "pfr_id": pd.array(["MahoPa00"], dtype=_PYARROW_STR),
            "draft_age": pd.array([21.5], dtype=pd.Float64Dtype()),
        }
    )
    validated = DraftPicksSchema.validate(df)
    assert len(validated) == 1


def test_draft_picks_schema_rejects_duplicate_gsis_id() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0033873", "00-0033873"], dtype=_PYARROW_STR),
            "draft_year": pd.array([2017, 2018], dtype=pd.Int64Dtype()),
            "draft_round": pd.array([1, 2], dtype=pd.Int64Dtype()),
            "draft_overall_pick": pd.array([10, 50], dtype=pd.Int64Dtype()),
            "pfr_id": pd.array([pd.NA, pd.NA], dtype=_PYARROW_STR),
            "draft_age": pd.array([21.5, 22.0], dtype=pd.Float64Dtype()),
        }
    )
    with pytest.raises(SchemaError):
        DraftPicksSchema.validate(df)


def test_draft_picks_schema_rejects_malformed_gsis_id() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["malformed-id"], dtype=_PYARROW_STR),
            "draft_year": pd.array([2017], dtype=pd.Int64Dtype()),
            "draft_round": pd.array([1], dtype=pd.Int64Dtype()),
            "draft_overall_pick": pd.array([10], dtype=pd.Int64Dtype()),
            "pfr_id": pd.array([pd.NA], dtype=_PYARROW_STR),
            "draft_age": pd.array([21.5], dtype=pd.Float64Dtype()),
        }
    )
    with pytest.raises(SchemaError):
        DraftPicksSchema.validate(df)


def test_draft_picks_schema_allows_nullable_optional_columns() -> None:
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0033873"], dtype=_PYARROW_STR),
            "draft_year": pd.array([2017], dtype=pd.Int64Dtype()),
            "draft_round": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "draft_overall_pick": pd.array([pd.NA], dtype=pd.Int64Dtype()),
            "pfr_id": pd.array([pd.NA], dtype=_PYARROW_STR),
            "draft_age": pd.array([float("nan")], dtype=pd.Float64Dtype()),
        }
    )
    DraftPicksSchema.validate(df)
