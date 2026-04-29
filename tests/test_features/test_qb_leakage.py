"""Leakage tests for build_qb_features. One assertion per input source."""

from __future__ import annotations

import pandas as pd

from projections.features import build_qb_features
from projections.schemas import _PYARROW_STR

_AS_OF_WEEK = 5
_SEASON = 2024


def _baseline(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> pd.DataFrame:
    return build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )


def test_no_leakage_from_weekly_stats(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    baseline = _baseline(
        qb_weekly_stats,
        qb_snap_counts,
        qb_depth_charts,
        qb_ngs_passing,
        qb_schedules,
        fake_pbp_df,
    )
    leaky = qb_weekly_stats.copy()
    mask_future = (leaky["gsis_id"] == "00-0034857") & (leaky["week"] >= _AS_OF_WEEK)
    leaky.loc[mask_future, "passing_yards"] = 999.0
    leaky.loc[mask_future, "attempts"] = 60
    leaky.loc[mask_future, "carries"] = 15
    after = build_qb_features(
        weekly_stats=leaky,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_snap_counts(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    baseline = _baseline(
        qb_weekly_stats,
        qb_snap_counts,
        qb_depth_charts,
        qb_ngs_passing,
        qb_schedules,
        fake_pbp_df,
    )
    leaky = qb_snap_counts.copy()
    mask_future = leaky["week"] >= _AS_OF_WEEK
    leaky.loc[mask_future, "offense_pct"] = 0.0
    leaky.loc[mask_future, "offense_snaps"] = 0
    after = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=leaky,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_ngs_passing(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    baseline = _baseline(
        qb_weekly_stats,
        qb_snap_counts,
        qb_depth_charts,
        qb_ngs_passing,
        qb_schedules,
        fake_pbp_df,
    )
    extra = pd.DataFrame(
        [
            {
                "gsis_id": "00-0034857",
                "season": 2024,
                "week": 5,
                "team": "KC",
                "position": "QB",
                "avg_time_to_throw": 99.0,
                "avg_completed_air_yards": 99.0,
                "avg_intended_air_yards": 99.0,
                "avg_air_yards_differential": 99.0,
                "aggressiveness": 99.0,
                "max_completed_air_distance": 99.0,
                "avg_air_yards_to_sticks": 99.0,
                "completion_percentage": 99.0,
                "expected_completion_percentage": 99.0,
                "completion_percentage_above_expectation": 99.0,
                "avg_air_distance": 99.0,
                "max_air_distance": 99.0,
            }
        ]
    ).astype({"gsis_id": _PYARROW_STR, "team": _PYARROW_STR, "position": _PYARROW_STR})
    leaky = pd.concat([qb_ngs_passing, extra], ignore_index=True)
    after = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=leaky,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_depth_charts_other_weeks(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    baseline = _baseline(
        qb_weekly_stats,
        qb_snap_counts,
        qb_depth_charts,
        qb_ngs_passing,
        qb_schedules,
        fake_pbp_df,
    )
    extra_weeks = pd.concat(
        [
            qb_depth_charts.assign(week=4, depth_rank=99, depth_team="QB99"),
            qb_depth_charts.assign(week=6, depth_rank=99, depth_team="QB99"),
        ],
        ignore_index=True,
    )
    leaky = pd.concat([qb_depth_charts, extra_weeks], ignore_index=True)
    after = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=leaky,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_schedules_other_weeks(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    baseline = _baseline(
        qb_weekly_stats,
        qb_snap_counts,
        qb_depth_charts,
        qb_ngs_passing,
        qb_schedules,
        fake_pbp_df,
    )
    extra_weeks = qb_schedules.assign(week=6, total_line=99.0, spread_line=99.0)
    leaky = pd.concat([qb_schedules, extra_weeks], ignore_index=True)
    after = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=leaky,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_pbp_other_weeks(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Future PBP rows (week >= as_of_week) must not influence the residual.

    Baseline uses only PBP rows from weeks < 5; the leaky frame appends a
    week-6 row with an extreme EPA. Builder must filter via prior_mask before
    calling opp_epa_allowed_residual."""
    baseline = _baseline(
        qb_weekly_stats,
        qb_snap_counts,
        qb_depth_charts,
        qb_ngs_passing,
        qb_schedules,
        fake_pbp_df,
    )
    extra = fake_pbp_df.iloc[[0]].copy()
    extra.loc[:, "week"] = 6
    extra.loc[:, "epa"] = 99.0
    extra.loc[:, "play_id"] = int(fake_pbp_df["play_id"].max()) + 1
    leaky = pd.concat([fake_pbp_df, extra], ignore_index=True)
    after = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=leaky,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)
