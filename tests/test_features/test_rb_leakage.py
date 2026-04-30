"""Leakage tests for build_rb_features. One assertion per input source."""

from __future__ import annotations

import pandas as pd

from projections.features import build_rb_features
from projections.schemas import _PYARROW_STR

_AS_OF_WEEK = 5
_SEASON = 2024


def _baseline(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> pd.DataFrame:
    return build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )


def test_no_leakage_from_weekly_stats(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    baseline = _baseline(
        rb_weekly_stats,
        rb_snap_counts,
        rb_depth_charts,
        rb_ngs_rushing,
        rb_schedules,
        fake_pbp_df,
    )
    leaky = rb_weekly_stats.copy()
    mask_future = (leaky["gsis_id"] == "00-0034796") & (leaky["week"] >= _AS_OF_WEEK)
    leaky.loc[mask_future, "carries"] = 99
    leaky.loc[mask_future, "rushing_yards"] = 999.0
    leaky.loc[mask_future, "targets"] = 99
    after = build_rb_features(
        weekly_stats=leaky,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_snap_counts(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    baseline = _baseline(
        rb_weekly_stats,
        rb_snap_counts,
        rb_depth_charts,
        rb_ngs_rushing,
        rb_schedules,
        fake_pbp_df,
    )
    leaky = rb_snap_counts.copy()
    mask_future = leaky["week"] >= _AS_OF_WEEK
    leaky.loc[mask_future, "offense_pct"] = 0.0
    leaky.loc[mask_future, "offense_snaps"] = 0
    after = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=leaky,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_ngs_rushing(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    baseline = _baseline(
        rb_weekly_stats,
        rb_snap_counts,
        rb_depth_charts,
        rb_ngs_rushing,
        rb_schedules,
        fake_pbp_df,
    )
    extra = pd.DataFrame(
        [
            {
                "gsis_id": "00-0034796",
                "season": 2024,
                "week": 5,
                "team": "PHI",
                "position": "RB",
                "efficiency": 99.0,
                "percent_attempts_gte_eight_defenders": 99.0,
                "avg_time_to_los": 99.0,
                "rush_attempts": 99,
                "rush_yards": 999,
                "expected_rush_yards": 99.0,
                "rush_yards_over_expected": 99.0,
                "avg_rush_yards": 99.0,
                "rush_yards_over_expected_per_att": 99.0,
                "rush_pct_over_expected": 99.0,
            }
        ]
    ).astype({"gsis_id": _PYARROW_STR, "team": _PYARROW_STR, "position": _PYARROW_STR})
    leaky = pd.concat([rb_ngs_rushing, extra], ignore_index=True)
    after = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=leaky,
        schedules=rb_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_depth_charts_other_weeks(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    baseline = _baseline(
        rb_weekly_stats,
        rb_snap_counts,
        rb_depth_charts,
        rb_ngs_rushing,
        rb_schedules,
        fake_pbp_df,
    )
    extra_weeks = pd.concat(
        [
            rb_depth_charts.assign(week=4, depth_rank=99, depth_team="RB99"),
            rb_depth_charts.assign(week=6, depth_rank=99, depth_team="RB99"),
        ],
        ignore_index=True,
    )
    leaky = pd.concat([rb_depth_charts, extra_weeks], ignore_index=True)
    after = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=leaky,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_schedules_other_weeks(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    baseline = _baseline(
        rb_weekly_stats,
        rb_snap_counts,
        rb_depth_charts,
        rb_ngs_rushing,
        rb_schedules,
        fake_pbp_df,
    )
    extra_weeks = rb_schedules.assign(week=6, total_line=99.0, spread_line=99.0)
    leaky = pd.concat([rb_schedules, extra_weeks], ignore_index=True)
    after = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=leaky,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)
