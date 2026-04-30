"""Leakage tests for build_wr_features.

Strategy: build features for (season=2024, as_of_week=5). Then for each input
frame independently, fabricate implausible rows for week >= 5, rebuild, and
assert the output is byte-equal to the original. Five tests, one per input
source - a leak in any single source surfaces with a precise failure.

Why byte-equal not "values match within tolerance": a leak by definition
changes the computation, so the output will differ. We want the strongest
possible signal."""

from __future__ import annotations

import pandas as pd

from projections.features import build_wr_features
from projections.schemas import _PYARROW_STR

_AS_OF_WEEK = 5
_SEASON = 2024


def _baseline(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> pd.DataFrame:
    return build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )


def test_no_leakage_from_weekly_stats(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    baseline = _baseline(
        wr_weekly_stats,
        wr_snap_counts,
        wr_depth_charts,
        wr_ngs_receiving,
        wr_schedules,
        fake_pbp_df,
    )

    # Inject implausible weeks 5-8: Jefferson records 1000 receiving yards every week
    leaky = wr_weekly_stats.copy()
    mask_future = (leaky["gsis_id"] == "00-0036322") & (leaky["week"] >= _AS_OF_WEEK)
    leaky.loc[mask_future, "receiving_yards"] = 1000.0
    leaky.loc[mask_future, "targets"] = 30
    leaky.loc[mask_future, "receptions"] = 25
    leaky.loc[mask_future, "carries"] = 10

    after = build_wr_features(
        weekly_stats=leaky,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_snap_counts(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    baseline = _baseline(
        wr_weekly_stats,
        wr_snap_counts,
        wr_depth_charts,
        wr_ngs_receiving,
        wr_schedules,
        fake_pbp_df,
    )

    leaky = wr_snap_counts.copy()
    mask_future = leaky["week"] >= _AS_OF_WEEK
    leaky.loc[mask_future, "offense_pct"] = 0.0  # implausible: no snaps for any WR
    leaky.loc[mask_future, "offense_snaps"] = 0

    after = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=leaky,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_ngs_receiving(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    baseline = _baseline(
        wr_weekly_stats,
        wr_snap_counts,
        wr_depth_charts,
        wr_ngs_receiving,
        wr_schedules,
        fake_pbp_df,
    )

    # Inject NGS rows for week 5+ (the fixture only goes through week 4).
    extra = pd.DataFrame(
        [
            {
                "gsis_id": "00-0036322",
                "season": 2024,
                "week": 5,
                "team": "MIN",
                "position": "WR",
                "avg_cushion": 99.0,
                "avg_separation": 99.0,
                "avg_intended_air_yards": 99.0,
                "percent_share_of_intended_air_yards": 99.0,
                "receptions": 99,
                "targets": 99,
                "catch_percentage": 99.0,
                "yards": 999,
                "rec_touchdowns": 9,
                "avg_yac": 99.0,
                "avg_expected_yac": 99.0,
                "avg_yac_above_expectation": 99.0,
            }
        ]
    ).astype({"gsis_id": _PYARROW_STR, "team": _PYARROW_STR, "position": _PYARROW_STR})
    leaky = pd.concat([wr_ngs_receiving, extra], ignore_index=True)

    after = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=leaky,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_depth_charts_other_weeks(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Depth chart from OTHER weeks must not affect the as_of_week=5 build -
    we read only the target-week snapshot."""
    baseline = _baseline(
        wr_weekly_stats,
        wr_snap_counts,
        wr_depth_charts,
        wr_ngs_receiving,
        wr_schedules,
        fake_pbp_df,
    )

    extra_weeks = pd.concat(
        [
            wr_depth_charts.assign(week=4, depth_rank=99, depth_team="WR99"),
            wr_depth_charts.assign(week=6, depth_rank=99, depth_team="WR99"),
        ],
        ignore_index=True,
    )
    leaky = pd.concat([wr_depth_charts, extra_weeks], ignore_index=True)
    # depth_rank=99 violates schema, so skip schema validate and pass raw -
    # build_wr_features filters first by exact_week_mask.

    after = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=leaky,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_schedules_other_weeks(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Schedule rows from OTHER weeks must not affect the as_of_week=5 build."""
    baseline = _baseline(
        wr_weekly_stats,
        wr_snap_counts,
        wr_depth_charts,
        wr_ngs_receiving,
        wr_schedules,
        fake_pbp_df,
    )

    extra_weeks = wr_schedules.assign(week=6, total_line=99.0, spread_line=99.0)
    leaky = pd.concat([wr_schedules, extra_weeks], ignore_index=True)

    after = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=leaky,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)
