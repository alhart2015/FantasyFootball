"""Leakage tests for build_te_features. One assertion per input source."""

from __future__ import annotations

import pandas as pd

from projections.features import build_te_features
from projections.schemas import _PYARROW_STR

_AS_OF_WEEK = 5
_SEASON = 2024


def _baseline(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> pd.DataFrame:
    return build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )


def test_no_leakage_from_weekly_stats(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    baseline = _baseline(
        te_weekly_stats, te_snap_counts, te_depth_charts, te_ngs_receiving, te_schedules
    )
    leaky = te_weekly_stats.copy()
    mask_future = (leaky["gsis_id"] == "00-0030506") & (leaky["week"] >= _AS_OF_WEEK)
    leaky.loc[mask_future, "targets"] = 99
    leaky.loc[mask_future, "receiving_yards"] = 999.0
    leaky.loc[mask_future, "receiving_tds"] = 9
    after = build_te_features(
        weekly_stats=leaky,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_snap_counts(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    baseline = _baseline(
        te_weekly_stats, te_snap_counts, te_depth_charts, te_ngs_receiving, te_schedules
    )
    leaky = te_snap_counts.copy()
    mask_future = leaky["week"] >= _AS_OF_WEEK
    leaky.loc[mask_future, "offense_pct"] = 0.0
    leaky.loc[mask_future, "offense_snaps"] = 0
    after = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=leaky,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_ngs_receiving(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    baseline = _baseline(
        te_weekly_stats, te_snap_counts, te_depth_charts, te_ngs_receiving, te_schedules
    )
    extra = pd.DataFrame(
        [
            {
                "gsis_id": "00-0030506",
                "season": 2024,
                "week": 5,
                "team": "KC",
                "position": "TE",
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
    leaky = pd.concat([te_ngs_receiving, extra], ignore_index=True)
    after = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=leaky,
        schedules=te_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_depth_charts_other_weeks(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    baseline = _baseline(
        te_weekly_stats, te_snap_counts, te_depth_charts, te_ngs_receiving, te_schedules
    )
    extra_weeks = pd.concat(
        [
            te_depth_charts.assign(week=4, depth_rank=99, depth_team="TE99"),
            te_depth_charts.assign(week=6, depth_rank=99, depth_team="TE99"),
        ],
        ignore_index=True,
    )
    leaky = pd.concat([te_depth_charts, extra_weeks], ignore_index=True)
    after = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=leaky,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)


def test_no_leakage_from_schedules_other_weeks(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    baseline = _baseline(
        te_weekly_stats, te_snap_counts, te_depth_charts, te_ngs_receiving, te_schedules
    )
    extra_weeks = te_schedules.assign(week=6, total_line=99.0, spread_line=99.0)
    leaky = pd.concat([te_schedules, extra_weeks], ignore_index=True)
    after = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=leaky,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )
    pd.testing.assert_frame_equal(baseline, after, check_like=True)
