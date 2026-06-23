"""Second-surface leakage guard for build_wr_features (spec §5.3(b)).

The first-surface guard (test_wr_leakage.py) asserts no leak into the
volume/snap/ngs/depth/schedule rollups. This guard targets the
trajectory + Vegas team-context feature surface specifically: inject
implausible FUTURE-week rows into the full frames that
attach_trajectory_features / the Vegas team-context join consume
(weekly_stats, schedules) and assert those feature columns are
byte-identical. A leak by definition changes the computation, so
byte-equality is the strongest possible signal.

If a test here FAILS, that is a real leak and a §5.3 hard stop — report
it; do NOT weaken the assertion.
"""

from __future__ import annotations

import pandas as pd

from projections.features import build_wr_features

_SEASON = 2024
_AS_OF_WEEK = 5

# Trajectory + Vegas team-context columns on WrFeaturesSchema (verified
# against src/projections/schemas.py).
_TRAJ_VEGAS_COLS = [
    "volume_trend_l4_minus_prior_l4",
    "snap_pct_change_l4_vs_prior_l4",
    "season_avg_implied_team_total",
    "season_avg_spread",
]


def _build(
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    schedules: pd.DataFrame,
    depth_charts: pd.DataFrame,
    ngs_receiving: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> pd.DataFrame:
    return build_wr_features(
        weekly_stats=weekly_stats,
        snap_counts=snap_counts,
        depth_charts=depth_charts,
        ngs_receiving=ngs_receiving,
        schedules=schedules,
        pbp=fake_pbp_df,
        season=_SEASON,
        as_of_week=_AS_OF_WEEK,
    )


def test_future_weekly_stats_do_not_leak_into_trajectory(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    base = _build(
        wr_weekly_stats,
        wr_snap_counts,
        wr_schedules,
        wr_depth_charts,
        wr_ngs_receiving,
        fake_pbp_df,
    )
    leaky = wr_weekly_stats.copy()
    fut = leaky["week"] >= _AS_OF_WEEK
    leaky.loc[fut, ["receiving_yards", "targets", "receptions"]] = 999.0
    after = _build(
        leaky,
        wr_snap_counts,
        wr_schedules,
        wr_depth_charts,
        wr_ngs_receiving,
        fake_pbp_df,
    )
    pd.testing.assert_frame_equal(base[_TRAJ_VEGAS_COLS], after[_TRAJ_VEGAS_COLS], check_like=True)


def test_future_schedules_do_not_leak_into_vegas(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    base = _build(
        wr_weekly_stats,
        wr_snap_counts,
        wr_schedules,
        wr_depth_charts,
        wr_ngs_receiving,
        fake_pbp_df,
    )
    leaky = pd.concat(
        [wr_schedules, wr_schedules.assign(week=6, total_line=999.0, spread_line=999.0)],
        ignore_index=True,
    )
    after = _build(
        wr_weekly_stats,
        wr_snap_counts,
        leaky,
        wr_depth_charts,
        wr_ngs_receiving,
        fake_pbp_df,
    )
    pd.testing.assert_frame_equal(base[_TRAJ_VEGAS_COLS], after[_TRAJ_VEGAS_COLS], check_like=True)
