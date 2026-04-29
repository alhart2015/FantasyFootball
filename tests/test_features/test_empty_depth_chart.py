"""Regression tests: feature builders must validate cleanly on empty input.

When a slate has no rostered players at a position (e.g., a partial-season
view, or any week where the depth chart hasn't been produced yet), the
builder takes the empty-output branch and validates an empty DataFrame.
Without `coerce=True` on the feature schema, that validation crashes —
``pd.DataFrame(columns=...)`` produces object-dtype columns and pandera
will not silently re-type them. These tests guard the early-return path
for all four position builders."""

from __future__ import annotations

import pandas as pd

from projections.features import (
    build_qb_features,
    build_rb_features,
    build_te_features,
    build_wr_features,
)
from projections.schemas import (
    QbFeaturesSchema,
    RbFeaturesSchema,
    TeFeaturesSchema,
    WrFeaturesSchema,
)


def test_build_wr_features_empty_depth_chart_does_not_crash(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,  # contains only QBs — no WR rows in target week
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
) -> None:
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        season=2024,
        as_of_week=5,
    )
    assert out.empty
    WrFeaturesSchema.validate(out)


def test_build_qb_features_empty_depth_chart_does_not_crash(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,  # contains only WRs — no QB rows in target week
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    assert out.empty
    QbFeaturesSchema.validate(out)


def test_build_rb_features_empty_depth_chart_does_not_crash(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,  # contains only WRs — no RB rows in target week
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    assert out.empty
    RbFeaturesSchema.validate(out)


def test_build_te_features_empty_depth_chart_does_not_crash(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,  # contains only WRs — no TE rows in target week
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    assert out.empty
    TeFeaturesSchema.validate(out)
