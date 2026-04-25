"""RB feature builder tests (non-leakage). Leakage tests live in test_rb_leakage.py."""

from __future__ import annotations

import pandas as pd

from projections.features import build_rb_features
from projections.schemas import RbFeaturesSchema


def test_build_rb_features_returns_validated_frame(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    RbFeaturesSchema.validate(out)


def test_build_rb_features_one_row_per_rostered_rb(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    assert len(out) == 2
    assert set(out["gsis_id"]) == {"00-0034796", "00-0036650"}


def test_build_rb_features_carries_per_game_l4_correct(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    """Saquon weeks 1-4: 20 carries/game uniformly → mean = 20.0."""
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    saquon = out[out["gsis_id"] == "00-0034796"].iloc[0]
    assert saquon["carries_per_game_l4"] == 20.0


def test_build_rb_features_rush_share_l4_solo_rb_is_one(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    """Each fixture team has only one RB in the fixture → rush_share_l4 = 1.0."""
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    assert (out["rush_share_l4"] == 1.0).all()


def test_build_rb_features_passing_down_back_true_above_threshold(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    """CMC has 6 targets/game → passing_down_back == True (>=4.0)."""
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    cmc = out[out["gsis_id"] == "00-0036650"].iloc[0]
    assert bool(cmc["passing_down_back"]) is True


def test_build_rb_features_passing_down_back_false_below_threshold(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    """Saquon has 2 targets/game → passing_down_back == False."""
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    saquon = out[out["gsis_id"] == "00-0034796"].iloc[0]
    assert bool(saquon["passing_down_back"]) is False


def test_build_rb_features_target_share_against_full_pass_catching_group(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
) -> None:
    """target_share denominator must include WR + RB + TE on the team.
    The fixture has only RB rows; if no other receivers, RB target_share = 1.0
    (or 0 if RB has 0 targets — but Saquon has 2/game, CMC has 6/game)."""
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        season=2024,
        as_of_week=5,
    )
    # With no WR/TE rows in fixture, RB share against the (RB-only) pass-catching
    # set is 1.0 for both.
    assert (out["target_share_l4"] == 1.0).all()
