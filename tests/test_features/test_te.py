"""TE feature builder tests (non-leakage). Leakage tests live in test_te_leakage.py."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.features import build_te_features
from projections.schemas import TeFeaturesSchema


def test_build_te_features_returns_validated_frame(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    TeFeaturesSchema.validate(out)


def test_build_te_features_one_row_per_rostered_te(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    assert len(out) == 2
    assert set(out["gsis_id"]) == {"00-0030506", "00-0033084"}


def test_build_te_features_targets_per_game_l4_correct(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    """Kelce: 8 targets/game -> mean = 8.0."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    kelce = out[out["gsis_id"] == "00-0030506"].iloc[0]
    assert kelce["targets_per_game_l4"] == 8.0


def test_build_te_features_target_share_against_full_pass_catching_group(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    """Kelce 8 targets + Rice 8 targets = 16 KC team total. Kelce share = 8/16 = 0.5.
    Kittle 6 targets + Aiyuk 6 targets = 12 SF team total. Kittle share = 6/12 = 0.5."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    kelce = out[out["gsis_id"] == "00-0030506"].iloc[0]
    kittle = out[out["gsis_id"] == "00-0033084"].iloc[0]
    assert kelce["target_share_l4"] == pytest.approx(0.5, abs=1e-6)
    assert kittle["target_share_l4"] == pytest.approx(0.5, abs=1e-6)


def test_build_te_features_implied_team_total_from_schedules(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    """KC @ ARI, total=49, spread_line=-7.5 (KC away favored).
    KC implied = (49 - (-7.5))/2 = 28.25."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    kelce = out[out["gsis_id"] == "00-0030506"].iloc[0]
    assert kelce["implied_team_total"] == pytest.approx(28.25, abs=1e-6)


def test_build_te_features_roof_dome_true_for_closed_roof(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    """ARI has roof='closed' in fixture -> roof_dome == True."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    assert out["roof_dome"].all()


def test_build_te_features_ngs_separation_propagates(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
) -> None:
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        season=2024,
        as_of_week=5,
    )
    kelce = out[out["gsis_id"] == "00-0030506"].iloc[0]
    # Fixture sets avg_separation=2.8 for Kelce.
    assert kelce["avg_separation_std"] == pytest.approx(2.8, abs=1e-6)
