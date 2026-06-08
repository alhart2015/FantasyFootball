"""QB feature builder tests (non-leakage). Leakage tests live in test_qb_leakage.py."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.features import build_qb_features
from projections.schemas import QbFeaturesSchema


def test_build_qb_features_returns_validated_frame(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    QbFeaturesSchema.validate(out)


def test_build_qb_features_one_row_per_rostered_qb(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    # 2 QBs on rosters in week 5 → 2 rows.
    assert len(out) == 2
    assert set(out["gsis_id"]) == {"00-0034857", "00-0033106"}


def test_build_qb_features_pass_attempts_per_game_l4_correct(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Mahomes weeks 1-4: 36/38/40/42 → mean = 39.0."""
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    mahomes = out[out["gsis_id"] == "00-0034857"].iloc[0]
    assert mahomes["pass_attempts_per_game_l4"] == 39.0


def test_build_qb_features_rushing_qb_false_for_pocket_qbs(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Both fixture QBs have 0 carries → rushing_qb == False."""
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    assert not out["rushing_qb"].any()


def test_build_qb_features_rushing_qb_true_above_threshold(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Inject 6 carries/game over weeks 1-4 for Mahomes (24 trailing-4 / 4 = 6.0 ≥ 5.0)."""
    ws = qb_weekly_stats.copy()
    mask = (ws["gsis_id"] == "00-0034857") & (ws["week"] <= 4)
    ws.loc[mask, "carries"] = 6
    ws.loc[mask, "rushing_yards"] = 30.0
    out = build_qb_features(
        weekly_stats=ws,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    mahomes = out[out["gsis_id"] == "00-0034857"].iloc[0]
    assert bool(mahomes["rushing_qb"]) is True


def test_build_qb_features_implied_team_total_from_schedules(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """KC @ CHI, total=51, spread_line=-7.5 (KC away favored).
    KC implied = (51 - (-7.5))/2 = 29.25; KC spread = +(-7.5) = -7.5."""
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    mahomes = out[out["gsis_id"] == "00-0034857"].iloc[0]
    assert mahomes["implied_team_total"] == pytest.approx(29.25, abs=1e-6)
    assert mahomes["spread"] == pytest.approx(-7.5, abs=1e-6)
    assert bool(mahomes["is_home"]) is False


def test_build_qb_features_ngs_aggressiveness_propagates(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """The latest NGS snapshot's `aggressiveness` is propagated as `aggressiveness_std`."""
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    mahomes = out[out["gsis_id"] == "00-0034857"].iloc[0]
    # Fixture sets aggressiveness=12.5 for Mahomes through week 4.
    assert mahomes["aggressiveness_std"] == pytest.approx(12.5, abs=1e-6)


def test_build_qb_features_emits_vegas_team_context_cols(
    qb_weekly_stats: pd.DataFrame,
    qb_snap_counts: pd.DataFrame,
    qb_depth_charts: pd.DataFrame,
    qb_ngs_passing: pd.DataFrame,
    qb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """build_qb_features attaches the four Vegas team-context cols and the
    output validates against the extended QbFeaturesSchema. With a 1-week
    fixture, preseason_* values equal the only-week values; season_avg_*
    is NaN (expanding mean of a single game .shift(1) is NaN)."""
    out = build_qb_features(
        weekly_stats=qb_weekly_stats,
        snap_counts=qb_snap_counts,
        depth_charts=qb_depth_charts,
        ngs_passing=qb_ngs_passing,
        schedules=qb_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    for col in (
        "preseason_implied_team_total",
        "preseason_spread",
        "season_avg_implied_team_total",
        "season_avg_spread",
    ):
        assert col in out.columns

    mahomes = out[out["gsis_id"] == "00-0034857"].iloc[0]
    # KC implied_team_total = (51 - (-7.5))/2 = 29.25; spread = -7.5.
    # preseason_* equals current week (week 5 is the only week in the fixture).
    assert mahomes["preseason_implied_team_total"] == pytest.approx(29.25, abs=1e-6)
    assert mahomes["preseason_spread"] == pytest.approx(-7.5, abs=1e-6)
    # season_avg_* is NaN with a single-game schedule (expanding mean .shift(1)).
    assert pd.isna(mahomes["season_avg_implied_team_total"])
    assert pd.isna(mahomes["season_avg_spread"])
