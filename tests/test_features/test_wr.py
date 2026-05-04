"""WR feature builder tests (non-leakage). Leakage tests live in test_wr_leakage.py
so a leak surfaces with a precise failure independent of shape/correctness checks."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.features import build_wr_features
from projections.schemas import WrFeaturesSchema


def test_build_wr_features_returns_validated_frame(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    WrFeaturesSchema.validate(out)


def test_build_wr_features_one_row_per_rostered_wr(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    # 3 WRs on rosters in week 5 → 3 rows.
    assert len(out) == 3
    assert set(out["gsis_id"]) == {"00-0036322", "00-0036323", "00-0034950"}


def test_build_wr_features_targets_per_game_l4_correct(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Jefferson weeks 1-4: 12/10/8/6 -> mean = 9.0."""
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    jef = out[out["gsis_id"] == "00-0036322"].iloc[0]
    assert jef["targets_per_game_l4"] == 9.0


def test_build_wr_features_target_share_l4_correct(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Jefferson MIN trailing-4 targets: 12+10+8+6 = 36.
    Reed MIN trailing-4 targets: 4+4+4+4 = 16. MIN total = 52.
    Jefferson share = 36/52 ~ 0.6923."""
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    jef = out[out["gsis_id"] == "00-0036322"].iloc[0]
    assert jef["target_share_l4"] == pytest.approx(36 / 52, abs=1e-6)


def test_build_wr_features_designed_rusher_false_for_pure_wrs(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """All 3 fixture WRs have 0 carries -> designed_rusher == False."""
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    assert not out["designed_rusher"].any()


def test_build_wr_features_designed_rusher_true_above_threshold(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Inject 2 carries/game over weeks 1-4 for Jefferson (8 carries total
    in trailing 4 -> 2.0 carries/game >= 1.5 threshold)."""
    ws = wr_weekly_stats.copy()
    mask = (ws["gsis_id"] == "00-0036322") & (ws["week"] <= 4)
    ws.loc[mask, "carries"] = 2

    out = build_wr_features(
        weekly_stats=ws,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    jef = out[out["gsis_id"] == "00-0036322"].iloc[0]
    assert bool(jef["designed_rusher"]) is True


def test_build_wr_features_implied_team_total_from_schedules(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """MIN implied total = (total + |spread|)/2 when MIN favored.
    total=48.5, MIN favored by 3.5 -> MIN implied = 26.0."""
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    jef = out[out["gsis_id"] == "00-0036322"].iloc[0]
    assert jef["implied_team_total"] == pytest.approx(26.0, abs=1e-6)
    assert bool(jef["is_home"]) is False  # MIN is the away team


def test_build_wr_features_rookie_with_no_prior_games_zeros(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Inject a rookie WR on KC with NO prior weeks of data; depth chart shows
    them at WR2. They get a row with l4 stats == 0 (or NaN), no crash."""
    extra_dc = pd.concat(
        [
            wr_depth_charts,
            pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0099777",
                        "season": 2024,
                        "week": 5,
                        "team": "KC",
                        "position": "WR",
                        "depth_team": "WR2",
                        "depth_rank": 2,
                    }
                ]
            ).astype({"gsis_id": "string[pyarrow]"}),
        ],
        ignore_index=True,
    )

    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=extra_dc,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    rookie = out[out["gsis_id"] == "00-0099777"].iloc[0]
    assert rookie["targets_per_game_l4"] == 0.0
    assert rookie["receiving_yards_per_game_l4"] == 0.0
    assert bool(rookie["designed_rusher"]) is False


def test_build_wr_features_attach_trajectory_drafted_veteran(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    wr_draft_picks: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Justin Jefferson: drafted 2020 at age 21; in 2024 should be age 25, is_rookie=0."""
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        draft_picks=wr_draft_picks,
        season=2024,
        as_of_week=5,
    )
    jef = out[out["gsis_id"] == "00-0036322"].iloc[0]
    assert jef["age"] == 25.0  # 21 + (2024 - 2020)
    assert jef["is_rookie"] == 0.0


def test_build_wr_features_attach_trajectory_rookie(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    wr_draft_picks: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Inject rookie WR 00-0099777 (drafted 2024 in fixture); is_rookie=1.0,
    age=22.0, no prior 8 games so volume_trend / snap_pct_change are NaN.

    The rookie needs at least one prior weekly_stats row in 2024 so that
    ``compute_age`` / ``compute_is_rookie`` (keyed by weekly_stats
    (gsis_id, season) pairs) emit a row for the rookie. A single week-1
    appearance is enough; the trend cols still NaN at week 5 because we
    have only 1 prior active game, far below the 8 needed."""
    extra_dc = pd.concat(
        [
            wr_depth_charts,
            pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0099777",
                        "season": 2024,
                        "week": 5,
                        "team": "KC",
                        "position": "WR",
                        "depth_team": "WR2",
                        "depth_rank": 2,
                    }
                ]
            ).astype({"gsis_id": "string[pyarrow]"}),
        ],
        ignore_index=True,
    )
    extra_ws = pd.concat(
        [
            wr_weekly_stats,
            pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0099777",
                        "season": 2024,
                        "week": 1,
                        "position": "WR",
                        "team": "KC",
                        "opponent": "DET",
                        "passing_yards": 0.0,
                        "passing_tds": 0,
                        "interceptions": 0,
                        "rushing_yards": 0.0,
                        "rushing_tds": 0,
                        "carries": 0,
                        "receptions": 2,
                        "receiving_yards": 25.0,
                        "receiving_tds": 0,
                        "receiving_air_yards": 30.0,
                        "targets": 3,
                        "fumbles_lost": 0,
                    }
                ]
            ).astype(
                {
                    "gsis_id": "string[pyarrow]",
                    "position": "string[pyarrow]",
                    "team": "string[pyarrow]",
                    "opponent": "string[pyarrow]",
                }
            ),
        ],
        ignore_index=True,
    )

    out = build_wr_features(
        weekly_stats=extra_ws,
        snap_counts=wr_snap_counts,
        depth_charts=extra_dc,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        draft_picks=wr_draft_picks,
        season=2024,
        as_of_week=5,
    )
    rookie = out[out["gsis_id"] == "00-0099777"].iloc[0]
    assert rookie["is_rookie"] == 1.0
    assert rookie["age"] == 22.0
    assert pd.isna(rookie["volume_trend_l4_minus_prior_l4"])
    assert pd.isna(rookie["snap_pct_change_l4_vs_prior_l4"])
