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
) -> None:
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
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
) -> None:
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
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
) -> None:
    """Jefferson weeks 1-4: 12/10/8/6 -> mean = 9.0."""
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
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
) -> None:
    """All 3 fixture WRs have 0 carries -> designed_rusher == False."""
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
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
) -> None:
    """MIN implied total = (total + |spread|)/2 when MIN favored.
    total=48.5, MIN favored by 3.5 -> MIN implied = 26.0."""
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
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
        season=2024,
        as_of_week=5,
    )
    rookie = out[out["gsis_id"] == "00-0099777"].iloc[0]
    assert rookie["targets_per_game_l4"] == 0.0
    assert rookie["receiving_yards_per_game_l4"] == 0.0
    assert bool(rookie["designed_rusher"]) is False
