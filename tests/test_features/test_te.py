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
    fake_pbp_df: pd.DataFrame,
) -> None:
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
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
    fake_pbp_df: pd.DataFrame,
) -> None:
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
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
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Kelce: 8 targets/game -> mean = 8.0."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    kelce = out[out["gsis_id"] == "00-0030506"].iloc[0]
    assert kelce["targets_per_game_l4"] == 8.0


def test_build_te_features_emits_rushing_columns(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """TE schema includes rushing_attempts_per_game_l4 and rushing_yards_per_game_l4
    so the TE Model A baseline (Plan 3b) can capture Taysom-Hill-style rushing
    contribution. The columns are populated from the same WeeklyStatsSchema
    rushing source as RB."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    assert "rushing_attempts_per_game_l4" in out.columns
    assert "rushing_yards_per_game_l4" in out.columns
    # Kittle has uniform carries=6 and rushing_yards=28.0 for weeks 1-4 of 2024,
    # so the trailing-4 mean at as_of_week=5 is exactly 6.0 / 28.0.
    hill = out[out["gsis_id"] == "00-0033084"]
    assert not hill.empty
    assert hill["rushing_attempts_per_game_l4"].iloc[0] == pytest.approx(6.0, abs=1e-6)
    assert hill["rushing_yards_per_game_l4"].iloc[0] == pytest.approx(28.0, abs=1e-6)


def test_build_te_features_target_share_against_full_pass_catching_group(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Kelce 8 targets + Rice 8 targets = 16 KC team total. Kelce share = 8/16 = 0.5.
    Kittle 6 targets + Aiyuk 6 targets = 12 SF team total. Kittle share = 6/12 = 0.5."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
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
    fake_pbp_df: pd.DataFrame,
) -> None:
    """KC @ ARI, total=49, spread_line=-7.5 (KC away favored).
    KC implied = (49 - (-7.5))/2 = 28.25."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
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
    fake_pbp_df: pd.DataFrame,
) -> None:
    """ARI has roof='closed' in fixture -> roof_dome == True."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
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
    fake_pbp_df: pd.DataFrame,
) -> None:
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    kelce = out[out["gsis_id"] == "00-0030506"].iloc[0]
    # Fixture sets avg_separation=2.8 for Kelce.
    assert kelce["avg_separation_std"] == pytest.approx(2.8, abs=1e-6)


def test_build_te_features_attach_trajectory_drafted_veteran(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
    te_draft_picks: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Travis Kelce: drafted 2013 at age 23; in 2024 should be age 34, is_rookie=0."""
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
        draft_picks=te_draft_picks,
        season=2024,
        as_of_week=5,
    )
    kelce = out[out["gsis_id"] == "00-0030506"].iloc[0]
    assert kelce["age"] == 34.0  # 23 + (2024 - 2013)
    assert kelce["is_rookie"] == 0.0


def test_build_te_features_attach_trajectory_rookie(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
    te_draft_picks: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Inject rookie TE 00-0099778 (drafted 2024 in fixture); is_rookie=1.0,
    age=22.0, no prior 8 games so volume_trend / snap_pct_change are NaN.

    The rookie needs at least one prior weekly_stats row in 2024 so that
    ``compute_age`` / ``compute_is_rookie`` (keyed by weekly_stats
    (gsis_id, season) pairs) emit a row for the rookie. A single week-1
    appearance is enough; the trend cols still NaN at week 5 because we
    have only 1 prior active game, far below the 8 needed. Mirrors PR
    #26's rookie-WR test precedent."""
    extra_dc = pd.concat(
        [
            te_depth_charts,
            pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0099778",
                        "season": 2024,
                        "week": 5,
                        "team": "KC",
                        "position": "TE",
                        "depth_team": "TE2",
                        "depth_rank": 2,
                    }
                ]
            ).astype({"gsis_id": "string[pyarrow]"}),
        ],
        ignore_index=True,
    )
    extra_ws = pd.concat(
        [
            te_weekly_stats,
            pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0099778",
                        "season": 2024,
                        "week": 1,
                        "position": "TE",
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

    out = build_te_features(
        weekly_stats=extra_ws,
        snap_counts=te_snap_counts,
        depth_charts=extra_dc,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
        draft_picks=te_draft_picks,
        season=2024,
        as_of_week=5,
    )
    rookie = out[out["gsis_id"] == "00-0099778"].iloc[0]
    assert rookie["is_rookie"] == 1.0
    assert rookie["age"] == 22.0
    assert pd.isna(rookie["volume_trend_l4_minus_prior_l4"])
    assert pd.isna(rookie["snap_pct_change_l4_vs_prior_l4"])


def test_build_te_features_attach_trajectory_udfa_fallback(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """UDFA TE (not in draft_picks fixture) falls back to inferred draft year
    = earliest weekly_stats appearance. Kelce appears in fixture from week 1
    of 2024, so inferred_year = 2024 -> is_rookie = 1.0, age = 2024 - 2024 +
    22.0 = 22.0."""
    udfa_picks = pd.DataFrame(
        columns=[
            "gsis_id",
            "draft_year",
            "draft_round",
            "draft_overall_pick",
            "pfr_id",
            "draft_age",
        ]
    ).astype(
        {
            "gsis_id": "string[pyarrow]",
            "draft_year": pd.Int64Dtype(),
            "draft_round": pd.Int64Dtype(),
            "draft_overall_pick": pd.Int64Dtype(),
            "pfr_id": "string[pyarrow]",
            "draft_age": pd.Float64Dtype(),
        }
    )
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
        draft_picks=udfa_picks,
        season=2024,
        as_of_week=5,
    )
    kelce = out[out["gsis_id"] == "00-0030506"].iloc[0]
    # Kelce's earliest appearance in synthetic fixture is 2024 week 1
    # (no historical seasons in te_weekly_stats), so inferred_year = 2024.
    assert kelce["age"] == 22.0  # 2024 - 2024 + 22.0
    assert kelce["is_rookie"] == 1.0


def test_build_te_features_empty_draft_picks_default(
    te_weekly_stats: pd.DataFrame,
    te_snap_counts: pd.DataFrame,
    te_depth_charts: pd.DataFrame,
    te_ngs_receiving: pd.DataFrame,
    te_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Calling build_te_features without draft_picks (default empty) must
    not raise. Every row falls through inferred-draft-year and the schema
    validates with non-NaN age (~22.0 from the offset) for everyone."""
    # Note: NO draft_picks kwarg passed.
    out = build_te_features(
        weekly_stats=te_weekly_stats,
        snap_counts=te_snap_counts,
        depth_charts=te_depth_charts,
        ngs_receiving=te_ngs_receiving,
        schedules=te_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    TeFeaturesSchema.validate(out)
    assert "age" in out.columns
    # All rows should have age = inferred-fallback (22.0) since the
    # synthetic fixture's earliest-week is the target season.
    assert (out["age"] == 22.0).all()
