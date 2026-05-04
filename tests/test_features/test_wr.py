"""WR feature builder tests (non-leakage). Leakage tests live in test_wr_leakage.py
so a leak surfaces with a precise failure independent of shape/correctness checks."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.features import build_wr_features
from projections.schemas import _PYARROW_STR, WrFeaturesSchema


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


def test_build_wr_features_attach_trajectory_udfa_fallback(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """UDFA WR (not in draft_picks fixture) falls back to inferred draft year
    = earliest weekly_stats appearance. Jefferson appears in fixture from
    week 1 of 2024, so inferred_year = 2024 -> is_rookie = 1.0,
    age = 2024 - 2024 + 22.0 = 22.0."""
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
    out = build_wr_features(
        weekly_stats=wr_weekly_stats,
        snap_counts=wr_snap_counts,
        depth_charts=wr_depth_charts,
        ngs_receiving=wr_ngs_receiving,
        schedules=wr_schedules,
        pbp=fake_pbp_df,
        draft_picks=udfa_picks,
        season=2024,
        as_of_week=5,
    )
    jef = out[out["gsis_id"] == "00-0036322"].iloc[0]
    # Jefferson's earliest appearance in synthetic fixture is 2024 week 1
    # (no historical seasons in the fixture), so inferred_year = 2024.
    assert jef["age"] == 22.0  # 2024 - 2024 + 22.0
    assert jef["is_rookie"] == 1.0


def test_build_wr_features_empty_draft_picks_default(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_depth_charts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Calling build_wr_features without draft_picks (default empty) must
    not raise. Every row falls through inferred-draft-year and the schema
    validates with non-NaN age (~22.0 from the offset) for everyone."""
    # Note: NO draft_picks kwarg passed.
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
    assert "age" in out.columns
    # All rows should have age = inferred-fallback (22.0) since the
    # synthetic fixture's earliest-week is the target season.
    assert (out["age"] == 22.0).all()


def test_build_wr_features_attach_trajectory_veteran_with_8plus_history_yields_non_nan_trend(
    wr_weekly_stats: pd.DataFrame,
    wr_snap_counts: pd.DataFrame,
    wr_ngs_receiving: pd.DataFrame,
    wr_draft_picks: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Direct regression test for the bug fixed in commit d1b3092.

    The bug: ``build_wr_features`` was passing ``prior_mask``-filtered
    ``ws`` / ``sc`` to ``attach_trajectory_features``. The helper's
    internal ``_volume_trend`` and ``compute_snap_pct_change`` already use
    ``.rolling(4).mean().shift(1)`` for leakage safety, so double-filtering
    stripped the current-week index row out of the trend output and every
    trend value resolved as NaN at merge time.

    Existing WR tests that asserted "trend is NaN" passed regardless
    because their fixtures (weeks 1-8 of 2024 with as_of_week=5) couldn't
    produce 8+ prior active games of history per player anyway. This test
    constructs a 9-week 2024 fixture for Justin Jefferson (drafted 2020,
    so a veteran in 2024) and queries at as_of_week=9 — the earliest week
    where 8 prior active games exist (weeks 1-8) AND the index row at
    week 9 needs to come back from the trend output.

    Hand-computed expected values from ``wr_weekly_stats`` targets:
      Jefferson weeks 1-8 = 12, 10, 8, 6, 14, 12, 10, 8.
      L4 at week 9 (mean of weeks 5-8) = (14+12+10+8) / 4 = 11.0.
      prior_l4 at week 9 (mean of weeks 1-4) = (12+10+8+6) / 4 = 9.0.
      volume_trend_l4_minus_prior_l4 = 11.0 - 9.0 = 2.0.

    Snap pct is uniform 0.95 in the fixture, so the snap_pct change is
    0.0 (mean L4 minus mean prior L4, both 0.95).

    A future change that re-introduces double-filtering (e.g., a "cleanup"
    swapping ``weekly_stats`` back to ``ws`` in the call site) would
    silently revert the trend cols to NaN at week 9 and fail this test.
    """
    as_of_week = 9

    # Add a week=9 weekly_stats row for Jefferson (and Reed for symmetry —
    # both MIN WRs need a row at week 9 if anything in build_wr_features
    # joins on that). The trend value at week 9 is computed from prior
    # weeks via .shift(1), so the week-9 target value itself is irrelevant
    # to the assertion (it does NOT enter the L4 / prior_l4 windows).
    extra_ws = pd.concat(
        [
            wr_weekly_stats,
            pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0036322",
                        "season": 2024,
                        "week": as_of_week,
                        "position": "WR",
                        "team": "MIN",
                        "opponent": "GB",
                        "passing_yards": 0.0,
                        "passing_tds": 0,
                        "interceptions": 0,
                        "rushing_yards": 0.0,
                        "rushing_tds": 0,
                        "carries": 0,
                        "receptions": 7,
                        "receiving_yards": 95.0,
                        "receiving_tds": 1,
                        "receiving_air_yards": 120.0,
                        "targets": 9,
                        "fumbles_lost": 0,
                    },
                ]
            ).astype(
                {
                    "gsis_id": _PYARROW_STR,
                    "position": _PYARROW_STR,
                    "team": _PYARROW_STR,
                    "opponent": _PYARROW_STR,
                }
            ),
        ],
        ignore_index=True,
    )

    # Mirror the weekly_stats coverage in snap_counts. Uniform 0.95 so the
    # L4 vs prior-L4 snap-pct delta is exactly 0.0.
    extra_sc = pd.concat(
        [
            wr_snap_counts,
            pd.DataFrame(
                [
                    {
                        "gsis_id": "00-0036322",
                        "season": 2024,
                        "week": as_of_week,
                        "team": "MIN",
                        "opponent": "GB",
                        "position": "WR",
                        "offense_snaps": 60,
                        "offense_pct": 0.95,
                        "defense_snaps": 0,
                        "defense_pct": 0.0,
                        "st_snaps": 2,
                        "st_pct": 0.05,
                    },
                ]
            ).astype(
                {
                    "gsis_id": _PYARROW_STR,
                    "team": _PYARROW_STR,
                    "opponent": _PYARROW_STR,
                    "position": _PYARROW_STR,
                }
            ),
        ],
        ignore_index=True,
    )

    # Custom depth chart at the target week — only Jefferson, keeps the
    # test row set tight to the assertion subject.
    dc_week9 = pd.DataFrame(
        [
            {
                "gsis_id": "00-0036322",
                "season": 2024,
                "week": as_of_week,
                "team": "MIN",
                "position": "WR",
                "depth_team": "WR1",
                "depth_rank": 1,
            },
        ]
    ).astype(
        {
            "gsis_id": _PYARROW_STR,
            "team": _PYARROW_STR,
            "position": _PYARROW_STR,
            "depth_team": _PYARROW_STR,
        }
    )

    # Custom schedule for week 9: MIN @ GB. Uses the same nfl_data_py
    # spread_line sign convention as the existing wr_schedules fixture
    # (negative spread_line => away favored, MIN here).
    sch_week9 = pd.DataFrame(
        {
            "season": [2024],
            "week": [as_of_week],
            "game_id": pd.array(["2024_09_MIN_GB"], dtype=_PYARROW_STR),
            "home_team": pd.array(["GB"], dtype=_PYARROW_STR),
            "away_team": pd.array(["MIN"], dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(["2024-11-03T17:00:00Z"], utc=True).as_unit("us"),
            "spread_line": [-2.5],
            "total_line": [47.5],
            "home_moneyline": pd.array([130], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-150], dtype=pd.Int64Dtype()),
            "surface": pd.array(["grass"], dtype=_PYARROW_STR),
            "roof": pd.array(["outdoors"], dtype=_PYARROW_STR),
            "temp": pd.array([45], dtype=pd.Int64Dtype()),
            "wind": pd.array([10], dtype=pd.Int64Dtype()),
        }
    )

    out = build_wr_features(
        weekly_stats=extra_ws,
        snap_counts=extra_sc,
        depth_charts=dc_week9,
        ngs_receiving=wr_ngs_receiving,
        schedules=sch_week9,
        pbp=fake_pbp_df,
        draft_picks=wr_draft_picks,
        season=2024,
        as_of_week=as_of_week,
    )

    jef = out[out["gsis_id"] == "00-0036322"].iloc[0]

    # Trajectory cols must be non-NaN — the d1b3092 bug would yield NaN here.
    assert pd.notna(jef["volume_trend_l4_minus_prior_l4"])
    assert pd.notna(jef["snap_pct_change_l4_vs_prior_l4"])

    # Hand-computed values: catches not just "non-NaN" but "correct value".
    # A wrong shift offset / window size would slip past a NaN-only check.
    assert jef["volume_trend_l4_minus_prior_l4"] == pytest.approx(2.0, abs=1e-9)
    assert jef["snap_pct_change_l4_vs_prior_l4"] == pytest.approx(0.0, abs=1e-9)

    # Sanity: Jefferson is a veteran (drafted 2020 at 21 -> age 25 in 2024,
    # is_rookie = 0). Confirms the trajectory append wired the draft_picks
    # path correctly even when 8+ history is present.
    assert jef["age"] == 25.0
    assert jef["is_rookie"] == 0.0
