"""RB feature builder tests (non-leakage). Leakage tests live in test_rb_leakage.py."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.features import build_rb_features
from projections.schemas import _PYARROW_STR, RbFeaturesSchema


def test_build_rb_features_returns_validated_frame(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    RbFeaturesSchema.validate(out)
    # The 4 PBP family cols (spec 2026-05-01) must appear in the output
    # frame and be float-typed. The fake_pbp_df fixture has no PHI/SF/SEA
    # plays, so the values themselves come back NaN — but the columns
    # must exist.
    for col in ("pace_l4", "proe_l4", "team_ayps_l4", "team_def_epa_resid_l4"):
        assert col in out.columns, f"missing {col}"
        assert out[col].dtype == float, f"{col} dtype is {out[col].dtype}, expected float"


def test_build_rb_features_one_row_per_rostered_rb(
    rb_weekly_stats: pd.DataFrame,
    rb_snap_counts: pd.DataFrame,
    rb_depth_charts: pd.DataFrame,
    rb_ngs_rushing: pd.DataFrame,
    rb_schedules: pd.DataFrame,
    fake_pbp_df: pd.DataFrame,
) -> None:
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        pbp=fake_pbp_df,
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
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Saquon weeks 1-4: 20 carries/game uniformly → mean = 20.0."""
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        pbp=fake_pbp_df,
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
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Each fixture team has only one RB in the fixture → rush_share_l4 = 1.0."""
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        pbp=fake_pbp_df,
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
    fake_pbp_df: pd.DataFrame,
) -> None:
    """CMC has 6 targets/game → passing_down_back == True (>=4.0)."""
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        pbp=fake_pbp_df,
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
    fake_pbp_df: pd.DataFrame,
) -> None:
    """Saquon has 2 targets/game → passing_down_back == False."""
    out = build_rb_features(
        weekly_stats=rb_weekly_stats,
        snap_counts=rb_snap_counts,
        depth_charts=rb_depth_charts,
        ngs_rushing=rb_ngs_rushing,
        schedules=rb_schedules,
        pbp=fake_pbp_df,
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
    fake_pbp_df: pd.DataFrame,
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
        pbp=fake_pbp_df,
        season=2024,
        as_of_week=5,
    )
    # With no WR/TE rows in fixture, RB share against the (RB-only) pass-catching
    # set is 1.0 for both.
    assert (out["target_share_l4"] == 1.0).all()


def _make_pbp_rows_for_join_side_test() -> pd.DataFrame:
    """Synthetic PBP: 10 weeks of plays for KC (offense) vs BAL (defense)
    constructed so the 4 PBP-family computes yield distinct, predictable
    values per team. KC plays BAL at week 5 per the schedule fixture below;
    the trailing-4 window covers weeks 1-4 where KC and BAL faced off
    head-to-head.

    Math (mirrors test_assembler_emits_4_columns_with_correct_join_sides
    in test_pbp_team_features.py):
    - Weeks 1-4: KC offense vs BAL defense at epa=0.1; pass_oe=5.0, air_yards=8.0.
    - Weeks 1-4: BAL offense vs KC defense at epa=0.4; pass_oe=-3.0, air_yards=6.0.
    - Weeks 5-9 filler vs MIA: KC posteam epa=0.5; BAL posteam epa=0.0.
    - KC season-avg posteam EPA = (80*0.1 + 100*0.5)/180 ≈ 0.322
    - BAL season-avg posteam EPA = (80*0.4 + 100*0.0)/180 ≈ 0.178
    - At target week 5 trailing-4 (over wks 1-4):
        - KC pace_l4 = 20 plays/game; KC proe_l4 = 5.0; KC team_ayps_l4 = 8.0
        - BAL def-allowed (vs KC) = 0.1; per-game residual = 0.1 - 0.322 = -0.222
        - That is what the RB-on-KC-vs-BAL row should pick up for
          team_def_epa_resid_l4 (joined on opp=BAL).
    """
    rows: list[dict[str, object]] = []
    play_id = 1

    def _add(
        week: int,
        posteam: str,
        defteam: str,
        epa: float,
        pass_oe: float,
        air_yards: float,
    ) -> None:
        nonlocal play_id
        for _ in range(20):
            rows.append(
                {
                    "play_id": play_id,
                    "game_id": f"2024_{week:02d}_{posteam}_{defteam}",
                    "season": 2024,
                    "week": week,
                    "posteam": posteam,
                    "defteam": defteam,
                    "play_type": "pass",
                    "qb_dropback": 1.0,
                    "qb_scramble": 0.0,
                    "sack": 0.0,
                    "rush_attempt": 0.0,
                    "pass_attempt": 1.0,
                    "epa": epa,
                    "wpa": 0.0,
                    "success": 0.0,
                    "air_yards": air_yards,
                    "yards_after_catch": 0.0,
                    "complete_pass": 1.0,
                    "xpass": 0.5,
                    "pass_oe": pass_oe,
                    "down": 1.0,
                    "ydstogo": 10,
                    "yardline_100": 50.0,
                    "half_seconds_remaining": 600.0,
                    "passer_player_id": "00-0011111",
                    "rusher_player_id": None,
                    "receiver_player_id": "00-0022222",
                }
            )
            play_id += 1

    for wk in range(1, 5):
        _add(wk, posteam="KC", defteam="BAL", epa=0.1, pass_oe=5.0, air_yards=8.0)
        _add(wk, posteam="BAL", defteam="KC", epa=0.4, pass_oe=-3.0, air_yards=6.0)
    for wk in range(5, 10):
        _add(wk, posteam="KC", defteam="MIA", epa=0.5, pass_oe=5.0, air_yards=8.0)
        _add(wk, posteam="BAL", defteam="MIA", epa=0.0, pass_oe=-3.0, air_yards=6.0)
        # MIA posteam vs BAL/KC defteam at epa=0 — gives BAL and KC def rows at
        # weeks 5-9 so the trailing-4 shift in compute_team_def_epa_residual
        # has a row at week 5 to land on. Without this, BAL has only weeks 1-4
        # def rows, the shift drops the rolling-4 value, and the joined
        # team_def_epa_resid_l4 at week 5 comes back NaN. MIA's epa=0 here
        # makes its season-avg offensive EPA = 0, so BAL's residual at weeks
        # 5-9 = (0 - 0) = 0 — outside the trailing-4 window at week 5 anyway.
        _add(wk, posteam="MIA", defteam="BAL", epa=0.0, pass_oe=0.0, air_yards=5.0)
        _add(wk, posteam="MIA", defteam="KC", epa=0.0, pass_oe=0.0, air_yards=5.0)
    return pd.DataFrame(rows)


def test_build_rb_features_attach_pbp_family_join_sides() -> None:
    """RB on team KC facing opp BAL at week 5 picks up:
    - KC's offensive PBP features (pace_l4, proe_l4, team_ayps_l4)
    - BAL's defensive residual (team_def_epa_resid_l4, joined on opp)
    """
    pbp = _make_pbp_rows_for_join_side_test()

    # 4 weeks of weekly_stats so the RB has rolling carries history.
    ws_rows = []
    for wk in range(1, 5):
        ws_rows.append(
            {
                "gsis_id": "00-0099999",
                "season": 2024,
                "week": wk,
                "position": "RB",
                "team": "KC",
                "opponent": "BAL",
                "passing_yards": 0.0,
                "passing_tds": 0,
                "interceptions": 0,
                "attempts": 0,
                "completions": 0,
                "sacks": 0,
                "rushing_yards": 50.0,
                "rushing_tds": 0,
                "carries": 12,
                "receptions": 1,
                "receiving_yards": 5.0,
                "receiving_tds": 0,
                "receiving_air_yards": 3.0,
                "targets": 1,
                "fumbles_lost": 0,
            }
        )
    ws = pd.DataFrame(ws_rows)
    for c in ("gsis_id", "position", "team", "opponent"):
        ws[c] = ws[c].astype(_PYARROW_STR)

    sc_rows = []
    for wk in range(1, 5):
        sc_rows.append(
            {
                "gsis_id": "00-0099999",
                "season": 2024,
                "week": wk,
                "team": "KC",
                "opponent": "BAL",
                "position": "RB",
                "offense_snaps": 50,
                "offense_pct": 0.80,
                "defense_snaps": 0,
                "defense_pct": 0.0,
                "st_snaps": 0,
                "st_pct": 0.0,
            }
        )
    sc = pd.DataFrame(sc_rows)
    for c in ("gsis_id", "team", "opponent", "position"):
        sc[c] = sc[c].astype(_PYARROW_STR)

    dc = pd.DataFrame(
        [
            {
                "gsis_id": "00-0099999",
                "season": 2024,
                "week": 5,
                "team": "KC",
                "position": "RB",
                "depth_team": "RB1",
                "depth_rank": 1,
            }
        ]
    )
    for c in ("gsis_id", "team", "position", "depth_team"):
        dc[c] = dc[c].astype(_PYARROW_STR)

    sch = pd.DataFrame(
        {
            "season": [2024],
            "week": [5],
            "game_id": pd.array(["2024_05_KC_BAL"], dtype=_PYARROW_STR),
            "home_team": pd.array(["BAL"], dtype=_PYARROW_STR),
            "away_team": pd.array(["KC"], dtype=_PYARROW_STR),
            "kickoff": pd.to_datetime(["2024-10-06T17:00:00Z"], utc=True).as_unit("us"),
            "spread_line": [-2.5],
            "total_line": [45.0],
            "home_moneyline": pd.array([135], dtype=pd.Int64Dtype()),
            "away_moneyline": pd.array([-160], dtype=pd.Int64Dtype()),
            "surface": pd.array(["fieldturf"], dtype=_PYARROW_STR),
            "roof": pd.array(["outdoors"], dtype=_PYARROW_STR),
            "temp": pd.array([62], dtype=pd.Int64Dtype()),
            "wind": pd.array([6], dtype=pd.Int64Dtype()),
        }
    )

    ngs = pd.DataFrame(
        {
            "gsis_id": pd.array([], dtype=_PYARROW_STR),
            "season": pd.array([], dtype="Int64"),
            "week": pd.array([], dtype="Int64"),
            "team": pd.array([], dtype=_PYARROW_STR),
            "position": pd.array([], dtype=_PYARROW_STR),
            "efficiency": pd.array([], dtype=float),
            "percent_attempts_gte_eight_defenders": pd.array([], dtype=float),
            "avg_time_to_los": pd.array([], dtype=float),
            "rush_attempts": pd.array([], dtype="Int64"),
            "rush_yards": pd.array([], dtype="Int64"),
            "expected_rush_yards": pd.array([], dtype=float),
            "rush_yards_over_expected": pd.array([], dtype=float),
            "avg_rush_yards": pd.array([], dtype=float),
            "rush_yards_over_expected_per_att": pd.array([], dtype=float),
            "rush_pct_over_expected": pd.array([], dtype=float),
        }
    )

    out = build_rb_features(
        weekly_stats=ws,
        snap_counts=sc,
        depth_charts=dc,
        ngs_rushing=ngs,
        schedules=sch,
        pbp=pbp,
        season=2024,
        as_of_week=5,
    )

    assert len(out) == 1
    row = out.iloc[0]
    # KC offensive features (joined on team=KC).
    assert row["pace_l4"] == pytest.approx(20.0)
    assert row["proe_l4"] == pytest.approx(5.0)
    assert row["team_ayps_l4"] == pytest.approx(8.0)
    # BAL def-residual (joined on opp=BAL).
    # KC season-avg posteam EPA = (80*0.1 + 100*0.5)/180 = 0.32222...
    # BAL def-allowed vs KC = 0.1; per-game residual = 0.1 - 0.32222 = -0.22222
    assert row["team_def_epa_resid_l4"] == pytest.approx(-0.2222, abs=1e-3)
