"""PBP team-level feature computes — tests."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.schemas import _PYARROW_STR


def _make_pbp_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a synthetic PBP frame with sane defaults for unspecified columns.

    The compute fns only read a subset of PbpSchema columns; tests fill in
    only the columns the function under test uses, defaults the rest."""
    defaults = {
        "play_id": 1,
        "game_id": "2024_01_KC_BAL",
        "season": 2024,
        "week": 1,
        "posteam": "KC",
        "defteam": "BAL",
        "play_type": "pass",
        "qb_dropback": 1.0,
        "qb_scramble": 0.0,
        "sack": 0.0,
        "rush_attempt": 0.0,
        "pass_attempt": 1.0,
        "epa": 0.0,
        "wpa": 0.0,
        "success": 0.0,
        "air_yards": 8.0,
        "yards_after_catch": 0.0,
        "complete_pass": 1.0,
        "xpass": 0.5,
        "pass_oe": 0.0,
        "down": 1.0,
        "ydstogo": 10,
        "yardline_100": 50.0,
        "half_seconds_remaining": 600.0,
        "passer_player_id": "00-0011111",
        "rusher_player_id": None,
        "receiver_player_id": "00-0022222",
    }
    out = []
    for r in rows:
        merged = {**defaults, **r}
        out.append(merged)
    return pd.DataFrame(out)


def test_pace_counts_pass_and_run_only() -> None:
    """Kickoffs / punts / FGs / no_play do not count toward pace."""
    from projections.features.pbp_team_features import compute_team_pace

    # Build 4 prior weeks for KC so trailing-4 has a window, then test week 5.
    # Each prior week: 50 pass+run plays + 10 special-teams plays.
    rows: list[dict[str, object]] = []
    for wk in range(1, 6):
        for i in range(50):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "pass" if i % 2 == 0 else "run",
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(10):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "kickoff",
                    "play_id": 100 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_pace(pbp)
    # Week 5's pace_l4 should be 50 (mean of 50 plays/game over wk1-4),
    # not 60 (which would include kickoffs).
    wk5_kc = out.query("team == 'KC' and season == 2024 and week == 5")
    assert len(wk5_kc) == 1
    assert wk5_kc["pace_l4"].iloc[0] == pytest.approx(50.0)


def test_pace_trailing_window_excludes_current_week() -> None:
    """pace_l4 at week W is computed from weeks 1..W-1, not 1..W."""
    from projections.features.pbp_team_features import compute_team_pace

    rows: list[dict[str, object]] = []
    # KC plays 60 plays in wk1, 60 in wk2, 60 in wk3, 60 in wk4, then 100 in wk5.
    for wk, count in [(1, 60), (2, 60), (3, 60), (4, 60), (5, 100)]:
        for i in range(count):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "pass",
                    "play_id": 1000 * wk + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_pace(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    # Trailing-4 of wk5 = mean(wk1..wk4) = 60, NOT mean(wk2..wk5) = 70.
    assert wk5["pace_l4"].iloc[0] == pytest.approx(60.0)


def test_pace_returns_nan_for_first_4_weeks_when_no_prior_history() -> None:
    """min_periods=4: weeks with fewer than 4 prior games emit NaN."""
    from projections.features.pbp_team_features import compute_team_pace

    rows: list[dict[str, object]] = []
    for wk in range(1, 4):
        for i in range(50):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "pass",
                    "play_id": 100 * wk + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_pace(pbp)
    # Wk1 has 0 prior games, wk2 has 1, wk3 has 2 — all NaN under min_periods=4.
    for wk in (1, 2, 3):
        row = out.query(f"team == 'KC' and season == 2024 and week == {wk}")
        assert pd.isna(row["pace_l4"].iloc[0])


def test_pace_does_not_leak_across_team_boundaries() -> None:
    """Two-team frame: each team's pace_l4 reflects only that team's history."""
    from projections.features.pbp_team_features import compute_team_pace

    rows: list[dict[str, object]] = []
    for wk in range(1, 6):
        for i in range(10):  # KC: 10 plays/wk
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "pass",
                    "play_id": 1000 * wk + i,
                }
            )
        for i in range(100):  # BAL: 100 plays/wk
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "play_type": "pass",
                    "play_id": 2000 * wk + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_pace(pbp)
    # KC week 1 must be NaN — no prior KC history. Buggy global-shift impl yields 100.0.
    assert pd.isna(out.query("team == 'KC' and week == 1")["pace_l4"].iloc[0])
    # KC week 5 must reflect KC's history alone (mean of 10,10,10,10).
    assert out.query("team == 'KC' and week == 5")["pace_l4"].iloc[0] == pytest.approx(10.0)
    # BAL week 5 must reflect BAL's history alone.
    assert out.query("team == 'BAL' and week == 5")["pace_l4"].iloc[0] == pytest.approx(100.0)


def test_proe_uses_pass_oe_mean_directly() -> None:
    """proe_l4 is the per-team rolling-4 mean of nflfastR's pass_oe column."""
    from projections.features.pbp_team_features import compute_team_proe

    # KC: 4 prior weeks with mean pass_oe = +5.0; week 5 should report +5.0.
    # BAL: 4 prior weeks with mean pass_oe = -3.0; week 5 should report -3.0.
    rows: list[dict[str, object]] = []
    for team, oe in [("KC", 5.0), ("BAL", -3.0)]:
        for wk in range(1, 6):
            for i in range(20):
                rows.append(
                    {
                        "season": 2024,
                        "week": wk,
                        "posteam": team,
                        "pass_oe": oe,
                        "play_type": "pass",
                        "play_id": 1000 * wk + i,
                    }
                )
    pbp = _make_pbp_rows(rows)
    out = compute_team_proe(pbp)

    kc_wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    bal_wk5 = out.query("team == 'BAL' and season == 2024 and week == 5")
    assert kc_wk5["proe_l4"].iloc[0] == pytest.approx(5.0)
    assert bal_wk5["proe_l4"].iloc[0] == pytest.approx(-3.0)


def test_proe_drops_nan_pass_oe_rows() -> None:
    """pass_oe NaN (e.g., kickoffs, no-plays) are excluded from the mean."""
    from projections.features.pbp_team_features import compute_team_proe

    rows: list[dict[str, object]] = []
    # KC wk1-4: 10 plays with pass_oe=10.0, plus 90 plays with pass_oe=NaN.
    # Mean over non-NaN should be 10.0.
    for wk in range(1, 6):
        for i in range(10):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "pass_oe": 10.0,
                    "play_type": "pass",
                    "play_id": 1000 * wk + i,
                }
            )
        for i in range(90):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "pass_oe": float("nan"),
                    "play_type": "kickoff",
                    "play_id": 2000 * wk + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_proe(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5["proe_l4"].iloc[0] == pytest.approx(10.0)


def test_ayps_only_counts_pass_attempts() -> None:
    """team_ayps_l4 averages air_yards over plays where pass_attempt == 1.0."""
    from projections.features.pbp_team_features import compute_team_ayps

    rows: list[dict[str, object]] = []
    # KC wk1-5: each week, 10 pass attempts at air_yards=15.0 (mean=15.0)
    #          + 50 rushing plays at air_yards=NaN, pass_attempt=0.0.
    for wk in range(1, 6):
        for i in range(10):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "pass_attempt": 1.0,
                    "air_yards": 15.0,
                    "play_type": "pass",
                    "play_id": 1000 * wk + i,
                }
            )
        for i in range(50):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "pass_attempt": 0.0,
                    "air_yards": float("nan"),
                    "play_type": "run",
                    "play_id": 2000 * wk + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_ayps(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5["team_ayps_l4"].iloc[0] == pytest.approx(15.0)


def test_ayps_drops_nan_air_yards_pass_attempts() -> None:
    """A pass attempt with air_yards=NaN (sack, throw-away) is excluded."""
    from projections.features.pbp_team_features import compute_team_ayps

    rows: list[dict[str, object]] = []
    # 10 valid pass attempts (air_yards=10.0) + 5 NaN-air-yards pass attempts
    # per week. Mean should be 10.0, not (10*10 + 5*0) / 15 = 6.67.
    for wk in range(1, 6):
        for i in range(10):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "pass_attempt": 1.0,
                    "air_yards": 10.0,
                    "play_type": "pass",
                    "play_id": 1000 * wk + i,
                }
            )
        for i in range(5):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "pass_attempt": 1.0,
                    "air_yards": float("nan"),
                    "play_type": "pass",
                    "play_id": 2000 * wk + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_ayps(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5["team_ayps_l4"].iloc[0] == pytest.approx(10.0)


def test_def_epa_residual_subtracts_schedule_strength() -> None:
    """Two defenses allow identical raw EPA, but BAL faces a top offense
    (KC, season-avg EPA = +0.3) while CIN faces a bottom offense (JAX,
    season-avg EPA = -0.1). With both allowing +0.1 EPA per play, BAL's
    residual is NEGATIVE (allowing less than expected against tough
    offenses) and CIN's POSITIVE.

    NOTE on data construction: PBP's ``epa`` is a single signed value per
    play, the same value when read from either team's perspective. So the
    "offense's season-average EPA" and "defense's per-game EPA-allowed"
    necessarily share the same row sets. To make the math clean:
    - BAL vs KC weeks 1-5: every play epa = +0.1 (BAL's def-allowed = 0.1
      AND those 50 plays contribute to KC's season-avg as 0.1).
    - KC vs LV (filler) weeks 11-15: every play epa = +0.5, chosen so
      KC's full-season avg = (50*0.1 + 50*0.5)/100 = +0.3.
    - CIN vs JAX weeks 1-5: every play epa = +0.1.
    - JAX vs LV weeks 16-20: every play epa = -0.3, chosen so JAX's
      full-season avg = (50*0.1 + 50*-0.3)/100 = -0.1.
    """
    from projections.features.pbp_team_features import compute_team_def_epa_residual

    rows: list[dict[str, object]] = []

    def add_game(season: int, week: int, off: str, defn: str, epa: float) -> None:
        for i in range(10):
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "posteam": off,
                    "defteam": defn,
                    "epa": epa,
                    "play_id": 100000 * week + i,
                }
            )

    # BAL defense vs KC offense: BAL allows +0.1 each play, weeks 1-5.
    for wk in range(1, 6):
        add_game(2024, wk, off="KC", defn="BAL", epa=0.1)
    # KC offense filler vs LV: weeks 11-15 at +0.5 → KC season-avg = +0.3.
    for wk in range(11, 16):
        add_game(2024, wk, off="KC", defn="LV", epa=0.5)

    # CIN defense vs JAX offense: CIN allows +0.1 each play, weeks 1-5.
    for wk in range(1, 6):
        add_game(2024, wk, off="JAX", defn="CIN", epa=0.1)
    # JAX offense filler vs LV: weeks 16-20 at -0.3 → JAX season-avg = -0.1.
    for wk in range(16, 21):
        add_game(2024, wk, off="JAX", defn="LV", epa=-0.3)

    pbp = _make_pbp_rows(rows)
    out = compute_team_def_epa_residual(pbp)

    bal_wk5 = out.query("team == 'BAL' and season == 2024 and week == 5")
    cin_wk5 = out.query("team == 'CIN' and season == 2024 and week == 5")
    # BAL trailing-4 residual at wk5 = mean of wks 1-4's residuals.
    # Each game: allowed +0.1 vs KC (season EPA +0.3) → residual = -0.2.
    assert bal_wk5["team_def_epa_resid_l4"].iloc[0] == pytest.approx(-0.2, abs=0.01)
    # CIN: allowed +0.1 vs JAX (season EPA -0.1) → residual = +0.2.
    assert cin_wk5["team_def_epa_resid_l4"].iloc[0] == pytest.approx(+0.2, abs=0.01)


def _make_player_team_week_index(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a (gsis_id, season, week, team, opp) frame with GSIS-format IDs."""
    out = pd.DataFrame(rows)
    out["gsis_id"] = out["gsis_id"].astype(_PYARROW_STR)
    out["team"] = out["team"].astype(_PYARROW_STR)
    out["opp"] = out["opp"].astype(_PYARROW_STR)
    out["season"] = out["season"].astype("Int64")
    out["week"] = out["week"].astype("Int64")
    return out


def test_assembler_emits_4_columns_with_correct_join_sides() -> None:
    """pace/proe/team_ayps join on the player's TEAM; team_def_epa_resid
    joins on the player's OPPONENT.

    The fixture is constructed so KC and BAL have DISTINCT def-residuals
    (-0.2 vs +0.2). A wrong-side join (on team instead of opp) would swap
    them; an all-NaN merge regression (wrong key, dtype mismatch) would
    fail both assertions. This is stronger than the prior fixture (every
    play epa=0.1) under which both teams' residuals collapsed to 0.0 and
    a wrong-side join would silently pass.

    Construction:
    - Weeks 1-5: KC posteam vs BAL defteam at epa=0.1 (20 plays/wk).
    - Weeks 1-5: BAL posteam vs KC defteam at epa=0.4 (20 plays/wk).
    - Weeks 6-10 (filler vs MIA): KC posteam at epa=0.5; BAL posteam at
      epa=0.0. These pull each team's *season-avg* offensive EPA away
      from its *def-allowed* mean, so the residuals are non-zero.

    Math:
    - KC season-avg posteam EPA = (100*0.1 + 100*0.5)/200 = 0.3
    - BAL season-avg posteam EPA = (100*0.4 + 100*0.0)/200 = 0.2
    - BAL def-allowed (weeks 1-5 vs KC) = 0.1; per-game residual = 0.1 - 0.3 = -0.2
    - KC def-allowed (weeks 1-5 vs BAL) = 0.4; per-game residual = 0.4 - 0.2 = +0.2
    - At week 5 (trailing-4 over weeks 1-4): both means are constant -0.2 / +0.2.
    """
    from projections.features.pbp_team_features import build_pbp_family_overrides

    pbp_rows: list[dict[str, object]] = []

    def _add_plays(
        *,
        season: int,
        week: int,
        posteam: str,
        defteam: str,
        epa: float,
        pass_oe: float,
        air_yards: float,
        play_id_base: int,
    ) -> None:
        for i in range(20):
            pbp_rows.append(
                {
                    "season": season,
                    "week": week,
                    "posteam": posteam,
                    "defteam": defteam,
                    "play_type": "pass",
                    "pass_attempt": 1.0,
                    "air_yards": air_yards,
                    "pass_oe": pass_oe,
                    "epa": epa,
                    "play_id": play_id_base + i,
                }
            )

    # Weeks 1-5: KC offense vs BAL defense at epa=0.1; pass_oe=5.0, air_yards=8.0.
    # Weeks 1-5: BAL offense vs KC defense at epa=0.4; pass_oe=-3.0, air_yards=6.0.
    for wk in range(1, 6):
        _add_plays(
            season=2024,
            week=wk,
            posteam="KC",
            defteam="BAL",
            epa=0.1,
            pass_oe=5.0,
            air_yards=8.0,
            play_id_base=100_000 + 1000 * wk,
        )
        _add_plays(
            season=2024,
            week=wk,
            posteam="BAL",
            defteam="KC",
            epa=0.4,
            pass_oe=-3.0,
            air_yards=6.0,
            play_id_base=200_000 + 1000 * wk,
        )

    # Weeks 6-10 filler vs MIA. epa values pull each team's season-avg
    # offensive EPA away from its def-allowed mean, making residuals non-zero.
    # pass_oe / air_yards values here are outside the trailing-4 window at
    # week 5, so they don't affect proe_l4 / team_ayps_l4 assertions below.
    for wk in range(6, 11):
        _add_plays(
            season=2024,
            week=wk,
            posteam="KC",
            defteam="MIA",
            epa=0.5,
            pass_oe=5.0,
            air_yards=8.0,
            play_id_base=300_000 + 1000 * wk,
        )
        _add_plays(
            season=2024,
            week=wk,
            posteam="BAL",
            defteam="MIA",
            epa=0.0,
            pass_oe=-3.0,
            air_yards=6.0,
            play_id_base=400_000 + 1000 * wk,
        )

    pbp = _make_pbp_rows(pbp_rows)

    # One player on KC, one on BAL — both at week 5 (trailing-4 over wks 1-4).
    idx = _make_player_team_week_index(
        [
            {"gsis_id": "00-0011111", "season": 2024, "week": 5, "team": "KC", "opp": "BAL"},
            {"gsis_id": "00-0022222", "season": 2024, "week": 5, "team": "BAL", "opp": "KC"},
        ]
    )

    out = build_pbp_family_overrides(pbp, idx)

    assert set(out.columns) == {
        "gsis_id",
        "season",
        "week",
        "pace_l4",
        "proe_l4",
        "team_ayps_l4",
        "team_def_epa_resid_l4",
    }
    assert len(out) == 2

    kc_player = out.query("gsis_id == '00-0011111'")
    bal_player = out.query("gsis_id == '00-0022222'")

    # KC player gets KC's offensive features.
    assert kc_player["proe_l4"].iloc[0] == pytest.approx(5.0)
    assert kc_player["team_ayps_l4"].iloc[0] == pytest.approx(8.0)

    # BAL player gets BAL's offensive features.
    assert bal_player["proe_l4"].iloc[0] == pytest.approx(-3.0)
    assert bal_player["team_ayps_l4"].iloc[0] == pytest.approx(6.0)

    # KC player's def_resid is the OPPONENT's (BAL's) def-residual = -0.2.
    # BAL player's def_resid is the OPPONENT's (KC's) def-residual = +0.2.
    # A wrong-side join (on team instead of opp) would swap signs.
    assert kc_player["team_def_epa_resid_l4"].iloc[0] == pytest.approx(-0.2, abs=1e-9)
    assert bal_player["team_def_epa_resid_l4"].iloc[0] == pytest.approx(+0.2, abs=1e-9)


def test_assembler_rejects_invalid_gsis_id() -> None:
    """build_pbp_family_overrides raises if any input gsis_id violates the
    GSIS_ID_PATTERN."""
    from projections.features.pbp_team_features import build_pbp_family_overrides

    pbp = _make_pbp_rows([{"season": 2024, "week": 1, "posteam": "KC"}])
    idx = _make_player_team_week_index(
        [
            {"gsis_id": "BOGUS_ID", "season": 2024, "week": 1, "team": "KC", "opp": "BAL"},
        ]
    )

    with pytest.raises(ValueError, match="gsis_id"):
        build_pbp_family_overrides(pbp, idx)


def test_assembler_rejects_duplicate_keys() -> None:
    """Two rows with the same (gsis_id, season, week) is a programmer
    error in the index — assembler refuses."""
    from projections.features.pbp_team_features import build_pbp_family_overrides

    pbp = _make_pbp_rows([{"season": 2024, "week": 1, "posteam": "KC"}])
    idx = _make_player_team_week_index(
        [
            {"gsis_id": "00-0011111", "season": 2024, "week": 5, "team": "KC", "opp": "BAL"},
            {"gsis_id": "00-0011111", "season": 2024, "week": 5, "team": "BAL", "opp": "KC"},
        ]
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_pbp_family_overrides(pbp, idx)
