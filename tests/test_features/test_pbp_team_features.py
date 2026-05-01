"""PBP team-level feature computes — tests."""

from __future__ import annotations

import pandas as pd
import pytest


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
