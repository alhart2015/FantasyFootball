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
        assert row["pace_l4"].iloc[0] != row["pace_l4"].iloc[0]  # NaN check
