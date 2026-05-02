"""PBP red-zone feature computes — tests."""

from __future__ import annotations

import pandas as pd
import pytest


def _make_pbp_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a synthetic PBP frame with sane defaults for unspecified columns.

    The compute fns only read a subset of PbpSchema columns; tests fill in
    only the columns the function under test uses, defaults the rest.
    """
    defaults: dict[str, object] = {
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
        "yardline_100": 15.0,  # default to RZ for these tests
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


def test_rz_pace_filters_yardline_100_gt_20() -> None:
    """Plays outside the red zone (yardline_100 > 20) do not count toward pace."""
    from projections.features.pbp_redzone_features import compute_team_rz_pace

    rows: list[dict[str, object]] = []
    # 4 prior weeks: 5 RZ plays + 5 non-RZ plays per week.
    for wk in range(1, 6):
        for i in range(5):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "pass",
                    "yardline_100": 10.0,  # RZ
                    "play_id": 100 * wk + i,
                }
            )
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "pass",
                    "yardline_100": 50.0,  # non-RZ
                    "play_id": 100 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_rz_pace(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    # Trailing-4 of wk5 = mean of wk1..wk4 RZ plays/game = 5.
    assert len(wk5) == 1
    assert wk5["team_rz_pace_l4"].iloc[0] == pytest.approx(5.0)


def test_rz_pace_excludes_special_teams_in_zone() -> None:
    """Kickoff/punt/FG plays at RZ yardlines are excluded by play_type filter."""
    from projections.features.pbp_redzone_features import compute_team_rz_pace

    rows: list[dict[str, object]] = []
    for wk in range(1, 6):
        for i in range(5):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "pass",
                    "yardline_100": 10.0,
                    "play_id": 100 * wk + i,
                }
            )
        # Add 3 FG attempts in RZ — should NOT count.
        for i in range(3):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "field_goal",
                    "yardline_100": 5.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_rz_pace(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    # 5 RZ pass plays per game, FG plays excluded → trailing-4 = 5.0.
    assert wk5["team_rz_pace_l4"].iloc[0] == pytest.approx(5.0)


def test_rz_pace_returns_nan_when_min_periods_unmet() -> None:
    """Weeks with fewer than 4 prior games emit NaN under min_periods=4."""
    from projections.features.pbp_redzone_features import compute_team_rz_pace

    rows: list[dict[str, object]] = []
    for wk in range(1, 4):  # only 3 weeks
        for i in range(5):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "pass",
                    "yardline_100": 10.0,
                    "play_id": 100 * wk + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_rz_pace(pbp)
    for wk in (1, 2, 3):
        row = out.query(f"team == 'KC' and season == 2024 and week == {wk}")
        assert pd.isna(row["team_rz_pace_l4"].iloc[0])


def test_rz_pace_does_not_leak_across_team_boundaries() -> None:
    """Two-team frame: each team's RZ pace_l4 reflects only its own history."""
    from projections.features.pbp_redzone_features import compute_team_rz_pace

    rows: list[dict[str, object]] = []
    for wk in range(1, 6):
        for i in range(3):  # KC: 3 RZ plays/wk
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "pass",
                    "yardline_100": 10.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(8):  # BAL: 8 RZ plays/wk
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "play_type": "run",
                    "yardline_100": 5.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_rz_pace(pbp)
    wk5_kc = out.query("team == 'KC' and season == 2024 and week == 5")
    wk5_bal = out.query("team == 'BAL' and season == 2024 and week == 5")
    assert wk5_kc["team_rz_pace_l4"].iloc[0] == pytest.approx(3.0)
    assert wk5_bal["team_rz_pace_l4"].iloc[0] == pytest.approx(8.0)


def test_rz_pass_rate_basic() -> None:
    """Pass rate is mean of pass_attempt over RZ pass+run plays."""
    from projections.features.pbp_redzone_features import compute_team_rz_pass_rate

    rows: list[dict[str, object]] = []
    # 4 prior weeks: 6 RZ plays/wk, 4 pass + 2 run → pass_rate = 4/6 ≈ 0.667.
    for wk in range(1, 6):
        for i in range(4):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "pass",
                    "pass_attempt": 1.0,
                    "yardline_100": 10.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(2):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "run",
                    "pass_attempt": 0.0,
                    "yardline_100": 10.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_rz_pass_rate(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5["team_rz_pass_rate_l4"].iloc[0] == pytest.approx(4.0 / 6.0)


def test_rz_pass_rate_filters_out_of_zone_plays() -> None:
    """Non-RZ plays do not affect the RZ pass-rate mean."""
    from projections.features.pbp_redzone_features import compute_team_rz_pass_rate

    rows: list[dict[str, object]] = []
    # 4 prior weeks: 4 RZ pass + 2 RZ run + 100 non-RZ run plays per week.
    # RZ-only pass_rate = 4/6; if the filter is broken, non-RZ runs would
    # drag the mean toward 4/106.
    for wk in range(1, 6):
        for i in range(4):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "pass",
                    "pass_attempt": 1.0,
                    "yardline_100": 10.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(2):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "run",
                    "pass_attempt": 0.0,
                    "yardline_100": 10.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
        for i in range(100):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "play_type": "run",
                    "pass_attempt": 0.0,
                    "yardline_100": 50.0,  # non-RZ
                    "play_id": 1000 * wk + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_rz_pass_rate(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5["team_rz_pass_rate_l4"].iloc[0] == pytest.approx(4.0 / 6.0)


def test_def_rz_epa_allowed_basic() -> None:
    """Defensive RZ EPA allowed = mean of opponent's RZ EPA per game,
    grouped by defteam."""
    from projections.features.pbp_redzone_features import compute_team_def_rz_epa_allowed

    rows: list[dict[str, object]] = []
    # KC defense plays BAL each week: 5 RZ plays/wk with EPA = 0.5 each.
    for wk in range(1, 6):
        for i in range(5):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "defteam": "KC",
                    "play_type": "pass",
                    "epa": 0.5,
                    "yardline_100": 10.0,
                    "play_id": 100 * wk + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_def_rz_epa_allowed(pbp)
    # KC's trailing-4 def_rz_epa_allowed_l4 at wk5 should be 0.5 (mean of wk1..wk4).
    wk5_kc = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5_kc["team_def_rz_epa_allowed_l4"].iloc[0] == pytest.approx(0.5)


def test_def_rz_epa_allowed_excludes_nan_epa() -> None:
    """Plays with NaN EPA do not contribute to the per-game mean."""
    from projections.features.pbp_redzone_features import compute_team_def_rz_epa_allowed

    rows: list[dict[str, object]] = []
    # 4 RZ plays per game, 2 with EPA=1.0 + 2 with NaN EPA.
    # Per-game mean = mean of [1.0, 1.0] = 1.0, NOT mean of [1.0, NaN, 1.0, NaN].
    for wk in range(1, 6):
        for i in range(2):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "defteam": "KC",
                    "play_type": "pass",
                    "epa": 1.0,
                    "yardline_100": 10.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(2):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "defteam": "KC",
                    "play_type": "pass",
                    "epa": float("nan"),
                    "yardline_100": 10.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_def_rz_epa_allowed(pbp)
    wk5_kc = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5_kc["team_def_rz_epa_allowed_l4"].iloc[0] == pytest.approx(1.0)


def test_def_rz_epa_allowed_excludes_special_teams() -> None:
    """Kickoff/punt/FG plays are excluded by play_type filter, even at
    RZ yardlines and with non-NaN EPA."""
    from projections.features.pbp_redzone_features import compute_team_def_rz_epa_allowed

    rows: list[dict[str, object]] = []
    # 5 RZ pass plays/wk with EPA=0.5 + 3 RZ FG plays/wk with EPA=2.0.
    # If FG plays leaked in, per-game mean would be (5*0.5 + 3*2.0)/8 ≈ 1.06.
    for wk in range(1, 6):
        for i in range(5):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "defteam": "KC",
                    "play_type": "pass",
                    "epa": 0.5,
                    "yardline_100": 10.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(3):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "defteam": "KC",
                    "play_type": "field_goal",
                    "epa": 2.0,
                    "yardline_100": 5.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_def_rz_epa_allowed(pbp)
    wk5_kc = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5_kc["team_def_rz_epa_allowed_l4"].iloc[0] == pytest.approx(0.5)
