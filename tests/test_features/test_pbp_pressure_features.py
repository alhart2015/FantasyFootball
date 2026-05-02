"""PBP pressure feature computes — tests."""

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


def test_sack_rate_allowed_basic() -> None:
    """Sack rate = sum(sack) / sum(qb_dropback) over rows where posteam == team."""
    from projections.features.pbp_pressure_features import compute_team_sack_rate_allowed

    rows: list[dict[str, object]] = []
    # 4 prior weeks: 10 dropbacks/wk, 2 of them sacks → sack_rate = 0.20.
    for wk in range(1, 6):
        for i in range(2):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 1.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(8):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 0.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_sack_rate_allowed(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    assert len(wk5) == 1
    assert wk5["team_sack_rate_allowed_l4"].iloc[0] == pytest.approx(0.20)


def test_sack_rate_allowed_excludes_non_dropback_plays() -> None:
    """Plays with qb_dropback=0 (handoffs, kneels, spikes) excluded from
    BOTH numerator and denominator."""
    from projections.features.pbp_pressure_features import compute_team_sack_rate_allowed

    rows: list[dict[str, object]] = []
    for wk in range(1, 6):
        for i in range(2):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 1.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(8):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 0.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
        for i in range(50):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 0.0,
                    "sack": 0.0,
                    "play_id": 100 * wk + 200 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_sack_rate_allowed(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5["team_sack_rate_allowed_l4"].iloc[0] == pytest.approx(0.20)


def test_sack_rate_allowed_excludes_nan_dropback() -> None:
    """Rows with NaN qb_dropback excluded."""
    from projections.features.pbp_pressure_features import compute_team_sack_rate_allowed

    rows: list[dict[str, object]] = []
    for wk in range(1, 6):
        for i in range(2):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 1.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(8):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 0.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
        for i in range(5):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": float("nan"),
                    "sack": 0.0,
                    "play_id": 100 * wk + 300 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_sack_rate_allowed(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5["team_sack_rate_allowed_l4"].iloc[0] == pytest.approx(0.20)


def test_trailing_4_min_periods_4() -> None:
    """Weeks with fewer than 4 prior games emit NaN."""
    from projections.features.pbp_pressure_features import compute_team_sack_rate_allowed

    rows: list[dict[str, object]] = []
    for wk in range(1, 4):
        for i in range(2):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 1.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(8):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 0.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_sack_rate_allowed(pbp)
    for wk in (1, 2, 3):
        row = out.query(f"team == 'KC' and season == 2024 and week == {wk}")
        assert pd.isna(row["team_sack_rate_allowed_l4"].iloc[0])


def test_trailing_4_within_team_boundary() -> None:
    """Two-team frame: rolling+shift does not leak across team boundary."""
    from projections.features.pbp_pressure_features import compute_team_sack_rate_allowed

    rows: list[dict[str, object]] = []
    for wk in range(1, 6):
        for i in range(1):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 1.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(9):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 0.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
        for i in range(4):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "qb_dropback": 1.0,
                    "sack": 1.0,
                    "play_id": 200 * wk + i,
                }
            )
        for i in range(6):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "qb_dropback": 1.0,
                    "sack": 0.0,
                    "play_id": 200 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_sack_rate_allowed(pbp)
    wk5_kc = out.query("team == 'KC' and season == 2024 and week == 5")
    wk5_bal = out.query("team == 'BAL' and season == 2024 and week == 5")
    assert wk5_kc["team_sack_rate_allowed_l4"].iloc[0] == pytest.approx(0.10)
    assert wk5_bal["team_sack_rate_allowed_l4"].iloc[0] == pytest.approx(0.40)


def test_qb_scramble_rate_basic() -> None:
    """QB scramble rate = sum(qb_scramble) / sum(qb_dropback)."""
    from projections.features.pbp_pressure_features import compute_team_qb_scramble_rate

    rows: list[dict[str, object]] = []
    # 4 prior weeks: 10 dropbacks/wk, 3 of them scrambles → scramble_rate = 0.30.
    for wk in range(1, 6):
        for i in range(3):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "qb_scramble": 1.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(7):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "qb_scramble": 0.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_qb_scramble_rate(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5["team_qb_scramble_rate_l4"].iloc[0] == pytest.approx(0.30)


def test_qb_scramble_rate_excludes_non_dropback_plays() -> None:
    """qb_dropback=0 plays are excluded from both numerator and denominator."""
    from projections.features.pbp_pressure_features import compute_team_qb_scramble_rate

    rows: list[dict[str, object]] = []
    # 10 dropbacks/wk (3 scrambles), 50 handoffs/wk (qb_dropback=0).
    for wk in range(1, 6):
        for i in range(3):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "qb_scramble": 1.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(7):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "qb_scramble": 0.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
        for i in range(50):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 0.0,
                    "qb_scramble": 0.0,
                    "play_id": 100 * wk + 200 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_qb_scramble_rate(pbp)
    wk5 = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5["team_qb_scramble_rate_l4"].iloc[0] == pytest.approx(0.30)
