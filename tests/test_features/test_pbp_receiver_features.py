"""PBP receiver-level feature computes — tests."""

from __future__ import annotations

import pandas as pd
import pytest


def _make_pbp_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a synthetic PBP frame with sane defaults for unspecified columns.

    The compute fns only read a subset of PbpSchema columns; tests fill in
    only the columns the function under test uses, default the rest. Mirrors
    `tests/test_features/test_pbp_team_features.py::_make_pbp_rows`.
    """
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
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_trailing_4_asof_within_player_no_leakage() -> None:
    """Rolling-4 stays within gsis_id; row N for player A doesn't see player B."""
    from projections.features.pbp_receiver_features import _trailing_4_per_player_asof

    # per_game: 5 rows for A (values 10..14), 5 rows for B (values 100..104).
    per_game = pd.DataFrame(
        [{"gsis_id": "00-0000001", "season": 2024, "week": w, "val": 9 + w} for w in range(1, 6)]
        + [{"gsis_id": "00-0000002", "season": 2024, "week": w, "val": 99 + w} for w in range(1, 6)]
    )
    # index: one row per (player, season, week) for both players, weeks 1-6.
    index = pd.DataFrame(
        [
            {"gsis_id": gid, "season": 2024, "week": w}
            for gid in ("00-0000001", "00-0000002")
            for w in range(1, 7)
        ]
    )
    out = _trailing_4_per_player_asof(per_game, index, value_col="val", out_col="val_l4")

    # Player A, week 5: trailing-4 at most-recent receiver-active game strictly
    # before (2024, w5) is row at (2024, w4) with rolling = mean(w1..w4 vals
    # 10,11,12,13) = 11.5.
    a_w5 = out.query("gsis_id == '00-0000001' and season == 2024 and week == 5")
    assert a_w5["val_l4"].iloc[0] == pytest.approx(11.5)

    # Player B, week 5: rolling-4 at (2024, w4) for B is mean(100,101,102,103)
    # = 101.5. NOT contaminated by A's values.
    b_w5 = out.query("gsis_id == '00-0000002' and season == 2024 and week == 5")
    assert b_w5["val_l4"].iloc[0] == pytest.approx(101.5)


def test_trailing_4_asof_inactive_week_uses_last_active() -> None:
    """For an index row where the player wasn't receiver-active, value comes
    from the last receiver-active game strictly before (season, week)."""
    from projections.features.pbp_receiver_features import _trailing_4_per_player_asof

    # Player active in weeks 1, 2, 3, 4, 5, 7 (skipped week 6).
    per_game = pd.DataFrame(
        [
            {"gsis_id": "00-0000001", "season": 2024, "week": w, "val": float(w)}
            for w in (1, 2, 3, 4, 5, 7)
        ]
    )
    # Index has rostered weeks 1-8; week 6 is a non-active week for the player.
    index = pd.DataFrame(
        [{"gsis_id": "00-0000001", "season": 2024, "week": w} for w in range(1, 9)]
    )
    out = _trailing_4_per_player_asof(per_game, index, value_col="val", out_col="val_l4")

    # Week 6 (inactive): last active game strictly before w6 is w5.
    # rolling-4 at w5 = mean(2, 3, 4, 5) = 3.5. So index w6 gets 3.5.
    w6 = out.query("week == 6")
    assert w6["val_l4"].iloc[0] == pytest.approx(3.5)

    # Week 7 (active again): last active game strictly before w7 is w5.
    # Same value as w6: 3.5.
    w7 = out.query("week == 7")
    assert w7["val_l4"].iloc[0] == pytest.approx(3.5)

    # Week 8: last active game strictly before w8 is w7. rolling-4 at w7 =
    # mean of 4 most recent receiver-active games up to and including w7 =
    # mean(3, 4, 5, 7) = 4.75.
    w8 = out.query("week == 8")
    assert w8["val_l4"].iloc[0] == pytest.approx(4.75)


def test_trailing_4_asof_min_periods_4() -> None:
    """Fewer than 4 prior receiver-active games yield NaN."""
    from projections.features.pbp_receiver_features import _trailing_4_per_player_asof

    # Player active in only weeks 1, 2, 3 (3 games, not 4).
    per_game = pd.DataFrame(
        [{"gsis_id": "00-0000001", "season": 2024, "week": w, "val": float(w)} for w in (1, 2, 3)]
    )
    index = pd.DataFrame(
        [{"gsis_id": "00-0000001", "season": 2024, "week": w} for w in range(1, 6)]
    )
    out = _trailing_4_per_player_asof(per_game, index, value_col="val", out_col="val_l4")

    # All output values should be NaN — never enough prior receiver-active games.
    assert out["val_l4"].isna().all()


def test_trailing_4_asof_cross_season() -> None:
    """Trailing-4 wraps across season boundary if Y has fewer than 4 active games."""
    from projections.features.pbp_receiver_features import _trailing_4_per_player_asof

    # Player active in 2023 weeks 16, 17 and 2024 weeks 1, 2, 3.
    per_game = pd.DataFrame(
        [
            {"gsis_id": "00-0000001", "season": 2023, "week": 16, "val": 16.0},
            {"gsis_id": "00-0000001", "season": 2023, "week": 17, "val": 17.0},
            {"gsis_id": "00-0000001", "season": 2024, "week": 1, "val": 1.0},
            {"gsis_id": "00-0000001", "season": 2024, "week": 2, "val": 2.0},
            {"gsis_id": "00-0000001", "season": 2024, "week": 3, "val": 3.0},
        ]
    )
    index = pd.DataFrame(
        [{"gsis_id": "00-0000001", "season": 2024, "week": w} for w in range(1, 5)]
    )
    out = _trailing_4_per_player_asof(per_game, index, value_col="val", out_col="val_l4")

    # 2024 w4: latest active game strictly before is 2024 w3. rolling-4 at w3 =
    # mean of 4 most recent up-to-and-including w3 = mean(2023 w17, 2024 w1,
    # 2024 w2, 2024 w3) = mean(17, 1, 2, 3) = 5.75.
    w4 = out.query("season == 2024 and week == 4")
    assert w4["val_l4"].iloc[0] == pytest.approx(5.75)
