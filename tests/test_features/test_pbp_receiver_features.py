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


def test_adot_air_yards_only() -> None:
    """aDOT averages only over rows with non-NaN air_yards (excludes sacks /
    throwaways with NaN air_yards upstream)."""
    from projections.features.pbp_receiver_features import compute_receiver_adot

    rows: list[dict[str, object]] = []
    # Build 4 prior weeks of receiver-active games for player A so trailing-4
    # has a window. Each week: 5 targets with air_yards = 10, plus 1 sack
    # (NaN air_yards). Mean over weeks 1-4 should be 10.0 (sacks excluded).
    gid = "00-0000001"
    for wk in range(1, 5):
        for i in range(5):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "play_id": 1000 * wk + i,
                    "receiver_player_id": gid,
                    "pass_attempt": 1.0,
                    "air_yards": 10.0,
                }
            )
        # 1 sack on the same player (would not actually credit air_yards to
        # receiver, but defensive: NaN air_yards excluded from mean).
        rows.append(
            {
                "season": 2024,
                "week": wk,
                "play_id": 1000 * wk + 99,
                "receiver_player_id": gid,
                "pass_attempt": 0.0,  # not a target — sack
                "air_yards": float("nan"),
            }
        )

    pbp = _make_pbp_rows(rows)
    per_game = compute_receiver_adot(pbp)
    # per_game has rows for receiver-active games (weeks 1-4) but no rolling
    # column attached — that's the helper's job. compute_receiver_adot returns
    # the per-game-mean frame ready for _trailing_4_per_player_asof.
    assert set(per_game.columns) == {"gsis_id", "season", "week", "aDOT_l4"}
    # Per-game mean for week 1: 5 targets at 10 yards = mean 10.0.
    wk1 = per_game.query("week == 1")
    assert wk1["aDOT_l4"].iloc[0] == pytest.approx(10.0)


def test_adot_trailing_4_within_player() -> None:
    """6 receiver-active games for A and B; per-game means stay within gsis_id."""
    from projections.features.pbp_receiver_features import compute_receiver_adot

    rows: list[dict[str, object]] = []
    for gid, base_yards in [("00-0000001", 5.0), ("00-0000002", 15.0)]:
        for wk in range(1, 7):
            for i in range(3):  # 3 targets per game
                rows.append(
                    {
                        "season": 2024,
                        "week": wk,
                        "play_id": 1000 * wk + (0 if gid == "00-0000001" else 500) + i,
                        "receiver_player_id": gid,
                        "pass_attempt": 1.0,
                        "air_yards": base_yards + wk,  # A: 6,7,...; B: 16,17,...
                    }
                )

    pbp = _make_pbp_rows(rows)
    per_game = compute_receiver_adot(pbp)
    # per_game has 12 rows (6 weeks x 2 players).
    assert len(per_game) == 12
    # Per-game mean for player A, week 1: 5+1 = 6.0.
    a_wk1 = per_game.query("gsis_id == '00-0000001' and week == 1")
    assert a_wk1["aDOT_l4"].iloc[0] == pytest.approx(6.0)


def test_adot_zero_air_yards_targets_no_per_game_row() -> None:
    """If a receiver has zero non-NaN air_yards targets in a week, no per-game
    row is emitted (the player has no per-game value for that week)."""
    from projections.features.pbp_receiver_features import compute_receiver_adot

    gid = "00-0000001"
    # Week 1: 0 valid targets (only sacks). Week 2: 3 valid targets.
    rows: list[dict[str, object]] = [
        {
            "season": 2024,
            "week": 1,
            "play_id": 100,
            "receiver_player_id": gid,
            "pass_attempt": 0.0,
            "air_yards": float("nan"),
        },
        {
            "season": 2024,
            "week": 2,
            "play_id": 200,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "air_yards": 12.0,
        },
        {
            "season": 2024,
            "week": 2,
            "play_id": 201,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "air_yards": 18.0,
        },
    ]
    pbp = _make_pbp_rows(rows)
    per_game = compute_receiver_adot(pbp)
    # No row for week 1 (no valid targets); one row for week 2.
    assert len(per_game) == 1
    assert per_game.iloc[0]["week"] == 2
    assert per_game.iloc[0]["aDOT_l4"] == pytest.approx(15.0)


def test_deep_target_share_threshold() -> None:
    """Targets with air_yards >= 20 count as deep; < 20 do not. Throwaways
    (pass_attempt=1.0 with NaN air_yards) excluded from both numerator and
    denominator."""
    from projections.features.pbp_receiver_features import compute_receiver_deep_target_share

    gid = "00-0000001"
    # 6 valid targets at depths 5/15/19/20/25/35 (3 deep, 3 shallow); plus
    # 1 throwaway (pass_attempt=1.0 but NaN air_yards) that must be excluded
    # from both numerator and denominator. Expected share: 3/6 = 0.5.
    depths = [5.0, 15.0, 19.0, 20.0, 25.0, 35.0]
    rows: list[dict[str, object]] = [
        {
            "season": 2024,
            "week": 1,
            "play_id": 100 + i,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "air_yards": d,
        }
        for i, d in enumerate(depths)
    ]
    # Throwaway row — must be filtered out (NaN air_yards excluded), not
    # treated as a 0-depth shallow target.
    rows.append(
        {
            "season": 2024,
            "week": 1,
            "play_id": 200,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "air_yards": float("nan"),
        }
    )
    pbp = _make_pbp_rows(rows)
    per_game = compute_receiver_deep_target_share(pbp)
    assert len(per_game) == 1
    # 3 deep (20, 25, 35) / 6 valid targets (throwaway excluded) = 0.5.
    assert per_game.iloc[0]["deep_target_share_l4"] == pytest.approx(3 / 6)


def test_deep_target_share_zero_targets_no_per_game_row() -> None:
    """A receiver with 0 valid targets in a week contributes no per-game row."""
    from projections.features.pbp_receiver_features import compute_receiver_deep_target_share

    gid = "00-0000001"
    # Only a sack (no valid targets).
    rows = [
        {
            "season": 2024,
            "week": 1,
            "play_id": 100,
            "receiver_player_id": gid,
            "pass_attempt": 0.0,
            "air_yards": float("nan"),
        }
    ]
    pbp = _make_pbp_rows(rows)
    per_game = compute_receiver_deep_target_share(pbp)
    assert len(per_game) == 0


def test_yac_completions_only() -> None:
    """YAC averages only over completions (complete_pass == 1.0); incompletions
    excluded even when receiver_player_id is set."""
    from projections.features.pbp_receiver_features import compute_receiver_yac_per_reception

    gid = "00-0000001"
    rows = [
        # 3 completions with YAC 5, 10, 15.
        {
            "season": 2024,
            "week": 1,
            "play_id": 100,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "complete_pass": 1.0,
            "yards_after_catch": 5.0,
        },
        {
            "season": 2024,
            "week": 1,
            "play_id": 101,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "complete_pass": 1.0,
            "yards_after_catch": 10.0,
        },
        {
            "season": 2024,
            "week": 1,
            "play_id": 102,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "complete_pass": 1.0,
            "yards_after_catch": 15.0,
        },
        # 2 incompletions (NaN YAC). Should NOT contribute.
        {
            "season": 2024,
            "week": 1,
            "play_id": 103,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "complete_pass": 0.0,
            "yards_after_catch": float("nan"),
        },
        {
            "season": 2024,
            "week": 1,
            "play_id": 104,
            "receiver_player_id": gid,
            "pass_attempt": 1.0,
            "complete_pass": 0.0,
            "yards_after_catch": float("nan"),
        },
    ]
    pbp = _make_pbp_rows(rows)
    per_game = compute_receiver_yac_per_reception(pbp)
    # Mean of (5, 10, 15) = 10.0.
    assert len(per_game) == 1
    assert per_game.iloc[0]["yac_per_reception_l4"] == pytest.approx(10.0)
