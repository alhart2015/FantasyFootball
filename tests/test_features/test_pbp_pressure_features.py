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


def test_def_sack_rate_basic() -> None:
    """Defensive sack rate = sum(sack) / sum(qb_dropback) grouped by defteam."""
    from projections.features.pbp_pressure_features import compute_team_def_sack_rate

    rows: list[dict[str, object]] = []
    # KC defense plays BAL each week: 10 dropbacks/wk, 3 sacks → 0.30.
    for wk in range(1, 6):
        for i in range(3):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "defteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 1.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(7):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "defteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 0.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_def_sack_rate(pbp)
    wk5_kc = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5_kc["team_def_sack_rate_l4"].iloc[0] == pytest.approx(0.30)


def test_def_sack_rate_groups_by_defteam() -> None:
    """Two defenses facing different offensive scripts produce different rates."""
    from projections.features.pbp_pressure_features import compute_team_def_sack_rate

    rows: list[dict[str, object]] = []
    # KC defense gets 4 sacks per 10 dropbacks vs BAL.
    # CIN defense gets 1 sack per 10 dropbacks vs CLE.
    for wk in range(1, 6):
        for i in range(4):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "defteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 1.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(6):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "defteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 0.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
        for i in range(1):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "CLE",
                    "defteam": "CIN",
                    "qb_dropback": 1.0,
                    "sack": 1.0,
                    "play_id": 200 * wk + i,
                }
            )
        for i in range(9):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "CLE",
                    "defteam": "CIN",
                    "qb_dropback": 1.0,
                    "sack": 0.0,
                    "play_id": 200 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_def_sack_rate(pbp)
    wk5_kc = out.query("team == 'KC' and season == 2024 and week == 5")
    wk5_cin = out.query("team == 'CIN' and season == 2024 and week == 5")
    assert wk5_kc["team_def_sack_rate_l4"].iloc[0] == pytest.approx(0.40)
    assert wk5_cin["team_def_sack_rate_l4"].iloc[0] == pytest.approx(0.10)


def test_def_scramble_rate_basic() -> None:
    """Defensive scramble rate = sum(qb_scramble) / sum(qb_dropback) grouped by defteam."""
    from projections.features.pbp_pressure_features import compute_team_def_scramble_rate

    rows: list[dict[str, object]] = []
    # KC defense forces opposing QBs to scramble at 0.20 rate.
    for wk in range(1, 6):
        for i in range(2):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "defteam": "KC",
                    "qb_dropback": 1.0,
                    "qb_scramble": 1.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(8):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "BAL",
                    "defteam": "KC",
                    "qb_dropback": 1.0,
                    "qb_scramble": 0.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    out = compute_team_def_scramble_rate(pbp)
    wk5_kc = out.query("team == 'KC' and season == 2024 and week == 5")
    assert wk5_kc["team_def_scramble_rate_l4"].iloc[0] == pytest.approx(0.20)


def test_trailing_4_crosses_season_boundary_via_concat() -> None:
    """When the caller concats Y-1 + Y PBP, the trailing-4 helper carries
    history forward by date order — week 1 of Y reflects the last 4 games
    of Y-1 (matches PR #20's backfill semantics)."""
    from projections.features.pbp_pressure_features import compute_team_sack_rate_allowed

    rows: list[dict[str, object]] = []
    # KC: weeks 14-17 of 2023 (4 prior games), then week 1 of 2024.
    # 2023 sack rate is 0.20 each week; week 1 2024 trailing-4 should be 0.20.
    for wk in range(14, 18):
        for i in range(2):
            rows.append(
                {
                    "season": 2023,
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
                    "season": 2023,
                    "week": wk,
                    "posteam": "KC",
                    "qb_dropback": 1.0,
                    "sack": 0.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
    rows.append(
        {
            "season": 2024,
            "week": 1,
            "posteam": "KC",
            "qb_dropback": 1.0,
            "sack": 0.0,
            "play_id": 1000,
        }
    )
    pbp = _make_pbp_rows(rows)
    out = compute_team_sack_rate_allowed(pbp)
    wk1_2024 = out.query("team == 'KC' and season == 2024 and week == 1")
    assert wk1_2024["team_sack_rate_allowed_l4"].iloc[0] == pytest.approx(0.20)


def _make_index_row(gsis_id: str, season: int, week: int, team: str, opp: str) -> dict[str, object]:
    return {"gsis_id": gsis_id, "season": season, "week": week, "team": team, "opp": opp}


def test_attach_offensive_features_join_on_team() -> None:
    """team_sack_rate_allowed_l4 / team_qb_scramble_rate_l4 attach via the player's
    own team."""
    from projections.features.pbp_pressure_features import attach_pbp_pressure_features

    rows: list[dict[str, object]] = []
    # 4 prior weeks for KC: 10 dropbacks/wk, 2 sacks + 3 scrambles per week.
    # Expected: sack_rate=0.20, scramble_rate=0.30.
    for wk in range(1, 6):
        for i in range(2):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "defteam": "BAL",
                    "qb_dropback": 1.0,
                    "qb_scramble": 0.0,
                    "sack": 1.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(3):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "defteam": "BAL",
                    "qb_dropback": 1.0,
                    "qb_scramble": 1.0,
                    "sack": 0.0,
                    "play_id": 100 * wk + 30 + i,
                }
            )
        for i in range(5):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "defteam": "BAL",
                    "qb_dropback": 1.0,
                    "qb_scramble": 0.0,
                    "sack": 0.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    index = pd.DataFrame([_make_index_row("00-0011111", 2024, 5, "KC", "BAL")])
    out = attach_pbp_pressure_features(index, pbp)
    assert len(out) == 1
    assert out["team_sack_rate_allowed_l4"].iloc[0] == pytest.approx(0.20)
    assert out["team_qb_scramble_rate_l4"].iloc[0] == pytest.approx(0.30)


def test_attach_defensive_features_join_on_opp() -> None:
    """team_def_sack_rate_l4 / team_def_scramble_rate_l4 attach via the
    player's *opponent's* team."""
    from projections.features.pbp_pressure_features import attach_pbp_pressure_features

    rows: list[dict[str, object]] = []
    # 4 prior weeks: BAL defense forces 0.40 sack rate and 0.10 scramble rate.
    # KC plays BAL in wk5; the player's row should pick up BAL's def rates.
    for wk in range(1, 6):
        for i in range(4):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "defteam": "BAL",
                    "qb_dropback": 1.0,
                    "qb_scramble": 0.0,
                    "sack": 1.0,
                    "play_id": 100 * wk + i,
                }
            )
        for i in range(1):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "defteam": "BAL",
                    "qb_dropback": 1.0,
                    "qb_scramble": 1.0,
                    "sack": 0.0,
                    "play_id": 100 * wk + 30 + i,
                }
            )
        for i in range(5):
            rows.append(
                {
                    "season": 2024,
                    "week": wk,
                    "posteam": "KC",
                    "defteam": "BAL",
                    "qb_dropback": 1.0,
                    "qb_scramble": 0.0,
                    "sack": 0.0,
                    "play_id": 100 * wk + 50 + i,
                }
            )
    pbp = _make_pbp_rows(rows)
    index = pd.DataFrame([_make_index_row("00-0011111", 2024, 5, "KC", "BAL")])
    out = attach_pbp_pressure_features(index, pbp)
    assert len(out) == 1
    assert out["team_def_sack_rate_l4"].iloc[0] == pytest.approx(0.40)
    assert out["team_def_scramble_rate_l4"].iloc[0] == pytest.approx(0.10)


def test_attach_empty_pbp_short_circuits_to_nan() -> None:
    """Empty PBP frame → all-NaN columns appended; row count preserved."""
    from projections.features.pbp_pressure_features import attach_pbp_pressure_features

    pbp = _make_pbp_rows([{"play_id": 1}]).iloc[0:0]  # truly empty, schema preserved
    index = pd.DataFrame(
        [
            _make_index_row("00-0011111", 2024, 5, "KC", "BAL"),
            _make_index_row("00-0022222", 2024, 5, "KC", "BAL"),
        ]
    )
    out = attach_pbp_pressure_features(index, pbp)
    assert len(out) == 2
    for col in (
        "team_sack_rate_allowed_l4",
        "team_qb_scramble_rate_l4",
        "team_def_sack_rate_l4",
        "team_def_scramble_rate_l4",
    ):
        assert out[col].isna().all()


def test_assembler_returns_4_feature_columns() -> None:
    """Output schema matches the spec: gsis_id, season, week +
    4 pressure feature columns in declared order."""
    from projections.features.pbp_pressure_features import build_pbp_pressure_overrides

    rows: list[dict[str, object]] = []
    for wk in range(1, 6):
        rows.append(
            {
                "season": 2024,
                "week": wk,
                "posteam": "KC",
                "defteam": "BAL",
                "qb_dropback": 1.0,
                "qb_scramble": 0.0,
                "sack": 0.0,
                "play_id": 100 * wk,
            }
        )
    pbp = _make_pbp_rows(rows)
    index = pd.DataFrame([_make_index_row("00-0011111", 2024, 5, "KC", "BAL")])
    out = build_pbp_pressure_overrides(pbp, index)
    assert list(out.columns) == [
        "gsis_id",
        "season",
        "week",
        "team_sack_rate_allowed_l4",
        "team_qb_scramble_rate_l4",
        "team_def_sack_rate_l4",
        "team_def_scramble_rate_l4",
    ]


def test_assembler_canonical_teams_pass_through() -> None:
    """Canonical team codes (per normalize_team_code; e.g. JAC not JAX)
    pass through cleanly. The assembler is a passive consumer — ingest
    schemas validate canonical codes upstream."""
    from projections.features.pbp_pressure_features import build_pbp_pressure_overrides

    rows: list[dict[str, object]] = []
    for wk in range(1, 6):
        rows.append(
            {
                "season": 2024,
                "week": wk,
                "posteam": "JAC",
                "defteam": "TEN",
                "qb_dropback": 1.0,
                "qb_scramble": 0.0,
                "sack": 0.0,
                "play_id": 100 * wk,
            }
        )
    pbp = _make_pbp_rows(rows)
    index = pd.DataFrame([_make_index_row("00-0011111", 2024, 5, "JAC", "TEN")])
    out = build_pbp_pressure_overrides(pbp, index)
    assert len(out) == 1


def test_assembler_raises_on_invalid_gsis_id() -> None:
    """Malformed gsis_id strings raise ValueError."""
    from projections.features.pbp_pressure_features import build_pbp_pressure_overrides

    pbp = _make_pbp_rows([{"season": 2024, "week": 1, "posteam": "KC"}])
    index = pd.DataFrame([_make_index_row("not-a-real-id", 2024, 1, "KC", "BAL")])
    with pytest.raises(ValueError, match="invalid gsis_id format"):
        build_pbp_pressure_overrides(pbp, index)


def test_assembler_raises_on_duplicate_keys() -> None:
    """Duplicate (gsis_id, season, week) in the index raises ValueError."""
    from projections.features.pbp_pressure_features import build_pbp_pressure_overrides

    pbp = _make_pbp_rows([{"season": 2024, "week": 1, "posteam": "KC"}])
    index = pd.DataFrame(
        [
            _make_index_row("00-0011111", 2024, 1, "KC", "BAL"),
            _make_index_row("00-0011111", 2024, 1, "KC", "BAL"),  # dup
        ]
    )
    with pytest.raises(ValueError, match="duplicate \\(gsis_id, season, week\\) keys"):
        build_pbp_pressure_overrides(pbp, index)


def test_assembler_preserves_row_count() -> None:
    """Output row count matches the input index row count exactly."""
    from projections.features.pbp_pressure_features import build_pbp_pressure_overrides

    rows: list[dict[str, object]] = []
    for wk in range(1, 6):
        rows.append(
            {
                "season": 2024,
                "week": wk,
                "posteam": "KC",
                "defteam": "BAL",
                "qb_dropback": 1.0,
                "qb_scramble": 0.0,
                "sack": 0.0,
                "play_id": 100 * wk,
            }
        )
    pbp = _make_pbp_rows(rows)
    index = pd.DataFrame(
        [
            _make_index_row("00-0011111", 2024, 5, "KC", "BAL"),
            _make_index_row("00-0022222", 2024, 5, "KC", "BAL"),
            _make_index_row("00-0033333", 2024, 5, "KC", "BAL"),
        ]
    )
    out = build_pbp_pressure_overrides(pbp, index)
    assert len(out) == 3
