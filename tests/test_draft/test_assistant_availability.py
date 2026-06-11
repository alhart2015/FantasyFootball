"""Tests for the per-player season availability model."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.draft.assistant.availability import build_availability
from projections.schemas import _PYARROW_STR


def _weekly_stats(rows: list[tuple[str, int, int, str]]) -> pd.DataFrame:
    """rows = [(gsis_id, season, week, position), ...]."""
    return pd.DataFrame(
        {
            "gsis_id": pd.array([r[0] for r in rows], dtype=_PYARROW_STR),
            "season": [r[1] for r in rows],
            "week": [r[2] for r in rows],
            "position": pd.array([r[3] for r in rows], dtype=_PYARROW_STR),
        }
    )


def _schedules(season: int, byes: dict[str, int], n_weeks: int = 18) -> pd.DataFrame:
    """Build a schedule where each team in `byes` is missing exactly its bye week.
    Two teams (A1/A2) pair up every week except their byes; enough to exercise the
    'week with no game row for the team' rule."""
    rows = []
    teams = list(byes)
    for w in range(1, n_weeks + 1):
        playing = [t for t in teams if byes[t] != w]
        # pair them arbitrarily; odd one out plays a filler 'ZZ'
        for i in range(0, len(playing) - 1, 2):
            rows.append((season, w, playing[i], playing[i + 1]))
        if len(playing) % 2 == 1:
            rows.append((season, w, playing[-1], "ZZ"))
    return pd.DataFrame(
        {
            "season": [r[0] for r in rows],
            "week": [r[1] for r in rows],
            "home_team": pd.array([r[2] for r in rows], dtype=_PYARROW_STR),
            "away_team": pd.array([r[3] for r in rows], dtype=_PYARROW_STR),
        }
    )


def _id_map(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """rows = [(gsis_id, team), ...]."""
    return pd.DataFrame(
        {
            "gsis_id": pd.array([r[0] for r in rows], dtype=_PYARROW_STR),
            "team": pd.array([r[1] for r in rows], dtype=_PYARROW_STR),
        }
    )


def _pool(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """rows = [(gsis_id, position), ...]."""
    return pd.DataFrame(
        {
            "gsis_id": pd.array([r[0] for r in rows], dtype=_PYARROW_STR),
            "position": pd.array([r[1] for r in rows], dtype=_PYARROW_STR),
        }
    )


def test_workhorse_and_injury_prone_and_rookie() -> None:
    # A: 17/17 in a 17-game season (clamped to hi). B: 9/17 (injury-prone). R: rookie, no history.
    ws = _weekly_stats(
        [("00-0000001", 2022, w, "RB") for w in range(1, 18)]  # A plays all 17
        + [("00-0000002", 2022, w, "RB") for w in range(1, 10)]  # B plays 9
    )
    sched = _schedules(2026, {"AA": 7, "BB": 9, "RR": 5})
    id_map = _id_map([("00-0000001", "AA"), ("00-0000002", "BB"), ("00-0000009", "RR")])
    pool = _pool([("00-0000001", "RB"), ("00-0000002", "RB"), ("00-0000009", "RB")])

    avail = build_availability(ws, sched, id_map, pool, season=2026)

    assert avail.p_week("00-0000001") == pytest.approx(0.97)  # clamped to hi
    assert avail.p_week("00-0000002") == pytest.approx(9 / 17, abs=1e-9)
    assert 0.4 <= avail.p_week("00-0000009") <= 0.97  # rookie -> RB position default
    assert avail.bye_week("00-0000001") == 7
    assert avail.bye_week("00-0000002") == 9
    assert avail.bye_week("00-0000009") == 5


def test_16_game_era_is_normalized() -> None:
    # A plays all 16 of a 2019 (16-game) season -> frac 1.0, not 16/17.
    ws = _weekly_stats([("00-0000001", 2019, w, "WR") for w in range(1, 17)])
    sched = _schedules(2026, {"AA": 6})
    avail = build_availability(
        ws, sched, _id_map([("00-0000001", "AA")]), _pool([("00-0000001", "WR")]), season=2026
    )
    assert avail.p_week("00-0000001") == pytest.approx(0.97)  # 16/16 = 1.0 -> clamp hi


def test_missing_schedule_degrades_to_no_byes() -> None:
    ws = _weekly_stats([("00-0000001", 2022, w, "RB") for w in range(1, 12)])
    sched = _schedules(2025, {"AA": 7})  # wrong season -> no 2026 rows
    with pytest.warns(UserWarning, match="no schedules for season 2026"):
        avail = build_availability(
            ws, sched, _id_map([("00-0000001", "AA")]), _pool([("00-0000001", "RB")]), season=2026
        )
    assert avail.bye_week("00-0000001") is None
    assert avail.p_week("00-0000001") == pytest.approx(11 / 17, abs=1e-9)


def test_cross_era_averaging() -> None:
    # All 16 of a 2019 (16-game) season AND all 17 of a 2022 (17-game) season:
    # per-season fracs are 1.0 and 1.0 -> mean 1.0 -> clamp hi.
    ws = _weekly_stats(
        [("00-0000001", 2019, w, "RB") for w in range(1, 17)]
        + [("00-0000001", 2022, w, "RB") for w in range(1, 18)]
    )
    sched = _schedules(2026, {"AA": 8})
    avail = build_availability(
        ws, sched, _id_map([("00-0000001", "AA")]), _pool([("00-0000001", "RB")]), season=2026
    )
    assert avail.p_week("00-0000001") == pytest.approx(0.97)


def test_pool_player_absent_from_id_map_has_no_bye() -> None:
    ws = _weekly_stats([("00-0000001", 2022, w, "WR") for w in range(1, 15)])
    sched = _schedules(2026, {"AA": 9})
    # id_map does NOT contain 00-0000001
    avail = build_availability(
        ws, sched, _id_map([("00-0000099", "AA")]), _pool([("00-0000001", "WR")]), season=2026
    )
    assert avail.bye_week("00-0000001") is None
    assert avail.p_week("00-0000001") == pytest.approx(14 / 17, abs=1e-9)
