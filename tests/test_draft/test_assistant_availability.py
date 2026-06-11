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


def test_byes_ignore_playoff_weeks() -> None:
    # Real schedule partitions include playoff weeks 19-22; the single-gap bye
    # rule must restrict to the regular season (spec 3.1: weeks 1..18) or every
    # team "misses" the playoff weeks it didn't reach and no bye is ever found.
    base = _schedules(2026, {"AA": 7, "BB": 9})  # regular-season weeks 1-18
    playoffs = pd.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026],
            "week": [19, 20, 21, 22],
            "home_team": pd.array(["CC", "CC", "CC", "CC"], dtype=_PYARROW_STR),
            "away_team": pd.array(["DD", "DD", "DD", "DD"], dtype=_PYARROW_STR),
        }
    )
    sched = pd.concat([base, playoffs], ignore_index=True)
    ws = _weekly_stats([("00-0000001", 2022, w, "RB") for w in range(1, 18)])
    avail = build_availability(
        ws, sched, _id_map([("00-0000001", "AA")]), _pool([("00-0000001", "RB")]), season=2026
    )
    assert avail.bye_week("00-0000001") == 7  # AA's bye, not lost to the playoff rows


def test_playoff_weeks_excluded_from_availability() -> None:
    # weekly_stats partitions carry playoff weeks (19-22). Availability is a
    # regular-season concept, so playoff rows must not count as games played
    # (which inflates the numerator and can mask a regular-season injury) nor set
    # first_week (a playoff-only season would otherwise collapse to a spurious 1.0).
    ws = _weekly_stats(
        # P1: 10 regular-season games (out after wk 10) + 3 playoff games.
        [("00-0000001", 2023, w, "RB") for w in range(1, 11)]
        + [("00-0000001", 2023, w, "RB") for w in (19, 20, 21)]
        # P2: a 9/17 regular 2022 season + a playoff-ONLY 2023 (no reg-season rows).
        + [("00-0000002", 2022, w, "RB") for w in range(1, 10)]
        + [("00-0000002", 2023, w, "RB") for w in (19, 20, 21)]
    )
    sched = _schedules(2026, {"AA": 7, "BB": 9})
    id_map = _id_map([("00-0000001", "AA"), ("00-0000002", "BB")])
    pool = _pool([("00-0000001", "RB"), ("00-0000002", "RB")])
    avail = build_availability(ws, sched, id_map, pool, season=2026)
    # P1: only the 10 regular-season games count -> 10/17, not (10+3)/17.
    assert avail.p_week("00-0000001") == pytest.approx(10 / 17, abs=1e-9)
    # P2: the playoff-only 2023 season is dropped -> p is the 2022 frac 9/17,
    # not mean(9/17, 1.0) that a 1-week playoff span would have produced.
    assert avail.p_week("00-0000002") == pytest.approx(9 / 17, abs=1e-9)


def test_midseason_debut_not_penalized_as_injury() -> None:
    # Availability is measured over the player's ACTIVE span, so a mid-season
    # debut (rookie / trade / call-up) is not conflated with injury risk.
    # MID debuts week 6 then plays every game (12 games, bye at 10) -> fully
    # available once active. INJ debuts week 1, plays 12, then is injured out for
    # the rest -> genuinely missed games. Same game count, very different p.
    mid_weeks = [w for w in range(6, 19) if w != 10]  # weeks 6-18 minus bye 10 = 12 games
    inj_weeks = list(range(1, 13))  # weeks 1-12 = 12 games, then out
    ws = _weekly_stats(
        [("00-0000010", 2024, w, "RB") for w in mid_weeks]
        + [("00-0000011", 2024, w, "RB") for w in inj_weeks]
    )
    sched = _schedules(2026, {"MM": 7, "NN": 8})
    id_map = _id_map([("00-0000010", "MM"), ("00-0000011", "NN")])
    pool = _pool([("00-0000010", "RB"), ("00-0000011", "RB")])
    avail = build_availability(ws, sched, id_map, pool, season=2026)
    # MID: active span = weeks 6..18 -> denom 17-(6-1)=12, games 12 -> frac 1.0 -> clamp hi
    assert avail.p_week("00-0000010") == pytest.approx(0.97)
    # INJ: debut week 1 -> denom 17, games 12 -> 12/17, the injury weeks count
    assert avail.p_week("00-0000011") == pytest.approx(12 / 17, abs=1e-9)


def test_pool_player_absent_from_id_map_has_no_bye() -> None:
    ws = _weekly_stats([("00-0000001", 2022, w, "WR") for w in range(1, 15)])
    sched = _schedules(2026, {"AA": 9})
    # id_map does NOT contain 00-0000001
    avail = build_availability(
        ws, sched, _id_map([("00-0000099", "AA")]), _pool([("00-0000001", "WR")]), season=2026
    )
    assert avail.bye_week("00-0000001") is None
    assert avail.p_week("00-0000001") == pytest.approx(14 / 17, abs=1e-9)
