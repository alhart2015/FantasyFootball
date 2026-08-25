"""LeagueCalendar, and the guarantee that parameterising it changed no existing behaviour.

The bracket in `league_projection` used to be hard-coded for exactly six playoff teams with two
byes, a wildcard week, a semifinal week, and a two-week final. It is now a general
single-elimination ladder driven by a `LeagueCalendar`. The load-bearing test here is
`test_default_calendar_bracket_matches_the_old_hardcoded_one`, which re-implements the old
bracket literally and asserts the general one agrees on 500 random seedings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.league_projection import (
    DEFAULT_CALENDAR,
    SeasonOutcomes,
    resolve_bracket,
    simulate_seasons,
)
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.league_calendar import LeagueCalendar, _byes_for, usable_int
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset


def test_default_calendar_reproduces_the_previous_module_constants() -> None:
    """These were literal module constants before. Drift here silently changes every
    championship percentage the repo has ever reported, so they are pinned to the numbers."""
    c = DEFAULT_CALENDAR
    assert c.reg_week_numbers == tuple(range(1, 14))
    assert c.wildcard_week == 14
    assert c.round_week(1) == 15  # the old SEMIFINAL_WEEK
    assert c.championship_weeks == (16, 17)
    assert c.all_week_numbers == tuple(range(1, 18))
    assert (c.playoff_size, c.n_byes) == (6, 2)
    assert c.n_playoff_rounds == 3


def test_critts_calendar_from_espn_settings() -> None:
    """The league this was built for: ESPN reports 14 regular weeks and one-week playoff
    rounds, against the defaults' 13 and two. That difference is why this class exists."""
    c = LeagueCalendar.from_espn_settings(
        {"matchupPeriodCount": 14, "playoffTeamCount": 6, "playoffMatchupPeriodLength": 1}
    )
    assert (c.reg_weeks, c.playoff_size, c.n_byes, c.final_weeks) == (14, 6, 2, 1)
    assert c.wildcard_week == 15
    assert c.championship_weeks == (17,)
    assert c.total_weeks == 17


def test_from_espn_settings_falls_back_rather_than_raising_on_a_partial_payload() -> None:
    c = LeagueCalendar.from_espn_settings({"matchupPeriodCount": 14})
    assert c.reg_weeks == 14
    assert (c.playoff_size, c.n_byes, c.final_weeks) == (6, 2, 2)


@pytest.mark.parametrize(
    ("playoff_size", "expected_byes"),
    [(2, 0), (4, 0), (6, 2), (8, 0), (10, 6), (12, 4), (16, 0)],
)
def test_byes_are_whatever_makes_the_first_round_a_power_of_two(
    playoff_size: int, expected_byes: int
) -> None:
    byes = _byes_for(playoff_size)
    assert byes == expected_byes
    playing = playoff_size - byes
    assert playing % 2 == 0
    advancing = byes + playing // 2
    assert advancing & (advancing - 1) == 0, "a later round would strand a team"


@pytest.mark.parametrize(
    ("playoff_size", "n_byes", "expected_rounds"),
    [(2, 0, 1), (4, 0, 2), (6, 2, 3), (8, 0, 3), (12, 4, 4), (16, 0, 4)],
)
def test_round_count(playoff_size: int, n_byes: int, expected_rounds: int) -> None:
    assert LeagueCalendar(playoff_size=playoff_size, n_byes=n_byes).n_playoff_rounds == (
        expected_rounds
    )


def test_rejects_a_bracket_where_every_playoff_team_has_a_bye() -> None:
    with pytest.raises(ValueError, match="fewer than playoff_size"):
        LeagueCalendar(playoff_size=4, n_byes=4)


def test_rejects_an_odd_first_round() -> None:
    """An odd `playoff_size - n_byes` leaves one team unpaired, which the zip would silently
    drop rather than fail on."""
    with pytest.raises(ValueError, match="must be even"):
        LeagueCalendar(playoff_size=6, n_byes=1)


# --- the equivalence guarantee -------------------------------------------------------------

_Pts = dict[tuple[int, int], float]


def _old_bracket(seed_order: list[int], pts: _Pts) -> tuple[int, int]:
    """The 6-team/2-bye bracket exactly as written before parameterisation.

    Wildcard wk14 3v6 and 4v5; wk15 reseeded semis, seed 1 against the lower survivor and
    seed 2 against the higher; final summed over wk16+wk17.
    """
    seedpos = {s: r for r, s in enumerate(seed_order)}
    s1, s2, s3, s4, s5, s6 = seed_order[:6]
    win_a = s3 if pts[(s3, 14)] >= pts[(s6, 14)] else s6
    win_b = s4 if pts[(s4, 14)] >= pts[(s5, 14)] else s5
    hi, lo = sorted([win_a, win_b], key=lambda s: seedpos[s])
    f1 = s1 if pts[(s1, 15)] >= pts[(lo, 15)] else lo
    f2 = s2 if pts[(s2, 15)] >= pts[(hi, 15)] else hi
    f1_total = pts[(f1, 16)] + pts[(f1, 17)]
    f2_total = pts[(f2, 16)] + pts[(f2, 17)]
    return (f1, f2) if f1_total >= f2_total else (f2, f1)


def test_default_calendar_bracket_matches_the_old_hardcoded_one() -> None:
    """The regression guard for this whole change.

    Runs the previous hard-coded bracket and the generalized ladder over the same weekly
    points, across 500 random seedings. "The default constants are unchanged" is an assertion
    about numbers only — it would not catch a reseeding or pairing bug inside the new loop.
    """
    rng = np.random.default_rng(11)
    for _ in range(500):
        seed_order = [int(x) for x in rng.permutation(np.arange(1, 9))]
        # Continuous points: an exact tie is the one case the two implementations are allowed
        # to differ on (the old code broke it to seed 1's half, the new one to the better
        # seed), and it does not occur here.
        pts: _Pts = {
            (slot, week): float(rng.normal(110.0, 25.0))
            for slot in seed_order
            for week in (14, 15, 16, 17)
        }

        # `table` bound as a default so the function does not close over the loop variable
        # (ruff B023) — every iteration would otherwise read the last round's points.
        def points(slot: int, week: int, table: _Pts = pts) -> float:
            return table[(slot, week)]

        assert resolve_bracket(seed_order, points, DEFAULT_CALENDAR) == _old_bracket(
            seed_order, pts
        )


def test_bracket_gives_byes_a_first_round_off() -> None:
    """Seeds 1 and 2 must not play in the wildcard round. Losing badly in wk14 and winning
    in wk15 would eliminate them if the bye were not honoured."""
    seed_order = [1, 2, 3, 4, 5, 6]
    pts: _Pts = {}
    for slot in seed_order:
        pts[(slot, 14)] = 1.0 if slot in (1, 2) else 100.0 + slot
        pts[(slot, 15)] = 500.0 if slot in (1, 2) else 10.0
        pts[(slot, 16)] = 300.0 if slot == 1 else 10.0
        pts[(slot, 17)] = 300.0 if slot == 1 else 10.0
    assert resolve_bracket(seed_order, lambda s, w: pts[(s, w)], DEFAULT_CALENDAR) == (1, 2)


def test_bracket_honours_a_one_week_final() -> None:
    """The Critts shape. Seed 1 wins the first championship week by a mile and seed 2 takes
    the second; a two-week final gives it to seed 1, a one-week final to seed 2."""
    one_week = LeagueCalendar(reg_weeks=13, playoff_size=6, n_byes=2, final_weeks=1)
    assert one_week.championship_weeks == (16,)

    seed_order = [1, 2, 3, 4, 5, 6]
    pts: _Pts = {}
    for slot in seed_order:
        pts[(slot, 14)] = 100.0 + slot
        pts[(slot, 15)] = 500.0 if slot in (1, 2) else 10.0
        pts[(slot, 16)] = 5.0 if slot == 1 else 50.0  # seed 2 takes the single final week
        pts[(slot, 17)] = 900.0 if slot == 1 else 1.0  # only counted by a two-week final
    assert resolve_bracket(seed_order, lambda s, w: pts[(s, w)], one_week) == (2, 1)
    assert resolve_bracket(seed_order, lambda s, w: pts[(s, w)], DEFAULT_CALENDAR) == (1, 2)


def test_bracket_handles_a_four_team_no_bye_field() -> None:
    cal = LeagueCalendar(reg_weeks=13, playoff_size=4, n_byes=0, final_weeks=1)
    assert (cal.n_playoff_rounds, cal.championship_weeks) == (2, (15,))
    seed_order = [1, 2, 3, 4, 5, 6, 7, 8]
    # Round 1 (wk14) pairs 1v4 and 2v3; seeds 3 and 4 win, then 3 takes the final.
    pts: _Pts = {}
    for slot in seed_order:
        pts[(slot, 14)] = 200.0 if slot in (3, 4) else 1.0
        pts[(slot, 15)] = 200.0 if slot == 3 else 1.0
    assert resolve_bracket(seed_order, lambda s, w: pts[(s, w)], cal) == (3, 4)


# --- end-to-end through simulate_seasons ---------------------------------------------------


def _roster_frame(n_teams: int) -> tuple[dict[int, list[str]], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    rosters: dict[int, list[str]] = {}
    for seat in range(1, n_teams + 1):
        ids: list[str] = []
        for j, pos in enumerate(("QB", "RB", "RB", "WR", "WR", "TE")):
            gid = f"00-{seat:04d}{j:03d}"
            ids.append(gid)
            rows.append(
                {
                    "gsis_id": gid,
                    "position": pos,
                    "season_mean_fpts": 260.0 - 6.0 * seat - 2.0 * j,
                    "is_rookie": False,
                }
            )
        rosters[seat] = ids
    pool = pd.DataFrame(rows)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    return rosters, pool


def _config(n_teams: int) -> LeagueConfig:
    return LeagueConfig(
        name="calendar_test",
        n_teams=n_teams,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
        },
        ruleset=Ruleset.espn_half(),
    )


def _run(n_teams: int, calendar: LeagueCalendar | None) -> SeasonOutcomes:
    rosters, pool = _roster_frame(n_teams)
    # Everyone always available, no byes: isolates the calendar as the only thing that varies
    # between runs, so a win-count difference can only come from the week count.
    availability = PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={})
    return simulate_seasons(
        rosters,
        pool,
        availability,
        VarianceParams.load(),
        league_config=_config(n_teams),
        n_sims=40,
        rng=np.random.default_rng(7),
        calendar=calendar,
    )


def test_passing_no_calendar_is_identical_to_passing_the_default() -> None:
    """Every existing caller passes nothing. That path must stay byte-identical."""
    implicit, explicit = _run(8, None), _run(8, DEFAULT_CALENDAR)
    assert np.array_equal(implicit.wins, explicit.wins)
    assert np.array_equal(implicit.seed, explicit.seed)
    assert np.array_equal(implicit.champion, explicit.champion)
    assert np.array_equal(implicit.runner_up, explicit.runner_up)


def test_a_longer_regular_season_actually_plays_more_games() -> None:
    """The point of the change. Every week is exactly one win per matchup, so a team's league
    total is n_teams/2 per week — 14 regular weeks must produce 14 games' worth, not 13."""
    short, long = _run(8, LeagueCalendar(reg_weeks=13)), _run(8, LeagueCalendar(reg_weeks=14))
    assert short.wins.sum(axis=1)[0] == 13 * 4
    assert long.wins.sum(axis=1)[0] == 14 * 4


def test_outcomes_carry_the_calendar_they_were_simulated_under() -> None:
    """`seed <= playoff_size` means nothing without knowing which playoff_size, so the result
    carries its own calendar instead of letting a caller assume the module default."""
    res = _run(8, LeagueCalendar(reg_weeks=14, playoff_size=4, n_byes=0, final_weeks=1))
    assert res.calendar.playoff_size == 4
    assert res.made_playoffs(1).sum() == (res.seed[:, 0] <= 4).sum()


# --- usable_int ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (14, 14),
        ("14", 14),
        (14.0, 14),
        # `bool` is an `int` in Python, so a naive type test lets True through as 1 -- a
        # one-week regular season, with every later week discarded from the locked record.
        (True, None),
        (False, None),
        (None, None),
        ("full", None),
        (float("nan"), None),
        (float("inf"), None),
        ([], None),
        ({}, None),
        # LeagueCalendar's own gt=0 would reject these, but only by raising mid-call.
        (0, None),
        (-3, None),
    ],
)
def test_usable_int_accepts_only_what_can_actually_be_a_week_count(
    raw: object, expected: int | None
) -> None:
    assert usable_int(raw) == expected


def test_an_unusable_week_count_falls_back_rather_than_raising() -> None:
    """`from_espn_settings` must degrade to its default, not abort its caller mid-write."""
    for raw in (True, "full", float("nan"), 0, -3, []):
        assert LeagueCalendar.from_espn_settings({"matchupPeriodCount": raw}).reg_weeks == 13
