"""Real fixture list and locked already-played weeks in `simulate_seasons`.

Step 2 of the projected-standings spec. Two capabilities, each with a way of being subtly
wrong that a smoke test would not catch:

- **The real schedule must actually be used.** Passing one and having it silently ignored
  leaves the gauntlet round-robin in place and every number looks plausible. The test here
  gives one team an easy draw and one a brutal one and requires the projection to notice.
- **Locked weeks must be facts, not priors.** With every week played the answer has to be
  the record itself with zero spread — not a tight distribution around it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.league_projection import (
    LockedRecord,
    SeasonOutcomes,
    gauntlet_schedule,
    simulate_seasons,
)
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.league_calendar import LeagueCalendar
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset

_N_TEAMS = 6
_CAL = LeagueCalendar(reg_weeks=4, playoff_size=2, n_byes=0, final_weeks=1)


def _rosters_and_pool(strengths: dict[int, float]) -> tuple[dict[int, list[str]], pd.DataFrame]:
    """One roster per slot, every player scaled by that slot's strength multiplier."""
    rows: list[dict[str, object]] = []
    rosters: dict[int, list[str]] = {}
    for seat, mult in strengths.items():
        ids: list[str] = []
        for j, pos in enumerate(("QB", "RB", "RB", "WR", "WR", "TE")):
            gid = f"00-{seat:04d}{j:03d}"
            ids.append(gid)
            rows.append(
                {
                    "gsis_id": gid,
                    "position": pos,
                    "season_mean_fpts": 200.0 * mult,
                    "is_rookie": False,
                }
            )
        rosters[seat] = ids
    pool = pd.DataFrame(rows)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    return rosters, pool


def _config() -> LeagueConfig:
    return LeagueConfig(
        name="in_season_test",
        n_teams=_N_TEAMS,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
        },
        ruleset=Ruleset.espn_half(),
    )


def _run(
    strengths: dict[int, float],
    *,
    schedule: list[list[tuple[int, int]]] | None = None,
    locked: dict[int, LockedRecord] | None = None,
    first_unplayed_week: int = 1,
    n_sims: int = 200,
) -> SeasonOutcomes:
    rosters, pool = _rosters_and_pool(strengths)
    availability = PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={})
    return simulate_seasons(
        rosters,
        pool,
        availability,
        VarianceParams.load(),
        league_config=_config(),
        n_sims=n_sims,
        rng=np.random.default_rng(3),
        calendar=_CAL,
        schedule=schedule,
        locked=locked,
        first_unplayed_week=first_unplayed_week,
    )


_EVEN = dict.fromkeys(range(1, _N_TEAMS + 1), 1.0)


def test_strength_of_schedule_moves_the_projection() -> None:
    """The load-bearing test for the real fixture list.

    Slot 1 and slot 2 have identical rosters. Slot 1 is scheduled against the two weakest
    teams every week; slot 2 against the two strongest. If the schedule were ignored and the
    gauntlet used instead, they would project the same — that is exactly the silent failure
    this guards.
    """
    strengths = {1: 1.0, 2: 1.0, 3: 1.6, 4: 1.6, 5: 0.4, 6: 0.4}
    # Slot 1 always draws a 0.4 team; slot 2 always draws a 1.6 team.
    easy_vs_hard: list[list[tuple[int, int]]] = [
        [(1, 5), (2, 3), (4, 6)],
        [(1, 6), (2, 4), (3, 5)],
        [(1, 5), (2, 3), (4, 6)],
        [(1, 6), (2, 4), (3, 5)],
    ]
    res = _run(strengths, schedule=easy_vs_hard)
    slot1_wins = res.wins[:, res.slots.index(1)].mean()
    slot2_wins = res.wins[:, res.slots.index(2)].mean()
    assert slot1_wins > slot2_wins + 1.0, (
        f"identical rosters, opposite draws: slot 1 {slot1_wins:.2f} vs slot 2 "
        f"{slot2_wins:.2f} — the schedule is not being used"
    )


def test_a_passed_schedule_is_not_the_gauntlet() -> None:
    """Same rosters, two different fixture lists, must not produce the same records."""
    strengths = {1: 1.0, 2: 1.0, 3: 1.6, 4: 1.6, 5: 0.4, 6: 0.4}
    easy_first: list[list[tuple[int, int]]] = [
        [(1, 5), (2, 3), (4, 6)],
        [(1, 6), (2, 4), (3, 5)],
        [(1, 5), (2, 3), (4, 6)],
        [(1, 6), (2, 4), (3, 5)],
    ]
    custom = _run(strengths, schedule=easy_first)
    default = _run(strengths)
    assert not np.array_equal(custom.wins, default.wins)


def test_schedule_length_must_match_the_calendar() -> None:
    """A three-week schedule against a four-week calendar is a caller bug that would
    otherwise drop a week silently."""
    with pytest.raises(ValueError, match="week-for-week"):
        _run(_EVEN, schedule=[[(1, 2), (3, 4), (5, 6)]] * 3)


def test_passing_no_schedule_still_uses_the_gauntlet() -> None:
    explicit = _run(_EVEN, schedule=gauntlet_schedule(_N_TEAMS, _CAL.reg_weeks))
    implicit = _run(_EVEN)
    assert np.array_equal(implicit.wins, explicit.wins)


# --- locked weeks --------------------------------------------------------------------------


def test_a_fully_played_season_has_no_uncertainty_left() -> None:
    """Every regular-season week locked: projected wins must BE the actual record, identically
    across every simulation. A tight distribution around it would mean weeks were re-simulated
    rather than treated as facts."""
    locked = {
        1: LockedRecord(wins=4, losses=0, points_for=600.0),
        2: LockedRecord(wins=3, losses=1, points_for=500.0),
        3: LockedRecord(wins=2, losses=2, points_for=400.0),
        4: LockedRecord(wins=2, losses=2, points_for=390.0),
        5: LockedRecord(wins=1, losses=3, points_for=300.0),
        6: LockedRecord(wins=0, losses=4, points_for=200.0),
    }
    res = _run(_EVEN, locked=locked, first_unplayed_week=5)  # 5 > reg_weeks: nothing to sim
    for slot, record in locked.items():
        col = res.slots.index(slot)
        assert res.wins[:, col].std() == 0.0, f"slot {slot} record still varies"
        assert res.wins[:, col][0] == record.wins
        assert res.points_for[:, col][0] == pytest.approx(record.points_for)


def test_a_fully_played_season_gives_deterministic_seeds() -> None:
    """The corollary: with the record known, so is the seeding. Playoff odds become 0 or 1."""
    locked = {
        1: LockedRecord(wins=4, losses=0, points_for=600.0),
        2: LockedRecord(wins=3, losses=1, points_for=500.0),
        3: LockedRecord(wins=2, losses=2, points_for=400.0),
        4: LockedRecord(wins=2, losses=2, points_for=390.0),
        5: LockedRecord(wins=1, losses=3, points_for=300.0),
        6: LockedRecord(wins=0, losses=4, points_for=200.0),
    }
    res = _run(_EVEN, locked=locked, first_unplayed_week=5)
    # playoff_size is 2, so exactly slots 1 and 2 are in, in every sim.
    assert res.made_playoffs(1).all()
    assert res.made_playoffs(2).all()
    for slot in (3, 4, 5, 6):
        assert not res.made_playoffs(slot).any()
    # Points-for breaks the 2-2 tie between slots 3 and 4, deterministically.
    assert (res.seed[:, res.slots.index(3)] < res.seed[:, res.slots.index(4)]).all()


def test_banked_wins_carry_into_a_half_played_season() -> None:
    """Week 3 of 4, identical rosters, opposite records. The projection difference must be
    the banked wins — the remaining schedule cannot make up a 2-0 vs 0-2 start."""
    locked = {
        1: LockedRecord(wins=2, losses=0, points_for=260.0),
        2: LockedRecord(wins=0, losses=2, points_for=180.0),
        3: LockedRecord(wins=1, losses=1, points_for=220.0),
        4: LockedRecord(wins=1, losses=1, points_for=220.0),
        5: LockedRecord(wins=1, losses=1, points_for=220.0),
        6: LockedRecord(wins=1, losses=1, points_for=220.0),
    }
    res = _run(_EVEN, locked=locked, first_unplayed_week=3)
    hot = res.wins[:, res.slots.index(1)].mean()
    cold = res.wins[:, res.slots.index(2)].mean()
    assert hot > cold
    # Only weeks 3 and 4 are simulated, so nobody can gain more than 2 wins.
    assert res.wins[:, res.slots.index(2)].max() <= 2
    assert res.wins[:, res.slots.index(1)].min() >= 2


def test_locked_points_seed_the_points_for_tally() -> None:
    """Points-for is the seeding tiebreak, so banked points have to be in it from the start."""
    locked = {1: LockedRecord(wins=1, losses=1, points_for=5000.0)}
    res = _run(_EVEN, locked=locked, first_unplayed_week=3)
    assert res.points_for[:, res.slots.index(1)].min() >= 5000.0
    assert res.points_for[:, res.slots.index(2)].max() < 5000.0


def test_no_locked_records_is_the_preseason_path() -> None:
    """`locked=None, first_unplayed_week=1` must be byte-identical to not passing them."""
    with_none = _run(_EVEN)
    explicit = _run(_EVEN, locked={}, first_unplayed_week=1)
    assert np.array_equal(with_none.wins, explicit.wins)
    assert np.array_equal(with_none.champion, explicit.champion)


def test_locked_record_counts_ties_in_games_played() -> None:
    assert LockedRecord(wins=2, losses=1, ties=1).games_played == 4


# --- ties ----------------------------------------------------------------------------------


def test_a_tie_counts_as_half_a_win_for_seeding() -> None:
    """ESPN seeds on winning percentage, where a tie is half a win.

    `LockedRecord` carried `ties` into the dataclass, the schema and the printed record, and
    then seeding used `wins` alone -- so a 6-1-1 team (.8125) and a 6-2-0 team (.750) were
    treated as identical and separated on points-for instead. That flips playoff berths and
    byes. The dataclass docstring asserted "a tie is neither a win nor a loss for seeding" as
    if it were a decision; the spec says the opposite.
    """
    locked = {
        1: LockedRecord(wins=2, losses=0, ties=2, points_for=400.0),  # 3.0 -> ahead
        2: LockedRecord(wins=3, losses=1, ties=0, points_for=900.0),  # 3.0, more points
        3: LockedRecord(wins=0, losses=4, points_for=100.0),
        4: LockedRecord(wins=0, losses=4, points_for=100.0),
        5: LockedRecord(wins=0, losses=4, points_for=100.0),
        6: LockedRecord(wins=0, losses=4, points_for=100.0),
    }
    res = _run(_EVEN, locked=locked, first_unplayed_week=5)
    # 2 wins + 2 ties == 3.0 credited wins, the same as 3-1-0, so points-for decides and slot
    # 2 leads. The point is that the tie counted at all: on `wins` alone slot 1 sat on 2.0.
    assert res.wins[:, res.slots.index(1)][0] == pytest.approx(3.0)
    assert res.wins[:, res.slots.index(2)][0] == pytest.approx(3.0)


def test_ties_break_ahead_of_a_worse_record_with_more_points() -> None:
    """The case that actually flips a berth: 1-0-2 (.667) must outrank 1-2-0 (.333) even when
    the losing team has scored far more."""
    locked = {
        1: LockedRecord(wins=1, losses=0, ties=2, points_for=100.0),
        2: LockedRecord(wins=1, losses=2, ties=0, points_for=9000.0),
        3: LockedRecord(wins=0, losses=3, points_for=50.0),
        4: LockedRecord(wins=0, losses=3, points_for=50.0),
        5: LockedRecord(wins=0, losses=3, points_for=50.0),
        6: LockedRecord(wins=0, losses=3, points_for=50.0),
    }
    res = _run(_EVEN, locked=locked, first_unplayed_week=5)
    seed_1 = res.seed[:, res.slots.index(1)]
    seed_2 = res.seed[:, res.slots.index(2)]
    assert (seed_1 < seed_2).all(), "a tie must outweigh points-for from a worse record"


def test_head_to_head_of_two_scoreless_teams_is_not_a_double_win() -> None:
    """`head_to_head` uses `>=`, so identical point vectors report 1.0 for BOTH sides and
    P(A beats B) + P(B beats A) = 2.0. The docstring dismissed this as unreachable via
    continuous totals, but `sample_weekly_points` returns exactly 0.0 for a non-positive
    projection, and a zeroed pool (a finished regular season) produces deterministic
    all-zero columns."""
    zeroed = dict.fromkeys(range(1, _N_TEAMS + 1), 0.0)
    res = _run(zeroed, n_sims=40)
    forward = res.head_to_head(1, 2, 1)
    reverse = res.head_to_head(2, 1, 1)
    assert forward + reverse == pytest.approx(1.0)
