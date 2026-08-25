"""Projected standings and matchup odds: the assembly layer.

Step 4/5 of the projected-standings spec. The interesting failures here are translation
failures, not maths ones — ESPN speaks in arbitrary team ids and the simulator in contiguous
slots, and a mapping that is merely *plausible* rather than correct produces standings that
attribute the right numbers to the wrong teams.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.league_projection import (
    LockedRecord,
    SeasonOutcomes,
    simulate_seasons,
)
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.league_calendar import LeagueCalendar
from projections.draft.league_config import LeagueConfig
from projections.midseason.standings import (
    SlotMap,
    rosters_to_slots,
    build_matchup_odds,
    build_standings,
    first_unplayed_week,
    locked_by_slot,
    schedule_to_slots,
)
from projections.schemas import (
    _PYARROW_STR,
    MatchupOddsSchema,
    ProjectedStandingsSchema,
    RosterSlot,
    Ruleset,
)
from projections.store import read_partition, write_partition

_CAL = LeagueCalendar(reg_weeks=4, playoff_size=2, n_byes=0, final_weeks=1)
#: Deliberately non-contiguous and unsorted, like a real ESPN league with deleted franchises.
_TEAM_IDS = [17, 3, 11, 5, 9, 1]


def _schedule(played_weeks: int = 0) -> pd.DataFrame:
    """Round-robin over `_TEAM_IDS`, with the first `played_weeks` weeks decided."""
    pairings = [
        [(17, 3), (11, 5), (9, 1)],
        [(17, 11), (3, 9), (5, 1)],
        [(17, 5), (11, 9), (3, 1)],
        [(17, 9), (5, 3), (11, 1)],
    ]
    rows: list[dict[str, object]] = []
    for week, games in enumerate(pairings, start=1):
        for home, away in games:
            played = week <= played_weeks
            rows.append(
                {
                    "week": week,
                    "home_team_id": home,
                    "away_team_id": away,
                    "home_team": f"T{home}",
                    "away_team": f"T{away}",
                    "home_points": 120.0 if played else 0.0,
                    "away_points": 90.0 if played else 0.0,
                    "winner": "HOME" if played else "UNDECIDED",
                    "is_played": played,
                }
            )
    frame = pd.DataFrame(rows)
    for column in ("home_team", "away_team", "winner"):
        frame[column] = frame[column].astype(_PYARROW_STR)
    return frame


# --- SlotMap ------------------------------------------------------------------------------


def test_slot_map_round_trips_every_team_id() -> None:
    slots = SlotMap.from_team_ids(_TEAM_IDS)
    for team_id in _TEAM_IDS:
        assert slots.team_id(slots.slot(team_id)) == team_id


def test_slot_map_is_contiguous_from_one_whatever_the_ids_are() -> None:
    """The simulator requires slots 1..n_teams. Non-contiguous ESPN ids (Critts runs 1..17
    with a gap) must not leak through as slot numbers."""
    slots = SlotMap.from_team_ids(_TEAM_IDS)
    assert sorted(slots.slot(t) for t in _TEAM_IDS) == list(range(1, len(_TEAM_IDS) + 1))


def test_slot_map_is_stable_across_input_order() -> None:
    """A snapshot written this week has to be comparable with last week's, so the same league
    must always map the same way regardless of the order ids arrive in."""
    a = SlotMap.from_team_ids(_TEAM_IDS)
    b = SlotMap.from_team_ids(list(reversed(_TEAM_IDS)))
    assert a.team_ids == b.team_ids


# --- week detection -----------------------------------------------------------------------


@pytest.mark.parametrize("played", [0, 1, 2, 3])
def test_first_unplayed_week_follows_the_results(played: int) -> None:
    assert first_unplayed_week(_schedule(played_weeks=played), _CAL) == played + 1


def test_a_finished_regular_season_reports_one_past_the_end() -> None:
    """Leaves the simulator nothing to simulate, which is right: the standings are known."""
    assert first_unplayed_week(_schedule(played_weeks=4), _CAL) == _CAL.reg_weeks + 1


# --- schedule translation -----------------------------------------------------------------


def test_schedule_to_slots_covers_every_week_in_slot_space() -> None:
    slots = SlotMap.from_team_ids(_TEAM_IDS)
    fixtures = schedule_to_slots(_schedule(), slots, _CAL)
    assert len(fixtures) == _CAL.reg_weeks
    for week_games in fixtures:
        seats = [s for pair in week_games for s in pair]
        assert sorted(seats) == list(range(1, len(_TEAM_IDS) + 1)), "every team plays once"


def test_a_missing_week_raises_rather_than_shortening_the_season() -> None:
    """Silently dropping a week would lower every record with no indication why."""
    slots = SlotMap.from_team_ids(_TEAM_IDS)
    partial = _schedule()
    partial = partial[partial["week"] != 3]
    with pytest.raises(ValueError, match="week"):
        schedule_to_slots(partial, slots, _CAL)


def test_locked_by_slot_translates_records() -> None:
    slots = SlotMap.from_team_ids(_TEAM_IDS)
    records = pd.DataFrame(
        [{"team_id": 17, "wins": 2, "losses": 1, "ties": 0, "points_for": 300.0}]
    )
    locked = locked_by_slot(records, slots)
    assert locked[slots.slot(17)] == LockedRecord(wins=2, losses=1, ties=0, points_for=300.0)


# --- end to end ---------------------------------------------------------------------------


def _pool_and_rosters() -> tuple[dict[int, list[str]], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    rosters: dict[int, list[str]] = {}
    for slot in range(1, len(_TEAM_IDS) + 1):
        ids: list[str] = []
        for j, pos in enumerate(("QB", "RB", "RB", "WR", "WR", "TE")):
            gid = f"00-{slot:04d}{j:03d}"
            ids.append(gid)
            rows.append(
                {
                    "gsis_id": gid,
                    "position": pos,
                    "season_mean_fpts": 240.0 - 20.0 * slot,
                    "is_rookie": False,
                }
            )
        rosters[slot] = ids
    pool = pd.DataFrame(rows)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    return rosters, pool


def _run(
    played_weeks: int = 2, n_sims: int = 150
) -> tuple[SeasonOutcomes, pd.DataFrame, SlotMap, dict[int, LockedRecord], int, dict[int, str]]:
    schedule = _schedule(played_weeks=played_weeks)
    slots = SlotMap.from_team_ids(_TEAM_IDS)
    rosters, pool = _pool_and_rosters()
    week = first_unplayed_week(schedule, _CAL)
    locked = {
        slot: LockedRecord(wins=1, losses=1, points_for=200.0 + slot)
        for slot in range(1, len(_TEAM_IDS) + 1)
    }
    config = LeagueConfig(
        name="standings_test",
        n_teams=len(_TEAM_IDS),
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
        },
        ruleset=Ruleset.espn_half(),
    )
    outcomes = simulate_seasons(
        rosters,
        pool,
        PlayerAvailability(p={g: 1.0 for g in pool["gsis_id"].astype(str)}, bye={}),
        VarianceParams.load(),
        league_config=config,
        n_sims=n_sims,
        rng=np.random.default_rng(5),
        calendar=_CAL,
        schedule=schedule_to_slots(schedule, slots, _CAL),
        locked=locked,
        first_unplayed_week=week,
    )
    names = {t: f"T{t}" for t in _TEAM_IDS}
    return outcomes, schedule, slots, locked, week, names


def test_standings_validate_and_cover_every_team() -> None:
    outcomes, _, slots, locked, week, names = _run()
    frame = build_standings(outcomes, slots, names, locked, season=2026, snapshot_week=week)
    assert len(frame) == len(_TEAM_IDS)
    assert set(frame["team_id"]) == set(_TEAM_IDS)
    assert frame["make_playoffs_pct"].between(0, 1).all()


def test_playoff_odds_sum_to_the_number_of_playoff_spots() -> None:
    """An identity, not an approximation: exactly `playoff_size` teams make it in every
    simulation, so the per-team probabilities must total that."""
    outcomes, _, slots, locked, week, names = _run()
    frame = build_standings(outcomes, slots, names, locked, season=2026, snapshot_week=week)
    assert frame["make_playoffs_pct"].sum() == pytest.approx(_CAL.playoff_size)
    assert frame["champ_pct"].sum() == pytest.approx(1.0)


def test_standings_attribute_results_to_the_right_teams() -> None:
    """The translation failure that would otherwise look like a plausible table: slot 1 has
    the strongest roster and maps to the lowest team id (1), so team 1 must lead."""
    outcomes, _, slots, locked, week, names = _run()
    frame = build_standings(outcomes, slots, names, locked, season=2026, snapshot_week=week)
    best = frame.iloc[0]
    assert best["team_id"] == min(_TEAM_IDS)
    assert best["team_name"] == f"T{min(_TEAM_IDS)}"


def test_banked_record_appears_alongside_the_projection() -> None:
    """A reader that cannot see both cannot tell a 1-1 start from a 1-1 projection."""
    outcomes, _, slots, locked, week, names = _run()
    frame = build_standings(outcomes, slots, names, locked, season=2026, snapshot_week=week)
    assert (frame["wins"] == 1).all()
    assert (frame["games_played"] == 2).all()
    # Two weeks banked, two simulated: nobody can finish below their banked wins.
    assert (frame["projected_wins"] >= frame["wins"]).all()


# --- matchup odds -------------------------------------------------------------------------


def test_matchup_odds_cover_only_remaining_games() -> None:
    outcomes, schedule, slots, _, week, _ = _run()
    odds = build_matchup_odds(outcomes, schedule, slots, season=2026, snapshot_week=week)
    assert (odds["week"] >= week).all()
    assert len(odds) == len(schedule[~schedule["is_played"]])


def test_head_to_head_is_symmetric() -> None:
    """P(A beats B) and P(B beats A) must sum to 1 over the same simulations. They cannot,
    if the two probabilities came from different runs."""
    outcomes, _, _, _, _, _ = _run()
    a, b, week = 1, 2, 3
    forward = outcomes.head_to_head(a, b, week)
    reverse = outcomes.head_to_head(b, a, week)
    assert forward + reverse == pytest.approx(1.0)


def test_the_stronger_roster_is_favoured() -> None:
    """Slot 1 outprojects slot 6 by a wide margin, so it must be better than a coin flip."""
    outcomes, _, _, _, _, _ = _run()
    assert outcomes.head_to_head(1, len(_TEAM_IDS), 3) > 0.5


def test_matchup_odds_come_from_the_same_run_as_the_standings() -> None:
    """Not a second engine: the table's probability must equal the accessor's, exactly."""
    outcomes, schedule, slots, _, week, _ = _run()
    odds = build_matchup_odds(outcomes, schedule, slots, season=2026, snapshot_week=week)
    row = odds.iloc[0]
    expected = outcomes.head_to_head(
        slots.slot(int(row["home_team_id"])), slots.slot(int(row["away_team_id"])), int(row["week"])
    )
    assert row["home_win_pct"] == pytest.approx(expected)


# --- snapshot persistence -----------------------------------------------------------------


def test_snapshot_round_trips_through_the_store(tmp_path: Path) -> None:
    """The trend line is a read of accumulated partitions, so a snapshot has to come back out
    validating against the same schema it went in under."""
    outcomes, schedule, slots, locked, week, names = _run()
    standings = build_standings(outcomes, slots, names, locked, season=2026, snapshot_week=week)
    odds = build_matchup_odds(outcomes, schedule, slots, season=2026, snapshot_week=week)

    for table, frame, schema in (
        ("projected_standings", standings, ProjectedStandingsSchema),
        ("matchup_odds", odds, MatchupOddsSchema),
    ):
        write_partition(tmp_path, table, frame, season=2026, week=week)
        back = read_partition(tmp_path, table, season=2026, week=week)
        assert len(back) == len(frame)
        schema.validate(back)


def test_two_weeks_of_snapshots_read_back_as_a_trend(tmp_path: Path) -> None:
    """The point of persisting: several weeks accumulate into one frame that can be plotted.
    A season-wide read must return every week, not just the newest."""
    outcomes, _, slots, locked, _, names = _run()
    for week in (3, 4):
        frame = build_standings(outcomes, slots, names, locked, season=2026, snapshot_week=week)
        write_partition(tmp_path, "projected_standings", frame, season=2026, week=week)

    trend = read_partition(tmp_path, "projected_standings", season=2026)
    assert sorted(trend["week"].unique()) == [3, 4]
    assert len(trend) == 2 * len(_TEAM_IDS)


# --- roster resolution: ESPN player ids -> gsis ---------------------------------------------


def _id_map(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Minimal (espn_id, gsis_id) crosswalk, the two columns rosters_to_slots joins on."""
    frame = pd.DataFrame(
        {
            "espn_id": pd.Series([e for e, _ in pairs], dtype=_PYARROW_STR),
            "gsis_id": pd.Series([g for _, g in pairs], dtype=_PYARROW_STR),
        }
    )
    return frame


def _espn_rosters(rows: list[tuple[int, str]]) -> pd.DataFrame:
    """The shape `parse_rosters` actually returns: ESPN `player_id`, and NO gsis_id."""
    frame = pd.DataFrame(
        {
            "team_id": [t for t, _ in rows],
            "player_id": [int(p) for _, p in rows],
            "player": pd.Series([f"P{p}" for _, p in rows], dtype=_PYARROW_STR),
            "pos": pd.Series(["RB"] * len(rows), dtype=_PYARROW_STR),
            "nfl_team": pd.Series([""] * len(rows), dtype=_PYARROW_STR),
            "lineup_slot": pd.Series([""] * len(rows), dtype=_PYARROW_STR),
            "acquisition_type": pd.Series([""] * len(rows), dtype=_PYARROW_STR),
        }
    )
    return frame


def test_rosters_are_resolved_through_the_id_map_not_a_gsis_column() -> None:
    """`parse_rosters` emits ESPN `player_id`; the pool is keyed on `gsis_id`. Nothing in
    `espn_league.py` produces a gsis at all, so reading a `gsis_id` column off an ESPN roster
    silently matches nothing and every team comes back empty.

    That failure is not loud. Empty rosters make `team_weekly_points` return all zeros, every
    matchup resolves `0 >= 0` so the HOME team wins every game in every simulation, and the
    output is a fully populated standings table whose playoff and title percentages are
    determined entirely by how many home fixtures each team happens to have.
    """
    rosters = _espn_rosters([(17, "4001"), (17, "4002"), (3, "4003")])
    id_map = _id_map([("4001", "00-0000001"), ("4002", "00-0000002"), ("4003", "00-0000003")])
    slots = SlotMap.from_team_ids([17, 3])
    pool_ids = {"00-0000001", "00-0000002", "00-0000003"}

    by_slot, dropped = rosters_to_slots(rosters, id_map, slots, pool_ids)

    assert by_slot[slots.slot(17)] == ["00-0000001", "00-0000002"]
    assert by_slot[slots.slot(3)] == ["00-0000003"]
    assert dropped == 0


def test_players_outside_the_pool_are_dropped_and_counted() -> None:
    """K and D/ST are rostered but unprojectable, so dropping them is right — but the count
    has to be reported, because 'everything was dropped' and 'two kickers were dropped' print
    the same way otherwise."""
    rosters = _espn_rosters([(17, "4001"), (17, "9999")])
    id_map = _id_map([("4001", "00-0000001"), ("9999", "00-0000009")])
    slots = SlotMap.from_team_ids([17])

    by_slot, dropped = rosters_to_slots(rosters, id_map, slots, {"00-0000001"})

    assert by_slot[slots.slot(17)] == ["00-0000001"]
    assert dropped == 1


def test_an_all_empty_resolution_raises_rather_than_simulating_zeros() -> None:
    """The guard the old `note:` line was not. If NOTHING resolves, the id_map join is broken,
    and continuing produces a confident table built on zero-point rosters."""
    rosters = _espn_rosters([(17, "4001"), (3, "4002")])
    slots = SlotMap.from_team_ids([17, 3])

    with pytest.raises(ValueError, match="no rostered player"):
        rosters_to_slots(rosters, _id_map([]), slots, {"00-0000001"})
