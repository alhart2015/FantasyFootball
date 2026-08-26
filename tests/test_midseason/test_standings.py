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
from projections.draft.league_config import LeagueConfig
from projections.ingest.espn_league import team_records
from projections.midseason.standings import (
    SlotMap,
    build_matchup_odds,
    build_standings,
    first_unplayed_week,
    locked_by_slot,
    rosters_to_slots,
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
from tests.test_midseason.conftest import CALENDAR, TEAM_IDS, schedule_frame

_CAL = CALENDAR

_TEAM_IDS = TEAM_IDS


def _schedule(played_weeks: int = 0) -> pd.DataFrame:
    """Alias over the shared builder, so the existing call sites read unchanged."""
    return schedule_frame(played_weeks)


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
    outcomes, _, slots, _locked, week, names = _run()
    frame = build_standings(outcomes, slots, names, season=2026, snapshot_week=week)
    assert len(frame) == len(_TEAM_IDS)
    assert set(frame["team_id"]) == set(_TEAM_IDS)
    assert frame["make_playoffs_pct"].between(0, 1).all()


def test_playoff_odds_sum_to_the_number_of_playoff_spots() -> None:
    """An identity, not an approximation: exactly `playoff_size` teams make it in every
    simulation, so the per-team probabilities must total that."""
    outcomes, _, slots, _locked, week, names = _run()
    frame = build_standings(outcomes, slots, names, season=2026, snapshot_week=week)
    assert frame["make_playoffs_pct"].sum() == pytest.approx(_CAL.playoff_size)
    assert frame["champ_pct"].sum() == pytest.approx(1.0)


def test_standings_attribute_results_to_the_right_teams() -> None:
    """The translation failure that would otherwise look like a plausible table: slot 1 has
    the strongest roster and maps to the lowest team id (1), so team 1 must lead."""
    outcomes, _, slots, _locked, week, names = _run()
    frame = build_standings(outcomes, slots, names, season=2026, snapshot_week=week)
    best = frame.iloc[0]
    assert best["team_id"] == min(_TEAM_IDS)
    assert best["team_name"] == f"T{min(_TEAM_IDS)}"


def test_banked_record_appears_alongside_the_projection() -> None:
    """A reader that cannot see both cannot tell a 1-1 start from a 1-1 projection."""
    outcomes, _, slots, _locked, week, names = _run()
    frame = build_standings(outcomes, slots, names, season=2026, snapshot_week=week)
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
    outcomes, schedule, slots, _locked, week, names = _run()
    standings = build_standings(outcomes, slots, names, season=2026, snapshot_week=week)
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
    outcomes, _, slots, _locked, _, names = _run()
    for week in (3, 4):
        frame = build_standings(outcomes, slots, names, season=2026, snapshot_week=week)
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

    with pytest.raises(ValueError, match="no projectable players"):
        rosters_to_slots(rosters, _id_map([]), slots, {"00-0000001"})


# --- schedule validation, snapshot keys, seeding horizon ------------------------------------


def test_a_week_where_a_team_plays_twice_is_rejected() -> None:
    """The old guard checked only that a week was non-empty, which is not the invariant. Every
    slot must play exactly once per week: a duplicated matchup row lets one team bank two
    games in a week and finish with more wins than the season has games, and nothing fires."""
    slots = SlotMap.from_team_ids(_TEAM_IDS)
    doubled = pd.concat(
        [_schedule(), _schedule().head(1)], ignore_index=True
    )  # week 1 now holds 17-vs-3 twice
    with pytest.raises(ValueError, match="at most once"):
        schedule_to_slots(doubled, slots, _CAL)


def test_a_week_missing_half_its_matchups_is_rejected() -> None:
    """The complement, and the failure the old message actually named: a week ESPN returned
    only partially passes a non-empty check while half the league silently sits out."""
    slots = SlotMap.from_team_ids(_TEAM_IDS)
    partial = _schedule()
    # Drop one of week 2's three matchups: the week is still non-empty.
    drop = partial[(partial["week"] == 2)].index[0]
    with pytest.raises(ValueError, match="at most once"):
        schedule_to_slots(partial.drop(index=drop), slots, _CAL)


def test_matchup_odds_carry_the_snapshot_week_they_were_produced_in() -> None:
    """`week` on this table is the MATCHUP week, but the partition key is the SNAPSHOT week.
    Without a column recording which snapshot produced a row, two snapshots both containing
    the same future fixture concatenate into a frame where the same (season, week, teams) key
    appears twice with different probabilities and nothing says which is current."""
    outcomes, schedule, slots, _, week, _ = _run()
    odds = build_matchup_odds(outcomes, schedule, slots, season=2026, snapshot_week=week)
    assert (odds["snapshot_week"] == week).all()
    # And the matchup week is still its own column, distinct from the snapshot.
    assert (odds["week"] >= week).all()
    assert odds["week"].nunique() > 1


def test_two_matchup_odds_snapshots_stay_distinguishable(tmp_path: Path) -> None:
    outcomes, schedule, slots, _, _, _ = _run()
    for snap in (3, 4):
        frame = build_matchup_odds(outcomes, schedule, slots, season=2026, snapshot_week=snap)
        write_partition(tmp_path, "matchup_odds", frame, season=2026, week=snap)
    back = read_partition(tmp_path, "matchup_odds", season=2026)
    assert sorted(back["snapshot_week"].unique()) == [3, 4]
    # The same fixture appears once per snapshot, and the snapshot column separates them.
    key = ["season", "snapshot_week", "week", "home_team_id", "away_team_id"]
    assert not back.duplicated(subset=key).any()


def test_standings_are_ordered_by_projected_points_not_banked_points() -> None:
    """The simulator seeds on total (locked + simulated) points-for, so ordering the display
    on banked points alone makes the row order disagree with the `mean_seed` printed beside
    it -- and preseason, where every banked figure is 0.0, the order is arbitrary."""
    outcomes, _, slots, _locked, week, names = _run()
    frame = build_standings(outcomes, slots, names, season=2026, snapshot_week=week)
    assert "projected_points_for" in frame.columns
    assert (frame["projected_points_for"] >= frame["points_for"]).all()
    ordered = frame.sort_values(
        ["projected_wins", "projected_points_for"], ascending=False, ignore_index=True
    )
    assert list(frame["team_id"]) == list(ordered["team_id"])


def test_standings_read_the_banked_record_off_the_simulation_that_used_it() -> None:
    """`build_standings` used to take `locked` as its own argument, and the CLI passed
    `locked_by_slot(...)` twice -- once into the simulator and once here. Nothing checked the
    two agreed, so a mismatch would print a banked record that disagreed with the projection
    sitting next to it, silently. The outcomes object now carries the records it was built
    with, so there is only one source."""
    outcomes, _, slots, locked, week, names = _run()
    assert outcomes.locked == locked
    frame = build_standings(outcomes, slots, names, season=2026, snapshot_week=week)
    for row in frame.itertuples():
        record = locked[slots.slot(int(row.team_id))]
        assert row.wins == record.wins
        assert row.points_for == pytest.approx(record.points_for)


def test_the_end_to_end_fixture_derives_its_record_from_its_own_schedule() -> None:
    """The fixture used to hand-write locked records that contradicted the schedule beside
    them (schedule said 2-0 at 240 points, fixture asserted 1-1 at ~200), so
    `team_records -> locked_by_slot` -- the real integration seam -- was never exercised
    together. Deriving one from the other covers it and removes the invented numbers."""
    schedule = _schedule(played_weeks=2)
    slots = SlotMap.from_team_ids(_TEAM_IDS)
    derived = locked_by_slot(team_records(schedule, through_week=2), slots)
    # Every home team won both weeks at 120-90, so home sides are 2-0 and away sides 0-2.
    home_slot = slots.slot(17)  # home in weeks 1 and 3 of the fixture pairing
    assert derived[home_slot].games_played == 2
    assert derived[home_slot].wins + derived[home_slot].losses == 2
    assert sum(r.wins for r in derived.values()) == sum(r.losses for r in derived.values())


def test_an_odd_sized_league_gives_one_team_a_bye_each_week() -> None:
    """`parse_schedule` DELIBERATELY drops ESPN's bye rows (a matchup with no away side), so in
    an odd-sized league every week is one slot short. The every-slot-plays-once guard was
    written against even leagues and rejected an 11-team league on week 1 -- a perfectly legal
    setup, and the old non-empty check had tolerated it."""
    odd_ids = [17, 3, 11, 5, 9]  # five teams: one sits out each week
    slots = SlotMap.from_team_ids(odd_ids)
    pairings = [
        [(17, 3), (11, 5)],  # 9 has the bye
        [(17, 11), (5, 9)],  # 3 has the bye
        [(17, 5), (3, 9)],  # 11 has the bye
        [(17, 9), (11, 3)],  # 5 has the bye
    ]
    rows: list[dict[str, object]] = []
    for week, games in enumerate(pairings, start=1):
        for home, away in games:
            rows.append(
                {
                    "week": week,
                    "home_team_id": home,
                    "away_team_id": away,
                    "home_team": f"T{home}",
                    "away_team": f"T{away}",
                    "home_points": 0.0,
                    "away_points": 0.0,
                    "winner": "UNDECIDED",
                    "is_played": False,
                }
            )
    frame = pd.DataFrame(rows)
    for column in ("home_team", "away_team", "winner"):
        frame[column] = frame[column].astype(_PYARROW_STR)

    fixtures = schedule_to_slots(frame, slots, _CAL)
    assert len(fixtures) == _CAL.reg_weeks
    for week_games in fixtures:
        seats = [seat for pair in week_games for seat in pair]
        assert len(seats) == len(set(seats)), "nobody plays twice"
        assert len(seats) == len(odd_ids) - 1, "exactly one team sits out"


def test_a_duplicated_espn_id_does_not_roster_a_player_twice() -> None:
    """`IdMapSchema` marks only `gsis_id` unique; `espn_id` is nullable and non-unique, and the
    live `data/raw/id_map.parquet` genuinely holds two ESPN ids that each map to two different
    players. An undeduplicated inner join fans out on those, puts a player on a roster twice --
    where he can fill two starting slots at once and inflate that team's weekly points -- and
    makes the dropped count negative, so the CLI prints 'note: -1 rostered players'."""
    rosters = _espn_rosters([(17, "4001"), (17, "4002")])
    id_map = pd.DataFrame(
        {
            "espn_id": pd.Series(["4001", "4001", "4002"], dtype=_PYARROW_STR),
            "gsis_id": pd.Series(["00-0000001", "00-0000099", "00-0000002"], dtype=_PYARROW_STR),
        }
    )
    slots = SlotMap.from_team_ids([17])
    pool_ids = {"00-0000001", "00-0000002", "00-0000099"}

    by_slot, dropped = rosters_to_slots(rosters, id_map, slots, pool_ids)

    assert len(by_slot[slots.slot(17)]) == 2, "two roster rows must yield two players"
    assert len(set(by_slot[slots.slot(17)])) == 2
    assert dropped >= 0


def test_one_team_resolving_to_nothing_raises_rather_than_being_simulated_at_zero() -> None:
    """The guard used to fire only when EVERY team failed. One team whose players are all
    missing from the id_map -- a new manager's roster of just-signed players -- was simulated
    at zero points, losing every week, with champ_pct 0.0 and no message anywhere. That is the
    same failure the guard exists to prevent, just retail instead of wholesale."""
    rosters = _espn_rosters([(17, "4001"), (3, "9998"), (3, "9999")])
    id_map = _id_map([("4001", "00-0000001")])  # team 3 resolves to nothing
    slots = SlotMap.from_team_ids([17, 3])

    with pytest.raises(ValueError, match="no projectable players"):
        rosters_to_slots(rosters, id_map, slots, {"00-0000001"})
