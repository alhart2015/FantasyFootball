"""Is anyone on waivers better than someone on my team -- starter or bench?

Two horizons, two lists. The season list compares rest-of-season points against the weakest
player I roster and names him as the drop. The weekly list compares this week's projection
against the weakest player I have who is playing, and names no drop.

The tests that matter most are still about what the tool refuses: a player it cannot price is
not worth zero, a bye is not a zero-point week, and an IR player is neither a starter nor a drop.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from projections.draft.league_config import LeagueConfig
from projections.ingest.espn_league import parse_free_agents, parse_rosters
from projections.midseason.waivers import (
    Candidate,
    remaining_points_by_espn_id,
    season_upgrades,
    weekly_projections_by_espn_id,
    weekly_upgrades,
)
from projections.schemas import (
    _PYARROW_STR,
    InjuryStatus,
    RosterSlot,
    Ruleset,
)
from tests.test_midseason.conftest import MY_TEAM_ID, espn_payload

#: Critts: seven starters, five bench, two IR.
LEAGUE = LeagueConfig(
    name="test",
    n_teams=16,
    roster_slots={
        RosterSlot.QB: 1,
        RosterSlot.RB: 2,
        RosterSlot.WR: 2,
        RosterSlot.TE: 1,
        RosterSlot.FLEX: 1,
        RosterSlot.BENCH: 5,
        RosterSlot.IR: 2,
    },
    ruleset=Ruleset.espn_half(),
)


def _players(rows: list[tuple[Any, ...]]) -> pd.DataFrame:
    """`(player_id, name, position, injury_status[, lineup_slot])` -> a roster or FA frame."""
    return pd.DataFrame(
        {
            "player_id": [r[0] for r in rows],
            "player": pd.Series([r[1] for r in rows], dtype=_PYARROW_STR),
            "pos": pd.Series([r[2] for r in rows], dtype=_PYARROW_STR),
            "nfl_team": pd.Series(["KC"] * len(rows), dtype=_PYARROW_STR),
            "injury_status": pd.Series([r[3] for r in rows], dtype=_PYARROW_STR),
            # `parse_rosters` always produces this, and `is_on_ir` reads it.
            "lineup_slot": pd.Series(
                [r[4] if len(r) > 4 else "" for r in rows], dtype=_PYARROW_STR
            ),
            "percent_owned": [50.0] * len(rows),
            "on_waivers": [False] * len(rows),
        }
    )


def _full_roster() -> pd.DataFrame:
    """Twelve players, filling every non-IR spot. Nobody can be added for free."""
    rows = [
        (1, "QB1", "QB", "ACTIVE"),
        (2, "RB1", "RB", "ACTIVE"),
        (3, "RB2", "RB", "ACTIVE"),
        (4, "WR1", "WR", "ACTIVE"),
        (5, "WR2", "WR", "ACTIVE"),
        (6, "TE1", "TE", "ACTIVE"),
        (7, "FlexRB", "RB", "ACTIVE"),
    ]
    rows += [(10 + i, f"Bench{i}", "WR", "ACTIVE") for i in range(5)]
    return _players(rows)


#: Starters project well, bench players badly.
BASE_PROJECTIONS: dict[str, float] = {
    "1": 20.0,
    "2": 18.0,
    "3": 14.0,
    "4": 16.0,
    "5": 12.0,
    "6": 9.0,
    "7": 11.0,
    **{str(10 + i): 3.0 for i in range(5)},
}

#: Rest-of-season value. The bench is cheap (Bench0 cheapest at 20), starters are not.
BASE_REMAINING: dict[str, float] = {
    "1": 200.0,
    "2": 180.0,
    "3": 140.0,
    "4": 160.0,
    "5": 120.0,
    "6": 90.0,
    "7": 110.0,
    **{str(10 + i): 20.0 + i for i in range(5)},
}


def _season(
    free_agents: pd.DataFrame,
    *,
    projections: dict[str, float] | None = None,
    remaining: dict[str, float] | None = None,
    roster: pd.DataFrame | None = None,
    min_margin: float = 5.0,
) -> list[Candidate]:
    candidates, _ = season_upgrades(
        _full_roster() if roster is None else roster,
        free_agents,
        {**BASE_PROJECTIONS, **(projections or {})},
        {**BASE_REMAINING, **(remaining or {})},
        LEAGUE,
        min_margin=min_margin,
    )
    return candidates


def _weekly(
    free_agents: pd.DataFrame,
    *,
    projections: dict[str, float] | None = None,
    base: dict[str, float] | None = None,
    remaining: dict[str, float] | None = None,
    roster: pd.DataFrame | None = None,
    source_is_injury_aware: bool = True,
    min_margin: float = 0.5,
) -> list[Candidate]:
    return weekly_upgrades(
        _full_roster() if roster is None else roster,
        free_agents,
        {**(BASE_PROJECTIONS if base is None else base), **(projections or {})},
        BASE_REMAINING if remaining is None else remaining,
        LEAGUE,
        source_is_injury_aware=source_is_injury_aware,
        min_margin=min_margin,
    )


def _open_spots(roster: pd.DataFrame) -> int:
    _, spots = season_upgrades(roster, _players([]), BASE_PROJECTIONS, BASE_REMAINING, LEAGUE)
    return spots


# --- the season list ----------------------------------------------------------------------------


def test_a_free_agent_better_than_my_worst_bench_player_is_listed_though_he_would_not_start() -> (
    None
):
    """The whole reason for this change. He beats Bench0 for the rest of the year but would
    not start this week, and the old lineup-only filter scored him as exactly nothing."""
    agents = _players([(99, "Decent WR", "WR", "ACTIVE")])
    [candidate] = _season(agents, projections={"99": 10.0}, remaining={"99": 60.0})
    assert candidate.beats_player == "Bench0"
    assert candidate.margin == pytest.approx(40.0)
    assert candidate.season_points == pytest.approx(60.0)
    assert candidate.lineup_gain == pytest.approx(0.0), "a bench upgrade does not move the lineup"
    assert candidate.drop_player == "Bench0"
    assert candidate.drop_cost == pytest.approx(20.0)
    assert not candidate.is_free


def test_a_free_agent_worse_than_everyone_i_roster_is_not_listed() -> None:
    agents = _players([(99, "Scrub WR", "WR", "ACTIVE")])
    assert _season(agents, projections={"99": 10.0}, remaining={"99": 15.0}) == []


def test_a_margin_under_the_floor_is_noise() -> None:
    """Two season points is a few hundredths of a hundredth of a win. The floor is a flag."""
    agents = _players([(99, "Marginal WR", "WR", "ACTIVE")])
    assert _season(agents, remaining={"99": 22.0}) == []
    assert [c.player for c in _season(agents, remaining={"99": 22.0}, min_margin=1.0)] == [
        "Marginal WR"
    ]


def test_the_weakest_player_is_the_drop_even_when_he_starts() -> None:
    """Starter or bench. If my starting WR2 is the worst receiver I roster for the rest of the
    year, he is the one a better free agent should replace -- and the row says what that costs
    this week: the add (10.0) starts in his place (12.0)."""
    agents = _players([(99, "Decent WR", "WR", "ACTIVE")])
    [candidate] = _season(agents, projections={"99": 10.0}, remaining={"5": 5.0, "99": 60.0})
    assert candidate.drop_player == "WR2"
    assert candidate.lineup_gain == pytest.approx(-2.0)


def test_raw_points_do_not_compare_across_positions() -> None:
    """Found on a real league: backup quarterbacks filled the list because a 190-point QB
    "beat" a 55-point bench back. He is compared with my QB, whom he does not beat."""
    agents = _players([(99, "Backup QB", "QB", "ACTIVE")])
    assert _season(agents, projections={"99": 15.0}, remaining={"99": 150.0}) == []
    assert _weekly(agents, projections={"99": 15.0}) == []


def test_a_free_agent_the_pool_cannot_price_is_not_on_the_season_list() -> None:
    agents = _players([(99, "Unknown WR", "WR", "ACTIVE")])
    assert _season(agents, projections={"99": 30.0}) == []


def test_a_rostered_player_the_pool_cannot_price_is_never_the_drop() -> None:
    """A kicker, a defense, a back nobody projects yet -- absent from `remaining_points`.

    Defaulting him to 0.0 made him the cheapest player by construction, so the tool recommended
    dropping him "at no cost". "We have no number for him" is not "he is worth nothing".
    """
    remaining = {k: v for k, v in BASE_REMAINING.items() if k != "10"}  # Bench0 unpriced
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    candidates, _ = season_upgrades(
        _full_roster(), agents, BASE_PROJECTIONS, {**remaining, "99": 100.0}, LEAGUE
    )
    assert candidates[0].drop_player == "Bench1"


def test_nobody_priced_at_his_position_means_no_season_row() -> None:
    """Nobody to beat and no like-for-like drop. Listing him "free" on a full roster would be
    the lie `is_free` once told; the weekly list is where a positional hole shows up."""
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    remaining = {k: v for k, v in BASE_REMAINING.items() if int(k) < 4 or k in {"6", "7"}}
    candidates, _ = season_upgrades(
        _full_roster(), agents, BASE_PROJECTIONS, {**remaining, "99": 100.0}, LEAGUE
    )
    assert candidates == []


def test_an_open_roster_spot_means_nobody_is_dropped() -> None:
    roster = _full_roster().iloc[:-1]  # one bench spot free
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    [candidate] = _season(agents, remaining={"99": 100.0}, roster=roster)
    assert candidate.is_free
    assert candidate.drop_player == ""
    assert candidate.drop_cost == 0.0


def test_ir_slots_do_not_count_as_open_roster_spots() -> None:
    """Counting them made a full 12-man roster look like it had two spaces going spare, so
    every recommendation came back free and no drop was ever named."""
    assert _open_spots(_full_roster()) == 0


def test_the_caller_is_told_how_many_spots_are_actually_open() -> None:
    """Every `needs_no_drop` candidate is claiming the SAME spot."""
    assert _open_spots(_full_roster().iloc[:-2]) == 2


def test_season_candidates_come_back_best_margin_first() -> None:
    agents = _players([(98, "Good WR", "WR", "ACTIVE"), (99, "Better WR", "WR", "ACTIVE")])
    ranked = _season(agents, remaining={"98": 50.0, "99": 90.0})
    assert [c.player for c in ranked] == ["Better WR", "Good WR"]


def test_a_position_the_league_cannot_start_is_never_listed() -> None:
    """A kicker in a kicker-less league beats nobody, however many points he scores."""
    agents = _players([(99, "Big Leg", "K", "ACTIVE")])
    assert _season(agents, projections={"99": 12.0}, remaining={"99": 150.0}) == []
    assert _weekly(agents, projections={"99": 12.0}) == []


def test_an_empty_wire_is_an_empty_list_not_an_error() -> None:
    assert _season(_players([])) == []
    assert _weekly(_players([])) == []


def test_a_float_player_id_column_does_not_crash() -> None:
    """Any frame that has been through a merge introducing an NA becomes float64."""
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    agents["player_id"] = agents["player_id"].astype("float64")
    [candidate] = _season(agents, remaining={"99": 100.0})
    assert candidate.player_id == 99


# --- the weekly list ----------------------------------------------------------------------------


def test_a_free_agent_who_outprojects_my_bench_this_week_is_listed() -> None:
    """He would not start, and still belongs on the list: the reader wants to see who is
    projected to do well this week. No drop is named -- that is the season list's call."""
    agents = _players([(99, "Hot WR", "WR", "ACTIVE")])
    [candidate] = _weekly(agents, projections={"99": 10.0})
    assert candidate.beats_player == "Bench0"
    assert candidate.margin == pytest.approx(7.0)
    assert candidate.lineup_gain == pytest.approx(0.0)
    assert candidate.drop_player_id is None


def test_the_weekly_list_ignores_rest_of_season_value() -> None:
    """A one-week wonder the pool does not even project for the season still shows up."""
    agents = _players([(99, "One Week Wonder", "WR", "ACTIVE")])
    [candidate] = _weekly(agents, projections={"99": 10.0})
    assert candidate.season_points is None


def test_a_free_agent_below_everyone_this_week_is_not_listed() -> None:
    agents = _players([(99, "Scrub WR", "WR", "ACTIVE")])
    assert _weekly(agents, projections={"99": 2.0}) == []


def test_a_player_with_no_projection_is_never_on_the_weekly_list() -> None:
    """A bye, or nobody projected him. Unstartable, not zero."""
    agents = _players([(99, "On Bye", "WR", "ACTIVE")])
    assert _weekly(agents) == []


def test_my_players_on_bye_are_not_the_weakest_at_zero() -> None:
    """A bye has no projection. Scoring it 0.0 would make every free agent on the wire "better"
    than him, and the list would be the whole wire."""
    base = {k: v for k, v in BASE_PROJECTIONS.items() if not k.startswith("1") or k == "1"}
    agents = _players([(99, "Meh WR", "WR", "ACTIVE")])
    # The bench receivers are all on bye; the weakest receiver PLAYING is WR2 at 12.0.
    assert _weekly(agents, base=base, projections={"99": 5.0}) == []
    [candidate] = _weekly(agents, base=base, projections={"99": 13.0})
    assert candidate.beats_player == "WR2"


def test_a_bye_week_hole_is_compared_against_an_empty_spot() -> None:
    """My only TE is on bye, so a streamer at 2.0 out-projects the nobody I have playing
    there, and fills the slot for the whole of his projection."""
    base = {k: v for k, v in BASE_PROJECTIONS.items() if k != "6"}
    agents = _players([(99, "Streamer TE", "TE", "ACTIVE")])
    [candidate] = _weekly(agents, base=base, projections={"99": 2.0})
    assert candidate.beats_player == ""
    assert candidate.margin == pytest.approx(2.0)
    assert candidate.lineup_gain == pytest.approx(2.0)


def test_displacing_a_starter_cascades_down_the_lineup() -> None:
    """A 12.2 receiver beats WR2 (12.0) by two tenths, but WR2 moves into the flex and pushes
    FlexRB (11.0) out, so the LINEUP gains 1.2 -- the number a reader checking by hand will not
    expect, and the one the row prints."""
    agents = _players([(99, "Slight WR", "WR", "ACTIVE")])
    [candidate] = _weekly(agents, projections={"99": 12.2})
    assert candidate.lineup_gain == pytest.approx(1.2)


def test_a_questionable_free_agent_is_discounted() -> None:
    """0.86 on a single week. At 4.0 healthy he clears my bench (3.0) by a point; tagged
    Questionable he projects 3.44 and clears it by less than the floor."""
    healthy = _players([(99, "Healthy WR", "WR", "ACTIVE")])
    tagged = _players([(99, "Questionable WR", "WR", "QUESTIONABLE")])
    assert _weekly(healthy, projections={"99": 4.0})[0].margin == pytest.approx(1.0)
    assert _weekly(tagged, projections={"99": 4.0}) == []


def test_espn_priced_statuses_are_not_discounted_twice() -> None:
    """ESPN's weekly feed already zeroes players it lists as Out. With the default, an Out
    player carrying a projection is left alone; told the source is naive, he is zeroed."""
    agents = _players([(99, "Out WR", "WR", "OUT")])
    assert _weekly(agents, projections={"99": 20.0}, source_is_injury_aware=True)
    assert _weekly(agents, projections={"99": 20.0}, source_is_injury_aware=False) == []


def test_weekly_candidates_come_back_best_first() -> None:
    agents = _players([(98, "Good WR", "WR", "ACTIVE"), (99, "Better WR", "WR", "ACTIVE")])
    ranked = _weekly(agents, projections={"98": 8.0, "99": 15.0})
    assert [c.player for c in ranked] == ["Better WR", "Good WR"]


# --- end to end, through the real parsers ------------------------------------------------------


def _fa_payload(
    player_id: int, name: str, position_id: int, *, status: str = "ACTIVE"
) -> dict[str, Any]:
    """`kona_player_info` shape, as `parse_free_agents` reads it."""
    return {
        "players": [
            {
                "id": player_id,
                "status": "FREEAGENT",
                "player": {
                    "id": player_id,
                    "fullName": name,
                    "defaultPositionId": position_id,
                    "proTeamId": 1,
                    "injuryStatus": status,
                    "ownership": {"percentOwned": 12.0},
                },
            }
        ]
    }


def test_the_pipeline_runs_on_parser_output_not_hand_built_frames() -> None:
    """Every other test here builds its frames by hand, which means none of them would notice
    `parse_rosters` renaming a column or `parse_free_agents` producing a different id dtype."""
    payload = espn_payload(played_weeks=0)
    roster = parse_rosters(payload)
    roster = roster[roster["team_id"] == MY_TEAM_ID]
    assert not roster.empty, "the fixture league has rosters"

    free_agents, warning = parse_free_agents(_fa_payload(900_002, "Wire Stud", 3), limit=50)
    assert warning is None

    projections = {str(pid): 8.0 for pid in roster["player_id"]}
    projections["900002"] = 30.0
    remaining = {str(pid): 50.0 for pid in roster["player_id"]}
    remaining["900002"] = 120.0

    season, _ = season_upgrades(roster, free_agents, projections, remaining, LEAGUE)
    weekly = weekly_upgrades(roster, free_agents, projections, remaining, LEAGUE)
    for candidates in (season, weekly):
        assert [c.player for c in candidates] == ["Wire Stud"]
        assert candidates[0].position == "WR"
        assert candidates[0].percent_owned == pytest.approx(12.0)


def test_an_injured_free_agent_survives_the_parsers_with_his_status() -> None:
    """His status has to reach the recommender from the parser rather than from a fixture that
    happens to spell it right."""
    payload = espn_payload(played_weeks=0)
    roster = parse_rosters(payload)
    roster = roster[roster["team_id"] == MY_TEAM_ID]
    free_agents, _ = parse_free_agents(
        _fa_payload(900_003, "Hurt Stud", 3, status="QUESTIONABLE"), limit=50
    )
    projections = {str(pid): 8.0 for pid in roster["player_id"]}
    projections["900003"] = 30.0

    [candidate] = weekly_upgrades(roster, free_agents, projections, {}, LEAGUE)
    assert candidate.injury_status is InjuryStatus.QUESTIONABLE
    assert candidate.projected == pytest.approx(30.0 * 0.86)


# --- one roster model: IR is not an active spot, not a starter, and not a drop ----------------


def _roster_with_ir() -> pd.DataFrame:
    """Eleven active players plus one parked on IR. Twelve rows, eleven active spots used."""
    rows: list[tuple[Any, ...]] = [
        (1, "QB1", "QB", "ACTIVE", "QB"),
        (2, "RB1", "RB", "ACTIVE", "RB"),
        (3, "RB2", "RB", "ACTIVE", "RB"),
        (4, "WR1", "WR", "ACTIVE", "WR"),
        (5, "WR2", "WR", "ACTIVE", "WR"),
        (6, "TE1", "TE", "ACTIVE", "TE"),
        (7, "FlexRB", "RB", "ACTIVE", "FLEX"),
    ]
    rows += [(10 + i, f"Bench{i}", "WR", "ACTIVE", "BENCH") for i in range(4)]
    rows += [(20, "Hurt WR", "WR", "INJURY_RESERVE", "IR")]
    return _players(rows)


def test_a_player_on_ir_does_not_occupy_an_active_roster_spot() -> None:
    """Counting IR PLAYERS against active capacity made a roster with someone on IR look full,
    so the tool named a drop nobody had to make."""
    assert _open_spots(_roster_with_ir()) == 1
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    [candidate] = _season(agents, remaining={"99": 100.0}, roster=_roster_with_ir())
    assert candidate.is_free, "eleven active players in twelve spots: nobody has to go"


def test_a_player_on_ir_cannot_hold_a_starting_slot() -> None:
    """The morning-after case. My WR1 is on IR but ESPN still projects him; if the lineup
    counts him, a replacement looks like he changes nothing."""
    roster = _roster_with_ir()
    roster.loc[roster["player"] == "WR1", "lineup_slot"] = "IR"
    roster.loc[roster["player"] == "WR1", "injury_status"] = "INJURY_RESERVE"

    agents = _players([(99, "Replacement WR", "WR", "ACTIVE")])
    [candidate] = _weekly(agents, projections={"99": 13.0}, roster=roster)
    assert candidate.lineup_gain > 0


def test_a_player_on_ir_is_never_the_weakest_on_either_list() -> None:
    """Dropping an IR player frees an IR slot, not the ACTIVE spot the add needs, and his
    injury-discounted numbers make him the likeliest to be named. He is neither the drop nor
    the player a free agent "beats"."""
    roster = pd.concat(
        [_roster_with_ir(), _players([(30, "Bench4", "WR", "ACTIVE", "BENCH")])],
        ignore_index=True,
    )
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    [season] = _season(agents, remaining={"20": 1.0, "30": 40.0, "99": 100.0}, roster=roster)
    assert not season.is_free, "the active roster is full"
    # The POSITIVE assertion: `!= "Hurt WR"` also passes when no drop is found at all.
    assert season.drop_player == "Bench0"
    [weekly] = _weekly(agents, projections={"20": 0.5, "30": 3.0, "99": 10.0}, roster=roster)
    assert weekly.beats_player != "Hurt WR"


# --- the two inputs, now in src and therefore testable -----------------------------------------


def _kona(rows: list[tuple[int, int, dict[str, float] | None]]) -> dict[str, Any]:
    """`(espn_id, defaultPositionId, week-1 raw stats or None)` -> a kona_player_info payload."""
    players = []
    for espn_id, position_id, stats in rows:
        player: dict[str, Any] = {
            "id": espn_id,
            "fullName": f"Player {espn_id}",
            "defaultPositionId": position_id,
            "proTeamId": 1,
        }
        if stats is not None:
            player["stats"] = [
                {
                    "scoringPeriodId": 1,
                    "statSourceId": 1,
                    "statSplitTypeId": 1,
                    "stats": stats,
                }
            ]
        players.append({"id": espn_id, "status": "FREEAGENT", "player": player})
    return {"players": players}


def test_weekly_projections_are_scored_under_the_league_ruleset() -> None:
    """Not read off ESPN's `appliedTotal`. A free agent and a rostered player have to be valued
    the same way, and the same way the rest of this repo values anybody."""
    # statId 42 is receiving yards, 53 is receptions. Half-PPR: 100 yards + 4 catches = 12.0.
    payload = _kona([(1, 3, {"42": 100.0, "53": 4.0})])
    projections = weekly_projections_by_espn_id(payload, 1, Ruleset.espn_half())
    assert projections["1"] == pytest.approx(12.0)


def test_a_player_with_no_projection_is_absent_rather_than_zero() -> None:
    """That absence is what makes him unstartable downstream -- which is how bye weeks work
    without anything in this repo having a rule about bye weeks."""
    payload = _kona([(1, 3, {"42": 100.0}), (2, 3, None)])
    projections = weekly_projections_by_espn_id(payload, 1, Ruleset.espn_half())
    assert "1" in projections
    assert "2" not in projections


def test_kickers_and_defenses_are_priced_even_though_they_cannot_be_ranked() -> None:
    """This feed prices MY roster as well as the wire. A kicker the tool cannot price is a
    kicker it treats as unstartable, leaving a hole in the baseline lineup and inflating every
    candidate's gain."""
    payload = _kona([(1, 5, {"42": 0.0}), (2, 16, {"42": 0.0})])  # K, DST
    projections = weekly_projections_by_espn_id(payload, 1, Ruleset.espn_half())
    assert set(projections) == {"1", "2"}


def _run_state(week: int = 5, *, ir_player: str | None = None) -> Any:
    """A minimal `MyTeamRun`-shaped object: `remaining_points_by_espn_id` reads two frames."""
    roster = pd.DataFrame(
        {
            "gsis_id": pd.Series(["00-0000001", "00-0000002"], dtype=_PYARROW_STR),
            "player": pd.Series(["Fit RB", "Hurt RB"], dtype=_PYARROW_STR),
            "injury_status": pd.Series(["ACTIVE", ir_player or "ACTIVE"], dtype=_PYARROW_STR),
        }
    )
    ros = pd.DataFrame(
        {
            "gsis_id": pd.Series(["00-0000001", "00-0000002"], dtype=_PYARROW_STR),
            "season_mean_fpts": [100.0, 100.0],
        }
    )
    return SimpleNamespace(roster=roster, ros=ros, week=week)


def _small_id_map() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "espn_id": pd.Series(["1", "2"], dtype=_PYARROW_STR),
            "gsis_id": pd.Series(["00-0000001", "00-0000002"], dtype=_PYARROW_STR),
        }
    )


def test_a_drop_cost_is_discounted_by_the_injury_that_makes_him_droppable() -> None:
    """The whole point of the column: you are looking for a drop BECAUSE somebody is hurt, and
    a player on IR is worth less for the rest of the season than his projection says."""
    remaining = remaining_points_by_espn_id(
        _run_state(week=5, ir_player="INJURY_RESERVE"), _small_id_map()
    )
    assert remaining["1"] == pytest.approx(100.0)
    # 13 games left, 4 missed: 9/13 of his projection.
    assert remaining["2"] == pytest.approx(100.0 * 9 / 13)


def test_the_horizon_is_the_one_the_ros_frame_was_built_over() -> None:
    """`run.ros` holds remaining points as of `run.week`, so the multiplier has to use the same
    week. Scaling that total by a discount derived from a DIFFERENT week applies it over a
    denominator the numerator was never built for -- which an earlier version did, in the name
    of making a `--week` override move "everything"."""
    late = remaining_points_by_espn_id(
        _run_state(week=12, ir_player="INJURY_RESERVE"), _small_id_map()
    )
    assert late["2"] == pytest.approx(100.0 * 2 / 6), "six games left at week 12, four missed"


def test_a_player_the_pool_cannot_price_is_absent_from_the_mapping() -> None:
    """Not zero. `waivers._weakest` reads the absence as "cannot price him" rather than "he is
    worthless", which is what stops a kicker being recommended as a free drop."""
    state = _run_state()
    state.ros = state.ros.iloc[:1]
    remaining = remaining_points_by_espn_id(state, _small_id_map())
    assert set(remaining) == {"1"}


def test_a_free_agents_season_points_are_discounted_by_his_own_injury() -> None:
    """The ADD side of a season swap. Without the free agents' statuses an injured one was
    compared at full health against a roster that was not."""
    state = _run_state(week=5)
    state.roster = state.roster.iloc[:1]  # player 2 is on the wire, not my roster
    free_agents = pd.DataFrame(
        {"player_id": [2], "injury_status": pd.Series(["INJURY_RESERVE"], dtype=_PYARROW_STR)}
    )
    assert remaining_points_by_espn_id(state, _small_id_map())["2"] == pytest.approx(100.0)
    discounted = remaining_points_by_espn_id(state, _small_id_map(), free_agents)
    assert discounted["2"] == pytest.approx(100.0 * 9 / 13)
