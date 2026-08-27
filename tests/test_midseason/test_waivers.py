"""Stage 1: does this free agent crack my starting lineup?

The tests that matter here are the ones about what the tool refuses to recommend. In a
16-team league the honest answer most weeks is "nobody", and a recommender that cannot say
that is worse than no recommender.
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
    rank_free_agents,
    remaining_points_by_espn_id,
    weekly_projections_by_espn_id,
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
    """`(player_id, name, position, injury_status)` -> a roster or free-agent frame."""
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


#: Starters project well, bench players badly, so a free agent has to beat a real starter.
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

#: Rest-of-season value, which is what a drop costs. The bench is cheap, starters are not.
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


def _rank(
    free_agents: pd.DataFrame,
    *,
    projections: dict[str, float] | None = None,
    remaining: dict[str, float] | None = None,
    roster: pd.DataFrame | None = None,
    source_is_injury_aware: bool = True,
    min_gain: float = 0.5,
) -> list[Candidate]:
    """Explicit keywords rather than `**kwargs: object`.

    The kwargs form threw away every argument type and then needed a
    `# type: ignore[arg-type]` at each call site to hide the result -- seven of them, in a repo
    whose CLAUDE.md forbids broad ignores to make things pass.
    """
    candidates, _ = rank_free_agents(
        _full_roster() if roster is None else roster,
        free_agents,
        {**BASE_PROJECTIONS, **(projections or {})},
        BASE_REMAINING if remaining is None else remaining,
        LEAGUE,
        source_is_injury_aware=source_is_injury_aware,
        min_gain=min_gain,
    )
    return candidates


def _open_spots(
    free_agents: pd.DataFrame,
    *,
    projections: dict[str, float] | None = None,
    roster: pd.DataFrame | None = None,
) -> int:
    _, spots = rank_free_agents(
        _full_roster() if roster is None else roster,
        free_agents,
        {**BASE_PROJECTIONS, **(projections or {})},
        BASE_REMAINING,
        LEAGUE,
    )
    return spots


# --- the refusals, which are the point ----------------------------------------------------------


def test_a_good_free_agent_who_would_not_start_scores_nothing() -> None:
    """The honest answer most weeks in a 16-team league, and the reason this is a filter.

    He projects 10 points, which is better than every player on my bench -- and worse than the
    seven I am already starting, so adding him changes nothing about my week.
    """
    agents = _players([(99, "Decent WR", "WR", "ACTIVE")])
    assert _rank(agents, projections={"99": 10.0}) == []


def test_a_player_with_no_projection_is_never_recommended() -> None:
    """A bye week, or nobody projected him. Unstartable, which is a different fact from being
    projected at zero, and the lineup chooser depends on the difference."""
    agents = _players([(99, "On Bye", "WR", "ACTIVE")])
    assert _rank(agents) == []


def test_noise_is_filtered_out() -> None:
    """Half a point of lineup gain is not a roster move, and a list that includes it trains the
    reader to skip the list."""
    agents = _players([(99, "Marginal WR", "WR", "ACTIVE")])
    # He beats WR2 by 0.2 -- but that pushes WR2 into the flex, which pushes FlexRB out, so
    # the LINEUP gains 1.2. See `test_displacing_a_starter_cascades_down_the_lineup`.
    assert _rank(agents, projections={"99": 12.2}, min_gain=2.0) == []
    assert _rank(agents, projections={"99": 12.2})[0].lineup_gain == pytest.approx(1.2)


# --- the recommendations -----------------------------------------------------------------------


def test_a_free_agent_who_beats_a_starter_is_recommended_with_the_margin() -> None:
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    [candidate] = _rank(agents, projections={"99": 20.0})
    assert candidate.player == "Stud WR"
    # He displaces WR2 (12.0) into the flex, which displaces FlexRB (11.0) out entirely.
    assert candidate.lineup_gain == pytest.approx(9.0)


def test_the_flex_is_handled_without_a_rule_about_it() -> None:
    """A receiver who beats my flex running back counts, and one who does not, does not.
    Nothing in this module knows what a flex is -- `choose_starters` does."""
    agents = _players([(99, "Flex WR", "WR", "ACTIVE")])
    assert _rank(agents, projections={"99": 11.5})[0].lineup_gain == pytest.approx(0.5)
    assert _rank(agents, projections={"99": 10.5}) == []


def test_displacing_a_starter_cascades_down_the_lineup() -> None:
    """The subtlety worth pinning, because it makes every gain bigger than the head-to-head
    margin suggests and a reader checking by hand will not expect it.

    A 12.2 receiver beats WR2 (12.0) by two tenths. But WR2 is not cut -- he moves into the
    flex, where he beats FlexRB (11.0) by a full point, and FlexRB leaves the lineup. So the
    lineup gains 1.2, not 0.2. Comparing the add against the man he directly replaces
    understates every recommendation this tool makes.
    """
    agents = _players([(99, "Slight WR", "WR", "ACTIVE")])
    [candidate] = _rank(agents, projections={"99": 12.2})
    assert candidate.lineup_gain == pytest.approx(1.2)


def test_a_bye_week_hole_makes_a_replacement_valuable() -> None:
    """The case that actually drives streaming. My TE has no projection this week, so the slot
    is empty and any startable tight end is pure gain."""
    projections = dict(BASE_PROJECTIONS)
    del projections["6"]  # TE1 on bye
    agents = _players([(99, "Streamer TE", "TE", "ACTIVE")])
    (candidate,), _ = rank_free_agents(
        _full_roster(), agents, {**projections, "99": 7.0}, BASE_REMAINING, LEAGUE
    )
    assert candidate.lineup_gain == pytest.approx(7.0), "an empty slot means the whole projection"


# --- the drop side -----------------------------------------------------------------------------


def test_the_drop_is_the_cheapest_player_who_is_not_starting() -> None:
    """Never someone the lineup uses. Picking "my worst player" would happily suggest dropping
    somebody who is still in the starting eleven."""
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    [candidate] = _rank(agents, projections={"99": 20.0})
    assert candidate.drop_player == "Bench0", "cheapest bench player by rest-of-season value"
    assert candidate.drop_cost == pytest.approx(20.0)
    assert not candidate.is_free


def test_a_displaced_starter_becomes_droppable_but_only_if_he_is_cheapest() -> None:
    """FlexRB is pushed out of the lineup by the add, so he IS droppable -- but he is worth 110
    rest-of-season points and the bench is worth 20, so the tool does not suggest him."""
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    [candidate] = _rank(agents, projections={"99": 20.0})
    assert candidate.drop_player != "FlexRB"


def test_ir_slots_do_not_count_as_open_roster_spots() -> None:
    """They hold a player who is already hurt; a healthy add cannot be parked there. Counting
    them made a full 12-man roster look like it had two spaces going spare, so every
    recommendation came back free and no drop was ever named."""
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    [candidate] = _rank(agents, projections={"99": 20.0})
    assert not candidate.is_free, "the roster is full; somebody has to go"


def test_an_open_roster_spot_means_nobody_is_dropped() -> None:
    """Categorically different from every other recommendation, so it is checked before a drop
    is chosen rather than by pricing a drop nobody has to make."""
    roster = _full_roster().iloc[:-1]  # one bench spot free
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    [candidate] = _rank(agents, projections={"99": 20.0}, roster=roster)
    assert candidate.is_free
    assert candidate.drop_player == ""
    assert candidate.drop_cost == 0.0


def test_the_free_agent_is_never_his_own_drop_candidate() -> None:
    """He is the last row, so a candidate who does not crack the lineup would otherwise be his
    own cheapest leftover -- and the tool would recommend adding and immediately dropping him."""
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    [candidate] = _rank(agents, projections={"99": 20.0})
    assert candidate.drop_player_id != 99


# --- injuries ----------------------------------------------------------------------------------


def test_a_questionable_free_agent_is_discounted() -> None:
    """0.86 on a single week, which is exactly the size that decides a close call: he projects
    ahead of my starter on paper and behind him once the designation is priced."""
    agents = _players([(99, "Questionable WR", "WR", "QUESTIONABLE")])
    healthy = _players([(99, "Healthy WR", "WR", "ACTIVE")])
    # Healthy at 12.5 he beats WR2 (12.0), and the cascade puts WR2 in the flex over FlexRB
    # (11.0): the lineup gains 1.5 and this is a move.
    assert _rank(healthy, projections={"99": 12.5})[0].lineup_gain == pytest.approx(1.5)
    # The same player carrying a Questionable tag projects 12.5 x 0.86 = 10.75, which beats
    # nobody I already start. Same projection, opposite recommendation.
    assert _rank(agents, projections={"99": 12.5}) == []


def test_the_discount_can_demote_a_move_without_killing_it() -> None:
    """The in-between case, and the reason 0.86 is worth measuring rather than rounding to 1.

    At 13.5 healthy he displaces a starter and cascades, worth 2.5. Discounted to 11.61 he no
    longer beats WR2 -- but he still beats FlexRB, so he is a marginal flex upgrade worth 0.61
    rather than either a headline add or nothing at all.
    """
    healthy = _players([(99, "Healthy WR", "WR", "ACTIVE")])
    agents = _players([(99, "Questionable WR", "WR", "QUESTIONABLE")])
    assert _rank(healthy, projections={"99": 13.5})[0].lineup_gain == pytest.approx(2.5)
    assert _rank(agents, projections={"99": 13.5})[0].lineup_gain == pytest.approx(0.61)


def test_espn_priced_statuses_are_not_discounted_twice() -> None:
    """ESPN's weekly feed already zeroes players it lists as Out. This asserts the flag reaches
    the multiplier: with the default, an Out player carrying a projection is left alone; told
    the source is naive, he is zeroed."""
    agents = _players([(99, "Out WR", "WR", "OUT")])
    priced = _rank(agents, projections={"99": 20.0}, source_is_injury_aware=True)
    naive = _rank(agents, projections={"99": 20.0}, source_is_injury_aware=False)
    assert priced and priced[0].lineup_gain == pytest.approx(9.0)
    assert naive == []


def test_an_injured_player_on_my_roster_opens_the_hole_he_leaves() -> None:
    """The motivating case. My WR1 is Out, so ESPN projects him at nothing this week and a
    replacement is worth the whole slot rather than the margin over him."""
    roster = _full_roster()
    roster.loc[roster["player"] == "WR1", "injury_status"] = "OUT"
    projections = {**BASE_PROJECTIONS, "4": 0.0, "99": 10.0}
    agents = _players([(99, "Replacement WR", "WR", "ACTIVE")])
    (candidate,), _ = rank_free_agents(
        roster, agents, projections, BASE_REMAINING, LEAGUE, source_is_injury_aware=False
    )
    assert candidate.lineup_gain > 0
    assert candidate.injury_status is InjuryStatus.ACTIVE


# --- ordering and shape ------------------------------------------------------------------------


def test_candidates_come_back_best_first() -> None:
    agents = _players(
        [
            (98, "Better WR", "WR", "ACTIVE"),
            (99, "Good WR", "WR", "ACTIVE"),
        ]
    )
    ranked = _rank(agents, projections={"98": 22.0, "99": 15.0})
    assert [c.player for c in ranked] == ["Better WR", "Good WR"]
    assert ranked[0].lineup_gain > ranked[1].lineup_gain


def test_an_empty_wire_is_an_empty_list_not_an_error() -> None:
    assert _rank(_players([])) == []


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
    """The wiring test the web UI taught us to write.

    Every other test here builds its frames by hand, which means none of them would notice
    `parse_rosters` renaming a column or `parse_free_agents` producing a different id dtype.
    This one starts from ESPN-shaped payloads and goes through the real parsers, so the seam
    between ingest and the recommender is covered by something.
    """
    payload = espn_payload(played_weeks=0)
    roster = parse_rosters(payload)
    roster = roster[roster["team_id"] == MY_TEAM_ID]
    assert not roster.empty, "the fixture league has rosters"

    free_agents, warning = parse_free_agents(_fa_payload(900_002, "Wire Stud", 3), limit=50)
    assert warning is None

    # Everyone on my roster projects modestly; the free agent projects far above them, so he
    # must crack the lineup whatever the fixture's slot layout happens to be.
    projections = {str(pid): 8.0 for pid in roster["player_id"]}
    projections["900002"] = 30.0
    remaining = {str(pid): 50.0 for pid in roster["player_id"]}

    candidates, _ = rank_free_agents(roster, free_agents, projections, remaining, LEAGUE)
    assert [c.player for c in candidates] == ["Wire Stud"]
    assert candidates[0].lineup_gain > 0
    assert candidates[0].position == "WR"
    assert candidates[0].percent_owned == pytest.approx(12.0)


def test_an_injured_free_agent_survives_the_parsers_with_his_status() -> None:
    """Both sides of a swap are compared on `injury_status`, and it has to reach the
    recommender from the parser rather than from a fixture that happens to spell it right."""
    payload = espn_payload(played_weeks=0)
    roster = parse_rosters(payload)
    roster = roster[roster["team_id"] == MY_TEAM_ID]
    free_agents, _ = parse_free_agents(
        _fa_payload(900_003, "Hurt Stud", 3, status="QUESTIONABLE"), limit=50
    )
    projections = {str(pid): 8.0 for pid in roster["player_id"]}
    projections["900003"] = 30.0
    remaining = {str(pid): 50.0 for pid in roster["player_id"]}

    (candidate,), _ = rank_free_agents(roster, free_agents, projections, remaining, LEAGUE)
    assert candidate.injury_status is InjuryStatus.QUESTIONABLE
    # 30.0 x 0.86 = 25.8, so the gain is smaller than the healthy version would give.
    healthy, _ = parse_free_agents(_fa_payload(900_003, "Fit Stud", 3), limit=50)
    (fit,), _ = rank_free_agents(roster, healthy, projections, remaining, LEAGUE)
    assert candidate.lineup_gain < fit.lineup_gain


# --- one roster model: IR is not an active spot, and an IR player is not startable ------------


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
    """The mirror of the bug the last fix introduced. Counting IR SLOTS as capacity made a full
    roster look like it had spares, so every recommendation came back free. Counting IR PLAYERS
    against active capacity made a roster with someone on IR look full, so the tool named a
    drop nobody had to make. A spot is active, and so is the player who fills it."""
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    assert _open_spots(agents, projections={"99": 20.0}, roster=_roster_with_ir()) == 1
    [candidate] = _rank(agents, projections={"99": 20.0}, roster=_roster_with_ir())
    assert candidate.is_free, "eleven active players in twelve spots: nobody has to go"


def test_a_player_on_ir_cannot_hold_a_starting_slot() -> None:
    """The morning-after case, and the one the whole tool exists for.

    My WR1 is on IR. ESPN still projects him -- `weekly_multiplier` deliberately leaves an
    IR player alone when the source already prices injuries -- so if the lineup counts him,
    every wire receiver's gain falls under `min_gain` and the tool reports "nothing on the
    wire would change your lineup" on exactly the day it should be shouting.
    """
    roster = _roster_with_ir()
    roster.loc[roster["player"] == "WR1", "lineup_slot"] = "IR"
    roster.loc[roster["player"] == "WR1", "injury_status"] = "INJURY_RESERVE"

    agents = _players([(99, "Replacement WR", "WR", "ACTIVE")])
    [candidate] = _rank(agents, projections={"99": 13.0}, roster=roster)
    # WR1 (16.0) is unstartable, so the slot is filled by WR2 (12.0) and the replacement takes
    # the other one. Without the IR rule WR1 holds his slot and 13.0 beats nobody.
    assert candidate.lineup_gain > 0


# --- a player we cannot price is not a player worth zero ---------------------------------------


def test_a_player_the_pool_cannot_price_is_never_the_recommended_drop() -> None:
    """A kicker, a defense, a back nobody projects yet -- absent from `remaining_points`.

    Defaulting him to 0.0 made him the cheapest leftover by construction, so the tool
    recommended dropping him and printed "costs 0 rest-of-season points" underneath. Those are
    exactly the players a waiver tool should be careful with, and "we have no number for him"
    is not "he is worth nothing".
    """
    roster = _full_roster()
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    # Bench0 is the cheapest player we CAN price; Bench1 has no price at all.
    remaining = {k: v for k, v in BASE_REMAINING.items() if k != "11"}
    [candidate] = _rank(agents, projections={"99": 20.0}, remaining=remaining, roster=roster)
    assert candidate.drop_player == "Bench0"
    assert candidate.drop_player != "Bench1"


def test_no_droppable_player_is_not_the_same_as_needing_no_drop() -> None:
    """`is_free` used to report both as "roster spot open", which is a lie in one of them --
    and it is the one the module calls the first thing worth telling a reader."""
    roster = _full_roster()
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    # Nobody who could be dropped can be priced -- the bench, and the flex RB the add
    # displaces. The roster is still full.
    remaining = {k: v for k, v in BASE_REMAINING.items() if int(k) < 7}
    [candidate] = _rank(agents, projections={"99": 20.0}, remaining=remaining, roster=roster)
    assert not candidate.is_free, "the roster is full; we simply could not price a drop"
    assert candidate.drop_player_id is None


def test_the_caller_is_told_how_many_spots_are_actually_open() -> None:
    """Every `needs_no_drop` candidate is claiming the SAME spot. Three of them printed without
    that number invites a roster overfill."""
    roster = _full_roster().iloc[:-2]  # two bench spots free
    agents = _players([(98, "Stud WR", "WR", "ACTIVE"), (99, "Other WR", "WR", "ACTIVE")])
    spots = _open_spots(agents, projections={"98": 20.0, "99": 19.0}, roster=roster)
    assert spots == 2
    ranked = _rank(agents, projections={"98": 20.0, "99": 19.0}, roster=roster)
    assert all(c.is_free for c in ranked)


def test_a_float_player_id_column_does_not_crash() -> None:
    """Any frame that has been through a merge introducing an NA becomes float64, and
    `int("12345.0")` raises -- the exact shape the old comment claimed to handle."""
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    agents["player_id"] = agents["player_id"].astype("float64")
    [candidate] = _rank(agents, projections={"99": 20.0})
    assert candidate.player_id == 99


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
    """Not zero. `_drop_candidate` reads the absence as "cannot price him" rather than "he is
    worthless", which is what stops a kicker being recommended as a free drop."""
    state = _run_state()
    state.ros = state.ros.iloc[:1]
    remaining = remaining_points_by_espn_id(state, _small_id_map())
    assert set(remaining) == {"1"}


def test_a_player_on_ir_is_never_the_recommended_drop() -> None:
    """The third place `is_on_ir` says needs it, and the one an earlier version left out.

    Dropping an IR player frees an IR slot, not the ACTIVE spot the add needs -- so the move
    does not fit and the recommendation is unactionable. He is also the likeliest player to be
    named: his projection is forced to `None` so he is a permanent leftover, and his cost is
    injury-discounted so he is often the cheapest one on the roster.
    """
    roster = _roster_with_ir()
    # Fill the last active spot so a drop is genuinely required.
    roster = pd.concat(
        [roster, _players([(30, "Bench4", "WR", "ACTIVE", "BENCH")])], ignore_index=True
    )
    remaining = {**BASE_REMAINING, "20": 1.0, "30": 40.0}
    agents = _players([(99, "Stud WR", "WR", "ACTIVE")])
    [candidate] = _rank(agents, projections={"99": 20.0}, remaining=remaining, roster=roster)
    assert not candidate.is_free, "the active roster is full"
    # The POSITIVE assertion. `!= "Hurt WR"` also passes when no drop is found at all, which is
    # a strictly worse outcome than naming him -- so it would have missed a regression that
    # broke `_drop_candidate` entirely.
    assert candidate.drop_player == "Bench0", "the cheapest priceable non-IR leftover"
