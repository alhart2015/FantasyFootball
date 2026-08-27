"""Stage 1: does this free agent crack my starting lineup?

The tests that matter here are the ones about what the tool refuses to recommend. In a
16-team league the honest answer most weeks is "nobody", and a recommender that cannot say
that is worse than no recommender.
"""

from __future__ import annotations

import pandas as pd
import pytest

from projections.draft.league_config import LeagueConfig
from projections.midseason.waivers import Candidate, rank_free_agents
from projections.schemas import _PYARROW_STR, InjuryStatus, RosterSlot, Ruleset

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


def _players(rows: list[tuple[int, str, str, str]]) -> pd.DataFrame:
    """`(player_id, name, position, injury_status)` -> a roster or free-agent frame."""
    frame = pd.DataFrame(
        {
            "player_id": [r[0] for r in rows],
            "player": pd.Series([r[1] for r in rows], dtype=_PYARROW_STR),
            "pos": pd.Series([r[2] for r in rows], dtype=_PYARROW_STR),
            "nfl_team": pd.Series(["KC"] * len(rows), dtype=_PYARROW_STR),
            "injury_status": pd.Series([r[3] for r in rows], dtype=_PYARROW_STR),
            "percent_owned": [50.0] * len(rows),
            "on_waivers": [False] * len(rows),
        }
    )
    return frame


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
    roster: pd.DataFrame | None = None,
    **kwargs: object,
) -> list[Candidate]:
    return rank_free_agents(
        _full_roster() if roster is None else roster,
        free_agents,
        {**BASE_PROJECTIONS, **(projections or {})},
        BASE_REMAINING,
        LEAGUE,
        **kwargs,  # type: ignore[arg-type]
    )


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
    [candidate] = rank_free_agents(
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
    [candidate] = rank_free_agents(
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
