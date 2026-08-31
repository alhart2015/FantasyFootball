"""Trade analyzer: valuation, roster shape, and trade generation.

Three failure modes carry the weight, all of the plausible-wrong-answer kind:

- Scoring the two valuations under different rulesets, which manufactures an edge on every
  pass-catcher at once and looks like insight.
- Comparing a player to his direct backup instead of re-optimising the lineup, which
  understates every trade by the size of the cascade.
- Losing the pairing on the simulation by changing roster order, which the waiver work measured
  making a no-op swap read 0.05 wins.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from projections.midseason.roster_shape import (
    TeamShape,
    lineup_points,
    median_starters,
    need,
    surplus,
    team_shapes,
)
from projections.midseason.trades import TradeProposal, _swap, generate
from projections.midseason.valuation import PlayerValue, espn_season_points
from projections.schemas import InjuryStatus, RosterSlot, Ruleset

SLOTS: dict[RosterSlot, int] = {
    RosterSlot.QB: 1,
    RosterSlot.RB: 2,
    RosterSlot.WR: 2,
    RosterSlot.TE: 1,
    RosterSlot.FLEX: 1,
    RosterSlot.BENCH: 5,
}


def pv(
    name: str, pos: str, ours: float, market: float | None = None, espn_id: int = 0
) -> PlayerValue:
    return PlayerValue(
        gsis_id=f"00-{abs(hash(name)) % 10_000_000:07d}",
        espn_id=espn_id or abs(hash(name)) % 100_000,
        full_name=name,
        position=pos,
        market=ours if market is None else market,
        ours=ours,
        injury_status=InjuryStatus.ACTIVE,
        injury_raw="",
    )


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------


def _external(**stats: float) -> pd.DataFrame:
    base = {
        "source": ["ESPN"],
        "gsis_id": ["00-0000001"],
        "passing_yards": [0.0],
        "passing_tds": [0.0],
        "interceptions": [0.0],
        "rushing_yards": [0.0],
        "rushing_tds": [0.0],
        "receptions": [0.0],
        "receiving_yards": [0.0],
        "receiving_tds": [0.0],
        "fumbles_lost": [0.0],
    }
    for key, value in stats.items():
        base[key] = [value]
    return pd.DataFrame(base)


def test_espn_points_are_scored_under_the_league_ruleset_not_espns() -> None:
    """The units error that looks exactly like an edge, and would look like one on every
    pass-catcher simultaneously: half-PPR against full-PPR is 0.5 points per reception."""
    line = _external(receptions=100.0, receiving_yards=1000.0)

    half = espn_season_points(line, Ruleset.espn_half())["00-0000001"]
    ppr = espn_season_points(line, Ruleset.espn_ppr())["00-0000001"]

    assert ppr - half == pytest.approx(50.0)


def test_edge_is_zero_when_both_sources_agree() -> None:
    """The identity the whole feature rests on. If a player's ESPN line scores to exactly what
    our pool says, there is no disagreement and `edge` must be 0.0 -- not a small residue from
    a ruleset or rounding mismatch."""
    espn = espn_season_points(
        _external(rushing_yards=1000.0, rushing_tds=10.0), Ruleset.espn_half()
    )["00-0000001"]

    assert pv("x", "RB", ours=espn, market=espn).edge == pytest.approx(0.0)


def test_a_row_with_no_stat_line_is_omitted_not_scored_as_zero() -> None:
    """Sleeper carries ADP only. Scoring its null line as 0.0 would make every Sleeper-only
    player look like a confident projection of nothing rather than an absence of an opinion."""
    frame = pd.DataFrame({"source": ["SLEEPER"], "gsis_id": ["00-0000002"]})

    assert espn_season_points(frame, Ruleset.espn_half()) == {}


# ---------------------------------------------------------------------------
# Surplus and need
# ---------------------------------------------------------------------------


def _roster() -> list[PlayerValue]:
    return [
        pv("QB1", "QB", 300.0),
        pv("RB1", "RB", 200.0),
        pv("RB2", "RB", 150.0),
        pv("RB3", "RB", 120.0),
        pv("RB4", "RB", 100.0),
        pv("WR1", "WR", 180.0),
        pv("WR2", "WR", 140.0),
        pv("TE1", "TE", 110.0),
    ]


def test_a_benched_player_has_zero_surplus() -> None:
    """RB4 cannot start: RB1/RB2 fill the RB slots and RB3 takes the flex. Giving him away
    costs the lineup nothing, which is exactly what makes him a trade chip."""
    roster = _roster()

    assert surplus(roster, roster[4], SLOTS) == pytest.approx(0.0)


def test_surplus_measures_the_cascade_not_the_direct_backup() -> None:
    """Remove RB2 and RB3 slides into the RB slot while RB4 enters the flex. The lineup loses
    150 - 100 = 50, not the 150 - 120 = 30 a direct-backup comparison would report."""
    roster = _roster()

    assert surplus(roster, roster[2], SLOTS) == pytest.approx(50.0)


def test_the_only_quarterback_has_surplus_equal_to_his_whole_projection() -> None:
    """No replacement exists on this roster, so the QB slot empties entirely."""
    roster = _roster()

    assert surplus(roster, roster[0], SLOTS) == pytest.approx(300.0)


def test_need_is_zero_where_an_upgrade_would_not_start() -> None:
    roster = _roster()

    assert need(roster, "RB", 90.0, SLOTS) == pytest.approx(0.0)


def test_need_is_the_upgrade_over_the_man_he_displaces() -> None:
    """A 160-point TE replaces the 110-point starter: +50."""
    roster = _roster()

    assert need(roster, "TE", 160.0, SLOTS) == pytest.approx(50.0)


def test_need_is_never_negative() -> None:
    """The optimiser may leave an addition on the bench, so an add cannot hurt."""
    roster = _roster()

    assert need(roster, "WR", 1.0, SLOTS) >= 0.0


def test_median_starters_ignores_the_bench() -> None:
    """Taken over all rostered players the reference would sit near replacement level and every
    team would look equally needy at every position."""
    rosters = {1: _roster(), 2: _roster()}

    medians = median_starters(rosters, SLOTS)

    # Starting RBs are 200/150/120 (flex) on each roster; RB4 at 100 never starts.
    assert medians["RB"] == pytest.approx(150.0)


def test_shape_is_computed_the_same_way_for_every_team() -> None:
    """A separate code path for "my" roster is how the two sides of a trade come to be scored
    by subtly different rules."""
    rosters = {1: _roster(), 2: _roster()}

    shapes = team_shapes(rosters, {1: "A", 2: "B"}, SLOTS)

    assert shapes[1].surplus == shapes[2].surplus
    assert shapes[1].need == shapes[2].need


# ---------------------------------------------------------------------------
# Trade generation
# ---------------------------------------------------------------------------


def test_swap_replaces_in_place_and_preserves_roster_order() -> None:
    """Roster order reaches the simulator, which draws per player in order. Appending instead
    of replacing shifts every later player onto a different draw and silently unpairs the
    comparison -- the bug that made a no-op swap read 0.05 wins."""
    roster = _roster()
    incoming = pv("NEW", "RB", 175.0)

    swapped = _swap(roster, [roster[2]], [incoming])

    assert [p.full_name for p in swapped] == [
        "QB1",
        "RB1",
        "NEW",
        "RB3",
        "RB4",
        "WR1",
        "WR2",
        "TE1",
    ]


def test_a_no_op_swap_changes_the_lineup_by_exactly_zero() -> None:
    """The gate the spec names. Anything but an exact zero means the machinery is leaking."""
    roster = _roster()

    swapped = _swap(roster, [roster[3]], [roster[3]])

    assert lineup_points(swapped, SLOTS) == lineup_points(roster, SLOTS)


def _shapes_for_a_fit_trade() -> dict[int, TeamShape]:
    """I am four deep at RB and starting a 90-point WR2; they are the mirror.

    The numbers matter. A 1-for-1 of equal value is **neutral** whenever the flex can absorb
    it — send a 180 and receive a 180 and the flex simply swaps which one it holds. A real fit
    gain needs the outgoing player to be one who *cannot start for me* (surplus 0, because the
    flex is already taken by someone better) and the incoming one to displace a genuinely weak
    starter. That is also what makes it sellable: both sides give up a bench body.
    """
    mine = [
        pv("MyQB", "QB", 300.0),
        pv("MyRB1", "RB", 200.0),
        pv("MyRB2", "RB", 190.0),
        pv("MyRB3", "RB", 185.0),
        pv("MyRB4", "RB", 180.0),  # benched: the flex already holds MyRB3
        pv("MyWR1", "WR", 100.0),
        pv("MyWR2", "WR", 90.0),
        pv("MyTE", "TE", 110.0),
    ]
    theirs = [
        pv("TheirQB", "QB", 300.0),
        pv("TheirWR1", "WR", 200.0),
        pv("TheirWR2", "WR", 190.0),
        pv("TheirWR3", "WR", 185.0),
        pv("TheirWR4", "WR", 180.0),  # benched for the same reason
        pv("TheirRB1", "RB", 100.0),
        pv("TheirRB2", "RB", 90.0),
        pv("TheirTE", "TE", 110.0),
    ]
    return team_shapes({1: mine, 2: theirs}, {1: "me", 2: "them"}, SLOTS)


def test_generate_finds_the_mutually_beneficial_positional_swap() -> None:
    shapes = _shapes_for_a_fit_trade()

    proposals = generate(shapes[1], shapes[2], SLOTS, max_players=1)

    assert proposals, "a surplus-RB-for-surplus-WR swap should clear both filters"
    best = max(proposals, key=lambda t: t.my_lineup_gain)
    assert best.send[0].position == "RB"
    assert best.receive[0].position == "WR"
    assert best.their_lineup_gain_market > 0.0


def test_generate_rejects_trades_that_do_not_help_the_partner() -> None:
    """They evaluate on ESPN's numbers. A trade that does not improve their own lineup there is
    not one they have any reason to accept, however good it is for me."""
    shapes = _shapes_for_a_fit_trade()

    for proposal in generate(shapes[1], shapes[2], SLOTS, max_players=2):
        assert proposal.their_lineup_gain_market > 0.0


def test_generate_rejects_lowballs_beyond_the_tolerance() -> None:
    shapes = _shapes_for_a_fit_trade()

    for proposal in generate(shapes[1], shapes[2], SLOTS, max_players=2, espn_tolerance=0.0):
        assert proposal.espn_balance <= 0.0


def test_fit_and_edge_decompose_the_gain() -> None:
    """A trade between players everyone values identically is pure fit: the edge term must be
    exactly zero, so a fit trade is never mistaken for a bet on our model."""
    shapes = _shapes_for_a_fit_trade()

    for proposal in generate(shapes[1], shapes[2], SLOTS, max_players=1):
        assert proposal.edge_gain == pytest.approx(0.0)
        assert proposal.fit_gain == pytest.approx(proposal.my_lineup_gain)


def test_edge_is_attributed_when_we_disagree_with_the_market() -> None:
    """Same lineup gain, but half of it comes from us rating the incoming player above ESPN."""
    incoming = pv("Sleeper", "WR", ours=200.0, market=180.0)
    outgoing = pv("Hyped", "RB", ours=100.0, market=100.0)
    proposal = TradeProposal(
        partner_id=2,
        partner_name="them",
        send=(outgoing,),
        receive=(incoming,),
        my_lineup_gain=100.0,
        their_lineup_gain_market=1.0,
        espn_balance=80.0,
    )

    assert proposal.edge_gain == pytest.approx(20.0)
    assert proposal.fit_gain == pytest.approx(80.0)
    assert proposal.is_lowball


def test_a_proposal_below_the_noise_floor_is_flagged() -> None:
    small = TradeProposal(
        partner_id=2,
        partner_name="them",
        send=(),
        receive=(),
        my_lineup_gain=1.0,
        their_lineup_gain_market=1.0,
        espn_balance=0.0,
        delta_wins=0.05,
    )
    real = TradeProposal(
        partner_id=2,
        partner_name="them",
        send=(),
        receive=(),
        my_lineup_gain=1.0,
        their_lineup_gain_market=1.0,
        espn_balance=0.0,
        delta_wins=0.40,
    )

    assert not small.above_noise
    assert real.above_noise


# ---------------------------------------------------------------------------
# The payload swap — where the pairing is won or lost
# ---------------------------------------------------------------------------


def _payload_proposal(payload: dict[str, Any], mine: int, theirs: int) -> TradeProposal:
    """A 1-for-1 between two real entries in `payload`, as a proposal."""

    def first_player(team_id: int) -> PlayerValue:
        team = next(t for t in payload["teams"] if int(t["id"]) == team_id)
        entry = team["roster"]["entries"][0]
        player = entry["playerPoolEntry"]["player"]
        return pv(str(player["fullName"]), "RB", 100.0, espn_id=int(player["id"]))

    return TradeProposal(
        partner_id=theirs,
        partner_name="them",
        send=(first_player(mine),),
        receive=(first_player(theirs),),
        my_lineup_gain=1.0,
        their_lineup_gain_market=1.0,
        espn_balance=0.0,
    )


def test_the_payload_swap_is_an_exact_involution() -> None:
    """Apply a trade, then apply its inverse: the payload must come back **byte-identical**,
    ids and list positions included.

    This is the gate. Roster order reaches the simulator, which draws per player in order, so a
    swap that reorders a roster silently unpairs the comparison — the waiver work measured that
    failure making a no-op swap read 0.05 wins, and only an exact-equality assertion catches it.
    A round trip tests order and identity together, which comparing totals cannot.
    """
    from projections.midseason.trades import payload_with_trade
    from tests.test_midseason.conftest import TEAM_IDS, espn_payload

    payload = espn_payload()
    mine, theirs = TEAM_IDS[0], TEAM_IDS[1]
    forward = _payload_proposal(payload, mine, theirs)

    once = payload_with_trade(payload, forward, my_team_id=mine)
    back = TradeProposal(
        partner_id=theirs,
        partner_name="them",
        send=forward.receive,
        receive=forward.send,
        my_lineup_gain=1.0,
        their_lineup_gain_market=1.0,
        espn_balance=0.0,
    )
    twice = payload_with_trade(once, back, my_team_id=mine)

    def roster(pay: dict[str, Any], team_id: int) -> list[int]:
        team = next(t for t in pay["teams"] if int(t["id"]) == team_id)
        return [int(e["playerPoolEntry"]["player"]["id"]) for e in team["roster"]["entries"]]

    assert roster(twice, mine) == roster(payload, mine)
    assert roster(twice, theirs) == roster(payload, theirs)
    # ...and the one-way trip really did move the players, so the round trip is not vacuous.
    assert roster(once, mine) != roster(payload, mine)


def test_the_payload_swap_does_not_mutate_the_caller() -> None:
    """A mutated payload would make the baseline and every later proposal compare against a
    roster that has been silently accumulating trades, and every delta would look plausible."""
    from projections.midseason.trades import payload_with_trade
    from tests.test_midseason.conftest import TEAM_IDS, espn_payload

    payload = espn_payload()
    before = [
        int(e["playerPoolEntry"]["player"]["id"]) for e in payload["teams"][0]["roster"]["entries"]
    ]

    payload_with_trade(
        payload, _payload_proposal(payload, TEAM_IDS[0], TEAM_IDS[1]), my_team_id=TEAM_IDS[0]
    )

    after = [
        int(e["playerPoolEntry"]["player"]["id"]) for e in payload["teams"][0]["roster"]["entries"]
    ]
    assert after == before


def test_a_player_who_is_not_on_the_named_roster_is_refused() -> None:
    """Absorbing it would change a roster's size and unpair the comparison, producing a
    plausible-looking delta and no error."""
    from projections.midseason.standings import ProjectionInputError
    from projections.midseason.trades import payload_with_trade
    from tests.test_midseason.conftest import TEAM_IDS, espn_payload

    payload = espn_payload()
    bogus = TradeProposal(
        partner_id=TEAM_IDS[1],
        partner_name="them",
        send=(pv("Nobody", "RB", 100.0, espn_id=999_999_999),),
        receive=(pv("Also Nobody", "RB", 100.0, espn_id=999_999_998),),
        my_lineup_gain=1.0,
        their_lineup_gain_market=1.0,
        espn_balance=0.0,
    )

    with pytest.raises(ProjectionInputError, match="no slot to occupy"):
        payload_with_trade(payload, bogus, my_team_id=TEAM_IDS[0])
