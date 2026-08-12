"""LiveAuctionSession — the live auction board's controller.

Covers the three things the board asks of it: price a player the way the engine would,
say who to nominate, and record awards without ever letting the room go insolvent.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from projections.draft.assistant.auction.bid_strategy import (
    AuctionView,
    BalancedValueBid,
    build_engine_dollars,
)
from projections.draft.assistant.auction.live import (
    BOARD_BID_MODELS,
    DEFAULT_BID_MODEL,
    LiveAuctionSession,
    build_market_dollars,
)
from projections.draft.assistant.auction.registry import BID_MODELS
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, Position, RosterSlot, Ruleset

_N = 96
_IDS = [f"00-000{i:04d}" for i in range(1, _N + 1)]
_POSITIONS = ["RB", "WR", "QB", "TE"] * (_N // 4)


def _id_map() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array(_IDS, dtype=_PYARROW_STR),
            "position": pd.array(_POSITIONS, dtype=_PYARROW_STR),
            "full_name": pd.array([f"Player {i}" for i in range(1, _N + 1)], dtype=_PYARROW_STR),
            "team": pd.array(["KC"] * _N, dtype=_PYARROW_STR),
        }
    )


def _pool(*, espn: bool = True) -> pd.DataFrame:
    cols = {
        "gsis_id": pd.array(_IDS, dtype=_PYARROW_STR),
        "position": pd.array(_POSITIONS, dtype=_PYARROW_STR),
        "season_mean_fpts": [300.0 - i for i in range(_N)],
        "vorp": [150.0 - i for i in range(_N)],
        "replacement_fpts": [100.0] * _N,
        "consensus_adp": pd.array([float(i + 1) for i in range(_N)], dtype=pd.Float64Dtype()),
        "full_name": pd.array([f"Player {i}" for i in range(1, _N + 1)], dtype=_PYARROW_STR),
    }
    if espn:
        # Deliberately a DIFFERENT shape from our VORP order so model vs market is separable:
        # ESPN loves the 4th player and is lukewarm on the 1st.
        espn_vals = [10 if i == 0 else max(1, 80 - 2 * i) for i in range(_N)]
        espn_vals[3] = 90
        cols["espn_auction_dollars"] = pd.array(espn_vals, dtype=pd.Int64Dtype())
    return pd.DataFrame(cols)


def _league(n_teams: int = 8) -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=n_teams,
        budget=200,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 4,
        },
        ruleset=Ruleset.espn_half(),
    )


def _session(**kw) -> LiveAuctionSession:  # type: ignore[no-untyped-def]
    espn = kw.pop("espn", True)
    n_teams = kw.pop("n_teams", 8)
    return LiveAuctionSession(
        league=kw.pop("league", _league(n_teams)),
        my_seat=kw.pop("my_seat", 1),
        id_map=_id_map(),
        pool=kw.pop("pool", _pool(espn=espn)),
        strategy=kw.pop("strategy", BOARD_BID_MODELS[DEFAULT_BID_MODEL]),
        strategy_name=kw.pop("strategy_name", DEFAULT_BID_MODEL),
        **kw,
    )


@dataclass(frozen=True)
class _FixedBid:
    """A bid model that always desires the same number — isolates the engine clamp."""

    amount: int

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        return self.amount


# --------------------------------------------------------------------------- registry


def test_registry_is_the_tournament_roster() -> None:
    """The board offers the tournament's own models, so a name means one thing everywhere.

    Asserts the contract, not object identity: `_MODELS is BID_MODELS` would have passed with
    a broken registry and failed on a harmless defensive copy.
    """
    assert set(BID_MODELS) <= set(BOARD_BID_MODELS)
    for name, model in BID_MODELS.items():
        assert BOARD_BID_MODELS[name] is model, f"{name} is not the strategy the tournament raced"
    assert "overbid_noramp" in BOARD_BID_MODELS  # the printed cheat sheet's plan


def test_registry_mappings_are_read_only() -> None:
    """Every consumer binds the same object, so a mutation would redefine the tournament
    roster and the live board's menu process-wide."""
    with pytest.raises(TypeError):  # mappingproxy rejects item assignment
        BID_MODELS["static"] = BID_MODELS["balanced"]  # type: ignore[index]
    with pytest.raises(AttributeError):  # ...and has no mutating methods at all
        BOARD_BID_MODELS.pop("balanced")  # type: ignore[attr-defined]
    assert dict(BID_MODELS)  # a copy is still the sanctioned way to build a variant


def test_build_engine_dollars_falls_back_to_our_value() -> None:
    baseline = pd.DataFrame(
        {
            "gsis_id": ["a", "b"],
            "auction_dollars": pd.array([10, 20], dtype=pd.Int64Dtype()),
        }
    )
    partial = pd.Series([7], index=pd.Index(["a"], name="gsis_id"), dtype=pd.Int64Dtype())
    bd = build_engine_dollars(baseline, partial)
    assert int(bd.loc["a", "bot_dollars"]) == 7  # ESPN-priced
    assert int(bd.loc["b", "bot_dollars"]) == 20  # unpriced -> our own value
    assert list(build_engine_dollars(baseline)["bot_dollars"]) == [10, 20]


def test_build_market_dollars_warns_and_falls_back_without_espn() -> None:
    pool, league = _pool(espn=False), _league()
    with pytest.warns(UserWarning, match="no usable espn_auction_dollars"):
        baseline, bot = build_market_dollars(pool, league, market="espn")
    assert bot is None
    assert len(baseline) == len(pool)
    with pytest.raises(ValueError, match="market must be"):
        build_market_dollars(pool, league, market="nope")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- state


def test_recording_a_purchase_moves_budget_slots_and_pool() -> None:
    s = _session()
    before = len(s.available_pool())
    s.record_purchase(_IDS[0], 2, 40)
    assert s.budget(2) == 160
    assert s.open_slots(2) == s.league.roster_size - 1
    assert s.budget(1) == 200  # untouched seat
    assert _IDS[0] in s.drafted_ids
    assert len(s.available_pool()) == before - 1
    assert s.roster_ids(2) == [_IDS[0]]


def test_feasible_max_reserves_a_dollar_for_every_other_slot() -> None:
    s = _session()
    rs = s.league.roster_size
    assert s.feasible_max(1) == 200 - (rs - 1)
    s.record_purchase(_IDS[0], 1, 100)
    assert s.feasible_max(1) == 100 - (rs - 2)


def test_feasible_max_is_zero_for_a_seat_with_no_open_slot() -> None:
    """A full seat cannot bid. The bare formula reads budget + min_bid at zero open slots --
    more than the seat has -- which the board printed as "Your max bid" while the budget
    ledger, which guards the same expression, showed 0 for that seat on the same screen."""
    s = _session()
    for i in range(s.league.roster_size):
        s.record_purchase(_IDS[i], 1, 1)
    assert s.open_slots(1) == 0
    assert s.budget(1) > 0  # money left over, so the unguarded formula would be positive
    assert s.feasible_max(1) == 0


def test_my_roster_view_keeps_a_player_with_no_allocatable_slot() -> None:
    """`record_purchase` records what the room did without enforcing positional eligibility,
    so a roster can hold a player `allocate_roster_slots` has no slot for. He must still
    appear -- `spent` counts his price either way, and dropping him loses a bought player."""
    s = _session()
    qbs = [g for g in _IDS if s._position_by_id[g] is Position.QB]
    # QB1 + 4 bench = 5 placeable (QB is not FLEX-eligible), so the sixth has no slot.
    bought = qbs[:6]
    for gid in bought:
        s.record_purchase(gid, s.my_seat, 5)
    from projections.draft.roster_eligibility import allocate_roster_slots

    placements, _, _ = allocate_roster_slots(
        ((g, Position.QB) for g in bought), s.league.roster_slots
    )
    assert len(placements) < len(bought), "fixture no longer overflows; the test would be vacuous"
    view = s.my_roster_view()
    assert set(view.filled["gsis_id"]) == set(bought)
    assert view.spent == 30
    assert view.filled["price"].sum() == view.spent


@pytest.mark.parametrize(
    ("gsis", "seat", "price", "match"),
    [
        ("00-9999999", 1, 5, "not in the draft pool"),
        (_IDS[0], 99, 5, "seat must be in"),
        (_IDS[0], 1, 0, "at least"),
        (_IDS[0], 1, 500, "unable to fill its roster"),
    ],
)
def test_record_purchase_rejects_illegal_awards(
    gsis: str, seat: int, price: int, match: str
) -> None:
    s = _session()
    with pytest.raises(ValueError, match=match):
        s.record_purchase(gsis, seat, price)
    assert s.purchases == []


def test_record_purchase_rejects_a_second_sale_and_a_full_roster() -> None:
    s = _session()
    s.record_purchase(_IDS[0], 2, 10)
    with pytest.raises(ValueError, match="already drafted"):
        s.record_purchase(_IDS[0], 3, 10)
    full = _session()
    for i in range(full.league.roster_size):
        full.record_purchase(_IDS[i], 3, 1)
    with pytest.raises(ValueError, match="full roster"):
        full.record_purchase(_IDS[50], 3, 1)


def test_undo_restores_the_prior_state_and_invalidates_the_seat_memo() -> None:
    s = _session()
    s.record_purchase(_IDS[0], 2, 40)
    assert s.budget(2) == 160
    popped = s.undo()
    assert popped is not None and str(popped.gsis_id) == _IDS[0]
    assert s.budget(2) == 200 and s.drafted_ids == frozenset()
    # An undo followed by a DIFFERENT award of the same length must not serve the stale memo.
    s.record_purchase(_IDS[1], 3, 55)
    assert s.budget(2) == 200
    assert s.budget(3) == 145
    assert s.roster_ids(3) == [_IDS[1]]
    assert s.undo() is not None and s.undo() is None


def test_my_seat_must_be_in_range() -> None:
    with pytest.raises(ValueError, match="my_seat must be"):
        _session(my_seat=99)
    with pytest.raises(ValueError, match="nomination_mode"):
        _session(nomination_mode="bogus")


# --------------------------------------------------------------------------- bid advice


def test_advice_is_the_models_desire_clamped_to_the_engine_window() -> None:
    rich = _session(strategy=_FixedBid(10_000), strategy_name="fixed")
    a = rich.advise(_IDS[0])
    assert a.desired == 10_000
    assert a.max_bid == rich.feasible_max(1)  # clamped down to solvency, never above
    poor = _session(strategy=_FixedBid(-5), strategy_name="fixed")
    assert poor.advise(_IDS[0]).max_bid == poor.league.min_bid  # floored at min_bid


def test_advice_reports_our_value_and_the_rooms_price_separately() -> None:
    s = _session()
    a = s.advise(_IDS[3])  # the fixture player ESPN loves and we do not
    assert a.market_value > a.fair_value
    model = _session(market="model")
    m = model.advise(_IDS[3])
    assert m.market_value == m.fair_value  # no ESPN anchor -> the room bids our board


def test_advice_passes_on_a_position_my_roster_can_no_longer_take() -> None:
    s = _session()
    qbs = [g for g, p in zip(_IDS, _POSITIONS, strict=True) if p == "QB"]
    # Buy QBs until the roster-discipline gate closes the position.
    for gid in qbs:
        if Position.QB not in s.eligible_positions(1):
            break
        s.record_purchase(gid, 1, 1)
    assert Position.QB not in s.eligible_positions(1)
    a = s.advise(qbs[-1])
    assert not a.eligible and a.max_bid == 0


def test_advise_rejects_an_unknown_or_sold_player() -> None:
    s = _session()
    with pytest.raises(ValueError, match="not in the draft pool"):
        s.advise("00-9999999")
    s.record_purchase(_IDS[0], 2, 5)
    with pytest.raises(ValueError, match="already drafted"):
        s.advise(_IDS[0])


def test_room_ceiling_is_the_best_rival_not_me() -> None:
    s = _session()
    assert s.room_ceiling(Position.RB) == s.feasible_max(2)
    for seat in range(2, s.league.n_teams + 1):  # drain every rival
        s.record_purchase(_IDS[seat * 3], seat, s.feasible_max(seat))
    assert s.room_ceiling(Position.RB) == s.league.min_bid
    assert s.feasible_max(1) > s.room_ceiling(Position.RB)  # I can outbid the whole room


def test_bid_board_filters_sorts_and_caps() -> None:
    s = _session()
    board = s.bid_board(top=10)
    assert len(board) == 10
    assert list(board["max_bid"]) == sorted(board["max_bid"], reverse=True)
    assert (board["edge"] == board["max_bid"] - board["market"]).all()
    rbs = s.bid_board(position=Position.RB, top=5)
    assert set(rbs["position"]) == {"RB"}
    one = s.bid_board(query="Player 7", top=40)
    assert not one.empty and one["full_name"].str.contains("Player 7").all()
    assert s.bid_board(query="nobody at all").empty


def test_bid_board_drops_sold_players() -> None:
    s = _session()
    s.record_purchase(_IDS[0], 2, 30)
    assert _IDS[0] not in set(s.bid_board(top=40)["gsis_id"])


# --------------------------------------------------------------------------- nomination


def test_value_nomination_picks_the_most_valuable_nominable_player() -> None:
    s = _session()
    assert s.suggested_nomination() == _IDS[0]
    s.record_purchase(_IDS[0], 2, 30)
    assert s.suggested_nomination() == _IDS[1]


def test_drain_max_nominates_the_priciest_by_the_rooms_board_not_ours() -> None:
    s = _session(nomination_mode="drain_max")
    assert s.suggested_nomination() == _IDS[3]  # the ESPN darling, not our own #1


def test_drain_off_position_prefers_a_position_i_have_already_filled() -> None:
    s = _session(nomination_mode="drain_off_position")
    rbs = [g for g, p in zip(_IDS, _POSITIONS, strict=True) if p == "RB"]
    for gid in rbs[:4]:  # fill my RB starter requirement
        s.record_purchase(gid, 1, 1)
    sug = s.suggested_nomination()
    assert sug is not None
    assert s._position_by_id[sug] == Position.RB


def test_nomination_board_flags_what_i_want() -> None:
    s = _session()
    board = s.nomination_board(top=6)
    assert len(board) == 6
    assert set(board.columns) >= {"full_name", "value", "market", "max_bid", "room_max", "i_want"}
    assert board["i_want"].dtype == bool


def test_nomination_dries_up_when_the_auction_is_over() -> None:
    s = _session(n_teams=2)  # 2 x 11 = 22 lots, fixture has 96 players
    seat = 1
    for gid in _IDS:
        if s.is_complete:
            break
        if s.open_slots(seat) == 0:
            seat += 1
        if s.suggested_nomination() is None:
            break
        s.record_purchase(s.suggested_nomination() or gid, seat, 1)
    assert s.is_complete
    assert s.suggested_nomination() is None
    assert s.nomination_board().empty


# --------------------------------------------------------------------------- views


def test_roster_view_carries_prices_and_open_starting_slots() -> None:
    s = _session()
    s.record_purchase(_IDS[0], 1, 40)  # RB
    s.record_purchase(_IDS[2], 1, 12)  # QB
    view = s.my_roster_view()
    assert view.spent == 52
    assert list(view.filled["price"]) == [40, 12]
    assert set(view.filled["slot"]) == {"RB", "QB"}
    assert RosterSlot.WR in view.open_slots


def test_budget_table_and_log_report_every_seat_and_lot() -> None:
    s = _session(team_names=("Alden", "", "Will"))
    s.record_purchase(_IDS[0], 3, 40)
    table = s.budget_table()
    assert len(table) == s.league.n_teams
    assert list(table["team"])[:3] == ["Alden", "Team 2", "Will"]
    assert next(iter(table["you"])) == "★"
    assert int(table.loc[table["seat"] == 3, "budget"].iloc[0]) == 160
    log = s.purchase_log()
    assert list(log["player"]) == ["Player 1"]
    assert int(log["over"].iloc[0]) == 40 - int(log["value"].iloc[0])


def test_inflation_rises_when_the_room_keeps_its_money() -> None:
    s = _session()
    start = s.inflation()
    for seat in range(2, s.league.n_teams + 1):  # everyone buys cheap: money left, board thinner
        s.record_purchase(_IDS[seat], seat, 1)
    assert s.inflation() > start


def test_nominating_seat_rotates_and_skips_full_rosters() -> None:
    s = _session()
    assert s.nominating_seat == 1 and s.is_my_nomination
    s.record_purchase(_IDS[0], 5, 10)
    assert s.nominating_seat == 2
    full = _session()
    for i in range(full.league.roster_size):
        full.record_purchase(_IDS[i], 2, 1)
    # 11 lots sold -> rotation points at seat 4 (11 % 8 + 1); seat 2 is full and gets skipped
    assert full.open_slots(2) == 0
    assert full.nominating_seat != 2


# --------------------------------------------------------------------------- persistence


def test_save_load_round_trip(tmp_path: Path) -> None:
    league_path = tmp_path / "l.json"
    league_path.write_text(_league().model_dump_json())
    s = _session(
        my_seat=3,
        nomination_mode="drain_max",
        team_names=("A", "B"),
        league_config_path=league_path,
    )
    s.record_purchase(_IDS[0], 3, 44)
    s.record_purchase(_IDS[1], 2, 11)
    path = tmp_path / "auction.json"
    s.save(path)
    back = LiveAuctionSession.load(path, id_map=_id_map(), pool=_pool())
    assert back.my_seat == 3
    assert back.strategy_name == s.strategy_name
    assert back.nomination_mode == "drain_max"
    assert back.team_names == ("A", "B")
    assert [(str(p.gsis_id), p.seat, p.price) for p in back.purchases] == [
        (_IDS[0], 3, 44),
        (_IDS[1], 2, 11),
    ]
    assert back.budget(3) == 156


def test_load_rejects_an_unknown_bid_model(tmp_path: Path) -> None:
    league_path = tmp_path / "l.json"
    league_path.write_text(_league().model_dump_json())
    s = _session(strategy=BalancedValueBid(), strategy_name="nope", league_config_path=league_path)
    path = tmp_path / "a.json"
    s.save(path)
    with pytest.raises(ValueError, match="unknown bid model"):
        LiveAuctionSession.load(path, id_map=_id_map(), pool=_pool())


def test_forced_lot_mirrors_the_engine_when_the_pool_goes_thin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no open seat can roster any remaining position, the engine forces the top
    undrafted player and lets every seat bid ungated rather than dead-ending
    (`simulation._simulate_to_state`, the `forced` branch). The board must do the same: it
    reported "nothing to nominate" and a $0 max bid while the room was still selling lots.

    The gate is closed directly. Reaching it through a realistic draft needs a pathological
    pool, and the behaviour under test is what happens *once* it is closed.
    """
    s = _session()
    monkeypatch.setattr(type(s), "eligible_positions", lambda self, seat: frozenset())
    assert s._gated_candidates() == []
    assert s.is_forced_lot

    nom = s.suggested_nomination()
    assert nom is not None, "a forced lot must still name a nominee"
    assert nom == s._undrafted_in_value_order()[0], "the engine forces the top undrafted player"
    assert len(s._nomination_candidates()) == 1, "exactly one forced nominee, as in the engine"

    advice = s.advise(nom)
    assert advice.eligible, "the hero is ungated on a forced lot, as every seat is in the engine"
    assert advice.max_bid >= s.league.min_bid


def test_live_view_matches_the_engine_view_field_for_field() -> None:
    """The board's whole premise is that it scores the strategy against *the same*
    `AuctionView` the simulation builds. The two are constructed independently
    (`live._view` vs `simulation._build_view`), so pin them together: a divergence here means
    the board's number stops being the number the measured strategy would have bid."""
    from projections.draft.assistant.auction.simulation import AuctionState, _build_view

    s = _session()
    awards = [(_IDS[0], 1, 40), (_IDS[1], 2, 30), (_IDS[5], 1, 12), (_IDS[9], 3, 7)]
    for gid, seat, price in awards:
        s.record_purchase(gid, seat, price)

    # The engine's equivalent state for the same awards (0-based seats).
    state = AuctionState.initial(s.league)
    for gid, seat, price in awards:
        state.budgets[seat - 1] -= price
        state.rosters[seat - 1].append((gid, s._position_by_id[gid].value, price))
        state.drafted.add(gid)

    mine = _build_view(state, s.my_seat - 1, s.pool, s.engine_dollars, s.league)
    theirs = s._view()
    assert theirs.my_budget == mine.my_budget
    assert theirs.my_open_slots == mine.my_open_slots
    assert theirs.my_positions == mine.my_positions
    assert theirs.drafted == mine.drafted
    assert theirs.budgets_by_seat == mine.budgets_by_seat
    assert list(theirs.my_roster["gsis_id"]) == list(mine.my_roster["gsis_id"])
    assert theirs.baseline_dollars.equals(mine.baseline_dollars)
    # Every field is covered above -- fail loudly if someone adds one.
    assert {f.name for f in dataclasses.fields(AuctionView)} == {
        "my_budget",
        "my_open_slots",
        "my_positions",
        "my_roster",
        "drafted",
        "budgets_by_seat",
        "baseline_dollars",
    }
