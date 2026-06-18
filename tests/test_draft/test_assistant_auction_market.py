import numpy as np
import pandas as pd

from projections.draft.assistant.auction.market import SeatView, bot_max_bid, resolve_bids
from projections.draft.league_config import LeagueConfig
from projections.schemas import Position, RosterSlot, Ruleset


def _config(min_bid: int = 1) -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=2,
        budget=100,
        min_bid=min_bid,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def _baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {"in_pool": [True], "auction_dollars": [40]},
        index=pd.Index(["00-0000001"], name="gsis_id"),
    )


def _player() -> pd.Series:
    return pd.Series({"gsis_id": "00-0000001", "position": "RB", "season_mean_fpts": 200.0})


def test_bot_centers_on_baseline_with_zero_jitter() -> None:
    bid = bot_max_bid(
        SeatView(open_slots=3),
        _player(),
        _baseline(),
        _config(),
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert bid == 40


def test_bot_floors_at_min_bid() -> None:
    base = pd.DataFrame(
        {"in_pool": [False], "auction_dollars": [0]},
        index=pd.Index(["00-0000001"], name="gsis_id"),
    )
    bid = bot_max_bid(
        SeatView(open_slots=3),
        _player(),
        base,
        _config(min_bid=2),
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert bid == 2


def test_full_seat_abstains() -> None:
    bid = bot_max_bid(
        SeatView(open_slots=0),
        _player(),
        _baseline(),
        _config(),
        np.random.default_rng(0),
        price_jitter=0.5,
    )
    assert bid == 0


def test_resolve_second_price_plus_min_bid() -> None:
    winner, price = resolve_bids({0: 40, 1: 25, 2: 10}, min_bid=1)
    assert winner == 0
    assert price == 26  # second-highest (25) + min_bid (1)


def test_resolve_caps_at_winner_max() -> None:
    winner, price = resolve_bids({0: 5, 1: 4}, min_bid=3)
    assert winner == 0
    assert price == min(5, 4 + 3)  # == 5, never above the winner's own ceiling


def test_resolve_lone_bidder_pays_min_bid() -> None:
    assert resolve_bids({2: 80}, min_bid=1) == (2, 1)


def test_resolve_ties_break_on_seat_index() -> None:
    winner, _ = resolve_bids({3: 20, 1: 20}, min_bid=1)
    assert winner == 1


def test_bot_abstains_when_position_not_eligible() -> None:
    bid = bot_max_bid(
        SeatView(open_slots=3, eligible_positions=frozenset({Position.WR})),  # RB not eligible
        _player(),  # position "RB"
        _baseline(),
        _config(),
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert bid == 0


def test_bot_bids_when_position_eligible() -> None:
    bid = bot_max_bid(
        SeatView(open_slots=3, eligible_positions=frozenset({Position.RB})),
        _player(),  # position "RB"
        _baseline(),
        _config(),
        np.random.default_rng(0),
        price_jitter=0.0,
    )
    assert bid == 40  # baseline, unchanged when eligible
