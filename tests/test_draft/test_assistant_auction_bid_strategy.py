from collections import Counter

import pandas as pd

from projections.draft.assistant.auction.bid_strategy import (
    AuctionView,
    InflationBid,
    MarginalValueBid,
    StaticDollarBid,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset


def _config() -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=2,
        budget=100,
        min_bid=1,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": pd.array(
                ["00-0000001", "00-0000002", "00-0000003", "00-0000004"], dtype=_PYARROW_STR
            ),
            "position": pd.array(["RB", "WR", "RB", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [200.0, 180.0, 50.0, 40.0],
        }
    )


def _baseline(in_pool: list[bool], dollars: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "in_pool": in_pool,
            "auction_dollars": dollars,
        },
        index=pd.Index(["00-0000001", "00-0000002", "00-0000003", "00-0000004"], name="gsis_id"),
    )


def _view(
    my_roster: pd.DataFrame, *, budget: int, drafted: set[str], baseline: pd.DataFrame
) -> AuctionView:
    return AuctionView(
        my_budget=budget,
        my_open_slots=3 - len(my_roster),
        my_positions=Counter(my_roster["position"].astype(str)),
        my_roster=my_roster,
        drafted=frozenset(drafted),
        budgets_by_seat=(budget, budget),
        baseline_dollars=baseline,
    )


def test_static_bids_the_baseline_dollar() -> None:
    pool = _pool()
    baseline = _baseline([True, True, False, False], [60, 40, 0, 0])
    view = _view(pool.iloc[:0], budget=100, drafted=set(), baseline=baseline)
    bid = StaticDollarBid().max_bid(view, pool.iloc[0], pool, _config())
    assert bid == 60


def test_inflation_below_one_when_room_overspent() -> None:
    pool = _pool()
    baseline = _baseline([True, True, False, False], [60, 40, 0, 0])
    # Both seats have spent down to $10 each but nothing is drafted yet (contrived overspend):
    view = AuctionView(
        my_budget=10,
        my_open_slots=3,
        my_positions=Counter(),
        my_roster=pool.iloc[:0],
        drafted=frozenset(),
        budgets_by_seat=(10, 10),
        baseline_dollars=baseline,
    )
    bid = InflationBid().max_bid(view, pool.iloc[0], pool, _config())
    assert bid < 60  # inflation < 1 -> below the static dollar


def test_inflation_falls_back_to_one_when_no_surplus_value() -> None:
    pool = _pool()
    # Only out-of-pool players left undrafted -> remaining_surplus_value == 0 -> factor 1.0
    baseline = _baseline([True, True, False, False], [60, 40, 0, 0])
    view = AuctionView(
        my_budget=100,
        my_open_slots=3,
        my_positions=Counter(),
        my_roster=pool.iloc[:0],
        drafted=frozenset({"00-0000001", "00-0000002"}),  # both in-pool already drafted
        budgets_by_seat=(100, 100),
        baseline_dollars=baseline,
    )
    # pricing an out-of-pool player: base==0 -> min_bid + (0-1)*1.0 = 0 -> still returns an int
    bid = InflationBid().max_bid(view, pool.iloc[2], pool, _config())
    assert isinstance(bid, int)


def test_marginal_zero_lift_player_bids_min_bid() -> None:
    pool = _pool()
    baseline = _baseline([True, True, True, True], [60, 40, 5, 5])
    # Hero already holds the best RB and WR (starters full); a worse RB adds 0 lineup lift.
    my_roster = pool.iloc[[0, 1]]
    view = _view(my_roster, budget=50, drafted={"00-0000001", "00-0000002"}, baseline=baseline)
    bid = MarginalValueBid().max_bid(view, pool.iloc[2], pool, _config())
    assert bid == _config().min_bid


def test_marginal_improving_player_bids_above_min_bid() -> None:
    pool = _pool()
    baseline = _baseline([True, True, True, True], [60, 40, 5, 5])
    # Empty roster: the best RB cracks the lineup -> lift > 0 -> bid > min_bid.
    view = _view(pool.iloc[:0], budget=100, drafted=set(), baseline=baseline)
    bid = MarginalValueBid().max_bid(view, pool.iloc[0], pool, _config())
    assert bid > _config().min_bid
