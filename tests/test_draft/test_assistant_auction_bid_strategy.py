from collections import Counter

import pandas as pd

from projections.draft.assistant.auction.bid_strategy import (
    AnchorBudgetBid,
    AuctionView,
    InflationBid,
    MarginalValueBid,
    OverbidValueBid,
    PatientValueBid,
    StaticDollarBid,
    VorpShareBid,
    _undrafted,
    _vorp_threshold,
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


# ---------------------------------------------------------------------------
# VORP-bearing fixtures and tests for AnchorBudgetBid, OverbidValueBid, VorpShareBid
# ---------------------------------------------------------------------------


def _vpool() -> pd.DataFrame:
    # vorp strictly descending by row: 120,110,90,20,10,5
    return pd.DataFrame(
        {
            "gsis_id": pd.array(
                [
                    "00-0000001",
                    "00-0000002",
                    "00-0000003",
                    "00-0000004",
                    "00-0000005",
                    "00-0000006",
                ],
                dtype=_PYARROW_STR,
            ),
            "position": pd.array(["RB", "WR", "QB", "RB", "WR", "TE"], dtype=_PYARROW_STR),
            "season_mean_fpts": [250.0, 240.0, 280.0, 120.0, 110.0, 100.0],
            "vorp": [120.0, 110.0, 90.0, 20.0, 10.0, 5.0],
        }
    )


def _vbaseline() -> pd.DataFrame:
    return pd.DataFrame(
        {"in_pool": [True] * 6, "auction_dollars": [30, 28, 25, 5, 3, 2]},
        index=pd.Index(
            ["00-0000001", "00-0000002", "00-0000003", "00-0000004", "00-0000005", "00-0000006"],
            name="gsis_id",
        ),
    )


def _vconfig() -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=2,
        budget=100,
        min_bid=1,
        roster_slots={
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.QB: 1,
            RosterSlot.BENCH: 2,
        },
        ruleset=Ruleset.espn_ppr(),
    )


def _aview(
    pool: pd.DataFrame,
    *,
    my_ids: tuple[str, ...] = (),
    budget: int = 100,
    open_slots: int = 8,
    drafted: tuple[str, ...] = (),
) -> AuctionView:
    my_roster = pool[pool["gsis_id"].isin(list(my_ids))]
    return AuctionView(
        my_budget=budget,
        my_open_slots=open_slots,
        my_positions=Counter(),
        my_roster=my_roster,
        drafted=frozenset(drafted),
        budgets_by_seat=(budget, budget),
        baseline_dollars=_vbaseline(),
    )


def test_vorp_threshold_kth_highest() -> None:
    pool = _vpool()
    assert _vorp_threshold(pool, 1) == 120.0
    assert _vorp_threshold(pool, 3) == 90.0
    assert _vorp_threshold(pool, 4) == 20.0


def test_vorp_threshold_pool_smaller_than_k_returns_min() -> None:
    assert _vorp_threshold(_vpool(), 8) == 5.0  # len 6 <= 8 -> pool min (all anchor-grade)


def test_vorp_threshold_nonpositive_k_is_inf() -> None:
    assert _vorp_threshold(_vpool(), 0) == float("inf")


def test_undrafted_filters_drafted_ids() -> None:
    pool = _vpool()
    assert len(_undrafted(pool, frozenset())) == 6
    left = _undrafted(pool, frozenset({"00-0000001"}))
    assert len(left) == 5 and "00-0000001" not in {str(g) for g in left["gsis_id"]}


def test_anchor_bids_above_market_for_a_top_vorp_player() -> None:
    # n_anchors=2, n_teams=2 -> league_anchor_count 4 -> threshold = 4th vorp = 20.
    # Empty roster, budget 100, open 8: reserve=1*(8-2)=6, cap=(100-6)/2=47.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    bid = AnchorBudgetBid(n_anchors=2).max_bid(view, pool.iloc[0], pool, _vconfig())
    assert bid == 47
    assert bid > 30  # overbids the $30 market value to actually win the anchor


def test_anchor_bids_min_for_a_scrub() -> None:
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    bid = AnchorBudgetBid(n_anchors=2).max_bid(view, pool.iloc[4], pool, _vconfig())  # vorp 10 < 20
    assert bid == _vconfig().min_bid


def test_anchor_switches_to_scrubs_once_anchors_held() -> None:
    # Hero already holds 2 anchors (vorp 120,110 >= 20); n_anchors=2 -> anchors_remaining 0.
    pool = _vpool()
    view = _aview(
        pool,
        my_ids=("00-0000001", "00-0000002"),
        budget=60,
        open_slots=6,
        drafted=("00-0000001", "00-0000002"),
    )
    bid = AnchorBudgetBid(n_anchors=2).max_bid(view, pool.iloc[2], pool, _vconfig())  # anchor-grade
    assert bid == _vconfig().min_bid


def test_overbid_pays_up_for_studs_value_for_others() -> None:
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    cfg = _vconfig()
    strat = OverbidValueBid(k=1.5, stud_count=3)  # threshold = 3rd vorp = 90
    assert strat.max_bid(view, pool.iloc[0], pool, cfg) == round(30 * 1.5)  # stud (vorp 120) -> 45
    assert strat.max_bid(view, pool.iloc[3], pool, cfg) == 5  # non-stud (vorp 20) -> value 5


def test_overbid_default_stud_count_unset() -> None:
    assert OverbidValueBid().stud_count is None  # resolved to 3*n_teams at call time


def test_vorpshare_concentrates_on_top_targets() -> None:
    # Empty roster, open 8 -> targets = all 6 undrafted. denom = 120+110+90+20+10+5 = 355.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    cfg = _vconfig()
    assert VorpShareBid().max_bid(view, pool.iloc[0], pool, cfg) == round(100 * 120 / 355)  # 34
    assert VorpShareBid().max_bid(view, pool.iloc[5], pool, cfg) == cfg.min_bid  # vorp 5 -> ~1


def test_vorpshare_off_target_player_bids_min() -> None:
    # open 2 -> targets = top-2 vorp (players 1,2). Player 3 (vorp 90) is off-target.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=2)
    assert VorpShareBid().max_bid(view, pool.iloc[2], pool, _vconfig()) == _vconfig().min_bid


def test_vorpshare_zero_target_vorp_bids_min() -> None:
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000007", "00-0000008"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [80.0, 70.0],
            "vorp": [0.0, -5.0],
        }
    )
    view = AuctionView(
        my_budget=100,
        my_open_slots=2,
        my_positions=Counter(),
        my_roster=pool.iloc[:0],
        drafted=frozenset(),
        budgets_by_seat=(100, 100),
        baseline_dollars=pd.DataFrame(
            {"in_pool": [True, True], "auction_dollars": [1, 1]},
            index=pd.Index(["00-0000007", "00-0000008"], name="gsis_id"),
        ),
    )
    assert VorpShareBid().max_bid(view, pool.iloc[0], pool, _vconfig()) == _vconfig().min_bid


# ---------------------------------------------------------------------------
# PatientValueBid tests
# ---------------------------------------------------------------------------


def test_patient_hero_holds_on_a_stud() -> None:
    # _vpool vorps: 120,110,90,20,10,5 (6 players). stud_frac 0.10 -> round(0.6)=1 -> top-1 cutoff
    # = _vorp_threshold(pool,1)=120; vorp 120 is a stud -> min_bid.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    assert PatientValueBid().max_bid(view, pool.iloc[0], pool, _vconfig()) == _vconfig().min_bid


def test_patient_hero_pays_premium_for_midtier_with_reserve() -> None:
    # scrub_frac 0.50 -> (1-0.50)*6=3 -> scrub cutoff = _vorp_threshold(pool,3)=90; stud cutoff 120.
    # vorp 110 (player 2) is in (90,120) -> mid. auction_dollars for it = 28 -> round(28*1.35)=38.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    assert PatientValueBid().max_bid(view, pool.iloc[1], pool, _vconfig()) == round(28 * 1.35)


def test_patient_hero_midtier_without_reserve_bids_min() -> None:
    pool = _vpool()
    view = _aview(pool, budget=8, open_slots=8)  # reserve = 8 - 1*7 = 1 < the premium bid
    assert PatientValueBid().max_bid(view, pool.iloc[1], pool, _vconfig()) == _vconfig().min_bid


def test_patient_hero_scrub_bids_min() -> None:
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    # vorp 5 (player 6) is below the scrub cutoff (90) -> scrub -> min_bid.
    assert PatientValueBid().max_bid(view, pool.iloc[5], pool, _vconfig()) == _vconfig().min_bid
