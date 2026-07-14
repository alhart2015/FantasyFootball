from collections import Counter

import pandas as pd
import pytest

from projections.draft.assistant.auction.bid_strategy import (
    URGENCY_GAIN,
    AnchorBudgetBid,
    AuctionView,
    BalancedValueBid,
    InflationBid,
    MarginalValueBid,
    OverbidValueBid,
    PatientValueBid,
    StaticDollarBid,
    StudsAndDepthBid,
    VorpShareBid,
    _budget_urgency,
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
    cfg = _config()
    bid = MarginalValueBid().max_bid(view, pool.iloc[2], pool, cfg)
    # urgency feature: late-draft (open 1 of 3, surplus 49) the zero-lift base min_bid is scaled up
    assert bid == round(cfg.min_bid * _budget_urgency(view, cfg))
    assert bid > cfg.min_bid


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
    cfg = _vconfig()
    bid = AnchorBudgetBid(n_anchors=2).max_bid(view, pool.iloc[2], pool, cfg)  # anchor-grade
    # urgency feature: anchors held (anchors_remaining 0) -> base min_bid, scaled up late (open 6/8)
    assert bid == round(cfg.min_bid * _budget_urgency(view, cfg))


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
    cfg = _vconfig()
    bid = VorpShareBid().max_bid(view, pool.iloc[2], pool, cfg)
    # urgency feature: off-target base min_bid scaled up late (open 2/8, surplus 98)
    assert bid == round(cfg.min_bid * _budget_urgency(view, cfg))


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
    cfg = _vconfig()
    bid = VorpShareBid().max_bid(view, pool.iloc[0], pool, cfg)
    # urgency feature: zero-target-vorp base min_bid scaled up late (open 2/8, surplus 98)
    assert bid == round(cfg.min_bid * _budget_urgency(view, cfg))


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


# ---------------------------------------------------------------------------
# _budget_urgency (shared late-draft deployment factor)
# ---------------------------------------------------------------------------


def test_budget_urgency_is_one_at_draft_start() -> None:
    # progress == 0 (my_open_slots == roster_size) -> exactly 1.0, regardless of surplus.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)  # roster_size 8 -> progress 0
    assert _budget_urgency(view, _vconfig()) == 1.0


def test_budget_urgency_is_one_when_broke() -> None:
    # surplus = budget - min_bid*open_slots <= 0 -> 1.0 (don't escalate what you can't afford).
    pool = _vpool()
    view = _aview(pool, budget=5, open_slots=8)  # 5 - 1*8 = -3 <= 0
    assert _budget_urgency(view, _vconfig()) == 1.0


def test_budget_urgency_exceeds_one_for_overfunded_partial_roster() -> None:
    pool = _vpool()
    view = _aview(pool, budget=90, open_slots=4)  # surplus 86 > 0, progress 0.5
    assert _budget_urgency(view, _vconfig()) > 1.0


def test_budget_urgency_increases_with_progress() -> None:
    pool = _vpool()
    cfg = _vconfig()
    fewer_slots = _budget_urgency(_aview(pool, budget=100, open_slots=2), cfg)  # progress 0.75
    more_slots = _budget_urgency(_aview(pool, budget=100, open_slots=6), cfg)  # progress 0.25
    assert fewer_slots > more_slots


def test_budget_urgency_increases_with_surplus() -> None:
    pool = _vpool()
    cfg = _vconfig()
    rich = _budget_urgency(_aview(pool, budget=100, open_slots=4), cfg)  # ratio 96/100
    poor = _budget_urgency(_aview(pool, budget=20, open_slots=4), cfg)  # ratio 16/20
    assert rich > poor


def test_budget_urgency_is_bounded_below_one_plus_gain() -> None:
    pool = _vpool()
    cfg = _vconfig()
    extreme = _budget_urgency(_aview(pool, budget=10_000, open_slots=1), cfg)
    assert 1.0 <= extreme < 1.0 + URGENCY_GAIN


# ---------------------------------------------------------------------------
# StudsAndDepthBid (the "good bot as a hero")
# ---------------------------------------------------------------------------


def test_studs_premium_for_a_stud() -> None:
    # vorp 120 >= stud_cut 120 -> auction_dollars 30 * (1 + 0.2). Empty roster -> urgency 1.0.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    bid = StudsAndDepthBid().max_bid(view, pool.iloc[0], pool, _vconfig())
    assert bid == round(30 * (1.0 + 0.2))  # 36


def test_studs_fair_value_for_midtier() -> None:
    # vorp 110 in (10, 120) -> fair value = auction_dollars 28, no $1-dump. Empty -> urgency 1.0.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    bid = StudsAndDepthBid().max_bid(view, pool.iloc[1], pool, _vconfig())
    assert bid == 28


def test_studs_min_bid_for_a_scrub() -> None:
    # vorp 5 < scrub_cut 10 -> min_bid. Empty roster -> urgency 1.0.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    bid = StudsAndDepthBid().max_bid(view, pool.iloc[5], pool, _vconfig())
    assert bid == _vconfig().min_bid


def test_studs_depth_scales_up_under_overfunded_partial_roster() -> None:
    # Same mid-tier player, but a partial overfunded view (open 5/8, surplus 85) -> urgency > 1.
    pool = _vpool()
    cfg = _vconfig()
    view = _aview(pool, budget=90, open_slots=5)
    bid = StudsAndDepthBid().max_bid(view, pool.iloc[1], pool, cfg)
    assert bid == round(28 * _budget_urgency(view, cfg))
    assert bid > 28  # deploys the surplus rather than leaving it idle


def test_studs_depth_tiny_pool_has_no_studs() -> None:
    # round(stud_frac * 1) == 0 -> _vorp_threshold(pool, 0) == +inf -> nothing clears the stud bar.
    one = _vpool().iloc[[0]].reset_index(drop=True)
    view = _aview(one, budget=100, open_slots=8)
    # vorp 120 is NOT a stud here (stud_cut +inf); scrub_cut = pool min 120 -> v<120 false -> mid.
    bid = StudsAndDepthBid().max_bid(view, one.iloc[0], one, _vconfig())
    assert bid == 30  # fair value (auction_dollars), urgency 1.0


def test_studs_depth_satisfies_protocol() -> None:
    from projections.draft.assistant.auction.bid_strategy import AuctionBidStrategy

    assert isinstance(StudsAndDepthBid(), AuctionBidStrategy)


# ---------------------------------------------------------------------------
# BalancedValueBid (balanced-breadth hero: premium over fair, capped by pace)
# ---------------------------------------------------------------------------


def test_balanced_premium_wins_contested_value() -> None:
    # mid-tier under the cap -> bid a premium over fair (out-bids a fair-value bidder)
    pool = _pool()
    baseline = _baseline([True, True, False, False], [20, 40, 0, 0])
    view = _view(pool.iloc[:0], budget=100, drafted=set(), baseline=baseline)
    bid = BalancedValueBid(premium=0.15, pace=2.0).max_bid(view, pool.iloc[0], pool, _config())
    assert bid == 23  # round(20 * 1.15); cap = 2*(100/3) = 66.7 does not bind
    assert bid > 20  # strictly above fair value


def test_balanced_cap_forces_spread_on_studs() -> None:
    # a stud whose premium'd value exceeds the pace cap -> bid the cap (< fair)
    pool = _pool()
    baseline = _baseline([True, True, False, False], [80, 40, 0, 0])
    view = _view(pool.iloc[:0], budget=100, drafted=set(), baseline=baseline)
    bid = BalancedValueBid(premium=0.15, pace=2.0).max_bid(view, pool.iloc[0], pool, _config())
    assert bid == 67  # round(2 * 100/3) = round(66.67); 80*1.15=92 does not win
    assert bid < 80  # strictly below fair value (capped)


def test_balanced_cap_tracks_remaining_budget() -> None:
    pool = _pool()
    baseline = _baseline([True, True, False, False], [80, 40, 0, 0])
    strat = BalancedValueBid(premium=0.15, pace=2.0)
    rich = strat.max_bid(
        _view(pool.iloc[:0], budget=100, drafted=set(), baseline=baseline),
        pool.iloc[0],
        pool,
        _config(),
    )
    poor = strat.max_bid(
        _view(pool.iloc[:0], budget=30, drafted=set(), baseline=baseline),
        pool.iloc[0],
        pool,
        _config(),
    )
    assert poor < rich  # cap shrinks with budget: 2*(30/3)=20 vs 2*(100/3)=67


def test_balanced_does_not_apply_urgency() -> None:
    # partial roster + idle cash => _budget_urgency > 1; the bid must NOT be inflated by it.
    pool = _pool()
    baseline = _baseline([True, True, False, False], [20, 40, 0, 0])
    view = _view(
        pool.iloc[[2]], budget=100, drafted={"00-0000003"}, baseline=baseline
    )  # 1 held -> 2 open
    config = _config()
    urgency = _budget_urgency(view, config)
    assert urgency > 1.5  # sanity: this state carries a real urgency ramp
    bid = BalancedValueBid(premium=0.15, pace=2.0).max_bid(view, pool.iloc[0], pool, config)
    assert bid == 23  # round(20 * 1.15); cap = 2*(100/2) = 100 does not bind
    assert bid < round(23 * urgency)  # NOT multiplied by the urgency ramp


def test_balanced_is_deterministic() -> None:
    pool = _pool()
    baseline = _baseline([True, True, False, False], [20, 40, 0, 0])
    view = _view(pool.iloc[:0], budget=100, drafted=set(), baseline=baseline)
    strat, config = BalancedValueBid(), _config()
    assert strat.max_bid(view, pool.iloc[0], pool, config) == strat.max_bid(
        view, pool.iloc[0], pool, config
    )


def test_balanced_rejects_bad_tuning() -> None:
    with pytest.raises(ValueError):
        BalancedValueBid(premium=-0.1)
    with pytest.raises(ValueError):
        BalancedValueBid(pace=0.0)
    with pytest.raises(ValueError):
        BalancedValueBid(pace=float("inf"))
    with pytest.raises(ValueError):
        BalancedValueBid(premium=float("nan"))


def test_balanced_default_is_retuned_premium_one() -> None:
    # Retuned 2026-07-14 (cap-vs-premium sweep): default premium=1.0 bids up to the low pace cap on
    # the mid-tier so the budget spreads (the winning behavior in inflated markets); pace stays 2.0
    # (raising the cap backfires). See reports/auction_tournament_validation_2026.md.
    strat = BalancedValueBid()
    assert (strat.premium, strat.pace) == (1.0, 2.0)
    pool = _pool()
    baseline = _baseline([True, True, False, False], [20, 40, 0, 0])
    view = _view(pool.iloc[:0], budget=100, drafted=set(), baseline=baseline)
    # fair=20 -> 20*(1+1.0)=40 (premium doubles the value bid); cap=2*(100/3)=66.7 does not bind
    assert strat.max_bid(view, pool.iloc[0], pool, _config()) == 40
