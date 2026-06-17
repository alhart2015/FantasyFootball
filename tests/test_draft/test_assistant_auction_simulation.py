import warnings

import numpy as np
import pandas as pd
import pytest

from projections.draft.assistant.auction.bid_strategy import AuctionView, StaticDollarBid
from projections.draft.assistant.auction.simulation import (
    AuctionState,
    _build_view,
    _feasible_max,
    _simulate_to_state,
    simulate_auction,
    validate_auction_inputs,
)
from projections.draft.auction import generate_auction_values
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset


class _MaxBidStub:
    """A hero that always desires an astronomical bid; the engine clamps it to feasible_max."""

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        return 10**9


def _config(
    n_teams: int = 4,
    budget: int = 100,
    min_bid: int = 1,
    roster_slots: dict[RosterSlot, int] | None = None,
) -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=n_teams,
        budget=budget,
        min_bid=min_bid,
        roster_slots=roster_slots or {RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def _pool(n: int = 40) -> pd.DataFrame:
    # per-position digit keeps synthetic gsis within the canonical \d{2}-\d{7} regex
    pos = ["RB" if i % 2 else "WR" for i in range(n)]
    prefix = {"RB": 2, "WR": 3}
    gsis = [f"00-{prefix[pos[i]]}{i:06d}" for i in range(n)]
    return pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "position": pd.array(pos, dtype=_PYARROW_STR),
            "season_mean_fpts": [float(300 - i) for i in range(n)],
            "vorp": [float(150 - i) for i in range(n)],
            "replacement_fpts": [100.0] * n,
        }
    )


def _baseline(pool: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
    return generate_auction_values(pool, config)


def test_validate_rejects_thin_pool() -> None:
    cfg = _config(n_teams=4)  # need 4*3 = 12
    with pytest.raises(ValueError, match="need >= 12"):
        validate_auction_inputs(_pool(8), cfg)


def test_validate_rejects_insolvent_budget() -> None:
    cfg = _config(n_teams=4, budget=2, min_bid=1)  # roster_size 3 > budget 2
    with pytest.raises(ValueError, match="can't afford min_bid"):
        validate_auction_inputs(_pool(40), cfg)


def test_returns_full_league_each_seat_full() -> None:
    cfg = _config(n_teams=4)
    pool = _pool(40)
    league = simulate_auction(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=_baseline(pool, cfg),
        price_jitter=0.1,
        rng=np.random.default_rng(0),
    )
    assert set(league) == {1, 2, 3, 4}
    assert all(len(r) == cfg.roster_size for r in league.values())
    all_ids = [g for r in league.values() for g in r]
    assert len(all_ids) == len(set(all_ids))  # no player drafted twice


def test_determinism_same_seed_same_league() -> None:
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    a = simulate_auction(
        StaticDollarBid(),
        2,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.2,
        rng=np.random.default_rng(7),
    )
    b = simulate_auction(
        StaticDollarBid(),
        2,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.2,
        rng=np.random.default_rng(7),
    )
    assert a == b


def test_returned_dict_key_matches_internal_zero_based_state() -> None:
    # Off-by-one guard (dict side): for EVERY seat, the 1-based league key k holds exactly the
    # ids stored at the 0-based AuctionState.rosters[k-1]. A swapped conversion would diverge.
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    my_seat = 3
    state = _simulate_to_state(
        StaticDollarBid(),
        my_seat,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        rng=np.random.default_rng(5),
    )
    league = simulate_auction(
        StaticDollarBid(),
        my_seat,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        rng=np.random.default_rng(5),
    )
    for k, ids in league.items():
        assert ids == [g for (g, _p, _pr) in state.rosters[k - 1]]


def test_build_view_reads_the_hero_zero_based_seat() -> None:
    # Off-by-one guard (hero side): _build_view(hero0=my_seat-1) reads the 0-based state seat.
    cfg = _config(n_teams=4)  # roster_size 3
    pool = _pool(40)
    bd = _baseline(pool, cfg).set_index("gsis_id")
    state = AuctionState.initial(cfg)
    rb = str(pool.iloc[0]["gsis_id"])
    state.rosters[2] = [(rb, "RB", 10)]  # seat index 2  ==  my_seat 3 - 1
    state.budgets[2] = 90
    view = _build_view(state, 2, pool, bd, cfg)
    assert view.my_budget == 90
    assert list(view.my_roster["gsis_id"].astype(str)) == [rb]


def test_feasible_max_reserves_min_bid_for_remaining_slots() -> None:
    cfg = _config(n_teams=4, min_bid=1)  # roster_size 3, all slots open, budget 100
    state = AuctionState.initial(cfg)
    # feasible_max = budget - min_bid*(open_slots-1) = 100 - 1*2 = 98
    assert _feasible_max(state, 0, cfg.roster_size, cfg.min_bid) == 98


def test_feasible_max_one_slot_left_is_whole_budget() -> None:
    cfg = _config(n_teams=4)  # roster_size 3
    state = AuctionState.initial(cfg)
    state.rosters[0] = [("00-2000000", "RB", 10), ("00-3000001", "WR", 5)]  # 2 filled -> 1 open
    state.budgets[0] = 85
    assert _feasible_max(state, 0, cfg.roster_size, cfg.min_bid) == 85


def test_feasible_max_at_reserve_floor_equals_min_bid() -> None:
    cfg = _config(n_teams=4, budget=100, min_bid=1)  # roster_size 3
    state = AuctionState.initial(cfg)
    state.budgets[0] = 3  # budget == min_bid * open_slots(3) -> can only bid min_bid
    assert _feasible_max(state, 0, cfg.roster_size, cfg.min_bid) == 1


def test_over_budget_desired_is_clamped_no_overspend() -> None:
    # A max-desiring hero never overspends: every seat's budget stays >= 0 and the hero still
    # fills a full roster (the [min_bid, feasible_max] clamp held on every win).
    cfg = _config(n_teams=4)
    pool = _pool(40)
    state = _simulate_to_state(
        _MaxBidStub(),
        1,
        pool,
        cfg,
        baseline_dollars=_baseline(pool, cfg),
        price_jitter=0.0,
        rng=np.random.default_rng(0),
    )
    assert all(b >= 0 for b in state.budgets)
    assert len(state.rosters[0]) == cfg.roster_size


def test_solvency_holds_no_negative_budget_path() -> None:
    # A tight budget where every seat must reserve min_bid for remaining slots still completes.
    cfg = _config(n_teams=4, budget=3, min_bid=1)  # budget == min_bid*roster_size, endgame all $1
    pool = _pool(40)
    league = simulate_auction(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=_baseline(pool, cfg),
        price_jitter=0.0,
        rng=np.random.default_rng(0),
    )
    assert all(len(r) == cfg.roster_size for r in league.values())


def test_total_spend_within_budget() -> None:
    # Conservation (spec §4): per-seat spend <= budget, so total spend <= total_budget.
    cfg = _config(n_teams=4)
    pool = _pool(40)
    state = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=_baseline(pool, cfg),
        price_jitter=0.2,
        rng=np.random.default_rng(3),
    )
    per_seat = [sum(price for (_g, _p, price) in seat) for seat in state.rosters]
    assert all(spend <= cfg.budget for spend in per_seat)
    assert sum(per_seat) <= cfg.total_budget


class _MinBidStub:
    """Hero that always bids the minimum — loses every contested player to the bots."""

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        return config.min_bid


def _bd(ids: list[str], dollars: list[int]) -> pd.DataFrame:
    """Hand-built baseline_dollars (gsis_id + auction_dollars) — bypasses generate_auction_values,
    so a deliberately thin/skewed pool isn't rejected by its bench-fill check. The engine only reads
    `auction_dollars` off this frame (set_index on gsis_id)."""
    return pd.DataFrame({"gsis_id": pd.array(ids, dtype=_PYARROW_STR), "auction_dollars": dollars})


def _thin_pool(ids: list[str], positions: list[str]) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame(
        {
            "gsis_id": pd.array(ids, dtype=_PYARROW_STR),
            "position": pd.array(positions, dtype=_PYARROW_STR),
            "season_mean_fpts": [float(100 - i) for i in range(n)],
            "vorp": [float(50 - i) for i in range(n)],
            "replacement_fpts": [100.0] * n,
        }
    )


# 5 RB (ids start 00-2…) + 1 WR (00-3…); RBs carry the top baseline so the WR is nominated last.
_RB5_WR1 = ["00-2000000", "00-2000001", "00-2000002", "00-2000003", "00-2000004", "00-3000000"]
_RB5_WR1_POS = ["RB", "RB", "RB", "RB", "RB", "WR"]


def test_every_bot_roster_is_startable() -> None:
    # n_teams=2, {RB:1, WR:1, BENCH:1}. A min-bidding hero (seat index 1, via my_seat=2) loses
    # every RB, so the UNGATED bot (seat index 0) hoards 3 RB and strands its WR starter -> RED
    # before Task 5. After Task 5 the gate forces the bot to reserve a WR -> GREEN.
    cfg = _config(n_teams=2, roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1})
    pool = _thin_pool(_RB5_WR1, _RB5_WR1_POS)
    baseline = _bd(_RB5_WR1, [20, 19, 18, 17, 16, 5])
    state = _simulate_to_state(
        _MinBidStub(),
        2,
        pool,
        cfg,
        baseline_dollars=baseline,
        price_jitter=0.0,
        rng=np.random.default_rng(0),
    )
    bot = [p for (_g, p, _pr) in state.rosters[0]]  # seat index 0 is a bot (hero is index 1)
    assert bot.count("RB") >= 1 and bot.count("WR") >= 1  # a fillable starting lineup


def test_forced_pick_completes_and_warns_when_pool_thin() -> None:
    # n_teams=2, {RB:1, BENCH:1}: bench-eligible is RB only (no QB slot), so a QB is never
    # bot-rosterable. With 2 RB + 2 QB, once both RBs are gone a bot's only eligible position
    # (RB) is exhausted -> the forced-pick path fires (ungated) so the auction completes.
    # Hand-built baseline so the thin pool isn't rejected by generate_auction_values' fill check.
    cfg = _config(n_teams=2, roster_slots={RosterSlot.RB: 1, RosterSlot.BENCH: 1})
    ids = ["00-2000000", "00-2000001", "00-1000000", "00-1000001"]
    pool = _thin_pool(ids, ["RB", "RB", "QB", "QB"])
    baseline = _bd(ids, [50, 40, 0, 0])  # QBs out-of-pool ($0) -> nominated last
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        league = simulate_auction(
            StaticDollarBid(),
            1,
            pool,
            cfg,
            baseline_dollars=baseline,
            price_jitter=0.0,
            rng=np.random.default_rng(0),
        )
    assert all(len(r) == cfg.roster_size for r in league.values())  # every seat filled
    assert any("pool thin" in str(w.message) for w in caught)  # forced-pick warned


def test_hero_is_not_gated() -> None:
    # Same 5-RB/1-WR pool with CHEAP RB baselines so a max-bidding hero (seat index 0, my_seat=1)
    # affords all three. RBs are nominated before the WR, so the hero fills on 3 RB and strands
    # its own WR starter — exceeding the bot RB cap (2) and ending non-startable, which the gate
    # forbids for bots. Proves the hero has no eligibility gate / starter reservation; the gated
    # bot still gets its WR. (Green before and after — guard that the hero stays ungated post-T5.)
    cfg = _config(n_teams=2, roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1})
    pool = _thin_pool(_RB5_WR1, _RB5_WR1_POS)
    baseline = _bd(_RB5_WR1, [10, 9, 8, 7, 6, 5])  # cheap RBs (hero affords 3), WR last
    state = _simulate_to_state(
        _MaxBidStub(),
        1,
        pool,
        cfg,
        baseline_dollars=baseline,
        price_jitter=0.0,
        rng=np.random.default_rng(0),
    )
    hero = [p for (_g, p, _pr) in state.rosters[0]]
    bot = [p for (_g, p, _pr) in state.rosters[1]]
    # ungated: hero exceeds bot RB cap, WR starter stranded
    assert hero.count("RB") == 3 and hero.count("WR") == 0
    assert bot.count("WR") >= 1  # the gated bot still reserves its WR starter
