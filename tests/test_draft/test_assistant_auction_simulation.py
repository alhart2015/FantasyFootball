import warnings
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from projections.draft.assistant.auction.bid_strategy import (
    AnchorBudgetBid,
    AuctionView,
    BigStackBid,
    StackRatioBid,
    StaticDollarBid,
    StudsAndDepthBid,
)
from projections.draft.assistant.auction.market import (
    AggressiveBot,
    BalancedBot,
    PatientValueBot,
    _value_tier,
)
from projections.draft.assistant.auction.simulation import (
    AuctionState,
    PickRecord,
    _build_view,
    _feasible_max,
    _sample_nominee,
    _simulate_to_state,
    simulate_auction,
    validate_auction_inputs,
)
from projections.draft.assistant.auction.snake_bot import SnakeBoard
from projections.draft.assistant.auction.tournament import _SNAKE_SUBSTREAM
from projections.draft.auction import generate_auction_values
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import bot_position_bounds
from projections.schemas import _PYARROW_STR, Position, RosterSlot, Ruleset


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


def _pool_with_adp(n: int = 40) -> pd.DataFrame:
    # _pool() + an ascending, all-positive consensus_adp column so the snake regime activates.
    pool = _pool(n)
    pool["consensus_adp"] = pd.array(range(1, n + 1), dtype="Float64")
    return pool


def _broke_config() -> LeagueConfig:
    # budget == min_bid * roster_size (=1*3=3) => surplus 0 in generate_auction_values (all $1),
    # AND every seat is broke from pick 1 (feasible_max == min_bid throughout). The whole auction is
    # then governed by the snake regime — the only way to actually exercise broke behavior.
    return _config(budget=3)


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


def test_trace_records_every_pick_in_draft_order() -> None:
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    trace: list[PickRecord] = []
    state = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        rng=np.random.default_rng(0),
        trace=trace,
    )
    total = cfg.roster_size * cfg.n_teams
    assert len(trace) == total  # one record per award
    assert [r.pick for r in trace] == list(range(1, total + 1))  # 1-based draft order
    priced = {(str(g), pr) for seat in state.rosters for (g, _p, pr) in seat}
    assert all((r.gsis_id, r.price) in priced for r in trace)  # trace price == awarded price
    # room_budget is the running sum of remaining budgets: strictly decreasing to sum(state.budgets)
    budgets = [r.room_budget for r in trace]
    assert budgets == sorted(budgets, reverse=True)
    assert trace[-1].room_budget == sum(state.budgets)


def test_trace_none_is_behaviour_preserving() -> None:
    # Appending to `trace` draws no RNG, so a traced run must be byte-identical to an untraced one.
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    untraced = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        rng=np.random.default_rng(3),
    )
    trace: list[PickRecord] = []
    traced = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        rng=np.random.default_rng(3),
        trace=trace,
    )
    assert traced.rosters == untraced.rosters  # identical draft
    assert traced.budgets == untraced.budgets
    assert len(trace) == cfg.roster_size * cfg.n_teams


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


def test_hero_is_gated_like_a_bot() -> None:
    # SUPERSEDES test_hero_is_not_gated (sane-bots slice, spec R7a). That slice deliberately left
    # the hero ungated to isolate the bid model; this slice (Goal 1) gates it with the SAME
    # bot_eligible/bot_position_bounds rule. Same cheap-RB pool: pre-gate the max-bidding hero took
    # 3 RB / 0 WR; gated it must stop at the RB max (2 for {RB:1,WR:1,BENCH:1}) and reserve its WR.
    cfg = _config(n_teams=2, roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1})
    pool = _thin_pool(_RB5_WR1, _RB5_WR1_POS)
    baseline = _bd(_RB5_WR1, [10, 9, 8, 7, 6, 5])
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
    assert hero.count("RB") <= 2  # gated: never exceeds the RB max
    assert hero.count("WR") >= 1  # gated: reserves the WR starter (no empty starting slot)


def test_gated_anchor_hero_builds_a_startable_roster() -> None:
    # AnchorBudgetBid through the full engine: respects the gate (startable, within max) and still
    # concentrates budget (pays up for an anchor instead of spreading $1s). _pool carries vorp.
    cfg = _config(n_teams=4, roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.BENCH: 2})
    pool = _pool(40)  # RB/WR only; need >= 4*6 = 24 players
    baseline = _baseline(pool, cfg)
    state = _simulate_to_state(
        AnchorBudgetBid(),
        1,
        pool,
        cfg,
        baseline_dollars=baseline,
        price_jitter=0.0,
        rng=np.random.default_rng(0),
    )
    hero = [p for (_g, p, _pr) in state.rosters[0]]
    assert len(hero) == cfg.roster_size  # filled
    assert hero.count("RB") <= 3 and hero.count("WR") <= 3  # within the gate max (min+bench share)
    assert hero.count("RB") >= 2 and hero.count("WR") >= 2  # minimum starters reserved


def test_gated_hero_is_deterministic() -> None:
    cfg = _config(n_teams=4, roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.BENCH: 2})
    pool = _pool(40)
    baseline = _baseline(pool, cfg)
    a = _simulate_to_state(
        AnchorBudgetBid(),
        1,
        pool,
        cfg,
        baseline_dollars=baseline,
        price_jitter=0.15,
        rng=np.random.default_rng(7),
    )
    b = _simulate_to_state(
        AnchorBudgetBid(),
        1,
        pool,
        cfg,
        baseline_dollars=baseline,
        price_jitter=0.15,
        rng=np.random.default_rng(7),
    )
    assert a.rosters == b.rosters  # same seed -> identical draft


def test_sample_nominee_temp_zero_is_argmax() -> None:
    # candidates are pre-sorted value-desc; temp=0 must return the first (no RNG draw).
    cands = ["A", "B", "C"]
    val = {"A": 50.0, "B": 20.0, "C": 1.0}
    assert _sample_nominee(cands, val, 0.0, np.random.default_rng(0)) == "A"


def test_sample_nominee_single_candidate() -> None:
    assert _sample_nominee(["X"], {"X": 0.0}, 1.0, np.random.default_rng(0)) == "X"


def test_sample_nominee_temp_one_favors_value_but_samples_tail() -> None:
    cands = ["hi", "lo1", "lo2"]
    val = {"hi": 100.0, "lo1": 1.0, "lo2": 1.0}
    rng = np.random.default_rng(0)
    picks = [_sample_nominee(cands, val, 1.0, rng) for _ in range(500)]
    assert picks.count("hi") > picks.count("lo1") + picks.count("lo2")  # value-weighted
    assert (picks.count("lo1") + picks.count("lo2")) > 0  # but the tail does come up


def test_nomination_temp_zero_is_deterministic() -> None:
    # temp=0 consumes no nomination RNG, so two runs at the same seed are identical.
    # (Backward-compat to pre-change behavior is guarded by the engine suite at default temp=0.)
    cfg = _config(n_teams=4, roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.BENCH: 2})
    pool = _pool(40)
    baseline = _baseline(pool, cfg)
    kw = dict(baseline_dollars=baseline, price_jitter=0.15)
    legacy = _simulate_to_state(StaticDollarBid(), 1, pool, cfg, rng=np.random.default_rng(3), **kw)
    temp0 = _simulate_to_state(
        StaticDollarBid(), 1, pool, cfg, rng=np.random.default_rng(3), nomination_temp=0.0, **kw
    )
    assert legacy.rosters == temp0.rosters


def _realistic_baseline(pool: pd.DataFrame) -> pd.DataFrame:
    """Hand-crafted baseline with a realistic stud/mid/scrub value spread.

    generate_auction_values() compresses all 80-player values into $11-$17 for this
    synthetic pool (equal fpts spacing => equal budget allocation), so every mid-tier
    player clears above $1 under any bot field — zero discriminating power.
    A proper spread ($1-$83) puts lower mid-tier players near the floor so
    PatientValueBot's 1.35x premium genuinely lifts them above min_bid.
    """
    gids = list(pool["gsis_id"].astype(str))
    n = len(gids)
    dollars: list[int] = []
    for i in range(n):
        if i < 8:  # studs: $55-$83
            dollars.append(55 + (7 - i) * 4)
        elif i < 35:  # mid-tier: $15-$4
            dollars.append(max(4, int(15 - (i - 8) * 0.4)))
        else:  # scrubs: $1-$3
            dollars.append(max(1, 3 - min(i - 35, 2)))
    return pd.DataFrame(
        {
            "gsis_id": pd.array(gids, dtype=_PYARROW_STR),
            "auction_dollars": dollars,
            "bot_dollars": dollars,
            "in_pool": [True] * n,
        }
    )


def test_mixed_field_bids_midtier_off_the_dollar_floor() -> None:
    # THE CORE FIX: legacy (all-aggressive) vs realistic (mixed field with PatientValueBot).
    # Mid-tier players should clear ABOVE min_bid far more often under the mixed field.
    # Uses _realistic_baseline() — a hand-crafted stud/mid/scrub spread — because
    # generate_auction_values() compresses all values to $11-$17 for this synthetic pool,
    # making every mid-tier player clear above $1 regardless of field (no discriminating power).
    cfg = _config(n_teams=8, roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.BENCH: 3})
    pool = _pool(80)
    baseline = _realistic_baseline(pool)
    bd_idx = baseline.set_index("gsis_id")

    def midtier_above_floor(state: AuctionState) -> int:
        n = 0
        for seat in range(cfg.n_teams):
            for gsis, _pos, price in state.rosters[seat]:
                val = float(bd_idx.loc[gsis, "auction_dollars"])
                if _value_tier(val, baseline, 0.10, 0.50) == "mid" and price > cfg.min_bid:
                    n += 1
        return n

    legacy = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=baseline,
        price_jitter=0.15,
        rng=np.random.default_rng(11),
    )
    mixed = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=baseline,
        price_jitter=0.15,
        rng=np.random.default_rng(11),
        nomination_temp=1.0,
        bot_archetypes=[AggressiveBot(), PatientValueBot(), BalancedBot()],
    )
    assert midtier_above_floor(mixed) > midtier_above_floor(legacy)
    assert midtier_above_floor(mixed) >= 1


def test_mixed_field_is_deterministic() -> None:
    cfg = _config(n_teams=8, roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.BENCH: 3})
    pool = _pool(80)
    bl = _realistic_baseline(pool)
    kw = dict(
        baseline_dollars=bl,
        price_jitter=0.15,
        nomination_temp=1.0,
        bot_archetypes=[AggressiveBot(), PatientValueBot(), BalancedBot()],
    )
    a = _simulate_to_state(StaticDollarBid(), 1, pool, cfg, rng=np.random.default_rng(5), **kw)
    b = _simulate_to_state(StaticDollarBid(), 1, pool, cfg, rng=np.random.default_rng(5), **kw)
    assert a.rosters == b.rosters


def test_studs_and_depth_deploys_budget_and_is_startable() -> None:
    # Realistic market (mixed field + value-weighted-random nomination). StudsAndDepthBid should
    # deploy most of its budget (the Run-E failure mode was idle cash) and field a startable roster,
    # spending far more than a floor-bidder hero on the same seed.
    cfg = _config(n_teams=8, roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.BENCH: 3})
    pool = _pool(80)
    baseline = _realistic_baseline(pool)
    kw = dict(
        baseline_dollars=baseline,
        price_jitter=0.15,
        nomination_temp=1.0,
        bot_archetypes=[AggressiveBot(), PatientValueBot(), BalancedBot()],
    )
    studs = _simulate_to_state(
        StudsAndDepthBid(), 1, pool, cfg, rng=np.random.default_rng(11), **kw
    )
    floor = _simulate_to_state(_MinBidStub(), 1, pool, cfg, rng=np.random.default_rng(11), **kw)
    studs_spend = cfg.budget - studs.budgets[0]
    floor_spend = cfg.budget - floor.budgets[0]
    assert studs_spend > floor_spend  # actually deploys budget vs a min-bidder
    assert studs_spend >= 0.75 * cfg.budget  # ~full-budget deployment (no large idle cash)
    hero = [p for (_g, p, _pr) in studs.rosters[0]]
    assert len(hero) == cfg.roster_size  # filled
    assert hero.count("RB") >= 2 and hero.count("WR") >= 2  # startable starting lineup


def test_studs_and_depth_is_deterministic() -> None:
    cfg = _config(n_teams=8, roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.BENCH: 3})
    pool = _pool(80)
    bl = _realistic_baseline(pool)
    kw = dict(
        baseline_dollars=bl,
        price_jitter=0.15,
        nomination_temp=1.0,
        bot_archetypes=[AggressiveBot(), PatientValueBot(), BalancedBot()],
    )
    a = _simulate_to_state(StudsAndDepthBid(), 1, pool, cfg, rng=np.random.default_rng(5), **kw)
    b = _simulate_to_state(StudsAndDepthBid(), 1, pool, cfg, rng=np.random.default_rng(5), **kw)
    assert a.rosters == b.rosters


def _flat_bot_dollars(pool: pd.DataFrame, value: int) -> pd.Series:
    """A bot_dollars Series over every pool gsis_id, all equal -> bots value all players same."""
    return pd.Series(
        pd.array([value] * len(pool), dtype=pd.Int64Dtype()),
        index=pd.Index(pool["gsis_id"], name="gsis_id"),
    )


def test_bot_dollars_none_reproduces_baseline() -> None:
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    a = simulate_auction(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        rng=np.random.default_rng(0),
    )
    b = simulate_auction(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        rng=np.random.default_rng(0),
        bot_dollars=None,
    )
    assert a == b  # explicit None is identical to the default


def test_bot_dollars_changes_the_bot_market() -> None:
    # Flat bot_dollars makes bots value every player equally -> a different market than SOS,
    # so the resulting league differs from the bot_dollars=None (SOS) run at the same seed.
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    sos = simulate_auction(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        rng=np.random.default_rng(0),
    )
    flat = simulate_auction(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        rng=np.random.default_rng(0),
        bot_dollars=_flat_bot_dollars(pool, 20),
    )
    assert sos != flat  # bot pricing changed -> different rosters/prices


def test_snake_rng_param_does_not_change_flush_rosters() -> None:
    # Boards are built (ADP present) but UNUSED in Task 3, and flush bots never enter the snake
    # path: rosters are identical whether snake_rng is defaulted or given, and whether consensus_adp
    # is present or dropped.
    pool, config = _pool_with_adp(), _config()  # flush (budget 100)
    # NOTE: simulate_auction signature is (strategy, my_seat, pool, config, *, baseline_dollars,
    # price_jitter, rng, nomination_temp, ...). my_seat is the 2nd POSITIONAL arg.
    bd = _baseline(pool, config)
    common = dict(baseline_dollars=bd, price_jitter=0.15, nomination_temp=1.0)
    a = simulate_auction(StaticDollarBid(), 1, pool, config, rng=np.random.default_rng(0), **common)
    b = simulate_auction(
        StaticDollarBid(),
        1,
        pool,
        config,
        rng=np.random.default_rng(0),
        snake_rng=np.random.default_rng([0, 7]),
        **common,
    )
    no_adp = pool.drop(columns=["consensus_adp"])
    c = simulate_auction(
        StaticDollarBid(),
        1,
        no_adp,
        config,
        rng=np.random.default_rng(0),
        baseline_dollars=_baseline(no_adp, config),
        price_jitter=0.15,
        nomination_temp=1.0,
    )
    assert a == b == c


def test_all_broke_auction_completes_no_assert() -> None:
    # With every seat broke and abstaining off-target, the engine must NOT hit `assert bids` — the
    # nominator backstop (or the broke nominator self-bidding its target) keeps each round
    # non-empty, and every roster fills.
    pool, config = _pool_with_adp(), _broke_config()
    league = simulate_auction(
        StaticDollarBid(),
        1,
        pool,
        config,
        baseline_dollars=_baseline(pool, config),
        price_jitter=0.15,
        rng=np.random.default_rng(3),
        nomination_temp=1.0,
        snake_rng=np.random.default_rng([3, 7]),
    )
    assert all(len(r) == config.roster_size for r in league.values())


def test_broke_regime_respects_position_caps() -> None:
    # Behavioral guard: broke bots never roster an off-position scrub — every seat's roster respects
    # the position-cap maxima (a blind $1-scrub grab would blow a cap or strand a needed slot).
    pool, config = _pool_with_adp(), _broke_config()
    state = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        config,
        baseline_dollars=_baseline(pool, config),
        price_jitter=0.15,
        rng=np.random.default_rng(11),
        nomination_temp=1.0,
        snake_rng=np.random.default_rng([11, 7]),
    )
    _minimums, maximums = bot_position_bounds(config.roster_slots)
    for roster in state.rosters:
        counts = Counter(Position(pos) for _g, pos, _pr in roster)
        for p, c in counts.items():
            assert c <= maximums[p]


def test_broke_regime_drafts_better_adp_depth_than_dollar_grab() -> None:
    # THE FEATURE'S VALUE (no-scrub property): broke bots draft their best-AVAILABLE-by-ADP player
    # at a needed position rather than blindly $1-grabbing whatever is nominated. Run the SAME
    # all-broke seed twice — once with the snake regime ACTIVE (consensus_adp present) and once with
    # it DISABLED (consensus_adp dropped -> today's archetype $1-grab behavior). Join each bot's won
    # gsis_ids back to consensus_adp; the snake-regime bots must roster a strictly lower mean ADP
    # (earlier/better players) than the no-ADP bots. The hero seat (index 0) is excluded — only the
    # broke BOTS are governed by the snake regime.
    config = _broke_config()
    adp_pool = _pool_with_adp()
    no_adp = adp_pool.drop(columns=["consensus_adp"])
    adp_by_id = {
        str(g): float(a)
        for g, a in zip(adp_pool["gsis_id"], adp_pool["consensus_adp"], strict=True)
    }

    snake = _simulate_to_state(
        StaticDollarBid(),
        1,
        adp_pool,
        config,
        baseline_dollars=_baseline(adp_pool, config),
        price_jitter=0.15,
        rng=np.random.default_rng(11),
        nomination_temp=1.0,
        snake_rng=np.random.default_rng([11, 7]),
    )
    grab = _simulate_to_state(
        StaticDollarBid(),
        1,
        no_adp,
        config,
        baseline_dollars=_baseline(no_adp, config),
        price_jitter=0.15,
        rng=np.random.default_rng(11),
        nomination_temp=1.0,
        snake_rng=np.random.default_rng([11, 7]),
    )

    def bot_mean_adp(state: AuctionState) -> float:
        # bot seats are every seat except the hero (index 0); join won ids back to consensus_adp.
        adps = [
            adp_by_id[str(g)]
            for seat in range(1, config.n_teams)
            for (g, _p, _pr) in state.rosters[seat]
        ]
        return float(np.mean(adps))

    assert bot_mean_adp(snake) < bot_mean_adp(grab)  # snake regime -> earlier/better depth

    # Keep the cheap cap guard too: the better depth never comes at the cost of a blown cap.
    _minimums, maximums = bot_position_bounds(config.roster_slots)
    for roster in snake.rosters:
        counts = Counter(Position(pos) for _g, pos, _pr in roster)
        for p, c in counts.items():
            assert c <= maximums[p]


def test_backstop_else_branch_awards_to_eligible_non_nominator_seat() -> None:
    # Reaches the non-forced empty-bids ELSE backstop (simulation.py): a FLUSH nominator nominates a
    # room-union player it personally CANNOT roster (its own position capped) while every responding
    # open seat is a broke bot not targeting it -> `bids` is empty -> the player is awarded at
    # min_bid to the lowest-index open seat that CAN roster it (not the nominator). budget=4 (one $1
    # above the all-broke floor of 3) is the spec's mixed regime: seats start flush and go broke
    # after one buy. Found by a bounded seed search (configs x seeds 0..200); this exact
    # (config, seed) is the pinned hit. The auction must complete cleanly through that branch with
    # no AssertionError escaping the backstop's `assert eligible_seat is not None`.
    pool = _pool_with_adp()
    config = _config(n_teams=4, budget=4)  # mixed regime: surplus $1 over the all-broke floor
    state = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        config,
        baseline_dollars=_baseline(pool, config),
        price_jitter=0.15,
        rng=np.random.default_rng(25),  # pinned: this seed fires the ELSE backstop once
        nomination_temp=1.0,
        snake_rng=np.random.default_rng([25, 7]),
    )
    # (a) auction completed: every roster full. (b) no AssertionError escaped (we got here).
    assert all(len(r) == config.roster_size for r in state.rosters)
    # (c) no position cap violated by the backstop award.
    _minimums, maximums = bot_position_bounds(config.roster_slots)
    for roster in state.rosters:
        counts = Counter(Position(pos) for _g, pos, _pr in roster)
        for p, c in counts.items():
            assert c <= maximums[p]


def test_snake_regime_changes_outcomes_vs_no_adp() -> None:
    # The regime must DO something: an all-broke auction with usable ADP must produce a different
    # outcome than the same all-broke auction with the regime disabled (consensus_adp dropped ->
    # today's archetype/central behavior). Both are reproducible at a fixed seed.
    config = _broke_config()
    adp_pool = _pool_with_adp()
    no_adp = adp_pool.drop(columns=["consensus_adp"])
    # my_seat is positional (2nd arg); price_jitter / nomination_temp held common across both runs.
    with_adp = simulate_auction(
        StaticDollarBid(),
        1,
        adp_pool,
        config,
        rng=np.random.default_rng(2),
        baseline_dollars=_baseline(adp_pool, config),
        snake_rng=np.random.default_rng([2, 7]),
        price_jitter=0.15,
        nomination_temp=1.0,
    )
    without = simulate_auction(
        StaticDollarBid(),
        1,
        no_adp,
        config,
        rng=np.random.default_rng(2),
        baseline_dollars=_baseline(no_adp, config),
        snake_rng=np.random.default_rng([2, 7]),
        price_jitter=0.15,
        nomination_temp=1.0,
    )
    assert with_adp != without  # the snake regime re-routes who drafts whom


def test_broke_nominator_auction_completes() -> None:
    # All-broke auction with a broke bot on the clock each round: the broke-nominator override must
    # produce a valid, rosterable nominee every round and fill every roster (no KeyError / assert).
    pool, config = _pool_with_adp(), _broke_config()
    state = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        config,
        baseline_dollars=_baseline(pool, config),
        price_jitter=0.15,
        rng=np.random.default_rng(5),
        nomination_temp=1.0,
        snake_rng=np.random.default_rng([5, 7]),
    )
    assert all(len(r) == config.roster_size for r in state.rosters)


def test_no_adp_pool_is_byte_identical_regardless_of_snake_rng() -> None:
    # Regime fully disabled without ADP -> the snake_rng value cannot matter (boards never built),
    # even in an all-broke config where the regime WOULD otherwise be active.
    config = _broke_config()
    no_adp = _pool_with_adp().drop(columns=["consensus_adp"])
    a = simulate_auction(
        StaticDollarBid(),
        1,
        no_adp,
        config,
        rng=np.random.default_rng(9),
        baseline_dollars=_baseline(no_adp, config),
        price_jitter=0.15,
        nomination_temp=1.0,
    )
    b = simulate_auction(
        StaticDollarBid(),
        1,
        no_adp,
        config,
        rng=np.random.default_rng(9),
        snake_rng=np.random.default_rng([9, 7]),
        baseline_dollars=_baseline(no_adp, config),
        price_jitter=0.15,
        nomination_temp=1.0,
    )
    assert a == b


def test_snake_substream_is_seed_only_and_shared() -> None:
    # The dedicated substream depends on the seed alone, so the bot field is identical across models
    # at a fixed seed (CRN). Build boards the way the tournament does and confirm equality; also
    # confirm the substream is distinct from the bidding stream (a different board).
    pool = _pool_with_adp()
    elig = frozenset(Position)
    base = 0
    snake_a = np.random.default_rng([base, _SNAKE_SUBSTREAM])
    snake_b = np.random.default_rng([base, _SNAKE_SUBSTREAM])
    bidding = np.random.default_rng(base)  # the scalar bidding seed
    pick = SnakeBoard(pool, snake_a).best_available(frozenset(), elig)
    assert pick == SnakeBoard(pool, snake_b).best_available(frozenset(), elig)  # CRN: seed-only
    # The substream is a DIFFERENT stream than the scalar bidding seed (else the hero's bidding RNG
    # would perturb the bot field). Compare the streams' first draws directly — deterministic, not a
    # flaky board comparison: the two fixed seeds produce two fixed, distinct values.
    assert bidding.random() != np.random.default_rng([base, _SNAKE_SUBSTREAM]).random()


def test_hero_nominator_none_matches_default() -> None:
    # None hook is byte-identical to no hook: same rosters and budgets (spec R1).
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    a = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        rng=np.random.default_rng(0),
    )
    b = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        rng=np.random.default_rng(0),
        hero_nominator=None,
    )
    assert a.rosters == b.rosters
    assert a.budgets == b.budgets


def test_hero_nominator_choice_changes_the_draft() -> None:
    # Two different hero nominators (priciest vs cheapest) diverge deterministically -> the hook
    # fires and the returned id is actually used (spec R2 behavioral evidence).
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    priciest = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        nomination_temp=0.0,
        rng=np.random.default_rng(0),
        hero_nominator=lambda c, ctx: max(c, key=lambda g: ctx.value_by_id[str(g)]),
    )
    cheapest = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        nomination_temp=0.0,
        rng=np.random.default_rng(0),
        hero_nominator=lambda c, ctx: min(c, key=lambda g: ctx.value_by_id[str(g)]),
    )
    assert priciest.rosters != cheapest.rosters
    for state in (priciest, cheapest):  # both are still valid, full drafts
        assert all(len(r) == cfg.roster_size for r in state.rosters)
        ids = [g for r in state.rosters for (g, _p, _pr) in r]
        assert len(ids) == len(set(ids))


def test_hero_nominator_receives_only_valid_candidate_lists() -> None:
    # Every candidate list the hero sees is non-empty and dup-free (spec R3, checked live).
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    calls = 0

    def spy(candidates: list[str], ctx: object) -> str:
        nonlocal calls
        calls += 1
        assert candidates
        assert len({str(g) for g in candidates}) == len(candidates)
        return candidates[0]

    _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        rng=np.random.default_rng(0),
        hero_nominator=spy,
    )
    assert calls > 0


def test_hero_nominator_draws_central_rng_for_crn(monkeypatch: pytest.MonkeyPatch) -> None:
    # The override path must still call _sample_nominee (advancing the shared rng) so the control
    # (None) and poison arms stay CRN-paired at nomination_temp>0. Assert the hook draws the central
    # nominee exactly as many times as the baseline — _pool(40) has no adp (no snake path) and can't
    # be exhausted (no forced path), so all 12 nominations go through _sample_nominee.
    import projections.draft.assistant.auction.simulation as sim

    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    calls = {"n": 0}
    real = sim._sample_nominee

    def counting(
        candidates: list[str], val_by_id: dict[str, float], temp: float, rng: np.random.Generator
    ) -> str:
        calls["n"] += 1
        return real(candidates, val_by_id, temp, rng)

    monkeypatch.setattr(sim, "_sample_nominee", counting)

    sim._simulate_to_state(  # baseline run starts with the counter at its init 0
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        nomination_temp=1.0,
        rng=np.random.default_rng(0),
        hero_nominator=None,
    )
    baseline = calls["n"]
    calls["n"] = 0
    sim._simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        nomination_temp=1.0,
        rng=np.random.default_rng(0),
        hero_nominator=lambda c, ctx: c[0],
    )
    hooked = calls["n"]
    assert hooked == baseline == cfg.roster_size * cfg.n_teams


def test_hero_nominator_override_branch_consumes_no_extra_rng() -> None:
    # Stronger CRN guard than the call-count test: at temp=0 the central draw is deterministic
    # (candidates[0], no rng) and a hook returning candidates[0] picks the SAME nominee, so the
    # override branch must yield a BYTE-IDENTICAL draft to the None baseline. Any future rng.* added
    # inside the `nom == hero0` override block would draw in the hook arm only, diverging this draft
    # and tripping the test — a desync the _sample_nominee call-count parity check cannot see.
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    baseline = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        nomination_temp=0.0,
        rng=np.random.default_rng(0),
        hero_nominator=None,
    )
    overridden = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        nomination_temp=0.0,
        rng=np.random.default_rng(0),
        hero_nominator=lambda c, ctx: c[0],
    )
    assert overridden.rosters == baseline.rosters
    assert overridden.budgets == baseline.budgets


def test_market_adp_jitter_none_matches_default() -> None:
    # market_adp_jitter=None is byte-identical to omitting it (value nomination unchanged).
    cfg = _config(n_teams=4)
    pool = _pool_with_adp(40)
    bd = _baseline(pool, cfg)
    a = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        nomination_temp=1.0,
        rng=np.random.default_rng(0),
    )
    b = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        nomination_temp=1.0,
        rng=np.random.default_rng(0),
        market_adp_jitter=None,
    )
    assert a.rosters == b.rosters
    assert a.budgets == b.budgets


def test_market_adp_nomination_nominates_lowest_adp_first() -> None:
    # With jitter=0 the shared market board is strict ADP order, so the very first (non-forced)
    # nominee is the lowest-consensus_adp player — proof the ADP-market override replaces the
    # value-weighted nominee. _pool_with_adp sets consensus_adp = 1..n ascending.
    cfg = _config(n_teams=4)
    pool = _pool_with_adp(40)
    bd = _baseline(pool, cfg)
    trace: list[PickRecord] = []
    _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        nomination_temp=1.0,
        rng=np.random.default_rng(0),
        market_adp_jitter=0.0,
        trace=trace,
    )
    lowest_adp = str(pool.sort_values("consensus_adp")["gsis_id"].iloc[0])
    assert str(trace[0].gsis_id) == lowest_adp


def test_market_board_override_consumes_the_unconditional_value_draw() -> None:
    # CRN invariant: `_sample_nominee` is drawn UNCONDITIONALLY before the market-board override, so
    # the shared rng advances identically whether or not the board overrides — the pairing the
    # value-nom-vs-ADP-nom and control-vs-poison verdicts depend on. Proof at temp>0 (the production
    # sweep regime): the jitter=0 board picks the SAME nominees regardless of nomination_temp (first
    # assert), but the discarded value draw consumes rng ONLY at temp>0 (temp=0 returns
    # candidates[0] with no rng), so the two runs' downstream bids/prices — hence rosters — MUST
    # diverge (second assert). Identical nominees + divergent rosters isolates the cause to the rng
    # the value draw consumes. If a refactor guarded that draw on the board (`if market_board is
    # None: ...`), nomination_temp would stop mattering and the rosters would collapse to identical,
    # tripping this test. (A temp=0 byte-identical test can't see it: at temp=0 the draw is free.)
    cfg = _config(n_teams=4)
    pool = _pool_with_adp(40)
    bd = _baseline(pool, cfg)
    trace0: list[PickRecord] = []
    trace1: list[PickRecord] = []
    at_temp0 = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        nomination_temp=0.0,
        rng=np.random.default_rng(0),
        market_adp_jitter=0.0,
        trace=trace0,
    )
    at_temp1 = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        nomination_temp=1.0,
        rng=np.random.default_rng(0),
        market_adp_jitter=0.0,
        trace=trace1,
    )
    assert [r.gsis_id for r in trace0] == [r.gsis_id for r in trace1]  # same board-driven nominees
    assert at_temp0.rosters != at_temp1.rosters  # only the discarded value draw's rng differs


def test_market_board_leaves_broke_bot_snake_nominations_untouched() -> None:
    # Broke bots take the snake-target path (nominee_id set BEFORE the `if nominee_id is None`
    # guard), so the shared market board — nested inside that guard — must never override them. Both
    # runs pin the hero's nomination with a hero_nominator (which fires AFTER the board), making the
    # hero's own market override a guaranteed no-op; the comparison then isolates the BROKE-BOT
    # path, so it can't false-pass on an incidental hero value==ADP coincidence. Setting the jitter
    # must be byte-identical to leaving it off. If the override were hoisted out of the broke guard,
    # broke bots would nominate the market's global lowest-ADP player instead of their snake target
    # and this draft would diverge.
    # budget == min_bid*roster_size -> every bot broke from pick 1 (snake path)
    cfg = _broke_config()
    pool = _pool_with_adp(40)
    bd = _baseline(pool, cfg)
    without = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        nomination_temp=0.0,
        rng=np.random.default_rng(0),
        hero_nominator=lambda c, ctx: c[0],
    )
    with_market = _simulate_to_state(
        StaticDollarBid(),
        1,
        pool,
        cfg,
        baseline_dollars=bd,
        price_jitter=0.1,
        nomination_temp=0.0,
        rng=np.random.default_rng(0),
        market_adp_jitter=0.0,
        hero_nominator=lambda c, ctx: c[0],
    )
    assert without.rosters == with_market.rosters
    assert without.budgets == with_market.budgets


def test_market_adp_jitter_rejects_invalid_values() -> None:
    # A NaN/inf/negative jitter must raise, not silently produce gsis-alphabetical nomination (NaN)
    # or crash deep in SnakeBoard (negative std) while the chunk still records an "ADP-market" run.
    cfg = _config(n_teams=4)
    pool = _pool_with_adp(40)
    bd = _baseline(pool, cfg)
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="market_adp_jitter"):
            _simulate_to_state(
                StaticDollarBid(),
                1,
                pool,
                cfg,
                baseline_dollars=bd,
                price_jitter=0.1,
                rng=np.random.default_rng(0),
                market_adp_jitter=bad,
            )


def test_market_adp_jitter_without_usable_adp_raises() -> None:
    # Asking for ADP-market nomination on a pool with no consensus_adp must raise rather than
    # silently fall back to value nomination (which would mislabel the run as ADP-nom).
    cfg = _config(n_teams=4)
    pool = _pool(40)  # no consensus_adp column -> adp_usable is False
    bd = _baseline(pool, cfg)
    with pytest.raises(ValueError, match="consensus_adp"):
        _simulate_to_state(
            StaticDollarBid(),
            1,
            pool,
            cfg,
            baseline_dollars=bd,
            price_jitter=0.1,
            rng=np.random.default_rng(0),
            market_adp_jitter=8.0,
        )


def test_bigstack_produces_a_legal_full_roster() -> None:
    # The engine clamps BigStackBid's (possibly huge) desired bids to feasible_max, so the hero
    # always fills a legal, full roster with no duplicate players. Spec R4. BOTH references are run
    # through a real auction: max_opp is the branch whose _advantage calls opp.remove()/max(), which
    # field_avg never touches, so it needs its own integration coverage.
    cfg = _config(n_teams=4)
    pool = _pool(40)
    for reference in ("field_avg", "max_opp"):
        league = simulate_auction(
            BigStackBid(reference=reference, overpay_gain=2.0),
            1,
            pool,
            cfg,
            baseline_dollars=_baseline(pool, cfg),
            price_jitter=0.1,
            rng=np.random.default_rng(0),
        )
        assert all(len(r) == cfg.roster_size for r in league.values())
        ids = [g for r in league.values() for g in r]
        assert len(ids) == len(set(ids))


def test_stackratio_produces_a_legal_full_roster() -> None:
    # The engine clamps StackRatioBid's (possibly huge) desired bids to feasible_max, so the hero
    # always fills a legal, full, dup-free roster. Spec R4. Both a linear (curve=1) and a convex
    # (curve=3) config run through a real auction.
    cfg = _config(n_teams=4)
    pool = _pool(40)
    for gain, curve in ((2.0, 1.0), (1.0, 3.0)):
        league = simulate_auction(
            StackRatioBid(gain=gain, curve=curve),
            1,
            pool,
            cfg,
            baseline_dollars=_baseline(pool, cfg),
            price_jitter=0.1,
            rng=np.random.default_rng(0),
        )
        assert all(len(r) == cfg.roster_size for r in league.values())
        ids = [g for r in league.values() for g in r]
        assert len(ids) == len(set(ids))
