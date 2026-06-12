import pandas as pd  # noqa: F401

from projections.draft.backtest.harness import aggregate, collect_results, run_backtest
from projections.draft.backtest.league import Calendar
from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot
from tests.test_draft.test_backtest.test_availability_stub import stub_availability
from tests.test_draft.test_backtest.test_draft_field import _synthetic_pool


def _cfg16() -> LeagueConfig:
    return LeagueConfig(
        name="t16",
        n_teams=16,
        budget=200,
        min_bid=1,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 4,
        },
        ruleset="espn_half",  # type: ignore[arg-type]
    )


def test_rates_bounded_and_seat_weighted_champions_sum_to_one() -> None:
    cfg, pool = _cfg16(), _synthetic_pool(n_per_pos=60)
    cal = Calendar(regular_weeks=tuple(range(1, 6)), playoff_weeks=(6, 7, 8), playoff_size=6)
    proj = {
        (g, wk): float(m)
        for g, m in zip(pool["gsis_id"], pool["season_mean_fpts"], strict=False)
        for wk in range(1, 9)
    }
    actual = dict(proj)
    res = run_backtest(
        n_seeds=4,
        pool=pool,
        config=cfg,
        availability=stub_availability(pool),
        proj_lookup=proj,
        actual_lookup=actual,
        calendar=cal,
        jitter=8.0,
        strategy_n_sims=5,
        base_seed=0,
    )
    # Both metric tables are present with the same strategy keys.
    for table in (res.by_strategy_actual, res.by_strategy_projected):
        assert set(table) == {"now_or_never", "season_value", "bot"}
        for m in table.values():
            for iv in (m.championship, m.playoff, m.win_pct):
                assert 0.0 <= iv.lo_95 <= iv.point <= iv.hi_95 <= 1.0
        # Exactly one champion per league => seat-weighted championship rates sum to 1.
        weighted = (
            4 * table["now_or_never"].championship.point
            + 4 * table["season_value"].championship.point
            + 8 * table["bot"].championship.point
        )
        assert abs(weighted - 1.0) < 1e-9


def test_chunked_collection_matches_monolithic() -> None:
    """Collecting seeds in chunks then aggregating == one monolithic run_backtest.

    This is the invariant the resumable chunk-runner relies on: a crash that loses a
    chunk and is retried must reconstruct byte-identical results to an uninterrupted run.
    """
    cfg, pool = _cfg16(), _synthetic_pool(n_per_pos=60)
    cal = Calendar(regular_weeks=tuple(range(1, 6)), playoff_weeks=(6, 7, 8), playoff_size=6)
    proj = {
        (g, wk): float(m)
        for g, m in zip(pool["gsis_id"], pool["season_mean_fpts"], strict=False)
        for wk in range(1, 9)
    }
    actual = dict(proj)
    kw = dict(
        pool=pool,
        config=cfg,
        availability=stub_availability(pool),
        proj_lookup=proj,
        actual_lookup=actual,
        calendar=cal,
        jitter=8.0,
        strategy_n_sims=5,
        base_seed=0,
    )
    full = run_backtest(n_seeds=4, **kw)
    a1, p1 = collect_results(seed_lo=0, seed_hi=2, **kw)
    a2, p2 = collect_results(seed_lo=2, seed_hi=4, **kw)
    chunked = aggregate(a1 + a2, p1 + p2, n_seeds=4, base_seed=0)
    assert chunked.by_strategy_actual == full.by_strategy_actual
    assert chunked.by_strategy_projected == full.by_strategy_projected
