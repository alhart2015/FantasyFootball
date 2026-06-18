import pandas as pd

from projections.draft.assistant.auction.bid_strategy import AuctionView, StaticDollarBid
from projections.draft.assistant.auction.tournament import (
    METRICS,
    AuctionTournamentResult,
    run_auction_tournament,
)
from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, RosterSlot, Ruleset


def _config(
    n_teams: int = 6,
    budget: int = 100,
    roster_slots: dict[RosterSlot, int] | None = None,
) -> LeagueConfig:
    # n_teams >= 6 (and even) — project_draft requires >= PLAYOFF_SIZE teams and an even
    # count for gauntlet_schedule; a smaller league raises in scoring.
    return LeagueConfig(
        name="t",
        n_teams=n_teams,
        budget=budget,
        min_bid=1,
        roster_slots=roster_slots or {RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def _pool(n: int = 40) -> pd.DataFrame:
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
            "is_rookie": [False] * n,
        }
    )


def _avail(pool: pd.DataFrame) -> PlayerAvailability:
    return PlayerAvailability(p={g: 0.95 for g in pool["gsis_id"].astype(str)}, bye={})


def test_result_has_per_model_per_metric_intervals_and_no_winner() -> None:
    pool = _pool(40)
    cfg = _config(6)
    result = run_auction_tournament(
        {"static": StaticDollarBid()},
        pool,
        cfg,
        my_seat=1,
        n_seeds=4,
        price_jitter=0.1,
        base_seed=0,
        n_sims=50,
        availability=_avail(pool),
        params=VarianceParams.load(),
    )
    assert isinstance(result, AuctionTournamentResult)
    assert set(result.summaries["static"]) == set(METRICS)
    assert not hasattr(result, "winner")  # data-gathering: no winner field exists
    assert result.season_base_seed == 0 + 1_000_000


def test_paired_diffs_recorded_for_each_pair() -> None:
    pool = _pool(40)
    cfg = _config(6)
    result = run_auction_tournament(
        {"a": StaticDollarBid(), "b": StaticDollarBid()},
        pool,
        cfg,
        my_seat=1,
        n_seeds=3,
        price_jitter=0.1,
        base_seed=0,
        n_sims=40,
        availability=_avail(pool),
        params=VarianceParams.load(),
    )
    assert "a_vs_b" in result.paired_diffs
    assert set(result.paired_diffs["a_vs_b"]) == set(METRICS)
    # identical models, paired: every metric diff is ~0
    assert abs(result.paired_diffs["a_vs_b"]["mean_points"].point) < 1e-9


def test_league_driven_runs_under_two_configs() -> None:
    pool = _pool(60)
    avail, params = _avail(pool), VarianceParams.load()
    for cfg in (
        _config(6, budget=100),
        _config(
            8,
            budget=50,
            roster_slots={
                RosterSlot.QB: 1,
                RosterSlot.RB: 1,
                RosterSlot.WR: 1,
                RosterSlot.BENCH: 2,
            },
        ),
    ):
        # ensure the QB config has QBs in the pool
        p = pool.copy()
        if RosterSlot.QB in cfg.roster_slots:
            p.loc[p.index[:10], "position"] = "QB"
        res = run_auction_tournament(
            {"static": StaticDollarBid()},
            p,
            cfg,
            my_seat=1,
            n_seeds=2,
            price_jitter=0.1,
            base_seed=0,
            n_sims=30,
            availability=avail,
            params=params,
        )
        assert set(res.summaries["static"]) == set(METRICS)


class _MinBidStub:
    """A hero that always bids the minimum — it loses every contested nomination, so at a
    fixed seed it reliably ends with a worse roster than a value-bidding hero."""

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        return config.min_bid


def test_constant_edge_paired_diff_excludes_zero() -> None:
    # Spec §4 "recorded comparison": a model with a constant per-seed edge -> the paired-diff CI
    # excludes 0, and the diff is RECORDED (not labeled a winner).
    pool = _pool(40)
    cfg = _config(6)
    result = run_auction_tournament(
        {"static": StaticDollarBid(), "min": _MinBidStub()},
        pool,
        cfg,
        my_seat=1,
        n_seeds=6,
        price_jitter=0.0,
        base_seed=0,
        n_sims=60,
        availability=_avail(pool),
        params=VarianceParams.load(),
    )
    diff = result.paired_diffs["static_vs_min"]["mean_points"]
    assert diff.lo_95 > 0.0  # static reliably out-rosters a min-bidding hero
    assert not hasattr(result, "winner")  # recorded as data; no winner declared
