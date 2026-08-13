import pandas as pd
import pytest

from projections.draft.assistant.auction.bid_strategy import (
    AuctionView,
    BalancedValueBid,
    StaticDollarBid,
)
from projections.draft.assistant.auction.nomination import HeroNominator, drain_max
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


def test_balanced_contestant_races_and_is_scored() -> None:
    pool = _pool(40)
    cfg = _config(6)
    result = run_auction_tournament(
        {"balanced": BalancedValueBid()},
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
    assert set(result.summaries["balanced"]) == set(METRICS)


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


def _pool_with_inverted_espn(n: int = 40) -> pd.DataFrame:
    """Pool whose ESPN $ are INVERTED vs vorp (best-vorp player gets the lowest ESPN $), so the
    ESPN-anchored bot market diverges hard from SOS."""
    p = _pool(n)
    espn = [int(5 + i) for i in range(n)]  # ascending: worst SOS player priced highest
    p["espn_auction_dollars"] = pd.array(espn, dtype="Int64")
    return p


def test_bot_prices_unknown_raises() -> None:
    pool = _pool(40)
    cfg = _config(6)
    with pytest.raises(ValueError, match="bot_prices"):
        run_auction_tournament(
            {"static": StaticDollarBid()},
            pool,
            cfg,
            my_seat=1,
            n_seeds=2,
            price_jitter=0.1,
            base_seed=0,
            n_sims=20,
            availability=_avail(pool),
            params=VarianceParams.load(),
            bot_prices="sos",  # type: ignore[arg-type]  # deliberately invalid: exercises the runtime guard
        )


def test_bot_prices_espn_without_column_warns_and_matches_model() -> None:
    pool = _pool(40)
    cfg = _config(6)
    params = VarianceParams.load()
    avail = _avail(pool)
    with pytest.warns(UserWarning, match="espn"):
        espn = run_auction_tournament(
            {"static": StaticDollarBid()},
            pool,
            cfg,
            my_seat=1,
            n_seeds=3,
            price_jitter=0.1,
            base_seed=0,
            n_sims=30,
            availability=avail,
            params=params,
            bot_prices="espn",
        )
    model = run_auction_tournament(
        {"static": StaticDollarBid()},
        pool,
        cfg,
        my_seat=1,
        n_seeds=3,
        price_jitter=0.1,
        base_seed=0,
        n_sims=30,
        availability=avail,
        params=params,
        bot_prices="model",
    )
    assert (
        espn.summaries["static"]["mean_points"].point
        == model.summaries["static"]["mean_points"].point
    )


def test_bot_prices_espn_with_column_differs_from_model() -> None:
    pool = _pool_with_inverted_espn(40)
    cfg = _config(6)
    params = VarianceParams.load()
    avail = _avail(pool)
    espn = run_auction_tournament(
        {"static": StaticDollarBid()},
        pool,
        cfg,
        my_seat=1,
        n_seeds=4,
        price_jitter=0.1,
        base_seed=0,
        n_sims=30,
        availability=avail,
        params=params,
        bot_prices="espn",
    )
    model = run_auction_tournament(
        {"static": StaticDollarBid()},
        pool,
        cfg,
        my_seat=1,
        n_seeds=4,
        price_jitter=0.1,
        base_seed=0,
        n_sims=30,
        availability=avail,
        params=params,
        bot_prices="model",
    )
    assert (
        espn.summaries["static"]["mean_points"].point
        != model.summaries["static"]["mean_points"].point
    )


# --- hero_nominators (Slice 2b: race NOMINATORS at a fixed bid) --------------------------------


def _nominator_race(
    hero_nominators: dict[str, HeroNominator | None] | None,
) -> AuctionTournamentResult:
    """Two contestants under the SAME bid model, so any difference is the nominator's alone."""
    pool = _pool(40)
    cfg = _config(6)
    return run_auction_tournament(
        {"control": StaticDollarBid(), "poison": StaticDollarBid()},
        pool,
        cfg,
        my_seat=1,
        n_seeds=4,
        price_jitter=0.1,
        base_seed=0,
        n_sims=50,
        availability=_avail(pool),
        params=VarianceParams.load(),
        nomination_temp=1.0,  # the temp>0 path, where a desynced rng draw would show up
        hero_nominators=hero_nominators,
    )


def test_hero_nominators_none_is_identity() -> None:
    """Passing the parameter at all must not perturb the default path (Run O's R1)."""
    without = _nominator_race(None)
    mapped_to_none = _nominator_race({"control": None, "poison": None})
    for name in ("control", "poison"):
        for m in METRICS:
            assert without.summaries[name][m].point == mapped_to_none.summaries[name][m].point, (
                f"{name}/{m} moved when hero_nominators was supplied as all-None"
            )


def test_hero_nominators_changes_only_the_mapped_contestant() -> None:
    """The control keeps the engine's nomination; the poisoned seat diverges from it.

    Also the CRN check that matters: `control` must be bit-identical to the no-hook run, which only
    holds because the engine draws the central nominee before overriding it (commit f9ccb0e).
    """
    baseline = _nominator_race(None)
    poisoned = _nominator_race({"poison": drain_max})
    assert (
        poisoned.summaries["control"]["mean_points"].point
        == baseline.summaries["control"]["mean_points"].point
    )
    assert (
        poisoned.summaries["poison"]["mean_points"].point
        != baseline.summaries["poison"]["mean_points"].point
    )


def test_hero_nominators_unknown_contestant_raises() -> None:
    # A typo'd name silently dropped would leave the probe comparing two identical controls.
    with pytest.raises(ValueError, match="unknown"):
        _nominator_race({"typo": drain_max})
