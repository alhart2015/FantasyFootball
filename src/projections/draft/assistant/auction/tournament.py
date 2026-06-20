"""Data-gathering auction tournament: race bid models, score each league with project_draft.

Records per-model per-metric means + bootstrap CIs and paired per-seed diffs. Declares NO
winner (spec §5.1) — the adopt decision is the user's, in September.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd

from projections.draft.assistant._compare import Interval, bootstrap_mean
from projections.draft.assistant.auction.bid_strategy import AuctionBidStrategy
from projections.draft.assistant.auction.market import BotArchetype
from projections.draft.assistant.auction.simulation import simulate_auction, validate_auction_inputs
from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.league_projection import project_draft
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.auction import (
    espn_anchored_bot_prices,
    generate_auction_values,
    has_usable_espn_prices,
)
from projections.draft.league_config import LeagueConfig

_SNAKE_SUBSTREAM = 20260619  # dedicated sub-key for the broke-bot ADP noise (CRN: shared bot field)

METRICS: tuple[str, ...] = (
    "mean_points",
    "reg_win_pct",
    "make_playoffs_pct",
    "bye_pct",
    "champ_pct",
)


@dataclass(frozen=True)
class AuctionTournamentResult:
    """Per-model per-metric means/CIs + paired diffs. No winner (data-gathering)."""

    summaries: dict[str, dict[str, Interval]]
    paired_diffs: dict[str, dict[str, Interval]]
    n_seeds: int
    price_jitter: float
    base_seed: int
    season_base_seed: int
    n_sims: int
    my_seat: int
    budget: int
    min_bid: int


def _validate(
    config: LeagueConfig,
    *,
    my_seat: int,
    n_seeds: int,
    price_jitter: float,
    n_sims: int,
) -> None:
    if not 1 <= my_seat <= config.n_teams:
        raise ValueError(f"my_seat must be in 1..{config.n_teams}; got {my_seat}")
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1; got {n_seeds}")
    if price_jitter < 0:
        raise ValueError(f"price_jitter must be >= 0; got {price_jitter}")
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1; got {n_sims}")


def run_auction_tournament(
    strategies: Mapping[str, AuctionBidStrategy],
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    my_seat: int,
    n_seeds: int,
    price_jitter: float,
    base_seed: int,
    n_sims: int,
    availability: PlayerAvailability,
    params: VarianceParams,
    season_base_seed: int | None = None,
    nomination_temp: float = 0.0,
    bot_archetypes: Sequence[BotArchetype] | None = None,
    bot_prices: Literal["espn", "model"] = "espn",
    unranked_discount: float | None = None,
) -> AuctionTournamentResult:
    if season_base_seed is None:
        season_base_seed = base_seed + 1_000_000
    validate_auction_inputs(pool, config)
    _validate(config, my_seat=my_seat, n_seeds=n_seeds, price_jitter=price_jitter, n_sims=n_sims)
    baseline_dollars = generate_auction_values(pool, config)  # config-determined; computed once
    if bot_prices not in ("espn", "model"):
        raise ValueError(f"bot_prices must be 'espn' or 'model'; got {bot_prices!r}")
    bot_dollars: pd.Series | None = None
    if bot_prices == "espn":
        if not has_usable_espn_prices(pool):
            warnings.warn(
                "bot_prices='espn' but pool has no usable espn_auction_dollars; "
                "falling back to model (shared-value) bot pricing.",
                stacklevel=2,
            )
        else:
            try:
                bot_dollars = espn_anchored_bot_prices(
                    pool, config, model_values=baseline_dollars, unranked_discount=unranked_discount
                )
            except ValueError as exc:
                warnings.warn(
                    f"espn_anchored_bot_prices failed ({exc}); falling back to model pricing.",
                    stacklevel=2,
                )
                bot_dollars = None

    per: dict[str, dict[str, np.ndarray]] = {
        name: {m: np.empty(n_seeds, dtype=np.float64) for m in METRICS} for name in strategies
    }
    for name, strat in strategies.items():
        for s in range(n_seeds):
            league = simulate_auction(
                strat,
                my_seat,
                pool,
                config,
                baseline_dollars=baseline_dollars,
                price_jitter=price_jitter,
                rng=np.random.default_rng(base_seed + s),
                snake_rng=np.random.default_rng([base_seed + s, _SNAKE_SUBSTREAM]),
                nomination_temp=nomination_temp,
                bot_archetypes=bot_archetypes,
                bot_dollars=bot_dollars,
            )
            proj = project_draft(
                league,
                pool,
                availability,
                params,
                league_config=config,
                n_sims=n_sims,
                rng=np.random.default_rng(season_base_seed + s),  # CRN: shared across strategies
            )
            sp = proj[my_seat]
            for m in METRICS:
                per[name][m][s] = float(getattr(sp, m))

    summaries = {
        name: {m: bootstrap_mean(per[name][m], seed=base_seed) for m in METRICS}
        for name in strategies
    }
    paired: dict[str, dict[str, Interval]] = {}
    for a, b in combinations(strategies, 2):
        paired[f"{a}_vs_{b}"] = {
            m: bootstrap_mean(per[a][m] - per[b][m], seed=base_seed) for m in METRICS
        }

    return AuctionTournamentResult(
        summaries=summaries,
        paired_diffs=paired,
        n_seeds=n_seeds,
        price_jitter=price_jitter,
        base_seed=base_seed,
        season_base_seed=season_base_seed,
        n_sims=n_sims,
        my_seat=my_seat,
        budget=config.budget,
        min_bid=config.min_bid,
    )
