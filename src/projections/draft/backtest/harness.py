"""Run simulate_league over many seeds (mirrored seat layout).

Aggregates per-strategy rates + bootstrap CIs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.strategy import (
    DraftStrategy,
    NowOrNeverStrategy,
    SeasonValueStrategy,
)
from projections.draft.assistant.survival import LogisticSurvival, default_sigma
from projections.draft.assistant.tournament import Interval, _bootstrap_mean
from projections.draft.backtest.draft_field import seat_layout
from projections.draft.backtest.league import Calendar, LeagueResult, simulate_league
from projections.draft.league_config import LeagueConfig


@dataclass(frozen=True)
class StrategyMetrics:
    championship: Interval
    playoff: Interval
    win_pct: Interval
    points_for: Interval


@dataclass(frozen=True)
class BacktestResult:
    by_strategy_actual: dict[str, StrategyMetrics]
    by_strategy_projected: dict[str, StrategyMetrics]
    n_seeds: int


def collect_results(
    *,
    seed_lo: int,
    seed_hi: int,
    pool: pd.DataFrame,
    config: LeagueConfig,
    availability: PlayerAvailability,
    proj_lookup: Mapping[tuple[str, int], float],
    actual_lookup: Mapping[tuple[str, int], float],
    calendar: Calendar,
    jitter: float = 8.0,
    strategy_n_sims: int = 200,
    base_seed: int = 0,
) -> tuple[list[LeagueResult], list[LeagueResult]]:
    """Simulate seed indices [seed_lo, seed_hi) and return raw (actual, projected) results.

    Seed index ``s`` uses ``seat_layout(s)`` and league seed ``base_seed + s`` — identical to
    the slice run_backtest would produce for those indices, so disjoint chunks pooled in order
    reconstruct a monolithic run exactly. This is the unit the resumable chunk-runner serializes.
    """
    sigma = default_sigma(config.n_teams)
    nn = NowOrNeverStrategy(LogisticSurvival(sigma=sigma))
    sv = SeasonValueStrategy(availability, n_sims=strategy_n_sims, base_seed=base_seed)
    label_to_strategy: dict[str, DraftStrategy | None] = {
        "now_or_never": nn,
        "season_value": sv,
        "bot": None,
    }
    results_actual: list[LeagueResult] = []
    results_projected: list[LeagueResult] = []
    for s in range(seed_lo, seed_hi):
        layout = seat_layout(s)
        seat_strategies = {seat: label_to_strategy[label] for seat, label in layout.items()}
        outcome = simulate_league(
            base_seed + s,
            seat_strategies=seat_strategies,
            strategy_labels=layout,
            pool=pool,
            config=config,
            proj_lookup=proj_lookup,
            actual_lookup=actual_lookup,
            calendar=calendar,
            jitter=jitter,
        )
        results_actual += outcome.actual
        results_projected += outcome.projected
    return results_actual, results_projected


def aggregate(
    results_actual: list[LeagueResult],
    results_projected: list[LeagueResult],
    *,
    n_seeds: int,
    base_seed: int = 0,
) -> BacktestResult:
    """Aggregate raw per-seed results into per-strategy bootstrap metrics for both scorings."""

    def _metrics(results: list[LeagueResult], label: str) -> StrategyMetrics:
        rs = [r for r in results if r.strategy == label]
        champ = np.array([1.0 if r.is_champion else 0.0 for r in rs])
        playoff = np.array([1.0 if r.made_playoffs else 0.0 for r in rs])
        winp = np.array([r.wins / (r.wins + r.losses) for r in rs])
        pf = np.array([r.points_for for r in rs])
        return StrategyMetrics(
            championship=_bootstrap_mean(champ, seed=base_seed),
            playoff=_bootstrap_mean(playoff, seed=base_seed),
            win_pct=_bootstrap_mean(winp, seed=base_seed),
            points_for=_bootstrap_mean(pf, seed=base_seed),
        )

    labels = ("now_or_never", "season_value", "bot")

    def _table(results: list[LeagueResult]) -> dict[str, StrategyMetrics]:
        return {label: _metrics(results, label) for label in labels}

    return BacktestResult(
        by_strategy_actual=_table(results_actual),
        by_strategy_projected=_table(results_projected),
        n_seeds=n_seeds,
    )


def run_backtest(
    *,
    n_seeds: int,
    pool: pd.DataFrame,
    config: LeagueConfig,
    availability: PlayerAvailability,
    proj_lookup: Mapping[tuple[str, int], float],
    actual_lookup: Mapping[tuple[str, int], float],
    calendar: Calendar,
    jitter: float = 8.0,
    strategy_n_sims: int = 200,
    base_seed: int = 0,
) -> BacktestResult:
    """Run all n_seeds in one process and aggregate. See collect_results for chunked runs."""
    results_actual, results_projected = collect_results(
        seed_lo=0,
        seed_hi=n_seeds,
        pool=pool,
        config=config,
        availability=availability,
        proj_lookup=proj_lookup,
        actual_lookup=actual_lookup,
        calendar=calendar,
        jitter=jitter,
        strategy_n_sims=strategy_n_sims,
        base_seed=base_seed,
    )
    return aggregate(results_actual, results_projected, n_seeds=n_seeds, base_seed=base_seed)
