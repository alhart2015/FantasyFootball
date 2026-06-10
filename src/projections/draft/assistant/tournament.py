"""Compare draft strategies empirically (spec §3.5-§3.6).

Run each strategy over many seeded drafts against an ADP field, score the hero
roster by its optimal starting lineup, and declare a winner on the paired
per-seed difference (percentile bootstrap, mirroring adoption_gate.py). The same
seed index gives every strategy the same bot field -- the paired counterfactual.
`tune_sigma` sweeps the survival sigma the same way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.assistant.roster_score import optimal_lineup_points
from projections.draft.assistant.simulation import simulate_draft
from projections.draft.assistant.strategy import DraftStrategy, NowOrNeverStrategy
from projections.draft.assistant.survival import LogisticSurvival
from projections.draft.league_config import LeagueConfig

_N_BOOTSTRAP = 1000
_CI_PCTILES = (2.5, 97.5)


@dataclass(frozen=True)
class Interval:
    """A point estimate with a central 95% bootstrap CI."""

    point: float
    lo_95: float
    hi_95: float


@dataclass(frozen=True)
class TournamentResult:
    """Per-strategy mean starting-lineup points, the top-two paired diff, the winner."""

    summaries: dict[str, Interval]
    diff: Interval | None  # top-vs-second paired difference; None if <2 strategies
    winner: str | None  # named iff diff.lo_95 > 0 (CI excludes 0)
    n_seeds: int
    adp_jitter: float
    base_seed: int
    my_slot: int


@dataclass(frozen=True)
class SigmaTuningResult:
    """(sigma, mean hero value) grid + the argmax."""

    grid: list[tuple[float, float]]
    best_sigma: float
    n_seeds: int
    adp_jitter: float
    base_seed: int
    my_slot: int


def _validate_pool(pool: pd.DataFrame, config: LeagueConfig) -> None:
    """Hard preconditions shared by both entry points (spec §3.1, §3.3)."""
    if "consensus_adp" not in pool.columns or bool(pool["consensus_adp"].isna().all()):
        raise ValueError(
            "pool has no consensus_adp signal; the tournament needs market ADP to drive the field"
        )
    need = config.n_teams * config.roster_size
    if len(pool) < need:
        raise ValueError(f"pool has {len(pool)} players; need >= {need} to fill a full draft")


def _bootstrap(values: np.ndarray, *, seed: int) -> Interval:
    """Percentile-bootstrap CI of the mean of `values`."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = v.shape[0]
    boot = np.empty(_N_BOOTSTRAP, dtype=np.float64)
    for b in range(_N_BOOTSTRAP):
        boot[b] = v[rng.integers(0, n, size=n)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return Interval(point=float(v.mean()), lo_95=float(lo), hi_95=float(hi))


def _paired_diff_ci(
    a: np.ndarray, b: np.ndarray, *, n_bootstrap: int = _N_BOOTSTRAP, seed: int
) -> Interval:
    """Percentile-bootstrap CI of the paired mean difference `a - b`."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = d.shape[0]
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for b_i in range(n_bootstrap):
        boot[b_i] = d[rng.integers(0, n, size=n)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return Interval(point=float(d.mean()), lo_95=float(lo), hi_95=float(hi))


def _strategy_values(
    strategy: DraftStrategy,
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    my_slot: int,
    n_seeds: int,
    adp_jitter: float,
    base_seed: int,
) -> np.ndarray:
    """Optimal-lineup points of the hero roster for each paired seed."""
    out = np.empty(n_seeds, dtype=np.float64)
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        roster = simulate_draft(strategy, my_slot, pool, config, adp_jitter=adp_jitter, rng=rng)
        out[s] = optimal_lineup_points(roster, config.roster_slots)
    return out


def run_tournament(
    strategies: Mapping[str, DraftStrategy],
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    my_slot: int,
    n_seeds: int,
    adp_jitter: float,
    base_seed: int,
) -> TournamentResult:
    """Compare `strategies` over `n_seeds` paired drafts; declare a winner."""
    _validate_pool(pool, config)
    values = {
        name: _strategy_values(
            strat,
            pool,
            config,
            my_slot=my_slot,
            n_seeds=n_seeds,
            adp_jitter=adp_jitter,
            base_seed=base_seed,
        )
        for name, strat in strategies.items()
    }
    summaries = {name: _bootstrap(v, seed=base_seed) for name, v in values.items()}
    ranked = sorted(summaries, key=lambda n: summaries[n].point, reverse=True)

    diff: Interval | None = None
    winner: str | None = ranked[0] if ranked else None
    if len(ranked) >= 2:
        top, second = ranked[0], ranked[1]
        diff = _paired_diff_ci(values[top], values[second], seed=base_seed)
        winner = top if diff.lo_95 > 0 else None

    return TournamentResult(
        summaries=summaries,
        diff=diff,
        winner=winner,
        n_seeds=n_seeds,
        adp_jitter=adp_jitter,
        base_seed=base_seed,
        my_slot=my_slot,
    )


def tune_sigma(
    sigma_grid: Sequence[float],
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    my_slot: int,
    n_seeds: int,
    adp_jitter: float,
    base_seed: int,
) -> SigmaTuningResult:
    """Sweep the survival sigma for NowOrNeverStrategy; return the (sigma, mean) grid + argmax."""
    _validate_pool(pool, config)
    grid: list[tuple[float, float]] = []
    for sigma in sigma_grid:
        strat = NowOrNeverStrategy(LogisticSurvival(sigma=float(sigma)))
        vals = _strategy_values(
            strat,
            pool,
            config,
            my_slot=my_slot,
            n_seeds=n_seeds,
            adp_jitter=adp_jitter,
            base_seed=base_seed,
        )
        grid.append((float(sigma), float(vals.mean())))
    best_sigma = max(grid, key=lambda r: r[1])[0]
    return SigmaTuningResult(
        grid=grid,
        best_sigma=best_sigma,
        n_seeds=n_seeds,
        adp_jitter=adp_jitter,
        base_seed=base_seed,
        my_slot=my_slot,
    )
