"""Compare draft strategies empirically (spec §3.5-§3.6).

Run each strategy over many seeded drafts against an ADP field, score the hero
roster via a `RosterValuer` (default: optimal starting lineup), and declare a
winner on the paired per-seed difference (percentile bootstrap, mirroring
adoption_gate.py). The same seed index gives every strategy the same bot field
-- the paired counterfactual. `tune_sigma` sweeps the survival sigma the same
way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.assistant._compare import (
    Interval as Interval,
)
from projections.draft.assistant._compare import (
    bootstrap_mean,
    validate_pool_size,
)
from projections.draft.assistant.simulation import simulate_draft
from projections.draft.assistant.strategy import DraftStrategy, NowOrNeverStrategy
from projections.draft.assistant.survival import LogisticSurvival
from projections.draft.assistant.valuer import RosterValuer, StartersValuer
from projections.draft.league_config import LeagueConfig

# Private alias kept for backward-compat with backtest and test modules that import it.
_bootstrap_mean = bootstrap_mean


@dataclass(frozen=True)
class TournamentResult:
    """Per-strategy mean roster value (per the valuer), the top-two paired diff, the winner."""

    summaries: dict[str, Interval]
    diff: Interval | None  # top-vs-second paired difference; None if <2 strategies
    winner: str | None  # top strategy iff diff.lo_95 > 0; the sole strategy if only one supplied
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
    validate_pool_size(pool, config)
    if "consensus_adp" not in pool.columns or bool(pool["consensus_adp"].isna().all()):
        raise ValueError(
            "pool has no consensus_adp signal; the tournament needs market ADP to drive the field"
        )


def _validate_run_params(
    config: LeagueConfig, *, my_slot: int, n_seeds: int, adp_jitter: float
) -> None:
    """Guard the run parameters both entry points share.

    An out-of-range my_slot would never own a snake pick, so the hero drafts
    nobody, scores 0, and the harness would report a confidently-wrong verdict.
    A negative adp_jitter would crash deep in numpy (`scale < 0`); 0 is allowed
    (a deterministic, zero-noise field).
    """
    if not 1 <= my_slot <= config.n_teams:
        raise ValueError(f"my_slot must be in 1..{config.n_teams}; got {my_slot}")
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1; got {n_seeds}")
    if adp_jitter < 0:
        raise ValueError(f"adp_jitter must be >= 0; got {adp_jitter}")


def _strategy_values(
    strategy: DraftStrategy,
    pool: pd.DataFrame,
    config: LeagueConfig,
    *,
    my_slot: int,
    n_seeds: int,
    adp_jitter: float,
    base_seed: int,
    valuer: RosterValuer,
) -> np.ndarray:
    """Roster value (per the valuer) of the hero roster for each paired seed."""
    out = np.empty(n_seeds, dtype=np.float64)
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        roster = simulate_draft(strategy, my_slot, pool, config, adp_jitter=adp_jitter, rng=rng)
        out[s] = valuer.value(roster, config.roster_slots)
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
    valuer: RosterValuer | None = None,
) -> TournamentResult:
    """Compare `strategies` over `n_seeds` paired drafts; declare a winner."""
    _validate_pool(pool, config)
    _validate_run_params(config, my_slot=my_slot, n_seeds=n_seeds, adp_jitter=adp_jitter)
    valuer = valuer if valuer is not None else StartersValuer()
    values = {
        name: _strategy_values(
            strat,
            pool,
            config,
            my_slot=my_slot,
            n_seeds=n_seeds,
            adp_jitter=adp_jitter,
            base_seed=base_seed,
            valuer=valuer,
        )
        for name, strat in strategies.items()
    }
    summaries = {name: _bootstrap_mean(v, seed=base_seed) for name, v in values.items()}
    ranked = sorted(summaries, key=lambda n: summaries[n].point, reverse=True)

    diff: Interval | None = None
    winner: str | None = ranked[0] if ranked else None
    if len(ranked) >= 2:
        top, second = ranked[0], ranked[1]
        diff = _bootstrap_mean(values[top] - values[second], seed=base_seed)
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
    valuer: RosterValuer | None = None,
) -> SigmaTuningResult:
    """Sweep the survival sigma for NowOrNeverStrategy; return the (sigma, mean) grid + argmax."""
    _validate_pool(pool, config)
    _validate_run_params(config, my_slot=my_slot, n_seeds=n_seeds, adp_jitter=adp_jitter)
    if not sigma_grid:
        raise ValueError("sigma_grid must be non-empty")
    if any(s <= 0 for s in sigma_grid):
        # LogisticSurvival requires sigma > 0; reject the whole grid up front rather
        # than mid-loop at strategy construction (protects every caller, not just the CLI).
        raise ValueError(f"sigma_grid values must all be > 0; got {list(sigma_grid)}")
    valuer = valuer if valuer is not None else StartersValuer()
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
            valuer=valuer,
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
