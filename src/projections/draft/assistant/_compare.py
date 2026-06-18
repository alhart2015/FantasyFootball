"""Mechanism-agnostic comparison helpers shared by the snake and auction tournaments.

Promoted from tournament.py so both harnesses bootstrap and size-validate identically.
The snake tournament keeps its ADP-specific arms and winner-labeling on top of these;
the auction harness records metrics without declaring a winner (spec §5.1, §5.7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.draft.league_config import LeagueConfig

_N_BOOTSTRAP = 1000
_CI_PCTILES = (2.5, 97.5)


@dataclass(frozen=True)
class Interval:
    """A point estimate with a central 95% bootstrap CI."""

    point: float
    lo_95: float
    hi_95: float


def bootstrap_mean(values: np.ndarray, *, n_bootstrap: int = _N_BOOTSTRAP, seed: int) -> Interval:
    """Percentile-bootstrap CI of the mean of `values` (pass `a - b` for a paired diff)."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = v.shape[0]
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        boot[b] = v[rng.integers(0, n, size=n)].mean()
    lo, hi = np.percentile(boot, _CI_PCTILES)
    return Interval(point=float(v.mean()), lo_95=float(lo), hi_95=float(hi))


def validate_pool_size(pool: pd.DataFrame, config: LeagueConfig) -> None:
    """Mechanism-agnostic pool-size precondition: enough players to fill every roster spot."""
    need = config.n_teams * config.roster_size
    if len(pool) < need:
        raise ValueError(f"pool has {len(pool)} players; need >= {need} to fill a full draft")
