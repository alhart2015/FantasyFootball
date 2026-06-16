"""Shared is_rookie attachment: a player is a 'rookie' (for the variance model's higher-variance
tier) if they have no prior-season weekly_stats appearance. Used by the backtest input loader and
the live-board projected eval."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.store import read_partition


def prior_appearance_gsis(season: int, data_root: Path, *, since: int = 2018) -> set[str]:
    """gsis_ids appearing in any weekly_stats season in [since, season); missing partitions skip."""
    seen: set[str] = set()
    for yr in range(since, season):
        try:
            ws = read_partition(data_root / "raw", "weekly_stats", season=yr)
        except (FileNotFoundError, ValueError):
            continue
        seen.update(ws["gsis_id"].astype(str).tolist())
    return seen


def attach_is_rookie(pool: pd.DataFrame, *, season: int, data_root: Path) -> pd.DataFrame:
    """Return a copy of `pool` with a boolean `is_rookie` column (True = no prior appearance)."""
    out = pool.copy()
    prior = prior_appearance_gsis(season, data_root)
    out["is_rookie"] = ~out["gsis_id"].astype(str).isin(prior)
    return out


__all__ = ["attach_is_rookie", "prior_appearance_gsis"]
