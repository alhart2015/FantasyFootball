"""Trailing-window helpers used by feature builders.

All functions are pure: they don't mutate inputs and don't perform I/O.
Designed for the WR builder in 2a but applicable to every position builder
in 2b — the contract here is the load-bearing one."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def last_n_per_group(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    sort_cols: Sequence[str],
    n: int,
) -> pd.DataFrame:
    """Return only the last ``n`` rows per group, sorted by ``sort_cols``.

    Used to compute trailing-N-game statistics: group by player, take the
    last 4 entries by (season, week), then mean a stat column.

    Caller order doesn't matter — we sort internally.
    """
    if df.empty:
        return df.copy()
    group_cols_l = list(group_cols)
    sort_cols_l = list(sort_cols)
    return (
        df.sort_values(group_cols_l + sort_cols_l)
        .groupby(group_cols_l, as_index=False, sort=False)
        .tail(n)
        .copy()
    )


def per_game_rate(df: pd.DataFrame, *, num_col: str, denom_col: str) -> pd.Series:
    """Safe division: ``df[num_col] / df[denom_col]``.

    Zeros and NaN denominators yield ``0.0`` (instead of ``inf`` or ``NaN``).
    """
    num = df[num_col].fillna(0).astype(float)
    denom = df[denom_col].fillna(0).astype(float)
    return pd.Series(
        [n / d if d > 0 else 0.0 for n, d in zip(num, denom, strict=True)],
        index=df.index,
        dtype=float,
    )


def season_to_date_mean(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    sort_cols: Sequence[str],
    value_col: str,
) -> pd.Series:
    """Running per-group mean of ``value_col`` within each season.

    Returns a Series aligned to the input row index (after internal sort).
    ``group_cols`` MUST include season so the running mean resets at season
    boundaries — that's the caller's responsibility, not the helper's.
    """
    if df.empty:
        return pd.Series([], dtype=float)
    group_cols_l = list(group_cols)
    sorted_df = df.sort_values(group_cols_l + list(sort_cols))
    result = (
        sorted_df.groupby(group_cols_l, sort=False)[value_col]
        .expanding()
        .mean()
        .reset_index(level=list(range(len(group_cols_l))), drop=True)
    )
    return result
