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


def trailing_4_per_player(weekly_stats: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-player frame with ``mean_l4`` = mean of ``value_col`` over the
    trailing 4 games. Players with 0 prior games are simply absent — the
    caller fills their value with 0.0 after the merge.

    Hardcoded N=4 because every position builder uses the same window
    today; if that ever needs to vary, change the signature and add an
    ``n`` keyword (call sites also depend on the literal ``mean_l4``
    column name in their ``.rename(columns=...)`` calls)."""
    if weekly_stats.empty:
        # Local import avoids any future cycle if schemas grows to import
        # from _rolling — schemas does not currently import from here.
        from projections.schemas import _PYARROW_STR

        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "mean_l4": pd.array([], dtype=float),
            }
        )
    last4 = last_n_per_group(
        weekly_stats,
        group_cols=["gsis_id"],
        sort_cols=["season", "week"],
        n=4,
    )
    return (
        last4.groupby("gsis_id", as_index=False, observed=True)[value_col]
        .mean()
        .rename(columns={value_col: "mean_l4"})
    )


def latest_ngs_snapshot(ngs: pd.DataFrame) -> pd.DataFrame:
    """Per-player most-recent NGS row. The caller is responsible for
    filtering ``ngs`` to leakage-safe rows before calling — this helper
    just picks the latest (season, week) per ``gsis_id``."""
    if ngs.empty:
        return pd.DataFrame()
    return (
        ngs.sort_values(["season", "week"])
        .groupby("gsis_id", as_index=False, observed=True)
        .tail(1)
        .copy()
    )


def trailing_n_share_in_group(
    weekly_stats: pd.DataFrame,
    *,
    value_col: str,
    n: int = 4,
) -> pd.DataFrame:
    """Per-player share of ``value_col`` within their team over the trailing N games.

    Numerator: each player's trailing-N sum of ``value_col``.
    Denominator: sum across all players in ``weekly_stats`` on the same team
    (over the same trailing-N windows).

    Returns a frame keyed by ``gsis_id`` with column ``share_l<n>`` (``share_l4``
    when n=4, the default).

    The caller controls the share-group by pre-filtering ``weekly_stats``:

    - WR target_share among the team's WRs: filter input to ``position == WR``.
    - RB target_share among the team's pass-catchers: filter input to
      ``position in {WR, RB, TE}``, then keep only the RB rows from the output.
    - RB rush_share among the team's RBs: filter input to ``position == RB``.
    """
    out_col = f"share_l{n}"
    if weekly_stats.empty:
        # Schema-friendly empty frame so callers can merge without dtype churn.
        # Local import avoids any future cycle if schemas grows to import from
        # _rolling — schemas does not currently import from here.
        from projections.schemas import _PYARROW_STR

        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                out_col: pd.array([], dtype=float),
            }
        )
    last_n_player = last_n_per_group(
        weekly_stats,
        group_cols=["gsis_id"],
        sort_cols=["season", "week"],
        n=n,
    )
    player_sum = last_n_player.groupby(["gsis_id", "team"], as_index=False, observed=True)[
        value_col
    ].sum()
    team_sum = (
        player_sum.groupby("team", as_index=False, observed=True)[value_col]
        .sum()
        .rename(columns={value_col: "team_total"})
    )
    merged = player_sum.merge(team_sum, on="team", how="left")
    merged[out_col] = (merged[value_col].astype(float) / merged["team_total"].astype(float)).where(
        merged["team_total"] > 0, 0.0
    )
    return merged[["gsis_id", out_col]]
