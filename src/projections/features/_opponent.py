"""Opponent-strength proxy: average fantasy points allowed to a position
over a trailing window. v1 substitute for true opponent-adjusted EPA
(which would need play-by-play ingest, deferred to a later plan)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from projections.schemas import Position, Ruleset
from projections.scoring.score import StatLine, score


def _row_to_statline(row: pd.Series[Any]) -> StatLine:
    """Build a StatLine from a weekly_stats row. Defaults to 0 for any
    field not present in weekly_stats (e.g., 2pt conversions, return_tds,
    which the foundations-era schema doesn't track)."""
    return StatLine(
        passing_yards=float(row.get("passing_yards", 0.0) or 0.0),
        passing_tds=int(row.get("passing_tds", 0) or 0),
        interceptions=int(row.get("interceptions", 0) or 0),
        rushing_yards=float(row.get("rushing_yards", 0.0) or 0.0),
        rushing_tds=int(row.get("rushing_tds", 0) or 0),
        receptions=int(row.get("receptions", 0) or 0),
        receiving_yards=float(row.get("receiving_yards", 0.0) or 0.0),
        receiving_tds=int(row.get("receiving_tds", 0) or 0),
        fumbles_lost=int(row.get("fumbles_lost", 0) or 0),
    )


def opp_allowed_fppg(
    weekly_stats: pd.DataFrame,
    *,
    position: Position,
    ruleset: Ruleset,
    n_weeks: int,
) -> pd.DataFrame:
    """For each `(opp_team, season, week)`, the mean fantasy points allowed
    to `position` over the trailing `n_weeks`.

    Returns a DataFrame with columns `(season, week, opp_team, opp_allowed_fppg)`,
    where `week` is the week being scored against (NOT included in the
    trailing window). Joining onto a feature row uses `(season, week, opponent)`
    on the offense side to retrieve the opponent's allowed-points proxy.
    """
    pos_stats = weekly_stats[weekly_stats["position"] == position.value].copy()
    if pos_stats.empty:
        return pd.DataFrame(columns=["season", "week", "opp_team", "opp_allowed_fppg"]).astype(
            {"season": int, "week": int, "opp_allowed_fppg": float}
        )

    # Score each per-game line.
    pos_stats["fpts"] = pos_stats.apply(lambda r: score(_row_to_statline(r), ruleset), axis=1)

    # Sum per (opp_team, season, week) — that's all `position`-players' points
    # allowed by `opp_team` in that week.
    weekly_allowed = (
        pos_stats.groupby(["opponent", "season", "week"], as_index=False)["fpts"]
        .sum()
        .rename(columns={"opponent": "opp_team"})
    )

    # Trailing-N mean per opp_team, BUT the result is associated with the NEXT
    # week (the one where the opponent will face this defense).
    # Approach: keep the trailing window, then shift the resulting mean to
    # week+1 of the same season.
    rows: list[dict[str, object]] = []
    for (opp_team, season), g in weekly_allowed.groupby(["opp_team", "season"], sort=False):
        g_sorted = g.sort_values("week").reset_index(drop=True)
        for i in range(len(g_sorted)):
            window = g_sorted.iloc[max(0, i - n_weeks + 1) : i + 1]
            mean_fppg = float(window["fpts"].mean())
            target_week = int(g_sorted.iloc[i]["week"]) + 1
            rows.append(
                {
                    "season": int(season),
                    "week": target_week,
                    "opp_team": opp_team,
                    "opp_allowed_fppg": mean_fppg,
                }
            )

    return pd.DataFrame(rows, columns=["season", "week", "opp_team", "opp_allowed_fppg"])
