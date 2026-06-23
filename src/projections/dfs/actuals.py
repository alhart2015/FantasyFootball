"""DraftKings-base weekly actual points (era-aware, skill positions).

A sibling of draft/backtest/weekly_actuals.build_weekly_actuals that (a) is
ruleset-parameterized for DK base, (b) uses the era-aware regular-season cutoff
(18 weeks for 2021+, not a hard week-17 cap), and (c) retains `position` for the
edge study's per-position bucketing. All point math routes through scoring.score.
"""

from __future__ import annotations

import pandas as pd

from projections.schemas import _PYARROW_STR, Position, Ruleset
from projections.scoring import dk_actuals_bonus
from projections.scoring.score import StatLine, score
from projections.season_calendar import last_regular_week

_SKILL = {p.value for p in (Position.QB, Position.RB, Position.WR, Position.TE)}


def dk_weekly_actuals(weekly_stats: pd.DataFrame, *, ruleset: Ruleset) -> pd.DataFrame:
    """One row per (gsis_id, season, week, position) of realized DK-base points
    (+ a bonus-inclusive column for the sensitivity check), regular-season weeks
    only, skill positions only."""
    ws = weekly_stats[weekly_stats["position"].isin(_SKILL)].copy()
    if not ws.empty:
        cutoff = ws["season"].map(last_regular_week)
        ws = ws[ws["week"] <= cutoff].copy()

    if ws.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "season": pd.array([], dtype="Int64"),
                "week": pd.array([], dtype="Int64"),
                "position": pd.array([], dtype=_PYARROW_STR),
                "actual_points": pd.array([], dtype="Float64"),
                "actual_points_with_bonus": pd.array([], dtype="Float64"),
            }
        )

    points: list[float] = []
    with_bonus: list[float] = []
    for _, row in ws.iterrows():
        line = StatLine(
            passing_yards=float(row["passing_yards"]),
            passing_tds=int(row["passing_tds"]),
            interceptions=int(row["interceptions"]),
            rushing_yards=float(row["rushing_yards"]),
            rushing_tds=int(row["rushing_tds"]),
            receptions=int(row["receptions"]),
            receiving_yards=float(row["receiving_yards"]),
            receiving_tds=int(row["receiving_tds"]),
            fumbles_lost=int(row["fumbles_lost"]),
        )
        base = score(line, ruleset)
        bonus = dk_actuals_bonus(
            passing_yards=float(row["passing_yards"]),
            rushing_yards=float(row["rushing_yards"]),
            receiving_yards=float(row["receiving_yards"]),
        )
        points.append(base)
        with_bonus.append(base + bonus)

    return pd.DataFrame(
        {
            "gsis_id": ws["gsis_id"].astype(_PYARROW_STR).to_numpy(),
            "season": ws["season"].astype("Int64").to_numpy(),
            "week": ws["week"].astype("Int64").to_numpy(),
            "position": ws["position"].astype(_PYARROW_STR).to_numpy(),
            "actual_points": pd.array(points, dtype="Float64"),
            "actual_points_with_bonus": pd.array(with_bonus, dtype="Float64"),
        }
    )
