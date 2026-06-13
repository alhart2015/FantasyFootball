"""Score a weekly_stats frame to per-(gsis_id, week) half-PPR actual points (weeks 1-17)."""

from __future__ import annotations

import pandas as pd

from projections.schemas import _PYARROW_STR, Ruleset, WeeklyActualSchema
from projections.scoring.score import StatLine, score

_MAX_WEEK = 17


def build_weekly_actuals(weekly_stats: pd.DataFrame, *, ruleset: Ruleset) -> pd.DataFrame:
    """Return a validated WeeklyActualSchema frame with one row per (gsis_id, week).

    Weeks outside 1-17 are dropped before scoring. Accepts an empty or
    all-week-18 input and returns a zero-row frame with correct dtypes.
    """
    ws = weekly_stats[weekly_stats["week"] <= _MAX_WEEK].copy()

    if ws.empty:
        return WeeklyActualSchema.validate(
            pd.DataFrame(
                {
                    "gsis_id": pd.array([], dtype=_PYARROW_STR),
                    "season": pd.array([], dtype="Int64"),
                    "week": pd.array([], dtype="Int64"),
                    "actual_points": pd.array([], dtype="Float64"),
                }
            )
        )

    points: list[float] = []
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
        points.append(score(line, ruleset))

    out = pd.DataFrame(
        {
            "gsis_id": ws["gsis_id"].astype(_PYARROW_STR).to_numpy(),
            "season": ws["season"].astype("Int64").to_numpy(),
            "week": ws["week"].astype("Int64").to_numpy(),
            "actual_points": pd.array(points, dtype="Float64"),
        }
    )
    return WeeklyActualSchema.validate(out)
