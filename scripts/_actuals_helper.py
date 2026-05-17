"""Shared helper: convert a weekly_stats frame to per-(gsis_id, position) actual
fantasy-point totals under a Ruleset. Lifted from scripts/compare_predictions_to_actuals.py
so both that script and scripts/diagnose_upside_ranking.py can call it.

Public name: actual_ppr_total (dropped the leading underscore since the function
is now cross-script).
"""

from __future__ import annotations

import pandas as pd

from projections.schemas import Ruleset
from projections.scoring import score
from projections.scoring.score import StatLine


def actual_ppr_total(weekly_stats: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    """Per-row fantasy points, summed per (gsis_id, position) -> actual season total.

    Returns one row per (gsis_id, position) with columns:
      - gsis_id: str
      - position: str (raw from input)
      - actual_total: float (sum of weekly scored points under `ruleset`)
      - actual_n_weeks: int (distinct weeks present for that player-position)
    """
    points: list[float] = []
    for _, row in weekly_stats.iterrows():
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
    ws = weekly_stats.copy()
    ws["actual_ppr"] = points
    return ws.groupby(["gsis_id", "position"], as_index=False).agg(
        actual_total=("actual_ppr", "sum"),
        actual_n_weeks=("week", "nunique"),
    )
