"""Sum per-row stat lines to per-(gsis_id, position) season actual fantasy points
under a Ruleset. Used by retrospective comparison and diagnostic scripts.

Pure function: takes a weekly_stats frame in, returns a season-totals frame out.
Lives in projections.scoring because it's a fantasy-points-scoring helper, not
script glue.
"""

from __future__ import annotations

import pandas as pd

from projections.schemas import Ruleset
from projections.scoring.score import StatLine, score


def actual_season_total(weekly_stats: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    """Score each row's StatLine, sum per (gsis_id, position) -> season total.

    Returns one row per (gsis_id, position) with columns:
      - gsis_id: str
      - position: str (raw from input)
      - actual_total: float (sum of weekly scored points under `ruleset`)
      - actual_n_weeks: int (distinct weeks present for that player-position)

    The output column name `actual_total` is preserved from the legacy inline
    helper at scripts/compare_predictions_to_actuals.py so downstream code that
    merges on it doesn't break.
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
    ws["actual_points_per_week"] = points
    return ws.groupby(["gsis_id", "position"], as_index=False).agg(
        actual_total=("actual_points_per_week", "sum"),
        actual_n_weeks=("week", "nunique"),
    )
