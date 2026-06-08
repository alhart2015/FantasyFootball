# scripts/benchmark_projections.py
"""Spike: benchmark our BaselineModel preseason projection vs ESPN's preseason
projection at predicting actual 2024 fantasy outcomes. Emits a verdict report.

Inputs:
  - data/external_projections/{season}/espn.parquet   (from pull_external_projections.py)
  - data/external_projections/{season}/sleeper_adp.parquet
  - reports/season_projection_{season}.csv            (from project_season.py --out)
  - data/raw weekly_stats + id_map                    (in-house)

Output:
  - reports/external_projection_benchmark_{season}.md

Preseason-vs-preseason only. Every stat line is scored through OUR PPR ruleset so
the comparison is under one scoring rule. Pure transforms are unit-tested; the
end-to-end run is a manual phase.

Usage:
    python scripts/benchmark_projections.py --season 2024
"""

from __future__ import annotations

import pandas as pd

from projections.schemas import Ruleset
from projections.scoring.score import StatLine, score

_STAT_FIELDS = (
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
)
_COUNT_FIELDS = frozenset(
    {
        "passing_tds",
        "interceptions",
        "rushing_tds",
        "receptions",
        "receiving_tds",
        "fumbles_lost",
    }
)


def _score_row(row: pd.Series[object], ruleset: Ruleset) -> float:
    sl = StatLine(
        passing_yards=float(row["passing_yards"]),
        passing_tds=round(float(row["passing_tds"])),
        interceptions=round(float(row["interceptions"])),
        rushing_yards=float(row["rushing_yards"]),
        rushing_tds=round(float(row["rushing_tds"])),
        receptions=round(float(row["receptions"])),
        receiving_yards=float(row["receiving_yards"]),
        receiving_tds=round(float(row["receiving_tds"])),
        fumbles_lost=round(float(row["fumbles_lost"])),
    )
    return score(sl, ruleset)


def actual_season_points(weekly_stats: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    """Sum each player's weekly stat lines to a season total and score under `ruleset`.
    Position is the modal value across the player's weeks."""
    agg = {f: "sum" for f in _STAT_FIELDS}
    summed = weekly_stats.groupby("gsis_id", as_index=False).agg(agg)
    pos = weekly_stats.groupby("gsis_id")["position"].agg(lambda s: s.mode().iloc[0]).reset_index()
    out = summed.merge(pos, on="gsis_id", how="left")
    out["actual_pts"] = out.apply(lambda r: _score_row(r, ruleset), axis=1)
    return out[["gsis_id", "position", "actual_pts"]]


def our_season_points(csv_df: pd.DataFrame) -> pd.DataFrame:
    """Our model's CSV: season_total_mean is already PPR fantasy points."""
    out = csv_df[["gsis_id", "position", "season_total_mean"]].copy()
    out = out.rename(columns={"season_total_mean": "our_pts"})
    return out
