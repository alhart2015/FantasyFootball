"""Phase 1 diagnostic for TODO #33d. Reads weekly-distribution parquet +
distributions CSV from project_season.py output + actuals from data/raw/weekly_stats,
computes ranking under four metrics (mean / p90 / blend_70_30 / p_elite), and
writes a markdown report with a Phase-2-decision verdict.

See docs/superpowers/specs/2026-05-16-upside-sensitive-ranking-diagnostic-design.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

from projections.schemas import Position, Ruleset
from projections.scoring import actual_season_total
from projections.store import read_partition


def _compute_elite_thresholds(
    *,
    raw_root: Path,
    seasons: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023),
    ruleset: Ruleset,
    min_games: int = 8,
) -> dict[Position, float]:
    """Per-position elite threshold = mean over `seasons` of the 5th-highest
    actual season fantasy points at that position, computed over players with
    >= min_games games played that season."""
    per_season_top5: dict[Position, list[float]] = {
        p: [] for p in (Position.QB, Position.RB, Position.WR, Position.TE)
    }
    for season in seasons:
        ws = read_partition(raw_root, "weekly_stats", season=season)
        actuals = actual_season_total(ws, ruleset)
        actuals = actuals[actuals["actual_n_weeks"] >= min_games]
        for pos in per_season_top5:
            pos_rows = actuals[actuals["position"] == pos.value].sort_values(
                "actual_total", ascending=False
            )
            if len(pos_rows) >= 5:
                per_season_top5[pos].append(float(pos_rows["actual_total"].iloc[4]))
    out: dict[Position, float] = {}
    for pos, vals in per_season_top5.items():
        if not vals:
            raise ValueError(
                f"No seasons in {seasons} produced >= 5 players with "
                f">= {min_games} games at {pos.value}"
            )
        out[pos] = sum(vals) / len(vals)
    return out


def top_k_overlap(pred_rank: pd.Series, actual_rank: pd.Series, *, k: int) -> float:
    """|predicted_top_k ∩ actual_top_k| / k. Ranks are 1-based, smallest = best."""
    pred_top = set(pred_rank.nsmallest(k).index)
    actual_top = set(actual_rank.nsmallest(k).index)
    return len(pred_top & actual_top) / k


def top5_rank_err(pred_rank: pd.Series, actual_rank: pd.Series) -> float:
    """For each player in actual top-5: median(|predicted_rank - actual_rank|)."""
    top5 = actual_rank.nsmallest(5).index
    return float((pred_rank.loc[top5] - actual_rank.loc[top5]).abs().median())


def kendall_tau_filtered(
    pred_score: pd.Series,
    actual_score: pd.Series,
    n_weeks: pd.Series,
    *,
    min_n_weeks: int,
) -> tuple[float, int]:
    """Kendall's tau over players with n_weeks >= min_n_weeks. Returns (tau, n)."""
    eligible = n_weeks[n_weeks >= min_n_weeks].index
    pred_e = pred_score.loc[eligible]
    actual_e = actual_score.loc[eligible]
    result = kendalltau(pred_e.to_numpy(), actual_e.to_numpy())
    return float(result.statistic), len(eligible)
