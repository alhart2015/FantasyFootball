"""Preseason backtest harness — walk-forward eval over target seasons.

Produces:
- A PreseasonBacktestSchema-validated CSV at reports/backtest_preseason_<model>.csv.
- A human-readable markdown report at reports/backtest_preseason_<model>.md.

See `docs/superpowers/specs/2026-05-17-preseason-projections-design.md` §7.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from projections.preseason.project import project_preseason
from projections.schemas import (
    Position,
    PreseasonBacktestSchema,
    Ruleset,
)
from projections.scoring import scoring_coefficients
from projections.store import read_partition

logger = logging.getLogger(__name__)


def compute_rmse_and_spearman(
    *,
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    top_n: int,
) -> tuple[float, float, int]:
    """Compute RMSE on full inner-join + Spearman on top-N actuals.

    `predicted` requires columns: `gsis_id`, `season_total_fpts_mean`.
    `actual` requires columns: `gsis_id`, `actual_season_total_fpts`.

    Returns:
        (rmse, spearman_top_n, n_players). If the inner join is empty, returns
        (NaN, NaN, 0). If fewer than 2 top-N rows, Spearman is NaN.
    """
    merged = predicted.merge(actual, on="gsis_id", how="inner")
    if merged.empty:
        return float("nan"), float("nan"), 0

    err = merged["season_total_fpts_mean"] - merged["actual_season_total_fpts"]
    rmse = float(np.sqrt((err**2).mean()))

    top_actual = merged.nlargest(top_n, "actual_season_total_fpts")
    if len(top_actual) < 2:
        spearman = float("nan")
    else:
        rho, _ = spearmanr(
            top_actual["actual_season_total_fpts"].to_numpy(),
            top_actual["season_total_fpts_mean"].to_numpy(),
        )
        spearman = float(rho)
    return rmse, spearman, len(merged)


def determine_verdict(
    *, rmse_delta_pct: float, spearman_top50: float
) -> Literal["ADOPT", "NULL", "DO_NOT_ADOPT"]:
    """Apply the per-cell verdict logic from spec §7.3.

    - ADOPT       if rmse_delta_pct < 0 AND spearman_top50 >= 0.70
    - DO_NOT_ADOPT if rmse_delta_pct >= 0 OR spearman_top50 < 0.50
    - NULL         otherwise
    """
    if rmse_delta_pct < 0 and spearman_top50 >= 0.70:
        return "ADOPT"
    if rmse_delta_pct >= 0 or spearman_top50 < 0.50:
        return "DO_NOT_ADOPT"
    return "NULL"


def _aggregate_actuals(
    weekly_stats: pd.DataFrame, *, ruleset: Ruleset, season: int
) -> pd.DataFrame:
    """Aggregate weekly_stats[season=Y] to per-player actual season-total fpts.

    Returns: DataFrame with columns gsis_id, position, actual_season_total_fpts.
    """
    season_rows = weekly_stats.loc[weekly_stats["season"] == season].copy()
    if season_rows.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.Series([], dtype="string[pyarrow]"),
                "position": pd.Series([], dtype="string[pyarrow]"),
                "actual_season_total_fpts": pd.Series([], dtype="float32"),
            }
        )

    coef = scoring_coefficients(ruleset)
    fpts = pd.Series(0.0, index=season_rows.index, dtype="float64")
    for stat, val in coef.items():
        col = stat.value
        if col in season_rows.columns:
            fpts = fpts + season_rows[col].fillna(0).astype("float64") * val

    season_rows["weekly_fpts"] = fpts
    agg = (
        season_rows.groupby(["gsis_id", "position"], as_index=False)["weekly_fpts"]
        .sum()
        .rename(columns={"weekly_fpts": "actual_season_total_fpts"})
    )
    agg["actual_season_total_fpts"] = (
        agg["actual_season_total_fpts"].clip(lower=0).astype("float32")
    )
    return agg


def walk_forward_backtest(
    *,
    raw_root: Path,
    projections_root: Path,
    target_seasons: list[int],
    train_start: int,
    ruleset: Ruleset,
) -> pd.DataFrame:
    """Run walk-forward eval over `target_seasons`.

    For each target_season:
      1. project_preseason(target_season) using train_start..target_season-1.
      2. Aggregate weekly_stats[season=target_season] to actuals.
      3. Inner-join, compute per-(position) RMSE + Spearman + coverage diff.
      4. Compute rmse_naive_baseline (prior_1 per-game * 16).
      5. Build verdict.

    Returns a PreseasonBacktestSchema-validated frame.
    """
    rows: list[dict[str, object]] = []
    for target_season in target_seasons:
        projections = project_preseason(
            raw_root=raw_root,
            projections_root=projections_root,
            target_season=target_season,
            train_start=train_start,
            ruleset=ruleset,
        )

        actuals_weekly = read_partition(raw_root, "weekly_stats", season=target_season)
        actuals_weekly["season"] = actuals_weekly["season"].astype("int32")
        actuals = _aggregate_actuals(actuals_weekly, ruleset=ruleset, season=target_season)

        # Naive baseline: prior_1_per_game * 16 from target_season-1 actuals.
        prior_weekly = read_partition(raw_root, "weekly_stats", season=target_season - 1)
        naive_actuals = _aggregate_actuals(prior_weekly, ruleset=ruleset, season=target_season - 1)
        # games_played from prior season for per-game * 16 conversion
        prior_games = prior_weekly.groupby("gsis_id")["week"].count()
        naive_actuals["games_played"] = (
            naive_actuals["gsis_id"].map(prior_games).fillna(1).astype("float64")
        )
        naive_actuals["season_total_fpts_mean"] = (
            naive_actuals["actual_season_total_fpts"].astype("float64")
            / naive_actuals["games_played"]
            * 16
        ).astype("float32")
        naive_actuals = naive_actuals[["gsis_id", "position", "season_total_fpts_mean"]]

        for position in (Position.QB, Position.RB, Position.WR, Position.TE):
            pred_pos = projections.loc[projections["position"] == position.value]
            actual_pos = actuals.loc[actuals["position"] == position.value]
            naive_pos = naive_actuals.loc[naive_actuals["position"] == position.value]

            rmse, spearman_top50, n_players = compute_rmse_and_spearman(
                predicted=pred_pos, actual=actual_pos, top_n=50
            )
            rmse_naive, _, _ = compute_rmse_and_spearman(
                predicted=naive_pos, actual=actual_pos, top_n=50
            )
            rmse_delta_pct = (
                float("nan")
                if rmse_naive == 0 or np.isnan(rmse_naive)
                else (rmse - rmse_naive) / rmse_naive * 100
            )

            projected_not_played = int(
                len(pred_pos) - len(pred_pos.merge(actual_pos, on="gsis_id", how="inner"))
            )
            played_not_projected = int(
                len(actual_pos) - len(actual_pos.merge(pred_pos, on="gsis_id", how="inner"))
            )

            verdict: str = "NULL"
            if not np.isnan(rmse_delta_pct) and not np.isnan(spearman_top50):
                verdict = determine_verdict(
                    rmse_delta_pct=rmse_delta_pct, spearman_top50=spearman_top50
                )

            rows.append(
                {
                    "target_season": target_season,
                    "position": position.value,
                    "model_class": "naive-preseason-v1",
                    "ruleset": ruleset.name,
                    "rmse": float(rmse) if not np.isnan(rmse) else 0.0,
                    "rmse_naive_baseline": (float(rmse_naive) if not np.isnan(rmse_naive) else 0.0),
                    "rmse_delta_pct": (
                        float(rmse_delta_pct) if not np.isnan(rmse_delta_pct) else 0.0
                    ),
                    "spearman_top50": (
                        float(spearman_top50) if not np.isnan(spearman_top50) else 0.0
                    ),
                    "n_players": n_players,
                    "coverage_diff_projected_not_played": projected_not_played,
                    "coverage_diff_played_not_projected": played_not_projected,
                    "verdict": verdict,
                }
            )

    out = pd.DataFrame(rows)
    out = out.astype(
        {
            "target_season": "int32",
            "rmse": "float32",
            "rmse_naive_baseline": "float32",
            "rmse_delta_pct": "float32",
            "spearman_top50": "float32",
            "n_players": "Int64",
            "coverage_diff_projected_not_played": "Int64",
            "coverage_diff_played_not_projected": "Int64",
        }
    )
    out["model_class"] = out["model_class"].astype("string[pyarrow]")
    out = PreseasonBacktestSchema.validate(out)
    return out
