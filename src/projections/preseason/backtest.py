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

from projections.preseason.model import (
    NaivePreseasonModel,
    NaivePriorOnlyModel,
    PreseasonModel,
)
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
    model_under_test: PreseasonModel | None = None,
    baseline_model: PreseasonModel | None = None,
) -> pd.DataFrame:
    """Walk-forward eval over `target_seasons`.

    `model_under_test` defaults to `NaivePreseasonModel` (the v1.0 production
    model — what actually goes into the canonical partition write).
    `baseline_model` defaults to `NaivePriorOnlyModel` and provides the
    comparison floor that gates the per-cell verdict.

    For v1.0, model_under_test == NaivePreseasonModel and baseline_model
    == NaivePriorOnlyModel; the resulting `rmse_delta_pct` measures the diff
    between "full naive with rookies + multi-tier prior fallback" and "strict
    veteran prior_1 only". For v1.5+, swap `model_under_test` for the trained
    model and keep the same NaivePriorOnlyModel floor.

    Both models run through the same `project_preseason` pipeline; the
    canonical partition is written for `model_under_test` only — the baseline
    runs with `write_partition=False` so its in-memory frame doesn't clobber
    the disk artifact.

    Returns a PreseasonBacktestSchema-validated frame.
    """
    rows: list[dict[str, object]] = []
    mut = model_under_test if model_under_test is not None else NaivePreseasonModel()
    baseline = baseline_model if baseline_model is not None else NaivePriorOnlyModel()
    model_class_id = mut.model_id

    for target_season in target_seasons:
        # Production projection — written to disk.
        projections = project_preseason(
            raw_root=raw_root,
            projections_root=projections_root,
            target_season=target_season,
            train_start=train_start,
            ruleset=ruleset,
            model=mut,
            write_partition=True,
        )
        # Baseline projection — in-memory only; floor for the verdict gate.
        baseline_projections = project_preseason(
            raw_root=raw_root,
            projections_root=projections_root,
            target_season=target_season,
            train_start=train_start,
            ruleset=ruleset,
            model=baseline,
            write_partition=False,
        )

        actuals_weekly = read_partition(raw_root, "weekly_stats", season=target_season)
        actuals_weekly["season"] = actuals_weekly["season"].astype("int32")
        actuals = _aggregate_actuals(actuals_weekly, ruleset=ruleset, season=target_season)

        for position in (Position.QB, Position.RB, Position.WR, Position.TE):
            pred_pos = projections.loc[projections["position"] == position.value]
            base_pos = baseline_projections.loc[baseline_projections["position"] == position.value]
            actual_pos = actuals.loc[actuals["position"] == position.value]

            rmse, spearman_top50, n_players = compute_rmse_and_spearman(
                predicted=pred_pos, actual=actual_pos, top_n=50
            )
            rmse_naive, _, _ = compute_rmse_and_spearman(
                predicted=base_pos, actual=actual_pos, top_n=50
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
                    "model_class": model_class_id,
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


def write_backtest_report(backtest_df: pd.DataFrame, path: Path) -> None:
    """Render a PreseasonBacktestSchema frame as a markdown report.

    Spec §7.6 calls for additional per-position top-20 spot-check tables and
    player-name coverage sidebars; those are deferred to v1.1. v1.0 ships the
    verdict tables + per-cell metrics + coverage-diff counts, which is the
    gate-relevant minimum.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Preseason Backtest Report",
        "",
        f"**Model class:** {backtest_df['model_class'].iloc[0]}  ",
        f"**Ruleset:** {backtest_df['ruleset'].iloc[0]}  ",
        f"**Target seasons:** {sorted(set(backtest_df['target_season'].tolist()))}  ",
        "",
        "## Per-cell verdicts",
        "",
        "| target_season | position | rmse | rmse_naive | rmse_delta_pct | spearman_top50 | n_players | verdict |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in backtest_df.iterrows():
        lines.append(
            f"| {row['target_season']} | {row['position']} | {row['rmse']:.2f} | "
            f"{row['rmse_naive_baseline']:.2f} | {row['rmse_delta_pct']:+.2f}% | "
            f"{row['spearman_top50']:.3f} | {row['n_players']} | "
            f"**{row['verdict']}** |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- ADOPT cells:        {(backtest_df['verdict'] == 'ADOPT').sum()}",
            f"- NULL cells:         {(backtest_df['verdict'] == 'NULL').sum()}",
            f"- DO_NOT_ADOPT cells: {(backtest_df['verdict'] == 'DO_NOT_ADOPT').sum()}",
            "",
            "## Coverage diff",
            "",
            "| target_season | position | projected_not_played | played_not_projected |",
            "|---|---|---|---|",
        ]
    )
    for _, row in backtest_df.iterrows():
        lines.append(
            f"| {row['target_season']} | {row['position']} | "
            f"{row['coverage_diff_projected_not_played']} | "
            f"{row['coverage_diff_played_not_projected']} |"
        )
    path.write_text("\n".join(lines) + "\n")
