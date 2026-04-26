"""Walk-forward backtest driver.

For each (position, year) in the cartesian product, train Model A on
cached features for [train_start, year-1], predict every week of `year`
from cached features, score against actuals from data/raw/weekly_stats,
and return a BacktestRun with model + naive metrics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from projections.backtest.metrics import compute_all_metrics
from projections.backtest.naive import compute_naive_predictions
from projections.features.cache import read_features
from projections.models import POSITION_DISPATCH
from projections.schemas import Position, Ruleset, Stat
from projections.scoring import INTEGER_STATS, score
from projections.scoring.score import StatLine
from projections.store import read_partition

_METRICS_COLUMNS: tuple[str, ...] = ("position", "year", "metric", "value")


@dataclass(frozen=True, slots=True)
class BacktestRun:
    """Result of a single walk-forward backtest invocation.

    Attributes:
        timestamp: UTC time the run started; used to name diagnostic
            output directories under data/backtest/run_<ts>/.
        metrics: long-form DataFrame with columns
            (position, year, metric, value) -- the model's metrics across
            (position, year, metric) cells. Becomes the snapshot input.
        naive_metrics: same shape; computed alongside model metrics for
            informational reporting. Not gated.
        per_row_results: per-(position, year, week, gsis_id) row of
            actuals + model predictions for diagnosis. Plan 3c writes
            this to data/backtest/run_<ts>/results.parquet (gitignored).
    """

    timestamp: pd.Timestamp
    metrics: pd.DataFrame
    naive_metrics: pd.DataFrame
    per_row_results: pd.DataFrame


def _realized_ppr_points(weekly_stats: pd.DataFrame, ruleset: Ruleset) -> pd.Series:
    """Compute realized PPR points per row of weekly_stats. Mirrors
    scripts/sanity_check_baseline.py's helper of the same name."""
    points: list[float] = []
    for _idx, row in weekly_stats.iterrows():
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
    return pd.Series(points, index=weekly_stats.index, name="actual_ppr")


def _build_eval_df(
    *,
    predictions: pd.DataFrame,
    per_stat_pred_means: pd.DataFrame,
    held_out_pos: pd.DataFrame,
    target_stats: tuple[Stat, ...],
) -> pd.DataFrame:
    """Inner-join model preds + per-stat predicted means + actuals on
    (gsis_id, season, week). Result has {stat}_pred / {stat}_actual
    columns, mean / p10 / p90, and actual_ppr."""
    keep = ["gsis_id", "season", "week", "actual_ppr"] + [s.value for s in target_stats]
    pred_with_per_stat = predictions.merge(
        per_stat_pred_means, on=["gsis_id", "season", "week"], how="left"
    )
    return pred_with_per_stat.merge(
        held_out_pos[keep],
        on=["gsis_id", "season", "week"],
        how="inner",
        suffixes=("_pred", "_actual"),
    )


def _naive_metrics_for_cell(
    *,
    train_actuals: pd.DataFrame,
    holdout_actuals: pd.DataFrame,
    position: Position,
    target_stats: tuple[Stat, ...],
    held_out_year: int,
    ruleset: Ruleset,
) -> dict[str, float]:
    """Compute the naive per-stat predictions, build an eval_df-shaped
    frame, and run compute_all_metrics on it."""
    naive_per_stat = compute_naive_predictions(
        train_actuals=train_actuals,
        holdout_actuals=holdout_actuals,
        position=position,
        target_stats=target_stats,
        held_out_year=held_out_year,
    )
    holdout_pos = holdout_actuals[holdout_actuals["position"] == position.value].copy()
    holdout_pos["actual_ppr"] = _realized_ppr_points(holdout_pos, ruleset)

    # Naive composite point prediction: feed naive per-stat into score().
    naive_composite: list[float] = []
    for _idx, row in naive_per_stat.iterrows():
        kwargs_clean: dict[str, float | int] = {}
        for stat in target_stats:
            v = row[stat.value]
            if stat in INTEGER_STATS:
                kwargs_clean[stat.value] = round(float(v))
            else:
                kwargs_clean[stat.value] = float(v)
        line = StatLine(**kwargs_clean)  # type: ignore[arg-type]
        naive_composite.append(score(line, ruleset))
    naive_per_stat = naive_per_stat.copy()
    naive_per_stat["mean"] = naive_composite
    # p10/p90 are not meaningful for a point baseline; populate with mean
    # so calibration metrics return 1.0 (informational only -- never gated).
    naive_per_stat["p10"] = naive_per_stat["mean"]
    naive_per_stat["p90"] = naive_per_stat["mean"]

    eval_df = _build_eval_df(
        predictions=naive_per_stat[["gsis_id", "season", "week", "mean", "p10", "p90"]],
        per_stat_pred_means=naive_per_stat.drop(columns=["mean", "p10", "p90"]),
        held_out_pos=holdout_pos,
        target_stats=target_stats,
    )
    return compute_all_metrics(eval_df, target_stats=target_stats)


def _model_metrics_for_cell(
    *,
    train_features: pd.DataFrame,
    train_actuals: pd.DataFrame,
    predict_features: pd.DataFrame,
    holdout_actuals: pd.DataFrame,
    position: Position,
    ruleset: Ruleset,
) -> tuple[dict[str, float], pd.DataFrame, tuple[Stat, ...]]:
    """Train BaselineModel on train_features+train_actuals, predict each
    week of predict_features, score against holdout_actuals. Returns
    (metrics_dict, eval_df, target_stats) -- the third tuple element lets
    the caller pass target_stats to the naive computation without
    rebuilding a model instance."""
    dispatch = POSITION_DISPATCH[position]
    model = dispatch.factory()
    model.fit(train_features, train_actuals)

    predictions = model.predict_distribution(predict_features, ruleset=ruleset)
    stat_dists_per_row = model.build_stat_distributions(predict_features)
    per_stat_pred_means = pd.DataFrame(
        {stat.value: [d[stat].mean() for d in stat_dists_per_row] for stat in model.target_stats}
    )
    per_stat_pred_means["gsis_id"] = predict_features["gsis_id"].values
    per_stat_pred_means["season"] = predict_features["season"].astype(int).values
    per_stat_pred_means["week"] = predict_features["week"].astype(int).values

    holdout_pos = holdout_actuals[holdout_actuals["position"] == position.value].copy()
    holdout_pos["actual_ppr"] = _realized_ppr_points(holdout_pos, ruleset)

    target_stats = tuple(model.target_stats)
    eval_df = _build_eval_df(
        predictions=predictions,
        per_stat_pred_means=per_stat_pred_means,
        held_out_pos=holdout_pos,
        target_stats=target_stats,
    )
    metrics = compute_all_metrics(eval_df, target_stats=target_stats)
    return metrics, eval_df, target_stats


def run_backtest(
    *,
    held_out_years: Iterable[int] = (2021, 2022, 2023, 2024),
    positions: Iterable[Position] | None = None,
    train_start: int = 2018,
    features_root: Path = Path("data/features"),
    raw_root: Path = Path("data/raw"),
    ruleset: Ruleset | None = None,
) -> BacktestRun:
    """Walk-forward backtest. Spec section 2.3."""
    if ruleset is None:
        ruleset = Ruleset.espn_ppr()
    if positions is None:
        positions = (Position.QB, Position.RB, Position.TE, Position.WR)

    timestamp = pd.Timestamp(datetime.now(UTC))
    positions_list = list(positions)
    years_list = list(held_out_years)

    metrics_rows: list[dict[str, object]] = []
    naive_rows: list[dict[str, object]] = []
    per_row_frames: list[pd.DataFrame] = []

    for position in positions_list:
        for year in years_list:
            train_seasons = list(range(train_start, year))
            train_features = pd.concat(
                [read_features(position, s, features_root=features_root) for s in train_seasons],
                ignore_index=True,
            )
            train_actuals = pd.concat(
                [read_partition(raw_root, "weekly_stats", season=s) for s in train_seasons],
                ignore_index=True,
            )
            predict_features = read_features(position, year, features_root=features_root)
            holdout_actuals = read_partition(raw_root, "weekly_stats", season=year)

            model_metrics, eval_df, target_stats = _model_metrics_for_cell(
                train_features=train_features,
                train_actuals=train_actuals,
                predict_features=predict_features,
                holdout_actuals=holdout_actuals,
                position=position,
                ruleset=ruleset,
            )
            for metric_name, value in model_metrics.items():
                metrics_rows.append(
                    {
                        "position": position.value,
                        "year": year,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )

            naive_metrics = _naive_metrics_for_cell(
                train_actuals=train_actuals,
                holdout_actuals=holdout_actuals,
                position=position,
                target_stats=target_stats,
                held_out_year=year,
                ruleset=ruleset,
            )
            for metric_name, value in naive_metrics.items():
                naive_rows.append(
                    {
                        "position": position.value,
                        "year": year,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )

            eval_df = eval_df.assign(position=position.value)
            per_row_frames.append(eval_df)

    metrics_df = pd.DataFrame(metrics_rows, columns=list(_METRICS_COLUMNS))
    naive_metrics_df = pd.DataFrame(naive_rows, columns=list(_METRICS_COLUMNS))
    per_row_results = (
        pd.concat(per_row_frames, ignore_index=True) if per_row_frames else pd.DataFrame()
    )
    return BacktestRun(
        timestamp=timestamp,
        metrics=metrics_df,
        naive_metrics=naive_metrics_df,
        per_row_results=per_row_results,
    )
