"""Walk-forward backtest driver.

For each (position, year, model_class) in the cartesian product, train
the selected model class on cached features for [train_start, year-1],
predict every week of `year` from cached features, score against actuals
from data/raw/weekly_stats, and return a BacktestRun with per-model
metrics + a single naive baseline per (position, year).

Plan 5 Task 12: the inner per-fold loop iterates over `model_classes`
(default ``("baseline",)`` for backward compat). Per-row metrics +
per-row results gain a ``model_class`` column so callers can
filter/aggregate by which model class produced the prediction.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from projections.aggregation import aggregate_to_season
from projections.backtest.metrics import (
    compute_all_metrics,
    compute_season_calibration_metrics,
)
from projections.backtest.naive import compute_naive_predictions
from projections.distributions import unpack_per_stat_params
from projections.features.cache import read_features
from projections.models import (
    POSITION_DISPATCH,
    BaselineModel,
    DecomposedBaselineModel,
    EnsembleModel,
    LightGBMModel,
)
from projections.schemas import DistributionFamily, Position, Ruleset, Stat
from projections.scoring import INTEGER_STATS, score
from projections.scoring.score import StatLine
from projections.store import read_partition

# Plan 5 Task 12: per-row metrics rows now carry `model_class`.
_METRICS_COLUMNS: tuple[str, ...] = ("position", "year", "metric", "model_class", "value")
# Naive baseline is model-class-agnostic (it's a fixed point predictor
# computed from train_actuals); naive rows remain 4-column.
_NAIVE_COLUMNS: tuple[str, ...] = ("position", "year", "metric", "value")


@dataclass(frozen=True, slots=True)
class BacktestRun:
    """Result of a single walk-forward backtest invocation.

    Attributes:
        timestamp: UTC time the run started; used to name diagnostic
            output directories under data/backtest/run_<ts>/.
        metrics: long-form DataFrame with columns
            (position, year, metric, model_class, value) -- the model's
            metrics across (position, year, metric, model_class) cells.
            Becomes the snapshot input.
        naive_metrics: long-form DataFrame with columns
            (position, year, metric, value); computed once per
            (position, year) regardless of model_classes selected.
            Informational; not gated.
        per_row_results: per-(position, year, week, gsis_id, model_class)
            row of actuals + model predictions for diagnosis. Plan 3c
            writes this to data/backtest/run_<ts>/results.parquet
            (gitignored).
        per_player_results: per-(position, year, gsis_id, model_class)
            season eval row for diagnosis. Plan 3d writes this to
            data/backtest/run_<ts>/season_results.parquet (gitignored).
            NOTE: only populated for baseline today; aggregate_to_season
            currently requires DistributionFamily.SAMPLED_SUMMARY and
            LightGBMModel emits QUANTILE — Plan 5 Task 18 to file a
            follow-up TODO for widening the season aggregator.
    """

    timestamp: pd.Timestamp
    metrics: pd.DataFrame
    naive_metrics: pd.DataFrame
    per_row_results: pd.DataFrame
    per_player_results: pd.DataFrame


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


def _per_stat_means_from_predictions(
    predictions: pd.DataFrame,
    *,
    target_stats: tuple[Stat, ...],
) -> pd.DataFrame:
    """Decode predictions[params] back to per-stat distributions and emit a
    DataFrame of per-stat predicted means keyed on (gsis_id, season, week).

    Model-class-agnostic: works for BaselineModel (SAMPLED_SUMMARY family,
    parametric per-stat) and LightGBMModel (QUANTILE family, per-stat
    quantile vectors) uniformly via the codec round-trip. Replaces the
    BaselineModel-only ``model.build_stat_distributions(...)`` call that
    Task 11 had as a bridge.
    """
    rows: list[dict[str, float | int | str]] = []
    for _idx, pred_row in predictions.iterrows():
        per_stat_dists = unpack_per_stat_params(bytes(pred_row["params"]))
        out: dict[str, float | int | str] = {
            "gsis_id": str(pred_row["gsis_id"]),
            "season": int(pred_row["season"]),
            "week": int(pred_row["week"]),
        }
        for stat in target_stats:
            out[stat.value] = float(per_stat_dists[stat].mean())
        rows.append(out)
    return pd.DataFrame(rows)


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


def run_backtest(
    *,
    held_out_years: Iterable[int] = (2021, 2022, 2023, 2024),
    positions: Iterable[Position] | None = None,
    train_start: int = 2018,
    model_classes: Iterable[str] = ("baseline",),
    features_root: Path = Path("data/features"),
    raw_root: Path = Path("data/raw"),
    ruleset: Ruleset | None = None,
) -> BacktestRun:
    """Walk-forward backtest. Spec section 2.3.

    Args:
        model_classes: which model classes to run per fold. Defaults to
            ``("baseline",)`` so legacy callers keep their behavior. Plan
            5 adds ``"lightgbm"``; ``("baseline", "lightgbm")`` runs both
            side by side and tags every output row with ``model_class``.
    """
    if ruleset is None:
        ruleset = Ruleset.espn_ppr()
    if positions is None:
        positions = (Position.QB, Position.RB, Position.TE, Position.WR)

    timestamp = pd.Timestamp(datetime.now(UTC))
    positions_list = list(positions)
    years_list = list(held_out_years)
    model_classes_list = list(model_classes)

    metrics_rows: list[dict[str, object]] = []
    naive_rows: list[dict[str, object]] = []
    per_row_frames: list[pd.DataFrame] = []
    per_player_frames: list[pd.DataFrame] = []

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

            holdout_pos = holdout_actuals[holdout_actuals["position"] == position.value].copy()
            holdout_pos["actual_ppr"] = _realized_ppr_points(holdout_pos, ruleset)

            dispatch = POSITION_DISPATCH[position]

            # Track naive computation: target_stats is identical across
            # model classes (per position), so we compute naive metrics
            # once per cell using the first model's target_stats.
            naive_target_stats: tuple[Stat, ...] | None = None

            for model_class in model_classes_list:
                # Plan 5: factories return the Model protocol; narrow to
                # the union of concrete classes so we can read
                # `target_stats` (not on the Model protocol).
                model = cast(
                    BaselineModel | DecomposedBaselineModel | LightGBMModel | EnsembleModel,
                    dispatch.factories[model_class](),
                )
                model.fit(train_features, train_actuals)
                predictions = model.predict_distribution(predict_features, ruleset=ruleset)
                target_stats = tuple(model.target_stats)
                if naive_target_stats is None:
                    naive_target_stats = target_stats

                # Per-stat predicted means via codec round-trip --
                # model-class-agnostic (replaces BaselineModel-only
                # build_stat_distributions in Task 11's bridge).
                per_stat_pred_means = _per_stat_means_from_predictions(
                    predictions, target_stats=target_stats
                )

                eval_df = _build_eval_df(
                    predictions=predictions,
                    per_stat_pred_means=per_stat_pred_means,
                    held_out_pos=holdout_pos,
                    target_stats=target_stats,
                )
                model_metrics = compute_all_metrics(eval_df, target_stats=target_stats)
                for metric_name, value in model_metrics.items():
                    metrics_rows.append(
                        {
                            "position": position.value,
                            "year": year,
                            "metric": metric_name,
                            "model_class": model_class,
                            "value": float(value),
                        }
                    )

                # Season aggregation: today only the SAMPLED_SUMMARY
                # family (BaselineModel) round-trips through
                # aggregate_to_season. LightGBMModel emits QUANTILE; Plan
                # 5 Task 18 to file a follow-up TODO for widening the
                # season aggregator. Skip the season-calibration rows
                # for any other family rather than crash mid-loop.
                if (predictions["family"] == DistributionFamily.SAMPLED_SUMMARY.value).all():
                    season_predictions = aggregate_to_season(predictions, ruleset=ruleset)
                    season_actuals = (
                        holdout_pos.groupby("gsis_id", as_index=False)["actual_ppr"]
                        .sum()
                        .rename(columns={"actual_ppr": "actual_season_total"})
                    )
                    season_eval_df = season_predictions.merge(
                        season_actuals, on="gsis_id", how="inner"
                    )
                    season_metrics = compute_season_calibration_metrics(season_eval_df)
                    for metric_name, value in season_metrics.items():
                        metrics_rows.append(
                            {
                                "position": position.value,
                                "year": year,
                                "metric": metric_name,
                                "model_class": model_class,
                                "value": float(value),
                            }
                        )
                    season_eval_df = season_eval_df.assign(
                        position=position.value, model_class=model_class
                    )
                    per_player_frames.append(season_eval_df)

                eval_df = eval_df.assign(position=position.value, model_class=model_class)
                per_row_frames.append(eval_df)

            # Naive metrics (independent of model_class). Computed once
            # per (position, year). target_stats is identical across
            # model classes per position so the choice is irrelevant.
            assert naive_target_stats is not None, "model_classes must be non-empty"
            naive_metrics = _naive_metrics_for_cell(
                train_actuals=train_actuals,
                holdout_actuals=holdout_actuals,
                position=position,
                target_stats=naive_target_stats,
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

    metrics_df = pd.DataFrame(metrics_rows, columns=list(_METRICS_COLUMNS))
    naive_metrics_df = pd.DataFrame(naive_rows, columns=list(_NAIVE_COLUMNS))
    per_row_results = (
        pd.concat(per_row_frames, ignore_index=True) if per_row_frames else pd.DataFrame()
    )
    per_player_results = (
        pd.concat(per_player_frames, ignore_index=True) if per_player_frames else pd.DataFrame()
    )
    return BacktestRun(
        timestamp=timestamp,
        metrics=metrics_df,
        naive_metrics=naive_metrics_df,
        per_row_results=per_row_results,
        per_player_results=per_player_results,
    )
