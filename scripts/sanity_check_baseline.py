"""Plan 3b -- sanity-check eval of Model A baseline (or Plan 5 Model C
lightgbm) for a position against the held-out 2024 season. Stdout-only;
not a CI gate (Plan 3c builds the backtest harness with thresholds).

Replaces scripts/sanity_check_wr_baseline.py.

Usage (after train_baseline.py {pos} [--model lightgbm]):
    python scripts/sanity_check_baseline.py {qb|rb|te|wr} [--model {baseline|lightgbm}]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from projections.models import POSITION_DISPATCH, BaselineModel, LightGBMModel
from projections.schemas import Position, Ruleset
from projections.scoring import score
from projections.scoring.score import StatLine
from projections.store import read_partition

_HELD_OUT_SEASON = 2024


def _find_artifact(artifacts_root: Path, position: Position, model_class: str) -> Path:
    pattern = f"{model_class}-{position.value.lower()}-*.joblib"
    matches = sorted(artifacts_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No {pattern} in {artifacts_root}. "
            f"Run scripts/train_baseline.py {position.value.lower()} "
            f"--model {model_class} first."
        )
    return matches[-1]


def _realized_ppr_points(weekly_stats: pd.DataFrame, ruleset: Ruleset) -> pd.Series:
    """Compute realized PPR points per row of weekly_stats."""
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
    return pd.Series(points, index=weekly_stats.index, name="actual_ppr")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check Model A baseline or Model C lightgbm on 2024 for a position."
    )
    parser.add_argument("position", choices=["qb", "rb", "te", "wr"], help="Target position.")
    parser.add_argument(
        "--model",
        choices=["baseline", "lightgbm"],
        default="baseline",
        help="Which model class artifact to load (Model A or Model C). Default baseline.",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("models/artifacts"))
    args = parser.parse_args()

    position = Position(args.position.upper())
    dispatch = POSITION_DISPATCH[position]
    builder = dispatch.feature_builder
    ngs_kwarg = {
        "passing": "ngs_passing",
        "rushing": "ngs_rushing",
        "receiving": "ngs_receiving",
    }[dispatch.ngs_stat_type]
    ngs_table = f"ngs_{dispatch.ngs_stat_type}"

    artifact = _find_artifact(args.artifacts_root, position, args.model)
    print(f"Loading artifact: {artifact}")
    model: BaselineModel | LightGBMModel
    if args.model == "baseline":
        model = BaselineModel.load(artifact)
    else:
        model = LightGBMModel.load(artifact)
    print(f"model_id: {model.model_id}")

    raw_root = args.raw_root

    # Build features for every week of 2024 (predict-time inputs use prior
    # weeks within 2024; training data through 2023 is implicit in the fit).
    ws_held = read_partition(raw_root, "weekly_stats", season=_HELD_OUT_SEASON)
    sc_held = read_partition(raw_root, "snap_counts", season=_HELD_OUT_SEASON)
    dc_held = read_partition(raw_root, "depth_charts", season=_HELD_OUT_SEASON)
    ngs_held = read_partition(raw_root, ngs_table, season=_HELD_OUT_SEASON)
    sch_held = read_partition(raw_root, "schedules", season=_HELD_OUT_SEASON)

    # Concatenate prior season for rolling windows (one season is enough for L4).
    ws_prior = read_partition(raw_root, "weekly_stats", season=_HELD_OUT_SEASON - 1)
    sc_prior = read_partition(raw_root, "snap_counts", season=_HELD_OUT_SEASON - 1)
    ngs_prior = read_partition(raw_root, ngs_table, season=_HELD_OUT_SEASON - 1)
    ws_full = pd.concat([ws_prior, ws_held], ignore_index=True)
    sc_full = pd.concat([sc_prior, sc_held], ignore_index=True)
    ngs_full = pd.concat([ngs_prior, ngs_held], ignore_index=True)
    # Plan 9: PBP for opp-defensive EPA features. Degrade gracefully if a
    # season's partition doesn't exist (e.g., pre-Plan-9 ingest hasn't run);
    # builders accept an empty frame and emit NaN-filled values.
    pbp_frames: list[pd.DataFrame] = []
    for s in (_HELD_OUT_SEASON - 1, _HELD_OUT_SEASON):
        try:
            pbp_frames.append(read_partition(raw_root, "pbp", season=s))
        except FileNotFoundError:
            pass
    pbp_full = pd.concat(pbp_frames, ignore_index=True) if pbp_frames else pd.DataFrame()

    weeks = sorted(dc_held["week"].unique())
    rows: list[pd.DataFrame] = []
    for week in weeks:
        kwargs: dict[str, Any] = {
            "weekly_stats": ws_full,
            "snap_counts": sc_full,
            "depth_charts": dc_held,
            "schedules": sch_held,
            "season": _HELD_OUT_SEASON,
            "as_of_week": int(week),
            "pbp": pbp_full,
            ngs_kwarg: ngs_full,
        }
        feats = builder(**kwargs)
        if feats.empty:
            continue
        preds = model.predict_distribution(feats, ruleset=Ruleset.espn_ppr())
        if isinstance(model, BaselineModel):
            # Per-stat point predictions for fit metrics. BaselineModel exposes
            # `build_stat_distributions`; LightGBMModel does not (per-stat
            # quantile boosters live behind `_sub_models`). The composite
            # PPR / calibration metrics below still run for lightgbm; only
            # the per-stat fit table is skipped.
            stat_dists_per_row = model.build_stat_distributions(feats)
            per_stat_means = pd.DataFrame(
                {
                    stat.value: [d[stat].mean() for d in stat_dists_per_row]
                    for stat in model.target_stats
                }
            )
            per_stat_means["gsis_id"] = feats["gsis_id"].values
            per_stat_means["season"] = _HELD_OUT_SEASON
            per_stat_means["week"] = int(week)
            joined = preds.merge(per_stat_means, on=["gsis_id", "season", "week"], how="left")
        else:
            joined = preds
        rows.append(joined)

    all_preds = pd.concat(rows, ignore_index=True)

    # Inner-join to actual weekly stats (filter to the requested position).
    actual = ws_held[ws_held["position"] == position.value].copy()
    actual["actual_ppr"] = _realized_ppr_points(actual, Ruleset.espn_ppr())
    if isinstance(model, BaselineModel):
        keep = ["gsis_id", "season", "week", "actual_ppr"] + [s.value for s in model.target_stats]
        eval_df = all_preds.merge(
            actual[keep],
            on=["gsis_id", "season", "week"],
            how="inner",
            suffixes=("_pred", "_actual"),
        )
    else:
        # No per-stat predicted columns merged in for lightgbm; keep only the
        # composite columns from `predict_distribution` plus actual_ppr.
        eval_df = all_preds.merge(
            actual[["gsis_id", "season", "week", "actual_ppr"]],
            on=["gsis_id", "season", "week"],
            how="inner",
        )

    print(
        f"\n=== {position.value} {_HELD_OUT_SEASON} sanity check "
        f"(n={len(eval_df)} player-weeks) ==="
    )

    # Per-stat fit (BaselineModel only; lightgbm has no per-stat point-pred path here).
    if isinstance(model, BaselineModel):
        print("\n-- Per-stat fit --")
        for stat in model.target_stats:
            pred_col = f"{stat.value}_pred"
            actual_col = f"{stat.value}_actual"
            rmse = float(np.sqrt(((eval_df[pred_col] - eval_df[actual_col]) ** 2).mean()))
            mae = float((eval_df[pred_col] - eval_df[actual_col]).abs().mean())
            print(
                f"  {stat.value:>20s}  rmse={rmse:6.3f}  mae={mae:6.3f}  "
                f"mean_pred={eval_df[pred_col].mean():6.3f}  "
                f"mean_actual={eval_df[actual_col].mean():6.3f}"
            )
    else:
        print("\n-- Per-stat fit -- (skipped for lightgbm; no per-stat point-pred helper)")

    # Composite -- PPR.
    print("\n-- Composite (PPR points) --")
    rmse = float(np.sqrt(((eval_df["mean"] - eval_df["actual_ppr"]) ** 2).mean()))
    mae = float((eval_df["mean"] - eval_df["actual_ppr"]).abs().mean())
    print(f"  mean prediction:  rmse={rmse:.3f}  mae={mae:.3f}")
    # Top-N rank correlation across the entire held-out year.
    pred_rank = eval_df.groupby("gsis_id")["mean"].sum().rank()
    actual_rank = eval_df.groupby("gsis_id")["actual_ppr"].sum().rank()
    common = pred_rank.index.intersection(actual_rank.index)
    spearman = float(np.corrcoef(pred_rank.loc[common], actual_rank.loc[common])[0, 1])
    print(
        f"  top-N season-total rank correlation (Spearman, all {position.value}s): {spearman:.3f}"
    )

    # Calibration.
    print("\n-- Calibration --")
    in_p10p90 = (
        (eval_df["actual_ppr"] >= eval_df["p10"]) & (eval_df["actual_ppr"] <= eval_df["p90"])
    ).mean()
    le_p90 = (eval_df["actual_ppr"] <= eval_df["p90"]).mean()
    print(f"  fraction in [p10, p90]: {in_p10p90:.3f}  (target ~ 0.80)")
    print(f"  fraction <= p90:        {le_p90:.3f}  (target ~ 0.90)")

    print("\n=== End sanity check (informational; not a CI gate) ===")


if __name__ == "__main__":
    main()
