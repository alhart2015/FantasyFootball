"""Plan 3b -- write 2024 weekly projections for a specified position to
data/projections/weekly/.

Replaces scripts/predict_2024_wr.py.

Usage:
    python scripts/predict_2024.py {qb|rb|te|wr} [--model {baseline|lightgbm}]

Loads the trained artifact ({model}-{pos}-...joblib), builds features
for each week of 2024, predicts, and writes one parquet partition per
(season, week) using store.write_partition. Validated against
ProjectionWeeklySchema.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from projections.models import POSITION_DISPATCH, BaselineModel, LightGBMModel
from projections.schemas import Position, ProjectionWeeklySchema, Ruleset
from projections.store import read_partition, write_partition

_PROJECTION_SEASON = 2024


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict 2024 weekly projections for a position.")
    parser.add_argument("position", choices=["qb", "rb", "te", "wr"], help="Target position.")
    parser.add_argument(
        "--model",
        choices=["baseline", "lightgbm"],
        default="baseline",
        help="Which model class artifact to load (Model A or Model C). Default baseline.",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--projections-root", type=Path, default=Path("data/projections"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--ruleset", type=str, default="espn_ppr")
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

    ruleset_map = {
        "espn_ppr": Ruleset.espn_ppr(),
        "espn_half": Ruleset.espn_half(),
        "standard": Ruleset.standard(),
    }
    ruleset = ruleset_map[args.ruleset]

    raw_root = args.raw_root
    ws_prior = read_partition(raw_root, "weekly_stats", season=_PROJECTION_SEASON - 1)
    sc_prior = read_partition(raw_root, "snap_counts", season=_PROJECTION_SEASON - 1)
    ngs_prior = read_partition(raw_root, ngs_table, season=_PROJECTION_SEASON - 1)
    ws_curr = read_partition(raw_root, "weekly_stats", season=_PROJECTION_SEASON)
    sc_curr = read_partition(raw_root, "snap_counts", season=_PROJECTION_SEASON)
    dc_curr = read_partition(raw_root, "depth_charts", season=_PROJECTION_SEASON)
    ngs_curr = read_partition(raw_root, ngs_table, season=_PROJECTION_SEASON)
    sch_curr = read_partition(raw_root, "schedules", season=_PROJECTION_SEASON)

    ws_full = pd.concat([ws_prior, ws_curr], ignore_index=True)
    sc_full = pd.concat([sc_prior, sc_curr], ignore_index=True)
    ngs_full = pd.concat([ngs_prior, ngs_curr], ignore_index=True)
    # Plan 9: PBP for opp-defensive EPA features. Degrade gracefully if a
    # season's partition doesn't exist (e.g., pre-Plan-9 ingest hasn't run);
    # builders accept an empty frame and emit NaN-filled values.
    pbp_frames: list[pd.DataFrame] = []
    for s in (_PROJECTION_SEASON - 1, _PROJECTION_SEASON):
        try:
            pbp_frames.append(read_partition(raw_root, "pbp", season=s))
        except FileNotFoundError:
            pass
    pbp_full = pd.concat(pbp_frames, ignore_index=True) if pbp_frames else pd.DataFrame()

    weeks = sorted(dc_curr["week"].unique())
    rule_partition = ruleset.name  # e.g., "ESPN_PPR"
    for week in weeks:
        kwargs: dict[str, Any] = {
            "weekly_stats": ws_full,
            "snap_counts": sc_full,
            "depth_charts": dc_curr,
            "schedules": sch_curr,
            "season": _PROJECTION_SEASON,
            "as_of_week": int(week),
            "pbp": pbp_full,
            ngs_kwarg: ngs_full,
        }
        feats = builder(**kwargs)
        if feats.empty:
            print(f"  Week {week}: no rostered {position.value}s; skipping")
            continue
        preds = model.predict_distribution(feats, ruleset=ruleset)
        ProjectionWeeklySchema.validate(preds)
        target = write_partition(
            args.projections_root,
            f"weekly/ruleset={rule_partition}",
            preds,
            season=_PROJECTION_SEASON,
            week=int(week),
        )
        print(f"  Week {week}: wrote {len(preds)} rows -> {target}")

    print("Done.")


if __name__ == "__main__":
    main()
