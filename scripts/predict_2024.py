"""Plan 3b -- write 2024 weekly projections for a specified position to
data/projections/weekly/.

Replaces scripts/predict_2024_wr.py.

Usage:
    python scripts/predict_2024.py {qb|rb|te|wr}

Loads the trained artifact (baseline-{pos}-...joblib), builds features
for each week of 2024, predicts, and writes one parquet partition per
(season, week) using store.write_partition. Validated against
ProjectionWeeklySchema.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from projections.models import POSITION_DISPATCH, BaselineModel
from projections.schemas import Position, ProjectionWeeklySchema, Ruleset
from projections.store import read_partition, write_partition

_PROJECTION_SEASON = 2024


def _find_artifact(artifacts_root: Path, position: Position) -> Path:
    pattern = f"baseline-{position.value.lower()}-*.joblib"
    matches = sorted(artifacts_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No {pattern} in {artifacts_root}. "
            f"Run scripts/train_baseline.py {position.value.lower()} first."
        )
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict 2024 weekly projections for a position.")
    parser.add_argument("position", choices=["qb", "rb", "te", "wr"], help="Target position.")
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

    artifact = _find_artifact(args.artifacts_root, position)
    print(f"Loading artifact: {artifact}")
    model = BaselineModel.load(artifact)

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
