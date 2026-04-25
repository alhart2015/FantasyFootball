"""Plan 3a -- write 2024 WR projections to data/projections/weekly/.

Loads the trained artifact, builds features for each week of 2024, predicts,
and writes one parquet partition per (season, week) using store.write_partition.
ProjectionWeeklySchema-validated.

Note: spec called for 2025; nfl_data_py hasn't published 2025 yet, so we
emit 2024 projections (also the held-out year used for the sanity check).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.features import build_wr_features
from projections.models import BaselineModel
from projections.schemas import ProjectionWeeklySchema, Ruleset
from projections.store import read_partition, write_partition

_PROJECTION_SEASON = 2024


def _find_artifact(artifacts_root: Path) -> Path:
    matches = sorted(artifacts_root.glob("wr-baseline-*.joblib"))
    if not matches:
        raise FileNotFoundError(f"No wr-baseline-*.joblib in {artifacts_root}.")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict 2024 WR weekly projections.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--projections-root", type=Path, default=Path("data/projections"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("models/artifacts"))
    parser.add_argument("--ruleset", type=str, default="espn_ppr")
    args = parser.parse_args()

    artifact = _find_artifact(args.artifacts_root)
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
    ngs_prior = read_partition(raw_root, "ngs_receiving", season=_PROJECTION_SEASON - 1)
    ws_curr = read_partition(raw_root, "weekly_stats", season=_PROJECTION_SEASON)
    sc_curr = read_partition(raw_root, "snap_counts", season=_PROJECTION_SEASON)
    dc_curr = read_partition(raw_root, "depth_charts", season=_PROJECTION_SEASON)
    ngs_curr = read_partition(raw_root, "ngs_receiving", season=_PROJECTION_SEASON)
    sch_curr = read_partition(raw_root, "schedules", season=_PROJECTION_SEASON)

    ws_full = pd.concat([ws_prior, ws_curr], ignore_index=True)
    sc_full = pd.concat([sc_prior, sc_curr], ignore_index=True)
    ngs_full = pd.concat([ngs_prior, ngs_curr], ignore_index=True)

    weeks = sorted(dc_curr["week"].unique())
    rule_partition = ruleset.name  # e.g., "ESPN_PPR"
    for week in weeks:
        feats = build_wr_features(
            weekly_stats=ws_full,
            snap_counts=sc_full,
            depth_charts=dc_curr,
            ngs_receiving=ngs_full,
            schedules=sch_curr,
            season=_PROJECTION_SEASON,
            as_of_week=int(week),
        )
        if feats.empty:
            print(f"  Week {week}: no rostered WRs; skipping")
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
