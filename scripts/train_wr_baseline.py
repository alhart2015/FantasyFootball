"""Plan 3a -- train WR Model A baseline on 2018-2023, persist to models/artifacts/.

Run from the repo root:
    python scripts/train_wr_baseline.py

Reads ingested raw partitions from ``data/raw/``, builds WR features for every
week of 2018-2023, fits BaselineModel, saves the joblib artifact under a name
that includes the train window and code hash so reruns produce a comparable
file even after model code changes.

Note: the spec calls for training on 2018-2024 with 2025 as the held-out
sanity-check year, but nfl_data_py has not yet published 2025 data as of
this run. The actual training window is 2018-2023, with 2024 held out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from projections.features import build_wr_features
from projections.models import wr_baseline
from projections.store import read_partition

_TRAIN_SEASONS = range(2018, 2024)  # 2018..2023 inclusive (2024 held out)


def _build_training_features(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a stacked WR feature DataFrame across (season, week) pairs in the
    training window, plus the matching weekly_stats truth across the same
    seasons. Caller passes both into ``BaselineModel.fit``."""
    feature_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    for season in _TRAIN_SEASONS:
        ws = read_partition(raw_root, "weekly_stats", season=season)
        sc = read_partition(raw_root, "snap_counts", season=season)
        dc = read_partition(raw_root, "depth_charts", season=season)
        ngs = read_partition(raw_root, "ngs_receiving", season=season)
        sch = read_partition(raw_root, "schedules", season=season)
        truth_frames.append(ws)

        weeks = sorted(dc["week"].unique())
        for week in weeks:
            f = build_wr_features(
                weekly_stats=ws,
                snap_counts=sc,
                depth_charts=dc,
                ngs_receiving=ngs,
                schedules=sch,
                season=int(season),
                as_of_week=int(week),
            )
            if not f.empty:
                feature_frames.append(f)
        print(f"  Built features for season {season}: {len(weeks)} weeks")

    features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    weekly_stats = pd.concat(truth_frames, ignore_index=True) if truth_frames else pd.DataFrame()
    return features, weekly_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WR Model A baseline.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("models/artifacts"))
    args = parser.parse_args()

    print(f"Reading raw partitions from {args.raw_root}")
    features, weekly_stats = _build_training_features(args.raw_root)
    print(f"Total WR feature rows: {len(features)}; weekly_stats rows: {len(weekly_stats)}")

    model = wr_baseline()
    model.fit(features=features, weekly_stats=weekly_stats)
    print(f"model_id: {model.model_id}")
    for stat in model.target_stats:
        print(f"  {stat.value}: variance_params = {model.variance_params[stat]}")

    train_start, train_end = model.train_seasons or (0, 0)
    artifact = (
        args.artifacts_root / f"wr-baseline-{train_start}-{train_end}-{model.code_hash}.joblib"
    )
    model.save(artifact)
    print(f"Saved artifact: {artifact}")


if __name__ == "__main__":
    main()
