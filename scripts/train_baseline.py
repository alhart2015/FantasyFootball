"""Plan 3b -- train Model A baseline (or Plan 5 Model C lightgbm) for a
specified position on 2018-2023, persist to models/artifacts/.

Replaces scripts/train_wr_baseline.py with a position-arg-driven version.
Usage:
    python scripts/train_baseline.py {qb|rb|te|wr} [--model {baseline|lightgbm}]

Reads ingested raw partitions from data/raw/, builds features for every
week of 2018-2023, fits the selected model via
POSITION_DISPATCH[pos].factories[args.model](), saves the joblib artifact to:
    models/artifacts/{model}-{pos}-{train_start}-{train_end}-{hash}.joblib

Held-out: 2024 (sanity_check_baseline.py {pos} consumes the artifact).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd

from projections.models import POSITION_DISPATCH, BaselineModel, LightGBMModel
from projections.schemas import Position
from projections.store import read_partition

_TRAIN_SEASONS = range(2018, 2024)  # 2018..2023 inclusive (2024 held out)


def _build_training_features(
    raw_root: Path, position: Position
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a stacked feature DataFrame across (season, week) pairs in the
    training window plus the matching weekly_stats truth across the same
    seasons. Caller passes both into BaselineModel.fit."""
    dispatch = POSITION_DISPATCH[position]
    builder = dispatch.feature_builder
    ngs_table = f"ngs_{dispatch.ngs_stat_type}"
    # Each builder takes a kwarg named ngs_{stat_type}; unify by mapping.
    ngs_kwarg = {
        "passing": "ngs_passing",
        "rushing": "ngs_rushing",
        "receiving": "ngs_receiving",
    }[dispatch.ngs_stat_type]

    # 2026-05-03 WR trajectory: load draft_picks once across the full
    # nfl_data_py-supported range (1980+) so the WR trajectory features see
    # every drafted player's birth_date / draft_year on the join. Degrade
    # gracefully if a season's partition is missing — the builder routes
    # missing rows to the inferred-draft-year fallback.
    max_season = max(_TRAIN_SEASONS)
    draft_picks_frames: list[pd.DataFrame] = []
    for s in range(1980, max_season + 1):
        try:
            draft_picks_frames.append(read_partition(raw_root, "draft_picks", season=s))
        except FileNotFoundError:
            continue
    draft_picks = (
        pd.concat(draft_picks_frames, ignore_index=True) if draft_picks_frames else pd.DataFrame()
    )

    feature_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    for season in _TRAIN_SEASONS:
        ws = read_partition(raw_root, "weekly_stats", season=season)
        sc = read_partition(raw_root, "snap_counts", season=season)
        dc = read_partition(raw_root, "depth_charts", season=season)
        ngs = read_partition(raw_root, ngs_table, season=season)
        sch = read_partition(raw_root, "schedules", season=season)
        # Plan 9: PBP for opp-defensive EPA features. Degrade gracefully if a
        # season's partition doesn't exist (e.g., pre-Plan-9 ingest hasn't run);
        # builders accept an empty frame and emit NaN-filled values.
        try:
            pbp = read_partition(raw_root, "pbp", season=season)
        except FileNotFoundError:
            pbp = pd.DataFrame()
        truth_frames.append(ws)

        weeks = sorted(dc["week"].unique())
        for week in weeks:
            kwargs: dict[str, Any] = {
                "weekly_stats": ws,
                "snap_counts": sc,
                "depth_charts": dc,
                "schedules": sch,
                "season": int(season),
                "as_of_week": int(week),
                "pbp": pbp,
                "draft_picks": draft_picks,
                ngs_kwarg: ngs,
            }
            f = builder(**kwargs)
            if not f.empty:
                feature_frames.append(f)
        print(f"  Built {position.value} features for season {season}: {len(weeks)} weeks")

    features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    weekly_stats = pd.concat(truth_frames, ignore_index=True) if truth_frames else pd.DataFrame()
    return features, weekly_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Model A baseline or Model C lightgbm for a position."
    )
    parser.add_argument("position", choices=["qb", "rb", "te", "wr"], help="Target position.")
    parser.add_argument(
        "--model",
        choices=["baseline", "lightgbm"],
        default="baseline",
        help="Which model class to train (Model A or Model C). Default baseline.",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("models/artifacts"))
    args = parser.parse_args()

    position = Position(args.position.upper())
    print(f"Training {position.value} {args.model}; reading raw partitions from {args.raw_root}")

    features, weekly_stats = _build_training_features(args.raw_root, position)
    print(
        f"Total {position.value} feature rows: {len(features)}; "
        f"weekly_stats rows: {len(weekly_stats)}"
    )

    # Dispatch returns the Model Protocol; cast to the concrete-class union so
    # we can read BaselineModel-/LightGBMModel-specific properties
    # (target_stats, train_seasons, code_hash) that aren't part of the Protocol.
    # Both concrete classes are exhaustively listed in factories above, so the
    # cast is sound by construction.
    model = cast(BaselineModel | LightGBMModel, POSITION_DISPATCH[position].factories[args.model]())
    model.fit(features=features, weekly_stats=weekly_stats)
    print(f"model_id: {model.model_id}")
    if isinstance(model, BaselineModel):
        # variance_params is BaselineModel-specific (per-stat parametric residual
        # variance); LightGBMModel quantile regressors don't expose an analogue.
        for stat in model.target_stats:
            print(f"  {stat.value}: variance_params = {model.variance_params[stat]}")

    train_start, train_end = model.train_seasons or (0, 0)
    pos_lower = position.value.lower()
    artifact = (
        args.artifacts_root
        / f"{args.model}-{pos_lower}-{train_start}-{train_end}-{model.code_hash}.joblib"
    )
    model.save(artifact)
    print(f"Saved artifact: {artifact}")


if __name__ == "__main__":
    main()
