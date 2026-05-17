"""End-to-end preseason projection driver.

Reads raw inputs, builds features, fits the model, predicts, and writes the
parquet partition. Returns the in-memory frame for downstream use (e.g., the
backtest harness in Task 21).

See `docs/superpowers/specs/2026-05-17-preseason-projections-design.md` §5.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from projections.preseason.features import build_preseason_features
from projections.preseason.model import NaivePreseasonModel, PreseasonModel
from projections.schemas import (
    PreseasonProjectionSchema,
    Ruleset,
)
from projections.store import read_partition, write_partition

logger = logging.getLogger(__name__)


def project_preseason(
    *,
    raw_root: Path,
    projections_root: Path,
    target_season: int,
    train_start: int,
    ruleset: Ruleset,
    model: PreseasonModel | None = None,
    dropped_csv_path: Path | None = None,
) -> pd.DataFrame:
    """Run the end-to-end preseason pipeline. Returns the projection frame.

    Args:
        raw_root: data/raw root directory.
        projections_root: data/projections root directory.
        target_season: the year to project (e.g., 2026).
        train_start: earliest season in the training window (default 2018).
        ruleset: scoring rules.
        model: optional pre-fit model. If None, fits a fresh NaivePreseasonModel.
        dropped_csv_path: optional path to write dropped-player side-channel CSV.

    Returns:
        DataFrame validated against PreseasonProjectionSchema.
    """
    weekly_stats_frames: list[pd.DataFrame] = []
    for s in range(train_start, target_season):
        try:
            weekly_stats_frames.append(read_partition(raw_root, "weekly_stats", season=s))
        except FileNotFoundError:
            logger.warning("weekly_stats season=%d missing; skipping in training window", s)
    if not weekly_stats_frames:
        raise FileNotFoundError(
            f"No weekly_stats partitions found under {raw_root} for "
            f"seasons {train_start}..{target_season - 1}."
        )
    weekly_stats = pd.concat(weekly_stats_frames, ignore_index=True)

    try:
        depth_charts_target = read_partition(raw_root, "depth_charts", season=target_season)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"depth_charts season={target_season} not found at "
            f"{raw_root}/depth_charts/season={target_season}/. "
            f"Run refresh_depth_charts([{target_season}]) first."
        ) from e

    draft_picks_frames: list[pd.DataFrame] = []
    for s in range(1980, target_season + 1):
        try:
            draft_picks_frames.append(read_partition(raw_root, "draft_picks", season=s))
        except FileNotFoundError:
            continue
    draft_picks = (
        pd.concat(draft_picks_frames, ignore_index=True) if draft_picks_frames else pd.DataFrame()
    )

    id_map = read_partition(raw_root, "id_map")

    features = build_preseason_features(
        weekly_stats=weekly_stats,
        depth_charts_target=depth_charts_target,
        draft_picks=draft_picks,
        id_map=id_map,
        target_season=target_season,
        dropped_csv_path=dropped_csv_path,
    )

    if model is None:
        model = NaivePreseasonModel()
        model.fit(weekly_stats=weekly_stats, draft_picks=draft_picks, id_map=id_map)
    projections = model.predict_season_distribution(features, ruleset=ruleset)

    projections = PreseasonProjectionSchema.validate(projections)
    table = f"preseason/ruleset={ruleset.name}"
    target = write_partition(
        projections_root,
        table,
        projections,
        season=target_season,
        week=None,
    )
    logger.info("project_preseason: wrote %d rows -> %s", len(projections), target)
    return projections
