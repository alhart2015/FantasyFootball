"""Baseline model — per-stat Ridge regressions composed into a points
distribution via the existing scoring layer.

One BaselineModel class parameterized by (position, target_stats,
feature_columns, dist_families); per-position factories (wr_baseline,
qb_baseline, rb_baseline, te_baseline) construct correctly-configured
instances. Plan 3a only ships wr_baseline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from projections.schemas import DistributionFamily, Position, Stat

_WR_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.RECEPTIONS,
    Stat.RECEIVING_YARDS,
    Stat.RECEIVING_TDS,
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.FUMBLES_LOST,
)

_WR_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.RECEPTIONS: DistributionFamily.GAMMA,
    Stat.RECEIVING_YARDS: DistributionFamily.NORMAL,
    Stat.RECEIVING_TDS: DistributionFamily.GAMMA,
    Stat.RUSHING_YARDS: DistributionFamily.NORMAL,
    Stat.RUSHING_TDS: DistributionFamily.GAMMA,
    Stat.FUMBLES_LOST: DistributionFamily.GAMMA,
}

# Feature columns from WrFeaturesSchema, minus identity columns
# (gsis_id / season / week / team / opponent). Boolean columns are coerced
# to 0/1 by fit/predict.
_WR_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "depth_rank",
    "targets_per_game_l4",
    "targets_per_game_std",
    "target_share_l4",
    "air_yards_share_l4",
    "receptions_per_game_l4",
    "receiving_yards_per_game_l4",
    "receiving_tds_per_game_l4",
    "rushing_attempts_per_game_l4",
    "rushing_yards_per_game_l4",
    "designed_rusher",
    "snap_pct_l4",
    "avg_separation_std",
    "avg_intended_air_yards_std",
    "percent_share_intended_air_yards_std",
    "avg_yac_above_expectation_std",
    "implied_team_total",
    "spread",
    "is_home",
    "roof_dome",
    "opp_allowed_wr_fppg_l4",
)


@dataclass
class BaselineModel:
    """Per-stat Ridge baseline. Construct via per-position factories
    (wr_baseline, etc.); do not call __init__ directly."""

    position: Position
    target_stats: tuple[Stat, ...]
    feature_columns: tuple[str, ...]
    dist_families: Mapping[Stat, DistributionFamily]

    # Populated by .fit() — None on an unfitted instance.
    feature_means: pd.Series | None = field(default=None)
    ridges: dict[Stat, object] = field(default_factory=dict)
    variance_params: dict[Stat, dict[str, float]] = field(default_factory=dict)
    train_seasons: tuple[int, int] | None = field(default=None)
    code_hash: str | None = field(default=None)

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """Train one RidgeCV per target stat. See spec §3.1 for the pipeline."""
        # Schema validation — defensive at the boundary even though our caller
        # is supposed to have already validated.
        from projections.schemas import WeeklyStatsSchema, WrFeaturesSchema

        features = WrFeaturesSchema.validate(features)
        weekly_stats = WeeklyStatsSchema.validate(weekly_stats)

        # Inner-join features with truth on (gsis_id, season, week). Players in
        # the depth chart who didn't actually play that week have no truth and
        # are silently dropped — that's correct, the model only learns from
        # players who played.
        ws = weekly_stats[weekly_stats["position"] == self.position.value].copy()
        joined = features.merge(
            ws[
                [
                    "gsis_id",
                    "season",
                    "week",
                    *(s.value for s in self.target_stats),
                ]
            ],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )

        # Build feature matrix, persist column order. Drop rows with NaN in any
        # feature column at fit time (mostly week-1-of-season rows where
        # rolling features have no prior history).
        x_cols = list(self.feature_columns)
        feature_frame = joined[x_cols].copy()
        # Coerce booleans to 0/1.
        for col in feature_frame.columns:
            if feature_frame[col].dtype == bool:
                feature_frame[col] = feature_frame[col].astype(np.int8)
        # Persist medians BEFORE dropping NaN rows so predict-time imputation
        # uses the broadest possible signal.
        self.feature_means = feature_frame.median(skipna=True).astype(float)

        feature_frame = feature_frame.dropna()
        if feature_frame.empty:
            raise ValueError(
                "After dropping NaN feature rows, no training data remains. "
                "Check the feature builder and inputs."
            )
        truth_frame = joined.loc[feature_frame.index, [s.value for s in self.target_stats]]

        x = feature_frame.to_numpy(dtype=np.float64)

        # Fit one RidgeCV per stat.
        alphas = np.logspace(-3, 3, 13)
        for stat in self.target_stats:
            y = truth_frame[stat.value].to_numpy(dtype=np.float64)
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(x, y)
            self.ridges[stat] = ridge

        # Record the season range we trained on.
        seasons = sorted(joined["season"].unique().tolist())
        self.train_seasons = (int(seasons[0]), int(seasons[-1]))


def wr_baseline() -> BaselineModel:
    """Construct an unfitted WR-baseline model. Caller invokes .fit(features,
    weekly_stats) and then .save(path)."""
    return BaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
    )
