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

from projections.distributions import Distribution, ParametricGamma, ParametricNormal
from projections.schemas import DistributionFamily, Position, Stat

_GAMMA_ALPHA_CLIP: Final[tuple[float, float]] = (0.01, 100.0)


def _gamma_alpha_from_residuals(*, mu_hat: np.ndarray, residuals: np.ndarray) -> float:
    """Method-of-moments shape parameter alpha for a Gamma distribution
    parameterized by (alpha, beta=alpha/mu_hat). var = mu_hat^2 / alpha, so
    alpha_hat = mean(mu_hat)^2 / var(residuals).

    Clipped to [0.01, 100] for numerical safety; the MoM estimator is
    degenerate for very rare events. Spec section 3.4 documents this choice.
    """
    mean_mu = float(mu_hat.mean())
    var_resid = float(residuals.var())  # population variance (ddof=0)
    if mean_mu == 0.0 or var_resid == 0.0:
        # Degenerate; pick the most permissive clip.
        return _GAMMA_ALPHA_CLIP[1] if var_resid == 0.0 else _GAMMA_ALPHA_CLIP[0]
    raw = (mean_mu * mean_mu) / var_resid
    return float(min(max(raw, _GAMMA_ALPHA_CLIP[0]), _GAMMA_ALPHA_CLIP[1]))


def _normal_std_from_residuals(residuals: np.ndarray) -> float:
    """Global per-stat residual std for the Normal family. Floored at a
    tiny positive epsilon so ParametricNormal's std>0 invariant always holds."""
    s = float(residuals.std())
    return max(s, 1e-6)


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
    ridges: dict[Stat, RidgeCV] = field(default_factory=dict)
    variance_params: dict[Stat, dict[str, float]] = field(default_factory=dict)
    train_seasons: tuple[int, int] | None = field(default=None)
    code_hash: str | None = field(default=None)

    # Floor for predicted Gamma mean (mu_hat). Ridge can predict <=0 for low-mean
    # stats (e.g., fumbles_lost). Clamping keeps the rate parameter (alpha/mu)
    # well-defined. Spec section 3.2 step 3.
    _GAMMA_MU_FLOOR: Final[float] = 1e-3

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

        if joined.empty:
            raise ValueError(
                "Inner-join of features and weekly_stats produced 0 rows. "
                "Check that (gsis_id, season, week) tuples overlap and that "
                f"weekly_stats contains position={self.position.value} rows."
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

        # Variance estimation (spec section 3.4).
        for stat in self.target_stats:
            ridge = self.ridges[stat]
            mu_hat = ridge.predict(x).astype(np.float64)
            y = truth_frame[stat.value].to_numpy(dtype=np.float64)
            residuals = y - mu_hat
            family = self.dist_families[stat]
            if family is DistributionFamily.NORMAL:
                self.variance_params[stat] = {"std": _normal_std_from_residuals(residuals)}
            elif family is DistributionFamily.GAMMA:
                self.variance_params[stat] = {
                    "shape": _gamma_alpha_from_residuals(mu_hat=mu_hat, residuals=residuals)
                }
            else:  # pragma: no cover — only NORMAL/GAMMA configured today
                raise ValueError(f"Unsupported family {family} for stat {stat}")

        # Record the season range we trained on (post-dropna training set).
        trained_seasons = sorted(joined.loc[feature_frame.index, "season"].unique().tolist())
        self.train_seasons = (int(trained_seasons[0]), int(trained_seasons[-1]))

    def _build_stat_distributions(self, features: pd.DataFrame) -> list[dict[Stat, Distribution]]:
        """Build per-row dicts of {Stat -> Distribution} from fitted regressors.

        Pure function over the fitted state. Does not call score_distribution
        (that's predict_distribution's job, Task 9). Useful for unit tests and
        for any caller that wants per-stat dists for analysis.
        """
        if not self.ridges or self.feature_means is None:
            raise RuntimeError("Model is not fitted; call fit() before predict.")

        # Build feature matrix with same column order, impute, coerce bools.
        x_cols = list(self.feature_columns)
        x_frame = features[x_cols].copy()
        for col in x_frame.columns:
            if x_frame[col].dtype == bool:
                x_frame[col] = x_frame[col].astype(np.int8)
        x_frame = x_frame.fillna(self.feature_means)
        x = x_frame.to_numpy(dtype=np.float64)

        # Per-stat predict + Distribution construction.
        per_stat_mu: dict[Stat, np.ndarray] = {}
        for stat in self.target_stats:
            mu = self.ridges[stat].predict(x).astype(np.float64)
            if self.dist_families[stat] is DistributionFamily.GAMMA:
                mu = np.maximum(mu, self._GAMMA_MU_FLOOR)
            per_stat_mu[stat] = mu

        out: list[dict[Stat, Distribution]] = []
        for i in range(len(x)):
            row: dict[Stat, Distribution] = {}
            for stat in self.target_stats:
                mu_i = float(per_stat_mu[stat][i])
                family = self.dist_families[stat]
                params = self.variance_params[stat]
                if family is DistributionFamily.NORMAL:
                    row[stat] = ParametricNormal(mean=mu_i, std=params["std"])
                elif family is DistributionFamily.GAMMA:
                    shape = params["shape"]
                    # rate = alpha / mu; scale = 1/rate = mu / alpha
                    scale = mu_i / shape
                    row[stat] = ParametricGamma(shape=shape, scale=scale)
                else:  # pragma: no cover
                    raise ValueError(f"Unsupported family {family}")
            out.append(row)
        return out


def wr_baseline() -> BaselineModel:
    """Construct an unfitted WR-baseline model. Caller invokes .fit(features,
    weekly_stats) and then .save(path)."""
    return BaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
    )
