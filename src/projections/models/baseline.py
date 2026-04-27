"""Baseline model — per-stat Ridge regressions composed into a points
distribution via the existing scoring layer.

One BaselineModel class parameterized by (position, target_stats,
feature_columns, dist_families); per-position factories (wr_baseline,
qb_baseline, rb_baseline, te_baseline) construct correctly-configured
instances.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pandera.pandas as pa
from scipy import stats as scipy_stats
from scipy.optimize import minimize_scalar
from sklearn.linear_model import RidgeCV

from projections.distributions import (
    Distribution,
    ParametricGamma,
    ParametricNormal,
    pack_per_stat_params,
)
from projections.models.base import compute_code_hash
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    Position,
    ProjectionWeeklySchema,
    QbFeaturesSchema,
    RbFeaturesSchema,
    Ruleset,
    Stat,
    TeFeaturesSchema,
    WeeklyStatsSchema,
    WrFeaturesSchema,
)
from projections.scoring import derive_row_seed, score_distribution

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


_NB_DISPERSION_CLIP: Final[tuple[float, float]] = (0.01, 1000.0)


def _negative_binomial_dispersion_from_residuals(
    *, mu_hat: np.ndarray, actual: np.ndarray
) -> float:
    """Conditional MLE for NB dispersion given per-row mean = mu_hat.

    Maximizes sum(nbinom.logpmf(actual_i; n_i, p_i)) over a single global
    `dispersion` where n_i = mu_hat_i^2 / dispersion and p_i = n_i / (n_i + mu_hat_i).

    Coerces actual to non-negative integers (counts upstream may carry float
    dtype). Returns the dispersion clipped to ``_NB_DISPERSION_CLIP``.
    """
    counts = np.clip(np.round(actual), 0, None).astype(np.int64)
    mu_clipped = np.maximum(mu_hat, 1e-3)

    if counts.size < 2:
        return _NB_DISPERSION_CLIP[1]

    def neg_log_lik(dispersion: float) -> float:
        if dispersion <= 0:
            return float("inf")
        n_size = mu_clipped * mu_clipped / dispersion
        p = n_size / (n_size + mu_clipped)
        return -float(np.sum(scipy_stats.nbinom.logpmf(counts, n=n_size, p=p)))

    result = minimize_scalar(
        neg_log_lik,
        bounds=_NB_DISPERSION_CLIP,
        method="bounded",
        options={"xatol": 1e-3},
    )
    if not result.success or not np.isfinite(result.fun):
        return _NB_DISPERSION_CLIP[1]
    fitted = float(np.clip(result.x, *_NB_DISPERSION_CLIP))
    # Snap to a clip endpoint when the bounded minimizer stops within its xatol
    # of the boundary: degenerate inputs (e.g. all-zero actuals) drive the
    # likelihood monotonically toward an endpoint, but `minimize_scalar` returns
    # a value just inside the bound rather than the bound itself.
    snap_tol = 2e-3
    if fitted - _NB_DISPERSION_CLIP[0] <= snap_tol:
        return _NB_DISPERSION_CLIP[0]
    if _NB_DISPERSION_CLIP[1] - fitted <= snap_tol:
        return _NB_DISPERSION_CLIP[1]
    return fitted


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


_QB_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.PASSING_YARDS,
    Stat.PASSING_TDS,
    Stat.INTERCEPTIONS,
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.FUMBLES_LOST,
)

_QB_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.PASSING_YARDS: DistributionFamily.NORMAL,
    Stat.PASSING_TDS: DistributionFamily.GAMMA,
    Stat.INTERCEPTIONS: DistributionFamily.GAMMA,
    Stat.RUSHING_YARDS: DistributionFamily.NORMAL,
    Stat.RUSHING_TDS: DistributionFamily.GAMMA,
    Stat.FUMBLES_LOST: DistributionFamily.GAMMA,
}

# Feature columns from QbFeaturesSchema, minus identity (gsis_id/season/week/team/opponent).
# Boolean columns (rushing_qb / is_home / roof_dome) are coerced to 0/1 by fit/predict.
_QB_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "depth_rank",
    "pass_attempts_per_game_l4",
    "passing_yards_per_game_l4",
    "passing_tds_per_game_l4",
    "interceptions_per_game_l4",
    "sacks_per_game_l4",
    "passing_yards_per_game_std",
    "rushing_attempts_per_game_l4",
    "rushing_yards_per_game_l4",
    "rushing_qb",
    "snap_pct_l4",
    "aggressiveness_std",
    "completion_percentage_above_expectation_std",
    "avg_intended_air_yards_std",
    "avg_time_to_throw_std",
    "implied_team_total",
    "spread",
    "is_home",
    "roof_dome",
    "opp_allowed_qb_fppg_l4",
)


_RB_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.RECEPTIONS,
    Stat.RECEIVING_YARDS,
    Stat.RECEIVING_TDS,
    Stat.FUMBLES_LOST,
)

_RB_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.RUSHING_YARDS: DistributionFamily.NORMAL,
    Stat.RUSHING_TDS: DistributionFamily.GAMMA,
    Stat.RECEPTIONS: DistributionFamily.GAMMA,
    Stat.RECEIVING_YARDS: DistributionFamily.NORMAL,
    Stat.RECEIVING_TDS: DistributionFamily.GAMMA,
    Stat.FUMBLES_LOST: DistributionFamily.GAMMA,
}

_RB_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "depth_rank",
    "carries_per_game_l4",
    "rushing_yards_per_game_l4",
    "rushing_tds_per_game_l4",
    "rush_share_l4",
    "targets_per_game_l4",
    "receptions_per_game_l4",
    "receiving_yards_per_game_l4",
    "target_share_l4",
    "targets_per_game_std",
    "passing_down_back",
    "snap_pct_l4",
    "efficiency_std",
    "rush_yards_over_expected_per_att_std",
    "percent_attempts_gte_eight_defenders_std",
    "implied_team_total",
    "spread",
    "is_home",
    "roof_dome",
    "opp_allowed_rb_fppg_l4",
)


_TE_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.RECEPTIONS,
    Stat.RECEIVING_YARDS,
    Stat.RECEIVING_TDS,
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.FUMBLES_LOST,
)

_TE_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.RECEPTIONS: DistributionFamily.GAMMA,
    Stat.RECEIVING_YARDS: DistributionFamily.NORMAL,
    Stat.RECEIVING_TDS: DistributionFamily.GAMMA,
    Stat.RUSHING_YARDS: DistributionFamily.NORMAL,
    Stat.RUSHING_TDS: DistributionFamily.GAMMA,
    Stat.FUMBLES_LOST: DistributionFamily.GAMMA,
}

# Feature columns include rushing_*_per_game_l4 (added to TeFeaturesSchema in Plan 3b Phase 1).
_TE_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "depth_rank",
    "targets_per_game_l4",
    "targets_per_game_std",
    "target_share_l4",
    "receptions_per_game_l4",
    "receiving_yards_per_game_l4",
    "receiving_tds_per_game_l4",
    "rushing_attempts_per_game_l4",
    "rushing_yards_per_game_l4",
    "snap_pct_l4",
    "avg_separation_std",
    "avg_intended_air_yards_std",
    "avg_yac_above_expectation_std",
    "implied_team_total",
    "spread",
    "is_home",
    "roof_dome",
    "opp_allowed_te_fppg_l4",
)


def _default_code_hash_files(position_module: str) -> tuple[Path, ...]:
    """Build the canonical code-hash file tuple for a position factory.

    The eight files are: models/base.py, models/baseline.py,
    features/{position_module}, features/_shared.py, features/_rolling.py,
    features/_opponent.py, scoring/score.py, scoring/score_distribution.py.

    Used by every position factory so the per-factory call site is one line.
    """
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root / "src" / "projections" / "models" / "base.py",
        repo_root / "src" / "projections" / "models" / "baseline.py",
        repo_root / "src" / "projections" / "features" / position_module,
        repo_root / "src" / "projections" / "features" / "_shared.py",
        repo_root / "src" / "projections" / "features" / "_rolling.py",
        repo_root / "src" / "projections" / "features" / "_opponent.py",
        repo_root / "src" / "projections" / "scoring" / "score.py",
        repo_root / "src" / "projections" / "scoring" / "score_distribution.py",
    )


@dataclass
class BaselineModel:
    """Per-stat Ridge baseline. Construct via per-position factories
    (wr_baseline, etc.); do not call __init__ directly."""

    position: Position
    target_stats: tuple[Stat, ...]
    feature_columns: tuple[str, ...]
    dist_families: Mapping[Stat, DistributionFamily]
    feature_schema: type[pa.DataFrameModel]
    code_hash_files: tuple[Path, ...]

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

    def _x_frame_with_bool_coercion(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select feature_columns (preserving order) and coerce bool to int8.
        Caller decides NaN policy (fit drops; predict imputes).
        """
        x_cols = list(self.feature_columns)
        x_frame = features[x_cols].copy()
        for col in x_frame.columns:
            if x_frame[col].dtype == bool:
                x_frame[col] = x_frame[col].astype(np.int8)
        return x_frame

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """Train one RidgeCV per target stat. See spec section 3.1 for the
        pipeline."""
        # Schema validation -- defensive at the boundary even though our caller
        # is supposed to have already validated.
        features = self.feature_schema.validate(features)
        weekly_stats = WeeklyStatsSchema.validate(weekly_stats)

        # Inner-join features with truth on (gsis_id, season, week). Players in
        # the depth chart who didn't actually play that week have no truth and
        # are silently dropped -- that's correct, the model only learns from
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
        feature_frame = self._x_frame_with_bool_coercion(joined)
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
            else:  # pragma: no cover -- only NORMAL/GAMMA configured today
                raise ValueError(f"Unsupported family {family} for stat {stat}")

        # Record the season range we trained on (post-dropna training set).
        trained_seasons = sorted(joined.loc[feature_frame.index, "season"].unique().tolist())
        self.train_seasons = (int(trained_seasons[0]), int(trained_seasons[-1]))

        # Code hash over the file list this factory declared. The exact set
        # is per-position (each factory passes its own features/{pos}.py).
        self.code_hash = compute_code_hash(self.code_hash_files)

    def build_stat_distributions(self, features: pd.DataFrame) -> list[dict[Stat, Distribution]]:
        """Build per-row dicts of {Stat -> Distribution} from fitted regressors.

        Pure function over the fitted state. Does not call score_distribution
        (that's predict_distribution's job, Task 9). Useful for unit tests and
        for any caller that wants per-stat dists for analysis.
        """
        if not self.ridges or self.feature_means is None:
            raise RuntimeError("Model is not fitted; call fit() before predict.")

        # Build feature matrix with same column order, impute, coerce bools.
        x_frame = self._x_frame_with_bool_coercion(features)
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

    @property
    def model_id(self) -> str:
        """Stable identifier of the form
        ``"baseline:<position>:<8-char-code-hash>:<train-start>-<train-end>"``.

        Undefined for unfitted models (raises RuntimeError); both ``code_hash``
        and ``train_seasons`` are populated by ``fit()``.
        """
        if self.code_hash is None or self.train_seasons is None:
            raise RuntimeError("model_id is undefined for unfitted models")
        return (
            f"baseline:{self.position.value.lower()}:{self.code_hash}"
            f":{self.train_seasons[0]}-{self.train_seasons[1]}"
        )

    def predict_distribution(self, features: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
        """Predict per-player-week fantasy-points distributions under ``ruleset``.

        Returns a DataFrame validated against ``ProjectionWeeklySchema`` with one
        row per ``features`` row. Each row's persisted ``mean`` / ``p10`` / ``p50``
        / ``p90`` columns are the canonical per-row distributional summary; the
        ``params`` blob carries per-stat distribution parameters via
        ``pack_per_stat_params`` so a downstream consumer can rehydrate the
        per-stat distributions and regenerate samples deterministically.

        Per-row Monte Carlo seed is derived from ``(gsis_id, season, week,
        ruleset.name)`` via ``derive_row_seed``, giving cross-process reproducible
        and cross-row independent samples.
        """
        features = self.feature_schema.validate(features)
        if features.empty:
            empty_cols = list(ProjectionWeeklySchema.to_schema().columns.keys())
            return ProjectionWeeklySchema.validate(pd.DataFrame(columns=empty_cols))

        stat_dists_per_row = self.build_stat_distributions(features)

        rows: list[dict[str, object]] = []
        generated_at = datetime.now(UTC)
        for (_idx, feat_row), stat_dists in zip(
            features.reset_index(drop=True).iterrows(), stat_dists_per_row, strict=True
        ):
            seed = derive_row_seed(
                gsis_id=str(feat_row["gsis_id"]),
                season=int(feat_row["season"]),
                week=int(feat_row["week"]),
                ruleset_name=ruleset.name,
            )
            points = score_distribution(stat_dists, ruleset, n_samples=10_000, seed=seed)
            family_blob = pack_per_stat_params(stat_dists)
            rows.append(
                {
                    "gsis_id": feat_row["gsis_id"],
                    "season": int(feat_row["season"]),
                    "week": int(feat_row["week"]),
                    "position": self.position.value,
                    "team": feat_row["team"],
                    "opponent": feat_row["opponent"],
                    "ruleset": ruleset.name,
                    "family": DistributionFamily.SAMPLED_SUMMARY.value,
                    "params": family_blob,
                    "mean": points.mean(),
                    "p10": points.quantile(0.1),
                    "p50": points.quantile(0.5),
                    "p90": points.quantile(0.9),
                    "model_id": self.model_id,
                    "generated_at": pd.Timestamp(generated_at).as_unit("us"),
                }
            )

        out = pd.DataFrame(rows)
        for col in ("gsis_id", "team", "opponent", "ruleset", "family", "model_id"):
            out[col] = out[col].astype(_PYARROW_STR)
        out["position"] = out["position"].astype(_PYARROW_STR)
        return ProjectionWeeklySchema.validate(out)

    def save(self, path: Path) -> None:
        """Serialize the fitted model to ``path`` via joblib.

        Refuses to save an unfitted instance: an artifact without a
        ``code_hash`` / ``train_seasons`` cannot derive a stable ``model_id``
        and would silently produce un-traceable projections on load.
        """
        import joblib

        if self.code_hash is None or self.train_seasons is None:
            raise RuntimeError("Cannot save an unfitted BaselineModel")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> BaselineModel:
        """Deserialize a BaselineModel from ``path``. Raises ``TypeError`` if
        the artifact decodes to a different class -- defends against loading a
        future GBM artifact through this entrypoint."""
        import joblib

        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected BaselineModel, got {type(loaded).__name__}")
        return loaded


def wr_baseline() -> BaselineModel:
    """Construct an unfitted WR-baseline model. Caller invokes .fit(features,
    weekly_stats) and then .save(path)."""
    return BaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
        feature_schema=WrFeaturesSchema,
        code_hash_files=_default_code_hash_files("wr.py"),
    )


def qb_baseline() -> BaselineModel:
    """Construct an unfitted QB-baseline model."""
    return BaselineModel(
        position=Position.QB,
        target_stats=_QB_TARGET_STATS,
        feature_columns=_QB_FEATURE_COLUMNS,
        dist_families=_QB_DIST_FAMILIES,
        feature_schema=QbFeaturesSchema,
        code_hash_files=_default_code_hash_files("qb.py"),
    )


def rb_baseline() -> BaselineModel:
    """Construct an unfitted RB-baseline model."""
    return BaselineModel(
        position=Position.RB,
        target_stats=_RB_TARGET_STATS,
        feature_columns=_RB_FEATURE_COLUMNS,
        dist_families=_RB_DIST_FAMILIES,
        feature_schema=RbFeaturesSchema,
        code_hash_files=_default_code_hash_files("rb.py"),
    )


def te_baseline() -> BaselineModel:
    """Construct an unfitted TE-baseline model."""
    return BaselineModel(
        position=Position.TE,
        target_stats=_TE_TARGET_STATS,
        feature_columns=_TE_FEATURE_COLUMNS,
        dist_families=_TE_DIST_FAMILIES,
        feature_schema=TeFeaturesSchema,
        code_hash_files=_default_code_hash_files("te.py"),
    )
