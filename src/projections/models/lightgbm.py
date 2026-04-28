"""LightGBM-based per-stat quantile regression (Model C).

Plan 5 — coexists with BaselineModel (Model A) under the existing Model
Protocol. Trains 5 LightGBM quantile sub-models per (position, stat) at
quantiles [0.05, 0.10, 0.50, 0.90, 0.95]. Per-row prediction:
    1. Predict 5 quantiles.
    2. Sort to enforce non-crossing.
    3. Clip to [0, inf) for `non_negative` stats.
    4. Wrap in QuantileDistribution.
    5. Run through the existing scoring layer to get composite PPR points.

The whole prediction path beneath score_distribution is unchanged; the new
QuantileDistribution satisfies the Distribution Protocol structurally.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pandera.pandas as pa

from projections.distributions import (
    QuantileDistribution,
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
from projections.scoring.score_distribution import (
    derive_row_seed,
    score_distribution,
)

LGBM_DEFAULTS: Final[dict[str, Any]] = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,  # required to actually apply subsample
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "verbose": -1,
    "random_state": 42,
}

EARLY_STOPPING_ROUNDS: Final[int] = 50
QUANTILE_GRID: Final[tuple[float, ...]] = (0.05, 0.10, 0.50, 0.90, 0.95)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class _LightGBMConfig:
    """Per-position config for LightGBMModel.

    Attributes:
        position: which Position this config trains.
        target_stats: stats predicted by per-stat sub-models (matches BaselineModel).
        feature_columns: ordered list of feature columns the model consumes.
        feature_schema: pandera schema validated on input to fit/predict.
        non_negative_stats: stats whose predicted quantiles are clipped to [0, inf).
    """

    position: Position
    target_stats: tuple[Stat, ...]
    feature_columns: tuple[str, ...]
    feature_schema: type[pa.DataFrameModel]
    non_negative_stats: frozenset[Stat]


def _code_hash_files(position: Position) -> tuple[Path, ...]:
    """Source files whose content is hashed into model_id for invalidation tracking."""
    src = _PROJECT_ROOT / "src" / "projections"
    feat_module = {
        Position.QB: "qb.py",
        Position.RB: "rb.py",
        Position.TE: "te.py",
        Position.WR: "wr.py",
    }[position]
    return (
        src / "models" / "lightgbm.py",
        src / "models" / "base.py",
        src / "distributions" / "quantile.py",
        src / "distributions" / "codec.py",
        src / "distributions" / "parametric.py",
        src / "features" / feat_module,
        src / "features" / "_shared.py",
        src / "features" / "_rolling.py",
        src / "features" / "_opponent.py",
        src / "scoring" / "score.py",
        src / "scoring" / "score_distribution.py",
    )


# Per-position feature columns mirror BaselineModel's per-position lists.
# Source of truth for the ordering: the corresponding *FeaturesSchema column order.
_QB_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(QbFeaturesSchema.to_schema().columns.keys())
_RB_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(RbFeaturesSchema.to_schema().columns.keys())
_TE_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(TeFeaturesSchema.to_schema().columns.keys())
_WR_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(WrFeaturesSchema.to_schema().columns.keys())

# Drop identifier / target / context columns — only true model features go in.
_NON_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset(
    {"gsis_id", "season", "week", "team", "opponent", "position"}
)


def _filter_features(cols: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(c for c in cols if c not in _NON_FEATURE_COLUMNS)


_QB_NON_NEGATIVE: Final[frozenset[Stat]] = frozenset(
    {Stat.PASSING_TDS, Stat.INTERCEPTIONS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST}
)
_RB_NON_NEGATIVE: Final[frozenset[Stat]] = frozenset(
    {Stat.RECEPTIONS, Stat.RUSHING_TDS, Stat.RECEIVING_TDS, Stat.FUMBLES_LOST}
)
_TE_NON_NEGATIVE: Final[frozenset[Stat]] = frozenset(
    {Stat.RECEPTIONS, Stat.RECEIVING_TDS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST}
)
_WR_NON_NEGATIVE: Final[frozenset[Stat]] = frozenset(
    {Stat.RECEPTIONS, Stat.RECEIVING_TDS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST}
)


_QB_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.PASSING_YARDS,
    Stat.PASSING_TDS,
    Stat.INTERCEPTIONS,
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.FUMBLES_LOST,
)
_RB_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.RECEPTIONS,
    Stat.RECEIVING_YARDS,
    Stat.RECEIVING_TDS,
    Stat.FUMBLES_LOST,
)
_TE_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.RECEPTIONS,
    Stat.RECEIVING_YARDS,
    Stat.RECEIVING_TDS,
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.FUMBLES_LOST,
)
_WR_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.RECEPTIONS,
    Stat.RECEIVING_YARDS,
    Stat.RECEIVING_TDS,
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.FUMBLES_LOST,
)


def qb_lightgbm() -> LightGBMModel:
    return LightGBMModel(
        config=_LightGBMConfig(
            position=Position.QB,
            target_stats=_QB_TARGET_STATS,
            feature_columns=_filter_features(_QB_FEATURE_COLUMNS),
            feature_schema=QbFeaturesSchema,
            non_negative_stats=_QB_NON_NEGATIVE,
        )
    )


def rb_lightgbm() -> LightGBMModel:
    return LightGBMModel(
        config=_LightGBMConfig(
            position=Position.RB,
            target_stats=_RB_TARGET_STATS,
            feature_columns=_filter_features(_RB_FEATURE_COLUMNS),
            feature_schema=RbFeaturesSchema,
            non_negative_stats=_RB_NON_NEGATIVE,
        )
    )


def te_lightgbm() -> LightGBMModel:
    return LightGBMModel(
        config=_LightGBMConfig(
            position=Position.TE,
            target_stats=_TE_TARGET_STATS,
            feature_columns=_filter_features(_TE_FEATURE_COLUMNS),
            feature_schema=TeFeaturesSchema,
            non_negative_stats=_TE_NON_NEGATIVE,
        )
    )


def wr_lightgbm() -> LightGBMModel:
    return LightGBMModel(
        config=_LightGBMConfig(
            position=Position.WR,
            target_stats=_WR_TARGET_STATS,
            feature_columns=_filter_features(_WR_FEATURE_COLUMNS),
            feature_schema=WrFeaturesSchema,
            non_negative_stats=_WR_NON_NEGATIVE,
        )
    )


class LightGBMModel:
    """Per-stat LightGBM quantile-regression model. Implements Model Protocol structurally.

    Use the per-position factories (qb_lightgbm, rb_lightgbm, te_lightgbm, wr_lightgbm)
    rather than constructing directly.
    """

    def __init__(self, *, config: _LightGBMConfig) -> None:
        self._config = config
        self._sub_models: dict[Stat, dict[float, lgb.Booster]] = {}
        self._best_iters: dict[tuple[Stat, float], int] = {}
        self._train_start: int | None = None
        self._train_end: int | None = None
        self._is_fitted: bool = False

    @property
    def position(self) -> Position:
        return self._config.position

    @property
    def target_stats(self) -> tuple[Stat, ...]:
        return self._config.target_stats

    @property
    def train_seasons(self) -> tuple[int, int] | None:
        """(train_start, train_end) recorded at fit time, or None if unfitted.

        Mirrors `BaselineModel.train_seasons` so both implementations expose
        the same attribute for artifact naming and metadata reporting.
        """
        if not self._is_fitted:
            return None
        assert self._train_start is not None and self._train_end is not None
        return (self._train_start, self._train_end)

    @property
    def code_hash(self) -> str:
        """SHA-256 (first 8 chars) of the source files this model depends on.

        Mirrors `BaselineModel.code_hash`. Computed on demand; callers may
        access this on an unfitted instance to compare against an existing
        artifact's expected hash.
        """
        return compute_code_hash(_code_hash_files(self._config.position))

    @property
    def model_id(self) -> str:
        if not self._is_fitted:
            raise RuntimeError(
                "model_id not available before fit() — depends on training-time state"
            )
        assert self._train_start is not None and self._train_end is not None
        return (
            f"lightgbm:{self._config.position.value.lower()}:{self.code_hash}"
            f":{self._train_start}-{self._train_end}"
        )

    def _hyperparams_for(self, stat: Stat) -> dict[str, Any]:
        """Return LightGBM kwargs for the given stat's sub-models.

        Subclasses override to provide tuned per-(position, stat) hyperparameters.
        The base implementation returns a copy of LGBM_DEFAULTS so all sub-models
        share the same baseline settings.
        """
        return dict(LGBM_DEFAULTS)

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """Train per-stat per-quantile sub-models with early stopping on the last
        training season. Stores boosters in self._sub_models and best iterations
        in self._best_iters.

        Raises:
            ValueError: empty join, or fewer than 2 training seasons (needed to carve
                the validation slice).
        """
        # Validate features against the position schema.
        features = self._config.feature_schema.validate(features)
        weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
        # Filter to only this position's truth (matches BaselineModel pattern).
        weekly_stats = weekly_stats[weekly_stats["position"] == self._config.position.value].copy()

        # Inner-join on (gsis_id, season, week).
        target_cols = [s.value for s in self._config.target_stats]
        joined = features.merge(
            weekly_stats[["gsis_id", "season", "week", *target_cols]],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        if joined.empty:
            raise ValueError("Empty training set after feature/weekly_stats join")

        # Need >=2 seasons to carve a last-season validation slice.
        seasons = sorted(joined["season"].unique())
        if len(seasons) < 2:
            raise ValueError(
                f"Need >=2 training seasons for early-stopping validation slice; got {len(seasons)}"
            )

        val_season = seasons[-1]
        train_mask = joined["season"] != val_season
        val_mask = joined["season"] == val_season

        feat_cols = list(self._config.feature_columns)
        x_train = joined.loc[train_mask, feat_cols].to_numpy(dtype=np.float64)
        x_val = joined.loc[val_mask, feat_cols].to_numpy(dtype=np.float64)

        for stat in self._config.target_stats:
            self._sub_models[stat] = {}
            y_train = joined.loc[train_mask, stat.value].to_numpy(dtype=np.float64)
            y_val = joined.loc[val_mask, stat.value].to_numpy(dtype=np.float64)
            stat_params = self._hyperparams_for(stat)
            for q in QUANTILE_GRID:
                regressor = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    **stat_params,
                )
                regressor.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_val, y_val)],
                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
                )
                # `regressor.booster_` exposes the trained Booster after fit.
                best_iter = int(regressor.best_iteration_ or 0)
                if best_iter == 0:
                    warnings.warn(
                        f"LightGBMModel.fit: best_iter=0 for "
                        f"{self._config.position.value}/{stat.value}/q={q}; "
                        "early stopping fired immediately. "
                        "Sub-model will predict at constant baseline.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                self._sub_models[stat][q] = regressor.booster_
                self._best_iters[(stat, q)] = best_iter

        self._train_start = int(seasons[0])
        self._train_end = int(seasons[-1])
        self._is_fitted = True

    def predict_distribution(self, features: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
        """Predict per-row composite fantasy-points distribution.

        Pipeline:
            1. Verify feature columns match training (raise ValueError on mismatch
               BEFORE schema validation so a missing column surfaces as a model-
               level error, not a deep pandera SchemaError).
            2. Validate features against the position schema.
            3. For each row, predict 5 quantiles via the 5 sub-models per stat.
            4. Sort per-row to enforce non-crossing.
            5. Clip to [0, inf) for stats in `non_negative_stats`.
            6. Wrap each per-stat (quantiles, values) in a `QuantileDistribution`.
            7. Run through `score_distribution` -> composite mean / p10 / p50 / p90.

        Returns:
            DataFrame validated against `ProjectionWeeklySchema`.
        """
        if not self._is_fitted:
            raise RuntimeError("predict_distribution requires fit() first")

        feat_cols = list(self._config.feature_columns)
        actual_cols = set(features.columns)
        missing = set(feat_cols) - actual_cols
        if missing:
            raise ValueError(
                f"Feature columns differ from training: missing={sorted(missing)}; "
                f"expected feature_columns={feat_cols}"
            )

        features = self._config.feature_schema.validate(features)

        if features.empty:
            empty_cols = list(ProjectionWeeklySchema.to_schema().columns.keys())
            return ProjectionWeeklySchema.validate(pd.DataFrame(columns=empty_cols))

        x = features[feat_cols].to_numpy(dtype=np.float64)
        n_rows = x.shape[0]
        quant_arr = np.array(QUANTILE_GRID, dtype=np.float64)

        # Predict every (stat, quantile) into a per-stat (n_rows, n_quantiles) array.
        per_stat_pred: dict[Stat, np.ndarray[Any, np.dtype[np.float64]]] = {}
        for stat in self._config.target_stats:
            preds_per_q = np.column_stack(
                [self._sub_models[stat][q].predict(x) for q in QUANTILE_GRID]
            ).astype(np.float64)
            # Sort per-row to enforce non-crossing (in-place is fine — fresh array).
            preds_per_q.sort(axis=1)
            # Clip to >=0 for non-negative stats.
            if stat in self._config.non_negative_stats:
                np.maximum(preds_per_q, 0.0, out=preds_per_q)
            per_stat_pred[stat] = preds_per_q

        # Build per-row per-stat distributions and run scoring.
        out_rows: list[dict[str, Any]] = []
        generated_at = datetime.now(UTC)
        gsis_id_col = features["gsis_id"].to_numpy()
        season_col = features["season"].to_numpy()
        week_col = features["week"].to_numpy()
        team_col = features["team"].to_numpy()
        opponent_col = features["opponent"].to_numpy()
        for row_idx in range(n_rows):
            per_stat_dists: dict[Stat, QuantileDistribution] = {}
            for stat in self._config.target_stats:
                per_stat_dists[stat] = QuantileDistribution(
                    quantiles=quant_arr,
                    values=per_stat_pred[stat][row_idx],
                )

            seed = derive_row_seed(
                gsis_id=str(gsis_id_col[row_idx]),
                season=int(season_col[row_idx]),
                week=int(week_col[row_idx]),
                ruleset_name=ruleset.name,
            )
            composite = score_distribution(per_stat_dists, ruleset, seed=seed)

            out_rows.append(
                {
                    "gsis_id": str(gsis_id_col[row_idx]),
                    "season": int(season_col[row_idx]),
                    "week": int(week_col[row_idx]),
                    "position": self._config.position.value,
                    "team": str(team_col[row_idx]),
                    "opponent": str(opponent_col[row_idx]),
                    "ruleset": ruleset.name,
                    "family": DistributionFamily.QUANTILE.value,
                    "params": pack_per_stat_params(per_stat_dists),
                    "mean": composite.mean(),
                    "p10": composite.quantile(0.10),
                    "p50": composite.quantile(0.50),
                    "p90": composite.quantile(0.90),
                    "model_id": self.model_id,
                    "generated_at": pd.Timestamp(generated_at).as_unit("us"),
                }
            )
        out = pd.DataFrame(out_rows)
        for col in ("gsis_id", "team", "opponent", "ruleset", "family", "model_id"):
            out[col] = out[col].astype(_PYARROW_STR)
        out["position"] = out["position"].astype(_PYARROW_STR)
        return ProjectionWeeklySchema.validate(out)

    def save(self, path: Path) -> None:
        """Joblib-serialize the entire model.

        lgb.Booster instances pickle cleanly (lightgbm registers reduce/setstate
        hooks), so the whole LightGBMModel — config, sub-models, train window,
        and fitted flag — round-trips through joblib.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save() an unfitted model")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> LightGBMModel:
        """Inverse of save(). Returns the same instance shape as the original."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Loaded object is {type(loaded).__name__}, expected {cls.__name__}")
        return loaded
