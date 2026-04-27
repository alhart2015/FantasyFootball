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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import lightgbm as lgb
import pandas as pd
import pandera.pandas as pa

from projections.distributions import (
    QuantileDistribution,  # noqa: F401  -- consumed by Task 7's predict_distribution
    pack_per_stat_params,  # noqa: F401  -- consumed by Task 7's predict_distribution
)
from projections.features.qb import build_qb_features
from projections.features.rb import build_rb_features
from projections.features.te import build_te_features
from projections.features.wr import build_wr_features
from projections.models.base import compute_code_hash
from projections.schemas import (
    DistributionFamily,  # noqa: F401  -- consumed by Task 7's predict_distribution
    Position,
    ProjectionWeeklySchema,  # noqa: F401  -- consumed by Task 7's predict_distribution
    QbFeaturesSchema,
    RbFeaturesSchema,
    Ruleset,
    Stat,
    TeFeaturesSchema,
    WrFeaturesSchema,
)
from projections.scoring.score_distribution import (
    derive_row_seed,  # noqa: F401  -- consumed by Task 7's predict_distribution
    score_distribution,  # noqa: F401  -- consumed by Task 7's predict_distribution
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
        feature_builder: position-specific build_*_features for code-hash purposes.
    """

    position: Position
    target_stats: tuple[Stat, ...]
    feature_columns: tuple[str, ...]
    feature_schema: type[pa.DataFrameModel]
    non_negative_stats: frozenset[Stat]
    feature_builder: Any  # callable; signature varies per position


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
            feature_builder=build_qb_features,
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
            feature_builder=build_rb_features,
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
            feature_builder=build_te_features,
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
            feature_builder=build_wr_features,
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
    def model_id(self) -> str:
        if not self._is_fitted:
            raise RuntimeError(
                "model_id not available before fit() — depends on training-time state"
            )
        code_hash = compute_code_hash(_code_hash_files(self._config.position))
        assert self._train_start is not None and self._train_end is not None
        return (
            f"lightgbm:{self._config.position.value.lower()}:{code_hash}"
            f":{self._train_start}-{self._train_end}"
        )

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        raise NotImplementedError("Plan 5 Task 6")

    def predict_distribution(self, features: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
        raise NotImplementedError("Plan 5 Task 7")

    def save(self, path: Path) -> None:
        raise NotImplementedError("Plan 5 Task 8")

    @classmethod
    def load(cls, path: Path) -> LightGBMModel:
        raise NotImplementedError("Plan 5 Task 8")
