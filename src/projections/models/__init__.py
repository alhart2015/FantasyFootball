"""Public surface for the models package.

Plan 3b adds POSITION_DISPATCH so callers (CLI scripts, Plan 3c backtest
harness) can dispatch by Position to the correct factory + feature builder
+ feature schema + NGS source. Adding a new position is one new line in
this registry plus a corresponding factory in baseline.py and a feature
builder in features/{pos}.py.

Plan 5 generalizes the registry: each position now carries a `factories`
mapping keyed by model class ("baseline" -> Model A, "lightgbm" -> Model C)
so the same dispatch tables drive both training pipelines.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandera.pandas as pa

from projections.features.qb import build_qb_features
from projections.features.rb import build_rb_features
from projections.features.te import build_te_features
from projections.features.wr import build_wr_features
from projections.ingest.ngs import NgsStatType
from projections.models.base import Model, compute_code_hash
from projections.models.baseline import (
    BaselineModel,
    qb_baseline,
    rb_baseline,
    te_baseline,
    wr_baseline,
)
from projections.models.lightgbm import (
    LightGBMModel,
    qb_lightgbm,
    rb_lightgbm,
    te_lightgbm,
    wr_lightgbm,
)
from projections.models.lightgbm_tuned import (
    LightGBMTunedModel,
    qb_lightgbm_tuned,
    rb_lightgbm_tuned,
    te_lightgbm_tuned,
    wr_lightgbm_tuned,
)
from projections.schemas import (
    Position,
    QbFeaturesSchema,
    RbFeaturesSchema,
    TeFeaturesSchema,
    WrFeaturesSchema,
)

__all__ = [
    "POSITION_DISPATCH",
    "BaselineModel",
    "LightGBMModel",
    "LightGBMTunedModel",
    "Model",
    "compute_code_hash",
    "qb_baseline",
    "qb_lightgbm",
    "qb_lightgbm_tuned",
    "rb_baseline",
    "rb_lightgbm",
    "rb_lightgbm_tuned",
    "te_baseline",
    "te_lightgbm",
    "te_lightgbm_tuned",
    "wr_baseline",
    "wr_lightgbm",
    "wr_lightgbm_tuned",
]


@dataclass(frozen=True)
class _PositionDispatch:
    """Per-position bundle of "what's needed to train and predict" entries.

    Consumed by the CLI scripts (scripts/train_baseline.py etc.) and Plan
    3c's backtest harness. Frozen so callers can't mutate the registry by
    accident.

    Attributes:
        factories: mapping of model-class identifier -> zero-arg callable
            returning an unfitted model. Keys today: "baseline" (Model A,
            BaselineModel) and "lightgbm" (Model C, LightGBMModel).
            Callers select via `dispatch.factories[args.model]()`.
        feature_builder: position-specific build_*_features function.
        feature_schema: pandera schema for the feature builder's output.
        ngs_stat_type: which NGS partition the feature builder consumes
            ("passing" / "rushing" / "receiving").
    """

    factories: Mapping[str, Callable[[], Model]]
    feature_builder: Callable[..., Any]
    feature_schema: type[pa.DataFrameModel]
    ngs_stat_type: NgsStatType


# Explicit per-position factories dicts. Annotated as
# `dict[str, Callable[[], Model]]` so the BaselineModel / LightGBMModel
# return types widen to the Model Protocol — `Callable` is covariant in its
# return type, but `dict` value types are invariant, so we declare the
# widened dict explicitly rather than relying on inference.
_QB_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": qb_baseline,
    "lightgbm": qb_lightgbm,
    "lightgbm-tuned": qb_lightgbm_tuned,
}
_RB_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": rb_baseline,
    "lightgbm": rb_lightgbm,
    "lightgbm-tuned": rb_lightgbm_tuned,
}
_TE_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": te_baseline,
    "lightgbm": te_lightgbm,
    "lightgbm-tuned": te_lightgbm_tuned,
}
_WR_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": wr_baseline,
    "lightgbm": wr_lightgbm,
    "lightgbm-tuned": wr_lightgbm_tuned,
}


POSITION_DISPATCH: Mapping[Position, _PositionDispatch] = {
    Position.QB: _PositionDispatch(
        factories=_QB_FACTORIES,
        feature_builder=build_qb_features,
        feature_schema=QbFeaturesSchema,
        ngs_stat_type="passing",
    ),
    Position.RB: _PositionDispatch(
        factories=_RB_FACTORIES,
        feature_builder=build_rb_features,
        feature_schema=RbFeaturesSchema,
        ngs_stat_type="rushing",
    ),
    Position.TE: _PositionDispatch(
        factories=_TE_FACTORIES,
        feature_builder=build_te_features,
        feature_schema=TeFeaturesSchema,
        ngs_stat_type="receiving",
    ),
    Position.WR: _PositionDispatch(
        factories=_WR_FACTORIES,
        feature_builder=build_wr_features,
        feature_schema=WrFeaturesSchema,
        ngs_stat_type="receiving",
    ),
}
