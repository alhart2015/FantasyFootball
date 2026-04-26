"""Public surface for the models package.

Plan 3b adds POSITION_DISPATCH so callers (CLI scripts, Plan 3c backtest
harness) can dispatch by Position to the correct factory + feature builder
+ feature schema + NGS source. Adding a new position is one new line in
this registry plus a corresponding factory in baseline.py and a feature
builder in features/{pos}.py.
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
    "Model",
    "compute_code_hash",
    "qb_baseline",
    "rb_baseline",
    "te_baseline",
    "wr_baseline",
]


@dataclass(frozen=True)
class _PositionDispatch:
    """Per-position bundle of "what's needed to train and predict" entries.

    Consumed by the CLI scripts (scripts/train_baseline.py etc.) and
    intended to back Plan 3c's backtest harness. Frozen so callers can't
    mutate the registry by accident.

    Attributes:
        factory: zero-arg callable returning an unfitted BaselineModel.
        feature_builder: position-specific build_*_features function.
        feature_schema: pandera schema for the feature builder's output.
        ngs_stat_type: which NGS partition the feature builder consumes
            ("passing" / "rushing" / "receiving").
    """

    factory: Callable[[], BaselineModel]
    feature_builder: Callable[..., Any]
    feature_schema: type[pa.DataFrameModel]
    ngs_stat_type: NgsStatType


POSITION_DISPATCH: Mapping[Position, _PositionDispatch] = {
    Position.QB: _PositionDispatch(
        factory=qb_baseline,
        feature_builder=build_qb_features,
        feature_schema=QbFeaturesSchema,
        ngs_stat_type="passing",
    ),
    Position.RB: _PositionDispatch(
        factory=rb_baseline,
        feature_builder=build_rb_features,
        feature_schema=RbFeaturesSchema,
        ngs_stat_type="rushing",
    ),
    Position.TE: _PositionDispatch(
        factory=te_baseline,
        feature_builder=build_te_features,
        feature_schema=TeFeaturesSchema,
        ngs_stat_type="receiving",
    ),
    Position.WR: _PositionDispatch(
        factory=wr_baseline,
        feature_builder=build_wr_features,
        feature_schema=WrFeaturesSchema,
        ngs_stat_type="receiving",
    ),
}
