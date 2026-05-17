"""Preseason projections — season-total per-player distributions produced from
data available before any games are played in the target season.

Parallel to the in-season weekly pipeline in `src/projections/models/`. Sized
to the season-total target rather than per-week. See
`docs/superpowers/specs/2026-05-17-preseason-projections-design.md`.
"""

from projections.preseason.features import build_preseason_features
from projections.preseason.model import (
    NaivePreseasonModel,
    NaivePriorOnlyModel,
    PreseasonModel,
)

__all__ = [
    "NaivePreseasonModel",
    "NaivePriorOnlyModel",
    "PreseasonModel",
    "build_preseason_features",
]
