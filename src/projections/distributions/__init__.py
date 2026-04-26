"""Distribution layer — interface + parametric implementations + codec."""

from __future__ import annotations

from projections.distributions.base import Distribution
from projections.distributions.codec import pack_per_stat_params, unpack_per_stat_params
from projections.distributions.parametric import ParametricGamma, ParametricNormal

__all__ = [
    "Distribution",
    "ParametricGamma",
    "ParametricNormal",
    "pack_per_stat_params",
    "unpack_per_stat_params",
]
