"""Distribution layer — interface + parametric implementations + mixture + codec."""

from __future__ import annotations

from projections.distributions.base import Distribution
from projections.distributions.codec import pack_per_stat_params, unpack_per_stat_params
from projections.distributions.mixture import MixtureDistribution
from projections.distributions.parametric import (
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    ParametricStudentT,
)
from projections.distributions.quantile import QuantileDistribution
from projections.distributions.sampled import FrozenSampledDistribution

__all__ = [
    "Distribution",
    "FrozenSampledDistribution",
    "MixtureDistribution",
    "ParametricGamma",
    "ParametricNegativeBinomial",
    "ParametricNormal",
    "ParametricStudentT",
    "QuantileDistribution",
    "pack_per_stat_params",
    "unpack_per_stat_params",
]
