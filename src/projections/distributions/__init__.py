"""Distribution layer — interface + parametric implementations."""

from __future__ import annotations

from projections.distributions.base import Distribution
from projections.distributions.parametric import ParametricGamma, ParametricNormal

__all__ = ["Distribution", "ParametricGamma", "ParametricNormal"]
