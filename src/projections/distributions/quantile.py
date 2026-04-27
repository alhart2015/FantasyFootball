"""QuantileDistribution — interpolated CDF/quantile/sample backed by stored knots.

Plan 5 (Model C / LightGBM with quantile regression). Each row's stat
distribution is represented by a sorted (quantiles, values) array pair.
quantile(q) linearly interpolates between adjacent knots; mean()/std()
are computed by trapezoid integration of the quantile function;
sample(n, rng) uses inverse-CDF (uniform draws -> np.interp).

No scipy dependency — pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

# Trapezoid integration grid for mean()/std(). 100 points over (0.01, 0.99)
# leaves a small tail at each end that is dominated by the linear-extrapolation
# from the outermost stored knots; sufficient for E[X] / Var[X] to within
# the per-cell snapshot tolerance.
_INTEGRATION_GRID: Final[NDArray[np.float64]] = np.linspace(0.01, 0.99, 100)


@dataclass(slots=True, frozen=True, init=False, eq=False)
class QuantileDistribution:
    """Distribution backed by a sorted set of (quantile, value) knots.

    Implements the Distribution Protocol structurally (mean, std, quantile, sample).

    ``eq=False`` is required because the auto-generated ``__eq__`` from
    ``@dataclass`` does field-by-field ``==``, which on ``NDArray`` returns an
    array (not bool) and raises ``ValueError``. We override ``__eq__`` and
    ``__hash__`` explicitly below.
    """

    quantiles_: NDArray[np.float64]
    values_: NDArray[np.float64]

    def __init__(
        self,
        quantiles: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> None:
        q = np.asarray(quantiles, dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)
        if q.shape != v.shape:
            raise ValueError(
                f"quantiles and values must have matching length: {q.shape} vs {v.shape}"
            )
        if q.size < 2:
            raise ValueError(f"need at least 2 knots, got {q.size}")
        if not np.all(np.diff(q) > 0):
            raise ValueError(f"quantiles must be strictly ascending, got {q}")
        if not np.all((q > 0.0) & (q < 1.0)):
            raise ValueError(f"quantiles must lie strictly in (0, 1), got {q}")
        if not np.all(np.diff(v) >= 0):
            raise ValueError(f"values must be non-decreasing, got {v}")
        object.__setattr__(self, "quantiles_", q)
        object.__setattr__(self, "values_", v)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuantileDistribution):
            return NotImplemented
        return np.array_equal(self.quantiles_, other.quantiles_) and np.array_equal(
            self.values_, other.values_
        )

    def __hash__(self) -> int:
        # tobytes() of an immutable numpy array gives a stable hash; the arrays
        # are the value identity since they back quantile/sample/mean/std.
        return hash((self.quantiles_.tobytes(), self.values_.tobytes()))

    def quantile(self, q: float) -> float:
        """Return the value at quantile q.

        For q within [q_min, q_max], linearly interpolates between adjacent stored knots.
        For q outside that range, linearly extrapolates from the two nearest knots.
        """
        # np.interp handles in-range linear interpolation. For out-of-range,
        # we extrapolate manually using the slope of the nearest two knots.
        qs = self.quantiles_
        vs = self.values_
        if qs[0] <= q <= qs[-1]:
            return float(np.interp(q, qs, vs))
        if q < qs[0]:
            slope = (vs[1] - vs[0]) / (qs[1] - qs[0])
            return float(vs[0] + slope * (q - qs[0]))
        # q > qs[-1]
        slope = (vs[-1] - vs[-2]) / (qs[-1] - qs[-2])
        return float(vs[-1] + slope * (q - qs[-1]))

    def _quantile_vec(self, qs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Vectorized quantile() for the integration grid + sampling.

        In-range points use np.interp; out-of-range points are linearly
        extrapolated from the two-nearest knots on the relevant side.
        """
        out = np.interp(qs, self.quantiles_, self.values_)
        # Lower-tail extrapolation
        below = qs < self.quantiles_[0]
        if below.any():
            slope_lo = (self.values_[1] - self.values_[0]) / (
                self.quantiles_[1] - self.quantiles_[0]
            )
            out[below] = self.values_[0] + slope_lo * (qs[below] - self.quantiles_[0])
        # Upper-tail extrapolation
        above = qs > self.quantiles_[-1]
        if above.any():
            slope_hi = (self.values_[-1] - self.values_[-2]) / (
                self.quantiles_[-1] - self.quantiles_[-2]
            )
            out[above] = self.values_[-1] + slope_hi * (qs[above] - self.quantiles_[-1])
        return out

    def mean(self) -> float:
        """E[X] via trapezoid integration of the quantile function on _INTEGRATION_GRID."""
        vs = self._quantile_vec(_INTEGRATION_GRID)
        return float(
            np.trapezoid(vs, _INTEGRATION_GRID) / (_INTEGRATION_GRID[-1] - _INTEGRATION_GRID[0])
        )

    def std(self) -> float:
        """Var[X] = E[X^2] - mean^2; std = sqrt(Var). Same integration approach as mean()."""
        vs = self._quantile_vec(_INTEGRATION_GRID)
        e_x = np.trapezoid(vs, _INTEGRATION_GRID) / (_INTEGRATION_GRID[-1] - _INTEGRATION_GRID[0])
        e_x2 = np.trapezoid(vs * vs, _INTEGRATION_GRID) / (
            _INTEGRATION_GRID[-1] - _INTEGRATION_GRID[0]
        )
        var = max(e_x2 - e_x * e_x, 0.0)
        return float(np.sqrt(var))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        """Inverse-CDF sampling: u ~ U(0, 1); return quantile(u) elementwise."""
        rng = rng if rng is not None else np.random.default_rng()
        u = rng.uniform(0.0, 1.0, size=n)
        return self._quantile_vec(u)
