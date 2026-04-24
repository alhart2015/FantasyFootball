"""Parametric distribution backings: Normal and Gamma."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(slots=True, frozen=True, init=False)
class ParametricNormal:
    mean_: float
    std_: float

    def __init__(self, mean: float, std: float) -> None:
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")
        object.__setattr__(self, "mean_", float(mean))
        object.__setattr__(self, "std_", float(std))

    def mean(self) -> float:
        return self.mean_

    def std(self) -> float:
        return self.std_

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        return float(stats.norm.ppf(q, loc=self.mean_, scale=self.std_))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        rng = rng if rng is not None else np.random.default_rng()
        return rng.normal(loc=self.mean_, scale=self.std_, size=n).astype(np.float64)


@dataclass(slots=True, frozen=True)
class ParametricGamma:
    """Shape (k) / scale (theta) parameterization. mean = k*theta, var = k*theta^2."""

    shape: float
    scale: float

    def __post_init__(self) -> None:
        if self.shape <= 0 or self.scale <= 0:
            raise ValueError(f"shape and scale must be positive; got {self.shape}, {self.scale}")

    def mean(self) -> float:
        return self.shape * self.scale

    def std(self) -> float:
        return float(np.sqrt(self.shape) * self.scale)

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        return float(stats.gamma.ppf(q, a=self.shape, scale=self.scale))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        rng = rng if rng is not None else np.random.default_rng()
        return rng.gamma(shape=self.shape, scale=self.scale, size=n).astype(np.float64)
