"""Distribution interface — value object exposing mean/quantile/sample."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Distribution(Protocol):
    """A probability distribution over a single player's fantasy points (or
    underlying stat). Backings: parametric (Normal/Gamma), empirical-quantile,
    or sampled. Same surface regardless."""

    def mean(self) -> float: ...
    def std(self) -> float: ...
    def quantile(self, q: float) -> float: ...
    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]: ...
