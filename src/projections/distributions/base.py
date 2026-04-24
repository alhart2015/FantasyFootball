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

    # NOTE: @runtime_checkable enables isinstance() checks but performs structural
    # (attribute-presence) checking only — it does NOT verify method signatures or
    # return types. Trust mypy for that, not isinstance.

    def mean(self) -> float: ...
    def std(self) -> float: ...
    def quantile(self, q: float) -> float: ...
    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]: ...
