"""Sampled-distribution implementations.

Hosts:
- ``FrozenSampledDistribution`` — empirical distribution whose ``.sample(n)``
  returns the underlying ``samples`` array verbatim when ``n == len(samples)``,
  preserving any external row-aligned correlation structure. Used by
  ``DecomposedBaselineModel`` to keep within-row cross-stat correlation
  intact when ``score_distribution`` consumes multiple decomposed-stat
  distributions that share an underlying volume draw.

Cf. ``projections.scoring.score_distribution.SampledDistribution``, which
always re-samples via ``rng.choice``. The two classes are deliberately
separate: ``SampledDistribution`` is the *output* of scoring (a points
distribution that callers re-sample from), ``FrozenSampledDistribution``
is an *input* to scoring (a per-stat distribution whose internal sample
ordering carries information that must not be shuffled).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class FrozenSampledDistribution:
    """Distribution-Protocol-conforming dataclass backed by a frozen samples array.

    ``sample(n)`` returns ``self.samples`` directly when ``n == len(self.samples)``;
    otherwise falls back to ``rng.choice`` for bootstrap-style re-sampling.

    The ``n == len`` branch is the architectural guarantee that lets
    ``DecomposedBaselineModel`` plumb cross-stat correlation through
    ``score_distribution`` without any modifications to the scoring path.
    """

    samples: NDArray[np.float64]

    def mean(self) -> float:
        return float(self.samples.mean())

    def std(self) -> float:
        return float(self.samples.std())

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        return float(np.quantile(self.samples, q))

    def cdf(self, x: float) -> float:
        return float((self.samples <= x).mean())

    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        if n == len(self.samples):
            return self.samples
        rng = rng if rng is not None else np.random.default_rng()
        return rng.choice(self.samples, size=n, replace=True)
