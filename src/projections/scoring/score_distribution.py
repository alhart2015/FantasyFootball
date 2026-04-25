"""Convert a dict of per-stat distributions into a single fantasy-points distribution.

Strategy: Monte Carlo. Sample n times from each underlying-stat distribution,
score each row through the scoring function, and return an empirical (sampled)
distribution backed by the resulting array.

This is intentionally NOT analytic: real per-stat distributions are not Gaussian
and don't combine cleanly. Sampling lets us re-score under any ruleset for free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from projections.distributions import Distribution
from projections.schemas import Ruleset, Stat
from projections.scoring.score import StatLine, score


@dataclass(slots=True, frozen=True)
class SampledDistribution:
    """Empirical distribution backed by a samples array."""

    samples: NDArray[np.float64]

    def mean(self) -> float:
        return float(self.samples.mean())

    def std(self) -> float:
        return float(self.samples.std())

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        return float(np.quantile(self.samples, q))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        rng = rng if rng is not None else np.random.default_rng()
        return rng.choice(self.samples, size=n, replace=True)


def _derive_integer_stats() -> frozenset[Stat]:
    """Programmatically determine which Stat enum members map to integer
    fields on StatLine. Single source of truth: if StatLine adds a new int
    field, this set updates without manual edits."""
    int_field_names = {
        name for name, field in StatLine.model_fields.items() if field.annotation is int
    }
    return frozenset(stat for stat in Stat if stat.value in int_field_names)


_INTEGER_STATS: frozenset[Stat] = _derive_integer_stats()


def score_distribution(
    stat_dists: Mapping[Stat, Distribution],
    ruleset: Ruleset,
    *,
    n_samples: int = 10_000,
    seed: int | None = None,
) -> SampledDistribution:
    """Build a points distribution by sampling each stat distribution and scoring."""
    rng = np.random.default_rng(seed)

    # Sample each stat n_samples times. Missing stats default to 0.
    samples_per_stat: dict[Stat, NDArray[np.float64]] = {}
    for stat, dist in stat_dists.items():
        s = dist.sample(n_samples, rng=rng)
        if stat in _INTEGER_STATS:
            # Round to non-negative integers; floor at 0 since count stats can't be negative.
            s = np.maximum(np.rint(s), 0.0)
        samples_per_stat[stat] = s

    # Score each row.
    # TODO(perf): vectorize this Python loop. Each iteration constructs a
    # Pydantic StatLine instance (~5-10us); at backtest scale (~500 players x
    # 17 weeks x 10k samples = 85M iterations) this dominates. The score()
    # math is linear and can be expressed as a dot product over per-stat
    # arrays without per-row object construction.
    points = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        kwargs: dict[str, float | int] = {}
        for stat, arr in samples_per_stat.items():
            kwargs[stat.value] = arr[i] if stat not in _INTEGER_STATS else int(arr[i])
        points[i] = score(StatLine(**kwargs), ruleset)  # type: ignore[arg-type]

    return SampledDistribution(samples=points)
