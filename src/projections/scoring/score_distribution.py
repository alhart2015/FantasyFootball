"""Convert a dict of per-stat distributions into a single fantasy-points distribution.

Strategy: Monte Carlo. Sample n times from each underlying-stat distribution,
score each sample under the ruleset, and return an empirical (sampled)
distribution backed by the resulting array.

Vectorized: the scoring rule is linear over per-stat counts/yards, so
points = sum_over_stats(coefficient(stat, ruleset) * samples_per_stat[stat]).
This eliminates the per-sample StatLine construction that previously
dominated runtime at backtest scale.

This is intentionally NOT analytic: real per-stat distributions are not Gaussian
and don't combine cleanly. Sampling lets us re-score under any ruleset for free.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from projections.distributions import Distribution
from projections.schemas import Ruleset, Stat
from projections.scoring.score import StatLine


def derive_row_seed(*, gsis_id: str, season: int, week: int, ruleset_name: str) -> int:
    """Stable 32-bit seed from (gsis_id, season, week, ruleset_name).

    Used by BaselineModel.predict_distribution and aggregate_to_season to keep
    per-row Monte Carlo draws independent and reproducible.

    Properties:
      - Deterministic across processes. Python's built-in hash() is
        salt-randomized via PYTHONHASHSEED; this uses sha256 instead.
      - Independent: changes to any of the four inputs change the seed.
      - Reproducible: identical inputs always produce identical samples
        downstream.

    Returns:
        An int in [0, 2**32).
    """
    h = hashlib.sha256(f"{gsis_id}|{season}|{week}|{ruleset_name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


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


INTEGER_STATS: Final[frozenset[Stat]] = _derive_integer_stats()


def _scoring_coefficients(ruleset: Ruleset) -> dict[Stat, float]:
    """Per-stat fantasy-point coefficient under ``ruleset``. Mirrors the
    13 contributions in ``score()`` exactly. Yards stats have inverse
    coefficients (1 / yds_per_pt) because score divides; everything else
    is a direct multiplier.

    Stats not in this map are not part of the scoring rule and would have
    been silently defaulted to 0 in the per-sample StatLine path or have
    raised Pydantic ValidationError if passed; the vectorized path
    fail-fasts in the validation step below.
    """
    return {
        Stat.PASSING_YARDS: 1.0 / ruleset.passing_yds_per_pt,
        Stat.PASSING_TDS: ruleset.passing_td_pts,
        Stat.INTERCEPTIONS: ruleset.interception_pts,
        Stat.PASSING_2PT: ruleset.two_pt_pts,
        Stat.RUSHING_YARDS: 1.0 / ruleset.rushing_yds_per_pt,
        Stat.RUSHING_TDS: ruleset.rushing_td_pts,
        Stat.RUSHING_2PT: ruleset.two_pt_pts,
        Stat.RECEPTIONS: ruleset.reception_pts,
        Stat.RECEIVING_YARDS: 1.0 / ruleset.receiving_yds_per_pt,
        Stat.RECEIVING_TDS: ruleset.receiving_td_pts,
        Stat.RECEIVING_2PT: ruleset.two_pt_pts,
        Stat.FUMBLES_LOST: ruleset.fumble_lost_pts,
        Stat.RETURN_TDS: ruleset.return_td_pts,
    }


def score_distribution(
    stat_dists: Mapping[Stat, Distribution],
    ruleset: Ruleset,
    *,
    n_samples: int = 10_000,
    seed: int | None = None,
) -> SampledDistribution:
    """Build a points distribution by sampling each stat distribution and scoring.

    Vectorized: math equivalent to summing per-stat (coefficient * samples)
    arrays. Stats with INTEGER_STATS membership are rounded and floored at 0
    before scoring (same convention as the prior per-sample loop).

    Raises:
        ValueError: ``stat_dists`` contains a stat that has no coefficient
            in the ruleset (would have raised Pydantic ValidationError in
            the legacy code path).
    """
    rng = np.random.default_rng(seed)
    coef_map = _scoring_coefficients(ruleset)

    unknown = set(stat_dists.keys()) - set(coef_map.keys())
    if unknown:
        names = sorted(s.value for s in unknown)
        raise ValueError(
            f"Cannot score stats not in scoring rule: {names}. "
            f"Known scorable stats: {sorted(s.value for s in coef_map)}"
        )

    points = np.zeros(n_samples, dtype=np.float64)
    for stat, dist in stat_dists.items():
        s = dist.sample(n_samples, rng=rng)
        if stat in INTEGER_STATS:
            # Round to non-negative integers; floor at 0 since count stats can't be negative.
            s = np.maximum(np.rint(s), 0.0)
        coef = coef_map[stat]
        points += s * coef

    return SampledDistribution(samples=points)
