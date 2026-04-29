"""MixtureDistribution — weighted mixture of two child distributions.

Plan 6 (Model D ensemble). For per-(position, stat) ensemble of Model A and
Model C-NB, each row's per-stat distribution is a MixtureDistribution wrapping
the two child distributions and a scalar weight.

Mathematics:
    mean()      = w * F_a.mean() + (1-w) * F_b.mean()
    variance()  = w * F_a.var() + (1-w) * F_b.var()
                  + w * (1-w) * (F_a.mean() - F_b.mean())^2
    cdf(x)      = w * F_a.cdf(x) + (1-w) * F_b.cdf(x)
    quantile(q) = brentq solving cdf(x) - q = 0 over a bracket spanning both
                  components' tails.
    sample(n)   = vectorized Bernoulli(w) mask -> per-element child sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from projections.distributions.base import Distribution

_QUANTILE_EPS = 1e-9
_BRACKET_PADDING = 100.0


@dataclass(slots=True, frozen=True, init=False)
class MixtureDistribution:
    """Weighted mixture: P(X) = w * F_a(X) + (1-w) * F_b(X)."""

    component_a: Distribution
    component_b: Distribution
    weight: float

    def __init__(
        self,
        *,
        component_a: Distribution,
        component_b: Distribution,
        weight: float,
    ) -> None:
        if not (0.0 < weight < 1.0):
            raise ValueError(f"weight must lie strictly in (0, 1), got {weight}")
        object.__setattr__(self, "component_a", component_a)
        object.__setattr__(self, "component_b", component_b)
        object.__setattr__(self, "weight", float(weight))

    def mean(self) -> float:
        w = self.weight
        return w * self.component_a.mean() + (1.0 - w) * self.component_b.mean()

    def std(self) -> float:
        w = self.weight
        var_a = self.component_a.std() ** 2
        var_b = self.component_b.std() ** 2
        mean_a = self.component_a.mean()
        mean_b = self.component_b.mean()
        var = w * var_a + (1.0 - w) * var_b + w * (1.0 - w) * (mean_a - mean_b) ** 2
        # Numerical guard: floating-point cancellation can produce tiny negatives.
        return float(np.sqrt(max(var, 0.0)))

    def cdf(self, x: float) -> float:
        w = self.weight
        return w * self.component_a.cdf(x) + (1.0 - w) * self.component_b.cdf(x)

    def quantile(self, q: float) -> float:
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must lie strictly in (0, 1), got {q}")
        # Build a bracket [lo, hi] guaranteed to contain the q-quantile.
        # Distribution Protocol guarantees quantile(q) is finite for q in (0, 1),
        # so unconditional calls are safe.
        lo, hi = _bracket_for_components(self.component_a, self.component_b)
        return _quantile_with_bracket(self, q, lo, hi)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        rng = rng if rng is not None else np.random.default_rng()
        # Vectorized Bernoulli draw of which component to use.
        use_a = rng.random(size=n) < self.weight
        n_a = int(use_a.sum())
        n_b = n - n_a
        out = np.empty(n, dtype=np.float64)
        if n_a > 0:
            out[use_a] = self.component_a.sample(n=n_a, rng=rng)
        if n_b > 0:
            out[~use_a] = self.component_b.sample(n=n_b, rng=rng)
        return out


def _bracket_for_components(
    component_a: Distribution, component_b: Distribution
) -> tuple[float, float]:
    """Compute [lo, hi] bracket for any mixture of these two components,
    invariant in the mixture weight. Used by callers that build many
    mixtures over the same components with varying weights (e.g., the
    pinball weight optimizer)."""
    a_low = component_a.quantile(_QUANTILE_EPS)
    a_high = component_a.quantile(1.0 - _QUANTILE_EPS)
    b_low = component_b.quantile(_QUANTILE_EPS)
    b_high = component_b.quantile(1.0 - _QUANTILE_EPS)
    return (
        min(a_low, b_low) - _BRACKET_PADDING,
        max(a_high, b_high) + _BRACKET_PADDING,
    )


def _quantile_with_bracket(mix: MixtureDistribution, q: float, lo: float, hi: float) -> float:
    """Inverse of cdf at q given a pre-computed bracket [lo, hi].

    Caller is responsible for ensuring the bracket spans the support of the
    mixture's combined cdf. Used by EnsembleModel weight fitting to avoid
    re-deriving brackets on every optimizer iteration; for the standalone
    quantile() method, the bracket is computed inline from the children's
    quantiles. Both paths share the same brentq root-finder afterwards.

    Raises ValueError if q lies outside the joint support represented by
    the bracket (cdf(lo) > q or cdf(hi) < q).
    """
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must lie strictly in (0, 1), got {q}")
    # cdf is non-decreasing, so brentq on cdf(x) - q is well-defined --
    # provided q lies inside the joint support of the mixture. Children
    # whose cdf clamps (e.g., QuantileDistribution clamping at qs[0]) can
    # produce a mixture cdf that does not reach 0 or 1 at the bracket
    # endpoints. Raise explicitly in that case rather than silently
    # returning the endpoint.
    f_lo = mix.cdf(lo) - q
    f_hi = mix.cdf(hi) - q
    if f_lo > 0.0:
        raise ValueError(
            f"q={q} lies below joint support of mixture: cdf(lo={lo:.6g})={f_lo + q:.6g} "
            "(can happen when both children's cdfs clamp at a common minimum)"
        )
    if f_hi < 0.0:
        raise ValueError(
            f"q={q} lies above joint support of mixture: cdf(hi={hi:.6g})={f_hi + q:.6g} "
            "(can happen when both children's cdfs clamp at a common maximum)"
        )
    return float(brentq(lambda x: mix.cdf(x) - q, lo, hi, xtol=1e-9, rtol=1e-9))
