"""Parametric distribution backings: Normal, Gamma, Negative Binomial, Student-t."""

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
        # Coerce ints to floats so callers passing `shape=4` get float behavior — matches
        # ParametricNormal's __init__ discipline and avoids surprise int returns from mean().
        object.__setattr__(self, "shape", float(self.shape))
        object.__setattr__(self, "scale", float(self.scale))
        if self.shape <= 0:
            raise ValueError(f"shape must be positive, got {self.shape}")
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

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


@dataclass(slots=True, frozen=True, init=False)
class ParametricNegativeBinomial:
    """Negative Binomial parameterized as (mean, dispersion).

    Standard NB-2 / "size" parameterization. Internally uses scipy's (n, p):
        n = dispersion                        (the "size" / shape parameter)
        p = dispersion / (dispersion + mean)
    Variance: var = mean + mean^2 / dispersion (overdispersed vs Poisson when
    dispersion is finite; recovers Poisson as dispersion -> inf).

    Suitable for low-mean integer counts where the assumed GAMMA family
    cannot represent a point mass at zero (Plan 3e Phase 1 use case).
    """

    mean_: float
    dispersion_: float

    def __init__(self, mean: float, dispersion: float) -> None:
        if mean <= 0:
            raise ValueError(f"mean must be positive, got {mean}")
        if dispersion <= 0:
            raise ValueError(f"dispersion must be positive, got {dispersion}")
        object.__setattr__(self, "mean_", float(mean))
        object.__setattr__(self, "dispersion_", float(dispersion))

    def _scipy_n_p(self) -> tuple[float, float]:
        # Standard NB-2: n = dispersion (the "size" parameter), p such that
        # mean = n*(1-p)/p ⟹ p = dispersion / (dispersion + mean).
        # Yields var = n*(1-p)/p² = mean + mean²/dispersion (matches std()).
        n = self.dispersion_
        p = self.dispersion_ / (self.dispersion_ + self.mean_)
        return n, p

    def mean(self) -> float:
        return self.mean_

    def std(self) -> float:
        var = self.mean_ + self.mean_ * self.mean_ / self.dispersion_
        return float(np.sqrt(var))

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        n, p = self._scipy_n_p()
        return float(stats.nbinom.ppf(q, n=n, p=p))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        # `n` is the protocol-conforming sample-count param. Rename the NB
        # shape parameter to `n_size` locally to avoid the collision.
        rng = rng if rng is not None else np.random.default_rng()
        n_size, p = self._scipy_n_p()
        # scipy's rvs is typed as Any; wrap in np.asarray so the return type is
        # a properly-typed NDArray[np.float64] instead of Any.
        raw = stats.nbinom.rvs(n=n_size, p=p, size=n, random_state=rng)
        return np.asarray(raw, dtype=np.float64)


@dataclass(slots=True, frozen=True, init=False)
class ParametricStudentT:
    """Student-t parameterized as (loc, scale, df).

    Mean = loc (for df > 1).
    Variance = scale^2 * df / (df - 2) (for df > 2).

    Plan 3e Phase 2 routes heavy-tailed continuous stats (yards-shaped) here,
    using the per-row Ridge prediction as `loc` and globally-fit (scale, df)
    from training residuals.
    """

    loc_: float
    scale_: float
    df_: float

    def __init__(self, loc: float, scale: float, df: float) -> None:
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if df <= 2:
            raise ValueError(f"df must be greater than 2 for finite variance, got {df}")
        object.__setattr__(self, "loc_", float(loc))
        object.__setattr__(self, "scale_", float(scale))
        object.__setattr__(self, "df_", float(df))

    def mean(self) -> float:
        return self.loc_

    def std(self) -> float:
        return float(self.scale_ * np.sqrt(self.df_ / (self.df_ - 2.0)))

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        return float(stats.t.ppf(q, df=self.df_, loc=self.loc_, scale=self.scale_))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        rng = rng if rng is not None else np.random.default_rng()
        # scipy's rvs is typed as Any; wrap in np.asarray so the return type is
        # a properly-typed NDArray[np.float64] instead of Any.
        raw = stats.t.rvs(df=self.df_, loc=self.loc_, scale=self.scale_, size=n, random_state=rng)
        return np.asarray(raw, dtype=np.float64)
