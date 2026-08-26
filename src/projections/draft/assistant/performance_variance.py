"""Two-component weekly fantasy-point variance model (spec 2026-06-14).

Component ii: mean-preserving lognormal season-mean multiplier ``m`` (E[m]=1, log-SD by
position x rookie). Component i: per-game Gamma weekly noise, ``std = a_pos*pg + b_pos``.
Params are fit offline (``scripts/fit_performance_variance.py``) and committed to
``configs/performance_variance_params.json``; this module only loads + samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Where the fitted parameters ship. Relative, so it resolves against the process CWD --
#: callers that may run from elsewhere should check it exists and say so rather than
#: letting `load()` raise a bare FileNotFoundError.
DEFAULT_PARAMS_PATH = Path("configs/performance_variance_params.json")
#: Games the variance model spreads a season projection over. Public because callers that
#: build a season_mean_fpts figure must express it over the SAME horizon this divides by --
#: `rest_of_season.rest_of_season_pool` re-scales a partial-season pace through it.
SEASON_GAMES = 17
_GAMES = SEASON_GAMES  # retained: referenced below and in tests


@dataclass(frozen=True)
class VarianceParams:
    """Fitted variance parameters. ``weekly_std_affine`` maps position -> {"a","b"} (with a
    required "default" key); ``mean_mult_log_sd`` maps "<pos>|<veteran|rookie>" -> lognormal
    log-SD (with required "default|veteran" / "default|rookie" keys)."""

    weekly_std_affine: dict[str, dict[str, float]]
    mean_mult_log_sd: dict[str, float]

    @classmethod
    def load(cls, path: Path = DEFAULT_PARAMS_PATH) -> VarianceParams:
        blob = json.loads(Path(path).read_text())
        return cls(blob["weekly_std_affine"], blob["mean_mult_log_sd"])

    def weekly_std(self, position: str, per_game_mean: float) -> float:
        """Per-game weekly std at ``per_game_mean`` for ``position`` (default affine if unknown)."""
        coef = self.weekly_std_affine.get(position) or self.weekly_std_affine["default"]
        return coef["a"] * per_game_mean + coef["b"]

    def log_sd(self, position: str, *, is_rookie: bool) -> float:
        """Lognormal log-SD of the season-mean multiplier for (position, rookie tier)."""
        tier = "rookie" if is_rookie else "veteran"
        return self.mean_mult_log_sd.get(
            f"{position}|{tier}", self.mean_mult_log_sd[f"default|{tier}"]
        )


def sample_weekly_points(
    params: VarianceParams,
    positions: np.ndarray,
    projected_means: np.ndarray,
    is_rookie: np.ndarray,
    *,
    n_sims: int,
    n_weeks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``(n_sims, n_weeks, n_players)`` non-negative sampled weekly points.

    Component ii (per sim, player): mean-preserving lognormal ``m`` with ``E[m]=1`` ->
    ``true_mean = projected_mean * m``. Component i (per sim, week, player): Gamma with
    per-game mean ``pg = true_mean/GAMES`` and ``std = f_pos(pg)``. ``projected_mean <= 0`` -> 0.
    """
    n = len(positions)
    log_sd = np.array(
        [
            params.log_sd(str(p), is_rookie=bool(r))
            for p, r in zip(positions, is_rookie, strict=True)
        ],
        dtype=np.float64,
    )
    # Mean-preserving lognormal: mu = -sigma^2/2 so E[m] = 1.
    mu = -0.5 * log_sd**2
    m = rng.lognormal(mean=mu, sigma=log_sd, size=(n_sims, n))  # (n_sims, n)
    true_mean = projected_means.astype(np.float64)[None, :] * m  # (n_sims, n)
    pg = true_mean / _GAMES  # per-game mean

    coefs = [
        params.weekly_std_affine.get(str(p)) or params.weekly_std_affine["default"]
        for p in positions
    ]
    a = np.array([c["a"] for c in coefs])
    b = np.array([c["b"] for c in coefs])
    std = np.maximum(a[None, :] * pg + b[None, :], 1e-9)  # (n_sims, n) per-game weekly std

    valid = pg > 0
    # Gamma(shape=k, scale=theta): mean = k*theta = pg, var = k*theta^2 = std^2
    # -> k = (pg/std)^2, theta = std^2/pg. Guard invalid (pg<=0) cells, zeroed below.
    safe_pg = np.where(valid, pg, 1.0)
    k = np.where(valid, (pg / std) ** 2, 1.0)
    theta = np.where(valid, std**2 / safe_pg, 0.0)
    shape3 = np.broadcast_to(k[:, None, :], (n_sims, n_weeks, n))
    scale3 = np.broadcast_to(theta[:, None, :], (n_sims, n_weeks, n))
    pts = rng.gamma(shape=shape3, scale=scale3)  # 0 where theta == 0
    return np.where(valid[:, None, :], pts, 0.0)
