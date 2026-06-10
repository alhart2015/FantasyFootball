"""Probability a player is still available at a future pick, from market ADP.

The default `LogisticSurvival` uses a logistic CDF around ADP with a single
global spread `sigma` (in picks). The exact CDF shape is not load-bearing — it
is monotone and deterministic, and `sigma` is tuned empirically in Slice 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class SurvivalModel(Protocol):
    def p_available(self, adp: float, at_pick: int) -> float:
        """P(player with this ADP is still on the board *at* `at_pick`)."""
        ...


def default_sigma(n_teams: int) -> float:
    """Spread default ≈ two-thirds of one round (picks)."""
    return (2.0 / 3.0) * n_teams


def _sigmoid(x: float) -> float:
    # Numerically stable logistic.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class LogisticSurvival:
    """Logistic survival in ADP space. `sigma` is the spread in picks."""

    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0; got {self.sigma}")

    def p_available(self, adp: float, at_pick: int) -> float:
        # No market signal → treat as "won't be taken soon".
        if adp is None or math.isnan(adp):
            return 1.0
        # Available *at* `at_pick` ⇔ not taken on or before `at_pick - 1`.
        return 1.0 - _sigmoid((at_pick - 1 - adp) / self.sigma)
