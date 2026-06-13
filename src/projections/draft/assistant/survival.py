"""Probability a player is still available at a future pick, from market ADP.

The default `LogisticSurvival` uses a logistic CDF around ADP with a single
global spread `sigma` (in picks). The exact CDF shape is not load-bearing — it
is monotone and deterministic, and `sigma` is tuned empirically in Slice 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import groupby
from typing import Protocol, runtime_checkable

import numpy as np


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
        # `nan <= 0` is False, so guard NaN explicitly — otherwise every
        # p_available silently returns NaN and the whole column degrades to null.
        if math.isnan(self.sigma) or self.sigma <= 0:
            raise ValueError(f"sigma must be a positive number; got {self.sigma}")

    def p_available(self, adp: float, at_pick: int) -> float:
        # No market signal (missing ADP arrives as NaN from pandas) → treat as
        # "won't be taken soon".
        if math.isnan(adp):
            return 1.0
        # Available *at* `at_pick` ⇔ not taken on or before `at_pick - 1`.
        return 1.0 - _sigmoid((at_pick - 1 - adp) / self.sigma)


def expected_best_by_position(
    positions: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    tiebreak: np.ndarray,
) -> dict[str, float]:
    """Expected value of the best *surviving* player at each position.

    For each position, players are sorted by value descending (deterministic
    `tiebreak`, ascending, breaks ties), and the expected max over survivors is
    accumulated sequentially: ``value_i * p_i * prod_{better j}(1 - p_j)``.
    Canonical pre-sort makes the result independent of input row order; the
    sequential ``+=`` accumulation (rather than ``np.sum``) is bit-for-bit reproducible.
    Shared by now_or_never (values = VORP) and the season-value timing
    strategy (values = marginal season points).
    """
    if not len(positions) == len(values) == len(probs) == len(tiebreak):
        raise ValueError(
            "positions, values, probs, tiebreak must be equal length; got "
            f"{len(positions)}, {len(values)}, {len(probs)}, {len(tiebreak)}"
        )
    order = np.lexsort((tiebreak, -values, positions))
    out: dict[str, float] = {}
    rows = zip(
        positions[order].tolist(),
        values[order].tolist(),
        probs[order].tolist(),
        strict=True,
    )
    for position, group in groupby(rows, key=lambda r: r[0]):
        expected = 0.0
        prob_all_better_gone = 1.0
        for _, value_i, p_i in group:
            expected += value_i * p_i * prob_all_better_gone
            prob_all_better_gone *= 1.0 - p_i
        out[str(position)] = expected
    return out
