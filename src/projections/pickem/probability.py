"""Win probabilities from consensus moneylines.

This module is the *only* source of "who is likely to win" in the pick'em
pipeline. The organizer's spread never reaches it — that spread decides which
team counts as the underdog and nothing else.

Why moneylines rather than a spread-to-probability curve: the moneyline already
*is* the market's probability estimate, so no model is needed. Availability was
audited across 2010-2025 (4,175 regular-season games) plus 2026 upcoming games:
there is not one row carrying a spread but no moneyline, so a spread-based
fallback would be dead code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.pickem._validate import require_schedule_columns

# `game_id` is required, not incidental: both raises below name the offending
# game_id, so a frame without it would die on a bare KeyError instead of the
# diagnostic this module exists to give.
_REQUIRED_COLUMNS = ("game_id", "home_moneyline", "away_moneyline")


def american_to_implied(odds: int) -> float:
    """Convert American odds to the implied probability the bet wins.

    Negative odds are the favorite (risk `-odds` to win 100); positive odds are
    the underdog (risk 100 to win `odds`). The result still contains the
    bookmaker's margin — see `devig_pair` to remove it.
    """
    if odds == 0:
        raise ValueError("American odds of 0 are not a valid price")
    if odds < 0:
        return float(-odds) / (float(-odds) + 100.0)
    return 100.0 / (float(odds) + 100.0)


def devig_pair(home_odds: int, away_odds: int) -> tuple[float, float]:
    """Strip the bookmaker's margin from a two-way market, returning fair
    (home, away) probabilities that sum to 1.

    Both implied probabilities are inflated so the book profits either way;
    their sum exceeds 1 by roughly the margin. We remove it by proportional
    normalization, the standard treatment for a two-way market.

    Proportional devigging slightly overstates heavy favorites relative to
    methods that model the margin as non-uniform (Shin, power). At the margins
    this pipeline works with, the difference is well under the noise in the
    picks themselves; revisit only if heavy favorites start looking mispriced.
    """
    home_raw = american_to_implied(home_odds)
    away_raw = american_to_implied(away_odds)
    total = home_raw + away_raw
    if total <= 0:  # pragma: no cover - unreachable for valid American odds
        raise ValueError(f"non-positive implied total for odds {home_odds}/{away_odds}")
    return home_raw / total, away_raw / total


def _implied_array(odds: np.ndarray) -> np.ndarray:
    """Vectorized `american_to_implied`. Callers must reject zeros first.

    Each branch is evaluated only on its own subset rather than via `np.where`.
    `np.where` computes both sides for every element, and at even money (a
    moneyline of exactly +100 or -100, a common price) the discarded branch
    divides by zero — harmless in the result but it raises a RuntimeWarning and
    builds an inf on the way through.
    """
    out = np.empty(odds.shape, dtype="float64")
    favorite = odds < 0
    out[favorite] = -odds[favorite] / (-odds[favorite] + 100.0)
    out[~favorite] = 100.0 / (odds[~favorite] + 100.0)
    return out


def add_win_probs(schedules: pd.DataFrame) -> pd.DataFrame:
    """Return `schedules` with `home_win_prob` / `away_win_prob` appended.

    Raises rather than falling back when a moneyline is missing or zero. Per the
    availability audit this should never happen, and a silently wrong pick is
    far worse than a loud failure the morning picks are due.
    """
    require_schedule_columns(schedules, _REQUIRED_COLUMNS, needed_for="pick'em win probabilities")
    out = schedules.copy()

    missing = out["home_moneyline"].isna() | out["away_moneyline"].isna()
    if bool(missing.any()):
        raise ValueError(
            "missing moneyline for game(s): "
            f"{sorted(out.loc[missing, 'game_id'].astype(str))}. "
            "Cannot derive a win probability without a price."
        )

    home_odds = out["home_moneyline"].to_numpy(dtype="float64")
    away_odds = out["away_moneyline"].to_numpy(dtype="float64")

    zero = (home_odds == 0) | (away_odds == 0)
    if bool(zero.any()):
        raise ValueError(
            "American odds of 0 are not a valid price; game(s): "
            f"{sorted(out.loc[zero, 'game_id'].astype(str))}"
        )

    home_raw = _implied_array(home_odds)
    away_raw = _implied_array(away_odds)
    total = home_raw + away_raw

    out["home_win_prob"] = home_raw / total
    out["away_win_prob"] = away_raw / total
    return out
