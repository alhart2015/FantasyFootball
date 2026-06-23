"""DraftKings yardage-bonus scoring (skill positions).

DraftKings awards +3 for a 300+ yard passing game, +3 for 100+ rushing, +3 for
100+ receiving. These stack. This is *deterministic actuals* logic: it takes
realized yards, so no probability model is involved. It is intentionally NOT a
`Ruleset` field — the base projection comparison excludes bonuses (a point
projection cannot express E[bonus]); this helper is used only to score actuals
for the edge study's bonus sensitivity check.
"""

from __future__ import annotations

_BONUS = 3.0
_PASS_THRESHOLD = 300.0
_RUSH_THRESHOLD = 100.0
_REC_THRESHOLD = 100.0


def dk_actuals_bonus(
    *, passing_yards: float, rushing_yards: float, receiving_yards: float
) -> float:
    """Total DK yardage bonus for a realized stat line (0/3/6/9)."""
    bonus = 0.0
    if passing_yards >= _PASS_THRESHOLD:
        bonus += _BONUS
    if rushing_yards >= _RUSH_THRESHOLD:
        bonus += _BONUS
    if receiving_yards >= _REC_THRESHOLD:
        bonus += _BONUS
    return bonus
