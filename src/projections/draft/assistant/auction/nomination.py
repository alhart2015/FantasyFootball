"""Hero nomination strategies for the auction (Slice 2 feasibility probe — NOT ADOPTED).

A `HeroNominator` picks the hero's nominee from the room-rosterable `candidates`. The poison
heuristics aim to drain opponents' budgets: `drain_max` nominates the priciest player (forcing the
room to spend on a stud the capped hero would lose anyway); `drain_off_position` nominates the
priciest player at a position the hero has already filled, so the drain lands on opponents who still
need that slot. See docs/superpowers/specs/2026-07-15-auction-nomination-poisoning-design.md.

PROBE VERDICT (Run O, reports/auction_tournament_validation_2026.md) — NO-GO, retained as a probe.
A 40-seed CRN-paired sweep found `drain_max` harmful and `drain_off_position` a genuine but
MARKET-SPECIFIC edge: real in the symmetric model market (+0.015 reg_win_pct, 95% CI-separated from
0) yet indistinguishable from zero in the realistic ESPN market (+0.007, CI spans 0). It fails the
pre-registered both-markets robustness bar, so the shipped `balanced` p0.0 hero does NOT wire in a
`hero_nominator` (the engine hook defaults to None = today's value-weighted-random nomination). This
module + hook are kept as a tested, opt-in extension point for any future ESPN-targeted nomination
work, not as an adopted strategy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from projections.schemas import Position


@dataclass(frozen=True)
class NominationContext:
    """What a HeroNominator reads to choose the hero's nominee.

    `value_by_id` is the market value the room bids on (`bot_dollars`), so "priciest" means "biggest
    drain" in both the model and ESPN markets (they differ under ESPN anchoring). `hero_positions`
    is the hero's drafted position counts; `position_minimums` the per-position starter requirement
    (`bot_position_bounds`); `position_by_id` each candidate's position.
    """

    hero_positions: Counter[Position]
    value_by_id: Mapping[str, float]
    position_by_id: Mapping[str, Position]
    position_minimums: Mapping[Position, int]


HeroNominator = Callable[[list[str], NominationContext], str]


def drain_max(candidates: list[str], ctx: NominationContext) -> str:
    """Nominate the priciest room-rosterable player (max value the room bids on)."""
    return max(candidates, key=lambda g: ctx.value_by_id[str(g)])


def drain_off_position(candidates: list[str], ctx: NominationContext) -> str:
    """Nominate the priciest candidate at a position the hero has already filled to its starter
    requirement; fall back to `drain_max` (priciest overall) when none qualifies."""
    off = [
        g
        for g in candidates
        if ctx.hero_positions[ctx.position_by_id[str(g)]]
        >= ctx.position_minimums.get(ctx.position_by_id[str(g)], 0)
    ]
    return max(off or candidates, key=lambda g: ctx.value_by_id[str(g)])
