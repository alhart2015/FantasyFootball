"""Hero nomination strategies for the auction (feasibility probes — NONE ADOPTED).

A `HeroNominator` picks the hero's nominee from the room-rosterable `candidates`. The poison
heuristics aim to drain opponents' budgets. Two families, from two probes:

*Price-ranked (Slice 2, Run O).* `drain_max` nominates the priciest player (forcing the room to
spend on a stud the capped hero would lose anyway); `drain_off_position` nominates the priciest
player at a position the hero has already filled, so the drain lands on opponents who still need
that slot. See docs/superpowers/specs/2026-07-15-auction-nomination-poisoning-design.md.

*Gap-ranked (Slice 2b).* `drain_value_gap` and `drain_value_gap_off_position` rank by how far the
room's price exceeds OUR value (`value_by_id - hero_value_by_id`) rather than by price alone, so the
drain lands only where the room is wrong in our favour and never on a player we ourselves want. The
gap is identically zero in the model market (the room prices off our own numbers there), so these
are ESPN-market heuristics by construction. See
docs/superpowers/specs/2026-08-12-auction-value-gap-nomination-design.md.

PROBE VERDICT (Run O, reports/auction_tournament_validation_2026.md) — NO-GO, retained as a probe.
An 80-seed CRN-paired sweep found `drain_max` harmful and `drain_off_position` a genuine but
MARKET-SPECIFIC edge: real in the symmetric model market (+0.010 reg_win_pct, 95% CI-separated from
0, robust across 40->80 seeds) yet NOT distinguishable from zero in the realistic ESPN market
(+0.005, CI [-0.001, +0.011] spans 0 once firmed up). It fails the pre-registered both-markets
robustness bar, so the shipped `balanced` p0.0 hero does NOT wire in a `hero_nominator` (the engine
hook defaults to None = today's value-weighted-random nomination). This module + hook are kept as a
tested, opt-in extension point for any future ESPN-targeted nomination work, not as an adopted
strategy. (History: a CRN-desync bug in the hook briefly flipped the interim verdict to a marginal
GO; the fix + the 80-seed firm-up settled it here. See Run O.)
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
    drain" in both the model and ESPN markets (they differ under ESPN anchoring). `hero_value_by_id`
    is OUR value (`auction_dollars`) for those players, so `value_by_id - hero_value_by_id` is the
    room's overpay relative to our board — zero everywhere in the model market, where the room
    prices off our numbers. `hero_positions` is the hero's drafted position counts;
    `position_minimums` the per-position starter requirement (`bot_position_bounds`);
    `position_by_id` each candidate's position.
    """

    hero_positions: Counter[Position]
    value_by_id: Mapping[str, float]
    position_by_id: Mapping[str, Position]
    position_minimums: Mapping[Position, int]
    hero_value_by_id: Mapping[str, float]


HeroNominator = Callable[[list[str], NominationContext], str]


def drain_max(candidates: list[str], ctx: NominationContext) -> str:
    """Nominate the priciest room-rosterable player (max value the room bids on)."""
    return max(candidates, key=lambda g: ctx.value_by_id[str(g)])


def _off_position(candidates: list[str], ctx: NominationContext) -> list[str]:
    """The candidates at a position the hero has already filled to its starter requirement.

    May be empty (early draft, before the hero has filled anything) — every caller falls back to the
    unrestricted candidate list rather than nominating nothing.
    """
    return [
        g
        for g in candidates
        if ctx.hero_positions[ctx.position_by_id[str(g)]]
        >= ctx.position_minimums.get(ctx.position_by_id[str(g)], 0)
    ]


def drain_off_position(candidates: list[str], ctx: NominationContext) -> str:
    """Nominate the priciest candidate at a position the hero has already filled to its starter
    requirement; fall back to `drain_max` (priciest overall) when none qualifies."""
    off = _off_position(candidates, ctx)
    return drain_max(off or candidates, ctx)  # priciest among the off-position set (or all)


def _gap(ctx: NominationContext, gsis_id: str) -> float:
    """How far the room's price exceeds our own value — the room's overpay, in dollars.

    ABSOLUTE dollars, deliberately not a ratio: the hypothesis is about how much money leaves the
    room per nomination, and a ratio would rank a $3 player the room prices at $9 ("200% over")
    above a $40 player it prices at $55, draining $6 instead of $55.
    """
    return ctx.value_by_id[gsis_id] - ctx.hero_value_by_id[gsis_id]


def drain_value_gap(candidates: list[str], ctx: NominationContext) -> str:
    """Nominate the candidate the room most overvalues relative to our board.

    Unlike `drain_max` this never surfaces a player *we* rate highly (our own value is subtracted),
    so the drain lands on money we were never going to compete for. Degenerate in the model market,
    where the room prices off our numbers and every gap is 0.
    """
    return max(candidates, key=lambda g: _gap(ctx, str(g)))


def drain_value_gap_off_position(candidates: list[str], ctx: NominationContext) -> str:
    """`drain_value_gap` restricted to positions the hero has already filled to its starter
    requirement; falls back to the unrestricted `drain_value_gap` when none qualifies.

    Composes the two signals that each showed something on their own: Run O's off-position targeting
    (aim the drain at opponents who still need the slot) and the room-disagreement gap.
    """
    return drain_value_gap(_off_position(candidates, ctx) or candidates, ctx)
