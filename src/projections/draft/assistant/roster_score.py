"""Value a completed roster by its optimal starting lineup (spec §3.4).

Fill order is load-bearing: single-position slots, then FLEX, then SUPER_FLEX
(ascending eligibility breadth). The eligibility sets are laminar, so this
restrictive-first greedy is optimal -- no assignment solver needed. Filling a
wider slot first can strand a player and undercount.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from projections.draft.roster_eligibility import FLEX_SLOTS, POSITION_SLOTS
from projections.schemas import Position, RosterSlot


def optimal_lineup_points(
    roster_rows: pd.DataFrame, roster_slots: Mapping[RosterSlot, int]
) -> float:
    """Sum the `season_mean_fpts` of the optimal legal starting lineup.

    `roster_rows` needs columns `gsis_id`, `position`, `season_mean_fpts`
    (no NaN in `season_mean_fpts`; the upstream pandera schema enforces this).
    Bench/IR slots contribute nothing; a starting slot no player can fill scores 0.

    **This is a second implementation of `roster_eligibility.choose_starters`**, kept
    deliberately: this one is DataFrame-shaped and sits in the auction bidding inner loop,
    where converting per call would plausibly cost more than the duplication does. That cost
    is ASSERTED, not measured -- collapsing the two is its own change with its own benchmark.
    `assistant.season_value` holds a third, vectorised over availability draws. If you change
    the fill order or the tie-break here, change it in all three.
    """
    # Per-position points, best-first, deterministic gsis_id tie-break.
    ordered = roster_rows.sort_values(["season_mean_fpts", "gsis_id"], ascending=[False, True])
    by_pos: dict[Position, list[float]] = {}
    for pos in Position:
        by_pos[pos] = [
            float(v) for v in ordered.loc[ordered["position"] == pos.value, "season_mean_fpts"]
        ]
    cursor: dict[Position, int] = {pos: 0 for pos in Position}

    total = 0.0
    # 1) Single-position starting slots.
    for slot in POSITION_SLOTS:
        pos = Position(slot.value)
        for _ in range(roster_slots.get(slot, 0)):
            if cursor[pos] < len(by_pos[pos]):
                total += by_pos[pos][cursor[pos]]
                cursor[pos] += 1
    # 2) Flex tiers, narrowest first; each takes the best remaining eligible player.
    for slot, eligible in FLEX_SLOTS:
        for _ in range(roster_slots.get(slot, 0)):
            best_pos: Position | None = None
            best_val = float("-inf")
            # stable pos order when top-remaining fpts tie
            for pos in sorted(eligible, key=lambda p: p.value):
                if cursor[pos] < len(by_pos[pos]) and by_pos[pos][cursor[pos]] > best_val:
                    best_pos, best_val = pos, by_pos[pos][cursor[pos]]
            if best_pos is not None:
                total += best_val
                cursor[best_pos] += 1
    return total
