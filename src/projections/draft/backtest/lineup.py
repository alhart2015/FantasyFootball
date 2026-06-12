"""Set a weekly lineup by PROJECTION (the manager's decision), score it by ACTUAL points.

Mirrors roster_score.optimal_lineup_points' restrictive-slot-first greedy, but assigns by
`projected` and sums `actual`. Players with a null projection are unstartable (bye/inactive);
a started player with a null actual contributes 0; unfilled slots contribute 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from projections.draft.roster_eligibility import FLEX_ELIGIBLE, POSITION_SLOTS, SUPER_FLEX_ELIGIBLE
from projections.schemas import Position, RosterSlot

_FLEX_SLOTS: tuple[tuple[RosterSlot, frozenset[Position]], ...] = (
    (RosterSlot.FLEX, FLEX_ELIGIBLE),
    (RosterSlot.SUPER_FLEX, SUPER_FLEX_ELIGIBLE),
)


def weekly_lineup_points(
    roster: Sequence[Mapping[str, Any]], roster_slots: Mapping[RosterSlot, int]
) -> float:
    """Sum the `actual` points of the lineup chosen by highest `projected` values.

    Fill order: single-position slots (restrictive first), then FLEX, then SUPER_FLEX.
    Players with a null `projected` are unstartable. A started player whose `actual`
    is null contributes 0. Slots with no eligible player also contribute 0.
    """
    startable = [p for p in roster if p.get("projected") is not None]

    by_pos: dict[Position, list[Mapping[str, Any]]] = {pos: [] for pos in Position}
    for p in startable:
        by_pos[Position(p["position"])].append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: float(p["projected"]), reverse=True)

    cursor: dict[Position, int] = {pos: 0 for pos in Position}

    def _actual(p: Mapping[str, Any]) -> float:
        a = p.get("actual")
        return 0.0 if a is None else float(a)

    total = 0.0

    # 1) Single-position starting slots (most restrictive first).
    for slot in POSITION_SLOTS:
        pos = Position(slot.value)
        for _ in range(roster_slots.get(slot, 0)):
            if cursor[pos] < len(by_pos[pos]):
                total += _actual(by_pos[pos][cursor[pos]])
                cursor[pos] += 1

    # 2) Flex tiers, narrowest eligibility first; each takes the highest-projected
    #    remaining eligible player and scores their actual points.
    for slot, eligible in _FLEX_SLOTS:
        for _ in range(roster_slots.get(slot, 0)):
            best_pos: Position | None = None
            best_proj = float("-inf")
            for pos in sorted(eligible, key=lambda p: p.value):
                if cursor[pos] < len(by_pos[pos]):
                    proj = float(by_pos[pos][cursor[pos]]["projected"])
                    if proj > best_proj:
                        best_pos, best_proj = pos, proj
            if best_pos is not None:
                total += _actual(by_pos[best_pos][cursor[best_pos]])
                cursor[best_pos] += 1

    return total
