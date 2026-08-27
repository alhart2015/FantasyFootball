"""Set a weekly lineup by PROJECTION (the manager's decision), score it by ACTUAL points.

The choice itself is `roster_eligibility.choose_starters`, shared with the waiver recommender;
this module is the "set by projection, score by actual" wrapper around it. Players with a null
projection are unstartable (bye/inactive); a started player with a null actual contributes 0;
unfilled slots contribute 0.

`assistant.roster_score.optimal_lineup_points` still carries its own copy of the same greedy,
over a DataFrame of season points. Not merged here on purpose: it sits in the auction bidding
inner loop where the per-call frame conversion would cost more than the duplication does, and
`assistant.season_value` holds a third, vectorised over availability draws. Three shapes of one
algorithm is worth knowing about; collapsing them is its own change with its own benchmark.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from projections.draft.roster_eligibility import choose_starters
from projections.schemas import RosterSlot


def weekly_lineup_points(
    roster: Sequence[Mapping[str, Any]],
    roster_slots: Mapping[RosterSlot, int],
    *,
    score_by: Literal["actual", "projected"] = "actual",
) -> float:
    """Score the lineup chosen by highest `projected` values, summing the `score_by` field.

    The lineup is always *set* by projection (the no-hindsight manager decision). The
    points *summed* are taken from `score_by`:
      - ``"actual"`` (default): real outcome points — the realistic fantasy objective.
      - ``"projected"``: the projected points of the started lineup — "who drafted better"
        under the shared projections, with outcome luck and projection error removed.

    Fill order: single-position slots (restrictive first), then FLEX, then SUPER_FLEX.
    Players with a null `projected` are unstartable. A started player whose `score_by`
    value is null contributes 0. Slots with no eligible player also contribute 0.
    """
    chosen = choose_starters(
        list(roster),
        roster_slots,
        value=lambda p: None if p.get("projected") is None else float(p["projected"]),
        position=lambda p: str(p["position"]),
    )
    total = 0.0
    for index in chosen:
        # NOTE: score_by must be a key present in each roster dict ("projected"/"actual").
        # This coupling is by convention, not type-checked -- a missing key reads as 0.0
        # (None), silently zeroing every player. If you rename the roster dict keys, update
        # the score_by Literal in this signature to match.
        points = roster[index].get(score_by)
        total += 0.0 if points is None else float(points)
    return total
