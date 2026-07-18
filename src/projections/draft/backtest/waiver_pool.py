"""Characterize the undrafted ("waiver wire") pool by position after a draft.

Pure, per-draft core for the waiver-pool assessment (spec 2026-07-17): given the
drafted rosters and the VORP pool, report per skill position how good the best
still-available players are (top-3 VORP + the leader's projected points), how much
startable-quality depth remains (count above replacement), and how hard the field
drained the position (drain_rate). The driver (scripts/waiver_pool_assessment.py)
runs this over many simulated drafts and aggregates.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from projections.draft.league_config import LeagueConfig
from projections.schemas import Position, VorpTableSchema, WaiverPoolSchema

_SKILL_POSITIONS = (Position.QB, Position.RB, Position.WR, Position.TE)


def undrafted_pool_by_position(
    rosters: Mapping[int, list[str]],
    pool: pd.DataFrame,
    config: LeagueConfig,
) -> pd.DataFrame:
    """Per-position undrafted-pool metrics for one draft (WaiverPoolSchema-valid).

    `rosters` is {seat: [gsis_id, ...]} from `draft_mixed_field`; `pool` is a
    VorpTableSchema-valid consensus VORP table. `config` is accepted for interface
    stability; v1 always reports the four skill positions (QB/RB/WR/TE). See the
    module and WaiverPoolSchema docstrings for the metric definitions and edge cases.
    """
    # Deliberately validated per call (not only at the driver): this is a reusable src/
    # boundary, and rejecting a malformed pool -- e.g. a NaN `vorp`, which would make
    # top1_vorp NaN via np.sort while best_avail_proj_pts stays real -- is worth more than
    # the small redundant-revalidation cost in the seed loop. Correctness over speed; keep.
    pool = VorpTableSchema.validate(pool)
    drafted = {str(g) for roster in rosters.values() for g in roster}
    pos = pool["position"].astype(str)
    vorp = pool["vorp"]
    undrafted = ~pool["gsis_id"].isin(drafted)

    rows: list[dict[str, object]] = []
    for p in _SKILL_POSITIONS:
        at_pos = pos == p.value
        above = at_pos & (vorp > 0)
        total_above = int(above.sum())
        drafted_above = int((above & ~undrafted).sum())

        undf = pool[at_pos & undrafted]
        uv = undf["vorp"].to_numpy()
        top = np.sort(uv)[::-1]  # descending
        top1 = float(top[0]) if len(top) > 0 else float("nan")
        top2 = float(top[1]) if len(top) > 1 else float("nan")
        top3 = float(top[2]) if len(top) > 2 else float("nan")

        # Positional argmax (not idxmax/.loc) so a non-unique DataFrame index can't make
        # this return a multi-row Series; the validated pool has no NaN vorp to confuse it.
        if len(undf):
            best_proj = float(undf["season_mean_fpts"].to_numpy()[int(uv.argmax())])
        else:
            best_proj = float("nan")

        rows.append(
            {
                "position": p.value,
                "top1_vorp": top1,
                "top2_vorp": top2,
                "top3_vorp": top3,
                "best_avail_proj_pts": best_proj,
                "n_above_replacement": total_above - drafted_above,
                "drain_rate": (drafted_above / total_above) if total_above else float("nan"),
            }
        )

    return WaiverPoolSchema.validate(pd.DataFrame(rows))
