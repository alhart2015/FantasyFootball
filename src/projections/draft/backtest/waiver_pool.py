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
    pool = VorpTableSchema.validate(pool)
    drafted = {str(g) for roster in rosters.values() for g in roster}
    pos = pool["position"].astype(str)
    vorp = pool["vorp"]
    undrafted = ~pool["gsis_id"].astype(str).isin(drafted)

    rows: list[dict[str, object]] = []
    for p in _SKILL_POSITIONS:
        at_pos = pos == p.value
        above = at_pos & (vorp > 0)
        total_above = int(above.sum())
        drafted_above = int((above & ~undrafted).sum())

        undf = pool[at_pos & undrafted]
        top = np.sort(undf["vorp"].to_numpy())[::-1]  # descending
        top1 = float(top[0]) if len(top) > 0 else float("nan")
        top2 = float(top[1]) if len(top) > 1 else float("nan")
        top3 = float(top[2]) if len(top) > 2 else float("nan")

        if len(undf):
            best_proj = float(undf.loc[undf["vorp"].idxmax(), "season_mean_fpts"])
        else:
            best_proj = float("nan")

        rows.append(
            {
                "position": p.value,
                "top1_vorp": top1,
                "top2_vorp": top2,
                "top3_vorp": top3,
                "best_avail_proj_pts": best_proj,
                "n_above_replacement": int((undf["vorp"] > 0).sum()),
                "drain_rate": (drafted_above / total_above) if total_above else float("nan"),
            }
        )

    return WaiverPoolSchema.validate(pd.DataFrame(rows))
