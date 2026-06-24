"""Actual-usage frame for the edge-study universe floor (spec §6.5)."""

from __future__ import annotations

import pandas as pd


def build_usage(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """(gsis_id, season, week, touches_targets) where touches_targets = carries
    + targets (actual usage — never a projection, to avoid endogenous selection)."""
    out = weekly_stats[["gsis_id", "season", "week"]].copy()
    out["touches_targets"] = (
        weekly_stats["carries"].fillna(0) + weekly_stats["targets"].fillna(0)
    ).astype("Float64")
    return out
