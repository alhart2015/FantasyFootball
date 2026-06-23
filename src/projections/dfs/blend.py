"""Blend home-grown + Sleeper weekly projections in stat-line space.

Matches consensus.blend: average per-stat means (here a weighted average),
assemble one stat line, score once. weight_ours=1.0 -> home-grown-only,
0.0 -> Sleeper-only.
"""

from __future__ import annotations

import pandas as pd

from projections.schemas import Ruleset, Stat
from projections.scoring import expected_points

# canonical stat fields shared by both sources (Sleeper uses these names; our
# emitter uses Stat.value, which are the same strings).
_BLEND_FIELDS = [
    Stat.PASSING_YARDS.value,
    Stat.PASSING_TDS.value,
    Stat.INTERCEPTIONS.value,
    Stat.RUSHING_YARDS.value,
    Stat.RUSHING_TDS.value,
    Stat.RECEPTIONS.value,
    Stat.RECEIVING_YARDS.value,
    Stat.RECEIVING_TDS.value,
    Stat.FUMBLES_LOST.value,
]
_KEY = ["gsis_id", "season", "week"]


def blend_statlines(
    ours: pd.DataFrame, sleeper: pd.DataFrame, *, weight_ours: float, ruleset: Ruleset
) -> pd.DataFrame:
    """Weighted stat-line blend -> one `blended_pts` per (gsis_id, season, week)."""
    # Reindex both sides to the full field set (absent fields -> NaN) so every
    # blend field is present in BOTH frames and therefore gets a _ours/_slp
    # suffix on merge. Without this, a field present in only one source stays
    # unsuffixed and `row.get(f"{field}_slp")` silently reads None -> that
    # source's contribution for the stat is dropped.
    o = ours.reindex(columns=_KEY + _BLEND_FIELDS)
    s = sleeper.reindex(columns=_KEY + _BLEND_FIELDS)
    merged = o.merge(s, on=_KEY, how="inner", suffixes=("_ours", "_slp"))

    pts: list[float] = []
    for _, row in merged.iterrows():
        line: dict[str, float] = {}
        for field in _BLEND_FIELDS:
            ov, sv = row.get(f"{field}_ours"), row.get(f"{field}_slp")
            vals = [(weight_ours, ov), (1.0 - weight_ours, sv)]
            num = sum(w * float(v) for w, v in vals if pd.notna(v))
            wsum = sum(w for w, v in vals if pd.notna(v))
            if wsum > 0:
                line[field] = num / wsum
        pts.append(expected_points(line, ruleset))

    out = merged[_KEY].copy()
    out["blended_pts"] = pts
    return out


def sleeper_weekly_points(sleeper: pd.DataFrame, *, ruleset: Ruleset) -> pd.DataFrame:
    """Score the Sleeper weekly stat-line frame to DK-base points per cell.

    This is the `sleeper_pts` frame Task 10's build_universe consumes.
    """
    present = [c for c in _BLEND_FIELDS if c in sleeper.columns]
    pts = [
        expected_points({c: float(row[c]) for c in present if pd.notna(row[c])}, ruleset)
        for _, row in sleeper.iterrows()
    ]
    out = sleeper[_KEY].copy()
    out["sleeper_pts"] = pts
    return out
