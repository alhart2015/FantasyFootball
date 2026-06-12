"""Pull + parse ESPN weekly projected stat lines (weeks 1-17) -> half-PPR weekly projections.

Mirrors scripts/pull_external_projections.py's ESPN parse but reads the single-week projection
entry (statSourceId=1, statSplitTypeId=1, scoringPeriodId=wk) and scores the fractional stat
line via scoring.expected_points. The espn_id->gsis_id crosswalk + store write live in
refresh_espn_weekly_projections (a later task).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from projections.ingest.external_projections import ESPN_POSITIONS, ESPN_STAT_IDS
from projections.schemas import Ruleset
from projections.scoring.score import expected_points


def _weekly_proj_stats(player: dict[str, Any], week: int) -> dict[str, float] | None:
    """Return the raw ESPN stat dict for the single weekly projection entry, or None if absent.

    Matches statSourceId=1 (projected) + statSplitTypeId=1 (weekly) + scoringPeriodId=week.
    """
    for s in player.get("stats", []):
        if (
            s.get("scoringPeriodId") == week
            and s.get("statSourceId") == 1
            and s.get("statSplitTypeId") == 1
        ):
            return s.get("stats") or {}
    return None


def _statline_dict(raw: dict[str, float]) -> dict[str, float]:
    """Map ESPN string stat-ids to canonical StatLine field names.

    ESPN_STAT_IDS keys are strings (e.g. "24") matching the JSON stats dict keys directly.
    Fields absent from raw default to 0.0; fields not in ESPN_STAT_IDS are ignored.
    """
    line = {field: 0.0 for field in ESPN_STAT_IDS.values()}
    for sid, field in ESPN_STAT_IDS.items():
        if sid in raw:
            line[field] = float(raw[sid])
    return line


def parse_espn_weekly(
    payload: dict[str, Any], *, season: int, week: int, ruleset: Ruleset
) -> pd.DataFrame:
    """Parse a single-week ESPN kona_player_info payload -> one row per QB/RB/WR/TE.

    Columns: espn_id (str), season (int), week (int), position (str), projected_points (float|None).
    projected_points is None when no weekly projection entry exists for the player (e.g. bye week
    or missing data). A zero-stat projection entry produces 0.0, not None.
    """
    rows: list[dict[str, Any]] = []
    for pl in payload.get("players", []):
        p = pl.get("player", {})
        position = ESPN_POSITIONS.get(p.get("defaultPositionId"))
        if position is None:
            continue
        raw = _weekly_proj_stats(p, week)
        proj = expected_points(_statline_dict(raw), ruleset) if raw is not None else None
        rows.append(
            {
                "espn_id": str(p.get("id")),
                "season": season,
                "week": week,
                "position": position,
                "projected_points": proj,
            }
        )
    return pd.DataFrame(rows)
