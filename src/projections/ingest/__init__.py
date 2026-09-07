"""Ingest layer — the only module that talks to nflreadpy.

`sources.INGEST_SOURCES` is the single registry of what we ingest and in what order. Everything
that refreshes data drives it: `refresh()` here (fail-fast, importable) and
`scripts/refresh_data.py` (isolated per source, classified, summarised).
"""

from __future__ import annotations

from projections.ingest.depth_charts import refresh_depth_charts
from projections.ingest.draft_picks import refresh_draft_picks
from projections.ingest.id_map import build_id_map
from projections.ingest.ngs import refresh_ngs
from projections.ingest.pbp import refresh_pbp
from projections.ingest.schedules import refresh_schedules
from projections.ingest.snap_counts import refresh_snap_counts
from projections.ingest.sources import (
    INGEST_SOURCES,
    IngestSource,
    games_played,
    refresh,
    season_start_date,
    selected_sources,
)
from projections.ingest.weekly_stats import refresh_weekly_stats

__all__ = [
    "INGEST_SOURCES",
    "IngestSource",
    "build_id_map",
    "games_played",
    "refresh",
    "refresh_depth_charts",
    "refresh_draft_picks",
    "refresh_ngs",
    "refresh_pbp",
    "refresh_schedules",
    "refresh_snap_counts",
    "refresh_weekly_stats",
    "season_start_date",
    "selected_sources",
]
