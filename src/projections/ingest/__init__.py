"""Ingest layer — the only module that talks to nflreadpy."""

from __future__ import annotations

from projections.ingest.depth_charts import refresh_depth_charts
from projections.ingest.draft_picks import refresh_draft_picks
from projections.ingest.id_map import build_id_map
from projections.ingest.ngs import refresh_ngs
from projections.ingest.pbp import refresh_pbp
from projections.ingest.refresh import refresh
from projections.ingest.schedules import refresh_schedules
from projections.ingest.snap_counts import refresh_snap_counts
from projections.ingest.weekly_stats import refresh_weekly_stats

__all__ = [
    "build_id_map",
    "refresh",
    "refresh_depth_charts",
    "refresh_draft_picks",
    "refresh_ngs",
    "refresh_pbp",
    "refresh_schedules",
    "refresh_snap_counts",
    "refresh_weekly_stats",
]
