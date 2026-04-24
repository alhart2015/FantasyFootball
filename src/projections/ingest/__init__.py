"""Ingest layer — the only module that talks to nfl_data_py."""

from __future__ import annotations

from projections.ingest.depth_charts import refresh_depth_charts
from projections.ingest.id_map import build_id_map
from projections.ingest.schedules import refresh_schedules
from projections.ingest.snap_counts import refresh_snap_counts
from projections.ingest.weekly_stats import refresh_weekly_stats

__all__ = [
    "build_id_map",
    "refresh_depth_charts",
    "refresh_schedules",
    "refresh_snap_counts",
    "refresh_weekly_stats",
]
