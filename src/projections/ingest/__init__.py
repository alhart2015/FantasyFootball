"""Ingest layer — the only module that talks to nfl_data_py."""

from __future__ import annotations

from projections.ingest.id_map import build_id_map
from projections.ingest.schedules import refresh_schedules
from projections.ingest.weekly_stats import refresh_weekly_stats

__all__ = [
    "build_id_map",
    "refresh_schedules",
    "refresh_weekly_stats",
]
