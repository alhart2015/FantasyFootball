"""Top-level ingest orchestrator. Fans out to every per-source refresh
function. Plan 3a's first real-data pull uses this entrypoint.

Note on signatures: the per-source refresh functions take ``data_root``
positionally, with ``seasons`` (and where applicable ``stat_type``) as
keyword-only arguments. ``build_id_map`` does not take a ``seasons``
argument at all -- the id_map is a single roster-wide table, not per-season.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from projections.ingest.depth_charts import refresh_depth_charts
from projections.ingest.id_map import build_id_map
from projections.ingest.ngs import STAT_TYPES as NGS_STAT_TYPES
from projections.ingest.ngs import refresh_ngs
from projections.ingest.schedules import refresh_schedules
from projections.ingest.snap_counts import refresh_snap_counts
from projections.ingest.weekly_stats import refresh_weekly_stats


def refresh(seasons: Iterable[int], *, data_root: Path) -> None:
    """Refresh every ingest source for ``seasons`` under ``data_root``.

    Order matters: ``build_id_map`` must run before ``refresh_snap_counts``
    because snap_counts ingest joins on the gsis_id <-> pfr_id translation
    table written by build_id_map and raises ``FileNotFoundError`` if it is
    missing. Within that constraint the remaining sources are independent
    and could in principle run in any order; we run weekly_stats and
    schedules first because they are the cheapest pulls and surface
    network/auth failures fastest.

    Fail-fast: aborts on the first source failure. Re-running after a
    partial failure is safe but will repeat the work of already-completed
    sources (per-source writes are idempotent in place).
    """
    # Materialize: per-source calls each iterate ``seasons``; a generator
    # would be exhausted after the first call.
    season_list = list(seasons)
    refresh_weekly_stats(data_root, seasons=season_list)
    refresh_schedules(data_root, seasons=season_list)
    refresh_depth_charts(data_root, seasons=season_list)
    for stat_type in NGS_STAT_TYPES:
        refresh_ngs(data_root, stat_type=stat_type, seasons=season_list)
    build_id_map(data_root)
    refresh_snap_counts(data_root, seasons=season_list)
