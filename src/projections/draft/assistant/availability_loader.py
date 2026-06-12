"""Load per-player availability from the store (the shared CLI construction point).

Reads historical weekly_stats + the target-season schedules + id_map under
`<data_root>/raw`, then builds a `PlayerAvailability` for `pool`. A missing
weekly_stats history is a hard error (fail loud — spec §6); a missing
target-season schedule degrades to no byes (build_availability warns).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability, build_availability
from projections.store import read_partition

# weekly_stats coverage for the availability model. Bump the upper bound each season
# once the new year's weekly_stats partition is ingested.
_HISTORY_SEASONS = range(2018, 2025)


def load_store_availability(
    pool: pd.DataFrame,
    *,
    season: int,
    data_root: Path,
    history_seasons: range = _HISTORY_SEASONS,
) -> PlayerAvailability:
    """Build `PlayerAvailability` for `pool` from store partitions under `data_root`.

    `history_seasons` is the weekly_stats span for the injury model (default the full
    ingested range); overridable so tests can stub a single season.
    """
    raw = data_root / "raw"
    frames: list[pd.DataFrame] = []
    for yr in history_seasons:
        try:
            frames.append(read_partition(raw, "weekly_stats", season=yr))
        except FileNotFoundError:
            continue
    if not frames:
        raise FileNotFoundError(
            f"no weekly_stats partitions under {raw} for seasons "
            f"{history_seasons.start}-{history_seasons.stop - 1}; check --data-root"
        )
    weekly_stats = pd.concat(frames, ignore_index=True)
    try:
        schedules = read_partition(raw, "schedules", season=season)
    except FileNotFoundError:
        # A missing target-season schedule degrades to no byes (build_availability
        # warns and the injury model still applies), not a hard fail.
        schedules = pd.DataFrame(columns=["season", "week", "home_team", "away_team"])
    # build_availability only reads gsis_id + team, so full IdMapSchema validation is skipped.
    # id_map is a single file (not a season-partitioned store table), so read it directly.
    # Guard on .exists() rather than catching FileNotFoundError, which would also swallow a
    # parquet-internal missing-file error and misattribute it to the id_map path.
    id_map_path = raw / "id_map.parquet"
    if not id_map_path.exists():
        raise FileNotFoundError(f"id_map.parquet not found at {id_map_path}; check --data-root")
    id_map = pd.read_parquet(id_map_path)
    return build_availability(weekly_stats, schedules, id_map, pool, season=season)
