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

# Upper bound of the weekly_stats span the availability model may read. Bump it each
# season once the new year's partition is ingested AND complete.
#
# 2025 added 2026-08-16. It had been ingested (weeks 1-22, 5,557 rows) but was not being
# read, so every availability probability came from a history one full season stale. The
# effect was not neutral - it made the model systematically pessimistic, e.g. De'Von
# Achane 0.844 -> 0.876 and Jahmyr Gibbs 0.941 -> 0.961.
#
# The bound is a ceiling, not a target: `load_store_availability` additionally drops every
# season at or after the one being built for. See `_history_for`.
_HISTORY_SEASONS = range(2018, 2026)


def _history_for(season: int, history_seasons: range) -> list[int]:
    """Seasons strictly before `season`, within `history_seasons`.

    Two distinct problems, one rule.

    **Lookahead.** `draft.backtest.inputs` builds availability with `season=` the season
    being simulated. Reading that season's weekly_stats makes the drafting agent's ex-ante
    injury prior depend on the outcomes it is about to be graded on: a player who missed
    half of 2025 is pre-marked risky in the 2025 draft, flattering any strategy that gates
    on availability.

    **Partial seasons.** `build_availability` divides games played by the FULL scheduled
    span (`regular_season_games(season) - first_week + 1`), not by the weeks present on
    disk. An in-progress partition — which any in-season `refresh` writes — therefore
    scores every player at `weeks_so_far / 17`. Three weeks into 2026 that is 0.18, and
    since `p_raw` is an unweighted mean across seasons it would drag a durable back from
    ~0.96 to ~0.76 and slam much of the pool into the `lo` clamp.

    Excluding the target season and everything after it removes both at once, and makes
    the model safe to run mid-season.
    """
    return [yr for yr in history_seasons if yr < season]


def load_store_availability(
    pool: pd.DataFrame,
    *,
    season: int,
    data_root: Path,
    history_seasons: range = _HISTORY_SEASONS,
) -> PlayerAvailability:
    """Build `PlayerAvailability` for `pool` from store partitions under `data_root`.

    `history_seasons` is the weekly_stats ceiling for the injury model (default the full
    ingested range); overridable so tests can stub a single season. Seasons at or after
    `season` are dropped regardless — see `_history_for`.
    """
    raw = data_root / "raw"
    usable = _history_for(season, history_seasons)
    frames: list[pd.DataFrame] = []
    for yr in usable:
        try:
            frames.append(read_partition(raw, "weekly_stats", season=yr))
        except FileNotFoundError:
            continue
    if not frames:
        span = f"{usable[0]}-{usable[-1]}" if usable else "(none before target season)"
        raise FileNotFoundError(
            f"no weekly_stats partitions under {raw} for seasons {span} "
            f"(history capped at {history_seasons.stop - 1}, target season {season}); "
            "check --data-root"
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
