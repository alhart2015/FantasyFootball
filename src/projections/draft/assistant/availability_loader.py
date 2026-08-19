"""Load per-player availability from the store (the shared CLI construction point).

Reads historical weekly_stats + the target-season schedules + id_map under
`<data_root>/raw`, then builds a `PlayerAvailability` for `pool`. A missing
weekly_stats history is a hard error (fail loud - spec §6); a missing
target-season schedule degrades to no byes (build_availability warns).

Which seasons count as history is DERIVED, not configured. See `_usable_history`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability, build_availability
from projections.season_calendar import last_regular_week
from projections.store import read_partition

# Earliest ingested weekly_stats season. A floor, not a ceiling: there is deliberately
# no upper bound to hand-maintain, because that is exactly how the 2025 bug happened -
# the partition was ingested, the constant still said 2024, and `read_partition`'s
# skip-if-missing behaviour made the too-narrow range indistinguishable from a correct
# one at runtime. The upper bound is now the target season itself.
_HISTORY_FLOOR = 2018


class _History(NamedTuple):
    """What `_usable_history` found, split by why each season is or is not usable."""

    frames: list[pd.DataFrame]
    incomplete: list[int]
    gaps: list[int]
    span: range


def _usable_history(raw: Path, season: int, candidates: range | None = None) -> _History:
    """Read every COMPLETED weekly_stats season strictly before `season`.

    Two rules, both enforced here rather than by a constant a human has to remember
    to update:

    **Strictly before `season`** - reading the target season's own weekly_stats is
    lookahead. `draft.backtest.inputs` builds availability with `season=` the season
    being simulated, so without this the drafting agent's ex-ante injury prior depends
    on the outcomes it is about to be graded on: a player who missed half of 2025 is
    pre-marked risky in the 2025 draft, flattering any strategy that gates on
    availability.

    **Completed only** - `build_availability` divides games played by the FULL
    scheduled span (`regular_season_games(season) - first_week + 1`), not by the weeks
    present on disk. Any in-season `refresh` writes a partial partition, and reading
    one scores every player in it at `weeks_so_far / 17`. Ten weeks into a season that
    is 0.59, and since `p_raw` is an unweighted mean across seasons it would drag a
    durable back from ~0.96 toward the `lo` clamp. Completeness is checked against
    `last_regular_week`, so a partial season is skipped whether or not it is the
    target - the target-season rule alone would not catch a stale partial partition
    from an earlier year.

    `gaps` are seasons with no partition at all that sit BETWEEN seasons that do have
    one. A store that simply starts late is normal and says nothing; a hole in the
    middle means a partition went missing — `write_partition` unlinks before writing,
    so an interrupted refresh leaves nothing behind. Without this, a vanished season
    would narrow the history silently, which is the same failure this module exists to
    prevent, only from a different cause.
    """
    frames: list[pd.DataFrame] = []
    incomplete: list[int] = []
    absent: list[int] = []
    seen: list[int] = []

    span = range(_HISTORY_FLOOR, season) if candidates is None else candidates
    for yr in span:
        if yr >= season:
            continue
        try:
            df = read_partition(raw, "weekly_stats", season=yr)
        except FileNotFoundError:
            absent.append(yr)
            continue
        seen.append(yr)
        if df.empty or int(df["week"].max()) < last_regular_week(yr):
            incomplete.append(yr)
            continue
        frames.append(df)

    gaps = [yr for yr in absent if seen and min(seen) < yr < max(seen)]
    return _History(frames, incomplete, gaps, span)


def load_store_availability(
    pool: pd.DataFrame,
    *,
    season: int,
    data_root: Path,
    history_seasons: range | None = None,
) -> PlayerAvailability:
    """Build `PlayerAvailability` for `pool` from store partitions under `data_root`.

    History is derived: every completed weekly_stats season strictly before `season`.
    `history_seasons` narrows the candidate span (tests stub a single season); it can
    never widen past `season`, and never admits an incomplete partition.

    A history that is narrower than expected degrades the model rather than failing,
    and warns. That is softer than the hard error for NO history, deliberately: with
    some good seasons present the model is usable, just weaker. Note the warning goes
    to stderr, so the Streamlit callers do not surface it in their UI.
    """
    raw = data_root / "raw"
    found = _usable_history(raw, season, history_seasons)
    frames = found.frames

    if found.incomplete:
        warnings.warn(
            f"weekly_stats season(s) {found.incomplete} are on disk but incomplete (they do "
            f"not run through the end of the regular season) and were EXCLUDED from the "
            f"availability model. Reading a partial season would score every player in it "
            f"at weeks-so-far / full-season. Re-ingest once the season finishes.",
            stacklevel=2,
        )

    if found.gaps:
        warnings.warn(
            f"weekly_stats season(s) {found.gaps} have no partition at all, but sit between "
            f"seasons that do. A missing season narrows the availability history silently "
            f"(an interrupted refresh leaves nothing behind, since write_partition unlinks "
            f"before writing). Re-ingest them.",
            stacklevel=2,
        )

    if not frames:
        if season <= _HISTORY_FLOOR:
            raise FileNotFoundError(
                f"no weekly_stats history available before season {season}: the ingested "
                f"span starts at {_HISTORY_FLOOR}, so there is nothing earlier to learn "
                "availability from. This is a season-range problem, not a data-root problem."
            )
        examined = found.span
        raise FileNotFoundError(
            f"no complete weekly_stats partitions under {raw} for seasons "
            f"{examined.start}-{min(examined.stop, season) - 1}; check --data-root"
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
