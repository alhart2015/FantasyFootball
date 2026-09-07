"""The one list of ingest sources, and the fail-fast orchestrator over it.

`INGEST_SOURCES` is the single source of truth for "everything we ingest", in dependency order.
Everything that refreshes data drives it: `refresh()` here (fail-fast, importable) and
`scripts/refresh_data.py` (isolated per source, classified, summarised).

**Add a source by adding one entry to `INGEST_SOURCES`.** That is the whole point of the registry.
Two hand-maintained lists of "every source" is how this module ended up missing `draft_picks`,
`external_projections`, and `pbp` while a second orchestrator had them.

Two facts about a source decide when it runs, and both live on the entry rather than in the
caller:

- **`needs_games_played`** — the per-game sources (`weekly_stats`, `snap_counts`, `ngs_*`,
  `depth_charts`, `pbp`) have no rows upstream until the season is underway, and nflverse
  publishes each week's release on a lag after that. Asking for them before kickoff yields a 404
  or a bare `ValueError` from `nflreadpy`, neither distinguishable from a real defect. The
  market-facing sources (`id_map`, `schedules`, `draft_picks`, `external_projections`) are
  published year-round, which is what makes a preseason refresh worth running at all.
- **`heavy`** — `pbp` is hundreds of MB per season. Opt in explicitly; never pulled by default.

Order matters and is encoded by position:

- `build_id_map` must precede `refresh_snap_counts`, which joins on the gsis_id <-> pfr_id
  translation table it writes and raises `FileNotFoundError` without it.
- `refresh_schedules` must precede `refresh_depth_charts`, which needs schedules to resolve
  `(season, week)` from the 2025+ snapshot format.

Within those constraints the cheap pulls run first, so a network or auth failure surfaces fast.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from projections.ingest.depth_charts import refresh_depth_charts
from projections.ingest.draft_picks import refresh_draft_picks
from projections.ingest.external_projections import refresh_external_projections
from projections.ingest.id_map import build_id_map
from projections.ingest.ngs import STAT_TYPES as NGS_STAT_TYPES
from projections.ingest.ngs import NgsStatType, refresh_ngs
from projections.ingest.pbp import refresh_pbp
from projections.ingest.schedules import refresh_schedules
from projections.ingest.snap_counts import refresh_snap_counts
from projections.ingest.weekly_stats import refresh_weekly_stats


@dataclass(frozen=True)
class IngestSource:
    """One ingest source, with the two facts that decide when it may run.

    `run` is normalised to `(data_root, seasons) -> list[Path]` so callers can iterate the
    registry uniformly. The underlying signatures are not uniform — `build_id_map` takes no
    seasons, `refresh_ngs` takes a `stat_type`, and `refresh_external_projections` takes one
    season at a time — and the adapters below are where that is absorbed, once.
    """

    name: str
    #: Rows only exist once the season is underway. See the module docstring.
    needs_games_played: bool
    #: Hundreds of MB; pulled only when explicitly requested.
    heavy: bool
    run: Callable[[Path, list[int]], list[Path]]


def _run_id_map(data_root: Path, seasons: list[int]) -> list[Path]:
    """The id_map is a single roster-wide table, not per-season, so `seasons` is unused."""
    del seasons
    return [build_id_map(data_root)]


def _ngs_runner(stat_type: NgsStatType) -> Callable[[Path, list[int]], list[Path]]:
    """Bind `stat_type` eagerly — a closure over the loop variable would run every NGS source
    against whichever stat type the comprehension happened to end on."""

    def _run(data_root: Path, seasons: list[int]) -> list[Path]:
        return refresh_ngs(data_root, stat_type=stat_type, seasons=seasons)

    return _run


def _run_external_projections(data_root: Path, seasons: list[int]) -> list[Path]:
    """ESPN + Sleeper take one season per call and return a single snapshot path each."""
    return [refresh_external_projections(data_root, season=season) for season in seasons]


INGEST_SOURCES: tuple[IngestSource, ...] = (
    # Always available, and dependencies of the per-game sources below.
    IngestSource("id_map", needs_games_played=False, heavy=False, run=_run_id_map),
    IngestSource(
        "schedules",
        needs_games_played=False,
        heavy=False,
        run=lambda root, seasons: refresh_schedules(root, seasons=seasons),
    ),
    IngestSource(
        "draft_picks",
        needs_games_played=False,
        heavy=False,
        run=lambda root, seasons: refresh_draft_picks(root, seasons=seasons),
    ),
    IngestSource(
        "external_projections",
        needs_games_played=False,
        heavy=False,
        run=_run_external_projections,
    ),
    # Per-game: nothing upstream until the season is underway.
    IngestSource(
        "weekly_stats",
        needs_games_played=True,
        heavy=False,
        run=lambda root, seasons: refresh_weekly_stats(root, seasons=seasons),
    ),
    IngestSource(
        "depth_charts",
        needs_games_played=True,
        heavy=False,
        run=lambda root, seasons: refresh_depth_charts(root, seasons=seasons),
    ),
    IngestSource(
        "snap_counts",
        needs_games_played=True,
        heavy=False,
        run=lambda root, seasons: refresh_snap_counts(root, seasons=seasons),
    ),
    *(
        IngestSource(
            f"ngs_{stat_type}", needs_games_played=True, heavy=False, run=_ngs_runner(stat_type)
        )
        for stat_type in NGS_STAT_TYPES
    ),
    IngestSource(
        "pbp",
        needs_games_played=True,
        heavy=True,
        run=lambda root, seasons: refresh_pbp(root, seasons=seasons),
    ),
)


def season_start_date(season: int) -> date:
    """Kickoff Thursday -- the Thursday after Labor Day -- mirroring `nflreadpy`'s own rollover.

    Kept here rather than deferring to `nflreadpy.get_current_season()` because that function
    takes no date argument, so nothing depending on it can be pinned to a fixed point in the
    calendar by a test. The arithmetic is byte-for-byte the same.
    """
    labor_day = next(
        date(season, 9, day) for day in range(1, 8) if date(season, 9, day).weekday() == 0
    )
    return date(season, 9, labor_day.day + 3)


def games_played(season: int, *, today: date | None = None) -> bool:
    """Whether `season` has begun, so its per-game sources could have rows upstream."""
    return (today or date.today()) >= season_start_date(season)


def selected_sources(*, with_pbp: bool = False) -> tuple[IngestSource, ...]:
    """The registry minus what this run opted out of. Order is preserved."""
    return tuple(s for s in INGEST_SOURCES if with_pbp or not s.heavy)


def refresh(
    seasons: Iterable[int],
    *,
    data_root: Path,
    with_pbp: bool = False,
    today: date | None = None,
) -> None:
    """Refresh every ingest source for ``seasons`` under ``data_root``, in registry order.

    **Fail-fast**: aborts on the first source failure. Re-running after a partial failure is safe
    but repeats the work of already-completed sources (per-source writes are idempotent in place).
    For an unattended run that isolates each source, classifies "not published yet" against a real
    defect, and prints a summary, use ``scripts/refresh_data.py`` — it drives this same registry.

    Sources whose rows cannot exist yet are skipped rather than attempted: asking nflverse for a
    season that has not kicked off raises an error indistinguishable from a real defect, and
    fail-fast means that error would abort every source after it.
    """
    # Materialize: per-source calls each iterate ``seasons``; a generator would be exhausted
    # after the first call.
    season_list = list(seasons)
    playable = [s for s in season_list if games_played(s, today=today)]
    for source in selected_sources(with_pbp=with_pbp):
        applicable = playable if source.needs_games_played else season_list
        if not applicable:
            continue
        source.run(data_root, applicable)
