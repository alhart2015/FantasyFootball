"""`projections.ingest.sources` — the one ingest-source registry and the fail-fast orchestrator.

These tests assert against `INGEST_SOURCES` itself rather than a hand-written list of expected
calls. That is the point: a second hand-maintained list of "every source" — in a test or in another
orchestrator — is exactly how this module ended up missing `draft_picks`, `external_projections`,
and `pbp` while `scripts/refresh_data.py` had them. A test that enumerates sources independently
would have to be updated alongside the registry, and would pass while drifting from it.

Sources are mocked at the registry level so nothing here touches the network.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from projections.ingest import sources as sources_mod
from projections.ingest.sources import (
    INGEST_SOURCES,
    IngestSource,
    games_played,
    season_start_date,
    selected_sources,
)


def _record(calls: list[tuple[str, list[int]]], source: IngestSource) -> IngestSource:
    """A registry entry whose `run` records the seasons it was asked for and writes nothing."""

    def _run(data_root: Path, seasons: list[int]) -> list[Path]:
        del data_root
        calls.append((source.name, list(seasons)))
        return []

    return IngestSource(
        name=source.name,
        needs_games_played=source.needs_games_played,
        heavy=source.heavy,
        run=_run,
    )


def _patched_registry(calls: list[tuple[str, list[int]]]) -> Any:
    return patch.object(
        sources_mod, "INGEST_SOURCES", tuple(_record(calls, s) for s in INGEST_SOURCES)
    )


# --- the registry itself ---------------------------------------------------------------------


def test_registry_covers_every_source_we_ingest() -> None:
    """The list this module is the single source of truth for. `draft_picks`,
    `external_projections`, and `pbp` are named explicitly because their absence is the concrete
    drift that motivated consolidating the two orchestrators."""
    names = {s.name for s in INGEST_SOURCES}
    assert names == {
        "id_map",
        "schedules",
        "draft_picks",
        "external_projections",
        "weekly_stats",
        "depth_charts",
        "snap_counts",
        "ngs_passing",
        "ngs_rushing",
        "ngs_receiving",
        "pbp",
    }


def test_registry_names_are_unique() -> None:
    """Names address a source in the summary and in the `PROJECTIONS_SOURCE` gate; a duplicate
    would make one of them unreachable."""
    names = [s.name for s in INGEST_SOURCES]
    assert len(names) == len(set(names))


def test_id_map_precedes_snap_counts() -> None:
    """snap_counts joins on the gsis_id <-> pfr_id table build_id_map writes, and raises
    FileNotFoundError without it. Order is encoded by position in the registry."""
    names = [s.name for s in INGEST_SOURCES]
    assert names.index("id_map") < names.index("snap_counts")


def test_schedules_precedes_depth_charts() -> None:
    """depth_charts needs schedules to resolve (season, week) from the 2025+ snapshot format."""
    names = [s.name for s in INGEST_SOURCES]
    assert names.index("schedules") < names.index("depth_charts")


def test_only_pbp_is_heavy() -> None:
    assert [s.name for s in INGEST_SOURCES if s.heavy] == ["pbp"]


def test_market_facing_sources_do_not_need_games_played() -> None:
    """These four are what make a preseason refresh worth running; gating them on kickoff would
    leave nothing to do in August."""
    always = {s.name for s in INGEST_SOURCES if not s.needs_games_played}
    assert always == {"id_map", "schedules", "draft_picks", "external_projections"}


def test_selected_sources_excludes_pbp_by_default() -> None:
    assert "pbp" not in {s.name for s in selected_sources()}
    assert "pbp" in {s.name for s in selected_sources(with_pbp=True)}


def test_selected_sources_preserves_registry_order() -> None:
    selected = [s.name for s in selected_sources(with_pbp=True)]
    assert selected == [s.name for s in INGEST_SOURCES]


# --- refresh() -------------------------------------------------------------------------------


def test_refresh_runs_every_source_in_the_registry(tmp_path: Path) -> None:
    calls: list[tuple[str, list[int]]] = []
    seasons = [2018, 2019]
    with _patched_registry(calls):
        sources_mod.refresh(seasons=seasons, data_root=tmp_path)

    # Every non-heavy source, once, with the materialized season list.
    assert [name for name, _ in calls] == [s.name for s in selected_sources()]
    assert all(got == seasons for _, got in calls)


def test_refresh_materializes_a_generator(tmp_path: Path) -> None:
    """Each source iterates `seasons`; a generator would be exhausted after the first one and
    every later source would silently receive nothing."""
    calls: list[tuple[str, list[int]]] = []
    with _patched_registry(calls):
        sources_mod.refresh(seasons=(s for s in [2018, 2019]), data_root=tmp_path)

    assert len(calls) == len(selected_sources())
    assert all(got == [2018, 2019] for _, got in calls)


def test_refresh_skips_pbp_unless_asked(tmp_path: Path) -> None:
    calls: list[tuple[str, list[int]]] = []
    with _patched_registry(calls):
        sources_mod.refresh(seasons=[2024], data_root=tmp_path)
    assert "pbp" not in {name for name, _ in calls}

    calls.clear()
    with _patched_registry(calls):
        sources_mod.refresh(seasons=[2024], data_root=tmp_path, with_pbp=True)
    assert "pbp" in {name for name, _ in calls}


def test_refresh_does_not_ask_for_a_season_that_has_not_kicked_off(tmp_path: Path) -> None:
    """Fail-fast plus an unplayable season would abort every source after the first per-game one.
    The market-facing sources still run — that is the whole value of a preseason refresh."""
    calls: list[tuple[str, list[int]]] = []
    with _patched_registry(calls):
        sources_mod.refresh(seasons=[2026], data_root=tmp_path, today=date(2026, 9, 7))

    ran = {name for name, _ in calls}
    assert ran == {"id_map", "schedules", "draft_picks", "external_projections"}


def test_refresh_asks_per_game_sources_only_for_seasons_that_started(tmp_path: Path) -> None:
    calls: list[tuple[str, list[int]]] = []
    with _patched_registry(calls):
        sources_mod.refresh(seasons=[2024, 2025, 2026], data_root=tmp_path, today=date(2026, 9, 7))

    by_name = dict(calls)
    assert by_name["weekly_stats"] == [2024, 2025]
    # Market-facing sources still get every requested season.
    assert by_name["schedules"] == [2024, 2025, 2026]


def test_refresh_is_fail_fast(tmp_path: Path) -> None:
    """The library entrypoint aborts on the first failure; `scripts/refresh_data.py` is the
    isolating runner. Asserting this keeps the two contracts distinct."""
    calls: list[tuple[str, list[int]]] = []

    def _boom(data_root: Path, seasons: list[int]) -> list[Path]:
        raise RuntimeError("upstream is down")

    sources = tuple(
        IngestSource(s.name, s.needs_games_played, s.heavy, _boom)
        if s.name == "schedules"
        else _record(calls, s)
        for s in INGEST_SOURCES
    )
    with patch.object(sources_mod, "INGEST_SOURCES", sources), pytest.raises(RuntimeError):
        sources_mod.refresh(seasons=[2024], data_root=tmp_path)

    # id_map ran (it precedes schedules); nothing after schedules did.
    assert [name for name, _ in calls] == ["id_map"]


# --- calendar --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("season", "expected"),
    [
        (2026, date(2026, 9, 10)),  # Labor Day 2026-09-07 -> kickoff Thursday 09-10
        (2025, date(2025, 9, 4)),  # Labor Day 2025-09-01 -> kickoff Thursday 09-04
        (2024, date(2024, 9, 5)),  # Labor Day 2024-09-02 -> kickoff Thursday 09-05
    ],
)
def test_season_start_is_the_thursday_after_labor_day(season: int, expected: date) -> None:
    assert season_start_date(season) == expected


def test_games_played_flips_on_kickoff_day() -> None:
    assert not games_played(2026, today=date(2026, 9, 7))
    assert not games_played(2026, today=date(2026, 9, 9))
    assert games_played(2026, today=date(2026, 9, 10))
    assert games_played(2026, today=date(2026, 12, 1))
