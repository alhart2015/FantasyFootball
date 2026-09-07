"""Tests for scripts/refresh_data.py.

The behaviour worth pinning is not "does it call the ingest functions" -- it is the classification
that makes the script trustworthy to run unattended: an upstream 404 must come back SKIPPED and
exit 0, a schema regression must come back FAILED and exit 1, and one source's failure must never
cost the others their run. Getting that backwards in either direction is the bug this script
exists to prevent, so every case below is a boundary of `classify_error` or of the isolation.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

# script import (scripts/ on sys.path via conftest)
from refresh_data import (
    Status,
    StepResult,
    classify_error,
    default_season,
    parse_seasons,
    refresh_derived,
    refresh_sources,
    render_summary,
    run_step,
)

from projections.ingest.sources import INGEST_SOURCES, IngestSource

# --- classify_error -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        # The exact shapes seen in the wild, verbatim from a real 2026 refresh attempt.
        ConnectionError(
            "Failed to download https://github.com/nflverse/nflverse-data/releases/download/"
            "stats_player/stats_player_week_2026.parquet: 404 Client Error: Not Found for url: "
            "https://github.com/nflverse/nflverse-data/releases/download/stats_player/"
            "stats_player_week_2026.parquet"
        ),
        ValueError("Season must be between 2012 and 2025"),
        ValueError("Season must be between 2016 and 2025"),
    ],
)
def test_upstream_has_no_data_yet_is_skipped(exc: Exception) -> None:
    assert classify_error(exc) is Status.SKIPPED


@pytest.mark.parametrize(
    "exc",
    [
        # A pandera failure is the depth_charts placeholder-id bug: a real defect that must not
        # be laundered into "nothing to fetch" just because it happened during an ingest.
        ValueError("Column 'gsis_id' failed element-wise validator number 0: str_matches(...)"),
        KeyError("season"),
        RuntimeError("External API error for season 2026: 500"),
        # Near-miss wording: a message that merely mentions a season must not match the marker.
        ValueError("Season 2026 is not between our supported bounds"),
    ],
)
def test_real_defects_are_failures(exc: Exception) -> None:
    assert classify_error(exc) is Status.FAILED


# --- default_season -------------------------------------------------------------------------


def test_default_season_prefers_the_configured_league(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented primary behaviour: one loadable profile wins over the calendar.

    Monkeypatched rather than written to disk because `load_profile` also resolves the league
    config; this asserts the season-selection branch, which is what `default_season` owns. An
    earlier version of this test wrote an unparseable profile and so silently exercised the
    calendar fallback instead -- the branch below could have been deleted with the suite green.
    """
    import refresh_data

    profile = SimpleNamespace(key="critts_2025_2026", season=2025)
    monkeypatch.setattr(refresh_data, "discover_profiles", lambda _root: ([profile], []))
    # 2025 despite a September-2026 calendar: the profile is the authority.
    assert default_season(tmp_path, today=date(2026, 9, 7)) == 2025


def test_default_season_falls_back_when_profiles_disagree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two profiles naming different seasons is ambiguous, so neither wins."""
    import refresh_data

    profiles = [
        SimpleNamespace(key="a", season=2025),
        SimpleNamespace(key="b", season=2026),
    ]
    monkeypatch.setattr(refresh_data, "discover_profiles", lambda _root: (profiles, []))
    assert default_season(tmp_path, today=date(2026, 9, 7)) == 2026


def test_default_season_survives_an_unreadable_profile(tmp_path: Path) -> None:
    """A typo'd profile falls back to the calendar rather than raising -- refreshing the wrong
    season costs time, not correctness, and `refresh_derived` reports the mismatch by name."""
    league_dir = tmp_path / "leagues" / "some_league"
    league_dir.mkdir(parents=True)
    (league_dir / "board_profile.json").write_text("{ not json")
    assert default_season(tmp_path / "leagues", today=date(2026, 9, 7)) == 2026


def test_default_season_falls_back_to_the_calendar(tmp_path: Path) -> None:
    empty = tmp_path / "no_leagues_here"
    assert default_season(empty, today=date(2026, 9, 7)) == 2026
    # January still belongs to the season that just ended, not the one 8 months away.
    assert default_season(empty, today=date(2027, 1, 15)) == 2026
    assert default_season(empty, today=date(2027, 3, 1)) == 2027


# --- parse_seasons --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026", [2026]),
        ("2021-2025", [2021, 2022, 2023, 2024, 2025]),
        ("2021,2026", [2021, 2026]),
        ("2024-2025,2021", [2021, 2024, 2025]),
        ("2026,2026", [2026]),  # de-duplicated
    ],
)
def test_parse_seasons(raw: str, expected: list[int]) -> None:
    assert parse_seasons(raw) == expected


@pytest.mark.parametrize("raw", ["2025-2021", "twenty-twentysix", ""])
def test_parse_seasons_rejects_garbage(raw: str) -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        parse_seasons(raw)


# --- isolation ------------------------------------------------------------------------------


def test_run_step_converts_an_exception_into_a_result() -> None:
    def _boom() -> str:
        raise ValueError("Season must be between 2012 and 2025")

    result = run_step("weekly_stats", _boom)
    assert result.status is Status.SKIPPED
    assert result.name == "weekly_stats"
    assert "Season must be between" in result.detail


def test_run_step_catches_systemexit_from_a_sibling_script() -> None:
    """`generate_preset_vorp_tables.main` raises `SystemExit` when the id_map is missing, and
    `SystemExit` is a `BaseException`. A bare `except Exception` let it terminate the process
    after the entire raw pull -- discarding every collected result and printing no summary."""

    def _boom() -> str:
        raise SystemExit("No id_map at data/raw/id_map.parquet")

    result = run_step("vorp_presets", _boom)
    assert result.status is Status.FAILED
    assert "No id_map" in result.detail


def test_run_step_still_lets_keyboardinterrupt_through() -> None:
    """Ctrl-C must stop the run, not be recorded as one source quietly failing."""

    def _boom() -> str:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_step("weekly_stats", _boom)


def test_run_step_truncates_a_novel_length_traceback_message() -> None:
    """A pandera SchemaError lists every failure case; unabridged it buries the summary table."""

    def _boom() -> str:
        raise ValueError("x" * 5000)

    result = run_step("depth_charts", _boom)
    assert result.status is Status.FAILED
    assert len(result.detail) <= 200


def test_run_step_keeps_only_the_first_line() -> None:
    def _boom() -> str:
        raise ValueError("headline\nstack noise\nmore noise")

    assert run_step("x", _boom).detail == "ValueError: headline"


# --- derived gating -------------------------------------------------------------------------


def test_derived_tables_are_skipped_when_projections_did_not_refresh(tmp_path: Path) -> None:
    """The gate that stops a stale pool from looking fresh. Rebuilding off an unchanged snapshot
    writes new mtimes over identical numbers, which is indistinguishable from a real refresh."""
    results = refresh_derived(
        data_root=tmp_path,
        season=2026,
        projections_ok=False,
        profile_root=tmp_path / "leagues",
    )
    assert [r.status for r in results] == [Status.SKIPPED]
    assert "did not refresh" in results[0].detail


def test_skipped_derived_names_every_league_it_did_not_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single generic "presets skipped" line reads as though the league pools were fine. The
    league pool is the one the in-season CLIs actually load, so it must appear by name."""
    import refresh_data

    fake = SimpleNamespace(key="critts_2025_2026", season=2026)
    monkeypatch.setattr(refresh_data, "discover_profiles", lambda _root: ([fake], []))

    results = refresh_derived(
        data_root=tmp_path, season=2026, projections_ok=False, profile_root=tmp_path / "leagues"
    )
    names = [r.name for r in results]
    assert names == ["vorp_presets", "league:critts_2025_2026"]
    assert all(r.status is Status.SKIPPED for r in results)


def test_a_malformed_profile_is_reported_even_when_projections_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken profile is broken regardless of whether the projections refreshed. Reporting it
    only on the happy path means the user discovers the typo on some later run, after fixing an
    unrelated problem -- while that league quietly serves last week's numbers in the meantime."""
    import refresh_data

    err = SimpleNamespace(
        path=tmp_path / "leagues" / "critts" / "board_profile.json", message="bad key 'my_slott'"
    )
    monkeypatch.setattr(refresh_data, "discover_profiles", lambda _root: ([], [err]))

    results = refresh_derived(
        data_root=tmp_path, season=2026, projections_ok=False, profile_root=tmp_path / "leagues"
    )
    failures = [r for r in results if r.status is Status.FAILED]
    assert [r.name for r in failures] == ["league:critts"]
    assert "my_slott" in failures[0].detail


# --- driving the shared registry --------------------------------------------------------------


def _stub_registry(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, list[int]]]) -> None:
    """Replace every registry entry's `run` with a recorder, keeping names and flags.

    Patched at the registry rather than per-function so this test cannot go stale when a source
    is added -- which is the entire reason the registry exists.
    """
    import refresh_data

    def _recorder(name: str) -> Callable[[Path, list[int]], list[Path]]:
        def _run(data_root: Path, seasons: list[int]) -> list[Path]:
            del data_root
            calls.append((name, list(seasons)))
            return []

        return _run

    stubbed = tuple(
        IngestSource(s.name, s.needs_games_played, s.heavy, _recorder(s.name))
        for s in INGEST_SOURCES
    )
    monkeypatch.setattr(refresh_data, "selected_sources", lambda *, with_pbp=False: stubbed)


def test_every_registry_source_gets_a_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The script reports on the registry, not on a list of its own. Adding a source to
    `INGEST_SOURCES` must make it appear here with no change to the script."""
    calls: list[tuple[str, list[int]]] = []
    _stub_registry(monkeypatch, calls)

    results = refresh_sources(data_root=tmp_path, seasons=[2024], today=date(2026, 9, 7))
    assert {r.name for r in results} == {s.name for s in INGEST_SOURCES}


def test_a_season_that_has_not_kicked_off_is_reported_even_when_others_have(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--seasons 2024-2026` today must not let 2026 vanish. Running the rest and reporting them
    OK is a clean zero-failure run that says nothing about a season the user explicitly asked
    for."""
    calls: list[tuple[str, list[int]]] = []
    _stub_registry(monkeypatch, calls)

    results = refresh_sources(
        data_root=tmp_path, seasons=[2024, 2025, 2026], today=date(2026, 9, 7)
    )
    skipped = [r for r in results if r.status is Status.SKIPPED]
    assert [r.name for r in skipped] == ["game_stats(not started)"]
    assert "2026 kicks off 2026-09-10" in skipped[0].detail
    # The seasons that DID run are named too, so "ran for [2024, 2025] only" is unambiguous.
    assert "[2024, 2025]" in skipped[0].detail

    by_name = dict(calls)
    assert by_name["weekly_stats"] == [2024, 2025]  # per-game: playable seasons only
    assert by_name["schedules"] == [2024, 2025, 2026]  # market-facing: every requested season


def test_no_extra_row_when_every_requested_season_has_kicked_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, list[int]]] = []
    _stub_registry(monkeypatch, calls)

    results = refresh_sources(data_root=tmp_path, seasons=[2024, 2025], today=date(2026, 9, 7))
    assert not [r for r in results if r.status is Status.SKIPPED]


def test_per_game_sources_are_skipped_by_name_before_kickoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2026-09-07 case. Each per-game source gets its own SKIPPED row carrying the kickoff
    date, and the market-facing four still run -- that is what makes a preseason refresh useful."""
    calls: list[tuple[str, list[int]]] = []
    _stub_registry(monkeypatch, calls)

    results = refresh_sources(data_root=tmp_path, seasons=[2026], today=date(2026, 9, 7))
    by_status = {r.name: r.status for r in results}

    assert by_status["weekly_stats"] is Status.SKIPPED
    assert by_status["ngs_passing"] is Status.SKIPPED
    assert by_status["external_projections"] is Status.OK
    assert by_status["id_map"] is Status.OK
    assert {name for name, _ in calls} == {
        "id_map",
        "schedules",
        "draft_picks",
        "external_projections",
    }
    detail = next(r.detail for r in results if r.name == "weekly_stats")
    assert "2026 kicks off 2026-09-10" in detail


# --- summary --------------------------------------------------------------------------------


def test_summary_counts_and_calls_out_failures() -> None:
    out = render_summary(
        [
            StepResult("id_map", Status.OK, "data/raw/id_map.parquet"),
            StepResult("weekly_stats", Status.SKIPPED, "no games played yet"),
            StepResult("depth_charts", Status.FAILED, "SchemaError: bad gsis_id"),
        ]
    )
    assert "1 refreshed" in out
    assert "1 skipped" in out
    assert "1 failed" in out
    # A failure must be repeated in its own block; a reader scanning the table can miss one row.
    assert out.count("depth_charts") == 2
    assert "--verbose" in out


def test_summary_of_an_all_clear_run_says_nothing_about_failures() -> None:
    out = render_summary([StepResult("id_map", Status.OK, "ok")])
    assert "FAILED means" not in out
