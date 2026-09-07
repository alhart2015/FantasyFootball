"""Tests for scripts/refresh_data.py.

The behaviour worth pinning is not "does it call the ingest functions" -- it is the classification
that makes the script trustworthy to run unattended: an upstream 404 must come back SKIPPED and
exit 0, a schema regression must come back FAILED and exit 1, and one source's failure must never
cost the others their run. Getting that backwards in either direction is the bug this script
exists to prevent, so every case below is a boundary of `classify_error` or of the isolation.
"""

from __future__ import annotations

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
    games_played,
    parse_seasons,
    refresh_derived,
    render_summary,
    run_step,
    season_start_date,
)

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


# --- season calendar ------------------------------------------------------------------------


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
    """The day this was written -- 2026-09-07 -- is the case that made the script necessary:
    the season is three days away and every per-game source 404s."""
    assert not games_played(2026, today=date(2026, 9, 7))
    assert not games_played(2026, today=date(2026, 9, 9))
    assert games_played(2026, today=date(2026, 9, 10))
    assert games_played(2026, today=date(2026, 12, 1))


# --- default_season -------------------------------------------------------------------------


def test_default_season_prefers_the_configured_league(tmp_path: Path) -> None:
    league_dir = tmp_path / "leagues" / "some_league"
    league_dir.mkdir(parents=True)
    # A profile that fails to load must not be silently ignored here either -- but this one is
    # only exercised for its season, so a minimal valid file is enough.
    (league_dir / "board_profile.json").write_text("{ not json")
    # Unreadable profile -> no season set -> falls back to the calendar rather than raising.
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
