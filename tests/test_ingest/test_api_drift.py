"""Opt-in API-drift smoke tests for `nfl_data_py`.

Marked `@pytest.mark.network`; skipped by default. Run with:

    pytest -m network --run-network -q

Each test fetches a tiny live slice (one season) via the same private
`_fetch_raw_*` hook that the CI tests monkeypatch, asserts that every raw
column our normalize step depends on is present, and then runs the
normalize end-to-end (schema validation gates this — pandera will throw
on dtype / value drift). Designed to be the canonical post-bump check
after upgrading the `nfl_data_py` pin in `pyproject.toml`.

Test season: 2023 — old enough to be stable upstream, recent enough to
exercise the full column set including `attempts` / `completions` /
`sacks` (added to `WeeklyStatsSchema` in Plan 2b) and the modern
rank-based depth-chart labels. Seasons before ~2018 used different
column conventions and would force the smokes to handle two contracts.

When a smoke fails: read the assertion message, confirm the raw column
rename or addition upstream, and patch the corresponding ingest module
(rename map, _KEEP list, schema, or normalize step). Add a TODO #16-style
note to `TODO.md` if the drift was non-trivial."""

from __future__ import annotations

from pathlib import Path

import pytest

from projections.ingest.depth_charts import (
    _fetch_raw_depth_charts,
)
from projections.ingest.depth_charts import (
    _normalize_one_season as _normalize_depth_charts,
)
from projections.ingest.id_map import _fetch_raw_id_map, build_id_map
from projections.ingest.ngs import (
    _fetch_raw_ngs,
)
from projections.ingest.ngs import (
    _normalize_one_season as _normalize_ngs,
)
from projections.ingest.schedules import (
    _fetch_raw_schedules,
)
from projections.ingest.schedules import (
    _normalize_one_season as _normalize_schedules,
)
from projections.ingest.snap_counts import (
    _fetch_raw_snap_counts,
)
from projections.ingest.snap_counts import (
    _normalize_one_season as _normalize_snap_counts,
)
from projections.ingest.weekly_stats import (
    _fetch_raw_weekly,
)
from projections.ingest.weekly_stats import (
    _normalize_one_season as _normalize_weekly,
)

pytestmark = pytest.mark.network

_DRIFT_SEASON = 2023


def _assert_columns_present(raw_columns: set[str], expected: set[str], source: str) -> None:
    """Friendly diff message: which columns we expect that the API didn't
    return. Surfaces drift before the normalize step swallows it via the
    defensive `[c for c in _KEEP if c in df.columns]` filters."""
    missing = expected - raw_columns
    assert not missing, (
        f"nfl_data_py {source} missing expected columns {sorted(missing)}; "
        f"got {sorted(raw_columns)}"
    )


def test_weekly_stats_api_columns_and_schema() -> None:
    raw = _fetch_raw_weekly([_DRIFT_SEASON])
    expected = {
        "player_id",
        "recent_team",
        "opponent_team",
        "season",
        "week",
        "position",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "attempts",
        "completions",
        "sacks",
        "rushing_yards",
        "rushing_tds",
        "carries",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "receiving_air_yards",
        "targets",
        # fumbles_lost is derived from these source-specific cols (no aggregated col upstream).
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
        "sack_fumbles_lost",
    }
    _assert_columns_present(set(raw.columns), expected, "import_weekly_data")
    df = _normalize_weekly(raw)
    assert not df.empty


def test_depth_charts_api_columns_and_schema() -> None:
    raw = _fetch_raw_depth_charts([_DRIFT_SEASON])
    expected = {
        "gsis_id",
        "season",
        "week",
        "club_code",
        "position",
        "depth_team",
        "depth_position",
    }
    _assert_columns_present(set(raw.columns), expected, "import_depth_charts")
    df = _normalize_depth_charts(raw)
    assert not df.empty


@pytest.mark.parametrize("stat_type", ["passing", "rushing", "receiving"])
def test_ngs_api_columns_and_schema(stat_type: str) -> None:
    raw = _fetch_raw_ngs(stat_type, [_DRIFT_SEASON])  # type: ignore[arg-type]
    common = {
        "player_gsis_id",
        "season",
        "week",
        "team_abbr",
        "player_position",
    }
    per_stat: dict[str, set[str]] = {
        "passing": {
            "avg_time_to_throw",
            "avg_completed_air_yards",
            "avg_intended_air_yards",
            "completion_percentage",
            "expected_completion_percentage",
            "completion_percentage_above_expectation",
        },
        "rushing": {
            "efficiency",
            "percent_attempts_gte_eight_defenders",
            "rush_yards_over_expected_per_att",
            "rush_attempts",
            "rush_yards",
        },
        "receiving": {
            "avg_separation",
            "avg_intended_air_yards",
            "percent_share_of_intended_air_yards",
            "avg_yac_above_expectation",
            "receptions",
            "targets",
        },
    }
    _assert_columns_present(
        set(raw.columns),
        common | per_stat[stat_type],
        f"import_ngs_data({stat_type!r})",
    )
    df = _normalize_ngs(stat_type, raw)  # type: ignore[arg-type]
    assert not df.empty


def test_schedules_api_columns_and_schema() -> None:
    raw = _fetch_raw_schedules([_DRIFT_SEASON])
    expected = {
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "gameday",
        "gametime",
        "spread_line",
        "total_line",
        "home_moneyline",
        "away_moneyline",
        "surface",
        "roof",
        "temp",
        "wind",
    }
    _assert_columns_present(set(raw.columns), expected, "import_schedules")
    df = _normalize_schedules(raw)
    assert not df.empty


def test_id_map_api_columns_and_schema(tmp_path: Path) -> None:
    """`build_id_map` is the only one without a separate `_normalize_one_season`
    — its full body runs `import_ids()` and writes a partition. Drift surfaces
    via the schema validation inside `build_id_map`."""
    raw = _fetch_raw_id_map()
    expected = {"gsis_id", "espn_id", "sleeper_id", "pfr_id", "name", "position", "team"}
    _assert_columns_present(set(raw.columns), expected, "import_ids")
    out = build_id_map(tmp_path)
    assert out.exists()


def test_snap_counts_api_columns_and_schema(tmp_path: Path) -> None:
    """Snap counts joins on `id_map.pfr_id`, so we have to build the id_map
    first (using the live API) before we can normalize."""
    raw = _fetch_raw_snap_counts([_DRIFT_SEASON])
    expected = {
        "pfr_player_id",
        "season",
        "week",
        "team",
        "opponent",
        "position",
        "offense_snaps",
        "offense_pct",
        "defense_snaps",
        "defense_pct",
        "st_snaps",
        "st_pct",
    }
    _assert_columns_present(set(raw.columns), expected, "import_snap_counts")
    build_id_map(tmp_path)
    df = _normalize_snap_counts(raw, tmp_path)
    assert not df.empty
