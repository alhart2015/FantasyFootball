"""Shared pytest fixtures.

The `fake_*_df` fixtures below mimic raw `nfl_data_py` responses (one row
per per-week slice) and are consumed by both the per-module ingest tests
under `tests/test_ingest/` and the end-to-end smoke test at this level.
They were promoted from `tests/test_ingest/conftest.py` so the top-level
`test_smoke_2a.py` can request them via pytest's hierarchical fixture
resolution.

The `baseline_features_wr` / `baseline_weekly_stats_wr` fixtures (and
their helpers `_GSIS_IDS`, `_TEAMS`, `_TARGETS_BASE`, `_wr_weekly_stats_row`)
were promoted from `tests/test_models/conftest.py` for the same reason:
the round-trip smoke test at this level needs to consume them, and pytest
fixtures only inherit from parent conftests, not siblings.

Plan 3b adds sibling per-position fixtures (`baseline_weekly_stats_{qb,
rb,te}` + `baseline_features_{qb,rb,te}`) at root scope so the
parametrized smoke test can resolve them across all four positions.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from projections.schemas import _PYARROW_STR


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register `--run-network` so the live nfl_data_py API-drift smokes
    only run when explicitly requested. The marker itself is declared in
    `pyproject.toml`; this option toggles the skip behavior in
    `pytest_collection_modifyitems` below."""
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run @pytest.mark.network tests that hit the live nfl_data_py API.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip network-marked tests unless `--run-network` was passed."""
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="needs --run-network to hit the live API")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture
def fake_id_map_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_ids()` — a row per player with cross-platform IDs.

    Kelce (TE/KC) is included so the smoke test can exercise the TE feature
    builder end-to-end through the snap_counts pfr_id -> gsis_id join.
    """
    return pd.DataFrame(
        {
            "gsis_id": ["00-0036322", "00-0034857", "00-0034796", "00-0030506"],
            "espn_id": ["4262921", "3915511", "4035687", "15847"],
            "sleeper_id": ["6794", "5849", "5045", "1466"],
            "pfr_id": ["JeffJu00", "MahoPa00", "BarkSa00", "KelcTr00"],
            "name": [
                "Justin Jefferson",
                "Patrick Mahomes",
                "Saquon Barkley",
                "Travis Kelce",
            ],
            "position": ["WR", "QB", "RB", "TE"],
            "team": ["MIN", "KC", "PHI", "KC"],
        }
    )


@pytest.fixture
def fake_weekly_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_weekly_data([2024])` — 3 player-weeks.

    Kelce (TE/KC) is included so the smoke test can exercise the TE feature
    builder against a non-empty weekly stats slice.
    """
    return pd.DataFrame(
        {
            "player_id": ["00-0036322", "00-0034857", "00-0030506"],
            "season": [2024, 2024, 2024],
            "week": [3, 3, 3],
            "position": ["WR", "QB", "TE"],
            "recent_team": ["MIN", "KC", "KC"],
            "opponent_team": ["HOU", "ATL", "ATL"],
            "passing_yards": [0.0, 286.0, 0.0],
            "passing_tds": [0, 2, 0],
            "interceptions": [0, 1, 0],
            "attempts": [0, 38, 0],
            "completions": [0, 24, 0],
            "sacks": [0, 2, 0],
            "rushing_yards": [0.0, 12.0, 0.0],
            "rushing_tds": [0, 0, 0],
            "carries": [0, 3, 0],
            "receptions": [9, 0, 5],
            "receiving_yards": [110.0, 0.0, 58.0],
            "receiving_tds": [1, 0, 1],
            "receiving_air_yards": [145.0, 0.0, 70.0],
            "targets": [12, 0, 7],
            "fumbles_lost": [0, 0, 0],
        }
    )


@pytest.fixture
def fake_schedules_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_schedules([2024])` — 3 games for week 3.

    Raw column names (before our _RENAME): nfl_data_py uses `gameday` (date)
    and `gametime` (HH:MM string) for kickoff; the ingest module combines them
    into a UTC `kickoff` timestamp.

    A PHI@TB game is included so the smoke test's RB builder (Barkley) has a
    real schedule row for the game-environment join.
    """
    return pd.DataFrame(
        {
            "game_id": ["2024_03_KC_ATL", "2024_03_MIN_HOU", "2024_03_PHI_TB"],
            "season": [2024, 2024, 2024],
            "week": [3, 3, 3],
            "home_team": ["ATL", "HOU", "TB"],
            "away_team": ["KC", "MIN", "PHI"],
            "gameday": ["2024-09-22", "2024-09-22", "2024-09-22"],
            "gametime": ["20:20", "13:00", "13:00"],
            "spread_line": [3.5, -2.5, 3.5],
            "total_line": [48.5, 44.0, 45.5],
            "home_moneyline": [155, -125, 150],
            "away_moneyline": [-180, 105, -175],
            "surface": ["fieldturf", "matrixturf", "grass"],
            "roof": ["dome", "dome", "outdoors"],
            "temp": [72, 72, 85],
            "wind": [0, 0, 6],
        }
    )


@pytest.fixture
def fake_snap_counts_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_snap_counts([2024])` — 2 player-weeks.

    NOTE (API drift vs. original spec): the real `nfl_data_py.import_snap_counts`
    output does NOT contain `gsis_id`; it contains `pfr_player_id` instead.
    The snap_counts ingest module (Task 10) must join on `pfr_id` through
    the id_map to produce the `gsis_id` required by `SnapCountsSchema`.
    The `pfr_player_id` values here correspond to `fake_id_map_df.pfr_id`.
    """
    return pd.DataFrame(
        {
            "game_id": [
                "2024_03_KC_ATL",
                "2024_03_MIN_HOU",
                "2024_03_KC_ATL",
            ],
            "season": [2024, 2024, 2024],
            "week": [3, 3, 3],
            "player": ["Patrick Mahomes", "Justin Jefferson", "Travis Kelce"],
            "position": ["QB", "WR", "TE"],
            "team": ["KC", "MIN", "KC"],
            "opponent": ["ATL", "HOU", "ATL"],
            "offense_snaps": [71, 62, 58],
            "offense_pct": [1.0, 0.95, 0.89],
            "defense_snaps": [0, 0, 0],
            "defense_pct": [0.0, 0.0, 0.0],
            "st_snaps": [0, 3, 0],
            "st_pct": [0.0, 0.10, 0.0],
            "pfr_player_id": ["MahoPa00", "JeffJu00", "KelcTr00"],
        }
    )


@pytest.fixture
def fake_depth_charts_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_depth_charts([2024])` — 4 player-weeks.

    Raw column names: `club_code` is the team (renamed to `team`); `depth_team`
    is the raw slot label (e.g., 'WR1', 'LWR'); `depth_position` may already
    be a numeric rank — the ingest module prefers `depth_position` if present
    and otherwise parses the trailing digit from `depth_team`.

    Kelce (TE/KC) is included so the smoke test can exercise the TE feature
    builder end-to-end.
    """
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "club_code": ["MIN", "KC", "PHI", "KC"],
            "week": [3, 3, 3, 3],
            "depth_team": ["WR1", "QB1", "RB1", "TE1"],
            "last_name": ["Jefferson", "Mahomes", "Barkley", "Kelce"],
            "first_name": ["Justin", "Patrick", "Saquon", "Travis"],
            "formation": ["Offense", "Offense", "Offense", "Offense"],
            "gsis_id": ["00-0036322", "00-0034857", "00-0034796", "00-0030506"],
            "jersey_number": [18, 15, 26, 87],
            "position": ["WR", "QB", "RB", "TE"],
            "elias_id": ["JEF845899", "MAH335103", "BAR123456", "KEL235109"],
            "depth_position": [1, 1, 1, 1],
            "football_name": [
                "Justin Jefferson",
                "Patrick Mahomes",
                "Saquon Barkley",
                "Travis Kelce",
            ],
        }
    )


@pytest.fixture
def fake_ngs_passing_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_ngs_data('passing', [2024])` — 1 QB-week.

    Raw column names: `player_gsis_id` (renamed to `gsis_id`),
    `team_abbr` (renamed to `team`), `player_position` (renamed to `position`).
    """
    return pd.DataFrame(
        {
            "season": [2024],
            "season_type": ["REG"],
            "week": [3],
            "player_display_name": ["Patrick Mahomes"],
            "player_position": ["QB"],
            "team_abbr": ["KC"],
            "avg_time_to_throw": [2.71],
            "avg_completed_air_yards": [6.2],
            "avg_intended_air_yards": [8.1],
            "avg_air_yards_differential": [-1.9],
            "aggressiveness": [12.5],
            "max_completed_air_distance": [42.0],
            "avg_air_yards_to_sticks": [-0.4],
            "completion_percentage": [68.5],
            "expected_completion_percentage": [65.2],
            "completion_percentage_above_expectation": [3.3],
            "avg_air_distance": [9.5],
            "max_air_distance": [55.0],
            "player_gsis_id": ["00-0034857"],
        }
    )


@pytest.fixture
def fake_ngs_rushing_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_ngs_data('rushing', [2024])` — 1 RB-week."""
    return pd.DataFrame(
        {
            "season": [2024],
            "season_type": ["REG"],
            "week": [3],
            "player_display_name": ["Saquon Barkley"],
            "player_position": ["RB"],
            "team_abbr": ["PHI"],
            "efficiency": [3.1],
            "percent_attempts_gte_eight_defenders": [22.5],
            "avg_time_to_los": [2.95],
            "rush_attempts": [18],
            "rush_yards": [102],
            "expected_rush_yards": [85.4],
            "rush_yards_over_expected": [16.6],
            "avg_rush_yards": [5.7],
            "rush_yards_over_expected_per_att": [0.9],
            "rush_pct_over_expected": [12.0],
            "player_gsis_id": ["00-0034796"],
        }
    )


@pytest.fixture
def fake_ngs_receiving_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_ngs_data('receiving', [2024])` — 1 WR-week."""
    return pd.DataFrame(
        {
            "season": [2024],
            "season_type": ["REG"],
            "week": [3],
            "player_display_name": ["Justin Jefferson"],
            "player_position": ["WR"],
            "team_abbr": ["MIN"],
            "avg_cushion": [5.4],
            "avg_separation": [3.2],
            "avg_intended_air_yards": [12.1],
            "percent_share_of_intended_air_yards": [29.5],
            "receptions": [9],
            "targets": [12],
            "catch_percentage": [75.0],
            "yards": [110],
            "rec_touchdowns": [1],
            "avg_yac": [4.0],
            "avg_expected_yac": [3.5],
            "avg_yac_above_expectation": [0.5],
            "player_gsis_id": ["00-0036322"],
        }
    )


# ---------------------------------------------------------------------------
# Synthetic features + truth fixtures for BaselineModel unit tests.
# Promoted from tests/test_models/conftest.py so sibling test directories
# (e.g. the top-level smoke test) can consume them via pytest's hierarchical
# fixture resolution.
# ---------------------------------------------------------------------------

# 5 synthetic WRs across 2 teams; 8 weeks of 2024 + 4 weeks of 2025.
_GSIS_IDS = ["00-0010001", "00-0010002", "00-0010003", "00-0010004", "00-0010005"]
_TEAMS = ["KC", "KC", "MIN", "MIN", "MIN"]
_TARGETS_BASE = [10.0, 6.0, 9.0, 4.0, 3.0]  # per-player target rate


def _wr_weekly_stats_row(
    *, gsis_id: str, season: int, week: int, team: str, opponent: str, base_targets: float
) -> dict[str, object]:
    """Build a single WeeklyStatsSchema-valid row with stat values that scale
    plausibly with the per-player base target rate. Random jitter is added by
    the caller via week index so trailing-4 means are non-constant."""
    targets_jitter = (week % 3) - 1  # -1, 0, +1, repeating
    targets = max(0, int(base_targets + targets_jitter))
    receptions = max(0, int(targets * 0.65))
    rec_yards = float(receptions * 12.0)
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": "WR",
        "team": team,
        "opponent": opponent,
        "passing_yards": 0.0,
        "passing_tds": 0,
        "interceptions": 0,
        "attempts": 0,
        "completions": 0,
        "sacks": 0,
        "rushing_yards": 0.0,
        "rushing_tds": 0,
        "carries": 0,
        "receptions": receptions,
        "receiving_yards": rec_yards,
        "receiving_tds": 1 if (week + int(gsis_id[-1])) % 4 == 0 else 0,
        "receiving_air_yards": float(targets * 13.0),
        "targets": targets,
        "fumbles_lost": 0,
    }


@pytest.fixture
def baseline_weekly_stats_wr() -> pd.DataFrame:
    """8 weeks of 2024 + 4 weeks of 2025 WR-shaped stats for 5 synthetic WRs.

    2024 = training universe; 2025 = held-out. KC and MIN play each other
    every week so the opponent-allowed-fppg proxy resolves (MIN's allowed WR
    FPPG comes from KC's WRs and vice versa).
    """
    rows: list[dict[str, object]] = []
    for season, weeks in [(2024, range(1, 9)), (2025, range(1, 5))]:
        for week in weeks:
            for gsis_id, team, base_targets in zip(_GSIS_IDS, _TEAMS, _TARGETS_BASE, strict=True):
                opponent = "MIN" if team == "KC" else "KC"
                rows.append(
                    _wr_weekly_stats_row(
                        gsis_id=gsis_id,
                        season=season,
                        week=week,
                        team=team,
                        opponent=opponent,
                        base_targets=base_targets,
                    )
                )

    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def baseline_features_wr(baseline_weekly_stats_wr: pd.DataFrame) -> pd.DataFrame:
    """WR feature rows produced by build_wr_features for every (season, week)
    in the WR training fixture. Built up-front so tests don't pay the cost
    individually."""
    from projections.features import build_wr_features

    # We need supporting fixtures (snap_counts, depth_charts, ngs, schedules)
    # for the builder. Keep them minimal.
    snap_rows = [
        {
            "gsis_id": r["gsis_id"],
            "season": r["season"],
            "week": r["week"],
            "team": r["team"],
            "opponent": r["opponent"],
            "position": "WR",
            "offense_snaps": 60,
            "offense_pct": 0.95,
            "defense_snaps": 0,
            "defense_pct": 0.0,
            "st_snaps": 2,
            "st_pct": 0.05,
        }
        for _, r in baseline_weekly_stats_wr.iterrows()
    ]
    snap_counts = pd.DataFrame(snap_rows)
    for col in ("gsis_id", "team", "opponent", "position"):
        snap_counts[col] = snap_counts[col].astype(_PYARROW_STR)

    # Depth charts: every player as their team's WR1/WR2 (deterministic by
    # base_targets ranking) for every (season, week).
    dc_rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            for gsis_id, team, _base_targets in zip(_GSIS_IDS, _TEAMS, _TARGETS_BASE, strict=True):
                # Rank within team by base_targets descending.
                team_pool = sorted(
                    [
                        (g, t, b)
                        for g, t, b in zip(_GSIS_IDS, _TEAMS, _TARGETS_BASE, strict=True)
                        if t == team
                    ],
                    key=lambda x: -x[2],
                )
                rank = next(i for i, (g, _, _) in enumerate(team_pool, start=1) if g == gsis_id)
                dc_rows.append(
                    {
                        "gsis_id": gsis_id,
                        "season": season,
                        "week": week,
                        "team": team,
                        "position": "WR",
                        "depth_team": f"WR{rank}",
                        "depth_rank": rank,
                    }
                )
    depth = pd.DataFrame(dc_rows)
    for col in ("gsis_id", "team", "position", "depth_team"):
        depth[col] = depth[col].astype(_PYARROW_STR)

    # NGS: per-(season, week) snapshot per WR with stable, per-player synthetic
    # values keyed off base_targets so the rows are non-NaN at fit time.
    # The wr.py builder applies prior_mask and then takes the latest snapshot
    # per gsis_id; values vary slightly week-to-week so the snapshot isn't
    # week-1-only.
    ngs_rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            for gsis_id, team, base_targets in zip(_GSIS_IDS, _TEAMS, _TARGETS_BASE, strict=True):
                ngs_rows.append(
                    {
                        "gsis_id": gsis_id,
                        "season": season,
                        "week": week,
                        "team": team,
                        "position": "WR",
                        "avg_separation": 2.5 + base_targets * 0.05,
                        "avg_intended_air_yards": 9.0 + base_targets * 0.2,
                        # 0-100 scale (wr.py divides by 100 for the schema's 0-1 range).
                        "percent_share_of_intended_air_yards": min(95.0, 5.0 + base_targets * 4.0),
                        "avg_yac_above_expectation": -0.2 + base_targets * 0.05,
                    }
                )
    ngs = pd.DataFrame(ngs_rows)
    for col in ("gsis_id", "team", "position"):
        ngs[col] = ngs[col].astype(_PYARROW_STR)

    # Schedules: KC hosts MIN every week. Single game per (season, week)
    # covers both teams' WRs.
    sch_rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            sch_rows.append(
                {
                    "season": season,
                    "week": week,
                    "game_id": f"{season}_{week:02d}_KC_MIN",
                    "home_team": "KC",
                    "away_team": "MIN",
                    "kickoff": pd.Timestamp(f"{season}-09-{week + 1:02d}T17:00:00Z")
                    .tz_convert("UTC")
                    .as_unit("us"),
                    "spread_line": -3.0,
                    "total_line": 47.0,
                    "home_moneyline": -150,
                    "away_moneyline": 130,
                    "surface": "grass",
                    "roof": "outdoors",
                    "temp": 60,
                    "wind": 5,
                }
            )
    schedules = pd.DataFrame(sch_rows)
    for col in ("game_id", "home_team", "away_team", "surface", "roof"):
        schedules[col] = schedules[col].astype(_PYARROW_STR)
    schedules["temp"] = schedules["temp"].astype(pd.Int64Dtype())
    schedules["wind"] = schedules["wind"].astype(pd.Int64Dtype())
    schedules["home_moneyline"] = schedules["home_moneyline"].astype(pd.Int64Dtype())
    schedules["away_moneyline"] = schedules["away_moneyline"].astype(pd.Int64Dtype())

    feat_frames: list[pd.DataFrame] = []
    for season, weeks in [(2024, range(1, 9)), (2025, range(1, 5))]:
        for week in weeks:
            f = build_wr_features(
                weekly_stats=baseline_weekly_stats_wr,
                snap_counts=snap_counts,
                depth_charts=depth,
                ngs_receiving=ngs,
                schedules=schedules,
                season=season,
                as_of_week=week,
            )
            feat_frames.append(f)
    return pd.concat(feat_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-position baseline fixtures (Plan 3b).
# Each {position} variant provides a baseline_weekly_stats_{position} +
# baseline_features_{position} pair shaped to that position's target stats
# and feature schema. The smoke test (test_smoke.py) parametrizes across
# all four positions, so every fixture must live at root scope; pytest
# fixtures only inherit downward.
# ---------------------------------------------------------------------------

# Synthetic universe per position. 5 players across 2 teams (KC/MIN), 8
# weeks of 2024 + 4 weeks of 2025. KC and MIN play each other every week
# so the opp_allowed_*_fppg_l4 proxy resolves.
_POSITION_BASE_RATES: dict[str, list[float]] = {
    "QB": [38.0, 32.0, 36.0, 26.0, 22.0],  # pass attempts/game baseline
    "RB": [16.0, 12.0, 14.0, 8.0, 6.0],  # carries/game baseline
    "TE": [9.0, 6.0, 8.0, 3.0, 2.0],  # targets/game baseline
}


def _qb_weekly_stats_row(
    *,
    gsis_id: str,
    season: int,
    week: int,
    team: str,
    opponent: str,
    base_attempts: float,
) -> dict[str, object]:
    jitter = (week % 3) - 1
    attempts = max(0, int(base_attempts + jitter))
    completions = max(0, int(attempts * 0.65))
    pass_yards = float(completions * 11.0)
    pass_tds = 1 if (week + int(gsis_id[-1])) % 3 == 0 else 0
    interceptions = 1 if (week + int(gsis_id[-1])) % 5 == 0 else 0
    sacks = 1 if week % 4 == 0 else 0
    rush_attempts = 3 + (week % 2)
    rush_yards = float(rush_attempts * 4.0)
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": "QB",
        "team": team,
        "opponent": opponent,
        "passing_yards": pass_yards,
        "passing_tds": pass_tds,
        "interceptions": interceptions,
        "attempts": attempts,
        "completions": completions,
        "sacks": sacks,
        "rushing_yards": rush_yards,
        "rushing_tds": 0,
        "carries": rush_attempts,
        "receptions": 0,
        "receiving_yards": 0.0,
        "receiving_tds": 0,
        "receiving_air_yards": 0.0,
        "targets": 0,
        "fumbles_lost": 0,
    }


def _rb_weekly_stats_row(
    *,
    gsis_id: str,
    season: int,
    week: int,
    team: str,
    opponent: str,
    base_carries: float,
) -> dict[str, object]:
    jitter = (week % 3) - 1
    carries = max(0, int(base_carries + jitter))
    rush_yards = float(carries * 4.5)
    rush_tds = 1 if (week + int(gsis_id[-1])) % 4 == 0 else 0
    targets = max(0, int(base_carries * 0.25))
    receptions = max(0, int(targets * 0.7))
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": "RB",
        "team": team,
        "opponent": opponent,
        "passing_yards": 0.0,
        "passing_tds": 0,
        "interceptions": 0,
        "attempts": 0,
        "completions": 0,
        "sacks": 0,
        "rushing_yards": rush_yards,
        "rushing_tds": rush_tds,
        "carries": carries,
        "receptions": receptions,
        "receiving_yards": float(receptions * 7.0),
        "receiving_tds": 0,
        "receiving_air_yards": float(targets * 5.0),
        "targets": targets,
        "fumbles_lost": 0,
    }


def _te_weekly_stats_row(
    *,
    gsis_id: str,
    season: int,
    week: int,
    team: str,
    opponent: str,
    base_targets: float,
) -> dict[str, object]:
    jitter = (week % 3) - 1
    targets = max(0, int(base_targets + jitter))
    receptions = max(0, int(targets * 0.65))
    rec_yards = float(receptions * 11.0)
    rec_tds = 1 if (week + int(gsis_id[-1])) % 4 == 0 else 0
    # One synthetic TE rushes (Taysom-Hill-shape) when the gsis_id ends in
    # "3"; this gives the new TeFeaturesSchema rushing columns non-zero
    # rolling means.
    is_rushing_te = gsis_id.endswith("3")
    carries = 4 + (week % 2) if is_rushing_te else 0
    rush_yards = float(carries * 4.0)
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": "TE",
        "team": team,
        "opponent": opponent,
        "passing_yards": 0.0,
        "passing_tds": 0,
        "interceptions": 0,
        "attempts": 0,
        "completions": 0,
        "sacks": 0,
        "rushing_yards": rush_yards,
        "rushing_tds": 1 if is_rushing_te and week % 4 == 0 else 0,
        "carries": carries,
        "receptions": receptions,
        "receiving_yards": rec_yards,
        "receiving_tds": rec_tds,
        "receiving_air_yards": float(targets * 12.0),
        "targets": targets,
        "fumbles_lost": 0,
    }


# Callable[..., ...] (vs. a concrete signature) is the explicit escape hatch
# for the dynamic-kwarg dispatch in `_build_position_weekly_stats`: each
# builder takes a different position-specific keyword (`base_attempts` /
# `base_carries` / `base_targets`), and we resolve which one to pass via
# `_POSITION_BASE_KW`. mypy can't validate kwarg-name dispatch, so we narrow
# the type to "any callable returning a row dict".
_POSITION_ROW_BUILDERS: dict[str, Callable[..., dict[str, object]]] = {
    "QB": _qb_weekly_stats_row,
    "RB": _rb_weekly_stats_row,
    "TE": _te_weekly_stats_row,
}

_POSITION_BASE_KW: dict[str, str] = {
    "QB": "base_attempts",
    "RB": "base_carries",
    "TE": "base_targets",
}


def _build_position_weekly_stats(position: str) -> pd.DataFrame:
    """Stack 8 weeks of 2024 + 4 weeks of 2025 for the synthetic universe,
    using the row-builder registered for `position`."""
    builder = _POSITION_ROW_BUILDERS[position]
    base_rates = _POSITION_BASE_RATES[position]
    base_kw = _POSITION_BASE_KW[position]
    rows: list[dict[str, object]] = []
    for season, weeks in [(2024, range(1, 9)), (2025, range(1, 5))]:
        for week in weeks:
            for gsis_id, team, base_rate in zip(_GSIS_IDS, _TEAMS, base_rates, strict=True):
                opponent = "MIN" if team == "KC" else "KC"
                rows.append(
                    builder(
                        gsis_id=gsis_id,
                        season=season,
                        week=week,
                        team=team,
                        opponent=opponent,
                        **{base_kw: base_rate},
                    )
                )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    return df


@pytest.fixture
def baseline_weekly_stats_qb() -> pd.DataFrame:
    """8 weeks of 2024 + 4 weeks of 2025 QB-shaped stats for 5 synthetic QBs."""
    return _build_position_weekly_stats("QB")


@pytest.fixture
def baseline_weekly_stats_rb() -> pd.DataFrame:
    """8 weeks of 2024 + 4 weeks of 2025 RB-shaped stats for 5 synthetic RBs."""
    return _build_position_weekly_stats("RB")


@pytest.fixture
def baseline_weekly_stats_te() -> pd.DataFrame:
    """8 weeks of 2024 + 4 weeks of 2025 TE-shaped stats for 5 synthetic TEs.

    The TE whose gsis_id ends in "3" rushes (Taysom-Hill-shape) so the new
    TeFeaturesSchema rushing columns from Phase 1 carry non-zero rolling
    means.
    """
    return _build_position_weekly_stats("TE")


def _build_position_supporting_frames(
    weekly_stats: pd.DataFrame, position: str
) -> dict[str, pd.DataFrame]:
    """Build snap_counts / depth_charts / ngs (passing or rushing or receiving)
    / schedules sub-frames matching the synthetic universe."""
    base_rates = _POSITION_BASE_RATES[position]

    snap_rows = [
        {
            "gsis_id": r["gsis_id"],
            "season": r["season"],
            "week": r["week"],
            "team": r["team"],
            "opponent": r["opponent"],
            "position": position,
            "offense_snaps": 60,
            "offense_pct": 0.95,
            "defense_snaps": 0,
            "defense_pct": 0.0,
            "st_snaps": 2,
            "st_pct": 0.05,
        }
        for _, r in weekly_stats.iterrows()
    ]
    snap_counts = pd.DataFrame(snap_rows)
    for col in ("gsis_id", "team", "opponent", "position"):
        snap_counts[col] = snap_counts[col].astype(_PYARROW_STR)

    dc_rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            for gsis_id, team, _base in zip(_GSIS_IDS, _TEAMS, base_rates, strict=True):
                team_pool = sorted(
                    [
                        (g, t, b)
                        for g, t, b in zip(_GSIS_IDS, _TEAMS, base_rates, strict=True)
                        if t == team
                    ],
                    key=lambda x: -x[2],
                )
                rank = next(i for i, (g, _, _) in enumerate(team_pool, start=1) if g == gsis_id)
                dc_rows.append(
                    {
                        "gsis_id": gsis_id,
                        "season": season,
                        "week": week,
                        "team": team,
                        "position": position,
                        "depth_team": f"{position}{rank}",
                        "depth_rank": rank,
                    }
                )
    depth = pd.DataFrame(dc_rows)
    for col in ("gsis_id", "team", "position", "depth_team"):
        depth[col] = depth[col].astype(_PYARROW_STR)

    # NGS source depends on position: QB->passing, RB->rushing, TE->receiving.
    ngs_rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            for gsis_id, team, base in zip(_GSIS_IDS, _TEAMS, base_rates, strict=True):
                row: dict[str, object] = {
                    "gsis_id": gsis_id,
                    "season": season,
                    "week": week,
                    "team": team,
                    "position": position,
                }
                if position == "QB":
                    row.update(
                        {
                            "avg_time_to_throw": 2.7 + base * 0.001,
                            "avg_intended_air_yards": 8.0 + base * 0.05,
                            "completion_percentage_above_expectation": -1.0 + base * 0.1,
                            "aggressiveness": 12.0 + base * 0.05,
                        }
                    )
                elif position == "RB":
                    row.update(
                        {
                            "efficiency": 3.0 + base * 0.02,
                            "rush_yards_over_expected_per_att": -0.5 + base * 0.05,
                            "percent_attempts_gte_eight_defenders": 18.0 + base * 0.1,
                        }
                    )
                else:  # TE
                    row.update(
                        {
                            "avg_separation": 2.5 + base * 0.05,
                            "avg_intended_air_yards": 9.0 + base * 0.2,
                            "avg_yac_above_expectation": -0.2 + base * 0.05,
                        }
                    )
                ngs_rows.append(row)
    ngs = pd.DataFrame(ngs_rows)
    for col in ("gsis_id", "team", "position"):
        ngs[col] = ngs[col].astype(_PYARROW_STR)

    sch_rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        weeks = range(1, 9) if season == 2024 else range(1, 5)
        for week in weeks:
            sch_rows.append(
                {
                    "season": season,
                    "week": week,
                    "game_id": f"{season}_{week:02d}_KC_MIN",
                    "home_team": "KC",
                    "away_team": "MIN",
                    "kickoff": pd.Timestamp(f"{season}-09-{week + 1:02d}T17:00:00Z")
                    .tz_convert("UTC")
                    .as_unit("us"),
                    "spread_line": -3.0,
                    "total_line": 47.0,
                    "home_moneyline": -150,
                    "away_moneyline": 130,
                    "surface": "grass",
                    "roof": "outdoors",
                    "temp": 60,
                    "wind": 5,
                }
            )
    schedules = pd.DataFrame(sch_rows)
    for col in ("game_id", "home_team", "away_team", "surface", "roof"):
        schedules[col] = schedules[col].astype(_PYARROW_STR)
    for col in ("temp", "wind", "home_moneyline", "away_moneyline"):
        schedules[col] = schedules[col].astype(pd.Int64Dtype())

    return {
        "snap_counts": snap_counts,
        "depth_charts": depth,
        "ngs": ngs,
        "schedules": schedules,
    }


@pytest.fixture
def baseline_features_qb(baseline_weekly_stats_qb: pd.DataFrame) -> pd.DataFrame:
    """QB feature rows produced by build_qb_features for every (season, week)."""
    from projections.features import build_qb_features

    aux = _build_position_supporting_frames(baseline_weekly_stats_qb, "QB")
    feat_frames: list[pd.DataFrame] = []
    for season, weeks in [(2024, range(1, 9)), (2025, range(1, 5))]:
        for week in weeks:
            f = build_qb_features(
                weekly_stats=baseline_weekly_stats_qb,
                snap_counts=aux["snap_counts"],
                depth_charts=aux["depth_charts"],
                ngs_passing=aux["ngs"],
                schedules=aux["schedules"],
                season=season,
                as_of_week=week,
            )
            if not f.empty:
                feat_frames.append(f)
    return pd.concat(feat_frames, ignore_index=True) if feat_frames else pd.DataFrame()


@pytest.fixture
def baseline_features_rb(baseline_weekly_stats_rb: pd.DataFrame) -> pd.DataFrame:
    """RB feature rows produced by build_rb_features for every (season, week)."""
    from projections.features import build_rb_features

    aux = _build_position_supporting_frames(baseline_weekly_stats_rb, "RB")
    feat_frames: list[pd.DataFrame] = []
    for season, weeks in [(2024, range(1, 9)), (2025, range(1, 5))]:
        for week in weeks:
            f = build_rb_features(
                weekly_stats=baseline_weekly_stats_rb,
                snap_counts=aux["snap_counts"],
                depth_charts=aux["depth_charts"],
                ngs_rushing=aux["ngs"],
                schedules=aux["schedules"],
                season=season,
                as_of_week=week,
            )
            if not f.empty:
                feat_frames.append(f)
    return pd.concat(feat_frames, ignore_index=True) if feat_frames else pd.DataFrame()


@pytest.fixture
def baseline_features_te(baseline_weekly_stats_te: pd.DataFrame) -> pd.DataFrame:
    """TE feature rows produced by build_te_features for every (season, week)."""
    from projections.features import build_te_features

    aux = _build_position_supporting_frames(baseline_weekly_stats_te, "TE")
    feat_frames: list[pd.DataFrame] = []
    for season, weeks in [(2024, range(1, 9)), (2025, range(1, 5))]:
        for week in weeks:
            f = build_te_features(
                weekly_stats=baseline_weekly_stats_te,
                snap_counts=aux["snap_counts"],
                depth_charts=aux["depth_charts"],
                ngs_receiving=aux["ngs"],
                schedules=aux["schedules"],
                season=season,
                as_of_week=week,
            )
            if not f.empty:
                feat_frames.append(f)
    return pd.concat(feat_frames, ignore_index=True) if feat_frames else pd.DataFrame()
