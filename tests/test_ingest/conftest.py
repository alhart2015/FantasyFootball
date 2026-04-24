"""Shared ingest test fixtures — fake `nfl_data_py` responses."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def fake_id_map_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_ids()` — a row per player with cross-platform IDs."""
    return pd.DataFrame(
        {
            "gsis_id": ["00-0036322", "00-0034857", "00-0034796"],
            "espn_id": ["4262921", "3915511", "4035687"],
            "sleeper_id": ["6794", "5849", "5045"],
            "pfr_id": ["JeffJu00", "MahoPa00", "BarkSa00"],
            "name": ["Justin Jefferson", "Patrick Mahomes", "Saquon Barkley"],
            "position": ["WR", "QB", "RB"],
            "team": ["MIN", "KC", "PHI"],
        }
    )


@pytest.fixture
def fake_weekly_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_weekly_data([2024])` — 2 player-weeks."""
    return pd.DataFrame(
        {
            "player_id": ["00-0036322", "00-0034857"],
            "season": [2024, 2024],
            "week": [3, 3],
            "position": ["WR", "QB"],
            "recent_team": ["MIN", "KC"],
            "opponent_team": ["HOU", "ATL"],
            "passing_yards": [0.0, 286.0],
            "passing_tds": [0, 2],
            "interceptions": [0, 1],
            "rushing_yards": [0.0, 12.0],
            "rushing_tds": [0, 0],
            "carries": [0, 3],
            "receptions": [9, 0],
            "receiving_yards": [110.0, 0.0],
            "receiving_tds": [1, 0],
            "receiving_air_yards": [145.0, 0.0],
            "targets": [12, 0],
            "fumbles_lost": [0, 0],
        }
    )


@pytest.fixture
def fake_schedules_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_schedules([2024])` — 2 games for week 3.

    Raw column names (before our _RENAME): nfl_data_py uses `gameday` (date)
    and `gametime` (HH:MM string) for kickoff; the ingest module combines them
    into a UTC `kickoff` timestamp.
    """
    return pd.DataFrame(
        {
            "game_id": ["2024_03_KC_ATL", "2024_03_MIN_HOU"],
            "season": [2024, 2024],
            "week": [3, 3],
            "home_team": ["ATL", "HOU"],
            "away_team": ["KC", "MIN"],
            "gameday": ["2024-09-22", "2024-09-22"],
            "gametime": ["20:20", "13:00"],
            "spread_line": [3.5, -2.5],
            "total_line": [48.5, 44.0],
            "home_moneyline": [155, -125],
            "away_moneyline": [-180, 105],
            "surface": ["fieldturf", "matrixturf"],
            "roof": ["dome", "dome"],
            "temp": [72, 72],
            "wind": [0, 0],
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
            "game_id": ["2024_03_KC_ATL", "2024_03_MIN_HOU"],
            "season": [2024, 2024],
            "week": [3, 3],
            "player": ["Patrick Mahomes", "Justin Jefferson"],
            "position": ["QB", "WR"],
            "team": ["KC", "MIN"],
            "opponent": ["ATL", "HOU"],
            "offense_snaps": [71, 62],
            "offense_pct": [1.0, 0.95],
            "defense_snaps": [0, 0],
            "defense_pct": [0.0, 0.0],
            "st_snaps": [0, 3],
            "st_pct": [0.0, 0.10],
            "pfr_player_id": ["MahoPa00", "JeffJu00"],
        }
    )


@pytest.fixture
def fake_depth_charts_df() -> pd.DataFrame:
    """Mimics `nfl_data_py.import_depth_charts([2024])` — 3 player-weeks.

    Raw column names: `club_code` is the team (renamed to `team`); `depth_team`
    is the raw slot label (e.g., 'WR1', 'LWR'); `depth_position` may already
    be a numeric rank — the ingest module prefers `depth_position` if present
    and otherwise parses the trailing digit from `depth_team`.
    """
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "club_code": ["MIN", "KC", "PHI"],
            "week": [3, 3, 3],
            "depth_team": ["WR1", "QB1", "RB1"],
            "last_name": ["Jefferson", "Mahomes", "Barkley"],
            "first_name": ["Justin", "Patrick", "Saquon"],
            "formation": ["Offense", "Offense", "Offense"],
            "gsis_id": ["00-0036322", "00-0034857", "00-0034796"],
            "jersey_number": [18, 15, 26],
            "position": ["WR", "QB", "RB"],
            "elias_id": ["JEF845899", "MAH335103", "BAR123456"],
            "depth_position": [1, 1, 1],
            "football_name": ["Justin Jefferson", "Patrick Mahomes", "Saquon Barkley"],
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
