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
            "receptions": [9, 0],
            "receiving_yards": [110.0, 0.0],
            "receiving_tds": [1, 0],
            "fumbles_lost": [0, 0],
        }
    )
