"""Shared NFL regular-season calendar helpers.

The regular season is 16 games (17 calendar weeks incl. the bye) through 2020,
and 17 games (18 weeks) from 2021 on. Ingested `weekly_stats`/`schedules`
number playoff weeks above this (up to 22), so `last_regular_week` is the
"regular season only" cutoff on either. Centralized here per TODO #41.
"""

from __future__ import annotations


def regular_season_games(season: int) -> int:
    """Number of regular-season games: 16 through 2020, 17 from 2021 on."""
    return 16 if season <= 2020 else 17


def last_regular_week(season: int) -> int:
    """Last regular-season calendar week (games + the one bye)."""
    return regular_season_games(season) + 1
