"""Helpers shared by every per-position feature builder.

These were previously private to ``features.wr`` and imported across modules
(qb / rb / te all reached into wr's underscore-prefixed names). Living here
makes that intent explicit and prevents wr from becoming a de-facto utility
module.

All functions are pure — no I/O, no caching, no mutation of inputs."""

from __future__ import annotations

import pandas as pd


def prior_mask(df: pd.DataFrame, *, season: int, as_of_week: int) -> pd.Series:
    """Boolean mask selecting rows STRICTLY before week ``as_of_week`` of
    ``season`` — i.e., every prior season + the in-season weeks earlier than
    ``as_of_week``. Used to filter all stat-bearing inputs to leakage-safe
    rows before any rolling computation."""
    return (df["season"] < season) | ((df["season"] == season) & (df["week"] < as_of_week))


def exact_week_mask(df: pd.DataFrame, *, season: int, as_of_week: int) -> pd.Series:
    """Boolean mask selecting only ``(season, as_of_week)`` rows. Used for
    inputs that describe the target week itself: the depth chart that drives
    the row set, and the schedule that supplies game-environment features."""
    return (df["season"] == season) & (df["week"] == as_of_week)


def build_game_environment(schedules: pd.DataFrame) -> pd.DataFrame:
    """Per-team game-environment row from a per-game schedules frame.

    Output columns (one row per team-game, two per game):
        season, week, team, opp_team, is_home, spread, implied_team_total,
        roof_dome.

    Sign convention: empirically verified against
    ``nfl_data_py.import_schedules([2023])``, ``spread_line`` is positive when
    the HOME team is favored and negative when the AWAY team is favored. This
    INVERTS the standard sportsbook convention. Concretely:

        spread_line = +13.0 (KC home vs CHI, home_moneyline=-750) -> KC favored
        spread_line = -11.5 (ARI home vs DAL, home_moneyline=+470) -> ARI dog

    Implied team totals follow directly from spread_line:

        home_implied = (total_line + spread_line) / 2
        away_implied = (total_line - spread_line) / 2

    The per-team ``spread`` column we expose downstream is the team's own signed
    spread in the standard convention (favorite negative, dog positive):

        home_spread = -spread_line   # home favored -> negative
        away_spread = +spread_line   # away favored -> negative (since spread_line is negative)
    """
    home = schedules[
        ["season", "week", "home_team", "away_team", "spread_line", "total_line", "roof"]
    ].rename(columns={"home_team": "team", "away_team": "opp_team"})
    home["is_home"] = True
    home["spread"] = -home["spread_line"].astype(float)
    home["implied_team_total"] = (
        home["total_line"].astype(float) + home["spread_line"].astype(float)
    ) / 2.0

    away = schedules[
        ["season", "week", "home_team", "away_team", "spread_line", "total_line", "roof"]
    ].rename(columns={"away_team": "team", "home_team": "opp_team"})
    away["is_home"] = False
    away["spread"] = away["spread_line"].astype(float)
    away["implied_team_total"] = (
        away["total_line"].astype(float) - away["spread_line"].astype(float)
    ) / 2.0

    game_env = pd.concat([home, away], ignore_index=True)
    game_env["roof_dome"] = game_env["roof"].isin(["dome", "closed"]).fillna(False).astype(bool)
    return game_env[
        [
            "season",
            "week",
            "team",
            "opp_team",
            "is_home",
            "spread",
            "implied_team_total",
            "roof_dome",
        ]
    ]
