"""Weather feature computes for the weather family probe.

Sourced from `SchedulesSchema` columns (`wind`, `temp`, `roof`, `surface`)
already in `data/raw/schedules`. Dome / closed-roof games are filled per
spec §3.5: a controlled environment has no weather, so wind=0 / temp=70
is semantically correct, not "imputed missing."

Probe-only — features land in the override parquet, not in
`*FeaturesSchema`. Integration follow-up is conditional on the family-probe
verdict per `docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md`.
"""

from __future__ import annotations

import pandas as pd

_HIGH_WIND_MPH = 20.0
_DOME_FILL_TEMP_F = 70.0
_DOME_FILL_WIND_MPH = 0.0


def compute_weather_features(schedules: pd.DataFrame) -> pd.DataFrame:
    """Per-team-game frame with four weather features.

    One row per (game, team) — each schedules row produces two output rows
    (home + away). Weather is a game-level attribute, so both teams in a
    matchup carry identical wind/temp/surface values.

    Dome / closed-roof handling (spec §3.5):
        wind_speed_mph = 0.0
        temperature_f = 70.0
        is_high_wind = 0.0 (falls out of wind_speed_mph = 0.0)
        is_grass_surface = surface == 'grass' (no override)

    Outdoor handling: NaN wind / temp propagate; is_high_wind preserves NaN.

    Args:
        schedules: frame validated against `SchedulesSchema` (must carry
            season, week, home_team, away_team, wind, temp, roof, surface).

    Returns:
        DataFrame with columns:
            season, week, team,
            wind_speed_mph, is_high_wind, temperature_f, is_grass_surface
        All four feature columns are nullable Float64. season / week are
        Int64; team is StringDtype("pyarrow") (inherited from inputs).
    """
    cols = ["season", "week", "wind", "temp", "roof", "surface"]
    home = schedules[[*cols, "home_team"]].rename(columns={"home_team": "team"}).copy()
    away = schedules[[*cols, "away_team"]].rename(columns={"away_team": "team"}).copy()
    games = pd.concat([home, away], ignore_index=True)

    # Dome / closed-roof predicate matches `_shared.build_game_environment`'s
    # logic exactly: any roof not in {dome, closed} (including NaN) is treated
    # as outdoor.
    is_indoor = games["roof"].isin(["dome", "closed"]).fillna(False)

    wind_f = games["wind"].astype("Float64")
    games["wind_speed_mph"] = wind_f.where(~is_indoor, _DOME_FILL_WIND_MPH)

    temp_f = games["temp"].astype("Float64")
    games["temperature_f"] = temp_f.where(~is_indoor, _DOME_FILL_TEMP_F)

    # NaN-preserving threshold (spec §3.2). pandas Float64 + NA propagates
    # the comparison: (NA >= 20.0) -> NA -> NaN in resulting Float64.
    wind_speed = games["wind_speed_mph"]
    games["is_high_wind"] = (wind_speed >= _HIGH_WIND_MPH).astype("Float64")

    games["is_grass_surface"] = (games["surface"] == "grass").fillna(False).astype("Float64")

    return games[
        [
            "season",
            "week",
            "team",
            "wind_speed_mph",
            "is_high_wind",
            "temperature_f",
            "is_grass_surface",
        ]
    ].reset_index(drop=True)


def attach_weather_features(
    index: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge the four weather features onto a player-team-week index.

    Args:
        index: frame with at least (season, week, team) columns. Typically
            the player-team-week index from
            `scripts.build_weather_override._build_player_team_week_index`,
            carrying (gsis_id, season, week, team, opp, position).
        schedules: frame validated against `SchedulesSchema`.

    Returns:
        Copy of index with four nullable Float64 cols appended:
        wind_speed_mph, is_high_wind, temperature_f, is_grass_surface.
        Index rows without a matching (season, week, team) in schedules
        retain NaN in all four cols.
    """
    weather = compute_weather_features(schedules)
    return index.merge(weather, on=["season", "week", "team"], how="left")


_REQUIRED_INDEX_COLS = ("gsis_id", "season", "week", "team", "opp", "position")


def build_weather_overrides(
    schedules: pd.DataFrame,
    player_team_week_index: pd.DataFrame,
) -> pd.DataFrame:
    """Build the weather override frame from a schedules table + a
    player-team-week index.

    Args:
        schedules: validated against `SchedulesSchema`.
        player_team_week_index: frame from `_build_player_team_week_index`
            with columns (gsis_id, season, week, team, opp, position).
            Must have unique (gsis_id, season, week) keys.

    Returns:
        Frame with columns
            (gsis_id, season, week, position,
             wind_speed_mph, is_high_wind, temperature_f, is_grass_surface)
        — one row per index input row. Designed to feed
        `scripts.probe_feature_signal --override`.

    Raises:
        ValueError: index missing a required column or carrying duplicate
            (gsis_id, season, week) keys.
    """
    missing = [c for c in _REQUIRED_INDEX_COLS if c not in player_team_week_index.columns]
    if missing:
        raise ValueError(f"player_team_week_index missing required column(s): {missing}")

    key_cols = ["gsis_id", "season", "week"]
    dups = player_team_week_index.duplicated(subset=key_cols)
    if dups.any():
        n = int(dups.sum())
        raise ValueError(f"player_team_week_index has {n} duplicate (gsis_id, season, week) keys")

    attached = attach_weather_features(player_team_week_index, schedules)
    return attached[
        [
            "gsis_id",
            "season",
            "week",
            "position",
            "wind_speed_mph",
            "is_high_wind",
            "temperature_f",
            "is_grass_surface",
        ]
    ].reset_index(drop=True)
