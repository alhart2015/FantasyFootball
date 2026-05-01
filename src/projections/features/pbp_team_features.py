"""PBP-derived team-level features for the PBP family probe.

Pure-pandas computes consumed by build_pbp_family_overrides (this module's
public assembler). Each compute returns a (team, season, week, <metric>_l4)
frame with one row per (team, season, week) where the team has a scheduled
game, computed as the rolling mean over the trailing 4 prior games (min 4).

The trailing-4 backfill across season boundaries is handled by the caller
feeding multiple seasons of PBP concat'd together — see
scripts/build_pbp_family_override.py.

Spec: docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md §6.1.
"""

from __future__ import annotations

import pandas as pd

_OFFENSIVE_PLAY_TYPES: frozenset[str] = frozenset({"pass", "run"})


def _trailing_4_mean(per_game: pd.DataFrame, *, value_col: str, out_col: str) -> pd.DataFrame:
    """Rolling-4 mean of value_col per team, shifted so row at week W
    reflects the mean over W-4..W-1 (NOT W). min_periods=4 → fewer than 4
    prior games yield NaN. Input must have columns (team, season, week,
    value_col); output is (team, season, week, out_col).
    """
    sorted_df = per_game.sort_values(["team", "season", "week"]).reset_index(drop=True)
    rolled = (
        sorted_df.groupby("team", sort=False)[value_col]
        .rolling(window=4, min_periods=4)
        .mean()
        .shift(1)  # row at index N reflects rolling mean of N-4..N-1
        .reset_index(level=0, drop=True)
    )
    sorted_df[out_col] = rolled
    return sorted_df[["team", "season", "week", out_col]]


def compute_team_pace(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level offensive plays per game, trailing 4 prior games.

    Plays counted: rows where ``play_type in {'pass', 'run'}``. Excludes
    kickoff / punt / field_goal / no_play. No neutral-script filter — the
    curated PbpSchema lacks ``wp`` / ``qtr`` / ``score_differential``.
    """
    offensive = pbp[pbp["play_type"].isin(_OFFENSIVE_PLAY_TYPES)]
    per_game = (
        offensive.groupby(["posteam", "season", "week"], as_index=False)
        .size()
        .rename(columns={"posteam": "team", "size": "plays"})
    )
    return _trailing_4_mean(per_game, value_col="plays", out_col="pace_l4")
