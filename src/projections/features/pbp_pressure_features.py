"""PBP-derived team-level pressure features for the PBP pressure family probe.

Pure-pandas computes consumed by build_pbp_pressure_overrides (this module's
public assembler). Each compute returns a (team, season, week, <metric>_l4)
frame with one row per (team, season, week) where the team has a scheduled
game, computed as the rolling mean over the trailing 4 prior games (min 4).

Pressure events are denominated on qb_dropback == 1 (the canonical
nflfastR pressure-event denominator: pass attempts + sacks + scrambles).
Plays with qb_dropback == 0 (handoffs, kneels, spikes) and NaN qb_dropback
rows are excluded from both numerator and denominator.

The trailing-4 backfill across season boundaries is handled by the caller
feeding multiple seasons of PBP concat'd together — see
scripts/build_pbp_pressure_override.py.

Spec: docs/superpowers/specs/2026-05-02-pbp-pressure-feature-family-probe-design.md §6.1.
"""

from __future__ import annotations

import re
from typing import Final

import pandas as pd

from projections.schemas import GSIS_ID_PATTERN

_GSIS_RE: Final[re.Pattern[str]] = re.compile(rf"^{GSIS_ID_PATTERN}$")
_PBP_COLUMNS_USED: Final[tuple[str, ...]] = (
    "posteam",
    "defteam",
    "season",
    "week",
    "qb_dropback",
    "qb_scramble",
    "sack",
)


def _trailing_4_mean(per_game: pd.DataFrame, *, value_col: str, out_col: str) -> pd.DataFrame:
    """Rolling-4 mean of value_col per team, shifted so row at week W
    reflects the mean over W-4..W-1 (NOT W). min_periods=4 → fewer than 4
    prior games yield NaN.

    Both the rolling AND the shift are within-team (via groupby+transform);
    a global .shift(1) would leak the last row of one team into the first
    row of the next.
    """
    sorted_df = per_game.sort_values(["team", "season", "week"]).reset_index(drop=True)
    rolled = sorted_df.groupby("team", sort=False)[value_col].transform(
        lambda s: s.rolling(window=4, min_periods=4).mean().shift(1)
    )
    sorted_df[out_col] = rolled
    return sorted_df[["team", "season", "week", out_col]]


def _per_game_rate(
    pbp: pd.DataFrame, *, team_key: str, num_col: str, denom_col: str, out_col: str
) -> pd.DataFrame:
    """Compute per-game rate = sum(num_col) / sum(denom_col) where
    denom_col == 1, grouped by (team_key, season, week). Excludes NaN
    denom_col rows. Output: (team, season, week, out_col)."""
    valid = pbp[(pbp[denom_col] == 1.0) & pbp[denom_col].notna()]
    per_game = valid.groupby([team_key, "season", "week"], as_index=False).agg(
        num=(num_col, "sum"), denom=(denom_col, "sum")
    )
    per_game[out_col] = per_game["num"] / per_game["denom"]
    return per_game.rename(columns={team_key: "team"})[["team", "season", "week", out_col]]


def compute_team_sack_rate_allowed(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level offensive sack rate allowed, trailing 4 prior games.

    Per (posteam, season, week): sum(sack) / sum(qb_dropback) over rows
    where posteam == team AND qb_dropback == 1. Plays with qb_dropback == 0
    or NaN are excluded from both numerator and denominator.
    """
    per_game = _per_game_rate(
        pbp,
        team_key="posteam",
        num_col="sack",
        denom_col="qb_dropback",
        out_col="sack_rate",
    )
    return _trailing_4_mean(per_game, value_col="sack_rate", out_col="team_sack_rate_allowed_l4")


def compute_team_qb_scramble_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level offensive QB scramble rate, trailing 4 prior games.

    Per (posteam, season, week): sum(qb_scramble) / sum(qb_dropback) over
    rows where posteam == team AND qb_dropback == 1. Plays with
    qb_dropback == 0 or NaN are excluded from both numerator and denominator.

    Output: (team, season, week, team_qb_scramble_rate_l4)
    """
    per_game = _per_game_rate(
        pbp,
        team_key="posteam",
        num_col="qb_scramble",
        denom_col="qb_dropback",
        out_col="scramble_rate",
    )
    return _trailing_4_mean(per_game, value_col="scramble_rate", out_col="team_qb_scramble_rate_l4")


def compute_team_def_sack_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Defensive sack rate forced, trailing 4 prior games.

    Per (defteam, season, week): sum(sack) / sum(qb_dropback) over rows
    where defteam == team AND qb_dropback == 1. Plays with qb_dropback == 0
    or NaN are excluded from both numerator and denominator.

    Output: (team, season, week, team_def_sack_rate_l4) where ``team`` is
    the DEFENSE's team code; the joiner attaches each player's *opponent's*
    row.
    """
    per_game = _per_game_rate(
        pbp,
        team_key="defteam",
        num_col="sack",
        denom_col="qb_dropback",
        out_col="def_sack_rate",
    )
    return _trailing_4_mean(per_game, value_col="def_sack_rate", out_col="team_def_sack_rate_l4")


def compute_team_def_scramble_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Defensive scramble rate forced, trailing 4 prior games.

    Per (defteam, season, week): sum(qb_scramble) / sum(qb_dropback) over
    rows where defteam == team AND qb_dropback == 1. Plays with
    qb_dropback == 0 or NaN are excluded from both numerator and denominator.

    Output: (team, season, week, team_def_scramble_rate_l4) where ``team``
    is the DEFENSE's team code.
    """
    per_game = _per_game_rate(
        pbp,
        team_key="defteam",
        num_col="qb_scramble",
        denom_col="qb_dropback",
        out_col="def_scramble_rate",
    )
    return _trailing_4_mean(
        per_game, value_col="def_scramble_rate", out_col="team_def_scramble_rate_l4"
    )
