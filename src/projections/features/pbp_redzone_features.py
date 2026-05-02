"""PBP-derived team-level red-zone features for the PBP red-zone family probe.

Pure-pandas computes consumed by build_pbp_redzone_overrides (this module's
public assembler). Each compute returns a (team, season, week, <metric>_l4)
frame with one row per (team, season, week) where the team has a scheduled
game, computed as the rolling mean over the trailing 4 prior games (min 4).

Red zone defined as yardline_100 <= 20 (NFL standard).

The trailing-4 backfill across season boundaries is handled by the caller
feeding multiple seasons of PBP concat'd together — see
scripts/build_pbp_redzone_override.py.

Spec: docs/superpowers/specs/2026-05-02-pbp-redzone-feature-family-probe-design.md §6.1.
"""

from __future__ import annotations

import re
from typing import Final

import pandas as pd

from projections.schemas import GSIS_ID_PATTERN

_OFFENSIVE_PLAY_TYPES: Final[frozenset[str]] = frozenset({"pass", "run"})
_RZ_THRESHOLD: Final[float] = 20.0
_GSIS_RE: Final[re.Pattern[str]] = re.compile(rf"^{GSIS_ID_PATTERN}$")
_PBP_COLUMNS_USED: Final[tuple[str, ...]] = (
    "posteam",
    "defteam",
    "season",
    "week",
    "play_type",
    "pass_attempt",
    "epa",
    "yardline_100",
)


def _trailing_4_mean(per_game: pd.DataFrame, *, value_col: str, out_col: str) -> pd.DataFrame:
    """Rolling-4 mean of value_col per team, shifted so row at week W
    reflects the mean over W-4..W-1 (NOT W). min_periods=4 → fewer than 4
    prior games yield NaN. Input must have columns (team, season, week,
    value_col); output is (team, season, week, out_col).

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


def compute_team_rz_pace(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level offensive RZ plays per game, trailing 4 prior games.

    Plays counted: rows where ``play_type in {'pass', 'run'}`` AND
    ``yardline_100 <= 20``. Excludes kickoffs, punts, FGs, no-plays.
    No neutral-script filter — the curated PbpSchema lacks ``wp`` /
    ``qtr`` / ``score_differential``.
    """
    rz = pbp[pbp["play_type"].isin(_OFFENSIVE_PLAY_TYPES) & (pbp["yardline_100"] <= _RZ_THRESHOLD)]
    per_game = (
        rz.groupby(["posteam", "season", "week"], as_index=False)
        .size()
        .rename(columns={"posteam": "team", "size": "rz_plays"})
    )
    return _trailing_4_mean(per_game, value_col="rz_plays", out_col="team_rz_pace_l4")


def compute_team_rz_pass_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level RZ pass rate, trailing 4 prior games.

    Mean of ``pass_attempt`` (1.0/0.0) over rows where
    ``play_type in {'pass', 'run'}`` AND ``yardline_100 <= 20``. The
    play_type filter excludes special-teams plays where ``pass_attempt``
    is undefined / NaN.
    """
    rz = pbp[pbp["play_type"].isin(_OFFENSIVE_PLAY_TYPES) & (pbp["yardline_100"] <= _RZ_THRESHOLD)]
    per_game = (
        rz.groupby(["posteam", "season", "week"], as_index=False)["pass_attempt"]
        .mean()
        .rename(columns={"posteam": "team", "pass_attempt": "rz_pass_rate"})
    )
    return _trailing_4_mean(per_game, value_col="rz_pass_rate", out_col="team_rz_pass_rate_l4")


def compute_team_def_rz_epa_allowed(pbp: pd.DataFrame) -> pd.DataFrame:
    """Defensive RZ EPA-per-play allowed, trailing 4 prior games.

    Per (defteam, season, week): mean of ``epa`` across rows where
    ``epa`` is non-NaN AND ``yardline_100 <= 20`` AND
    ``play_type in {'pass', 'run'}``. Special-teams plays are excluded
    (they have non-NaN EPA in nflfastR but are not pass/run scrimmage
    plays). Then rolling-4 mean of the per-game series per team,
    shifted so row at week W reflects the last 4 prior games.

    Output: (team, season, week, team_def_rz_epa_allowed_l4) where
    ``team`` is the DEFENSE's team code; the joiner attaches each
    player's *opponent's* row.
    """
    rz_epa = pbp[
        pbp["play_type"].isin(_OFFENSIVE_PLAY_TYPES)
        & (pbp["yardline_100"] <= _RZ_THRESHOLD)
        & pbp["epa"].notna()
    ]
    per_game = (
        rz_epa.groupby(["defteam", "season", "week"], as_index=False)["epa"]
        .mean()
        .rename(columns={"defteam": "team", "epa": "def_rz_epa"})
    )
    return _trailing_4_mean(per_game, value_col="def_rz_epa", out_col="team_def_rz_epa_allowed_l4")
