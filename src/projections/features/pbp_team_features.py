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

import re
from typing import Final

import pandas as pd

from projections.schemas import GSIS_ID_PATTERN

_OFFENSIVE_PLAY_TYPES: Final[frozenset[str]] = frozenset({"pass", "run"})
_GSIS_RE: Final[re.Pattern[str]] = re.compile(rf"^{GSIS_ID_PATTERN}$")
_PBP_COLUMNS_USED: Final[tuple[str, ...]] = (
    "posteam",
    "defteam",
    "season",
    "week",
    "play_type",
    "pass_oe",
    "pass_attempt",
    "air_yards",
    "epa",
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


def compute_team_proe(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level pass rate over expected, trailing 4 prior games.

    Mean of ``pass_oe`` (nflfastR's pass-over-expected, percentage points)
    across rows where ``posteam == team`` and ``pass_oe`` is non-NaN.
    Upstream's xpass model already game-state-controls the per-play
    pass_oe value, so the per-play mean is itself a properly-controlled
    PROE — no further bucketing required here.
    """
    plays = pbp[pbp["pass_oe"].notna()]
    per_game = (
        plays.groupby(["posteam", "season", "week"], as_index=False)["pass_oe"]
        .mean()
        .rename(columns={"posteam": "team", "pass_oe": "proe"})
    )
    return _trailing_4_mean(per_game, value_col="proe", out_col="proe_l4")


def compute_team_ayps(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-level mean air yards per pass attempt, trailing 4 prior games.

    Plays counted: rows where ``posteam == team``, ``pass_attempt == 1.0``,
    and ``air_yards`` is non-NaN. Sacks and throw-aways have NaN air_yards
    upstream and are excluded from the per-game mean.
    """
    plays = pbp[(pbp["pass_attempt"] == 1.0) & (pbp["air_yards"].notna())]
    per_game = (
        plays.groupby(["posteam", "season", "week"], as_index=False)["air_yards"]
        .mean()
        .rename(columns={"posteam": "team", "air_yards": "ayps"})
    )
    return _trailing_4_mean(per_game, value_col="ayps", out_col="team_ayps_l4")


def compute_team_def_epa_residual(pbp: pd.DataFrame) -> pd.DataFrame:
    """Defensive EPA-allowed-per-play residual vs offensive-opponent
    season-average EPA, trailing 4 prior games.

    Per (defteam, season, week): mean of ``epa`` across rows where
    ``defteam == team`` and ``epa`` is non-NaN. Per (posteam, season):
    season-average mean of ``epa`` on offense (the opponent's strength
    signal). Per-game residual = (mean def-allowed EPA) - (offensive
    opponent's season-average EPA-on-offense). Then rolling-4 mean of
    the residual series per team, shifted so row at week W reflects the
    last 4 prior games.

    Plain (non-regression) residual: subtracting the opp's season-avg
    EPA from each game's def-allowed EPA gives the "above-or-below
    expected" residual for that game. Same shape as Plan 9's per-position
    EPA-residual but pooled across all plays.

    Output schema: (team, season, week, team_def_epa_resid_l4) where
    ``team`` is the DEFENSE's team code; the joiner attaches each
    player's *opponent's* row.
    """
    epa_plays = pbp[pbp["epa"].notna()]

    # Per (defteam, season, week): mean EPA allowed; first opponent
    # (one opponent per defense per week in real NFL).
    def_per_game = (
        epa_plays.groupby(["defteam", "season", "week"], as_index=False)
        .agg(def_epa_mean=("epa", "mean"), opp=("posteam", "first"))
        .rename(columns={"defteam": "team"})
    )

    # Per (posteam, season): full-season average offensive EPA. The
    # opponent's "strength expectation" for any game.
    off_season_avg = (
        epa_plays.groupby(["posteam", "season"], as_index=False)["epa"]
        .mean()
        .rename(columns={"posteam": "opp", "epa": "opp_season_off_epa"})
    )

    merged = def_per_game.merge(off_season_avg, on=["opp", "season"], how="left")
    merged["resid"] = merged["def_epa_mean"] - merged["opp_season_off_epa"]

    return _trailing_4_mean(
        merged[["team", "season", "week", "resid"]],
        value_col="resid",
        out_col="team_def_epa_resid_l4",
    )


def build_pbp_family_overrides(
    pbp: pd.DataFrame,
    player_team_week_index: pd.DataFrame,
) -> pd.DataFrame:
    """Public assembler. Returns the 4-column override frame ready to write.

    Args:
        pbp: PBP frame matching ``PbpSchema``. Must include the seasons
            spanning the index plus one prior season for trailing-4 backfill.
            Team codes (``posteam``, ``defteam``) are assumed canonical per
            ``_TEAM_VALUES`` — schema validation at ingest is the contract.
        player_team_week_index: ``(gsis_id, season, week, team, opp)`` —
            one row per player-week. Team codes are assumed canonical per the
            ingest schemas (DepthChartsSchema / SchedulesSchema both validate
            against ``_TEAM_VALUES``).

    Returns:
        ``(gsis_id, season, week, pace_l4, proe_l4, team_ayps_l4,
        team_def_epa_resid_l4)`` — one row per input index row.

    Raises:
        ValueError: gsis_id format violations or duplicate
            (gsis_id, season, week) keys in the index.
        AssertionError: row-count mismatch after merges (internal-invariant
            violation; a future compute regression that introduces duplicate
            (team, season, week) keys would trigger this).

    Per-position coverage validation is the probe's responsibility (the
    assembler has no access to the per-position feature parquets); see
    spec §1.3 criterion 1 + §3.3 step 2.
    """
    bad_ids = [g for g in player_team_week_index["gsis_id"].dropna() if not _GSIS_RE.match(str(g))]
    if bad_ids:
        raise ValueError(
            f"invalid gsis_id format(s): {bad_ids[:3]} (and {max(0, len(bad_ids) - 3)} more)"
        )

    dup_mask = player_team_week_index.duplicated(subset=["gsis_id", "season", "week"], keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        raise ValueError(f"duplicate (gsis_id, season, week) keys in index: {n_dup} rows")

    pbp_proj = pbp[list(_PBP_COLUMNS_USED)]
    pace = compute_team_pace(pbp_proj)
    proe = compute_team_proe(pbp_proj)
    ayps = compute_team_ayps(pbp_proj)
    def_resid = compute_team_def_epa_residual(pbp_proj)

    out = player_team_week_index.merge(pace, on=["team", "season", "week"], how="left")
    out = out.merge(proe, on=["team", "season", "week"], how="left")
    out = out.merge(ayps, on=["team", "season", "week"], how="left")
    out = out.merge(
        def_resid.rename(columns={"team": "opp"}),
        on=["opp", "season", "week"],
        how="left",
    )

    if len(out) != len(player_team_week_index):
        raise AssertionError(
            f"row count mismatch: input index had {len(player_team_week_index)} rows, "
            f"output has {len(out)}; suggests a many-to-many merge regression"
        )

    return out[
        [
            "gsis_id",
            "season",
            "week",
            "pace_l4",
            "proe_l4",
            "team_ayps_l4",
            "team_def_epa_resid_l4",
        ]
    ].reset_index(drop=True)
