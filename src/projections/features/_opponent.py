"""Opponent-strength helper: schedule-of-strength-adjusted EPA-per-play
residual, computed from play-by-play data.

Replaces the v1 `opp_allowed_fppg` (Plan 2a) which used team-week fppg
trailing means without schedule-of-strength adjustment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from projections.schemas import Position, Ruleset


def _is_pass_play(df: pd.DataFrame) -> pd.Series[bool]:
    """A play is pass-classified if play_type=pass OR a sack OR a scramble."""
    return (
        (df["play_type"] == "pass")
        | (df["sack"].fillna(0).astype(int) == 1)
        | (df["qb_scramble"].fillna(0).astype(int) == 1)
    )


def _is_run_play(df: pd.DataFrame) -> pd.Series[bool]:
    """A play is run-classified if play_type=run AND not a scramble."""
    return (df["play_type"] == "run") & (df["qb_scramble"].fillna(0).astype(int) != 1)


def opp_epa_allowed_residual(
    pbp: pd.DataFrame,
    *,
    play_type: Literal["pass", "run"],
    n_weeks: int,
) -> pd.DataFrame:
    """Schedule-of-strength-adjusted EPA-allowed per play type.

    Per-play residual = EPA(p) - mean_EPA_for(posteam, play_type, in_window),
    where mean_EPA_for is that offense's overall pass/run EPA-per-play in the
    same trailing window. The residual answers: "given who they faced, how
    much better/worse than expected did this defense play?"

    Aggregation follows spec §5.1 in two stages:

    1. Group per-play residuals by (defteam, season, week) and take the
       mean. This yields a single per-week mean residual per defense and
       prevents weeks with abnormally many plays (e.g. a high-pace game)
       from dominating the trailing-window estimate.
    2. Take the mean of those per-week means across the trailing window.
       This is the value emitted as ``opp_epa_allowed_residual``.

    A one-stage per-play mean across the entire window would weight weeks
    by their play count, which the spec deliberately avoids.

    Returns one row per (season, target_week, opp_team) with target_week
    shifted +1 from the trailing window's last week, mirroring the v1
    `opp_allowed_fppg` join interface. The opp_team column carries the
    defense; join onto offense-side feature rows on (season, week, opponent).

    Target weeks emitted per (defteam, season) span ``range(2, span_end)``
    where ``span_end = max(max_observed_week, n_weeks) + 2`` — i.e. weeks
    2 through whichever of "the last week with any observation" or "the
    natural lookback target ``n_weeks + 1``" is larger. This handles both
    early-season expanding-window emissions and bye weeks where the defense
    hasn't played in the immediate prior week but still needs a residual
    for prediction.
    """
    empty_out = pd.DataFrame(
        columns=["season", "week", "opp_team", "opp_epa_allowed_residual"]
    ).astype({"season": "int64", "week": "int64", "opp_epa_allowed_residual": float})

    if pbp.empty:
        return empty_out

    # Broad filter: drop pre-snap penalties / bad rows. Used to identify
    # which (defteam, season) pairs exist in the dataset.
    broad = pbp[
        pbp["epa"].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
        & (pbp["play_type"] != "no_play")
    ].copy()

    if broad.empty:
        return empty_out

    # Play-type-filtered subset: residuals are computed *only* over plays of
    # the requested type. The window's mean residual is what we emit.
    if play_type == "pass":
        typed = broad[_is_pass_play(broad)].copy()
    else:
        typed = broad[_is_run_play(broad)].copy()

    if typed.empty:
        return empty_out

    rows: list[dict[str, object]] = []
    for season, season_broad in broad.groupby("season", sort=False):
        max_week = int(season_broad["week"].max())
        # Target weeks span 2 through whichever of `max_week + 1` or
        # `n_weeks + 1` is larger. The latter ensures the natural
        # "predict-week-(n+1)" target is always emitted even if the dataset
        # ends earlier than that.
        span_end = max(max_week, n_weeks) + 2  # exclusive upper bound for range
        target_weeks = list(range(2, span_end))

        defteams_in_season = sorted(season_broad["defteam"].unique())
        season_typed = typed[typed["season"] == season]

        for defteam in defteams_in_season:
            for target_week in target_weeks:
                window_min = max(1, target_week - n_weeks)
                window_max = target_week - 1
                window_weeks = list(range(window_min, window_max + 1))

                # Plays this defense allowed in window_weeks (typed only).
                mask_def_window = (season_typed["defteam"] == defteam) & (
                    season_typed["week"].isin(window_weeks)
                )
                window_plays = season_typed[mask_def_window]
                if window_plays.empty:
                    continue

                # Offense overall mean (typed only) in the same window
                # across ALL defenses faced — schedule-of-strength baseline.
                mask_off_window = season_typed["week"].isin(window_weeks)
                off_window = season_typed[mask_off_window]
                off_means = off_window.groupby("posteam")["epa"].mean().rename("off_window_mean")

                joined = window_plays.merge(
                    off_means.to_frame(),
                    left_on="posteam",
                    right_index=True,
                    how="left",
                )
                joined["residual"] = joined["epa"] - joined["off_window_mean"]
                # Spec §5.1 step 6: per-week mean of per-play residuals.
                weekly_residuals = joined.groupby("week", as_index=False)["residual"].mean()
                # Spec §5.1 step 7: trailing-N-week mean over the per-week means.
                mean_residual = float(weekly_residuals["residual"].mean())

                rows.append(
                    {
                        "season": int(season),
                        "week": target_week,
                        "opp_team": defteam,
                        "opp_epa_allowed_residual": mean_residual,
                    }
                )

    if not rows:
        return empty_out

    out = pd.DataFrame(rows, columns=["season", "week", "opp_team", "opp_epa_allowed_residual"])
    out["season"] = out["season"].astype("int64")
    out["week"] = out["week"].astype("int64")
    return out


# --- Transitional shim — REMOVED IN PLAN 9 PHASE 3 -------------------------
# The v1 `opp_allowed_fppg` is no longer implemented; only a typed shim
# remains so that the per-position builders (qb.py, rb.py, wr.py, te.py)
# still type-check while Tasks 6-9 migrate each to `opp_epa_allowed_residual`.
# At runtime the shim raises immediately — this matches the Phase 3 design
# where each builder's tests intentionally fail at the start of its task and
# pass after the swap. Delete this once Task 9 (te.py) lands.
def opp_allowed_fppg(
    weekly_stats: pd.DataFrame,
    *,
    position: Position,
    ruleset: Ruleset,
    n_weeks: int,
) -> pd.DataFrame:
    """Removed in Plan 9. Use ``opp_epa_allowed_residual`` instead."""
    del weekly_stats, position, ruleset, n_weeks  # silence unused-arg
    raise NotImplementedError(
        "opp_allowed_fppg was replaced by opp_epa_allowed_residual in Plan 9. "
        "Per-position builders (qb/rb/wr/te) are migrated in Tasks 6-9."
    )
