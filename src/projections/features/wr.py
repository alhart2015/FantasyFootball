"""WR feature builder. Pure function — no I/O, no caching.

Output is one row per (gsis_id, season, week=as_of_week) for every WR on
a roster in week as_of_week of season. Validates against WrFeaturesSchema."""

from __future__ import annotations

from typing import Final

import pandas as pd

from projections.features._opponent import opp_allowed_fppg
from projections.features._rolling import (
    latest_ngs_snapshot,
    trailing_4_per_player,
    trailing_n_share_in_group,
)
from projections.features._shared import build_game_environment, exact_week_mask, prior_mask
from projections.features.trajectory_features import (
    attach_trajectory_features,
    build_draft_lookup,
)
from projections.schemas import (
    _PYARROW_STR,
    Position,
    Ruleset,
    Stat,
    WrFeaturesSchema,
)

# Module-level singletons for the optional builder kwargs. Per ruff B008,
# we cannot put `pd.DataFrame()` directly in function defaults.
_EMPTY_PBP: Final[pd.DataFrame] = pd.DataFrame()
_EMPTY_DRAFT_PICKS: Final[pd.DataFrame] = pd.DataFrame()

_DESIGNED_RUSHER_THRESHOLD: Final = 1.5  # carries/game over trailing 4

# Columns that represent rolling-window per-game means; rookies with no prior
# games receive 0.0 for these (the schema disallows NaN on them).
_ROLLING_ZERO_FILL_COLS: tuple[str, ...] = (
    "targets_per_game_l4",
    "targets_per_game_std",
    "receptions_per_game_l4",
    "receiving_yards_per_game_l4",
    "receiving_tds_per_game_l4",
    "rushing_attempts_per_game_l4",
    "rushing_yards_per_game_l4",
    "target_share_l4",
)


def build_wr_features(
    *,
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    depth_charts: pd.DataFrame,
    ngs_receiving: pd.DataFrame,
    schedules: pd.DataFrame,
    season: int,
    as_of_week: int,
    # pbp: reserved plumbing for future PBP-driven features (Plan 9 Phase 6
    # negative result, kept threaded for future plans). Currently unused.
    pbp: pd.DataFrame = _EMPTY_PBP,
    # draft_picks: consumed by the trajectory features (PR #25 + 2026-05-03
    # WR integration). Empty -> all rows route to inferred-draft-year fallback.
    draft_picks: pd.DataFrame = _EMPTY_DRAFT_PICKS,
) -> pd.DataFrame:
    """Build the WR feature DataFrame for week `as_of_week` of `season`.

    Inputs are validated against their respective schemas (caller's
    responsibility). The function filters every input to leakage-safe rows
    before computing anything — see prior_mask / exact_week_mask.
    """
    # --- Leakage-safe input filtering -------------------------------------
    ws = weekly_stats[prior_mask(weekly_stats, season=season, as_of_week=as_of_week)].copy()
    sc = snap_counts[prior_mask(snap_counts, season=season, as_of_week=as_of_week)].copy()
    ngs = ngs_receiving[prior_mask(ngs_receiving, season=season, as_of_week=as_of_week)].copy()
    dc = depth_charts[exact_week_mask(depth_charts, season=season, as_of_week=as_of_week)].copy()
    sch = schedules[exact_week_mask(schedules, season=season, as_of_week=as_of_week)].copy()

    # --- Rostered WRs in target week (depth chart drives roster set) ------
    # Restrict to teams that have a schedule row this week — bye-week WRs
    # have no opponent / is_home / roof_dome to populate, and the schema
    # rejects NaN on those columns. Closing TODO #9a from PR #4 review.
    sch_teams = set(sch["home_team"].astype(str)) | set(sch["away_team"].astype(str))
    wr_dc = dc[
        (dc["position"] == Position.WR.value) & (dc["team"].astype(str).isin(sch_teams))
    ].copy()
    # Dedupe: a player can appear multiple times in the same depth chart (e.g.,
    # listed under multiple slot labels like LWR + SWR, or traded mid-week).
    # Keep the lowest depth_rank (the player's primary listing). Closing
    # TODO #9c from PR #4 review.
    wr_dc = (
        wr_dc.sort_values(["gsis_id", "season", "week", "depth_rank"])
        .drop_duplicates(subset=["gsis_id", "season", "week"], keep="first")
        .copy()
    )
    if wr_dc.empty:
        empty_cols = list(WrFeaturesSchema.to_schema().columns.keys())
        return WrFeaturesSchema.validate(pd.DataFrame(columns=empty_cols))

    # Restrict prior frames to WR position so RB carries / TE targets don't
    # contaminate the rolling means or team-share denominators.
    ws_wr = ws[ws["position"] == Position.WR.value].copy()
    sc_wr = sc[sc["position"] == Position.WR.value].copy()

    # --- Per-player rolling features --------------------------------------
    targets_l4 = trailing_4_per_player(ws_wr, Stat.TARGETS.value).rename(
        columns={"mean_l4": "targets_per_game_l4"}
    )
    rec_l4 = trailing_4_per_player(ws_wr, Stat.RECEPTIONS.value).rename(
        columns={"mean_l4": "receptions_per_game_l4"}
    )
    rec_yd_l4 = trailing_4_per_player(ws_wr, Stat.RECEIVING_YARDS.value).rename(
        columns={"mean_l4": "receiving_yards_per_game_l4"}
    )
    rec_td_l4 = trailing_4_per_player(ws_wr, Stat.RECEIVING_TDS.value).rename(
        columns={"mean_l4": "receiving_tds_per_game_l4"}
    )
    rush_att_l4 = trailing_4_per_player(ws_wr, Stat.CARRIES.value).rename(
        columns={"mean_l4": "rushing_attempts_per_game_l4"}
    )
    rush_yd_l4 = trailing_4_per_player(ws_wr, Stat.RUSHING_YARDS.value).rename(
        columns={"mean_l4": "rushing_yards_per_game_l4"}
    )

    # Season-to-date targets-per-game = mean across all prior in-season weeks.
    ws_this_season = ws_wr[ws_wr["season"] == season]
    if ws_this_season.empty:
        targets_std = pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "targets_per_game_std": pd.array([], dtype=float),
            }
        )
    else:
        targets_std = (
            ws_this_season.groupby("gsis_id", as_index=False, observed=True)[Stat.TARGETS.value]
            .mean()
            .rename(columns={Stat.TARGETS.value: "targets_per_game_std"})
        )

    # Helper returns one row per (gsis_id, team); the join on ["gsis_id", "team"]
    # below picks the share for the player's depth-chart-current team and
    # naturally drops the secondary row for players traded mid-window.
    target_share = trailing_n_share_in_group(ws_wr, value_col=Stat.TARGETS.value).rename(
        columns={"share_l4": "target_share_l4"}
    )
    air_yards_share = trailing_n_share_in_group(
        ws_wr, value_col=Stat.RECEIVING_AIR_YARDS.value
    ).rename(columns={"share_l4": "air_yards_share_l4"})
    # receiving_air_yards can be slightly negative on a TFL behind LOS, so the
    # per-player share can dip negative even though semantically we want a
    # [0, 1] feature. Clip — the schema's lower bound is 0.
    air_yards_share["air_yards_share_l4"] = air_yards_share["air_yards_share_l4"].clip(0.0, 1.0)

    snap_l4 = trailing_4_per_player(sc_wr, Stat.OFFENSE_PCT.value).rename(
        columns={"mean_l4": "snap_pct_l4"}
    )

    # --- NGS latest snapshot per player → *_std columns -------------------
    ngs_latest = latest_ngs_snapshot(ngs)
    ngs_std_cols = (
        "avg_separation_std",
        "avg_intended_air_yards_std",
        "percent_share_intended_air_yards_std",
        "avg_yac_above_expectation_std",
    )
    if ngs_latest.empty:
        ngs_cols = pd.DataFrame(
            {"gsis_id": pd.array([], dtype=_PYARROW_STR)}
            | {c: pd.array([], dtype=float) for c in ngs_std_cols}
        )
    else:
        ngs_cols = ngs_latest[
            [
                "gsis_id",
                "avg_separation",
                "avg_intended_air_yards",
                "percent_share_of_intended_air_yards",
                "avg_yac_above_expectation",
            ]
        ].rename(
            columns={
                "avg_separation": "avg_separation_std",
                "avg_intended_air_yards": "avg_intended_air_yards_std",
                "percent_share_of_intended_air_yards": ("percent_share_intended_air_yards_std"),
                "avg_yac_above_expectation": "avg_yac_above_expectation_std",
            }
        )
        # NGS reports share as a 0-100 percentage; the schema range is 0-1.
        # Clip — real-data shares can dip slightly negative when team total
        # air yards aggregates a TFL play, same situation as air_yards_share_l4.
        ngs_cols["percent_share_intended_air_yards_std"] = (
            ngs_cols["percent_share_intended_air_yards_std"].astype(float) / 100.0
        ).clip(0.0, 1.0)

    # --- Game environment from schedules ---------------------------------
    game_env = build_game_environment(sch)

    # --- Opponent strength proxy ------------------------------------------
    opp_proxy_full = opp_allowed_fppg(
        ws_wr, position=Position.WR, ruleset=Ruleset.espn_ppr(), n_weeks=4
    )
    opp_proxy = opp_proxy_full[
        (opp_proxy_full["season"] == season) & (opp_proxy_full["week"] == as_of_week)
    ].rename(columns={"opp_allowed_fppg": "opp_allowed_wr_fppg_l4"})

    # --- Assemble: depth chart drives the row set, join everything else ---
    out = wr_dc[["gsis_id", "season", "week", "team", "depth_rank"]].copy()
    out = out.merge(game_env, on=["season", "week", "team"], how="left")
    out = out.rename(columns={"opp_team": "opponent"})

    out = out.merge(targets_l4, on="gsis_id", how="left")
    out = out.merge(targets_std, on="gsis_id", how="left")
    out = out.merge(target_share, on=["gsis_id", "team"], how="left")
    out = out.merge(air_yards_share, on=["gsis_id", "team"], how="left")
    out = out.merge(rec_l4, on="gsis_id", how="left")
    out = out.merge(rec_yd_l4, on="gsis_id", how="left")
    out = out.merge(rec_td_l4, on="gsis_id", how="left")
    out = out.merge(rush_att_l4, on="gsis_id", how="left")
    out = out.merge(rush_yd_l4, on="gsis_id", how="left")
    out = out.merge(snap_l4, on="gsis_id", how="left")
    out = out.merge(ngs_cols, on="gsis_id", how="left")
    out = out.merge(
        opp_proxy[["season", "week", "opp_team", "opp_allowed_wr_fppg_l4"]].rename(
            columns={"opp_team": "opponent"}
        ),
        on=["season", "week", "opponent"],
        how="left",
    )

    # Rookies / players with no prior games: fill schema-required floats with 0.
    # Real-data weekly stats permit slightly-negative yardage (TFL behind
    # LOS); the feature schema declares these rolling means as ``ge=0``, so
    # clip after the fillna. The clip is essentially a no-op for clean fits
    # — only fires on the few rows where 4-week-mean rushing/receiving
    # yardage went negative, which is noise we can safely floor at 0.
    for c in _ROLLING_ZERO_FILL_COLS:
        out[c] = out[c].fillna(0.0).astype(float).clip(lower=0.0)

    out["designed_rusher"] = out["rushing_attempts_per_game_l4"] >= _DESIGNED_RUSHER_THRESHOLD

    # depth_rank may have lost dtype through merges; restore nullable Int64
    # (the schema declares Series[int] but allows NaN via Pandera's promotion).
    out["depth_rank"] = out["depth_rank"].astype(pd.Int64Dtype())

    # Schema enforces Series[str] for team/opponent — our merges may have
    # introduced object dtype on those columns.
    for col in ("team", "opponent"):
        out[col] = out[col].astype(_PYARROW_STR)

    # --- Trajectory features (PR #25 + 2026-05-03 WR integration) ---------
    # Helper does its own position filter + rolling, so pass the unfiltered
    # (but prior-mask-filtered) weekly_stats / snap_counts.
    draft_lookup = build_draft_lookup(draft_picks)
    traj_idx = out[["gsis_id", "season", "week", "team", "opponent"]].rename(
        columns={"opponent": "opp"}
    )
    traj = attach_trajectory_features(traj_idx, ws, sc, draft_lookup, Position.WR)
    out = out.merge(
        traj[
            [
                "gsis_id",
                "season",
                "week",
                "age",
                "is_rookie",
                "volume_trend_l4_minus_prior_l4",
                "snap_pct_change_l4_vs_prior_l4",
            ]
        ],
        on=["gsis_id", "season", "week"],
        how="left",
    )

    return WrFeaturesSchema.validate(out)
