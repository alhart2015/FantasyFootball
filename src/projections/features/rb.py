"""RB feature builder. Pure function — no I/O, no caching."""

from __future__ import annotations

from typing import Final

import pandas as pd

from projections.features._opponent import opp_epa_allowed_residual
from projections.features._rolling import (
    latest_ngs_snapshot,
    trailing_4_per_player,
    trailing_n_share_in_group,
)
from projections.features._shared import build_game_environment, exact_week_mask, prior_mask
from projections.schemas import (
    _PYARROW_STR,
    Position,
    RbFeaturesSchema,
    Stat,
)

_PASSING_DOWN_BACK_THRESHOLD: Final = 4.0  # targets/game over trailing 4

_PASS_CATCHING_POSITIONS: tuple[str, ...] = (
    Position.WR.value,
    Position.RB.value,
    Position.TE.value,
)

_ROLLING_ZERO_FILL_COLS: tuple[str, ...] = (
    "carries_per_game_l4",
    "rushing_yards_per_game_l4",
    "rushing_tds_per_game_l4",
    "rush_share_l4",
    "targets_per_game_l4",
    "receptions_per_game_l4",
    "receiving_yards_per_game_l4",
    "target_share_l4",
    "targets_per_game_std",
)


def build_rb_features(
    *,
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    depth_charts: pd.DataFrame,
    ngs_rushing: pd.DataFrame,
    schedules: pd.DataFrame,
    pbp: pd.DataFrame,
    season: int,
    as_of_week: int,
) -> pd.DataFrame:
    """Build the RB feature DataFrame for week `as_of_week` of `season`.

    `pbp` is the play-by-play frame produced by `ingest.pbp` (PbpSchema).
    It feeds the schedule-of-strength-adjusted opponent run-EPA residual
    (Plan 9), which replaces the v1 ``opp_allowed_rb_fppg_l4``.
    """
    ws = weekly_stats[prior_mask(weekly_stats, season=season, as_of_week=as_of_week)].copy()
    sc = snap_counts[prior_mask(snap_counts, season=season, as_of_week=as_of_week)].copy()
    ngs = ngs_rushing[prior_mask(ngs_rushing, season=season, as_of_week=as_of_week)].copy()
    dc = depth_charts[exact_week_mask(depth_charts, season=season, as_of_week=as_of_week)].copy()
    sch = schedules[exact_week_mask(schedules, season=season, as_of_week=as_of_week)].copy()

    # Restrict to teams that have a schedule row this week — bye-week RBs
    # have no opponent / is_home / roof_dome to populate, and the schema
    # rejects NaN on those columns. Mirrors the WR filter (PR #4 review).
    sch_teams = set(sch["home_team"].astype(str)) | set(sch["away_team"].astype(str))
    rb_dc = dc[
        (dc["position"] == Position.RB.value) & (dc["team"].astype(str).isin(sch_teams))
    ].copy()
    # Dedupe: a player can appear multiple times in the same depth chart (e.g.,
    # listed under multiple slot labels, or traded mid-week). Keep the lowest
    # depth_rank (the player's primary listing). Mirrors the WR dedupe (PR #4
    # review, TODO #9c).
    rb_dc = (
        rb_dc.sort_values(["gsis_id", "season", "week", "depth_rank"])
        .drop_duplicates(subset=["gsis_id", "season", "week"], keep="first")
        .copy()
    )
    if rb_dc.empty:
        empty_cols = list(RbFeaturesSchema.to_schema().columns.keys())
        return RbFeaturesSchema.validate(pd.DataFrame(columns=empty_cols))

    ws_rb = ws[ws["position"] == Position.RB.value].copy()
    sc_rb = sc[sc["position"] == Position.RB.value].copy()
    ws_pass_catchers = ws[ws["position"].isin(_PASS_CATCHING_POSITIONS)].copy()

    # --- Rolling per-player rushing/receiving features --------------------
    carries_l4 = trailing_4_per_player(ws_rb, Stat.CARRIES.value).rename(
        columns={"mean_l4": "carries_per_game_l4"}
    )
    rush_yd_l4 = trailing_4_per_player(ws_rb, Stat.RUSHING_YARDS.value).rename(
        columns={"mean_l4": "rushing_yards_per_game_l4"}
    )
    rush_td_l4 = trailing_4_per_player(ws_rb, Stat.RUSHING_TDS.value).rename(
        columns={"mean_l4": "rushing_tds_per_game_l4"}
    )
    targets_l4 = trailing_4_per_player(ws_rb, Stat.TARGETS.value).rename(
        columns={"mean_l4": "targets_per_game_l4"}
    )
    rec_l4 = trailing_4_per_player(ws_rb, Stat.RECEPTIONS.value).rename(
        columns={"mean_l4": "receptions_per_game_l4"}
    )
    rec_yd_l4 = trailing_4_per_player(ws_rb, Stat.RECEIVING_YARDS.value).rename(
        columns={"mean_l4": "receiving_yards_per_game_l4"}
    )

    # --- Shares -----------------------------------------------------------
    # Helper returns one row per (gsis_id, team); the (gsis_id, team) join in
    # the assemble step below picks the share for the player's current team
    # and naturally drops non-RB rows from the pass-catcher input as well as
    # secondary rows for players traded mid-window.
    rush_share = trailing_n_share_in_group(ws_rb, value_col=Stat.CARRIES.value).rename(
        columns={"share_l4": "rush_share_l4"}
    )
    target_share = trailing_n_share_in_group(ws_pass_catchers, value_col=Stat.TARGETS.value).rename(
        columns={"share_l4": "target_share_l4"}
    )

    # --- Season-to-date targets-per-game (RBs only) -----------------------
    ws_this_season = ws_rb[ws_rb["season"] == season]
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

    snap_l4 = trailing_4_per_player(sc_rb, Stat.OFFENSE_PCT.value).rename(
        columns={"mean_l4": "snap_pct_l4"}
    )

    # --- NGS rushing latest snapshot --------------------------------------
    ngs_latest = latest_ngs_snapshot(ngs)
    ngs_std_cols = (
        "efficiency_std",
        "rush_yards_over_expected_per_att_std",
        "percent_attempts_gte_eight_defenders_std",
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
                "efficiency",
                "rush_yards_over_expected_per_att",
                "percent_attempts_gte_eight_defenders",
            ]
        ].rename(
            columns={
                "efficiency": "efficiency_std",
                "rush_yards_over_expected_per_att": "rush_yards_over_expected_per_att_std",
                "percent_attempts_gte_eight_defenders": "percent_attempts_gte_eight_defenders_std",
            }
        )

    # --- Game environment from schedules ---------------------------------
    game_env = build_game_environment(sch)

    # --- Opponent strength: opp-adjusted run-EPA residual (Plan 9) -------
    # Filter to just the trailing window weeks before passing to the helper
    # (see qb.py for rationale).
    n_pbp_weeks = 4
    pbp_window = pbp[
        (pbp["season"] == season)
        & (pbp["week"] >= as_of_week - n_pbp_weeks)
        & (pbp["week"] < as_of_week)
    ].copy()
    opp_proxy_full = opp_epa_allowed_residual(pbp_window, play_type="run", n_weeks=n_pbp_weeks)
    opp_proxy = opp_proxy_full[
        (opp_proxy_full["season"] == season) & (opp_proxy_full["week"] == as_of_week)
    ].rename(columns={"opp_epa_allowed_residual": "opp_run_epa_allowed_l4"})

    # --- Assemble ---------------------------------------------------------
    out = rb_dc[["gsis_id", "season", "week", "team", "depth_rank"]].copy()
    out = out.merge(game_env, on=["season", "week", "team"], how="left")
    out = out.rename(columns={"opp_team": "opponent"})

    out = out.merge(carries_l4, on="gsis_id", how="left")
    out = out.merge(rush_yd_l4, on="gsis_id", how="left")
    out = out.merge(rush_td_l4, on="gsis_id", how="left")
    out = out.merge(rush_share, on=["gsis_id", "team"], how="left")
    out = out.merge(targets_l4, on="gsis_id", how="left")
    out = out.merge(rec_l4, on="gsis_id", how="left")
    out = out.merge(rec_yd_l4, on="gsis_id", how="left")
    out = out.merge(target_share, on=["gsis_id", "team"], how="left")
    out = out.merge(targets_std, on="gsis_id", how="left")
    out = out.merge(snap_l4, on="gsis_id", how="left")
    out = out.merge(ngs_cols, on="gsis_id", how="left")
    out = out.merge(
        opp_proxy[["season", "week", "opp_team", "opp_run_epa_allowed_l4"]].rename(
            columns={"opp_team": "opponent"}
        ),
        on=["season", "week", "opponent"],
        how="left",
    )

    for c in _ROLLING_ZERO_FILL_COLS:
        out[c] = out[c].fillna(0.0).astype(float)

    out["passing_down_back"] = out["targets_per_game_l4"] >= _PASSING_DOWN_BACK_THRESHOLD
    out["depth_rank"] = out["depth_rank"].astype(pd.Int64Dtype())

    for col in ("team", "opponent"):
        out[col] = out[col].astype(_PYARROW_STR)

    return RbFeaturesSchema.validate(out)
