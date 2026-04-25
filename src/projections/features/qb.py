"""QB feature builder. Pure function — no I/O, no caching.

Output is one row per (gsis_id, season, week=as_of_week) for every QB on
a roster in week as_of_week of season. Validates against QbFeaturesSchema."""

from __future__ import annotations

from typing import Final

import pandas as pd

from projections.features._opponent import opp_allowed_fppg
from projections.features._rolling import last_n_per_group
from projections.features.wr import _build_game_environment, _exact_week_mask, _prior_mask
from projections.schemas import (
    _PYARROW_STR,
    Position,
    QbFeaturesSchema,
    Ruleset,
    Stat,
)

_RUSHING_QB_THRESHOLD: Final = 5.0  # carries/game over trailing 4

_ROLLING_ZERO_FILL_COLS: tuple[str, ...] = (
    "pass_attempts_per_game_l4",
    "passing_yards_per_game_l4",
    "passing_tds_per_game_l4",
    "interceptions_per_game_l4",
    "sacks_per_game_l4",
    "passing_yards_per_game_std",
    "rushing_attempts_per_game_l4",
    "rushing_yards_per_game_l4",
)


def _trailing_4_per_player(weekly_stats: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-player mean of `value_col` over the trailing 4 games."""
    if weekly_stats.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "mean_l4": pd.array([], dtype=float),
            }
        )
    last4 = last_n_per_group(
        weekly_stats,
        group_cols=["gsis_id"],
        sort_cols=["season", "week"],
        n=4,
    )
    return (
        last4.groupby("gsis_id", as_index=False, observed=True)[value_col]
        .mean()
        .rename(columns={value_col: "mean_l4"})
    )


def _latest_ngs_snapshot(ngs: pd.DataFrame) -> pd.DataFrame:
    """Per-player most-recent NGS row (assumes ngs already prior-filtered)."""
    if ngs.empty:
        return pd.DataFrame()
    return (
        ngs.sort_values(["season", "week"])
        .groupby("gsis_id", as_index=False, observed=True)
        .tail(1)
        .copy()
    )


def build_qb_features(
    *,
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    depth_charts: pd.DataFrame,
    ngs_passing: pd.DataFrame,
    schedules: pd.DataFrame,
    season: int,
    as_of_week: int,
) -> pd.DataFrame:
    """Build the QB feature DataFrame for week `as_of_week` of `season`."""
    # --- Leakage-safe input filtering -------------------------------------
    ws = weekly_stats[_prior_mask(weekly_stats, season=season, as_of_week=as_of_week)].copy()
    sc = snap_counts[_prior_mask(snap_counts, season=season, as_of_week=as_of_week)].copy()
    ngs = ngs_passing[_prior_mask(ngs_passing, season=season, as_of_week=as_of_week)].copy()
    dc = depth_charts[_exact_week_mask(depth_charts, season=season, as_of_week=as_of_week)].copy()
    sch = schedules[_exact_week_mask(schedules, season=season, as_of_week=as_of_week)].copy()

    # --- Rostered QBs in target week (depth chart drives roster set) ------
    qb_dc = dc[dc["position"] == Position.QB.value].copy()
    if qb_dc.empty:
        empty_cols = list(QbFeaturesSchema.to_schema().columns.keys())
        return QbFeaturesSchema.validate(pd.DataFrame(columns=empty_cols))

    ws_qb = ws[ws["position"] == Position.QB.value].copy()
    sc_qb = sc[sc["position"] == Position.QB.value].copy()

    # --- Per-player rolling features --------------------------------------
    pass_att_l4 = _trailing_4_per_player(ws_qb, Stat.PASSING_ATTEMPTS.value).rename(
        columns={"mean_l4": "pass_attempts_per_game_l4"}
    )
    pass_yd_l4 = _trailing_4_per_player(ws_qb, Stat.PASSING_YARDS.value).rename(
        columns={"mean_l4": "passing_yards_per_game_l4"}
    )
    pass_td_l4 = _trailing_4_per_player(ws_qb, Stat.PASSING_TDS.value).rename(
        columns={"mean_l4": "passing_tds_per_game_l4"}
    )
    int_l4 = _trailing_4_per_player(ws_qb, Stat.INTERCEPTIONS.value).rename(
        columns={"mean_l4": "interceptions_per_game_l4"}
    )
    sacks_l4 = _trailing_4_per_player(ws_qb, Stat.SACKS.value).rename(
        columns={"mean_l4": "sacks_per_game_l4"}
    )
    rush_att_l4 = _trailing_4_per_player(ws_qb, Stat.CARRIES.value).rename(
        columns={"mean_l4": "rushing_attempts_per_game_l4"}
    )
    rush_yd_l4 = _trailing_4_per_player(ws_qb, Stat.RUSHING_YARDS.value).rename(
        columns={"mean_l4": "rushing_yards_per_game_l4"}
    )

    ws_this_season = ws_qb[ws_qb["season"] == season]
    if ws_this_season.empty:
        pass_yd_std = pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "passing_yards_per_game_std": pd.array([], dtype=float),
            }
        )
    else:
        pass_yd_std = (
            ws_this_season.groupby("gsis_id", as_index=False, observed=True)[
                Stat.PASSING_YARDS.value
            ]
            .mean()
            .rename(columns={Stat.PASSING_YARDS.value: "passing_yards_per_game_std"})
        )

    snap_l4 = _trailing_4_per_player(sc_qb, Stat.OFFENSE_PCT.value).rename(
        columns={"mean_l4": "snap_pct_l4"}
    )

    # --- NGS latest snapshot per player → *_std columns -------------------
    ngs_latest = _latest_ngs_snapshot(ngs)
    ngs_std_cols = (
        "aggressiveness_std",
        "completion_percentage_above_expectation_std",
        "avg_intended_air_yards_std",
        "avg_time_to_throw_std",
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
                "aggressiveness",
                "completion_percentage_above_expectation",
                "avg_intended_air_yards",
                "avg_time_to_throw",
            ]
        ].rename(
            columns={
                "aggressiveness": "aggressiveness_std",
                "completion_percentage_above_expectation": (
                    "completion_percentage_above_expectation_std"
                ),
                "avg_intended_air_yards": "avg_intended_air_yards_std",
                "avg_time_to_throw": "avg_time_to_throw_std",
            }
        )

    # --- Game environment from schedules ---------------------------------
    game_env = _build_game_environment(sch)

    # --- Opponent strength proxy ------------------------------------------
    opp_proxy_full = opp_allowed_fppg(
        ws_qb, position=Position.QB, ruleset=Ruleset.espn_ppr(), n_weeks=4
    )
    opp_proxy = opp_proxy_full[
        (opp_proxy_full["season"] == season) & (opp_proxy_full["week"] == as_of_week)
    ].rename(columns={"opp_allowed_fppg": "opp_allowed_qb_fppg_l4"})

    # --- Assemble: depth chart drives the row set, join everything else ---
    out = qb_dc[["gsis_id", "season", "week", "team", "depth_rank"]].copy()
    out = out.merge(game_env, on=["season", "week", "team"], how="left")
    out = out.rename(columns={"opp_team": "opponent"})

    out = out.merge(pass_att_l4, on="gsis_id", how="left")
    out = out.merge(pass_yd_l4, on="gsis_id", how="left")
    out = out.merge(pass_td_l4, on="gsis_id", how="left")
    out = out.merge(int_l4, on="gsis_id", how="left")
    out = out.merge(sacks_l4, on="gsis_id", how="left")
    out = out.merge(pass_yd_std, on="gsis_id", how="left")
    out = out.merge(rush_att_l4, on="gsis_id", how="left")
    out = out.merge(rush_yd_l4, on="gsis_id", how="left")
    out = out.merge(snap_l4, on="gsis_id", how="left")
    out = out.merge(ngs_cols, on="gsis_id", how="left")
    out = out.merge(
        opp_proxy[["season", "week", "opp_team", "opp_allowed_qb_fppg_l4"]].rename(
            columns={"opp_team": "opponent"}
        ),
        on=["season", "week", "opponent"],
        how="left",
    )

    for c in _ROLLING_ZERO_FILL_COLS:
        out[c] = out[c].fillna(0.0).astype(float)

    out["rushing_qb"] = out["rushing_attempts_per_game_l4"] >= _RUSHING_QB_THRESHOLD
    out["depth_rank"] = out["depth_rank"].astype(pd.Int64Dtype())

    for col in ("team", "opponent"):
        out[col] = out[col].astype(_PYARROW_STR)

    return QbFeaturesSchema.validate(out)
