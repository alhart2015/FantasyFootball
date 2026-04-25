"""TE feature builder. Pure function — no I/O, no caching."""

from __future__ import annotations

import pandas as pd

from projections.features._opponent import opp_allowed_fppg
from projections.features._rolling import last_n_per_group, trailing_n_share_in_group
from projections.features._shared import build_game_environment, exact_week_mask, prior_mask
from projections.schemas import (
    _PYARROW_STR,
    Position,
    Ruleset,
    Stat,
    TeFeaturesSchema,
)

_PASS_CATCHING_POSITIONS: tuple[str, ...] = (
    Position.WR.value,
    Position.RB.value,
    Position.TE.value,
)

_ROLLING_ZERO_FILL_COLS: tuple[str, ...] = (
    "targets_per_game_l4",
    "targets_per_game_std",
    "target_share_l4",
    "receptions_per_game_l4",
    "receiving_yards_per_game_l4",
    "receiving_tds_per_game_l4",
)


def _trailing_4_per_player(weekly_stats: pd.DataFrame, value_col: str) -> pd.DataFrame:
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
    if ngs.empty:
        return pd.DataFrame()
    return (
        ngs.sort_values(["season", "week"])
        .groupby("gsis_id", as_index=False, observed=True)
        .tail(1)
        .copy()
    )


def build_te_features(
    *,
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    depth_charts: pd.DataFrame,
    ngs_receiving: pd.DataFrame,
    schedules: pd.DataFrame,
    season: int,
    as_of_week: int,
) -> pd.DataFrame:
    """Build the TE feature DataFrame for week `as_of_week` of `season`."""
    ws = weekly_stats[prior_mask(weekly_stats, season=season, as_of_week=as_of_week)].copy()
    sc = snap_counts[prior_mask(snap_counts, season=season, as_of_week=as_of_week)].copy()
    ngs = ngs_receiving[prior_mask(ngs_receiving, season=season, as_of_week=as_of_week)].copy()
    dc = depth_charts[exact_week_mask(depth_charts, season=season, as_of_week=as_of_week)].copy()
    sch = schedules[exact_week_mask(schedules, season=season, as_of_week=as_of_week)].copy()

    te_dc = dc[dc["position"] == Position.TE.value].copy()
    if te_dc.empty:
        empty_cols = list(TeFeaturesSchema.to_schema().columns.keys())
        return TeFeaturesSchema.validate(pd.DataFrame(columns=empty_cols))

    ws_te = ws[ws["position"] == Position.TE.value].copy()
    sc_te = sc[sc["position"] == Position.TE.value].copy()
    ws_pass_catchers = ws[ws["position"].isin(_PASS_CATCHING_POSITIONS)].copy()

    # Rolling per-player receiving features (TE rows only)
    targets_l4 = _trailing_4_per_player(ws_te, Stat.TARGETS.value).rename(
        columns={"mean_l4": "targets_per_game_l4"}
    )
    rec_l4 = _trailing_4_per_player(ws_te, Stat.RECEPTIONS.value).rename(
        columns={"mean_l4": "receptions_per_game_l4"}
    )
    rec_yd_l4 = _trailing_4_per_player(ws_te, Stat.RECEIVING_YARDS.value).rename(
        columns={"mean_l4": "receiving_yards_per_game_l4"}
    )
    rec_td_l4 = _trailing_4_per_player(ws_te, Stat.RECEIVING_TDS.value).rename(
        columns={"mean_l4": "receiving_tds_per_game_l4"}
    )

    # target_share against the full team pass-catching group
    all_target_share = trailing_n_share_in_group(
        ws_pass_catchers, value_col=Stat.TARGETS.value
    ).rename(columns={"share_l4": "target_share_l4"})
    te_gsis_ids = set(ws_te["gsis_id"].unique())
    target_share = all_target_share[all_target_share["gsis_id"].isin(te_gsis_ids)].copy()

    # Season-to-date targets/game (TEs only)
    ws_this_season = ws_te[ws_te["season"] == season]
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

    snap_l4 = _trailing_4_per_player(sc_te, Stat.OFFENSE_PCT.value).rename(
        columns={"mean_l4": "snap_pct_l4"}
    )

    # NGS receiving latest snapshot
    ngs_latest = _latest_ngs_snapshot(ngs)
    ngs_std_cols = (
        "avg_separation_std",
        "avg_intended_air_yards_std",
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
                "avg_yac_above_expectation",
            ]
        ].rename(
            columns={
                "avg_separation": "avg_separation_std",
                "avg_intended_air_yards": "avg_intended_air_yards_std",
                "avg_yac_above_expectation": "avg_yac_above_expectation_std",
            }
        )

    game_env = build_game_environment(sch)
    opp_proxy_full = opp_allowed_fppg(
        ws_te, position=Position.TE, ruleset=Ruleset.espn_ppr(), n_weeks=4
    )
    opp_proxy = opp_proxy_full[
        (opp_proxy_full["season"] == season) & (opp_proxy_full["week"] == as_of_week)
    ].rename(columns={"opp_allowed_fppg": "opp_allowed_te_fppg_l4"})

    out = te_dc[["gsis_id", "season", "week", "team", "depth_rank"]].copy()
    out = out.merge(game_env, on=["season", "week", "team"], how="left")
    out = out.rename(columns={"opp_team": "opponent"})

    out = out.merge(targets_l4, on="gsis_id", how="left")
    out = out.merge(targets_std, on="gsis_id", how="left")
    out = out.merge(target_share, on="gsis_id", how="left")
    out = out.merge(rec_l4, on="gsis_id", how="left")
    out = out.merge(rec_yd_l4, on="gsis_id", how="left")
    out = out.merge(rec_td_l4, on="gsis_id", how="left")
    out = out.merge(snap_l4, on="gsis_id", how="left")
    out = out.merge(ngs_cols, on="gsis_id", how="left")
    out = out.merge(
        opp_proxy[["season", "week", "opp_team", "opp_allowed_te_fppg_l4"]].rename(
            columns={"opp_team": "opponent"}
        ),
        on=["season", "week", "opponent"],
        how="left",
    )

    for c in _ROLLING_ZERO_FILL_COLS:
        out[c] = out[c].fillna(0.0).astype(float)

    out["depth_rank"] = out["depth_rank"].astype(pd.Int64Dtype())
    for col in ("team", "opponent"):
        out[col] = out[col].astype(_PYARROW_STR)

    return TeFeaturesSchema.validate(out)
