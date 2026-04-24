"""WR feature builder. Pure function — no I/O, no caching.

Output is one row per (gsis_id, season, week=as_of_week) for every WR on
a roster in week as_of_week of season. Validates against WrFeaturesSchema."""

from __future__ import annotations

import pandas as pd

from projections.features._opponent import opp_allowed_fppg
from projections.features._rolling import last_n_per_group
from projections.schemas import (
    _PYARROW_STR,
    Position,
    Ruleset,
    WrFeaturesSchema,
)

_DESIGNED_RUSHER_THRESHOLD = 1.5  # carries/game over trailing 4

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


def _prior_mask(df: pd.DataFrame, *, season: int, as_of_week: int) -> pd.Series:
    return (df["season"] < season) | ((df["season"] == season) & (df["week"] < as_of_week))


def _exact_week_mask(df: pd.DataFrame, *, season: int, as_of_week: int) -> pd.Series:
    return (df["season"] == season) & (df["week"] == as_of_week)


def _trailing_4_per_player(weekly_stats: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Return a per-player frame with `mean_l4` = mean of `value_col` over the
    trailing 4 games. Players with 0 prior games are simply absent — the caller
    fills their value with 0.0 after the merge."""
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


def _trailing_4_share_per_team(weekly_stats: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Trailing-4 player share of `value_col` within their team's WR group.

    For each player, we sum `value_col` over their last 4 games. The team total
    is the sum of those per-player trailing-4 sums across all WRs on the team.
    The share is per-player-sum / team-sum (0 if team-sum is 0).
    """
    if weekly_stats.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "share_l4": pd.array([], dtype=float),
            }
        )
    last4_player = last_n_per_group(
        weekly_stats,
        group_cols=["gsis_id"],
        sort_cols=["season", "week"],
        n=4,
    )
    player_sum = last4_player.groupby(["gsis_id", "team"], as_index=False, observed=True)[
        value_col
    ].sum()
    team_sum = (
        player_sum.groupby("team", as_index=False, observed=True)[value_col]
        .sum()
        .rename(columns={value_col: "team_total"})
    )
    merged = player_sum.merge(team_sum, on="team", how="left")
    merged["share_l4"] = (
        merged[value_col].astype(float) / merged["team_total"].astype(float)
    ).where(merged["team_total"] > 0, 0.0)
    return merged[["gsis_id", "share_l4"]]


def _latest_ngs_snapshot(ngs: pd.DataFrame) -> pd.DataFrame:
    """Per-player most-recent NGS snapshot (the season-to-date columns the WR
    builder propagates as `*_std` features). `ngs` is assumed already filtered
    to leakage-safe rows by the caller."""
    if ngs.empty:
        return pd.DataFrame()
    return (
        ngs.sort_values(["season", "week"])
        .groupby("gsis_id", as_index=False, observed=True)
        .tail(1)
        .copy()
    )


def _build_game_environment(schedules: pd.DataFrame) -> pd.DataFrame:
    """Per-team game-environment row from a per-game schedules frame.

    Output columns (one row per team-game, two per game):
        season, week, team, opp_team, is_home, spread, implied_team_total,
        roof_dome.

    Sign convention: `nfl_data_py.spread_line` is the HOME team's spread in
    standard sportsbook terms (negative = home favored, positive = home dog /
    away favored). We follow that convention end-to-end:

        home_implied = (total_line - spread_line) / 2
        away_implied = (total_line + spread_line) / 2

    The per-team `spread` column we expose downstream is the team's own signed
    spread (negative = that team favored). Therefore:

        home_spread = +spread_line   # home dog gets positive
        away_spread = -spread_line   # away favorite gets negative
    """
    home = schedules[
        ["season", "week", "home_team", "away_team", "spread_line", "total_line", "roof"]
    ].rename(columns={"home_team": "team", "away_team": "opp_team"})
    home["is_home"] = True
    home["spread"] = home["spread_line"].astype(float)

    away = schedules[
        ["season", "week", "home_team", "away_team", "spread_line", "total_line", "roof"]
    ].rename(columns={"away_team": "team", "home_team": "opp_team"})
    away["is_home"] = False
    away["spread"] = -away["spread_line"].astype(float)

    game_env = pd.concat([home, away], ignore_index=True)
    # Implied team total = (total - team's signed spread) / 2. The favored team
    # has a negative spread, so it correctly gets the higher implied total.
    game_env["implied_team_total"] = (
        game_env["total_line"].astype(float) - game_env["spread"].astype(float)
    ) / 2.0
    game_env["roof_dome"] = game_env["roof"].isin(["dome", "closed"]).fillna(False).astype(bool)
    return game_env[
        [
            "season",
            "week",
            "team",
            "opp_team",
            "is_home",
            "spread",
            "implied_team_total",
            "roof_dome",
        ]
    ]


def build_wr_features(
    *,
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    depth_charts: pd.DataFrame,
    ngs_receiving: pd.DataFrame,
    schedules: pd.DataFrame,
    season: int,
    as_of_week: int,
) -> pd.DataFrame:
    """Build the WR feature DataFrame for week `as_of_week` of `season`.

    Inputs are validated against their respective schemas (caller's
    responsibility). The function filters every input to leakage-safe rows
    before computing anything — see _prior_mask / _exact_week_mask.
    """
    # --- Leakage-safe input filtering -------------------------------------
    ws = weekly_stats[_prior_mask(weekly_stats, season=season, as_of_week=as_of_week)].copy()
    sc = snap_counts[_prior_mask(snap_counts, season=season, as_of_week=as_of_week)].copy()
    ngs = ngs_receiving[_prior_mask(ngs_receiving, season=season, as_of_week=as_of_week)].copy()
    dc = depth_charts[_exact_week_mask(depth_charts, season=season, as_of_week=as_of_week)].copy()
    sch = schedules[_exact_week_mask(schedules, season=season, as_of_week=as_of_week)].copy()

    # --- Rostered WRs in target week (depth chart drives roster set) ------
    wr_dc = dc[dc["position"] == Position.WR.value].copy()
    if wr_dc.empty:
        empty_cols = list(WrFeaturesSchema.to_schema().columns.keys())
        return WrFeaturesSchema.validate(pd.DataFrame(columns=empty_cols))

    # Restrict prior frames to WR position so RB carries / TE targets don't
    # contaminate the rolling means or team-share denominators.
    ws_wr = ws[ws["position"] == Position.WR.value].copy()
    sc_wr = sc[sc["position"] == Position.WR.value].copy()

    # --- Per-player rolling features --------------------------------------
    targets_l4 = _trailing_4_per_player(ws_wr, "targets").rename(
        columns={"mean_l4": "targets_per_game_l4"}
    )
    rec_l4 = _trailing_4_per_player(ws_wr, "receptions").rename(
        columns={"mean_l4": "receptions_per_game_l4"}
    )
    rec_yd_l4 = _trailing_4_per_player(ws_wr, "receiving_yards").rename(
        columns={"mean_l4": "receiving_yards_per_game_l4"}
    )
    rec_td_l4 = _trailing_4_per_player(ws_wr, "receiving_tds").rename(
        columns={"mean_l4": "receiving_tds_per_game_l4"}
    )
    rush_att_l4 = _trailing_4_per_player(ws_wr, "carries").rename(
        columns={"mean_l4": "rushing_attempts_per_game_l4"}
    )
    rush_yd_l4 = _trailing_4_per_player(ws_wr, "rushing_yards").rename(
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
            ws_this_season.groupby("gsis_id", as_index=False, observed=True)["targets"]
            .mean()
            .rename(columns={"targets": "targets_per_game_std"})
        )

    target_share = _trailing_4_share_per_team(ws_wr, "targets").rename(
        columns={"share_l4": "target_share_l4"}
    )
    air_yards_share = _trailing_4_share_per_team(ws_wr, "receiving_air_yards").rename(
        columns={"share_l4": "air_yards_share_l4"}
    )

    snap_l4 = _trailing_4_per_player(sc_wr, "offense_pct").rename(
        columns={"mean_l4": "snap_pct_l4"}
    )

    # --- NGS latest snapshot per player → *_std columns -------------------
    ngs_latest = _latest_ngs_snapshot(ngs)
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
        ngs_cols["percent_share_intended_air_yards_std"] = (
            ngs_cols["percent_share_intended_air_yards_std"].astype(float) / 100.0
        )

    # --- Game environment from schedules ---------------------------------
    game_env = _build_game_environment(sch)

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
    out = out.merge(target_share, on="gsis_id", how="left")
    out = out.merge(air_yards_share, on="gsis_id", how="left")
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
    for c in _ROLLING_ZERO_FILL_COLS:
        out[c] = out[c].fillna(0.0).astype(float)

    out["designed_rusher"] = out["rushing_attempts_per_game_l4"] >= _DESIGNED_RUSHER_THRESHOLD

    # depth_rank may have lost dtype through merges; restore nullable Int64
    # (the schema declares Series[int] but allows NaN via Pandera's promotion).
    out["depth_rank"] = out["depth_rank"].astype(pd.Int64Dtype())

    # Schema enforces Series[str] for team/opponent — our merges may have
    # introduced object dtype on those columns.
    for col in ("team", "opponent"):
        out[col] = out[col].astype(_PYARROW_STR)

    return WrFeaturesSchema.validate(out)
