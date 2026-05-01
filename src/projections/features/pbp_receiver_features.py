"""PBP-derived player-level (receiver) features for the WR/TE PBP family probe.

Pure-pandas computes consumed by build_pbp_receiver_overrides (this module's
public assembler). Each compute returns a (gsis_id, season, week, <metric>_l4)
frame attached to a depth-chart-derived (gsis_id, season, week) index via an
as-of trailing-4 lookup over the player's prior receiver-active games.

The trailing-4 backfill across season boundaries is handled by the caller
feeding multiple seasons of PBP concat'd together — see
scripts/build_pbp_receiver_override.py.

Spec: docs/superpowers/specs/2026-05-01-wr-te-pbp-receiver-features-design.md §6.1.
"""

from __future__ import annotations

import re
from typing import Final

import pandas as pd

from projections.schemas import GSIS_ID_PATTERN

_GSIS_RE: Final[re.Pattern[str]] = re.compile(rf"^{GSIS_ID_PATTERN}$")
_DEEP_AIR_YARDS: Final[float] = 20.0
_RED_ZONE_YARDLINE: Final[int] = 20
_PBP_COLUMNS_USED: Final[tuple[str, ...]] = (
    "receiver_player_id",
    "season",
    "week",
    "pass_attempt",
    "complete_pass",
    "air_yards",
    "yards_after_catch",
    "yardline_100",
)


def _trailing_4_per_player_asof(
    per_game: pd.DataFrame,
    index: pd.DataFrame,
    *,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    """Attach a trailing-4-receiver-active-game mean to a wider index via as-of join.

    Args:
        per_game: (gsis_id, season, week, value_col) — one row per
            receiver-active game with the per-game stat.
        index: (gsis_id, season, week) — one row per (rostered) player-week
            in the override; may include weeks where the player was not
            receiver-active.
        value_col: the per-game stat column name in ``per_game``.
        out_col: the output column name to attach to ``index``.

    Returns:
        A (gsis_id, season, week, out_col) frame with one row per input
        ``index`` row. ``out_col`` is the mean of the player's last 4
        receiver-active games strictly before (season, week); NaN if the
        player has fewer than 4 prior receiver-active games.

    Implementation: rolling-4 mean at per_game (no shift; "value at row N =
    mean of N-3 through N inclusive"), then merge_asof with
    direction='backward' and allow_exact_matches=False. The strict
    less-than semantics ensure (s, w) for an index row sees only games
    chronologically before it.
    """
    # 1. Sort per_game and compute rolling-4 (no shift) within player.
    pg = per_game.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True).copy()
    pg["_rolled"] = pg.groupby("gsis_id", sort=False)[value_col].transform(
        lambda s: s.rolling(window=4, min_periods=4).mean()
    )

    # 2. Build a sortable timestamp from (season, week) — season*100+week is
    # monotonic provided week <= 22 (PbpSchema enforces).
    pg["_t"] = pg["season"].astype("int64") * 100 + pg["week"].astype("int64")

    # 3. As-of merge; preserve the original index row order via reset_index.
    idx = index.copy()
    idx["_t"] = idx["season"].astype("int64") * 100 + idx["week"].astype("int64")
    idx_sorted = idx.sort_values("_t").reset_index().rename(columns={"index": "_orig_idx"})
    pg_sorted = pg[["gsis_id", "_t", "_rolled"]].sort_values("_t")

    merged = pd.merge_asof(
        idx_sorted,
        pg_sorted,
        on="_t",
        by="gsis_id",
        direction="backward",
        allow_exact_matches=False,
    )

    out = merged.sort_values("_orig_idx").drop(columns=["_t", "_orig_idx"])
    out = out.rename(columns={"_rolled": out_col})
    return out[["gsis_id", "season", "week", out_col]].reset_index(drop=True)


def compute_receiver_adot(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-receiver mean depth of target, per receiver-active game.

    Per (gsis_id, season, week): mean of ``air_yards`` across rows where
    ``receiver_player_id == gsis_id`` AND ``pass_attempt == 1.0`` AND
    ``air_yards.notna()``. NaN ``air_yards`` (sacks, throw-aways, no-plays)
    excluded.

    Output is the per-game mean frame, NOT the trailing-4 lookup. The
    assembler ``attach_pbp_receiver_features`` calls
    ``_trailing_4_per_player_asof`` to compute the trailing-4 against an
    index. The output column is named ``aDOT_l4`` to match the final
    override-column name; the helper preserves the name.

    Output: (gsis_id, season, week, aDOT_l4) — one row per receiver-active
    game (player had at least 1 valid target).
    """
    plays = pbp[
        (pbp["receiver_player_id"].notna())
        & (pbp["pass_attempt"] == 1.0)
        & (pbp["air_yards"].notna())
    ]
    per_game = (
        plays.groupby(["receiver_player_id", "season", "week"], as_index=False)["air_yards"]
        .mean()
        .rename(columns={"receiver_player_id": "gsis_id", "air_yards": "aDOT_l4"})
    )
    return per_game[["gsis_id", "season", "week", "aDOT_l4"]]


def compute_receiver_deep_target_share(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-receiver share of targets with air_yards >= 20, per
    receiver-active game.

    Per (gsis_id, season, week):
      total_valid_targets = count rows where receiver_player_id == gsis_id
                            AND pass_attempt == 1.0 AND air_yards.notna()
      deep_targets        = count rows where receiver_player_id == gsis_id
                            AND pass_attempt == 1.0 AND air_yards >= 20.0
      share               = deep_targets / total_valid_targets

    A receiver with zero valid targets in a week contributes no per-game row
    (no division-by-zero; the player simply doesn't appear in per_game for
    that week). The 20-yard cutoff is the conventional "deep" threshold.

    Output: (gsis_id, season, week, deep_target_share_l4) — one row per
    receiver-active game.
    """
    valid = pbp[
        (pbp["receiver_player_id"].notna())
        & (pbp["pass_attempt"] == 1.0)
        & (pbp["air_yards"].notna())
    ].copy()
    valid["_is_deep"] = (valid["air_yards"] >= _DEEP_AIR_YARDS).astype("float64")
    per_game = (
        valid.groupby(["receiver_player_id", "season", "week"], as_index=False)["_is_deep"]
        .mean()
        .rename(columns={"receiver_player_id": "gsis_id", "_is_deep": "deep_target_share_l4"})
    )
    return per_game[["gsis_id", "season", "week", "deep_target_share_l4"]]
