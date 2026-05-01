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


def compute_receiver_yac_per_reception(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-receiver mean yards-after-catch per completion, per receiver-active
    game.

    Per (gsis_id, season, week): mean of ``yards_after_catch`` across rows
    where ``receiver_player_id == gsis_id`` AND ``complete_pass == 1.0`` AND
    ``yards_after_catch.notna()``. Filtered to completions — YAC only exists
    when the ball is caught.

    Receivers with no catches in a week contribute no per-game row.

    Output: (gsis_id, season, week, yac_per_reception_l4) — one row per
    receiver-active game.
    """
    completions = pbp[
        (pbp["receiver_player_id"].notna())
        & (pbp["complete_pass"] == 1.0)
        & (pbp["yards_after_catch"].notna())
    ]
    per_game = (
        completions.groupby(["receiver_player_id", "season", "week"], as_index=False)[
            "yards_after_catch"
        ]
        .mean()
        .rename(
            columns={
                "receiver_player_id": "gsis_id",
                "yards_after_catch": "yac_per_reception_l4",
            }
        )
    )
    return per_game[["gsis_id", "season", "week", "yac_per_reception_l4"]]


def compute_receiver_red_zone_target_share(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-receiver share of targets at yardline_100 <= 20, per
    receiver-active game.

    Per (gsis_id, season, week):
      total_targets = count rows where receiver_player_id == gsis_id AND
                      pass_attempt == 1.0 AND yardline_100.notna()
      rz_targets    = count rows where receiver_player_id == gsis_id AND
                      pass_attempt == 1.0 AND yardline_100 <= 20
      share         = rz_targets / total_targets

    yardline_100 = 20 is the standard NFL red-zone definition (yards from
    the opponent's goal line). This is the receiver's RZ target share, not
    the team's RZ target rate; captures whether the player is the offense's
    preferred end-zone target.

    Output: (gsis_id, season, week, red_zone_target_share_l4) — one row per
    receiver-active game.
    """
    targets = pbp[
        (pbp["receiver_player_id"].notna())
        & (pbp["pass_attempt"] == 1.0)
        & (pbp["yardline_100"].notna())
    ].copy()
    targets["_is_rz"] = (targets["yardline_100"] <= _RED_ZONE_YARDLINE).astype("float64")
    per_game = (
        targets.groupby(["receiver_player_id", "season", "week"], as_index=False)["_is_rz"]
        .mean()
        .rename(
            columns={
                "receiver_player_id": "gsis_id",
                "_is_rz": "red_zone_target_share_l4",
            }
        )
    )
    return per_game[["gsis_id", "season", "week", "red_zone_target_share_l4"]]


_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "aDOT_l4",
    "deep_target_share_l4",
    "yac_per_reception_l4",
    "red_zone_target_share_l4",
)


def attach_pbp_receiver_features(
    index: pd.DataFrame,
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the 4 PBP receiver features to a (gsis_id, season, week) index.

    Args:
        index: ``(gsis_id, season, week)`` — one row per receiver-week.
            Built from ``depth_charts`` filtered to position in {WR, TE}.
        pbp: PBP frame matching ``PbpSchema``, projected to or wider than
            the receiver-features column set. Must include the seasons
            spanning the index plus one prior season for trailing-4 backfill.

    Returns:
        A copy of ``index`` with 4 columns appended in order:
        ``aDOT_l4``, ``deep_target_share_l4``, ``yac_per_reception_l4``,
        ``red_zone_target_share_l4``. Row count equals ``len(index)``.
        All 4 columns are float64 (NaN where trailing-4 has fewer than 4
        prior receiver-active games or the player has no PBP rows at all).

    All four computes key on ``receiver_player_id``; no team / opponent
    join required.

    Empty ``pbp`` short-circuits to all-NaN columns — same shape as a
    successful call where every row's trailing-4 has fewer than 4 prior
    receiver-active games. Schema ``nullable=True`` covers this.
    """
    if pbp.empty:
        out = index.copy()
        for col in _OUTPUT_COLUMNS:
            out[col] = float("nan")
        return out.reset_index(drop=True)

    pbp_proj = pbp[list(_PBP_COLUMNS_USED)]
    adot_pg = compute_receiver_adot(pbp_proj)
    deep_pg = compute_receiver_deep_target_share(pbp_proj)
    yac_pg = compute_receiver_yac_per_reception(pbp_proj)
    rz_pg = compute_receiver_red_zone_target_share(pbp_proj)

    out = index.copy()
    for per_game, col in (
        (adot_pg, "aDOT_l4"),
        (deep_pg, "deep_target_share_l4"),
        (yac_pg, "yac_per_reception_l4"),
        (rz_pg, "red_zone_target_share_l4"),
    ):
        attached = _trailing_4_per_player_asof(per_game, index, value_col=col, out_col=col)
        # `attached` has one row per index row, keyed by (gsis_id, season,
        # week). The index is presumed unique on that key per the
        # depth-chart-derived contract upstream; validate="one_to_one"
        # converts that assumption into a runtime guard.
        out = out.merge(
            attached,
            on=["gsis_id", "season", "week"],
            how="left",
            validate="one_to_one",
        )

    return out[["gsis_id", "season", "week", *_OUTPUT_COLUMNS]].reset_index(drop=True)


def build_pbp_receiver_overrides(
    pbp: pd.DataFrame,
    receiver_index: pd.DataFrame,
) -> pd.DataFrame:
    """Public assembler. Returns the 4-column override frame ready to write.

    Args:
        pbp: PBP frame matching ``PbpSchema``. Must include the seasons
            spanning the index plus one prior season for trailing-4 backfill.
        receiver_index: ``(gsis_id, season, week)`` — one row per
            receiver-week. Built by the override script from ``depth_charts``
            filtered to ``position in {WR, TE}``.

    Returns:
        ``(gsis_id, season, week, aDOT_l4, deep_target_share_l4,
        yac_per_reception_l4, red_zone_target_share_l4)`` — one row per
        input index row.

    Raises:
        ValueError: gsis_id format violations or duplicate
            (gsis_id, season, week) keys in the index.
        AssertionError: row-count mismatch after merges (internal-invariant
            violation; a future compute regression that introduces duplicate
            (gsis_id, season, week) keys would trigger this).

    Per-position coverage validation is the probe's responsibility; see
    spec §1.3 criterion 1 + §3.3 step 2.
    """
    bad_ids = [g for g in receiver_index["gsis_id"].dropna() if not _GSIS_RE.match(str(g))]
    if bad_ids:
        raise ValueError(
            f"invalid gsis_id format(s): {bad_ids[:3]} (and {max(0, len(bad_ids) - 3)} more)"
        )

    dup_mask = receiver_index.duplicated(subset=["gsis_id", "season", "week"], keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        raise ValueError(f"duplicate (gsis_id, season, week) keys in index: {n_dup} rows")

    out = attach_pbp_receiver_features(receiver_index, pbp)

    if len(out) != len(receiver_index):
        raise AssertionError(
            f"row count mismatch: input index had {len(receiver_index)} rows, "
            f"output has {len(out)}; suggests a many-to-many merge regression"
        )

    return out[["gsis_id", "season", "week", *_OUTPUT_COLUMNS]].reset_index(drop=True)
