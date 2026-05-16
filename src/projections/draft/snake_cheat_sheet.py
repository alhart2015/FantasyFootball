"""Snake-draft cheat sheet — per-position rankings with gap-based tiers.

Pure transform over the VORP table (`VorpTableSchema`). Reuses `_select_pool`
for in-pool identification. Emits `SnakeCheatSheetSchema`-validated output.

Spec: docs/superpowers/specs/2026-05-16-snake-cheat-sheet-design.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from projections.draft._pool import _select_pool
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, Position, SnakeCheatSheetSchema, VorpTableSchema

_POSITION_ORDER: tuple[Position, ...] = (
    Position.QB,
    Position.RB,
    Position.WR,
    Position.TE,
    Position.K,
    Position.DST,
)

_DISPLAY_NAME_FALLBACK = "—"


def _assign_tiers(
    vorp_desc: NDArray[np.float64],
    n_tiers: int,
) -> NDArray[np.int64]:
    """Gap-based tier assignment over a VORP-descending array.

    Returns a 1-indexed int64 array of the same length as `vorp_desc` giving
    the tier (1..N) for each in-pool player. See spec §3.2 for the algorithm.

    Tie-break: when multiple gaps share the value that is competing for the
    `(N-1)`th-largest slot, the earlier (higher-rank) gap-index wins.
    Deterministic.
    """
    n = len(vorp_desc)
    if n == 0:
        return np.array([], dtype=np.int64)
    if n <= n_tiers:
        return np.arange(1, n + 1, dtype=np.int64)

    gaps = vorp_desc[:-1] - vorp_desc[1:]
    # lexsort: primary key (last arg) sorts ascending; -gaps ascending = gaps
    # descending. Ties broken by np.arange (gap-index) ascending.
    order = np.lexsort((np.arange(n - 1), -gaps))
    cut_indices = np.sort(order[: n_tiers - 1])

    tier = np.empty(n, dtype=np.int64)
    start = 0
    for t, cut in enumerate(cut_indices, start=1):
        tier[start : cut + 1] = t
        start = int(cut) + 1
    tier[start:] = n_tiers
    return tier


def generate_snake_cheat_sheet(
    vorp_table: pd.DataFrame,
    league_config: LeagueConfig,
    display_names: pd.DataFrame | None = None,
    tiers_per_position: int = 8,
) -> pd.DataFrame:
    """Build a per-position snake-draft cheat sheet from a VORP table.

    Pure transform. See spec §3.1 for the three-stage algorithm:
    (1) call _select_pool to flag in-pool rows; (2) compute positional_rank
    across all rows by vorp desc; (3) gap-based tier breaks within each
    position's in-pool subset; out-of-pool rows get tier=NA.

    Args:
      vorp_table: VorpTableSchema-validated frame.
      league_config: drives _select_pool's in-pool definition.
      display_names: optional (gsis_id, display_name) frame. If None, all
        display_name values become '—'.
      tiers_per_position: positive int; default 8.

    Returns: SnakeCheatSheetSchema-validated frame, sorted by
      (position canonical, positional_rank).
    """
    if tiers_per_position <= 0:
        raise ValueError(f"tiers_per_position must be >= 1; got {tiers_per_position}")

    vorp = VorpTableSchema.validate(vorp_table)

    # Stage 1: in-pool flag via _select_pool (which itself enforces config-
    # required positions present, raising "cannot fill N {slot} slots" if not).
    in_pool_ids = set(_select_pool(vorp, league_config))
    df = vorp.copy()
    df["is_in_pool"] = df["gsis_id"].isin(in_pool_ids)

    # Stage 2: positional_rank across all rows (in-pool + out), by vorp desc
    # with gsis_id ascending tie-break (matches _select_pool tie-break).
    df = df.sort_values(["position", "vorp", "gsis_id"], ascending=[True, False, True])
    df["positional_rank"] = df.groupby("position", sort=False).cumcount() + 1
    df["positional_rank"] = df["positional_rank"].astype(pd.Int64Dtype())

    # Stage 3: gap-based tiers within each position's in-pool subset.
    tier_col = pd.array([pd.NA] * len(df), dtype=pd.Int64Dtype())
    df = df.reset_index(drop=True)
    for pos_value in df["position"].unique():
        pos_mask = df["position"] == pos_value
        in_pool_mask = pos_mask & df["is_in_pool"]
        in_pool_idx = df.index[in_pool_mask].to_numpy()
        if len(in_pool_idx) == 0:
            continue
        # in_pool_idx is already in positional_rank order because we sorted
        # the whole frame by (position, vorp desc, gsis_id) above.
        vorps = df.loc[in_pool_idx, "vorp"].to_numpy(dtype=np.float64)
        tiers = _assign_tiers(vorps, tiers_per_position)
        for idx, t in zip(in_pool_idx, tiers, strict=True):
            tier_col[idx] = int(t)
    df["tier"] = tier_col

    # Display names: left-join optional map; fallback "—".
    if display_names is None or display_names.empty:
        df["display_name"] = pd.Series([_DISPLAY_NAME_FALLBACK] * len(df), dtype=_PYARROW_STR)
    else:
        names = display_names.set_index("gsis_id")["display_name"]
        mapped = df["gsis_id"].map(names).fillna(_DISPLAY_NAME_FALLBACK)
        df["display_name"] = mapped.astype(_PYARROW_STR)

    # Final sort: position canonical order, then positional_rank ascending.
    position_rank = {pos.value: i for i, pos in enumerate(_POSITION_ORDER)}
    df["_pos_sort"] = df["position"].map(position_rank)
    df = df.sort_values(["_pos_sort", "positional_rank"], ascending=[True, True])
    df = df.drop(columns=["_pos_sort"]).reset_index(drop=True)

    # Column order: align to schema declaration order so downstream consumers
    # (and the round-trip test) get a stable layout. Pandera's strict="filter"
    # drops extras but does not reorder.
    df = df[list(SnakeCheatSheetSchema.to_schema().columns)]

    return SnakeCheatSheetSchema.validate(df)
