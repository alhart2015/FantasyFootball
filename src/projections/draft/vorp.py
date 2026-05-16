"""Value Over Replacement Player generator.

Public surface: `generate_vorp_table(season_projections, league_config)`.

Pool-boundary replacement-level: for each position present in both the input
season-projections and the LeagueConfig.roster_slots, `replacement_fpts(pos)`
is the `season_mean_fpts` of the best non-pool player at that position, where
"pool" is the auction-values `_select_pool` output. Rows at positions absent
from `LeagueConfig.roster_slots` are dropped.

Spec: docs/superpowers/specs/2026-05-16-vorp-design.md
"""

from __future__ import annotations

import pandas as pd

from projections.draft._pool import (
    _POSITION_SLOTS,
    _reject_duplicate_gsis_ids,
    _select_pool,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import (
    _PYARROW_STR,
    ProjectionSeasonSchema,
    VorpTableSchema,
)

_OUTPUT_COLUMNS: tuple[str, ...] = (
    "gsis_id",
    "position",
    "season_mean_fpts",
    "vorp",
    "replacement_fpts",
)


def _in_scope_positions(league_config: LeagueConfig) -> frozenset[str]:
    """The set of `Position.value`s the league cares about.

    Excludes FLEX / SUPER_FLEX / BENCH / IR (those slots accept multiple
    positions or are not draft-position slots at all).
    """
    return frozenset(
        slot.value
        for slot in league_config.roster_slots
        if slot in _POSITION_SLOTS and league_config.roster_slots[slot] > 0
    )


def _validate_input(season_projections: pd.DataFrame, league_config: LeagueConfig) -> pd.DataFrame:
    df = ProjectionSeasonSchema.validate(season_projections)
    if df.empty:
        return df

    rulesets = df["ruleset"].unique()
    if len(rulesets) > 1:
        raise ValueError(f"season_projections contains mixed rulesets: {sorted(rulesets)}")
    observed_ruleset = str(rulesets[0])
    if observed_ruleset != league_config.ruleset.name:
        raise ValueError(
            f"season_projections ruleset {observed_ruleset!r} does not match "
            f"league_config.ruleset {league_config.ruleset.name!r}"
        )

    seasons = df["season"].unique()
    if len(seasons) > 1:
        raise ValueError(
            f"season_projections contains multiple seasons: {sorted(seasons.tolist())}"
        )

    _reject_duplicate_gsis_ids(df, "season_projections")
    return df


def generate_vorp_table(
    season_projections: pd.DataFrame,
    league_config: LeagueConfig,
) -> pd.DataFrame:
    """Convert per-player season projections into a VORP table under `league_config`.

    Returns a DataFrame validated against `VorpTableSchema`. One row per input
    player whose `position` is in `league_config.roster_slots`. Rows at out-of-scope
    positions are dropped. See spec §3 for the pool-boundary algorithm.

    Raises:
        ValueError: If `season_projections` contains mixed rulesets, a ruleset
            that does not match `league_config.ruleset.name`, multiple seasons,
            or duplicate `gsis_id` rows. These preconditions are enforced by
            `_validate_input`.
    """
    df = _validate_input(season_projections, league_config)

    in_scope = _in_scope_positions(league_config)
    df = df[df["position"].isin(in_scope)]

    if df.empty:
        empty = pd.DataFrame(columns=list(_OUTPUT_COLUMNS))
        empty["gsis_id"] = empty["gsis_id"].astype(_PYARROW_STR)
        empty["position"] = empty["position"].astype(_PYARROW_STR)
        for col in ("season_mean_fpts", "vorp", "replacement_fpts"):
            empty[col] = empty[col].astype("float64")
        return VorpTableSchema.validate(empty)

    # ProjectionSeasonSchema uses `season_mean`; the auction layer (and _select_pool)
    # uses `season_mean_fpts`. Bridge here so all downstream code sees one name.
    pool_input = df[["gsis_id", "position", "season_mean"]].rename(
        columns={"season_mean": "season_mean_fpts"}
    )
    pool_input["vorp"] = 0.0
    pool_ids = set(_select_pool(pool_input, league_config))

    replacement_by_position: dict[str, float] = {}
    for pos in in_scope:
        pos_rows = df[df["position"] == pos]
        if pos_rows.empty:
            continue
        non_pool = pos_rows[~pos_rows["gsis_id"].isin(pool_ids)]
        if non_pool.empty:
            # All players at this position are in the pool — replacement is the
            # min projection (bottom player lands at VORP 0). Spec §3.1 step 3.
            replacement_by_position[pos] = float(pos_rows["season_mean"].min())
        else:
            replacement_by_position[pos] = float(non_pool["season_mean"].max())

    season_mean_fpts = df["season_mean"].astype("float64").to_numpy()
    replacement_fpts = df["position"].map(replacement_by_position).astype("float64").to_numpy()
    out = pd.DataFrame(
        {
            "gsis_id": df["gsis_id"].astype(_PYARROW_STR).to_numpy(),
            "position": df["position"].astype(_PYARROW_STR).to_numpy(),
            "season_mean_fpts": season_mean_fpts,
            "vorp": season_mean_fpts - replacement_fpts,
            "replacement_fpts": replacement_fpts,
        }
    )
    return VorpTableSchema.validate(out[list(_OUTPUT_COLUMNS)])


__all__ = ["generate_vorp_table"]
