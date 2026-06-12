"""Value Over Replacement Player generator.

Public surface: `generate_vorp_table(season_projections, league_config)`.

Starter-demand replacement-level: for each position present in both the input
season-projections and the LeagueConfig.roster_slots, `replacement_fpts(pos)` is
the `season_mean_fpts` of the player at the *cushioned starter-demand* rank --
`round(bench_cushion * starter_demand(pos))`, where `starter_demand(pos)` is how
many players the position fills in the STARTER pool (dedicated + FLEX + SUPER_FLEX,
bench excluded). This is deliberately independent of bench depth: a deeper bench
must not make a position's replacement deeper. Rows at positions absent from
`LeagueConfig.roster_slots` are dropped.

The cushion (default 1.3) is a small backup allowance over pure starter demand --
e.g. some teams carry a backup QB. It is a placeholder for an empirically-fit value
(from prior league draft histories); exposed as `bench_cushion` so a future slice
can tune it without touching this module.

Spec: docs/superpowers/specs/2026-05-16-vorp-design.md
"""

from __future__ import annotations

import pandas as pd

from projections.draft._pool import (
    _reject_duplicate_gsis_ids,
    _select_pool,
)
from projections.draft.league_config import LeagueConfig
from projections.draft.roster_eligibility import bench_eligible_positions
from projections.schemas import (
    _PYARROW_STR,
    ProjectionSeasonSchema,
    RosterSlot,
    VorpTableSchema,
)

# Backup allowance over starter demand for the replacement level. Placeholder for an
# empirically-fit value (prior-season draft histories) -- TODO follow-up. Overridable
# per call via generate_vorp_table(..., bench_cushion=...).
_BENCH_CUSHION = 1.3

_OUTPUT_COLUMNS: tuple[str, ...] = (
    "gsis_id",
    "position",
    "season_mean_fpts",
    "vorp",
    "replacement_fpts",
)


def _starter_demand(df: pd.DataFrame, league_config: LeagueConfig) -> dict[str, int]:
    """Per-position count in the STARTER pool (dedicated + FLEX + SUPER_FLEX, no bench).

    Reuses `_select_pool` on a bench-stripped league so FLEX/SUPER_FLEX allocate to
    positions exactly as in a real draft. Bench is excluded on purpose: the replacement
    level must track *starter* demand, not how deep the bench is. Propagates
    `_select_pool`'s "cannot fill N <slot> slots" ValueError when a required starter slot
    has too few players (same failure surface as before, minus the bench slots).
    """
    starter_slots = {
        slot: count
        for slot, count in league_config.roster_slots.items()
        if slot != RosterSlot.BENCH and count > 0
    }
    if not starter_slots:
        return {}
    starter_cfg = league_config.model_copy(update={"roster_slots": starter_slots})
    pool_input = df[["gsis_id", "position", "season_mean"]].rename(
        columns={"season_mean": "season_mean_fpts"}
    )
    pool_input["vorp"] = 0.0
    starter_ids = set(_select_pool(pool_input, starter_cfg))
    counts = df[df["gsis_id"].isin(starter_ids)]["position"].value_counts()
    return {str(pos): int(n) for pos, n in counts.items()}


def _replacement_level(
    pos_means_desc: pd.Series, starter_demand: int, bench_cushion: float
) -> float:
    """Replacement = season_mean of the player at rank `round(bench_cushion * starter_demand)`.

    The player at that (cushioned) rank is the last roster-worthy one at the position, so
    it lands at VORP 0 and everyone above is positive. Falls back to the minimum projection
    when the cushioned demand meets or exceeds the supply (every player is roster-worthy).
    """
    rank = round(bench_cushion * starter_demand)
    if rank <= 0 or rank >= len(pos_means_desc):
        return float(pos_means_desc.iloc[-1])
    return float(pos_means_desc.iloc[rank - 1])


def _in_scope_positions(league_config: LeagueConfig) -> frozenset[str]:
    """The set of `Position.value`s the league cares about.

    Excludes FLEX / SUPER_FLEX / BENCH / IR (those slots accept multiple
    positions or are not draft-position slots at all). Delegates to the shared
    bench-eligibility rule so VORP and the pool selector cannot drift.
    """
    return frozenset(pos.value for pos in bench_eligible_positions(league_config.roster_slots))


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
    *,
    bench_cushion: float = _BENCH_CUSHION,
) -> pd.DataFrame:
    """Convert per-player season projections into a VORP table under `league_config`.

    Returns a DataFrame validated against `VorpTableSchema`. One row per input
    player whose `position` is in `league_config.roster_slots`. Rows at out-of-scope
    positions are dropped.

    `replacement_fpts(pos)` is the projection at the cushioned starter-demand rank
    (`round(bench_cushion * starter_demand(pos))`), so bench depth never deepens a
    position's replacement. `bench_cushion` (default 1.3) is the backup allowance over
    pure starter demand; override it to tune (e.g. an empirically-fit value).

    Raises:
        ValueError: If `season_projections` contains mixed rulesets, a ruleset
            that does not match `league_config.ruleset.name`, multiple seasons,
            or duplicate `gsis_id` rows (enforced by `_validate_input`); or if a
            required starter slot cannot be filled (from `_select_pool`).
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

    demand = _starter_demand(df, league_config)

    replacement_by_position: dict[str, float] = {}
    for pos in in_scope:
        pos_means = (
            df[df["position"] == pos]["season_mean"].astype("float64").sort_values(ascending=False)
        )
        if pos_means.empty:
            continue
        replacement_by_position[pos] = _replacement_level(
            pos_means, demand.get(pos, 0), bench_cushion
        )

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
