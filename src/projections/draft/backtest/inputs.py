"""Shared input loading for the H2H backtest (CLI + chunked runner).

Both entry points must build byte-identical pool / projection / actual / availability
inputs, so the loading lives here once rather than being duplicated per script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.backtest.draft_basis import build_draft_basis
from projections.draft.backtest.league import Calendar
from projections.draft.backtest.weekly_actuals import build_weekly_actuals
from projections.draft.league_config import LeagueConfig
from projections.schemas import ExternalProjectionSchema, WeeklyProjectionSchema
from projections.store import read_latest_partition, read_partition

# Weeks 1-14 regular season, 15-17 playoffs, top-6 bracket (spec §3, §5.6). Week 18 excluded.
DEFAULT_CALENDAR = Calendar(
    regular_weeks=tuple(range(1, 15)),
    playoff_weeks=(15, 16, 17),
    playoff_size=6,
)


def _attach_is_rookie(pool: pd.DataFrame, prior_gsis: set[str]) -> pd.DataFrame:
    """Return a copy of `pool` with a boolean `is_rookie` column: True for players absent from
    `prior_gsis` (the set of gsis_ids that appeared in any earlier season's weekly_stats).
    Undeterminable defaults to rookie only if truly never seen; a present player is veteran."""
    out = pool.copy()
    out["is_rookie"] = ~out["gsis_id"].astype(str).isin(prior_gsis)
    return out


def _prior_appearance_gsis(season: int, data_root: Path, *, since: int = 2018) -> set[str]:
    """gsis_ids appearing in any weekly_stats season in [since, season); missing partitions skip."""
    seen: set[str] = set()
    for yr in range(since, season):
        try:
            ws = read_partition(data_root / "raw", "weekly_stats", season=yr)
        except (FileNotFoundError, ValueError):
            continue
        seen.update(ws["gsis_id"].astype(str).tolist())
    return seen


@dataclass(frozen=True)
class BacktestInputs:
    pool: pd.DataFrame
    proj_lookup: dict[tuple[str, int], float]
    actual_lookup: dict[tuple[str, int], float]
    availability: PlayerAvailability
    calendar: Calendar


def load_inputs(*, season: int, config: LeagueConfig, data_root: Path) -> BacktestInputs:
    """Load every fixed input the backtest needs for ``season`` (deterministic given the store)."""
    external = ExternalProjectionSchema.validate(
        read_latest_partition(data_root / "raw", "external_projections", season=season)
    )
    pool = build_draft_basis(external, league_config=config)
    pool = _attach_is_rookie(pool, _prior_appearance_gsis(season, data_root))

    proj_df = WeeklyProjectionSchema.validate(
        read_partition(data_root / "processed", "espn_weekly_projections", season=season)
    )
    weekly_stats = read_partition(data_root / "raw", "weekly_stats", season=season)
    actual_df = build_weekly_actuals(weekly_stats, ruleset=config.ruleset)

    proj_lookup = {
        (str(r.gsis_id), int(r.week)): float(r.projected_points)
        for r in proj_df.itertuples(index=False)
        if pd.notna(r.projected_points)
    }
    actual_lookup = {
        (str(r.gsis_id), int(r.week)): float(r.actual_points)
        for r in actual_df.itertuples(index=False)
    }
    availability = load_store_availability(pool, season=season, data_root=data_root)
    return BacktestInputs(
        pool=pool,
        proj_lookup=proj_lookup,
        actual_lookup=actual_lookup,
        availability=availability,
        calendar=DEFAULT_CALENDAR,
    )
