"""Refresh per-season expected fantasy production from `nflreadpy.load_ff_opportunity`.

nflverse's `ffopportunity` model scores every play for what it was *worth*, then sums per
player-week. That gives an expected value beside each actual: expected receptions, expected
yards, expected touchdowns. The gap between the two is the only luck-regression signal in this
repo -- `weekly_stats` cannot tell a three-catch game that ended in a 60-yard catch-and-run from
a nine-target grind, and they are very different evidence about next week.

**Stats, not points.** Upstream ships its own `total_fantasy_points*`; those columns are stored
for diagnostic comparison and nothing else. `src/projections/scoring/` converts the `_exp` stat
columns under our league's ruleset -- see `FfOpportunitySchema`.

Upstream dtypes here are looser than anywhere else in this repo and the normalization order
matters; `_normalize_one_season` documents each step at the point it happens.

Usage:
    python -m projections.ingest.ff_opportunity --seasons 2024 2025
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import nflreadpy
import pandas as pd

from projections.ingest.identity import drop_placeholder_gsis_rows
from projections.ingest.manifest import record as record_manifest
from projections.schemas import (
    _PYARROW_STR,
    FfOpportunitySchema,
    Position,
    normalize_team_code,
)
from projections.store import write_partition

_RENAME: Final[dict[str, str]] = {"player_id": "gsis_id", "posteam": "team"}

#: The actual/expected pairs we keep, plus the opportunity counts they are built on. Upstream
#: ships 159 columns; the `_diff` family is a subtraction of two of these and the `*_team`
#: family is a groupby of them, so neither is stored. See `FfOpportunitySchema`.
_STAT_COLS: Final[tuple[str, ...]] = (
    "pass_attempt",
    "rec_attempt",
    "rush_attempt",
    "pass_completions",
    "pass_completions_exp",
    "receptions",
    "receptions_exp",
    "pass_yards_gained",
    "pass_yards_gained_exp",
    "rec_yards_gained",
    "rec_yards_gained_exp",
    "rush_yards_gained",
    "rush_yards_gained_exp",
    "pass_touchdown",
    "pass_touchdown_exp",
    "rec_touchdown",
    "rec_touchdown_exp",
    "rush_touchdown",
    "rush_touchdown_exp",
    "pass_interception",
    "pass_interception_exp",
    "total_fantasy_points",
    "total_fantasy_points_exp",
)

_KEEP: Final[tuple[str, ...]] = ("gsis_id", "season", "week", "team", "position", *_STAT_COLS)


def _fetch_raw_ff_opportunity(seasons: list[int]) -> pd.DataFrame:
    """Thin wrapper around the nflreadpy call; tests monkey-patch this."""
    return nflreadpy.load_ff_opportunity(seasons=seasons, stat_type="weekly").to_pandas()


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_RENAME).copy()

    # 1. Team-level aggregate rows carry no player and no position. They must go before the
    #    gsis pattern check, which would otherwise see ~420 nulls per season.
    df = drop_placeholder_gsis_rows(df, source="ff_opportunity")

    # 2. Upstream returns `season` as a *string* and `week` as a *float* -- unique to this
    #    source; every other nflverse release here gives int32. Neither survives a
    #    `Series[int]` schema without an explicit cast, and the float must go via a
    #    null-drop first because `astype("int64")` on NaN raises rather than propagating.
    df = df[df["season"].notna() & df["week"].notna()].copy()
    df["season"] = df["season"].astype("int64")
    df["week"] = df["week"].astype("float64").astype("int64")

    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].map(lambda v: normalize_team_code(v).value).astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)

    # 3. The model scores every player who touched the ball, so `position` carries DB, LB, DL,
    #    OL and P alongside the six we model. Filter before validation.
    df = df[df["position"].isin([p.value for p in Position])].copy()

    for stat_col in _STAT_COLS:
        df[stat_col] = df[stat_col].astype("float64")

    df = df[[c for c in _KEEP if c in df.columns]].copy()
    # IMPORTANT: reassign -- strict="filter" returns a new DataFrame.
    df = FfOpportunitySchema.validate(df)
    return df


def refresh_ff_opportunity(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write expected production for each season. One partition per season. Idempotent."""
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_ff_opportunity([season])
        df = _normalize_one_season(raw)
        path = write_partition(data_root / "raw", "ff_opportunity", df, season=season, week=None)
        record_manifest(data_root, table="ff_opportunity", season=season, df=df)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    for path in refresh_ff_opportunity(args.data_root, seasons=args.seasons):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
