"""The only place that knows the pick'em partition layout.

    data/pickem/sheet/season=YYYY/week=WW/part.parquet
    data/pickem/picks/season=YYYY/week=WW/part.parquet

Picks are stored under the same partition through their whole life: written on
Thursday when entered, overwritten on Monday once graded. `write_partition` is
idempotent, so re-grading a week replaces it rather than appending.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.schemas import PickemPicksSchema, PickemSheetSchema
from projections.store import read_partition, write_partition

SHEET_TABLE = "sheet"
PICKS_TABLE = "picks"


def _pickem_root(data_root: Path) -> Path:
    return data_root / "pickem"


def _sole_season_week(df: pd.DataFrame, *, table: str) -> tuple[int, int]:
    """A partition holds exactly one week; a frame spanning several would
    silently write only the last one's coordinates."""
    pairs = df[["season", "week"]].drop_duplicates()
    if len(pairs) != 1:
        raise ValueError(
            f"{table} frame must cover exactly one (season, week); got "
            f"{sorted(map(tuple, pairs.to_numpy().tolist()))}"
        )
    return int(pairs.iloc[0]["season"]), int(pairs.iloc[0]["week"])


def write_sheet(data_root: Path, sheet: pd.DataFrame) -> Path:
    """Persist the organizer's sheet for one week."""
    validated: pd.DataFrame = PickemSheetSchema.validate(sheet)
    season, week = _sole_season_week(validated, table=SHEET_TABLE)
    return write_partition(
        _pickem_root(data_root), SHEET_TABLE, validated, season=season, week=week
    )


def read_sheet_partition(data_root: Path, *, season: int, week: int) -> pd.DataFrame:
    """Read back a stored organizer sheet."""
    df = read_partition(_pickem_root(data_root), SHEET_TABLE, season=season, week=week)
    validated: pd.DataFrame = PickemSheetSchema.validate(df)
    return validated


def write_picks(data_root: Path, picks: pd.DataFrame) -> Path:
    """Persist picks for one week, graded or not."""
    validated: pd.DataFrame = PickemPicksSchema.validate(picks)
    season, week = _sole_season_week(validated, table=PICKS_TABLE)
    return write_partition(
        _pickem_root(data_root), PICKS_TABLE, validated, season=season, week=week
    )


def read_picks(data_root: Path, *, season: int, week: int) -> pd.DataFrame:
    """Read back stored picks for one week."""
    df = read_partition(_pickem_root(data_root), PICKS_TABLE, season=season, week=week)
    validated: pd.DataFrame = PickemPicksSchema.validate(df)
    return validated


def read_picks_season(data_root: Path, *, season: int) -> pd.DataFrame:
    """Read every stored week of picks for a season — the season-long record."""
    df = read_partition(_pickem_root(data_root), PICKS_TABLE, season=season)
    validated: pd.DataFrame = PickemPicksSchema.validate(df)
    return validated
