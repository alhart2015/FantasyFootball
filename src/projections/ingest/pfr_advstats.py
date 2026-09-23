"""Refresh per-season PFR advanced charting from `nflreadpy.load_pfr_advstats`.

Parameterized by `stat_type` in {"pass", "rush", "rec"}; produces three partition tables
(`pfr_pass`, `pfr_rush`, `pfr_rec`), mirroring how `ngs` splits.

These are **charted** numbers, not derived ones: a human watched the play and recorded that the
quarterback was hurried, that the throw was bad, that the back broke a tackle. That is what makes
them worth ingesting alongside NGS, which is tracking-chip telemetry, and `weekly_stats`, which
is the box score. A sack is pressure that succeeded; `times_pressured` is pressure that was
applied, and the second carries forward to next week where the first mostly does not.

**Keyed on `pfr_player_id`, not `gsis_id`.** Resolved through `identity.resolve_gsis_via_id_map`,
the same crosswalk `snap_counts` uses, so `build_id_map` must have run first.

**`def` is deliberately not ingested here.** `load_pfr_advstats(stat_type="def")` is
individual-defender coverage and pass-rush data -- the right raw material for a real D/ST model,
which needs its own design work on team aggregation and opponent adjustment. Adding it as a
fourth table now would land data shaped for a model nobody has specified.

Usage:
    python -m projections.ingest.pfr_advstats --stat-type rush --seasons 2024 2025
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Final, Literal

import nflreadpy
import pandas as pd
import pandera.pandas as pa

from projections.ingest.identity import resolve_gsis_via_id_map
from projections.ingest.manifest import record as record_manifest
from projections.schemas import (
    _PYARROW_STR,
    PfrPassingSchema,
    PfrReceivingSchema,
    PfrRushingSchema,
    normalize_team_code,
)
from projections.store import write_partition

PfrStatType = Literal["pass", "rush", "rec"]
STAT_TYPES: Final[tuple[PfrStatType, ...]] = ("pass", "rush", "rec")

_KEEP_COMMON: Final[tuple[str, ...]] = ("gsis_id", "season", "week", "team", "opponent")

#: Each stat type's payload. Upstream ships every type's columns in every table, with the
#: irrelevant ones entirely null -- `pass` carries an all-null `receiving_drop`, `rec` carries an
#: all-null `passing_drops`. Selecting per type is what keeps a column that means something in
#: one table from becoming a column of nulls in another.
_KEEP_FOR: Final[dict[PfrStatType, tuple[str, ...]]] = {
    "pass": (
        *_KEEP_COMMON,
        "passing_drops",
        "passing_drop_pct",
        "passing_bad_throws",
        "passing_bad_throw_pct",
        "times_sacked",
        "times_blitzed",
        "times_hurried",
        "times_hit",
        "times_pressured",
        "times_pressured_pct",
    ),
    "rush": (
        *_KEEP_COMMON,
        "carries",
        "rushing_yards_before_contact",
        "rushing_yards_before_contact_avg",
        "rushing_yards_after_contact",
        "rushing_yards_after_contact_avg",
        "rushing_broken_tackles",
    ),
    "rec": (
        *_KEEP_COMMON,
        "receiving_broken_tackles",
        "receiving_drop",
        "receiving_drop_pct",
        "receiving_int",
        "receiving_rat",
    ),
}

_SCHEMA_FOR: Final[dict[PfrStatType, type[pa.DataFrameModel]]] = {
    "pass": PfrPassingSchema,
    "rush": PfrRushingSchema,
    "rec": PfrReceivingSchema,
}


def _fetch_raw_pfr_advstats(stat_type: PfrStatType, seasons: list[int]) -> pd.DataFrame:
    """Thin wrapper around the nflreadpy call; tests monkey-patch this."""
    return nflreadpy.load_pfr_advstats(
        seasons=seasons, stat_type=stat_type, summary_level="week"
    ).to_pandas()


def _normalize_team(v: str) -> str:
    return normalize_team_code(v).value


def _normalize_one_season(
    stat_type: PfrStatType, raw: pd.DataFrame, data_root: Path
) -> pd.DataFrame:
    df = raw.copy()

    # Drop rows missing pfr_player_id before the join, as snap_counts does.
    df = df[df["pfr_player_id"].notna()].copy()

    # Resolve pfr_player_id -> gsis_id via id_map. Unmatched rows (players we do not track at
    # all) are dropped; the count is logged below so a collapsed crosswalk is visible.
    n_before = len(df)
    df = resolve_gsis_via_id_map(df, data_root)
    _log_crosswalk_loss(stat_type, n_before=n_before, n_after=len(df))

    df = df[df["season"].notna() & df["week"].notna()].copy()
    # Upstream returns int32 for season/week; pandera Series[int] requires int64.
    for int_col in ("season", "week"):
        df[int_col] = df[int_col].astype("int64")

    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].map(_normalize_team).astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].map(_normalize_team).astype(_PYARROW_STR)

    keep = _KEEP_FOR[stat_type]
    for float_col in keep[len(_KEEP_COMMON) :]:
        df[float_col] = df[float_col].astype("float64")

    df = df[[c for c in keep if c in df.columns]].copy()
    # IMPORTANT: reassign -- strict="filter" returns a new DataFrame.
    df = _SCHEMA_FOR[stat_type].validate(df)
    return df


def _log_crosswalk_loss(stat_type: PfrStatType, *, n_before: int, n_after: int) -> None:
    """Report how much of the payload the id_map join dropped.

    Unmatched rows are expected and harmless in small numbers -- a deep-bench player nobody
    tracks. A large share means `id_map` is stale or upstream changed its id format, and that is
    invisible without a count: the partition simply comes out thin and still reports success.
    """
    lost = n_before - n_after
    if not lost:
        return
    share = lost / n_before if n_before else 0.0
    log = logging.getLogger(__name__)
    message = (
        "pfr_advstats (%s): %d of %d row(s) (%.1f%%) had no id_map entry for their "
        "pfr_player_id and were dropped."
    )
    if share > 0.25:
        log.warning(
            message + " That is high enough to suspect a stale id_map rather than bench players;"
            " re-run build_id_map before trusting this partition.",
            stat_type,
            lost,
            n_before,
            share * 100,
        )
    else:
        log.info(message, stat_type, lost, n_before, share * 100)


def refresh_pfr_advstats(
    data_root: Path,
    *,
    stat_type: PfrStatType,
    seasons: Iterable[int],
) -> list[Path]:
    """Fetch and write PFR advanced stats for `stat_type` and each season. Idempotent.

    Writes to `data/raw/pfr_{stat_type}/season=YYYY/part.parquet`. Requires `id_map.parquet`
    to already exist in `data_root/raw/` (built by `build_id_map`).
    """
    if stat_type not in STAT_TYPES:
        raise ValueError(f"stat_type must be one of {STAT_TYPES}, got {stat_type!r}")

    table = f"pfr_{stat_type}"
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_pfr_advstats(stat_type, [season])
        df = _normalize_one_season(stat_type, raw, data_root)
        path = write_partition(data_root / "raw", table, df, season=season, week=None)
        record_manifest(data_root, table=table, season=season, df=df)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stat-type", choices=STAT_TYPES, required=True)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    for path in refresh_pfr_advstats(
        args.data_root, stat_type=args.stat_type, seasons=args.seasons
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
