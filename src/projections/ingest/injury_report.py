"""Refresh the per-week NFL injury report from `nflreadpy.load_injuries`.

Writes one parquet partition per season. What the report carries that nothing else here does is
the pair of columns: the **designation** (Questionable/Doubtful/Out) that the team commits to
before Sunday, and the **practice participation** that led to it. A Questionable who practised
in full and a Questionable who did not practise all week are the same designation and very
different players, and only this source can tell them apart.

**Not `ingest.injury_news`.** That module fetches ESPN's beat-reporter write-up behind a
designation -- prose, for a human to overrule a number with. This is the structured weekly
report: no text, every player, joinable on `(gsis_id, season, week)`.

**Healthy players have no row.** Absence from this table is the healthy signal. Nothing here
manufactures a "healthy" row, and no consumer should read a missing row as a missing
measurement -- see `InjuryReportSchema` for what each flavour of null means.

`scripts/measure_injury_impact.py` predates this module and still calls `load_injuries`
directly, with a *different* filter (it keeps postseason rows). Pointing it at this partition
would change the constants in `midseason.injuries` that it produced, so it is deliberately left
alone here and tracked separately.

Usage:
    python -m projections.ingest.injury_report --seasons 2024 2025
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
    InjuryReportSchema,
    InjuryStatus,
    Position,
    PracticeStatus,
    normalize_team_code,
)
from projections.store import write_partition

_log = logging.getLogger(__name__)

_KEEP: Final = [
    "gsis_id",
    "season",
    "week",
    "team",
    "position",
    "report_status",
    "report_primary_injury",
    "report_secondary_injury",
    "practice_status",
    "practice_primary_injury",
    "practice_secondary_injury",
]

#: Upstream spells practice participation as a sentence. The mapping is exhaustive on purpose:
#: unlike ESPN's open-ended status field, this column has exactly three spellings, so a fourth
#: is a real change in the source rather than a value to absorb. `_to_practice_status` raises on
#: one instead of silently nulling it, because a null here already means something specific
#: ("on the report, no practice line") and quietly widening that would corrupt the column.
_PRACTICE_LABELS: Final[dict[str, PracticeStatus]] = {
    "full participation in practice": PracticeStatus.FULL,
    "limited participation in practice": PracticeStatus.LIMITED,
    "did not participate in practice": PracticeStatus.DNP,
}


def _fetch_raw_injuries(seasons: list[int]) -> pd.DataFrame:
    """Thin wrapper around the nflreadpy call; tests monkey-patch this."""
    return nflreadpy.load_injuries(seasons=seasons).to_pandas()


def _to_injury_status(raw: object) -> object:
    """One upstream designation -> an `InjuryStatus` value, preserving null.

    Deliberately **not** `schemas.parse_injury_status`: that helper maps empty to `ACTIVE`
    because ESPN omits the field for uninjured players. Here an empty value means the opposite
    -- the player is on the report without a Sunday designation yet -- so it must stay null.

    An unrecognised designation maps to `UNKNOWN` and warns rather than raising, matching how
    `InjuryStatus.UNKNOWN` is used everywhere else: a status we have no mapping for is a gap in
    our table, not evidence about the player, and it must not take a season's ingest down.
    """
    if pd.isna(raw):
        return pd.NA
    text = str(raw).strip()
    if not text:
        return pd.NA
    try:
        return InjuryStatus(text.upper()).value
    except ValueError:
        _log.warning(
            "injury_report: unrecognised report_status %r mapped to %s. If upstream has added a "
            "designation, add it to InjuryStatus rather than leaving it unknown.",
            text,
            InjuryStatus.UNKNOWN.value,
        )
        return InjuryStatus.UNKNOWN.value


def _to_practice_status(raw: object) -> object:
    """One upstream practice sentence -> a `PracticeStatus` value, preserving null.

    Raises on an unrecognised label. See `_PRACTICE_LABELS` for why this is strict where
    `_to_injury_status` is forgiving.
    """
    if pd.isna(raw):
        return pd.NA
    text = str(raw).strip()
    if not text:
        return pd.NA
    try:
        return _PRACTICE_LABELS[text.lower()].value
    except KeyError:
        raise ValueError(
            f"injury_report: unrecognised practice_status {text!r}. Upstream has historically "
            f"emitted exactly {sorted(_PRACTICE_LABELS)}, so a new value is a source change, "
            f"not a row to drop -- add it to _PRACTICE_LABELS and PracticeStatus."
        ) from None


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # Regular season only. Upstream mixes POST rows in with a restarted week numbering, so a
    # Week 1 postseason row would collide with a Week 1 regular-season row on any
    # (gsis_id, season, week) join.
    df = df[df["season_type"] == "REG"].copy()

    df = drop_placeholder_gsis_rows(df, source="injury_report")

    # Drop rows with NaN season/week before int64 coercion.
    df = df[df["season"].notna() & df["week"].notna()].copy()
    # Upstream returns int32; pandera Series[int] requires int64.
    for int_col in ("season", "week"):
        df[int_col] = df[int_col].astype("int64")

    df["team"] = df["team"].map(lambda v: normalize_team_code(v).value).astype(_PYARROW_STR)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)

    df["report_status"] = df["report_status"].map(_to_injury_status).astype(_PYARROW_STR)
    df["practice_status"] = df["practice_status"].map(_to_practice_status).astype(_PYARROW_STR)
    for text_col in (
        "report_primary_injury",
        "report_secondary_injury",
        "practice_primary_injury",
        "practice_secondary_injury",
    ):
        df[text_col] = df[text_col].astype(_PYARROW_STR)

    # Filter rows at unsupported positions BEFORE schema validation. The report covers every
    # position on the roster -- linemen, defenders, punters -- and we model six.
    df = df[df["position"].isin([p.value for p in Position])].copy()

    df = df[[c for c in _KEEP if c in df.columns]].copy()
    # IMPORTANT: reassign -- strict="filter" returns a new DataFrame.
    df = InjuryReportSchema.validate(df)
    return df


def refresh_injury_report(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write the injury report for each season. One partition per season. Idempotent."""
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_injuries([season])
        df = _normalize_one_season(raw)
        path = write_partition(data_root / "raw", "injury_report", df, season=season, week=None)
        record_manifest(data_root, table="injury_report", season=season, df=df)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    for path in refresh_injury_report(args.data_root, seasons=args.seasons):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
