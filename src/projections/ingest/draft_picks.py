"""Refresh per-season draft picks from `nflreadpy.load_draft_picks`.

Writes one parquet partition per season (curated subset). Snapshot
semantics — a season's draft never changes after the draft completes,
so re-running a season overwrites that partition only.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

import nflreadpy
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import (
    _PYARROW_STR,
    GSIS_ID_PATTERN,
    DraftPicksSchema,
)
from projections.store import write_partition

logger = logging.getLogger(__name__)

_GSIS_RE = re.compile(rf"^{GSIS_ID_PATTERN}$")


def _fetch_raw_draft_picks(seasons: list[int]) -> pd.DataFrame:
    """Thin wrapper around nflreadpy; tests monkey-patch this."""
    if not seasons:
        return pd.DataFrame()
    return nflreadpy.load_draft_picks(seasons=seasons).to_pandas()


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "draft_year": pd.array([], dtype=pd.Int64Dtype()),
                "draft_round": pd.array([], dtype=pd.Int64Dtype()),
                "draft_overall_pick": pd.array([], dtype=pd.Int64Dtype()),
                "pfr_id": pd.array([], dtype=_PYARROW_STR),
                "draft_age": pd.array([], dtype=pd.Float64Dtype()),
            }
        )

    df = raw.rename(
        columns={
            "season": "draft_year",
            "round": "draft_round",
            "pick": "draft_overall_pick",
            "pfr_player_id": "pfr_id",
            "age": "draft_age",
        }
    )

    # Filter rows without a valid gsis_id (older drafts may have nulls).
    df = df[df["gsis_id"].notna()].copy()
    n_pre_regex = len(df)
    df = df[df["gsis_id"].astype(str).str.match(_GSIS_RE)].copy()
    n_filtered = n_pre_regex - len(df)
    if n_filtered > 0:
        # nflverse carries PFR-style placeholder ids (e.g. "MEN516487") for
        # the current draft class until NFL.com assigns real gsis_ids around
        # training camp (~July). Surface the filter so a 0-row partition for
        # a freshly-drafted class isn't a silent diagnostic chase.
        logger.warning(
            "refresh_draft_picks: filtered %d row(s) with non-GSIS placeholder ids "
            "(typical of pre-camp rookies for the current draft class — nflverse holds "
            "PFR-style placeholders until NFL assigns real gsis_ids ~July). Re-ingest "
            "after training camps to capture these players.",
            n_filtered,
        )

    # Coerce dtypes: source returns int32 for season/round/pick and
    # float64 for age; pandera schema expects Int64/Float64 nullable types.
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["draft_year"] = df["draft_year"].astype(pd.Int64Dtype())
    df["draft_round"] = df["draft_round"].astype(pd.Int64Dtype())
    df["draft_overall_pick"] = df["draft_overall_pick"].astype(pd.Int64Dtype())
    df["pfr_id"] = df["pfr_id"].where(df["pfr_id"].notna(), other=pd.NA).astype(_PYARROW_STR)
    df["draft_age"] = df["draft_age"].astype(pd.Float64Dtype())

    df = (
        df[
            [
                "gsis_id",
                "draft_year",
                "draft_round",
                "draft_overall_pick",
                "pfr_id",
                "draft_age",
            ]
        ]
        .drop_duplicates(subset=["gsis_id"], keep="first")
        .reset_index(drop=True)
    )

    df = DraftPicksSchema.validate(df)
    return df


def refresh_draft_picks(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write draft pick data for each season.

    One partition per season. Idempotent — re-running a season overwrites
    that partition only.
    """
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_draft_picks([season])
        df = _normalize_one_season(raw)
        path = write_partition(data_root / "raw", "draft_picks", df, season=season, week=None)
        record_manifest(data_root, table="draft_picks", season=season, df=df)
        written.append(path)
    return written
