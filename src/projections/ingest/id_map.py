"""Build the canonical id_map.parquet from `nflreadpy.load_ff_playerids()`.

`_fetch_raw_id_map` is split out so tests can monkeypatch it instead of
hitting the network.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nflreadpy
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import (
    _NO_TEAM_CODES,
    _PYARROW_STR,
    IdMapSchema,
    Position,
    normalize_team_code,
)
from projections.store import write_partition

logger = logging.getLogger(__name__)


def _coerce_external_id(s: pd.Series) -> pd.Series:
    """Persist an external id column as a clean integer-string. Upstream returns the numeric
    ids (espn_id/sleeper_id) as float64 (NaNs force float), so a plain .astype(str) yields
    '4374302.0'. When every non-null value parses as a number, round-trip through nullable
    Int64 to drop the spurious '.0'. Genuinely-string id columns (pfr_id, e.g. 'ChASEJa00')
    are left unchanged — they pass through as nullable pyarrow strings."""
    numeric = pd.to_numeric(s, errors="coerce")
    every_nonnull_is_numeric = bool((numeric.notna() | s.isna()).all())
    if every_nonnull_is_numeric:
        s = numeric.astype("Int64")
    return s.where(s.notna(), other=pd.NA).astype(_PYARROW_STR)


def _fetch_raw_id_map() -> pd.DataFrame:
    return nflreadpy.load_ff_playerids().to_pandas()


def _normalize_team(v: str | None) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v)
    if s.lower() in _NO_TEAM_CODES:
        return None
    return normalize_team_code(s).value


def build_id_map(data_root: Path) -> Path:
    """Fetch player IDs across platforms, normalize, and write to id_map.parquet.

    Idempotent — re-running overwrites the existing file.
    """
    raw = _fetch_raw_id_map()

    cols = ["gsis_id", "espn_id", "sleeper_id", "pfr_id", "name", "position", "team"]
    df = raw[[c for c in cols if c in raw.columns]].copy()
    df = df.rename(columns={"name": "full_name"})

    # Drop rows without canonical id; downstream joins are unusable without it.
    df = df[df["gsis_id"].notna()].copy()

    # Drop rows whose gsis_id does not match the canonical pattern. Two
    # populations hit this: legacy PFR-style IDs for very old players, and
    # PFR-style placeholders that nflverse holds for the current draft class
    # until NFL.com assigns real gsis_ids around training camp (~July).
    from projections.schemas import GSIS_ID_PATTERN

    n_pre_regex = len(df)
    df = df[df["gsis_id"].astype(str).str.match(rf"^{GSIS_ID_PATTERN}$")].copy()
    n_filtered = n_pre_regex - len(df)
    if n_filtered > 0:
        logger.warning(
            "build_id_map: filtered %d row(s) with non-GSIS placeholder ids "
            "(typical of pre-camp rookies for the current draft class — nflverse holds "
            "PFR-style placeholders until NFL assigns real gsis_ids ~July). Re-ingest "
            "after training camps to capture these players.",
            n_filtered,
        )

    # Drop players at positions outside our covered set (offensive line, punters, etc.)
    # load_ff_playerids() returns roster-wide rows; we only model the positions
    # downstream consumers care about.
    df = df[df["position"].isin([p.value for p in Position])].copy()

    # Coerce all string columns to string[pyarrow] so pandera is satisfied.
    # Nullable ID columns get pd.NA for missing values (compatible with StringDtype).
    for col in ("espn_id", "sleeper_id", "pfr_id"):
        if col in df.columns:
            df[col] = _coerce_external_id(df[col])

    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["full_name"] = df["full_name"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)

    # team is nullable — map to canonical value or pd.NA
    df["team"] = df["team"].map(_normalize_team).astype(_PYARROW_STR)

    df = df.drop_duplicates(subset=["gsis_id"], keep="first").reset_index(drop=True)

    df = IdMapSchema.validate(df)
    out = write_partition(data_root / "raw", "id_map", df, season=None, week=None)
    record_manifest(data_root, table="id_map", season=None, df=df)
    return out
