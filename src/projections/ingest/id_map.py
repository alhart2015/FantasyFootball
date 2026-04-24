"""Build the canonical id_map.parquet from `nfl_data_py.import_ids()`.

`_fetch_raw_id_map` is split out so tests can monkeypatch it instead of
hitting the network.
"""

from __future__ import annotations

from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import IdMapSchema, normalize_team_code
from projections.store import write_partition

_PYARROW_STR = pd.StringDtype("pyarrow")


def _fetch_raw_id_map() -> pd.DataFrame:
    return nfl.import_ids()


def _normalize_team(v: str | None) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return normalize_team_code(str(v)).value


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

    # Coerce all string columns to string[pyarrow] so pandera is satisfied.
    # Nullable ID columns get pd.NA for missing values (compatible with StringDtype).
    for col in ("espn_id", "sleeper_id", "pfr_id"):
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), other=pd.NA).astype(_PYARROW_STR)

    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["full_name"] = df["full_name"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)

    # team is nullable — map to canonical value or pd.NA
    df["team"] = (
        df["team"]
        .map(_normalize_team)
        .astype(_PYARROW_STR)
    )

    df = df.drop_duplicates(subset=["gsis_id"], keep="first").reset_index(drop=True)

    IdMapSchema.validate(df)
    out = write_partition(data_root / "raw", "id_map", df, season=None, week=None)
    record_manifest(data_root, table="id_map", season=None, df=df)
    return out
