"""Track every ingest write in `data/manifests/ingest_manifest.parquet`.

Schema: (table, season, fetched_at, rowcount, checksum). One row per
(table, season) pair — re-runs replace the row in place so the manifest
always reflects the *current* on-disk state."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

_MANIFEST_FILE = Path("manifests") / "ingest_manifest.parquet"


def _manifest_path(data_root: Path) -> Path:
    return data_root / _MANIFEST_FILE


def compute_checksum(df: pd.DataFrame) -> str:
    """SHA-256 over the parquet bytes of `df` — stable identifier for content."""
    blob = df.to_parquet(index=False)
    return hashlib.sha256(blob).hexdigest()


def record(
    data_root: Path,
    *,
    table: str,
    season: int | None,
    df: pd.DataFrame,
) -> None:
    """Upsert a manifest row for `(table, season)`. Replaces any existing row."""
    path = _manifest_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame(
        [
            {
                "table": table,
                "season": pd.NA if season is None else int(season),
                "fetched_at": datetime.now(tz=UTC),
                "rowcount": len(df),
                "checksum": compute_checksum(df),
            }
        ]
    )

    if path.exists():
        existing = pd.read_parquet(path)
        if season is None:
            mask = (existing["table"] == table) & existing["season"].isna()
        else:
            mask = (existing["table"] == table) & (existing["season"] == int(season))
        existing = existing[~mask]
        out = pd.concat([existing, new_row], ignore_index=True)
    else:
        out = new_row

    out["season"] = out["season"].astype("Int64")
    out.to_parquet(path, index=False)


def read_manifest(data_root: Path) -> pd.DataFrame:
    path = _manifest_path(data_root)
    if not path.exists():
        return pd.DataFrame(columns=["table", "season", "fetched_at", "rowcount", "checksum"])
    return pd.read_parquet(path)
