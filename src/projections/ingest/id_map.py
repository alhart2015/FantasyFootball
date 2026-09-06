"""Build the canonical id_map.parquet from `nflreadpy.load_ff_playerids()`.

`_fetch_raw_id_map` is split out so tests can monkeypatch it instead of
hitting the network.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import nflreadpy
import pandas as pd

from projections.ingest.espn_league import ESPN_PRO_TEAMS
from projections.ingest.manifest import record as record_manifest
from projections.schemas import (
    _NO_TEAM_CODES,
    _PYARROW_STR,
    DST_GSIS_IDS,
    IdMapSchema,
    Position,
    Team,
    normalize_team_code,
)
from projections.store import write_partition

logger = logging.getLogger(__name__)


def _coerce_external_id(s: pd.Series, *, numeric: bool) -> pd.Series:
    """Persist an external id column as a clean nullable pyarrow string. NUMERIC id columns
    (espn_id/sleeper_id) round-trip the numeric values through Int64 to drop the spurious '.0'
    that upstream float64 dtype produces; any genuinely non-numeric id (e.g. a team-defense or
    future alphanumeric source id) is PRESERVED verbatim rather than silently nulled, so its
    crosswalk mapping survives. String id columns (pfr_id, e.g. 'ChASEJa00', or leading-zero
    ids) pass through unchanged."""
    if numeric:
        as_num = pd.to_numeric(s, errors="coerce")
        # numeric values -> clean int-string ('4374302', not '4374302.0'); where coercion
        # failed, keep the original string (a non-numeric id) verbatim, NA where truly missing.
        cleaned = as_num.astype("Int64").astype(_PYARROW_STR)
        return cleaned.where(as_num.notna(), other=s.astype(_PYARROW_STR))
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


#: ESPN's D/ST player ids are negative and derived from `proTeamId`: `-(16000 + proTeamId)`.
#: Verified against all 32 defenses in the live 2026 payload on 2026-09-06 (Falcons proTeamId 1
#: -> -16001, Texans 34 -> -16034).
_ESPN_DST_ID_BASE: Final = 16000


def dst_id_map_rows() -> pd.DataFrame:
    """The 32 team-defense rows, which the upstream player-id source does not carry.

    Without these a rostered defense cannot be resolved from an ESPN roster entry, and every
    mid-season tool reports it as an unknown player and skips it (issue #166).

    Sleeper identifies a defense by the bare team code (`"SEA"`), ESPN by the negative id above.
    `full_name` matches ESPN's own label so a name-based fallback lookup resolves too.
    """
    espn_by_team = {
        normalize_team_code(code): -(_ESPN_DST_ID_BASE + pro_id)
        for pro_id, code in ESPN_PRO_TEAMS.items()
    }
    return pd.DataFrame(
        {
            "gsis_id": pd.Series([DST_GSIS_IDS[t] for t in Team], dtype=_PYARROW_STR),
            "espn_id": pd.Series(
                [str(espn_by_team[t]) if t in espn_by_team else pd.NA for t in Team],
                dtype=_PYARROW_STR,
            ),
            "sleeper_id": pd.Series([t.value for t in Team], dtype=_PYARROW_STR),
            "pfr_id": pd.Series([pd.NA] * len(Team), dtype=_PYARROW_STR),
            "full_name": pd.Series([f"{t.value} D/ST" for t in Team], dtype=_PYARROW_STR),
            "position": pd.Series([Position.DST.value] * len(Team), dtype=_PYARROW_STR),
            "team": pd.Series([t.value for t in Team], dtype=_PYARROW_STR),
        }
    )


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
    for col in ("espn_id", "sleeper_id"):
        if col in df.columns:
            df[col] = _coerce_external_id(df[col], numeric=True)
    if "pfr_id" in df.columns:
        df["pfr_id"] = _coerce_external_id(df["pfr_id"], numeric=False)

    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["full_name"] = df["full_name"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)

    # team is nullable — map to canonical value or pd.NA
    df["team"] = df["team"].map(_normalize_team).astype(_PYARROW_STR)

    df = pd.concat([df, dst_id_map_rows()], ignore_index=True)
    df = df.drop_duplicates(subset=["gsis_id"], keep="first").reset_index(drop=True)

    df = IdMapSchema.validate(df)
    out = write_partition(data_root / "raw", "id_map", df, season=None, week=None)
    record_manifest(data_root, table="id_map", season=None, df=df)
    return out
