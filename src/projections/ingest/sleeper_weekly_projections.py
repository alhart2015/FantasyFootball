"""Ingest Sleeper *weekly* projections (historical, retrospective).

Unlike the season endpoint (ADP-only), the weekly endpoint
`api.sleeper.com/projections/nfl/<season>/<week>` returns a per-player stat
line. We map it to canonical stat fields, attach gsis_id, and store a weekly
partition. Skill positions only.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from projections.ingest.external_projections import (
    SLEEPER_STAT_FIELDS,
    _make_placeholder_gsis,  # reuse the same placeholder scheme
)
from projections.ingest.identity import normalize_join_id
from projections.schemas import (
    ExternalProjectionWeeklySchema,
    Position,
    ProjectionSource,
)
from projections.store import read_partition, write_partition

_SLEEPER_WEEKLY_URL = "https://api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular"
_SKILL_POSITIONS = {
    Position.QB.value,
    Position.RB.value,
    Position.WR.value,
    Position.TE.value,
}
_STAT_FIELDS = list(SLEEPER_STAT_FIELDS.values())


class SleeperWeeklyError(RuntimeError):
    """Raised when the Sleeper weekly endpoint fetch/parse fails."""


def parse_sleeper_weekly(payload: list[dict[str, Any]], *, season: int, week: int) -> pd.DataFrame:
    """Parse the weekly payload into canonical columns. Pure (no I/O)."""
    rows: list[dict[str, Any]] = []
    for entry in payload:
        player = entry.get("player") or {}
        position = player.get("position")
        if position not in _SKILL_POSITIONS:
            continue
        stats = entry.get("stats") or {}
        mapped = {
            canonical: float(stats[key])
            for key, canonical in SLEEPER_STAT_FIELDS.items()
            if key in stats and stats[key] is not None
        }
        if not mapped:  # ADP-only / empty stat line
            continue
        full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
        row: dict[str, Any] = {
            "sleeper_id": str(entry["player_id"]),
            "full_name": full_name,
            "position": position,
            "season": season,
            "week": week,
        }
        for field in _STAT_FIELDS:
            row[field] = mapped.get(field, pd.NA)
        rows.append(row)

    columns = ["sleeper_id", "full_name", "position", "season", "week", *_STAT_FIELDS]
    return pd.DataFrame(rows, columns=columns)


def _fetch_sleeper_weekly(season: int, week: int) -> list[dict[str, Any]]:
    url = _SLEEPER_WEEKLY_URL.format(season=season, week=week)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # trusted host
            data = json.load(resp)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SleeperWeeklyError(
            f"Sleeper weekly fetch failed for {season} wk{week}: {exc}"
        ) from exc
    if not isinstance(data, list):
        raise SleeperWeeklyError(f"Unexpected Sleeper weekly payload type: {type(data).__name__}")
    return data


def _attach_gsis(df: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    crosswalk = (
        id_map[["gsis_id", "sleeper_id"]]
        .dropna(subset=["sleeper_id"])
        .drop_duplicates("sleeper_id")
        .copy()
    )
    crosswalk["sleeper_id"] = normalize_join_id(crosswalk["sleeper_id"])
    df = df.copy()
    df["sleeper_id"] = normalize_join_id(df["sleeper_id"])
    merged = df.merge(crosswalk, on="sleeper_id", how="left")
    mask = merged["gsis_id"].isna()
    merged["is_placeholder_gsis"] = mask
    merged["gsis_id"] = merged["gsis_id"].astype("object")
    merged.loc[mask, "gsis_id"] = [
        _make_placeholder_gsis(name, pos)
        for name, pos in zip(
            merged.loc[mask, "full_name"], merged.loc[mask, "position"], strict=True
        )
    ]
    return merged


def refresh_sleeper_weekly(data_root: Path, *, season: int, week: int) -> Path:
    """Fetch, parse, attach gsis, validate, and store one weekly partition."""
    payload = _fetch_sleeper_weekly(season, week)
    parsed = parse_sleeper_weekly(payload, season=season, week=week)
    id_map = read_partition(data_root, "id_map", season=None)
    attached = _attach_gsis(parsed, id_map)
    attached = attached.rename(columns={"sleeper_id": "source_player_id"})
    attached["source"] = ProjectionSource.SLEEPER.value
    # dtype hygiene so concat/validate are stable (no all-NA object inference)
    for field in _STAT_FIELDS:
        attached[field] = attached[field].astype("Float64")
    frame = ExternalProjectionWeeklySchema.validate(attached)
    return write_partition(data_root, "sleeper_weekly_projections", frame, season=season, week=week)
