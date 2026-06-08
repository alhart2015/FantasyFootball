"""Ingest source for external preseason projections (ESPN + Sleeper).

Repeatable, dated-snapshot ingest: each `refresh_external_projections(...)` writes one
`ExternalProjectionSchema` snapshot under data/raw/external_projections/season=YYYY/
asof=YYYY-MM-DD/. Veterans get their real gsis_id via the id_map crosswalk; pre-camp
rookies get a deterministic placeholder (99-XXXXXXX, flagged is_placeholder_gsis) that
auto-reconciles on later refreshes once id_map propagates the real id. Stat lines are
stored, not fantasy points (the scoring layer converts downstream).

Network-dependent; the pure parsers/normalizers are unit-tested with synthetic payloads.

Usage:
    python -m projections.ingest.external_projections --season 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from projections.schemas import (
    _PYARROW_STR,
    ExternalProjectionSchema,
    ProjectionSource,
)
from projections.store import read_partition, write_partition


class ExternalProjectionError(RuntimeError):
    """Raised by refresh_external_projections on an API failure or empty pull. The CLI
    (main) converts it to SystemExit; programmatic callers can catch it normally."""


_ESPN_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/"
    "{season}/segments/0/leaguedefaults/3?view=kona_player_info"
)
_SLEEPER_URL = "https://api.sleeper.com/projections/nfl/{season}?season_type=regular"
_UA = "Mozilla/5.0"

# ESPN numeric stat-id -> StatLine field (common scoring set). Verified against real data.
ESPN_STAT_IDS: dict[str, str] = {
    "3": "passing_yards",
    "4": "passing_tds",
    "20": "interceptions",
    "24": "rushing_yards",
    "25": "rushing_tds",
    "53": "receptions",
    "42": "receiving_yards",
    "43": "receiving_tds",
    "72": "fumbles_lost",
}
STAT_FIELDS: tuple[str, ...] = (
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
)
_COUNT_FIELDS = frozenset(
    {"passing_tds", "interceptions", "rushing_tds", "receptions", "receiving_tds", "fumbles_lost"}
)
_ESPN_POSITIONS: dict[int, str] = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}
_SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})


def _round_count(value: float) -> int:
    """Half-up rounding for non-negative projected count stats (Python's round() is banker's).
    Clamps at 0 — count stats (TDs, receptions, INTs, fumbles) are never negative."""
    return max(0, int(value + 0.5))


def _espn_stats_to_statline(stats: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {f: 0.0 for f in STAT_FIELDS}
    for sid, field in ESPN_STAT_IDS.items():
        if sid in stats:
            val = float(stats[sid])
            out[field] = float(_round_count(val)) if field in _COUNT_FIELDS else val
    return out


def parse_espn_players(payload: dict[str, Any], season: int) -> pd.DataFrame:
    """Tidy one ESPN kona_player_info payload -> one row per QB/RB/WR/TE with a preseason
    projected stat line + espn_id + ADP + PPR draft rank."""
    rows: list[dict[str, object]] = []
    for entry in payload.get("players", []):
        pl = entry.get("player", {})
        position = _ESPN_POSITIONS.get(pl.get("defaultPositionId"))
        if position is None:
            continue
        espn_id = pl.get("id")
        if espn_id is None:
            continue
        full_name = pl.get("fullName")
        if not full_name:
            continue
        proj_stats: dict[str, float] | None = None
        for s in pl.get("stats", []):
            if s.get("seasonId") != season or s.get("statSplitTypeId") != 0:
                continue
            if s.get("statSourceId") == 1:
                proj_stats = s.get("stats", {})
        if not proj_stats:
            continue
        ownership = pl.get("ownership") or {}
        ppr_rank = ((pl.get("draftRanksByRankType") or {}).get("PPR") or {}).get("rank")
        row: dict[str, object] = {
            "espn_id": str(espn_id),
            "full_name": full_name,
            "position": position,
            "espn_adp": ownership.get("averageDraftPosition"),
            "espn_pos_rank": ppr_rank,
        }
        row.update(_espn_stats_to_statline(proj_stats))
        rows.append(row)
    return pd.DataFrame(rows)


# NOTE (v1 position choice): rows use `player.position` (the primary position), not
# `fantasy_positions`. This is deliberate — it's simpler and covers the common case.
# For flex/hybrid players (e.g., a RB with WR eligibility) `player.position` and
# `fantasy_positions` can diverge; revisit if multi-eligibility matters for the Draft Hub.
def parse_sleeper_projections(payload: list[dict[str, Any]]) -> pd.DataFrame:
    """Tidy Sleeper season projections -> one row per QB/RB/WR/TE with sleeper_id + name +
    position + PPR ADP (Sleeper has no stat line at the season level)."""
    rows: list[dict[str, object]] = []
    for item in payload:
        pid = item.get("player_id")
        if pid is None:
            continue
        pl = item.get("player") or {}
        position = pl.get("position")
        if position not in _SKILL_POSITIONS:
            continue
        first = pl.get("first_name") or ""
        last = pl.get("last_name") or ""
        full_name = f"{first} {last}".strip()
        if not full_name:
            continue
        stats = item.get("stats") or {}
        rows.append(
            {
                "sleeper_id": str(pid),
                "full_name": full_name,
                "position": position,
                "sleeper_adp": stats.get("adp_ppr"),
            }
        )
    return pd.DataFrame(rows)


def _make_placeholder_gsis(source: str, source_player_id: str) -> str:
    """Deterministic synthetic gsis_id for a player not in id_map (e.g., a pre-camp rookie).
    Matches GSIS_ID_PATTERN with a reserved 99- prefix. Source-scoped so an ESPN and a
    Sleeper id never collide into the same placeholder."""
    digest = hashlib.sha1(f"{source}:{source_player_id}".encode()).hexdigest()
    return f"99-{int(digest, 16) % 10_000_000:07d}"


def _attach_gsis_id(
    df: pd.DataFrame, id_map: pd.DataFrame, *, source: str, id_col: str
) -> pd.DataFrame:
    """Left-join df to id_map on `id_col` (espn_id/sleeper_id) to attach a real gsis_id;
    unmatched rows get a deterministic placeholder. Adds `gsis_id` + `is_placeholder_gsis`.
    Dedupes the crosswalk on `id_col` so a duplicate id_map mapping can't multiply rows."""
    crosswalk = id_map[["gsis_id", id_col]].dropna(subset=[id_col]).drop_duplicates(subset=[id_col])
    merged = df.merge(crosswalk, on=id_col, how="left")
    mask = merged["gsis_id"].isna()
    merged["is_placeholder_gsis"] = mask
    merged["gsis_id"] = merged["gsis_id"].astype("object")  # allow filling pyarrow-NA with a str
    merged.loc[mask, "gsis_id"] = merged.loc[mask, id_col].map(
        lambda pid: _make_placeholder_gsis(source, pid)
    )
    return merged


_CANONICAL_STR_COLS = ("source", "source_player_id", "gsis_id", "full_name", "position", "asof")


def _finish_canonical(df: pd.DataFrame, *, season: int, asof: date) -> pd.DataFrame:
    df = df.copy()
    df["season"] = season
    df["asof"] = asof.isoformat()
    for c in _CANONICAL_STR_COLS:
        df[c] = df[c].astype(_PYARROW_STR)
    return df


def _espn_to_canonical(
    espn: pd.DataFrame, *, season: int, asof: date, id_map: pd.DataFrame
) -> pd.DataFrame:
    keyed = _attach_gsis_id(espn, id_map, source=ProjectionSource.ESPN.value, id_col="espn_id")
    out = pd.DataFrame(
        {
            "source": ProjectionSource.ESPN.value,
            "source_player_id": keyed["espn_id"],
            "gsis_id": keyed["gsis_id"],
            "is_placeholder_gsis": keyed["is_placeholder_gsis"],
            "full_name": keyed["full_name"],
            "position": keyed["position"],
            "adp": keyed["espn_adp"].astype(float),
            "espn_draft_rank": keyed["espn_pos_rank"].astype(float),
        }
    )
    for f in STAT_FIELDS:
        out[f] = keyed[f].astype(float)
    return _finish_canonical(out, season=season, asof=asof)


def _sleeper_to_canonical(
    sleeper: pd.DataFrame, *, season: int, asof: date, id_map: pd.DataFrame
) -> pd.DataFrame:
    keyed = _attach_gsis_id(
        sleeper, id_map, source=ProjectionSource.SLEEPER.value, id_col="sleeper_id"
    )
    out = pd.DataFrame(
        {
            "source": ProjectionSource.SLEEPER.value,
            "source_player_id": keyed["sleeper_id"],
            "gsis_id": keyed["gsis_id"],
            "is_placeholder_gsis": keyed["is_placeholder_gsis"],
            "full_name": keyed["full_name"],
            "position": keyed["position"],
            "adp": keyed["sleeper_adp"].astype(float),
            "espn_draft_rank": pd.array([pd.NA] * len(keyed), dtype="Float64"),
        }
    )
    for f in STAT_FIELDS:
        out[f] = pd.array([pd.NA] * len(keyed), dtype="Float64")
    return _finish_canonical(out, season=season, asof=asof)


def fetch_espn(season: int, limit: int = 800) -> dict[str, Any]:
    flt = {"players": {"limit": limit, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
    req = urllib.request.Request(
        _ESPN_URL.format(season=season),
        headers={"User-Agent": _UA, "X-Fantasy-Filter": json.dumps(flt)},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)  # type: ignore[no-any-return]


def fetch_sleeper_season(season: int) -> list[dict[str, Any]]:
    req = urllib.request.Request(_SLEEPER_URL.format(season=season), headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)  # type: ignore[no-any-return]


def refresh_external_projections(data_root: Path, *, season: int, asof: date | None = None) -> Path:
    """Fetch ESPN + Sleeper preseason projections, crosswalk to gsis_id (placeholder for
    rookies), validate, and write one dated snapshot. `asof` defaults to today (UTC)."""
    asof = asof or datetime.now(UTC).date()
    try:
        espn_payload = fetch_espn(season)
        sleeper_payload = fetch_sleeper_season(season)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        if isinstance(exc, urllib.error.HTTPError):
            detail = f"HTTP {exc.code}"
        elif isinstance(exc, urllib.error.URLError):
            detail = str(exc.reason)
        else:
            detail = f"non-JSON response ({exc})"
        raise ExternalProjectionError(f"External API error for season {season}: {detail}") from exc

    espn = parse_espn_players(espn_payload, season)
    sleeper = parse_sleeper_projections(sleeper_payload)
    if espn.empty or sleeper.empty:
        raise ExternalProjectionError(
            f"Empty pull for {season} (espn={len(espn)} rows, sleeper={len(sleeper)} rows) — "
            f"refusing to write an empty asof snapshot."
        )

    id_map = read_partition(data_root / "raw", "id_map")
    frame = pd.concat(
        [
            _espn_to_canonical(espn, season=season, asof=asof, id_map=id_map),
            _sleeper_to_canonical(sleeper, season=season, asof=asof, id_map=id_map),
        ],
        ignore_index=True,
    )
    frame = ExternalProjectionSchema.validate(frame)
    return write_partition(
        data_root / "raw", "external_projections", frame, season=season, asof=asof
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest external preseason projections (one snapshot)."
    )
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument(
        "--asof",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="Pull-date partition (ISO YYYY-MM-DD); defaults to today (UTC).",
    )
    args = ap.parse_args()
    try:
        path = refresh_external_projections(args.data_root, season=args.season, asof=args.asof)
    except ExternalProjectionError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote external-projection snapshot: {path}", flush=True)


if __name__ == "__main__":
    main()
