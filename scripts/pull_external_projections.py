"""Spike: pull preseason projections + ADP from free public sources for one season.

Writes intermediate parquet under data/external_projections/{season}/:
  - espn.parquet       : per-player preseason projected stat line + ADP/rank + ESPN actual total
  - sleeper_adp.parquet : per-player preseason ADP (rank reference only)

ESPN's season projection (statSourceId=1, statSplitTypeId=0) is the genuine
preseason forecast (verified against 2024: rookie/breakout misses, not
contaminated end-of-season values). Sleeper exposes only ADP at the season
level, so it is a rank reference, not a stat-line source.

Network-dependent; the pure parse helpers are unit-tested with synthetic payloads.

Usage:
    python scripts/pull_external_projections.py --season 2024
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

# ESPN numeric stat-id -> StatLine field (common scoring set only). Decoded
# empirically against known 2024 players; reconstructing through ESPN PPR
# matches appliedTotal within ~1 pt.
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
_COUNT_FIELDS = frozenset(
    {"passing_tds", "interceptions", "rushing_tds", "receptions", "receiving_tds", "fumbles_lost"}
)
_ALL_FIELDS = (
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


def espn_stats_to_statline_dict(stats: dict[str, float]) -> dict[str, float]:
    """Map ESPN's numeric stat dict to our StatLine field names. Missing ids -> 0.
    Count fields are rounded to the nearest integer; yards stay float."""
    out: dict[str, float] = {f: 0.0 for f in _ALL_FIELDS}
    for sid, field in ESPN_STAT_IDS.items():
        if sid in stats:
            val = float(stats[sid])
            out[field] = float(round(val)) if field in _COUNT_FIELDS else val
    return out


_ESPN_POSITIONS: dict[int, str] = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}


def parse_espn_players(payload: dict[str, Any], season: int) -> pd.DataFrame:
    """Tidy one ESPN kona_player_info payload into one row per QB/RB/WR/TE with a
    preseason projected season stat line. Players without a season-proj entry, or
    not in QB/RB/WR/TE, are dropped."""
    rows: list[dict[str, object]] = []
    for entry in payload.get("players", []):
        pl = entry.get("player", {})
        position = _ESPN_POSITIONS.get(pl.get("defaultPositionId"))
        if position is None:
            continue
        proj_stats: dict[str, float] | None = None
        actual_total: float | None = None
        for s in pl.get("stats", []):
            if s.get("seasonId") != season or s.get("statSplitTypeId") != 0:
                continue
            if s.get("statSourceId") == 1:
                proj_stats = s.get("stats", {})
            elif s.get("statSourceId") == 0:
                actual_total = s.get("appliedTotal")
        if proj_stats is None:
            continue
        ownership = pl.get("ownership") or {}
        ppr_rank = ((pl.get("draftRanksByRankType") or {}).get("PPR") or {}).get("rank")
        row: dict[str, object] = {
            "espn_id": str(pl.get("id")),
            "full_name": pl.get("fullName"),
            "position": position,
            "espn_adp": ownership.get("averageDraftPosition"),
            "espn_pos_rank": ppr_rank,
            "espn_actual_applied_total": actual_total,
        }
        row.update(espn_stats_to_statline_dict(proj_stats))
        rows.append(row)
    return pd.DataFrame(rows)


_ESPN_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/"
    "{season}/segments/0/leaguedefaults/3?view=kona_player_info"
)
_SLEEPER_URL = "https://api.sleeper.com/projections/nfl/{season}?season_type=regular"
_UA = "Mozilla/5.0"


def parse_sleeper_adp(payload: list[dict[str, Any]]) -> pd.DataFrame:
    """Keep sleeper player id + PPR ADP. Rows without a player id are dropped."""
    rows: list[dict[str, object]] = []
    for item in payload:
        pid = item.get("player_id")
        if pid is None:
            continue
        stats = item.get("stats") or {}
        rows.append({"sleeper_id": str(pid), "sleeper_adp": stats.get("adp_ppr")})
    return pd.DataFrame(rows)


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull preseason external projections for one season.")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--out-root", type=Path, default=Path("data/external_projections"))
    args = ap.parse_args()

    out_dir = args.out_root / str(args.season)
    out_dir.mkdir(parents=True, exist_ok=True)

    espn = parse_espn_players(fetch_espn(args.season), args.season)
    espn.to_parquet(out_dir / "espn.parquet", index=False)
    print(f"ESPN: {len(espn)} players -> {out_dir / 'espn.parquet'}", flush=True)

    sleeper = parse_sleeper_adp(fetch_sleeper_season(args.season))
    sleeper.to_parquet(out_dir / "sleeper_adp.parquet", index=False)
    print(f"Sleeper ADP: {len(sleeper)} players -> {out_dir / 'sleeper_adp.parquet'}", flush=True)


if __name__ == "__main__":
    main()
