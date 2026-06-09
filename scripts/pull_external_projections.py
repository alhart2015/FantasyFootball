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
import urllib.error
from pathlib import Path
from typing import Any

import pandas as pd

# Single-source the network I/O + decode constants from the ingest module. The PARSE helpers
# below intentionally stay separate: this spike feeds the (frozen) RMSE benchmark, so it rounds
# count stats and captures each player's ESPN ACTUAL applied total — neither of which the ingest
# module wants (it stores raw fractional projections, no actuals). Keep these in sync by hand
# only where the ESPN payload SHAPE changes; the shared fetch/URLs/positions move together.
from projections.ingest.external_projections import (
    _ESPN_POSITIONS,
    COUNT_FIELDS,
    ESPN_STAT_IDS,
    STAT_FIELDS,
    fetch_espn,
    fetch_sleeper_season,
    round_count,
)

# Explicit re-exports so downstream importers (benchmark_projections.py) and
# mypy see these as public names of this module.
__all__ = ["COUNT_FIELDS", "ESPN_STAT_IDS", "STAT_FIELDS", "round_count"]


def espn_stats_to_statline_dict(stats: dict[str, float]) -> dict[str, float]:
    """Map ESPN's numeric stat dict to our StatLine field names. Missing ids -> 0.
    Count fields are half-up rounded to int; yards stay float. (Benchmark-specific: the ingest
    module stores RAW fractional counts — see external_projections._espn_stats_to_statline.)"""
    out: dict[str, float] = {f: 0.0 for f in STAT_FIELDS}
    for sid, field in ESPN_STAT_IDS.items():
        if sid in stats:
            val = float(stats[sid])
            out[field] = float(round_count(val)) if field in COUNT_FIELDS else val
    return out


def parse_espn_players(payload: dict[str, Any], season: int) -> pd.DataFrame:
    """Tidy one ESPN kona_player_info payload into one row per QB/RB/WR/TE with a preseason
    projected season stat line PLUS the season's actual applied total (for RMSE benchmarking).
    Players without a season-proj entry, or not in QB/RB/WR/TE, are dropped. (Diverges from the
    ingest module's parse_espn_players, which omits actuals and stores raw fractional counts.)"""
    rows: list[dict[str, object]] = []
    for entry in payload.get("players", []):
        pl = entry.get("player", {})
        position = _ESPN_POSITIONS.get(pl.get("defaultPositionId"))
        if position is None:
            continue
        espn_id = pl.get("id")
        if espn_id is None:
            continue
        proj_stats: dict[str, float] | None = None
        actual_total: float | None = None
        for s in pl.get("stats", []):
            if s.get("seasonId") != season or s.get("statSplitTypeId") != 0:
                continue
            if s.get("statSourceId") == 1:
                # last-write-wins; ESPN provides at most one season-proj entry per player
                proj_stats = s.get("stats", {})
            elif s.get("statSourceId") == 0:
                actual_total = s.get("appliedTotal")
        if not proj_stats:
            # None (no season-proj entry) or {} (entry present but no inner stat
            # dict — seen on injured/placeholder players); both mean "no usable
            # projection", so drop rather than keep an all-zero line that would
            # pollute ESPN's error metrics.
            continue
        ownership = pl.get("ownership") or {}
        ppr_rank = ((pl.get("draftRanksByRankType") or {}).get("PPR") or {}).get("rank")
        row: dict[str, object] = {
            "espn_id": str(espn_id),
            "full_name": pl.get("fullName"),
            "position": position,
            "espn_adp": ownership.get("averageDraftPosition"),
            "espn_pos_rank": ppr_rank,
            "espn_actual_applied_total": actual_total,
        }
        row.update(espn_stats_to_statline_dict(proj_stats))
        rows.append(row)
    return pd.DataFrame(rows)


def parse_sleeper_adp(payload: list[dict[str, Any]]) -> pd.DataFrame:
    """Keep sleeper player id + PPR ADP only (the benchmark uses Sleeper as a rank reference,
    not a stat-line source — unlike the ingest module's parse_sleeper_projections, which also
    keeps name/position). Rows without a player id are dropped."""
    rows: list[dict[str, object]] = []
    for item in payload:
        pid = item.get("player_id")
        if pid is None:
            continue
        stats = item.get("stats") or {}
        rows.append({"sleeper_id": str(pid), "sleeper_adp": stats.get("adp_ppr")})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull preseason external projections for one season.")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--out-root", type=Path, default=Path("data/external_projections"))
    args = ap.parse_args()

    out_dir = args.out_root / str(args.season)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        espn_payload = fetch_espn(args.season)
        sleeper_payload = fetch_sleeper_season(args.season)
    except urllib.error.URLError as exc:  # HTTPError + connection/DNS/timeout failures
        detail = f"HTTP {exc.code}" if isinstance(exc, urllib.error.HTTPError) else str(exc.reason)
        raise SystemExit(f"External API error for season {args.season}: {detail}") from exc

    espn = parse_espn_players(espn_payload, args.season)
    if espn.empty:
        raise SystemExit(
            f"ESPN returned 0 players for {args.season} (possible rate-limit/soft-block or bad "
            f"season). Refusing to write an empty parquet."
        )
    espn.to_parquet(out_dir / "espn.parquet", index=False)
    print(f"ESPN: {len(espn)} players -> {out_dir / 'espn.parquet'}", flush=True)

    sleeper = parse_sleeper_adp(sleeper_payload)
    if sleeper.empty:
        raise SystemExit(
            f"Sleeper returned 0 rows for {args.season}. Refusing to write an empty parquet."
        )
    sleeper.to_parquet(out_dir / "sleeper_adp.parquet", index=False)
    print(f"Sleeper ADP: {len(sleeper)} players -> {out_dir / 'sleeper_adp.parquet'}", flush=True)


if __name__ == "__main__":
    main()
