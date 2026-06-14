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
import logging
import urllib.error
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from projections.ingest.identity import placeholder_name_key
from projections.ingest.manifest import record as record_manifest
from projections.schemas import (
    _PYARROW_STR,
    STAT_FIELDS,
    ExternalProjectionSchema,
    Position,
    ProjectionSource,
)
from projections.store import read_partition, write_partition

_log = logging.getLogger(__name__)


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
COUNT_FIELDS = frozenset(
    {"passing_tds", "interceptions", "rushing_tds", "receptions", "receiving_tds", "fumbles_lost"}
)
# ESPN numeric position-id -> canonical position string (reference the enum, not the literal).
ESPN_POSITIONS: dict[int, str] = {
    1: Position.QB.value,
    2: Position.RB.value,
    3: Position.WR.value,
    4: Position.TE.value,
}
_SKILL_POSITIONS = frozenset(ESPN_POSITIONS.values())


def round_count(value: float) -> int:
    """Half-up rounding for non-negative projected count stats (Python's round() is banker's).
    Clamps at 0 — count stats (TDs, receptions, INTs, fumbles) are never negative."""
    return max(0, int(value + 0.5))


def _map_stats(raw: dict[str, float], mapping: dict[str, str]) -> dict[str, float]:
    """Map a source's raw stat dict to the canonical STAT_FIELDS via `mapping` (source key ->
    canonical field), zero-filling absent fields. Stores RAW fractional values (e.g. 8.4 receiving
    TDs), never rounded — rounding is irreversible and biases season totals; the scoring layer is
    the only place that turns a projected stat into points. (round_count/COUNT_FIELDS are retained
    for the benchmark spike, which rounds for a different, frozen purpose.)"""
    out: dict[str, float] = {field: 0.0 for field in STAT_FIELDS}
    for key, field in mapping.items():
        if key in raw:
            out[field] = float(raw[key])
    return out


def _espn_stats_to_statline(stats: dict[str, float]) -> dict[str, float]:
    return _map_stats(stats, ESPN_STAT_IDS)


# Sleeper's raw projected stat keys -> canonical STAT_FIELDS. Verified live against the Sleeper
# projections API. Values are STAT_FIELDS members (same convention as ESPN_STAT_IDS).
SLEEPER_STAT_FIELDS: dict[str, str] = {
    "pass_yd": "passing_yards",
    "pass_td": "passing_tds",
    "pass_int": "interceptions",
    "rush_yd": "rushing_yards",
    "rush_td": "rushing_tds",
    "rec": "receptions",
    "rec_yd": "receiving_yards",
    "rec_td": "receiving_tds",
    "fum_lost": "fumbles_lost",
}


def _sleeper_stats_to_statline(stats: dict[str, float]) -> dict[str, float] | None:
    """Map Sleeper's raw projected stat line to the canonical STAT_FIELDS, raw (no rounding) —
    mirroring _espn_stats_to_statline. Returns None when `stats` carries none of the mapped keys
    (an ADP-only Sleeper row with no real projection), so the caller stores NA rather than a
    fabricated all-zero line. Non-mapped keys (gp, cmp_pct, *_fd, bonus_*, *_2pt, adp_*) are
    ignored."""
    if not any(key in stats for key in SLEEPER_STAT_FIELDS):
        return None
    return _map_stats(stats, SLEEPER_STAT_FIELDS)


def parse_espn_players(payload: dict[str, Any], season: int) -> pd.DataFrame:
    """Tidy one ESPN kona_player_info payload -> one row per QB/RB/WR/TE with a preseason
    projected stat line + espn_id + ADP + PPR draft rank."""
    rows: list[dict[str, object]] = []
    n_skill = 0  # skill-position players seen (the population we expect projections for)
    n_no_projection = 0  # skill players dropped for a missing/empty projection block
    for entry in payload.get("players", []):
        pl = entry.get("player", {})
        position = ESPN_POSITIONS.get(pl.get("defaultPositionId"))
        if position is None:
            continue
        n_skill += 1
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
            n_no_projection += 1
            continue
        ownership = pl.get("ownership") or {}
        ppr_rank = ((pl.get("draftRanksByRankType") or {}).get("PPR") or {}).get("rank")
        # ESPN encodes "undrafted / no draft data" as ADP 0; normalize non-positive to None so the
        # raw table stores honest null (adp is nullable) rather than an in-band sentinel that every
        # downstream consumer would have to re-discover.
        espn_adp = ownership.get("averageDraftPosition")
        if espn_adp is not None and espn_adp <= 0:
            espn_adp = None
        row: dict[str, object] = {
            "espn_id": str(espn_id),
            "full_name": full_name,
            "position": position,
            "espn_adp": espn_adp,
            "espn_pos_rank": ppr_rank,
        }
        row.update(_espn_stats_to_statline(proj_stats))
        rows.append(row)
    # Surface silent drops: a degraded payload (projections for stars only) writes a
    # plausible-but-truncated snapshot otherwise. Loud when coverage looks suspicious.
    if n_skill:
        level = logging.WARNING if len(rows) < n_skill // 2 else logging.INFO
        _log.log(
            level,
            "ESPN parse season=%s: kept %d of %d skill players (%d had no usable projection).",
            season,
            len(rows),
            n_skill,
            n_no_projection,
        )
    return pd.DataFrame(rows)


# NOTE (v1 position choice): rows use `player.position` (the primary position), not
# `fantasy_positions`. This is deliberate — it's simpler and covers the common case.
# For flex/hybrid players (e.g., a RB with WR eligibility) `player.position` and
# `fantasy_positions` can diverge; revisit if multi-eligibility matters for the Draft Hub.
def parse_sleeper_projections(payload: list[dict[str, Any]]) -> pd.DataFrame:
    """Tidy Sleeper season projections -> one row per QB/RB/WR/TE with sleeper_id + name +
    position + PPR ADP. Sleeper carries a raw season stat line (mapped via
    _sleeper_stats_to_statline); ADP-only rows leave the stat columns NA."""
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
        row: dict[str, object] = {
            "sleeper_id": str(pid),
            "full_name": full_name,
            "position": position,
            "sleeper_adp": stats.get("adp_ppr"),
        }
        statline = _sleeper_stats_to_statline(stats)
        if statline is not None:
            row.update(statline)
        rows.append(row)
    # Fixed column set so the STAT_FIELDS columns exist (NA) even for ADP-only rows / an empty
    # pull — the has_stats=True path in _to_canonical reads them unconditionally.
    return pd.DataFrame(
        rows, columns=["sleeper_id", "full_name", "position", "sleeper_adp", *STAT_FIELDS]
    )


def _make_placeholder_gsis(full_name: str, position: str) -> str:
    """Deterministic synthetic gsis_id for a player not in id_map (e.g. a pre-camp rookie).
    Matches GSIS_ID_PATTERN with a reserved 99- prefix. Keyed on the normalized (full_name,
    position) — NOT the per-source player id — so the SAME rookie gets the SAME placeholder
    from ESPN and Sleeper and a gsis_id join still unifies them (a source-scoped key would
    fork every rookie into two phantom players). Residual limits: two distinct players sharing
    a normalized name+position collide (rare), and the 99-XXXXXXX space is 10^7, so a within-
    pull hash collision is possible at hundreds of rookies — refresh logs any that occur."""
    digest = hashlib.sha1(placeholder_name_key(full_name, position).encode()).hexdigest()
    return f"99-{int(digest, 16) % 10_000_000:07d}"


def _attach_gsis_id(df: pd.DataFrame, id_map: pd.DataFrame, *, id_col: str) -> pd.DataFrame:
    """Left-join df to id_map on `id_col` (espn_id/sleeper_id) to attach a real gsis_id;
    unmatched rows get a deterministic placeholder keyed on (full_name, position). Adds
    `gsis_id` + `is_placeholder_gsis`. Dedupes the crosswalk on `id_col` so a duplicate
    id_map mapping can't multiply rows."""
    crosswalk = id_map[["gsis_id", id_col]].dropna(subset=[id_col]).drop_duplicates(subset=[id_col])
    # Align the join-key dtype on both sides (parsed ids are object str, id_map ids are pyarrow
    # string) so the merge can't silently miss on a cross-extension-dtype compare and send every
    # veteran down the placeholder path.
    df = df.copy()
    df[id_col] = df[id_col].astype(_PYARROW_STR)
    crosswalk[id_col] = crosswalk[id_col].astype(_PYARROW_STR)
    merged = df.merge(crosswalk, on=id_col, how="left")
    mask = merged["gsis_id"].isna()
    merged["is_placeholder_gsis"] = mask
    # Cast to object so .loc can fill NA slots with a plain str (pyarrow StringDtype rejects
    # mixed assignment); _finish_canonical recasts the column back to _PYARROW_STR downstream.
    merged["gsis_id"] = merged["gsis_id"].astype("object")
    merged.loc[mask, "gsis_id"] = [
        _make_placeholder_gsis(name, pos)
        for name, pos in zip(
            merged.loc[mask, "full_name"], merged.loc[mask, "position"], strict=True
        )
    ]
    return merged


_CANONICAL_STR_COLS = ("source", "source_player_id", "gsis_id", "full_name", "position", "asof")


def _finish_canonical(df: pd.DataFrame, *, season: int, asof: date) -> pd.DataFrame:
    # Mutates df in place; callers always pass a freshly built frame, so no copy needed.
    df["season"] = season
    df["asof"] = asof.isoformat()
    for c in _CANONICAL_STR_COLS:
        df[c] = df[c].astype(_PYARROW_STR)
    return df


def _to_canonical(
    df: pd.DataFrame,
    *,
    source: ProjectionSource,
    id_col: str,
    adp_col: str,
    rank_col: str | None,
    has_stats: bool,
    season: int,
    asof: date,
    id_map: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize a parsed source frame to the canonical ExternalProjectionSchema shape: attach
    gsis_id (real or placeholder), rename source-specific columns onto canonical names, and null
    out whatever the source doesn't carry (Sleeper has no stat line or draft rank). Numeric
    columns pass through untouched — ExternalProjectionSchema(coerce=True) casts them to Float64."""
    keyed = _attach_gsis_id(df, id_map, id_col=id_col)
    null_col = pd.array([pd.NA] * len(keyed), dtype="Float64")
    out = pd.DataFrame(
        {
            "source": source.value,
            "source_player_id": keyed[id_col],
            "gsis_id": keyed["gsis_id"],
            "is_placeholder_gsis": keyed["is_placeholder_gsis"],
            "full_name": keyed["full_name"],
            "position": keyed["position"],
            "adp": keyed[adp_col],
            "espn_draft_rank": keyed[rank_col] if rank_col else null_col,
        }
    )
    for f in STAT_FIELDS:
        out[f] = keyed[f] if has_stats else null_col
    # Uniform nullable-float dtype across all source frames so pd.concat needs no dtype inference
    # over all-NA columns (e.g. Sleeper's espn_draft_rank) — avoids the all-NA-column FutureWarning.
    for col in ("adp", "espn_draft_rank", *STAT_FIELDS):
        out[col] = out[col].astype("Float64")
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


def _warn_on_placeholder_collisions(frame: pd.DataFrame) -> None:
    """Log if two DISTINCT rookies hashed to the same placeholder gsis_id (bounded 10^7 space).
    Distinct = different normalized name+position; the same rookie appearing under ESPN and
    Sleeper shares one key by design and is NOT a collision."""
    placeholders = frame.loc[frame["is_placeholder_gsis"], ["gsis_id", "full_name", "position"]]
    if placeholders.empty:
        return
    keys = [
        placeholder_name_key(name, pos)
        for name, pos in zip(placeholders["full_name"], placeholders["position"], strict=True)
    ]
    # Distinct (gsis_id, name-key) pairs; a gsis_id appearing in more than one pair means two
    # different rookies hashed to the same placeholder.
    pairs = pd.DataFrame(
        {"gsis_id": placeholders["gsis_id"].to_numpy(), "key": keys}
    ).drop_duplicates()
    counts = pairs["gsis_id"].value_counts()
    colliding = counts[counts > 1]
    if not colliding.empty:
        _log.warning(
            "external_projections: %d placeholder gsis_id(s) shared by distinct rookies "
            "(hash collision in the bounded 99-XXXXXXX space): %s",
            len(colliding),
            colliding.index.tolist(),
        )


def refresh_external_projections(data_root: Path, *, season: int, asof: date | None = None) -> Path:
    """Fetch ESPN + Sleeper preseason projections, crosswalk to gsis_id (placeholder for
    rookies), validate, and write one dated snapshot. `asof` defaults to today (UTC). A pull is
    refused only if BOTH sources are empty; a single empty source is logged and the other is
    written (losing a good single-source snapshot would be worse than a partial one)."""
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
    if espn.empty and sleeper.empty:
        raise ExternalProjectionError(
            f"Empty pull for {season} (both ESPN and Sleeper returned 0 rows) — "
            f"refusing to write an empty asof snapshot."
        )

    id_map = read_partition(data_root / "raw", "id_map")
    # (parsed frame, source, id_col, adp_col, rank_col, has_stats). A single empty source is
    # logged and skipped; only the all-empty pull above is refused.
    source_specs: list[tuple[pd.DataFrame, ProjectionSource, str, str, str | None, bool]] = [
        (espn, ProjectionSource.ESPN, "espn_id", "espn_adp", "espn_pos_rank", True),
        (sleeper, ProjectionSource.SLEEPER, "sleeper_id", "sleeper_adp", None, True),
    ]
    frames: list[pd.DataFrame] = []
    for parsed, source, id_col, adp_col, rank_col, has_stats in source_specs:
        if parsed.empty:
            _log.warning(
                "%s returned 0 rows for %s; writing the snapshot without it.", source.value, season
            )
            continue
        frames.append(
            _to_canonical(
                parsed,
                source=source,
                id_col=id_col,
                adp_col=adp_col,
                rank_col=rank_col,
                has_stats=has_stats,
                season=season,
                asof=asof,
                id_map=id_map,
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    frame = ExternalProjectionSchema.validate(frame)
    _warn_on_placeholder_collisions(frame)
    _log.info(
        "external_projections season=%s asof=%s: wrote %d rows (espn=%d, sleeper=%d, "
        "placeholders=%d).",
        season,
        asof.isoformat(),
        len(frame),
        len(espn),
        len(sleeper),
        int(frame["is_placeholder_gsis"].sum()),
    )
    out = write_partition(
        data_root / "raw", "external_projections", frame, season=season, asof=asof
    )
    # Manifest keys on (table, season), so for this asof-snapshotted table it records the LATEST
    # refresh (newest snapshot's rowcount/checksum) — the freshness signal that matters. Older
    # snapshots stay on disk under their own asof= partitions; the manifest doesn't enumerate them.
    record_manifest(data_root, table="external_projections", season=season, df=frame)
    return out


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
