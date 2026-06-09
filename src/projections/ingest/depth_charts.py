"""Refresh per-season depth charts.

Two upstream formats are supported, dispatched on the columns present in the
raw payload returned by `nflreadpy.load_depth_charts`:

- **Legacy (≤2024)** — weekly per-team rows keyed on
  `(season, week, club_code, depth_team, depth_position)`. `depth_team` is the
  raw slot label (e.g., "WR1", "LWR"); `depth_position` is the numeric depth
  rank. `_parse_depth_rank` resolves these into a single canonical
  `depth_rank` int.
- **Snapshot-by-timestamp (2025+)** — `(dt, team, gsis_id, pos_abb, pos_slot,
  pos_rank, ...)`. No `season`/`week`/`club_code`. `_derive_weekly_snapshots_from_new_format`
  derives the legacy shape by joining each team-week's kickoff (from schedules)
  against `dt` and picking the closest-prior snapshot per team-week.
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
    DepthChartsSchema,
    Position,
    SchedulesSchema,
    normalize_team_code,
)
from projections.store import read_partition, write_partition

_log = logging.getLogger(__name__)

_KEEP = ["gsis_id", "season", "week", "team", "position", "depth_team", "depth_rank"]
_RENAME = {"club_code": "team"}
_TRAILING_DIGITS = re.compile(r"(\d+)$")

# Signature columns identifying the 2025+ snapshot-by-timestamp release.
_NEW_FORMAT_REQUIRED_COLS = frozenset({"dt", "team", "gsis_id", "pos_abb", "pos_slot", "pos_rank"})


def _fetch_raw_depth_charts(seasons: list[int]) -> pd.DataFrame:
    return nflreadpy.load_depth_charts(seasons=seasons).to_pandas()


def _normalize_team(v: str) -> str:
    return normalize_team_code(v).value


def _parse_depth_rank(*, depth_team: str | None, depth_position: int | None) -> tuple[int, bool]:
    """Resolve a numeric `depth_rank` from raw inputs.

    Returns (rank, warned). `warned` is True if we had to fall back to 1
    because the inputs were unrankable, OR if the parsed rank was clamped
    from an out-of-range value, so the caller can log once with a
    representative example.
    """
    if depth_position is not None and not pd.isna(depth_position):
        try:
            return min(10, max(1, int(depth_position))), False
        except (ValueError, TypeError):
            pass
    if depth_team is not None and not pd.isna(depth_team):
        match = _TRAILING_DIGITS.search(str(depth_team))
        if match:
            parsed = int(match.group(1))
            if parsed >= 1:
                return min(10, parsed), parsed > 10
    return 1, True


_LEGACY_REQUIRED_COLS = {"club_code", "depth_team", "depth_position", "season", "week"}


def _derive_weekly_snapshots_from_new_format(
    raw: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    """Derive `DepthChartsSchema`-shaped rows from the 2025+ snapshot feed.

    For each `(team, season, week)` in `schedules`, picks the snapshot with the
    largest `dt` strictly before that team's kickoff (closest-prior snapshot),
    pulls the snapshot's rows for that team, filters to `Position`-enum
    positions, and synthesizes legacy `depth_team` / `depth_rank` from
    `pos_rank`.

    `raw` must contain `_NEW_FORMAT_REQUIRED_COLS`; `schedules` must be
    `SchedulesSchema`-shaped (validated tz-UTC `kickoff`).
    """
    missing_cols = _NEW_FORMAT_REQUIRED_COLS - set(raw.columns)
    if missing_cols:
        raise ValueError(f"new-format payload missing required columns: {sorted(missing_cols)}")

    raw = raw.copy()
    # Coerce to microsecond resolution to match SchedulesSchema's `kickoff` (unit="us",
    # from polars/nflreadpy `.to_pandas()`). pandas 2.3 `merge_asof` rejects mixed
    # datetime64[us] / [ns] keys, so both sides of the per-team merge must agree.
    raw["dt"] = pd.to_datetime(raw["dt"], utc=True).dt.as_unit("us")
    # Normalize team codes BEFORE the per-team groupby/merge — raw nflverse
    # uses `JAX` and `LA` while validated schedules use `JAC` and `LAR`
    # (`normalize_team_code` semantics). Without this, the team-match drops
    # all JAC/LAR depth-chart rows silently.
    raw["team"] = raw["team"].map(_normalize_team)

    # Validate schedules to guarantee tz-UTC `kickoff` and required columns.
    schedules = SchedulesSchema.validate(schedules)

    # Melt schedules to (season, week, team, kickoff) — one row per team-game.
    home = schedules[["season", "week", "home_team", "kickoff"]].rename(
        columns={"home_team": "team"}
    )
    away = schedules[["season", "week", "away_team", "kickoff"]].rename(
        columns={"away_team": "team"}
    )
    team_kickoffs = pd.concat([home, away], ignore_index=True)
    team_kickoffs = team_kickoffs[team_kickoffs["kickoff"].notna()].copy()

    # Per-team merge_asof: find largest dt strictly before each team-week kickoff.
    # 32 teams * 22 weeks ~= 700 iterations -- trivial overhead.
    pieces: list[pd.DataFrame] = []
    missing_team_weeks = 0
    for team, snapshots in raw.groupby("team", sort=False):
        dt_keys = snapshots[["dt"]].drop_duplicates().sort_values("dt").reset_index(drop=True)
        tw = team_kickoffs[team_kickoffs["team"] == team].sort_values("kickoff")
        if tw.empty:
            continue
        matched = pd.merge_asof(
            tw,
            dt_keys.rename(columns={"dt": "snapshot_dt"}),
            left_on="kickoff",
            right_on="snapshot_dt",
            direction="backward",
            allow_exact_matches=False,
        )
        if matched["snapshot_dt"].isna().any():
            missing_team_weeks += int(matched["snapshot_dt"].isna().sum())
            matched = matched[matched["snapshot_dt"].notna()].copy()
        if matched.empty:
            continue
        enriched = matched.merge(
            snapshots.rename(columns={"dt": "snapshot_dt"}),
            on=["team", "snapshot_dt"],
            how="left",
        )
        pieces.append(enriched)

    if missing_team_weeks:
        _log.warning(
            "Skipped %d (team, week) cells with no closest-prior depth-chart snapshot.",
            missing_team_weeks,
        )

    if not pieces:
        empty = pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=_PYARROW_STR),
                "season": pd.Series([], dtype="int64"),
                "week": pd.Series([], dtype="int64"),
                "team": pd.array([], dtype=_PYARROW_STR),
                "position": pd.array([], dtype=_PYARROW_STR),
                "depth_team": pd.array([], dtype=_PYARROW_STR),
                "depth_rank": pd.Series([], dtype="int64"),
            }
        )
        return DepthChartsSchema.validate(empty)

    out = pd.concat(pieces, ignore_index=True)

    pos_values = {p.value for p in Position}
    out = out[out["pos_abb"].isin(pos_values)].copy()
    out = out[out["gsis_id"].notna()].copy()

    depth_rank = out["pos_rank"].clip(lower=1, upper=10).astype("int64")
    out["depth_rank"] = depth_rank
    out["depth_team"] = depth_rank.astype(str).astype(_PYARROW_STR)
    out["position"] = out["pos_abb"].astype(_PYARROW_STR)
    # `team` is already normalized (see early-loop normalization above) — just
    # coerce the dtype here.
    out["team"] = out["team"].astype(_PYARROW_STR)
    out["gsis_id"] = out["gsis_id"].astype(_PYARROW_STR)
    out["season"] = out["season"].astype("int64")
    out["week"] = out["week"].astype("int64")

    out = out[_KEEP].copy()
    # Defensive dedupe: a player should appear at most once per team-snapshot.
    out = out.drop_duplicates(subset=["gsis_id", "season", "week", "team"]).reset_index(drop=True)
    out = DepthChartsSchema.validate(out)
    return out


def _normalize_one_season(raw: pd.DataFrame, schedules: pd.DataFrame | None = None) -> pd.DataFrame:
    missing = _LEGACY_REQUIRED_COLS - set(raw.columns)
    if missing:
        if "dt" in raw.columns and not (_NEW_FORMAT_REQUIRED_COLS - set(raw.columns)):
            if schedules is None:
                raise ValueError(
                    "post-2025 depth_charts payload requires a schedules frame; "
                    "pass schedules=... or use refresh_depth_charts (which reads "
                    "data/raw/schedules from disk)."
                )
            return _derive_weekly_snapshots_from_new_format(raw, schedules)
        raise NotImplementedError(
            f"depth_charts upstream missing legacy columns {sorted(missing)} — and "
            "the new-format signature columns are not present either. Unknown release "
            "shape; investigate."
        )

    df = raw.rename(columns=_RENAME).copy()

    # Resolve depth_rank row-by-row; track if any rows fell back to 1 unranked.
    ranks: list[int] = []
    fallback_count = 0
    sample_label: str | None = None
    for _, row in df.iterrows():
        rank, warned = _parse_depth_rank(
            depth_team=row.get("depth_team"),
            depth_position=row.get("depth_position"),
        )
        ranks.append(rank)
        if warned:
            fallback_count += 1
            if sample_label is None:
                sample_label = str(row.get("depth_team"))
    if fallback_count:
        _log.warning(
            "Fell back to depth_rank=1 for %d rows (e.g., depth_team=%r). "
            "These are unrankable labels (alignment-based or out-of-range numeric).",
            fallback_count,
            sample_label,
        )
    df["depth_rank"] = ranks

    # Drop rows with NaN season/week (corrupt rows that would coerce to 0
    # and fail schema validation downstream).
    df = df[df["season"].notna() & df["week"].notna()].copy()
    # Upstream returns int32 for season/week; pandera Series[int] requires int64.
    for int_col in ("season", "week", "depth_rank"):
        if int_col in df.columns:
            df[int_col] = df[int_col].astype("int64")

    df = df[df["gsis_id"].notna()].copy()
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].map(_normalize_team).astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)
    df["depth_team"] = df["depth_team"].astype(_PYARROW_STR)

    df = df[df["position"].isin([p.value for p in Position])].copy()
    df = df[[c for c in _KEEP if c in df.columns]].copy()
    df = DepthChartsSchema.validate(df)
    return df


def refresh_depth_charts(
    data_root: Path,
    *,
    seasons: Iterable[int],
    schedules: pd.DataFrame | None = None,
) -> list[Path]:
    """Fetch and write depth charts for each season. Idempotent.

    For 2025+ seasons (snapshot-by-timestamp upstream), `_normalize_one_season`
    requires a schedules frame to resolve `(season, week, team)` from snapshot
    `dt`. When `schedules` is None, reads `data/raw/schedules/season=<s>/` from
    disk. Tests pass `schedules` directly. Pre-2025 seasons ignore `schedules`.
    """
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_depth_charts([season])
        season_schedules = schedules
        if (
            season_schedules is None
            and not (_NEW_FORMAT_REQUIRED_COLS - set(raw.columns))
            and "dt" in raw.columns
        ):
            try:
                season_schedules = read_partition(data_root / "raw", "schedules", season=season)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"depth_charts season={season} uses the post-2025 snapshot format and "
                    "requires the schedules partition to derive (season, week); ingest "
                    "schedules first (refresh_schedules) or pass schedules=...."
                ) from exc
        df = _normalize_one_season(raw, schedules=season_schedules)
        path = write_partition(data_root / "raw", "depth_charts", df, season=season, week=None)
        record_manifest(data_root, table="depth_charts", season=season, df=df)
        written.append(path)
    return written
