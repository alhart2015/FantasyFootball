"""Pull + parse ESPN weekly projected stat lines (weeks 1-17) -> half-PPR weekly projections.

Mirrors scripts/pull_external_projections.py's ESPN parse but reads the single-week projection
entry (statSourceId=1, statSplitTypeId=1, scoringPeriodId=wk) and scores the fractional stat
line via scoring.expected_points. The espn_id->gsis_id crosswalk + store write live in
refresh_espn_weekly_projections.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from projections.ingest.external_projections import (
    _ESPN_URL,
    _UA,
    ESPN_POSITIONS,
    ESPN_STAT_IDS,
)
from projections.schemas import _PYARROW_STR, Ruleset, WeeklyProjectionSchema
from projections.scoring.score import expected_points
from projections.store import write_partition

_DEFAULT_WEEKS: range = range(1, 18)


def _weekly_proj_stats(player: dict[str, Any], week: int) -> dict[str, float] | None:
    """Return the raw ESPN stat dict for the single weekly projection entry, or None if absent.

    Matches statSourceId=1 (projected) + statSplitTypeId=1 (weekly) + scoringPeriodId=week.
    """
    for s in player.get("stats", []):
        if (
            s.get("scoringPeriodId") == week
            and s.get("statSourceId") == 1
            and s.get("statSplitTypeId") == 1
        ):
            return s.get("stats") or {}
    return None


def _statline_dict(raw: dict[str, float]) -> dict[str, float]:
    """Map ESPN string stat-ids to canonical StatLine field names.

    ESPN_STAT_IDS keys are strings (e.g. "24") matching the JSON stats dict keys directly.
    Fields absent from raw default to 0.0; fields not in ESPN_STAT_IDS are ignored.
    """
    line = {field: 0.0 for field in ESPN_STAT_IDS.values()}
    for sid, field in ESPN_STAT_IDS.items():
        if sid in raw:
            line[field] = float(raw[sid])
    return line


def parse_espn_weekly(
    payload: dict[str, Any],
    *,
    season: int,
    week: int,
    ruleset: Ruleset,
    skill_positions_only: bool = True,
) -> pd.DataFrame:
    """Parse a single-week ESPN kona_player_info payload -> one row per player.

    Columns: espn_id (str), season (int), week (int), position (str), projected_points (float|None).
    projected_points is None when no weekly projection entry exists for the player (e.g. bye week
    or missing data). A zero-stat projection entry produces 0.0, not None.

    `skill_positions_only=False` keeps kickers and defenses, whose `defaultPositionId` is not in
    `ESPN_POSITIONS`; they come back with an empty `position`. The waiver recommender needs them
    because it prices MY roster from this same feed, and a starter it cannot price is a starter
    it treats as unstartable.
    """
    rows: list[dict[str, Any]] = []
    for pl in payload.get("players", []):
        p = pl.get("player", {})
        position = ESPN_POSITIONS.get(p.get("defaultPositionId"))
        if position is None and skill_positions_only:
            continue
        raw = _weekly_proj_stats(p, week)
        proj = expected_points(_statline_dict(raw), ruleset) if raw is not None else None
        rows.append(
            {
                "espn_id": str(p.get("id")),
                "season": season,
                "week": week,
                "position": position or "",
                "projected_points": proj,
            }
        )
    return pd.DataFrame(rows)


def espn_weekly_statlines(
    payload: dict[str, Any], *, week: int, skill_positions_only: bool = False
) -> pd.DataFrame:
    """Parse the same weekly entry as `parse_espn_weekly` but return the line UNSCORED.

    Columns: espn_id (str), position (str), + the nine canonical stat fields.

    `parse_espn_weekly` scores this line immediately, which is right for a projection store
    and wrong for a blend: averaging two already-scored totals bakes in each source's scoring
    assumptions before the league's own `Ruleset` can override them. The start/sit blend
    averages per-stat and scores once (`midseason.start_sit.blend_weekly_points`), matching
    `consensus.blend` and `dfs.blend.blend_statlines`.

    **A player with no weekly projection entry is ABSENT, not present with a zero line.**
    `parse_espn_weekly` keeps him with `projected_points = None` because its schema wants one
    row per player; here the mapping is consumed by `choose_starters`, for which absence is
    unstartability. That is how bye weeks work with no rule about bye weeks.

    `skill_positions_only` defaults to **False**, the opposite of `parse_espn_weekly`, because
    every caller of this function prices a real roster and a roster starts a K and a D/ST.
    They arrive with an empty `position`, as they do from `parse_espn_weekly(...,
    skill_positions_only=False)`.
    """
    fields = list(ESPN_STAT_IDS.values())
    rows: list[dict[str, Any]] = []
    for pl in payload.get("players", []):
        p = pl.get("player", {})
        position = ESPN_POSITIONS.get(p.get("defaultPositionId"))
        if position is None and skill_positions_only:
            continue
        raw = _weekly_proj_stats(p, week)
        # A stat line sharing NO ids with `ESPN_STAT_IDS` cannot be priced from these nine
        # fields, and `_statline_dict` would return all zeros -- a fabricated 0.0 that reads
        # exactly like a real projection of nothing. D/ST is the live case: ESPN reports it in
        # the 100-block ids (sacks, points allowed) with zero overlap here, so a defense was
        # arriving priced at 0.0 rather than unpriced. `parse_espn_weekly` has the same hole
        # and does not care, because it is fed to a schema that expects one row per player;
        # here the value becomes a lineup decision.
        if raw is None or not set(raw) & set(ESPN_STAT_IDS):
            continue
        rows.append(
            {
                "espn_id": str(p.get("id")),
                "position": position or "",
                **_statline_dict(raw),
            }
        )
    return pd.DataFrame(rows, columns=["espn_id", "position", *fields])


def _fetch_espn_week(season: int, week: int, *, limit: int = 800) -> dict[str, Any]:
    """Fetch the ESPN kona_player_info payload for a single scoring period.

    Network-only; monkeypatched in tests. The URL is built from trusted int args
    only — no user input is interpolated.
    """
    flt = {
        "players": {
            "limit": limit,
            "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
        }
    }
    url = _ESPN_URL.format(season=season) + f"&scoringPeriodId={week}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _UA, "X-Fantasy-Filter": json.dumps(flt)},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)  # type: ignore[no-any-return]


def weekly_projections_for_weeks(
    *,
    season: int,
    weeks: Iterable[int],
    ruleset: Ruleset,
    id_map: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble weekly projections across multiple weeks with espn_id -> gsis_id crosswalk.

    Parses each week via parse_espn_weekly, inner-joins on espn_id, and returns a
    WeeklyProjectionSchema-validated DataFrame. Players not present in id_map are dropped.
    """
    cross = id_map[["espn_id", "gsis_id"]].dropna().astype({"espn_id": str})
    frames: list[pd.DataFrame] = []
    for wk in weeks:
        parsed = parse_espn_weekly(
            _fetch_espn_week(season, wk), season=season, week=wk, ruleset=ruleset
        )
        merged = parsed.merge(cross, on="espn_id", how="inner")
        frames.append(merged[["gsis_id", "season", "week", "position", "projected_points"]])
    out = pd.concat(frames, ignore_index=True)
    out["gsis_id"] = out["gsis_id"].astype(_PYARROW_STR)
    return WeeklyProjectionSchema.validate(out)


def refresh_espn_weekly_projections(
    *,
    season: int,
    ruleset: Ruleset,
    id_map: pd.DataFrame,
    data_root: Path,
    weeks: Iterable[int] = _DEFAULT_WEEKS,
) -> pd.DataFrame:
    """Fetch, crosswalk, validate, and persist ESPN weekly projections for all requested weeks.

    Writes to data_root/processed/espn_weekly_projections/season=YYYY/.
    Returns the validated DataFrame.
    """
    out = weekly_projections_for_weeks(season=season, weeks=weeks, ruleset=ruleset, id_map=id_map)
    write_partition(data_root / "processed", "espn_weekly_projections", out, season=season)
    return out
