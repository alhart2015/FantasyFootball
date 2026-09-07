"""Should I change my lineup this week?

The blend is the reason this tool exists. A start/sit report built on ESPN's weekly
projection alone tells a manager what the ESPN app already shows him, which is no reason to
run anything. Two independent sources, scored under *this* league's ruleset, disagreeing
where the call is close — that is a number he cannot get anywhere else.

Keyed by **ESPN player id, not gsis_id**, for the reason `waivers` records: going through the
crosswalk drops exactly the just-signed players an in-season tool is about. Sleeper arrives
keyed by `sleeper_id` and is crosswalked *in*; a Sleeper row the `id_map` cannot place falls
back to ESPN alone rather than vanishing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from projections.ingest.external_projections import WEEKLY_BLEND_FIELDS
from projections.ingest.identity import normalize_join_id
from projections.schemas import Ruleset
from projections.scoring.score import expected_points

_ESPN_SUFFIX = "_espn"
_SLEEPER_SUFFIX = "_slp"


@dataclass(frozen=True)
class BlendedPoints:
    """One player's weekly points, and the evidence behind them."""

    #: Blended points under the league ruleset. The number the lineup is solved on.
    points: float
    #: ESPN's line scored alone, or None when ESPN did not price him.
    espn: float | None
    #: Sleeper's line scored alone, or None when Sleeper did not price him.
    sleeper: float | None
    #: "both" | "espn" | "sleeper" — printed, so a missing spread is legible rather than
    #: mysterious. A K or D/ST is always "espn": `parse_sleeper_weekly` filters to QB/RB/WR/TE.
    sources: str

    @property
    def spread(self) -> float | None:
        """How far apart the sources are. None unless both priced him."""
        if self.espn is None or self.sleeper is None:
            return None
        return abs(self.espn - self.sleeper)


def _score(line: Mapping[str, float], ruleset: Ruleset) -> float:
    return float(expected_points(dict(line), ruleset))


def _line(row: Mapping[str, Any], suffix: str) -> dict[str, float]:
    """The present, non-null stat fields of one side of the merge."""
    out: dict[str, float] = {}
    for field in WEEKLY_BLEND_FIELDS:
        value = row.get(f"{field}{suffix}")
        # `is None` first: the merge leaves a missing side as NaN/NA, but a column absent
        # entirely returns None from `.get`, and `pd.notna(None)` is False either way while
        # only the explicit check narrows the type for mypy.
        if value is None or pd.isna(value):
            continue
        out[field] = float(value)
    return out


def _sleeper_keyed_by_espn_id(sleeper: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """Crosswalk `sleeper_id` -> `espn_id`, dropping rows the map cannot place.

    Dropping is right here and would be wrong in the other direction: an unplaceable Sleeper
    row means we lose one source for a player ESPN still prices, which the `sources` column
    reports. Losing the ESPN row would lose the player.
    """
    crosswalk = (
        id_map[["espn_id", "sleeper_id"]]
        .dropna(subset=["espn_id", "sleeper_id"])
        .drop_duplicates("sleeper_id")
        .copy()
    )
    crosswalk["sleeper_id"] = normalize_join_id(crosswalk["sleeper_id"])
    crosswalk["espn_id"] = crosswalk["espn_id"].astype(str)

    keyed = sleeper.copy()
    keyed["sleeper_id"] = normalize_join_id(keyed["sleeper_id"])
    keyed = keyed.merge(crosswalk, on="sleeper_id", how="inner")
    return keyed.reindex(columns=["espn_id", *WEEKLY_BLEND_FIELDS])


def blend_weekly_points(
    espn: pd.DataFrame,
    sleeper: pd.DataFrame,
    id_map: pd.DataFrame,
    *,
    weight_espn: float,
    ruleset: Ruleset,
) -> dict[str, BlendedPoints]:
    """Blend two weekly stat lines per-stat, score once, key by ESPN id.

    `espn` is `espn_weekly.espn_weekly_statlines` shape; `sleeper` is
    `ingest.sleeper_weekly_projections.parse_sleeper_weekly` shape.

    **Per-stat, not per-player.** `Ruleset` is linear, so the two agree whenever both sources
    report the same fields — the difference is the field only one source carries. Sleeper omits
    receptions for a player and ESPN does not: that reception count belongs in the blend at
    ESPN's full weight, and averaging the two scored totals instead would read low by half of
    ESPN's reception points with nothing on screen to say so. Same rule as
    `dfs.blend.blend_statlines`.

    **A player neither source prices is absent from the result, not present at 0.0.**
    `choose_starters` reads a missing value as unstartable, which is how bye weeks work here
    without a rule about bye weeks; a 0.0 would instead let him fill a slot nobody else is
    eligible for.
    """
    if not 0.0 <= weight_espn <= 1.0:
        raise ValueError(f"weight_espn must be in [0, 1], got {weight_espn}")

    left = espn.reindex(columns=["espn_id", *WEEKLY_BLEND_FIELDS]).copy()
    left["espn_id"] = left["espn_id"].astype(str)
    right = _sleeper_keyed_by_espn_id(sleeper, id_map)

    # Reindexing both sides to the FULL field set above is what guarantees every field gets a
    # suffix on the merge. Without it a field present in only one frame stays unsuffixed and
    # `row.get(f"{field}_slp")` silently reads None -- that source's contribution vanishes.
    # `dfs.blend` documents the same trap.
    merged = left.merge(
        right,
        on="espn_id",
        how="outer",
        suffixes=(_ESPN_SUFFIX, _SLEEPER_SUFFIX),
        indicator=True,
    )

    out: dict[str, BlendedPoints] = {}
    for _, row in merged.iterrows():
        espn_line = _line(row, _ESPN_SUFFIX)
        sleeper_line = _line(row, _SLEEPER_SUFFIX)
        side = str(row["_merge"])
        has_espn = side in {"left_only", "both"}
        has_sleeper = side in {"right_only", "both"}

        blended: dict[str, float] = {}
        for field in WEEKLY_BLEND_FIELDS:
            weighted = [
                (weight_espn, espn_line.get(field)),
                (1.0 - weight_espn, sleeper_line.get(field)),
            ]
            total = sum(w * v for w, v in weighted if v is not None)
            weights = sum(w for w, v in weighted if v is not None)
            if weights > 0:
                blended[field] = total / weights

        out[str(row["espn_id"])] = BlendedPoints(
            points=_score(blended, ruleset),
            espn=_score(espn_line, ruleset) if has_espn else None,
            sleeper=_score(sleeper_line, ruleset) if has_sleeper else None,
            sources="both" if has_espn and has_sleeper else ("espn" if has_espn else "sleeper"),
        )
    return out
