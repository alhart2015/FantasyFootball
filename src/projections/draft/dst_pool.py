"""Turn the stored D/ST stat vectors into pool rows the VORP generator can consume.

The bridge between `ingest.external_projections.refresh_dst_projections` (which stores stats)
and `draft.vorp.generate_vorp_table` (which wants one season projection per player). Scoring
happens here, at read time, under the caller's ruleset -- which is the whole reason the stored
table holds stats rather than points: two leagues score the same defense differently.

Without this, a league with a D/ST roster slot has no defenses in its pool, and every
downstream tool reports each rostered defense as an unprojectable player and skips it
(issue #166).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from projections.schemas import (
    _PYARROW_STR,
    DST_TEAM_BY_GSIS,
    DstProjectionSchema,
    GsisId,
    Position,
    ProjectionSeasonSchema,
    Ruleset,
)
from projections.scoring.dst import score_dst
from projections.store import read_latest_partition, read_partition

#: Model id stamped on the rows, so a pool row's provenance is readable on the table itself.
MODEL_ID = "espn_dst_passthrough"


#: What `full_name` reads for a defense. Matches `ingest.id_map.dst_id_map_rows` so a
#: name-based lookup resolves to the same string on both sides.
def dst_display_name(gsis_id: str) -> str:
    return f"{DST_TEAM_BY_GSIS[GsisId(gsis_id)].value} D/ST"


class DstPoolError(RuntimeError):
    """Raised when defenses are needed but cannot be produced."""


def load_dst_season_projections(
    data_root: Path,
    *,
    season: int,
    ruleset: Ruleset,
    generated_at: pd.Timestamp,
    asof: date | None = None,
) -> pd.DataFrame:
    """Stored D/ST stat vectors -> `ProjectionSeasonSchema` rows, plus a `full_name` column.

    One row per defense. `season_mean` is `score_dst` of that defense's whole stat vector under
    `ruleset`. The percentile columns are left at the mean: the season simulators do not read
    them (they sample through `performance_variance`, which is keyed off `season_mean_fpts`),
    and inventing a spread here would put an unfitted number where a fitted one already exists.

    Raises:
        DstPoolError: If the snapshot is missing, or `ruleset` scores no D/ST categories. Both
            would otherwise surface as a league that quietly has no defenses.
    """
    # Checked before reading: score_dst would raise DstScoringError deep inside the loop, which
    # callers do not catch (they catch DstPoolError, as this docstring promises). A league with
    # a DST slot and no D/ST scoring must fail with the actionable message, not a traceback.
    if not ruleset.scores_dst:
        raise DstPoolError(
            f"Ruleset {ruleset.name!r} scores no D/ST categories, so defenses cannot be priced. "
            "If this league rosters a D/ST, re-derive its config from the ESPN payload "
            "(`python -m projections.ingest.espn_league`) so pointsOverrides['16'] is parsed."
        )
    try:
        # read_partition for an explicit asof, read_latest_partition otherwise. Filtering the
        # latest frame by an older asof would silently return nothing and report it as "empty",
        # never having read the snapshot the caller asked for.
        raw = (
            read_partition(data_root / "raw", "dst_projections", season=season, asof=asof)
            if asof is not None
            else read_latest_partition(data_root / "raw", "dst_projections", season=season)
        )
    except (FileNotFoundError, ValueError) as exc:
        raise DstPoolError(
            f"No dst_projections snapshot for season {season} under {data_root / 'raw'}"
            f"{f' at asof={asof.isoformat()}' if asof else ''}. "
            "Build it with `python -m projections.ingest.external_projections "
            f"--season {season}`."
        ) from exc
    raw = DstProjectionSchema.validate(raw)
    if raw.empty:
        raise DstPoolError(f"dst_projections for season {season} is empty.")

    rows: list[dict[str, object]] = []
    for gsis_id, group in raw.groupby("gsis_id", sort=True):
        points = score_dst(dict(zip(group["stat_id"], group["value"], strict=True)), ruleset)
        rows.append(
            {
                "gsis_id": str(gsis_id),
                "season": season,
                "position": Position.DST.value,
                "ruleset": ruleset.name,
                "n_weeks": 17,
                "season_mean": points,
                "season_p10": points,
                "season_p50": points,
                "season_p90": points,
                "model_id": MODEL_ID,
                "generated_at": generated_at,
                "full_name": dst_display_name(str(gsis_id)),
            }
        )

    out = pd.DataFrame(rows)
    out["gsis_id"] = out["gsis_id"].astype(_PYARROW_STR)
    out["position"] = out["position"].astype(_PYARROW_STR)
    out["full_name"] = out["full_name"].astype(_PYARROW_STR)
    # Validate the schema-owned columns, then restore full_name: ProjectionSeasonSchema is
    # strict="filter" and would drop it, and the caller needs it for the pool's display column.
    names = out[["gsis_id", "full_name"]]
    validated = ProjectionSeasonSchema.validate(out)
    return validated.merge(names, on="gsis_id", how="left")


__all__ = ["MODEL_ID", "DstPoolError", "dst_display_name", "load_dst_season_projections"]
