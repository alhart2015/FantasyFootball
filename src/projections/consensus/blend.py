"""Pure blend of one external_projections snapshot into per-player consensus rows.

Groups by gsis_id: mean ADP across sources (-> ordinal consensus_rank), mean stat line across
the sources that carry one (-> fantasy points via scoring.expected_points). Union coverage: every
player ranked by >=1 source appears. No I/O; the orchestrator (refresh.py) handles
read/validate/write.
"""

from __future__ import annotations

import pandas as pd

from projections.schemas import STAT_FIELDS, Ruleset
from projections.scoring import expected_points

MIN_STAT_FIELDS = 2  # a source row must have >= this many non-null, non-zero STAT_FIELDS to count


def _stat_bearing_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: True for rows with >= MIN_STAT_FIELDS STAT_FIELDS values that are non-null
    AND non-zero. Both source statline constructors zero-fill absent fields, so a degenerate stub
    arrives as an all-0.0 (fully non-null) row; testing non-zero is what keeps it out of the blend.
    Computed in one vectorized pass over the whole frame (not per-row)."""
    vals = df[list(STAT_FIELDS)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return (vals != 0).sum(axis=1) >= MIN_STAT_FIELDS


_OUTPUT_COLUMNS: tuple[str, ...] = (
    "gsis_id",
    "season",
    "asof",
    "full_name",
    "position",
    "consensus_adp",
    "consensus_rank",
    "n_adp_sources",
    "has_points",
    "projected_points_ppr",
    *STAT_FIELDS,
    "is_placeholder_gsis",
    "ruleset",
)


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in _OUTPUT_COLUMNS})


def build_consensus(external: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
    """Blend one validated external_projections snapshot into ConsensusProjectionSchema-shaped rows.

    `external` carries one row per (source, gsis_id), all sharing one season + asof. Returns one row
    per gsis_id; pass the result through ConsensusProjectionSchema.validate at the call site.
    """
    if external.empty:
        return _empty_output()

    season = int(external["season"].iloc[0])
    asof = str(external["asof"].iloc[0])

    # Stat-bearing rows (>= MIN_STAT_FIELDS non-null, non-zero STAT_FIELDS) — computed once,
    # vectorized, over the whole frame; degenerate all-zero stubs (e.g. ESPN's 2023 historical
    # season projection) are excluded so they can't pull the per-field blend mean toward zero.
    bearing_mask = _stat_bearing_mask(external)

    records: list[dict[str, object]] = []
    for gsis_id, grp in external.groupby("gsis_id", sort=False):
        # Defensive: a published consensus_adp must be > 0 (ConsensusProjectionSchema), so no
        # source's ADP may be non-positive. The known ESPN "undrafted = 0" sentinel is normalized
        # to null at the ingest parser; this guard keeps the contract robust for any other source.
        adp_vals = grp["adp"].dropna()
        adp_vals = adp_vals[adp_vals > 0]
        n_adp_sources = int(adp_vals.shape[0])
        consensus_adp: float | None = float(adp_vals.mean()) if n_adp_sources > 0 else None

        bearing = grp[bearing_mask.loc[grp.index]]

        # Prefer a stat-bearing row for identity (full_name/position); fall back to the first row.
        identity_row = bearing.iloc[0] if not bearing.empty else grp.iloc[0]

        statline: dict[str, float] = {}
        has_points = False
        for field in STAT_FIELDS:
            vals = bearing[field].dropna()
            if not vals.empty:
                statline[field] = float(vals.mean())
                has_points = True

        projected = expected_points(statline, ruleset) if has_points else None

        rec: dict[str, object] = {
            "gsis_id": str(gsis_id),
            "season": season,
            "asof": asof,
            "full_name": str(identity_row["full_name"]),
            "position": str(identity_row["position"]),
            "consensus_adp": consensus_adp,
            "n_adp_sources": n_adp_sources,
            "has_points": has_points,
            "projected_points_ppr": projected,
            "is_placeholder_gsis": bool(identity_row["is_placeholder_gsis"]),
            "ruleset": ruleset.name,
        }
        for field in STAT_FIELDS:
            rec[field] = statline.get(field, pd.NA)
        records.append(rec)

    df = pd.DataFrame.from_records(records)

    # Ordinal rank over non-null consensus_adp, deterministic tie-break on gsis_id.
    df = df.sort_values(["consensus_adp", "gsis_id"], na_position="last").reset_index(drop=True)
    ranked = df["consensus_adp"].notna()
    df["consensus_rank"] = pd.NA
    df.loc[ranked, "consensus_rank"] = range(1, int(ranked.sum()) + 1)

    # Nullable-aware dtypes (schema coerces, but be explicit so pd.NA survives).
    df["season"] = df["season"].astype("Int64")
    df["n_adp_sources"] = df["n_adp_sources"].astype("Int64")
    df["consensus_rank"] = df["consensus_rank"].astype("Int64")

    return df[list(_OUTPUT_COLUMNS)]
