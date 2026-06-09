"""Adapter: published consensus projection -> the season-projection contract VORP consumes.

The Draft Hub's `generate_vorp_table` accepts a `ProjectionSeasonSchema` frame; the consensus
layer (PR #55) publishes `ConsensusProjectionSchema`. This pure adapter bridges them so the
draft tooling is fed the draft-valid external consensus instead of the in-season model.

Point estimates only: `season_p10 == season_p50 == season_p90 == season_mean` is the honest
representation of a single-source point projection (a real band waits for >=2 stat-line
sources, a later consensus slice). See
docs/superpowers/specs/2026-06-09-draft-hub-consensus-design.md.
"""

from __future__ import annotations

import pandas as pd

from projections.schemas import _PYARROW_STR, ConsensusProjectionSchema, ProjectionSeasonSchema

# Sentinel n_weeks for a consensus-derived season total: "full season", not a per-week
# aggregate. 17 reads as a complete season (vs the misleading 1 = "one week of data").
# VORP does not read n_weeks; no consumer should filter consensus rows on it.
_FULL_SEASON_WEEKS = 17


def consensus_to_season_projections(consensus: pd.DataFrame) -> pd.DataFrame:
    """Convert one consensus snapshot into a ProjectionSeasonSchema frame.

    Filters to `has_points` players (VORP needs a non-null season-points total),
    maps `projected_points_ppr` -> `season_mean` with a degenerate point-mass band,
    and stamps provenance (`model_id = "consensus:<asof>"`). Carries the snapshot's
    `ruleset` so `generate_vorp_table`'s ruleset-match guard fires on a mismatch.

    Raises ValueError if the frame mixes `asof` snapshots or seasons (a caller bug;
    the CLI reads exactly one partition).
    """
    df = ConsensusProjectionSchema.validate(consensus)

    if not df.empty:
        asofs = df["asof"].unique()
        if len(asofs) > 1:
            raise ValueError(f"consensus frame mixes asof snapshots: {sorted(asofs)}")
        seasons = df["season"].unique()
        if len(seasons) > 1:
            raise ValueError(f"consensus frame mixes seasons: {sorted(seasons.tolist())}")
        asof = str(asofs[0])
        season = int(seasons[0])
    else:
        # Fully-empty input: produce a 0-row conforming frame (placeholders never reach a row).
        asof, season = "", 0

    points = df[df["has_points"]]
    season_mean = points["projected_points_ppr"].astype("float64").to_numpy()

    out = pd.DataFrame(
        {
            "gsis_id": points["gsis_id"].to_numpy(),
            "season": season,
            "position": points["position"].to_numpy(),
            "ruleset": points["ruleset"].to_numpy(),
            "n_weeks": _FULL_SEASON_WEEKS,
            "season_mean": season_mean,
            "season_p10": season_mean,
            "season_p50": season_mean,
            "season_p90": season_mean,
            "model_id": f"consensus:{asof}",
            "generated_at": pd.Timestamp.now(tz="UTC").as_unit("us"),
        }
    )
    for col in ("gsis_id", "position", "ruleset", "model_id"):
        out[col] = out[col].astype(_PYARROW_STR)
    return ProjectionSeasonSchema.validate(out)


__all__ = ["consensus_to_season_projections"]
