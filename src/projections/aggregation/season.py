"""Aggregate weekly per-player projections into season-total distributions.

Pure function over ProjectionWeeklySchema rows (no model coupling, no parquet
I/O). For each (gsis_id, season) group the function:

  - Decodes each week's per-stat distribution params via unpack_per_stat_params.
  - Re-derives each week's per-row seed via derive_row_seed.
  - Calls score_distribution(...) to regenerate that week's points samples.
  - Sums per-week sample arrays positionally -> n_samples season-total samples.
  - Summarizes mean / p10 / p50 / p90.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from projections.distributions import unpack_per_stat_params
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    ProjectionSeasonSchema,
    ProjectionWeeklySchema,
    Ruleset,
)
from projections.scoring import derive_row_seed, score_distribution


def aggregate_to_season(
    weekly: pd.DataFrame,
    *,
    ruleset: Ruleset,
    n_samples: int = 10_000,
) -> pd.DataFrame:
    """Aggregate weekly per-player projections into season-total distributions.

    The input is validated against ProjectionWeeklySchema. Every row must have
    family == DistributionFamily.SAMPLED_SUMMARY and ruleset == ruleset.name --
    a mixed-ruleset frame or a row written before Plan 3d's codec swap raises
    ValueError immediately.

    Returns a ProjectionSeasonSchema-validated DataFrame with one row per
    (gsis_id, season). position is the modal value across the input rows for
    that gsis_id (handles in-season position changes deterministically).

    Empty input returns an empty validated frame.
    """
    weekly = ProjectionWeeklySchema.validate(weekly)
    if weekly.empty:
        empty_cols = list(ProjectionSeasonSchema.to_schema().columns.keys())
        return ProjectionSeasonSchema.validate(pd.DataFrame(columns=empty_cols))

    bad_family = weekly[weekly["family"] != DistributionFamily.SAMPLED_SUMMARY.value]
    if not bad_family.empty:
        raise ValueError(
            f"aggregate_to_season requires family={DistributionFamily.SAMPLED_SUMMARY.value}, "
            f"found {bad_family['family'].unique().tolist()}"
        )

    bad_ruleset = weekly[weekly["ruleset"] != ruleset.name]
    if not bad_ruleset.empty:
        raise ValueError(
            f"Mixed-ruleset input: expected {ruleset.name}, "
            f"found {bad_ruleset['ruleset'].unique().tolist()}"
        )

    rows: list[dict[str, object]] = []
    generated_at = datetime.now(UTC)
    for (gsis_id, season), group in weekly.groupby(["gsis_id", "season"], sort=False):
        season_samples = np.zeros(n_samples, dtype=np.float64)
        for _idx, week_row in group.iterrows():
            per_stat_dists = unpack_per_stat_params(bytes(week_row["params"]))
            seed = derive_row_seed(
                gsis_id=str(gsis_id),
                season=int(season),
                week=int(week_row["week"]),
                ruleset_name=ruleset.name,
            )
            # NOTE: for DecomposedBaselineModel outputs, the params blob stores
            # decomposed stats as QuantileDistribution summaries (see
            # DecomposedBaselineModel._persistable_dists_for_packing in
            # src/projections/models/decomposed_baseline.py). The cross-stat
            # correlation from the original FrozenSampledDistribution's shared
            # volume draw is NOT recoverable here — score_distribution will draw
            # per-stat samples independently and the season-level variance for
            # stats that shared a volume axis will be slightly compressed
            # (smaller p10..p90 spread than the live weekly predict produced).
            # This is the v1 acceptable trade-off; revisit if a ProductDistribution
            # codec branch is added.
            week_dist = score_distribution(per_stat_dists, ruleset, n_samples=n_samples, seed=seed)
            season_samples += week_dist.samples

        position = group["position"].mode().iloc[0]
        rows.append(
            {
                "gsis_id": gsis_id,
                "season": int(season),
                "position": position,
                "ruleset": ruleset.name,
                "n_weeks": len(group),
                "season_mean": float(season_samples.mean()),
                "season_p10": float(np.quantile(season_samples, 0.1)),
                "season_p50": float(np.quantile(season_samples, 0.5)),
                "season_p90": float(np.quantile(season_samples, 0.9)),
                "model_id": group["model_id"].iloc[0],
                "generated_at": pd.Timestamp(generated_at).as_unit("us"),
            }
        )

    out = pd.DataFrame(rows)
    for col in ("gsis_id", "position", "ruleset", "model_id"):
        out[col] = out[col].astype(_PYARROW_STR)
    return ProjectionSeasonSchema.validate(out)
