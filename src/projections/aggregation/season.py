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
from typing import Literal, overload

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


@overload
def aggregate_to_season(
    weekly: pd.DataFrame,
    *,
    ruleset: Ruleset,
    n_samples: int = ...,
    return_samples: Literal[False] = ...,
) -> pd.DataFrame: ...


@overload
def aggregate_to_season(
    weekly: pd.DataFrame,
    *,
    ruleset: Ruleset,
    n_samples: int = ...,
    return_samples: Literal[True],
) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]: ...


def aggregate_to_season(
    weekly: pd.DataFrame,
    *,
    ruleset: Ruleset,
    n_samples: int = 10_000,
    return_samples: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    """Aggregate weekly per-player projections into season-total distributions.

    The input is validated against ProjectionWeeklySchema. Every row must have
    family in {SAMPLED_SUMMARY, QUANTILE, MIXED} and ruleset == ruleset.name --
    a mixed-ruleset frame or a row with an unsupported family raises ValueError
    immediately. The codec in projections.distributions handles each family's
    params blob; this function composes per-week samples and sums to season
    totals regardless of the row-level family tag.

    Returns a ProjectionSeasonSchema-validated DataFrame with one row per
    (gsis_id, season). position is the modal value across the input rows for
    that gsis_id (handles in-season position changes deterministically).

    Empty input returns an empty validated frame.

    When ``return_samples=True`` returns a 2-tuple ``(summary, samples)`` where
    ``samples`` is a dict keyed by ``(gsis_id, season)`` mapping to the
    per-player season-total MC sample array (length ``n_samples``). Useful for
    downstream consumers that need tail-probability estimates from the same
    draws used to compute the summary quantiles.
    """
    weekly = ProjectionWeeklySchema.validate(weekly)
    if weekly.empty:
        empty_cols = list(ProjectionSeasonSchema.to_schema().columns.keys())
        empty_validated = ProjectionSeasonSchema.validate(pd.DataFrame(columns=empty_cols))
        if return_samples:
            return empty_validated, {}
        return empty_validated

    _allowed_families = {
        DistributionFamily.SAMPLED_SUMMARY.value,
        DistributionFamily.QUANTILE.value,
        DistributionFamily.MIXED.value,
    }
    bad_family = weekly[~weekly["family"].isin(_allowed_families)]
    if not bad_family.empty:
        raise ValueError(
            f"aggregate_to_season requires family in {sorted(_allowed_families)}, "
            f"found {bad_family['family'].unique().tolist()}"
        )

    bad_ruleset = weekly[weekly["ruleset"] != ruleset.name]
    if not bad_ruleset.empty:
        raise ValueError(
            f"Mixed-ruleset input: expected {ruleset.name}, "
            f"found {bad_ruleset['ruleset'].unique().tolist()}"
        )

    rows: list[dict[str, object]] = []
    samples_out: dict[tuple[str, int], np.ndarray] = {}
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
        if return_samples:
            samples_out[(str(gsis_id), int(season))] = season_samples

    out = pd.DataFrame(rows)
    for col in ("gsis_id", "position", "ruleset", "model_id"):
        out[col] = out[col].astype(_PYARROW_STR)
    validated = ProjectionSeasonSchema.validate(out)
    if return_samples:
        return validated, samples_out
    return validated
