"""Trajectory feature family — career-arc / role / volume-trend signals.

Probe-only at this stage: the override produced by build_trajectory_overrides
is consumed by scripts/probe_feature_signal.py via the standard --override
mechanism. Schema integration into per-position FeaturesSchemas is deferred
to a SIGNAL-greenlit follow-up.

Each compute_* function returns every (gsis_id, season[, week]) combo with
the feature value. The assembler merges all per-week feature frames onto
the player-team-week index in one pass.

Spec: docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md.
"""

# pd and Position are referenced by compute_* fns added in subsequent tasks
# (Tasks 5-12); imported here so each task is a pure addition.
from __future__ import annotations

import re
from typing import Final

import pandas as pd

from projections.schemas import (  # noqa: F401  # Position used in subsequent tasks
    GSIS_ID_PATTERN,
    Position,
)

# DraftLookup maps gsis_id -> (draft_year, draft_age). draft_age may be NaN
# (drafted-but-missing-age, rare). Missing key: UDFA / pre-coverage; falls
# back to inferred draft year from earliest weekly_stats appearance.
DraftLookup = dict[str, tuple[int, float]]

_GSIS_RE: Final[re.Pattern[str]] = re.compile(rf"^{GSIS_ID_PATTERN}$")
_AGE_OFFSET_FALLBACK: Final[float] = 22.0  # mean entry age for inferred path


def compute_age(
    weekly_stats: pd.DataFrame,
    draft_lookup: DraftLookup,
) -> pd.DataFrame:
    """Per-(player, season) biological age in the target season.

    Primary path: if gsis_id is in draft_lookup AND draft_age is finite,
    age = draft_age + (season - draft_year).

    Fallback path: missing key OR NaN draft_age → uses inferred_draft_year
    (earliest weekly_stats season for the player); age = season -
    inferred_draft_year + _AGE_OFFSET_FALLBACK. The draft_year_inferred
    column is True for these rows so the override audit can track fallback
    frequency.

    Output: (gsis_id, season, age, draft_year_inferred).
    One row per (player, season) where the player has at least one
    weekly_stats row that season.
    """
    if weekly_stats.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=pd.StringDtype("pyarrow")),
                "season": pd.array([], dtype=pd.Int64Dtype()),
                "age": pd.array([], dtype=pd.Float64Dtype()),
                "draft_year_inferred": pd.array([], dtype=bool),
            }
        )

    # Earliest-season-played per player (for the fallback path).
    earliest = (
        weekly_stats.groupby("gsis_id", as_index=False, observed=True)["season"]
        .min()
        .rename(columns={"season": "inferred_draft_year"})
    )

    distinct = weekly_stats[["gsis_id", "season"]].drop_duplicates()
    merged = distinct.merge(earliest, on="gsis_id", how="left")

    def _age_row(row: pd.Series) -> tuple[float, bool]:
        entry = draft_lookup.get(row["gsis_id"])
        if entry is not None:
            draft_year, draft_age = entry
            if pd.notna(draft_age):
                return float(draft_age) + (int(row["season"]) - int(draft_year)), False
        # Fallback path.
        inferred = int(row["inferred_draft_year"])
        return float(row["season"]) - inferred + _AGE_OFFSET_FALLBACK, True

    age_inferred = merged.apply(_age_row, axis=1, result_type="expand")
    age_inferred.columns = ["age", "draft_year_inferred"]
    out = pd.concat([merged[["gsis_id", "season"]].reset_index(drop=True), age_inferred], axis=1)
    return out[["gsis_id", "season", "age", "draft_year_inferred"]].reset_index(drop=True)
