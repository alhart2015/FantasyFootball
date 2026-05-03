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

from __future__ import annotations

import re
from typing import Final

import pandas as pd

from projections.schemas import (
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
                "draft_year_inferred": pd.array([], dtype=pd.BooleanDtype()),
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
    return (
        out[["gsis_id", "season", "age", "draft_year_inferred"]]
        .astype(
            {
                "gsis_id": pd.StringDtype("pyarrow"),
                "season": pd.Int64Dtype(),
                "age": pd.Float64Dtype(),
                "draft_year_inferred": pd.BooleanDtype(),
            }
        )
        .reset_index(drop=True)
    )


def compute_is_rookie(
    weekly_stats: pd.DataFrame,
    draft_lookup: DraftLookup,
) -> pd.DataFrame:
    """Per-(player, season) rookie flag (1.0 if season == draft_year, else 0.0).

    For UDFAs / missing-from-lookup, uses the same inferred_draft_year
    fallback as compute_age (earliest weekly_stats season).

    Output: (gsis_id, season, is_rookie) — one row per (player, season)
    where the player has at least one weekly_stats row. is_rookie is
    Float64 for ML-compat.
    """
    if weekly_stats.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=pd.StringDtype("pyarrow")),
                "season": pd.array([], dtype=pd.Int64Dtype()),
                "is_rookie": pd.array([], dtype=pd.Float64Dtype()),
            }
        )

    earliest = (
        weekly_stats.groupby("gsis_id", as_index=False, observed=True)["season"]
        .min()
        .rename(columns={"season": "inferred_draft_year"})
    )
    distinct = weekly_stats[["gsis_id", "season"]].drop_duplicates()
    merged = distinct.merge(earliest, on="gsis_id", how="left")

    def _rookie_year(row: pd.Series) -> int:
        entry = draft_lookup.get(row["gsis_id"])
        if entry is not None:
            return int(entry[0])
        return int(row["inferred_draft_year"])

    rookie_years = merged.apply(_rookie_year, axis=1)
    out = pd.DataFrame(
        {
            "gsis_id": merged["gsis_id"].values,
            "season": merged["season"].values,
            "is_rookie": (merged["season"].values == rookie_years.values).astype(float),
        }
    )
    return out.astype(
        {
            "gsis_id": pd.StringDtype("pyarrow"),
            "season": pd.Int64Dtype(),
            "is_rookie": pd.Float64Dtype(),
        }
    ).reset_index(drop=True)


def _volume_trend(
    weekly_stats: pd.DataFrame,
    *,
    position: Position | tuple[Position, ...],
    value_col: str,
) -> pd.DataFrame:
    """Per-(player, season, week) volume trend on `value_col`, defined as
    mean over trailing-4 active games minus mean over prior-4 active games.

    Active game = game with a weekly_stats row for this player. Bye / IR /
    inactive weeks are not in weekly_stats and therefore excluded from the
    rolling denominator (treated as gaps, NOT as 0-value games).

    Within-player rolling: groups by gsis_id, sorts by (season, week). The
    trailing-4 window uses .rolling(4).mean().shift(1) — the row at week W
    reflects the mean over W-4..W-1 (NOT W). The prior-4 window uses
    .shift(5) — mean over W-8..W-5. Fewer than 8 prior active games yields
    NaN for prior_l4 (and therefore NaN for the trend).

    Position is a Position enum or tuple of Position enums; rows whose
    position is not in that set are excluded before the rolling computation.
    """
    if weekly_stats.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=pd.StringDtype("pyarrow")),
                "season": pd.array([], dtype=pd.Int64Dtype()),
                "week": pd.array([], dtype=pd.Int64Dtype()),
                "volume_trend_l4_minus_prior_l4": pd.array([], dtype=pd.Float64Dtype()),
            }
        )

    positions = (
        (position.value,) if isinstance(position, Position) else tuple(p.value for p in position)
    )
    filtered = weekly_stats[weekly_stats["position"].isin(positions)].copy()
    if filtered.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=pd.StringDtype("pyarrow")),
                "season": pd.array([], dtype=pd.Int64Dtype()),
                "week": pd.array([], dtype=pd.Int64Dtype()),
                "volume_trend_l4_minus_prior_l4": pd.array([], dtype=pd.Float64Dtype()),
            }
        )

    sorted_df = filtered.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)
    grouped = sorted_df.groupby("gsis_id", sort=False)[value_col]
    l4 = grouped.transform(
        lambda s: s.astype(float).rolling(window=4, min_periods=4).mean().shift(1)
    )
    prior_l4 = grouped.transform(
        lambda s: s.astype(float).rolling(window=4, min_periods=4).mean().shift(5)
    )
    sorted_df["volume_trend_l4_minus_prior_l4"] = l4 - prior_l4
    out = sorted_df[["gsis_id", "season", "week", "volume_trend_l4_minus_prior_l4"]]
    return out.astype(
        {
            "gsis_id": pd.StringDtype("pyarrow"),
            "season": pd.Int64Dtype(),
            "week": pd.Int64Dtype(),
            "volume_trend_l4_minus_prior_l4": pd.Float64Dtype(),
        }
    ).reset_index(drop=True)


def compute_qb_volume_trend(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """QB volume trend on `attempts`, trailing-4 minus prior-4 (active games)."""
    return _volume_trend(weekly_stats, position=Position.QB, value_col="attempts")


def compute_rb_volume_trend(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """RB volume trend on `carries`, trailing-4 minus prior-4 (active games)."""
    return _volume_trend(weekly_stats, position=Position.RB, value_col="carries")


def compute_wr_te_volume_trend(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """WR/TE volume trend on `targets`, trailing-4 minus prior-4 (active games)."""
    return _volume_trend(weekly_stats, position=(Position.WR, Position.TE), value_col="targets")


def compute_snap_pct_change(snap_counts: pd.DataFrame) -> pd.DataFrame:
    """Per-(player, season, week) change in offensive snap share, trailing-4
    minus prior-4 (active games — where active = has a snap_counts row).

    Output: (gsis_id, season, week, snap_pct_change_l4_vs_prior_l4).
    Players inactive that week (no snap_counts row) are skipped (not 0).
    """
    if snap_counts.empty:
        return pd.DataFrame(
            {
                "gsis_id": pd.array([], dtype=pd.StringDtype("pyarrow")),
                "season": pd.array([], dtype=pd.Int64Dtype()),
                "week": pd.array([], dtype=pd.Int64Dtype()),
                "snap_pct_change_l4_vs_prior_l4": pd.array([], dtype=pd.Float64Dtype()),
            }
        )

    sorted_df = snap_counts.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)
    grouped = sorted_df.groupby("gsis_id", sort=False)["offense_pct"]
    l4 = grouped.transform(
        lambda s: s.astype(float).rolling(window=4, min_periods=4).mean().shift(1)
    )
    prior_l4 = grouped.transform(
        lambda s: s.astype(float).rolling(window=4, min_periods=4).mean().shift(5)
    )
    sorted_df["snap_pct_change_l4_vs_prior_l4"] = l4 - prior_l4
    out = sorted_df[["gsis_id", "season", "week", "snap_pct_change_l4_vs_prior_l4"]]
    return out.astype(
        {
            "gsis_id": pd.StringDtype("pyarrow"),
            "season": pd.Int64Dtype(),
            "week": pd.Int64Dtype(),
            "snap_pct_change_l4_vs_prior_l4": pd.Float64Dtype(),
        }
    ).reset_index(drop=True)


def attach_trajectory_features(
    index: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    draft_lookup: DraftLookup,
    position: Position,
) -> pd.DataFrame:
    """Append the 4 trajectory features (+ informational draft_year_inferred)
    to a player-team-week index for one position.

    Args:
        index: (gsis_id, season, week, team, opp) — one row per player-week.
        weekly_stats: full weekly_stats frame (multiple positions OK; the
            position-specific volume_trend filters internally).
        snap_counts: full snap_counts frame.
        draft_lookup: gsis_id -> (draft_year, draft_age) lookup.
        position: which position is being processed (selects volume_trend variant).

    Returns:
        A copy of `index` with 5 columns appended:
            age, is_rookie, volume_trend_l4_minus_prior_l4,
            snap_pct_change_l4_vs_prior_l4, draft_year_inferred.
        Row count equals len(index).

    Raises:
        ValueError: position not in {QB, RB, WR, TE}.
    """
    if position == Position.QB:
        trend = compute_qb_volume_trend(weekly_stats)
    elif position == Position.RB:
        trend = compute_rb_volume_trend(weekly_stats)
    elif position in (Position.WR, Position.TE):
        trend = compute_wr_te_volume_trend(weekly_stats)
    else:
        raise ValueError(f"unsupported position for trajectory features: {position!r}")

    age = compute_age(weekly_stats, draft_lookup)
    is_rookie = compute_is_rookie(weekly_stats, draft_lookup)
    snap_change = compute_snap_pct_change(snap_counts)

    out = index.merge(age, on=["gsis_id", "season"], how="left")
    out = out.merge(is_rookie, on=["gsis_id", "season"], how="left")
    out = out.merge(trend, on=["gsis_id", "season", "week"], how="left")
    out = out.merge(snap_change, on=["gsis_id", "season", "week"], how="left")

    if len(out) != len(index):
        raise AssertionError(
            f"row count mismatch in attach_trajectory_features: input {len(index)}, "
            f"output {len(out)}; suggests a many-to-many merge regression"
        )

    return out
