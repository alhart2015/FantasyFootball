"""Vegas team-context features for the 33c family probe.

Sourced from `SchedulesSchema` columns (`spread_line`, `total_line`) already
in `data/raw/schedules`. Wraps `_shared.build_game_environment` to produce
per-team-game `(spread, implied_team_total)` rows in the canonical
sign convention (team-perspective: favorite negative, dog positive).

Two mechanism axes per spec §3:
  1. `preseason_*` — team's week-1 game values, broadcast across all weeks of
     the season. Vegas's preseason team-strength view.
  2. `season_avg_*` — leakage-safe expanding mean of weeks 1..N-1. As-of-time
     market view of team strength.

Probe-only — features land in the override parquet, not in `*FeaturesSchema`.
Integration follow-up is conditional on the family-probe verdict per
`docs/superpowers/specs/2026-05-17-vegas-team-context-probe-design.md`.
"""

from __future__ import annotations

import re
from typing import Final

import pandas as pd

from projections.features._shared import build_game_environment
from projections.schemas import GSIS_ID_PATTERN

# Used by build_vegas_team_context_overrides for gsis_id validation.
_GSIS_RE: Final[re.Pattern[str]] = re.compile(rf"^{GSIS_ID_PATTERN}$")

# Used by build_vegas_team_context_overrides for output column ordering.
_FEATURE_COLS: Final[tuple[str, ...]] = (
    "preseason_implied_team_total",
    "preseason_spread",
    "season_avg_implied_team_total",
    "season_avg_spread",
)

# Required columns on the player-team-week index passed to
# build_vegas_team_context_overrides.
_REQUIRED_INDEX_COLS: Final[tuple[str, ...]] = (
    "gsis_id",
    "season",
    "week",
    "team",
    "opp",
    "position",
)


def compute_vegas_team_context_features(schedules: pd.DataFrame) -> pd.DataFrame:
    """Per-team-game frame with four Vegas team-context features.

    One row per (game, team) — each schedules row produces two output rows
    (home + away). Both teams in a matchup carry their own
    team-perspective values (the home team's spread is the negative of the
    away team's spread, etc.).

    Spec §3:
      - `preseason_*`: broadcast week-1-game values across all weeks of the
        season for that team.
      - `season_avg_*`: expanding mean of weeks 1..N-1 (leakage-safe via
        .shift(1)). NaN at week 1.

    Args:
        schedules: frame validated against `SchedulesSchema` (must carry
            season, week, home_team, away_team, spread_line, total_line, roof).

    Returns:
        DataFrame sorted by (season, week, team) with columns:
            season, week, team,
            preseason_implied_team_total, preseason_spread,
            season_avg_implied_team_total, season_avg_spread.
        All four feature columns are nullable Float64. ``season`` / ``week`` /
        ``team`` are not re-cast here; they pass through as whatever
        ``_shared.build_game_environment`` returned (numpy int64 / object).
        Downstream attach / override builders re-cast to canonical dtypes.
    """
    game_env = build_game_environment(schedules)
    # Keep only the columns we need; cast spread / implied_team_total to Float64
    # for downstream-Float64 consistency (build_game_environment returns float).
    games = game_env[["season", "week", "team", "spread", "implied_team_total"]].copy()
    games["spread"] = games["spread"].astype("Float64")
    games["implied_team_total"] = games["implied_team_total"].astype("Float64")

    # Preseason broadcast: for each (season, team), look up the min-week row,
    # broadcast its spread + implied_team_total across all weeks.
    sorted_games = games.sort_values(["season", "team", "week"])
    first_week_idx = sorted_games.groupby(["season", "team"], as_index=False).head(1)
    preseason = first_week_idx[["season", "team", "spread", "implied_team_total"]].rename(
        columns={
            "spread": "preseason_spread",
            "implied_team_total": "preseason_implied_team_total",
        }
    )
    out = games.merge(preseason, on=["season", "team"], how="left")

    # season_avg_*: expanding mean shifted by 1 (so week-N row sees only
    # weeks 1..N-1). NaN at week 1.
    out = out.sort_values(["season", "team", "week"]).reset_index(drop=True)
    grouped = out.groupby(["season", "team"], group_keys=False)
    out["season_avg_spread"] = (
        grouped["spread"].apply(lambda s: s.expanding().mean().shift(1)).astype("Float64")
    )
    out["season_avg_implied_team_total"] = (
        grouped["implied_team_total"]
        .apply(lambda s: s.expanding().mean().shift(1))
        .astype("Float64")
    )

    return (
        out[
            [
                "season",
                "week",
                "team",
                "preseason_implied_team_total",
                "preseason_spread",
                "season_avg_implied_team_total",
                "season_avg_spread",
            ]
        ]
        .sort_values(["season", "week", "team"])
        .reset_index(drop=True)
    )


def attach_vegas_team_context_features(
    index: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge the four Vegas team-context features onto a player-team-week index.

    Args:
        index: frame with at least (season, week, team) columns. Typically
            the player-team-week index from
            `scripts.build_vegas_team_context_override._build_player_team_week_index`,
            carrying (gsis_id, season, week, team, opp, position).
        schedules: frame validated against `SchedulesSchema`.

    Returns:
        Copy of index with four nullable Float64 cols appended:
        preseason_implied_team_total, preseason_spread,
        season_avg_implied_team_total, season_avg_spread.
        Index rows without a matching (season, week, team) in schedules
        retain NaN in all four cols.
    """
    feats = compute_vegas_team_context_features(schedules)
    return index.merge(feats, on=["season", "week", "team"], how="left")


def build_vegas_team_context_overrides(
    schedules: pd.DataFrame,
    player_team_week_index: pd.DataFrame,
) -> pd.DataFrame:
    """Build the Vegas team-context override frame.

    Args:
        schedules: validated against `SchedulesSchema`.
        player_team_week_index: frame with columns (gsis_id, season, week,
            team, opp, position). Must have unique (gsis_id, season, week)
            keys.

    Returns:
        Frame with columns (gsis_id, season, week, position,
        preseason_implied_team_total, preseason_spread,
        season_avg_implied_team_total, season_avg_spread) — one row per
        index input row. Feeds `scripts.probe_feature_signal --override`.

    Raises:
        ValueError: index missing a required column, carrying a malformed
            gsis_id, or carrying duplicate (gsis_id, season, week) keys.
        AssertionError: row-count mismatch after the feature merge
            (internal-invariant violation; signals a regression introducing
            duplicate (season, week, team) keys in compute).
    """
    missing = [c for c in _REQUIRED_INDEX_COLS if c not in player_team_week_index.columns]
    if missing:
        raise ValueError(f"player_team_week_index missing required column(s): {missing}")

    bad_ids = [g for g in player_team_week_index["gsis_id"].dropna() if not _GSIS_RE.match(str(g))]
    if bad_ids:
        raise ValueError(
            f"invalid gsis_id format(s): {bad_ids[:3]} (and {max(0, len(bad_ids) - 3)} more)"
        )

    key_cols = ["gsis_id", "season", "week"]
    dups = player_team_week_index.duplicated(subset=key_cols)
    if dups.any():
        n = int(dups.sum())
        raise ValueError(f"player_team_week_index has {n} duplicate (gsis_id, season, week) keys")

    attached = attach_vegas_team_context_features(player_team_week_index, schedules)
    if len(attached) != len(player_team_week_index):
        raise AssertionError(
            f"row count mismatch: input had {len(player_team_week_index)} rows, "
            f"output has {len(attached)}; suggests a many-to-many merge regression"
        )

    return attached[["gsis_id", "season", "week", "position", *_FEATURE_COLS]].reset_index(
        drop=True
    )
