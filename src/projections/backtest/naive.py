"""Naive baseline for the walk-forward backtest harness.

Per-player trailing-4-game stat mean, with cold-start fallback to per-
position mean across the train window. Used for informational comparison;
never gated.

Plan 3c spec section 3 (naive baseline definition).
"""

from __future__ import annotations

import pandas as pd

from projections.schemas import Position, Stat

_TRAILING_N: int = 4


def _per_position_means(
    train_actuals: pd.DataFrame,
    *,
    position: Position,
    target_stats: tuple[Stat, ...],
) -> dict[str, float]:
    """Per-stat mean across the train window for the given position.
    Cold-start fallback when a player has fewer than _TRAILING_N prior games.
    """
    pos_rows = train_actuals[train_actuals["position"] == position.value]
    return {stat.value: float(pos_rows[stat.value].mean()) for stat in target_stats}


def compute_naive_predictions(
    *,
    train_actuals: pd.DataFrame,
    holdout_actuals: pd.DataFrame,
    position: Position,
    target_stats: tuple[Stat, ...],
    held_out_year: int,
) -> pd.DataFrame:
    """For each (gsis_id, week) in holdout_actuals, produce a per-stat
    naive prediction equal to the player's trailing-4-game mean of that
    stat across all games strictly prior to (held_out_year, week).

    Earlier weeks of held_out_year are allowed in the trailing window
    (no leakage — they're already observed at the simulated time of
    prediction). Cold start (< 4 prior games) falls back to per-position
    mean across train_actuals (held_out_year excluded).

    Returns:
        DataFrame with columns ``gsis_id``, ``season``, ``week``, plus
        one float column per target stat. Same row count as holdout_actuals
        (filtered to the position).
    """
    holdout_pos = holdout_actuals[holdout_actuals["position"] == position.value].copy()
    cold_means = _per_position_means(train_actuals, position=position, target_stats=target_stats)

    # Combine train + holdout-prior for the trailing window. We re-filter
    # per (gsis_id, week) below.
    combined = pd.concat(
        [
            train_actuals[train_actuals["position"] == position.value],
            holdout_pos,
        ],
        ignore_index=True,
    )
    combined = combined.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for _idx, hold_row in holdout_pos.iterrows():
        gsis_id = hold_row["gsis_id"]
        season = int(hold_row["season"])
        week = int(hold_row["week"])

        # Strictly-prior mask: same player, and (season < held_out_year)
        # OR (season == held_out_year AND week < target week).
        prior = combined[
            (combined["gsis_id"] == gsis_id)
            & (
                (combined["season"] < held_out_year)
                | ((combined["season"] == held_out_year) & (combined["week"] < week))
            )
        ]
        prior = prior.tail(_TRAILING_N)

        out_row: dict[str, object] = {
            "gsis_id": gsis_id,
            "season": season,
            "week": week,
        }
        if len(prior) >= _TRAILING_N:
            for stat in target_stats:
                out_row[stat.value] = float(prior[stat.value].mean())
        else:
            for stat in target_stats:
                out_row[stat.value] = cold_means[stat.value]
        rows.append(out_row)

    return pd.DataFrame(rows)
