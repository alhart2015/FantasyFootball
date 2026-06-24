"""Walk-forward home-grown weekly projections for the DFS edge study.

Reuses the model-backtest fit/predict path (backtest.harness) but collects
per-stat predicted means rather than metrics. Scores DK-base points from those
means via the scoring layer (exact: DK base scoring is linear in stats, and the
blend in dfs.blend consumes the same per-stat means).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from projections.backtest.harness import _per_stat_means_from_predictions
from projections.features.cache import read_features
from projections.models import POSITION_DISPATCH
from projections.schemas import Position, Ruleset, Stat
from projections.scoring import expected_points
from projections.store import read_partition

_STAT_COLS = [s.value for s in Stat]


class _HasTargetStats(Protocol):
    """Structural view of the per-position production models, which all expose
    ``target_stats`` (not on the base ``Model`` protocol). Casting to this
    avoids importing/maintaining the concrete 4-class union the harness uses."""

    target_stats: tuple[Stat, ...]


def _emit_one_cell(
    position: Position,
    year: int,
    *,
    train_start: int,
    model_class: str | None,
    features_root: Path,
    raw_root: Path,
    ruleset: Ruleset,
) -> pd.DataFrame:
    """Fit on seasons < year, predict all weeks of `year`, return per-stat means.

    ``model_class=None`` -> the position's production model
    (``default_model_class``)."""
    dispatch = POSITION_DISPATCH[position]
    resolved_class = model_class or dispatch.default_model_class

    train_seasons = list(range(train_start, year))
    train_features = pd.concat(
        [read_features(position, s, features_root=features_root) for s in train_seasons],
        ignore_index=True,
    )
    train_actuals = pd.concat(
        [read_partition(raw_root, "weekly_stats", season=s) for s in train_seasons],
        ignore_index=True,
    )
    predict_features = read_features(position, year, features_root=features_root)

    model = dispatch.factories[resolved_class]()
    model.fit(train_features, train_actuals)
    predictions = model.predict_distribution(predict_features, ruleset=ruleset)
    target_stats = tuple(cast(_HasTargetStats, model).target_stats)
    means = _per_stat_means_from_predictions(predictions, target_stats=target_stats)
    means["position"] = position.value
    return means


def emit_weekly_projections(
    *,
    seasons: list[int],
    positions: list[Position],
    train_start: int = 2018,
    model_class: str | None = None,
    features_root: Path | str,
    raw_root: Path | str,
    ruleset: Ruleset,
) -> pd.DataFrame:
    """Per-(gsis_id, season, week, position) per-stat means + DK-base `our_pts`.

    ``model_class=None`` uses each position's production model."""
    frames: list[pd.DataFrame] = []
    for position in positions:
        for year in seasons:
            frames.append(
                _emit_one_cell(
                    position,
                    year,
                    train_start=train_start,
                    model_class=model_class,
                    features_root=Path(features_root),
                    raw_root=Path(raw_root),
                    ruleset=ruleset,
                )
            )
    out = pd.concat(frames, ignore_index=True)
    stat_cols = [c for c in out.columns if c in _STAT_COLS]
    out["our_pts"] = out[stat_cols].apply(
        lambda r: expected_points({k: float(r[k]) for k in stat_cols if pd.notna(r[k])}, ruleset),
        axis=1,
    )
    return out
