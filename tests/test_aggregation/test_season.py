"""aggregate_to_season — pure-function tests over ProjectionWeeklySchema rows."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from projections.aggregation import aggregate_to_season
from projections.distributions import (
    Distribution,
    ParametricGamma,
    ParametricNormal,
    pack_per_stat_params,
)
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    ProjectionSeasonSchema,
    Ruleset,
    Stat,
)
from projections.scoring import derive_row_seed, score_distribution

_RULESET = Ruleset.espn_ppr()


def _build_weekly_row(
    *,
    gsis_id: str = "00-0033873",
    season: int = 2024,
    week: int = 1,
    position: str = "WR",
    team: str = "KC",
    opponent: str = "BAL",
    family: str = DistributionFamily.SAMPLED_SUMMARY.value,
    ruleset_name: str | None = None,
    rec_yards_mean: float = 50.0,
    rec_yards_std: float = 18.0,
    rec_shape: float = 4.0,
    rec_scale: float = 0.7,
    model_id: str = "baseline:wr:abcdef12:2018-2023",
) -> dict[str, Any]:
    rs_name = ruleset_name if ruleset_name is not None else _RULESET.name
    per_stat_dists: dict[Stat, Distribution] = {
        Stat.RECEIVING_YARDS: ParametricNormal(mean=rec_yards_mean, std=rec_yards_std),
        Stat.RECEPTIONS: ParametricGamma(shape=rec_shape, scale=rec_scale),
    }
    blob = pack_per_stat_params(per_stat_dists)
    seed = derive_row_seed(gsis_id=gsis_id, season=season, week=week, ruleset_name=rs_name)
    points = score_distribution(per_stat_dists, _RULESET, n_samples=10_000, seed=seed)
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": position,
        "team": team,
        "opponent": opponent,
        "ruleset": rs_name,
        "family": family,
        "params": blob,
        "mean": points.mean(),
        "p10": points.quantile(0.1),
        "p50": points.quantile(0.5),
        "p90": points.quantile(0.9),
        "model_id": model_id,
        "generated_at": pd.Timestamp(datetime.now(UTC)).as_unit("us"),
    }


def _to_weekly_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "team", "opponent", "ruleset", "family", "model_id", "position"):
        df[col] = df[col].astype(_PYARROW_STR)
    return df


def test_empty_input_returns_empty_validated_frame() -> None:
    empty = pd.DataFrame(
        columns=[
            "gsis_id",
            "season",
            "week",
            "position",
            "team",
            "opponent",
            "ruleset",
            "family",
            "params",
            "mean",
            "p10",
            "p50",
            "p90",
            "model_id",
            "generated_at",
        ]
    )
    out = aggregate_to_season(empty, ruleset=_RULESET)
    assert out.empty
    ProjectionSeasonSchema.validate(out)
