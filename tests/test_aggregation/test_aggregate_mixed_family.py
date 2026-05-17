"""aggregate_to_season — coverage for family=MIXED and family=QUANTILE rows.

Plan 5c production models (lightgbm-nb, ensemble-decomposed) emit per-row
family=MIXED because their params blob can hold multiple per-stat families
(NORMAL + NEGATIVE_BINOMIAL + QUANTILE etc.). The codec handles each per-stat
family generically, so aggregate_to_season composes weeks correctly regardless
of the row-level family tag. This module verifies that wiring end-to-end.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from projections.aggregation import aggregate_to_season
from projections.distributions import (
    Distribution,
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    QuantileDistribution,
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


def _build_mixed_row(
    *,
    gsis_id: str,
    season: int,
    week: int,
    family_tag: str,
    per_stat: dict[Stat, Distribution],
    position: str = "WR",
    model_id: str = "test:mixed:abcdef12:2018-2023",
) -> dict[str, Any]:
    """Build a ProjectionWeeklySchema-shaped row from a per-stat dict.

    Mirrors tests/test_aggregation/test_season.py:_build_weekly_row but lets the
    caller pick the per-stat distributions and the row-level family tag so we
    can exercise MIXED/QUANTILE branches without hardcoding stat combinations.
    """
    blob = pack_per_stat_params(per_stat)
    seed = derive_row_seed(gsis_id=gsis_id, season=season, week=week, ruleset_name=_RULESET.name)
    points = score_distribution(per_stat, _RULESET, n_samples=10_000, seed=seed)
    return {
        "gsis_id": gsis_id,
        "season": season,
        "week": week,
        "position": position,
        "team": "KC",
        "opponent": "BAL",
        "ruleset": _RULESET.name,
        "family": family_tag,
        "params": blob,
        "mean": points.mean(),
        "p10": points.quantile(0.1),
        "p50": points.quantile(0.5),
        "p90": points.quantile(0.9),
        "model_id": model_id,
        "generated_at": pd.Timestamp(datetime.now(UTC)).as_unit("us"),
    }


def _to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("gsis_id", "team", "opponent", "ruleset", "family", "model_id", "position"):
        df[col] = df[col].astype(_PYARROW_STR)
    return df


def test_mixed_family_row_is_accepted() -> None:
    """aggregate_to_season accepts family=MIXED rows; the params blob can hold a
    mix of per-stat families (NORMAL + GAMMA + NEGATIVE_BINOMIAL here) and the
    codec decodes each one via _unpack_single regardless of the row-level tag."""
    per_stat: dict[Stat, Distribution] = {
        Stat.RECEIVING_YARDS: ParametricNormal(mean=80.0, std=25.0),
        Stat.RECEPTIONS: ParametricGamma(shape=5.0, scale=1.2),
        Stat.RECEIVING_TDS: ParametricNegativeBinomial(mean=0.5, dispersion=0.7),
    }
    rows = [
        _build_mixed_row(
            gsis_id="00-0000001",
            season=2024,
            week=w,
            family_tag=DistributionFamily.MIXED.value,
            per_stat=per_stat,
        )
        for w in (1, 2, 3)
    ]
    weekly = _to_frame(rows)

    out = aggregate_to_season(weekly, ruleset=_RULESET, n_samples=200)

    assert len(out) == 1
    ProjectionSeasonSchema.validate(out)
    season_mean = float(out["season_mean"].iloc[0])
    p10 = float(out["season_p10"].iloc[0])
    p50 = float(out["season_p50"].iloc[0])
    p90 = float(out["season_p90"].iloc[0])
    assert season_mean > 0
    assert p10 <= p50 <= p90
    assert int(out["n_weeks"].iloc[0]) == 3


def test_quantile_family_row_is_accepted() -> None:
    """family=QUANTILE rows (Plan 5 lightgbm quantile-regression model) also
    flow through. The per-stat blob holds a QuantileDistribution which the
    codec round-trips."""
    # Synthesize a quantile sketch consistent with ~70 receiving yards median.
    per_stat: dict[Stat, Distribution] = {
        Stat.RECEIVING_YARDS: QuantileDistribution(
            quantiles=np.array([0.1, 0.5, 0.9], dtype=np.float64),
            values=np.array([35.0, 70.0, 115.0], dtype=np.float64),
        ),
        Stat.RECEPTIONS: QuantileDistribution(
            quantiles=np.array([0.1, 0.5, 0.9], dtype=np.float64),
            values=np.array([3.0, 5.0, 8.0], dtype=np.float64),
        ),
    }
    rows = [
        _build_mixed_row(
            gsis_id="00-0000002",
            season=2024,
            week=w,
            family_tag=DistributionFamily.QUANTILE.value,
            per_stat=per_stat,
        )
        for w in (1, 2)
    ]
    weekly = _to_frame(rows)

    out = aggregate_to_season(weekly, ruleset=_RULESET, n_samples=200)

    assert len(out) == 1
    ProjectionSeasonSchema.validate(out)
    season_mean = float(out["season_mean"].iloc[0])
    p10 = float(out["season_p10"].iloc[0])
    p90 = float(out["season_p90"].iloc[0])
    assert season_mean > 0
    assert p90 >= p10


def test_mixed_family_aggregate_is_deterministic() -> None:
    """Same MIXED-family input twice => identical summary quantiles
    (sample seeds derive deterministically from ids + ruleset)."""
    per_stat: dict[Stat, Distribution] = {
        Stat.RECEIVING_YARDS: ParametricNormal(mean=60.0, std=20.0),
        Stat.RECEIVING_TDS: ParametricNegativeBinomial(mean=0.4, dispersion=0.6),
    }
    rows = [
        _build_mixed_row(
            gsis_id="00-0000003",
            season=2024,
            week=w,
            family_tag=DistributionFamily.MIXED.value,
            per_stat=per_stat,
        )
        for w in (1, 2, 3, 4)
    ]
    weekly = _to_frame(rows)
    out_a = aggregate_to_season(weekly, ruleset=_RULESET, n_samples=300)
    out_b = aggregate_to_season(weekly, ruleset=_RULESET, n_samples=300)
    for col in ("season_mean", "season_p10", "season_p50", "season_p90"):
        assert (out_a[col].to_numpy() == out_b[col].to_numpy()).all()
