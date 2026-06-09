"""Scoring engine -- pure stat -> points math, ruleset-parameterized."""

from __future__ import annotations

from projections.scoring.actuals import actual_season_total
from projections.scoring.score import StatLine, expected_points, score
from projections.scoring.score_distribution import (
    INTEGER_STATS,
    SampledDistribution,
    derive_row_seed,
    score_distribution,
)
from projections.scoring.score_distribution import (
    _scoring_coefficients as scoring_coefficients,
)

__all__ = [
    "INTEGER_STATS",
    "SampledDistribution",
    "StatLine",
    "actual_season_total",
    "derive_row_seed",
    "expected_points",
    "score",
    "score_distribution",
    "scoring_coefficients",
]
