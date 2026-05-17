"""Scoring engine -- pure stat -> points math, ruleset-parameterized."""

from __future__ import annotations

from projections.scoring.score import StatLine, score
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
    "derive_row_seed",
    "score",
    "score_distribution",
    "scoring_coefficients",
]
