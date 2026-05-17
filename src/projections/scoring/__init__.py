"""Scoring engine -- pure stat -> points math, ruleset-parameterized."""

from __future__ import annotations

from projections.scoring.actuals import actual_season_total
from projections.scoring.score import StatLine, score
from projections.scoring.score_distribution import (
    INTEGER_STATS,
    SampledDistribution,
    derive_row_seed,
    score_distribution,
)

__all__ = [
    "INTEGER_STATS",
    "SampledDistribution",
    "StatLine",
    "actual_season_total",
    "derive_row_seed",
    "score",
    "score_distribution",
]
