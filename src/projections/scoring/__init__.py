"""Scoring engine -- pure stat -> points math, ruleset-parameterized."""

from __future__ import annotations

from projections.scoring.score import StatLine, score
from projections.scoring.score_distribution import (
    INTEGER_STATS,
    SampledDistribution,
    score_distribution,
)

__all__ = ["INTEGER_STATS", "SampledDistribution", "StatLine", "score", "score_distribution"]
