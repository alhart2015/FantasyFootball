"""Scoring engine -- pure stat -> points math, ruleset-parameterized."""

from __future__ import annotations

from projections.scoring.score import StatLine, score
from projections.scoring.score_distribution import SampledDistribution, score_distribution

__all__ = ["SampledDistribution", "StatLine", "score", "score_distribution"]
