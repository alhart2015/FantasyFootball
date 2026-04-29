"""Plan 8 — adoption gate.

Paired-bootstrap CI machinery for comparing two model classes on per-row
backtest output. Pure numpy/scipy/pandas — no IO. Consumed by
scripts/adoption_gate.py (the CLI orchestrator).

Spec: docs/superpowers/specs/2026-04-29-plan-8-gate-redesign-design.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from projections.schemas import Position

VerdictLabel = Literal["ADOPT", "MARGINAL", "DO_NOT_ADOPT"]


@dataclass(frozen=True, slots=True)
class BootstrapDelta:
    """Result of a paired bootstrap on a metric delta (candidate - incumbent).

    Sign convention is metric-specific: for RMSE, negative ``point`` means
    the candidate wins (lower error). For Spearman, positive ``point`` means
    the candidate wins (higher rank correlation).
    """

    point: float
    lo_95: float
    hi_95: float
    n_paired_rows: int
    n_bootstrap: int


@dataclass(frozen=True, slots=True)
class PositionVerdict:
    """Per-position adoption verdict bundling RMSE, Spearman, and per-year breakdown."""

    position: Position
    incumbent_class: str
    candidate_class: str
    rmse_delta: BootstrapDelta
    spearman_delta: BootstrapDelta
    verdict: VerdictLabel
    reason: str
    per_year_breakdown: pd.DataFrame
