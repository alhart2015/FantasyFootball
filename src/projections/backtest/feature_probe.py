# src/projections/backtest/feature_probe.py
"""Feature signal probe — pre-spec screening tool.

Two-phase probe of a candidate feature column or column-set against the
production baseline features. Phase 1 fits per-stat Ridge regressors and
emits a paired-bootstrap CI on the per-stat Δ-CV-RMSE; Phase 2 (gated on
any Phase-1 SIGNAL cell) runs the configured production model class
(BaselineModel or LightGBMNbModel) on both feature sets walk-forward and
emits an adoption-gate-shaped composite verdict.

Pure numpy/scipy/pandas/sklearn. Reuses
``src/projections/backtest/adoption_gate.py``'s paired-bootstrap helpers
unchanged. Consumed by ``scripts/probe_feature_signal.py``.

Spec: docs/superpowers/specs/2026-04-30-feature-signal-probe-design.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from projections.backtest.adoption_gate import BootstrapDelta, PositionVerdict
from projections.schemas import Position, Stat

PerStatLabel = Literal["SIGNAL", "NULL", "REGRESSION"]


@dataclass(frozen=True, slots=True)
class PerStatVerdict:
    """Per-stat per-(year-or-pooled) screening verdict.

    ``verdict == "SIGNAL"`` iff ``rmse_delta.hi_95 < 0`` (candidate strictly
    improves CV-RMSE on this cell). ``REGRESSION`` iff ``lo_95 > 0``. ``NULL``
    otherwise — the bootstrap CI brackets zero, so the per-stat effect is
    indistinguishable from sampling noise on this dataset.
    """

    position: Position
    stat: Stat
    year_or_pooled: int | Literal["pooled"]
    n_paired: int
    rmse_delta: BootstrapDelta
    r_squared_delta: float  # in-sample, diagnostic only
    verdict: PerStatLabel


@dataclass(frozen=True, slots=True)
class ProbeReport:
    """Bundled probe report for rendering. ``phase2`` is None iff Phase 1
    returned no SIGNAL cell or the user passed ``--no-composite``."""

    candidate_name: str
    model_class: str
    baseline_features_path: str
    override_paths: tuple[str, ...]
    drop_columns: tuple[str, ...]
    phase1: list[PerStatVerdict]
    phase2: list[PositionVerdict] | None


def phase1_should_fire_phase2(verdicts: list[PerStatVerdict]) -> bool:
    """Return True iff any per-cell or pooled verdict in the Phase-1 result
    set is ``SIGNAL``. NULL and REGRESSION cells do not fire Phase 2 — the
    probe runs Phase 2 only when there's plausibly a real effect to evaluate
    at the composite level."""
    return any(v.verdict == "SIGNAL" for v in verdicts)
