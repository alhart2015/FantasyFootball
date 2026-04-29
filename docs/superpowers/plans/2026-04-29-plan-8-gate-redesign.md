# Plan 8 — Adoption Gate Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the §1.3 adoption gate with a paired-bootstrap CI–based, per-position gate; ship per-position routing in `POSITION_DISPATCH`; re-evaluate the four existing peer models (Model C / C-tuned / C-NB / D) under the new gate and update production defaults for ADOPT verdicts.

**Architecture:** A pure-stats module (`src/projections/backtest/adoption_gate.py`) provides paired-bootstrap CI functions and a verdict rule; a thin CLI (`scripts/adoption_gate.py`) reads the existing per-row backtest parquet, pairs rows on `(gsis_id, season, week)`, and emits per-position verdicts. `_PositionDispatch` gains a `default_model_class` field with a `production_model_for(position)` helper so production defaults can vary per position. No model code, feature pipeline, or harness changes.

**Tech Stack:** Python 3.11, numpy, scipy (stats), pandas, pyarrow (parquet), pandera (existing schemas), argparse (CLI), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-29-plan-8-gate-redesign-design.md`

**Branch:** `feat/plan-8-gate-redesign` (cut from `main` at `995d43f`).

---

## Phase 1 — Pure-stats module: paired-bootstrap CIs and verdict rule

**Phase scope:** 1 production file, 1 test file, ~5 commits. No external IO. Pure numpy/scipy. Reusable for any future plan that wants paired CIs.

### Task 1.1 — Module scaffold + dataclasses

**Files:**
- Create: `src/projections/backtest/adoption_gate.py`
- Create: `tests/test_backtest/test_adoption_gate.py`

- [ ] **Step 1: Write the failing test for the dataclasses**

```python
# tests/test_backtest/test_adoption_gate.py
"""Plan 8 — adoption gate tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.backtest.adoption_gate import (
    BootstrapDelta,
    PositionVerdict,
)
from projections.schemas import Position


def test_bootstrap_delta_is_frozen_dataclass() -> None:
    bd = BootstrapDelta(
        point=-0.5, lo_95=-1.2, hi_95=0.1, n_paired_rows=1000, n_bootstrap=500
    )
    assert bd.point == -0.5
    assert bd.n_bootstrap == 500
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        bd.point = 0.0  # type: ignore[misc]


def test_position_verdict_bundles_metrics() -> None:
    rmse = BootstrapDelta(point=-0.3, lo_95=-0.5, hi_95=-0.1, n_paired_rows=500, n_bootstrap=1000)
    spear = BootstrapDelta(point=0.01, lo_95=-0.005, hi_95=0.025, n_paired_rows=500, n_bootstrap=1000)
    breakdown = pd.DataFrame(
        {
            "year": [2021, 2022],
            "rmse_delta_point": [-0.4, -0.2],
            "rmse_delta_lo": [-0.7, -0.4],
            "rmse_delta_hi": [-0.1, 0.0],
            "spearman_delta_point": [0.02, 0.0],
            "spearman_delta_lo": [-0.01, -0.02],
            "spearman_delta_hi": [0.04, 0.02],
        }
    )
    pv = PositionVerdict(
        position=Position.QB,
        incumbent_class="baseline",
        candidate_class="ensemble",
        rmse_delta=rmse,
        spearman_delta=spear,
        verdict="ADOPT",
        reason="RMSE win, Spearman within floor",
        per_year_breakdown=breakdown,
    )
    assert pv.position is Position.QB
    assert pv.verdict == "ADOPT"
    assert len(pv.per_year_breakdown) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest/test_adoption_gate.py -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError` for `projections.backtest.adoption_gate`.

- [ ] **Step 3: Create the module with dataclasses**

```python
# src/projections/backtest/adoption_gate.py
"""Plan 8 — adoption gate.

Paired-bootstrap CI machinery for comparing two model classes on per-row
backtest output. Pure numpy/scipy/pandas — no IO. Consumed by
scripts/adoption_gate.py (the CLI orchestrator).

Spec: docs/superpowers/specs/2026-04-29-plan-8-gate-redesign-design.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from projections.schemas import Position

VerdictLabel = Literal["ADOPT", "MARGINAL", "DO_NOT_ADOPT"]


@dataclass(frozen=True, slots=True)
class BootstrapDelta:
    """Result of a paired bootstrap on a metric delta (candidate - incumbent).

    Sign convention is metric-specific: for RMSE, negative `point` means
    the candidate wins (lower error). For Spearman, positive `point` means
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest/test_adoption_gate.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/projections/backtest/adoption_gate.py tests/test_backtest/test_adoption_gate.py
git commit -m "feat(adoption-gate): scaffold module + dataclasses — Plan 8 Phase 1"
```

### Task 1.2 — `paired_bootstrap_rmse_delta`

**Files:**
- Modify: `src/projections/backtest/adoption_gate.py`
- Modify: `tests/test_backtest/test_adoption_gate.py`

- [ ] **Step 1: Write failing tests for the RMSE bootstrap**

Append to `tests/test_backtest/test_adoption_gate.py`:

```python
from projections.backtest.adoption_gate import paired_bootstrap_rmse_delta


def test_rmse_delta_identical_residuals_brackets_zero() -> None:
    rng = np.random.default_rng(0)
    residuals = rng.normal(size=2000)
    bd = paired_bootstrap_rmse_delta(residuals, residuals, n_bootstrap=500, seed=42)
    assert bd.point == 0.0
    assert bd.lo_95 <= 0.0 <= bd.hi_95
    assert bd.n_paired_rows == 2000
    assert bd.n_bootstrap == 500


def test_rmse_delta_candidate_strictly_better_has_negative_ci() -> None:
    rng = np.random.default_rng(0)
    incumbent_residuals = rng.normal(scale=2.0, size=3000)
    candidate_residuals = incumbent_residuals / 2.0  # half the variance
    bd = paired_bootstrap_rmse_delta(
        incumbent_residuals, candidate_residuals, n_bootstrap=500, seed=42
    )
    assert bd.point < 0.0
    assert bd.hi_95 < 0.0  # 95% CI entirely below zero


def test_rmse_delta_candidate_strictly_worse_has_positive_ci() -> None:
    rng = np.random.default_rng(0)
    incumbent_residuals = rng.normal(scale=2.0, size=3000)
    candidate_residuals = incumbent_residuals * 2.0
    bd = paired_bootstrap_rmse_delta(
        incumbent_residuals, candidate_residuals, n_bootstrap=500, seed=42
    )
    assert bd.point > 0.0
    assert bd.lo_95 > 0.0


def test_rmse_delta_deterministic_under_same_seed() -> None:
    rng = np.random.default_rng(0)
    inc = rng.normal(size=500)
    cand = inc + rng.normal(scale=0.1, size=500)
    bd1 = paired_bootstrap_rmse_delta(inc, cand, n_bootstrap=200, seed=99)
    bd2 = paired_bootstrap_rmse_delta(inc, cand, n_bootstrap=200, seed=99)
    assert bd1 == bd2


def test_rmse_delta_raises_on_too_few_rows() -> None:
    inc = np.zeros(50)
    cand = np.zeros(50)
    with pytest.raises(ValueError, match="at least 100 paired rows"):
        paired_bootstrap_rmse_delta(inc, cand)


def test_rmse_delta_raises_on_length_mismatch() -> None:
    inc = np.zeros(200)
    cand = np.zeros(199)
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_rmse_delta(inc, cand)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest/test_adoption_gate.py -v -k "rmse"`
Expected: 6 failures (function not yet defined).

- [ ] **Step 3: Implement `paired_bootstrap_rmse_delta`**

Append to `src/projections/backtest/adoption_gate.py`:

```python
_MIN_PAIRED_ROWS = 100


def paired_bootstrap_rmse_delta(
    residuals_incumbent: np.ndarray,
    residuals_candidate: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BootstrapDelta:
    """Paired bootstrap CI on RMSE(candidate) - RMSE(incumbent).

    Both residual arrays must be aligned on the same rows (same player-week
    in the same order). The same bootstrap-sampled indices are applied to
    both arrays each draw — that's the "paired" structure.

    Args:
        residuals_incumbent: shape (n,), real-valued, ``actual - predicted``.
        residuals_candidate: shape (n,), real-valued, same row order.
        n_bootstrap: number of bootstrap resamples. Default 1000.
        seed: RNG seed for reproducibility. Default 42.

    Returns:
        BootstrapDelta with `point` = observed delta on full sample
        (no resampling), `lo_95` and `hi_95` the central 95% CI bounds
        across the bootstrap distribution.

    Raises:
        ValueError: lengths mismatch, or fewer than 100 paired rows.
    """
    inc = np.asarray(residuals_incumbent, dtype=np.float64)
    cand = np.asarray(residuals_candidate, dtype=np.float64)
    if inc.shape != cand.shape:
        raise ValueError(
            f"residuals must have the same length, got incumbent={inc.shape} "
            f"vs candidate={cand.shape}"
        )
    n = inc.shape[0]
    if n < _MIN_PAIRED_ROWS:
        raise ValueError(
            f"need at least {_MIN_PAIRED_ROWS} paired rows for a meaningful "
            f"bootstrap, got {n}"
        )

    point = float(np.sqrt(np.mean(cand**2)) - np.sqrt(np.mean(inc**2)))

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rmse_inc = np.sqrt(np.mean(inc[idx] ** 2))
        rmse_cand = np.sqrt(np.mean(cand[idx] ** 2))
        deltas[b] = rmse_cand - rmse_inc

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return BootstrapDelta(
        point=point,
        lo_95=float(lo),
        hi_95=float(hi),
        n_paired_rows=n,
        n_bootstrap=n_bootstrap,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest/test_adoption_gate.py -v -k "rmse"`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/backtest/adoption_gate.py tests/test_backtest/test_adoption_gate.py
git commit -m "feat(adoption-gate): paired_bootstrap_rmse_delta — Plan 8 Phase 1"
```

### Task 1.3 — `paired_bootstrap_spearman_delta`

**Files:**
- Modify: `src/projections/backtest/adoption_gate.py`
- Modify: `tests/test_backtest/test_adoption_gate.py`

- [ ] **Step 1: Write failing tests for the Spearman bootstrap**

Append to `tests/test_backtest/test_adoption_gate.py`:

```python
from projections.backtest.adoption_gate import paired_bootstrap_spearman_delta


def test_spearman_delta_identical_predictions_brackets_zero() -> None:
    rng = np.random.default_rng(0)
    actual = rng.normal(size=2000)
    pred = actual + rng.normal(scale=0.5, size=2000)
    grouping = np.repeat([2021, 2022, 2023, 2024], 500)
    bd = paired_bootstrap_spearman_delta(
        pred, pred, actual, grouping, n_bootstrap=300, seed=42
    )
    assert bd.point == 0.0
    assert bd.lo_95 <= 0.0 <= bd.hi_95


def test_spearman_delta_candidate_perfect_vs_incumbent_random() -> None:
    rng = np.random.default_rng(0)
    actual = np.arange(2000, dtype=np.float64)
    candidate = actual.copy()           # perfect rank
    incumbent = rng.normal(size=2000)   # random rank
    grouping = np.repeat([2021, 2022, 2023, 2024], 500)
    bd = paired_bootstrap_spearman_delta(
        incumbent, candidate, actual, grouping, n_bootstrap=300, seed=42
    )
    assert bd.point > 0.5
    assert bd.lo_95 > 0.0


def test_spearman_delta_per_year_averaging_cancels_opposite_year_wins() -> None:
    rng = np.random.default_rng(0)
    n_per_year = 600
    actual = rng.normal(size=n_per_year * 2)
    incumbent = actual + rng.normal(scale=0.5, size=n_per_year * 2)
    # Year 1: candidate wins by ~0.2 Spearman; Year 2: incumbent wins by ~0.2.
    candidate = incumbent.copy()
    # Year 1: candidate is closer to actual.
    candidate[:n_per_year] = actual[:n_per_year] + rng.normal(scale=0.2, size=n_per_year)
    # Year 2: candidate is much noisier (much worse Spearman).
    candidate[n_per_year:] = rng.normal(size=n_per_year)
    grouping = np.repeat([2021, 2022], n_per_year)
    bd = paired_bootstrap_spearman_delta(
        incumbent, candidate, actual, grouping, n_bootstrap=300, seed=42
    )
    # Year 1 win + Year 2 loss → averaged delta lives near zero.
    assert -0.6 < bd.point < 0.6


def test_spearman_delta_constant_candidate_propagates_nan() -> None:
    rng = np.random.default_rng(0)
    actual = rng.normal(size=2000)
    incumbent = actual + rng.normal(scale=0.5, size=2000)
    candidate = np.full(2000, 7.0)
    grouping = np.repeat([2021, 2022, 2023, 2024], 500)
    bd = paired_bootstrap_spearman_delta(
        incumbent, candidate, actual, grouping, n_bootstrap=200, seed=42
    )
    assert np.isnan(bd.point) or np.isnan(bd.lo_95) or np.isnan(bd.hi_95)


def test_spearman_delta_deterministic_under_same_seed() -> None:
    rng = np.random.default_rng(0)
    actual = rng.normal(size=500)
    inc = actual + rng.normal(scale=0.5, size=500)
    cand = actual + rng.normal(scale=0.4, size=500)
    grouping = np.repeat([2021, 2022], 250)
    bd1 = paired_bootstrap_spearman_delta(
        inc, cand, actual, grouping, n_bootstrap=200, seed=99
    )
    bd2 = paired_bootstrap_spearman_delta(
        inc, cand, actual, grouping, n_bootstrap=200, seed=99
    )
    assert bd1 == bd2 or (
        np.isnan(bd1.point) and np.isnan(bd2.point)
    )  # exact equality unless degenerate
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest/test_adoption_gate.py -v -k "spearman"`
Expected: 5 failures.

- [ ] **Step 3: Implement `paired_bootstrap_spearman_delta`**

Append to `src/projections/backtest/adoption_gate.py`:

```python
def _per_group_mean_spearman(
    predicted: np.ndarray,
    actual: np.ndarray,
    grouping: np.ndarray,
) -> float:
    """Spearman correlation per group, averaged unweighted across groups.

    Returns NaN if any group's correlation is undefined (constant input,
    or empty). The verdict_for_position rule downgrades NaN to DO_NOT_ADOPT.
    """
    groups = np.unique(grouping)
    if groups.size == 0:
        return float("nan")
    rhos = np.empty(groups.size, dtype=np.float64)
    for i, g in enumerate(groups):
        mask = grouping == g
        if mask.sum() < 2:
            return float("nan")
        rho = spearmanr(predicted[mask], actual[mask]).statistic
        if np.isnan(rho):
            return float("nan")
        rhos[i] = rho
    return float(rhos.mean())


def paired_bootstrap_spearman_delta(
    predicted_incumbent: np.ndarray,
    predicted_candidate: np.ndarray,
    actual: np.ndarray,
    grouping: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BootstrapDelta:
    """Paired bootstrap CI on (per-group-mean Spearman) delta candidate - incumbent.

    Per-year (group) Spearman is computed within each group, then averaged
    unweighted across groups. Pooling across years would mix populations
    because the player set rotates between held-out years.

    Args:
        predicted_incumbent: shape (n,) per-row predicted composite from incumbent.
        predicted_candidate: shape (n,) per-row predicted composite from candidate.
        actual: shape (n,) per-row actual composite.
        grouping: shape (n,) integer/string per-row group key (held-out year).
        n_bootstrap: number of bootstrap resamples. Default 1000.
        seed: RNG seed. Default 42.

    Returns:
        BootstrapDelta. NaN values propagate when either model produces a
        constant prediction within a group.

    Raises:
        ValueError: arrays have inconsistent lengths or fewer than 100 rows.
    """
    inc = np.asarray(predicted_incumbent, dtype=np.float64)
    cand = np.asarray(predicted_candidate, dtype=np.float64)
    act = np.asarray(actual, dtype=np.float64)
    grp = np.asarray(grouping)
    n = inc.shape[0]
    if not (cand.shape[0] == act.shape[0] == grp.shape[0] == n):
        raise ValueError(
            "predicted_incumbent, predicted_candidate, actual, grouping must "
            f"have the same length; got {inc.shape[0]}, {cand.shape[0]}, "
            f"{act.shape[0]}, {grp.shape[0]}"
        )
    if n < _MIN_PAIRED_ROWS:
        raise ValueError(
            f"need at least {_MIN_PAIRED_ROWS} paired rows, got {n}"
        )

    point = _per_group_mean_spearman(cand, act, grp) - _per_group_mean_spearman(inc, act, grp)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s_inc = _per_group_mean_spearman(inc[idx], act[idx], grp[idx])
        s_cand = _per_group_mean_spearman(cand[idx], act[idx], grp[idx])
        deltas[b] = s_cand - s_inc

    if np.isnan(point) or np.isnan(deltas).any():
        return BootstrapDelta(
            point=float("nan"),
            lo_95=float("nan"),
            hi_95=float("nan"),
            n_paired_rows=n,
            n_bootstrap=n_bootstrap,
        )

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return BootstrapDelta(
        point=point,
        lo_95=float(lo),
        hi_95=float(hi),
        n_paired_rows=n,
        n_bootstrap=n_bootstrap,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest/test_adoption_gate.py -v -k "spearman"`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/backtest/adoption_gate.py tests/test_backtest/test_adoption_gate.py
git commit -m "feat(adoption-gate): paired_bootstrap_spearman_delta — Plan 8 Phase 1"
```

### Task 1.4 — `verdict_for_position` rule

**Files:**
- Modify: `src/projections/backtest/adoption_gate.py`
- Modify: `tests/test_backtest/test_adoption_gate.py`

- [ ] **Step 1: Write failing tests for the verdict truth table**

Append to `tests/test_backtest/test_adoption_gate.py`:

```python
from projections.backtest.adoption_gate import verdict_for_position


def _bd(point: float, lo: float, hi: float) -> BootstrapDelta:
    return BootstrapDelta(
        point=point, lo_95=lo, hi_95=hi, n_paired_rows=1000, n_bootstrap=1000
    )


def test_verdict_adopt_when_rmse_wins_and_spearman_within_floor() -> None:
    rmse = _bd(point=-0.5, lo=-0.8, hi=-0.1)   # PASS_RMSE: hi_95 < 0
    spear = _bd(point=0.0, lo=-0.01, hi=0.01)  # PASS_SPEARMAN: lo_95 > -0.02
    label, reason = verdict_for_position(rmse, spear)
    assert label == "ADOPT"
    assert "RMSE" in reason or "rmse" in reason


def test_verdict_marginal_when_rmse_wins_but_spearman_regresses() -> None:
    rmse = _bd(point=-0.5, lo=-0.8, hi=-0.1)
    spear = _bd(point=-0.05, lo=-0.08, hi=-0.03)  # FAIL_SPEARMAN: lo_95 < -0.02
    label, reason = verdict_for_position(rmse, spear)
    assert label == "MARGINAL"
    assert "Spearman" in reason


def test_verdict_do_not_adopt_when_rmse_inconclusive() -> None:
    rmse = _bd(point=-0.1, lo=-0.4, hi=0.2)  # FAIL_RMSE: CI brackets zero
    spear = _bd(point=0.01, lo=-0.005, hi=0.025)
    label, reason = verdict_for_position(rmse, spear)
    assert label == "DO_NOT_ADOPT"


def test_verdict_do_not_adopt_when_both_fail() -> None:
    rmse = _bd(point=0.5, lo=0.2, hi=0.8)
    spear = _bd(point=-0.05, lo=-0.08, hi=-0.03)
    label, reason = verdict_for_position(rmse, spear)
    assert label == "DO_NOT_ADOPT"


def test_verdict_respects_custom_spearman_floor() -> None:
    rmse = _bd(point=-0.5, lo=-0.8, hi=-0.1)
    spear = _bd(point=-0.04, lo=-0.06, hi=-0.02)  # lo_95 = -0.06
    label_strict, _ = verdict_for_position(rmse, spear, spearman_floor=-0.02)
    assert label_strict == "MARGINAL"
    label_loose, _ = verdict_for_position(rmse, spear, spearman_floor=-0.10)
    assert label_loose == "ADOPT"


def test_verdict_degenerate_when_nan_inputs() -> None:
    rmse = _bd(point=float("nan"), lo=float("nan"), hi=float("nan"))
    spear = _bd(point=float("nan"), lo=float("nan"), hi=float("nan"))
    label, reason = verdict_for_position(rmse, spear)
    assert label == "DO_NOT_ADOPT"
    assert "degenerate" in reason.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest/test_adoption_gate.py -v -k "verdict"`
Expected: 6 failures (function not yet defined).

- [ ] **Step 3: Implement `verdict_for_position`**

Append to `src/projections/backtest/adoption_gate.py`:

```python
def verdict_for_position(
    rmse: BootstrapDelta,
    spearman: BootstrapDelta,
    *,
    spearman_floor: float = -0.02,
) -> tuple[VerdictLabel, str]:
    """Apply the §1.3-replacement rule.

    Rule (per spec §1.3):
        PASS_RMSE     := rmse.hi_95     <  0.0
        PASS_SPEARMAN := spearman.lo_95 > spearman_floor

        if  PASS_RMSE and  PASS_SPEARMAN: ADOPT
        if  PASS_RMSE and !PASS_SPEARMAN: MARGINAL
        if !PASS_RMSE and  PASS_SPEARMAN: DO_NOT_ADOPT
        if !PASS_RMSE and !PASS_SPEARMAN: DO_NOT_ADOPT

    Returns:
        (verdict_label, one-line human-readable reason).
    """
    if (
        np.isnan(rmse.point) or np.isnan(rmse.hi_95)
        or np.isnan(spearman.point) or np.isnan(spearman.lo_95)
    ):
        return ("DO_NOT_ADOPT", "degenerate prediction (NaN bootstrap statistics)")

    pass_rmse = rmse.hi_95 < 0.0
    pass_spearman = spearman.lo_95 > spearman_floor

    if pass_rmse and pass_spearman:
        return (
            "ADOPT",
            f"RMSE delta {rmse.point:+.3f} (95% CI [{rmse.lo_95:+.3f}, {rmse.hi_95:+.3f}]); "
            f"Spearman lo_95 {spearman.lo_95:+.4f} > floor {spearman_floor:+.3f}",
        )
    if pass_rmse and not pass_spearman:
        return (
            "MARGINAL",
            f"RMSE wins ({rmse.point:+.3f}) but Spearman lo_95 {spearman.lo_95:+.4f} "
            f"breaks floor {spearman_floor:+.3f}; investigate before adopting",
        )
    if not pass_rmse and pass_spearman:
        return (
            "DO_NOT_ADOPT",
            f"RMSE inconclusive: 95% CI [{rmse.lo_95:+.3f}, {rmse.hi_95:+.3f}] brackets / "
            f"exceeds zero",
        )
    return (
        "DO_NOT_ADOPT",
        f"RMSE worse ({rmse.point:+.3f}) and Spearman regresses "
        f"(lo_95 {spearman.lo_95:+.4f} < floor {spearman_floor:+.3f})",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest/test_adoption_gate.py -v -k "verdict"`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/backtest/adoption_gate.py tests/test_backtest/test_adoption_gate.py
git commit -m "feat(adoption-gate): verdict_for_position rule — Plan 8 Phase 1"
```

### Task 1.5 — Phase 1 verification

- [ ] **Step 1: Run full module test file**

Run: `pytest tests/test_backtest/test_adoption_gate.py -v`
Expected: all (~17) tests PASS.

- [ ] **Step 2: Run mypy + ruff on the new files**

```bash
mypy src/projections/backtest/adoption_gate.py tests/test_backtest/test_adoption_gate.py
ruff check src/projections/backtest/adoption_gate.py tests/test_backtest/test_adoption_gate.py
ruff format --check src/projections/backtest/adoption_gate.py tests/test_backtest/test_adoption_gate.py
```

Expected: zero violations across all three commands.

- [ ] **Step 3: Run wider test suite to confirm no collateral damage**

Run: `pytest tests/test_backtest -v`
Expected: all PASS, no new failures vs main.

---

## Phase 2 — CLI script: read parquet, pair rows, compute verdicts

**Phase scope:** 1 new script, 1 new test file with synthetic-parquet fixtures, ~5 commits.

### Task 2.1 — CLI scaffold + parquet load + model_class validation

**Files:**
- Create: `scripts/adoption_gate.py`
- Create: `tests/test_scripts/test_adoption_gate_cli.py`

- [ ] **Step 1: Write failing test for argparse + load + missing-class error**

```python
# tests/test_scripts/test_adoption_gate_cli.py
"""Plan 8 — adoption gate CLI tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.adoption_gate import load_run_parquet, validate_model_classes_present


def _make_synthetic_run(tmp_path: Path, model_classes: list[str], n_per_class: int = 800) -> Path:
    """Build a tiny per-row results.parquet with the specified model_classes."""
    rng = np.random.default_rng(0)
    rows = []
    positions = ["QB", "RB", "TE", "WR"]
    seasons = [2021, 2022, 2023, 2024]
    rows_per_pos = n_per_class // (len(positions) * len(seasons))
    for cls in model_classes:
        for pos in positions:
            for season in seasons:
                for w in range(rows_per_pos):
                    rows.append(
                        {
                            "gsis_id": f"00-{pos[0]}-{w:04d}",
                            "season": season,
                            "week": (w % 17) + 1,
                            "position": pos,
                            "model_class": cls,
                            "mean": float(rng.normal(loc=10.0, scale=5.0)),
                            "actual_ppr": float(rng.normal(loc=10.0, scale=6.0)),
                        }
                    )
    df = pd.DataFrame(rows)
    run_dir = tmp_path / "run_synthetic"
    run_dir.mkdir()
    df.to_parquet(run_dir / "results.parquet")
    return run_dir


def test_load_run_parquet_returns_frame_with_expected_columns(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"])
    df = load_run_parquet(run_dir)
    assert "model_class" in df.columns
    assert set(df["model_class"].unique()) == {"baseline", "ensemble"}


def test_load_run_parquet_raises_on_missing_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="results.parquet"):
        load_run_parquet(run_dir)


def test_validate_model_classes_present_succeeds_when_both_present(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"])
    df = load_run_parquet(run_dir)
    validate_model_classes_present(df, incumbent="baseline", candidate="ensemble")  # no raise


def test_validate_model_classes_present_raises_on_missing_candidate(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline"])
    df = load_run_parquet(run_dir)
    with pytest.raises(ValueError, match="candidate.*not present"):
        validate_model_classes_present(df, incumbent="baseline", candidate="ensemble")


def test_validate_model_classes_present_raises_on_missing_incumbent(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["ensemble"])
    df = load_run_parquet(run_dir)
    with pytest.raises(ValueError, match="incumbent.*not present"):
        validate_model_classes_present(df, incumbent="baseline", candidate="ensemble")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scripts/test_adoption_gate_cli.py -v`
Expected: 5 failures (`scripts.adoption_gate` import errors).

- [ ] **Step 3: Create the CLI scaffold**

```python
# scripts/adoption_gate.py
"""Plan 8 — adoption gate CLI.

Reads a backtest run's per-row results.parquet, pairs rows on
(gsis_id, season, week) between two model classes, and emits per-position
adoption verdicts via paired-bootstrap CIs.

Usage:
    python -m scripts.adoption_gate \\
        --run data/backtest/run_<ts> \\
        --candidate <model_class> \\
        [--incumbent baseline] \\
        [--position QB|RB|TE|WR|all] \\
        [--csv-out reports/adoption_gate_<cand>_<ts>.csv] \\
        [--n-bootstrap 1000] \\
        [--seed 42]

Spec: docs/superpowers/specs/2026-04-29-plan-8-gate-redesign-design.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_run_parquet(run_dir: Path) -> pd.DataFrame:
    """Load <run_dir>/results.parquet.

    Raises:
        FileNotFoundError: results.parquet missing under run_dir.
    """
    results_path = run_dir / "results.parquet"
    if not results_path.is_file():
        raise FileNotFoundError(
            f"results.parquet missing under {run_dir}; this CLI expects per-row "
            f"backtest output produced by scripts/backtest.py."
        )
    return pd.read_parquet(results_path)


def validate_model_classes_present(
    df: pd.DataFrame, *, incumbent: str, candidate: str
) -> None:
    """Raise ValueError if either incumbent or candidate is not in df['model_class']."""
    present = set(df["model_class"].unique())
    if candidate not in present:
        raise ValueError(
            f"candidate model_class '{candidate}' not present in run; "
            f"present classes: {sorted(present)}"
        )
    if incumbent not in present:
        raise ValueError(
            f"incumbent model_class '{incumbent}' not present in run; "
            f"present classes: {sorted(present)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan 8 adoption gate.")
    parser.add_argument("--run", type=Path, required=True, help="run_<ts> directory")
    parser.add_argument("--candidate", type=str, required=True, help="candidate model_class")
    parser.add_argument(
        "--incumbent", type=str, default="baseline", help="incumbent model_class (default baseline)"
    )
    parser.add_argument(
        "--position",
        type=str,
        choices=["QB", "RB", "TE", "WR", "all"],
        default="all",
        help="position to evaluate (default all)",
    )
    parser.add_argument("--csv-out", type=Path, default=None, help="optional CSV output path")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_run_parquet(args.run)
    validate_model_classes_present(df, incumbent=args.incumbent, candidate=args.candidate)
    print(f"Loaded {len(df)} rows from {args.run / 'results.parquet'}.")
    print(f"Model classes present: {sorted(df['model_class'].unique())}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scripts/test_adoption_gate_cli.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/adoption_gate.py tests/test_scripts/test_adoption_gate_cli.py
git commit -m "feat(adoption-gate-cli): scaffold + parquet load + class validation — Plan 8 Phase 2"
```

### Task 2.2 — Pairing logic + per-position iteration

**Files:**
- Modify: `scripts/adoption_gate.py`
- Modify: `tests/test_scripts/test_adoption_gate_cli.py`

- [ ] **Step 1: Write failing tests for pairing**

Append to `tests/test_scripts/test_adoption_gate_cli.py`:

```python
from scripts.adoption_gate import pair_rows, evaluate_position
from projections.schemas import Position


def test_pair_rows_returns_aligned_arrays_for_matched_keys(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"], n_per_class=400)
    df = load_run_parquet(run_dir)
    pos_df = df[df["position"] == "QB"]
    inc_pred, cand_pred, actual, grouping, n_dropped = pair_rows(
        pos_df, incumbent="baseline", candidate="ensemble"
    )
    assert len(inc_pred) == len(cand_pred) == len(actual) == len(grouping)
    assert len(inc_pred) > 0
    assert n_dropped == 0


def test_pair_rows_drops_unmatched_rows_with_count(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = [
        # Both classes for these keys → paired.
        {"gsis_id": "A", "season": 2021, "week": 1, "position": "QB",
         "model_class": "baseline", "mean": 10.0, "actual_ppr": 12.0},
        {"gsis_id": "A", "season": 2021, "week": 1, "position": "QB",
         "model_class": "ensemble", "mean": 11.0, "actual_ppr": 12.0},
        # Only baseline → dropped.
        {"gsis_id": "B", "season": 2021, "week": 1, "position": "QB",
         "model_class": "baseline", "mean": 10.0, "actual_ppr": 12.0},
        # Only ensemble → dropped.
        {"gsis_id": "C", "season": 2021, "week": 1, "position": "QB",
         "model_class": "ensemble", "mean": 11.0, "actual_ppr": 12.0},
    ]
    df = pd.DataFrame(rows)
    inc, cand, actual, grouping, n_dropped = pair_rows(
        df, incumbent="baseline", candidate="ensemble"
    )
    assert len(inc) == 1
    assert n_dropped == 2  # B (baseline-only) + C (ensemble-only)


def test_evaluate_position_emits_position_verdict(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"], n_per_class=2000)
    df = load_run_parquet(run_dir)
    pv = evaluate_position(
        df,
        position=Position.QB,
        incumbent="baseline",
        candidate="ensemble",
        n_bootstrap=200,
        seed=42,
    )
    assert pv.position is Position.QB
    assert pv.verdict in {"ADOPT", "MARGINAL", "DO_NOT_ADOPT"}
    assert pv.rmse_delta.n_bootstrap == 200
    assert len(pv.per_year_breakdown) == 4  # 4 held-out years
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scripts/test_adoption_gate_cli.py -v -k "pair_rows or evaluate"`
Expected: 3 failures.

- [ ] **Step 3: Implement pairing + evaluation**

Append to `scripts/adoption_gate.py`:

```python
import warnings

import numpy as np

from projections.backtest.adoption_gate import (
    PositionVerdict,
    paired_bootstrap_rmse_delta,
    paired_bootstrap_spearman_delta,
    verdict_for_position,
)
from projections.schemas import Position


def pair_rows(
    position_df: pd.DataFrame,
    *,
    incumbent: str,
    candidate: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Pair rows on (gsis_id, season, week) between incumbent and candidate.

    Args:
        position_df: rows for a single position (caller filters).
        incumbent: incumbent model_class.
        candidate: candidate model_class.

    Returns:
        (predicted_incumbent, predicted_candidate, actual, grouping, n_dropped)
        as 1-d numpy arrays plus the count of unpaired rows that were dropped.
        grouping is the held-out year per row.
    """
    inc_rows = position_df[position_df["model_class"] == incumbent]
    cand_rows = position_df[position_df["model_class"] == candidate]
    keys = ["gsis_id", "season", "week"]
    paired = inc_rows.merge(
        cand_rows,
        on=keys,
        how="inner",
        suffixes=("_inc", "_cand"),
        validate="one_to_one",
    )
    n_inc = len(inc_rows)
    n_cand = len(cand_rows)
    n_paired = len(paired)
    # Unpaired = (rows in either side) - (paired count counted once).
    n_dropped = (n_inc - n_paired) + (n_cand - n_paired)
    if n_dropped > 0:
        warnings.warn(
            f"pair_rows dropped {n_dropped} unpaired rows (incumbent={n_inc}, "
            f"candidate={n_cand}, paired={n_paired}); both model classes should "
            f"be scored on identical (gsis_id, season, week) inputs.",
            stacklevel=2,
        )
    return (
        paired["mean_inc"].to_numpy(dtype=np.float64),
        paired["mean_cand"].to_numpy(dtype=np.float64),
        paired["actual_ppr_inc"].to_numpy(dtype=np.float64),
        paired["season"].to_numpy(),
        n_dropped,
    )


def _per_year_breakdown(
    inc_pred: np.ndarray,
    cand_pred: np.ndarray,
    actual: np.ndarray,
    grouping: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """One row per year with per-year-only bootstrap CIs (informational)."""
    years = np.unique(grouping)
    rows = []
    for y in years:
        mask = grouping == y
        inc_y = (actual[mask] - inc_pred[mask])
        cand_y = (actual[mask] - cand_pred[mask])
        if mask.sum() < 100:
            rows.append(
                {
                    "year": int(y),
                    "n_paired": int(mask.sum()),
                    "rmse_delta_point": float("nan"),
                    "rmse_delta_lo": float("nan"),
                    "rmse_delta_hi": float("nan"),
                    "spearman_delta_point": float("nan"),
                    "spearman_delta_lo": float("nan"),
                    "spearman_delta_hi": float("nan"),
                }
            )
            continue
        rmse = paired_bootstrap_rmse_delta(inc_y, cand_y, n_bootstrap=n_bootstrap, seed=seed)
        # Per-year Spearman uses a single-group bootstrap.
        spear = paired_bootstrap_spearman_delta(
            inc_pred[mask],
            cand_pred[mask],
            actual[mask],
            np.full(mask.sum(), y),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        rows.append(
            {
                "year": int(y),
                "n_paired": int(mask.sum()),
                "rmse_delta_point": rmse.point,
                "rmse_delta_lo": rmse.lo_95,
                "rmse_delta_hi": rmse.hi_95,
                "spearman_delta_point": spear.point,
                "spearman_delta_lo": spear.lo_95,
                "spearman_delta_hi": spear.hi_95,
            }
        )
    return pd.DataFrame(rows)


def evaluate_position(
    df: pd.DataFrame,
    *,
    position: Position,
    incumbent: str,
    candidate: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> PositionVerdict:
    """Build a PositionVerdict for a single position from the run frame."""
    pos_df = df[df["position"] == position.value]
    inc_pred, cand_pred, actual, grouping, _ = pair_rows(
        pos_df, incumbent=incumbent, candidate=candidate
    )
    inc_residuals = actual - inc_pred
    cand_residuals = actual - cand_pred
    rmse = paired_bootstrap_rmse_delta(
        inc_residuals, cand_residuals, n_bootstrap=n_bootstrap, seed=seed
    )
    spear = paired_bootstrap_spearman_delta(
        inc_pred, cand_pred, actual, grouping, n_bootstrap=n_bootstrap, seed=seed
    )
    label, reason = verdict_for_position(rmse, spear)
    breakdown = _per_year_breakdown(
        inc_pred, cand_pred, actual, grouping, n_bootstrap=n_bootstrap, seed=seed
    )
    return PositionVerdict(
        position=position,
        incumbent_class=incumbent,
        candidate_class=candidate,
        rmse_delta=rmse,
        spearman_delta=spear,
        verdict=label,
        reason=reason,
        per_year_breakdown=breakdown,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scripts/test_adoption_gate_cli.py -v -k "pair_rows or evaluate"`
Expected: 3 PASS (the warning will appear in stderr — acceptable).

- [ ] **Step 5: Commit**

```bash
git add scripts/adoption_gate.py tests/test_scripts/test_adoption_gate_cli.py
git commit -m "feat(adoption-gate-cli): pair rows + evaluate_position — Plan 8 Phase 2"
```

### Task 2.3 — Markdown report formatter + main() wire-up

**Files:**
- Modify: `scripts/adoption_gate.py`
- Modify: `tests/test_scripts/test_adoption_gate_cli.py`

- [ ] **Step 1: Write failing tests for the formatter + end-to-end CLI**

Append to `tests/test_scripts/test_adoption_gate_cli.py`:

```python
from scripts.adoption_gate import format_position_report
import subprocess
import sys


def test_format_position_report_contains_verdict_and_breakdown(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"], n_per_class=2000)
    df = load_run_parquet(run_dir)
    pv = evaluate_position(
        df, position=Position.QB, incumbent="baseline", candidate="ensemble",
        n_bootstrap=200, seed=42,
    )
    text = format_position_report(pv)
    assert "QB" in text
    assert pv.verdict in text
    assert "RMSE" in text
    assert "Spearman" in text
    # Per-year breakdown table includes year numbers.
    assert "2021" in text and "2024" in text


def test_cli_smoke_exits_zero_with_verdict_in_stdout(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"], n_per_class=2000)
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.adoption_gate",
            "--run", str(run_dir),
            "--candidate", "ensemble",
            "--n-bootstrap", "200",
            "--position", "QB",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "QB" in result.stdout
    assert any(v in result.stdout for v in ("ADOPT", "MARGINAL", "DO_NOT_ADOPT"))


def test_cli_missing_candidate_exits_nonzero(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline"], n_per_class=2000)
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.adoption_gate",
            "--run", str(run_dir),
            "--candidate", "ensemble",
            "--n-bootstrap", "200",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "ensemble" in result.stderr or "ensemble" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scripts/test_adoption_gate_cli.py -v -k "format_position_report or cli_smoke or cli_missing"`
Expected: 3 failures (first ImportError, others CalledProcessError or stdout mismatch).

- [ ] **Step 3: Implement formatter + wire main()**

Append to `scripts/adoption_gate.py` (replace `main` body):

```python
def format_position_report(pv: PositionVerdict) -> str:
    """One-position markdown report: verdict line + per-year breakdown table."""
    lines: list[str] = []
    lines.append(
        f"### {pv.position.value} — {pv.candidate_class} vs {pv.incumbent_class}: "
        f"**{pv.verdict}**"
    )
    lines.append("")
    lines.append(f"_{pv.reason}_")
    lines.append("")
    lines.append(
        f"- n_paired: {pv.rmse_delta.n_paired_rows}; n_bootstrap: {pv.rmse_delta.n_bootstrap}"
    )
    lines.append(
        f"- RMSE delta: {pv.rmse_delta.point:+.4f} "
        f"(95% CI [{pv.rmse_delta.lo_95:+.4f}, {pv.rmse_delta.hi_95:+.4f}])"
    )
    lines.append(
        f"- Spearman delta: {pv.spearman_delta.point:+.4f} "
        f"(95% CI [{pv.spearman_delta.lo_95:+.4f}, {pv.spearman_delta.hi_95:+.4f}])"
    )
    lines.append("")
    lines.append("Per-year breakdown (informational):")
    lines.append("")
    lines.append(pv.per_year_breakdown.to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan 8 adoption gate.")
    parser.add_argument("--run", type=Path, required=True, help="run_<ts> directory")
    parser.add_argument("--candidate", type=str, required=True, help="candidate model_class")
    parser.add_argument(
        "--incumbent", type=str, default="baseline", help="incumbent model_class (default baseline)"
    )
    parser.add_argument(
        "--position",
        type=str,
        choices=["QB", "RB", "TE", "WR", "all"],
        default="all",
        help="position to evaluate (default all)",
    )
    parser.add_argument("--csv-out", type=Path, default=None, help="optional CSV output path")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_run_parquet(args.run)
    validate_model_classes_present(df, incumbent=args.incumbent, candidate=args.candidate)

    positions = (
        [Position.QB, Position.RB, Position.TE, Position.WR]
        if args.position == "all"
        else [Position(args.position)]
    )

    print(f"# Adoption gate report — {args.candidate} vs {args.incumbent}")
    print()
    print(f"Run: `{args.run}`")
    print(f"n_bootstrap: {args.n_bootstrap}, seed: {args.seed}")
    print()

    verdicts: list[PositionVerdict] = []
    for pos in positions:
        pv = evaluate_position(
            df,
            position=pos,
            incumbent=args.incumbent,
            candidate=args.candidate,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        print(format_position_report(pv))
        verdicts.append(pv)

    if args.csv_out is not None:
        _write_csv(verdicts, args.csv_out)
        print(f"Wrote CSV: {args.csv_out}")


def _write_csv(verdicts: list[PositionVerdict], path: Path) -> None:
    """Long-form CSV: one row per (position, metric, year-or-pooled)."""
    rows: list[dict[str, object]] = []
    for pv in verdicts:
        rows.append(
            {
                "position": pv.position.value,
                "incumbent": pv.incumbent_class,
                "candidate": pv.candidate_class,
                "year": "pooled",
                "metric": "rmse_delta",
                "point": pv.rmse_delta.point,
                "lo_95": pv.rmse_delta.lo_95,
                "hi_95": pv.rmse_delta.hi_95,
                "n_paired": pv.rmse_delta.n_paired_rows,
                "verdict": pv.verdict,
                "reason": pv.reason,
            }
        )
        rows.append(
            {
                "position": pv.position.value,
                "incumbent": pv.incumbent_class,
                "candidate": pv.candidate_class,
                "year": "pooled",
                "metric": "spearman_delta",
                "point": pv.spearman_delta.point,
                "lo_95": pv.spearman_delta.lo_95,
                "hi_95": pv.spearman_delta.hi_95,
                "n_paired": pv.spearman_delta.n_paired_rows,
                "verdict": pv.verdict,
                "reason": pv.reason,
            }
        )
        for _, by in pv.per_year_breakdown.iterrows():
            for metric in ("rmse_delta", "spearman_delta"):
                rows.append(
                    {
                        "position": pv.position.value,
                        "incumbent": pv.incumbent_class,
                        "candidate": pv.candidate_class,
                        "year": int(by["year"]),
                        "metric": metric,
                        "point": by[f"{metric}_point"],
                        "lo_95": by[f"{metric}_lo"],
                        "hi_95": by[f"{metric}_hi"],
                        "n_paired": int(by["n_paired"]),
                        "verdict": "",
                        "reason": "",
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scripts/test_adoption_gate_cli.py -v -k "format_position_report or cli_smoke or cli_missing"`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/adoption_gate.py tests/test_scripts/test_adoption_gate_cli.py
git commit -m "feat(adoption-gate-cli): markdown report + CSV output + main wired — Plan 8 Phase 2"
```

### Task 2.4 — CSV output test + position filter test

**Files:**
- Modify: `tests/test_scripts/test_adoption_gate_cli.py`

- [ ] **Step 1: Write failing tests for CSV output and `--position QB` filter**

Append to `tests/test_scripts/test_adoption_gate_cli.py`:

```python
def test_cli_writes_csv_when_csv_out_provided(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"], n_per_class=2000)
    csv_path = tmp_path / "out.csv"
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.adoption_gate",
            "--run", str(run_dir),
            "--candidate", "ensemble",
            "--n-bootstrap", "200",
            "--position", "QB",
            "--csv-out", str(csv_path),
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert csv_path.is_file()
    csv = pd.read_csv(csv_path)
    expected_cols = {"position", "incumbent", "candidate", "year", "metric",
                     "point", "lo_95", "hi_95", "n_paired", "verdict", "reason"}
    assert expected_cols <= set(csv.columns)
    # Two pooled rows (rmse, spearman) + 4 years × 2 metrics for QB.
    assert (csv["position"] == "QB").sum() == 10


def test_cli_position_filter_only_runs_one_position(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path, ["baseline", "ensemble"], n_per_class=2000)
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.adoption_gate",
            "--run", str(run_dir),
            "--candidate", "ensemble",
            "--n-bootstrap", "200",
            "--position", "RB",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "### RB" in result.stdout
    assert "### QB" not in result.stdout
    assert "### TE" not in result.stdout
    assert "### WR" not in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail or pass**

Run: `pytest tests/test_scripts/test_adoption_gate_cli.py -v -k "csv or position_filter"`
Expected: PASS (the implementation in Task 2.3 already supports both; this confirms it).

If they fail, implementation is wrong — revisit Task 2.3.

- [ ] **Step 3: Commit**

```bash
git add tests/test_scripts/test_adoption_gate_cli.py
git commit -m "test(adoption-gate-cli): csv output + position filter — Plan 8 Phase 2"
```

### Task 2.5 — Phase 2 verification

- [ ] **Step 1: Full test run**

```bash
pytest tests/test_scripts/test_adoption_gate_cli.py tests/test_backtest/test_adoption_gate.py -v
```

Expected: all PASS.

- [ ] **Step 2: Type + lint**

```bash
mypy scripts/adoption_gate.py tests/test_scripts/test_adoption_gate_cli.py
ruff check scripts/adoption_gate.py tests/test_scripts/test_adoption_gate_cli.py
ruff format --check scripts/adoption_gate.py tests/test_scripts/test_adoption_gate_cli.py
```

Expected: zero violations.

---

## Phase 3 — Per-position routing field on `_PositionDispatch`

**Phase scope:** 1 modified production file, 1 new test file, ~3 commits.

**Audit finding:** No script in the codebase hardcodes `factories["baseline"]()` for "the production model" — all use `factories[args.model]()` with argparse-driven selection (verified 2026-04-29 against `scripts/{train_baseline,sanity_check_baseline,predict_2024,backtest,refresh_features,diagnose_calibration,diagnose_calibration_breakdown}.py`). Phase 3 therefore adds the new field + helper + tests; **no consumer migration is required**. The field becomes load-bearing in Phase 4 when re-evaluation updates QB's default to `ensemble`.

### Task 3.1 — Add `default_model_class` field with post-init validation

**Files:**
- Modify: `src/projections/models/__init__.py`
- Create: `tests/test_models/test_position_dispatch.py`

- [ ] **Step 1: Re-read the existing `_PositionDispatch` and `POSITION_DISPATCH` definitions**

```bash
sed -n '103,189p' src/projections/models/__init__.py
```

Note: the dataclass currently has 4 fields; you are adding a 5th. The 4 dispatch entries below the dataclass need updates too.

- [ ] **Step 2: Write failing tests**

```python
# tests/test_models/test_position_dispatch.py
"""Plan 8 — per-position routing field on _PositionDispatch."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from projections.models import POSITION_DISPATCH, BaselineModel
from projections.models.base import Model
from projections.schemas import Position


def test_every_dispatch_entry_has_default_model_class() -> None:
    for pos, dispatch in POSITION_DISPATCH.items():
        assert hasattr(dispatch, "default_model_class"), (
            f"{pos} dispatch missing default_model_class"
        )
        assert dispatch.default_model_class in dispatch.factories, (
            f"{pos} default_model_class={dispatch.default_model_class!r} "
            f"not present in factories keys {sorted(dispatch.factories)}"
        )


def test_initial_default_is_baseline_for_every_position() -> None:
    for pos, dispatch in POSITION_DISPATCH.items():
        assert dispatch.default_model_class == "baseline", (
            f"{pos} default_model_class is {dispatch.default_model_class!r}, "
            f"expected 'baseline' (initial Plan 8 state — Phase 4 will update)"
        )


def test_post_init_raises_when_default_not_in_factories() -> None:
    """The factory dict's value is irrelevant for post-init validation
    (only the keys are inspected). Use the real qb_baseline factory so
    the dict typechecks; the factory itself is never invoked here.
    """
    from projections.models import _PositionDispatch, qb_baseline

    factories: dict[str, Callable[[], Model]] = {"baseline": qb_baseline}
    with pytest.raises(ValueError, match="default_model_class.*not in factories"):
        _PositionDispatch(
            factories=factories,
            feature_builder=lambda: None,  # type: ignore[arg-type]
            feature_schema=None,  # type: ignore[arg-type]
            ngs_stat_type="passing",
            default_model_class="lightgbm",  # not in factories
        )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_models/test_position_dispatch.py -v`
Expected: 3 failures (`AttributeError` for `default_model_class`).

- [ ] **Step 4: Add the field + post-init validation**

Modify `src/projections/models/__init__.py` — replace the `_PositionDispatch` dataclass:

```python
@dataclass(frozen=True)
class _PositionDispatch:
    """Per-position bundle of "what's needed to train and predict" entries.

    Plan 8 adds `default_model_class`: the production-default model class for
    this position. Consumers asking for "the production model" go through
    `production_model_for(position)`; consumers asking for a specific class
    keep going through `dispatch.factories[name]()`.

    Attributes:
        factories: model-class identifier -> zero-arg factory returning unfitted Model.
        feature_builder: position-specific build_*_features function.
        feature_schema: pandera schema for the feature builder's output.
        ngs_stat_type: NGS partition consumed by the feature builder.
        default_model_class: factories key for this position's production default.
            Must be present in `factories` (validated post-init).
    """

    factories: Mapping[str, Callable[[], Model]]
    feature_builder: Callable[..., Any]
    feature_schema: type[pa.DataFrameModel]
    ngs_stat_type: NgsStatType
    default_model_class: str

    def __post_init__(self) -> None:
        if self.default_model_class not in self.factories:
            raise ValueError(
                f"default_model_class={self.default_model_class!r} not in factories "
                f"(available: {sorted(self.factories)})"
            )
```

Also update each `POSITION_DISPATCH` entry (4 entries) to add `default_model_class="baseline"`:

```python
POSITION_DISPATCH: Mapping[Position, _PositionDispatch] = {
    Position.QB: _PositionDispatch(
        factories=_QB_FACTORIES,
        feature_builder=build_qb_features,
        feature_schema=QbFeaturesSchema,
        ngs_stat_type="passing",
        default_model_class="baseline",
    ),
    Position.RB: _PositionDispatch(
        factories=_RB_FACTORIES,
        feature_builder=build_rb_features,
        feature_schema=RbFeaturesSchema,
        ngs_stat_type="rushing",
        default_model_class="baseline",
    ),
    Position.TE: _PositionDispatch(
        factories=_TE_FACTORIES,
        feature_builder=build_te_features,
        feature_schema=TeFeaturesSchema,
        ngs_stat_type="receiving",
        default_model_class="baseline",
    ),
    Position.WR: _PositionDispatch(
        factories=_WR_FACTORIES,
        feature_builder=build_wr_features,
        feature_schema=WrFeaturesSchema,
        ngs_stat_type="receiving",
        default_model_class="baseline",
    ),
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_models/test_position_dispatch.py -v`
Expected: 3 PASS.

Also run the existing model tests to confirm nothing broke:

```bash
pytest tests/test_models -v -x
```

Expected: all PASS (the new field is additive).

- [ ] **Step 6: Commit**

```bash
git add src/projections/models/__init__.py tests/test_models/test_position_dispatch.py
git commit -m "feat(models): add default_model_class to _PositionDispatch — Plan 8 Phase 3"
```

### Task 3.2 — Add `production_model_for(position)` helper

**Files:**
- Modify: `src/projections/models/__init__.py`
- Modify: `tests/test_models/test_position_dispatch.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_models/test_position_dispatch.py`:

```python
from projections.models import production_model_for


def test_production_model_for_returns_baseline_instance_initially() -> None:
    for pos in [Position.QB, Position.RB, Position.TE, Position.WR]:
        model = production_model_for(pos)
        assert isinstance(model, BaselineModel), (
            f"{pos} initial production model should be a BaselineModel, "
            f"got {type(model).__name__}"
        )


def test_production_model_for_respects_default_model_class_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If we patch QB's default to 'ensemble', production_model_for(QB) returns
    an EnsembleModel instance."""
    from projections.models import _PositionDispatch
    from projections.models.ensemble import EnsembleModel

    qb_dispatch = POSITION_DISPATCH[Position.QB]
    patched = _PositionDispatch(
        factories=qb_dispatch.factories,
        feature_builder=qb_dispatch.feature_builder,
        feature_schema=qb_dispatch.feature_schema,
        ngs_stat_type=qb_dispatch.ngs_stat_type,
        default_model_class="ensemble",
    )
    monkeypatch.setitem(POSITION_DISPATCH, Position.QB, patched)
    model = production_model_for(Position.QB)
    assert isinstance(model, EnsembleModel)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_position_dispatch.py -v -k "production_model_for"`
Expected: 2 failures (`ImportError` for `production_model_for`).

- [ ] **Step 3: Implement the helper**

Append to `src/projections/models/__init__.py` (after `POSITION_DISPATCH`):

```python
def production_model_for(position: Position) -> Model:
    """Return a freshly instantiated production-default model for the position.

    Reads `_PositionDispatch.default_model_class` and calls the matching
    factory. The single sanctioned entry point for "the production model
    for this position" — callers asking for a specific class continue to
    use `POSITION_DISPATCH[pos].factories[name]()` directly.
    """
    dispatch = POSITION_DISPATCH[position]
    return dispatch.factories[dispatch.default_model_class]()
```

Add to `__all__`:

```python
__all__ = [
    "POSITION_DISPATCH",
    "BaselineModel",
    "EnsembleModel",
    "LightGBMModel",
    "LightGBMNbModel",
    "LightGBMTunedModel",
    "Model",
    "compute_code_hash",
    "production_model_for",  # NEW
    # ... existing factory exports
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models/test_position_dispatch.py -v`
Expected: 5 PASS (3 from Task 3.1 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/projections/models/__init__.py tests/test_models/test_position_dispatch.py
git commit -m "feat(models): production_model_for(position) helper — Plan 8 Phase 3"
```

### Task 3.3 — Phase 3 verification

- [ ] **Step 1: Run full models test suite**

```bash
pytest tests/test_models -v
```

Expected: all PASS, no regressions.

- [ ] **Step 2: Run mypy across the touched files**

```bash
mypy src/projections/models/__init__.py tests/test_models/test_position_dispatch.py
```

Expected: zero violations.

- [ ] **Step 3: Lint**

```bash
ruff check src/projections/models/__init__.py tests/test_models/test_position_dispatch.py
ruff format --check src/projections/models/__init__.py tests/test_models/test_position_dispatch.py
```

Expected: zero violations.

---

## Phase 4 — Re-evaluation: run gate, capture verdicts, update routing

**Phase scope:** Run the new CLI for all (4 candidates × 4 positions) = 16 reports; build a verdict table; update `default_model_class` for ADOPT verdicts; document everything in `project_management.md`. ~2 commits (re-evaluation report + routing update).

### Task 4.1 — Run the gate for all four candidates

**Files:**
- Create: `reports/adoption_gate_lightgbm.csv`
- Create: `reports/adoption_gate_lightgbm-tuned.csv`
- Create: `reports/adoption_gate_lightgbm-nb.csv`
- Create: `reports/adoption_gate_ensemble.csv`
- Create: `reports/adoption_gate_summary.md`

- [ ] **Step 1: Confirm the latest backtest run is the Plan 6 run**

```bash
ls data/backtest/ | tail -1
```

Expected: `run_20260429T003552Z` (or newer if a re-run happened — use whatever is most recent).

- [ ] **Step 2: Sanity-check the parquet's mean column is in fantasy-points scale**

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/backtest/run_20260429T003552Z/results.parquet')
for cls in df['model_class'].unique():
    sub = df[df['model_class'] == cls]
    print(f'{cls:20s} mean range [{sub[\"mean\"].min():.2f}, {sub[\"mean\"].max():.2f}]; '
          f'actual_ppr range [{sub[\"actual_ppr\"].min():.2f}, {sub[\"actual_ppr\"].max():.2f}]')
"
```

Expected: every class has `mean` in roughly `[0, 60]` and `actual_ppr` in roughly `[-5, 80]`. If a class has `mean` in a wildly different scale (e.g., raw stat units instead of fantasy points), STOP and investigate — `evaluate_position` would silently produce a meaningless RMSE delta.

- [ ] **Step 3: Run the gate for each candidate × all positions**

```bash
mkdir -p reports
for cand in lightgbm lightgbm-tuned lightgbm-nb ensemble; do
    python -m scripts.adoption_gate \
        --run data/backtest/run_20260429T003552Z \
        --candidate $cand \
        --csv-out reports/adoption_gate_$cand.csv \
        > reports/adoption_gate_$cand.md
done
```

Expected: 4 markdown files + 4 CSV files written. Each markdown contains 4 per-position sections.

- [ ] **Step 4: Build the summary table**

Create `reports/adoption_gate_summary.md` by hand from the four markdown reports. Format:

```markdown
# Plan 8 — adoption gate re-evaluation (run_20260429T003552Z)

| Position | Candidate | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |
|---|---|---|---|---|---|
| QB | lightgbm       | <FILL> | <FILL> | <FILL> | <FILL> |
| QB | lightgbm-tuned | <FILL> | <FILL> | <FILL> | <FILL> |
| QB | lightgbm-nb    | <FILL> | <FILL> | <FILL> | <FILL> |
| QB | ensemble       | <FILL> | <FILL> | <FILL> | <FILL> |
| RB | lightgbm       | <FILL> | <FILL> | <FILL> | <FILL> |
| ... (16 rows total)
```

Then append a "Contender chain" section per position that ADOPTs:

```markdown
## Contender chains

### QB
- Candidates that ADOPT: <list>
- Tie-break (most-negative rmse_delta.point): <winner>
- Routing change: `_PositionDispatch[Position.QB].default_model_class = "<winner>"`

### RB / TE / WR
- (likewise; omit any position with no ADOPT verdict)
```

- [ ] **Step 5: Commit the reports**

```bash
git add reports/
git commit -m "chore(plan-8): adoption gate re-evaluation reports for 4 candidates × 4 positions — Plan 8 Phase 4"
```

### Task 4.2 — Update `default_model_class` for ADOPT verdicts

**Files:**
- Modify: `src/projections/models/__init__.py`
- Modify: `tests/test_models/test_position_dispatch.py`

- [ ] **Step 1: Re-read `reports/adoption_gate_summary.md` to identify routing changes**

For each position with an ADOPT verdict (after tie-break), determine the candidate to route through.

- [ ] **Step 2: Update the loosened-tie test**

Replace `test_initial_default_is_baseline_for_every_position` with a per-position assertion driven by the actual gate verdicts. For example, if QB adopts `ensemble`:

```python
def test_default_model_class_per_position() -> None:
    """Defaults reflect Plan 8 Phase 4 re-evaluation verdicts.
    See reports/adoption_gate_summary.md for the full report.
    """
    expected = {
        Position.QB: "ensemble",     # if QB ADOPTs ensemble (illustrative)
        Position.RB: "baseline",     # no ADOPT verdict
        Position.TE: "baseline",
        Position.WR: "baseline",
    }
    for pos, want in expected.items():
        got = POSITION_DISPATCH[pos].default_model_class
        assert got == want, f"{pos} default_model_class={got!r} expected {want!r}"
```

(The actual `expected` dict comes from the re-evaluation in Task 4.1. If no candidate ADOPTs for any position, leave all four as `"baseline"` and the test reduces to a no-op assertion of the initial state.)

- [ ] **Step 3: Run test to verify it fails (initially) for the new expected values**

Run: `pytest tests/test_models/test_position_dispatch.py::test_default_model_class_per_position -v`
Expected: FAIL on positions whose default needs to change.

- [ ] **Step 4: Update `POSITION_DISPATCH` entries in `src/projections/models/__init__.py`**

For each position with an ADOPT verdict, change `default_model_class="baseline"` to `default_model_class="<adopt_winner>"` (e.g., `"ensemble"`).

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_models/test_position_dispatch.py -v`
Expected: all PASS.

- [ ] **Step 6: Run wider tests to confirm no consumer breaks**

```bash
pytest tests/test_models tests/test_backtest tests/test_scripts -v
```

Expected: all PASS. (No script hardcodes `"baseline"`; per Task 3 audit. Backtest harness takes `--model` arg explicitly.)

- [ ] **Step 7: Commit**

```bash
git add src/projections/models/__init__.py tests/test_models/test_position_dispatch.py
git commit -m "feat(models): per-position default routing — Plan 8 Phase 4 verdicts applied"
```

### Task 4.3 — Update `project_management.md` with full re-evaluation results

**Files:**
- Modify: `project_management.md`

- [ ] **Step 1: Replace the placeholder section in the Plan 8 entry with the actual results**

The Plan 8 entry at the top of `project_management.md` currently has a "Status: brainstorming → spec → plan → execute" header. After Phase 4, replace that with the real verdict tables and routing changes.

Example structure (substitute actual data from `reports/adoption_gate_summary.md`):

```markdown
## Plan 8 — Adoption Gate Redesign — re-evaluation complete (2026-04-29)

**Status:** Phases 1-4 complete; ready for PR after Phases 5-6.

### Re-evaluation verdicts (16 reports)

| Position | Candidate      | Verdict       | RMSE delta (95% CI)       | Spearman delta (95% CI)  |
|----------|----------------|---------------|---------------------------|--------------------------|
| QB       | lightgbm       | <FILL>        | <FILL>                    | <FILL>                   |
| QB       | lightgbm-tuned | <FILL>        | <FILL>                    | <FILL>                   |
| QB       | lightgbm-nb    | <FILL>        | <FILL>                    | <FILL>                   |
| QB       | ensemble       | <FILL>        | <FILL>                    | <FILL>                   |
| RB       | ... (16 rows)
...

### Per-position routing updates

| Position | Pre-Plan-8 default | Post-Plan-8 default | Reason                           |
|----------|--------------------|---------------------|----------------------------------|
| QB       | baseline           | <FILL>              | <FILL contender-chain reasoning> |
| RB       | baseline           | baseline            | <FILL — no ADOPT, or ADOPT but tie-break>
| TE       | baseline           | <FILL>              | <FILL>
| WR       | baseline           | <FILL>              | <FILL>
```

- [ ] **Step 2: Commit**

```bash
git add project_management.md
git commit -m "docs(plan-8): re-evaluation verdicts + routing updates — Plan 8 Phase 4"
```

---

## Phase 5 — Snapshot regression gate audit (read-only)

**Phase scope:** Read `src/projections/backtest/snapshot.py`; compare default tolerances against measured noise floor; document conclusion in `project_management.md`. Zero code changes — opens a follow-up TODO if the snapshot tolerances are too tight, but does not change them in this PR.

### Task 5.1 — Read tolerances and compare to measured noise floor

**Files:**
- Read-only: `src/projections/backtest/snapshot.py`
- Modify: `project_management.md`
- (Maybe modify) `TODO.md`

- [ ] **Step 1: Find the default tolerance values**

```bash
grep -n -E "tolerance|TOLERANCE|_DEFAULT" src/projections/backtest/snapshot.py | head -40
```

Look for the values keyed by `tolerance_kind` (`rmse_relative`, `mae_relative`, `spearman_absolute`, `calibration_absolute`, `mean_pred_relative`).

- [ ] **Step 2: Estimate the per-cell noise floor from Phase 4 reports**

For each (position, year) cell in the per-year breakdowns, the bootstrap CI half-width on RMSE is a direct estimate of the per-cell noise floor. Compute the median CI half-width across the 16 cells from the per-year breakdowns. (Roughly: `(hi_95 - lo_95) / 2` per row in the per-year breakdown; aggregate.)

```bash
python -c "
import pandas as pd
import numpy as np
csvs = [
    'reports/adoption_gate_lightgbm.csv',
    'reports/adoption_gate_lightgbm-tuned.csv',
    'reports/adoption_gate_lightgbm-nb.csv',
    'reports/adoption_gate_ensemble.csv',
]
half_widths = []
for c in csvs:
    df = pd.read_csv(c)
    per_year = df[(df['year'] != 'pooled') & (df['metric'] == 'rmse_delta')]
    half_widths.append(((per_year['hi_95'] - per_year['lo_95']) / 2).abs())
all_hw = pd.concat(half_widths)
print(f'per-cell RMSE delta CI half-width: median={all_hw.median():.4f}, '
      f'p75={all_hw.quantile(0.75):.4f}, max={all_hw.max():.4f}')
"
```

- [ ] **Step 3: Compare to snapshot.py's `rmse_relative` default tolerance**

If the median noise-floor half-width (in absolute fantasy-point units) divided by typical per-cell RMSE (≈ 5–8 in fantasy points) gives a relative noise floor in the same ballpark as the snapshot's default `rmse_relative` tolerance, the snapshot gate is fine. If the snapshot tolerance is much smaller (e.g., 0.5% when noise is 1.5%), file a TODO to revisit.

- [ ] **Step 4: Document the audit conclusion in `project_management.md`'s Plan 8 entry**

Add a section:

```markdown
### Snapshot regression gate audit

`src/projections/backtest/snapshot.py` default tolerances:
- `rmse_relative`: <FILL>
- `mae_relative`: <FILL>
- `spearman_absolute`: <FILL>
- `calibration_absolute`: <FILL>
- `mean_pred_relative`: <FILL>

Measured per-cell RMSE delta CI half-width across 16 cells × 4 candidates:
- median: <FILL>
- p75: <FILL>
- max: <FILL>

Translated to relative scale (CI half-width / typical per-cell RMSE ≈ <FILL>):
- median: <FILL>%

Conclusion: <"Snapshot tolerances are above the noise floor (fine as-is)" OR
"Snapshot rmse_relative tolerance is below the noise floor — file follow-up TODO">.
```

- [ ] **Step 5: If conclusion = "below noise floor", add a TODO**

Append to `TODO.md`:

```markdown
### 31. Snapshot regression-gate tolerances are below per-cell noise floor

Plan 8 Phase 5 audit found that `src/projections/backtest/snapshot.py`'s default
`rmse_relative` tolerance (<X%>) is below the measured per-cell RMSE delta noise
floor of <Y%> (median across 16 cells × 4 candidates from Plan 8's bootstrap).
This means the snapshot regression gate may flag false-positive regressions when
the underlying model is unchanged (e.g., refitting after a code-hash rotation
that touches no math).

Fix candidates:
- Loosen the default `rmse_relative` tolerance to ~2× the measured noise floor.
- Or replace the fixed-tolerance check with a paired-bootstrap CI on the
  per-row residuals (port Plan 8's `paired_bootstrap_rmse_delta`).

Defer until a real false-positive snapshot regression occurs, or until Plan 8
Phase 5's measurement gives us a stable noise-floor estimate to argue from.
```

- [ ] **Step 6: Commit**

```bash
git add project_management.md TODO.md
git commit -m "docs(plan-8): snapshot regression gate audit — Plan 8 Phase 5"
```

(If only `project_management.md` was touched — no TODO needed — drop `TODO.md` from the add and adjust the message.)

---

## Phase 6 — §1.3 spec template

**Phase scope:** 1 new template file. Single commit.

### Task 6.1 — Create the §1.3 template for future model-class specs

**Files:**
- Create: `docs/superpowers/specs/_adoption_gate_template.md`

- [ ] **Step 1: Write the template**

```markdown
# §1.3 Adoption gate template (Plan 8)

**For author:** copy the body below into your new spec's §1.3 section.
Replace `<CANDIDATE_MODEL_CLASS>` and any other angle-bracket placeholders
with concrete values. Do **not** include / symlink this file — copy inline,
so your spec carries the gate it was evaluated under as record-of-decision.

**Spec history:** introduced in Plan 8
(`docs/superpowers/specs/2026-04-29-plan-8-gate-redesign-design.md`),
replacing the prior §1.3 "three-criteria, no-significance-test" gate that
killed Plans 3e / 5 / 5b / 5c / 7 / 6 from sampling noise on a metric
no consumer needs.

---

## 1.3 Adoption gate

Adoption decisions are **per position**. For each `Position P ∈ {QB, RB, TE, WR}`,
the adoption gate compares `<CANDIDATE_MODEL_CLASS>` against the incumbent
`_PositionDispatch[P].default_model_class`.

**Inputs.** Per-row predictions from both classes for the same
`(gsis_id, season, week)` rows across all held-out years (currently 2021–2024),
pulled from a single backtest run's `results.parquet`. After pairing, position P
contributes ~3,000–8,000 paired rows.

**Statistical machinery.** Paired bootstrap with `n_bootstrap=1000`,
deterministic seed `42`. Resampling unit is the paired player-week — both
candidate and incumbent are scored on the same draw.

**Per-position metrics.**
- **RMSE delta** (`candidate - incumbent`): pooled across all held-out years.
  Negative = candidate wins.
- **Spearman delta**: per-year Spearman computed within each held-out year,
  then averaged unweighted across years.

Per-cell breakdowns (one row per held-out year) are emitted for inspection
but do **not** gate adoption.

**Verdict rule.**
```
PASS_RMSE      := rmse_delta.hi_95     <  0.0
PASS_SPEARMAN  := spearman_delta.lo_95 > -0.02

if  PASS_RMSE and  PASS_SPEARMAN:  ADOPT
if  PASS_RMSE and !PASS_SPEARMAN:  MARGINAL — investigate before adopting
if !PASS_RMSE and  PASS_SPEARMAN:  DO_NOT_ADOPT
if !PASS_RMSE and !PASS_SPEARMAN:  DO_NOT_ADOPT
```

**What this gate does not check:**
- No per-cell pass/fail — per-year deltas are informational; only the
  position-pooled CI gates.
- No Spearman-improvement requirement — only the catastrophic-regression floor.
- No calibration check at all. `weekly_calibration_*` and
  `season_calibration_*` continue to be emitted into the snapshot for
  monitoring; the adoption decision ignores them.
- No "max worse cell" floor — sampling variation on a single year is not
  adoption-blocking.

**Tie-breaking** when multiple candidates ADOPT for the same position: the
candidate with the most-negative `rmse_delta.point` is selected. Document
the contender chain in this spec.

**Adoption is manual.** `scripts/adoption_gate.py` emits a report; a human
reads the verdicts and edits `_PositionDispatch[P].default_model_class` for
any position where the verdict is `ADOPT`. The CLI never writes to source.

**Tooling.** Run:
```
python -m scripts.adoption_gate \\
  --run data/backtest/run_<ts> \\
  --candidate <CANDIDATE_MODEL_CLASS> \\
  --csv-out reports/adoption_gate_<CANDIDATE_MODEL_CLASS>.csv
```

Capture the per-(position) verdict + CI table in this spec's verdict section.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/_adoption_gate_template.md
git commit -m "docs(plan-8): §1.3 adoption gate template — Plan 8 Phase 6"
```

---

## Phase 7 — Final verification + PR

**Phase scope:** End-to-end check of all gates (spec §3 of CLAUDE.md "FORCED VERIFICATION"). PR creation.

### Task 7.1 — Full pytest + mypy + ruff sweep

- [ ] **Step 1: Run the full test suite**

```bash
pytest -v
```

Expected: all PASS. If anything fails, **stop and fix** — do not commit / PR a failing build.

- [ ] **Step 2: Run mypy**

```bash
mypy src tests
```

Expected: zero violations.

- [ ] **Step 3: Run ruff check + format check**

```bash
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 4: Run the ingest-touching subset (defense-in-depth per CLAUDE.md)**

```bash
pytest -v -k "ingest or store or schemas"
```

Expected: all PASS. (Plan 8 doesn't touch ingest / store / schemas, so this is a no-regression check.)

### Task 7.2 — Update Plan 8 PM entry with final state

**Files:**
- Modify: `project_management.md`

- [ ] **Step 1: Update the Plan 8 entry's "Status" line**

Replace `Status: brainstorming → spec → plan → execute. No code yet.` (or the Phase-4 status from Task 4.3) with:

```markdown
**Status:** complete; ready for PR. All 7 phases shipped.
```

- [ ] **Step 2: Confirm the bottom-of-file "Next action" still points at Track 2 (TODO #3 PBP / EPA features)**

Already updated 2026-04-29; no edit needed unless a Phase-5 audit follow-up changed it.

- [ ] **Step 3: Commit**

```bash
git add project_management.md
git commit -m "docs(plan-8): mark Plan 8 complete in PM doc — Plan 8 Phase 7"
```

### Task 7.3 — PR

- [ ] **Step 1: Push branch**

```bash
git push -u origin feat/plan-8-gate-redesign
```

- [ ] **Step 2: Create PR**

```bash
gh pr create --title "Plan 8 — Adoption gate redesign + per-position routing" --body "$(cat <<'EOF'
## Summary
- Replaces §1.3 with paired-bootstrap CI gate (RMSE one-sided 95%; Spearman catastrophic-regression-only floor; calibration informational)
- Adds per-position routing: `_PositionDispatch.default_model_class` + `production_model_for(position)` helper
- Re-evaluates 4 existing peer models (C, C-tuned, C-NB, D) under the new gate; updates `default_model_class` for ADOPT verdicts
- New CLI `scripts/adoption_gate.py` reads any backtest run's per-row parquet
- §1.3 template for future model-class specs at `docs/superpowers/specs/_adoption_gate_template.md`
- Snapshot regression gate audit (read-only) documented in PM doc

Diagnosis behind this plan: PRs 10-15 (Plans 3e/5/5b/5c/7/6) all failed adoption from below-noise-floor thresholds + a calibration metric no consumer needs. See `docs/superpowers/specs/2026-04-29-plan-8-gate-redesign-design.md` for the full diagnosis + design.

## Test plan
- [ ] All `tests/test_backtest/test_adoption_gate.py` pass (~17 tests)
- [ ] All `tests/test_scripts/test_adoption_gate_cli.py` pass (~10 tests)
- [ ] All `tests/test_models/test_position_dispatch.py` pass (~6 tests)
- [ ] `pytest -v` (full suite) passes with zero new failures
- [ ] `mypy src tests` zero violations
- [ ] `ruff check src tests && ruff format --check src tests` zero violations
- [ ] Re-evaluation reports in `reports/` reviewed against expected `Strong prior` in spec §6

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Confirm PR URL printed; no other action**

Stop here. Reviewer takes over.

---

## Self-review notes (added during plan writing)

- **Spec coverage:** Every section of `2026-04-29-plan-8-gate-redesign-design.md` maps to at least one task. §1.1 Goals → Phases 1, 2, 3, 4, 6. §1.2 Non-goals → enforced by what tasks do NOT touch (no model code, no harness changes). §1.3 → Phase 6 (template) + Phase 1.4 (verdict rule implementation). §2 Architecture → Phases 1, 2, 3 file layout. §3 Component contracts → Phase 1 (stats) + Phase 2 (CLI). §4 Error handling → Tasks 1.2/1.3/1.4 raise/NaN tests + Task 2.1 missing-class test. §5 Testing → all phase tests. §6 Re-evaluation → Phase 4. §7 Open questions → noted in tasks (mean-column scale check at Task 4.1 Step 2; year-block bootstrap risk explicitly deferred per spec §7).
- **Placeholder scan:** `<FILL>` markers exist in Task 4.3 and Task 5.1 — these are intentional placeholders for *runtime data* that the implementer fills from the actual gate output. Not "TBD" / "implement later" placeholders for code or design decisions. All code blocks are concrete and executable as written.
- **Type consistency:** `BootstrapDelta`, `PositionVerdict`, `paired_bootstrap_rmse_delta`, `paired_bootstrap_spearman_delta`, `verdict_for_position`, `production_model_for`, `_PositionDispatch.default_model_class`, `pair_rows`, `evaluate_position`, `format_position_report`, `load_run_parquet`, `validate_model_classes_present`, `_per_year_breakdown`, `_write_csv`, `_per_group_mean_spearman`, `_MIN_PAIRED_ROWS`, `VerdictLabel` are referenced consistently across tasks (no rename drift).
- **Dependency order:** Task 1.x → 2.x → 3.x → 4.x → 5 → 6 → 7. Phase 4 depends on Phases 1, 2, 3 all complete (CLI must work + per-position field must exist). Tasks within a phase are TDD-ordered (test → fail → impl → pass → commit).
