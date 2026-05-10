# WR Receiving Stats Target Decomposition Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Probe whether decomposing WR receiving stats into a shared `targets` volume × per-stat efficiency factor (`catch_rate`, `yards_per_target`, `td_rate_per_target`) beats the current per-stat RidgeCV on out-of-sample mean prediction. Returns three per-stat verdicts (SIGNAL / NULL / REGRESSION) over the 2021–2024 walk-forward eval window. NULL on all three closes target decomposition cheaply at this unit; SIGNAL on any one greenlights an integration plan.

**Architecture:** New module `src/projections/backtest/target_decomposition_probe.py` is a self-contained CV harness that mirrors the shape of `feature_probe.py`'s Phase 1 but compares two prediction recipes (direct vs decomposed RidgeCV) instead of two feature sets. Pure numpy / pandas / sklearn. No new ingest, no schema changes, no model-class additions, no scoring-layer changes. Reuses `paired_bootstrap_rmse_delta` and `BootstrapDelta` from `adoption_gate.py` unchanged. CLI loads the WR feature cache + weekly stats and dispatches the harness. The probe deliberately tests Ridge-only (matches `BaselineModel`'s algorithmic family) so any SIGNAL is attributable to *decomposition itself*, not to a model-class change.

**Tech Stack:** numpy, pandas, scikit-learn (RidgeCV), pandera, pytest, mypy strict, ruff. Reuses `projections.features.cache.read_features`, `projections.store.read_partition`, `projections.backtest.adoption_gate.paired_bootstrap_rmse_delta`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md`.

**Branch:** `feat/probe-target-decomposition` (already created; spec already committed at `a37e761`).

---

## File Structure

**Create:**
- `src/projections/backtest/target_decomposition_probe.py` — probe module: sub-model fitters, predictors, walk-forward harness, verdict mapping, report renderers, dataclasses. Single-responsibility module ≤ 600 LOC.
- `tests/test_backtest/test_target_decomposition_probe.py` — unit tests for every public callable in the module + edge cases.
- `scripts/probe_target_decomposition.py` — argparse CLI: load feature cache + weekly stats, invoke harness, write reports.
- `tests/test_scripts/test_probe_target_decomposition_cli.py` — CLI argparse + dispatch tests.

**Generated at run time (Task 5, committed):**
- `reports/feature_probe_target_decomposition_summary.md` — hand-written family summary including verdict + decision narrative + composite-fpts translation + factor residual correlation note.
- `reports/feature_probe_target_decomposition_per_stat.csv` — machine-readable per-stat verdict table.
- `reports/feature_probe_target_decomposition_receptions.md`
- `reports/feature_probe_target_decomposition_receiving_yards.md`
- `reports/feature_probe_target_decomposition_receiving_tds.md`

**Modify:**
- `project_management.md` — prepend a "WR Receiving Stats Target Decomposition Probe" decision-log entry (Task 6).
- `TODO.md` — append an Update under #23 with verdict + follow-up disposition (Task 6).

**Untouched (deliberately):**
- `src/projections/schemas.py` — no new schemas; existing `WrFeaturesSchema` and `WeeklyStatsSchema` cover all inputs.
- `src/projections/models/baseline.py` — probe imports `_WR_FEATURE_COLUMNS` but does not modify it.
- `src/projections/models/__init__.py` — no new factories, no `_WR_FACTORIES` change.
- `src/projections/distributions/*` — no new distribution classes; probe is point-prediction only.
- `src/projections/scoring/*` — probe does not invoke `score_distribution`.
- `scripts/probe_feature_signal.py` — feature-override probe, structurally different from this one; not touched.
- `scripts/adoption_gate.py` — gate is for the integration plan, not the probe.

---

## Task 1: Sub-model fit / predict primitives

**Files:**
- Create: `src/projections/backtest/target_decomposition_probe.py`
- Create: `tests/test_backtest/test_target_decomposition_probe.py`

The two fitters and two predictors are pure numpy/sklearn. Test each with a tiny synthetic frame so the unit tests run in milliseconds.

- [ ] **Step 1: Write the failing test for `_StatDecomp` and the decomp registry**

```python
# tests/test_backtest/test_target_decomposition_probe.py
"""Tests for src/projections/backtest/target_decomposition_probe.py.

Spec: docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.backtest.target_decomposition_probe import (
    _WR_RECEIVING_DECOMPS,
    _StatDecomp,
)
from projections.schemas import Stat


def test_wr_receiving_decomps_registry_has_three_stats() -> None:
    """Three decomposed stats — receptions, receiving_yards, receiving_tds.

    Each shares Stat.TARGETS as its volume factor; efficiency clip-hi is 1.0
    for ratios and +inf for yards-per-target.
    """
    assert set(_WR_RECEIVING_DECOMPS.keys()) == {
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    }
    for stat, decomp in _WR_RECEIVING_DECOMPS.items():
        assert decomp.volume_stat is Stat.TARGETS
        assert decomp.numerator_stat is stat
    assert _WR_RECEIVING_DECOMPS[Stat.RECEPTIONS].efficiency_clip_hi == 1.0
    assert _WR_RECEIVING_DECOMPS[Stat.RECEIVING_TDS].efficiency_clip_hi == 1.0
    assert _WR_RECEIVING_DECOMPS[Stat.RECEIVING_YARDS].efficiency_clip_hi == float("inf")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py::test_wr_receiving_decomps_registry_has_three_stats -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'projections.backtest.target_decomposition_probe'`.

- [ ] **Step 3: Create the module skeleton with `_StatDecomp` and the registry**

```python
# src/projections/backtest/target_decomposition_probe.py
"""Target decomposition probe — model architecture probe.

Tests whether decomposing WR receiving stats into a shared `targets` volume
factor times a per-stat efficiency factor beats per-stat direct RidgeCV on
out-of-sample mean prediction. Per-stat Δ-CV-RMSE × 3 stats × walk-forward
eval window 2021–2024, paired-bootstrap CI on pooled residuals.

Architecturally distinct from `feature_probe.py`: that module probes feature
additions via override parquets; this module probes a prediction recipe
change with no override layer. Reuses `paired_bootstrap_rmse_delta` and
`BootstrapDelta` from `adoption_gate.py`.

Spec: docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.linear_model import RidgeCV

from projections.schemas import Stat

# Same alpha grid as BaselineModel.fit (src/projections/models/baseline.py:563).
_RIDGE_ALPHAS: Final = np.logspace(-3, 3, 13)


@dataclass(frozen=True, slots=True)
class _StatDecomp:
    """Per-stat decomposition spec.

    The probe measures one decomposed prediction per stat:
        mu_decomposed[stat] = clip(volume_ridge.predict(X), 0, +inf)
                            * clip(efficiency_ridge.predict(X), 0, efficiency_clip_hi)

    The shared volume sub-model is fit once per training window (against
    `volume_stat` directly); each efficiency sub-model is fit on rows where
    `volume_stat > 0` against `numerator_stat / volume_stat`.
    """

    volume_stat: Stat
    efficiency_label: str
    efficiency_clip_hi: float
    numerator_stat: Stat


_WR_RECEIVING_DECOMPS: Final[dict[Stat, _StatDecomp]] = {
    Stat.RECEPTIONS: _StatDecomp(
        volume_stat=Stat.TARGETS,
        efficiency_label="catch_rate",
        efficiency_clip_hi=1.0,
        numerator_stat=Stat.RECEPTIONS,
    ),
    Stat.RECEIVING_YARDS: _StatDecomp(
        volume_stat=Stat.TARGETS,
        efficiency_label="yards_per_target",
        efficiency_clip_hi=float("inf"),
        numerator_stat=Stat.RECEIVING_YARDS,
    ),
    Stat.RECEIVING_TDS: _StatDecomp(
        volume_stat=Stat.TARGETS,
        efficiency_label="td_rate_per_target",
        efficiency_clip_hi=1.0,
        numerator_stat=Stat.RECEIVING_TDS,
    ),
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py::test_wr_receiving_decomps_registry_has_three_stats -v`
Expected: PASS.

- [ ] **Step 5: Write failing tests for `_fit_direct` and `_fit_decomposed_volume`**

Append to `tests/test_backtest/test_target_decomposition_probe.py`:

```python
from projections.backtest.target_decomposition_probe import (
    _fit_decomposed_efficiency,
    _fit_decomposed_volume,
    _fit_direct,
    _predict_decomposed,
    _predict_direct,
    _RIDGE_ALPHAS,
)


def _synthetic_xy(n: int = 50, n_features: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_features))
    coef = np.array([1.0, -0.5, 0.25, 0.0])
    y = x @ coef + 0.1 * rng.standard_normal(n)
    return x, y


def test_fit_direct_returns_ridgecv_with_canonical_alphas() -> None:
    x, y = _synthetic_xy()
    ridge = _fit_direct(x, y)
    assert isinstance(ridge, RidgeCV)
    np.testing.assert_array_equal(ridge.alphas, _RIDGE_ALPHAS)
    # Sanity: prediction shape matches input rows.
    assert ridge.predict(x).shape == (len(x),)


def test_fit_decomposed_volume_targets_y_is_volume_stat() -> None:
    x, _ = _synthetic_xy()
    targets = np.arange(len(x), dtype=np.int64)  # arbitrary non-trivial targets
    ridge = _fit_decomposed_volume(x, targets)
    assert isinstance(ridge, RidgeCV)
    # Coefficient is a 4-vector matching n_features.
    assert ridge.coef_.shape == (4,)
```

Add the import at the top:
```python
from sklearn.linear_model import RidgeCV
```

- [ ] **Step 6: Run the tests, expect failures**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py -v`
Expected: FAIL on the two new tests (`AttributeError` on `_fit_direct`).

- [ ] **Step 7: Implement `_fit_direct` and `_fit_decomposed_volume`**

Append to `src/projections/backtest/target_decomposition_probe.py`:

```python
def _fit_direct(x: np.ndarray, y: np.ndarray) -> RidgeCV:
    """Fit RidgeCV(_RIDGE_ALPHAS) on (x, y).

    Caller is responsible for: NaN drop, bool-to-int8 coercion of x cols,
    targets-positive filtering when fitting an efficiency factor. This helper
    is pure: arrays in, fitted ridge out.
    """
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
    ridge.fit(x, y.astype(np.float64))
    return ridge


def _fit_decomposed_volume(x: np.ndarray, targets: np.ndarray) -> RidgeCV:
    """Fit the shared volume sub-model on `targets` directly.

    Trained on the un-filtered training rows (zero-target rows are legitimate
    observations of low-volume players, and the volume model needs to predict
    them).
    """
    return _fit_direct(x, targets.astype(np.float64))
```

- [ ] **Step 8: Run tests, verify pass**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py -v`
Expected: 3 PASS.

- [ ] **Step 9: Write failing test for `_fit_decomposed_efficiency`**

```python
def test_fit_decomposed_efficiency_filters_to_targets_positive() -> None:
    """Efficiency arm trains only on rows where targets > 0; ratio is
    numerator / targets on those rows.
    """
    x, _ = _synthetic_xy(n=20, seed=1)
    # Half the rows have targets > 0; the rest are zero-target.
    targets = np.array([5, 0, 3, 0, 8, 0, 2, 0, 4, 0, 6, 0, 1, 0, 7, 0, 9, 0, 3, 0])
    numerator = targets * 0.5  # so true catch_rate is 0.5 on every targets > 0 row
    numerator[targets == 0] = 0  # well-defined ratio nonexistent here, but predict won't see these
    ridge = _fit_decomposed_efficiency(x, numerator, targets)
    assert isinstance(ridge, RidgeCV)
    # On the targets > 0 subset, ratio is constant 0.5; ridge should predict ~0.5 everywhere.
    pred = ridge.predict(x)
    assert np.allclose(pred, 0.5, atol=0.05)


def test_fit_decomposed_efficiency_raises_when_no_positive_targets() -> None:
    """All-zero-targets training set is malformed; raise rather than silently
    return a ridge fit on zero rows.
    """
    x, _ = _synthetic_xy(n=10)
    targets = np.zeros(10, dtype=np.int64)
    numerator = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError, match="targets > 0"):
        _fit_decomposed_efficiency(x, numerator, targets)
```

- [ ] **Step 10: Run tests, expect failures**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py -v`
Expected: 2 FAIL on the new tests, 3 PASS on the prior ones.

- [ ] **Step 11: Implement `_fit_decomposed_efficiency`**

Append:

```python
def _fit_decomposed_efficiency(
    x: np.ndarray,
    numerator: np.ndarray,
    targets: np.ndarray,
) -> RidgeCV:
    """Fit an efficiency sub-model on rows where targets > 0.

    Ratio = numerator / targets on those rows. Caller passes `numerator`
    already aligned with `x` and `targets`; this helper handles the masking.

    Raises:
        ValueError: no rows in the training set have targets > 0.
    """
    mask = targets > 0
    if not mask.any():
        raise ValueError(
            "Cannot fit efficiency factor: no training rows with targets > 0. "
            "Check the training-window filter."
        )
    x_pos = x[mask]
    targets_pos = targets[mask].astype(np.float64)
    numerator_pos = numerator[mask].astype(np.float64)
    ratio = numerator_pos / targets_pos
    return _fit_direct(x_pos, ratio)
```

- [ ] **Step 12: Run tests, verify pass**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py -v`
Expected: 5 PASS.

- [ ] **Step 13: Write failing tests for `_predict_direct` and `_predict_decomposed`**

```python
def test_predict_direct_returns_float64_array_of_eval_shape() -> None:
    x_train, y_train = _synthetic_xy(seed=2)
    ridge = _fit_direct(x_train, y_train)
    x_eval, _ = _synthetic_xy(n=30, seed=3)
    pred = _predict_direct(ridge, x_eval)
    assert pred.dtype == np.float64
    assert pred.shape == (30,)


def test_predict_decomposed_clips_volume_at_zero_and_efficiency_at_clip_hi() -> None:
    """Volume floored at 0; efficiency clipped to [0, efficiency_clip_hi].

    Construct a contrived volume_ridge that predicts negatives on some rows
    and an efficiency_ridge that predicts > 1 on some rows, with
    efficiency_clip_hi = 1.0. Verify the product respects both clips.
    """
    # Train ridges with extreme synthetic data so we control the predictions.
    rng = np.random.default_rng(4)
    n = 8
    x = rng.standard_normal((n, 2))

    # Volume ridge: y_train = -x[:, 0] * 5 (negatives appear when x[:, 0] > 0)
    volume_y = -x[:, 0] * 5.0
    volume_ridge = _fit_direct(x, volume_y)

    # Efficiency ridge: y_train = x[:, 1] * 0.5 + 0.5, range roughly [-0.5, 1.5]
    efficiency_y = x[:, 1] * 0.5 + 0.5
    efficiency_ridge = _fit_direct(x, efficiency_y)

    pred = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=efficiency_ridge,
        x=x,
        efficiency_clip_hi=1.0,
    )
    assert pred.dtype == np.float64
    assert pred.shape == (n,)
    # Product of clipped factors is non-negative and bounded above by max(volume_clip, +inf) * 1.0.
    assert (pred >= 0.0).all()
    # On rows where volume_ridge predicts < 0, the product is exactly 0.
    raw_volume = volume_ridge.predict(x)
    assert np.all(pred[raw_volume < 0] == 0.0)


def test_predict_decomposed_no_clip_on_efficiency_hi_when_inf() -> None:
    """yards_per_target case: efficiency_clip_hi = +inf; Ridge predictions
    above ~15 (empirical max) are not clipped on the high side."""
    rng = np.random.default_rng(5)
    n = 6
    x = rng.standard_normal((n, 2))
    volume_y = np.full(n, 10.0)  # volume ridge predicts ~10 everywhere
    volume_ridge = _fit_direct(x, volume_y)
    efficiency_y = np.full(n, 25.0)  # efficiency ridge predicts ~25 everywhere
    efficiency_ridge = _fit_direct(x, efficiency_y)

    pred = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=efficiency_ridge,
        x=x,
        efficiency_clip_hi=float("inf"),
    )
    # No upper clip; product is ~250.
    assert np.all(pred > 200.0)
```

- [ ] **Step 14: Run tests, expect failures**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py -v`
Expected: 3 FAIL on the new tests, 5 PASS prior.

- [ ] **Step 15: Implement `_predict_direct` and `_predict_decomposed`**

Append:

```python
def _predict_direct(ridge: RidgeCV, x: np.ndarray) -> np.ndarray:
    """Direct per-row mu prediction; matches BaselineModel.fit's predict semantics
    (no clipping; downstream Distribution constructor handles family floors).
    """
    return ridge.predict(x).astype(np.float64)


def _predict_decomposed(
    *,
    volume_ridge: RidgeCV,
    efficiency_ridge: RidgeCV,
    x: np.ndarray,
    efficiency_clip_hi: float,
) -> np.ndarray:
    """Decomposed per-row mu prediction.

    mu = clip(volume.predict(x), 0, +inf) * clip(efficiency.predict(x), 0, hi)

    Both clips engage on the *low* side; the high side engages only for
    bounded-rate efficiency factors (catch_rate, td_rate_per_target with
    efficiency_clip_hi=1.0). yards_per_target uses efficiency_clip_hi=+inf so
    the high side is a no-op.
    """
    volume = np.maximum(volume_ridge.predict(x).astype(np.float64), 0.0)
    eff = efficiency_ridge.predict(x).astype(np.float64)
    eff = np.clip(eff, 0.0, efficiency_clip_hi)
    return volume * eff
```

- [ ] **Step 16: Run tests, verify all 8 pass**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py -v`
Expected: 8 PASS.

- [ ] **Step 17: Run mypy + ruff on the new module**

Run: `PATH="$(pwd)/.venv/Scripts:$PATH" mypy src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py && ruff check src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py && ruff format --check src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py`
Expected: zero errors on each.

- [ ] **Step 18: Commit**

```bash
PATH="$(pwd)/.venv/Scripts:$PATH" git add src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py
PATH="$(pwd)/.venv/Scripts:$PATH" git commit -m "feat(probe-target-decomposition): sub-model fit/predict primitives — Task 1

Probe core: _StatDecomp registry pinning the WR receiving 3-stat 2-factor
decomposition, _fit_direct/_fit_decomposed_volume/_fit_decomposed_efficiency
RidgeCV fitters (alphas match BaselineModel.fit), and _predict_direct/_predict_decomposed
predictors with predict-time clipping (volume floor 0; efficiency clip [0, hi]
keyed off _StatDecomp.efficiency_clip_hi).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Walk-forward harness

**Files:**
- Modify: `src/projections/backtest/target_decomposition_probe.py` (append).
- Modify: `tests/test_backtest/test_target_decomposition_probe.py` (append).

The harness consumes a per-year features dict + weekly_stats DataFrame, walk-forwards over eval years, and emits per-stat residual buffers + per-year coverage. Pure dataclass-out, no IO.

- [ ] **Step 1: Write failing test for the `WalkForwardOutput` dataclass shape**

```python
from projections.backtest.target_decomposition_probe import (
    WalkForwardOutput,
    walk_forward_residuals,
)


def _make_synthetic_walk_forward_inputs(
    seasons: tuple[int, ...] = (2018, 2019, 2020, 2021),
    n_per_season: int = 25,
    seed: int = 6,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Build a tiny features dict and weekly_stats frame whose schema is
    compatible with the harness. Identifying cols are stable (same gsis_ids
    across seasons) so the inner-join joins.
    """
    rng = np.random.default_rng(seed)
    feature_cols = ["targets_per_game_l4", "receiving_yards_per_game_l4"]
    features_by_year: dict[int, pd.DataFrame] = {}
    weekly_rows: list[dict[str, object]] = []
    for season in seasons:
        n = n_per_season
        gsis_ids = [f"00-{season:04d}{i:03d}" for i in range(n)]
        feat_df = pd.DataFrame(
            {
                "gsis_id": gsis_ids,
                "season": season,
                "week": rng.integers(1, 18, size=n),
                "team": "KC",
                "opponent": "DEN",
                **{c: rng.standard_normal(n) for c in feature_cols},
            }
        )
        features_by_year[season] = feat_df
        # Construct weekly stats so targets is non-zero on most rows.
        targets = rng.integers(0, 12, size=n)
        for i, gid in enumerate(gsis_ids):
            t = int(targets[i])
            weekly_rows.append(
                {
                    "gsis_id": gid,
                    "season": season,
                    "week": int(feat_df["week"].iloc[i]),
                    "position": "WR",
                    "team": "KC",
                    "opponent": "DEN",
                    "targets": t,
                    "receptions": int(t * 0.6),
                    "receiving_yards": float(t * 8.0),
                    "receiving_tds": int(t * 0.05),
                    "passing_yards": 0.0,
                    "passing_tds": 0,
                    "interceptions": 0,
                    "attempts": 0,
                    "completions": 0,
                    "sacks": 0,
                    "rushing_yards": 0.0,
                    "rushing_tds": 0,
                    "carries": 0,
                    "receiving_air_yards": 0.0,
                    "fumbles_lost": 0,
                }
            )
    weekly_stats = pd.DataFrame(weekly_rows)
    return features_by_year, weekly_stats


def test_walk_forward_output_shape_matches_three_stats() -> None:
    """Output is a WalkForwardOutput with one entry per decomposed stat
    (receptions, receiving_yards, receiving_tds), each holding aligned
    arrays of (actual, mu_direct, mu_decomposed) of equal length n_paired.
    """
    features_by_year, weekly_stats = _make_synthetic_walk_forward_inputs()
    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
        eval_years=(2021,),
        train_start=2018,
    )
    assert isinstance(out, WalkForwardOutput)
    assert set(out.per_stat.keys()) == {
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    }
    for stat, residuals in out.per_stat.items():
        assert residuals.actual.shape == residuals.mu_direct.shape
        assert residuals.actual.shape == residuals.mu_decomposed.shape
        assert residuals.n_paired == len(residuals.actual)
```

- [ ] **Step 2: Run test, expect failure**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py::test_walk_forward_output_shape_matches_three_stats -v`
Expected: FAIL on `ImportError: cannot import name 'WalkForwardOutput'`.

- [ ] **Step 3: Add the dataclasses**

Append to `src/projections/backtest/target_decomposition_probe.py`:

```python
import pandas as pd

from collections.abc import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class StatResiduals:
    """Pooled residuals for one decomposed stat across all eval years.

    Each array is row-aligned: row i is the same eval-year row in both arms.
    """

    actual: np.ndarray
    mu_direct: np.ndarray
    mu_decomposed: np.ndarray
    n_paired: int


@dataclass(frozen=True, slots=True)
class FactorResidualsByYear:
    """Per-eval-year (volume_residual, efficiency_residual) pairs for one stat.

    Used for the §5 risk #2 Pearson correlation diagnostic. Volume residual is
    actual_targets - predicted_targets; efficiency residual is
    actual_efficiency_ratio - predicted_efficiency on rows with targets > 0.
    """

    eval_year: int
    volume_residuals: np.ndarray
    efficiency_residuals: np.ndarray


@dataclass(frozen=True, slots=True)
class CoverageByYear:
    """Per-eval-year `targets > 0` rate on WR rows."""

    eval_year: int
    targets_positive_rate: float
    n_eval_rows: int


@dataclass(frozen=True, slots=True)
class WalkForwardOutput:
    """Bundle of all walk-forward outputs.

    `per_stat` keys: the 3 entries in `_WR_RECEIVING_DECOMPS`.
    `factor_residuals_by_year`: per-stat per-year pairs of factor residuals
        for the orthogonality-correlation diagnostic.
    `coverage_by_year`: targets > 0 rate per eval year on the eval rows.
    `train_coverage_by_year`: targets > 0 rate per eval year on the *training*
        rows for that walk-forward iteration. Same threshold applies.
    """

    per_stat: Mapping[Stat, StatResiduals]
    factor_residuals_by_year: Mapping[Stat, Sequence[FactorResidualsByYear]]
    coverage_by_year: Sequence[CoverageByYear]
    train_coverage_by_year: Sequence[CoverageByYear]
    eval_years: tuple[int, ...]
```

- [ ] **Step 4: Run test, still failing on `walk_forward_residuals` not implemented**

Run: same command. Expected: FAIL on `ImportError: cannot import name 'walk_forward_residuals'`.

- [ ] **Step 5: Implement `walk_forward_residuals`**

Append:

```python
def walk_forward_residuals(
    *,
    features_by_year: Mapping[int, pd.DataFrame],
    weekly_stats: pd.DataFrame,
    feature_columns: Sequence[str],
    eval_years: Sequence[int],
    train_start: int,
) -> WalkForwardOutput:
    """Walk-forward residual collection.

    For each eval year Y in `eval_years`:
        1. Train rows = features ∩ weekly_stats inner-join on (gsis_id, season,
           week), filtered to season in [train_start, Y - 1] and position WR.
        2. Eval rows = same join filtered to season == Y, position WR.
        3. Fit 1 shared volume + 3 efficiency + 3 direct comparators on train rows.
        4. Predict per-row mu for both arms on eval rows.
        5. Append per-row tuples to per-stat residual buffers.
        6. Record per-year (volume, efficiency) factor residuals + coverage.

    Strict separation invariant: train rows for eval Y contain no row from
    season Y. Asserted at runtime for defense in depth.

    Args:
        features_by_year: Per-season WR feature DataFrames (already validated
            against WrFeaturesSchema). Each must have columns {gsis_id, season,
            week, *feature_columns}.
        weekly_stats: Canonical WeeklyStatsSchema-validated frame; will be
            filtered to position == WR internally.
        feature_columns: Ordered tuple of feature columns to use as X. Must
            match `_WR_FEATURE_COLUMNS` for an apples-to-apples comparison vs
            the production WR baseline. Boolean columns are coerced to int8.
        eval_years: Walk-forward eval years (e.g., (2021, 2022, 2023, 2024)).
        train_start: Earliest season included in any training window
            (inclusive lower bound).
    """
    weekly_stats_wr = weekly_stats[weekly_stats["position"] == "WR"].copy()

    per_stat_buffers: dict[Stat, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        stat: [] for stat in _WR_RECEIVING_DECOMPS
    }
    factor_residuals_by_year: dict[Stat, list[FactorResidualsByYear]] = {
        stat: [] for stat in _WR_RECEIVING_DECOMPS
    }
    eval_coverage: list[CoverageByYear] = []
    train_coverage: list[CoverageByYear] = []

    for eval_year in eval_years:
        # Build train + eval row sets via the same join recipe BaselineModel.fit uses.
        train_features = pd.concat(
            [
                features_by_year[s]
                for s in features_by_year
                if train_start <= s <= eval_year - 1
            ],
            ignore_index=True,
        )
        eval_features = features_by_year[eval_year]

        # Inner-join features ↔ weekly stats; drop NaN feature rows.
        truth_cols = [
            "gsis_id",
            "season",
            "week",
            Stat.TARGETS.value,
            Stat.RECEPTIONS.value,
            Stat.RECEIVING_YARDS.value,
            Stat.RECEIVING_TDS.value,
        ]
        train_joined = train_features.merge(
            weekly_stats_wr[truth_cols],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        eval_joined = eval_features.merge(
            weekly_stats_wr[truth_cols],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )

        # Strict separation defense in depth.
        if not train_joined.empty:
            assert int(train_joined["season"].max()) < eval_year, (
                f"Train rows for eval {eval_year} contain season >= eval year; "
                f"max train season is {int(train_joined['season'].max())}"
            )
        assert (eval_joined["season"] == eval_year).all(), (
            f"Eval rows for {eval_year} contain rows from other seasons"
        )

        # X matrix construction (mirrors BaselineModel._x_frame_with_bool_coercion).
        train_x_frame = train_joined[list(feature_columns)].copy()
        eval_x_frame = eval_joined[list(feature_columns)].copy()
        for col in train_x_frame.columns:
            if train_x_frame[col].dtype == bool:
                train_x_frame[col] = train_x_frame[col].astype(np.int8)
                eval_x_frame[col] = eval_x_frame[col].astype(np.int8)

        # Persist train medians for predict-time imputation; drop train NaN rows.
        train_medians = train_x_frame.median(skipna=True).astype(float)
        train_keep_mask = train_x_frame.notna().all(axis=1).to_numpy()
        train_x_frame = train_x_frame.loc[train_keep_mask]
        train_truth_keep = train_joined.loc[train_keep_mask]

        eval_x_frame = eval_x_frame.fillna(train_medians)
        x_train = train_x_frame.to_numpy(dtype=np.float64)
        x_eval = eval_x_frame.to_numpy(dtype=np.float64)

        # Targets and per-stat numerators (post-NaN-drop on train).
        targets_train = train_truth_keep[Stat.TARGETS.value].to_numpy(dtype=np.int64)
        targets_eval = eval_joined[Stat.TARGETS.value].to_numpy(dtype=np.int64)

        # Coverage: targets > 0 rates.
        train_coverage.append(
            CoverageByYear(
                eval_year=eval_year,
                targets_positive_rate=float((targets_train > 0).mean()) if len(targets_train) else 0.0,
                n_eval_rows=len(targets_train),
            )
        )
        eval_coverage.append(
            CoverageByYear(
                eval_year=eval_year,
                targets_positive_rate=float((targets_eval > 0).mean()) if len(targets_eval) else 0.0,
                n_eval_rows=len(targets_eval),
            )
        )

        # Fit shared volume sub-model.
        volume_ridge = _fit_decomposed_volume(x_train, targets_train)

        # Per-stat: fit efficiency + direct, predict on eval, record residuals.
        for stat, decomp in _WR_RECEIVING_DECOMPS.items():
            actual_eval = eval_joined[stat.value].to_numpy(dtype=np.float64)
            numerator_train = train_truth_keep[stat.value].to_numpy(dtype=np.int64)

            efficiency_ridge = _fit_decomposed_efficiency(
                x_train, numerator_train, targets_train
            )
            direct_ridge = _fit_direct(x_train, numerator_train.astype(np.float64))

            mu_decomposed = _predict_decomposed(
                volume_ridge=volume_ridge,
                efficiency_ridge=efficiency_ridge,
                x=x_eval,
                efficiency_clip_hi=decomp.efficiency_clip_hi,
            )
            mu_direct = _predict_direct(direct_ridge, x_eval)

            per_stat_buffers[stat].append((actual_eval, mu_direct, mu_decomposed))

            # Factor residuals on eval rows where targets > 0
            # (efficiency factor is undefined where targets == 0).
            mask_pos = targets_eval > 0
            volume_resid = (
                targets_eval[mask_pos].astype(np.float64)
                - volume_ridge.predict(x_eval[mask_pos]).astype(np.float64)
            )
            actual_ratio = (
                actual_eval[mask_pos]
                / targets_eval[mask_pos].astype(np.float64)
            )
            predicted_ratio = np.clip(
                efficiency_ridge.predict(x_eval[mask_pos]).astype(np.float64),
                0.0,
                decomp.efficiency_clip_hi,
            )
            efficiency_resid = actual_ratio - predicted_ratio
            factor_residuals_by_year[stat].append(
                FactorResidualsByYear(
                    eval_year=eval_year,
                    volume_residuals=volume_resid,
                    efficiency_residuals=efficiency_resid,
                )
            )

    per_stat_residuals = {
        stat: StatResiduals(
            actual=np.concatenate([b[0] for b in buffers]),
            mu_direct=np.concatenate([b[1] for b in buffers]),
            mu_decomposed=np.concatenate([b[2] for b in buffers]),
            n_paired=int(sum(len(b[0]) for b in buffers)),
        )
        for stat, buffers in per_stat_buffers.items()
    }

    return WalkForwardOutput(
        per_stat=per_stat_residuals,
        factor_residuals_by_year=factor_residuals_by_year,
        coverage_by_year=eval_coverage,
        train_coverage_by_year=train_coverage,
        eval_years=tuple(eval_years),
    )
```

- [ ] **Step 6: Run test, verify pass**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py -v`
Expected: 9 PASS.

- [ ] **Step 7: Write failing test for train/eval season separation**

```python
def test_walk_forward_train_eval_strict_separation_assertion() -> None:
    """Defense-in-depth: train rows for eval Y must not contain season Y rows.

    Construct synthetic data where features for season 2021 leak into the
    train pool for eval year 2021; verify the harness catches it.
    """
    features_by_year, weekly_stats = _make_synthetic_walk_forward_inputs(
        seasons=(2019, 2020, 2021)
    )
    # Inject a row tagged season=2021 into the 2020 features frame.
    leak_row = features_by_year[2020].iloc[[0]].copy()
    leak_row["season"] = 2021
    features_by_year[2020] = pd.concat(
        [features_by_year[2020], leak_row], ignore_index=True
    )
    # Add the corresponding weekly_stats row.
    leak_ws = weekly_stats[
        (weekly_stats["gsis_id"] == leak_row["gsis_id"].iloc[0])
        & (weekly_stats["season"] == 2020)
    ].iloc[[0]].copy()
    leak_ws["season"] = 2021
    weekly_stats = pd.concat([weekly_stats, leak_ws], ignore_index=True)

    with pytest.raises(AssertionError, match="contain season >= eval year"):
        walk_forward_residuals(
            features_by_year=features_by_year,
            weekly_stats=weekly_stats,
            feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
            eval_years=(2021,),
            train_start=2019,
        )
```

- [ ] **Step 8: Run, verify pass (assertion already implemented)**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py -v`
Expected: 10 PASS.

- [ ] **Step 9: Write failing test for coverage measurement**

```python
def test_walk_forward_coverage_measured_per_year() -> None:
    """Coverage = (targets > 0).mean() per eval year on eval rows; same on train rows.

    Construct synthetic data where 80% of 2020 eval rows have targets > 0;
    verify coverage_by_year[2020] reflects that.
    """
    rng = np.random.default_rng(42)
    n = 50
    seasons = (2018, 2019, 2020)
    features_by_year: dict[int, pd.DataFrame] = {}
    weekly_rows: list[dict[str, object]] = []
    for season in seasons:
        gsis_ids = [f"00-{season:04d}{i:03d}" for i in range(n)]
        feat_df = pd.DataFrame(
            {
                "gsis_id": gsis_ids,
                "season": season,
                "week": np.arange(1, n + 1) % 18 + 1,
                "team": "KC",
                "opponent": "DEN",
                "targets_per_game_l4": rng.standard_normal(n),
                "receiving_yards_per_game_l4": rng.standard_normal(n),
            }
        )
        features_by_year[season] = feat_df
        # 2020: exactly 40 rows with targets > 0 (80%); other seasons: ~95%.
        if season == 2020:
            targets = np.array([3] * 40 + [0] * 10)
        else:
            targets = rng.choice([0, 5], size=n, p=[0.05, 0.95])
        for i, gid in enumerate(gsis_ids):
            t = int(targets[i])
            weekly_rows.append(
                {
                    "gsis_id": gid,
                    "season": season,
                    "week": int(feat_df["week"].iloc[i]),
                    "position": "WR",
                    "team": "KC",
                    "opponent": "DEN",
                    "targets": t,
                    "receptions": int(t * 0.6),
                    "receiving_yards": float(t * 8.0),
                    "receiving_tds": int(t * 0.05),
                    "passing_yards": 0.0,
                    "passing_tds": 0,
                    "interceptions": 0,
                    "attempts": 0,
                    "completions": 0,
                    "sacks": 0,
                    "rushing_yards": 0.0,
                    "rushing_tds": 0,
                    "carries": 0,
                    "receiving_air_yards": 0.0,
                    "fumbles_lost": 0,
                }
            )
    weekly_stats = pd.DataFrame(weekly_rows)

    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
        eval_years=(2020,),
        train_start=2018,
    )
    cov = {c.eval_year: c for c in out.coverage_by_year}
    assert cov[2020].targets_positive_rate == pytest.approx(0.80, abs=0.01)
    assert cov[2020].n_eval_rows == n
```

- [ ] **Step 10: Run, verify pass (already implemented)**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py -v`
Expected: 11 PASS.

- [ ] **Step 11: Write failing test for factor-residual record per stat per year**

```python
def test_walk_forward_factor_residuals_recorded_per_stat_per_year() -> None:
    """For each (stat, eval_year), volume_residuals + efficiency_residuals
    are recorded on rows where targets > 0 (efficiency residual undefined
    where targets == 0).
    """
    features_by_year, weekly_stats = _make_synthetic_walk_forward_inputs(
        seasons=(2018, 2019, 2020)
    )
    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
        eval_years=(2019, 2020),
        train_start=2018,
    )
    for stat in (Stat.RECEPTIONS, Stat.RECEIVING_YARDS, Stat.RECEIVING_TDS):
        per_year = out.factor_residuals_by_year[stat]
        assert len(per_year) == 2  # one entry per eval year
        years = sorted(p.eval_year for p in per_year)
        assert years == [2019, 2020]
        for entry in per_year:
            assert entry.volume_residuals.shape == entry.efficiency_residuals.shape
            assert entry.volume_residuals.dtype == np.float64
```

- [ ] **Step 12: Run, verify pass (already implemented)**

Run: same. Expected: 12 PASS.

- [ ] **Step 13: Run mypy + ruff on the module**

Run: `PATH="$(pwd)/.venv/Scripts:$PATH" mypy src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py && ruff check src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py && ruff format --check src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py`
Expected: zero errors.

- [ ] **Step 14: Commit**

```bash
PATH="$(pwd)/.venv/Scripts:$PATH" git add src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py
PATH="$(pwd)/.venv/Scripts:$PATH" git commit -m "feat(probe-target-decomposition): walk-forward harness — Task 2

Walk-forward residual collection over eval_years × per-stat decomposed
recipes, with strict train/eval season separation assertion + per-year
coverage measurement (targets > 0 rate on both train and eval populations)
+ per-stat per-year (volume, efficiency) factor residuals for the §5 risk
#2 orthogonality-correlation diagnostic. Pure dataclass output; no IO.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Verdict mapping + report rendering

**Files:**
- Modify: `src/projections/backtest/target_decomposition_probe.py` (append).
- Modify: `tests/test_backtest/test_target_decomposition_probe.py` (append).

Verdict mapping per stat (matches `feature_probe.py:53` Phase 1 logic). Report rendering: per-stat markdown + summary csv. Includes the §5 risk #1 composite-fpts translation column and the §5 risk #2 factor residual Pearson correlation diagnostic.

- [ ] **Step 1: Write failing tests for verdict mapping**

```python
from projections.backtest.adoption_gate import BootstrapDelta
from projections.backtest.target_decomposition_probe import (
    ProbeReport,
    ProbeVerdictLabel,
    StatProbeVerdict,
    _verdict_from_delta,
    render_probe_report,
)


def _delta(point: float, lo_95: float, hi_95: float) -> BootstrapDelta:
    """Test helper. BootstrapDelta has 5 required fields; the verdict logic
    only reads point/lo_95/hi_95, so pin the others at constants."""
    return BootstrapDelta(
        point=point,
        lo_95=lo_95,
        hi_95=hi_95,
        n_paired_rows=100,
        n_bootstrap=200,
    )


def test_verdict_signal_when_hi_95_strictly_negative() -> None:
    assert _verdict_from_delta(_delta(-0.5, -0.8, -0.1)) == "SIGNAL"


def test_verdict_regression_when_lo_95_strictly_positive() -> None:
    assert _verdict_from_delta(_delta(0.5, 0.1, 0.8)) == "REGRESSION"


def test_verdict_null_when_ci_brackets_zero() -> None:
    assert _verdict_from_delta(_delta(-0.05, -0.3, 0.2)) == "NULL"


def test_verdict_null_at_exact_zero_boundaries() -> None:
    """SIGNAL requires hi_95 strictly < 0; lo_95 strictly > 0 for REGRESSION.
    Exact-zero boundaries fall into NULL."""
    assert _verdict_from_delta(_delta(-0.5, -1.0, 0.0)) == "NULL"
    assert _verdict_from_delta(_delta(0.5, 0.0, 1.0)) == "NULL"
```

- [ ] **Step 2: Run, expect failure**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py -v`
Expected: 4 FAIL on the new tests; 12 PASS prior.

- [ ] **Step 3: Implement verdict mapping + dataclasses**

Append to `src/projections/backtest/target_decomposition_probe.py`:

```python
from typing import Literal

from projections.backtest.adoption_gate import (
    BootstrapDelta,
    paired_bootstrap_rmse_delta,
)


ProbeVerdictLabel = Literal["SIGNAL", "NULL", "REGRESSION"]


def _verdict_from_delta(delta: BootstrapDelta) -> ProbeVerdictLabel:
    """Pure CI-based per-stat verdict (no effect-size floor).

    SIGNAL iff hi_95 < 0 (decomposed strictly improves).
    REGRESSION iff lo_95 > 0 (decomposed strictly regresses).
    NULL otherwise.
    """
    if delta.hi_95 < 0:
        return "SIGNAL"
    if delta.lo_95 > 0:
        return "REGRESSION"
    return "NULL"


@dataclass(frozen=True, slots=True)
class StatProbeVerdict:
    """Per-stat probe verdict + diagnostic numbers."""

    stat: Stat
    n_paired: int
    rmse_delta: BootstrapDelta
    rmse_direct: float
    rmse_decomposed: float
    verdict: ProbeVerdictLabel
    expected_composite_fpts_delta: float
    """RMSE delta translated to expected composite-fpts contribution
    (rmse_delta.point × scoring coefficient for this stat under ESPN PPR).
    Per §5 risk #1: surfaces probe-vs-gate magnitude calibration."""
    factor_residual_correlation_by_year: Mapping[int, float]
    """Per-eval-year Pearson ρ between volume residual and efficiency residual.
    |ρ| > 0.2 in any year is a documented caveat per §5 risk #2."""


@dataclass(frozen=True, slots=True)
class ProbeReport:
    """Bundle of all per-stat verdicts + the walk-forward output for context.

    Renders to:
    - 1 summary markdown (hand-written from this struct; see Task 5).
    - 1 per-stat csv.
    - 3 per-stat markdown details.
    """

    verdicts: Mapping[Stat, StatProbeVerdict]
    walk_forward: WalkForwardOutput
    bootstrap_n: int
    seed: int
```

- [ ] **Step 4: Run, verify pass**

Run: same. Expected: 16 PASS.

- [ ] **Step 5: Write failing test for `render_probe_report` end-to-end**

```python
def test_render_probe_report_returns_three_stat_verdicts() -> None:
    """End-to-end: walk_forward_residuals → render_probe_report yields 3 verdicts,
    each with finite n_paired and RMSE deltas."""
    features_by_year, weekly_stats = _make_synthetic_walk_forward_inputs()
    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
        eval_years=(2021,),
        train_start=2018,
    )
    report = render_probe_report(out, bootstrap_n=200, seed=42)
    assert isinstance(report, ProbeReport)
    assert set(report.verdicts.keys()) == {
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    }
    for stat, v in report.verdicts.items():
        assert v.stat is stat
        assert v.n_paired > 0
        assert np.isfinite(v.rmse_delta.point)
        assert np.isfinite(v.rmse_delta.lo_95)
        assert np.isfinite(v.rmse_delta.hi_95)
        assert v.verdict in ("SIGNAL", "NULL", "REGRESSION")


def test_render_probe_report_is_deterministic_under_fixed_seed() -> None:
    features_by_year, weekly_stats = _make_synthetic_walk_forward_inputs()
    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
        eval_years=(2021,),
        train_start=2018,
    )
    a = render_probe_report(out, bootstrap_n=200, seed=123)
    b = render_probe_report(out, bootstrap_n=200, seed=123)
    for stat in (Stat.RECEPTIONS, Stat.RECEIVING_YARDS, Stat.RECEIVING_TDS):
        assert a.verdicts[stat].rmse_delta.point == b.verdicts[stat].rmse_delta.point
        assert a.verdicts[stat].rmse_delta.lo_95 == b.verdicts[stat].rmse_delta.lo_95
        assert a.verdicts[stat].rmse_delta.hi_95 == b.verdicts[stat].rmse_delta.hi_95
```

- [ ] **Step 6: Run, expect failure**

Run: same. Expected: 2 FAIL; 16 PASS.

- [ ] **Step 7: Implement `render_probe_report`**

Append:

```python
from projections.schemas import Ruleset


def _rmse(actual: np.ndarray, mu: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - mu) ** 2)))


# Cached coefficient lookup. We import lazily to avoid an import cycle (scoring
# imports schemas which is also imported by this module — direct import is fine,
# but we centralize the coefficient table here for clarity).
def _stat_scoring_coefficient(stat: Stat) -> float:
    """ESPN PPR coefficient for `stat`. Mirrors
    `projections.scoring.score_distribution._scoring_coefficients`.

    Used to translate per-stat RMSE delta to expected composite-fpts impact:
    a per-stat RMSE delta of -0.1 yds with coefficient 0.1 fpt/yd implies
    the stat contributes roughly -0.01 fpts to composite-fpts RMSE (rough,
    not exact — composite-fpts RMSE depends on cross-stat covariance).
    """
    ruleset = Ruleset.espn_ppr()
    coefficients = {
        Stat.RECEPTIONS: ruleset.reception_pts,
        Stat.RECEIVING_YARDS: 1.0 / ruleset.receiving_yds_per_pt,
        Stat.RECEIVING_TDS: ruleset.receiving_td_pts,
    }
    return coefficients[stat]


def _factor_residual_correlation(
    factor_residuals: Sequence[FactorResidualsByYear],
) -> dict[int, float]:
    """Per-eval-year Pearson ρ of (volume_residual, efficiency_residual)."""
    out: dict[int, float] = {}
    for entry in factor_residuals:
        if len(entry.volume_residuals) < 2:
            out[entry.eval_year] = float("nan")
            continue
        vol = entry.volume_residuals
        eff = entry.efficiency_residuals
        std_v = vol.std(ddof=0)
        std_e = eff.std(ddof=0)
        if std_v == 0 or std_e == 0:
            out[entry.eval_year] = float("nan")
            continue
        rho = float(np.cov(vol, eff, ddof=0)[0, 1] / (std_v * std_e))
        out[entry.eval_year] = rho
    return out


def render_probe_report(
    walk_forward: WalkForwardOutput,
    *,
    bootstrap_n: int = 5000,
    seed: int = 0xD3C0,
) -> ProbeReport:
    """Per-stat verdicts via paired-bootstrap on the pooled residuals.

    Args:
        walk_forward: output from `walk_forward_residuals`.
        bootstrap_n: paired-bootstrap resample count (default 5000, matches
            `feature_probe.py`).
        seed: RNG seed for the bootstrap; deterministic across runs.
    """
    verdicts: dict[Stat, StatProbeVerdict] = {}
    for stat, residuals in walk_forward.per_stat.items():
        # paired_bootstrap_rmse_delta consumes residual arrays (actual - mu);
        # compute them up front from the per-row buffers.
        residuals_direct = residuals.actual - residuals.mu_direct
        residuals_decomposed = residuals.actual - residuals.mu_decomposed
        # Sign convention: candidate - incumbent. We name "decomposed" the
        # candidate (we want it to win), so direct is incumbent.
        delta = paired_bootstrap_rmse_delta(
            residuals_incumbent=residuals_direct,
            residuals_candidate=residuals_decomposed,
            n_bootstrap=bootstrap_n,
            seed=seed,
        )
        rmse_direct = _rmse(residuals.actual, residuals.mu_direct)
        rmse_decomposed = _rmse(residuals.actual, residuals.mu_decomposed)
        coef = _stat_scoring_coefficient(stat)
        per_year_rho = _factor_residual_correlation(
            walk_forward.factor_residuals_by_year[stat]
        )
        verdicts[stat] = StatProbeVerdict(
            stat=stat,
            n_paired=residuals.n_paired,
            rmse_delta=delta,
            rmse_direct=rmse_direct,
            rmse_decomposed=rmse_decomposed,
            verdict=_verdict_from_delta(delta),
            expected_composite_fpts_delta=delta.point * coef,
            factor_residual_correlation_by_year=per_year_rho,
        )

    return ProbeReport(
        verdicts=verdicts,
        walk_forward=walk_forward,
        bootstrap_n=bootstrap_n,
        seed=seed,
    )
```

- [ ] **Step 8: Run tests, verify pass**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py -v`
Expected: 18 PASS.

- [ ] **Step 9: Write failing tests for csv + per-stat markdown rendering**

```python
def test_per_stat_csv_has_one_row_per_decomposed_stat(tmp_path) -> None:
    """Per-stat CSV: one row per decomposed stat with verdict columns."""
    features_by_year, weekly_stats = _make_synthetic_walk_forward_inputs()
    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
        eval_years=(2021,),
        train_start=2018,
    )
    report = render_probe_report(out, bootstrap_n=200, seed=42)
    csv_path = tmp_path / "per_stat.csv"
    write_per_stat_csv(report, csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert set(df["stat"]) == {"receptions", "receiving_yards", "receiving_tds"}
    for col in (
        "n_paired",
        "rmse_direct",
        "rmse_decomposed",
        "rmse_delta_point",
        "rmse_delta_lo_95",
        "rmse_delta_hi_95",
        "verdict",
        "expected_composite_fpts_delta",
    ):
        assert col in df.columns


def test_per_stat_markdown_renders_verdict_and_diagnostics(tmp_path) -> None:
    """Per-stat markdown: verdict header + RMSE table + per-year coverage table
    + per-year factor-residual ρ."""
    features_by_year, weekly_stats = _make_synthetic_walk_forward_inputs()
    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=("targets_per_game_l4", "receiving_yards_per_game_l4"),
        eval_years=(2021,),
        train_start=2018,
    )
    report = render_probe_report(out, bootstrap_n=200, seed=42)
    md_path = tmp_path / "receiving_yards.md"
    write_per_stat_markdown(report, Stat.RECEIVING_YARDS, md_path)
    text = md_path.read_text()
    assert "receiving_yards" in text
    assert report.verdicts[Stat.RECEIVING_YARDS].verdict in text
    # Composite-fpts translation must surface in the body.
    assert "Expected composite-fpts" in text or "expected_composite_fpts" in text
    # Factor residual correlation surfaces.
    assert "Factor residual" in text or "ρ" in text
```

- [ ] **Step 10: Run, expect failure**

Run: same. Expected: 2 FAIL.

- [ ] **Step 11: Implement `write_per_stat_csv` and `write_per_stat_markdown`**

Append:

```python
from pathlib import Path


def write_per_stat_csv(report: ProbeReport, path: Path) -> None:
    """Write a 3-row CSV: one row per decomposed stat with verdict + diagnostics."""
    rows: list[dict[str, object]] = []
    for stat, v in report.verdicts.items():
        rows.append(
            {
                "stat": stat.value,
                "n_paired": v.n_paired,
                "rmse_direct": v.rmse_direct,
                "rmse_decomposed": v.rmse_decomposed,
                "rmse_delta_point": v.rmse_delta.point,
                "rmse_delta_lo_95": v.rmse_delta.lo_95,
                "rmse_delta_hi_95": v.rmse_delta.hi_95,
                "verdict": v.verdict,
                "expected_composite_fpts_delta": v.expected_composite_fpts_delta,
            }
        )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_per_stat_markdown(
    report: ProbeReport, stat: Stat, path: Path
) -> None:
    """Render the per-stat detail markdown."""
    v = report.verdicts[stat]
    coverage = {c.eval_year: c for c in report.walk_forward.coverage_by_year}
    train_coverage = {c.eval_year: c for c in report.walk_forward.train_coverage_by_year}
    eval_years = report.walk_forward.eval_years

    lines: list[str] = []
    lines.append(f"# Target Decomposition Probe — {stat.value}")
    lines.append("")
    lines.append(f"**Verdict:** {v.verdict}")
    lines.append("")
    lines.append("## Pooled per-stat verdict")
    lines.append("")
    lines.append("| n_paired | RMSE direct | RMSE decomposed | Δ-RMSE | 95% CI | Verdict |")
    lines.append("|---:|---:|---:|---:|---|:---:|")
    lines.append(
        f"| {v.n_paired} | {v.rmse_direct:.4f} | {v.rmse_decomposed:.4f} | "
        f"{v.rmse_delta.point:+.4f} | "
        f"[{v.rmse_delta.lo_95:+.4f}, {v.rmse_delta.hi_95:+.4f}] | "
        f"**{v.verdict}** |"
    )
    lines.append("")
    lines.append(
        f"**Expected composite-fpts Δ (rough)**: "
        f"{v.expected_composite_fpts_delta:+.4f} fpts "
        f"(stat RMSE Δ × ESPN PPR coefficient {_stat_scoring_coefficient(stat):+.4f}). "
        f"Per §5 risk #1, magnitudes < 0.005 fpts under coverage relaxation should "
        f"be treated as MARGINAL, not SIGNAL."
    )
    lines.append("")
    lines.append("## Per-eval-year coverage")
    lines.append("")
    lines.append("| Year | Eval n | Eval (targets > 0) | Train n | Train (targets > 0) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for year in eval_years:
        e = coverage[year]
        t = train_coverage[year]
        lines.append(
            f"| {year} | {e.n_eval_rows} | {e.targets_positive_rate:.3f} | "
            f"{t.n_eval_rows} | {t.targets_positive_rate:.3f} |"
        )
    lines.append("")
    lines.append("## Factor residual correlation (Pearson ρ)")
    lines.append("")
    lines.append(
        "Per-eval-year Pearson ρ between (predicted-volume residual, "
        "predicted-efficiency residual) on rows with targets > 0. |ρ| > 0.2 in "
        "any year is a documented caveat per §5 risk #2."
    )
    lines.append("")
    lines.append("| Year | ρ |")
    lines.append("|---:|---:|")
    for year in eval_years:
        rho = v.factor_residual_correlation_by_year.get(year, float("nan"))
        lines.append(f"| {year} | {rho:+.3f} |")
    lines.append("")
    lines.append(
        f"_Bootstrap n_resamples = {report.bootstrap_n}, seed = {report.seed}._"
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
```

- [ ] **Step 12: Run tests, verify pass**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py -v`
Expected: 20 PASS.

- [ ] **Step 13: Run mypy + ruff**

Run: `PATH="$(pwd)/.venv/Scripts:$PATH" mypy src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py && ruff check src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py && ruff format --check src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py`
Expected: zero errors.

- [ ] **Step 14: Commit**

```bash
PATH="$(pwd)/.venv/Scripts:$PATH" git add src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py
PATH="$(pwd)/.venv/Scripts:$PATH" git commit -m "feat(probe-target-decomposition): verdict mapping + report rendering — Task 3

Per-stat verdict mapping (SIGNAL/NULL/REGRESSION, pure CI-based; matches
feature_probe.py:53). render_probe_report runs paired_bootstrap_rmse_delta
on pooled residuals and packages StatProbeVerdict per stat with composite-fpts
translation (§5 risk #1) + factor residual Pearson ρ per eval year (§5 risk
#2). write_per_stat_csv + write_per_stat_markdown render the canonical
report layout.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: CLI script

**Files:**
- Create: `scripts/probe_target_decomposition.py`
- Create: `tests/test_scripts/test_probe_target_decomposition_cli.py`

argparse-based CLI: load WR feature cache + weekly stats from canonical paths, dispatch the harness, render outputs to `--output-dir`. Most loading logic mirrors `scripts/refresh_features.py` and `BaselineModel.fit`'s join recipe.

- [ ] **Step 1: Write the failing CLI argparse test**

```python
# tests/test_scripts/test_probe_target_decomposition_cli.py
"""Unit tests for scripts/probe_target_decomposition.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.probe_target_decomposition import _build_arg_parser, _parse_args


def test_arg_parser_defaults_match_spec() -> None:
    """Defaults: --eval-years 2021 2022 2023 2024 --train-start 2018
    --bootstrap-n 5000 --coverage-threshold 0.95."""
    parser = _build_arg_parser()
    args = parser.parse_args(["--output-dir", "reports/probe"])
    assert args.output_dir == Path("reports/probe")
    assert args.eval_years == [2021, 2022, 2023, 2024]
    assert args.train_start == 2018
    assert args.bootstrap_n == 5000
    assert args.coverage_threshold == 0.95
    assert args.seed == 0xD3C0


def test_arg_parser_overrides() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            "out",
            "--eval-years",
            "2022",
            "2023",
            "--train-start",
            "2019",
            "--bootstrap-n",
            "1000",
            "--coverage-threshold",
            "0.90",
            "--seed",
            "42",
        ]
    )
    assert args.eval_years == [2022, 2023]
    assert args.train_start == 2019
    assert args.bootstrap_n == 1000
    assert args.coverage_threshold == 0.90
    assert args.seed == 42
```

- [ ] **Step 2: Run, expect failure**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_scripts/test_probe_target_decomposition_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.probe_target_decomposition'`.

- [ ] **Step 3: Implement `_build_arg_parser` and `_parse_args` in the CLI module**

```python
# scripts/probe_target_decomposition.py
"""WR receiving-stats target decomposition probe — CLI.

Loads WR feature cache (data/features/wr/season=YYYY/week=WW/part.parquet)
and weekly stats (data/raw/weekly_stats/season=YYYY/part.parquet); runs the
walk-forward harness in `projections.backtest.target_decomposition_probe`;
renders per-stat markdown + CSV reports into --output-dir.

Spec: docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md
Plan: docs/superpowers/plans/2026-05-10-target-decomposition-probe.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from projections.backtest.target_decomposition_probe import (
    render_probe_report,
    walk_forward_residuals,
    write_per_stat_csv,
    write_per_stat_markdown,
)
from projections.features.cache import read_features
from projections.models.baseline import _WR_FEATURE_COLUMNS
from projections.schemas import Position, Stat, WeeklyStatsSchema
from projections.store import read_partition

_LOG = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the rendered reports (created if missing).",
    )
    p.add_argument(
        "--eval-years",
        type=int,
        nargs="+",
        default=[2021, 2022, 2023, 2024],
        help="Walk-forward eval years (default: 2021 2022 2023 2024).",
    )
    p.add_argument(
        "--train-start",
        type=int,
        default=2018,
        help="Inclusive lower bound for any training window (default: 2018).",
    )
    p.add_argument(
        "--bootstrap-n",
        type=int,
        default=5000,
        help="Paired-bootstrap resample count (default: 5000).",
    )
    p.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.95,
        help="Per-eval-year and per-train-window targets > 0 rate floor "
        "(default: 0.95). Relaxation must be documented in the report.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0xD3C0,
        help="Bootstrap seed (default: 0xD3C0). Pin to reproduce.",
    )
    p.add_argument(
        "--features-root",
        type=Path,
        default=Path("data/features"),
        help="Feature cache root (default: data/features).",
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="Raw partition root (default: data/raw). Weekly stats live at "
        "<raw-root>/weekly_stats/season=YYYY/part.parquet — read via "
        "projections.store.read_partition(raw_root, 'weekly_stats', season=Y).",
    )
    return p


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_arg_parser().parse_args(argv)
```

- [ ] **Step 4: Run, verify pass**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_scripts/test_probe_target_decomposition_cli.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Write failing test for the CLI dispatch (loader → harness → outputs)**

```python
def test_main_writes_expected_outputs(tmp_path, monkeypatch):
    """End-to-end: monkey-patch the loader to return synthetic data; verify
    main() writes the 4 expected output files to --output-dir."""
    import numpy as np
    from projections.schemas import Stat
    from scripts.probe_target_decomposition import main

    # Synthetic loader returns (features_by_year, weekly_stats)
    rng = np.random.default_rng(7)
    seasons = (2018, 2019, 2020, 2021)
    n = 30
    features_by_year = {}
    weekly_rows = []
    feature_cols = ["targets_per_game_l4", "receiving_yards_per_game_l4"]
    for season in seasons:
        ids = [f"00-{season:04d}{i:03d}" for i in range(n)]
        features_by_year[season] = pd.DataFrame(
            {
                "gsis_id": ids,
                "season": season,
                "week": rng.integers(1, 18, size=n),
                "team": "KC",
                "opponent": "DEN",
                **{c: rng.standard_normal(n) for c in feature_cols},
            }
        )
        for i, gid in enumerate(ids):
            t = int(rng.integers(0, 10))
            weekly_rows.append(
                {
                    "gsis_id": gid,
                    "season": season,
                    "week": int(features_by_year[season]["week"].iloc[i]),
                    "position": "WR",
                    "team": "KC",
                    "opponent": "DEN",
                    "targets": t,
                    "receptions": int(t * 0.6),
                    "receiving_yards": float(t * 8.0),
                    "receiving_tds": int(t * 0.05),
                    "passing_yards": 0.0,
                    "passing_tds": 0,
                    "interceptions": 0,
                    "attempts": 0,
                    "completions": 0,
                    "sacks": 0,
                    "rushing_yards": 0.0,
                    "rushing_tds": 0,
                    "carries": 0,
                    "receiving_air_yards": 0.0,
                    "fumbles_lost": 0,
                }
            )
    weekly_stats = pd.DataFrame(weekly_rows)

    def fake_load(args):
        return features_by_year, weekly_stats, list(feature_cols)

    monkeypatch.setattr(
        "scripts.probe_target_decomposition._load_inputs", fake_load
    )

    out_dir = tmp_path / "reports"
    rc = main(
        [
            "--output-dir",
            str(out_dir),
            "--eval-years",
            "2021",
            "--train-start",
            "2018",
            "--bootstrap-n",
            "200",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    assert (out_dir / "feature_probe_target_decomposition_per_stat.csv").exists()
    for stat_name in ("receptions", "receiving_yards", "receiving_tds"):
        md = out_dir / f"feature_probe_target_decomposition_{stat_name}.md"
        assert md.exists()
        text = md.read_text()
        assert stat_name in text
```

- [ ] **Step 6: Run, expect failure**

Run: same. Expected: FAIL on `AttributeError` or `ImportError`.

- [ ] **Step 7: Implement `_load_inputs` and `main`**

Append to `scripts/probe_target_decomposition.py`:

```python
def _load_inputs(
    args: argparse.Namespace,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, list[str]]:
    """Load WR features + weekly stats from canonical paths.

    Returns (features_by_year, weekly_stats, feature_columns).
    """
    needed_seasons = list(range(args.train_start, max(args.eval_years) + 1))
    features_by_year: dict[int, pd.DataFrame] = {}
    for season in needed_seasons:
        features_by_year[season] = read_features(
            position=Position.WR,
            season=season,
            features_root=args.features_root,
        )

    # Weekly stats: load all needed seasons via store.read_partition.
    # Pattern matches scripts/predict_2024.py:83 — read_partition(raw_root,
    # "weekly_stats", season=Y) returns the canonical WeeklyStatsSchema frame.
    weekly_frames = [
        read_partition(args.raw_root, "weekly_stats", season=s)
        for s in needed_seasons
    ]
    weekly_stats = pd.concat(weekly_frames, ignore_index=True)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)

    return features_by_year, weekly_stats, list(_WR_FEATURE_COLUMNS)


def _enforce_coverage(
    walk_forward_output, threshold: float
) -> list[str]:
    """Return a list of human-readable warnings for any per-eval-year
    coverage below `threshold`. Empty list if all clear."""
    warnings: list[str] = []
    for c in walk_forward_output.coverage_by_year:
        if c.targets_positive_rate < threshold:
            warnings.append(
                f"Eval year {c.eval_year}: targets > 0 rate "
                f"{c.targets_positive_rate:.3f} below threshold {threshold:.2f}"
            )
    for c in walk_forward_output.train_coverage_by_year:
        if c.targets_positive_rate < threshold:
            warnings.append(
                f"Train window for eval {c.eval_year}: targets > 0 rate "
                f"{c.targets_positive_rate:.3f} below threshold {threshold:.2f}"
            )
    return warnings


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    features_by_year, weekly_stats, feature_columns = _load_inputs(args)

    out = walk_forward_residuals(
        features_by_year=features_by_year,
        weekly_stats=weekly_stats,
        feature_columns=feature_columns,
        eval_years=tuple(args.eval_years),
        train_start=args.train_start,
    )

    coverage_warnings = _enforce_coverage(out, args.coverage_threshold)
    if coverage_warnings:
        _LOG.warning(
            "Coverage threshold %.2f not met:\n  %s",
            args.coverage_threshold,
            "\n  ".join(coverage_warnings),
        )
        _LOG.warning("(Continuing per the PR #31 retrospective rule — relaxation "
                     "must be documented in the summary report.)")

    report = render_probe_report(
        out, bootstrap_n=args.bootstrap_n, seed=args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "feature_probe_target_decomposition_per_stat.csv"
    write_per_stat_csv(report, csv_path)
    for stat in (Stat.RECEPTIONS, Stat.RECEIVING_YARDS, Stat.RECEIVING_TDS):
        md_path = (
            args.output_dir
            / f"feature_probe_target_decomposition_{stat.value}.md"
        )
        write_per_stat_markdown(report, stat, md_path)

    _LOG.info("Wrote outputs to %s", args.output_dir)
    for stat, v in report.verdicts.items():
        _LOG.info(
            "  %s: %s (Δ-RMSE %+.4f, CI [%+.4f, %+.4f], n=%d)",
            stat.value,
            v.verdict,
            v.rmse_delta.point,
            v.rmse_delta.lo_95,
            v.rmse_delta.hi_95,
            v.n_paired,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8: Run, verify pass**

Run: `PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_scripts/test_probe_target_decomposition_cli.py -v`
Expected: 3 PASS.

- [ ] **Step 9: Run all probe-related tests + mypy + ruff**

Run:
```bash
PYTHONPATH=src .venv/Scripts/python -m pytest tests/test_backtest/test_target_decomposition_probe.py tests/test_scripts/test_probe_target_decomposition_cli.py -v
PATH="$(pwd)/.venv/Scripts:$PATH" mypy src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py scripts/probe_target_decomposition.py tests/test_scripts/test_probe_target_decomposition_cli.py
PATH="$(pwd)/.venv/Scripts:$PATH" ruff check src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py scripts/probe_target_decomposition.py tests/test_scripts/test_probe_target_decomposition_cli.py
PATH="$(pwd)/.venv/Scripts:$PATH" ruff format --check src/projections/backtest/target_decomposition_probe.py tests/test_backtest/test_target_decomposition_probe.py scripts/probe_target_decomposition.py tests/test_scripts/test_probe_target_decomposition_cli.py
```
Expected: 23 PASS, mypy clean, ruff clean.

- [ ] **Step 10: Commit**

```bash
PATH="$(pwd)/.venv/Scripts:$PATH" git add scripts/probe_target_decomposition.py tests/test_scripts/test_probe_target_decomposition_cli.py
PATH="$(pwd)/.venv/Scripts:$PATH" git commit -m "feat(probe-target-decomposition): CLI script — Task 4

argparse CLI loads WR feature cache + weekly stats via canonical paths,
dispatches the walk-forward harness, renders per-stat CSV + 3 per-stat
markdown reports to --output-dir. Coverage warnings emitted when threshold
not met (defaults to 0.95 per spec §1.3.1; relaxation must be documented
in the summary report per the PR #31 retrospective rule). _WR_FEATURE_COLUMNS
imported from baseline.py at run time so feature drift fails fast at the
schema-validate step.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Real-data probe run + summary writeup

**Files:**
- Create: `reports/feature_probe_target_decomposition_per_stat.csv` (CLI output, committed).
- Create: `reports/feature_probe_target_decomposition_{receptions,receiving_yards,receiving_tds}.md` (CLI outputs, committed).
- Create: `reports/feature_probe_target_decomposition_summary.md` (hand-written, committed).

This task is the only one that requires real data (the WR feature cache + weekly stats). Verify the cache exists, run the CLI, inspect outputs, write the summary report.

- [ ] **Step 1: Verify the WR feature cache + weekly_stats are present locally**

```bash
ls data/features/wr/season=2018/ data/features/wr/season=2024/ 2>/dev/null
ls data/raw/weekly_stats/season=2018/ data/raw/weekly_stats/season=2024/ 2>/dev/null
```
Expected: directories exist with `part.parquet` files.

If missing, refresh:
```bash
PYTHONPATH=src .venv/Scripts/python scripts/refresh_features.py wr --seasons 2018 2019 2020 2021 2022 2023 2024
PYTHONPATH=src .venv/Scripts/python -c "
from pathlib import Path
from projections.ingest.refresh import refresh
refresh(data_root=Path('data'), seasons=range(2018, 2025), tables=['weekly_stats'])
"
```

- [ ] **Step 2: Run the CLI with default flags**

```bash
PYTHONPATH=src .venv/Scripts/python scripts/probe_target_decomposition.py \
  --output-dir reports/ \
  --coverage-threshold 0.95 \
  --bootstrap-n 5000 \
  --eval-years 2021 2022 2023 2024 \
  --train-start 2018 \
  --seed 54208
```

Expected stdout: 3 lines logging per-stat verdict + Δ-RMSE + CI + n_paired. If coverage threshold isn't met, a warning logs the affected (year, rate) pairs.

Capture the stdout — it goes into the summary report.

- [ ] **Step 3: Inspect the 3 per-stat markdowns + the CSV**

```bash
cat reports/feature_probe_target_decomposition_per_stat.csv
cat reports/feature_probe_target_decomposition_receptions.md
cat reports/feature_probe_target_decomposition_receiving_yards.md
cat reports/feature_probe_target_decomposition_receiving_tds.md
```
Expected: each markdown has the verdict header, pooled-verdict table, per-eval-year coverage table, factor residual ρ table.

- [ ] **Step 4: Write the summary report by hand**

Use this template, filling in the verdict + numbers from the CSV. Save as `reports/feature_probe_target_decomposition_summary.md`:

```markdown
# WR Receiving Stats Target Decomposition Probe — Summary

**Date:** 2026-05-10
**Branch:** `feat/probe-target-decomposition`
**Spec:** `docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md`
**Plan:** `docs/superpowers/plans/2026-05-10-target-decomposition-probe.md`
**Commit:** [fill in after final commit]

## Verdict

[Family verdict: SIGNAL / NULL / REGRESSION / mixed. Greenlights / closes / branches accordingly per spec §4.]

## Per-stat verdicts (pooled across 2021–2024)

| Stat | n_paired | RMSE direct | RMSE decomposed | Δ-RMSE | 95% CI | Verdict | Expected composite-fpts Δ |
|---|---:|---:|---:|---:|---|:---:|---:|
| receptions | [n] | [direct] | [decomposed] | [delta] | [lo, hi] | [SIGNAL/NULL/REGRESSION] | [delta * 1.0] |
| receiving_yards | [n] | [direct] | [decomposed] | [delta] | [lo, hi] | [verdict] | [delta * 0.1] |
| receiving_tds | [n] | [direct] | [decomposed] | [delta] | [lo, hi] | [verdict] | [delta * 6.0] |

## Coverage cross-check

| Eval year | Eval n | Eval (targets > 0) | Train n | Train (targets > 0) |
|---:|---:|---:|---:|---:|
| 2021 | [n] | [rate] | [n] | [rate] |
| 2022 | [n] | [rate] | [n] | [rate] |
| 2023 | [n] | [rate] | [n] | [rate] |
| 2024 | [n] | [rate] | [n] | [rate] |

[State whether coverage threshold (0.95 default) was met; if relaxed, note the per-(year, population) figures and per the PR #31 retrospective rule, treat any SIGNAL with magnitude < 0.005 fpts as MARGINAL not SIGNAL.]

## Factor residual correlation (Pearson ρ per eval year)

[For each stat, summarize the per-eval-year ρ. |ρ| > 0.2 in any year is a documented caveat per §5 risk #2.]

| Stat | 2021 | 2022 | 2023 | 2024 |
|---|---:|---:|---:|---:|
| receptions | [ρ] | [ρ] | [ρ] | [ρ] |
| receiving_yards | [ρ] | [ρ] | [ρ] | [ρ] |
| receiving_tds | [ρ] | [ρ] | [ρ] | [ρ] |

## Decision log

[1-2 paragraphs: which spec §4 branch fires, what closes vs greenlights, whether refined-unit candidates are queued, what the integration plan must address (per §7 named follow-ups: DecomposedBaselineModel peer, ProductDistribution + coherent within-row sampling, factor-appropriate sub-models as a separate probe + integration cycle).]

## Recurring "QB augment regression" check

Not applicable — this probe has no augment / swap modes; it tests a model architecture change on WR only. The recurring QB augment regression pattern documented across PRs #23 / #24 / #25 / #28 is feature-additions-on-QB-specific and does not apply here.

## Spec gaps caught + fixed during execution

[Any deviations from spec discovered during implementation. If none, write "None — spec executed clean."]
```

- [ ] **Step 5: Run the full verification gate suite**

```bash
PYTHONPATH=src .venv/Scripts/python -m pytest -v
PATH="$(pwd)/.venv/Scripts:$PATH" mypy src tests
PATH="$(pwd)/.venv/Scripts:$PATH" ruff check src tests scripts
PATH="$(pwd)/.venv/Scripts:$PATH" ruff format --check src tests scripts
```
Expected: full test suite green, mypy clean, ruff clean. Note any pre-existing slow lightgbm/ensemble tests that already skip (PR #29 / #31 documented these as out-of-scope for narrowly-scoped work).

- [ ] **Step 6: Commit reports**

```bash
PATH="$(pwd)/.venv/Scripts:$PATH" git add reports/feature_probe_target_decomposition_*.md reports/feature_probe_target_decomposition_*.csv
PATH="$(pwd)/.venv/Scripts:$PATH" git commit -m "report(probe-target-decomposition): probe verdict + per-stat detail + summary — Task 5

Real-data probe run on WR feature cache + weekly stats. Per-stat verdicts:
[fill in: receptions=X, receiving_yards=Y, receiving_tds=Z]. Family verdict:
[SIGNAL/NULL/MIXED]. [One-line decision narrative.]

Verification gates: pytest [n] passed, mypy clean, ruff clean,
ruff format clean. Wall-clock end-to-end [seconds].

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: PM/TODO update + final verification

**Files:**
- Modify: `project_management.md` (prepend a new entry).
- Modify: `TODO.md` (append an Update under #23).

- [ ] **Step 1: Read the current top of `project_management.md` to confirm the header pattern**

```bash
head -10 project_management.md
```
Expected: lines 1–5 are the file's preamble; line 7 begins the most recent decision entry (PR #31 weather refined-unit RB+WR integration verdict).

- [ ] **Step 2: Prepend the new PM entry**

Edit `project_management.md` to insert a new section *between* line 5 (`---`) and line 7 (the prior entry's H2). The new section template:

```markdown
## WR Receiving Stats Target Decomposition Probe — verdict [SIGNAL/NULL/MIXED] (2026-05-10, on branch `feat/probe-target-decomposition`)

**Status:** Probe-only spec shipped per `docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md` and plan `docs/superpowers/plans/2026-05-10-target-decomposition-probe.md`. **First model-architecture probe in the project** (Track 2A's prior probes have all measured feature additions via override parquets; this probe measures a prediction-recipe change with no override layer). New module `src/projections/backtest/target_decomposition_probe.py` walks forward over 2021–2024, fitting one shared volume RidgeCV (on `targets`) plus three efficiency RidgeCVs (on `catch_rate`, `yards_per_target`, `td_rate_per_target` filtered to `targets > 0`) plus three direct-comparator RidgeCVs (matching `BaselineModel.fit`). Sub-model class deliberately Ridge-only across both arms so any SIGNAL is attributable to *decomposition itself*, not to a model-class change. No new ingest, no schema changes, no production builders touched, no model factory added.

**Per-stat verdicts (pooled 2021–2024, paired-bootstrap CI on RMSE delta):**

| Stat | n_paired | RMSE direct | RMSE decomposed | Δ-RMSE | 95% CI | Verdict | Expected composite-fpts Δ |
|---|---:|---:|---:|---:|---|:---:|---:|
| receptions | [n] | [d] | [c] | [δ] | [lo, hi] | [V] | [δ × 1.0] |
| receiving_yards | [n] | [d] | [c] | [δ] | [lo, hi] | [V] | [δ × 0.1] |
| receiving_tds | [n] | [d] | [c] | [δ] | [lo, hi] | [V] | [δ × 6.0] |

**Family verdict:** [SIGNAL on N of 3 / NULL on all 3 / MIXED with REGRESSION on M]. [Decision branch fired per spec §4: greenlights integration plan / closes at this unit / writes a tighter follow-up probe.]

**Probe-vs-gate calibration risk (per spec §5 risk #1).** Per-stat RMSE Δ translated to expected composite-fpts contribution via the ESPN PPR coefficients (1.0 fpt/rec, 0.1 fpt/yd, 6.0 fpt/td). [Net expected composite-fpts magnitude × across the 3 stats]. Per the PR #31 retrospective rule, magnitudes < 0.005 fpts under coverage relaxation should be treated as MARGINAL, not SIGNAL. [Apply the rule to the strongest cell.]

**Factor orthogonality check (per spec §5 risk #2).** Pearson ρ between volume residual and efficiency residual per eval year, on rows with `targets > 0`. [Summary: |ρ| < 0.2 in all year × stat cells / |ρ| > 0.2 caveat in N cells documented]. [Implication: decomposition cleanly separates the two signal axes / efficiency factor partly redundant with predicted-volume residual; integration plan should account.]

**Coverage:** [targets > 0 rate per (eval year, train window). Default --coverage-threshold 0.95 met / relaxed to X with documentation per the PR #31 retrospective rule].

**Plan-vs-execution deviation.** [Any spec gaps caught + fixed during execution. None expected on a self-contained probe.]

**What this closes:** [If NULL: closes target decomposition at the WR receiving stats × 2-factor unit. None queued; refined decompositions and other positions remain open under TODO #23 but require independent mechanism evidence before re-probing per the post-PR-31 retrospective rule.] [If SIGNAL: greenlights integration plan with named follow-ups — `DecomposedBaselineModel` peer, within-row coherent factor sampling via `ProductDistribution`, factor-appropriate sub-model classes (logistic / log-link / Poisson) as a separate probe + integration cycle.]

**Reports:** `reports/feature_probe_target_decomposition_summary.md`, `reports/feature_probe_target_decomposition_per_stat.csv`, 3 per-stat `.md` reports.

---
```

(Fill in the bracketed placeholders from the actual probe run results.)

- [ ] **Step 3: Append a TODO #23 update**

Open `TODO.md`, find the `### 23. Target decomposition (volume × efficiency)` section, and append at the end of that section (before `### 24.`):

```markdown
**Update 2026-05-10 (target decomposition probe, branch `feat/probe-target-decomposition`):** First model-architecture probe in the project shipped per `docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md`. Bundled three WR receiving stats (`receptions`, `receiving_yards`, `receiving_tds`) decomposed as `targets × {catch_rate, yards_per_target, td_rate_per_target}` and probed via a new lite walk-forward CV harness (`src/projections/backtest/target_decomposition_probe.py`). RidgeCV on every sub-model + direct comparator (matches `BaselineModel.fit`'s algorithmic family) so verdicts are attributable to decomposition itself. **Family verdict: [SIGNAL/NULL/MIXED]** — [per-stat tally]. [Decision branch + follow-up disposition. If SIGNAL: integration plan name and roadmap (DecomposedBaselineModel + ProductDistribution + coherent within-row sampling; factor-appropriate sub-models as a deferred separate cycle). If NULL: refined decompositions + other-position decompositions remain open under this TODO but require independent mechanism evidence before re-probing per post-PR-31 retrospective rule. None queued.] See `reports/feature_probe_target_decomposition_summary.md`.
```

- [ ] **Step 4: Run the full verification gates one more time**

```bash
PYTHONPATH=src .venv/Scripts/python -m pytest -v
PATH="$(pwd)/.venv/Scripts:$PATH" mypy src tests
PATH="$(pwd)/.venv/Scripts:$PATH" ruff check src tests scripts
PATH="$(pwd)/.venv/Scripts:$PATH" ruff format --check src tests scripts
```
Expected: clean across all four. Capture the exit codes for the commit message evidence per CLAUDE.md "Forced verification" rule.

- [ ] **Step 5: Commit PM/TODO**

```bash
PATH="$(pwd)/.venv/Scripts:$PATH" git add project_management.md TODO.md
PATH="$(pwd)/.venv/Scripts:$PATH" git commit -m "docs(pm,todo): record target decomposition probe verdict — [SIGNAL/NULL]

Verification gates: pytest [n] passed, mypy clean, ruff clean, ruff format
clean. [One-line summary of probe verdict + decision branch.] Follow-up
disposition documented under TODO #23.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Push and open PR**

```bash
PATH="$(pwd)/.venv/Scripts:$PATH" git push -u origin feat/probe-target-decomposition
PATH="$(pwd)/.venv/Scripts:$PATH" gh pr create \
  --title "probe: WR receiving stats target decomposition — verdict [SIGNAL/NULL]" \
  --body "$(cat <<'EOF'
## Summary
- First **model-architecture probe** in the project (prior Track 2A probes all test feature additions via override parquets; this tests a prediction-recipe change with no override layer).
- Decomposes WR receiving stats into `targets × {catch_rate, yards_per_target, td_rate_per_target}`. RidgeCV on every sub-model + direct comparator so verdicts are attributable to decomposition itself, not to a model-class change.
- Walk-forward 2021–2024, paired-bootstrap CI on pooled per-stat RMSE delta.
- **Family verdict: [SIGNAL/NULL/MIXED].**

## Per-stat verdicts (pooled 2021–2024)

| Stat | Δ-RMSE | 95% CI | Verdict |
|---|---:|---|:---:|
| receptions | [δ] | [lo, hi] | [V] |
| receiving_yards | [δ] | [lo, hi] | [V] |
| receiving_tds | [δ] | [lo, hi] | [V] |

## Decision branch (spec §4)

[Greenlights integration plan with named follow-ups / closes target decomposition at the WR receiving × 2-factor unit / writes a tighter follow-up probe.]

## What ships in this PR

- `src/projections/backtest/target_decomposition_probe.py` — probe core.
- `tests/test_backtest/test_target_decomposition_probe.py` — unit + walk-forward tests.
- `scripts/probe_target_decomposition.py` — CLI.
- `tests/test_scripts/test_probe_target_decomposition_cli.py` — CLI tests.
- `reports/feature_probe_target_decomposition_*.{md,csv}` — verdict + per-stat detail + summary.
- `project_management.md` + `TODO.md` — verdict log.

## Test plan

- [x] `pytest -v` clean.
- [x] `mypy src tests` zero violations.
- [x] `ruff check src tests scripts` zero violations.
- [x] `ruff format --check src tests scripts` no drift.
- [x] CLI dispatched against the production WR feature cache + weekly stats; outputs inspected.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review checklist

After execution, the implementer should sanity-check the plan against the spec:

- [ ] Spec §1.1 (3 decomposed stats with shared volume + per-stat efficiency) → implemented in Task 1's `_WR_RECEIVING_DECOMPS` registry; consumed by Task 2's harness.
- [ ] Spec §1.3.1 (coverage threshold 0.95 across eval years and train windows) → enforced in Task 4's `_enforce_coverage`.
- [ ] Spec §1.3.2 (probe completeness — 3 per-stat reports + summary) → produced by Task 4's CLI + Task 5's writeup.
- [ ] Spec §1.3.3 (verdict mapping pure CI-based) → implemented in Task 3's `_verdict_from_delta`.
- [ ] Spec §1.3.4 (verification gates) → run in Tasks 1, 2, 3, 4 (per-task narrow scope) and Tasks 5, 6 (full suite).
- [ ] Spec §3.2 (alphas match BaselineModel.fit; bool→int8 coercion; train medians for predict-time imputation) → implemented in Task 1's `_RIDGE_ALPHAS` and Task 2's harness.
- [ ] Spec §3.3 (walk-forward; train/eval strict separation assertion) → implemented + tested in Task 2.
- [ ] Spec §3.4 (paired-bootstrap CI; verdict mapping) → implemented in Task 3.
- [ ] Spec §3.5 (CLI flags + defaults) → implemented in Task 4.
- [ ] Spec §3.6 (report layout) → implemented in Tasks 3 + 4.
- [ ] Spec §4 (decision branches) → addressed in Task 5's summary template + Task 6's PM/TODO entry templates.
- [ ] Spec §5 risk #1 (composite-fpts translation) → implemented in Task 3's `expected_composite_fpts_delta` field + per-stat markdown rendering.
- [ ] Spec §5 risk #2 (factor residual ρ per eval year) → implemented in Task 3's `factor_residual_correlation_by_year` + per-stat markdown.
- [ ] Spec §5 risk #5 (`_WR_FEATURE_COLUMNS` drift) → addressed by Task 4's runtime import; mitigation note documented.
- [ ] Spec §5 risk #6 (train/eval season strict separation) → implemented + tested in Task 2.
- [ ] Spec §6 (test coverage) → implemented across Tasks 1, 2, 3, 4.
- [ ] Spec §7 (named follow-ups) → carried forward in Task 5's summary template + Task 6's PM/TODO entries (not built in the probe).
- [ ] Spec §8 (estimated 5–6 plan tasks) → matches: 6 tasks, each ≤ 5 files per CLAUDE.md "phased execution" rule.
