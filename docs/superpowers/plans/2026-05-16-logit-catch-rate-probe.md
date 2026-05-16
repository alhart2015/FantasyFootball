# Logit catch_rate Sub-Model Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Probe whether replacing the catch_rate efficiency sub-model class from `RidgeCV` on the ratio + clipped-Normal predict-time to `LogisticRegressionCV` via Bernoulli-trial row expansion lowers per-stat receptions RMSE on WR rows, 2021-2024 walk-forward.

**Architecture:** Two-arm probe in a new pure-numpy/pandas/sklearn module. Both arms share the same `RidgeCV` on `targets`; only the catch_rate efficiency sub-model class differs. Per-row predictions emitted; pooled paired-bootstrap CI on the receptions Δ-RMSE. No production code touched.

**Tech Stack:** Python, numpy, pandas, sklearn (`RidgeCV`, `LogisticRegressionCV`, `StandardScaler`), pandera schemas, pytest, mypy strict, ruff.

**Spec:** `docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md`

---

## Pre-task: re-point venv editable install

The `.venv` at `C:/Users/alden/FantasyFootball/.venv` is editable-installed against a different worktree (`feat-wr-ensemble-decomposed-child` from the merged PR #38). Any test invocation from inside the new worktree resolves imports against THAT directory, not the current one — silent staleness that wasted ~10 min in PR #38 Task 1.

Run this from the new worktree's root before starting Task 1:

```bash
cd C:/Users/alden/FantasyFootball/.worktrees/feat-probe-logit-catch-rate
../../.venv/Scripts/python.exe -m pip install -e . --no-deps
```

Verify:

```bash
../../.venv/Scripts/python.exe -c "from importlib.metadata import distribution; import json; print(json.loads(distribution('projections').read_text('direct_url.json'))['url'])"
```

Expected output: `file:///C:/Users/alden/FantasyFootball/.worktrees/feat-probe-logit-catch-rate`

If the URL still points at any other worktree, re-run the `pip install -e .` from this worktree's root.

---

## File Map

**Create:**
- `src/projections/backtest/logit_catch_rate_probe.py` — probe core: row expansion, sub-model fit helpers, prediction helpers, walk-forward, verdict
- `scripts/probe_logit_catch_rate.py` — CLI driver
- `tests/test_backtest/test_logit_catch_rate_probe.py` — unit + integration tests for the probe module
- `tests/test_scripts/test_probe_logit_catch_rate_cli.py` — CLI smoke
- `reports/feature_probe_logit_catch_rate_summary.md` — Task 5 output
- `reports/feature_probe_logit_catch_rate.csv` — Task 5 output

**Modify (Task 5 only):**
- `project_management.md` — top-of-file decision-log entry
- `TODO.md` — update entry under #23 / #33b with probe verdict

---

## Task 1: Bernoulli-trial row expansion

**Files:**
- Create: `src/projections/backtest/logit_catch_rate_probe.py`
- Create: `tests/test_backtest/test_logit_catch_rate_probe.py`

Scope: only the `_expand_to_trials` helper + its unit tests. No fit logic yet.

- [ ] **Step 1.1: Re-point venv editable install per pre-task block above.** Confirm with the verify command before proceeding.

- [ ] **Step 1.2: Write the failing tests**

Create `tests/test_backtest/test_logit_catch_rate_probe.py`:

```python
"""Tests for the logit catch_rate probe.

Spec: docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md.
"""

from __future__ import annotations

import numpy as np

from projections.backtest.logit_catch_rate_probe import _expand_to_trials


def test_expand_to_trials_basic_shape_and_labels() -> None:
    """3 rows with (T, S) = (4, 3), (2, 0), (5, 5). Expansion yields 11 trial
    rows: 3+1+0+2+5+0 = 7 successes and 1+2+0 = 3 failures, sharing X.
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    successes = np.array([3, 0, 5], dtype=np.int64)
    trials = np.array([4, 2, 5], dtype=np.int64)

    x_trials, y_trials = _expand_to_trials(x, successes, trials)

    assert x_trials.shape == (11, 2)
    assert y_trials.shape == (11,)
    # Row 0: 4 copies of [1, 2] -> first 3 y=1, last 1 y=0
    assert np.allclose(x_trials[:4], np.tile([1.0, 2.0], (4, 1)))
    assert np.array_equal(y_trials[:4], np.array([1, 1, 1, 0]))
    # Row 1: 2 copies of [3, 4] -> both y=0
    assert np.allclose(x_trials[4:6], np.tile([3.0, 4.0], (2, 1)))
    assert np.array_equal(y_trials[4:6], np.array([0, 0]))
    # Row 2: 5 copies of [5, 6] -> all y=1
    assert np.allclose(x_trials[6:11], np.tile([5.0, 6.0], (5, 1)))
    assert np.array_equal(y_trials[6:11], np.array([1, 1, 1, 1, 1]))


def test_expand_to_trials_zero_trials_dropped() -> None:
    """A row with T=0 must be dropped from the expansion entirely
    (rather than panicking on shape mismatch in the per-row alloc).
    """
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    successes = np.array([2, 0, 1], dtype=np.int64)
    trials = np.array([2, 0, 3], dtype=np.int64)

    x_trials, y_trials = _expand_to_trials(x, successes, trials)

    # Row 0: 2 successes; Row 1: 0 trials (dropped); Row 2: 1 success + 2 failures.
    assert x_trials.shape == (5, 1)
    assert y_trials.shape == (5,)
    assert np.allclose(x_trials[:2], np.array([[1.0], [1.0]]))
    assert np.array_equal(y_trials[:2], np.array([1, 1]))
    assert np.allclose(x_trials[2:], np.array([[3.0], [3.0], [3.0]]))
    assert np.array_equal(y_trials[2:], np.array([1, 0, 0]))


def test_expand_to_trials_validates_successes_le_trials() -> None:
    """successes[i] > trials[i] is a bug in the caller. Raise ValueError
    rather than producing a corrupt expansion.
    """
    import pytest

    x = np.array([[1.0]], dtype=np.float64)
    successes = np.array([5], dtype=np.int64)
    trials = np.array([3], dtype=np.int64)

    with pytest.raises(ValueError, match=r"successes\[0\]=5 > trials\[0\]=3"):
        _expand_to_trials(x, successes, trials)


def test_expand_to_trials_handles_empty_input() -> None:
    """Empty (X, successes, trials) returns empty arrays of the right shape."""
    x = np.empty((0, 3), dtype=np.float64)
    successes = np.empty((0,), dtype=np.int64)
    trials = np.empty((0,), dtype=np.int64)

    x_trials, y_trials = _expand_to_trials(x, successes, trials)

    assert x_trials.shape == (0, 3)
    assert y_trials.shape == (0,)
```

- [ ] **Step 1.3: Run tests — expect ImportError**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_logit_catch_rate_probe.py -v
```

Expected: collection error, `ImportError: cannot import name '_expand_to_trials' from 'projections.backtest.logit_catch_rate_probe'`.

- [ ] **Step 1.4: Create the module skeleton + implement `_expand_to_trials`**

Create `src/projections/backtest/logit_catch_rate_probe.py`:

```python
"""Logit catch_rate sub-model probe — factor-appropriate efficiency factor.

Two-arm probe comparing the production catch_rate sub-model class (RidgeCV
on the ratio with predict-time clipping to [0, 1]) against a binomial-logit
fit (LogisticRegressionCV via Bernoulli-trial row expansion). Per-stat
receptions Delta-CV-RMSE, walk-forward eval window 2021-2024, paired-bootstrap
CI on pooled residuals.

Mirrors `target_decomposition_probe.py`'s shape; reuses
`paired_bootstrap_rmse_delta` and `BootstrapDelta` from `adoption_gate.py`.

Spec: docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md.
"""

from __future__ import annotations

import numpy as np


def _expand_to_trials(
    x: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand each row into individual Bernoulli trials for a binomial-logit fit.

    For row i with `trials[i] = T` and `successes[i] = S`, emit T copies of
    `x[i]` — the first S with `y=1`, the remaining (T - S) with `y=0`. Rows
    with `trials[i] = 0` are dropped entirely.

    The expanded (X_trials, y_trials) pair is the LogisticRegressionCV input
    that recovers the same MLE as a binomial-logit GLM by likelihood
    factorization.

    Args:
        x: (n, n_features) feature matrix.
        successes: (n,) int array; count of successful trials per row.
        trials: (n,) int array; total trials per row.

    Returns:
        (x_trials, y_trials) where x_trials has shape (sum(trials), n_features)
        and y_trials has shape (sum(trials),) with int 0/1 values.

    Raises:
        ValueError: if any successes[i] > trials[i].
    """
    if x.shape[0] != successes.shape[0] or x.shape[0] != trials.shape[0]:
        raise ValueError(
            f"row count mismatch: x={x.shape[0]}, successes={successes.shape[0]}, "
            f"trials={trials.shape[0]}"
        )
    overflow = successes > trials
    if overflow.any():
        bad = int(np.argmax(overflow))
        raise ValueError(
            f"successes[{bad}]={int(successes[bad])} > trials[{bad}]={int(trials[bad])}"
        )

    keep = trials > 0
    x_kept = x[keep]
    successes_kept = successes[keep].astype(np.int64)
    trials_kept = trials[keep].astype(np.int64)

    # Repeat each kept row T times along axis 0.
    x_trials = np.repeat(x_kept, trials_kept, axis=0)

    # Build y per kept row: S ones followed by (T - S) zeros.
    failures_kept = trials_kept - successes_kept
    y_trials_parts: list[np.ndarray] = []
    for s, f in zip(successes_kept, failures_kept, strict=True):
        y_trials_parts.append(np.ones(int(s), dtype=np.int64))
        y_trials_parts.append(np.zeros(int(f), dtype=np.int64))
    y_trials = (
        np.concatenate(y_trials_parts) if y_trials_parts else np.empty((0,), dtype=np.int64)
    )

    return x_trials, y_trials
```

- [ ] **Step 1.5: Run tests — expect PASS**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_logit_catch_rate_probe.py -v
```

Expected: 4 tests pass.

- [ ] **Step 1.6: Lint + typecheck**

```bash
../../.venv/Scripts/python.exe -m ruff check src tests
../../.venv/Scripts/python.exe -m ruff format --check src tests
../../.venv/Scripts/python.exe -m mypy src tests
```

Expected: zero violations across all three.

- [ ] **Step 1.7: Commit**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  src/projections/backtest/logit_catch_rate_probe.py \
  tests/test_backtest/test_logit_catch_rate_probe.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(probe): _expand_to_trials helper for binomial-logit row expansion

For each (X, T trials, S successes) row, emits T copies of X with the first
S as y=1 and the remaining (T - S) as y=0. Drops T=0 rows. Validates S <= T.
Foundation for the binomial-logit catch_rate sub-model in subsequent tasks.

Spec: docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md (Task 1).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Sub-model fit + predict helpers

**Files:**
- Modify: `src/projections/backtest/logit_catch_rate_probe.py` (extend)
- Modify: `tests/test_backtest/test_logit_catch_rate_probe.py` (extend)

Scope: shared volume + ridge efficiency + logit efficiency fit functions, plus per-row receptions predictors for both arms.

- [ ] **Step 2.1: Append the failing tests**

Append to `tests/test_backtest/test_logit_catch_rate_probe.py`:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline

from projections.backtest.logit_catch_rate_probe import (
    _LOGIT_CS,
    _RIDGE_ALPHAS,
    _fit_logit_efficiency,
    _fit_ridge_efficiency,
    _fit_shared_volume,
    _predict_receptions_logit,
    _predict_receptions_ridge,
)


def test_fit_shared_volume_returns_fitted_ridgecv() -> None:
    """Shared volume fit on a linear-target synthetic frame recovers the slope."""
    rng = np.random.default_rng(seed=2026)
    x = rng.uniform(0.0, 5.0, size=(200, 3)).astype(np.float64)
    # true: targets = 2 * x[:, 0] + 1 * x[:, 1] - 0.5 * x[:, 2] + noise
    targets = (
        2.0 * x[:, 0]
        + 1.0 * x[:, 1]
        - 0.5 * x[:, 2]
        + rng.normal(0, 0.2, size=200)
    ).astype(np.float64)

    ridge = _fit_shared_volume(x, targets)

    assert isinstance(ridge, RidgeCV)
    assert ridge.alpha_ in _RIDGE_ALPHAS
    # Coefficient recovery sanity.
    assert abs(ridge.coef_[0] - 2.0) < 0.1
    assert abs(ridge.coef_[1] - 1.0) < 0.1


def test_fit_ridge_efficiency_matches_pinned_alpha_grid() -> None:
    """Ridge efficiency fit uses the same alpha grid as BaselineModel.fit."""
    rng = np.random.default_rng(seed=2027)
    x = rng.uniform(0.0, 1.0, size=(150, 2)).astype(np.float64)
    ratio = (0.6 + 0.3 * x[:, 0] + rng.normal(0, 0.05, size=150)).astype(np.float64)

    ridge = _fit_ridge_efficiency(x, ratio)

    assert isinstance(ridge, RidgeCV)
    assert ridge.alpha_ in _RIDGE_ALPHAS


def test_fit_logit_efficiency_recovers_true_p() -> None:
    """Generate (X, S, T) from a known logit; verify predict_proba on a holdout
    matches the true catch_rate within tolerance.
    """
    rng = np.random.default_rng(seed=2028)
    n_rows = 400
    x = rng.uniform(-1.0, 1.0, size=(n_rows, 2)).astype(np.float64)
    # true: logit(p) = -0.4 + 1.2 * x[:, 0] - 0.6 * x[:, 1]
    logit_p = -0.4 + 1.2 * x[:, 0] - 0.6 * x[:, 1]
    true_p = 1.0 / (1.0 + np.exp(-logit_p))
    trials = rng.integers(1, 15, size=n_rows).astype(np.int64)
    successes = rng.binomial(trials, true_p).astype(np.int64)

    x_trials, y_trials = _expand_to_trials(x, successes, trials)
    logit = _fit_logit_efficiency(x_trials, y_trials)

    # Pipeline wraps StandardScaler + LogisticRegressionCV per spec §5 risk #6.
    assert isinstance(logit, Pipeline)
    assert "scaler" in logit.named_steps
    inner = logit.named_steps["logit"]
    assert isinstance(inner, LogisticRegressionCV)
    assert inner.C_[0] in _LOGIT_CS

    # predict_proba on the Pipeline transparently scales x before the logit step.
    pred_p = logit.predict_proba(x)[:, 1]
    mae = float(np.abs(pred_p - true_p).mean())
    assert mae < 0.05, f"binomial-logit MAE on true_p too large: {mae}"


def test_predict_receptions_ridge_clips_to_unit_interval() -> None:
    """Ridge predictions outside [0, 1] are clipped before multiplying by volume."""
    rng = np.random.default_rng(seed=2029)
    x_train = rng.uniform(0.0, 1.0, size=(100, 2)).astype(np.float64)
    # Synthetic ratio = 0.5 + 1.5 * x[:, 0]: forces unclipped predictions > 1
    # on the high-x test rows, so the clip is exercised.
    ratio_train = (0.5 + 1.5 * x_train[:, 0]).astype(np.float64)
    ridge_eff = _fit_ridge_efficiency(x_train, ratio_train)

    x_eval = np.array([[1.0, 0.5], [0.9, 0.5]], dtype=np.float64)
    mu_targets = np.array([10.0, 8.0], dtype=np.float64)

    pred = _predict_receptions_ridge(mu_targets, x_eval, ridge_eff)

    # mu_eff at x=1.0 is ~2.0 (unclipped) -> clipped to 1.0 -> receptions = 10.0
    assert pred[0] == 10.0
    # mu_eff at x=0.9 is ~1.85 (unclipped) -> clipped to 1.0 -> receptions = 8.0
    assert pred[1] == 8.0


def test_predict_receptions_logit_uses_predict_proba() -> None:
    """Logit prediction equals mu_targets * predict_proba(X)[:, 1]."""
    rng = np.random.default_rng(seed=2030)
    n_rows = 200
    x = rng.uniform(-1.0, 1.0, size=(n_rows, 2)).astype(np.float64)
    trials = rng.integers(2, 10, size=n_rows).astype(np.int64)
    true_p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x[:, 0])))
    successes = rng.binomial(trials, true_p).astype(np.int64)

    x_trials, y_trials = _expand_to_trials(x, successes, trials)
    logit_eff = _fit_logit_efficiency(x_trials, y_trials)

    x_eval = rng.uniform(-1.0, 1.0, size=(10, 2)).astype(np.float64)
    mu_targets = rng.uniform(2.0, 10.0, size=10).astype(np.float64)

    pred = _predict_receptions_logit(mu_targets, x_eval, logit_eff)

    expected = mu_targets * logit_eff.predict_proba(x_eval)[:, 1]
    assert np.allclose(pred, expected)
    assert (pred >= 0).all()
```

- [ ] **Step 2.2: Run tests — expect import / attribute errors**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_logit_catch_rate_probe.py -v
```

Expected: collection error or import failures on `_RIDGE_ALPHAS`, `_LOGIT_CS`, `_fit_shared_volume`, etc.

- [ ] **Step 2.3: Extend the probe module with fit + predict helpers**

Append to `src/projections/backtest/logit_catch_rate_probe.py`:

```python
from typing import Final

from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Same alpha grid as `BaselineModel.fit` (src/projections/models/baseline.py) and
# `target_decomposition_probe._fit_direct` so the two probe arms differ only in
# the catch_rate sub-model class, not the regularization scale.
_RIDGE_ALPHAS: Final[np.ndarray] = np.logspace(-3, 3, 13)

# Cs grid for LogisticRegressionCV. C = 1 / alpha (sklearn's inverse-penalty
# convention). 5 points spanning 3 orders of magnitude — matches the
# effective regularization range of the Ridge alpha grid for the row-expanded
# Bernoulli trials.
_LOGIT_CS: Final[tuple[float, ...]] = (0.01, 0.1, 1.0, 10.0, 100.0)


def _fit_shared_volume(x: np.ndarray, targets: np.ndarray) -> RidgeCV:
    """Fit the shared volume sub-model: `targets ~ X` via RidgeCV.

    Trained on all rows (no targets > 0 filter); zero-target rows are valid
    observations of low-volume players and the volume model must predict them.
    Identical recipe to `target_decomposition_probe._fit_decomposed_volume`.
    """
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
    ridge.fit(x, targets.astype(np.float64))
    return ridge


def _fit_ridge_efficiency(x_pos: np.ndarray, ratio: np.ndarray) -> RidgeCV:
    """Fit the Ridge-on-ratio efficiency sub-model (incumbent arm).

    Matches the catch_rate fit in `decomposed_baseline.py` exactly: RidgeCV
    on `receptions / targets` (computed only on rows with `targets > 0`).
    """
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
    ridge.fit(x_pos, ratio.astype(np.float64))
    return ridge


def _fit_logit_efficiency(x_trials: np.ndarray, y_trials: np.ndarray) -> Pipeline:
    """Fit the binomial-logit efficiency sub-model (candidate arm).

    Expects row-expanded Bernoulli trials from `_expand_to_trials`. The fit is
    mathematically equivalent to a binomial-logit GLM on (S, T-S) via MLE.

    Wraps StandardScaler + LogisticRegressionCV in a sklearn Pipeline per
    spec §5 risk #6 mitigation: LogisticRegression's L2 penalty is
    scale-dependent (whereas Ridge's CV-selected alpha is approximately
    scale-invariant); scaling the features stops the regularization scale
    from becoming a confounder between sub-model class and feature scale.
    Scaler is fit on the trial-expanded rows and persisted inside the
    Pipeline for predict-time use.

    Uses L2 regularization (matching Ridge's penalty family) and 5-fold CV
    across the `_LOGIT_CS` grid. Default solver `lbfgs` works well for L2
    logistic on this row count.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegressionCV(
                    Cs=list(_LOGIT_CS),
                    cv=5,
                    penalty="l2",
                    scoring="neg_log_loss",
                    solver="lbfgs",
                    max_iter=1000,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(x_trials, y_trials)
    return pipeline


def _predict_receptions_ridge(
    mu_targets: np.ndarray, x_eval: np.ndarray, ridge_eff: RidgeCV
) -> np.ndarray:
    """Incumbent-arm receptions prediction: mu_targets * clip(mu_ratio, 0, 1).

    Matches the production `decomposed_baseline.py` predict path for the
    mean of the receptions distribution (the predict-time variance/sampling
    is not exercised here — the probe is mean-only).
    """
    mu_ratio = ridge_eff.predict(x_eval).astype(np.float64)
    mu_ratio_clipped = np.clip(mu_ratio, 0.0, 1.0)
    return mu_targets * mu_ratio_clipped


def _predict_receptions_logit(
    mu_targets: np.ndarray, x_eval: np.ndarray, logit_eff: Pipeline
) -> np.ndarray:
    """Candidate-arm receptions prediction: mu_targets * P(success | x).

    Uses Pipeline.predict_proba, which applies the fitted StandardScaler to
    x_eval before LogisticRegressionCV.predict_proba. The second column is
    P(y=1) under sklearn's binary classification convention.
    """
    p = logit_eff.predict_proba(x_eval)[:, 1].astype(np.float64)
    return mu_targets * p
```

- [ ] **Step 2.4: Run tests — expect PASS**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_logit_catch_rate_probe.py -v
```

Expected: 9 tests pass (4 Task-1 + 5 Task-2). If any logit test produces a `ConvergenceWarning`, the test still passes but the implementer should review whether `max_iter=1000` is sufficient on the synthetic fixture (it is for these row counts; the warning would indicate a real bug).

- [ ] **Step 2.5: Lint + typecheck**

```bash
../../.venv/Scripts/python.exe -m ruff check src tests
../../.venv/Scripts/python.exe -m ruff format --check src tests
../../.venv/Scripts/python.exe -m mypy src tests
```

Expected: zero violations.

- [ ] **Step 2.6: Commit**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  src/projections/backtest/logit_catch_rate_probe.py \
  tests/test_backtest/test_logit_catch_rate_probe.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(probe): sub-model fit + predict helpers for both arms

_fit_shared_volume + _fit_ridge_efficiency match the existing
target_decomposition_probe recipe (RidgeCV with the same alpha grid as
BaselineModel.fit). _fit_logit_efficiency wires LogisticRegressionCV
(L2, Cs=5-point log grid, cv=5) on row-expanded Bernoulli trials —
mathematically equivalent to binomial-logit GLM by MLE.

_predict_receptions_ridge replicates the production clipped-Normal mean
prediction (mu_targets * clip(mu_ratio, 0, 1)).
_predict_receptions_logit uses predict_proba.

Spec: docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md (Task 2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Walk-forward + verdict

**Files:**
- Modify: `src/projections/backtest/logit_catch_rate_probe.py` (extend)
- Modify: `tests/test_backtest/test_logit_catch_rate_probe.py` (extend)

Scope: the full walk-forward harness that fits both arms per train window and produces pooled per-row residuals; the verdict mapping.

- [ ] **Step 3.1: Append the failing tests**

Append to `tests/test_backtest/test_logit_catch_rate_probe.py`:

```python
from collections.abc import Sequence

from projections.backtest.logit_catch_rate_probe import (
    PerStatVerdict,
    ProbeResults,
    compute_verdict,
    walk_forward_residuals,
)


def _synthetic_wr_inputs(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small synthetic WR (features, weekly_stats) pair for walk-forward
    integration testing.

    4 seasons × 4 weeks × 8 players = 128 rows. Features are uniform random;
    truth uses a known catch_rate-from-features generative model with targets
    correlated with one feature so volume_ridge has signal.
    """
    from projections.schemas import WeeklyStatsSchema, _PYARROW_STR

    rng = np.random.default_rng(seed=seed)
    rows: list[dict[str, object]] = []
    for season in range(2018, 2022):
        for week in range(1, 5):
            for p in range(8):
                rows.append(
                    {
                        "gsis_id": f"00-{1_000_000 + p:07d}",
                        "season": np.int64(season),
                        "week": np.int64(week),
                        "team": "KC",
                        "opponent": "DEN",
                    }
                )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)

    # Use the WR feature schema's columns; fill plausibles for the rest.
    from projections.models import POSITION_DISPATCH
    from projections.schemas import Position
    feature_schema = POSITION_DISPATCH[Position.WR].feature_schema
    schema_cols = feature_schema.to_schema().columns
    for col_name, col in schema_cols.items():
        if col_name in df.columns:
            continue
        dtype_str = str(col.dtype)
        if "bool" in dtype_str.lower():
            df[col_name] = rng.integers(0, 2, size=len(df)).astype(bool)
        elif "int" in dtype_str.lower():
            df[col_name] = rng.integers(1, 6, size=len(df)).astype(np.int64)
        elif col_name == "age":
            df[col_name] = rng.uniform(22.0, 30.0, size=len(df)).astype(np.float64)
        else:
            df[col_name] = rng.uniform(0.0, 1.0, size=len(df)).astype(np.float64)
    features = feature_schema.validate(df)

    ws = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    ws["position"] = "WR"
    n = len(ws)
    target_lambda = rng.uniform(3.0, 12.0, size=n)
    targets = np.maximum(1, rng.poisson(target_lambda)).astype(np.int64)
    true_p = 1.0 / (1.0 + np.exp(-(0.4 + 0.5 * target_lambda / 10.0)))
    receptions = np.array(
        [int(rng.binomial(t, p)) for t, p in zip(targets, true_p, strict=True)],
        dtype=np.int64,
    )
    ws["targets"] = targets
    ws["receptions"] = receptions
    ws["receiving_yards"] = np.maximum(0.0, receptions * rng.normal(11.0, 3.0, size=n)).astype(
        np.float64
    )
    ws["receiving_tds"] = np.where(
        rng.uniform(0, 1, size=n) < np.minimum(targets * 0.05, 1.0), 1, 0
    ).astype(np.int64)
    ws["rushing_yards"] = np.zeros(n, dtype=np.float64)
    ws["rushing_tds"] = np.zeros(n, dtype=np.int64)
    ws["fumbles_lost"] = np.zeros(n, dtype=np.int64)
    ws["passing_yards"] = 0.0
    ws["passing_tds"] = np.int64(0)
    ws["interceptions"] = np.int64(0)

    schema_cols_ws = WeeklyStatsSchema.to_schema().columns
    for col_name, col in schema_cols_ws.items():
        if col_name in ws.columns:
            continue
        dtype_str = str(col.dtype)
        if "int" in dtype_str.lower():
            ws[col_name] = np.zeros(n, dtype=np.int64)
        elif "float" in dtype_str.lower():
            ws[col_name] = np.zeros(n, dtype=np.float64)
        else:
            ws[col_name] = 0
    return features, WeeklyStatsSchema.validate(ws)


def test_walk_forward_residuals_produces_paired_arrays() -> None:
    """Both ridge and logit prediction buffers are populated and length-matched
    on every eval year.
    """
    features, weekly_stats = _synthetic_wr_inputs()
    results = walk_forward_residuals(
        features, weekly_stats, eval_years=(2020, 2021)
    )

    assert isinstance(results, ProbeResults)
    assert len(results.actual_receptions) > 0
    assert results.actual_receptions.shape == results.pred_ridge.shape
    assert results.actual_receptions.shape == results.pred_logit.shape
    assert results.year.shape == results.actual_receptions.shape
    # Coverage stats present for each eval year.
    assert set(results.coverage_per_year.keys()) == {2020, 2021}
    # All predictions non-negative.
    assert (results.pred_ridge >= 0).all()
    assert (results.pred_logit >= 0).all()


def test_walk_forward_residuals_arms_differ_on_some_rows() -> None:
    """The two arms should not produce identical predictions everywhere. This
    sanity-checks that the logit arm is actually exercised (a no-op fall-through
    to the ridge arm would be a silent bug).
    """
    features, weekly_stats = _synthetic_wr_inputs(seed=12345)
    results = walk_forward_residuals(features, weekly_stats, eval_years=(2021,))

    n_different = int(np.sum(np.abs(results.pred_ridge - results.pred_logit) > 1e-6))
    assert n_different > 0, "ridge and logit arms produced identical predictions"


def test_compute_verdict_signal_when_ci_strictly_negative() -> None:
    """Synthetic ProbeResults where the logit clearly beats the ridge -> SIGNAL."""
    rng = np.random.default_rng(seed=2031)
    n = 500
    actual = rng.uniform(0.0, 10.0, size=n)
    # ridge has 0.5 systematic bias; logit is unbiased.
    pred_ridge = actual + 0.5 + rng.normal(0, 0.1, size=n)
    pred_logit = actual + rng.normal(0, 0.1, size=n)
    year = np.full(n, 2024, dtype=np.int64)
    results = ProbeResults(
        actual_receptions=actual,
        pred_ridge=pred_ridge,
        pred_logit=pred_logit,
        year=year,
        coverage_per_year={2024: 1.0},
    )

    verdict = compute_verdict(results, n_bootstrap=200, seed=42)

    assert isinstance(verdict, PerStatVerdict)
    assert verdict.verdict == "SIGNAL"
    assert verdict.rmse_delta.hi_95 < 0


def test_compute_verdict_null_when_ci_brackets_zero() -> None:
    """Random noise -> NULL."""
    rng = np.random.default_rng(seed=2032)
    n = 300
    actual = rng.uniform(0.0, 10.0, size=n)
    pred_ridge = actual + rng.normal(0, 1.0, size=n)
    pred_logit = actual + rng.normal(0, 1.0, size=n)
    year = np.full(n, 2024, dtype=np.int64)
    results = ProbeResults(
        actual_receptions=actual,
        pred_ridge=pred_ridge,
        pred_logit=pred_logit,
        year=year,
        coverage_per_year={2024: 1.0},
    )

    verdict = compute_verdict(results, n_bootstrap=200, seed=42)

    assert verdict.verdict == "NULL"
    assert verdict.rmse_delta.lo_95 < 0 < verdict.rmse_delta.hi_95


def test_compute_verdict_regression_when_ci_strictly_positive() -> None:
    """Logit systematically worse than Ridge -> REGRESSION."""
    rng = np.random.default_rng(seed=2033)
    n = 500
    actual = rng.uniform(0.0, 10.0, size=n)
    pred_ridge = actual + rng.normal(0, 0.1, size=n)
    pred_logit = actual + 0.5 + rng.normal(0, 0.1, size=n)
    year = np.full(n, 2024, dtype=np.int64)
    results = ProbeResults(
        actual_receptions=actual,
        pred_ridge=pred_ridge,
        pred_logit=pred_logit,
        year=year,
        coverage_per_year={2024: 1.0},
    )

    verdict = compute_verdict(results, n_bootstrap=200, seed=42)

    assert verdict.verdict == "REGRESSION"
    assert verdict.rmse_delta.lo_95 > 0
```

- [ ] **Step 3.2: Run tests — expect import errors**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_logit_catch_rate_probe.py -v
```

Expected: collection / import errors on `walk_forward_residuals`, `ProbeResults`, `compute_verdict`, `PerStatVerdict`.

- [ ] **Step 3.3: Extend the module with walk-forward + verdict**

Append to `src/projections/backtest/logit_catch_rate_probe.py`:

```python
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from projections.backtest.adoption_gate import (
    BootstrapDelta,
    paired_bootstrap_rmse_delta,
)
from projections.schemas import Position, Stat, WeeklyStatsSchema


VerdictLabel = Literal["SIGNAL", "NULL", "REGRESSION"]


@dataclass(slots=True)
class ProbeResults:
    """Pooled per-row buffers from a walk-forward run.

    Attributes:
        actual_receptions: (N,) ground-truth receptions (float64 for residual math).
        pred_ridge: (N,) incumbent-arm receptions predictions.
        pred_logit: (N,) candidate-arm receptions predictions.
        year: (N,) eval year per row (int64).
        coverage_per_year: per-eval-year fraction of WR rows with targets > 0.
    """

    actual_receptions: np.ndarray
    pred_ridge: np.ndarray
    pred_logit: np.ndarray
    year: np.ndarray
    coverage_per_year: dict[int, float]


@dataclass(slots=True, frozen=True)
class PerStatVerdict:
    """Per-stat verdict on the receptions Δ-RMSE (logit − ridge).

    Mirrors `feature_probe.PerStatVerdict` but is local to this module so the
    probe is self-contained.
    """

    stat: Stat
    n_paired: int
    rmse_delta: BootstrapDelta
    verdict: VerdictLabel


def _x_frame_for_features(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract the X matrix from a WR features frame.

    Reuses `BaselineModel`'s feature-column set so the probe's X matches
    production. Bool columns are coerced to int8; NaN rows are dropped.

    Returns:
        (x, columns) where x is shape (n_rows, n_features) and columns is the
        ordered list of feature names retained.
    """
    from projections.models.baseline import _WR_FEATURE_COLUMNS

    cols = list(_WR_FEATURE_COLUMNS)
    sub = features[cols].copy()
    for c in cols:
        if pd.api.types.is_bool_dtype(sub[c]):
            sub[c] = sub[c].astype(np.int8)
    keep = sub.notna().all(axis=1)
    sub = sub.loc[keep]
    return sub.to_numpy(dtype=np.float64), cols


def walk_forward_residuals(
    features: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    eval_years: Iterable[int],
) -> ProbeResults:
    """For each eval year, train both arms on prior seasons, predict on the
    eval year, collect per-row residuals.

    Spec: probe-design §3.1 walk_forward_residuals.
    """
    eval_years_list = sorted(int(y) for y in eval_years)
    actual_buffer: list[np.ndarray] = []
    ridge_buffer: list[np.ndarray] = []
    logit_buffer: list[np.ndarray] = []
    year_buffer: list[np.ndarray] = []
    coverage_per_year: dict[int, float] = {}

    features_validated = features  # caller is responsible for schema validation
    ws = WeeklyStatsSchema.validate(weekly_stats)
    ws_wr = ws[ws["position"] == Position.WR.value].copy()

    all_seasons = sorted(int(s) for s in features_validated["season"].unique())

    for eval_year in eval_years_list:
        train_seasons = [s for s in all_seasons if s < eval_year]
        if not train_seasons:
            continue

        # Train-window join (features <-> weekly_stats on (gsis_id, season, week)).
        train_feat = features_validated[features_validated["season"].isin(train_seasons)]
        train_ws = ws_wr[ws_wr["season"].isin(train_seasons)]
        train_join = train_feat.merge(
            train_ws[["gsis_id", "season", "week", "targets", "receptions"]],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        x_train, _ = _x_frame_for_features(train_join)
        # _x_frame_for_features may drop rows on NaN — realign train_join.
        train_join = train_join.iloc[: x_train.shape[0]] if False else train_join.loc[
            features_validated[features_validated["season"].isin(train_seasons)]
            [list(train_join.columns) if False else train_join.columns.tolist()]
            .index.intersection(train_join.index)
        ]
        # Re-derive via the helper to keep mask alignment exact.
        from projections.models.baseline import _WR_FEATURE_COLUMNS
        keep_mask = train_join[list(_WR_FEATURE_COLUMNS)].notna().all(axis=1)
        train_join = train_join.loc[keep_mask]
        x_train = train_join[list(_WR_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        targets_train = train_join["targets"].to_numpy(dtype=np.int64)
        receptions_train = train_join["receptions"].to_numpy(dtype=np.int64)

        # Shared volume fit.
        volume = _fit_shared_volume(x_train, targets_train)

        # Efficiency fits on rows with targets > 0.
        pos_mask = targets_train > 0
        x_pos = x_train[pos_mask]
        targets_pos = targets_train[pos_mask]
        receptions_pos = receptions_train[pos_mask]
        ratio_pos = receptions_pos.astype(np.float64) / targets_pos.astype(np.float64)

        ridge_eff = _fit_ridge_efficiency(x_pos, ratio_pos)

        x_trials, y_trials = _expand_to_trials(x_pos, receptions_pos, targets_pos)
        logit_eff = _fit_logit_efficiency(x_trials, y_trials)

        # Eval-year join + prediction.
        eval_feat = features_validated[features_validated["season"] == eval_year]
        eval_ws = ws_wr[ws_wr["season"] == eval_year]
        eval_join = eval_feat.merge(
            eval_ws[["gsis_id", "season", "week", "targets", "receptions"]],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        eval_keep = eval_join[list(_WR_FEATURE_COLUMNS)].notna().all(axis=1)
        eval_join = eval_join.loc[eval_keep]
        x_eval = eval_join[list(_WR_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        eval_targets = eval_join["targets"].to_numpy(dtype=np.int64)
        eval_receptions = eval_join["receptions"].to_numpy(dtype=np.float64)

        mu_targets = volume.predict(x_eval).astype(np.float64)
        pred_ridge = _predict_receptions_ridge(mu_targets, x_eval, ridge_eff)
        pred_logit = _predict_receptions_logit(mu_targets, x_eval, logit_eff)

        actual_buffer.append(eval_receptions)
        ridge_buffer.append(pred_ridge)
        logit_buffer.append(pred_logit)
        year_buffer.append(np.full(eval_receptions.shape, eval_year, dtype=np.int64))

        # Coverage: fraction of eval rows with targets > 0.
        coverage_per_year[eval_year] = (
            float((eval_targets > 0).mean()) if eval_targets.size > 0 else 0.0
        )

    return ProbeResults(
        actual_receptions=np.concatenate(actual_buffer) if actual_buffer else np.array([]),
        pred_ridge=np.concatenate(ridge_buffer) if ridge_buffer else np.array([]),
        pred_logit=np.concatenate(logit_buffer) if logit_buffer else np.array([]),
        year=np.concatenate(year_buffer) if year_buffer else np.array([], dtype=np.int64),
        coverage_per_year=coverage_per_year,
    )


def compute_verdict(
    results: ProbeResults, *, n_bootstrap: int = 1000, seed: int = 42
) -> PerStatVerdict:
    """Pooled paired-bootstrap CI on the receptions Δ-RMSE (logit − ridge).

    The signed residuals fed to paired_bootstrap_rmse_delta are (actual − pred);
    the function computes RMSE on each arm and returns (candidate − incumbent),
    which matches our convention (logit − ridge).
    """
    inc_residuals = results.actual_receptions - results.pred_ridge
    cand_residuals = results.actual_receptions - results.pred_logit
    rmse_delta = paired_bootstrap_rmse_delta(
        inc_residuals, cand_residuals, n_bootstrap=n_bootstrap, seed=seed
    )

    if rmse_delta.hi_95 < 0:
        label: VerdictLabel = "SIGNAL"
    elif rmse_delta.lo_95 > 0:
        label = "REGRESSION"
    else:
        label = "NULL"

    return PerStatVerdict(
        stat=Stat.RECEPTIONS,
        n_paired=int(results.actual_receptions.shape[0]),
        rmse_delta=rmse_delta,
        verdict=label,
    )
```

Note on the `walk_forward_residuals` implementation: the train_join NaN-mask alignment is a touchpoint that can trip on `_x_frame_for_features` dropping rows. The implementation above re-derives the keep mask in-place using the same `_WR_FEATURE_COLUMNS` constant — avoids the double-mask drift bug that PR #29's plan-vs-execution notes flagged.

- [ ] **Step 3.4: Run tests — expect PASS**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_logit_catch_rate_probe.py -v
```

Expected: 14 tests pass (4 Task-1 + 5 Task-2 + 5 Task-3).

- [ ] **Step 3.5: Lint + typecheck**

```bash
../../.venv/Scripts/python.exe -m ruff check src tests
../../.venv/Scripts/python.exe -m ruff format --check src tests
../../.venv/Scripts/python.exe -m mypy src tests
```

Expected: zero violations. If mypy complains about `Literal["SIGNAL", "NULL", "REGRESSION"]` not being a runtime type, narrow with `cast(VerdictLabel, "SIGNAL")` or use the existing pattern from `feature_probe.py`.

- [ ] **Step 3.6: Commit**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  src/projections/backtest/logit_catch_rate_probe.py \
  tests/test_backtest/test_logit_catch_rate_probe.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(probe): walk_forward_residuals + compute_verdict for the catch_rate probe

walk_forward_residuals fits both arms on each train window (shared volume +
ridge_eff incumbent + logit_eff candidate) and emits pooled per-row buffers.
compute_verdict runs paired-bootstrap CI on the residuals delta and maps to
SIGNAL / NULL / REGRESSION per spec §1.3 verdict rule.

Spec: docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md (Task 3).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: CLI driver + CLI smoke

**Files:**
- Create: `scripts/probe_logit_catch_rate.py`
- Create: `tests/test_scripts/test_probe_logit_catch_rate_cli.py`

Scope: argparse driver that wires `walk_forward_residuals` + `compute_verdict` + report writing. Mocked CLI smoke test.

- [ ] **Step 4.1: Write the failing CLI test**

Create `tests/test_scripts/test_probe_logit_catch_rate_cli.py`:

```python
"""CLI smoke for scripts/probe_logit_catch_rate.py.

Mocks walk_forward_residuals to avoid real data; verifies argparse + report
writing.
"""

from __future__ import annotations

import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

import scripts.probe_logit_catch_rate as probe_cli
from projections.backtest.adoption_gate import BootstrapDelta
from projections.backtest.logit_catch_rate_probe import PerStatVerdict, ProbeResults
from projections.schemas import Stat


def _fake_results() -> ProbeResults:
    return ProbeResults(
        actual_receptions=np.array([5.0, 3.0, 7.0], dtype=np.float64),
        pred_ridge=np.array([5.1, 3.2, 6.8], dtype=np.float64),
        pred_logit=np.array([5.0, 3.0, 6.9], dtype=np.float64),
        year=np.array([2021, 2021, 2022], dtype=np.int64),
        coverage_per_year={2021: 0.98, 2022: 0.99},
    )


def _fake_verdict() -> PerStatVerdict:
    return PerStatVerdict(
        stat=Stat.RECEPTIONS,
        n_paired=3,
        rmse_delta=BootstrapDelta(
            point=-0.05, lo_95=-0.08, hi_95=-0.01, n_paired_rows=3, n_bootstrap=1000
        ),
        verdict="SIGNAL",
    )


def test_cli_writes_summary_and_csv(tmp_path: Path) -> None:
    """CLI invocation produces both the summary .md and the .csv report."""
    summary_path = tmp_path / "summary.md"
    csv_path = tmp_path / "deltas.csv"

    fake_features = mock.MagicMock(name="features")
    fake_weekly_stats = mock.MagicMock(name="weekly_stats")

    with (
        mock.patch.object(probe_cli, "_load_inputs", return_value=(fake_features, fake_weekly_stats)),
        mock.patch.object(probe_cli, "walk_forward_residuals", return_value=_fake_results()),
        mock.patch.object(probe_cli, "compute_verdict", return_value=_fake_verdict()),
        mock.patch.object(
            sys,
            "argv",
            [
                "probe_logit_catch_rate",
                "--summary-out",
                str(summary_path),
                "--csv-out",
                str(csv_path),
            ],
        ),
    ):
        probe_cli.main()

    assert summary_path.exists()
    summary = summary_path.read_text(encoding="utf-8")
    assert "SIGNAL" in summary
    assert "n_paired: 3" in summary

    assert csv_path.exists()
    csv = csv_path.read_text(encoding="utf-8")
    # Per-year rows + pooled.
    assert "2021" in csv
    assert "2022" in csv
    assert "pooled" in csv.lower() or "all" in csv.lower()


def test_cli_rejects_unknown_year(tmp_path: Path) -> None:
    """argparse should choke on an out-of-range --eval-years value."""
    with mock.patch.object(
        sys,
        "argv",
        [
            "probe_logit_catch_rate",
            "--eval-years",
            "2099",
            "--summary-out",
            str(tmp_path / "out.md"),
        ],
    ):
        with pytest.raises(SystemExit):
            probe_cli.main()
```

- [ ] **Step 4.2: Run test — expect FAIL**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_scripts/test_probe_logit_catch_rate_cli.py -v
```

Expected: import error on `scripts.probe_logit_catch_rate`.

- [ ] **Step 4.3: Implement the CLI driver**

Create `scripts/probe_logit_catch_rate.py`:

```python
"""CLI driver for the logit catch_rate sub-model probe.

Spec: docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md.

Reads WR features + weekly_stats from disk, runs walk_forward_residuals on
2021-2024 by default, computes the verdict, writes a summary markdown +
per-year CSV.

Usage:
    python scripts/probe_logit_catch_rate.py \\
        --summary-out reports/feature_probe_logit_catch_rate_summary.md \\
        --csv-out reports/feature_probe_logit_catch_rate.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from projections.backtest.logit_catch_rate_probe import (
    PerStatVerdict,
    ProbeResults,
    compute_verdict,
    walk_forward_residuals,
)
from projections.features.cache import read_features
from projections.schemas import Position
from projections.store import read_partition

_DEFAULT_EVAL_YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024)
_VALID_YEARS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
_COVERAGE_THRESHOLD: float = 0.95
_MARGINAL_ZONE_THRESHOLD: float = 0.005  # receptions; per PR #31 retrospective


def _load_inputs(
    *,
    eval_years: Sequence[int],
    features_root: Path,
    raw_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load WR features (for all seasons needed for the eval window) +
    weekly_stats for the union of train and eval years.

    Train span starts at 2018 (matches BaselineModel.fit's lower bound).
    """
    seasons_needed = sorted({*_VALID_YEARS[: _VALID_YEARS.index(max(eval_years)) + 1]})
    feat_parts = [
        read_features(Position.WR, s, features_root=features_root) for s in seasons_needed
    ]
    features = pd.concat(feat_parts, ignore_index=True)

    ws_parts = [
        read_partition(raw_root, "weekly_stats", season=s) for s in seasons_needed
    ]
    weekly_stats = pd.concat(ws_parts, ignore_index=True)
    return features, weekly_stats


def _per_year_breakdown(
    results: ProbeResults, *, n_bootstrap: int, seed: int
) -> pd.DataFrame:
    """One-row-per-year breakdown of Δ-RMSE point + CI."""
    from projections.backtest.adoption_gate import paired_bootstrap_rmse_delta

    rows: list[dict[str, object]] = []
    for year in np.unique(results.year):
        mask = results.year == year
        if mask.sum() < 100:
            rows.append(
                {
                    "year": int(year),
                    "n_paired": int(mask.sum()),
                    "rmse_delta_point": float("nan"),
                    "rmse_delta_lo": float("nan"),
                    "rmse_delta_hi": float("nan"),
                    "coverage": results.coverage_per_year.get(int(year), float("nan")),
                }
            )
            continue
        inc_residuals = results.actual_receptions[mask] - results.pred_ridge[mask]
        cand_residuals = results.actual_receptions[mask] - results.pred_logit[mask]
        delta = paired_bootstrap_rmse_delta(
            inc_residuals, cand_residuals, n_bootstrap=n_bootstrap, seed=seed
        )
        rows.append(
            {
                "year": int(year),
                "n_paired": int(mask.sum()),
                "rmse_delta_point": delta.point,
                "rmse_delta_lo": delta.lo_95,
                "rmse_delta_hi": delta.hi_95,
                "coverage": results.coverage_per_year.get(int(year), float("nan")),
            }
        )
    return pd.DataFrame(rows)


def _write_summary(
    path: Path,
    *,
    verdict: PerStatVerdict,
    results: ProbeResults,
    per_year: pd.DataFrame,
    coverage_threshold: float,
    args: argparse.Namespace,
) -> None:
    """Markdown summary report."""
    lines: list[str] = [
        "# Logit catch_rate Probe — Summary",
        "",
        f"**Spec:** `docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md`",
        f"**Eval years:** {sorted(set(int(y) for y in results.year))}",
        f"**n_bootstrap:** {args.n_bootstrap}, seed: {args.seed}",
        "",
        f"## Verdict: **{verdict.verdict}**",
        "",
        f"- n_paired: {verdict.n_paired}",
        (
            f"- RMSE Δ (logit − ridge): {verdict.rmse_delta.point:+.4f} "
            f"(95% CI [{verdict.rmse_delta.lo_95:+.4f}, {verdict.rmse_delta.hi_95:+.4f}])"
        ),
        "",
    ]
    if abs(verdict.rmse_delta.point) < _MARGINAL_ZONE_THRESHOLD:
        lines.append(
            f"**Magnitude flag:** |Δ| {abs(verdict.rmse_delta.point):.4f} < {_MARGINAL_ZONE_THRESHOLD:.3f} "
            "receptions — in the marginal zone per PR #31's retrospective rule. "
            "Integration go/no-go must weight CI strength against magnitude."
        )
        lines.append("")

    lines.append("## Per-year breakdown")
    lines.append("")
    lines.append(per_year.to_string(index=False))
    lines.append("")

    lines.append("## Coverage")
    lines.append("")
    lines.append(f"Coverage threshold: {coverage_threshold:.2f} (`targets > 0` rate per eval year).")
    lines.append("")
    for year in sorted(results.coverage_per_year):
        rate = results.coverage_per_year[year]
        flag = "" if rate >= coverage_threshold else " — BELOW THRESHOLD"
        lines.append(f"- {year}: {rate:.4f}{flag}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, per_year: pd.DataFrame, verdict: PerStatVerdict) -> None:
    """Long-form CSV: per-year rows + one pooled row."""
    pooled = pd.DataFrame(
        [
            {
                "year": "pooled",
                "n_paired": verdict.n_paired,
                "rmse_delta_point": verdict.rmse_delta.point,
                "rmse_delta_lo": verdict.rmse_delta.lo_95,
                "rmse_delta_hi": verdict.rmse_delta.hi_95,
                "coverage": float("nan"),
            }
        ]
    )
    out = pd.concat([per_year, pooled], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logit catch_rate sub-model probe.")
    parser.add_argument(
        "--eval-years",
        type=int,
        nargs="+",
        choices=_VALID_YEARS,
        default=list(_DEFAULT_EVAL_YEARS),
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path("data/features"),
        help="root dir for the feature cache",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="root dir for the weekly_stats parquet store",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("reports/feature_probe_logit_catch_rate_summary.md"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("reports/feature_probe_logit_catch_rate.csv"),
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=_COVERAGE_THRESHOLD,
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()

    features, weekly_stats = _load_inputs(
        eval_years=args.eval_years,
        features_root=args.features_root,
        raw_root=args.raw_root,
    )
    results = walk_forward_residuals(features, weekly_stats, eval_years=args.eval_years)
    verdict = compute_verdict(results, n_bootstrap=args.n_bootstrap, seed=args.seed)

    per_year = _per_year_breakdown(results, n_bootstrap=args.n_bootstrap, seed=args.seed)
    _write_summary(
        args.summary_out,
        verdict=verdict,
        results=results,
        per_year=per_year,
        coverage_threshold=args.coverage_threshold,
        args=args,
    )
    _write_csv(args.csv_out, per_year, verdict)

    print(f"Verdict: {verdict.verdict}")
    print(
        f"  RMSE Δ {verdict.rmse_delta.point:+.4f} "
        f"(CI [{verdict.rmse_delta.lo_95:+.4f}, {verdict.rmse_delta.hi_95:+.4f}])"
    )
    print(f"  Summary: {args.summary_out}")
    print(f"  CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.4: Run CLI test — expect PASS**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_scripts/test_probe_logit_catch_rate_cli.py -v
```

Expected: 2 tests pass.

- [ ] **Step 4.5: Lint + typecheck**

```bash
../../.venv/Scripts/python.exe -m ruff check src tests scripts
../../.venv/Scripts/python.exe -m ruff format --check src tests scripts
../../.venv/Scripts/python.exe -m mypy src tests
```

Expected: zero violations.

- [ ] **Step 4.6: Commit**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  scripts/probe_logit_catch_rate.py \
  tests/test_scripts/test_probe_logit_catch_rate_cli.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(scripts): probe_logit_catch_rate CLI driver

argparse driver mirroring scripts/probe_target_decomposition.py: loads
WR features + weekly_stats, runs walk_forward_residuals over 2021-2024 by
default, computes verdict via paired-bootstrap CI, writes summary markdown
+ per-year CSV. Magnitude flag fires when |Δ| < 0.005 receptions per
PR #31's retrospective rule.

Spec: docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md (Task 4).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Real-data probe run + report + PM/TODO

**Files:**
- Create: `reports/feature_probe_logit_catch_rate_summary.md` (overwritten by the CLI)
- Create: `reports/feature_probe_logit_catch_rate.csv` (overwritten by the CLI)
- Modify: `project_management.md`
- Modify: `TODO.md`

Scope: operational. Run the CLI against the real feature cache, inspect the verdict, write PM/TODO entries.

- [ ] **Step 5.1: Run the probe from main repo cwd**

The worktree lacks data/raw and data/features (those live in the main repo). Run from main repo cwd so relative paths resolve correctly; invoke the worktree's script directly:

```bash
cd /c/Users/alden/FantasyFootball
.venv/Scripts/python.exe /c/Users/alden/FantasyFootball/.worktrees/feat-probe-logit-catch-rate/scripts/probe_logit_catch_rate.py \
  --summary-out /c/Users/alden/FantasyFootball/.worktrees/feat-probe-logit-catch-rate/reports/feature_probe_logit_catch_rate_summary.md \
  --csv-out /c/Users/alden/FantasyFootball/.worktrees/feat-probe-logit-catch-rate/reports/feature_probe_logit_catch_rate.csv
```

Expected runtime: ~1-3 min (4 years × small ridge + small logit fit). Captures both arms' predictions on real WR data 2021-2024.

If you see a `ConvergenceWarning` from LogisticRegressionCV, the run still succeeds; document the warning in the summary report's plan-vs-execution-deviations section and re-run with `max_iter=2000` only if `pred_logit` looks wrong (sanity check: mean of `pred_logit / mu_targets` should land between 0.55 and 0.75 — the expected catch_rate range).

- [ ] **Step 5.2: Read the summary + verdict**

```bash
cat /c/Users/alden/FantasyFootball/.worktrees/feat-probe-logit-catch-rate/reports/feature_probe_logit_catch_rate_summary.md
```

Map the verdict to the §1.3 spec rule:
- **SIGNAL** (CI strictly negative): recommend integration plan.
- **NULL** (CI brackets zero): recommend closing the catch_rate factor-appropriate direction; next slot is `yards_per_target` factor-appropriate probe (separate cycle).
- **REGRESSION** (CI strictly positive): close the catch_rate factor-appropriate direction in stronger terms.

Note the magnitude flag if it fired.

- [ ] **Step 5.3: Update `project_management.md`**

Add a top-of-file decision-log entry (after the `---` divider following the intro). Read recent entries for tone:

```markdown
## Logit catch_rate Probe — verdict `<VERDICT>` (2026-05-16, on branch `feat/probe-logit-catch-rate`)

**Status:** New probe `src/projections/backtest/logit_catch_rate_probe.py` tests whether replacing the catch_rate efficiency sub-model class from `RidgeCV` on the ratio with predict-time clip to `LogisticRegressionCV` via Bernoulli-trial row expansion lowers per-stat receptions RMSE on WR rows. Spec at `docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md`.

**Verdict:** `<SIGNAL | NULL | REGRESSION>` — RMSE Δ <X.XXXX> receptions (95% CI [<X.XXXX>, <X.XXXX>]), n_paired = <N>. <Magnitude-flag-line-if-fired>.

**Mechanism interpretation:** <one paragraph: if SIGNAL, the logit's proper [0, 1] support beat the Ridge-on-clipped-ratio on receptions; if NULL, the Ridge approximation is good enough on this data; if REGRESSION, the logit fit is worse — possibly due to feature-scale mismatch (Ridge is approximately scale-invariant; LogisticRegression's L2 penalty is scale-dependent).>

**Recommended next direction:** <per verdict above — integration plan / yards_per_target probe / close the factor-appropriate direction>.

See `reports/feature_probe_logit_catch_rate_summary.md` for the full decision log + per-year tables.
```

Fill the bracketed values from Step 5.2.

- [ ] **Step 5.4: Update `TODO.md`**

Add an `**Update 2026-05-16 (logit catch_rate probe, branch `feat/probe-logit-catch-rate`)**:` line under the existing TODO #23 follow-up section (the one with the 2026-05-15 ensemble-decomposed update). Mirror the style of prior `**Update ...**` lines. Cite the verdict + recommended next direction. Close (`✓`) any TODO entry that was waiting on this probe's verdict.

- [ ] **Step 5.5: Final verification**

```bash
cd /c/Users/alden/FantasyFootball/.worktrees/feat-probe-logit-catch-rate
../../.venv/Scripts/python.exe -m pytest -v
../../.venv/Scripts/python.exe -m mypy src tests
../../.venv/Scripts/python.exe -m ruff check src tests scripts
../../.venv/Scripts/python.exe -m ruff format --check src tests scripts
../../.venv/Scripts/python.exe -m pytest -v -k "ingest or store or schemas"
```

All green. Paste concise summaries into the final report.

- [ ] **Step 5.6: Commit reports + PM/TODO**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  reports/feature_probe_logit_catch_rate_summary.md \
  reports/feature_probe_logit_catch_rate.csv \
  project_management.md \
  TODO.md
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
data(probe): logit catch_rate verdict <VERDICT> — RMSE Δ <X.XXXX> receptions

Pooled 4-year walk-forward (2021-2024) on real WR data. Two arms: Ridge-on-
ratio + clip (incumbent, matches current production) vs LogisticRegressionCV
via Bernoulli-trial row expansion (candidate). Per-stat receptions Δ-RMSE
verdict <VERDICT> at 95% CI [<X.XXXX>, <X.XXXX>], n_paired = <N>.

<Optional marginal-zone flag line if fired>

Recommended next direction: <integration plan / yards_per_target probe /
close the factor-appropriate direction>.

Spec: docs/superpowers/specs/2026-05-15-logit-catch-rate-probe-design.md (Task 5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Fill in the bracketed values from the actual verdict at commit time.)

- [ ] **Step 5.7: Push + open PR (pending user authorization)**

The PR creation is a visible/shared action per project conventions. Do NOT push or `gh pr create` without explicit user confirmation. Wait for the orchestrator (or user) to request the push.

When authorized, the invocation:

```bash
git push -u origin feat/probe-logit-catch-rate

gh pr create --title "feat: logit catch_rate sub-model probe (verdict <VERDICT>)" --body "$(cat <<'EOF'
## Summary

- New probe `src/projections/backtest/logit_catch_rate_probe.py` testing factor-appropriate sub-model for `catch_rate`: LogisticRegressionCV via Bernoulli-trial row expansion (candidate) vs RidgeCV on the ratio + clip (incumbent, matches current production).
- Both arms share the same shared-volume RidgeCV on `targets`; only the catch_rate efficiency sub-model class differs.
- Verdict: **<VERDICT>** — RMSE Δ <X.XXXX> receptions (95% CI [<X.XXXX>, <X.XXXX>]); n_paired = <N>.
- <Magnitude flag note if fired>

## Test plan

- [x] All tests pass: `pytest -v`
- [x] `mypy src tests` — zero violations
- [x] `ruff check src tests scripts` + `ruff format --check src tests scripts` — clean
- [x] Integration-seam smoke (`pytest -k "ingest or store or schemas"`): clean

## Reports

- `reports/feature_probe_logit_catch_rate_summary.md` — verdict + per-year + coverage + magnitude flag note
- `reports/feature_probe_logit_catch_rate.csv` — long-form per-year deltas

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review checks (post-plan)

- **Spec coverage:** Tasks 1-5 cover §1.1 (the two-arm probe), §1.3 (verdict rule + coverage), §3.1 (module surface), §3.2 (CLI), §4 (testing), §6 (reports), §7 (estimated scope). Spec's §1.2 (mechanism prior) is informational and doesn't require implementation work. Spec's §1.4 (deferred follow-ups) is explicitly out of scope.
- **Placeholders:** Steps 5.3 / 5.4 / 5.6 / 5.7 use bracketed `<VERDICT>` / `<X.XXXX>` / `<N>` placeholders for the actual verdict numbers — intentional because they're only known after Step 5.1. Step 4.3's argparse `choices=_VALID_YEARS` pins valid years to a known constant.
- **Type consistency:** `ProbeResults` and `PerStatVerdict` defined in Task 3 are referenced in Task 4's CLI imports — names match. `walk_forward_residuals` signature in Task 3 matches Task 4's invocation. `compute_verdict` signature matches.
- **Scope boundaries:** Each task touches ≤ 5 files per CLAUDE.md "phased execution" rule (Task 5 touches 4: 2 reports + PM + TODO). Pre-task venv re-install is a single command, not counted against the file-touch budget.
- **Feature scaling per spec §5 risk #6:** Task 2 wraps `_fit_logit_efficiency` in a sklearn Pipeline of `[StandardScaler, LogisticRegressionCV]`. Scaler is fit on the trial-expanded rows and persisted inside the Pipeline; `_predict_receptions_logit` invokes `Pipeline.predict_proba` which transparently scales eval rows before the logit step. The Ridge arm is intentionally NOT scaled (Ridge's CV-selected alpha is approximately scale-invariant; the existing `target_decomposition_probe.py` doesn't scale either). The probe report's plan-vs-execution-deviations section should note this preprocessing asymmetry between arms.
