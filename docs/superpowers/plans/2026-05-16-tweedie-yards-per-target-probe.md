# Tweedie yards_per_target Sub-Model Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Probe whether replacing the yards_per_target efficiency sub-model class from `RidgeCV` on the ratio + clip(>=0) to `TweedieRegressor(power=1.5, link="log")` with alpha CV-selected lowers per-stat receiving_yards RMSE on WR rows, 2021-2024 walk-forward.

**Architecture:** Two-arm probe in a new pure-numpy/pandas/sklearn module. Both arms share the same `RidgeCV` on `targets`; only the yards_per_target efficiency sub-model class differs. Per-row predictions emitted; pooled paired-bootstrap CI on the receiving_yards Delta-RMSE. No production code touched.

**Tech Stack:** Python, numpy, pandas, sklearn (`RidgeCV`, `TweedieRegressor`, `GridSearchCV`, `StandardScaler`, `Pipeline`, `make_scorer`, `mean_tweedie_deviance`), pandera schemas, pytest, mypy strict, ruff.

**Spec:** `docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md`

---

## Pre-task: re-point venv editable install

The `.venv` at `C:/Users/alden/FantasyFootball/.venv` is editable-installed against a different worktree (e.g., `feat-probe-logit-catch-rate` from PR #39). Any test invocation from inside the new worktree resolves imports against THAT directory, not the current one — silent staleness that wastes ~10 min if missed.

Run this from the new worktree's root before starting Task 1:

```bash
cd C:/Users/alden/FantasyFootball/.worktrees/feat-probe-tweedie-yards-per-target
../../.venv/Scripts/python.exe -m pip install -e . --no-deps
```

Verify:

```bash
../../.venv/Scripts/python.exe -c "from importlib.metadata import distribution; import json; print(json.loads(distribution('projections').read_text('direct_url.json'))['url'])"
```

Expected output: `file:///C:/Users/alden/FantasyFootball/.worktrees/feat-probe-tweedie-yards-per-target`

If the URL still points at any other worktree, re-run the `pip install -e .` from this worktree's root.

---

## File Map

**Create:**
- `src/projections/backtest/tweedie_yards_per_target_probe.py` — probe core: sub-model fit helpers, prediction helpers, walk-forward, verdict
- `scripts/probe_tweedie_yards_per_target.py` — CLI driver
- `tests/test_backtest/test_tweedie_yards_per_target_probe.py` — unit + integration tests for the probe module
- `tests/test_scripts/test_probe_tweedie_yards_per_target_cli.py` — CLI smoke
- `reports/feature_probe_tweedie_yards_per_target_summary.md` — Task 5 output
- `reports/feature_probe_tweedie_yards_per_target.csv` — Task 5 output

**Modify (Task 5 only):**
- `project_management.md` — top-of-file decision-log entry
- `TODO.md` — update entry with probe verdict + next-slot recommendation

---

## Task 1: Shared volume + Ridge efficiency helpers + grid constants

**Files:**
- Create: `src/projections/backtest/tweedie_yards_per_target_probe.py`
- Create: `tests/test_backtest/test_tweedie_yards_per_target_probe.py`

Scope: module skeleton + `_RIDGE_ALPHAS` constant + `_fit_shared_volume` + `_fit_ridge_efficiency` + their unit tests. No Tweedie code yet.

- [ ] **Step 1.1: Re-point venv editable install per pre-task block above.** Confirm with the verify command before proceeding.

- [ ] **Step 1.2: Write the failing tests**

Create `tests/test_backtest/test_tweedie_yards_per_target_probe.py`:

```python
"""Tests for the Tweedie yards_per_target probe.

Spec: docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV

from projections.backtest.tweedie_yards_per_target_probe import (
    _RIDGE_ALPHAS,
    _fit_ridge_efficiency,
    _fit_shared_volume,
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
    # ratio = yards_per_target ~ 8 + 4 * x[:, 0] (unbounded positive)
    ratio = (8.0 + 4.0 * x[:, 0] + rng.normal(0, 0.5, size=150)).astype(np.float64)

    ridge = _fit_ridge_efficiency(x, ratio)

    assert isinstance(ridge, RidgeCV)
    assert ridge.alpha_ in _RIDGE_ALPHAS
    # Coefficient recovery sanity for the unbounded yards-per-target response.
    assert abs(ridge.coef_[0] - 4.0) < 0.5
```

- [ ] **Step 1.3: Run tests — expect ImportError**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_tweedie_yards_per_target_probe.py -v
```

Expected: collection error, `ImportError: cannot import name '_RIDGE_ALPHAS' from 'projections.backtest.tweedie_yards_per_target_probe'`.

- [ ] **Step 1.4: Create the module skeleton + implement shared volume + ridge efficiency**

Create `src/projections/backtest/tweedie_yards_per_target_probe.py`:

```python
"""Tweedie yards_per_target sub-model probe — factor-appropriate efficiency factor.

Two-arm probe comparing a Ridge-on-ratio efficiency sub-model (incumbent;
the recipe used by DecomposedBaselineModel when configured for unbounded
efficiency factors per src/projections/models/decomposed_baseline.py) against
a TweedieRegressor(power=1.5, link="log") fit on the same ratio (candidate).
Per-stat receiving_yards Delta-CV-RMSE, walk-forward eval window 2021-2024,
paired-bootstrap CI on pooled residuals.

Mirrors logit_catch_rate_probe.py's shape; reuses paired_bootstrap_rmse_delta
and BootstrapDelta from adoption_gate.py.

Spec: docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from sklearn.linear_model import RidgeCV

# Same alpha grid as BaselineModel.fit (src/projections/models/baseline.py) and
# logit_catch_rate_probe._RIDGE_ALPHAS so the volume + Ridge arm differ from
# production only in the per-stat dispatch, not the regularization scale.
_RIDGE_ALPHAS: Final[np.ndarray] = np.logspace(-3, 3, 13)


def _fit_shared_volume(x: np.ndarray, targets: np.ndarray) -> RidgeCV:
    """Fit the shared volume sub-model: targets ~ X via RidgeCV.

    Trained on all rows (no targets > 0 filter); zero-target rows are valid
    observations of low-volume players and the volume model must predict them.
    Identical recipe to logit_catch_rate_probe._fit_shared_volume and to
    target_decomposition_probe._fit_decomposed_volume.
    """
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
    ridge.fit(x, targets.astype(np.float64))
    return ridge


def _fit_ridge_efficiency(x_pos: np.ndarray, ratio: np.ndarray) -> RidgeCV:
    """Fit the Ridge-on-ratio efficiency sub-model (incumbent arm).

    Matches the unbounded-efficiency code path in decomposed_baseline.py
    (efficiency_clip_hi = float("inf")): RidgeCV on
    `receiving_yards / targets` (computed only on rows with `targets > 0`).
    """
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
    ridge.fit(x_pos, ratio.astype(np.float64))
    return ridge
```

- [ ] **Step 1.5: Run tests — expect PASS**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_tweedie_yards_per_target_probe.py -v
```

Expected: 2 tests pass.

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
  src/projections/backtest/tweedie_yards_per_target_probe.py \
  tests/test_backtest/test_tweedie_yards_per_target_probe.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(probe): shared volume + Ridge efficiency helpers for tweedie probe

_fit_shared_volume + _fit_ridge_efficiency replicate the existing
target_decomposition_probe / logit_catch_rate_probe recipe (RidgeCV with
the same alpha grid as BaselineModel.fit). Establishes the incumbent arm
of the two-arm Tweedie probe; candidate arm follows in Task 2.

Spec: docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md (Task 1).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Tweedie efficiency fit + predict helpers (both arms)

**Files:**
- Modify: `src/projections/backtest/tweedie_yards_per_target_probe.py` (extend)
- Modify: `tests/test_backtest/test_tweedie_yards_per_target_probe.py` (extend)

Scope: `_fit_tweedie_efficiency` (Pipeline + GridSearchCV) + both predict helpers + their tests. Tests include a Tweedie-generative-fixture coefficient-recovery test and a zero-yards-row handling test.

- [ ] **Step 2.1: Append the failing tests**

Append to `tests/test_backtest/test_tweedie_yards_per_target_probe.py`:

```python
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from projections.backtest.tweedie_yards_per_target_probe import (
    _TWEEDIE_ALPHAS,
    _TWEEDIE_POWER,
    _fit_tweedie_efficiency,
    _predict_yards_ridge,
    _predict_yards_tweedie,
)


def _synthetic_tweedie_fixture(
    rng: np.random.Generator,
    n: int,
    b0: float,
    b1: float,
    *,
    phi: float = 1.0,
    power: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (X, y, true_mu) from a compound-Poisson-Gamma Tweedie with
    a known relationship `mu = exp(b0 + b1 * x)`.

    For Tweedie p in (1, 2): if N ~ Pois(lambda) and each claim G_i iid
    Gamma(alpha, scale), the sum y = sum_{i=1..N} G_i is Tweedie-distributed.
    Closed-form params:
        lambda = mu^(2-p) / (phi * (2-p))
        alpha  = (2-p) / (p-1)
        scale  = phi * (p-1) * mu^(p-1)
    y has a point mass at 0 when N=0 (and continuous positive support otherwise).
    """
    x = rng.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float64)
    mu = np.exp(b0 + b1 * x[:, 0])
    lam = mu ** (2 - power) / (phi * (2 - power))
    alpha = (2 - power) / (power - 1)
    n_claims = rng.poisson(lam)
    y = np.zeros(n, dtype=np.float64)
    nonzero = n_claims > 0
    scale_nonzero = phi * (power - 1) * mu[nonzero] ** (power - 1)
    # rng.gamma's `shape` argument is alpha * n_claims (sum of n Gamma(alpha, scale)
    # is Gamma(n*alpha, scale) when scale matches).
    y[nonzero] = rng.gamma(alpha * n_claims[nonzero], scale_nonzero)
    return x, y, mu


def test_fit_tweedie_efficiency_recovers_true_mu() -> None:
    """Generate (X, y) from a known Tweedie GLM; verify the Pipeline.predict
    recovers true mu within tolerance.
    """
    rng = np.random.default_rng(seed=2031)
    # Larger n for Tweedie (variance is intrinsic to the family).
    x, y, true_mu = _synthetic_tweedie_fixture(rng, n=500, b0=2.0, b1=0.5)

    pipeline = _fit_tweedie_efficiency(x, y)

    assert isinstance(pipeline, Pipeline)
    assert "scaler" in pipeline.named_steps
    assert "gscv" in pipeline.named_steps
    inner = pipeline.named_steps["gscv"]
    assert isinstance(inner, GridSearchCV)
    assert isinstance(inner.best_estimator_, TweedieRegressor)
    assert inner.best_params_["alpha"] in _TWEEDIE_ALPHAS
    assert inner.best_estimator_.power == _TWEEDIE_POWER
    assert inner.best_estimator_.link == "log"

    pred_mu = pipeline.predict(x).astype(np.float64)
    # Tweedie variance is intrinsic; relative MAE up to 30% on average.
    relative_errors = np.abs(pred_mu - true_mu) / np.maximum(true_mu, 1e-6)
    assert (
        float(relative_errors.mean()) < 0.30
    ), f"Tweedie fit relative MAE {float(relative_errors.mean()):.4f} too large"
    # All predictions strictly positive (log link guarantees this).
    assert (pred_mu > 0).all()


def test_fit_tweedie_efficiency_handles_zero_yards_rows() -> None:
    """Tweedie p=1.5 must handle yards_per_target == 0 rows without raising.

    Pins the entire motivation for choosing Tweedie over Gamma (spec §1.4 #8).
    """
    rng = np.random.default_rng(seed=2032)
    n = 300
    x = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float64)
    # ~40% zero rows (mirrors realistic incompletion rate among targets-positive
    # WR-weeks); ~60% positive yards_per_target rows.
    zero_mask = rng.uniform(0, 1, size=n) < 0.4
    y = np.where(
        zero_mask,
        0.0,
        rng.gamma(shape=2.5, scale=4.0, size=n),
    ).astype(np.float64)

    pipeline = _fit_tweedie_efficiency(x, y)

    pred_mu = pipeline.predict(x).astype(np.float64)
    assert (pred_mu > 0).all()
    # Predictions in a reasonable yards_per_target range (mean of positive
    # rows is shape*scale = 10; with 40% zeros the overall mean is ~6).
    assert 1.0 < float(pred_mu.mean()) < 25.0


def test_predict_yards_ridge_clips_negative_to_zero() -> None:
    """Ridge predictions below 0 are clipped before multiplying by volume.

    yards_per_target is bounded below by 0 (no negative yards in a target's
    average); the Ridge incumbent's predict-time clip enforces this.
    """
    rng = np.random.default_rng(seed=2033)
    x_train = rng.uniform(0.0, 1.0, size=(100, 2)).astype(np.float64)
    # ratio = -2 + 4 * x[:, 0]: negative predictions on the low-x test rows,
    # so the >= 0 clip is exercised.
    ratio_train = (-2.0 + 4.0 * x_train[:, 0]).astype(np.float64)
    ridge_eff = _fit_ridge_efficiency(x_train, ratio_train)

    x_eval = np.array([[0.0, 0.5], [0.1, 0.5]], dtype=np.float64)
    mu_targets = np.array([10.0, 8.0], dtype=np.float64)

    pred = _predict_yards_ridge(mu_targets, x_eval, ridge_eff)

    # mu_eff at x=0.0 is ~-2.0 (unclipped) -> clipped to 0 -> yards = 0.
    assert pred[0] == 0.0
    # mu_eff at x=0.1 is ~-1.6 (unclipped) -> clipped to 0 -> yards = 0.
    assert pred[1] == 0.0


def test_predict_yards_tweedie_uses_inverse_log_link() -> None:
    """Tweedie prediction equals mu_targets * pipeline.predict(X)."""
    rng = np.random.default_rng(seed=2034)
    n = 250
    x, y, _ = _synthetic_tweedie_fixture(rng, n=n, b0=1.8, b1=0.4)

    pipeline = _fit_tweedie_efficiency(x, y)

    x_eval = rng.uniform(-1.0, 1.0, size=(10, 1)).astype(np.float64)
    mu_targets = rng.uniform(2.0, 10.0, size=10).astype(np.float64)

    pred = _predict_yards_tweedie(mu_targets, x_eval, pipeline)

    expected = mu_targets * pipeline.predict(x_eval).astype(np.float64)
    assert np.allclose(pred, expected)
    # All Tweedie predictions strictly positive (log link).
    assert (pred > 0).all()
```

- [ ] **Step 2.2: Run tests — expect import errors**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_tweedie_yards_per_target_probe.py -v
```

Expected: collection / import errors on `_TWEEDIE_ALPHAS`, `_TWEEDIE_POWER`, `_fit_tweedie_efficiency`, `_predict_yards_ridge`, `_predict_yards_tweedie`.

- [ ] **Step 2.3: Extend the probe module with Tweedie + predict helpers**

Append to `src/projections/backtest/tweedie_yards_per_target_probe.py`:

```python
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import make_scorer, mean_tweedie_deviance
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Tweedie alpha grid: 7 points spanning 6 orders of magnitude. Ridge uses 13
# because RidgeCV is fast; Tweedie fits are ~5-10x slower per fit (iterative
# Newton/LBFGS on the GLM likelihood), so we trade resolution for runtime.
_TWEEDIE_ALPHAS: Final[tuple[float, ...]] = tuple(
    float(a) for a in np.logspace(-3, 3, 7)
)

# Tweedie variance power. p=1.5 is the standard compound-Poisson-Gamma default
# (point mass at 0 + continuous positive support), matching yards_per_target's
# distribution shape. Fixed (not CV-searched) per spec §1.4 #7.
_TWEEDIE_POWER: Final[float] = 1.5


def _fit_tweedie_efficiency(x_pos: np.ndarray, ratio: np.ndarray) -> Pipeline:
    """Fit the Tweedie GLM efficiency sub-model (candidate arm).

    Wraps StandardScaler + GridSearchCV(TweedieRegressor) in a sklearn Pipeline.

    Why a Pipeline:
    - StandardScaler upstream: TweedieRegressor's L2 penalty is scale-dependent
      (whereas Ridge's CV-selected alpha is approximately scale-invariant);
      scaling stops the regularization scale from confounding the sub-model-
      class comparison. The scaler is fit on the full training rows here (same
      pattern as PR #39's logit probe); the inner-CV-fold leakage on a stable
      StandardScaler mean/std on ~5K-8K rows is negligible.
    - GridSearchCV downstream: TweedieRegressor lacks a built-in CV variant
      (no TweedieRegressorCV); GridSearchCV with `refit=True` produces a final
      estimator trained on the full train fold at the CV-selected alpha. The
      inner 5-fold CV is on the training fold only — no leakage with the outer
      walk-forward eval-year split.

    Scoring: mean_tweedie_deviance with matching power=1.5 (deviance is loss-
    style, so `greater_is_better=False`). This is the canonical Tweedie GLM
    fit objective.

    Solver: TweedieRegressor's default is `lbfgs`; max_iter=200 (sklearn
    default 100) for safety against convergence warnings on rows with
    extreme features.
    """
    scorer = make_scorer(
        mean_tweedie_deviance, power=_TWEEDIE_POWER, greater_is_better=False
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gscv",
                GridSearchCV(
                    estimator=TweedieRegressor(
                        power=_TWEEDIE_POWER,
                        link="log",
                        max_iter=200,
                    ),
                    param_grid={"alpha": list(_TWEEDIE_ALPHAS)},
                    cv=5,
                    scoring=scorer,
                    refit=True,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(x_pos, ratio.astype(np.float64))
    return pipeline


def _predict_yards_ridge(
    mu_targets: np.ndarray, x_eval: np.ndarray, ridge_eff: RidgeCV
) -> np.ndarray:
    """Incumbent-arm receiving_yards prediction: mu_targets * clip(mu_ratio, 0, +inf).

    Matches the unbounded-efficiency predict path in decomposed_baseline.py
    (efficiency_clip_hi = float("inf")): the >=0 floor still applies because
    yards_per_target cannot be negative; the upper clip is a no-op.
    """
    mu_ratio = ridge_eff.predict(x_eval).astype(np.float64)
    mu_ratio_clipped = np.maximum(mu_ratio, 0.0)
    return mu_targets * mu_ratio_clipped


def _predict_yards_tweedie(
    mu_targets: np.ndarray, x_eval: np.ndarray, tweedie_eff: Pipeline
) -> np.ndarray:
    """Candidate-arm receiving_yards prediction: mu_targets * exp(scaled X @ beta).

    Pipeline.predict applies the fitted StandardScaler to x_eval, then the
    GridSearchCV.best_estimator_ (a refit TweedieRegressor with the CV-selected
    alpha) applies the inverse-log link (`exp(scaled_X @ beta)`). No manual
    exp() needed.
    """
    mu_ratio = tweedie_eff.predict(x_eval).astype(np.float64)
    return mu_targets * mu_ratio
```

- [ ] **Step 2.4: Run tests — expect PASS**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_tweedie_yards_per_target_probe.py -v
```

Expected: 6 tests pass (2 Task-1 + 4 Task-2). If any Tweedie test produces a `ConvergenceWarning`, the test still passes but verify `max_iter=200` is sufficient (it is for synthetic-fixture row counts; a warning on real data is acceptable as long as predictions look sane).

- [ ] **Step 2.5: Lint + typecheck**

```bash
../../.venv/Scripts/python.exe -m ruff check src tests
../../.venv/Scripts/python.exe -m ruff format --check src tests
../../.venv/Scripts/python.exe -m mypy src tests
```

Expected: zero violations. If mypy complains about `make_scorer` or `mean_tweedie_deviance` returning `Any`, narrow with `cast` if needed, but sklearn's stubs typically handle these cleanly.

- [ ] **Step 2.6: Commit**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  src/projections/backtest/tweedie_yards_per_target_probe.py \
  tests/test_backtest/test_tweedie_yards_per_target_probe.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(probe): _fit_tweedie_efficiency + _predict_* helpers for both arms

_fit_tweedie_efficiency wraps StandardScaler + GridSearchCV(TweedieRegressor,
power=1.5, link=log) in a Pipeline. mean_tweedie_deviance(power=1.5) scoring;
7-point alpha grid; cv=5 inner. Coefficient-recovery test on a synthetic
compound-Poisson-Gamma fixture pins that the candidate path actually fits a
log-link Tweedie. Zero-yards-row test pins that p=1.5 handles the point mass
at zero natively (the entire motivation for Tweedie over Gamma).

_predict_yards_ridge applies the >=0 clip (Ridge incumbent).
_predict_yards_tweedie composes mu_targets with Pipeline.predict (which
applies StandardScaler.transform then inverse-log-link via TweedieRegressor).

Spec: docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md (Task 2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Walk-forward + verdict

**Files:**
- Modify: `src/projections/backtest/tweedie_yards_per_target_probe.py` (extend)
- Modify: `tests/test_backtest/test_tweedie_yards_per_target_probe.py` (extend)

Scope: full walk-forward harness that fits both arms per train window and produces pooled per-row residuals; the verdict mapping.

- [ ] **Step 3.1: Append the failing tests**

Append to `tests/test_backtest/test_tweedie_yards_per_target_probe.py`:

```python
import pandas as pd

from projections.backtest.tweedie_yards_per_target_probe import (
    PerStatVerdict,
    ProbeResults,
    compute_verdict,
    walk_forward_residuals,
)


def _synthetic_wr_inputs(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small synthetic WR (features, weekly_stats) pair for walk-forward
    integration testing.

    4 seasons * 4 weeks * 8 players = 128 rows. Features uniform random; truth
    uses a known yards-per-target generative model with targets correlated with
    one feature so the volume_ridge has signal.
    """
    from projections.models import POSITION_DISPATCH
    from projections.schemas import Position, WeeklyStatsSchema, _PYARROW_STR

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
    # Yards-per-target: log-link from target_lambda (so volume signal carries
    # through to yards). ~30% zero point mass.
    true_ypt = np.exp(1.5 + 0.05 * target_lambda)
    zero_mask = rng.uniform(0, 1, size=n) < 0.30
    ypt_realized = np.where(
        zero_mask, 0.0, rng.gamma(shape=2.0, scale=true_ypt / 2.0)
    )
    receiving_yards = np.maximum(0.0, ypt_realized * targets).astype(np.float64)
    receptions = np.minimum(
        targets,
        np.maximum(
            0,
            (receiving_yards / np.maximum(11.0, 1.0)).astype(np.int64),
        ),
    ).astype(np.int64)
    ws["targets"] = targets
    ws["receptions"] = receptions
    ws["receiving_yards"] = receiving_yards
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
    """Both ridge and tweedie prediction buffers are populated and length-matched
    on every eval year.
    """
    features, weekly_stats = _synthetic_wr_inputs()
    results = walk_forward_residuals(
        features, weekly_stats, eval_years=(2020, 2021)
    )

    assert isinstance(results, ProbeResults)
    assert len(results.actual_yards) > 0
    assert results.actual_yards.shape == results.pred_ridge.shape
    assert results.actual_yards.shape == results.pred_tweedie.shape
    assert results.year.shape == results.actual_yards.shape
    assert set(results.coverage_per_year.keys()) == {2020, 2021}
    # Ridge predictions >= 0 (clipped); Tweedie predictions > 0 (log link).
    assert (results.pred_ridge >= 0).all()
    assert (results.pred_tweedie > 0).all()


def test_walk_forward_residuals_arms_differ_on_some_rows() -> None:
    """The two arms must not produce identical predictions everywhere. A no-op
    fall-through to the ridge arm would be a silent bug.
    """
    features, weekly_stats = _synthetic_wr_inputs(seed=12345)
    results = walk_forward_residuals(features, weekly_stats, eval_years=(2021,))

    n_different = int(np.sum(np.abs(results.pred_ridge - results.pred_tweedie) > 1e-6))
    assert n_different > 0, "ridge and tweedie arms produced identical predictions"


def test_compute_verdict_signal_when_ci_strictly_negative() -> None:
    """Synthetic ProbeResults where tweedie clearly beats ridge -> SIGNAL."""
    rng = np.random.default_rng(seed=2041)
    n = 500
    actual = rng.uniform(0.0, 100.0, size=n)
    # ridge has 5-yard systematic bias; tweedie is unbiased.
    pred_ridge = actual + 5.0 + rng.normal(0, 1.0, size=n)
    pred_tweedie = actual + rng.normal(0, 1.0, size=n)
    year = np.full(n, 2024, dtype=np.int64)
    results = ProbeResults(
        actual_yards=actual,
        pred_ridge=pred_ridge,
        pred_tweedie=pred_tweedie,
        year=year,
        coverage_per_year={2024: 1.0},
    )

    verdict = compute_verdict(results, n_bootstrap=200, seed=42)

    assert isinstance(verdict, PerStatVerdict)
    assert verdict.verdict == "SIGNAL"
    assert verdict.rmse_delta.hi_95 < 0


def test_compute_verdict_null_when_ci_brackets_zero() -> None:
    """Random noise -> NULL."""
    rng = np.random.default_rng(seed=2042)
    n = 300
    actual = rng.uniform(0.0, 100.0, size=n)
    pred_ridge = actual + rng.normal(0, 5.0, size=n)
    pred_tweedie = actual + rng.normal(0, 5.0, size=n)
    year = np.full(n, 2024, dtype=np.int64)
    results = ProbeResults(
        actual_yards=actual,
        pred_ridge=pred_ridge,
        pred_tweedie=pred_tweedie,
        year=year,
        coverage_per_year={2024: 1.0},
    )

    verdict = compute_verdict(results, n_bootstrap=200, seed=42)

    assert verdict.verdict == "NULL"
    assert verdict.rmse_delta.lo_95 < 0 < verdict.rmse_delta.hi_95


def test_compute_verdict_regression_when_ci_strictly_positive() -> None:
    """Tweedie systematically worse than ridge -> REGRESSION."""
    rng = np.random.default_rng(seed=2043)
    n = 500
    actual = rng.uniform(0.0, 100.0, size=n)
    pred_ridge = actual + rng.normal(0, 1.0, size=n)
    pred_tweedie = actual + 5.0 + rng.normal(0, 1.0, size=n)
    year = np.full(n, 2024, dtype=np.int64)
    results = ProbeResults(
        actual_yards=actual,
        pred_ridge=pred_ridge,
        pred_tweedie=pred_tweedie,
        year=year,
        coverage_per_year={2024: 1.0},
    )

    verdict = compute_verdict(results, n_bootstrap=200, seed=42)

    assert verdict.verdict == "REGRESSION"
    assert verdict.rmse_delta.lo_95 > 0
```

- [ ] **Step 3.2: Run tests — expect import errors**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_tweedie_yards_per_target_probe.py -v
```

Expected: collection / import errors on `walk_forward_residuals`, `ProbeResults`, `compute_verdict`, `PerStatVerdict`.

- [ ] **Step 3.3: Extend the module with walk-forward + verdict**

Append to `src/projections/backtest/tweedie_yards_per_target_probe.py`:

```python
from collections.abc import Iterable
from dataclasses import dataclass
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
        actual_yards: (N,) ground-truth receiving_yards (float64).
        pred_ridge: (N,) incumbent-arm receiving_yards predictions.
        pred_tweedie: (N,) candidate-arm receiving_yards predictions.
        year: (N,) eval year per row (int64).
        coverage_per_year: per-eval-year fraction of WR rows with targets > 0.
    """

    actual_yards: np.ndarray
    pred_ridge: np.ndarray
    pred_tweedie: np.ndarray
    year: np.ndarray
    coverage_per_year: dict[int, float]


@dataclass(slots=True, frozen=True)
class PerStatVerdict:
    """Per-stat verdict on the receiving_yards Delta-RMSE (tweedie - ridge)."""

    stat: Stat
    n_paired: int
    rmse_delta: BootstrapDelta
    verdict: VerdictLabel


def walk_forward_residuals(
    features: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    eval_years: Iterable[int],
) -> ProbeResults:
    """For each eval year, train both arms on prior seasons, predict on the
    eval year, collect per-row residuals.

    Spec: probe-design §3.1 walk_forward_residuals.
    """
    from projections.models.baseline import _WR_FEATURE_COLUMNS

    eval_years_list = sorted(int(y) for y in eval_years)
    actual_buffer: list[np.ndarray] = []
    ridge_buffer: list[np.ndarray] = []
    tweedie_buffer: list[np.ndarray] = []
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

        # Train-window join.
        train_feat = features_validated[
            features_validated["season"].isin(train_seasons)
        ]
        train_ws = ws_wr[ws_wr["season"].isin(train_seasons)]
        train_join = train_feat.merge(
            train_ws[["gsis_id", "season", "week", "targets", "receiving_yards"]],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        keep_mask = train_join[list(_WR_FEATURE_COLUMNS)].notna().all(axis=1)
        train_join = train_join.loc[keep_mask]
        x_train = train_join[list(_WR_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        targets_train = train_join["targets"].to_numpy(dtype=np.int64)
        yards_train = train_join["receiving_yards"].to_numpy(dtype=np.float64)

        # Shared volume fit (on all rows, no targets > 0 filter).
        volume = _fit_shared_volume(x_train, targets_train)

        # Efficiency fits on rows with targets > 0.
        pos_mask = targets_train > 0
        x_pos = x_train[pos_mask]
        targets_pos = targets_train[pos_mask].astype(np.float64)
        yards_pos = yards_train[pos_mask]
        ratio_pos = yards_pos / targets_pos

        ridge_eff = _fit_ridge_efficiency(x_pos, ratio_pos)
        tweedie_eff = _fit_tweedie_efficiency(x_pos, ratio_pos)

        # Eval-year join + prediction.
        eval_feat = features_validated[features_validated["season"] == eval_year]
        eval_ws = ws_wr[ws_wr["season"] == eval_year]
        eval_join = eval_feat.merge(
            eval_ws[["gsis_id", "season", "week", "targets", "receiving_yards"]],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        eval_keep = eval_join[list(_WR_FEATURE_COLUMNS)].notna().all(axis=1)
        eval_join = eval_join.loc[eval_keep]
        x_eval = eval_join[list(_WR_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        eval_targets = eval_join["targets"].to_numpy(dtype=np.int64)
        eval_yards = eval_join["receiving_yards"].to_numpy(dtype=np.float64)

        mu_targets = volume.predict(x_eval).astype(np.float64)
        pred_ridge = _predict_yards_ridge(mu_targets, x_eval, ridge_eff)
        pred_tweedie = _predict_yards_tweedie(mu_targets, x_eval, tweedie_eff)

        actual_buffer.append(eval_yards)
        ridge_buffer.append(pred_ridge)
        tweedie_buffer.append(pred_tweedie)
        year_buffer.append(
            np.full(eval_yards.shape, eval_year, dtype=np.int64)
        )

        # Coverage: fraction of eval rows with targets > 0.
        coverage_per_year[eval_year] = (
            float((eval_targets > 0).mean()) if eval_targets.size > 0 else 0.0
        )

    return ProbeResults(
        actual_yards=np.concatenate(actual_buffer) if actual_buffer else np.array([]),
        pred_ridge=np.concatenate(ridge_buffer) if ridge_buffer else np.array([]),
        pred_tweedie=(
            np.concatenate(tweedie_buffer) if tweedie_buffer else np.array([])
        ),
        year=(
            np.concatenate(year_buffer) if year_buffer else np.array([], dtype=np.int64)
        ),
        coverage_per_year=coverage_per_year,
    )


def compute_verdict(
    results: ProbeResults, *, n_bootstrap: int = 1000, seed: int = 42
) -> PerStatVerdict:
    """Pooled paired-bootstrap CI on receiving_yards Delta-RMSE (tweedie - ridge).

    Signed residuals are (actual - pred); paired_bootstrap_rmse_delta computes
    RMSE on each arm and returns (candidate - incumbent), matching our
    convention (tweedie - ridge).
    """
    inc_residuals = results.actual_yards - results.pred_ridge
    cand_residuals = results.actual_yards - results.pred_tweedie
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
        stat=Stat.RECEIVING_YARDS,
        n_paired=int(results.actual_yards.shape[0]),
        rmse_delta=rmse_delta,
        verdict=label,
    )
```

- [ ] **Step 3.4: Run tests — expect PASS**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_tweedie_yards_per_target_probe.py -v
```

Expected: 11 tests pass (2 Task-1 + 4 Task-2 + 5 Task-3).

- [ ] **Step 3.5: Lint + typecheck**

```bash
../../.venv/Scripts/python.exe -m ruff check src tests
../../.venv/Scripts/python.exe -m ruff format --check src tests
../../.venv/Scripts/python.exe -m mypy src tests
```

Expected: zero violations. If mypy complains about `Literal["SIGNAL", "NULL", "REGRESSION"]` not being a runtime type, narrow with `cast(VerdictLabel, "SIGNAL")` or use the same pattern as `logit_catch_rate_probe.py`.

- [ ] **Step 3.6: Commit**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  src/projections/backtest/tweedie_yards_per_target_probe.py \
  tests/test_backtest/test_tweedie_yards_per_target_probe.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(probe): walk_forward_residuals + compute_verdict for tweedie probe

walk_forward_residuals fits both arms on each train window (shared volume +
ridge_eff incumbent + tweedie_eff candidate) and emits pooled per-row
receiving_yards prediction buffers. compute_verdict runs paired-bootstrap CI
on the residuals delta (tweedie - ridge) and maps to SIGNAL / NULL / REGRESSION
per spec §1.3 verdict rule.

Spec: docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md (Task 3).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: CLI driver + CLI smoke

**Files:**
- Create: `scripts/probe_tweedie_yards_per_target.py`
- Create: `tests/test_scripts/test_probe_tweedie_yards_per_target_cli.py`

Scope: argparse driver that wires `walk_forward_residuals` + `compute_verdict` + report writing. Mocked CLI smoke test.

- [ ] **Step 4.1: Write the failing CLI test**

Create `tests/test_scripts/test_probe_tweedie_yards_per_target_cli.py`:

```python
"""CLI smoke for scripts/probe_tweedie_yards_per_target.py.

Mocks walk_forward_residuals to avoid real data; verifies argparse + report
writing.
"""

from __future__ import annotations

import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

import scripts.probe_tweedie_yards_per_target as probe_cli
from projections.backtest.adoption_gate import BootstrapDelta
from projections.backtest.tweedie_yards_per_target_probe import (
    PerStatVerdict,
    ProbeResults,
)
from projections.schemas import Stat


def _fake_results() -> ProbeResults:
    return ProbeResults(
        actual_yards=np.array([55.0, 30.0, 72.0], dtype=np.float64),
        pred_ridge=np.array([55.1, 31.2, 70.8], dtype=np.float64),
        pred_tweedie=np.array([54.9, 30.5, 71.4], dtype=np.float64),
        year=np.array([2021, 2021, 2022], dtype=np.int64),
        coverage_per_year={2021: 0.98, 2022: 0.99},
    )


def _fake_verdict() -> PerStatVerdict:
    return PerStatVerdict(
        stat=Stat.RECEIVING_YARDS,
        n_paired=3,
        rmse_delta=BootstrapDelta(
            point=-0.5,
            lo_95=-0.8,
            hi_95=-0.1,
            n_paired_rows=3,
            n_bootstrap=1000,
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
        mock.patch.object(
            probe_cli,
            "_load_inputs",
            return_value=(fake_features, fake_weekly_stats),
        ),
        mock.patch.object(
            probe_cli, "walk_forward_residuals", return_value=_fake_results()
        ),
        mock.patch.object(probe_cli, "compute_verdict", return_value=_fake_verdict()),
        mock.patch.object(
            sys,
            "argv",
            [
                "probe_tweedie_yards_per_target",
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
    assert "2021" in csv
    assert "2022" in csv
    assert "pooled" in csv.lower() or "all" in csv.lower()


def test_cli_rejects_unknown_year(tmp_path: Path) -> None:
    """argparse should choke on an out-of-range --eval-years value."""
    with mock.patch.object(
        sys,
        "argv",
        [
            "probe_tweedie_yards_per_target",
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
../../.venv/Scripts/python.exe -m pytest tests/test_scripts/test_probe_tweedie_yards_per_target_cli.py -v
```

Expected: import error on `scripts.probe_tweedie_yards_per_target`.

- [ ] **Step 4.3: Implement the CLI driver**

Create `scripts/probe_tweedie_yards_per_target.py`:

```python
"""CLI driver for the Tweedie yards_per_target sub-model probe.

Spec: docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md.

Reads WR features + weekly_stats from disk, runs walk_forward_residuals on
2021-2024 by default, computes the verdict, writes a summary markdown +
per-year CSV.

Usage:
    python scripts/probe_tweedie_yards_per_target.py \\
        --summary-out reports/feature_probe_tweedie_yards_per_target_summary.md \\
        --csv-out reports/feature_probe_tweedie_yards_per_target.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from projections.backtest.tweedie_yards_per_target_probe import (
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

# Marginal-zone threshold on receiving_yards: 0.05 yards is the per-stat
# equivalent of PR #31's 0.005 fpts composite-fpts threshold, given PPR yards
# coefficient = 0.1 fpts/yard. ASCII text only in stdout/file output to avoid
# Windows cp1252 encoding crashes (spec §5 risk #8).
_MARGINAL_ZONE_THRESHOLD: float = 0.05


def _load_inputs(
    *,
    eval_years: Sequence[int],
    features_root: Path,
    raw_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load WR features + weekly_stats for the seasons needed by the walk-forward.

    Train span starts at 2018 (matches BaselineModel.fit's lower bound).
    """
    seasons_needed = sorted(
        {*_VALID_YEARS[: _VALID_YEARS.index(max(eval_years)) + 1]}
    )
    feat_parts = [
        read_features(Position.WR, s, features_root=features_root)
        for s in seasons_needed
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
    """One-row-per-year breakdown of Delta-RMSE point + CI."""
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
                    "coverage": results.coverage_per_year.get(
                        int(year), float("nan")
                    ),
                }
            )
            continue
        inc_residuals = results.actual_yards[mask] - results.pred_ridge[mask]
        cand_residuals = results.actual_yards[mask] - results.pred_tweedie[mask]
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
    """Markdown summary report. ASCII-only stdout for Windows cp1252 safety."""
    lines: list[str] = [
        "# Tweedie yards_per_target Probe -- Summary",
        "",
        f"**Spec:** `docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md`",
        f"**Eval years:** {sorted(set(int(y) for y in results.year))}",
        f"**n_bootstrap:** {args.n_bootstrap}, seed: {args.seed}",
        "",
        f"## Verdict: **{verdict.verdict}**",
        "",
        f"- n_paired: {verdict.n_paired}",
        (
            f"- RMSE delta (tweedie - ridge): {verdict.rmse_delta.point:+.4f} yards "
            f"(95% CI [{verdict.rmse_delta.lo_95:+.4f}, "
            f"{verdict.rmse_delta.hi_95:+.4f}])"
        ),
        (
            f"- Composite-fpts equivalent (yards * 0.1): "
            f"{verdict.rmse_delta.point * 0.1:+.4f} fpts"
        ),
        "",
    ]
    if abs(verdict.rmse_delta.point) < _MARGINAL_ZONE_THRESHOLD:
        lines.append(
            f"**Magnitude flag:** |delta| {abs(verdict.rmse_delta.point):.4f} < "
            f"{_MARGINAL_ZONE_THRESHOLD:.3f} yards "
            f"(|delta_fpts| < 0.005) -- in the marginal zone per PR #31's "
            "retrospective rule. Integration go/no-go must weight CI strength "
            "against magnitude."
        )
        lines.append("")

    lines.append("## Mechanism caveat")
    lines.append("")
    lines.append(
        "Incumbent arm is Ridge-decomp (a probe-internal construction), NOT "
        "current production. Current production for receiving_yards is direct "
        "RidgeCV (via `ensemble-decomposed`, which decomposes Stat.RECEPTIONS "
        "only per PR #36/#38). A SIGNAL verdict here does NOT imply "
        "Tweedie-decomp beats current production; that comparison is the "
        "integration adoption-gate's question on a separate cycle."
    )
    lines.append("")

    lines.append("## Per-year breakdown")
    lines.append("")
    lines.append(per_year.to_string(index=False))
    lines.append("")

    lines.append("## Coverage")
    lines.append("")
    lines.append(
        f"Coverage threshold: {coverage_threshold:.2f} (`targets > 0` rate per eval year)."
    )
    lines.append("")
    for year in sorted(results.coverage_per_year):
        rate = results.coverage_per_year[year]
        flag = "" if rate >= coverage_threshold else " -- BELOW THRESHOLD"
        lines.append(f"- {year}: {rate:.4f}{flag}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(
    path: Path, per_year: pd.DataFrame, verdict: PerStatVerdict
) -> None:
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
    parser = argparse.ArgumentParser(
        description="Tweedie yards_per_target sub-model probe."
    )
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
        default=Path("reports/feature_probe_tweedie_yards_per_target_summary.md"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("reports/feature_probe_tweedie_yards_per_target.csv"),
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
    results = walk_forward_residuals(
        features, weekly_stats, eval_years=args.eval_years
    )
    verdict = compute_verdict(
        results, n_bootstrap=args.n_bootstrap, seed=args.seed
    )

    per_year = _per_year_breakdown(
        results, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    _write_summary(
        args.summary_out,
        verdict=verdict,
        results=results,
        per_year=per_year,
        coverage_threshold=args.coverage_threshold,
        args=args,
    )
    _write_csv(args.csv_out, per_year, verdict)

    # ASCII-only stdout; Windows cp1252 crashed on the catch_rate probe's
    # Delta symbol per PR #39's follow-up flag.
    print(f"Verdict: {verdict.verdict}")
    print(
        f"  RMSE delta {verdict.rmse_delta.point:+.4f} yards "
        f"(CI [{verdict.rmse_delta.lo_95:+.4f}, {verdict.rmse_delta.hi_95:+.4f}])"
    )
    print(f"  Summary: {args.summary_out}")
    print(f"  CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.4: Run CLI test — expect PASS**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_scripts/test_probe_tweedie_yards_per_target_cli.py -v
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
  scripts/probe_tweedie_yards_per_target.py \
  tests/test_scripts/test_probe_tweedie_yards_per_target_cli.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(scripts): probe_tweedie_yards_per_target CLI driver

argparse driver mirroring scripts/probe_logit_catch_rate.py: loads WR features
+ weekly_stats, runs walk_forward_residuals over 2021-2024 by default,
computes verdict via paired-bootstrap CI, writes summary markdown + per-year
CSV. Magnitude flag fires when |delta_yards| < 0.05 (composite-fpts < 0.005)
per PR #31's retrospective rule.

ASCII-only stdout (Windows cp1252 crashed on the catch_rate probe's Delta
symbol per PR #39 follow-up flag); reports written with encoding="utf-8".
Summary includes the spec §5 risk #6 mechanism caveat (incumbent arm is NOT
current production).

Spec: docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md (Task 4).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Real-data probe run + report + PM/TODO

**Files:**
- Create: `reports/feature_probe_tweedie_yards_per_target_summary.md` (overwritten by the CLI)
- Create: `reports/feature_probe_tweedie_yards_per_target.csv` (overwritten by the CLI)
- Modify: `project_management.md`
- Modify: `TODO.md`

Scope: operational. Run the CLI against the real feature cache, inspect the verdict, write PM/TODO entries.

- [ ] **Step 5.1: Run the probe from main repo cwd**

The worktree lacks `data/raw` and `data/features` (those live in the main repo). Run from main repo cwd so relative paths resolve correctly; invoke the worktree's script directly:

```bash
cd /c/Users/alden/FantasyFootball
.venv/Scripts/python.exe /c/Users/alden/FantasyFootball/.worktrees/feat-probe-tweedie-yards-per-target/scripts/probe_tweedie_yards_per_target.py \
  --summary-out /c/Users/alden/FantasyFootball/.worktrees/feat-probe-tweedie-yards-per-target/reports/feature_probe_tweedie_yards_per_target_summary.md \
  --csv-out /c/Users/alden/FantasyFootball/.worktrees/feat-probe-tweedie-yards-per-target/reports/feature_probe_tweedie_yards_per_target.csv
```

Expected runtime: ~5-15 min (4 years x small ridge + 7-point alpha grid x 5-fold CV TweedieRegressor fits on ~6K-8K rows). Captures both arms' predictions on real WR data 2021-2024.

If you see a `ConvergenceWarning` from TweedieRegressor, the run still succeeds; document the warning in the summary report's plan-vs-execution-deviations section and re-run with `max_iter=400` only if `pred_tweedie` looks wrong (sanity check: mean of `pred_tweedie / mu_targets` should land between 5.0 and 10.0 yards-per-target — the expected WR range).

- [ ] **Step 5.2: Read the summary + verdict**

```bash
cat /c/Users/alden/FantasyFootball/.worktrees/feat-probe-tweedie-yards-per-target/reports/feature_probe_tweedie_yards_per_target_summary.md
```

Map the verdict to the §1.3 spec rule:
- **SIGNAL** (CI strictly negative): recommend integration plan (separate adoption gate vs current production).
- **NULL** (CI brackets zero): recommend closing the yards_per_target factor-appropriate direction; next slot is `td_rate_per_target` factor-appropriate probe (Poisson / logistic; separate cycle).
- **REGRESSION** (CI strictly positive): close the yards_per_target factor-appropriate direction in stronger terms; Tweedie with CV-selected power is the only remaining sub-direction.

Note the magnitude flag if it fired. Note the mechanism caveat in the report (incumbent NOT production).

- [ ] **Step 5.3: Update `project_management.md`**

Add a top-of-file decision-log entry (after the `---` divider following the intro). Read recent entries for tone; the catch_rate probe entry (PR #39) is the closest precedent.

```markdown
## Tweedie yards_per_target Probe -- verdict `<VERDICT>` (2026-05-16, on branch `feat/probe-tweedie-yards-per-target`)

**Status:** New probe `src/projections/backtest/tweedie_yards_per_target_probe.py` tests whether replacing the yards_per_target efficiency sub-model class from `RidgeCV` on the ratio + clip(>=0) to `TweedieRegressor(power=1.5, link="log")` with alpha CV-selected lowers per-stat receiving_yards RMSE on WR rows. Spec at `docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md`.

**Verdict:** `<SIGNAL | NULL | REGRESSION>` -- RMSE delta <+/-X.XXXX> yards (95% CI [<+/-X.XXXX>, <+/-X.XXXX>]), n_paired = <N>. Composite-fpts equivalent <+/-0.XXXX> fpts. <Magnitude-flag-line-if-fired>.

**Mechanism interpretation:** <one paragraph: if SIGNAL, the Tweedie log-link's proper compound-Poisson-Gamma shape beat Ridge-on-clipped-ratio on yards_per_target; if NULL, Ridge approximates the Tweedie mean well enough on this data; if REGRESSION, the Tweedie fit is worse -- possibly due to feature-scale interaction with the GLM L2 penalty, or because Tweedie's variance assumption mu^1.5 doesn't match the empirical heteroscedasticity.>

**Mechanism caveat:** Incumbent arm (Ridge-decomp) is NOT current production for receiving_yards; production is direct RidgeCV via `ensemble-decomposed` (which decomposes Stat.RECEPTIONS only per PR #36/#38). SIGNAL verdict at this gate does NOT imply Tweedie-decomp beats current production; that comparison is the integration adoption-gate's question.

**Recommended next direction:** <per verdict above: if SIGNAL, integration plan with new adoption gate vs production ensemble-decomposed; if NULL, close yards_per_target factor-appropriate slot and name td_rate_per_target as next factor; if REGRESSION, close yards_per_target strongly, Tweedie-power-CV is the only follow-up>.

See `reports/feature_probe_tweedie_yards_per_target_summary.md` for the full decision log + per-year tables + coverage + magnitude flag.
```

Fill the bracketed values from Step 5.2.

- [ ] **Step 5.4: Update `TODO.md`**

Add an `**Update 2026-05-16 (Tweedie yards_per_target probe, branch `feat/probe-tweedie-yards-per-target`)**:` line under the existing TODO entry that tracks the factor-appropriate sub-model chain (TODO #33b or wherever PR #39's update is logged). Mirror the style of prior `**Update ...**` lines. Cite the verdict + recommended next direction. Close (`[x]` or strikethrough) any TODO entry that was waiting on this probe's verdict.

- [ ] **Step 5.5: Final verification**

```bash
cd /c/Users/alden/FantasyFootball/.worktrees/feat-probe-tweedie-yards-per-target
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
  reports/feature_probe_tweedie_yards_per_target_summary.md \
  reports/feature_probe_tweedie_yards_per_target.csv \
  project_management.md \
  TODO.md
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
data(probe): tweedie yards_per_target verdict <VERDICT> -- RMSE delta <+/-X.XXXX> yards

Pooled 4-year walk-forward (2021-2024) on real WR data. Two arms: RidgeCV
on yards_per_target ratio + clip(>=0) (incumbent) vs TweedieRegressor
(power=1.5, link=log) with alpha CV-selected via GridSearchCV (candidate).
Per-stat receiving_yards delta-RMSE verdict <VERDICT> at 95% CI
[<+/-X.XXXX>, <+/-X.XXXX>] yards, n_paired = <N>. Composite-fpts equivalent
<+/-0.XXXX> fpts.

<Optional marginal-zone flag line if fired>

Mechanism caveat: incumbent arm is Ridge-decomp, NOT current production
(direct Ridge on receiving_yards). Integration adoption-gate cycle separate.

Recommended next direction: <per verdict: integration plan / td_rate_per_target
probe / close yards_per_target factor-appropriate slot strongly>.

Spec: docs/superpowers/specs/2026-05-16-tweedie-yards-per-target-probe-design.md (Task 5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Fill in the bracketed values from the actual verdict at commit time.)

- [ ] **Step 5.7: Push + open PR (pending user authorization)**

The PR creation is a visible/shared action per project conventions. Do NOT push or `gh pr create` without explicit user confirmation. Wait for the orchestrator (or user) to request the push.

When authorized, the invocation:

```bash
git push -u origin feat/probe-tweedie-yards-per-target

gh pr create --title "feat: tweedie yards_per_target sub-model probe (verdict <VERDICT>)" --body "$(cat <<'EOF'
## Summary

- New probe `src/projections/backtest/tweedie_yards_per_target_probe.py` testing factor-appropriate sub-model for `yards_per_target`: TweedieRegressor (power=1.5, link=log) with alpha CV-selected via GridSearchCV (candidate) vs RidgeCV on the ratio + clip(>=0) (incumbent, matches the unbounded-efficiency code path in DecomposedBaselineModel).
- Both arms share the same shared-volume RidgeCV on `targets`; only the yards_per_target efficiency sub-model class differs.
- Verdict: **<VERDICT>** -- RMSE delta <+/-X.XXXX> yards (95% CI [<+/-X.XXXX>, <+/-X.XXXX>]); n_paired = <N>; composite-fpts equivalent <+/-0.XXXX> fpts.
- <Magnitude flag note if fired>

## Mechanism caveat

Incumbent arm (Ridge-decomp on yards_per_target) is NOT current production for receiving_yards; production is direct RidgeCV via `ensemble-decomposed` (which decomposes Stat.RECEPTIONS only per PR #36/#38). SIGNAL at this gate does NOT imply Tweedie-decomp beats current production; that comparison is the integration adoption-gate's question.

## Test plan

- [x] All tests pass: `pytest -v`
- [x] `mypy src tests` -- zero violations
- [x] `ruff check src tests scripts` + `ruff format --check src tests scripts` -- clean
- [x] Integration-seam smoke (`pytest -k "ingest or store or schemas"`): clean

## Reports

- `reports/feature_probe_tweedie_yards_per_target_summary.md` -- verdict + per-year + coverage + magnitude flag + mechanism caveat
- `reports/feature_probe_tweedie_yards_per_target.csv` -- long-form per-year deltas

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review checks (post-plan)

- **Spec coverage:** Tasks 1-5 cover spec §1.1 (the two-arm probe), §1.3 (verdict rule + coverage), §3.1 (module surface), §3.2 (CLI), §4 (testing), §6 (reports), §7 (estimated scope). Spec §1.2 (mechanism prior) is informational and doesn't require implementation work. Spec §1.4 (deferred follow-ups) is explicitly out of scope. Spec §3.5 (sklearn details — Pipeline, GridSearchCV, scorer) is implemented in Task 2.
- **Placeholders:** Steps 5.3 / 5.4 / 5.6 / 5.7 use bracketed `<VERDICT>` / `<+/-X.XXXX>` / `<N>` placeholders for the actual verdict numbers — intentional because they're only known after Step 5.1. Step 4.3's argparse `choices=_VALID_YEARS` pins valid years to a known constant.
- **Type consistency:** `ProbeResults` and `PerStatVerdict` defined in Task 3 are referenced in Task 4's CLI imports — names match. `walk_forward_residuals` signature in Task 3 matches Task 4's invocation. `compute_verdict` signature matches. `_fit_tweedie_efficiency` returns `Pipeline` (Task 2); Task 3's call site passes the Pipeline through to `_predict_yards_tweedie` which accepts `Pipeline`.
- **Scope boundaries:** Each task touches ≤ 5 files per CLAUDE.md "phased execution" rule (Task 5 touches 4: 2 reports + PM + TODO). Pre-task venv re-install is a single command, not counted against the file-touch budget.
- **Feature scaling per spec §5 risk #7:** Task 2 wraps `_fit_tweedie_efficiency` in a sklearn Pipeline of `[StandardScaler, GridSearchCV(TweedieRegressor)]`. Scaler is fit on the full training rows; the inner-CV-fold leakage on a stable StandardScaler is negligible (same pattern as PR #39's logit probe). The Ridge arm is intentionally NOT scaled (Ridge's CV-selected alpha is approximately scale-invariant; the existing decomposed_baseline.py recipe doesn't scale either). The probe report's plan-vs-execution-deviations section should note this preprocessing asymmetry between arms.
- **Zero-yards handling per spec §1.4 #8 + §5 risk #2:** Task 2 Test 2 (`test_fit_tweedie_efficiency_handles_zero_yards_rows`) pins that Tweedie p=1.5 fits without raising on yards_per_target == 0 rows. This is the entire motivation for Tweedie over Gamma (which would require filtering). Documented in the test and in the function docstring.
- **Mechanism caveat per spec §5 risk #6:** Task 4's CLI summary writer (`_write_summary`) explicitly notes that the incumbent arm is NOT current production. Task 5's PM and PR body both include the caveat. Without this, future readers might misinterpret a SIGNAL verdict as "Tweedie-decomp beats current production".
