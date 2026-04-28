# Plan 5c — Hybrid LightGBM with NB-2 for Count Stats — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `LightGBMNbModel` (Model C-NB) as a fourth peer model class. Replaces `QuantileDistribution` with `ParametricNegativeBinomial` for the 13 count cells Plan 3e routes through NB-2 in Ridge; yards stats stay on QuantileDistribution unchanged from Model C-tuned.

**Architecture:** Subclass `LightGBMTunedModel`. For count stats: train one `objective="poisson"` regressor; fit NB-2 dispersion on training residuals; wrap in `ParametricNegativeBinomial(mu, dispersion)` at predict time. lgb's poisson `predict(X)` returns mu (the mean) directly in original scale — no `np.exp` needed. For yards stats: inherited 5-quantile + `QuantileDistribution` behavior unchanged. Reuses Plan 5b's tuned hyperparameters via the inherited `_hyperparams_for(stat)` hook. New `DistributionFamily.MIXED` row-level family for the per-stat-mixed rows.

**Tech Stack:** Python 3.12, LightGBM ≥4.0 (`objective="poisson"` is built-in), scipy.stats / scipy.optimize (already in use), pandera, joblib, pytest, mypy strict, ruff.

**Spec:** `docs/superpowers/specs/2026-04-28-plan-5c-nb-counts-design.md`.

**Branch:** `feat/plan-5c-nb-counts` (branched from `feat/plan-5b-tuning`).

---

## Phase 0 — Public NB dispersion helper

Goal: relocate the NB-2 dispersion estimator from `models/baseline.py` to `distributions/parametric.py` and rename to public `nb_dispersion_from_residuals`. Behavior-preserving move. Existing Ridge tests stay green; the new public helper gets focused tests.

### Task 1: Extract `nb_dispersion_from_residuals` to `parametric.py`

**Files:**
- Modify: `src/projections/distributions/parametric.py` — add public function + 2 module constants.
- Modify: `src/projections/models/baseline.py` — remove the function + constants; update 4 call sites to import from `parametric.py`.
- Create: `tests/test_distributions/test_nb_dispersion.py` — new focused tests for the public helper.

The current state in `baseline.py`:
- `_NB_DISPERSION_CLIP: Final[tuple[float, float]] = (0.01, 1000.0)` (line 104).
- `_NB_MU_FLOOR: Final[float] = 1e-3` (line 107).
- `def _negative_binomial_dispersion_from_residuals(*, mu_hat, actual) -> float:` (lines 110-155).
- Call sites: lines 261, 269, 609 (call the function); line 670 (uses `_NB_MU_FLOOR` directly in `build_stat_distributions`).

- [ ] **Step 1: Write failing test for the new public helper**

Create `tests/test_distributions/test_nb_dispersion.py` with:

```python
"""Tests for the public NB-2 dispersion helper relocated to parametric.py
(Plan 5c Phase 0). The function body is unchanged from baseline.py's prior
private implementation; these tests pin the behavior at the new location.
"""

from __future__ import annotations

import numpy as np
import pytest

from projections.distributions.parametric import (
    _NB_DISPERSION_CLIP,
    _NB_MU_FLOOR,
    nb_dispersion_from_residuals,
)


def test_returns_clip_endpoint_on_empty_input() -> None:
    """Fewer than 2 rows -> degenerate; helper returns clip top to keep downstream NB defined."""
    out = nb_dispersion_from_residuals(mu_hat=np.array([0.5]), actual=np.array([0]))
    assert out == _NB_DISPERSION_CLIP[1]


def test_returns_clip_lower_bound_on_all_zero_actuals() -> None:
    """All-zero actuals drive the likelihood toward the lower clip endpoint."""
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.1, 1.0, size=200)
    out = nb_dispersion_from_residuals(mu_hat=mu, actual=np.zeros(200, dtype=np.int64))
    assert out == _NB_DISPERSION_CLIP[0]


def test_recovers_dispersion_within_30pct_on_known_nb_data() -> None:
    """On samples drawn from a known NB-2 with mean=mu and dispersion=5,
    the MLE should land within ~30% of the truth on a reasonable-size sample."""
    rng = np.random.default_rng(42)
    n = 5000
    mu = rng.uniform(0.5, 2.0, size=n)
    true_dispersion = 5.0
    # Sample NB-2 row-wise: var = mu + mu^2 / dispersion.
    p = true_dispersion / (true_dispersion + mu)
    actual = rng.negative_binomial(n=true_dispersion, p=p, size=n)

    fitted = nb_dispersion_from_residuals(mu_hat=mu, actual=actual.astype(np.float64))

    assert _NB_DISPERSION_CLIP[0] < fitted < _NB_DISPERSION_CLIP[1]
    assert abs(fitted - true_dispersion) / true_dispersion < 0.3


def test_clips_negative_actuals_to_zero() -> None:
    """The helper rounds + clips actuals to non-negative integers before
    fitting NB. Pass a negative value and confirm the fit still returns
    a value in the clip range (i.e., the negative was treated as 0)."""
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.5, 1.0, size=100)
    actual = np.array([-1.0, -0.5, 0.0, 1.0, 2.0] * 20)
    out = nb_dispersion_from_residuals(mu_hat=mu, actual=actual)
    assert _NB_DISPERSION_CLIP[0] <= out <= _NB_DISPERSION_CLIP[1]


def test_constants_exposed_for_baseline_import() -> None:
    """baseline.py imports both constants; pin their existence + types."""
    assert isinstance(_NB_DISPERSION_CLIP, tuple)
    assert len(_NB_DISPERSION_CLIP) == 2
    assert _NB_DISPERSION_CLIP[0] < _NB_DISPERSION_CLIP[1]
    assert isinstance(_NB_MU_FLOOR, float)
    assert _NB_MU_FLOOR > 0
```

- [ ] **Step 2: Run new tests to confirm they fail (function not yet in parametric.py)**

```bash
. .venv/Scripts/activate
pytest -v tests/test_distributions/test_nb_dispersion.py
```

Expected: ImportError or AttributeError on `from projections.distributions.parametric import nb_dispersion_from_residuals` (the symbol doesn't exist yet).

- [ ] **Step 3: Move the function and constants to `parametric.py`**

Open `src/projections/distributions/parametric.py`. At the top of the file, after the existing imports (around line 9), add:

```python
from scipy.optimize import minimize_scalar
from typing import Final
```

Then near the top of the file, after the imports and before the `ParametricNormal` class (around line 11), add the constants and the function:

```python
_NB_DISPERSION_CLIP: Final[tuple[float, float]] = (0.01, 1000.0)
# Floor for the NB rate parameter mu. Kept module-scope so estimator + predict-time
# consumer share one definition.
_NB_MU_FLOOR: Final[float] = 1e-3


def nb_dispersion_from_residuals(*, mu_hat: np.ndarray, actual: np.ndarray) -> float:
    """Conditional MLE for NB-2 dispersion given per-row mean = mu_hat.

    Maximizes sum(nbinom.logpmf(actual_i; n=dispersion, p_i)) over a single
    global ``dispersion`` (the standard NB-2 / "size" parameter), where
    p_i = dispersion / (dispersion + mu_hat_i). Yields per-row var =
    mu_hat_i + mu_hat_i^2 / dispersion -- matches ParametricNegativeBinomial.

    Coerces actual to non-negative integers (counts upstream may carry float
    dtype). Returns the dispersion clipped to ``_NB_DISPERSION_CLIP``.
    """
    counts = np.clip(np.round(actual), 0, None).astype(np.int64)
    mu_clipped = np.maximum(mu_hat, _NB_MU_FLOOR)

    if counts.size < 2:
        return _NB_DISPERSION_CLIP[1]

    def neg_log_lik(dispersion: float) -> float:
        if dispersion <= 0:
            return float("inf")
        # Standard NB-2: n = dispersion (size param, scalar), p per-row.
        # scipy broadcasts the scalar n across the per-row p.
        p = dispersion / (dispersion + mu_clipped)
        return -float(np.sum(stats.nbinom.logpmf(counts, n=dispersion, p=p)))

    result = minimize_scalar(
        neg_log_lik,
        bounds=_NB_DISPERSION_CLIP,
        method="bounded",
        options={"xatol": 1e-3},
    )
    if not result.success or not np.isfinite(result.fun):
        return _NB_DISPERSION_CLIP[1]
    fitted = float(np.clip(result.x, *_NB_DISPERSION_CLIP))
    # Snap to a clip endpoint when the bounded minimizer stops within its xatol
    # of the boundary: degenerate inputs (e.g. all-zero actuals) drive the
    # likelihood monotonically toward an endpoint, but `minimize_scalar` returns
    # a value just inside the bound rather than the bound itself.
    snap_tol = 2e-3
    if fitted - _NB_DISPERSION_CLIP[0] <= snap_tol:
        return _NB_DISPERSION_CLIP[0]
    if _NB_DISPERSION_CLIP[1] - fitted <= snap_tol:
        return _NB_DISPERSION_CLIP[1]
    return fitted
```

Note: `parametric.py` already imports `from scipy import stats`, so `stats.nbinom.logpmf` is the right reference (replaces `scipy_stats.nbinom.logpmf` from baseline.py).

- [ ] **Step 4: Update `baseline.py` to remove the relocated code and import from `parametric.py`**

Open `src/projections/models/baseline.py`. Make these edits:

1. Add the import to the top of the file (after `from scipy.optimize import minimize_scalar` — actually that import goes away too):

```python
from projections.distributions.parametric import (
    _NB_DISPERSION_CLIP,
    _NB_MU_FLOOR,
    nb_dispersion_from_residuals,
)
```

2. Remove `from scipy.optimize import minimize_scalar` (line 50) — it's no longer used in `baseline.py` after the move (verify with grep before deleting).

3. Delete the constants and function (lines 104-155):

```python
# DELETE these lines:
_NB_DISPERSION_CLIP: Final[tuple[float, float]] = (0.01, 1000.0)
# Floor for the NB rate parameter mu. ...
_NB_MU_FLOOR: Final[float] = 1e-3


def _negative_binomial_dispersion_from_residuals(
    *, mu_hat: np.ndarray, actual: np.ndarray
) -> float:
    ...  # 45 lines body
```

4. Update the 3 call sites that referenced `_negative_binomial_dispersion_from_residuals` — lines 261, 269, 609 (verify line numbers with grep — they shift after the delete). Replace with `nb_dispersion_from_residuals` (no leading underscore):

```python
# Around line 261 (in _per_bucket_nb_dispersion_from_residuals):
global_d = nb_dispersion_from_residuals(mu_hat=mu_hat, actual=actual)

# Around line 269 (in same function, inside the per-bucket loop):
nb_dispersion_from_residuals(
    mu_hat=mu_hat[mask], actual=actual[mask]
)

# Around line 609 (in BaselineModel.fit):
self.variance_params[stat] = {
    "dispersion": nb_dispersion_from_residuals(
        mu_hat=mu_hat, actual=y
    )
}
```

5. The reference at line 670 (`mu_safe = max(mu_i, _NB_MU_FLOOR)`) uses the imported `_NB_MU_FLOOR` constant — no change needed (the import in step 1 makes the name available at module scope just like before).

- [ ] **Step 5: Run the new helper tests to confirm they pass**

```bash
pytest -v tests/test_distributions/test_nb_dispersion.py
```

Expected: 5 PASS.

- [ ] **Step 6: Run the existing baseline tests to confirm no regression**

```bash
pytest -v tests/test_models/test_baseline.py tests/test_models/test_baseline_qb.py tests/test_models/test_baseline_rb.py tests/test_models/test_baseline_te.py
```

Expected: all PASS. The function move is behavior-preserving.

- [ ] **Step 7: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 8: Commit**

```bash
git add src/projections/distributions/parametric.py src/projections/models/baseline.py tests/test_distributions/test_nb_dispersion.py
git commit -m "$(cat <<'EOF'
refactor(distributions): extract nb_dispersion_from_residuals to parametric.py — Plan 5c Phase 0

The conditional-MLE NB-2 dispersion estimator was previously private in
baseline.py with a leading underscore. Plan 5c needs it in
LightGBMNbModel as well, so it moves next to ParametricNegativeBinomial
in parametric.py with a public name.

Behavior-preserving move: function body unchanged; constants
(_NB_DISPERSION_CLIP, _NB_MU_FLOOR) move alongside; baseline.py imports
both. Existing Ridge regression tests pass without modification.

New focused tests for the public helper land in
tests/test_distributions/test_nb_dispersion.py (clip endpoints on
degenerate inputs; recovers true dispersion within 30% on known NB-2
samples; handles negative actuals).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1 — `LightGBMNbModel` + `DistributionFamily.MIXED` + factories + dispatch

Goal: new model class that fits poisson regressors for count stats and NB-2 dispersion on residuals; predicts `ParametricNegativeBinomial` per row; yards stats inherit `LightGBMTunedModel` behavior unchanged.

### Task 2: Add `DistributionFamily.MIXED` enum value

**Files:**
- Modify: `src/projections/schemas.py` — append `MIXED` to the `DistributionFamily` StrEnum.

- [ ] **Step 1: Edit `src/projections/schemas.py`**

Find the `class DistributionFamily(StrEnum):` block (currently around lines 145-154) and add `MIXED` as the last entry:

```python
class DistributionFamily(StrEnum):
    """Backing representation of a `Distribution`."""

    NORMAL = "NORMAL"
    GAMMA = "GAMMA"
    NEGATIVE_BINOMIAL = "NEGATIVE_BINOMIAL"  # NEW (Plan 3e Phase 1)
    STUDENT_T = "STUDENT_T"  # NEW (Plan 3e Phase 2) — heavy-tailed continuous
    SAMPLED = "SAMPLED"  # explicit sample array
    SAMPLED_SUMMARY = "SAMPLED_SUMMARY"  # per-stat dist params + summary in mean/p10/p50/p90
    QUANTILE = "QUANTILE"  # NEW (Plan 5) — Model C (LightGBM quantile regression)
    MIXED = "MIXED"  # NEW (Plan 5c) — per-row distribution mixes families per stat
```

- [ ] **Step 2: Verify the new enum value loads**

```bash
. .venv/Scripts/activate
python -c "from projections.schemas import DistributionFamily; print(DistributionFamily.MIXED.value)"
```

Expected: `MIXED`.

- [ ] **Step 3: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 4: Commit**

```bash
git add src/projections/schemas.py
git commit -m "$(cat <<'EOF'
feat(schemas): DistributionFamily.MIXED — Plan 5c Phase 1 prep

Marks rows whose per-stat distributions span multiple families. The codec's
pack_per_stat_params already supports per-stat families inside the params
blob; MIXED is the row-level metadata for downstream consumers reading the
parquet directly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Implement `LightGBMNbModel` + per-position factories + POSITION_DISPATCH wiring

**Files:**
- Create: `src/projections/models/lightgbm_nb.py` — class + factories.
- Modify: `src/projections/models/__init__.py` — extend imports, `__all__`, and `_<POS>_FACTORIES` dicts.

- [ ] **Step 1: Create `src/projections/models/lightgbm_nb.py`**

Write the complete content:

```python
"""Hybrid LightGBM with NB-2 for count stats — Plan 5c.

Subclass of LightGBMTunedModel. For zero-inflated count stats (the 13 cells
Plan 3e routes through NB-2 in Ridge — passing_tds / rushing_tds /
receiving_tds / interceptions / fumbles_lost, intersected with each
position's target_stats), trains one lgb.LGBMRegressor with
``objective="poisson"``, reads predicted mu directly from
``regressor.predict(X)`` (lgb's poisson predict returns the mean in
original scale, already exponentiated), and fits NB-2 dispersion on
training residuals via ``nb_dispersion_from_residuals``. Predict-time
distribution per count stat: ``ParametricNegativeBinomial(mu, dispersion)``.

For yards / receptions stats: 5-quantile sub-models exactly as
LightGBMTunedModel does today. Predict-time distribution:
``QuantileDistribution`` (unchanged).

Per-row ``ProjectionWeeklySchema.family`` is set to
``DistributionFamily.MIXED``; per-stat families remain encoded
individually inside the params blob via the codec's existing per-stat
dispatch.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import lightgbm as lgb
import numpy as np
import pandas as pd

from projections.distributions import ParametricNegativeBinomial, pack_per_stat_params
from projections.distributions.parametric import (
    _NB_MU_FLOOR,
    nb_dispersion_from_residuals,
)
from projections.distributions.quantile import QuantileDistribution
from projections.models.base import compute_code_hash
from projections.models.lightgbm import (
    QUANTILE_GRID,
    _LightGBMConfig,
    _filter_features,
    _QB_FEATURE_COLUMNS,
    _QB_NON_NEGATIVE,
    _QB_TARGET_STATS,
    _RB_FEATURE_COLUMNS,
    _RB_NON_NEGATIVE,
    _RB_TARGET_STATS,
    _TE_FEATURE_COLUMNS,
    _TE_NON_NEGATIVE,
    _TE_TARGET_STATS,
    _WR_FEATURE_COLUMNS,
    _WR_NON_NEGATIVE,
    _WR_TARGET_STATS,
)
from projections.models.lightgbm_tuned import (
    LightGBMTunedModel,
    _TUNED_PARAMS_PATH,
)
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    Position,
    ProjectionWeeklySchema,
    QbFeaturesSchema,
    RbFeaturesSchema,
    Ruleset,
    Stat,
    TeFeaturesSchema,
    WeeklyStatsSchema,
    WrFeaturesSchema,
)
from projections.scoring.score_distribution import (
    derive_row_seed,
    score_distribution,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

# Stats Plan 3e routes through NB-2 in Ridge's _<POS>_DIST_FAMILIES.
# Per-position intersection with target_stats yields 13 cells: QB 4
# (passing_tds, interceptions, rushing_tds, fumbles_lost); RB / TE / WR
# 3 each (receiving_tds, rushing_tds, fumbles_lost).
COUNT_STATS_FOR_NB: Final[frozenset[Stat]] = frozenset(
    {
        Stat.PASSING_TDS,
        Stat.RUSHING_TDS,
        Stat.RECEIVING_TDS,
        Stat.INTERCEPTIONS,
        Stat.FUMBLES_LOST,
    }
)


def _code_hash_files_nb(position: Position) -> tuple[Path, ...]:
    """Source files whose content is hashed into the NB model's model_id.

    Mirrors LightGBMTunedModel's set + adds lightgbm_nb.py. parametric.py is
    already in the parent's set (it owns the QuantileDistribution codec
    that the parent uses); after Plan 5c Phase 0 it also owns
    nb_dispersion_from_residuals + ParametricNegativeBinomial, so the
    same path covers both reasons.
    """
    src = _PROJECT_ROOT / "src" / "projections"
    feat_module = {
        Position.QB: "qb.py",
        Position.RB: "rb.py",
        Position.TE: "te.py",
        Position.WR: "wr.py",
    }[position]
    return (
        src / "models" / "lightgbm_nb.py",
        src / "models" / "lightgbm_tuned.py",
        src / "models" / "lightgbm.py",
        src / "models" / "base.py",
        src / "distributions" / "quantile.py",
        src / "distributions" / "codec.py",
        src / "distributions" / "parametric.py",
        src / "features" / feat_module,
        src / "features" / "_shared.py",
        src / "features" / "_rolling.py",
        src / "features" / "_opponent.py",
        src / "scoring" / "score.py",
        src / "scoring" / "score_distribution.py",
        _TUNED_PARAMS_PATH,
    )


class LightGBMNbModel(LightGBMTunedModel):
    """LightGBM with NB-2 for count stats and QuantileDistribution for yards stats.

    Inherits from LightGBMTunedModel: tuned-params loader, _hyperparams_for(stat)
    hook, joblib save/load, feature/weekly_stats join. Overrides fit and
    predict_distribution to branch per stat between count (NB-2) and yards
    (5-quantile) paths. Overrides code_hash and model_id to reflect the
    lightgbm-nb: prefix and the lightgbm_nb.py file in the hash.
    """

    def __init__(
        self,
        *,
        config: _LightGBMConfig,
        tuned_params_path: Path = _TUNED_PARAMS_PATH,
    ) -> None:
        super().__init__(config=config, tuned_params_path=tuned_params_path)
        self._count_models: dict[Stat, lgb.Booster] = {}
        self._count_dispersions: dict[Stat, float] = {}
        self._count_best_iters: dict[Stat, int] = {}

    @property
    def code_hash(self) -> str:
        return compute_code_hash(_code_hash_files_nb(self._config.position))

    @property
    def model_id(self) -> str:
        if not self._is_fitted:
            raise RuntimeError(
                "model_id not available before fit() — depends on "
                "training-time state"
            )
        assert self._train_start is not None and self._train_end is not None
        return (
            f"lightgbm-nb:{self._config.position.value.lower()}:"
            f"{self.code_hash}:{self._train_start}-{self._train_end}"
        )

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """Train per-stat sub-models with hybrid count/yards routing.

        For stat in COUNT_STATS_FOR_NB: train one lgb.LGBMRegressor with
        objective="poisson" + tuned hyperparameters; fit NB-2 dispersion
        on training residuals via nb_dispersion_from_residuals.

        Otherwise: 5 quantile sub-models, identical to LightGBMTunedModel.
        """
        # Validate features against the position schema (mirrors parent).
        features = self._config.feature_schema.validate(features)
        weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
        weekly_stats = weekly_stats[
            weekly_stats["position"] == self._config.position.value
        ].copy()

        target_cols = [s.value for s in self._config.target_stats]
        joined = features.merge(
            weekly_stats[["gsis_id", "season", "week", *target_cols]],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        if joined.empty:
            raise ValueError("Empty training set after feature/weekly_stats join")

        seasons = sorted(joined["season"].unique())
        if len(seasons) < 2:
            raise ValueError(
                f"Need >=2 training seasons for early-stopping validation slice; "
                f"got {len(seasons)}"
            )

        val_season = seasons[-1]
        train_mask = joined["season"] != val_season
        val_mask = joined["season"] == val_season

        feat_cols = list(self._config.feature_columns)
        x_train = joined.loc[train_mask, feat_cols].to_numpy(dtype=np.float64)
        x_val = joined.loc[val_mask, feat_cols].to_numpy(dtype=np.float64)

        for stat in self._config.target_stats:
            stat_params = self._hyperparams_for(stat)
            y_train = joined.loc[train_mask, stat.value].to_numpy(dtype=np.float64)
            y_val = joined.loc[val_mask, stat.value].to_numpy(dtype=np.float64)

            if stat in COUNT_STATS_FOR_NB:
                # Single poisson regressor; predicts log-mu.
                regressor = lgb.LGBMRegressor(
                    objective="poisson",
                    **stat_params,
                )
                regressor.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                best_iter = int(regressor.best_iteration_ or 0)
                if best_iter == 0:
                    warnings.warn(
                        f"LightGBMNbModel.fit: best_iter=0 for "
                        f"{self._config.position.value}/{stat.value} (poisson); "
                        "early stopping fired immediately. Sub-model will "
                        "predict at constant baseline.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                self._count_models[stat] = regressor.booster_
                self._count_best_iters[stat] = best_iter
                # Fit NB-2 dispersion on training residuals.
                mu_hat_train = np.exp(regressor.predict(x_train))
                dispersion = nb_dispersion_from_residuals(
                    mu_hat=mu_hat_train, actual=y_train
                )
                self._count_dispersions[stat] = dispersion
            else:
                # Inherited quantile-stat behavior: 5 sub-models.
                self._sub_models[stat] = {}
                for q in QUANTILE_GRID:
                    regressor = lgb.LGBMRegressor(
                        objective="quantile",
                        alpha=q,
                        **stat_params,
                    )
                    regressor.fit(
                        x_train,
                        y_train,
                        eval_set=[(x_val, y_val)],
                        callbacks=[lgb.early_stopping(50, verbose=False)],
                    )
                    best_iter = int(regressor.best_iteration_ or 0)
                    if best_iter == 0:
                        warnings.warn(
                            f"LightGBMNbModel.fit: best_iter=0 for "
                            f"{self._config.position.value}/{stat.value}/q={q}; "
                            "early stopping fired immediately. Sub-model will "
                            "predict at constant baseline.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    self._sub_models[stat][q] = regressor.booster_
                    self._best_iters[(stat, q)] = best_iter

        self._train_start = int(seasons[0])
        self._train_end = int(seasons[-1])
        self._is_fitted = True

    def predict_distribution(
        self, features: pd.DataFrame, ruleset: Ruleset
    ) -> pd.DataFrame:
        """Predict per-row composite fantasy-points distribution with hybrid families."""
        if not self._is_fitted:
            raise RuntimeError("predict_distribution requires fit() first")

        feat_cols = list(self._config.feature_columns)
        actual_cols = set(features.columns)
        missing = set(feat_cols) - actual_cols
        if missing:
            raise ValueError(
                f"Feature columns differ from training: missing={sorted(missing)}; "
                f"expected feature_columns={feat_cols}"
            )

        features = self._config.feature_schema.validate(features)
        if features.empty:
            empty_cols = list(ProjectionWeeklySchema.to_schema().columns.keys())
            return ProjectionWeeklySchema.validate(pd.DataFrame(columns=empty_cols))

        x = features[feat_cols].to_numpy(dtype=np.float64)
        n_rows = x.shape[0]
        quant_arr = np.array(QUANTILE_GRID, dtype=np.float64)

        # Per-stat predictions: count stats get mu_hat; yards stats get sorted/clipped quantiles.
        per_stat_count_mu: dict[Stat, np.ndarray[Any, np.dtype[np.float64]]] = {}
        per_stat_quantile_pred: dict[Stat, np.ndarray[Any, np.dtype[np.float64]]] = {}
        for stat in self._config.target_stats:
            if stat in COUNT_STATS_FOR_NB:
                log_mu = self._count_models[stat].predict(x).astype(np.float64)
                mu_hat = np.maximum(np.exp(log_mu), _NB_MU_FLOOR)
                per_stat_count_mu[stat] = mu_hat
            else:
                preds_per_q = np.column_stack(
                    [self._sub_models[stat][q].predict(x) for q in QUANTILE_GRID]
                ).astype(np.float64)
                preds_per_q.sort(axis=1)
                if stat in self._config.non_negative_stats:
                    np.maximum(preds_per_q, 0.0, out=preds_per_q)
                per_stat_quantile_pred[stat] = preds_per_q

        out_rows: list[dict[str, Any]] = []
        generated_at = datetime.now(UTC)
        gsis_id_col = features["gsis_id"].to_numpy()
        season_col = features["season"].to_numpy()
        week_col = features["week"].to_numpy()
        team_col = features["team"].to_numpy()
        opponent_col = features["opponent"].to_numpy()

        for row_idx in range(n_rows):
            per_stat_dists: dict[Stat, Any] = {}
            for stat in self._config.target_stats:
                if stat in COUNT_STATS_FOR_NB:
                    per_stat_dists[stat] = ParametricNegativeBinomial(
                        mean=float(per_stat_count_mu[stat][row_idx]),
                        dispersion=self._count_dispersions[stat],
                    )
                else:
                    per_stat_dists[stat] = QuantileDistribution(
                        quantiles=quant_arr,
                        values=per_stat_quantile_pred[stat][row_idx],
                    )

            seed = derive_row_seed(
                gsis_id=str(gsis_id_col[row_idx]),
                season=int(season_col[row_idx]),
                week=int(week_col[row_idx]),
                ruleset_name=ruleset.name,
            )
            composite = score_distribution(per_stat_dists, ruleset, seed=seed)

            out_rows.append(
                {
                    "gsis_id": str(gsis_id_col[row_idx]),
                    "season": int(season_col[row_idx]),
                    "week": int(week_col[row_idx]),
                    "position": self._config.position.value,
                    "team": str(team_col[row_idx]),
                    "opponent": str(opponent_col[row_idx]),
                    "ruleset": ruleset.name,
                    "family": DistributionFamily.MIXED.value,
                    "params": pack_per_stat_params(per_stat_dists),
                    "mean": composite.mean(),
                    "p10": composite.quantile(0.10),
                    "p50": composite.quantile(0.50),
                    "p90": composite.quantile(0.90),
                    "model_id": self.model_id,
                    "generated_at": pd.Timestamp(generated_at).as_unit("us"),
                }
            )
        out = pd.DataFrame(out_rows)
        for col in ("gsis_id", "team", "opponent", "ruleset", "family", "model_id"):
            out[col] = out[col].astype(_PYARROW_STR)
        out["position"] = out["position"].astype(_PYARROW_STR)
        return ProjectionWeeklySchema.validate(out)


def qb_lightgbm_nb() -> LightGBMNbModel:
    return LightGBMNbModel(
        config=_LightGBMConfig(
            position=Position.QB,
            target_stats=_QB_TARGET_STATS,
            feature_columns=_filter_features(_QB_FEATURE_COLUMNS),
            feature_schema=QbFeaturesSchema,
            non_negative_stats=_QB_NON_NEGATIVE,
        )
    )


def rb_lightgbm_nb() -> LightGBMNbModel:
    return LightGBMNbModel(
        config=_LightGBMConfig(
            position=Position.RB,
            target_stats=_RB_TARGET_STATS,
            feature_columns=_filter_features(_RB_FEATURE_COLUMNS),
            feature_schema=RbFeaturesSchema,
            non_negative_stats=_RB_NON_NEGATIVE,
        )
    )


def te_lightgbm_nb() -> LightGBMNbModel:
    return LightGBMNbModel(
        config=_LightGBMConfig(
            position=Position.TE,
            target_stats=_TE_TARGET_STATS,
            feature_columns=_filter_features(_TE_FEATURE_COLUMNS),
            feature_schema=TeFeaturesSchema,
            non_negative_stats=_TE_NON_NEGATIVE,
        )
    )


def wr_lightgbm_nb() -> LightGBMNbModel:
    return LightGBMNbModel(
        config=_LightGBMConfig(
            position=Position.WR,
            target_stats=_WR_TARGET_STATS,
            feature_columns=_filter_features(_WR_FEATURE_COLUMNS),
            feature_schema=WrFeaturesSchema,
            non_negative_stats=_WR_NON_NEGATIVE,
        )
    )
```

- [ ] **Step 2: Wire factories into `POSITION_DISPATCH`**

Open `src/projections/models/__init__.py`. Three edits:

(a) Import block — add the `LightGBMNbModel` import alongside the existing tuned import:

```python
from projections.models.lightgbm_nb import (
    LightGBMNbModel,
    qb_lightgbm_nb,
    rb_lightgbm_nb,
    te_lightgbm_nb,
    wr_lightgbm_nb,
)
```

(b) `__all__` — add 5 entries (alphabetical with the existing entries):

```python
__all__ = [
    "POSITION_DISPATCH",
    "BaselineModel",
    "LightGBMModel",
    "LightGBMNbModel",
    "LightGBMTunedModel",
    "Model",
    "compute_code_hash",
    "qb_baseline",
    "qb_lightgbm",
    "qb_lightgbm_nb",
    "qb_lightgbm_tuned",
    "rb_baseline",
    "rb_lightgbm",
    "rb_lightgbm_nb",
    "rb_lightgbm_tuned",
    "te_baseline",
    "te_lightgbm",
    "te_lightgbm_nb",
    "te_lightgbm_tuned",
    "wr_baseline",
    "wr_lightgbm",
    "wr_lightgbm_nb",
    "wr_lightgbm_tuned",
]
```

(c) Each of the four `_<POS>_FACTORIES` dicts gains a `"lightgbm-nb"` key:

```python
_QB_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": qb_baseline,
    "lightgbm": qb_lightgbm,
    "lightgbm-tuned": qb_lightgbm_tuned,
    "lightgbm-nb": qb_lightgbm_nb,
}
_RB_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": rb_baseline,
    "lightgbm": rb_lightgbm,
    "lightgbm-tuned": rb_lightgbm_tuned,
    "lightgbm-nb": rb_lightgbm_nb,
}
_TE_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": te_baseline,
    "lightgbm": te_lightgbm,
    "lightgbm-tuned": te_lightgbm_tuned,
    "lightgbm-nb": te_lightgbm_nb,
}
_WR_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": wr_baseline,
    "lightgbm": wr_lightgbm,
    "lightgbm-tuned": wr_lightgbm_tuned,
    "lightgbm-nb": wr_lightgbm_nb,
}
```

- [ ] **Step 3: Verify factories construct + POSITION_DISPATCH wiring**

```bash
. .venv/Scripts/activate
python -c "from projections.models import qb_lightgbm_nb, rb_lightgbm_nb, te_lightgbm_nb, wr_lightgbm_nb; print('factories ok:', [f().__class__.__name__ for f in (qb_lightgbm_nb, rb_lightgbm_nb, te_lightgbm_nb, wr_lightgbm_nb)])"
```

Expected: `factories ok: ['LightGBMNbModel', 'LightGBMNbModel', 'LightGBMNbModel', 'LightGBMNbModel']`.

```bash
python -c "from projections.models import POSITION_DISPATCH; from projections.schemas import Position; [print(p.value, sorted(POSITION_DISPATCH[p].factories.keys())) for p in Position if p in POSITION_DISPATCH]"
```

Expected: each position prints `['baseline', 'lightgbm', 'lightgbm-nb', 'lightgbm-tuned']`.

- [ ] **Step 4: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 5: Commit**

```bash
git add src/projections/models/lightgbm_nb.py src/projections/models/__init__.py
git commit -m "$(cat <<'EOF'
feat(models): LightGBMNbModel + POSITION_DISPATCH wiring — Plan 5c Phase 1

Subclass of LightGBMTunedModel. For zero-inflated count stats (the 13 cells
Plan 3e routes to NB-2 in Ridge — passing_tds/rushing_tds/receiving_tds/
interceptions/fumbles_lost intersected with each position's target_stats),
trains one lgb.LGBMRegressor with objective="poisson" using tuned
hyperparameters; predicts log-mu, exponentiates to mu_hat, fits NB-2
dispersion on training residuals; wraps in ParametricNegativeBinomial at
predict time.

For yards/receptions stats: 5 quantile sub-models exactly as
LightGBMTunedModel does today; QuantileDistribution at predict time.
Per-row family is DistributionFamily.MIXED; per-stat families remain
encoded individually inside the params blob via the codec's existing
per-stat dispatch.

POSITION_DISPATCH.factories gains "lightgbm-nb" per position.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Cross-cutting tests for `LightGBMNbModel`

**Files:**
- Create: `tests/test_models/test_lightgbm_nb.py`

- [ ] **Step 1: Write the test file**

Create `tests/test_models/test_lightgbm_nb.py`:

```python
"""Cross-cutting tests for LightGBMNbModel (Model C-NB, Plan 5c).

Per-position parametrized smoke lives in test_lightgbm_nb_smoke.py (Phase 2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from projections.distributions import (
    ParametricNegativeBinomial,
    QuantileDistribution,
    unpack_per_stat_params,
)
from projections.models.lightgbm import wr_lightgbm
from projections.models.lightgbm_nb import (
    COUNT_STATS_FOR_NB,
    LightGBMNbModel,
    qb_lightgbm_nb,
    rb_lightgbm_nb,
    te_lightgbm_nb,
    wr_lightgbm_nb,
)
from projections.models.lightgbm_tuned import wr_lightgbm_tuned
from projections.schemas import DistributionFamily, Ruleset, Stat


# ---------------- Synthetic fixtures (re-using the test_lightgbm.py shape) ----------------


def _build_synthetic_wr_features(
    n_seasons: int = 4, n_weeks: int = 17, n_players: int = 30
) -> pd.DataFrame:
    """Synthetic WrFeaturesSchema-shaped DataFrame for fit/predict smoke tests."""
    from projections.schemas import _PYARROW_STR, WrFeaturesSchema

    rng = np.random.default_rng(42)
    rows = []
    for season in range(2018, 2018 + n_seasons):
        for week in range(1, n_weeks + 1):
            for p in range(n_players):
                rows.append(
                    {
                        "gsis_id": f"00-{p:07d}",
                        "season": season,
                        "week": week,
                        "team": "KC",
                        "opponent": "DEN",
                    }
                )
    df = pd.DataFrame(rows)
    df["season"] = df["season"].astype(np.int64)
    df["week"] = df["week"].astype(np.int64)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)

    schema_cols = WrFeaturesSchema.to_schema().columns
    for col_name, col in schema_cols.items():
        if col_name in df.columns:
            continue
        dtype_str = str(col.dtype)
        if "bool" in dtype_str.lower():
            df[col_name] = rng.integers(0, 2, size=len(df)).astype(bool)
        elif "int" in dtype_str.lower():
            df[col_name] = rng.integers(1, 6, size=len(df)).astype(np.int64)
        else:
            df[col_name] = rng.uniform(0.0, 0.5, size=len(df)).astype(np.float64)
    return WrFeaturesSchema.validate(df)


def _build_synthetic_wr_weekly_stats(features: pd.DataFrame) -> pd.DataFrame:
    """WeeklyStatsSchema-shaped truth for the synthetic WR features."""
    from projections.schemas import WeeklyStatsSchema

    rng = np.random.default_rng(43)
    n = len(features)
    df = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    df["position"] = "WR"
    df["receptions"] = rng.integers(0, 8, size=n).astype(np.int64)
    df["targets"] = (df["receptions"] + rng.integers(0, 4, size=n)).astype(np.int64)
    df["receiving_yards"] = rng.integers(0, 100, size=n).astype(np.float64)
    df["receiving_tds"] = rng.integers(0, 2, size=n).astype(np.int64)
    df["receiving_air_yards"] = rng.integers(0, 80, size=n).astype(np.float64)
    df["carries"] = rng.integers(0, 2, size=n).astype(np.int64)
    df["rushing_yards"] = rng.integers(0, 20, size=n).astype(np.float64)
    df["rushing_tds"] = rng.integers(0, 1, size=n).astype(np.int64)
    df["passing_yards"] = np.zeros(n, dtype=np.float64)
    df["passing_tds"] = np.zeros(n, dtype=np.int64)
    df["interceptions"] = np.zeros(n, dtype=np.int64)
    df["attempts"] = np.zeros(n, dtype=np.int64)
    df["completions"] = np.zeros(n, dtype=np.int64)
    df["sacks"] = np.zeros(n, dtype=np.int64)
    df["fumbles_lost"] = rng.integers(0, 1, size=n).astype(np.int64)
    return WeeklyStatsSchema.validate(df)


# ---------------- Tests ----------------


def test_factories_construct() -> None:
    for factory in (qb_lightgbm_nb, rb_lightgbm_nb, te_lightgbm_nb, wr_lightgbm_nb):
        m = factory()
        assert isinstance(m, LightGBMNbModel)


def test_count_stats_set() -> None:
    """COUNT_STATS_FOR_NB pins the 5 Stat values Plan 3e routes to NB-2 in Ridge."""
    assert COUNT_STATS_FOR_NB == frozenset(
        {
            Stat.PASSING_TDS,
            Stat.RUSHING_TDS,
            Stat.RECEIVING_TDS,
            Stat.INTERCEPTIONS,
            Stat.FUMBLES_LOST,
        }
    )


def test_fit_populates_count_and_yards_models() -> None:
    """After fit, count stats land in _count_models + _count_dispersions; yards
    stats land in _sub_models. WR has 3 count stats (receiving_tds, rushing_tds,
    fumbles_lost) and 3 yards/recs stats (receptions, receiving_yards, rushing_yards)."""
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)

    model = wr_lightgbm_nb()
    model.fit(features, weekly)

    expected_counts = {Stat.RECEIVING_TDS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST}
    assert set(model._count_models.keys()) == expected_counts
    assert set(model._count_dispersions.keys()) == expected_counts
    for stat, dispersion in model._count_dispersions.items():
        assert 0.0 < dispersion < 10000.0, f"{stat}: dispersion {dispersion} outside clip"

    expected_yards = {Stat.RECEPTIONS, Stat.RECEIVING_YARDS, Stat.RUSHING_YARDS}
    assert set(model._sub_models.keys()) == expected_yards
    for stat in expected_yards:
        assert len(model._sub_models[stat]) == 5  # 5 quantiles


def test_predict_distribution_emits_mixed_family() -> None:
    """Every predicted row's `family` column is MIXED."""
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)

    model = wr_lightgbm_nb()
    model.fit(features, weekly)
    pred = model.predict_distribution(features, ruleset=Ruleset.espn_ppr())

    assert (pred["family"] == DistributionFamily.MIXED.value).all()


def test_codec_round_trip_yields_correct_per_stat_distribution_types() -> None:
    """unpack_per_stat_params on a Model C-NB row gives NB for count stats
    and QuantileDistribution for yards stats."""
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)

    model = wr_lightgbm_nb()
    model.fit(features, weekly)
    pred = model.predict_distribution(features, ruleset=Ruleset.espn_ppr())

    blob = pred["params"].iloc[0]
    per_stat = unpack_per_stat_params(blob)

    # WR count stats -> ParametricNegativeBinomial.
    for stat in (Stat.RECEIVING_TDS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST):
        assert isinstance(per_stat[stat], ParametricNegativeBinomial), (
            f"{stat}: expected NB, got {type(per_stat[stat]).__name__}"
        )

    # WR yards / receptions stats -> QuantileDistribution.
    for stat in (Stat.RECEPTIONS, Stat.RECEIVING_YARDS, Stat.RUSHING_YARDS):
        assert isinstance(per_stat[stat], QuantileDistribution), (
            f"{stat}: expected QuantileDistribution, got {type(per_stat[stat]).__name__}"
        )


def test_yards_stat_predictions_match_tuned_baseline() -> None:
    """Yards-stat predictions from LightGBMNbModel should be bit-exact identical
    to LightGBMTunedModel's on the same fixture, since LightGBMNbModel inherits
    yards-stat training and only overrides count-stat training. Checks the
    `p10`, `p50`, `p90` columns are NOT bit-exact at the composite level (count
    stats differ between the two), but per-stat yards quantile predictions are.

    The simplest way to verify this without exposing internals: compare
    yards-stat sub-models' best_iters between the two models. They should match
    when fitted on identical data with identical hyperparameters."""
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)

    nb = wr_lightgbm_nb()
    tuned = wr_lightgbm_tuned()
    nb.fit(features, weekly)
    tuned.fit(features, weekly)

    yards_stats = (Stat.RECEPTIONS, Stat.RECEIVING_YARDS, Stat.RUSHING_YARDS)
    for stat in yards_stats:
        for q in (0.05, 0.10, 0.50, 0.90, 0.95):
            assert nb._best_iters[(stat, q)] == tuned._best_iters[(stat, q)], (
                f"{stat} q={q}: NB best_iter {nb._best_iters[(stat, q)]} "
                f"differs from tuned {tuned._best_iters[(stat, q)]}"
            )


def test_model_id_uses_nb_prefix() -> None:
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)
    model = wr_lightgbm_nb()
    model.fit(features, weekly)
    assert model.model_id.startswith("lightgbm-nb:wr:")


def test_code_hash_differs_from_tuned() -> None:
    """The NB subclass hashes a different file set (adds lightgbm_nb.py),
    so its code_hash differs from LightGBMTunedModel's."""
    nb = wr_lightgbm_nb()
    tuned = wr_lightgbm_tuned()
    assert nb.code_hash != tuned.code_hash


def test_save_load_round_trip(tmp_path: Path) -> None:
    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)
    original = wr_lightgbm_nb()
    original.fit(features, weekly)
    artifact = tmp_path / "nb.joblib"
    original.save(artifact)

    loaded = LightGBMNbModel.load(artifact)
    assert loaded.model_id == original.model_id

    pred_orig = original.predict_distribution(features, ruleset=Ruleset.espn_ppr())
    pred_loaded = loaded.predict_distribution(features, ruleset=Ruleset.espn_ppr())
    for col in ("mean", "p10", "p50", "p90"):
        np.testing.assert_array_equal(
            pred_orig[col].to_numpy(), pred_loaded[col].to_numpy()
        )


def test_predict_distribution_validates_against_schema() -> None:
    """The output DataFrame validates against ProjectionWeeklySchema."""
    from projections.schemas import ProjectionWeeklySchema

    features = _build_synthetic_wr_features()
    weekly = _build_synthetic_wr_weekly_stats(features)
    model = wr_lightgbm_nb()
    model.fit(features, weekly)

    pred = model.predict_distribution(features, ruleset=Ruleset.espn_ppr())
    # Re-validate to confirm no silent dtype drift.
    revalidated = ProjectionWeeklySchema.validate(pred)
    assert len(revalidated) == len(features)
```

- [ ] **Step 2: Run the new tests**

```bash
. .venv/Scripts/activate
pytest -v tests/test_models/test_lightgbm_nb.py
```

Expected: 9 tests PASS.

- [ ] **Step 3: Run the full models test suite for regression**

```bash
pytest -v tests/test_models/
```

Expected: all PASS. The new model class and tests are additive; existing tests for LightGBMModel, LightGBMTunedModel, BaselineModel are unaffected.

- [ ] **Step 4: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 5: Commit**

```bash
git add tests/test_models/test_lightgbm_nb.py
git commit -m "$(cat <<'EOF'
test(models): cross-cutting tests for LightGBMNbModel — Plan 5c Phase 1

Verifies (a) factories construct; (b) COUNT_STATS_FOR_NB pins the 5 Stat
values matching Plan 3e's Ridge routing; (c) fit populates _count_models +
_count_dispersions for count stats AND _sub_models for yards stats; (d) every
predicted row carries family=MIXED; (e) codec round-trip gives the right
per-stat distribution types (NB for counts, Quantile for yards); (f) yards-
stat sub-model training matches LightGBMTunedModel bit-for-bit on best_iters
(yards path is inherited unchanged); (g) model_id uses lightgbm-nb: prefix;
(h) code_hash differs from tuned; (i) save/load round-trip; (j) output
validates against ProjectionWeeklySchema.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — Backtest harness wiring + smokes

Goal: harness CLI accepts `--model lightgbm-nb` and `--model all` (4 classes); default-on smoke covers all 4; quad-model harness end-to-end test; per-position parametrized smoke for the new class.

### Task 5: Update backtest harness CLI

**Files:**
- Modify: `scripts/backtest.py`

- [ ] **Step 1: Edit the `--model` argparse block**

Open `scripts/backtest.py`. Find the `--model` argparse block (currently around lines 117-126 — choices `["baseline", "lightgbm", "lightgbm-tuned", "both", "all"]`). Update:

```python
    parser.add_argument(
        "--model",
        choices=["baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb", "both", "all"],
        default="both",
        help=(
            "Which model class(es) to run. "
            "'both' = Model A + Model C (legacy default). "
            "'all' = Model A + Model C + Model C-tuned + Model C-NB."
        ),
    )
```

Then find the model-class translation block (currently around lines 125-132):

```python
    if args.model == "both":
        model_classes: tuple[str, ...] = ("baseline", "lightgbm")
    elif args.model == "all":
        model_classes = ("baseline", "lightgbm", "lightgbm-tuned")
    else:
        model_classes = (args.model,)
```

Replace with:

```python
    if args.model == "both":
        model_classes: tuple[str, ...] = ("baseline", "lightgbm")
    elif args.model == "all":
        model_classes = ("baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb")
    else:
        model_classes = (args.model,)
```

- [ ] **Step 2: Smoke-test the CLI parsing**

```bash
. .venv/Scripts/activate
python scripts/backtest.py --help 2>&1 | head -30
```

Expected: help text shows `--model {baseline,lightgbm,lightgbm-tuned,lightgbm-nb,both,all}`.

```bash
python -c "
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
parser.add_argument('--model', choices=['baseline', 'lightgbm', 'lightgbm-tuned', 'lightgbm-nb', 'both', 'all'], default='both')
args = parser.parse_args(['--model', 'lightgbm-nb', '--check'])
print('parsed model=', args.model)
"
```

Expected: `parsed model= lightgbm-nb`.

- [ ] **Step 3: Run mypy + ruff**

```bash
mypy src tests scripts/backtest.py
ruff check src tests scripts/backtest.py
ruff format --check src tests scripts/backtest.py
```

Expected: zero violations.

- [ ] **Step 4: Commit**

```bash
git add scripts/backtest.py
git commit -m "$(cat <<'EOF'
feat(backtest): --model lightgbm-nb; --model all expands to 4 classes — Plan 5c Phase 2

'both' preserves legacy A+C selection. 'all' now runs A+C+C-tuned+C-NB.
Default remains 'both' so existing CI invocations are unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Extend default-on smoke to cover all 4 model classes

**Files:**
- Modify: `tests/backtest/test_backtest_smoke.py`

The current smoke (Plan 5b Task 9) passes `model_classes=("baseline", "lightgbm", "lightgbm-tuned")`. Plan 5c extends to 4. Also adds an asymmetry pin for `lightgbm-nb` matching the existing `lightgbm` and `lightgbm-tuned` pins (LightGBMNbModel emits family=MIXED, which is also gated out of season aggregation today).

- [ ] **Step 1: Edit the smoke**

Open `tests/backtest/test_backtest_smoke.py`. Make four edits:

**(a)** Update the `model_classes` argument:

```python
        model_classes=("baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb"),
```

**(b)** Update the unique-class assertion:

```python
    assert set(out.metrics["model_class"].unique()) == {"baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb"}
```

**(c)** Update the per-model core-metrics loop:

```python
    for model_class in ("baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb"):
```

**(d)** After the existing `tuned_season_rows` block (which asserts `lightgbm-tuned` does not emit `season_calibration_*` rows), add an analogous block for the NB class:

```python
    # Plan 5c: lightgbm-nb emits family=MIXED rows which (like SAMPLED_SUMMARY-vs-non
    # gate today) are skipped by aggregate_to_season. TODO #28 widening still open.
    nb_metrics = out.metrics[out.metrics["model_class"] == "lightgbm-nb"]
    nb_season_rows = nb_metrics[
        nb_metrics["metric"].isin(["season_calibration_p10p90", "season_calibration_le_p90"])
    ]
    assert nb_season_rows.empty, (
        "lightgbm-nb is not expected to emit season_calibration_* rows yet; "
        f"got: {nb_season_rows.to_dict('records')}"
    )
```

Place this immediately after the existing `lightgbm-tuned` asymmetry block.

- [ ] **Step 2: Run the smoke**

```bash
. .venv/Scripts/activate
pytest -v tests/backtest/test_backtest_smoke.py
```

Expected: PASS. Runtime budget rises ~50s → ~65–80s (one extra LightGBM-NB fit on the (WR, 2024) cell). Within the <2min CI budget.

If the test is skipped because `data/features/wr/season=2024` doesn't exist, run `python scripts/refresh_features.py wr` first.

- [ ] **Step 3: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 4: Commit**

```bash
git add tests/backtest/test_backtest_smoke.py
git commit -m "$(cat <<'EOF'
test(backtest): smoke covers all 4 model classes — Plan 5c Phase 2

Default-on smoke now asserts baseline + lightgbm + lightgbm-tuned + lightgbm-nb
all produce finite metrics on (WR, 2024). Pins the MIXED-family
season-aggregation asymmetry for lightgbm-nb mirroring the existing pins
for lightgbm and lightgbm-tuned. Runtime ~65-80s; still within the <2min
CI budget.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Quad-model harness end-to-end test

**Files:**
- Create: `tests/test_backtest/test_harness_quad_model.py`

Mirrors `tests/test_backtest/test_harness_triple_model.py` exactly, with the fourth `model_class`. Gated by `@pytest.mark.backtest`.

- [ ] **Step 1: Create the file**

```python
"""End-to-end harness fold under all 4 model classes — Plan 5c Phase 2.

Mirrors test_harness_triple_model.py: single (WR, 2024) fold under
model_classes=(baseline, lightgbm, lightgbm-tuned, lightgbm-nb). Verifies
all four contribute rows for the cell and that the same model-class-
agnostic metric set is emitted for each.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from projections.backtest.harness import run_backtest
from projections.schemas import Position

_FEATURES_ROOT = Path("data/features")
_RAW_ROOT = Path("data/raw")


def _cache_present() -> bool:
    return (_FEATURES_ROOT / "wr" / "season=2024").exists() and (
        _RAW_ROOT / "weekly_stats" / "season=2024"
    ).exists()


@pytest.mark.backtest
@pytest.mark.skipif(
    not _cache_present(),
    reason="data/features/wr/season=2024 missing — run scripts/refresh_features.py wr",
)
def test_harness_runs_all_four_models_for_one_cell() -> None:
    """Run one (WR, 2024) fold with all 4 model classes; assert all four
    appear in the long-form metrics frame and that the core metric set
    is the same for each."""
    result = run_backtest(
        positions=[Position.WR],
        held_out_years=[2024],
        train_start=2018,
        model_classes=("baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb"),
    )
    df = result.metrics
    assert set(df["model_class"].unique()) == {
        "baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb",
    }
    core_metrics = {
        "composite_rmse",
        "composite_mae",
        "spearman_topN",
        "calibration_p10p90",
    }
    for model_class in ("baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb"):
        per_model = set(df[df["model_class"] == model_class]["metric"].unique())
        assert core_metrics.issubset(per_model), (
            f"model_class={model_class!r} missing core metrics; got {sorted(per_model)}"
        )
```

- [ ] **Step 2: Run the test**

```bash
. .venv/Scripts/activate
pytest -v -m backtest --run-backtest tests/test_backtest/test_harness_quad_model.py
```

Expected: PASS. Runtime ~70-90s.

- [ ] **Step 3: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 4: Commit**

```bash
git add tests/test_backtest/test_harness_quad_model.py
git commit -m "$(cat <<'EOF'
test(backtest): quad-model harness end-to-end — Plan 5c Phase 2

Mirrors test_harness_triple_model.py with the 4th model_class. Gated by
@pytest.mark.backtest + cache-present skipif.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Per-position parametrized smoke for `LightGBMNbModel`

**Files:**
- Create: `tests/test_models/test_lightgbm_nb_smoke.py`

Verbatim copy of `tests/test_models/test_lightgbm_tuned_smoke.py` with the factory key swapped to `"lightgbm-nb"` and a docstring update.

- [ ] **Step 1: Create the file**

Read the existing `tests/test_models/test_lightgbm_tuned_smoke.py` and create `tests/test_models/test_lightgbm_nb_smoke.py` with the same content, applying these single-token edits:

1. Update the module docstring at the top (line 1-12 area):
   - Replace "Plan 5b" → "Plan 5c".
   - Replace "LightGBMTunedModel" → "LightGBMNbModel".
   - Replace `factories["lightgbm-tuned"]` → `factories["lightgbm-nb"]` everywhere.
2. In the test body, replace every occurrence of `factories["lightgbm-tuned"]` with `factories["lightgbm-nb"]` (there's typically one such occurrence — the factory lookup inside the parametrized test body).

The resulting file should be ~121 lines, structurally identical to `test_lightgbm_tuned_smoke.py`.

- [ ] **Step 2: Run the smoke**

```bash
. .venv/Scripts/activate
pytest -v tests/test_models/test_lightgbm_nb_smoke.py
```

Expected: PASS for all 4 parametrized positions. Runtime ~30–45s.

- [ ] **Step 3: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 4: Commit**

```bash
git add tests/test_models/test_lightgbm_nb_smoke.py
git commit -m "$(cat <<'EOF'
test(models): parametrized per-position smoke for LightGBMNbModel — Plan 5c Phase 2

Verbatim copy of test_lightgbm_tuned_smoke.py with the factory key swapped
to 'lightgbm-nb'. Catches dispatch-table regressions across all 4 positions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — Run snapshot regen

Goal: regenerate `tests/backtest/model_metrics.json` with all 4 model classes; commit. ~95–155 minutes wall time on a typical dev box.

### Task 9: Regenerate the backtest snapshot

**Files:**
- Modify: `tests/backtest/model_metrics.json`

- [ ] **Step 1: Confirm feature cache is populated for 2018-2024**

```bash
. .venv/Scripts/activate
ls data/features/wr/season=2024/ data/features/qb/season=2024/ data/features/rb/season=2024/ data/features/te/season=2024/
```

Each directory should contain at least one `week=*/part.parquet`. If any are missing, run:

```bash
python scripts/refresh_features.py wr --seasons 2018 2019 2020 2021 2022 2023 2024
python scripts/refresh_features.py qb --seasons 2018 2019 2020 2021 2022 2023 2024
python scripts/refresh_features.py rb --seasons 2018 2019 2020 2021 2022 2023 2024
python scripts/refresh_features.py te --seasons 2018 2019 2020 2021 2022 2023 2024
```

- [ ] **Step 2: Run the full backtest with all 4 model classes**

```bash
python scripts/backtest.py --update-snapshot --model all
```

Expected: ~95–155 minutes wall time. Final stdout reports `Wrote tests/backtest/model_metrics.json` with the new row count.

- [ ] **Step 3: Verify the new snapshot has 1504 rows**

```bash
python -c "
import json
from collections import Counter
d = json.load(open('tests/backtest/model_metrics.json'))
mc_counts = Counter(r['model_class'] for r in d)
print('total rows:', len(d))
print('per-class:', dict(mc_counts))
"
```

Expected: total 1504 rows; per-class `{baseline: 400, lightgbm: 368, lightgbm-tuned: 368, lightgbm-nb: 368}`.

- [ ] **Step 4: Run the gate against the new snapshot**

```bash
pytest -v tests/backtest/test_backtest_gate.py
```

Expected: PASS (the gate compares against itself on the first run).

- [ ] **Step 5: Commit the snapshot**

```bash
git add tests/backtest/model_metrics.json
git commit -m "$(cat <<'EOF'
chore(backtest): regenerate snapshot with C-NB rows — Plan 5c Phase 3

1504 rows: 400 baseline + 368 lightgbm + 368 lightgbm-tuned + 368 lightgbm-nb.
The 32 season_calibration_* metrics for lightgbm-nb are skipped per the
existing SAMPLED_SUMMARY-only family gate (TODO #28 still open).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — Diagnostic report

Goal: per-cell A vs C vs C-tuned vs C-NB comparison; gate verdict; project_management.md + TODO.md updated.

### Task 10: Build the per-cell A vs C vs C-tuned vs C-NB comparison + verdict

**Files:**
- Modify: `project_management.md`

- [ ] **Step 1: Compute the §1.3 adoption-gate verdict for C-NB vs A**

```bash
. .venv/Scripts/activate
python <<'EOF'
import json, collections

d = json.load(open('tests/backtest/model_metrics.json'))
g = collections.defaultdict(dict)
for r in d:
    g[(r['position'], r['year'], r['metric'])][r['model_class']] = r['value']

positions = ['QB', 'RB', 'TE', 'WR']
years = [2021, 2022, 2023, 2024]

# Criterion 1: composite_rmse strictly lower on >=12/16; max worse <=1%.
rmse_lower = 0
max_worse = 0.0
for pos in positions:
    for year in years:
        a = g[(pos, year, 'composite_rmse')]['baseline']
        nb = g[(pos, year, 'composite_rmse')]['lightgbm-nb']
        pct = (nb - a) / a * 100
        if nb < a:
            rmse_lower += 1
        max_worse = max(max_worse, pct)
print(f"Criterion 1 (RMSE): C-NB < A on {rmse_lower}/16; max worse {max_worse:+.2f}%; "
      f"PASS={rmse_lower >= 12 and max_worse <= 1.0}")

# Criterion 2: spearman within +-0.005 on every cell.
sp_outside = 0
sp_max_abs = 0.0
for pos in positions:
    for year in years:
        a = g[(pos, year, 'spearman_topN')]['baseline']
        nb = g[(pos, year, 'spearman_topN')]['lightgbm-nb']
        delta = nb - a
        if abs(delta) > 0.005:
            sp_outside += 1
        sp_max_abs = max(sp_max_abs, abs(delta))
print(f"Criterion 2 (Spearman): {sp_outside}/16 outside +-0.005; max |delta| {sp_max_abs:.4f}; "
      f"PASS={sp_outside == 0}")

# Criterion 3: calibration no worse on any cell; mean delta >= +0.02.
calib_deltas = []
calib_worse = 0
for pos in positions:
    for year in years:
        a = g[(pos, year, 'calibration_p10p90')]['baseline']
        nb = g[(pos, year, 'calibration_p10p90')]['lightgbm-nb']
        delta = nb - a
        calib_deltas.append(delta)
        if delta < -0.005:
            calib_worse += 1
mean_delta = sum(calib_deltas) / len(calib_deltas)
print(f"Criterion 3 (Calib): worse on {calib_worse}/16; mean delta {mean_delta:+.4f}; "
      f"PASS={calib_worse == 0 and mean_delta >= 0.02}")
EOF
```

Capture the three pass/fail verdicts.

- [ ] **Step 2: Generate the per-cell comparison table (16 cells × 4 metrics × 4 models)**

```bash
PYTHONIOENCODING=utf-8 python <<'EOF' > /tmp/plan-5c-table.md
import json, collections
d = json.load(open('tests/backtest/model_metrics.json'))
g = collections.defaultdict(dict)
for r in d:
    g[(r['position'], r['year'], r['metric'])][r['model_class']] = r['value']

positions = ['QB', 'RB', 'TE', 'WR']
years = [2021, 2022, 2023, 2024]

print("| Cell | RMSE A | RMSE C | RMSE Ctuned | RMSE Cnb | RMSE Cnb-A % | "
      "Spearman A | Spearman Cnb | Spearman Cnb-A | Calib A | Calib Cnb | Calib Cnb-A |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for pos in positions:
    for year in years:
        a_rmse = g[(pos, year, 'composite_rmse')]['baseline']
        c_rmse = g[(pos, year, 'composite_rmse')]['lightgbm']
        ct_rmse = g[(pos, year, 'composite_rmse')]['lightgbm-tuned']
        nb_rmse = g[(pos, year, 'composite_rmse')]['lightgbm-nb']
        a_sp = g[(pos, year, 'spearman_topN')]['baseline']
        nb_sp = g[(pos, year, 'spearman_topN')]['lightgbm-nb']
        a_cal = g[(pos, year, 'calibration_p10p90')]['baseline']
        nb_cal = g[(pos, year, 'calibration_p10p90')]['lightgbm-nb']
        rmse_pct = (nb_rmse - a_rmse) / a_rmse * 100
        sp_delta = nb_sp - a_sp
        cal_delta = nb_cal - a_cal
        print(f"| {pos} {year} | {a_rmse:.4f} | {c_rmse:.4f} | {ct_rmse:.4f} | "
              f"{nb_rmse:.4f} | {rmse_pct:+.2f}% | {a_sp:.4f} | {nb_sp:.4f} | "
              f"{sp_delta:+.4f} | {a_cal:.4f} | {nb_cal:.4f} | {cal_delta:+.4f} |")
EOF
cat /tmp/plan-5c-table.md
```

- [ ] **Step 3: Capture per-position model_ids for the new C-NB models**

```bash
python -c "
from projections.models import qb_lightgbm_nb, rb_lightgbm_nb, te_lightgbm_nb, wr_lightgbm_nb
for f in (qb_lightgbm_nb, rb_lightgbm_nb, te_lightgbm_nb, wr_lightgbm_nb):
    m = f()
    print(f'{m.position.value}: code_hash={m.code_hash}')
"
```

Capture the 4 hashes for the report.

- [ ] **Step 4: Append the report section to `project_management.md`**

Open `project_management.md`. Insert a new top section (above the existing Plan 5b section) following this template, filling in the values from Steps 1-3:

```markdown
## Plan 5c — Hybrid LightGBM with NB-2 for Count Stats (Model C-NB) — shipped (run YYYY-MM-DD)

**Closes:** the count-stat-bias mechanism identified in Plan 5b's diagnostic.

`LightGBMNbModel` (Model C-NB) lands as a fourth peer of Models A, C,
C-tuned. Subclasses `LightGBMTunedModel`; for the 13 count cells Plan 3e
routes through NB-2 in Ridge, trains one `lgb.LGBMRegressor(objective="poisson")`
per stat, fits NB-2 dispersion on training residuals, predicts via
`ParametricNegativeBinomial`. Yards/receptions stats unchanged from
Model C-tuned (5-quantile + QuantileDistribution). Reuses Plan 5b's tuned
hyperparameters from `data/tuned_params/lightgbm.json`. Per-row family
is `MIXED`; per-stat families remain encoded inside the params blob.

Snapshot extended 1136 → 1504 rows (368 new lightgbm-nb rows).

### Per-position model_ids

| Position | Model A | Model C | Model C-tuned | Model C-NB |
|---|---|---|---|---|
| WR | `baseline:wr:6d955427:2018-2023` | `lightgbm:wr:a4dd5a82:2018-2023` | `lightgbm-tuned:wr:62df14ad:2018-2023` | `lightgbm-nb:wr:<hash>:2018-2023` |
| QB | `baseline:qb:c98738f3:2018-2023` | `lightgbm:qb:06fadb3f:2018-2023` | `lightgbm-tuned:qb:fc902ed6:2018-2023` | `lightgbm-nb:qb:<hash>:2018-2023` |
| RB | `baseline:rb:5a86c8ee:2018-2023` | `lightgbm:rb:fb169c0e:2018-2023` | `lightgbm-tuned:rb:5d69fdfe:2018-2023` | `lightgbm-nb:rb:<hash>:2018-2023` |
| TE | `baseline:te:9c00025b:2018-2023` | `lightgbm:te:bd4c2a5b:2018-2023` | `lightgbm-tuned:te:89dafdb6:2018-2023` | `lightgbm-nb:te:<hash>:2018-2023` |

(Replace the `<hash>` placeholders with the values from Step 3.)

### Adoption-gate verdict — Model C-NB vs Model A

| Criterion | Threshold | Actual (C-NB vs A) | Pass? |
|---|---|---|---|
| Composite RMSE strictly lower on >=12/16; max +1% worse | C-NB < A on 12+; max +1% worse | <fill from Step 1> | <PASS/FAIL> |
| Spearman top-N within +-0.005 on every cell | All within ±0.005 | <fill> | <PASS/FAIL> |
| Calibration no worse on any cell; mean delta >= +0.02 | No regressions; mean ≥ +0.02 | <fill> | <PASS/FAIL> |

### Side-by-side per-cell comparison

<paste table from Step 2>

### Why this should work / does it work

The Plan 5b diagnostic identified the mechanism: 5-knot QuantileDistribution
linear interpolation + sort + clip produces a biased empirical mean on
zero-inflated count stats (over-prediction of 30-60% on TE/WR receiving_tds,
QB rushing_tds/fumbles, etc.). NB-2 with mean = mu_hat and dispersion fit on
training residuals does not have this bias — the empirical mean of NB-2
samples ≈ mu_hat by construction.

<2-3 sentence analysis: did the per-cell results match the prediction? Did
the count-stat per-stat RMSE deltas collapse as predicted? Are yards stats
unchanged from Model C-tuned (they should be, since the yards-stat fit path
is inherited unchanged)?>

### Decision

<If gate passes: "Adoption verdict: ADOPT Model C-NB as the production default.
File a follow-up housekeeping commit for the production-default switch and
prune Model C (now strictly dominated by Model C-tuned and C-NB)." If gate
fails: "Adoption verdict: <fail-mode>. <Next-step decision: Plan 5d for
poisson-objective retuning, Plan 6 ensemble, or pivot to feature work.>">

---
```

(Replace `<...>` placeholders with values from Steps 1-2 and the analysis text.)

- [ ] **Step 5: Update the Next-action section in `project_management.md`**

Find the "## Next action" header (search for `^## Next action`). Replace its content with:

```markdown
## Next action

**Plan 5c (Model C-NB — hybrid LightGBM with NB-2 for count stats) shipped.**

<If gate passed: "Adoption verdict: PASS. File the production-default switch
in a housekeeping commit; consider Plan 6 (ensemble of A + C-NB) as the next
model-improvement track.">

<If gate failed: "Adoption verdict: FAIL on <criteria>. Three remaining tracks:
Plan 5d (re-tune for poisson), Plan 6 (ensemble), TODO #3 (PBP features).">

After model-improvement work: Plan 4 (public Python API + CLI verbs +
free-tier hosting), then Draft Hub.
```

- [ ] **Step 6: Commit**

```bash
git add project_management.md
git commit -m "$(cat <<'EOF'
docs(plan-5c): record Model C-NB ship + adoption-gate verdict — Plan 5c Phase 4

Per-cell A vs C vs C-tuned vs C-NB table; the three §1.3 criteria evaluated
for C-NB vs A; analysis of count-stat per-stat RMSE deltas confirms the
diagnostic prediction (or documents how the actual deltas differed from
the predicted ones). Next-action section updated with the post-5c verdict.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Update TODO.md

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Edit `TODO.md` based on the gate verdict**

Open `TODO.md`. Locate TODO #26 (the Plan 5 / 5b consolidated entry). Append a Plan 5c section:

```markdown
**Plan 5c (Model C-NB — hybrid LightGBM with NB-2 for count stats):**
`LightGBMNbModel` subclass overriding only count-stat training and prediction;
yards stats inherited unchanged from `LightGBMTunedModel`. Snapshot extended
(1136 → 1504 rows). **Adoption verdict: <PASS or FAIL on each of the §1.3
criteria>.** <If PASS: "Production-default switch and Model C pruning land
in a follow-up housekeeping commit.">. <If FAIL: "Pivot to <Plan 5d
re-tuning / Plan 6 ensemble / feature work>.">
```

If the gate **passed**, also file a new TODO for the production-default switch + Model C pruning:

```markdown
### 30. Production-default switch to Model C-NB + Model C pruning

Plan 5c's Model C-NB passed the §1.3 adoption gate. Production default
should switch from Model A to Model C-NB. Concrete tasks:

- Update the default factory dispatch in `src/projections/models/__init__.py`
  if applicable (the default is currently the `"baseline"` key in
  POSITION_DISPATCH; production callers select via `factories[<arg>]`).
- Update CLI defaults in `scripts/backtest.py`, `scripts/train_baseline.py`,
  `scripts/predict_2024.py`, `scripts/sanity_check_baseline.py` to default
  to `"lightgbm-nb"` where they currently default to `"baseline"` or `"both"`.
- Prune Model C (`"lightgbm"`) from POSITION_DISPATCH — now strictly
  dominated by Model C-tuned (Plan 5b verdict) and Model C-NB (Plan 5c
  verdict). Drop `LightGBMModel` factories or keep as historical anchors.
- Delete or migrate Plan 5 / 5b backtest snapshot rows under model_class=
  "lightgbm" if pruned.

Estimated complexity: small (~1 day). One PR.
```

- [ ] **Step 2: Commit**

```bash
git add TODO.md
git commit -m "$(cat <<'EOF'
docs(todo): record Plan 5c adoption verdict; <file Plan 5d/Plan 6 follow-up | file production-default switch> — Plan 5c Phase 4

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

After Task 11 commits:

- [ ] **Step 1: Run the full test suite**

```bash
. .venv/Scripts/activate
pytest -v
```

Expected: all tests PASS. The 12 skipped tests (the opt-in `--run-backtest` and `--run-network` gates) are unchanged.

- [ ] **Step 2: Run mypy + ruff**

```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin feat/plan-5c-nb-counts
gh pr create --title "Plan 5c: hybrid LightGBM with NB-2 for count stats (Model C-NB)" --body "$(cat <<'EOF'
## Summary
- Adds `LightGBMNbModel` (Model C-NB) as a fourth peer model class. Subclasses `LightGBMTunedModel`.
- For count stats (the 13 cells Plan 3e routes through NB-2 in Ridge): trains `objective="poisson"` regressor with Plan 5b's tuned hyperparameters; fits NB-2 dispersion on training residuals; predicts via `ParametricNegativeBinomial`. Yards stats inherit from Model C-tuned unchanged.
- Refactors `_negative_binomial_dispersion_from_residuals` from `models/baseline.py` to a public `nb_dispersion_from_residuals` in `distributions/parametric.py`. Behavior-preserving move.
- Adds `DistributionFamily.MIXED` row-level family for mixed-per-stat-family rows.
- Backtest snapshot extended 1136 → 1504 rows.
- Adoption-gate verdict: <PASS / FAIL — fill in from project_management.md>.

## Spec / plan
- Spec: `docs/superpowers/specs/2026-04-28-plan-5c-nb-counts-design.md`
- Plan: `docs/superpowers/plans/2026-04-28-plan-5c-nb-counts.md`
- Builds on Plan 5b (PR #12).

## Test plan
- [x] `pytest -v` clean
- [x] `mypy src tests` clean
- [x] `ruff check src tests` and `ruff format --check src tests` clean
- [x] `pytest -m backtest --run-backtest` clean against the regenerated `model_metrics.json`
- [x] Default-on smoke covers all 4 model classes
- [x] Per-position parametrized smoke covers all 4 positions
- [x] Cross-cutting tests verify count vs yards stat routing, MIXED family, codec round-trip

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Summary of files touched

| Phase | New files | Modified files |
|---|---|---|
| 0 | `tests/test_distributions/test_nb_dispersion.py` | `src/projections/distributions/parametric.py`, `src/projections/models/baseline.py` |
| 1 | `src/projections/models/lightgbm_nb.py`, `tests/test_models/test_lightgbm_nb.py` | `src/projections/schemas.py`, `src/projections/models/__init__.py` |
| 2 | `tests/test_backtest/test_harness_quad_model.py`, `tests/test_models/test_lightgbm_nb_smoke.py` | `scripts/backtest.py`, `tests/backtest/test_backtest_smoke.py` |
| 3 | — | `tests/backtest/model_metrics.json` |
| 4 | — | `project_management.md`, `TODO.md` |

Each phase ≤5 files per the CLAUDE.md "PHASED EXECUTION" rule.
