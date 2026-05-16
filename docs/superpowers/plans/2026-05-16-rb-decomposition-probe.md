# RB Rushing + Receiving Decomposition Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Probe whether decomposing RB stats into two shared volume axes (`carries`, `targets`) × per-stat efficiency factors beats the current per-stat RidgeCV on out-of-sample mean prediction. Returns five per-stat verdicts (SIGNAL / NULL / REGRESSION) over 2021-2024 walk-forward. NULL on all five closes RB decomposition cheaply; SIGNAL on any stat greenlights an integration plan.

**Architecture:** New module `src/projections/backtest/rb_decomposition_probe.py` — a self-contained CV harness extending PR #32's pattern (`target_decomposition_probe.py`) to two volume axes and five composed stats. Pure numpy/pandas/sklearn. Both arms use RidgeCV everywhere with predict-time clipping; any SIGNAL is attributable to *decomposition itself*, not a model-class change. No new ingest, schema, codec, or factory changes. CLI loads RB feature cache + weekly stats and dispatches the harness.

**Tech Stack:** Python, numpy, pandas, sklearn (`RidgeCV`), pandera, pytest, mypy strict, ruff. Reuses `paired_bootstrap_rmse_delta` + `BootstrapDelta` from `adoption_gate.py`; imports `_RIDGE_ALPHA_GRID` + `_RB_FEATURE_COLUMNS` from `baseline.py`.

**Spec:** `docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md`

**Branch:** `feat/probe-rb-decomposition` (worktree at `.worktrees/feat-probe-rb-decomposition`).

---

## Pre-task: re-point venv editable install

The `.venv` at `C:/Users/alden/FantasyFootball/.venv` is editable-installed against a different worktree. Re-point it before starting Task 1:

```bash
cd C:/Users/alden/FantasyFootball/.worktrees/feat-probe-rb-decomposition
../../.venv/Scripts/python.exe -m pip install -e . --no-deps
```

Verify:

```bash
../../.venv/Scripts/python.exe -c "from importlib.metadata import distribution; import json; print(json.loads(distribution('projections').read_text('direct_url.json'))['url'])"
```

Expected output: `file:///C:/Users/alden/FantasyFootball/.worktrees/feat-probe-rb-decomposition`

---

## File Map

**Create:**
- `src/projections/backtest/rb_decomposition_probe.py` — probe core: `_RB_DECOMPS` registry, fit/predict helpers, dataclasses, walk-forward, verdict mapping.
- `scripts/probe_rb_decomposition.py` — argparse CLI driver.
- `tests/test_backtest/test_rb_decomposition_probe.py` — unit + integration tests.
- `tests/test_scripts/test_probe_rb_decomposition_cli.py` — CLI smoke.
- `reports/feature_probe_rb_decomposition_summary.md` — Task 5 output (CLI-written).
- `reports/feature_probe_rb_decomposition_per_stat.csv` — Task 5 output (CLI-written).

**Modify (Task 5 only):**
- `project_management.md` — top-of-file decision-log entry.
- `TODO.md` — update entry under the factor-appropriate / decomposition chain with verdict + next direction.

**Untouched (deliberately):**
- `src/projections/models/baseline.py` — probe imports `_RIDGE_ALPHA_GRID` and `_RB_FEATURE_COLUMNS` but does not modify.
- `src/projections/models/decomposed_baseline.py` — integration territory if SIGNAL.
- `src/projections/backtest/target_decomposition_probe.py` / `logit_catch_rate_probe.py` / `tweedie_yards_per_target_probe.py` — siblings; probe is its own module.
- `src/projections/schemas.py` — no new schemas.

---

## Task 1: Sub-model primitives + registry + dataclasses

**Files:**
- Create: `src/projections/backtest/rb_decomposition_probe.py`
- Create: `tests/test_backtest/test_rb_decomposition_probe.py`

Scope: module skeleton, `_RB_DECOMPS` registry (5 entries), `_StatDecomp` dataclass, `_fit_direct`, `_fit_decomposed_volume`, `_fit_decomposed_efficiency`, `_predict_direct`, `_predict_decomposed`. Plus their unit tests.

- [ ] **Step 1.1: Re-point venv per pre-task block above.** Confirm with the verify command.

- [ ] **Step 1.2: Write the failing tests**

Create `tests/test_backtest/test_rb_decomposition_probe.py`:

```python
"""Tests for src/projections/backtest/rb_decomposition_probe.py.

Spec: docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV

from projections.backtest.rb_decomposition_probe import (
    _RB_DECOMPS,
    _RIDGE_ALPHAS,
    _fit_decomposed_efficiency,
    _fit_decomposed_volume,
    _fit_direct,
    _predict_decomposed,
    _predict_direct,
    _StatDecomp,
)
from projections.schemas import Stat


def test_rb_decomps_registry_has_five_stats_across_two_volume_axes() -> None:
    """Five composed stats: 2 rushing (carries axis) + 3 receiving (targets axis)."""
    assert set(_RB_DECOMPS.keys()) == {
        Stat.RUSHING_YARDS,
        Stat.RUSHING_TDS,
        Stat.RECEPTIONS,
        Stat.RECEIVING_YARDS,
        Stat.RECEIVING_TDS,
    }
    # Rushing axis.
    assert _RB_DECOMPS[Stat.RUSHING_YARDS].volume_stat is Stat.CARRIES
    assert _RB_DECOMPS[Stat.RUSHING_TDS].volume_stat is Stat.CARRIES
    # Receiving axis.
    assert _RB_DECOMPS[Stat.RECEPTIONS].volume_stat is Stat.TARGETS
    assert _RB_DECOMPS[Stat.RECEIVING_YARDS].volume_stat is Stat.TARGETS
    assert _RB_DECOMPS[Stat.RECEIVING_TDS].volume_stat is Stat.TARGETS

    # Clip semantics: rate factors -> 1.0; unbounded efficiency -> +inf.
    assert _RB_DECOMPS[Stat.RUSHING_YARDS].efficiency_clip_hi == float("inf")
    assert _RB_DECOMPS[Stat.RUSHING_TDS].efficiency_clip_hi == 1.0
    assert _RB_DECOMPS[Stat.RECEPTIONS].efficiency_clip_hi == 1.0
    assert _RB_DECOMPS[Stat.RECEIVING_YARDS].efficiency_clip_hi == float("inf")
    assert _RB_DECOMPS[Stat.RECEIVING_TDS].efficiency_clip_hi == 1.0

    # numerator_stat == the key for every entry (composition invariant).
    for stat, decomp in _RB_DECOMPS.items():
        assert decomp.numerator_stat is stat


def test_fit_direct_fits_ridgecv_on_synthetic_linear_relationship() -> None:
    """Direct fit recovers a clean linear slope within tolerance."""
    rng = np.random.default_rng(seed=2026)
    x = rng.uniform(0.0, 5.0, size=(200, 3)).astype(np.float64)
    # true: y = 2 * x[:, 0] + 1 * x[:, 1] - 0.5 * x[:, 2] + noise
    y = (
        2.0 * x[:, 0]
        + 1.0 * x[:, 1]
        - 0.5 * x[:, 2]
        + rng.normal(0, 0.2, size=200)
    ).astype(np.float64)

    ridge = _fit_direct(x, y)

    assert isinstance(ridge, RidgeCV)
    assert ridge.alpha_ in _RIDGE_ALPHAS
    assert abs(ridge.coef_[0] - 2.0) < 0.1
    assert abs(ridge.coef_[1] - 1.0) < 0.1


def test_fit_decomposed_volume_fits_on_unfiltered_rows() -> None:
    """Volume sub-model trains on ALL rows including volume == 0
    (zero-volume rows are legitimate observations of low-volume players).
    """
    rng = np.random.default_rng(seed=2027)
    x = rng.uniform(0.0, 1.0, size=(150, 2)).astype(np.float64)
    # carries integer with ~30% zeros
    carries = np.where(
        rng.uniform(0, 1, size=150) < 0.3,
        0,
        rng.poisson(10, size=150),
    ).astype(np.int64)

    ridge = _fit_decomposed_volume(x, carries)

    assert isinstance(ridge, RidgeCV)
    # n_samples_seen_ on RidgeCV is not exposed; just verify the predictions
    # are non-trivial (i.e. the model used all the rows).
    pred = ridge.predict(x)
    assert pred.shape == (150,)


def test_fit_decomposed_efficiency_fits_only_on_positive_volume_rows() -> None:
    """Efficiency sub-model trains only on rows where volume > 0."""
    rng = np.random.default_rng(seed=2028)
    x = rng.uniform(0.0, 1.0, size=(100, 2)).astype(np.float64)
    volume = np.where(
        rng.uniform(0, 1, size=100) < 0.5,
        0,
        rng.poisson(8, size=100),
    ).astype(np.int64)
    # numerator ~ volume * (4 + 2 * x[:, 0])
    rate = 4.0 + 2.0 * x[:, 0]
    numerator = (volume.astype(np.float64) * rate).astype(np.int64)

    ridge = _fit_decomposed_efficiency(x, numerator, volume)

    assert isinstance(ridge, RidgeCV)
    # The slope on x[:, 0] should be ~2.0 (the rate's coefficient).
    assert abs(ridge.coef_[0] - 2.0) < 0.5


def test_fit_decomposed_efficiency_raises_if_no_positive_volume() -> None:
    """If every training row has volume == 0, the efficiency fit cannot
    proceed (division-by-zero in the ratio).
    """
    import pytest

    x = np.zeros((10, 3), dtype=np.float64)
    volume = np.zeros(10, dtype=np.int64)
    numerator = np.zeros(10, dtype=np.int64)

    with pytest.raises(ValueError, match=r"no training rows with volume > 0"):
        _fit_decomposed_efficiency(x, numerator, volume)


def test_predict_direct_passes_through_no_clipping() -> None:
    """Direct prediction does NOT clip (mirrors BaselineModel; downstream
    Distribution constructor handles family floors).
    """
    rng = np.random.default_rng(seed=2029)
    x_train = rng.uniform(0.0, 1.0, size=(50, 2)).astype(np.float64)
    # Force negative predictions: train y = -1 + ...
    y_train = (-1.0 + rng.normal(0, 0.05, size=50)).astype(np.float64)
    ridge = _fit_direct(x_train, y_train)

    x_eval = rng.uniform(0.0, 1.0, size=(5, 2)).astype(np.float64)
    pred = _predict_direct(ridge, x_eval)

    assert pred.shape == (5,)
    # Predictions are around -1 — no clip applied.
    assert (pred < 0).all()


def test_predict_decomposed_applies_two_sided_clip_on_rate_efficiency() -> None:
    """Decomposed prediction clips efficiency to [0, clip_hi]; volume to [0, +inf).

    Two-sided clip engages on rate factors (clip_hi = 1.0); only low-side on
    yards_per_volume (clip_hi = +inf).
    """
    rng = np.random.default_rng(seed=2030)
    x_train = rng.uniform(0.0, 1.0, size=(100, 2)).astype(np.float64)
    # Volume Ridge fit on a synthetic carries response.
    volume_y = (10.0 + 5.0 * x_train[:, 0] + rng.normal(0, 0.5, size=100)).astype(np.float64)
    volume_ridge = _fit_direct(x_train, volume_y)
    # Efficiency Ridge that will predict > 1 on high-x_eval (forces clip_hi=1.0).
    rate_y = (0.5 + 1.5 * x_train[:, 0] + rng.normal(0, 0.05, size=100)).astype(np.float64)
    rate_ridge = _fit_direct(x_train, rate_y)

    x_eval = np.array([[1.0, 0.5]], dtype=np.float64)  # rate prediction ~ 2.0 unclipped.

    # Rate factor with clip_hi=1.0: predicted efficiency caps at 1.0.
    pred_clipped = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=rate_ridge,
        x=x_eval,
        efficiency_clip_hi=1.0,
    )
    # Volume at x=1.0 is ~15; efficiency clipped to 1.0 -> result ~ 15.0
    assert 10.0 < pred_clipped[0] < 20.0

    # Same data, clip_hi=+inf: predicted efficiency is unclipped.
    pred_unclipped = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=rate_ridge,
        x=x_eval,
        efficiency_clip_hi=float("inf"),
    )
    # Now efficiency ~ 2.0 -> result ~ 30.
    assert pred_unclipped[0] > pred_clipped[0]


def test_predict_decomposed_clips_negative_volume_to_zero() -> None:
    """Volume predictions below 0 are clipped (can't have negative carries)."""
    rng = np.random.default_rng(seed=2031)
    x_train = rng.uniform(0.0, 1.0, size=(50, 2)).astype(np.float64)
    # Force negative volume predictions on the eval-low-x side.
    volume_y = (-5.0 + 10.0 * x_train[:, 0] + rng.normal(0, 0.5, size=50)).astype(
        np.float64
    )
    volume_ridge = _fit_direct(x_train, volume_y)
    eff_y = (5.0 + 2.0 * x_train[:, 0] + rng.normal(0, 0.1, size=50)).astype(np.float64)
    eff_ridge = _fit_direct(x_train, eff_y)

    x_eval = np.array([[0.0, 0.5]], dtype=np.float64)  # volume ~ -5 unclipped.

    pred = _predict_decomposed(
        volume_ridge=volume_ridge,
        efficiency_ridge=eff_ridge,
        x=x_eval,
        efficiency_clip_hi=float("inf"),
    )
    # Volume clipped to 0 -> result is 0.
    assert pred[0] == 0.0
```

- [ ] **Step 1.3: Run tests — expect ImportError**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_rb_decomposition_probe.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'projections.backtest.rb_decomposition_probe'`.

- [ ] **Step 1.4: Create the module with registry + helpers**

Create `src/projections/backtest/rb_decomposition_probe.py`:

```python
"""RB rushing + receiving decomposition probe — model architecture probe.

Two-arm probe extending PR #32's target_decomposition_probe to RB across two
shared volume axes:
  - rushing: carries x (yards_per_carry, td_rate_per_carry)
  - receiving: targets x (catch_rate, yards_per_target, td_rate_per_target)

Five composed stats, two shared volume sub-models per training window
(carries, targets). Per-stat Delta-RMSE x 5 stats x walk-forward eval window
2021-2024, paired-bootstrap CI on pooled residuals.

Sub-model = RidgeCV everywhere with predict-time clipping; any SIGNAL is
attributable to decomposition itself, not a model-class change.

Spec: docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.linear_model import RidgeCV

from projections.models.baseline import _RIDGE_ALPHA_GRID
from projections.schemas import Stat

# Canonical Ridge alpha grid shared with BaselineModel.fit. The probe's
# incumbent (direct) arm and decomposed arm both use this grid, so any SIGNAL
# is attributable to the decomposition recipe, not a regularization-scale
# difference. PR #44 review-fix: import from baseline, do not redefine.
_RIDGE_ALPHAS: Final[np.ndarray] = _RIDGE_ALPHA_GRID


@dataclass(frozen=True, slots=True)
class _StatDecomp:
    """Per-stat decomposition spec.

    The probe measures one decomposed prediction per stat:
        mu_decomposed[stat] = clip(volume_ridge.predict(X), 0, +inf)
                            * clip(efficiency_ridge.predict(X), 0, efficiency_clip_hi)

    Volume sub-models (one per unique `volume_stat`) are fit once per training
    window. Efficiency sub-models are fit on the volume_stat > 0 row subset.
    """

    volume_stat: Stat
    efficiency_label: str
    efficiency_clip_hi: float
    numerator_stat: Stat


_RB_DECOMPS: Final[dict[Stat, _StatDecomp]] = {
    Stat.RUSHING_YARDS: _StatDecomp(
        volume_stat=Stat.CARRIES,
        efficiency_label="yards_per_carry",
        efficiency_clip_hi=float("inf"),
        numerator_stat=Stat.RUSHING_YARDS,
    ),
    Stat.RUSHING_TDS: _StatDecomp(
        volume_stat=Stat.CARRIES,
        efficiency_label="td_rate_per_carry",
        efficiency_clip_hi=1.0,
        numerator_stat=Stat.RUSHING_TDS,
    ),
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


def _fit_direct(x: np.ndarray, y: np.ndarray) -> RidgeCV:
    """Fit RidgeCV(_RIDGE_ALPHAS) on (x, y).

    Caller is responsible for: NaN drop, bool-to-int8 coercion of x cols,
    volume-positive filtering when fitting an efficiency factor. Pure: arrays
    in, fitted ridge out.
    """
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS)
    ridge.fit(x, y.astype(np.float64))
    return ridge


def _fit_decomposed_volume(x: np.ndarray, volume: np.ndarray) -> RidgeCV:
    """Fit a shared volume sub-model on `volume` directly.

    Trained on UN-FILTERED training rows (zero-volume rows are legitimate
    observations of low-volume / out-of-rotation players).

    Called twice per training window: once with volume = carries (for the
    rushing axis), once with volume = targets (for the receiving axis).
    """
    return _fit_direct(x, volume.astype(np.float64))


def _fit_decomposed_efficiency(
    x: np.ndarray,
    numerator: np.ndarray,
    volume: np.ndarray,
) -> RidgeCV:
    """Fit an efficiency sub-model on rows where volume > 0.

    Ratio = numerator / volume on the masked subset. Caller passes `numerator`
    already aligned with `x` and `volume`; this helper handles the masking.

    Raises:
        ValueError: no rows in the training set have volume > 0.
    """
    mask = volume > 0
    if not mask.any():
        raise ValueError(
            "Cannot fit efficiency factor: no training rows with volume > 0. "
            "Check the training-window filter."
        )
    x_pos = x[mask]
    volume_pos = volume[mask].astype(np.float64)
    numerator_pos = numerator[mask].astype(np.float64)
    ratio = numerator_pos / volume_pos
    return _fit_direct(x_pos, ratio)


def _predict_direct(ridge: RidgeCV, x: np.ndarray) -> np.ndarray:
    """Direct per-row mu prediction. No clipping (matches BaselineModel)."""
    pred: np.ndarray = ridge.predict(x).astype(np.float64)
    return pred


def _predict_decomposed(
    *,
    volume_ridge: RidgeCV,
    efficiency_ridge: RidgeCV,
    x: np.ndarray,
    efficiency_clip_hi: float,
) -> np.ndarray:
    """Decomposed per-row mu prediction.

    mu = clip(volume.predict(x), 0, +inf) * clip(efficiency.predict(x), 0, hi)

    Volume clip floors at 0 (negative carries / targets impossible). Efficiency
    clip floors at 0 for all factors; ceiling is 1.0 for rate factors
    (catch_rate, td_rate_per_carry, td_rate_per_target) and +inf for unbounded
    efficiency (yards_per_carry, yards_per_target).
    """
    volume_raw: np.ndarray = volume_ridge.predict(x).astype(np.float64)
    volume = np.maximum(volume_raw, 0.0)
    eff_raw: np.ndarray = efficiency_ridge.predict(x).astype(np.float64)
    eff = np.clip(eff_raw, 0.0, efficiency_clip_hi)
    result: np.ndarray = volume * eff
    return result
```

- [ ] **Step 1.5: Run tests — expect PASS**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_rb_decomposition_probe.py -v
```

Expected: 8 tests pass.

- [ ] **Step 1.6: Lint + typecheck**

```bash
../../.venv/Scripts/python.exe -m ruff check src tests
../../.venv/Scripts/python.exe -m ruff format --check src tests
../../.venv/Scripts/python.exe -m mypy src tests
```

Expected: zero violations.

- [ ] **Step 1.7: Commit**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  src/projections/backtest/rb_decomposition_probe.py \
  tests/test_backtest/test_rb_decomposition_probe.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(probe): RB decomposition probe primitives + 5-stat registry

_RB_DECOMPS maps 5 composed stats to (volume_stat, efficiency_label,
efficiency_clip_hi, numerator_stat): 2 rushing entries (carries axis) and
3 receiving entries (targets axis). _fit_direct / _fit_decomposed_volume /
_fit_decomposed_efficiency / _predict_direct / _predict_decomposed are the
pure-array primitives the walk-forward harness composes in Task 2.

_fit_decomposed_efficiency raises ValueError if no rows have volume > 0;
both predict helpers clip volume at 0 (negative carries/targets impossible);
efficiency clip ceiling is per-_StatDecomp (1.0 for rate factors, +inf for
unbounded efficiency).

Spec: docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md (Task 1).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Dataclasses + walk-forward harness + integration tests

**Files:**
- Modify: `src/projections/backtest/rb_decomposition_probe.py` (extend)
- Modify: `tests/test_backtest/test_rb_decomposition_probe.py` (extend)

Scope: `StatResiduals`, `FactorResidualsByYear`, `CoverageByYear`, `WalkForwardOutput` dataclasses + `walk_forward_residuals` function + integration tests.

- [ ] **Step 2.1: Append failing tests**

Append to `tests/test_backtest/test_rb_decomposition_probe.py`:

```python
import pandas as pd

from projections.backtest.rb_decomposition_probe import (
    CoverageByYear,
    FactorResidualsByYear,
    StatResiduals,
    WalkForwardOutput,
    walk_forward_residuals,
)


def _synthetic_rb_inputs(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small synthetic RB (features, weekly_stats) pair for walk-forward
    integration testing.

    4 seasons x 4 weeks x 8 players = 128 rows. Features uniform random;
    truth uses two correlated volume axes (carries, targets) and matched
    efficiency factors so both arms have signal to extract.
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

    feature_schema = POSITION_DISPATCH[Position.RB].feature_schema
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
    ws["position"] = "RB"
    n = len(ws)
    carries_lambda = rng.uniform(8.0, 16.0, size=n)
    carries = np.maximum(1, rng.poisson(carries_lambda)).astype(np.int64)
    targets_lambda = rng.uniform(2.0, 6.0, size=n)
    targets = np.maximum(1, rng.poisson(targets_lambda)).astype(np.int64)
    # Yards per carry ~ N(4.3, 1.0); per-row mean from carries_lambda for signal.
    ypc = np.maximum(0.0, 4.3 + 0.05 * carries_lambda + rng.normal(0, 0.5, size=n))
    rushing_yards = np.clip(carries.astype(np.float64) * ypc, -50.0, 400.0)
    # TD rate per carry — small, ~0.03; bumped a bit by goal-line indicator (proxy
    # via carries_lambda).
    td_rate_carry = np.clip(0.02 + 0.005 * (carries_lambda - 12.0), 0.0, 0.1)
    rushing_tds = rng.binomial(carries, td_rate_carry).astype(np.int64)
    # Catch_rate ~ 0.7 for RBs; small variability.
    catch_rate = np.clip(0.7 + 0.05 * rng.normal(0, 1.0, size=n), 0.1, 1.0)
    receptions = rng.binomial(targets, catch_rate).astype(np.int64)
    # Yards per target ~ 6 yards (short routes for RBs).
    ypt = np.maximum(0.0, 6.0 + 0.1 * targets_lambda + rng.normal(0, 0.5, size=n))
    receiving_yards = np.clip(targets.astype(np.float64) * ypt, -50.0, 400.0).astype(
        np.float64
    )
    td_rate_target = np.clip(0.04 + rng.normal(0, 0.02, size=n), 0.0, 0.2)
    receiving_tds = rng.binomial(targets, td_rate_target).astype(np.int64)

    ws["carries"] = carries
    ws["targets"] = targets
    ws["rushing_yards"] = rushing_yards
    ws["rushing_tds"] = rushing_tds
    ws["receptions"] = receptions
    ws["receiving_yards"] = receiving_yards
    ws["receiving_tds"] = receiving_tds
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


def test_walk_forward_residuals_populates_all_five_stat_buffers() -> None:
    """Every entry in _RB_DECOMPS gets a StatResiduals; both arms populated."""
    features, weekly_stats = _synthetic_rb_inputs()
    output = walk_forward_residuals(
        features, weekly_stats, eval_years=(2020, 2021)
    )

    assert isinstance(output, WalkForwardOutput)
    assert set(output.per_stat.keys()) == set(_RB_DECOMPS.keys())
    for stat, residuals in output.per_stat.items():
        assert isinstance(residuals, StatResiduals)
        assert residuals.actual.shape == residuals.mu_direct.shape
        assert residuals.actual.shape == residuals.mu_decomposed.shape
        assert residuals.n_paired == residuals.actual.shape[0]
        assert residuals.n_paired > 0
        # Decomposed predictions: rate-factor stats >= 0 (volume clip floors).
        assert (residuals.mu_decomposed >= 0).all()


def test_walk_forward_residuals_tracks_two_coverage_axes() -> None:
    """Coverage tracked separately for the carries and targets axes."""
    features, weekly_stats = _synthetic_rb_inputs()
    output = walk_forward_residuals(features, weekly_stats, eval_years=(2020, 2021))

    assert set(output.coverage_carries_by_year.keys()) == {2020, 2021}
    assert set(output.coverage_targets_by_year.keys()) == {2020, 2021}
    for year in {2020, 2021}:
        assert 0.0 <= output.coverage_carries_by_year[year] <= 1.0
        assert 0.0 <= output.coverage_targets_by_year[year] <= 1.0


def test_walk_forward_residuals_arms_differ_on_some_rows() -> None:
    """The decomposed and direct arms must NOT produce identical predictions
    everywhere — a no-op fall-through would be a silent bug.
    """
    features, weekly_stats = _synthetic_rb_inputs(seed=12345)
    output = walk_forward_residuals(features, weekly_stats, eval_years=(2021,))

    for stat, residuals in output.per_stat.items():
        n_different = int(
            np.sum(np.abs(residuals.mu_direct - residuals.mu_decomposed) > 1e-6)
        )
        assert n_different > 0, (
            f"direct and decomposed arms produced identical predictions for {stat.value}"
        )


def test_walk_forward_residuals_skips_eval_year_with_no_train_data() -> None:
    """When eval_year is the earliest season, no train data exists — skip cleanly."""
    features, weekly_stats = _synthetic_rb_inputs()
    output = walk_forward_residuals(features, weekly_stats, eval_years=(2018,))

    # 2018 is the earliest season; train_seasons is empty -> coverage dict empty.
    assert output.coverage_carries_by_year == {}
    assert output.coverage_targets_by_year == {}
    for residuals in output.per_stat.values():
        assert residuals.n_paired == 0


def test_walk_forward_residuals_emits_factor_residuals_per_stat_per_year() -> None:
    """Per-stat per-year (volume_residual, efficiency_residual) for orthogonality check."""
    features, weekly_stats = _synthetic_rb_inputs()
    output = walk_forward_residuals(features, weekly_stats, eval_years=(2020, 2021))

    for stat in _RB_DECOMPS:
        assert stat in output.factor_residuals_by_year
        per_year_list = output.factor_residuals_by_year[stat]
        assert len(per_year_list) == 2  # one entry per eval year
        for entry in per_year_list:
            assert isinstance(entry, FactorResidualsByYear)
            assert entry.eval_year in {2020, 2021}
            assert entry.volume_residuals.shape == entry.efficiency_residuals.shape
```

- [ ] **Step 2.2: Run tests — expect import errors**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_rb_decomposition_probe.py -v
```

Expected: import errors on `StatResiduals`, `FactorResidualsByYear`, `CoverageByYear`, `WalkForwardOutput`, `walk_forward_residuals`.

- [ ] **Step 2.3: Extend the probe module with dataclasses + walk-forward harness**

Append to `src/projections/backtest/rb_decomposition_probe.py`:

```python
from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

from projections.models.baseline import _RB_FEATURE_COLUMNS
from projections.schemas import Position, WeeklyStatsSchema


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

    Used for the spec section 5 risk #2 orthogonality diagnostic. Volume residual
    is actual_volume - predicted_volume; efficiency residual is
    actual_efficiency_ratio - predicted_efficiency on rows with volume > 0.
    """

    eval_year: int
    volume_residuals: np.ndarray
    efficiency_residuals: np.ndarray


@dataclass(frozen=True, slots=True)
class WalkForwardOutput:
    """Bundle of all walk-forward outputs.

    Attributes:
        per_stat: {stat: StatResiduals} for the 5 entries in _RB_DECOMPS.
        factor_residuals_by_year: {stat: [FactorResidualsByYear per eval year]}.
        coverage_carries_by_year: per-eval-year carries > 0 rate (eval rows).
        coverage_targets_by_year: per-eval-year targets > 0 rate (eval rows).
        eval_years: the eval years actually computed (skipped years with no
            training data are absent from the coverage dicts).
    """

    per_stat: Mapping[Stat, StatResiduals]
    factor_residuals_by_year: Mapping[Stat, Sequence[FactorResidualsByYear]]
    coverage_carries_by_year: dict[int, float]
    coverage_targets_by_year: dict[int, float]
    eval_years: tuple[int, ...]


def walk_forward_residuals(
    features: pd.DataFrame,
    weekly_stats: pd.DataFrame,
    eval_years: Iterable[int],
) -> WalkForwardOutput:
    """For each eval year, train both arms on prior seasons, predict on the
    eval year, collect per-row residuals + per-volume-axis coverage stats.

    Two shared volume sub-models per training window (carries, targets);
    five efficiency sub-models on their respective volume > 0 subsets;
    five direct comparator sub-models on the full train rows. Eval predictions
    use the same X matrix for both arms.

    Spec: probe-design section 3.3.
    """
    eval_years_list = sorted(int(y) for y in eval_years)
    features_validated = features  # caller is responsible for schema validation
    ws = WeeklyStatsSchema.validate(weekly_stats)
    ws_rb = ws[ws["position"] == Position.RB.value].copy()

    per_stat_buffers: dict[Stat, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        stat: [] for stat in _RB_DECOMPS
    }
    factor_residuals_by_year: dict[Stat, list[FactorResidualsByYear]] = {
        stat: [] for stat in _RB_DECOMPS
    }
    coverage_carries_by_year: dict[int, float] = {}
    coverage_targets_by_year: dict[int, float] = {}

    all_seasons = sorted(int(s) for s in features_validated["season"].unique())
    feat_cols = list(_RB_FEATURE_COLUMNS)
    # The seven stat columns the harness joins from weekly_stats.
    stat_cols = [
        Stat.CARRIES.value,
        Stat.TARGETS.value,
        Stat.RUSHING_YARDS.value,
        Stat.RUSHING_TDS.value,
        Stat.RECEPTIONS.value,
        Stat.RECEIVING_YARDS.value,
        Stat.RECEIVING_TDS.value,
    ]

    def _join_and_filter(seasons: list[int]) -> pd.DataFrame | None:
        """Inner-join features <-> weekly_stats, drop NaN feature rows."""
        if not seasons:
            return None
        feat_slice = features_validated[features_validated["season"].isin(seasons)]
        ws_slice = ws_rb[ws_rb["season"].isin(seasons)]
        joined = feat_slice.merge(
            ws_slice[["gsis_id", "season", "week", *stat_cols]],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        if joined.empty:
            return None
        keep_mask = joined[feat_cols].notna().all(axis=1)
        joined = joined.loc[keep_mask]
        return joined if not joined.empty else None

    for eval_year in eval_years_list:
        train_seasons = [s for s in all_seasons if s < eval_year]
        train_join = _join_and_filter(train_seasons)
        eval_join = _join_and_filter([eval_year])
        if train_join is None or eval_join is None:
            continue

        # Boolean -> int8 coercion (matches BaselineModel.fit recipe).
        train_x_frame = train_join[feat_cols].copy()
        eval_x_frame = eval_join[feat_cols].copy()
        for col in train_x_frame.columns:
            if train_x_frame[col].dtype == bool:
                train_x_frame[col] = train_x_frame[col].astype(np.int8)
                eval_x_frame[col] = eval_x_frame[col].astype(np.int8)
        x_train = train_x_frame.to_numpy(dtype=np.float64)
        x_eval = eval_x_frame.to_numpy(dtype=np.float64)

        carries_train = train_join[Stat.CARRIES.value].to_numpy(dtype=np.int64)
        targets_train = train_join[Stat.TARGETS.value].to_numpy(dtype=np.int64)
        carries_eval = eval_join[Stat.CARRIES.value].to_numpy(dtype=np.int64)
        targets_eval = eval_join[Stat.TARGETS.value].to_numpy(dtype=np.int64)

        # Two shared volume sub-models.
        volume_carries = _fit_decomposed_volume(x_train, carries_train)
        volume_targets = _fit_decomposed_volume(x_train, targets_train)
        volume_ridges: dict[Stat, RidgeCV] = {
            Stat.CARRIES: volume_carries,
            Stat.TARGETS: volume_targets,
        }

        # Per-stat: fit direct + efficiency, predict on eval rows.
        for stat, decomp in _RB_DECOMPS.items():
            numerator_train = train_join[stat.value].to_numpy(dtype=np.float64)
            volume_train = train_join[decomp.volume_stat.value].to_numpy(
                dtype=np.int64
            )

            direct_ridge = _fit_direct(x_train, numerator_train)
            efficiency_ridge = _fit_decomposed_efficiency(
                x_train, numerator_train.astype(np.int64), volume_train
            )

            mu_direct = _predict_direct(direct_ridge, x_eval)
            mu_decomposed = _predict_decomposed(
                volume_ridge=volume_ridges[decomp.volume_stat],
                efficiency_ridge=efficiency_ridge,
                x=x_eval,
                efficiency_clip_hi=decomp.efficiency_clip_hi,
            )
            actual_eval = eval_join[stat.value].to_numpy(dtype=np.float64)

            per_stat_buffers[stat].append((actual_eval, mu_direct, mu_decomposed))

            # Factor residuals on eval rows where the relevant volume > 0.
            volume_eval = (
                carries_eval if decomp.volume_stat is Stat.CARRIES else targets_eval
            )
            mask_pos = volume_eval > 0
            if mask_pos.any():
                volume_pred: np.ndarray = volume_ridges[decomp.volume_stat].predict(
                    x_eval[mask_pos]
                ).astype(np.float64)
                volume_resid = (
                    volume_eval[mask_pos].astype(np.float64) - volume_pred
                )
                actual_ratio = actual_eval[mask_pos] / volume_eval[mask_pos].astype(
                    np.float64
                )
                eff_pred: np.ndarray = efficiency_ridge.predict(
                    x_eval[mask_pos]
                ).astype(np.float64)
                predicted_ratio = np.clip(eff_pred, 0.0, decomp.efficiency_clip_hi)
                efficiency_resid = actual_ratio - predicted_ratio
            else:
                volume_resid = np.array([], dtype=np.float64)
                efficiency_resid = np.array([], dtype=np.float64)

            factor_residuals_by_year[stat].append(
                FactorResidualsByYear(
                    eval_year=eval_year,
                    volume_residuals=volume_resid,
                    efficiency_residuals=efficiency_resid,
                )
            )

        # Coverage on eval rows.
        coverage_carries_by_year[eval_year] = (
            float((carries_eval > 0).mean()) if carries_eval.size > 0 else 0.0
        )
        coverage_targets_by_year[eval_year] = (
            float((targets_eval > 0).mean()) if targets_eval.size > 0 else 0.0
        )

    per_stat_residuals: dict[Stat, StatResiduals] = {}
    for stat, buffers in per_stat_buffers.items():
        if not buffers:
            per_stat_residuals[stat] = StatResiduals(
                actual=np.array([], dtype=np.float64),
                mu_direct=np.array([], dtype=np.float64),
                mu_decomposed=np.array([], dtype=np.float64),
                n_paired=0,
            )
            continue
        per_stat_residuals[stat] = StatResiduals(
            actual=np.concatenate([b[0] for b in buffers]),
            mu_direct=np.concatenate([b[1] for b in buffers]),
            mu_decomposed=np.concatenate([b[2] for b in buffers]),
            n_paired=int(sum(len(b[0]) for b in buffers)),
        )

    return WalkForwardOutput(
        per_stat=per_stat_residuals,
        factor_residuals_by_year=factor_residuals_by_year,
        coverage_carries_by_year=coverage_carries_by_year,
        coverage_targets_by_year=coverage_targets_by_year,
        eval_years=tuple(eval_years_list),
    )
```

- [ ] **Step 2.4: Run tests — expect PASS**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_rb_decomposition_probe.py -v
```

Expected: 13 tests pass (8 Task-1 + 5 Task-2).

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
  src/projections/backtest/rb_decomposition_probe.py \
  tests/test_backtest/test_rb_decomposition_probe.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(probe): walk_forward_residuals + dataclasses for RB probe

walk_forward_residuals fits both arms on each train window across two shared
volume axes (carries, targets) and 5 composed stats. Emits pooled per-row
buffers (per stat), per-year per-stat factor residuals (for orthogonality
check), and per-volume-axis per-year coverage stats. Mirrors PR #32's
WR-target probe but generalized to 5 stats x 2 volume axes.

StatResiduals / FactorResidualsByYear / WalkForwardOutput dataclasses are
frozen + slotted. Empty-train and empty-eval-join cases skip cleanly without
raising; affected coverage dict entries are absent.

Spec: docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md (Task 2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Verdict mapping + per-stat verdict helper

**Files:**
- Modify: `src/projections/backtest/rb_decomposition_probe.py` (extend)
- Modify: `tests/test_backtest/test_rb_decomposition_probe.py` (extend)

Scope: `VerdictLabel = Literal[...]`, `PerStatVerdict` dataclass, `compute_verdicts(output)` that returns a list of 5 PerStatVerdict (one per `_RB_DECOMPS` entry). Verdict mapping mirrors PR #32 / PR #44 exactly.

- [ ] **Step 3.1: Append failing tests**

Append to `tests/test_backtest/test_rb_decomposition_probe.py`:

```python
from projections.backtest.adoption_gate import BootstrapDelta

from projections.backtest.rb_decomposition_probe import (
    PerStatVerdict,
    compute_verdicts,
)


def _make_output_with_residuals(
    *,
    per_stat_residuals: dict[Stat, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> WalkForwardOutput:
    """Build a WalkForwardOutput from raw per-stat (actual, mu_direct, mu_decomp) arrays."""
    return WalkForwardOutput(
        per_stat={
            stat: StatResiduals(
                actual=a, mu_direct=md, mu_decomposed=mdc, n_paired=a.shape[0]
            )
            for stat, (a, md, mdc) in per_stat_residuals.items()
        },
        factor_residuals_by_year={stat: [] for stat in per_stat_residuals},
        coverage_carries_by_year={2024: 1.0},
        coverage_targets_by_year={2024: 1.0},
        eval_years=(2024,),
    )


def test_compute_verdicts_signal_when_decomposed_strictly_better() -> None:
    """SIGNAL iff hi_95 < 0."""
    rng = np.random.default_rng(seed=2041)
    n = 500
    actual = rng.uniform(0.0, 100.0, size=n)
    mu_direct = actual + 5.0 + rng.normal(0, 1.0, size=n)  # 5-unit bias
    mu_decomp = actual + rng.normal(0, 1.0, size=n)  # unbiased

    output = _make_output_with_residuals(
        per_stat_residuals={Stat.RUSHING_YARDS: (actual, mu_direct, mu_decomp)}
    )
    verdicts = compute_verdicts(output, n_bootstrap=200, seed=42)

    assert len(verdicts) == 1
    v = verdicts[0]
    assert isinstance(v, PerStatVerdict)
    assert v.stat is Stat.RUSHING_YARDS
    assert v.verdict == "SIGNAL"
    assert v.rmse_delta.hi_95 < 0


def test_compute_verdicts_null_when_ci_brackets_zero() -> None:
    """NULL when both arms produce equivalent residuals."""
    rng = np.random.default_rng(seed=2042)
    n = 300
    actual = rng.uniform(0.0, 100.0, size=n)
    mu_direct = actual + rng.normal(0, 5.0, size=n)
    mu_decomp = actual + rng.normal(0, 5.0, size=n)

    output = _make_output_with_residuals(
        per_stat_residuals={Stat.RECEIVING_YARDS: (actual, mu_direct, mu_decomp)}
    )
    verdicts = compute_verdicts(output, n_bootstrap=200, seed=42)

    v = verdicts[0]
    assert v.verdict == "NULL"
    assert v.rmse_delta.lo_95 < 0 < v.rmse_delta.hi_95


def test_compute_verdicts_regression_when_decomposed_strictly_worse() -> None:
    """REGRESSION iff lo_95 > 0."""
    rng = np.random.default_rng(seed=2043)
    n = 500
    actual = rng.uniform(0.0, 100.0, size=n)
    mu_direct = actual + rng.normal(0, 1.0, size=n)  # unbiased
    mu_decomp = actual + 5.0 + rng.normal(0, 1.0, size=n)  # 5-unit bias

    output = _make_output_with_residuals(
        per_stat_residuals={Stat.RUSHING_TDS: (actual, mu_direct, mu_decomp)}
    )
    verdicts = compute_verdicts(output, n_bootstrap=200, seed=42)

    v = verdicts[0]
    assert v.verdict == "REGRESSION"
    assert v.rmse_delta.lo_95 > 0


def test_compute_verdicts_returns_one_per_stat_in_output() -> None:
    """compute_verdicts returns len(per_stat) verdicts, one per stat."""
    features, weekly_stats = _synthetic_rb_inputs()
    output = walk_forward_residuals(features, weekly_stats, eval_years=(2020, 2021))

    verdicts = compute_verdicts(output, n_bootstrap=200, seed=42)

    assert len(verdicts) == 5
    assert {v.stat for v in verdicts} == set(_RB_DECOMPS.keys())
    for v in verdicts:
        assert v.verdict in {"SIGNAL", "NULL", "REGRESSION"}
        assert isinstance(v.rmse_delta, BootstrapDelta)
```

- [ ] **Step 3.2: Run tests — expect import errors**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_rb_decomposition_probe.py -v
```

Expected: import errors on `PerStatVerdict`, `compute_verdicts`.

- [ ] **Step 3.3: Extend module with verdict mapping**

Append to `src/projections/backtest/rb_decomposition_probe.py`:

```python
from typing import Literal

from projections.backtest.adoption_gate import (
    BootstrapDelta,
    paired_bootstrap_rmse_delta,
)


VerdictLabel = Literal["SIGNAL", "NULL", "REGRESSION"]


@dataclass(frozen=True, slots=True)
class PerStatVerdict:
    """Per-stat verdict on the per-stat Delta-RMSE (decomposed - direct)."""

    stat: Stat
    n_paired: int
    rmse_delta: BootstrapDelta
    verdict: VerdictLabel


def _verdict_from_delta(delta: BootstrapDelta) -> VerdictLabel:
    """SIGNAL iff hi_95 < 0; REGRESSION iff lo_95 > 0; else NULL."""
    if delta.hi_95 < 0:
        return "SIGNAL"
    if delta.lo_95 > 0:
        return "REGRESSION"
    return "NULL"


def compute_verdicts(
    output: WalkForwardOutput, *, n_bootstrap: int = 1000, seed: int = 42
) -> list[PerStatVerdict]:
    """Pooled paired-bootstrap CI per stat on (decomposed - direct) Delta-RMSE.

    Signed residuals are (actual - pred); paired_bootstrap_rmse_delta(inc, cand)
    returns RMSE(cand) - RMSE(inc), so direct=inc and decomposed=cand gives the
    convention (decomposed - direct).
    """
    verdicts: list[PerStatVerdict] = []
    for stat in _RB_DECOMPS:
        residuals = output.per_stat[stat]
        if residuals.n_paired == 0:
            # Empty residuals -> a sentinel "NULL" with zero CI.
            verdicts.append(
                PerStatVerdict(
                    stat=stat,
                    n_paired=0,
                    rmse_delta=BootstrapDelta(
                        point=0.0,
                        lo_95=0.0,
                        hi_95=0.0,
                        n_paired_rows=0,
                        n_bootstrap=0,
                    ),
                    verdict="NULL",
                )
            )
            continue
        inc_residuals = residuals.actual - residuals.mu_direct
        cand_residuals = residuals.actual - residuals.mu_decomposed
        rmse_delta = paired_bootstrap_rmse_delta(
            inc_residuals,
            cand_residuals,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        verdicts.append(
            PerStatVerdict(
                stat=stat,
                n_paired=residuals.n_paired,
                rmse_delta=rmse_delta,
                verdict=_verdict_from_delta(rmse_delta),
            )
        )
    return verdicts
```

- [ ] **Step 3.4: Run tests — expect PASS**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_rb_decomposition_probe.py -v
```

Expected: 17 tests pass (8 Task-1 + 5 Task-2 + 4 Task-3).

- [ ] **Step 3.5: Lint + typecheck**

```bash
../../.venv/Scripts/python.exe -m ruff check src tests
../../.venv/Scripts/python.exe -m ruff format --check src tests
../../.venv/Scripts/python.exe -m mypy src tests
```

Expected: zero violations.

- [ ] **Step 3.6: Commit**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  src/projections/backtest/rb_decomposition_probe.py \
  tests/test_backtest/test_rb_decomposition_probe.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(probe): compute_verdicts + PerStatVerdict for RB probe

compute_verdicts runs paired-bootstrap CI on (decomposed - direct) residuals
for each of the 5 stats and maps to SIGNAL / NULL / REGRESSION per spec
section 1.3 verdict rule. Empty-residuals case (early eval year with no
training data) returns a sentinel NULL verdict with zero CI to keep the
output shape stable.

Spec: docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md (Task 3).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: CLI driver + report writers + CLI smoke

**Files:**
- Create: `scripts/probe_rb_decomposition.py`
- Create: `tests/test_scripts/test_probe_rb_decomposition_cli.py`

Scope: argparse driver wiring `walk_forward_residuals` + `compute_verdicts` + summary report + per-stat CSV. Mocked CLI smoke test.

- [ ] **Step 4.1: Write failing CLI test**

Create `tests/test_scripts/test_probe_rb_decomposition_cli.py`:

```python
"""CLI smoke for scripts/probe_rb_decomposition.py.

Mocks walk_forward_residuals + compute_verdicts to avoid real data; verifies
argparse + report writing.
"""

from __future__ import annotations

import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

import scripts.probe_rb_decomposition as probe_cli
from projections.backtest.adoption_gate import BootstrapDelta
from projections.backtest.rb_decomposition_probe import (
    PerStatVerdict,
    StatResiduals,
    WalkForwardOutput,
)
from projections.schemas import Stat


def _fake_output() -> WalkForwardOutput:
    actual = np.array([55.0, 30.0, 72.0], dtype=np.float64)
    md = np.array([55.1, 31.2, 70.8], dtype=np.float64)
    mdc = np.array([54.9, 30.5, 71.4], dtype=np.float64)
    per_stat = {
        s: StatResiduals(actual=actual, mu_direct=md, mu_decomposed=mdc, n_paired=3)
        for s in (
            Stat.RUSHING_YARDS,
            Stat.RUSHING_TDS,
            Stat.RECEPTIONS,
            Stat.RECEIVING_YARDS,
            Stat.RECEIVING_TDS,
        )
    }
    return WalkForwardOutput(
        per_stat=per_stat,
        factor_residuals_by_year={s: [] for s in per_stat},
        coverage_carries_by_year={2021: 0.92, 2022: 0.94},
        coverage_targets_by_year={2021: 0.88, 2022: 0.90},
        eval_years=(2021, 2022),
    )


def _fake_verdicts() -> list[PerStatVerdict]:
    return [
        PerStatVerdict(
            stat=stat,
            n_paired=3,
            rmse_delta=BootstrapDelta(
                point=-0.5, lo_95=-0.8, hi_95=-0.1, n_paired_rows=3, n_bootstrap=1000
            ),
            verdict="SIGNAL",
        )
        for stat in (
            Stat.RUSHING_YARDS,
            Stat.RUSHING_TDS,
            Stat.RECEPTIONS,
            Stat.RECEIVING_YARDS,
            Stat.RECEIVING_TDS,
        )
    ]


def test_cli_writes_summary_and_csv(tmp_path: Path) -> None:
    """CLI invocation produces both the summary .md and the .csv report."""
    summary_path = tmp_path / "summary.md"
    csv_path = tmp_path / "per_stat.csv"

    with (
        mock.patch.object(
            probe_cli,
            "_load_inputs",
            return_value=(mock.MagicMock(), mock.MagicMock()),
        ),
        mock.patch.object(probe_cli, "walk_forward_residuals", return_value=_fake_output()),
        mock.patch.object(probe_cli, "compute_verdicts", return_value=_fake_verdicts()),
        mock.patch.object(
            sys,
            "argv",
            [
                "probe_rb_decomposition",
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
    # All 5 stats appear, all SIGNAL verdicts.
    for stat_value in ("rushing_yards", "rushing_tds", "receptions", "receiving_yards", "receiving_tds"):
        assert stat_value in summary
    assert summary.count("SIGNAL") >= 5
    # Coverage section surfaces below-threshold flag (0.88 < 0.95).
    assert "BELOW THRESHOLD" in summary

    assert csv_path.exists()
    csv = csv_path.read_text(encoding="utf-8")
    for stat_value in ("rushing_yards", "rushing_tds", "receptions", "receiving_yards", "receiving_tds"):
        assert stat_value in csv


def test_cli_rejects_unknown_year(tmp_path: Path) -> None:
    """argparse should choke on an out-of-range --eval-years value."""
    with mock.patch.object(
        sys,
        "argv",
        [
            "probe_rb_decomposition",
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
../../.venv/Scripts/python.exe -m pytest tests/test_scripts/test_probe_rb_decomposition_cli.py -v
```

Expected: import error on `scripts.probe_rb_decomposition`.

- [ ] **Step 4.3: Implement the CLI driver**

Create `scripts/probe_rb_decomposition.py`:

```python
"""CLI driver for the RB rushing + receiving decomposition probe.

Spec: docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md.

Reads RB features + weekly_stats from disk, runs walk_forward_residuals on
2021-2024 by default, computes per-stat verdicts, writes a summary markdown
+ per-stat CSV.

Usage:
    python scripts/probe_rb_decomposition.py \\
        --summary-out reports/feature_probe_rb_decomposition_summary.md \\
        --csv-out reports/feature_probe_rb_decomposition_per_stat.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from projections.backtest.rb_decomposition_probe import (
    PerStatVerdict,
    WalkForwardOutput,
    compute_verdicts,
    walk_forward_residuals,
)
from projections.features.cache import read_features
from projections.schemas import Position, Ruleset
from projections.store import read_partition

_DEFAULT_EVAL_YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024)
_VALID_YEARS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
_COVERAGE_THRESHOLD: float = 0.95

# Marginal-zone threshold per PR #31's retrospective rule: |delta_fpts| < 0.005.
# ASCII text only in stdout/file output to avoid Windows cp1252 encoding crashes
# (spec section 5 risk #6 / PR #39 follow-up).
_MARGINAL_ZONE_FPTS: float = 0.005
_RULESET: Ruleset = Ruleset.espn_ppr()


def _load_inputs(
    *,
    eval_years: Sequence[int],
    features_root: Path,
    raw_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load RB features + weekly_stats for the seasons needed by the walk-forward."""
    seasons_needed = [y for y in _VALID_YEARS if y <= max(eval_years)]
    feat_parts = [
        read_features(Position.RB, s, features_root=features_root)
        for s in seasons_needed
    ]
    features = pd.concat(feat_parts, ignore_index=True)
    ws_parts = [
        read_partition(raw_root, "weekly_stats", season=s) for s in seasons_needed
    ]
    weekly_stats = pd.concat(ws_parts, ignore_index=True)
    return features, weekly_stats


def _stat_to_fpts(stat_value: str, delta_yards_or_count: float) -> float:
    """Translate per-stat delta to composite-fpts delta via Ruleset.espn_ppr()."""
    if stat_value == "rushing_yards":
        return delta_yards_or_count / _RULESET.rushing_yds_per_pt
    if stat_value == "receiving_yards":
        return delta_yards_or_count / _RULESET.receiving_yds_per_pt
    if stat_value == "rushing_tds":
        return delta_yards_or_count * _RULESET.rushing_td_pts
    if stat_value == "receiving_tds":
        return delta_yards_or_count * _RULESET.receiving_td_pts
    if stat_value == "receptions":
        return delta_yards_or_count * _RULESET.reception_pts
    raise ValueError(f"unknown stat value for fpts translation: {stat_value}")


def _write_summary(
    path: Path,
    *,
    verdicts: list[PerStatVerdict],
    output: WalkForwardOutput,
    coverage_threshold: float,
    args: argparse.Namespace,
) -> None:
    """Markdown summary report. ASCII-only stdout for Windows cp1252 safety."""
    lines: list[str] = [
        "# RB Rushing + Receiving Decomposition Probe -- Summary",
        "",
        "**Spec:** `docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md`",
        f"**Eval years:** {list(output.eval_years)}",
        f"**n_bootstrap:** {args.n_bootstrap}, seed: {args.seed}",
        "",
        "## Per-stat verdicts",
        "",
        "| Stat | n_paired | RMSE delta (decomp - direct) | 95% CI | Composite-fpts equiv | Magnitude flag | Verdict |",
        "|---|---:|---:|---|---:|---|:---:|",
    ]
    for v in verdicts:
        fpts_delta = _stat_to_fpts(v.stat.value, v.rmse_delta.point)
        mag_flag = (
            "MARGINAL" if abs(fpts_delta) < _MARGINAL_ZONE_FPTS else ""
        )
        lines.append(
            f"| {v.stat.value} | {v.n_paired} | "
            f"{v.rmse_delta.point:+.4f} | "
            f"[{v.rmse_delta.lo_95:+.4f}, {v.rmse_delta.hi_95:+.4f}] | "
            f"{fpts_delta:+.4f} | {mag_flag} | **{v.verdict}** |"
        )
    lines.append("")

    lines.append("## Coverage (eval rows)")
    lines.append("")
    lines.append(
        f"Coverage threshold: {coverage_threshold:.2f} per volume axis per eval year."
    )
    lines.append("")
    lines.append("### Carries > 0 rate (rushing axis)")
    for year in sorted(output.coverage_carries_by_year):
        rate = output.coverage_carries_by_year[year]
        flag = "" if rate >= coverage_threshold else " -- BELOW THRESHOLD"
        lines.append(f"- {year}: {rate:.4f}{flag}")
    lines.append("")
    lines.append("### Targets > 0 rate (receiving axis)")
    for year in sorted(output.coverage_targets_by_year):
        rate = output.coverage_targets_by_year[year]
        flag = "" if rate >= coverage_threshold else " -- BELOW THRESHOLD"
        lines.append(f"- {year}: {rate:.4f}{flag}")
    lines.append("")

    lines.append("## Mechanism caveat")
    lines.append("")
    lines.append(
        "This probe tests decomposition with RidgeCV everywhere (the same model "
        "class on both arms). Factor-appropriate sub-model classes (Poisson, "
        "Gamma / Tweedie, logistic) are separate probe + integration cycles per "
        "spec section 1.4 #3. PR #39 / PR #44 closed two of these on WR with NULL "
        "verdicts; RB-side factor-class probes remain independent tests if any "
        "stat here SIGNALs."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, verdicts: list[PerStatVerdict]) -> None:
    """Long-form CSV: one row per stat with delta + CI + composite-fpts equiv."""
    rows: list[dict[str, object]] = []
    for v in verdicts:
        rows.append(
            {
                "stat": v.stat.value,
                "n_paired": v.n_paired,
                "rmse_delta_point": v.rmse_delta.point,
                "rmse_delta_lo": v.rmse_delta.lo_95,
                "rmse_delta_hi": v.rmse_delta.hi_95,
                "composite_fpts_equivalent": _stat_to_fpts(
                    v.stat.value, v.rmse_delta.point
                ),
                "verdict": v.verdict,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RB rushing + receiving decomposition probe."
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
        default=Path("reports/feature_probe_rb_decomposition_summary.md"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("reports/feature_probe_rb_decomposition_per_stat.csv"),
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
    output = walk_forward_residuals(
        features, weekly_stats, eval_years=args.eval_years
    )
    verdicts = compute_verdicts(output, n_bootstrap=args.n_bootstrap, seed=args.seed)

    _write_summary(
        args.summary_out,
        verdicts=verdicts,
        output=output,
        coverage_threshold=args.coverage_threshold,
        args=args,
    )
    _write_csv(args.csv_out, verdicts)

    # ASCII-only stdout (Windows cp1252 guard per PR #39 follow-up).
    print("Per-stat verdicts:")
    for v in verdicts:
        print(
            f"  {v.stat.value:<18s} -> {v.verdict:<10s} "
            f"(delta {v.rmse_delta.point:+.4f}, "
            f"CI [{v.rmse_delta.lo_95:+.4f}, {v.rmse_delta.hi_95:+.4f}], "
            f"n_paired {v.n_paired})"
        )
    print(f"  Summary: {args.summary_out}")
    print(f"  CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.4: Run CLI tests — expect PASS**

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_scripts/test_probe_rb_decomposition_cli.py -v
```

Expected: 2 tests pass.

- [ ] **Step 4.5: Lint + typecheck**

```bash
../../.venv/Scripts/python.exe -m ruff check src tests scripts
../../.venv/Scripts/python.exe -m ruff format --check src tests scripts
../../.venv/Scripts/python.exe -m mypy src tests
```

Expected: zero violations. If mypy complains about `_stat_to_fpts`'s catch-all `raise ValueError`, narrow with `# type: ignore[unreachable]` or use a `match`-statement; but the canonical pattern is fine.

- [ ] **Step 4.6: Commit**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  scripts/probe_rb_decomposition.py \
  tests/test_scripts/test_probe_rb_decomposition_cli.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(scripts): probe_rb_decomposition CLI driver

argparse driver mirroring scripts/probe_target_decomposition.py: loads RB
features + weekly_stats, runs walk_forward_residuals + compute_verdicts
across 5 stats x 2 volume axes, writes summary markdown + per-stat CSV.

Composite-fpts equivalents derived from Ruleset.espn_ppr() per-stat
coefficients (no hardcoded multipliers per CLAUDE.md scoring-layer rule
from PR #44 review). Magnitude flag fires when |delta_fpts| < 0.005 per
PR #31's retrospective. ASCII-only stdout (Windows cp1252 guard).

Summary surfaces per-volume-axis coverage with BELOW THRESHOLD flag per
spec section 1.3 #1 (carries > 0 may dip below 0.95 due to pass-catching
backs; symmetric concern on targets > 0).

Spec: docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md (Task 4).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Real-data probe run + report + PM/TODO

**Files:**
- Create (CLI-written): `reports/feature_probe_rb_decomposition_summary.md`
- Create (CLI-written): `reports/feature_probe_rb_decomposition_per_stat.csv`
- Modify: `project_management.md`
- Modify: `TODO.md`

Scope: operational. Run the CLI against the real RB feature cache, inspect verdicts, write PM/TODO entries.

- [ ] **Step 5.1: Run the probe from main repo cwd**

The worktree lacks `data/raw` and `data/features`. Run from main repo cwd so relative paths resolve correctly; invoke the worktree's script directly:

```bash
cd /c/Users/alden/FantasyFootball
.venv/Scripts/python.exe /c/Users/alden/FantasyFootball/.worktrees/feat-probe-rb-decomposition/scripts/probe_rb_decomposition.py \
  --summary-out /c/Users/alden/FantasyFootball/.worktrees/feat-probe-rb-decomposition/reports/feature_probe_rb_decomposition_summary.md \
  --csv-out /c/Users/alden/FantasyFootball/.worktrees/feat-probe-rb-decomposition/reports/feature_probe_rb_decomposition_per_stat.csv
```

Expected runtime: 30s-2min (5 stats x 2 arms x 4 years, plus bootstrap). On real RB data 2021-2024.

Sanity check the verdict output: per-stat means should land in plausible NFL ranges (mean RB rushing_yards ~ 50-70 yards per game-week, receptions ~ 2-4, etc.). If any stat returns wildly out-of-range predictions, investigate before continuing.

- [ ] **Step 5.2: Read the summary + verdicts**

```bash
cat /c/Users/alden/FantasyFootball/.worktrees/feat-probe-rb-decomposition/reports/feature_probe_rb_decomposition_summary.md
```

Map each per-stat verdict to spec section 4 decision branches:
- **All 5 NULL**: close RB decomposition at 2-factor unit; refined decompositions remain on TODO but require independent mechanism evidence.
- **Rushing SIGNAL**: integration plan for the SIGNAL stat(s) (add to `_RB_FACTORIES["decomposed-baseline"]` `decomposed_stats` mapping). Composite-fpts adoption gate vs current production.
- **RB receiving SIGNAL** (especially receptions): generalization finding from WR's PR #36. Integration plan opts in to the SIGNAL stats only.
- **Mixed**: tighter follow-up probe documented per stat.

Note coverage flags if either axis dips below 0.95 — PR #31 retrospective MARGINAL rule applies.

- [ ] **Step 5.3: Update `project_management.md`**

Add a top-of-file decision-log entry (after the `---` divider following the intro). Read recent entries for tone; the Tweedie probe entry (PR #44) is the closest precedent.

Template (fill bracketed values from Step 5.2):

```markdown
## RB Rushing + Receiving Decomposition Probe -- verdicts `<5 verdicts>` (2026-05-16, on branch `feat/probe-rb-decomposition`)

**Status:** New probe `src/projections/backtest/rb_decomposition_probe.py` tests whether decomposing RB stats into two shared volume axes (carries, targets) x per-stat efficiency factors beats per-stat direct RidgeCV. 5 composed stats: rushing_yards, rushing_tds (carries axis) + receptions, receiving_yards, receiving_tds (targets axis). Sub-model = RidgeCV everywhere (decomposition-only test; factor-appropriate sub-models are separate cycles). Spec at `docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md`.

**Per-stat verdicts:**

| Stat | n_paired | RMSE delta | 95% CI | Composite-fpts equiv | Verdict |
|---|---:|---:|---|---:|:---:|
| rushing_yards | <N> | <+/-X.XXXX> | [<+/-X.XXXX>, <+/-X.XXXX>] | <+/-0.XXXX> fpts | <SIGNAL/NULL/REGRESSION> |
| rushing_tds | <N> | <+/-X.XXXX> | [<+/-X.XXXX>, <+/-X.XXXX>] | <+/-0.XXXX> fpts | <SIGNAL/NULL/REGRESSION> |
| receptions | <N> | <+/-X.XXXX> | [<+/-X.XXXX>, <+/-X.XXXX>] | <+/-0.XXXX> fpts | <SIGNAL/NULL/REGRESSION> |
| receiving_yards | <N> | <+/-X.XXXX> | [<+/-X.XXXX>, <+/-X.XXXX>] | <+/-0.XXXX> fpts | <SIGNAL/NULL/REGRESSION> |
| receiving_tds | <N> | <+/-X.XXXX> | [<+/-X.XXXX>, <+/-X.XXXX>] | <+/-0.XXXX> fpts | <SIGNAL/NULL/REGRESSION> |

**Coverage:** carries > 0 rate <X.XXXX-X.XXXX> across 2021-2024; targets > 0 rate <X.XXXX-X.XXXX>. <Below-threshold-note-if-applicable>.

**Mechanism interpretation:** <one paragraph per outcome class.>

**Recommended next direction:** <per verdict cluster.>

See `reports/feature_probe_rb_decomposition_summary.md` for full per-stat tables + coverage flags + plan-vs-execution-deviations.
```

Fill bracketed values from the actual run.

- [ ] **Step 5.4: Update `TODO.md`**

Append an `**Update 2026-05-16 (RB decomposition probe, branch `feat/probe-rb-decomposition`)**:` line under the decomposition / factor-appropriate chain entry (TODO #23 or wherever the WR PR #32 / PR #36 / PR #38 / PR #39 / PR #44 cluster lives). Cite the 5 verdicts + recommended next direction. Close (`[x]`) any TODO entry that was waiting on this probe's outcome.

- [ ] **Step 5.5: Final verification**

```bash
cd /c/Users/alden/FantasyFootball/.worktrees/feat-probe-rb-decomposition
../../.venv/Scripts/python.exe -m pytest tests/test_backtest/test_rb_decomposition_probe.py tests/test_scripts/test_probe_rb_decomposition_cli.py -v
../../.venv/Scripts/python.exe -m mypy src tests
../../.venv/Scripts/python.exe -m ruff check src tests scripts
../../.venv/Scripts/python.exe -m ruff format --check src tests scripts
../../.venv/Scripts/python.exe -m pytest -v -k "ingest or store or schemas"
```

All green.

- [ ] **Step 5.6: Commit reports + PM/TODO**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  reports/feature_probe_rb_decomposition_summary.md \
  reports/feature_probe_rb_decomposition_per_stat.csv \
  project_management.md \
  TODO.md
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
data(probe): RB decomposition verdicts <5 verdicts> on 2021-2024 walk-forward

Pooled 4-year walk-forward on real RB data, 5 composed stats x 2 arms.
Per-stat verdicts: rushing_yards <V>, rushing_tds <V>, receptions <V>,
receiving_yards <V>, receiving_tds <V>. Coverage: carries > 0 rate
<X.XXXX-X.XXXX>, targets > 0 rate <X.XXXX-X.XXXX>.

<Optional below-threshold flag(s) per axis>

Mechanism: <one-line interpretation>

Recommended next direction: <per verdict cluster>.

Spec: docs/superpowers/specs/2026-05-16-rb-decomposition-probe-design.md (Task 5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Fill bracketed values from the actual run at commit time.)

- [ ] **Step 5.7: DO NOT push or open PR.** Wait for user authorization per project convention (PR creation is a visible/shared action).

---

## Self-review checks (post-plan)

- **Spec coverage:**
  - §1.1 (5-stat / 2-axis goal) — Task 1 registry + Task 2 walk-forward.
  - §1.3 (success criteria) — Tasks 2-5 cover coverage tracking, verdict mapping, gate verification.
  - §1.4 (out of scope) — explicitly skipped (no new factory, no schema, no codec, no production routing).
  - §2 (source data) — Task 4's `_load_inputs` + Task 2's `_join_and_filter`.
  - §3.1 (module architecture) — Tasks 1-3.
  - §3.2 (sub-model fitting) — Task 1.
  - §3.3 (walk-forward harness) — Task 2.
  - §3.4 (verdict + bootstrap) — Task 3.
  - §3.5 (CLI) — Task 4.
  - §3.6 (no edits to existing) — preserved (only new files + Task 5 PM/TODO updates).
  - §3.7 (no schema/codec/factory) — preserved.
  - §5 (risk register) — all 6 risks have mitigations in the implementation (coverage tracking, orthogonality residuals, clipping, raise-on-no-positive-volume guard, ASCII stdout, composite-fpts conversion via Ruleset).
  - §6 (reports) — Task 4 summary + CSV writers.
  - §7 (estimated scope) — matches 5 tasks.

- **Placeholder scan:** Step 5.3 / 5.4 / 5.6 use bracketed `<5 verdicts>` / `<+/-X.XXXX>` / `<N>` — intentional and documented because they're only known after Step 5.1. No other placeholders in code blocks.

- **Type consistency:**
  - `_StatDecomp` defined in Task 1; consumed by Task 2 (`_RB_DECOMPS`) + Task 3 (verdict iteration).
  - `StatResiduals`, `FactorResidualsByYear`, `WalkForwardOutput` defined in Task 2; consumed by Task 3 (compute_verdicts) + Task 4 (CLI summary writer).
  - `PerStatVerdict` defined in Task 3; consumed by Task 4 (CLI).
  - All function signatures consistent across tasks.
  - `Stat.CARRIES.value` and `Stat.TARGETS.value` referenced consistently (verified against schemas.py lines 170-173).

- **Scope boundaries:** Each task touches ≤ 5 files per CLAUDE.md "phased execution" rule. Tasks 1-3 touch 2 files each (source + test); Task 4 creates 2 new files; Task 5 touches 4 (2 reports + PM + TODO).

- **Scoring-layer rule (CLAUDE.md):** `_stat_to_fpts` in Task 4 derives all coefficients from `Ruleset.espn_ppr()` — no hardcoded multipliers (review fix from PR #44).

- **ASCII-only stdout:** all `print()` lines in Task 4's CLI use plain text; no Greek letters or em-dashes (Windows cp1252 guard per PR #39 follow-up).
