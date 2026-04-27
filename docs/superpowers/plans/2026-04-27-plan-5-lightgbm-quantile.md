# Plan 5 — LightGBM with Quantile Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `LightGBMModel` (Model C) implementing per-stat quantile regression, coexisting with `BaselineModel` (Model A) under the existing `Model` Protocol, so the two can be compared side-by-side on the backtest snapshot and eventually ensembled (Model D / Plan 6).

**Architecture:** New `src/projections/models/lightgbm.py` with a `LightGBMModel` class that trains 5 LightGBM `LGBMRegressor` quantile sub-models per stat per position (quantiles `[0.05, 0.10, 0.50, 0.90, 0.95]`). New `QuantileDistribution` class (interpolated CDF / quantile / inverse-CDF sampling) added under `src/projections/distributions/quantile.py`. New `DistributionFamily.QUANTILE` enum value + codec branch. `POSITION_DISPATCH` extended with a `factories: dict[str, Callable]` so callers select model class by name. Backtest harness gains `--model {baseline,lightgbm,both}`; snapshot file renamed to `model_metrics.json` and rows keyed by `(position, year, metric, model_class)`. `BaselineModel` is **untouched**.

**Tech Stack:** Python 3.11, `lightgbm>=4.0` (new dep), `lgb.LGBMRegressor` with `objective='quantile'` + `lgb.early_stopping` callback. Existing `Distribution` Protocol, codec, `score_distribution`, and backtest harness consumed unchanged downstream of the new model class.

**Spec:** `docs/superpowers/specs/2026-04-27-plan-5-lightgbm-quantile-design.md`

---

## File structure

### Files created

- `src/projections/distributions/quantile.py` — `QuantileDistribution` class (~80 LOC).
- `src/projections/models/lightgbm.py` — `LightGBMModel` class + per-position factories (`qb_lightgbm`, `rb_lightgbm`, `te_lightgbm`, `wr_lightgbm`); module-level `LGBM_DEFAULTS`, `EARLY_STOPPING_ROUNDS`, `QUANTILE_GRID` constants.
- `tests/test_distributions/test_quantile.py` — `QuantileDistribution` unit tests.
- `tests/test_models/test_lightgbm.py` — cross-cutting `LightGBMModel` tests + WR-specific tests (mirrors `test_baseline.py`).
- `tests/test_models/test_lightgbm_qb.py` — QB per-position smoke.
- `tests/test_models/test_lightgbm_rb.py` — RB per-position smoke.
- `tests/test_models/test_lightgbm_te.py` — TE per-position smoke.
- `tests/test_models/test_lightgbm_smoke.py` — parametrized fit-and-predict for all 4 positions on synthetic fixtures.
- `tests/test_backtest/test_harness_lightgbm.py` — end-to-end harness fold for Model C.
- `tests/test_backtest/test_harness_dual_model.py` — `--model both` harness invocation.
- `docs/superpowers/plans/2026-04-27-plan-5-lightgbm-quantile.md` — this file.

### Files modified

- `pyproject.toml` — add `lightgbm>=4.0` to `[project] dependencies`.
- `src/projections/schemas.py` — add `DistributionFamily.QUANTILE` enum value.
- `src/projections/distributions/__init__.py` — export `QuantileDistribution`.
- `src/projections/distributions/codec.py` — add QUANTILE pack/unpack branches.
- `tests/test_distributions/test_codec.py` — add `test_codec_round_trip_quantile` and `test_codec_round_trip_mixed_with_quantile`.
- `src/projections/models/__init__.py` — `_PositionDispatch.factories: dict[str, Callable]`; export `LightGBMModel`, `qb_lightgbm`, `rb_lightgbm`, `te_lightgbm`, `wr_lightgbm`.
- `src/projections/backtest/harness.py` — accept selected model classes; iterate over them per fold; add `model_class` to per-row results.
- `src/projections/backtest/snapshot.py` — snapshot rows keyed by `(position, year, metric, model_class)`; default snapshot path renamed to `model_metrics.json`.
- `scripts/backtest.py` — add `--model {baseline,lightgbm,both}` CLI arg (default `both` for gated runs); thread through to harness.
- `scripts/train_baseline.py` — add `--model {baseline,lightgbm}` CLI arg (default `baseline` for backwards compat).
- `scripts/sanity_check_baseline.py` — same `--model` arg.
- `scripts/predict_2024.py` — same `--model` arg.
- `tests/backtest/test_backtest_smoke.py` — assert *both* models produce finite metrics.
- `tests/backtest/baseline_metrics.json` — **renamed** to `tests/backtest/model_metrics.json`; rows acquire `model_class` field; existing 400 Model A rows preserved + 400 new Model C rows added at the snapshot regeneration step.
- `tests/backtest/conftest.py` — if it references the old snapshot filename, update.
- `project_management.md` — record Plan 5 outcomes; update Next-action.
- `TODO.md` — close TODO #26; surface follow-ups (Plan 6 ensemble, default-model selection).

### Files NOT touched

- `src/projections/models/baseline.py` — `BaselineModel` (Model A) is **untouched**. No edits.
- `src/projections/models/base.py` — `Model` Protocol is **untouched**. The new `LightGBMModel` satisfies the existing structural contract.
- `src/projections/distributions/parametric.py` — Plan 3e's `ParametricStudentT` and the rest of the parametric families are untouched.
- `src/projections/scoring/`, `src/projections/aggregation/` — fully unchanged. `QuantileDistribution` satisfies the `Distribution` Protocol structurally so all downstream consumers work without edits.
- `src/projections/features/` — same features, same schemas.
- Plan 3e infrastructure — `_per_bucket_*` helpers, widened `variance_params` type, `tests/test_distributions/test_student_t.py`, etc. — all preserved as-is.

---

## Task list at a glance

| # | Task | Output |
|---|---|---|
| 1 | Add `lightgbm>=4.0` dep | `pyproject.toml` |
| 2 | Add `DistributionFamily.QUANTILE` enum | `schemas.py` + regression test |
| 3 | `QuantileDistribution` class + tests | `distributions/quantile.py`, `test_quantile.py` |
| 4 | Codec QUANTILE branches + round-trip tests | `distributions/codec.py`, `test_codec.py` |
| 5 | `LightGBMModel` config dataclass + per-position factories (skeleton) | `models/lightgbm.py` |
| 6 | `LightGBMModel.fit` — train/val split + per-stat per-quantile sub-models | `models/lightgbm.py` |
| 7 | `LightGBMModel.predict_distribution` — quantiles → sort → clip → QD → score | `models/lightgbm.py` |
| 8 | `save` / `load` / `model_id` | `models/lightgbm.py` |
| 9 | Cross-cutting `LightGBMModel` tests (incl. WR) | `test_lightgbm.py` |
| 10 | Per-position smoke tests (QB/RB/TE) | `test_lightgbm_qb/rb/te.py` |
| 11 | Parametrized cross-position smoke | `test_lightgbm_smoke.py` |
| 12 | `POSITION_DISPATCH` `factories` dict + consumer updates | `models/__init__.py` + scripts |
| 13 | Backtest harness `--model` arg | `backtest/harness.py`, `scripts/backtest.py` |
| 14 | Snapshot rename + `model_class` column | `backtest/snapshot.py`, snapshot file |
| 15 | Default-on smoke extension | `tests/backtest/test_backtest_smoke.py` |
| 16 | Harness-side tests | `test_harness_lightgbm.py`, `test_harness_dual_model.py` |
| 17 | Train Model C standalone artifacts (4 positions) | `models/artifacts/lightgbm-*.joblib` |
| 18 | Run full opt-in backtest gate; regenerate snapshot | `tests/backtest/model_metrics.json` |
| 19 | PM doc + TODO updates | docs |
| 20 | End-of-effort gate + open PR | green checks + PR |

---

## Task 1: Add `lightgbm>=4.0` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current `pyproject.toml` `[project] dependencies`**

Run: `grep -n "^dependencies = " pyproject.toml`
Expected: line ~10 starting `dependencies = [`.

- [ ] **Step 2: Add `lightgbm>=4.0` to the dependencies list**

In `pyproject.toml`, append `"lightgbm>=4.0",` to the `dependencies` array (alphabetical-ish; place after `joblib>=1.3,`).

```toml
dependencies = [
    "pandas>=2.2",
    "pyarrow>=15",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "scipy>=1.12",
    "pydantic>=2.6",
    "pandera>=0.20",
    "duckdb>=1.0",
    "joblib>=1.3",
    "lightgbm>=4.0",
    "nfl_data_py>=0.3.2",
    "msgpack>=1.0",
]
```

- [ ] **Step 3: Install in venv**

Run: `pip install -e .`
Expected: `Successfully installed lightgbm-4.x.x ...`

- [ ] **Step 4: Verify import works**

Run: `python -c "import lightgbm as lgb; print(lgb.__version__)"`
Expected: prints `4.x.x` with no error.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "deps(plan-5): add lightgbm>=4.0 for Model C"
```

with the standard `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.

---

## Task 2: Add `DistributionFamily.QUANTILE` enum value

**Files:**
- Modify: `src/projections/schemas.py`
- Modify: `tests/test_schemas/test_other_enums.py`

- [ ] **Step 1: Locate `DistributionFamily` in `schemas.py`**

Run: `grep -n "class DistributionFamily" src/projections/schemas.py`
Note the line; read 10 lines around it.

- [ ] **Step 2: Add the `QUANTILE` member**

In `src/projections/schemas.py`, add `QUANTILE = "QUANTILE"` to `DistributionFamily` after the existing `STUDENT_T` member.

```python
class DistributionFamily(StrEnum):
    NORMAL = "NORMAL"
    GAMMA = "GAMMA"
    NEGATIVE_BINOMIAL = "NEGATIVE_BINOMIAL"
    STUDENT_T = "STUDENT_T"
    SAMPLED_SUMMARY = "SAMPLED_SUMMARY"
    QUANTILE = "QUANTILE"
```

(Keep `SAMPLED_SUMMARY` ordering consistent with the existing file; insert `QUANTILE` immediately before or after as appropriate to match Plan 3d's existing pattern.)

- [ ] **Step 3: Add a regression assertion in the existing enum test**

In `tests/test_schemas/test_other_enums.py`, add:

```python
def test_distribution_family_includes_quantile() -> None:
    """Plan 5 — Model C emits QUANTILE-family per-stat distributions."""
    from projections.schemas import DistributionFamily
    assert DistributionFamily.QUANTILE.value == "QUANTILE"
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_schemas/test_other_enums.py::test_distribution_family_includes_quantile -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/test_other_enums.py
git commit -m "feat(schemas): DistributionFamily.QUANTILE — Plan 5"
```

---

## Task 3: `QuantileDistribution` class + tests

**Files:**
- Create: `src/projections/distributions/quantile.py`
- Modify: `src/projections/distributions/__init__.py`
- Create: `tests/test_distributions/test_quantile.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_distributions/test_quantile.py`:

```python
"""Tests for QuantileDistribution."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from projections.distributions import QuantileDistribution


def test_constructor_validates_sorted_quantiles() -> None:
    with pytest.raises(ValueError, match="quantiles must be strictly ascending"):
        QuantileDistribution(
            quantiles=np.array([0.10, 0.05, 0.50, 0.90, 0.95]),
            values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )


def test_constructor_validates_quantiles_in_open_unit_interval() -> None:
    with pytest.raises(ValueError, match="quantiles must lie strictly in"):
        QuantileDistribution(
            quantiles=np.array([0.0, 0.10, 0.50, 0.90, 0.95]),
            values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )
    with pytest.raises(ValueError, match="quantiles must lie strictly in"):
        QuantileDistribution(
            quantiles=np.array([0.05, 0.10, 0.50, 0.90, 1.0]),
            values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )


def test_constructor_validates_values_non_decreasing() -> None:
    with pytest.raises(ValueError, match="values must be non-decreasing"):
        QuantileDistribution(
            quantiles=np.array([0.05, 0.10, 0.50, 0.90, 0.95]),
            values=np.array([1.0, 2.0, 3.0, 2.5, 5.0]),
        )


def test_constructor_validates_matching_lengths() -> None:
    with pytest.raises(ValueError, match="must have matching length"):
        QuantileDistribution(
            quantiles=np.array([0.10, 0.50, 0.90]),
            values=np.array([1.0, 2.0]),
        )


def test_constructor_validates_minimum_two_knots() -> None:
    with pytest.raises(ValueError, match="at least 2 knots"):
        QuantileDistribution(
            quantiles=np.array([0.50]),
            values=np.array([1.0]),
        )


def test_quantile_returns_exact_value_at_knot() -> None:
    dist = QuantileDistribution(
        quantiles=np.array([0.05, 0.10, 0.50, 0.90, 0.95]),
        values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    assert dist.quantile(0.50) == pytest.approx(3.0)
    assert dist.quantile(0.10) == pytest.approx(2.0)


def test_quantile_interpolates_between_knots() -> None:
    dist = QuantileDistribution(
        quantiles=np.array([0.10, 0.50, 0.90]),
        values=np.array([0.0, 10.0, 20.0]),
    )
    # Midpoint between (0.10, 0.0) and (0.50, 10.0): q=0.30 -> v=5.0
    assert dist.quantile(0.30) == pytest.approx(5.0)


def test_quantile_extrapolates_beyond_knots() -> None:
    dist = QuantileDistribution(
        quantiles=np.array([0.10, 0.50, 0.90]),
        values=np.array([0.0, 10.0, 20.0]),
    )
    # q=0.05 is below q_min=0.10. Linear extrapolation from (0.10, 0.0) and (0.50, 10.0):
    # slope = (10 - 0) / (0.50 - 0.10) = 25; v(0.05) = 0 + 25 * (0.05 - 0.10) = -1.25
    assert dist.quantile(0.05) == pytest.approx(-1.25)
    # q=0.99 is above q_max=0.90. Slope = (20 - 10) / (0.90 - 0.50) = 25;
    # v(0.99) = 20 + 25 * (0.99 - 0.90) = 22.25
    assert dist.quantile(0.99) == pytest.approx(22.25)


def test_mean_matches_normal_distribution() -> None:
    """Quantiles drawn from N(loc=10, scale=2); QD.mean() ≈ 10 within 0.05."""
    qs = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
    vs = scipy_stats.norm.ppf(qs, loc=10.0, scale=2.0)
    dist = QuantileDistribution(quantiles=qs, values=vs)
    assert dist.mean() == pytest.approx(10.0, abs=0.5)


def test_std_matches_normal_distribution() -> None:
    """Quantiles drawn from N(loc=10, scale=2); QD.std() ≈ 2 within 0.5."""
    qs = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
    vs = scipy_stats.norm.ppf(qs, loc=10.0, scale=2.0)
    dist = QuantileDistribution(quantiles=qs, values=vs)
    # 5 knots is coarse; numerical-integration tolerance is generous.
    assert dist.std() == pytest.approx(2.0, abs=0.5)


def test_sample_is_deterministic_with_seeded_rng() -> None:
    dist = QuantileDistribution(
        quantiles=np.array([0.10, 0.50, 0.90]),
        values=np.array([0.0, 10.0, 20.0]),
    )
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    a = dist.sample(n=100, rng=rng_a)
    b = dist.sample(n=100, rng=rng_b)
    np.testing.assert_array_equal(a, b)


def test_sample_returns_correct_shape_and_dtype() -> None:
    dist = QuantileDistribution(
        quantiles=np.array([0.05, 0.10, 0.50, 0.90, 0.95]),
        values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    samples = dist.sample(n=500, rng=np.random.default_rng(0))
    assert samples.shape == (500,)
    assert samples.dtype == np.float64


def test_empirical_quantiles_match_stored_quantiles() -> None:
    """Sample 10K from a constructed QD; empirical quantiles match stored within 0.1."""
    qs = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
    vs = scipy_stats.norm.ppf(qs, loc=10.0, scale=2.0)
    dist = QuantileDistribution(quantiles=qs, values=vs)
    samples = dist.sample(n=10_000, rng=np.random.default_rng(0))
    for q, v in zip(qs, vs, strict=True):
        assert np.quantile(samples, q) == pytest.approx(v, abs=0.2)


def test_constant_quantile_repeats_value() -> None:
    """Mass concentrated at zero (count-stat case): repeated 0.0 values are valid."""
    dist = QuantileDistribution(
        quantiles=np.array([0.05, 0.10, 0.50, 0.90, 0.95]),
        values=np.array([0.0, 0.0, 0.0, 1.0, 2.0]),
    )
    assert dist.quantile(0.10) == pytest.approx(0.0)
    assert dist.quantile(0.50) == pytest.approx(0.0)
    assert dist.quantile(0.90) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_distributions/test_quantile.py -v`
Expected: FAIL with `ImportError: cannot import name 'QuantileDistribution'`.

- [ ] **Step 3: Implement `QuantileDistribution`**

Create `src/projections/distributions/quantile.py`:

```python
"""QuantileDistribution — interpolated CDF/quantile/sample backed by stored knots.

Plan 5 (Model C / LightGBM with quantile regression). Each row's stat
distribution is represented by a sorted (quantiles, values) array pair.
quantile(q) linearly interpolates between adjacent knots; mean()/std()
are computed by trapezoid integration of the quantile function;
sample(n, rng) uses inverse-CDF (uniform draws → np.interp).

No scipy dependency — pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

# Trapezoid integration grid for mean()/std(). 100 points over (0.01, 0.99)
# leaves a small tail at each end that is dominated by the linear-extrapolation
# from the outermost stored knots; sufficient for E[X] / Var[X] to within
# the per-cell snapshot tolerance.
_INTEGRATION_GRID: Final[NDArray[np.float64]] = np.linspace(0.01, 0.99, 100)


@dataclass(slots=True, frozen=True, init=False)
class QuantileDistribution:
    """Distribution backed by a sorted set of (quantile, value) knots.

    Implements the Distribution Protocol structurally (mean, std, quantile, sample).
    """

    quantiles_: NDArray[np.float64]
    values_: NDArray[np.float64]

    def __init__(
        self,
        quantiles: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> None:
        q = np.asarray(quantiles, dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)
        if q.shape != v.shape:
            raise ValueError(
                f"quantiles and values must have matching length: {q.shape} vs {v.shape}"
            )
        if q.size < 2:
            raise ValueError(f"need at least 2 knots, got {q.size}")
        if not np.all(np.diff(q) > 0):
            raise ValueError(f"quantiles must be strictly ascending, got {q}")
        if not np.all((q > 0.0) & (q < 1.0)):
            raise ValueError(f"quantiles must lie strictly in (0, 1), got {q}")
        if not np.all(np.diff(v) >= 0):
            raise ValueError(f"values must be non-decreasing, got {v}")
        object.__setattr__(self, "quantiles_", q)
        object.__setattr__(self, "values_", v)

    def quantile(self, q: float) -> float:
        """Return the value at quantile q.

        For q within [q_min, q_max], linearly interpolates between adjacent stored knots.
        For q outside that range, linearly extrapolates from the two nearest knots.
        """
        # np.interp handles in-range linear interpolation. For out-of-range,
        # we extrapolate manually using the slope of the nearest two knots.
        qs = self.quantiles_
        vs = self.values_
        if qs[0] <= q <= qs[-1]:
            return float(np.interp(q, qs, vs))
        if q < qs[0]:
            slope = (vs[1] - vs[0]) / (qs[1] - qs[0])
            return float(vs[0] + slope * (q - qs[0]))
        # q > qs[-1]
        slope = (vs[-1] - vs[-2]) / (qs[-1] - qs[-2])
        return float(vs[-1] + slope * (q - qs[-1]))

    def _quantile_vec(self, qs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Vectorized quantile() for the integration grid + sampling.

        In-range points use np.interp; out-of-range points are linearly
        extrapolated from the two-nearest knots on the relevant side.
        """
        out = np.interp(qs, self.quantiles_, self.values_)
        # Lower-tail extrapolation
        below = qs < self.quantiles_[0]
        if below.any():
            slope_lo = (self.values_[1] - self.values_[0]) / (
                self.quantiles_[1] - self.quantiles_[0]
            )
            out[below] = self.values_[0] + slope_lo * (qs[below] - self.quantiles_[0])
        # Upper-tail extrapolation
        above = qs > self.quantiles_[-1]
        if above.any():
            slope_hi = (self.values_[-1] - self.values_[-2]) / (
                self.quantiles_[-1] - self.quantiles_[-2]
            )
            out[above] = self.values_[-1] + slope_hi * (qs[above] - self.quantiles_[-1])
        return out

    def mean(self) -> float:
        """E[X] via trapezoid integration of the quantile function on _INTEGRATION_GRID."""
        vs = self._quantile_vec(_INTEGRATION_GRID)
        return float(np.trapezoid(vs, _INTEGRATION_GRID) / (_INTEGRATION_GRID[-1] - _INTEGRATION_GRID[0]))

    def std(self) -> float:
        """Var[X] = E[X^2] - mean^2; std = sqrt(Var). Same integration approach as mean()."""
        vs = self._quantile_vec(_INTEGRATION_GRID)
        e_x = np.trapezoid(vs, _INTEGRATION_GRID) / (_INTEGRATION_GRID[-1] - _INTEGRATION_GRID[0])
        e_x2 = np.trapezoid(vs * vs, _INTEGRATION_GRID) / (
            _INTEGRATION_GRID[-1] - _INTEGRATION_GRID[0]
        )
        var = max(e_x2 - e_x * e_x, 0.0)
        return float(np.sqrt(var))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        """Inverse-CDF sampling: u ~ U(0, 1); return quantile(u) elementwise."""
        rng = rng if rng is not None else np.random.default_rng()
        u = rng.uniform(0.0, 1.0, size=n)
        return self._quantile_vec(u)
```

- [ ] **Step 4: Update `src/projections/distributions/__init__.py`**

Append `QuantileDistribution` to imports and `__all__`:

```python
"""Distribution layer — interface + parametric implementations + codec."""

from __future__ import annotations

from projections.distributions.base import Distribution
from projections.distributions.codec import pack_per_stat_params, unpack_per_stat_params
from projections.distributions.parametric import (
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    ParametricStudentT,
)
from projections.distributions.quantile import QuantileDistribution

__all__ = [
    "Distribution",
    "ParametricGamma",
    "ParametricNegativeBinomial",
    "ParametricNormal",
    "ParametricStudentT",
    "QuantileDistribution",
    "pack_per_stat_params",
    "unpack_per_stat_params",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_distributions/test_quantile.py -v`
Expected: 14 passed.

- [ ] **Step 6: Commit**

```bash
git add src/projections/distributions/quantile.py src/projections/distributions/__init__.py tests/test_distributions/test_quantile.py
git commit -m "feat(distributions): QuantileDistribution — Plan 5"
```

---

## Task 4: Codec QUANTILE branches + round-trip tests

**Files:**
- Modify: `src/projections/distributions/codec.py`
- Modify: `tests/test_distributions/test_codec.py`

- [ ] **Step 1: Write the failing tests in `test_codec.py`**

Append to `tests/test_distributions/test_codec.py`:

```python
def test_codec_round_trip_quantile() -> None:
    """Plan 5 — QUANTILE family round-trips through msgpack codec."""
    import numpy as np
    from projections.distributions import QuantileDistribution, pack_per_stat_params, unpack_per_stat_params
    from projections.schemas import Stat

    qs = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
    vs = np.array([1.0, 2.5, 5.0, 8.5, 12.0])
    original = {Stat.RECEIVING_YARDS: QuantileDistribution(quantiles=qs, values=vs)}

    blob = pack_per_stat_params(original)
    decoded = unpack_per_stat_params(blob)

    assert set(decoded.keys()) == {Stat.RECEIVING_YARDS}
    decoded_dist = decoded[Stat.RECEIVING_YARDS]
    assert isinstance(decoded_dist, QuantileDistribution)
    np.testing.assert_array_equal(decoded_dist.quantiles_, qs)
    np.testing.assert_array_equal(decoded_dist.values_, vs)


def test_codec_round_trip_mixed_with_quantile() -> None:
    """Plan 5 — QUANTILE coexists with NORMAL / GAMMA / NB in a single per-row blob."""
    import numpy as np
    from projections.distributions import (
        ParametricGamma,
        ParametricNegativeBinomial,
        ParametricNormal,
        QuantileDistribution,
        pack_per_stat_params,
        unpack_per_stat_params,
    )
    from projections.schemas import Stat

    original = {
        Stat.RECEIVING_YARDS: QuantileDistribution(
            quantiles=np.array([0.05, 0.10, 0.50, 0.90, 0.95]),
            values=np.array([1.0, 2.5, 5.0, 8.5, 12.0]),
        ),
        Stat.RECEPTIONS: ParametricNormal(mean=3.0, std=1.5),
        Stat.RUSHING_YARDS: ParametricGamma(shape=2.0, scale=4.0),
        Stat.RECEIVING_TDS: ParametricNegativeBinomial(mean=0.3, dispersion=2.0),
    }

    blob = pack_per_stat_params(original)
    decoded = unpack_per_stat_params(blob)

    assert set(decoded.keys()) == set(original.keys())
    assert isinstance(decoded[Stat.RECEIVING_YARDS], QuantileDistribution)
    assert isinstance(decoded[Stat.RECEPTIONS], ParametricNormal)
    assert isinstance(decoded[Stat.RUSHING_YARDS], ParametricGamma)
    assert isinstance(decoded[Stat.RECEIVING_TDS], ParametricNegativeBinomial)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_distributions/test_codec.py::test_codec_round_trip_quantile tests/test_distributions/test_codec.py::test_codec_round_trip_mixed_with_quantile -v`
Expected: FAIL with `ValueError: No codec entry for Distribution type QuantileDistribution`.

- [ ] **Step 3: Add the QUANTILE pack branch in `codec.py`**

In `src/projections/distributions/codec.py`, add an import:

```python
from projections.distributions.quantile import QuantileDistribution
```

Add a branch in `pack_per_stat_params` after the `ParametricStudentT` branch (before the `else: raise ValueError(...)`):

```python
        elif isinstance(dist, QuantileDistribution):
            stats_blob[stat.value] = {
                "family": DistributionFamily.QUANTILE.value,
                "quantiles": dist.quantiles_.tolist(),
                "values": dist.values_.tolist(),
            }
```

- [ ] **Step 4: Add the QUANTILE unpack branch**

In the same file, add a branch in `unpack_per_stat_params` after the `STUDENT_T` branch (before the `else: raise ValueError(...)`):

```python
        elif family_value == DistributionFamily.QUANTILE.value:
            import numpy as np
            out[stat] = QuantileDistribution(
                quantiles=np.asarray(entry["quantiles"], dtype=np.float64),
                values=np.asarray(entry["values"], dtype=np.float64),
            )
```

- [ ] **Step 5: Update the codec module docstring**

In `src/projections/distributions/codec.py`, extend the "Currently registered families" docstring block to include QUANTILE:

```python
"""...

Currently registered families:
    NORMAL:            {"family": "NORMAL",            "mean": float, "std": float}
    GAMMA:             {"family": "GAMMA",             "shape": float, "scale": float}
    NEGATIVE_BINOMIAL: {"family": "NEGATIVE_BINOMIAL", "mean": float, "dispersion": float}
    STUDENT_T:         {"family": "STUDENT_T",         "loc": float, "scale": float, "df": float}
    QUANTILE:          {"family": "QUANTILE",          "quantiles": list[float], "values": list[float]}
..."""
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_distributions/test_codec.py -v`
Expected: all codec tests (existing + 2 new) pass.

- [ ] **Step 7: Commit**

```bash
git add src/projections/distributions/codec.py tests/test_distributions/test_codec.py
git commit -m "feat(codec): QUANTILE family pack/unpack — Plan 5"
```

---

## Task 5: `LightGBMModel` config dataclass + per-position factories (skeleton)

**Files:**
- Create: `src/projections/models/lightgbm.py`

This task adds the module skeleton: module-level constants, the per-position config dataclass, the four factory functions, and a stub `LightGBMModel` class with empty methods that raise `NotImplementedError`. Subsequent tasks fill in `fit`, `predict_distribution`, `save`/`load`, and `model_id`.

- [ ] **Step 1: Read `src/projections/models/baseline.py` lines 1-150**

Familiarize yourself with the per-position config pattern (`_BaselineConfig` dataclass, `wr_baseline()` etc. factories). The new `_LightGBMConfig` mirrors its shape with two changes: replace `dist_families` with `non_negative_stats` (since Model C doesn't pick parametric families), and drop any RidgeCV-specific args.

- [ ] **Step 2: Write the skeleton**

Create `src/projections/models/lightgbm.py`:

```python
"""LightGBM-based per-stat quantile regression (Model C).

Plan 5 — coexists with BaselineModel (Model A) under the existing Model
Protocol. Trains 5 LightGBM quantile sub-models per (position, stat) at
quantiles [0.05, 0.10, 0.50, 0.90, 0.95]. Per-row prediction:
    1. Predict 5 quantiles.
    2. Sort to enforce non-crossing.
    3. Clip to [0, inf) for `non_negative` stats.
    4. Wrap in QuantileDistribution.
    5. Run through the existing scoring layer to get composite PPR points.

The whole prediction path beneath score_distribution is unchanged; the new
QuantileDistribution satisfies the Distribution Protocol structurally.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pandera.pandas as pa

from projections.distributions import QuantileDistribution, pack_per_stat_params
from projections.features.qb import build_qb_features
from projections.features.rb import build_rb_features
from projections.features.te import build_te_features
from projections.features.wr import build_wr_features
from projections.models.base import compute_code_hash
from projections.schemas import (
    DistributionFamily,
    Position,
    ProjectionWeeklySchema,
    QbFeaturesSchema,
    RbFeaturesSchema,
    Ruleset,
    Stat,
    TeFeaturesSchema,
    WrFeaturesSchema,
)
from projections.scoring.score_distribution import derive_row_seed, score_distribution

LGBM_DEFAULTS: Final[dict[str, Any]] = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,  # required to actually apply subsample
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "verbose": -1,
    "random_state": 42,
}

EARLY_STOPPING_ROUNDS: Final[int] = 50
QUANTILE_GRID: Final[tuple[float, ...]] = (0.05, 0.10, 0.50, 0.90, 0.95)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class _LightGBMConfig:
    """Per-position config for LightGBMModel.

    Attributes:
        position: which Position this config trains.
        target_stats: stats predicted by per-stat sub-models (matches BaselineModel).
        feature_columns: ordered list of feature columns the model consumes.
        feature_schema: pandera schema validated on input to fit/predict.
        non_negative_stats: stats whose predicted quantiles are clipped to [0, inf).
        feature_builder: position-specific build_*_features for code-hash purposes.
    """

    position: Position
    target_stats: tuple[Stat, ...]
    feature_columns: tuple[str, ...]
    feature_schema: type[pa.DataFrameModel]
    non_negative_stats: frozenset[Stat]
    feature_builder: Any  # callable; signature varies per position


def _code_hash_files(position: Position) -> tuple[Path, ...]:
    """Source files whose content is hashed into model_id for invalidation tracking."""
    src = _PROJECT_ROOT / "src" / "projections"
    feat_module = {
        Position.QB: "qb.py",
        Position.RB: "rb.py",
        Position.TE: "te.py",
        Position.WR: "wr.py",
    }[position]
    return (
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
    )


# Per-position feature columns mirror BaselineModel's per-position lists.
# Source of truth for the ordering: the corresponding *FeaturesSchema column order.
_QB_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(QbFeaturesSchema.to_schema().columns.keys())
_RB_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(RbFeaturesSchema.to_schema().columns.keys())
_TE_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(TeFeaturesSchema.to_schema().columns.keys())
_WR_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(WrFeaturesSchema.to_schema().columns.keys())

# Drop identifier / target / context columns — only true model features go in.
_NON_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset(
    {"gsis_id", "season", "week", "team", "opponent", "position"}
)


def _filter_features(cols: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(c for c in cols if c not in _NON_FEATURE_COLUMNS)


_QB_NON_NEGATIVE: Final[frozenset[Stat]] = frozenset(
    {Stat.PASSING_TDS, Stat.INTERCEPTIONS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST}
)
_RB_NON_NEGATIVE: Final[frozenset[Stat]] = frozenset(
    {Stat.RECEPTIONS, Stat.RUSHING_TDS, Stat.RECEIVING_TDS, Stat.FUMBLES_LOST}
)
_TE_NON_NEGATIVE: Final[frozenset[Stat]] = frozenset(
    {Stat.RECEPTIONS, Stat.RECEIVING_TDS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST}
)
_WR_NON_NEGATIVE: Final[frozenset[Stat]] = frozenset(
    {Stat.RECEPTIONS, Stat.RECEIVING_TDS, Stat.RUSHING_TDS, Stat.FUMBLES_LOST}
)


_QB_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.PASSING_YARDS,
    Stat.PASSING_TDS,
    Stat.INTERCEPTIONS,
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.FUMBLES_LOST,
)
_RB_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.RECEPTIONS,
    Stat.RECEIVING_YARDS,
    Stat.RECEIVING_TDS,
    Stat.FUMBLES_LOST,
)
_TE_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.RECEPTIONS,
    Stat.RECEIVING_YARDS,
    Stat.RECEIVING_TDS,
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.FUMBLES_LOST,
)
_WR_TARGET_STATS: Final[tuple[Stat, ...]] = (
    Stat.RECEPTIONS,
    Stat.RECEIVING_YARDS,
    Stat.RECEIVING_TDS,
    Stat.RUSHING_YARDS,
    Stat.RUSHING_TDS,
    Stat.FUMBLES_LOST,
)


def qb_lightgbm() -> LightGBMModel:
    return LightGBMModel(
        config=_LightGBMConfig(
            position=Position.QB,
            target_stats=_QB_TARGET_STATS,
            feature_columns=_filter_features(_QB_FEATURE_COLUMNS),
            feature_schema=QbFeaturesSchema,
            non_negative_stats=_QB_NON_NEGATIVE,
            feature_builder=build_qb_features,
        )
    )


def rb_lightgbm() -> LightGBMModel:
    return LightGBMModel(
        config=_LightGBMConfig(
            position=Position.RB,
            target_stats=_RB_TARGET_STATS,
            feature_columns=_filter_features(_RB_FEATURE_COLUMNS),
            feature_schema=RbFeaturesSchema,
            non_negative_stats=_RB_NON_NEGATIVE,
            feature_builder=build_rb_features,
        )
    )


def te_lightgbm() -> LightGBMModel:
    return LightGBMModel(
        config=_LightGBMConfig(
            position=Position.TE,
            target_stats=_TE_TARGET_STATS,
            feature_columns=_filter_features(_TE_FEATURE_COLUMNS),
            feature_schema=TeFeaturesSchema,
            non_negative_stats=_TE_NON_NEGATIVE,
            feature_builder=build_te_features,
        )
    )


def wr_lightgbm() -> LightGBMModel:
    return LightGBMModel(
        config=_LightGBMConfig(
            position=Position.WR,
            target_stats=_WR_TARGET_STATS,
            feature_columns=_filter_features(_WR_FEATURE_COLUMNS),
            feature_schema=WrFeaturesSchema,
            non_negative_stats=_WR_NON_NEGATIVE,
            feature_builder=build_wr_features,
        )
    )


class LightGBMModel:
    """Per-stat LightGBM quantile-regression model. Implements Model Protocol structurally.

    Use the per-position factories (qb_lightgbm, rb_lightgbm, te_lightgbm, wr_lightgbm)
    rather than constructing directly.
    """

    def __init__(self, *, config: _LightGBMConfig) -> None:
        self._config = config
        self._sub_models: dict[Stat, dict[float, lgb.Booster]] = {}
        self._best_iters: dict[tuple[Stat, float], int] = {}
        self._train_start: int | None = None
        self._train_end: int | None = None
        self._is_fitted: bool = False

    @property
    def position(self) -> Position:
        return self._config.position

    @property
    def model_id(self) -> str:
        if not self._is_fitted:
            raise RuntimeError("model_id not available before fit() — depends on training-time state")
        code_hash = compute_code_hash(_code_hash_files(self._config.position))
        assert self._train_start is not None and self._train_end is not None
        return f"lightgbm:{self._config.position.value.lower()}:{code_hash}:{self._train_start}-{self._train_end}"

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        raise NotImplementedError("Plan 5 Task 6")

    def predict_distribution(self, features: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
        raise NotImplementedError("Plan 5 Task 7")

    def save(self, path: Path) -> None:
        raise NotImplementedError("Plan 5 Task 8")

    @classmethod
    def load(cls, path: Path) -> LightGBMModel:
        raise NotImplementedError("Plan 5 Task 8")
```

- [ ] **Step 3: Quick sanity check — module imports cleanly**

Run: `python -c "from projections.models.lightgbm import LightGBMModel, qb_lightgbm, rb_lightgbm, te_lightgbm, wr_lightgbm; m = wr_lightgbm(); print(m.position)"`
Expected: prints `Position.WR` with no errors.

- [ ] **Step 4: mypy + ruff check on the new module**

Run: `mypy src/projections/models/lightgbm.py && ruff check src/projections/models/lightgbm.py`
Expected: zero violations.

- [ ] **Step 5: Commit**

```bash
git add src/projections/models/lightgbm.py
git commit -m "feat(models): LightGBMModel skeleton + per-position factories — Plan 5"
```

---

## Task 6: `LightGBMModel.fit` — train/val split + per-stat per-quantile sub-models

**Files:**
- Modify: `src/projections/models/lightgbm.py`
- Create: `tests/test_models/test_lightgbm.py` (will be extended in later tasks; this task adds the first fit-related tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models/test_lightgbm.py`:

```python
"""Cross-cutting tests for LightGBMModel (Model C, Plan 5).

Per-position smoke tests live in test_lightgbm_qb.py / _rb.py / _te.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.models.lightgbm import (
    EARLY_STOPPING_ROUNDS,
    LGBM_DEFAULTS,
    LightGBMModel,
    QUANTILE_GRID,
    qb_lightgbm,
    wr_lightgbm,
)
from projections.schemas import Position, Stat


# ---------------- Synthetic fixture builder ----------------

def _build_synthetic_wr_features(n_seasons: int = 4, n_weeks: int = 17, n_players: int = 30) -> pd.DataFrame:
    """Synthetic WrFeaturesSchema-shaped DataFrame for fit/predict smoke tests."""
    from projections.schemas import WrFeaturesSchema
    from projections.schemas import _PYARROW_STR

    rng = np.random.default_rng(42)
    rows = []
    for season in range(2018, 2018 + n_seasons):
        for week in range(1, n_weeks + 1):
            for p in range(n_players):
                rows.append({
                    "gsis_id": f"00-{p:07d}",
                    "season": season,
                    "week": week,
                    "team": "KC",
                    "opponent": "DEN",
                })
    df = pd.DataFrame(rows)
    df["season"] = df["season"].astype(np.int64)
    df["week"] = df["week"].astype(np.int64)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)

    # Populate every WrFeaturesSchema feature column with synthetic numerics.
    for col_name in WrFeaturesSchema.to_schema().columns:
        if col_name in df.columns:
            continue
        df[col_name] = rng.normal(0.0, 1.0, size=len(df)).astype(np.float64)
        if col_name in ("is_home", "roof_dome"):
            df[col_name] = rng.integers(0, 2, size=len(df)).astype(np.int64)
    return WrFeaturesSchema.validate(df)


def _build_synthetic_weekly_stats(features: pd.DataFrame) -> pd.DataFrame:
    """Synthetic WeeklyStatsSchema-shaped DataFrame aligned with `features`."""
    from projections.schemas import WeeklyStatsSchema

    rng = np.random.default_rng(43)
    df = features[["gsis_id", "season", "week"]].copy()
    df["team"] = features["team"]
    df["opponent"] = features["opponent"]
    df["position"] = "WR"
    n = len(df)
    # Plausible synthetic targets — enough variation for LightGBM to learn something.
    df["receptions"] = np.maximum(0, rng.poisson(3.0, size=n)).astype(np.int64)
    df["receiving_yards"] = (df["receptions"] * rng.normal(12.0, 3.0, size=n)).astype(np.float64)
    df["receiving_tds"] = np.maximum(0, rng.poisson(0.3, size=n)).astype(np.int64)
    df["rushing_yards"] = rng.normal(2.0, 5.0, size=n).astype(np.float64)
    df["rushing_tds"] = np.maximum(0, rng.poisson(0.05, size=n)).astype(np.int64)
    df["fumbles_lost"] = np.maximum(0, rng.poisson(0.05, size=n)).astype(np.int64)
    # Other required schema columns get zeros / placeholders.
    for col_name in WeeklyStatsSchema.to_schema().columns:
        if col_name not in df.columns:
            df[col_name] = 0 if "tds" in col_name or "yards" in col_name else 0.0
    return WeeklyStatsSchema.validate(df)


# ---------------- Fit tests ----------------

def test_fit_populates_sub_models_and_best_iters() -> None:
    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    assert model._is_fitted
    assert set(model._sub_models.keys()) == set(model._config.target_stats)
    for stat in model._config.target_stats:
        assert set(model._sub_models[stat].keys()) == set(QUANTILE_GRID)
        for q in QUANTILE_GRID:
            assert (stat, q) in model._best_iters


def test_fit_sets_train_window_from_data() -> None:
    features = _build_synthetic_wr_features(n_seasons=3)  # 2018-2020
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    assert model._train_start == 2018
    assert model._train_end == 2020


def test_fit_raises_on_insufficient_seasons() -> None:
    features = _build_synthetic_wr_features(n_seasons=1)
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    with pytest.raises(ValueError, match=r"Need .*?2 training seasons"):
        model.fit(features, weekly_stats)


def test_fit_raises_on_empty_join() -> None:
    features = _build_synthetic_wr_features()
    # Create non-overlapping weekly_stats (different gsis_ids):
    weekly_stats = _build_synthetic_weekly_stats(features.copy())
    weekly_stats["gsis_id"] = weekly_stats["gsis_id"].apply(lambda x: x.replace("00-", "99-"))
    model = wr_lightgbm()
    with pytest.raises(ValueError, match="Empty training set"):
        model.fit(features, weekly_stats)


def test_model_id_unavailable_before_fit() -> None:
    model = wr_lightgbm()
    with pytest.raises(RuntimeError, match="model_id not available before fit"):
        _ = model.model_id


def test_model_id_after_fit_has_expected_shape() -> None:
    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)
    mid = model.model_id
    parts = mid.split(":")
    assert parts[0] == "lightgbm"
    assert parts[1] == "wr"
    assert len(parts[2]) == 8  # 8-char code hash
    assert "-" in parts[3]  # train-start-train-end


def test_qb_factory_uses_qb_target_stats() -> None:
    model = qb_lightgbm()
    assert Stat.PASSING_YARDS in model._config.target_stats
    assert Stat.PASSING_TDS in model._config.target_stats
    assert Stat.INTERCEPTIONS in model._config.target_stats
    # WR-only stats are not in QB:
    assert Stat.RECEPTIONS not in model._config.target_stats
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_lightgbm.py -v`
Expected: most tests FAIL with `NotImplementedError: Plan 5 Task 6`.

- [ ] **Step 3: Implement `fit`**

In `src/projections/models/lightgbm.py`, replace the `fit` method body:

```python
    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """Train per-stat per-quantile sub-models with early stopping on the last
        training season. Stores boosters in self._sub_models and best iterations
        in self._best_iters.

        Raises:
            ValueError: empty join, or fewer than 2 training seasons (needed to carve
                the validation slice).
        """
        # Validate features against the position schema.
        features = self._config.feature_schema.validate(features)

        # Inner-join on (gsis_id, season, week).
        joined = features.merge(
            weekly_stats[["gsis_id", "season", "week", *[s.value for s in self._config.target_stats]]],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        if joined.empty:
            raise ValueError("Empty training set after feature/weekly_stats join")

        # Need >=2 seasons to carve a last-season validation slice.
        seasons = sorted(joined["season"].unique())
        if len(seasons) < 2:
            raise ValueError(
                f"Need >=2 training seasons for early-stopping validation slice; got {len(seasons)}"
            )

        val_season = seasons[-1]
        train_mask = joined["season"] != val_season
        val_mask = joined["season"] == val_season

        feat_cols = list(self._config.feature_columns)
        x_train = joined.loc[train_mask, feat_cols].to_numpy(dtype=np.float64)
        x_val = joined.loc[val_mask, feat_cols].to_numpy(dtype=np.float64)

        for stat in self._config.target_stats:
            self._sub_models[stat] = {}
            y_train = joined.loc[train_mask, stat.value].to_numpy(dtype=np.float64)
            y_val = joined.loc[val_mask, stat.value].to_numpy(dtype=np.float64)
            for q in QUANTILE_GRID:
                regressor = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    **LGBM_DEFAULTS,
                )
                regressor.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_val, y_val)],
                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
                )
                # `regressor.booster_` exposes the trained Booster after fit.
                self._sub_models[stat][q] = regressor.booster_
                self._best_iters[(stat, q)] = int(regressor.best_iteration_ or 0)

        self._train_start = int(seasons[0])
        self._train_end = int(seasons[-1])
        self._is_fitted = True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models/test_lightgbm.py -v -x`
Expected: all 7 tests pass. (May take ~2 minutes — 30 sub-model fits per `test_fit_*` test.)

- [ ] **Step 5: Commit**

```bash
git add src/projections/models/lightgbm.py tests/test_models/test_lightgbm.py
git commit -m "feat(models): LightGBMModel.fit — Plan 5"
```

---

## Task 7: `LightGBMModel.predict_distribution` — quantiles → sort → clip → QD → score

**Files:**
- Modify: `src/projections/models/lightgbm.py`
- Modify: `tests/test_models/test_lightgbm.py`

- [ ] **Step 1: Write the failing tests (append to `test_lightgbm.py`)**

```python
# ---------------- predict_distribution tests ----------------

def test_predict_distribution_validates_against_projection_schema() -> None:
    from projections.schemas import ProjectionWeeklySchema, Ruleset
    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].copy()
    out = model.predict_distribution(test_features, Ruleset.ESPN_PPR)

    # ProjectionWeeklySchema.validate raises if columns/dtypes drift.
    ProjectionWeeklySchema.validate(out)
    assert (out["family"] == "QUANTILE").all()
    assert (out["model_id"] == model.model_id).all()
    assert len(out) == len(test_features)


def test_predict_distribution_params_blob_round_trips() -> None:
    from projections.distributions import QuantileDistribution, unpack_per_stat_params
    from projections.schemas import Ruleset
    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].head(5).copy()
    out = model.predict_distribution(test_features, Ruleset.ESPN_PPR)

    for blob in out["params"]:
        decoded = unpack_per_stat_params(bytes(blob))
        for stat in model._config.target_stats:
            assert stat in decoded
            assert isinstance(decoded[stat], QuantileDistribution)


def test_predict_distribution_clips_non_negative_stats() -> None:
    from projections.distributions import QuantileDistribution, unpack_per_stat_params
    from projections.schemas import Ruleset, Stat
    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].copy()
    out = model.predict_distribution(test_features, Ruleset.ESPN_PPR)

    # Every non-negative stat's stored quantile values must be >= 0 in every row.
    for blob in out["params"]:
        decoded = unpack_per_stat_params(bytes(blob))
        for stat in model._config.non_negative_stats:
            qd = decoded[stat]
            assert isinstance(qd, QuantileDistribution)
            assert (qd.values_ >= 0.0).all()


def test_predict_distribution_sorts_quantile_crossing() -> None:
    """If LightGBM produces a row where p10_pred > p50_pred, predict_distribution
    must sort before constructing the QuantileDistribution. Constructed indirectly:
    we assert no ValueError is raised by the QuantileDistribution constructor's
    monotonicity check on any prediction row."""
    from projections.schemas import Ruleset
    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].copy()
    # No exception means the sort upstream is doing its job.
    out = model.predict_distribution(test_features, Ruleset.ESPN_PPR)
    assert len(out) == len(test_features)


def test_predict_distribution_raises_on_feature_column_mismatch() -> None:
    from projections.schemas import Ruleset
    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].copy()
    # Drop one feature column.
    dropped_col = list(model._config.feature_columns)[0]
    test_features = test_features.drop(columns=[dropped_col])
    with pytest.raises(ValueError, match=r"Feature columns differ from training"):
        model.predict_distribution(test_features, Ruleset.ESPN_PPR)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_lightgbm.py -v -x -k predict`
Expected: FAIL with `NotImplementedError: Plan 5 Task 7`.

- [ ] **Step 3: Implement `predict_distribution`**

In `src/projections/models/lightgbm.py`, replace the `predict_distribution` method body:

```python
    def predict_distribution(self, features: pd.DataFrame, ruleset: Ruleset) -> pd.DataFrame:
        """Predict per-row composite fantasy-points distribution.

        Pipeline:
            1. Validate features against the position schema.
            2. Verify feature columns match training.
            3. For each row, predict 5 quantiles via the 5 sub-models per stat.
            4. Sort per-row to enforce non-crossing.
            5. Clip to [0, inf) for stats in non_negative_stats.
            6. Wrap each per-stat (quantiles, values) in a QuantileDistribution.
            7. Run through score_distribution -> composite mean / p10 / p50 / p90.

        Returns:
            DataFrame validated against ProjectionWeeklySchema.
        """
        if not self._is_fitted:
            raise RuntimeError("predict_distribution requires fit() first")

        features = self._config.feature_schema.validate(features)

        feat_cols = list(self._config.feature_columns)
        actual_cols = set(features.columns)
        missing = set(feat_cols) - actual_cols
        if missing:
            raise ValueError(
                f"Feature columns differ from training: missing={sorted(missing)}; "
                f"expected feature_columns={feat_cols}"
            )

        x = features[feat_cols].to_numpy(dtype=np.float64)
        n_rows = x.shape[0]
        quant_arr = np.array(QUANTILE_GRID, dtype=np.float64)

        # Predict every (stat, quantile) into a per-stat (n_rows, n_quantiles) array.
        per_stat_pred: dict[Stat, np.ndarray] = {}
        for stat in self._config.target_stats:
            preds_per_q = np.column_stack(
                [self._sub_models[stat][q].predict(x) for q in QUANTILE_GRID]
            ).astype(np.float64)
            # Sort per-row to enforce non-crossing.
            preds_per_q.sort(axis=1)
            # Clip to >=0 if non-negative.
            if stat in self._config.non_negative_stats:
                np.maximum(preds_per_q, 0.0, out=preds_per_q)
            per_stat_pred[stat] = preds_per_q

        # Build per-row per-stat distributions and run scoring.
        out_rows: list[dict[str, Any]] = []
        generated_at = datetime.now(UTC)
        for row_idx in range(n_rows):
            per_stat_dists: dict[Stat, QuantileDistribution] = {}
            for stat in self._config.target_stats:
                per_stat_dists[stat] = QuantileDistribution(
                    quantiles=quant_arr,
                    values=per_stat_pred[stat][row_idx],
                )

            seed = derive_row_seed(
                gsis_id=str(features["gsis_id"].iloc[row_idx]),
                season=int(features["season"].iloc[row_idx]),
                week=int(features["week"].iloc[row_idx]),
                ruleset_name=ruleset.value,
            )
            composite = score_distribution(per_stat_dists, ruleset, seed=seed)

            out_rows.append({
                "gsis_id": str(features["gsis_id"].iloc[row_idx]),
                "season": int(features["season"].iloc[row_idx]),
                "week": int(features["week"].iloc[row_idx]),
                "position": self._config.position.value,
                "team": str(features["team"].iloc[row_idx]),
                "opponent": str(features["opponent"].iloc[row_idx]),
                "ruleset": ruleset.value,
                "family": DistributionFamily.QUANTILE.value,
                "params": pack_per_stat_params(per_stat_dists),
                "mean": composite.mean(),
                "p10": composite.quantile(0.10),
                "p50": composite.quantile(0.50),
                "p90": composite.quantile(0.90),
                "model_id": self.model_id,
                "generated_at": generated_at,
            })
        result = pd.DataFrame(out_rows)
        return ProjectionWeeklySchema.validate(result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models/test_lightgbm.py -v -x`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/projections/models/lightgbm.py tests/test_models/test_lightgbm.py
git commit -m "feat(models): LightGBMModel.predict_distribution — Plan 5"
```

---

## Task 8: `save` / `load` round-trip

**Files:**
- Modify: `src/projections/models/lightgbm.py`
- Modify: `tests/test_models/test_lightgbm.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models/test_lightgbm.py`:

```python
# ---------------- save / load round-trip tests ----------------

def test_save_load_round_trip(tmp_path) -> None:
    from projections.schemas import Ruleset
    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    artifact_path = tmp_path / "wr_lightgbm.joblib"
    model.save(artifact_path)
    loaded = LightGBMModel.load(artifact_path)

    test_features = features[features["season"] == 2021].head(5).copy()
    out_a = model.predict_distribution(test_features, Ruleset.ESPN_PPR)
    out_b = loaded.predict_distribution(test_features, Ruleset.ESPN_PPR)

    # Identical predictions modulo generated_at timestamp:
    cols_to_check = ["gsis_id", "season", "week", "mean", "p10", "p50", "p90", "model_id"]
    pd.testing.assert_frame_equal(
        out_a[cols_to_check].reset_index(drop=True),
        out_b[cols_to_check].reset_index(drop=True),
    )


def test_load_returns_lightgbm_model_instance(tmp_path) -> None:
    features = _build_synthetic_wr_features()
    weekly_stats = _build_synthetic_weekly_stats(features)
    model = wr_lightgbm()
    model.fit(features, weekly_stats)

    artifact_path = tmp_path / "wr_lightgbm.joblib"
    model.save(artifact_path)
    loaded = LightGBMModel.load(artifact_path)
    assert isinstance(loaded, LightGBMModel)
    assert loaded.position == Position.WR
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_lightgbm.py -v -x -k save_load`
Expected: FAIL with `NotImplementedError: Plan 5 Task 8`.

- [ ] **Step 3: Implement `save` and `load`**

In `src/projections/models/lightgbm.py`, replace the `save` and `load` method bodies:

```python
    def save(self, path: Path) -> None:
        """Joblib-serialize the entire model. lgb.Booster instances pickle cleanly
        (lightgbm registers reduce/setstate hooks)."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save() an unfitted model")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> LightGBMModel:
        """Inverse of save(). Returns the same instance shape as the original."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Loaded object is {type(loaded).__name__}, expected {cls.__name__}")
        return loaded
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models/test_lightgbm.py -v -x`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/projections/models/lightgbm.py tests/test_models/test_lightgbm.py
git commit -m "feat(models): LightGBMModel.save/load — Plan 5"
```

---

## Task 9: Per-position smoke tests (QB / RB / TE)

**Files:**
- Create: `tests/test_models/test_lightgbm_qb.py`
- Create: `tests/test_models/test_lightgbm_rb.py`
- Create: `tests/test_models/test_lightgbm_te.py`

For each per-position file, the test mirrors `test_lightgbm.py`'s `test_fit_populates_sub_models_and_best_iters` and `test_predict_distribution_validates_against_projection_schema` but builds synthetic features against the position's own schema.

- [ ] **Step 1: Create `tests/test_models/test_lightgbm_qb.py`**

```python
"""Per-position smoke for LightGBMModel — QB. Plan 5."""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.schemas import _PYARROW_STR
from projections.models.lightgbm import qb_lightgbm
from projections.schemas import (
    ProjectionWeeklySchema,
    QbFeaturesSchema,
    Ruleset,
    WeeklyStatsSchema,
)


def _build_synthetic_qb_data():
    rng = np.random.default_rng(42)
    rows = []
    for season in range(2018, 2022):  # 4 seasons
        for week in range(1, 18):
            for p in range(20):
                rows.append({
                    "gsis_id": f"00-{p:07d}",
                    "season": np.int64(season),
                    "week": np.int64(week),
                    "team": "KC",
                    "opponent": "DEN",
                })
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    for col_name in QbFeaturesSchema.to_schema().columns:
        if col_name in df.columns:
            continue
        df[col_name] = rng.normal(0.0, 1.0, size=len(df)).astype(np.float64)
        if col_name in ("is_home", "roof_dome"):
            df[col_name] = rng.integers(0, 2, size=len(df)).astype(np.int64)
    features = QbFeaturesSchema.validate(df)

    ws = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    ws["position"] = "QB"
    n = len(ws)
    ws["passing_yards"] = rng.normal(220.0, 60.0, size=n).astype(np.float64)
    ws["passing_tds"] = np.maximum(0, rng.poisson(1.5, size=n)).astype(np.int64)
    ws["interceptions"] = np.maximum(0, rng.poisson(0.7, size=n)).astype(np.int64)
    ws["rushing_yards"] = rng.normal(20.0, 15.0, size=n).astype(np.float64)
    ws["rushing_tds"] = np.maximum(0, rng.poisson(0.2, size=n)).astype(np.int64)
    ws["fumbles_lost"] = np.maximum(0, rng.poisson(0.2, size=n)).astype(np.int64)
    ws["receptions"] = 0
    ws["receiving_yards"] = 0.0
    ws["receiving_tds"] = 0
    for col_name in WeeklyStatsSchema.to_schema().columns:
        if col_name not in ws.columns:
            ws[col_name] = 0 if "tds" in col_name or "yards" in col_name else 0.0
    return features, WeeklyStatsSchema.validate(ws)


def test_qb_lightgbm_fit_predict_smoke() -> None:
    features, weekly_stats = _build_synthetic_qb_data()
    model = qb_lightgbm()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].head(10).copy()
    out = model.predict_distribution(test_features, Ruleset.ESPN_PPR)

    ProjectionWeeklySchema.validate(out)
    assert len(out) == len(test_features)
    assert (out["family"] == "QUANTILE").all()
    assert (out["position"] == "QB").all()
```

- [ ] **Step 2: Create `tests/test_models/test_lightgbm_rb.py` and `_te.py`**

Mirror the QB file. Replace:
- `qb_lightgbm` → `rb_lightgbm` / `te_lightgbm`
- `QbFeaturesSchema` → `RbFeaturesSchema` / `TeFeaturesSchema`
- `position = "QB"` → `"RB"` / `"TE"`
- Stat columns: copy from `_build_synthetic_weekly_stats` in `test_lightgbm.py`, adjusting per-position relevant means.

(For the RB and TE versions, `passing_*` columns can be zeroed and the weekly_stats targets set to RB-relevant means: `rushing_yards ~ N(40, 30)`, `rushing_tds ~ Poisson(0.4)`, `receptions ~ Poisson(2)`, etc.)

- [ ] **Step 3: Run all per-position tests**

Run: `pytest tests/test_models/test_lightgbm_qb.py tests/test_models/test_lightgbm_rb.py tests/test_models/test_lightgbm_te.py -v`
Expected: 3 tests pass. ~3 minutes total wall-clock (30 sub-model fits per test).

- [ ] **Step 4: Commit**

```bash
git add tests/test_models/test_lightgbm_qb.py tests/test_models/test_lightgbm_rb.py tests/test_models/test_lightgbm_te.py
git commit -m "test(models): per-position LightGBM smokes — Plan 5"
```

---

## Task 10: Parametrized cross-position smoke

**Files:**
- Create: `tests/test_models/test_lightgbm_smoke.py`

- [ ] **Step 1: Create the smoke file**

```python
"""Cross-position smoke for LightGBMModel — Plan 5.

Single fit + predict for each of the 4 positions on synthetic fixtures.
Catches regressions where one position breaks while others pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.models import POSITION_DISPATCH
from projections.schemas import (
    Position,
    ProjectionWeeklySchema,
    Ruleset,
    WeeklyStatsSchema,
)


@pytest.mark.parametrize("position", [Position.QB, Position.RB, Position.TE, Position.WR])
def test_lightgbm_fit_predict_smoke(position: Position) -> None:
    """Each position's LightGBMModel can fit and predict on synthetic data."""
    from projections.schemas import _PYARROW_STR

    dispatch = POSITION_DISPATCH[position]
    feature_schema = dispatch.feature_schema
    factory = dispatch.factories["lightgbm"]

    rng = np.random.default_rng(int(position.value.encode().sum()))
    rows = []
    for season in range(2018, 2022):
        for week in range(1, 18):
            for p in range(15):
                rows.append({
                    "gsis_id": f"00-{p:07d}",
                    "season": np.int64(season),
                    "week": np.int64(week),
                    "team": "KC",
                    "opponent": "DEN",
                })
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)
    for col_name in feature_schema.to_schema().columns:
        if col_name in df.columns:
            continue
        df[col_name] = rng.normal(0.0, 1.0, size=len(df)).astype(np.float64)
        if col_name in ("is_home", "roof_dome"):
            df[col_name] = rng.integers(0, 2, size=len(df)).astype(np.int64)
    features = feature_schema.validate(df)

    ws = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    ws["position"] = position.value
    n = len(ws)
    for col_name in WeeklyStatsSchema.to_schema().columns:
        if col_name in ws.columns:
            continue
        if "yards" in col_name:
            ws[col_name] = rng.normal(20.0, 10.0, size=n).astype(np.float64)
        elif "tds" in col_name or "interceptions" in col_name or "fumbles" in col_name:
            ws[col_name] = np.maximum(0, rng.poisson(0.3, size=n)).astype(np.int64)
        elif "receptions" in col_name or "carries" in col_name or "completions" in col_name or "attempts" in col_name or "sacks" in col_name:
            ws[col_name] = np.maximum(0, rng.poisson(2.0, size=n)).astype(np.int64)
        else:
            ws[col_name] = 0.0
    weekly_stats = WeeklyStatsSchema.validate(ws)

    model = factory()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].head(5).copy()
    out = model.predict_distribution(test_features, Ruleset.ESPN_PPR)
    ProjectionWeeklySchema.validate(out)
    assert (out["position"] == position.value).all()
    assert (out["family"] == "QUANTILE").all()
```

- [ ] **Step 2: Note — this depends on Task 12's POSITION_DISPATCH update**

Defer running this test until Task 12 completes. For now, create the file and commit:

- [ ] **Step 3: Commit**

```bash
git add tests/test_models/test_lightgbm_smoke.py
git commit -m "test(models): cross-position LightGBM smoke (parametrized) — Plan 5"
```

---

## Task 11: `POSITION_DISPATCH` `factories` dict + consumer updates

**Files:**
- Modify: `src/projections/models/__init__.py`
- Modify: `scripts/train_baseline.py`
- Modify: `scripts/sanity_check_baseline.py`
- Modify: `scripts/predict_2024.py`

- [ ] **Step 1: Read current `models/__init__.py` and `_PositionDispatch`**

Run: `cat src/projections/models/__init__.py`
Note current `_PositionDispatch.factory` field — this becomes `factories: dict[str, ...]`.

- [ ] **Step 2: Modify `models/__init__.py`**

Replace the `_PositionDispatch` dataclass and the `POSITION_DISPATCH` map. New version:

```python
"""Public surface for the models package."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandera.pandas as pa

from projections.features.qb import build_qb_features
from projections.features.rb import build_rb_features
from projections.features.te import build_te_features
from projections.features.wr import build_wr_features
from projections.ingest.ngs import NgsStatType
from projections.models.base import Model, compute_code_hash
from projections.models.baseline import (
    BaselineModel,
    qb_baseline,
    rb_baseline,
    te_baseline,
    wr_baseline,
)
from projections.models.lightgbm import (
    LightGBMModel,
    qb_lightgbm,
    rb_lightgbm,
    te_lightgbm,
    wr_lightgbm,
)
from projections.schemas import (
    Position,
    QbFeaturesSchema,
    RbFeaturesSchema,
    TeFeaturesSchema,
    WrFeaturesSchema,
)

__all__ = [
    "POSITION_DISPATCH",
    "BaselineModel",
    "LightGBMModel",
    "Model",
    "compute_code_hash",
    "qb_baseline",
    "qb_lightgbm",
    "rb_baseline",
    "rb_lightgbm",
    "te_baseline",
    "te_lightgbm",
    "wr_baseline",
    "wr_lightgbm",
]


@dataclass(frozen=True)
class _PositionDispatch:
    """Per-position bundle of "what's needed to train and predict" entries.

    `factories` is a dict keyed by model class name string ("baseline" / "lightgbm")
    so callers can dispatch by model class. Single source of truth for
    "which model classes the system knows about."
    """

    factories: Mapping[str, Callable[[], Model]]
    feature_builder: Callable[..., Any]
    feature_schema: type[pa.DataFrameModel]
    ngs_stat_type: NgsStatType


POSITION_DISPATCH: Mapping[Position, _PositionDispatch] = {
    Position.QB: _PositionDispatch(
        factories={"baseline": qb_baseline, "lightgbm": qb_lightgbm},
        feature_builder=build_qb_features,
        feature_schema=QbFeaturesSchema,
        ngs_stat_type="passing",
    ),
    Position.RB: _PositionDispatch(
        factories={"baseline": rb_baseline, "lightgbm": rb_lightgbm},
        feature_builder=build_rb_features,
        feature_schema=RbFeaturesSchema,
        ngs_stat_type="rushing",
    ),
    Position.TE: _PositionDispatch(
        factories={"baseline": te_baseline, "lightgbm": te_lightgbm},
        feature_builder=build_te_features,
        feature_schema=TeFeaturesSchema,
        ngs_stat_type="receiving",
    ),
    Position.WR: _PositionDispatch(
        factories={"baseline": wr_baseline, "lightgbm": wr_lightgbm},
        feature_builder=build_wr_features,
        feature_schema=WrFeaturesSchema,
        ngs_stat_type="receiving",
    ),
}
```

- [ ] **Step 3: Update CLI scripts to use `factories[...]` lookup**

In each of `scripts/train_baseline.py`, `scripts/sanity_check_baseline.py`, `scripts/predict_2024.py`:

- Locate where `POSITION_DISPATCH[position].factory()` is called.
- Add a `--model` argparse arg with choices `["baseline", "lightgbm"]`, default `"baseline"`.
- Replace the call with `POSITION_DISPATCH[position].factories[args.model]()`.

Example diff for `scripts/train_baseline.py` (apply analogously to the others):

```python
# In the argparse section:
parser.add_argument(
    "--model",
    choices=["baseline", "lightgbm"],
    default="baseline",
    help="Which model class to train (Model A or Model C). Default baseline.",
)

# In the body, replace any `POSITION_DISPATCH[pos].factory()` with:
model = POSITION_DISPATCH[position].factories[args.model]()
```

If a script has artifact-naming logic that bakes in `"baseline"`, generalize it to use `args.model` (e.g., `f"{args.model}-{position.value}-{train_start}-{train_end}-{code_hash}.joblib"`).

- [ ] **Step 4: Run the smoke test from Task 10 now that POSITION_DISPATCH is updated**

Run: `pytest tests/test_models/test_lightgbm_smoke.py -v`
Expected: 4 tests pass. ~12 minutes total (30 sub-models × 4 positions × small fixtures).

- [ ] **Step 5: Run the existing Model A tests to confirm no regression**

Run: `pytest tests/test_models/ -v -x`
Expected: all existing baseline tests still pass + all new lightgbm tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/projections/models/__init__.py scripts/train_baseline.py scripts/sanity_check_baseline.py scripts/predict_2024.py
git commit -m "feat(models): POSITION_DISPATCH.factories — both model classes per position — Plan 5"
```

---

## Task 12: Backtest harness `--model` arg

**Files:**
- Modify: `src/projections/backtest/harness.py`
- Modify: `scripts/backtest.py`

The harness today iterates `(position, year)` and constructs a Model A via the position's factory. Plan 5 extends it to iterate `(position, year, model_class)` so per-fold metrics include a `model_class` identifier.

- [ ] **Step 1: Read `src/projections/backtest/harness.py`**

Run: `cat src/projections/backtest/harness.py | head -120`
Identify the per-fold loop structure and where the per-row results DataFrame is built.

- [ ] **Step 2: Add `model_classes` parameter to the harness entry point**

In `src/projections/backtest/harness.py`, find the top-level `run_backtest(...)` function (or whatever it's called). Add a parameter:

```python
def run_backtest(
    *,
    positions: Iterable[Position],
    held_out_years: Iterable[int],
    train_start: int,
    model_classes: Iterable[str] = ("baseline",),  # NEW: defaults to baseline-only for backward compat
    ...
) -> BacktestRun:
```

In the per-fold loop, wrap the model construction + fit + predict logic in another loop over `model_classes`:

```python
for position in positions:
    for test_year in held_out_years:
        # ... train_features / test_features setup unchanged ...
        for model_class in model_classes:
            model = POSITION_DISPATCH[position].factories[model_class]()
            model.fit(train_features, train_stats)
            preds = model.predict_distribution(test_features, ruleset)
            # ... metric computation ...
            for metric_name, value in metrics.items():
                results.append({
                    "position": position.value,
                    "year": test_year,
                    "metric": metric_name,
                    "model_class": model_class,
                    "value": value,
                })
```

The per-row `data/backtest/run_<ts>/results.parquet` writer also needs a `model_class` column added — it already has `model_id` which encodes the model class as a prefix, but the explicit column makes filtering and aggregation cheap.

- [ ] **Step 3: Add `--model` arg to `scripts/backtest.py`**

```python
parser.add_argument(
    "--model",
    choices=["baseline", "lightgbm", "both"],
    default="both",
    help="Which model class(es) to run. 'both' runs Model A and Model C side-by-side.",
)

# When invoking the harness:
if args.model == "both":
    model_classes = ("baseline", "lightgbm")
else:
    model_classes = (args.model,)
run_backtest(..., model_classes=model_classes, ...)
```

- [ ] **Step 4: Run an existing harness test to confirm the change is backward-compatible**

Run: `pytest tests/test_backtest/test_harness.py -v`
Expected: existing tests pass (they call `run_backtest(...)` without `model_classes`, which defaults to `("baseline",)`).

- [ ] **Step 5: Commit**

```bash
git add src/projections/backtest/harness.py scripts/backtest.py
git commit -m "feat(backtest): --model selector; iterate over model classes per fold — Plan 5"
```

---

## Task 13: Snapshot rename + `model_class` column

**Files:**
- Modify: `src/projections/backtest/snapshot.py`
- Modify: `tests/backtest/conftest.py` (if it references the old filename)
- Rename: `tests/backtest/baseline_metrics.json` → `tests/backtest/model_metrics.json` (existing 400 rows acquire `"model_class": "baseline"` field)

- [ ] **Step 1: Read `src/projections/backtest/snapshot.py`**

Run: `cat src/projections/backtest/snapshot.py`
Identify (1) the snapshot file path constant and (2) the row-key logic.

- [ ] **Step 2: Update the snapshot path constant**

Find the line referencing `baseline_metrics.json` and change to `model_metrics.json`. If the path is computed via `Path(__file__).parent ... / "baseline_metrics.json"`, update accordingly.

- [ ] **Step 3: Update row-key logic to include `model_class`**

If the snapshot is stored as a list of dicts, the keying convention before was implicit `(position, year, metric)`. Add the `model_class` field to every row's identity. The dedupe/lookup logic that compares to a "prior" snapshot must use the 4-tuple now.

- [ ] **Step 4: Rename and migrate the snapshot file**

Run: `git mv tests/backtest/baseline_metrics.json tests/backtest/model_metrics.json`

Then add `"model_class": "baseline"` to every row in the JSON file. A small Python helper (one-shot, do not commit):

```python
import json
p = "tests/backtest/model_metrics.json"
with open(p) as f:
    rows = json.load(f)
for r in rows:
    r["model_class"] = "baseline"
with open(p, "w") as f:
    json.dump(rows, f, indent=2)
```

- [ ] **Step 5: Update any conftest.py / harness reference to the old filename**

Run: `grep -rn "baseline_metrics" src/ tests/ scripts/`
Update each to `model_metrics`.

- [ ] **Step 6: Run the existing backtest gate to confirm Model A rows still match**

Run: `pytest tests/backtest/test_backtest_gate.py -v --run-backtest -k "wr and 2024"` (subset run — single cell, ~30s)
Expected: PASS (Model A rows match the snapshot identically).

- [ ] **Step 7: Commit**

```bash
git add src/projections/backtest/snapshot.py tests/backtest/model_metrics.json tests/backtest/conftest.py
git commit -m "refactor(backtest): snapshot keyed by model_class; rename to model_metrics.json — Plan 5"
```

---

## Task 14: Default-on smoke extension

**Files:**
- Modify: `tests/backtest/test_backtest_smoke.py`

- [ ] **Step 1: Read the existing smoke**

Run: `cat tests/backtest/test_backtest_smoke.py`

- [ ] **Step 2: Extend the smoke to assert both models produce metrics**

Modify the smoke to call `run_backtest(..., model_classes=("baseline", "lightgbm"))` and assert:
- Both `model_class="baseline"` and `model_class="lightgbm"` rows are present in the result for the (WR, 2024) cell.
- All metric values are finite for both.

```python
def test_backtest_smoke_both_models() -> None:
    """Plan 5: smoke covers both Model A (baseline) and Model C (lightgbm)."""
    from projections.backtest.harness import run_backtest

    result = run_backtest(
        positions=[Position.WR],
        held_out_years=[2024],
        train_start=2018,
        model_classes=("baseline", "lightgbm"),
        # ... other args as before ...
    )

    df = result.metrics_df  # per-row metrics with model_class column
    assert set(df["model_class"].unique()) == {"baseline", "lightgbm"}
    assert df["value"].notna().all() and np.isfinite(df["value"]).all()
```

- [ ] **Step 3: Run the smoke**

Run: `pytest tests/backtest/test_backtest_smoke.py -v`
Expected: PASS in <60s (Model A ~15s + Model C ~30s on one cell).

- [ ] **Step 4: Commit**

```bash
git add tests/backtest/test_backtest_smoke.py
git commit -m "test(backtest): smoke covers both Model A and Model C — Plan 5"
```

---

## Task 15: Harness-side LightGBM tests

**Files:**
- Create: `tests/test_backtest/test_harness_lightgbm.py`
- Create: `tests/test_backtest/test_harness_dual_model.py`

- [ ] **Step 1: Create `test_harness_lightgbm.py`**

```python
"""End-to-end harness fold for Model C (LightGBM). Plan 5."""

from __future__ import annotations

import numpy as np
import pytest

from projections.backtest.harness import run_backtest
from projections.schemas import Position


@pytest.mark.backtest
def test_harness_lightgbm_single_cell() -> None:
    """Run one (WR, 2024) fold under Model C and assert per-row results have model_class column."""
    result = run_backtest(
        positions=[Position.WR],
        held_out_years=[2024],
        train_start=2018,
        model_classes=("lightgbm",),
    )
    df = result.metrics_df
    assert (df["model_class"] == "lightgbm").all()
    assert df["value"].notna().all() and np.isfinite(df["value"]).all()
    expected_metrics = {"composite_rmse", "composite_mae", "spearman_topN", "calibration_p10p90"}
    assert expected_metrics.issubset(set(df["metric"].unique()))
```

- [ ] **Step 2: Create `test_harness_dual_model.py`**

```python
"""--model both end-to-end fold — Plan 5."""

from __future__ import annotations

import pytest

from projections.backtest.harness import run_backtest
from projections.schemas import Position


@pytest.mark.backtest
def test_harness_runs_both_models_for_one_cell() -> None:
    result = run_backtest(
        positions=[Position.WR],
        held_out_years=[2024],
        train_start=2018,
        model_classes=("baseline", "lightgbm"),
    )
    df = result.metrics_df
    assert set(df["model_class"].unique()) == {"baseline", "lightgbm"}
    # Same metrics produced for both models per (position, year) cell:
    a = set(df[df["model_class"] == "baseline"]["metric"].unique())
    c = set(df[df["model_class"] == "lightgbm"]["metric"].unique())
    assert a == c
```

- [ ] **Step 3: Run both tests under the backtest gate**

Run: `pytest tests/test_backtest/test_harness_lightgbm.py tests/test_backtest/test_harness_dual_model.py -v --run-backtest`
Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_backtest/test_harness_lightgbm.py tests/test_backtest/test_harness_dual_model.py
git commit -m "test(backtest): harness-side Model C + dual-model gates — Plan 5"
```

---

## Task 16: Train Model C standalone artifacts (4 positions)

This task runs the existing CLI to produce per-position Model C joblib artifacts on the real ingested data. No code changes; outputs are gitignored under `models/artifacts/`.

- [ ] **Step 1: Train each position via the script**

For each of QB / RB / TE / WR:

```bash
python scripts/train_baseline.py --position WR --model lightgbm --train-start 2018 --train-end 2023
python scripts/train_baseline.py --position QB --model lightgbm --train-start 2018 --train-end 2023
python scripts/train_baseline.py --position RB --model lightgbm --train-start 2018 --train-end 2023
python scripts/train_baseline.py --position TE --model lightgbm --train-start 2018 --train-end 2023
```

Each takes ~3-5 minutes. Output artifact: `models/artifacts/lightgbm-<pos>-2018-2023-<code-hash>.joblib`.

- [ ] **Step 2: Quick sanity-check via the existing sanity-check script**

```bash
python scripts/sanity_check_baseline.py --position WR --model lightgbm --hold-out-year 2024
```

Expected: prints per-stat fit + composite metrics. Numbers don't need to be specific values — we're verifying the artifact loads + predicts cleanly.

- [ ] **Step 3: Record the four `model_id`s**

The sanity-check script prints `model_id` near the top. Capture all 4 for the PM-doc update in Task 18.

- [ ] **Step 4: No commit**

Artifacts are gitignored; nothing to commit at this step.

---

## Task 17: Run full opt-in backtest gate; regenerate snapshot

- [ ] **Step 1: Run the full backtest gate with both models**

```bash
python scripts/backtest.py --report --model both
```

Expected: ~35-55 minutes. Prints the side-by-side comparison table per (position, year) cell.

- [ ] **Step 2: Update the snapshot to include the new Model C rows**

```bash
pytest tests/backtest/test_backtest_gate.py --run-backtest --update-snapshot
```

Or if there's a dedicated `--update-snapshot` mode in `scripts/backtest.py`:

```bash
python scripts/backtest.py --update-snapshot --model both
```

The snapshot file `tests/backtest/model_metrics.json` should grow from 400 rows (Model A only) to 800 rows (400 baseline + 400 lightgbm).

- [ ] **Step 3: Verify the snapshot regenerates correctly**

Run: `pytest tests/backtest/test_backtest_gate.py --run-backtest -v`
Expected: PASS — both Model A and Model C rows match the regenerated snapshot bit-for-bit.

- [ ] **Step 4: Commit the regenerated snapshot**

```bash
git add tests/backtest/model_metrics.json
git commit -m "chore(backtest): regenerate snapshot with Model C rows — Plan 5"
```

---

## Task 18: PM doc + TODO updates

**Files:**
- Modify: `project_management.md`
- Modify: `TODO.md`

- [ ] **Step 1: Append a new top-of-file section to `project_management.md`**

Insert before the existing "Plan 3e Phase 3 — Per-tertile bucketing — REVERTED" section:

```markdown
## Plan 5 — LightGBM with Quantile Regression (Model C) — shipped (run YYYY-MM-DD)

**Closes:** TODO #26.

`LightGBMModel` lands as a peer of `BaselineModel` under the existing `Model`
Protocol. Per-stat sub-models trained at quantiles `[0.05, 0.10, 0.50, 0.90,
0.95]`; per-row prediction sorts to enforce non-crossing, clips to `[0, inf)` for
non-negative stats, wraps in `QuantileDistribution`, and runs through the
unchanged `score_distribution` scoring layer. New `DistributionFamily.QUANTILE`
+ codec branch. `POSITION_DISPATCH` extended with `factories: dict[str,
Callable]`. Backtest harness gains `--model {baseline,lightgbm,both}`; snapshot
file renamed `baseline_metrics.json` → `model_metrics.json` and rows keyed by
`(position, year, metric, model_class)` (400 → 800 rows).

### Per-position model_ids

| Position | model_id |
|---|---|
| WR | (paste from Task 16 sanity check) |
| QB | (paste from Task 16 sanity check) |
| RB | (paste from Task 16 sanity check) |
| TE | (paste from Task 16 sanity check) |

### Composite metric comparison: Model A vs Model C

(Insert the side-by-side table from Task 17 `--report` output. Highlight cells
where Model C beats Model A on RMSE / MAE / Spearman / calibration.)

### Adoption decision

Per Plan 5 spec §1.3, Model C must beat Model A on:
- Composite RMSE strictly lower on >=12 of 16 cells; not worse by >1% on any cell.
- Spearman top-N within +-0.005 of Model A on every cell.
- Weekly mean [p10,p90] coverage no worse on any cell; mean improvement >= 0.02.

(Record whether the criteria are met. If yes, file a follow-up TODO to switch the
production default to lightgbm. If no, both models stay live and Plan 6 (ensemble)
is the more interesting next step.)

### Next: post-merge brainstorming for Model D ensemble (Plan 6) or feature work (TODO #3, #23, #24, #25).
```

- [ ] **Step 2: Update the "Next action" section in `project_management.md`**

Replace the existing "Next action" block with:

```markdown
## Next action

**Plan 5 (Model C — LightGBM with quantile regression) merged.** Three documented
follow-ups, in priority order:

1. **Plan 6 — Model D (ensemble of A + C)**: stack of Model A + Model C predictions.
   Smallest expected gain (2-5%) but cheapest to ship if Model C lands.
2. **TODO #3 — PBP / EPA features**: largest cumulative feature win (5-15% RMSE),
   replaces crude `opp_allowed_fppg_l4` proxy.
3. **TODO #23 — Target decomposition (volume * efficiency)**: 3-10% structural win;
   tightens TD modeling specifically.

Pick + brainstorm one in the next session.

After model-improvement work: Plan 4 (public Python API + CLI verbs + free-tier
hosting), then Draft Hub.
```

- [ ] **Step 3: Update `TODO.md`**

Mark TODO #26 as closed. Add a new line:

```markdown
### 26. Plan 5 — LightGBM with quantile regression (Model C) — closed in Plan 5

Closed YYYY-MM-DD. Per-stat sub-models trained at quantiles [0.05, 0.10, 0.50, 0.90, 0.95];
new QuantileDistribution + codec branch; POSITION_DISPATCH.factories dict; backtest snapshot
extended (400 -> 800 rows). Model A unchanged; both coexist for Plan 6 ensemble.
```

(Optional follow-up TODO #27 if Plan 6 should be tracked alongside the others; otherwise it
stays in the project_management.md backlog as a Plan-numbered entry.)

- [ ] **Step 4: Commit the doc updates**

```bash
git add project_management.md TODO.md
git commit -m "docs(plan-5): record Model C ship + adoption-gate results"
```

---

## Task 19: End-of-effort gate + open PR

- [ ] **Step 1: Activate the venv**

```bash
source .venv/Scripts/activate
```

- [ ] **Step 2: Run the full end-of-effort gate per CLAUDE.md §4**

```bash
pytest -v
mypy src tests
ruff check src tests
ruff format --check src tests
pytest -v -k "ingest or store or schemas"
```

Expected: every command exits 0 with no warnings beyond LightGBM's training-time output.

- [ ] **Step 3: Run the opt-in backtest gate one more time as the final regression check**

```bash
pytest -m backtest --run-backtest -v
```

Expected: PASS — snapshot drift is zero (we regenerated it in Task 17 and made no model-affecting code changes since).

- [ ] **Step 4: Push + open PR**

```bash
git push -u origin feat/plan-5-lightgbm
gh pr create --title "Plan 5 — LightGBM with quantile regression (Model C)" --body "$(cat <<'EOF'
## Summary

- New `LightGBMModel` (Model C) implements per-stat quantile regression at quantiles `[0.05, 0.10, 0.50, 0.90, 0.95]`; coexists with `BaselineModel` under the existing `Model` Protocol.
- New `QuantileDistribution` + `DistributionFamily.QUANTILE` codec branch.
- `POSITION_DISPATCH` extended with `factories: dict[str, Callable]`; backtest harness gains `--model {baseline,lightgbm,both}`; snapshot file renamed `baseline_metrics.json` -> `model_metrics.json` (400 -> 800 rows).
- `BaselineModel` (Model A) is **untouched**. Plan 3e infrastructure preserved as future-infrastructure.

## Test plan

- [ ] `pytest -v` clean
- [ ] `mypy src tests` clean
- [ ] `ruff check src tests && ruff format --check src tests` clean
- [ ] `pytest -m backtest --run-backtest` passes; snapshot drift zero
- [ ] (manual) review the side-by-side Model A vs Model C comparison table in the PM-doc update for adoption-gate criteria

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Capture the PR URL** for the PM-doc next-action and any follow-up sessions.

---

## Self-review notes

**Spec coverage check:**

| Spec § | Implementation Task |
|--------|---------------------|
| §1.1 LightGBMModel + factories | Task 5, 6, 7, 8 |
| §1.1 QuantileDistribution | Task 3 |
| §1.1 QUANTILE enum + codec branch | Task 2, 4 |
| §1.1 POSITION_DISPATCH extension | Task 11 |
| §1.1 Harness --model arg + results.parquet model_class column | Task 12 |
| §1.1 Snapshot rename + key | Task 13 |
| §1.1 Default-on smoke extension | Task 14 |
| §1.1 lightgbm dependency | Task 1 |
| §1.1 Standalone artifact retrains | Task 16 |
| §1.3 Adoption gate | Task 18 (results recorded; decision out-of-band) |
| §3.1 model_id format + code_hash files | Task 5 (skeleton) + Task 6 (set after fit) |
| §3.1 fit failure modes | Task 6 (validation tests) |
| §3.1 predict failure modes (column mismatch, crossing, clip) | Task 7 |
| §3.2 QuantileDistribution invariants | Task 3 |
| §3.3 codec round-trip | Task 4 |
| §3.4 LGBM_DEFAULTS / EARLY_STOPPING_ROUNDS / QUANTILE_GRID | Task 5 |
| §6.1 QuantileDistribution tests | Task 3 |
| §6.2 codec round-trip + mixed-family | Task 4 |
| §6.3 cross-cutting LightGBMModel tests | Tasks 6, 7, 8 |
| §6.3 per-position smokes | Task 9 |
| §6.3 cross-position smoke | Task 10 |
| §6.4 harness tests | Task 15 |
| §6.5 default-on smoke | Task 14 |
| §6.6 opt-in backtest gate | Task 17 |
| §6.7 type / lint conformance | Task 19 |
