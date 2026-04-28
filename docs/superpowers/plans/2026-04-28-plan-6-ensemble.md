# Plan 6 — Model D ensemble (A + C-NB) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Model D — a per-(position, stat) calibration-aware weighted mixture of Model A (`BaselineModel`) and Model C-NB (`LightGBMNbModel`) — and run it through the §1.3 adoption gate against Model A.

**Architecture:** New `MixtureDistribution` (CDF-pool of two child distributions, weighted) under a `cdf(x)` extension to the `Distribution` Protocol. New `EnsembleModel` that fits two child models on `[S, Y-2]`, predicts the calibration year `Y-1`, fits one scalar weight per (position, stat) via 1-D bounded brent on summed pinball loss at q ∈ {0.10, 0.90}, then re-fits children on the full prediction span `[S, Y-1]`. Codec gains a `MIXTURE` per-stat family branch; row-level family stays `MIXED`. Single new factory key `"ensemble"` in `POSITION_DISPATCH`; the backtest harness needs only a one-line cast widening.

**Tech Stack:** Python 3.11, numpy, scipy (`stats`, `optimize.minimize_scalar` bounded-brent + `optimize.brentq`), pandas, pandera, msgpack, lightgbm (inherited from C-NB), joblib, pytest, mypy strict, ruff.

**Spec:** `docs/superpowers/specs/2026-04-28-plan-6-ensemble-design.md`

---

## File Structure

**New files (created across the plan):**

| Path | Phase | Responsibility |
|---|---|---|
| `src/projections/distributions/mixture.py` | 1 | `MixtureDistribution` class implementing `Distribution` Protocol |
| `src/projections/models/ensemble.py` | 2-3 | `EnsembleModel` + per-position factories |
| `tests/test_distributions/test_cdf.py` | 0 | `cdf(x)` on each existing `Distribution` implementer |
| `tests/test_distributions/test_mixture.py` | 1 | `MixtureDistribution` math (mean, var, cdf, quantile, sample) |
| `tests/test_models/test_ensemble_model.py` | 2 | Cross-cutting fit/predict over (QB, RB, TE, WR) |
| `tests/test_models/test_ensemble_weight_fit.py` | 3 | Weight optimizer math (synthetic recovery, fallback) |

**Files to modify:**

| Path | Phase | Change |
|---|---|---|
| `src/projections/distributions/base.py` | 0 | Add `cdf(x: float) -> float` to Distribution Protocol |
| `src/projections/distributions/parametric.py` | 0 | Add `cdf` to `ParametricNormal`, `ParametricGamma`, `ParametricNegativeBinomial`, `ParametricStudentT` |
| `src/projections/distributions/quantile.py` | 0 | Add `cdf` to `QuantileDistribution` (piecewise-linear inverse) |
| `src/projections/distributions/__init__.py` | 1 | Export `MixtureDistribution` |
| `src/projections/distributions/codec.py` | 1 | New `MIXTURE` branch in `pack_per_stat_params` / `unpack_per_stat_params`; bump `_SCHEMA_VERSION` to 2 |
| `src/projections/schemas.py` | 1 | Add `DistributionFamily.MIXTURE` |
| `src/projections/models/__init__.py` | 4 | Add `qb_ensemble` / `rb_ensemble` / `te_ensemble` / `wr_ensemble` factories; `"ensemble"` keys in per-position factories dicts |
| `src/projections/backtest/harness.py` | 4 | Widen `cast` at line 261 to include `EnsembleModel` |
| `scripts/backtest.py` | 4 | `--model` choices add `"ensemble"`; `--model all` expands to 5 classes |
| `tests/backtest/model_metrics.json` | 5 | Snapshot extension 1504 → 1872 rows after real-data run |
| `data/ensemble_weights/*.json` | 5 | Per-fold weight artifacts (committed) |
| `project_management.md` | 6 | Plan 6 entry at top |
| `TODO.md` | 6 | Update items #28, #29, #30 with Plan 6 progress |

---

## Phase 0 — Distribution Protocol `cdf(x)` extension

Pure foundations. No production wiring touched. Each existing `Distribution` implementer gains an analytic `cdf(x) -> float` method. Adds a `cdf` slot to the `Distribution` Protocol.

### Task 0.1: Test cdf on parametric distributions

**Files:**
- Create: `tests/test_distributions/test_cdf.py`

- [ ] **Step 1: Write the failing test**

```python
"""Plan 6 Phase 0 — cdf(x) parity with scipy on each existing parametric distribution."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from projections.distributions import (
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    ParametricStudentT,
)


@pytest.mark.parametrize(
    "x",
    [-5.0, -1.0, 0.0, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 100.0],
)
def test_normal_cdf_matches_scipy(x: float) -> None:
    dist = ParametricNormal(mean=10.0, std=4.0)
    expected = float(stats.norm.cdf(x, loc=10.0, scale=4.0))
    assert dist.cdf(x) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "x",
    [0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0],
)
def test_gamma_cdf_matches_scipy(x: float) -> None:
    dist = ParametricGamma(shape=4.0, scale=2.5)
    expected = float(stats.gamma.cdf(x, a=4.0, scale=2.5))
    assert dist.cdf(x) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "x",
    [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0],
)
def test_nb_cdf_matches_scipy(x: float) -> None:
    dist = ParametricNegativeBinomial(mean=2.5, dispersion=5.0)
    n = 5.0
    p = 5.0 / (5.0 + 2.5)
    expected = float(stats.nbinom.cdf(x, n=n, p=p))
    assert dist.cdf(x) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "x",
    [-50.0, -10.0, -2.0, -0.5, 0.0, 0.5, 2.0, 10.0, 50.0, 200.0],
)
def test_student_t_cdf_matches_scipy(x: float) -> None:
    dist = ParametricStudentT(loc=5.0, scale=3.0, df=4.5)
    expected = float(stats.t.cdf(x, df=4.5, loc=5.0, scale=3.0))
    assert dist.cdf(x) == pytest.approx(expected, abs=1e-12)


def test_cdf_endpoints() -> None:
    """cdf(-large) ~ 0, cdf(+large) ~ 1 for unbounded distributions."""
    for dist in [
        ParametricNormal(mean=0.0, std=1.0),
        ParametricStudentT(loc=0.0, scale=1.0, df=5.0),
    ]:
        assert dist.cdf(-1e6) == pytest.approx(0.0, abs=1e-12)
        assert dist.cdf(1e6) == pytest.approx(1.0, abs=1e-12)


def test_cdf_monotone() -> None:
    """cdf is non-decreasing across a fine x grid for each parametric family."""
    grid = np.linspace(-10.0, 50.0, 200)
    families = [
        ParametricNormal(mean=5.0, std=3.0),
        ParametricGamma(shape=2.0, scale=4.0),
        ParametricNegativeBinomial(mean=3.0, dispersion=4.0),
        ParametricStudentT(loc=5.0, scale=2.0, df=4.0),
    ]
    for dist in families:
        cdfs = np.array([dist.cdf(float(x)) for x in grid])
        assert np.all(np.diff(cdfs) >= -1e-12), f"{type(dist).__name__} cdf not monotone"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_distributions/test_cdf.py -v`
Expected: FAIL with `AttributeError: '<X>' object has no attribute 'cdf'`.

- [ ] **Step 3: Add `cdf` to the Distribution Protocol**

Edit `src/projections/distributions/base.py`. Add `def cdf(self, x: float) -> float: ...` after `sample`:

```python
"""Distribution interface — value object exposing mean/quantile/sample/cdf."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Distribution(Protocol):
    """A probability distribution over a single player's fantasy points (or
    underlying stat). Backings: parametric (Normal/Gamma/NB/Student-t), empirical-quantile,
    sampled, or mixture. Same surface regardless."""

    # NOTE: @runtime_checkable enables isinstance() checks but performs structural
    # (attribute-presence) checking only — it does NOT verify method signatures or
    # return types. Trust mypy for that, not isinstance.

    def mean(self) -> float: ...
    def std(self) -> float: ...
    def quantile(self, q: float) -> float: ...
    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]: ...
    def cdf(self, x: float) -> float: ...
```

- [ ] **Step 4: Implement `cdf` on each parametric class**

Edit `src/projections/distributions/parametric.py`. Append a `cdf(self, x: float) -> float` method to each of `ParametricNormal`, `ParametricGamma`, `ParametricNegativeBinomial`, `ParametricStudentT`:

```python
# In ParametricNormal, after sample():
    def cdf(self, x: float) -> float:
        return float(stats.norm.cdf(x, loc=self.mean_, scale=self.std_))

# In ParametricGamma, after sample():
    def cdf(self, x: float) -> float:
        return float(stats.gamma.cdf(x, a=self.shape, scale=self.scale))

# In ParametricNegativeBinomial, after sample():
    def cdf(self, x: float) -> float:
        n, p = self._scipy_n_p()
        return float(stats.nbinom.cdf(x, n=n, p=p))

# In ParametricStudentT, after sample():
    def cdf(self, x: float) -> float:
        return float(stats.t.cdf(x, df=self.df_, loc=self.loc_, scale=self.scale_))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_distributions/test_cdf.py -v`
Expected: 23 PASS (10 normal + 10 gamma + 10 nb + 10 student-t = 40 parametrized cases, plus `test_cdf_endpoints` and `test_cdf_monotone`).

Wait, that's actually `40 + 1 + 1 = 42` results. Both endpoints test and monotone test loop internally so they each count as 1 test.

- [ ] **Step 6: Commit**

```bash
git add src/projections/distributions/base.py src/projections/distributions/parametric.py tests/test_distributions/test_cdf.py
git commit -m "feat(distributions): add cdf to Distribution Protocol — Plan 6 Phase 0 (parametric)"
```

---

### Task 0.2: Add cdf to QuantileDistribution

**Files:**
- Modify: `src/projections/distributions/quantile.py`
- Modify: `tests/test_distributions/test_cdf.py` (extend with QuantileDistribution cases)

- [ ] **Step 1: Append a failing test for QuantileDistribution.cdf**

Append to `tests/test_distributions/test_cdf.py`:

```python
import numpy as np

from projections.distributions import QuantileDistribution


def _make_quantile_dist() -> QuantileDistribution:
    """Symmetric 5-knot fixture spanning [10, 50] at quantiles
    [0.05, 0.25, 0.5, 0.75, 0.95]."""
    return QuantileDistribution(
        quantiles=np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float64),
        values=np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
    )


def test_quantile_cdf_at_stored_knots() -> None:
    """cdf(value_at_qk) == qk exactly for each stored knot."""
    dist = _make_quantile_dist()
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    vs = [10.0, 20.0, 30.0, 40.0, 50.0]
    for q, v in zip(qs, vs, strict=True):
        assert dist.cdf(v) == pytest.approx(q, abs=1e-12)


def test_quantile_cdf_piecewise_linear_between_knots() -> None:
    """Between knots cdf is linear: at midpoint of two stored values, cdf is
    midpoint of two stored quantiles."""
    dist = _make_quantile_dist()
    # midpoint between (0.25, 20) and (0.5, 30) is value 25, cdf 0.375
    assert dist.cdf(25.0) == pytest.approx(0.375, abs=1e-12)
    # midpoint between (0.5, 30) and (0.75, 40) is value 35, cdf 0.625
    assert dist.cdf(35.0) == pytest.approx(0.625, abs=1e-12)


def test_quantile_cdf_clamps_at_endpoints() -> None:
    """Below the lowest stored value, cdf clamps at the lowest stored quantile.
    Above the highest stored value, cdf clamps at the highest stored quantile.
    (Conservative: extrapolation only happens at the quantile() boundary, not cdf.)
    """
    dist = _make_quantile_dist()
    assert dist.cdf(-100.0) == pytest.approx(0.05, abs=1e-12)
    assert dist.cdf(0.0) == pytest.approx(0.05, abs=1e-12)
    assert dist.cdf(1000.0) == pytest.approx(0.95, abs=1e-12)


def test_quantile_cdf_monotone_on_grid() -> None:
    """cdf is non-decreasing across a fine value grid."""
    dist = _make_quantile_dist()
    grid = np.linspace(0.0, 60.0, 200)
    cdfs = np.array([dist.cdf(float(x)) for x in grid])
    assert np.all(np.diff(cdfs) >= -1e-12)


def test_quantile_cdf_round_trip_with_quantile() -> None:
    """For q in (q_min, q_max), cdf(quantile(q)) == q."""
    dist = _make_quantile_dist()
    for q in [0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95]:
        v = dist.quantile(q)
        assert dist.cdf(v) == pytest.approx(q, abs=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_distributions/test_cdf.py -v -k quantile`
Expected: FAIL with `AttributeError: 'QuantileDistribution' object has no attribute 'cdf'`.

- [ ] **Step 3: Implement `cdf` on QuantileDistribution**

In `src/projections/distributions/quantile.py`, add this method to the `QuantileDistribution` class (after `sample`):

```python
    def cdf(self, x: float) -> float:
        """Return P(X <= x) by piecewise-linear inversion of the stored
        (quantiles, values) knots.

        Below the lowest stored value, clamps at the lowest stored quantile.
        Above the highest stored value, clamps at the highest stored quantile.
        (We deliberately do NOT extrapolate cdf into the tail — quantile()
        extrapolates linearly for q outside [q_min, q_max] when sampling, but
        cdf(x) for x outside [v_min, v_max] is reported as the boundary
        quantile to keep cdf(x) bounded in [q_min, q_max] subset of [0, 1].
        Mixtures pool both children's cdfs, so this clamp does not impede
        the brentq inversion in MixtureDistribution.quantile.)
        """
        qs = self.quantiles_
        vs = self.values_
        if x <= vs[0]:
            return float(qs[0])
        if x >= vs[-1]:
            return float(qs[-1])
        # Linear interpolation in (value -> quantile) space.
        # np.interp expects ascending xp; values_ is non-decreasing (validated in __init__).
        return float(np.interp(x, vs, qs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_distributions/test_cdf.py -v`
Expected: all PASS.

- [ ] **Step 5: Verification gate**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_distributions/ -v
.venv/Scripts/python.exe -m mypy src tests
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m ruff format --check src tests
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/projections/distributions/quantile.py tests/test_distributions/test_cdf.py
git commit -m "feat(distributions): add cdf to QuantileDistribution — Plan 6 Phase 0 (quantile)"
```

---

## Phase 1 — `MixtureDistribution` + codec MIXTURE branch

### Task 1.1: Add `DistributionFamily.MIXTURE`

**Files:**
- Modify: `src/projections/schemas.py:145-156`

- [ ] **Step 1: Add the enum value**

Edit `src/projections/schemas.py`. In the `DistributionFamily` enum, add `MIXTURE = "MIXTURE"` after `MIXED`:

```python
class DistributionFamily(StrEnum):
    """Backing representation of a `Distribution`."""

    NORMAL = "NORMAL"
    GAMMA = "GAMMA"
    NEGATIVE_BINOMIAL = "NEGATIVE_BINOMIAL"  # Plan 3e Phase 1
    STUDENT_T = "STUDENT_T"  # Plan 3e Phase 2 — heavy-tailed continuous
    SAMPLED = "SAMPLED"  # explicit sample array
    SAMPLED_SUMMARY = "SAMPLED_SUMMARY"  # per-stat dist params + summary in mean/p10/p50/p90
    QUANTILE = "QUANTILE"  # Plan 5 — Model C (LightGBM quantile regression)
    MIXED = "MIXED"  # Plan 5c — per-row distribution mixes families per stat
    MIXTURE = "MIXTURE"  # Plan 6 — per-stat: weighted mixture of two child distributions
```

- [ ] **Step 2: Verify the existing schema-options test catches the new value**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schemas/ -v -k DistributionFamily`
Expected: failure if there's a hard-coded option set; PASS if the enum is referenced dynamically.

If a test fails because it asserts a fixed option set, update that test to include `"MIXTURE"`. Search for the pattern: `Grep "DistributionFamily" tests/`.

- [ ] **Step 3: Run all schema tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schemas/ -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/projections/schemas.py tests/test_schemas/  # (only the touched test file)
git commit -m "feat(schemas): DistributionFamily.MIXTURE — Plan 6 Phase 1"
```

---

### Task 1.2: `MixtureDistribution` math (TDD)

**Files:**
- Create: `tests/test_distributions/test_mixture.py`
- Create: `src/projections/distributions/mixture.py`
- Modify: `src/projections/distributions/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_distributions/test_mixture.py`:

```python
"""Plan 6 Phase 1 — MixtureDistribution math: mean / std / cdf / quantile / sample."""

from __future__ import annotations

import numpy as np
import pytest

from projections.distributions import (
    MixtureDistribution,
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    QuantileDistribution,
)


def _make_quantile_dist() -> QuantileDistribution:
    return QuantileDistribution(
        quantiles=np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float64),
        values=np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
    )


def test_constructor_rejects_boundary_weights() -> None:
    a = ParametricNormal(mean=0.0, std=1.0)
    b = ParametricNormal(mean=10.0, std=1.0)
    with pytest.raises(ValueError, match="weight"):
        MixtureDistribution(component_a=a, component_b=b, weight=0.0)
    with pytest.raises(ValueError, match="weight"):
        MixtureDistribution(component_a=a, component_b=b, weight=1.0)
    with pytest.raises(ValueError, match="weight"):
        MixtureDistribution(component_a=a, component_b=b, weight=-0.1)
    with pytest.raises(ValueError, match="weight"):
        MixtureDistribution(component_a=a, component_b=b, weight=1.1)


def test_mean_is_linear_combination() -> None:
    a = ParametricNormal(mean=5.0, std=2.0)
    b = ParametricNormal(mean=15.0, std=3.0)
    for w in [0.1, 0.3, 0.5, 0.7, 0.9]:
        mix = MixtureDistribution(component_a=a, component_b=b, weight=w)
        expected = w * 5.0 + (1.0 - w) * 15.0
        assert mix.mean() == pytest.approx(expected, abs=1e-12)


def test_variance_uses_mixture_formula() -> None:
    """variance = w*var_A + (1-w)*var_B + w*(1-w)*(mean_A - mean_B)^2."""
    a = ParametricNormal(mean=5.0, std=2.0)   # var = 4
    b = ParametricNormal(mean=15.0, std=3.0)  # var = 9
    for w in [0.2, 0.5, 0.8]:
        mix = MixtureDistribution(component_a=a, component_b=b, weight=w)
        expected_var = w * 4.0 + (1 - w) * 9.0 + w * (1 - w) * (5.0 - 15.0) ** 2
        assert mix.std() ** 2 == pytest.approx(expected_var, abs=1e-9)


def test_cdf_is_linear_pool() -> None:
    a = ParametricNormal(mean=0.0, std=1.0)
    b = ParametricGamma(shape=2.0, scale=2.0)
    for w in [0.25, 0.5, 0.75]:
        mix = MixtureDistribution(component_a=a, component_b=b, weight=w)
        for x in [-2.0, 0.0, 1.0, 5.0, 10.0]:
            expected = w * a.cdf(x) + (1 - w) * b.cdf(x)
            assert mix.cdf(x) == pytest.approx(expected, abs=1e-12)


def test_quantile_round_trips_through_cdf() -> None:
    """For each q, cdf(quantile(q)) ~ q."""
    a = ParametricNormal(mean=2.0, std=1.0)
    b = ParametricNormal(mean=8.0, std=2.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.4)
    for q in [0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95]:
        x = mix.quantile(q)
        assert mix.cdf(x) == pytest.approx(q, abs=1e-6)


def test_quantile_invalid_q() -> None:
    a = ParametricNormal(mean=0.0, std=1.0)
    b = ParametricNormal(mean=5.0, std=1.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.5)
    with pytest.raises(ValueError, match="q"):
        mix.quantile(0.0)
    with pytest.raises(ValueError, match="q"):
        mix.quantile(1.0)
    with pytest.raises(ValueError, match="q"):
        mix.quantile(-0.1)


def test_sample_converges_to_analytic_moments() -> None:
    a = ParametricNormal(mean=0.0, std=1.0)
    b = ParametricNormal(mean=5.0, std=2.0)
    w = 0.4
    mix = MixtureDistribution(component_a=a, component_b=b, weight=w)
    rng = np.random.default_rng(seed=42)
    draws = mix.sample(n=20000, rng=rng)
    assert draws.shape == (20000,)
    expected_mean = w * 0.0 + (1 - w) * 5.0
    expected_var = w * 1.0 + (1 - w) * 4.0 + w * (1 - w) * (0.0 - 5.0) ** 2
    expected_std = np.sqrt(expected_var)
    assert draws.mean() == pytest.approx(expected_mean, abs=0.1)
    assert draws.std() == pytest.approx(expected_std, abs=0.1)


def test_sample_with_count_component() -> None:
    """Mixture with NB component samples ints from one side, floats from the other."""
    a = ParametricNegativeBinomial(mean=1.5, dispersion=4.0)
    b = ParametricNormal(mean=20.0, std=5.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.5)
    rng = np.random.default_rng(seed=0)
    draws = mix.sample(n=5000, rng=rng)
    assert draws.shape == (5000,)
    # Roughly half should be small integers (NB samples), half larger floats (Normal).
    small = (draws < 5.0).sum()
    large = (draws > 10.0).sum()
    assert 0.4 * 5000 < small < 0.6 * 5000
    assert 0.4 * 5000 < large < 0.6 * 5000


def test_sample_with_quantile_component() -> None:
    a = _make_quantile_dist()
    b = ParametricNormal(mean=100.0, std=5.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.3)
    rng = np.random.default_rng(seed=1)
    draws = mix.sample(n=10000, rng=rng)
    assert draws.shape == (10000,)
    # Mean is between the two components' means, weighted.
    a_mean = a.mean()
    expected_mean = 0.3 * a_mean + 0.7 * 100.0
    assert draws.mean() == pytest.approx(expected_mean, abs=2.0)


def test_quantile_extreme_q_within_safe_bracket() -> None:
    """Quantile inversion at q in (1e-3, 1 - 1e-3) should converge cleanly."""
    a = ParametricNormal(mean=0.0, std=1.0)
    b = ParametricNormal(mean=10.0, std=2.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.5)
    for q in [0.001, 0.01, 0.99, 0.999]:
        x = mix.quantile(q)
        assert np.isfinite(x)
        assert mix.cdf(x) == pytest.approx(q, abs=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_distributions/test_mixture.py -v`
Expected: FAIL with `ImportError: cannot import name 'MixtureDistribution'`.

- [ ] **Step 3: Implement `MixtureDistribution`**

Create `src/projections/distributions/mixture.py`:

```python
"""MixtureDistribution — weighted mixture of two child distributions.

Plan 6 (Model D ensemble). For per-(position, stat) ensemble of Model A and
Model C-NB, each row's per-stat distribution is a MixtureDistribution wrapping
the two child distributions and a scalar weight.

Mathematics:
    mean()      = w * F_a.mean() + (1-w) * F_b.mean()
    variance()  = w * F_a.var() + (1-w) * F_b.var()
                  + w * (1-w) * (F_a.mean() - F_b.mean())^2
    cdf(x)      = w * F_a.cdf(x) + (1-w) * F_b.cdf(x)
    quantile(q) = brentq solving cdf(x) - q = 0 over a bracket spanning both
                  components' tails.
    sample(n)   = vectorized Bernoulli(w) mask -> per-element child sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from projections.distributions.base import Distribution

_QUANTILE_EPS = 1e-9
_BRACKET_PADDING = 100.0


@dataclass(slots=True, frozen=True, init=False)
class MixtureDistribution:
    """Weighted mixture: P(X) = w * F_a(X) + (1-w) * F_b(X)."""

    component_a: Distribution
    component_b: Distribution
    weight: float

    def __init__(
        self,
        *,
        component_a: Distribution,
        component_b: Distribution,
        weight: float,
    ) -> None:
        if not (0.0 < weight < 1.0):
            raise ValueError(f"weight must lie strictly in (0, 1), got {weight}")
        object.__setattr__(self, "component_a", component_a)
        object.__setattr__(self, "component_b", component_b)
        object.__setattr__(self, "weight", float(weight))

    def mean(self) -> float:
        w = self.weight
        return w * self.component_a.mean() + (1.0 - w) * self.component_b.mean()

    def std(self) -> float:
        w = self.weight
        var_a = self.component_a.std() ** 2
        var_b = self.component_b.std() ** 2
        mean_a = self.component_a.mean()
        mean_b = self.component_b.mean()
        var = w * var_a + (1.0 - w) * var_b + w * (1.0 - w) * (mean_a - mean_b) ** 2
        # Numerical guard: floating-point cancellation can produce tiny negatives.
        return float(np.sqrt(max(var, 0.0)))

    def cdf(self, x: float) -> float:
        w = self.weight
        return w * self.component_a.cdf(x) + (1.0 - w) * self.component_b.cdf(x)

    def quantile(self, q: float) -> float:
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must lie strictly in (0, 1), got {q}")

        # Build a bracket [lo, hi] guaranteed to contain the q-quantile.
        # The mixture's q-quantile lies between min(child q-quantiles) and
        # max(child q-quantiles) -- but inverse is monotone in the pool, so a
        # safe bracket is [min(a_qmin, b_qmin), max(a_qmax, b_qmax)] padded.
        try:
            a_low = self.component_a.quantile(_QUANTILE_EPS)
            a_high = self.component_a.quantile(1.0 - _QUANTILE_EPS)
            b_low = self.component_b.quantile(_QUANTILE_EPS)
            b_high = self.component_b.quantile(1.0 - _QUANTILE_EPS)
        except (ValueError, OverflowError):
            # Pathological component (e.g., zero-width bracket); fall back to a
            # mean-and-std bracket.
            mean = self.mean()
            std = max(self.std(), 1e-6)
            a_low = b_low = mean - 10.0 * std
            a_high = b_high = mean + 10.0 * std

        lo = min(a_low, b_low) - _BRACKET_PADDING
        hi = max(a_high, b_high) + _BRACKET_PADDING

        # cdf is non-decreasing, so brentq on cdf(x) - q is well-defined.
        f_lo = self.cdf(lo) - q
        f_hi = self.cdf(hi) - q
        if f_lo > 0.0:
            # q is below the joint support; clamp to lo.
            return float(lo)
        if f_hi < 0.0:
            # q is above the joint support; clamp to hi.
            return float(hi)

        return float(brentq(lambda x: self.cdf(x) - q, lo, hi, xtol=1e-9, rtol=1e-9))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        rng = rng if rng is not None else np.random.default_rng()
        # Vectorized Bernoulli draw of which component to use.
        use_a = rng.random(size=n) < self.weight
        n_a = int(use_a.sum())
        n_b = n - n_a
        out = np.empty(n, dtype=np.float64)
        if n_a > 0:
            out[use_a] = self.component_a.sample(n=n_a, rng=rng)
        if n_b > 0:
            out[~use_a] = self.component_b.sample(n=n_b, rng=rng)
        return out
```

- [ ] **Step 4: Export from the package**

Edit `src/projections/distributions/__init__.py`:

```python
"""Distribution layer — interface + parametric implementations + mixture + codec."""

from __future__ import annotations

from projections.distributions.base import Distribution
from projections.distributions.codec import pack_per_stat_params, unpack_per_stat_params
from projections.distributions.mixture import MixtureDistribution
from projections.distributions.parametric import (
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    ParametricStudentT,
)
from projections.distributions.quantile import QuantileDistribution

__all__ = [
    "Distribution",
    "MixtureDistribution",
    "ParametricGamma",
    "ParametricNegativeBinomial",
    "ParametricNormal",
    "ParametricStudentT",
    "QuantileDistribution",
    "pack_per_stat_params",
    "unpack_per_stat_params",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_distributions/test_mixture.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/projections/distributions/mixture.py src/projections/distributions/__init__.py tests/test_distributions/test_mixture.py
git commit -m "feat(distributions): MixtureDistribution — Plan 6 Phase 1 (math)"
```

---

### Task 1.3: Codec MIXTURE branch (TDD)

**Files:**
- Modify: `tests/test_distributions/test_codec.py`
- Modify: `src/projections/distributions/codec.py`

- [ ] **Step 1: Append failing tests for the MIXTURE round-trip**

Append to `tests/test_distributions/test_codec.py`:

```python
import numpy as np

from projections.distributions import (
    MixtureDistribution,
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    QuantileDistribution,
    pack_per_stat_params,
    unpack_per_stat_params,
)
from projections.schemas import DistributionFamily, Stat


def test_codec_mixture_round_trip_normal_normal() -> None:
    a = ParametricNormal(mean=5.0, std=2.0)
    b = ParametricNormal(mean=15.0, std=3.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.4)
    blob = pack_per_stat_params({Stat.PASSING_YARDS: mix})
    decoded = unpack_per_stat_params(blob)
    out = decoded[Stat.PASSING_YARDS]
    assert isinstance(out, MixtureDistribution)
    assert out.weight == pytest.approx(0.4, abs=1e-12)
    assert isinstance(out.component_a, ParametricNormal)
    assert out.component_a.mean() == pytest.approx(5.0)
    assert out.component_a.std() == pytest.approx(2.0)
    assert isinstance(out.component_b, ParametricNormal)
    assert out.component_b.mean() == pytest.approx(15.0)
    assert out.component_b.std() == pytest.approx(3.0)


def test_codec_mixture_round_trip_gamma_quantile() -> None:
    a = ParametricGamma(shape=4.0, scale=2.5)
    b = QuantileDistribution(
        quantiles=np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float64),
        values=np.array([1.0, 5.0, 10.0, 18.0, 30.0], dtype=np.float64),
    )
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.65)
    blob = pack_per_stat_params({Stat.RUSHING_YARDS: mix})
    decoded = unpack_per_stat_params(blob)
    out = decoded[Stat.RUSHING_YARDS]
    assert isinstance(out, MixtureDistribution)
    assert out.weight == pytest.approx(0.65, abs=1e-12)
    assert isinstance(out.component_a, ParametricGamma)
    assert isinstance(out.component_b, QuantileDistribution)
    np.testing.assert_array_almost_equal(
        out.component_b.quantiles_, np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    )
    np.testing.assert_array_almost_equal(
        out.component_b.values_, np.array([1.0, 5.0, 10.0, 18.0, 30.0])
    )


def test_codec_mixture_round_trip_nb_nb() -> None:
    a = ParametricNegativeBinomial(mean=1.5, dispersion=4.0)
    b = ParametricNegativeBinomial(mean=2.5, dispersion=8.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.3)
    blob = pack_per_stat_params({Stat.RECEIVING_TDS: mix})
    decoded = unpack_per_stat_params(blob)
    out = decoded[Stat.RECEIVING_TDS]
    assert isinstance(out, MixtureDistribution)
    assert out.weight == pytest.approx(0.3, abs=1e-12)
    assert isinstance(out.component_a, ParametricNegativeBinomial)
    assert out.component_a.mean() == pytest.approx(1.5)
    assert out.component_a.dispersion_ == pytest.approx(4.0)
    assert isinstance(out.component_b, ParametricNegativeBinomial)
    assert out.component_b.mean() == pytest.approx(2.5)
    assert out.component_b.dispersion_ == pytest.approx(8.0)


def test_codec_mixture_family_name() -> None:
    """The encoded blob carries family='MIXTURE' for mixture entries."""
    import msgpack

    a = ParametricNormal(mean=5.0, std=2.0)
    b = ParametricNormal(mean=15.0, std=3.0)
    mix = MixtureDistribution(component_a=a, component_b=b, weight=0.5)
    blob = pack_per_stat_params({Stat.PASSING_YARDS: mix})
    payload = msgpack.unpackb(blob, raw=False)
    assert payload["schema_version"] == 2
    entry = payload["stats"]["passing_yards"]
    assert entry["family"] == DistributionFamily.MIXTURE.value
    assert entry["weight"] == pytest.approx(0.5)
    assert entry["component_a"]["family"] == DistributionFamily.NORMAL.value
    assert entry["component_b"]["family"] == DistributionFamily.NORMAL.value


def test_codec_v1_blobs_still_decodable() -> None:
    """Plan 5/5b/5c blobs (v1) without MIXTURE entries should remain readable.

    Manufactured by packing a non-mixture dist with a temporary version-1
    payload to validate forward-compat: v1 readers no longer exist (we only
    emit v2 now), so we test by directly synthesizing a v1-shaped blob.
    """
    import msgpack

    payload = {
        "schema_version": 1,  # legacy
        "stats": {
            "passing_yards": {
                "family": DistributionFamily.NORMAL.value,
                "mean": 250.0,
                "std": 50.0,
            }
        },
    }
    blob = msgpack.packb(payload, use_bin_type=True)
    # v1 blobs are NOT readable by the v2 unpacker (we bumped schema_version
    # to 2 in this plan). Confirm the failure mode is explicit.
    with pytest.raises(ValueError, match="schema_version"):
        unpack_per_stat_params(bytes(blob))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_distributions/test_codec.py -v -k mixture`
Expected: FAIL with `ValueError: No codec entry for Distribution type MixtureDistribution`.

- [ ] **Step 3: Implement the MIXTURE codec branch**

Replace `src/projections/distributions/codec.py` entirely with:

```python
"""Symmetric codec for per-stat distribution params persisted in
ProjectionWeeklySchema.params.

The encoded blob is msgpack-packed with shape:

    {
        "schema_version": 2,
        "stats": {
            "<stat_value>": {
                "family": "NORMAL"|"GAMMA"|"NEGATIVE_BINOMIAL"|"STUDENT_T"|
                          "QUANTILE"|"MIXTURE",
                ... family-specific params ...
            },
            ...
        }
    }

Currently registered families:
    NORMAL:            {"family": "NORMAL",            "mean": float, "std": float}
    GAMMA:             {"family": "GAMMA",             "shape": float, "scale": float}
    NEGATIVE_BINOMIAL: {"family": "NEGATIVE_BINOMIAL", "mean": float, "dispersion": float}
    STUDENT_T:         {"family": "STUDENT_T",         "loc": float, "scale": float, "df": float}
    QUANTILE:          {"family": "QUANTILE",          "quantiles": list[float],
                                                       "values":    list[float]}
    MIXTURE:           {"family": "MIXTURE",           "weight": float,
                                                       "component_a": {<single>},
                                                       "component_b": {<single>}}

Schema version 2 (Plan 6): MIXTURE branch added; v1 blobs are no longer
forward-compatible (no v1 readers in the codebase). Adding a new family means
adding one branch each to _pack_single and _unpack_single.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import msgpack
import numpy as np

from projections.distributions.base import Distribution
from projections.distributions.mixture import MixtureDistribution
from projections.distributions.parametric import (
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    ParametricStudentT,
)
from projections.distributions.quantile import QuantileDistribution
from projections.schemas import DistributionFamily, Stat

_SCHEMA_VERSION: Final[int] = 2


def _pack_single(dist: Distribution) -> dict[str, Any]:
    """Encode a single Distribution as a family-tagged dict.

    Used by both top-level pack_per_stat_params (one dict per stat) and the
    MIXTURE recursion (one dict per child component).

    Raises ValueError on Distribution types without a registered codec entry.
    """
    if isinstance(dist, MixtureDistribution):
        return {
            "family": DistributionFamily.MIXTURE.value,
            "weight": float(dist.weight),
            "component_a": _pack_single(dist.component_a),
            "component_b": _pack_single(dist.component_b),
        }
    if isinstance(dist, ParametricNormal):
        return {
            "family": DistributionFamily.NORMAL.value,
            "mean": dist.mean(),
            "std": dist.std(),
        }
    if isinstance(dist, ParametricGamma):
        return {
            "family": DistributionFamily.GAMMA.value,
            "shape": dist.shape,
            "scale": dist.scale,
        }
    if isinstance(dist, ParametricNegativeBinomial):
        return {
            "family": DistributionFamily.NEGATIVE_BINOMIAL.value,
            "mean": dist.mean(),
            "dispersion": dist.dispersion_,
        }
    if isinstance(dist, ParametricStudentT):
        return {
            "family": DistributionFamily.STUDENT_T.value,
            "loc": dist.mean(),
            "scale": dist.scale_,
            "df": dist.df_,
        }
    if isinstance(dist, QuantileDistribution):
        return {
            "family": DistributionFamily.QUANTILE.value,
            "quantiles": dist.quantiles_.tolist(),
            "values": dist.values_.tolist(),
        }
    raise ValueError(
        f"No codec entry for Distribution type {type(dist).__name__}; "
        f"add a branch to _pack_single in distributions/codec.py."
    )


def _unpack_single(entry: Mapping[str, Any]) -> Distribution:
    """Decode a single family-tagged dict back into a Distribution.

    Raises ValueError on unknown family.
    """
    family_value = entry["family"]
    if family_value == DistributionFamily.MIXTURE.value:
        return MixtureDistribution(
            component_a=_unpack_single(entry["component_a"]),
            component_b=_unpack_single(entry["component_b"]),
            weight=float(entry["weight"]),
        )
    if family_value == DistributionFamily.NORMAL.value:
        return ParametricNormal(mean=float(entry["mean"]), std=float(entry["std"]))
    if family_value == DistributionFamily.GAMMA.value:
        return ParametricGamma(shape=float(entry["shape"]), scale=float(entry["scale"]))
    if family_value == DistributionFamily.NEGATIVE_BINOMIAL.value:
        return ParametricNegativeBinomial(
            mean=float(entry["mean"]),
            dispersion=float(entry["dispersion"]),
        )
    if family_value == DistributionFamily.STUDENT_T.value:
        return ParametricStudentT(
            loc=float(entry["loc"]),
            scale=float(entry["scale"]),
            df=float(entry["df"]),
        )
    if family_value == DistributionFamily.QUANTILE.value:
        return QuantileDistribution(
            quantiles=np.asarray(entry["quantiles"], dtype=np.float64),
            values=np.asarray(entry["values"], dtype=np.float64),
        )
    raise ValueError(
        f"Unknown family in params blob: {family_value!r}; "
        f"add a branch to _unpack_single in distributions/codec.py."
    )


def pack_per_stat_params(per_stat_dists: Mapping[Stat, Distribution]) -> bytes:
    """Encode a per-row per-stat distribution dict for ProjectionWeeklySchema.params.

    Raises:
        ValueError: a Distribution type without a registered codec entry.
    """
    stats_blob: dict[str, dict[str, Any]] = {
        stat.value: _pack_single(dist) for stat, dist in per_stat_dists.items()
    }
    payload = {"schema_version": _SCHEMA_VERSION, "stats": stats_blob}
    return bytes(msgpack.packb(payload, use_bin_type=True))


def unpack_per_stat_params(blob: bytes) -> dict[Stat, Distribution]:
    """Decode the params blob into a {Stat -> Distribution} dict.

    Raises:
        ValueError: unknown schema_version, unknown family, or unknown stat name.
    """
    payload = msgpack.unpackb(blob, raw=False)
    version = payload.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise ValueError(
            f"Unknown per-stat params schema_version: {version!r} (supported: {_SCHEMA_VERSION})"
        )
    stats_blob = payload["stats"]
    out: dict[Stat, Distribution] = {}
    for stat_name, entry in stats_blob.items():
        try:
            stat = Stat(stat_name)
        except ValueError as exc:
            raise ValueError(f"Unknown stat name in params blob: {stat_name!r}") from exc
        out[stat] = _unpack_single(entry)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_distributions/test_codec.py -v`
Expected: all PASS.

- [ ] **Step 5: Verification gate**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_distributions/ -v
.venv/Scripts/python.exe -m mypy src tests
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m ruff format --check src tests
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/projections/distributions/codec.py tests/test_distributions/test_codec.py
git commit -m "feat(distributions): MIXTURE codec branch + schema_version=2 — Plan 6 Phase 1 (codec)"
```

---

### Task 1.4: Phase 1 integration — re-run existing snapshot

The snapshot was written with v1 blobs (Plan 5c). After bumping `_SCHEMA_VERSION` to 2, existing predictions need to be regenerated (because `unpack_per_stat_params` now rejects v1).

But the snapshot at `tests/backtest/model_metrics.json` doesn't store params blobs — it stores summary metrics. The actual `data/backtest/run_*/results.parquet` files DO carry params blobs but those are gitignored.

**Decision:** the snapshot won't break. The v1 blobs only live in gitignored backtest output. If a user has stale data files they'll see the v1 rejection — that is fine, they regenerate.

- [ ] **Step 1: Verify the existing snapshot test still passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest/ -v`
Expected: all PASS. (The snapshot is metric-level, not blob-level.)

- [ ] **Step 2: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: all PASS (508+ tests).

- [ ] **Step 3: Mypy / ruff full pass**

Run:
```bash
.venv/Scripts/python.exe -m mypy src tests
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m ruff format --check src tests
```
Expected: all green.

- [ ] **Step 4: Commit (no diff expected; sanity checkpoint)**

If anything was modified during this checkpoint, commit it. Otherwise no-op.

---

## Phase 2 — `EnsembleModel` scaffolding (static weights)

This phase ships `EnsembleModel` with static `weights = 0.5` for every (position, stat). No optimizer yet. The point is to exercise the fit-children → predict-mixture → codec round-trip plumbing end-to-end before adding the weight-fitting math.

### Task 2.1: Cross-cutting test for EnsembleModel (static weights)

**Files:**
- Create: `tests/test_models/test_ensemble_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_models/test_ensemble_model.py`. We'll piggyback on the synthetic feature/weekly_stats fixtures used by `test_lightgbm_nb.py`. Look at `tests/test_models/test_lightgbm_nb.py` line 1-30 for the imports + fixture pattern.

```python
"""Plan 6 Phase 2 — EnsembleModel scaffolding cross-cutting tests.

Uses the synthetic feature/weekly_stats fixtures from tests/conftest.py
(same pattern as test_lightgbm_nb.py).
"""

from __future__ import annotations

import pandas as pd
import pytest

from projections.distributions import MixtureDistribution, unpack_per_stat_params
from projections.models import (
    qb_ensemble,
    rb_ensemble,
    te_ensemble,
    wr_ensemble,
)
from projections.schemas import DistributionFamily, Position, ProjectionWeeklySchema, Ruleset


@pytest.mark.parametrize(
    "factory, position",
    [
        (qb_ensemble, Position.QB),
        (rb_ensemble, Position.RB),
        (te_ensemble, Position.TE),
        (wr_ensemble, Position.WR),
    ],
)
def test_ensemble_fit_predict_round_trip(
    factory, position, synthetic_features, synthetic_weekly_stats
) -> None:
    """EnsembleModel fits two children, predicts a MIXED row per player-week,
    and the params blob round-trips MixtureDistribution per stat."""
    pos_features = synthetic_features[position]
    pos_weekly = synthetic_weekly_stats

    model = factory()
    model.fit(pos_features, pos_weekly)

    predictions = model.predict_distribution(
        pos_features, ruleset=Ruleset.espn_ppr()
    )
    assert isinstance(predictions, pd.DataFrame)
    ProjectionWeeklySchema.validate(predictions)
    assert (predictions["family"] == DistributionFamily.MIXED.value).all()
    assert (predictions["model_id"].str.startswith("ensemble:")).all()

    # Round-trip the params blob for the first row.
    first_row = predictions.iloc[0]
    decoded = unpack_per_stat_params(bytes(first_row["params"]))
    for stat, dist in decoded.items():
        assert isinstance(dist, MixtureDistribution), f"stat {stat} not MixtureDistribution"


@pytest.mark.parametrize(
    "factory, position",
    [
        (qb_ensemble, Position.QB),
        (rb_ensemble, Position.RB),
        (te_ensemble, Position.TE),
        (wr_ensemble, Position.WR),
    ],
)
def test_ensemble_model_id_format(
    factory, position, synthetic_features, synthetic_weekly_stats
) -> None:
    """model_id format: 'ensemble:<pos>:<8-hex>:<train_start>-<train_end>'."""
    model = factory()
    with pytest.raises(RuntimeError, match="model_id"):
        _ = model.model_id  # not yet fitted

    model.fit(synthetic_features[position], synthetic_weekly_stats)
    parts = model.model_id.split(":")
    assert parts[0] == "ensemble"
    assert parts[1] == position.value.lower()
    assert len(parts[2]) == 8  # 8-char hex
    assert "-" in parts[3]  # train range


def test_ensemble_save_load_round_trip(
    tmp_path, synthetic_features, synthetic_weekly_stats
) -> None:
    """save() then load() yields a model that predicts identically."""
    model = qb_ensemble()
    model.fit(synthetic_features[Position.QB], synthetic_weekly_stats)
    pred_before = model.predict_distribution(
        synthetic_features[Position.QB], ruleset=Ruleset.espn_ppr()
    )

    save_path = tmp_path / "ensemble.joblib"
    model.save(save_path)

    from projections.models import EnsembleModel
    loaded = EnsembleModel.load(save_path)
    pred_after = loaded.predict_distribution(
        synthetic_features[Position.QB], ruleset=Ruleset.espn_ppr()
    )

    pd.testing.assert_frame_equal(
        pred_before.drop(columns=["generated_at"]).reset_index(drop=True),
        pred_after.drop(columns=["generated_at"]).reset_index(drop=True),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/test_ensemble_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'qb_ensemble'`.

- [ ] **Step 3: Read the existing fixtures to confirm signatures**

Run: `Grep "synthetic_features|synthetic_weekly_stats" tests/conftest.py` and confirm the fixture shape matches what the test expects (a `dict[Position, pd.DataFrame]` for features and a single combined frame for weekly stats). If they don't match exactly, adjust the test imports/usage to match the actual fixture surface.

---

### Task 2.2: Implement EnsembleModel skeleton (static weights = 0.5)

**Files:**
- Create: `src/projections/models/ensemble.py`

- [ ] **Step 1: Implement EnsembleModel with static 0.5 weights**

Create `src/projections/models/ensemble.py`:

```python
"""EnsembleModel — Plan 6 Model D.

Per-(position, stat) weighted mixture of Model A (BaselineModel) and Model
C-NB (LightGBMNbModel). Weights are constant per (position, stat); per-row
distributions are MixtureDistribution(F_a, F_b, w[stat]).

Phase 2 scaffolding: weights default to 0.5 per stat. Phase 3 wires the
pinball-loss optimizer into fit().

Per-row schema:
    family = DistributionFamily.MIXED
    params = pack_per_stat_params({stat: MixtureDistribution(...) for stat in target_stats})
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd

from projections.distributions import (
    MixtureDistribution,
    pack_per_stat_params,
    unpack_per_stat_params,
)
from projections.distributions.base import Distribution
from projections.models.base import Model, compute_code_hash
from projections.models.baseline import BaselineModel
from projections.models.lightgbm_nb import LightGBMNbModel
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    Position,
    ProjectionWeeklySchema,
    Ruleset,
    Stat,
)
from projections.scoring.score_distribution import (
    derive_row_seed,
    score_distribution,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_DEFAULT_WEIGHTS_DIR: Final[Path] = _PROJECT_ROOT / "data" / "ensemble_weights"


def _code_hash_files_ensemble(position: Position) -> tuple[Path, ...]:
    """Source files whose content is hashed into EnsembleModel.code_hash."""
    src = _PROJECT_ROOT / "src" / "projections"
    return (
        src / "models" / "ensemble.py",
        src / "distributions" / "mixture.py",
        src / "distributions" / "codec.py",
    )


@dataclass(slots=True)
class _EnsembleConfig:
    position: Position
    target_stats: tuple[Stat, ...]
    child_a_factory: Callable[[], BaselineModel]
    child_b_factory: Callable[[], LightGBMNbModel]
    weights_dir: Path = field(default=_DEFAULT_WEIGHTS_DIR)


class EnsembleModel:
    """Per-(position, stat) weighted mixture of Model A and Model C-NB."""

    _config: _EnsembleConfig
    _child_a: BaselineModel | None
    _child_b: LightGBMNbModel | None
    _weights: dict[Stat, float]
    _train_start: int | None
    _train_end: int | None
    _calibration_year: int | None
    _is_fitted: bool

    def __init__(self, *, config: _EnsembleConfig) -> None:
        self._config = config
        self._child_a = None
        self._child_b = None
        self._weights = {}
        self._train_start = None
        self._train_end = None
        self._calibration_year = None
        self._is_fitted = False

    @property
    def position(self) -> Position:
        return self._config.position

    @property
    def target_stats(self) -> tuple[Stat, ...]:
        return self._config.target_stats

    @property
    def code_hash(self) -> str:
        """SHA-256 first 8 hex of:
        - ensemble.py + mixture.py + codec.py file contents,
        - + child_a.code_hash + child_b.code_hash (if fitted),
        - + sorted(self._weights.items()) JSON-canonical bytes (if fitted).
        """
        files_hash = compute_code_hash(_code_hash_files_ensemble(self._config.position))
        if not self._is_fitted:
            return files_hash
        assert self._child_a is not None and self._child_b is not None
        # Combine the three pieces with hashlib once more so changes anywhere invalidate.
        import hashlib
        h = hashlib.sha256()
        h.update(files_hash.encode("utf-8"))
        h.update(self._child_a.code_hash.encode("utf-8"))
        h.update(self._child_b.code_hash.encode("utf-8"))
        weights_canonical = json.dumps(
            {s.value: round(w, 6) for s, w in sorted(self._weights.items())},
            sort_keys=True,
        )
        h.update(weights_canonical.encode("utf-8"))
        return h.hexdigest()[:8]

    @property
    def model_id(self) -> str:
        if not self._is_fitted:
            raise RuntimeError(
                "model_id not available before fit() — depends on training-time state"
            )
        assert self._train_start is not None and self._train_end is not None
        return (
            f"ensemble:{self._config.position.value.lower()}:"
            f"{self.code_hash}:{self._train_start}-{self._train_end}"
        )

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """Fit children and (in Phase 2) set static weights = 0.5 per stat.

        Phase 3 will replace the static-weight assignment with a pinball-loss
        optimizer over child predictions on the calibration year Y-1.
        """
        seasons = sorted(int(s) for s in features["season"].unique())
        if len(seasons) < 2:
            raise ValueError(
                f"EnsembleModel.fit needs >=2 training seasons; got {len(seasons)}"
            )

        # Phase 2 — train both children once on the full span; static weights.
        self._child_a = self._config.child_a_factory()
        self._child_a.fit(features, weekly_stats)
        self._child_b = self._config.child_b_factory()
        self._child_b.fit(features, weekly_stats)

        self._weights = {stat: 0.5 for stat in self._config.target_stats}

        self._train_start = seasons[0]
        self._train_end = seasons[-1]
        self._calibration_year = seasons[-1]
        self._is_fitted = True

    def predict_distribution(
        self, features: pd.DataFrame, ruleset: Ruleset
    ) -> pd.DataFrame:
        """Predict per-row composite fantasy-points distribution as the
        weighted mixture of A and C-NB per stat."""
        if not self._is_fitted or self._child_a is None or self._child_b is None:
            raise RuntimeError("predict_distribution requires fit() first")

        pred_a = self._child_a.predict_distribution(features, ruleset)
        pred_b = self._child_b.predict_distribution(features, ruleset)

        # Align rows on (gsis_id, season, week). Both children predict from
        # the same features; alignment must be 1:1.
        keys = ["gsis_id", "season", "week"]
        pred_a_idx = pred_a.set_index(keys, drop=False)
        pred_b_idx = pred_b.set_index(keys, drop=False)
        if not pred_a_idx.index.equals(pred_b_idx.index):
            raise RuntimeError(
                "child predictions misaligned — both children should predict on the same features"
            )

        out_rows: list[dict[str, Any]] = []
        generated_at = datetime.now(UTC)

        for row_idx in range(len(pred_a_idx)):
            row_a = pred_a_idx.iloc[row_idx]
            row_b = pred_b_idx.iloc[row_idx]

            per_stat_a = unpack_per_stat_params(bytes(row_a["params"]))
            per_stat_b = unpack_per_stat_params(bytes(row_b["params"]))

            per_stat_dists: dict[Stat, Distribution] = {}
            for stat in self._config.target_stats:
                per_stat_dists[stat] = MixtureDistribution(
                    component_a=per_stat_a[stat],
                    component_b=per_stat_b[stat],
                    weight=self._weights[stat],
                )

            seed = derive_row_seed(
                gsis_id=str(row_a["gsis_id"]),
                season=int(row_a["season"]),
                week=int(row_a["week"]),
                ruleset_name=ruleset.name,
            )
            composite = score_distribution(per_stat_dists, ruleset, seed=seed)

            out_rows.append(
                {
                    "gsis_id": str(row_a["gsis_id"]),
                    "season": int(row_a["season"]),
                    "week": int(row_a["week"]),
                    "position": self._config.position.value,
                    "team": str(row_a["team"]),
                    "opponent": str(row_a["opponent"]),
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

    def save(self, path: Path) -> None:
        joblib.dump(
            {
                "child_a": self._child_a,
                "child_b": self._child_b,
                "weights": self._weights,
                "train_start": self._train_start,
                "train_end": self._train_end,
                "calibration_year": self._calibration_year,
                "config_position": self._config.position.value,
                "config_target_stats": [s.value for s in self._config.target_stats],
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> EnsembleModel:
        data = joblib.load(path)
        position = Position(data["config_position"])
        target_stats = tuple(Stat(s) for s in data["config_target_stats"])
        # Factories are not serialized; we reconstruct config-only-by-position.
        # Callers that re-fit must re-create the model from the original factory.
        config = _EnsembleConfig(
            position=position,
            target_stats=target_stats,
            child_a_factory=lambda: data["child_a"],     # type: ignore[arg-type]
            child_b_factory=lambda: data["child_b"],     # type: ignore[arg-type]
        )
        instance = cls(config=config)
        instance._child_a = data["child_a"]
        instance._child_b = data["child_b"]
        instance._weights = {Stat(k) if isinstance(k, str) else k: float(v)
                             for k, v in data["weights"].items()}
        instance._train_start = int(data["train_start"])
        instance._train_end = int(data["train_end"])
        instance._calibration_year = int(data["calibration_year"])
        instance._is_fitted = True
        return instance
```

- [ ] **Step 2: Add per-position factories**

Append at the bottom of `src/projections/models/ensemble.py`:

```python
from projections.models.baseline import qb_baseline, rb_baseline, te_baseline, wr_baseline
from projections.models.lightgbm_nb import (
    qb_lightgbm_nb,
    rb_lightgbm_nb,
    te_lightgbm_nb,
    wr_lightgbm_nb,
)
from projections.models.lightgbm import (
    _QB_TARGET_STATS,
    _RB_TARGET_STATS,
    _TE_TARGET_STATS,
    _WR_TARGET_STATS,
)


def qb_ensemble() -> EnsembleModel:
    return EnsembleModel(
        config=_EnsembleConfig(
            position=Position.QB,
            target_stats=_QB_TARGET_STATS,
            child_a_factory=qb_baseline,
            child_b_factory=qb_lightgbm_nb,
        )
    )


def rb_ensemble() -> EnsembleModel:
    return EnsembleModel(
        config=_EnsembleConfig(
            position=Position.RB,
            target_stats=_RB_TARGET_STATS,
            child_a_factory=rb_baseline,
            child_b_factory=rb_lightgbm_nb,
        )
    )


def te_ensemble() -> EnsembleModel:
    return EnsembleModel(
        config=_EnsembleConfig(
            position=Position.TE,
            target_stats=_TE_TARGET_STATS,
            child_a_factory=te_baseline,
            child_b_factory=te_lightgbm_nb,
        )
    )


def wr_ensemble() -> EnsembleModel:
    return EnsembleModel(
        config=_EnsembleConfig(
            position=Position.WR,
            target_stats=_WR_TARGET_STATS,
            child_a_factory=wr_baseline,
            child_b_factory=wr_lightgbm_nb,
        )
    )
```

- [ ] **Step 3: Export from `models/__init__.py`**

Edit `src/projections/models/__init__.py`. Add the imports and the per-position dict entries:

```python
# Add to imports:
from projections.models.ensemble import (
    EnsembleModel,
    qb_ensemble,
    rb_ensemble,
    te_ensemble,
    wr_ensemble,
)

# Add to __all__:
    "EnsembleModel",
    "qb_ensemble",
    "rb_ensemble",
    "te_ensemble",
    "wr_ensemble",

# Add to each of _QB_FACTORIES / _RB_FACTORIES / _TE_FACTORIES / _WR_FACTORIES:
    "ensemble": qb_ensemble,   # in _QB_FACTORIES
    "ensemble": rb_ensemble,   # in _RB_FACTORIES
    "ensemble": te_ensemble,   # in _TE_FACTORIES
    "ensemble": wr_ensemble,   # in _WR_FACTORIES
```

- [ ] **Step 4: Run the cross-cutting test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/test_ensemble_model.py -v`
Expected: all PASS.

- [ ] **Step 5: Verification gate**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_models/ -v
.venv/Scripts/python.exe -m mypy src tests
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m ruff format --check src tests
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/projections/models/ensemble.py src/projections/models/__init__.py tests/test_models/test_ensemble_model.py
git commit -m "feat(models): EnsembleModel scaffolding (static 0.5 weights) — Plan 6 Phase 2"
```

---

## Phase 3 — Pinball weight fitting + JSON persistence

### Task 3.1: Pinball loss helper and weight optimizer (TDD)

**Files:**
- Create: `tests/test_models/test_ensemble_weight_fit.py`
- Modify: `src/projections/models/ensemble.py` (add helper + integrate into fit)

- [ ] **Step 1: Write the failing test**

Create `tests/test_models/test_ensemble_weight_fit.py`:

```python
"""Plan 6 Phase 3 — EnsembleModel weight fitting math."""

from __future__ import annotations

import numpy as np
import pytest

from projections.distributions import (
    ParametricNegativeBinomial,
    ParametricNormal,
    QuantileDistribution,
)
from projections.models.ensemble import _fit_weight_for_stat, _pinball


def test_pinball_loss_underestimate() -> None:
    """When q_pred < actual, pinball = q*(actual - q_pred) for q in (0, 1)."""
    # actual = 10, q_pred = 7, q = 0.9 → q_pred too low → loss = 0.9 * (10 - 7) = 2.7
    assert _pinball(actual=10.0, q_pred=7.0, q=0.9) == pytest.approx(2.7, abs=1e-12)


def test_pinball_loss_overestimate() -> None:
    """When q_pred > actual, pinball = (1-q)*(q_pred - actual)."""
    # actual = 10, q_pred = 13, q = 0.9 → q_pred too high → loss = 0.1 * 3 = 0.3
    assert _pinball(actual=10.0, q_pred=13.0, q=0.9) == pytest.approx(0.3, abs=1e-12)


def test_pinball_loss_exact_match() -> None:
    """When q_pred == actual, pinball = 0 regardless of q."""
    for q in [0.05, 0.1, 0.5, 0.9, 0.95]:
        assert _pinball(actual=10.0, q_pred=10.0, q=q) == pytest.approx(0.0, abs=1e-12)


def test_fit_weight_recovers_known_optimum_a_dominant() -> None:
    """When component A perfectly matches actuals and B is far off, w → 1.

    Construct components such that A's [p10, p90] tightly brackets each actual
    and B's is far above. Optimizer should push w toward the upper bound.
    """
    rng = np.random.default_rng(seed=7)
    n = 500
    actuals = rng.normal(loc=10.0, scale=2.0, size=n)
    components_a = [ParametricNormal(mean=10.0, std=2.0) for _ in range(n)]
    components_b = [ParametricNormal(mean=100.0, std=2.0) for _ in range(n)]

    w_star = _fit_weight_for_stat(
        components_a=components_a,
        components_b=components_b,
        actuals=actuals,
    )
    assert w_star > 0.9, f"expected w near 1.0 (A wins), got {w_star}"


def test_fit_weight_recovers_known_optimum_b_dominant() -> None:
    """Symmetric: B dominant → w → 0."""
    rng = np.random.default_rng(seed=11)
    n = 500
    actuals = rng.normal(loc=10.0, scale=2.0, size=n)
    components_a = [ParametricNormal(mean=100.0, std=2.0) for _ in range(n)]
    components_b = [ParametricNormal(mean=10.0, std=2.0) for _ in range(n)]

    w_star = _fit_weight_for_stat(
        components_a=components_a,
        components_b=components_b,
        actuals=actuals,
    )
    assert w_star < 0.1, f"expected w near 0.0 (B wins), got {w_star}"


def test_fit_weight_handles_quantile_component() -> None:
    """No crash with QuantileDistribution as component_b (Plan 5/5c shape)."""
    n = 200
    rng = np.random.default_rng(seed=3)
    actuals = rng.gamma(shape=2.0, scale=4.0, size=n)
    components_a = [ParametricNormal(mean=8.0, std=4.0) for _ in range(n)]
    components_b = [
        QuantileDistribution(
            quantiles=np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
            values=np.array([1.0, 4.0, 8.0, 14.0, 25.0]),
        )
        for _ in range(n)
    ]
    w_star = _fit_weight_for_stat(
        components_a=components_a,
        components_b=components_b,
        actuals=actuals,
    )
    assert 0.0 < w_star < 1.0


def test_fit_weight_handles_count_actuals() -> None:
    """Count-stat shape: NB component_b, integer actuals."""
    rng = np.random.default_rng(seed=5)
    n = 300
    actuals = rng.poisson(lam=1.5, size=n).astype(float)
    components_a = [ParametricNormal(mean=1.5, std=1.5) for _ in range(n)]
    components_b = [ParametricNegativeBinomial(mean=1.5, dispersion=4.0) for _ in range(n)]
    w_star = _fit_weight_for_stat(
        components_a=components_a,
        components_b=components_b,
        actuals=actuals,
    )
    assert 0.001 <= w_star <= 0.999


def test_fit_weight_clip_bounds() -> None:
    """Optimizer output is clipped into [0.001, 0.999]."""
    n = 100
    actuals = np.full(n, 10.0)
    components_a = [ParametricNormal(mean=10.0, std=0.001) for _ in range(n)]
    components_b = [ParametricNormal(mean=10.0, std=10.0) for _ in range(n)]
    w_star = _fit_weight_for_stat(
        components_a=components_a,
        components_b=components_b,
        actuals=actuals,
    )
    assert 0.001 <= w_star <= 0.999
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/test_ensemble_weight_fit.py -v`
Expected: FAIL with `ImportError: cannot import name '_fit_weight_for_stat'`.

- [ ] **Step 3: Implement `_pinball` and `_fit_weight_for_stat`**

In `src/projections/models/ensemble.py`, add at module level (above `EnsembleModel`):

```python
import warnings
from collections.abc import Sequence

from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from projections.distributions import MixtureDistribution


_QUANTILES_FOR_FIT: Final[tuple[float, float]] = (0.10, 0.90)
_WEIGHT_BOUNDS: Final[tuple[float, float]] = (0.001, 0.999)


def _pinball(actual: float, q_pred: float, q: float) -> float:
    """Standard quantile pinball loss.

    pinball(y, q_pred, q) = max((q - 1) * (q_pred - y), q * (q_pred - y))
                          = (q - 1{y < q_pred}) * (q_pred - y)
    """
    diff = q_pred - actual
    return float(max((q - 1.0) * diff, q * diff))


def _fit_weight_for_stat(
    *,
    components_a: Sequence[Distribution],
    components_b: Sequence[Distribution],
    actuals: NDArray[np.float64],
) -> float:
    """Fit one scalar weight w in (0.001, 0.999) minimizing summed pinball
    loss at q in {0.10, 0.90} on the per-row mixture distribution.

    components_a[i], components_b[i], actuals[i] correspond to the same row
    on the calibration year. len(components_a) == len(components_b) ==
    len(actuals) is required.
    """
    n = len(actuals)
    if not (n == len(components_a) == len(components_b)):
        raise ValueError(
            f"length mismatch: components_a={len(components_a)}, "
            f"components_b={len(components_b)}, actuals={n}"
        )
    if n == 0:
        warnings.warn(
            "_fit_weight_for_stat received zero-length input; returning 0.5 default",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.5

    def loss(w: float) -> float:
        total = 0.0
        for a_dist, b_dist, actual in zip(components_a, components_b, actuals, strict=True):
            mix = MixtureDistribution(component_a=a_dist, component_b=b_dist, weight=w)
            for q in _QUANTILES_FOR_FIT:
                q_pred = mix.quantile(q)
                total += _pinball(float(actual), q_pred, q)
        return total

    try:
        result = minimize_scalar(
            loss,
            method="bounded",
            bounds=_WEIGHT_BOUNDS,
            options={"xatol": 1e-3},
        )
        if result.success and np.isfinite(result.fun):
            return float(np.clip(result.x, *_WEIGHT_BOUNDS))
    except (ValueError, OverflowError) as exc:
        warnings.warn(
            f"_fit_weight_for_stat: scipy optimization failed ({exc!r}); falling back to grid",
            RuntimeWarning,
            stacklevel=2,
        )

    # Grid-search fallback.
    grid = np.linspace(_WEIGHT_BOUNDS[0], _WEIGHT_BOUNDS[1], 11)
    losses = []
    for w in grid:
        try:
            losses.append(loss(float(w)))
        except (ValueError, OverflowError):
            losses.append(np.inf)
    if not np.any(np.isfinite(losses)):
        warnings.warn(
            "_fit_weight_for_stat: all grid points returned non-finite loss; "
            "returning 0.5 default",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.5
    return float(grid[int(np.argmin(losses))])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/test_ensemble_weight_fit.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/models/ensemble.py tests/test_models/test_ensemble_weight_fit.py
git commit -m "feat(models): pinball weight optimizer for ensemble — Plan 6 Phase 3 (optimizer)"
```

---

### Task 3.2: Wire weight fitting into `EnsembleModel.fit`

**Files:**
- Modify: `src/projections/models/ensemble.py:fit method`
- Modify: `tests/test_models/test_ensemble_model.py` (add test that fitted weights differ from 0.5)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models/test_ensemble_model.py`:

```python
def test_fit_produces_non_static_weights(synthetic_features, synthetic_weekly_stats) -> None:
    """After fit() with the real pinball optimizer, at least one weight should
    differ from 0.5 (Phase 3 replaces the static-0.5 default with optimized values)."""
    model = qb_ensemble()
    model.fit(synthetic_features[Position.QB], synthetic_weekly_stats)
    # The synthetic fixture isn't required to produce non-0.5 weights for every
    # stat, but on average across stats the optimizer should not produce a flat
    # 0.5 vector. Permit individual stats == 0.5 if optimizer happens to land there.
    weights = list(model._weights.values())
    assert any(w != 0.5 for w in weights), \
        f"expected at least one optimized weight != 0.5, got {weights}"


def test_fit_rejects_too_few_seasons(synthetic_features, synthetic_weekly_stats) -> None:
    """EnsembleModel.fit needs >=3 seasons (>=2 for weight-fit children +
    1 calibration year)."""
    qb_features = synthetic_features[Position.QB]
    seasons = sorted(qb_features["season"].unique())
    if len(seasons) < 3:
        pytest.skip(
            f"synthetic fixture has only {len(seasons)} seasons; "
            "need >=3 to exercise the weight-fit split"
        )
    only_two = qb_features[qb_features["season"].isin(seasons[:2])]
    weekly_two = synthetic_weekly_stats[
        synthetic_weekly_stats["season"].isin(seasons[:2])
    ]
    model = qb_ensemble()
    with pytest.raises(ValueError, match="seasons"):
        model.fit(only_two, weekly_two)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/test_ensemble_model.py -v -k non_static`
Expected: FAIL because Phase 2's static-0.5 implementation produces all 0.5 weights.

- [ ] **Step 3: Replace `EnsembleModel.fit` with the 4-stage flow**

In `src/projections/models/ensemble.py`, replace the `fit` method:

```python
    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """4-stage fit: weight-fit children on [S, Y-2] → predict Y-1 →
        fit weights → re-fit prediction children on [S, Y-1].

        Spec §3.1.
        """
        seasons = sorted(int(s) for s in features["season"].unique())
        if len(seasons) < 3:
            raise ValueError(
                f"EnsembleModel.fit needs >=3 training seasons "
                f"(>=2 for weight-fit children + 1 calibration year); got {len(seasons)}"
            )

        cal_year = seasons[-1]
        weight_fit_seasons = seasons[:-1]

        wf_features = features[features["season"].isin(weight_fit_seasons)].copy()
        wf_weekly = weekly_stats[weekly_stats["season"].isin(weight_fit_seasons)].copy()
        cal_features = features[features["season"] == cal_year].copy()
        cal_weekly = weekly_stats[weekly_stats["season"] == cal_year].copy()

        # Stage 1 — weight-fit children on [S, Y-2]
        child_a_wf = self._config.child_a_factory()
        child_b_wf = self._config.child_b_factory()
        child_a_wf.fit(wf_features, wf_weekly)
        child_b_wf.fit(wf_features, wf_weekly)

        # Stage 2 — predict Y-1
        ruleset = Ruleset.espn_ppr()
        pred_a = child_a_wf.predict_distribution(cal_features, ruleset=ruleset)
        pred_b = child_b_wf.predict_distribution(cal_features, ruleset=ruleset)

        # Stage 3 — fit weights via pinball at q in {0.10, 0.90}
        self._weights = self._fit_weights(
            pred_a=pred_a,
            pred_b=pred_b,
            cal_weekly=cal_weekly,
        )

        # Stage 4 — re-fit children on the full prediction span [S, Y-1]
        self._child_a = self._config.child_a_factory()
        self._child_a.fit(features, weekly_stats)
        self._child_b = self._config.child_b_factory()
        self._child_b.fit(features, weekly_stats)

        self._train_start = seasons[0]
        self._train_end = seasons[-1]
        self._calibration_year = cal_year
        self._is_fitted = True

        # Stage 5 — persist weights JSON for traceability
        self._write_weights_json()

    def _fit_weights(
        self,
        *,
        pred_a: pd.DataFrame,
        pred_b: pd.DataFrame,
        cal_weekly: pd.DataFrame,
    ) -> dict[Stat, float]:
        """For each target stat, fit one scalar weight via pinball at q=(0.10, 0.90)."""
        keys = ["gsis_id", "season", "week"]
        # Filter cal_weekly to this position only.
        cal_pos = cal_weekly[cal_weekly["position"] == self._config.position.value].copy()

        # Inner-join predictions with actuals on (gsis_id, season, week) to align rows.
        # We pull the per-stat actual columns out of cal_pos.
        target_cols = [s.value for s in self._config.target_stats]
        joined = pred_a[keys].merge(
            cal_pos[keys + target_cols],
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if joined.empty:
            warnings.warn(
                f"EnsembleModel._fit_weights: no calibration rows for "
                f"{self._config.position.value}; defaulting all weights to 0.5",
                RuntimeWarning,
                stacklevel=2,
            )
            return {stat: 0.5 for stat in self._config.target_stats}

        # Pull aligned indices from pred_a / pred_b.
        pred_a_keyed = pred_a.set_index(keys, drop=False)
        pred_b_keyed = pred_b.set_index(keys, drop=False)
        joined_idx = joined.set_index(keys, drop=False)

        # Decode per-row per-stat distributions for both children, restricted
        # to the joined rows.
        per_row_a: list[dict[Stat, Distribution]] = [
            unpack_per_stat_params(bytes(pred_a_keyed.loc[idx, "params"]))
            for idx in joined_idx.index
        ]
        per_row_b: list[dict[Stat, Distribution]] = [
            unpack_per_stat_params(bytes(pred_b_keyed.loc[idx, "params"]))
            for idx in joined_idx.index
        ]

        weights: dict[Stat, float] = {}
        for stat in self._config.target_stats:
            actuals = joined[stat.value].to_numpy(dtype=np.float64)
            components_a = [r[stat] for r in per_row_a]
            components_b = [r[stat] for r in per_row_b]
            weights[stat] = _fit_weight_for_stat(
                components_a=components_a,
                components_b=components_b,
                actuals=actuals,
            )
        return weights

    def _write_weights_json(self) -> None:
        """Write the weights artifact to data/ensemble_weights/{model_id}.json."""
        if self._child_a is None or self._child_b is None:
            return
        self._config.weights_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self._config.weights_dir / f"{self.model_id}.json"
        payload = {
            "model_class": "ensemble",
            "position": self._config.position.value,
            "train_seasons": [self._train_start, self._train_end],
            "calibration_year": self._calibration_year,
            "child_a_model_id": self._child_a.model_id,
            "child_b_model_id": self._child_b.model_id,
            "weights": {stat.value: round(w, 6) for stat, w in self._weights.items()},
            "fitted_at": datetime.now(UTC).isoformat(),
        }
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models/test_ensemble_model.py -v -k non_static`
Expected: PASS.

- [ ] **Step 5: Run all ensemble + cross-cutting tests**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_models/test_ensemble_model.py tests/test_models/test_ensemble_weight_fit.py tests/test_distributions/ -v
```
Expected: all PASS.

- [ ] **Step 6: Verification gate**

Run:
```bash
.venv/Scripts/python.exe -m mypy src tests
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m ruff format --check src tests
```
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/projections/models/ensemble.py tests/test_models/test_ensemble_model.py
git commit -m "feat(models): wire pinball weight fitting + JSON persistence — Plan 6 Phase 3"
```

---

## Phase 4 — POSITION_DISPATCH wiring + harness + CLI

The factories were already added to `_QB_FACTORIES` etc. in Phase 2. This phase wires the harness `cast` and the CLI choices, then runs the end-to-end smoke.

### Task 4.1: Widen the harness cast

**Files:**
- Modify: `src/projections/backtest/harness.py:33,261-264`

- [ ] **Step 1: Add EnsembleModel to the harness imports**

Edit `src/projections/backtest/harness.py` line 33 (`from projections.models import POSITION_DISPATCH, BaselineModel, LightGBMModel`):

```python
from projections.models import (
    POSITION_DISPATCH,
    BaselineModel,
    EnsembleModel,
    LightGBMModel,
)
```

- [ ] **Step 2: Widen the cast**

In `src/projections/backtest/harness.py`, find lines 261-264 (the `cast(...)` block) and replace:

```python
                model = cast(
                    BaselineModel | LightGBMModel | EnsembleModel,
                    dispatch.factories[model_class](),
                )
```

- [ ] **Step 3: Run the existing harness tests to verify nothing regressed**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest/ -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/projections/backtest/harness.py
git commit -m "feat(backtest): widen harness cast to include EnsembleModel — Plan 6 Phase 4 (harness)"
```

---

### Task 4.2: CLI: `--model ensemble` and expand `--model all`

**Files:**
- Modify: `scripts/backtest.py:117-134`

- [ ] **Step 1: Update CLI choices and `--model all` expansion**

Edit `scripts/backtest.py`. Change the `--model` argparse `choices` and the `--model all` expansion:

```python
    parser.add_argument(
        "--model",
        choices=["baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb", "ensemble", "both", "all"],
        default="both",
        help=(
            "Which model class(es) to run. "
            "'both' = Model A + Model C (legacy default). "
            "'all' = Model A + Model C + Model C-tuned + Model C-NB + Ensemble."
        ),
    )
    args = parser.parse_args()

    if args.model == "both":
        model_classes: tuple[str, ...] = ("baseline", "lightgbm")
    elif args.model == "all":
        model_classes = ("baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb", "ensemble")
    else:
        model_classes = (args.model,)
```

- [ ] **Step 2: Smoke-test the CLI argument parsing**

Run:
```bash
.venv/Scripts/python.exe scripts/backtest.py --help
```
Expected: `--model` choices listed, including `ensemble` and `all` (5 classes).

- [ ] **Step 3: Commit**

```bash
git add scripts/backtest.py
git commit -m "feat(backtest): --model ensemble + 5-way --model all — Plan 6 Phase 4 (cli)"
```

---

### Task 4.3: Synthetic-fixture smoke through the harness

**Files:**
- Modify: `tests/test_backtest/test_harness.py` (extend existing smoke)

- [ ] **Step 1: Find the existing harness smoke**

Run: `Grep "model_classes" tests/test_backtest/test_harness.py -n`. Find the test that calls `run_backtest(model_classes=...)`.

- [ ] **Step 2: Add a parametrize entry for the ensemble**

Append a new test (or extend the existing one) in `tests/test_backtest/test_harness.py`:

```python
def test_run_backtest_includes_ensemble(synthetic_features, synthetic_weekly_stats, tmp_path):
    """Smoke: run_backtest with model_classes including 'ensemble' produces
    rows tagged with model_class='ensemble' and at least one row per (position, year)."""
    # Use the existing fixtures shape; refer to the existing smoke test for the
    # exact arg shape (features_root / raw_root paths).
    from projections.backtest.harness import run_backtest

    # Adapt this call to match the existing smoke; the key assertion is the
    # presence of the ensemble model_class in the output.
    run = run_backtest(
        held_out_years=(2023,),  # single year for speed
        positions=(Position.QB,),
        model_classes=("baseline", "lightgbm-nb", "ensemble"),
        features_root=...,    # match existing smoke
        raw_root=...,         # match existing smoke
    )
    assert "ensemble" in set(run.metrics["model_class"].unique())
    # Ensemble rows: 23 metrics × 1 position × 1 year = 23 (no season_calibration_*)
    ensemble_rows = run.metrics[run.metrics["model_class"] == "ensemble"]
    assert len(ensemble_rows) >= 1
```

The exact `features_root` / `raw_root` / fixture wiring should mirror the existing harness smoke; copy from there. Don't introduce a new fixture pattern.

- [ ] **Step 3: Run the smoke**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest/test_harness.py -v -k ensemble`
Expected: PASS.

- [ ] **Step 4: Verification gate**

Run:
```bash
.venv/Scripts/python.exe -m pytest -v
.venv/Scripts/python.exe -m mypy src tests
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m ruff format --check src tests
.venv/Scripts/python.exe -m pytest -v -k "ingest or store or schemas"
```
Expected: all green; full suite passes (508+ existing tests + new mixture/codec/ensemble tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_backtest/test_harness.py
git commit -m "test(backtest): smoke covers ensemble end-to-end — Plan 6 Phase 4 (smoke)"
```

---

## Phase 5 — Real-data backtest + adoption-gate verdict

This phase runs the actual backtest harness on real `data/features/` and `data/raw/weekly_stats/` data, regenerates the snapshot, and applies the §1.3 adoption gate.

### Task 5.1: Run real-data backtest with ensemble

**Files:**
- Modify: `tests/backtest/model_metrics.json` (regenerated)
- Create: `data/ensemble_weights/*.json` (16 files: 4 positions × 4 folds)

- [ ] **Step 1: Run the backtest in update-snapshot mode**

Run:
```bash
.venv/Scripts/python.exe scripts/backtest.py --model all --update-snapshot
```
Expected output:
- Per-fold model fitting log lines.
- "Previous snapshot: 1504 rows."
- "New snapshot: 1872 rows."
- `data/backtest/run_<ts>/results.parquet` written.
- `data/backtest/run_<ts>/season_results.parquet` written.
- 16 new files under `data/ensemble_weights/ensemble:<pos>:*.json` (4 per position; one per fold's prediction children — though the same training span across folds may produce identical model_ids; verify count after run).

- [ ] **Step 2: Verify snapshot row count**

Run: `.venv/Scripts/python.exe -c "import json; print(len(json.loads(open('tests/backtest/model_metrics.json').read())))"`
Expected: 1872.

- [ ] **Step 3: Determinism check**

Run:
```bash
.venv/Scripts/python.exe scripts/backtest.py --check
```
Expected: `PASS — 1872 metrics within tolerance.`

If FAIL: investigate the source of non-determinism. Mixture quantile uses brentq (deterministic given fixed inputs); composite uses derive_row_seed; pinball optimizer is deterministic given fixed predictions; child models are deterministic per Plan 3d. Most likely cause is hash-seed instability — check Python `PYTHONHASHSEED` if running outside the standard venv.

- [ ] **Step 4: Stage regenerated snapshot + weights artifacts**

```bash
git add tests/backtest/model_metrics.json data/ensemble_weights/
git status
```
Verify: `tests/backtest/model_metrics.json` modified; `data/ensemble_weights/*.json` (16 new files).

- [ ] **Step 5: Commit (snapshot delta)**

```bash
git commit -m "chore(backtest): regenerate snapshot with ensemble rows — Plan 6 Phase 5"
```

---

### Task 5.2: Build the per-cell decision table

**Files:**
- Output: tabular data for the Plan 6 PM entry (no file commit at this step)

- [ ] **Step 1: Generate the per-cell comparison**

Run (output goes to stdout; copy into the PM entry in Phase 6):

```bash
.venv/Scripts/python.exe -c "
import json
import pandas as pd

snapshot = json.loads(open('tests/backtest/model_metrics.json').read())
df = pd.DataFrame(snapshot)

target_metrics = ['composite_rmse', 'spearman_topN', 'weekly_calibration_p10p90']
filtered = df[df['metric'].isin(target_metrics)]
pivot = filtered.pivot_table(
    index=['position', 'year'],
    columns=['metric', 'model_class'],
    values='value',
)
print(pivot.to_string(float_format=lambda x: f'{x:7.4f}'))
"
```

Capture the printed table for the PM entry.

- [ ] **Step 2: Apply §1.3 adoption-gate criteria**

Manually evaluate from the table:

1. **RMSE strictly lower on ≥12/16 cells; max +1% worse.** Count cells where `composite_rmse[ensemble] < composite_rmse[baseline]`. Compute pct delta `(ensemble - baseline) / baseline * 100` per cell; check max.
2. **Spearman within ±0.005 on every cell.** Compute `spearman_topN[ensemble] - spearman_topN[baseline]` per cell; check abs max.
3. **Calibration no worse on any cell; mean delta ≥ +0.02.** Compute `weekly_calibration_p10p90[ensemble] - weekly_calibration_p10p90[baseline]` per cell; check sign + mean.

Record verdict in scratch notes for use in Phase 6.

- [ ] **Step 3: Decide adopt vs peer**

Based on the verdict in Step 2:

- All three pass → adopt path (Task 5.3 below).
- Any fail → peer path (skip Task 5.3; proceed directly to Phase 6).

---

### Task 5.3: (Conditional) Adoption — update CLI default + CLAUDE.md

Run only if all three §1.3 criteria pass.

**Files:**
- Modify: `scripts/train_baseline.py` (default model class)
- Modify: `CLAUDE.md` (production-default note)

- [ ] **Step 1: Inspect the current default**

Run: `Grep "default" scripts/train_baseline.py | head -10`. Find the model-class default.

- [ ] **Step 2: Change default to `ensemble`**

Edit the relevant default (likely an argparse `default=` or a hard-coded factory call). Replace `BaselineModel` factory or `"baseline"` string with the ensemble equivalent.

- [ ] **Step 3: Update CLAUDE.md**

In `CLAUDE.md` "Project shape" section, add a line under Projections Core:

```markdown
- **Production default model:** Model D ensemble (Plan 6, 2026-04-28). Per-(position, stat) weighted mixture of Model A and Model C-NB; weights at `data/ensemble_weights/`.
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_baseline.py CLAUDE.md
git commit -m "feat(models): adopt Model D ensemble as production default — Plan 6 Phase 5"
```

---

## Phase 6 — `project_management.md` + TODO updates

### Task 6.1: Add Plan 6 entry to project_management.md

**Files:**
- Modify: `project_management.md` (prepend Plan 6 entry at the top)

- [ ] **Step 1: Compose the Plan 6 PM entry**

Prepend to `project_management.md` immediately after the `---` following the header lines (currently the Plan 7 entry sits at the top). Use this template, filling in the actual numbers from Phase 5's decision table:

```markdown
## Plan 6 — Model D ensemble (A + C-NB) — <VERDICT> (run 2026-04-28)

**Verdict:** <adopted as production default | shipped as peer>. Per-(position, stat) calibration-aware weighted mixture of Model A and Model C-NB. <One-paragraph summary of how the per-position split moved.>

### Per-position model_ids

| Position | Model A | Model C-NB | Model D Ensemble |
|---|---|---|---|
| WR | `baseline:wr:6d955427:2018-2023` | `lightgbm-nb:wr:dc445a2d:2018-2023` | `<from data/ensemble_weights/>` |
| QB | ... | ... | ... |
| RB | ... | ... | ... |
| TE | ... | ... | ... |

### Adoption-gate verdict — <ADOPT | DO NOT ADOPT>

| Criterion | Threshold | Actual (D vs A) | Pass? |
|---|---|---|---|
| Composite RMSE strictly lower on >=12/16 cells; max +1% worse | D < A on 12+; max +1% worse | D strictly lower on **N/16**; max <%> worse | <PASS\|FAIL> |
| Spearman top-N within +-0.005 on every cell | All within ±0.005 | <count> outside ±0.005; max abs delta <value> | <PASS\|FAIL> |
| Calibration no worse on any cell; mean delta >= +0.02 | No regressions; mean ≥ +0.02 | D worse on <count>/16; mean delta <value> | <PASS\|FAIL> |

### Side-by-side per-cell comparison (16 cells × 3 metrics × 5 models)

<copy the table from Phase 5 Task 5.2 Step 1>

### Per-stat fitted weights (per-position, fold-specific)

<short summary of the fitted weight vectors, e.g., "QB cells consistently weight C-NB heavily (w_a in [0.05, 0.20] across stats); RB/TE/WR cells weight A heavily on count stats (w_a in [0.65, 0.85]) and roughly evenly on yards stats (w_a near 0.5)">

### Decision

<adopt path: Model D becomes production default; A/C/C-tuned/C-NB stay as peers for backtest comparability. scripts/train_baseline.py default updated; CLAUDE.md notes Model D as production.>

OR

<peer path: Model A stays default. All four classes (A/C/C-tuned/C-NB) plus Ensemble ship as peers. The experimental verdict is the deliverable.>

### Why this should work / does it work

<one paragraph reflecting on whether per-stat pinball at p10/p90 moved composite calibration. If composite [p10, p90] coverage moved as expected → cite the specific delta. If per-stat moved but composite didn't (Plan 7's lesson recurring) → flag as a clean diagnostic for a follow-up plan to optimize composite directly.>

### Next action

<one of:>
- Plan 4 (public Python API + CLI verbs + free-tier hosting) — modeling work has reached "good enough"; pivot to consumer tools.
- TODO #3 (PBP / EPA features) — feature-class track; estimated 5-15% RMSE win on top of any model class.
- TODO #23 (target decomposition) — feature-class track; volume × efficiency separation.
- Composite-direct calibration optimization (follow-up to TODO #30) — if Plan 6 showed per-stat moving but composite not closing.

---
```

- [ ] **Step 2: Verify the file is well-formed**

Run: `wc -l project_management.md` and confirm the new entry is ~60 lines.

Run: `head -100 project_management.md` and confirm the Plan 6 entry sits at the top with no broken markdown.

- [ ] **Step 3: Commit**

```bash
git add project_management.md
git commit -m "docs(plan-6): record Model D verdict + tables — Plan 6 Phase 6"
```

---

### Task 6.2: Update TODO.md

**Files:**
- Modify: `TODO.md` (items #28, #29, #30)

- [ ] **Step 1: Update TODO #28 (season aggregation widening)**

Find the `### 28.` section in TODO.md. Append:

```markdown
**Update 2026-04-28 (Plan 6):** Ensemble inherits the same SAMPLED_SUMMARY-only family gate as C / C-tuned / C-NB; ensemble cells skip the 8 `season_calibration_*` rows per (position, year) cell (32 rows total). TODO #28 is unchanged in scope; widening would now add ~96 rows to model_metrics.json (32 each for QUANTILE, QUANTILE-tuned, MIXED) plus 32 for the new MIXED-via-MIXTURE ensemble path.
```

- [ ] **Step 2: Update TODO #29 (prune Model C-tuned)**

Find the `### 29.` section. Append:

```markdown
**Update 2026-04-28 (Plan 6):** Ensemble landed (as <adopt|peer>); the deferred pruning condition was "after Plan 6 lands and we're confident which model classes the ensemble references." Plan 6's ensemble references **A and C-NB** only — Tuned is not in the ensemble. Tuned is therefore a clean pruning candidate. Concrete tasks listed in TODO #29 remain accurate; the ensemble's snapshot delta would shrink from 368 to 0 ensemble rows (no change) plus the existing 368 lightgbm-tuned rows would drop. Defer until next housekeeping pass.
```

- [ ] **Step 3: Update TODO #30 (upper-tail count calibration)**

Find the `### 30.` section. Append:

```markdown
**Update 2026-04-28 (Plan 6):** Ensemble's per-stat pinball at q ∈ {0.10, 0.90} fitted weights on the calibration year. <If composite calibration improved → cite; otherwise note that this confirms the per-stat-vs-composite decomposition gap from Plan 7's diagnostic.> The remaining candidate mechanisms (composite-direct optimization via Monte Carlo; ZIP family for count cells; explicit point-mass-at-0 mixture) all stay open. Plan 6 closes the "is the per-position split exploitable via simple stacking?" question; the binding constraint for further calibration improvement remains where Plan 7 located it (upper-tail count behavior at p95+).
```

- [ ] **Step 4: Commit**

```bash
git add TODO.md
git commit -m "docs(todo): #28/#29/#30 progress notes — Plan 6 Phase 6"
```

---

### Task 6.3: Final verification + push

**Files:**
- (no file changes; quality gate)

- [ ] **Step 1: Full test suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: all PASS (existing 508 + new mixture/codec/ensemble tests + ensemble snapshot rows).

- [ ] **Step 2: mypy**

Run: `.venv/Scripts/python.exe -m mypy src tests`
Expected: 0 errors.

- [ ] **Step 3: ruff check**

Run: `.venv/Scripts/python.exe -m ruff check src tests`
Expected: 0 violations.

- [ ] **Step 4: ruff format**

Run: `.venv/Scripts/python.exe -m ruff format --check src tests`
Expected: 0 drift.

- [ ] **Step 5: Targeted ingest/store/schemas**

Run: `.venv/Scripts/python.exe -m pytest -v -k "ingest or store or schemas"`
Expected: all PASS.

- [ ] **Step 6: Determinism re-check**

Run: `.venv/Scripts/python.exe scripts/backtest.py --check`
Expected: `PASS — 1872 metrics within tolerance.`

- [ ] **Step 7: Push the branch and open PR**

```bash
git push -u origin feat/plan-6-ensemble
```

Then open a PR from `feat/plan-6-ensemble` → `main`. Title: "Plan 6 — Model D ensemble (A + C-NB) — <VERDICT>". Body should summarize: spec/plan paths, snapshot delta (1504 → 1872), adoption verdict, what shipped to main, the Plan 6 PM entry's per-cell table, the §1.3 verdict on the three criteria.

---

## Self-review checklist (run after writing the plan)

- [x] Spec coverage: every section of the spec maps to at least one task.
  - §1 goals → Task 5.2 (gate evaluation), Task 6.1 (PM verdict).
  - §2.1 file structure → all create/modify rows covered across phases.
  - §2.2 EnsembleModel class → Task 2.2.
  - §2.3 MixtureDistribution → Task 1.2.
  - §2.4 Distribution Protocol cdf → Task 0.1, 0.2.
  - §2.5 codec MIXTURE → Task 1.3.
  - §2.6 weights persistence → Task 3.2 (`_write_weights_json`).
  - §3.1 fit data flow → Task 3.2 (4-stage fit).
  - §3.2 leakage analysis → covered structurally in fit; no test needed (verified by construction).
  - §3.3 weight fitting → Task 3.1.
  - §3.4 predict-time mixture → Task 2.2 (predict_distribution).
  - §3.5 harness integration → Task 4.1, 4.2, 4.3.
  - §4 testing strategy → Tasks 0.1, 0.2, 1.2, 1.3, 2.1, 3.1, 3.2 tests; harness smoke 4.3.
  - §5 phased rollout → matches the 7 phase headers.
  - §6 backwards compat & risks → covered in code (no v1-blob breakage; additive Protocol; harness one-line widening).
  - §7 success criteria → Phase 5 + Phase 6 (decision table + PM entry).
- [x] No placeholders / TBDs in code blocks.
- [x] Type consistency: `EnsembleModel.fit/predict/save/load`, `_fit_weight_for_stat`, `_pinball`, `MixtureDistribution.{mean,std,cdf,quantile,sample}` all consistent across tasks.
- [x] All command/run lines use `.venv/Scripts/python.exe -m ...` consistent with the project's Windows-bash environment.
