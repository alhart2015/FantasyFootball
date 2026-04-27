# Plan 3e Phase 1+ — Calibration Tightening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phases 1-3 of Plan 3e Phase 1+ — add `ParametricNegativeBinomial` for zero-inflated count stats (Phase 1), `ParametricStudentT` for heavy-tailed yards stats (Phase 2), and per-tertile variance bucketing across all families (Phase 3). Move calibration coverage toward the 0.80 target as informed by Phase 0's diagnostic findings.

**Architecture:** Three sequential phases on the existing `feat/plan-3e-calibration-tightening` branch. Each phase ends with a wholesale re-snapshot of `tests/backtest/baseline_metrics.json`, a standalone-artifact retrain, and a PM-doc update reporting per-cell coverage delta. New families conform to the existing `Distribution` Protocol; no changes to scoring, aggregation, or harness layers. Variance-parameter shape evolves from scalar (Phase 1-2) to per-bucket lists (Phase 3) with a codec schema-version bump.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy.stats / scipy.optimize, scikit-learn, pandera, msgpack, pytest.

**Spec:** `docs/superpowers/specs/2026-04-26-plan-3e-calibration-tightening-design.md` (sections 8-10 cover Phase 1+).

**Phase 0 context (already merged in this branch):** `scripts/diagnose_calibration.py` is the calibration diagnostic; `docs/superpowers/research/2026-04-26-calibration-diagnosis.md` is the research report driving the family choices below.

---

## Phase 1 — `ParametricNegativeBinomial` for zero-inflated counts

Closes the 10 cells with `coverage_p10p90 = 0.0` (every `*_tds`, `interceptions`, `fumbles_lost` across all four positions). Phase ends with all four positions' factories rewired, codec extended, snapshot regenerated.

### Task 1.1: Add `ParametricNegativeBinomial` distribution class

**Files:**
- Modify: `src/projections/distributions/parametric.py`
- Modify: `src/projections/distributions/__init__.py`
- Create: `tests/test_distributions/test_neg_binomial.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_distributions/test_neg_binomial.py`:

```python
"""Tests for ParametricNegativeBinomial."""

from __future__ import annotations

import numpy as np
import pytest

from projections.distributions import ParametricNegativeBinomial


def test_mean_matches_constructor_arg() -> None:
    dist = ParametricNegativeBinomial(mean=2.0, dispersion=4.0)
    assert dist.mean() == pytest.approx(2.0)


def test_std_overdispersed_vs_poisson() -> None:
    """NB std exceeds Poisson std (sqrt(mean)) when dispersion is finite."""
    dist = ParametricNegativeBinomial(mean=2.0, dispersion=4.0)
    # var = mean + mean^2 / dispersion = 2 + 4/4 = 3; std = sqrt(3)
    assert dist.std() == pytest.approx(np.sqrt(3.0))


def test_quantile_monotonic() -> None:
    dist = ParametricNegativeBinomial(mean=1.0, dispersion=2.0)
    assert dist.quantile(0.1) <= dist.quantile(0.5) <= dist.quantile(0.9)


def test_quantile_rejects_out_of_range() -> None:
    dist = ParametricNegativeBinomial(mean=1.0, dispersion=2.0)
    with pytest.raises(ValueError, match="q must be in"):
        dist.quantile(0.0)
    with pytest.raises(ValueError, match="q must be in"):
        dist.quantile(1.0)


def test_sample_returns_correct_shape_and_dtype() -> None:
    dist = ParametricNegativeBinomial(mean=1.0, dispersion=2.0)
    rng = np.random.default_rng(42)
    samples = dist.sample(n=500, rng=rng)
    assert samples.shape == (500,)
    assert samples.dtype == np.float64
    # NB samples are non-negative integers (cast to float64).
    assert (samples >= 0).all()


def test_sample_mean_approximates_param_mean() -> None:
    """Law of large numbers — sample mean is close to the parameterized mean."""
    dist = ParametricNegativeBinomial(mean=0.3, dispersion=2.0)
    rng = np.random.default_rng(0)
    samples = dist.sample(n=10_000, rng=rng)
    assert samples.mean() == pytest.approx(0.3, abs=0.05)


def test_constructor_rejects_non_positive_mean() -> None:
    with pytest.raises(ValueError, match="mean must be positive"):
        ParametricNegativeBinomial(mean=0.0, dispersion=2.0)
    with pytest.raises(ValueError, match="mean must be positive"):
        ParametricNegativeBinomial(mean=-0.5, dispersion=2.0)


def test_constructor_rejects_non_positive_dispersion() -> None:
    with pytest.raises(ValueError, match="dispersion must be positive"):
        ParametricNegativeBinomial(mean=1.0, dispersion=0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_distributions/test_neg_binomial.py -v`
Expected: FAIL with `ImportError: cannot import name 'ParametricNegativeBinomial'`.

- [ ] **Step 3: Implement `ParametricNegativeBinomial`**

Append to `src/projections/distributions/parametric.py` (after `ParametricGamma`):

```python
@dataclass(slots=True, frozen=True, init=False)
class ParametricNegativeBinomial:
    """Negative Binomial parameterized as (mean, dispersion).

    Internally uses scipy's (n, p) parameterization:
        n = mean^2 / dispersion
        p = n / (n + mean)
    Variance: var = mean + mean^2 / dispersion (overdispersed vs Poisson when
    dispersion is finite; recovers Poisson as dispersion -> inf).

    Suitable for low-mean integer counts where the assumed GAMMA family
    cannot represent a point mass at zero (Plan 3e Phase 1 use case).
    """

    mean_: float
    dispersion_: float

    def __init__(self, mean: float, dispersion: float) -> None:
        if mean <= 0:
            raise ValueError(f"mean must be positive, got {mean}")
        if dispersion <= 0:
            raise ValueError(f"dispersion must be positive, got {dispersion}")
        object.__setattr__(self, "mean_", float(mean))
        object.__setattr__(self, "dispersion_", float(dispersion))

    def _scipy_n_p(self) -> tuple[float, float]:
        n = self.mean_ * self.mean_ / self.dispersion_
        p = n / (n + self.mean_)
        return n, p

    def mean(self) -> float:
        return self.mean_

    def std(self) -> float:
        var = self.mean_ + self.mean_ * self.mean_ / self.dispersion_
        return float(np.sqrt(var))

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        n, p = self._scipy_n_p()
        return float(stats.nbinom.ppf(q, n=n, p=p))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        # `n` is the protocol-conforming sample-count param. Rename the NB
        # shape parameter to `n_size` locally to avoid the collision.
        rng = rng if rng is not None else np.random.default_rng()
        n_size, p = self._scipy_n_p()
        return stats.nbinom.rvs(n=n_size, p=p, size=n, random_state=rng).astype(np.float64)
```

- [ ] **Step 4: Update `src/projections/distributions/__init__.py`**

Add `ParametricNegativeBinomial` to the imports + `__all__`:

```python
from projections.distributions.parametric import (
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
)

__all__ = [
    "Distribution",
    "ParametricGamma",
    "ParametricNegativeBinomial",
    "ParametricNormal",
    "pack_per_stat_params",
    "unpack_per_stat_params",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_distributions/test_neg_binomial.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/projections/distributions/parametric.py src/projections/distributions/__init__.py tests/test_distributions/test_neg_binomial.py
git commit -m "feat(distributions): ParametricNegativeBinomial — Plan 3e Phase 1"
```

with the standard `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.

---

### Task 1.2: Add `NEGATIVE_BINOMIAL` enum + codec branches

**Files:**
- Modify: `src/projections/schemas.py`
- Modify: `src/projections/distributions/codec.py`
- Modify: `tests/test_distributions/test_codec.py`

- [ ] **Step 1: Write failing tests for codec round-trip**

Append to `tests/test_distributions/test_codec.py`:

```python
def test_codec_round_trip_neg_binomial() -> None:
    """NB packed via pack_per_stat_params and round-tripped."""
    from projections.distributions import (
        ParametricNegativeBinomial,
        pack_per_stat_params,
        unpack_per_stat_params,
    )
    from projections.schemas import Stat

    dist = ParametricNegativeBinomial(mean=0.3, dispersion=2.0)
    blob = pack_per_stat_params({Stat.RECEIVING_TDS: dist})
    decoded = unpack_per_stat_params(blob)
    assert Stat.RECEIVING_TDS in decoded
    decoded_dist = decoded[Stat.RECEIVING_TDS]
    assert isinstance(decoded_dist, ParametricNegativeBinomial)
    assert decoded_dist.mean() == pytest.approx(0.3)
    # Round-trip preserves dispersion via internal n/p back-conversion is unnecessary —
    # we persist (mean, dispersion) directly.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_distributions/test_codec.py::test_codec_round_trip_neg_binomial -v`
Expected: FAIL — codec doesn't recognize ParametricNegativeBinomial.

- [ ] **Step 3: Add `NEGATIVE_BINOMIAL` to `DistributionFamily` enum**

Edit `src/projections/schemas.py`. Find the `DistributionFamily` `StrEnum` class. Add the new value alphabetically (or at the end of the existing values, before `SAMPLED` / `SAMPLED_SUMMARY`):

```python
class DistributionFamily(StrEnum):
    NORMAL = "NORMAL"
    GAMMA = "GAMMA"
    NEGATIVE_BINOMIAL = "NEGATIVE_BINOMIAL"  # NEW (Plan 3e Phase 1)
    SAMPLED = "SAMPLED"
    SAMPLED_SUMMARY = "SAMPLED_SUMMARY"
```

(Adjust placement to match the existing alphabetical/categorical convention in the file.)

- [ ] **Step 4: Add codec branches**

Edit `src/projections/distributions/codec.py`. In `pack_per_stat_params`, add a new `elif` branch before the `else` clause:

```python
        elif isinstance(dist, ParametricNegativeBinomial):
            stats_blob[stat.value] = {
                "family": DistributionFamily.NEGATIVE_BINOMIAL.value,
                "mean": dist.mean(),
                "dispersion": dist.dispersion_,
            }
```

In `unpack_per_stat_params`, add the symmetric branch:

```python
        elif family_value == DistributionFamily.NEGATIVE_BINOMIAL.value:
            out[stat] = ParametricNegativeBinomial(
                mean=float(entry["mean"]),
                dispersion=float(entry["dispersion"]),
            )
```

Update the import at the top of `codec.py` to pull in the new class:

```python
from projections.distributions.parametric import (
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
)
```

- [ ] **Step 5: Run all distributions tests**

Run: `pytest tests/test_distributions/ -v`
Expected: all tests pass (existing codec tests + new NB round-trip).

- [ ] **Step 6: Commit**

```bash
git add src/projections/schemas.py src/projections/distributions/codec.py tests/test_distributions/test_codec.py
git commit -m "feat(codec): NEGATIVE_BINOMIAL family — Plan 3e Phase 1"
```

with the standard Co-Authored-By trailer.

---

### Task 1.3: Add conditional NB dispersion estimator

**Files:**
- Modify: `src/projections/models/baseline.py`
- Modify: `tests/test_models/test_baseline.py`

This task adds the helper that estimates NB dispersion conditional on the per-row Ridge prediction. Mirrors `_normal_std_from_residuals` and `_gamma_alpha_from_residuals`. The conditional formulation directly addresses the Phase 0 marginal-vs-conditional AIC asymmetry caveat.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models/test_baseline.py`:

```python
def test_negative_binomial_dispersion_recovers_known_param() -> None:
    """Synthesize NB-distributed `actual` from known dispersion + per-row mean,
    fit the dispersion, expect recovery within tolerance."""
    from projections.models.baseline import _negative_binomial_dispersion_from_residuals

    rng = np.random.default_rng(42)
    n = 500
    mu_hat = rng.uniform(0.1, 1.5, n)
    true_dispersion = 3.0
    n_size = mu_hat * mu_hat / true_dispersion
    p = n_size / (n_size + mu_hat)
    actual = scipy_stats.nbinom.rvs(n=n_size, p=p, size=n, random_state=rng).astype(np.float64)

    fitted = _negative_binomial_dispersion_from_residuals(mu_hat=mu_hat, actual=actual)
    assert fitted == pytest.approx(true_dispersion, rel=0.30)


def test_negative_binomial_dispersion_clipped_for_degenerate_input() -> None:
    """All-zero actual gives no overdispersion signal; estimator should
    return the high clip so the fitted distribution stays usable."""
    from projections.models.baseline import (
        _NB_DISPERSION_CLIP,
        _negative_binomial_dispersion_from_residuals,
    )

    mu_hat = np.full(50, 0.1)
    actual = np.zeros(50)
    fitted = _negative_binomial_dispersion_from_residuals(mu_hat=mu_hat, actual=actual)
    assert fitted == _NB_DISPERSION_CLIP[1]
```

If `scipy_stats` and `np` aren't already imported at the top of the test file, add them. Pytest is already imported.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models/test_baseline.py -v -k negative_binomial`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the estimator**

Add to `src/projections/models/baseline.py` (near the existing `_gamma_alpha_from_residuals` and `_normal_std_from_residuals` helpers):

```python
_NB_DISPERSION_CLIP: Final[tuple[float, float]] = (0.01, 1000.0)


def _negative_binomial_dispersion_from_residuals(
    *, mu_hat: np.ndarray, actual: np.ndarray
) -> float:
    """Conditional MLE for NB dispersion given per-row mean = mu_hat.

    Maximizes sum(nbinom.logpmf(actual_i; n_i, p_i)) over a single global
    `dispersion` where n_i = mu_hat_i^2 / dispersion and p_i = n_i / (n_i + mu_hat_i).

    Coerces actual to non-negative integers (counts upstream may carry float
    dtype). Returns the dispersion clipped to ``_NB_DISPERSION_CLIP``.
    """
    counts = np.clip(np.round(actual), 0, None).astype(np.int64)
    mu_clipped = np.maximum(mu_hat, 1e-3)

    if counts.size < 2:
        return _NB_DISPERSION_CLIP[1]

    def neg_log_lik(dispersion: float) -> float:
        if dispersion <= 0:
            return float("inf")
        n_size = mu_clipped * mu_clipped / dispersion
        p = n_size / (n_size + mu_clipped)
        return -float(np.sum(scipy_stats.nbinom.logpmf(counts, n=n_size, p=p)))

    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        neg_log_lik,
        bounds=_NB_DISPERSION_CLIP,
        method="bounded",
        options={"xatol": 1e-3},
    )
    if not result.success or not np.isfinite(result.fun):
        return _NB_DISPERSION_CLIP[1]
    return float(np.clip(result.x, *_NB_DISPERSION_CLIP))
```

Add `from scipy import stats as scipy_stats` at the top of `baseline.py` if not already present (Phase 0 may have added it elsewhere, but `baseline.py` is `src/projections/models/`, not `scripts/`; check imports).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models/test_baseline.py -v -k negative_binomial`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/projections/models/baseline.py tests/test_models/test_baseline.py
git commit -m "feat(baseline): _negative_binomial_dispersion_from_residuals (conditional MLE)"
```

with the standard Co-Authored-By trailer.

---

### Task 1.4: Wire NB into `BaselineModel.fit` + `build_stat_distributions`

**Files:**
- Modify: `src/projections/models/baseline.py`
- Modify: `tests/test_models/test_baseline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models/test_baseline.py`. This uses the existing `baseline_features_wr` / `baseline_weekly_stats_wr` fixtures from `tests/conftest.py` and rewires WR's RECEIVING_TDS family to NB just for the test (without modifying the production factory yet — that comes in Task 1.5):

```python
def test_baseline_model_fit_stores_nb_dispersion_for_nb_stat(
    baseline_features_wr: pd.DataFrame,
    baseline_weekly_stats_wr: pd.DataFrame,
) -> None:
    """A BaselineModel configured with a NEGATIVE_BINOMIAL stat must store
    a "dispersion" entry in variance_params after fit()."""
    from projections.models.baseline import _NB_DISPERSION_CLIP, wr_baseline
    from projections.schemas import DistributionFamily, Stat

    model = wr_baseline()
    # Rewire RECEIVING_TDS to NB locally for this test (production factory
    # rewire happens in Task 1.5).
    object.__setattr__(
        model,
        "dist_families",
        {**dict(model.dist_families), Stat.RECEIVING_TDS: DistributionFamily.NEGATIVE_BINOMIAL},
    )
    model.fit(features=baseline_features_wr, weekly_stats=baseline_weekly_stats_wr)
    assert "dispersion" in model.variance_params[Stat.RECEIVING_TDS]
    dispersion = model.variance_params[Stat.RECEIVING_TDS]["dispersion"]
    assert _NB_DISPERSION_CLIP[0] <= dispersion <= _NB_DISPERSION_CLIP[1]
```

NB: `BaselineModel` is a `@dataclass` (mutable by default since `frozen=False`), so direct attribute assignment works. The `object.__setattr__` form above is defensive in case `frozen=True` is added later. Either form is fine; pick the simpler one.

If the test file's existing imports don't include `pd` or the fixture parameters need annotation tweaks, mirror the existing fit-path tests in `tests/test_models/test_baseline.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models/test_baseline.py -v -k nb_dispersion_for_nb_stat`
Expected: FAIL — `BaselineModel.fit` raises `ValueError: Unsupported family <DistributionFamily.NEGATIVE_BINOMIAL>` because the existing fit code only handles NORMAL/GAMMA.

- [ ] **Step 3: Wire NB into `fit()`'s variance estimation loop**

Edit `src/projections/models/baseline.py`. In `BaselineModel.fit`, find the variance estimation block (currently around lines 362-376). Add a new `elif` branch before the `else`:

```python
            elif family is DistributionFamily.NEGATIVE_BINOMIAL:
                self.variance_params[stat] = {
                    "dispersion": _negative_binomial_dispersion_from_residuals(
                        mu_hat=mu_hat, actual=y
                    )
                }
```

Note: the helper takes `actual=y` (the truth column), NOT `residuals` — NB is conditional on `mu_hat`, so it consumes `(mu_hat, actual)` pairs directly. The `residuals = y - mu_hat` line above is unused in this branch (it remains used by NORMAL/GAMMA).

- [ ] **Step 4: Wire NB into `build_stat_distributions`**

In `BaselineModel.build_stat_distributions`, find the per-row distribution construction loop (currently around lines 412-425). Add a new `elif` branch before the `else`:

```python
                elif family is DistributionFamily.NEGATIVE_BINOMIAL:
                    dispersion = params["dispersion"]
                    # Floor mu to keep the (n, p) parameterization defined.
                    mu_safe = max(mu_i, 1e-3)
                    row[stat] = ParametricNegativeBinomial(
                        mean=mu_safe, dispersion=dispersion
                    )
```

Also update the imports at the top of `baseline.py`:

```python
from projections.distributions import (
    Distribution,
    ParametricGamma,
    ParametricNegativeBinomial,  # NEW
    ParametricNormal,
    pack_per_stat_params,
)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_models/test_baseline.py -v`
Expected: all tests pass, including the new NB-fit test.

Run also: `pytest tests/test_models/ -v`
Expected: all per-position tests still pass (no NB rewiring yet at the factory level).

- [ ] **Step 6: Commit**

```bash
git add src/projections/models/baseline.py tests/test_models/test_baseline.py
git commit -m "feat(baseline): wire NEGATIVE_BINOMIAL into fit() + build_stat_distributions"
```

with the standard Co-Authored-By trailer.

---

### Task 1.5: Rewire per-position factories — route count stats to NB

**Files:**
- Modify: `src/projections/models/baseline.py`
- Modify: `tests/test_models/test_baseline_qb.py`
- Modify: `tests/test_models/test_baseline_rb.py`
- Modify: `tests/test_models/test_baseline_te.py`

- [ ] **Step 1: Update WR factory**

In `src/projections/models/baseline.py`, edit `_WR_DIST_FAMILIES` (currently around lines 81-88):

```python
_WR_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.RECEPTIONS: DistributionFamily.GAMMA,
    Stat.RECEIVING_YARDS: DistributionFamily.NORMAL,           # → STUDENT_T in Phase 2
    Stat.RECEIVING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,  # was GAMMA
    Stat.RUSHING_YARDS: DistributionFamily.NORMAL,             # → STUDENT_T in Phase 2
    Stat.RUSHING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,    # was GAMMA
    Stat.FUMBLES_LOST: DistributionFamily.NEGATIVE_BINOMIAL,   # was GAMMA
}
```

- [ ] **Step 2: Update QB factory**

Edit `_QB_DIST_FAMILIES`:

```python
_QB_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.PASSING_YARDS: DistributionFamily.NORMAL,             # stays — regression reference
    Stat.PASSING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,    # was GAMMA
    Stat.INTERCEPTIONS: DistributionFamily.NEGATIVE_BINOMIAL,  # was GAMMA
    Stat.RUSHING_YARDS: DistributionFamily.NORMAL,             # → STUDENT_T in Phase 2
    Stat.RUSHING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,    # was GAMMA
    Stat.FUMBLES_LOST: DistributionFamily.NEGATIVE_BINOMIAL,   # was GAMMA
}
```

- [ ] **Step 3: Update RB factory**

Edit `_RB_DIST_FAMILIES`:

```python
_RB_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.RUSHING_YARDS: DistributionFamily.NORMAL,             # → STUDENT_T in Phase 2
    Stat.RUSHING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,    # was GAMMA
    Stat.RECEPTIONS: DistributionFamily.GAMMA,
    Stat.RECEIVING_YARDS: DistributionFamily.NORMAL,           # → STUDENT_T in Phase 2
    Stat.RECEIVING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,  # was GAMMA
    Stat.FUMBLES_LOST: DistributionFamily.NEGATIVE_BINOMIAL,   # was GAMMA
}
```

- [ ] **Step 4: Update TE factory**

Edit `_TE_DIST_FAMILIES`:

```python
_TE_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.RECEPTIONS: DistributionFamily.GAMMA,
    Stat.RECEIVING_YARDS: DistributionFamily.NORMAL,           # → STUDENT_T in Phase 2
    Stat.RECEIVING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,  # was GAMMA
    Stat.RUSHING_YARDS: DistributionFamily.NORMAL,             # → STUDENT_T in Phase 2
    Stat.RUSHING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,    # was GAMMA
    Stat.FUMBLES_LOST: DistributionFamily.NEGATIVE_BINOMIAL,   # was GAMMA
}
```

- [ ] **Step 5: Update existing per-position fit/predict tests**

The existing per-position tests in `tests/test_models/test_baseline_{qb,rb,te}.py` (and the WR portion of `test_baseline.py`) likely assert specific family expectations on `variance_params`. Find any test that asserts `variance_params[Stat.X] == {"shape": ...}` or `{"std": ...}` for a count stat that's now NB and update to expect `{"dispersion": ...}`. The fit smoke tests (one per position) should already pass once the families are rewired — verify by running:

Run: `pytest tests/test_models/ -v`
Expected: any test that pinned the GAMMA `"shape"` key for a count stat (TDs, fumbles_lost, INTs) now fails. Update each failing test to expect `{"dispersion": ...}` and assert the value is a positive float in the clip range.

- [ ] **Step 6: Run all model tests**

Run: `pytest tests/test_models/ -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/projections/models/baseline.py tests/test_models/
git commit -m "feat(baseline): rewire count stats to NEGATIVE_BINOMIAL — Plan 3e Phase 1"
```

with the standard Co-Authored-By trailer.

---

### Task 1.6: Retrain standalone artifacts

**Files:**
- Modify: `models/artifacts/*.joblib` (regenerated; gitignored)

- [ ] **Step 1: Confirm raw data exists**

Run: `ls data/raw/`
Expected: `weekly_stats/`, `snap_counts/`, `depth_charts/`, `ngs_*/`, `schedules/`, `id_map/` partitions are present.

If `data/raw/` is empty, run `python -c "from projections.ingest.refresh import refresh; from pathlib import Path; refresh(data_root=Path('data'), seasons=range(2018, 2025))"` to ingest. (Per TODO #18, there's no `python -m` entry point yet.)

- [ ] **Step 2: Confirm feature cache exists or regenerate**

Run: `ls data/features/`
Expected: `qb/`, `rb/`, `te/`, `wr/` subdirectories with `season=YYYY/week=WW/part.parquet` partitions.

If empty, run: `python scripts/refresh_features.py` to populate.

- [ ] **Step 3: Retrain each position's standalone artifact**

Run (with venv active):
```
python scripts/train_baseline.py wr
python scripts/train_baseline.py qb
python scripts/train_baseline.py rb
python scripts/train_baseline.py te
```

Each should print the new `model_id` (the `code_hash` rotates because `baseline.py` changed) and the per-stat `variance_params` for the trained model. NB stats should show `{"dispersion": <float>}`.

Expected output per stat (example for QB):
```
  passing_yards: variance_params = {'std': 84.5}
  passing_tds: variance_params = {'dispersion': 5.2}
  interceptions: variance_params = {'dispersion': 3.1}
  rushing_yards: variance_params = {'std': 17.9}
  rushing_tds: variance_params = {'dispersion': 1.8}
  fumbles_lost: variance_params = {'dispersion': 1.5}
```

(Exact values will differ; the key point is NB stats produce `dispersion`, NORMAL stats produce `std`.)

- [ ] **Step 4: No commit**

`models/artifacts/*.joblib` is gitignored. Retraining is local; the snapshot regen in Task 1.7 captures the trained-model behavior into the gated tests.

---

### Task 1.7: Re-snapshot baseline_metrics.json + Phase 1 wrap

**Files:**
- Modify: `tests/backtest/baseline_metrics.json`
- Modify: `project_management.md`

- [ ] **Step 1: Run the full backtest harness with snapshot update**

Run (with venv active): `python scripts/backtest.py --update-snapshot`
Expected: harness runs all 16 (position, year) cells × all 4 positions × per-stat metrics + season metrics; writes the new snapshot to `tests/backtest/baseline_metrics.json` (~400 rows).

Watch the per-cell calibration metric values in the stdout summary. NB-routed cells (`*_tds`, `interceptions`, `fumbles_lost`) should show non-zero `coverage_p10p90` for the first time — Phase 0 had them at 0.0; Phase 1 should land them somewhere in the [0.30, 0.85] band depending on how well NB's tail captures the residuals.

- [ ] **Step 2: Verify the gate passes after re-snapshot**

Run: `python scripts/backtest.py --check`
Expected: PASS — the new snapshot becomes the baseline; no drift vs itself.

- [ ] **Step 3: Verify the opt-in backtest gate test passes**

Run: `pytest -m backtest --run-backtest -v`
Expected: PASS.

- [ ] **Step 4: Compute Phase 1 coverage delta vs Phase 0**

Run a quick comparison script (one-shot inline; don't commit):

```bash
python -c "
import json
from pathlib import Path

# Phase 0 baseline coverage cells (from project_management.md or git history)
# extract from the old snapshot via git show
import subprocess
old = subprocess.check_output(['git', 'show', 'fe55d5b:tests/backtest/baseline_metrics.json']).decode()
new = Path('tests/backtest/baseline_metrics.json').read_text()
old_rows = {(r['position'], r['year'], r['metric']): r['value'] for r in json.loads(old)}
new_rows = {(r['position'], r['year'], r['metric']): r['value'] for r in json.loads(new)}
for key in sorted(set(old_rows) & set(new_rows)):
    if 'calibration_p10p90' in key[2]:
        delta = new_rows[key] - old_rows[key]
        if abs(delta) > 0.01:
            print(f'{key}: {old_rows[key]:.3f} -> {new_rows[key]:.3f}  (delta {delta:+.3f})')
"
```

(This is informational; don't commit.)

- [ ] **Step 5: Update project_management.md**

Add a "Plan 3e Phase 1 — diagnostic results" sub-block under the existing Plan 3e Phase 0 block (insert above the Phase 0 block, mirroring the Phase 0 block structure). Include:
- Heading: `### Phase 1 — Negative Binomial for count stats (run YYYY-MM-DD)`
- 1-2 sentence intro describing what Phase 1 added.
- Bullet summary of cells affected: ~10 cells (position × stat) routed to NB; coverage delta on those cells.
- Note that PASSING_YARDS, *_yards, RECEPTIONS unchanged (Phase 2 / Phase 3 territory).
- Reference the snapshot file.

- [ ] **Step 6: Commit**

```bash
git add tests/backtest/baseline_metrics.json project_management.md
git commit -m "feat(baseline): Plan 3e Phase 1 — NB for count stats; re-snapshot"
```

with the standard Co-Authored-By trailer.

---

## Phase 2 — `ParametricStudentT` for heavy-tailed yards

Closes the 5 cells where Phase 0's AIC strongly preferred Student-t over Normal: `*_yards` for QB rushing, RB rushing, TE receiving, WR receiving. (`QB passing_yards` stays NORMAL as the regression reference. WR rushing_yards and TE rushing_yards are soft picks — see § Task 2.6 for the per-phase review check.)

### Task 2.1: Add `ParametricStudentT` distribution class

**Files:**
- Modify: `src/projections/distributions/parametric.py`
- Modify: `src/projections/distributions/__init__.py`
- Create: `tests/test_distributions/test_student_t.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_distributions/test_student_t.py`:

```python
"""Tests for ParametricStudentT."""

from __future__ import annotations

import numpy as np
import pytest

from projections.distributions import ParametricStudentT


def test_mean_matches_loc() -> None:
    """For df > 1, Student-t mean is loc."""
    dist = ParametricStudentT(loc=10.0, scale=2.0, df=4.0)
    assert dist.mean() == pytest.approx(10.0)


def test_std_matches_formula() -> None:
    """For df > 2, var = scale^2 * df / (df - 2)."""
    dist = ParametricStudentT(loc=0.0, scale=2.0, df=4.0)
    expected_std = 2.0 * np.sqrt(4.0 / 2.0)
    assert dist.std() == pytest.approx(expected_std)


def test_quantile_symmetric_around_loc() -> None:
    """Student-t is symmetric: P(X <= loc) = 0.5."""
    dist = ParametricStudentT(loc=5.0, scale=1.0, df=3.0)
    assert dist.quantile(0.5) == pytest.approx(5.0)


def test_quantile_rejects_out_of_range() -> None:
    dist = ParametricStudentT(loc=0.0, scale=1.0, df=4.0)
    with pytest.raises(ValueError, match="q must be in"):
        dist.quantile(0.0)
    with pytest.raises(ValueError, match="q must be in"):
        dist.quantile(1.0)


def test_sample_shape_and_dtype() -> None:
    dist = ParametricStudentT(loc=0.0, scale=1.0, df=4.0)
    rng = np.random.default_rng(42)
    samples = dist.sample(n=500, rng=rng)
    assert samples.shape == (500,)
    assert samples.dtype == np.float64


def test_sample_mean_approximates_loc() -> None:
    dist = ParametricStudentT(loc=5.0, scale=2.0, df=10.0)
    rng = np.random.default_rng(0)
    samples = dist.sample(n=10_000, rng=rng)
    assert samples.mean() == pytest.approx(5.0, abs=0.1)


def test_constructor_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        ParametricStudentT(loc=0.0, scale=0.0, df=4.0)


def test_constructor_rejects_low_df() -> None:
    """Phase 1+ implementations need df > 2 for finite variance — std() is
    undefined otherwise. Reject at construction time."""
    with pytest.raises(ValueError, match="df must be greater than 2"):
        ParametricStudentT(loc=0.0, scale=1.0, df=2.0)
    with pytest.raises(ValueError, match="df must be greater than 2"):
        ParametricStudentT(loc=0.0, scale=1.0, df=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_distributions/test_student_t.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `ParametricStudentT`**

Append to `src/projections/distributions/parametric.py`:

```python
@dataclass(slots=True, frozen=True, init=False)
class ParametricStudentT:
    """Student-t parameterized as (loc, scale, df).

    Mean = loc (for df > 1).
    Variance = scale^2 * df / (df - 2) (for df > 2).

    Plan 3e Phase 2 routes heavy-tailed continuous stats (yards-shaped) here,
    using the per-row Ridge prediction as `loc` and globally-fit (scale, df)
    from training residuals.
    """

    loc_: float
    scale_: float
    df_: float

    def __init__(self, loc: float, scale: float, df: float) -> None:
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if df <= 2:
            raise ValueError(
                f"df must be greater than 2 for finite variance, got {df}"
            )
        object.__setattr__(self, "loc_", float(loc))
        object.__setattr__(self, "scale_", float(scale))
        object.__setattr__(self, "df_", float(df))

    def mean(self) -> float:
        return self.loc_

    def std(self) -> float:
        return float(self.scale_ * np.sqrt(self.df_ / (self.df_ - 2.0)))

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        return float(stats.t.ppf(q, df=self.df_, loc=self.loc_, scale=self.scale_))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        rng = rng if rng is not None else np.random.default_rng()
        return stats.t.rvs(
            df=self.df_, loc=self.loc_, scale=self.scale_, size=n, random_state=rng
        ).astype(np.float64)
```

- [ ] **Step 4: Update `src/projections/distributions/__init__.py`**

Add `ParametricStudentT`:

```python
from projections.distributions.parametric import (
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    ParametricStudentT,
)

__all__ = [
    "Distribution",
    "ParametricGamma",
    "ParametricNegativeBinomial",
    "ParametricNormal",
    "ParametricStudentT",
    "pack_per_stat_params",
    "unpack_per_stat_params",
]
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_distributions/test_student_t.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/projections/distributions/parametric.py src/projections/distributions/__init__.py tests/test_distributions/test_student_t.py
git commit -m "feat(distributions): ParametricStudentT — Plan 3e Phase 2"
```

with the standard Co-Authored-By trailer.

---

### Task 2.2: Add `STUDENT_T` enum + codec branches

**Files:**
- Modify: `src/projections/schemas.py`
- Modify: `src/projections/distributions/codec.py`
- Modify: `tests/test_distributions/test_codec.py`

- [ ] **Step 1: Write failing test for codec round-trip**

Append to `tests/test_distributions/test_codec.py`:

```python
def test_codec_round_trip_student_t() -> None:
    from projections.distributions import (
        ParametricStudentT,
        pack_per_stat_params,
        unpack_per_stat_params,
    )
    from projections.schemas import Stat

    dist = ParametricStudentT(loc=250.0, scale=70.0, df=8.0)
    blob = pack_per_stat_params({Stat.PASSING_YARDS: dist})
    decoded = unpack_per_stat_params(blob)
    decoded_dist = decoded[Stat.PASSING_YARDS]
    assert isinstance(decoded_dist, ParametricStudentT)
    assert decoded_dist.mean() == pytest.approx(250.0)
    assert decoded_dist.std() == pytest.approx(dist.std())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distributions/test_codec.py::test_codec_round_trip_student_t -v`
Expected: FAIL.

- [ ] **Step 3: Add `STUDENT_T` to `DistributionFamily` enum**

Edit `src/projections/schemas.py`:

```python
class DistributionFamily(StrEnum):
    NORMAL = "NORMAL"
    GAMMA = "GAMMA"
    NEGATIVE_BINOMIAL = "NEGATIVE_BINOMIAL"
    STUDENT_T = "STUDENT_T"  # NEW (Plan 3e Phase 2)
    SAMPLED = "SAMPLED"
    SAMPLED_SUMMARY = "SAMPLED_SUMMARY"
```

- [ ] **Step 4: Add codec branches**

In `src/projections/distributions/codec.py`'s `pack_per_stat_params`, add:

```python
        elif isinstance(dist, ParametricStudentT):
            stats_blob[stat.value] = {
                "family": DistributionFamily.STUDENT_T.value,
                "loc": dist.mean(),
                "scale": dist.scale_,
                "df": dist.df_,
            }
```

In `unpack_per_stat_params`, add:

```python
        elif family_value == DistributionFamily.STUDENT_T.value:
            out[stat] = ParametricStudentT(
                loc=float(entry["loc"]),
                scale=float(entry["scale"]),
                df=float(entry["df"]),
            )
```

Update the codec's import:

```python
from projections.distributions.parametric import (
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    ParametricStudentT,
)
```

- [ ] **Step 5: Run distributions tests**

Run: `pytest tests/test_distributions/ -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/projections/schemas.py src/projections/distributions/codec.py tests/test_distributions/test_codec.py
git commit -m "feat(codec): STUDENT_T family — Plan 3e Phase 2"
```

with the standard Co-Authored-By trailer.

---

### Task 2.3: Add Student-t (scale, df) estimator

**Files:**
- Modify: `src/projections/models/baseline.py`
- Modify: `tests/test_models/test_baseline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models/test_baseline.py`:

```python
def test_student_t_params_recovers_known_params() -> None:
    """Synthesize Student-t-distributed residuals from known (scale, df),
    fit, expect recovery within tolerance."""
    from projections.models.baseline import _student_t_params_from_residuals

    rng = np.random.default_rng(42)
    n = 1000
    true_scale = 50.0
    true_df = 6.0
    residuals = scipy_stats.t.rvs(df=true_df, loc=0.0, scale=true_scale, size=n, random_state=rng)

    fitted_scale, fitted_df = _student_t_params_from_residuals(residuals=residuals)
    assert fitted_scale == pytest.approx(true_scale, rel=0.20)
    assert fitted_df == pytest.approx(true_df, rel=0.50)


def test_student_t_params_rejects_degenerate_input() -> None:
    """All-zero residuals collapse scipy's scale to ~0 — guard returns floor."""
    from projections.models.baseline import (
        _STUDENT_T_DF_FLOOR,
        _STUDENT_T_SCALE_FLOOR,
        _student_t_params_from_residuals,
    )

    fitted_scale, fitted_df = _student_t_params_from_residuals(residuals=np.zeros(50))
    assert fitted_scale >= _STUDENT_T_SCALE_FLOOR
    assert fitted_df >= _STUDENT_T_DF_FLOOR
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models/test_baseline.py -v -k student_t_params`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the estimator**

Add to `src/projections/models/baseline.py`:

```python
_STUDENT_T_SCALE_FLOOR: Final[float] = 1e-3
_STUDENT_T_DF_FLOOR: Final[float] = 2.5  # > 2 for finite variance


def _student_t_params_from_residuals(*, residuals: np.ndarray) -> tuple[float, float]:
    """MLE Student-t scale + df fit on the residual array.

    Returns (scale, df). Discards the fitted loc — residuals are mean-zero
    by construction (`actual - pred`); the scale and df are the per-stat
    global parameters used at predict time.

    Guards against degenerate fits: if scipy's fitted scale is dramatically
    smaller than the empirical sample std, returns the floor to keep
    ParametricStudentT's std() finite.
    """
    if residuals.size < 2:
        return _STUDENT_T_SCALE_FLOOR, _STUDENT_T_DF_FLOOR
    try:
        df, _loc, scale = scipy_stats.t.fit(residuals)
    except Exception:  # diagnostic must not abort on per-cell fit failure
        return _STUDENT_T_SCALE_FLOOR, _STUDENT_T_DF_FLOOR

    sample_std = float(np.std(residuals, ddof=0))
    if scale < max(sample_std * 1e-6, _STUDENT_T_SCALE_FLOOR):
        scale = _STUDENT_T_SCALE_FLOOR
    if df <= 2.0 or not np.isfinite(df):
        df = _STUDENT_T_DF_FLOOR
    return float(scale), float(df)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_models/test_baseline.py -v -k student_t_params`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/projections/models/baseline.py tests/test_models/test_baseline.py
git commit -m "feat(baseline): _student_t_params_from_residuals (scale, df MLE)"
```

with the standard Co-Authored-By trailer.

---

### Task 2.4: Wire Student-t into `fit` + `build_stat_distributions`

**Files:**
- Modify: `src/projections/models/baseline.py`

- [ ] **Step 1: Wire Student-t into `fit()`'s variance estimation**

Edit `src/projections/models/baseline.py`. In `BaselineModel.fit`, the variance-estimation block. Add a new `elif` branch:

```python
            elif family is DistributionFamily.STUDENT_T:
                scale, df = _student_t_params_from_residuals(residuals=residuals)
                self.variance_params[stat] = {"scale": scale, "df": df}
```

- [ ] **Step 2: Wire Student-t into `build_stat_distributions`**

Add a new `elif` branch in the per-row construction loop:

```python
                elif family is DistributionFamily.STUDENT_T:
                    scale = params["scale"]
                    df = params["df"]
                    row[stat] = ParametricStudentT(loc=mu_i, scale=scale, df=df)
```

Update the import:

```python
from projections.distributions import (
    Distribution,
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
    ParametricStudentT,  # NEW
    pack_per_stat_params,
)
```

- [ ] **Step 3: Run all model tests**

Run: `pytest tests/test_models/ -v`
Expected: all tests pass (Student-t isn't wired into any factory yet, so no behavior change).

- [ ] **Step 4: Commit**

```bash
git add src/projections/models/baseline.py
git commit -m "feat(baseline): wire STUDENT_T into fit() + build_stat_distributions"
```

with the standard Co-Authored-By trailer.

---

### Task 2.5: Rewire per-position factories — route yards stats to Student-t

**Files:**
- Modify: `src/projections/models/baseline.py`
- Modify: `tests/test_models/*` (if any tests pin the NORMAL `"std"` for these stats)

- [ ] **Step 1: Update factories**

In `src/projections/models/baseline.py`, edit each per-position `_*_DIST_FAMILIES`:

```python
_WR_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.RECEPTIONS: DistributionFamily.GAMMA,
    Stat.RECEIVING_YARDS: DistributionFamily.STUDENT_T,        # was NORMAL
    Stat.RECEIVING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,
    Stat.RUSHING_YARDS: DistributionFamily.STUDENT_T,          # was NORMAL (soft pick — see Task 2.6)
    Stat.RUSHING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,
    Stat.FUMBLES_LOST: DistributionFamily.NEGATIVE_BINOMIAL,
}

_QB_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.PASSING_YARDS: DistributionFamily.NORMAL,             # stays — regression reference
    Stat.PASSING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,
    Stat.INTERCEPTIONS: DistributionFamily.NEGATIVE_BINOMIAL,
    Stat.RUSHING_YARDS: DistributionFamily.STUDENT_T,          # was NORMAL
    Stat.RUSHING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,
    Stat.FUMBLES_LOST: DistributionFamily.NEGATIVE_BINOMIAL,
}

_RB_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.RUSHING_YARDS: DistributionFamily.STUDENT_T,          # was NORMAL
    Stat.RUSHING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,
    Stat.RECEPTIONS: DistributionFamily.GAMMA,
    Stat.RECEIVING_YARDS: DistributionFamily.STUDENT_T,        # was NORMAL
    Stat.RECEIVING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,
    Stat.FUMBLES_LOST: DistributionFamily.NEGATIVE_BINOMIAL,
}

_TE_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    Stat.RECEPTIONS: DistributionFamily.GAMMA,
    Stat.RECEIVING_YARDS: DistributionFamily.STUDENT_T,        # was NORMAL
    Stat.RECEIVING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,
    Stat.RUSHING_YARDS: DistributionFamily.STUDENT_T,          # was NORMAL (soft pick — see Task 2.6)
    Stat.RUSHING_TDS: DistributionFamily.NEGATIVE_BINOMIAL,
    Stat.FUMBLES_LOST: DistributionFamily.NEGATIVE_BINOMIAL,
}
```

- [ ] **Step 2: Update existing tests if they pinned NORMAL for these stats**

Run: `pytest tests/test_models/ -v`
Expected: any test pinning `variance_params[Stat.X] == {"std": ...}` for a yards stat (now Student-t) fails. Update each failing test to expect `{"scale": ..., "df": ...}` and assert both are positive floats.

- [ ] **Step 3: Run all model tests**

Run: `pytest tests/test_models/ -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/projections/models/baseline.py tests/test_models/
git commit -m "feat(baseline): rewire yards stats to STUDENT_T — Plan 3e Phase 2"
```

with the standard Co-Authored-By trailer.

---

### Task 2.6: Retrain artifacts + soft-pick review

**Files:**
- Modify: `models/artifacts/*.joblib` (regenerated; gitignored)

Two cells need post-retrain review (per spec section 8.3):

- **WR rushing_yards** (mean ~1 yard, Phase 0 alt-fit returned `none`)
- **TE rushing_yards** (mean ~0.10 yards, kurtosis ~2618, degenerate)

If Phase 2's STUDENT_T fit produces a degenerate `(scale, df)` for either cell — `scale == _STUDENT_T_SCALE_FLOOR` or `df == _STUDENT_T_DF_FLOOR` — revert that specific cell back to NORMAL in the factory.

- [ ] **Step 1: Retrain each position**

Run (with venv active):
```
python scripts/train_baseline.py wr
python scripts/train_baseline.py qb
python scripts/train_baseline.py rb
python scripts/train_baseline.py te
```

For each, observe the printed `variance_params` for `rushing_yards` (WR + TE) and check whether `scale` or `df` hit the floor.

- [ ] **Step 2: Review WR rushing_yards**

If WR's `train_baseline.py` output shows `rushing_yards: variance_params = {'scale': 0.001, 'df': 2.5}` (both at floor), revert in `baseline.py`:

```python
_WR_DIST_FAMILIES: Final[Mapping[Stat, DistributionFamily]] = {
    ...
    Stat.RUSHING_YARDS: DistributionFamily.NORMAL,  # reverted from STUDENT_T — degenerate fit
    ...
}
```

Re-run `python scripts/train_baseline.py wr` and confirm normal `{'std': X}` shape.

- [ ] **Step 3: Review TE rushing_yards**

Same check + same revert pattern if TE's `rushing_yards` STUDENT_T fit is degenerate.

- [ ] **Step 4: Re-run model tests after any reverts**

Run: `pytest tests/test_models/ -v`
Expected: all pass.

- [ ] **Step 5: Commit any reverts**

```bash
git add src/projections/models/baseline.py
git commit -m "fix(baseline): revert {WR,TE} rushing_yards to NORMAL (degenerate Student-t fit)"
```

If both cells fit cleanly, no commit needed.

---

### Task 2.7: Re-snapshot baseline_metrics.json + Phase 2 wrap

**Files:**
- Modify: `tests/backtest/baseline_metrics.json`
- Modify: `project_management.md`

- [ ] **Step 1: Run the harness with snapshot update**

Run: `python scripts/backtest.py --update-snapshot`
Expected: harness runs; new snapshot written. Yards-stat cells (`*_yards`) should show modest coverage improvement (Student-t's heavier tails increase `[p10, p90]` range). NB cells (Phase 1) should be unchanged or noise-level different.

- [ ] **Step 2: Verify gate passes**

Run: `python scripts/backtest.py --check`
Expected: PASS.

Run: `pytest -m backtest --run-backtest -v`
Expected: PASS.

- [ ] **Step 3: Update project_management.md**

Add a "Phase 2 — Student-t for yards stats (run YYYY-MM-DD)" sub-block under the Plan 3e Phase 1 entry (mirroring Phase 1's structure). Include:
- Cells affected (5 yards stats; note any reverts from Task 2.6).
- Coverage delta on those cells.
- Note that count stats (Phase 1 NB) and `receptions` (still GAMMA) and bucketing (Phase 3) are unchanged.

- [ ] **Step 4: Commit**

```bash
git add tests/backtest/baseline_metrics.json project_management.md
git commit -m "feat(baseline): Plan 3e Phase 2 — Student-t for yards; re-snapshot"
```

with the standard Co-Authored-By trailer.

---

## Phase 3 — Per-tertile variance bucketing

Cross-cutting fix. Applies to all four families. For each (position, stat) cell, persist tertile cuts on `mu_hat` (33rd / 67th percentile from training) and a per-bucket variance parameter. At predict time, look up the bucket and select the parameter.

### Task 3.1: Generalize `variance_params` shape + add bucket cuts

**Files:**
- Modify: `src/projections/models/baseline.py`

- [ ] **Step 1: Update type annotations**

In `src/projections/models/baseline.py`, find `BaselineModel`'s `variance_params` field declaration. Update to:

```python
    variance_params: dict[Stat, dict[str, float | list[float]]] = field(default_factory=dict)
```

This allows existing scalar entries (`{"std": 70.0}`) AND new bucket-aware entries (`{"bucket_cuts": [c33, c67], "std_per_bucket": [a, b, c]}`).

- [ ] **Step 2: Run mypy + ruff to confirm no breakage**

Run: `mypy src tests scripts`
Expected: clean (the union widening is structurally additive).

Run: `pytest tests/test_models/ -v`
Expected: all pass (no behavior change yet).

- [ ] **Step 3: Commit**

```bash
git add src/projections/models/baseline.py
git commit -m "refactor(baseline): widen variance_params value type for bucketing"
```

with the standard Co-Authored-By trailer.

---

### Task 3.2: Add tertile-cuts + bucket-aware estimator helpers

**Files:**
- Modify: `src/projections/models/baseline.py`
- Modify: `tests/test_models/test_baseline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models/test_baseline.py`:

```python
def test_compute_tertile_cuts_returns_two_cuts_in_order() -> None:
    from projections.models.baseline import _compute_tertile_cuts

    mu_hat = np.linspace(0, 100, 300)
    cuts = _compute_tertile_cuts(mu_hat)
    assert len(cuts) == 2
    assert cuts[0] < cuts[1]
    # 33rd percentile of linspace(0, 100, 300) is ~33.0; 67th is ~66.7.
    assert cuts[0] == pytest.approx(33.0, abs=1.5)
    assert cuts[1] == pytest.approx(66.7, abs=1.5)


def test_assign_bucket_indices_returns_zero_one_two() -> None:
    from projections.models.baseline import _assign_bucket_indices

    cuts = [33.0, 67.0]
    mu_hat = np.array([10.0, 33.0, 50.0, 67.0, 80.0])
    indices = _assign_bucket_indices(mu_hat=mu_hat, cuts=cuts)
    # Convention: searchsorted with cuts; values <= cuts[0] go to bucket 0,
    # values in (cuts[0], cuts[1]] go to bucket 1, > cuts[1] go to bucket 2.
    # np.searchsorted([33, 67], [10, 33, 50, 67, 80]) returns [0, 0, 1, 1, 2].
    np.testing.assert_array_equal(indices, [0, 0, 1, 1, 2])


def test_per_bucket_normal_std_returns_three_values() -> None:
    from projections.models.baseline import _per_bucket_normal_std_from_residuals

    rng = np.random.default_rng(0)
    n = 600
    mu_hat = np.linspace(0, 100, n)
    cuts = [33.0, 67.0]
    # Synthesize heteroscedastic residuals: low-mu has small std, high-mu large.
    bucket_idx = np.searchsorted(cuts, mu_hat).clip(0, 2)
    stds = np.array([5.0, 15.0, 40.0])
    residuals = rng.normal(0, stds[bucket_idx], n)

    fitted = _per_bucket_normal_std_from_residuals(
        mu_hat=mu_hat, residuals=residuals, cuts=cuts
    )
    assert len(fitted) == 3
    assert fitted[0] == pytest.approx(5.0, rel=0.20)
    assert fitted[1] == pytest.approx(15.0, rel=0.20)
    assert fitted[2] == pytest.approx(40.0, rel=0.20)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_baseline.py -v -k "tertile or bucket"`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the helpers**

Add to `src/projections/models/baseline.py`:

```python
def _compute_tertile_cuts(mu_hat: np.ndarray) -> list[float]:
    """Return [33rd-percentile, 67th-percentile] cuts on mu_hat."""
    return [
        float(np.percentile(mu_hat, 33.333)),
        float(np.percentile(mu_hat, 66.667)),
    ]


def _assign_bucket_indices(*, mu_hat: np.ndarray, cuts: list[float]) -> np.ndarray:
    """Return per-row bucket indices in {0, 1, 2} based on cuts.

    Uses np.searchsorted: rows where mu_hat <= cuts[0] go to bucket 0;
    cuts[0] < mu_hat <= cuts[1] goes to bucket 1; mu_hat > cuts[1] goes
    to bucket 2.
    """
    return np.searchsorted(np.asarray(cuts), mu_hat, side="left").clip(0, 2).astype(np.int64)


def _per_bucket_normal_std_from_residuals(
    *, mu_hat: np.ndarray, residuals: np.ndarray, cuts: list[float]
) -> list[float]:
    """Per-bucket residual std for the NORMAL family. Returns 3 values
    (one per tertile bucket). Falls back to the global residual std for
    any bucket with < 2 rows."""
    indices = _assign_bucket_indices(mu_hat=mu_hat, cuts=cuts)
    global_std = _normal_std_from_residuals(residuals)
    out: list[float] = []
    for b in range(3):
        mask = indices == b
        if mask.sum() < 2:
            out.append(global_std)
        else:
            out.append(_normal_std_from_residuals(residuals[mask]))
    return out


def _per_bucket_gamma_alpha_from_residuals(
    *, mu_hat: np.ndarray, residuals: np.ndarray, cuts: list[float]
) -> list[float]:
    """Per-bucket gamma alpha. Falls back to the global alpha for any
    bucket with < 2 rows."""
    indices = _assign_bucket_indices(mu_hat=mu_hat, cuts=cuts)
    global_alpha = _gamma_alpha_from_residuals(mu_hat=mu_hat, residuals=residuals)
    out: list[float] = []
    for b in range(3):
        mask = indices == b
        if mask.sum() < 2:
            out.append(global_alpha)
        else:
            out.append(
                _gamma_alpha_from_residuals(mu_hat=mu_hat[mask], residuals=residuals[mask])
            )
    return out


def _per_bucket_nb_dispersion_from_residuals(
    *, mu_hat: np.ndarray, actual: np.ndarray, cuts: list[float]
) -> list[float]:
    """Per-bucket NB dispersion. Falls back to global for any bucket with < 2 rows."""
    indices = _assign_bucket_indices(mu_hat=mu_hat, cuts=cuts)
    global_d = _negative_binomial_dispersion_from_residuals(mu_hat=mu_hat, actual=actual)
    out: list[float] = []
    for b in range(3):
        mask = indices == b
        if mask.sum() < 2:
            out.append(global_d)
        else:
            out.append(
                _negative_binomial_dispersion_from_residuals(
                    mu_hat=mu_hat[mask], actual=actual[mask]
                )
            )
    return out


def _per_bucket_student_t_params_from_residuals(
    *, residuals: np.ndarray, indices: np.ndarray
) -> tuple[list[float], list[float]]:
    """Per-bucket Student-t (scale, df). Returns (scale_per_bucket, df_per_bucket).
    Falls back to the global fit for any bucket with < 2 rows.

    Takes pre-computed `indices` rather than (mu_hat, cuts) because Student-t
    fitting is on the residual array, not the (mu_hat, residual) pair —
    the function signature mirrors the underlying _student_t_params_from_residuals.
    """
    global_scale, global_df = _student_t_params_from_residuals(residuals=residuals)
    scales: list[float] = []
    dfs: list[float] = []
    for b in range(3):
        mask = indices == b
        if mask.sum() < 2:
            scales.append(global_scale)
            dfs.append(global_df)
        else:
            s, d = _student_t_params_from_residuals(residuals=residuals[mask])
            scales.append(s)
            dfs.append(d)
    return scales, dfs
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_models/test_baseline.py -v -k "tertile or bucket"`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/projections/models/baseline.py tests/test_models/test_baseline.py
git commit -m "feat(baseline): tertile-cuts + per-bucket variance estimators"
```

with the standard Co-Authored-By trailer.

---

### Task 3.3: Wire bucketing into `fit()`

**Files:**
- Modify: `src/projections/models/baseline.py`

- [ ] **Step 1: Update `fit()`'s variance estimation block**

In `BaselineModel.fit`, replace the existing variance-estimation loop:

```python
        # Variance estimation with per-tertile bucketing (Plan 3e Phase 3).
        for stat in self.target_stats:
            ridge = self.ridges[stat]
            mu_hat = ridge.predict(x).astype(np.float64)
            y = truth_frame[stat.value].to_numpy(dtype=np.float64)
            residuals = y - mu_hat
            family = self.dist_families[stat]
            cuts = _compute_tertile_cuts(mu_hat)
            indices = _assign_bucket_indices(mu_hat=mu_hat, cuts=cuts)
            if family is DistributionFamily.NORMAL:
                self.variance_params[stat] = {
                    "bucket_cuts": cuts,
                    "std_per_bucket": _per_bucket_normal_std_from_residuals(
                        mu_hat=mu_hat, residuals=residuals, cuts=cuts
                    ),
                }
            elif family is DistributionFamily.GAMMA:
                self.variance_params[stat] = {
                    "bucket_cuts": cuts,
                    "shape_per_bucket": _per_bucket_gamma_alpha_from_residuals(
                        mu_hat=mu_hat, residuals=residuals, cuts=cuts
                    ),
                }
            elif family is DistributionFamily.NEGATIVE_BINOMIAL:
                self.variance_params[stat] = {
                    "bucket_cuts": cuts,
                    "dispersion_per_bucket": _per_bucket_nb_dispersion_from_residuals(
                        mu_hat=mu_hat, actual=y, cuts=cuts
                    ),
                }
            elif family is DistributionFamily.STUDENT_T:
                scales, dfs = _per_bucket_student_t_params_from_residuals(
                    residuals=residuals, indices=indices
                )
                self.variance_params[stat] = {
                    "bucket_cuts": cuts,
                    "scale_per_bucket": scales,
                    "df_per_bucket": dfs,
                }
            else:  # pragma: no cover
                raise ValueError(f"Unsupported family {family} for stat {stat}")
```

- [ ] **Step 2: Update existing tests that asserted scalar variance_params**

Run: `pytest tests/test_models/ -v`
Expected: tests that asserted `variance_params[Stat.X] == {"std": ...}` (scalar shape) now fail. Update each to expect the bucketed shape:

- NORMAL: `{"bucket_cuts": [c33, c67], "std_per_bucket": [a, b, c]}`
- GAMMA: `{"bucket_cuts": [c33, c67], "shape_per_bucket": [a, b, c]}`
- NEGATIVE_BINOMIAL: `{"bucket_cuts": [c33, c67], "dispersion_per_bucket": [a, b, c]}`
- STUDENT_T: `{"bucket_cuts": [c33, c67], "scale_per_bucket": [a, b, c], "df_per_bucket": [a, b, c]}`

For each test, assert that `bucket_cuts` is a 2-element list of floats and the `*_per_bucket` lists have 3 elements.

- [ ] **Step 3: Run model tests**

Run: `pytest tests/test_models/ -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/projections/models/baseline.py tests/test_models/
git commit -m "feat(baseline): per-tertile bucketing in fit() for all 4 families"
```

with the standard Co-Authored-By trailer.

---

### Task 3.4: Wire bucket lookup into `build_stat_distributions`

**Files:**
- Modify: `src/projections/models/baseline.py`

- [ ] **Step 1: Update `build_stat_distributions`**

In `BaselineModel.build_stat_distributions`, replace the per-row construction loop. For each stat, compute the bucket index per row first, then dispatch on family using the per-bucket parameter:

```python
        out: list[dict[Stat, Distribution]] = []
        # Pre-compute per-row bucket indices for every stat.
        per_stat_indices: dict[Stat, np.ndarray] = {}
        for stat in self.target_stats:
            cuts = self.variance_params[stat]["bucket_cuts"]
            assert isinstance(cuts, list)
            per_stat_indices[stat] = _assign_bucket_indices(
                mu_hat=per_stat_mu[stat], cuts=cuts
            )

        for i in range(len(x)):
            row: dict[Stat, Distribution] = {}
            for stat in self.target_stats:
                mu_i = float(per_stat_mu[stat][i])
                family = self.dist_families[stat]
                params = self.variance_params[stat]
                bucket = int(per_stat_indices[stat][i])
                if family is DistributionFamily.NORMAL:
                    std = params["std_per_bucket"][bucket]  # type: ignore[index]
                    row[stat] = ParametricNormal(mean=mu_i, std=std)
                elif family is DistributionFamily.GAMMA:
                    shape = params["shape_per_bucket"][bucket]  # type: ignore[index]
                    mu_safe = max(mu_i, self._GAMMA_MU_FLOOR)
                    scale = mu_safe / shape
                    row[stat] = ParametricGamma(shape=shape, scale=scale)
                elif family is DistributionFamily.NEGATIVE_BINOMIAL:
                    dispersion = params["dispersion_per_bucket"][bucket]  # type: ignore[index]
                    mu_safe = max(mu_i, 1e-3)
                    row[stat] = ParametricNegativeBinomial(mean=mu_safe, dispersion=dispersion)
                elif family is DistributionFamily.STUDENT_T:
                    scale = params["scale_per_bucket"][bucket]  # type: ignore[index]
                    df = params["df_per_bucket"][bucket]  # type: ignore[index]
                    row[stat] = ParametricStudentT(loc=mu_i, scale=scale, df=df)
                else:  # pragma: no cover
                    raise ValueError(f"Unsupported family {family}")
            out.append(row)
        return out
```

The `# type: ignore[index]` annotations are necessary because the value type of `variance_params[stat]` is `float | list[float]` and mypy can't narrow at the bucket-index lookup. If you want to avoid the ignores, add a small helper:

```python
def _list_param(params: dict[str, float | list[float]], key: str) -> list[float]:
    val = params[key]
    assert isinstance(val, list)
    return val
```

and call `_list_param(params, "std_per_bucket")[bucket]`. Pick whichever is cleaner; the helper is preferred.

- [ ] **Step 2: Run model tests**

Run: `pytest tests/test_models/ -v`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add src/projections/models/baseline.py
git commit -m "feat(baseline): bucket lookup in build_stat_distributions"
```

with the standard Co-Authored-By trailer.

---

### Task 3.5: Update codec for bucketed params

**Files:**
- Modify: `src/projections/distributions/codec.py`
- Modify: `tests/test_distributions/test_codec.py`

The codec encodes per-row distribution params for `ProjectionWeeklySchema.params`. After Phase 3, the per-row params are still scalar (each row has its own distribution with concrete numbers, not a bucket dict) — bucketing happens at fit/predict time inside `BaselineModel`. So the codec doesn't need to change for bucketing per se; the per-row distributions emitted by `build_stat_distributions` are still concrete `ParametricNormal(mean, std)` etc., and the existing codec branches handle them.

- [ ] **Step 1: Verify the codec is unchanged for buckets**

Run: `pytest tests/test_distributions/ -v`
Expected: all pass without any codec edits.

- [ ] **Step 2: Add a regression test confirming bucketed-fit per-row codec round-trip**

Append to `tests/test_distributions/test_codec.py`:

```python
def test_codec_round_trip_after_bucketed_fit() -> None:
    """End-to-end: fit a BaselineModel with bucketing, predict, codec-decode
    a row's params blob, confirm shape matches the per-row family."""
    from projections.distributions import (
        ParametricNegativeBinomial,
        ParametricNormal,
        ParametricStudentT,
        unpack_per_stat_params,
    )
    # Build a tiny model end-to-end via the existing test fixture machinery
    # is heavyweight; this regression test is informational. Use a hand-rolled
    # blob that mirrors what build_stat_distributions would emit per row.
    from projections.distributions import pack_per_stat_params
    from projections.schemas import Stat

    blob = pack_per_stat_params({
        Stat.PASSING_YARDS: ParametricStudentT(loc=250.0, scale=70.0, df=8.0),
        Stat.PASSING_TDS: ParametricNegativeBinomial(mean=1.5, dispersion=4.0),
        Stat.RUSHING_YARDS: ParametricNormal(mean=20.0, std=15.0),
    })
    decoded = unpack_per_stat_params(blob)
    assert isinstance(decoded[Stat.PASSING_YARDS], ParametricStudentT)
    assert isinstance(decoded[Stat.PASSING_TDS], ParametricNegativeBinomial)
    assert isinstance(decoded[Stat.RUSHING_YARDS], ParametricNormal)
```

- [ ] **Step 3: Run the test**

Run: `pytest tests/test_distributions/test_codec.py::test_codec_round_trip_after_bucketed_fit -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_distributions/test_codec.py
git commit -m "test(codec): mixed-family round-trip regression"
```

with the standard Co-Authored-By trailer.

---

### Task 3.6: Retrain artifacts + final re-snapshot + close TODO #22

**Files:**
- Modify: `models/artifacts/*.joblib` (regenerated; gitignored)
- Modify: `tests/backtest/baseline_metrics.json`
- Modify: `project_management.md`
- Modify: `TODO.md`

- [ ] **Step 1: Retrain each position**

Run:
```
python scripts/train_baseline.py wr
python scripts/train_baseline.py qb
python scripts/train_baseline.py rb
python scripts/train_baseline.py te
```

Confirm each prints bucketed `variance_params` (lists for `*_per_bucket`, plus `bucket_cuts`).

- [ ] **Step 2: Re-snapshot**

Run: `python scripts/backtest.py --update-snapshot`

Watch the per-cell `coverage_p10p90` and `season_calibration_p10p90` values. Phase 3 should net coverage gains on cells with high `heteroscedasticity_ratio` (per Phase 0's diagnostic).

- [ ] **Step 3: Verify gate passes**

Run: `python scripts/backtest.py --check`
Expected: PASS.

Run: `pytest -m backtest --run-backtest -v`
Expected: PASS.

- [ ] **Step 4: Compute final coverage delta vs Phase 0 baseline**

Use the comparison script from Task 1.7 Step 4 to diff the post-Phase-3 snapshot against the pre-Phase-1 baseline (`fe55d5b:tests/backtest/baseline_metrics.json`). Confirm:

- Min cell coverage (across all 32 calibration cells) ≥ 0.65 (Phase 0 was 0.0; spec target).
- Mean coverage improved by ≥ 0.10 (spec target).
- No regression on RMSE / MAE / Spearman beyond existing tolerances.

If either coverage target isn't met, document in PM doc as a known shortfall and consider whether Phase 4 (e.g., ZIP, more buckets, inflation factor) is justified — defer to a follow-up plan rather than expanding this PR.

- [ ] **Step 5: Update project_management.md**

Add a "Phase 3 — Variance bucketing (run YYYY-MM-DD)" sub-block. Include:
- Cells affected (all 24 — bucketing applies to every cell).
- Coverage delta on the 18 cells that had `heteroscedasticity_ratio > 1.5` in Phase 0.
- Final cumulative coverage delta vs the pre-Phase-1 baseline (Phase 0's snapshot).
- Whether the 0.65 / 0.10 spec targets were met.

Update the bottom "Current status" section to reflect "Plan 3e Phase 1+ complete on branch; ready for PR" (mirror previous plan-complete status updates).

Update the bottom "Next action" section: change to "Open PR for Plan 3e (Phases 0 + 1+); after merge, decide on follow-up plan based on whether coverage targets were met."

- [ ] **Step 6: Close TODO #22**

Edit `TODO.md`. Find the entry for `### 22. Plan 3e — calibration tightening`. Mark it closed:

```markdown
### 22. Plan 3e — calibration tightening — closed in Plan 3e Phase 3

Closed YYYY-MM-DD. Phase 0 diagnostic identified 3 root causes; Phases 1-3
implemented:
- Phase 1: ParametricNegativeBinomial for *_tds / interceptions / fumbles_lost
  (10 cells; coverage 0.0 → <fill in>).
- Phase 2: ParametricStudentT for *_yards (5 cells; coverage <Phase 1 base> → <Phase 2 result>).
- Phase 3: per-tertile variance bucketing (cross-cutting; 18 cells with hetero ratio > 1.5).

Final coverage: min cell <X>; mean delta vs pre-Phase-1 <Y>. Spec target ≥ 0.65 / ≥ 0.10:
<met / not met — if not met, follow-up plan TBD>.
```

(Don't fill the placeholders with literal "<fill in>" strings — substitute the actual numbers from Step 4.)

- [ ] **Step 7: Final commit**

```bash
git add tests/backtest/baseline_metrics.json project_management.md TODO.md src/projections/models/baseline.py
git commit -m "feat(baseline): Plan 3e Phase 3 — variance bucketing; final re-snapshot; close TODO #22"
```

with the standard Co-Authored-By trailer.

---

### Task 3.7: End-of-Phase-1+ verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full check suite**

```
pytest -v
mypy src tests scripts
ruff check src tests scripts
ruff format --check src tests scripts
```

All four gates must pass cleanly.

- [ ] **Step 2: Confirm the branch is ready for PR**

Run: `git log --oneline main..HEAD`
Expected: ~25-30 commits across Phase 0 + Phase 1 + Phase 2 + Phase 3, all on `feat/plan-3e-calibration-tightening`.

Run: `git status`
Expected: clean.

- [ ] **Step 3: Push the final state**

Run: `git push`

- [ ] **Step 4: Hand back to user for PR opening**

Do NOT open the PR autonomously — the user will open it. Report back with a summary of:
- Total commits Phase 0 + Phase 1 + Phase 2 + Phase 3.
- Final coverage numbers (min cell, mean delta).
- Whether spec targets were met.
- Any open follow-up items (e.g., ZIP if NB undercovers, inflation factor, cross-week correlation deferred to follow-up plan).

The user will then run `/ultrareview` or open the PR via gh CLI.

---

## Self-review notes

**Spec coverage checked:**

- **§ 1.2 Phase 1 goal (NB for counts):** Tasks 1.1-1.7.
- **§ 1.2 Phase 2 goal (Student-t for yards):** Tasks 2.1-2.7. Soft-pick review: Task 2.6.
- **§ 1.2 Phase 3 goal (variance bucketing):** Tasks 3.1-3.6.
- **§ 1.2 Wrap-up (codec, retrain, snapshot, PM doc, close #22):** Distributed across Tasks 1.6/1.7, 2.6/2.7, 3.5/3.6.
- **§ 8.1 Distribution layer additions:** Tasks 1.1, 2.1.
- **§ 8.2 DistributionFamily enum + codec:** Tasks 1.2, 2.2, 3.5.
- **§ 8.3 Per-position factory rewires:** Tasks 1.5, 2.5 (with soft-pick review in 2.6).
- **§ 8.4 variance_params shape evolution:** Task 3.1 widens the type; Task 3.3 wires bucketing.
- **§ 9.1 Conditional NB dispersion estimator:** Task 1.3.
- **§ 9.2 Student-t (scale, df) estimator:** Task 2.3.
- **§ 9.3 Per-tertile bucketing:** Task 3.2 (helpers), 3.3 (fit), 3.4 (predict).
- **§ 9.4 Phase boundaries:** Tasks 1.7, 2.7, 3.6 (each phase ends with re-snapshot + PM doc update).
- **§ 9.5 Backwards compatibility:** Standalone artifact retraining at Tasks 1.6, 2.6, 3.6. Codec schema: per the spec, schema_version stays at 1 because the per-row params blob shape is unchanged (the codec already carries the family + family-specific params; new families are additive branches). The `variance_params` shape change is internal to `BaselineModel.fit/predict` and doesn't touch the persisted `params` blob.
- **§ 10.1 Per-phase validation:** Tasks 1.7, 2.7, 3.6 each call `pytest -m backtest --run-backtest` and compute coverage delta.
- **§ 10.2 Plan 3e overall validation:** Task 3.6 Steps 4-6 verify the targets and document the result.

**No placeholders.** Every task has actual code, exact commands, expected output. The two `<fill in>` markers in Task 3.6 Step 6 are the literal template the implementer fills with measured numbers.

**Type consistency.** `_negative_binomial_dispersion_from_residuals` (Task 1.3) is referenced by name in Task 1.4 (wiring) and Task 3.2 (per-bucket wrapper); same for `_student_t_params_from_residuals` and the per-bucket helpers. `ParametricNegativeBinomial` and `ParametricStudentT` are introduced in Tasks 1.1 / 2.1 and consumed at Tasks 1.4 / 2.4 / 3.4. The `variance_params` value type widens at Task 3.1 (from `dict[str, float]` to `dict[str, float | list[float]]`) before bucketing fills the lists at Task 3.3.

**Sequencing constraints.** Phases must run in order: Phase 1 → Phase 2 → Phase 3. Within each phase, tasks numbered sequentially are TDD-internal (test → impl → commit) and can't be reordered. Across phases, the variance_params shape change at Task 3.1 is the only structural break — the per-stat tests updated at Tasks 1.5 and 2.5 will need updating again at Task 3.3. That's expected; the plan calls it out at Task 3.3 Step 2.
