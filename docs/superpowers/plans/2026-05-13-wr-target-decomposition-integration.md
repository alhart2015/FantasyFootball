# WR Target Decomposition Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `DecomposedBaselineModel` as a peer to `BaselineModel` with per-stat decomposition opt-in. Register `wr_decomposed_baseline` factory for receptions-only decomposition (volume = `targets`, efficiency = `catch_rate`). Run binding adoption gate vs `EnsembleModel` (WR's current production) + informational gate vs `BaselineModel`. Ship per §1.3.5: ADOPT flips `_PositionDispatch[WR].default_model_class`; MARGINAL/DO_NOT_ADOPT ships infra-only; REGRESSION full-reverts.

**Architecture:** New `DecomposedBaselineModel` (subclass of `BaselineModel`) with constructor arg `decomposed_stats: Mapping[Stat, DecompositionSpec]`. Direct ridges for non-decomposed stats (unchanged); shared volume RidgeCV + per-stat efficiency RidgeCV for decomposed stats. New `FrozenSampledDistribution` carries per-row composed samples through `score_distribution` (its `.sample(n)` returns `self.samples` directly when `n == len`, preserving within-row cross-stat correlation). Persistence converts `FrozenSampledDistribution` → `QuantileDistribution` for the existing `QUANTILE` codec branch — no codec edits. WR factory configured for receptions only; the opt-in architecture supports v2 expansion to receiving_yards/receiving_tds without re-architecting.

**Tech Stack:** Python 3.11, pandas (pyarrow strings + nullable Int64/Float64), pandera (`DataFrameModel` + `strict="filter"`), scikit-learn (`RidgeCV`), numpy, pytest, mypy strict, ruff. Pre-commit hooks (ruff format, mypy strict). Reuses `projections.distributions.{Distribution, QuantileDistribution, pack_per_stat_params}`, `projections.scoring.{derive_row_seed, score_distribution}`, `projections.backtest.harness.run_backtest`, `scripts/adoption_gate.py`'s `--run` single-run mode (Plan 8 mechanic).

**Spec:** `docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md` (committed at `27ee83d`).

**Branch:** `feat/wr-target-decomposition` (worktree at `.claude/worktrees/feat+wr-target-decomposition/`).

**Python invocation convention.** All Python invocations from the worktree use:

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m <module> <args>
```

The `PYTHONPATH=src` prefix routes the worktree's `src/projections/` ahead of the main repo's editable-installed `src/projections/` on `sys.path`, so the worktree's source is the one being tested. The relative path `../../../.venv/Scripts/python.exe` walks from the worktree at `.claude/worktrees/<name>/` back to the main repo's `.venv/`. **Every shell command in this plan uses this exact prefix.**

---

## File Structure (decomposition lock-in)

**Create:**
- `src/projections/distributions/sampled.py` — net-new module hosting `FrozenSampledDistribution`. ~40 LOC.
- `tests/test_distributions/test_sampled.py` — tests for `FrozenSampledDistribution`.
- `src/projections/models/decomposed_baseline.py` — net-new module hosting `DecompositionSpec`, `DecomposedBaselineModel`, `wr_decomposed_baseline`. ~250 LOC.
- `tests/test_models/test_decomposed_baseline.py` — tests for `DecomposedBaselineModel`.

**Modify:**
- `src/projections/distributions/__init__.py` — export `FrozenSampledDistribution`.
- `src/projections/models/baseline.py` — extract one tiny hook method `_persistable_dists_for_packing(stat_dists)` from `predict_distribution`'s pack step. Default returns `stat_dists` unchanged. `DecomposedBaselineModel` overrides to convert `FrozenSampledDistribution` → `QuantileDistribution` before persistence.
- `src/projections/models/__init__.py` — import `DecomposedBaselineModel`, `wr_decomposed_baseline`; add `"decomposed-baseline": wr_decomposed_baseline` to `_WR_FACTORIES`; update `__all__`.
- `scripts/backtest.py` — add `"decomposed-baseline"` to `--model` argparse `choices`. One-line change so the CLI can target the new class.
- (Conditional on §1.3.5 outcome) `src/projections/models/__init__.py` again — flip `_PositionDispatch[Position.WR].default_model_class` from `"ensemble"` to `"decomposed-baseline"` on ADOPT; revert on REGRESSION; no change on MARGINAL/DO_NOT_ADOPT.
- `project_management.md` — top decision-log entry (Phase 6).
- `TODO.md` — Update under #23 (Phase 6).

**Generated at run time (Phase 5–6, committed where committable):**
- `reports/adoption_gate_wr_decomposed_baseline_vs_ensemble.{md,csv}` — binding cell verdict.
- `reports/adoption_gate_wr_decomposed_baseline_vs_baseline.{md,csv}` — informational cell verdict.
- `reports/wr_target_decomposition_summary.md` — decision log + per-cell tables + probe-vs-gate calibration + §1.3.5 outcome narrative.

**Untouched (deliberately):**
- `src/projections/schemas.py` — no new schemas; `WrFeaturesSchema`, `WeeklyStatsSchema`, `ProjectionWeeklySchema` cover everything.
- `src/projections/distributions/codec.py` — `QUANTILE` branch already exists.
- `src/projections/scoring/score_distribution.py` — `SampledDistribution` unchanged; `FrozenSampledDistribution` lives in `distributions/sampled.py`, not here.
- `src/projections/features/*` — no feature changes.
- `scripts/refresh_features.py`, `scripts/_run_single_backtest.py`, `scripts/backtest_dual.py`, `scripts/adoption_gate.py` — all consumed as-is; Phase 5 uses `run_backtest` Python API directly + `adoption_gate.py --run` single-run mode.
- `data/ensemble_weights/` — existing ensemble weights stay in tree regardless of §1.3.5 outcome.
- `tests/conftest.py`, `tests/test_features/conftest.py` — no fixture changes; `BaselineModel` fixtures already cover the train-window requirements.
- `tests/backtest/model_metrics.json` — backtest snapshot is NOT updated in this PR. Updating snapshot is conditional on routing flip and handled by `--update-snapshot` at the executor's discretion in Phase 6.

---

## Phase 1 — `FrozenSampledDistribution`

Goal: net-new Distribution class whose `.sample(n)` returns `self.samples` verbatim when `n == len(self.samples)`, enabling within-row cross-stat correlation preservation when `score_distribution` calls `.sample(n_samples)` on multiple decomposed-stat distributions. Foundation for Phase 3.

### Task 1: Create `FrozenSampledDistribution`

**Files:**
- Create: `src/projections/distributions/sampled.py`
- Modify: `src/projections/distributions/__init__.py`
- Create: `tests/test_distributions/test_sampled.py`

- [ ] **Step 1: Write the failing test for `FrozenSampledDistribution.sample` semantics**

Create `tests/test_distributions/test_sampled.py`:

```python
"""Tests for src/projections/distributions/sampled.py.

Spec: docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md §3.2.
"""
from __future__ import annotations

import numpy as np
import pytest

from projections.distributions.sampled import FrozenSampledDistribution


def test_sample_returns_underlying_array_when_n_equals_len() -> None:
    """The n == len branch is the load-bearing property for cross-stat coherence.

    score_distribution calls .sample(n_samples=10_000) on each per-stat distribution;
    when n matches len(samples), FrozenSampledDistribution returns the underlying
    array directly, preserving any cross-stat correlation baked into the array.
    """
    rng = np.random.default_rng(seed=42)
    samples = rng.normal(loc=10.0, scale=2.0, size=100)
    dist = FrozenSampledDistribution(samples=samples)
    out = dist.sample(n=100)
    assert out is samples or np.shares_memory(out, samples)
    assert np.array_equal(out, samples)


def test_sample_resamples_when_n_differs_from_len() -> None:
    """Fallback to rng.choice when n != len(samples)."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dist = FrozenSampledDistribution(samples=samples)
    rng = np.random.default_rng(seed=0)
    out = dist.sample(n=20, rng=rng)
    assert out.shape == (20,)
    assert set(out.tolist()).issubset(set(samples.tolist()))


def test_sample_resamples_use_provided_rng_for_determinism() -> None:
    samples = np.arange(100, dtype=np.float64)
    dist = FrozenSampledDistribution(samples=samples)
    out1 = dist.sample(n=50, rng=np.random.default_rng(seed=7))
    out2 = dist.sample(n=50, rng=np.random.default_rng(seed=7))
    assert np.array_equal(out1, out2)


def test_mean_std_quantile_cdf_match_numpy_reference() -> None:
    rng = np.random.default_rng(seed=1)
    samples = rng.normal(loc=5.0, scale=1.5, size=10_000)
    dist = FrozenSampledDistribution(samples=samples)
    assert dist.mean() == pytest.approx(float(samples.mean()))
    assert dist.std() == pytest.approx(float(samples.std()))
    assert dist.quantile(0.5) == pytest.approx(float(np.quantile(samples, 0.5)))
    assert dist.cdf(5.0) == pytest.approx(float((samples <= 5.0).mean()))


def test_quantile_rejects_endpoints() -> None:
    dist = FrozenSampledDistribution(samples=np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="must be in"):
        dist.quantile(0.0)
    with pytest.raises(ValueError, match="must be in"):
        dist.quantile(1.0)


def test_satisfies_distribution_protocol() -> None:
    """Runtime structural isinstance check against the Distribution Protocol.

    Distribution is @runtime_checkable; isinstance checks attribute presence.
    """
    from projections.distributions.base import Distribution

    dist = FrozenSampledDistribution(samples=np.array([1.0, 2.0, 3.0]))
    assert isinstance(dist, Distribution)
```

- [ ] **Step 2: Run the tests; expect ModuleNotFoundError**

Run:
```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_distributions/test_sampled.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'projections.distributions.sampled'`.

- [ ] **Step 3: Create `src/projections/distributions/sampled.py`**

```python
"""Sampled-distribution implementations.

Hosts:
- ``FrozenSampledDistribution`` — empirical distribution whose ``.sample(n)``
  returns the underlying ``samples`` array verbatim when ``n == len(samples)``,
  preserving any external row-aligned correlation structure. Used by
  ``DecomposedBaselineModel`` to keep within-row cross-stat correlation
  intact when ``score_distribution`` consumes multiple decomposed-stat
  distributions that share an underlying volume draw.

Cf. ``projections.scoring.score_distribution.SampledDistribution``, which
always re-samples via ``rng.choice``. The two classes are deliberately
separate: ``SampledDistribution`` is the *output* of scoring (a points
distribution that callers re-sample from), ``FrozenSampledDistribution``
is an *input* to scoring (a per-stat distribution whose internal sample
ordering carries information that must not be shuffled).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class FrozenSampledDistribution:
    """Distribution-Protocol-conforming dataclass backed by a frozen samples array.

    ``sample(n)`` returns ``self.samples`` directly when ``n == len(self.samples)``;
    otherwise falls back to ``rng.choice`` for bootstrap-style re-sampling.

    The ``n == len`` branch is the architectural guarantee that lets
    ``DecomposedBaselineModel`` plumb cross-stat correlation through
    ``score_distribution`` without any modifications to the scoring path.
    """

    samples: NDArray[np.float64]

    def mean(self) -> float:
        return float(self.samples.mean())

    def std(self) -> float:
        return float(self.samples.std())

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        return float(np.quantile(self.samples, q))

    def cdf(self, x: float) -> float:
        return float((self.samples <= x).mean())

    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        if n == len(self.samples):
            return self.samples
        rng = rng if rng is not None else np.random.default_rng()
        return rng.choice(self.samples, size=n, replace=True)
```

- [ ] **Step 4: Add to the package `__init__`**

Edit `src/projections/distributions/__init__.py`. Find:

```python
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

Replace with:

```python
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
from projections.distributions.sampled import FrozenSampledDistribution

__all__ = [
    "Distribution",
    "FrozenSampledDistribution",
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

- [ ] **Step 5: Run the tests; all should pass**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_distributions/test_sampled.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Verification gates**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m mypy src tests
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff check src tests
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff format --check src tests
```

Expected: zero violations / no drift across all three commands.

- [ ] **Step 7: Commit**

```
git add src/projections/distributions/sampled.py \
        src/projections/distributions/__init__.py \
        tests/test_distributions/test_sampled.py
git commit -m "feat(distributions): add FrozenSampledDistribution for cross-stat coherent sampling"
```

---

## Phase 2 — `DecompositionSpec` + `DecomposedBaselineModel` skeleton + `fit`

Goal: ship the new model class with the train-side decomposition recipe wired. Tests verify fit populates the volume + efficiency sub-models correctly. No predict_distribution yet.

### Task 2: `DecompositionSpec`, class scaffold, `fit` extension, fit tests

**Files:**
- Create: `src/projections/models/decomposed_baseline.py`
- Create: `tests/test_models/test_decomposed_baseline.py`

- [ ] **Step 1: Write the failing test for `DecompositionSpec` semantics**

Create `tests/test_models/test_decomposed_baseline.py`:

```python
"""Tests for src/projections/models/decomposed_baseline.py.

Spec: docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md.
"""
from __future__ import annotations

import dataclasses

import pytest

from projections.models.decomposed_baseline import DecompositionSpec
from projections.schemas import Stat


def test_decomposition_spec_is_frozen() -> None:
    spec = DecompositionSpec(
        volume_stat=Stat.TARGETS,
        efficiency_label="catch_rate",
        efficiency_clip_hi=1.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.efficiency_clip_hi = 2.0  # type: ignore[misc]


def test_decomposition_spec_equality_by_value() -> None:
    a = DecompositionSpec(Stat.TARGETS, "catch_rate", 1.0)
    b = DecompositionSpec(Stat.TARGETS, "catch_rate", 1.0)
    c = DecompositionSpec(Stat.TARGETS, "yards_per_target", float("inf"))
    assert a == b
    assert a != c
```

- [ ] **Step 2: Run; expect ModuleNotFoundError**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/test_decomposed_baseline.py::test_decomposition_spec_is_frozen -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'projections.models.decomposed_baseline'`.

- [ ] **Step 3: Create the module skeleton with `DecompositionSpec` only**

```python
# src/projections/models/decomposed_baseline.py
"""DecomposedBaselineModel — per-stat volume × efficiency decomposition.

Subclass of ``BaselineModel`` with a constructor argument
``decomposed_stats: Mapping[Stat, DecompositionSpec]`` mapping composite stats
to their (volume, efficiency) decomposition recipe. Stats absent from
``decomposed_stats`` fall through to the inherited direct-RidgeCV path.

Per-row prediction for decomposed stats samples volume × efficiency factors
with within-row coherent sampling (a single shared volume draw flows into
every decomposed stat with the same ``volume_stat``). Persistence uses
``QuantileDistribution`` summaries via the existing codec branch — no codec
edits required.

Spec: docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from projections.schemas import Stat


@dataclass(frozen=True, slots=True)
class DecompositionSpec:
    """Per-stat decomposition recipe.

    Attributes:
        volume_stat: the volume axis (e.g., ``Stat.TARGETS``). Multiple
            decomposed stats sharing the same ``volume_stat`` get a shared
            per-row volume draw at predict time.
        efficiency_label: human-readable label of the efficiency factor
            (e.g., ``"catch_rate"``, ``"yards_per_target"``). Used in logs
            and diagnostics.
        efficiency_clip_hi: upper bound for sample-time efficiency clipping.
            ``1.0`` for ratio efficiency factors (catch_rate, td_rate);
            ``float("inf")`` for unbounded efficiency factors (yards_per_target).
    """

    volume_stat: Stat
    efficiency_label: str
    efficiency_clip_hi: float
```

- [ ] **Step 4: Run; tests pass**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/test_decomposed_baseline.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Write failing tests for `DecomposedBaselineModel` construction + fit**

Append to `tests/test_models/test_decomposed_baseline.py`:

```python
import numpy as np
import pandas as pd

from projections.models.decomposed_baseline import DecomposedBaselineModel


def _synthetic_wr_fit_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic WR features + weekly_stats spanning 2018-2020 for fit testing.

    Designed to be cheap and small: 3 seasons × 3 weeks × 4 players = 36 rows.
    Feature values are plausible but synthetic. Volume (targets) varies enough
    to give the volume ridge a signal; catch_rate varies independently.
    """
    rows: list[dict[str, object]] = []
    weekly_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed=0xD3C0)
    for season in (2018, 2019, 2020):
        for week in (1, 2, 3):
            for pid in range(4):
                gsis = f"00-{season}-{pid:02d}"
                team = ("KC", "BUF", "DAL", "PHI")[pid]
                opp = ("LV", "NE", "NYG", "WAS")[pid]
                targets_per_game_l4 = float(rng.uniform(3.0, 12.0))
                features_row = {
                    "gsis_id": gsis,
                    "season": season,
                    "week": week,
                    "team": team,
                    "opponent": opp,
                    "depth_rank": float(rng.integers(1, 4)),
                    "targets_per_game_l4": targets_per_game_l4,
                    "targets_per_game_std": float(rng.uniform(0.5, 3.0)),
                }
                rows.append(features_row)
                # Truth: targets ~ Poisson(targets_per_game_l4); receptions ~
                # Binomial(targets, 0.65); receiving_yards ~ targets * Normal(11, 3);
                # receiving_tds ~ Bernoulli(targets * 0.05).
                tgt = int(rng.poisson(targets_per_game_l4))
                rec = int(rng.binomial(max(tgt, 0), 0.65))
                yds = float(max(rec, 0) * rng.normal(11.0, 3.0))
                tds = int(rng.random() < min(tgt * 0.05, 1.0))
                weekly_rows.append({
                    "gsis_id": gsis,
                    "season": season,
                    "week": week,
                    "team": team,
                    "opponent": opp,
                    "position": "WR",
                    "targets": tgt,
                    "receptions": rec,
                    "receiving_yards": yds,
                    "receiving_tds": tds,
                    "rushing_yards": 0.0,
                    "rushing_tds": 0,
                    "fumbles_lost": 0,
                })
    features = pd.DataFrame(rows)
    weekly_stats = pd.DataFrame(weekly_rows)
    return features, weekly_stats


def test_decomposed_baseline_constructs_with_empty_decomposed_stats() -> None:
    """An empty ``decomposed_stats`` mapping should make the model behaviorally
    indistinguishable from BaselineModel on the train side (only direct ridges
    fit). Verifies the opt-in arch's empty-config case.
    """
    from projections.models.baseline import (
        _WR_DIST_FAMILIES,
        _WR_FEATURE_COLUMNS,
        _WR_TARGET_STATS,
        _default_code_hash_files,
    )
    from projections.schemas import Position, WrFeaturesSchema

    model = DecomposedBaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
        feature_schema=WrFeaturesSchema,
        code_hash_files=_default_code_hash_files("wr.py"),
        decomposed_stats={},
    )
    assert model.decomposed_stats == {}
    assert model.volume_ridges == {}
    assert model.efficiency_ridges == {}
```

This first test uses the inherited `BaselineModel` config knobs (target_stats, feature_columns, etc.) — they are imported privately. **Note: the test imports from `projections.models.baseline` private names — this is intentional and load-bearing for synthetic-fixture tests; do not refactor the imports.** The test currently fails because `DecomposedBaselineModel` is not defined.

- [ ] **Step 6: Run; expect ImportError**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/test_decomposed_baseline.py::test_decomposed_baseline_constructs_with_empty_decomposed_stats -v
```

Expected: FAIL — `ImportError: cannot import name 'DecomposedBaselineModel'`.

- [ ] **Step 7: Add `DecomposedBaselineModel` class scaffold (no fit logic yet)**

Append to `src/projections/models/decomposed_baseline.py`:

```python
from collections.abc import Mapping
from dataclasses import field

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from projections.models.baseline import BaselineModel
from projections.schemas import WeeklyStatsSchema


@dataclass
class DecomposedBaselineModel(BaselineModel):
    """BaselineModel + per-stat decomposition.

    See module docstring for the architectural overview.

    Per-stat decomposition opt-in via ``decomposed_stats``. Stats not in the
    mapping fall through to the inherited direct-RidgeCV path. Stats in the
    mapping are predicted as ``mu_volume * mu_efficiency`` at the mean level
    and as ``volume_samples * efficiency_samples`` at the distribution level
    (with the volume samples shared across all decomposed stats with the same
    ``volume_stat``, baking within-row cross-stat correlation into the per-row
    sample arrays).
    """

    decomposed_stats: Mapping[Stat, DecompositionSpec] = field(default_factory=dict)
    volume_ridges: dict[Stat, RidgeCV] = field(default_factory=dict)
    efficiency_ridges: dict[Stat, RidgeCV] = field(default_factory=dict)
    volume_variance: dict[Stat, float] = field(default_factory=dict)
    efficiency_variance: dict[Stat, float] = field(default_factory=dict)
```

- [ ] **Step 8: Run; the empty-decomposed_stats construction test passes**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/test_decomposed_baseline.py -v
```

Expected: 3 passed.

- [ ] **Step 9: Write the failing test for `fit` populating decomposition sub-models**

Append to `tests/test_models/test_decomposed_baseline.py`:

```python
def _wr_decomp_model_receptions_only() -> DecomposedBaselineModel:
    from projections.models.baseline import (
        _WR_DIST_FAMILIES,
        _WR_FEATURE_COLUMNS,
        _WR_TARGET_STATS,
        _default_code_hash_files,
    )
    from projections.schemas import Position, WrFeaturesSchema

    return DecomposedBaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
        feature_schema=WrFeaturesSchema,
        code_hash_files=_default_code_hash_files("wr.py"),
        decomposed_stats={
            Stat.RECEPTIONS: DecompositionSpec(
                volume_stat=Stat.TARGETS,
                efficiency_label="catch_rate",
                efficiency_clip_hi=1.0,
            ),
        },
    )


def test_fit_populates_direct_ridges_for_all_target_stats() -> None:
    """All target_stats (including decomposed ones) get a direct-comparator
    ridge. Decomposed stats get those plus volume + efficiency sub-models.
    See spec §3.1.3 'fit both arms' rationale.
    """
    model = _wr_decomp_model_receptions_only()
    features, weekly_stats = _synthetic_wr_fit_inputs()
    # Synthetic features need all the WR feature columns. Backfill the rest
    # with finite values so the schema's no-op fill matches BaselineModel.fit.
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = 0.0
    # Validate against the schema before passing — matches BaselineModel.fit
    # boundary.
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)
    assert set(model.ridges) == set(model.target_stats)


def test_fit_populates_decomposition_sub_models_for_receptions() -> None:
    """Decomposed stats get a shared volume RidgeCV and a per-stat efficiency
    RidgeCV; both are stored in ``volume_ridges`` / ``efficiency_ridges`` keyed
    by ``volume_stat`` and the composite stat respectively. Residual stds are
    persisted in ``volume_variance`` / ``efficiency_variance``.
    """
    model = _wr_decomp_model_receptions_only()
    features, weekly_stats = _synthetic_wr_fit_inputs()
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = 0.0
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)

    assert Stat.TARGETS in model.volume_ridges
    assert isinstance(model.volume_ridges[Stat.TARGETS], RidgeCV)
    assert Stat.RECEPTIONS in model.efficiency_ridges
    assert isinstance(model.efficiency_ridges[Stat.RECEPTIONS], RidgeCV)
    assert model.volume_variance[Stat.TARGETS] > 0.0
    assert model.efficiency_variance[Stat.RECEPTIONS] > 0.0


def test_fit_model_id_uses_decomposed_baseline_prefix() -> None:
    model = _wr_decomp_model_receptions_only()
    features, weekly_stats = _synthetic_wr_fit_inputs()
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = 0.0
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)
    assert model.model_id.startswith("decomposed-baseline:wr:")
```

- [ ] **Step 10: Run; expect failures**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/test_decomposed_baseline.py -v
```

Expected: 3 failures — `fit` is not yet overridden, and `model_id` falls back to `BaselineModel.model_id`'s `"baseline:"` prefix.

- [ ] **Step 11: Implement `fit` override + `model_id` property**

Append to `src/projections/models/decomposed_baseline.py`:

```python
    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """Train direct ridges (parent) + decomposition sub-models.

        Parent ``BaselineModel.fit`` handles schema validation, the
        ``(gsis_id, season, week)`` inner-join, NaN-drop, median-imputation, and
        per-stat direct ridges. We extend it with the volume + efficiency
        sub-models for each entry in ``decomposed_stats``.
        """
        # Run parent fit first — populates self.ridges, self.feature_means,
        # self.variance_params, self.train_seasons, self.code_hash.
        super().fit(features, weekly_stats)

        if not self.decomposed_stats:
            return

        # Re-build the same train frame parent used. We need the joined truth
        # rows (post-NaN-drop) to fit volume + efficiency ridges against.
        features_validated = self.feature_schema.validate(features)
        weekly_validated = WeeklyStatsSchema.validate(weekly_stats)
        ws = weekly_validated[weekly_validated["position"] == self.position.value].copy()
        truth_cols = ["gsis_id", "season", "week"] + [
            s.value for s in self.target_stats
        ] + [
            spec.volume_stat.value
            for spec in self.decomposed_stats.values()
        ]
        # Dedupe in case a volume_stat is already in target_stats (it is for
        # WR receptions/yards/tds — volume_stat is TARGETS which is NOT in
        # _WR_TARGET_STATS, but defensive dedupe keeps the schema-select fast).
        truth_cols = list(dict.fromkeys(truth_cols))
        joined = features_validated.merge(
            ws[truth_cols],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        feature_frame = self._x_frame_with_bool_coercion(joined)
        # Apply the same NaN-drop the parent applied. ``self.feature_means``
        # is populated by parent fit on the pre-dropna frame, so post-dropna
        # alignment is implicit.
        keep_mask = feature_frame.notna().all(axis=1)
        feature_frame = feature_frame.loc[keep_mask]
        joined_kept = joined.loc[feature_frame.index]
        x_train = feature_frame.to_numpy(dtype=np.float64)

        alphas = np.logspace(-3, 3, 13)

        # Fit shared volume sub-models (one per unique volume_stat).
        unique_volume_stats = {spec.volume_stat for spec in self.decomposed_stats.values()}
        for volume_stat in unique_volume_stats:
            y_vol = joined_kept[volume_stat.value].to_numpy(dtype=np.float64)
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(x_train, y_vol)
            self.volume_ridges[volume_stat] = ridge
            mu_vol: np.ndarray = ridge.predict(x_train).astype(np.float64)
            residuals_vol = y_vol - mu_vol
            self.volume_variance[volume_stat] = max(float(residuals_vol.std()), 1e-6)

        # Fit per-stat efficiency sub-models on rows with volume_stat > 0.
        for composite_stat, spec in self.decomposed_stats.items():
            vol_col = joined_kept[spec.volume_stat.value].to_numpy(dtype=np.float64)
            mask = vol_col > 0
            if not mask.any():
                raise ValueError(
                    f"Cannot fit efficiency factor for {composite_stat.value}: "
                    f"no training rows with {spec.volume_stat.value} > 0."
                )
            x_pos = x_train[mask]
            vol_pos = vol_col[mask]
            num_pos = joined_kept.loc[mask, composite_stat.value].to_numpy(dtype=np.float64)
            ratio = num_pos / vol_pos
            ridge_eff = RidgeCV(alphas=alphas)
            ridge_eff.fit(x_pos, ratio)
            self.efficiency_ridges[composite_stat] = ridge_eff
            mu_eff: np.ndarray = ridge_eff.predict(x_pos).astype(np.float64)
            residuals_eff = ratio - mu_eff
            self.efficiency_variance[composite_stat] = max(float(residuals_eff.std()), 1e-6)

    @property
    def model_id(self) -> str:
        """Stable identifier of the form
        ``"decomposed-baseline:<position>:<8-char-code-hash>:<train-start>-<train-end>"``.

        Mirrors ``BaselineModel.model_id`` except for the class-name prefix.
        """
        if self.code_hash is None or self.train_seasons is None:
            raise RuntimeError("model_id is undefined for unfitted models")
        return (
            f"decomposed-baseline:{self.position.value.lower()}:{self.code_hash}"
            f":{self.train_seasons[0]}-{self.train_seasons[1]}"
        )
```

- [ ] **Step 12: Run; all 6 tests should pass**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/test_decomposed_baseline.py -v
```

Expected: 6 passed.

- [ ] **Step 13: Verification gates**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m mypy src tests
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff check src tests
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 14: Commit**

```
git add src/projections/models/decomposed_baseline.py \
        tests/test_models/test_decomposed_baseline.py
git commit -m "feat(models): DecomposedBaselineModel scaffold + fit — direct + decomposed sub-models"
```

---

## Phase 3 — `build_stat_distributions` + `predict_distribution` override

Goal: emit `FrozenSampledDistribution` for decomposed stats with within-row coherent volume sampling; override `predict_distribution`'s persistence step to convert to `QuantileDistribution` for the params blob.

### Task 3a: Extract `_persistable_dists_for_packing` hook on `BaselineModel`

**Files:**
- Modify: `src/projections/models/baseline.py` (one new method + one-line call site change in `predict_distribution`)
- Modify: `tests/test_models/test_baseline.py` if it has a `predict_distribution` test; otherwise no test change for this micro-refactor.

- [ ] **Step 1: Re-read the relevant lines of `predict_distribution`**

```
sed -n '676,735p' src/projections/models/baseline.py
```

Expected: shows the per-row loop that calls `pack_per_stat_params(stat_dists)` at line ~709.

- [ ] **Step 2: Extract the hook method**

Edit `src/projections/models/baseline.py`. In the `BaselineModel` class, find the per-row loop in `predict_distribution`:

```python
            seed = derive_row_seed(
                gsis_id=str(feat_row["gsis_id"]),
                season=int(feat_row["season"]),
                week=int(feat_row["week"]),
                ruleset_name=ruleset.name,
            )
            points = score_distribution(stat_dists, ruleset, n_samples=10_000, seed=seed)
            family_blob = pack_per_stat_params(stat_dists)
```

Replace with:

```python
            seed = derive_row_seed(
                gsis_id=str(feat_row["gsis_id"]),
                season=int(feat_row["season"]),
                week=int(feat_row["week"]),
                ruleset_name=ruleset.name,
            )
            points = score_distribution(stat_dists, ruleset, n_samples=10_000, seed=seed)
            family_blob = pack_per_stat_params(self._persistable_dists_for_packing(stat_dists))
```

Then add the hook method to the `BaselineModel` class (above `predict_distribution`, after `model_id`):

```python
    def _persistable_dists_for_packing(
        self, stat_dists: Mapping[Stat, Distribution]
    ) -> Mapping[Stat, Distribution]:
        """Hook for subclasses to convert non-codec-supported Distribution types
        (e.g., FrozenSampledDistribution) into supported ones (QuantileDistribution)
        before persistence. Default returns ``stat_dists`` unchanged.

        BaselineModel emits only parametric distributions and QuantileDistribution
        in stat_dists, all of which are directly codec-supported, so no conversion
        is needed at this level.
        """
        return stat_dists
```

This refactor preserves BaselineModel's behavior exactly — the default override is a no-op pass-through.

- [ ] **Step 3: Run full test suite to verify no regression**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/ -v
```

Expected: same pass count as before the change (zero regressions).

- [ ] **Step 4: Verification gates**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m mypy src tests
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff check src tests
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 5: Commit**

```
git add src/projections/models/baseline.py
git commit -m "refactor(baseline): extract _persistable_dists_for_packing hook for subclass override"
```

### Task 3b: `build_stat_distributions` override + cross-stat coherence test

**Files:**
- Modify: `src/projections/models/decomposed_baseline.py`
- Modify: `tests/test_models/test_decomposed_baseline.py`

The override emits `FrozenSampledDistribution` for decomposed stats (per-row 10,000-element sample arrays with within-row volume sharing) and inherits the parametric path for non-decomposed stats.

- [ ] **Step 1: Write the failing test for `build_stat_distributions` type signature**

Append to `tests/test_models/test_decomposed_baseline.py`:

```python
from projections.distributions import FrozenSampledDistribution, QuantileDistribution
from projections.distributions.parametric import (
    ParametricGamma,
    ParametricNegativeBinomial,
    ParametricNormal,
)


def test_build_stat_distributions_emits_frozen_sampled_for_decomposed_stat() -> None:
    """build_stat_distributions emits FrozenSampledDistribution for the
    decomposed stat (carrying the per-row composed sample array) and
    parametric distributions for non-decomposed stats (unchanged path).
    """
    model = _wr_decomp_model_receptions_only()
    features, weekly_stats = _synthetic_wr_fit_inputs()
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = 0.0
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)

    # Predict on the same frame (a stand-in for an eval-row frame).
    per_row = model.build_stat_distributions(features)
    assert len(per_row) == len(features)

    row0 = per_row[0]
    # Receptions is decomposed → FrozenSampledDistribution carrying 10_000 samples.
    assert isinstance(row0[Stat.RECEPTIONS], FrozenSampledDistribution)
    assert len(row0[Stat.RECEPTIONS].samples) == 10_000
    # Non-decomposed stats keep parametric forms per _WR_DIST_FAMILIES.
    assert isinstance(row0[Stat.RECEIVING_YARDS], ParametricNormal)
    assert isinstance(row0[Stat.RECEIVING_TDS], ParametricNegativeBinomial)
    assert isinstance(row0[Stat.RUSHING_YARDS], ParametricNormal)
    assert isinstance(row0[Stat.RUSHING_TDS], ParametricNegativeBinomial)
    assert isinstance(row0[Stat.FUMBLES_LOST], ParametricNegativeBinomial)


def test_within_row_cross_stat_coherence_two_stat_synthetic_config() -> None:
    """Architectural guarantee: two decomposed stats sharing the same
    volume_stat produce FrozenSampledDistribution instances with strongly
    correlated samples within a row (Pearson rho > 0.5).

    v1 production decomposes only receptions, but this test exercises the
    coherence code path that v2 will exercise in production. If this test
    fails, the cross-stat coherence guarantee is broken.
    """
    from projections.models.baseline import (
        _WR_DIST_FAMILIES,
        _WR_FEATURE_COLUMNS,
        _WR_TARGET_STATS,
        _default_code_hash_files,
    )
    from projections.schemas import Position, WrFeaturesSchema

    model = DecomposedBaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
        feature_schema=WrFeaturesSchema,
        code_hash_files=_default_code_hash_files("wr.py"),
        decomposed_stats={
            Stat.RECEPTIONS: DecompositionSpec(
                volume_stat=Stat.TARGETS,
                efficiency_label="catch_rate",
                efficiency_clip_hi=1.0,
            ),
            Stat.RECEIVING_YARDS: DecompositionSpec(
                volume_stat=Stat.TARGETS,
                efficiency_label="yards_per_target",
                efficiency_clip_hi=float("inf"),
            ),
        },
    )
    features, weekly_stats = _synthetic_wr_fit_inputs()
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = 0.0
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)

    per_row = model.build_stat_distributions(features)
    row0 = per_row[0]
    rec_dist = row0[Stat.RECEPTIONS]
    yds_dist = row0[Stat.RECEIVING_YARDS]
    assert isinstance(rec_dist, FrozenSampledDistribution)
    assert isinstance(yds_dist, FrozenSampledDistribution)
    # Both compose against the same per-row volume draw → strong positive
    # element-wise correlation.
    rho = float(np.corrcoef(rec_dist.samples, yds_dist.samples)[0, 1])
    assert rho > 0.5, f"expected within-row Pearson rho > 0.5, got {rho:.3f}"


def test_build_stat_distributions_is_deterministic_per_row() -> None:
    """Same model state + same features → same sample arrays. Per-row seed
    derivation via derive_row_seed makes this byte-stable.
    """
    model = _wr_decomp_model_receptions_only()
    features, weekly_stats = _synthetic_wr_fit_inputs()
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = 0.0
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)
    per_row_a = model.build_stat_distributions(features)
    per_row_b = model.build_stat_distributions(features)
    for i in range(len(features)):
        rec_a = per_row_a[i][Stat.RECEPTIONS]
        rec_b = per_row_b[i][Stat.RECEPTIONS]
        assert isinstance(rec_a, FrozenSampledDistribution)
        assert isinstance(rec_b, FrozenSampledDistribution)
        assert np.array_equal(rec_a.samples, rec_b.samples)
```

- [ ] **Step 2: Run; expect failures (method not overridden)**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/test_decomposed_baseline.py -v
```

Expected: the new tests fail because `build_stat_distributions` is inherited (returns parametric for all stats, not `FrozenSampledDistribution` for decomposed stats).

- [ ] **Step 3: Implement `build_stat_distributions` override**

Append to `src/projections/models/decomposed_baseline.py`:

```python
from typing import Final

from projections.distributions import Distribution, FrozenSampledDistribution
from projections.scoring.score_distribution import derive_row_seed


_N_SAMPLES: Final[int] = 10_000


# Append the method to the DecomposedBaselineModel class.
    def build_stat_distributions(
        self, features: pd.DataFrame
    ) -> list[dict[Stat, Distribution]]:
        """Per-row dicts of {Stat -> Distribution}.

        Non-decomposed stats: inherited parametric path (Normal/Gamma/NB per
        ``dist_families``).
        Decomposed stats: per-row FrozenSampledDistribution with 10_000 samples.

        Within-row cross-stat coherence: all decomposed stats sharing a
        ``volume_stat`` reuse the same per-row volume draw, baking element-wise
        correlation into their composed sample arrays. ``score_distribution``
        consumes the FrozenSampledDistributions via ``.sample(n=10_000)``;
        the ``n == len`` branch returns the underlying arrays verbatim,
        preserving the correlation through scoring.
        """
        # Parent path emits parametric distributions for all target_stats —
        # including the decomposed ones, which we will overwrite below.
        per_row_parametric = super().build_stat_distributions(features)

        if not self.decomposed_stats:
            return per_row_parametric

        # Feature matrix (impute with train medians; bool → int8).
        x_frame = self._x_frame_with_bool_coercion(features)
        if self.feature_means is None:
            raise RuntimeError(
                "feature_means is None; model is not fitted. Call fit() before predict."
            )
        x_frame = x_frame.fillna(self.feature_means)
        x = x_frame.to_numpy(dtype=np.float64)

        # Vectorized volume + efficiency predictions.
        unique_volume_stats = {spec.volume_stat for spec in self.decomposed_stats.values()}
        per_volume_mu: dict[Stat, np.ndarray] = {}
        for vs in unique_volume_stats:
            mu_v: np.ndarray = self.volume_ridges[vs].predict(x).astype(np.float64)
            per_volume_mu[vs] = mu_v

        per_decomposed_mu_eff: dict[Stat, np.ndarray] = {}
        for cs in self.decomposed_stats:
            mu_e: np.ndarray = self.efficiency_ridges[cs].predict(x).astype(np.float64)
            per_decomposed_mu_eff[cs] = mu_e

        # Per-row coherent sampling.
        features_iter = features.reset_index(drop=True)
        for i in range(len(features_iter)):
            feat_row = features_iter.iloc[i]
            row_seed = derive_row_seed(
                gsis_id=str(feat_row["gsis_id"]),
                season=int(feat_row["season"]),
                week=int(feat_row["week"]),
                ruleset_name="__decomp_build__",
            )
            # Per-row volume draws (one per volume_stat). Shared across all
            # decomposed stats whose spec.volume_stat is this volume_stat.
            vol_samples: dict[Stat, np.ndarray] = {}
            for vs, mu_arr in per_volume_mu.items():
                sigma_v = self.volume_variance[vs]
                rng_v = np.random.default_rng(row_seed)
                vs_raw = rng_v.normal(loc=float(mu_arr[i]), scale=sigma_v, size=_N_SAMPLES)
                vol_samples[vs] = np.maximum(vs_raw, 0.0)

            # Per-stat efficiency draws + composition.
            for j, (composite_stat, spec) in enumerate(self.decomposed_stats.items(), start=1):
                sigma_e = self.efficiency_variance[composite_stat]
                rng_e = np.random.default_rng(row_seed + j)
                eff_raw = rng_e.normal(
                    loc=float(per_decomposed_mu_eff[composite_stat][i]),
                    scale=sigma_e,
                    size=_N_SAMPLES,
                )
                eff_samples = np.clip(eff_raw, 0.0, spec.efficiency_clip_hi)
                composed = vol_samples[spec.volume_stat] * eff_samples
                # Replace the parametric entry with the live FrozenSampled.
                per_row_parametric[i][composite_stat] = FrozenSampledDistribution(
                    samples=composed
                )

        return per_row_parametric
```

The per-row seed strategy: `row_seed + 0` for the volume RNG, `row_seed + j` (j = 1..K) for each of K efficiency RNGs. This keeps factor draws independent within a row while preserving determinism across runs. The `ruleset_name` argument to `derive_row_seed` is set to `"__decomp_build__"` (a constant) so the build-time per-row seed is independent of the runtime ruleset — `predict_distribution`'s call to `derive_row_seed(ruleset_name=ruleset.name)` for the *scoring* seed remains independent of the build seed.

- [ ] **Step 4: Run the three new tests**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/test_decomposed_baseline.py -v
```

Expected: 9 passed (6 prior + 3 new).

- [ ] **Step 5: Verification gates**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m mypy src tests
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff check src tests
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff format --check src tests
```

Expected: zero violations.

- [ ] **Step 6: Commit**

```
git add src/projections/models/decomposed_baseline.py \
        tests/test_models/test_decomposed_baseline.py
git commit -m "feat(models): DecomposedBaselineModel.build_stat_distributions — within-row coherent sampling"
```

### Task 3c: Override `_persistable_dists_for_packing` + predict_distribution round-trip test

**Files:**
- Modify: `src/projections/models/decomposed_baseline.py`
- Modify: `tests/test_models/test_decomposed_baseline.py`

- [ ] **Step 1: Write the failing test for `predict_distribution` round-trip**

Append to `tests/test_models/test_decomposed_baseline.py`:

```python
from projections.distributions import unpack_per_stat_params
from projections.schemas import DistributionFamily, ProjectionWeeklySchema, Ruleset


def test_predict_distribution_round_trip_through_quantile_codec() -> None:
    """predict_distribution validates against ProjectionWeeklySchema; the
    params blob decodes back with the decomposed stat as a QuantileDistribution
    (the persisted form), non-decomposed stats as their parametric form.
    """
    model = _wr_decomp_model_receptions_only()
    features, weekly_stats = _synthetic_wr_fit_inputs()
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = 0.0
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)

    ruleset = Ruleset.espn_ppr()
    pred = model.predict_distribution(features, ruleset=ruleset)
    # Schema validation already happens inside predict_distribution; if we get
    # here without raising, the frame is schema-conformant.
    assert len(pred) == len(features)
    assert (pred["family"] == DistributionFamily.SAMPLED_SUMMARY.value).all()

    # Decode the first row's params blob.
    blob = pred.iloc[0]["params"]
    decoded = unpack_per_stat_params(blob)
    assert isinstance(decoded[Stat.RECEPTIONS], QuantileDistribution)
    assert isinstance(decoded[Stat.RECEIVING_YARDS], ParametricNormal)
    assert isinstance(decoded[Stat.RECEIVING_TDS], ParametricNegativeBinomial)


def test_predict_distribution_mean_p10_p90_finite() -> None:
    """The summarized per-row mean/p10/p50/p90 should be finite and ordered
    p10 < p50 < p90 for nontrivial predictions."""
    model = _wr_decomp_model_receptions_only()
    features, weekly_stats = _synthetic_wr_fit_inputs()
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = 0.0
    features = model.feature_schema.validate(features)
    weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
    model.fit(features, weekly_stats)
    pred = model.predict_distribution(features, ruleset=Ruleset.espn_ppr())

    for col in ("mean", "p10", "p50", "p90"):
        assert pred[col].notna().all()
        assert np.isfinite(pred[col]).all()
    assert (pred["p10"] <= pred["p50"]).all()
    assert (pred["p50"] <= pred["p90"]).all()
```

- [ ] **Step 2: Run; expect failure (no codec branch for FrozenSampledDistribution)**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/test_decomposed_baseline.py::test_predict_distribution_round_trip_through_quantile_codec -v
```

Expected: FAIL — `pack_per_stat_params` raises `ValueError: No codec entry for Distribution type FrozenSampledDistribution`.

- [ ] **Step 3: Override `_persistable_dists_for_packing`**

Append to `src/projections/models/decomposed_baseline.py`:

```python
from projections.distributions import QuantileDistribution

# Persisted quantile grid: 19 knots at q ∈ {0.05, 0.10, …, 0.95}. Per-row
# persistence cost is 19 floats per decomposed stat — small relative to the
# existing per-stat parametric encoding. QuantileDistribution recomposes via
# linear interpolation between knots.
_PERSISTED_QUANTILES: Final[np.ndarray] = np.arange(0.05, 0.96, 0.05)


# Append the method to the DecomposedBaselineModel class.
    def _persistable_dists_for_packing(
        self, stat_dists: Mapping[Stat, Distribution]
    ) -> Mapping[Stat, Distribution]:
        """Convert FrozenSampledDistribution entries (decomposed stats) into
        QuantileDistribution summaries for persistence via the existing
        QUANTILE codec branch.

        The cross-stat correlation baked into the FrozenSampledDistribution's
        sample array is lost at the persistence boundary — see spec §3.1.5 +
        §5 risk #4. This is acceptable for v1 (no post-hoc re-scoring from
        persisted params blobs). The scoring step has already consumed the
        live in-memory FrozenSampledDistributions via score_distribution
        upstream; this conversion is downstream of that.
        """
        out: dict[Stat, Distribution] = {}
        for stat, dist in stat_dists.items():
            if isinstance(dist, FrozenSampledDistribution):
                values = np.quantile(dist.samples, _PERSISTED_QUANTILES).astype(np.float64)
                # QuantileDistribution requires values to be non-decreasing
                # (validated in its __init__); np.quantile already produces
                # sorted output for ascending quantiles.
                out[stat] = QuantileDistribution(
                    quantiles=_PERSISTED_QUANTILES.copy(),
                    values=values,
                )
            else:
                out[stat] = dist
        return out
```

- [ ] **Step 4: Run the two new tests**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/test_decomposed_baseline.py -v
```

Expected: 11 passed (9 prior + 2 new).

- [ ] **Step 5: Edge case — `QuantileDistribution` rejects equal-valued knots when all samples are identical**

QuantileDistribution's `__init__` requires `np.all(np.diff(v) >= 0)` (non-decreasing values). All-zero samples produce all-zero values, which satisfies the non-strict ≥ check. Equal values across knots are fine. **However**, QuantileDistribution also requires `np.all(np.diff(q) > 0)` (strictly ascending quantiles), which `_PERSISTED_QUANTILES = np.arange(0.05, 0.96, 0.05)` satisfies.

Add a defensive test:

```python
def test_persistable_dists_handles_all_zero_samples_gracefully() -> None:
    """If a per-row composed sample array is all zero (e.g., zero predicted
    volume on a deep WR4), the quantile values are all zero. QuantileDistribution
    accepts non-decreasing values (equal is allowed); this test guards against
    a regression where _PERSISTED_QUANTILES is changed to require strictly
    increasing values.
    """
    from projections.models.decomposed_baseline import (
        _PERSISTED_QUANTILES,
        DecomposedBaselineModel,
    )

    zero_samples = np.zeros(10_000, dtype=np.float64)
    frozen = FrozenSampledDistribution(samples=zero_samples)

    # Minimal model just to expose _persistable_dists_for_packing as bound.
    from projections.models.baseline import (
        _WR_DIST_FAMILIES,
        _WR_FEATURE_COLUMNS,
        _WR_TARGET_STATS,
        _default_code_hash_files,
    )
    from projections.schemas import Position, WrFeaturesSchema

    model = DecomposedBaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
        feature_schema=WrFeaturesSchema,
        code_hash_files=_default_code_hash_files("wr.py"),
        decomposed_stats={
            Stat.RECEPTIONS: DecompositionSpec(
                volume_stat=Stat.TARGETS,
                efficiency_label="catch_rate",
                efficiency_clip_hi=1.0,
            ),
        },
    )
    out = model._persistable_dists_for_packing({Stat.RECEPTIONS: frozen})
    quant = out[Stat.RECEPTIONS]
    assert isinstance(quant, QuantileDistribution)
    assert np.array_equal(quant.values_, np.zeros_like(_PERSISTED_QUANTILES))
```

Run:

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/test_decomposed_baseline.py::test_persistable_dists_handles_all_zero_samples_gracefully -v
```

Expected: PASS.

- [ ] **Step 6: Full Phase 3 verification**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/ -v -k "decomposed or sampled or test_baseline"
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m mypy src tests
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff check src tests
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff format --check src tests
```

Expected: all green.

- [ ] **Step 7: Commit**

```
git add src/projections/models/decomposed_baseline.py \
        tests/test_models/test_decomposed_baseline.py
git commit -m "feat(models): DecomposedBaselineModel.predict_distribution — QuantileDistribution persistence"
```

---

## Phase 4 — Factory + dispatch + Model-protocol conformance

Goal: register `wr_decomposed_baseline` in `_WR_FACTORIES["decomposed-baseline"]` so `run_backtest(model_classes=("decomposed-baseline",), positions=(Position.WR,))` dispatches correctly. Add `"decomposed-baseline"` to `scripts/backtest.py`'s argparse choices so the CLI works.

### Task 4: `wr_decomposed_baseline` factory + dispatch registration

**Files:**
- Modify: `src/projections/models/decomposed_baseline.py` (factory + code_hash_files helper)
- Modify: `src/projections/models/__init__.py` (register + export)
- Modify: `scripts/backtest.py` (argparse choices)
- Modify: `tests/test_models/test_decomposed_baseline.py` (factory + Model-protocol conformance tests)

- [ ] **Step 1: Write failing tests for factory + registration**

Append to `tests/test_models/test_decomposed_baseline.py`:

```python
def test_wr_decomposed_baseline_factory_returns_unfitted_model() -> None:
    from projections.models.decomposed_baseline import wr_decomposed_baseline

    model = wr_decomposed_baseline()
    assert isinstance(model, DecomposedBaselineModel)
    assert model.position.value == "WR"
    # v1 config: receptions decomposed, others fall through.
    assert set(model.decomposed_stats.keys()) == {Stat.RECEPTIONS}
    rec_spec = model.decomposed_stats[Stat.RECEPTIONS]
    assert rec_spec.volume_stat is Stat.TARGETS
    assert rec_spec.efficiency_label == "catch_rate"
    assert rec_spec.efficiency_clip_hi == 1.0
    # Unfitted state.
    assert model.code_hash is None
    assert model.train_seasons is None


def test_dispatch_registers_decomposed_baseline_for_wr() -> None:
    """POSITION_DISPATCH[Position.WR].factories['decomposed-baseline'] is
    callable and returns the same shape the factory returns directly.
    """
    from projections.models import POSITION_DISPATCH
    from projections.schemas import Position

    factory = POSITION_DISPATCH[Position.WR].factories["decomposed-baseline"]
    model = factory()
    assert isinstance(model, DecomposedBaselineModel)


def test_dispatch_default_model_class_for_wr_is_unchanged() -> None:
    """Pre-gate state: WR routes to ensemble. The flip (if ADOPT) is
    explicitly a Phase 6 action, not landed in this PR's initial commits.
    """
    from projections.models import POSITION_DISPATCH
    from projections.schemas import Position

    assert POSITION_DISPATCH[Position.WR].default_model_class == "ensemble"


def test_decomposed_baseline_satisfies_model_protocol() -> None:
    """DecomposedBaselineModel must implement Model: position, model_id, fit,
    predict_distribution, save, load. fit + predict_distribution end-to-end
    exercise on a synthetic frame is the strongest structural assertion.
    """
    from projections.models.base import Model

    model = _wr_decomp_model_receptions_only()
    # Structural conformance: getattr each Protocol member without error.
    for attr in ("position", "model_id", "fit", "predict_distribution", "save", "load"):
        assert hasattr(model, attr), f"missing Model attr {attr!r}"
    # Mypy enforces signatures; this is a smoke check.
    _: Model = model  # type-check assignment
```

- [ ] **Step 2: Run; expect factory ImportError**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/test_decomposed_baseline.py::test_wr_decomposed_baseline_factory_returns_unfitted_model -v
```

Expected: FAIL — `ImportError: cannot import name 'wr_decomposed_baseline'`.

- [ ] **Step 3: Add the factory + code_hash helper to `decomposed_baseline.py`**

Append to `src/projections/models/decomposed_baseline.py`:

```python
from pathlib import Path

from projections.models.baseline import (
    _WR_DIST_FAMILIES,
    _WR_FEATURE_COLUMNS,
    _WR_TARGET_STATS,
    _default_code_hash_files,
)
from projections.schemas import Position, WrFeaturesSchema


_WR_RECEIVING_DECOMPOSITION: Final[Mapping[Stat, DecompositionSpec]] = {
    Stat.RECEPTIONS: DecompositionSpec(
        volume_stat=Stat.TARGETS,
        efficiency_label="catch_rate",
        efficiency_clip_hi=1.0,
    ),
}


def _decomposed_baseline_code_hash_files(position_module: str) -> tuple[Path, ...]:
    """Extend BaselineModel's code-hash file tuple with decomposed_baseline.py.

    Any edit to this module must invalidate fitted artifacts' model_id, so it
    must be in the code-hash file tuple. Builds on _default_code_hash_files
    rather than duplicating its 8-file list.
    """
    project_root = Path(__file__).resolve().parents[3]
    return _default_code_hash_files(position_module) + (
        project_root / "src" / "projections" / "models" / "decomposed_baseline.py",
    )


def wr_decomposed_baseline() -> DecomposedBaselineModel:
    """Construct an unfitted WR decomposed-baseline model.

    v1 config: receptions decomposed via TARGETS × catch_rate; all other WR
    target stats fall through to direct RidgeCV (identical to BaselineModel).
    """
    return DecomposedBaselineModel(
        position=Position.WR,
        target_stats=_WR_TARGET_STATS,
        feature_columns=_WR_FEATURE_COLUMNS,
        dist_families=_WR_DIST_FAMILIES,
        feature_schema=WrFeaturesSchema,
        code_hash_files=_decomposed_baseline_code_hash_files("wr.py"),
        decomposed_stats=_WR_RECEIVING_DECOMPOSITION,
    )
```

- [ ] **Step 4: Run the factory test; expect PASS**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/test_decomposed_baseline.py::test_wr_decomposed_baseline_factory_returns_unfitted_model -v
```

Expected: PASS.

- [ ] **Step 5: Register in `_WR_FACTORIES`**

Edit `src/projections/models/__init__.py`. Find:

```python
from projections.models.baseline import (
    BaselineModel,
    qb_baseline,
    rb_baseline,
    te_baseline,
    wr_baseline,
)
```

Replace with:

```python
from projections.models.baseline import (
    BaselineModel,
    qb_baseline,
    rb_baseline,
    te_baseline,
    wr_baseline,
)
from projections.models.decomposed_baseline import (
    DecomposedBaselineModel,
    wr_decomposed_baseline,
)
```

Find:

```python
_WR_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": wr_baseline,
    "lightgbm": wr_lightgbm,
    "lightgbm-tuned": wr_lightgbm_tuned,
    "lightgbm-nb": wr_lightgbm_nb,
    "ensemble": wr_ensemble,
}
```

Replace with:

```python
_WR_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": wr_baseline,
    "decomposed-baseline": wr_decomposed_baseline,
    "lightgbm": wr_lightgbm,
    "lightgbm-tuned": wr_lightgbm_tuned,
    "lightgbm-nb": wr_lightgbm_nb,
    "ensemble": wr_ensemble,
}
```

Find:

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
    "production_model_for",
    "qb_baseline",
    "qb_ensemble",
    "qb_lightgbm",
    "qb_lightgbm_nb",
    "qb_lightgbm_tuned",
    "rb_baseline",
    "rb_ensemble",
    "rb_lightgbm",
    "rb_lightgbm_nb",
    "rb_lightgbm_tuned",
    "te_baseline",
    "te_ensemble",
    "te_lightgbm",
    "te_lightgbm_nb",
    "te_lightgbm_tuned",
    "wr_baseline",
    "wr_ensemble",
    "wr_lightgbm",
    "wr_lightgbm_nb",
    "wr_lightgbm_tuned",
]
```

Replace with:

```python
__all__ = [
    "POSITION_DISPATCH",
    "BaselineModel",
    "DecomposedBaselineModel",
    "EnsembleModel",
    "LightGBMModel",
    "LightGBMNbModel",
    "LightGBMTunedModel",
    "Model",
    "compute_code_hash",
    "production_model_for",
    "qb_baseline",
    "qb_ensemble",
    "qb_lightgbm",
    "qb_lightgbm_nb",
    "qb_lightgbm_tuned",
    "rb_baseline",
    "rb_ensemble",
    "rb_lightgbm",
    "rb_lightgbm_nb",
    "rb_lightgbm_tuned",
    "te_baseline",
    "te_ensemble",
    "te_lightgbm",
    "te_lightgbm_nb",
    "te_lightgbm_tuned",
    "wr_baseline",
    "wr_decomposed_baseline",
    "wr_ensemble",
    "wr_lightgbm",
    "wr_lightgbm_nb",
    "wr_lightgbm_tuned",
]
```

- [ ] **Step 6: Add `decomposed-baseline` to `scripts/backtest.py` argparse choices**

Edit `scripts/backtest.py`. Find:

```python
    parser.add_argument(
        "--model",
        choices=[
            "baseline",
            "lightgbm",
            "lightgbm-tuned",
            "lightgbm-nb",
            "ensemble",
            "both",
            "all",
        ],
        default="both",
```

Replace with:

```python
    parser.add_argument(
        "--model",
        choices=[
            "baseline",
            "decomposed-baseline",
            "lightgbm",
            "lightgbm-tuned",
            "lightgbm-nb",
            "ensemble",
            "both",
            "all",
        ],
        default="both",
```

(The `"all"` branch is left unchanged — `"all"` does not include `decomposed-baseline` because the integration's binding gate runs WR-only and won't benefit from the other 4 positions × decomposed-baseline cells. Phase 5 invokes `decomposed-baseline` via the Python API, not via `--model all`.)

- [ ] **Step 7: Run all Phase-4 tests**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest tests/test_models/test_decomposed_baseline.py -v
```

Expected: all tests pass (now 15+ depending on count from Phase 3).

- [ ] **Step 8: Smoke test — `production_model_for(WR)` is still ensemble**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -c "
from projections.models import production_model_for
from projections.schemas import Position
m = production_model_for(Position.WR)
print(type(m).__name__)
"
```

Expected: `EnsembleModel` — confirms the pre-gate state. The flip (if any) is a Phase 6 action.

- [ ] **Step 9: Full verification gates**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest -v
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m mypy src tests scripts
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff check src tests scripts
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff format --check src tests scripts
```

Expected: full suite passes; zero mypy / ruff violations.

- [ ] **Step 10: Commit**

```
git add src/projections/models/decomposed_baseline.py \
        src/projections/models/__init__.py \
        scripts/backtest.py \
        tests/test_models/test_decomposed_baseline.py
git commit -m "feat(models): register wr_decomposed_baseline factory in _WR_FACTORIES + backtest CLI choice"
```

---

## Phase 5 — Real-data backtest + adoption gate

Goal: run a single 3-class WR-only backtest producing per-(model_class) per-row predictions; run `scripts/adoption_gate.py --run` mode twice to produce the binding cell (vs ensemble) and informational cell (vs baseline) verdicts.

### Task 5: Real-data dual-cell adoption gate

**Files:**
- Modify (run artifacts, not committed wholesale): `data/backtest/run_wr_decomp_<ts>/results.parquet`, `metrics.parquet`
- Create (committed): `reports/adoption_gate_wr_decomposed_baseline_vs_ensemble.md`, `reports/adoption_gate_wr_decomposed_baseline_vs_ensemble.csv`, `reports/adoption_gate_wr_decomposed_baseline_vs_baseline.md`, `reports/adoption_gate_wr_decomposed_baseline_vs_baseline.csv`

- [ ] **Step 1: Confirm the WR feature cache is current**

```
ls data/features/wr/season=2024/week=01/
```

Expected: `part.parquet` present. If absent, run `refresh_features.py` first:

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe scripts/refresh_features.py wr --seasons 2018-2024
```

(Expected to be already-current per the post-PR-35 nflreadpy state on main; only re-run if the partition is missing.)

- [ ] **Step 2: Run the WR-only 3-class backtest via the Python API**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -c "
from datetime import datetime, UTC
from pathlib import Path
from projections.backtest.harness import run_backtest
from projections.schemas import Position

ts = datetime.now(UTC).strftime('%Y%m%dT%H%M%S')
out = Path(f'data/backtest/run_wr_decomp_{ts}')
out.mkdir(parents=True, exist_ok=True)
run = run_backtest(
    model_classes=('baseline', 'ensemble', 'decomposed-baseline'),
    positions=(Position.WR,),
)
run.per_row_results.to_parquet(out / 'results.parquet')
run.metrics.to_parquet(out / 'metrics.parquet')
print(f'wrote {out}/results.parquet ({len(run.per_row_results)} rows, '
      f'model_classes={sorted(run.per_row_results[\"model_class\"].unique())})')
"
```

Expected output: a directory under `data/backtest/run_wr_decomp_<ts>/` containing `results.parquet` (~24K rows: ~8K per class × 3 classes) and `metrics.parquet`. **Capture `<ts>` for the next step.**

Wall-clock: ~5-15 minutes (3 model classes × 4 walk-forward folds × ensemble's 4-stage fit). Significantly longer than a baseline-only run because ensemble fit is the bottleneck.

- [ ] **Step 3: Run binding cell adoption gate (decomposed-baseline vs ensemble)**

```
RUN_DIR=data/backtest/run_wr_decomp_<ts>  # substitute actual timestamp
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m scripts.adoption_gate \
  --run "$RUN_DIR" \
  --candidate decomposed-baseline \
  --incumbent ensemble \
  --position WR \
  --csv-out reports/adoption_gate_wr_decomposed_baseline_vs_ensemble.csv
```

Expected: prints a per-position table (just WR) with `rmse_delta` (point + 95% CI) and `spearman_delta` (point + 95% CI). Verdict label printed: `ADOPT` / `MARGINAL` / `DO_NOT_ADOPT`. CSV written to the reports/ path. **Capture the printed verdict + CIs.**

- [ ] **Step 4: Run informational cell adoption gate (decomposed-baseline vs baseline)**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m scripts.adoption_gate \
  --run "$RUN_DIR" \
  --candidate decomposed-baseline \
  --incumbent baseline \
  --position WR \
  --csv-out reports/adoption_gate_wr_decomposed_baseline_vs_baseline.csv
```

Expected: same shape as Step 3 but with baseline as incumbent. **Capture the printed verdict + CIs.**

- [ ] **Step 5: Hand-write `reports/adoption_gate_wr_decomposed_baseline_vs_ensemble.md`**

The CSV from Step 3 is machine-readable; the markdown is the human-readable companion. Follow the format used in `reports/adoption_gate_weather_features_rb_wr.md` (PR #29) — a header block + the position table + a one-paragraph verdict narrative. Template:

```markdown
# Adoption Gate — `decomposed-baseline` vs `ensemble` (WR)

**Spec:** `docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md`
**Plan:** `docs/superpowers/plans/2026-05-13-wr-target-decomposition-integration.md`
**Run:** `data/backtest/run_wr_decomp_<ts>/`
**Date:** YYYY-MM-DD
**Branch:** `feat/wr-target-decomposition`

## Verdict

**`<VERDICT>`** — (one sentence reading off the CSV).

## Per-position metrics

| Position | n_paired | RMSE delta | RMSE 95% CI | Spearman delta | Spearman 95% CI | Verdict |
|---|---:|---:|---|---:|---|:---:|
| WR | <fill> | <fill> | <fill> | <fill> | <fill> | **<fill>** |

(Single-position gate; binding cell per spec §1.3.5 contingency matrix.)
```

Fill in values from the CSV.

- [ ] **Step 6: Hand-write `reports/adoption_gate_wr_decomposed_baseline_vs_baseline.md`**

Identical shape; informational cell.

- [ ] **Step 7: Commit reports**

```
git add reports/adoption_gate_wr_decomposed_baseline_vs_ensemble.md \
        reports/adoption_gate_wr_decomposed_baseline_vs_ensemble.csv \
        reports/adoption_gate_wr_decomposed_baseline_vs_baseline.md \
        reports/adoption_gate_wr_decomposed_baseline_vs_baseline.csv
git commit -m "data(adoption-gate): WR decomposed-baseline vs ensemble + vs baseline"
```

---

## Phase 6 — §1.3.5 outcome execution + writeup + PM/TODO update

Goal: execute the per-verdict branch from spec §1.3.5 + write the consolidated summary report + update the project decision log.

### Task 6: Per-verdict action + summary + PM/TODO

**Files:**
- (Conditional) Modify: `src/projections/models/__init__.py` (default_model_class flip per §1.3.5 ADOPT branch; full revert per REGRESSION; no change for MARGINAL/DO_NOT_ADOPT)
- Create: `reports/wr_target_decomposition_summary.md`
- Modify: `project_management.md` (prepend decision-log entry)
- Modify: `TODO.md` (update under #23 — target decomposition direction)

- [ ] **Step 1: Read both adoption-gate CSVs to determine the §1.3.5 branch**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -c "
import pandas as pd
b = pd.read_csv('reports/adoption_gate_wr_decomposed_baseline_vs_ensemble.csv')
i = pd.read_csv('reports/adoption_gate_wr_decomposed_baseline_vs_baseline.csv')
print('binding cell (vs ensemble):')
print(b.to_string(index=False))
print()
print('informational cell (vs baseline):')
print(i.to_string(index=False))
"
```

Read the printed verdicts. Branch in spec §1.3.5:
- `ADOPT` (binding) → Step 2a
- `MARGINAL` (binding RMSE PASS, Spearman fail) → Step 2b
- `DO_NOT_ADOPT` (binding) + check informational → Step 2c
- `REGRESSION` (binding CI strictly > 0) → Step 2d

- [ ] **Step 2a: ADOPT branch — flip `default_model_class` to `decomposed-baseline`**

Edit `src/projections/models/__init__.py`. Find:

```python
    Position.WR: _PositionDispatch(
        factories=_WR_FACTORIES,
        feature_builder=build_wr_features,
        feature_schema=WrFeaturesSchema,
        ngs_stat_type="receiving",
        default_model_class="ensemble",
    ),
```

Replace with:

```python
    Position.WR: _PositionDispatch(
        factories=_WR_FACTORIES,
        feature_builder=build_wr_features,
        feature_schema=WrFeaturesSchema,
        ngs_stat_type="receiving",
        default_model_class="decomposed-baseline",
    ),
```

Confirm the existing test `test_dispatch_default_model_class_for_wr_is_unchanged` now fails (expected — flip the assertion or split into "pre-flip" / "post-flip" testing). For this PR, **delete the pre-flip test** since the dispatch state is changing in this commit:

Edit `tests/test_models/test_decomposed_baseline.py`. Remove:

```python
def test_dispatch_default_model_class_for_wr_is_unchanged() -> None:
    """Pre-gate state: WR routes to ensemble. The flip (if ADOPT) is
    explicitly a Phase 6 action, not landed in this PR's initial commits.
    """
    from projections.models import POSITION_DISPATCH
    from projections.schemas import Position

    assert POSITION_DISPATCH[Position.WR].default_model_class == "ensemble"
```

Replace with:

```python
def test_dispatch_default_model_class_for_wr_after_adopt_flip() -> None:
    """Post-§1.3.5 ADOPT outcome: WR routes to decomposed-baseline. Pre-flip
    assertion is in this PR's git history (Phase 4 commit).
    """
    from projections.models import POSITION_DISPATCH
    from projections.schemas import Position

    assert POSITION_DISPATCH[Position.WR].default_model_class == "decomposed-baseline"
```

Skip to Step 3 (writeup).

- [ ] **Step 2b: MARGINAL branch — keep ensemble, ship infra-only**

No source edits. The factory + class are already registered (Phase 4) and available for any future plan that wants to invoke them. The summary report documents the verdict + Spearman-floor-violation magnitude. Skip to Step 3.

- [ ] **Step 2c: DO_NOT_ADOPT branch — keep ensemble, evaluate informational cell**

No `default_model_class` change. The informational cell decides the recommended follow-up phrasing in the summary report (Step 3):
- **Informational ADOPT** → recommend "ensemble-child swap" next plan.
- **Informational MARGINAL / DO_NOT_ADOPT** → close target-decomposition WR-receiving-2-factor direction.
- **Informational REGRESSION** → full revert (jump to Step 2d).

Skip to Step 3 unless informational REGRESSION fires.

- [ ] **Step 2d: REGRESSION branch — full revert of production-routed code**

If either the binding cell is REGRESSION OR (DO_NOT_ADOPT binding + REGRESSION informational), full-revert per PR #31 precedent:

Edit `src/projections/models/__init__.py`. Remove the `decomposed_baseline` import and the `_WR_FACTORIES["decomposed-baseline"]` entry and the `DecomposedBaselineModel` / `wr_decomposed_baseline` `__all__` entries. Edit `scripts/backtest.py` to remove `"decomposed-baseline"` from `--model` choices.

Keep `src/projections/models/decomposed_baseline.py`, `src/projections/distributions/sampled.py`, `tests/test_models/test_decomposed_baseline.py`, `tests/test_distributions/test_sampled.py`, and `src/projections/distributions/__init__.py`'s `FrozenSampledDistribution` export — these are draft infrastructure, useful for any future revisit. **The PR's `git diff main` should show zero net change to `src/projections/models/__init__.py` and `scripts/backtest.py`** after this commit.

Edit the `tests/test_models/test_decomposed_baseline.py::test_dispatch_registers_decomposed_baseline_for_wr` test to skip with `pytest.skip("Reverted per §1.3.5 REGRESSION outcome — see reports/wr_target_decomposition_summary.md")` so the in-tree module isn't reachable through dispatch.

Skip to Step 3 (writeup).

- [ ] **Step 3: Write `reports/wr_target_decomposition_summary.md`**

Synthesize across the two adoption-gate CSVs + the PR #32 probe predictions + the §1.3.5 outcome. Template (fill bracketed values from the CSVs):

```markdown
# WR Target Decomposition Integration — verdict `<VERDICT>` (YYYY-MM-DD, on branch `feat/wr-target-decomposition`)

**Spec:** `docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md`
**Plan:** `docs/superpowers/plans/2026-05-13-wr-target-decomposition-integration.md`
**Builds on:** PR #32 (WR target decomposition probe; SIGNAL on receptions cell, NULL on receiving_yards / _tds).

**Status.** Production integration of `DecomposedBaselineModel` (peer to `BaselineModel`) with per-stat decomposition opt-in. v1 ships WR receptions-only decomposition (volume = `targets`, efficiency = `catch_rate`). `FrozenSampledDistribution` carries within-row coherent sampling through `score_distribution`; persistence uses `QuantileDistribution` summaries via the existing codec branch (no codec edits). Binding gate `(DecomposedBaselineModel, WR)` vs `(EnsembleModel, WR)` returned **`<VERDICT>`**; informational gate `(DecomposedBaselineModel, WR)` vs `(BaselineModel, WR)` returned **`<VERDICT>`**.

## Per-cell verdicts

| Cell | Incumbent | Candidate | n_paired | RMSE delta | RMSE 95% CI | Spearman delta | Spearman 95% CI | Verdict |
|---|---|---|---:|---:|---|---:|---|:---:|
| **Binding** (gates routing) | ensemble | decomposed-baseline | <n> | <pt> | <CI> | <pt> | <CI> | **<v>** |
| Informational (probe-equivalent) | baseline | decomposed-baseline | <n> | <pt> | <CI> | <pt> | <CI> | <v> |

## Probe-vs-gate calibration

The probe (PR #32) predicted Δ-RMSE on receptions only: `-0.0042 fpts` × ESPN PPR reception coefficient (`1.0 fpt/rec`) = `-0.0042 fpts` composite-fpts contribution. The gate's binding-cell composite-fpts magnitude was `<pt> fpts` (CI `<CI>`).

- Probe-vs-gate sign agreement: `<yes / flipped>`.
- Probe-vs-gate magnitude shift: `<delta> fpts` (positive = gate measured a smaller candidate advantage than probe; negative = larger).
- Marginal-zone flag (per PR #31 retrospective rule): binding-cell composite-fpts magnitude `<is / is not>` under `0.005 fpts`. (Coverage was strictly above 0.95 across all eval years, so the rule does not strictly fire — magnitude alone is the flag.)

## Coverage

`targets > 0` rate on WR rows per eval year:

| Year | Eval n | Eval coverage | Train n | Train coverage |
|---:|---:|---:|---:|---:|
| 2021 | <n> | <rate> | <n> | <rate> |
| 2022 | <n> | <rate> | <n> | <rate> |
| 2023 | <n> | <rate> | <n> | <rate> |
| 2024 | <n> | <rate> | <n> | <rate> |

Threshold `0.95` met with margin on every cell. No `--coverage-threshold` relaxation invoked.

## §1.3.5 outcome

`<one of:>`

- **(ADOPT)** `_PositionDispatch[Position.WR].default_model_class` flipped from `"ensemble"` to `"decomposed-baseline"`. Ensemble factory remains available in `_WR_FACTORIES`. Existing `data/ensemble_weights/ensemble_wr_*.json` artifacts retained in-tree.
- **(MARGINAL)** Production routing unchanged (WR stays on `ensemble`). `DecomposedBaselineModel` + `wr_decomposed_baseline` factory shipped as available infrastructure. Spearman-floor magnitude `<value>` documented.
- **(DO_NOT_ADOPT, informational ADOPT)** Production routing unchanged. Recommended next plan: swap `BaselineModel → DecomposedBaselineModel` inside `EnsembleModel`'s child A factory and re-fit ensemble weights; new dual-run gate on `(EnsembleModel-with-decomposed-baseline, WR)` vs current `(EnsembleModel, WR)` measures the ensemble-internal contribution of decomposition.
- **(DO_NOT_ADOPT, informational MARGINAL/DO_NOT_ADOPT)** Production routing unchanged. Closing target-decomposition at the WR receiving 2-factor unit. Refined-unit candidates (3-factor decomposition; factor-appropriate sub-model classes per PR #32 spec §7.4) remain open under TODO #23 but require independent mechanism evidence before re-probing per PR #31's retrospective rule.
- **(DO_NOT_ADOPT, informational REGRESSION OR binding REGRESSION)** Full revert of production-routed code. `decomposed_baseline.py` + `sampled.py` + factory + tests retained in-tree as historical record. PM logs the WR decomposition direction closed; future revisits require either factor-appropriate sub-models or a refined volume axis.

## Mechanism interpretation

`<2-3 sentences interpreting the verdicts.>`

## Deferred follow-ups

1. **Factor-appropriate sub-model classes** (logistic for catch_rate, log-link Gamma for yards_per_target, Poisson for targets) per PR #32 probe spec §7.4. Gated on this integration's verdict per §1.3.5.
2. **Decomposition for receiving_yards / receiving_tds** (NULL in probe). Conditional on factor-appropriate sub-models closing those NULL probes first.
3. **Other positions** — RB / QB / TE decomposition, each its own probe + integration cycle.

## Plan-vs-execution deviations (if any)

`<list any minor deviations caught during execution — see PR #29 / PR #32 precedent.>`
```

- [ ] **Step 4: Prepend the PM decision-log entry**

Edit `project_management.md`. Insert after the leading `---` separator (becoming the new top entry):

```markdown
## WR Target Decomposition Integration — verdict `<VERDICT>` (YYYY-MM-DD, on branch `feat/wr-target-decomposition`)

**Status:** Production integration of `DecomposedBaselineModel` (peer to `BaselineModel`) with per-stat decomposition opt-in. v1 ships WR receptions-only decomposition (volume `targets` × efficiency `catch_rate` with sample-time clip `[0, 1]`). New `FrozenSampledDistribution` carries within-row coherent sampling through `score_distribution`; persistence uses `QuantileDistribution` summaries via the existing codec branch (no codec edits). Spec at `docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md`.

**Per-cell verdicts:**

| Cell | Incumbent | Candidate | n_paired | RMSE Δ (fpts) | RMSE 95% CI | Spearman Δ | Verdict |
|---|---|---|---:|---:|---|---:|:---:|
| **Binding** (gates routing) | ensemble | decomposed-baseline | `<n>` | `<pt>` | `<CI>` | `<pt>` | **`<v>`** |
| Informational (probe-equivalent) | baseline | decomposed-baseline | `<n>` | `<pt>` | `<CI>` | `<pt>` | `<v>` |

**§1.3.5 outcome:** `<one-line summary; flip / no-flip / revert + the production-routing state after this PR>`.

**Probe-vs-gate calibration:** probe predicted `-0.0042 fpts` on receptions (PR #32); gate measured `<pt>` on the binding cell + `<pt>` on the informational cell. `<sign-agreement / divergence narrative>`. Magnitude `<is / is not>` in the PR #31 marginal-zone (`< 0.005 fpts`).

**Recommended next direction.** `<per §1.3.5 branch — ensemble-child swap; factor-appropriate sub-models; close direction; full revert>`.

**Reports:** `reports/wr_target_decomposition_summary.md`, `reports/adoption_gate_wr_decomposed_baseline_vs_ensemble.{md,csv}`, `reports/adoption_gate_wr_decomposed_baseline_vs_baseline.{md,csv}`.

---
```

- [ ] **Step 5: Update `TODO.md` under #23**

Edit `TODO.md`. Find the `### 23.` section (target decomposition; the probe's PM entry references this). Append an Update entry under that section:

```markdown
**Update YYYY-MM-DD (WR target decomposition integration, branch `feat/wr-target-decomposition`):** Production integration shipped per `docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md`. Binding gate `(DecomposedBaselineModel, WR)` vs `(EnsembleModel, WR)` verdict: **`<v>`** (composite RMSE Δ `<pt>` fpts, CI `<CI>`). Informational gate `(DecomposedBaselineModel, WR)` vs `(BaselineModel, WR)` verdict: `<v>` (`<pt>` fpts, CI `<CI>`). `<§1.3.5 outcome one-liner>`. `<recommended-next-direction one-liner per §1.3.5 branch>`. See `reports/wr_target_decomposition_summary.md`.
```

If the TODO file does not yet have a `### 23.` section (e.g., it was assigned in the original TODO list but never realized as a heading), add it under "Open" with a brief context paragraph and then the Update entry. Confirm by grepping first:

```
grep -n "target decomposition\|### 23\|TODO #23" TODO.md
```

If the grep returns nothing, the target-decomposition direction was tracked at the probe-spec level only and didn't get a TODO heading — in that case add an "Open" subsection `### 23. Target decomposition follow-up` with a one-paragraph context summary + the Update entry.

- [ ] **Step 6: Full verification gates**

```
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest -v
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m mypy src tests scripts
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff check src tests scripts
PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff format --check src tests scripts
```

Expected: full suite passes; zero violations. **If the ADOPT branch fired and the `default_model_class` flipped to `decomposed-baseline`, the existing backtest snapshot at `tests/backtest/model_metrics.json` may be stale** — re-run `python scripts/backtest.py --update-snapshot` to regenerate. This step is mechanical and is the same shape PR #35 / PR #29 used.

- [ ] **Step 7: Commit Phase-6 changes**

```
git add reports/wr_target_decomposition_summary.md \
        project_management.md \
        TODO.md \
        src/projections/models/__init__.py \
        tests/test_models/test_decomposed_baseline.py
# Plus, conditionally:
#   tests/backtest/model_metrics.json   (only if --update-snapshot was run on ADOPT)
git commit -m "data(target-decomposition): §1.3.5 outcome <VERDICT> — <one-line summary>"
```

- [ ] **Step 8: Push the branch + open the PR**

```
git push -u origin feat/wr-target-decomposition
gh pr create --title "feat: WR target decomposition integration (DecomposedBaselineModel, <VERDICT>)" \
  --body "$(cat <<'EOF'
## Summary
- New `DecomposedBaselineModel` peer to `BaselineModel` with per-stat decomposition opt-in.
- `wr_decomposed_baseline` factory configured for receptions-only decomposition.
- `FrozenSampledDistribution` carries within-row coherent sampling through `score_distribution`.
- Adoption-gate verdicts: binding `<v>`, informational `<v>`. See `reports/wr_target_decomposition_summary.md`.

## §1.3.5 outcome
<one-line: flip / no-flip / revert>

## Verification
- `pytest -v`: all pass
- `mypy src tests scripts`: 0 violations
- `ruff check / format --check src tests scripts`: 0 violations

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Notes for the executor

1. **Worktree venv routing.** Every Python invocation prefixes `PYTHONPATH=src` to route the worktree's `src/projections/` ahead of the main repo's editable-installed copy. The main repo's `.venv` is shared (located at `../../../.venv/` from the worktree). If you forget the prefix on a test invocation, you'll silently test the *main repo's* source instead of the worktree's. The verification-gate steps fail loudly if the routing is broken, but per-step tests will silently mislead.

2. **Pre-commit hooks.** Per CLAUDE.md, pre-commit hooks may use system Python; the workaround per PR #29 is `PATH="/.venv/Scripts:$PATH" git commit ...` or equivalent. If hooks fail with pydantic v1/v2 import errors, that's the pre-existing system-Python conflict — apply the PATH workaround, not `--no-verify`.

3. **Phase 5 wall-clock.** The 3-class backtest is slow because ensemble fit runs the 4-stage flow. Expect ~5-15 minutes; do not interrupt. The verification gate steps in earlier phases are fast (~30s each).

4. **Phase 6 branching.** Steps 2a–2d are mutually exclusive based on the gate verdicts. The executor reads the CSVs in Step 1 and follows exactly one branch. Do not commit the §1.3.5 source edit until the verdict is confirmed.

5. **Snapshot update on ADOPT.** If the routing flip happens, run `python scripts/backtest.py --update-snapshot` to regenerate `tests/backtest/model_metrics.json`. The metric rows for `decomposed-baseline` will replace nothing pre-existing (it's a new class); the rows for `ensemble` may shift slightly due to ordering. Commit the snapshot diff in the same Phase-6 commit.

6. **No new ingest, no schema changes, no caller-script changes.** The integration is entirely model-layer + distribution-layer. The feature cache, weekly_stats, schedules, draft_picks partitions are all consumed as-is.

7. **CLAUDE.md verification checklist** before final commit:
   - `PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest -v` — all pass
   - `PYTHONPATH=src ../../../.venv/Scripts/python.exe -m mypy src tests scripts` — 0 violations
   - `PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff check src tests scripts` — 0 violations
   - `PYTHONPATH=src ../../../.venv/Scripts/python.exe -m ruff format --check src tests scripts` — no drift
   - `PYTHONPATH=src ../../../.venv/Scripts/python.exe -m pytest -v -k "ingest or store or schemas"` — clean (per CLAUDE.md "Forced verification" rule, run even though this PR doesn't touch ingest/store/schemas — the integration test of those seams catches dtype regressions)

Paste the output (or summary) of each into the PR body per CLAUDE.md "Forced verification — END-OF-EFFORT CHECKLIST."
