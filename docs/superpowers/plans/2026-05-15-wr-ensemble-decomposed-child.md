# WR Ensemble — Decomposed-Baseline Child A Swap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `wr_ensemble_decomposed` factory that wires `EnsembleModel`'s child A to `wr_decomposed_baseline` (instead of `wr_baseline`), run a dual-cell adoption gate, and execute the §1.3.5 outcome.

**Architecture:** Single-factory addition in `ensemble.py` + one-line registration in `_WR_FACTORIES`. `EnsembleModel` itself is child-agnostic — no changes to fit/predict/codec/mixture machinery (unless the mixture-of-`QuantileDistribution`-with-`NegativeBinomial` tail-quantile tests in Task 2 surface a numerical bug in `_bracket_for_components`, in which case the fix is scoped there). Persistence keys separate via distinct `model_id` code-hashes; mixture-of-`QuantileDistribution`-with-`NegativeBinomial` already round-trips through `codec._pack_single`/`_unpack_single`'s `MIXTURE` branch.

**Tech Stack:** Python, pandas, numpy, sklearn `RidgeCV`, LightGBM, pandera schemas, msgpack codec, pytest, mypy strict, ruff.

**Spec:** `docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md`

---

## File Map

**Create:**
- `tests/test_models/test_ensemble_decomposed.py` — factory wiring + fit/predict + code_hash divergence + mixture-tail sanity tests
- `reports/adoption_gate_wr_ensemble_decomposed_vs_ensemble.{md,csv}` — binding cell (Task 4 output)
- `reports/adoption_gate_wr_ensemble_decomposed_vs_decomposed_baseline.{md,csv}` — informational cell (Task 4 output)
- `reports/wr_ensemble_decomposed_summary.md` — Task 5 writeup

**Modify:**
- `src/projections/models/ensemble.py` — add `wr_ensemble_decomposed` factory (append after `wr_ensemble`)
- `src/projections/models/__init__.py` — import + `__all__` + `_WR_FACTORIES["ensemble-decomposed"]` entry; conditionally flip `default_model_class` in Task 5 on ADOPT
- `scripts/backtest.py` — `--model` choice + WR-only positions restriction
- `tests/test_scripts/test_backtest_cli.py` — add mirror test for `ensemble-decomposed`
- `tests/test_models/test_position_dispatch.py` — update `expected` dict on §1.3.5 ADOPT (Task 5 only)
- `tests/backtest/model_metrics.json` — snapshot update on §1.3.5 ADOPT (Task 5 only)
- `project_management.md`, `TODO.md` — Task 5 decision-log entries

**Conditionally modify (only if Task 2's mixture-tail tests fail):**
- `src/projections/distributions/mixture.py` — fix `_bracket_for_components` for `QuantileDistribution` components

---

## Task 1: Factory + registration + wiring tests

**Files:**
- Create: `tests/test_models/test_ensemble_decomposed.py`
- Modify: `src/projections/models/ensemble.py` (append new factory after `wr_ensemble`, ~line 593-601)
- Modify: `src/projections/models/__init__.py` (import, `__all__`, `_WR_FACTORIES`)

Scope: net-new factory + registration; no behavior change to `EnsembleModel`. No fit/predict in this task — that's Task 2.

- [ ] **Step 1.1: Write failing factory-wiring tests**

Create `tests/test_models/test_ensemble_decomposed.py` with:

```python
"""Tests for the wr_ensemble_decomposed factory.

Spec: docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md.

Task 1 covers factory wiring + registry. Fit/predict tests (Task 2) live in
the same file and share the synthetic-data helper below.
"""

from __future__ import annotations

from projections.models import (
    POSITION_DISPATCH,
    DecomposedBaselineModel,
    EnsembleModel,
    LightGBMNbModel,
    wr_ensemble_decomposed,
)
from projections.models.ensemble import _EnsembleConfig
from projections.schemas import Position


def test_wr_ensemble_decomposed_returns_ensemble_with_decomposed_child_a() -> None:
    """Factory returns an EnsembleModel whose child A factory yields a
    DecomposedBaselineModel and child B factory yields an LightGBMNbModel.
    """
    model = wr_ensemble_decomposed()
    assert isinstance(model, EnsembleModel)
    assert model.position == Position.WR

    # Children are constructed by the factories on demand; instantiate to
    # verify type (not the same as the lazily-fit children inside fit()).
    config: _EnsembleConfig = model._config
    child_a = config.child_a_factory()
    child_b = config.child_b_factory()
    assert isinstance(child_a, DecomposedBaselineModel), (
        f"child A should be DecomposedBaselineModel, got {type(child_a).__name__}"
    )
    assert isinstance(child_b, LightGBMNbModel), (
        f"child B should be LightGBMNbModel, got {type(child_b).__name__}"
    )


def test_wr_ensemble_decomposed_registered_in_factories() -> None:
    """_WR_FACTORIES has an 'ensemble-decomposed' entry resolving to a freshly
    instantiated EnsembleModel with the decomposed child A wiring.
    """
    wr_dispatch = POSITION_DISPATCH[Position.WR]
    assert "ensemble-decomposed" in wr_dispatch.factories, (
        f"_WR_FACTORIES missing 'ensemble-decomposed'; "
        f"available: {sorted(wr_dispatch.factories)}"
    )
    model = wr_dispatch.factories["ensemble-decomposed"]()
    assert isinstance(model, EnsembleModel)
    child_a = model._config.child_a_factory()
    assert isinstance(child_a, DecomposedBaselineModel)


def test_default_model_class_unchanged_after_registration() -> None:
    """Registering 'ensemble-decomposed' does NOT flip production routing.
    The flip is the §1.3.5 ADOPT outcome (Task 5).
    """
    assert POSITION_DISPATCH[Position.WR].default_model_class == "ensemble"


def test_wr_ensemble_decomposed_in_models_all() -> None:
    """wr_ensemble_decomposed is exported via projections.models.__all__."""
    import projections.models as models_pkg

    assert "wr_ensemble_decomposed" in models_pkg.__all__, (
        f"wr_ensemble_decomposed missing from __all__; got {sorted(models_pkg.__all__)}"
    )
```

- [ ] **Step 1.2: Run tests — expect ImportError**

```bash
.venv/Scripts/python.exe -m pytest tests/test_models/test_ensemble_decomposed.py -v
```

Expected: collection error or `ImportError: cannot import name 'wr_ensemble_decomposed' from 'projections.models'`.

- [ ] **Step 1.3: Add the factory in `ensemble.py`**

Append at the bottom of `src/projections/models/ensemble.py` (after `wr_ensemble`, around line 601):

```python
def wr_ensemble_decomposed() -> EnsembleModel:
    """Construct an unfitted WR ensemble whose child A is decomposed-baseline.

    Differs from `wr_ensemble` only in `child_a_factory`. The pinball-weight-fit
    calibration step (Stage 3 of EnsembleModel.fit) re-runs against the
    decomposed-baseline-vs-lgb-nb children, producing per-stat weights tuned
    to the new child A.

    Spec: docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md.
    """
    return EnsembleModel(
        config=_EnsembleConfig(
            position=Position.WR,
            target_stats=_WR_TARGET_STATS,
            child_a_factory=wr_decomposed_baseline,
            child_b_factory=wr_lightgbm_nb,
        )
    )
```

In the same file, add `wr_decomposed_baseline` to the existing `from projections.models.baseline import (...)` block — wait — `wr_decomposed_baseline` lives in `decomposed_baseline.py`, not `baseline.py`. Add a new import line near the existing `from projections.models.baseline import (` block:

```python
from projections.models.decomposed_baseline import wr_decomposed_baseline
```

Place this import in alphabetical order within the existing `from projections.models...` import block.

- [ ] **Step 1.4: Register in `_WR_FACTORIES` + `__all__`**

In `src/projections/models/__init__.py`:

1. Add `wr_ensemble_decomposed` to the existing `from projections.models.ensemble import (...)` block. The block currently is:

```python
from projections.models.ensemble import (
    EnsembleModel,
    qb_ensemble,
    rb_ensemble,
    te_ensemble,
    wr_ensemble,
)
```

Change to:

```python
from projections.models.ensemble import (
    EnsembleModel,
    qb_ensemble,
    rb_ensemble,
    te_ensemble,
    wr_ensemble,
    wr_ensemble_decomposed,
)
```

2. Insert `"wr_ensemble_decomposed"` into `__all__` in alphabetical position (between `"wr_ensemble"` and `"wr_lightgbm"`):

```python
    "wr_ensemble",
    "wr_ensemble_decomposed",
    "wr_lightgbm",
```

3. Add `"ensemble-decomposed"` entry to `_WR_FACTORIES`. The current dict (lines ~168-175) is:

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

Change to (insert after `"ensemble"`):

```python
_WR_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": wr_baseline,
    "decomposed-baseline": wr_decomposed_baseline,
    "lightgbm": wr_lightgbm,
    "lightgbm-tuned": wr_lightgbm_tuned,
    "lightgbm-nb": wr_lightgbm_nb,
    "ensemble": wr_ensemble,
    "ensemble-decomposed": wr_ensemble_decomposed,
}
```

- [ ] **Step 1.5: Run tests — expect PASS**

```bash
.venv/Scripts/python.exe -m pytest tests/test_models/test_ensemble_decomposed.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 1.6: Run the dispatch-table conformance tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_models/test_position_dispatch.py -v
```

Expected: all existing tests pass. (`test_default_model_class_per_position` pins WR to `"ensemble"` — Task 5 will update it if ADOPT.)

- [ ] **Step 1.7: Lint + typecheck**

```bash
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m ruff format --check src tests
.venv/Scripts/python.exe -m mypy src tests
```

Expected: zero violations across all three.

- [ ] **Step 1.8: Commit**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  src/projections/models/ensemble.py \
  src/projections/models/__init__.py \
  tests/test_models/test_ensemble_decomposed.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(models): wr_ensemble_decomposed factory + registration

Adds a new EnsembleModel factory wired to wr_decomposed_baseline (instead of
wr_baseline) for child A; registers as _WR_FACTORIES["ensemble-decomposed"].
No behavior change to EnsembleModel; default_model_class stays on "ensemble".

Spec: docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md (Task 1).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Synthetic-data fit/predict + mixture-tail sanity

**Files:**
- Modify: `tests/test_models/test_ensemble_decomposed.py` (extend with fit/predict + tail tests)
- Conditionally modify: `src/projections/distributions/mixture.py` (only if Step 2.5 surfaces a numerical bug)

Scope: end-to-end fit on synthetic WR-with-non-zero-targets data; verify the `MixtureDistribution(QuantileDistribution, ParametricNegativeBinomial)` code path round-trips; sanity-check tail quantiles. This is the task most likely to surface a `mixture.py` numerical issue.

- [ ] **Step 2.1: Add the synthetic-data helper**

Append to `tests/test_models/test_ensemble_decomposed.py`:

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

from projections.distributions import (
    MixtureDistribution,
    QuantileDistribution,
    unpack_per_stat_params,
)
from projections.distributions.parametric import ParametricNegativeBinomial
from projections.schemas import (
    _PYARROW_STR,
    DistributionFamily,
    ProjectionWeeklySchema,
    Ruleset,
    Stat,
    WeeklyStatsSchema,
)


def _synthetic_wr_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic WR features + weekly_stats with positive `targets` per row,
    sized for ensemble fit (≥3 seasons, enough rows per season for lgb-nb).

    Mirrors test_ensemble_model_smoke._build_synthetic_data's size pattern
    (4 seasons x 17 weeks x 20 players = 1360 rows) but populates `targets`
    correlated with a feature so DecomposedBaselineModel's volume sub-model
    has signal. `receptions = Binomial(targets, 0.65)`; receiving yards/tds
    derived similarly so the truth columns are coherent.
    """
    from projections.models import POSITION_DISPATCH

    rng = np.random.default_rng(seed=20260515)
    feature_schema = POSITION_DISPATCH[Position.WR].feature_schema

    rows: list[dict[str, object]] = []
    for season in range(2018, 2022):  # 4 seasons
        for week in range(1, 18):
            for p in range(20):
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
            df[col_name] = rng.uniform(0.0, 0.5, size=len(df)).astype(np.float64)
    features = feature_schema.validate(df)

    ws = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    ws["position"] = "WR"
    n = len(ws)
    # Targets correlated with a feature; lambda in [3, 12] to mirror real WR
    # target rates. Decomposed-baseline's mask = (targets > 0) holds.
    target_lambda = rng.uniform(3.0, 12.0, size=n)
    targets = np.maximum(1, rng.poisson(target_lambda)).astype(np.int64)
    receptions = np.array(
        [int(rng.binomial(t, 0.65)) for t in targets], dtype=np.int64
    )
    ws["targets"] = targets
    ws["receptions"] = receptions
    ws["receiving_yards"] = np.maximum(
        0.0, receptions * rng.normal(11.0, 3.0, size=n)
    ).astype(np.float64)
    ws["receiving_tds"] = np.where(
        rng.uniform(0, 1, size=n) < np.minimum(targets * 0.05, 1.0), 1, 0
    ).astype(np.int64)
    ws["rushing_yards"] = np.clip(
        rng.normal(2.0, 4.0, size=n), 0.0, 100.0
    ).astype(np.float64)
    ws["rushing_tds"] = np.maximum(0, rng.poisson(0.05, size=n)).astype(np.int64)
    ws["fumbles_lost"] = np.maximum(0, rng.poisson(0.05, size=n)).astype(np.int64)
    # Zero-fill QB stats (required by WeeklyStatsSchema).
    ws["passing_yards"] = 0.0
    ws["passing_tds"] = np.int64(0)
    ws["interceptions"] = np.int64(0)

    # Zero-fill any remaining required columns.
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
```

(Add `from projections.schemas import Position` if not already imported.)

- [ ] **Step 2.2: Write the fit+predict round-trip test (failing)**

Append:

```python
def test_wr_ensemble_decomposed_fit_predict_round_trip(tmp_path: Path) -> None:
    """End-to-end fit + predict on synthetic WR data.

    Validates the production code path:
      DecomposedBaselineModel (child A) emits QuantileDistribution for
      RECEPTIONS at persist time; EnsembleModel wraps each per-stat dist
      from child A with the matching dist from child B (LightGBMNbModel,
      ParametricNegativeBinomial for counting stats) inside a
      MixtureDistribution. The codec's MIXTURE branch packs each component
      via _pack_single — including the QuantileDistribution branch for
      RECEPTIONS.
    """
    features, weekly_stats = _synthetic_wr_inputs()
    model = wr_ensemble_decomposed()
    model._config.weights_dir = tmp_path
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].head(5).copy()
    out = model.predict_distribution(test_features, Ruleset.espn_ppr())

    ProjectionWeeklySchema.validate(out)
    assert len(out) == 5
    assert (out["family"] == DistributionFamily.MIXED.value).all()
    assert (out["position"] == "WR").all()
    assert (out["model_id"].str.startswith("ensemble:wr:")).all()

    # Round-trip a row's params blob: per-stat dists must be Mixture, and
    # for RECEPTIONS specifically, component_a must be QuantileDistribution
    # (the persistable form of the decomposed FrozenSampledDistribution).
    decoded = unpack_per_stat_params(bytes(out.iloc[0]["params"]))
    assert Stat.RECEPTIONS in decoded
    rec_dist = decoded[Stat.RECEPTIONS]
    assert isinstance(rec_dist, MixtureDistribution), (
        f"RECEPTIONS dist should be MixtureDistribution, got {type(rec_dist).__name__}"
    )
    assert isinstance(rec_dist.component_a, QuantileDistribution), (
        f"RECEPTIONS component_a should be QuantileDistribution (from "
        f"decomposed-baseline child), got {type(rec_dist.component_a).__name__}"
    )
    # Component B is from lgb-nb. For RECEPTIONS (a count stat) this is
    # ParametricNegativeBinomial.
    assert isinstance(rec_dist.component_b, ParametricNegativeBinomial), (
        f"RECEPTIONS component_b should be ParametricNegativeBinomial, "
        f"got {type(rec_dist.component_b).__name__}"
    )
```

- [ ] **Step 2.3: Write the code_hash divergence test (failing)**

Append:

```python
def test_wr_ensemble_decomposed_code_hash_differs_from_wr_ensemble(
    tmp_path: Path,
) -> None:
    """The two ensembles must have distinct code_hash + model_id post-fit
    because their child A code_hashes differ. This separates their
    data/ensemble_weights/ artifacts cleanly.
    """
    from projections.models import wr_ensemble
    from projections.models.ensemble import _weights_artifact_path

    features, weekly_stats = _synthetic_wr_inputs()

    ensemble_standard = wr_ensemble()
    ensemble_standard._config.weights_dir = tmp_path / "standard"
    ensemble_standard.fit(features, weekly_stats)

    ensemble_decomposed = wr_ensemble_decomposed()
    ensemble_decomposed._config.weights_dir = tmp_path / "decomposed"
    ensemble_decomposed.fit(features, weekly_stats)

    assert ensemble_standard.code_hash != ensemble_decomposed.code_hash, (
        "ensembles with different child A factories must have different "
        "code_hash; got identical hash {h}".format(h=ensemble_standard.code_hash)
    )
    assert ensemble_standard.model_id != ensemble_decomposed.model_id

    # Sanity: weight artifacts land at distinct paths (model_id contains hash).
    standard_path = _weights_artifact_path(
        ensemble_standard._config.weights_dir, ensemble_standard.model_id
    )
    decomposed_path = _weights_artifact_path(
        ensemble_decomposed._config.weights_dir, ensemble_decomposed.model_id
    )
    assert standard_path != decomposed_path
    assert standard_path.exists()
    assert decomposed_path.exists()
```

- [ ] **Step 2.4: Write the mixture tail-quantile sanity tests (failing)**

Append:

```python
def test_mixture_of_quantile_and_negative_binomial_tail_quantiles_are_finite() -> None:
    """Direct construction of MixtureDistribution(QuantileDistribution,
    ParametricNegativeBinomial). Verifies the mixture's quantile lookups
    return finite values across q ∈ {0.10, 0.50, 0.90, 0.99}.

    This is the code path that decomposed-baseline-as-child-A introduces
    to production. If this fails, the fix scope is mixture._bracket_for_components
    (likely needs to query QuantileDistribution components at q ∈ [0.05, 0.95]
    rather than wider tails since the persisted knots cap at q=0.95).
    """
    quantile_dist = QuantileDistribution(
        quantiles=np.arange(0.05, 0.96, 0.05),
        values=np.linspace(0.0, 10.0, 19),
    )
    nb_dist = ParametricNegativeBinomial(mean=5.0, dispersion=2.0)
    mix = MixtureDistribution(
        component_a=quantile_dist, component_b=nb_dist, weight=0.5
    )

    for q in (0.10, 0.50, 0.90, 0.99):
        value = mix.quantile(q)
        assert np.isfinite(value), (
            f"MixtureDistribution(Quantile, NB).quantile({q}) returned "
            f"non-finite value: {value}"
        )

    # Mean + std are computed via sampling under MixtureDistribution;
    # both should be finite.
    assert np.isfinite(mix.mean())
    assert np.isfinite(mix.std())


def test_fit_weight_for_stat_handles_quantile_distribution_components() -> None:
    """_fit_weight_for_stat must produce a finite per-stat weight when
    components_a is a list of QuantileDistribution and components_b is a
    list of ParametricNegativeBinomial. Exercises _bracket_for_components +
    _quantile_with_bracket against a Quantile component, which is the
    inner-loop machinery during EnsembleModel.fit Stage 3.
    """
    from projections.models.ensemble import _fit_weight_for_stat

    rng = np.random.default_rng(seed=1234)
    n_rows = 20
    components_a: list[QuantileDistribution] = []
    components_b: list[ParametricNegativeBinomial] = []
    for _ in range(n_rows):
        values = np.sort(rng.uniform(0.0, 10.0, size=19))
        components_a.append(
            QuantileDistribution(
                quantiles=np.arange(0.05, 0.96, 0.05),
                values=values,
            )
        )
        components_b.append(
            ParametricNegativeBinomial(
                mean=float(rng.uniform(2.0, 8.0)),
                dispersion=float(rng.uniform(1.0, 3.0)),
            )
        )
    actuals = rng.integers(0, 15, size=n_rows).astype(np.float64)

    weight = _fit_weight_for_stat(
        components_a=components_a,
        components_b=components_b,
        actuals=actuals,
    )
    assert 0.001 <= weight <= 0.999, (
        f"_fit_weight_for_stat returned out-of-bounds weight: {weight}"
    )
    assert np.isfinite(weight)
```

- [ ] **Step 2.5: Run tests — interpret the result**

```bash
.venv/Scripts/python.exe -m pytest tests/test_models/test_ensemble_decomposed.py -v
```

Two outcomes are possible:

**Outcome A: all tests pass.** Proceed directly to Step 2.7 (skip 2.6). Mixture machinery handles the new component pair cleanly.

**Outcome B: one or both mixture-tail tests fail with non-finite values, ValueError from scipy's brentq, or pinned-edge weights.** Proceed to Step 2.6 to scope a fix in `mixture.py`.

- [ ] **Step 2.6: (conditional) Fix `_bracket_for_components` for `QuantileDistribution`**

Read `src/projections/distributions/mixture.py` to locate `_bracket_for_components`. If the helper queries components at `q ∈ {0.001, 0.999}` (or similarly tight tails), it will overshoot a `QuantileDistribution`'s persisted knot range (q=0.05 to q=0.95). The fix: introduce a per-component max-knot detection or simply cap the bracket queries to q ∈ [0.05, 0.95] for `QuantileDistribution` components.

Specific guidance:
- If the existing helper just does `a.quantile(0.001)` and `b.quantile(0.999)`, introduce a tiny `_safe_bracket_q(dist)` helper that returns `(0.05, 0.95)` when `isinstance(dist, QuantileDistribution)` and `(0.001, 0.999)` otherwise.
- Add a focused unit test in `tests/test_distributions/test_mixture.py` (create if missing) exercising the fix at a `QuantileDistribution`-component edge case.
- Re-run the Step 2.5 tests and verify they now pass.

Commit this fix as its own commit BEFORE proceeding to Step 2.7:

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  src/projections/distributions/mixture.py \
  tests/test_distributions/test_mixture.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
fix(mixture): bracket lookup must respect QuantileDistribution knot range

MixtureDistribution.quantile uses _bracket_for_components which previously
queried each component at q ∈ {0.001, 0.999}. QuantileDistribution's
persisted knots cap at q=0.95, so the wide bracket overshot the supported
range and caused [describe symptom — NaN/inf, pinned-edge, ValueError].

Caps the per-component bracket query to q ∈ [0.05, 0.95] when the
component is a QuantileDistribution; preserves the wider bracket for
parametric components.

Surfaced by the wr_ensemble_decomposed integration: it is the first
production code path that mixes QuantileDistribution with
ParametricNegativeBinomial.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2.7: Lint + typecheck**

```bash
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m ruff format --check src tests
.venv/Scripts/python.exe -m mypy src tests
```

Expected: zero violations.

- [ ] **Step 2.8: Commit Task 2 tests**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  tests/test_models/test_ensemble_decomposed.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
test(models): wr_ensemble_decomposed fit + predict + mixture-tail sanity

Adds end-to-end fit + predict round-trip on synthetic WR data, code_hash
divergence vs wr_ensemble, and direct unit tests on
MixtureDistribution(QuantileDistribution, ParametricNegativeBinomial)
tail-quantile behavior. The latter is the first production code path
mixing a quantile-summary distribution with a parametric one.

Spec: docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md (Task 2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: CLI + harness wiring + CLI test

**Files:**
- Modify: `scripts/backtest.py`
- Modify: `tests/test_scripts/test_backtest_cli.py`

Scope: CLI choice + WR-only positions restriction mirror of PR #36's `decomposed-baseline` handling. No source changes to harness.py (the existing `EnsembleModel` cast covers the new factory's return type).

- [ ] **Step 3.1: Write the failing CLI test**

Append to `tests/test_scripts/test_backtest_cli.py`:

```python
def test_backtest_cli_ensemble_decomposed_restricts_to_wr_only() -> None:
    """When --model ensemble-decomposed is selected, the CLI restricts
    positions to (Position.WR,) — mirroring the decomposed-baseline
    restriction, since ensemble-decomposed depends on
    _WR_FACTORIES["decomposed-baseline"] as child A.
    """
    captured: dict[str, object] = {}

    def fake_run_backtest(**kwargs: object) -> object:
        captured.update(kwargs)
        raise SystemExit(0)

    with mock.patch.object(backtest, "run_backtest", fake_run_backtest):
        with mock.patch.object(
            sys, "argv", ["backtest", "--report", "--model", "ensemble-decomposed"]
        ):
            with pytest.raises(SystemExit) as ex_info:
                backtest.main()
            assert ex_info.value.code == 0

    assert captured.get("model_classes") == ("ensemble-decomposed",)
    assert captured.get("positions") == (Position.WR,)
```

- [ ] **Step 3.2: Run test — expect FAIL**

```bash
.venv/Scripts/python.exe -m pytest tests/test_scripts/test_backtest_cli.py::test_backtest_cli_ensemble_decomposed_restricts_to_wr_only -v
```

Expected: failure because `"ensemble-decomposed"` is not yet in the `--model` `choices`.

- [ ] **Step 3.3: Add `"ensemble-decomposed"` to `scripts/backtest.py`**

Read `scripts/backtest.py` to locate the `--model` argparse `choices=[...]` block and the WR-only positions restriction block. Specifically:

1. Find the `add_argument("--model", ...)` call. Its `choices` list currently includes `"decomposed-baseline"`. Add `"ensemble-decomposed"` to that list (alphabetically near `"ensemble"`).

2. Find the conditional block that sets `positions=(Position.WR,)` when `args.model == "decomposed-baseline"`. Extend it to also fire on `args.model == "ensemble-decomposed"`. If the existing logic uses `if args.model == "decomposed-baseline":`, change to `if args.model in ("decomposed-baseline", "ensemble-decomposed"):`.

(Do not guess at line numbers — the `Read` tool result is the source of truth at execution time. Both edit points should be within the `main()` function near the model-class dispatch.)

- [ ] **Step 3.4: Run CLI tests — expect PASS**

```bash
.venv/Scripts/python.exe -m pytest tests/test_scripts/test_backtest_cli.py -v
```

Expected: all 4 tests pass (3 existing + 1 new).

- [ ] **Step 3.5: Lint + typecheck**

```bash
.venv/Scripts/python.exe -m ruff check src tests scripts
.venv/Scripts/python.exe -m ruff format --check src tests scripts
.venv/Scripts/python.exe -m mypy src tests scripts
```

Expected: zero violations.

- [ ] **Step 3.6: Commit**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  scripts/backtest.py \
  tests/test_scripts/test_backtest_cli.py
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(scripts): backtest --model ensemble-decomposed + WR-only restriction

Mirrors the decomposed-baseline CLI handling shipped in PR #36; the new
factory consumes wr_decomposed_baseline as its child A so the WR-only
restriction applies for the same reason (no factory exists for other
positions).

Spec: docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md (Task 3).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Real-data dual-run backtest + adoption gate

**Files:** (data artifacts; not committed wholesale)
- Create: `data/backtest/run_<ts>/` (one or two run directories from dual-run)
- Create: `reports/adoption_gate_wr_ensemble_decomposed_vs_ensemble.{md,csv}`
- Create: `reports/adoption_gate_wr_ensemble_decomposed_vs_decomposed_baseline.{md,csv}`

Scope: operational — no source edits. Produce the binding + informational cell adoption-gate reports. Expected wall-clock ~25-35 min per PR #36's deviation #3.

- [ ] **Step 4.1: Verify the existing backtest CLI signature**

Run a help check to confirm the dual-run mode is available and to capture the exact arg names used in PR #36:

```bash
.venv/Scripts/python.exe scripts/backtest_dual.py --help
```

Capture the exact `--model-a` / `--model-b` (or equivalent) arg names for use in the invocation in Step 4.2.

- [ ] **Step 4.2: Run the dual-run backtest (ensemble vs ensemble-decomposed)**

The binding cell needs predictions from both `ensemble` and `ensemble-decomposed` on the same WR rows across 2021-2024. Use the same invocation pattern PR #36 documented in its plan-vs-execution deviations § (the Windows absolute-path workaround for `features_root` / `raw_root`):

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/backtest_dual.py \
  --model-a ensemble \
  --model-b ensemble-decomposed \
  --positions wr \
  --eval-years 2021-2024 \
  --features-root C:/Users/alden/FantasyFootball/data/features \
  --raw-root C:/Users/alden/FantasyFootball/data/raw \
  --output-root C:/Users/alden/FantasyFootball/data/backtest
```

(The exact CLI shape may differ slightly from PR #36 — adapt to whatever `--help` revealed in Step 4.1. The principle: produce two `data/backtest/run_<ts>_<model>/results.parquet` files with paired WR rows.)

Expected wall-clock: 25-35 min. Do not interrupt the run; the ensemble's 4-stage fit per fold per year dominates the cost.

Capture the two output run directory paths for use in Step 4.3.

- [ ] **Step 4.3: Generate the binding cell adoption-gate report**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m scripts.adoption_gate \
  --baseline-run data/backtest/run_<ts_ensemble> \
  --candidate-run data/backtest/run_<ts_ensemble_decomposed> \
  --candidate ensemble-decomposed \
  --csv-out reports/adoption_gate_wr_ensemble_decomposed_vs_ensemble.csv
```

This emits both the `.csv` (raw deltas + CIs) and the `.md` (human-readable summary) in `reports/`. Verify both files were produced.

- [ ] **Step 4.4: Re-run for the informational cell (decomposed-baseline as incumbent)**

The informational cell compares the new ensemble against `decomposed-baseline` alone. Either:
- (a) re-run a second backtest with `decomposed-baseline` predictions (if you don't already have a cached run from PR #36's gate)
- (b) or reuse PR #36's run dir if it's still on disk (check `data/backtest/run_wr_decomp_*`)

For (b), confirm the existing `decomposed-baseline` run dir contains 2021-2024 paired WR rows. If yes:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m scripts.adoption_gate \
  --baseline-run data/backtest/run_<ts_decomposed_baseline_from_pr36> \
  --candidate-run data/backtest/run_<ts_ensemble_decomposed_from_step_4.2> \
  --candidate ensemble-decomposed \
  --csv-out reports/adoption_gate_wr_ensemble_decomposed_vs_decomposed_baseline.csv
```

If (a) is required, append `decomposed-baseline` to the dual-run invocation in Step 4.2 as a third model (if supported) or run a second pair-run.

- [ ] **Step 4.5: Verify the report content**

Read both report `.md` files and confirm:
- n_paired is in the expected range (~8400 paired WR rows, per PR #36)
- RMSE delta + 95% CI is present
- Spearman delta + 95% CI is present
- Per-year breakdowns are present
- The reports name the correct candidate/incumbent at the top

If any of these are missing, the `adoption_gate.py` invocation likely had a CLI shape issue — re-run with the correct flags.

- [ ] **Step 4.6: Stage the report files**

The data/backtest/run_* dirs are NOT committed wholesale; only the reports are. Stage:

```bash
git add \
  reports/adoption_gate_wr_ensemble_decomposed_vs_ensemble.md \
  reports/adoption_gate_wr_ensemble_decomposed_vs_ensemble.csv \
  reports/adoption_gate_wr_ensemble_decomposed_vs_decomposed_baseline.md \
  reports/adoption_gate_wr_ensemble_decomposed_vs_decomposed_baseline.csv
```

Also stage any new `data/ensemble_weights/*.json` artifacts generated by the backtest (these are pre-existing convention per PR #36 deviation #4):

```bash
git status data/ensemble_weights/
git add data/ensemble_weights/ensemble_wr_*.json  # only NEW files
```

- [ ] **Step 4.7: Commit reports**

```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
data(adoption-gate): WR ensemble-decomposed vs ensemble + vs decomposed-baseline

Binding cell (vs ensemble): [paste verdict + RMSE delta + CI from the report].
Informational cell (vs decomposed-baseline): [paste verdict + RMSE delta + CI].

Spec: docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md (Task 4).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(The placeholder bracket text is intentional — fill it from the actual reports at commit time.)

---

## Task 5: §1.3.5 outcome execution + summary + PM/TODO

**Files:** (branch-dependent on Task 4 verdict)
- Create: `reports/wr_ensemble_decomposed_summary.md`
- Modify: `project_management.md`, `TODO.md`
- Conditionally modify (ADOPT branch): `src/projections/models/__init__.py`, `tests/test_models/test_position_dispatch.py`, `tests/backtest/model_metrics.json`
- Conditionally modify (REGRESSION branch): `src/projections/models/ensemble.py`, `src/projections/models/__init__.py`, `scripts/backtest.py`, `tests/test_scripts/test_backtest_cli.py`, `tests/test_models/test_ensemble_decomposed.py`

Scope: execute the §1.3.5 outcome matrix per the binding cell verdict from Task 4.

- [ ] **Step 5.1: Read both Task 4 reports and identify the verdict**

Read:
- `reports/adoption_gate_wr_ensemble_decomposed_vs_ensemble.md` — binding cell
- `reports/adoption_gate_wr_ensemble_decomposed_vs_decomposed_baseline.md` — informational cell

Map to the §1.3.5 verdict matrix from the spec:
- **ADOPT (binding)**: RMSE delta.hi_95 < 0 AND Spearman delta.lo_95 > -0.02
- **MARGINAL**: RMSE PASS, Spearman fail
- **DO_NOT_ADOPT**: RMSE fail (CI brackets zero or is positive)
- **REGRESSION**: RMSE CI strictly > 0

Note the magnitude. If binding cell RMSE delta is in the marginal zone (|delta| < 0.005 fpts) with a borderline CI, document the marginal-zone flag in the summary report.

- [ ] **Step 5.2: Write `reports/wr_ensemble_decomposed_summary.md`**

Create the summary report. Use PR #36's `reports/wr_target_decomposition_summary.md` as the structural template. Required sections:

1. **Status** — verdict (ADOPT / MARGINAL / DO_NOT_ADOPT + informational sub-branch / REGRESSION), magnitude, n_paired.
2. **Per-cell verdicts** — table with both cells: RMSE delta + CI, Spearman delta + CI, verdict.
3. **§1.3.5 outcome** — narrative on what was executed (routing flip / no flip / full revert) and why.
4. **Mechanism interpretation** — what does the binding-cell magnitude (combined with the informational cell) tell us about whether ensemble compounds with decomposition or not.
5. **Probe-vs-gate magnitude flag** — was the binding-cell delta < 0.005 fpts? If yes, note it.
6. **Recommended next direction** — per the §1.3.5 outcome branch:
   - ADOPT → close the ensemble-child swap follow-up; next direction is factor-appropriate sub-models or yards/TDs decomposition (deferred-follow-up #2 from PR #36).
   - MARGINAL / DO_NOT_ADOPT informational ADOPT → close the ensemble-child swap direction; next slot is factor-appropriate sub-models per the spec's §1.3.5.
   - DO_NOT_ADOPT with informational MARGINAL or worse → close target-decomposition at the WR cell entirely; require independent mechanism evidence before re-probing.
   - REGRESSION → close + log the mechanism hypothesis.
7. **Deferred follow-ups (named)** — same list from PR #36 unless this run closes one of them.
8. **Plan-vs-execution deviations** — if any.
9. **Reports** — pointers to the two gate reports.

- [ ] **Step 5.3: Execute the §1.3.5 conditional source edits**

Branch on the binding-cell verdict:

**If ADOPT:**

1. Edit `src/projections/models/__init__.py` to change WR's `default_model_class` from `"ensemble"` to `"ensemble-decomposed"`:

```python
    Position.WR: _PositionDispatch(
        factories=_WR_FACTORIES,
        feature_builder=build_wr_features,
        feature_schema=WrFeaturesSchema,
        ngs_stat_type="receiving",
        default_model_class="ensemble-decomposed",  # was "ensemble"; flipped on adoption 2026-05-XX
    ),
```

2. Edit `tests/test_models/test_position_dispatch.py` line 46 to update the `expected` dict:

```python
    expected = {
        Position.QB: "lightgbm-nb",
        Position.RB: "baseline",
        Position.TE: "baseline",
        Position.WR: "ensemble-decomposed",  # flipped on 2026-05-XX adoption
    }
```

3. Run the dispatch tests:

```bash
.venv/Scripts/python.exe -m pytest tests/test_models/test_position_dispatch.py -v
```

Expected: all pass (including `test_production_model_for_returns_expected_class_per_position` — the new factory still returns an `EnsembleModel` instance).

4. Update `tests/backtest/model_metrics.json` snapshot. Read the current file; identify the WR ensemble-production-routed metric values; re-run the relevant production-default snapshot (if there's a script that regenerates it) OR manually update WR's pinned values to match the new production model's metrics from Task 4's binding-cell candidate run. **State explicitly** in the commit message that the snapshot was regenerated from the Task 4 run.

5. Run the full test suite to verify no regressions:

```bash
.venv/Scripts/python.exe -m pytest -v
```

Expected: all tests pass.

**If MARGINAL / DO_NOT_ADOPT (any informational sub-branch except REGRESSION):**

No source edits — `default_model_class` stays on `"ensemble"`. The factory + registration stay in tree as available infrastructure. Skip to Step 5.4 (PM/TODO).

**If REGRESSION (binding) OR informational REGRESSION:**

Full revert. Remove:

1. `wr_ensemble_decomposed` factory from `src/projections/models/ensemble.py` (delete the function added in Task 1.3).
2. `wr_ensemble_decomposed` import + `__all__` entry + `_WR_FACTORIES["ensemble-decomposed"]` from `src/projections/models/__init__.py`.
3. `"ensemble-decomposed"` from `scripts/backtest.py`'s `--model` choices and WR-only restriction.
4. The new test in `tests/test_scripts/test_backtest_cli.py`.
5. `tests/test_models/test_ensemble_decomposed.py` (entire file).

Keep the spec, plan, summary report, and Task 4 gate reports as historical record.

Run lint + typecheck + full test suite after the revert.

- [ ] **Step 5.4: Update `project_management.md` and `TODO.md`**

Add a top-of-file decision-log entry to `project_management.md` (use the established style — read recent entries from PR #36 / #35 / #32 for tone):

```markdown
**2026-05-XX — WR ensemble-decomposed-child verdict: [ADOPT / MARGINAL / DO_NOT_ADOPT / REGRESSION].**
Binding cell `(EnsembleModel-with-decomposed-baseline-child, WR) vs (EnsembleModel, WR)`:
RMSE Δ [+/-X.XXXX] fpts (CI [...]), Spearman Δ [...]. Informational cell vs decomposed-baseline:
[verdict + delta]. Per spec §1.3.5: [executed action — flip default_model_class /
no-flip ship infra / full revert]. Recommended next slot: [factor-appropriate sub-models /
closure / other].
```

Update `TODO.md`:
- Close (✓) or update the "ensemble-child swap (PR #36 follow-up)" entry.
- Add a new follow-up entry per the §1.3.5 outcome (e.g., "factor-appropriate sub-models for catch_rate efficiency" if MARGINAL / DO_NOT_ADOPT informational ADOPT).

- [ ] **Step 5.5: Final verification**

Run the full forced-verification checklist from `CLAUDE.md`:

```bash
.venv/Scripts/python.exe -m pytest -v
.venv/Scripts/python.exe -m mypy src tests
.venv/Scripts/python.exe -m ruff check src tests scripts
.venv/Scripts/python.exe -m ruff format --check src tests scripts
.venv/Scripts/python.exe -m pytest -v -k "ingest or store or schemas"  # integration-seam smoke
```

Expected: all green. Paste a concise summary into the final message per CLAUDE.md "Forced verification — End-of-effort checklist" rule.

- [ ] **Step 5.6: Commit the outcome execution**

The exact `git add` set depends on the branch:

**ADOPT branch:**
```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  src/projections/models/__init__.py \
  tests/test_models/test_position_dispatch.py \
  tests/backtest/model_metrics.json \
  reports/wr_ensemble_decomposed_summary.md \
  project_management.md \
  TODO.md
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
feat(models): flip WR production routing to ensemble-decomposed — §1.3.5 ADOPT

Binding cell verdict ADOPT (RMSE Δ [paste], Spearman Δ [paste]).
Informational cell [paste verdict + delta].

Updates POSITION_DISPATCH[WR].default_model_class from "ensemble" to
"ensemble-decomposed"; updates test_position_dispatch.py's expected dict;
regenerates tests/backtest/model_metrics.json from the Task 4 binding-cell
candidate run.

Spec: docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md (Task 5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**MARGINAL / DO_NOT_ADOPT branch:**
```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  reports/wr_ensemble_decomposed_summary.md \
  project_management.md \
  TODO.md
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
docs(ensemble-decomposed): §1.3.5 outcome — [MARGINAL/DO_NOT_ADOPT]; infra-only ship

Binding cell verdict [MARGINAL/DO_NOT_ADOPT] (RMSE Δ [paste], CI [paste];
Spearman Δ [paste]). Informational cell [paste verdict + delta].

Production WR routing stays on "ensemble". wr_ensemble_decomposed factory +
_WR_FACTORIES["ensemble-decomposed"] registration stay in tree as available
infrastructure. Recommended next slot: [factor-appropriate sub-models / other
per §1.3.5 sub-branch].

Spec: docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md (Task 5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**REGRESSION branch:**
```bash
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git add \
  src/projections/models/ensemble.py \
  src/projections/models/__init__.py \
  scripts/backtest.py \
  tests/test_scripts/test_backtest_cli.py \
  tests/test_models/test_ensemble_decomposed.py \
  reports/wr_ensemble_decomposed_summary.md \
  project_management.md \
  TODO.md
PATH="/c/Users/alden/FantasyFootball/.venv/Scripts:$PATH" git commit -m "$(cat <<'EOF'
revert(ensemble-decomposed): §1.3.5 REGRESSION — full revert

Binding cell verdict REGRESSION (RMSE CI strictly > 0). Removes
wr_ensemble_decomposed factory, _WR_FACTORIES["ensemble-decomposed"]
registration, backtest CLI choice, and unit tests.

Spec + plan + summary report stay as historical record per PR #31 / PR #36
precedent.

Spec: docs/superpowers/specs/2026-05-15-wr-ensemble-decomposed-child-design.md (Task 5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5.7: Open the PR**

```bash
git push -u origin feat/wr-ensemble-decomposed-child

gh pr create --title "feat: WR ensemble-decomposed child A swap" --body "$(cat <<'EOF'
## Summary

- New `wr_ensemble_decomposed()` factory swaps `EnsembleModel`'s child A from
  `wr_baseline` to `wr_decomposed_baseline`; registered as
  `_WR_FACTORIES["ensemble-decomposed"]`.
- Adoption gate dual-cell verdict: [paste binding + informational].
- §1.3.5 outcome: [ADOPT routing flip / infra-only ship / full revert per branch].

## Test plan

- [x] All tests pass: `pytest -v`
- [x] `mypy src tests scripts` — zero violations
- [x] `ruff check src tests scripts` + `ruff format --check src tests scripts` — clean
- [x] Integration-seam smoke (`pytest -k "ingest or store or schemas"`): clean
- [x] Adoption gate verdicts match spec §1.3.5
- [x] Production WR routing: [paste current state — `ensemble` or `ensemble-decomposed`]

## Reports

- `reports/wr_ensemble_decomposed_summary.md` — verdicts, §1.3.5 outcome, mechanism interpretation
- `reports/adoption_gate_wr_ensemble_decomposed_vs_ensemble.{md,csv}` — binding
- `reports/adoption_gate_wr_ensemble_decomposed_vs_decomposed_baseline.{md,csv}` — informational

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review checks (post-plan)

- **Spec coverage:** Tasks 1-5 cover §1.1 (factory + registration + tests), §1.3 (adoption gate), §1.3.5 (outcome execution), §1.4 (success criteria), §3.4 (mixture-tail verification), and §6 (testing). The §3.4 risk is wired into Task 2's mixture-tail tests with a scoped fix path in Step 2.6.
- **Placeholders:** Step 4.7's commit message has `[paste verdict]` placeholders — intentional, since the verdict is only known at execution time. Same pattern in Step 5.6 commit-message bodies. Step 5.2 summary-report sections are described, not pre-written, since they branch on verdict. Step 4.2's CLI invocation has `<ts>` placeholders for the run-directory timestamps which are runtime-generated.
- **Type consistency:** `wr_ensemble_decomposed` name used consistently in all tasks; `EnsembleModel.code_hash`, `model_id`, `_config.weights_dir` all match the existing class surface verified in Task 1. `_fit_weight_for_stat` keyword args match the function signature in `ensemble.py:107`.
- **Scope boundaries:** Each task touches ≤ 5 files per CLAUDE.md "phased execution" rule (Task 5 ADOPT branch touches 6 — `__init__.py` + dispatch test + snapshot + summary + PM + TODO. PM + TODO are documentation files updated together with the source change, which fits the spirit of the rule; flag this for execution if the executor wants to split Task 5 into 5a (source flip) + 5b (PM/TODO docs)).
