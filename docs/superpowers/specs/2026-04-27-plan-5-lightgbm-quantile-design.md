# Plan 5 — LightGBM with Quantile Regression (Model C) — Design

**Status:** approved (brainstorming, 2026-04-27).
**Date:** 2026-04-27
**Author:** alden + claude
**Builds on:** Plan 3e (merged at `b541e5b`, PR #10) for the per-row deterministic seed + per-stat params codec + walk-forward backtest gate it depends on. Closes TODO #26.

**Plan-3 / Plan-5 series context:**

- **Plan 3a (merged at `598ab9c`):** Model A (Ridge) on WR only.
- **Plan 3b (merged at `c4a0401`):** Model A generalized to QB / RB / TE.
- **Plan 3c (merged at `3db71a6`):** Walk-forward backtest harness + snapshot-diff gate.
- **Plan 3d (merged at `fe55d5b`):** Real Monte Carlo season-distribution aggregation; per-row deterministic seeds; per-stat params codec; season-total calibration in the gate.
- **Plan 3e (merged at `b541e5b`):** Calibration tightening. Phase 0 diagnostic + Phase 1 NB-2 for count stats shipped; Phase 2 (Student-t) and Phase 3 (per-tertile bucketing) attempted-and-reverted (infrastructure preserved).
- **Plan 5 (this design):** New model class. LightGBM-based per-stat quantile regression coexisting with Model A under the existing `Model` Protocol.
- **Plan 4 (deferred until model-improvement track is further along):** Public Python API + CLI verbs + free-tier web hosting.
- **Plan 6 (future, depends on this):** Model D — ensemble of Model A + Model C.

---

## 1. Overview

The post-Plan-3e brainstorm (2026-04-27) concluded that within-week parametric calibration tightening has reached its natural ceiling. Empirical investigation showed:

- **Mean prediction quality is the high-leverage lever.** Composite RMSE / Spearman drive every downstream tool (draft, start/sit, DFS cash). Plan 3e moved calibration metrics; it did not move the mean.
- **Cross-week residual correlation is too weak to chase.** Lag-1 autocorrelation of standardized residuals is +0.02 to +0.10 across positions; lag-2+ ≈ noise. AR(1) with the observed lag-1 ρ would explain ≤10% of the season variance gap.
- **Calibration-aware fitting risks distorting tails.** Optimizing variance to hit `[p10, p90]` coverage leaves the upper tail unconstrained — and the upper tail is exactly what DFS GPP construction depends on.

Three model-improvement tracks were identified as the next forward direction (TODOs #3, #23, #26). Plan 5 is the first of them: the LightGBM model class swap. The expected single-step gain is 5-15% RMSE reduction on top of current features, plus calibration improvement as a side effect of training quantiles directly via pinball loss rather than fitting them post-hoc to residuals.

This plan delivers Model C (LightGBM + quantile regression) as a peer of Model A (Ridge), coexisting under the existing `Model` Protocol. Both models train on the same features and target stats; the backtest snapshot is extended to track both side-by-side so the comparison is direct and reversible. Model A is not removed — both feed the eventual Model D ensemble.

### 1.1 Goals (in scope)

- New `LightGBMModel` implementing the existing `Model` Protocol, with per-position factories (`qb_lightgbm`, `rb_lightgbm`, `te_lightgbm`, `wr_lightgbm`) mirroring the `BaselineModel` factory shape.
- New `QuantileDistribution` implementing the `Distribution` Protocol — interpolated CDF/quantile/sample backed by stored `(quantiles, values)` arrays.
- New `DistributionFamily.QUANTILE` enum value + codec branch in `pack_per_stat_params` / `unpack_per_stat_params`.
- `POSITION_DISPATCH` registry extended to carry both factories per position (keyed by model class name).
- Backtest harness extended with `--model {baseline,lightgbm,both}` arg; results.parquet gains a `model_class` column.
- Snapshot file extended to key rows by `(position, year, metric, model_class)`; existing 400 Model A rows preserved; 400 new Model C rows added.
- Default-on smoke test extended to assert both models produce finite metrics.
- `lightgbm` added to `pyproject.toml` dependencies.
- Standalone artifacts retrained for the four positions under Model C; Model A artifacts untouched.

### 1.2 Non-goals (deferred to later plans)

- **No new features.** Same `Wr/Qb/Rb/TeFeaturesSchema` columns as Model A consumes today. Feature additions are TODO #3 (PBP/EPA), TODO #24 (player-trajectory), TODO #25 (weather).
- **No target decomposition.** Same per-stat regression targets as Model A. Volume × efficiency decomposition is TODO #23.
- **No ensembling.** Combining Model A + Model C predictions is Model D / Plan 6.
- **No Bayesian / hierarchical hyperparameter selection.** Hand-set defaults + early stopping; tuning is its own future work if results justify it.
- **No removal or deprecation of Model A.** Both models stay live indefinitely; Model A is the regression floor for Model C and a future ensemble component.
- **No new ingest.** No PBP, no draft picks, no weather plumbing. Strictly model class swap.

### 1.3 Adoption gate

Per the A → C → D modeling roadmap, Model C must beat Model A on the backtest snapshot before being adopted as the production default. "Beat" is defined as:

- **Composite RMSE:** strictly lower on at least 12 of 16 (position, year) cells; not worse by more than 1% on any cell.
- **Spearman top-N:** within ±0.005 of Model A on every cell (we don't expect rank gains; we don't want regressions).
- **Calibration:** weekly mean `[p10, p90]` coverage no worse than Model A's on any cell; mean coverage across cells improves by ≥ 0.02.

The adoption decision is post-merge. Plan 5 ships both models side-by-side; the snapshot makes the comparison data committed and durable. Default model selection is a follow-up config change, not part of this plan.

---

## 2. Architecture

```
src/projections/
├── distributions/
│   ├── parametric.py           [unchanged: Normal, Gamma, NB, StudentT]
│   ├── quantile.py             [NEW: QuantileDistribution]
│   ├── codec.py                [+1 branch each in pack/unpack for QUANTILE]
│   └── __init__.py             [+ QuantileDistribution export]
├── models/
│   ├── base.py                 [unchanged: Model Protocol, compute_code_hash]
│   ├── baseline.py             [unchanged: Model A]
│   ├── lightgbm.py             [NEW: LightGBMModel + per-position factories]
│   └── __init__.py             [POSITION_DISPATCH extended with factories dict]
├── schemas.py                  [+ DistributionFamily.QUANTILE]
└── backtest/
    ├── harness.py              [+ model selector arg; iterate over selected models]
    └── snapshot.py             [+ model_class column in snapshot rows]
```

`POSITION_DISPATCH` is extended so each position carries factories for both model classes:

```python
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

The `factories` dict is keyed by the model class name string (`"baseline"`, `"lightgbm"`); the value is a zero-arg callable returning an unfitted instance of that class. Single source of truth for "which model classes does the system know about."

CLI scripts (`scripts/train_baseline.py`, `scripts/sanity_check_baseline.py`, `scripts/predict_2024.py`) gain a `--model {baseline,lightgbm}` arg; each script reads `POSITION_DISPATCH[position].factories[model_arg]()` to instantiate the right model. `scripts/backtest.py` accepts `--model {baseline,lightgbm,both}` (default `both` for a gated run).

---

## 3. Components

### 3.1 `LightGBMModel` (`src/projections/models/lightgbm.py`)

Implements the `Model` Protocol structurally. Per-position factories construct instances configured with that position's `target_stats` and `feature_columns` — same per-position config dataclass shape as `BaselineModel` for consistency.

**Internal state after `fit()`:**

- `self.sub_models: dict[Stat, dict[float, lgb.Booster]]` — keyed by `(stat, quantile)`. 5 quantiles × N stats per position. WR / TE / RB ≈ 30 boosters each; QB ≈ 30; full set ≈ 120 across all positions.
- `self.feature_columns: list[str]` — canonical feature order at training time, persisted with the model.
- `self.train_start: int`, `self.train_end: int` — for `model_id`.
- `self.best_iters: dict[tuple[Stat, float], int]` — early-stopping winning round per sub-model, persisted for traceability.

**`fit(features, weekly_stats)`:**

1. Inner-join on `(gsis_id, season, week)`. Same join logic as `BaselineModel`. Empty-result raises `ValueError`.
2. Carve last training season as validation slice. For training years `[2018, 2019, 2020, 2021, 2022, 2023]`, train rows are `season ≤ 2022` and validation rows are `season == 2023`. Training slice cannot be empty (raises `ValueError` if it is); validation slice cannot be empty (raises `ValueError` if it is).
3. For each `Stat` in `target_stats`, for each `q in [0.05, 0.10, 0.50, 0.90, 0.95]`:
    - Construct `lgb.LGBMRegressor(objective='quantile', alpha=q, **LGBM_DEFAULTS)`.
    - `.fit(X_train, y_train_stat, eval_set=[(X_val, y_val_stat)], callbacks=[lgb.early_stopping(50)])`.
    - Store the trained booster + `best_iter` in `self.sub_models` and `self.best_iters`.

**`predict_distribution(features, ruleset)`:**

1. Validate features against the position's `feature_schema` (with reassignment).
2. Verify `set(features.columns) ⊇ set(self.feature_columns)`; raise `ValueError` on mismatch with a diagnostic message.
3. For each row, for each `Stat`:
    - Predict 5 quantiles via the 5 sub-models.
    - **Sort the 5 predictions per row** to enforce non-crossing.
    - **Clip to `[0, ∞)`** if the stat is configured as `non_negative` (count stats; `receptions`).
    - Construct `QuantileDistribution(quantiles=[0.05, 0.10, 0.50, 0.90, 0.95], values=[v5, v10, v50, v90, v95])`.
4. Pass per-stat `Distribution` dict through `score_distribution(per_stat_dists, ruleset, derive_row_seed(row))` — unchanged scoring layer.
5. Return `ProjectionWeeklySchema`-validated DataFrame with `family="QUANTILE"`, `params=pack_per_stat_params(per_stat_dists)`, plus the same `mean / p10 / p50 / p90 / model_id / generated_at` columns Model A produces.

**`save` / `load`:** joblib serialization. `lgb.Booster` is joblib-serializable.

**`model_id`:** `"lightgbm:<pos>:<8-char-code-hash>:<train-start>-<train-end>"`. Same shape as Model A's prefix; different prefix string for trace clarity. `code_hash_files` covers:

- `models/lightgbm.py`
- `models/base.py`
- `distributions/quantile.py`
- `distributions/codec.py`
- `distributions/parametric.py` (QuantileDistribution interpolation may borrow numerical helpers; safe to include)
- `features/<pos>.py`
- `features/_shared.py`, `features/_rolling.py`, `features/_opponent.py`
- `scoring/score.py`, `scoring/score_distribution.py`

11 files. Captures everything whose change should invalidate a Model C artifact: the model class itself, its model-base helpers, the new distribution and codec it emits, the parametric distributions whose helpers it may reference, the position's feature builder + shared feature helpers, and the scoring layer that composes per-stat distributions.

**Per-stat `non_negative` config:** A small flag in the per-stat config struct. Defaults: `True` for receptions and all `*_tds` / interceptions / fumbles_lost; `False` for all yards stats (which can be negative from sacks/TFL/kneels). Single source of truth alongside the existing `target_stats` and `feature_columns` per-position config.

### 3.2 `QuantileDistribution` (`src/projections/distributions/quantile.py`)

Implements the `Distribution` Protocol. Constructed with sorted `(quantiles, values)` arrays.

```python
class QuantileDistribution:
    def __init__(self, quantiles: NDArray[np.float64], values: NDArray[np.float64]) -> None:
        # Validates: quantiles strictly ascending in (0, 1); values non-decreasing;
        # len(quantiles) == len(values); both ≥ 2.
        ...

    def quantile(self, q: float) -> float:
        # q in [q_min, q_max]: linear interpolation between adjacent stored knots.
        # q < q_min or q > q_max: linear extrapolation from the two nearest knots.
        ...

    def mean(self) -> float:
        # Trapezoid integration: E[X] = ∫_0^1 quantile(p) dp on a 100-point grid over [0.01, 0.99].
        ...

    def std(self) -> float:
        # Same integration approach for E[X^2] - mean^2.
        ...

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        # Inverse-CDF sampling: u = rng.uniform(0, 1, n); return np.interp(u, quantiles, values).
        # Vectorized; no scipy dependency.
        ...
```

Pure-functional, ~80 lines + tests. No scipy dependency (numpy.interp is sufficient). Numerical-integration tolerance for `mean()`/`std()` is O(1/N²) → adequate for E[X] given the use cases (Monte Carlo aggregation re-samples via inverse CDF, not via mean/std).

### 3.3 Codec branch

`DistributionFamily.QUANTILE` enum value added to `schemas.py`. One branch in `pack_per_stat_params` and one in `unpack_per_stat_params` for the new family. Persisted shape:

```
{"family": "QUANTILE", "quantiles": [0.05, 0.10, 0.50, 0.90, 0.95], "values": [v5, v10, v50, v90, v95]}
```

Quantiles array is fixed across all rows in a given Plan 5 model output, but persisted per-row for codec self-containment (future Plan 6 ensembles may produce different quantile grids per model component).

### 3.4 LightGBM hyperparameter defaults

Module-level constant in `models/lightgbm.py`:

```python
LGBM_DEFAULTS: Final[dict[str, object]] = {
    "n_estimators": 2000,         # capped; early-stop usually picks 100-500
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "verbose": -1,
    "random_state": 42,           # deterministic across runs
}
EARLY_STOPPING_ROUNDS: Final[int] = 50
QUANTILE_GRID: Final[tuple[float, ...]] = (0.05, 0.10, 0.50, 0.90, 0.95)
```

Single source of truth; all 30 sub-models per position share these defaults. `random_state=42` makes the LightGBM training deterministic across runs — combined with the existing per-row seed for `score_distribution`, the entire prediction pipeline is reproducible.

---

## 4. Data flow

### 4.1 Training (one position, one fold)

```
features parquet                    weekly_stats parquet
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
              inner-join on (gsis_id, season, week)
                       ▼
        split: train years vs validation year (last train season)
                       ▼
        For each (Stat, q in QUANTILE_GRID):
          ├─ lgb.LGBMRegressor(objective='quantile', alpha=q, **LGBM_DEFAULTS)
          ├─ .fit(X_train, y_train_stat,
          │       eval_set=[(X_val, y_val_stat)],
          │       callbacks=[lgb.early_stopping(50)])
          └─ store (booster, best_iter)
                       ▼
              joblib-serialize → models/artifacts/lightgbm-<pos>-<train-start>-<train-end>-<code-hash>.joblib
```

For WR (`target_stats = (receptions, receiving_yards, receiving_tds, rushing_yards, rushing_tds, fumbles_lost)`), 30 booster fits per fold. Each booster on ~6-15K rows trains in <5s with the configured defaults; one fold ≈ 2-3 minutes per position. Full backtest (4 positions × 4 held-out years × Model C) ≈ 30-50 minutes. With `--model both` the gate also re-runs Model A (Plan 3d's 292s baseline), keeping total runtime within an acceptable opt-in pre-PR window.

### 4.2 Prediction (one position, one slate)

```
features parquet (target weeks)
        ▼
For each row, for each Stat:
  ├─ predict 5 quantiles via the 5 sub-models
  ├─ sort to enforce non-crossing
  ├─ clip to [0, ∞) if stat is non_negative
  └─ wrap in QuantileDistribution(quantiles, values)
        ▼
For each row:
  per_stat_dists = {Stat.RECEPTIONS: QD(...), Stat.RECEIVING_YARDS: QD(...), ...}
        ▼
score_distribution(per_stat_dists, ruleset, derive_row_seed(row))
  ├─ samples 200 draws from each per-stat QuantileDistribution via inverse-CDF
  ├─ scores each sample under ruleset
  └─ returns composite mean, p10, p50, p90 in fantasy points
        ▼
ProjectionWeeklySchema row:
  ├─ family = "QUANTILE"
  ├─ params = pack_per_stat_params({Stat.X: QD(...), ...})  → msgpack blob
  ├─ mean, p10, p50, p90, model_id, generated_at, ...
        ▼
Validated DataFrame returned by predict_distribution()
```

The entire prediction path beneath `score_distribution` is **unchanged**. `QuantileDistribution` satisfies the `Distribution` Protocol structurally, so `score_distribution` calls `.sample(n, rng)` and gets samples drawn from the interpolated quantile function. No changes in `scoring/`, `aggregation/`, or any downstream consumer.

### 4.3 Backtest harness extension

`scripts/backtest.py` gains `--model {baseline,lightgbm,both}` (default `both` for gated runs):

```python
selected_models = parse_model_arg(args.model)   # ["baseline"], ["lightgbm"], or ["baseline", "lightgbm"]
for position in positions:
    for test_year in held_out_years:
        for model_class in selected_models:
            model = POSITION_DISPATCH[position].factories[model_class]()
            model.fit(train_features, train_stats)
            metrics = score_metrics(model.predict_distribution(test_features), test_stats)
            for metric_name, value in metrics.items():
                results.append({
                    "position": position.value,
                    "year": test_year,
                    "metric": metric_name,
                    "model_class": model_class,
                    "value": value,
                })
```

Per-row `data/backtest/run_<ts>/results.parquet` gains a `model_class` column. The `--report` mode aggregates by `(position, year, model_class)` and prints a side-by-side comparison table.

### 4.4 Snapshot extension

`tests/backtest/baseline_metrics.json` is renamed to `tests/backtest/model_metrics.json`. Each row is now keyed by `(position, year, metric, model_class)`. Existing 400 Model A rows are preserved exactly (same metric values; only the new `model_class="baseline"` column added); 400 new Model C rows added under `model_class="lightgbm"`. Total: 800 rows.

The tolerance classifier in `backtest/snapshot.py` operates per-row regardless of `model_class`, so no logic changes — only the row identity widens. The default-on smoke test asserts both models produce metrics for the (WR, 2024) cell.

The first snapshot regeneration after Plan 5 lands establishes the Model C baseline rows; subsequent runs regress against them with the same tolerance rules used today.

---

## 5. Error handling

### 5.1 `fit()` failure modes

- **Empty training data** (no rows after inner-join): `ValueError("Empty training set after feature/weekly_stats join")`.
- **Missing target stat in `weekly_stats`** (someone deleted a column): `KeyError(f"Stat {stat.value} not in weekly_stats columns")` with a hint to check schema.
- **Insufficient training years to carve a validation slice** (training data spans only one season — the last-season carve would empty the training slice). Raise `ValueError("Need ≥2 training seasons for early-stopping validation slice; got {n_seasons}")`. Backtest folds always satisfy this (held-out years 2021-2024 → ≥3 train seasons each); the check is defense for ad-hoc training.
- **LightGBM convergence failure / `best_iter == 0`** (early stopping fires immediately on a degenerate sub-model). Don't abort the whole fit; log a warning to stderr and proceed. The sub-model still predicts (just at the constant baseline) and the resulting `QuantileDistribution` reflects that.

### 5.2 `predict_distribution()` failure modes

- **Feature column mismatch** (post-fit, the input features have different columns than training). `set(features.columns)` vs `self.feature_columns`; raise `ValueError(f"Feature columns differ from training: missing={...}, extra={...}")`. Same check `BaselineModel` performs.
- **Quantile crossing on a row** (predicted p10 > p50, etc.). **Always sort per-row before constructing `QuantileDistribution`**; never raise. Sort is deterministic and idempotent. Log per-fit metric `pct_rows_with_crossing` to per-row results parquet for diagnostics — not a snapshot column, but useful when investigating odd predictions.
- **Negative predictions for non-negative stats** (LightGBM doesn't enforce ≥ 0). Clip predicted quantiles to `[0, ∞)` per the per-stat `non_negative` config flag before constructing `QuantileDistribution`.

### 5.3 `QuantileDistribution` invariants

Constructor validates:

- `quantiles` is sorted ascending, all in `(0, 1)` (open interval — endpoints would imply known bounded support).
- `values` is non-decreasing (after the per-row sort upstream this holds; constructor still checks).
- `len(quantiles) == len(values)` and both ≥ 2 (need at least two knots to interpolate).

Behavior:

- `quantile(q)` for `q in [q_min, q_max]`: linear interpolation between adjacent knots.
- `quantile(q)` for `q < q_min` or `q > q_max`: linear extrapolation from the two nearest knots. Tail beyond p5 / p95 is small but non-zero; extrapolation produces a continuous distribution rather than mass-at-endpoints.
- `mean()` / `std()`: trapezoid integration on a 100-point grid over `[0.01, 0.99]`. Coarser than knot resolution elsewhere but sufficient for E[X]; error O(1/N²).
- `sample(n, rng)`: vectorized inverse-CDF via `np.interp`.

### 5.4 Snapshot drift handling

The 400 new Model C rows are *new* in the snapshot; no prior baseline to drift against on first run. The tolerance classifier already handles new rows (treats absent prior as no-drift). The first `--update-snapshot` after Plan 5 lands establishes the Model C baseline.

---

## 6. Testing

### 6.1 `QuantileDistribution` (`tests/test_distributions/test_quantile.py`)

- **Constructor invariants**: rejects unsorted quantiles, non-monotone values, mismatched lengths, length < 2, q outside (0, 1).
- **`quantile(q)` interpolation**: at-knot returns exact stored value; midpoint returns midpoint of values; extrapolation beyond q_min / q_max linearly extends.
- **`mean()` / `std()`**: against constructed Normal-shaped quantile set (`scipy.stats.norm.ppf(q, loc=10, scale=2)`); assert mean ≈ 10 within 0.05 and std ≈ 2 within 0.05. Numerical-integration tolerance, not bit-exact.
- **`sample(n, rng)`**: with seeded rng, samples are deterministic; empirical quantiles of 10K samples match stored quantiles within ~0.1.
- **Sample-then-fit round-trip**: sample 10K from a known distribution → re-estimate quantiles from samples → assert close to original within tolerance.

### 6.2 Codec round-trip (`tests/test_distributions/test_codec.py`)

- Add `test_codec_round_trip_quantile`: pack `{Stat.X: QuantileDistribution(...)}` → unpack → assert distributions are equivalent (quantiles + values arrays match exactly).
- Add `test_codec_round_trip_mixed_with_quantile`: mixed-family per-stat dict including QuantileDistribution alongside NORMAL / GAMMA / NB to confirm the codec doesn't regress on existing families when the new branch is added.

### 6.3 `LightGBMModel` (`tests/test_models/test_lightgbm*.py`)

Mirror `tests/test_models/test_baseline*.py` shape:

- **`test_lightgbm.py`** (cross-cutting):
    - `fit` produces 5 sub-models per stat per position; `best_iters` populated; `model_id` present and stable.
    - `predict_distribution` output validates against `ProjectionWeeklySchema`; `family` column is `"QUANTILE"` for every row; `params` blob round-trips through codec.
    - Quantile-crossing test: construct synthetic data where one stat shows crossing; assert sort is applied and no exception raised.
    - Negative-prediction clip test: count-stat features that produce negative quantile predictions are clipped to 0 before `QuantileDistribution` construction.
    - Save / load round-trip: fit → save → load → predict produces identical output.
    - `model_id` stable across runs given same training data + same code state.
    - **Empty-validation-slice guardrail**: training on a single season raises `ValueError`.

- **Per-position files** (`test_lightgbm_qb.py`, `_rb.py`, `_te.py`, `_wr.py`):
    - Smoke fit on synthetic per-position fixtures (reuse Plan 3a / 3b fixtures); assert per-stat sub-models exist for that position's `target_stats`; assert prediction output non-empty and validates.

- **Smoke parametrized across positions** (`tests/test_models/test_lightgbm_smoke.py`): single fit + predict for each of 4 positions, ~30s total.

### 6.4 Backtest harness (`tests/test_backtest/`)

- Existing tests pass unchanged (Model A path).
- New `test_harness_lightgbm.py`: end-to-end fold for (WR, 2024) on Model C; assert metrics produced + finite + match `ProjectionWeeklySchema` row shape.
- `test_harness_dual_model.py`: invoke harness with `--model both`; results.parquet has `model_class` column with both values present; per-cell metrics for both models.
- Snapshot file shape change: assert `model_metrics.json` rows are keyed by `(position, year, metric, model_class)`.

### 6.5 Default-on smoke (`tests/backtest/test_backtest_smoke.py`)

Extend the existing (WR, 2024) smoke to assert *both* models produce finite metrics. Total runtime stays in the ~30s budget — Model C fit on one position is well under 30s.

### 6.6 Backtest snapshot gate (`pytest -m backtest --run-backtest`)

Full opt-in backtest produces 800 metric rows (400 Model A + 400 Model C). Model C portion is ~30-50 minutes; Model A portion is unchanged (Plan 3d's 292s); combined wall-time is ~35-55 minutes for `--model both`. Acceptable for an opt-in pre-PR gate.

### 6.7 Type / lint conformance

- `mypy src tests` clean — `LightGBMModel` types annotated, including the `lgb.Booster` import.
- `ruff check src tests` and `ruff format --check src tests` clean.
- `lightgbm` added to `pyproject.toml` dependencies as `lightgbm>=4.0` (matches the `>=` convention used by every other dep in the file).

### 6.8 Plan 3e infrastructure not exercised by Plan 5

The Phase 2 / 3 reverted infrastructure (`ParametricStudentT`, bucketing helpers, widened `variance_params` type, related tests) is **not touched** by Plan 5; it stays as future-infrastructure. Tests for that infrastructure continue to run unchanged in regular `pytest` invocations.

---

## 7. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-27 | Side-by-side coexistence with Model A (not replacement) | Eventual Model D ensemble requires both; backtest comparison is direct and reversible; deletion forecloses options without saving meaningful work. |
| 2026-04-27 | New `QuantileDistribution` + `DistributionFamily.QUANTILE` (not parametric variance fit) | Realizes the structural calibration benefit of training quantiles directly via pinball loss; sidesteps Plan 3e's central-interval-vs-tails dilemma; ~80 lines. |
| 2026-04-27 | Per-stat sub-models (not composite PPR) | Preserves scoring-layer separation and ruleset flexibility (ESPN_PPR vs DraftKings without retrain). |
| 2026-04-27 | 5 quantiles: p5 / p10 / p50 / p90 / p95 | p10 / p90 keeps existing snapshot metrics; p50 anchors the center; p5 / p95 explicitly carry tail signal critical for DFS GPP. |
| 2026-04-27 | Hand-set defaults + early stopping (not grid search / Optuna) | At "does it beat Ridge at all" stage; defaults usually within a few % of optimal; tuning is its own future improvement if results justify. |
| 2026-04-27 | Validation = last training season; no refit on full window | Mirrors backtest's walk-forward philosophy; one fit per sub-model is simpler than refit-after-early-stopping. Marginal signal from one extra season is small. |
| 2026-04-27 | Uniform quantile regression for all stats including counts | GAMMA's catastrophic count miscalibration was due to PDF inability to put mass at 0; quantile regression has no such constraint and represents zero-mass naturally. |
| 2026-04-27 | Strictly model-class swap (no features, no decomposition) | Keeps backtest comparison clean — any RMSE win attributable to model class, not bundled changes. |
| 2026-04-27 | `POSITION_DISPATCH` extended with `factories: dict[str, Callable]` | Single source of truth for "which model classes the system knows about"; CLI scripts and harness consume the dict by string key. |
| 2026-04-27 | Snapshot file renamed `baseline_metrics.json` → `model_metrics.json` | Single snapshot is the right shape long-term; we're touching this layer anyway; cascades through harness / snapshot references. |
| 2026-04-27 | Quantile crossing handled by per-row sort (not constrained training) | Sort is deterministic and idempotent; affects ~1-3% of rows in practice; constrained training would be invasive and is unnecessary at this stage. |
| 2026-04-27 | Negative-prediction clip via per-stat `non_negative` config flag | Yards stats can be negative (sacks, TFL, kneels); count stats cannot. Per-stat config keeps clip policy where the family policy lives. |
| 2026-04-27 | LightGBM `random_state=42` for determinism | Training-time determinism — re-fitting the same data produces the same boosters across runs. Orthogonal to `score_distribution`'s per-row seeds, which control sampling-time determinism; together the two make the full pipeline reproducible. |

---

## 8. Out-of-scope / explicit future work

- **Hyperparameter tuning** (grid / Optuna): only worth doing if Plan 5 results show enough signal that the marginal 1-3% gain matters.
- **Refitting on full training window after early stopping picks `n_estimators`**: small expected gain; defer until measured.
- **Per-stat Poisson / Tweedie objectives for count stats**: deferred unless quantile-on-counts shows measurable miscalibration.
- **Plan 6 — Model D ensemble**: stack of Model A + Model C predictions. Depends on Plan 5 landing.
- **Default-model selection**: currently the gated backtest runs both. Choosing a default for production projections is a follow-up config change, post-adoption-decision.
- **Removing Model A**: deferred indefinitely. Both models stay live as ensemble components.
