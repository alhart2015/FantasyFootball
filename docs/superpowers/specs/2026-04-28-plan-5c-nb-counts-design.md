# Plan 5c — Hybrid LightGBM with NB-2 for Count Stats — Design

**Status:** approved (brainstorming, 2026-04-28).
**Date:** 2026-04-28
**Author:** alden + claude
**Builds on:** Plan 5b (PR #12, branch `feat/plan-5b-tuning`) — depends on `LightGBMTunedModel`, `data/tuned_params/lightgbm.json`, the `POSITION_DISPATCH.factories` dict, and the snapshot keying by `(position, year, metric, model_class)`. Branched from `feat/plan-5b-tuning`; rebases or merges main forward when Plan 5b lands.

---

## 1. Overview

Plan 5b's diagnostic identified the precise reason Model C (LightGBM quantile regression) and Model C-tuned both lose to Ridge on RB / TE / WR composite RMSE: **`QuantileDistribution` with 5 sorted-and-clipped quantile knots produces a systematically biased empirical mean on zero-inflated count stats** (TDs, interceptions, fumbles_lost). The bias is in the prediction *output format* — linear interpolation between (p50=0, p90=1) places ~40% of distribution mass uniformly in [0, 1], a region that doesn't physically exist for integer count stats — and is invariant to hyperparameter tuning. TD stats carry weight ×6 in fantasy scoring, so even small per-stat RMSE deltas propagate to large composite-RMSE deltas. This is consistent across positions:

| Position | Composite Δ vs A | TD-stat contribution | Yards-stat contribution |
|---|---|---|---|
| QB | -0.118 (C-tuned wins) | rushing_tds +0.095, fumbles +0.036 | passing_yards **-0.097** |
| RB | +0.114 | rushing_tds +0.068 | rushing_yards +0.022 |
| TE | +0.088 | **receiving_tds +0.139** | receiving_yards −0.019 |
| WR | +0.071 | receiving_tds +0.083 | receiving_yards +0.003 |

The fix is structural: replace `QuantileDistribution` with `ParametricNegativeBinomial` (NB-2) for the count stats Plan 3e already routes through NB-2 in Ridge. The NB-routed stats are `PASSING_TDS`, `INTERCEPTIONS`, `RUSHING_TDS`, `RECEIVING_TDS`, `FUMBLES_LOST` (5 `Stat` values; per-position intersection with `target_stats` yields 13 (position, stat) cells: 4 for QB, 3 each for RB / TE / WR — verified against `_<POS>_DIST_FAMILIES` in `src/projections/models/baseline.py`). NB-2 with mean = mu_hat and dispersion fit on training residuals produces an empirical sample mean ≈ mu_hat — no shape-induced bias. Yards/receptions stats are competitive under quantile regression and stay on `QuantileDistribution`.

This plan delivers Model C-NB (`LightGBMNbModel`) as a fourth peer model class coexisting with Models A, C, C-tuned. Hyperparameters reuse Plan 5b's `data/tuned_params/lightgbm.json` for both the count-stat poisson sub-models and the yards-stat quantile sub-models — the diagnostic finding is that the bias lives in the output format, not in fit quality, so re-tuning is unlikely to be the dominant lever.

### 1.1 Goals (in scope)

- New `LightGBMNbModel` subclassing `LightGBMTunedModel`. Trains:
  - **Count stats** (13 cells across 4 positions; QB has 4, RB / TE / WR have 3 each): one `lgb.LGBMRegressor(objective="poisson", **tuned_params)` per stat. The booster's `predict(X)` already returns mu (the mean) in original scale — lgb's poisson objective exponentiates the leaf scores internally. Read `mu_hat = regressor.predict(X)` directly (no `np.exp`); clip to `[_NB_MU_FLOOR, ∞)`. NB-2 dispersion fit on training set via `nb_dispersion_from_residuals(mu_hat=mu_hat_train, actual=y_train)` — the same conditional-MLE estimator Ridge uses (signature unchanged from the relocated helper). Predict-time distribution: `ParametricNegativeBinomial(mu=mu_hat, alpha=dispersion)`.
  - **Yards / receptions stats**: 5-quantile sub-models exactly as `LightGBMTunedModel` does today. Predict-time distribution: `QuantileDistribution(quantiles, sorted_clipped_values)`. **No behavior change** for these stats.
- Per-position factories `qb_lightgbm_nb`, `rb_lightgbm_nb`, `te_lightgbm_nb`, `wr_lightgbm_nb` mirroring the Plan 5b factory shape.
- New `DistributionFamily.MIXED` enum value to mark rows whose per-stat distributions span multiple families. The `ProjectionWeeklySchema.family` column is set to `MIXED` for `LightGBMNbModel` rows; per-stat families remain encoded individually inside `params` (the existing codec already supports this — each stats_blob entry carries its own `family`).
- Public helper `nb_dispersion_from_residuals(y, mu_hat) -> float` extracted from `_negative_binomial_dispersion_from_residuals` in `models/baseline.py` to `distributions/parametric.py`. Behavior-preserving move; `BaselineModel` updated to import from the new location.
- `POSITION_DISPATCH.factories` extended with `"lightgbm-nb"` per position.
- Backtest harness `--model` selector accepts `lightgbm-nb`. The `--model all` value runs `baseline + lightgbm + lightgbm-tuned + lightgbm-nb` (4 model classes).
- Backtest snapshot extended 1136 → 1504 rows (368 new rows under `model_class="lightgbm-nb"`; same 32 `season_calibration_*` rows skipped per the existing SAMPLED_SUMMARY family gate, applying equivalently to MIXED — TODO #28 still open and unrelated).
- Default-on smoke covers all 4 model classes.
- Per-position parametrized smoke for the new class.
- Diagnostic report appended to `project_management.md` with the per-cell A vs C vs C-tuned vs C-NB comparison; gate verdict recorded.
- TODO.md updated with Plan 5c verdict.

### 1.2 Non-goals (deferred)

- **No re-tuning of hyperparameters for the poisson objective.** Reuse Plan 5b's quantile-tuned `data/tuned_params/lightgbm.json`. If the gate fails marginally, re-tuning is a follow-up plan candidate (Plan 5d).
- **No changes to yards-stat training or prediction.** The diagnostic showed yards stats are competitive on RMSE; switching them would muddy the comparison.
- **No removal or deprecation of any existing model class.** Models A, C, C-tuned, C-NB all ship as peers. Pruning Model C (untuned) — now strictly dominated by Model C-tuned — is housekeeping for a follow-up commit, separate from this plan's adoption decision.
- **No widening of `aggregate_to_season` to MIXED or QUANTILE families** (TODO #28). Model C-NB rows skip the same 32 season_calibration_* metrics that Model C and Model C-tuned skip today.
- **No new ingest, no PBP / EPA, no target decomposition, no multi-output training.**
- **No per-fold tuning** (still deferred from Plan 5b).
- **No new training objectives beyond `objective="poisson"` for count stats.** Alternative count objectives (Tweedie, regression with log link) deferred unless poisson proves inadequate.

### 1.3 Adoption gate

Same Plan 5 §1.3 criteria, applied to Model C-NB vs Model A:

- **Composite RMSE:** strictly lower on at least 12 of 16 (position, year) cells; not worse by more than 1% on any cell.
- **Spearman top-N:** within ±0.005 of Model A on every cell.
- **Calibration:** weekly mean `[p10, p90]` coverage no worse than Model A's on any cell; mean coverage across cells improves by ≥ 0.02.

The diagnostic predicts Model C-NB will pass criterion 1 and substantially close criteria 2-3 because:
- **Criterion 1** depends on composite RMSE. NB-2 removes the count-stat empirical-mean bias that drives the current 0.07–0.14 fantasy-point-RMSE delta on TD stats. Yards stats stay where they were (competitive). Model C-tuned already wins QB on RMSE (-1.56% mean delta); RB/TE/WR sit at +1.07% to +1.74% — well within striking distance of −1% if the count-stat losses collapse.
- **Criterion 2** depends on Spearman ordering. NB-2's monotone-in-mu_hat property preserves whatever ordering the underlying poisson booster produces. Model C-tuned was already at 7/16 cells outside the ±0.005 tolerance; expect modest improvement.
- **Criterion 3** depends on `[p10, p90]` calibration. NB-2's analytical quantile function is well-calibrated by construction when dispersion is fit correctly. Plan 3e's NB-2 routing in Ridge produced positive calibration deltas; the same mechanism should apply to LightGBM-NB.

Plan 5c is **complete on either** of:

- **Model C-NB passes the §1.3 adoption gate.** → File a follow-up housekeeping commit prune Model C (untuned), document Model C-NB as the new production default candidate, and proceed to deprecating Models A vs C-NB as appropriate. (The actual production-default switch is its own small change — likely a one-line config — and can land in a separate PR for clarity.)
- **Model C-NB fails the gate.** → Document the per-cell deltas; identify which mechanism the residual gap is attributable to (e.g., is the failure on yards-stat RMSE? On QB-only criterion 2 marginal? On count-stat coverage rather than mean?). Pivot to Plan 6 (ensemble of A + C-tuned + C-NB) or to feature work (TODO #3, TODO #23).

---

## 2. Architecture

```
src/projections/
├── distributions/
│   ├── parametric.py        [+ public nb_dispersion_from_residuals (extracted from baseline.py)]
│   └── codec.py             [unchanged: pack_per_stat_params already supports per-stat families]
├── models/
│   ├── baseline.py          [refactor: import nb_dispersion_from_residuals from parametric.py]
│   ├── lightgbm.py          [unchanged]
│   ├── lightgbm_tuned.py    [unchanged]
│   ├── lightgbm_nb.py       [NEW: LightGBMNbModel + 4 per-position factories]
│   └── __init__.py          [POSITION_DISPATCH.factories += "lightgbm-nb"]
├── schemas.py               [+ DistributionFamily.MIXED]
└── backtest/
    └── harness.py           [unchanged: model_classes is already a generic Iterable[str]]

scripts/
└── backtest.py              [--model accepts "lightgbm-nb"; --model all expands to 4 classes]
```

`POSITION_DISPATCH.factories` extension:

```python
_QB_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": qb_baseline,
    "lightgbm": qb_lightgbm,
    "lightgbm-tuned": qb_lightgbm_tuned,
    "lightgbm-nb": qb_lightgbm_nb,
}
# (same for RB / TE / WR)
```

CLI scripts and the harness consume `factories[model_arg]()` unchanged; only the `--model` arg-parsing surface in `scripts/backtest.py` widens.

---

## 3. Components

### 3.1 `LightGBMNbModel` (`src/projections/models/lightgbm_nb.py`)

Subclasses `LightGBMTunedModel`. Inherits the tuned-params loader, the `_hyperparams_for(stat)` hook, the joblib save/load, the feature/weekly_stats join, and the per-stat scoring composition. Overrides:

- **Internal state extension.** Adds `self._count_models: dict[Stat, lgb.Booster]` and `self._count_dispersions: dict[Stat, float]`. The inherited `self._sub_models` continues to hold the 5-quantile boosters for yards stats. `self._best_iters` continues to record per-(stat, q) early-stopping iterations for yards stats only; a parallel `self._count_best_iters: dict[Stat, int]` is added for count stats.
- **`fit(features, weekly_stats)`.** Same outer plumbing as parent (validate, join, season-split). Per-stat branch:
  - If `stat in COUNT_STATS_FOR_NB`: fit one `lgb.LGBMRegressor(objective="poisson", **self._hyperparams_for(stat))` with early stopping on the last training season. Predict on training rows to obtain `mu_hat_train = np.maximum(regressor.predict(X_train), _NB_MU_FLOOR)` — lgb's poisson `predict()` already returns mu in original scale, so no `np.exp` is applied. Fit dispersion via `nb_dispersion_from_residuals(y_train, mu_hat_train)` from `parametric.py`. Store the booster in `self._count_models[stat]` and the dispersion in `self._count_dispersions[stat]`.
  - Else (yards/receptions): identical behavior to `LightGBMTunedModel.fit` for that stat. The 5 quantile boosters land in `self._sub_models[stat]` exactly as before.
- **`predict_distribution(features, ruleset)`.** Same outer plumbing as parent (feature-column check, schema validate, score_distribution wrap). Per-stat branch:
  - If `stat in COUNT_STATS_FOR_NB`: predict mu directly from `self._count_models[stat]` (lgb poisson `predict()` returns the mean in original scale), clip to `[_NB_MU_FLOOR, ∞)` (matches the floor used in `nb_dispersion_from_residuals`), construct `ParametricNegativeBinomial(mu=mu_hat[row], alpha=self._count_dispersions[stat])` per row.
  - Else: 5 quantile predictions per row, sort, clip to `[0, ∞)` if `stat in non_negative_stats`, wrap in `QuantileDistribution`. Identical to parent.
  - Per-stat dict goes into `score_distribution(per_stat_dists, ruleset, seed=derive_row_seed(row))` exactly as before. Per-row family at the schema level is `DistributionFamily.MIXED.value`.
- **`COUNT_STATS_FOR_NB`.** Module-level `frozenset[Stat]` of the 5 count `Stat` values Plan 3e's `_<POS>_DIST_FAMILIES` routes to NB-2 in Ridge:
  - `Stat.PASSING_TDS`, `Stat.RUSHING_TDS`, `Stat.RECEIVING_TDS`
  - `Stat.INTERCEPTIONS`, `Stat.FUMBLES_LOST`

  Per-position intersection with `target_stats` happens implicitly inside `fit` / `predict_distribution` (each position iterates over its own `target_stats`; only the count subset is routed to NB-2). The resulting per-position cell counts: QB 4 cells (passing_tds, interceptions, rushing_tds, fumbles_lost); RB / TE / WR 3 cells each (receiving_tds, rushing_tds, fumbles_lost). 13 cells total — matching the Ridge NB-2 routing.

- **`code_hash_files`.** Inherits from parent's `_code_hash_files_tuned` and adds `lightgbm_nb.py` and `parametric.py` (the latter because Plan 5c moves a function into it). Final list:
  - `models/lightgbm_nb.py` (new)
  - `models/lightgbm_tuned.py`, `models/lightgbm.py`, `models/base.py`
  - `distributions/quantile.py`, `distributions/codec.py`, `distributions/parametric.py`
  - `features/<pos>.py`, `features/_shared.py`, `features/_rolling.py`, `features/_opponent.py`
  - `scoring/score.py`, `scoring/score_distribution.py`
  - `data/tuned_params/lightgbm.json`

  14 paths.

- **`model_id` prefix.** `"lightgbm-nb:"`. Format: `"lightgbm-nb:<pos>:<8-char-code-hash>:<train-start>-<train-end>"`.

- **Disk artifacts.** `models/artifacts/lightgbm-nb-<pos>-<train-start>-<train-end>-<code-hash>.joblib`.

- **Per-position factories.** Four factories returning `LightGBMNbModel` instances configured with the same per-position `_LightGBMConfig` as `LightGBMTunedModel` (identical `target_stats`, `feature_columns`, `feature_schema`, `non_negative_stats`).

### 3.2 Public `nb_dispersion_from_residuals` helper (`src/projections/distributions/parametric.py`)

Extract the existing `_negative_binomial_dispersion_from_residuals` function from `src/projections/models/baseline.py` (lines 110-155) verbatim. Move it to `distributions/parametric.py` with a public name (drop the leading underscore):

```python
def nb_dispersion_from_residuals(*, mu_hat: np.ndarray, actual: np.ndarray) -> float:
    """Conditional MLE for NB-2 dispersion given per-row mean = mu_hat.

    See ParametricNegativeBinomial for the dispersion semantics. Returns the
    fitted dispersion clipped to a minimum/maximum to keep the predicted
    quantile function well-defined on degenerate inputs.
    """
    # body identical to baseline.py's _negative_binomial_dispersion_from_residuals
```

The associated module-level constants `_NB_MU_FLOOR` and `_NB_DISPERSION_CLIP` move with the function (they're tightly coupled to the dispersion-fitting math, not to Ridge's training pipeline).

`baseline.py` updates the existing call sites — line 261, 269, 609 — to import from `parametric.py`. The internal `_per_bucket_nb_dispersion_from_residuals` helper (line 256) calls into the new public function and is otherwise unchanged.

**Behavior-preserving.** Existing Ridge tests stay green; the move is mechanically a relocation + visibility change.

### 3.3 `DistributionFamily.MIXED` (`src/projections/schemas.py`)

New enum member appended to the existing `DistributionFamily` StrEnum:

```python
class DistributionFamily(StrEnum):
    NORMAL = "NORMAL"
    GAMMA = "GAMMA"
    NEGATIVE_BINOMIAL = "NEGATIVE_BINOMIAL"
    STUDENT_T = "STUDENT_T"
    SAMPLED = "SAMPLED"
    SAMPLED_SUMMARY = "SAMPLED_SUMMARY"
    QUANTILE = "QUANTILE"
    MIXED = "MIXED"  # NEW (Plan 5c) — per-row distribution mixes families per stat
```

Marks rows whose per-stat distributions span multiple families. The codec doesn't care (each per-stat blob carries its own family); the harness's existing SAMPLED_SUMMARY-only gate for season aggregation continues to skip these rows. No codec branch needed.

### 3.4 `POSITION_DISPATCH.factories` extension

In `src/projections/models/__init__.py`:

- Import the four `<pos>_lightgbm_nb` factories and `LightGBMNbModel` from `lightgbm_nb`.
- Add the 5 names to `__all__` in alphabetical position with the existing entries.
- Add `"lightgbm-nb": <pos>_lightgbm_nb` to each of the four `_<POS>_FACTORIES` dicts.

### 3.5 Backtest harness CLI (`scripts/backtest.py`)

Update the `--model` argparse block to extend `choices` with `"lightgbm-nb"`:

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

And extend the `args.model == "all"` branch:

```python
elif args.model == "all":
    model_classes = ("baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb")
```

The harness's `run_backtest(model_classes=...)` is generic over the iterable; no changes inside `src/projections/backtest/harness.py`.

### 3.6 Snapshot extension

`tests/backtest/model_metrics.json` row keying unchanged. Plan 5c adds 368 rows under `model_class="lightgbm-nb"` (same metric set as Model C / Model C-tuned: 23 weekly metrics × 16 cells = 368 rows, skipping 32 season_calibration_* rows per the existing family gate). Pre-Plan-5c row count: 1136. Post-Plan-5c: 1504.

The default-on smoke and the harness end-to-end test both extend to assert all 4 model classes produce metrics; the SAMPLED_SUMMARY-vs-non-SAMPLED_SUMMARY asymmetry is pinned for `lightgbm-nb` the same way it is for `lightgbm` and `lightgbm-tuned`.

---

## 4. Data flow

### 4.1 Training (one position, one fold) — count-stat branch

```
features parquet                weekly_stats parquet
        │                                │
        └────────────┬───────────────────┘
                     ▼
       inner-join on (gsis_id, season, week); position filter
                     ▼
       split: train years vs validation year (last train season)
                     ▼
       For each Stat in target_stats:
         if Stat in COUNT_STATS_FOR_NB:
           ├─ params = self._hyperparams_for(Stat)
           ├─ regressor = lgb.LGBMRegressor(objective="poisson", **_FIXED_PARAMS, **params)
           ├─ regressor.fit(X_train, y_train_stat,
           │                eval_set=[(X_val, y_val_stat)],
           │                callbacks=[lgb.early_stopping(50, verbose=False)])
           ├─ store regressor.booster_ in self._count_models[Stat]
           ├─ mu_hat_train = np.maximum(regressor.predict(X_train), _NB_MU_FLOOR)  # lgb poisson predict returns mu directly
           ├─ dispersion = nb_dispersion_from_residuals(mu_hat=mu_hat_train, actual=y_train_stat)
           └─ store dispersion in self._count_dispersions[Stat]
         else:
           [unchanged: 5-quantile sub-model fit as in LightGBMTunedModel.fit]
                     ▼
              joblib-serialize → models/artifacts/lightgbm-nb-<pos>-<...>.joblib
```

### 4.2 Prediction (one position, one slate)

```
features parquet (target weeks)
        ▼
For each row, for each Stat in target_stats:
  if Stat in COUNT_STATS_FOR_NB:
    ├─ mu_hat = np.maximum(self._count_models[Stat].predict(X[row]), _NB_MU_FLOOR)  # lgb poisson predict returns mu directly
    └─ per_stat_dists[Stat] = ParametricNegativeBinomial(
                                  mu=mu_hat, alpha=self._count_dispersions[Stat]
                              )
  else:
    [unchanged: 5-quantile predict, sort, clip, wrap in QuantileDistribution]
        ▼
For each row:
  per_stat_dists = {Stat.PASSING_TDS: NB(...), Stat.PASSING_YARDS: QD(...), ...}
        ▼
  score_distribution(per_stat_dists, ruleset, derive_row_seed(row))
    ├─ samples 200 draws from each per-stat distribution
    ├─ scores each sample under the ruleset
    └─ returns composite mean / p10 / p50 / p90 in fantasy points
        ▼
  ProjectionWeeklySchema row:
    ├─ family = "MIXED"
    ├─ params = pack_per_stat_params({Stat.X: NB(...), Stat.Y: QD(...), ...})
    ├─ mean, p10, p50, p90, model_id (prefix "lightgbm-nb:"), generated_at
        ▼
Validated DataFrame returned by predict_distribution()
```

### 4.3 Snapshot regeneration

`scripts/backtest.py --update-snapshot --model all` runs the four model classes through 4 held-out years × 4 positions = 16 cells per model. Total 1504 rows written to `tests/backtest/model_metrics.json`. The `--check` mode regresses subsequent runs against this baseline.

---

## 5. Error handling

### 5.1 `LightGBMNbModel.fit` failure modes

Inherits all parent failure modes (empty join, insufficient seasons, feature column mismatch, schema validation). New failure modes specific to count-stat training:

- **Poisson convergence failure on count stat.** `lgb.LGBMRegressor.fit` with `objective="poisson"` requires non-negative targets. If an upstream change introduces negative values (would violate `WeeklyStatsSchema` validation already, but defense-in-depth) the fit raises a LightGBM error. Bubble up; don't catch.
- **`best_iteration_ == 0` on a count-stat fit.** Same warning the parent emits for quantile fits (`RuntimeWarning` to stderr, sub-model still predicts at constant baseline). Reuse the parent's warning pattern; emit per-(position, stat) for diagnosability.
- **`nb_dispersion_from_residuals` returns clip endpoint.** The MLE-then-clip pattern can return either `_NB_DISPERSION_CLIP[0]` (over-dispersed; long tail) or `_NB_DISPERSION_CLIP[1]` (effectively Poisson). Both are valid; no warning.

### 5.2 `LightGBMNbModel.predict_distribution` failure modes

Inherits parent's failure modes. New count-stat-specific:

- **`mu_hat` underflow.** lgb's poisson `predict()` can return values arbitrarily close to 0 if the booster predicts a very small mean. Clip to `[_NB_MU_FLOOR, ∞)` to keep `ParametricNegativeBinomial` well-defined (the same clip Plan 3e applies in Ridge's NB path). No warning.
- **`mu_hat` non-finite.** lgb's poisson `predict()` should always return finite, non-negative values for valid inputs, but a corrupt booster artifact could return NaN/inf. The downstream `ParametricNegativeBinomial(mean=..., dispersion=...)` constructor will surface a violation; if it ever fires the artifact is corrupt and the model should be retrained.

### 5.3 Mixed-family per-row

`ProjectionWeeklySchema.family` set to `MIXED` for every Plan 5c row. The codec's `pack_per_stat_params` handles each per-stat distribution independently — the row-level family value is metadata for downstream consumers reading the parquet directly without going through `unpack_per_stat_params`. The harness's existing season-aggregation gate (`SAMPLED_SUMMARY` family only) skips these rows the same way it skips Plan 5 / Plan 5b QUANTILE rows.

### 5.4 Snapshot drift handling

The 368 new Model C-NB rows are *new* on first regeneration; no prior baseline to drift against. The tolerance classifier already handles new rows. After the first `--update-snapshot --model all` post-Plan-5c, subsequent `--check` runs gate against the established Model C-NB values.

### 5.5 Determinism

- LightGBM `random_state=42` (inherited from `LGBM_DEFAULTS` via `_FIXED_PARAMS`-style merge) makes every booster deterministic on the same data. `objective="poisson"` adds no new stochasticity.
- `nb_dispersion_from_residuals` uses `scipy.optimize.minimize_scalar` with a bounded search; deterministic.
- Per-row sampling seeds inside `score_distribution` are derived from `(gsis_id, season, week, ruleset_name)` — already deterministic.

Re-running `--update-snapshot --model all` reproduces the same metrics bit-for-bit (within the existing snapshot tolerances).

---

## 6. Testing

### 6.1 `nb_dispersion_from_residuals` (`tests/test_distributions/test_parametric.py`)

Add a focused test for the relocated public helper:

- Invariant: returns dispersion in `_NB_DISPERSION_CLIP`.
- Smoke: passes a (mu_hat, actual) pair drawn from a known NB-2 distribution; assert the fitted dispersion is within ~30% of the truth (MLE has finite-sample variance).
- Edge cases: empty array → returns clip endpoint; all-zero actuals → returns clip endpoint.
- The `BaselineModel.fit` regression tests in `tests/test_models/test_baseline*.py` continue to pass without modification — the function move is behavior-preserving.

### 6.2 `LightGBMNbModel` (`tests/test_models/test_lightgbm_nb*.py`)

Mirrors `tests/test_models/test_lightgbm_tuned*.py` shape:

- **`test_lightgbm_nb.py`** (cross-cutting):
  - `fit` populates both `self._count_models` (one booster per count stat) and `self._sub_models` (5 boosters per yards stat).
  - `fit` populates `self._count_dispersions` for every count stat with a value in `_NB_DISPERSION_CLIP`.
  - `predict_distribution` output validates against `ProjectionWeeklySchema`; `family` column is `"MIXED"` for every row.
  - `params` blob round-trips through codec; reconstructed per-stat distributions match the originals (count stats → `ParametricNegativeBinomial`; yards stats → `QuantileDistribution`).
  - `model_id` prefix is `"lightgbm-nb:"`.
  - `code_hash` differs from `LightGBMTunedModel`'s code_hash (different file set hashed).
  - `save` / `load` round-trip preserves `model_id`, `_count_models`, `_count_dispersions`, `_sub_models`, and predictions.
  - **Synthetic stat-mix fit/predict**: with the existing synthetic WR fixture (counts at low mean ~0.4 for receiving_tds, yards at higher mean), assert (a) count-stat per-row mean predictions are positive and finite; (b) yards-stat predictions match what `LightGBMTunedModel` would produce on the same fixture (no behavior change on yards).

- **Per-position files** (`test_lightgbm_nb_qb.py`, `_rb.py`, `_te.py`, `_wr.py`):
  - Smoke fit on synthetic per-position fixtures (reuse Plan 3a/3b fixtures); assert per-stat sub-models (count + yards) exist for that position's `target_stats`; assert prediction output validates.

- **Smoke parametrized across positions** (`tests/test_models/test_lightgbm_nb_smoke.py`):
  - Single fit + predict for each of 4 positions on synthetic data; ~30s total.

### 6.3 Backtest harness (`tests/test_backtest/`)

Existing tests pass unchanged. Extend:

- `tests/test_backtest/test_harness_quad_model.py` (new — or rename `test_harness_triple_model.py` to `_quad_model.py`): asserts `run_backtest(model_classes=("baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb"))` produces metrics for all 4 model classes. Gated by `@pytest.mark.backtest`.

### 6.4 Default-on smoke (`tests/backtest/test_backtest_smoke.py`)

Extend to:
- Pass `model_classes=("baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb")` to `run_backtest`.
- Assert unique model_class set is `{baseline, lightgbm, lightgbm-tuned, lightgbm-nb}`.
- Iterate the per-model core-metrics check across all 4 classes.
- Pin the SAMPLED_SUMMARY-vs-non-SAMPLED_SUMMARY asymmetry: assert `lightgbm-nb` does not emit `season_calibration_*` rows (same expectation as `lightgbm` and `lightgbm-tuned`).

Runtime budget: rises from ~50s (Plan 5b smoke) to ~65–80s (one extra model fit on the (WR, 2024) cell). Within the <2min CI budget.

### 6.5 Backtest snapshot gate (`pytest -m backtest --run-backtest`)

Full opt-in backtest produces 1504 metric rows (400 baseline + 368 lightgbm + 368 lightgbm-tuned + 368 lightgbm-nb). Wall time ≈ A 5min + C 30-50min + C-tuned 30-50min + C-NB ~30-50min → 95–155 minutes for `--model all`. Acceptable for an opt-in pre-PR gate.

### 6.6 Type / lint conformance

- `mypy src tests` clean (strict mode).
- `ruff check src tests` and `ruff format --check src tests` clean.
- No new dependencies (`scipy.stats.nbinom`, `scipy.optimize.minimize_scalar` already used by Ridge; `lgb.LGBMRegressor(objective="poisson")` already supported by installed LightGBM).

---

## 7. Phasing

Each phase ≤5 files per the CLAUDE.md "PHASED EXECUTION" rule.

### Phase 0 — Public NB dispersion helper

Files (3):
1. `src/projections/distributions/parametric.py` — add public `nb_dispersion_from_residuals` with the body of `_negative_binomial_dispersion_from_residuals`; move `_NB_MU_FLOOR` + `_NB_DISPERSION_CLIP` constants too.
2. `src/projections/models/baseline.py` — update 3 call sites to import from `distributions.parametric`; remove the relocated function and constants. Behavior-preserving move.
3. `tests/test_distributions/test_parametric.py` — add focused tests for the new public helper.

**Exit criterion:** all existing tests pass (Ridge regression unchanged); the new helper tests pass; `mypy src tests`, `ruff check src tests`, `ruff format --check src tests` clean.

### Phase 1 — `LightGBMNbModel` + `DistributionFamily.MIXED` + factories + dispatch

Files (4):
1. `src/projections/schemas.py` — add `DistributionFamily.MIXED`.
2. `src/projections/models/lightgbm_nb.py` (new) — class + four factories.
3. `src/projections/models/__init__.py` — extend imports, `__all__`, and `_<POS>_FACTORIES`.
4. `tests/test_models/test_lightgbm_nb.py` (new) — cross-cutting tests.

**Exit criterion:** `pytest -v tests/test_models/test_lightgbm_nb.py` passes (cross-cutting test exercises real fit + predict on synthetic WR data); mypy + ruff clean.

### Phase 2 — Backtest harness wiring + cross-position smoke

Files (4):
1. `scripts/backtest.py` — extend `--model` choices and `args.model == "all"` branch.
2. `tests/backtest/test_backtest_smoke.py` — extend to all 4 model classes + tuned-vs-NB asymmetry pin.
3. `tests/test_backtest/test_harness_quad_model.py` (new) — quad-model harness end-to-end. Mirrors the existing `test_harness_triple_model.py` shape with a fourth `model_class`. (Could alternatively rename triple to quad; the new file is preferred to keep the Plan 5b artifact intact and the Plan 5c PR diff narrow.)
4. `tests/test_models/test_lightgbm_nb_smoke.py` (new) — parametrized cross-position smoke covering all 4 positions.

Per-position smoke files (`test_lightgbm_nb_{qb,rb,te,wr}.py`) are **not** added; the parametrized cross-position smoke is sufficient. This matches Plan 5b's final shape (the per-position files for `LightGBMTunedModel` were collapsed into `test_lightgbm_tuned_smoke.py` during implementation).

**Exit criterion:** `pytest -v` clean; default-on smoke covers all 4 model classes; harness opt-in test passes with `--run-backtest`.

### Phase 3 — Regenerate snapshot

Files (1):
1. `tests/backtest/model_metrics.json` — overwritten by `scripts/backtest.py --update-snapshot --model all`.

**Operations:**
1. Run `python scripts/backtest.py --update-snapshot --model all`. Wall time ~95–155min.
2. Inspect the snapshot diff: 1136 → 1504 rows; 368 new under `model_class="lightgbm-nb"`.
3. Re-run `python scripts/backtest.py --check --model all` to confirm zero drift on the same data.
4. Commit the snapshot.

### Phase 4 — Diagnostic report

Files (2):
1. `project_management.md` — append per-cell A vs C vs C-tuned vs C-NB comparison; record §1.3 verdict.
2. `TODO.md` — close the Plan 5c follow-up; document the verdict.

**Operations:**
1. Generate the per-cell A vs C vs C-tuned vs C-NB table (same shape as Plan 5b's §1.3 comparison; 16 rows × 4 metrics × 4 models).
2. Compute the §1.3 adoption-gate verdict for C-NB vs A.
3. Per-stat attribution for any remaining composite-RMSE gap (mirrors the diagnostic done at the end of Plan 5b — confirms whether the count-stat fix landed and identifies any remaining contributors).
4. Append "Plan 5c — Hybrid LightGBM with NB-2 for Count Stats — shipped (run YYYY-MM-DD)" section to `project_management.md`.
5. If the gate passes: file a follow-up housekeeping commit for production-default switch + Model C pruning. If the gate fails: document deltas; the next-step decision (Plan 6 ensemble vs. TODO #3 / TODO #23) goes back to the user.

---

## 8. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-28 | Side-by-side coexistence (4 model classes total) | Plan 5b's diagnostic was sharper for having A vs C vs C-tuned committed in one place; Plan 5c benefits from the same pattern. Cost is 368 snapshot rows + one factory key. Pruning Model C (untuned) is housekeeping for after Plan 5c lands. |
| 2026-04-28 | Reuse Plan 5b's tuned hyperparameters for poisson sub-models | The diagnostic finding is that the bias is in the prediction *output format* (5-knot interp → biased empirical mean), not in fit quality. Replacing the output format addresses the bias directly; hyperparameter quality is secondary. Re-tuning is the cheap escalation if the gate fails marginally. |
| 2026-04-28 | NB-2 (not Poisson) at predict time for count stats | Plan 3e identified that pure Poisson under-disperses real NFL count stats (the 0/1/2 weekly TD distribution has variance > mean — over-dispersed). NB-2 with `dispersion` parameter handles this cleanly. The Poisson loss at fit time is fine because the booster's `mu_hat` is unbiased even when the data is over-dispersed; dispersion is fit separately on residuals. |
| 2026-04-28 | Yards/receptions stats unchanged from Model C-tuned | Diagnostic showed yards stats are competitive on RMSE under quantile regression. Switching them would muddy the comparison and add risk for no expected gain. |
| 2026-04-28 | New `DistributionFamily.MIXED` row-level value (not reuse of SAMPLED_SUMMARY) | SAMPLED_SUMMARY semantically means "every per-stat dist has summary stats packed in mean/p10/p50/p90"; that's true here but the per-stat *families* differ. MIXED makes the row-level metadata accurate without overloading SAMPLED_SUMMARY. The codec doesn't care (per-stat families are independent inside the params blob). |
| 2026-04-28 | Move `_negative_binomial_dispersion_from_residuals` from baseline.py to parametric.py as public `nb_dispersion_from_residuals` | The function is mathematical (NB-2 dispersion MLE) — not specific to Ridge. Two callers now (Ridge and Plan 5c LightGBM); the right home is next to `ParametricNegativeBinomial`. |
| 2026-04-28 | Subclass `LightGBMTunedModel` (not `LightGBMModel`) | Inherits the tuned-params loader, the JSON path resolution, and the `_hyperparams_for(stat)` hook that yields tuned values. Plan 5c is a strict extension of Plan 5b's surface. |
| 2026-04-28 | Branch from `feat/plan-5b-tuning` (not main) | Plan 5c structurally depends on Plan 5b's `LightGBMTunedModel`. When Plan 5b merges to main, Plan 5c rebases or merges main forward. |
| 2026-04-28 | No re-tuning, no per-fold tuning, no yards-stat changes | All three are deferred per the diagnostic finding that the count-stat bias is the dominant lever. Each is a candidate for its own follow-up plan if Plan 5c's gate is marginal. |
| 2026-04-28 | Adoption gate same as Plan 5 / 5b §1.3 | Consistency across the modeling track makes per-plan verdicts directly comparable. Production-default switch is post-adoption housekeeping, separate from the gate decision. |

---

## 9. Out-of-scope / explicit future work

- **Re-tuning hyperparameters for the poisson objective.** Plan 5d candidate if the gate fails marginally.
- **Pruning Model C (untuned) from POSITION_DISPATCH.** Housekeeping commit after Plan 5c lands; Model C is now strictly dominated by Model C-tuned (Plan 5b verdict) and would be further dominated by Model C-NB (expected Plan 5c verdict).
- **Production-default switch from Model A to Model C-NB.** Separate small change after the gate passes; likely a one-line config update plus possibly a CLI default flag.
- **TODO #28 — widen `aggregate_to_season` to QUANTILE / MIXED families.** Independent of Plan 5c. Would add 32 + 32 + 32 = 96 snapshot rows for the three non-baseline model classes once landed.
- **Plan 6 — Model D ensemble (A + best of {C, C-tuned, C-NB}).** Cheap given the infrastructure that will exist post-Plan-5c; pursue if Plan 5c marginally passes or marginally fails.
- **Multi-output LightGBM training.** Would address the per-stat-sub-model "no shared prior" mechanism that Plan 5 / 5b's quantile training inherited. Not yet specced.
- **Other count-stat objectives (Tweedie, regression with log link).** `objective="poisson"` is the canonical first choice; alternatives are deferred unless poisson proves inadequate.
