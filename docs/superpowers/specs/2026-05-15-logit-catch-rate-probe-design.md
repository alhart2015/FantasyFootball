# Logit catch_rate Sub-Model Probe — Design

**Status:** draft (brainstorming, 2026-05-15). Ready for user review.
**Date:** 2026-05-15
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Builds on:**
- WR Ensemble — Decomposed-Baseline Child A Swap (PR #38, merged 2026-05-16). Shipped `wr_ensemble_decomposed` as WR production default. Binding cell ADOPT at RMSE Δ -0.0038 fpts (marginal-zone magnitude per PR #31's retrospective). PR #38 §1.3.5 named factor-appropriate sub-models for `catch_rate` as the recommended next slot to lift the small magnitude.
- WR Target Decomposition Integration (PR #36, merged 2026-05-14). Shipped `DecomposedBaselineModel` with `catch_rate` decomposed via `RidgeCV` on the ratio + clipped-Normal predict-time sampling. PR #36 §1.2 deferred "factor-appropriate sub-model classes" to this probe + integration cycle.
- WR Target Decomposition Probe (PR #32, merged 2026-05-10). Established the probe pattern (`src/projections/backtest/target_decomposition_probe.py` + `scripts/probe_target_decomposition.py`). PR #32 §1.4 #3 explicitly named factor-appropriate sub-models as a deferred follow-up to a SIGNAL integration.

**Branch:** `feat/probe-logit-catch-rate` cut from `main` at `4317c66` (PR #38 merge commit).

---

## 1. Goals & success criteria

### 1.1 Goal

Probe whether replacing the **catch_rate efficiency sub-model class** with a binomial-logit model lowers per-stat **receptions** RMSE on out-of-sample WR rows. Two arms are compared, both built on the same shared-volume `RidgeCV` for `targets`:

| Arm | catch_rate sub-model | receptions prediction |
|---|---|---|
| **Incumbent (Ridge-decomp)** | `RidgeCV` on `receptions/targets` ratio | `mu_targets × clip(mu_ratio, 0, 1)` |
| **Candidate (Logit-decomp)** | `LogisticRegressionCV` via Bernoulli-trial row expansion (`L2` penalty, 5-point `Cs` grid) | `mu_targets × sigmoid(X β_logit)` |

The two arms differ in **exactly one design choice**: the catch_rate efficiency sub-model class. Both arms use the same shared-volume RidgeCV on `targets`, the same train rows, the same train-time mask (`targets > 0` for efficiency fits), the same eval splits. Any per-stat RMSE Δ on receptions is attributable solely to the sub-model class change.

This probes the same architectural-prior axis PR #32 named in §1.4 #3: **does swapping `catch_rate` from a Ridge-on-ratio + clipped-Normal predict (current production via PR #36 / #38) to a binomial-logit fit (proper for ratio-of-counts response with known trial counts) improve mean prediction accuracy?**

### 1.2 Architectural prior

The current production `catch_rate` model has two structural problems that a binomial-logit addresses:

1. **Ridge-on-ratio mis-models a [0, 1]-bounded response.** Ridge minimizes MSE on the unbounded real line; predicted ratios can be negative or > 1, then get hard-clipped at predict time. The asymmetric clip cuts off tails unevenly and biases the mean for rows with extreme features.
2. **Population-level Normal residual std mis-calibrates per-row variance.** True catch_rate variance is `p × (1 - p) / N` (binomial) where `N = targets` — heteroscedastic in both `p` and `N`. The constant Normal σ is too narrow at moderate `p` (when `p × (1-p)` is large) and too wide at extreme `p`. This shows up in the integration's p10/p90 calibration, not in the probe's mean-RMSE.

The probe targets problem 1 (the **mean** consequence of the bounded-response misfit). Problem 2 is integration territory — variance/distribution calibration is not measured by per-stat mean RMSE.

Expected magnitude (mechanism prior): plausibly small (probably 0.001–0.005 fpts on the receptions stat). Ridge-on-clipped-ratio is already a reasonable approximation for most rows; the logit fix is most consequential for rows where `mu_ratio_ridge` lands near 0 or near 1 before clipping. Those rows are a minority of the WR row population. A SIGNAL verdict at any magnitude greenlights the integration cycle; a marginal-zone magnitude (|Δ| < 0.005 receptions) gets the PR #31 retrospective flag in the report. NULL is plausible — the probe is genuinely uncertain.

### 1.3 Success criteria

The probe **ships** when all four pass. The verdict (SIGNAL / NULL / REGRESSION) is informational, not a ship gate.

1. **Coverage:** WR rows with `targets > 0` per eval year ≥ 0.95 across all four eval years 2021–2024. Same default `--coverage-threshold 0.95` as PR #32. PR #32's run measured 0.981+ per year; no relaxation expected. Relaxation triggers the PR #31 retrospective rule (marginal-zone magnitudes under coverage relaxation are MARGINAL not SIGNAL).
2. **Probe completeness:** Walk-forward over eval years `{2021, 2022, 2023, 2024}`; per-row residuals populated for both arms on every eval year; pooled paired-bootstrap CI on receptions Δ-RMSE rendered. One per-stat report (`reports/feature_probe_logit_catch_rate_receptions.md`) plus one summary report (`reports/feature_probe_logit_catch_rate_summary.md`).
3. **Verdict rule on receptions Δ-RMSE (candidate − incumbent, i.e., logit − ridge):**
   - **SIGNAL** iff `rmse_delta.hi_95 < 0` (logit CI strictly negative)
   - **REGRESSION** iff `rmse_delta.lo_95 > 0` (logit CI strictly positive)
   - **NULL** otherwise (CI brackets zero)
   - No effect-size floor at the per-stat level (matches PR #32). PR #31 retrospective flag applies at integration go/no-go time, not at probe verdict time.
4. **Verification gates green:** `mypy src tests scripts` strict + `ruff check` + `ruff format --check` clean. Relevant pytest subset green (probe module + walk-forward harness tests + CLI tests).

### 1.4 Out of scope (deferred follow-ups)

1. **Composite-fpts measurement.** Per-stat RMSE only. Composite-fpts is the integration's adoption-gate concern (`(EnsembleModel-with-logit-catch-rate, WR) vs (EnsembleModel-with-ridge-catch-rate, WR)` if SIGNAL).
2. **Variance / distribution comparison at predict time.** Binomial(targets, p) vs clipped-Normal(mu, σ). Important for the integration's p10/p90 calibration story but not gated by this probe. The integration plan can include a side-by-side calibration table.
3. **`yards_per_target` factor-appropriate sub-model** (log-link Gamma). Separate probe cycle; same harness shape but a different efficiency factor and different sub-model class. Gated on this probe's outcome — if catch_rate SIGNAL ships an integration, yards_per_target probe naturally follows.
4. **`td_rate_per_target` factor-appropriate sub-model** (logistic / Poisson-NB). Same as #3 — separate cycle.
5. **Integration build-out.** No changes to `src/projections/models/decomposed_baseline.py` in this probe. If SIGNAL: the integration plan layers a swap inside `wr_decomposed_baseline` (replacing the catch_rate efficiency Ridge with the logit model) + variance-handling at predict time (`np.random.binomial(targets, p)` instead of clipped-Normal). New adoption gate vs the just-shipped `ensemble-decomposed` production.
6. **Other positions.** RB / TE have receiving factors that could decompose similarly; QB has a different (passing-chain) decomposition entirely. Each is its own probe + integration cycle. WR-only here.
7. **Other stats on WR.** `rushing_yards` / `rushing_tds` / `fumbles_lost` are out of scope as in PR #32 §1.4 #5.
8. **Beta regression or beta-binomial as alternatives.** Binomial-logit is the simplest factor-appropriate model for a counts-out-of-trials response; beta and beta-binomial allow overdispersion / fractional responses but add scope. Defer to a separate probe if binomial NULLs but the mechanism prior holds.

---

## 2. Source data — already ingested

All inputs already exist in the feature cache and weekly stats:

| Column | Type | Source | Use |
|---|---|---|---|
| WR features | `WrFeaturesSchema` | `data/features/wr/season=YYYY/week=WW/part.parquet` | X for both volume RidgeCV and efficiency sub-models |
| `targets` | `Series[int]` | `WeeklyStatsSchema.targets` | y for shared volume RidgeCV; trial count N for Bernoulli expansion; filter (`> 0`) for efficiency fits |
| `receptions` | `Series[int]` | `WeeklyStatsSchema.receptions` | numerator for `catch_rate = receptions / targets`; success count for Bernoulli expansion |

**No new ingest, no schema changes, no override parquet.** The probe consumes the existing per-week WR feature parquet and joins to `weekly_stats` on `(gsis_id, season, week)` exactly as `BaselineModel.fit` does (and as `target_decomposition_probe.py` does).

---

## 3. Architecture

### 3.1 New module — `src/projections/backtest/logit_catch_rate_probe.py`

Mirrors `src/projections/backtest/target_decomposition_probe.py`'s shape. Pure numpy / pandas / sklearn. Reuses `paired_bootstrap_rmse_delta` and `BootstrapDelta` from `src/projections/backtest/adoption_gate.py` unchanged.

```
logit_catch_rate_probe.py
├── _fit_shared_volume(X, y_targets, alphas) -> RidgeCV
│       # shared volume sub-model on targets ~ X; same recipe as target_decomposition_probe._fit_decomposed_volume
├── _fit_ridge_efficiency(X_pos, ratio, alphas) -> RidgeCV
│       # Arm A — identical to current production decomposed_baseline.py catch_rate fit
├── _expand_to_trials(X_pos, successes, trials) -> tuple[np.ndarray, np.ndarray]
│       # Bernoulli-trial expansion. For each row with T trials and S successes,
│       # emit T copies of the X row, S with y=1 and (T-S) with y=0.
│       # Returns (X_trials, y_trials). Pure numpy via np.repeat + np.concatenate.
├── _fit_logit_efficiency(X_trials, y_trials, Cs) -> LogisticRegressionCV
│       # Arm B — sklearn.linear_model.LogisticRegressionCV
│       # penalty="l2", Cs=5-point log grid, cv=5, scoring="neg_log_loss".
│       # Trains on row-expanded Bernoulli trials (mathematically equivalent
│       # to binomial-logit GLM via MLE).
├── _predict_receptions_ridge(mu_targets, X_eval, ridge_eff) -> np.ndarray
│       # mu_targets × clip(ridge_eff.predict(X_eval), 0, 1)
├── _predict_receptions_logit(mu_targets, X_eval, logit_eff) -> np.ndarray
│       # mu_targets × logit_eff.predict_proba(X_eval)[:, 1]
├── walk_forward_residuals(features, weekly_stats, eval_years) -> ProbeResults
│       # For each year in eval_years:
│       #   - train on seasons [start..year-1]
│       #   - fit shared volume Ridge on all WR train rows
│       #   - mask train rows on targets > 0
│       #   - fit ridge_eff on (X_pos, catch_rate_ratio)
│       #   - fit logit_eff on _expand_to_trials(X_pos, receptions_pos, targets_pos)
│       #   - eval on year:
│       #       mu_targets = volume.predict(X_eval)
│       #       pred_ridge[year] = _predict_receptions_ridge(mu_targets, X_eval, ridge_eff)
│       #       pred_logit[year] = _predict_receptions_logit(mu_targets, X_eval, logit_eff)
│       #       actual[year] = receptions
│       # Concatenate per-year arrays into pooled (actual, pred_ridge, pred_logit, year_id)
│       # buffers. Returns ProbeResults with these buffers + per-year coverage stats.
└── compute_verdict(results: ProbeResults, n_bootstrap=1000, seed=42) -> PerStatVerdict
        # Pooled paired-bootstrap CI on (residuals_logit - residuals_ridge) where
        # residuals = (actual - prediction). Returns BootstrapDelta + verdict label.
```

**`PerStatVerdict` and `ProbeResults` shapes:** match `feature_probe.py` Phase 1's PerStatVerdict dataclass (`src/projections/backtest/feature_probe.py:57`). Add a `RidgeAlphaGrid` / `LogitCsGrid` constants block at the top of the module — mirrors `target_decomposition_probe.py`'s grid constants.

**Row-expansion bookkeeping.** For ~8,400 WR rows with avg ~7 targets/row, expanded count ≈ 50K-60K trial rows. Numpy can build the expansion in one pass: `X_trials = np.repeat(X_pos, trials, axis=0)` and `y_trials = np.concatenate([np.ones(S_i), np.zeros(T_i - S_i)] for each row)`. Memory: ~60K × ~20 features × 8 bytes ≈ 10 MB. Trivial.

### 3.2 New CLI script — `scripts/probe_logit_catch_rate.py`

Mirrors `scripts/probe_target_decomposition.py`. argparse for `--eval-years`, `--coverage-threshold`, `--seed`, `--n-bootstrap`. Runs `walk_forward_residuals` then `compute_verdict`. Writes:

- `reports/feature_probe_logit_catch_rate_summary.md` — verdict + 95% CI + n_paired + per-year coverage table + magnitude flag note if |Δ| < 0.005.
- `reports/feature_probe_logit_catch_rate.csv` — per-year and pooled rmse/spearman deltas (matches PR #32's CSV shape).

### 3.3 No edits to existing code

- `src/projections/models/decomposed_baseline.py` is **not touched** by this probe. The integration plan handles any production swap.
- `src/projections/backtest/target_decomposition_probe.py` is **not touched**. The new probe is its own module; reuse is via `from projections.backtest.adoption_gate import paired_bootstrap_rmse_delta, BootstrapDelta`.
- `src/projections/backtest/feature_probe.py` is **not touched**. PerStatVerdict can be either imported or re-defined locally — plan picks; either works.

### 3.4 No schema, no codec, no factory changes

The probe operates in-memory against sklearn estimators fit on numpy arrays. No new Distribution classes, no codec edits, no new `_WR_FACTORIES` registration. The probe is strictly a **mechanism test**.

---

## 4. Testing

`tests/test_backtest/test_logit_catch_rate_probe.py`:

1. **`_expand_to_trials` correctness** — synthetic input: 3 rows with (T=4, S=3), (T=2, S=0), (T=5, S=5). Expected expanded shape: (11, n_features); expected y_trials: [1,1,1,0, 0,0, 1,1,1,1,1]. Verifies the expansion preserves binomial likelihood (sum-by-original-row should match input).
2. **`_expand_to_trials` zero-trials guard** — a row with `T=0` should be excluded from expansion (matches the upstream `targets > 0` mask) and not raise.
3. **`_fit_ridge_efficiency` matches production** — fit on a small synthetic ratio frame; verify the returned RidgeCV's `predict` matches `decomposed_baseline._fit` (same RidgeCV alpha grid, same data). This is a pin against accidental drift in the incumbent arm.
4. **`_fit_logit_efficiency` mean prediction sanity** — fit on a small synthetic frame with a known true coefficient (e.g., true catch_rate = sigmoid(β₀ + β₁ x₁), generate trials via np.random.binomial). Verify `predict_proba(X)[:, 1]` recovers the true probabilities within tolerance (say MAE < 0.05 on a 200-row fixture). Tests that the row-expansion + LogisticRegressionCV path actually fits a binomial-logit.
5. **`_predict_receptions_ridge` vs `_predict_receptions_logit` divergence test** — on the synthetic fixture, both produce predictions; verify they differ on rows with extreme features (where the Ridge predicts ratio outside [0, 1] before clip). Spec property: at extreme features the logit prediction will land closer to the binomial mean than the ridge-clipped prediction.
6. **`walk_forward_residuals` integration** — synthetic 4-season frame; verify both pred buffers are populated, length matches actual, coverage stat populated.
7. **`compute_verdict` verdict mapping** — three crafted ProbeResults inputs (synthetic residuals) hitting each of SIGNAL / NULL / REGRESSION. Pins the verdict-mapping logic.
8. **CLI smoke** — `pytest tests/test_scripts/test_probe_logit_catch_rate_cli.py` (new). Mocks `walk_forward_residuals`, verifies argparse + report writing.

Real-data smoke + the actual probe verdict captured in the PR per CLAUDE.md "Forced verification" rule.

---

## 5. Risk register

1. **Magnitude prior is small.** Per §1.2, expected |Δ| is plausibly < 0.005 receptions — marginal-zone per PR #31. If SIGNAL is small, the integration's adoption-gate verdict may also land marginal. **Mitigation:** the probe ships either way; the integration gate weights magnitude separately. No mitigation in the probe itself beyond clear reporting.
2. **NULL on receptions.** Plausible: Ridge-on-clipped-ratio may already approximate the binomial mean within Monte-Carlo noise on this dataset. **Mitigation:** NULL closes the catch_rate-only factor-appropriate direction at the WR cell. Yards-per-target and td-rate-per-target probes (separate cycles) remain open under TODO #33b. NULL doesn't close the broader factor-appropriate-sub-model direction.
3. **Row-expansion + LogisticRegressionCV runtime.** 50K-60K samples × CV=5 folds × Cs=5 grid points = 1,500K fits per cross-validation. Each fit is fast on a small feature matrix (~20 cols), so total ~5-30s per walk-forward year. With 4 years, total probe runtime ~1-3 min. **Mitigation:** none needed.
4. **Sub-model regularization scale mismatch.** Ridge uses alpha (penalty strength); LogisticRegressionCV uses C = 1/alpha (inverse penalty). The two grids span different optimization landscapes. **Mitigation:** verify Cs grid covers an analogous range of effective regularization. The plan task picks `Cs=[0.01, 0.1, 1.0, 10.0, 100.0]` to span ~3 orders of magnitude, matching Ridge alpha grid's effective range.
5. **Sklearn LogisticRegressionCV solver choice.** Default is `lbfgs`; for binary with L2 it works fine on this data size. **Mitigation:** plan task uses default solver; if convergence warnings fire, switch to `liblinear` or `saga` per sklearn docs.
6. **Feature scaling.** Ridge is scale-invariant up to penalty (it implicitly handles via alpha); logistic-regression's L2 penalty is scale-dependent. WR features are NOT pre-scaled in `BaselineModel.fit`. **Mitigation:** add a `StandardScaler` upstream of the logistic fit (probe-internal; doesn't affect Ridge arm). Document the difference in the report.

---

## 6. Reports

`reports/feature_probe_logit_catch_rate_summary.md`:

- Verdict (SIGNAL / NULL / REGRESSION) + 95% CI on receptions Δ-RMSE.
- Per-year breakdown (informational): Δ-RMSE point + CI per eval year.
- Coverage stats: `targets > 0` rate per year (eval), per train window (train).
- Magnitude flag (informational): is |Δ| < 0.005 receptions?
- Recommended next direction per verdict: SIGNAL → integration plan; NULL → close cell, name `yards_per_target` probe as next slot; REGRESSION → close cell strongly, no follow-up at this factor.

`reports/feature_probe_logit_catch_rate.csv` — long-form per-year deltas matching PR #32's CSV shape.

---

## 7. Estimated scope

~5 plan tasks. Single session. Real-data probe run is fast (~1-3 min per year × 4 years = small).

| Task | Surface | Files touched |
|---|---|---|
| 1. `_expand_to_trials` + unit tests | probe core | `src/projections/backtest/logit_catch_rate_probe.py` (new), `tests/test_backtest/test_logit_catch_rate_probe.py` (new) |
| 2. `_fit_ridge_efficiency` + `_fit_logit_efficiency` + `_predict_*` + unit tests | probe core | extend the module from Task 1, extend the test from Task 1 |
| 3. `walk_forward_residuals` + `compute_verdict` + integration tests | probe core | extend module + tests |
| 4. CLI script + CLI smoke | scripts + tests | `scripts/probe_logit_catch_rate.py` (new), `tests/test_scripts/test_probe_logit_catch_rate_cli.py` (new) |
| 5. Real-data probe run + report + PM/TODO updates | reports + PM | `reports/feature_probe_logit_catch_rate_summary.md`, `reports/feature_probe_logit_catch_rate.csv`, `project_management.md`, `TODO.md` |

End-to-end: 1 focused session.

---

## 8. Implementation plan handoff

After spec approval and commit on `feat/probe-logit-catch-rate`, the next step is the writing-plans skill to produce `docs/superpowers/plans/2026-05-15-logit-catch-rate-probe.md` decomposing the 5 tasks above into per-task implementation steps with per-task verification commands.
