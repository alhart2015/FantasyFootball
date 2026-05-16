# Tweedie yards_per_target Sub-Model Probe — Design

**Status:** draft (brainstorming, 2026-05-16). Ready for user review.
**Date:** 2026-05-16
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Builds on:**
- Logit catch_rate Sub-Model Probe (PR #39, merged 2026-05-16). Verdict NULL on receptions (RMSE Delta -0.0018, CI [-0.0047, +0.0009]). PR #39 §1.4 #3 named `yards_per_target` factor-appropriate probe (log-link Gamma / Tweedie family) as the next slot. PR #39's recommended-next-direction quotes "higher prior — `yards_per_target` carries more receiving-yards variance than `catch_rate` carries receptions variance, AND a Gaussian-on-ratio Ridge is a much worse approximation to a Gamma response than to a Bernoulli mean."
- WR Ensemble — Decomposed-Baseline Child A Swap (PR #38, merged 2026-05-16). Production WR routing now `ensemble-decomposed`, which decomposes `Stat.RECEPTIONS` only. `receiving_yards` and `receiving_tds` still predicted via direct RidgeCV. So Ridge-decomp on `yards_per_target` is NOT in production; this probe's incumbent arm is a probe-internal construction.
- WR Target Decomposition Integration (PR #36, merged 2026-05-14). Shipped `DecomposedBaselineModel` infrastructure including the unbounded-efficiency code path (`efficiency_clip_hi=float("inf")`) used by `yards_per_target` decomp.
- WR Target Decomposition Probe (PR #32, merged 2026-05-10). Measured Ridge-decomp on `yards_per_target` vs direct RidgeCV on `receiving_yards` as NULL (per-stat RMSE Delta -0.0054 yards, CI [-0.0601, +0.0492]). The CI bracketed zero with width ~0.11 yards — wider than catch_rate's by an order of magnitude.

**Branch:** `feat/probe-tweedie-yards-per-target` cut from `origin/main` at `efa2588` (PR #39 merge commit).

---

## 1. Goals & success criteria

### 1.1 Goal

Probe whether replacing the **yards_per_target efficiency sub-model class** with a Tweedie GLM (log link, power=1.5) lowers per-stat **receiving_yards** RMSE on out-of-sample WR rows. Two arms are compared, both built on the same shared-volume `RidgeCV` for `targets`:

| Arm | yards_per_target sub-model | receiving_yards prediction |
|---|---|---|
| **Incumbent (Ridge-decomp)** | `RidgeCV` on `receiving_yards / targets` ratio | `mu_targets * clip(mu_ratio, 0, +inf)` |
| **Candidate (Tweedie-decomp)** | `TweedieRegressor(power=1.5, link="log")` with alpha CV-selected via `GridSearchCV` | `mu_targets * tweedie_eff.predict(X_eval)` |

The two arms differ in **exactly one design choice**: the yards_per_target efficiency sub-model class. Both arms use the same shared-volume RidgeCV on `targets`, the same train rows, the same train-time mask (`targets > 0` for efficiency fits), the same eval splits. Any per-stat RMSE Delta on receiving_yards is attributable solely to the sub-model class change.

This is the factor-appropriate-sub-model probe one slot down PR #39's chain. The architectural axis is the same as catch_rate's (does the factor-appropriate GLM beat Ridge-on-ratio?), but on a different response shape: yards_per_target has continuous positive support with a point mass at zero (incomplete passes, zero-yard catches), so the factor-appropriate model class is Tweedie p=1.5 (compound Poisson-Gamma) rather than binomial-logit.

### 1.2 Architectural prior

Two structural problems with Ridge-on-ratio that Tweedie p=1.5 with log link addresses:

1. **Gaussian-Ridge mis-models a right-skewed non-negative response.** `yards_per_target` has a point mass near 0 (incomplete passes, 0-yard catches) and a long right tail (long completions). Ridge minimizes Gaussian MSE on the unbounded real line; predicted ratios can be negative and get hard-clipped at 0 at predict time. The clip-asymmetry biases the mean for rows with extreme features.
2. **Tweedie p=1.5 is the principled choice for compound Poisson-Gamma responses.** A response that mixes a point mass at zero with continuous positive support fits the Tweedie family with `1 < p < 2`. The log link guarantees positivity of predictions; variance scales as `mu^p`, matching how `yards_per_target` variance grows with mean (rookies with stable low-volume receiving have tight efficiency distributions; high-volume veterans have wide ones).

PR #39 made the same case for binomial-logit on catch_rate. PR #39's verdict was NULL (RMSE Delta -0.0018 receptions, CI brackets zero), suggesting Ridge-on-clipped-ratio approximates the binomial mean well enough on this dataset.

**Why this probe is not dead on arrival despite PR #39's NULL.** PR #39's argument for *higher* prior on yards_per_target rests on two claims:
- `yards_per_target` carries more receiving-yards variance than `catch_rate` carries receptions variance. Composite-fpts contribution: receiving_yards at 0.1 fpt/yard scales any per-stat lift by 0.1; receptions at 1.0 fpt/rec scales by 1.0. A 0.05-yard per-stat Delta on receiving_yards translates to 0.005 fpts composite — same composite scale as 0.005 receptions on catch_rate, which is what PR #39 was hunting for. So composite-fpts magnitude requires a *bigger* per-stat lift on yards_per_target; the prior must clear a higher hurdle.
- *Gaussian-on-ratio Ridge is a worse approximation to a Gamma response than to a Bernoulli mean.* This is the substantive prior. For Bernoulli, the Ridge-clipped-mean and the logistic-mean agree to first order around p=0.5 (where catch_rate clusters); they only diverge at the tails. For Gamma, the Ridge-mean diverges from the log-link-mean *uniformly* — the log link is the canonical link for the multiplicative variance structure, and Ridge has no analog. The yards_per_target distribution has more mass in the divergent regime than catch_rate has in the binomial-vs-Ridge divergent regime.

The probe tests this prior. NULL closes the factor-appropriate-yards_per_target slot but doesn't close the broader direction (td_rate_per_target remains).

**Expected magnitude.** Plausibly 0.005-0.05 fpts composite, i.e., 0.05-0.5 yards per-stat. PR #32's Ridge-decomp-vs-direct measurement on receiving_yards (RMSE Delta -0.0054 yards, CI half-width ~0.05 yards) sets the noise floor. The Ridge-vs-Tweedie comparison should be similarly noisy. NULL remains plausible — the probe is genuinely uncertain.

### 1.3 Success criteria

The probe **ships** when all four pass. The verdict (SIGNAL / NULL / REGRESSION) is informational, not a ship gate.

1. **Coverage:** WR rows with `targets > 0` per eval year >= 0.95 across all four eval years 2021-2024. Same default `--coverage-threshold 0.95` as PR #39. PR #39's run measured 0.989+ per year; no relaxation expected. Relaxation triggers the PR #31 retrospective rule (marginal-zone magnitudes under coverage relaxation are MARGINAL not SIGNAL).
2. **Probe completeness:** Walk-forward over eval years `{2021, 2022, 2023, 2024}`; per-row residuals populated for both arms on every eval year; pooled paired-bootstrap CI on receiving_yards Delta-RMSE rendered. One summary report (`reports/feature_probe_tweedie_yards_per_target_summary.md`) plus one CSV (`reports/feature_probe_tweedie_yards_per_target.csv`).
3. **Verdict rule on receiving_yards Delta-RMSE (candidate − incumbent, i.e., tweedie − ridge):**
   - **SIGNAL** iff `rmse_delta.hi_95 < 0` (tweedie CI strictly negative)
   - **REGRESSION** iff `rmse_delta.lo_95 > 0` (tweedie CI strictly positive)
   - **NULL** otherwise (CI brackets zero)
   - No effect-size floor at the per-stat level (matches PR #32, PR #39). PR #31 retrospective flag applies at integration go/no-go time, not at probe verdict time.
4. **Verification gates green:** `mypy src tests scripts` strict + `ruff check` + `ruff format --check` clean. Relevant pytest subset green (probe module + walk-forward harness tests + CLI tests).

### 1.4 Out of scope (deferred follow-ups)

1. **Composite-fpts measurement.** Per-stat RMSE only. Composite-fpts is the integration adoption-gate concern (`(EnsembleModel with yards-per-target Tweedie-decomp, WR) vs current production (EnsembleModel with direct-Ridge on receiving_yards, WR)` if SIGNAL).
2. **Variance / distribution comparison at predict time.** Tweedie p=1.5 distribution vs clipped-Normal(mu, sigma). Important for the integration's p10/p90 calibration story but not gated by this probe.
3. **`td_rate_per_target` factor-appropriate sub-model** (Poisson / logistic). Separate probe cycle.
4. **Other receiving stats.** `receptions` is decomposed in production (PR #36/#38); not re-probed here. `rushing_yards` / `rushing_tds` / `fumbles_lost` are out of scope as in PR #32 §1.4 #5.
5. **Other positions.** RB / TE have rushing/receiving factors that could decompose similarly; QB has a different (passing-chain) decomposition entirely. Each is its own probe + integration cycle. WR-only here.
6. **Integration build-out.** No changes to `src/projections/models/decomposed_baseline.py` in this probe. If SIGNAL: the integration plan adds `Stat.RECEIVING_YARDS` to `_WR_FACTORIES["decomposed-baseline"]`'s `decomposed_stats` mapping with the Tweedie efficiency sub-model wired in. Composite gate vs current production `ensemble-decomposed` (which decomposes RECEPTIONS only).
7. **Tweedie with CV-selected power.** Power fixed at 1.5 (standard compound Poisson-Gamma default). If Tweedie p=1.5 NULLs but mechanism prior remains strong, a separate probe with `power in {1.3, 1.5, 1.7}` CV-selected is a follow-up.
8. **Gamma GLM with filtered zeros.** Considered and rejected in brainstorming: Tweedie p=1.5 handles zeros natively without the selection bias of filtering rows with `receiving_yards == 0`.

---

## 2. Source data — already ingested

All inputs already exist in the feature cache and weekly stats. **Identical to PR #39's data surface.**

| Column | Type | Source | Use |
|---|---|---|---|
| WR features | `WrFeaturesSchema` | `data/features/wr/season=YYYY/week=WW/part.parquet` | X for both volume RidgeCV and efficiency sub-models |
| `targets` | `Series[int]` | `WeeklyStatsSchema.targets` | y for shared volume RidgeCV; filter (`> 0`) for efficiency fits |
| `receiving_yards` | `Series[int]` | `WeeklyStatsSchema.receiving_yards` | numerator for `yards_per_target = receiving_yards / targets`; final per-stat actual |

**No new ingest, no schema changes, no override parquet.** The probe consumes the existing per-week WR feature parquet and joins to `weekly_stats` on `(gsis_id, season, week)` exactly as `BaselineModel.fit` does (and as `logit_catch_rate_probe.py` does).

---

## 3. Architecture

### 3.1 New module — `src/projections/backtest/tweedie_yards_per_target_probe.py`

Mirrors `src/projections/backtest/logit_catch_rate_probe.py`'s shape. Pure numpy / pandas / sklearn. Reuses `paired_bootstrap_rmse_delta` and `BootstrapDelta` from `src/projections/backtest/adoption_gate.py` unchanged.

```
tweedie_yards_per_target_probe.py
├── _fit_shared_volume(X, y_targets, alphas) -> RidgeCV
│       # shared volume sub-model on targets ~ X; identical to logit_catch_rate_probe._fit_shared_volume
├── _fit_ridge_efficiency(X_pos, ratio, alphas) -> RidgeCV
│       # Arm A — RidgeCV on yards_per_target ratio. Matches the DecomposedBaselineModel
│       # recipe (efficiency_clip_hi=+inf branch from decomposed_baseline.py).
├── _fit_tweedie_efficiency(X_pos, ratio, alphas, power=1.5) -> Pipeline
│       # Arm B — sklearn Pipeline:
│       #   StandardScaler() -> GridSearchCV(TweedieRegressor(power=1.5, link="log",
│       #                       max_iter=200), param_grid={"alpha": alphas}, cv=5,
│       #                       scoring=make_scorer(mean_tweedie_deviance, power=1.5,
│       #                       greater_is_better=False), refit=True)
│       # Returns the fitted Pipeline; downstream `predict()` handles scaling
│       # internally and TweedieRegressor.predict applies inverse-log link.
├── _predict_yards_ridge(mu_targets, X_eval, ridge_eff) -> np.ndarray
│       # mu_targets * clip(ridge_eff.predict(X_eval), 0, +inf)
├── _predict_yards_tweedie(mu_targets, X_eval, tweedie_eff) -> np.ndarray
│       # mu_targets * tweedie_eff.predict(X_eval)   (no manual exp; Pipeline + GLM handle it)
├── walk_forward_residuals(features, weekly_stats, eval_years) -> ProbeResults
│       # For each year in eval_years:
│       #   - train on seasons [start..year-1]
│       #   - fit shared volume Ridge on all WR train rows
│       #   - mask train rows on targets > 0
│       #   - compute ratio = receiving_yards / targets on masked rows
│       #   - fit ridge_eff on (X_pos, ratio)
│       #   - fit tweedie_eff via Pipeline on (X_pos, ratio)  -- same rows, same y
│       #   - eval on year:
│       #       mu_targets = volume.predict(X_eval)
│       #       pred_ridge[year] = _predict_yards_ridge(mu_targets, X_eval, ridge_eff)
│       #       pred_tweedie[year] = _predict_yards_tweedie(mu_targets, X_eval, tweedie_eff)
│       #       actual[year] = receiving_yards
│       # Concatenate per-year arrays into pooled (actual, pred_ridge, pred_tweedie, year_id)
│       # buffers. Returns ProbeResults with these buffers + per-year coverage stats.
└── compute_verdict(results: ProbeResults, n_bootstrap=1000, seed=42) -> PerStatVerdict
        # Pooled paired-bootstrap CI on (residuals_tweedie - residuals_ridge) where
        # residuals_arm = (actual - pred_arm). Returns BootstrapDelta + verdict label.
```

**`PerStatVerdict` and `ProbeResults` shapes:** match `logit_catch_rate_probe.py`'s dataclass shapes. May `from projections.backtest.logit_catch_rate_probe import PerStatVerdict, VerdictLabel, ProbeResults` if those types are public; otherwise re-define locally. Plan task decides — both paths are clean.

**Both arms fit on the SAME train rows.** The `targets > 0` mask defines the efficiency-fit row set for both arms. The Tweedie arm does NOT additionally filter `receiving_yards > 0` — power=1.5 handles the zero point mass natively via compound Poisson-Gamma likelihood. This is the entire point of choosing Tweedie over Gamma.

### 3.2 New CLI script — `scripts/probe_tweedie_yards_per_target.py`

Mirrors `scripts/probe_logit_catch_rate.py`. argparse for `--eval-years`, `--coverage-threshold`, `--seed`, `--n-bootstrap`. Runs `walk_forward_residuals` then `compute_verdict`. Writes:

- `reports/feature_probe_tweedie_yards_per_target_summary.md` — verdict + 95% CI + n_paired + per-year coverage table + magnitude flag note if |Delta_yards| < 0.05 (the receiving_yards-per-stat equivalent of PR #31's 0.005 fpts composite-fpts threshold, given PPR yards coefficient 0.1).
- `reports/feature_probe_tweedie_yards_per_target.csv` — per-year and pooled rmse deltas (matches PR #39's CSV shape).

### 3.3 No edits to existing code

- `src/projections/models/decomposed_baseline.py` is **not touched** by this probe. The integration plan handles any production swap.
- `src/projections/backtest/logit_catch_rate_probe.py` is **not touched**. The new probe is its own module; reuse is via `from projections.backtest.adoption_gate import paired_bootstrap_rmse_delta, BootstrapDelta` and (optionally) `from projections.backtest.logit_catch_rate_probe import PerStatVerdict, VerdictLabel`.
- `src/projections/backtest/target_decomposition_probe.py` is **not touched**.

### 3.4 No schema, no codec, no factory changes

The probe operates in-memory against sklearn estimators fit on numpy arrays. No new Distribution classes, no codec edits, no new `_WR_FACTORIES` registration. The probe is strictly a **mechanism test**.

### 3.5 sklearn Tweedie / Pipeline / GridSearchCV details

- **`TweedieRegressor(power=1.5, link="log", max_iter=200)`** — sklearn default solver is `lbfgs`; `max_iter=200` (sklearn default is 100) for safety against convergence warnings on rows with extreme features. `link="log"` is the canonical link for compound Poisson-Gamma.
- **alpha grid: `np.logspace(-3, 3, 7)`** — 7 points spanning 6 orders of magnitude. Ridge uses 13 points but Tweedie fits are ~5-10x slower; 7 covers the same effective regularization range. Plan task can widen to 9-11 if real-data runtime budget permits.
- **`GridSearchCV(..., cv=5, scoring=make_scorer(mean_tweedie_deviance, power=1.5, greater_is_better=False), refit=True)`** — 5-fold inner CV on the training fold (NOT cross-fold-leaking with the outer walk-forward window). `refit=True` produces a final estimator trained on the full train fold with the selected alpha. Scoring uses `sklearn.metrics.mean_tweedie_deviance` with matching `power=1.5`; `greater_is_better=False` because deviance is loss-style (lower = better).
- **`StandardScaler()` upstream of TweedieRegressor.** Tweedie's L2 penalty is scale-dependent (alpha-times-norm-of-coefs); WR features are not pre-scaled in `BaselineModel.fit`. The Ridge arm doesn't need scaling because Ridge's CV-selected alpha absorbs the scale; documenting the asymmetry in the spec means reviewers don't flag it as an arm-fairness issue. Standard practice in sklearn GLM examples.
- **`Pipeline([("scale", StandardScaler()), ("gscv", GridSearchCV(...))])`** — Pipeline composes `.predict()`: scaler transforms test X then GridSearchCV's best estimator predicts. Inverse-log-link is applied by `TweedieRegressor.predict()` internally; the probe code does NOT manually call `exp()`.

---

## 4. Testing

`tests/test_backtest/test_tweedie_yards_per_target_probe.py`:

1. **`_fit_ridge_efficiency` matches production** — fit on a small synthetic ratio frame; verify the returned RidgeCV's `predict` matches `decomposed_baseline._fit` (same RidgeCV alpha grid, same data). Pin against accidental drift in the incumbent arm. Direct analog to PR #39's "ridge matches production" test.
2. **`_fit_tweedie_efficiency` recovers known coefficients** — synthetic 300-row Tweedie-generated fixture: draw `targets ~ Poisson(5)`, on rows with `targets > 0` draw `yards_per_target ~ Tweedie(mu=exp(b0 + b1 * x1), p=1.5, dispersion=phi)` for chosen `(b0, b1, phi)`. Fit Pipeline; verify `pipeline.predict(X)` recovers true mu within tolerance (relative error < 20% on average; Tweedie fits are noisier than Bernoulli, so tolerance is generous). Pins that the Pipeline + GridSearchCV + TweedieRegressor path actually fits a log-link Tweedie.
3. **`_fit_tweedie_efficiency` handles zero-yards rows** — synthetic fixture with `yards_per_target == 0` on ~30% of rows (mirrors realistic incompletion rate). Verify the Pipeline fits without raising and `pipeline.predict(X) > 0` for all eval X. Pins Tweedie's zero-handling.
4. **`_predict_yards_ridge` vs `_predict_yards_tweedie` divergence test** — on a synthetic fixture with extreme features (where Ridge predicts ratio < 0 before clipping), verify the two arms produce different predictions. Property: at extreme features the Tweedie prediction stays in (0, mu_max) while the Ridge prediction lands at 0 after clip. The integration plan would later test that this difference improves on average across the real WR row distribution.
5. **`walk_forward_residuals` integration** — synthetic 4-season frame; verify both pred buffers are populated, length matches actual, coverage stat populated, both pred arrays are strictly positive (Tweedie) and >= 0 (Ridge).
6. **`compute_verdict` verdict mapping** — three crafted ProbeResults inputs (synthetic residuals) hitting each of SIGNAL / NULL / REGRESSION. Pins the verdict-mapping logic. Trivially adapted from PR #39's identical test.
7. **CLI smoke** — `tests/test_scripts/test_probe_tweedie_yards_per_target_cli.py` (new). Mocks `walk_forward_residuals`, verifies argparse + report writing.

Real-data smoke + the actual probe verdict captured in the PR per CLAUDE.md "Forced verification" rule.

---

## 5. Risk register

1. **Magnitude prior is small relative to noise floor.** Per §1.2, expected per-stat |Delta_yards| is plausibly 0.05-0.5; the noise floor (PR #32 CI half-width) is ~0.05 yards. The probe's discriminative power on yards_per_target is lower than on catch_rate. NULL is the most likely outcome even with a real Tweedie lift, unless the lift sits in the upper half of the expected range. **Mitigation:** the probe ships either way; the integration gate weights magnitude separately. Magnitude flag in the report if |Delta_yards| < 0.05 (composite-fpts equivalent 0.005, per PR #31's threshold).
2. **NULL on receiving_yards.** Plausible: Ridge-on-clipped-ratio may already approximate the Tweedie mean within Monte-Carlo noise on this dataset, just as it did for Bernoulli mean in PR #39. **Mitigation:** NULL closes the yards_per_target factor-appropriate slot at the WR cell. Next slot per PR #39's chain: `td_rate_per_target` factor-appropriate probe (logistic / Poisson, since td_rate is a [0,1] rate with extreme sparsity). NULL doesn't close the broader factor-appropriate-sub-model direction.
3. **TweedieRegressor convergence warnings.** Default solver is `lbfgs`; bumping `max_iter=200` (vs sklearn default 100) is the standard safety guard. Convergence warnings on a few feature configurations are acceptable as long as the converged fit is well-behaved on validation rows. **Mitigation:** plan Task 2 verifies on real data that warnings (if any) don't correlate with bad eval-row predictions. Add to `pyproject.toml` `filterwarnings` only after verifying convergence isn't pathological.
4. **GridSearchCV inner-CV vs walk-forward outer-CV time-leakage.** GridSearchCV's inner cv=5 partitions the training fold (seasons <= year-1); does NOT include eval-year rows. Same structure as PR #39's `LogisticRegressionCV(cv=5)`. No leakage. **Mitigation:** plan task tests verify train-row season range against eval-year-minus-1.
5. **Runtime.** Tweedie GLM fits ~5-10x slower than Ridge per fit. 7-point alpha grid × 5 CV folds × 4 walk-forward years ≈ 140 Tweedie fits on ~6K-8K rows × ~20 features. Estimate: 5-15 min total walk-forward (catch_rate probe was ~3 min; Tweedie is slower per fit but Pipeline + GridSearchCV is well-optimized). **Mitigation:** none needed; CLI prints per-year wall-clock and total.
6. **PR #32 already measured NULL for Ridge-decomp vs direct on receiving_yards.** This probe's incumbent arm (Ridge-decomp) is NOT current production; current production is direct RidgeCV. A SIGNAL verdict here (Tweedie > Ridge-decomp) does NOT imply Tweedie-decomp beats current production; that's the integration adoption-gate's question. **Mitigation:** spec §1.4 #1 names this explicitly. The integration plan if SIGNAL must run a separate adoption gate vs current production `ensemble-decomposed`. Failing to mention this caveat in the summary report would mislead future readers.
7. **Sub-model regularization scale mismatch.** Ridge uses alpha (penalty strength on raw coefs); TweedieRegressor uses alpha on coefs of *scaled* features (due to Pipeline's StandardScaler). The two grids span different optimization landscapes. **Mitigation:** plan Task 2's alpha grid `np.logspace(-3, 3, 7)` covers 6 orders of magnitude on the scaled-feature space, which spans the realistic range of effective regularization for a 20-feature × 6K-8K row regression. Same approach as PR #39's `Cs=[0.01, 0.1, 1.0, 10.0, 100.0]` vs Ridge's `alphas=np.logspace(-3, 3, 13)`. Magnitude of `alpha` is not directly comparable across arms; only the *fit quality* is.
8. **Unicode print error on Windows cp1252** (PR #39 surfaced this). The CLI's stdout summary may print `Delta` symbol or unicode em-dashes; Windows cp1252 will crash. **Mitigation:** plan Task 4 uses ASCII-only stdout (write `delta` not `Delta`; use `--` not em-dash); reports are written with `encoding="utf-8"` explicitly.

---

## 6. Reports

`reports/feature_probe_tweedie_yards_per_target_summary.md`:

- Verdict (SIGNAL / NULL / REGRESSION) + 95% CI on receiving_yards Delta-RMSE (yards units).
- Per-year breakdown (informational): Delta-RMSE point + CI per eval year.
- Coverage stats: `targets > 0` rate per year (eval), per train window (train).
- Magnitude flag (informational): is |Delta_yards| < 0.05? Composite-fpts equivalent (multiply by 0.1) flagged.
- Mechanism caveat: explicit note that incumbent arm is Ridge-decomp, NOT current production (direct Ridge); integration adoption-gate comparison is separate.
- Recommended next direction per verdict:
  - SIGNAL → integration plan; new adoption gate vs production `ensemble-decomposed`.
  - NULL → close cell; name `td_rate_per_target` probe (logistic / Poisson, [0,1] rate with extreme sparsity) as next slot.
  - REGRESSION → close cell strongly. Tweedie with CV-selected power could be a separate follow-up if mechanism prior remains strong.

`reports/feature_probe_tweedie_yards_per_target.csv` — long-form per-year deltas matching PR #39's CSV shape.

---

## 7. Estimated scope

~5 plan tasks. Single session. Real-data probe runtime estimated 5-15 min walk-forward across 4 eval years.

| Task | Surface | Files touched |
|---|---|---|
| 1. `_fit_shared_volume` + `_fit_ridge_efficiency` + unit tests | probe core | `src/projections/backtest/tweedie_yards_per_target_probe.py` (new), `tests/test_backtest/test_tweedie_yards_per_target_probe.py` (new) |
| 2. `_fit_tweedie_efficiency` (Pipeline + GridSearchCV) + unit tests | probe core | extend module + tests; verify Tweedie-known-coef recovery and zero-handling |
| 3. `_predict_*` + `walk_forward_residuals` + `compute_verdict` + integration tests | probe core | extend module + tests |
| 4. CLI script + CLI smoke | scripts + tests | `scripts/probe_tweedie_yards_per_target.py` (new), `tests/test_scripts/test_probe_tweedie_yards_per_target_cli.py` (new) |
| 5. Real-data probe run + report + PM/TODO updates | reports + PM | `reports/feature_probe_tweedie_yards_per_target_summary.md`, `reports/feature_probe_tweedie_yards_per_target.csv`, `project_management.md`, `TODO.md` |

End-to-end: 1 focused session.

---

## 8. Implementation plan handoff

After spec approval and commit on `feat/probe-tweedie-yards-per-target`, the next step is the writing-plans skill to produce `docs/superpowers/plans/2026-05-16-tweedie-yards-per-target-probe.md` decomposing the 5 tasks above into per-task implementation steps with per-task verification commands.
