# Plan 3e — Calibration tightening — Design

**Status:** approved (brainstorming). Phase 0 complete; Phase 1+ amendment landed 2026-04-26 after Phase 0 evidence (see `docs/superpowers/research/2026-04-26-calibration-diagnosis.md`).
**Date:** 2026-04-26
**Author:** alden + claude
**Builds on:** `2026-04-26-plan-3d-monte-carlo-season-design.md` (the per-row seeds + per-stat params blob + season aggregator this plan tightens). Closes TODO #22.

**Plan-3 series context:**

- **Plan 3a (merged at `598ab9c`):** Model A on WR only.
- **Plan 3b (merged at `c4a0401`):** Model A generalized to QB / RB / TE.
- **Plan 3c (merged at `3db71a6`):** Walk-forward backtest harness + snapshot-diff CI gate.
- **Plan 3d (merged at `fe55d5b`):** Real Monte Carlo season-distribution aggregation; per-row deterministic seeds; season-total calibration added to the gate.
- **Plan 3e (this design):** Calibration tightening. Two-pass spec: Phase 0 = diagnostic, Phase 1+ = TBD amendment after Phase 0 evidence lands.
- **Plan 4 (next, not in this design):** Public Python API + CLI verbs + free-tier web hosting.

---

## 1. Overview

Plan 3d's snapshot pinned the regression floor at the calibration coverage Model A actually produces, but those numbers are well below target:

- **Weekly `[p10, p90]` coverage:** 0.67–0.80 across the 16 (position, year) cells (target 0.80). Worst on WR / QB.
- **Season `[p10, p90]` coverage:** 0.30–0.55 across the 16 (position, year) cells (target 0.80). Worst on QB seasons (0.31–0.39).

The under-dispersion is structural. When independent under-dispersed weekly distributions are summed across the season — variances add, but the systematic miss does not cancel — the season distribution under-disperses further. So weekly tightening pays compound interest on season calibration. Conversely, no amount of season-level tweaking compensates for weekly under-dispersion.

The fix could be in any of several places, with very different surface areas:

- **Variance estimator quality** — `_gamma_alpha_from_residuals` uses method-of-moments; an MLE fit may produce a tighter alpha. Likely small effect because MoM and MLE for gamma usually agree on the first two moments.
- **Heteroscedasticity** — both `_normal_std_from_residuals` and `_gamma_alpha_from_residuals` produce a single per-stat parameter; in reality, residual variance probably scales with predicted mean. A per-tertile bucketing fix is mechanical.
- **Family misspecification** — NORMAL has too-thin tails for football outcomes (long TDs, big games are heavy-tail events); GAMMA's constant-CoV assumption (`var = mu^2 / alpha`) is rigid. Fixing this needs new families (Student-t, log-normal, Negative Binomial) and is the largest surface area.
- **Low-mean rare-event distortion** — `_GAMMA_MU_FLOOR = 1e-3` keeps `scale = mu / alpha` defined, but for `mu_hat << 1` (receiving_tds, fumbles_lost) the resulting distribution is degenerate. Real low-mean integer counts are not gamma-shaped at all.

Picking the wrong knob would ship a parametric tightening that moves the dial by 3% when we needed 15%. Picking blindly across all four would balloon the plan. So Plan 3e leads with a **diagnostic phase** that characterizes each (position, stat) cell's residual distribution against the assumed family and ranks alternative families. The implementation phases are deferred to a **spec amendment** written after Phase 0 evidence is in hand.

This document specifies Phase 0 in full. The amendment will land in this same branch / PR.

### 1.1 Goals (Phase 0 only)

- Add `scripts/diagnose_calibration.py`: a CLI that loads the most recent backtest run's `results.parquet`, computes per-(position, stat) residual diagnostics for the held-out years, fits and ranks 2–3 alternative families, and writes structured artifacts + plots to `data/diagnostics/calibration_<ts>/`.
- Add a human-written summary report at `docs/superpowers/research/2026-04-26-calibration-diagnosis.md`. Per-(position, stat) findings table + recommended-fix matrix + one-paragraph synthesis. The script runs first; the human (alden + claude) reads the artifacts and writes the report.
- Add `tests/test_scripts/test_diagnose_calibration.py`: smoke test that runs the script against a tiny synthetic `results.parquet`, asserts artifacts are produced, asserts no exceptions.
- Update `.gitignore` so `data/diagnostics/` is gitignored (mirrors `data/backtest/`).
- Decision-gate the spec on the diagnostic report: after Phase 0 lands, the brainstorming skill is re-invoked in this branch to amend the spec with Phase 1+ implementation phases.

### 1.2 Goals (Phase 1+, locked in by 2026-04-26 amendment)

Phase 0's diagnostic identified three distinct root causes ([report](../research/2026-04-26-calibration-diagnosis.md)). Phase 1+ addresses them in three sequential phases, biggest-bang-first:

- **Phase 1 — `ParametricNegativeBinomial` for zero-inflated counts.** Closes the 10 cells (`*_tds`, `interceptions`, `fumbles_lost` across all four positions) where coverage is currently 0.0 because GAMMA can't represent a point mass at zero.
- **Phase 2 — `ParametricStudentT` for heavy-tailed yards.** Closes the 5 cells where Phase 0's AIC strongly preferred Student-t over Normal (`*_yards` for QB rushing, RB rushing, TE receiving, WR receiving, marginally QB passing).
- **Phase 3 — Per-tertile variance bucketing.** Cross-cutting; applies to all families. Addresses the 18-of-24 cells with `heteroscedasticity_ratio > 1.5` — high-pred tertiles carry 1.5–2.7× the residual std of low-pred tertiles, indicating constant per-stat sigma is the wrong scale model.

Each phase ends with a wholesale re-snapshot of `tests/backtest/baseline_metrics.json` and a PM-doc update reporting the per-cell coverage delta. The existing 0.03 absolute calibration tolerance carries forward as the no-backsliding floor at each new step.

After all three phases land:

- Update `pack_per_stat_params` codec for the two new families.
- Retrain `models/artifacts/*.joblib` standalone artifacts (same pattern as 3b → 3a — old artifacts unloadable; the backtest harness regenerates fold artifacts from the feature cache, so the gate is unaffected).
- Update `project_management.md`; close TODO #22.

### 1.3 Non-goals (deferred)

- **Model-architecture changes** (GBM, quantile regression) → Plan 5 territory. Plan 3e stays within the parametric-distribution + per-stat Ridge frame.
- **Joint correlations between players or stats** → TODO #1, DFS Engine. Plan 3e is per-player marginal calibration only.
- **Cross-week residual correlation modeling.** Plan 3d's season aggregator assumes within-player cross-week independence. If Phase 0 reveals that cross-week correlation is the dominant driver of season under-dispersion (rather than weekly under-dispersion compounding), the diagnostic report flags it and the amendment defers to a follow-up plan; Plan 3e does not introduce a time-series component.
- **Hard coverage targets per cell.** Plan 3e is best-effort: minimum-cell coverage improves to ≥ 0.65 and mean coverage across all 32 cells improves by ≥ 0.10. Coverage that lands above the floor is the win; further tightening, if needed, is a 3f follow-up.
- **Inflation-factor calibration** (multiply std by a constant to hit target). If parametric tightening doesn't reach the floor, the amendment can revisit; not on the table at v1.
- **K / DST positions** → TODO #10.

---

## 2. Architecture (Phase 0)

### 2.1 New files

```
scripts/
└── diagnose_calibration.py            # NEW: CLI
docs/superpowers/research/             # NEW DIRECTORY
└── 2026-04-26-calibration-diagnosis.md  # NEW: human-written report
tests/test_scripts/                    # NEW DIRECTORY (if absent)
├── __init__.py
└── test_diagnose_calibration.py       # NEW: smoke test
.gitignore                             # extended: data/diagnostics/
```

No source code under `src/projections/` is modified in Phase 0. The model, the harness, the gate, the snapshot — all unchanged. Phase 0 is pure analysis on existing artifacts.

### 2.2 Inputs

The diagnostic operates on the latest `data/backtest/run_<ts>/results.parquet` produced by `scripts/backtest.py` (default-on smoke or `--run-backtest` gate). Each row has, per stat configured for the position:

- `<stat>_pred` — `mu_hat` from the fitted Ridge.
- `<stat>_actual` — observed value.
- `mean`, `p10`, `p50`, `p90` — composite-points distributional summary.
- `family` — the assumed distribution family (`SAMPLED_SUMMARY` for 3d output).
- `params` — per-stat distribution parameter blob, decodable via `unpack_per_stat_params` (3d's codec).

For each (position, stat) cell, residuals are computed as `<stat>_actual - <stat>_pred`, and the assumed-family parameters per row come from unpacking `params`.

If `data/backtest/run_<ts>/` is empty (no recent run on the box), the script exits with a clear error pointing at `python scripts/backtest.py --report` to generate one. The script does not refit the model; that's `backtest.py`'s job.

### 2.3 Outputs

```
data/diagnostics/calibration_<ts>/     # gitignored; <ts> = run timestamp
├── residuals.parquet                  # per (position, stat, gsis_id, season, week)
├── summary.parquet                    # per (position, stat) aggregate diagnostics
└── plots/
    ├── <position>_<stat>_hist.png     # residual histogram + assumed-family overlay
    └── <position>_<stat>_qq.png       # Q-Q plot vs assumed family
```

`residuals.parquet` schema:

| column | type | notes |
|---|---|---|
| position | str | from results.parquet |
| stat | str | one of the position's target stats (e.g., `passing_yards`) |
| gsis_id | str | |
| season | int | held-out year |
| week | int | |
| pred | float | `<stat>_pred` |
| actual | float | `<stat>_actual` |
| residual | float | `actual - pred` |
| assumed_family | str | `NORMAL` / `GAMMA` |
| assumed_param_a | float | `std` (NORMAL) or `shape` (GAMMA) |
| assumed_param_b | float | `nan` (NORMAL — only one param) or `scale` (GAMMA, `= mu / shape`) |

`summary.parquet` schema:

| column | type | notes |
|---|---|---|
| position | str | |
| stat | str | |
| n | int | row count |
| mean_pred | float | |
| mean_actual | float | |
| residual_mean | float | bias signal |
| residual_std | float | global per-stat std (the homoscedastic estimate) |
| residual_skew | float | Fisher's; pos = right-skewed |
| residual_excess_kurtosis | float | Pearson - 3; pos = heavy-tailed |
| std_tertile_low | float | residual std in `mu_hat`'s lower tertile |
| std_tertile_mid | float | |
| std_tertile_high | float | |
| heteroscedasticity_ratio | float | `std_tertile_high / std_tertile_low` (>1.5 ≈ meaningful) |
| coverage_p10p90 | float | per-stat individual coverage of the assumed-family `[p10, p90]` |
| coverage_le_p90 | float | per-stat individual coverage of the assumed-family `[-inf, p90]` |
| ks_assumed_stat | float | KS statistic of standardized residuals vs assumed-family CDF |
| ks_assumed_pvalue | float | |
| best_alt_family | str | `student_t` / `log_normal` / `neg_binomial` / `none` |
| best_alt_aic | float | AIC of best-fit alternative |
| assumed_aic | float | AIC of the assumed family fit on residuals |
| aic_delta | float | `assumed_aic - best_alt_aic`; positive = alternative fits better |
| recommended_fix | str | `variance_bucket` / `family_swap` / `combined` / `no_change` (family name lives in `best_alt_family`) |
| assumed_family | str | informational — the family the per-row params blob assumed (`NORMAL` / `GAMMA`); useful for the report writer to know which `assumed_aic` was computed against which family |

### 2.4 Plots

Two PNGs per (position, stat):

- **`*_hist.png`** — histogram of residuals (or, where appropriate, of standardized residuals); assumed-family density overlaid on the same axes; alternative-family density overlaid in a contrasting color. Tail mismatch is visually obvious.
- **`*_qq.png`** — Q-Q plot of standardized residuals vs assumed-family quantiles. Heavy tails curve away from the diagonal at the extremes.

Plots are static PNGs (matplotlib + savefig); no interactive backend, so the script runs headless. Plots are not load-bearing for the recommended-fix decision — they're for human visual confirmation when alden + claude write the report.

### 2.5 Recommended-fix decision rule

Per (position, stat), inside the script:

```
if heteroscedasticity_ratio > 1.5 and aic_delta < 5:
    recommended_fix = "variance_bucket"
elif heteroscedasticity_ratio <= 1.5 and aic_delta >= 5:
    recommended_fix = "family_swap"  # family name lives in best_alt_family
elif heteroscedasticity_ratio > 1.5 and aic_delta >= 5:
    recommended_fix = "combined"
else:
    recommended_fix = "no_change"
```

Thresholds are deliberately rough; the report-writing pass overrides where visual / domain reasoning disagrees. The numeric rule's job is to bucket cells for the report, not to make the call.

### 2.6 Alternative-family menu

For each (position, stat), the script attempts to MLE-fit:

- **Continuous yards stats** (`passing_yards`, `rushing_yards`, `receiving_yards`):
  - Student-t (heavy-tailed, symmetric): `scipy.stats.t.fit` with location and scale.
  - Log-normal (positive support, right-skewed): `scipy.stats.lognorm.fit`. Skipped if `min(actual) <= 0`.
- **Discrete low-mean counts** (`passing_tds`, `interceptions`, `rushing_tds`, `receiving_tds`, `fumbles_lost`):
  - Negative Binomial: MLE via `scipy.optimize.minimize` on the log-likelihood (`scipy.stats.nbinom` does not support `.fit`). Two parameters: `n` (dispersion), `p` (success probability).
- **Higher-mean counts** (`receptions`):
  - Negative Binomial (as above; comparison to existing GAMMA-via-continuous treatment).

(`sacks_per_game_l4` is a QB *feature*, not a target stat; it does not appear in any position's `target_stats` and is not in scope.)

AIC = `2k - 2 * log_likelihood`, where `k` is the number of fitted parameters. Lower is better. Computed on the same data the assumed family was fit to (the residuals or the actuals, family-dependent — Section 4 documents the convention).

If a candidate fit fails (singular, non-finite log-likelihood, `scipy` exception), the script logs a warning and records `nan` for that family's AIC; the row's `best_alt_family` falls back to whichever candidate did fit, or `none`.

---

## 3. Decision gate: Phase 0 → Phase 1+

After Phase 0 lands (the script is committed, runs cleanly, produces artifacts; the human-written report is committed):

1. Re-invoke `superpowers:brainstorming` in this branch with Phase 0's findings as input.
2. The brainstorming session selects the implementation strategy: which (position, stat) cells get bucket fixes, which get family swaps, which stay unchanged.
3. The spec is amended in-branch with Phase 1+ phases (numbered Phase 1, Phase 2, etc.; Phase 0 is preserved as historical record).
4. Implementation begins only after the amendment lands.

The amendment is a separate brainstorming pass; this spec deliberately does not pre-write Phase 1 alternatives. Doing so would foreclose on findings the diagnostic surfaces.

---

## 4. Implementation notes (Phase 0)

### 4.1 Residual standardization

For per-row alternative-family fits and KS tests, residuals are standardized against the assumed-family scale:

- **NORMAL stats:** `z_i = (actual_i - pred_i) / sigma`, where `sigma` is the assumed `std` (constant across rows under current homoscedastic estimator). KS test is against `scipy.stats.norm.cdf`.
- **GAMMA stats:** comparison is direct, not standardized. The assumed gamma is row-specific (`shape`, `scale = mu_i / shape`). Per-row CDF transform gives `u_i = scipy.stats.gamma.cdf(actual_i, shape, scale=scale_i)`. Under H0 (the assumed family is correct), `u_i ~ Uniform(0, 1)`. KS test is against `scipy.stats.uniform.cdf`. Heavy-tail or shape-misfit signals show up as deviation from uniform.

### 4.2 AIC consistency

The assumed-family AIC and alternative-family AIC must be computed on the same data, otherwise the comparison is meaningless. The script:

- For NORMAL stats: assumed log-likelihood is `sum(scipy.stats.norm.logpdf(actual_i, loc=pred_i, scale=sigma))`. Alternative families fit on the residuals (`actual - pred`) and their log-likelihoods evaluated there. Both are likelihoods over the same `actual` values, conditional on `pred`; AIC compares them validly.
- For GAMMA stats: assumed log-likelihood is `sum(scipy.stats.gamma.logpdf(actual_i, shape, scale=scale_i))`. Negative Binomial alternative fit on `actual_i` directly (integer counts). Same data, same scale.

A short comment block in the script documents this; getting it wrong is a silent bug.

**Caveat — marginal-vs-conditional asymmetry in the implemented fits.** The Phase 0 implementation cuts a corner that callers must be aware of: `_fit_student_t`, `_fit_log_normal`, and `_fit_neg_binomial` all fit the alternative family **marginally on `actual` alone**, discarding `pred`. The assumed-family AIC, by contrast, is **conditional** on `pred` (it uses `pred` as the per-row location). As long as `pred` carries any signal, the conditional assumed-family log-likelihood will tend to dominate the marginal alternative-family log-likelihood, biasing `aic_delta` away from `family_swap` recommendations. The diagnostic compensates by reporting `coverage_p10p90` and `ks_assumed_pvalue` as orthogonal signals that don't suffer this asymmetry, and the report-writing pass overrides the mechanical `recommended_fix` column when those signals disagree. Phase 1+ implementations that want a true apples-to-apples AIC comparison must also fit the alternative families conditionally (e.g., a Student-t with `loc = pred`) — or accept the asymmetric framing and lean on coverage / KS for the family-fit signal.

### 4.3 Tertile binning

`std_tertile_*` columns are computed as: sort rows by `pred`, split into 3 equal-count buckets, compute residual std within each. `heteroscedasticity_ratio = std_tertile_high / std_tertile_low` (clamped to a small positive denominator to avoid division by zero on degenerate stats).

### 4.4 CLI

```
python scripts/diagnose_calibration.py [--run-dir PATH] [--out-dir PATH]
```

- `--run-dir`: path to a `data/backtest/run_<ts>/` directory. Defaults to the latest by timestamp under `data/backtest/`.
- `--out-dir`: path to write diagnostic artifacts. Defaults to `data/diagnostics/calibration_<ts>/` matching the run timestamp.

The script prints the summary table to stdout (compact), then exits 0. No exit-code gating.

### 4.5 Dependencies

- `scipy.stats` — `scipy>=1.12` is already in `pyproject.toml`'s main `dependencies`. The diagnostic adds the first direct `scipy.stats` import in this project.
- `matplotlib` for plots — currently NOT in `pyproject.toml`. Phase 0 adds it under `[project.optional-dependencies].dev`. Production runtime never imports `scripts/`, so dev-only is correct and keeps the install footprint clean for the eventual web-hosting plan.

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    ...,
    "matplotlib>=3.8",
]
```

### 4.6 Smoke test

`tests/test_scripts/test_diagnose_calibration.py`:

- Builds a tiny synthetic `results.parquet` in a `tmp_path` directory with 3 positions × 2 stats × ~20 rows.
- Invokes the script as a module via `subprocess.run([sys.executable, "scripts/diagnose_calibration.py", "--run-dir", tmp_path, "--out-dir", out])`.
- Asserts exit code 0.
- Asserts `residuals.parquet`, `summary.parquet`, and at least one PNG exist.
- Asserts the `summary.parquet` has a row per (position, stat) with all required columns and no `nan` in `recommended_fix`.

The smoke test does not validate diagnostic correctness — that's the human report's job.

---

## 5. Validation surface

### 5.1 Phase 0

- `python scripts/diagnose_calibration.py` runs cleanly against a real `data/backtest/run_<ts>/` and produces `summary.parquet`, `residuals.parquet`, and the full set of plots.
- `pytest -v tests/test_scripts/test_diagnose_calibration.py` passes.
- `mypy src tests scripts` — zero violations on the new files.
- `ruff check src tests scripts` — zero violations.
- `ruff format --check src tests scripts` — no drift.
- `docs/superpowers/research/2026-04-26-calibration-diagnosis.md` exists, has at minimum: per-(position, stat) findings table mirroring `summary.parquet`, recommended-fix matrix, synthesis paragraph identifying the 1–3 dominant root causes the amendment should address.

### 5.2 Plan 3e overall (best-effort)

After the amendment + Phase 1+ implementation:

- Min-cell coverage across all 32 calibration cells improves from current 0.31 to **≥ 0.65**.
- Mean coverage across all 32 cells improves by **≥ 0.10**.
- No regression on RMSE / MAE / Spearman beyond existing snapshot tolerances (5% rel RMSE/MAE; 0.02 abs Spearman).
- `pytest -m backtest --run-backtest` passes after re-snapshot.
- TODO #22 closed.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Phase 0 reveals dominant cause is cross-week residual correlation (which we can't fix in 3e). | The diagnostic report flags it explicitly. The amendment then defers to a follow-up plan (Plan 3f or Plan 5) and Plan 3e ships the parametric tightening that is in scope; we accept that the season-coverage target may stay below 0.80. |
| Alternative-family MLE fits fail silently on degenerate cells (`fumbles_lost` is mostly zero). | Script logs warnings, records `nan` for failed fits, does not abort. Recommended-fix rule treats `nan` AIC as no improvement. |
| `matplotlib` adds a heavy install footprint to dev. | Plots are nice-to-have, not load-bearing; `recommended_fix` uses numeric criteria. If the install is awful in CI later, plots can move behind a `--with-plots` flag. |
| The amendment becomes a second long brainstorming session after Phase 0. | That's the point; the alternative is writing a worse spec now. The brainstorming skill's terminal step is invoke writing-plans, so the amendment cycle is bounded. |
| Re-snapshot churn: Phase 1+ will shift every weekly row's calibration metric. | Same shape as Plan 3d's drift section — re-snapshot wholesale, document max abs/rel drift in PM doc, accept. The 0.03 calibration tolerance + 5% RMSE/MAE tolerance carry forward unchanged as the new no-backsliding floor. |

---

## 7. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-26 | Plan 3e leads with a diagnostic phase rather than implementing TODO #22's "MLE + variance buckets" directly. | TODO #22 lists candidate fixes but doesn't identify root cause. Three plausible causes (heteroscedasticity, family misspecification, low-mean distortion) need different fixes; throwing parametric tightening at the wrong cause moves the dial 3% when we needed 15%. Diagnostic is ~½ day; amortizes across Plan 3f and any future model-quality work. |
| 2026-04-26 | One PR, one branch, two-pass spec (Phase 0 now; Phase 1+ amendment after evidence). | Matches existing one-plan-one-PR project pattern. Splitting into two PRs (3e diagnostic + 3f tightening) adds plan/PR overhead. The amendment is a short brainstorming pass, not a full design redo. |
| 2026-04-26 | Diagnostic reads existing `data/backtest/run_<ts>/results.parquet` rather than refitting models. | `results.parquet` already has per-row predicted means + actuals + per-stat distribution params (Plan 3d shipped this). Refitting is unnecessary; reading is fast (~20K rows total) and keeps the diagnostic dependency-light. |
| 2026-04-26 | Medium-scope diagnostic: heteroscedasticity buckets + Q-Q + KS + alternative-family AIC for 2–3 candidates. | Narrow forecloses on family-swap as a fix (the most likely root cause for the under-dispersion pattern). Wide (running ablation backtests inside Phase 0) overlaps with Phase 1+ implementation. Medium answers "is the family wrong, and which fits better" without burning the implementation budget twice. |
| 2026-04-26 | Best-effort success criterion (min cell ≥ 0.65, mean improvement ≥ 0.10). | Hard target (e.g., ≥ 0.78 everywhere) risks blocking 3e on a structural problem (cross-week correlation) that no parametric fix solves. Best-effort + snapshot floor + the 0.65 / 0.10 sanity bar give a clear ship signal without constraining the toolset. |
| 2026-04-26 | Snapshot strategy: re-snapshot wholesale once Phase 1+ lands; existing 0.03 absolute calibration tolerance carries forward as the no-backsliding gate. | The gate's job is regression detection. "Coverage moving toward 0.80" is the plan's whole point and belongs in the PR description / PM doc, not in tests. Existing tolerances naturally enforce no-backsliding from whatever floor we land at. |
| 2026-04-26 | Standalone artifact retraining: same pattern as 3b → 3a. Saved `models/artifacts/*.joblib` become unloadable after `BaselineModel.variance_params` shape changes; mechanical retrain in a Phase N task. | Backtest harness is unaffected (regenerates fold artifacts from the feature cache). Standalone artifacts are only consumed by `scripts/sanity_check_baseline.py` and `scripts/predict_2024.py`; retraining is one CLI invocation. Maintaining backward compat for an internal-only artifact format is not worth the surface area. |
| 2026-04-26 | Alternative-family menu: Student-t + log-normal for continuous, Negative Binomial for low-mean integer counts. | Three families cover the three most plausible alternatives to NORMAL/GAMMA: heavy-tailed symmetric (Student-t), heavy-tailed positive-skewed (log-normal), heavy-tailed integer (NB). Adding more (Tweedie, generalized Pareto, mixture) is overfitting the menu before evidence demands it. |
| 2026-04-26 | `matplotlib` added to `[project.optional-dependencies].dev`, not main `dependencies`. | Plots are diagnostic-only; production runtime never imports `scripts/`. Dev-only keeps the install footprint clean for the eventual web-hosting plan. |
| 2026-04-26 | Phase ordering: NegBinomial first, Student-t second, variance bucketing third. | Biggest-bang-first. NB closes 10 cells where coverage is 0.0 (highest absolute coverage gain). Student-t closes 5 cells where heavy tails are confirmed. Bucketing is cross-cutting and operates on whatever families ended up after Phases 1-2; doing bucketing first would mean re-bucketing after the family swap. |
| 2026-04-26 | Skip Zero-Inflated Negative Binomial at v1; use plain NB. | NB with low mean already produces high mass at zero; adding a separate zero-component (ZIP) is a strict generalization but adds a parameter. If Phase 1 doesn't reach the 0.65 floor on any count cell, revisit ZIP in a follow-up plan. |
| 2026-04-26 | `receptions` stays on GAMMA at Phase 1; revisit only if it remains under-calibrated. | Phase 0 showed `receptions` cells at coverage 0.60-0.70 (under-calibrated but not catastrophic). Continuous-ish stat with mean ~1-3; GAMMA is borderline appropriate. Avoids over-scoping Phase 1; revisit empirically. |
| 2026-04-26 | NegativeBinomial parameterization: `(mean, dispersion)` with mean = per-row Ridge prediction (clamped) and dispersion fit globally per stat from training residuals. | Mirrors the existing per-row-conditional shape used by NORMAL (loc=pred, scale fit) and GAMMA (scale=pred/alpha, alpha fit). Keeps the row-level prediction → distribution mapping consistent across families. Conditional formulation also fixes Phase 0's marginal-vs-conditional AIC asymmetry caveat. |
| 2026-04-26 | Student-t parameterization: `(loc=pred, scale, df)` with scale + df fit globally per stat via `scipy.stats.t.fit` on residuals. | `loc = pred` matches the conditional NORMAL framing. Fitting df allows the family to flex toward Normal (df → ∞) where residuals are well-described as Normal, or toward heavy tails (df ≈ 4-10) where they aren't. Per-stat global df is the simplest sufficient statistic; per-bucket df is a Phase 4 concern if needed. |
| 2026-04-26 | Variance bucketing: 3 quantile-based buckets per (position, stat) on `mu_hat`; bucket boundaries persisted in the model artifact. | 3 buckets is the minimum that captures monotonic heteroscedasticity (Phase 0 reported `std_tertile_low/mid/high`). 5+ buckets adds parameter count without strong empirical justification. Quantile-based avoids degenerate buckets on long-tailed predictor distributions. Persisting boundaries is necessary for predict-time bucket lookup. |
| 2026-04-26 | Single family per (position, stat); buckets share the family. | Bucketing varies the variance parameter; family swaps vary the distribution shape. Allowing per-bucket family would explode configuration surface and complicate the codec for marginal gain. |

---

## 8. Architecture (Phase 1+)

### 8.1 Distribution layer additions

Two new parametric families added to `src/projections/distributions/parametric.py`, mirroring the existing `ParametricNormal` / `ParametricGamma` shape:

```python
@dataclass(slots=True, frozen=True, init=False)
class ParametricNegativeBinomial:
    """Negative Binomial parameterized as (mean, dispersion).

    n = mean^2 / max(dispersion, eps)
    p = n / (n + mean)
    var = mean + mean^2 / dispersion (overdispersion vs Poisson when dispersion < inf)
    """
    mean_: float
    dispersion_: float
    # ... mean(), std(), quantile(q), sample(n, rng) per Distribution Protocol


@dataclass(slots=True, frozen=True, init=False)
class ParametricStudentT:
    """Student-t parameterized as (loc, scale, df).

    var = scale^2 * df / (df - 2)  for df > 2
    """
    loc_: float
    scale_: float
    df_: float
    # ... mean(), std(), quantile(q), sample(n, rng) per Distribution Protocol
```

Both new classes implement the `Distribution` Protocol (`mean`, `std`, `quantile`, `sample`). Existing scoring layer (`scoring/score_distribution.py`) and aggregation layer (`aggregation/season.py`) consume the protocol; no changes needed in those modules.

### 8.2 `DistributionFamily` enum + codec

Add two enum values to `src/projections/schemas.py`:

```python
class DistributionFamily(StrEnum):
    NORMAL = "NORMAL"
    GAMMA = "GAMMA"
    NEGATIVE_BINOMIAL = "NEGATIVE_BINOMIAL"   # NEW (Phase 1)
    STUDENT_T = "STUDENT_T"                   # NEW (Phase 2)
    SAMPLED = "SAMPLED"
    SAMPLED_SUMMARY = "SAMPLED_SUMMARY"
```

Update `pack_per_stat_params` / `unpack_per_stat_params` in `src/projections/distributions/codec.py` with branches for each new family. Schema version bumps from 1 to 2 (codec rejects old blobs with a clear "unsupported schema_version" error so any old persisted projections fail loudly).

### 8.3 `BaselineModel` per-position factory updates

`src/projections/models/baseline.py` per-position factories rewire `dist_families`:

- **WR**: `RECEPTIONS` → GAMMA (unchanged, Phase 0 evidence borderline OK), `RECEIVING_YARDS` → STUDENT_T, `RECEIVING_TDS` → NEGATIVE_BINOMIAL, `RUSHING_YARDS` → STUDENT_T, `RUSHING_TDS` → NEGATIVE_BINOMIAL, `FUMBLES_LOST` → NEGATIVE_BINOMIAL.
- **QB**: `PASSING_YARDS` → keep NORMAL (Phase 0 essentially calibrated; use as regression reference), `PASSING_TDS` → NEGATIVE_BINOMIAL, `INTERCEPTIONS` → NEGATIVE_BINOMIAL, `RUSHING_YARDS` → STUDENT_T, `RUSHING_TDS` → NEGATIVE_BINOMIAL, `FUMBLES_LOST` → NEGATIVE_BINOMIAL.
- **RB**: `RUSHING_YARDS` → STUDENT_T, `RUSHING_TDS` → NEGATIVE_BINOMIAL, `RECEPTIONS` → GAMMA (unchanged), `RECEIVING_YARDS` → STUDENT_T, `RECEIVING_TDS` → NEGATIVE_BINOMIAL, `FUMBLES_LOST` → NEGATIVE_BINOMIAL.
- **TE**: `RECEPTIONS` → GAMMA (unchanged), `RECEIVING_YARDS` → STUDENT_T, `RECEIVING_TDS` → NEGATIVE_BINOMIAL, `RUSHING_YARDS` → STUDENT_T, `RUSHING_TDS` → NEGATIVE_BINOMIAL, `FUMBLES_LOST` → NEGATIVE_BINOMIAL.

Two cells are "soft" picks the per-phase review must reconfirm: (1) `WR rushing_yards` is rare (mean ~1 yard) and the Phase 0 alt-fitter returned `none`; if Phase 2's Student-t fit is degenerate, revert this cell to NORMAL in the Phase 2 wrap. (2) `TE rushing_yards` is degenerate (mean ~0.10 yards, kurtosis ~2618 per Phase 0); Phase 2 attempts STUDENT_T, but if the fit is degenerate or the post-Phase-2 coverage on this cell drops below its Phase 1 baseline, revert to NORMAL in the Phase 2 wrap. Document the choice in the per-phase PM doc update.

### 8.4 `BaselineModel.variance_params` generalization

Current shape: `dict[Stat, dict[str, float]]` — one parameter (`std` or `shape`) per stat.

Phase 1-2 shape: `dict[Stat, dict[str, float]]` extends to support new family parameters:
- NORMAL: `{"std": float}` (unchanged)
- GAMMA: `{"shape": float}` (unchanged)
- NEGATIVE_BINOMIAL: `{"dispersion": float}`
- STUDENT_T: `{"scale": float, "df": float}`

Phase 3 shape: `dict[Stat, dict[str, float | list[float]]]` — bucket boundaries + per-bucket parameter:
- e.g. NORMAL with bucketing: `{"bucket_cuts": [low_pred, high_pred], "std_per_bucket": [std_low, std_mid, std_high]}`
- GAMMA with bucketing: `{"bucket_cuts": [...], "shape_per_bucket": [a_low, a_mid, a_high]}`
- NB with bucketing: `{"bucket_cuts": [...], "dispersion_per_bucket": [d_low, d_mid, d_high]}`
- Student-t with bucketing: `{"bucket_cuts": [...], "scale_per_bucket": [...], "df_per_bucket": [...]}`

Codec extended in lockstep. `BaselineModel.predict_distribution` looks up which bucket each row falls into via `np.searchsorted(bucket_cuts, mu_hat)` and selects the corresponding parameter value.

---

## 9. Implementation notes (Phase 1+)

### 9.1 NegativeBinomial dispersion estimator

Per stat, fit dispersion by maximizing the log-likelihood `sum(nbinom.logpmf(actual_i, n_i, p_i))` over a single global dispersion parameter, where `n_i = mean_i^2 / dispersion` and `p_i = n_i / (n_i + mean_i)`, and `mean_i = max(pred_i, eps)` is the per-row Ridge prediction. `scipy.optimize.minimize_scalar` on a 1-D dispersion parameter; bracket `[1e-3, 1e3]`.

This is a CONDITIONAL fit (per-row mean from `pred`), unlike Phase 0's marginal `_fit_neg_binomial`. The conditional formulation directly fixes the marginal-vs-conditional AIC asymmetry caveat from Phase 0.

### 9.2 Student-t (scale, df) estimator

Per stat, fit scale + df by `scipy.stats.t.fit` on the residual array `actual - pred`. Returns `(df, loc, scale)`; we discard the fitted `loc` (always near zero on residuals) and persist `(scale, df)`. Apply the same sample-std floor guard from Phase 0's `_fit_student_t` to reject degenerate fits.

### 9.3 Per-tertile bucketing

Per (position, stat), compute the 33rd and 67th percentile of `mu_hat` from the training set. Persist these two cuts as `bucket_cuts: [c33, c67]`. Within each bucket (rows where `c33 < mu_hat <= c67` etc.), refit the variance parameter using the cell's existing single-parameter estimator (e.g., `_normal_std_from_residuals` for NORMAL, the dispersion fitter from §9.1 for NB). Persist as `*_per_bucket: [low_value, mid_value, high_value]`.

At predict time: `bucket_idx = np.searchsorted(bucket_cuts, mu_hat).clip(0, 2)`. Pick parameter value from `params_per_bucket[bucket_idx]`. Construct the per-row distribution with that parameter.

### 9.4 Phase boundaries

- **Phase 1 ends** with all `*_tds` / `*_interceptions` / `*_fumbles_lost` cells routed to NB, codec updated, models retrained, snapshot regenerated, PM doc updated. Stop here, run gate, verify min-cell coverage moved from 0.0 to something positive (target: ≥ 0.50 for the affected cells; final ≥ 0.65 awaits Phase 3).
- **Phase 2 ends** with all `*_yards` cells (except `QB passing_yards`) routed to Student-t, snapshot regenerated. Verify Student-t cells improved without regressing the NB cells from Phase 1.
- **Phase 3 ends** with bucketing applied to every (position, stat) cell whose `heteroscedasticity_ratio > 1.5` per Phase 0 (18 cells). Final snapshot. PM doc reports the cumulative coverage delta vs the Phase 0 baseline.

### 9.5 Backwards compatibility

- **Codec schema_version bump (1 → 2):** old blobs (Plan 3d artifacts) decode with a clear `ValueError("Unknown per-stat params schema_version: 1")`. There are no production consumers of those blobs other than the diagnostic CLI (which reads `data/backtest/run_<ts>/results.parquet` from a recent harness run, and the post-Phase-1+ harness writes v=2). No migration path.
- **`BaselineModel.variance_params` shape change:** `dict[Stat, dict[str, float]]` → `dict[Stat, dict[str, float | list[float]]]`. Saved `models/artifacts/*.joblib` from 3b/3c/3d become unloadable on the structural change. Same pattern as 3b → 3a; mechanical retrain in a Phase N task.
- **No backwards-compat shims.** Same posture as previous plans.

---

## 10. Validation surface (Phase 1+)

### 10.1 Per-phase

- **After each phase:** `pytest -v` clean; `mypy src tests scripts` clean; `ruff check` + `ruff format --check` clean.
- **Per-phase gate:** `python scripts/backtest.py --check` passes after re-snapshot. `pytest -m backtest --run-backtest` passes.
- **PM doc update per phase:** report the cells affected, the coverage delta vs the previous phase, the snapshot drift summary (max abs drift on calibration metrics, max rel drift on RMSE).

### 10.2 Plan 3e overall

(Unchanged from § 5.2.) Min-cell coverage ≥ 0.65; mean coverage improves by ≥ 0.10; no regression on RMSE/MAE/Spearman beyond existing tolerances; TODO #22 closed.

### 10.3 Out of scope (deferred to follow-up plans)

- Cross-week residual correlation (joint/copula territory) → follow-up plan if season coverage stays bad after Phase 3.
- Inflation-factor calibration → only if parametric tightening doesn't reach floor.
- Per-bucket family selection (single family per cell stays).
- Conditional alt-family AIC in the Phase 0 diagnostic (caveat documented in §4.2; not load-bearing for Phases 1-3).
- ZIP / Zero-Inflated NB → revisit if NB alone leaves count cells under-calibrated.
- More than 3 buckets → revisit if Phase 3's coverage gain is short of target.

---
