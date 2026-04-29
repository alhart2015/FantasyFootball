# Project Management

Running log of project status, decisions, and next steps. Append new entries at the top; keep the bottom as the long-tail backlog. Specs live in `docs/superpowers/specs/`, plans in `docs/superpowers/plans/`, single-task TODOs in `TODO.md`.

---

## Plan 8 — Adoption gate redesign — re-evaluation complete (2026-04-29, on branch `feat/plan-8-gate-redesign`)

**Status:** Phases 1–4 complete; Phases 5 (snapshot.py audit) + 6 (§1.3 spec template) + 7 (PR) remain. **Three production routing changes shipped** (per Phase 4 verdicts below).

### Diagnosis recap

Plans 3e / 5 / 5b / 5c / 7 / 6 all failed the prior §1.3 adoption gate. The streak decomposes into two compounding structural problems:

1. **§1.3 thresholds sit below the per-cell noise floor.** "Composite RMSE strictly lower on ≥12/16 cells AND not worse by >1% on any cell" + "weekly calibration no worse on any cell, mean delta ≥ +0.02" treat sampling variation as systematic regression. Smoking gun: Plan 6 hit 12/16 RMSE wins (meets the count!) but failed because TE 2024 was +1.24% worse — 0.24pp over the no-regression line, on a 1081-week cell. There is no significance test gating this; the noise floor on a single cell's RMSE is plausibly ≥1%.
2. **The calibration metric isn't load-bearing for any planned consumer.** Plan 5c PM and Plan 6 §96–99 already note this out loud — Draft Hub, start/sit, and the DFS lineup optimizer all consume mean and rank, not `[p10, p90]` coverage. Five plans of calibration work optimized a metric whose failure has no downstream cost. Plan 7's Phase 0 separately showed the assumed calibration mechanism was wrong (per-stat coverage doesn't decompose to composite coverage), so multiple plans were also pulling the wrong end of the distribution.

### What shipped (Phases 1–4)

- **Pure-stats module** at `src/projections/backtest/adoption_gate.py`: `BootstrapDelta`, `PositionVerdict`, `paired_bootstrap_rmse_delta`, `paired_bootstrap_spearman_delta`, `verdict_for_position` with the §1.3-replacement rule (RMSE: 95% CI strictly below 0; Spearman: lower CI > -0.02 catastrophic-regression floor; calibration informational, not gating).
- **CLI** at `scripts/adoption_gate.py`: reads any backtest run's per-row `results.parquet`, pairs rows on `(gsis_id, season, week)`, emits per-position adoption verdicts as markdown to stdout + optional CSV via `--csv-out`.
- **Per-position routing**: `_PositionDispatch.default_model_class` field with `__post_init__` validation; `production_model_for(position)` helper (single sanctioned entry point for "the production model for this position"); per-position defaults set per re-evaluation verdicts.
- **37 new tests** (Phase 1: 19 stats tests; Phase 2: 13 CLI tests; Phase 3: 5 routing tests). All pass; mypy + ruff + format clean across 145 source files.
- **Re-evaluation reports** committed under `reports/adoption_gate_*.{md,csv,stderr}` and `reports/adoption_gate_summary.md`.

### Phase 4 re-evaluation verdicts (run_20260429T003552Z)

Paired bootstrap, n_bootstrap=1000, seed=42. Pairing key `(gsis_id, season, week)`. ADOPT requires `rmse_delta` 95% CI strictly below 0 AND `spearman_delta` 95% CI strictly above -0.02.

| Position | Candidate      | Verdict       | RMSE delta (95% CI)             | Spearman delta (95% CI)        | n_paired |
|---|---|---|---|---|---:|
| QB | lightgbm       | DO_NOT_ADOPT  | -0.0233 ([-0.1239, +0.0758])    | +0.0155 ([+0.0010, +0.0296])   | 2676 |
| QB | lightgbm-tuned | **ADOPT**     | -0.1189 ([-0.2063, -0.0310])    | +0.0177 ([+0.0046, +0.0304])   | 2676 |
| QB | lightgbm-nb    | **ADOPT**     | **-0.1933 ([-0.2719, -0.1102])** | +0.0183 ([+0.0045, +0.0313])   | 2676 |
| QB | ensemble       | **ADOPT**     | -0.1760 ([-0.2274, -0.1242])    | +0.0184 ([+0.0098, +0.0262])   | 2676 |
| RB | lightgbm       | DO_NOT_ADOPT  | +0.1916 ([+0.1438, +0.2421])    | -0.0023 ([-0.0082, +0.0028])   | 5273 |
| RB | lightgbm-tuned | DO_NOT_ADOPT  | +0.1144 ([+0.0798, +0.1520])    | -0.0043 ([-0.0098, +0.0009])   | 5273 |
| RB | lightgbm-nb    | DO_NOT_ADOPT  | +0.0420 ([+0.0133, +0.0740])    | -0.0012 ([-0.0068, +0.0039])   | 5273 |
| RB | ensemble       | DO_NOT_ADOPT  | +0.0212 ([-0.0021, +0.0455])    | +0.0003 ([-0.0037, +0.0043])   | 5273 |
| TE | lightgbm       | DO_NOT_ADOPT  | +0.1553 ([+0.1096, +0.2060])    | +0.0043 ([-0.0052, +0.0132])   | 4257 |
| TE | lightgbm-tuned | DO_NOT_ADOPT  | +0.0879 ([+0.0468, +0.1322])    | +0.0082 ([-0.0003, +0.0170])   | 4257 |
| TE | lightgbm-nb    | DO_NOT_ADOPT  | +0.0028 ([-0.0289, +0.0422])    | +0.0071 ([-0.0014, +0.0160])   | 4257 |
| TE | ensemble       | DO_NOT_ADOPT  | -0.0208 ([-0.0454, +0.0097])    | +0.0076 ([+0.0016, +0.0137])   | 4257 |
| WR | lightgbm       | DO_NOT_ADOPT  | +0.1338 ([+0.0963, +0.1721])    | +0.0045 ([-0.0012, +0.0101])   | 8460 |
| WR | lightgbm-tuned | DO_NOT_ADOPT  | +0.0711 ([+0.0397, +0.1046])    | +0.0044 ([-0.0012, +0.0099])   | 8460 |
| WR | lightgbm-nb    | DO_NOT_ADOPT  | -0.0016 ([-0.0316, +0.0291])    | +0.0027 ([-0.0032, +0.0080])   | 8460 |
| WR | ensemble       | **ADOPT**     | -0.0320 ([-0.0531, -0.0092])    | +0.0069 ([+0.0028, +0.0109])   | 8460 |

### Per-position routing changes shipped

| Position | Pre-Plan-8 default | Post-Plan-8 default | Reason                           |
|----------|--------------------|---------------------|----------------------------------|
| QB       | baseline           | **lightgbm-nb**     | 3 ADOPTers; mechanical tie-break (most-negative rmse_delta.point) selects lightgbm-nb (-0.1933) over ensemble (-0.1760) and tuned (-0.1189). |
| RB       | baseline           | baseline            | No ADOPT verdict; every candidate regresses RB or has rank-correlation issues. |
| TE       | baseline           | baseline            | No ADOPT verdict; ensemble is closest (rmse_delta -0.021) but CI brackets zero. |
| WR       | baseline           | **ensemble**        | Sole ADOPTer; rmse_delta -0.032 fpts, both CIs strictly clear zero. n=8460 paired rows give it the statistical power. |

### Surprises vs spec §6 strong prior

Two findings deviate from the spec's prediction:

1. **QB winner is `lightgbm-nb`, not `ensemble`.** The spec expected ensemble to win QB. Reality: NB beats ensemble's RMSE point estimate by ~0.017 fpts; CIs overlap heavily but mechanical tie-break selects NB. Side benefit: NB is structurally simpler than ensemble (no MixtureDistribution / per-stat weight optimizer / 4-stage fit) — simpler is better when stat-equivalent.
2. **WR ADOPTs `ensemble`.** The spec said "WR's improvements were ≤0.55% per cell, pooled CI may or may not clear zero." It cleared. Larger sample (n=8460) gives the bootstrap enough power for a small-but-clean win.

### Snapshot regression gate audit (Phase 5)

`tests/backtest/tolerances.json` defaults vs measured per-cell noise floor (from Plan 8's bootstrap reports under `reports/adoption_gate_*.csv`, per-year breakdown rows):

| Metric kind            | snapshot default | Measured per-cell RMSE-delta CI half-width (absolute, fpts) | Translated to relative (÷ typical per-cell RMSE ≈ 6 fpts) | Verdict |
|---|---|---|---|---|
| `rmse_relative`        | 5.0%   | median 0.076; p75 0.100; max 0.217 | median 1.26%; p75 1.66%; max 3.61% | **Fine — ~3-4× headroom over the median noise floor** |
| `mae_relative`         | 5.0%   | (same scale as rmse) | (same) | **Fine** |
| `spearman_absolute`    | 0.02   | (Spearman per-year deltas have CI half-widths ~0.005–0.015) | n/a | **Fine — matches Plan 8's catastrophic-regression floor** |
| `calibration_absolute` | 0.03   | (calibration is informational under Plan 8; not load-bearing for adoption) | n/a | **Fine for code-regression purposes** |
| `mean_pred_relative`   | 10.0%  | (broader window, intentionally — guards against unintended mean shifts) | n/a | **Fine** |

**Conclusion**: snapshot.py's tolerances are above the per-cell noise floor with comfortable headroom; the regression gate is doing the right job (catching real code-induced numeric drift, not flagging sampling noise). No changes needed; no follow-up TODO filed.

The snapshot regression gate (catches code regression on a frozen model) and the new adoption gate (decides which class is the production default) answer different questions and stay independent. Both ship as-is.

### Remaining work

- **Phase 6**: Create `docs/superpowers/specs/_adoption_gate_template.md` for future model-class specs to inline-copy.
- **Phase 7**: Final pytest + mypy + ruff sweep; PR.

### Next track after Plan 8

Feature-class work starting with **TODO #3 (PBP / EPA features)**. Five model-class swaps on identical features hit the same information ceiling; the next real RMSE lift (estimated 5–15%) lives in features, not in model class.

---

## Plan 6 — Model D ensemble (A + C-NB) — shipped as peer (run 2026-04-29)

**Verdict:** ship as peer. Per-(position, stat) calibration-aware weighted mixture of Model A and Model C-NB landed cleanly with the per-stat pinball optimizer behaving exactly as designed — yards stats heavily favor C-NB's tight QuantileDistribution; TD stats favor A's wider parametric distributions. **All three §1.3 adoption criteria failed** by narrow margins. The per-position split that motivated the plan is preserved (QB cells improve on every metric; RB/TE/WR cells regress on calibration), confirming the mechanism but also confirming Plan 7's lesson — per-stat coverage at [p10, p90] does not algebraically decompose to composite [p10, p90] coverage.

`EnsembleModel` (Model D) lands as a fifth peer of Models A, C, C-tuned, C-NB. Wraps a `BaselineModel` + `LightGBMNbModel` pair via `_EnsembleConfig` factories. The 4-stage `fit()` per spec §3.1 trains weight-fit children on `[S, Y-2]`, predicts the calibration year `Y-1`, fits per-(position, stat) weights via `scipy.optimize.minimize_scalar` on summed pinball loss at q ∈ {0.10, 0.90}, then re-fits prediction children on the full `[S, Y-1]` span. `MixtureDistribution` (new in `src/projections/distributions/mixture.py`) implements the `Distribution` Protocol structurally — pure CDF-pool composition with brentq-based quantile inversion. New codec MIXTURE branch persists `{family, weight, component_a, component_b}` recursively; `schema_version` bumps 1 → 2.

Snapshot extended 1504 → 1872 rows (368 new ensemble rows: 23 metrics × 4 positions × 4 years). 16 weight artifacts at `data/ensemble_weights/{model_id}.json` (4 positions × 4 folds, filename sanitizes `:` → `_` for NTFS). The 32 `season_calibration_*` rows still skip via the existing `SAMPLED_SUMMARY`-only family gate; TODO #28 stays open.

### Per-position model_ids (final fold, train 2018-2023, predict 2024)

| Position | Model A | Model C-NB | Model D Ensemble |
|---|---|---|---|
| QB | `baseline:qb:5e8fe380:2018-2023` | `lightgbm-nb:qb:4f40329c:2018-2023` | `ensemble:qb:3494f28a:2018-2023` |
| RB | `baseline:rb:...:2018-2023` | `lightgbm-nb:rb:...:2018-2023` | `ensemble:rb:9dec620c:2018-2023` |
| TE | `baseline:te:...:2018-2023` | `lightgbm-nb:te:...:2018-2023` | `ensemble:te:da0287a2:2018-2023` |
| WR | `baseline:wr:730abe91:2018-2023` | `lightgbm-nb:wr:b751ce19:2018-2023` | `ensemble:wr:6f075552:2018-2023` |

(Full model_ids per fold are in the `data/ensemble_weights/*.json` artifacts.)

### Adoption-gate verdict — DO NOT ADOPT Model D as default

Spec §1.3 required Model D to beat Model A on three criteria. **All three failed.**

| Criterion | Threshold | Actual (D vs A) | Pass? |
|---|---|---|---|
| Composite RMSE strictly lower on >=12/16 cells; max +1% worse | D < A on 12+; max +1% worse | D strictly lower on **12/16** (meets count); max +1.24% on TE 2024 (exceeds +1.0%) | **FAIL** (margin) |
| Spearman top-N within ±0.005 on every cell | All within ±0.005 | 4/16 outside ±0.005; max abs delta +0.0131 (QB 2021, **a +0.0131 IMPROVEMENT**) | **FAIL** |
| Calibration no worse on any cell; mean delta >= +0.02 | No regressions; mean ≥ +0.02 | D worse on 13/16; mean delta -0.058 | **FAIL** |

The Spearman criterion's "within ±0.005" is symmetric — it fails ensembles that improve rank ordering by >0.005, not just those that regress it. All 4 of Plan 6's Spearman violations are positive deltas (rank ordering *improves* on those cells). For purposes of "does D beat A on rank?" Plan 6 ties or wins on every cell except RB 2021 (delta -0.0055).

### Side-by-side per-cell comparison (16 cells)

`RMSE D-A %` is the percentage delta on composite RMSE (negative = D wins; threshold +1.00%). Spearman / Calib columns show D's value and the (D − A) delta.

| Cell | RMSE A | RMSE D | RMSE D-A % | Spearman A | Spearman D | Spearman D-A | Calib A | Calib D | Calib D-A |
|---|---|---|---|---|---|---|---|---|---|
| QB 2021 | 7.8342 | 7.6396 | -2.49% | 0.9342 | 0.9473 | +0.0131 | 0.6947 | 0.7143 | +0.0196 |
| QB 2022 | 7.2261 | 7.0432 | -2.53% | 0.9669 | 0.9667 | -0.0002 | 0.7458 | 0.7808 | +0.0350 |
| QB 2023 | 7.3092 | 7.1780 | -1.80% | 0.9454 | 0.9570 | +0.0116 | 0.7313 | 0.7299 | -0.0014 |
| QB 2024 | 7.6995 | 7.5061 | -2.51% | 0.9383 | 0.9437 | +0.0054 | 0.7018 | 0.7222 | +0.0204 |
| RB 2021 | 6.8486 | 6.8948 | +0.67% | 0.9700 | 0.9645 | -0.0055 | 0.7475 | 0.6563 | -0.0912 |
| RB 2022 | 6.6359 | 6.6197 | -0.24% | 0.9658 | 0.9680 | +0.0022 | 0.7415 | 0.6536 | -0.0879 |
| RB 2023 | 6.3143 | 6.3485 | +0.54% | 0.9665 | 0.9657 | -0.0008 | 0.7872 | 0.6789 | -0.1083 |
| RB 2024 | 6.4860 | 6.5070 | +0.32% | 0.9753 | 0.9762 | +0.0009 | 0.7568 | 0.6573 | -0.0995 |
| TE 2021 | 5.3365 | 5.2750 | -1.15% | 0.9655 | 0.9659 | +0.0004 | 0.7350 | 0.6748 | -0.0602 |
| TE 2022 | 5.2498 | 5.2024 | -0.90% | 0.9615 | 0.9642 | +0.0027 | 0.7647 | 0.6590 | -0.1057 |
| TE 2023 | 4.9422 | 4.9041 | -0.77% | 0.9704 | 0.9751 | +0.0047 | 0.7561 | 0.6767 | -0.0794 |
| TE 2024 | 5.0804 | 5.1435 | +1.24% | 0.9620 | 0.9593 | -0.0027 | 0.7345 | 0.6605 | -0.0740 |
| WR 2021 | 6.7333 | 6.6966 | -0.55% | 0.9699 | 0.9699 | +0.0000 | 0.6956 | 0.6339 | -0.0617 |
| WR 2022 | 6.6255 | 6.5910 | -0.52% | 0.9767 | 0.9754 | -0.0013 | 0.6970 | 0.6275 | -0.0695 |
| WR 2023 | 6.5159 | 6.4920 | -0.37% | 0.9680 | 0.9671 | -0.0009 | 0.7256 | 0.6415 | -0.0841 |
| WR 2024 | 6.6728 | 6.6398 | -0.49% | 0.9739 | 0.9721 | -0.0018 | 0.7109 | 0.6309 | -0.0800 |

### Per-position split — QB clean win on every metric; RB/TE/WR RMSE wins paired with calibration regressions

| Position | RMSE wins vs A | Mean Spearman delta | Mean calib delta vs A |
|---|---|---|---|
| QB | 4/4 | +0.0075 | **+0.0184** (positive) |
| RB | 1/4 | -0.0008 | -0.0968 |
| TE | 3/4 | +0.0013 | -0.0798 |
| WR | 4/4 | -0.0010 | -0.0738 |

**QB is the only position where Model D cleanly beats A on every metric on every fold.** RMSE -1.8% to -2.5% across 4/4 years; calibration mean +0.018 (3 of 4 years positive); Spearman gains in 3 of 4 years. Per the final-fold weight vector, the QB optimizer pulls passing_yards (0.20) and rushing_yards (0.12) heavily toward C-NB while leaving TDs and interceptions near-balanced — exactly the direction QB-specific gains in Plan 5c suggested.

**RB / TE / WR show the same pattern across the board:** RMSE improves on most cells (RB 1/4, TE 3/4, WR 4/4 wins), but [p10, p90] calibration regresses 6-11 percentage points. The mechanism is the same one Plan 5c diagnosed and Plan 7's Phase 0 confirmed empirically: NB-2 dispersion fitted on training residuals produces tight predictive intervals that don't survive held-out variance on RB/TE/WR; the per-stat pinball optimizer correctly identifies that yards distributions should pull heavily toward C-NB (where per-stat coverage is good), but the convolution into composite fantasy points doesn't preserve [p10, p90] coverage.

### Per-stat fitted weights — final fold

Across all 4 positions, the optimizer learned a clean per-stat pattern:

- **Yards stats** (passing / rushing / receiving): w_a ∈ [0.001, 0.20] — heavily C-NB.
- **TD stats** (passing / rushing / receiving TDs): w_a ∈ [0.61, 0.77] — moderately A.
- **Other counts** (interceptions / receptions / fumbles_lost): mixed, position-dependent.

This validates the design hypothesis from spec §1.1: A's wider parametric distributions help TD calibration; C-NB's tight QuantileDistribution distributions match yards p10/p90 well. What the design did NOT predict is that this per-stat optimum would not propagate to composite calibration on RB/TE/WR.

### Why this should work / does it work

Spec §1.1's mechanism hypothesis was correct in isolation (per-stat). The mixture variance formula `w·var_A + (1-w)·var_B + w(1-w)(mean_A − mean_B)²` does widen calibration intervals when component means differ. The pinball optimizer correctly identifies the per-stat optimum — visible in the clean yards-vs-TDs split.

**The composite [p10, p90] coverage problem is upstream of any per-stat fix.** Plan 7's diagnostic established that per-stat coverage at the central interval (p10/p90) doesn't decompose to composite coverage at the central interval — composite p10/p90 width is dominated by yards (weight ~6-8 fp per 100 yards), composite tail weight by counts (TD weight × 6 = single-row 6-18 fp jumps). When ensemble narrows yards (good for yards p10/p90) and widens TDs (good for TD p10/p90), the composite [p10, p90] band tightens around yards width but the composite tail behavior shifts in a way that increases the rate of actuals falling outside composite [p10, p90].

**TODO #30 follow-up #1 (composite-direct optimization via Monte Carlo) is the right next experiment if calibration is the priority.** Plan 6 confirms what Plan 7's diagnostic predicted: any per-stat-decoupled fix is fundamentally limited.

### Decision

**Default model selection:** Model A stays the production default. Models C, C-tuned, C-NB, and D all ship as peers; none is adopted. **Model D's QB cells beat Model A on every metric — if the project ever adopts a per-position default selection, the QB row of `POSITION_DISPATCH` could route through Model D while leaving RB/TE/WR routed through A.** Not implemented in this plan; flagged as a future routing experiment.

**Pivot:** The next track is determined by what we want from the modeling stack:
1. **Calibration priority** → composite-direct weight optimization via MC (TODO #30 follow-up #1). Same EnsembleModel infrastructure, replace pinball-on-per-stat with composite-Brier-on-MC. ~5-10x slower per fold; risks but might break the per-stat-vs-composite barrier.
2. **Mean-prediction priority** → feature-class tracks (TODO #3 PBP/EPA, TODO #23 target decomposition). Estimated 5-15% RMSE win on top of any model class. Independent of model class.
3. **Pivot to consumer tools** → Plan 4 (public Python API + CLI verbs). Modeling has reached "good enough" for downstream consumers; all four planned tools (Draft Hub, start/sit, DFS) consume mean and rank, not [p10, p90] coverage.

Pick one in the next session.

### Per-position model_ids on disk

Standalone artifacts at `data/ensemble_weights/ensemble_{pos}_{8hex}_{S}-{E}.json` (filename sanitizes `:` → `_` for NTFS). The joblib pickle at `models/artifacts/ensemble-{pos}-...joblib` is only created on `scripts/train_baseline.py`-style invocations; the backtest harness regenerates per-fold artifacts in-memory and does not write standalone files.

### Operational notes

- Backtest run: 2026-04-29, ~5h45m wall-clock for the full `--model all` regeneration on real data (5 model classes × 4 positions × 4 folds). Ensemble's 4-child-per-fold fit + per-stat pinball optimizer is the bottleneck; weight optimization alone is ~3h of the 5h45m total.
- Determinism re-check (`--check` after `--update-snapshot`) is **deferred** — re-running takes 5+ hours wall-clock. Plan 6 ships the snapshot from this single run; future re-runs (e.g., a touch on `ensemble.py`) should re-validate determinism before merging the resulting snapshot.
- Test runtime cost: `tests/test_models/test_ensemble_model_smoke.py` 4 new fit-based tests added ~14 min in CI. Phase 6 keeps the existing `@pytest.mark.backtest` gating; the quint-model smoke at `tests/test_backtest/test_harness_quint_model.py` is gated on real-data caches and does NOT run in lightweight CI.

---

## Plan 7 — Calibration-aware NB-2 fitting (Model C-NB-cal) — STOPPED at Phase 0 (2026-04-28)

**Verdict:** stop the plan. Spec premise was misaligned with empirical reality. Phase 0 ships as research output; Phase 1+ unexecuted. Branch `feat/plan-7-calibration-aware-nb` proposed for merge with just the diagnostic CLI + spec + plan + research note (record-of-decision). Filed TODO #30 for the right follow-up plan.

### What happened

Plan 7's spec assumed Plan 5c's "NB-2 distribution too narrow at the [p10, p90] tails" claim mapped directly to per-stat NB-2 distributions being too narrow at p10/p90. The Phase 0 diagnostic measured per-stat empirical [p10, p90] coverage on Plan 5c's C-NB output and showed the opposite: count NB-2 distributions are *over*-covering at [p10, p90] by ~16pp (mean gap **-0.169**; range -0.188 to -0.154 across all 16 cells). Yards distributions are well-calibrated (mean gap **+0.011**).

Pinball-loss fitting at q=0.10 / q=0.90 — Plan 7's exact mechanism — would tighten count distributions toward 0.80 nominal, which is the opposite direction needed to close the composite [p10, p90] coverage gap (-0.062 mean vs A). The composite gap mechanism lives in **upper-tail behavior beyond p90** — outside what Plan 7's loss function targets.

### Why per-stat over-coverage coexists with composite under-coverage

For low-mean count NB-2 (μ ≈ 0.4 typical for RB receiving_tds), discrete support concentrates ~95% of mass at {0, 1}, so the predicted [p10, p90] = [0, 1] trivially over-covers at the 80% nominal. The thin upper tail (P(X≥2) ≈ 0.05 model vs ~7-10% empirical) is what Plan 5c described as "too narrow" — the wording is imprecise; the narrowness is at p95-p99, not at p10/p90. Composite [p10, p90] under-coverage comes from upper-tail count outliers (TD weight × 6 = 12-18 fp jumps) exceeding composite p90 set by yards width.

Per-stat coverage at [p10, p90] does NOT decompose to composite coverage at [p10, p90] — convolution behavior at the central interval is dominated by the wider distribution's mean shift, not by the narrower distribution's own [p10, p90] coverage.

### Diagnostic CLI ships as reusable research output

`scripts/diagnose_calibration_breakdown.py` reads any per-row backtest parquet, filters by `model_id` prefix, and emits a per-(position, year) CSV decomposing per-stat coverage. It can be re-pointed at any future model-class output. 8 unit tests; mypy / ruff clean.

### What was missed at scoping

1. Trusted Plan 5c's mechanism statement without empirically checking it. A two-line python check at scoping would have caught the per-stat-over-coverage finding.
2. Per-stat coverage doesn't decompose composite coverage (convolution effect). The right diagnostic was counterfactual replacement (swap count distributions for A's; re-sample composite; measure closure). Shipped diagnostic measures a related-but-not-identical quantity that happens to surface the right verdict.
3. Missed the discreteness math: at μ ≈ 0.4, NB-2 mass at {0,1} trivially over-covers any [p10, p90] band at 80% nominal. Mechanical, catchable in 5 minutes.

Cost of getting it wrong: ~30 min conversation + 10 min compute. Cheap vs. building Phase 1 + Phase 2 and discovering empirically.

### Per-cell breakdown

Pasted into the research note: `docs/superpowers/research/2026-04-28-calibration-breakdown.md`.

### Next action

The composite calibration shortfall remains the binding constraint. Three candidate follow-up tracks (TODO #30 captures all three):

1. **Pinball-loss dispersion fit at upper-tail quantiles** (q=0.90, q=0.95 or q=0.95 only). Same machinery as Plan 7; right mechanism for the actual gap location.
2. **ZIP (zero-inflated Poisson) for count cells.** Handles zero mass + thin tail decoupled rather than via NB-2's single overdispersion knob. Fundamental family change.
3. **Mixture model: point mass at 0 + heavier-tailed integer distribution.** Most flexible, most code; defer until 1 and 2 are tried.

Or: accept the calibration shortfall as a known limitation and pivot to feature-class tracks (TODO #3, TODO #23) or Plan 6 (ensemble) instead. None of the planned downstream consumers (Draft Hub, start/sit, DFS) actually depend on a perfectly calibrated [p10, p90] — they consume mean and rank. Plan 5c PM already framed this as acceptable.

---

## Plan 5c — Hybrid LightGBM with NB-2 for Count Stats (Model C-NB) — shipped (run 2026-04-28)

**Closes:** the count-stat-bias mechanism identified in Plan 5b's diagnostic. Model C-NB strictly dominates Model C-tuned on RMSE (16/16 cells better) but still fails Model A's adoption gate — the gap moved from "tuning regression" to "calibration regression."

`LightGBMNbModel` (Model C-NB) lands as a fourth peer of Models A, C, C-tuned. Subclasses `LightGBMTunedModel`; for the 13 count cells Plan 3e routes through NB-2 in Ridge (`passing_tds`/`rushing_tds`/`receiving_tds`/`interceptions`/`fumbles_lost` × per-position target_stats), trains one `lgb.LGBMRegressor(objective="poisson")` per stat, fits NB-2 dispersion via the public `nb_dispersion_from_residuals` (relocated from `models/baseline.py` to `distributions/parametric.py` in Phase 0), and predicts via `ParametricNegativeBinomial(mu, dispersion)`. Yards/receptions stats unchanged from Model C-tuned (5-quantile + `QuantileDistribution`). Reuses Plan 5b's tuned hyperparameters from `data/tuned_params/lightgbm.json`. Per-row family is `MIXED`; per-stat families remain encoded inside the params blob via the existing codec.

Snapshot extended 1136 → 1504 rows (368 new lightgbm-nb rows; the 32 `season_calibration_*` rows are skipped by the SAMPLED_SUMMARY-only family gate, TODO #28 still open).

### Per-position model_ids

| Position | Model A | Model C | Model C-tuned | Model C-NB |
|---|---|---|---|---|
| WR | `baseline:wr:6d955427:2018-2023` | `lightgbm:wr:a4dd5a82:2018-2023` | `lightgbm-tuned:wr:62df14ad:2018-2023` | `lightgbm-nb:wr:dc445a2d:2018-2023` |
| QB | `baseline:qb:c98738f3:2018-2023` | `lightgbm:qb:06fadb3f:2018-2023` | `lightgbm-tuned:qb:fc902ed6:2018-2023` | `lightgbm-nb:qb:3ae5b940:2018-2023` |
| RB | `baseline:rb:5a86c8ee:2018-2023` | `lightgbm:rb:fb169c0e:2018-2023` | `lightgbm-tuned:rb:5d69fdfe:2018-2023` | `lightgbm-nb:rb:ba2e35cc:2018-2023` |
| TE | `baseline:te:9c00025b:2018-2023` | `lightgbm:te:bd4c2a5b:2018-2023` | `lightgbm-tuned:te:89dafdb6:2018-2023` | `lightgbm-nb:te:e76e590a:2018-2023` |

### Adoption-gate verdict — DO NOT ADOPT Model C-NB as default

Spec §1.3 required Model C-NB to beat Model A on three criteria. **All three failed.**

| Criterion | Threshold | Actual (C-NB vs A) | Pass? |
|---|---|---|---|
| Composite RMSE strictly lower on >=12/16 cells; max +1% worse | C-NB < A on 12+; max +1% worse | C-NB strictly lower on **11/16**; max +1.69% worse (TE 2024); 4/16 cells exceed 1% | **FAIL** |
| Spearman top-N within +-0.005 on every cell | All within ±0.005 | 4/16 outside ±0.005; max abs delta 0.0204 (QB 2021, **a +0.0204 IMPROVEMENT**) | **FAIL** |
| Calibration no worse on any cell; mean delta >= +0.02 | No regressions; mean ≥ +0.02 | C-NB worse on 13/16; mean delta -0.0617 | **FAIL** |

### Side-by-side per-cell comparison (16 cells × 4 metrics × 4 models)

`RMSE Cnb-A %` is the percentage delta on composite RMSE (negative = NB wins; threshold +1.00%). Spearman / Calib columns show NB's value and the (NB − A) delta.

| Cell | RMSE A | RMSE C | RMSE Ctuned | RMSE Cnb | RMSE Cnb-A % | Spearman A | Spearman Cnb | Spearman Cnb-A | Calib A | Calib Cnb | Calib Cnb-A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QB 2021 | 7.8342 | 7.8386 | 7.7320 | 7.5651 | -3.43% | 0.9342 | 0.9546 | +0.0204 | 0.6947 | 0.7173 | +0.0226 |
| QB 2022 | 7.2261 | 7.2718 | 7.1220 | 7.0637 | -2.25% | 0.9669 | 0.9672 | +0.0003 | 0.7458 | 0.7686 | +0.0228 |
| QB 2023 | 7.3092 | 7.3213 | 7.2602 | 7.2152 | -1.29% | 0.9454 | 0.9569 | +0.0115 | 0.7313 | 0.7194 | -0.0119 |
| QB 2024 | 7.6995 | 7.5523 | 7.4845 | 7.4597 | -3.12% | 0.9383 | 0.9452 | +0.0069 | 0.7018 | 0.7178 | +0.0161 |
| RB 2021 | 6.8486 | 7.0688 | 6.9895 | 6.9292 | +1.18% | 0.9700 | 0.9623 | -0.0077 | 0.7475 | 0.6327 | -0.1148 |
| RB 2022 | 6.6359 | 6.8370 | 6.7032 | 6.6331 | -0.04% | 0.9658 | 0.9692 | +0.0034 | 0.7415 | 0.6446 | -0.0969 |
| RB 2023 | 6.3143 | 6.5069 | 6.4255 | 6.3744 | +0.95% | 0.9665 | 0.9640 | -0.0025 | 0.7872 | 0.6751 | -0.1121 |
| RB 2024 | 6.4860 | 6.6370 | 6.6242 | 6.5157 | +0.46% | 0.9753 | 0.9763 | +0.0010 | 0.7568 | 0.6512 | -0.1056 |
| TE 2021 | 5.3365 | 5.4636 | 5.3835 | 5.3137 | -0.43% | 0.9655 | 0.9640 | -0.0015 | 0.7350 | 0.6602 | -0.0748 |
| TE 2022 | 5.2498 | 5.3710 | 5.3197 | 5.2092 | -0.78% | 0.9615 | 0.9629 | +0.0015 | 0.7647 | 0.6710 | -0.0938 |
| TE 2023 | 4.9422 | 5.1324 | 5.0272 | 4.9314 | -0.22% | 0.9704 | 0.9742 | +0.0038 | 0.7561 | 0.6805 | -0.0756 |
| TE 2024 | 5.0804 | 5.2661 | 5.2302 | 5.1662 | +1.69% | 0.9620 | 0.9580 | -0.0040 | 0.7345 | 0.6586 | -0.0759 |
| WR 2021 | 6.7333 | 6.8837 | 6.8025 | 6.7244 | -0.13% | 0.9699 | 0.9679 | -0.0020 | 0.6956 | 0.6354 | -0.0602 |
| WR 2022 | 6.6255 | 6.7479 | 6.7149 | 6.6189 | -0.10% | 0.9767 | 0.9729 | -0.0038 | 0.6970 | 0.6256 | -0.0714 |
| WR 2023 | 6.5159 | 6.7129 | 6.5922 | 6.5380 | +0.34% | 0.9680 | 0.9645 | -0.0035 | 0.7256 | 0.6447 | -0.0809 |
| WR 2024 | 6.6728 | 6.7339 | 6.7218 | 6.6589 | -0.21% | 0.9739 | 0.9709 | -0.0030 | 0.7109 | 0.6362 | -0.0747 |

### Aggregate movement: NB strictly dominates Tuned on RMSE; calibration unchanged in aggregate

| Metric | Tuned vs A | NB vs A | NB vs Tuned |
|---|---|---|---|
| RMSE: cells where C-* beats A | 4/16 | 11/16 | 16/16 strict dominance |
| RMSE: max pct worse vs A | +2.95% | +1.69% | — |
| Spearman: cells outside ±0.005 vs A | 7/16 | 4/16 | improved |
| Spearman: max abs delta vs A | 0.0163 | 0.0204 (a +0.0204 *gain* on QB 2021) | — |
| Calibration: cells where C-* worse than A | 12/16 | 13/16 | NB on average +0.0013 vs Tuned |
| Calibration: mean delta vs A | -0.0630 | -0.0617 | +0.0013 (essentially unchanged) |

**NB strictly dominates Tuned on RMSE on every cell.** Replacing the 5-knot quantile prediction for count stats with a poisson-objective regressor + NB-2 dispersion eliminated the count-stat over-prediction Plan 5b diagnosed. The mean RMSE pct vs A moved from "Tuned regresses on 12/16 cells" to "NB beats A on 11/16 cells." But that improvement does not propagate to calibration — the mean p10/p90 coverage delta vs A stayed essentially flat (-0.0630 → -0.0617).

### Per-position split — QB clean win; RB/TE/WR are RMSE wins paired with calibration regressions

| Position | RMSE wins vs A | Mean calib delta vs A |
|---|---|---|
| QB | 4/4 | **+0.0124 (positive)** |
| RB | 1/4 | -0.1074 |
| TE | 3/4 | -0.0800 |
| WR | 3/4 | -0.0718 |

**QB is the only position where C-NB cleanly beats A on every metric:** RMSE -1.3% to -3.4% across 4/4 years, calibration +0.012 mean (3 of 4 years positive), Spearman +0 to +0.02. QB's count-stat distributions (passing_tds with mean ~1.5, interceptions ~0.7) are the exact zero-inflated count stats NB-2 was designed for, and the per-row mean is the dominant signal in QB scoring (rushing yards and passing yards together account for ~85% of fantasy points; passing TDs are the next ~10%).

**RB / TE / WR show the same pattern across the board:** RMSE improves (RB 1/4 wins, TE 3/4, WR 3/4 — close to a 50/50 split), but [p10, p90] calibration regresses 6-12 percentage points. The mechanism: NB-2 dispersion fitted on training residuals via conditional MLE produces tight predictive intervals when the per-row mean is well-fit (which the poisson booster does), but the residual variance on test data — particularly RB/TE/WR which have higher target variance and more regime drift between seasons — exceeds the training-fit dispersion. The fitted NB-2 distribution is therefore too narrow at the [p10, p90] tails on held-out years.

A practical illustration: RB 2024 has C-NB calib 0.6512 vs A 0.7568 — a 10 pp coverage drop. The NB-2 receiving_tds distribution for an average-volume RB with mu_hat ≈ 0.4 and fitted dispersion ≈ 5 has [p10, p90] of [0, 1] (NB-2 mode at 0; long right tail). When test-set actuals scatter to 2-3 TDs (a realistic RB game), they fall outside p90. The booster predicts the mean correctly; the dispersion under-estimates the heavy right tail.

### Why this should work / does it work

The Plan 5b diagnostic identified the mechanism: 5-knot QuantileDistribution linear interpolation + sort + clip produces a biased empirical mean on zero-inflated count stats (over-prediction of 30-60% on TE/WR receiving_tds, QB rushing_tds/fumbles, etc.). NB-2 with mean = mu_hat and dispersion fit on training residuals does not have this bias — the empirical mean of NB-2 samples ≈ mu_hat by construction.

**The mean-prediction fix worked.** NB strictly dominates Tuned on RMSE on every cell — closing the entire mean-prediction gap Plan 5 / 5b's quantile sub-models couldn't. RMSE moved from "C-tuned 4/16 wins, max +2.95% worse" to "C-NB 11/16 wins, max +1.69% worse" — a step closer to the §1.3 threshold but still short of 12/16.

**The calibration regression that Plan 5 / 5b had against A on the [p10, p90] interval did NOT close.** NB-2's narrow predictive interval at low mu trades RMSE for coverage. Plan 3e Phase 1 saw the same shape in Ridge (NB cells nudged calibration in the right direction in aggregate but did not solve the problem). Replacing GAMMA with NB-2 in Ridge, or quantile regression with NB-2 in LightGBM, both improve mean-prediction RMSE without solving the underlying coverage problem because the underlying residual variance on held-out years exceeds what a well-fit conditional distribution can represent without overfitting the noise floor.

Yards stats are unchanged from Model C-tuned (the test `test_yards_stat_predictions_match_tuned_baseline` pins yards-stat best_iters to be bit-exact identical between the two models) — the calibration regression on RB/TE/WR is fully attributable to the count-stat NB-2 path, not to anything in the inherited yards path.

### Decision

**Default model selection:** Model A stays the production default. Models C, C-tuned, and C-NB all ship as peers; none is adopted. Model C-NB strictly dominates Model C-tuned on RMSE — Model C-tuned is now arguably prunable (TODO followup).

**Pivot:** the next model-improvement track stays one of the three remaining options. With C-NB now showing what mean-prediction fixes alone can do, the calibration gap is the unambiguous binding constraint. Three remaining tracks:

- **Plan 6 — Model D ensemble.** Stack of (Model A, Model C-NB) per (position, stat) with calibration-aware weighting. Cheapest given Plan 5c's infrastructure. Covers the case where C-NB's mean-prediction wins on QB and Model A's calibration wins on RB/TE/WR.
- **Calibration-aware NB fitting.** Fit NB-2 dispersion to optimize p10/p90 coverage directly (quantile loss) rather than likelihood. Preserves C-NB's RMSE wins; targets the calibration regression directly. Risk: overfit to validation noise.
- **TODO #3 (PBP / EPA features)** + **TODO #23 (target decomposition)** — feature-class tracks. Independent of model class. Estimated 5-15% RMSE win on top of any model class.

Pick one in the next session. Plan 4 (public Python API + CLI verbs + free-tier hosting) remains the post-modeling milestone.

### Per-position model_ids on disk

Standalone artifacts at `models/artifacts/lightgbm-nb-{pos}-2018-2023-{hash}.joblib` (only created on `scripts/train_baseline.py`-style invocations; the backtest harness regenerates per-fold artifacts from the feature cache and does not write standalone files).

---

## Plan 5b — Optuna Tuning of Model C (Model C-tuned) — shipped (run 2026-04-28)

**Closes:** TODO #26 follow-up "if tuning closes the gap, revisit adoption."

`LightGBMTunedModel` (Model C-tuned) lands as a third peer of `BaselineModel` (Model A) and `LightGBMModel` (Model C) under the existing `Model` Protocol. Subclass overrides only `_hyperparams_for(stat)`, `code_hash`, and `model_id`; all training and prediction logic is inherited from `LightGBMModel`. Tuned hyperparameters live in `data/tuned_params/lightgbm.json` (checked in, dense across all 24 (position, stat) entries, content-hashed into `model_id`). 24 per-(position, stat) Optuna studies × 50 trials with TPE sampler + median pruner via `LightGBMPruningCallback`; trial scorer = sum of 5 pinball losses on 2023 val. Train 2018-2021; early-stop val 2022; trial scorer 2023. Tuned params reused across all 4 backtest folds. Backtest snapshot keyed by `(position, year, metric, model_class)` extended from 768 → 1136 rows (368 new lightgbm-tuned rows; SAMPLED_SUMMARY-vs-QUANTILE family asymmetry pinned, TODO #28 still open).

### Per-position model_ids

| Position | Model A model_id | Model C model_id | Model C-tuned model_id |
|---|---|---|---|
| WR | `baseline:wr:6d955427:2018-2023` | `lightgbm:wr:a4dd5a82:2018-2023` | `lightgbm-tuned:wr:62df14ad:2018-2023` |
| QB | `baseline:qb:c98738f3:2018-2023` | `lightgbm:qb:06fadb3f:2018-2023` | `lightgbm-tuned:qb:fc902ed6:2018-2023` |
| RB | `baseline:rb:5a86c8ee:2018-2023` | `lightgbm:rb:fb169c0e:2018-2023` | `lightgbm-tuned:rb:5d69fdfe:2018-2023` |
| TE | `baseline:te:9c00025b:2018-2023` | `lightgbm:te:bd4c2a5b:2018-2023` | `lightgbm-tuned:te:89dafdb6:2018-2023` |

### Adoption-gate verdict — DO NOT ADOPT Model C-tuned as default

Spec §1.3 required Model C-tuned to beat Model A on three criteria. **All three failed.**

| Criterion | Threshold | Actual (C-tuned vs A) | Pass? |
|---|---|---|---|
| Composite RMSE strictly lower on >=12 of 16 cells; not worse by >1% on any cell | C-tuned < A on 12+ cells; max +1% worse | C-tuned strictly lower on 4/16; max +2.95% worse (TE 2024); 8/16 cells exceed 1% | **FAIL** |
| Spearman top-N within +-0.005 of A on every cell | All 16 within +-0.005 | 9/16 within tolerance; 7 fail; worst +0.0163 (QB 2021) | **FAIL** |
| Weekly mean [p10,p90] coverage no worse on any cell; mean improvement >= +0.02 | No regressions; mean delta >= +0.02 | C-tuned no worse on 5/16 (all 4 QB cells + TE 2021 marginal); mean delta -0.0630 | **FAIL** |

### Side-by-side metric comparison (16 cells)

A vs C vs C-tuned with the C-tuned − A deltas on RMSE, Spearman, and calibration. `RMSE Ctuned-A %` is the percentage delta on composite RMSE; positive = C-tuned worse; threshold is +1.00% per criterion 1.

| Cell | RMSE A | RMSE C | RMSE C-tuned | RMSE Ctuned-A % | MAE A | MAE C | MAE Ctuned | Spearman A | Spearman C | Spearman Ctuned | Spearman Ctuned-A | Calib A | Calib C | Calib Ctuned | Calib Ctuned-A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QB 2021 | 7.8342 | 7.8386 | 7.7320 | -1.30% | 6.3606 | 6.4652 | 6.3581 | 0.9342 | 0.9501 | 0.9505 | +0.0163 | 0.6947 | 0.6857 | 0.7128 | +0.0180 |
| QB 2022 | 7.2261 | 7.2718 | 7.1220 | -1.44% | 5.7093 | 5.7649 | 5.6199 | 0.9669 | 0.9655 | 0.9668 | -0.0001 | 0.7458 | 0.7595 | 0.7854 | +0.0396 |
| QB 2023 | 7.3092 | 7.3213 | 7.2602 | -0.67% | 5.8796 | 5.9636 | 5.8887 | 0.9454 | 0.9560 | 0.9576 | +0.0122 | 0.7313 | 0.6955 | 0.7448 | +0.0134 |
| QB 2024 | 7.6995 | 7.5523 | 7.4845 | -2.79% | 6.0788 | 6.1338 | 6.0594 | 0.9383 | 0.9450 | 0.9460 | +0.0078 | 0.7018 | 0.6944 | 0.7281 | +0.0263 |
| RB 2021 | 6.8486 | 7.0688 | 6.9895 | +2.06% | 5.2108 | 5.6643 | 5.5522 | 0.9700 | 0.9565 | 0.9611 | -0.0089 | 0.7475 | 0.6008 | 0.6167 | -0.1308 |
| RB 2022 | 6.6359 | 6.8370 | 6.7032 | +1.01% | 5.0383 | 5.4613 | 5.3138 | 0.9658 | 0.9603 | 0.9674 | +0.0016 | 0.7415 | 0.6221 | 0.6379 | -0.1037 |
| RB 2023 | 6.3143 | 6.5069 | 6.4255 | +1.76% | 4.7179 | 5.0392 | 5.0039 | 0.9665 | 0.9600 | 0.9634 | -0.0032 | 0.7872 | 0.6354 | 0.6590 | -0.1281 |
| RB 2024 | 6.4860 | 6.6370 | 6.6242 | +2.13% | 4.9290 | 5.1591 | 5.1982 | 0.9753 | 0.9746 | 0.9763 | +0.0009 | 0.7568 | 0.6231 | 0.6307 | -0.1261 |
| TE 2021 | 5.3365 | 5.4636 | 5.3835 | +0.88% | 3.8932 | 4.1989 | 4.1390 | 0.9655 | 0.9598 | 0.9617 | -0.0038 | 0.7350 | 0.6398 | 0.6670 | -0.0680 |
| TE 2022 | 5.2498 | 5.3710 | 5.3197 | +1.33% | 3.6970 | 4.1489 | 4.0954 | 0.9615 | 0.9606 | 0.9621 | +0.0006 | 0.7647 | 0.6415 | 0.6700 | -0.0947 |
| TE 2023 | 4.9422 | 5.1324 | 5.0272 | +1.72% | 3.5439 | 4.0228 | 3.9442 | 0.9704 | 0.9709 | 0.9743 | +0.0039 | 0.7561 | 0.6711 | 0.6900 | -0.0662 |
| TE 2024 | 5.0804 | 5.2661 | 5.2302 | +2.95% | 3.7446 | 4.1408 | 4.1022 | 0.9620 | 0.9568 | 0.9582 | -0.0038 | 0.7345 | 0.6401 | 0.6735 | -0.0611 |
| WR 2021 | 6.7333 | 6.8837 | 6.8025 | +1.03% | 5.0891 | 5.4583 | 5.3996 | 0.9699 | 0.9624 | 0.9651 | -0.0048 | 0.6956 | 0.6107 | 0.6321 | -0.0635 |
| WR 2022 | 6.6255 | 6.7479 | 6.7149 | +1.35% | 5.0221 | 5.3711 | 5.3580 | 0.9767 | 0.9670 | 0.9690 | -0.0076 | 0.6970 | 0.6004 | 0.6151 | -0.0818 |
| WR 2023 | 6.5159 | 6.7129 | 6.5922 | +1.17% | 4.7814 | 5.2292 | 5.0860 | 0.9680 | 0.9590 | 0.9628 | -0.0051 | 0.7256 | 0.6220 | 0.6379 | -0.0877 |
| WR 2024 | 6.6728 | 6.7339 | 6.7218 | +0.74% | 4.9437 | 5.1907 | 5.2066 | 0.9739 | 0.9669 | 0.9673 | -0.0066 | 0.7109 | 0.6128 | 0.6177 | -0.0933 |

### Tuning helped — but not enough

Aggregate movement vs untuned Model C (Plan 5):

| Metric | Untuned C vs A | Tuned C vs A | Delta from tuning |
|---|---|---|---|
| RMSE: cells where C beats A | 1/16 | 4/16 | +3 cells |
| RMSE: max pct worse | +3.85% | +2.95% | -0.90 pp |
| Spearman: cells outside ±0.005 | 12/16 | 7/16 | -5 cells |
| Spearman: max abs delta | 0.0135 | 0.0163 | +0.0028 (worse on outlier) |
| Calibration: cells where C-tuned worse | 15/16 | 11/16 | -4 cells |
| Calibration: mean delta | -0.0857 | -0.0630 | +0.0227 |

**QB cells responded strongly to tuning.** All 4 QB years now strictly beat Model A on RMSE (-0.67% to -2.79%) and on calibration (+0.013 to +0.040 deltas; all positive). The hand-set defaults from Plan 5 had QBs landing in only 1 cell of "C wins"; tuning produced 4-for-4. QB tuning preferred shallow trees (`max_depth` mostly 3) with moderate `num_leaves` (33-127) and meaningful `reg_alpha` on count stats — i.e., regularize harder than the defaults did.

**RB / TE / WR cells improved but did not flip.** All three positions still regress on RMSE (1-3% worse), Spearman (mostly within tolerance now but some still outside), and calibration (still 6-13 pp under A). Tuning compressed the gaps but didn't eliminate them. The Plan 5 post-mortem hypothesis #1 (per-stat sub-models lack a shared prior; small-data positions overfit) and #4 (multi-output training would let the model borrow strength across stats) are the most plausible remaining mechanisms — tuning operates *within* per-stat sub-models and so cannot address the fundamental "each sub-model fits its own noise" problem.

### Why Model C-tuned still lost (refined hypothesis)

Plan 5's post-mortem listed four candidate mechanisms; tuning addresses only #2:

1. **No shared prior across the 5 quantile sub-models per (position, stat).** Tuning made each sub-model's hyperparameters more conservative on average (lower `learning_rate`, more aggressive `min_child_samples`), but each still fits independently. **Not addressed.**
2. **Hand-set hyperparameters were sub-optimal.** Tuning addressed this directly. QBs benefited materially; smaller-data positions partially benefited.
3. **5-quantile interpolation is too coarse.** Tuning didn't change the quantile grid. **Not addressed.**
4. **Per-stat independent training discards shared signal across stats.** Tuning operates at the per-stat level. **Not addressed.**

Mechanisms #1 and #4 jointly explain why QB tunes well while WR/RB/TE don't: QB has fewer rows but each row has one strong target (passing yards) with high signal-to-noise; the other QB stats are zero-inflated counts that benefit from the tuning's harder regularization. RB/TE/WR have similar-sized datasets but each (position, stat) sub-model is a small-data fit on its own — total dataset size is fine, but per-sub-model is starved. A multi-output gradient-boosted model trained jointly across the 6 stats per position (and arguably across all 4 positions) is the natural next experiment.

### Decision

**Default model selection:** Model A stays the production default. Model C and Model C-tuned both ship as peers; neither is adopted. **No Plan 5c is filed** — the diagnostic verdict is unambiguous on the criteria, and per-fold tuning would not address mechanisms #1 / #3 / #4.

**Pivot:** the next model-improvement track is one of:
- **Plan 6 — Model D ensemble.** Stacked predictor (Model A + Model C + Model C-tuned). Even with all three losing head-to-head against A on most cells, a per-cell weighted ensemble could beat A — particularly on QB where C-tuned now has a robust edge. Cheapest experiment given Plan 5b's infrastructure.
- **TODO #3 (PBP / EPA features).** Feature track. Independent of model class. Estimated 5-15% RMSE win on top of any model class.
- **TODO #23 (target decomposition).** Volume × efficiency factorization. Independent of model class. Estimated 3-10% RMSE win.
- **Future Plan: multi-output LightGBM / shared-prior training** — addresses mechanisms #1 / #4 directly. Not yet specced; would inherit the LightGBM machinery from Plan 5.

The user picks one in the next session. Plan 4 (public Python API + CLI verbs + free-tier hosting) remains the post-modeling milestone.

### Per-position model_ids on disk

Standalone artifacts at `models/artifacts/lightgbm-tuned-{pos}-2018-2023-{hash}.joblib`. Backtest harness regenerates per-fold artifacts via the feature cache; standalone artifacts are for ad-hoc prediction / sanity checks.

---

## Plan 5 — LightGBM with Quantile Regression (Model C) — shipped (run 2026-04-27)

**Closes:** TODO #26.

`LightGBMModel` (Model C) lands as a peer of `BaselineModel` (Model A) under
the existing `Model` Protocol. Per-stat sub-models trained at quantiles
[0.05, 0.10, 0.50, 0.90, 0.95]; per-row prediction sorts to enforce
non-crossing, clips to [0, inf) for non-negative stats, wraps in
`QuantileDistribution`, and runs through the unchanged `score_distribution`
scoring layer. New `DistributionFamily.QUANTILE` + codec branch.
`POSITION_DISPATCH` extended with `factories: dict[str, Callable]` keyed by
model class name. Backtest harness gains `--model {baseline,lightgbm,both}`;
snapshot file renamed `baseline_metrics.json` → `model_metrics.json` and
rows keyed by `(position, year, metric, model_class)` (400 → 768 rows; LightGBM
skips 32 season_calibration_* rows per the harness gate that limits
season-aggregation to SAMPLED_SUMMARY family — see Task 18 follow-up).

### Per-position model_ids

| Position | Model A model_id (current) | Model C model_id (this plan) |
|---|---|---|
| WR | (Plan 3e Phase 1: `baseline:wr:6d955427:2018-2023`) | `lightgbm:wr:a4dd5a82:2018-2023` |
| QB | (Plan 3e Phase 1: `baseline:qb:c98738f3:2018-2023`) | `lightgbm:qb:06fadb3f:2018-2023` |
| RB | (Plan 3e Phase 1: `baseline:rb:5a86c8ee:2018-2023`) | `lightgbm:rb:fb169c0e:2018-2023` |
| TE | (Plan 3e Phase 1: `baseline:te:9c00025b:2018-2023`) | `lightgbm:te:bd4c2a5b:2018-2023` |

### Adoption-gate verdict — DO NOT ADOPT Model C as default

Spec §1.3 required Model C to beat Model A on three criteria. **All three failed.**

| Criterion | Threshold | Actual | Pass? |
|---|---|---|---|
| Composite RMSE strictly lower on >=12 of 16 cells; not worse by >1% on any cell | C <= A on 12+ cells; max +1% worse | C strictly lower on 1/16; max +3.85% worse (TE 2023); 11/16 cells exceed 1% | **FAIL** |
| Spearman top-N within +-0.005 of A on every cell | All 16 within +-0.005 | 4/16 within tolerance; 12 fail; worst -0.0135 (RB 2021) | **FAIL** |
| Weekly mean [p10,p90] coverage no worse on any cell; mean improvement >= +0.02 | No regressions; mean delta >= +0.02 | C no worse on 1/16 (QB 2022 +0.0137); mean delta -0.0857 | **FAIL** |

### Side-by-side metric comparison (16 cells)

Per-cell deltas (Model C - Model A) and the cell winner. `tie` indicates the absolute pct-delta is within the tolerance band (0-1% on RMSE/MAE; ±0.005 on Spearman; ±0.005 on calibration). `A` / `C` indicate a strict winner.

| Cell | composite_rmse (A) | composite_rmse (C) | RMSE pct delta | composite_mae A | composite_mae C | spearman A | spearman C | spearman delta | calib_p10p90 A | calib_p10p90 C | calib delta | RMSE winner | MAE winner | Spearman winner | Calib winner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QB 2021 | 7.8342 | 7.8386 | +0.06% | 6.3606 | 6.4652 | 0.9342 | 0.9501 | +0.0159 | 0.6947 | 0.6857 | -0.0090 | tie | A | C | A |
| QB 2022 | 7.2261 | 7.2718 | +0.63% | 5.7093 | 5.7649 | 0.9669 | 0.9655 | -0.0014 | 0.7458 | 0.7595 | +0.0137 | tie | tie | tie | C |
| QB 2023 | 7.3092 | 7.3213 | +0.17% | 5.8796 | 5.9636 | 0.9454 | 0.9560 | +0.0106 | 0.7313 | 0.6955 | -0.0358 | tie | A | C | A |
| QB 2024 | 7.6995 | 7.5523 | -1.91% | 6.0788 | 6.1338 | 0.9383 | 0.9450 | +0.0067 | 0.7018 | 0.6944 | -0.0073 | C | tie | C | A |
| RB 2021 | 6.8486 | 7.0688 | +3.22% | 5.2108 | 5.6643 | 0.9700 | 0.9565 | -0.0135 | 0.7475 | 0.6008 | -0.1468 | A | A | A | A |
| RB 2022 | 6.6359 | 6.8370 | +3.03% | 5.0383 | 5.4613 | 0.9658 | 0.9603 | -0.0055 | 0.7415 | 0.6221 | -0.1195 | A | A | A | A |
| RB 2023 | 6.3143 | 6.5069 | +3.05% | 4.7179 | 5.0392 | 0.9665 | 0.9600 | -0.0065 | 0.7872 | 0.6354 | -0.1518 | A | A | A | A |
| RB 2024 | 6.4860 | 6.6370 | +2.33% | 4.9290 | 5.1591 | 0.9753 | 0.9746 | -0.0007 | 0.7568 | 0.6231 | -0.1337 | A | A | tie | A |
| TE 2021 | 5.3365 | 5.4636 | +2.38% | 3.8932 | 4.1989 | 0.9655 | 0.9598 | -0.0057 | 0.7350 | 0.6398 | -0.0951 | A | A | A | A |
| TE 2022 | 5.2498 | 5.3710 | +2.31% | 3.6970 | 4.1489 | 0.9615 | 0.9606 | -0.0009 | 0.7647 | 0.6415 | -0.1232 | A | A | tie | A |
| TE 2023 | 4.9422 | 5.1324 | +3.85% | 3.5439 | 4.0228 | 0.9704 | 0.9709 | +0.0005 | 0.7561 | 0.6711 | -0.0851 | A | A | tie | A |
| TE 2024 | 5.0804 | 5.2661 | +3.66% | 3.7446 | 4.1408 | 0.9620 | 0.9568 | -0.0052 | 0.7345 | 0.6401 | -0.0944 | A | A | A | A |
| WR 2021 | 6.7333 | 6.8837 | +2.23% | 5.0891 | 5.4583 | 0.9699 | 0.9624 | -0.0076 | 0.6956 | 0.6107 | -0.0849 | A | A | A | A |
| WR 2022 | 6.6255 | 6.7479 | +1.85% | 5.0221 | 5.3711 | 0.9767 | 0.9670 | -0.0096 | 0.6970 | 0.6004 | -0.0966 | A | A | A | A |
| WR 2023 | 6.5159 | 6.7129 | +3.02% | 4.7814 | 5.2292 | 0.9680 | 0.9590 | -0.0090 | 0.7256 | 0.6220 | -0.1036 | A | A | A | A |
| WR 2024 | 6.6728 | 6.7339 | +0.92% | 4.9437 | 5.1907 | 0.9739 | 0.9669 | -0.0070 | 0.7109 | 0.6128 | -0.0981 | tie | A | A | A |

### Why Model C lost — initial analysis

LightGBM-with-defaults systematically under-covers and underperforms Ridge on RB/TE/WR; only QBs see meaningful improvement. Plausible causes (none investigated in Plan 5; deferred):

1. **Quantile-loss training does not regularize against under-confidence.** The per-stat sub-models train independently at p5 / p10 / p50 / p90 / p95, with no shared prior. With ~6-15K rows per (position, stat), each sub-model fits noise that pushes the predicted interval inward. Ridge's L2 prior + post-hoc parametric variance (Plan 3e NB-2 for counts; Normal/Gamma for the rest) is more conservative.
2. **Hand-set hyperparameters, not tuned.** Plan 5 §1.3 explicitly deferred hyperparameter tuning to a focused follow-up "if results justify." `n_estimators=2000` + `learning_rate=0.05` + `num_leaves=31` is a reasonable starting point but not optimal for any specific stat.
3. **5-quantile interpolation is too coarse.** Tail accuracy depends on knot density; 5 knots over [0.05, 0.95] interpolates linearly between p10 and p50 (40% mass) and between p50 and p90 (40% mass) — coarse enough to lose structure where the underlying distribution has skew.
4. **Per-stat independent training discards shared signal across stats.** A multi-output model trained jointly across the 6 stats per position would let it borrow strength.

### Next steps

**Default model selection**: keep Model A as the production default. Model C ships as a peer for future iteration but is not adopted today.

**Followup plans (none in scope for Plan 5):**
1. **Plan 5b — Hyperparameter tuning for Model C.** Optuna-based search per (position, stat, quantile) sub-model. If tuning closes the gap, revisit adoption.
2. **Plan 6 — Model D (ensemble of Model A + Model C).** Even though Model C lost head-to-head, a stacked predictor (e.g., per-cell weighted average with weights fit on a held-out year) could still beat A alone, particularly on QB where C has a real edge. Worth trying once the adoption-gate infra is in place.
3. **TODO #28 (filed below)** — widen `aggregate_to_season` to accept `QUANTILE` family so LightGBM cells get season_calibration_* metrics.

After Plan 5 + (5b? 6?): Plan 4 (public Python API + CLI verbs + free-tier hosting), then Draft Hub.

### Per-position model_ids on disk

Standalone artifacts at `models/artifacts/lightgbm-{pos}-2018-2023-{hash}.joblib`. Backtest harness regenerates per-fold artifacts via the feature cache; standalone artifacts are for ad-hoc prediction / sanity checks.

---

## Plan 3e Phase 3 — Per-tertile bucketing — REVERTED (run 2026-04-27)

**Closes:** Nothing new — the routing change did not survive validation. TODO #22 stays closed against Plan 3e overall, but the shipped Plan 3e state is now Phase 0 (diagnostic CLI) + Phase 1 (NB for count stats); Phase 2 + Phase 3 are both attempted-and-reverted (infrastructure preserved).

After Phase 3 merged on-branch, the empirical signal was unambiguous: per-tertile variance bucketing regressed weekly mean coverage by **0.016** (0.726 → 0.710) and season mean coverage by **0.062** (0.461 → 0.399) vs the Plan 3d baseline. QB cells gained ~+0.02 weekly across all 4 years (their residuals are more uniformly homoscedastic across `mu_hat` tertiles, so bucketing produced near-equal per-bucket params and was harmless). RB/WR/TE cells regressed substantially (-0.025 to -0.037 weekly per cell, and as much as -0.135 on the worst season cell).

### Mechanism

The per-bucket variance estimator does not capture within-bucket residual asymmetry on positions whose low-pred buckets mix mostly-zero actuals with occasional big-game actuals. The bottom bucket gets a tighter std/shape/dispersion, which narrows the [p10, p90] interval on rows with the smallest predicted means — exactly where residuals are heteroscedastic *upward* (zero-inflated tails on count stats; right-skew on small-yards rows). Result: the central interval shrinks where actuals don't, and coverage drops. The right answer is quantile-based fitting (Plan 5 / quantile regression territory) — fit variance to minimize p10/p90 quantile loss directly rather than maximize residual likelihood.

### Decision

Revert Phase 3's routing in `BaselineModel.fit` and `BaselineModel.build_stat_distributions`. **Keep** the bucketing helpers (`_compute_tertile_cuts`, `_assign_bucket_indices`, `_per_bucket_normal_std_from_residuals`, `_per_bucket_gamma_alpha_from_residuals`, `_per_bucket_nb_dispersion_from_residuals`, `_per_bucket_student_t_params_from_residuals`), the widened `variance_params` value type (`float | list[float]`), and their unit tests as future infrastructure for plans that combine bucketing with non-symmetric within-bucket estimators or quantile-based fits.

### Verification

After revert + retrain + re-snapshot, the snapshot returns to Phase 1's baseline (commit `0078223`) **bit-for-bit**: weekly mean 0.733 → 0.733; season mean 0.428 → 0.428; max abs delta across all 400 metrics is 0.00000. Variance_params reverts to scalar shape (`{"std": X}` / `{"shape": X}` / `{"dispersion": X}`); per-position model_ids change because the source-file hash changes, but the underlying numbers don't.

### Final shipped state for Plan 3e

- Phase 0: diagnostic CLI (`scripts/diagnose_calibration.py`) + research report.
- Phase 1: `ParametricNegativeBinomial` for the 10 zero-inflated count stats (`*_tds`, `interceptions`, `fumbles_lost`); conditional MLE dispersion estimator with NB-2 / "size" parameterization; codec branch for the new family.
- Phase 2: Student-t routing for `*_yards` — attempted, reverted. `ParametricStudentT` class, codec branch, and `_student_t_params_from_residuals` estimator preserved as infrastructure.
- Phase 3: per-tertile variance bucketing across all routed families — attempted, reverted. Bucketing helpers + widened `variance_params` type preserved as infrastructure.

Spec calibration targets (min cell coverage ≥ 0.65; mean delta ≥ +0.10) still **not met** by the shipped state. Follow-up plans below.

### Follow-up plan candidates (post-merge brainstorming)

1. **ZIP (zero-inflated Poisson) for count cells** if NB still undercovers — handles the zero mass directly rather than via dispersion.
2. **Cross-week residual correlation modeling for season under-dispersion.** Season aggregation currently sums independent weekly draws; in reality, a player's good/bad weeks correlate (matchup quality, role, health). Modeling that correlation would widen season-total variance directly without touching weekly distributions. This is the canonical fix for the 0.30–0.50 season under-dispersion that has persisted through every Plan 3e attempt.
3. **Calibration-aware fitting.** Plan 3e fitted variance via residual MLE / method-of-moments; the empirical signal then says coverage missed. Direct fits that minimize a calibration loss (e.g., quantile loss at p10/p90) rather than a likelihood would close the loop. This is a structural shift in the fitting paradigm and worth its own spec — the bucketing infrastructure preserved on-branch is a natural building block here.

---

## Plan 3e Phase 3 — Per-tertile variance bucketing (run 2026-04-27, on branch `feat/plan-3e-calibration-tightening`)

**Closes:** Plan 3e overall (Phases 0 + 1 + 2-attempted-and-reverted + 3); TODO #22 closed.

Phase 3 is the cross-cutting fix: every (position, stat) cell now persists 33rd/67th-percentile cuts on `mu_hat` from the training set + a 3-element list of variance parameters (one per tertile bucket). At predict time, each row is routed to its bucket via `np.searchsorted` and the corresponding parameter is selected. Applies to all families currently in use (NORMAL, GAMMA, NEGATIVE_BINOMIAL).

**Phase 3 delivered:**
- `BaselineModel.variance_params` shape generalized from `dict[Stat, dict[str, float]]` to `dict[Stat, dict[str, float | list[float]]]`.
- 5 new helpers: `_compute_tertile_cuts`, `_assign_bucket_indices`, `_per_bucket_normal_std_from_residuals`, `_per_bucket_gamma_alpha_from_residuals`, `_per_bucket_nb_dispersion_from_residuals` (and `_per_bucket_student_t_params_from_residuals` as future infrastructure).
- `BaselineModel.fit` rewritten to compute tertile cuts + per-bucket parameters per family.
- `BaselineModel.build_stat_distributions` rewritten to look up bucket per row + select per-bucket parameter.
- Codec unchanged (per-row distributions still emit concrete scalar params); mixed-family regression test added.
- Standalone artifacts retrained.
- Snapshot regenerated.

### Coverage delta vs Plan 3d baseline (pre-Plan-3e at commit `fe55d5b`)

| Metric | Pre-Plan-3e (3d at `fe55d5b`) | Post-Phase-3 | Delta |
|---|---|---|---|
| Weekly mean `calibration_p10p90` | 0.726 | 0.710 | **-0.016** |
| Weekly min `calibration_p10p90` | 0.675 (QB/2021) | 0.663 (WR/2022) | -0.012 |
| Season mean `season_calibration_p10p90` | 0.461 | 0.399 | **-0.062** |
| Season min `season_calibration_p10p90` | 0.313 (QB/2022) | 0.293 (QB/2021) | -0.020 |
| ALL-32-cells mean `[p10, p90]` coverage delta | — | — | **-0.039** |
| ALL-32-cells min coverage | 0.313 | 0.293 | -0.020 |

**Compared to Phase 1 alone** (snapshot at `0078223`, pre-bucketing): weekly mean 0.733 → 0.710 (-0.023); season mean 0.428 → 0.399 (-0.030); all-32-cells mean delta -0.026.

**Per-cell weekly highlights:**
- QB cells gained on bucketing: 2021 +0.020, 2022 +0.024, 2023 +0.025, 2024 +0.019 (QB cells are now the only weekly cells with positive deltas vs Plan 3d).
- RB cells regressed -0.025 to -0.033 across all 4 years.
- TE cells regressed -0.017 to -0.032 across all 4 years.
- WR cells regressed -0.030 to -0.037 across all 4 years (WR/2024 is the worst weekly miss).

**Per-cell season highlights:**
- Worst season-coverage regressions: WR/2022 -0.135, RB/2024 -0.121, WR/2023 -0.119, RB/2021 -0.090, WR/2021 -0.096.
- Only QB/2023 (+0.013) and QB/2022 (0.000) season cells held or improved.

### Per-position model_ids

| Position | model_id |
|---|---|
| WR | `baseline:wr:a1fe2727:2018-2023` |
| QB | `baseline:qb:5333a44e:2018-2023` |
| RB | `baseline:rb:078c171c:2018-2023` |
| TE | `baseline:te:f460c50f:2018-2023` |

### Sample variance_params shape (one stat per family per position)

- WR receiving_yards (NORMAL): `{'bucket_cuts': [38.288, 55.137], 'std_per_bucket': [25.599, 33.781, 41.593]}`
- WR receptions (GAMMA): `{'bucket_cuts': [2.984, 4.253], 'shape_per_bucket': [1.752, 2.787, 3.822]}`
- WR receiving_tds (NEGATIVE_BINOMIAL): `{'bucket_cuts': [0.226, 0.334], 'dispersion_per_bucket': [4.828, 1000.0, 1000.0]}`
- QB passing_yards (NORMAL): `{'bucket_cuts': [220.099, 250.908], 'std_per_bucket': [87.075, 75.772, 76.945]}`
- QB passing_tds (NEGATIVE_BINOMIAL): `{'bucket_cuts': [1.294, 1.650], 'dispersion_per_bucket': [1000.0, 1000.0, 1000.0]}`
- RB rushing_yards (NORMAL): `{'bucket_cuts': [40.814, 57.406], 'std_per_bucket': [28.511, 33.948, 37.936]}`
- TE receiving_yards (NORMAL): `{'bucket_cuts': [27.605, 40.259], 'std_per_bucket': [19.853, 24.328, 33.730]}`

### Spec target verification

**Both spec targets MISSED — and Phase 3 regressed coverage rather than improving it.**

- Min cell coverage across all 32 cells: 0.293 (target ≥ 0.65). **Not met.** No appreciable movement from Plan 3d (0.313).
- Mean coverage delta across all 32 cells: -0.039 (target ≥ +0.10). **Not met; regressed.**

### Mechanism of the regression

Per-tertile bucketing reduces variance in the bottom + middle buckets relative to the unbucketed pooled estimate. The bottom bucket now uses a tighter std/shape/dispersion, which narrows the [p10, p90] interval on the half of the dataset with the smallest predicted means — exactly the half where residuals are heteroscedastic *upward* (zero-inflated tails on count stats; right-skew on small-yards rows). Result: the central interval shrinks where the actuals don't, and coverage drops.

QB cells are the exception (uniform +0.02 weekly gains): QB residual variance is more uniformly homoscedastic across mu_hat tertiles than RB/WR/TE, so bucketing produces ~equal per-bucket params and avoids the asymmetric narrowing effect. RB/WR/TE — where heteroscedasticity is sharpest — are exactly where bucketing hurts most.

### Known shortfalls / follow-up plans

Recommended follow-up plans (none of these is in scope for Plan 3e — they are post-merge work):

1. **Revert Phase 3 if RB/WR/TE coverage matters more than QB.** The Phase-1 snapshot (`0078223`) had better mean coverage than Phase 3 (0.733 vs 0.710 weekly; 0.428 vs 0.399 season). A clean revert to Phase 1 is a reasonable call. Plan 3e Phase 3 ships the per-tertile mechanism + tests; reversing the routing is a one-commit follow-up.
2. **Asymmetric residual modeling.** Bucketing collapses the residual distribution to a single std/shape per bucket, which still assumes symmetric tails within each bucket. The data has zero-inflation (count stats) and right-skew (small-yards rows) that bucketing on its own can't capture. Follow-up plans:
   - **ZIP (zero-inflated Poisson) for count cells** if NB still undercovers — handles the zero mass directly rather than via dispersion.
   - **Per-bucket family choice** rather than per-cell — e.g., use NORMAL on the high-mean bucket of receiving_yards but Student-t on the low-mean bucket where the long right tail dominates.
3. **Cross-week residual correlation modeling for season under-dispersion.** Season aggregation currently sums independent weekly draws; in reality, a player's good/bad weeks correlate (matchup quality, role, health). Modeling that correlation would widen season-total variance directly without touching weekly distributions. This is the canonical fix for the 0.30–0.50 season under-dispersion that has persisted through every Phase 3e attempt.
4. **Calibration-aware fitting.** Plan 3e fitted variance via residual MLE / method-of-moments; the empirical signal then says coverage missed. Direct fits that minimize a calibration loss (e.g., quantile loss at p10/p90) rather than a likelihood would close the loop. This is a structural shift in the fitting paradigm and worth its own spec.

---

## Plan 3e Phase 2 — Student-t for yards stats — ATTEMPTED + REVERTED (run 2026-04-27)

**Closes:** Nothing — the routing change did not survive validation. TODO #22
remains in progress; Phase 3 (variance bucketing) is the next attempt.

Phase 2 attempted to route every `*_yards` stat (passing/rushing/receiving
yards across QB/RB/TE/WR) from `NORMAL` to `STUDENT_T` based on Phase 0's
per-cell AIC signal favoring heavy tails (delta `[-2160, -317]` across the 5
yards-stat cells). The new `ParametricStudentT(loc, scale, df)` distribution
class, `DistributionFamily.STUDENT_T` enum value, codec branches, and
`_student_t_params_from_residuals` MLE estimator were all built and wired
through `BaselineModel.fit` and `build_stat_distributions`.

### Empirical finding: weekly coverage regressed by ~1.5–2 pts uniformly

After retraining the standalone artifacts and regenerating the snapshot,
weekly `calibration_p10p90` dropped roughly 1.5–2 pts uniformly across
RB/WR/TE cells with no offsetting season-coverage gain. The regression was
not noise: it appeared on every position-year cell that contained a `*_yards`
stat in the points decomposition.

### Root cause: heavy tails narrow the [p10, p90] shoulder

The mechanism is structural, not a bug. Student-t with the data's empirical
tail shape (df ~5–8 across the yards stats) puts more probability mass in
the extreme outer tails and *less* in the central shoulder of the
distribution than `NORMAL` at similar total std. Since our success metric
is `[p10, p90]` coverage — i.e. the share of actuals that land in the
central 80% interval — Student-t's heavier extremes shrink that interval
and lose coverage even when its full-distribution likelihood is better.

Phase 0's AIC signal was correct on its own terms (Student-t is a closer
fit to the full residual distribution), but **AIC is not a calibration
metric for the central interval.** The two objectives can diverge structurally
when the underlying data has heavy tails — preferring the heavier-tailed family
on AIC simultaneously deprefers it on `[p10, p90]` coverage.

### Decision: revert Phase 2 routing; keep the infrastructure

Per user decision, the factory routing was reverted in this commit. After
revert, **zero stats route to `STUDENT_T`** across all 4 positions. Yards
stats are back to `NORMAL` everywhere; the snapshot returns bit-exactly to
the Phase 1 baseline at commit `0078223` (verified via Step 7 coverage
delta = 0.000 on weekly mean / season mean / weekly min / season min).

The `ParametricStudentT` class, `DistributionFamily.STUDENT_T` enum value,
codec round-trip, `_student_t_params_from_residuals` estimator, and the
`STUDENT_T` branches in `BaselineModel.fit` / `build_stat_distributions`
all remain in-tree as future infrastructure. Their dedicated unit tests
(`tests/test_distributions/test_student_t.py`,
`tests/test_distributions/test_codec.py::test_codec_round_trip_student_t`,
and the two estimator tests in `tests/test_models/test_baseline.py`) are
unchanged. Any future plan can wire them up; the current code is correct
and validated.

### Lesson learned

Phase 0's family-fit AIC signal preferred Student-t for yards stats, and
that signal was technically correct: Student-t *is* a better full-
distribution fit than Normal on these residuals. But AIC measures full-
distribution agreement, not central `[p10, p90]` coverage — and Plan 3e's
success metric is calibration of the central interval. When the underlying
data is heavy-tailed, the two objectives can diverge structurally: the
heavier-tailed family wins on AIC and loses on central coverage. **For
Plan 3e and any future calibration-tightening phase, the family choice
must be evaluated against the calibration metric directly, not via AIC
proxy.**

### Forward pointer

Phase 3 (per-tertile variance bucketing) is the next attempt at improving
weekly coverage. It addresses a different Phase 0 root cause (pervasive
heteroscedasticity, 18 of 24 cells with variance-bucket ratio > 1.5) and
operates orthogonally to family choice — it can be wired on top of any
future family swap.

### Per-position model_ids (after revert)

| Position | model_id |
|---|---|
| WR | `baseline:wr:6d955427:2018-2023` |
| QB | `baseline:qb:c98738f3:2018-2023` |
| RB | `baseline:rb:5a86c8ee:2018-2023` |
| TE | `baseline:te:9c00025b:2018-2023` |

(Code hashes rotate from Phase 1's because the `baseline.py` module docstring
+ `_*_DIST_FAMILIES` dicts changed.)

---

## Plan 3e Phase 1 — Negative Binomial for count stats (run 2026-04-27, on branch `feat/plan-3e-calibration-tightening`)

**Closes:** TODO #22 progress; Phase 0 complete; Phases 2-3 in progress on this branch.

Phase 1 routes the 10 zero-inflated count stats (every `*_tds` + `interceptions` + `fumbles_lost` across QB/RB/TE/WR) from GAMMA to NEGATIVE_BINOMIAL via the new `ParametricNegativeBinomial` family. Conditional MLE estimator (`_negative_binomial_dispersion_from_residuals`) fits dispersion per stat, addressing Phase 0's marginal-vs-conditional AIC asymmetry caveat in production.

**Phase 1 delivered:**
- `ParametricNegativeBinomial(mean, dispersion)` distribution class implementing the Distribution Protocol; standard NB-2 parameterization (var = mean + mean²/dispersion).
- `DistributionFamily.NEGATIVE_BINOMIAL` enum value + codec branches in `pack_per_stat_params` / `unpack_per_stat_params`.
- `_negative_binomial_dispersion_from_residuals` conditional-MLE estimator (`scipy.optimize.minimize_scalar` bounded; `_NB_DISPERSION_CLIP = (0.01, 1000.0)`).
- `BaselineModel.fit` and `BaselineModel.build_stat_distributions` route NB stats correctly.
- All 4 per-position factories (_WR/QB/RB/TE_DIST_FAMILIES) updated.
- Standalone artifacts retrained (4 `models/artifacts/baseline-{pos}-...joblib` files; new `model_id` per position because `code_hash` rotates).
- Snapshot regenerated; gate passes.
- Bug fix landed mid-phase (commit `865ccfb`): inverted `_scipy_n_p()` conversion was producing wrong NB variance; fixed to standard NB-2.

### Coverage delta vs Phase 0 baseline

Pre-Phase-1 baseline = Plan 3d's snapshot at merge commit `fe55d5b`.

| Metric | Pre-Phase-1 | Post-Phase-1 | Delta |
|---|---|---|---|
| Weekly mean `calibration_p10p90` | 0.726 | 0.733 | +0.007 |
| Weekly min `calibration_p10p90` | 0.675 (QB/2021) | 0.695 (QB/2021) | +0.020 |
| Season mean `season_calibration_p10p90` | 0.461 | 0.428 | -0.033 |
| Season min `season_calibration_p10p90` | 0.313 (QB/2022) | 0.293 (QB/2021) | -0.020 |

**Weekly coverage improved modestly across all positions, with the largest gains on QB and TE:**
- QB cells: 2021 +0.020, 2023 +0.022 (the Phase 0 diagnostic flagged QB as the worst-calibrated position).
- TE cells: 2022 +0.015, 2023 +0.021, 2024 +0.018.
- WR/RB cells: small mixed deltas in `[-0.009, +0.004]`, all within tolerance.

**Season coverage regressed across most cells.** This is an expected secondary effect: Phase 0's GAMMA fits had inflated variance on count stats, so when independent weekly distributions were summed for the season Monte Carlo, the over-wide weekly tails partially compensated for the missing inter-week covariance. Replacing GAMMA with NB-2 (which correctly tightens count-stat variance per the conditional MLE fit) removes that compensating slack, exposing the true season-aggregation under-dispersion. Worst-affected cells: WR/2022 -0.074, WR/2023 -0.075, RB/2024 -0.073, RB/2022 -0.059. Phase 2 (Student-t for yards) and Phase 3 (variance bucketing) should not directly address this, but season-level inter-week correlation (a Plan-3e follow-up or post-3e item) will.

**Per-stat MAE/RMSE shifts on NB-routed stats are below the 0.01 noise floor across all 16 cells** — NB-2 and GAMMA agree on the conditional mean by construction; only the variance/shape changes, which feeds into calibration metrics, not point-prediction metrics.

### Per-position model_ids

| Position | model_id |
|---|---|
| WR | `baseline:wr:6964f45a:2018-2023` |
| QB | `baseline:qb:178a0438:2018-2023` |
| RB | `baseline:rb:0d8180b1:2018-2023` |
| TE | `baseline:te:ae33da15:2018-2023` |

### Next: Phase 2 (Student-t for yards stats) on this same branch.

---

## Plan 3e Phase 0 — Calibration diagnostic (run 2026-04-26, on branch `feat/plan-3e-calibration-tightening`)

**Closes:** None. TODO #22 (Plan 3e calibration tightening) stays open — Phase 0
delivers the diagnostic only; the full Plan 3e tightening closes #22. Phase 0
surfaced 3 root causes that the spec amendment (next gate) will translate into
Phase 1+ implementation work.

Phase 0 = a `scripts/diagnose_calibration.py` CLI plus a research report
(`docs/superpowers/research/2026-04-26-calibration-diagnosis.md`) that fits
alternative distribution families against per-row residuals from the latest
backtest run and identifies why weekly + season `[p10, p90]` coverage
under-disperses to 0.30–0.55 vs the 0.80 target. The spec amendment that adds
Phase 1+ implementation phases to
`docs/superpowers/specs/2026-04-26-plan-3e-calibration-tightening-design.md` is
the next gate before any model code changes (per spec section 3 decision gate).

### Diagnostic findings (3 root causes)

1. **Zero-inflated count stats are catastrophically miscalibrated under
   GAMMA.** `coverage_p10p90 = 0.0` across every (position, stat) cell for
   `*_tds`, `interceptions`, and `fumbles_lost` — the fitted GAMMA's p10 sits
   above zero while the modal residual is exactly zero. Root cause is family
   choice; the recommendation is a family swap to negative-binomial / zero-
   inflated negative binomial.
2. **Continuous yards stats are heavy-tailed.** Student-t fits beat Normal on
   AIC by `delta in [-2160, -317]` across 5 yards-stat cells (passing/rushing/
   receiving × position). Recommendation is a family swap to Student-t for the
   `*_yards` stats.
3. **Heteroscedasticity is pervasive.** 18 of 24 (position, stat) cells have
   variance-bucket ratio > 1.5 (top vs bottom predicted-mean tertile).
   Variance bucketing is needed independent of family choice and combines with
   the family swaps above.

See `docs/superpowers/research/2026-04-26-calibration-diagnosis.md` for the full
per-cell table, recommended fixes, and selection methodology.

### Next gate

The spec amendment (Plan 3e Phase 1+) is the next gate before any model code
changes. Re-invocation of `superpowers:brainstorming` happens in the next
user-driven session to scope Phase 1 (family-family swaps), Phase 2 (variance
bucketing), and a final regression-gate phase against the 3d snapshot.

---

## Plan 3d — Real Monte Carlo season aggregation (run 2026-04-26, on branch `feat/plan-3d-monte-carlo-season`)

**Closes:** TODO #13 (per-row seeds), TODO #14 (SAMPLED_SUMMARY family), TODO #19 (gate non-determinism by demonstration).

Held-out years: 2021–2024 (same as Plan 3c). Snapshot at 400 rows
(368 weekly metrics from 3c + 32 new season-calibration rows from 3d).
Full gate runtime: 292.73 seconds.

### Composite metrics by (position, year)

| Position | Year | composite_rmse | composite_mae | spearman_topN | calibration_p10p90 | calibration_le_p90 |
|---|---|---|---|---|---|---|
| QB | 2021 | 7.841 | 6.357 | 0.933 | 0.675 | 0.857 |
| QB | 2022 | 7.240 | 5.703 | 0.968 | 0.737 | 0.845 |
| QB | 2023 | 7.324 | 5.868 | 0.945 | 0.709 | 0.831 |
| QB | 2024 | 7.722 | 6.072 | 0.938 | 0.699 | 0.842 |
| RB | 2021 | 6.864 | 5.147 | 0.970 | 0.745 | 0.846 |
| RB | 2022 | 6.631 | 4.965 | 0.967 | 0.746 | 0.851 |
| RB | 2023 | 6.322 | 4.641 | 0.967 | 0.791 | 0.867 |
| RB | 2024 | 6.487 | 4.853 | 0.975 | 0.766 | 0.863 |
| TE | 2021 | 5.352 | 3.856 | 0.966 | 0.727 | 0.845 |
| TE | 2022 | 5.282 | 3.670 | 0.960 | 0.750 | 0.830 |
| TE | 2023 | 4.978 | 3.527 | 0.969 | 0.735 | 0.821 |
| TE | 2024 | 5.101 | 3.712 | 0.962 | 0.717 | 0.823 |
| WR | 2021 | 6.746 | 5.040 | 0.970 | 0.700 | 0.827 |
| WR | 2022 | 6.633 | 4.975 | 0.977 | 0.693 | 0.831 |
| WR | 2023 | 6.531 | 4.737 | 0.968 | 0.723 | 0.832 |
| WR | 2024 | 6.693 | 4.899 | 0.975 | 0.707 | 0.825 |

Drift from Plan 3c snapshot was within tolerance for every weekly metric
(largest absolute drift: `RB/2021/calibration_p10p90` 0.7536 -> 0.7452,
abs delta 0.0084 vs 0.03 tolerance; largest relative drift: `RB/2024/composite_mae`
+0.165% vs 5% tolerance). 77 of 368 existing rows show non-zero drift; all
are within tolerance. Cause: the per-row seed change in `score_distribution`
(closes TODO #13) reorders Monte Carlo draws, but the underlying regression
math is unchanged. See `/tmp/3d-pre-snapshot-drift.txt` for the raw
`--check` output.

### Season-total calibration (new in Plan 3d)

| Position | Year | season_calibration_p10p90 | season_calibration_le_p90 |
|---|---|---|---|
| QB | 2021 | 0.317 | 0.976 |
| QB | 2022 | 0.313 | 0.928 |
| QB | 2023 | 0.388 | 0.900 |
| QB | 2024 | 0.377 | 0.935 |
| RB | 2021 | 0.521 | 0.896 |
| RB | 2022 | 0.478 | 0.853 |
| RB | 2023 | 0.413 | 0.857 |
| RB | 2024 | 0.516 | 0.879 |
| TE | 2021 | 0.500 | 0.925 |
| TE | 2022 | 0.474 | 0.853 |
| TE | 2023 | 0.540 | 0.876 |
| TE | 2024 | 0.432 | 0.890 |
| WR | 2021 | 0.505 | 0.881 |
| WR | 2022 | 0.563 | 0.898 |
| WR | 2023 | 0.562 | 0.881 |
| WR | 2024 | 0.479 | 0.863 |

Season-total `[p10, p90]` coverage is well below target (0.80) — typically
0.30–0.55 across cells, worst on QB (0.31–0.39). This inherits 3c's weekly
under-dispersion: when independent under-dispersed weekly distributions are
summed (with no covariance), the season distribution under-disperses further
because variances add but the systematic miss does not cancel. `<= p90`
coverage is closer to target (0.85–0.98) — the upper-tail stretch from
gamma summation partially masks the under-dispersion at p10. Plan 3e is
the calibration-tightening follow-up.

### Decision log (Plan 3d)

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-26 | params blob format = per-stat distribution params | Three orders of magnitude smaller than persisting full sample arrays; decomposable; deterministic regeneration via seed. |
| 2026-04-26 | Per-row seed = sha256 of `(gsis_id, season, week, ruleset.name)` truncated to 32 bits | Deterministic across processes (Python `hash()` is salt-randomized via PYTHONHASHSEED); independent across rows; reproducible. |
| 2026-04-26 | Aggregator regenerates per-week samples rather than persisting them | Storage 1000x smaller; regeneration is O(seconds); samples are deterministic given seed. |
| 2026-04-26 | Modal-position resolution for traded players | Deterministic; rare edge case; documented in docstring. |
| 2026-04-26 | Calibration tightening (MLE gamma alpha / variance buckets) explicitly deferred to Plan 3e | 3d's snapshot reflects under-dispersed calibration as the regression floor; tightening is a separable model-quality improvement. |

### Current status (as of 2026-04-26)

**Projections Core — Plan 3d (real Monte Carlo season aggregation) merged to `main` at commit `fe55d5b` (PR #9).**

**Predecessors:**
- Plan 1 (Foundations) merged at `8f02a6c`.
- Plan 2a merged at `7926090`; Plan 2b merged at `af325ea`.
- Plan 3a (WR Model A baseline) merged at `598ab9c`.
- Plan 3b (QB / RB / TE Model A baselines) merged at `c4a0401`.
- Plan 3c (walk-forward backtest gate) merged at `3db71a6` (PR #8).

### Next action

**Recommended: Plan 3e — calibration tightening.** Replace
`_gamma_alpha_from_residuals`'s method-of-moments with an MLE fit, and/or
add per-stat residual-variance bucketing by predicted-mean tertile, to
move weekly + season calibration coverage toward 0.80. The under-dispersion
shows up most acutely on QB season totals (p10–p90 coverage 0.31–0.39);
expect the largest tightening to come from QB-stat MLE fits.

---

## Plan 3c — Walk-forward backtest gate (run 2026-04-26, on branch `feat/plan-3c-backtest-harness`)

Held-out years: 2021, 2022, 2023, 2024 (4 years × 4 positions = 16 fits per gate run).
Train window: expanding from 2018 → year-1.
Snapshot file: `tests/backtest/baseline_metrics.json` (368 rows committed).
Gate: `pytest -m backtest --run-backtest` — opt-in, pre-PR. Full run: 133 seconds.
Default-on smoke: `tests/backtest/test_backtest_smoke.py` — one (WR, 2024) cell, ~15s.

### Composite metrics by (position, year)

| Position | Year | composite_rmse | composite_mae | spearman_topN | calibration_p10p90 | calibration_le_p90 |
|---|---|---|---|---|---|---|
| QB | 2021 | 7.846 | 6.364 | 0.933 | 0.677 | 0.860 |
| QB | 2022 | 7.234 | 5.702 | 0.967 | 0.740 | 0.848 |
| QB | 2023 | 7.323 | 5.868 | 0.945 | 0.712 | 0.834 |
| QB | 2024 | 7.714 | 6.068 | 0.939 | 0.702 | 0.844 |
| RB | 2021 | 6.868 | 5.143 | 0.970 | 0.754 | 0.849 |
| RB | 2022 | 6.635 | 4.963 | 0.967 | 0.753 | 0.851 |
| RB | 2023 | 6.324 | 4.636 | 0.967 | 0.796 | 0.868 |
| RB | 2024 | 6.486 | 4.845 | 0.975 | 0.769 | 0.862 |
| TE | 2021 | 5.351 | 3.857 | 0.966 | 0.720 | 0.841 |
| TE | 2022 | 5.278 | 3.671 | 0.960 | 0.753 | 0.831 |
| TE | 2023 | 4.973 | 3.527 | 0.970 | 0.738 | 0.825 |
| TE | 2024 | 5.098 | 3.714 | 0.962 | 0.716 | 0.821 |
| WR | 2021 | 6.743 | 5.044 | 0.970 | 0.698 | 0.827 |
| WR | 2022 | 6.631 | 4.979 | 0.977 | 0.694 | 0.833 |
| WR | 2023 | 6.529 | 4.742 | 0.968 | 0.726 | 0.833 |
| WR | 2024 | 6.691 | 4.903 | 0.975 | 0.702 | 0.825 |

### Naive baseline comparison (informational)

Naive = per-player trailing-4-game stat mean, with cold-start fallback to
per-position mean. **Model A beats naive on composite RMSE by 5–11% on
every (position, year) cell** — no inverted cells.

| Position | Naive composite RMSE range | Model A vs naive |
|---|---|---|
| QB | 7.83 – 8.53 | -6.2% to -10.8% |
| RB | 6.77 – 7.45 | -5.4% to -7.8% |
| TE | 5.30 – 5.67 | -5.6% to -6.2% |
| WR | 7.02 – 7.17 | -5.5% to -7.3% |

Spearman top-N: model and naive are tied within ±0.01 across all 16 cells.
Trailing-4-mean is already a very strong rank-correlation baseline (because
"good players keep being good"); Model A's value-add is in lower
RMSE / MAE on per-stat and composite metrics, not in ranking signal.

Calibration (`[p10, p90]` coverage): 0.67–0.80 across cells, target 0.80 — under-dispersed
in the same direction as 3a/3b's WR sanity check. Plan 3d's MLE-fit gamma α / variance
bucketing should tighten this; Plan 3c locks the current numbers in as the regression floor.

### Phase 6 unplanned-but-necessary fixes

Two issues surfaced during the first end-to-end run; both fixed in scope:

- **`score_distribution` perf vectorization** (commit `dc122a7`). The original per-sample Python loop building a Pydantic StatLine per sample × per stat × per row dominated harness runtime — 20–30 minutes for the full 16-cell run. Spec section 1.2 had deferred this perf TODO to Plan 3d, but the runtime made the gate functionally unrunnable, so vectorization was pulled forward. Math is bit-identical (linear scoring rule + same RNG draw order); existing scoring tests pass unchanged. Full gate now runs in 133s.
- **`tests/conftest.py` marker filter** (commit `4b5aea0`). The original `"backtest" in item.keywords` filter over-matched any test under `tests/backtest/` (pytest's keywords include path-derived components), wrongly skipping the default-on smoke test under the `--run-backtest` gate. Fixed by switching to `item.get_closest_marker("backtest")`. Network filter switched to the same idiom for consistency.

---

## Plan 3b — 2024 sanity check (run on branch `feat/plan-3b-qb-rb-te-baseline`)

Held-out year is 2024 (same as 3a; `nfl_data_py` has not yet published 2025). Each position trained on 2018-2023. Per-position evals are stdout-only — Plan 3c owns CI threshold gating.

### WR (retrained under Plan 3b's `BaselineModel` constructor)

```
Loading artifact: models\artifacts\baseline-wr-2018-2023-a2f581cf.joblib
model_id: baseline:wr:a2f581cf:2018-2023

=== WR 2024 sanity check (n=2048 player-weeks) ===

-- Per-stat fit --
            receptions  rmse= 2.051  mae= 1.543  mean_pred= 2.892  mean_actual= 3.116
       receiving_yards  rmse=31.198  mae=22.938  mean_pred=36.237  mean_actual=39.204
         receiving_tds  rmse= 0.495  mae= 0.347  mean_pred= 0.212  mean_actual= 0.256
         rushing_yards  rmse= 3.944  mae= 1.914  mean_pred= 1.311  mean_actual= 1.005
           rushing_tds  rmse= 0.086  mae= 0.017  mean_pred= 0.010  mean_actual= 0.007
          fumbles_lost  rmse= 0.122  mae= 0.033  mean_pred= 0.018  mean_actual= 0.015

-- Composite (PPR points) --
  mean prediction:  rmse=6.780  mae=4.910
  top-N season-total rank correlation (Spearman, all WRs): 0.971

-- Calibration --
  fraction in [p10, p90]: 0.708  (target ~ 0.80)
  fraction <= p90:        0.815  (target ~ 0.90)

=== End sanity check (informational; not a CI gate) ===
```

### QB

```
Loading artifact: models\artifacts\baseline-qb-2018-2023-3907548e.joblib
model_id: baseline:qb:3907548e:2018-2023

=== QB 2024 sanity check (n=684 player-weeks) ===

-- Per-stat fit --
         passing_yards  rmse=84.538  mae=68.175  mean_pred=199.516  mean_actual=192.405
           passing_tds  rmse= 1.068  mae= 0.866  mean_pred= 1.219  mean_actual= 1.219
         interceptions  rmse= 0.829  mae= 0.699  mean_pred= 0.684  mean_actual= 0.585
         rushing_yards  rmse=17.880  mae=13.369  mean_pred=18.163  mean_actual=17.197
           rushing_tds  rmse= 0.440  mae= 0.287  mean_pred= 0.191  mean_actual= 0.171
          fumbles_lost  rmse= 0.396  mae= 0.304  mean_pred= 0.205  mean_actual= 0.171

-- Composite (PPR points) --
  mean prediction:  rmse=7.810  mae=6.281
  top-N season-total rank correlation (Spearman, all QBs): 0.928

-- Calibration --
  fraction in [p10, p90]: 0.667  (target ~ 0.80)
  fraction <= p90:        0.860  (target ~ 0.90)

=== End sanity check (informational; not a CI gate) ===
```

### RB

```
Loading artifact: models\artifacts\baseline-rb-2018-2023-a7f565e9.joblib
model_id: baseline:rb:a7f565e9:2018-2023

=== RB 2024 sanity check (n=1316 player-weeks) ===

-- Per-stat fit --
         rushing_yards  rmse=30.294  mae=22.628  mean_pred=38.617  mean_actual=39.458
           rushing_tds  rmse= 0.531  mae= 0.373  mean_pred= 0.267  mean_actual= 0.296
            receptions  rmse= 1.523  mae= 1.174  mean_pred= 1.751  mean_actual= 1.734
       receiving_yards  rmse=15.410  mae=11.127  mean_pred=12.767  mean_actual=13.127
         receiving_tds  rmse= 0.248  mae= 0.118  mean_pred= 0.065  mean_actual= 0.064
          fumbles_lost  rmse= 0.213  mae= 0.093  mean_pred= 0.052  mean_actual= 0.047

-- Composite (PPR points) --
  mean prediction:  rmse=6.517  mae=4.802
  top-N season-total rank correlation (Spearman, all RBs): 0.975

-- Calibration --
  fraction in [p10, p90]: 0.773  (target ~ 0.80)
  fraction <= p90:        0.851  (target ~ 0.90)

=== End sanity check (informational; not a CI gate) ===
```

### TE

```
Loading artifact: models\artifacts\baseline-te-2018-2023-4706d589.joblib
model_id: baseline:te:4706d589:2018-2023

=== TE 2024 sanity check (n=1081 player-weeks) ===

-- Per-stat fit --
            receptions  rmse= 1.911  mae= 1.372  mean_pred= 2.271  mean_actual= 2.596
       receiving_yards  rmse=22.476  mae=16.371  mean_pred=23.030  mean_actual=26.175
         receiving_tds  rmse= 0.397  mae= 0.286  mean_pred= 0.191  mean_actual= 0.166
         rushing_yards  rmse= 4.423  mae= 0.399  mean_pred= 0.131  mean_actual= 0.256
           rushing_tds  rmse= 0.114  mae= 0.008  mean_pred= 0.002  mean_actual= 0.006
          fumbles_lost  rmse= 0.138  mae= 0.035  mean_pred= 0.016  mean_actual= 0.019

-- Composite (PPR points) --
  mean prediction:  rmse=5.143  mae=3.716
  top-N season-total rank correlation (Spearman, all TEs): 0.960

-- Calibration --
  fraction in [p10, p90]: 0.741  (target ~ 0.80)
  fraction <= p90:        0.821  (target ~ 0.90)

=== End sanity check (informational; not a CI gate) ===
```

The WR retrain in Phase 6 produced a new `model_id` (`a2f581cf` vs 3a's `925f492b`) because Plan 3b modified `baseline.py` (which is part of the hashed code-files list); substantively the predictions match the merged 3a artifact's output to within numerical noise.

---

## Plan 3a — 2024 WR sanity check (run 2026-04-25, on branch `feat/plan-3a-wr-model-a`)

Held-out year is 2024 not 2025 (spec called for 2025; `nfl_data_py` has not yet published 2025 data).

```
Loading artifact: models/artifacts/wr-baseline-2018-2023-925f492b.joblib
model_id: baseline:wr:925f492b:2018-2023

=== 2024 sanity check (n=2048 player-weeks) ===

-- Per-stat fit --
            receptions  rmse= 2.049  mae= 1.541  mean_pred= 2.900  mean_actual= 3.116
       receiving_yards  rmse=31.186  mae=22.946  mean_pred=36.331  mean_actual=39.204
         receiving_tds  rmse= 0.495  mae= 0.348  mean_pred= 0.212  mean_actual= 0.256
         rushing_yards  rmse= 3.945  mae= 1.917  mean_pred= 1.314  mean_actual= 1.005
           rushing_tds  rmse= 0.086  mae= 0.017  mean_pred= 0.010  mean_actual= 0.007
          fumbles_lost  rmse= 0.122  mae= 0.033  mean_pred= 0.018  mean_actual= 0.015

-- Composite (PPR points) --
  mean prediction:  rmse=6.775  mae=4.908
  top-N season-total rank correlation (Spearman, all WRs): 0.971

-- Calibration --
  fraction in [p10, p90]: 0.708  (target ~ 0.80)
  fraction <= p90:        0.816  (target ~ 0.90)
```

Soft-threshold check vs. spec §6.3:
- Spearman top-30 correlation ≥ 0.4 — **MET** (0.971 — very high, the model captures relative WR ranking well).
- Calibration `[p10, p90]` coverage in 70–90% range — **borderline MET** (70.8%; right at the lower bound). The predicted distributions are slightly too narrow (under-dispersed). Plan 3c's backtest harness can formalize this and motivate either MLE-fit gamma α (TODO note in spec §3.4) or per-stat residual variance buckets.
- Per-stat RMSE within 2× of naive-baseline RMSE — **n/a until we compute the naive baseline**; track for future.

Per-stat means are systematically slightly *under* actual (e.g., receptions 2.90 vs 3.12, receiving_yards 36.3 vs 39.2) — Ridge has shrunk toward the league mean, which is expected behavior. The bias is small enough that the rank correlation is preserved.

**Plan 3a deliverable: pipeline works end-to-end on real data.** Bad numbers would feed into Plan 3c's threshold-setting; the sanity numbers here are good enough that the pipeline is the load-bearing artifact, not the model itself.

---

## Current status (as of 2026-04-27)

**Projections Core — Plan 3e shipped state = Phase 0 + Phase 1; Phase 2 + Phase 3 attempted-and-reverted (infrastructure preserved). Branch ready for PR.** Phase 1 swapped 10 zero-inflated count stats from GAMMA to NB with conditional MLE dispersion fitting (weekly mean coverage 0.726 → 0.733; season mean 0.461 → 0.428). Phase 2 attempted Student-t for `*_yards` and was reverted after empirical coverage regressed (`ParametricStudentT` + codec branch + estimator preserved in-tree). Phase 3 wired per-tertile variance bucketing across all routed families and was reverted after empirical coverage regressed (-0.016 weekly mean / -0.062 season mean vs Plan 3d; bucketing helpers + widened `variance_params` type preserved as future infrastructure for quantile-based fitting). Snapshot returns bit-for-bit to Phase 1 baseline (commit `0078223`). TODO #22 closed against Plan 3e overall; follow-up plans documented under the Phase 3 revert block (ZIP for count cells, cross-week correlation, calibration-aware fitting).

**Plan 3e Phase 0 (calibration diagnostic) complete on same branch.** Diagnostic CLI + research report committed.

**Plan 3d (real Monte Carlo season aggregation) merged to `main` at commit `fe55d5b` (PR #9).**

**Predecessors:**
- Plan 1 (Foundations) merged at `8f02a6c`.
- Plan 2a merged at `7926090`; Plan 2b merged at `af325ea`.
- Plan 3a (WR Model A baseline) merged at `598ab9c`.
- Plan 3b (QB / RB / TE Model A baselines) merged at `c4a0401`.
- Plan 3c (walk-forward backtest gate) merged at `3db71a6` (PR #8).

**Plan 3e Phase 0 delivered (current branch, not yet merged):**
- New `scripts/diagnose_calibration.py` CLI: loads the latest `data/backtest/run_<ts>/` per-row results, fits alternative distribution families (Student-t, lognormal, negative-binomial) against per-stat residuals, and emits a per-cell summary CSV + per-cell QQ/residual plots + a recommendation column.
- New `tests/test_scripts/test_diagnose_calibration.py` (21 tests) covering the loader, residual extraction, summary stats, alternative-family fits (including the degenerate Student-t guardrail), recommendation logic, and a smoke test of `main()`.
- Research report committed at `docs/superpowers/research/2026-04-26-calibration-diagnosis.md` identifying 3 root causes (zero-inflation under GAMMA, heavy tails on `*_yards`, pervasive heteroscedasticity).
- TODO #22 (Plan 3e — calibration tightening) stays open; the diagnostic surfaces root causes but does not implement fixes.

**Plan 3d delivered:**
- New `src/projections/aggregation/season.py`: `aggregate_to_season` real Monte Carlo season aggregator (regenerates per-week samples from the per-row seed; modal-position resolution for traded players).
- New `src/projections/distributions/codec.py`: `pack_per_stat_params` / `unpack_per_stat_params` codec for `ProjectionWeeklySchema.params`.
- `derive_row_seed` in `src/projections/scoring/score_distribution.py`: stable 32-bit per-row seed via sha256 of `(gsis_id, season, week, ruleset.name)`. Consumed by both `predict_distribution` and `aggregate_to_season`.
- `BaselineModel.predict_distribution` now writes per-row seeds + per-stat params blob.
- New `DistributionFamily.SAMPLED_SUMMARY` enum value; new `ProjectionSeasonSchema`.
- Season-calibration metrics (`season_calibration_p10p90`, `season_calibration_le_p90`) wired into the harness; pinned to `calibration_absolute` tolerance classifier. Snapshot expanded from 368 → 400 rows.
- `scripts/backtest.py` writes per-row + per-player results to `data/backtest/run_<ts>/`.
- Default-on smoke asserts season metrics are present and finite.
- Full gate runtime: 292.73s. Drift on the 368 existing weekly rows is within tolerance for every cell (max abs delta 0.0084 vs 0.03 tolerance; max rel delta +0.165% vs 5% tolerance). Cause: per-row seed change reorders Monte Carlo draws; underlying regression math unchanged.
- Season `[p10, p90]` coverage well below target (0.30–0.55 vs 0.80) — inherits 3c's weekly under-dispersion. Plan 3e is the calibration-tightening follow-up.
- TODO #13 (per-row seeds), TODO #14 (SAMPLED_SUMMARY family), TODO #19 (gate non-determinism by demonstration) closed.
- TODO #22 filed (Plan 3e — calibration tightening).

## Next action

**Plan 8 — adoption gate redesign — in progress on `feat/plan-8-gate-redesign`.**
Diagnosis 2026-04-29 traced the PR-10-through-PR-15 model losing streak
(Plans 3e / 5 / 5b / 5c / 7 / 6 — all failed §1.3) to two compounding causes:
(1) §1.3 thresholds sit below the per-cell noise floor, so a model that's
better in expectation routinely fails from sampling variation alone — Plan 6
hit 12/16 RMSE wins but failed because TE 2024 was 0.24pp over the
+1.0% no-regression line; (2) the weekly `[p10, p90]` calibration metric
isn't load-bearing for any planned downstream consumer (Draft Hub,
start/sit, DFS all consume mean and rank, not coverage). See the Plan 8
entry at the top of this file for full context. Plan 8 ships a redesigned
gate plus a re-evaluation of existing peers (C, C-tuned, C-NB, D) under it.

**Track 2 — feature-class work, starting with TODO #3 (PBP / EPA features).**
Five model-class swaps on identical features extract the same signal and
hit the same ceiling. The next real RMSE lift (estimated 5-15%) lives in
features. TODO #3 covers play-by-play ingest + the family of opponent-
adjusted EPA / pace / PROE / air-yards features it unlocks. Brainstorm and
spec after Plan 8 lands.

**Followup housekeeping (low-priority):** Model C-tuned is strictly
dominated by Model C-NB on RMSE — TODO #29 captures the pruning when
ready.

After Plans 8 + the feature-class work: Plan 4 (public API + CLI verbs +
free-tier hosting), then Draft Hub.

---

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-25 | Plan 3a held-out year is 2024, not 2025 | `nfl_data_py` has not yet published 2025 data despite the simulated date being post-2025-season. Training window shifted to 2018-2023. Architecture unaffected; 3c's walk-forward backtest will revisit. |
| 2026-04-25 | Per-stat independent `RidgeCV` sub-models for Model A | Closest match to spec wording (§3.1); per-stat residuals are debuggable; per-stat-independence assumption is "option D" / TODO #1 territory. |
| 2026-04-25 | `Model` as `typing.Protocol` (not `abc.ABC`); not `@runtime_checkable` | Structural typing matches existing `Distribution` Protocol; no isinstance checks needed in callers. |
| 2026-04-25 | One `BaselineModel` class with per-position factories (`wr_baseline()`, future `qb_/rb_/te_baseline()`) | Minimizes 3a→3b copy; per-position quirks expressed as config (`target_stats`, `feature_columns`, `dist_families`). |
| 2026-04-25 | `model_id = "baseline:<pos>:<8-char-code-hash>:<train-start>-<train-end>"` written into every projection row | Stable, reproducible, traceable. Persisted into `ProjectionWeeklySchema.model_id` so we always know which model produced which projection. |
| 2026-04-25 | `code_hash` covers 8 source files | `models/base.py`, `models/baseline.py`, `features/wr.py`, `features/_shared.py`, `features/_rolling.py`, `features/_opponent.py`, `scoring/score.py`, `scoring/score_distribution.py`. Anything whose change should invalidate the artifact. |
| 2026-04-25 | Method of moments for gamma α with clip to `[0.01, 100]` | Closed-form; MLE via `scipy.optimize` is a follow-up if calibration is bad. Plan 3a's calibration is borderline (70.8% in [p10, p90]) — TODO note for 3c. |
| 2026-04-25 | Greek letters in source converted to ASCII (`alpha`, `mu`) | Ruff RUF002/RUF003 flag Greek letters as ambiguous-unicode. Spec/plan markdown can keep them; source files use ASCII transliterations. |
| 2026-04-25 | Per-row sample seed in `score_distribution` is fixed at `42` for v1 | Documented in `predict_distribution` docstring + TODO #13. Cross-row sample correlation; fine for per-row stats; matters when callers combine samples (DFS lineup variance). Defer fix to Plan 3c or DFS work. |
| 2026-04-25 | `family="SAMPLED"` but `params` is summary-only blob | Documented in `predict_distribution` docstring + TODO #14. Per-row p-quantile columns carry the actual distributional info. Decide between SAMPLED_SUMMARY enum value vs. full samples blob before Plan 3c's backtest output consumes the rows. |
| 2026-04-25 | WR builder's traded-player fix: dedupe shares to highest share per gsis_id | v1 hack documented inline + TODO #15. Proper fix restructures `trailing_n_share_in_group` to expose team, lets callers join on (gsis_id, team). Tackle in Plan 3b. |
| 2026-04-25 | TODO #15 closed before Plan 3b kickoff: helper returns `[gsis_id, team, share_l<n>]`; WR/RB/TE builders join on `(gsis_id, team)` | Picks the share for the player's depth-chart-current team — semantically more correct than the v1 highest-share proxy and removes the dedupe hack. RB/TE builders inherit the fix automatically when 3b trains them on real data. |
| 2026-04-25 | TODO #8 closed before Plan 3b kickoff: opt-in `pytest -m network --run-network` smokes per ingest source | One smoke per source (weekly_stats, depth_charts, ngs × 3 stat_types, schedules, id_map, snap_counts) asserts every raw column the normalize step depends on is present, then runs normalize end-to-end so pandera surfaces dtype drift. Post-bump procedure documented in `CONTRIBUTING.md`. |
| 2026-04-25 | Plan 3b: BaselineModel gains required `feature_schema` + `code_hash_files` constructor args | Replaces hardcoded WR references; per-position config stays per-factory. Existing 3a artifact unloadable; retrain in Phase 6 (TODO #17 closed). |
| 2026-04-25 | Plan 3b: TE model includes rushing as target stat (Taysom Hill) | Q3 brainstorm decision; Phase 1 added `rushing_*_per_game_l4` to `TeFeaturesSchema` and `build_te_features`; cost is two columns and a fixture row. |
| 2026-04-25 | Plan 3b: NORMAL/GAMMA convention extended mechanically; POISSON deferred | WR's family choices carry to QB/RB/TE without per-position tuning. POISSON for low-mean integer counts (interceptions, fumbles_lost) deferred to 3c contingent on calibration evidence. |
| 2026-04-25 | Plan 3b: centralized `POSITION_DISPATCH` registry in `models/__init__.py` | One canonical "what positions the system knows about" answer. Reused by CLI scripts and future 3c backtest harness. Adding a position is one new line. |
| 2026-04-25 | Plan 3b: per-position test files (mirrors `tests/test_features/`) | Q6 brainstorm decision. Six new files; failure isolation per position is worth ~210 lines of necessary duplication. |
| 2026-04-25 | Plan 3b: smoke test parametrized across all four positions | Q6 brainstorm B; catches "I broke RB silently" earlier than the per-position test files. ~20s smoke runtime acceptable. |
| 2026-04-25 | Plan 3b: three WR-specific scripts deleted; replaced by position-arg-driven generalized scripts | Q1 brainstorm C. Avoids producing four near-duplicate scripts after 3b. |
| 2026-04-25 | Plan 3b real-data drift: `*_yards_per_game_l4` schema bound dropped to allow negative trailing means | Underlying weekly_stats yards columns allow negative values (sacks/TFL/kneels); commits `fa864ac` and `e25eb57` relax the bound on the trailing means and on `passing_yards_per_game_std`. |
| 2026-04-25 | Plan 3b real-data drift: bye-week + dedupe filters ported from WR to QB/RB/TE | WR had these in 3a (TODO #9a, #9c); QB/RB/TE feature builders inherit the same shape. Commits `f79806a` (bye filter) and `54b6d95` (dedupe). |
| 2026-04-26 | Plan 3c gate is opt-in `pytest -m backtest --run-backtest`, not default-on | A full gate run is ~2 minutes; default-on adds material drag to every dev iteration. Default-on smoke covering one (WR, 2024) cell catches harness wiring bugs cheaply. |
| 2026-04-26 | Snapshot at (position, year, metric) granularity (368 rows); per-metric-type tolerances | Per-year visibility is the whole point of multi-year backtest; aggregating loses the "regressed only on 2022" signal. Tolerances grouped by metric type keeps maintenance low; per-row overrides added empirically as we observe noise. |
| 2026-04-26 | Held-out years 2021-2024 (skip 2019 / 2020) | 2019's 1-season train window is too small; 2020 is COVID-shortened structural outlier. Each held-out year has at least 3 seasons of training history. |
| 2026-04-26 | Plan 3c uses summed weekly means as season totals (degenerate aggregation); real Monte Carlo aggregation deferred to Plan 3d | Decouples gating infrastructure from season-distribution design. Plan 3d converges TODOs #13 / #14 and calibration tightening. |
| 2026-04-26 | Feature cache invalidation is manual via `scripts/refresh_features.py`; auto-invalidation deferred (TODO #21) | Manual is documented in CONTRIBUTING.md and produces a clear FileNotFoundError pointing at the refresh command. Auto-invalidation via code-hash is straightforward but adds surface area; defer until manual produces a real-world bug. |
| 2026-04-26 | `score_distribution` perf vectorization pulled forward from Plan 3d into Plan 3c | Spec section 1.2 deferred the perf TODO under "feature caching means we predict once per (player-week, year), not per training fold." Phase 6 demonstrated this was wrong: the per-sample Python loop still dominated at 20-30 minutes for the full harness. Math is bit-identical (linear scoring rule); fix is mechanically safe. |
| 2026-04-29 | Pivot to gate redesign (Plan 8) before any further model-improvement work; feature-class track (TODO #3 PBP / EPA) is the next-up modeling lift after Plan 8 lands | PR-10-through-PR-15 diagnosis: §1.3's per-cell thresholds are below the noise floor (Plan 6 failed by 0.24pp on a single noisy cell after winning 12/16 RMSE) and the calibration criterion isn't load-bearing for any planned downstream consumer. Five model-class swaps on identical features hit the same information ceiling — feature work is the next real lift, not another model class. |

---

## Plan 2b — historical (as of 2026-04-24)

**Projections Core — Plan 2b (QB/RB/TE feature builders) merged to `main` at commit `af325ea`.**

**Predecessors:**
- Plan 1 (Foundations) merged at `8f02a6c`.
- Dev tooling merged via `feat/dev-tooling`.
- Plan 2a (Ingest expansion + WR feature builder) merged at `7926090`.

**Plan 2b delivered:**
- `build_qb_features`, `build_rb_features`, `build_te_features` — pure-function builders mirroring `build_wr_features`'s shape.
- Three new feature schemas (`QbFeaturesSchema`, `RbFeaturesSchema`, `TeFeaturesSchema`).
- `WeeklyStatsSchema` extended with `attempts`, `completions`, `sacks` for QB features.
- Generalized `trailing_n_share_in_group` helper in `_rolling.py` (migrated from `wr.py`'s local helper).
- ~45 new tests (~200 total). 5 leakage tests per position (15 new).

---

## Next action

**Recommended: Plan 3 — Model A baseline + season aggregation + first-class backtest harness.**

All 4 offensive skill positions (QB/RB/WR/TE) now have feature builders. Plan 3 trains the v1 model per position, aggregates weekly outputs to season distributions (Monte Carlo with bye + availability), and stands up the backtest harness that gates future model changes.

K and DST builders (TODO #10) can land in parallel with Plan 3 — they're independent.

---

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-24 | `rushing_qb` boolean threshold = 5.0 carries/game over trailing 4; `passing_down_back` = 4.0 targets/game | Rough heuristics from feel. Not load-bearing; revisit at backtest time if categorization matters |
| 2026-04-24 | TE target_share denominator includes WR + RB + TE (full pass-catching group) | TEs usually have only one fantasy-relevant player per team, so same-position-share would be ~1.0 and useless. Full-group share captures meaningful gradient |
| 2026-04-24 | RB target_share denominator includes WR + RB + TE (full pass-catching group) | A workhorse RB getting 5 targets/game on a 30-target offense is meaningfully different from one getting 5 on a 20-target offense. Full-group denominator captures team passing volume, not just RB-on-RB share |
| 2026-04-24 | Migrate `_trailing_4_share_per_team` from `wr.py` to `_rolling.py` as `trailing_n_share_in_group` | RB needs target_share against the full pass-catching group (not just RBs); TE needs the same. Generalize once, in the shared helper module, rather than duplicate in three builders |
| 2026-04-24 | Extend `WeeklyStatsSchema` with `attempts`, `completions`, `sacks` | QB feature builder needs these source columns. All three are present in raw `nfl_data_py.import_weekly_data` output. Same incremental pattern as 2a's extension for `targets`/`carries`/`receiving_air_yards` |
| 2026-04-24 | One bundled PR for QB/RB/TE (not three per-position PRs) | Repetitive, interlinked work; reviewing all three together catches drift. Each position lands as its own commit inside the bundle for easy retrospection |
| 2026-04-24 | All 4 position builders use parallel files (no WR/TE shared base) | Each position's feature list will diverge over time as we add play-by-play-derived features. Premature DRY hurts later. Shared logic lives in `_rolling.py` / `_opponent.py` / `_shared.py` (the latter added 2026-04-25 in PR #4 cleanup, hoisting `prior_mask` / `exact_week_mask` / `build_game_environment` out of `wr.py`) |
| 2026-04-24 | K and DST split out into a future plan; 2b covers QB/RB/TE only | K needs FG-attempt data not in `WeeklyStatsSchema`; DST is team-level not player-level and needs play-by-play. Both should wait for the data they need rather than ship degraded v0 features |
| 2026-04-24 | `nfl_data_py.import_snap_counts` returns `pfr_player_id` not `gsis_id`; ingest joins on id_map | Discovered during fixture-construction (Task 8). Snap_counts ingest now reads id_map.parquet and inner-joins pfr_id → gsis_id; bench/practice players with no id_map match are dropped silently |
| 2026-04-24 | `spread_line` from `nfl_data_py` is positive when home favored (inverts standard sportsbook) | Discovered during code review of Task 15. Empirically verified against import_schedules([2023]). `_build_game_environment` in features/wr.py uses the empirically-correct convention; team-perspective `spread` follows standard "favorite is negative" |
| 2026-04-24 | Split Plan 2 into 2a (ingest expansion + WR feature builder) and 2b (QB / RB / TE / K / DST feature builders) | Validate the feature-builder pattern end-to-end on one position before copy-pasting across five files; isolate ingest (mechanical) from features (greenfield design) |
| 2026-04-24 | WR is the first end-to-end position | Exercises every new ingest source (snap_counts, depth_charts, NGS receiving) in one builder; surfaces design issues before propagating to other positions |
| 2026-04-24 | Feature builders are pure functions in 2a — no parquet storage | Output is small (~1.8K rows/season for WR) and computes in milliseconds; defer caching until backtest performance demands it (Plan 3+) |
| 2026-04-24 | Ingest all three NGS stat types (passing, rushing, receiving) in 2a, even though only NGS receiving is consumed by WR | The hard part of NGS ingest is the snapshot/partition decision; make it once across all three rather than three times |
| 2026-04-24 | Opponent strength via `opp_allowed_fppg_l4` proxy in 2a, not play-by-play EPA | True EPA needs play-by-play ingest (separate concern, deferred); the FPPG-allowed proxy is sufficient for v1 baseline |
| 2026-04-24 | Shared `_rolling.py` and `_opponent.py` helpers built and tested in 2a | Pin helper API on the first builder so 2b's five other builders consume a stable contract |
| 2026-04-24 | Schedule ingest captures Vegas lines (spread, total, moneyline) | "Implied team total" is a load-bearing feature for every offensive position |
| 2026-04-24 | Drive-by cleanups (`_PYARROW_STR` to `schemas.py`, programmatic `_INTEGER_STATS`, ingest `__all__`) folded into 2a | We're touching every ingest module anyway; cheaper to clean up once than across two PRs |
| 2026-04-24 | Extend `WeeklyStatsSchema` with `targets`, `receiving_air_yards`, `carries` | Discovered during plan-writing: WR feature builder needs these source columns and the foundations-era schema didn't include them. All three are present in raw `nfl_data_py.import_weekly_data` output |
| 2026-04-24 | Test fixtures are synthetic in-memory `pd.DataFrame`s, not real-data parquet snapshots | Matches existing convention from foundations (`fake_weekly_df` etc.); simpler maintenance; `nfl_data_py` API drift is handled separately by opt-in network smoke tests (TODO #8) |
| 2026-04-24 | Decompose project into 4 sub-projects (Projections Core, Draft Hub, Mid-season Manager, DFS Engine) | Each subsystem has different consumer logic; shared dependency is a probabilistic projection engine. Keeps any single design doc executable. |
| 2026-04-24 | Build Projections Core first | Earliest dependency for everything else. |
| 2026-04-24 | `nfl_data_py` as primary data source | Free, comprehensive, modern; Python-native. Paid feeds (PFF, FantasyPros API) deferred until we've validated need. |
| 2026-04-24 | Full per-player distributions (option C from brainstorming), not point estimates | Subsumes point estimates for free; required for DFS GPP work later. Joint correlations (option D) deferred to TODO #1 — schema designed so D is additive. |
| 2026-04-24 | Weekly model as foundation; season aggregates as derived layer | Weekly is where play-by-play signal lives; season is Monte Carlo aggregation with bye + availability. |
| 2026-04-24 | A → C → D modeling roadmap | Baseline regression first (Model A) to establish data pipeline + backtest harness; gradient boosted (Model C) only if it beats baseline; ensemble (Model D) reserved for last. |
| 2026-04-24 | Strong typing posture: pandera schemas at module boundaries, pydantic models for configs/records, NewType per ID flavor, mypy strict, enums for every reused string-keyed concept | User had prior pain with stringly-typed/dict-laden code. Catch errors at boundaries, not three modules deep. |
| 2026-04-24 | Parquet + DuckDB storage | Friendly to free-tier hosting (Streamlit Community Cloud, HF Spaces, DuckDB-WASM in browser). |
| 2026-04-24 | Subagent-driven execution for foundations plan | Faster iteration, fresh context per task, two-stage review (spec then code quality) at higher-risk tasks. |
| 2026-04-24 | Pre-commit hooks (ruff lint+format, mypy, housekeeping); no GitHub Actions CI; pytest manual before PR | Catches the regressions that matter at commit time without slowing commits with full pytest. CI deferred indefinitely per user direction. |
| 2026-04-24 | No direct commits to `main` — specs, plans, and implementation all on feature branch via PR | User correction after I committed a spec to main. Conventions encoded in CONTRIBUTING.md and CLAUDE.md. |
| 2026-04-24 | `CLAUDE.md` trimmed; `CONTRIBUTING.md` is the deep contributor doc | CLAUDE.md auto-loads into Claude's context every interaction; every line costs context budget. Detail moves to CONTRIBUTING.md. |

---

## Backlog (longer-term)

Roughly in order. Each is its own brainstorm → spec → plan cycle.

### Projections Core (remaining)

- **Plan 2** — Ingest expansion (schedules, snap_counts, depth_charts, NGS) + per-position feature builders.
- **Plan 3** — Model A baseline (per-position regressions) + season aggregation (Monte Carlo with bye + availability) + first-class backtest harness.
- **Plan 4** — Public Python API + CLI verbs (`refresh`, `project`, `backtest`, `query`) + free-tier web hosting setup (likely Streamlit on Community Cloud).
- **Plan 5** — Model C (LightGBM with quantile regression). Adopt only if it beats Model A on the backtest harness. Detailed scope in TODO #26. One of three model-improvement tracks identified post-Plan-3e (alongside TODO #3 PBP features and TODO #23 target decomposition).

### Subsequent sub-projects

- **Draft Hub** — pre-draft rankings, ADP, tier breaks, VORP, mock-draft sim, live draft assistant (consumes Projections Core + ESPN league API).
- **Mid-season Manager** — weekly start/sit, waiver-wire valuator, trade analyzer, schedule strength.
- **DFS Engine** — slate projections, ownership, salary-constrained lineup optimizer, multi-lineup portfolio. Triggers TODO #1 (joint correlations) work.

### Cross-cutting

- **TODO #1** — option D exploration: joint-correlation projections (covariance / scenario sim / factor / copula). Decide before DFS Engine.
- **`score_distribution` vectorization** — TODO marker in code; needed before backtest scale (~85M Pydantic instantiations otherwise).
- Minor cleanups from foundations review: `_PYARROW_STR` to `schemas.py`, programmatic `_INTEGER_STATS`, drop ingest helpers from `__all__`.
- ESPN league API integration (year-long league sync). Belongs in Draft Hub / Mid-season Manager sub-projects.
- Pyarrow strings everywhere story: pandera 0.31 enforces `string[pyarrow]` for `Series[str]`. Consider whether a future schema or storage shift makes this implicit rather than per-module.
