# RB Rushing + Receiving Decomposition Probe — Design

**Status:** draft (brainstorming, 2026-05-16). Ready for user review.
**Date:** 2026-05-16
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Projections Core
**Branch:** `feat/probe-rb-decomposition` cut from `origin/main` at `9c36507` (PR #44 merge commit).

**Builds on:**
- WR Receiving Target Decomposition Probe (PR #32, merged 2026-05-10). Canonical template. Receptions ADOPT (SIGNAL); receiving_yards / receiving_tds NULL. Spec at `docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md`. This RB probe mirrors PR #32's architecture, generalized to two shared volume axes (carries for rushing; targets for receiving) and 5 composed stats.
- Tweedie yards_per_target Probe (PR #44, merged 2026-05-16). Established that factor-class swaps on the WR receiving efficiency factor are NULL on this dataset. Reinforces the "test decomposition itself before testing factor-appropriate sub-models" sequencing rule from PR #32 §1.2.
- Logit catch_rate Probe (PR #39, merged 2026-05-16). Same — factor-class swap on WR catch_rate NULL. Two consecutive NULLs at the factor-class axis. The decomposition-recipe axis (this PR's territory) has more open prior.

---

## 1. Goals & success criteria

### 1.1 Goal

Probe whether decomposing RB stats into **two shared volume axes × per-stat efficiency factors** beats the current `BaselineModel`'s direct per-stat RidgeCV on out-of-sample mean prediction. Five composed predictions are tested across two volume axes:

| Stat | Volume factor | Efficiency factor | Trained on |
|---|---|---|---|
| `rushing_yards` | `carries` | `yards_per_carry` | RB rows with `carries > 0` |
| `rushing_tds` | `carries` | `td_rate_per_carry` | RB rows with `carries > 0` |
| `receptions` | `targets` | `catch_rate` (= receptions / targets) | RB rows with `targets > 0` |
| `receiving_yards` | `targets` | `yards_per_target` | RB rows with `targets > 0` |
| `receiving_tds` | `targets` | `td_rate_per_target` | RB rows with `targets > 0` |

Two shared volume sub-models fit once per training window (`carries ~ X`, `targets ~ X`); five efficiency sub-models fit on their respective non-zero-volume subsets; five direct comparators identical to today's `BaselineModel.fit` recipe for the five composed stats.

The architectural hypothesis is the same as PR #32: volume (team-/scheme-driven) and efficiency (player-driven) carry distinct signal axes that a single combined Ridge cannot fully extract. This probe extends PR #32 to RB along two new dimensions: (a) a rushing-specific volume axis (`carries`) that has no analog in PR #32, and (b) a generalization test of the WR receiving decomposition on a structurally different sample (RB receivers).

**Single sub-model class: RidgeCV everywhere** with predict-time clipping per PR #32 §1.2. Any SIGNAL is attributable to *decomposition itself*, not to a model-class change. Factor-appropriate sub-models (Poisson on carries, log-link Gamma on yards_per_carry, logistic on catch_rate / td_rate_per_carry / td_rate_per_target, Tweedie on yards_per_target) are explicitly out of scope — each is a separate probe + integration cycle conditional on this probe's verdict.

### 1.2 Architectural prior

Two structural differences from WR receiving (PR #32) that affect the prior:

1. **No completions-count analog on rushing.** WR's strongest decomposition win was `receptions` (the count) — the [0, 1]-bounded `catch_rate` efficiency is clean, and the volume signal (`targets`) is fully observed. Rushing has no direct analog: every carry generates yardage (positive or negative); there is no "incomplete carry" axis. So the structural argument that produced WR's only ADOPT does not transfer to rushing. The rushing prior leans on a different mechanism — `carries` is more team/scheme-driven than `targets` (target share encodes player "trust"; carry distribution encodes game-script and rotational decisions), and `yards_per_carry` has a broader tail than `yards_per_target` (long runs are more dramatic than long catches). Whether those two effects compound into a Ridge-decomp win is what this probe tests.

2. **RB receiving may NOT mirror WR receiving.** WR's PR #32 verdict was receptions ADOPT, receiving_yards / receiving_tds NULL. RBs catch passes differently — shorter average routes, more checkdowns, higher catch_rate baseline (~0.75 for RBs vs ~0.65 for WRs), and a smaller sample size per RB (RBs average fewer targets/game than WRs). Whether the receptions-decomposition signal generalizes from WR to RB is an open empirical question; testing it on RB data is the only way to know.

**Magnitude prior.** Plausibly 1–5% per-stat RMSE on the rushing factors (similar to or slightly smaller than WR target decomposition's per-stat magnitudes); ~0–3% on RB receiving (smaller samples than WR; smaller signal). NULL is plausible on multiple stats; SIGNAL on at least one is plausible. The probe is genuinely uncertain.

### 1.3 Success criteria

The probe **ships** when all four pass. Per-stat verdicts (SIGNAL / NULL / REGRESSION) are informational, not ship gates.

1. **Coverage:** RB rows with `carries > 0` per eval year ≥ 0.95 AND RB rows with `targets > 0` per eval year ≥ 0.95, across all four eval years 2021–2024. Default `--coverage-threshold 0.95`. **Realistic concern:** pass-catching RBs (James-White-style 3rd-down backs) and 3rd-string backs have 0-carry weeks; the `carries > 0` rate may be materially below 0.95 on the RB population. Symmetrically, run-only backs may have 0-target weeks. Relaxation on either axis triggers the PR #31 retrospective rule (probe binding-cell magnitudes <0.005 fpts composite-fpts under coverage relaxation should be treated as MARGINAL, not SIGNAL). The report must surface per-volume-axis per-eval-year coverage rates.

2. **Probe completeness:** Walk-forward over eval years `{2021, 2022, 2023, 2024}`; per-stat residual buffers populated for both arms (direct, decomposed) on every eval year; pooled paired-bootstrap CI rendered per stat. Five per-stat reports (`rushing_yards`, `rushing_tds`, `receptions`, `receiving_yards`, `receiving_tds`) plus one summary report and one machine-readable CSV.

3. **Per-stat verdict mapping** (matches PR #32 / PR #39 / PR #44):
   - **SIGNAL** iff `rmse_delta.hi_95 < 0` (decomposed CI strictly negative).
   - **REGRESSION** iff `rmse_delta.lo_95 > 0` (decomposed CI strictly positive).
   - **NULL** otherwise (CI brackets zero).
   - No effect-size floor at the per-stat level — per-stat RMSE units are heterogeneous (~20 yards for rushing_yards, ~0.3 for rushing_tds, ~1.5 for receptions, ~15 for receiving_yards, ~0.2 for receiving_tds). Pure CI-based verdict mirrors PR #32's PerStatVerdict logic.

4. **Verification gates green:** `mypy src tests scripts` strict + `ruff check src tests scripts` + `ruff format --check src tests scripts` clean. Relevant pytest subset green (probe module + walk-forward harness tests + CLI smoke).

### 1.4 Out of scope (deferred follow-ups)

1. **Composite-fpts measurement.** Per-stat RMSE only. Composite-fpts is the integration-adoption-gate concern (`(DecomposedBaselineModel-for-RB, RB) vs (BaselineModel, RB)` if any stat SIGNALs).
2. **Within-row coherent factor sampling.** Shared per-row `carries` and `targets` draws flowing into composed `SampledDistribution`s. Integration concern; lifts composite-fpts CI accuracy, not per-stat mean RMSE.
3. **Factor-appropriate sub-model classes.** Poisson on `carries`; log-link Gamma or Tweedie on `yards_per_carry`; logistic on `catch_rate` and `td_rate_per_carry` / `td_rate_per_target`; Tweedie on `yards_per_target`. Each a separate probe + integration cycle. PR #39 / PR #44 closed two of these on WR with NULL verdicts; the priors on the RB-side factor-class probes are correspondingly weakened, but they remain independent tests.
4. **`fumbles_lost` and other rare events.** `fumbles_lost` is in `_RB_TARGET_STATS` but has no clean volume axis (fumbles can come from carries, receptions, or returns; the rate is tiny and zero-inflated). Out of scope; integration plan keeps it direct.
5. **RB-specific receiving-quality features.** Target share, route-depth, two-back vs single-back splits. Feature-engineering territory, not architecture. Separate spec.
6. **New `DecomposedBaselineModel` registration for RB / `_RB_FACTORIES` updates.** The existing `DecomposedBaselineModel` peer is already infrastructure (PR #36 shipped it for WR receptions). If RB SIGNALs, the integration plan reuses the same class with a different `decomposed_stats` mapping. No code touched in this probe.
7. **Other positions.** TE rushing / QB rushing / QB passing. Each its own probe + integration cycle.
8. **RB rushing-receiving correlation features.** RBs who catch more may also run more (or vice versa, depending on scheme). A joint volume model is conceivable but adds complexity. Out of scope.

---

## 2. Source data — already ingested

All inputs already exist in the feature cache and weekly stats:

| Column | Type | Source | Use |
|---|---|---|---|
| RB features | `RbFeaturesSchema` | `data/features/rb/season=YYYY/week=WW/part.parquet` | X for the 7 ridges (2 shared volume + 5 efficiency) and the 5 direct comparators. |
| `carries` | `Series[int]` | `WeeklyStatsSchema.carries` (`ge=0, le=50`) | y for the carries volume sub-model; filter (`> 0`) for rushing efficiency. |
| `targets` | `Series[int]` | `WeeklyStatsSchema.targets` | y for the targets volume sub-model; filter (`> 0`) for receiving efficiency. |
| `rushing_yards` | `Series[float]` | `WeeklyStatsSchema.rushing_yards` (`ge=-50, le=400`) | y for `yards_per_carry` and the rushing_yards direct comparator. Note: schema permits negative values (TFL / fumble recoveries / kneels); Ridge handles negative y natively. |
| `rushing_tds` | `Series[int]` | `WeeklyStatsSchema.rushing_tds` (`ge=0, le=10`) | y for `td_rate_per_carry` and the rushing_tds direct comparator. |
| `receptions` | `Series[int]` | `WeeklyStatsSchema.receptions` | y for `catch_rate` and the receptions direct comparator. |
| `receiving_yards` | `Series[float]` | `WeeklyStatsSchema.receiving_yards` | y for `yards_per_target` and the receiving_yards direct comparator. |
| `receiving_tds` | `Series[int]` | `WeeklyStatsSchema.receiving_tds` | y for `td_rate_per_target` and the receiving_tds direct comparator. |

**No new ingest, no schema changes, no override parquet.** The probe consumes the existing per-week RB feature parquet and joins to `weekly_stats` on `(gsis_id, season, week)` exactly as `BaselineModel.fit` does for the RB position dispatch.

---

## 3. Architecture

### 3.1 New module — `src/projections/backtest/rb_decomposition_probe.py`

Mirrors `src/projections/backtest/target_decomposition_probe.py`'s shape, generalized to two volume axes and five composed stats. Pure numpy / pandas / sklearn. Reuses `paired_bootstrap_rmse_delta` and `BootstrapDelta` from `src/projections/backtest/adoption_gate.py` unchanged.

```
rb_decomposition_probe.py
├── _RB_DECOMPS — frozen mapping of 5 Stat -> _StatDecomp(volume_stat, efficiency_label,
│                                                          efficiency_clip_hi, numerator_stat)
│       # {RUSHING_YARDS: (CARRIES, "yards_per_carry", +inf, RUSHING_YARDS),
│       #  RUSHING_TDS:   (CARRIES, "td_rate_per_carry", 1.0, RUSHING_TDS),
│       #  RECEPTIONS:    (TARGETS, "catch_rate", 1.0, RECEPTIONS),
│       #  RECEIVING_YARDS: (TARGETS, "yards_per_target", +inf, RECEIVING_YARDS),
│       #  RECEIVING_TDS: (TARGETS, "td_rate_per_target", 1.0, RECEIVING_TDS)}
├── _fit_direct(x, y) -> RidgeCV
│       # identical recipe to BaselineModel.fit per stat (no clipping at fit or predict).
├── _fit_decomposed_volume(x_all, volume_y) -> RidgeCV
│       # shared volume sub-model; trained on all train rows (no filter).
│       # Called twice per train window: once for carries, once for targets.
├── _fit_decomposed_efficiency(x_all, numerator, volume) -> RidgeCV
│       # ratio = numerator / volume on rows with volume > 0.
│       # Raises ValueError if no rows have volume > 0 (caller-side guard).
├── _predict_direct(ridge, x) -> np.ndarray
├── _predict_decomposed(volume_ridge, efficiency_ridge, x, efficiency_clip_hi) -> np.ndarray
│       # mu = clip(volume.predict, 0, +inf) * clip(efficiency.predict, 0, efficiency_clip_hi)
├── walk_forward_residuals(features, weekly_stats, eval_years) -> WalkForwardOutput
│       # Per train window: fit 2 shared volume Ridges (carries, targets),
│       #                   5 efficiency Ridges (per _RB_DECOMPS),
│       #                   5 direct comparator Ridges.
│       # Per eval year: per-row residuals for each of 5 stats x 2 arms.
│       # Concatenates per-year buffers into pooled arrays.
└── compute_verdicts(output, *, n_bootstrap, seed) -> list[PerStatVerdict]
        # One PerStatVerdict per of the 5 stats; paired-bootstrap CI on
        # (residuals_decomposed - residuals_direct) per stat.
```

`WalkForwardOutput` is a frozen dataclass with per-stat `StatResiduals` (matching PR #32's shape: `actual`, `mu_direct`, `mu_decomposed`, `n_paired`) plus per-volume-axis per-year coverage (`coverage_carries_by_year: dict[int, float]`, `coverage_targets_by_year: dict[int, float]`).

`PerStatVerdict` is a frozen dataclass: `stat: Stat`, `n_paired: int`, `rmse_delta: BootstrapDelta`, `verdict: VerdictLabel` where `VerdictLabel = Literal["SIGNAL", "NULL", "REGRESSION"]`. Matches PR #32's PerStatVerdict shape.

### 3.2 Sub-model fitting details

- `alphas` matches `BaselineModel.fit` via `from projections.models.baseline import _RIDGE_ALPHA_GRID` (canonical import — PR #44 fix). Aliased locally as `_RIDGE_ALPHAS: Final[np.ndarray] = _RIDGE_ALPHA_GRID`.
- Feature columns match `_RB_FEATURE_COLUMNS` from `projections.models.baseline`. Same bool-to-int8 coercion as `BaselineModel`'s `_x_frame_with_bool_coercion`. Train-time NaN-row drop matches `BaselineModel.fit`. Predict-time imputation uses train-set medians.
- Carries volume sub-model `y = weekly_stats["carries"]` (int, ≥0). Targets volume sub-model `y = weekly_stats["targets"]` (int, ≥0). Both volume sub-models train on the **un-filtered** RB train rows (zero-volume rows are legitimate observations of low-volume / out-of-rotation players).
- Efficiency-factor filter: per-volume-axis subset of train rows. For rushing efficiency (`yards_per_carry`, `td_rate_per_carry`): rows where `carries > 0`. For receiving efficiency (`catch_rate`, `yards_per_target`, `td_rate_per_target`): rows where `targets > 0`. The two subsets are different row sets; the probe maintains them separately.
- Ratio computation: `ratio = numerator / volume` on the volume > 0 subset, where the denominator is non-zero by filter construction.

### 3.3 Walk-forward harness

For each eval year Y in `{2021, 2022, 2023, 2024}`:

1. **Train rows:** RB feature-cache ∩ RB weekly_stats inner-join on `(gsis_id, season, week)`, season ∈ [2018, Y-1].
2. **Eval rows:** same intersection, season = Y.
3. **Fit on train rows:**
   - 2 shared volume RidgeCVs: `carries ~ X`, `targets ~ X`.
   - 5 efficiency RidgeCVs: one per entry in `_RB_DECOMPS`, fit on the volume > 0 subset for the entry's volume_stat.
   - 5 direct comparator RidgeCVs: one per composed stat, identical recipe to today's `BaselineModel`.
   - Persist train-set feature medians for predict-time imputation.
4. **Predict per-row mu on eval rows for both arms:**
   - direct: `mu_direct[stat] = direct_ridge[stat].predict(X_eval)` (no clip).
   - decomposed: `mu_decomposed[stat] = clip(volume_ridge.predict(X_eval), 0, +inf) * clip(efficiency_ridge[stat].predict(X_eval), 0, _RB_DECOMPS[stat].efficiency_clip_hi)`.
5. **Append per-row `(actual, mu_direct, mu_decomposed)` to per-stat residual buffers** (one buffer per of the 5 stats). Record per-year coverage on each volume axis: `(carries > 0).mean()` and `(targets > 0).mean()` on the eval rows.

After the loop: pool residuals across 4 years per stat (no row weighting). Pooled `n_paired` per stat = sum across years. Two coverage dicts (one per volume axis) emitted in the WalkForwardOutput.

### 3.4 Verdict + bootstrap CI

For each of 5 stats:

1. `rmse_delta = paired_bootstrap_rmse_delta(residuals_direct, residuals_decomposed, n_bootstrap=1000, seed=42)` where `residuals_arm = actual - pred_arm`. Returns `BootstrapDelta(point, lo_95, hi_95, ...)`. The helper computes (cand − inc) RMSE under our convention; sign convention matches PR #32 / PR #39 / PR #44.
2. Verdict mapping: `hi_95 < 0 → SIGNAL`; `lo_95 > 0 → REGRESSION`; else `NULL`.
3. Per-stat reports stitched into the summary with verdict, point, CI, n_paired, per-volume-axis per-year coverage, and the magnitude-flag conditional (informational only — surfacing when |Δ_fpts_equiv| < 0.005, per PR #31 retrospective).

Seed: 42 (matches PR #44 / PR #39 default). Documented in CLI args.

### 3.5 CLI — `scripts/probe_rb_decomposition.py`

Mirrors `scripts/probe_target_decomposition.py`. argparse flags:

```
--eval-years YEARS         default [2021, 2022, 2023, 2024]; choices in _VALID_YEARS
--features-root PATH       default data/features
--raw-root PATH            default data/raw
--summary-out PATH         default reports/feature_probe_rb_decomposition_summary.md
--csv-out PATH             default reports/feature_probe_rb_decomposition.csv
--coverage-threshold FLOAT default 0.95
--n-bootstrap INT          default 1000
--seed INT                 default 42
```

Reports written with `encoding="utf-8"` (Windows cp1252 guard per PR #39 follow-up). All stdout strings ASCII-only.

### 3.6 No edits to existing code

- `src/projections/models/baseline.py` untouched (probe imports `_RIDGE_ALPHA_GRID` and `_RB_FEATURE_COLUMNS`).
- `src/projections/models/decomposed_baseline.py` untouched (integration plan territory if SIGNAL).
- `src/projections/backtest/target_decomposition_probe.py` / `logit_catch_rate_probe.py` / `tweedie_yards_per_target_probe.py` untouched (siblings; probe is its own module). Reuse via `from projections.backtest.adoption_gate import BootstrapDelta, paired_bootstrap_rmse_delta` only.

### 3.7 No schema, codec, or factory changes

Probe operates in-memory against sklearn estimators fit on numpy arrays. No new Distribution classes, no codec edits, no new `_RB_FACTORIES` registration. The probe is strictly a **mechanism test**.

---

## 4. Decision branches

Per §1.3's per-stat verdicts (5 stats × 3 outcomes = 15 possible per-stat states, but typical patterns):

- **All 5 NULL** — close RB decomposition at the 2-factor unit. Refined decompositions (red-zone-split carries, goal-line carries, snap-share-conditioned efficiency, two-back vs single-back splits) remain open but require independent mechanism evidence before re-probing per PR #31 retrospective rule. None queued.
- **≥ 1 rushing SIGNAL, no REGRESSION** — greenlight integration plan for RB rushing decomposition. Plan must:
  1. Add `Stat.RUSHING_YARDS` and/or `Stat.RUSHING_TDS` (whichever signaled) to `_RB_FACTORIES["decomposed-baseline"]`'s `decomposed_stats` mapping with the appropriate `DecompositionSpec(volume_stat=Stat.CARRIES, efficiency_label=..., efficiency_clip_hi=...)`.
  2. Composite-fpts adoption gate on `(DecomposedBaselineModel-for-RB, RB) vs (BaselineModel, RB)` at production scope.
  3. Per-position routing decision: RB is currently routed to `BaselineModel` (per Plan 8); if the gate SIGNALs at composite-fpts, flip RB's `default_model_class` to `"decomposed-baseline"`.
  4. Factor-appropriate sub-models named as a follow-up cycle (Poisson on carries; Gamma / Tweedie on yards_per_carry; logistic on td_rate_per_carry). PR #39 / PR #44 NULL priors weaken these but do not close them — RB-side factor-class probes are independent tests.
- **≥ 1 RB receiving SIGNAL** — interesting generalization finding from WR. The integration plan can opt in to the SIGNAL stats only (typically `receptions`, if it mirrors WR). The probe-vs-gate calibration narrative for RB receiving should explicitly contrast with WR's PR #32 baseline (composite-fpts magnitude on RB will be smaller because RB receiving carries less variance per game than WR receiving).
- **Mixed SIGNAL + REGRESSION on different stats** — write a tighter follow-up probe; document per-stat verdicts. Most likely cause: a stat-specific data-quality issue (e.g., `td_rate_per_carry` is extremely zero-inflated on most RB-weeks).

---

## 5. Risk register

1. **`carries > 0` coverage below 0.95.** Most likely real concern. Pass-catching backs (Austin Ekeler, James White, Theo Riddick types) have 0-carry weeks; 3rd-string backs likewise. Symmetric concern on `targets > 0` for run-only backs (Henry, Chubb in pure-rushing weeks). **Mitigation:** the CLI emits per-volume-axis per-eval-year coverage rates in the summary report. If either coverage falls below 0.95 for any eval year, the PR #31 retrospective rule fires: per-stat magnitudes affected by relaxation get the MARGINAL flag in addition to the CI-based verdict. The integration plan (if any stat SIGNALs) must weight CI strength against the coverage caveat.

2. **Independent-factor false-confidence.** Decomposition assumes volume and efficiency are orthogonal in the residual sense. If `yards_per_carry` is correlated with feature-implied `carries` (e.g., game-script: blowouts → more carries AND lower yards_per_carry as defenses sell out to stop the run), the multiplicative composition double-counts the suppression and over-shrinks the predicted mean. **Mitigation:** the report includes per-eval-year scatter of (predicted volume residual, predicted efficiency residual) and a Pearson correlation per (stat, volume_axis); |ρ| > 0.2 across years documented as a probe caveat. Same mitigation pattern as PR #32 §5.2.

3. **Predict-time clipping bias.** Clipping efficiency at [0, 1] for rate factors and [0, +∞) for unbounded factors is asymmetric. Ridge predictions in the right tail are not clipped (yards_per_carry can predict 8+ yards/carry even though empirical max is ~7). **Mitigation:** the report includes per-stat clipped-row counts per eval year; if >2% of rows are clipped on any stat, document as a caveat. The factor-appropriate sub-model follow-up addresses this structurally.

4. **`rushing_yards` schema permits negative values.** `pa.Field(ge=-50, le=400)`: real-data laterals, tackles for loss, and fumble recoveries on the line of scrimmage produce negative rushing_yards. Ridge handles negative y natively (Gaussian assumption is unbothered). **Specific implication:** the efficiency-fit row mask is `carries > 0` only — NOT `(carries > 0) & (rushing_yards >= 0)`. Negative-yards rows are valid training data for the Ridge-on-ratio recipe. If this probe SIGNALs and a Tweedie follow-up is greenlit, the Tweedie probe will face the same y >= 0 filter issue as PR #44 (~0.6% of WR rows; the rate on RB rushing is likely higher because TFL is more frequent than 0-yard catches). That's a future-spec concern, not this probe's.

5. **5-stat scope = larger report surface.** The summary report has 5 verdict lines + 5 per-stat sections + 2 coverage tables + the magnitude-flag table. Runtime is still small (5 efficiency RidgeCVs + 5 direct RidgeCVs per fold × 4 folds + bootstrap is sub-minute). **Mitigation:** none needed; the architecture is well-rehearsed from PR #32 and reports are templated.

6. **Probe-vs-gate calibration risk.** Per PR #31 retrospective. Per-stat RMSE Δ translated to expected composite-fpts contribution via the ESPN PPR scoring coefficients (`Ruleset.espn_ppr()`: 1.0 fpt/rec, 0.1 fpt/yd receiving, 0.1 fpt/yd rushing, 6.0 fpt/td receiving or rushing). Composite-fpts magnitude per stat is `per_stat_rmse_delta / yds_per_pt` (for yards) or `per_stat_rmse_delta × pts_per_unit` (for receptions / tds). **Mitigation:** the report includes per-stat composite-fpts equivalent inline next to the yards/receptions/tds delta, and the magnitude-flag fires when |Δ_fpts| < 0.005 (per PR #31 retrospective).

---

## 6. Reports

`reports/feature_probe_rb_decomposition_summary.md`:

- Verdict table (1 row per of 5 stats): stat, n_paired, RMSE Δ point, CI, verdict, magnitude flag.
- Per-volume-axis per-eval-year coverage table (carries > 0 rate and targets > 0 rate per year, with the threshold comparison).
- Mechanism caveat: this probe tests decomposition *with Ridge sub-models everywhere*. Factor-appropriate sub-model probes are separate cycles per spec §1.4 #3.
- Recommended next direction per verdict: SIGNAL → integration plan; NULL → close cell + name td_rate_per_carry factor-appropriate probe (or yards_per_carry factor-appropriate probe) as candidate follow-ups; REGRESSION → close strongly.
- Plan-vs-execution deviations section (template).

Per-stat reports (`reports/feature_probe_rb_decomposition_{rushing_yards,rushing_tds,receptions,receiving_yards,receiving_tds}.md`): verdict, per-year RMSE delta breakdown, factor-residual orthogonality check, clipping-bias diagnostic.

`reports/feature_probe_rb_decomposition_per_stat.csv`: long-form per-stat × per-year + pooled rows. Schema mirrors PR #32's CSV.

---

## 7. Estimated scope

5-6 plan tasks. Single session. Real-data probe runtime estimated 30s–2min (vs PR #44's 7s — 5 stats × 2 arms × 4 years with bootstrap is larger but still trivial).

| Task | Surface | Files touched |
|---|---|---|
| 1. `_RB_DECOMPS` + `_fit_direct` + `_fit_decomposed_volume` + `_fit_decomposed_efficiency` + unit tests | probe core | `src/projections/backtest/rb_decomposition_probe.py` (new), `tests/test_backtest/test_rb_decomposition_probe.py` (new) |
| 2. `_predict_direct` + `_predict_decomposed` + unit tests | probe core | extend Task 1 files |
| 3. `walk_forward_residuals` + `compute_verdicts` + integration tests | probe core | extend Task 1 files |
| 4. CLI script + CLI smoke | scripts + tests | `scripts/probe_rb_decomposition.py` (new), `tests/test_scripts/test_probe_rb_decomposition_cli.py` (new) |
| 5. Real-data probe run + report + PM/TODO updates | reports + PM | `reports/feature_probe_rb_decomposition_summary.md`, 5 per-stat reports, `reports/feature_probe_rb_decomposition_per_stat.csv`, `project_management.md`, `TODO.md` |

End-to-end: 1 focused session.

---

## 8. Implementation plan handoff

After spec approval and commit on `feat/probe-rb-decomposition`, the next step is the writing-plans skill to produce `docs/superpowers/plans/2026-05-16-rb-decomposition-probe.md` decomposing the 5 tasks above into per-task implementation steps with per-task verification commands.
