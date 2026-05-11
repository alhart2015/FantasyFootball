# WR Receiving Stats Target Decomposition Probe — Design

**Date:** 2026-05-10
**Branch:** `feat/probe-target-decomposition`
**Status:** spec
**Predecessor:** PR #31 (refined-unit weather strict-replace integration — verdict full-revert × 2; closes the broad-cut weather direction on RB/WR with the retrospective takeaway that probe binding-cell magnitudes <0.005 fpts under coverage relaxation should be treated as MARGINAL, not SIGNAL). With the weather track closed, target decomposition (TODO #23) is the next untouched model-improvement axis named in the post-Plan-3e brainstorm. This is the first probe in the project to test a **model architecture change** (volume × efficiency factor decomposition) rather than a **feature addition**.

---

## 1. Goals & success criteria

### 1.1 Goal

Probe whether decomposing WR receiving stats into a **shared volume factor × per-stat efficiency factor** beats the current `BaselineModel`'s direct per-stat RidgeCV on out-of-sample mean prediction. Three composed predictions are tested:

| Stat | Volume factor | Efficiency factor | Trained on |
|---|---|---|---|
| `receptions` | `targets` | `catch_rate` (catches / target) | WR rows with `targets > 0` |
| `receiving_yards` | `targets` | `yards_per_target` | WR rows with `targets > 0` |
| `receiving_tds` | `targets` | `td_rate_per_target` | WR rows with `targets > 0` |

The shared `targets` sub-model is fit once per training window and reused for all three composed predictions. The **architectural hypothesis** is two-part: (a) volume (team-driven) and efficiency (player-driven) carry distinct signal axes that a single combined Ridge cannot fully extract, and (b) the shared volume factor unlocks within-row coherent sampling at the integration stage (currently each stat is sampled independently in `score_distribution`). This probe tests only (a) — the mean-RMSE consequence of decomposition. Within-row coherence is a distribution-shape effect that lifts composite-fpts CI accuracy, not per-stat mean RMSE; it lives in the integration plan.

This is the **first model-architecture probe in the project**. Prior probes (Track 2A: PBP, trajectory, weather × broad-cut + refined-unit) all measured feature additions via override parquets through `scripts/probe_feature_signal.py`. Target decomposition has no override parquet — the probe is a new lite walk-forward CV harness, mirroring `feature_probe.py` Phase 1's shape but adapted to comparing two prediction recipes rather than two feature sets.

### 1.2 Architectural prior framework

The decomposition is the canonical "volume × efficiency" frame from the post-Plan-3e brainstorm and TODO #23. Estimated win: 3–10% RMSE on the affected stats, model-class-independent. Compare to integrated feature-family wins (PR #21 RB PBP -1.24% on RMSE composite-fpts; PR #26 WR trajectory -3.71% composite-fpts; PR #29 weather -0.77% RB / -1.04% WR composite-fpts). Per-stat RMSE deltas at the family-magnitude prior would be larger (composite-fpts dilutes per-stat lift across 6 stats and Monte-Carlo noise).

**Sub-model class for the probe: RidgeCV everywhere with predict-time clipping.** This deliberately matches `BaselineModel`'s algorithmic family — any SIGNAL is then attributable to *decomposition itself*, not to a model-class change. Per-factor predict-time clips:

| Sub-model | Predict-time clip |
|---|---|
| `targets` (volume, shared) | clip(predict, 0, +∞) |
| `yards_per_target` (efficiency) | clip(predict, 0, +∞) |
| `catch_rate` (efficiency) | clip(predict, 0, 1) |
| `td_rate_per_target` (efficiency) | clip(predict, 0, 1) |
| 3 direct comparators (RidgeCV → stat directly) | no clip — matches today's BaselineModel exactly |

If the probe returns SIGNAL, **switching efficiency factors to factor-appropriate sub-model classes (logistic for `catch_rate` / `td_rate_per_target`, log-link Gamma for `yards_per_target`, Poisson for `targets`) is a named follow-up**, layered on top of the decomposition win in a separate probe + integration cycle. The probe deliberately does not bundle factor-appropriate models with decomposition because attribution would be muddied: a SIGNAL could come from either change, and disentangling requires a doubled control arm.

### 1.3 Success criteria

The probe **ships** when all four pass. The verdict (SIGNAL/NULL/REGRESSION per stat) is informational, not a ship gate.

1. **Coverage:** WR rows with `targets > 0` per eval year (the share of eval-year rows on which the efficiency factor is well-defined) ≥ 0.95 across all four eval years 2021–2024. Default `--coverage-threshold 0.95`. The same threshold applies separately to per-train-window coverage (each walk-forward iteration's training set must hit ≥ 0.95). Relaxation on either set requires explicit documentation in the report per the PR #31 retrospective rule (probe binding-cell magnitudes under ~0.005 fpts with coverage relaxation should be treated as MARGINAL, not SIGNAL).
2. **Probe completeness:** Walk-forward over eval years {2021, 2022, 2023, 2024}; per-stat residual buffers populated for both arms (direct, decomposed) on all four years; pooled paired-bootstrap CI rendered per stat. Three per-stat reports (`receptions`, `receiving_yards`, `receiving_tds`) plus one summary report.
3. **Verdict mapping per stat:**
   - **SIGNAL** iff `rmse_delta.hi_95 < 0` (decomposed CI strictly negative).
   - **REGRESSION** iff `rmse_delta.lo_95 > 0` (decomposed CI strictly positive).
   - **NULL** otherwise (CI brackets zero).
   - No effect-size floor at the per-stat level — per-stat RMSE units are heterogeneous (~30 yds for receiving_yards, ~3 for receptions, ~0.5 for receiving_tds). Pure CI-based verdict matches `feature_probe.py`'s Phase 1 PerStatVerdict mapping (`src/projections/backtest/feature_probe.py:53`).
4. **Verification gates green:** mypy strict + ruff + ruff format clean across `src/`, `scripts/`, `tests/`. Relevant pytest subset clean (probe module + walk-forward harness tests).

### 1.4 Out of scope (deferred)

The following are explicitly out of scope for this probe. Items 1–3 are gated follow-ups conditional on a SIGNAL verdict; items 4–7 are deferred independently of verdict.

1. **Composite-fpts measurement.** The probe measures per-stat Δ-CV-RMSE only. The composite-fpts metric (sum-of-stats × scoring-coefficients via `score_distribution`) is the integration plan's adoption-gate concern. Per-stat RMSE win is *necessary but not sufficient* for a composite-fpts win; per-stat RMSE NULL on all three is near-conclusive close (composite-fpts mean ≈ sum of per-stat means; the only way to win composite-fpts without per-stat is via within-row correlation lifting CI accuracy on a noisy-but-zero-mean stat sum, which is small in practice and isn't measured by per-stat mean RMSE anyway).
2. **Within-row coherent factor sampling.** The integration plan must wire the same per-row `targets` draw into all three composed distributions, so receptions / receiving_yards / receiving_tds become correlated within a row at scoring time (currently independent). This lifts composite-fpts CI accuracy, separate from the mean-RMSE story tested here. Names a `ProductDistribution` or coherent-sampling helper that produces three `SampledDistribution` instances from a single shared (targets, catch_rate, ypt, tdrate) draw set.
3. **Factor-appropriate sub-model classes** (logistic for `catch_rate` / `td_rate_per_target`, log-link Gamma for `yards_per_target`, Poisson for `targets`). Named follow-up to a SIGNAL integration. Tests a separate hypothesis (model class) on top of the decomposition (architectural recipe).
4. **Other positions.** RB (`carries × yards_per_carry` for rushing; `targets × yards_per_target` for receiving; per-touch TD rate), QB (full passing chain plus rushing/sack/scramble adjustments), TE (mirrors WR receiving). Each is its own probe + integration cycle. WR-first because it's the largest sample, has the cleanest 2-factor receiving-stat structure, and is currently routed to `EnsembleModel` in production (the highest bar in the dispatch table — a binding-cell win there is the most informative).
5. **Other stats on WR.** `rushing_yards` / `rushing_tds` / `fumbles_lost` are in `_WR_TARGET_STATS` but lack a clean volume axis on WR (rushing carries on WR are mostly 0; fumbles_lost is rare). Decomposing them is awkward and the prior is thin. Out of scope; the integration plan keeps these direct.
6. **New model class.** `DecomposedBaselineModel` (the integration plan's deliverable) is not built in the probe. The probe operates inline against ridges fit in-memory, not against a registered model factory in `_WR_FACTORIES`.
7. **`SampledDistribution` / `ProductDistribution`.** No new distribution classes in the probe — point predictions only (per-stat mu_decomposed = mu_volume × mu_efficiency, per-row scalar).

---

## 2. Source data — already ingested

All inputs already exist in the feature cache and weekly stats:

| Column | Type | Source | Use |
|---|---|---|---|
| WR features | `WrFeaturesSchema` | `data/features/wr/season=YYYY/week=WW/part.parquet` (populated by `scripts/refresh_features.py`) | X matrix for all 7 ridges (1 shared volume + 3 efficiency + 3 direct). |
| `targets` | `Series[int]` | `WeeklyStatsSchema.targets` | y for the volume sub-model; filter predicate (`> 0`) for the efficiency sub-models. |
| `receptions` | `Series[int]` | `WeeklyStatsSchema.receptions` | y for `catch_rate` (= receptions / targets) and the receptions direct comparator. |
| `receiving_yards` | `Series[float]` | `WeeklyStatsSchema.receiving_yards` | y for `yards_per_target` (= receiving_yards / targets) and the receiving_yards direct comparator. |
| `receiving_tds` | `Series[int]` | `WeeklyStatsSchema.receiving_tds` | y for `td_rate_per_target` (= receiving_tds / targets) and the receiving_tds direct comparator. |

**No new ingest, no schema changes, no override parquet.** The probe consumes the existing per-week WR feature parquet and joins to `weekly_stats` on `(gsis_id, season, week)` exactly as `BaselineModel.fit` does.

---

## 3. Architecture

### 3.1 New module — `src/projections/backtest/target_decomposition_probe.py`

Mirrors `src/projections/backtest/feature_probe.py`'s shape. Pure numpy / pandas / sklearn. Reuses `paired_bootstrap_rmse_delta` and `BootstrapDelta` from `src/projections/backtest/adoption_gate.py` unchanged.

```
target_decomposition_probe.py
├── _fit_direct(X, y, alphas) -> RidgeCV
│       # baseline comparator — identical recipe to BaselineModel.fit per stat
├── _fit_decomposed_volume(X_all, targets_all, alphas) -> RidgeCV
│       # shared volume sub-model; trained on all WR rows in train years
├── _fit_decomposed_efficiency(X_targets_pos, ratio, alphas) -> RidgeCV
│       # efficiency sub-model; trained on rows with targets > 0
│       # ratio = receptions / targets (catch_rate), receiving_yards / targets (yards_per_target),
│       #        receiving_tds / targets (td_rate_per_target)
├── _predict_direct(ridge, X) -> np.ndarray
├── _predict_decomposed(volume, efficiency, X, stat) -> np.ndarray
│       # mu = clip(volume.predict, 0, +∞) * clip(efficiency.predict, lo, hi)
│       # bounds[stat] keyed off (lo=0, hi=+∞) for yards_per_target,
│       #                       (lo=0, hi=1) for catch_rate / td_rate_per_target
├── walk_forward_residuals(features_by_year, weekly_stats,
│                          eval_years=(2021, 2022, 2023, 2024),
│                          train_start=2018) -> WalkForwardOutput
│       # per stat: pooled (actual, mu_direct, mu_decomposed) tuples across eval years
└── render_probe_report(walk_forward_output, bootstrap_n=5000, seed=...)
        -> ProbeReport
```

`WalkForwardOutput` is a frozen dataclass with one entry per stat: `(actual: np.ndarray, mu_direct: np.ndarray, mu_decomposed: np.ndarray, n_paired: int, coverage_by_year: dict[int, float])`. `ProbeReport` is rendered to markdown + csv via the existing reporting helpers.

### 3.2 Sub-model fitting details

- `alphas` matches `BaselineModel.fit`: `np.logspace(-3, 3, 13)`. (`src/projections/models/baseline.py:563`.)
- Feature columns match `_WR_FEATURE_COLUMNS` exactly (`src/projections/models/baseline.py:266`). The X matrix is constructed via the same `_x_frame_with_bool_coercion` recipe — boolean cols coerced to int8 — so the comparison is apples-to-apples vs current production WR baseline.
- Train-time NaN policy: rows with NaN in any feature column are dropped before fit (matches `BaselineModel.fit:552`). Predict-time imputation uses train-set medians (matches `BaselineModel.feature_means`).
- Volume sub-model `y` is `targets` directly (an int column in `WeeklyStatsSchema`). Efficiency sub-model `y` is the ratio computed on the targets > 0 subset, where the denominator is non-zero by filter construction.
- Efficiency-factor filter: `weekly_stats[weekly_stats["targets"] > 0]` — joined to features on `(gsis_id, season, week)` after the filter. The shared volume sub-model trains on the un-filtered set (so it sees zero-target rows as legitimate observations of low-volume players).

### 3.3 Walk-forward harness

For each eval year Y in `eval_years`:

1. Train rows: WR feature-cache ∩ WR weekly_stats inner-join on `(gsis_id, season, week)`, season ∈ [`train_start`, Y-1].
2. Eval rows: same intersection, season = Y.
3. Fit on train rows: 1 shared volume RidgeCV + 3 efficiency RidgeCVs (each on its targets > 0 subset) + 3 direct comparator RidgeCVs (one per receiving stat). Persist train-set feature medians for predict-time imputation.
4. Predict per-row mu on eval rows for both arms:
   - direct: `mu_direct[stat] = direct_ridge[stat].predict(X_eval)`
   - decomposed: `mu_decomposed[stat] = clip(volume.predict(X_eval), 0, +∞) * clip(efficiency_ridge[stat].predict(X_eval), bounds[stat])`
5. Append `(actual, mu_direct, mu_decomposed)` per row to per-stat residual buffers. Record per-year coverage: `(targets > 0).mean()` on the eval rows.

After the loop: pool residuals across 4 years per stat (no row weighting; each eval year contributes its full WR row count). Pooled `n_paired` per stat is the sum across years.

### 3.4 Verdict + bootstrap CI

For each of 3 stats:

1. `rmse_delta = paired_bootstrap_rmse_delta(actual, mu_direct, mu_decomposed, n_resamples=5000, seed=...)` — returns `BootstrapDelta(point, lo_95, hi_95)`. The helper is the same one used by `feature_probe.py` and `adoption_gate.py`; `point < 0` means decomposed has lower RMSE.
2. Verdict mapping (matches `feature_probe.py:_DEFAULT_EFFECT_SIZE_FLOOR`-free Phase 1 logic):
   - `hi_95 < 0` → SIGNAL
   - `lo_95 > 0` → REGRESSION
   - else → NULL
3. Per-stat reports stitched into the summary report with the verdict, point estimate, CI, n_paired, and per-year coverage.

Seed for reproducibility: SHA-256(branch_name + "target_decomposition_probe_seed") truncated to 32 bits — same recipe as `derive_row_seed` in scoring (`src/projections/scoring/score_distribution.py:31`). Documented in the report.

### 3.5 CLI

```
python scripts/probe_target_decomposition.py \
  --output-dir reports/probe_target_decomposition/ \
  --coverage-threshold 0.95 \
  --bootstrap-n 5000 \
  --eval-years 2021 2022 2023 2024 \
  --train-start 2018
```

Defaults match the gate eval window (PR #8 onward; 2021–2024 walk-forward). All flags overridable for sensitivity probes (e.g., narrower train window) but the published report uses defaults.

### 3.6 Outputs

- `reports/feature_probe_target_decomposition_summary.md` — verdict per stat (3 rows), per-year coverage table, follow-up disposition narrative, decision-log entry.
- `reports/feature_probe_target_decomposition_per_stat.csv` — machine-readable: `stat,n_paired,rmse_delta_point,rmse_delta_lo_95,rmse_delta_hi_95,verdict`.
- `reports/feature_probe_target_decomposition_{receptions,receiving_yards,receiving_tds}.md` — per-stat detail, including direct-RMSE / decomposed-RMSE / n_paired per eval year + pooled.

---

## 4. Decision branches

Per §1.3.2's three per-stat verdicts:

- **All 3 NULL** → close target decomposition at the WR receiving cell on the 2-factor unit. PM logs the verdict + the closed direction. Refined decompositions (3-factor `targets × catch_rate × ypr`, red-zone-shares × RZ TD rate, RB-style dual-mode rushing+receiving for RB/QB/TE) remain open under TODO #23 but require independent mechanism evidence before re-probing — the post-PR-31 retrospective rule. None queued.
- **≥ 1 SIGNAL, no REGRESSION** → greenlight the integration plan. Plan must explicitly name and scope:
  1. New `DecomposedBaselineModel` peer (subclass of `BaselineModel`) with per-stat decomposition opt-in via constructor arg.
  2. Within-row coherent factor sampling — shared per-row `targets` draw flowing into all decomposed stats' composed `SampledDistribution`s. New `ProductDistribution` or sample-set helper.
  3. Factor-appropriate sub-model classes as a named follow-up (separate probe + integration cycle, conditional on this integration's gate verdict).
  4. Production routing decision: WR is currently routed to `ensemble`; the integration plan's binding cell is `(DecomposedBaselineModel, WR)` vs `(EnsembleModel, WR)`. Per-position routing flip is a §1.3.5 contingency, same shape as PR #29's RB+WR weather contingency matrix.
- **Mixed SIGNAL + REGRESSION on different stats** → write a tighter follow-up probe specified in PM. The most likely cause is a stat-specific data-quality issue (e.g., `td_rate_per_target` is extremely zero-inflated and might regress while `yards_per_target` cleanly wins). Document the per-stat verdicts and let the integration plan opt in to only the SIGNAL stats.

---

## 5. Risk register

1. **Probe-vs-gate calibration risk.** PR #31 retrospective: probe binding cells <0.005 fpts under coverage relaxation should be MARGINAL. This probe's per-stat RMSE units are 30 yds / 3 receptions / 0.5 TDs — large enough that a 1% RMSE delta is >0.005 in absolute terms. But composite-fpts dilution at the integration gate could shrink absolute fpts magnitude well below that. **Mitigation:** the report explicitly translates per-stat RMSE Δ to expected composite-fpts Δ (per-stat-RMSE × scoring-coefficient, summed) and flags any cell where the implied composite-fpts magnitude is <0.005 fpts. The integration plan's go/no-go decision should weight that figure against the probe's CI strength.
2. **Independent-factor false-confidence.** Decomposition assumes targets and efficiency factors are orthogonal in the residual sense. If catch_rate is correlated with feature-implied volume (e.g., low-target weeks are also low-catch-rate weeks, both reflecting opponent suppression), the multiplicative composition double-counts the suppression and over-shrinks the predicted mean. **Mitigation:** the report includes a per-eval-year scatter of (predicted volume residual, predicted efficiency residual) and a Pearson correlation; a |ρ| > 0.2 across years is documented as a probe caveat.
3. **Predict-time clipping bias.** Clipping efficiency at [0, 1] and volume at [0, +∞) is asymmetric. Ridge predictions in the right tail are not clipped (yards_per_target can predict 30+ yds/target even though empirical max is ~15); only left-tail clipping is engaged. **Mitigation:** the report includes per-stat clipped-row counts per eval year; if >2% of rows are clipped on any stat, document as a caveat. Clipping is the right behavior for a probe that's testing the "decomposition recipe" as a recipe; the integration plan's factor-appropriate sub-models address this structurally.
4. **Recurring QB-style augment regression.** Not directly applicable — the probe doesn't have augment/swap modes. But if per-stat RMSE on `td_rate_per_target` regresses (not unprecedented for zero-inflated ratios), document as a stat-specific finding and let the integration plan opt out of decomposing that stat.
5. **Feature column drift.** `_WR_FEATURE_COLUMNS` evolves across integrations (most recently expanded with trajectory cols then v1 weather cols). The probe must read the current `WrFeaturesSchema` and `_WR_FEATURE_COLUMNS` at run time, not pin a snapshot. **Mitigation:** the module imports `_WR_FEATURE_COLUMNS` directly from `projections.models.baseline`; if that module reorders or renames cols mid-run, the probe fails fast at the schema-validate step. A regression test pins parity (see §6 test 7).
6. **Walk-forward train-window leakage.** Rolling features in the WR feature cache use trailing-N windows that look back to season Y-1's late weeks at season Y's early weeks. The walk-forward train cutoff `season ≤ Y-1` is per-row, so train rows for eval Y do not include any row from season Y. The trailing features themselves are computed in `build_wr_features` and shouldn't leak — but the probe should sanity-check by computing train/eval feature-min/max on `season` and asserting strict separation. **Mitigation:** assertion in `walk_forward_residuals`.

---

## 6. Testing

`tests/test_backtest/test_target_decomposition_probe.py`:

1. **Unit — `_predict_decomposed`:** synthetic 2-row frame, mock volume + efficiency ridges with known coefficients, verify `mu = clip(volume_pred, 0, +∞) * clip(efficiency_pred, bounds[stat])` row-wise; verify clipping engages on both arms (negative volume → 0 product; efficiency > 1 on a rate stat → 1).
2. **Unit — `_fit_decomposed_efficiency`:** synthetic frame with mixed targets > 0 and targets == 0 rows; verify ridge fits on the targets > 0 subset only; verify ratio is computed correctly per stat.
3. **Walk-forward — eval-year row counts:** synthetic 4-year frame; verify each eval year's row count matches expected; verify train rows for eval Y exclude season Y entirely.
4. **Determinism:** identical seed → identical bootstrap CI on a fixed synthetic input.
5. **Edge — zero-volume row at predict time:** synthetic eval row where direct ridge predicts negative `targets`; verify clipped product is 0; verify the row contributes a finite paired-bootstrap residual.
6. **Coverage flag:** synthetic data with engineered `targets == 0` rate of 10% (below threshold); verify report flags the year and includes the relaxation note. Synthetic data at 2% verifies pass.
7. **Schema-revalidation:** the probe reads `_WR_FEATURE_COLUMNS` at run time; a regression test pins that the probe's X matrix col list equals `BaselineModel(position=WR, ...).feature_columns`. Failure here means a future schema change broke the probe's apples-to-apples claim.

`pytest -v tests/test_backtest/test_target_decomposition_probe.py` runs in <5s on dev hardware.

Real-data smoke: a CLI invocation against the production feature cache, recorded in the summary report, with all four verification gates green (`pytest`, `mypy src tests scripts`, `ruff check`, `ruff format --check`) per CLAUDE.md "Forced verification" rule. Evidence pasted into the PR.

---

## 7. Out of scope, named (gated on SIGNAL)

These are not done in the probe. They are named here so the integration plan, if scoped, has a fixed roadmap.

1. **`DecomposedBaselineModel`** — subclass of `BaselineModel` with `decomposed_stats: Mapping[Stat, tuple[Stat, str]]` keyed by composite stat → (volume sub-stat, efficiency expression). Override `fit` to fit both volume + efficiency ridges per decomposed stat and direct ridges per non-decomposed stat. Override `build_stat_distributions` to compose decomposed stats via `ProductDistribution` from coherent per-row volume + efficiency draws.
2. **`ProductDistribution`** — a `Distribution`-Protocol-conforming class whose `.sample(n, rng)` returns `volume.sample(n, rng) * efficiency.sample(n, rng)`. Mean and std implemented analytically under independence, with a fallback to empirical (sample-then-summarize) for non-independent cases.
3. **Coherent-sampling helper** — shared per-row (volume_samples, efficiency_samples) draw set so the same `targets` array flows into all 3 composed stats' product distributions. Direct shape: `build_stat_distributions` returns a per-row dict where the receptions / receiving_yards / receiving_tds entries are linked `SampledDistribution`s sharing the same underlying random state.
4. **Factor-appropriate sub-model classes** — logistic for `catch_rate` / `td_rate_per_target`, log-link Gamma for `yards_per_target`, Poisson (or NB-2) for `targets`. Separate probe + integration cycle gated on the decomposition probe's verdict; deliberately out of scope for the first probe to attribute decomposition wins cleanly.
5. **Other positions / other stats.** RB / QB / TE each get their own decomposition probe + integration cycle; this PR closes only the WR receiving-stats branch.
6. **Composite-fpts adoption gate.** Standard dual-run gate (`scripts/adoption_gate.py --baseline-run ... --candidate-run ...`) on `(DecomposedBaselineModel, WR)` vs current production WR `(EnsembleModel, WR)`. Per-position §1.3.5 contingency matrix on the WR cell.

---

## 8. Estimated scope

~5–6 plan tasks, single session, no overnight backtests. Per CLAUDE.md "phased execution" each task touches ≤ 5 files.

| Task | Surface | Files touched |
|---|---|---|
| 1. Module skeleton + sub-model fit/predict + unit tests | Probe core | `src/projections/backtest/target_decomposition_probe.py`, `tests/test_backtest/test_target_decomposition_probe.py` |
| 2. Walk-forward harness + walk-forward tests | Walk-forward | same module + test file extended |
| 3. Probe report rendering (markdown + csv) | Report layer | same module + test file extended |
| 4. CLI script | Operator surface | `scripts/probe_target_decomposition.py`, optional smoke test |
| 5. Real-data probe run + writeup | Reports + decision log | `reports/feature_probe_target_decomposition_*.{md,csv}` |
| 6. PM/TODO update | Decision log | `project_management.md`, `TODO.md` |

End-to-end wall-clock: 7 small RidgeCVs (each ~hundreds of WR rows × 2018–2023 = ~50K rows × 31 features) × 4 walk-forward iterations + 5000-resample paired bootstrap × 3 stats. Well under 1 minute per probe run on dev hardware.

---

## 9. Implementation plan handoff

After this spec is approved and committed on `feat/probe-target-decomposition`, the next step is the writing-plans skill to produce `docs/superpowers/plans/2026-05-10-target-decomposition-probe.md` decomposing the 6 tasks above into per-task implementation steps with explicit phase boundaries and per-task verification commands.
