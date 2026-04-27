# Project Management

Running log of project status, decisions, and next steps. Append new entries at the top; keep the bottom as the long-tail backlog. Specs live in `docs/superpowers/specs/`, plans in `docs/superpowers/plans/`, single-task TODOs in `TODO.md`.

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

**Open PR for Plan 3e (Phases 0 + 1; Phases 2 + 3 attempted-and-reverted, infrastructure preserved); after merge, brainstorm follow-up plans for the remaining coverage shortfalls.**

Plan 3e shipped Phase 0 + Phase 1 on branch `feat/plan-3e-calibration-tightening` and closed TODO #22. Phase 2 (Student-t) and Phase 3 (per-tertile bucketing) were both attempted and reverted; their infrastructure (`ParametricStudentT` + codec branch + Student-t estimator + bucketing helpers + widened `variance_params` type) remains in-tree as future building blocks. The two spec calibration targets (min cell coverage ≥ 0.65; mean delta ≥ +0.10) were not met by the shipped state. Three candidate follow-up plans documented under the Phase 3 revert block (ZIP for count cells, cross-week residual correlation, calibration-aware fitting); pick + scope one in the next brainstorming session post-merge.

After 3e: Plan 4 (public Python API + CLI verbs + free-tier web hosting).

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
- **Plan 5** — Model C (LightGBM with quantile regression). Adopt only if it beats Model A on the backtest harness.

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
