# PBP Red-Zone Family Probe — Summary

**Date:** 2026-05-02
**Branch:** `feat/probe-pbp-redzone`
**Spec:** `docs/superpowers/specs/2026-05-02-pbp-redzone-feature-family-probe-design.md`
**Plan:** `docs/superpowers/plans/2026-05-02-pbp-redzone-feature-family-probe.md`
**Override:** `data/features_probe/pbp_redzone.parquet` (regenerable; not committed; mtime 2026-05-02 00:10:53)
**Override generator:** `scripts/build_pbp_redzone_override.py`

The four PBP-derived team-level red-zone features `team_rz_pace_l4`, `team_rz_pass_rate_l4`, `team_def_rz_epa_allowed_l4`, `team_def_rz_pass_rate_allowed_l4` were bundled into a single override and probed in two modes (augment, swap) at the BaselineModel level always and at the lightgbm-nb level conditionally with `--force-composite` (per spec §1.3 criterion 3, since the baseline modes returned `NULL`). Family verdict is the §4 rule applied across the executed reports. Red zone defined as `yardline_100 <= 20` (NFL standard); all four features are trailing-4 prior-regular-season-game means with one-prior-season backfill.

## Per-mode summary

Each row reports pooled-Phase-1 SIGNAL / REGRESSION counts across that mode's 24 `(position, stat)` cells (4 positions × 6 position-relevant stats), and the Phase 2 composite verdict if Phase 2 ran.

| Model    | Mode    | Pooled-Phase-1 cells | Phase 2 verdicts (4 positions)                        |
|----------|---------|---------------------|-------------------------------------------------------|
| baseline | augment | 0 SIGNAL / 0 REGR   | skipped — Phase 1 NULL                                |
| baseline | swap    | 0 SIGNAL / 0 REGR   | skipped — Phase 1 NULL                                |
| lgb-nb   | augment | 0 SIGNAL / 0 REGR   | 4× DO_NOT_ADOPT (QB cell in REGRESSION territory)     |
| lgb-nb   | swap    | 0 SIGNAL / 0 REGR   | 4× DO_NOT_ADOPT (all 4 brackets zero)                 |

### Phase 2 composite RMSE deltas (lgb-nb only; positive = regression)

| Mode    | Pos | RMSE delta (fpts) | 95% CI               | Verdict      |
|---------|-----|-------------------|----------------------|--------------|
| augment | QB  | **+0.0268**       | **[+0.0082, +0.0449]** — strictly above 0 | DO_NOT_ADOPT (regression) |
| augment | RB  | -0.0041           | [-0.0129, +0.0044]   | DO_NOT_ADOPT (null)       |
| augment | WR  | +0.0007           | [-0.0051, +0.0071]   | DO_NOT_ADOPT (null)       |
| augment | TE  | +0.0027           | [-0.0063, +0.0112]   | DO_NOT_ADOPT (null)       |
| swap    | QB  | +0.0114           | [-0.0062, +0.0293]   | DO_NOT_ADOPT (null)       |
| swap    | RB  | +0.0019           | [-0.0067, +0.0116]   | DO_NOT_ADOPT (null)       |
| swap    | WR  | +0.0004           | [-0.0057, +0.0067]   | DO_NOT_ADOPT (null)       |
| swap    | TE  | +0.0006           | [-0.0082, +0.0094]   | DO_NOT_ADOPT (null)       |

**Only QB augment is statistically significant at the composite level**, and it's a *regression* (~0.027 fpts worse than the baseline feature set). The other 7 cells have CIs straddling zero with point estimates near zero — clean nulls. Swap mode is uniformly milder than augment mode (smaller magnitudes, all bracketing zero), suggesting the v1 `opp_allowed_*_fppg_l4` features are not what the RZ family is colliding with on QB.

### Per-year cells (informational; not gate-binding)

Per-year Phase-1 cells fire on a handful of (position, stat, year) triples — these reflect single-year sampling variation and the pooled bootstrap correctly washes them out at family level. The same per-year set appears in baseline and lgb-nb output because Phase 1 is RidgeCV-only regardless of `--model` (probe spec convention; not a bug):

| Direction | Position | Stat | Year | RMSE delta (augment) | RMSE delta (swap) |
|-----------|----------|------|------|----------------------|-------------------|
| SIGNAL    | QB       | rushing_yards | 2022   | -0.0553 ([-0.105, -0.004]) | (not pooled-significant) |
| SIGNAL    | TE       | receiving_yards | 2023 | -0.1130 ([-0.178, -0.049]) | (not pooled-significant) |
| REGRESSION | QB      | rushing_yards | 2021   | +0.0751 ([+0.017, +0.132]) | +0.0662 ([+0.023, +0.111]) |
| REGRESSION | RB      | rushing_yards | 2021   | (not pooled-significant)   | +0.1050 ([+0.003, +0.211]) |
| REGRESSION | RB      | receiving_yards | 2023 | +0.0633 ([+0.024, +0.103]) | +0.0576 ([+0.011, +0.108]) |
| REGRESSION | TE      | receiving_yards | 2022 | (not pooled-significant)   | +0.1192 ([+0.013, +0.223]) |

QB rushing_yards 2021 is the most consistent per-year signal — it regresses in both augment and swap, BL and lgb-nb. The pooled estimate brackets zero across QB rushing_yards (no durable QB-rushing-yards effect across years), but the 2021 single-year regression is real and points the same direction as the QB-augment-composite regression at lgb-nb (+0.027 fpts). One coherent story: RZ features carry slight QB-axis noise that doesn't sustain across years, especially under augment.

## Coverage caveat

Per spec §1.3 criterion 1, the override coverage check should pass at ≥ 95% per (position, season) pair. **The probe was invoked with `--coverage-threshold 0.90`** because the probe's hardcoded check is *pooled* across all seasons rather than per-(position, season), and the structurally cold-start 2018 (no Y-1 backfill, NaN coverage 73-77% per position) drags the pooled coverage to 94.7%. **Per-season 2019–2024 coverage is 96.9–98.9% across all 4 positions** — uniformly above 95% in the eval window where the probe actually evaluates against `--holdout-years 2021-2024`. Same precedent as PR #22 (which used `--coverage-threshold 0.70` for the same 2018 cold-start reason). The threshold relaxation does not affect the comparison validity.

This is a known precedent gap between spec text ("≥ 95% per (position, season) pair") and probe implementation (pooled coverage check); not blocking and not unique to this spec.

## Family verdict

**`NULL` (durable per spec §1.3 criterion 3 — both baseline modes ran AND both lgb-nb modes ran with `--force-composite`).**

Computed by the §4 rule via the pandas one-liner over the 4 committed CSVs:

```python
phase1_signal = ((combined['phase'] == 'phase1') & (combined['year_or_pooled'] == 'pooled') & (combined['verdict'] == 'SIGNAL')).any()  # False
phase2_adopt = ((combined['phase'] == 'phase2') & (combined['verdict'].isin(['ADOPT', 'MARGINAL']))).any()  # False
verdict = 'SIGNAL' if (phase1_signal or phase2_adopt) else 'NULL'  # 'NULL'
```

The `family_verdict_from_reports` helper (PR #20) is the canonical implementation; this CSV-based check transcribes the same §4 rule onto the probe's long-format output. Both arrive at `NULL`.

## Mechanism annotation (per §4.1)

The bundle's predicted mechanism was **TD efficiency** — RZ pace, RZ pass rate, defensive RZ EPA allowed, and defensive RZ pass rate forced were chosen to drive `*_tds` cells across all four positions, since TDs are Plan 5c's noisiest unmoved cell.

**No `*_tds` cell fired SIGNAL anywhere** (zero pooled SIGNAL cells across all 16 (model, mode, position) combinations). The only directional cell is QB-augment composite REGRESSION at lgb-nb (driven by stat-set composite, not TDs specifically). Mechanism: **predicted mechanism not observed; nor was any unexpected mechanism**. The four RZ features carry no orthogonal signal under either Ridge or lgb-nb composite at the trailing-4-team-game unit on the standard `yardline_100 ≤ 20` cut.

## Decision

**Family closed at the RZ-broad cut, durable across BaselineModel + lgb-nb.** The four bundled team-level RZ features do not carry orthogonal signal beyond v1 + already-shipped PBP team features. Closes TODO #3c's RZ-context sub-question.

### Refined-unit candidates beyond the RZ-broad cut that remain unexplored

- **Goal-line (`yardline_100 ≤ 5`)** specifically. Plausible mechanism: TD-rate density is much higher inside the 5 than across the full 20-yard zone, so trailing-4 means may be tighter signal-to-noise. Counterargument: spec §1.2 deferred goal-line as a refined-unit follow-up if RZ-broad probed SIGNAL — the current NULL across both model classes is evidence that goal-line is unlikely to clear what RZ-broad couldn't. **Do not queue without independent evidence** (e.g., a published study or third-party benchmark) suggesting the unit choice was the binding constraint.
- **Per-stat splits.** The bundle was tested as 4 features against 4 positions × 6 stats jointly. A finer probe could test (RZ pass rate) only on `*_pass_tds` cells, (RZ pace) only on `*_yards`, etc. Counterargument: Plan 9 + the family-level prior framework explicitly bundle to escape the per-cell noise floor; un-bundling reverses that lesson. **Defer.**
- **RZ EPA-residual** vs. schedule strength (analogous to PR #20's `team_def_epa_resid_l4` but RZ-restricted). Plausible mechanism: RZ-specific opponent strength might decouple from full-field `team_def_epa_resid_l4`. Counterargument: PR #20's full-field `team_def_epa_resid_l4` already shipped to RB and clears its adoption gate; the RZ-subset is unlikely to carry orthogonal signal beyond the full-field version it's correlated with. **Defer indefinitely.**

### What this closes

- TODO #3c's "Red-zone usage shares (separate from full-field share)" candidate, at the team-level RZ-broad cut.
- The `--force-composite` lgb-nb question for RZ features specifically: lgb-nb composite does not extract RZ-bundle signal that Ridge baseline missed. Same finding as Plan 9 retro option C (PR #19) for opp-EPA-residual: model class change does not unlock signal that the feature class fundamentally lacks.

### What remains open in TODO #3c

- **Pressure rate allowed by O-line** — the third unexplored team-level family from TODO #3c's list, distinct from RZ context. Different mechanism (offensive line proxy via sack rate / scramble rate, not TD distribution). Curated PBP has `sack`, `qb_dropback`, `qb_scramble` so a pressure-proxy bundle is buildable without ingest extension.
- A future bundled probe for the pressure family would follow the same workflow as this spec.

## Reports

- `reports/feature_probe_pbp_redzone_augment.{md,csv}` — baseline augment (4 positions, long format).
- `reports/feature_probe_pbp_redzone_swap.{md,csv}` — baseline swap (drops `opp_allowed_*_fppg_l4`).
- `reports/feature_probe_pbp_redzone_lgbnb_augment.{md,csv}` — lgb-nb composite, augment.
- `reports/feature_probe_pbp_redzone_lgbnb_swap.{md,csv}` — lgb-nb composite, swap.
