# PBP Feature Family Probe — Summary

**Date:** 2026-04-30
**Branch:** `feat/probe-pbp-family`
**Spec:** `docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md`
**Plan:** `docs/superpowers/plans/2026-04-30-pbp-feature-family-probe.md`
**Override:** `data/features_probe/pbp_family.parquet` (regenerable; not committed)
**Override generator:** `scripts/build_pbp_family_override.py`

The four PBP-derived team-level features `pace_l4`, `proe_l4`, `team_ayps_l4`,
`team_def_epa_resid_l4` were bundled into a single override and probed in
two modes (augment, swap) at the BaselineModel level against the v1 baseline
features. Family verdict is the §4 rule applied across the executed reports.

## Per-mode summary

| Model | Mode | Pos | Pooled SIGNAL cells | Pooled REGRESSION cells | Composite Phase 2 verdict | RMSE delta (fpts) | RMSE 95% CI |
|---|---|---|---:|---:|---|---:|---|
| baseline | augment | QB | 0 | 1 (`passing_yards`) | DO_NOT_ADOPT | +0.0108 | [-0.0106, +0.0312] |
| baseline | augment | RB | 1 (`rushing_yards`) | 0 | **ADOPT** | -0.0124 | [-0.0249, -0.0001] |
| baseline | augment | WR | 0 | 0 | DO_NOT_ADOPT | -0.0038 | [-0.0124, +0.0053] |
| baseline | augment | TE | 0 | 0 | DO_NOT_ADOPT | +0.0055 | [-0.0048, +0.0163] |
| baseline | swap    | QB | 0 | 0 | DO_NOT_ADOPT | -0.0022 | [-0.0261, +0.0223] |
| baseline | swap    | RB | 1 (`rushing_yards`) | 0 | **ADOPT** | -0.0131 | [-0.0259, -0.0003] |
| baseline | swap    | WR | 0 | 0 | DO_NOT_ADOPT | -0.0028 | [-0.0120, +0.0069] |
| baseline | swap    | TE | 0 | 0 | DO_NOT_ADOPT | +0.0079 | [-0.0025, +0.0184] |

### Phase 1 SIGNAL cells (pooled across years)

| Mode | Position | Stat | RMSE delta (fpts) | CI |
|---|---|---|---:|---|
| augment | RB | rushing_yards | -0.0709 | [-0.1304, -0.0078] |
| swap    | RB | rushing_yards | -0.1047 | [-0.1754, -0.0397] |

### Phase 1 REGRESSION cells (pooled)

| Mode | Position | Stat | RMSE delta (fpts) | CI |
|---|---|---|---:|---|
| augment | QB | passing_yards | +0.4537 | [+0.1354, +0.7551] |

QB passing-yards regression in augment mode is meaningful — the four team-level
PBP features systematically *hurt* the existing QB passing-yards prediction by
~0.45 fpts when added on top of v1. The swap mode (which drops the v1
`opp_allowed_qb_fppg_l4` column alongside) does not regress, but it also does
not gain on QB. Team-level pace / PROE / AYPS / opp-EPA-resid carry no
orthogonal signal for QB passing yards under Ridge regularization.

## Family verdict

**`SIGNAL`** (computed by the spec §4 rule via `family_verdict_from_reports`).

The verdict fires on two independent triggers:
- **Phase 1 pooled SIGNAL** on `(RB, rushing_yards)` in both augment and swap.
- **Phase 2 ADOPT** on `RB` in both augment and swap (composite RMSE delta -0.012 to -0.013 fpts strictly below zero).

The signal is **concentrated on RB**. QB / WR / TE all return DO_NOT_ADOPT
at the composite level. QB even regresses on passing yards in augment mode.

Per spec §3.2, the conditional `lightgbm-nb` runs are **not required** because
the baseline already returned SIGNAL. The family is greenlit at the
BaselineModel level.

## Decision

**Greenlit. Scope a follow-up production-builder plan for RB.**

The probe verdict justifies writing a production plan that integrates the
PBP family into the RB feature builder. Suggested scope for the follow-up:

1. **RB-only integration first.** Add `pace_l4`, `proe_l4`, `team_ayps_l4`,
   `team_def_epa_resid_l4` to `RbFeaturesSchema` and `build_rb_features`.
   Run the full backtest + adoption gate on RB to verify the probe's ADOPT
   prediction holds under the production model class (ensemble Model D and
   any newly tuned variant).
2. **Investigate which subset carries the RB rushing-yards signal.** The
   probe tested all four columns bundled. The production plan should
   verify whether the signal lives in one feature (e.g., `pace_l4` for
   plays-per-game volume), in two (`pace_l4` + `team_def_epa_resid_l4`),
   or genuinely across all four. Per Plan 9 lesson, family-level priors
   don't pin per-feature priors.
3. **QB regression caveat.** Augment-mode QB passing-yards regression
   (+0.45 fpts) means the production plan must NOT integrate these
   features into QB without a separate gate run that verifies the swap
   semantics on QB. The probe's swap-mode QB result (DO_NOT_ADOPT at
   -0.002 ± 0.024) is the honest answer: net-zero on QB. Don't ship to QB.
4. **WR / TE deferred.** Both return DO_NOT_ADOPT at composite. The
   production plan can either (a) ship RB-only, leaving WR/TE on v1
   features, or (b) probe a refined per-position version (player aDOT
   for receivers) before committing to either inclusion or exclusion.
   Recommendation: ship RB-only in the first iteration; revisit
   WR / TE as a separate spec with refined per-position units.

This closes the family-level question opened by TODO #3c. The
production-builder follow-up plan is the natural next step.

## Cross-references

- Per-position augment reports: `reports/feature_probe_pbp_family_augment.{md,csv}`
- Per-position swap reports: `reports/feature_probe_pbp_family_swap.{md,csv}`
- Spec: `docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md`
- Plan: `docs/superpowers/plans/2026-04-30-pbp-feature-family-probe.md`
- TODO #3c: cross-reference will be appended in the same commit cluster.
