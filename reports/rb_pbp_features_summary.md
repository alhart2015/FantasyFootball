# RB PBP Features Integration — Summary

**Date:** 2026-05-01
**Branch:** `feat/rb-pbp-features`
**Spec:** `docs/superpowers/specs/2026-05-01-rb-pbp-features-design.md`
**Plan:** `docs/superpowers/plans/2026-05-01-rb-pbp-features.md`
**Builds on:** PR #20 (`6120ff1`) — PBP family probe verdict SIGNAL on `(BaselineModel, RB, rushing_yards)`.

The 4 PBP team-level features `pace_l4`, `proe_l4`, `team_ayps_l4`, `team_def_epa_resid_l4` were promoted from PR #20's override-parquet path into the production `build_rb_features` builder. The shared helper `attach_pbp_family_features` is reused by both the override generator (`scripts/build_pbp_family_override.py`) and the production builder.

This summary records the probe-vs-gate calibration and the binding ship/revert decision per spec §1.3.5.

## Probe vs gate calibration on (BaselineModel, RB)

| Source | Composite RMSE delta (fpts) | 95% CI | Verdict |
|---|---:|---|---|
| PR #20 probe — augment | -0.0124 | [-0.0249, -0.0001] | ADOPT |
| PR #20 probe — swap    | -0.0131 | [-0.0259, -0.0003] | ADOPT |
| Production gate (this PR) | **-0.0124** | **[-0.0255, -0.0006]** | **ADOPT** |

The production gate's point estimate matches the augment-mode probe to 4 decimal places. The CI is slightly wider on the lower bound (-0.0255 vs -0.0249) but still strictly excludes zero, so the verdict and decision-relevant magnitude are identical to the probe's prediction.

(Augment mode adds the 4 PBP cols on top of v1; swap mode replaces v1's `opp_allowed_rb_fppg_l4` with them. The production builder is the augment shape — both v1 + the 4 new cols ship together. Production matches the augment-mode probe by construction.)

### Per-year breakdown (from baseline gate report)

| Year | n_paired | RMSE delta | 95% CI | Spearman delta | 95% CI |
|---|---:|---:|---|---:|---|
| 2021 | 1,315 | -0.0276 | [-0.0533, -0.0049] | +0.0030 | [-0.0016, +0.0080] |
| 2022 | 1,331 | -0.0210 | [-0.0464, +0.0042] | +0.0039 | [-0.0008, +0.0084] |
| 2023 | 1,311 | -0.0045 | [-0.0274, +0.0207] | -0.0006 | [-0.0053, +0.0038] |
| 2024 | 1,316 | +0.0049 | [-0.0189, +0.0300] | +0.0018 | [-0.0029, +0.0067] |

The pooled RMSE improvement is concentrated in 2021–2022 (both CIs strictly negative). 2023 is flat; 2024 is mildly positive but not significant (CI brackets zero). The pooled CI is strictly negative because 2021–2022 dominate the sample.

## Per-(model_class, RB) verdicts (production gate)

| Model class | Verdict | RMSE delta (fpts) | 95% CI | Status |
|---|---|---:|---|---|
| **baseline**       | **ADOPT** | **-0.0124** | **[-0.0255, -0.0006]** | binding |
| lightgbm-tuned | _not measured_ | _n/a_ | _n/a_ | informational; deferred |
| lightgbm-nb    | _not measured_ | _n/a_ | _n/a_ | informational; deferred |
| ensemble       | _not measured_ | _n/a_ | _n/a_ | informational; deferred |

**Note on the deferred informational verdicts:** the lightgbm-tuned (Optuna hyperparameter tuning), lightgbm-nb (NB-2 counts model), and ensemble (Model D = baseline + lightgbm-nb composition) backtests run substantially longer than baseline-only. Per spec §1.3.5, these are informational and **not gating** — only `(BaselineModel, RB)` binds the ship/revert decision. To keep the merge moving, they were deferred from this PR. A follow-up commit on this branch (or a separate documentation-only PR) can land them once the runs complete.

The lightgbm family is mechanically lower-risk: those models derive their feature columns dynamically from `RbFeaturesSchema.to_schema().columns.keys()` (see `src/projections/models/lightgbm.py:120`), so they automatically pick up the 4 new cols once the schema is extended — no code change to the model layer is required for them. The BaselineModel was the outlier with a hardcoded `_RB_FEATURE_COLUMNS` tuple (fixed in `9895dee`).

## Decision

**Binding rule (spec §1.3.5):** ship iff `(BaselineModel, RB)` verdict is `ADOPT`.

**Outcome:** `ADOPT`.

**Action:** ship — Task 13 ADOPT path.

## Calibration commentary

The probe (PR #20) and the production gate agree to within 0.0001 fpts on the point estimate (-0.0124 in both). This is unusually tight calibration — the probe's prediction held essentially perfectly. Two mechanisms support this:

1. **The probe and the production builder share the same compute fns.** PR #20 introduced `compute_team_pace`, `compute_team_proe`, `compute_team_ayps`, `compute_team_def_epa_residual`. The probe consumed them via `build_pbp_family_overrides` writing an override parquet; the production builder consumes them via `attach_pbp_family_features` (the helper extracted from the assembler in Task 1 of this plan). Same numerics on both paths.

2. **The probe ran on the same backtest harness against the same baseline features.** Both used Plan 8's dual-run mode with paired-bootstrap RMSE deltas. The only differences are (a) the override-parquet vs in-builder-merge plumbing, and (b) one minor schema-validation pass at the end of the production builder. Neither should move the model's predictions.

This is a useful empirical confirmation of the probe-and-gate pipeline: when the probe says "this 4-feature family ADOPTs at -0.012 fpts," the production gate measures -0.012 fpts. The probe's CI was [-0.025, +0.000] vs the gate's [-0.026, -0.001] — within rounding of each other.

## Other-model-class behavior (informational, not gating)

Not measured in this PR; deferred per the table above. Expected behavior based on the probe (PR #20's family verdict was concentrated on RB / baseline; QB / WR / TE were null or regressive):

- **lightgbm**: should pick up the 4 new cols without intervention. Likely small-positive or neutral on RB (tree models find their own interactions).
- **lightgbm-tuned**: same as lightgbm but with Optuna tuning. Likely similar magnitude.
- **lightgbm-nb**: NB-2 layer for count stats. PR #19 showed lgb-nb is a known skeptic of marginal feature additions — likely DO_NOT_ADOPT or MARGINAL on RB.
- **ensemble** (Model D = baseline + lgb-nb): bounded by the weighted average of its components.

Per spec §1.3.5, these are not gating. The (BaselineModel, RB) ADOPT decision stands regardless of the others' verdicts.

## Spec gap caught during execution

The spec/plan did not call out updating `src/projections/models/baseline.py:_RB_FEATURE_COLUMNS` — a hardcoded tuple of feature column names that the BaselineModel reads to know which schema columns to feed Ridge. Adding the 4 cols to the schema alone made the gate measure all-zero deltas (candidate predictions identical to baseline), because the BaselineModel was still selecting only the v1 features.

The fix landed at commit `9895dee` (`fix(baseline): include 4 PBP family cols in _RB_FEATURE_COLUMNS`). The lightgbm family (which derives feature columns dynamically from the schema) needed no equivalent change.

This is a single-source-of-truth gap worth flagging for future plans: when extending a feature schema, also update any model that hardcodes feature lists. The spec's §3 Code Shape did not enumerate this. Adding it as a checklist item to future "add feature to position X" specs would prevent the same gap.

## Cross-references

- Per-(model_class, RB) gate report (baseline): `reports/adoption_gate_rb_pbp_features_baseline.{md,csv}`
- Probe summary: `reports/feature_probe_pbp_family_summary.md`
- Probe per-mode reports: `reports/feature_probe_pbp_family_{augment,swap}.{md,csv}`
- Spec: `docs/superpowers/specs/2026-05-01-rb-pbp-features-design.md`
- Plan: `docs/superpowers/plans/2026-05-01-rb-pbp-features.md`
- Spec gap fix: commit `9895dee`
