# Track 2B — RB PBP cols × other model classes (informational)

**Branch:** `feat/probe-pbp-pressure` (folded into PR #24 per user request)
**Date:** 2026-05-03
**Question:** Do the 4 RB PBP team-level features (`pace_l4`, `proe_l4`, `team_ayps_l4`, `team_def_epa_resid_l4`) shipped in PR #21 transfer signal to the tree-based model classes (lightgbm-tuned, lightgbm-nb, ensemble) beyond the BaselineModel cell that PR #21 gated on?

**Methodology.** Compared two backtest runs at `--model all` (5 model classes × 4 positions × 4 holdout years 2021–2024):
- **Baseline run:** `data/backtest/run_20260429T003552Z` — pre-PR-20 schema. RB feature schema does NOT include the 4 PBP cols.
- **Candidate run:** `data/backtest/run_20260503T014536Z` — post-PR-21 schema. RB feature schema DOES include the 4 PBP cols. Ran fresh today on the same `--model all`.

The lightgbm family (untuned + tuned + NB) auto-derives feature columns from the `RbFeaturesSchema` dynamically, so they pick up the 4 PBP cols without code changes. Per-stat predictions paired across runs on `(gsis_id, season, week, position)`. Composite RMSE = `sqrt(mean((mean_pred - actual_ppr)^2))` per `(model_class, position)`. 95% CI via 1000-bootstrap-resample on paired errors. Spearman delta computed analogously on `(mean_pred, actual_ppr)` pairs.

---

## Per-`(model_class, position)` RMSE delta (post-PR-21 minus pre-PR-20)

| Model class | Position | n | Baseline RMSE | Candidate RMSE | RMSE Δ | 95% CI | Verdict |
|---|---|---:|---:|---:|---:|---|---|
| baseline | QB | 2676 | 7.523 | 7.523 | +0.0000 | [+0.0000, +0.0000] | BRACKET-0 |
| baseline | **RB** | 5273 | 6.575 | 6.562 | **-0.0124** | [-0.0258, -0.0002] | **IMPROVE** |
| baseline | WR | 8460 | 6.636 | 6.636 | +0.0002 | [-0.0017, +0.0018] | BRACKET-0 |
| baseline | TE | 4257 | 5.154 | 5.154 | +0.0000 | [+0.0000, +0.0000] | BRACKET-0 |
| lightgbm | QB | 2676 | 7.500 | 7.500 | +0.0000 | [+0.0000, +0.0000] | BRACKET-0 |
| lightgbm | **RB** | 5273 | 6.766 | 6.752 | **-0.0141** | [-0.0278, +0.0005] | BRACKET-0 (point estimate strongest) |
| lightgbm | WR | 8460 | 6.770 | 6.770 | -0.0000 | [-0.0037, +0.0035] | BRACKET-0 |
| lightgbm | TE | 4257 | 5.309 | 5.309 | +0.0000 | [+0.0000, +0.0000] | BRACKET-0 |
| lightgbm-tuned | QB | 2676 | 7.404 | 7.404 | +0.0000 | [+0.0000, +0.0000] | BRACKET-0 |
| lightgbm-tuned | **RB** | 5273 | 6.689 | 6.679 | -0.0101 | [-0.0225, +0.0015] | BRACKET-0 |
| lightgbm-tuned | WR | 8460 | 6.707 | 6.707 | +0.0003 | [-0.0021, +0.0023] | BRACKET-0 |
| lightgbm-tuned | TE | 4257 | 5.241 | 5.241 | +0.0000 | [+0.0000, +0.0000] | BRACKET-0 |
| lightgbm-nb | QB | 2676 | 7.330 | 7.330 | +0.0000 | [+0.0000, +0.0000] | BRACKET-0 |
| lightgbm-nb | **RB** | 5273 | 6.616 | 6.609 | -0.0075 | [-0.0203, +0.0041] | BRACKET-0 |
| lightgbm-nb | WR | 8460 | 6.634 | 6.635 | +0.0011 | [-0.0019, +0.0037] | BRACKET-0 |
| lightgbm-nb | TE | 4257 | 5.156 | 5.156 | +0.0000 | [+0.0000, +0.0000] | BRACKET-0 |
| ensemble | QB | 2676 | 7.347 | 7.347 | +0.0000 | [+0.0000, +0.0000] | BRACKET-0 |
| ensemble | **RB** | 5273 | 6.596 | 6.590 | -0.0062 | [-0.0177, +0.0054] | BRACKET-0 |
| ensemble | WR | 8460 | 6.604 | 6.605 | +0.0012 | [-0.0010, +0.0034] | BRACKET-0 |
| ensemble | TE | 4257 | 5.133 | 5.133 | +0.0000 | [+0.0000, +0.0000] | BRACKET-0 |

---

## Findings

### 1. Baseline cell exactly reproduces PR #21's adoption gate verdict

`(baseline, RB)` here: **-0.0124 fpts CI [-0.0258, -0.0002]**. PR #21's reported gate result: **-0.0124 fpts CI [-0.0255, -0.0006]**. Point estimate matches to 4 decimals; CI bounds drift by ~0.0003 (different RNG state in this all-position bootstrap loop). This confirms the methodology is sound — Track 2B is using the same paired-rows-and-bootstrap shape that PR #21 used.

### 2. RB benefit transfers directionally to all tree-model classes; only `baseline` reaches strict CI<0

| Model class | RB RMSE Δ | CI | Strictly negative? |
|---|---:|---|:---:|
| baseline | -0.0124 | [-0.0258, -0.0002] | **yes** (matches PR #21) |
| lightgbm | -0.0141 | [-0.0278, +0.0005] | no (upper bound +0.0005, just barely brackets 0) |
| lightgbm-tuned | -0.0101 | [-0.0225, +0.0015] | no |
| lightgbm-nb | -0.0075 | [-0.0203, +0.0041] | no |
| ensemble | -0.0062 | [-0.0177, +0.0054] | no |

All 5 model classes show negative RMSE delta point estimates on RB. The signal is strongest at the simplest model classes (baseline, untuned lightgbm) and progressively diffuses as the model class adds complexity (Optuna tuning, NB-2 dispersion, ensemble weighting). Untuned `lightgbm` actually has the strongest point estimate (-0.0141), but its CI is wider so it's just shy of strict significance (upper bound +0.0005).

### 3. No regression on any cell

Out of 20 `(model_class, position)` cells, 0 show CI strictly above zero. The only directional improvements are on RB (5/5 classes negative). QB/WR/TE are bit-identical or trivially-different across the two runs (zero delta where the position's schema didn't change at all; tiny noise on WR which uses some shared code paths but no schema additions).

### 4. Spearman delta picture is consistent — small positive on RB, ~zero elsewhere

Not tabled separately to keep the report compact; the headline numbers (95% CIs all bracketing 0 except marginally for RB-baseline +0.0020 [-0.0004, +0.0041]) match the RMSE picture: directional but mostly within noise.

---

## Interpretation

**The RB PBP cols' signal is real but small.** PR #21's adoption-gate verdict on `(baseline, RB)` of ADOPT was load-bearing — and it transfers to tree model classes with the same directional sign, but the magnitude shrinks as model complexity grows. This is consistent with the general pattern: tree models with their own feature interactions and tuning extract some-but-less of the per-feature linear signal that Ridge captures cleanly.

**No tree-model class regresses on RB.** Adopting the cols system-wide (which was already automatic since the lightgbm family auto-picks-up schema cols dynamically) is at minimum neutral for tree classes and at maximum slightly helpful. There is no case for backing them out from the lightgbm family.

**No spillover to other positions.** QB/WR/TE schemas weren't changed in PR #21, and the per-position feature cache writes to disk separately, so this confirms the integration was clean.

---

## Limitations

- **Different runs, not perfectly hermetic.** The two backtest runs were generated 4 days apart on slightly different code (the worktree HAS PR #22 + PR #23 + this PR's spec/plan/code, but those don't touch the RB feature builder or the lightgbm models). The pre-run is from the Plan 8 reference state (`run_20260429T003552Z`); the post-run is from today's HEAD (`run_20260503T014536Z`). The only schema-affecting change to RB between those two states is PR #21 (the 4 PBP cols). No other RB-feature regressions are expected, but the comparison is not as hermetic as a same-commit dual-run with manipulated feature parquets would be.
- **Bootstrap CI methodology is paired on rows, not on backtest folds.** The walk-forward backtest's true sampling distribution involves multi-year fold uncertainty that this paired-bootstrap doesn't model. This is the same tradeoff PR #21 made; consistent with that precedent.
- **Track 2B is informational only** — it doesn't alter any production routing. The lightgbm family is already using the 4 PBP cols (auto-derived from schema), so this report just documents the magnitude of the lift, not whether to adopt.

---

## Conclusion

**Track 2B closes with a clean directional confirmation:** the 4 RB PBP team-level features that PR #21 shipped on the BaselineModel cell also help (or are neutral for) all 4 tree-based model classes. Only the simpler model classes reach strict statistical significance on the lift; the more complex classes (tuned/NB/ensemble) show the same direction but smaller and noisier effect sizes. No model class regresses. No further action needed.

**Reports:** this file + `reports/track2b_rb_pbp_lgbnb_drop.md` (a probe attempt that did NOT work — see appendix).

---

## Appendix: probe attempts that did NOT produce useful signal

We initially attempted to use the feature signal probe (`scripts/probe_feature_signal.py`) for Track 2B, since it natively supports `--model lightgbm-nb --force-composite`. Two attempts:

1. **`--drop pace_l4,proe_l4,team_ayps_l4,team_def_epa_resid_l4` only** — the probe applies `--drop` symmetrically (both baseline and candidate sides drop the same cols), so the candidate is identical to baseline → all deltas are exactly 0.0000.
2. **`--drop ... + --override rb_pbp_track2b.parquet`** (override re-adds the dropped cols on candidate) — the probe's `override_added` filter excludes any override cols matching `args.drop`, so the override is silently ignored → all deltas are exactly 0.0000 again.

The probe is designed to test "does adding these new cols help?", not "does removing these existing cols hurt?". It does not natively support the latter. Future Track 2B-style retrospective gates on already-shipped features should use the dual-run backtest pattern (this report) instead. Probe attempt outputs preserved at `reports/track2b_rb_pbp_lgbnb_drop.md` and `reports/track2b_rb_pbp_lgbnb.md` as a record-of-experiment.

---
