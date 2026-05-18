# TODO #33c integration summary — QB + WR Vegas team-context

**Verdict: DO_NOT_ADOPT across all three gates.** The probe's predicted improvements did not generalize to the production gate; the integration regresses on QB lgb-nb by +0.111 fpts (sign-flipped from the probe's predicted −0.0587), and the WR cases land null/inconclusive.

- **Branch:** `feat/qb-wr-vegas-team-context-integration`.
- **Spec:** `docs/superpowers/specs/2026-05-17-qb-wr-vegas-team-context-integration-design.md`.
- **Plan:** `docs/superpowers/plans/2026-05-17-qb-wr-vegas-team-context-integration.md`.
- **Predecessor probe:** PR #49 (`reports/feature_probe_vegas_team_context_summary.md`).
- **Gate runs:** `data/backtest/run_{pre,post}_vegas_{lgbnb,ensdec}` (post-rebase onto current `main` HEAD `f961ab6`).

## Gate verdicts

| Gate # | (model, position) | Probe ΔRMSE | Observed ΔRMSE | 95% CI | n_paired | Verdict |
|---|---|---|---|---|---|---|
| 1 | (lgb-nb, QB) | −0.0587 | **+0.1112** | [+0.0735, +0.1482] | 2676 | **DO_NOT_ADOPT — regression** |
| 2 | (lgb-nb, WR) | −0.0130 | +0.0068 | [−0.0031, +0.0170] | 8460 | **DO_NOT_ADOPT — null/inconclusive** |
| 3 | (ensemble-decomposed, WR) | n/a — production route | +0.0004 | [−0.0060, +0.0073] | 8460 | **DO_NOT_ADOPT — null** |

Spearman deltas (informational):

| Gate # | Spearman delta | CI | Notes |
|---|---|---|---|
| 1 | −0.0222 | [−0.0307, −0.0131] | Strictly below the −0.020 floor — rank regression on QB |
| 2 | −0.0014 | [−0.0033, +0.0002] | Brackets zero — no rank effect on lgb-nb WR |
| 3 | +0.0002 | [−0.0010, +0.0012] | Brackets zero — no rank effect on production WR route |

## Probe-vs-integration replication

The probe and gate measure the same composite ΔRMSE metric on the same model class (lgb-nb), with the same feature swap. Their disagreement is the load-bearing finding:

| Position | Probe point estimate | Gate point estimate | Within ±50% band? |
|---|---|---|---|
| QB | −0.0587 | +0.1112 | **No — sign-flipped, 290% miss** |
| WR | −0.0130 | +0.0068 | **No — sign-flipped, 152% miss** |

## Builder correctness verification (rules out the obvious bug class)

Before accepting the regression as real, I verified the integrated feature builder produces values byte-identical to the probe's override parquet:

- Joined `data/features_probe/vegas_team_context.parquet` (9,379 QB rows, probe-built override) against the integration's QB feature parquets (`data/features/qb/season=*/week=*/part.parquet`) on `(gsis_id, season, week)`.
- All four Vegas cols (`preseason_implied_team_total`, `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread`) match: max absolute delta = 0.0 across 9,379 rows. Zero mismatches.

The feature builder is correct. The regression is not a builder bug.

## Mechanism hypothesis

The probe ran on `data/features` parquets generated at commit `e8e2c6d` (PR #49 merge) — i.e., the **pre-PR-#50** ingest state. The integration's gate ran on parquets refreshed against the **post-PR-#50** ingest (current `main` HEAD `f961ab6`). PR #50 (`fix/ingest-placeholder-gsis-warning`) tightened how `id_map` and `draft_picks` ingest filter placeholder gsis-ids — this changes which player-weeks land in the training and eval sets, especially for pre-camp rookies.

n_paired evidence consistent with this hypothesis: probe Phase 2 reported QB n=2692; gate reports n_paired=2676 (Δ=16 rows). The same delta direction (16 fewer paired rows post-#50) appears across positions. A small fraction of rows changed, but the model's response to the feature swap is apparently sensitive to it.

The mechanism this most likely reflects: lgb-nb's per-stat tree training is sensitive to training-set composition near zero (rookies / depth players). The probe's signal — `preseason_*` + `season_avg_*` outperforming per-game `implied_team_total` + `spread` — held against the pre-#50 training set but inverts against the post-#50 set. This is the "feature-set wins are not invariant to training-set drift" failure mode that the dual-run gate is specifically designed to catch.

Alternatively: the probe ran via `probe_composite` which is a separate harness from `run_backtest`; though both call `lgb-nb.fit/.predict_distribution` on the same model class, their orchestration (walk-forward year sequencing, train_mask boundary semantics, calibration-year slicing in lgb-tuned's parent class) could differ in some way I didn't trace deeply. The gate is the production-truth signal regardless.

## Ship decision: **do not ship as-is**

Per the plan's ship/stop policy:

> Gate 1/2 REGRESSION → stop, debug feature-builder bug

The feature builder is verified correct (§ "Builder correctness verification" above), so this is not a debug-and-retry scenario. The integration's hypothesized benefit (lgb-nb on smoother Vegas signals beats lgb-nb on per-game Vegas) does not survive contact with the production gate.

Three options for closing this branch:

1. **Close PR without merging.** Leaves the branch as a documented negative result. Schemas + builders + lgb-nb feature-list overrides all stay on the branch, regenerable for future re-investigation.
2. **Merge Phase 0 only (schemas + builders), revert Phase 1 (factory swap).** Schemas + builder wire-up are harmless and may help a future probe / integration. Phase 1's lgb-nb factory change is what the gate says don't ship. Cleanest from a "leave the codebase ready for future work" lens.
3. **Merge as-is + flip default routing back to baseline if needed.** Risky — leaves a known-regression model class wired into the production factory. Don't recommend.

**Recommendation: option 2.** Phase 0 makes future probes cheaper (no need to re-wire builders); Phase 1 ships nothing the gate doesn't endorse.

## Caveats (carried from probe verdict, still load-bearing)

- ΔRMSE −0.06 fpts at QB was ≈ 1–2% per-week composite even if it had ADOPTED — **the Chase 250→403 elite-magnitude gap is not closed by this feature class.** Necessary-but-not-sufficient assumption from the spec stands: the elite-magnitude problem lives in *feature signal coverage*, not in per-game-vs-smoothed Vegas data.
- The next genuinely unexplored direction is **external preseason Vegas data** (May win totals, OC/HC tenure, FA-acquisition flag, projected pace, projected pass rate). This is a separate spec, distinct from re-derivations of `spread_line` / `total_line` that are already ingested.
- RB just-missed-ADOPT in the predecessor probe; a follow-up `preseason_*`-only probe was queued. Given this integration's gate reversal, the RB follow-up's prior is weaker — its probe verdict may also fail to generalize. Worth running the RB follow-up but with the dual-run gate as the load-bearing decision criterion, not the probe.
- TE was NULL in the probe — closed.

## What was learned about the probe → gate generalization gap

This is the **first observed case** where a `--force-composite` probe Phase-2 ADOPT did not replicate in the dual-run adoption gate. Prior cases (RB PBP integration — PR #21 — and Plan 9's various negative probes) replicated cleanly. The probe → gate gap matters for the framework's reliability:

- The probe is structurally a walk-forward composite-fpts ΔRMSE bootstrap (`probe_composite`), exactly matching the gate's bootstrap shape. The disagreement is therefore in *data*, not in *measurement procedure*.
- The most likely causal factor here is PR #50's ingest tightening shifting the training set composition. This is the failure mode the integration gate is specifically supposed to catch — and it did, correctly.
- Future probe specs should consider documenting the data state (commit hash of `data/features/` source) so probe → gate disagreement can be attributed cleanly.

## Files

**Gate CSVs:**
- `reports/gate_33c_lgbnb_qb.csv`
- `reports/gate_33c_lgbnb_wr.csv`
- `reports/gate_33c_ensdec_wr.csv`

**Source files modified on this branch:**
- `src/projections/schemas.py` (+8 fields across QbFeaturesSchema + WrFeaturesSchema)
- `src/projections/features/qb.py` (+1 import, +3 lines)
- `src/projections/features/wr.py` (+1 import, +3 lines)
- `src/projections/models/lightgbm_nb.py` (+3 module-level constants + `_swap_for` helper + 2 factory line changes + 1 line in `_code_hash_files_nb`)

**Source files NOT modified (preserves Ridge children of `wr_ensemble_decomposed`):**
- `src/projections/models/baseline.py`
- `src/projections/models/decomposed_baseline.py`
- `src/projections/models/lightgbm.py`
- `src/projections/models/lightgbm_tuned.py`
- `src/projections/models/ensemble.py`

**Tests added:**
- `tests/test_schemas/test_qb_features_schema.py` (4 tests)
- `tests/test_schemas/test_wr_features_schema.py` (4 tests)
- `tests/test_features/test_qb.py` (+1 test)
- `tests/test_features/test_wr.py` (+1 test)
- `tests/test_models/test_lightgbm_nb.py` (+6 tests; +1 rewritten — `test_yards_stat_predictions_match_tuned_baseline` now pins the intentional feature-list divergence between lgb-nb and lgb-tuned for WR)
- `tests/test_models/test_lightgbm.py` (+2 augment-pin tests for QB)
