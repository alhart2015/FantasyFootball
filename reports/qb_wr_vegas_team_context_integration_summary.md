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

## Mechanism (root cause traced)

Initial hypothesis: PR #50's `id_map` / `draft_picks` placeholder filter drifted the data between probe time and integration time. **This hypothesis was wrong.** PR #50's full diff is purely additive logging (`import logging` + `logger.warning(...)` calls) — the filter logic is identical to pre-#50. Verified by reading the actual diff and by observing pre and post backtest runs have byte-identical row coverage (2676 QB rows each, perfectly aligned).

**Actual mechanism: harness-pairing divergence between `probe_composite` and `run_backtest`. The probe is structurally optimistic relative to the production gate when a feature swap differentially helps or hurts player-week rows that the gate's position filter excludes.**

Diagnostic evidence:

- Re-ran the probe on current data with `--force-composite --drop implied_team_total spread --model lightgbm-nb --position QB`. Probe **reproduced** its original verdict: ΔRMSE = −0.0587, CI [−0.092, −0.028], n_paired = 2692, ADOPT.
- Gate on the same data state returns ΔRMSE = +0.1112, CI [+0.0735, +0.1482], n_paired = 2676 — sign-flipped.
- The probe's candidate feature columns match the integration's `_QB_FEATURE_COLUMNS_NB` byte-for-byte (same 22 cols, same order). The probe's candidate Vegas col values match the integration's feature parquet Vegas col values byte-for-byte (0 mismatches across 9,379 QB rows). Same model class, same data, different pairing.

**Where the 16-row delta lives:** `probe_composite` (`src/projections/backtest/feature_probe.py:505`) merges predictions with `weekly_stats` on `(gsis_id, season, week)` **without a position filter**. `run_backtest` (`src/projections/backtest/harness.py:253`) filters `holdout_pos = holdout_actuals[holdout_actuals["position"] == "QB"]` **before** the merge. The 16 extra rows in the probe are **all Taysom Hill** (`gsis_id 00-0033357`) in 2023 — he's on the QB depth chart (so the QB feature builder includes him) but his 2023 weekly_stats rows are labeled `position == "TE"`. Production gate filters him out; probe keeps him.

**Math sanity check:** gate ΔSSE = 2676 × (7.433² − 7.322²) ≈ +4,282 across 2676 rows. Probe ΔSSE ≈ 2692 × (7.26² − 7.32²) ≈ −2,342 across 2692 rows. Combined swing of ~6,624 SSE units across the 16 Taysom Hill rows ≈ **20-fpts residual difference per row** — entirely plausible for his 2023 utility-back-as-QB pattern, one of the most asymmetric player profiles in the dataset. For the Vegas-context swap specifically, the candidate model apparently predicted Taysom Hill closer to actuals than the incumbent did, and those 16 rows alone reversed the QB verdict.

**The gate's +0.1112 ΔRMSE is the production-truth signal.** Production never pairs Taysom Hill's QB-depth-chart predictions with his TE weekly_stats rows; production filters on `position` first. So shipping the lgb-nb swap to QB would yield +0.111 fpts of regression on the production path, not −0.0587 of improvement.

**Implication for the probe framework:** the probe's `--force-composite` Phase-2 verdict can be artifactually optimistic when (a) the feature swap differentially affects depth-chart-mislabeled players and (b) those players have high residual variance. Future probe specs should run a parallel `run_backtest` + `adoption_gate.py` dual-run BEFORE shipping the integration, treating any probe Phase-2 ADOPT as preliminary until the gate confirms it.

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

This is the **first observed case** where a `--force-composite` probe Phase-2 ADOPT did not replicate in the dual-run adoption gate. Prior cases (RB PBP integration — PR #21 — and Plan 9's various negative probes) replicated cleanly.

The actual disagreement is in **measurement procedure**, not in data state:

- `probe_composite` pairs predictions with `weekly_stats` on `(gsis_id, season, week)` only — no position filter.
- `run_backtest` filters `weekly_stats` to `position == <target>` before pairing.
- When the candidate model's predictions differ from incumbent's on rows where a player is on one position's depth chart but recorded under another position's stats (e.g., Taysom Hill 2023: QB depth chart, TE stats), the probe's bootstrap sees those rows; the gate doesn't.
- For QB 2021–2024 lgb-nb specifically, 16 Taysom Hill rows alone account for the full ~0.17-fpts ΔRMSE swing between probe and gate verdicts.

**Concrete framework fix to consider (out of scope for this PR):** align `probe_composite`'s pairing semantics with `run_backtest`'s by adding a position filter to `probe_composite`'s truth merge. This would make the probe's Phase-2 verdict a more faithful predictor of the production gate. Without that fix, any candidate that improves predictions for cross-position-mislabeled rows will be artifactually advantaged by the probe.

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
